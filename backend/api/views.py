import os
import sys
import json
import shutil
import mimetypes
import subprocess

from django.conf import settings
from django.http import StreamingHttpResponse, HttpResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, JSONParser
from rest_framework.response import Response
from rest_framework import status

from .models import Job, Clip
from .serializers import JobSerializer, JobListSerializer, ClipSerializer
from . import runner

# ── Platform & Frame Layout constants (mirror pipeline) ─────
PLATFORM_PRESETS = {
    "instagram_reel": {"label": "Instagram Reel",  "width": 1080, "height": 1920},
    "youtube_short":  {"label": "YouTube Short",   "width": 1080, "height": 1920},
    "youtube_full":   {"label": "YouTube Full",    "width": 1920, "height": 1080},
}

FRAME_LAYOUTS = {
    "torn_card":   "Torn Card   — Video + Red headline card + Image",
    "split_frame": "Split Frame — Thumbnail + Video on colored background",
    "follow_bar":  "Follow Bar  — Text top + Square video + Social follow bar",
}


# ── Config endpoints ─────────────────────────────────────────
@api_view(["GET"])
def platforms(request):
    return Response(PLATFORM_PRESETS)

@api_view(["GET"])
def frame_layouts(request):
    return Response(FRAME_LAYOUTS)


# ── Job CRUD ─────────────────────────────────────────────────
@api_view(["GET"])
def job_list(request):
    jobs = Job.objects.all()
    return Response(JobListSerializer(jobs, many=True).data)


@api_view(["POST"])
@parser_classes([MultiPartParser])
def job_create(request):
    """Upload video + choose platform/frame → create Job, start pipeline immediately."""
    video_file   = request.FILES.get("video")
    platform     = request.data.get("platform", "instagram_reel")
    frame_layout = request.data.get("frame_layout", "torn_card")

    if not video_file:
        return Response({"error": "No video file"}, status=400)
    if platform not in PLATFORM_PRESETS:
        return Response({"error": f"Unknown platform: {platform}"}, status=400)
    if frame_layout not in FRAME_LAYOUTS:
        return Response({"error": f"Unknown frame_layout: {frame_layout}"}, status=400)

    # Save video to media/uploads/
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    video_path = os.path.join(upload_dir, video_file.name)
    with open(video_path, "wb") as f:
        for chunk in video_file.chunks():
            f.write(chunk)

    job = Job.objects.create(
        video_name   = video_file.name,
        video_path   = video_path,
        platform     = platform,
        frame_layout = frame_layout,
        status       = "pending",
    )

    # Auto-start pipeline
    runner.start_pipeline(job.id, video_path, platform, frame_layout)
    Job.objects.filter(id=job.id).update(status="running")

    return Response({"id": str(job.id)}, status=201)


@api_view(["GET"])
def job_detail(request, job_id):
    try:
        job = Job.objects.get(id=job_id)
    except Job.DoesNotExist:
        return Response({"error": "Not found"}, status=404)
    return Response(JobSerializer(job).data)


@api_view(["GET"])
def job_status(request, job_id):
    """Lightweight polling endpoint — just status, pct, last 50 log lines."""
    try:
        job = Job.objects.get(id=job_id)
    except Job.DoesNotExist:
        return Response({"error": "Not found"}, status=404)

    live = runner.get_live_log(str(job_id))
    lines = live if live else job.log.splitlines()
    return Response({
        "status":       job.status,
        "progress_pct": job.progress_pct,
        "log_lines":    lines[-60:],
        "error":        job.error,
    })


@api_view(["DELETE"])
def job_delete(request, job_id):
    try:
        job = Job.objects.get(id=job_id)
    except Job.DoesNotExist:
        return Response({"error": "Not found"}, status=404)
    if job.output_dir and os.path.isdir(job.output_dir):
        shutil.rmtree(job.output_dir, ignore_errors=True)
    job.delete()
    return Response({"deleted": True})


# ── Clip editing ─────────────────────────────────────────────
@api_view(["GET"])
def clip_detail(request, clip_id):
    try:
        clip = Clip.objects.get(id=clip_id)
    except Clip.DoesNotExist:
        return Response({"error": "Not found"}, status=404)
    return Response(ClipSerializer(clip).data)


@api_view(["POST"])
@parser_classes([JSONParser])
def clip_rerender(request, clip_id):
    """Re-render a clip with new edits (text, font, color, sections, card style)."""
    try:
        clip = Clip.objects.get(id=clip_id)
    except Clip.DoesNotExist:
        return Response({"error": "Not found"}, status=404)

    edits = request.data

    # Build meta dict that rerender_clip() expects
    clip_dict = {
        "clip_path":    clip.clip_path,
        "raw_path":     clip.raw_path,
        "image_path":   clip.image_path,
        "text":         clip.text,
        "frame_type":   clip.frame_type,
        "card_params":  clip.card_params,
        "section_pct":  clip.section_pct,
        "follow_params": clip.follow_params,
        "split_params": clip.split_params,
        "preset":       clip.preset or clip.job.preset if hasattr(clip.job, 'preset') else {},
    }

    try:
        meta = {"clips": [clip_dict], "preset": clip.preset}
        new_path = _rerender(clip_dict, edits, meta)

        # Regenerate thumbnail
        thumb_path = new_path.replace(".mp4", "_thumb.jpg")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", new_path, "-vframes", "1", "-q:v", "2", thumb_path],
                capture_output=True, check=True
            )
            clip.thumb_path = os.path.abspath(thumb_path)
        except Exception:
            pass

        # Update clip record
        clip.clip_path    = os.path.abspath(new_path)
        clip.text         = edits.get("text", clip.text)
        clip.frame_type   = edits.get("frame_type", clip.frame_type)
        if "section_pct"  in edits: clip.section_pct  = edits["section_pct"]
        if "card_params"  in edits: clip.card_params   = edits["card_params"]
        if "follow_params"in edits: clip.follow_params = edits["follow_params"]
        if "split_params" in edits: clip.split_params  = edits["split_params"]
        clip.save()

        return Response(ClipSerializer(clip).data)

    except Exception as e:
        import traceback
        return Response({"error": str(e), "detail": traceback.format_exc()}, status=500)


def _rerender(clip_dict, edits, meta):
    """Import rerender_clip from pipeline_core/editor.py (local copy) and run it."""
    from importlib import util as _ilu
    editor_path = os.path.join(settings.BASE_DIR, "pipeline_core", "editor.py")
    _orig_argv = sys.argv[:]
    sys.argv = ["editor.py"]
    try:
        spec = _ilu.spec_from_file_location("kn_editor", editor_path)
        mod  = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.META = meta
        return mod.rerender_clip(clip_dict, edits)
    finally:
        sys.argv = _orig_argv


@api_view(["POST"])
@parser_classes([MultiPartParser])
def clip_upload_image(request, clip_id):
    """Replace the image for a clip."""
    try:
        clip = Clip.objects.get(id=clip_id)
    except Clip.DoesNotExist:
        return Response({"error": "Not found"}, status=404)

    img_file = request.FILES.get("image")
    if not img_file:
        return Response({"error": "No image file"}, status=400)

    out_dir = os.path.dirname(clip.clip_path) or os.path.join(settings.MEDIA_ROOT, "clips")
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f"user_img_{clip.index + 1:02d}.jpg")
    with open(img_path, "wb") as f:
        for chunk in img_file.chunks():
            f.write(chunk)

    clip.image_path = img_path
    clip.save()
    return Response({"image_path": img_path, "image_url": f"/api/file/?path={img_path}"})


# ── Export ───────────────────────────────────────────────────
@api_view(["POST"])
def job_export(request, job_id):
    """Copy all clip files to an export/ subfolder."""
    try:
        job = Job.objects.get(id=job_id)
    except Job.DoesNotExist:
        return Response({"error": "Not found"}, status=404)

    export_dir = os.path.join(job.output_dir or settings.PIPELINE_OUTPUT_ROOT, "export")
    os.makedirs(export_dir, exist_ok=True)

    count = 0
    paths = []
    for clip in job.clips.all():
        src = clip.clip_path
        if src and os.path.isfile(src):
            dst = os.path.join(export_dir, f"kaizer_clip_{clip.index + 1:02d}.mp4")
            shutil.copy2(src, dst)
            paths.append(dst)
            count += 1

    return Response({"count": count, "export_dir": export_dir, "files": paths})


# ── File serving with Range support ─────────────────────────
def serve_file(request):
    """
    GET /api/file/?path=<absolute_path>
    Serves any pipeline output file with proper Range support for video playback.
    """
    path = request.GET.get("path", "")
    if not path or not os.path.isfile(path):
        raise Http404

    # Security: must be inside the project root or output root
    project_root = str(settings.BASE_DIR.parent.parent)
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(project_root):
        raise Http404

    ctype = mimetypes.guess_type(abs_path)[0] or "application/octet-stream"
    size  = os.path.getsize(abs_path)

    range_header = request.META.get("HTTP_RANGE", "")
    if range_header and range_header.startswith("bytes="):
        try:
            rng   = range_header[6:].split("-")
            start = int(rng[0]) if rng[0] else 0
            end   = int(rng[1]) if len(rng) > 1 and rng[1] else size - 1
            end   = min(end, size - 1)
            chunk = end - start + 1

            def _stream(s, c):
                with open(abs_path, "rb") as f:
                    f.seek(s)
                    remaining = c
                    while remaining > 0:
                        buf = f.read(min(65536, remaining))
                        if not buf:
                            break
                        yield buf
                        remaining -= len(buf)

            resp = StreamingHttpResponse(_stream(start, chunk), status=206, content_type=ctype)
            resp["Content-Range"]  = f"bytes {start}-{end}/{size}"
            resp["Content-Length"] = str(chunk)
            resp["Accept-Ranges"]  = "bytes"
            return resp
        except Exception:
            return HttpResponse(status=416)

    resp = StreamingHttpResponse(
        open(abs_path, "rb"), content_type=ctype
    )
    resp["Content-Length"] = str(size)
    resp["Accept-Ranges"]  = "bytes"
    return resp


# ── Font serving ─────────────────────────────────────────────
def serve_font(request, filename):
    fonts_dir = os.path.join(settings.PIPELINE_RESOURCES, "fonts")
    path = os.path.join(fonts_dir, filename)
    if not os.path.isfile(path):
        raise Http404
    return StreamingHttpResponse(open(path, "rb"), content_type="font/ttf")
