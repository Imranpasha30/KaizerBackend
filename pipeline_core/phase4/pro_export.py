"""
kaizer.pipeline.phase4.pro_export
==================================
Final Cut Pro X / Adobe Premiere Pro project exporter.

Power users finish in pro tools. Exporting an FCPX XML (.fcpxml) or
Premiere Pro XML (.xml / Project Panel interchange) with:
  - Primary video track (the rendered master or sliced source segments)
  - Caption track (SRT-equivalent + style metadata)
  - B-roll secondary track with timestamps from pipeline_core.broll
  - Markers at narrative turning points from clip_boundaries /
    narrative.py
  - CTA overlay as a title-effect compound clip

Gives creators the best of both worlds: Kaizer does 90% of the work;
they polish the last 10% in their DAW.

FCPX XML spec: https://developer.apple.com/fcpxml/
Premiere XML: based on Final Cut Pro 7 XML interchange format.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from math import gcd
from pathlib import PurePosixPath
from typing import Optional

logger = logging.getLogger("kaizer.pipeline.phase4.pro_export")


SUPPORTED_FORMATS = ("fcpxml", "prproj_xml")


@dataclass
class ExportResult:
    output_path: str
    format: str                 # one of SUPPORTED_FORMATS
    tracks: int                 # how many tracks emitted
    markers: int                # how many markers emitted
    warnings: list


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _path_to_file_uri(path: str) -> str:
    """Convert an absolute path to a file:// URI with forward slashes.

    Windows:  e:/foo/bar.mp4  ->  file:///e:/foo/bar.mp4
    Unix:     /tmp/foo.mp4    ->  file:///tmp/foo.mp4
    """
    # Normalise to forward slashes
    fwd = path.replace("\\", "/")
    # Strip any leading slashes so we can re-add the right number
    stripped = fwd.lstrip("/")
    # Detect Windows drive letter (e.g. e:/…)
    if len(stripped) >= 2 and stripped[1] == ":":
        # Windows absolute path: file:///e:/foo/bar.mp4
        return "file:///" + stripped
    else:
        # Unix absolute path: file:///tmp/foo.mp4
        return "file:///" + stripped


def _rational(seconds: float, fps: float = 1.0) -> str:
    """Return an FCPX rational-time string such as '45/1s' or '1001/30000s'.

    For whole-second values we produce n/1s.
    For sub-second we use the frame-rate denominator when fps is provided,
    otherwise we fall back to a millisecond-precision fraction.
    """
    if seconds == 0.0:
        return "0s"

    # Try to express as a clean fraction based on fps
    # duration_frames = seconds * fps_num / fps_den
    # We accept fps as a float and find a good rational.
    # Round to nearest frame at the given fps.
    if fps > 0:
        # Represent fps as a rational: find num/den
        # Common fps values: 24, 25, 30, 60, 23.976 (24000/1001), 29.97 (30000/1001)
        # Approximate fps to a rational with den <= 1001
        for den in (1, 1001, 1000, 100, 10):
            num = round(fps * den)
            if abs(num / den - fps) < 0.001:
                break
        else:
            num, den = round(fps * 1001), 1001

        # duration = seconds * num / den  frames
        # FCPX time = frames / fps_rational = frames * den / num  seconds
        # Express as: (seconds * num) / num  * den / den
        # Simplest: numerator = round(seconds * num), denominator = num
        # and reduce by gcd.
        numer = round(seconds * num)
        denom = num
        g = gcd(abs(numer), denom)
        numer //= g
        denom //= g
        if denom == 1:
            return f"{numer}s"
        return f"{numer}/{denom}s"
    else:
        # Fallback: milliseconds
        numer = round(seconds * 1000)
        denom = 1000
        g = gcd(abs(numer), denom)
        numer //= g
        denom //= g
        if denom == 1:
            return f"{numer}s"
        return f"{numer}/{denom}s"


def _probe_source(source_path: str) -> dict:
    """Run ffprobe on source_path and return a dict with width, height, fps, duration.

    Raises ValueError if ffprobe fails or the file is not readable.
    """
    from pipeline_core.qa import FFPROBE_BIN

    if not os.path.exists(source_path):
        raise ValueError(f"source_path does not exist: {source_path!r}")

    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        source_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise ValueError(f"ffprobe binary not found: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"ffprobe timed out on {source_path!r}") from exc

    if result.returncode != 0:
        raise ValueError(
            f"ffprobe failed on {source_path!r} "
            f"(exit {result.returncode}): {result.stderr.strip()}"
        )

    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe returned non-JSON output for {source_path!r}: {exc}") from exc

    streams = probe.get("streams", [])
    fmt = probe.get("format", {})
    video_streams = [s for s in streams if s.get("codec_type") == "video"]

    if not video_streams:
        raise ValueError(f"No video stream found in {source_path!r}")

    vs = video_streams[0]

    # Parse fps
    r_frame_rate = vs.get("r_frame_rate") or vs.get("avg_frame_rate") or "30/1"
    parts = r_frame_rate.split("/")
    try:
        fps = float(parts[0]) / float(parts[1]) if len(parts) == 2 else float(parts[0])
    except (ValueError, ZeroDivisionError, IndexError):
        fps = 30.0

    # Duration
    try:
        duration = float(fmt.get("duration") or vs.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    if duration <= 0.0:
        raise ValueError(f"Could not determine duration for {source_path!r}")

    width = int(vs.get("width") or 0)
    height = int(vs.get("height") or 0)

    if width <= 0 or height <= 0:
        raise ValueError(f"Could not determine dimensions for {source_path!r}")

    return {"width": width, "height": height, "fps": fps, "duration": duration}


def _parse_srt(srt_path: str) -> list[dict]:
    """Parse an SRT file into a list of caption dicts.

    Each dict: {"index": int, "start": float_s, "end": float_s, "text": str}
    Returns [] on any parse error (caller handles warnings).
    """
    try:
        with open(srt_path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except OSError:
        return []

    captions = []
    # SRT blocks separated by blank lines
    blocks = re.split(r"\n\s*\n", raw.strip())
    _ts_re = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        # First line: index
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        # Second line: timecode
        m = _ts_re.match(lines[1].strip())
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(x) for x in m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
        text = "\n".join(lines[2:])
        captions.append({"index": idx, "start": start, "end": end, "text": text})
    return captions


# ---------------------------------------------------------------------------
# FCPX builder
# ---------------------------------------------------------------------------

# FCPX standard colour names recognised by FCPXML
_FCPX_COLOR_MAP = {
    "red":    "red",
    "orange": "orange",
    "yellow": "yellow",
    "green":  "green",
    "blue":   "blue",
    "purple": "purple",
}

_FCPX_FORMAT_NAME_MAP = {
    # (width, height, fps_rounded) -> FCPX format name
    (1080, 1920, 30): "FFVideoFormat1080p30",
    (1080, 1920, 60): "FFVideoFormat1080p60",
    (1920, 1080, 30): "FFVideoFormat1080p30",
    (1920, 1080, 60): "FFVideoFormat1080p60",
    (3840, 2160, 30): "FFVideoFormat4Kp30",
    (3840, 2160, 60): "FFVideoFormat4Kp60",
}


def _fcpx_format_name(width: int, height: int, fps: float) -> str:
    fps_r = round(fps)
    key = (width, height, fps_r)
    return _FCPX_FORMAT_NAME_MAP.get(key, f"FFVideoFormatCustom_{width}x{height}p{fps_r}")


def _build_fcpxml(
    source_path: str,
    probe: dict,
    markers: list[dict],
    captions: list[dict],
    broll_tracks: list[dict],
    warnings: list[str],
) -> ET.Element:
    """Build and return the <fcpxml> root Element."""
    fps = probe["fps"]
    duration = probe["duration"]
    width = probe["width"]
    height = probe["height"]

    dur_rat = _rational(duration, fps)
    fps_r = round(fps)

    # Build fps denominator string for frameDuration: e.g. "1001/30000s" for 29.97
    # For clean fps: "1/30s"
    for den in (1, 1001):
        num_fps = round(fps * den)
        if abs(num_fps / den - fps) < 0.001:
            break
    else:
        num_fps, den = round(fps * 1001), 1001
    if den == 1:
        frame_dur = f"1/{num_fps}s"
    else:
        frame_dur = f"{den}/{num_fps}s"

    # Root element
    root = ET.Element("fcpxml", attrib={"version": "1.11"})

    # --- <resources> ---
    resources = ET.SubElement(root, "resources")

    fmt_id = "r1"
    ET.SubElement(resources, "format", attrib={
        "id": fmt_id,
        "name": _fcpx_format_name(width, height, fps),
        "frameDuration": frame_dur,
        "width": str(width),
        "height": str(height),
    })

    main_asset_id = "r2"
    main_uri = _path_to_file_uri(os.path.abspath(source_path))
    main_name = os.path.splitext(os.path.basename(source_path))[0]
    ET.SubElement(resources, "asset", attrib={
        "id": main_asset_id,
        "name": main_name,
        "src": main_uri,
        "start": "0s",
        "duration": dur_rat,
        "hasVideo": "1",
        "hasAudio": "1",
        "format": fmt_id,
    })

    # B-roll assets
    broll_asset_ids = []
    for i, br in enumerate(broll_tracks):
        br_path = br.get("path", "")
        if not os.path.exists(br_path):
            warnings.append(f"B-roll file not found, skipping asset: {br_path!r}")
            broll_asset_ids.append(None)
            continue
        br_id = f"r{3 + i}"
        br_uri = _path_to_file_uri(os.path.abspath(br_path))
        br_name = os.path.splitext(os.path.basename(br_path))[0]
        br_duration = br.get("duration", 5.0)
        ET.SubElement(resources, "asset", attrib={
            "id": br_id,
            "name": br_name,
            "src": br_uri,
            "start": "0s",
            "duration": _rational(br_duration, fps),
            "hasVideo": "1",
            "hasAudio": "1",
            "format": fmt_id,
        })
        broll_asset_ids.append(br_id)

    # --- <library><event><project><sequence><spine> ---
    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", attrib={"name": "Kaizer Export"})
    project = ET.SubElement(event, "project", attrib={"name": "kaizer_project"})
    sequence = ET.SubElement(project, "sequence", attrib={
        "format": fmt_id,
        "duration": dur_rat,
    })
    spine = ET.SubElement(sequence, "spine")

    # Main asset-clip (lane 0, implicit)
    main_clip = ET.SubElement(spine, "asset-clip", attrib={
        "ref": main_asset_id,
        "offset": "0s",
        "duration": dur_rat,
        "name": main_name,
    })

    # Chapter markers on main clip
    for mk in markers:
        t = float(mk.get("t", 0.0))
        label = str(mk.get("label", ""))
        color = _FCPX_COLOR_MAP.get(str(mk.get("color", "red")).lower(), "red")
        ET.SubElement(main_clip, "chapter-marker", attrib={
            "start": _rational(t, fps),
            "value": label,
            "note": color,
        })

    # SRT captions as <title> elements on main clip
    for cap in captions:
        t_start = cap["start"]
        t_end = cap["end"]
        cap_dur = max(t_end - t_start, 0.0)
        title_elem = ET.SubElement(main_clip, "title", attrib={
            "start": _rational(t_start, fps),
            "duration": _rational(cap_dur, fps),
            "name": f"Caption {cap['index']}",
            "lane": "2",
        })
        ET.SubElement(title_elem, "text").text = cap["text"]

    # B-roll clips on lane 1
    for i, br in enumerate(broll_tracks):
        br_asset_id = broll_asset_ids[i] if i < len(broll_asset_ids) else None
        if br_asset_id is None:
            continue
        t_start = float(br.get("t_start", 0.0))
        br_dur = float(br.get("duration", 5.0))
        br_name = os.path.splitext(os.path.basename(br.get("path", f"broll{i+1}")))[0]
        ET.SubElement(spine, "asset-clip", attrib={
            "ref": br_asset_id,
            "offset": _rational(t_start, fps),
            "duration": _rational(br_dur, fps),
            "lane": "1",
            "name": br_name,
        })

    return root


# ---------------------------------------------------------------------------
# Premiere (FCP7 XML) builder
# ---------------------------------------------------------------------------

def _frames(seconds: float, fps: float) -> int:
    """Convert seconds to frame count using round()."""
    return round(seconds * fps)


def _build_prproj_xml(
    source_path: str,
    probe: dict,
    markers: list[dict],
    captions: list[dict],
    broll_tracks: list[dict],
    warnings: list[str],
) -> ET.Element:
    """Build and return the <xmeml> root Element for Premiere / FCP7 XML."""
    fps = probe["fps"]
    duration = probe["duration"]

    fps_r = round(fps)
    # For 29.97, ntsc is TRUE; for clean fps it is FALSE
    is_ntsc = abs(fps - round(fps)) > 0.01
    ntsc_str = "TRUE" if is_ntsc else "FALSE"
    timebase = fps_r
    total_frames = _frames(duration, fps)

    main_uri = _path_to_file_uri(os.path.abspath(source_path))

    root = ET.Element("xmeml", attrib={"version": "5"})
    sequence = ET.SubElement(root, "sequence")

    # Sequence rate
    seq_rate = ET.SubElement(sequence, "rate")
    ET.SubElement(seq_rate, "timebase").text = str(timebase)
    ET.SubElement(seq_rate, "ntsc").text = ntsc_str

    media = ET.SubElement(sequence, "media")
    video = ET.SubElement(media, "video")

    # Track 1: main clip
    track1 = ET.SubElement(video, "track")
    clip1 = ET.SubElement(track1, "clipitem", attrib={"id": "clip-1"})

    file1 = ET.SubElement(clip1, "file", attrib={"id": "file-1"})
    ET.SubElement(file1, "pathurl").text = main_uri
    file_rate = ET.SubElement(file1, "rate")
    ET.SubElement(file_rate, "timebase").text = str(timebase)
    ET.SubElement(file_rate, "ntsc").text = ntsc_str

    clip_rate = ET.SubElement(clip1, "rate")
    ET.SubElement(clip_rate, "timebase").text = str(timebase)
    ET.SubElement(clip_rate, "ntsc").text = ntsc_str

    ET.SubElement(clip1, "in").text = "0"
    ET.SubElement(clip1, "out").text = str(total_frames)

    # Markers on clip
    for mk in markers:
        t = float(mk.get("t", 0.0))
        label = str(mk.get("label", ""))
        frame_in = _frames(t, fps)
        marker_elem = ET.SubElement(clip1, "marker")
        ET.SubElement(marker_elem, "name").text = label
        ET.SubElement(marker_elem, "in").text = str(frame_in)
        ET.SubElement(marker_elem, "out").text = "-1"

    # SRT captions as a separate text track
    if captions:
        cap_track = ET.SubElement(video, "track")
        for i, cap in enumerate(captions):
            t_start = cap["start"]
            t_end = cap["end"]
            in_frame = _frames(t_start, fps)
            out_frame = _frames(t_end, fps)
            cap_clip = ET.SubElement(cap_track, "clipitem", attrib={"id": f"caption-{i+1}"})
            ET.SubElement(cap_clip, "in").text = str(in_frame)
            ET.SubElement(cap_clip, "out").text = str(out_frame)
            ET.SubElement(cap_clip, "name").text = cap["text"]

    # Track 2: b-roll
    if broll_tracks:
        broll_track = ET.SubElement(video, "track")
        valid_broll = 0
        for i, br in enumerate(broll_tracks):
            br_path = br.get("path", "")
            if not os.path.exists(br_path):
                warnings.append(f"B-roll file not found, skipping clip: {br_path!r}")
                continue
            valid_broll += 1
            t_start = float(br.get("t_start", 0.0))
            br_dur = float(br.get("duration", 5.0))
            in_frame = _frames(t_start, fps)
            out_frame = _frames(t_start + br_dur, fps)
            br_uri = _path_to_file_uri(os.path.abspath(br_path))
            br_clip = ET.SubElement(broll_track, "clipitem", attrib={"id": f"bcroll-{i+1}"})
            br_file = ET.SubElement(br_clip, "file", attrib={"id": f"bfile-{i+1}"})
            ET.SubElement(br_file, "pathurl").text = br_uri
            ET.SubElement(br_clip, "in").text = str(in_frame)
            ET.SubElement(br_clip, "out").text = str(out_frame)

    return root


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_project(
    source_path: str,
    *,
    markers: list[dict] | None = None,
    caption_srt_path: Optional[str] = None,
    broll_tracks: list[dict] | None = None,
    output_path: str,
    format: str = "fcpxml",
) -> ExportResult:
    """Emit a FCPX or Premiere XML project file.

    - Validates source_path exists and is readable via ffprobe (width, height,
      fps, duration).
    - Builds an XML tree conforming to the target format's spec:
        * fcpxml: FCPX v1.11 minimum (<fcpxml version="1.11">), with a
          <resources> section (<format>, <asset> for main + b-rolls), then
          <library>/<event>/<project>/<sequence>/<spine>. Main asset on
          lane 0; b-rolls on lane 1 using <asset-clip>; markers as
          <chapter-marker> children on the main clip.
        * prproj_xml: Final Cut Pro 7 XML interchange (<xmeml version="5">),
          <sequence><media><video><track><clipitem> ... with <rate>, <in>,
          <out>, <marker> children.
    - Markers: passed dicts with t, label, color. Default color 'red'.
    - Caption SRT: inlined as a <title> element (FCPX) or a separate text
      track (Premiere). If parsing SRT fails, emit a warning but continue.
    - broll_tracks: b-rolls inserted on secondary track (lane 1 in FCPX,
      track 2 in Premiere) with explicit in/out times.
    - Writes the XML to output_path (pretty-printed, utf-8 w/ BOM not needed).
    - Returns ExportResult(output_path, format, tracks=2 if broll else 1,
      markers=len(markers), warnings=<list>).

    Raises ValueError on unsupported format or unreadable source.
    Never silently swallows errors — validation failures raise, but optional
    enrichments (SRT parse, missing b-roll file on disk) degrade with warnings.
    """
    if format not in SUPPORTED_FORMATS:
        raise ValueError(f"format must be one of {SUPPORTED_FORMATS}, got {format!r}")

    # Normalise optional args
    markers = list(markers) if markers else []
    broll_tracks = list(broll_tracks) if broll_tracks else []
    warnings: list[str] = []

    # Validate + probe source
    probe = _probe_source(source_path)

    # Parse SRT captions (soft failure)
    captions: list[dict] = []
    if caption_srt_path is not None:
        if not os.path.exists(caption_srt_path):
            warnings.append(
                f"caption_srt_path does not exist, captions skipped: {caption_srt_path!r}"
            )
        else:
            captions = _parse_srt(caption_srt_path)
            if not captions:
                warnings.append(
                    f"SRT file could not be parsed or is empty: {caption_srt_path!r}"
                )

    # Determine track count
    has_broll = bool(broll_tracks)
    track_count = 2 if has_broll else 1

    # Build the XML tree
    if format == "fcpxml":
        root = _build_fcpxml(
            source_path, probe, markers, captions, broll_tracks, warnings
        )
        tree = ET.ElementTree(root)
        # DOCTYPE declaration + XML declaration handled via manual write
        ET.indent(tree, space="  ")
        xml_bytes = ET.tostring(root, encoding="unicode", xml_declaration=False)
        output_text = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE fcpxml>\n'
            + xml_bytes
        )

    elif format == "prproj_xml":
        root = _build_prproj_xml(
            source_path, probe, markers, captions, broll_tracks, warnings
        )
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        xml_bytes = ET.tostring(root, encoding="unicode", xml_declaration=False)
        output_text = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE xmeml>\n'
            + xml_bytes
        )

    else:
        # Should never reach here because we check SUPPORTED_FORMATS above
        raise ValueError(f"format must be one of {SUPPORTED_FORMATS}, got {format!r}")

    # Self-verify: the output must be parseable
    try:
        ET.fromstring(output_text)
    except ET.ParseError as exc:
        raise ValueError(f"Internal error: generated XML is not well-formed: {exc}") from exc

    # Write output file (UTF-8, no BOM)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(output_text)

    logger.info(
        "pro_export: wrote %s format=%s tracks=%d markers=%d warnings=%d path=%s",
        format, format, track_count, len(markers), len(warnings), output_path,
    )

    return ExportResult(
        output_path=output_path,
        format=format,
        tracks=track_count,
        markers=len(markers),
        warnings=warnings,
    )
