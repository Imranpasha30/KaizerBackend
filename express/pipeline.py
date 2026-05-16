"""Express Mode pipeline orchestrator.

Direct port of the teammate's ``runAutopubPipeline`` — but only the
"publish-as-is" branch in Session 1. The full flow (AI trim + shorts
cut + cinematic + color grade) lands in Session 2.

Session 1 flow
--------------
1. Extract audio (ffmpeg)
2. Whisper transcribe with brief+names hint biasing
3. Claude SEO writer (title / description / tags)
4. Postiz upload of the original source video
5. Postiz post creation on each selected integration
6. Mark job done

Runs in a background thread; the router fires-and-returns. Progress
is reported via ``express.state.set_step()`` which the status endpoint
polls every 2-5s.

Tenancy
-------
Every Express Mode job is owned by the Kaizer user who started it.
The owner check happens at the router boundary; this module trusts
its inputs and only updates the job state it was handed.
"""
from __future__ import annotations

import os
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import ai_image, claude, cut_clip, render_longform, state, whisper


# ─── Global render concurrency cap ────────────────────────────────
# ffmpeg with our filter graphs (esp. cinematic Ken Burns + zoompan)
# can hold 1.5-2.5 GB of RAM per render. On a 16 GB workstation,
# letting more than 1-2 jobs run in parallel oversaturates the box,
# crashes Chrome, and produces stuck zombie ffmpegs. The semaphore
# below caps concurrent Express Mode jobs across ALL pipelines.
# Override via env for cloud deployments with more RAM headroom.
_RENDER_CONCURRENCY = max(1, int(
    (os.environ.get("KAIZER_EXPRESS_CONCURRENCY") or "1").strip()
))
_RENDER_SLOT = threading.BoundedSemaphore(_RENDER_CONCURRENCY)


def _acquire_slot(jid: str) -> None:
    """Block on the render semaphore. Updates the job's message so the
    UI shows "waiting for render slot" instead of "starting" forever."""
    if not _RENDER_SLOT.acquire(blocking=False):
        # Slot taken — log + block. The user will see the bar at 0%
        # with "waiting for render slot" message until the in-flight
        # job releases.
        state.set_step(jid, "queued", 0,
                       f"waiting for render slot (concurrency={_RENDER_CONCURRENCY})…")
        _RENDER_SLOT.acquire()


def _release_slot() -> None:
    try:
        _RENDER_SLOT.release()
    except ValueError:
        # Already released — defensive no-op.
        pass


def _cleanup(paths: list[str]) -> None:
    """Best-effort delete of temp files. Logs but doesn't raise."""
    for p in paths:
        if not p:
            continue
        try:
            Path(p).unlink(missing_ok=True)
        except OSError as exc:
            print(f"[express] cleanup skip {p}: {exc}")


# Per-call cap on integrations sent to Postiz's /public/v1/posts. Each
# integration adds one full ``posts[i]`` entry (text + settings + tags
# + image refs) to the JSON body. The user's self-hosted Postiz has
# been 413-ing at 10 integrations per call — drop to 3 so the body
# stays small regardless of description length. Override per-job via
# the env var if needed.
_POSTIZ_INTEGRATION_CHUNK = int(
    (os.environ.get("KAIZER_POSTIZ_INTEGRATION_CHUNK") or "3").strip()
)

# YouTube's video description displays cap at ~5000 chars but the
# Postiz body multiplies by N integrations. Cap aggressively for the
# wire body — the FULL description still lands on each YouTube post
# because Postiz fans it out from the single payload.
_POSTIZ_DESCRIPTION_CAP = 1500


def _schedule_post_chunked(
    *,
    integration_ids: list[str],
    log_prefix: str,
    append_log,
    chunk: int = _POSTIZ_INTEGRATION_CHUNK,
    **kwargs,
) -> list[dict]:
    """Call ``postiz_client.schedule_post`` in batches of ``chunk``
    integrations to keep each request body under Postiz's body-size
    cap. Per-chunk failures are logged but don't block the rest.

    Returns the list of per-chunk response dicts (one entry per
    chunk). Empty list iff EVERY chunk failed — caller can treat
    that as "post creation failed entirely".
    """
    from clients import postiz as postiz_client

    # Cap description length inside this helper so all 3 modes
    # benefit — long descriptions multiplied by N integrations are a
    # major contributor to 413s.
    if "text" in kwargs and isinstance(kwargs["text"], str):
        if len(kwargs["text"]) > _POSTIZ_DESCRIPTION_CAP:
            kwargs["text"] = kwargs["text"][:_POSTIZ_DESCRIPTION_CAP].rstrip() + "…"

    responses: list[dict] = []
    failures = 0
    total_chunks = max(1, (len(integration_ids) + chunk - 1) // chunk)
    append_log(f"{log_prefix} posting to {len(integration_ids)} integrations in {total_chunks} chunk(s) of {chunk}")
    for ci in range(total_chunks):
        slice_ids = integration_ids[ci * chunk : (ci + 1) * chunk]
        if not slice_ids:
            continue
        try:
            resp = postiz_client.schedule_post(
                integration_ids=slice_ids,
                **kwargs,
            )
            responses.append({"chunk": ci, "size": len(slice_ids), "resp": resp})
            append_log(f"{log_prefix} chunk {ci+1}/{total_chunks} ({len(slice_ids)} integrations) ok")
        except postiz_client.PostizError as exc:
            failures += 1
            append_log(f"{log_prefix} chunk {ci+1}/{total_chunks} failed: {exc}")
    if failures == total_chunks:
        # Every chunk failed → caller treats as full failure.
        return []
    return responses


def run_publish_as_is(
    *,
    jid: str,
    video_path: str,
    integration_ids: list[str],
    brief: str = "",
    names_hint: str = "",
    style_guide: str = "",
    language: Optional[str] = None,
    privacy: str = "private",          # public | unlisted | private
    made_for_kids: bool = False,
    title_override: Optional[str] = None,
    description_override: Optional[str] = None,
    tags_override: Optional[list[str]] = None,
    schedule_at_iso: Optional[str] = None,
    # AI keys passed in from the request (per-user, not from env)
    anthropic_api_key: str = "",
    transcription_provider: str = "groq",
    transcription_api_key: str = "",
    transcription_base_url: Optional[str] = None,
    transcription_model: Optional[str] = None,
) -> None:
    """Execute the publish-as-is pipeline synchronously.

    The caller is expected to invoke this in a background thread so
    the HTTP request can return immediately with the job id.
    """
    cleanups: list[str] = []
    _acquire_slot(jid)
    try:
        # Sanity-check inputs ──────────────────────────────────────
        if not os.path.isfile(video_path):
            raise RuntimeError(f"video_path not found: {video_path}")
        if not integration_ids:
            raise RuntimeError("no Postiz integrations selected")

        # 1) Audio extract + Whisper ────────────────────────────────
        state.set_step(jid, "transcribe", 5, "Extracting audio…")
        audio_path = video_path + ".mp3"
        whisper.extract_audio_mp3(video_path, audio_path)
        cleanups.append(audio_path)

        state.set_step(jid, "transcribe", 10, "Transcribing audio (Whisper)…")
        # Combine brief + names hint as the Whisper prompt (mirrors
        # teammate's whisperPromptParts logic). Brief first because
        # it carries the higher-signal topic context.
        wp_parts: list[str] = []
        if brief: wp_parts.append(brief)
        if names_hint: wp_parts.append(names_hint)
        whisper_prompt = ("\n".join(wp_parts))[:800] if wp_parts else None

        try:
            tr = whisper.transcribe(
                audio_path,
                api_key=transcription_api_key,
                provider=transcription_provider,
                base_url=transcription_base_url,
                model=transcription_model,
                language=language or None,
                names_hint=whisper_prompt,
            )
        except whisper.WhisperError as exc:
            # Preserve the audio file so we can inspect why the provider
            # rejected it (e.g. malformed frames, silent audio). Moves
            # it out of the cleanups list so the finally-block leaves
            # it alone.
            import shutil as _sh
            debug_dir = os.path.join(
                os.path.dirname(audio_path), "_express_debug"
            )
            try:
                os.makedirs(debug_dir, exist_ok=True)
                kept = os.path.join(debug_dir, f"{jid}.mp3")
                _sh.copy2(audio_path, kept)
                state.append_log(jid, f"[transcribe] failed audio kept at {kept}")
            except OSError as cp_exc:
                state.append_log(jid, f"[transcribe] could not stash audio: {cp_exc}")
            raise RuntimeError(f"transcription failed: {exc}") from exc

        transcript = tr.get("text", "") or ""
        duration   = float(tr.get("duration") or 0.0)
        state.append_log(jid, f"[transcribe] {len(transcript)} chars, {duration:.1f}s duration")

        # 2) Claude SEO writer ──────────────────────────────────────
        state.set_step(jid, "plan", 30, "Writing SEO (Claude)…")
        try:
            seo = claude.write_seo(
                api_key=anthropic_api_key,
                transcript=transcript,
                brief=brief,
                names_hint=names_hint,
                style_guide=style_guide,
            )
        except claude.ClaudeError as exc:
            raise RuntimeError(f"SEO generation failed: {exc}") from exc

        # Apply user overrides on top of Claude's SEO.
        title = (title_override or seo.get("title") or "Untitled")[:100]
        description = description_override if description_override is not None else seo.get("description", "")
        tags = tags_override if tags_override is not None else seo.get("tags") or []
        state.append_log(jid, f"[plan] title={title!r} tags={len(tags)}")

        # 3) Postiz upload ──────────────────────────────────────────
        state.set_step(jid, "upload", 55, "Uploading to Postiz…")
        # Use Kaizer's existing Postiz client — same wire format,
        # already battle-tested by the per-channel publish flow.
        from clients import postiz as postiz_client
        try:
            media = postiz_client.upload_file(video_path, "video/mp4")
        except postiz_client.PostizError as exc:
            raise RuntimeError(f"Postiz upload failed: {exc}") from exc

        media_id   = media.get("id", "")
        media_path = media.get("path", "")
        state.append_log(jid, f"[upload] media_id={media_id} ({len(integration_ids)} integration(s))")

        # 4) Postiz post creation ──────────────────────────────────
        state.set_step(jid, "publish", 80, "Creating Postiz posts…")
        # Postiz expects type="now" for immediate post, "scheduled"
        # when a future date is provided.
        post_type = "scheduled" if schedule_at_iso else "now"
        post_resp = _schedule_post_chunked(
            integration_ids=integration_ids,
            log_prefix="[publish]",
            append_log=lambda m, _jid=jid: state.append_log(_jid, m),
            text=description or title,
            media_id=media_id,
            media_path=media_path,
            schedule_at_iso=schedule_at_iso,
            type_=post_type,
            yt_title=title,
            yt_privacy=privacy or "private",
            yt_tags=tags,
            yt_made_for_kids=bool(made_for_kids),
        )
        if not post_resp:
            raise RuntimeError("Postiz post creation failed on every chunk")

        # 5) Done ──────────────────────────────────────────────────
        state.mark_done(jid, {
            "mode":            "publish-as-is",
            "title":           title,
            "description":     description,
            "tags":            tags,
            "integration_ids": integration_ids,
            "scheduled_at":    schedule_at_iso,
            "postiz":          {
                "media_id":  media_id,
                "media_url": media_path,
                "post":      post_resp,
            },
            "duration_s":      duration,
        })

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[express] job={jid} FAILED:\n{tb}")
        state.mark_failed(jid, str(exc))
    finally:
        _cleanup(cleanups)
        _release_slot()


# ─── AI Trim mode ───────────────────────────────────────────────────
#
# 1. Audio extract + Whisper (same as publish-as-is, with SEGMENTS used)
# 2. Claude SEO writer  (parallel-ish; we do serial for simplicity)
# 3. Claude trim plan   — which segments to keep
# 4. ffmpeg renderLongformTrim — concat the keep plan, apply color
#    grade, optionally cinematic xfade + Ken Burns + grain
# 5. Postiz upload of the TRIMMED video
# 6. Postiz post creation
# 7. Mark done

def run_ai_trim(
    *,
    jid: str,
    video_path: str,
    integration_ids: list[str],
    brief: str = "",
    names_hint: str = "",
    style_guide: str = "",
    language: Optional[str] = None,
    privacy: str = "private",
    made_for_kids: bool = False,
    title_override: Optional[str] = None,
    description_override: Optional[str] = None,
    tags_override: Optional[list[str]] = None,
    schedule_at_iso: Optional[str] = None,
    anthropic_api_key: str = "",
    transcription_provider: str = "groq",
    transcription_api_key: str = "",
    transcription_base_url: Optional[str] = None,
    transcription_model: Optional[str] = None,
    color_grade_preset: str = "subtle",
    cinematic_edit: bool = False,
    logo_path: Optional[str] = None,
    # Session 3 additions:
    openai_api_key: str = "",                 # for gpt-image-1 thumbnail
    thumbnail_strategy: str = "none",         # none | ai
) -> None:
    """Full AI-trim → publish path. Same contract as run_publish_as_is
    (background thread, status via state.set_step, results via state.mark_done)."""
    cleanups: list[str] = []
    _acquire_slot(jid)
    try:
        if not os.path.isfile(video_path):
            raise RuntimeError(f"video_path not found: {video_path}")
        if not integration_ids:
            raise RuntimeError("no Postiz integrations selected")

        # 1) Audio + Whisper (request SEGMENTS for the trim planner)
        state.set_step(jid, "transcribe", 5, "Extracting audio…")
        audio_path = video_path + ".mp3"
        whisper.extract_audio_mp3(video_path, audio_path)
        cleanups.append(audio_path)

        state.set_step(jid, "transcribe", 12, "Transcribing audio (Whisper)…")
        wp_parts = [p for p in (brief, names_hint) if p]
        whisper_prompt = ("\n".join(wp_parts))[:800] if wp_parts else None
        try:
            tr = whisper.transcribe(
                audio_path,
                api_key=transcription_api_key,
                provider=transcription_provider,
                base_url=transcription_base_url,
                model=transcription_model,
                language=language or None,
                names_hint=whisper_prompt,
            )
        except whisper.WhisperError as exc:
            raise RuntimeError(f"transcription failed: {exc}") from exc

        transcript = tr.get("text", "") or ""
        segments   = tr.get("segments") or []
        duration   = float(tr.get("duration") or 0.0)
        tr_lang    = tr.get("language") or language or ""
        state.append_log(jid, f"[transcribe] {len(transcript)} chars, {len(segments)} segments, {duration:.1f}s")

        if not segments or duration < 5:
            raise RuntimeError("Whisper returned no usable segments — can't plan a trim")

        # 2) Claude SEO
        state.set_step(jid, "plan", 25, "Writing SEO (Claude)…")
        try:
            seo = claude.write_seo(
                api_key=anthropic_api_key,
                transcript=transcript,
                brief=brief,
                names_hint=names_hint,
                style_guide=style_guide,
            )
        except claude.ClaudeError as exc:
            raise RuntimeError(f"SEO generation failed: {exc}") from exc
        title = (title_override or seo.get("title") or "Untitled")[:100]
        description = description_override if description_override is not None else seo.get("description", "")
        tags = tags_override if tags_override is not None else seo.get("tags") or []

        # 3) Claude trim plan
        state.set_step(jid, "plan", 38, "Planning trim (Claude)…")
        window = claude.target_trim_window(duration)
        try:
            plan = claude.plan_longform_trim(
                api_key=anthropic_api_key,
                segments=segments,
                language=tr_lang,
                duration=duration,
                target_min=window["min_sec"],
                target_max=window["max_sec"],
                style_guide=style_guide,
            )
        except claude.ClaudeError as exc:
            raise RuntimeError(f"trim plan failed: {exc}") from exc

        keep = plan.get("keep") or []
        if not keep:
            raise RuntimeError("Claude returned no segments to keep — cannot render")
        kept_total = sum(s["end"] - s["start"] for s in keep)
        state.append_log(jid,
            f"[plan] kept {len(keep)} segments = {kept_total:.0f}s "
            f"(target {window['label']}). {plan.get('summary','')}")

        # 4) ffmpeg renderLongformTrim
        state.set_step(jid, "render-trim", 50,
                       f"Rendering trimmed video ({kept_total:.0f}s, grade={color_grade_preset}"
                       + (", cinematic" if cinematic_edit else "") + ")…")
        trimmed_path = video_path + ".trimmed.mp4"
        cleanups.append(trimmed_path)
        try:
            render_longform.render_longform_trim(
                input_path=video_path,
                output_path=trimmed_path,
                keep=keep,
                logo_path=logo_path,
                color_grade_preset=color_grade_preset,
                cinematic_edit=cinematic_edit,
            )
        except render_longform.RenderError as exc:
            raise RuntimeError(f"trim render failed: {exc}") from exc

        # 4b) Optional AI thumbnail (gpt-image-1). Soft-fail — if it
        # doesn't land in time, the trimmed video still ships, just
        # without a custom thumbnail (YouTube will auto-pick a frame).
        thumb_path: Optional[str] = None
        if thumbnail_strategy == "ai" and openai_api_key:
            state.set_step(jid, "thumbnail", 70, "Generating AI thumbnail (gpt-image-1)…")
            tp = trimmed_path + ".thumb.png"
            try:
                ok = ai_image.generate_thumbnail(
                    api_key=openai_api_key,
                    title=title,
                    brief=brief,
                    names_hint=names_hint,
                    output_path=tp,
                )
            except Exception as exc:
                state.append_log(jid, f"[thumbnail] exception: {exc}")
                ok = None
            if ok:
                thumb_path = ok
                cleanups.append(tp)
                state.append_log(jid, f"[thumbnail] generated {os.path.getsize(tp)} bytes")
            else:
                state.append_log(jid, "[thumbnail] gen failed — shipping without thumbnail")

        # 5) Postiz upload of the TRIMMED video
        state.set_step(jid, "upload", 78, "Uploading trimmed video to Postiz…")
        from clients import postiz as postiz_client
        try:
            media = postiz_client.upload_file(trimmed_path, "video/mp4")
        except postiz_client.PostizError as exc:
            raise RuntimeError(f"Postiz upload failed: {exc}") from exc

        media_id   = media.get("id", "")
        media_path = media.get("path", "")
        state.append_log(jid, f"[upload] media_id={media_id} ({len(integration_ids)} integrations)")

        # Upload the thumbnail file too, if we have one. Postiz returns
        # a separate file id — we currently don't pass it to schedule_post
        # because Kaizer's postiz client signature doesn't accept it
        # (Session 4 enhancement if needed). For now we log the URL.
        thumb_url = ""
        if thumb_path:
            try:
                t_media = postiz_client.upload_file(thumb_path, "image/png")
                thumb_url = t_media.get("path", "")
                state.append_log(jid, f"[thumbnail] uploaded → {thumb_url}")
            except postiz_client.PostizError as exc:
                state.append_log(jid, f"[thumbnail] postiz upload skipped: {exc}")

        # 6) Postiz post creation
        state.set_step(jid, "publish", 90, "Creating Postiz posts…")
        post_type = "scheduled" if schedule_at_iso else "now"
        post_resp = _schedule_post_chunked(
            integration_ids=integration_ids,
            log_prefix="[publish]",
            append_log=lambda m, _jid=jid: state.append_log(_jid, m),
            text=description or title,
            media_id=media_id,
            media_path=media_path,
            schedule_at_iso=schedule_at_iso,
            type_=post_type,
            yt_title=title,
            yt_privacy=privacy or "private",
            yt_tags=tags,
            yt_made_for_kids=bool(made_for_kids),
        )
        if not post_resp:
            raise RuntimeError("Postiz post creation failed on every chunk")

        # 7) Done
        state.mark_done(jid, {
            "mode":             "ai-trim",
            "title":            title,
            "description":      description,
            "tags":             tags,
            "integration_ids":  integration_ids,
            "scheduled_at":     schedule_at_iso,
            "color_grade":      color_grade_preset,
            "cinematic_edit":   bool(cinematic_edit),
            "duration_s":       duration,
            "trimmed_s":        kept_total,
            "removed_s":        plan.get("removedSeconds") or (duration - kept_total),
            "trim_summary":     plan.get("summary", ""),
            "kept_segments":    len(keep),
            "thumbnail_url":    thumb_url,
            "postiz":           {
                "media_id":  media_id,
                "media_url": media_path,
                "post":      post_resp,
            },
        })

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[express] job={jid} (ai-trim) FAILED:\n{tb}")
        state.mark_failed(jid, str(exc))
    finally:
        _cleanup(cleanups)
        _release_slot()


# ─── Shorts mode ────────────────────────────────────────────────────
#
# 1. Audio extract + Whisper (with segments)
# 2. Claude SEO (shared title/description used for the parent video)
# 3. Claude shorts plan — picks 3-5 self-contained moments
# 4. For each clip:
#       - extract midpoint frame as inset (Session 3 will add AI gen)
#       - cut_clip → 1080×1920 with TV news split panel
#       - Postiz upload + post creation
# 5. Mark done with one results entry per published short

def run_shorts(
    *,
    jid: str,
    video_path: str,
    integration_ids: list[str],
    brief: str = "",
    names_hint: str = "",
    style_guide: str = "",
    language: Optional[str] = None,
    privacy: str = "private",
    made_for_kids: bool = False,
    title_override: Optional[str] = None,
    description_override: Optional[str] = None,
    tags_override: Optional[list[str]] = None,
    schedule_at_iso: Optional[str] = None,
    anthropic_api_key: str = "",
    transcription_provider: str = "groq",
    transcription_api_key: str = "",
    transcription_base_url: Optional[str] = None,
    transcription_model: Optional[str] = None,
    color_grade_preset: str = "subtle",
    cinematic_edit: bool = False,
    panel_color: str = "#dc2626",
    footer_text: str = "KAIZER NEWS NETWORK",
    logo_path: Optional[str] = None,
    short_count_override: Optional[int] = None,
    # Session 3 additions:
    openai_api_key: str = "",                 # for gpt-image-1 inset photos
    inset_strategy: str = "frame",            # frame | ai (per-short AI inset)
    layout: str = "news",                     # news (TV split-panel) | branded (blurred bg)
    logo_corner: str = "top-right",           # branded layout only
) -> None:
    """Full Shorts pipeline. Same contract as the other run_* funcs."""
    cleanups: list[str] = []
    _acquire_slot(jid)
    try:
        if not os.path.isfile(video_path):
            raise RuntimeError(f"video_path not found: {video_path}")
        if not integration_ids:
            raise RuntimeError("no Postiz integrations selected")

        # 1) Audio + Whisper (need segments for shorts planner)
        state.set_step(jid, "transcribe", 5, "Extracting audio…")
        audio_path = video_path + ".mp3"
        whisper.extract_audio_mp3(video_path, audio_path)
        cleanups.append(audio_path)

        state.set_step(jid, "transcribe", 12, "Transcribing (Whisper)…")
        wp_parts = [p for p in (brief, names_hint) if p]
        whisper_prompt = ("\n".join(wp_parts))[:800] if wp_parts else None
        try:
            tr = whisper.transcribe(
                audio_path,
                api_key=transcription_api_key,
                provider=transcription_provider,
                base_url=transcription_base_url,
                model=transcription_model,
                language=language or None,
                names_hint=whisper_prompt,
            )
        except whisper.WhisperError as exc:
            raise RuntimeError(f"transcription failed: {exc}") from exc

        transcript = tr.get("text", "") or ""
        segments   = tr.get("segments") or []
        duration   = float(tr.get("duration") or 0.0)
        tr_lang    = tr.get("language") or language or ""
        state.append_log(jid, f"[transcribe] {len(transcript)} chars, {len(segments)} segments, {duration:.1f}s")

        if not segments:
            raise RuntimeError("Whisper returned no segments — can't plan shorts")

        # 2) Claude SEO (for the parent — used as fallback per-short title)
        state.set_step(jid, "plan", 20, "Writing SEO (Claude)…")
        try:
            seo = claude.write_seo(
                api_key=anthropic_api_key,
                transcript=transcript,
                brief=brief,
                names_hint=names_hint,
                style_guide=style_guide,
            )
        except claude.ClaudeError as exc:
            raise RuntimeError(f"SEO generation failed: {exc}") from exc

        # 3) Claude shorts plan
        n_shorts = short_count_override or claude.target_short_count(duration)
        state.set_step(jid, "plan", 32, f"Planning {n_shorts} shorts (Claude)…")
        try:
            plan = claude.plan_shorts(
                api_key=anthropic_api_key,
                segments=segments,
                language=tr_lang,
                duration=duration,
                count=n_shorts,
                style_guide=style_guide,
            )
        except claude.ClaudeError as exc:
            raise RuntimeError(f"shorts plan failed: {exc}") from exc
        if not plan:
            raise RuntimeError("Claude returned an empty shorts plan")
        state.append_log(jid, f"[plan] {len(plan)} shorts selected")

        # 4) Cut + publish each short
        from clients import postiz as postiz_client
        results: list[dict] = []
        post_type = "scheduled" if schedule_at_iso else "now"

        for i, clip in enumerate(plan):
            base = 35 + int((i / max(len(plan), 1)) * 55)

            # Optional AI inset per short. Soft-fail to video frame.
            custom_snap: Optional[str] = None
            if inset_strategy == "ai" and openai_api_key and clip.get("imagePrompt"):
                state.set_step(jid, "render-shorts", base,
                               f"Short {i+1}/{len(plan)}: generating AI inset image (gpt-image-1)…")
                snap_path = video_path + f".short-{i:02d}.inset.png"
                try:
                    ok = ai_image.generate_short_inset(
                        api_key=openai_api_key,
                        raw_prompt=clip["imagePrompt"],
                        output_path=snap_path,
                    )
                except Exception as exc:
                    state.append_log(jid, f"[short {i+1}] inset gen exception: {exc}")
                    ok = None
                if ok:
                    custom_snap = ok
                    cleanups.append(snap_path)
                    state.append_log(jid, f"[short {i+1}] AI inset ready ({os.path.getsize(snap_path)} bytes)")
                else:
                    state.append_log(jid, f"[short {i+1}] AI inset gen failed — falling back to video frame")

            state.set_step(jid, "render-shorts", base,
                           f"Short {i+1}/{len(plan)}: cutting [{clip['start']:.1f}–{clip['end']:.1f}s]…")

            clip_out = video_path + f".short-{i:02d}.mp4"
            # Don't auto-cleanup the rendered Shorts — preserve them
            # in a debug dir so the user can VIEW the news-panel layout
            # before / regardless of Postiz upload success. Same dir
            # the failed-audio rescue uses.
            try:
                cut_clip.cut_clip(
                    input_path=video_path,
                    output_path=clip_out,
                    start_sec=clip["start"],
                    end_sec=clip["end"],
                    title=clip.get("title", ""),
                    hook=clip.get("hook", ""),
                    logo_path=logo_path,
                    panel_color=panel_color,
                    footer_text=footer_text,
                    custom_snap_path=custom_snap,
                    color_grade=color_grade_preset,
                    cinematic_edit=bool(cinematic_edit),
                    layout=layout,
                    logo_corner=logo_corner,
                )
            except cut_clip.CutClipError as exc:
                state.append_log(jid, f"[short {i+1}] cut failed: {exc}")
                continue

            # Preserve a copy of the rendered Short in a debug dir so
            # the user can preview the news-panel layout regardless of
            # Postiz upload success/fail. Idempotent — overwrites prior
            # renders of the same short index.
            try:
                import shutil as _sh
                debug_dir = os.path.join(os.path.dirname(video_path), "_express_renders")
                os.makedirs(debug_dir, exist_ok=True)
                kept = os.path.join(debug_dir, f"{jid}-short-{i:02d}.mp4")
                _sh.copy2(clip_out, kept)
                state.append_log(jid, f"[short {i+1}] render saved → {kept}")
            except OSError as exc:
                state.append_log(jid, f"[short {i+1}] could not save preview copy: {exc}")

            # Postiz upload + post
            state.set_step(jid, "upload", base + 3,
                           f"Short {i+1}/{len(plan)}: uploading to Postiz…")
            try:
                media = postiz_client.upload_file(clip_out, "video/mp4")
            except postiz_client.PostizError as exc:
                state.append_log(jid, f"[short {i+1}] postiz upload failed: {exc}")
                continue

            short_title = (clip.get("title") or seo.get("title") or f"Short {i+1}")[:100]
            short_descr = clip.get("description") or seo.get("description") or ""
            short_tags  = clip.get("tags") or seo.get("tags") or []

            post_resp = _schedule_post_chunked(
                integration_ids=integration_ids,
                log_prefix=f"[short {i+1}]",
                append_log=lambda m, _jid=jid: state.append_log(_jid, m),
                text=short_descr or short_title,
                media_id=media.get("id", ""),
                media_path=media.get("path", ""),
                schedule_at_iso=schedule_at_iso,
                type_=post_type,
                yt_title=short_title,
                yt_privacy=privacy or "private",
                yt_tags=short_tags,
                yt_made_for_kids=bool(made_for_kids),
            )
            if not post_resp:
                state.append_log(jid, f"[short {i+1}] postiz post failed on every chunk")
                continue

            results.append({
                "index":     i,
                "title":     short_title,
                "start":     clip["start"],
                "end":       clip["end"],
                "subject":   clip.get("subject", ""),
                "media_id":  media.get("id", ""),
                "media_url": media.get("path", ""),
                "post":      post_resp,
            })
            state.append_log(jid, f"[short {i+1}] published — title={short_title!r}")

        if not results:
            raise RuntimeError("all shorts failed to render or upload — see log_tail")

        state.mark_done(jid, {
            "mode":            "shorts",
            "integration_ids": integration_ids,
            "scheduled_at":    schedule_at_iso,
            "color_grade":     color_grade_preset,
            "cinematic_edit":  bool(cinematic_edit),
            "panel_color":     panel_color,
            "footer_text":     footer_text,
            "duration_s":      duration,
            "shorts":          results,
            "shorts_planned":  len(plan),
            "shorts_published": len(results),
        })

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[express] job={jid} (shorts) FAILED:\n{tb}")
        state.mark_failed(jid, str(exc))
    finally:
        _cleanup(cleanups)
        _release_slot()
