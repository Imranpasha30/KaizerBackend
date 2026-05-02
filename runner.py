import os
import sys
import json
import threading
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR        = Path(__file__).parent
PIPELINE_SCRIPT = BASE_DIR / "pipeline_core" / "pipeline.py"
# Use /tmp so clips survive the session but don't fill the deploy image
OUTPUT_ROOT     = Path(os.getenv("KAIZER_OUTPUT_ROOT", "/tmp/kaizer_output"))


def run_pipeline(job_id: int, video_path: str, platform: str, frame: str,
                 db_session_factory, language: str = "te",
                 default_image: str = "",
                 default_logo: str = ""):
    """Launch pipeline as subprocess, stream stdout into Job.log.

    - `default_image` (non-empty absolute path) → the pipeline uses this
      image for every clip instead of fetching stock photos.
    - `default_logo` (non-empty absolute path) → the pipeline overlays this
      logo on every clip video.  Empty / missing = NO logo overlay.
    """

    def _run():
        from models import Job, Clip

        db = db_session_factory()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            job.status = "running"
            from datetime import datetime as _dt, timezone as _tz
            job.started_at = _dt.now(_tz.utc)
            db.commit()

            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

            # pipeline.py uses positional 'video', --platform, --frame, --language.
            # -u forces unbuffered stdout so log lines stream through the pipe
            # in real time instead of sitting in Python's 8KB block buffer.
            cmd = [
                sys.executable, "-u", str(PIPELINE_SCRIPT),
                video_path,
                "--platform", platform,
                "--frame", frame,
                "--language", language,
            ]
            if default_image:
                cmd += ["--default-image", default_image]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                # On Windows `text=True` defaults to cp1252 which crashes as
                # soon as the child emits Telugu / Hindi / any non-ASCII byte.
                # Match the child's PYTHONIOENCODING=utf-8 explicitly, and
                # replace any rogue bytes instead of raising.
                encoding="utf-8",
                errors="replace",
                env={
                    **os.environ,
                    "KAIZER_OUTPUT_ROOT": str(OUTPUT_ROOT),
                    # Per-job logo path resolved from channel.logo_asset —
                    # pipeline reads this and overlays on every clip.  Empty
                    # = no logo overlay (deliberate SaaS default).
                    "KAIZER_DEFAULT_LOGO": default_logo or "",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
                cwd=str(BASE_DIR),
            )

            log_lines = []
            captured_meta_path: str = ""
            for line in process.stdout:
                stripped = line.rstrip()
                log_lines.append(stripped)
                # Sniff for the machine-parseable marker line the pipeline emits
                # right after writing editor_meta.json.  Canonical source for
                # import — removes rglob-picks-a-stale-file ambiguity.
                if stripped.startswith("[kaizer:meta] "):
                    captured_meta_path = stripped[len("[kaizer:meta] "):].strip()
                # Flush every line for live log visibility (testing mode).
                # Swap back to `% 5 == 0` for less DB load in production.
                db2 = db_session_factory()
                j = db2.query(Job).filter(Job.id == job_id).first()
                if j:
                    j.log = "\n".join(log_lines)
                    db2.commit()
                db2.close()

            process.wait()

            db2 = db_session_factory()
            j = db2.query(Job).filter(Job.id == job_id).first()
            j.log = "\n".join(log_lines)

            if process.returncode == 0:
                try:
                    meta_override = Path(captured_meta_path) if captured_meta_path else None
                    _import_clips(j, db2, meta_override=meta_override)
                    if not j.clips:
                        j.status = "failed"
                        j.error = ("Pipeline finished but 0 clips were imported. "
                                   "Check editor_meta.json and the runner log.")
                    else:
                        j.status = "done"
                        j.error = ""
                except Exception as import_err:
                    j.status = "failed"
                    j.error = f"Clip import failed: {import_err}"
            else:
                j.status = "failed"
                j.error = "\n".join(log_lines[-20:])   # last 20 lines as error summary

            # Stamp completion wall-clock time on every terminal state.
            from datetime import datetime as _dt, timezone as _tz
            j.finished_at = _dt.now(_tz.utc)
            db2.commit()
            db2.close()

            # Auto-enqueue any campaigns attached to this job (Phase A).
            if process.returncode == 0:
                try:
                    from campaigns import orchestrator as _campaigns_orch
                    _campaigns_orch.auto_enqueue_async(job_id)
                except Exception as e:
                    print(f"[runner] campaign auto-enqueue skipped: {e}")

        except Exception as e:
            db3 = db_session_factory()
            j = db3.query(Job).filter(Job.id == job_id).first()
            if j:
                from datetime import datetime as _dt, timezone as _tz
                j.status = "failed"
                j.error = str(e)
                j.finished_at = _dt.now(_tz.utc)
                db3.commit()
            db3.close()
        finally:
            # ─── Aggressive cleanup ─────────────────────────────────────
            # Railway's container has a small ephemeral disk cap. Without
            # this the disk fills up after a handful of jobs and the
            # container crashes ("Container is exceeding maximum
            # ephemeral storage"). All bytes worth keeping are in R2 by
            # this point — the source video at sources/{user}/...,
            # rendered clips + thumbs + images at clips/{job}/{idx}/...
            # Anything still on local disk is now dead weight.
            try:
                import shutil as _shutil
                # 1. Source video that create_job dropped in MEDIA_ROOT/uploads/
                if video_path and os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except OSError as _e:
                        print(f"[runner] cleanup: failed to remove source {video_path!r}: {_e}")
                # 2. Pipeline output dir for this run (job.output_dir set by _import_clips)
                try:
                    db_clean = db_session_factory()
                    j_clean = db_clean.query(Job).filter(Job.id == job_id).first()
                    out_dir = j_clean.output_dir if j_clean else ""
                    db_clean.close()
                    if out_dir and os.path.isdir(out_dir):
                        # Don't delete OUTPUT_ROOT itself, only the per-job subfolder
                        if os.path.abspath(out_dir) != os.path.abspath(str(OUTPUT_ROOT)):
                            _shutil.rmtree(out_dir, ignore_errors=True)
                            print(f"[runner] cleanup: dropped pipeline output {out_dir!r}")
                except Exception as _e:
                    print(f"[runner] cleanup: output_dir removal warning: {_e}")
            except Exception as _cleanup_e:
                # Cleanup failures are never fatal — container will get
                # wiped on next redeploy worst case.
                print(f"[runner] cleanup warning: {_cleanup_e}")
            db.close()

    threading.Thread(target=_run, daemon=True).start()


def _import_clips(job, db, meta_override: Path | None = None):
    """Find editor_meta.json, read it as UTF-8, create Clip rows.

    Priority for locating the metadata file:
      1. Explicit `meta_override` (set by the runner from the `[kaizer:meta]`
         marker line emitted by pipeline.py — canonical source).
      2. `job.output_dir` if it points directly at a directory containing the file.
      3. rglob from `job.output_dir`, then rglob from the global OUTPUT_ROOT.

    Exceptions propagate to the caller so the runner can mark the job failed
    with a useful message instead of silently ending up with zero clips.
    """
    from models import Clip

    meta_path: Path | None = None

    if meta_override and Path(meta_override).exists():
        meta_path = Path(meta_override)
    else:
        search_root = Path(job.output_dir) if job.output_dir else OUTPUT_ROOT
        # Direct-hit: the dir itself contains editor_meta.json
        direct = search_root / "editor_meta.json"
        if direct.exists():
            meta_path = direct
        else:
            for p in search_root.rglob("editor_meta.json"):
                meta_path = p
                break
        if not meta_path or not meta_path.exists():
            for p in OUTPUT_ROOT.rglob("editor_meta.json"):
                meta_path = p
                break

    if not meta_path or not meta_path.exists():
        raise FileNotFoundError(
            f"editor_meta.json not found (searched override={meta_override!r}, "
            f"job.output_dir={job.output_dir!r}, OUTPUT_ROOT={OUTPUT_ROOT!s})"
        )

    # Must be utf-8 — editor_meta.json now contains native-script fields
    # (Telugu, Devanagari, Tamil, …).  Windows default cp1252 will crash.
    raw = meta_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    # Lazy R2 import — pipeline-emitted images get mirrored to R2 so they
    # survive container restarts AND don't break when the user opens the
    # clip from a different machine. Each image lands at a clip-specific
    # key (clips/{job_id}/{clip_index}/<filename>) so the mapping is
    # permanent — even if the user later changes their default ad asset
    # or deletes the source file, the clip's image_storage_url still
    # resolves to the exact image used at render time.
    try:
        from pipeline_core.storage import get_storage_provider
        _r2 = get_storage_provider("r2")
    except Exception as exc:
        print(f"[runner] R2 provider unavailable, images will only live "
              f"on local disk: {exc}")
        _r2 = None

    def _r2_upload(local_path: str, key: str, ct: str) -> str:
        """Upload to R2 with permanent clip-specific key. Returns URL or
        empty string on failure / no R2 / no file."""
        if not _r2 or not local_path or not Path(local_path).exists():
            return ""
        try:
            return _r2.upload(local_path, key, content_type=ct).url
        except Exception as exc:
            print(f"[runner] R2 upload failed for {local_path!r}: {exc}")
            return ""

    clips_data = data.get("clips", [])
    imported = 0
    skipped  = 0
    for i, c in enumerate(clips_data):
        clip_path = c.get("clip_path", "")
        # Drop clips whose mp4 file is missing on disk — avoids broken cards
        # in the UI that 404 on every thumb + video fetch.
        if not clip_path or not Path(clip_path).exists():
            print(f"[runner] skipping clip {i}: file missing ({clip_path!r})")
            skipped += 1
            continue

        thumb_path = c.get("thumb_path", "") or ""
        image_path = c.get("image_path", "") or ""

        # Pipeline already populates storage_* for the video clip. We add
        # mirrors for the thumbnail + the editorial image so all three
        # bytes-on-disk artefacts of this clip live in R2.
        thumb_storage_url = _r2_upload(
            thumb_path,
            f"clips/{job.id}/{i:02d}/{Path(thumb_path).name}" if thumb_path else "",
            "image/jpeg",
        )
        image_storage_url = ""
        if image_path:
            ext = Path(image_path).suffix.lower()
            ct = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(ext, "image/jpeg")
            image_storage_url = _r2_upload(
                image_path,
                f"clips/{job.id}/{i:02d}/{Path(image_path).name}",
                ct,
            )

        clip = Clip(
            job_id=job.id,
            clip_index=i,
            filename=Path(clip_path).name,
            file_path=clip_path,
            thumb_path=thumb_path,
            image_path=image_path,
            duration=float(c.get("duration", 0)),
            frame_type=c.get("frame_type", ""),
            text=c.get("text", ""),
            sentiment=c.get("sentiment", ""),
            entities=json.dumps(c.get("entities", [])),
            card_params=json.dumps(c.get("card_params", {})),
            section_pct=json.dumps(c.get("section_pct", {})),
            follow_params=json.dumps(c.get("follow_params", {})),
            meta=json.dumps(c),
            # Phase 5 storage fields — empty strings when backend is local
            storage_url=c.get("storage_url", ""),
            storage_key=c.get("storage_key", ""),
            storage_backend=c.get("storage_backend", ""),
            # Image mirrors — permanent clip-specific R2 keys so the
            # mapping survives default-asset rotation and container
            # rebuilds.
            thumb_storage_url=thumb_storage_url,
            image_storage_url=image_storage_url,
        )
        db.add(clip)
        imported += 1

    job.output_dir = str(meta_path.parent)
    db.commit()
    print(f"[runner] imported {imported} clip(s), skipped {skipped} for job {job.id}")
