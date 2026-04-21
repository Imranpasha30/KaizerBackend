"""After an upload completes, variant 0 is applied at upload time.
At +6h, the swapper inspects early performance (views). If views are flat
(below a floor), it calls `thumbnails.set` with variant 1 and marks the
row `swapped_in`. This is a cheap experiment — single swap, not full A/B.

Called hourly from learning/scheduler.py.
"""
from __future__ import annotations

import traceback
from datetime import datetime, timedelta, timezone

from googleapiclient.errors import HttpError
from sqlalchemy.orm import Session

import models
from database import SessionLocal
from youtube import oauth as yt_oauth, uploader


SWAP_AFTER_HOURS = 6
VIEW_FLOOR = 200   # if fewer than this after 6h, try the alternate thumbnail


def run_once() -> dict:
    """Single sweep — find `served` primaries whose upload is >6h old and views
    are below the floor; swap in the alternate thumbnail."""
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=SWAP_AFTER_HOURS)

        # Pick uploads that: have variant-0 served, have variant-1 alternate available
        alt_rows = (
            db.query(models.ThumbnailVariant)
              .filter(models.ThumbnailVariant.status == "alternate")
              .all()
        )
        swapped = 0
        errors = 0

        for alt in alt_rows:
            upload = db.query(models.UploadJob).filter(
                models.UploadJob.id == alt.upload_job_id,
                models.UploadJob.status == "done",
                models.UploadJob.video_id != "",
            ).first()
            if not upload or not upload.updated_at:
                continue

            upd = upload.updated_at if upload.updated_at.tzinfo else upload.updated_at.replace(tzinfo=timezone.utc)
            if upd > cutoff:
                continue  # not old enough

            # Find the latest perf sample
            perf = (
                db.query(models.ClipPerformance)
                  .filter(models.ClipPerformance.upload_job_id == upload.id)
                  .order_by(models.ClipPerformance.sampled_at.desc())
                  .first()
            )
            views = perf.views if perf else 0

            if views >= VIEW_FLOOR:
                alt.status = "skipped"
                alt.views_at_swap = views
                db.commit()
                continue

            # Under floor → swap in alternate thumbnail
            try:
                creds = yt_oauth.get_credentials(db, upload.channel_id)
                uploader.set_thumbnail(creds, upload.video_id, alt.image_path or upload.clip.thumb_path)
                alt.status = "swapped_in"
                alt.swapped_at = now
                alt.views_at_swap = views
                # Mark the original primary as swapped_out
                primary = (
                    db.query(models.ThumbnailVariant)
                      .filter(models.ThumbnailVariant.upload_job_id == upload.id,
                              models.ThumbnailVariant.variant_idx == 0)
                      .first()
                )
                if primary:
                    primary.status = "swapped_out"
                db.commit()
                swapped += 1
            except (HttpError, yt_oauth.OAuthError, uploader.UploadError) as e:
                alt.status = "pending"
                alt.views_at_swap = views
                db.commit()
                errors += 1
                print(f"[thumb-ab] swap failed upload={upload.id}: {e}")
            except Exception as e:
                errors += 1
                print(f"[thumb-ab] unexpected swap error upload={upload.id}: {e}")

        return {"ran_at": now.isoformat(), "swapped": swapped, "errors": errors, "candidates": len(alt_rows)}
    except Exception:
        traceback.print_exc()
        return {"ran_at": datetime.now(timezone.utc).isoformat(), "error": "swap loop failed"}
    finally:
        db.close()
