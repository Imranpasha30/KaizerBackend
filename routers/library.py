"""Company Library router — shared pool of source videos.

Creatives (and admins) upload videos here once; every logged-in user
can browse the grid and click "Use" to start a new job from that
video, going through the same NewJob wizard they would for a personal
upload.

Storage is intentionally R2-only (the cloud bucket is the single source
of truth) so dev and prod instances read the same library. Local file
storage is NOT supported here even when ``STORAGE_BACKEND=local`` is the
process-wide default — the call sites pin ``backend='r2'`` explicitly.

Endpoints
---------
GET    /api/library/                       — list (all users)
GET    /api/library/{item_id}              — single item
DELETE /api/library/{item_id}              — soft delete (uploader or admin)
POST   /api/library/upload-session         — issue a one-time signed token
                                              (creative + admin only)
POST   /api/library/upload/{token}         — upload using the token. The token
                                              is single-use, 5-min TTL, bound
                                              to the issuing user, and rejects
                                              cross-origin requests. So even
                                              if an attacker copies the cURL
                                              from devtools, the URL is dead
                                              within minutes or after one POST.

Hardening layers on every upload
--------------------------------
1.  Auth + role gate (creative or admin).
2.  Single-use HMAC-signed upload token in the URL path.
3.  Origin header allow-list (rejects cross-origin replays).
4.  Per-user sliding-window rate limit (10 uploads / 10 min).
5.  Filename sanitisation (no path components, restricted charset).
6.  Magic-byte content check (file MUST actually start with a known
    video container header — defeats "rename .exe to .mp4" attempts).
7.  Hard 4 GiB size cap.
8.  Audit log of every accept + reject decision.
"""
from __future__ import annotations

import hmac
import hashlib
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

import auth
import models
from database import get_db
from pipeline_core.storage import get_storage_provider

router = APIRouter(prefix="/api/library", tags=["library"])

log     = logging.getLogger("kaizer.library")
audit   = logging.getLogger("kaizer.library.audit")

ALLOWED_VIDEO_MIMES = {
    "video/mp4", "video/quicktime", "video/x-matroska",
    "video/webm", "video/x-msvideo",
}
ALLOWED_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
MAX_BYTES   = 4 * 1024 * 1024 * 1024   # 4 GiB hard cap

# ─── Hardening: secrets, rate limit, origin allow-list ──────────────────

# HMAC key for signing one-time upload tokens. ALWAYS read from env in
# prod; falling back to a fresh per-process random key in dev means
# tokens issued by one worker won't validate on another — fine for a
# single-worker uvicorn but logs a loud warning so a multi-worker
# deploy notices.
_UPLOAD_SIGNING_KEY = os.environ.get("KAIZER_UPLOAD_SIGNING_KEY", "").encode()
if not _UPLOAD_SIGNING_KEY:
    _UPLOAD_SIGNING_KEY = secrets.token_bytes(32)
    log.warning(
        "KAIZER_UPLOAD_SIGNING_KEY is not set — using an ephemeral per-process "
        "key. Tokens won't survive a restart and will fail in multi-worker "
        "deployments. Set the env var to a 64-char hex string in production."
    )

# In-memory single-use token store. token_str -> (user_id, expires_at_epoch).
# Tokens are deleted on consume. A periodic GC pass removes expired entries.
_token_store: dict[str, tuple[int, float]] = {}
_token_lock = threading.Lock()
_TOKEN_TTL_SECONDS = 5 * 60   # 5 minutes

# Sliding-window rate limit: user_id -> deque[float] of upload timestamps.
_recent_uploads: dict[int, deque] = {}
_recent_lock    = threading.Lock()
RATE_LIMIT_WINDOW_S = 600   # 10 min
RATE_LIMIT_MAX      = 10    # uploads per window per user

# Origin allow-list. Comma-separated list of scheme://host[:port] strings.
# Empty value = derived defaults (localhost dev + common prod hosts).
# Includes a permissive dev fallback so the check doesn't lock out
# laptops on weird ports during local work.
_ALLOWED_ORIGINS_RAW = os.environ.get("KAIZER_ALLOWED_ORIGINS", "").strip()
if _ALLOWED_ORIGINS_RAW:
    _ALLOWED_ORIGINS = {o.strip().rstrip("/") for o in _ALLOWED_ORIGINS_RAW.split(",") if o.strip()}
else:
    _ALLOWED_ORIGINS = set()  # interpreted as "any localhost"

# Magic-byte sniffers for the video container formats we accept.
# First 16 bytes of the file are read and matched against these patterns.
def _looks_like_video(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return False
    if len(head) < 8:
        return False
    # MP4 / MOV: "ftyp" box at offset 4
    if head[4:8] == b"ftyp":
        return True
    # MKV / WebM: EBML header
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return True
    # AVI: RIFF....AVI
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return True
    return False


def _safe_filename(raw: str) -> str:
    """Strip path components, restrict to safe charset, cap length."""
    base = os.path.basename((raw or "").strip())
    # Drop control chars + filesystem-hostile metachars; collapse repeats.
    safe = re.sub(r"[^\w \-.()]", "_", base)
    safe = re.sub(r"_{2,}", "_", safe).strip("._ ") or "upload.mp4"
    # Cap stem + extension lengths independently so a 250-char "name" can't
    # blow past filesystem limits.
    if "." in safe:
        stem, _, ext = safe.rpartition(".")
        ext = ext[:10].lower()
        stem = stem[:80]
        safe = f"{stem}.{ext}" if stem else f"upload.{ext}"
    else:
        safe = safe[:80] + ".mp4"
    return safe


def _check_origin(request: Request) -> None:
    """Reject cross-origin requests. Matches Origin header against the
    allow-list. Falls back to "any localhost" when the env var is empty
    (dev-friendly default)."""
    origin = (request.headers.get("origin") or "").strip().rstrip("/")
    if not origin:
        # No Origin = curl / Postman / native app. We rely on the
        # one-time token + auth to keep these safe; the missing header
        # alone is not a reason to reject (would break legitimate API
        # consumers).
        return
    if _ALLOWED_ORIGINS:
        if origin not in _ALLOWED_ORIGINS:
            audit.warning("origin reject: %r not in allow-list", origin)
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Origin not allowed")
        return
    # Default: only localhost during dev. Any prod deploy MUST set the env.
    if not (origin.startswith("http://localhost") or
            origin.startswith("http://127.0.0.1") or
            origin.startswith("https://localhost")):
        audit.warning("origin reject (no allow-list set): %r", origin)
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Origin not allowed. Set KAIZER_ALLOWED_ORIGINS in the server env.",
        )


def _rate_limit_check(user_id: int) -> None:
    """Sliding window. Raises 429 when the user has burst past the cap."""
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_S
    with _recent_lock:
        dq = _recent_uploads.setdefault(user_id, deque())
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= RATE_LIMIT_MAX:
            wait_s = int(dq[0] + RATE_LIMIT_WINDOW_S - now) + 1
            audit.warning("rate limit hit: user=%d count=%d", user_id, len(dq))
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"Upload rate limit exceeded ({RATE_LIMIT_MAX} per "
                f"{RATE_LIMIT_WINDOW_S // 60} min). Try again in {wait_s}s.",
            )
        dq.append(now)


def _gc_expired_tokens() -> None:
    now = time.time()
    with _token_lock:
        dead = [k for k, (_uid, exp) in _token_store.items() if exp < now]
        for k in dead:
            _token_store.pop(k, None)


def _issue_upload_token(user_id: int) -> tuple[str, float]:
    """Issue a single-use upload token for *user_id*. The token is the
    HMAC-SHA256 of (user_id || nonce || expires_at) plus a server-side
    record marking it active. Returns (token_str, expires_at_epoch)."""
    _gc_expired_tokens()
    nonce  = secrets.token_urlsafe(24)
    expires_at = time.time() + _TOKEN_TTL_SECONDS
    payload = f"{user_id}.{nonce}.{int(expires_at)}".encode()
    sig = hmac.new(_UPLOAD_SIGNING_KEY, payload, hashlib.sha256).hexdigest()
    token = f"{nonce}.{int(expires_at)}.{sig}"
    with _token_lock:
        _token_store[token] = (user_id, expires_at)
    return token, expires_at


def _consume_upload_token(token: str, user_id: int) -> None:
    """Verify the token and mark it consumed (single-use). Raises 401 on
    any failure. Verifies HMAC signature, expiry, and binding to caller."""
    _gc_expired_tokens()
    if not token or token.count(".") != 2:
        audit.warning("bad token format from user=%d", user_id)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid upload token")
    try:
        nonce, exp_str, sig = token.split(".")
        expires_at = int(exp_str)
    except (ValueError, TypeError):
        audit.warning("bad token parse from user=%d", user_id)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid upload token")

    expected = hmac.new(
        _UPLOAD_SIGNING_KEY,
        f"{user_id}.{nonce}.{expires_at}".encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, sig):
        audit.warning("token sig mismatch user=%d", user_id)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid upload token")
    if expires_at < time.time():
        audit.warning("token expired user=%d", user_id)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Upload token expired")
    # Single-use: pop the entry. If it's already gone, the token was used.
    with _token_lock:
        entry = _token_store.pop(token, None)
    if entry is None:
        audit.warning("token replay user=%d", user_id)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Upload token already used")
    bound_user_id, _ = entry
    if bound_user_id != user_id:
        audit.warning("token user mismatch issued_for=%d used_by=%d",
                      bound_user_id, user_id)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Upload token user mismatch")


# ─── Role gates ─────────────────────────────────────────────────────────

def _require_creative(user: models.User) -> None:
    """Library uploads are restricted to creative + admin roles."""
    if not (getattr(user, "is_creative", False) or getattr(user, "is_admin", False)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Library uploads are restricted to creative users.",
        )


# ─── Storage helpers ────────────────────────────────────────────────────

def _r2():
    """Return the R2 storage provider, pinned regardless of env override.

    The Library MUST be R2 — local files would defeat the "single source
    of truth across environments" guarantee. Raises HTTPException(503) if
    R2 isn't configured so the failure is visible at the API surface.
    """
    try:
        return get_storage_provider(backend="r2")
    except Exception as exc:   # ValueError when env vars missing
        log.error("library: R2 storage unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Library storage (Cloudflare R2) is not configured on this "
                "server. Set R2_BUCKET, R2_ENDPOINT, R2_ACCESS_KEY_ID, "
                "R2_SECRET_ACCESS_KEY in the environment and restart."
            ),
        ) from exc


# Object-key layout
# -----------------
# library/                       ← top-level folder for the company library
#   users/<user_id>/             ← one folder per uploader (visible audit trail
#                                  in the R2 dashboard — easy to tell who put
#                                  what in the bucket)
#     <item_id>/                 ← one folder per LibraryItem
#       video.<ext>
#       thumb.jpg
#
# Pre-existing rows uploaded before this layout was introduced keep their
# legacy key (``library/<id>/...``) in the DB — fetch_library_video_to
# and the serializer read the key off the row, so both schemes work.

def _video_key(item_id: int, ext: str, user_id: int) -> str:
    ext = (ext or ".mp4").lower()
    if not ext.startswith("."):
        ext = "." + ext
    return f"library/users/{user_id}/{item_id}/video{ext}"


def _thumb_key(item_id: int, user_id: int) -> str:
    return f"library/users/{user_id}/{item_id}/thumb.jpg"


# ─── ffmpeg / ffprobe shell-outs ───────────────────────────────────────

def _ffprobe(path: str) -> dict:
    """Lightweight duration + resolution probe. Returns empty dict on any
    failure — Library upload must not 500 because the user's file
    confuses ffprobe."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "stream=width,height:format=duration",
             "-of", "default=noprint_wrappers=1:nokey=0",
             path],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return {}
    info = {"duration": 0.0, "width": 0, "height": 0}
    for line in (out.stdout or "").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k == "duration":
            try:    info["duration"] = float(v)
            except: pass
        elif k == "width":
            try:    info["width"] = int(v)
            except: pass
        elif k == "height":
            try:    info["height"] = int(v)
            except: pass
    return info


def _extract_thumb(video_path: str, out_jpg: str, at_seconds: float = 1.0) -> bool:
    """Grab a single frame ~1 s in. Returns True on success."""
    try:
        # Try at the requested timestamp first; if the video is shorter,
        # fall back to t=0.
        for ts in (at_seconds, 0.0):
            rc = subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{ts:.2f}",
                 "-i", video_path,
                 "-vframes", "1", "-q:v", "3", "-vf",
                 "scale=1280:-2:force_original_aspect_ratio=decrease",
                 out_jpg],
                capture_output=True, timeout=60,
            ).returncode
            if rc == 0 and os.path.isfile(out_jpg) and os.path.getsize(out_jpg) > 0:
                return True
    except Exception as exc:
        log.warning("library: ffmpeg thumb extract failed: %s", exc)
    return False


# ─── Anti-download playback ────────────────────────────────────────────
#
# Bare R2 URLs are NEVER baked into list/get responses. Clients must
# POST /{id}/play-ticket on every play attempt — that endpoint returns
# a SHORT-LIVED signed R2 URL (60 s) that the <video> tag consumes
# directly. Why this shape:
#
#  • zero backend bandwidth — Cloudflare R2 serves the bytes; the
#    backend only spends a few ms per play to mint a signed URL
#  • single chokepoint for the future paywall / entitlement check —
#    deny the play-ticket call and the user literally has no URL
#  • URL is dead after 60 s, so devtools-copied curl commands rot fast
#  • <video controlsList="nodownload"> on the frontend nudges casual
#    users away from the "Save video as…" affordance
#
# The thumbnail is left as a normal public/signed URL — leaking a
# still frame is acceptable and the upside is one fewer round-trip
# on every grid card.

_PLAY_TTL_SECONDS     = 60
_DOWNLOAD_TTL_SECONDS = 300   # 5 min — long enough that a slow client can finish

# Free-tier downloads get a watermarked copy. Paid users (plan != "free")
# get the original. We render the watermark lazily on the first free
# download and cache the result on R2 — so popular videos pay the ffmpeg
# cost exactly once, and videos only downloaded by paid users never
# trigger a render at all.
#
# The watermark is baked into pixels via ffmpeg drawtext — there's no
# HTML overlay involved, so it can't be hidden client-side. We stamp:
#   • a prominent corner box "KAIZER X" (bottom-right)
#   • a center watermark at low opacity (makes cropping the corner pointless)


def _is_paid(user) -> bool:
    """A user is 'paid' when:
      • they're an admin (admins inherit paid privileges by policy), OR
      • their ``plan`` is anything other than 'free'.
    Future: extend with subscription expiry check + grace period."""
    if getattr(user, "is_admin", False):
        return True
    plan = (getattr(user, "plan", "") or "").strip().lower()
    return bool(plan) and plan != "free"


def _watermark_key_for(item_id: int, ext: str, user_id: int) -> str:
    ext = (ext or ".mp4").lower()
    if not ext.startswith("."):
        ext = "." + ext
    return f"library/users/{user_id}/{item_id}/video_watermarked{ext}"


def _render_watermark(in_path: str, out_path: str) -> bool:
    """Burn a 'kaizerx.com' watermark into the video. NVENC when
    configured, libx264 fallback. Returns True on success.

    Stylistic choices for a clean, professional look:
      • NotoSans-Bold for crisp Latin glyphs
      • Bottom-right corner — small, white with a subtle drop shadow
        for legibility on bright frames (no harsh box)
      • Center watermark at very low opacity — barely visible to the
        eye, but enough to defeat corner-crop attacks
    """
    # Resolve the bundled font. ffmpeg's libavfilter drawtext parser
    # treats unescaped spaces in the filter string as option separators,
    # so paths with spaces (very common on Windows: ``C:\Program Files``,
    # ``kaizer new data training``…) silently break the filter unless
    # the fontfile value is wrapped in single quotes. We also escape the
    # drive colon and convert backslashes so the parser doesn't try to
    # interpret them as option separators or backreferences. Falls back
    # to no fontfile when the font is missing — ffmpeg will then pick a
    # system default.
    font_path = (
        Path(__file__).resolve().parent.parent / "resources" / "fonts"
        / "NotoSans-Bold.ttf"
    )
    fontfile = ""
    if font_path.is_file():
        ff = str(font_path).replace("\\", "/").replace(":", r"\:")
        # Single quotes around the value let drawtext keep the spaces
        # inside the path as part of the fontfile name.
        fontfile = f"fontfile='{ff}':"

    drawtext_br = (
        f"drawtext={fontfile}"
        f"text='kaizerx.com':"
        f"fontsize=h*0.034:"
        f"fontcolor=white@0.92:"
        f"shadowcolor=black@0.7:shadowx=2:shadowy=2:"
        f"x=w-tw-32:y=h-th-26"
    )
    drawtext_center = (
        f"drawtext={fontfile}"
        f"text='kaizerx.com':"
        f"fontsize=h*0.075:"
        f"fontcolor=white@0.22:"
        f"shadowcolor=black@0.35:shadowx=2:shadowy=2:"
        f"x=(w-tw)/2:y=(h-th)/2"
    )
    vf = f"{drawtext_br},{drawtext_center}"

    encoder = (os.environ.get("KAIZER_VIDEO_ENCODER") or "libx264").lower()
    if encoder == "nvenc":
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    else:
        codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]

    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-vf", vf,
        *codec_args,
        "-c:a", "copy",
        "-movflags", "+faststart",
        out_path,
    ]
    try:
        rc = subprocess.run(cmd, capture_output=True, timeout=1800).returncode
        return rc == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    except Exception as exc:
        log.warning("library: watermark render failed: %s", exc)
        return False


def _signed_download_url(r2, key: str, filename: str) -> str:
    """Mint a signed R2 URL that forces the browser into 'Save as…' via
    Content-Disposition: attachment. boto3's generate_presigned_url takes
    ResponseContentDisposition as a query param, baked into the
    signature so attackers can't strip it."""
    safe_name = (filename or "video.mp4").replace('"', "")
    client = r2._get_client()
    full_key = r2._k(key)
    return client.generate_presigned_url(
        ClientMethod="get_object",
        Params={
            "Bucket": r2.bucket,
            "Key":    full_key,
            "ResponseContentDisposition": f'attachment; filename="{safe_name}"',
        },
        ExpiresIn=_DOWNLOAD_TTL_SECONDS,
    )


# ─── Trending cache (module-level, low cost) ───────────────────────────
#
# Trending = items with the most usage + ratings. Cached for 5 min so the
# list endpoint doesn't recompute on every request. Refreshed lazily on
# first hit after expiry — no background thread.

_trending_ids: set[int]      = set()
_trending_last_refresh: float = 0.0
_TRENDING_TTL_SECONDS         = 300
_TRENDING_TOP_N               = 12


def _refresh_trending(db: Session) -> set[int]:
    global _trending_ids, _trending_last_refresh
    now = time.time()
    if (now - _trending_last_refresh) < _TRENDING_TTL_SECONDS and _trending_ids:
        return _trending_ids
    from sqlalchemy import or_
    rows = (
        db.query(models.LibraryItem.id)
          .filter(models.LibraryItem.deleted_at.is_(None))
          .filter(models.LibraryItem.video_key != "")
          .filter(or_(
              models.LibraryItem.use_count > 0,
              models.LibraryItem.rating_count > 0,
          ))
          .order_by(
              (models.LibraryItem.use_count + models.LibraryItem.rating_count * 2).desc(),
              models.LibraryItem.created_at.desc(),
          )
          .limit(_TRENDING_TOP_N)
          .all()
    )
    _trending_ids = {r.id for r in rows}
    _trending_last_refresh = now
    return _trending_ids


def _dimension_label(w: int, h: int) -> str:
    """Human label for the library card. Squareness tolerance ~10 %."""
    if not w or not h:
        return "Unknown"
    if abs(w - h) <= max(w, h) * 0.1:
        return "Square"
    return "Vertical" if h > w else "Horizontal"


def _avatar_url(u: Optional[models.User], r2) -> str:
    if not u or not u.avatar_key:
        return ""
    try:
        return r2.get_url(u.avatar_key, signed=False)
    except Exception:
        return ""


# ─── Serialization ──────────────────────────────────────────────────────

def _to_dict(
    item: models.LibraryItem,
    r2,
    *,
    users_by_id:      dict,
    categories_by_id: dict,
    my_ratings:       dict,
    trending_ids:     set,
) -> dict:
    """Render one library item to JSON. Batch lookups are passed in by
    the list endpoint so a 24-item page costs O(1) extra queries."""
    uploader = users_by_id.get(item.uploader_id) if item.uploader_id else None
    category = categories_by_id.get(item.category_id) if item.category_id else None

    thumb_url = ""
    try:
        if item.thumb_key:
            thumb_url = r2.get_url(item.thumb_key, signed=False)
    except Exception:
        thumb_url = ""

    rating_avg = (item.rating_sum / item.rating_count) if item.rating_count else 0.0
    creator_rating_avg = (
        (uploader.creator_rating_sum / uploader.creator_rating_count)
        if uploader and uploader.creator_rating_count else 0.0
    )

    return {
        "id":             item.id,
        "title":          item.title or item.original_name or f"Untitled #{item.id}",
        "description":    item.description or "",
        # video_url removed — use POST /api/library/{id}/play-ticket
        "thumb_url":      thumb_url,
        "play_ticket_endpoint": f"/api/library/{item.id}/play-ticket",
        "duration_secs":  float(item.duration_secs or 0),
        "file_size":      int(item.file_size or 0),
        "width":          int(item.width or 0),
        "height":         int(item.height or 0),
        "aspect":         "9:16" if (item.height or 0) > (item.width or 0) > 0 else "16:9",
        "dimension_label": _dimension_label(item.width or 0, item.height or 0),
        "original_name":  item.original_name or "",
        "uploader_id":    item.uploader_id,
        "uploader_name":  (uploader.name or uploader.email.split("@")[0]) if uploader else "",
        "uploader_avatar_url": _avatar_url(uploader, r2),
        "creator_rating_avg":   round(creator_rating_avg, 2),
        "creator_rating_count": int(uploader.creator_rating_count) if uploader else 0,
        "category":      ({"id": category.id, "name": category.name, "color": category.color or ""}
                          if category else None),
        "rating_avg":    round(rating_avg, 2),
        "rating_count":  int(item.rating_count or 0),
        "my_rating":     my_ratings.get(item.id),
        "trending":      item.id in trending_ids,
        "use_count":     int(item.use_count or 0),
        "watch_count":   int(item.watch_count or 0),
        "created_at":    item.created_at.isoformat() if item.created_at else None,
    }


def _serialize_page(items: list, db: Session, viewer: models.User) -> list:
    """Batch-fetch related rows then serialize a page of library items.
    Caller passes the already-filtered+sorted query result."""
    if not items:
        return []
    r2 = _r2()
    trending_ids = _refresh_trending(db)
    uploader_ids = {it.uploader_id for it in items if it.uploader_id}
    category_ids = {it.category_id for it in items if it.category_id}
    item_ids     = [it.id for it in items]

    users_by_id = {}
    if uploader_ids:
        users_by_id = {
            u.id: u for u in db.query(models.User).filter(
                models.User.id.in_(uploader_ids)
            ).all()
        }
    cats_by_id = {}
    if category_ids:
        cats_by_id = {
            c.id: c for c in db.query(models.LibraryCategory).filter(
                models.LibraryCategory.id.in_(category_ids)
            ).all()
        }
    my_ratings = {}
    if item_ids:
        for r in db.query(models.LibraryItemRating).filter(
            models.LibraryItemRating.user_id == viewer.id,
            models.LibraryItemRating.item_id.in_(item_ids),
        ).all():
            my_ratings[r.item_id] = r.stars

    return [_to_dict(it, r2,
                     users_by_id=users_by_id,
                     categories_by_id=cats_by_id,
                     my_ratings=my_ratings,
                     trending_ids=trending_ids)
            for it in items]


# ─── Categories ────────────────────────────────────────────────────────


class CategoryIn(BaseModel):
    name:  Optional[str] = None
    color: Optional[str] = None
    sort_order: Optional[int] = None


@router.get("/categories")
def list_categories(
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    cats = (
        db.query(models.LibraryCategory)
          .filter(models.LibraryCategory.deleted_at.is_(None))
          .order_by(models.LibraryCategory.sort_order.asc(),
                    models.LibraryCategory.name.asc())
          .all()
    )
    # Quick count per category (single GROUP BY) so the chip can show
    # "(N)" without each card making its own request.
    from sqlalchemy import func as _func
    counts = dict(
        db.query(models.LibraryItem.category_id,
                 _func.count(models.LibraryItem.id))
          .filter(models.LibraryItem.deleted_at.is_(None))
          .filter(models.LibraryItem.video_key != "")
          .group_by(models.LibraryItem.category_id)
          .all()
    )
    return [
        {
            "id":         c.id,
            "name":       c.name,
            "color":      c.color or "",
            "sort_order": c.sort_order,
            "item_count": int(counts.get(c.id, 0)),
        }
        for c in cats
    ]


@router.post("/categories")
def create_category(
    body: CategoryIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    if not getattr(user, "is_admin", False):
        raise HTTPException(403, "Admin only")
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(422, "Category name is required")
    cat = models.LibraryCategory(
        name       = name[:80],
        color      = (body.color or "").strip()[:20],
        sort_order = body.sort_order if body.sort_order is not None else 999,
    )
    db.add(cat); db.commit(); db.refresh(cat)
    audit.info("category create id=%d name=%r by=%d", cat.id, cat.name, user.id)
    return {"id": cat.id, "name": cat.name, "color": cat.color, "sort_order": cat.sort_order}


@router.patch("/categories/{cat_id}")
def update_category(
    cat_id: int,
    body: CategoryIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    if not getattr(user, "is_admin", False):
        raise HTTPException(403, "Admin only")
    cat = db.query(models.LibraryCategory).get(cat_id)
    if not cat or cat.deleted_at is not None:
        raise HTTPException(404, "Category not found")
    if body.name is not None:
        new_name = body.name.strip()
        if not new_name:
            raise HTTPException(422, "Category name cannot be empty")
        # The video binding is by id, not name — rename is purely cosmetic.
        cat.name = new_name[:80]
    if body.color is not None:
        cat.color = (body.color or "").strip()[:20]
    if body.sort_order is not None:
        cat.sort_order = int(body.sort_order)
    db.commit()
    audit.info("category update id=%d by=%d", cat_id, user.id)
    return {"id": cat.id, "name": cat.name, "color": cat.color, "sort_order": cat.sort_order}


@router.delete("/categories/{cat_id}")
def delete_category(
    cat_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Soft-delete a category. Videos that used it are NOT removed —
    their ``category_id`` is just NULL'd so they show as uncategorised
    in the UI. This is what prevents the "delete the category, the
    videos disappear with it" footgun the operator called out."""
    if not getattr(user, "is_admin", False):
        raise HTTPException(403, "Admin only")
    cat = db.query(models.LibraryCategory).get(cat_id)
    if not cat or cat.deleted_at is not None:
        raise HTTPException(404, "Category not found")
    cat.deleted_at = datetime.now(timezone.utc)
    # NULL out FKs on related items — preserves the videos, removes the
    # broken binding. ON DELETE SET NULL would only fire on a hard
    # delete, so we do it manually here for soft-delete parity.
    from sqlalchemy import text as _text
    db.execute(_text("UPDATE library_items SET category_id = NULL WHERE category_id = :c"),
               {"c": cat_id})
    db.commit()
    audit.info("category soft-delete id=%d by=%d", cat_id, user.id)
    return {"ok": True, "id": cat_id}


# ─── Listing + creator profile ─────────────────────────────────────────


@router.get("/")
def list_library(
    q: str = "",
    category_id: Optional[int] = None,
    creator_id:  Optional[int] = None,
    sort: str = "newest",
    page: int = 1,
    page_size: int = 24,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Paginated + filtered + sorted library listing.

    Query params:
      • q              — search across title, description, creator name
      • category_id    — filter by category (None = any)
      • creator_id     — filter by uploader (None = any)
      • sort           — newest | trending | most_used | top_rated
      • page, page_size — 1-based pagination (page_size capped at 60)
    """
    from sqlalchemy import or_, func as _func
    page      = max(1, int(page))
    page_size = max(1, min(60, int(page_size)))

    qry = db.query(models.LibraryItem).filter(
        models.LibraryItem.deleted_at.is_(None),
        models.LibraryItem.video_key != "",
    )
    if category_id is not None:
        qry = qry.filter(models.LibraryItem.category_id == category_id)
    if creator_id is not None:
        qry = qry.filter(models.LibraryItem.uploader_id == creator_id)
    if (q or "").strip():
        like = f"%{q.strip().lower()}%"
        qry = qry.outerjoin(
            models.User, models.LibraryItem.uploader_id == models.User.id
        ).filter(or_(
            _func.lower(models.LibraryItem.title).like(like),
            _func.lower(models.LibraryItem.description).like(like),
            _func.lower(models.User.name).like(like),
            _func.lower(models.User.email).like(like),
        ))

    total = qry.count()

    if sort == "trending":
        qry = qry.order_by(
            (models.LibraryItem.use_count + models.LibraryItem.rating_count * 2).desc(),
            models.LibraryItem.created_at.desc(),
        )
    elif sort == "most_used":
        qry = qry.order_by(models.LibraryItem.use_count.desc(),
                           models.LibraryItem.created_at.desc())
    elif sort == "top_rated":
        qry = qry.order_by(models.LibraryItem.rating_sum.desc(),
                           models.LibraryItem.created_at.desc())
    else:  # newest (default)
        qry = qry.order_by(models.LibraryItem.created_at.desc())

    items = qry.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "items":       _serialize_page(items, db, user),
        "total":       int(total),
        "page":        page,
        "page_size":   page_size,
        "total_pages": (int(total) + page_size - 1) // page_size,
        "sort":        sort,
    }


@router.get("/creator/{creator_id}")
def get_creator_profile(
    creator_id: int,
    page: int = 1,
    page_size: int = 24,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Public creator profile — avatar, aggregate rating, paginated
    list of uploads. Click "rate this creator" on the frontend to fire
    POST /api/profile/creators/{id}/rate."""
    creator = db.query(models.User).get(creator_id)
    if not creator or not creator.is_active:
        raise HTTPException(404, "Creator not found")

    page      = max(1, int(page))
    page_size = max(1, min(60, int(page_size)))

    qry = db.query(models.LibraryItem).filter(
        models.LibraryItem.deleted_at.is_(None),
        models.LibraryItem.video_key != "",
        models.LibraryItem.uploader_id == creator_id,
    ).order_by(models.LibraryItem.created_at.desc())
    total = qry.count()
    items = qry.offset((page - 1) * page_size).limit(page_size).all()

    r2 = _r2()
    my_creator_rating = db.query(models.CreatorRating).filter_by(
        creator_id=creator_id, rater_id=user.id,
    ).first()

    creator_dict = {
        "id":           creator.id,
        "name":         creator.name or creator.email.split("@")[0],
        "avatar_url":   _avatar_url(creator, r2),
        "is_creative":  bool(creator.is_creative),
        "is_admin":     bool(creator.is_admin),
        "rating_avg":   round(
            creator.creator_rating_sum / creator.creator_rating_count, 2
        ) if creator.creator_rating_count else 0.0,
        "rating_count": int(creator.creator_rating_count or 0),
        "upload_count": int(total),
        "joined_at":    creator.created_at.isoformat() if creator.created_at else None,
        "my_rating":    my_creator_rating.stars if my_creator_rating else None,
    }

    return {
        "creator":     creator_dict,
        "items":       _serialize_page(items, db, user),
        "total":       int(total),
        "page":        page,
        "page_size":   page_size,
        "total_pages": (int(total) + page_size - 1) // page_size,
    }


@router.get("/{item_id}")
def get_library_item(
    item_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    item = db.query(models.LibraryItem).get(item_id)
    if not item or item.deleted_at is not None:
        raise HTTPException(404, "Library item not found")
    rows = _serialize_page([item], db, user)
    return rows[0] if rows else {}


# ─── Rate a library item ───────────────────────────────────────────────


class RateIn(BaseModel):
    stars: int


@router.post("/{item_id}/rate")
def rate_library_item(
    item_id: int,
    body: RateIn,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Upsert the caller's rating for this video. Cached
    rating_sum/rating_count on the item row are adjusted by the delta so
    the list endpoint never has to GROUP BY at query time."""
    stars = int(body.stars or 0)
    if not 1 <= stars <= 5:
        raise HTTPException(422, "stars must be in 1..5")
    item = db.query(models.LibraryItem).get(item_id)
    if not item or item.deleted_at is not None:
        raise HTTPException(404, "Library item not found")

    existing = db.query(models.LibraryItemRating).filter_by(
        item_id=item_id, user_id=user.id,
    ).first()
    if existing:
        delta = stars - int(existing.stars or 0)
        existing.stars = stars
        item.rating_sum = int(item.rating_sum or 0) + delta
    else:
        db.add(models.LibraryItemRating(
            item_id=item_id, user_id=user.id, stars=stars,
        ))
        item.rating_sum   = int(item.rating_sum or 0) + stars
        item.rating_count = int(item.rating_count or 0) + 1

    db.commit(); db.refresh(item)
    avg = (item.rating_sum / item.rating_count) if item.rating_count else 0.0
    return {
        "ok":           True,
        "stars":        stars,
        "rating_avg":   round(avg, 2),
        "rating_count": int(item.rating_count or 0),
    }


@router.post("/upload-session")
def create_upload_session(
    request: Request,
    user: models.User = Depends(auth.current_user),
):
    """Issue a one-time HMAC-signed upload token. The actual upload
    endpoint (``POST /upload/{token}``) refuses to accept a body without
    a fresh, unconsumed token bound to the caller.

    Token TTL: 5 minutes. Single-use. Inspecting the request URL in
    devtools gives an attacker nothing reusable.
    """
    _check_origin(request)
    _require_creative(user)
    _rate_limit_check(user.id)
    token, expires_at = _issue_upload_token(user.id)
    audit.info("upload-session issued user=%d ip=%s", user.id,
               request.client.host if request.client else "?")
    return {
        "token":      token,
        "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        "endpoint":   f"/api/library/upload/{token}",
        "ttl_seconds": _TOKEN_TTL_SECONDS,
    }


@router.post("/upload/{token}")
async def upload_library_item(
    token: str,
    request: Request,
    video: UploadFile = File(...),
    title: str        = Form(""),
    description: str  = Form(""),
    category_id: Optional[int] = Form(None),
    db: Session       = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Upload a video to the shared library.

    Hardening order (cheapest checks first so abusers cost us the least):
      1. Origin allow-list — instant reject on bad header
      2. Role gate (creative or admin)
      3. Token single-use validation (consumes the nonce)
      4. Rate limit (sliding window per user)
      5. Filename + extension sanitisation
      6. Stream body to disk, hard 4 GiB cap mid-stream
      7. Magic-byte sniff — file MUST start with a real video container
      8. ffprobe + thumbnail
      9. R2 upload (uploader-scoped key)
      10. DB row commit
    Every reject is logged to the audit channel with user_id + IP.
    """
    client_ip = request.client.host if request.client else "?"

    _check_origin(request)
    _require_creative(user)
    # Consume token AFTER origin + role checks so brute-forcing tokens
    # against an anonymous endpoint doesn't even reach the token store.
    _consume_upload_token(token, user.id)
    _rate_limit_check(user.id)

    # Sanitise filename + extension. Reject before touching disk.
    raw_name = video.filename or "upload.mp4"
    fname    = _safe_filename(raw_name)
    ext      = Path(fname).suffix.lower() or ".mp4"
    if ext not in ALLOWED_EXTS:
        audit.warning("ext reject user=%d ip=%s ext=%r", user.id, client_ip, ext)
        raise HTTPException(
            415,
            f"Unsupported video extension {ext!r}. "
            f"Allowed: {sorted(ALLOWED_EXTS)}",
        )
    if video.content_type and video.content_type.split(";")[0].strip() not in ALLOWED_VIDEO_MIMES:
        log.warning("library: unusual content-type %r for %r — accepting on extension",
                    video.content_type, fname)

    r2 = _r2()

    # 1) Create the DB row FIRST to obtain the id used in object keys.
    #    video_key stays empty until the upload completes so a failed
    #    upload doesn't leave a "ghost" row pointing at a missing object.
    # Validate the optional category_id — fall back to NULL if it doesn't
    # exist or has been soft-deleted, rather than 422-ing the upload.
    resolved_category_id: Optional[int] = None
    if category_id:
        cat = db.query(models.LibraryCategory).get(int(category_id))
        if cat and cat.deleted_at is None:
            resolved_category_id = cat.id

    item = models.LibraryItem(
        uploader_id   = user.id,
        title         = (title or "").strip()[:200],
        description   = (description or "").strip()[:2000],
        original_name = fname[:255],
        video_key     = "",
        thumb_key     = "",
        category_id   = resolved_category_id,
    )
    db.add(item); db.commit(); db.refresh(item)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"library_{item.id}_"))
    local_video = tmp_dir / f"video{ext}"
    local_thumb = tmp_dir / "thumb.jpg"
    try:
        # 2) Stream upload body to local tmp with mid-stream cap.
        bytes_written = 0
        with local_video.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_BYTES:
                    raise HTTPException(
                        413,
                        f"Video exceeds {MAX_BYTES // (1024*1024)} MiB limit",
                    )
                out.write(chunk)
        if bytes_written == 0:
            raise HTTPException(400, "Uploaded file is empty")

        # 3) Magic-byte sniff. The extension can lie — the bytes can't
        # easily. Rejects "evil.exe → rename to evil.mp4" attempts.
        if not _looks_like_video(str(local_video)):
            audit.warning(
                "magic reject user=%d ip=%s name=%r size=%d",
                user.id, client_ip, fname, bytes_written,
            )
            raise HTTPException(
                415,
                "Uploaded file does not look like a video container. "
                "Allowed: MP4/MOV/MKV/WebM/AVI.",
            )

        # 4) Probe + thumbnail (best-effort).
        info = _ffprobe(str(local_video))
        thumb_ok = _extract_thumb(str(local_video), str(local_thumb),
                                  at_seconds=1.0)

        # 5) Push video to R2. Key includes the uploader id so the R2
        # dashboard groups uploads by creative — easy audit trail.
        video_key = _video_key(item.id, ext, user.id)
        r2.upload(str(local_video), video_key,
                  content_type=video.content_type or "video/mp4")

        # 6) Push thumbnail to R2 if we got one.
        thumb_key = ""
        if thumb_ok:
            t_key = _thumb_key(item.id, user.id)
            r2.upload(str(local_thumb), t_key, content_type="image/jpeg")
            thumb_key = t_key

        # 7) Watermarked copy for free-tier downloads. We already have the
        # source on local disk so no re-download from R2 is needed. The
        # render is best-effort — if it fails the upload still succeeds
        # and the row keeps watermark_key='', which the download endpoint
        # treats as "render lazily on first free-tier hit" (fallback).
        watermark_key = ""
        local_wm = tmp_dir / f"video_watermarked{ext}"
        try:
            log.info("library: rendering watermark for item=%d (size=%d)",
                     item.id, bytes_written)
            if _render_watermark(str(local_video), str(local_wm)):
                wm_key = _watermark_key_for(item.id, ext, user.id)
                r2.upload(str(local_wm), wm_key, content_type="video/mp4")
                watermark_key = wm_key
                audit.info("watermark pre-render OK item=%d", item.id)
            else:
                log.warning("library: watermark render returned non-zero for item=%d", item.id)
        except Exception as _wm_exc:
            log.warning("library: watermark render failed for item=%d: %s",
                        item.id, _wm_exc)

        # 8) Persist keys + metadata on the row.
        item.video_key     = video_key
        item.thumb_key     = thumb_key
        item.watermark_key = watermark_key
        item.duration_secs = float(info.get("duration") or 0)
        item.width         = int(info.get("width") or 0)
        item.height        = int(info.get("height") or 0)
        item.file_size     = bytes_written
        db.commit(); db.refresh(item)

        audit.info(
            "upload OK user=%d ip=%s item=%d size=%d dur=%.1f %sx%s",
            user.id, client_ip, item.id, bytes_written,
            item.duration_secs, item.width, item.height,
        )
        log.info("library: uploaded item #%s (%d bytes, %.1fs, %sx%s)",
                 item.id, bytes_written, item.duration_secs,
                 item.width, item.height)

        # Use the same batch-fetch serializer the list endpoint uses so the
        # response shape is identical (category, dimension_label, etc.).
        rows = _serialize_page([item], db, user)
        return rows[0] if rows else {}

    except HTTPException:
        # Roll back the placeholder row on rejection.
        db.delete(item); db.commit()
        raise
    except Exception as exc:
        log.exception("library: upload failed for item #%s", item.id)
        audit.error("upload FAIL user=%d ip=%s item=%d err=%s",
                    user.id, client_ip, item.id, exc)
        db.delete(item); db.commit()
        raise HTTPException(500, f"Upload failed: {exc}")
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ─── Large-file path: presigned DIRECT-to-R2 upload (up to 2 GB) ───────────
# The streamed upload above runs the bytes THROUGH the app server, so a proxy
# (Cloudflare ~100 MB) 413s big files. This path hands the client a presigned
# R2 PUT URL — the bytes go straight to storage, bypassing the proxy — then a
# finalize call probes the object (rejecting a non-video), makes the thumbnail
# off a signed URL (ffmpeg reads it over HTTP range — no full download), and
# commits the row. Existing small-file uploads keep the hardened streamed path.
MAX_PRESIGN_BYTES = 2 * 1024 * 1024 * 1024   # 2 GiB


class PresignUploadIn(BaseModel):
    filename: str
    content_type: str = "video/mp4"
    size: int = 0
    title: str = ""
    description: str = ""
    category_id: Optional[int] = None


def _cleanup_presigned(r2, key: str, item, db) -> None:
    try:
        r2.delete(key)
    except Exception:
        pass
    try:
        db.delete(item); db.commit()
    except Exception:
        db.rollback()


@router.post("/presign-upload")
def presign_upload(body: PresignUploadIn, request: Request,
                   db: Session = Depends(get_db),
                   user: models.User = Depends(auth.current_user)):
    """Issue a presigned R2 PUT for a direct (up to 2 GB) upload. Creates a
    PENDING LibraryItem (video_key empty); the client PUTs the bytes to the URL,
    then calls /finalize-upload with the returned item_id."""
    _check_origin(request)
    _require_creative(user)
    _rate_limit_check(user.id)
    fname = _safe_filename(body.filename or "upload.mp4")
    ext = Path(fname).suffix.lower() or ".mp4"
    if ext not in ALLOWED_EXTS:
        raise HTTPException(415, f"Unsupported video extension {ext!r}. Allowed: {sorted(ALLOWED_EXTS)}")
    if body.size and body.size > MAX_PRESIGN_BYTES:
        raise HTTPException(413, f"Video exceeds {MAX_PRESIGN_BYTES // (1024*1024)} MiB limit")
    r2 = _r2()
    resolved_category_id: Optional[int] = None
    if body.category_id:
        cat = db.query(models.LibraryCategory).get(int(body.category_id))
        if cat and cat.deleted_at is None:
            resolved_category_id = cat.id
    item = models.LibraryItem(
        uploader_id   = user.id,
        title         = (body.title or "").strip()[:200],
        description   = (body.description or "").strip()[:2000],
        original_name = fname[:255],
        video_key     = "",
        thumb_key     = "",
        category_id   = resolved_category_id,
    )
    db.add(item); db.commit(); db.refresh(item)
    key = _video_key(item.id, ext, user.id)
    try:
        put_url = r2.presign_put(key, expires_s=3600)
    except Exception as exc:
        db.delete(item); db.commit()
        log.error("library: presign failed: %s", exc)
        raise HTTPException(503, f"Could not start the upload: {exc}")
    audit.info("presign issued user=%d item=%d ext=%s", user.id, item.id, ext)
    return {"item_id": item.id, "put_url": put_url, "key": key, "ext": ext, "expires_in": 3600}


class FinalizeUploadIn(BaseModel):
    item_id: int


@router.post("/finalize-upload")
def finalize_upload(body: FinalizeUploadIn, request: Request,
                    db: Session = Depends(get_db),
                    user: models.User = Depends(auth.current_user)):
    """Complete a presigned upload: verify the object exists + is within the cap,
    probe it (a non-video is rejected + swept), make a thumbnail off a signed URL,
    and commit the row. Idempotent once finalized."""
    _check_origin(request)
    _require_creative(user)
    item = db.query(models.LibraryItem).get(int(body.item_id))
    if not item or item.uploader_id != user.id:
        raise HTTPException(404, "not found")
    if item.video_key:  # already finalized — return idempotently
        rows = _serialize_page([item], db, user)
        return rows[0] if rows else {}
    ext = Path(item.original_name or "video.mp4").suffix.lower() or ".mp4"
    key = _video_key(item.id, ext, user.id)
    r2 = _r2()
    size = r2.head_size(key)
    if size < 0:
        raise HTTPException(400, "Upload not found in storage — the transfer may have failed. Try again.")
    if size == 0:
        _cleanup_presigned(r2, key, item, db)
        raise HTTPException(400, "Uploaded file is empty.")
    if size > MAX_PRESIGN_BYTES:
        _cleanup_presigned(r2, key, item, db)
        raise HTTPException(413, f"Video exceeds {MAX_PRESIGN_BYTES // (1024*1024)} MiB limit")
    # Probe + thumbnail straight off a short-lived signed URL (ffmpeg reads it
    # over HTTP range requests — the whole 2 GB is never downloaded here).
    get_url = r2.get_url(key, signed=True, expires_s=1800)
    info = _ffprobe(get_url)
    if not info or not (int(info.get("width") or 0) or float(info.get("duration") or 0)):
        _cleanup_presigned(r2, key, item, db)
        raise HTTPException(415, "That file doesn't look like a video. Allowed: MP4/MOV/MKV/WebM/AVI.")
    thumb_key = ""
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"library_fin_{item.id}_"))
    try:
        local_thumb = tmp_dir / "thumb.jpg"
        if _extract_thumb(get_url, str(local_thumb), at_seconds=1.0):
            t_key = _thumb_key(item.id, user.id)
            r2.upload(str(local_thumb), t_key, content_type="image/jpeg")
            thumb_key = t_key
    except Exception as exc:
        log.warning("library: finalize thumbnail failed item=%d: %s", item.id, exc)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    item.video_key     = key
    item.thumb_key     = thumb_key
    item.duration_secs = float(info.get("duration") or 0)
    item.width         = int(info.get("width") or 0)
    item.height        = int(info.get("height") or 0)
    item.file_size     = int(size)
    db.commit(); db.refresh(item)
    audit.info("finalize OK user=%d item=%d size=%d %sx%s",
               user.id, item.id, size, item.width, item.height)
    rows = _serialize_page([item], db, user)
    return rows[0] if rows else {}


# Per-user playback rate limit (independent from upload rate limit so
# heavy library browsing doesn't tax the upload bucket and vice versa).
# Sliding window: 60 ticket issuances / 60 s — enough headroom for a
# user to scrub a video aggressively without falling into 429.
_recent_plays: dict[int, deque] = {}
_recent_plays_lock = threading.Lock()
PLAY_RATE_WINDOW_S = 60
PLAY_RATE_MAX      = 60


def _play_rate_limit(user_id: int) -> None:
    now = time.time()
    cutoff = now - PLAY_RATE_WINDOW_S
    with _recent_plays_lock:
        dq = _recent_plays.setdefault(user_id, deque())
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= PLAY_RATE_MAX:
            audit.warning("play rate hit user=%d", user_id)
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Playback rate limit exceeded. Slow down and try again.",
            )
        dq.append(now)


@router.post("/{item_id}/play-ticket")
def issue_play_ticket(
    item_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Return a short-lived (60 s) signed R2 URL the client can hand
    straight to a <video> tag.

    This endpoint is the entire entitlement surface for playback —
    every play attempt routes through it. The future paywall check
    drops in here (return 402 on insufficient subscription, etc.)
    and the backend's role for the next 60 s is over: R2 serves the
    bytes via Cloudflare's edge with zero backend bandwidth.
    """
    _check_origin(request)
    _play_rate_limit(user.id)

    item = db.query(models.LibraryItem).get(item_id)
    if not item or item.deleted_at is not None or not item.video_key:
        raise HTTPException(404, "Library item not found")

    # ────────────────────────────────────────────────────────────
    # FUTURE: payment / entitlement check goes here.
    # if not user.has_active_subscription:
    #     raise HTTPException(402, "Subscription required to play library videos")
    # ────────────────────────────────────────────────────────────

    # Free users get the watermarked copy for streaming too — so even a
    # screen recording of the in-app player ends up with the
    # ``kaizerx.com`` watermark burned into the captured frames. Paid
    # users stream the original. Falls back to the original when the
    # watermark hasn't been rendered yet (legacy items or first-upload
    # in-flight) — better to serve the video than to 404.
    paid = _is_paid(user)
    use_key = item.video_key
    served_watermarked = False
    if not paid and item.watermark_key:
        use_key = item.watermark_key
        served_watermarked = True

    r2 = _r2()
    try:
        play_url = r2.get_url(
            use_key,
            signed=True,
            expires_s=_PLAY_TTL_SECONDS,
        )
    except Exception as exc:
        log.exception("library: signed-URL mint failed for item #%s", item_id)
        raise HTTPException(503, f"Could not issue playback URL: {exc}")

    # Bump watch count. Best-effort — never fail the play because the
    # counter update hit a transient DB blip.
    try:
        item.watch_count = int(item.watch_count or 0) + 1
        db.commit()
    except Exception:
        db.rollback()

    expires_at = int(time.time() + _PLAY_TTL_SECONDS)
    audit.info(
        "play user=%d ip=%s item=%d watermarked=%s",
        user.id, request.client.host if request.client else "?",
        item_id, served_watermarked,
    )
    return {
        "play_url":    play_url,
        "watermarked": served_watermarked,
        "expires_at":  datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        "ttl_seconds": _PLAY_TTL_SECONDS,
    }


@router.post("/{item_id}/download-ticket")
def issue_download_ticket(
    item_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Return a short-lived signed R2 URL that forces 'Save as…'.

    Tier behaviour:
      • Paid users (plan != "free") → original video, no watermark.
      • Free users → watermarked copy. Rendered lazily on first
        download, cached on R2, served from there on every subsequent
        download. So a video the team's paid users grab pays zero
        watermark-render cost ever.

    The download is gated through the same per-user playback rate
    limit as ``play-ticket`` so a script can't spam this endpoint
    either.
    """
    _check_origin(request)
    _play_rate_limit(user.id)

    item = db.query(models.LibraryItem).get(item_id)
    if not item or item.deleted_at is not None or not item.video_key:
        raise HTTPException(404, "Library item not found")

    r2 = _r2()
    paid = _is_paid(user)
    src_filename = item.original_name or f"library_{item.id}.mp4"
    # Drop the leading path components defensively.
    safe_name = os.path.basename(src_filename).replace("/", "_")

    if paid:
        try:
            url = _signed_download_url(r2, item.video_key, safe_name)
        except Exception as exc:
            log.exception("library: paid download URL mint failed (%d)", item_id)
            raise HTTPException(503, f"Could not issue download URL: {exc}")
        audit.info("download user=%d plan=%s item=%d original",
                   user.id, user.plan, item_id)
        return {
            "download_url": url,
            "watermarked":  False,
            "expires_at":   datetime.fromtimestamp(
                                int(time.time() + _DOWNLOAD_TTL_SECONDS),
                                tz=timezone.utc).isoformat(),
            "ttl_seconds":  _DOWNLOAD_TTL_SECONDS,
        }

    # ── Free tier: serve watermarked. Render + cache if not yet built.
    if not item.watermark_key:
        ext = Path(item.video_key).suffix.lower() or ".mp4"
        wm_key = _watermark_key_for(item.id, ext, item.uploader_id or user.id)

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"wm_{item.id}_"))
        src_path = tmp_dir / f"src{ext}"
        wm_path  = tmp_dir / f"wm{ext}"
        try:
            log.info("library: watermarking item=%d for user=%d", item.id, user.id)
            r2.download(item.video_key, str(src_path))
            ok = _render_watermark(str(src_path), str(wm_path))
            if not ok:
                raise HTTPException(503, "Watermark render failed; please try again")
            r2.upload(str(wm_path), wm_key, content_type="video/mp4")
            item.watermark_key = wm_key
            db.commit(); db.refresh(item)
            audit.info("watermark rendered user=%d item=%d key=%r",
                       user.id, item.id, wm_key)
        finally:
            try: shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception: pass

    # Always mint a fresh signed URL — last cached value would expire.
    try:
        url = _signed_download_url(r2, item.watermark_key, safe_name)
    except Exception as exc:
        log.exception("library: free download URL mint failed (%d)", item_id)
        raise HTTPException(503, f"Could not issue download URL: {exc}")

    audit.info("download user=%d plan=%s item=%d watermarked",
               user.id, user.plan, item_id)
    return {
        "download_url": url,
        "watermarked":  True,
        "expires_at":   datetime.fromtimestamp(
                            int(time.time() + _DOWNLOAD_TTL_SECONDS),
                            tz=timezone.utc).isoformat(),
        "ttl_seconds":  _DOWNLOAD_TTL_SECONDS,
    }


@router.delete("/{item_id}")
def delete_library_item(
    item_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.current_user),
):
    """Soft-delete a library item. Uploader or admin only."""
    item = db.query(models.LibraryItem).get(item_id)
    if not item or item.deleted_at is not None:
        raise HTTPException(404, "Library item not found")
    if item.uploader_id != user.id and not getattr(user, "is_admin", False):
        raise HTTPException(403, "Only the uploader or an admin can delete this item")
    item.deleted_at = datetime.now(timezone.utc)
    db.commit()
    return {"ok": True, "id": item_id}


# ─── Internal helper for create_job ────────────────────────────────────
# Imported by main.py's create_job to materialize a library video at a
# given local path. Kept in this module so the library is the only place
# that knows about its R2 keys.

def fetch_library_video_to(item_id: int, dest_dir: Path, db: Session) -> tuple[Path, models.LibraryItem]:
    """Download a library video from R2 to *dest_dir*. Returns
    (local_path, item). Raises HTTPException(404) when missing."""
    item = db.query(models.LibraryItem).get(item_id)
    if not item or item.deleted_at is not None or not item.video_key:
        raise HTTPException(404, f"Library item #{item_id} not available")
    r2 = _r2()
    ext = Path(item.original_name or "video.mp4").suffix.lower() or ".mp4"
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / f"library_{item.id}{ext}"
    r2.download(item.video_key, str(local))
    # Bump usage counter — non-blocking on failure.
    try:
        item.use_count = (item.use_count or 0) + 1
        db.commit()
    except Exception:
        db.rollback()
    return local, item
