"""Thin HTTP wrapper around the HeyGen v2 video-generation API.

We talk directly to ``api.heygen.com`` over httpx instead of pulling
in a HeyGen SDK — the surface we use is four endpoints, all REST,
and a hand-written wrapper is easier to audit + retry than a
1000-line generated client.

Endpoints used:
  POST /v2/video/generate         submit avatar+script -> {video_id}
  GET  /v1/video_status.get       poll progress       -> {status, video_url, ...}
  GET  /v2/avatars                list available avatars
  GET  /v2/voices                 list available voices

Auth header: ``x-api-key: <key>`` (no Bearer prefix — HeyGen-specific).

Same retry pattern as ``express.whisper.transcribe`` — 3 attempts
with exponential backoff on 429 / 5xx, fail fast on 4xx-other.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import httpx


_BASE = "https://api.heygen.com"
_TIMEOUT_S = 60


class HeyGenError(RuntimeError):
    """HeyGen API call failed or returned an unparseable body."""


class HeyGenAuthError(HeyGenError):
    """401/403 — wrong key or quota exhausted."""


# ── Internal request helper ────────────────────────────────────────

def _api_key(explicit: Optional[str]) -> str:
    """Per-request key wins, env fallback otherwise. Strip whitespace
    so a copy-paste with trailing newline doesn't 401."""
    key = (explicit or os.environ.get("HEYGEN_API_KEY") or "").strip()
    if not key:
        raise HeyGenAuthError("HEYGEN_API_KEY missing")
    return key


def _request(
    method: str,
    path: str,
    *,
    api_key: Optional[str],
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
    timeout_s: int = _TIMEOUT_S,
) -> dict:
    """One HTTP call with 3-attempt retry on transient failures."""
    key = _api_key(api_key)
    url = f"{_BASE}{path}"
    headers = {"x-api-key": key, "Accept": "application/json"}
    if json_body is not None:
        headers["Content-Type"] = "application/json"

    last_err = ""
    for attempt, wait_s in enumerate((0, 2, 5)):
        if wait_s:
            time.sleep(wait_s)
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=15)) as cli:
                r = cli.request(
                    method, url, headers=headers,
                    json=json_body, params=params,
                )
        except httpx.HTTPError as exc:
            last_err = f"network: {exc}"
            print(f"[heygen/client] attempt {attempt + 1}/3 network: {exc}")
            continue

        if r.status_code == 401 or r.status_code == 403:
            raise HeyGenAuthError(
                f"HeyGen rejected the API key (HTTP {r.status_code}): {r.text[:300]}"
            )
        if r.status_code < 400:
            try:
                return r.json()
            except ValueError as exc:
                raise HeyGenError(f"non-JSON response: {r.text[:300]}") from exc

        last_err = f"HTTP {r.status_code}: {r.text[:400]}"
        if r.status_code not in (408, 429, 500, 502, 503, 504):
            raise HeyGenError(f"HeyGen {method} {path} {last_err}")
        print(f"[heygen/client] attempt {attempt + 1}/3 got {r.status_code}; will retry")

    raise HeyGenError(f"HeyGen {method} {path} {last_err} (after 3 attempts)")


# ── Public endpoints ───────────────────────────────────────────────

def list_avatars(*, api_key: Optional[str] = None) -> dict:
    """Return ``{avatars: [...], talking_photos: [...]}`` from
    HeyGen's avatar library. Each avatar entry has:
      - avatar_id           (use this in generate_video)
      - avatar_name
      - gender
      - preview_image_url   (display in picker)
      - preview_video_url
      - premium             (bool — Creator+ tier required)
      - default_voice_id    (sensible voice paired with the avatar)
    """
    data = _request("GET", "/v2/avatars", api_key=api_key)
    payload = data.get("data") or {}
    if isinstance(payload, dict):
        return {
            "avatars":        payload.get("avatars") or [],
            "talking_photos": payload.get("talking_photos") or [],
        }
    raise HeyGenError(f"unexpected avatars payload: {str(data)[:300]}")


def list_voices(*, api_key: Optional[str] = None, limit: int = 500) -> list[dict]:
    """Return the available voices. Each voice has:
      - voice_id
      - language          (e.g. "Telugu", "Hindi", "English")
      - gender
      - name
      - preview_audio
      - support_pause     (bool — pauses in the script)
      - emotion_support   (bool)
    """
    data = _request("GET", "/v2/voices", api_key=api_key)
    payload = data.get("data") or {}
    voices = payload.get("voices") if isinstance(payload, dict) else None
    if voices is None and isinstance(data.get("data"), list):
        voices = data["data"]
    if voices is None:
        raise HeyGenError(f"unexpected voices payload: {str(data)[:300]}")
    return list(voices)[:limit]


def _verify_avatar_v3(avatar_id: str, *, api_key: Optional[str] = None) -> None:
    """Probe the v2/avatars list to confirm ``avatar_id`` is a public
    Avatar 3.0 (preview_video_url contains '/avatar/v3/'). Raises
    HeyGenError otherwise — keeps the pipeline from silently routing
    a talking_photo_id through the Avatar 3 path.

    Cached for the process lifetime so we only pay one /v2/avatars
    call across all generations.
    """
    cache = _verify_avatar_v3.__dict__.setdefault("_cache", {})
    if avatar_id in cache:
        if cache[avatar_id] is True:
            return
        raise HeyGenError(cache[avatar_id])

    try:
        data = _request("GET", "/v2/avatars", api_key=api_key)
        avs = ((data.get("data") or {}).get("avatars")) or []
        tps = ((data.get("data") or {}).get("talking_photos")) or []
    except HeyGenError as exc:
        # If the probe itself fails, don't block the generation —
        # let HeyGen reject it with its own error.
        print(f"[heygen/client] avatar verify probe failed: {exc}")
        return

    hit = next((a for a in avs if a.get("avatar_id") == avatar_id), None)
    if hit:
        is_v3 = "/avatar/v3/" in (hit.get("preview_video_url") or "")
        if is_v3:
            cache[avatar_id] = True
            return
        msg = (f"avatar_id={avatar_id} is in the public list but is NOT "
               f"Avatar 3.0 (engine looks like V2). Pick a different id "
               f"that has /avatar/v3/ in its preview URL.")
        cache[avatar_id] = msg
        raise HeyGenError(msg)

    # Maybe a talking_photo? Different error message so the user knows
    # which engine the id belongs to.
    tp = next((t for t in tps if t.get("talking_photo_id") == avatar_id), None)
    if tp:
        msg = (f"avatar_id={avatar_id} is a TALKING_PHOTO ({tp.get('talking_photo_name')}), "
               f"not an Avatar 3.0 public avatar. The 'Avatar 3 only' "
               f"policy rejects talking_photo ids. Pick an avatar from "
               f"/v2/avatars whose preview_video_url contains '/avatar/v3/'.")
        cache[avatar_id] = msg
        raise HeyGenError(msg)

    msg = (f"avatar_id={avatar_id} not found in /v2/avatars at all "
           f"(checked {len(avs)} public avatars + {len(tps)} talking_photos). "
           f"Likely a stale or revoked id.")
    cache[avatar_id] = msg
    raise HeyGenError(msg)


def generate_video(
    *,
    avatar_id: str,
    voice_id: str,
    script: str,
    api_key: Optional[str] = None,
    width: int = 1080,
    height: int = 1920,
    background_color: str = "#000000",
) -> str:
    """Submit an avatar generation. Returns the ``video_id`` to poll.

    Body shape per HeyGen v2 docs:

        {
          "video_inputs": [{
            "character": {"type": "avatar", "avatar_id": "..."},
            "voice":     {"type": "text", "voice_id": "...", "input_text": "..."},
            "background":{"type": "color", "value": "#000000"}
          }],
          "dimension": {"width": 1080, "height": 1920}
        }
    """
    if not avatar_id:
        raise HeyGenError("avatar_id required")
    if not voice_id:
        raise HeyGenError("voice_id required")
    # HARD GUARD: only Avatar 3.0 public avatars allowed. Rejects
    # talking_photo ids + stale ids before they hit HeyGen with a
    # confusing 400. Cached so this is one /v2/avatars round-trip
    # per process, not per generation.
    _verify_avatar_v3(avatar_id, api_key=api_key)
    script = (script or "").strip()
    if not script:
        raise HeyGenError("script is empty")
    if len(script) > 1500:
        # HeyGen's hard limit per their docs (errors at 1500+ chars).
        # We cap at 1450 to leave headroom for punctuation HeyGen may
        # auto-insert. Builder is supposed to cap at 700 anyway.
        script = script[:1450].rstrip() + "."

    body = {
        "video_inputs": [{
            "character": {"type": "avatar", "avatar_id": avatar_id},
            "voice":     {
                "type":       "text",
                "voice_id":   voice_id,
                "input_text": script,
            },
            "background": {"type": "color", "value": background_color},
        }],
        "dimension": {"width": int(width), "height": int(height)},
    }
    data = _request("POST", "/v2/video/generate",
                    api_key=api_key, json_body=body)
    if data.get("error"):
        raise HeyGenError(f"HeyGen error: {data['error']}")
    video_id = (data.get("data") or {}).get("video_id")
    if not video_id:
        raise HeyGenError(f"no video_id in response: {str(data)[:300]}")
    return str(video_id)


def get_status(*, video_id: str, api_key: Optional[str] = None) -> dict:
    """Poll the status of a generation. Returns a dict with:
      - status       : "pending" | "processing" | "completed" | "failed"
      - video_url    : str | None     (downloadable MP4, expires in 7 days)
      - thumbnail_url: str | None
      - duration     : float | None
      - error        : dict | None    (when status == "failed")
    """
    if not video_id:
        raise HeyGenError("video_id required")
    data = _request("GET", "/v1/video_status.get",
                    api_key=api_key, params={"video_id": video_id})
    payload = data.get("data") or {}
    if not isinstance(payload, dict):
        raise HeyGenError(f"unexpected status payload: {str(data)[:300]}")
    return {
        "status":        payload.get("status") or "unknown",
        "video_url":     payload.get("video_url"),
        "thumbnail_url": payload.get("thumbnail_url"),
        "duration":      float(payload.get("duration") or 0.0),
        "error":         payload.get("error"),
    }


def download_to_file(url: str, dest_path: str, timeout_s: int = 300) -> int:
    """Download the rendered video URL to ``dest_path``. Returns the
    bytes written. HeyGen URLs expire 7 days after generation, so we
    pull it immediately on ``completed``."""
    if not url:
        raise HeyGenError("download url is empty")
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=15), follow_redirects=True) as cli:
            with cli.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise HeyGenError(
                        f"download HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                written = 0
                with open(dest_path, "wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                        fh.write(chunk)
                        written += len(chunk)
                return written
    except httpx.HTTPError as exc:
        raise HeyGenError(f"download network error: {exc}") from exc
