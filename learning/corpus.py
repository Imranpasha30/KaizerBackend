"""Channel corpus miner — pulls top-performing videos + extracts patterns.

Flow (per channel):
  1. Resolve the channel's YouTube channel_id from its oauth_token row.
  2. List its uploads playlist (YouTube Data API v3, public).
  3. Fetch view counts → sort → take top 20% (min 5, max 30).
  4. Send the titles + descriptions to Gemini → JSON with:
        hook_patterns, emotional_triggers, power_words, political_framing.
  5. Upsert into `channel_corpus.payload`.

We prefer `YOUTUBE_DATA_API_KEY` (no OAuth cost to the user's channel) and
fall back to the channel's OAuth credentials if the key isn't set.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sqlalchemy.orm import Session
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)

import models
from config import settings
from youtube import oauth as yt_oauth
from learning.gemini_log import log_gemini_call


# ─── Config ───────────────────────────────────────────────────────────────

GEMINI_MODEL = os.environ.get("KAIZER_CORPUS_MODEL", "gemini-2.5-flash")

MAX_VIDEOS_FETCHED = 50      # Data API page size cap
TOP_PERCENT        = 0.20    # top 20% by views
MIN_TOP_N          = 5
MAX_TOP_N          = 30


# ─── Exceptions ───────────────────────────────────────────────────────────

class CorpusError(Exception):
    """Terminal error — no more retries."""


class TransientCorpusError(CorpusError):
    """Temporary error — retry with backoff."""


# ─── YouTube Data API client ──────────────────────────────────────────────

def _yt_service(db: Session, channel: models.Channel):
    """Build a youtube-v3 client. Prefer API key (public, no OAuth cost);
    fall back to the channel's OAuth credentials."""
    if settings.yt_data_api_key:
        return build("youtube", "v3", developerKey=settings.yt_data_api_key,
                     cache_discovery=False)
    try:
        creds = yt_oauth.get_credentials(db, channel.id)
    except yt_oauth.OAuthError as e:
        raise CorpusError(
            f"No YOUTUBE_DATA_API_KEY and channel '{channel.name}' is not connected: {e}"
        ) from e
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def _resolve_google_channel_id(channel: models.Channel) -> str:
    """Read the YouTube channel_id from the OAuth row (set during Connect)."""
    tok = channel.oauth_token
    if not tok or not tok.google_channel_id:
        raise CorpusError(
            f"Channel '{channel.name}' has no YouTube identity on file — Connect it first."
        )
    return tok.google_channel_id


def _fetch_top_videos(yt, google_channel_id: str) -> List[Dict[str, Any]]:
    """Return [{id,title,description,viewCount,publishedAt}, ...] sorted desc
    by viewCount. Raises TransientCorpusError on 5xx, CorpusError otherwise."""
    try:
        # 1. Resolve uploads playlist
        ch_resp = yt.channels().list(
            part="contentDetails", id=google_channel_id,
        ).execute()
        items = ch_resp.get("items") or []
        if not items:
            raise CorpusError(f"YouTube returned no channel data for id {google_channel_id}")
        uploads_pid = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

        # 2. Walk the uploads playlist (first page — up to 50 recent videos)
        pl_resp = yt.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_pid,
            maxResults=MAX_VIDEOS_FETCHED,
        ).execute()
        video_ids = [
            it["contentDetails"]["videoId"]
            for it in (pl_resp.get("items") or [])
            if it.get("contentDetails", {}).get("videoId")
        ]
        if not video_ids:
            return []

        # 3. Enrich with viewCount (videos.list is cheapest way)
        v_resp = yt.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids),
            maxResults=MAX_VIDEOS_FETCHED,
        ).execute()

        rows: List[Dict[str, Any]] = []
        for v in v_resp.get("items") or []:
            sn = v.get("snippet") or {}
            st = v.get("statistics") or {}
            rows.append({
                "id":           v.get("id", ""),
                "title":        (sn.get("title") or "").strip(),
                "description":  (sn.get("description") or "").strip(),
                "viewCount":    int(st.get("viewCount") or 0),
                "publishedAt":  sn.get("publishedAt") or "",
            })
        rows.sort(key=lambda r: r["viewCount"], reverse=True)
        return rows
    except HttpError as e:
        status = getattr(e.resp, "status", 0) if hasattr(e, "resp") else 0
        if status in (500, 502, 503, 504):
            raise TransientCorpusError(f"YouTube {status}: {e}") from e
        raise CorpusError(f"YouTube API error {status}: {e}") from e


def _top_slice(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    n = max(MIN_TOP_N, min(MAX_TOP_N, math.ceil(len(rows) * TOP_PERCENT)))
    return rows[:n]


# ─── Gemini pattern extraction ────────────────────────────────────────────

_PATTERN_SCHEMA = {
    "type": "object",
    "required": ["hook_patterns", "emotional_triggers", "power_words"],
    "properties": {
        "hook_patterns":      {"type": "array", "items": {"type": "string"}},
        "emotional_triggers": {"type": "array", "items": {"type": "string"}},
        "power_words":        {"type": "array", "items": {"type": "string"}},
        "political_framing":  {"type": "string"},
    },
}

_SYSTEM_PROMPT = (
    "You are an expert YouTube analyst specializing in Telugu news channels. "
    "From the given top-performing videos, extract the recurring editorial "
    "patterns that drive their CTR. Output STRICT JSON matching the schema. "
    "Keep lists short (3–8 items). Be specific, not generic. "
    "Patterns must be translatable into titles a model can write — not "
    "observations about the videos themselves."
)


def _build_pattern_prompt(channel: models.Channel, top_rows: List[Dict[str, Any]]) -> str:
    lines = [f"# Channel: {channel.name} ({channel.handle or ''})",
             f"# Language: {channel.language or 'te'}",
             f"# Analyzed: top {len(top_rows)} videos by viewCount", ""]
    for i, r in enumerate(top_rows, 1):
        views = f"{r['viewCount']:,}" if r['viewCount'] else "?"
        desc = (r['description'] or "").split("\n", 1)[0][:160]
        lines.append(f"{i}. [{views} views] {r['title']}")
        if desc:
            lines.append(f"   snippet: {desc}")
    lines.extend([
        "",
        "Extract:",
        "- hook_patterns: 4–8 title-opening formulas that recur (e.g. "
        "'SHOCKING: {Event}', '{Leader} vs {Leader} — {Issue}', 'BREAKING: {Topic}').",
        "- emotional_triggers: 3–6 emotions these titles invoke "
        "(e.g. 'anger', 'curiosity', 'fear', 'pride').",
        "- power_words: 6–12 Telugu or English words that repeatedly appear "
        "and correlate with high views.",
        "- political_framing: 1–2 sentences describing the channel's "
        "political lean / framing style.",
        "",
        "Return JSON only.",
    ])
    return "\n".join(lines)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=12),
    retry=retry_if_exception_type(TransientCorpusError),
)
def _extract_patterns(channel: models.Channel,
                      top_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not settings.gemini_api_key:
        raise CorpusError("GEMINI_API_KEY is not set")
    genai.configure(api_key=settings.gemini_api_key)

    user_prompt = _build_pattern_prompt(channel, top_rows)

    try:
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=_SYSTEM_PROMPT,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": _PATTERN_SCHEMA,
                "temperature": 0.4,
                "max_output_tokens": 2048,
            },
        )
        with log_gemini_call(
            db=None,  # corpus refresh runs in background scheduler — no request-scoped session
            user_id=getattr(channel, "user_id", None),
            model=GEMINI_MODEL, purpose="corpus",
        ) as _gcall:
            resp = model.generate_content(user_prompt)
            _gcall.record(resp)
        text = (resp.text or "").strip()
        if not text:
            raise TransientCorpusError("Gemini returned empty response")
        data = json.loads(text)
        if not isinstance(data, dict):
            raise CorpusError(f"Gemini returned non-object JSON: {type(data).__name__}")
        return data
    except json.JSONDecodeError as e:
        raise TransientCorpusError(f"JSON decode failed: {e}") from e
    except (TransientCorpusError, CorpusError):
        raise
    except Exception as e:
        msg = str(e).lower()
        if any(m in msg for m in ("503", "500", "502", "504", "rate", "deadline", "quota", "unavailable")):
            raise TransientCorpusError(f"transient gemini: {e}") from e
        raise CorpusError(f"gemini error: {e}") from e


# ─── Public API ───────────────────────────────────────────────────────────

def refresh_channel(db: Session, channel_id: int) -> Dict[str, Any]:
    """Mine top videos + extract patterns + upsert channel_corpus row.

    Returns the new payload dict. Raises CorpusError on terminal failure.
    """
    channel = db.query(models.Channel).filter(models.Channel.id == channel_id).first()
    if not channel:
        raise CorpusError(f"Channel id={channel_id} not found")

    google_id = _resolve_google_channel_id(channel)
    yt = _yt_service(db, channel)

    videos = _fetch_top_videos(yt, google_id)
    if not videos:
        raise CorpusError(f"No uploads found for '{channel.name}' — nothing to learn from")

    top = _top_slice(videos)
    patterns = _extract_patterns(channel, top)

    payload: Dict[str, Any] = {
        "top_titles":          [r["title"] for r in top if r["title"]],
        "top_descriptions":    [r["description"] for r in top if r["description"]][:10],
        "hook_patterns":       list(patterns.get("hook_patterns") or []),
        "emotional_triggers":  list(patterns.get("emotional_triggers") or []),
        "power_words":         list(patterns.get("power_words") or []),
        "political_framing":   (patterns.get("political_framing") or "").strip(),
        "sample_size":         len(top),
        "fetched_total":       len(videos),
        "youtube_channel_id":  google_id,
        "refreshed_at":        datetime.now(timezone.utc).isoformat(),
        "model":               GEMINI_MODEL,
    }

    row = channel.corpus
    if row is None:
        row = models.ChannelCorpus(channel_id=channel.id, payload=payload)
        db.add(row)
    else:
        row.payload = payload
        row.refreshed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return payload


def refresh_all_priority(db: Session) -> Dict[str, Any]:
    """Refresh every channel that's priority AND connected. Returns a summary."""
    rows = (
        db.query(models.Channel)
          .filter(models.Channel.is_priority == True)  # noqa: E712
          .all()
    )
    refreshed: List[str] = []
    failed:    List[Dict[str, str]] = []
    for ch in rows:
        if not ch.oauth_token or not ch.oauth_token.google_channel_id:
            continue  # silently skip unconnected priority channels
        try:
            refresh_channel(db, ch.id)
            refreshed.append(ch.name)
        except Exception as e:
            failed.append({"channel": ch.name, "error": str(e)[:200]})
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "refreshed":  refreshed,
        "failed":     failed,
    }
