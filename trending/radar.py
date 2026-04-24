"""Competitor watcher — polls the last uploads from every active competitor,
dedupes, summarizes with Gemini, writes TrendingTopic rows.

Runs every 2h from learning/scheduler.py. Can also be triggered on demand
via POST /api/trending/refresh.
"""
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone, timedelta
from typing import List, Dict

import google.generativeai as genai
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sqlalchemy.orm import Session

import models
from config import settings
from database import SessionLocal
from learning.gemini_log import log_gemini_call


FETCH_PER_CHANNEL = 10
TOPIC_MODEL = os.environ.get("KAIZER_TOPIC_MODEL", "gemini-2.5-flash")


_TOPIC_SCHEMA = {
    "type": "object",
    "required": ["summary", "keywords", "urgency"],
    "properties": {
        "summary":  {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "urgency":  {"type": "string"},     # hot | normal | low
    },
}


def _yt():
    if not settings.yt_data_api_key:
        raise RuntimeError(
            "YOUTUBE_DATA_API_KEY not set — trending radar requires the public Data API key."
        )
    return build("youtube", "v3", developerKey=settings.yt_data_api_key, cache_discovery=False)


def _fetch_recent(yt, youtube_channel_id: str) -> List[Dict]:
    try:
        ch_resp = yt.channels().list(
            part="contentDetails", id=youtube_channel_id,
        ).execute()
        items = ch_resp.get("items") or []
        if not items:
            return []
        uploads_pid = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

        pl = yt.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_pid,
            maxResults=FETCH_PER_CHANNEL,
        ).execute()
        video_ids = [
            i["contentDetails"]["videoId"] for i in (pl.get("items") or [])
            if i.get("contentDetails", {}).get("videoId")
        ]
        if not video_ids:
            return []

        v = yt.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids),
            maxResults=FETCH_PER_CHANNEL,
        ).execute()
        rows = []
        for item in v.get("items") or []:
            sn = item.get("snippet") or {}
            st = item.get("statistics") or {}
            rows.append({
                "video_id":     item.get("id", ""),
                "title":        sn.get("title", ""),
                "description":  sn.get("description", ""),
                "published_at": sn.get("publishedAt", ""),
                "view_count":   int(st.get("viewCount") or 0),
            })
        return rows
    except HttpError as e:
        print(f"[trending] videos fetch failed for {youtube_channel_id}: {e}")
        return []


def _summarize(title: str, description: str) -> Dict:
    """Short Gemini pass — topic + keywords + urgency label."""
    if not settings.gemini_api_key:
        return {"summary": title[:200], "keywords": [], "urgency": "normal"}
    try:
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(
            TOPIC_MODEL,
            system_instruction=(
                "Classify this Telugu news video into a terse topic summary and 3-6 "
                "keywords. Urgency: 'hot' if it's breaking/political-crisis, 'low' if "
                "filler/entertainment, 'normal' otherwise. Return STRICT JSON."
            ),
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": _TOPIC_SCHEMA,
                "temperature": 0.3,
                "max_output_tokens": 512,
            },
        )
        body = f"Title: {title}\nDescription: {(description or '')[:500]}"
        with log_gemini_call(
            db=None, model=TOPIC_MODEL, purpose="trending-topic",
        ) as _gcall:
            resp = model.generate_content(body)
            _gcall.record(resp)
        data = json.loads((resp.text or "").strip())
        u = (data.get("urgency") or "normal").lower()
        if u not in ("hot", "normal", "low"):
            u = "normal"
        return {
            "summary":  (data.get("summary") or title)[:500],
            "keywords": list(data.get("keywords") or [])[:8],
            "urgency":  u,
        }
    except Exception as e:
        print(f"[trending] summarize failed: {e}")
        return {"summary": title[:200], "keywords": [], "urgency": "normal"}


def refresh_all(since_hours: int | None = None) -> Dict:
    """Single sweep across all active competitor channels.

    When `since_hours` is set, only videos whose `published_at` falls inside
    that window are summarized + stored — older items are skipped entirely.
    This saves Gemini quota (no summary call for stale videos) and keeps the
    DB focused on the time range the user actually cares about.
    """
    db = SessionLocal()
    try:
        if not settings.yt_data_api_key:
            return {"error": "YOUTUBE_DATA_API_KEY not set"}

        cutoff = None
        if since_hours and since_hours > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=int(since_hours))

        yt = _yt()
        competitors = (
            db.query(models.CompetitorChannel)
              .filter(models.CompetitorChannel.active == True)  # noqa: E712
              .all()
        )
        added   = 0
        skipped_stale = 0
        for comp in competitors:
            try:
                videos = _fetch_recent(yt, comp.youtube_channel_id)
            except Exception as e:
                print(f"[trending] {comp.name} fetch failed: {e}")
                continue

            for v in videos:
                pub_at = None
                if v.get("published_at"):
                    try:
                        pub_at = datetime.fromisoformat(v["published_at"].replace("Z", "+00:00"))
                    except Exception:
                        pub_at = None

                # Window filter at source — skip before we spend a Gemini call
                if cutoff is not None and (pub_at is None or pub_at < cutoff):
                    skipped_stale += 1
                    continue

                exists = db.query(models.TrendingTopic).filter(
                    models.TrendingTopic.source_channel_id == comp.id,
                    models.TrendingTopic.video_id == v["video_id"],
                ).first()
                if exists:
                    continue

                meta = _summarize(v["title"], v["description"])

                row = models.TrendingTopic(
                    source_channel_id=comp.id,
                    video_id=v["video_id"],
                    video_title=v["title"],
                    video_url=f"https://youtu.be/{v['video_id']}",
                    published_at=pub_at,
                    view_count=v["view_count"],
                    topic_summary=meta["summary"],
                    keywords=meta["keywords"],
                    urgency=meta["urgency"],
                )
                db.add(row)
                added += 1
            db.commit()

        return {
            "ran_at":        datetime.now(timezone.utc).isoformat(),
            "competitors":   len(competitors),
            "new_topics":    added,
            "skipped_stale": skipped_stale,
            "window_hours":  since_hours or 0,
        }
    except Exception:
        traceback.print_exc()
        return {"ran_at": datetime.now(timezone.utc).isoformat(), "error": "refresh failed"}
    finally:
        db.close()
