"""Build a Veo-3-friendly text prompt from a TrendingTopic.

Veo rewards cinematic, visually concrete prompts.  Headlines alone are too
abstract ("Revanth Reddy criticises transfers"), so we rephrase them as a
SHOT description with subject, action, style and pacing hints.
"""
from __future__ import annotations

from typing import Optional

import models


# Vertical Shorts is the primary target; horizontal for `youtube_full`.
ASPECT_MAP = {
    "youtube_short":   "9:16",
    "instagram_reel":  "9:16",
    "youtube_full":    "16:9",
}


# Vertical prompt primer — Veo 3 handles on-screen text poorly, so we lean on
# non-text visuals (crowds, infographics, B-roll) and add the title as an
# overlay later in the compose step.  For tech topics specifically we nudge
# Veo toward moving logos, device close-ups, code flickering on screens, etc.
TECH_STYLE_HINT = (
    "Modern tech news b-roll, cinematic, 4K, smooth camera motion, "
    "neon accents, futuristic, depth of field. "
    "Do NOT render any on-screen text."
)


def build_prompt(topic: models.TrendingTopic, platform: str = "youtube_short") -> dict:
    """Return a dict with {prompt, aspect_ratio, duration_seconds, negative_prompt}."""
    title = (topic.video_title or "").strip()
    summary = (topic.topic_summary or title).strip()
    keywords = list(topic.keywords or [])
    kw_line = ", ".join(keywords[:6])

    # Core scene: one concrete subject + action grounded in the topic
    shot = (
        f"Cinematic tech news shot about: {title}. "
        f"Context: {summary}. "
    )
    if kw_line:
        shot += f"Visual cues: {kw_line}. "

    prompt = shot + TECH_STYLE_HINT

    return {
        "prompt":           prompt[:2000],  # Veo caps prompt length
        "aspect_ratio":     ASPECT_MAP.get(platform, "9:16"),
        "duration_seconds": 8,              # Veo 3 supports 4/6/8s
        # Things we actively don't want in the frame
        "negative_prompt":  "subtitles, captions, watermark, logos, text overlay, lower third",
    }
