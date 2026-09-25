"""Insights deep root-cause analysis engine.

Turns a snapshot's VideoMetric rows into an evidence-backed diagnosis: performance tiers,
the 10 root-cause dimensions (each with a quantified finding + the offending video_ids +
a specific fix), a correlation-ranked driver list, concrete targets, and a next-10
checklist. STARTER mode prescribes against niche benchmarks for new channels.

DESIGN: precision on what's controllable, honesty on what isn't. Findings are data-driven
(correlation, not proven causation — stated as such); every claim cites video_ids. The
report covers the WHOLE history; metrics that need owner analytics degrade gracefully in
public mode, and thumbnail CTR is labelled by its Reporting-API coverage.

The pure core ``analyze_videos`` is unit-tested in ``scripts/test_insights_analyze.py``;
``analyze_snapshot`` loads rows + persists an AnalysisRun.
"""
from __future__ import annotations

import logging
import re
from statistics import median as _median_raw
from typing import List, Optional

from insights import models as im
from insights import maturity as M

log = logging.getLogger("kaizer.insights.analyze")

ENGINE_VERSION = "v1"

# Curated public reference baselines for STARTER mode (DISCOVERY.md §8). Ballparks, not
# channel-specific — clearly labelled as such in the report.
NICHE_BENCHMARKS = {
    "default": {"ctr": 0.05, "avg_view_pct": 0.40, "label": "general YouTube"},
    "news":    {"ctr": 0.05, "avg_view_pct": 0.38, "label": "news / current-affairs"},
}

# Modest, extensible high-CTR vocabulary. Correlation is computed on THIS channel's data;
# the list just defines candidate features.
POWER_WORDS = {
    "shocking", "exposed", "breaking", "revealed", "secret", "exclusive", "truth",
    "warning", "alert", "huge", "massive", "viral", "stop", "never", "finally",
    "why", "how", "vs", "biggest", "first", "last", "real", "caught", "shock",
}

# Plain-language definitions so a non-expert understands the report. Surfaced in the
# report (a "What these words mean" section) + the UI glossary.
GLOSSARY = {
    "Early velocity": "How fast a video gets views in its first 2 days. A fast start tells "
                      "YouTube to show it to MORE people (Home & Suggested). A slow start usually "
                      "means it stays small. Think of it as the video's 'launch speed'.",
    "CTR (click-through rate)": "Out of everyone who SAW your thumbnail, the percentage who clicked. "
                                "Higher means your title + thumbnail are doing their job. ~4–6% is healthy for news.",
    "Retention": "Of the people who clicked, how much of the video they actually watched (as a %). "
                 "Higher means your hook and pacing are holding viewers.",
    "Reach ratio": "Views ÷ your subscriber count. Above 1 means the video reached BEYOND your "
                   "subscribers — i.e. the algorithm pushed it to new people. That's how videos go big.",
    "Browse": "Views from YouTube's Home page / feed — the algorithm actively recommending you.",
    "Suggested": "Views from the 'Up next' list shown beside other videos.",
    "Subscriber feed": "Views from your own subscribers. If almost ALL views come from here, the "
                       "video never escaped to new audiences — that's a reach ceiling.",
    "Impressions": "How many times YouTube showed your thumbnail to someone.",
}

# YouTube video category ids → name. High-volume categories are formats where posting many
# videos/day is NORMAL (news, entertainment, comedy, vlogs, gaming) — for these we NEVER tell
# the creator to post less; the strategy is to prioritize within the volume.
YT_CATEGORIES = {
    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", "15": "Pets & Animals",
    "17": "Sports", "18": "Short Movies", "19": "Travel & Events", "20": "Gaming",
    "21": "Videoblogging", "22": "People & Blogs", "23": "Comedy", "24": "Entertainment",
    "25": "News & Politics", "26": "Howto & Style", "27": "Education",
    "28": "Science & Technology", "29": "Nonprofits & Activism",
}
HIGH_VOLUME_CATEGORIES = {"25", "24", "23", "22", "20"}


def detect_category(videos: List[dict]) -> dict:
    """The channel's dominant YouTube category (mode of per-video categoryId) → drives a
    category-appropriate strategy (e.g. a News channel is high-volume; never told to post less)."""
    freq: dict = {}
    for v in videos:
        cid = (v.get("category_id") or "").strip()
        if cid:
            freq[cid] = freq.get(cid, 0) + 1
    if not freq:
        return {"id": "", "name": "this type of", "high_volume": False, "is_news": False}
    cid = max(freq, key=freq.get)
    return {"id": cid, "name": YT_CATEGORIES.get(cid, "this type of"),
            "high_volume": cid in HIGH_VOLUME_CATEGORIES, "is_news": cid == "25"}


# ── tiny stats (stdlib only) ─────────────────────────────────────────────

def _median(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]
    return float(_median_raw(xs)) if xs else 0.0


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    """Correlation over paired non-None samples. None when <3 points or zero variance."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sx = sum((p[0] - mx) ** 2 for p in pairs)
    sy = sum((p[1] - my) ** 2 for p in pairs)
    if sx <= 0 or sy <= 0:
        return None
    cov = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    return round(cov / (sx ** 0.5 * sy ** 0.5), 3)


def _ids(videos: List[dict], n: int = 8) -> List[str]:
    return [v["video_id"] for v in videos[:n] if v.get("video_id")]


_DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ── title linguistics (pure) ─────────────────────────────────────────────

def title_features(title: str) -> dict:
    t = title or ""
    words = re.findall(r"\S+", t)
    letters = [c for c in t if c.isalpha() and c.isascii()]
    caps = [c for c in letters if c.isupper()]
    toks = {w.strip(".,!?:;\"'()[]").lower() for w in words}
    return {
        "len_chars": len(t),
        "len_words": len(words),
        "has_number": 1 if re.search(r"\d", t) else 0,
        "has_question": 1 if "?" in t else 0,
        "has_exclaim": 1 if "!" in t else 0,
        "has_brackets": 1 if re.search(r"[\[\](){}]", t) else 0,
        "caps_ratio": round(len(caps) / len(letters), 3) if letters else 0.0,
        "power_words": len(toks & POWER_WORDS),
    }


# ── tiers ────────────────────────────────────────────────────────────────

def compute_tiers(videos: List[dict]) -> dict:
    """Breakout (>5× median views), Underperformer (<100 OR <0.2× median), Solid (rest).
    Long-form only when present (Shorts have different mechanics)."""
    base = [v for v in videos if not v.get("is_short")] or list(videos)
    med = _median([v.get("view_count") or 0 for v in base])
    out = {"breakout": [], "solid": [], "under": [], "median_views": med}
    for v in base:
        vc = v.get("view_count") or 0
        if med > 0 and vc > 5 * med:
            out["breakout"].append(v)
        elif vc < 100 or (med > 0 and vc < 0.2 * med):
            out["under"].append(v)
        else:
            out["solid"].append(v)
    for k in ("breakout", "solid", "under"):
        out[k].sort(key=lambda v: v.get("view_count") or 0, reverse=True)
    return out


def _tier_traits(vids: List[dict]) -> dict:
    if not vids:
        return {"n": 0}
    def share(bucket):
        ss = []
        for v in vids:
            ts = v.get("traffic_sources") or {}
            tot = sum(ts.values()) if ts else 0
            if tot > 0:
                ss.append(ts.get(bucket, 0) / tot)
        return _mean(ss)
    return {
        "n": len(vids),
        # All averages EXCLUDE videos missing that field (never count missing as 0). _mean
        # returns None when none have data → the report renders "—", not a fake "0".
        "avg_ctr": _mean([v.get("impressions_ctr") for v in vids]),
        "avg_retention_pct": _mean([v.get("avg_view_percentage") for v in vids]),
        # TRUE measured early velocity only (views_first_48h), never the lifetime fallback.
        "avg_velocity_vph": _mean([_true_vph(v) for v in vids]),
        "n_velocity": sum(1 for v in vids if _true_vph(v) is not None),
        "avg_title_len": _mean([len(v.get("title") or "") for v in vids]),
        "browse_share": share("browse"),
        "suggested_share": share("suggested"),
        "subscriber_share": share("subscriber"),
        "top_topics": _top_topics(vids, 3),
        "example_ids": _ids(vids, 5),
    }


def _top_topics(videos: List[dict], n: int) -> List[str]:
    freq: dict = {}
    for v in videos:
        t = v.get("topic_cluster") or "general"
        freq[t] = freq.get(t, 0) + 1
    return [t for t, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:n]]


def _finding(key, title, finding, *, impact="", metric=None, evidence=None, fix="", status="ok") -> dict:
    return {"key": key, "title": title, "finding": finding, "impact": impact,
            "metric": metric, "evidence": evidence or [], "fix": fix, "status": status}


# ── the 10 dimensions (pure) ─────────────────────────────────────────────

def dim_ctr(videos, tiers) -> dict:
    with_ctr = [v for v in videos if v.get("impressions_ctr") is not None]
    if len(with_ctr) < 3:
        return _finding("ctr", "Click-through rate", status="populating",
                        finding="Thumbnail CTR is still populating — your reach report is scheduled and "
                                "the exact Studio CTR arrives shortly (~24–48h after connect, then a ~30-day "
                                "window). Using early-velocity as a packaging proxy in the meantime.",
                        fix="No action needed — CTR fills in automatically once YouTube generates the report.")
    bo = [v["impressions_ctr"] for v in tiers["breakout"] if v.get("impressions_ctr") is not None]
    un = [v["impressions_ctr"] for v in tiers["under"] if v.get("impressions_ctr") is not None]
    thresh = round((( _median(bo) + _median(un)) / 2) if (bo and un) else _median([v["impressions_ctr"] for v in with_ctr]), 4)
    low = sorted([v for v in with_ctr if v["impressions_ctr"] < thresh],
                 key=lambda v: v["impressions_ctr"])
    return _finding(
        "ctr", "Click-through rate",
        finding=f"Your hits average {(_mean(bo) or 0)*100:.1f}% CTR vs {(_mean(un) or 0)*100:.1f}% on flops. "
                f"The line separating them on this channel is ~{thresh*100:.1f}% CTR.",
        impact=f"{len(low)} videos sit below the {thresh*100:.1f}% packaging line.",
        metric={"threshold_ctr": thresh, "breakout_ctr": _mean(bo), "under_ctr": _mean(un)},
        evidence=_ids(low),
        fix="Re-package the sub-threshold titles/thumbnails; match what your breakout thumbnails do.")


def dim_title(videos) -> dict:
    feats = [(v, title_features(v.get("title") or "")) for v in videos]
    views = [v.get("view_count") or 0 for v, _ in feats]
    corr = {}
    for f in ("len_chars", "has_number", "has_question", "caps_ratio", "has_brackets", "power_words"):
        corr[f] = _pearson([ft[f] for _, ft in feats], views)
    ranked = sorted([(k, c) for k, c in corr.items() if c is not None], key=lambda kv: abs(kv[1]), reverse=True)
    if not ranked:
        return _finding("title", "Title & wording", status="insufficient_data",
                        finding="Not enough variation to correlate title patterns yet.", fix="")
    top_k, top_c = ranked[0]
    direction = "more" if top_c > 0 else "fewer/less"
    return _finding(
        "title", "Title & wording",
        finding=f"On this channel, '{top_k}' correlates most with views (r={top_c:+.2f}) — "
                f"{direction} of it tracks with higher views.",
        impact="Title packaging is " + ("a strong" if abs(top_c) >= 0.3 else "a mild") + " lever here.",
        metric={"correlations": corr},
        evidence=_ids(sorted(videos, key=lambda v: v.get("view_count") or 0, reverse=True), 5),
        fix=f"Lean into the title patterns your top videos use; test '{top_k}' deliberately.")


def _thumbnail_vision(winners: list, losers: list):
    """Gemini-vision comparison of WINNING vs LOSING thumbnails → concrete,
    channel-specific rules in plain English. Returns (finding, fix) or None
    (fail-soft: no key, no images, download/vision error)."""
    try:
        import os as _os, httpx as _httpx
        from seo.generator import _gemini_client
        from google.genai import types as _gt
    except Exception:
        return None

    def _urls(vids, k):
        out = []
        for v in vids or []:
            u = (v.get("thumbnail_url") or "").strip()
            if u:
                out.append(u)
            if len(out) >= k:
                break
        return out

    win_u, los_u = _urls(winners, 6), _urls(losers, 6)
    if len(win_u) < 3 or len(los_u) < 3:
        return None
    try:
        client = _gemini_client()
    except Exception:
        return None

    def _parts(urls):
        ps = []
        for u in urls:
            try:
                r = _httpx.get(u, timeout=10)
                if r.status_code == 200 and r.content:
                    ps.append(_gt.Part.from_bytes(data=r.content, mime_type="image/jpeg"))
            except Exception:
                continue
        return ps

    win_p, los_p = _parts(win_u), _parts(los_u)
    if len(win_p) < 3 or len(los_p) < 3:
        return None
    model = _os.environ.get("KAIZER_INSIGHTS_VISION_MODEL", "gemini-2.5-flash")
    prompt = (
        "You are a YouTube thumbnail analyst. The FIRST images are this channel's WINNING "
        "thumbnails (high views); the SECOND set are its LOSING thumbnails (low views). "
        "In plain creator English, say what the WINNERS do differently — faces & expressions, "
        "amount and size of on-image text, colours, contrast, clutter, focal point — based on "
        "what you actually SEE. Then give 2-3 specific thumbnail rules for this channel. "
        "Under 110 words, no preamble, no headings.")
    contents = ["WINNING thumbnails:"] + win_p + ["LOSING thumbnails:"] + los_p + [prompt]
    try:
        resp = client.models.generate_content(
            model=model, contents=contents,
            config=_gt.GenerateContentConfig(temperature=0.4, max_output_tokens=400))
        txt = (resp.text or "").strip()
    except Exception:
        return None
    if not txt:
        return None
    return txt, ("Re-shoot or re-design the weakest thumbnails to follow these rules; "
                 "match what your winning thumbnails do.")


def dim_thumbnail(videos, tiers) -> dict:
    winners = (tiers.get("breakout") or
               sorted(videos, key=lambda v: v.get("view_count") or 0, reverse=True)[:8])
    losers = (tiers.get("under") or
              sorted(videos, key=lambda v: v.get("view_count") or 0)[:8])
    ev = _ids(sorted(videos, key=lambda v: v.get("view_count") or 0, reverse=True), 5)
    vis = _thumbnail_vision(winners, losers)
    if vis:
        finding, fix = vis
        return _finding("thumbnail", "Thumbnails", finding=finding, fix=fix, status="ok",
                        impact="Vision comparison of your winning vs losing thumbnails.",
                        evidence=ev)
    return _finding("thumbnail", "Thumbnails", status="needs_vision",
                    finding="Thumbnail comparison needs the thumbnail images — set a public YouTube "
                            "Data API key (YOUTUBE_DATA_API_KEY) and re-analyze to unlock it.",
                    fix="Until then, mirror the thumbnails of your breakout videos.",
                    evidence=ev)


def _avg_retention(videos):
    """THE single channel-average retention, used everywhere (tiers/section/targets share this
    population). Definition: per-video mean of averageViewPercentage (0–100), over LONG-FORM
    videos that have the data. Shorts retain very differently (and dominate this kind of
    channel), so they're excluded; missing data is excluded from the denominator, never 0."""
    vals = [v.get("avg_view_percentage") for v in videos
            if not v.get("is_short") and v.get("avg_view_percentage") is not None]
    return _mean(vals)


def dim_retention(videos) -> dict:
    longform = [v for v in videos
                if not v.get("is_short") and v.get("avg_view_percentage") is not None]
    if len(longform) < 3:
        return _finding("retention", "Audience retention", status="insufficient_data",
                        finding="Retention needs owner analytics access (re-connect with analytics scope).",
                        fix="Grant analytics access to unlock retention diagnosis.")
    avg = _avg_retention(videos) or 0.0
    weak = sorted([v for v in longform if (v["avg_view_percentage"] or 0) < avg],
                  key=lambda v: v["avg_view_percentage"])
    quad = [v for v in weak if (v["avg_view_percentage"] or 0) < avg * 0.8]
    return _finding(
        "retention", "Audience retention",
        finding=f"Average retention is {avg:.0f}%. {len(weak)} videos retain below your own average — "
                f"the hook/pacing is losing viewers early.",
        impact=f"{len(quad)} got the click but lost the watch (packaging worked, content/hook didn't).",
        metric={"avg_retention_pct": round(avg, 2)},
        evidence=_ids(quad or weak),
        fix="Tighten the first 15–30s hook and cut slow stretches on the flagged videos.")


def _true_vph(v):
    """REAL early velocity from the measured first-48h window (not the lifetime fallback)."""
    f = v.get("views_first_48h")
    return (f / 48.0) if f is not None else None


def dim_velocity(videos) -> dict:
    # Use ONLY the genuinely-measured first-48h data (available for recent uploads), never the
    # lifetime-views fallback — otherwise the signal is just views÷age (circular).
    with_v = [v for v in videos if _true_vph(v) is not None]
    if len(with_v) < 3:
        return _finding("velocity", "Early velocity (first 48h)", status="insufficient_data",
                        finding="Early velocity uses real first-48h data, which YouTube provides for "
                                "your most-recent uploads — not enough measured yet.", fix="")
    med = _median([_true_vph(v) for v in with_v])
    fast = sorted([v for v in with_v if _true_vph(v) > 2 * med], key=_true_vph, reverse=True)
    return _finding(
        "velocity", "Early velocity (first 48h)",
        finding=f"Median launch speed is ~{med:.0f} views/hr in the first 2 days (measured on your "
                f"{len(with_v)} most-recent uploads). Your fast starters launched at "
                f"{(_mean([_true_vph(v) for v in fast]) or 0):.0f}+ views/hr — that surge is what "
                f"tells YouTube to push a video onto Home & Suggested.",
        impact=f"{len(fast)} of those started fast; the rest stalled in their first hours.",
        metric={"median_vph": round(med, 2), "n_measured": len(with_v)},
        evidence=_ids(fast),
        fix="Post in your best window + push it to subscribers immediately so the first 2 hours spike.")


def dim_traffic(videos) -> dict:
    # Only videos that actually have traffic data (never average missing as 0). Buckets +
    # discovery set come from analytics_api (verified API enums), so "Browse" isn't a phantom.
    from insights.analytics_api import DISCOVERY_BUCKETS, TRAFFIC_BUCKETS
    with_t = [v for v in videos if v.get("traffic_sources")]
    if len(with_t) < 3:
        return _finding("traffic", "Traffic sources", status="insufficient_data",
                        finding="Traffic-source split needs owner analytics access.", fix="")
    cols = list(TRAFFIC_BUCKETS)

    def shares(v):
        ts = v["traffic_sources"]; tot = sum(ts.values()) or 1
        return {k: ts.get(k, 0) / tot for k in cols}

    avg = {b: (_mean([shares(v)[b] for v in with_t]) or 0.0) for b in cols}
    disc_avg = {b: avg.get(b, 0.0) for b in DISCOVERY_BUCKETS}
    top_src = max(disc_avg, key=disc_avg.get) if disc_avg else "discovery"
    avg_sub = avg.get("subscriber", 0.0)

    def disc_share(v):
        s = shares(v)
        return sum(s.get(b, 0.0) for b in DISCOVERY_BUCKETS)
    # "trapped" = mostly subscriber feed AND little reach into ANY discovery source.
    trapped = [v for v in with_t if shares(v)["subscriber"] > 0.6 and disc_share(v) < 0.2]
    return _finding(
        "traffic", "Traffic sources",
        finding=f"On average {avg[top_src]*100:.0f}% of views come from {top_src.title()} and "
                f"{avg_sub*100:.0f}% from your subscriber feed.",
        impact=f"{len(trapped)} videos were trapped in the subscriber feed and never reached "
               f"discovery — that's the reach ceiling.",
        metric={"avg_shares": {b: round(avg[b], 4) for b in cols}, "top_discovery": top_src,
                "avg_subscriber_share": round(avg_sub, 4)},
        evidence=_ids(trapped),
        fix="The trapped videos need stronger packaging + early velocity to earn a discovery push; "
            "broaden topics beyond what only existing subs click.")


def dim_timing(videos) -> dict:
    by_window: dict = {}
    for v in videos:
        dow, hr = v.get("publish_dow"), v.get("publish_hour_local")
        if dow is None or hr is None:
            continue
        sig = v.get("views_per_hour_48h")
        sig = sig if sig is not None else (v.get("view_count") or 0)
        by_window.setdefault((dow, hr), []).append(sig)
    if not by_window:
        return _finding("timing", "Publish timing", status="insufficient_data",
                        finding="No localized publish-time data yet.", fix="")
    scored = sorted([((d, h), _median(vs), len(vs)) for (d, h), vs in by_window.items() if len(vs) >= 1],
                    key=lambda x: x[1], reverse=True)
    best = [{"day": _DOW[d], "hour": h, "score": round(s, 1), "n": n} for (d, h), s, n in scored[:3]]
    worst = [{"day": _DOW[d], "hour": h, "score": round(s, 1), "n": n} for (d, h), s, n in scored[-3:]]
    top = best[0] if best else None
    # Full 7×24 grid for the heatmap chart (every cell that has data).
    heatmap = [{"dow": d, "hour": h, "score": round(_median(vs), 1), "n": len(vs)}
               for (d, h), vs in by_window.items()]
    # Best hour per weekday → a concrete weekly posting schedule (Mon … Sun).
    per_day_map = {}
    for (d, h), vs in by_window.items():
        sc = _median(vs)
        if d not in per_day_map or sc > per_day_map[d][1]:
            per_day_map[d] = (h, sc, len(vs))
    per_day_best = [{"dow": d, "day": _DOW[d], "hour": per_day_map[d][0],
                     "score": round(per_day_map[d][1], 1), "n": per_day_map[d][2]}
                    for d in sorted(per_day_map)]
    return _finding(
        "timing", "Publish timing",
        finding=(f"Your strongest window is {top['day']} ~{top['hour']:02d}:00 (your channel's local "
                 f"time) — videos posted then get the fastest early views." if top else "—"),
        impact="Posting in your best window instead of your worst measurably changes how fast a "
               "video takes off in its first hours (its launch speed).",
        metric={"best_windows": best, "worst_windows": worst,
                "heatmap": heatmap, "per_day_best": per_day_best},
        evidence=[],
        fix="Schedule uploads into the top windows below; avoid the weakest ones.")


def dim_cadence(videos, category: dict) -> dict:
    """Category-AWARE. A News (or other high-volume) channel is SUPPOSED to post a lot — we
    never tell them to cut volume; the fix is prioritizing within it. Low-volume channels get
    the spacing/cannibalization analysis."""
    dated = sorted([v for v in videos if v.get("published_at_utc")], key=lambda v: v["published_at_utc"])
    if len(dated) < 5:
        return _finding("cadence", "Upload cadence", status="insufficient_data",
                        finding="Not enough dated uploads to model cadence.", fix="")
    gaps, reach = [], []
    for i in range(1, len(dated)):
        prev, cur = dated[i - 1], dated[i]
        gap_h = (cur["published_at_utc"] - prev["published_at_utc"]).total_seconds() / 3600.0
        if gap_h <= 0:
            continue
        gaps.append(gap_h)
        reach.append(cur.get("reach_ratio") if cur.get("reach_ratio") is not None else (cur.get("view_count") or 0))
    corr = _pearson(gaps, reach)
    median_gap_h = _median(gaps)
    per_day = round(24.0 / median_gap_h, 2) if median_gap_h > 0 else 0
    metric = {"videos_per_day": per_day, "median_gap_hours": round(median_gap_h, 1),
              "gap_reach_corr": corr, "category": category.get("name"),
              "high_volume": category.get("high_volume", False)}

    if category.get("high_volume"):
        # News/high-volume: volume IS the format. Don't reduce it — prioritize within it.
        return _finding(
            "cadence", "Upload cadence & strategy",
            finding=f"You publish ~{per_day} videos/day — for a {category.get('name','news')} channel "
                    f"that high volume is normal and necessary (you have to cover everything). The "
                    f"problem is NOT how many you post.",
            impact="At this volume your best stories can get buried in your own feed — the ones that "
                   "deserve a big push don't always get the early momentum to earn it.",
            metric=metric, evidence=[],
            fix="Keep the volume. Each day, RANK your uploads by potential and give your top 2–3 the "
                "best publish window + your strongest title & thumbnail — so the big stories get the "
                "early surge instead of being lost in the stream. Treat the rest as steady coverage.")

    cannibal = (corr is not None and corr > 0.15)  # bigger gap → more reach ⇒ tight spacing hurts
    return _finding(
        "cadence", "Upload cadence",
        finding=f"You post ~{per_day} videos/day (median gap {median_gap_h:.0f}h). "
                + ("Posting closer together tracks with LOWER reach — your uploads are competing with "
                   "each other for the same Home-feed slots." if cannibal
                   else "Your spacing doesn't appear to be limiting reach right now."),
        impact=("Posting too close splits your Home-feed promotion across videos." if cannibal
                else "Cadence is not the bottleneck."),
        metric=metric, evidence=[],
        fix=("Space uploads further apart so each gets its own Home-feed window, and cut low-confidence posts."
             if cannibal else "Maintain your cadence; focus the fix elsewhere."))


def dim_topic(videos) -> dict:
    by: dict = {}
    for v in videos:
        by.setdefault(v.get("topic_cluster") or "general", []).append(v.get("view_count") or 0)
    if len(by) < 2:
        return _finding("topic", "Topics & freshness", status="insufficient_data",
                        finding="Not enough topic variety to rank.", fix="")
    ranked = sorted([(t, _median(vs), len(vs)) for t, vs in by.items()], key=lambda x: x[1], reverse=True)
    winners = [{"topic": t, "median_views": round(mv), "n": n} for t, mv, n in ranked[:3]]
    losers = [{"topic": t, "median_views": round(mv), "n": n} for t, mv, n in ranked[-3:]]
    return _finding(
        "topic", "Topics & freshness",
        finding=f"Your strongest topic cluster is '{winners[0]['topic']}' "
                f"(median {winners[0]['median_views']:,} views).",
        impact="Topic choice is a top reach driver for news — fresh, high-stakes angles win.",
        metric={"winning_topics": winners, "weak_topics": losers},
        evidence=[],
        fix="Do more of the winning clusters while fresh; drop or re-angle the weak ones. "
            "(Event→publish freshness scoring is a queued NLP enrichment.)")


WEAK_CORR = 0.05      # |r| below this = no measurable effect → not a "driver"
LOW_COVERAGE = 0.25   # a factor measured on <25% of the catalog is "limited sample"


def rank_drivers(videos) -> List[dict]:
    """Correlate each controllable factor with views — each correlation computed ONLY over the
    videos that actually have that factor's data (missing excluded, never treated as 0). Every
    driver carries its sample size `n` + coverage. Ranking: WELL-COVERED drivers (≥25% of the
    catalog) rank above LIMITED-SAMPLE ones regardless of |r| — so a metric measured on ~100%
    of videos (retention) isn't silently outranked by one on <5% (early velocity). Near-zero
    correlations (|r|<0.05) are dropped — they're not drivers. Correlation, not proven causation."""
    n_total = len(videos) or 1
    views = [v.get("view_count") or 0 for v in videos]

    def _browse_share(v):
        ts = v.get("traffic_sources")
        if not ts:
            return None
        tot = sum(ts.values()) or 1
        return ts.get("browse", 0) / tot

    feats = {
        "ctr": [v.get("impressions_ctr") for v in videos],
        "retention": [v.get("avg_view_percentage") for v in videos],
        # TRUE early velocity only (measured first-48h) — the lifetime fallback would be views÷age,
        # which correlates with views by construction (circular). None when not measured.
        "early_velocity": [_true_vph(v) for v in videos],
        "browse_share": [_browse_share(v) for v in videos],
        "title_length": [len(v.get("title") or "") for v in videos],
        "has_number_in_title": [title_features(v.get("title") or "")["has_number"] for v in videos],
    }
    out = []
    for k, xs in feats.items():
        c = _pearson(xs, views)
        if c is None or abs(c) < WEAK_CORR:          # drop near-zero / uncomputable (#4)
            continue
        n = sum(1 for x in xs if x is not None)
        cov = round(n / n_total, 4)
        out.append({
            "factor": k, "correlation": c, "n": n, "coverage": cov,
            "limited_sample": cov < LOW_COVERAGE,
            "direction": "higher → more views" if c > 0 else "higher → fewer views",
        })
    # well-covered first (limited_sample False sorts before True), then by |r| within each group.
    out.sort(key=lambda d: (d["limited_sample"], -abs(d["correlation"])))
    return out


def compute_targets(tiers, dims) -> dict:
    bo = _tier_traits(tiers["breakout"])
    timing = next((d for d in dims if d["key"] == "timing"), {})
    cadence = next((d for d in dims if d["key"] == "cadence"), {})
    return {
        "target_ctr": bo.get("avg_ctr"),
        "target_retention_pct": bo.get("avg_retention_pct"),
        "best_windows": (timing.get("metric") or {}).get("best_windows", []),
        "recommended_videos_per_day": (cadence.get("metric") or {}).get("videos_per_day"),
    }


def build_next_10(dims, targets) -> List[str]:
    out = []
    t = targets
    if t.get("target_ctr"):
        out.append(f"Design each thumbnail/title to beat {t['target_ctr']*100:.0f}% CTR (your winners' level).")
    if t.get("target_retention_pct"):
        out.append(f"Hook the first 30s to hold ≥{t['target_retention_pct']:.0f}% average retention.")
    if t.get("best_windows"):
        w = t["best_windows"][0]
        out.append(f"Publish in your top window: {w['day']} ~{w['hour']:02d}:00 local.")
    # cadence advice is category-aware (the dim itself knows news = high-volume) — pull its fix.
    for d in dims:
        if d["status"] == "ok" and d.get("fix") and d["key"] in ("cadence", "ctr", "retention", "traffic", "topic"):
            out.append(d["fix"])
    out.append("Lead the title with the freshest, highest-stakes angle within the first 3 words.")
    return out[:10]


# ── orchestration (pure core) ────────────────────────────────────────────

def analyze_videos(videos: List[dict], *, mode: str, subscriber_count: int = 0,
                   access_mode: str = "public", analytics_error: str = None) -> dict:
    """Compute the full diagnosis from VideoMetric-shaped dicts. Returns the structured
    results stored on AnalysisRun.results (the report generator renders prose over this)."""
    n = len(videos)
    n_ctr = sum(1 for v in videos if v.get("impressions_ctr") is not None)
    coverage = {
        "n_videos": n, "n_with_ctr": n_ctr, "access_mode": access_mode, "caveats": [],
    }
    if access_mode != "full":
        coverage["caveats"].append("Public mode: no CTR/retention/traffic (owner analytics not granted).")
    elif analytics_error == "analytics_api_disabled":
        coverage["caveats"].append(
            "Deep analytics are OFF — the YouTube Analytics API is disabled in your Google Cloud "
            "project. Enable the YouTube Analytics API (and YouTube Reporting API for CTR), then "
            "re-analyze to unlock retention, traffic sources, early velocity and timezone.")
    elif analytics_error in ("analytics_forbidden", "analytics_unavailable"):
        coverage["caveats"].append(
            "Deep analytics couldn't be read (permission/availability). Re-connect this channel "
            "granting analytics access, then re-analyze.")
    elif analytics_error == "reporting_unavailable":
        coverage["caveats"].append(
            "Thumbnail CTR needs the YouTube Reporting API enabled in your Google Cloud project "
            "(separate from Analytics). Once on, exact Studio CTR arrives ~24–48h and covers ~30 "
            "days. Retention, traffic and velocity below are live.")
    elif n_ctr == 0:
        # Job is scheduled; data hasn't landed yet. Per spec: show "populating", never 0 / error.
        coverage["caveats"].append(
            "Thumbnail CTR is populating — your reach report is scheduled; exact Studio CTR arrives "
            "~24–48h after connect, then covers ~30 days (older videos can't be backfilled). "
            "Retention, traffic and velocity below are live.")
    elif n_ctr < n:
        coverage["caveats"].append(
            f"Thumbnail CTR available for {n_ctr}/{n} videos (more lands daily; ~30-day window).")

    if mode == M.STARTER or n < 5:
        return _starter_plan(videos, coverage)

    category = detect_category(videos)
    tiers = compute_tiers(videos)
    dims = [
        dim_ctr(videos, tiers), dim_title(videos), dim_thumbnail(videos, tiers),
        dim_retention(videos), dim_velocity(videos), dim_traffic(videos),
        dim_timing(videos), dim_cadence(videos, category), dim_topic(videos),
    ]
    drivers = rank_drivers(videos)
    # synthesis = dimension #10
    top_driver = drivers[0] if drivers else None
    root = None
    if top_driver:
        root_dim = {"ctr": "ctr", "retention": "retention", "early_velocity": "velocity",
                    "browse_share": "traffic"}.get(top_driver["factor"])
        root = next((d for d in dims if d["key"] == root_dim), None) if root_dim else None
    synthesis = _finding(
        "synthesis", "Root-cause synthesis",
        finding=(f"The single biggest fixable driver of your view gap is **{(root or dims[0])['title']}** "
                 f"(strongest correlation with views: {top_driver['factor']} r={top_driver['correlation']:+.2f}, "
                 f"measured on {top_driver['n']:,} videos"
                 f"{' — limited sample' if top_driver.get('limited_sample') else ''})."
                 if top_driver else "Insufficient data to rank drivers."),
        impact="Fix this first — it compounds: better packaging → more early velocity → a Browse push → more reach.",
        metric={"ranked_drivers": drivers},
        evidence=(root or dims[0]).get("evidence", []),
        fix=(root or dims[0]).get("fix", ""))
    dims.append(synthesis)

    targets = compute_targets(tiers, dims)
    return {
        "mode": mode,
        "engine_version": ENGINE_VERSION,
        "tiers": {k: _tier_traits(tiers[k]) for k in ("breakout", "solid", "under")},
        "median_views": tiers["median_views"],
        "dimensions": dims,
        "drivers": drivers,
        "targets": targets,
        "next_10": build_next_10(dims, targets),
        "data_coverage": coverage,
        "headline": synthesis["finding"],
        "glossary": GLOSSARY,
        # Surfaced top-level for the UI charts (weekly heatmap + per-day schedule).
        "timing": next((d.get("metric") for d in dims if d["key"] == "timing"), {}),
        # Detected channel type → drives category-appropriate strategy (news = high-volume).
        "channel_profile": category,
    }


def _starter_plan(videos: List[dict], coverage: dict) -> dict:
    bm = NICHE_BENCHMARKS["news"]
    return {
        "mode": M.STARTER,
        "engine_version": ENGINE_VERSION,
        "headline": "Starter plan — not enough owned history to diagnose yet, so here's a "
                    "prescriptive first-90-days plan benchmarked to news/current-affairs norms.",
        "benchmarks": bm,
        "data_coverage": coverage,
        "plan": {
            "packaging": [f"Target ≥{bm['ctr']*100:.0f}% CTR: one clear subject, high contrast, ≤4 words of text on the thumbnail.",
                          "Lead titles with the freshest, highest-stakes angle in the first 3 words."],
            "hook": [f"Hold ≥{bm['avg_view_pct']*100:.0f}% average retention: state the payoff in the first 10 seconds."],
            "cadence": ["Start 1 video/day, consistent; space them so each gets its own Browse window."],
            "windows": ["Post when your target audience is active (early morning + evening local); refine from your own data as it accrues."],
            "browse": ["Earn Browse/Suggested with strong early velocity: notify subscribers, then let packaging carry it beyond the sub feed."],
        },
        "next_10": [
            f"Publish 10 videos with thumbnails built to beat {bm['ctr']*100:.0f}% CTR.",
            "Front-load the hook (payoff in 10s) on every one.",
            "Keep a steady daily cadence; don't bunch uploads.",
            "Lead every title with the freshest angle.",
            "Track which topics get picked up by Browse and double down.",
        ],
        "note": "This plan sharpens into a full root-cause diagnosis as your channel accumulates data.",
    }


# ── DB wrapper ───────────────────────────────────────────────────────────

def _row_to_dict(vm: im.VideoMetric) -> dict:
    return {c.name: getattr(vm, c.name) for c in vm.__table__.columns}


def analyze_snapshot(db, snapshot_id: int) -> im.AnalysisRun:
    """Load a snapshot's VideoMetric rows, route by maturity, compute, and persist an
    AnalysisRun. Raises if the snapshot is missing."""
    snap = db.query(im.ChannelSnapshot).filter(im.ChannelSnapshot.id == snapshot_id).first()
    if not snap:
        raise RuntimeError(f"snapshot {snapshot_id} not found")
    rows = db.query(im.VideoMetric).filter(im.VideoMetric.snapshot_id == snapshot_id).all()
    videos = [_row_to_dict(r) for r in rows]
    mode = M.classify(video_count=snap.video_count or 0, channel_age_days=snap.channel_age_days or 0,
                      total_views=snap.view_count or 0, analytics_granted=(snap.access_mode == "full")).mode
    run = im.AnalysisRun(snapshot_id=snap.id, user_id=snap.user_id, maturity=mode,
                         status="running", engine_version=ENGINE_VERSION, params={})
    db.add(run); db.flush()
    try:
        results = analyze_videos(videos, mode=mode, subscriber_count=snap.subscriber_count or 0,
                                 access_mode=snap.access_mode or "public", analytics_error=snap.error)
        run.results = results
        run.status = "ok"
        from sqlalchemy.sql import func as _func
        run.finished_at = _func.now()
    except Exception as exc:
        run.status = "failed"; run.error = str(exc)[:500]
        log.exception("insights analysis failed for snapshot %s", snapshot_id)
    db.commit(); db.refresh(run)
    return run
