"""REAL per-channel SEO learning — closes the loop that was collect-only.

Before this module, TrainingSample rows carried views/vph (and a broken
ctr=None) but NOTHING ever read them back: SEO generated the same way on
day 100 as on day 1 (operator: "learning built but never used — I need
real learning, not faking").

This module is the brain:

  1. ``ingest_real_ctr``   — fills TrainingSample.ctr/.impressions from the
     WORKING YouTube Reporting-API client (insights.reporting_api — real
     thumbnail impressions + CTR), replacing the silently-dead legacy
     analytics/ctr.py numbers.
  2. ``compute_channel_learning`` — deterministic aggregation over the
     channel's published history: which HOOK TYPE, TITLE SCRIPT and TITLE
     LENGTH actually earn views/hour and CTR on THIS channel, plus the
     keywords that ride its winners. No LLM — pure measured stats.
  3. ``snapshot`` — persists the aggregation per (channel, window) into
     SeoLearningSnapshot so the UI can graph the learning curve and the
     generator can read a stable policy.
  4. ``latest_policy`` — the learned policy the SEO prompts consume
     (seo/prompts.compose_seo ``learned=`` block): this is the wire that
     makes generation IMPROVE over time instead of staying static.

Everything is fail-soft: no data → empty payloads and None policies; the
SEO engine then behaves exactly as before learning existed.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

_NATIVE_RE = re.compile(r"[ऀ-෿]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_NUM_RE = re.compile(r"\d")

# Minimum samples in a bucket before we trust it enough to steer output.
MIN_BUCKET_N = 3
# Minimum total samples before a policy is emitted at all.
MIN_POLICY_N = 5

# Brand/beat tokens carry ZERO discovery value (they appear in nearly every
# title of the channel, so they dominate any frequency mining — the "kaizer,
# news, telugu on every card" problem). Filtered from keyword/topic mining;
# the channel's own name tokens are added at runtime.
_GENERIC_STOP = {
    # brand/beat
    "kaizer", "news", "telugu", "shorts", "short", "video", "videos",
    "live", "today", "latest", "update", "updates", "official", "channel",
    # hook words — they're HOOK style, not topics
    "breaking", "shocking", "viral", "exclusive", "big", "must", "watch",
    # glue / auxiliaries
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "with",
    "vs", "by", "at", "is", "are", "his", "her", "this", "that", "new",
    "will", "would", "can", "could", "should", "has", "have", "had",
    "was", "were", "been", "from", "into", "over", "after", "before",
    "its", "out", "off", "all", "you", "your", "our", "they", "them",
    "when", "what", "who", "how", "why", "says", "said", "not", "now",
}
_WORD_RE = re.compile(r"[A-Za-z]{2,}|[ऀ-෿]{2,}")


def _title_terms(title: str, brand_tokens: set) -> list[str]:
    """Searchable terms from a title: unigrams + bigrams, lowercased,
    brand/stop-filtered. Native-script runs count as words. Brand tokens
    are also rejected as SUBSTRINGS (>=5 chars) so concatenated forms
    like "kaizernews" can't sneak back in as 'topics'."""
    def _brandish(w: str) -> bool:
        if w in brand_tokens:
            return True
        return any(len(b) >= 5 and b in w for b in brand_tokens)

    words = [w.lower() for w in _WORD_RE.findall(title or "")]
    words = [w for w in words if w not in _GENERIC_STOP
             and not _brandish(w) and len(w) >= 3]
    terms = list(words)
    for i in range(len(words) - 1):
        terms.append(words[i] + " " + words[i + 1])
    return terms


def pick_hook(policy: Optional[dict], *, explore_rate: float = 0.2,
              rng=None) -> tuple[Optional[str], bool]:
    """Exploit the channel's best hook form most of the time; EXPLORE a
    different form at ``explore_rate`` so the policy can never freeze on a
    local optimum — each exploration is published and therefore measured,
    feeding back into the very stats that decide future policy.
    Returns (hook_form|None, explored)."""
    if not policy or not (policy.get("best_hooks") or []):
        return None, False
    import random as _random
    r = rng if rng is not None else _random
    best = list(policy["best_hooks"])
    if r.random() < max(0.0, min(0.5, explore_rate)):
        others = [h for h in ("question", "number", "quote", "power", "plain")
                  if h not in best]
        if others:
            return r.choice(others), True
    return best[0], False


def is_duplicate_title(title: str, recent_titles) -> bool:
    """True when ``title`` is the same as (or near-identical to) a recent
    one — the guard against publishing two videos with identical titles
    (it happened: the Vikarabad pair). Cheap + deterministic: normalized
    exact match or >=0.92 SequenceMatcher ratio."""
    from difflib import SequenceMatcher
    t = re.sub(r"\s+", " ", (title or "").strip().lower())
    if not t:
        return False
    for r0 in (recent_titles or []):
        r1 = re.sub(r"\s+", " ", str(r0 or "").strip().lower())
        if not r1:
            continue
        if t == r1 or SequenceMatcher(None, t, r1).ratio() >= 0.92:
            return True
    return False


# ── Title classifiers (deterministic — analytics must be cheap) ─────

def classify_script(title: str) -> str:
    t = title or ""
    latin = len(_LATIN_RE.findall(t)) >= 3
    native = len(_NATIVE_RE.findall(t)) >= 3
    if latin and native:
        return "mixed"
    if native:
        return "native"
    if latin:
        return "english"
    return "none"


def classify_hook(title: str) -> str:
    """Coarse hook-type buckets a human editor would recognize."""
    t = (title or "").strip()
    if not t:
        return "plain"
    if "?" in t:
        return "question"
    if _NUM_RE.search(t[:24]):
        return "number"
    if t.startswith(("“", '"', "'")) or t.count('"') >= 2:
        return "quote"
    try:
        from seo.verifier import _POWER_WORDS
        low = t.lower()
        if any(w in low for w in _POWER_WORDS):
            return "power"
    except Exception:
        pass
    return "plain"


def classify_len_band(title: str) -> str:
    n = len(title or "")
    if n < 50:
        return "short"
    if n <= 80:
        return "sweet"
    return "long"


# ── Real CTR ingestion (Reporting API v1 → TrainingSample) ──────────

def ingest_real_ctr(db, channel_id: int) -> int:
    """Fill ctr + impressions on this channel's TrainingSample rows from
    the YouTube Reporting API (real thumbnail data). Returns rows updated.
    Fail-soft: 0 on missing scope/creds/reports."""
    try:
        import models
        from insights.reporting_api import fetch_thumbnail_ctr
        m = fetch_thumbnail_ctr(db, int(channel_id))
        if not m:
            return 0
        rows = (db.query(models.TrainingSample)
                .filter(models.TrainingSample.channel_id == int(channel_id))
                .filter(models.TrainingSample.video_id.in_(list(m.keys())))
                .all())
        n = 0
        for r in rows:
            d = m.get(r.video_id) or {}
            ctr = d.get("ctr")
            imp = d.get("impressions")
            changed = False
            if ctr is not None and ctr != r.ctr:
                r.ctr = float(ctr)
                changed = True
            if imp is not None and getattr(r, "impressions", None) != int(imp):
                r.impressions = int(imp)
                changed = True
            n += 1 if changed else 0
        if n:
            db.commit()
        return n
    except Exception as exc:
        print(f"[seo-learning] real-CTR ingest soft-fail ch={channel_id}: "
              f"{str(exc)[:120]}", flush=True)
        try:
            db.rollback()
        except Exception:
            pass
        return 0


# ── Channel-history rows (operator directive: learn from the channel's
#    FULL YouTube history, not just Kaizer-published videos) ──────────

def _history_rows(db, channel_id: int, since) -> list:
    """The channel's cached YouTube catalogue (channel_videos — EVERY upload,
    383 videos on a real channel vs a handful of Kaizer publishes) as
    learning rows: title + lifetime views/hour. No CTR/keywords here — those
    ride the TrainingSample rows, which win on merge."""
    from types import SimpleNamespace
    import models
    try:
        ch = (db.query(models.Channel)
              .filter(models.Channel.id == int(channel_id)).first())
    except Exception:
        return []
    if ch is None:
        return []
    tok = (db.query(models.OAuthToken)
           .filter(models.OAuthToken.channel_id == ch.id).first())
    gcid = (getattr(tok, "google_channel_id", "") or "") if tok else ""
    if not gcid:
        return []
    return rows_for_gcid(db, gcid, since)


def rows_for_gcid(db, gcid: str, since) -> list:
    """Catalogue learning-rows for ANY YouTube channel id — the user's own
    channels and tracked competitors share this path (public stats either
    way). No user filter: the fullest catalogue may live under another
    user's sync (the 200-of-2803 bug); dedupe keeps one row per video."""
    from types import SimpleNamespace
    import models
    vids = (db.query(models.ChannelVideo)
            .filter(models.ChannelVideo.google_channel_id == gcid)
            .filter(models.ChannelVideo.published_at.isnot(None))
            .filter(models.ChannelVideo.published_at >= since)
            .all())
    _best: dict = {}
    for v in vids:
        prev = _best.get(v.video_id)
        if prev is None or (v.view_count or 0) > (prev.view_count or 0):
            _best[v.video_id] = v
    vids = list(_best.values())
    now = datetime.now(timezone.utc)
    out = []
    for v in vids:
        pub = v.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        age_h = max(1.0, (now - pub).total_seconds() / 3600.0)
        out.append(SimpleNamespace(
            video_id=v.video_id,
            seo_title=v.title or "",
            seo_keywords=[],
            tags=(v.tags or None),           # public tags (competitor harvest)
            # COMPARABILITY: divide by age CAPPED at 30 days. Lifetime
            # vph decays with age, while TrainingSample vph is measured
            # hours after publish — mixing raw values biased every stat
            # toward recent videos. The 720h cap makes both sources an
            # approximate "first-month velocity" (aggregation applies the
            # same cap to samples).
            views_per_hour=float(v.view_count or 0) / min(age_h, 720.0),
            _views=int(v.view_count or 0), _age_h=age_h,
            ctr=None, impressions=None,
            first_seen_at=pub))
    return out


def _row_vph(r) -> float:
    """Uniform first-month-velocity for ANY row source: views / age capped
    at 720h when views+age are known; the stored views_per_hour otherwise."""
    try:
        views = getattr(r, "_views", None)
        if views is None:
            views = getattr(r, "views", None)
        age = getattr(r, "_age_h", None)
        if age is None:
            age = getattr(r, "hours_since_publish", None)
        if views is not None and age:
            return max(0.0, float(views) / max(1.0, min(float(age), 720.0)))
    except (TypeError, ValueError):
        pass
    return max(0.0, float(getattr(r, "views_per_hour", 0.0) or 0.0))


def _insights_rows(db, channel_id: int, since) -> list:
    """FULL-catalogue rows from the channel's latest Channel Doctor ingest
    (insights VideoMetric) — the source of truth: ALL videos, real YouTube data
    (incl. real CTR when present). Same row shape as _history_rows. Returns []
    when no analysis exists yet, so merged_rows can fall back."""
    try:
        import insights.models as _im
        from types import SimpleNamespace
    except Exception:
        return []
    try:
        snap = (db.query(_im.ChannelSnapshot)
                .filter(_im.ChannelSnapshot.channel_id == int(channel_id),
                        _im.ChannelSnapshot.status.in_(("ok", "partial")))
                .order_by(_im.ChannelSnapshot.created_at.desc()).first())
        if snap is None:
            import models as _m
            ch = db.query(_m.Channel).filter(_m.Channel.id == int(channel_id)).first()
            gcid = ((getattr(getattr(ch, "oauth_token", None), "google_channel_id", "") or "")
                    if ch else "")
            if gcid:
                snap = (db.query(_im.ChannelSnapshot)
                        .filter(_im.ChannelSnapshot.google_channel_id == gcid,
                                _im.ChannelSnapshot.status.in_(("ok", "partial")))
                        .order_by(_im.ChannelSnapshot.created_at.desc()).first())
        if snap is None:
            return []
        vids = db.query(_im.VideoMetric).filter(_im.VideoMetric.snapshot_id == snap.id).all()
        now = datetime.now(timezone.utc)
        out = []
        for v in vids:
            pub = v.published_at_utc or v.published_at_local
            if pub is None:
                continue
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if since is not None and pub < since:
                continue
            age_h = max(1.0, (now - pub).total_seconds() / 3600.0)
            out.append(SimpleNamespace(
                video_id=v.video_id,
                seo_title=v.title or "",
                seo_keywords=[],
                tags=(v.tags or None),
                views_per_hour=float(v.view_count or 0) / min(age_h, 720.0),
                _views=int(v.view_count or 0), _age_h=age_h,
                ctr=v.impressions_ctr, impressions=v.impressions,
                first_seen_at=pub))
        return out
    except Exception:
        return []


def merged_rows(db, channel_id: int, since) -> list:
    """The channel's FULL catalogue, deduped by video_id, TrainingSamples
    overlaid last (they carry Kaizer's real CTR/keywords/A-B for published
    videos). Source of truth = Channel Doctor's complete ingest (all videos,
    real YouTube data); falls back to the ChannelVideo catalogue sync when no
    Channel Doctor analysis exists yet."""
    import models
    base = _insights_rows(db, channel_id, since)
    if not base:
        base = _history_rows(db, channel_id, since)      # fallback (capped sync)
    by_vid = {r.video_id: r for r in base}
    samples = (db.query(models.TrainingSample)
               .filter(models.TrainingSample.channel_id == int(channel_id))
               .filter(models.TrainingSample.first_seen_at >= since)
               .all())
    for s in samples:
        by_vid[s.video_id] = s
    return list(by_vid.values())


# ── Aggregation (the actual learning) ────────────────────────────────

def _bucket_stats(rows) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0, "avg_vph": 0.0, "avg_ctr": None, "ctr_n": 0}
    vph = [_row_vph(r) for r in rows]
    ctrs = [float(r.ctr) for r in rows if getattr(r, "ctr", None) is not None]
    return {
        "n": n,
        "avg_vph": round(sum(vph) / n, 2),
        "avg_ctr": (round(sum(ctrs) / len(ctrs), 3) if ctrs else None),
        "ctr_n": len(ctrs),
    }


def _best_bucket(stats: dict) -> Optional[str]:
    """Best bucket with enough samples. CTR-WEIGHTED once real CTR exists:
    when every qualifying bucket has >=3 CTR-measured videos, rank by
    click-truth (avg_ctr) with velocity as tiebreak; until then velocity
    leads — the honest proxy, never silently mixed."""
    good = {k: v for k, v in stats.items() if v["n"] >= MIN_BUCKET_N}
    if not good:
        return None
    if all(v.get("ctr_n", 0) >= 3 for v in good.values()):
        return max(good, key=lambda k: ((good[k]["avg_ctr"] or 0.0),
                                        good[k]["avg_vph"]))
    return max(good, key=lambda k: (good[k]["avg_vph"],
                                    good[k]["avg_ctr"] or 0.0))


def compute_channel_learning(db, channel_id: int,
                             window_days: int = 30) -> dict:
    """Aggregate this channel's real performance into learned stats.

    Rows = the channel's FULL YouTube catalogue (real titles, real view
    counts, real ages — 1000+ videos on a mature channel) merged with
    Kaizer's TrainingSamples (which add real CTR + keyword data). The
    catalogue merge is fail-soft: if its tables are unavailable the
    aggregation still runs on samples alone."""
    import models
    since = datetime.now(timezone.utc) - timedelta(days=int(window_days))
    try:
        rows = merged_rows(db, channel_id, since)
    except Exception:
        rows = (db.query(models.TrainingSample)
                .filter(models.TrainingSample.channel_id == int(channel_id))
                .filter(models.TrainingSample.first_seen_at >= since)
                .all())
    # Channel brand tokens join the stoplist (the channel's own name words
    # appear in most of its titles — mining them is the "brand echo" bug).
    brand_tokens: set = set()
    try:
        _chn = (db.query(models.Channel)
                .filter(models.Channel.id == int(channel_id)).first())
        if _chn is not None:
            brand_tokens = {w.lower() for w in
                            _WORD_RE.findall(_chn.name or "") if len(w) >= 2}
    except Exception:
        pass

    by_hook, by_script, by_len = (defaultdict(list), defaultdict(list),
                                  defaultdict(list))
    by_hour, by_dow = defaultdict(list), defaultdict(list)
    by_dow_hour: dict = defaultdict(list)   # (weekday, hour) → rows, IST
    kw_perf: dict[str, list] = defaultdict(list)
    topic_perf: dict[str, list] = defaultdict(list)
    tz_min = 330  # IST display buckets for upload-time learning
    for r in rows:
        t = r.seo_title or ""
        vph = _row_vph(r)                    # uniform first-month velocity
        by_hook[classify_hook(t)].append(r)
        by_script[classify_script(t)].append(r)
        by_len[classify_len_band(t)].append(r)
        # Upload-time buckets (local hour + weekday) from the publish stamp.
        ts = getattr(r, "first_seen_at", None)
        if ts is not None:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            loc = ts + timedelta(minutes=tz_min)
            by_hour[loc.hour].append(r)
            by_dow[loc.strftime("%a")].append(r)
            by_dow_hour[(loc.strftime("%a"), loc.hour)].append(r)
        # Keywords: REAL tags from samples + brand-filtered title terms from
        # the whole catalogue, all weighted by the video's measured vph.
        seen_terms = set()
        for kw in (r.seo_keywords or [])[:30]:
            k = str(kw).strip().lower()
            if 2 < len(k) <= 60 and k not in _GENERIC_STOP \
                    and k not in brand_tokens:
                kw_perf[k].append(vph)
                seen_terms.add(k)
        for term in _title_terms(t, brand_tokens):
            if term not in seen_terms:
                kw_perf[term].append(vph)
            if " " in term:                     # multi-word = topic-like
                topic_perf[term].append(vph)

    hook_stats = {k: _bucket_stats(v) for k, v in by_hook.items()}
    script_stats = {k: _bucket_stats(v) for k, v in by_script.items()}
    len_stats = {k: _bucket_stats(v) for k, v in by_len.items()}
    hour_stats = {str(k): _bucket_stats(v) for k, v in by_hour.items()}
    dow_stats = {k: _bucket_stats(v) for k, v in by_dow.items()}
    # Day×hour cross-tab (nested {weekday: {hour: stats}}) — the detailed
    # heatmap + per-day drill-down. Only non-empty cells stored; the report
    # layer fills the full 7×24 grid.
    dow_hour_stats: dict = {}
    for (dw, hr), rws in by_dow_hour.items():
        dow_hour_stats.setdefault(dw, {})[str(hr)] = _bucket_stats(rws)
    # Ranked by mean vph of the videos carrying the term (n>=2 so a single
    # viral outlier can't crown a keyword).
    top_kw = sorted(
        ((k, round(sum(v) / len(v), 2), len(v)) for k, v in kw_perf.items()
         if len(v) >= 2),
        key=lambda x: -x[1])[:12]
    top_topics = sorted(
        ((k, round(sum(v) / len(v), 2), len(v)) for k, v in topic_perf.items()
         if len(v) >= 2),
        key=lambda x: -x[1])[:8]
    best_hours = sorted(
        ((h, s) for h, s in hour_stats.items() if s["n"] >= MIN_BUCKET_N),
        key=lambda x: -x[1]["avg_vph"])[:2]

    n_total = len(rows)
    ctr_cov = sum(1 for r in rows if r.ctr is not None)
    policy = None
    if n_total >= MIN_POLICY_N:
        ranked_hooks = sorted(
            (k for k, v in hook_stats.items() if v["n"] >= MIN_BUCKET_N),
            key=lambda k: -(hook_stats[k]["avg_vph"]))
        policy = {
            "best_hooks": ranked_hooks[:2] or None,
            "best_script": _best_bucket(script_stats),
            "best_len_band": _best_bucket(len_stats),
            "top_keywords": [k for k, _, _ in top_kw[:8]],
            "top_topics": [k for k, _, _ in top_topics[:5]],
            "best_hours": [int(h) for h, _ in best_hours],
            "based_on": n_total,
            "ctr_coverage": ctr_cov,
        }
    # EXPLORATION LEDGER: every A/B probe (explored_hook stamped at
    # generation) vs the exploit baseline — measured, per hook form, so
    # the UI can show "12 experiments: question 8.2 vph vs usual 6.9"
    # and nobody has to take the A/B system on faith.
    _exp_rows = [r for r in rows if getattr(r, "explored_hook", None)]
    _base_rows = [r for r in rows if not getattr(r, "explored_hook", None)]
    explorations = {
        "n": len(_exp_rows),
        "baseline_avg_vph": _bucket_stats(_base_rows)["avg_vph"],
        "by_hook": {k: _bucket_stats(v) for k, v in
                    _group_by(_exp_rows, lambda r: r.explored_hook).items()},
    }

    return {
        "channel_id": int(channel_id),
        "window_days": int(window_days),
        "samples": n_total,
        "ctr_coverage": ctr_cov,
        "explorations": explorations,
        "by_hook": hook_stats,
        "by_script": script_stats,
        "by_len": len_stats,
        "by_hour": hour_stats,
        "by_dow": dow_stats,
        "by_dow_hour": dow_hour_stats,
        "top_keywords": [
            {"keyword": k, "avg_vph": v, "n": c} for k, v, c in top_kw],
        "top_topics": [
            {"topic": k, "avg_vph": v, "n": c} for k, v, c in top_topics],
        "policy": policy,
    }


# Weekday buckets are stored as C-locale %a ("Mon".."Sun"); this fixes the
# report ordering (Mon-first) and the human labels for the UI.
_DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_DOW_FULL = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday",
             "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday",
             "Sun": "Sunday"}


def _insights_timing_agg(db, channel_id: int):
    """Build an agg dict (by_hour / by_dow / by_dow_hour) from the channel's
    latest CHANNEL DOCTOR analysis, so the SEO-Settings best-time view shows the
    SAME numbers as Channel Doctor — the source of truth (more videos, the
    channel's REAL local time, launch-speed from real YouTube data). Returns
    (agg, tz_label) or (None, None) when no analysis exists yet (→ fallback)."""
    try:
        import models as _m
        import insights.models as _im
    except Exception:
        return None, None
    try:
        snap = (db.query(_im.ChannelSnapshot)
                .filter(_im.ChannelSnapshot.channel_id == int(channel_id),
                        _im.ChannelSnapshot.status.in_(("ok", "partial")))
                .order_by(_im.ChannelSnapshot.created_at.desc()).first())
        if snap is None:
            ch = db.query(_m.Channel).filter(_m.Channel.id == int(channel_id)).first()
            gcid = ((getattr(getattr(ch, "oauth_token", None), "google_channel_id", "") or "")
                    if ch else "")
            if gcid:
                snap = (db.query(_im.ChannelSnapshot)
                        .filter(_im.ChannelSnapshot.google_channel_id == gcid,
                                _im.ChannelSnapshot.status.in_(("ok", "partial")))
                        .order_by(_im.ChannelSnapshot.created_at.desc()).first())
        if snap is None:
            return None, None
        run = (db.query(_im.AnalysisRun)
               .filter(_im.AnalysisRun.snapshot_id == snap.id,
                       _im.AnalysisRun.status == "ok")
               .order_by(_im.AnalysisRun.created_at.desc()).first())
        heat = ((run.results or {}).get("timing") or {}).get("heatmap") if run else None
        if not heat:
            return None, None
        dow_hour: dict = {}
        hour_acc: dict = {}   # hour -> [weighted_sum, n]
        dow_acc: dict = {}    # dow_label -> [weighted_sum, n]
        for c in heat:
            d, h = c.get("dow"), c.get("hour")
            if d is None or h is None or not (0 <= int(d) < 7):
                continue
            dl = _DOW_ORDER[int(d)]
            sc = float(c.get("score") or 0.0)
            n = int(c.get("n") or 0)
            dow_hour.setdefault(dl, {})[str(int(h))] = {
                "n": n, "avg_vph": round(sc, 2), "avg_ctr": None, "ctr_n": 0}
            ha = hour_acc.setdefault(int(h), [0.0, 0]); ha[0] += sc * n; ha[1] += n
            da = dow_acc.setdefault(dl, [0.0, 0]); da[0] += sc * n; da[1] += n
        by_hour = {str(h): {"n": a[1], "avg_ctr": None, "ctr_n": 0,
                            "avg_vph": round(a[0] / a[1], 2) if a[1] else 0.0}
                   for h, a in hour_acc.items()}
        by_dow = {d: {"n": a[1], "avg_ctr": None, "ctr_n": 0,
                      "avg_vph": round(a[0] / a[1], 2) if a[1] else 0.0}
                  for d, a in dow_acc.items()}
        samples = int(snap.videos_ingested or snap.video_count or 0)
        tz = (snap.timezone or "").strip()
        tz_label = "IST" if tz.replace("Asia/", "") in ("Kolkata", "Calcutta") else "local time"
        agg = {"by_hour": by_hour, "by_dow": by_dow, "by_dow_hour": dow_hour,
               "samples": samples, "ctr_coverage": 0}
        return agg, tz_label
    except Exception:
        return None, None


def best_times_report(db, channel_id: int, window_days: int = 3650) -> dict:
    """The human-facing "best time to post" report — the full 24-hour + 7-weekday
    breakdowns (sample count + avg views/hour per bucket) + a recommendation.

    SOURCE OF TRUTH = Channel Doctor's measured timing (real YouTube data, the
    channel's real local time, launch speed) so this AGREES with the Channel
    Doctor view. FALLBACK = the SEO-learning history (IST) when Channel Doctor
    hasn't analyzed this channel yet. Buckets below MIN_BUCKET_N are shown but
    never crowned; honest empty-state when history is too thin.
    """
    import models
    # 1) Prefer Channel Doctor (source of truth) so both views agree.
    agg, tz_label = _insights_timing_agg(db, channel_id)
    source = "channel_doctor"
    # 2) Fallback: the SEO-learning snapshot / live compute (IST).
    if agg is None:
        source, tz_label = "learning", "IST"
        try:
            row = (db.query(models.SeoLearningSnapshot)
                   .filter(models.SeoLearningSnapshot.channel_id == int(channel_id),
                           models.SeoLearningSnapshot.kind == "own",
                           models.SeoLearningSnapshot.window_days == int(window_days))
                   .order_by(models.SeoLearningSnapshot.computed_at.desc())
                   .first())
            if row and row.payload and (row.payload.get("by_hour") is not None):
                agg = row.payload
        except Exception:
            agg = None
        if agg is None or agg.get("by_dow_hour") is None:
            agg = compute_channel_learning(db, channel_id, window_days)

    by_hour = agg.get("by_hour") or {}
    by_dow = agg.get("by_dow") or {}
    samples = int(agg.get("samples", 0) or 0)
    ctr_cov = int(agg.get("ctr_coverage", 0) or 0)

    def _cell(s):
        s = s or {}
        return {"n": int(s.get("n", 0) or 0),
                "avg_vph": float(s.get("avg_vph", 0.0) or 0.0),
                "avg_ctr": s.get("avg_ctr"),
                "ctr_n": int(s.get("ctr_n", 0) or 0)}

    # Full 24-hour array (every hour present so the chart shows real gaps).
    hours = [{"hour": h, **_cell(by_hour.get(str(h)))} for h in range(24)]
    days = [{"dow": d, "label": _DOW_FULL[d], **_cell(by_dow.get(d))}
            for d in _DOW_ORDER]

    qual_hours = {h["hour"]: h for h in hours if h["n"] >= MIN_BUCKET_N}
    qual_days = {d["dow"]: d for d in days if d["n"] >= MIN_BUCKET_N}

    # CTR-rank an axis ONLY when every one of ITS qualifying buckets has real
    # CTR (>=3 measured) — mirrors _best_bucket's honesty rule per-axis, so
    # the weekday axis never gets crowned by CTR the day buckets don't have.
    def _ctr_complete(qual):
        return bool(qual) and all(v["ctr_n"] >= 3 for v in qual.values())

    hours_ctr = _ctr_complete(qual_hours)
    days_ctr = _ctr_complete(qual_days)
    signal = "ctr" if hours_ctr else "views"  # headline signal = the hour axis

    def _rank(bucket_map, use_ctr):
        if use_ctr:
            return sorted(bucket_map,
                          key=lambda k: ((bucket_map[k]["avg_ctr"] or 0.0),
                                         bucket_map[k]["avg_vph"]),
                          reverse=True)
        return sorted(bucket_map,
                      key=lambda k: (bucket_map[k]["avg_vph"],
                                     bucket_map[k]["avg_ctr"] or 0.0),
                      reverse=True)

    best_hours = [int(h) for h in _rank(qual_hours, hours_ctr)[:3]]
    best_days = [_DOW_FULL[d] for d in _rank(qual_days, days_ctr)[:2]]
    enough = bool(qual_hours)

    # ── Day×hour detail: the full 7×24 grid + the single peak slot ──────
    dow_hour = agg.get("by_dow_hour") or {}
    grid = []
    peak = None
    for d in _DOW_ORDER:
        cells = dow_hour.get(d) or {}
        row_hours = []
        for h in range(24):
            c = _cell(cells.get(str(h)))
            c["hour"] = h
            row_hours.append(c)
            # Peak = strongest cell with enough samples AND real signal
            # (avg_vph, CTR tiebreak) — a dead 0-view slot is never crowned.
            if c["n"] >= MIN_BUCKET_N and (c["avg_vph"] > 0 or (c["avg_ctr"] or 0) > 0):
                key = (c["avg_vph"], c["avg_ctr"] or 0.0)
                if peak is None or key > (peak["avg_vph"], peak["avg_ctr"] or 0.0):
                    peak = {"dow": d, "label": _DOW_FULL[d], "hour": h,
                            "avg_vph": c["avg_vph"], "avg_ctr": c["avg_ctr"],
                            "n": c["n"]}
        grid.append({"dow": d, "label": _DOW_FULL[d], "hours": row_hours})

    if not enough:
        reason = (f"Not enough upload history yet — an hour needs at least "
                  f"{MIN_BUCKET_N} published videos before it can be trusted. "
                  f"Keep publishing and press “Learn now”.")
    elif signal == "ctr":
        reason = ("Ranked by real click-through rate (thumbnail CTR from "
                  "YouTube), with views/hour as the tiebreak.")
    else:
        reason = ("Ranked by average views per hour — the honest proxy until "
                  "enough real CTR data accrues on these time slots.")

    return {
        "channel_id": int(channel_id),
        "window_days": int(window_days),
        "timezone": tz_label,
        "source": source,     # "channel_doctor" (source of truth) | "learning" (fallback)
        "samples": samples,
        "ctr_coverage": ctr_cov,
        "signal": signal,
        "enough_data": enough,
        "hours": hours,
        "days": days,
        "grid": grid,          # 7×24 [{dow,label,hours:[{hour,n,avg_vph,avg_ctr,ctr_n}]}]
        "peak": peak,          # single strongest day×hour slot (or None)
        "recommendation": {
            "best_hours": best_hours,
            "best_days": best_days,
            "reason": reason,
        },
    }


def snapshot(db, channel_id: int, windows=(7, 30, 90, 3650)) -> int:
    """Compute + persist one SeoLearningSnapshot per window. Returns count."""
    import models
    n = 0
    for w in windows:
        try:
            payload = compute_channel_learning(db, channel_id, w)
            db.add(models.SeoLearningSnapshot(
                channel_id=int(channel_id), kind="own", window_days=int(w),
                samples=payload["samples"], payload=payload))
            n += 1
            # PRUNE: append-only would grow forever (4 rows per learn).
            # Keep the newest 30 per (channel, window) — months of curve
            # history, bounded storage.
            _ids = [r.id for r in
                    (db.query(models.SeoLearningSnapshot.id)
                     .filter(models.SeoLearningSnapshot.channel_id == int(channel_id),
                             models.SeoLearningSnapshot.kind == "own",
                             models.SeoLearningSnapshot.window_days == int(w))
                     .order_by(models.SeoLearningSnapshot.computed_at.desc())
                     .offset(30).all())]
            if _ids:
                (db.query(models.SeoLearningSnapshot)
                 .filter(models.SeoLearningSnapshot.id.in_(_ids))
                 .delete(synchronize_session=False))
        except Exception as exc:
            print(f"[seo-learning] snapshot soft-fail ch={channel_id} "
                  f"w={w}: {str(exc)[:120]}", flush=True)
    try:
        db.commit()
    except Exception:
        db.rollback()
    return n


def latest_policy(db, channel_id: int) -> Optional[dict]:
    """Freshest confident policy for a channel: prefer recent windows
    (30d, 90d), fall back to the ALL-TIME window (3650d — the channel's
    full real history, e.g. 1000+ catalogued videos), then the week.
    None = genuinely not enough data — generate as before."""
    import models
    for w in (30, 90, 3650, 7):
        row = (db.query(models.SeoLearningSnapshot)
               .filter(models.SeoLearningSnapshot.channel_id == int(channel_id))
               .filter(models.SeoLearningSnapshot.kind == "own")
               .filter(models.SeoLearningSnapshot.window_days == int(w))
               .order_by(models.SeoLearningSnapshot.computed_at.desc())
               .first())
        if row and (row.payload or {}).get("policy"):
            return row.payload["policy"]
    return None


def resolve_script_policy(db, channel_id: Optional[int] = None) -> str:
    """The title-script policy the deterministic verifier should score
    against — the SAME resolution seo/generator.py uses at generation time,
    so manual re-scores and Live Studio score titles consistently with how
    they were written. Precedence:
      1. env KAIZER_SEO_SCRIPT_POLICY = bilingual|english|native  (hard force)
      2. the channel's LEARNED best_script when confident (mixed→bilingual)
      3. 'bilingual' (operator's 2026-08 default)
    channel_id=None (e.g. a generic, channel-less clip) → env force or default.
    """
    import os
    sp_env = (os.environ.get("KAIZER_SEO_SCRIPT_POLICY", "learned")
              or "learned").strip().lower()
    if sp_env in ("bilingual", "english", "native"):
        return sp_env
    best = None
    if channel_id:
        try:
            best = (latest_policy(db, int(channel_id)) or {}).get("best_script")
        except Exception:
            best = None
    return {"mixed": "bilingual", "english": "english",
            "native": "native"}.get(best, "bilingual")


def _group_by(rows, keyfn) -> dict:
    out: dict = defaultdict(list)
    for r in rows:
        out[keyfn(r)].append(r)
    return dict(out)


def learn_channel(db, channel_id: int) -> dict:
    """One-call refresh: ingest real CTR, then snapshot all windows."""
    updated = ingest_real_ctr(db, channel_id)
    snaps = snapshot(db, channel_id)
    return {"ctr_rows_updated": updated, "snapshots": snaps}


def score_reality_audit(db, channel_id: int, window_days: int = 90) -> dict:
    """Does a high seo_score actually EARN more? The scorer's own honesty
    check: compares measured velocity of high-scored vs low-scored
    published videos. If high scorers don't win, the rubric is lying and
    must be revised — this is the anti-self-deception instrument."""
    import models
    since = datetime.now(timezone.utc) - timedelta(days=int(window_days))
    rows = (db.query(models.TrainingSample)
            .filter(models.TrainingSample.channel_id == int(channel_id))
            .filter(models.TrainingSample.first_seen_at >= since)
            .filter(models.TrainingSample.seo_score > 0)
            .all())
    hi = [r for r in rows if (r.seo_score or 0) >= 90]
    lo = [r for r in rows if 0 < (r.seo_score or 0) < 90]
    return {
        "window_days": int(window_days),
        "scored_videos": len(rows),
        "high": _bucket_stats(hi),
        "low": _bucket_stats(lo),
        "verdict": (None if len(hi) < MIN_BUCKET_N or len(lo) < MIN_BUCKET_N
                    else ("score_predicts" if _bucket_stats(hi)["avg_vph"]
                          > _bucket_stats(lo)["avg_vph"] else
                          "score_does_NOT_predict")),
    }


def uplift_report(db, channel_id: int) -> dict:
    """Recent 7 days vs the prior 30: the measured before/after that any
    'the SEO got better' claim must survive."""
    import models
    now = datetime.now(timezone.utc)
    recent = (db.query(models.TrainingSample)
              .filter(models.TrainingSample.channel_id == int(channel_id))
              .filter(models.TrainingSample.first_seen_at
                      >= now - timedelta(days=7)).all())
    prior = (db.query(models.TrainingSample)
             .filter(models.TrainingSample.channel_id == int(channel_id))
             .filter(models.TrainingSample.first_seen_at
                     < now - timedelta(days=7))
             .filter(models.TrainingSample.first_seen_at
                     >= now - timedelta(days=37)).all())
    r, p = _bucket_stats(recent), _bucket_stats(prior)
    return {
        "recent_7d": r, "prior_30d": p,
        "vph_uplift_pct": (round((r["avg_vph"] - p["avg_vph"])
                                 / p["avg_vph"] * 100, 1)
                           if p["avg_vph"] > 0 and r["n"] else None),
    }


def weekly_uplift(db, channel_id: int, weeks: int = 8) -> dict:
    """Week-by-week MEASURED rollup — the multi-week view behind the single
    7-vs-30 uplift chip. For each of the last `weeks` ISO weeks: published
    count, avg views/hour, real CTR (when present), and the week-over-week %
    change vs the previous non-empty week. Empty weeks are included
    (published=0) so cadence gaps are visible. Honest 'need >=2 measured
    weeks' state until there's enough history."""
    import models
    weeks = max(2, min(int(weeks), 26))
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=weeks * 7)
    try:
        rows = merged_rows(db, channel_id, since)
    except Exception:
        rows = (db.query(models.TrainingSample)
                .filter(models.TrainingSample.channel_id == int(channel_id))
                .filter(models.TrainingSample.first_seen_at >= since).all())

    # Bucket by the Monday of each row's ISO week (UTC), matching the curves
    # series' UTC dating so the two reconcile.
    buckets: dict = defaultdict(list)
    for r in rows:
        ts = getattr(r, "first_seen_at", None)
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        monday = (ts - timedelta(days=ts.weekday())).date()
        buckets[monday.isoformat()].append(r)

    this_monday = (now - timedelta(days=now.weekday())).date()
    series = []
    for i in range(weeks - 1, -1, -1):          # oldest → newest
        wk = (this_monday - timedelta(days=7 * i)).isoformat()
        st = _bucket_stats(buckets.get(wk, []))
        series.append({"week_start": wk, "published": st["n"],
                       "avg_vph": st["avg_vph"], "avg_ctr": st["avg_ctr"]})

    # Week-over-week % change vs the previous NON-EMPTY week (skips gap weeks
    # so a cadence pause doesn't read as a crash then a spike).
    prev_vph = None
    for w in series:
        if w["published"] > 0:
            w["wow_pct"] = (round((w["avg_vph"] - prev_vph) / prev_vph * 100, 1)
                            if prev_vph and prev_vph > 0 else None)
            prev_vph = w["avg_vph"]
        else:
            w["wow_pct"] = None

    nonempty = [w for w in series if w["published"] > 0]
    return {
        "channel_id": int(channel_id),
        "weeks": weeks,
        "series": series,
        "enough_data": len(nonempty) >= 2,
        "latest_wow_pct": (nonempty[-1]["wow_pct"] if nonempty else None),
    }
