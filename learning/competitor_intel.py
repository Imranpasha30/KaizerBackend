"""Competitor intelligence — public YouTube data → per-video SEO conquest.

Tier 1 of the competitor design (operator-approved):
  * ``sync_competitor``  — deep catalogue of a tracked rival via the PUBLIC
    Data API key (their titles, views, publish times AND tags — all public;
    no permission needed). Rows land in the shared channel_videos table.
  * ``learn_competitor`` — the SAME measured math as own-channel learning
    (hooks/script/length/topics), stored as SeoLearningSnapshot
    kind='competitor' (id namespace = CompetitorChannel.id).
  * ``topic_intel``      — the per-video weapon: given THIS video's terms,
    find the rivals' best-velocity videos on the SAME topic and return
    their titles (to DIFFERENTIATE from), harvested tags (to reuse when
    relevant) and cover-terms (queries their winners carry).

Honest limits, by construction: rival CTR/impressions are private and are
NEVER faked here — all competitor stats are views-velocity. All functions
fail soft (missing API key / quota → empty results, generation proceeds
without intel).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from learning.seo_learning import (
    MIN_BUCKET_N, MIN_POLICY_N, _best_bucket, _bucket_stats, _row_vph,
    _title_terms, _WORD_RE, classify_hook, classify_len_band,
    classify_script, rows_for_gcid,
)


# Frequent Telugu/Indian news entities in BOTH scripts. topic_intel matches
# clip terms against rival titles with a SQL substring (ILIKE); a clip whose
# terms are Latin ("Pawan Kalyan") otherwise misses pure-Telugu rival titles
# (and vice-versa). Expanding each term to its known script variants closes
# that gap. RECALL-ONLY — never changes ranking or scoring; extend freely.
_TRANSLIT_MAP: dict[str, list[str]] = {
    "pawan kalyan": ["పవన్ కళ్యాణ్", "పవన్"],
    "chandrababu": ["చంద్రబాబు", "బాబు"],
    "chandrababu naidu": ["చంద్రబాబు నాయుడు", "చంద్రబాబు"],
    "jagan": ["జగన్"],
    "ys jagan": ["వైఎస్ జగన్", "జగన్"],
    "revanth reddy": ["రేవంత్ రెడ్డి", "రేవంత్"],
    "kcr": ["కేసీఆర్"],
    "ktr": ["కేటీఆర్"],
    "modi": ["మోడీ", "నరేంద్ర మోడీ"],
    "narendra modi": ["నరేంద్ర మోడీ", "మోడీ"],
    "rahul gandhi": ["రాహుల్ గాంధీ", "రాహుల్"],
    "amit shah": ["అమిత్ షా"],
    "telangana": ["తెలంగాణ", "తెలంగాణా"],
    "andhra pradesh": ["ఆంధ్రప్రదేశ్", "ఆంధ్ర", "ఏపీ"],
    "andhra": ["ఆంధ్ర", "ఆంధ్రప్రదేశ్"],
    "hyderabad": ["హైదరాబాద్"],
    "amaravati": ["అమరావతి"],
    "vijayawada": ["విజయవాడ"],
    "visakhapatnam": ["విశాఖపట్నం", "విశాఖ"],
    "tirupati": ["తిరుపతి"],
    "tdp": ["టీడీపీ", "తెలుగుదేశం"],
    "ysrcp": ["వైఎస్సార్సీపీ", "వైసీపీ"],
    "brs": ["బీఆర్ఎస్"],
    "congress": ["కాంగ్రెస్"],
    "bjp": ["బీజేపీ"],
    "janasena": ["జనసేన"],
    "cricket": ["క్రికెట్"],
    "team india": ["టీమిండియా", "భారత జట్టు"],
}


def _build_reverse_translit(fwd: dict[str, list[str]]) -> dict[str, list[str]]:
    """native-lowercase → [english + sibling natives], so native clip terms
    also expand to their English form. Built once at import."""
    rev: dict[str, list[str]] = {}
    for eng, natives in fwd.items():
        for nat in natives:
            key = nat.lower()
            variants = rev.setdefault(key, [])
            if eng not in variants:
                variants.append(eng)
            for sib in natives:
                if sib.lower() != key and sib not in variants:
                    variants.append(sib)
    return rev


_TRANSLIT_REVERSE = _build_reverse_translit(_TRANSLIT_MAP)


def _expand_terms(terms: list[str], *, cap: int = 14) -> list[str]:
    """Add known cross-script variants for each term (Latin↔Telugu). Preserves
    order, de-dupes case-insensitively, bounded so the ILIKE OR stays small."""
    out: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        tl = (term or "").strip().lower()
        if tl and tl not in seen and len(out) < cap:
            seen.add(tl)
            out.append(term)

    for t in terms or []:
        _add(t)
        tl = (t or "").strip().lower()
        for variant in _TRANSLIT_MAP.get(tl, []):
            _add(variant)
        for variant in _TRANSLIT_REVERSE.get(tl, []):
            _add(variant)
    return out


def _yt_public():
    """Public Data-API client (API key — no OAuth). None when unset."""
    try:
        from config import settings
        if not getattr(settings, "yt_data_api_key", ""):
            return None
        from googleapiclient.discovery import build
        return build("youtube", "v3", developerKey=settings.yt_data_api_key,
                     cache_discovery=False)
    except Exception:
        return None


def sync_competitor(db, comp, *, max_videos: int = 1000) -> int:
    """Deep-sync a rival's public catalogue (incl. TAGS) into
    channel_videos. Returns rows upserted; 0 on any failure."""
    import models
    yt = _yt_public()
    if yt is None or not getattr(comp, "youtube_channel_id", ""):
        return 0
    gcid = comp.youtube_channel_id
    try:
        ch = yt.channels().list(part="contentDetails", id=gcid).execute()
        items = ch.get("items") or []
        uploads = (items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
                   if items else None)
        if not uploads:
            return 0
        vids: list[str] = []
        page = None
        while len(vids) < max_videos:
            pl = yt.playlistItems().list(
                part="contentDetails", playlistId=uploads,
                maxResults=50, pageToken=page).execute()
            vids += [i["contentDetails"]["videoId"]
                     for i in pl.get("items") or []]
            page = pl.get("nextPageToken")
            if not page:
                break
        n = 0
        for i in range(0, min(len(vids), max_videos), 50):
            batch = vids[i:i + 50]
            resp = yt.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(batch), maxResults=50).execute()
            for it in resp.get("items") or []:
                sn = it.get("snippet") or {}
                st = it.get("statistics") or {}
                vid = it.get("id")
                row = (db.query(models.ChannelVideo)
                       .filter(models.ChannelVideo.user_id == comp.user_id,
                               models.ChannelVideo.google_channel_id == gcid,
                               models.ChannelVideo.video_id == vid)
                       .first())
                if row is None:
                    row = models.ChannelVideo(
                        user_id=comp.user_id, google_channel_id=gcid,
                        video_id=vid)
                    db.add(row)
                row.title = sn.get("title") or ""
                row.description_short = (sn.get("description") or "")[:500]
                pub = (sn.get("publishedAt") or "").replace("Z", "+00:00")
                try:
                    row.published_at = datetime.fromisoformat(pub)
                except ValueError:
                    pass
                row.view_count = int(st.get("viewCount") or 0)
                row.like_count = int(st.get("likeCount") or 0)
                row.comment_count = int(st.get("commentCount") or 0)
                row.tags = (sn.get("tags") or [])[:40] or None
                n += 1
            db.commit()
        return n
    except Exception as exc:
        print(f"[competitor-intel] sync soft-fail {getattr(comp,'name','?')}: "
              f"{str(exc)[:140]}", flush=True)
        try:
            db.rollback()
        except Exception:
            pass
        return 0


def learn_competitor(db, comp) -> dict:
    """Same measured math as own-channel learning, over the rival's
    catalogue; snapshot stored under kind='competitor'."""
    import models
    since = datetime.now(timezone.utc) - timedelta(days=3650)
    rows = rows_for_gcid(db, comp.youtube_channel_id, since)
    brand = {w.lower() for w in _WORD_RE.findall(comp.name or "")
             if len(w) >= 2}
    by_hook, by_script, by_len = (defaultdict(list), defaultdict(list),
                                  defaultdict(list))
    kw: dict = defaultdict(list)
    topics: dict = defaultdict(list)
    tag_pool: dict = defaultdict(list)
    for r in rows:
        t = r.seo_title or ""
        vph = _row_vph(r)
        by_hook[classify_hook(t)].append(r)
        by_script[classify_script(t)].append(r)
        by_len[classify_len_band(t)].append(r)
        for term in _title_terms(t, brand):
            kw[term].append(vph)
            if " " in term:
                topics[term].append(vph)
        for tag in (getattr(r, "tags", None) or []):
            tg = str(tag).strip().lower()
            if 2 < len(tg) <= 60:
                tag_pool[tg].append(vph)

    def _rank(d, n_min=2, top=12):
        return [{"term": k, "avg_vph": round(sum(v) / len(v), 2), "n": len(v)}
                for k, v in sorted(d.items(),
                                   key=lambda kv: -(sum(kv[1]) / len(kv[1])))
                if len(v) >= n_min][:top]

    hook_stats = {k: _bucket_stats(v) for k, v in by_hook.items()}
    payload = {
        "competitor_id": comp.id,
        "name": comp.name,
        "samples": len(rows),
        "by_hook": hook_stats,
        "by_script": {k: _bucket_stats(v) for k, v in by_script.items()},
        "by_len": {k: _bucket_stats(v) for k, v in by_len.items()},
        "top_keywords": _rank(kw),
        "top_topics": _rank(topics, top=8),
        "top_tags": _rank(tag_pool, top=15),
        "policy": None,
    }
    if len(rows) >= MIN_POLICY_N:
        ranked = sorted((k for k, v in hook_stats.items()
                         if v["n"] >= MIN_BUCKET_N),
                        key=lambda k: -(hook_stats[k]["avg_vph"]))
        payload["policy"] = {
            "best_hooks": ranked[:2] or None,
            "best_script": _best_bucket(payload["by_script"]),
            "best_len_band": _best_bucket(payload["by_len"]),
            "based_on": len(rows),
        }
    try:
        db.add(models.SeoLearningSnapshot(
            channel_id=int(comp.id), kind="competitor", window_days=3650,
            samples=len(rows), payload=payload))
        _ids = [r.id for r in
                (db.query(models.SeoLearningSnapshot.id)
                 .filter(models.SeoLearningSnapshot.channel_id == int(comp.id),
                         models.SeoLearningSnapshot.kind == "competitor")
                 .order_by(models.SeoLearningSnapshot.computed_at.desc())
                 .offset(10).all())]
        if _ids:
            (db.query(models.SeoLearningSnapshot)
             .filter(models.SeoLearningSnapshot.id.in_(_ids))
             .delete(synchronize_session=False))
        db.commit()
    except Exception:
        db.rollback()
    return payload


def latest_competitor_payload(db, comp_id: int) -> Optional[dict]:
    import models
    row = (db.query(models.SeoLearningSnapshot)
           .filter(models.SeoLearningSnapshot.channel_id == int(comp_id),
                   models.SeoLearningSnapshot.kind == "competitor")
           .order_by(models.SeoLearningSnapshot.computed_at.desc())
           .first())
    return row.payload if row else None


def topic_intel(db, user_id: int, video_terms: list[str],
                *, limit: int = 6) -> Optional[dict]:
    """THE per-video weapon: rivals' best videos matching THIS video's
    terms. None when no tracked competitor / no match — generation then
    runs without the block (never with irrelevant intel)."""
    import models
    terms = [t for t in (video_terms or []) if t and len(t) >= 3][:6]
    if not terms:
        return None
    comps = (db.query(models.CompetitorChannel)
             .filter(models.CompetitorChannel.user_id == int(user_id),
                     models.CompetitorChannel.active.is_(True)).all())
    if not comps:
        return None
    gcid_to_name = {c.youtube_channel_id: c.name for c in comps}
    from sqlalchemy import or_
    # Cross-script recall: expand each term to its known Telugu↔English variants
    # so a Latin clip term still matches native-script rival titles.
    match_terms = _expand_terms(terms)
    # Escape LIKE wildcards — clip terms are LLM free-text ("100% hike",
    # "SC/ST_reservation"); an unescaped % / _ would silently over-broaden
    # (or neutralize) the substring match.
    def _esc_like(t: str) -> str:
        return (t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_"))
    conds = [models.ChannelVideo.title.ilike(f"%{_esc_like(t)}%", escape="\\")
             for t in match_terms]
    vids = (db.query(models.ChannelVideo)
            .filter(models.ChannelVideo.google_channel_id.in_(
                list(gcid_to_name.keys())))
            .filter(models.ChannelVideo.published_at.isnot(None))
            .filter(or_(*conds))
            .order_by(models.ChannelVideo.view_count.desc())
            .limit(60).all())
    if not vids:
        return None
    now = datetime.now(timezone.utc)

    def _vph(v):
        pub = v.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        return float(v.view_count or 0) / max(
            1.0, min((now - pub).total_seconds() / 3600.0, 720.0))

    ranked = sorted(vids, key=_vph, reverse=True)[:limit]
    tagc: dict = defaultdict(float)
    termc: dict = defaultdict(int)
    vset = {t.lower() for t in match_terms}
    for v in ranked:
        for tg in (v.tags or []):
            tg = str(tg).strip().lower()
            if 2 < len(tg) <= 60:
                tagc[tg] += _vph(v)
        for w in _title_terms(v.title or "", set()):
            if w not in vset:
                termc[w] += 1
    return {
        "rivals": [{"channel": gcid_to_name.get(v.google_channel_id, "?"),
                    "title": v.title, "vph": round(_vph(v), 1)}
                   for v in ranked],
        "harvest_tags": [k for k, _ in
                         sorted(tagc.items(), key=lambda kv: -kv[1])][:12],
        "cover_terms": [k for k, c in
                        sorted(termc.items(), key=lambda kv: -kv[1])
                        if c >= 2][:8],
        "rival_titles": [v.title for v in ranked],
    }
