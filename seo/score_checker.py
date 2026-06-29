"""SEO score-checker (Kaizer-native, dependency-light, advisory).

Powers the "is this SEO any good?" gate the operator asked for. Today we
GENERATE title/description/tags with AI but never CHECK them against (a) the
video's actual content or (b) real search demand. This scores generated SEO so
weak SEO can be flagged (and later auto-regenerated).

Tools used — ALL already installed, each wrapped so a missing/failed tool never
breaks scoring (graceful degrade, the operator's fallback principle):
  - scikit-learn TF-IDF  -> keyword extraction + content-relevance (offline,
                            works on Telugu unicode tokens, no network)
  - pytrends (Google Trends) -> real search demand for the title's key terms
                            (network + rate-limited -> advisory, optional)

Score is 0-100 (HIGHER is better) to match the existing seo_score convention;
verdict thresholds mirror the editor badge: >=90 strong, >=75 acceptable,
>=60 weak, else poor. Pure scoring — callers log it / decide to regenerate.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

log = logging.getLogger("kaizer.seo.score_checker")

_WORD = re.compile(r"(?u)\b\w\w+\b")


def _verdict(score: float) -> str:
    if score >= 90:
        return "strong"
    if score >= 75:
        return "acceptable"
    if score >= 60:
        return "weak"
    return "poor"


def _content_keywords(content_text: str, top: int = 15) -> list[str]:
    """Top TF-IDF terms from the video's own content (transcript/summaries).
    Returns [] on any failure so the caller degrades gracefully."""
    text = (content_text or "").strip()
    if len(text) < 20:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Split into pseudo-docs (lines) so TF-IDF has structure to weigh.
        docs = [ln for ln in re.split(r"[\n।.!?]+", text) if len(ln.strip()) > 3]
        if len(docs) < 2:
            docs = [text]
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", max_features=400,
                              ngram_range=(1, 2), stop_words="english")
        m = vec.fit_transform(docs)
        import numpy as np
        weights = np.asarray(m.sum(axis=0)).ravel()
        terms = vec.get_feature_names_out()
        order = weights.argsort()[::-1]
        return [str(terms[i]) for i in order[:top]]
    except Exception as exc:
        log.warning("score_checker: keyword extraction failed: %s", exc)
        return []


def _relevance(seo_text: str, content_text: str) -> Optional[float]:
    """TF-IDF cosine similarity between the SEO blob and the content. 0..1.
    None if it can't be computed."""
    a, b = (seo_text or "").strip(), (content_text or "").strip()
    if len(a) < 5 or len(b) < 20:
        return None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 2),
                              stop_words="english")
        m = vec.fit_transform([a, b])
        return float(cosine_similarity(m[0], m[1])[0][0])
    except Exception as exc:
        log.warning("score_checker: relevance calc failed: %s", exc)
        return None


def _trend_demand(terms: list[str], geo: str = "IN") -> Optional[float]:
    """Mean Google-Trends interest (0..1) for the top terms. ADVISORY: network
    + rate-limited, so any failure returns None and the caller neutral-scores."""
    terms = [t for t in (terms or []) if t][:4]
    if not terms:
        return None
    try:
        from pytrends.request import TrendReq
        py = TrendReq(hl="en-IN", tz=330, timeout=(4, 8), retries=1, backoff_factor=0.2)
        py.build_payload(terms, timeframe="now 7-d", geo=geo)
        df = py.interest_over_time()
        if df is None or df.empty:
            return None
        cols = [c for c in df.columns if c != "isPartial"]
        if not cols:
            return None
        mean = float(df[cols].mean().mean())  # 0..100
        return max(0.0, min(1.0, mean / 100.0))
    except Exception as exc:
        log.info("score_checker: trends unavailable (advisory): %s", str(exc)[:120])
        return None


def score_seo(
    *,
    title: str,
    description: str = "",
    tags: Optional[list[str]] = None,
    content_text: str = "",
    language: str = "te",
    use_trends: bool = True,
    geo: str = "IN",
) -> dict[str, Any]:
    """Score generated SEO 0-100. content_text = the video's own words
    (transcript / story summaries). Returns dimensions + suggestions."""
    tags = [t for t in (tags or []) if str(t).strip()]
    title = (title or "").strip()
    seo_blob = " ".join([title, description or "", " ".join(tags)]).strip()
    dims: dict[str, Any] = {}
    sugg: list[str] = []

    # ── relevance to the actual content (30 pts) ──
    rel = _relevance(seo_blob, content_text)
    if rel is None:
        dims["relevance"] = {"points": 21.0, "max": 30, "note": "no content to compare (neutral)"}
    else:
        pts = round(min(1.0, rel / 0.30) * 30, 1)  # 0.30 cosine ~ full marks on short news text
        dims["relevance"] = {"points": pts, "max": 30,
                             "note": f"content match {rel:.2f}"}
        if pts < 18:
            sugg.append("Title/description drift from the video content — pull in the actual topic words.")

    # ── keyword coverage: do the content's top terms appear in the SEO? (25) ──
    kws = _content_keywords(content_text)
    if not kws:
        dims["keyword_coverage"] = {"points": 17.5, "max": 25, "note": "no keywords extracted (neutral)"}
    else:
        blob_l = seo_blob.lower()
        hit = sum(1 for k in kws if k.lower() in blob_l)
        cov = hit / len(kws)
        pts = round(cov * 25, 1)
        dims["keyword_coverage"] = {"points": pts, "max": 25,
                                    "note": f"{hit}/{len(kws)} top content keywords used"}
        if cov < 0.4:
            miss = [k for k in kws if k.lower() not in blob_l][:5]
            sugg.append(f"Add these content keywords to tags/description: {', '.join(miss)}")

    # ── title quality (20) ──
    n = len(title)
    tq = 0.0
    if 40 <= n <= 80:
        tq += 12
    elif 25 <= n < 40 or 80 < n <= 95:
        tq += 8
    else:
        tq += 3
        sugg.append(f"Title length {n} is off — aim for ~40-80 chars.")
    if re.search(r"[0-9]", title) or re.search(r"[?!:|]", title) or any(
            w in title.lower() for w in ("shocking", "viral", "breaking", "ఆగ్రహం", "షాకింగ్")):
        tq += 8
    else:
        tq += 3
        sugg.append("Title has no hook (number, question, or strong word) — add one.")
    dims["title_quality"] = {"points": round(tq, 1), "max": 20, "note": f"{n} chars"}

    # ── tags (15) ──
    nt = len(tags)
    if 8 <= nt <= 20:
        tp = 15
    elif 4 <= nt < 8:
        tp = 9
        sugg.append(f"Only {nt} tags — add a few more (aim 8-15).")
    elif nt > 20:
        tp = 10
    else:
        tp = 2
        sugg.append("Very few/no tags — add 8-15 relevant tags.")
    dims["tags"] = {"points": float(tp), "max": 15, "note": f"{nt} tags"}

    # ── trend demand (10, advisory) ──
    trend = _trend_demand(kws or [title], geo=geo) if use_trends else None
    if trend is None:
        dims["trend_demand"] = {"points": 7.0, "max": 10, "note": "trends unavailable (neutral)"}
    else:
        pts = round(min(1.0, trend / 0.5) * 10, 1)  # 50/100 interest = full marks
        dims["trend_demand"] = {"points": pts, "max": 10, "note": f"search interest {trend:.2f}"}
        if pts < 4:
            sugg.append("Low search demand for these terms — consider a more-searched angle/keyword.")

    score = round(sum(d["points"] for d in dims.values()), 1)
    return {
        "score": score,
        "verdict": _verdict(score),
        "dimensions": dims,
        "suggestions": sugg,
        "keywords": kws,
    }
