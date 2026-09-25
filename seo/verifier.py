"""Deterministic SEO score verifier.

Scores a generic (brand-agnostic) SEO JSON on 0–100.  Rule-based, not Gemini —
the model cannot game its own score.  Return shape gives per-dimension points
+ specific failure reasons so the generator's retry loop can feed them back
into the next Gemini prompt as explicit fixes.

Scoring budget (100 total, 20 each):
  1. Title quality                (length, power word, hook placement, dedupe)
  2. Description structure        (length, hook-in-line-1, paragraphing, bilingual)
  3. Keyword tag coverage         (28-30 count, trends match, total char cap)
  4. Hashtag structure            (10-12 count, CamelCase, topic match)
  5. Topic relevance + freshness  (entity presence, news-match, trend-match,
                                   no channel-branding leaks)

Because this is GENERIC SEO (no channel name suffix, no mandatory hashtags,
no footer), channel-specific compliance is verified separately at publish
time by the composer.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


# ── Constants ────────────────────────────────────────────────────────────────

MIN_TITLE_LEN   = 50
MAX_TITLE_LEN   = 95     # leave ~5 chars for publish-time " | Name" suffix
MIN_DESC_LEN    = 700
MAX_DESC_LEN    = 1800
TAG_COUNT_MIN   = 28
TAG_COUNT_MAX   = 30
TAG_TOTAL_CAP   = 500
HASHTAG_MIN     = 10
HASHTAG_MAX     = 12

_POWER_WORDS = {
    # English
    "shocking", "breaking", "viral", "exclusive", "revealed", "stunning",
    "huge", "massive", "urgent", "live", "biggest", "must watch", "caught",
    # Telugu
    "బిగ్", "షాకింగ్", "బ్రేకింగ్", "వైరల్", "సంచలనం", "ఎక్స్క్లూజివ్",
    # Hindi
    "बड़ी", "बड़ा", "शॉकिंग", "ब्रेकिंग", "वायरल", "सनसनी",
    # Tamil / Kannada / Malayalam / Bengali / Marathi / Gujarati sprinkles
    "அதிரடி", "ಬ್ರೇಕಿಂಗ್", "ബ്രേക്കിംഗ്", "ব্রেকিং", "सनसनाटी", "બ્રેકિંગ",
}

_HASHTAG_PATTERN     = re.compile(r"^#[A-Z][A-Za-z0-9]*$")
_NUM_HOOK_RE         = re.compile(r"\d")   # number-led hook detector (first 24 chars)
# Channel-suffix = " | Something" at the END of the title (where YouTube
# clients render the channel tag).  Mid-title "|" used as a bilingual
# separator is NOT a channel leak — that's a high-CTR pattern we want to keep.
_CHANNEL_SUFFIX_RE   = re.compile(r"\s*\|\s*[A-Za-z0-9\u0900-\u0DFF][A-Za-z0-9\s\u0900-\u0DFF]{2,}\s*$")
_FORBIDDEN_BRANDS    = [
    # Common Telugu news channel brand names that style-source could leak
    "rtv", "tv9", "etv", "ntv", "abn", "sakshi", "mahaa", "hmtv", "10tv",
    "big tv", "v6", "suman tv", "raj news",
]


# ── Dimension scorers ────────────────────────────────────────────────────────

def _score_title(title: str,
                 script_policy: str = "bilingual") -> Tuple[int, List[str]]:
    pts = 0
    fails: List[str] = []
    t = (title or "").strip()
    if not t:
        return 0, ["title is empty"]

    # 7 pts: length in sweet spot
    if MIN_TITLE_LEN <= len(t) <= MAX_TITLE_LEN:
        pts += 7
    else:
        fails.append(
            f"title length is {len(t)} chars — target 75-90. "
            f"Current: {t!r}. Rewrite to end around 85 chars."
        )

    # 4 pts: a STRONG HOOK in any recognized form — power word, question,
    # number-led, or quote-led. (Was power-word-ONLY, which (a) forced the
    # "Shocking:" sameness formula and (b) rigged exploration: a QUESTION
    # A/B probe lost these 4 pts every time, so the retry loop fought the
    # explore directive and best-candidate selection was biased against
    # the very experiments the learning depends on.)
    low = t.lower()
    _has_power = any(w in low for w in _POWER_WORDS)
    _has_question = "?" in t
    _has_number = bool(_NUM_HOOK_RE.search(t[:24]))
    _has_quote = t.startswith(("“", '"', "'")) or t.count('"') >= 2
    if _has_power or _has_question or _has_number or _has_quote:
        pts += 4
    else:
        fails.append(
            "title has NO hook — use a power word (Shocking/Breaking/"
            "సంచలనం/షాకింగ్...), a QUESTION, a NUMBER-led opening, or a "
            "QUOTE-led opening."
        )

    # 4 pts: no channel-suffix leak — generic SEO must NOT include "| Name"
    m = _CHANNEL_SUFFIX_RE.search(t)
    if m:
        leaked = m.group(0).strip()
        fails.append(
            f"title ends with a channel suffix '{leaked}'. REMOVE IT COMPLETELY. "
            f"The title MUST NOT end with '| anything' — branding is injected at "
            f"publish time, never in generation.  Rewrite your title without any "
            f"trailing '| ' at all."
        )
    else:
        pts += 4

    # 3 pts: TITLE SCRIPT vs the channel's policy. Default 'bilingual'
    # (operator mandate 2026-08); channels whose MEASURED history favors
    # one script get 'english' or 'native' here (per-channel learnable \u2014
    # operator decision after Kaizer News Telugu's 2003-video history
    # showed English-leaning titles out-earning mixed ~3.5x).
    has_latin = len(re.findall(r"[A-Za-z]", t)) >= 3
    has_native = len(re.findall(r"[\u0900-\u0DFF]", t)) >= 3  # Devanagari..Sinhala range
    sp = (script_policy or "bilingual").lower()
    if sp == "english":
        if has_latin and not has_native:
            pts += 3
        elif has_latin:
            pts += 2   # mixed is fine-ish for an english-leaning channel
        else:
            fails.append("this channel's measured policy is ENGLISH-leaning "
                         "titles \u2014 rewrite mostly in English (Latin script)")
    elif sp == "native":
        if has_native and not has_latin:
            pts += 3
        elif has_native:
            pts += 2
        else:
            fails.append("this channel's measured policy is NATIVE-script "
                         "titles \u2014 rewrite in the native script")
    else:  # bilingual (default)
        if has_latin and has_native:
            pts += 3
        elif has_native:
            pts += 1
            fails.append(
                "title is native-script only \u2014 make it BILINGUAL: add the key "
                "person/place/topic in ENGLISH (Latin script) so it is "
                "searchable in both languages")
        elif has_latin:
            pts += 1
            fails.append(
                "title is English only \u2014 make it BILINGUAL: include the "
                "native-script core of the headline alongside the English terms")
        else:
            fails.append("title has no alphabetic content \u2014 write a BILINGUAL "
                         "(English + native-script) title")

    # 2 pts: has a hook-style separator (— or : or ? or !) that structures the title
    if re.search(r"[—:?!]", t):
        pts += 2

    return min(pts, 20), fails


def _score_description(desc: str, hook: str,
                       title: str = "") -> Tuple[int, List[str]]:
    pts = 0
    fails: List[str] = []
    d = (desc or "").strip()
    if not d:
        return 0, ["description is empty"]

    # 8 pts: length in sweet spot
    if MIN_DESC_LEN <= len(d) <= MAX_DESC_LEN:
        pts += 8
    elif len(d) < MIN_DESC_LEN:
        fails.append(f"description {len(d)} chars — target {MIN_DESC_LEN}-{MAX_DESC_LEN}; add 2-3 context paragraphs")
    else:
        fails.append(f"description {len(d)} chars — over cap, trim to ≤{MAX_DESC_LEN}")

    # 4 pts: hook appears in the first ~200 chars
    h = (hook or "").strip()
    if h and h.lower()[:50] in d.lower()[:250]:
        pts += 4
    elif h:
        fails.append("hook sentence not in first 200 chars of description — move it to line 1")
    else:
        fails.append("no hook provided (needed as description's opening line)")

    # 4 pts: paragraph structure — at least 2 blank-line paragraph breaks
    blank_breaks = len(re.findall(r"\n\s*\n", d))
    if blank_breaks >= 2:
        pts += 4
    elif blank_breaks == 1:
        pts += 2
        fails.append("description has only 1 paragraph break — aim for 3-4 paragraphs")
    else:
        fails.append("description is one run-on paragraph — split into 3 paragraphs")

    # 2 pts: FIRST-125-CHARS quality — the only part YouTube shows in
    # search/suggested before "…more", so it must SELL, not restate the
    # title. 1pt: line-1 structure exists; 1pt: the opening 125 chars are
    # not just the title again (normalized containment either way).
    if "\n" in d[:400]:
        pts += 1
    first125 = re.sub(r"\s+", " ", d[:125].strip().lower())
    tnorm = re.sub(r"\s+", " ", (title or "").strip().lower())
    if first125 and tnorm and (tnorm in first125 or first125 in tnorm):
        fails.append(
            "description's first 125 chars just repeat the title — that's "
            "the ONLY text YouTube shows in search before '…more'; open "
            "with a NEW selling detail instead")
    else:
        pts += 1

    # 2 pts: no channel-brand leaks (from style source)
    low = d.lower()
    leaks = [b for b in _FORBIDDEN_BRANDS if b in low]
    if leaks:
        fails.append(f"description contains style-source brand leaks: {leaks[:3]}")
    else:
        pts += 2

    return min(pts, 20), fails


def _score_keywords(tags: List[str], trend_keywords: List[str]) -> Tuple[int, List[str]]:
    pts = 0
    fails: List[str] = []
    t = [s.strip().lower() for s in (tags or []) if s and s.strip()]

    # 10 pts: count in range
    if TAG_COUNT_MIN <= len(t) <= TAG_COUNT_MAX:
        pts += 10
    else:
        fails.append(f"keywords count {len(t)} — target {TAG_COUNT_MIN}-{TAG_COUNT_MAX}")

    # 4 pts: total char budget respected
    total_chars = sum(len(s) for s in t) + 2 * max(len(t) - 1, 0)
    if total_chars <= TAG_TOTAL_CAP:
        pts += 4
    else:
        fails.append(f"keywords total {total_chars} chars — over YouTube's {TAG_TOTAL_CAP} cap")

    # 4 pts: overlap with Google Trends' related keywords for this topic
    trend_set = {s.lower() for s in (trend_keywords or []) if s}
    if trend_set:
        overlap = sum(1 for k in t if any(tk in k or k in tk for tk in trend_set))
        # Low threshold: any 2+ trending keywords present = full credit.  The
        # prompt tells Gemini to include 2-3 trending keywords, so 2 is enough.
        if overlap >= 2:
            pts += 4
        elif overlap >= 1:
            pts += 3
            fails.append(f"only {overlap} trending keyword(s) in tags — push to 2+ for SEO lift")
        else:
            pts += 1
            fails.append(f"tags ignore Google Trends (0 of {len(trend_set)} matched) — add trending terms")
    else:
        pts += 4  # no trends data — full credit, can't verify what isn't there

    # 2 pts: no brand-name leaks in tags
    leaks = [k for k in t if any(b in k for b in _FORBIDDEN_BRANDS)]
    if not leaks:
        pts += 2
    else:
        fails.append(f"tags contain style-source brand names: {leaks[:3]} — replace with topic terms")

    return min(pts, 20), fails


def _score_hashtags(hashtags: List[str]) -> Tuple[int, List[str]]:
    pts = 0
    fails: List[str] = []
    h = [s.strip() for s in (hashtags or []) if s and s.strip()]

    # 8 pts: count in range
    if HASHTAG_MIN <= len(h) <= HASHTAG_MAX:
        pts += 8
    else:
        fails.append(f"hashtags count {len(h)} — target {HASHTAG_MIN}-{HASHTAG_MAX}")

    # 6 pts: all are valid CamelCase #Hashtags
    valid_fmt = [s for s in h if _HASHTAG_PATTERN.match(s)]
    bad = [s for s in h if s not in valid_fmt]
    if not bad:
        pts += 6
    elif len(valid_fmt) >= len(h) * 0.8:
        pts += 3
        fails.append(f"hashtags wrong format: {bad[:3]} — need #CamelCase, no spaces/punctuation")
    else:
        fails.append(f"hashtags wrong format: {bad[:3]} — rewrite as '#CamelCaseWord'")

    # 4 pts: dedupe — lowercase set size equals list size
    if len({s.lower() for s in h}) == len(h):
        pts += 4
    else:
        fails.append("hashtags contain duplicates (case-insensitive)")

    # 2 pts: no brand-leak hashtags (e.g. #RTVTelugu, #TV9Telugu)
    leaks = [s for s in h if any(b.replace(" ", "") in s.lower() for b in _FORBIDDEN_BRANDS)]
    if not leaks:
        pts += 2
    else:
        fails.append(f"hashtags contain style-source brand names: {leaks} — replace with topic hashtags")

    return min(pts, 20), fails


def _score_relevance(
    seo: Dict[str, Any],
    clip_topic: str,
    trend_keywords: List[str],
    news_items: List[Dict[str, Any]],
    competitor_terms: List[str] | None = None,
) -> Tuple[int, List[str]]:
    """Topic relevance + freshness bundle (20)."""
    pts = 0
    fails: List[str] = []
    title = (seo.get("title") or "").lower()
    desc = (seo.get("description") or "").lower()
    topic = (clip_topic or "").strip().lower()
    combined = f"{title} {desc}"
    # COMPETITOR COVERAGE (only when intel was provided — scoring without
    # intel is byte-identical to before): the rivals' winning query terms
    # must be represented in the output. 2 bonus-capped pts folded into
    # this dimension's 20 via the shared min() cap at the caller.
    if competitor_terms:
        kw_blob = " ".join(str(k).lower()
                           for k in (seo.get("keywords") or []))
        hay = f"{combined} {kw_blob}"
        covered = sum(1 for t in competitor_terms[:12]
                      if str(t).lower() in hay)
        if covered >= 2:
            pts += 2
        else:
            fails.append(
                "competitor intel provided but the output covers "
                f"only {covered} of their winning query terms/tags — weave "
                "in at least 2 that fit this video")

    # 6 pts: at least one token from the clip's topic appears in title or description
    if topic:
        topic_tokens = [t for t in re.findall(r"\w{3,}", topic) if t not in {"the", "and", "was", "with", "from"}]
        if topic_tokens:
            hits = sum(1 for tok in topic_tokens if tok in combined)
            ratio = hits / len(topic_tokens)
            if ratio >= 0.4:
                pts += 6
            elif ratio >= 0.2:
                pts += 3
                fails.append(f"only {hits}/{len(topic_tokens)} clip-topic words in output — tie content to the video's actual subject")
            else:
                fails.append(f"output barely references clip topic ({hits}/{len(topic_tokens)} words) — rewrite around it")
        else:
            pts += 3
    else:
        pts += 3  # no topic to compare against

    # 6 pts: at least 1 Google News source topic appears in output
    # When news was fetched but returned zero items (niche topic / cold
    # Google response), we don't penalize — it's not the writer's fault
    # that no articles exist.  Only penalize when news IS available AND
    # the output ignores it.
    if news_items:
        news_tokens = set()
        for n in news_items[:5]:
            for tok in re.findall(r"\w{4,}", (n.get("title") or "").lower()):
                news_tokens.add(tok)
        hits = sum(1 for t in news_tokens if t in combined)
        if hits >= 2:
            pts += 6
        elif hits >= 1:
            pts += 4
            fails.append(f"only {hits} token overlap with live Google News — ground more in news_context")
        else:
            pts += 1
            fails.append("zero overlap with Google News context — description should reference news facts")
    else:
        pts += 6  # no external context to verify against → full credit

    # 4 pts: at least 1 trend keyword appears in title or description
    if trend_keywords:
        hits = sum(1 for k in trend_keywords if k.lower() in combined)
        # Show Gemini the EXACT terms it should incorporate (top 5) so the
        # retry prompt has actionable, named targets rather than generic advice.
        top_terms = ", ".join(f"'{k}'" for k in trend_keywords[:5])
        if hits >= 2:
            pts += 4
        elif hits >= 1:
            pts += 3
            fails.append(
                f"only {hits} trending keyword in output. MUST weave in 2+ of these "
                f"exact trending terms into the title and/or description: {top_terms}."
            )
        else:
            pts += 1
            fails.append(
                f"zero trending keywords in output. MUST include at least 2 of these "
                f"exact terms in the title or description (verbatim if possible): {top_terms}."
            )
    else:
        pts += 4  # no trends data → full credit

    # 4 pts: no brand-leaks anywhere (hard requirement for SaaS safety)
    low = combined
    leaks = [b for b in _FORBIDDEN_BRANDS if b in low]
    if not leaks:
        pts += 4
    else:
        fails.append(f"style-source brand leaked in content: {leaks[:3]}")

    return min(pts, 20), fails


# ── Public entry ─────────────────────────────────────────────────────────────

def verify(
    seo: Dict[str, Any],
    *,
    clip_topic: str = "",
    trend_keywords: List[str] | None = None,
    news_items: List[Dict[str, Any]] | None = None,
    script_policy: str = "bilingual",
    competitor_terms: List[str] | None = None,
) -> Dict[str, Any]:
    """Score a generic SEO JSON deterministically.

    ``script_policy``: 'bilingual' | 'english' | 'native' — per-channel
    LEARNABLE (operator decision 2026-08): channels whose measured history
    favors one script get scored against THAT policy instead of the
    global bilingual default.

    Returns:
        {
            "score": int (0-100),
            "breakdown": {title, description, keywords, hashtags, relevance},
            "reasons":   [str, ...]   # specific, actionable failures
        }
    """
    trend_keywords = trend_keywords or []
    news_items = news_items or []

    t_pts, t_fails = _score_title(seo.get("title", ""),
                                  script_policy=script_policy)
    d_pts, d_fails = _score_description(seo.get("description", ""),
                                        seo.get("hook", ""),
                                        title=seo.get("title", ""))
    k_pts, k_fails = _score_keywords(seo.get("keywords", []), trend_keywords)
    h_pts, h_fails = _score_hashtags(seo.get("hashtags", []))
    r_pts, r_fails = _score_relevance(seo, clip_topic, trend_keywords,
                                      news_items,
                                      competitor_terms=competitor_terms)
    r_pts = min(r_pts, 20)   # competitor coverage folds into the 20 cap

    total = t_pts + d_pts + k_pts + h_pts + r_pts

    reasons: List[str] = []
    reasons += t_fails + d_fails + k_fails + h_fails + r_fails

    # THUMBNAIL–TITLE COHERENCE (advisory, no points): CTR is thumbnail ×
    # title together — a thumbnail_text sharing zero words with the title
    # confuses the click. Retry loop fixes it without shifting the budget.
    _tt = str(seo.get("thumbnail_text") or "").lower()
    _ttl = str(seo.get("title") or "").lower()
    if _tt and _ttl:
        _tt_words = {w for w in re.findall(r"[\wऀ-෿]{3,}", _tt)}
        if _tt_words and not any(w in _ttl for w in _tt_words):
            reasons.append(
                "thumbnail_text shares NO word with the title — align them "
                "(same person/number/verdict) so the thumbnail and title "
                "sell the same click")

    return {
        "score": max(0, min(100, total)),
        "breakdown": {
            "title":       t_pts,
            "description": d_pts,
            "keywords":    k_pts,
            "hashtags":    h_pts,
            "relevance":   r_pts,
        },
        "reasons": reasons,
    }
