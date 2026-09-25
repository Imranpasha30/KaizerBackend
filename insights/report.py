"""Insights report generator — turns an AnalysisRun's structured results into a readable,
channel-specific strategy report (rendered in the V4 canvas + exportable).

Two layers:
  1. ``render_markdown`` — PURE, deterministic. Renders the computed facts (with cited
     video IDs) into a full report. Always works; this is the source of truth.
  2. ``generate_report`` — optionally has an LLM (Gemini/Claude) re-voice that report in
     plain, confident language, constrained to ONLY the provided facts + IDs (no invented
     numbers). Falls back to the deterministic markdown if no LLM is available.

VOICE: senior YouTube growth strategist — direct, unhedged on the DIAGNOSIS; honest that
outcomes also depend on uncontrollables (news cycle, timing luck, algorithmic variance).
Kaizer X = production & analytics; never framed as bulk-upload/monetization.

The deterministic renderer is unit-tested in ``scripts/test_insights_report.py``.
"""
from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

from insights import models as im

log = logging.getLogger("kaizer.insights.report")

_SYSTEM = """\
You are a senior YouTube growth strategist auditing ONE channel's performance for the
creator. Rewrite the provided analysis into a precise, plain-language report.

RULES:
- Use ONLY the facts, numbers, and video IDs in the JSON provided. NEVER invent a number,
  CTR, date, or video ID. If a section says data is insufficient, say so plainly.
- Be direct and confident in the DIAGNOSIS — name the problem, quantify it, rank it.
- State once, briefly, that views also depend on factors outside anyone's control (the news
  cycle, timing luck, algorithmic variance): the promise is the most accurate diagnosis and
  the highest-odds fix on every upload, not guaranteed views.
- Kaizer X is a video production & analytics tool — improving editorial/content performance.
  Never frame it as a system for bulk uploads or monetizing channels.
- Keep the structure: executive summary first, then each dimension (finding → impact →
  offending video IDs → fix), ranked fixes, concrete targets, and the next-10 checklist.
- Output GitHub-flavoured markdown. No preamble, no sign-off.
"""


def _yt(vid: str) -> str:
    return f"https://youtu.be/{vid}"


def _ids_md(ids: List[str], titles: Optional[dict] = None) -> str:
    """Render evidence videos. With a titles map we show the human TITLE (linked)
    instead of the raw video id — a creator can't read '[Tw1pRhMsFBM]'."""
    if not ids:
        return "_—_"
    titles = titles or {}
    parts = []
    for v in ids[:8]:
        t = (titles.get(v) or "").strip()
        label = ((t[:55] + "…") if len(t) > 56 else t) if t else v
        parts.append(f"[{label}]({_yt(v)})")
    return ", ".join(parts)


def _pct(x, nd=1) -> str:
    return f"{x*100:.{nd}f}%" if isinstance(x, (int, float)) else "—"


# Plain-English names for the internal feature/factor keys — a creator reads
# "title length", not "len_chars"; "how long people keep watching", not "retention r".
_FEATURE_LABELS = {
    "retention": "how long people keep watching",
    "len_chars": "title length",
    "title_length": "title length",
    "early_velocity": "how fast a video takes off in its first 2 days",
    "browse_share": "how often YouTube recommends it (Home / Browse)",
    "power_words": "punchy, emotional words in the title",
    "ctr": "how often people click your thumbnail",
    "has_number": "putting a number in the title",
    "has_question": "asking a question in the title",
    "caps_ratio": "using CAPS in the title",
    "has_brackets": "using [brackets] in the title",
}


def _label(f: str) -> str:
    return _FEATURE_LABELS.get(f, str(f).replace("_", " "))


def _strength(r) -> str:
    a = abs(r or 0)
    return "a strong" if a >= 0.30 else ("a clear" if a >= 0.15 else "a slight")


import re as _re

# Ugly internal codes → plain words (used only by the plain-report scrubber).
_CODE_SUBS = {
    "len_chars": "title length",
    "power_words": "punchy title words",
    "browse_share": "YouTube recommending it",
    "early_velocity": "how fast it takes off",
    "caps_ratio": "CAPS in the title",
    "has_number": "a number in the title",
    "has_question": "a question in the title",
    "has_brackets": "[brackets] in the title",
}


def _plainify(s: str) -> str:
    """Strip statistics jargon from a technical string for the PLAIN report:
    drop r-values / 'measured on N videos', swap ugly codes + jargon phrases."""
    s = str(s or "")
    for code, lab in _CODE_SUBS.items():
        s = _re.sub(r"'?\b" + _re.escape(code) + r"\b'?", lab, s)
    s = _re.sub(r"\(strongest correlation[^)]*\)", "", s)
    s = _re.sub(r"\(r\s*=\s*[+-]?\d*\.?\d+[^)]*\)", "", s)
    s = _re.sub(r"\br\s*=\s*[+-]?\d*\.?\d+", "", s)
    s = _re.sub(r",?\s*measured on [\d,]+ videos", "", s, flags=_re.I)
    s = s.replace("packaging line", "click-rate cutoff").replace("sub-threshold", "low-click")
    s = _re.sub(r"\s+([.;,])", r"\1", s)
    return _re.sub(r"\s{2,}", " ", s).strip()


def _titles_list_md(ids: List[str], titles: Optional[dict], n: int = 5) -> str:
    """A bulleted list of example videos by TITLE (linked) for the plain report."""
    if not ids:
        return ""
    titles = titles or {}
    lines = []
    for v in ids[:n]:
        t = (titles.get(v) or "").strip()
        label = ((t[:70] + "…") if len(t) > 71 else t) if t else v
        lines.append(f"- [{label}]({_yt(v)})")
    return "\n".join(lines)


def render_markdown_plain(results: dict, titles: Optional[dict] = None) -> str:
    """PLAIN-ENGLISH report for a creator — zero statistics jargon, no correlation
    numbers, video TITLES not ids, every point paired with one concrete action.
    Deterministic (always works); the analytic view keeps the precise numbers."""
    if not results:
        return "# Your channel report\n\n_No analysis available yet._"
    titles = titles or {}
    mode = results.get("mode", "deep")
    out: List[str] = []
    out.append("# Your channel — what to fix next (plain English)")
    out.append("")
    out.append("> The clearest read on what's holding your videos back and the one change most "
               "likely to help on your next upload. Views also swing on the news cycle and luck — "
               "this nails what you control.")
    out.append("")

    if mode == "starter":
        out.append("## Starter plan")
        out.append(results.get("headline") or "")
        bm = results.get("benchmarks", {})
        if bm:
            out.append(f"\nAim for a click rate around **{_pct(bm.get('ctr'))}** and keep people watching "
                       f"about **{_pct(bm.get('avg_view_pct'))}** of the video.")
        for sect, items in (results.get("plan", {}) or {}).items():
            out.append(f"\n### {sect.capitalize()}")
            for it in items:
                out.append(f"- {it}")
        out.append("\n## Your next 10 uploads")
        for i, step in enumerate(results.get("next_10", []), 1):
            out.append(f"{i}. {step}")
        return "\n".join(out)

    # Big picture
    prof = results.get("channel_profile") or {}
    cov = results.get("data_coverage", {})
    out.append("## The big picture")
    if prof.get("name"):
        out.append(f"- Your channel reads as **{prof['name']}**"
                   + (" (high-volume — the plan keeps your volume, it never tells you to post less)."
                      if prof.get("high_volume") else "."))
    out.append(f"- Looked at **{cov.get('n_videos', 0):,}** of your videos. A typical one gets about "
               f"**{int(results.get('median_views', 0)):,} views**.")
    if results.get("headline"):
        out.append(f"- {_plainify(results['headline'])}")

    # What your winners have in common
    tiers = results.get("tiers", {})
    bo = tiers.get("breakout", {})
    if bo.get("n"):
        tops = ", ".join(bo.get("top_topics", []) or []) or "your strongest topics"
        out.append("\n## What your best videos have in common")
        out.append(f"**{bo['n']:,}** of your videos took off (5×+ your usual views). They lean into: "
                   f"**{tops}**. Make more like these.")

    # Biggest opportunities (ranked drivers, in plain words)
    drivers = [d for d in (results.get("drivers", []) or []) if not d.get("limited_sample")][:4]
    if drivers:
        out.append("\n## Your biggest levers (most → least)")
        for i, dr in enumerate(drivers, 1):
            up = (dr.get("correlation", 0) or 0) >= 0
            more = "more of it goes with **more** views" if up else "more of it goes with **fewer** views"
            out.append(f"{i}. **{_label(dr['factor']).capitalize()}** — {_strength(dr.get('correlation'))} "
                       f"link with your views: {more}.")

    # Fix these first — dimensions with a real finding + action
    dims = [d for d in (results.get("dimensions", []) or [])
            if d.get("status") == "ok" and d.get("fix")]
    if dims:
        out.append("\n## Fix these first")
        for d in dims:
            out.append(f"\n### {d.get('title','')}")
            if d.get("impact"):
                out.append(_plainify(d["impact"]))
            out.append(f"**Do this:** {_plainify(d['fix'])}")
            ev = _titles_list_md(d.get("evidence") or [], titles, n=5)
            if ev:
                out.append("\nStart with these videos:")
                out.append(ev)

    # Best time to post — plain
    per_day = (results.get("timing") or {}).get("per_day_best") or []
    if per_day:
        out.append("\n## Best time to post")
        out.append("Your strongest hour each day (your channel's local time):")
        for d in per_day:
            out.append(f"- **{d['day']}:** {d['hour']:02d}:00 "
                       f"({((d['hour'] % 12) or 12)}{'AM' if d['hour'] < 12 else 'PM'})")

    # Targets — plain
    t = results.get("targets", {})
    tgt = []
    if t.get("target_ctr") is not None:
        tgt.append(f"get **{_pct(t['target_ctr'])}** of people who see your thumbnail to click")
    if t.get("target_retention_pct") is not None:
        tgt.append(f"keep people watching about **{round(t['target_retention_pct'])}%** of the video")
    if tgt:
        out.append("\n## What to aim for")
        out.append("Match your own best videos: " + "; ".join(tgt) + ".")

    out.append("\n## Your next 10 uploads")
    for i, step in enumerate(results.get("next_10", []), 1):
        out.append(f"{i}. {step}")
    return "\n".join(out)


def render_markdown(results: dict, titles: Optional[dict] = None) -> str:
    """Deterministic full report from the structured analysis results. Always evidence-backed.
    ANALYTIC view — keeps the precise numbers/correlations (the "Analytics" toggle)."""
    if not results:
        return "# Channel Insights\n\n_No analysis available._"
    mode = results.get("mode", "deep")
    cov = results.get("data_coverage", {})
    out: List[str] = []
    out.append("# Channel Insights — Trend Finder")
    out.append("")
    out.append("> The most accurate diagnosis of what's holding your videos back, and the "
               "highest-odds fix on every upload. Views also depend on things outside anyone's "
               "control (the news cycle, timing luck, algorithmic variance) — so this is total "
               "precision on what you *can* control, and honesty on what you can't.")
    out.append("")

    if mode == "starter":
        out.append("## Starter plan")
        out.append(results.get("headline") or "")
        bm = results.get("benchmarks", {})
        if bm:
            out.append(f"\n**Benchmark ({bm.get('label','')}):** target CTR "
                       f"~{_pct(bm.get('ctr'))}, average retention ~{_pct(bm.get('avg_view_pct'))}.")
        plan = results.get("plan", {})
        for sect, items in plan.items():
            out.append(f"\n### {sect.capitalize()}")
            for it in items:
                out.append(f"- {it}")
        out.append("\n## Your next 10 uploads")
        for i, step in enumerate(results.get("next_10", []), 1):
            out.append(f"{i}. {step}")
        if results.get("note"):
            out.append(f"\n_{results['note']}_")
        return "\n".join(out)

    # DEEP / BLEND
    out.append("## Executive summary")
    prof = results.get("channel_profile") or {}
    if prof.get("name"):
        vol = " (high-volume — strategy is tuned for that: prioritize within your volume, never post less)" if prof.get("high_volume") else ""
        out.append(f"**Channel type detected:** {prof['name']}{vol}.\n")
    out.append(results.get("headline") or "—")
    if cov.get("caveats"):
        out.append("\n**Data coverage:** " + " ".join(cov["caveats"]))
    out.append(f"\nAnalyzed **{cov.get('n_videos', 0)}** videos"
               + (f" · thumbnail CTR on {cov.get('n_with_ctr', 0)} of them" if cov.get("n_with_ctr") else "")
               + f" · median ~{int(results.get('median_views', 0)):,} views.")

    # Tiers
    tiers = results.get("tiers", {})
    out.append("\n## Performance tiers")
    out.append("| Tier | Videos | Avg CTR | Avg retention | Avg early vel. | Top topics |")
    out.append("|---|---|---|---|---|---|")
    for key, label in (("breakout", "Breakout (>5× median)"), ("solid", "Solid"), ("under", "Underperformer")):
        t = tiers.get(key, {})
        out.append(f"| {label} | {t.get('n', 0)} | {_pct(t.get('avg_ctr'))} | "
                   f"{(str(round(t['avg_retention_pct']))+'%') if t.get('avg_retention_pct') is not None else '—'} | "
                   f"{(str(round(t['avg_velocity_vph']))+'/hr') if t.get('avg_velocity_vph') is not None else '—'} | "
                   f"{', '.join(t.get('top_topics', []) or []) or '—'} |")

    # Dimensions
    out.append("\n## Diagnosis by dimension")
    for d in results.get("dimensions", []):
        flag = "" if d.get("status") == "ok" else f" _({d.get('status')})_"
        out.append(f"\n### {d.get('title','')}{flag}")
        out.append(d.get("finding", ""))
        if d.get("impact"):
            out.append(f"\n**Impact:** {d['impact']}")
        if d.get("evidence"):
            out.append(f"\n**Videos:** {_ids_md(d['evidence'], titles)}")
        if d.get("fix"):
            out.append(f"\n**Fix:** {d['fix']}")

    # Ranked drivers
    drivers = results.get("drivers", [])
    if drivers:
        out.append("\n## What actually moves views on this channel (ranked)")
        for i, dr in enumerate(drivers, 1):
            tag = " · ⚠ limited sample" if dr.get("limited_sample") else ""
            out.append(f"{i}. **{dr['factor']}** — r={dr['correlation']:+.2f}, measured on "
                       f"{dr.get('n', 0):,} videos{tag} ({dr['direction']})")
        out.append("\n_Ranked by correlation with views on your own data, with well-measured factors "
                   "first — strong association, not proven causation; act on the top ones first._")

    # Targets
    t = results.get("targets", {})
    out.append("\n## Your targets")
    if t.get("target_ctr") is not None:
        out.append(f"- **CTR:** aim for ≥ {_pct(t['target_ctr'])} (your breakouts' level).")
    if t.get("target_retention_pct") is not None:
        out.append(f"- **Retention:** aim for ≥ {round(t['target_retention_pct'])}%.")
    if t.get("best_windows"):
        ws = ", ".join(f"{w['day']} {w['hour']:02d}:00" for w in t["best_windows"])
        out.append(f"- **Best publish windows (local):** {ws}.")
    if t.get("recommended_videos_per_day") is not None:
        out.append(f"- **Cadence:** ~{t['recommended_videos_per_day']} videos/day, well-spaced.")

    # Best time to post — per weekday schedule (the chart lives in the app; here as a table).
    per_day = (results.get("timing") or {}).get("per_day_best") or []
    if per_day:
        out.append("\n## Best time to post — your week")
        out.append("Each day's strongest hour, from your own data (channel local time). Higher = videos "
                   "posted then tend to gain views fastest in their first hours.")
        out.append("\n| Day | Best hour | Strength (early views/hr) |")
        out.append("|---|---|---|")
        for d in per_day:
            out.append(f"| {d['day']} | {d['hour']:02d}:00 | {d.get('score', 0)} |")

    out.append("\n## Your next 10 uploads")
    for i, step in enumerate(results.get("next_10", []), 1):
        out.append(f"{i}. {step}")

    # Plain-language glossary so anyone can read this.
    gloss = results.get("glossary") or {}
    if gloss:
        out.append("\n## What these words mean")
        for term, desc in gloss.items():
            out.append(f"- **{term}:** {desc}")
    return "\n".join(out)


def _exec_summary(results: dict) -> str:
    return (results.get("headline") or "")[:1000]


# ── LLM polish (optional) ────────────────────────────────────────────────

_SYSTEM_PLAIN = """\
You are a friendly YouTube coach explaining ONE channel's report to the creator in PLAIN English.

RULES:
- Use ONLY the facts in the JSON + the draft. NEVER invent a number, CTR, date, title or video.
- NO jargon, NO statistics words: never write "correlation", "r=", "coefficient", "median",
  "packaging line", "velocity", or raw feature codes like "len_chars". Say it in everyday words.
- Refer to videos by their TITLE (already in the draft), never by a video id code.
- Every point = what's happening + ONE concrete thing to do next. Keep it warm and short.
- Say once, briefly, that views also depend on the news cycle and luck — this covers what they control.
- Keep the section headings from the draft. Output GitHub-flavoured markdown. No preamble, no sign-off.
"""


def _gemini_polish(results: dict, grounding_md: str, system: str = _SYSTEM):
    """Re-voice the deterministic report via Gemini, constrained to the given facts. Returns
    (markdown, tokens_in, tokens_out) or None on any failure."""
    try:
        from seo.generator import _gemini_client
        from google.genai import types as genai_types
    except Exception as exc:
        log.info("insights report: gemini unavailable (%s)", str(exc)[:120]); return None
    try:
        client = _gemini_client()
        model = os.environ.get("KAIZER_INSIGHTS_MODEL", "gemini-2.5-flash")
        user = ("Rewrite this channel analysis into the report. Use ONLY these facts + video titles.\n\n"
                "JSON facts:\n```json\n" + json.dumps(results, default=str)[:60000] + "\n```\n\n"
                "Draft to re-voice (keep every fact + video title):\n\n" + grounding_md[:40000])
        resp = client.models.generate_content(
            model=model, contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=system, temperature=0.5, max_output_tokens=8192),
        )
        text = (resp.text or "").strip()
        if not text:
            return None
        tin = tout = 0
        try:
            um = getattr(resp, "usage_metadata", None)
            tin = int(getattr(um, "prompt_token_count", 0) or 0)
            tout = int(getattr(um, "candidates_token_count", 0) or 0)
        except Exception:
            pass
        return text, tin, tout
    except Exception as exc:
        log.info("insights report: gemini polish failed (%s)", str(exc)[:160]); return None


def generate_report(db, analysis_run_id: int, *, provider: str = "gemini") -> im.ReportVersion:
    """Render the report for an AnalysisRun and persist a new ReportVersion. Always produces
    a deterministic report; uses the LLM to re-voice it when available (provider='gemini').
    provider='deterministic' skips the LLM."""
    run = db.query(im.AnalysisRun).filter(im.AnalysisRun.id == analysis_run_id).first()
    if not run:
        raise RuntimeError(f"analysis run {analysis_run_id} not found")
    results = run.results or {}

    # video_id → title, so both reports name videos instead of raw id codes.
    titles: dict = {}
    try:
        rows = (db.query(im.VideoMetric.video_id, im.VideoMetric.title)
                .filter(im.VideoMetric.snapshot_id == run.snapshot_id).all())
        titles = {vid: (t or "") for vid, t in rows}
    except Exception:
        titles = {}

    # ANALYTIC = deterministic (precise numbers, the "Analytics" toggle).
    analytic = render_markdown(results, titles)
    # PLAIN = default view; deterministic, optionally Gemini-polished for voice.
    plain = render_markdown_plain(results, titles)

    used_provider, tin, tout = "deterministic", 0, 0
    if provider == "gemini":
        polished = _gemini_polish(results, plain, system=_SYSTEM_PLAIN)
        if polished:
            plain, tin, tout = polished[0], polished[1], polished[2]
            used_provider = "gemini"

    # Carry the analytic narrative in report_json so the UI can toggle to it.
    # (Plain rides in report_md; the UI falls back to it — no need to duplicate.)
    results_out = dict(results)
    results_out["narrative_analytic"] = analytic

    prev = (db.query(im.ReportVersion)
            .filter(im.ReportVersion.analysis_run_id == analysis_run_id)
            .order_by(im.ReportVersion.version.desc()).first())
    version = (prev.version + 1) if prev else 1

    rv = im.ReportVersion(
        analysis_run_id=analysis_run_id, user_id=run.user_id, version=version,
        provider=used_provider, exec_summary=_exec_summary(results),
        report_md=plain, report_json=results_out, tokens_in=tin, tokens_out=tout, cost_usd=0.0,
    )
    db.add(rv); db.commit(); db.refresh(rv)
    return rv
