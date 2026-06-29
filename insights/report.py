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


def _ids_md(ids: List[str]) -> str:
    if not ids:
        return "_—_"
    return ", ".join(f"[{v}]({_yt(v)})" for v in ids[:8])


def _pct(x, nd=1) -> str:
    return f"{x*100:.{nd}f}%" if isinstance(x, (int, float)) else "—"


def render_markdown(results: dict) -> str:
    """Deterministic full report from the structured analysis results. Always evidence-backed."""
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
        out.append(results.get("headline", ""))
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
    out.append(results.get("headline", "—"))
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
            out.append(f"\n**Videos:** {_ids_md(d['evidence'])}")
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

def _gemini_polish(results: dict, grounding_md: str):
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
        user = ("Rewrite this channel analysis into the report. Use ONLY these facts + video IDs.\n\n"
                "JSON facts:\n```json\n" + json.dumps(results, default=str)[:60000] + "\n```\n\n"
                "Deterministic draft to re-voice (keep every number + ID):\n\n" + grounding_md[:40000])
        resp = client.models.generate_content(
            model=model, contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM, temperature=0.5, max_output_tokens=8192),
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
    deterministic = render_markdown(results)

    md, used_provider, tin, tout = deterministic, "deterministic", 0, 0
    if provider == "gemini":
        polished = _gemini_polish(results, deterministic)
        if polished:
            md, tin, tout = polished[0], polished[1], polished[2]
            used_provider = "gemini"

    prev = (db.query(im.ReportVersion)
            .filter(im.ReportVersion.analysis_run_id == analysis_run_id)
            .order_by(im.ReportVersion.version.desc()).first())
    version = (prev.version + 1) if prev else 1

    rv = im.ReportVersion(
        analysis_run_id=analysis_run_id, user_id=run.user_id, version=version,
        provider=used_provider, exec_summary=_exec_summary(results),
        report_md=md, report_json=results, tokens_in=tin, tokens_out=tout, cost_usd=0.0,
    )
    db.add(rv); db.commit(); db.refresh(rv)
    return rv
