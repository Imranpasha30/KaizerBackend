"""Admin observability page for the v2 publish/upload pipeline — Phase 3.G.

Single HTML page at ``GET /admin/upload-v2`` that surfaces, at a glance:

* Feature-flag header (KAIZER_NEW_PUBLISH_PATH, KAIZER_CLEAN_MASTER, …)
* Scheduler snapshot — queue depth + slot usage + in-flight count
* Branding cache — hit/miss ratio + last cleanup
* Burn-rate ledger — predicted vs actual totals + last-N rows
* Credit health — top users by balance + plan-tier distribution
* Quota gate — today's used / cap from ``youtube.quota_v2.daily_cap()``
* Idempotency — last-10 ``publish_attempts`` rows + recovered-orphan count

Auth: ``Depends(auth.admin_required)`` — mirrors every other ``routers/admin*``
endpoint. Disabled accounts → 403, non-admins → 403.

The page is plain server-rendered HTML (no React, no Jinja templates).
A 5-second meta-refresh keeps the page live without JavaScript so it
remains usable from low-power monitoring terminals.
"""
from __future__ import annotations

import html
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

import auth
import models
from database import get_db

log = logging.getLogger("kaizer.routers.admin_upload_v2")

router = APIRouter(tags=["admin"])


# ─── Helpers ────────────────────────────────────────────────────────────────


def _flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def _iso(dt: Optional[datetime]) -> str:
    if dt is None:
        return "-"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds")


def _esc(value: Any) -> str:
    """HTML-escape a value (None → empty string). Defensive against
    raw user data ending up in admin HTML."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


# ─── Snapshot pulls (read-only; never modifies DB) ──────────────────────────


def _scheduler_snapshot() -> dict:
    try:
        from services import scheduler as _sched
        return _sched.snapshot()
    except Exception as exc:
        log.warning("admin_upload_v2: scheduler.snapshot failed: %s", exc)
        return {"error": str(exc)}


def _branding_snapshot() -> dict:
    try:
        from services import branding as _branding
        return _branding.snapshot()
    except Exception as exc:
        log.warning("admin_upload_v2: branding.snapshot failed: %s", exc)
        return {"error": str(exc)}


def _durable_queue_enabled() -> bool:
    """Wave 2: panel only renders on KAIZER_DURABLE_QUEUE=1 deployments
    (re-read per request, same as the publish-path flag)."""
    return (os.environ.get("KAIZER_DURABLE_QUEUE", "0") or "0").strip() == "1"


def _durable_queue_snapshot(db: Session) -> dict:
    """Wave 2: Postgres-backed queue stats (Wave 1's job_queue). Queued/
    active/parked come from queue_depth_snapshot(); 'retrying' counts
    queued rows still waiting out their next_attempt_at backoff window."""
    try:
        from services.job_queue import queue_depth_snapshot
        snap = dict(queue_depth_snapshot())
    except Exception as exc:
        log.warning("admin_upload_v2: queue_depth_snapshot failed: %s", exc)
        return {"error": str(exc)}
    try:
        from sqlalchemy import text as _text
        snap["retrying"] = int(db.execute(_text(
            "SELECT count(*) FROM upload_jobs_v2 "
            "WHERE status = 'queued' AND next_attempt_at > CURRENT_TIMESTAMP"
        )).scalar() or 0)
    except Exception as exc:
        log.warning("admin_upload_v2: retrying count failed: %s", exc)
        snap["retrying"] = 0
    return snap


def _burn_snapshot(db: Session) -> dict:
    try:
        from services import burn_log as _burn
        return _burn.snapshot(db, window_minutes=60 * 24)  # last 24h
    except Exception as exc:
        log.warning("admin_upload_v2: burn_log.snapshot failed: %s", exc)
        return {"error": str(exc)}


def _quota_today(db: Session) -> dict:
    """Today's used quota units + the configured cap, via youtube.quota_v2.

    Reads the api_quota row(s) for today's UTC bucket. Returns a tuple of
    {used, cap, pct, by_key}.
    """
    try:
        from youtube import quota_v2 as _qv2
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = (
            db.query(models.ApiQuota.api_key_hash, models.ApiQuota.units_used)
            .filter(models.ApiQuota.date == today_str)
            .all()
        )
        by_key = {str(kh): int(u or 0) for kh, u in rows}
        used = sum(by_key.values())
        cap = _qv2.daily_cap()
        pct = round((used / cap) * 100.0, 2) if cap > 0 else 0.0
        return {"used": used, "cap": cap, "pct": pct, "by_key": by_key, "date": today_str}
    except Exception as exc:
        log.warning("admin_upload_v2: _quota_today failed: %s", exc)
        return {"error": str(exc)}


def _burn_recent(db: Session, limit: int = 20) -> list[dict]:
    try:
        rows = (
            db.query(models.QuotaBurnLog)
            .order_by(desc(models.QuotaBurnLog.created_at))
            .limit(limit)
            .all()
        )
        out = []
        for r in rows:
            out.append({
                "id": int(r.id),
                "upload_job_id": int(r.upload_job_id) if r.upload_job_id else None,
                "operation": str(r.operation or ""),
                "predicted": int(r.predicted_cost or 0),
                "actual": (int(r.reconciled_actual_cost) if r.reconciled_actual_cost is not None else None),
                "outcome": str(r.observed_outcome or ""),
                "http_status": (int(r.http_status) if r.http_status is not None else None),
                "created_at": _iso(r.created_at),
            })
        return out
    except Exception as exc:
        log.warning("admin_upload_v2: _burn_recent failed: %s", exc)
        return []


def _credit_top_users(db: Session, limit: int = 10) -> list[dict]:
    """Last balance_after per user (denormalized = current balance),
    joined to plan_tiers. Sorted DESC by balance."""
    try:
        latest_ids_subq = (
            db.query(func.max(models.CreditLedger.id).label("max_id"))
            .group_by(models.CreditLedger.user_id)
            .subquery()
        )
        rows = (
            db.query(
                models.CreditLedger.user_id,
                models.CreditLedger.balance_after,
                models.User.email,
                models.PlanTier.name,
            )
            .join(latest_ids_subq, models.CreditLedger.id == latest_ids_subq.c.max_id)
            .join(models.User, models.CreditLedger.user_id == models.User.id)
            .outerjoin(models.PlanTier, models.User.plan_tier_id == models.PlanTier.id)
            .order_by(desc(models.CreditLedger.balance_after))
            .limit(limit)
            .all()
        )
        return [
            {
                "user_id": int(uid),
                "email": str(email or ""),
                "balance": int(bal or 0),
                "plan_tier": str(plan or "-"),
            }
            for uid, bal, email, plan in rows
        ]
    except Exception as exc:
        log.warning("admin_upload_v2: _credit_top_users failed: %s", exc)
        return []


def _plan_tier_distribution(db: Session) -> dict:
    """How many users are on each plan_tier? Returns {tier_name: count}."""
    try:
        rows = (
            db.query(models.PlanTier.name, func.count(models.User.id))
            .outerjoin(models.User, models.User.plan_tier_id == models.PlanTier.id)
            .group_by(models.PlanTier.name)
            .all()
        )
        return {str(name or "unknown"): int(c or 0) for name, c in rows}
    except Exception as exc:
        log.warning("admin_upload_v2: _plan_tier_distribution failed: %s", exc)
        return {}


def _attempts_recent(db: Session, limit: int = 10) -> list[dict]:
    try:
        rows = (
            db.query(models.PublishAttempt)
            .order_by(desc(models.PublishAttempt.created_at))
            .limit(limit)
            .all()
        )
        out = []
        for r in rows:
            out.append({
                "id": int(r.id),
                "upload_job_id": int(r.upload_job_id) if r.upload_job_id else None,
                "status": str(r.status or ""),
                "attempt_no": int(r.attempt_no or 0),
                "worker_id": str(r.worker_id or ""),
                "youtube_video_id": str(r.youtube_video_id or ""),
                "created_at": _iso(r.created_at),
            })
        return out
    except Exception as exc:
        log.warning("admin_upload_v2: _attempts_recent failed: %s", exc)
        return []


def _orphan_count(db: Session) -> int:
    try:
        return int(
            db.query(func.count(models.PublishAttempt.id))
            .filter(models.PublishAttempt.status == "recovered")
            .scalar() or 0
        )
    except Exception:
        return 0


def _publish_task_status_counts(db: Session) -> dict:
    try:
        rows = (
            db.query(models.PublishTask.status, func.count(models.PublishTask.id))
            .group_by(models.PublishTask.status).all()
        )
        return {str(s or "-"): int(c or 0) for s, c in rows}
    except Exception:
        return {}


def _upload_job_status_counts(db: Session) -> dict:
    try:
        rows = (
            db.query(models.UploadJobV2.status, func.count(models.UploadJobV2.id))
            .group_by(models.UploadJobV2.status).all()
        )
        return {str(s or "-"): int(c or 0) for s, c in rows}
    except Exception:
        return {}


# ─── HTML rendering ─────────────────────────────────────────────────────────


def _render_kv_table(rows: list[tuple[str, Any]]) -> str:
    """Render a simple two-column key/value table."""
    body = "\n".join(
        f'<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>' for k, v in rows
    )
    return f'<table class="kv">{body}</table>'


def _render_dict_table(d: dict, *, key_label: str = "key", value_label: str = "value") -> str:
    if not d:
        return '<p class="muted">(empty)</p>'
    rows = "\n".join(
        f'<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>'
        for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))
    )
    return (
        f'<table class="kv"><thead><tr>'
        f'<th>{_esc(key_label)}</th><th>{_esc(value_label)}</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


def _render_burn_rows(rows: list[dict]) -> str:
    if not rows:
        return '<p class="muted">(no QuotaBurnLog rows yet)</p>'
    header = (
        '<tr><th>id</th><th>job</th><th>operation</th>'
        '<th>predicted</th><th>actual</th><th>outcome</th>'
        '<th>http</th><th>created_at</th></tr>'
    )
    body = "\n".join(
        f'<tr>'
        f'<td>{r["id"]}</td>'
        f'<td>{_esc(r["upload_job_id"]) if r["upload_job_id"] is not None else "-"}</td>'
        f'<td>{_esc(r["operation"])}</td>'
        f'<td class="num">{_esc(r["predicted"])}</td>'
        f'<td class="num">{_esc(r["actual"]) if r["actual"] is not None else "-"}</td>'
        f'<td class="outcome outcome-{_esc(r["outcome"])}">{_esc(r["outcome"])}</td>'
        f'<td>{_esc(r["http_status"]) if r["http_status"] is not None else "-"}</td>'
        f'<td class="ts">{_esc(r["created_at"])}</td>'
        f'</tr>'
        for r in rows
    )
    return f'<table class="data"><thead>{header}</thead><tbody>{body}</tbody></table>'


def _render_credit_rows(rows: list[dict]) -> str:
    if not rows:
        return '<p class="muted">(no credit_ledger rows yet)</p>'
    header = '<tr><th>user</th><th>email</th><th>plan</th><th>balance</th></tr>'
    body = "\n".join(
        f'<tr>'
        f'<td>{_esc(r["user_id"])}</td>'
        f'<td>{_esc(r["email"])}</td>'
        f'<td>{_esc(r["plan_tier"])}</td>'
        f'<td class="num">{_esc(r["balance"])}</td>'
        f'</tr>'
        for r in rows
    )
    return f'<table class="data"><thead>{header}</thead><tbody>{body}</tbody></table>'


def _render_attempts_rows(rows: list[dict]) -> str:
    if not rows:
        return '<p class="muted">(no publish_attempts rows yet)</p>'
    header = (
        '<tr><th>id</th><th>job</th><th>status</th><th>attempt#</th>'
        '<th>worker</th><th>yt_video_id</th><th>created_at</th></tr>'
    )
    body = "\n".join(
        f'<tr>'
        f'<td>{_esc(r["id"])}</td>'
        f'<td>{_esc(r["upload_job_id"]) if r["upload_job_id"] is not None else "-"}</td>'
        f'<td class="status status-{_esc(r["status"])}">{_esc(r["status"])}</td>'
        f'<td>{_esc(r["attempt_no"])}</td>'
        f'<td>{_esc(r["worker_id"])}</td>'
        f'<td>{_esc(r["youtube_video_id"]) or "-"}</td>'
        f'<td class="ts">{_esc(r["created_at"])}</td>'
        f'</tr>'
        for r in rows
    )
    return f'<table class="data"><thead>{header}</thead><tbody>{body}</tbody></table>'


_CSS = """
  body {
    font: 13px/1.45 -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0e1116; color: #d7dde7; margin: 0; padding: 20px;
  }
  h1 { font-size: 18px; margin: 0 0 4px; color: #e6edf3; }
  h2 { font-size: 14px; margin: 0 0 8px; color: #79c0ff; letter-spacing: 0.5px; text-transform: uppercase; }
  .meta { color: #8b949e; font-size: 11px; margin-bottom: 16px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 14px; }
  .card {
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 14px;
  }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  table.kv th { text-align: left; color: #79c0ff; font-weight: 500; padding: 3px 8px 3px 0; white-space: nowrap; vertical-align: top; }
  table.kv td { padding: 3px 0; color: #d7dde7; }
  table.data th { text-align: left; color: #8b949e; font-weight: 500; padding: 4px 8px; border-bottom: 1px solid #30363d; background: #1f242c; }
  table.data td { padding: 4px 8px; border-bottom: 1px solid #1f242c; }
  table.data td.num, td.num { text-align: right; font-variant-numeric: tabular-nums; }
  table.data td.ts, td.ts { color: #8b949e; font-size: 11px; }
  .muted { color: #6e7681; font-style: italic; }
  .flag { display: inline-block; padding: 1px 6px; margin-right: 4px; border-radius: 3px; font-family: ui-monospace, monospace; font-size: 11px; }
  .flag-on { background: #1f6f43; color: #b7f0c8; }
  .flag-off { background: #553e3a; color: #ffb3b3; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; }
  .outcome-success    { color: #a6e3a1; }
  .outcome-transient_error  { color: #f9e2af; }
  .outcome-quota_exceeded   { color: #f38ba8; font-weight: bold; }
  .outcome-permanent_error  { color: #f38ba8; }
  .status-in_flight   { color: #f9e2af; }
  .status-completed   { color: #a6e3a1; }
  .status-failed      { color: #f38ba8; }
  .status-recovered   { color: #89b4fa; }
  .progress { background: #1f242c; border-radius: 3px; overflow: hidden; height: 10px; margin-top: 6px; }
  .progress > div { background: #2da44e; height: 100%; transition: width .25s ease; }
  .progress > div.warn { background: #d29922; }
  .progress > div.crit { background: #f85149; }
  a, a:visited { color: #79c0ff; }
"""


def _render_page(
    *,
    flags: dict,
    sched: dict,
    branding: dict,
    burn: dict,
    burn_rows: list[dict],
    quota: dict,
    credit_users: list[dict],
    plan_dist: dict,
    attempts: list[dict],
    orphan_count: int,
    publish_status_counts: dict,
    upload_status_counts: dict,
    durable: Optional[dict] = None,
) -> str:
    """Assemble the HTML page. Pure string building — no template engine."""
    # Flag header
    def _flag_span(name: str, val: str, on_value: str = "1") -> str:
        klass = "flag-on" if val == on_value else "flag-off"
        return f'<span class="flag {klass}">{name}={_esc(val)}</span>'

    flag_html = " ".join(
        _flag_span(k, v) for k, v in flags.items()
    )

    # Quota gate gauge
    used = int(quota.get("used", 0))
    cap = int(quota.get("cap", 0))
    pct = float(quota.get("pct", 0.0))
    pct_klass = ""
    if pct >= 90:
        pct_klass = "crit"
    elif pct >= 70:
        pct_klass = "warn"
    bar_width = max(0.0, min(100.0, pct))
    quota_panel = _render_kv_table([
        ("Date (UTC)", quota.get("date", "-")),
        ("Used (units)", f"{used:,}"),
        ("Cap (units)", f"{cap:,}"),
        ("Used %", f"{pct:.2f}%"),
    ])
    quota_panel += (
        f'<div class="progress"><div class="{pct_klass}" '
        f'style="width:{bar_width:.2f}%"></div></div>'
    )
    by_key_panel = _render_dict_table(
        quota.get("by_key") or {},
        key_label="api_key_hash", value_label="units",
    )

    # Scheduler card
    slot = sched.get("slot_manager") or {}
    sched_panel = _render_kv_table([
        ("Started", sched.get("started", "-")),
        ("Queue depth", sched.get("queue_depth", "-")),
        ("In-flight", sched.get("in_flight_count", "-")),
        ("Oldest age (s)", round(sched.get("oldest_age_seconds") or 0, 2)),
        ("Aging step (min)", sched.get("aging_step_min", "-")),
        ("CPU tokens",  f'{slot.get("cpu_in_use", "-")}/{slot.get("cpu_total", "-")}'),
        ("Net tokens",  f'{slot.get("net_in_use", "-")}/{slot.get("net_total", "-")}'),
    ])
    sched_panel += "<h3>Queue by priority</h3>"
    sched_panel += _render_dict_table(
        sched.get("queue_by_priority") or {},
        key_label="priority", value_label="count",
    )
    sched_panel += "<h3>Reserved capacity</h3>"
    sched_panel += _render_dict_table(
        sched.get("reserved_capacity_pct") or {},
        key_label="priority", value_label="pct",
    )

    # Branding cache
    hits = int(branding.get("hits", 0))
    misses = int(branding.get("misses", 0))
    total = hits + misses
    hit_ratio = (hits / total * 100.0) if total > 0 else 0.0
    branding_panel = _render_kv_table([
        ("In-flight", branding.get("in_flight", 0)),
        ("Hits", hits),
        ("Misses", misses),
        ("Hit ratio %", f"{hit_ratio:.2f}"),
        ("ffmpeg invocations", branding.get("ffmpeg_invocations", 0)),
        ("Last cleanup at", branding.get("last_cleanup_at", "-")),
        ("Last cleanup removed", branding.get("last_cleanup_removed_count", 0)),
        ("Last ffmpeg duration (s)", branding.get("last_ffmpeg_duration_seconds", "-")),
    ])

    # Burn ledger summary
    burn_predicted = int(burn.get("predicted", 0))
    burn_actual = int(burn.get("actual", 0))
    burn_delta = burn_actual - burn_predicted
    burn_panel = _render_kv_table([
        ("Window (min)", burn.get("window_minutes", "-")),
        ("Rows total", burn.get("rows_total", 0)),
        ("Rows reconciled", burn.get("rows_reconciled", 0)),
        ("Predicted units", f"{burn_predicted:,}"),
        ("Actual units", f"{burn_actual:,}"),
        ("Delta", f"{burn_delta:+,}"),
    ])
    burn_panel += "<h3>By outcome (24h)</h3>"
    burn_panel += _render_dict_table(burn.get("by_outcome") or {})
    burn_panel += "<h3>Last 20 rows</h3>"
    burn_panel += _render_burn_rows(burn_rows)

    # Credit health
    credit_panel = "<h3>Top 10 by balance</h3>"
    credit_panel += _render_credit_rows(credit_users)
    credit_panel += "<h3>Users by plan tier</h3>"
    credit_panel += _render_dict_table(plan_dist, key_label="plan_tier", value_label="user_count")

    # Idempotency
    attempts_panel = _render_kv_table([
        ("Orphans recovered (total)", orphan_count),
        ("Last 10 attempts", ""),
    ])
    attempts_panel += _render_attempts_rows(attempts)

    # Status overview
    status_panel = "<h3>PublishTask status</h3>"
    status_panel += _render_dict_table(publish_status_counts, key_label="status", value_label="count")
    status_panel += "<h3>UploadJobV2 status</h3>"
    status_panel += _render_dict_table(upload_status_counts, key_label="status", value_label="count")

    # Durable queue (Wave 2) — only rendered when KAIZER_DURABLE_QUEUE=1
    durable_card = ""
    if durable is not None:
        if "error" in durable:
            durable_panel = f'<p class="muted">{_esc(durable["error"])}</p>'
        else:
            durable_panel = _render_kv_table([
                ("Queued (total)", durable.get("queued_total", 0)),
                ("Active", durable.get("active_total", 0)),
                ("Parked (quota)", durable.get("parked_quota", 0)),
                ("Retrying (backoff)", durable.get("retrying", 0)),
            ])
            durable_panel += "<h3>Queued by priority</h3>"
            durable_panel += _render_dict_table(
                durable.get("queued_by_priority") or {},
                key_label="priority", value_label="count",
            )
        durable_card = (
            '<div class="card">\n'
            '      <h2>Durable queue</h2>\n'
            f'      {durable_panel}\n'
            '    </div>'
        )

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Kaizer — Upload v2 Admin</title>
  <meta http-equiv="refresh" content="5">
  <style>{_CSS}</style>
</head>
<body>
  <h1>Kaizer — Publish/Upload v2 — Admin Observability</h1>
  <div class="meta">
    Auto-refresh every 5s &nbsp;·&nbsp; Server time (UTC): {_esc(now_iso)} &nbsp;·&nbsp;
    See also: <a href="/metrics">/metrics</a> (Prometheus) &nbsp;·&nbsp;
    <a href="/admin/work-monitor">/admin/work-monitor</a>
    <div style="margin-top:6px">{flag_html}</div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Scheduler</h2>
      {sched_panel}
    </div>
    {durable_card}
    <div class="card">
      <h2>Status overview</h2>
      {status_panel}
    </div>
    <div class="card">
      <h2>Quota gate (today)</h2>
      {quota_panel}
      <h3>By api_key_hash</h3>
      {by_key_panel}
    </div>
    <div class="card">
      <h2>Branding cache</h2>
      {branding_panel}
    </div>
    <div class="card" style="grid-column: 1 / -1">
      <h2>Burn-rate ledger (last 24h)</h2>
      {burn_panel}
    </div>
    <div class="card">
      <h2>Credit health</h2>
      {credit_panel}
    </div>
    <div class="card">
      <h2>Idempotency / Recovery</h2>
      {attempts_panel}
    </div>
  </div>
</body>
</html>
"""


# ─── Endpoint ───────────────────────────────────────────────────────────────


@router.get(
    "/admin/upload-v2",
    response_class=HTMLResponse,
    summary="Admin observability page for the v2 publish/upload pipeline",
)
def admin_upload_v2_page(
    db: Session = Depends(get_db),
    _user: models.User = Depends(auth.admin_required),
) -> HTMLResponse:
    """Server-rendered HTML page for ops. Auth: admin_required.

    Defensive: every snapshot pull is wrapped in try/except so a single
    broken sub-collector doesn't 500 the whole page. Errors surface as
    ``{"error": "..."}`` dicts the render code displays inline.
    """
    flags = {
        "KAIZER_NEW_PUBLISH_PATH": _flag("KAIZER_NEW_PUBLISH_PATH", "0"),
        "KAIZER_DURABLE_QUEUE": _flag("KAIZER_DURABLE_QUEUE", "0"),
        "KAIZER_CLEAN_MASTER": _flag("KAIZER_CLEAN_MASTER", "0"),
        "KAIZER_FREE_DIRECT_BLOCKED": _flag("KAIZER_FREE_DIRECT_BLOCKED", "1"),
        "KAIZER_YT_DAILY_QUOTA_CAP": _flag("KAIZER_YT_DAILY_QUOTA_CAP", "1000000"),
        "KAIZER_BRANDED_ARTIFACT_TTL_HOURS": _flag("KAIZER_BRANDED_ARTIFACT_TTL_HOURS", "24"),
        "KAIZER_PRIORITY_AGING_MIN": _flag("KAIZER_PRIORITY_AGING_MIN", "5"),
    }

    sched = _scheduler_snapshot()
    branding = _branding_snapshot()
    burn = _burn_snapshot(db)
    burn_rows = _burn_recent(db, limit=20)
    quota = _quota_today(db)
    credit_users = _credit_top_users(db, limit=10)
    plan_dist = _plan_tier_distribution(db)
    attempts = _attempts_recent(db, limit=10)
    orphan_count = _orphan_count(db)
    publish_status_counts = _publish_task_status_counts(db)
    upload_status_counts = _upload_job_status_counts(db)
    durable = _durable_queue_snapshot(db) if _durable_queue_enabled() else None

    page = _render_page(
        flags=flags,
        sched=sched,
        branding=branding,
        burn=burn,
        burn_rows=burn_rows,
        quota=quota,
        credit_users=credit_users,
        plan_dist=plan_dist,
        attempts=attempts,
        orphan_count=orphan_count,
        publish_status_counts=publish_status_counts,
        upload_status_counts=upload_status_counts,
        durable=durable,
    )
    return HTMLResponse(page)
