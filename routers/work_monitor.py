"""Work Monitor — live dashboard for watching Claude / agents work.

Three surfaces:
  - GET  /admin/work-monitor       → static HTML dashboard
  - WS   /ws/work-log               → broadcasts events as they're posted
  - POST /api/work-log/event        → append an event (used by Claude / agents)
  - POST /api/work-log/note         → save / append to the notes file
  - GET  /api/work-log/recent       → last N events on first page-load

Events are persisted as JSONL so they survive a backend restart. The
WebSocket fan-out is in-memory; new connections receive the last 200
events on connect (replay) then live new ones.

Auth: localhost-only by default (relies on Cloudflare Tunnel + admin
auth at the edge). Add the `current_admin` dep when this is exposed
publicly.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

router = APIRouter(tags=["work-monitor"])

# ─── Paths ───────────────────────────────────────────────────────────────────
_LOG_DIR = Path(os.environ.get("KAIZER_WORK_LOG_DIR", "logs"))
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_EVENTS_PATH = _LOG_DIR / "work_log.jsonl"
_NOTES_PATH  = _LOG_DIR / "work_notes.md"

# ─── In-memory pub/sub for WebSocket fan-out ─────────────────────────────────
_subscribers: set[WebSocket] = set()
_subscribers_lock = asyncio.Lock()


# ─── Models ──────────────────────────────────────────────────────────────────

class WorkEvent(BaseModel):
    kind: str = "log"        # log | task_start | task_done | error | commit | note | agent
    title: str = ""
    body: str = ""
    tag: str = ""            # short label, e.g. "issue-2", "agent-A"
    level: str = "info"      # info | warn | error | success


class NoteUpdate(BaseModel):
    text: str


# ─── Persistence helpers ─────────────────────────────────────────────────────

def _append_event(evt: dict) -> None:
    evt["ts"] = evt.get("ts") or datetime.now(timezone.utc).isoformat()
    with open(_EVENTS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(evt, ensure_ascii=False) + "\n")


def _load_recent(limit: int = 200) -> list[dict]:
    if not _EVENTS_PATH.exists():
        return []
    try:
        with open(_EVENTS_PATH, "r", encoding="utf-8") as fh:
            lines = fh.readlines()[-limit:]
        out: list[dict] = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out
    except Exception:
        return []


async def _broadcast(evt: dict) -> None:
    """Push an event to every connected WebSocket. Drops dead ones."""
    payload = json.dumps(evt, ensure_ascii=False)
    dead: list[WebSocket] = []
    async with _subscribers_lock:
        for ws in list(_subscribers):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _subscribers.discard(ws)


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/api/work-log/event")
async def post_event(evt: WorkEvent) -> dict:
    payload = evt.model_dump()
    _append_event(payload)
    await _broadcast(payload)
    return {"ok": True}


@router.get("/api/work-log/recent")
async def recent(limit: int = 200) -> JSONResponse:
    return JSONResponse(_load_recent(min(max(1, limit), 1000)))


@router.post("/api/work-log/note")
async def save_note(note: NoteUpdate) -> dict:
    """Append a note to work_notes.md with a UTC timestamp."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    line = f"\n## {stamp}\n\n{note.text.strip()}\n"
    with open(_NOTES_PATH, "a", encoding="utf-8") as fh:
        fh.write(line)
    # Echo as an event so the live UI shows it.
    payload = {
        "kind": "note", "title": "Note saved",
        "body": note.text.strip()[:400], "tag": "user-note",
        "level": "info",
    }
    _append_event(payload)
    await _broadcast(payload)
    return {"ok": True}


@router.get("/api/work-log/notes")
async def get_notes() -> JSONResponse:
    txt = _NOTES_PATH.read_text(encoding="utf-8") if _NOTES_PATH.exists() else ""
    return JSONResponse({"notes": txt})


@router.websocket("/ws/work-log")
async def ws_work_log(ws: WebSocket) -> None:
    await ws.accept()
    # Send last 200 events on connect so a freshly-opened tab has context.
    for evt in _load_recent(200):
        try:
            await ws.send_text(json.dumps(evt, ensure_ascii=False))
        except Exception:
            await ws.close()
            return

    async with _subscribers_lock:
        _subscribers.add(ws)
    try:
        while True:
            # Keep the connection alive; ignore any client messages.
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Heartbeat ping so proxies don't drop idle connections.
                await ws.send_text(json.dumps({"kind": "ping",
                                                "ts": datetime.now(timezone.utc).isoformat()}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        async with _subscribers_lock:
            _subscribers.discard(ws)


# ─── Static dashboard ────────────────────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Kaizer — Work Monitor</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root {
    --bg:#0b0d12; --panel:#13161d; --panel-2:#191d27; --line:#262b38;
    --txt:#e6e8ee; --mut:#8b91a3; --red:#e02538; --grn:#22c55e;
    --ylw:#facc15; --blu:#60a5fa; --org:#fb923c;
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0;background:var(--bg);color:var(--txt);
    font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  header{padding:14px 20px;background:var(--panel);
    border-bottom:1px solid var(--line);
    display:flex;align-items:center;gap:14px}
  .brand{font-weight:900;letter-spacing:.6px;color:var(--red)}
  .badge{font-size:11px;background:var(--panel-2);border:1px solid var(--line);
    border-radius:999px;padding:2px 10px;color:var(--mut)}
  #status{margin-left:auto;font-size:12px;color:var(--mut)}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;
    margin-right:6px;background:#888;vertical-align:middle}
  .dot.ok{background:var(--grn);box-shadow:0 0 8px rgba(34,197,94,.6)}
  .dot.bad{background:var(--red)}
  main{display:grid;grid-template-columns: 1fr 360px;
    gap:14px;padding:14px;height:calc(100vh - 53px)}
  section{background:var(--panel);border:1px solid var(--line);
    border-radius:10px;display:flex;flex-direction:column;overflow:hidden}
  section h2{margin:0;padding:11px 16px;border-bottom:1px solid var(--line);
    font-size:12px;letter-spacing:.6px;text-transform:uppercase;color:var(--mut)}
  #events{flex:1;overflow-y:auto;padding:6px}
  .ev{padding:8px 12px;border-radius:8px;margin:6px 0;
    background:var(--panel-2);border-left:3px solid var(--blu)}
  .ev.l-success{border-color:var(--grn)}
  .ev.l-warn{border-color:var(--ylw)}
  .ev.l-error{border-color:var(--red)}
  .ev.k-commit{border-color:var(--org)}
  .ev.k-task_start{border-color:var(--blu)}
  .ev.k-task_done{border-color:var(--grn)}
  .ev.k-note{border-color:#a78bfa}
  .ev .meta{font-size:11px;color:var(--mut);
    display:flex;gap:8px;margin-bottom:3px}
  .ev .ttl{font-weight:600;color:var(--txt)}
  .ev .body{color:#cdd1dc;white-space:pre-wrap;
    word-break:break-word;font-size:13px;
    margin-top:4px;font-family:ui-monospace,Menlo,Consolas,monospace}
  .ev .tag{color:#a78bfa}
  /* Right column */
  .col-right{display:flex;flex-direction:column;gap:14px;min-height:0}
  .col-right section{flex:1}
  #notes-text{flex:1;background:var(--panel-2);color:var(--txt);
    border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px;
    resize:none;font:13px ui-monospace,Menlo,Consolas,monospace;outline:none}
  #notes-text:focus{border-color:var(--blu)}
  #notes-bar{padding:10px 12px;border-top:1px solid var(--line);
    display:flex;gap:10px;align-items:center;font-size:12px;color:var(--mut)}
  #save-btn{background:var(--red);color:#fff;border:0;padding:7px 14px;
    border-radius:6px;font-weight:700;cursor:pointer;letter-spacing:.4px}
  #save-btn:hover{filter:brightness(1.1)}
  #save-btn:disabled{opacity:.5;cursor:default}
  #stats{padding:12px 16px;display:grid;grid-template-columns:1fr 1fr;
    gap:10px;font-size:12px}
  #stats div{background:var(--panel-2);border:1px solid var(--line);
    padding:8px 12px;border-radius:6px}
  #stats b{color:var(--txt);font-size:18px;display:block;margin-top:2px}
  #filter{margin:8px;display:flex;gap:6px}
  #filter button{background:var(--panel-2);border:1px solid var(--line);
    color:var(--mut);padding:4px 10px;border-radius:5px;cursor:pointer;
    font-size:11px;letter-spacing:.4px}
  #filter button.on{background:var(--red);color:#fff;border-color:var(--red)}
  ::-webkit-scrollbar{width:8px;height:8px}
  ::-webkit-scrollbar-thumb{background:#2a3040;border-radius:8px}
</style>
</head>
<body>
<header>
  <span class="brand">KAIZER · WORK MONITOR</span>
  <span class="badge">live</span>
  <span id="status"><span class="dot" id="conn-dot"></span><span id="conn-txt">connecting…</span></span>
</header>
<main>
  <section>
    <h2>Live event stream</h2>
    <div id="filter">
      <button data-f="all" class="on">All</button>
      <button data-f="task_start">Tasks</button>
      <button data-f="commit">Commits</button>
      <button data-f="error">Errors</button>
      <button data-f="agent">Agents</button>
      <button data-f="note">Notes</button>
    </div>
    <div id="events"></div>
  </section>
  <div class="col-right">
    <section>
      <h2>Stats</h2>
      <div id="stats">
        <div>Events <b id="s-events">0</b></div>
        <div>Errors <b id="s-errors">0</b></div>
        <div>Commits <b id="s-commits">0</b></div>
        <div>Last <b id="s-last">—</b></div>
      </div>
    </section>
    <section style="flex:2">
      <h2>Your notes (issues / requests)</h2>
      <textarea id="notes-text" placeholder="Type any issue you see — saved as you type. Press Cmd/Ctrl+S to flush a snapshot."></textarea>
      <div id="notes-bar">
        <span>Auto-saves every 5 s · also saved on Cmd/Ctrl+S</span>
        <button id="save-btn" style="margin-left:auto">Save snapshot</button>
      </div>
    </section>
  </div>
</main>

<script>
const evDiv = document.getElementById('events');
const sEvents = document.getElementById('s-events');
const sErrors = document.getElementById('s-errors');
const sCommits = document.getElementById('s-commits');
const sLast = document.getElementById('s-last');
const connDot = document.getElementById('conn-dot');
const connTxt = document.getElementById('conn-txt');
let curFilter = 'all';
let evCount = 0, errCount = 0, commitCount = 0;

function fmtTs(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
}

function render(evt) {
  if (evt.kind === 'ping') return;
  evCount++; sEvents.textContent = evCount;
  if (evt.level === 'error' || evt.kind === 'error') { errCount++; sErrors.textContent = errCount; }
  if (evt.kind === 'commit') { commitCount++; sCommits.textContent = commitCount; }
  sLast.textContent = fmtTs(evt.ts);

  if (curFilter !== 'all' && evt.kind !== curFilter) return;

  const div = document.createElement('div');
  div.className = `ev k-${evt.kind || 'log'} l-${evt.level || 'info'}`;
  div.innerHTML = `
    <div class="meta">
      <span>${fmtTs(evt.ts)}</span>
      ${evt.tag ? `<span class="tag">#${evt.tag}</span>` : ''}
      <span style="opacity:.5">${evt.kind || 'log'}</span>
    </div>
    ${evt.title ? `<div class="ttl">${escapeHtml(evt.title)}</div>` : ''}
    ${evt.body ? `<div class="body">${escapeHtml(evt.body)}</div>` : ''}
  `;
  evDiv.prepend(div);
  // Cap to 500 in DOM
  while (evDiv.childElementCount > 500) evDiv.lastChild.remove();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// ── WebSocket with auto-reconnect ──
let ws = null;
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${location.host}/ws/work-log`;
  ws = new WebSocket(url);
  ws.onopen = () => { connDot.className = 'dot ok'; connTxt.textContent = 'live'; };
  ws.onmessage = ev => {
    try { render(JSON.parse(ev.data)); } catch(_) {}
  };
  ws.onclose = () => {
    connDot.className = 'dot bad'; connTxt.textContent = 'reconnecting…';
    setTimeout(connect, 2000);
  };
  ws.onerror = () => { try { ws.close(); } catch(_){}};
}
connect();

// ── Filter ──
document.querySelectorAll('#filter button').forEach(b => {
  b.onclick = () => {
    document.querySelectorAll('#filter button').forEach(x => x.classList.remove('on'));
    b.classList.add('on');
    curFilter = b.dataset.f;
    evDiv.innerHTML = '';
    fetch('/api/work-log/recent?limit=200').then(r => r.json()).then(arr => {
      arr.forEach(render);
    });
  };
});

// ── Notes auto-save ──
const notes = document.getElementById('notes-text');
const saveBtn = document.getElementById('save-btn');
let lastSent = '';
async function saveNotes(force=false) {
  const txt = notes.value;
  if (!force && txt === lastSent) return;
  if (!txt.trim()) return;
  saveBtn.disabled = true;
  try {
    await fetch('/api/work-log/note', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text: txt})
    });
    lastSent = txt;
  } finally { saveBtn.disabled = false; }
}
setInterval(saveNotes, 5000);
saveBtn.onclick = () => saveNotes(true);
notes.addEventListener('keydown', ev => {
  if ((ev.ctrlKey||ev.metaKey) && ev.key === 's') {
    ev.preventDefault();
    saveNotes(true);
  }
});

// ── Initial replay ──
fetch('/api/work-log/recent?limit=200').then(r => r.json()).then(arr => arr.forEach(render));
</script>
</body>
</html>"""


@router.get("/admin/work-monitor", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(_DASHBOARD_HTML)
