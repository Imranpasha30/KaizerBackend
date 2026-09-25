"""routers.desktop_local — desktop-only local control surface.

Mounted by main.py ONLY when KAIZER_DESKTOP=1 (the SaaS never exposes
these routes). The desktop backend binds to 127.0.0.1 for a single local
user, so there is deliberately NO auth dependency here — the transport
itself is the boundary. Keep it that way: any endpoint added to this
router must stay safe to expose to "whoever can reach localhost".

Surface
-------
GET  /api/desktop-local/keys       — which AI keys are saved (masked last-4,
                                     never the full value).
POST /api/desktop-local/keys       — save/update/clear ONE whitelisted key.
                                     Writes the userData .env (KAIZER_ENV_DIR)
                                     AND os.environ so both the running
                                     process and future render subprocesses
                                     (which inherit os.environ) see it.
GET  /api/desktop-local/preflight  — per-feature readiness with honest,
                                     plain-language reasons.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

router = APIRouter(prefix="/api/desktop-local", tags=["desktop-local"])


class SPAStaticFiles(StaticFiles):
    """StaticFiles with an SPA fallback (desktop "/" mount in main.py).

    Client-side routes (/app, /jobs/5 ...) have no file on disk — a hard
    reload or deep link would 404. Any 404 whose path is NOT under api/ or
    media/ serves index.html instead so the SPA router can take over.
    api/ and media/ 404s stay REAL 404s (JSON), never a misleading HTML
    page. Real static files are always served normally."""

    _passthrough_prefixes = ("api", "media")

    def _spa_fallback_ok(self, path: str) -> bool:
        # StaticFiles.get_path hands us OS-specific separators ("api\\nope"
        # on Windows) — normalize before taking the first segment.
        norm = (path or "").replace("\\", "/").lstrip("/")
        head = norm.split("/", 1)[0].lower()
        return head not in self._passthrough_prefixes

    async def get_response(self, path, scope):
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            # Installed Starlette RAISES 404 for missing files.
            if exc.status_code == 404 and self._spa_fallback_ok(path):
                return await super().get_response("index.html", scope)
            raise
        # Older/other Starlette versions RETURN a 404 response instead.
        if getattr(response, "status_code", 200) == 404 and self._spa_fallback_ok(path):
            return await super().get_response("index.html", scope)
        return response


# The ONLY env names this surface may read or write. Everything else
# (DATABASE_URL, KAIZER_* toggles...) is owned by desktop_entry/config.
KEY_WHITELIST = (
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPGRAM_API_KEY",
    "KAIZER_V4_TRIM_PLANNER",
)

_KEY_LABELS = {
    "GEMINI_API_KEY":        "Google Gemini",
    "OPENAI_API_KEY":        "OpenAI",
    "ANTHROPIC_API_KEY":     "Anthropic Claude",
    "DEEPGRAM_API_KEY":      "Deepgram",
    "KAIZER_V4_TRIM_PLANNER": "Trim planner (which AI plans the cuts)",
}

# Plain SETTINGS (not secrets): returned by GET unmasked, and their value
# must come from the fixed choice list below. Everything else in
# KEY_WHITELIST is a secret API key (masked to last-4 on GET).
SETTING_VALUES = {
    "KAIZER_V4_TRIM_PLANNER": ("gemini", "claude"),
}

# Longest value we ever accept. Also: \r/\n are STRIPPED before writing —
# a newline inside a value would smuggle a second NAME=value line into the
# .env and bypass the whitelist entirely (.env injection).
_MAX_VALUE_LEN = 4096


def _sanitize_value(raw: str) -> str:
    cleaned = (raw or "").replace("\r", "").replace("\n", "").replace("\x00", "")
    return cleaned.strip()[:_MAX_VALUE_LEN]


def _env_path() -> Path:
    """The userData .env — resolved from KAIZER_ENV_DIR at call time (not
    import time) so it always matches the running desktop's data dir and
    stays testable with a tmp dir. Same rule config.py uses for ENV_PATH."""
    base = (os.environ.get("KAIZER_ENV_DIR", "") or "").strip()
    if base:
        return Path(base) / ".env"
    from config import ENV_PATH
    return ENV_PATH


def _get_key(name: str) -> str:
    return (os.environ.get(name, "") or "").strip()


def _mask(value: str) -> str:
    """Last 4 characters only — enough to recognize a key, never enough
    to use it."""
    if not value:
        return ""
    return "****" + value[-4:]


def _write_env_key(path: Path, name: str, value: str) -> None:
    """Set (or remove, when value is empty) NAME=value in the .env file.
    Preserves every other line; appends when the key is new."""
    lines: List[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    replaced = False
    for line in lines:
        if line.strip().startswith(f"{name}="):
            if value and not replaced:
                out.append(f"{name}={value}")
                replaced = True
            # empty value (or a duplicate line) → drop the line
            continue
        out.append(line)
    if value and not replaced:
        out.append(f"{name}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(out)
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


# ── Schemas ──────────────────────────────────────────────────────────────────

class KeyUpdate(BaseModel):
    name: str = Field(..., description="One of the supported key names.")
    value: str = Field(default="", max_length=_MAX_VALUE_LEN,
                       description="The key value. Empty = remove the key.")


# ── Endpoints ────────────────────────────────────────────────────────────────

def _row(name: str) -> Dict:
    """One GET row. Secrets are masked to last-4; plain settings (e.g. the
    trim-planner choice) are returned as-is — they aren't secret and the
    UI needs the real value to show the current choice."""
    value = _get_key(name)
    secret = name not in SETTING_VALUES
    row = {
        "name": name,
        "label": _KEY_LABELS.get(name, name),
        "set": bool(value),
        "masked": _mask(value) if secret else value,
        "secret": secret,
    }
    if not secret:
        row["value"] = value
        row["choices"] = list(SETTING_VALUES[name])
    return row


@router.get("/keys")
def list_keys() -> Dict:
    """Masked status of every supported AI key (+ plain settings)."""
    return {
        "keys": [_row(name) for name in KEY_WHITELIST],
        "env_file": str(_env_path()),
    }


@router.post("/keys")
def save_key(body: KeyUpdate) -> Dict:
    """Save one key to the userData .env AND the running process's
    environment (render subprocesses inherit os.environ, so new renders
    pick it up without a restart)."""
    name = (body.name or "").strip().upper()
    if name not in KEY_WHITELIST:
        supported = ", ".join(_KEY_LABELS[k] for k in KEY_WHITELIST)
        raise HTTPException(
            status_code=400,
            detail=f"That setting isn't one Kaizer X knows. "
                   f"Supported keys: {supported}.",
        )
    value = _sanitize_value(body.value)
    if name in SETTING_VALUES:
        value = value.lower()
        allowed = SETTING_VALUES[name]
        if value and value not in allowed:
            choices = " or ".join(allowed)
            raise HTTPException(
                status_code=400,
                detail=f"That isn't a choice Kaizer X understands for "
                       f"{_KEY_LABELS.get(name, name)}. Please pick "
                       f"{choices}.",
            )
    try:
        _write_env_key(_env_path(), name, value)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Couldn't save the key to your settings file. "
                   f"({exc}) Please check the app's data folder is writable.",
        )
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)
    if name in SETTING_VALUES:
        # Plain setting — echo the real value (nothing secret to hide).
        return {"name": name, "set": bool(value), "masked": value,
                "value": value}
    return {"name": name, "set": bool(value), "masked": _mask(value)}


@router.get("/preflight")
def preflight() -> Dict:
    """Per-feature readiness, with honest reasons a non-technical user can
    act on. The frontend shows these before letting a feature start."""
    gemini = bool(_get_key("GEMINI_API_KEY"))
    openai = bool(_get_key("OPENAI_API_KEY"))
    anthropic = bool(_get_key("ANTHROPIC_API_KEY"))
    deepgram = bool(_get_key("DEEPGRAM_API_KEY"))

    features = {
        "render": {
            "ready": gemini,
            "reason": "" if gemini else
                "Video rendering needs a Google Gemini API key. "
                "Add it in Settings to start rendering.",
        },
        "seo": {
            "ready": gemini,
            "reason": "" if gemini else
                "Writing titles and descriptions needs a Google Gemini "
                "API key. Add it in Settings.",
        },
        "image_generation": {
            "ready": gemini or openai,
            "reason": "" if (gemini or openai) else
                "Generating images needs either a Google Gemini or an "
                "OpenAI API key. Add one in Settings.",
        },
        "transcription": {
            "ready": deepgram,
            "reason": "" if deepgram else
                "Transcription (and the podcast editor) needs a Deepgram "
                "API key. Add it in Settings.",
        },
        "trim_planner": {
            # Claude is optional: without it the trim planner runs on
            # Gemini (runner defaults it there in desktop mode — an empty
            # Anthropic key would otherwise fail the render at Stage 1).
            "ready": gemini or anthropic,
            "claude_available": anthropic,
            "recommended": "claude" if anthropic else "gemini",
            "reason": "" if anthropic else
                "The Claude trim planner is unavailable without an "
                "Anthropic API key — Kaizer X will use Gemini instead, "
                "which works well.",
        },
    }
    return {"features": features, "ready": all(f["ready"] for f in features.values())}
