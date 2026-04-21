"""Centralized settings — loads from .env on import.

Uses plain os.environ to avoid a hard dependency on pydantic-settings when it's
not installed yet. Settings are lazily materialized so the module never raises
on import.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"


def _get(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


class Settings:
    """Runtime configuration. Values are read from env every access so
    hot-reloading after editing .env works without a process restart."""

    # ─── Core ──────────────────────────────────────────────────────────
    @property
    def gemini_api_key(self) -> str:
        return _get("GEMINI_API_KEY")

    @property
    def openai_api_key(self) -> str:
        return _get("OPENAI_API_KEY")

    @property
    def output_root(self) -> Path:
        return Path(_get("KAIZER_OUTPUT_ROOT", str(BASE_DIR / "output")))

    # ─── YouTube OAuth (Phase 4) ───────────────────────────────────────
    @property
    def yt_client_id(self) -> str:
        return _get("YOUTUBE_CLIENT_ID")

    @property
    def yt_client_secret(self) -> str:
        return _get("YOUTUBE_CLIENT_SECRET")

    @property
    def yt_redirect_uri(self) -> str:
        return _get("YOUTUBE_REDIRECT_URI", "http://localhost:8000/api/youtube/oauth/callback")

    @property
    def yt_data_api_key(self) -> str:
        """Public API key (no OAuth) used for learning-corpus / competitor queries."""
        return _get("YOUTUBE_DATA_API_KEY")

    @property
    def yt_oauth_configured(self) -> bool:
        return bool(self.yt_client_id and self.yt_client_secret)

    # ─── Encryption (Phase 4) ──────────────────────────────────────────
    @property
    def encryption_key(self) -> str:
        """Fernet key (32-byte urlsafe base64). Auto-generated on first boot."""
        key = _get("KAIZER_ENCRYPTION_KEY")
        if not key:
            key = self._provision_encryption_key()
        return key

    def _provision_encryption_key(self) -> str:
        """Generate a fresh Fernet key + append to .env. Idempotent — if another
        worker raced us and wrote one first, we use that instead."""
        from cryptography.fernet import Fernet

        # Re-read from file in case another process wrote it
        if ENV_PATH.exists():
            existing = ENV_PATH.read_text(encoding="utf-8")
            for line in existing.splitlines():
                if line.startswith("KAIZER_ENCRYPTION_KEY="):
                    val = line.split("=", 1)[1].strip()
                    if val:
                        os.environ["KAIZER_ENCRYPTION_KEY"] = val
                        return val

        fresh = Fernet.generate_key().decode("ascii")
        os.environ["KAIZER_ENCRYPTION_KEY"] = fresh

        try:
            with ENV_PATH.open("a", encoding="utf-8") as f:
                if ENV_PATH.stat().st_size > 0:
                    f.seek(0, 2)
                    f.write("\n")
                f.write(f"KAIZER_ENCRYPTION_KEY={fresh}\n")
            print(f"[config] Generated new KAIZER_ENCRYPTION_KEY and appended to {ENV_PATH.name}")
        except OSError as e:
            print(f"[config] WARN: Could not persist encryption key to .env: {e}. "
                  "Set KAIZER_ENCRYPTION_KEY manually to keep refresh tokens readable across restarts.")

        return fresh


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
