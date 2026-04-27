"""Shared helper: resolve a UserAsset to a filesystem path the local
process can read.

Why this exists
---------------
A UserAsset row stores ``file_path`` as the absolute on-disk location
where the bytes were originally written. That works on the host that
did the upload but breaks anywhere else: the user's local Windows box
has files at ``e:\\...``, Railway containers have files at
``/app/...``, and a fresh Railway redeploy doesn't even have the prior
container's ``/app/`` files.

After the R2 image migration every UserAsset also has ``storage_key``
+ ``storage_backend`` populated. This helper closes the loop: when
file_path doesn't exist locally, it pulls the bytes from R2 into a
tempfile and returns that path so callers can hand it to ffmpeg /
the pipeline subprocess unchanged.

Failure mode is empty-string return; callers must already have a
fallback (Pexels stock photos, no-logo upload, etc.) so we never
raise.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional


def materialize_asset_locally(asset) -> str:
    """Return an absolute filesystem path for *asset* the current
    process can open.

    Parameters
    ----------
    asset : models.UserAsset | None
        The asset row. Pass None to short-circuit to the empty string.

    Returns
    -------
    str
        Absolute path, or empty string when the asset has no usable
        bytes anywhere (no local file AND no R2 storage_key, or R2
        download failed).
    """
    if asset is None:
        return ""

    fp = getattr(asset, "file_path", "") or ""
    if fp and Path(fp).exists():
        return fp

    storage_key = getattr(asset, "storage_key", "") or ""
    storage_backend = getattr(asset, "storage_backend", "") or ""
    if not (storage_key and storage_backend):
        return ""

    try:
        from pipeline_core.storage import get_storage_provider
        provider = get_storage_provider(storage_backend)
        tmp_dir = tempfile.mkdtemp(prefix="kaizer_asset_")
        filename = Path(storage_key).name or f"asset_{getattr(asset, 'id', 'x')}"
        tmp_path = str(Path(tmp_dir) / filename)
        provider.download(storage_key, tmp_path)
        return tmp_path
    except Exception as exc:
        # Tempfiles are leaked here — fine on Railway (container is
        # ephemeral) and on Windows users will reboot eventually.
        print(
            f"[asset_resolver] failed to fetch asset "
            f"{getattr(asset, 'id', '?')} from "
            f"{storage_backend}/{storage_key!r}: {exc}"
        )
        return ""
