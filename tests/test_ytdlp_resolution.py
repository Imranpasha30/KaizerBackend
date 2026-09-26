"""yt-dlp must be found without relying on the child process's PATH.

FROM THE PRODUCTION LOG (2026-09-26), after the broadcast had already been
minted on the customer's real YouTube channel:

    err=ffmpeg push failed: yt-dlp not on PATH:
        [Errno 2] No such file or directory: 'yt-dlp'

Two faults sat behind that line and both are covered here: the package was
never declared in the requirements file the production deploy builds from,
and even when installed it was invoked by bare name, which resolves only if
the venv's bin directory happens to be exported to the child.

The important case is test_falls_back_to_the_module_runner: the console
script absent but the package importable is EXACTLY the production state.
"""
import pathlib
import sys

import pytest

from live_studio import streamer, url_ingest

MODULES = [streamer, url_ingest]
IDS = ["streamer", "url_ingest"]


@pytest.fixture(autouse=True)
def _no_inherited_override(monkeypatch):
    """This machine may set the override; the tests must not read it."""
    monkeypatch.delenv("YTDLP_BIN", raising=False)


@pytest.mark.parametrize("mod", MODULES, ids=IDS)
def test_an_explicit_override_wins(mod, monkeypatch):
    monkeypatch.setenv("YTDLP_BIN", "/opt/custom/yt-dlp")
    assert mod._resolve_ytdlp() == ["/opt/custom/yt-dlp"]


@pytest.mark.parametrize("mod", MODULES, ids=IDS)
def test_path_is_used_when_it_has_it(mod, monkeypatch):
    monkeypatch.setattr(mod._shutil, "which",
                        lambda n: "/usr/local/bin/yt-dlp" if n == "yt-dlp" else None)
    assert mod._resolve_ytdlp() == ["/usr/local/bin/yt-dlp"]


@pytest.mark.parametrize("mod", MODULES, ids=IDS)
def test_falls_back_to_the_module_runner(mod, monkeypatch):
    """THE PRODUCTION CASE: no console script on PATH, package installed.

    A buildpack deploy does not export the venv's bin directory to child
    processes, so `yt-dlp` is unfindable while `import yt_dlp` works fine.
    """
    monkeypatch.setattr(mod._shutil, "which", lambda n: None)
    assert mod._resolve_ytdlp() == [sys.executable, "-m", "yt_dlp"]


@pytest.mark.parametrize("mod", MODULES, ids=IDS)
def test_truly_absent_keeps_a_nameable_failure(mod, monkeypatch):
    """With nothing installed the bare name is right: the resulting error
    says 'yt-dlp', which is the thing an operator has to go install."""
    import importlib.util
    monkeypatch.setattr(mod._shutil, "which", lambda n: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda n: None)
    assert mod._resolve_ytdlp() == ["yt-dlp"]


@pytest.mark.parametrize("mod", MODULES, ids=IDS)
def test_resolution_survives_a_broken_importlib(mod, monkeypatch):
    """Resolution runs at import time; raising here would take the whole
    backend down rather than fail one broadcast."""
    import importlib.util
    monkeypatch.setattr(mod._shutil, "which", lambda n: None)
    def _boom(_n): raise RuntimeError("import machinery is unhappy")
    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    assert mod._resolve_ytdlp() == ["yt-dlp"]


@pytest.mark.parametrize("mod", MODULES, ids=IDS)
def test_the_module_actually_uses_the_resolved_command(mod):
    """A resolver nothing calls is the defect, not the fix."""
    import inspect
    src = inspect.getsource(mod)
    assert "*_YTDLP_CMD," in src, "the args list still hard-codes a bare name"
    assert '        "yt-dlp",\n' not in src


def test_the_error_names_the_command_and_the_remedy():
    import inspect
    src = inspect.getsource(streamer)
    assert "pip install yt-dlp" in src, "the failure should say how to fix it"


# ── the packaging half of the same bug ───────────────────────────────

def test_every_requirements_file_declares_ytdlp():
    """requirements.txt pinned it; requirements-railway.txt -- which the
    Procfile/runtime.txt buildpack deploy builds from -- did not, so the
    binary was simply absent from the container serving customers."""
    root = pathlib.Path(__file__).resolve().parent.parent
    files = sorted(root.glob("requirements*.txt"))
    assert files, "no requirements files found"
    missing = [f.name for f in files
               if "yt-dlp" not in f.read_text(encoding="utf-8")]
    assert not missing, f"yt-dlp is not declared in: {', '.join(missing)}"
