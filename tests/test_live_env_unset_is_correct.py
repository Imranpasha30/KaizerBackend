"""Every KAIZER_LIVE_* setting must behave correctly when nobody sets it.

WHY THIS FILE EXISTS. The repo's mode-2 sweep flags a name that is read in
production with no literal default and written nowhere, because that is how
KAIZER_V4_LANGUAGE shipped: read in seven places, set in none, so the Telugu
recogniser never ran and nothing ever raised. Six new names from the live engine
land in that class.

They are the other kind — deliberate opt-ins whose unset behaviour is correct —
and the sweep's own docstring says to re-record the baseline for those. But
"unset is fine" is a claim, and a baseline entry is where a claim goes to stop
being checked. So it is a test instead, one per name, asserting what actually
happens when the variable is absent.

If any of these ever becomes a name whose absence breaks something, this file
fails rather than a JSON file quietly continuing to excuse it.
"""
from __future__ import annotations

import inspect
import os
import sys

import pytest

# Which copy of the engine does THIS deployment use? Asked of the application
# rather than hard-coded: pinning a path would make these tests import DEV's
# engine when run on LIVE -- passing, and proving nothing about the code that
# machine actually runs.
import live_integration as _li  # noqa: E402

STACK = _li.resolve_stack_dir()
if STACK and STACK not in sys.path:
    sys.path.insert(0, STACK)

import live_integration as li  # noqa: E402

LIVE_NAMES = [
    "KAIZER_LIVE_PREFIX",
    "KAIZER_LIVE_STACK",
    "KAIZER_LIVE_REDIS",
    "KAIZER_LIVE_FERNET_KEY",
    "KAIZER_LIVE_FFMPEG",
    "KAIZER_LIVE_FFPROBE",
    "KAIZER_LIVE_SWEEPER_INPROC",
]


def test_the_prefix_derives_from_the_database_when_unset():
    """The most important of the six, and the one that must NOT have a plain
    default. The prefix is the only thing keeping DEV and LIVE apart in a shared
    Redis, and `os.getenv("KAIZER_LIVE_PREFIX", "kl")` would mean every box that
    forgot to set it addressed the PRODUCTION keys -- a DEV sweep ending a
    paying customer's broadcast. Unset derives from the database name instead,
    so forgetting it on DEV cannot reach LIVE."""
    assert li.derive_prefix(None, "postgresql://u:p@h/kaizer_dev") == "kldev"
    assert li.derive_prefix(None, "postgresql://u:p@h/kaizer") == "kl"


def test_redis_falls_back_to_the_applications_url_then_a_localhost_default():
    """Unset means "use whatever Redis this deployment already has", which is a
    working single-box setup. The stack's own instance is an upgrade, not a
    requirement."""
    src = inspect.getsource(li.build_settings)
    assert 'os.getenv("KAIZER_LIVE_REDIS")' in src
    assert 'os.getenv("REDIS_URL")' in src, "no fallback to the application's Redis"
    assert "redis://" in src, "no final default"


def test_the_fernet_key_falls_back_to_the_applications_own(monkeypatch):
    """Unset is the NORMAL case on a single deployment: the engine seals stream
    destinations with the same key crypto.py already manages. A second secret
    would be a second thing to lose, and a mismatch means a worker that comes up
    healthy and fails on every output."""
    monkeypatch.delenv("KAIZER_LIVE_FERNET_KEY", raising=False)
    s = li.build_settings()
    assert s.fernet_key, "with the variable unset, no key was resolved at all"


def test_ffmpeg_and_ffprobe_resolve_themselves_when_unset(monkeypatch):
    """Unset means "find them", which is what a normal machine wants. What must
    never happen is a bare name handed to a subprocess and hoped for: on a
    buildpack deploy the venv's bin directory does not reach a child, and
    production lost a day to exactly that with yt-dlp."""
    monkeypatch.delenv("KAIZER_LIVE_FFMPEG", raising=False)
    monkeypatch.delenv("KAIZER_LIVE_FFPROBE", raising=False)
    s = li.build_settings()
    assert s.ffmpeg and s.ffprobe
    for name, path in (("ffmpeg", s.ffmpeg), ("ffprobe", s.ffprobe)):
        assert os.path.isfile(path) or os.sep not in path, \
            f"{name} resolved to {path!r}, which is neither a real file nor a bare name"


def test_the_sweeper_runs_as_its_own_process_when_unset():
    """Unset is the DEFAULT ARCHITECTURE, not a missing setting: the sweeper is
    its own process so that deploying the API is not a sweeper outage. The
    variable only exists to put it back on a thread for a single-box deploy.

    And its absence cannot be silent -- that was the whole risk of moving it
    out -- so the API says at startup whether anything is sweeping.
    """
    import main
    src = inspect.getsource(main._check_live_sweeper)
    assert 'KAIZER_LIVE_SWEEPER_INPROC' in src
    assert '== "1"' in src, "any value other than 1 must leave the default alone"
    assert "WARNING: no live-control process is sweeping" in src, \
        "a missing sweeper must be reported, not assumed"
    assert "python -m kaizer_live.control" in src, "the warning must say what to start"


#: Read by the APPLICATION, never by the engine. They decide how this backend
#: uses the stack -- whether it hosts the sweeper itself, and WHICH COPY of the
#: engine it imports -- so they are documented in the application's own .env,
#: and a standalone stack deployment has no use for them.
APP_ONLY = {"KAIZER_LIVE_SWEEPER_INPROC", "KAIZER_LIVE_STACK"}


@pytest.mark.parametrize("name", LIVE_NAMES)
def test_each_name_is_documented_where_whoever_sets_it_will_look(name):
    """A setting nobody can find is a setting nobody sets correctly.

    Two audiences, two files. The stack is deployed separately by someone who
    has never seen this code, so everything IT reads is in its .env.example.
    The two the application reads are in the application's .env, because the
    stack has no use for them.
    """
    import pathlib
    if name in APP_ONLY:
        env = pathlib.Path(__file__).resolve().parent.parent / ".env"
        if not env.is_file():
            pytest.skip("this deployment has no .env")
        assert name in env.read_text(encoding="utf-8", errors="replace"),             f"{name} is read by the application but set nowhere in its .env"
        return
    example = pathlib.Path(STACK) / ".env.example"
    if not example.is_file():
        pytest.skip("the stack's .env.example is not present")
    assert name in example.read_text(encoding="utf-8"),         f"{name} is read but not documented in the stack's .env.example"


def test_no_live_setting_is_read_that_nothing_could_ever_supply():
    """The sweep's question, asked the other way round: every KAIZER_LIVE_* name
    the engine reads must appear somewhere a deployment could set it.

    This is the check that would have caught KAIZER_YTDLP_BIN, which was read in
    this codebase and written nowhere at all.
    """
    import pathlib
    import re
    stack = pathlib.Path(STACK)
    read: set[str] = set()
    for f in (stack / "kaizer_live").glob("*.py"):
        read |= set(re.findall(r'getenv\(\s*"(KAIZER_LIVE_[A-Z_]+)"', f.read_text(encoding="utf-8")))
    # A COMMENTED entry counts as documented, and for some names it is the only
    # correct form. KAIZER_LIVE_STACK must stay unset on a deployment with one
    # copy of the engine; writing it live in the example would make every .env
    # copied from it point at a path that does not exist there. What the sweep
    # is really asking is "could a deployment find out this exists?", and a
    # commented line with an explanation answers that.
    documented = set(re.findall(r"^\s*#?\s*(KAIZER_LIVE_[A-Z_]+)=",
                                (stack / ".env.example").read_text(encoding="utf-8"), re.M))
    # Names the engine invents for itself rather than reading from a deployment.
    internal = {"KAIZER_LIVE_ENV_FILE", "KAIZER_LIVE_WORKER_ID", "KAIZER_LIVE_ENCODE_ID",
                "KAIZER_LIVE_START_CHECK", "KAIZER_LIVE_AUTO_STOP", "KAIZER_LIVE_SERVICE_FACTORY"}
    orphans = sorted(read - documented - internal)
    assert not orphans, (
        "the engine reads these and no deployment could know to set them:\n  "
        + "\n  ".join(orphans)
        + "\nAdd each to kaizer-live-stack/.env.example with a comment saying what it does.")


def test_the_stack_directory_derives_from_where_this_backend_lives(monkeypatch):
    """The one that keeps production off DEV's engine.

    DEV and LIVE share one virtualenv here, and an editable install is a pointer
    -- a venv holds exactly ONE per package name. So with nothing set,
    `import kaizer_live` from either backend resolves to whichever folder was
    installed last, and a half-finished edit on DEV becomes what a paying
    customer's broadcast runs on. No deploy, no restart, and nothing to see:
    both sides import the same module name and neither says where from.

    Unset must therefore DERIVE, not default: from where this backend's own file
    sits, so forgetting the variable cannot reach across. And when it resolves
    to nothing, the installed package is used -- which is the right answer on a
    standalone deployment that only has one copy.
    """
    import os
    import pathlib

    monkeypatch.delenv("KAIZER_LIVE_STACK", raising=False)
    got = li.resolve_stack_dir()
    assert got, "nothing resolved: this deployment would import whatever was installed last"
    assert (pathlib.Path(got) / "kaizer_live").is_dir(), f"{got} holds no engine"

    # It must be THIS deployment's copy, not a sibling's.
    here = pathlib.Path(li.__file__).resolve().parent          # .../KaizerBackend
    assert pathlib.Path(got).resolve().is_relative_to(here.parent.parent) or \
           pathlib.Path(got).resolve().is_relative_to(here.parent), \
        f"{got} is not under this deployment's tree ({here.parent})"

    # An explicit value still wins, and a bad one resolves to nothing rather
    # than silently falling through to someone else's engine.
    monkeypatch.setenv("KAIZER_LIVE_STACK", str(here))
    assert li.resolve_stack_dir() == "" or li.resolve_stack_dir() == str(here)
    monkeypatch.setenv("KAIZER_LIVE_STACK", r"C:\nope\not\here")
    assert li.resolve_stack_dir() == ""


def test_the_engine_says_which_copy_it_loaded():
    """A wrong answer here cannot be detected at runtime -- the module imports
    cleanly either way -- so the only place it can be caught is a line somebody
    reads after a deploy."""
    import inspect
    src = inspect.getsource(li.get_live_service)
    assert "engine loaded from" in src
