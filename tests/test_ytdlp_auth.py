"""yt-dlp auth options: inert unless configured, and never leaky.

WHY THIS EXISTS. With yt-dlp finally installed, production hit YouTube's
anonymous-extraction block:

    ERROR: [youtube] V_b7R6a5D_w: Sign in to confirm you're not a bot.

That failure is a property of the SERVER'S IP, not of the video, so it cannot
be reproduced on the operator's desktop where the same URL fetches perfectly.
Hence configuration rather than a code workaround.

The two properties worth defending are that an unconfigured machine behaves
exactly as before (so this cannot break the desktop or DEV), and that cookies
— which are credentials — never leave this module as content.
"""
import os

import pytest

from live_studio import ytdlp_auth


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("YTDLP_COOKIES", "YTDLP_COOKIES_FILE",
              "YTDLP_PLAYER_CLIENT", "YTDLP_PROXY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(ytdlp_auth, "_CACHED", None)


# ── inert by default ─────────────────────────────────────────────────

def test_nothing_configured_adds_nothing():
    """The desktop and DEV set none of these; they must be unaffected."""
    assert ytdlp_auth.ytdlp_auth_args() == []


def test_blank_values_are_not_configuration():
    """A PaaS field left empty arrives as "" , not as unset."""
    os.environ["YTDLP_PLAYER_CLIENT"] = "   "
    os.environ["YTDLP_COOKIES"] = "\n"
    os.environ["YTDLP_PROXY"] = ""
    assert ytdlp_auth.ytdlp_auth_args() == []


# ── each knob ────────────────────────────────────────────────────────

def test_player_client_is_passed_as_an_extractor_arg():
    os.environ["YTDLP_PLAYER_CLIENT"] = "tv"
    assert ytdlp_auth.ytdlp_auth_args() == [
        "--extractor-args", "youtube:player_client=tv"]


def test_a_cookies_file_on_disk_is_used(tmp_path):
    f = tmp_path / "cookies.txt"
    f.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    os.environ["YTDLP_COOKIES_FILE"] = str(f)
    assert ytdlp_auth.ytdlp_auth_args() == ["--cookies", str(f)]


def test_a_missing_cookies_path_is_ignored_not_passed(tmp_path):
    """Handing yt-dlp a path that is not there turns a fixable config
    mistake into a confusing extractor error."""
    os.environ["YTDLP_COOKIES_FILE"] = str(tmp_path / "nope.txt")
    assert ytdlp_auth.ytdlp_auth_args() == []


def test_proxy_is_passed():
    os.environ["YTDLP_PROXY"] = "http://user:pw@host:8080"
    assert ytdlp_auth.ytdlp_auth_args() == ["--proxy", "http://user:pw@host:8080"]


def test_all_three_compose():
    os.environ["YTDLP_PLAYER_CLIENT"] = "web_safari"
    os.environ["YTDLP_PROXY"] = "socks5://h:1"
    os.environ["YTDLP_COOKIES"] = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\n"
    args = ytdlp_auth.ytdlp_auth_args()
    assert "--cookies" in args and "--extractor-args" in args and "--proxy" in args


# ── cookie CONTENT in an env var (the PaaS case) ─────────────────────

def test_cookie_content_becomes_a_file():
    os.environ["YTDLP_COOKIES"] = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tX\n"
    args = ytdlp_auth.ytdlp_auth_args()
    assert args[0] == "--cookies"
    assert os.path.isfile(args[1])
    assert "Netscape" in open(args[1], encoding="utf-8").read()


def test_escaped_newlines_are_restored():
    """A secret pasted into a PaaS field usually arrives with \\n escaped;
    yt-dlp needs a real Netscape file, one cookie per line."""
    os.environ["YTDLP_COOKIES"] = "# Netscape HTTP Cookie File\\n.youtube.com\\tTRUE\n"
    path = ytdlp_auth.ytdlp_auth_args()[1]
    assert open(path, encoding="utf-8").read().count("\n") >= 2


def test_the_temp_file_is_written_once_and_reused():
    """Rewriting it per broadcast would litter a credential across /tmp."""
    os.environ["YTDLP_COOKIES"] = "# Netscape HTTP Cookie File\n"
    first = ytdlp_auth.ytdlp_auth_args()[1]
    second = ytdlp_auth.ytdlp_auth_args()[1]
    assert first == second


# ── credentials must not leak ────────────────────────────────────────

def test_args_carry_a_path_never_the_secret():
    secret = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tSID\tSUPERSECRET\n"
    os.environ["YTDLP_COOKIES"] = secret
    assert "SUPERSECRET" not in " ".join(ytdlp_auth.ytdlp_auth_args())


def test_describe_says_whether_not_what():
    """describe() goes in the log, so it may say a cookie is configured and
    must never say what it is. Without it, 'still failing' cannot be told
    apart from 'the secret never reached the container'."""
    secret = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tSID\tSUPERSECRET\n"
    os.environ["YTDLP_COOKIES"] = secret
    os.environ["YTDLP_PROXY"] = "http://user:pw@host:8080"
    out = ytdlp_auth.describe()
    assert "cookies=yes" in out and "proxy=yes" in out
    assert "SUPERSECRET" not in out and "pw" not in out


def test_describe_is_honest_when_nothing_is_set():
    assert ytdlp_auth.describe() == "no yt-dlp auth configured"


# ── and the callers actually use it ──────────────────────────────────

@pytest.mark.parametrize("name", ["streamer", "url_ingest"])
def test_both_ytdlp_callers_pass_the_auth_args(name):
    import importlib, inspect
    src = inspect.getsource(importlib.import_module(f"live_studio.{name}"))
    assert "*_ytdlp_auth_args()," in src, f"{name} builds its args without auth"
