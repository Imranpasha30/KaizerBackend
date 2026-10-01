"""Every adminApi method the admin console CALLS must exist in its client.

This is the codebase's signature defect in its purest form. The operator's
Super Admin screen showed three at once:

    R.adminApi.accountRequests is not a function
    R.adminApi.usage is not a function
    R.adminApi.managedKeys is not a function

Ten were missing. The cause was a hand-maintained copy: the website's 13 admin
PAGE files were vendored into the desktop, but the api client they call was
not, so the desktop kept a pre-Super-Admin adminApi. Nothing failed at build
time -- JavaScript is happy to read an undefined property off an object -- so
it only surfaced when a panel was opened.

The two clients cannot simply be one file: the desktop's deliberately routes
/admin/ over the shell's IPC bridge to the hosted API, because the local engine
never mounts admin_router. Two copies is the design; this test is what keeps
them honest.

It is a STATIC check on purpose. It needs no node, no browser and no build, so
it runs in the ordinary backend gate the operator already runs.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

TREES = {
    "web": ROOT / "kaizerFrontned" / "src",
    "desktop": ROOT / "kaizer-desktop" / "ui" / "src" / "legacy" / "spa",
}

CALL = re.compile(r"\badminApi\s*\.\s*([A-Za-z_]\w*)")


def _client(tree: pathlib.Path):
    for name in ("client.js", "client.jsx", "client.ts"):
        p = tree / "api" / name
        if p.is_file():
            return p
    return None


def _defined(text: str):
    """The keys on the `adminApi` object literal, by brace matching -- a regex
    over the whole file would also catch the many other api objects."""
    m = re.search(r"(?:export\s+)?const\s+adminApi\s*=\s*\{", text)
    if not m:
        return None
    i = m.end() - 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                body = text[i:j + 1]
                break
    else:
        return None
    names = set(re.findall(r"^\s*(?:async\s+)?([A-Za-z_]\w*)\s*\(", body, re.M))
    names |= set(re.findall(r"^\s*([A-Za-z_]\w*)\s*:", body, re.M))
    return names


def _called(tree: pathlib.Path, client: pathlib.Path):
    out = {}
    for f in list(tree.rglob("*.jsx")) + list(tree.rglob("*.js")):
        if f == client:
            continue
        hits = set(CALL.findall(f.read_text(encoding="utf-8", errors="replace")))
        for h in hits:
            out.setdefault(h, []).append(f.relative_to(tree).as_posix())
    return out


@pytest.mark.parametrize("label", sorted(TREES))
def test_every_admin_api_method_called_is_defined(label):
    tree = TREES[label]
    if not tree.is_dir():
        pytest.skip(f"{label} tree not present at {tree}")
    client = _client(tree)
    assert client is not None, f"{label}: no api/client found under {tree}"

    defined = _defined(client.read_text(encoding="utf-8", errors="replace"))
    assert defined, f"{label}: could not read the adminApi object in {client}"

    called = _called(tree, client)
    missing = {k: v for k, v in called.items() if k not in defined}
    assert not missing, (
        f"{label}: {len(missing)} adminApi method(s) are called but not defined "
        f"in {client.relative_to(ROOT).as_posix()} -- each throws "
        f"'is not a function' the moment its panel opens:\n" +
        "\n".join(f"    adminApi.{k}  <- {', '.join(sorted(set(v))[:3])}"
                  for k, v in sorted(missing.items())))


def test_the_desktop_still_routes_admin_over_the_shell_bridge():
    """The reason the two clients cannot be merged into one file. If this ever
    goes, admin on the desktop talks to the LOCAL engine, which does not mount
    admin_router at all -- every call 404s."""
    client = _client(TREES["desktop"])
    if client is None:
        pytest.skip("desktop tree not present")
    text = client.read_text(encoding="utf-8", errors="replace")
    assert "cloudAdmin" in text, \
        "the desktop api client no longer routes /admin/ to the hosted API"


def test_the_two_clients_agree_on_the_admin_surface():
    """Not identical files -- identical adminApi SURFACE. A method added to the
    website's console must reach the desktop's copy, which is exactly the step
    that was missed."""
    if not all(t.is_dir() for t in TREES.values()):
        pytest.skip("both trees required")
    surfaces = {}
    for label, tree in TREES.items():
        c = _client(tree)
        surfaces[label] = _defined(c.read_text(encoding="utf-8", errors="replace")) or set()
    only_web = surfaces["web"] - surfaces["desktop"]
    assert not only_web, (
        "the website's admin console has method(s) the desktop's copy lacks, so "
        "a tab that works in the browser will break in the app:\n    " +
        ", ".join(sorted(only_web)))


# ── the right method on the WRONG object ─────────────────────────────
# AdminDesktopLicenses.jsx imported { api } and called
# api.adminDesktopLicenses(). Both that and adminRevokeDesktopLicense are
# defined on adminApi. `api.adminDesktopLicenses` was therefore undefined, and
# calling it threw SYNCHRONOUSLY inside a useCallback handed to useEffect -- so
# the .catch() never attached, React re-threw to the page-level error boundary,
# and the whole Super Admin console was replaced by "Something went wrong".
#
# The check above would never have caught it: the call is on `api`, not
# `adminApi`. This one looks for the mirror image -- a method taken off `api`
# that only exists on `adminApi`.
def _api_object(text: str, name: str):
    m = re.search(r"(?:export\s+)?const\s+" + name + r"\s*=\s*\{", text)
    if not m:
        return None
    i = m.end() - 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                body = text[i:j + 1]
                break
    else:
        return None
    names = set(re.findall(r"^\s*(?:async\s+)?([A-Za-z_]\w*)\s*\(", body, re.M))
    names |= set(re.findall(r"^\s*([A-Za-z_]\w*)\s*:", body, re.M))
    return names


PLAIN_CALL = re.compile(r"(?<![.\w])api\s*\.\s*([A-Za-z_]\w*)")


@pytest.mark.parametrize("label", sorted(TREES))
def test_no_admin_method_is_called_off_the_plain_api_object(label):
    tree = TREES[label]
    if not tree.is_dir():
        pytest.skip(f"{label} tree not present")
    client = _client(tree)
    text = client.read_text(encoding="utf-8", errors="replace")
    on_api = _api_object(text, "api") or set()
    on_admin = _api_object(text, "adminApi") or set()
    assert on_api and on_admin, f"{label}: could not read both api objects"

    admin_only = on_admin - on_api
    wrong = {}
    for f in list(tree.rglob("*.jsx")) + list(tree.rglob("*.js")):
        if f == client:
            continue
        body = f.read_text(encoding="utf-8", errors="replace")
        for name in set(PLAIN_CALL.findall(body)):
            if name in admin_only:
                wrong.setdefault(name, []).append(f.relative_to(tree).as_posix())

    assert not wrong, (
        f"{label}: method(s) taken off `api` that only exist on `adminApi` -- "
        f"undefined at runtime, and a throw inside an effect takes the whole "
        f"page down:\n" +
        "\n".join(f"    api.{k}  <- {', '.join(sorted(set(v)))}"
                  for k, v in sorted(wrong.items())))

# ── the same question, asked the other way round ─────────────────────

# NOT preceded by / . - or a word character. That is what separates a real
# `api.foo(` from the `api.` inside `https://api.qrserver.com/...` and from
# `something.api.foo(`. Learned by this guard reporting both as bugs.
ANY_CALL = re.compile(r"(?<![\w./-])(adminApi|api)\s*\.\s*([A-Za-z_]\w*)\s*\(")

_LINE_COMMENT = re.compile(r"^\s*(//|\*|/\*).*$", re.M)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def _code_only(text: str) -> str:
    """Comments removed, string literals kept.

    Prose about the API is not a call -- one finding here was a comment reading
    "from api.client `req`". String literals stay, because a call inside a
    template literal is still a call and stripping them would hide real ones.
    """
    text = _BLOCK_COMMENT.sub(" ", text)
    return _LINE_COMMENT.sub("", text)


def _object_keys(text: str, obj: str):
    """The keys of one exported object literal, by brace matching."""
    m = re.search(rf"(?:export\s+)?const\s+{re.escape(obj)}\s*=\s*\{{", text)
    if not m:
        return None
    i = m.end() - 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                body = text[i:j + 1]
                break
    else:
        return None
    names = set(re.findall(r"^\s*(?:async\s+)?([A-Za-z_]\w*)\s*\(", body, re.M))
    names |= set(re.findall(r"^\s*([A-Za-z_]\w*)\s*:", body, re.M))
    return names


@pytest.mark.parametrize("label", sorted(TREES))
def test_no_call_lands_on_an_object_that_does_not_have_it(label):
    """Both directions, because both have shipped.

    The test above catches `adminApi.x` where x is not on adminApi. LIVE shipped
    the opposite: `api.adminDesktopLicenses`, which exists on adminApi only. In
    JavaScript both read as `undefined` and both throw the moment the panel
    opens -- there is no build-time complaint about either.

    A name may legitimately be on BOTH objects (getJob and listJobs are), so the
    question is only ever "is it on the one it was taken off".
    """
    tree = TREES[label]
    if not tree.is_dir():
        pytest.skip(f"{label} tree not present at {tree}")
    client = _client(tree)
    assert client is not None, f"{label}: no api/client found under {tree}"
    text = client.read_text(encoding="utf-8", errors="replace")

    keys = {obj: (_object_keys(text, obj) or set()) for obj in ("api", "adminApi")}
    assert keys["api"] and keys["adminApi"], f"{label}: could not read both api objects"

    wrong = {}
    for f in list(tree.rglob("*.jsx")) + list(tree.rglob("*.js")):
        if f == client or "_archive" in f.parts:
            continue
        src = _code_only(f.read_text(encoding="utf-8", errors="replace"))
        for obj, meth in ANY_CALL.findall(src):
            if meth in keys[obj]:
                continue
            other = "adminApi" if obj == "api" else "api"
            where = f" (it is on {other})" if meth in keys[other] else ""
            wrong.setdefault(f"{obj}.{meth}{where}", set()).add(f.relative_to(tree).as_posix())

    assert not wrong, (
        f"{label}: {len(wrong)} call(s) land on an object that does not define them -- "
        f"undefined at runtime, and a throw inside an effect takes the whole page "
        f"down:\n" + "\n".join(f"    {call}  <- {', '.join(sorted(files)[:3])}"
                                for call, files in sorted(wrong.items())))
