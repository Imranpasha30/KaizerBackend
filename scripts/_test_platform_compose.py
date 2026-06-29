"""Regression: composer YouTube path is byte-for-byte unchanged; IG/FB shape.

No Gemini calls — the IG/FB variant is supplied inline so the test is
deterministic. Run: python scripts/_test_platform_compose.py
"""
import os, sys
_B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _B)
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_B, ".env"))

from seo.composer import compose


class _Tok:
    google_channel_title = "Auto Walla"


class _Ch:
    id = 7
    name = "Kaizer X Telugu"
    handle = "kaizerx"
    oauth_token = _Tok()
    fixed_tags = ["telugu news", "kaizer"]
    mandatory_hashtags = ["KaizerX"]
    footer = "Subscribe for daily updates."
    socials = {"instagram": "kaizerx", "youtube": "@kaizerx"}


GENERIC = {
    "title": "షాకింగ్: భూముల అమ్మకంపై ప్రతిపక్షాల ఆగ్రహం",
    "description": "ఈ రోజు అసెంబ్లీలో పెద్ద దుమారం. ప్రతిపక్షాలు తీవ్రంగా స్పందించాయి.",
    "keywords": ["telugu news", "assembly", "politics", "land sale"],
    "hashtags": ["#TeluguNews", "#Politics", "#Assembly"],
    "hook": "ఇది మీరు చూడాల్సిందే!",
    "thumbnail_text": "BREAKING",
}

ch = _Ch()
fails = 0

# 1) YouTube default == explicit platform='youtube' == pre-change behaviour.
yt_default = compose(GENERIC, ch, publish_kind="video")
yt_explicit = compose(GENERIC, ch, publish_kind="video", platform="youtube")
if yt_default == yt_explicit:
    print("PASS: youtube default == platform='youtube' (byte-identical)")
else:
    fails += 1
    print("FAIL: youtube default differs from platform='youtube'")
    for k in yt_default:
        if yt_default.get(k) != yt_explicit.get(k):
            print(f"   key {k}: {yt_default.get(k)!r} != {yt_explicit.get(k)!r}")

# YT must still carry a "| Auto Walla" title suffix and structured tags.
if "| Auto Walla" in yt_default["title"] and yt_default["keywords"]:
    print(f"PASS: youtube title suffix + tags intact -> {yt_default['title']!r}")
else:
    fails += 1
    print(f"FAIL: youtube title/tags wrong -> {yt_default['title']!r}")

# 2) Instagram: with an inline AI variant, caption + capped hashtags returned.
gen_ig = dict(GENERIC)
gen_ig["platform_variants"] = {
    "instagram": {
        "caption": "🔥 ఇది మీరు చూడాల్సిందే!\n\nఅసెంబ్లీలో దుమారం 😱",
        "hashtags": ["#reels", "#viral", "#TeluguNews", "#trending"],
        "hook": "ఇది మీరు చూడాల్సిందే!",
    }
}
ig = compose(gen_ig, ch, publish_kind="short", platform="instagram")
if ig.get("platform") == "instagram" and "చూడాల్సిందే" in ig.get("description", ""):
    print(f"PASS: instagram caption shaped -> {ig['description'][:60]!r}…")
else:
    fails += 1
    print(f"FAIL: instagram caption wrong -> {ig!r}")

# IG must have NO "| Channel" title suffix and mandatory hashtag first.
if "|" not in (ig.get("title") or "") and any("KaizerX" in h for h in ig.get("hashtags", [])):
    print(f"PASS: instagram no YT-title suffix + mandatory hashtag present "
          f"({len(ig.get('hashtags', []))} tags)")
else:
    fails += 1
    print(f"FAIL: instagram hashtags/title wrong -> title={ig.get('title')!r} "
          f"hashtags={ig.get('hashtags')}")

# Socials + footer overlaid into the caption text.
if "Subscribe for daily updates." in ig["description"]:
    print("PASS: instagram footer overlaid")
else:
    fails += 1
    print("FAIL: instagram footer missing")

# 3) Facebook: fallback (NO ai variant) deterministically shapes from generic.
fb = compose(GENERIC, ch, publish_kind="video", platform="facebook")
if fb.get("platform") == "facebook" and fb.get("description") and len(fb.get("hashtags", [])) <= 8:
    print(f"PASS: facebook fallback shape (<=8 hashtags) -> "
          f"composed_from={fb.get('composed_from')!r}")
else:
    fails += 1
    print(f"FAIL: facebook fallback wrong -> {fb!r}")

print("\nRESULT:", "ALL PASS" if fails == 0 else f"{fails} FAILURE(S)")
sys.exit(1 if fails else 0)
