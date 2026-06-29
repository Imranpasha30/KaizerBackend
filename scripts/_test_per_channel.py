import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from seo.per_channel import build_channel_previews

base = {"title": "Government land scam in the colony", "description": "A report.",
        "tags": ["news", "telugu"]}


class Ch:
    def __init__(s, i, n): s.id = i; s.name = n


chans = [Ch(1, "Kaizer 30"), Ch(2, "Kaizer 5")]
fb = {1: {"winning_keywords": ["Kukatpally", "encroachment", "acres"]}}

print("--- per_channel mode (paid) ---")
for p in build_channel_previews(base_seo=base, channels=chans, feedback_by_channel=fb,
                                content_keywords=["colony", "land"], mode="per_channel"):
    print(f"  {p['channel_name']}: source={p['source']}")
    print(f"    title: {p['title']}")
    print(f"    tags : {p['tags']}")
    print(f"    why  : {p['why']}")

print("--- shared mode (free, current behaviour) ---")
for p in build_channel_previews(base_seo=base, channels=chans, mode="shared"):
    print(f"  {p['channel_name']}: source={p['source']} title={p['title']!r} why={p['why']}")
