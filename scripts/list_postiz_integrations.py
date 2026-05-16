"""List every Postiz integration with the identifying info you need
to figure out which one matches which YouTube channel.

Postiz's auto-named integrations ('Kaizer 1', 'Kaizer 2', ...) don't
tell you which YouTube account they belong to.  But each integration
carries a ``profile`` field (the @handle Postiz captured during the
original YouTube OAuth flow) and a ``picture`` field (avatar URL).
Either is enough to match a row to a real channel — handles are
usually obvious, avatars require eyeballing the image.

Run from KaizerBackend:
    python scripts/list_postiz_integrations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable so we can use the existing client.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from clients import postiz as postiz_client


def main() -> int:
    if not postiz_client.is_enabled():
        print("POSTIZ_API_KEY is not set in .env — fill it in first.")
        return 1
    try:
        rows = postiz_client.list_integrations()
    except postiz_client.PostizError as exc:
        print(f"Postiz error: {exc}")
        return 2

    yt = [r for r in rows if (r.get("provider") or r.get("identifier") or "").lower() == "youtube"]
    print(f"Found {len(yt)} YouTube integration(s) on this Postiz account.")
    print("=" * 100)
    # Sort by display name so duplicates are obvious.
    for i, r in enumerate(sorted(yt, key=lambda x: (x.get("name") or "").lower()), 1):
        name    = r.get("name") or "(no name)"
        # Postiz adds a random suffix (-y5p, -f4i, …) to handles to avoid
        # collisions across Postiz tenants. The real handle is everything
        # before the last hyphen.
        profile_raw = r.get("profile") or ""
        handle      = profile_raw.split("-")[0] if "-" in profile_raw else profile_raw
        picture = r.get("picture") or ""
        disabled = r.get("disabled")
        iid = r.get("id", "")
        marker = " [DISABLED]" if disabled else ""
        print(f"\n#{i:>2}. {name}{marker}")
        print(f"     handle  : {handle}")
        print(f"     full    : {profile_raw}")
        print(f"     avatar  : {picture}")
        print(f"     postiz_id: {iid}")
    print("\nTip: open one of the avatar URLs in your browser to identify which YouTube channel it is.")
    print("     The @handle (above 'full') is usually obvious by itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
