"""One-off: prove per-channel SEO writes DISTINCT titles.

Mirrors the operator's "same title in every channel" complaint: takes ONE base
SEO + ONE bulletin text, then runs generate_channel_seo() for two real channels
and prints each channel's title / score / tags. If the titles differ, the
per-channel writer is doing its job. Run from KaizerBackend:

    venv\\Scripts\\python.exe scripts\\verify_per_channel_seo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import SessionLocal
from models import Channel
from seo.per_channel import generate_channel_seo

BASE_SEO = {
    "title": "Pawan Kalyan Big Announcement Shocks Andhra Politics",
    "description": "The deputy CM addressed the press today on the new policy. "
                   "Full breakdown of what it means for the state.",
    "tags": ["telugu news", "andhra pradesh", "pawan kalyan", "politics"],
}
CONTENT = (
    "Deputy Chief Minister Pawan Kalyan held a press conference in Vijayawada "
    "announcing a new welfare scheme for farmers. Opposition leaders reacted "
    "sharply. The scheme will roll out next month across all districts. "
    "Analysts say it could shift the political equation ahead of elections."
)


def main():
    db = SessionLocal()
    try:
        chans = (
            db.query(Channel)
            .filter(Channel.kind == "account")
            .order_by(Channel.id.asc())
            .limit(2)
            .all()
        )
        if len(chans) < 2:
            chans = db.query(Channel).order_by(Channel.id.asc()).limit(2).all()
        if len(chans) < 2:
            print("NEED >=2 channels in DB to compare; found", len(chans))
            return

        print(f"Comparing channels: "
              f"[{chans[0].id}] {chans[0].name!r}  vs  "
              f"[{chans[1].id}] {chans[1].name!r}")
        print("=" * 78)

        results = []
        for ch in chans:
            r = generate_channel_seo(
                base_seo=BASE_SEO,
                channel=ch,
                content_text=CONTENT,
                language=(ch.language or "te"),
                winning_keywords=[],   # no perf history needed to prove distinctness
            )
            results.append(r)
            print(f"\n--- [{ch.id}] {ch.name} ---")
            print(f"  source : {r.get('source')}")
            print(f"  score  : {r.get('seo_score')}")
            print(f"  TITLE  : {r.get('title')}")
            print(f"  desc   : {(r.get('description') or '')[:120]}")
            print(f"  tags   : {', '.join((r.get('tags') or [])[:8])}")

        print("\n" + "=" * 78)
        t0 = (results[0].get("title") or "").strip()
        t1 = (results[1].get("title") or "").strip()
        if t0 and t1 and t0 != t1:
            print("RESULT: PASS — titles are DISTINCT per channel.")
        elif not t0 or not t1:
            print("RESULT: INCONCLUSIVE — a generation returned an empty title (fell back).")
        else:
            print("RESULT: SAME TITLE — per-channel writer did NOT differentiate.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
