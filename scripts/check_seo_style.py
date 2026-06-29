"""End-to-end check that a style reference actually steers SEO output.

For a user's first style reference (kind:styles), this:
  1. prints what voice data it carries (title_formula / desc_style / corpus),
  2. runs pipeline_v4.seo_provider.generate_seo TWICE on the same input —
     once with style_source=None (default voice) and once with the style
     reference — and prints both titles/descriptions side by side, plus
     the recorded style_source_id.

This exercises the exact path V4's Canvas "Regenerate" and Quick Publish
now use. Needs a Gemini key in the env (same one the app uses).

  python scripts/check_seo_style.py [email]
"""
import sys
sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")

import models
from database import SessionLocal
from pipeline_v4.seo_provider import SeoInput, generate_seo

EMAIL = sys.argv[1] if len(sys.argv) > 1 else "imranpasha.ahmed@gmail.com"


def _short(s, n=90):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


db = SessionLocal()
try:
    u = db.query(models.User).filter(models.User.email == EMAIL).first()
    if not u:
        print(f"user {EMAIL} not found"); sys.exit(1)

    # First style reference = channel WITHOUT a usable oauth token.
    refs = [c for c in db.query(models.Channel)
            .filter(models.Channel.user_id == u.id)
            .order_by(models.Channel.id).all()
            if not (c.oauth_token and (c.oauth_token.refresh_token_enc or "").strip())]
    if not refs:
        print("no style references for this user — add one under SEO Settings")
        sys.exit(1)
    ref = refs[0]
    corp = getattr(ref, "corpus", None)
    payload = getattr(corp, "payload", None) if corp else None
    print(f"USER {u.id} <{u.email}>")
    print(f"STYLE REF #{ref.id} '{ref.name}'")
    print(f"  title_formula: {_short(getattr(ref, 'title_formula', ''))}")
    print(f"  desc_style:    {_short(getattr(ref, 'desc_style', ''))}")
    print(f"  corpus:        {'yes' if payload else 'no'}"
          + (f" (top_titles={len(payload.get('top_titles') or [])})" if payload else ""))
    print()

    inp_kw = dict(
        kind="short",
        language="te",
        title_native="హైదరాబాద్‌లో భారీ వర్షాలు — లోతట్టు ప్రాంతాలు జలమయం",
        title_english="Heavy rains in Hyderabad flood low-lying areas",
        summary="Heavy overnight rain flooded several low-lying colonies in "
                "Hyderabad; traffic halted, NDRF teams deployed for rescue.",
    )

    print("=== DEFAULT VOICE (style_source=None) ===")
    base = generate_seo(SeoInput(**inp_kw))
    print(f"  title:       {_short(base.get('title'))}")
    print(f"  description: {_short(base.get('description'), 120)}")
    print(f"  style_source_id: {base.get('style_source_id')}")
    print()

    print(f"=== IN THE STYLE OF '{ref.name}' (style_source=#{ref.id}) ===")
    styled = generate_seo(
        SeoInput(style_source_id=ref.id, **inp_kw), style_source=ref,
    )
    print(f"  title:       {_short(styled.get('title'))}")
    print(f"  description: {_short(styled.get('description'), 120)}")
    print(f"  style_source_id: {styled.get('style_source_id')}")
    print()

    ok = True
    if styled.get("style_source_id") != ref.id:
        print(f"  FAIL  style_source_id not recorded "
              f"(got {styled.get('style_source_id')!r})")
        ok = False
    else:
        print("  PASS  style_source_id recorded on styled output")
    if not (styled.get("title") or "").strip():
        print("  FAIL  styled output has no title"); ok = False
    else:
        print("  PASS  styled output has a usable title")
    # The reference channel's name must NOT leak into the copy.
    leak = (ref.name or "").strip().lower()
    blob = (styled.get("title", "") + " " + styled.get("description", "")).lower()
    if leak and leak in blob:
        print(f"  WARN  reference name '{ref.name}' appears in output — "
              f"sanitizer should have stripped it")
    else:
        print("  PASS  reference channel name not leaked into output")

    print("\nSTYLE WIRING OK" if ok else "\nSTYLE WIRING FAILED")
    sys.exit(0 if ok else 1)
finally:
    db.close()
