import os, sys, json
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_BACKEND, ".env"))
from seo.score_checker import score_seo
from pipeline_v4 import seo_provider

canvas = json.load(open(os.path.join(_BACKEND, "output", "full_video_shorts_v4",
                                      "job_160", "canvas.json"), encoding="utf-8"))
b = canvas["bulletin"]
stories = b.get("stories", [])
content = "\n".join([(s.get("title_native") or "") + " " + (s.get("summary") or "")
                     for s in stories]).strip()

# Generate fresh SEO via the working provider, then score it.
st = stories[0] if stories else {}
seo = seo_provider.generate_seo(seo_provider.SeoInput(
    kind="bulletin", language=canvas.get("language", "te"),
    title_native=st.get("title_native", "") or "", title_english=st.get("title_english", "") or "",
    summary=st.get("summary", "") or "",
    body="\n".join(f"- {s.get('title_native') or s.get('title_english')}" for s in stories),
    style_source_id=None), style_source=None)

print("Generated SEO:")
print("  title:", (seo.get("title") or "")[:80])
print("  tags :", len(seo.get("tags") or []))
print("  desc :", len(seo.get("description") or ""), "chars")

r = score_seo(
    title=seo.get("title", ""), description=seo.get("description", ""),
    tags=seo.get("tags") or [], content_text=content, language=canvas.get("language", "te"),
    use_trends=True,
)
print(f"\nSEO SCORE: {r['score']}/100  verdict={r['verdict']}")
for k, v in r["dimensions"].items():
    print(f"   {k:18} {v['points']}/{v['max']}  {v['note']}")
print("\nsuggestions:")
for s in r["suggestions"]:
    print("  -", s)
print("\ntop content keywords:", ", ".join(r["keywords"][:8]))
