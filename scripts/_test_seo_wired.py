import os, sys, json
_B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _B)
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_B, ".env"))
from pipeline_v4 import seo_provider

canvas = json.load(open(os.path.join(_B, "output", "full_video_shorts_v4",
                                      "job_160", "canvas.json"), encoding="utf-8"))
st = canvas["bulletin"]["stories"][0]
seo = seo_provider.generate_seo(seo_provider.SeoInput(
    kind="bulletin", language="te",
    title_native=st.get("title_native", "") or "", title_english=st.get("title_english", "") or "",
    summary=st.get("summary", "") or "",
    body="\n".join(f"- {s.get('title_native') or s.get('title_english')}" for s in canvas["bulletin"]["stories"]),
    style_source_id=None), style_source=None)

print("title:", (seo.get("title") or "")[:60])
print("tool_score:", seo.get("tool_score"), "verdict:", seo.get("tool_verdict"))
print("tool_suggestions:")
for s in (seo.get("tool_suggestions") or []):
    print("  -", s)
print("\nRESULT:", "PASS (score attached)" if seo.get("tool_score") is not None else "FAIL (no tool_score)")
