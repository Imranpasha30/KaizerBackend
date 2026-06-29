"""Extract the R&D report from the workflow output, save it to the repo,
and email it to the operator."""
import json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
sys.path.insert(0, _BACKEND)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SRC = r"C:\Users\user\AppData\Local\Temp\claude\e--kaizer-new-data-training\cbd30fda-c2d7-41d9-b130-761ae40794a9\tasks\wzbyyrk73.output"
data = json.load(open(SRC, encoding="utf-8"))
report = (data.get("result") or {}).get("report") or ""
# Strip the synthesizer's preamble before the real "# " title.
idx = report.find("# Kaizer X")
if idx > 0:
    report = report[idx:]
report = report.strip()

OUT = os.path.join(os.path.dirname(os.path.dirname(_BACKEND)), "KAIZER_X_RND_REPORT.md")
with open(OUT, "w", encoding="utf-8") as f:
    f.write(report + "\n")
print(f"wrote {OUT}  ({len(report)} chars)")

# Email the full report.
from send_update_email import send  # type: ignore
ok = send(
    "imranpasha.ahmed@gmail.com",
    "Kaizer X - R&D & competitive research report",
    report,
)
print("emailed:", ok)
