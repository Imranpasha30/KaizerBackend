"""Generic: extract a workflow {result:{report}} output, save to repo root,
and email it. Args: <src_output_file> <subject> <out_md_name> <title_anchor>"""
import json, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
sys.path.insert(0, _BACKEND)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

src, subject, out_name, anchor = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
data = json.load(open(src, encoding="utf-8"))
report = (data.get("result") or {}).get("report") or ""
idx = report.find(anchor)
if idx > 0:
    report = report[idx:]
report = report.strip()

OUT = os.path.join(os.path.dirname(os.path.dirname(_BACKEND)), out_name)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(report + "\n")
print(f"wrote {OUT}  ({len(report)} chars)")

from send_update_email import send  # type: ignore
print("emailed:", send("imranpasha.ahmed@gmail.com", subject, report))
