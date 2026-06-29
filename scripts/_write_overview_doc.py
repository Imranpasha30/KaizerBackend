"""Extract the system-overview markdown from the workflow result file and write
it to the repo as a document. One-off."""
import json, io, os

SRC = r"C:\Users\user\AppData\Local\Temp\claude\e--kaizer-new-data-training\cbd30fda-c2d7-41d9-b130-761ae40794a9\tasks\wjehk1mgb.output"
DST = r"e:\kaizer new data training\KAIZER_X_SYSTEM_OVERVIEW.md"

with io.open(SRC, "r", encoding="utf-8", errors="replace") as f:
    raw = f.read()

doc = None
try:
    data = json.loads(raw)
    doc = data.get("document")
    if not doc and isinstance(data.get("result"), dict):
        doc = data["result"].get("document")
except Exception:
    # Fallback: pull the "document":"..." value if the file isn't clean JSON.
    import re
    m = re.search(r'"document"\s*:\s*"(.*)"\s*,\s*"mapped"', raw, flags=re.DOTALL)
    if m:
        doc = json.loads('"' + m.group(1) + '"')

if not doc:
    print("FAILED: could not extract document; first 300 chars of source:")
    print(raw[:300])
    raise SystemExit(1)

# Strip any leading preamble before the title (model sometimes prepends a line).
idx = doc.find("# Kaizer X")
if idx > 0:
    doc = doc[idx:]
doc = doc.strip() + "\n"

with io.open(DST, "w", encoding="utf-8", newline="\n") as f:
    f.write(doc)

print("WROTE:", DST)
print("chars:", len(doc), "| lines:", doc.count("\n") + 1)
print("--- first 400 ---")
print(doc[:400])
print("--- last 300 ---")
print(doc[-300:])
