"""One-off: re-run the (now fixed) offline inference on every custom template's raw source
and update its stored entry HTML + contract + preview ONLY when the slot set actually
changes — so the over-marking fix (no phantom video/video:2 slots) applies to templates
uploaded before the fix. Safe: per-template try/rollback; unchanged templates untouched."""
import os, sys, json, shutil
os.environ.setdefault("PYTHONUTF8", "1")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from database import SessionLocal
import models
from services import custom_templates as ct
from services.custom_templates import infer as _infer
from services.custom_templates import bundle as _bundle


def sig(slots):
    return sorted((s.get("kind"), s.get("name") or s.get("key") or "") for s in (slots or []))


# Clear stale AI-understand cache (pre-fix cached markings could re-introduce phantom slots).
cache = os.path.join(HERE, "output", "custom_templates", "_aiunderstand_cache")
if os.path.isdir(cache):
    shutil.rmtree(cache, ignore_errors=True)
    print("cleared AI-understand cache")

db = SessionLocal()
changed = 0
try:
    for t in db.query(models.CustomTemplate).all():
        try:
            d = t.dir_path or ""
            entry = os.path.join(d, t.entry_rel or "index.html")
            raw = os.path.join(d, ".kaizer_src.html")
            src = raw if os.path.isfile(raw) else entry
            if not d or not os.path.isfile(src):
                print(f"tmpl {t.id}: no source, skip"); continue
            with open(src, encoding="utf-8", errors="replace") as fh:
                html = fh.read()
            norm, contract = _infer.normalize_and_discover(html)
            cj_new = contract.to_json()
            cur = t.contract_json if isinstance(t.contract_json, dict) else (
                json.loads(t.contract_json) if t.contract_json else {})
            if sig(cj_new.get("slots")) == sig((cur or {}).get("slots")):
                print(f"tmpl {t.id}: unchanged, skip")
                continue
            with open(entry, "w", encoding="utf-8") as fh:
                fh.write(norm or html)
            cj_new["slot_audit"] = ct.audit_template(contract)
            t.contract_json = cj_new
            t.canvas_w, t.canvas_h = contract.canvas_w, contract.canvas_h
            t.status = "ready" if (contract.video_slots or contract.background_slots) else "invalid"
            db.commit()
            changed += 1
            try:
                b = _bundle.Bundle(root_dir=d, entry_rel=t.entry_rel or "index.html", files=[])
                pv = t.preview_path or os.path.join(os.path.dirname(d), "preview.png")
                ct.render_preview(b, contract, pv)
            except Exception as e:
                print(f"  tmpl {t.id}: preview regen failed: {e!r}")
            print(f"tmpl {t.id}: UPDATED video={[s.key for s in contract.video_slots]} "
                  f"images={[s.key for s in contract.slots if s.kind == 'image']} status={t.status}")
        except Exception as e:
            db.rollback()
            print(f"tmpl {t.id}: ERROR {e!r}")
finally:
    db.close()
print(f"done — {changed} template(s) updated")
