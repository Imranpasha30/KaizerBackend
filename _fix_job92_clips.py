"""One-time backfill — rewrite job 92's Clip rows to point at the
newest youtube_full + youtube_short directories. Used after a
recompose-and-swap that didn't update Clip rows (bug fixed for
future recomposes; this is recovery for the existing breakage)."""
import sys, io, pathlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from database import SessionLocal
import models

db = SessionLocal()
j = db.query(models.Job).filter(models.Job.id == 92).first()
new_full = pathlib.Path(j.output_dir)
backend_root = pathlib.Path('.').resolve()

# Find the newest yt_short dir (non-.old) — the pair of the new yt_full.
ys_root = backend_root / 'output' / 'youtube_short'
new_short = None
if ys_root.is_dir():
    candidates = []
    nf_mt = new_full.stat().st_mtime
    for d in ys_root.iterdir():
        if not d.is_dir() or '.old' in d.name:
            continue
        gap = abs(d.stat().st_mtime - nf_mt)
        candidates.append((gap, d))
    candidates.sort()
    if candidates:
        new_short = candidates[0][1]

print('new_full:', new_full)
print('new_short:', new_short)

backslash = chr(92)

# Build prefixes to substitute.
old_full_rel = 'output' + backslash + 'youtube_full' + backslash + '20260602_154021'
old_full_abs = str((backend_root / 'output' / 'youtube_full' / '20260602_154021').resolve())
old_short_rel = 'output' + backslash + 'youtube_short' + backslash + '20260602_155205'
old_short_abs = str((backend_root / 'output' / 'youtube_short' / '20260602_155205').resolve())

subs = []
if new_full:
    new_full_rel = str(new_full.resolve().relative_to(backend_root))
    subs.append((old_full_rel, new_full_rel))
    subs.append((old_full_abs, str(new_full.resolve())))
if new_short:
    new_short_rel = str(new_short.resolve().relative_to(backend_root))
    subs.append((old_short_rel, new_short_rel))
    subs.append((old_short_abs, str(new_short.resolve())))

print('subs:')
for a, b in subs:
    print(' ', a, '->', b)

def repl(s):
    if not s: return s
    out = s
    for a, b in subs:
        if not a or not b: continue
        out = out.replace(a, b)
        # JSON-escaped paths inside meta column
        out = out.replace(a.replace(backslash, backslash + backslash),
                          b.replace(backslash, backslash + backslash))
    return out

clips = db.query(models.Clip).filter(models.Clip.job_id == 92).all()
rewritten = 0
for c in clips:
    chg = False
    for attr in ('file_path', 'thumb_path', 'image_path'):
        v = getattr(c, attr, None)
        if isinstance(v, str) and v:
            nv = repl(v)
            if nv != v:
                setattr(c, attr, nv)
                chg = True
    if c.meta:
        nm = repl(c.meta)
        if nm != c.meta:
            c.meta = nm
            chg = True
    if chg:
        rewritten += 1
db.commit()
print(f'rewrote {rewritten}/{len(clips)} clip rows')

c313 = db.query(models.Clip).filter(models.Clip.id == 313).first()
if c313:
    print('clip 313 file_path now:', c313.file_path)
    p = pathlib.Path(c313.file_path)
    if not p.is_absolute():
        p = backend_root / p
    print('  exists on disk:', p.exists())
