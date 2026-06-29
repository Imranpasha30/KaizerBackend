"""End-to-end: an element with data-kx-anim="slide_left" enters (slides in), not static.

A red box at (100,100,400,120) marked slide_left over 1s. After render: the box's final-spot
left edge is EMPTY at t=0.05 (still sliding in from the right) and RED at t=2.0 (settled) —
proving the entrance animates instead of the box being baked statically from t=0.

Run:  venv/Scripts/python.exe -m scripts.test_element_entrance
"""
from __future__ import annotations
import os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BX, BY, BW, BH = 100, 100, 400, 120

_HTML = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta name="kaizer:canvas" content="1280x720">
<style>html,body{{width:1280px;height:720px;margin:0;background:#0a1326;}}
.hl{{position:absolute;left:{BX}px;top:{BY}px;width:{BW}px;height:{BH}px;background:#dc1e1e;}}
.v{{position:absolute;left:100px;top:380px;width:500px;height:260px;background:#000;}}</style></head>
<body>
<div class="hl" data-kx-anim="slide_left" data-kx-anim-dur="1.0"></div>
<div class="v" data-kaizer="video"></div>
</body></html>"""


def _clip(path):
    ff = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run([ff, "-y", "-f", "lavfi", "-i", "color=c=gray:s=640x360:rate=30:d=3",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path],
                   capture_output=True, timeout=120)


def _pixel(mp4, t, x, y):
    from PIL import Image
    ff = shutil.which("ffmpeg") or "ffmpeg"
    tmp = mp4 + f".f{int(t*100)}.png"
    subprocess.run([ff, "-y", "-ss", str(t), "-i", mp4, "-frames:v", "1", tmp], capture_output=True, timeout=120)
    return Image.open(tmp).convert("RGB").getpixel((x, y))


def _is_red(p): return p[0] > 150 and p[1] < 90 and p[2] < 90


def main() -> int:
    import services.custom_templates as ct
    tmp = tempfile.mkdtemp(prefix="kx_anim_")
    with open(os.path.join(tmp, "t.html"), "w", encoding="utf-8") as fh:
        fh.write(_HTML)
    clip = os.path.join(tmp, "clip.mp4"); _clip(clip)
    b = ct.Bundle(root_dir=tmp, entry_rel="t.html", files=[])
    _n, contract = ct.normalize_and_discover(_HTML)
    out = os.path.join(tmp, "out.mp4")
    rep = ct.render_template(b, contract, ct.RenderRequest(videos={"video": clip}, main_slot="video", fps=30),
                             work_dir=os.path.join(tmp, "r"), out_path=out)
    print("report anim_entrances:", rep.get("anim_entrances"), "| out:", os.path.isfile(out))
    assert rep.get("anim_entrances") == 1, "engine did not composite an element entrance"
    lx, ly = BX + 30, BY + BH // 2      # near the LEFT edge of the box's final spot
    rx2, ry2 = BX + BW + 60, BY + BH // 2  # to the RIGHT of the final spot (where it slides from)
    p_left_start = _pixel(out, 0.05, lx, ly)   # entrance in progress -> not red yet (bg)
    p_right_start = _pixel(out, 0.05, rx2, ry2) # box is offset right at the start
    p_left_end = _pixel(out, 2.0, lx, ly)      # settled -> red
    print(f"left@0.05={p_left_start} (expect bg)  right@0.05={p_right_start} (expect red, offset)  left@2.0={p_left_end} (expect red)")
    ok = (not _is_red(p_left_start)) and _is_red(p_right_start) and _is_red(p_left_end)
    print("PASS ✓ element slides IN (not static from t=0)" if ok else "FAIL: entrance did not animate")
    assert ok
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("ENTRANCE TEST ERROR:", repr(e)); raise
