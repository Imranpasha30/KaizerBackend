"""
KAIZER NEWS — Simplified Web Editor (API Pipeline)
====================================================
Usage:
    python scripts/12_web_editor.py output/api_pipeline/.../editor_meta.json

Opens http://localhost:7654 in the browser.
Features:
  - Navigate all clips
  - Edit headline text inline
  - Font size + text color controls
  - Image replacement (drag-drop or click to browse)
  - Layout section drag + resize
  - Re-render clip with changes
  - Export all final clips
"""

import sys, os, json, shutil, subprocess, threading, webbrowser, time, math, random
import urllib.parse, mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS_DIR = os.path.join(BASE_DIR, "resources", "fonts")
PORT      = 7654

# ── Load metadata ────────────────────────────────────────
def load_meta(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

META_PATH = sys.argv[1] if len(sys.argv) > 1 else ""
META      = load_meta(META_PATH) if META_PATH and os.path.exists(META_PATH) else {"clips": []}


# ── Import compose functions from 11_api_pipeline ────────
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))
try:
    from importlib import util as _ilu
    _spec = _ilu.spec_from_file_location(
        "api_pipeline", os.path.join(BASE_DIR, "scripts", "11_api_pipeline.py"))
    _pipeline = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_pipeline)
    _compose_clip          = _pipeline.compose_clip
    _compose_split_frame   = _pipeline.compose_split_frame
    _compose_follow_bar    = _pipeline.compose_follow_bar
    _gen_torn_card         = _pipeline.generate_torn_paper_card
    _gen_news_card         = _pipeline.generate_news_card
    _e                     = _pipeline._e
    _get_video_info        = _pipeline.get_video_info
except Exception as _ex:
    print(f"  Warning: Could not import pipeline: {_ex}")
    _compose_clip = None


def rerender_clip(clip, edits):
    """Re-render a clip with new edits (text, font, color, section sizes, image)."""
    if _compose_clip is None:
        raise RuntimeError("Pipeline functions not loaded")

    out_dir   = os.path.dirname(clip["clip_path"])
    clip_name = os.path.splitext(os.path.basename(clip["clip_path"]))[0]
    # Always write to _edit.mp4 so raw is preserved
    new_clip  = os.path.join(out_dir, clip_name.replace("_edit", "") + "_edit.mp4")

    raw_path  = clip.get("raw_path", clip["clip_path"])
    img_path  = edits.get("image_path") or clip.get("image_path", "")

    preset = clip.get("preset") or META.get("preset", {
        "width": 1080, "height": 1920, "vertical": True
    })

    frame_type = edits.get("frame_type") or clip.get("frame_type", "torn_card")
    clip["frame_type"] = frame_type

    video_logo = edits.get("video_logo") or META.get("_custom_logo", "")

    if frame_type == "split_frame":
        split_params = edits.get("split_params") or clip.get("split_params", {})
        bg_color = split_params.get("bg_color", "#1a0a2e") if split_params else "#1a0a2e"
        _compose_split_frame(raw_path, img_path, new_clip, preset,
                             bg_color=bg_color, video_logo=video_logo or None)
        clip["split_params"] = split_params or {}
        if edits.get("image_path"):
            clip["image_path"] = edits["image_path"]

    elif frame_type == "follow_bar":
        fp = edits.get("follow_params") or clip.get("follow_params", {})
        _compose_follow_bar(
            raw_path, new_clip, preset,
            title_text=edits.get("text", clip.get("text", "")),
            font_file=edits.get("font_file", "Ponnala-Regular.ttf"),
            text_color=fp.get("text_color", "#ffff00"),
            text_size=int(edits.get("font_size", 60)),
            bg_color=fp.get("bg_color", "#1a0a2e"),
            follow_text=fp.get("follow_text", "FOLLOW KAIZER NEWS TELUGU"),
            follow_text_color=fp.get("follow_text_color", "#ffffff"),
            social_logos=fp.get("social_logos", []),
            video_logo=video_logo or None,
            velvet_style=fp.get("velvet_style"),
        )
        clip["text"] = edits.get("text", clip.get("text", ""))
        clip["follow_params"] = fp
        cp = clip.setdefault("card_params", {})
        cp.update({"font_size": edits.get("font_size", 60),
                   "font_file": edits.get("font_file", "Ponnala-Regular.ttf")})

    else:
        text      = edits.get("text", clip.get("text", "KAIZER NEWS"))
        sp        = edits.get("section_pct") or clip.get("section_pct")
        word_colors = edits.get("word_colors") or clip.get("card_params", {}).get("word_colors")
        card_style  = edits.get("card_style")  or clip.get("card_params", {}).get("card_style")

        _compose_clip(
            raw_path, img_path, text, new_clip, preset,
            font_size=edits.get("font_size"),
            text_color=edits.get("text_color"),
            font_file=edits.get("font_file") or "Ponnala-Regular.ttf",
            section_pct=sp,
            word_colors=word_colors,
            card_style={**(card_style or {}), **({"video_logo": video_logo} if video_logo else {})},
        )

        clip["text"]        = text
        clip["section_pct"] = sp or {"video": 0.4619, "text": 0.1691, "image": 0.3690}
        cp = clip.setdefault("card_params", {})
        cp.update({
            "font_size":  edits.get("font_size", cp.get("font_size", 32)),
            "font_file":  edits.get("font_file", cp.get("font_file", "Ponnala-Regular.ttf")),
            "text_color": edits.get("text_color", cp.get("text_color", "#ffffff")),
        })
        if word_colors is not None:
            cp["word_colors"] = word_colors
        if card_style is not None:
            cp["card_style"] = card_style
        if edits.get("image_path"):
            clip["image_path"] = edits["image_path"]

    clip["clip_path"] = os.path.abspath(new_clip)
    return new_clip


# ══════════════════════════════════════════════════════════
# HTML — Live Editor (Photoshop-style)
# ══════════════════════════════════════════════════════════

EDITOR_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>KAIZER NEWS &mdash; Live Editor</title>
<style>
@font-face{font-family:'KNTelugu';src:url('/fonts/NotoSansTelugu-Bold.ttf') format('truetype')}
@font-face{font-family:'KNTeluguSerif';src:url('/fonts/NotoSerifTelugu-Bold.ttf') format('truetype')}
@font-face{font-family:'KNTeluguReg';src:url('/fonts/NotoSansTelugu-Regular.ttf') format('truetype')}
@font-face{font-family:'KNTeluguSerifReg';src:url('/fonts/NotoSerifTelugu-Regular.ttf') format('truetype')}
@font-face{font-family:'Gurajada';src:url('/fonts/Gurajada-Regular.ttf') format('truetype')}
@font-face{font-family:'HindGunturBold';src:url('/fonts/HindGuntur-Bold.ttf') format('truetype')}
@font-face{font-family:'HindGunturSemiBold';src:url('/fonts/HindGuntur-SemiBold.ttf') format('truetype')}
@font-face{font-family:'HindGunturReg';src:url('/fonts/HindGuntur-Regular.ttf') format('truetype')}
@font-face{font-family:'LailaBold';src:url('/fonts/Laila-Bold.ttf') format('truetype')}
@font-face{font-family:'LailaReg';src:url('/fonts/Laila-Regular.ttf') format('truetype')}
@font-face{font-family:'NTR';src:url('/fonts/NTR-Regular.ttf') format('truetype')}
@font-face{font-family:'Ponnala';src:url('/fonts/Ponnala-Regular.ttf') format('truetype')}
@font-face{font-family:'Ramabhadra';src:url('/fonts/Ramabhadra-Regular.ttf') format('truetype')}
@font-face{font-family:'Ramaraja';src:url('/fonts/Ramaraja-Regular.ttf') format('truetype')}
@font-face{font-family:'SreeKrushnadevaraya';src:url('/fonts/SreeKrushnadevaraya-Regular.ttf') format('truetype')}
@font-face{font-family:'TenaliRamakrishna';src:url('/fonts/TenaliRamakrishna-Regular.ttf') format('truetype')}
@font-face{font-family:'Timmana';src:url('/fonts/Timmana-Regular.ttf') format('truetype')}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--acc:#c0392b;--acc2:#e74c3c;--bg:#080808;--txt:#ddd}
body{background:var(--bg);color:var(--txt);font-family:'Segoe UI',Arial,sans-serif;font-size:13px;display:flex;flex-direction:column;height:100vh;overflow:hidden}
#topbar{background:#0a0a0a;border-bottom:2px solid var(--acc);padding:0 16px;height:48px;display:flex;align-items:center;gap:10px;flex-shrink:0}
#topbar h1{color:var(--acc2);font-size:15px;letter-spacing:.1em;font-weight:700;margin-right:auto}
.tb-btn{padding:6px 14px;border:1px solid #333;border-radius:4px;cursor:pointer;font-size:12px;font-weight:600;color:#fff}
#btn-export{background:#0f3d20;color:#7fff9a;border-color:#1a5c30}
#btn-folder{background:#0d2040;color:#7ab8ff;border-color:#1a3a6b}
#status{color:#555;font-size:11px}#autosave{color:#444;font-size:10px;min-width:80px;text-align:right}
#workspace{display:flex;flex:1;min-height:0}
#panel-left{width:108px;background:#0c0c0c;border-right:1px solid #222;overflow-y:auto;padding:8px 6px;display:flex;flex-direction:column;gap:8px;flex-shrink:0}
.cthumb{cursor:pointer;border-radius:5px;overflow:hidden;border:2px solid #1e1e1e;position:relative}
.cthumb:hover{border-color:#444}.cthumb.active{border-color:var(--acc);box-shadow:0 0 8px rgba(192,57,43,.5)}
.cthumb img{width:100%;display:block;aspect-ratio:9/16;object-fit:cover;background:#111}
.cthumb .lbl{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.85);font-size:9px;padding:2px 3px;color:#bbb;text-align:center}
#panel-center{flex:1;display:flex;align-items:center;justify-content:center;padding:12px;background:#060606;overflow:hidden}
#frame{position:relative;overflow:hidden;border-radius:3px;background:#000;box-shadow:0 0 50px rgba(0,0,0,.95),0 0 2px rgba(192,57,43,.4);user-select:none;flex-shrink:0}
.sec{position:absolute;left:0;width:100%;overflow:hidden}
#sec-video{top:0;background:#000;z-index:1}
#sec-video img{width:100%;height:100%;object-fit:cover;display:block;pointer-events:none}
#logo-ov{position:absolute;top:0;right:0;z-index:5;pointer-events:none;object-fit:contain}
#sec-text{overflow:visible;z-index:10;pointer-events:none}
#torn-svg{width:100%;height:100%;display:block;overflow:visible;pointer-events:none}
#sec-image{background:#111;z-index:1}
#sec-image img{width:100%;height:100%;object-fit:cover;display:block;pointer-events:none}
.divh{position:absolute;left:0;right:0;height:14px;z-index:30;cursor:ns-resize;display:flex;align-items:center;justify-content:center}
.divh:hover{background:rgba(192,57,43,.3)}.divh.drag{background:rgba(192,57,43,.5)}
.divh::after{content:'';display:block;width:44px;height:3px;border-radius:2px;background:rgba(255,255,255,.2)}
.divh:hover::after,.divh.drag::after{background:var(--acc2)}
#sec-image.droptgt{outline:2px dashed var(--acc2)}
#panel-right{width:262px;background:#0e0e0e;border-left:1px solid #1a1a1a;overflow-y:auto;padding:12px 11px;flex-shrink:0}
.sec-hd{color:var(--acc);text-transform:uppercase;letter-spacing:.08em;font-size:10px;font-weight:700;margin:14px 0 5px;border-bottom:1px solid #1a1a1a;padding-bottom:3px}
.sec-hd:first-child{margin-top:0}
.row{display:flex;align-items:center;gap:7px;margin:3px 0}
.row label{flex:0 0 90px;color:#555;font-size:11px}
textarea#ctrl-text{width:100%;background:#141414;border:1px solid #252525;color:#ddd;padding:7px 9px;border-radius:4px;font-size:12px;font-family:inherit;min-height:55px;resize:vertical}
textarea#ctrl-text:focus{outline:1px solid var(--acc);border-color:var(--acc)}
.row input[type=range]{flex:1;accent-color:var(--acc);cursor:pointer}
.row input[type=color]{width:28px;height:24px;border:1px solid #2a2a2a;background:#111;cursor:pointer;border-radius:3px;padding:2px}
.row .val{flex:0 0 26px;text-align:right;color:#bbb;font-size:11px}
.row select{flex:1;background:#141414;border:1px solid #252525;color:#ddd;padding:4px;border-radius:4px;font-size:11px}
.img-btn{display:block;width:100%;padding:7px;background:#141414;border:1px dashed #2a2a2a;color:#555;cursor:pointer;border-radius:4px;font-size:11px;text-align:center}
.img-btn:hover{border-color:var(--acc);color:var(--acc)}
.ibox{background:#111;border:1px solid #1a1a1a;border-radius:4px;padding:8px;font-size:11px;color:#555;line-height:1.6;margin-top:4px}
.ibox b{color:#888}
.szbox{background:#111;border:1px solid #1a1a1a;border-radius:4px;padding:6px 8px;font-size:10px;color:#444;display:flex;gap:8px}
.szbox span{color:#777;font-weight:600}
/* Word chips */
#word-chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px;min-height:22px}
.wchip{cursor:pointer;padding:2px 6px;border-radius:3px;font-size:11px;border:1px solid #222;background:#111;transition:border-color .1s}
.wchip:hover{border-color:#555}
.wchip.sel{border-color:var(--acc);background:#1a0808}
#word-color-row{display:flex;align-items:center;gap:6px;margin-top:5px}
#word-color-row label{color:#555;font-size:11px;flex:1}
#ctrl-word-color{width:28px;height:24px;border:1px solid #2a2a2a;background:#111;cursor:pointer;border-radius:3px;padding:2px}
#btn-clear-wc{padding:3px 8px;background:#141414;border:1px solid #252525;color:#555;border-radius:3px;font-size:10px;cursor:pointer}
#btn-clear-wc:hover{border-color:var(--acc);color:var(--acc)}
/* ── Layout selection modal ── */
#layout-modal{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.92);z-index:500;align-items:center;justify-content:center}
#layout-modal.show{display:flex}
.lm-box{background:#141414;border:1px solid #2a2a2a;border-radius:8px;padding:28px 36px;text-align:center;min-width:380px}
.lm-title{color:var(--acc);font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin-bottom:22px}
.lm-opts{display:flex;gap:14px;justify-content:center}
.lm-btn{background:#0e0e0e;border:2px solid #2a2a2a;border-radius:6px;padding:14px 20px;cursor:pointer;color:#bbb;font-size:11px;min-width:155px;text-align:center;transition:border-color .15s,background .15s}
.lm-btn:hover{border-color:var(--acc2);background:#1a0808;color:#fff}
.lm-btn.active{border-color:var(--acc);background:#1a0808}
.lm-ico{font-size:22px;margin-bottom:7px}
.lm-name{font-weight:700;font-size:13px;margin-bottom:3px;color:#fff}
.lm-desc{font-size:10px;color:#555;line-height:1.4}
/* ── Split frame sections inside #frame ── */
#sec-sf-bg{position:absolute;top:0;left:0;width:100%;height:100%;z-index:1;display:none}
#sec-sf-thumb{position:absolute;z-index:2;overflow:hidden;background:#111;display:none}
#sec-sf-thumb img{width:100%;height:100%;object-fit:cover;display:block;pointer-events:none}
#sec-sf-video{position:absolute;z-index:2;overflow:hidden;background:#111;display:none}
#sec-sf-video img{width:100%;height:100%;object-fit:cover;display:block;pointer-events:none}
/* ── Follow bar sections inside #frame ── */
#sec-fb-bg{position:absolute;top:0;left:0;width:100%;height:100%;z-index:1;display:none}
#sec-fb-text{position:absolute;z-index:3;display:none;pointer-events:none;display:none;overflow:hidden;display:flex;flex-direction:column;align-items:center;justify-content:center}
#sec-fb-video{position:absolute;z-index:2;overflow:hidden;background:#111;display:none}
#sec-fb-video img{width:100%;height:100%;object-fit:cover;display:block;pointer-events:none}
#sec-fb-bar{position:absolute;z-index:3;display:none;background:#0d0d0d;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px}
#sec-fb-logo{position:absolute;z-index:10;pointer-events:none;display:none;object-fit:contain}
.fb-follow-txt{color:#fff;font-weight:700;text-align:center;font-size:11px;letter-spacing:.05em}
.fb-social-row{display:flex;gap:6px;align-items:center;justify-content:center}
.fb-ico{width:36px;height:36px;border-radius:50%;background:#fff;overflow:hidden;display:flex;align-items:center;justify-content:center}
.fb-ico img{width:100%;height:100%;object-fit:contain}
/* ── Logo upload slot ── */
.logo-slot{display:flex;align-items:center;gap:6px;margin:3px 0}
.logo-preview{width:40px;height:30px;object-fit:contain;background:#111;border:1px solid #222;border-radius:3px;cursor:pointer}
.logo-clear{background:none;border:none;color:#555;cursor:pointer;font-size:14px;padding:0 2px}
.logo-clear:hover{color:var(--acc)}
.social-slots{display:flex;flex-direction:column;gap:4px;margin-top:4px}
.social-slot-row{display:flex;align-items:center;gap:6px}
.social-slot-preview{width:36px;height:36px;border-radius:50%;object-fit:cover;background:#222;border:1px dashed #333;cursor:pointer}
.social-slot-preview:hover{border-color:var(--acc)}
</style></head><body>
<div id="topbar">
  <h1>&#9632; KAIZER NEWS&nbsp;<small style="color:#333;font-weight:400;font-size:11px">Live Editor</small></h1>
  <button class="tb-btn" id="btn-export" onclick="doExport()">&#8659; Export All</button>
  <button class="tb-btn" id="btn-folder" onclick="doOpenFolder()">&#128193; Folder</button>
  <span id="status">Loading&hellip;</span><span id="autosave"></span>
</div>
<div id="workspace">
  <div id="panel-left"></div>
  <div id="panel-center">
    <div id="frame">
      <div class="sec" id="sec-video"><img id="vid-thumb" src="" alt=""><img id="logo-ov" src="/logo" alt=""></div>
      <div class="divh" id="divh-top"></div>
      <div class="sec" id="sec-text">
        <svg id="torn-svg" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">
              <stop id="gStop0" offset="0%" stop-color="rgb(193,0,0)"/>
              <stop id="gStop1" offset="100%" stop-color="rgb(128,0,0)"/>
            </linearGradient>
            <clipPath id="cardClip"><path id="clip-shape"/></clipPath>
            <linearGradient id="vigLeft" x1="0" y1="0" x2="1" y2="0">
              <stop id="vlStop" offset="0%" stop-color="rgba(0,0,0,0.35)"/>
              <stop offset="100%" stop-color="transparent"/>
            </linearGradient>
            <linearGradient id="vigRight" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="transparent"/>
              <stop id="vrStop" offset="100%" stop-color="rgba(0,0,0,0.35)"/>
            </linearGradient>
            <radialGradient id="vigCornerGrad" cx="50%" cy="50%" r="70%">
              <stop offset="30%" stop-color="transparent"/>
              <stop id="vigCornerStop" offset="100%" stop-color="rgba(0,0,0,0.72)"/>
            </radialGradient>
          </defs>
          <path id="card-fill" fill="url(#bgGrad)"/>
          <path id="torn-top" fill="none" stroke="white" stroke-linejoin="round" stroke-linecap="round"/>
          <path id="torn-bot" fill="none" stroke="white" stroke-linejoin="round" stroke-linecap="round"/>
          <g id="vig-g" clip-path="url(#cardClip)">
            <rect id="vig-corner-r" x="0" y="0" width="1080" height="322" fill="url(#vigCornerGrad)"/>
            <rect id="vig-left-r"  x="0"   y="0" width="74"  height="322" fill="url(#vigLeft)"/>
            <rect id="vig-right-r" x="1006" y="0" width="74" height="322" fill="url(#vigRight)"/>
          </g>
          <g id="text-g" clip-path="url(#cardClip)"/>
        </svg>
      </div>
      <div class="divh" id="divh-bot"></div>
      <div class="sec" id="sec-image"><img id="img-preview" src="" alt=""></div>
      <!-- Split frame layout elements -->
      <div id="sec-sf-bg"></div>
      <div id="sec-sf-thumb"><img id="sf-thumb-img" src="" alt=""></div>
      <div id="sec-sf-video"><img id="sf-vid-img" src="" alt=""></div>
      <!-- Follow bar layout elements -->
      <div id="sec-fb-bg"><canvas id="fb-bg-canvas" style="position:absolute;top:0;left:0;width:100%;height:100%;display:block;"></canvas></div>
      <div id="sec-fb-text"></div>
      <div id="sec-fb-video"><img id="fb-vid-img" src="" alt=""></div>
      <div id="sec-fb-bar">
        <div class="fb-follow-txt" id="fb-follow-lbl">FOLLOW KAIZER NEWS TELUGU</div>
        <div class="fb-social-row" id="fb-social-row"></div>
      </div>
      <img id="sec-fb-logo" src="" alt="">
    </div>
  </div>
  <div id="panel-right">
    <div class="sec-hd">Headline Text</div>
    <textarea id="ctrl-text" placeholder="Edit Telugu headline..."></textarea>
    <div class="sec-hd">Word Colors <small style="color:#2a2a2a">(click word to select)</small></div>
    <div id="word-chips"></div>
    <div id="word-color-row">
      <label>Selected word color</label>
      <input type="color" id="ctrl-word-color" value="#ffff00" oninput="onWordColor()">
      <button id="btn-clear-wc" onclick="clearWordColors()">Reset all</button>
    </div>
    <div class="sec-hd">Font</div>
    <div class="row"><label>Size</label><input type="range" id="ctrl-fs" min="10" max="120" value="52" oninput="onCtrl()"><span class="val" id="fs-val">52</span></div>
    <div class="row"><label>Text color</label><input type="color" id="ctrl-color" value="#ffffff" oninput="onCtrl()"></div>
    <div class="row"><label>Font</label><select id="ctrl-font" onchange="onCtrl()">
      <option value="Ponnala-Regular.ttf" selected>Ponnala</option>
      <option value="NotoSansTelugu-Bold.ttf">Noto Sans Telugu Bold</option>
      <option value="NotoSansTelugu-Regular.ttf">Noto Sans Telugu Regular</option>
      <option value="NotoSerifTelugu-Bold.ttf">Noto Serif Telugu Bold</option>
      <option value="NotoSerifTelugu-Regular.ttf">Noto Serif Telugu Regular</option>
      <option value="HindGuntur-Bold.ttf">Hind Guntur Bold</option>
      <option value="HindGuntur-SemiBold.ttf">Hind Guntur SemiBold</option>
      <option value="HindGuntur-Regular.ttf">Hind Guntur Regular</option>
      <option value="Ramabhadra-Regular.ttf">Ramabhadra</option>
      <option value="Ramaraja-Regular.ttf">Ramaraja</option>
      <option value="TenaliRamakrishna-Regular.ttf">Tenali Ramakrishna</option>
      <option value="SreeKrushnadevaraya-Regular.ttf">Sree Krushnadevaraya</option>
      <option value="Gurajada-Regular.ttf">Gurajada</option>
      <option value="Timmana-Regular.ttf">Timmana</option>
      <option value="NTR-Regular.ttf">NTR</option>
      <option value="Mallanna-Regular.ttf">Mallanna</option>
      <option value="Mandali-Regular.ttf">Mandali</option>
      <option value="Dhurjati-Regular.ttf">Dhurjati</option>
      <option value="Laila-Bold.ttf">Laila Bold</option>
      <option value="Laila-Regular.ttf">Laila Regular</option>
      <option value="Roboto-Bold.ttf">Roboto Bold</option>
      <option value="Oswald-Bold.ttf">Oswald Bold</option>
    </select></div>
    <!-- Torn card specific controls (hidden in split_frame mode) -->
    <div id="torn-card-ctrls">
    <div class="sec-hd">Card Style</div>
    <div class="row"><label>Top color</label><input type="color" id="ctrl-card-c0" value="#c10000" oninput="onCtrl()"></div>
    <div class="row"><label>Bottom color</label><input type="color" id="ctrl-card-c1" value="#800000" oninput="onCtrl()"></div>
    <div class="row"><label>Edge height</label><input type="range" id="ctrl-edge" min="2" max="40" value="9" oninput="onCtrl()"><span class="val" id="edgev">9</span></div>
    <div class="row"><label>Jaggedness</label><input type="range" id="ctrl-jag" min="10" max="100" value="60" oninput="onCtrl()"><span class="val" id="jagv">60</span></div>
    <div class="row"><label>Edge seed</label><input type="range" id="ctrl-seed" min="0" max="99" value="7" oninput="onCtrl()"><span class="val" id="seedv">7</span></div>
    <div class="sec-hd">Vignette</div>
    <div class="row"><label>Side strength</label><input type="range" id="ctrl-vsid" min="0" max="80" value="35" oninput="onCtrl()"><span class="val" id="vsidv">35</span></div>
    <div class="row"><label>Corner strength</label><input type="range" id="ctrl-vcor" min="0" max="100" value="72" oninput="onCtrl()"><span class="val" id="vcorv">72</span></div>
    <div class="row"><label>Side width</label><input type="range" id="ctrl-vwid" min="20" max="300" value="74" oninput="onCtrl()"><span class="val" id="vwidv">74</span></div>
    </div><!-- end torn-card-ctrls -->
    <!-- Split frame specific controls -->
    <div id="split-frame-ctrls" style="display:none">
    <div class="sec-hd">Background</div>
    <div class="row"><label>BG Color</label><input type="color" id="ctrl-sf-bg" value="#1a0a2e" oninput="onSplitCtrl()"></div>
    </div><!-- end split-frame-ctrls -->
    <!-- Follow bar specific controls -->
    <div id="follow-bar-ctrls" style="display:none">
    <div class="sec-hd">Background</div>
    <div class="row"><label>BG Color</label><input type="color" id="ctrl-fb-bg" value="#1a0a2e" oninput="onFollowCtrl()"></div>
    <div class="sec-hd">Title Text</div>
    <div class="row"><label>Text color</label><input type="color" id="ctrl-fb-tc" value="#ffff00" oninput="onFollowCtrl()"></div>
    <div class="sec-hd">Follow Bar</div>
    <div class="row"><label>Follow text</label><input type="text" id="ctrl-fb-ftxt" value="FOLLOW KAIZER NEWS TELUGU" style="flex:1;background:#141414;border:1px solid #252525;color:#ddd;padding:3px 6px;border-radius:3px;font-size:11px" oninput="onFollowCtrl()"></div>
    <div class="row"><label>Text color</label><input type="color" id="ctrl-fb-ftc" value="#ffffff" oninput="onFollowCtrl()"></div>
    <div class="sec-hd">Social Logos <small style="color:#333">(up to 3)</small></div>
    <div class="social-slots">
      <div class="social-slot-row">
        <img class="social-slot-preview" id="soc-prev-0" src="" onerror="this.src=''" onclick="triggerSocialUpload(0)" title="Click to upload logo 1">
        <span style="font-size:10px;color:#555">Logo 1</span>
        <button class="logo-clear" onclick="clearSocialLogo(0)" title="Remove">&#10005;</button>
      </div>
      <div class="social-slot-row">
        <img class="social-slot-preview" id="soc-prev-1" src="" onerror="this.src=''" onclick="triggerSocialUpload(1)" title="Click to upload logo 2">
        <span style="font-size:10px;color:#555">Logo 2</span>
        <button class="logo-clear" onclick="clearSocialLogo(1)" title="Remove">&#10005;</button>
      </div>
      <div class="social-slot-row">
        <img class="social-slot-preview" id="soc-prev-2" src="" onerror="this.src=''" onclick="triggerSocialUpload(2)" title="Click to upload logo 3">
        <span style="font-size:10px;color:#555">Logo 3</span>
        <button class="logo-clear" onclick="clearSocialLogo(2)" title="Remove">&#10005;</button>
      </div>
    </div>
    <input type="file" id="soc-file-input" accept="image/*" style="display:none" onchange="onSocialFile(event)">
    </div><!-- end follow-bar-ctrls -->
    <!-- Video logo (all layouts) -->
    <div class="sec-hd">Video Logo</div>
    <div class="logo-slot">
      <img class="logo-preview" id="logo-prev" src="/logo" onerror="this.src=''" onclick="triggerLogoUpload()" title="Click to change logo">
      <span style="font-size:10px;color:#555">Top-right logo</span>
      <button class="logo-clear" onclick="clearLogo()" title="Use default">&#10227;</button>
    </div>
    <input type="file" id="logo-file-input" accept="image/*" style="display:none" onchange="onLogoFile(event)">
    <div class="sec-hd">Image</div>
    <input type="file" id="img-file" accept="image/*" style="display:none" onchange="onImgFile(event)">
    <button class="img-btn" onclick="document.getElementById('img-file').click()">&#128247; Click or drag image here</button>
    <div id="img-status" style="color:#444;font-size:10px;margin-top:3px;text-align:center"></div>
    <div id="torn-section-ctrls">
    <div class="sec-hd">Section Sizes <small style="color:#2a2a2a">(drag frame borders)</small></div>
    <div class="szbox">Video <span id="sz-v">49%</span>&nbsp; Text <span id="sz-t">17%</span>&nbsp; Image <span id="sz-i">34%</span></div>
    <div class="row"><label>Card overlap</label><input type="range" id="ctrl-overlap" min="5" max="80" value="20" oninput="onCtrl()"><span class="val" id="ovv">20</span>px</div>
    </div><!-- end torn-section-ctrls -->
    <div class="sec-hd" style="display:flex;align-items:center;justify-content:space-between">Frame Layout<button onclick="showLayoutModal(cur)" style="background:#0e0e0e;border:1px solid #2a2a2a;color:#888;padding:2px 8px;border-radius:3px;cursor:pointer;font-size:10px">Change</button></div>
    <div class="szbox" id="layout-info" style="color:#888">Torn Card</div>
    <div class="sec-hd">Clip Info</div>
    <div class="ibox" id="clip-info">Select a clip</div>
    <div class="ibox" id="clip-summary" style="max-height:80px;overflow-y:auto;margin-top:4px">&mdash;</div>
  </div>
</div>

<!-- Layout selection modal -->
<div id="layout-modal">
  <div class="lm-box">
    <div class="lm-title">&#9632; Select Frame Layout</div>
    <div class="lm-opts">
      <div class="lm-btn" id="lm-torn" onclick="selectLayout('torn_card')">
        <div class="lm-ico">&#9632;</div>
        <div class="lm-name">Torn Card</div>
        <div class="lm-desc">Video + Red card + Image<br>Classic KAIZER layout</div>
      </div>
      <div class="lm-btn" id="lm-split" onclick="selectLayout('split_frame')">
        <div class="lm-ico">&#9724;</div>
        <div class="lm-name">Split Frame</div>
        <div class="lm-desc">Thumbnail + Video<br>Colored background</div>
      </div>
      <div class="lm-btn" id="lm-follow" onclick="selectLayout('follow_bar')">
        <div class="lm-ico">&#128247;</div>
        <div class="lm-name">Follow Bar</div>
        <div class="lm-desc">Text + Square video<br>Social follow bar</div>
      </div>
    </div>
  </div>
</div>

<script>
var clips=[],META=null,cur=0,pendingImg=null;
var FW=0,FH=0,secPct={video:0.4619,text:0.1691,image:0.3690};
var saveTimer=null;
var wordColors={};   // {wordIndex: "#rrggbb"}
var selectedWords=new Set();
var SVG_W=1080, SVG_H=322;
var svgOvCur=0;
var frameType='torn_card';   // 'torn_card' | 'split_frame' | 'follow_bar'
var pendingLayoutClip=-1;
var socialLogos=['','',''];   // paths from server
var customLogo='';            // custom video logo path
var pendingSocialSlot=-1;
var FONT_CSS={
  'NotoSansTelugu-Bold.ttf':"'KNTelugu','Noto Sans Telugu',serif",
  'NotoSansTelugu-Regular.ttf':"'KNTeluguReg','Noto Sans Telugu',serif",
  'NotoSerifTelugu-Bold.ttf':"'KNTeluguSerif','Noto Serif Telugu',serif",
  'NotoSerifTelugu-Regular.ttf':"'KNTeluguSerifReg','Noto Serif Telugu',serif",
  'HindGuntur-Bold.ttf':"'HindGunturBold',sans-serif",
  'HindGuntur-SemiBold.ttf':"'HindGunturSemiBold',sans-serif",
  'HindGuntur-Regular.ttf':"'HindGunturReg',sans-serif",
  'Ponnala-Regular.ttf':"'Ponnala',sans-serif",
  'Ramabhadra-Regular.ttf':"'Ramabhadra',sans-serif",
  'Ramaraja-Regular.ttf':"'Ramaraja',sans-serif",
  'TenaliRamakrishna-Regular.ttf':"'TenaliRamakrishna',sans-serif",
  'SreeKrushnadevaraya-Regular.ttf':"'SreeKrushnadevaraya',sans-serif",
  'Gurajada-Regular.ttf':"'Gurajada',sans-serif",
  'Timmana-Regular.ttf':"'Timmana',sans-serif",
  'NTR-Regular.ttf':"'NTR',sans-serif",
  'Mallanna-Regular.ttf':"'Mallanna',sans-serif",
  'Mandali-Regular.ttf':"'Mandali',sans-serif",
  'Dhurjati-Regular.ttf':"'Dhurjati',sans-serif",
  'Laila-Bold.ttf':"'LailaBold',sans-serif",
  'Laila-Regular.ttf':"'LailaReg',sans-serif",
  'Roboto-Bold.ttf':"'Roboto',Arial,sans-serif",
  'Oswald-Bold.ttf':"'Oswald',sans-serif"
};
function ss(id){return document.getElementById(id);}
function setStatus(m){ss('status').textContent=m;}
function setAS(m){ss('autosave').textContent=m;}

/* ── PRNG (Mulberry32) ── */
function prng(seed){
  var s=seed>>>0;
  return function(){
    s+=0x6D2B79F5;
    var t=Math.imul(s^s>>>15,1|s);
    t^=t+Math.imul(t^t>>>7,61|t);
    return((t^t>>>14)>>>0)/4294967296;
  };
}

/* ── Torn edge points ── */
function tornPts(yBase,edgeH,amp,den,seed){
  var rand=prng(seed), pts=[[0,yBase]], x=0;
  while(x<SVG_W){
    var noise=(rand()*2-1)*(amp/100)*edgeH;
    var wave=edgeH*0.25*Math.sin(x*0.055);
    var jag=Math.max(0,Math.min(SVG_H-1,Math.round(yBase+noise+wave)));
    pts.push([x,jag]);
    x+=Math.max(3,Math.round(rand()*den*2));
  }
  pts.push([SVG_W,yBase]);
  return pts;
}

function ptsToD(pts){
  return 'M '+pts.map(function(p){return p[0]+','+p[1];}).join(' L ');
}

function cardPath(topPts,botPts){
  var d='M '+topPts[0][0]+','+topPts[0][1];
  for(var i=1;i<topPts.length;i++) d+=' L '+topPts[i][0]+','+topPts[i][1];
  d+=' L '+SVG_W+','+SVG_H;
  for(var j=botPts.length-1;j>=0;j--) d+=' L '+botPts[j][0]+','+botPts[j][1];
  d+=' Z';
  return d;
}

/* ── Render SVG torn card ── */
function renderCard(){
  var cardC0=ss('ctrl-card-c0').value||'#c10000';
  var cardC1=ss('ctrl-card-c1').value||'#800000';
  var edgeH=parseInt(ss('ctrl-edge').value);
  var jag=parseInt(ss('ctrl-jag').value);
  var seed=parseInt(ss('ctrl-seed').value);
  var vsid=parseInt(ss('ctrl-vsid').value)/100;
  var vcor=parseInt(ss('ctrl-vcor').value)/100;
  var vwid=parseInt(ss('ctrl-vwid').value);
  var fs=parseInt(ss('ctrl-fs').value)||32;
  var col=ss('ctrl-color').value||'#ffffff';
  var ff=ss('ctrl-font').value||'Ponnala-Regular.ttf';
  var text=ss('ctrl-text').value||'';

  /* update labels */
  /* card colors — no label spans needed, color pickers show live */
  ss('edgev').textContent=edgeH; ss('jagv').textContent=jag;
  ss('seedv').textContent=seed;
  ss('vsidv').textContent=Math.round(vsid*100);
  ss('vcorv').textContent=Math.round(vcor*100);
  ss('vwidv').textContent=vwid;
  ss('fs-val').textContent=fs;
  var ov2=parseInt(ss('ctrl-overlap').value)||0; ss('ovv').textContent=ov2;

  /* gradient */
  ss('gStop0').setAttribute('stop-color',cardC0);
  ss('gStop1').setAttribute('stop-color',cardC1);

  /* torn edges */
  var top=tornPts(edgeH,edgeH,jag,7,seed+7);
  var bot=tornPts(SVG_H-edgeH,edgeH,jag,7,seed+13);
  var cp=cardPath(top,bot);
  ss('card-fill').setAttribute('d',cp);
  ss('clip-shape').setAttribute('d',cp);
  ss('torn-top').setAttribute('d',ptsToD(top));
  ss('torn-bot').setAttribute('d',ptsToD(bot));
  ss('torn-top').setAttribute('stroke-width','2');
  ss('torn-bot').setAttribute('stroke-width','2');
  ss('torn-top').setAttribute('opacity','0.7');
  ss('torn-bot').setAttribute('opacity','0.7');

  /* vignette */
  ss('vlStop').setAttribute('stop-color','rgba(0,0,0,'+vsid.toFixed(2)+')');
  ss('vrStop').setAttribute('stop-color','rgba(0,0,0,'+vsid.toFixed(2)+')');
  ss('vigCornerStop').setAttribute('stop-color','rgba(0,0,0,'+vcor.toFixed(2)+')');
  ss('vig-left-r').setAttribute('width',vwid);
  ss('vig-right-r').setAttribute('width',vwid);
  ss('vig-right-r').setAttribute('x',SVG_W-vwid);
  ss('vig-corner-r').setAttribute('height',SVG_H);
  ss('vig-left-r').setAttribute('height',SVG_H);
  ss('vig-right-r').setAttribute('height',SVG_H);

  /* text */
  var tg=ss('text-g');
  while(tg.firstChild) tg.removeChild(tg.firstChild);
  var nsv='http://www.w3.org/2000/svg';
  var MARGIN=Math.max(32,SVG_W/28);
  var innerY=edgeH+8, innerH=SVG_H-2*edgeH-16;
  var words=text.split(/\s+/).filter(function(w){return w.length>0;});
  /* scale font to SVG internal coords (SVG_W=1080 base) */
  var scaledFs=Math.round(fs*(SVG_W/1080)*1.0);
  /* Balanced 2-line wrap: split words at the boundary that makes both lines
     as equal in character length as possible */
  var lines=[];
  if(words.length===0){
    lines=[['']];
  } else if(words.length===1){
    lines=[[words[0]]];
  } else {
    /* find split point k (1 ≤ k < n) minimising |len(line1) - len(line2)| */
    var bestSplit=1, bestDiff=Infinity;
    for(var si=1;si<words.length;si++){
      var l1=words.slice(0,si).join(' ').length;
      var l2=words.slice(si).join(' ').length;
      var diff=Math.abs(l1-l2);
      if(diff<bestDiff){bestDiff=diff;bestSplit=si;}
    }
    lines=[words.slice(0,bestSplit),words.slice(bestSplit)];
  }
  var GAP=8;
  var lineH=scaledFs*1.2;
  var total=lineH*lines.length+GAP*(lines.length-1);
  var cy=innerY+Math.max(0,(innerH-total)/2)+scaledFs;
  /* global word index for color lookup */
  var wi=0;
  lines.forEach(function(wArr){
    if(typeof wArr==='string') wArr=[wArr];
    /* measure total line width to center */
    /* use 0.55*fs per char approximation */
    var lineStr=wArr.join(' ');
    var lineW=lineStr.length*scaledFs*0.55;
    var lx=Math.max(MARGIN,(SVG_W-lineW)/2);
    /* shadow */
    var sha=document.createElementNS(nsv,'text');
    sha.textContent=lineStr;
    sha.setAttribute('x',SVG_W/2); sha.setAttribute('y',cy+3);
    sha.setAttribute('text-anchor','middle');
    sha.setAttribute('font-size',scaledFs);
    sha.setAttribute('font-weight','bold');
    sha.setAttribute('font-family',FONT_CSS[ff]||'serif');
    sha.setAttribute('fill','rgba(0,0,0,0.7)');
    tg.appendChild(sha);
    /* render each word with its color */
    var xCur=lx;
    wArr.forEach(function(w){
      var wColor=wordColors[wi]||col;
      var t=document.createElementNS(nsv,'text');
      t.textContent=w+' ';
      t.setAttribute('x',xCur); t.setAttribute('y',cy);
      t.setAttribute('font-size',scaledFs);
      t.setAttribute('font-weight','bold');
      t.setAttribute('font-family',FONT_CSS[ff]||'serif');
      t.setAttribute('fill',wColor);
      tg.appendChild(t);
      xCur+=w.length*scaledFs*0.6;
      wi++;
    });
    cy+=lineH+GAP;
  });

  /* ── Clip #sec-text to torn paper shape so video/image show through ── */
  (function(){
    var el=ss('sec-text');
    var w=el.offsetWidth||FW||SVG_W;
    var h=el.offsetHeight||1;
    var svgTotalH=SVG_H+2*svgOvCur;
    var sx=w/SVG_W;
    var sy=svgTotalH>0?h/svgTotalH:1;
    var dy=svgOvCur;
    /* transform x,y pairs in the card fill path */
    var cpath=cp.replace(/(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/g,function(m,x,y){
      return (parseFloat(x)*sx).toFixed(1)+','+((parseFloat(y)+dy)*sy).toFixed(1);
    });
    el.style.clipPath='path("'+cpath+'")';
  })();
}

/* ── Word chips ── */
function buildChips(){
  var words=(ss('ctrl-text').value||''). split(/\s+/).filter(function(w){return w.length>0;});
  var c=ss('word-chips'); c.innerHTML='';
  words.forEach(function(w,i){
    var sp=document.createElement('span');
    sp.className='wchip'+(selectedWords.has(i)?' sel':'');
    sp.textContent=w;
    sp.style.color=wordColors[i]||'#ffffff';
    sp.onclick=function(){
      if(selectedWords.has(i)) selectedWords.delete(i);
      else selectedWords.add(i);
      buildChips(); renderCard();
    };
    c.appendChild(sp);
  });
}

function onWordColor(){
  var col=ss('ctrl-word-color').value;
  selectedWords.forEach(function(i){wordColors[i]=col;});
  buildChips(); renderCard(); scheduleSave();
}
function clearWordColors(){wordColors={};selectedWords.clear();buildChips();renderCard();scheduleSave();}

/* ── Layout modal ── */
function showLayoutModal(i){
  var c=clips[i]; if(!c) return;
  /* if clip already has a layout saved, load directly without modal */
  if(c.frame_type){frameType=c.frame_type; showClip(i); return;}
  /* new clip — ask the user */
  pendingLayoutClip=i;
  ['lm-torn','lm-split','lm-follow'].forEach(function(id){ss(id).classList.remove('active');});
  ss('layout-modal').classList.add('show');
}
function selectLayout(type){
  frameType=type;
  ss('layout-modal').classList.remove('show');
  if(pendingLayoutClip>=0) showClip(pendingLayoutClip);
}
/* close modal on backdrop click */
ss('layout-modal').addEventListener('click',function(e){
  if(e.target===this) this.classList.remove('show');
});

/* ── Split frame layout ── */
function applySplitFrame(){
  if(!FW||!FH) return;
  var mx=Math.round(FW*60/1080);
  var my=Math.round(FH*60/1920);
  var tw=FW-2*mx;
  var th=Math.round(tw*9/16);
  var gap=Math.round(FH*30/1920);
  var vy=my+th+gap;
  var vh=FH-vy-my;
  var bg=ss('ctrl-sf-bg').value||'#1a0a2e';
  ss('sec-sf-bg').style.background=bg;
  var t=ss('sec-sf-thumb'); t.style.left=mx+'px'; t.style.top=my+'px';
  t.style.width=tw+'px'; t.style.height=th+'px';
  var v=ss('sec-sf-video'); v.style.left=mx+'px'; v.style.top=vy+'px';
  v.style.width=tw+'px'; v.style.height=vh+'px';
}
function onSplitCtrl(){applySplitFrame();scheduleSave();}

/* ── Velvet background renderer for Follow Bar ── */
(function(){
  var _NS=512;
  var _NT=(function(){
    var t=new Float32Array(_NS*_NS),s=987654321;
    function r(){s=((s^(s<<13))^(s>>17)^(s<<5))>>>0;return(s>>>0)/4294967296;}
    for(var i=0;i<t.length;i++) t[i]=r();
    return t;
  })();
  function _vn(x,y){
    var xi=((x%_NS)+_NS)%_NS, yi=((y%_NS)+_NS)%_NS;
    var x0=xi|0, x1=(x0+1)%_NS, y0=yi|0, y1=(y0+1)%_NS;
    var fx=xi-x0, fy=yi-y0;
    fx=fx*fx*(3-2*fx); fy=fy*fy*(3-2*fy);
    return _NT[y0*_NS+x0]*(1-fx)*(1-fy)+_NT[y0*_NS+x1]*fx*(1-fy)+
           _NT[y1*_NS+x0]*(1-fx)*fy    +_NT[y1*_NS+x1]*fx*fy;
  }
  function _wfbm(px,py,sc,oct,wa,ws){
    var wx=_vn(px/ws+3.7,py/ws+9.1)*wa-wa*0.5;
    var wy=_vn(px/ws+8.3,py/ws+2.4)*wa-wa*0.5;
    var v=0,amp=0.55,freq=1/sc,mx=0,qx=px+wx,qy=py+wy;
    for(var i=0;i<oct;i++){v+=_vn(qx*freq,qy*freq)*amp;mx+=amp;amp*=0.48;freq*=2.1;}
    return v/mx;
  }
  window.renderFollowBg=function(cnv,fw,fh,fbar_y){
    cnv.width=fw; cnv.height=fh;
    var cx=cnv.getContext('2d');
    // 1. Gradient background
    var g=cx.createLinearGradient(0,0,0,fh);
    g.addColorStop(0,'#2d0a4e'); g.addColorStop(1,'#1a0a2e');
    cx.fillStyle=g; cx.fillRect(0,0,fw,fh);
    // 2. Velvet texture on follow bar zone
    var fh2=fh-fbar_y; if(fh2<=0) return;
    var sc=fw/360.0;
    var psc=80*sc, wa=55*sc, ws=65*sc;
    var oct=5, con=1.07, bri=0.0, edD=0.33, gr=14;
    var imgd=cx.createImageData(fw,fh2), d=imgd.data;
    for(var py=0;py<fh2;py++){
      for(var px=0;px<fw;px++){
        var idx=(py*fw+px)*4;
        var ay=fbar_y+py;
        var n=_wfbm(px,ay,psc,oct,wa,ws);
        n=(n-0.5)*con+0.5+bri;
        n=n<0.5?2*n*n:1-2*(1-n)*(1-n);
        n=Math.min(1,Math.max(0,n));
        var ex=px/(fw-1), ey=py/(fh2-1);
        var ef=1-Math.min(1,ex*(1-ex)*ey*(1-ey)*22);
        n=Math.max(0,n-ef*edD*0.6);
        var gv=(Math.sin(px*127.1+ay*311.7)*43758.5453%1-0.5)*gr;
        d[idx]  =Math.min(255,Math.max(0,0x0a+(0x3d-0x0a)*n+gv));
        d[idx+1]=Math.min(255,Math.max(0,0x00+(0x00-0x00)*n+gv));
        d[idx+2]=Math.min(255,Math.max(0,0x1a+(0x60-0x1a)*n+gv));
        d[idx+3]=255;
      }
    }
    cx.putImageData(imgd,0,fbar_y,0,0,fw,fh2);
    // 3. Dot grids
    var dsp=18*sc, dr=5*sc, dop=0.38;
    cx.fillStyle='rgba(123,63,184,'+dop+')';
    var pad=dsp*0.6;
    for(var row=0;row<5;row++) for(var col=0;col<5;col++){
      cx.beginPath();cx.arc(fw-pad-col*dsp,pad+row*dsp,dr,0,Math.PI*2);cx.fill();
    }
    cx.save(); cx.beginPath(); cx.rect(0,fbar_y,fw,fh2); cx.clip();
    var bp=dsp*0.5;
    for(var row=0;row<5;row++) for(var col=0;col<5;col++){
      cx.beginPath();cx.arc(bp+col*dsp,fh-bp-row*dsp,dr,0,Math.PI*2);cx.fill();
    }
    cx.restore();
  };
})();

/* ── Follow bar layout ── */
function applyFollowBar(){
  if(!FW||!FH) return;
  var txt_mx  = Math.round(FW*79/1080);
  var top_my  = Math.round(FH*30/1920);
  var txt_h   = Math.round(FH*323/1920);
  var gap     = Math.round(FH*10/1920);
  var vid_mx  = Math.round(FW*16/1080);
  var vid_w   = FW - 2*vid_mx;
  var vid_h   = vid_w;  /* 1:1 square */
  var vid_y   = top_my + txt_h + gap;
  var fbar_y  = vid_y + vid_h;
  var fbar_h  = FH - fbar_y;
  var bg      = ss('ctrl-fb-bg').value||'#1a0a2e';
  var tc      = ss('ctrl-fb-tc').value||'#ffff00';
  var ftxt    = ss('ctrl-fb-ftxt').value||'FOLLOW KAIZER NEWS TELUGU';
  var ftc     = ss('ctrl-fb-ftc').value||'#ffffff';

  /* BG — velvet canvas */
  renderFollowBg(document.getElementById('fb-bg-canvas'),FW,FH,fbar_y);
  /* Text area */
  var te=ss('sec-fb-text');
  te.style.left=txt_mx+'px'; te.style.top=top_my+'px';
  te.style.width=(FW-2*txt_mx)+'px'; te.style.height=txt_h+'px';
  te.style.color=tc;
  te.style.fontFamily=FONT_CSS[ss('ctrl-font').value]||'Ponnala,sans-serif';
  te.style.fontSize=Math.round(parseInt(ss('ctrl-fs').value||52)*FW/1080)+'px';
  te.style.fontWeight='bold'; te.style.textAlign='center'; te.style.padding='4px 8px';
  te.style.textShadow='1px 1px 3px rgba(0,0,0,.8)';
  /* wrap text to 2 balanced lines */
  var words=(ss('ctrl-text').value||'').split(/\s+/).filter(function(w){return w;});
  var textContent='';
  if(words.length<=1){textContent=words.join('');}
  else{
    var bk=1,bd=Infinity;
    for(var si=1;si<words.length;si++){
      var d=Math.abs(words.slice(0,si).join(' ').length-words.slice(si).join(' ').length);
      if(d<bd){bd=d;bk=si;}
    }
    textContent=words.slice(0,bk).join(' ')+'<br>'+words.slice(bk).join(' ');
  }
  te.innerHTML=textContent;
  /* Video */
  var ve=ss('sec-fb-video');
  ve.style.left=vid_mx+'px'; ve.style.top=vid_y+'px';
  ve.style.width=vid_w+'px'; ve.style.height=vid_h+'px';
  /* Logo on video top-right */
  var lo=ss('sec-fb-logo');
  var lw=Math.round(vid_w*0.148), lh=Math.round(vid_h*0.14);
  lo.style.width=lw+'px'; lo.style.height=lh+'px';
  lo.style.top=(vid_y+Math.round(FH*0.013))+'px';
  lo.style.right=(vid_mx+Math.round(FW*0.022))+'px';
  lo.style.left='auto';
  lo.src=customLogo?('/media/'+customLogo):'/logo';
  /* Follow bar */
  var fb=ss('sec-fb-bar');
  fb.style.left='0'; fb.style.top=fbar_y+'px';
  fb.style.width=FW+'px'; fb.style.height=fbar_h+'px';
  ss('fb-follow-lbl').textContent=ftxt;
  ss('fb-follow-lbl').style.color=ftc;
  ss('fb-follow-lbl').style.fontSize=Math.round(fbar_h*0.18)+'px';
  /* Social icons */
  var row=ss('fb-social-row'); row.innerHTML='';
  var ico_r=Math.round(fbar_h*0.28);
  socialLogos.forEach(function(p,j){
    if(!p) return;
    var div=document.createElement('div');
    div.className='fb-ico';
    div.style.width=(ico_r*2)+'px'; div.style.height=(ico_r*2)+'px';
    var img_el=document.createElement('img');
    img_el.src='/social-logo/'+j+'?t='+Date.now();
    img_el.style.width='100%'; img_el.style.height='100%'; img_el.style.objectFit='contain';
    div.appendChild(img_el); row.appendChild(div);
  });
}
function onFollowCtrl(){applyFollowBar();scheduleSave();}

/* ── Logo upload handlers ── */
function triggerLogoUpload(){ss('logo-file-input').click();}
function onLogoFile(ev){
  var file=ev.target.files[0]; if(!file) return;
  var fd=new FormData(); fd.append('logo',file);
  fetch('/api/upload-logo',{method:'POST',body:fd}).then(r=>r.json()).then(function(d){
    if(d.path){customLogo=d.path; refreshLogo(); scheduleSave();}
  });
}
function clearLogo(){
  fetch('/api/clear-logo',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})
    .then(r=>r.json()).then(function(){customLogo=''; refreshLogo(); scheduleSave();});
}
function refreshLogo(){
  var t='?t='+Date.now();
  var src=customLogo?('/media/'+customLogo+t):('/logo'+t);
  ss('logo-ov').src=src; ss('logo-prev').src=src;
  ss('sec-fb-logo').src=src;
}
function triggerSocialUpload(slot){
  pendingSocialSlot=slot;
  ss('soc-file-input').value=''; ss('soc-file-input').click();
}
function onSocialFile(ev){
  var file=ev.target.files[0]; if(!file||pendingSocialSlot<0) return;
  var fd=new FormData(); fd.append('image',file); fd.append('slot',pendingSocialSlot);
  fetch('/api/upload-social-logo',{method:'POST',body:fd}).then(r=>r.json()).then(function(d){
    if(d.path){
      socialLogos[d.slot]=d.path;
      ss('soc-prev-'+d.slot).src='/social-logo/'+d.slot+'?t='+Date.now();
      applyFollowBar(); scheduleSave();
    }
  });
}
function clearSocialLogo(slot){
  fetch('/api/clear-social-logo',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({slot:slot})}).then(r=>r.json()).then(function(){
    socialLogos[slot]='';
    ss('soc-prev-'+slot).src='';
    applyFollowBar(); scheduleSave();
  });
}

/* ── Show/hide elements based on active layout ── */
function updateLayout(){
  var isTorn  =(frameType==='torn_card');
  var isSplit =(frameType==='split_frame');
  var isFollow=(frameType==='follow_bar');
  var TORN_IDS  =['sec-video','divh-top','sec-text','divh-bot','sec-image','logo-ov'];
  var SPLIT_IDS =['sec-sf-bg','sec-sf-thumb','sec-sf-video'];
  var FOLLOW_IDS=['sec-fb-bg','sec-fb-text','sec-fb-video','sec-fb-bar','sec-fb-logo'];

  TORN_IDS.forEach(function(id){ss(id).style.display=isTorn?'':'none';});
  SPLIT_IDS.forEach(function(id){ss(id).style.display=isSplit?'block':'none';});
  FOLLOW_IDS.forEach(function(id){ss(id).style.display=isFollow?'block':'none';});

  /* panel controls — strictly per layout */
  ss('torn-card-ctrls').style.display  =isTorn ?'':'none';
  ss('torn-section-ctrls').style.display=isTorn ?'':'none';
  ss('split-frame-ctrls').style.display =isSplit?'':'none';
  ss('follow-bar-ctrls').style.display  =isFollow?'':'none';

  /* headline text + font controls shown for torn + follow */
  var showText=isTorn||isFollow;
  ['ctrl-text','word-chips','word-color-row','ctrl-fs','ctrl-color','ctrl-font'].forEach(function(id){
    var el=ss(id); if(el) el.closest('.row,.sec-hd,textarea,#word-chips,#word-color-row') && (el.style.display='');
  });

  ss('layout-info').textContent=isTorn?'Torn Card':isSplit?'Split Frame':'Follow Bar';
  if(isSplit) applySplitFrame();
  else if(isFollow) applyFollowBar();
  else applySections();
}

/* ── INIT ── */
function init(){
  fetch('/api/meta').then(r=>r.json()).then(function(d){
    META=d; clips=d.clips||[];
    buildStrip();
    if(clips.length) requestAnimationFrame(function(){
      layoutFrame();
      showLayoutModal(0);  /* shows modal only if clip has no saved frame_type */
    });
    setStatus('Ready');
  }).catch(function(e){setStatus('Error: '+e);});
}

function buildStrip(){
  var p=ss('panel-left'); p.innerHTML='';
  clips.forEach(function(c,i){
    var d=document.createElement('div'); d.className='cthumb';
    d.onclick=function(){showLayoutModal(i);};
    var img=document.createElement('img');
    img.src='/clip/'+i+'/thumb?t='+Date.now();
    img.onerror=function(){this.style.background='#222';};
    var lbl=document.createElement('div'); lbl.className='lbl';
    lbl.textContent='Clip '+(i+1)+' ('+(c.duration||0).toFixed(0)+'s)';
    d.appendChild(img); d.appendChild(lbl); p.appendChild(d);
  });
}

function showClip(i){
  cur=i; pendingImg=null;
  var c=clips[i]; if(!c) return;
  document.querySelectorAll('.cthumb').forEach(function(el,j){el.classList.toggle('active',j===i);});

  /* restore frame type for this clip */
  frameType=c.frame_type||frameType||'torn_card';

  var sp=c.section_pct||{};
  secPct.video=sp.video||0.4619; secPct.text=sp.text||0.1691; secPct.image=sp.image||0.3690;
  /* word colors */
  wordColors=(c.card_params||{}).word_colors||{};
  selectedWords.clear();
  ss('vid-thumb').src='/clip/'+i+'/raw_thumb?t='+Date.now();
  ss('img-preview').src='/clip/'+i+'/image?t='+Date.now();
  var cp=c.card_params||{};
  ss('ctrl-text').value=c.text||'';
  var fs=cp.font_size||52;
  ss('ctrl-fs').value=fs; ss('fs-val').textContent=fs;
  ss('ctrl-color').value=cp.text_color||'#ffffff';
  var sel=ss('ctrl-font'); sel.value=cp.font_file||'Ponnala-Regular.ttf';
  if(!sel.value) sel.value='Ponnala-Regular.ttf';
  /* card style params */
  /* card gradient colors — new hex params; fallback to legacy bgr0/bgr1 (red-only) */
  ss('ctrl-card-c0').value=cp.card_c0||(cp.bgr0?'#'+('0'+Math.round(cp.bgr0).toString(16)).slice(-2)+'0000':'#c10000');
  ss('ctrl-card-c1').value=cp.card_c1||(cp.bgr1?'#'+('0'+Math.round(cp.bgr1).toString(16)).slice(-2)+'0000':'#800000');
  ss('ctrl-edge').value=cp.edge||9; ss('edgev').textContent=cp.edge||9;
  ss('ctrl-jag').value=cp.jag||60; ss('jagv').textContent=cp.jag||60;
  ss('ctrl-seed').value=cp.seed||7; ss('seedv').textContent=cp.seed||7;
  ss('ctrl-vsid').value=cp.vsid||35; ss('vsidv').textContent=cp.vsid||35;
  ss('ctrl-vcor').value=cp.vcor||72; ss('vcorv').textContent=cp.vcor||72;
  ss('ctrl-vwid').value=cp.vwid||74; ss('vwidv').textContent=cp.vwid||74;
  ss('ctrl-overlap').value=cp.overlap||20; ss('ovv').textContent=cp.overlap||20;
  /* split frame params */
  var sfp=c.split_params||{};
  ss('ctrl-sf-bg').value=sfp.bg_color||'#1a0a2e';
  ss('sf-thumb-img').src='/clip/'+i+'/image?t='+Date.now();
  ss('sf-vid-img').src='/clip/'+i+'/raw_thumb?t='+Date.now();
  /* follow bar params */
  var fbp=c.follow_params||{};
  ss('ctrl-fb-bg').value=fbp.bg_color||'#1a0a2e';
  ss('ctrl-fb-tc').value=fbp.text_color||'#ffff00';
  ss('ctrl-fb-ftxt').value=fbp.follow_text||'FOLLOW KAIZER NEWS TELUGU';
  ss('ctrl-fb-ftc').value=fbp.follow_text_color||'#ffffff';
  ss('fb-vid-img').src='/clip/'+i+'/raw_thumb?t='+Date.now();
  /* social logos */
  socialLogos=(META._social_logos||['','','']).slice(0,3);
  while(socialLogos.length<3) socialLogos.push('');
  for(var sl=0;sl<3;sl++){
    var sp_el=ss('soc-prev-'+sl);
    sp_el.src=socialLogos[sl]?('/social-logo/'+sl+'?t='+Date.now()):'';
  }
  /* custom video logo */
  customLogo=META._custom_logo||'';
  refreshLogo();
  ss('img-status').textContent='';
  ss('clip-info').innerHTML='<b>Type:</b> '+(c.video_type||META.video_type||'&mdash;')+'<br><b>Duration:</b> '+(c.duration||0).toFixed(1)+'s<br><b>Mood:</b> '+(c.mood||'&mdash;')+'<br><b>Importance:</b> '+(c.importance||'&mdash;')+'/10';
  ss('clip-summary').textContent=c.summary||'';
  buildChips(); updateLayout();
}

function layoutFrame(){
  var panel=ss('panel-center');
  var avW=panel.clientWidth-24,avH=panel.clientHeight-24;
  if(avW<=0||avH<=0){requestAnimationFrame(layoutFrame);return;}
  FW=Math.min(avW,Math.round(avH*9/16)); FH=Math.round(FW*16/9);
  var fr=ss('frame'); fr.style.width=FW+'px'; fr.style.height=FH+'px';
  var lw=Math.round(FW*0.148),lh=Math.round(FH*0.07);
  var lo=ss('logo-ov'); lo.style.width=lw+'px'; lo.style.height=lh+'px';
  lo.style.margin=Math.round(FH*0.013)+'px '+Math.round(FW*0.022)+'px 0 0';
  updateLayout();
}

function applySections(){
  if(!FH) return;
  var norm=secPct.video+secPct.text+secPct.image;
  var vH=Math.round(FH*secPct.video/norm);
  var tH=Math.max(18,Math.round(FH*secPct.text/norm));
  var iH=FH-vH-tH;
  var ov=parseInt(ss('ctrl-overlap').value)||0;
  // Video extends DOWN by ov — shows behind torn top edge
  ss('sec-video').style.height=(vH+ov)+'px';
  ss('divh-top').style.top=(vH-7)+'px';
  // Text card extends ov UP (into video) and ov DOWN (into image)
  ss('sec-text').style.top=(vH-ov)+'px'; ss('sec-text').style.height=(tH+2*ov)+'px';
  var svgOv=(tH>0?SVG_H*ov/tH:0);
  svgOvCur=svgOv;
  ss('torn-svg').setAttribute('viewBox','0 -'+svgOv+' '+SVG_W+' '+(SVG_H+2*svgOv));
  ss('divh-bot').style.top=(vH+tH-7)+'px';
  // Image extends UP by ov — shows behind torn bottom edge
  ss('sec-image').style.top=(vH+tH-ov)+'px'; ss('sec-image').style.height=(iH+ov)+'px';
  ss('sz-v').textContent=Math.round(secPct.video/norm*100)+'%';
  ss('sz-t').textContent=Math.round(secPct.text/norm*100)+'%';
  ss('sz-i').textContent=Math.round(secPct.image/norm*100)+'%';
  renderCard();
}

function updateFrame(){updateLayout();}

ss('ctrl-text').addEventListener('input',function(){buildChips();renderCard();scheduleSave();});
function onCtrl(){applySections();scheduleSave();}

function scheduleSave(){
  clearTimeout(saveTimer); setAS('\u25cf unsaved');
  saveTimer=setTimeout(doSave,1500);
}

function doSave(){
  var c=clips[cur]; if(!c) return;
  var edits={frame_type:frameType, video_logo:customLogo||null};
  if(frameType==='split_frame'){
    edits.split_params={bg_color:ss('ctrl-sf-bg').value};
  } else if(frameType==='follow_bar'){
    edits.text=ss('ctrl-text').value;
    edits.font_size=parseInt(ss('ctrl-fs').value);
    edits.font_file=ss('ctrl-font').value;
    var _existVS=(c.follow_params||{}).velvet_style||null;
    edits.follow_params={
      bg_color:ss('ctrl-fb-bg').value,
      text_color:ss('ctrl-fb-tc').value,
      follow_text:ss('ctrl-fb-ftxt').value,
      follow_text_color:ss('ctrl-fb-ftc').value,
      social_logos:socialLogos.filter(function(p){return p;}),
      velvet_style:_existVS
    };
  } else {
    edits.text=ss('ctrl-text').value;
    edits.text_color=ss('ctrl-color').value;
    edits.font_size=parseInt(ss('ctrl-fs').value);
    edits.font_file=ss('ctrl-font').value;
    edits.section_pct={video:secPct.video,text:secPct.text,image:secPct.image};
    edits.word_colors=wordColors;
    edits.card_style={card_c0:ss('ctrl-card-c0').value,card_c1:ss('ctrl-card-c1').value,
      edge:parseInt(ss('ctrl-edge').value),jag:parseInt(ss('ctrl-jag').value),
      seed:parseInt(ss('ctrl-seed').value),vsid:parseInt(ss('ctrl-vsid').value),
      vcor:parseInt(ss('ctrl-vcor').value),vwid:parseInt(ss('ctrl-vwid').value),
      overlap:parseInt(ss('ctrl-overlap').value)||0};
  }
  if(pendingImg) edits.image_path=pendingImg;
  setAS('\u23f3 saving\u2026');
  fetch('/api/rerender',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({clip_index:cur,edits:edits})
  }).then(r=>r.json()).then(function(d){
    if(d.success){
      clips[cur]=d.clip; pendingImg=null;
      document.querySelectorAll('.cthumb')[cur].querySelector('img').src='/clip/'+cur+'/thumb?t='+Date.now();
      setAS('\u2713 saved'); setStatus('Clip '+(cur+1)+' saved');
    } else {setAS('\u2717 error');setStatus('Error: '+(d.error||'?'));}
  }).catch(function(e){setAS('\u2717 error');setStatus('Failed: '+e);});
}

var dragging=null,dragY0=0,pct0={};
function initDrag(id,which){
  ss(id).addEventListener('mousedown',function(e){
    dragging=which; dragY0=e.clientY;
    pct0={video:secPct.video,text:secPct.text,image:secPct.image};
    ss(id).classList.add('drag'); e.preventDefault();
  });
}
initDrag('divh-top','top');
initDrag('divh-bot','bot');
document.addEventListener('mousemove',function(e){
  if(!dragging||!FH) return;
  var dp=(e.clientY-dragY0)/FH;
  if(dragging==='top'){
    secPct.video=Math.max(0.18,Math.min(0.75,pct0.video+dp));
    secPct.text=Math.max(0.06,pct0.text-(secPct.video-pct0.video));
    secPct.image=Math.max(0.08,1-secPct.video-secPct.text);
  } else {
    secPct.text=Math.max(0.06,Math.min(0.45,pct0.text+dp));
    secPct.image=Math.max(0.08,1-secPct.video-secPct.text);
  }
  applySections();
});
document.addEventListener('mouseup',function(){
  if(!dragging) return;
  document.querySelectorAll('.divh').forEach(function(d){d.classList.remove('drag');});
  dragging=null; scheduleSave();
});

function onImgFile(ev){
  var file=ev.target.files[0]; if(!file) return;
  ss('img-preview').src=URL.createObjectURL(file);
  ss('img-status').textContent='Uploading\u2026';
  var fd=new FormData(); fd.append('image',file); fd.append('clip_index',cur);
  fetch('/api/upload-image',{method:'POST',body:fd}).then(r=>r.json()).then(function(d){
    if(d.path){pendingImg=d.path;ss('img-status').textContent='Ready';scheduleSave();}
  }).catch(function(){ss('img-status').textContent='Upload failed';});
}
var si=ss('sec-image');
si.addEventListener('dragover',function(e){e.preventDefault();si.classList.add('droptgt');});
si.addEventListener('dragleave',function(){si.classList.remove('droptgt');});
si.addEventListener('drop',function(e){
  e.preventDefault(); si.classList.remove('droptgt');
  var file=e.dataTransfer.files[0];
  if(file&&file.type.startsWith('image/')){
    ss('img-preview').src=URL.createObjectURL(file);
    var fd=new FormData(); fd.append('image',file); fd.append('clip_index',cur);
    fetch('/api/upload-image',{method:'POST',body:fd}).then(r=>r.json()).then(function(d){
      if(d.path){pendingImg=d.path;scheduleSave();}
    });
  }
});

function doExport(){
  clearTimeout(saveTimer); doSave();
  setTimeout(function(){
    setStatus('Exporting\u2026');
    fetch('/api/export-all',{method:'POST'}).then(r=>r.json()).then(function(d){
      setStatus('Exported '+(d.count||0)+' clips \u2192 '+(d.path||'folder'));
    });
  },2500);
}
function doOpenFolder(){fetch('/api/open-folder',{method:'POST'});}
document.addEventListener('keydown',function(e){
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT') return;
  if(e.key==='ArrowLeft'&&cur>0) showLayoutModal(cur-1);
  if(e.key==='ArrowRight'&&cur<clips.length-1) showLayoutModal(cur+1);
  if((e.ctrlKey||e.metaKey)&&e.key==='s'){e.preventDefault();clearTimeout(saveTimer);doSave();}
});
window.addEventListener('resize',function(){layoutFrame();});
init();
</script>
</body></html>"""


# ══════════════════════════════════════════════════════════
# HTTP HANDLER
# ══════════════════════════════════════════════════════════

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path, ctype=None):
        if not os.path.isfile(path):
            self.send_error(404)
            return
        if not ctype:
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        size = os.path.getsize(path)
        range_header = self.headers.get("Range", "")
        if range_header and range_header.startswith("bytes="):
            # Parse range: bytes=start-end
            try:
                rng = range_header[6:].split("-")
                start = int(rng[0]) if rng[0] else 0
                end   = int(rng[1]) if len(rng) > 1 and rng[1] else size - 1
                end   = min(end, size - 1)
                chunk = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(chunk))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = chunk
                    while remaining > 0:
                        buf = f.read(min(65536, remaining))
                        if not buf:
                            break
                        self.wfile.write(buf)
                        remaining -= len(buf)
            except Exception:
                self.send_error(416)
        else:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/" or path == "":
            data = EDITOR_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        elif path == "/api/meta":
            self._json(META)

        elif path.startswith("/clip/"):
            # /clip/<idx>/thumb  or  /clip/<idx>/image  or  /clip/<idx>/video
            parts = path.split("/")
            if len(parts) >= 4:
                try:
                    idx = int(parts[2])
                    kind = parts[3]
                    clip = META.get("clips", [])[idx]
                    if kind == "thumb":
                        self._file(clip.get("thumb_path", ""))
                    elif kind == "raw_thumb":
                        raw = clip.get("raw_path", "")
                        rt = raw.replace(".mp4", "_raw_thumb.jpg") if raw else ""
                        if rt and not os.path.isfile(rt) and os.path.isfile(raw):
                            subprocess.run(["ffmpeg","-y","-ss","0","-i",raw,"-vframes","1","-q:v","2",rt],capture_output=True)
                        self._file(rt if os.path.isfile(rt) else clip.get("thumb_path", ""))
                    elif kind == "image":
                        self._file(clip.get("image_path", ""))
                    elif kind == "video":
                        self._file(clip.get("clip_path", ""))
                    elif kind == "raw":
                        self._file(clip.get("raw_path", ""))
                    else:
                        self.send_error(404)
                except (IndexError, ValueError):
                    self.send_error(404)
            else:
                self.send_error(404)

        elif path == "/logo":
            # serve custom logo if set, else default kaizer logo
            custom = META.get("_custom_logo", "")
            if custom and os.path.isfile(custom):
                self._file(custom)
            else:
                logo = os.path.join(BASE_DIR, "assests", "kaizer-logo.png")
                if os.path.exists(logo):
                    self._file(logo, "image/png")
                else:
                    self.send_error(404)

        elif path.startswith("/social-logo/"):
            # /social-logo/<n>  → serve nth social logo
            try:
                n = int(path.split("/")[-1])
                logos = META.get("_social_logos", [])
                p = logos[n] if n < len(logos) else ""
                if p and os.path.isfile(p):
                    self._file(p)
                else:
                    self.send_error(404)
            except (ValueError, IndexError):
                self.send_error(404)

        elif path.startswith("/media/"):
            fpath = urllib.parse.unquote(path[7:])
            self._file(fpath)

        elif path.startswith("/fonts/"):
            fname = urllib.parse.unquote(path[7:])
            fpath = os.path.join(FONTS_DIR, fname)
            self._file(fpath, "font/ttf")

        else:
            self.send_error(404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))

        if path == "/api/rerender":
            body = json.loads(self.rfile.read(length))
            idx = body.get("clip_index", 0)
            edits = body.get("edits", {})
            clip = META["clips"][idx]
            clip["preset"] = META.get("preset", {"width": 1080, "height": 1920})

            try:
                new_path = rerender_clip(clip, edits)
                # Regenerate thumbnail
                thumb_path = new_path.replace(".mp4", "_thumb.jpg")
                try:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", new_path,
                        "-vframes", "1", "-q:v", "2", thumb_path
                    ], capture_output=True, check=True)
                    clip["thumb_path"] = os.path.abspath(thumb_path)
                except Exception:
                    pass

                # Save updated metadata
                with open(META_PATH, "w", encoding="utf-8") as f:
                    json.dump(META, f, ensure_ascii=False, indent=2)

                self._json({"success": True, "clip": clip})
            except Exception as e:
                self._json({"success": False, "error": str(e)}, 500)

        elif path == "/api/upload-image":
            # Parse multipart form data
            content_type = self.headers.get("Content-Type", "")
            if "multipart" not in content_type:
                self._json({"error": "Expected multipart"}, 400)
                return

            boundary = content_type.split("boundary=")[1].encode()
            data = self.rfile.read(length)

            # Simple multipart parser
            parts = data.split(b"--" + boundary)
            img_data = None
            clip_idx = 0
            for part in parts:
                if b"filename=" in part:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end >= 0:
                        img_data = part[header_end + 4:]
                        if img_data.endswith(b"\r\n"):
                            img_data = img_data[:-2]
                elif b'name="clip_index"' in part:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end >= 0:
                        try:
                            clip_idx = int(part[header_end + 4:].strip().split(b"\r\n")[0])
                        except Exception:
                            pass

            if img_data:
                clip = META["clips"][clip_idx] if clip_idx < len(META["clips"]) else None
                out_dir = os.path.dirname(clip["clip_path"]) if clip else os.path.dirname(META_PATH)
                img_path = os.path.join(out_dir, f"user_image_{clip_idx + 1:02d}.jpg")

                with open(img_path, "wb") as f:
                    f.write(img_data)

                abs_path = os.path.abspath(img_path)
                if clip:
                    clip["image_path"] = abs_path
                self._json({"path": abs_path})
            else:
                self._json({"error": "No image data"}, 400)

        elif path == "/api/upload-logo":
            content_type = self.headers.get("Content-Type", "")
            if "multipart" not in content_type:
                self._json({"error": "Expected multipart"}, 400); return
            boundary = content_type.split("boundary=")[1].encode()
            data = self.rfile.read(length)
            parts = data.split(b"--" + boundary)
            img_data = None
            for part in parts:
                if b"filename=" in part:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end >= 0:
                        img_data = part[header_end + 4:]
                        if img_data.endswith(b"\r\n"): img_data = img_data[:-2]
            if img_data:
                out_dir = os.path.dirname(META_PATH)
                logo_path = os.path.join(out_dir, "custom_logo.png")
                with open(logo_path, "wb") as f: f.write(img_data)
                META["_custom_logo"] = os.path.abspath(logo_path)
                with open(META_PATH, "w", encoding="utf-8") as f:
                    json.dump(META, f, ensure_ascii=False, indent=2)
                self._json({"path": os.path.abspath(logo_path)})
            else:
                self._json({"error": "No data"}, 400)

        elif path == "/api/upload-social-logo":
            content_type = self.headers.get("Content-Type", "")
            if "multipart" not in content_type:
                self._json({"error": "Expected multipart"}, 400); return
            boundary = content_type.split("boundary=")[1].encode()
            data = self.rfile.read(length)
            parts = data.split(b"--" + boundary)
            img_data = None; slot = 0
            for part in parts:
                if b"filename=" in part:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end >= 0:
                        img_data = part[header_end + 4:]
                        if img_data.endswith(b"\r\n"): img_data = img_data[:-2]
                elif b'name="slot"' in part:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end >= 0:
                        try: slot = int(part[header_end + 4:].strip().split(b"\r\n")[0])
                        except: pass
            if img_data:
                out_dir = os.path.dirname(META_PATH)
                logo_path = os.path.join(out_dir, f"social_logo_{slot}.png")
                with open(logo_path, "wb") as f: f.write(img_data)
                logos = META.setdefault("_social_logos", ["", "", ""])
                while len(logos) <= slot: logos.append("")
                logos[slot] = os.path.abspath(logo_path)
                with open(META_PATH, "w", encoding="utf-8") as f:
                    json.dump(META, f, ensure_ascii=False, indent=2)
                self._json({"path": os.path.abspath(logo_path), "slot": slot})
            else:
                self._json({"error": "No data"}, 400)

        elif path == "/api/clear-logo":
            META.pop("_custom_logo", None)
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(META, f, ensure_ascii=False, indent=2)
            self._json({"ok": True})

        elif path == "/api/clear-social-logo":
            body = json.loads(self.rfile.read(length))
            slot = int(body.get("slot", 0))
            logos = META.get("_social_logos", ["", "", ""])
            if slot < len(logos): logos[slot] = ""
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(META, f, ensure_ascii=False, indent=2)
            self._json({"ok": True})

        elif path == "/api/export-all":
            export_dir = os.path.join(os.path.dirname(META_PATH), "export")
            os.makedirs(export_dir, exist_ok=True)
            count = 0
            for i, clip in enumerate(META.get("clips", [])):
                src = clip.get("clip_path", "")
                if src and os.path.exists(src):
                    dst = os.path.join(export_dir, f"kaizer_clip_{i+1:02d}.mp4")
                    shutil.copy2(src, dst)
                    count += 1
            self._json({"count": count, "path": export_dir})

        elif path == "/api/open-folder":
            folder = os.path.dirname(META_PATH)
            try:
                if sys.platform == "win32":
                    os.startfile(folder)
                else:
                    subprocess.Popen(["xdg-open", folder])
            except Exception:
                pass
            self._json({"ok": True})

        else:
            self.send_error(404)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    if not META_PATH:
        print("Usage: python scripts/12_web_editor.py <editor_meta.json>")
        sys.exit(1)

    print(f"  KAIZER NEWS Editor")
    print(f"  Clips: {len(META.get('clips', []))}")
    print(f"  Meta:  {META_PATH}")

    server = HTTPServer(("0.0.0.0", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"  URL:   {url}")

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Editor stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
