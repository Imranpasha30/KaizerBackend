# Kaizer X — Custom Template Contract (v2)

Build your own video template with plain **HTML + CSS**. Upload it, and Kaizer fills in
the videos / images / text per channel and renders a finished video. This page is the
contract your template should follow so the system understands and renders it cleanly.

## 0. The two things to remember

1. **HTML + CSS only. No JavaScript, no internet.** Any `<script>`, `on*` handler,
   `<iframe>`, external `https://`/CDN link, or web font is **removed on upload** (the
   renderer is fully sandboxed, offline, for safety). Bundle every asset; style with CSS.
2. **Design for a single still frame.** By default Kaizer captures your template as **one
   still frame** composited over the moving video (fast + light). CSS animations / a
   scrolling ticker **won't move** unless the operator turns on motion rendering — so make
   it look complete and correct frozen.

## 1. What you upload

A single **`.zip`**:

```
my-template.zip
├── index.html        ← required: your template (entry point)
├── style.css         ← optional (or put <style> inline in index.html)
└── assets/           ← optional: your gifs, images, fonts
    ├── frame.png
    └── MyFont.woff2
```

A lone `index.html` also works. Reference assets with **relative paths**
(`assets/frame.png`, `./style.css`) — external URLs are stripped.

## 2. The canvas (format is dynamic)

One meta tag sets the output size (defaults to `1080x1920` if omitted; max 4096 per side):

```html
<meta name="kaizer:canvas" content="1080x1920">   <!-- 9:16 Short -->
<meta name="kaizer:canvas" content="1920x1080">   <!-- 16:9 Full video -->
```

Make `<body>` exactly that size. The **aspect decides the kind** automatically: landscape
→ a Full-form template, portrait/square → a Short template. (Kaizer can also read the size
from a viewport tag or your inline/CSS `width`/`height` if you skip the meta.)

## 3. Slots — how Kaizer fills your design

**Best practice:** mark each fillable region with a **`data-kaizer`** attribute. You can
have as many as you like, including **multiple videos**.

| `data-kaizer` value | What Kaizer does |
|---|---|
| `video` (or `video:NAME`) | Drops a clip into this box (scaled to **cover**). Leave the element **empty/transparent** — the real video is composited *behind* your design through it, so anything you draw outside the box stays on top. Multiple allowed: `video:main`, `video:guest`. |
| `background` | A full-frame video/image behind everything. Can ship a bundled default. |
| `intro` | A clip played **before** the content (cold-open). Can ship a bundled default. |
| `image` (or `image:NAME`) | Sets an `<img>` source / a box's background image. |
| `headline`, `hook`, `subtitle`, `kicker`, `caption`, `cta`, `body`, `ticker`/`marquee` | Real story text is injected here. Put the *text-bearing leaf* element (the `<span>`/`<h1>` that holds the words), not a wrapper. |
| `logo` / `watermark` | The **channel's own** logo / watermark is placed **at this spot** (scaled to fit). Omit it and the brand is stamped at a default corner instead. |
| `data-kaizer-default="assets/x.mp4"` | A bundled default the creator can swap (for `video`/`background`/`intro`/`image`). |

**You don't strictly need markers.** Kaizer's AI also reads plain layouts — semantic tags
(`<video>`, `<h1>`, `<img>`, `<marquee>`), `id`/`class` names (`id="headline"`,
`class="news-ticker"`), and even unlabelled boxes — and figures out each region. **But
explicit `data-kaizer` markers are the most reliable**, and if you mark a region the AI
never overrides it. Opaque exports with only auto-generated class names (e.g. Figma
`x7f3a`) are the one case the AI can't guess — add markers there.

```html
<!-- one video filling the frame, a headline + ticker over it -->
<div class="stage">
  <video data-kaizer="video"></video>
  <h1 data-kaizer="headline">Headline goes here</h1>
</div>
<div class="lower"><span data-kaizer="ticker">Ticker text</span></div>
```

**Tip:** name your slots and put short placeholder text inside them
(`<h1 data-kaizer="headline">Headline here</h1>`). The upload step shows one media picker
per slot labelled by name, and your placeholder is auto-replaced with the real content
(it never leaks into the final video).

## 4. Text fits the box automatically

You don't have to size text perfectly. Kaizer **auto-shrinks** each text slot's font so the
full text fits its box, and an AI writes a **concise version sized to small boxes** (a tiny
kicker gets a short label, not a shrunk paragraph). Just give each text region a sensible
box and font; Kaizer makes the real text fit.

## 5. Brand variables

These CSS custom properties are set on `:root` from the channel's branding — use them so
your template matches each creator's look:

```css
:root {
  --kaizer-brand:  #ff5a3c;   /* primary brand color */
  --kaizer-accent: #3ad1c8;
  --kaizer-text:   #ffffff;
  --kaizer-bg:     #0b0d12;
  --kaizer-font:   'KaizerFont', sans-serif;
}
```

## 6. Sharing

When you save, choose visibility — **Private** (only you) or **Public** (shared library
for every Kaizer user). Change it anytime. Each template is also rateable.
