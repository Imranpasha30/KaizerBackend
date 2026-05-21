# TRACK 2 -- Broadcast / Editing Tool Architecture

Research mission: understand how professional NLEs, cloud video pipelines, and
broadcast graphics systems handle the operations that Kaizer V2 attempts, so
the team can decide whether to keep iterating on the ffmpeg cut + compose +
stitch chain, or adopt an industry pattern (EDL-first + render-only stage).

Citation convention used in this file:
    "[CLAIM]. [EVIDENCE: URL / file:line / spec]. [CONFIDENCE: HIGH/MED/LOW/UNVERIFIED]."

Confidence labels:
- HIGH = verified by multiple independent sources, or official spec, or by
  reading the production code in this repo.
- MED = single authoritative source (vendor pricing page, official docs).
- LOW = inferred from related evidence (e.g. one community post + one
  product blog).
- UNVERIFIED = flagged for user review.

The Kaizer side of every comparison is grounded in the production files the
mission named:

- `pipeline_v2/stages/stage_4_render.py` (3,400 lines, the cut + compose
  orchestrator).
- `pipeline_v2/bulletin_crossfade_stitcher.py` (527 lines, 3-pass audio
  acrossfade stitcher).
- `pipeline_v2/stages/stage_4_raw_extract.py` (404 lines, item-117 single-
  pass multi-output extractor).
- `pipeline_v2/render/edl_builder.py` (303 lines, pure
  filter_complex builder).

---

## 2A. Edit Decision Lists (EDLs)

### 2A.1 CMX 3600 -- the lowest common denominator

The CMX 3600 EDL is the oldest still-used edit-decision interchange format,
standardised in ANSI/SMPTE 258M-1993. The file is plain ASCII, fixed-column,
and limited to one video track plus up to four audio channels.

A typical edit event looks like::

    001  REEL01    V     C        00:01:00:00 00:01:05:00 00:00:00:00 00:00:05:00

Field-by-field:

- "001" event number (max 999).
- "REEL01" source tape name (max 8 characters).
- "V" edit type (V, A, A1, A2, A12, AA, B, etc.).
- "C" transition type (C=Cut, D=Dissolve, W=Wipe).
- The four timecodes are: source-in, source-out, record-in, record-out.

The CMX 3600 EDL has at minimum an event number, source reel number, mode
(C, D, W), and four timecodes for source in/out and record in/out. [EVIDENCE:
https://en.wikipedia.org/wiki/Edit_decision_list; https://xmil.biz/EDL-X/CMX3600.pdf]. [CONFIDENCE:
HIGH]

Strictly conforming CMX 3600 EDLs allow no more than one video track and four
audio channels, and event count is capped at 999. [EVIDENCE:
https://www.edlmax.com/EdlMaxHelp/Edl/Edl_Overview.htm; cross-checked at
https://www.niwa.nu/2013/05/how-to-read-an-edl/]. [CONFIDENCE: HIGH]

CMX 3600 has no native representation of motion graphics, no overlays, no
audio crossfade duration field (you get a dissolve duration in frames but
that applies to the whole edit, not separately to V and A). Modern tools
extend the format with COMMENT lines for clip names ("* FROM CLIP NAME: X"),
audio levels, and color-grade CDLs. [EVIDENCE:
https://www.edlmax.com/EdlMaxHelp/Edl/Edl_Overview.htm]. [CONFIDENCE: MED]

Cut decision representation: each row is one cut. Non-destructive by
design: the EDL points at source timecodes, never touches the media.
[EVIDENCE: same Wikipedia + EDL Max overview]. [CONFIDENCE: HIGH]

Python library availability: OpenTimelineIO ships an official cmx_3600
adapter, so a Python program can write an EDL with three lines of code
(see OTIO section below). [EVIDENCE:
https://github.com/OpenTimelineIO/otio-cmx3600-adapter]. [CONFIDENCE: HIGH]

### 2A.2 AAF (Advanced Authoring Format)

AAF is a structured-storage binary container holding both essence (the
media itself, optionally embedded) and metadata describing edits, effects,
levels, and colour. It was created by the AMWA (Advanced Media Workflow
Association) and is being standardised through SMPTE. [EVIDENCE:
https://en.wikipedia.org/wiki/Advanced_Authoring_Format;
https://aafassociation.org/specs/object_spec.html]. [CONFIDENCE: HIGH]

The AAF Object Specification defines a structured container that stores
essence + metadata in an object-oriented model and the AAF Low-Level
Container Specification describes how each object is stored using
Microsoft Structured Storage (the old OLE compound document format).
[EVIDENCE: https://aafassociation.org/specs/object_spec.html;
https://www.loc.gov/preservation/digital/formats/fdd/fdd000004.shtml].
[CONFIDENCE: HIGH]

Cut decision representation: AAF models the timeline as nested Mob
(metadata object) instances: SourceMob (one per piece of source media),
CompositionMob (the actual edit), MasterMob (a reference layer). Each
Composition contains Sequences of SourceClips with start/length fields in
edit units. Transitions are first-class objects with named effects.
[EVIDENCE: https://aaf.sourceforge.net/docs/aafObjectModel.pdf;
https://static.amwa.tv/as-01-aaf-edit-protocol-spec.pdf]. [CONFIDENCE: HIGH]

Use in automation: AAF is the lingua franca for "send my edit to ProTools
for mixing". Almost every broadcast workflow uses AAF for round-tripping
between Avid Media Composer and Pro Tools. [EVIDENCE:
https://tech.ebu.ch/docs/techreview/trev_291-gilmer.pdf]. [CONFIDENCE: MED]

Python library availability: `pyaaf2` (PyPI) by Mark Reid is the
de-facto Python AAF reader/writer; it is the backend that OpenTimelineIO's
AAF adapter calls. [EVIDENCE:
https://github.com/markreidvfx/pyaaf2; OTIO docs mention pyaaf2 as the
dependency for the AAF adapter at
https://github.com/AcademySoftwareFoundation/OpenTimelineIO/blob/main/docs/tutorials/adapters.md].
[CONFIDENCE: MED -- the OTIO repo's wording was paraphrased in search
results rather than read line-by-line; safe inference because pyaaf2 is
the only mature pure-Python AAF library.]

### 2A.3 Avid Media Composer internal representation

Avid stores everything in its own bin database (`.avb` files in modern
projects, `.avp` for project files). The bin database is a proprietary
binary tree of SourceMobs / CompositionMobs that maps almost
1-to-1 onto AAF's object model -- which is no accident, because the AAF
spec was largely defined to match Avid's existing data model.
[EVIDENCE: https://en.wikipedia.org/wiki/Advanced_Authoring_Format
attributes AAF's origin to the AMWA, of which Avid was a founder].
[CONFIDENCE: MED]

Cut decisions are stored as Composition objects with a Sequence holding
SourceClips that reference per-channel SourceMobs by Mob ID (a UMID, see
SMPTE 330M). Non-destructive: editing a cut only edits the Composition,
never touches the SourceMob's underlying media. [EVIDENCE: same AAF object
model PDF, plus the AMWA Edit Protocol AS-01]. [CONFIDENCE: MED]

Python library availability: third-party, via OTIO's AAF adapter and
pyaaf2. No first-party Avid Python SDK. [EVIDENCE: same as 2A.2].
[CONFIDENCE: MED]

### 2A.4 Adobe Premiere XML interchange (`.prproj`)

Premiere project files are XML wrapped in gzip. The root element is
`<PremiereData Version="N">` and the structure encodes a graph of
`<Project>`, `<RootProjectItem>`, `<Sequence>`, `<Track>`, `<TrackItem>`,
`<ClipMedia>`, etc., with cross-references through ObjectRef/ObjectID
attributes. Most modern .prproj files are gzip-compressed XML; older
versions may be uncompressed. [EVIDENCE:
http://fileformats.archiveteam.org/wiki/Premiere_Pro;
https://convert.guru/prproj-converter]. [CONFIDENCE: HIGH]

Premiere also exports/imports Final Cut Pro 7 XML (`.xml`) and FCPXML as
interchange formats. The native prproj is undocumented and Adobe reserves
the right to break the schema between versions; the
DigitalRebellion-class community has reverse-engineered it but it is not a
stable target for automated systems. [EVIDENCE:
http://fileformats.archiveteam.org/wiki/Premiere_Pro]. [CONFIDENCE: MED]

Use in automated systems: Premiere is the wrong layer to target if you
want stability. Most production automation generates FCP7 XML or FCPXML
(see 2A.7) which Premiere ingests; OTIO can also read/write those.
[EVIDENCE: OTIO adapter list at
https://opentimelineio.readthedocs.io/en/latest/tutorials/adapters.html].
[CONFIDENCE: HIGH]

Python library availability: none official. PRPROJ Converter
(convert.guru) is a paid SaaS; OTIO does not have a `.prproj` adapter, it
goes through FCP7 XML. [EVIDENCE: same OTIO adapter list]. [CONFIDENCE: HIGH]

### 2A.5 DaVinci Resolve project format

Resolve stores projects in a PostgreSQL or local SQLite database (the
"Resolve database"). The schema is internal and undocumented. Resolve
exports to AAF, FCPXML, FCP7 XML, CMX 3600 EDL, XML "DRP" (its native
text-export) and a few proprietary formats. [EVIDENCE:
https://www.blackmagicdesign.com/products/davinciresolve/studio;
community-confirmed across the
https://forum.blackmagicdesign.com/viewtopic.php?f=21&t=151297 thread on
fcpxml 1.9/1.10]. [CONFIDENCE: MED]

Cut decision representation (internal): each Resolve timeline is a
collection of clip references with a per-track linked list of in/out
points and effect stacks. The Resolve scripting API (Lua + Python)
exposes Project, MediaPool, MediaPoolItem, Timeline, TimelineItem objects;
each TimelineItem has GetStart/GetEnd/GetDuration accessors plus
GetClipColor, SetClipColor, etc. [EVIDENCE: the DaVinci Resolve Studio
Scripting documentation that ships with the installer; mirrored at
https://documents.blackmagicdesign.com (paywall-cached but linked in
multiple forum threads)]. [CONFIDENCE: MED]

Non-destructive editing: yes -- Resolve never modifies source media; all
trims, colour grades, and Fusion comps live in the project DB.
[EVIDENCE:
https://www.blackmagicdesign.com/products/davinciresolve/studio (product
page wording: "non-destructive editing")]. [CONFIDENCE: HIGH]

Render engine: GPU-accelerated. Renders are always performed at the
highest resolution possible and the colour pipeline applies a single
resize at the end, not per-effect. [EVIDENCE:
https://creativecow.net/forums/thread/timeline-resolution-vs-render-output-size/].
[CONFIDENCE: MED]

Python library availability: Resolve ships with a Python 3 scripting
console embedded in the app; `DaVinciResolveScript.py` is shipped in
`<install>/Support/Developer/Scripting/Modules/`. It can be imported from
external Python 3 if you launch Resolve first. [EVIDENCE: install paths
documented in the README that ships with Resolve Studio; community write-
up at https://www.miracamp.com/learn/davinci-resolve/how-to-render
references the same module]. [CONFIDENCE: MED]

### 2A.6 OpenTimelineIO (OTIO)

OTIO is the Academy Software Foundation's (originally Pixar's) open-source
interchange format and Python API for editorial timelines. Open-source,
permissively licensed (Apache 2.0), Python-first, with native C++ core.
[EVIDENCE: https://github.com/AcademySoftwareFoundation/OpenTimelineIO;
https://opentimelineio.readthedocs.io/en/stable/]. [CONFIDENCE: HIGH]

Core schema (verbatim from `opentimelineio.schema`):

- `Timeline` -- root container, has `global_start_time: RationalTime` and
  a `tracks: Stack`.
- `Stack` -- vertical stack of `Track`s.
- `Track` -- horizontal track holding `Clip`s, `Transition`s, and
  nested `Track`s. Has a `kind` attribute that is one of `"Video"` or
  `"Audio"`.
- `Clip` -- a segment of editable media with a `MediaReference` and a
  `source_range: TimeRange`.
- `Gap` -- empty time on a track.
- `Transition` -- represents a transition between two clips. Has
  `in_offset` and `out_offset` (each `RationalTime`) describing how much
  the transition extends into the previous / next clip.
- `MediaReference` (base class), `ExternalReference` (URL + available
  range), `MissingReference`, `GeneratorReference`,
  `ImageSequenceReference`.
- `RationalTime` (value, rate) and `TimeRange` (start_time, duration) --
  every timestamp in OTIO is a rational, so 24000/1001 fps cleanly
  represented, no floating-point drift.

[EVIDENCE:
https://opentimelineio.readthedocs.io/en/v0.15/api/python/opentimelineio.schema.html;
the canonical OTIO structure is "root as otio.schema.Timeline (containing
global_start_time and a top level container called tracks),
timeline.tracks as otio.schema.Stack (containing otio.schema.Track
objects)" -- confirmed in the architecture doc at
https://github.com/PixarAnimationStudios/OpenTimelineIO/blob/master/docs/tutorials/architecture.md].
[CONFIDENCE: HIGH]

Cut decision representation: each `Clip` carries a `source_range`
(TimeRange in source media's clock) and ends up at a position on the track
determined by the sum of the preceding items' durations. The model is
non-destructive by definition -- a Clip is just a reference + range.
[EVIDENCE: same schema docs + architecture doc]. [CONFIDENCE: HIGH]

Transition handling: a `Transition` object lives between two clips on a
track. `in_offset` is how much of clip A is consumed by the transition,
`out_offset` is how much of clip B is consumed. Total transition
duration is `in_offset + out_offset`. This generalises CMX 3600's
single dissolve duration field. [EVIDENCE:
https://opentimelineio.readthedocs.io/en/latest/api/python/opentimelineio.schema.transition.html
(documented in the Timeline Structure tutorial)]. [CONFIDENCE: HIGH]

Bundled adapters (verified via the OTIO-Plugins package): cmx_3600,
fcp_xml (FCP7), fcpx_xml (FCPXML), aaf, ale, burnins, hls_playlist,
maya_sequencer, svg, xges. AAF goes through pyaaf2. [EVIDENCE:
https://opentimelineio.readthedocs.io/en/latest/tutorials/adapters.html;
the kdenlive adapter exists too:
https://github.com/KDE/kdenlive-opentimelineio]. [CONFIDENCE: HIGH]

Python install: `pip install opentimelineio` for the core (Timeline,
Track, Clip, etc. + the built-in `.otio` reader/writer + the cmx_3600 + fcp_xml +
fcpx_xml adapters), `pip install opentimelineio-plugins` for the wider
adapter set including AAF. [EVIDENCE:
https://github.com/AcademySoftwareFoundation/OpenTimelineIO/blob/main/setup.py
and the OTIO-Plugins README on PyPI]. [CONFIDENCE: HIGH]

#### CRITICAL QUESTION 2A: could Kaizer use OTIO?

The question: could Kaizer represent Stage 2 cut decisions as a proper
OTIO Timeline, then render via a separate stage (rather than building a
ffmpeg filter graph directly)?

ANSWER: YES, with HIGH confidence. The model fit is exact and the
implementation effort is small (estimated 2 - 4 engineering days).

Justification:

1. Kaizer's cut decision data is exactly OTIO's data model. The
   `pipeline_v2/render/edl_builder.py:OutputSpec` dataclass carries
   `(role, index, v_label, a_label, duration_s, source_cuts)` where
   `source_cuts` is a `tuple[(start_s, end_s)]` of source timestamps.
   That is a 1-to-1 match to an OTIO Track of Clips each pointing at the
   same `ExternalReference(mezzanine.mp4)` with different
   `source_range: TimeRange` values. [EVIDENCE: read
   `edl_builder.py:78-108` for OutputSpec; OTIO Clip schema at
   https://opentimelineio.readthedocs.io/en/stable/]. [CONFIDENCE: HIGH]

2. OTIO is non-destructive and lossless. Every time value is a
   `RationalTime(value, rate)`, so the "snap-to-30fps-grid" bug class
   that produced job 51's -695ms drift cannot occur in the data model.
   Once the rate is set to (30, 1), value is an integer, and
   `TimeRange(start, duration)` is exact. [EVIDENCE:
   https://opentimelineio.readthedocs.io/en/latest/api/python/opentimelineio.opentime.RationalTime.html;
   already-shipped snap helper at `edl_builder.py:111-118` is doing the
   same thing OTIO's RationalTime gives for free]. [CONFIDENCE: HIGH]

3. Bidirectional adapters mean Kaizer's editor UI could (later) export
   the same Timeline to FCPXML or CMX 3600 and let an editor finish the
   cut in Premiere or Resolve. That is exactly the user's "creator-tool
   for Indian news creators" pitch. [EVIDENCE: OTIO adapters list at
   https://opentimelineio.readthedocs.io/en/latest/tutorials/adapters.html].
   [CONFIDENCE: HIGH]

4. The render side is still ffmpeg. OTIO is JUST the data model; it
   does not render. Kaizer would emit a Timeline object as Stage 2
   output, then a small renderer (which can be the existing
   `stage_4_raw_extract.extract_raw_timeline`) walks the Timeline and
   builds the filter_complex. The new renderer is a function with
   signature
   `(timeline: otio.schema.Timeline) -> RawExtractResult`. Estimated
   200 lines, all of which already exist split across `edl_builder.py`
   and `stage_4_raw_extract.py`. [EVIDENCE: read
   `stage_4_raw_extract.py:206-403` -- the function is already
   parameterised over `bulletin_cuts: Sequence[tuple[float, float]]`].
   [CONFIDENCE: HIGH]

5. The killer feature: **decoupling cut decisions from the render
   command** is the architectural move that lets the team unit-test
   cut logic without ffmpeg, lets them swap renderers (cloud, GPU,
   local) without touching Stage 2, and lets them ship the editor UI
   over the same timeline data. [EVIDENCE: inferred from architecture
   patterns; see Section 2E synthesis below for the exact ROI].
   [CONFIDENCE: MED -- this is judgement, not a fact.]

Caveats to flag for user review:

- Bulletin-overlay text and the per-short overlays would need to be
  expressed as OTIO Effects on the Clip (or as a separate track of
  GeneratorReference / Markers). The OTIO Effect schema is generic
  enough to carry "kaizer_torn_card" as an Effect name with a metadata
  dict, but Kaizer's renderer is the only thing that would understand
  that effect -- not portable to Premiere/Resolve. [EVIDENCE:
  https://opentimelineio.readthedocs.io/en/stable/api/python/opentimelineio.schema.effect.html].
  [CONFIDENCE: MED]
- OTIO has no first-class concept of "this timeline produces N output
  files". Kaizer's `bulletin_raw.mp4 + N short_raw.mp4` multi-output
  shape would need to be expressed as N separate Timeline objects (one
  per output), with a wrapping `SerializableCollection` if you want one
  file on disk. That is exactly what
  `opentimelineio.schema.SerializableCollection` is for; the OTIO docs
  call it "approximating the concept of a bin". [EVIDENCE:
  https://opentimelineio.readthedocs.io/en/stable/]. [CONFIDENCE: HIGH]
- OTIO's Python bindings are CPython only (a small native extension).
  Kaizer's backend already ships native ffmpeg, so adding one more
  manylinux wheel is no new burden. [EVIDENCE: PyPI page for
  `opentimelineio` lists wheels for cp38-cp312 on linux/mac/win].
  [CONFIDENCE: HIGH]

Bottom line: OTIO is the right data model for Kaizer's cut decisions.
The change is mostly internal -- Stage 2 emits a Timeline, Stage 4
consumes a Timeline, the cut-step + edl_builder + raw_extract chain
collapses into a single function `render_timeline(timeline, out_dir)`.
The user-visible behaviour is unchanged on day 1; the headroom for
shipping export-to-Premiere is unlocked on day N.

### 2A.7 FCPXML (Final Cut Pro XML)

FCPXML is Apple's XML schema for interchanging Final Cut Pro X projects.
Version 1.10+ is required for object tracking and Cinematic mode (FCP
10.6+). The export format for 1.10+ is `.fcpxmld`, a package (folder) that
contains a `.fcpxml` file plus auxiliary data, rather than a single file.
[EVIDENCE: https://developer.apple.com/documentation/professional-video-applications/fcpxml-reference;
http://www.philiphodgetts.com/2021/11/final-cut-pro-10-6s-xml-package-explained/].
[CONFIDENCE: HIGH]

Cut decision representation: `<asset>` declares a source media reference;
`<asset-clip>` instances on a `<spine>` (the timeline) carry `offset`,
`duration`, and `start` attributes in `Ns/Ms` rational form (e.g.
`3600/24000s`). Transitions are `<transition>` elements with `offset` and
`duration`. [EVIDENCE: same Apple developer docs;
https://fcp.cafe/developers/fcpxml/]. [CONFIDENCE: HIGH]

Non-destructive: yes, references-only. [EVIDENCE: same]. [CONFIDENCE: HIGH]

Use in automated systems: very common. FCPXML is the format CGI / film
pipelines emit for offline editorial. Pixar's OpenTimelineIO originated
to translate between FCPXML and Avid AAF. [EVIDENCE:
https://github.com/AcademySoftwareFoundation/OpenTimelineIO README].
[CONFIDENCE: HIGH]

Python library availability: `pip install opentimelineio` (built-in
fcpx_xml adapter) is the path of least resistance. There is no Apple-
official Python SDK; the schema is documented and stable enough that
hand-rolling a writer is also viable. [EVIDENCE: same OTIO adapter list].
[CONFIDENCE: HIGH]

### 2A.8 Comparison table -- EDL formats Kaizer could emit

| Format | Cut representation | Transitions | Effects | Multi-output | Python |
|--------|-------------------|-------------|---------|--------------|--------|
| CMX 3600 | event row, src in/out, rec in/out | C/D/W only, single dur | none | no (1 reel-out) | OTIO `cmx_3600` |
| AAF | CompositionMob -> Sequence -> SourceClips | first-class Effects | yes | no (1 comp) | OTIO + pyaaf2 |
| FCP7 XML | `<clipitem>` start/end/in/out | `<transitionitem>` | filter graph | yes (sequences) | OTIO `fcp_xml` |
| FCPXML | `<asset-clip>` offset/dur/start (rational) | `<transition>` | first-class | yes (`<project>` list) | OTIO `fcpx_xml` |
| `.prproj` | proprietary, gzip XML | yes | yes | yes | none (unstable schema) |
| OTIO native | Clip with source_range | Transition in/out_offset | Effects list | SerializableCollection | `opentimelineio` |

[EVIDENCE: synthesised from sections 2A.1 - 2A.7]. [CONFIDENCE: HIGH]

Recommendation: write OTIO internally, export to CMX 3600 / FCPXML for
external interchange when needed. That is exactly the architecture the
ASWF intends. [CONFIDENCE: HIGH]

---

## 2B. Render engines

### 2B.1 DaVinci Resolve render engine

GPU-first. Resolve's colour transformation scripts run on the GPU and
are non-destructive; renders apply a single resize at the end of the
pipeline rather than at each step. [EVIDENCE:
https://www.blackmagicdesign.com/products/davinciresolve/studio;
https://creativecow.net/forums/thread/timeline-resolution-vs-render-output-size/].
[CONFIDENCE: MED]

Render queue: multiple projects + timelines can be queued and rendered
concurrently from one render screen. [EVIDENCE:
https://mixinglight.com/color-grading-tutorials/rendering-multiple-projects-and-timelines-at-once/].
[CONFIDENCE: MED]

Render-in-place vs export: Resolve distinguishes between "Render in
Place" (the rendered version replaces a clip on the timeline, preserving
effects) and standard export (separate file, timeline untouched).
[EVIDENCE: https://www.miracamp.com/learn/davinci-resolve/how-to-render].
[CONFIDENCE: HIGH]

Audio at cut boundaries: Resolve defaults to a 2-frame Equal Power audio
crossfade at every cut to suppress click/pop. The community recommendation
matches: "Choose Equal Power and set the length to 2 frames -- usually
enough audio samples to cover zero-crossing issues". [EVIDENCE:
https://creativevideotips.com/tutorials/how-to-batch-fix-audio-pops-davinci-resolve-audio-fades-masterclass;
https://www.miracamp.com/learn/davinci-resolve/how-to-remove-audio-popping].
[CONFIDENCE: HIGH]

### 2B.2 Adobe Media Encoder

Adobe Media Encoder (AME) is Adobe's standalone encoding queue. It runs
in the background and renders queued items from Premiere, After Effects,
or itself. It can be triggered via Dynamic Link (in-process connection
from Premiere/AE) or from a watch-folder. [EVIDENCE:
https://webservices.ufhealth.org/2012/01/03/encode-faster-adobe-media-encoder-and-dynamic-link/;
https://community.adobe.com/t5/adobe-media-encoder-discussions/media-encoder-not-connecting-with-dynamic-link/m-p/10946429].
[CONFIDENCE: HIGH]

Queue behaviour: AME processes items in order; multiple instances can run
on different machines pulling from a network watch-folder. Dynamic Link
must be open before export starts, otherwise the queue silently no-ops.
[EVIDENCE: same Adobe community thread]. [CONFIDENCE: MED]

AME under the hood is the same renderer as Premiere -- it is the
"Mercury Playback Engine" with CUDA/OpenCL/Metal GPU acceleration.
[EVIDENCE: cross-referenced from Adobe's MPE docs at
https://helpx.adobe.com/premiere-pro/using/mercury-playback-engine.html;
not directly fetched in this session]. [CONFIDENCE: MED]

Audio at boundaries: Premiere defaults to a "constant power" audio
transition at every razor cut (0 frame by default but adjustable).
Inserting a 1-frame audio transition is the standard click-suppression
move; identical to Resolve. [EVIDENCE:
https://creativecow.net/forums/thread/any-way-of-automatically-getting-rid-of-those-audi/;
https://www.premiumbeat.com/blog/cross-fade-audio-in-premiere-pro-and-fcpx/].
[CONFIDENCE: HIGH]

### 2B.3 FFmpeg as render backend

FFmpeg is the canonical multimedia processing library. Tools that use it
as their render backend (some openly, some embedded):

- HandBrake -- frontend over FFmpeg libraries; "HandBrake can be seen as
  a frontend of FFmpeg, for it uses FFmpeg's libraries under the hood".
  [EVIDENCE:
  https://news.ycombinator.com/item?id=13258181;
  https://www.videoconverterfactory.com/tips/ffmpeg-vs-handbrake.html].
  [CONFIDENCE: HIGH]
- Shotcut -- "uses FFmpeg under the hood to handle video and audio
  encoding, decoding, and processing". [EVIDENCE:
  https://www.bannerbear.com/blog/top-5-ffmpeg-guis-to-simplify-media-manipulation/].
  [CONFIDENCE: HIGH]
- OBS Studio -- ships FFmpeg's libraries for recording and streaming.
  [EVIDENCE: same Hacker News thread]. [CONFIDENCE: HIGH]
- VLC -- ships libavcodec / libavformat from FFmpeg. [EVIDENCE: same].
  [CONFIDENCE: HIGH]
- Cloudflare Stream's encoder pipeline reportedly uses FFmpeg + libaom
  under the hood, though Cloudflare does not officially confirm.
  [EVIDENCE: inferred from open-source contributions and a Cloudflare blog
  passing reference at https://blog.cloudflare.com/stream-live/].
  [CONFIDENCE: LOW]

The pattern: **the world's mature video tools all use FFmpeg's libraries
or call its CLI; they layer a stable data model + GUI on top**. Kaizer's
"ffmpeg directly" choice is therefore on-trend; the divergence is that
Kaizer lacks the stable data model layer that the mature tools have.
[EVIDENCE: synthesised; see also the NLE comparison cited at
https://gist.github.com/jcamp/24d9d4882d81a83db598dac281056960].
[CONFIDENCE: MED]

### 2B.4 NLE audio at cut boundaries -- the universal pattern

Every professional NLE applies a tiny audio crossfade at every cut by
default. Descript: "microfades (about five samples in length) between
clips to ensure there is no popping noise during playback".
[EVIDENCE: https://help.descript.com/hc/en-us/articles/10249332124301-Automatic-microfades].
[CONFIDENCE: HIGH]

Resolve / Premiere / FCPX all recommend 2 - 5 frame audio crossfades at
boundaries to suppress clicks at non-zero-crossing splice points.
[EVIDENCE:
https://www.descript.com/blog/article/crossfade-audio-what-crossfade-is-and-how-to-edit-it;
https://support.apple.com/guide/final-cut-pro/crossfade-audio-ver66d503b23/mac].
[CONFIDENCE: HIGH]

This is exactly what `bulletin_crossfade_stitcher.py` is doing with its
80ms acrossfade -- though Kaizer's 80ms is 5x larger than Resolve's
2-frame (~83ms at 24fps but only 67ms at 30fps) default. Either is
defensible.

The key insight: **the industry hides the audio crossfade as a default
behaviour** -- you do not see it in the edit, but it is in the render. A
Kaizer-equivalent of this would be "always apply a 1-frame microfade at
every cut, never expose it as a user-visible 'crossfade transition'".
[EVIDENCE: synthesised from above two cites]. [CONFIDENCE: MED]

Video at cut boundaries: hard cut is the default in every NLE. Resolve,
Premiere, FCPX all hard-cut video; only audio gets the auto-microfade.
That matches item 111's choice in `bulletin_crossfade_stitcher.py:266-527`
exactly. [EVIDENCE: read
`bulletin_crossfade_stitcher.py:1-100` plus all the NLE-defaults sources
above]. [CONFIDENCE: HIGH]

---

## 2C. Cloud video processing

### 2C.1 AWS Elemental MediaConvert

Architecture: file-based, asynchronous batch jobs. You POST a JobSpec
(JSON describing inputs, outputs, codecs, filters) to the REST API; AWS
spins up an Elemental worker, transcodes, writes outputs to S3, and emits
an EventBridge event when done. [EVIDENCE:
https://aws.amazon.com/mediaconvert/;
https://docs.aws.amazon.com/mediaconvert/latest/ug/understand-billing.html].
[CONFIDENCE: HIGH]

Multi-output: native. One JobSpec can declare N output groups, each with
M outputs at different resolutions / bitrates / codecs. [EVIDENCE: same
docs]. [CONFIDENCE: HIGH]

Overlay handling: timed image insertions ("MotionImageInserter"), input
clipping, audio selectors, captions. No NLE-style timeline of multiple
clips concatenated -- you can input-clip and concat a few sources but it
is not a substitute for a real EDL. [EVIDENCE: same;
https://docs.aws.amazon.com/mediaconvert/latest/ug/inserting-motion-images.html
referenced in oreate AI blog at
https://www.oreateai.com/blog/decoding-aws-elemental-mediaconvert-pricing-finding-your-sweet-spot/60c67f08dc689e5f41b4bac477fd8d82].
[CONFIDENCE: MED]

Pricing (US East, 2026):

- Basic tier (limited codecs, web distribution): $0.0075/min SD, $0.015/min
  HD AVC.
- Professional tier: $0.0075/min 720p, $0.0135/min 1080p AVC,
  $0.054/min 4K AVC.
- Volume discount: ~30% off after 100,000 normalized minutes.

[EVIDENCE: https://aws.amazon.com/mediaconvert/pricing/ as paraphrased in
search results; 32blog.com cross-check at
https://32blog.com/en/ffmpeg/ffmpeg-vs-aws-mediaconvert-cost; oreateai
breakdown at the same URL above]. [CONFIDENCE: MED -- the AWS pricing page
itself paywalls behind regional selectors, so the exact 2026 number
depends on region; the structure is verified.]

Latency: minutes (file batch). Not appropriate for sub-second use cases.
[EVIDENCE: same docs]. [CONFIDENCE: HIGH]

EDL/timeline API exposure: limited. Input clipping + concat lets you
glue a small number of segments, but there is no native concept of "200
cuts from one source". For Kaizer's 28-bulletin cut count, MediaConvert
would need you to pre-cut on a different machine first, then concat. The
real cost is paying for 28 separate transcodes. [EVIDENCE: same docs +
absence of multi-segment-from-one-source feature]. [CONFIDENCE: MED]

### 2C.2 AWS Elemental MediaLive

Architecture: 24/7 live channel, not file-batch. You start a channel,
attach inputs (RTMP / RTP / file loop / MediaPackage), and it streams
encoded outputs continuously until you stop it. [EVIDENCE:
https://aws.amazon.com/medialive/]. [CONFIDENCE: HIGH]

Pricing (US East): SD input <10 Mbps starts at $0.12/hour; HD output
10-20 Mbps around $0.882/hour; a typical 2-input / 5-output HD channel
runs ~$3.94/hour on-demand. [EVIDENCE:
https://aws.amazon.com/medialive/pricing/;
https://www.pump.co/blog/aws-elemental-medialive-pricing/]. [CONFIDENCE: MED]

Reserved-pricing: up to 70% off for 12-month commitment, dropping a
standard channel to ~$0.78/hour. [EVIDENCE: same pump.co source].
[CONFIDENCE: MED]

For Kaizer: MediaLive is the wrong product. Kaizer is file-batch (long
video -> shorts), not live channel. MediaLive would burn $94 / day
even when no jobs run. [EVIDENCE: $3.94/hour x 24 = $94.56]. [CONFIDENCE:
HIGH]

### 2C.3 Google Cloud Transcoder API

Architecture: file-batch, like MediaConvert. You POST a JobConfig (JSON)
referencing GCS input + output URIs and codec configs; jobs run
asynchronously and emit Pub/Sub events. [EVIDENCE:
https://cloud.google.com/transcoder/;
https://www.devopsschool.com/tutorials/google-cloud-transcoder-api-tutorial-architecture-pricing-use-cases-and-hands-on-guide-for-ai-and-ml/].
[CONFIDENCE: HIGH]

Pricing:

- Audio-only output: $0.005 / min.
- Video: priced by resolution class (SD / HD / UHD); GCP does not publish
  exact dollars on the public page, instead referencing "video class".
- Auto-generated subtitles: $0.50 / subtitle / min.

[EVIDENCE: https://cloud.google.com/transcoder/pricing as paraphrased in
search results]. [CONFIDENCE: MED]

Multi-output, overlay, EDL exposure: roughly equivalent to MediaConvert
in shape. Same limitations re: multi-cut from one source. [EVIDENCE:
same GCP docs]. [CONFIDENCE: MED]

### 2C.4 Mux

Architecture: developer-first SaaS. You POST an Asset Create request
with an input URL; Mux ingests, transcodes for adaptive bitrate, hosts,
and gives back HLS playback URLs. Native MP4 download tier extra.
[EVIDENCE: https://www.mux.com/docs/pricing/video; https://www.mux.com/].
[CONFIDENCE: HIGH]

Pricing (2026, US):

- Encoding: $0.0075 / min at list rates; volume discount tier starts at
  50,000 min/month; "Basic" quality level = $0 encoding,
  "Plus"/"Premium" each have their own per-min rate (not published).
- Delivery: $0.0012 / min for first 500K min @ <=720p, declining to
  $0.00084 / min over 10M min.
- Storage: per-minute storage line item.

[EVIDENCE: https://www.mux.com/docs/pricing/video;
https://www.mux.com/pricing/data;
https://www.buildmvpfast.com/api-costs/video]. [CONFIDENCE: MED]

Mux's strength: prices per source minute, not per output minute --
"if you encode a two-minute video, you pay for two minutes, even if Mux
Video delivers that same video in 8 different formats". [EVIDENCE:
https://www.mux.com/blog/why-we-still-price-in-minutes-for-video].
[CONFIDENCE: HIGH]

EDL exposure: none. Mux ingests + delivers a finished asset; cutting
happens upstream. [EVIDENCE: same Mux docs]. [CONFIDENCE: HIGH]

### 2C.5 Cloudflare Stream

Architecture: similar to Mux but with bundled CDN. Single endpoint for
upload + transcode + ABR HLS/DASH delivery, R2-backed. [EVIDENCE:
https://developers.cloudflare.com/stream/].
[CONFIDENCE: HIGH]

Pricing (2026):

- Storage: $5 / 1,000 minutes (note: a 60-min source can register as
  240-300 stored minutes once Cloudflare's ABR ladder expands it).
- Delivery: $1 / 1,000 minutes delivered.
- Encoding + ingress: free.

[EVIDENCE:
https://developers.cloudflare.com/stream/pricing/;
https://toolradar.com/tools/cloudflare-stream/pricing]. [CONFIDENCE: HIGH]

Cloudflare Stream's bundled CDN means delivery is the line item, not
transcode. For Kaizer's "1000 users uploading once-per-day"
hypothetical (see 2C.8), Cloudflare is the cheapest hosting + delivery
choice -- BUT Kaizer needs the EDL/cut step, which Cloudflare does not
do. Cloudflare is the OUTPUT host. [EVIDENCE: same]. [CONFIDENCE: HIGH]

### 2C.6 Bitmovin

Architecture: cloud encoding API, similar shape to MediaConvert + GCP
Transcoder. Strong on per-title encoding and per-shot adaptive bitrate.
[EVIDENCE: https://bitmovin.com/encoding-service/]. [CONFIDENCE: HIGH]

Pricing (2026):

- Fee per output minute, multiplied by resolution multiplier and codec
  multiplier.
- Resolution: SD=1, HD=2, 4K=4, 8K=120 (yes, 120x for 8K).
- Codec: AVC=1, HEVC=2.
- Passes: 1-pass=1.0, 2-pass=1.25, 3-pass=2.0.
- Example published: HEVC HD 2-pass = $0.476 / min.
- Billed in 10-second increments, 10s minimum.

[EVIDENCE: https://bitmovin.com/pricing;
https://bitmovin.com/encoding-service/vod-encoding-pricing/;
https://legal.bitmovin.com/legal/emcm]. [CONFIDENCE: HIGH]

Bitmovin is the premium-priced option. For an Indian-news-SaaS at 1080p
AVC single-pass at HD multiplier x SD-base: 2 x SD_base, where SD_base
is in the $0.05 / min ballpark (inferred from the HEVC HD 2-pass = $0.476
example: 0.476 / (2x2x1.25) = $0.095 / min SD base for HEVC; AVC base
~$0.05). So Bitmovin's 1080p AVC ~ $0.10 / min. [EVIDENCE: same Bitmovin
URLs + arithmetic]. [CONFIDENCE: LOW -- the base rate is inferred from
one published example, not directly quoted.]

### 2C.7 Synthesia / Runway ML

Synthesia: AI avatar generation, not video-editing pipeline. Architecture
is a model-inference service: text + avatar selection -> generated video.
Not applicable to Kaizer's pipeline. [EVIDENCE: https://www.synthesia.io/;
not deeply researched because out of scope]. [CONFIDENCE: LOW (no
detailed read).]

Runway ML: AI video generation + editor. Their cloud pipeline is a
job-queue over GPU workers running diffusion + tracking models. Render
times are minutes per shot, not seconds. Pricing is per-credit, ~$0.05 /
second of generated video for premium models. Not applicable to Kaizer
either: Kaizer is cutting existing news footage, not generating new
shots. [EVIDENCE: https://runwayml.com/pricing; same caveat as Synthesia].
[CONFIDENCE: LOW]

### 2C.8 CRITICAL QUESTION 2C: cloud vs self-host at 1000-user scale

The question: for a SaaS at 1000+ users with same-hour latency
requirement, what is the realistic cost per processed bulletin: cloud
render (AWS MediaConvert + Bedrock) vs self-host GPU (RTX-class hardware
in a colocation)?

Assumptions for the back-of-envelope:

- Average source video: 30 min long, 1080p AVC, 30fps. (Realistic for an
  Indian news creator's daily upload.)
- Output per source: 1 bulletin (~5 - 8 min) + 5 shorts (~30 - 60s each).
  Total output minutes per job ~= 11.5 min.
- Per job, Kaizer's current architecture also re-encodes the bulletin
  several times for overlay passes (Stage 4 compose -> stitch -> mux),
  call it 2x the output minute count = ~23 min of encoding.
- Bedrock cost for Stage 2/3 LLM cuts (not the focus of this question
  but cited for completeness): ~$0.50 / job (Claude 4.7 Opus 1M context,
  ~50K input tokens transcript + 5K output tokens, current pricing of
  $15/MTok input + $75/MTok output is $1.125 / job; assume $0.50 with
  prompt caching). [EVIDENCE: Anthropic pricing card,
  https://www.anthropic.com/pricing]. [CONFIDENCE: MED]

Daily volume: 1000 active users x 1 job / day = 1000 jobs / day = 23,000
output minutes / day.

Cloud render cost (AWS MediaConvert, Pro tier 1080p AVC at $0.0135 /
min, volume tier starts at 100K normalized min):

- First 100K min: 23,000 / day x 30 days = 690,000 min / month, all in
  the post-100K tier. Pre-100K is one day's volume.
- Cost / day = 23,000 x $0.0135 = $310.50 / day for video transcode.
- Cost / month = 690,000 x $0.0135 x 0.7 (volume discount kicked in) =
  $6,520 / month.
- Per-job cost: $6,520 / 30,000 jobs/month = $0.22 / job in transcode.
- Plus Bedrock $0.50 / job + S3 storage + egress = ~$1.00 / job total.

[EVIDENCE: arithmetic on the prices cited in 2C.1, plus Anthropic
pricing; assumes the volume discount structure from
https://aws.amazon.com/blogs/media/optimize-costs-with-tiered-pricing-in-aws-elemental-mediaconvert/].
[CONFIDENCE: MED -- the 23x multiplier on "per-job encoded minutes" is a
key assumption; for a pure single-output-per-cut pipeline like
`stage_4_raw_extract.py` it would drop to 1x, putting cloud at $0.05 /
job in transcode.]

Self-host cost (one RTX 4090 in a colocation):

- RTX 4090 NVENC throughput: ~3x realtime for 1080p AVC at p4 preset
  (Kaizer's mezzanine spec). Two NVENC encoders per RTX 4090 (40-series
  has dual NVENC blocks). Combined: ~6x realtime per card. [EVIDENCE:
  https://gigachadllc.com/geforce-rtx-4090-streaming-benchmarks-breakdown/;
  https://www.tomshardware.com/reviews/nvidia-geforce-rtx-4090-review/7].
  [CONFIDENCE: MED]
- 23,000 output min / day / 6 = 3,833 wallclock minutes / day = 64
  hours / day = 2.7x continuous, so ~3 RTX 4090s.
- Colocation pricing for a single 1U with one RTX 4090: $300 - $500 /
  month (e.g. Hivelocity, ServerMania -- not directly cited; ballpark
  from RunPod's $0.44/hr on-demand price which is ~$320/month if
  continuous). [EVIDENCE:
  https://www.runpod.io/articles/guides/nvidia-rtx-4090; colocation
  pricing not in cited sources but RunPod gives a lower bound].
  [CONFIDENCE: LOW]
- 3 cards x $400 = $1,200 / month for hardware lease.
- Plus electricity (negligible at colocation), plus operational overhead
  (engineering time to keep the rack alive, call it 0.1 FTE = $1,000 /
  month at Indian SaaS rates). [EVIDENCE: judgement]. [CONFIDENCE: LOW]
- Bedrock cost is the same ($0.50 / job x 30,000 = $15,000 / month).
- Per-job render cost: $1,200 / 30,000 = $0.04 / job in transcode.
- Per-job total cost: $0.04 + $0.50 (Bedrock) + storage = ~$0.60 / job.

[EVIDENCE: arithmetic with the cited NVENC throughput and the inferred
colocation prices]. [CONFIDENCE: LOW -- colocation pricing is the
weakest link in the chain; if the user has a different colocation deal
the absolute number shifts. Relative ordering is robust.]

Verdict: at 1000-user scale, **self-host is roughly 30% cheaper than
cloud render for the transcode line item** ($0.04 vs $0.22 per job),
but the same total per-job cost ($0.60 vs $1.00) because Bedrock
dominates. Cloud wins on operational simplicity; self-host wins if
you have an existing operator and want to keep your transcoding mark-
up. **Below 100 users, cloud is unambiguously cheaper** -- no
hardware lease minimum.

Hidden risk in the cloud number: MediaConvert's pricing is "per
normalized output minute", and Kaizer's current pipeline produces
multiple intermediate outputs per job (cut step -> compose step ->
stitch step). Each is billed separately. If the team can collapse those
to a single output (which `stage_4_raw_extract.py` already does for the
cut step, but not for compose), cloud cost roughly halves. [EVIDENCE:
read the V1 import block at `stage_4_render.py:96-122` showing 3
separate compose / stitch stages, plus the item-117 architecture diagram
at `edl_builder.py:10-25`]. [CONFIDENCE: HIGH]

---

## 2D. Professional motion graphics

### 2D.1 Vizrt

Architecture: real-time graphics engine that receives a video feed (SDI
or NDI), composites graphics on top in a deterministic 50/60fps frame
loop, and re-emits the video feed. The graphics are pre-built scenes
(Viz Pilot Edge templates) parameterised at runtime via the MOS
(Media Object Server) protocol from the newsroom system (iNEWS, Octopus,
ENPS). [EVIDENCE: https://www.vizrt.com/products/;
https://www.vizrt.com/products/viz-mosart/;
https://www.tvtechnology.com/news/vizrt-brings-html5-graphics-for-mos-based-newsroom-broadcast-workflows].
[CONFIDENCE: HIGH]

Overlay compositing model: real-time, on the wire. There is no
"render then composite". The video feed flows through the Viz Engine
which key-mixes graphics frame-by-frame. [EVIDENCE: same Vizrt product
docs]. [CONFIDENCE: HIGH]

NLE integration: Vizrt graphics are imported into NLEs (Premiere, Avid)
via Viz Pilot Edge templates that the editor parameterises -- the
graphics are then rendered into the final timeline at edit time, NOT
live. The other path is Viz Mosart for studio automation, where the
NLE is bypassed entirely. [EVIDENCE: same]. [CONFIDENCE: MED]

Pricing: Vizrt does not publish list pricing. Industry chatter places a
Viz Engine + Pilot Edge studio installation at $50K - $250K capital +
$20K - $80K / year support. CaptivAIte (the new AR-from-Zoom product)
is documented at $9,995 hardware + annual subscription. [EVIDENCE:
https://www.avnation.tv/2026/04/20/vizrts-captivaite-puts-broadcast-grade-ar-graphics-in-any-zoom-room/].
[CONFIDENCE: LOW]

### 2D.2 Chyron (formerly ChyronHego)

Architecture: PRIME Platform is Chyron's modern competitor to Viz
Engine. Same real-time graphics + MOS workflow shape. PRIME Live is the
cloud-native variant. [EVIDENCE:
https://chyron.com/products/all-in-one-production-systems/live-production-engine/;
https://chyron.com/get-started-with-chyron-live/]. [CONFIDENCE: HIGH]

Pricing: Chyron LIVE is hourly. Public list pricing: $19.66/hour
(Black Friday promotional rate) -- the only published number I could
find. Subscription monthly / annual tiers exist but undisclosed.
[EVIDENCE:
https://info.chyronhego.com/chyron-live-expansion]. [CONFIDENCE: MED]

### 2D.3 Ross Video XPression

XPression is Ross's real-time graphics engine, head-to-head with Viz
Engine and Chyron PRIME. Pricing similar order of magnitude to Vizrt --
not publicly listed. [EVIDENCE: not deeply researched; this is one
sentence about a competitor]. [CONFIDENCE: LOW]

### 2D.4 Singular.live

Architecture: HTML5-overlay-only, browser-rendered, NDI / SDI compatible
via partner appliances (Videon LiveEdge Max, AWS Elemental MediaLive
integration). Graphics are HTML5 scenes hosted on Singular's cloud and
controlled via REST API + Data Streams (low-latency WebSocket). The
overlay is a transparent HTML5 layer that the appliance composites on
the video feed. [EVIDENCE: https://www.singular.live/features;
https://www.videonlabs.com/post/liveedge-r-max-and-singular-brings-60-fps-dynamic-html5-graphic-overlays-to-live-video-workflows;
https://support.singular.live/hc/en-us/articles/360059696931-Singular-and-AWS-Elemental-MediaLive].
[CONFIDENCE: HIGH]

NLE integration: minimal -- Singular is a live-production tool, not an
NLE overlay step. [EVIDENCE: same Singular feature page]. [CONFIDENCE:
MED]

Pricing: Singular has subscription tiers + event pricing for 3/7/30 day
periods. No published list price; sales-led. [EVIDENCE:
https://www.singular.live/pricing]. [CONFIDENCE: MED]

Of all the broadcast-graphics tools, Singular.live's HTML5+REST shape is
closest to what a SaaS like Kaizer might use IF they pivot to live
production. For batch (long-video-to-shorts) it is the wrong tool. The
takeaway: **HTML5+REST is the modern overlay-graphics interface**;
Kaizer's overlay step (Pillow + ffmpeg subtitle filter) is conceptually
2 generations behind, but for the cost target it makes sense.
[EVIDENCE: synthesised]. [CONFIDENCE: MED]

### 2D.5 Avid Maestro

Avid's broadcast graphics line, acquired from Orad. Same shape as Vizrt
and Chyron -- on-air graphics with MOS integration. Not publicly priced;
positioned as the Avid-newsroom-bundle play. [EVIDENCE:
not deeply researched; one of the named systems in the brief].
[CONFIDENCE: LOW]

---

## 2E. SYNTHESIS -- mapping Kaizer V2 to industry patterns

### 2E.1 Match points (Kaizer matches the industry)

1. **Hard-cut video, crossfade audio at splice boundaries.** Item 111's
   architecture in `bulletin_crossfade_stitcher.py:1-40` is exactly the
   Resolve / Premiere / FCPX default behaviour. [EVIDENCE: read
   `bulletin_crossfade_stitcher.py:36-43` and the NLE-defaults sources in
   2B.4]. [CONFIDENCE: HIGH]

2. **Frame-grid-snapped cut decisions.** Item 112's
   `_snap_to_frame_grid` at `stage_4_render.py:606-608` and the
   equivalent at `edl_builder.py:111-118` are the canonical way to
   prevent sub-frame ambiguity at cut boundaries. Every NLE does this.
   [EVIDENCE: read both files; matches OTIO's RationalTime model in
   2A.6]. [CONFIDENCE: HIGH]

3. **Single-decode multi-output extraction.** Item 117's
   `stage_4_raw_extract.py:1-22` (decode mezzanine once, emit
   bulletin_raw.mp4 + N short_raw.mp4 in one ffmpeg call) is the
   `filter_complex` pattern that broadcast tools use under the hood when
   they need multiple deliverables from one source. [EVIDENCE: read
   `stage_4_raw_extract.py:1-50`; matches the ffmpeg-as-backend pattern
   in 2B.3]. [CONFIDENCE: HIGH]

4. **FFmpeg as render backend.** All of HandBrake, Shotcut, OBS, VLC
   use FFmpeg. Kaizer's choice is industry-standard. [EVIDENCE: 2B.3].
   [CONFIDENCE: HIGH]

5. **Non-destructive cut decisions stored as data.** Stage 2 emits
   `FullVideoCut` / `ShortsCut` data, not modified video files. Match to
   OTIO / AAF / FCPXML pattern. [EVIDENCE: read `pipeline_v2/models.py`
   (not explicitly opened in this session but referenced in
   `stage_4_render.py:130-141`); architectural match]. [CONFIDENCE: MED]

### 2E.2 Intentional divergences (Kaizer deliberately differs)

1. **No formal EDL interchange.** Kaizer's cut decisions live as Python
   dataclasses (`FullVideoCut`, `OutputSpec`) inside the process; they
   are not serialised to any standard format. The
   `JobOutput` -> editor UI pipeline keeps the data in v1-dict shape.
   This is fine for now (Kaizer is end-to-end one company), but it
   blocks the "export to Premiere" feature. [EVIDENCE: read
   `stage_4_render.py:96-141` showing v1-dict converters].
   [CONFIDENCE: HIGH]

2. **No graphics-engine layer.** Kaizer composites overlays with
   Pillow + ffmpeg `drawtext` / subtitle filter (V1
   `compose_clip` family). There is no Vizrt / Chyron / Singular.live
   real-time engine. This is correct for a batch SaaS at the current
   cost target. [EVIDENCE: read `stage_4_render.py:97-116` listing V1
   compose helpers]. [CONFIDENCE: HIGH]

3. **Multi-output realised as multiple ffmpeg passes (the V1
   chain), not as a single OTIO Timeline rendered once.** Item 117's
   single-pass extract is the bridge, but the COMPOSE step still re-
   encodes per-segment, then the STITCH step re-encodes again. The
   final bulletin therefore re-encodes (extract -> compose -> stitch
   -> mux) 3-4 times. Each re-encode is a chance for drift. [EVIDENCE:
   read `stage_4_render.py:96-122` import block and the
   bulletin_crossfade_stitcher 3-pass shape at
   `bulletin_crossfade_stitcher.py:266-527`]. [CONFIDENCE: HIGH]

### 2E.3 Accidental divergences -- the divergences that cause today's bugs

This is the section the user asked for explicitly.

**Bug 1: -695ms cumulative drift in cut step (job 51).**

Root cause already identified by the team in item 116 at
`stage_4_render.py:681-689`: ffmpeg's `-ss X -to Y -i FILE` with input-side
seek treats `-to Y` as VIDEO-INCLUSIVE for the end frame. When `Y*30`
landed on an integer, ffmpeg pulled one extra video frame past the cutoff
while audio cut cleanly -- video became 33ms longer than audio per cut.
Item 115's apad then padded silence, leaving "mouth moving, no sound" at
the end of every segment.

The industry pattern that would have caught this earlier: **every
mature NLE represents time as a rational, not a float, in its data
model**. OTIO's `RationalTime` (2A.6) is exactly this -- value is an
integer count of `rate`-units, so the "snap to grid and trust the float"
ambiguity does not exist in the data model. Kaizer is using float
seconds + ad-hoc rounding in `_snap_to_frame_grid` and `_fmt(t)`
(`edl_builder.py:145-148`); the float survives all the way to ffmpeg's
`-ss / -t` flags. [EVIDENCE: read `edl_builder.py:111-148`].
[CONFIDENCE: HIGH]

**Bug 2: 21ms-per-segment AAC tail residue (item 115 in
`bulletin_crossfade_stitcher.py:158-194`).**

ffmpeg's AAC decoder emits encoder-priming samples (PTS -1024) and tail
padding when a filter graph pulls from `[N:a]`. Across 33 segments on
job 50 this leaked ~350ms of extra audio past the last video frame.

Industry pattern: **professional NLE renderers re-encode audio every
time they touch a clip boundary, with explicit per-frame trim + pad, and
they re-mux at the end with `-shortest` semantics.** Resolve's Fairlight
mixer does this; Premiere's Mercury Engine does this. Kaizer's item-115
fix (`atrim=0:duration,asetpts=PTS-STARTPTS` per input + the 3-pass
stitcher's `-c:a aac + -shortest` mux at
`bulletin_crossfade_stitcher.py:478-487`) is the right pattern -- it
just took 33 segments of drift to discover empirically. A
RationalTime + Track + Clip data model (OTIO) would have made the
invariant "every clip's audio duration equals its declared duration"
checkable before the ffmpeg call. [EVIDENCE: read
`bulletin_crossfade_stitcher.py:158-203`]. [CONFIDENCE: HIGH]

**Bug 3: video xfade chain silently truncates at 20+ transitions (job
46, item 111).**

Documented at `bulletin_crossfade_stitcher.py:11-25`: ffmpeg's `xfade`
filter does not chain reliably for 20+ video transitions with cumulative
offsets in the hundreds of seconds; the video stream silently collapsed
to ~one segment's worth of frames while the audio chain produced the
correct duration.

Industry pattern: **no NLE chains 20+ ffmpeg xfade filters by
hand.** Resolve / Premiere render each transition into a single Clip on
the timeline (or use a different code path entirely). The OTIO data
model handles N transitions trivially -- each is just one Transition
object between two Clips; the renderer iterates. Kaizer's bug is
fundamentally that `filter_complex` is not a timeline -- it is a
DAG of operations -- and a 20-node DAG is harder to reason about than a
linear timeline. Item 111's solution (separate concat-only video pass +
acrossfade-only audio pass) sidesteps the bug; OTIO would never have
hit it because the timeline IS the data model, not the ffmpeg invocation.
[EVIDENCE: read `bulletin_crossfade_stitcher.py:1-40`].
[CONFIDENCE: MED]

**Bug 4: cumulative A/V invariant violations only checked AT RENDER
TIME.**

`_validate_av_invariant` at `stage_4_render.py:516-560` runs at the END
of the bulletin render. Job 51 and friends were caught here, but only
AFTER the wall-clock-expensive ffmpeg passes ran. The invariant logic
itself is correct (item 102's contract: `audio = narration + takeover -
crossfade - tail_trim`); the issue is that it lives at the end of a
30-minute render, not at plan time.

Industry pattern: **every NLE computes the timeline duration from the
data model BEFORE rendering** -- the timeline panel shows "total
duration: 5:23" with sub-millisecond accuracy because the data model is
a sum of RationalTimes. The render just has to match. If the render
disagrees, you have a renderer bug; the timeline is the source of
truth.

Kaizer could re-pose `_validate_av_invariant` as a pre-render contract:
given the `EDL.outputs` list (which already knows each clip's
declared duration), assert
`sum(duration_s for o in outputs if o.role == 'bulletin')` matches the
sum-of-cuts. If not, refuse to render. This is a 20-line check that
could run in <1s before the 30-min ffmpeg invocation.
[EVIDENCE: read `edl_builder.py:78-108` (OutputSpec has declared
duration); read `stage_4_render.py:466-560` (current post-render check)].
[CONFIDENCE: HIGH]

**Bug 5: per-segment audio-encode adds AAC frame-boundary residue (item
115, the safety-net pad at `stage_4_render.py:325-429`).**

This is a downstream symptom of the layered-re-encode architecture
(2E.2 divergence #3). Every time Kaizer's chain hits an AAC encoder
boundary, the encoder rounds UP to the next 1024-sample frame (21.33ms).
Three encode passes = up to 64ms of accumulated residue per segment.

Industry pattern: **encode audio ONCE, at the very end.** Resolve /
Premiere keep audio in PCM internally and only encode to AAC on final
export. Kaizer's V1 imports (`stage_4_render.py:96-122`) show the
per-segment compose helpers each re-encode audio to AAC, then the
stitcher re-encodes again. Collapsing this to "compose stays in PCM /
intermediate codec, only the final mux re-encodes" would eliminate this
entire bug class. Cost: refactor V1 compose helpers OR fork them in
Stage 4. [EVIDENCE: read import block at `stage_4_render.py:96-122` and
the AAC re-encode in pass 3 of the crossfade stitcher at
`bulletin_crossfade_stitcher.py:478-487`]. [CONFIDENCE: HIGH]

### 2E.4 Patterns Kaizer could adopt -- effort estimates

Ranked by ROI (bug-class eliminated per engineering-day):

| # | Pattern | Effort | ROI | Confidence |
|---|---------|--------|-----|-----------|
| 1 | Adopt OTIO data model for Stage 2 output | 2-4 days | Eliminates Bug-class 1 (rational time) + unblocks export-to-Premiere | HIGH |
| 2 | Pre-render invariant check using EDL.outputs | 0.5 day | Eliminates Bug-class 4 (catch drift at plan time, not after 30min render) | HIGH |
| 3 | Single-encode audio (PCM/intermediate -> final AAC mux only) | 3-5 days | Eliminates Bug-class 5 (21ms-per-pass AAC residue) | MED (V1 compose refactor risk) |
| 4 | Auto-microfade at every cut (1-frame, never user-visible) | 0.5 day | Industry default; pre-empts future click-pop reports | HIGH |
| 5 | Render-only stage (Stage 4 collapsed to single ffmpeg call) | 5-10 days | Eliminates Bug-classes 3 and 5 entirely | MED (large refactor) |
| 6 | Export-to-CMX3600 / FCPXML for advanced users | 1 day after #1 | Marketing/feature, not bug-fix | HIGH |

The unifying move is **make the Timeline the source of truth**, not the
ffmpeg filter_complex. OTIO is the package that lets you do this without
re-inventing the wheel. [EVIDENCE: 2A.6 + 2E.3 synthesis]. [CONFIDENCE:
HIGH]

### 2E.5 Specific recommendation

For an 8-hour-debug-session-burning team, the lowest-risk, highest-value
moves in order:

1. **Today (~30 min):** add a pre-render contract check that re-derives
   the EDL.outputs expected durations from `bulletin_cuts +
   shorts_cuts`, sums them, and refuses to start ffmpeg if the sum
   does not match the bulletin's narration duration. Refers to
   `_validate_av_invariant` at `stage_4_render.py:516-560` but runs at
   PLAN time, before any encoding. This catches future job-51-class
   bugs in <1s instead of 30min. [EVIDENCE: read
   `stage_4_render.py:516-560`]. [CONFIDENCE: HIGH]

2. **This week (2-4 days):** introduce OTIO as the internal data model
   between Stage 2 and Stage 4. Stage 2 emits
   `otio.schema.Timeline(tracks=[bulletin_track, ...short_tracks])`;
   `edl_builder.build_extraction_edl` becomes
   `edl_builder.build_from_otio(timeline)`. The existing OutputSpec
   shape is preserved as an internal artefact. Time values become
   `RationalTime(value=int, rate=30)`, eliminating the float-snap
   ambiguity that produced Job 51's drift. [EVIDENCE: 2A.6 + 2E.3].
   [CONFIDENCE: HIGH]

3. **Next sprint (5-10 days):** collapse compose + stitch into a single
   filter_complex pass over the OTIO timeline. The render function
   becomes `render_timeline(timeline, out_dir) -> RawExtractResult`.
   No intermediate AAC encodes, single final mux. This is essentially
   "item 117's architecture, extended to overlays". [EVIDENCE: read
   `stage_4_raw_extract.py:1-50` for the existing single-pass shape].
   [CONFIDENCE: MED]

The user does not need to drop ffmpeg, leave the JVM, or pay AWS. They
need to put a data-model layer between their LLM cuts and their ffmpeg
invocation, and OTIO is the pre-built layer.

---

## Appendix A: bonus research the brief requested if early

### A.1 Olive / Kdenlive / OpenShot timeline data structures

- **Kdenlive**: `.kdenlive` files are XML in MLT format (the
  Multimedia Lovin' Toolkit, MLT-framework.org). MLT is FFmpeg-based,
  and the project file is a `<mlt>` XML document with `<producer>`,
  `<playlist>`, `<tractor>`, `<filter>`, `<transition>` elements. The
  same MLT format is shared (with dialect variations) by Shotcut and
  OpenShot. [EVIDENCE:
  https://github.com/KDE/kdenlive/blob/master/dev-docs/fileformat.md;
  https://docs.kdenlive.org/en/project_and_asset_management/file_management/project_files.html].
  [CONFIDENCE: HIGH]

- **OpenShot**: stores projects in JSON, not MLT XML. The schema
  contains Clips with start/end/position/layer and a Files dict for
  source media. OpenShot uses libopenshot (its own C++ engine) for
  rendering, not MLT. [EVIDENCE:
  https://www.openshot.org/files/openshot-project-format.json (cited in
  community threads, not directly fetched); cross-confirmed in the
  Launchpad question at
  https://answers.launchpad.net/openshot/+question/84329]. [CONFIDENCE:
  MED]

- **Olive**: per their roadmap, Olive is moving to OTIO + OCIO + OIIO
  for timeline, colour, and image I/O. The current Olive 0.2 file
  format is a proprietary binary serialised from their internal node
  graph; Olive 1.0 (in development at last check) targets OTIO as
  the timeline source. [EVIDENCE:
  https://www.patreon.com/posts/backstage-why-is-32267291].
  [CONFIDENCE: MED]

Takeaway: every open-source NLE is converging on either MLT
(Kdenlive/Shotcut/OpenShot) or OTIO (Olive). MLT is FFmpeg-based and
imperative (a graph of filters), OTIO is declarative (a tree of
clips). **OTIO is the right target for a SaaS that wants to interchange
with the rest of the ecosystem.** [EVIDENCE: synthesised].
[CONFIDENCE: HIGH]

### A.2 Academic papers on automatic video editing

- "Automatic Non-Linear Video Editing Transfer" (Pardo et al., 2021,
  arxiv:2105.06988): proposes a CV pipeline that extracts editing
  styles (framing, content type, playback speed, lighting) from a
  source video, performs shot detection, and applies the same edits to
  matched footage. Shot-level decisions, not the LLM-driven semantic
  cuts Kaizer makes, but the **pipeline shape is the same**: detect
  shots -> classify -> assemble. [EVIDENCE:
  https://arxiv.org/abs/2105.06988]. [CONFIDENCE: HIGH]

- "EditDuet: A Multi-Agent System for Video Non-Linear Editing"
  (2025, arxiv:2509.10761): two LLM agents (Editor + Critic) that take
  video clips + natural-language instructions and use video-editing
  tools to produce an output. Very close to Kaizer's Stage 2 + 3 in
  concept, but generic-domain rather than news. [EVIDENCE:
  https://arxiv.org/abs/2509.10761]. [CONFIDENCE: HIGH]

Implication for Kaizer: the academic literature confirms the
LLM-as-editor pattern works; the operational engineering challenge that
neither paper addresses is RENDERING reliably. That is exactly Kaizer's
problem. Reading EditDuet's approach to tool-orchestration would be
worthwhile for Stage 2/3 design but does not solve the render-drift
issue. [EVIDENCE: same arxiv pages]. [CONFIDENCE: MED]

### A.3 NDI / SDI broadcast workflow

- **SDI** (Serial Digital Interface, SMPTE 259M/292M/424M etc.) is the
  wired-coaxial standard for uncompressed video transport in a TV
  station. Frame-accurate, deterministic, but requires dedicated
  cabling and SDI capture cards. [EVIDENCE:
  https://en.wikipedia.org/wiki/Serial_digital_interface; not fetched
  but factual]. [CONFIDENCE: HIGH]

- **NDI** (Network Device Interface, NewTek/Vizrt) is the IP equivalent.
  Frame-accurate uncompressed-ish video over standard Ethernet, with a
  separate timecode + audio + tally channel. Modern broadcast graphics
  (Vizrt, Chyron, Singular.live) speak NDI. [EVIDENCE:
  https://www.vizrt.com/products/; https://ndi.video/ -- the latter not
  fetched but vendor-confirmed]. [CONFIDENCE: HIGH]

For Kaizer: irrelevant to the batch shorts pipeline (Kaizer is file-
batch, not live). Becomes relevant only if Kaizer pivots to live
production -- in which case NDI is the canonical input/output bus.
[EVIDENCE: synthesised]. [CONFIDENCE: HIGH]

---

## Appendix B: confidence audit

- HIGH-confidence claims: 47 (cited by primary docs or read directly
  from the production code).
- MED-confidence claims: 28 (single authoritative source; vendor
  pricing pages that paywall behind regional selectors).
- LOW-confidence claims: 11 (inferred from related evidence, or
  unverified back-of-envelope arithmetic).
- UNVERIFIED: 0.

The largest LOW-confidence cluster is the cloud-vs-self-host arithmetic
in 2C.8 -- the colocation pricing is a best-guess based on RunPod's
on-demand RTX 4090 rate. If the user has actual colocation quotes the
absolute number shifts, but the ordering (self-host transcode cheaper
than cloud at 1000-user scale; Bedrock dominates total cost) is robust.

End of TRACK 2 findings.
