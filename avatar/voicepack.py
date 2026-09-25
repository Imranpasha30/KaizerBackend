# Ported from kaizer-platform@d5fd482 server/avatar/voicepack.py.
# Changes from upstream: none (verbatim port; stdlib + ffmpeg only).
"""Telugu voice-pack minter — scales the studio catalog to 100 tagged voices.

Batch-mints IndicF5 clone-source voice folders (``ref.wav`` + ``ref.txt``
+ ``meta.json`` — the exact format avatar-studio's ``tools/make_voice.py``
produces) into the studio ``voices/`` dir, from the OpenSLR SLR66 corpus
zips already sitting in ``<studio>/corpus/slr66/``. Once a folder exists
the voice auto-appears in the catalog, the ``/sample`` endpoint, and
``generate.py`` — that IS the studio's registration mechanism, so no
router changes are needed per voice.

Sources & consent: SLR66 is crowdsourced Telugu speech recorded WITH the
speakers' consent and released CC-BY-SA-4.0 — no scraped anchors, no
real-person cloning. The corpus only has 24 female / 23 male distinct
speakers, so a 100-voice pack cannot be 100 distinct humans: the
remainder are pitch+tempo DSP variants (asetrate shifts formants along
with pitch, which is what makes a variant read as a different speaker,
not just the same one transposed). Variants are biased upward
(+0.9..+1.8 st, slightly faster) to fit the network's strong,
higher-pitch anchor delivery profile.

Target mix: 70 female / 30 male ``te_*`` voices (existing ``te_f_*`` /
``te_m_*`` folders count toward the target; minting is idempotent).

Run from the server dir:  .venv/Scripts/python -m avatar.voicepack
Only external dependency: ffmpeg on PATH (already required everywhere).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

TARGET = {"f": 70, "m": 30}          # tagged te_* voices per gender
SAMPLE_RATE = 24000                  # what IndicF5 refs use (make_voice.py)
_GOOD_MIN_S, _GOOD_MAX_S = 6.0, 15.0

# Digits (Latin or Indic) in a ref transcript garble IndicF5. make_voice.py
# only warns because a human reviews its input; here nobody does, so skip.
_DIGITS_RE = re.compile(r"[0-9০-৯०-९౦-౯]")

# Timbre variants, in fill order. Mostly upward — the anchor profile wants
# strong + higher-pitch — with one downward option so an all-variant
# stretch doesn't sound like a single transposed choir.
#   (id suffix, pitch semitones, tempo factor, label)
VARIANTS: tuple[tuple[str, float, float, str], ...] = (
    ("b", +0.9, 1.02, "bright"),
    ("h", +1.8, 1.04, "high"),
    ("w", -0.9, 0.99, "warm"),
)


@dataclass(frozen=True)
class MintSpec:
    """One voice folder to create."""
    voice_id: str                    # e.g. "te_f_01033" or "te_f_01033b"
    gender: str                      # "f" | "m"
    speaker: str                     # SLR66 speaker id, e.g. "01033"
    utt_id: str                      # zip member stem, e.g. "tef_01033_000..."
    transcript: str                  # EXACT text of that utterance
    pitch_st: float = 0.0            # 0.0 = base voice (no DSP)
    tempo: float = 1.0
    variant_label: str = ""          # "" for base voices


# ── corpus parsing ─────────────────────────────────────────────────

def parse_line_index(text: str) -> dict[str, list[tuple[str, str]]]:
    """SLR line_index.tsv -> {speaker: [(utt_id, transcript), ...]}.

    Lines look like ``tef_01033_00351357063\\t<telugu text>``; the middle
    token is the speaker. Malformed lines are dropped, not fatal — the
    corpus ships a handful of blank trailing lines.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    for line in text.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        utt_id, transcript = parts[0].strip(), parts[1].strip()
        bits = utt_id.split("_")
        if len(bits) < 3 or not transcript:
            continue
        out.setdefault(bits[1], []).append((utt_id, transcript))
    return out


def usable_utterances(utts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Digit-free utterances, longest transcript first.

    Transcript length is a proxy for audio length — SLR66 has no per-file
    durations in the index, and probing every wav inside a zip to pick
    one would cost more than the mint itself.
    """
    clean = [u for u in utts if not _DIGITS_RE.search(u[1])]
    return sorted(clean, key=lambda u: len(u[1]), reverse=True)


# ── pack planning (pure — this is what the tests pin down) ─────────

def plan_pack(
    gender: str,
    speakers: dict[str, list[tuple[str, str]]],
    existing_ids: set[str],
    target: int | None = None,
) -> list[MintSpec]:
    """Decide which voices to mint to close the gap to ``target``.

    Pass 1 mints a base voice per speaker that doesn't have one yet;
    pass 2 cycles VARIANTS across all speakers (a different utterance
    per variant, so refs differ in content as well as timbre) until the
    target is met. Deterministic: speakers in sorted order.
    """
    want = TARGET[gender] if target is None else target
    prefix = f"te_{gender}_"
    have = sum(1 for vid in existing_ids if vid.startswith(prefix))
    need = want - have
    if need <= 0:
        return []

    plan: list[MintSpec] = []
    ordered = sorted(speakers)
    pools = {spk: usable_utterances(speakers[spk]) for spk in ordered}

    def _add(spk: str, utt_idx: int, suffix: str, pitch: float,
             tempo: float, label: str) -> bool:
        pool = pools[spk]
        if utt_idx >= len(pool):
            return False               # speaker ran out of clean utterances
        vid = f"{prefix}{spk}{suffix}"
        if vid in existing_ids or any(s.voice_id == vid for s in plan):
            return False
        utt_id, transcript = pool[utt_idx]
        plan.append(MintSpec(
            voice_id=vid, gender=gender, speaker=spk, utt_id=utt_id,
            transcript=transcript, pitch_st=pitch, tempo=tempo,
            variant_label=label,
        ))
        return True

    # Pass 1 — one base voice per un-minted speaker.
    for spk in ordered:
        if len(plan) >= need:
            return plan
        _add(spk, 0, "", 0.0, 1.0, "")

    # Pass 2 — timbre variants, round-robin over variants then speakers.
    for vi, (suffix, pitch, tempo, label) in enumerate(VARIANTS):
        for spk in ordered:
            if len(plan) >= need:
                return plan
            _add(spk, vi + 1, suffix, pitch, tempo, label)

    return plan                        # corpus exhausted before target


# ── minting (effectful) ────────────────────────────────────────────

def _ffmpeg_bin() -> str:
    return os.environ.get("KAIZER_FFMPEG", "").strip() or shutil.which("ffmpeg") or ""


def _ffmpeg_convert(src: Path, dest: Path, pitch_st: float, tempo: float) -> None:
    """Corpus wav -> 24 kHz mono pcm_s16le ref, optionally timbre-shifted.

    asetrate scales pitch AND formants together (that's the point — a
    formant shift is what changes the perceived speaker), then atempo
    compensates duration so the variant isn't also sped up beyond its
    intended delivery pace. Combined atempo stays well inside ffmpeg's
    0.5-2.0 window for our ±1.8 st range.
    """
    ff = _ffmpeg_bin()
    if not ff:
        raise RuntimeError("ffmpeg not found — set KAIZER_FFMPEG or add to PATH")
    cmd = [ff, "-y", "-v", "error", "-i", str(src), "-ac", "1"]
    if pitch_st or tempo != 1.0:
        factor = 2.0 ** (pitch_st / 12.0)
        flt = (f"aresample={SAMPLE_RATE},"
               f"asetrate={int(SAMPLE_RATE * factor)},"
               f"aresample={SAMPLE_RATE},"
               f"atempo={tempo / factor:.6f}")
        cmd += ["-af", flt]
    else:
        cmd += ["-ar", str(SAMPLE_RATE)]
    cmd += ["-c:a", "pcm_s16le", str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120)
    if proc.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg failed for {dest.name}: {proc.stderr[:300]}")


def mint(spec: MintSpec, corpus_zip: Path, voices_root: Path) -> Path:
    """Materialize one voice folder. Idempotent: an existing ref.wav is
    the cache hit — nothing is re-synthesized or overwritten."""
    vdir = voices_root / spec.voice_id
    ref = vdir / "ref.wav"
    if ref.exists():
        return vdir
    vdir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(corpus_zip) as zf, \
            tempfile.TemporaryDirectory(prefix="voicepack_") as td:
        member = f"{spec.utt_id}.wav"
        src = Path(zf.extract(member, td))
        _ffmpeg_convert(src, ref, spec.pitch_st, spec.tempo)

    (vdir / "ref.txt").write_text(spec.transcript + "\n", encoding="utf-8")
    style = "natural spoken news read (SLR66)"
    source = f"OpenSLR SLR66 speaker {spec.speaker}"
    if spec.variant_label:
        style += f", {spec.variant_label} timbre variant"
        source += (f" ({spec.pitch_st:+.1f} st pitch, "
                   f"x{spec.tempo:.2f} tempo variant)")
    meta = {
        "gender": spec.gender.upper(),
        "style": style,
        "source": source,
        "license": "CC-BY-SA-4.0",
        "created": time.strftime("%Y-%m-%d"),
    }
    (vdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return vdir


def mint_pack(voices_root: Path, corpus_dir: Path) -> dict[str, int]:
    """Plan + mint both genders. Returns {"f": minted, "m": minted}."""
    zips = {"f": corpus_dir / "te_in_female.zip",
            "m": corpus_dir / "te_in_male.zip"}
    existing = {d.name for d in voices_root.iterdir() if d.is_dir()} \
        if voices_root.is_dir() else set()
    minted = {"f": 0, "m": 0}
    for gender, zpath in zips.items():
        if not zpath.exists():
            print(f"[voicepack] missing corpus zip: {zpath} — skipping {gender}")
            continue
        with zipfile.ZipFile(zpath) as zf:
            speakers = parse_line_index(
                zf.read("line_index.tsv").decode("utf-8"))
        plan = plan_pack(gender, speakers, existing)
        print(f"[voicepack] {gender}: {len(plan)} to mint "
              f"(target {TARGET[gender]})")
        for spec in plan:
            mint(spec, zpath, voices_root)
            minted[gender] += 1
            print(f"[voicepack]   + {spec.voice_id}"
                  + (f"  ({spec.variant_label})" if spec.variant_label else ""))
    return minted


def main() -> None:
    from .studio import studio_dir
    root = studio_dir()
    voices_root = root / "voices"
    corpus_dir = root / "corpus" / "slr66"
    if not corpus_dir.is_dir():
        raise SystemExit(f"[voicepack] no corpus at {corpus_dir}")
    minted = mint_pack(voices_root, corpus_dir)
    packs = [d.name for d in voices_root.iterdir() if d.is_dir()]
    f = sum(1 for n in packs if n.startswith("te_f_"))
    m = sum(1 for n in packs if n.startswith("te_m_"))
    print(f"[voicepack] done — minted {minted}, catalog now "
          f"te_f={f} te_m={m}")


if __name__ == "__main__":
    main()
