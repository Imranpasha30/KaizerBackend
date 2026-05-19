# step4_diag/ — historical evidence (abandoned local Whisper backend)

These JSONs are kept on purpose. They are NOT current production data.
They record the failure that prompted abandoning local
faster-whisper in favour of the multi-provider STT abstraction in
``pipeline_v2/stages/stt/``.

## Files

| File | Source | What it shows |
|---|---|---|
| `raw_a_vad_on.json`  | faster-whisper large-v3, default kwargs + VAD       | Severe hallucination loop. Same 6-word Telugu phrase emitted across 4 minutes of audio. Compression-ratio safety net didn't fire because the repeated phrase scored 1.378, below the 2.4 threshold. |
| `raw_b_vad_off.json` | faster-whisper large-v3, default kwargs minus VAD   | Hallucination loop persists. Confirms VAD remapping was not the cause. |
| `raw_c_postfix.json` | faster-whisper large-v3, 5 safeguard kwargs locked  | Loop broken: 524 words across 54 segments instead of 139 words across 22. Word-timestamp drift remained large enough that the user spot-check failed. Combined with cold-start cost (~3GB model download, 128.9s first load), led to abandoning local Whisper. |

## Why kept

If anyone later proposes "let's add local Whisper back as a free-tier
provider", point them here. The empirical answer to "does it work on
Telugu out of the box" is no, even with five carefully-tuned safeguard
kwargs the timestamps are still off by enough to fail a spot-check.

## Reproducing (do not unless investigating a similar bug)

The diagnostic script lives at
``pipeline_v2/scripts/step4_diagnose.py``. It still works -- it bypasses
the (now removed) pipeline_v2.stages.stage_1_transcribe wrapper and
calls ``faster_whisper.WhisperModel.transcribe()`` directly. To re-run
you would need to ``pip install faster-whisper`` (we removed it from
pyproject.toml in Step 4.0).

## Provenance

Captured on 2026-05-18 against fixture
``E:\kaizer new data training\videos\test.mp3`` (11:53 Telugu raw
podcast footage). RTX 5060, 7.96 GB VRAM, int8_float16, large-v3.
