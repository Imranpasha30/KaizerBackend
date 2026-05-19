"""Stage 1 -- transcription (multi-provider dispatcher redirect).

Originally this module hosted a faster-whisper large-v3 implementation
directly. In Step 4 we abandoned local-model STT in favour of a
multi-provider abstraction (Chirp 3, Whisper-Groq, Deepgram,
AssemblyAI, Sarvam) behind ``pipeline_v2.stages.stt``.

The module path ``pipeline_v2.stages.stage_1_transcribe`` is kept as a
stable import for the orchestrator (Step 10) and any other caller that
references Stage 1 by its conventional name. It re-exports the public
surface from the new ``stt`` package.

Why the rewrite happened:

  1. Local Whisper required GPU memory, model downloads, and CTranslate2
     kernel compilation; failed cold-start budget on slimmer hardware.
  2. On real Telugu audio it exhibited hallucination loops even with
     four safeguard kwargs locked in (see git history at
     ``tests/fixtures/step4_diag/`` for the empirical failure).
  3. Product strategy moved to tiered SaaS where the user picks an STT
     provider (Perplexity-style model dropdown). That doesn't fit a
     single-backend module.

If you need the old faster-whisper implementation, it lives in git
history before the Step 4 expansion commit.
"""

from pipeline_v2.stages.stt import (
    PROVIDERS,
    ProviderResponse,
    TranscriptionProvider,
    register,
    run_stage_1,
)

__all__ = [
    "PROVIDERS",
    "ProviderResponse",
    "TranscriptionProvider",
    "register",
    "run_stage_1",
]
