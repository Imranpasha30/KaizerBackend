"""Audio analyzer: RMS energy + webrtcvad voice activity."""
from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline_core.live_director.analyzers.base import Analyzer, AnalyzerConfig
from pipeline_core.live_director.signals import SignalFrame


class AudioAnalyzer(Analyzer):
    """Computes audio_rms (0-1) and vad_speaking every interval_s.

    Defaults to 0.3s interval, 0.5s audio window. webrtcvad is lazy-
    loaded on first tick — nothing heavy at import time.
    """

    name = "audio"

    def __init__(self, config: AnalyzerConfig, ring, bus) -> None:
        # Override default interval to match audio cadence
        if config.interval_s == 0.3:
            pass
        super().__init__(config, ring, bus)
        self._vad = None       # lazy: webrtcvad.Vad(aggressiveness)
        self._aggr: int = 2    # aggressiveness 0..3
        self._window_s: float = 0.5
        self._vad_frame_ms: int = 20   # webrtcvad supports 10/20/30 ms only

    def _ensure_vad(self):
        if self._vad is None:
            import webrtcvad  # type: ignore
            self._vad = webrtcvad.Vad(self._aggr)

    async def analyze(self) -> Optional[SignalFrame]:
        try:
            t, samples = await self.ring.latest_audio(seconds=self._window_s)
        except Exception as exc:
            self._log.debug("audio: latest_audio failed: %s", exc)
            return None
        if samples is None or len(samples) == 0:
            return None

        # RMS over the whole window, normalised to int16 peak.
        x = samples.astype(np.float32, copy=False)
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        audio_rms = min(1.0, rms / 32768.0)

        # VAD: slice into 20ms frames (320 samples @ 16kHz), majority-vote.
        self._ensure_vad()
        sr = 16000
        samples_per_frame = int(sr * self._vad_frame_ms / 1000)
        voiced = 0
        total = 0
        # webrtcvad wants bytes of int16
        for i in range(0, len(samples) - samples_per_frame + 1, samples_per_frame):
            chunk = samples[i : i + samples_per_frame]
            try:
                if self._vad.is_speech(chunk.tobytes(), sr):
                    voiced += 1
            except Exception:
                # malformed chunk length — skip
                pass
            total += 1
        vad_speaking = (total > 0) and ((voiced / total) >= 0.4)

        return SignalFrame(
            cam_id=self.config.cam_id,
            t=t,
            audio_rms=audio_rms,
            vad_speaking=vad_speaking,
        )
