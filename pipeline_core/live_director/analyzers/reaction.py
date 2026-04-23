"""Reaction analyzer: audience audio-event heuristic (laugh / cheer / clap)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline_core.live_director.analyzers.base import Analyzer
from pipeline_core.live_director.signals import SignalFrame


# Thresholds — tune later from real event telemetry
_SPIKE_RATIO = 1.5          # recent RMS vs trailing trend
_CENTROID_LAUGH = 3000.0    # Hz, spectral centroid above → laugh-like
_CENTROID_CLAP_LO = 1500.0
_CENTROID_CLAP_HI = 3000.0


class ReactionAnalyzer(Analyzer):
    name = "reaction"

    def __init__(self, config, ring, bus) -> None:
        super().__init__(config, ring, bus)
        self._window_s: float = 2.0
        self._sr: int = 16000

    async def analyze(self) -> Optional[SignalFrame]:
        try:
            t, samples = await self.ring.latest_audio(seconds=self._window_s)
        except Exception as exc:
            self._log.debug("reaction: latest_audio failed: %s", exc)
            return None
        if samples is None or len(samples) < self._sr // 2:
            return None

        x = samples.astype(np.float32, copy=False)

        # Recent (last 500ms) vs trailing (first 1500ms) RMS
        n = len(x)
        recent_n = min(self._sr // 2, n)
        recent = x[-recent_n:]
        trailing = x[:-recent_n] if n > recent_n else x
        recent_rms = float(np.sqrt(np.mean(recent * recent) + 1e-12))
        trailing_rms = float(np.sqrt(np.mean(trailing * trailing) + 1e-12))

        reaction: Optional[str] = None
        if trailing_rms > 1.0 and recent_rms > _SPIKE_RATIO * trailing_rms:
            # Spectral centroid on the recent window
            try:
                import librosa  # type: ignore
                centroids = librosa.feature.spectral_centroid(
                    y=recent / 32768.0, sr=self._sr, n_fft=1024, hop_length=256
                )
                centroid = float(np.mean(centroids)) if centroids.size else 0.0
            except Exception as exc:
                self._log.debug("reaction: centroid failed: %s", exc)
                centroid = 0.0

            if centroid > _CENTROID_LAUGH:
                reaction = "laugh"
            elif _CENTROID_CLAP_LO <= centroid <= _CENTROID_CLAP_HI:
                reaction = "clap"
            elif centroid < _CENTROID_CLAP_LO:
                reaction = "cheer"

        return SignalFrame(
            cam_id=self.config.cam_id,
            t=t,
            reaction=reaction,
        )
