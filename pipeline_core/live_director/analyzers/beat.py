"""Beat analyzer: librosa.beat.beat_track phase on rolling 8-second audio window."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from pipeline_core.live_director.analyzers.base import Analyzer
from pipeline_core.live_director.signals import SignalFrame


class BeatAnalyzer(Analyzer):
    name = "beat"

    def __init__(self, config, ring, bus) -> None:
        # Beat tracking wants longer window and slower cadence
        if config.interval_s == 0.3:   # default → override
            config.interval_s = 1.0
        super().__init__(config, ring, bus)
        self._window_s: float = 8.0
        self._sr: int = 16000

    async def analyze(self) -> Optional[SignalFrame]:
        try:
            t, samples = await self.ring.latest_audio(seconds=self._window_s)
        except Exception as exc:
            self._log.debug("beat: latest_audio failed: %s", exc)
            return None
        if samples is None or len(samples) < self._sr:  # need at least 1s
            return None

        x = samples.astype(np.float32, copy=False) / 32768.0

        try:
            import librosa  # type: ignore
            tempo, beat_times = librosa.beat.beat_track(
                y=x, sr=self._sr, units="time"
            )
        except Exception as exc:
            self._log.debug("beat: beat_track failed: %s", exc)
            return None

        tempo_f = float(tempo) if np.isscalar(tempo) else float(np.asarray(tempo).reshape(-1)[0])
        if tempo_f < 40.0 or tempo_f > 200.0 or len(beat_times) < 2:
            # Not plausibly musical
            return SignalFrame(
                cam_id=self.config.cam_id,
                t=t,
                beat_phase=None,
            )

        # Phase of `t` (end of window) relative to nearest beat interval
        intervals = np.diff(beat_times)
        beat_interval = float(np.median(intervals))
        # `t` here is the ring-buffer timestamp of the end of the window.
        # Find position of t relative to last beat_time in the window.
        last_beat = float(beat_times[-1])
        phase = ((t - last_beat) % beat_interval) / beat_interval if beat_interval > 0 else None
        if phase is not None:
            phase = max(0.0, min(1.0, phase))

        return SignalFrame(
            cam_id=self.config.cam_id,
            t=t,
            beat_phase=phase,
        )
