"""
kaizer.pipeline.live_director.director
=======================================
Rule-engine director: consumes SignalFrames from the SignalBus and emits
CameraSelection decisions.

Rule priority (highest → lowest; first match wins)
---------------------------------------------------
  1. Manual override in effect       → override.cam_id
  2. Critical reaction               → crowd cam with highest audio
     (reaction in {'laugh','cheer','clap'} on any camera,
      confidence ≥ reaction_threshold, min_shot elapsed)
  3. Designated speaker active       → camera with largest face bbox
     (vad_speaking=True + face_present=True for ≥ speaker_vad_hold_ms)
  4. Beat cut during music           → next-in-rotation by energy
     (beat_phase near 0 AND last_cut ≥ min_shot)
  5. Min-shot floor                  → stay (no cut under min_shot_s)
  6. Max-shot ceiling                → force cut to next highest-scoring
  7. Default                         → stay

Config is stored on Director.config; editable at runtime via
set_config() — takes effect on the next decision tick.

Manual overrides (operator control):
  - pin(cam_id)      : force this camera until unpin()
  - unpin()          : release pin
  - blacklist(cam_id): never cut to this camera
  - force_cut(cam_id): one-shot override cut (clears on next decision)
  - allow(cam_id)    : remove cam from blacklist
All overrides are logged as DirectorEvent(kind='override').

The director NEVER raises from its loop. Errors log and the next tick
re-evaluates from the last-known state.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from pipeline_core.live_director.signal_bus import SignalBus
from pipeline_core.live_director.signals import (
    CameraSelection,
    DirectorEvent,
    SignalFrame,
)

logger = logging.getLogger("kaizer.pipeline.live_director.director")


@dataclass
class DirectorConfig:
    """Tuneable decision parameters. Stored per-event in live_events.config_json."""
    min_shot_s: float = 2.5
    max_shot_s: float = 12.0
    reaction_threshold: float = 0.7
    speaker_vad_hold_ms: float = 400.0
    beat_cut_every_nth_bar: int = 2
    crossfade_on_scene_change: bool = True
    # Scores below this mute "interesting enough to cut to you" for rotation
    min_rotation_energy: float = 0.02


@dataclass
class _CamState:
    """Latest snapshot per camera + derived signals."""
    cam_id: str
    last_frame: Optional[SignalFrame] = None
    vad_speaking_since: Optional[float] = None   # timestamp when vad_speaking flipped True
    last_beat_phase: Optional[float] = None


class Director:
    """Rule-engine director. Instantiate once per live event, then call run()."""

    def __init__(
        self,
        event_id: int,
        camera_ids: list[str],
        bus: SignalBus,
        config: Optional[DirectorConfig] = None,
        *,
        on_selection: Optional[callable] = None,
        on_event: Optional[callable] = None,
    ) -> None:
        self.event_id = event_id
        self.camera_ids = list(camera_ids)
        self.bus = bus
        self.config = config or DirectorConfig()
        self._on_selection = on_selection   # async callback(CameraSelection)
        self._on_event = on_event           # async callback(DirectorEvent)

        # Decision state
        self._current_cam: Optional[str] = None
        self._last_cut_t: float = 0.0
        self._cam_states: dict[str, _CamState] = {
            cid: _CamState(cam_id=cid) for cid in self.camera_ids
        }

        # Override state
        self._pin: Optional[str] = None
        self._blacklist: set[str] = set()
        self._one_shot_cut: Optional[str] = None

        # Bar-count state for beat rule
        self._beats_since_last_cut: int = 0
        self._running: bool = False

    # ── Operator controls ────────────────────────────────────────────────────

    def pin(self, cam_id: str) -> None:
        if cam_id not in self.camera_ids:
            raise ValueError(f"Unknown camera {cam_id!r}")
        self._pin = cam_id
        self._emit_event(kind="override", cam_id=cam_id, reason="pinned")

    def unpin(self) -> None:
        prev = self._pin
        self._pin = None
        if prev:
            self._emit_event(kind="override", cam_id=prev, reason="unpinned")

    def blacklist(self, cam_id: str) -> None:
        if cam_id not in self.camera_ids:
            raise ValueError(f"Unknown camera {cam_id!r}")
        self._blacklist.add(cam_id)
        self._emit_event(kind="override", cam_id=cam_id, reason="blacklisted")

    def allow(self, cam_id: str) -> None:
        self._blacklist.discard(cam_id)
        self._emit_event(kind="override", cam_id=cam_id, reason="allowed")

    def force_cut(self, cam_id: str) -> None:
        if cam_id not in self.camera_ids:
            raise ValueError(f"Unknown camera {cam_id!r}")
        self._one_shot_cut = cam_id
        self._emit_event(kind="override", cam_id=cam_id, reason="force_cut")

    def set_config(self, config: DirectorConfig) -> None:
        self.config = config

    # ── Main loop ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._running = True
        try:
            async for frame in self.bus.subscribe():
                if not self._running:
                    break
                try:
                    self._ingest(frame)
                    selection = self._decide(now=frame.t)
                    if selection is not None:
                        await self._emit_selection(selection)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error(
                        "director: decision tick crashed: %s — continuing", exc,
                    )
        except asyncio.CancelledError:
            logger.debug("director: run() cancelled")
            return

    def stop(self) -> None:
        self._running = False

    # ── Internal: absorb a SignalFrame into per-camera state ────────────────

    def _ingest(self, frame: SignalFrame) -> None:
        if frame.cam_id not in self._cam_states:
            return
        state = self._cam_states[frame.cam_id]
        # Merge: keep previous fields when incoming is None (partial updates).
        prev = state.last_frame
        if prev is None:
            state.last_frame = frame
        else:
            merged = SignalFrame(
                cam_id=frame.cam_id,
                t=max(frame.t, prev.t),
                audio_rms=frame.audio_rms if frame.audio_rms else prev.audio_rms,
                vad_speaking=frame.vad_speaking or prev.vad_speaking,
                face_present=frame.face_present or prev.face_present,
                face_size_norm=max(frame.face_size_norm, prev.face_size_norm),
                face_identity=frame.face_identity or prev.face_identity,
                scene=frame.scene if frame.scene != "unknown" else prev.scene,
                motion_mag=max(frame.motion_mag, prev.motion_mag),
                reaction=frame.reaction or prev.reaction,
                beat_phase=frame.beat_phase if frame.beat_phase is not None else prev.beat_phase,
            )
            state.last_frame = merged

        # Track when the VAD-speaking window started (for the hold-time rule).
        if state.last_frame.vad_speaking:
            if state.vad_speaking_since is None:
                state.vad_speaking_since = frame.t
        else:
            state.vad_speaking_since = None

        if frame.beat_phase is not None:
            state.last_beat_phase = frame.beat_phase

    # ── Internal: run the rule ladder and return a CameraSelection ─────────

    def _decide(self, now: float) -> Optional[CameraSelection]:
        # Rule 1 — manual override (one-shot force > pin)
        if self._one_shot_cut is not None:
            cam = self._one_shot_cut
            self._one_shot_cut = None
            return self._select(cam, now, reason="force_cut override", confidence=1.0)
        if self._pin is not None and self._pin not in self._blacklist:
            if self._current_cam != self._pin:
                return self._select(self._pin, now, reason="pinned override", confidence=1.0)
            return None

        shot_elapsed = now - self._last_cut_t if self._current_cam else float("inf")

        # Rule 2 — critical reaction
        cam_with_reaction, reaction_score = self._find_reaction_cam()
        if (
            cam_with_reaction
            and reaction_score >= self.config.reaction_threshold
            and shot_elapsed >= self.config.min_shot_s
        ):
            return self._select(
                cam_with_reaction, now,
                reason=f"reaction:{self._cam_states[cam_with_reaction].last_frame.reaction}",
                confidence=reaction_score,
            )

        # Rule 3 — designated speaker active (vad+face held for hold_ms)
        speaker_cam = self._find_speaker_cam(now)
        if speaker_cam and shot_elapsed >= self.config.min_shot_s:
            return self._select(
                speaker_cam, now,
                reason="speaker_active (vad+face held)",
                confidence=0.9,
            )

        # Rule 4 — beat cut during music
        if (
            self._is_music_beat_now()
            and shot_elapsed >= self.config.min_shot_s
            and self._beats_since_last_cut >= self.config.beat_cut_every_nth_bar
        ):
            beat_cam = self._next_rotation_cam()
            if beat_cam:
                self._beats_since_last_cut = 0
                return self._select(
                    beat_cam, now,
                    reason=f"beat_cut (tempo phase≈0)",
                    confidence=0.5,
                )
        # Still advance the beat counter even when we don't cut
        if self._is_music_beat_now():
            self._beats_since_last_cut += 1

        # Rule 5 — min-shot floor
        if shot_elapsed < self.config.min_shot_s:
            return None

        # Rule 6 — max-shot ceiling
        if shot_elapsed >= self.config.max_shot_s:
            next_cam = self._next_rotation_cam()
            if next_cam and next_cam != self._current_cam:
                return self._select(
                    next_cam, now,
                    reason=f"max_shot exceeded ({shot_elapsed:.1f}s)",
                    confidence=0.4,
                )

        # Rule 7 — default: stay
        return None

    # ── Rule helpers ─────────────────────────────────────────────────────────

    def _find_reaction_cam(self) -> tuple[Optional[str], float]:
        best_cam: Optional[str] = None
        best_score = 0.0
        for cid, state in self._cam_states.items():
            if cid in self._blacklist:
                continue
            f = state.last_frame
            if not f or not f.reaction:
                continue
            # Confidence ≈ audio_rms (crowd cheer = loud, crowd cam = loud)
            score = max(f.audio_rms, 0.0)
            if score > best_score:
                best_score = score
                best_cam = cid
        return best_cam, best_score

    def _find_speaker_cam(self, now: float) -> Optional[str]:
        best_cam: Optional[str] = None
        best_size = 0.0
        hold_s = self.config.speaker_vad_hold_ms / 1000.0
        for cid, state in self._cam_states.items():
            if cid in self._blacklist:
                continue
            f = state.last_frame
            if not f:
                continue
            if state.vad_speaking_since is None:
                continue
            if (now - state.vad_speaking_since) < hold_s:
                continue
            if not f.face_present:
                continue
            if f.face_size_norm > best_size:
                best_size = f.face_size_norm
                best_cam = cid
        return best_cam

    def _is_music_beat_now(self) -> bool:
        """Heuristic: at least one camera reported a beat_phase near 0 recently."""
        for state in self._cam_states.values():
            if state.last_beat_phase is None:
                continue
            if state.last_beat_phase < 0.15 or state.last_beat_phase > 0.85:
                return True
        return False

    def _next_rotation_cam(self) -> Optional[str]:
        """Pick the next camera in rotation, scored by audio_rms + motion_mag,
        excluding blacklist + current."""
        scored: list[tuple[str, float]] = []
        for cid, state in self._cam_states.items():
            if cid in self._blacklist:
                continue
            if cid == self._current_cam:
                continue
            f = state.last_frame
            energy = (f.audio_rms if f else 0.0) + (f.motion_mag if f else 0.0)
            scored.append((cid, energy))
        if not scored:
            return None
        scored.sort(key=lambda x: x[1], reverse=True)
        top_cam, top_energy = scored[0]
        if top_energy < self.config.min_rotation_energy:
            # Nothing interesting; rotate round-robin
            idx = self.camera_ids.index(self._current_cam) if self._current_cam in self.camera_ids else -1
            for i in range(1, len(self.camera_ids) + 1):
                cand = self.camera_ids[(idx + i) % len(self.camera_ids)]
                if cand != self._current_cam and cand not in self._blacklist:
                    return cand
            return None
        return top_cam

    # ── Emission ─────────────────────────────────────────────────────────────

    def _select(
        self, cam_id: str, now: float, *, reason: str, confidence: float,
    ) -> Optional[CameraSelection]:
        if cam_id in self._blacklist:
            return None
        if cam_id == self._current_cam:
            return None  # stay (no emit)
        transition = "cut"
        if (
            self.config.crossfade_on_scene_change
            and self._current_cam is not None
            and self._is_scene_change(self._current_cam, cam_id)
        ):
            transition = "dissolve"
        self._current_cam = cam_id
        self._last_cut_t = now
        return CameraSelection(
            t=now, cam_id=cam_id,
            transition=transition,
            confidence=min(1.0, max(0.0, confidence)),
            reason=reason,
        )

    def _is_scene_change(self, old_cam: str, new_cam: str) -> bool:
        old_scene = self._cam_states.get(old_cam, _CamState(cam_id=old_cam))
        new_scene = self._cam_states.get(new_cam, _CamState(cam_id=new_cam))
        old_s = old_scene.last_frame.scene if old_scene.last_frame else "unknown"
        new_s = new_scene.last_frame.scene if new_scene.last_frame else "unknown"
        return old_s != new_s

    async def _emit_selection(self, selection: CameraSelection) -> None:
        logger.info(
            "director: select → %s (%s, conf=%.2f)",
            selection.cam_id, selection.reason, selection.confidence,
        )
        if self._on_selection is not None:
            try:
                await self._on_selection(selection)
            except Exception as exc:
                logger.error("director: on_selection callback crashed: %s", exc)

    def _emit_event(self, *, kind: str, cam_id: str = "", reason: str = "") -> None:
        ev = DirectorEvent(
            t=time.monotonic(),
            kind=kind,
            payload={"cam_id": cam_id, "reason": reason},
        )
        logger.info("director: event %s cam=%s reason=%r", kind, cam_id, reason)
        if self._on_event is None:
            return
        try:
            # on_event may be sync or async — support both.
            result = self._on_event(ev)
            if asyncio.iscoroutine(result):
                asyncio.get_event_loop().create_task(result)
        except Exception as exc:
            logger.error("director: on_event callback crashed: %s", exc)
