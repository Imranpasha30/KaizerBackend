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

    # ── Phase 7: layout intelligence ────────────────────────────────────────
    # Split-screen (artist + crowd, Q&A, joke-laugh, interaction moments)
    enable_split_screen: bool = True
    split_min_duration_s: float = 2.5
    # joke-laugh: artist stops speaking then crowd reacts within this window
    joke_laugh_window_s: float = 1.5
    # Q&A: alternating vad_speaking between stage & crowd within this window
    qa_alternation_window_s: float = 3.0
    # interaction: stage cam speaking + crowd cam motion/audio simultaneously
    interaction_min_motion: float = 0.15
    interaction_min_audio: float = 0.1

    # ── Phase 7: dead-air bridge ────────────────────────────────────────────
    enable_bridge: bool = True
    bridge_silence_threshold_s: float = 3.0
    # All cams below this RMS AND no vad_speaking for threshold_s → bridge
    bridge_silence_rms_ceiling: float = 0.02
    bridge_asset_url: str = ""          # image or looping video path
    bridge_min_duration_s: float = 4.0  # don't flip out of bridge too fast


@dataclass
class _CamState:
    """Latest snapshot per camera + derived signals."""
    cam_id: str
    last_frame: Optional[SignalFrame] = None
    vad_speaking_since: Optional[float] = None   # timestamp when vad_speaking flipped True
    vad_speaking_ended_t: Optional[float] = None # timestamp when vad flipped False (for joke-laugh)
    last_beat_phase: Optional[float] = None
    last_reaction_t: Optional[float] = None      # when a reaction last occurred


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

        # Phase 7: layout state
        self._current_layout: str = "single"
        self._current_layout_cams: list[str] = []
        self._layout_entered_t: float = 0.0
        self._in_bridge: bool = False
        self._bridge_entered_t: Optional[float] = None

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
            state.vad_speaking_ended_t = None
        else:
            # Record the moment VAD flipped off (for joke-laugh rule).
            if state.vad_speaking_since is not None:
                state.vad_speaking_ended_t = frame.t
            state.vad_speaking_since = None

        if frame.beat_phase is not None:
            state.last_beat_phase = frame.beat_phase

        # Track reactions (for joke-laugh timing)
        if frame.reaction:
            state.last_reaction_t = frame.t

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

        # ── Phase 7 rules — run BEFORE ordinary cut rules ─────────────────
        # Rule A — dead-air bridge (highest non-override)
        if self.config.enable_bridge:
            if self._should_enter_bridge(now):
                return self._enter_bridge(now)
            # If currently in bridge and the room comes alive, exit to whoever's hot
            if self._in_bridge and self._should_exit_bridge(now):
                exit_cam = self._next_rotation_cam() or self._any_cam()
                if exit_cam:
                    return self._select(
                        exit_cam, now,
                        reason="bridge_exit (signal resumed)",
                        confidence=0.6,
                        layout="single",
                    )
            # Stay in bridge — no emission
            if self._in_bridge:
                return None

        # Rule B — joke-laugh: artist just stopped, crowd reacted → split(artist+crowd)
        if self.config.enable_split_screen and shot_elapsed >= self.config.min_shot_s:
            jl = self._detect_joke_laugh(now)
            if jl is not None:
                artist_cam, crowd_cam = jl
                return self._select(
                    artist_cam, now,
                    reason=f"joke_laugh split (artist={artist_cam}+crowd={crowd_cam})",
                    confidence=0.85,
                    layout="split2_hstack",
                    layout_cams=[artist_cam, crowd_cam],
                )

            # Rule C — interaction: stage speaking + crowd motion/audio live
            inter = self._detect_interaction(now)
            if inter is not None:
                stage_cam, crowd_cam = inter
                return self._select(
                    stage_cam, now,
                    reason=f"interaction split ({stage_cam}+{crowd_cam})",
                    confidence=0.75,
                    layout="split2_hstack",
                    layout_cams=[stage_cam, crowd_cam],
                )

            # Rule D — Q&A: alternating speakers
            qa = self._detect_qa(now)
            if qa is not None:
                a, b = qa
                return self._select(
                    a, now,
                    reason=f"qa split ({a}+{b})",
                    confidence=0.7,
                    layout="split2_hstack",
                    layout_cams=[a, b],
                )

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
        self, cam_id: str, now: float, *,
        reason: str, confidence: float,
        layout: str = "single",
        layout_cams: Optional[list[str]] = None,
        bridge_asset_url: str = "",
    ) -> Optional[CameraSelection]:
        if cam_id in self._blacklist:
            return None
        # Filter any blacklisted secondaries out of layout_cams.
        cams: list[str] = list(layout_cams) if layout_cams else [cam_id]
        cams = [c for c in cams if c not in self._blacklist]
        if not cams:
            return None
        # Decide whether this selection is actually a change vs. the current state.
        same_cam = cam_id == self._current_cam
        same_layout = layout == self._current_layout and cams == self._current_layout_cams
        if same_cam and same_layout:
            return None  # stay (no emit)
        transition = "cut"
        if (
            self.config.crossfade_on_scene_change
            and self._current_cam is not None
            and cam_id != self._current_cam
            and self._is_scene_change(self._current_cam, cam_id)
        ):
            transition = "dissolve"
        self._current_cam = cam_id
        self._current_layout = layout
        self._current_layout_cams = cams
        self._last_cut_t = now
        self._layout_entered_t = now
        # Exiting bridge state on any non-bridge emission
        if layout != "bridge":
            self._in_bridge = False
            self._bridge_entered_t = None
        return CameraSelection(
            t=now, cam_id=cam_id,
            transition=transition,
            confidence=min(1.0, max(0.0, confidence)),
            reason=reason,
            layout=layout,
            layout_cams=cams,
            bridge_asset_url=bridge_asset_url,
        )

    # ── Phase 7: bridge + layout detectors ──────────────────────────────────

    def _should_enter_bridge(self, now: float) -> bool:
        """All cameras silent (low RMS + no VAD) for at least threshold seconds.
        Only fires when a bridge asset is configured — without an asset there is
        nothing to bridge to, so silent scenes keep using the current camera."""
        if self._in_bridge:
            return False
        if not self.config.bridge_asset_url:
            return False
        threshold_t = now - self.config.bridge_silence_threshold_s
        for state in self._cam_states.values():
            f = state.last_frame
            if f is None:
                continue
            # If any cam has VAD or non-trivial RMS more recently than threshold → not silent.
            if state.vad_speaking_since is not None and state.vad_speaking_since >= threshold_t:
                return False
            if f.t >= threshold_t and f.audio_rms > self.config.bridge_silence_rms_ceiling:
                return False
        # Also require we have SEEN at least one frame (avoid tripping on empty state)
        if not any(s.last_frame is not None for s in self._cam_states.values()):
            return False
        return True

    def _should_exit_bridge(self, now: float) -> bool:
        """Exit bridge only after bridge_min_duration_s, and when any cam has audio/VAD."""
        if not self._in_bridge:
            return False
        if self._bridge_entered_t is not None:
            if (now - self._bridge_entered_t) < self.config.bridge_min_duration_s:
                return False
        for state in self._cam_states.values():
            f = state.last_frame
            if f is None:
                continue
            if state.vad_speaking_since is not None:
                return True
            if f.audio_rms > self.config.bridge_silence_rms_ceiling * 2:
                return True
        return False

    def _enter_bridge(self, now: float) -> Optional[CameraSelection]:
        """Emit a CameraSelection carrying layout='bridge' + the configured asset url."""
        # Primary cam_id still required by the dataclass; use current or first cam.
        primary = self._current_cam or (self.camera_ids[0] if self.camera_ids else "")
        if not primary:
            return None
        self._in_bridge = True
        self._bridge_entered_t = now
        sel = CameraSelection(
            t=now, cam_id=primary,
            transition="cut",
            confidence=0.95,
            reason="bridge (dead_air)",
            layout="bridge",
            layout_cams=[primary],
            bridge_asset_url=self.config.bridge_asset_url,
        )
        self._current_layout = "bridge"
        self._current_layout_cams = [primary]
        self._last_cut_t = now
        self._layout_entered_t = now
        return sel

    def _any_cam(self) -> Optional[str]:
        for c in self.camera_ids:
            if c not in self._blacklist:
                return c
        return None

    def _detect_joke_laugh(self, now: float) -> Optional[tuple[str, str]]:
        """Artist just stopped speaking within window AND a crowd reaction exists
        within window → return (artist_cam, crowd_cam)."""
        w = self.config.joke_laugh_window_s
        artist_cam: Optional[str] = None
        best_end_t = -1.0
        for cid, state in self._cam_states.items():
            if cid in self._blacklist:
                continue
            if state.vad_speaking_ended_t is None:
                continue
            if (now - state.vad_speaking_ended_t) > w:
                continue
            if state.vad_speaking_ended_t > best_end_t:
                best_end_t = state.vad_speaking_ended_t
                artist_cam = cid
        if artist_cam is None:
            return None
        crowd_cam: Optional[str] = None
        best_reaction_t = -1.0
        for cid, state in self._cam_states.items():
            if cid == artist_cam or cid in self._blacklist:
                continue
            f = state.last_frame
            if not f or not f.reaction:
                continue
            if state.last_reaction_t is None:
                continue
            if (now - state.last_reaction_t) > w:
                continue
            if state.last_reaction_t > best_reaction_t:
                best_reaction_t = state.last_reaction_t
                crowd_cam = cid
        if crowd_cam is None:
            return None
        return (artist_cam, crowd_cam)

    def _detect_interaction(self, now: float) -> Optional[tuple[str, str]]:
        """Stage cam currently speaking + a crowd cam showing motion/audio.
        Returns (stage_cam, crowd_cam) or None."""
        stage_cam: Optional[str] = None
        hold_s = self.config.speaker_vad_hold_ms / 1000.0
        for cid, state in self._cam_states.items():
            if cid in self._blacklist:
                continue
            if state.vad_speaking_since is None:
                continue
            if (now - state.vad_speaking_since) < hold_s:
                continue
            f = state.last_frame
            if f and f.face_present:
                stage_cam = cid
                break
        if stage_cam is None:
            return None
        crowd_cam: Optional[str] = None
        best_score = 0.0
        for cid, state in self._cam_states.items():
            if cid == stage_cam or cid in self._blacklist:
                continue
            f = state.last_frame
            if not f:
                continue
            if f.motion_mag < self.config.interaction_min_motion and \
               f.audio_rms < self.config.interaction_min_audio:
                continue
            score = f.motion_mag + f.audio_rms
            if score > best_score:
                best_score = score
                crowd_cam = cid
        if crowd_cam is None:
            return None
        return (stage_cam, crowd_cam)

    def _detect_qa(self, now: float) -> Optional[tuple[str, str]]:
        """Alternating speakers within qa_alternation_window_s — return the
        two most-recent speaking cams."""
        w = self.config.qa_alternation_window_s
        speaking_cams: list[tuple[float, str]] = []
        for cid, state in self._cam_states.items():
            if cid in self._blacklist:
                continue
            if state.vad_speaking_since is not None:
                speaking_cams.append((state.vad_speaking_since, cid))
            elif state.vad_speaking_ended_t is not None and (now - state.vad_speaking_ended_t) < w:
                speaking_cams.append((state.vad_speaking_ended_t, cid))
        if len(speaking_cams) < 2:
            return None
        # Require both happened within the window
        speaking_cams.sort(key=lambda x: x[0], reverse=True)
        t1, cam1 = speaking_cams[0]
        t2, cam2 = speaking_cams[1]
        if (now - t2) > w:
            return None
        return (cam1, cam2)

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
