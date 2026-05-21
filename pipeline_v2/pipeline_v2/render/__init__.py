"""Render helpers for the item-117 architecture.

Phase 4a + 4b split: ``edl_builder`` produces the pure filter_complex
string consumed by ``stages.stage_4_raw_extract`` for the single-
ffmpeg multi-output decode pass. Phase 4b overlay steps live under
``stages.stage_4_bulletin_overlay`` + ``stages.stage_4_shorts_overlay``.
"""
