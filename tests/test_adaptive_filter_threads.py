"""_adaptive_filter_threads: long stories drop to fewer filtergraph threads
so 2-wide compose fits in RAM. Pure scheduling — output is byte-identical
regardless of thread count — so this only changes WALL-CLOCK, never
pixels/features. Caps relaxed (long 3, mid 4) after the framesync-heartbeat
fix (9d44cb2): the old multi-GB peaks were the dam, not thread queues."""
from pipeline_v4.v1_bridge import _adaptive_filter_threads as ft


def test_short_news_story_keeps_full_cap():
    # 30-90s news stories are unchanged (fast path).
    assert ft(5, 45.0) == 5
    assert ft(5, 149.9) == 5


def test_mid_story_drops_to_four():
    assert ft(5, 150.0) == 4
    assert ft(5, 240.0) == 4
    assert ft(5, 299.9) == 4


def test_long_podcast_story_drops_to_three():
    # The job-609/610 shape: 3-8 min stories.
    assert ft(5, 300.0) == 3
    assert ft(5, 470.0) == 3


def test_never_raises_above_base():
    # A caller who already lowered the cap is never overridden upward.
    assert ft(2, 470.0) == 2
    assert ft(1, 470.0) == 1
    assert ft(3, 240.0) == 3


def test_never_below_one():
    assert ft(1, 45.0) == 1
    assert ft(0, 470.0) == 1        # base 0 clamped up to 1


def test_bad_duration_treated_as_short():
    assert ft(5, None) == 5
    assert ft(5, "x") == 5
    assert ft(5, 0.0) == 5


def test_thresholds_are_tunable():
    # A caller can push the long threshold up so only huge stories drop.
    assert ft(5, 320.0, long_s=600.0, mid_s=300.0) == 4   # mid band now
    assert ft(5, 620.0, long_s=600.0, mid_s=300.0) == 3
