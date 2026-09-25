"""UA.2 — editorial taxonomy: ~90 spec subcategories resolve to packs."""
from __future__ import annotations

from pipeline_v4.trailer_styles import (DEFAULT_STYLE, STYLES,
                                        SUBCATEGORY_ALIASES, get_style,
                                        resolve_category)


def test_every_alias_targets_a_real_pack():
    assert len(SUBCATEGORY_ALIASES) >= 85, \
        f"spec taxonomy is ~90 subcategories, got {len(SUBCATEGORY_ALIASES)}"
    for sub, pack in SUBCATEGORY_ALIASES.items():
        assert pack in STYLES, f"{sub} -> {pack} (no such pack)"


def test_spec_named_examples_resolve_correctly():
    assert resolve_category("MURDER_HOMICIDE") == "crime"
    assert resolve_category("hurricane_cyclone") == "weather"
    assert resolve_category("STOCK_MARKET_CRASH") == "finance"
    assert resolve_category("true_crime") == "crime"
    assert resolve_category("esports") == "gaming"
    assert resolve_category("underdog_story") == "motivational"
    # pack keys pass through untouched
    assert resolve_category("horror") == "horror"
    # unknown → default, never an error
    assert resolve_category("alien_invasion") == DEFAULT_STYLE
    assert resolve_category("") == DEFAULT_STYLE


def test_get_style_accepts_subcategories():
    assert get_style("MURDER_HOMICIDE").key == "crime"
    assert get_style("pandemic_outbreak").key == "health"
