"""Round-2 brutal-review fixes, pinned:
1. vph comparability (30-day velocity cap for BOTH data sources)
2. exploration no longer punished (any strong hook form earns the 4 pts)
3. concatenated brand forms filtered ("kaizernews" topic bug)
4. snapshot pruning (bounded growth)
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import models
from learning import seo_learning as sl
from seo.verifier import _score_title


# ── 1. vph comparability ────────────────────────────────────────────

def test_row_vph_caps_lifetime_age_at_30_days():
    old = SimpleNamespace(_views=72000, _age_h=7200.0, views_per_hour=10.0)
    assert sl._row_vph(old) == 100.0        # 72000/720, NOT 72000/7200
    fresh = SimpleNamespace(_views=500, _age_h=5.0, views_per_hour=0.0)
    assert sl._row_vph(fresh) == 100.0      # young age used as-is


def test_row_vph_uses_sample_fields_uniformly():
    s = SimpleNamespace(views=1440, hours_since_publish=1440.0,
                        views_per_hour=1.0)
    assert sl._row_vph(s) == 2.0            # 1440/720 capped, not stored 1.0


def test_row_vph_falls_back_to_stored():
    bare = SimpleNamespace(views_per_hour=7.5)
    assert sl._row_vph(bare) == 7.5


# ── 2. exploration fairness in the verifier ─────────────────────────

def test_question_hook_earns_hook_points_without_power_word():
    t = "Will the court rule today? తీర్పు నేడు వస్తుందా? పూర్తి వివరాలు ఇక్కడ"
    _, fails = _score_title(t)
    assert not any("NO hook" in f or "POWER WORD" in f for f in fails)


def test_number_hook_earns_hook_points():
    t = "10 ఎకరాల భూమి వివాదం — Land Case Full Details And Records Explained"
    _, fails = _score_title(t)
    assert not any("NO hook" in f for f in fails)


def test_hookless_title_still_fails():
    t = "భూమి వివాదం వివరాలు — Land Case Details And Updates From Court Site"
    _, fails = _score_title(t)
    assert any("NO hook" in f for f in fails)


# ── 3. concatenated brand filtering ─────────────────────────────────

def test_concatenated_brand_form_filtered():
    terms = sl._title_terms("Ambedkar KaizerNews special report",
                            brand_tokens={"kaizer", "news", "telugu"})
    assert "kaizernews" not in terms
    assert not any("kaizernews" in t for t in terms)
    assert "ambedkar" in terms


def test_short_brand_token_not_substring_matched():
    # "news" (4 chars) must NOT nuke "newsworthy"-like words by substring.
    terms = sl._title_terms("Newsworthy verdict for farmers",
                            brand_tokens={"news"})
    assert "newsworthy" in terms


# ── 4. snapshot pruning ─────────────────────────────────────────────

@pytest.fixture()
def db():
    eng = create_engine("sqlite://")
    models.Base.metadata.create_all(
        eng, tables=[models.TrainingSample.__table__,
                     models.SeoLearningSnapshot.__table__])
    s = sessionmaker(bind=eng)()
    yield s
    s.close()


def test_snapshots_pruned_to_30_per_window(db):
    db.add(models.TrainingSample(
        video_id="v0", channel_id=5, seo_title="t? టెస్ట్",
        views_per_hour=1.0, first_seen_at=datetime.now(timezone.utc)))
    db.commit()
    for i in range(35):
        sl.snapshot(db, 5, windows=(30,))
    kept = (db.query(models.SeoLearningSnapshot)
            .filter(models.SeoLearningSnapshot.channel_id == 5,
                    models.SeoLearningSnapshot.window_days == 30).count())
    assert kept == 30
