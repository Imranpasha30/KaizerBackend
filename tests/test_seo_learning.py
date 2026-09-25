"""learning/seo_learning.py — the REAL learning brain.

Deterministic classifiers + aggregation + policy gating, offline against an
in-memory SQLite DB. The point pinned here: with enough real samples the
module emits a POLICY that steers generation; with too little data it emits
None and generation behaves exactly as before (no faking)."""
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import models
from learning import seo_learning as sl


# ── classifiers ────────────────────────────────────────────────────

def test_classify_script():
    assert sl.classify_script("Breaking: పవన్ కళ్యాణ్ భూ వివాదం") == "mixed"
    assert sl.classify_script("Pawan Kalyan Land Row Explained") == "english"
    assert sl.classify_script("పవన్ కళ్యాణ్ భూ వివాదంపై ఆరోపణలు") == "native"
    assert sl.classify_script("!!! 123") == "none"


def test_classify_hook():
    assert sl.classify_hook("పవన్ కళ్యాణ్ కు నైతిక హక్కు ఉందా?") == "question"
    assert sl.classify_hook("10 ఎకరాల భూమి వివాదం వెలుగులోకి") == "number"
    assert sl.classify_hook('"నేను వెనక్కి తగ్గను" అన్న పవన్ "సంచలనం"') == "quote"
    assert sl.classify_hook("Breaking: భూ వివాదంపై ఆరోపణలు") == "power"
    assert sl.classify_hook("హైదరాబాద్ లో కొత్త మెట్రో లైన్ పనులు") == "plain"


def test_classify_len_band():
    assert sl.classify_len_band("x" * 40) == "short"
    assert sl.classify_len_band("x" * 65) == "sweet"
    assert sl.classify_len_band("x" * 90) == "long"


# ── aggregation + policy over a real (SQLite) TrainingSample table ──

@pytest.fixture()
def db():
    eng = create_engine("sqlite://")
    models.Base.metadata.create_all(
        eng, tables=[models.TrainingSample.__table__,
                     models.SeoLearningSnapshot.__table__])
    s = sessionmaker(bind=eng)()
    yield s
    s.close()


def _sample(db, i, title, vph, ctr=None, kws=()):
    db.add(models.TrainingSample(
        video_id=f"v{i}", channel_id=7, seo_title=title,
        seo_keywords=list(kws), views_per_hour=vph, ctr=ctr,
        first_seen_at=datetime.now(timezone.utc)))
    db.commit()


def test_policy_none_below_min_samples(db):
    for i in range(sl.MIN_POLICY_N - 1):
        _sample(db, i, "Breaking: టెస్ట్ హెడ్లైన్ " + "x" * 40, vph=5.0)
    out = sl.compute_channel_learning(db, 7, 30)
    assert out["samples"] == sl.MIN_POLICY_N - 1
    assert out["policy"] is None            # honest: not enough data yet


def test_policy_learns_winning_hook_and_script(db):
    # questions in mixed script consistently outperform power-word natives
    for i in range(4):
        _sample(db, i, f"Will Pawan win? పవన్ గెలుస్తాడా? రౌండ్ {i}",
                vph=20.0 + i, ctr=0.06, kws=("pawan kalyan", "telangana"))
    for i in range(4, 8):
        _sample(db, i, f"షాకింగ్: సంచలన ఆరోపణలు వెల్లడి {i}",
                vph=4.0, ctr=0.02, kws=("pawan kalyan", "crime"))
    out = sl.compute_channel_learning(db, 7, 30)
    assert out["samples"] == 8
    pol = out["policy"]
    assert pol is not None
    assert pol["best_hooks"][0] == "question"
    assert pol["best_script"] == "mixed"
    # Performance-weighted mining: WINNER-only terms outrank terms that
    # also rode the losing videos ("pawan kalyan" was tagged on both
    # groups, so pure winner keywords like "telangana" lead).
    assert "telangana" in pol["top_keywords"]
    all_kw = [k["keyword"] for k in out["top_keywords"]]
    assert "pawan kalyan" in all_kw       # still present in the full list


def test_snapshot_and_latest_policy_roundtrip(db):
    for i in range(6):
        _sample(db, i, f"Court verdict today? తీర్పు నేడు వస్తుందా? {i}",
                vph=10.0, ctr=0.05, kws=("court", "verdict"))
    n = sl.snapshot(db, 7)
    assert n == 4                            # 7/30/90-day + ALL-TIME windows
    pol = sl.latest_policy(db, 7)
    assert pol and pol["best_hooks"]
    assert pol["based_on"] == 6


def test_learned_block_renders_into_prompt():
    from seo.prompts import build_user_prompt

    class _Clip:
        meta = "{}"
        text = "టెస్ట్ వార్త"
        sentiment = ""
        duration = 30.0

    p = build_user_prompt(
        clip=_Clip(), language="te",
        learned={"best_hooks": ["question"], "best_script": "mixed",
                 "best_len_band": "sweet",
                 "top_keywords": ["pawan kalyan", "telangana"],
                 "based_on": 12})
    assert "CHANNEL LEARNING" in p
    assert "QUESTION headline" in p
    assert "pawan kalyan" in p
    assert "12 published" in p


def test_no_learned_block_when_none():
    from seo.prompts import build_user_prompt

    class _Clip:
        meta = "{}"
        text = "టెస్ట్"
        sentiment = ""
        duration = 30.0

    p = build_user_prompt(clip=_Clip(), language="te", learned=None)
    assert "CHANNEL LEARNING" not in p       # absent data → prompt unchanged


# ── best-time-to-post report ───────────────────────────────────────

from datetime import timedelta


def _sample_at(db, i, vph, when, ctr=None):
    db.add(models.TrainingSample(
        video_id=f"t{i}", channel_id=9,
        seo_title="Court verdict today? తీర్పు నేడు వస్తుందా? " + "x" * 20,
        seo_keywords=[], views_per_hour=vph, ctr=ctr, first_seen_at=when))
    db.commit()


def test_best_times_empty_state_below_min_bucket(db):
    # Two videos in one slot — below MIN_BUCKET_N: nothing may be crowned.
    base = datetime(2026, 1, 5, 5, 30, tzinfo=timezone.utc)  # IST 11:00 Mon
    for i in range(2):
        _sample_at(db, i, 10.0, base + timedelta(days=i * 7))
    rep = sl.best_times_report(db, 9, 3650)
    assert rep["enough_data"] is False
    assert rep["recommendation"]["best_hours"] == []
    assert rep["signal"] == "views"
    assert len(rep["hours"]) == 24 and len(rep["days"]) == 7  # full grids always
    assert rep["peak"] is None                # nothing crowned below MIN_BUCKET_N
    assert len(rep["grid"]) == 7 and all(len(r["hours"]) == 24 for r in rep["grid"])


def test_best_times_recommends_high_vph_hour_ist(db):
    # 4 strong videos at UTC 05:30 == IST 11:00; 4 weak at UTC 12:30 == IST 18:00.
    strong = datetime(2026, 1, 6, 5, 30, tzinfo=timezone.utc)   # IST 11:00
    weak = datetime(2026, 1, 6, 12, 30, tzinfo=timezone.utc)    # IST 18:00
    for i in range(4):
        _sample_at(db, i, 40.0, strong + timedelta(days=i))
    for i in range(4, 8):
        _sample_at(db, i, 4.0, weak + timedelta(days=i))
    rep = sl.best_times_report(db, 9, 3650)
    assert rep["enough_data"] is True
    assert rep["signal"] == "views"                 # no CTR yet → honest proxy
    assert rep["recommendation"]["best_hours"][0] == 11
    # The 18:00 slot exists in the grid but never leads.
    assert rep["recommendation"]["best_hours"][0] != 18
    h11 = next(h for h in rep["hours"] if h["hour"] == 11)
    assert h11["n"] == 4 and h11["avg_vph"] == 40.0


def test_best_times_grid_and_peak(db):
    # 3 strong videos all on Tuesdays at IST 11:00 (same day×hour cell) beat a
    # separate weaker Wednesday slot → the peak is that single cell.
    for i, d in enumerate((6, 13, 20)):            # Jan 2026 Tuesdays
        _sample_at(db, i, 90.0, datetime(2026, 1, d, 5, 30, tzinfo=timezone.utc))
    for i, d in enumerate((7, 14, 21), start=3):   # Wednesdays, weaker
        _sample_at(db, i, 5.0, datetime(2026, 1, d, 12, 30, tzinfo=timezone.utc))
    rep = sl.best_times_report(db, 9, 3650)
    assert len(rep["grid"]) == 7
    assert all(len(r["hours"]) == 24 for r in rep["grid"])
    assert rep["peak"] is not None
    assert rep["peak"]["dow"] == "Tue" and rep["peak"]["hour"] == 11
    assert rep["peak"]["n"] == 3 and rep["peak"]["avg_vph"] == 90.0
    tue = next(r for r in rep["grid"] if r["dow"] == "Tue")
    cell = next(c for c in tue["hours"] if c["hour"] == 11)
    assert cell["n"] == 3 and cell["avg_vph"] == 90.0


def test_best_times_ctr_weighted_only_when_complete(db):
    # Every qualifying hour bucket has >=3 real CTR → rank by CTR, not vph.
    slotA = datetime(2026, 1, 6, 5, 30, tzinfo=timezone.utc)   # IST 11:00
    slotB = datetime(2026, 1, 6, 9, 30, tzinfo=timezone.utc)   # IST 15:00
    # slotB has lower vph but higher CTR — CTR must win when complete.
    for i in range(3):
        _sample_at(db, i, 50.0, slotA + timedelta(days=i), ctr=0.02)
    for i in range(3, 6):
        _sample_at(db, i, 10.0, slotB + timedelta(days=i), ctr=0.09)
    rep = sl.best_times_report(db, 9, 3650)
    assert rep["signal"] == "ctr"
    assert rep["recommendation"]["best_hours"][0] == 15   # CTR winner, not vph


def test_weekly_uplift_needs_two_measured_weeks(db):
    now = datetime.now(timezone.utc)
    # Two videos in the SAME (current) week → only one measured week.
    _sample_at(db, 0, 10.0, now)
    _sample_at(db, 1, 12.0, now - timedelta(hours=2))
    rep = sl.weekly_uplift(db, 9, 8)
    assert rep["enough_data"] is False
    assert rep["latest_wow_pct"] is None
    assert len(rep["series"]) == 8          # empty weeks still emitted


def test_weekly_uplift_wow_skips_empty_weeks(db):
    now = datetime.now(timezone.utc)
    # A measured week 2 weeks ago (vph 10) + this week (vph 20); last week is
    # empty and must be skipped, not read as a crash-then-spike.
    _sample_at(db, 0, 10.0, now - timedelta(days=14))
    _sample_at(db, 1, 20.0, now)
    rep = sl.weekly_uplift(db, 9, 8)
    assert rep["enough_data"] is True
    assert rep["latest_wow_pct"] == 100.0   # 10 → 20 vs the prior NON-EMPTY week


def test_resolve_script_policy_env_and_learned(db, monkeypatch):
    # No channel, no env override → operator's bilingual default.
    monkeypatch.delenv("KAIZER_SEO_SCRIPT_POLICY", raising=False)
    assert sl.resolve_script_policy(db) == "bilingual"
    # Env hard-force wins over any channel learning.
    monkeypatch.setenv("KAIZER_SEO_SCRIPT_POLICY", "english")
    assert sl.resolve_script_policy(db, 9) == "english"
    # 'learned' → follow the channel's measured best_script (mixed→bilingual).
    monkeypatch.setenv("KAIZER_SEO_SCRIPT_POLICY", "learned")
    db.add(models.SeoLearningSnapshot(
        channel_id=9, kind="own", window_days=30, samples=10,
        payload={"policy": {"best_script": "native"}}))
    db.commit()
    assert sl.resolve_script_policy(db, 9) == "native"
