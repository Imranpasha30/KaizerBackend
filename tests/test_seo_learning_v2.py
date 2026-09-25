"""seo_learning v2: brand-filtered mining, topics, upload times,
exploration, dedupe guard, per-channel script policy in the verifier."""
import random
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import models
from learning import seo_learning as sl
from seo.verifier import _score_title, verify


# ── brand/stop filtering (the "kaizer, news on every card" bug) ─────

def test_title_terms_filter_brand_and_stopwords():
    terms = sl._title_terms(
        "Kaizer News Telugu | Pawan Kalyan Land Dispute Breaking",
        brand_tokens={"kaizer"})
    assert "kaizer" not in terms and "news" not in terms
    assert "telugu" not in terms and "breaking" not in terms
    assert "pawan" in terms and "pawan kalyan" in terms
    assert "land dispute" in terms


# ── exploration ─────────────────────────────────────────────────────

def test_pick_hook_exploits_by_default():
    pol = {"best_hooks": ["power", "plain"]}
    rng = random.Random(42)
    picks = [sl.pick_hook(pol, explore_rate=0.0, rng=rng) for _ in range(10)]
    assert all(p == ("power", False) for p in picks)


def test_pick_hook_explores_at_rate():
    pol = {"best_hooks": ["power", "plain"]}
    rng = random.Random(7)
    picks = [sl.pick_hook(pol, explore_rate=0.5, rng=rng) for _ in range(200)]
    explored = [h for h, e in picks if e]
    assert explored, "exploration never fired at 50% rate"
    # explorations never pick a favored hook
    assert all(h not in ("power", "plain") for h in explored)


def test_pick_hook_no_policy_is_noop():
    assert sl.pick_hook(None) == (None, False)
    assert sl.pick_hook({"best_hooks": []}) == (None, False)


# ── dedupe guard ────────────────────────────────────────────────────

def test_duplicate_title_exact_and_near():
    recent = ["Breaking: వికారాబాద్ అభ్యర్థి నామినేషన్, ప్రజల ఆశీస్సులు!"]
    assert sl.is_duplicate_title(
        "breaking: వికారాబాద్ అభ్యర్థి నామినేషన్, ప్రజల ఆశీస్సులు!", recent)
    assert sl.is_duplicate_title(   # near-identical (punctuation drift)
        "Breaking: వికారాబాద్ అభ్యర్థి నామినేషన్ ప్రజల ఆశీస్సులు", recent)
    assert not sl.is_duplicate_title(
        "KTR విమర్శలపై మంత్రి స్పందన — పూర్తి వివరాలు", recent)
    assert not sl.is_duplicate_title("", recent)


# ── per-channel script policy in the verifier ──────────────────────

EN = "Breaking: Pawan Kalyan Land Dispute Sparks Huge Political Row Today!"
TE = "షాకింగ్: పవన్ కళ్యాణ్ భూ వివాదంపై సంచలన ఆరోపణలు వెల్లడి అయ్యాయి!"
MIX = "Breaking: పవన్ కళ్యాణ్ భూ వివాదం — Pawan Kalyan Land Row Explained!"


def test_script_policy_english_channel():
    pts_en, fails_en = _score_title(EN, script_policy="english")
    pts_mix, _ = _score_title(MIX, script_policy="english")
    assert not any("ENGLISH-leaning" in f for f in fails_en)
    assert pts_en > pts_mix - 2          # english wins or ties on this policy


def test_script_policy_native_channel():
    _, fails = _score_title(EN, script_policy="native")
    assert any("NATIVE-script" in f for f in fails)


def test_script_policy_default_is_bilingual():
    _, fails = _score_title(EN)          # no policy arg
    assert any("BILINGUAL" in f for f in fails)


def test_verify_threads_script_policy():
    seo = {"title": EN, "description": "x" * 800, "hook": "y",
           "keywords": ["a"] * 29, "hashtags": ["#TopicOne"] * 10}
    r_en = verify(seo, script_policy="english")
    r_bi = verify(seo, script_policy="bilingual")
    assert r_en["breakdown"]["title"] >= r_bi["breakdown"]["title"]


# ── first-125 description check ─────────────────────────────────────

def test_desc_first125_repeating_title_fails():
    from seo.verifier import _score_description
    title = "Breaking: Pawan Kalyan Land Dispute"
    desc = (title + "\n\n" + "para two " * 30 + "\n\n" + "para three " * 30)
    _, fails = _score_description(desc, "hook line", title=title)
    assert any("first 125 chars" in f for f in fails)


def test_desc_fresh_opening_passes():
    from seo.verifier import _score_description
    title = "Breaking: Pawan Kalyan Land Dispute"
    desc = ("Irrigation records name a 10-acre tank bed parcel; officials "
            "confirm a probe request.\n\n" + "para two " * 30 + "\n\n"
            + "para three " * 30)
    _, fails = _score_description(desc, "hook", title=title)
    assert not any("first 125 chars" in f for f in fails)


# ── extended policy fields over a real (SQLite) store ──────────────

@pytest.fixture()
def db():
    eng = create_engine("sqlite://")
    models.Base.metadata.create_all(
        eng, tables=[models.TrainingSample.__table__,
                     models.SeoLearningSnapshot.__table__])
    s = sessionmaker(bind=eng)()
    yield s
    s.close()


def test_policy_carries_topics_and_hours(db):
    for i in range(8):
        db.add(models.TrainingSample(
            video_id=f"v{i}", channel_id=9,
            seo_title=f"Court verdict updates today? తీర్పు వివరాలు {i}",
            seo_keywords=["court verdict"], views_per_hour=10.0 + i,
            first_seen_at=datetime(2026, 8, 10, 14, 30,
                                   tzinfo=timezone.utc)))
    db.commit()
    out = sl.compute_channel_learning(db, 9, 30)
    assert out["policy"] is not None
    assert "court verdict" in out["policy"]["top_keywords"]
    assert out["policy"]["top_topics"]          # multi-word terms learned
    assert out["policy"]["best_hours"]          # upload-hour learned (IST)
    assert out["by_hour"] and out["by_dow"]
