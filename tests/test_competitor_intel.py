"""Competitor intelligence + campaign wave: topic-matched retrieval,
learn-from-rival math, prompt block, coverage scoring, exploration ledger,
CTR-weighted buckets, thumbnail coherence. All offline (SQLite + shims)."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import models
from learning import competitor_intel as ci
from learning import seo_learning as sl


@pytest.fixture()
def db():
    eng = create_engine("sqlite://")
    models.Base.metadata.create_all(
        eng, tables=[models.TrainingSample.__table__,
                     models.SeoLearningSnapshot.__table__,
                     models.CompetitorChannel.__table__,
                     models.ChannelVideo.__table__])
    s = sessionmaker(bind=eng)()
    yield s
    s.close()


def _comp(db, uid=2, gcid="UC_rival1", name="TV9 Telugu"):
    c = models.CompetitorChannel(user_id=uid, name=name, handle="@tv9",
                                 youtube_channel_id=gcid, active=True)
    db.add(c)
    db.commit()
    return c


def _vid(db, gcid, vid, title, views, days_old=2, tags=None, uid=2):
    db.add(models.ChannelVideo(
        user_id=uid, google_channel_id=gcid, video_id=vid, title=title,
        view_count=views, tags=tags,
        published_at=datetime.now(timezone.utc) - timedelta(days=days_old)))
    db.commit()


# ── topic_intel: the per-video conquest retrieval ───────────────────

def test_topic_intel_matches_and_ranks(db):
    c = _comp(db)
    _vid(db, c.youtube_channel_id, "a1",
         "Pawan Kalyan Land Case Big Twist", 200000,
         tags=["pawan kalyan", "land case", "telangana"])
    _vid(db, c.youtube_channel_id, "a2",
         "RCB Match Highlights Today", 900000, tags=["rcb", "ipl"])
    out = ci.topic_intel(db, 2, ["Pawan Kalyan", "land"])
    assert out is not None
    titles = [r["title"] for r in out["rivals"]]
    assert any("Pawan Kalyan" in t for t in titles)
    assert all("RCB" not in t for t in titles)      # unmatched topic excluded
    assert "pawan kalyan" in out["harvest_tags"]
    assert out["rival_titles"]


def test_expand_terms_cross_script():
    out = ci._expand_terms(["Pawan Kalyan"])
    assert "Pawan Kalyan" in out
    assert "పవన్ కళ్యాణ్" in out          # Latin term → native variant added
    # Reverse: a native clip term expands to its English form.
    rev = ci._expand_terms(["తెలంగాణ"])
    assert any(v.lower() == "telangana" for v in rev)
    # Unknown terms pass through untouched, order preserved.
    assert ci._expand_terms(["random topic"]) == ["random topic"]


def test_topic_intel_matches_across_script(db):
    # Rival titles are pure Telugu; the clip's term is Latin. Without the
    # transliteration expansion this returns None (the old blind spot).
    c = _comp(db)
    _vid(db, c.youtube_channel_id, "b1",
         "పవన్ కళ్యాణ్ భూ వివాదం సంచలనం", 300000, tags=["land"])
    _vid(db, c.youtube_channel_id, "b2",
         "క్రికెట్ మ్యాచ్ హైలైట్స్", 500000, tags=["cricket"])
    out = ci.topic_intel(db, 2, ["Pawan Kalyan"])
    assert out is not None
    assert any("పవన్" in r["title"] for r in out["rivals"])
    assert all("క్రికెట్" not in r["title"] for r in out["rivals"])  # off-topic excluded


def test_topic_intel_escapes_like_wildcards(db):
    # An underscore in a clip term must be matched LITERALLY, not as the SQL
    # LIKE "any single char" wildcard (which would over-broaden the match).
    c = _comp(db)
    _vid(db, c.youtube_channel_id, "w1", "Big 50_off Sale Live", 100000)
    _vid(db, c.youtube_channel_id, "w2", "Big 50Xoff Sale Live", 500000)
    out = ci.topic_intel(db, 2, ["50_off"])
    assert out is not None
    titles = [r["title"] for r in out["rivals"]]
    assert any("50_off" in t for t in titles)
    assert all("50Xoff" not in t for t in titles)   # underscore not a wildcard


def test_topic_intel_none_without_competitors_or_terms(db):
    assert ci.topic_intel(db, 2, ["pawan"]) is None      # no rivals tracked
    _comp(db)
    assert ci.topic_intel(db, 2, []) is None             # no terms


# ── learn_competitor: same measured math, competitor namespace ──────

def test_learn_competitor_snapshot_and_policy(db):
    c = _comp(db)
    for i in range(6):
        _vid(db, c.youtube_channel_id, f"v{i}",
             f"10 Facts? నిజాలు ఇవే {i}", 50000 + i, tags=["facts"])
    payload = ci.learn_competitor(db, c)
    assert payload["samples"] == 6
    assert payload["policy"] is not None
    assert payload["top_tags"][0]["term"] == "facts"
    # stored under the competitor namespace — own-policy reads must miss it
    assert ci.latest_competitor_payload(db, c.id) is not None
    assert sl.latest_policy(db, c.id) is None


# ── prompt block + verifier coverage ────────────────────────────────

def test_competitor_block_renders():
    from seo.prompts import build_user_prompt

    class _Clip:
        meta = "{}"
        text = "టెస్ట్"
        sentiment = ""
        duration = 30.0

    p = build_user_prompt(clip=_Clip(), language="te", competitor={
        "rivals": [{"channel": "TV9", "title": "Their Winner", "vph": 42.0}],
        "harvest_tags": ["land case"], "cover_terms": ["high court"],
        "rival_titles": ["Their Winner"]})
    assert "MARKET INTELLIGENCE" in p
    assert "DIFFERENTIATE" in p and "high court" in p and "land case" in p


def test_verifier_competitor_coverage():
    from seo.verifier import verify
    seo = {"title": "Breaking: భూ వివాదం high court తీర్పు — Land Case!",
           "description": "x" * 800, "hook": "y",
           "keywords": ["land case"] + ["k"] * 28,
           "hashtags": ["#TopicOne"] * 10}
    r = verify(seo, competitor_terms=["high court", "land case", "verdict"])
    assert not any("competitor intel" in f for f in r["reasons"])
    seo2 = dict(seo, title="Breaking: వేరే విషయం పూర్తిగా — Other Topic!",
                keywords=["k"] * 29)
    r2 = verify(seo2, competitor_terms=["high court", "land case", "verdict"])
    assert any("competitor intel" in f for f in r2["reasons"])


def test_scoring_without_intel_unchanged():
    from seo.verifier import verify
    seo = {"title": "Breaking: భూ వివాదంపై తీర్పు — Court Verdict Today!",
           "description": "x" * 800, "hook": "y",
           "keywords": ["k"] * 29, "hashtags": ["#TopicOne"] * 10}
    assert (verify(seo)["score"]
            == verify(seo, competitor_terms=None)["score"])


# ── thumbnail–title coherence advisory ──────────────────────────────

def test_thumbnail_title_mismatch_flagged():
    from seo.verifier import verify
    seo = {"title": "Breaking: పవన్ కళ్యాణ్ భూ వివాదం Land Case!",
           "description": "x" * 800, "hook": "y",
           "keywords": ["k"] * 29, "hashtags": ["#TopicOne"] * 10,
           "thumbnail_text": "CRICKET SCORE"}
    r = verify(seo)
    assert any("thumbnail_text" in f for f in r["reasons"])


# ── CTR-weighted bucket ranking ─────────────────────────────────────

def test_best_bucket_prefers_ctr_when_measured():
    stats = {
        "power": {"n": 5, "avg_vph": 50.0, "avg_ctr": 0.02, "ctr_n": 4},
        "question": {"n": 5, "avg_vph": 30.0, "avg_ctr": 0.08, "ctr_n": 4},
    }
    assert sl._best_bucket(stats) == "question"   # click-truth outranks reach
    stats["question"]["ctr_n"] = 0                # CTR not measured → velocity
    assert sl._best_bucket(stats) == "power"


# ── exploration ledger in the payload ───────────────────────────────

def test_explorations_aggregated(db):
    for i in range(4):
        db.add(models.TrainingSample(
            video_id=f"e{i}", channel_id=3, seo_title=f"probe {i}?",
            views_per_hour=9.0, explored_hook="question",
            first_seen_at=datetime.now(timezone.utc)))
    for i in range(4):
        db.add(models.TrainingSample(
            video_id=f"b{i}", channel_id=3, seo_title=f"షాకింగ్ base {i}",
            views_per_hour=5.0, first_seen_at=datetime.now(timezone.utc)))
    db.commit()
    out = sl.compute_channel_learning(db, 3, 30)
    exp = out["explorations"]
    assert exp["n"] == 4
    assert exp["by_hook"]["question"]["avg_vph"] > exp["baseline_avg_vph"]
