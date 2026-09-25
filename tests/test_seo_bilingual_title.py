"""Bilingual title contract (operator mandate 2026-08): titles mix English
searchable terms + native-script core. Single-script titles lose points and
get a rewrite-directive fail so the retry loop fixes them."""
from seo.verifier import _score_title


BILINGUAL = "Breaking: పవన్ కళ్యాణ్ భూ వివాదంపై సంచలన ఆరోపణలు — Pawan Kalyan Land Row!"
ENGLISH_ONLY = "Breaking: Pawan Kalyan Land Controversy Sparks Huge Political Row!"
TELUGU_ONLY = "షాకింగ్: పవన్ కళ్యాణ్ భూ వివాదంపై సంచలన ఆరోపణలు వెల్లడి అయ్యాయి!"


def test_bilingual_title_gets_full_script_points():
    pts, fails = _score_title(BILINGUAL)
    assert not any("BILINGUAL" in f for f in fails)   # no script complaint
    assert pts >= 16                                   # len+power+nosuffix+script+hook


def test_english_only_title_penalized_with_directive():
    pts, fails = _score_title(ENGLISH_ONLY)
    assert any("English only" in f and "BILINGUAL" in f for f in fails)


def test_native_only_title_penalized_with_directive():
    pts, fails = _score_title(TELUGU_ONLY)
    assert any("native-script only" in f for f in fails)


def test_native_power_word_counts():
    # షాకింగ్ is in the power-word list — native-script hooks qualify.
    _, fails = _score_title(TELUGU_ONLY)
    assert not any("POWER WORD" in f for f in fails)
