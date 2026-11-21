"""The band waits one phrase when two keys are about level at its entry."""
from plume.bandentry import DRUMS, ENTRY_KEY_MARGIN, BandEntry
from plume.listen.harmony import HarmonyState


def h(key, margin):
    return HarmonyState(t_s=0.0, key=key, key_conf=0.8, chord=None, chord_conf=0.0,
                        register_low_midi=None, register_high_midi=None, key_margin=margin)


def entry(keys, margins):
    """(entry bar, entry key) for a detect-mode BandEntry fed these per-bar readings."""
    e = BandEntry("lofi", key_bars=4, after_first_heard_bars=6, phrase_bars=4)
    for bar, (k, m) in enumerate(zip(keys, margins)):
        s = e.update(bar, h(k, m), True)
        if s.stage != DRUMS:
            return bar, s.key
    return None, None


def test_a_clear_key_enters_where_it_always_did():
    assert entry(["A minor"] * 30, [0.3] * 30) == (8, "A minor")
    # a state with no margin (the machine's own, or older callers) is a sure one
    e = BandEntry("lofi", key_bars=4, after_first_heard_bars=6, phrase_bars=4)
    sure = HarmonyState(t_s=0.0, key="A minor", key_conf=0.8, chord=None, chord_conf=0.0,
                        register_low_midi=None, register_high_midi=None)
    assert [e.update(b, sure, True).stage != DRUMS for b in range(9)].index(True) == 8


def test_an_unsure_key_waits_one_phrase_and_only_one():
    # About level the whole take: one phrase later, then in regardless.
    assert entry(["A minor"] * 30, [ENTRY_KEY_MARGIN / 2] * 30) == (12, "A minor")


def test_after_the_wait_the_band_comes_in_on_what_the_tracker_then_holds():
    # The tracker opened on C major, was torn at the entry bar, and had moved to
    # A minor two bars later: the band comes in on A minor, not on the first guess.
    keys = ["C major"] * 9 + ["A minor"] * 21
    margins = [0.3] * 7 + [0.01] * 3 + [0.2] * 20
    assert entry(keys, margins) == (12, "A minor")


def test_the_tracker_says_how_far_its_key_is_ahead():
    import numpy as np
    from types import SimpleNamespace

    from plume.listen.harmony import HarmonyTracker
    c_major = np.array([1, 0, .5, 0, .8, .4, 0, .9, 0, .4, 0, .3])
    tr = HarmonyTracker()
    for i in range(400):
        st = tr.process([SimpleNamespace(t_s=i * 0.0116, chroma=c_major, chroma_low=c_major * .2,
                                         rms_db=-20.0, centroid_hz=800.0)])
    assert st.key == "C major"
    assert 0.0 < st.key_margin < 1.0
