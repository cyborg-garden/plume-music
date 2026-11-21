"""A style from how she plays (plume.playstyle): the rhythm ear, the scored rule, and
one player of each bundled style who gets it.

Unit tests on stroke times and on feature values, and a few short synthetic players
through the real feature extractor.  No take, no device.
"""
from __future__ import annotations

import numpy as np
import pytest

from plume.bandentry import StylePick, detect_style
from plume.playstyle import (MARGIN, PROFILES, SOFT, Playing, RhythmEar, explain,
                             score_styles, tempo_default, tempo_range)

SR = 48000


# ------------------------------------------------------------- the rhythm ear

def _rhythm(places, *, bpm=100.0, bars=7, late_s=0.0, tempo_off=0.0, jitter_s=0.004,
            seed=2):
    """A RhythmEar fed ``bars`` bars of strokes at ``places`` (beats into the bar).

    ``tempo_off``: she holds her own tempo, this share faster than the bar lines she is
    read against.  ``late_s``: every stroke is this late."""
    from plume.playstyle import STROKE_LAG_S
    rng = np.random.default_rng(seed)
    bar_s = 240.0 / bpm
    hers = bar_s / (1.0 + tempo_off)
    ear = RhythmEar()
    for b in range(bars):
        at = [b * hers + p * hers / 4.0 + late_s + STROKE_LAG_S
              + float(rng.normal(0.0, jitter_s)) for p in places]
        ear.bar(at, bar_start=b * bar_s, bar_s=bar_s)
    return ear


_EIGHTHS = tuple(i / 2.0 for i in range(8))
_QUARTERS = (0.0, 1.0, 2.0, 3.0)
_STABS = (0.5, 1.5, 2.5, 3.5)


def _swung(swing):
    return tuple(b + s for b in range(4) for s in (0.0, swing))


def _each_beat(*within):
    return tuple(b + w for b in range(4) for w in within)


def test_straight_eighths_read_straight_and_half_off_the_beat():
    ear = _rhythm(_EIGHTHS)
    assert ear.swing == pytest.approx(0.5, abs=0.02)
    assert ear.offbeat == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("swing", [0.58, 0.66])
@pytest.mark.parametrize("bpm", [80.0, 140.0, 200.0])
def test_swung_eighths_read_as_far_as_they_swing(swing, bpm):
    ear = _rhythm(_swung(swing), bpm=bpm)
    assert ear.swing == pytest.approx(swing, abs=0.025)
    assert ear.offbeat == pytest.approx(0.5, abs=0.02)


def test_strokes_on_the_beat_alone_say_nothing_of_swing():
    ear = _rhythm(_QUARTERS)
    assert ear.swing is None
    assert ear.offbeat == pytest.approx(0.0, abs=0.02)


def test_stabs_between_the_beats_are_all_off_it():
    ear = _rhythm(_STABS)
    assert ear.offbeat == pytest.approx(1.0, abs=0.02)
    assert ear.swing == pytest.approx(0.5, abs=0.02)


def test_sixteenths_are_read_by_their_eighths():
    """The last sixteenth of the beat is not a late eighth.  Three of her four strokes
    are off the beat."""
    ear = _rhythm(tuple(i / 4.0 for i in range(16)))
    assert ear.swing == pytest.approx(0.5, abs=0.02)
    assert ear.offbeat == pytest.approx(0.75, abs=0.03)


@pytest.mark.parametrize("bpm", [84.0, 120.0, 160.0])
@pytest.mark.parametrize("jitter_ms", [4.0, 12.0, 20.0])
def test_a_dotted_figure_is_not_swing(bpm, jitter_ms):
    """On the beat and on the last sixteenth of it.  A human hand spreads those
    sixteenths by a few hundredths of a beat, and the early ones land where a hard
    swing's eighths do: they are still sixteenths, and say nothing of her eighths."""
    for seed in (2, 3):
        ear = _rhythm(_each_beat(0.0, 0.75), bpm=bpm, jitter_s=jitter_ms / 1000.0, seed=seed)
        assert ear.swing is None, (seed, ear.swing)
        assert ear.offbeat == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize("jitter_ms", [4.0, 16.0])
def test_a_gallop_s_eighths_are_straight(jitter_ms):
    """On the beat, the eighth and the last sixteenth: the eighths are heard, and the
    sixteenths after them do not pull them late."""
    ear = _rhythm(_each_beat(0.0, 0.5, 0.75), bpm=140.0, jitter_s=jitter_ms / 1000.0)
    assert ear.swing == pytest.approx(0.5, abs=0.04)


def test_a_hard_swing_played_loosely_is_still_swing():
    ear = _rhythm(_swung(0.66), bpm=150.0, jitter_s=0.016)
    assert ear.swing == pytest.approx(0.66, abs=0.04)


def test_the_eighths_window_ends_where_the_late_sixteenths_begin():
    """Half way from a triplet swing to the last sixteenth, with nothing between the two
    windows and no bin in both."""
    from plume.playstyle import _AND_BINS, _LATE_BINS, PHASE_BINS
    assert _AND_BINS[1] == _LATE_BINS[0]
    assert _AND_BINS[1] / PHASE_BINS == pytest.approx((2 / 3 + 0.75) / 2, abs=0.5 / PHASE_BINS)


def test_triplets_are_not_swing():
    ear = _rhythm(_each_beat(0.0, 1 / 3, 2 / 3))
    assert ear.swing is None


def test_a_few_off_beat_eighths_say_nothing_of_swing():
    """Five in seven bars could be anything; one a bar is enough to hear."""
    few = _rhythm(_QUARTERS, bars=7)
    for b in range(5):
        few.bar([b * 2.4 + 0.66 * 0.6 + 0.0213], bar_start=b * 2.4, bar_s=2.4)
    assert few.swing is None
    enough = _rhythm((0.0, 1.0, 1.66, 2.0, 3.0), bars=7)
    assert enough.swing == pytest.approx(0.66, abs=0.03)


def test_nothing_heard_is_nothing_known():
    ear = RhythmEar()
    ear.bar([], bar_start=0.0, bar_s=2.4)
    assert ear.swing is None and ear.offbeat is None and ear.steady is None


@pytest.mark.parametrize("tempo_off", [-0.009, 0.009])
@pytest.mark.parametrize("places,offbeat,swing", [
    (_EIGHTHS, 0.5, 0.5),
    (_STABS, 1.0, 0.5),
    (_QUARTERS, 0.0, None),
    (_swung(0.66), 0.5, 0.66),
])
def test_a_player_who_holds_her_own_tempo_is_still_placed_in_the_beat(tempo_off, places,
                                                                     offbeat, swing):
    """A tempo taken from four knocks is off by up to a percent, and nine bars at that
    slide her a third of a beat against the bar lines: read against them alone, her
    strokes on the beat end up off it."""
    ear = _rhythm(places, bars=9, tempo_off=tempo_off)
    assert ear.offbeat == pytest.approx(offbeat, abs=0.06)
    if swing is None:
        assert ear.swing is None
    else:
        assert ear.swing == pytest.approx(swing, abs=0.03)


def test_an_input_that_runs_a_little_late_changes_nothing():
    for late_s in (-0.02, 0.025):
        ear = _rhythm(_swung(0.66), late_s=late_s)
        assert ear.swing == pytest.approx(0.66, abs=0.03), late_s
        assert ear.offbeat == pytest.approx(0.5, abs=0.03), late_s


@pytest.mark.parametrize("places,bpm,offbeat,swing", [
    (_EIGHTHS, 140.0, 0.5, 0.5),
    (_QUARTERS, 170.0, 0.0, None),
    (_STABS, 124.0, 1.0, 0.5),
    (_swung(0.66), 150.0, 0.5, 0.66),
    (tuple(i / 4.0 for i in range(16)), 100.0, 0.75, 0.5),
])
def test_an_input_that_runs_70_ms_late_is_still_placed_in_the_beat(places, bpm, offbeat,
                                                                  swing):
    """A slow output, or a player behind the click: 70 ms is a fifth of a beat at 170
    bpm, past the eighth of a beat that finds her at once, and straight eighths that
    late sit where a swing's would."""
    ear = _rhythm(places, bpm=bpm, late_s=0.070)
    assert ear.offbeat == pytest.approx(offbeat, abs=0.05)
    if swing is None:
        assert ear.swing is None
    else:
        assert ear.swing == pytest.approx(swing, abs=0.03)


def test_one_stray_stroke_far_from_the_beat_does_not_place_her():
    """A pickup on the last sixteenth before she starts: a late beat needs two strokes
    that agree, so the bar after it finds her where she is."""
    ear = RhythmEar()
    ear.bar([3.75 * 0.6 + 0.0213], bar_start=0.0, bar_s=2.4)
    for b in range(1, 8):
        ear.bar([b * 2.4 + p * 0.6 + 0.0213 for p in _QUARTERS], bar_start=b * 2.4, bar_s=2.4)
    assert ear.offbeat == pytest.approx(1 / 29, abs=0.01)


def test_an_even_pulse_is_steady_and_long_and_short_in_turn_are_not():
    assert _rhythm(_EIGHTHS).steady == pytest.approx(1.0, abs=0.03)
    assert _rhythm(_STABS).steady == pytest.approx(1.0, abs=0.03)
    assert _rhythm(_swung(0.66)).steady == pytest.approx(0.5, abs=0.05)
    # down, down-up, up-down-up: four of the six gaps are an eighth
    assert _rhythm((0.0, 1.0, 1.5, 2.5, 3.0, 3.5)).steady == pytest.approx(4 / 6, abs=0.05)
    assert _rhythm((0.0, 1.5, 2.0, 3.5)).steady == pytest.approx(0.5, abs=0.05)
    assert _rhythm(_QUARTERS, bars=2).steady is None        # seven gaps: too few


def test_the_rhythm_ear_keeps_a_bounded_memory():
    ear = _rhythm(_EIGHTHS, bars=400)
    assert ear.memory_items() < 100


# ------------------------------------------------------------------- the rule

#: A plausible player of each style, as the five things heard of her, and a tempo in the
#: style's own range.
_PLAYERS = {
    "midwest": (140.0, Playing(strokes=2.4, swing=0.50, offbeat=0.6, steady=0.6, strum=0.05)),
    "lofi": (78.0, Playing(strokes=0.6, swing=0.57, offbeat=0.4, steady=0.4, strum=None)),
    "folk": (100.0, Playing(strokes=1.4, swing=0.50, offbeat=0.45, steady=0.67, strum=0.45)),
    "synthwave": (96.0, Playing(strokes=2.0, swing=0.50, offbeat=0.5, steady=1.0, strum=0.02)),
    "hiphop": (94.0, Playing(strokes=0.9, swing=0.59, offbeat=0.75, steady=0.4, strum=None)),
    "rnb": (66.0, Playing(strokes=0.8, swing=0.58, offbeat=0.75, steady=0.4, strum=0.0)),
    "house": (124.0, Playing(strokes=1.0, swing=0.50, offbeat=1.0, steady=1.0, strum=0.03)),
    "trap": (140.0, Playing(strokes=0.5, swing=None, offbeat=0.45, steady=0.5, strum=None)),
    "jazz": (150.0, Playing(strokes=1.3, swing=0.65, offbeat=0.5, steady=0.5, strum=0.0)),
}


def test_every_bundled_style_has_a_profile_and_a_player():
    from plume.styles_lib import list_styles
    assert set(PROFILES) == set(list_styles()) == set(_PLAYERS)


@pytest.mark.parametrize("style", sorted(_PLAYERS))
@pytest.mark.parametrize("mode", ["major", "minor", None])
def test_a_player_of_each_style_gets_it_in_any_mode(style, mode):
    """On her playing: no style scores better, whatever the key says the tempo's
    default is."""
    bpm, playing = _PLAYERS[style]
    lo, hi = tempo_range(style)
    assert lo <= bpm <= hi
    scores = score_styles(bpm, mode, playing)
    assert scores[0].style == style, explain(scores)
    assert scores[0].score == max(s.score for s in scores) == 0.0, explain(scores)


def test_no_player_s_style_is_only_her_tempo_s_default():
    """But trap's, whose whole range is the tempo band that defaults to it."""
    for style, (bpm, _) in _PLAYERS.items():
        if style != "trap":
            assert any(tempo_default(bpm, mode) != style for mode in ("major", "minor")), style


def test_the_same_tempo_gets_the_style_she_plays():
    """85 bpm is inside seven styles' ranges, and used to be lofi or midwest."""
    got = {name: detect_style(85.0, "major", playing=p) for name, p in {
        "busy, on and off the beat": Playing(strokes=2.6, swing=0.5, offbeat=0.6, steady=0.6),
        "a busy even pulse": Playing(strokes=2.0, swing=0.5, offbeat=0.5, steady=1.0),
        "strummed": Playing(strokes=1.4, swing=0.5, offbeat=0.45, steady=0.67, strum=0.5),
        "sparse, uneven": Playing(strokes=0.5, swing=0.57, offbeat=0.4, steady=0.4),
        "off the beat with room": Playing(strokes=0.9, swing=0.59, offbeat=0.75, steady=0.4),
        "hard swing": Playing(strokes=1.3, swing=0.66, offbeat=0.5, steady=0.5),
    }.items()}
    assert got == {"busy, on and off the beat": "midwest", "a busy even pulse": "synthwave",
                   "strummed": "folk", "sparse, uneven": "lofi",
                   "off the beat with room": "hiphop", "hard swing": "jazz"}


def test_midwest_is_not_the_answer_for_most_players():
    for name, (bpm, playing) in _PLAYERS.items():
        if name != "midwest":
            assert detect_style(bpm, "major", playing=playing) != "midwest", name


@pytest.mark.parametrize("bpm,mode,style", [
    (50.0, None, "lofi"), (50.0, "minor", "lofi"), (60.0, "minor", "rnb"),
    (79.9, None, "lofi"), (79.9, "minor", "rnb"), (80.0, "major", "lofi"),
    (85.0, "minor", "hiphop"), (90.0, None, "midwest"), (95.0, "minor", "hiphop"),
    (100.0, "major", "midwest"), (100.0, "minor", "synthwave"), (117.9, "minor", "synthwave"),
    (118.0, "minor", "house"), (124.0, None, "midwest"), (129.9, "minor", "house"),
    (130.0, None, "trap"), (150.0, "minor", "trap"), (150.1, None, "midwest"),
    (220.0, "minor", "midwest"), (260.0, None, "midwest"),
])
def test_with_nothing_heard_the_tempo_answers(bpm, mode, style):
    assert tempo_default(bpm, mode) == style
    assert detect_style(bpm, mode) == style
    assert score_styles(bpm, mode)[0].score == 0.0


def test_a_near_tie_goes_to_the_tempo_s_default():
    """Busy and straight, in gaps a shade more even than a picked pattern's: synthwave
    fits it whole and midwest but for that shade.  What was heard does not tell them
    apart, so the tempo does."""
    playing = Playing(strokes=2.0, swing=0.5, offbeat=0.5, steady=0.82)
    scores = score_styles(110.0, "major", playing)
    assert tempo_default(110.0, "major") == "midwest"
    assert [s.style for s in scores[:2]] == ["midwest", "synthwave"], explain(scores)
    assert scores[1].score - MARGIN <= scores[0].score < scores[1].score
    assert explain(scores).endswith("a near-tie")
    # further out it is no longer a tie, and the style that fits takes it
    clearly = Playing(strokes=2.0, swing=0.5, offbeat=0.5, steady=1.0)
    assert detect_style(110.0, "major", playing=clearly) == "synthwave"
    assert not explain(score_styles(110.0, "major", clearly)).endswith("a near-tie")


def test_a_near_tie_without_the_default_goes_by_the_order_of_the_styles():
    """Off the beat with room at 85 is hip hop's and r&b's alike, and not lofi's, the
    tempo's default in a major key."""
    playing = Playing(strokes=0.9, swing=0.59, offbeat=0.85, steady=0.4, strum=0.0)
    scores = score_styles(85.0, "major", playing)
    assert [s.style for s in scores[:2]] == ["hiphop", "rnb"], explain(scores)
    assert scores[0].score == scores[1].score == 0.0
    order = list(PROFILES)
    assert order.index("hiphop") < order.index("rnb")


def test_when_nothing_fits_the_tempo_answers():
    """A stab between every other pair of beats at 96 is no style in range: each loses
    a whole point somewhere.  Nothing heard chose, so nothing but the tempo does."""
    stabs = Playing(strokes=0.5, swing=0.5, offbeat=1.0, steady=1.0, strum=0.0)
    for mode in ("major", "minor"):
        scores = score_styles(96.0, mode, stabs)
        assert max(s.score for s in scores) <= -1.0
        assert scores[0].style == tempo_default(96.0, mode), explain(scores)
        assert "no style fits" in explain(scores)
    assert detect_style(124.0, "major", playing=stabs) == "house"      # where one does


def test_no_style_answers_outside_its_own_range_but_the_default():
    from plume.styles_lib import list_styles
    for bpm in np.arange(40.0, 260.0, 1.5):
        for mode in (None, "major", "minor"):
            for _, playing in _PLAYERS.values():
                for scored in score_styles(float(bpm), mode, playing):
                    lo, hi = tempo_range(scored.style)
                    assert (lo <= bpm <= hi
                            or scored.style == tempo_default(float(bpm), mode)), scored
    assert set(list_styles()) == set(PROFILES)


def test_jazz_is_its_swing():
    """Fast is not jazz, and nor is leaving room: the swing is."""
    straight = Playing(strokes=1.0, swing=0.5, offbeat=0.5, steady=0.5)
    swung = Playing(strokes=1.0, swing=0.66, offbeat=0.5, steady=0.5)
    for bpm in (96.0, 140.0, 190.0):
        assert detect_style(bpm, "major", playing=straight) != "jazz"
        assert detect_style(bpm, "major", playing=swung) == "jazz"
        assert "jazz" not in [s.style for s in score_styles(bpm, "major", straight)]
    assert detect_style(170.0, "minor", 3.0) == "midwest"      # strokes alone, busy


@pytest.mark.parametrize("swing", [0.57, 0.59, 0.61])
def test_a_light_swing_is_not_jazz(swing):
    """Jazz needs its swing heard inside its band, not merely near it."""
    for bpm in np.arange(70.0, 221.0, 3.0):
        for strokes in (0.6, 1.3, 2.0):
            playing = Playing(strokes=strokes, swing=swing, offbeat=0.5, steady=0.5, strum=0.0)
            styles = [s.style for s in score_styles(float(bpm), "major", playing)]
            assert "jazz" not in styles, (bpm, strokes)
    assert detect_style(78.0, "major", playing=Playing(strokes=2.0, swing=0.59, offbeat=0.5,
                                                       steady=0.5, strum=0.0)) != "jazz"


@pytest.mark.parametrize("bpm", [110.0, 125.0, 170.0, 200.0])
@pytest.mark.parametrize("mode", ["major", "minor"])
def test_with_no_swing_heard_jazz_is_not_even_a_candidate(bpm, mode):
    """Short notes on every beat have no off-beat eighths to hear a swing in.  Jazz
    used to lose one point for that and win whenever every other style lost more.  It
    is plain playing, and fits nothing: the tempo and key answer."""
    quarters = Playing(strokes=1.0, swing=None, offbeat=0.0, steady=1.0, strum=0.0)
    scores = score_styles(bpm, mode, quarters)
    assert "jazz" not in [s.style for s in scores]
    assert scores[0].style == tempo_default(bpm, mode, 1.0), explain(scores)


def test_a_swing_counts_against_the_straight_styles_and_no_others():
    """Hip hop, r&b and lofi are often swung a little and as often not, so neither
    costs them; past a light swing it is jazz."""
    from dataclasses import replace
    for style in ("house", "synthwave", "trap", "midwest", "folk"):
        bpm, playing = _PLAYERS[style]
        lost = {s.style: dict(s.against)
                for s in score_styles(bpm, "major", replace(playing, swing=0.62))}
        assert lost[style].get("swing") == 1.0, (style, lost[style])
    for style in ("hiphop", "rnb", "lofi"):
        bpm, playing = _PLAYERS[style]
        for swing in (0.5, 0.62):
            lost = {s.style: dict(s.against)
                    for s in score_styles(bpm, "major", replace(playing, swing=swing))}
            assert "swing" not in lost[style], (style, swing)
        hard = {s.style: dict(s.against)
                for s in score_styles(bpm, "major", replace(playing, swing=0.69))}
        assert hard.get(style, {"swing": 1.0}).get("swing") == 1.0, style


def test_a_strum_turns_plain_playing_to_folk_in_any_key():
    """Busy, straight, on and off the beat in uneven gaps is midwest's, and folk's too
    for all that was heard.  Strummed, it is folk; picked, midwest.  It used to take an
    acoustic sound or a major key as well, which a guitar through a pickup in a minor
    key never had."""
    def player(strum):
        return Playing(strokes=1.5, swing=0.5, offbeat=0.5, steady=0.67, strum=strum)
    for mode in ("major", "minor", None):
        assert detect_style(96.0, mode, playing=player(0.5)) == "folk"
    for mode in ("major", None):
        assert detect_style(96.0, mode, playing=player(0.02)) == "midwest"
        assert detect_style(96.0, mode, playing=player(None)) == "midwest"
    lost = dict(next(s for s in score_styles(96.0, "major", player(0.5))
                     if s.style == "midwest").against)
    assert lost == {"strum": 1.0}
    assert detect_style(140.0, "major", playing=player(0.5)) != "folk"   # past its range


def test_a_strum_is_not_enough_for_folk():
    """Off the beat it is something else, strummed or not."""
    stabs = Playing(strokes=1.0, swing=0.5, offbeat=1.0, steady=1.0, strum=0.3)
    assert detect_style(124.0, "minor", playing=stabs) == "house"
    swung = Playing(strokes=1.3, swing=0.66, offbeat=0.5, steady=0.5, strum=0.3)
    assert detect_style(100.0, "major", playing=swung) == "jazz"


@pytest.mark.parametrize("strum", [None, 0.0, 0.1, 0.18])
def test_nothing_is_folk_without_a_strum(strum):
    """Plain playing, straight and on the beat, used to be folk wherever it fit
    nothing else.  Folk is a strum; without one it is not even a candidate."""
    for bpm in np.arange(60.0, 133.0, 2.0):
        for mode in ("major", "minor", None):
            for playing in (
                    Playing(strokes=1.0, swing=None, offbeat=0.0, steady=1.0, strum=strum),
                    Playing(strokes=0.5, swing=None, offbeat=0.0, steady=1.0, strum=strum),
                    Playing(strokes=2.0, swing=0.5, offbeat=0.5, steady=1.0, strum=strum),
                    Playing(strokes=1.4, swing=0.5, offbeat=0.45, steady=0.67, strum=strum)):
                styles = [s.style for s in score_styles(float(bpm), mode, playing)]
                assert "folk" not in styles, (bpm, mode, playing)


@pytest.mark.parametrize("bpm,mode,playing,style", [
    (110.0, "minor", Playing(strokes=1.0, swing=None, offbeat=0.0, steady=1.0), "synthwave"),
    (96.0, "major", Playing(strokes=1.0, swing=None, offbeat=0.0, steady=1.0), "midwest"),
    (92.0, "minor", Playing(strokes=1.0, swing=None, offbeat=0.0, steady=1.0), "hiphop"),
    (124.0, "minor", Playing(strokes=1.0, swing=None, offbeat=0.0, steady=1.0), "house"),
    (70.0, "minor", Playing(strokes=0.5, swing=None, offbeat=0.0, steady=1.0), "rnb"),
])
def test_plain_playing_gets_what_the_tempo_and_key_give_it(bpm, mode, playing, style):
    """Quarter-note chords, a bass on the roots, half notes: what was heard tells no
    style, so the answer is the one the count gave."""
    playing = Playing(**{**playing.__dict__, "strum": 0.0})
    scores = score_styles(bpm, mode, playing)
    assert scores[0].style == style == tempo_default(bpm, mode), explain(scores)


@pytest.mark.parametrize("mode", ["major", "minor", None])
@pytest.mark.parametrize("steady", [0.7, 0.85, 1.0])
@pytest.mark.parametrize("strum", [0.0, 0.5])
def test_a_busy_even_pulse_has_a_style_that_fits_at_every_tempo(mode, steady, strum):
    """Straight eighths, picked or strummed.  From 119 to 150 nothing used to take them:
    they were folk to 132 without a strum, and then trap, the style for leaving room."""
    playing = Playing(strokes=2.0, swing=0.5, offbeat=0.5, steady=steady, strum=strum)
    for bpm in np.arange(80.0, 150.5, 1.0):
        scores = score_styles(float(bpm), mode, playing)
        got = scores[0].style
        assert got != "trap", (bpm, explain(scores))
        assert got != "folk" or strum, (bpm, explain(scores))
        if bpm >= 133.0:
            assert got == "midwest", (bpm, explain(scores))
        elif not strum:
            assert scores[0].score > -1.0, (bpm, explain(scores))


def test_too_busy_for_trap_is_midwest_when_nothing_fits():
    """Trap's range is the tempo band that defaults to it, and trap leaves room."""
    assert tempo_default(140.0, "minor") == tempo_default(140.0, "minor", 0.5) == "trap"
    assert tempo_default(140.0, "minor", 2.0) == "midwest"
    odd = Playing(strokes=2.0, swing=0.5, offbeat=1.0, steady=0.3, strum=0.6)
    scores = score_styles(140.0, "minor", odd)
    assert max(s.score for s in scores) <= -1.0
    assert scores[0].style == "midwest", explain(scores)


def test_strokes_alone_choose_only_between_busy_and_sparse():
    """All a MIDI input knows.  Every other style is told by something it cannot hear,
    so none of them is a candidate on it but the tempo's own default."""
    assert detect_style(96.0, "major", 1.5) == "midwest"
    assert detect_style(140.0, "major", 0.4) == "trap"
    assert detect_style(140.0, "major", 2.0) == "midwest"
    for bpm, mode in ((85.0, "major"), (96.0, "minor"), (124.0, "minor")):
        for busy in (0.4, 1.0, 2.0):
            styles = {s.style for s in score_styles(bpm, mode, Playing(strokes=busy))}
            assert styles <= {"trap", "midwest", tempo_default(bpm, mode)}, (bpm, mode, styles)


def test_one_reading_a_little_outside_a_band_costs_a_little():
    near = score_styles(140.0, None, Playing(strokes=1.26))
    far = score_styles(140.0, None, Playing(strokes=0.75))
    cost = {s.style: -s.score for s in near}["midwest"]
    assert cost == pytest.approx(0.1)
    assert {s.style: -s.score for s in far}["midwest"] == 1.0     # never more than 1
    assert near[0].style == "midwest" and far[0].style == "trap"


def test_no_band_edge_sits_on_a_plain_figure():
    """Quarters are one stroke a beat and eighths two; none, half, three in four or all
    of them fall off the beat; an even pulse is all one gap and long-short pairs half.
    A reading moves a little with the input, so an edge on one of those would turn the
    style for the commonest players."""
    plain = {"strokes": (0.5, 1.0, 1.5, 2.0, 4.0), "offbeat": (0.0, 0.5, 0.75, 1.0),
             "steady": (0.5, 1.0), "swing": (0.5, 0.667)}
    for style, profile in PROFILES.items():
        for name, band in profile.items():
            for edge in band:
                for value in plain.get(name, ()):
                    assert edge == value == 0.0 or edge == value == 1.0 or \
                        abs(edge - value) >= 0.2 * SOFT[name], (style, name, edge)


def test_the_scores_say_what_each_style_lost_on():
    bpm, playing = _PLAYERS["house"]
    scores = score_styles(bpm, "major", playing)
    why = explain(scores)
    assert why.startswith("house 0.00 over ")
    assert scores[1].style in why and "against it:" in why
    assert all(cost > 0 for s in scores for _, cost in s.against)
    assert [s.score for s in scores] == sorted((s.score for s in scores), reverse=True)


# --------------------------------------------------------- synthetic players
#
# One short generated part per style through the real feature extractor and the ears,
# read over the seven bars heard before the band comes in.  These are the cheapest
# check that the ears and the rule meet: a part only has to be a plausible player of
# its style, and the bands were not set on these.

_HISS = 3e-4


def _part(places, *, bpm, key, ring_s=1.0, held_s=None, bars=8, chord=False, seed=4,
          gain_db=-6.0, jitter_s=0.004):
    """A picked part: a chord tone (or, ``chord``, three of them at once) at each of
    ``places`` (beats into the bar), ringing with time constant ``ring_s`` and damped
    after ``held_s``, peaking at ``gain_db`` over a noise floor that stays where it
    is."""
    from test_acoustic import _add, _string

    from plume.theory import parse_key, roman_to_chord
    rng = np.random.default_rng(seed)
    k = parse_key(key)
    numerals = ("i", "VI", "VII", "i") if k.mode == "minor" else ("I", "IV", "V", "I")
    beat = 60.0 / bpm
    buf = np.zeros(int(round((bars * 4 * beat + 2.0) * SR)))
    for b in range(bars):
        ch = roman_to_chord(numerals[b % 4], k)
        tones = [52 + (ch.root - 52) % 12 + iv for iv in ch.intervals]
        for i, at in enumerate(places):
            t = max(0.0, (b * 4 + at) * beat + float(rng.normal(0.0, jitter_s)))
            for midi in (tones if chord else [tones[(b + i) % len(tones)]]):
                y = _string(midi, dur_s=held_s or 1.5, velocity=float(rng.uniform(0.7, 0.9)),
                            rng=rng, attack_ms=float(rng.uniform(3.0, 6.0)), tau_s=ring_s)
                if held_s:
                    n = int(0.02 * SR)
                    y[-n:] *= np.linspace(1.0, 0.0, n)
                _add(buf, y, t)
    buf *= 10.0 ** (gain_db / 20.0) / float(np.max(np.abs(buf)))
    return (buf + rng.standard_normal(buf.size) * _HISS).astype(np.float32)


def _hear(audio, *, bpm, key, bars=7, tempo_off=0.0):
    """The first ``bars`` bars of ``audio`` into a direct input's StylePick, each bar
    with its bar line, and its pick as the band comes in.  ``tempo_off``: the bar lines
    run this share slow of her."""
    from plume.listen.features import HOP, FeatureExtractor
    fe = FeatureExtractor(sr=SR)
    frames = []
    for i in range(0, len(audio), 4800):
        frames.extend(fe.process(audio[i:i + 4800]))
    bar_s = 240.0 / bpm
    per = bar_s * SR / HOP
    pick = StylePick(trust_busy=True, hear_technique=True)
    for b in range(bars):
        bar = frames[int(round(b * per)):int(round((b + 1) * per))]
        pick.heard_bar(sum(bool(f.onset) for f in bar), frames=bar,
                       bar_start=b * bar_s * (1.0 + tempo_off), bar_s=bar_s * (1.0 + tempo_off))
    return pick, pick.at_entry(bpm * (1.0 - tempo_off), key)


def _strummed_folk(**kw):
    import folk_fixtures as ff
    audio = ff.make_take("strum", **{"bars": 8, **kw})
    rng = np.random.default_rng(1)
    return audio + rng.standard_normal(audio.size).astype(np.float32) * _HISS


#: style -> (the part, its tempo and key).  Each is what a player of the style might
#: play alone: jazz swings its eighths, house stabs between the beats, trap leaves
#: most of the bar empty, and so on.  The keys are the ones the tempo's default would
#: get wrong where there is one, so none of these is right by the tempo and key alone.
_PARTS = {
    "midwest": lambda: (_part(_EIGHTHS, bpm=140.0, key="E minor", ring_s=1.2), 140.0, "E minor"),
    "lofi": lambda: (_part((0.0, 1.57, 2.0, 3.57), bpm=76.0, key="A minor", ring_s=0.25,
                           held_s=0.4, chord=True), 76.0, "A minor"),
    "folk": lambda: (_strummed_folk(key="E minor", bpm=100.0,
                                    numerals=("i", "VII", "VI", "VII")), 100.0, "E minor"),
    "synthwave": lambda: (_part(_EIGHTHS, bpm=96.0, key="D major", ring_s=1.2),
                          96.0, "D major"),
    "hiphop": lambda: (_part((0.0, 0.58, 1.58, 2.58, 3.58), bpm=94.0, key="D major",
                             ring_s=0.2, held_s=0.25, chord=True), 94.0, "D major"),
    "rnb": lambda: (_part((0.58, 1.58, 2.0, 2.58, 3.58), bpm=68.0, key="F major",
                          ring_s=0.25, held_s=0.45, chord=True), 68.0, "F major"),
    "house": lambda: (_part((0.5, 1.5, 2.5, 3.5), bpm=124.0, key="C major", ring_s=0.15,
                            held_s=0.2, chord=True), 124.0, "C major"),
    "trap": lambda: (_part((0.0, 2.0), bpm=140.0, key="C minor", ring_s=0.3, held_s=0.6),
                     140.0, "C minor"),
    "jazz": lambda: (_part(_swung(0.66), bpm=150.0, key="Bb major", ring_s=0.5),
                     150.0, "Bb major"),
}


@pytest.mark.parametrize("style", sorted(_PARTS))
def test_a_synthetic_player_of_each_style_is_heard_as_it(style):
    audio, bpm, key = _PARTS[style]()
    pick, got = _hear(audio, bpm=bpm, key=key)
    assert got == style, pick.why
    assert pick.why.startswith(f"{style} "), pick.why


#: The players whose style used to turn with the gain knob, and a strummer: the part,
#: its tempo and key.  Each is heard at three input levels 40 dB apart over the same
#: noise floor.
_AT_ANY_GAIN = {
    "short notes on the beat": (dict(places=_QUARTERS, ring_s=0.2, held_s=0.25),
                                110.0, "E minor"),
    "stabs between the beats": (dict(places=_STABS, ring_s=0.15, held_s=0.2, chord=True),
                                85.0, "D major"),
    "two chords a bar": (dict(places=(0.0, 2.5), ring_s=0.4, held_s=0.8, chord=True),
                         124.0, "A minor"),
    "short chords on every eighth": (dict(places=_EIGHTHS, ring_s=0.15, held_s=0.2,
                                          chord=True), 72.0, "G major"),
    "strummed": (None, 96.0, "G major"),
}


@pytest.mark.parametrize("player", sorted(_AT_ANY_GAIN))
def test_the_style_does_not_turn_with_the_gain_knob(player):
    """Nothing heard of her is read from her level, so nothing is lost when she is
    quiet.  A reading taken from the level (the share of the time she sounds) went
    unknown under about -46 dBFS, and with it the style changed."""
    part, bpm, key = _AT_ANY_GAIN[player]
    seen = {}
    for gain_db in (-6.0, -26.0, -46.0):
        audio = (_strummed_folk(key=key, bpm=bpm, gain_db=gain_db) if part is None
                 else _part(bpm=bpm, key=key, gain_db=gain_db, **part))
        pick, got = _hear(audio, bpm=bpm, key=key)
        heard = pick.playing
        seen[gain_db] = (got, tuple(getattr(heard, f) is not None
                                    for f in ("strokes", "swing", "offbeat", "steady", "strum")))
    assert len(set(seen.values())) == 1, seen


def test_a_dotted_figure_played_by_hand_is_not_jazz():
    audio = _part(_each_beat(0.0, 0.75), bpm=120.0, key="E minor", ring_s=0.4,
                  jitter_s=0.016, seed=7)
    pick, got = _hear(audio, bpm=120.0, key="E minor")
    assert pick.playing.swing is None, pick.why
    assert got != "jazz", pick.why


def test_a_note_cut_off_just_before_the_next_is_timed_at_the_next():
    """A note damped as the next is struck leaves an edge of its own a few hops before
    the strike.  Timed there, her strokes on the beat ran a tenth of a beat early, her
    straight eighths read as swung, and a sparse line was jazz."""
    audio = _part((0.0, 1.5), bpm=96.0, key="D major", ring_s=1.0)
    pick, got = _hear(audio, bpm=96.0, key="D major")
    assert pick.playing.swing == pytest.approx(0.5, abs=0.04), pick.why
    assert got != "jazz", pick.why


def test_a_stroke_is_timed_where_it_was_played():
    """STROKE_LAG_S is how late the frame that holds a stroke's peak is stamped: without
    it every stroke reads a fifth of an eighth late at 140 bpm before anything else."""
    from plume.bandentry import StrokeEar
    from plume.listen.features import HOP, FeatureExtractor
    from plume.playstyle import STROKE_LAG_S
    bpm = 100.0
    audio = _part(_QUARTERS, bpm=bpm, key="A minor", ring_s=0.3, held_s=0.4, jitter_s=0.0)
    frames = FeatureExtractor(sr=SR).process(audio)
    per = 240.0 / bpm * SR / HOP
    ear, late = StrokeEar(), []
    for b in range(7):
        ear.bar(frames[int(round(b * per)):int(round((b + 1) * per))])
        late += [(t / (60.0 / bpm) + 0.5) % 1.0 - 0.5 for t in ear.last_at]
    assert len(late) >= 24
    assert float(np.median(late)) * 60.0 / bpm == pytest.approx(STROKE_LAG_S, abs=0.008)


def test_a_swung_player_is_jazz_at_a_tempo_that_used_to_be_midwest():
    audio = _part(_swung(0.66), bpm=110.0, key="F major", ring_s=0.5)
    pick, got = _hear(audio, bpm=110.0, key="F major")
    assert got == "jazz", pick.why
    assert pick.playing.swing == pytest.approx(0.66, abs=0.04)


def test_the_pick_holds_when_the_count_was_a_little_off_her_tempo():
    """She plays at 124 against bar lines counted at just under 123."""
    audio = _part((0.5, 1.5, 2.5, 3.5), bpm=124.0, key="A minor", ring_s=0.15, held_s=0.2,
                  chord=True)
    pick, got = _hear(audio, bpm=124.0, key="A minor", tempo_off=0.008)
    assert got == "house", pick.why
    assert pick.playing.offbeat > 0.9, pick.why


def test_the_log_gives_the_best_two_styles_and_everything_heard():
    audio, bpm, key = _PARTS["house"]()
    pick, got = _hear(audio, bpm=bpm, key=key)
    why = pick.why
    assert why.startswith("house ") and " over " in why, why
    for said in ("bpm 124.0", "mode major", f"busy {pick.busy:.2f} strokes a beat", "swing",
                 "off the beat", "steady", "strum"):
        assert said in why, (said, why)


def test_notes_alone_are_scored_on_how_many_there_are():
    """A MIDI input has no frames: each note is a stroke, and that is all it knows."""
    pick = StylePick(trust_busy=True)
    for _ in range(4):
        pick.heard_bar(2)
    assert pick.playing == Playing(strokes=0.5)
    assert pick.at_entry(140.0, "C minor") == "trap"
    busy = StylePick(trust_busy=True)
    for _ in range(4):
        busy.heard_bar(8)
    assert busy.at_entry(140.0, "C minor") == "midwest"


def _notes(places, *, bpm, bars=7, jitter_s=0.006, seed=5):
    """A MIDI input's StylePick fed ``bars`` bars of notes at ``places`` with their times,
    each stamped on the first frame at or after it, as MidiFrames stamps them."""
    from plume.listen.features import HOP
    rng = np.random.default_rng(seed)
    bar_s, hop = 240.0 / bpm, HOP / SR
    pick = StylePick(trust_busy=True)
    for b in range(bars):
        at = [b * bar_s + p * bar_s / 4.0 + float(rng.normal(0.0, jitter_s)) for p in places]
        stamped = [np.ceil(t / hop) * hop for t in at]
        pick.heard_bar(len(stamped), bar_start=b * bar_s, bar_s=bar_s, onset_times=stamped)
    return pick


@pytest.mark.parametrize("places,bpm,key,style", [
    (_swung(0.66), 120.0, "F major", "jazz"),
    (_STABS, 124.0, "C major", "house"),
    (_EIGHTHS, 96.0, "D major", "synthwave"),
    ((0.0, 0.58, 1.58, 2.58, 3.58), 94.0, "D major", "hiphop"),
])
def test_notes_with_their_times_are_placed_in_the_beat(places, bpm, key, style):
    """A MIDI input that hands on when each note was played is heard as a direct input
    is, but for the strum: notes alone could only ever say busy or sparse."""
    pick = _notes(places, bpm=bpm)
    assert pick.playing.strum is None and pick.playing.offbeat is not None, pick.why
    assert pick.at_entry(bpm, key) == style, pick.why
    assert "swing" in pick.why and "strum: not heard on this input" in pick.why, pick.why


@pytest.mark.parametrize("bpm,key", [(125.0, "C major"), (135.0, "A minor"),
                                     (145.0, "E major")])
def test_a_busy_even_picker_past_synthwave_s_range_is_midwest(bpm, key):
    """Straight eighths picked from 119 to 150 were folk, then trap, the style that
    leaves room: nothing fitted them.  (In a minor key up to 130 they are as near
    house's as midwest's, and the tempo's default, house, takes them.)"""
    audio = _part(_EIGHTHS, bpm=bpm, key=key, ring_s=0.6, held_s=None)
    pick, got = _hear(audio, bpm=bpm, key=key)
    assert got == "midwest", pick.why
    assert "no style fits" not in pick.why, pick.why


def test_a_room_mic_hears_nothing_of_how_she_plays():
    pick = StylePick(trust_busy=False, hear_technique=False)
    pick.heard_bar(16, frames=[], bar_start=0.0, bar_s=2.0)
    assert pick.playing is None
    assert pick.at_entry(124.0, "A minor") == tempo_default(124.0, "minor") == "house"
    assert "not hers alone" in pick.why


# ------------------------------------------------------------- a second look
#
# The style is picked as the band comes in, and may change once after, on how she plays
# now (StylePick.reconsider).  Fed one bar at a time, as the take feeds it.

def _take(feels, *, bpm, keys, pick=None, jitter_s=0.006, seed=5, entry_bars=7,
          tempo_off=0.0):
    """A MIDI input's StylePick fed ``entry_bars`` bars of the first feel, its pick as
    the band comes in, then bar by bar: ``feels`` is a list of (bars, places) and ``keys``
    the committed key of every bar after the band is in.  ``tempo_off``: she holds her
    own tempo, this share faster than the bar lines.  Returns the pick, its answer as
    the band came in, and every (bars in, style) it changed to."""
    from plume.listen.features import HOP
    rng = np.random.default_rng(seed)
    bar_s, hop = 240.0 / bpm, HOP / SR
    hers = bar_s / (1.0 + tempo_off)
    pick = pick or StylePick(trust_busy=True)
    plan = [places for n, places in feels for _ in range(n)]
    changes, at_entry, b = [], None, 0
    for i, places in enumerate(plan):
        if places is not None:
            at = [b * hers + p * hers / 4.0 + float(rng.normal(0.0, jitter_s))
                  for p in places]
            stamped = [np.ceil(t / hop) * hop for t in at]
            pick.heard_bar(len(stamped), bar_start=b * bar_s, bar_s=bar_s,
                           onset_times=stamped)
        b += 1
        if i + 1 == entry_bars:
            at_entry = pick.at_entry(bpm, keys[0])
        elif i + 1 > entry_bars:
            bars_in = i + 1 - entry_bars
            got = pick.reconsider(bars_in, bpm, keys[min(bars_in, len(keys) - 1)])
            if got is not None:
                changes.append((bars_in, got))
    return pick, at_entry, changes


_SPARSE = (0.0, 2.0)


@pytest.mark.parametrize("places,bpm,key", [
    (_SPARSE, 140.0, "C minor"), (_EIGHTHS, 96.0, "D major"), (_STABS, 124.0, "C major"),
    (_swung(0.66), 120.0, "F major"), ((0.0, 0.58, 1.58, 2.58, 3.58), 94.0, "D major"),
    (_QUARTERS, 85.0, "A minor"),
])
def test_a_player_who_keeps_her_feel_keeps_her_style(places, bpm, key):
    for seed in (5, 6, 7):
        pick, first, changes = _take([(60, places)], bpm=bpm, keys=[key], seed=seed,
                                     jitter_s=0.012)
        assert changes == [], (first, changes, pick.why)
        assert pick.style == first


@pytest.mark.parametrize("tempo_off", [0.001, 0.002, 0.003])
def test_a_player_who_drifts_off_the_bar_lines_keeps_her_style(tempo_off):
    """She swings, holding her own tempo a little faster than the count, and is a good
    part of a beat ahead of the bar lines well into the take.  Each part of a phrase is
    heard from where she is by then: placed from the bar line, her swung eighths
    landed off the beat and she was house."""
    pick, first, changes = _take([(60, _swung(0.66))], bpm=120.0, keys=["F major"],
                                 tempo_off=tempo_off)
    assert first == "jazz" and changes == [], (changes, pick.why)


def test_a_part_finds_her_again_where_following_lost_her():
    """Quarter notes on the beat, then from bar 6 a sixth of a beat late, as a player
    whose tempo is not quite the band's ends up in time: more than following moves in
    a bar, so the ear that followed her now hears every stroke off the beat.  A part
    that finds her again from there hears them on it."""
    from plume.playstyle import STROKE_LAG_S
    bpm, bar_s = 100.0, 2.4
    ear = RhythmEar()

    def bar(e, b, late):
        e.bar([b * bar_s + p * bar_s / 4.0 + late + STROKE_LAG_S for p in _QUARTERS],
              bar_start=b * bar_s, bar_s=bar_s)
    for b in range(6):
        bar(ear, b, 0.0)
    followed, found = ear.carry_on(), ear.carry_on(find=True)
    for b in range(6, 8):
        for e in (ear, followed, found):
            bar(e, b, bar_s / 4.0 / 6.0)
    assert followed.offbeat > 0.9
    assert found.offbeat == 0.0


def test_sparse_and_minor_then_dense_and_major_changes_style_once_on_a_phrase():
    """Two notes a bar in C minor, then even eighths in C major from 13 bars in: hip hop,
    which a minor key at 96 is when nothing else fits, then synthwave's even pulse."""
    keys = ["C minor"] * 13 + ["C major"] * 60
    pick, first, changes = _take([(20, _SPARSE), (60, _EIGHTHS)], bpm=96.0, keys=keys)
    assert first == "hiphop", pick.why
    assert len(changes) == 1, changes
    bars_in, style = changes[0]
    assert style == "synthwave", pick.why
    from plume.bandentry import section_line
    assert section_line(bars_in) and bars_in > 13
    assert pick.style == "synthwave"
    assert pick.why.startswith("synthwave ") and "led hiphop by" in pick.why, pick.why


def test_the_style_changes_never_twice():
    """Sparse, then an even pulse, then swung: three feels that each fit a style of
    their own, so a second change would have something to change to."""
    pick, first, changes = _take([(20, _SPARSE), (24, _EIGHTHS), (40, _swung(0.66))],
                                 bpm=96.0, keys=["C minor"])
    assert first == "hiphop"
    assert [s for _, s in changes] == ["synthwave"], changes
    assert pick.style == "synthwave"
    assert pick.reconsider(200, 96.0, "C minor") is None


def test_each_feel_after_the_first_could_change_the_style_on_its_own():
    """The two later feels of the take above each move a pick that hears them alone
    after the first: so only the limit keeps the third from changing the style."""
    for feel, style in ((_EIGHTHS, "synthwave"), (_swung(0.66), "jazz")):
        _, first, changes = _take([(20, _SPARSE), (40, feel)], bpm=96.0, keys=["C minor"])
        assert first == "hiphop" and [s for _, s in changes] == [style], (feel, changes)
    _, first, changes = _take([(20, _EIGHTHS), (40, _swung(0.66))], bpm=96.0,
                              keys=["C minor"])
    assert first == "synthwave" and [s for _, s in changes] == ["jazz"], changes


def test_never_in_the_first_phrase_after_the_band_comes_in():
    """She turns busy on the very bar the band comes in: the first phrase is the band's
    entrance, so the earliest change is at the end of the phrase after it."""
    from plume.bandentry import REGUESS_PHRASE_BARS
    pick, first, changes = _take([(7, _SPARSE), (60, _EIGHTHS)], bpm=96.0,
                                 keys=["C minor"])
    assert first == "hiphop"
    assert changes and changes[0][0] == 2 * REGUESS_PHRASE_BARS, changes


def test_a_fill_shorter_than_a_phrase_changes_nothing():
    """Four busy bars in the middle of a sparse take."""
    for at in range(13, 22):
        pick, first, changes = _take([(at, _SPARSE), (4, _EIGHTHS), (40, _SPARSE)],
                                     bpm=96.0, keys=["C minor"])
        assert first == "hiphop" and changes == [], (at, changes)


_DOTTED = (0.0, 0.75, 1.5, 2.0, 2.75, 3.5)
_OFF_EIGHTHS = (0.0, 0.58, 1.58, 2.58, 3.58)


def test_a_phrase_that_fits_another_style_barely_better_changes_nothing():
    """A dotted figure after an even pulse: midwest fits each part of the phrase only a
    little better than synthwave, well under REGUESS_LEAD."""
    for seed in (5, 6, 7):
        pick, first, changes = _take([(8, _EIGHTHS), (60, _DOTTED)], bpm=96.0,
                                     keys=["C major"], entry_bars=8, seed=seed)
        assert first == "synthwave" and changes == [], (seed, changes, pick.why)


def test_a_phrase_split_between_two_other_styles_changes_nothing():
    """Two bars off the beat, two bars swung, over and over: each part clearly fits
    another style than the one playing, hip hop then jazz, but no one style led the
    whole phrase."""
    feels = [(8, _EIGHTHS)] + [(2, _OFF_EIGHTHS), (2, _swung(0.66))] * 20
    for seed in (5, 6, 7):
        pick, first, changes = _take(feels, bpm=96.0, keys=["C major"], entry_bars=8,
                                     seed=seed)
        assert first == "synthwave" and changes == [], (seed, changes, pick.why)


@pytest.mark.parametrize("late", [1.0 / 6.0, 0.25])
def test_a_player_who_comes_back_in_behind_the_band_keeps_her_style(late):
    """Quarter notes, then the same a sixth of a beat or more behind the bar lines, as
    after a rest she comes back in a little late: further than following her moves in
    a bar.  Each part of a phrase finds her again where she is, so she still plays on
    her own beat; followed from where she was, she would be off it, and house."""
    behind = tuple(p + late for p in _QUARTERS)
    for seed in (5, 6, 7):
        pick, first, changes = _take([(20, _QUARTERS), (60, behind)], bpm=120.0,
                                     keys=["C major"], seed=seed)
        assert first == "midwest" and changes == [], (seed, changes, pick.why)


def test_a_rest_is_not_a_feel():
    """Bars she sat out say nothing: a phrase mostly resting changes nothing."""
    pick, first, changes = _take([(20, _SPARSE), (10, None), (2, _EIGHTHS), (30, None)],
                                 bpm=96.0, keys=["C minor"])
    assert changes == [], changes


def test_a_part_played_only_half_through_is_not_a_feel():
    """A busy bar, a bar's rest, over and over: every part of the phrase has a busy bar
    in it, but none was played all through, so none of them is heard as her feel."""
    feels = [(20, _SPARSE)] + [(1, _EIGHTHS), (1, None)] * 40
    for seed in (5, 6, 7):
        pick, first, changes = _take(feels, bpm=96.0, keys=["C minor"], seed=seed)
        assert first == "hiphop" and changes == [], (seed, changes, pick.why)
    # The same busy bars played through do change it, so it is the rests that held it.
    _, _, changes = _take([(20, _SPARSE), (80, _EIGHTHS)], bpm=96.0, keys=["C minor"])
    assert [s for _, s in changes] == ["synthwave"], changes


def test_the_first_phrase_holds_whatever_the_song_s_sections(monkeypatch):
    """Never in the first phrase after the band comes in, even where a section line
    would fall inside it."""
    import plume.bandentry as be
    monkeypatch.setattr(be, "section_line", lambda bars_in: True)
    pick, first, changes = _take([(7, _SPARSE), (60, _EIGHTHS)], bpm=96.0,
                                 keys=["C minor"])
    assert first == "hiphop"
    assert changes and changes[0][0] == 2 * be.REGUESS_PHRASE_BARS, changes


def test_a_room_mic_changes_style_when_the_key_s_mode_turns_it():
    """Nothing of how she plays is hers to hear, so only the key can move it: at 95 bpm
    a minor key is hip hop and a major one midwest."""
    pick = StylePick(trust_busy=False)
    keys = ["A minor"] * 10 + ["C major"] * 60
    pick, first, changes = _take([(60, _EIGHTHS)], bpm=95.0, keys=keys, pick=pick)
    assert first == "hiphop"
    from plume.bandentry import REGUESS_PHRASE_BARS, section_line
    assert [s for _, s in changes] == ["midwest"], changes
    # once the new mode has held a phrase, and on a section line
    assert section_line(changes[0][0])
    assert changes[0][0] >= 10 + REGUESS_PHRASE_BARS
    assert "mode major" in pick.why and "was minor" in pick.why, pick.why


def test_a_mode_that_does_not_turn_the_style_changes_nothing():
    """Swung eighths are jazz in either mode."""
    keys = ["F minor"] * 10 + ["F major"] * 60
    pick, first, changes = _take([(60, _swung(0.66))], bpm=120.0, keys=keys)
    assert first == "jazz" and changes == [], changes


@pytest.mark.parametrize("places,why", [
    (_DOTTED, "a near-tie: hip hop and midwest fit the dotted figure alike"),
    (_SPARSE, "nothing fits two notes a bar: only the tempo's default moves"),
])
def test_a_key_whose_mode_turns_moves_only_a_clear_answer(places, why):
    """On a direct input the key's mode only breaks ties and picks the default where
    nothing fits.  So a key that turns, rightly or as a long wobble, must not move the
    band on its own: the new answer has to fit what she played clearly better."""
    keys = ["C major"] * 10 + ["A minor"] * 60
    pick, first, changes = _take([(60, places)], bpm=95.0, keys=keys)
    assert first == "midwest" and changes == [], (why, changes, pick.why)
    assert score_styles(95.0, "minor", pick.playing)[0].style == "hiphop"


def test_a_key_whose_mode_turns_never_moves_to_a_style_that_fits_her_worse():
    """A figure hip hop fits a little in A minor at 95.  In C major hip hop is no
    candidate and midwest, the tempo's default, is the only one left, so it is the
    answer there without a tie: but it fits what she played worse than hip hop does."""
    from plume.playstyle import fit
    keys = ["A minor"] * 10 + ["C major"] * 60
    for seed in (5, 6, 7):
        pick, first, changes = _take([(60, (0.25, 1.0, 1.5, 2.0, 2.25))], bpm=95.0,
                                     keys=keys, seed=seed)
        assert first == "hiphop" and changes == [], (seed, changes, pick.why)
        major = score_styles(95.0, "major", pick.playing)
        assert major[0].style == "midwest"
        assert major[0].score <= fit("hiphop", pick.playing) + MARGIN


def test_a_key_whose_mode_turns_moves_a_style_that_now_fits_better():
    """Uneven gaps with notes off the beat at 95: hip hop's, but hip hop is no
    candidate in a major key, where nothing heard fits and midwest is the tempo's
    default.  The key turning minor makes hip hop the clear answer."""
    keys = ["C major"] * 10 + ["A minor"] * 60
    pick, first, changes = _take([(60, (0.0, 1.57, 2.0, 3.57))], bpm=95.0, keys=keys)
    assert first == "midwest"
    assert [s for _, s in changes] == ["hiphop"], (changes, pick.why)
    assert "was major" in pick.why, pick.why


def test_a_key_that_wobbles_back_before_a_phrase_changes_nothing():
    pick = StylePick(trust_busy=False)
    keys = ["A minor"] * 10 + ["C major"] * 5 + ["A minor"] * 60
    pick, first, changes = _take([(60, _EIGHTHS)], bpm=95.0, keys=keys, pick=pick)
    assert first == "hiphop" and changes == [], changes


def test_a_pick_that_has_not_seen_the_band_in_never_reconsiders():
    pick = _notes(_EIGHTHS, bpm=140.0)
    assert pick.reconsider(16, 140.0, "C major") is None


def test_the_style_changes_only_on_a_section_line():
    """The song's sections (SONG_ARC) after its first break start four bars off an
    eight-bar grid counted from the entry, at 44, 60 and 76: a change lands on one of
    those, never in the middle of a section."""
    from plume.bandentry import SONG_ARC, section_line, song_section
    lines = [b for b in range(1, 130) if section_line(b)]
    assert lines == [16, 32, 40, 44, 60, 76, 80, 104, 120]
    assert all(song_section(b)[0] != song_section(b - 1)[0] for b in lines[:-1])
    assert section_line(sum(bars for _, bars, _ in SONG_ARC))
    # She turns busy late, after the break: the change waits for the lift at 44.
    pick, first, changes = _take([(7 + 34, _SPARSE), (60, _EIGHTHS)], bpm=96.0,
                                 keys=["C minor"])
    assert first == "hiphop" and changes == [(44, "synthwave")], changes


def _played(parts, *, bpm, key):
    """Generated parts, (places, bars, _part's options), one after another, as a direct
    input's frames."""
    from plume.listen.features import FeatureExtractor
    bar_n = int(round(240.0 / bpm * SR))
    total = sum(bars for _, bars, _ in parts)
    audio = np.zeros(total * bar_n + 2 * SR, dtype=np.float32)
    at = 0
    for i, (places, bars, kw) in enumerate(parts):
        y = _part(places, bpm=bpm, key=key, bars=bars, seed=4 + i, **kw)
        n = min(y.size, audio.size - at * bar_n) if i == len(parts) - 1 else bars * bar_n
        audio[at * bar_n:at * bar_n + n] += y[:n]
        at += bars
    fe = FeatureExtractor(sr=SR)
    frames = []
    for i in range(0, len(audio), 4800):
        frames.extend(fe.process(audio[i:i + 4800]))
    return frames, total


@pytest.mark.parametrize("then,style", [
    ((_STABS, 18, {"ring_s": 0.15, "held_s": 0.2, "chord": True}), "house"),
    ((_QUARTERS, 18, {"ring_s": 1.2}), None),
])
def test_a_direct_input_hears_her_feel_change_from_her_frames(then, style):
    """The second look as a guitar input feeds it: frames through the stroke, rhythm and
    strum ears, each part of a phrase heard on from where the last left off.  Plain
    quarter notes, then stabs between the beats changes to house once; the same quarter
    notes all through change nothing."""
    from plume.listen.features import HOP
    bpm, key, entry = 120.0, "C major", 8
    frames, bars = _played([(_QUARTERS, entry, {"ring_s": 1.2}), then], bpm=bpm, key=key)
    bar_s = 240.0 / bpm
    per = bar_s * SR / HOP
    pick = StylePick(trust_busy=True, hear_technique=True)
    changes = []
    for b in range(bars):
        bar = frames[int(round(b * per)):int(round((b + 1) * per))]
        pick.heard_bar(sum(bool(f.onset) for f in bar), frames=bar,
                       bar_start=b * bar_s, bar_s=bar_s)
        if b + 1 == entry:
            assert pick.at_entry(bpm, key) == "midwest", pick.why
        elif b + 1 > entry:
            got = pick.reconsider(b + 1 - entry, bpm, key)
            if got is not None:
                changes.append((b + 1 - entry, got))
    assert changes == ([] if style is None else [(16, style)]), (changes, pick.why)
