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
    """On the beat with room, in gaps a shade more even than lofi's: folk fits it whole
    and lofi but for that shade.  What was heard does not tell them apart, so the tempo
    does."""
    playing = Playing(strokes=0.9, swing=0.5, offbeat=0.3, steady=0.73)
    scores = score_styles(85.0, "major", playing)
    assert tempo_default(85.0, "major") == "lofi"
    assert [s.style for s in scores[:2]] == ["lofi", "folk"], explain(scores)
    assert scores[1].score - MARGIN <= scores[0].score < scores[1].score
    assert explain(scores).endswith("a near-tie")
    # further out it is no longer a tie, and the style that fits takes it
    clearly = Playing(strokes=0.9, swing=0.5, offbeat=0.3, steady=0.9)
    assert detect_style(85.0, "major", playing=clearly) == "folk"
    assert not explain(score_styles(85.0, "major", clearly)).endswith("a near-tie")


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


@pytest.mark.parametrize("bpm", [110.0, 125.0, 170.0, 200.0])
@pytest.mark.parametrize("mode", ["major", "minor"])
def test_with_no_swing_heard_jazz_is_not_even_a_candidate(bpm, mode):
    """Short notes on every beat have no off-beat eighths to hear a swing in.  Jazz
    used to lose one point for that and win whenever every other style lost more."""
    quarters = Playing(strokes=1.0, swing=None, offbeat=0.0, steady=1.0, strum=0.0)
    scores = score_styles(bpm, mode, quarters)
    assert "jazz" not in [s.style for s in scores]
    assert scores[0].style in ("folk", "trap", "midwest"), explain(scores)


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


def test_plain_playing_on_the_beat_is_folk_without_a_strum():
    """Folk is where plain playing goes when it fits nothing else at her tempo: a note
    on every beat in a minor key at 110, where the tempo alone says synthwave."""
    quarters = Playing(strokes=1.0, swing=None, offbeat=0.0, steady=1.0, strum=0.0)
    scores = score_styles(110.0, "minor", quarters)
    assert scores[0].style == "folk" and scores[0].score == 0.0, explain(scores)


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


def test_a_room_mic_hears_nothing_of_how_she_plays():
    pick = StylePick(trust_busy=False, hear_technique=False)
    pick.heard_bar(16, frames=[], bar_start=0.0, bar_s=2.0)
    assert pick.playing is None
    assert pick.at_entry(124.0, "A minor") == tempo_default(124.0, "minor") == "house"
    assert "not hers alone" in pick.why
