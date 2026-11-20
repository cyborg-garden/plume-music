"""A style from how she plays (plume.playstyle): the rhythm ear, the scored rule, and
one player of each bundled style who gets it.

Unit tests on stroke times and on feature values, and a few short synthetic players
through the real feature extractor.  No take, no device.
"""
from __future__ import annotations

import numpy as np
import pytest

from plume.bandentry import StylePick, detect_style
from plume.playstyle import (PROFILES, Playing, RhythmEar, explain, score_styles,
                             tempo_default, tempo_range)

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


def _swung(swing):
    return tuple(b + s for b in range(4) for s in (0.0, swing))


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
    ear = _rhythm((0.0, 1.0, 2.0, 3.0))
    assert ear.swing is None
    assert ear.offbeat == pytest.approx(0.0, abs=0.02)


def test_stabs_between_the_beats_are_all_off_it():
    ear = _rhythm((0.5, 1.5, 2.5, 3.5))
    assert ear.offbeat == pytest.approx(1.0, abs=0.02)
    assert ear.swing == pytest.approx(0.5, abs=0.02)


def test_sixteenths_are_read_by_their_eighths():
    """The last sixteenth of the beat is not a late eighth.  Three of her four strokes
    are off the beat."""
    ear = _rhythm(tuple(i / 4.0 for i in range(16)))
    assert ear.swing == pytest.approx(0.5, abs=0.02)
    assert ear.offbeat == pytest.approx(0.75, abs=0.03)


def test_triplets_are_not_swing():
    ear = _rhythm(tuple(b + s for b in range(4) for s in (0.0, 1 / 3, 2 / 3)))
    assert ear.swing is None


def test_nothing_heard_is_nothing_known():
    ear = RhythmEar()
    ear.bar([], bar_start=0.0, bar_s=2.4)
    assert ear.swing is None and ear.offbeat is None


@pytest.mark.parametrize("tempo_off", [-0.009, 0.009])
@pytest.mark.parametrize("places,offbeat,swing", [
    (_EIGHTHS, 0.5, 0.5),
    ((0.5, 1.5, 2.5, 3.5), 1.0, 0.5),
    ((0.0, 1.0, 2.0, 3.0), 0.0, None),
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


def test_the_rhythm_ear_keeps_a_bounded_memory():
    ear = _rhythm(_EIGHTHS, bars=400)
    assert ear.memory_items() < 100


# ------------------------------------------------------------------- the rule

#: A plausible player of each style, as the five things heard of her, and a tempo in the
#: style's own range.
_PLAYERS = {
    "midwest": (140.0, Playing(strokes=1.9, swing=0.50, offbeat=0.5, sounding=1.0, strum=0.05)),
    "lofi": (78.0, Playing(strokes=0.6, swing=0.57, offbeat=0.5, sounding=0.6, strum=None)),
    "folk": (100.0, Playing(strokes=1.4, swing=0.50, offbeat=0.45, sounding=0.97, strum=0.45)),
    "synthwave": (84.0, Playing(strokes=2.0, swing=0.50, offbeat=0.5, sounding=1.0, strum=0.02)),
    "hiphop": (94.0, Playing(strokes=0.9, swing=0.59, offbeat=0.7, sounding=0.6, strum=None)),
    "rnb": (66.0, Playing(strokes=0.8, swing=0.58, offbeat=0.75, sounding=0.7, strum=0.0)),
    "house": (124.0, Playing(strokes=1.5, swing=0.50, offbeat=1.0, sounding=0.55, strum=0.03)),
    "trap": (140.0, Playing(strokes=0.5, swing=None, offbeat=0.45, sounding=0.75, strum=None)),
    "jazz": (150.0, Playing(strokes=1.3, swing=0.65, offbeat=0.5, sounding=0.8, strum=0.0)),
}


def test_every_bundled_style_has_a_profile_and_a_player():
    from plume.styles_lib import list_styles
    assert set(PROFILES) == set(list_styles()) == set(_PLAYERS)


@pytest.mark.parametrize("style", sorted(set(_PLAYERS) - {"synthwave"}))
@pytest.mark.parametrize("mode", ["major", "minor", None])
def test_a_player_of_each_style_gets_it_in_any_mode(style, mode):
    bpm, playing = _PLAYERS[style]
    lo, hi = tempo_range(style)
    assert lo <= bpm <= hi
    scores = score_styles(bpm, mode, playing)
    assert scores[0].style == style, explain(scores)
    assert scores[0].score > scores[1].score, "won on the tie-break, not on her playing"


def test_a_synthwave_player_gets_it_where_the_tempo_leans_that_way():
    """Nothing heard tells synthwave's driving eighths from midwest's, so the two tie
    and the tempo's default decides: synthwave in a minor key from 100 to 118."""
    _, playing = _PLAYERS["synthwave"]
    scores = score_styles(105.0, "minor", playing)
    assert [s.style for s in scores[:2]] == ["synthwave", "midwest"], explain(scores)
    assert scores[0].score == scores[1].score == 0.0
    assert score_styles(105.0, "major", playing)[0].style == "midwest"


def test_the_same_tempo_gets_the_style_she_plays():
    """85 bpm is inside seven styles' ranges, and used to be lofi or midwest."""
    got = {name: detect_style(85.0, "major", playing=p) for name, p in {
        "busy and ringing": Playing(strokes=2.6, swing=0.5, offbeat=0.6, sounding=1.0),
        "strummed": Playing(strokes=1.4, swing=0.5, offbeat=0.45, sounding=1.0, strum=0.5),
        "sparse, light swing": Playing(strokes=0.5, swing=0.57, offbeat=0.4, sounding=0.6),
        "light swing off the beat": Playing(strokes=0.9, swing=0.59, offbeat=0.75,
                                            sounding=0.6),
        "hard swing": Playing(strokes=1.3, swing=0.66, offbeat=0.5, sounding=0.8),
    }.items()}
    assert got == {"busy and ringing": "midwest", "strummed": "folk",
                   "sparse, light swing": "lofi", "light swing off the beat": "hiphop",
                   "hard swing": "jazz"}


def test_midwest_is_not_the_answer_for_most_players():
    for name, (bpm, playing) in _PLAYERS.items():
        if name not in ("midwest", "synthwave"):
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


def test_a_tie_goes_to_the_tempo_s_default():
    """Steady ringing eighths fit midwest and synthwave alike: the key turns it."""
    playing = Playing(strokes=2.0, swing=0.5, offbeat=0.5, sounding=1.0, strum=0.0)
    assert detect_style(105.0, "major", playing=playing) == "midwest"
    assert detect_style(105.0, "minor", playing=playing) == "synthwave"


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
    straight = Playing(strokes=1.0, swing=0.5, offbeat=0.5, sounding=0.8)
    swung = Playing(strokes=1.0, swing=0.66, offbeat=0.5, sounding=0.8)
    for bpm in (96.0, 140.0, 190.0):
        assert detect_style(bpm, "major", playing=straight) != "jazz"
        assert detect_style(bpm, "major", playing=swung) == "jazz"
    assert detect_style(170.0, "minor", 3.0) == "midwest"      # strokes alone, busy
    # No off-beat eighths to hear a swing in: jazz pays for that, and another style
    # that fits what was heard takes it.
    on_the_beat = Playing(strokes=1.0, swing=None, offbeat=0.0, sounding=1.0, strum=0.5)
    scores = score_styles(96.0, "major", on_the_beat)
    assert scores[0].style == "folk"
    assert dict(next(s for s in scores if s.style == "jazz").against)["swing"] == 1.0


def test_folk_is_the_strum_and_steady_strokes_in_any_key():
    """It used to need an acoustic sound or a major key as well, which a guitar through
    a pickup in a minor key never had."""
    strummed = Playing(strokes=1.4, swing=0.5, offbeat=0.45, sounding=0.97, strum=0.45)
    for mode in ("major", "minor", None):
        assert detect_style(96.0, mode, playing=strummed) == "folk"
    picked = Playing(strokes=1.9, swing=0.5, offbeat=0.5, sounding=1.0, strum=0.02)
    assert detect_style(96.0, "major", playing=picked) != "folk"
    # a strum is not enough: off the beat with gaps it is something else
    stabs = Playing(strokes=1.0, swing=0.5, offbeat=1.0, sounding=0.5, strum=0.45)
    assert detect_style(124.0, "minor", playing=stabs) == "house"
    assert detect_style(140.0, "major", playing=strummed) != "folk"   # past folk's range
    # and too few strokes to hear a strum in is not a strum
    sparse = Playing(strokes=0.46, swing=0.5, offbeat=0.5, sounding=0.9, strum=None)
    scores = score_styles(85.0, "minor", sparse)
    assert scores[0].style != "folk"
    assert dict(next(s for s in scores if s.style == "folk").against)["strum"] == 1.0


def test_strokes_and_a_strum_alone_stand_for_her_playing():
    assert detect_style(96.0, "major", 1.4, folk=0.5) == "folk"
    assert detect_style(96.0, "major", 1.4, folk=0.05) == "midwest"
    assert detect_style(96.0, "major", 1.4) == "midwest"
    assert detect_style(96.0, "major", folk=None) == tempo_default(96.0, "major")


def test_one_reading_a_little_outside_a_band_costs_a_little():
    near = score_styles(140.0, None, Playing(strokes=1.20))
    far = score_styles(140.0, None, Playing(strokes=0.75))
    cost = {s.style: -s.score for s in near}["midwest"]
    assert cost == pytest.approx(0.1)
    assert {s.style: -s.score for s in far}["midwest"] == 1.0     # never more than 1
    assert near[0].style == "midwest" and far[0].style == "trap"


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


def _part(places, *, bpm, key, ring_s=1.0, held_s=None, bars=8, chord=False, seed=4):
    """A picked part: a chord tone (or, ``chord``, three of them at once) at each of
    ``places`` (beats into the bar), ringing with time constant ``ring_s`` and damped
    after ``held_s``, over a noise floor."""
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
            t = (b * 4 + at) * beat + float(rng.normal(0.0, 0.004))
            for midi in (tones if chord else [tones[(b + i) % len(tones)]]):
                y = _string(midi, dur_s=held_s or 1.5, velocity=float(rng.uniform(0.7, 0.9)),
                            rng=rng, attack_ms=float(rng.uniform(3.0, 6.0)), tau_s=ring_s)
                if held_s:
                    n = int(0.02 * SR)
                    y[-n:] *= np.linspace(1.0, 0.0, n)
                _add(buf, y, t)
    buf *= 0.5 / float(np.max(np.abs(buf)))
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
    audio = ff.make_take("strum", bars=8, **kw)
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
    "synthwave": lambda: (_part(_EIGHTHS, bpm=105.0, key="A minor", ring_s=1.2),
                          105.0, "A minor"),
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
                 "off the beat", "sounding", "strum"):
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
