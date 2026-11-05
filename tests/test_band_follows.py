"""The band follows the player, with patience (Juniper, 2026-11-06).

"if i just make like a little mistake it doesnt assume thats the whole direction
we're going."  So nothing moves on one bar or two: a new way of playing has to hold
for :data:`FOLLOW_HOLD_BARS` before the band answers, and then it answers gradually
and within bounds.  Pure, so all of it is testable without a device.
"""
import pytest

from plume.bandentry import (FOLLOW, FOLLOW_CUE_HOLD_BARS, FOLLOW_DENSITY_MAX,
                             FOLLOW_DENSITY_MIN, FOLLOW_DENSITY_STEP, FOLLOW_GAIN_MAX_DB,
                             FOLLOW_GAIN_MIN_DB, FOLLOW_GAIN_STEP_DB, FOLLOW_HOLD_BARS,
                             FOLLOW_HOLD_BARS_ROOM, FOLLOW_LEARN_BARS, SONG_ARC, BandFollow,
                             song_section)

NORMAL_DB, NORMAL_BUSY = -24.0, 1.0


def learned(*, trusted=True, bars=FOLLOW_LEARN_BARS):
    f = BandFollow(trusted=trusted)
    for _ in range(bars):
        f.bar(NORMAL_DB, NORMAL_BUSY if trusted else None, playing=True)
    return f


def play(f, bars, db=NORMAL_DB, busy=NORMAL_BUSY, *, playing=True):
    busy = busy if f.trusted else None
    return [f.bar(db, busy, playing=playing) for _ in range(bars)]


def neutral(s):
    return s.gain_db == 0.0 and s.density == 1.0 and not s.moved


# ------------------------------------------------------------ patience

def test_steady_playing_moves_nothing():
    f = learned()
    assert all(neutral(s) for s in play(f, 40))


def test_one_wild_bar_moves_nothing():
    f = learned()
    states = play(f, 4) + play(f, 1, db=NORMAL_DB + 15, busy=4.0) + play(f, 12)
    assert all(neutral(s) for s in states)


@pytest.mark.parametrize("bars", range(1, FOLLOW_HOLD_BARS))
def test_a_short_mistake_moves_nothing(bars):
    f = learned()
    states = play(f, bars, db=NORMAL_DB - 12, busy=0.2) + play(f, 12)
    assert all(neutral(s) for s in states)


def test_nothing_follows_before_it_knows_how_they_normally_play():
    f = BandFollow(trusted=True)
    states = play(f, FOLLOW_LEARN_BARS - 1) + play(f, 2, db=NORMAL_DB - 12)
    assert all(neutral(s) for s in states)


def test_a_rest_is_no_evidence_either_way():
    f = learned()
    soft = NORMAL_DB - 9
    states = play(f, 2, db=soft) + play(f, 3, db=-90.0, playing=False) + play(f, 1, db=soft)
    assert states[-1].moved == ("softer",)


# ------------------------------------------------------------ following

def test_sustained_soft_playing_brings_the_band_down_to_find_them():
    f = learned()
    states = play(f, 10, db=NORMAL_DB - 9, busy=0.5)
    first = next(i for i, s in enumerate(states) if not neutral(s))
    assert first == FOLLOW_HOLD_BARS - 1          # lands on the third soft bar
    assert "softer" in states[first].moved
    assert states[-1].gain_db < -2.0 and states[-1].density < 0.9


def test_sustained_wild_playing_lifts_the_band():
    f = learned()
    states = play(f, 12, db=NORMAL_DB + 6, busy=2.0)
    assert states[-1].gain_db > 1.0 and states[-1].density > 1.1


def test_it_comes_back_up_when_they_do():
    f = learned()
    play(f, 12, db=NORMAL_DB - 9)
    back = play(f, 20)
    assert back[-1].gain_db == pytest.approx(0.0, abs=0.6)
    assert back[-1].density == pytest.approx(1.0, abs=0.05)
    assert any("louder" in s.moved for s in back)


@pytest.mark.parametrize("db,busy", [(NORMAL_DB + 40, 2.4), (NORMAL_DB - 50, 0.0)])
def test_bounds_hold_however_far_they_go(db, busy):
    f = learned()
    for s in play(f, 60, db=db, busy=busy):
        assert FOLLOW_GAIN_MIN_DB <= s.gain_db <= FOLLOW_GAIN_MAX_DB
        assert FOLLOW_DENSITY_MIN <= s.density <= FOLLOW_DENSITY_MAX


def test_it_never_buries_the_player_upward():
    # The band answers loud with less than it was given, and never by much.
    assert FOLLOW_GAIN_MAX_DB <= 3.0 and FOLLOW_GAIN_MAX_DB < -FOLLOW_GAIN_MIN_DB


def test_every_move_is_gradual():
    f = learned()
    states = play(f, 15, db=NORMAL_DB - 30, busy=0.0) + play(f, 15, db=NORMAL_DB + 30, busy=2.4)
    for a, b in zip(states, states[1:]):
        assert abs(b.gain_db - a.gain_db) <= FOLLOW_GAIN_STEP_DB + 1e-9
        assert abs(b.density - a.density) <= FOLLOW_DENSITY_STEP + 1e-9


def test_busier_playing_fills_the_band_in_without_getting_louder():
    f = learned()
    states = play(f, 12, busy=2.0)
    assert states[-1].density > 1.1 and states[-1].gain_db == 0.0


def test_a_torrent_of_notes_gets_room_not_more_band():
    # mind.py's inverse density coupling: past density_high, more band buries them.
    f = learned()
    states = play(f, 20, busy=3.5)
    assert all(s.density <= 1.0 for s in states)


# ------------------------------------------------------------ a room mic

def test_a_room_mic_waits_longer():
    f = learned(trusted=False)
    states = play(f, 10, db=NORMAL_DB - 9)
    first = next(i for i, s in enumerate(states) if not neutral(s))
    assert first == FOLLOW_HOLD_BARS_ROOM - 1 > FOLLOW_HOLD_BARS - 1


def test_a_room_mic_cannot_count_notes():
    f = learned(trusted=False)
    assert f.bar(NORMAL_DB, 3.0, playing=True).density == 1.0
    assert all(neutral(s) for s in play(f, 20))


def test_a_room_mic_does_not_chase_its_own_band():
    # The mic hears the band, so a band that lifts would read as a player who lifted.
    # A mic that hears nothing but the band must settle, not climb to the ceiling.
    f = learned(trusted=False)
    states = play(f, FOLLOW_HOLD_BARS_ROOM + 6, db=NORMAL_DB + 6)
    for _ in range(60):
        states.append(f.bar(NORMAL_DB + 6 + states[-1].gain_db, None, playing=True))
    assert states[-1].gain_db < FOLLOW_GAIN_MAX_DB
    assert states[-1].gain_db == pytest.approx(states[-20].gain_db, abs=0.01)


def test_a_room_mic_moves_less_far():
    near, room = learned(), learned(trusted=False)
    a = play(near, 40, db=NORMAL_DB - 30)[-1]
    b = play(room, 40, db=NORMAL_DB - 30)[-1]
    assert b.gain_db > a.gain_db and b.density > a.density


def test_a_room_mic_relearns_once_the_band_is_in():
    # Over drums alone its level is drums and player; with the band in it is louder.
    f = BandFollow(trusted=False)
    for _ in range(8):
        f.bar(NORMAL_DB - 10, None, playing=True, band_in=False)
    states = [f.bar(NORMAL_DB, None, playing=True) for _ in range(30)]
    assert all(neutral(s) for s in states)


# ------------------------------------------------------------ cues win

def test_sparser_wins_over_a_band_leaning_busier():
    f = learned()
    play(f, 12, db=NORMAL_DB + 6, busy=2.0)
    s = f.cue("sparser")
    assert s.density <= 1.0
    held = play(f, FOLLOW_CUE_HOLD_BARS, db=NORMAL_DB + 6, busy=2.0)
    assert all(x.density <= 1.0 for x in held)


def test_busier_wins_over_a_band_leaning_sparser():
    f = learned()
    play(f, 12, db=NORMAL_DB - 9, busy=0.3)
    assert f.cue("busier").density >= 1.0
    held = play(f, FOLLOW_CUE_HOLD_BARS, db=NORMAL_DB - 9, busy=0.3)
    assert all(x.density >= 1.0 for x in held)


def test_a_cue_keeps_a_lean_that_agrees_with_it():
    f = learned()
    leaning = play(f, 12, db=NORMAL_DB - 9, busy=0.3)[-1].density
    assert f.cue("sparser").density == pytest.approx(leaning)


# ------------------------------------------------------------ the seam

def test_follow_is_the_only_mode_for_now():
    assert BandFollow(trusted=True).mode == FOLLOW
    with pytest.raises(ValueError, match="roadmap"):
        BandFollow(trusted=True, mode="lead")


def test_following_softens_the_song_arc_but_keeps_its_shape():
    f = BandFollow(trusted=True)
    arc = {name: d for name, _, d in SONG_ARC}
    shaped = {name: f.shape(d) for name, d in arc.items()}
    assert shaped["break"] < shaped["verse"] < shaped["lift"]
    assert shaped["lift"] - shaped["break"] < arc["lift"] - arc["break"]
    assert song_section(16)[1] == SONG_ARC[1][2]    # the arc itself is unchanged
