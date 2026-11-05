"""How soon a take ends after the player stops.

A direct input or keys can really tell, so the take ends about 20 seconds later.  Bars
are read two bars late and an ordinary rest must not dip the band, so the tempo sets a
floor under that: the stated ranges below are what the numbers come to, and
docs/PLAYING.md repeats them.  A room mic cannot hear a player stop over the band, so it
keeps the long wait it always had.
"""
import json

import pytest
from conftest import fake_audio, patched_module, short_socket_path
from test_bandentry import _take

from plume import realtime
from plume import roles as roles_mod
from plume.bandentry import END_AFTER_S, MIN_REST_S, SilenceEnding

RIG = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]


def direct():
    return SilenceEnding(end_after_s=END_AFTER_S)


def seconds_to_the_end(bpm, stops_at, ending, *, back_at=None):
    """Drive an ending the way plan_one_bar does: bar b reads bar b - 2, and the last
    bar plays out.  She stops ``stops_at`` bars in.  Seconds from there to the silence."""
    bar_s = 240.0 / bpm
    b = 2
    while True:
        read = b - 2
        played = read < stops_at or (back_at is not None and back_at <= read < back_at + 4)
        _, done = ending.update(b * bar_s, played, bar_s)
        if done:
            return (b + 1) * bar_s - stops_at * bar_s
        b += 1
        assert b < 2000, "it never ended"


PHASES = [20 + i / 10 + 0.05 for i in range(10)]


@pytest.mark.parametrize("bpm,low,high", [
    (45, 37.0, 43.0),
    (60, 28.0, 32.0),
    (70, 24.0, 28.0),
    (75, 22.0, 26.0),
    (90, 21.0, 24.0),
    (100, 19.0, 22.0),
    (115, 19.0, 23.0),
    (130, 19.0, 23.0),
    (150, 19.0, 22.0),
    (170, 19.0, 22.0),
])
def test_it_ends_about_twenty_seconds_after_she_stops(bpm, low, high):
    for stops_at in PHASES:
        got = seconds_to_the_end(bpm, stops_at, direct())
        assert low <= got <= high, (bpm, stops_at, got)


@pytest.mark.parametrize("bpm,low,high", [(45, 69.0, 75.0), (115, 45.0, 48.0), (170, 40.0, 43.0)])
def test_a_room_mic_keeps_the_numbers_it_always_had(bpm, low, high):
    for stops_at in PHASES:
        assert low <= seconds_to_the_end(bpm, stops_at, SilenceEnding()) <= high


def dips_in_a_rest(bpm, stops_at, rest_s):
    """She rests ``rest_s`` seconds from ``stops_at`` bars in, then plays on.  A bar counts as
    played if any of it was, and bar b reads bar b - 2.  Does the band ever dip?"""
    bar_s = 240.0 / bpm
    back_at = stops_at + rest_s / bar_s
    e = direct()
    for b in range(2, int(back_at) + 12):
        read = b - 2
        played = read < stops_at or read + 1 > back_at
        gain, done = e.update(b * bar_s, played, bar_s)
        assert not done
        if gain < 0.0:
            return True
    return False


@pytest.mark.parametrize("bpm", [45, 60, 66, 70, 75, 80, 90, 100, 115, 130, 150, 170])
def test_an_ordinary_rest_never_dips_the_band_at_any_tempo(bpm):
    for stops_at in PHASES:
        assert not dips_in_a_rest(bpm, stops_at, MIN_REST_S - 0.01), (bpm, stops_at)


@pytest.mark.parametrize("bpm", [45, 66, 115, 170])
def test_one_silent_bar_is_never_enough(bpm):
    bar_s = 240.0 / bpm
    e = direct()
    e.update(0.0, True, bar_s)
    assert e.update(bar_s, False, bar_s)[0] == 0.0


@pytest.mark.parametrize("bpm", [66, 115])
def test_a_long_enough_rest_still_fades(bpm):
    assert dips_in_a_rest(bpm, 20.05, 25.0)


def test_a_room_mic_ending_is_unchanged_even_given_the_bar_length():
    e = SilenceEnding()
    assert e.update(0.0, True, 2.0) == (0.0, False)
    assert e.update(29.0, False, 2.0) == (0.0, False)
    assert e.update(30.0, False, 2.0) == (-9.0, False)
    assert e.update(32.0, False, 2.0) == (-18.0, False)
    assert e.update(34.0, False, 2.0) == (-27.0, False)
    assert e.update(36.0, False, 2.0) == (-36.0, False)
    assert e.update(38.0, False, 2.0) == (-36.0, False)
    assert e.update(40.0, False, 2.0) == (-36.0, True)


@pytest.mark.parametrize("bar_s,bars", [(5.33, 2), (4.0, 2), (2.0, 3), (1.5, 4), (1.2, 5), (0.9, 6), (0.5, 6)])
def test_the_fade_takes_about_six_seconds_in_bars(bar_s, bars):
    e = SilenceEnding(end_after_s=20.0, floor_db=-36.0)
    e.update(0.0, True, bar_s)
    t, gains = 0.0, []
    while len(gains) < bars:
        t += bar_s
        db, done = e.update(t, False, bar_s)
        assert not done
        if db < 0.0:
            gains.append(db)
    assert gains[-1] == pytest.approx(-36.0)
    assert all(g > -36.0 for g in gains[:-1])


def _first_dip(e, t, bar_s):
    """Seconds after ``t`` until the band first dips, with nobody playing."""
    start = t
    while True:
        t += bar_s
        if e.update(t, False, bar_s)[0] < 0.0:
            return t, t - start


@pytest.mark.parametrize("bar_s", [5.33, 2.087, 1.41])
def test_coming_back_during_the_fade_brings_the_band_straight_back(bar_s):
    e = direct()
    e.update(0.0, True, bar_s)
    t, waited = _first_dip(e, 0.0, bar_s)
    t += bar_s
    assert e.update(t, True, bar_s) == (0.0, False)
    # and the whole wait starts over
    assert _first_dip(e, t, bar_s)[1] == pytest.approx(waited)


def test_a_return_at_the_very_end_is_still_heard():
    e = direct()
    # 20 - (3 + 2 + 1.5) * 2 = 7 seconds, so it dips on the bar at 8 s: three bars of
    # fade, then two of listening, the last of which would end it.
    e.update(0.0, True, 2.0)
    assert e.update(6.0, False, 2.0) == (0.0, False)
    for t in (8.0, 10.0, 12.0, 14.0):
        assert e.update(t, False, 2.0)[1] is False
    assert e.update(16.0, True, 2.0) == (0.0, False)


def test_the_last_of_the_listening_ends_it():
    e = direct()
    e.update(0.0, True, 2.0)
    assert e.update(6.0, False, 2.0) == (0.0, False)
    for t in (8.0, 10.0, 12.0, 14.0):
        assert e.update(t, False, 2.0)[1] is False
    assert e.update(16.0, False, 2.0) == (-36.0, True)


@pytest.mark.parametrize("bar_s", [5.33, 2.087, 1.41])
def test_she_gets_the_old_thirty_seconds_until_she_has_played(bar_s):
    """She has only just knocked: her hands are on their way back to the strings."""
    e = direct()
    t = 0.0
    while t < 30.0 - bar_s:
        assert e.update(t, False, bar_s) == (0.0, False)
        t += bar_s
    e.update(t, False, bar_s)
    assert e.update(t + bar_s, False, bar_s)[0] < 0.0


def test_a_first_note_during_that_fade_brings_the_short_timing_in():
    e = direct()
    bar_s = 2.0
    assert e.update(0.0, False, bar_s) == (0.0, False)
    assert e.update(32.0, False, bar_s)[0] < 0.0
    assert e.update(34.0, True, bar_s) == (0.0, False)
    # 20 - (3 + 2 + 1.5) * 2 = 7 seconds from here, not 30
    assert e.update(40.0, False, bar_s) == (0.0, False)
    assert e.update(42.0, False, bar_s)[0] < 0.0


def test_a_tempo_cue_cannot_reshape_a_fade_already_running():
    e = direct()
    e.update(0.0, True, 2.0)
    assert e.update(8.0, False, 2.0)[0] == pytest.approx(-12.0)    # one bar of three
    assert e.update(10.0, False, 0.9)[0] == pytest.approx(-24.0)   # not one of six


# ------------------------------------------------------- the real transport, no device

@pytest.fixture
def consent(tmp_path, monkeypatch, plume_registry):
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only


def _played_rows(rec):
    rows = [json.loads(line) for line in (rec / "decisions.ndjson").read_text().splitlines()]
    return [r for r in rows if isinstance(r.get("bar"), int)]


@pytest.mark.parametrize("bpm", [115.0, 150.0])
def test_a_direct_take_ends_about_twenty_seconds_after_she_stops(tmp_path, consent, bpm):
    stops_at = 40.0
    human = _take(bpm=bpm, seconds=stops_at, room_s=80.0)
    # Her last strum rings on, so measure from the last bar the take heard her.
    with patched_module("sounddevice", fake_audio(human, devices=RIG, rate_x=12.0)):
        realtime.lead(prompt="midwest emo in C major", seed=0, tap_start=True, direct=True,
                      role=roles_mod.RoleParams(role="lead"), record_dir=str(tmp_path / "rec"),
                      status=False, socket_path=short_socket_path("fade"), max_minutes=0.7,
                      input_device="audio interface", input_channel=1)
    ending = json.loads((tmp_path / "rec" / "ending.json").read_text())
    assert ending["reason"] == realtime.END_SILENCE
    rows = _played_rows(tmp_path / "rec")
    heard = [r for r in rows if r["playing"]]
    bar_s = 240.0 / bpm
    # A playing row reads the bar two before it; the take ends when its last bar is over.
    last_heard_bar_ends = heard[-1]["t_s"] - bar_s
    over = rows[-1]["t_s"] + bar_s
    assert 16.0 <= over - last_heard_bar_ends <= 23.0
    assert any(r["fade_db"] < 0.0 for r in rows)
