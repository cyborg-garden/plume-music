"""Tap-started takes: drums, then a settled key, then the band."""
import numpy as np
import pytest

from plume.bandentry import (BAND, DRUMS, FULL, SONG_ARC, BandEntry, SilenceEnding,
                             count_start, first_bar_time, song_section, tap_tempo)
from plume.listen.harmony import HarmonyState


def h(key):
    return HarmonyState(t_s=0.0, key=key, key_conf=0.8, chord=None,
                        chord_conf=0.0, register_low_midi=None,
                        register_high_midi=None)


def run(entry, keys, *, start=0, playing=True):
    return [entry.update(start + i, h(k), playing) for i, k in enumerate(keys)]


def test_nothing_tonal_until_a_key_has_settled():
    states = run(BandEntry("lofi"), ["C major", "F major", "C major", "G major",
                                     "C major", "F major"])
    assert {s.stage for s in states} == {DRUMS}
    assert all("bass" in s.held_back and s.key is None for s in states)


def test_silence_is_not_evidence_of_a_key():
    """The bug she reported: a key decided before she had played a note."""
    e = BandEntry("lofi")
    states = run(e, ["F major"] * 12, playing=False)
    assert {s.stage for s in states} == {DRUMS}
    assert e.update(12, h("F major"), True).stage == DRUMS   # one real bar only


def test_band_then_melody_in_her_key():
    e = BandEntry("lofi", min_drum_bars=4, key_bars=4, melody_delay_bars=4,
                  after_first_heard_bars=4)
    states = run(e, ["D minor"] * 12)
    assert [s.stage for s in states[:3]] == [DRUMS] * 3
    assert states[4].stage == BAND and "bass" not in states[4].held_back
    assert "lead" in states[4].held_back
    assert states[8].stage == FULL and states[8].held_back == ()


def test_by_default_the_drums_get_eight_bars_alone():
    states = run(BandEntry("lofi"), ["D minor"] * 9)
    assert [s.stage for s in states] == [DRUMS] * 8 + [BAND]


def test_she_starts_late_and_the_band_waits_for_her():
    e = BandEntry("lofi", key_bars=2)
    run(e, ["D minor"] * 10, playing=False)
    states = run(e, ["D minor"] * 6, start=10, playing=True)
    assert [s.stage for s in states] == [DRUMS] * 5 + [BAND]


def test_a_count_starts_at_her_first_knock_even_if_it_was_quiet():
    """Her real count from 2026-11-02: confirmed from knock four."""
    taps = [28.09, 28.82, 29.62, 30.36]
    soft = [20.31, 20.9, 22.72, 24.39, 25.87, 26.61, 27.09, 27.32, 28.07]
    knocks = count_start(taps, soft)
    assert knocks[0] == pytest.approx(25.87)
    assert len(knocks) == 7
    bpm, _ = tap_tempo(knocks)
    assert knocks[0] + 8 * 60.0 / bpm == pytest.approx(31.87, abs=0.1)  # knock nine


def test_six_soft_knocks_start_it_but_four_do_not():
    from types import SimpleNamespace as F
    from plume.listen.features import CountInDetector

    def stream(n):
        hits = [2.0 + 0.75 * i for i in range(n)]
        return [F(t_s=i * 0.01, attack_breadth=0.6,
                  attack=(0.65 if any(abs(i * 0.01 - h) < 0.005 for h in hits)
                          else 0.02)) for i in range(1200)]

    def soft():
        return CountInDetector(floor=0.5, breadth=0.35, taps=6, waiting=True,
                               even_frac=0.15, bpm_range=(45.0, 170.0))
    assert soft().process(stream(4)) == []
    assert len(soft().process(stream(6))) == 6


def test_stray_hits_cannot_walk_the_count_off_its_grid():
    knocks = count_start([20.0, 21.0, 22.0, 23.0], [18.87, 17.74, 16.61])
    assert knocks[0] == 19.0, "the confirmed grid, not the stray hit's own time"


def test_count_start_stops_at_a_missing_beat_and_works_alone():
    assert count_start([10.0, 10.5, 11.0, 11.5], []) == [10.0, 10.5, 11.0, 11.5]
    assert count_start([10.0, 10.5, 11.0, 11.5], [9.5, 8.5])[0] == 9.5


def test_a_key_she_names_is_used_without_listening_for_one():
    e = BandEntry("lofi", min_drum_bars=4, known_key="F major",
                  after_first_heard_bars=4)
    states = run(e, ["A minor", "E minor", "D minor", "G major"] * 5, playing=True)
    assert [s.stage for s in states[:5]] == [DRUMS] * 4 + [BAND]
    assert {s.key for s in states[4:]} == {"F major"}, "it must never wander off"
    assert all(s.chord for s in states[4:])


def test_a_named_key_still_waits_for_her_to_play():
    e = BandEntry("lofi", min_drum_bars=4, known_key="F major")
    assert {s.stage for s in run(e, [None] * 12, playing=False)} == {DRUMS}
    states = run(e, [None] * 6, start=12, playing=True)
    assert [s.stage for s in states] == [DRUMS] * 5 + [BAND]
    assert states[5].key == "F major" and states[5].chord


def test_the_ear_tells_a_guitar_from_drums_alone():
    from plume.bandentry import GuitarEar
    flat = np.full(12, 1 / 12)
    f, c, g = 5, 0, 7
    played = np.full(12, 0.02)
    played[[f, c, g]] = [0.3, 0.25, 0.2]
    ear = GuitarEar()
    assert ear.bar([flat] * 40) is False and ear.notes() == ()
    assert ear.bar([played] * 40) is False, "one bar could be a ringing drum"
    assert ear.bar([flat] * 40) is False
    assert ear.bar([played] * 40) is True, "two within three bars is someone playing"
    assert ear.began_bars_ago == 0, "across a gap she starts on the confirming bar"
    assert ear.notes(3) == (f, c, g)
    assert ear.bar([]) is False and ear.last_bar_tonal is False
    for _ in range(4):
        ear.bar([flat] * 40)
    assert ear.bar([played] * 40) is False, "the confirmation expires"
    assert ear.bar([played] * 40) is True and ear.began_bars_ago == 1, \
        "two in a row: she began on the first"
    assert ear.last_bar_tonal is True, "but a fade can still hear this one bar"


def test_real_thresholds_separate_real_bars():
    """Measured on takes from 2026-11-02: none playing vs her guitar."""
    from plume.bandentry import TONAL_TOP3
    assert max(0.29, 0.31, 0.33, 0.37) < TONAL_TOP3 < min(0.46, 0.51, 0.76)


def test_drums_get_a_few_bars_alone_however_fast_the_key_settles():
    e = BandEntry("lofi", min_drum_bars=6, key_bars=1)
    states = run(e, ["C major"] * 7)
    assert [s.stage for s in states] == [DRUMS] * 6 + [BAND]


def test_band_stays_when_she_rests():
    e = BandEntry("lofi", min_drum_bars=4)
    run(e, ["C major"] * 6)
    rest = run(e, ["C major"] * 4, start=6, playing=False)
    assert all(s.stage != DRUMS and s.chord for s in rest)


def test_a_stale_key_belief_cannot_move_the_band_while_she_rests():
    e = BandEntry("lofi", min_drum_bars=4, release_bars=8)
    run(e, ["C major"] * 6)
    rest = run(e, ["G major"] * 12, start=6, playing=False)
    assert {s.key for s in rest} == {"C major"}


def test_a_rest_does_not_interrupt_a_real_key_change():
    e = BandEntry("lofi", min_drum_bars=4, release_bars=8)
    run(e, ["C major"] * 6)
    bar, last = 6, None
    for _ in range(9):
        last = e.update(bar, h("G major"), True)
        e.update(bar + 1, h("G major"), False)
        bar += 2
    assert last.key == "G major"


# The key waits the way the band's following does: a run of bars, and one bar
# that cannot say is no evidence.
def test_a_bar_the_tracker_cannot_name_does_not_reset_a_key_change():
    e = BandEntry("lofi", min_drum_bars=4, release_bars=8)
    for b in range(6):
        e.update(b, h("C major"), True)
    bar = 6
    for _ in range(4):
        e.update(bar, h("G major"), True)
        bar += 1
    e.update(bar, h(None), True)        # unsure, while playing: no evidence
    bar += 1
    last = None
    for _ in range(4):
        last = e.update(bar, h("G major"), True)
        bar += 1
    assert last.key == "G major"


def test_one_bar_back_home_still_holds_the_key():
    e = BandEntry("lofi", min_drum_bars=4, release_bars=8)
    for b in range(6):
        e.update(b, h("C major"), True)
    keys = ["G major"] * 7 + ["C major"] + ["G major"] * 7
    states = [e.update(6 + i, h(k), True) for i, k in enumerate(keys)]
    assert {s.key for s in states} == {"C major"}


def test_tap_tempo_ignores_one_rushed_knock():
    # her real knocks from 2026-11-02, clicks at 75 bpm
    bpm, nxt = tap_tempo([6.29, 6.93, 7.72, 8.53])
    assert bpm == pytest.approx(75.9, abs=0.1)
    assert nxt == pytest.approx(8.53 + 60 / bpm)


def test_a_slow_count_is_taken_at_its_word():
    """Doubling counts below 65 turned her real 51 bpm into 102."""
    bpm, _ = tap_tempo([10.0, 11.18, 12.36, 13.54])
    assert bpm == pytest.approx(50.85, abs=0.1)


def test_a_missed_first_downbeat_comes_in_a_whole_bar_later():
    assert first_bar_time(10.0, 2.0, 9.0, 0.317) == 10.0
    assert first_bar_time(10.0, 1.6, 9.7, 0.317) == pytest.approx(11.6)


def test_silence_waits_then_fades_then_ends():
    e = SilenceEnding(silence_s=30.0, fade_bars=4, floor_db=-36.0)
    assert e.update(0.0, True) == (0.0, False)
    assert e.update(29.0, False) == (0.0, False)
    assert e.update(30.0, False) == (-9.0, False)
    assert e.update(32.0, False) == (-18.0, False)
    assert e.update(34.0, False) == (-27.0, False)
    assert e.update(36.0, False) == (-36.0, False)  # faded, still listening
    assert e.update(38.0, False) == (-36.0, False)
    assert e.update(40.0, False) == (-36.0, True)


def test_a_return_at_the_very_end_of_the_fade_is_still_heard():
    e = SilenceEnding(silence_s=30.0)
    e.update(0.0, True)
    for t in (30.0, 32.0, 34.0, 36.0, 38.0):
        assert e.update(t, False)[1] is False
    assert e.update(40.0, True) == (0.0, False)


def test_playing_during_the_fade_brings_the_band_back():
    e = SilenceEnding(silence_s=30.0)
    e.update(0.0, True)
    assert e.update(32.0, False)[0] < 0.0
    assert e.update(34.0, True) == (0.0, False)
    assert e.update(36.0, False) == (0.0, False)   # the 30 s starts over


def test_the_clock_starts_on_the_first_bar_if_she_has_not_played():
    e = SilenceEnding(silence_s=30.0)
    assert e.update(100.0, False) == (0.0, False)
    assert e.update(129.0, False) == (0.0, False)
    assert e.update(130.0, False)[0] < 0.0


def test_waiting_for_knocks_needs_no_quiet_start_and_forgives_a_stray_hit():
    from types import SimpleNamespace as F
    from plume.listen.features import CountInDetector

    def frame(t, attack):
        return F(t_s=t, attack=attack, attack_breadth=0.9)
    # a noisy room, one stray hit, then four quiet-ish knocks at 0.65 s
    hits = [0.4, 0.9, 1.1, 1.8] + [3.0 + 0.65 * i for i in range(4)]
    frames = []
    for i in range(600):                # the quiet between hits, one frame per 10 ms
        t = i * 0.01
        frames.append(frame(t, 1.2 if any(abs(t - h) < 0.005 for h in hits)
                            else 0.02))
    assert CountInDetector(floor=1.0).process(frames) == []
    got = CountInDetector(floor=1.0, waiting=True).process(frames)
    assert got == pytest.approx([3.0, 3.65, 4.3, 4.95])


# ------------------------------------------------- the real transport, no device

def _take(sr=48000, bpm=75.0, seconds=45.0, room_s=0.0, strum_beats=12, rest=()):
    """Quiet, four knocks, a beat of nothing, a steady C major strum, then
    ``room_s`` of an empty room (the fake device loops whatever it is given).
    ``rest``: beats (from the first knock) between which she stops strumming."""
    import numpy as np
    rng = np.random.default_rng(1)
    x = np.zeros(int((seconds + room_s) * sr), dtype=np.float32)
    x += rng.standard_normal(x.shape[0]).astype(np.float32) * 3e-4
    period = 60.0 / bpm
    for i in range(4):
        at = int((2.0 + i * period) * sr)
        n = int(0.004 * sr)
        x[at:at + n] += rng.standard_normal(n).astype(np.float32) * 0.8
    t = np.arange(int(1.6 * sr)) / sr
    env = np.exp(-t * 2.5)
    chord = sum(np.sin(2 * np.pi * f * t) / (1 + 0.3 * j)
                for j, f in enumerate((130.81, 164.81, 196.0, 261.63, 329.63,
                                       392.0, 523.25)))
    chord = (chord * env * 0.08).astype(np.float32)
    at = 2.0 + strum_beats * period   # by default drums alone for two bars first
    while at + 2 < seconds:
        i = int(at * sr)
        if not (rest and 2.0 + rest[0] * period <= at < 2.0 + rest[1] * period):
            x[i:i + chord.shape[0]] += chord
        at += 2 * period
    return x


def test_tap_start_end_to_end_over_the_fake_device(tmp_path, monkeypatch,
                                                   plume_registry):
    import json
    from conftest import fake_audio, patched_module, short_socket_path
    from plume import realtime
    from plume import roles as roles_mod

    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
    fake = fake_audio(_take(seconds=90.0, room_s=120.0), rate_x=12.0)
    sock = short_socket_path("tap")
    with patched_module("sounddevice", fake):
        realtime.lead(prompt="lofi", seed=0, tap_start=True,
                      role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(tmp_path / "rec"),
                      socket_path=sock,
                      status=False, max_minutes=0.7)
    rows = [json.loads(line) for line in
            (tmp_path / "rec" / "decisions.ndjson").read_text().splitlines()]
    rows = [r for r in rows if isinstance(r.get("bar"), int)]
    assert rows, "no bars were planned"
    assert rows[0]["bar"] == 0 and rows[0]["t_s"] > 2.0 + 3 * 0.8
    assert all(r["bpm"] == pytest.approx(75.0, abs=1.5) for r in rows), \
        "the tapped tempo must hold for the whole take"
    assert rows[0]["entry"] == DRUMS and "bass" not in rows[0]["roles"]
    assert "kick" in rows[0]["roles"]
    entered = [r for r in rows if r["entry"] != DRUMS]
    assert entered, f"the band never came in: {[r['entry_reason'] for r in rows][-3:]}"
    assert entered[0]["bar"] >= 8 and "bass" in entered[0]["roles"]
    # drums and bass first, swelling in under the drums she was already hearing
    first = entered[0]["roles"]
    assert first["kick"] == 1.0 and first["bass"] < 0.5
    assert "pad" not in first and "keys" not in first
    assert entered[1]["roles"]["bass"] > first["bass"]
    assert entered[3]["roles"]["bass"] == 1.0 and "pad" in entered[3]["roles"]
    # on the top of one of her four-bar groups, counted from her first strum
    bar_s = rows[1]["t_s"] - rows[0]["t_s"]
    her_first = round((2.0 + 12 * 0.8 - rows[0]["t_s"]) / bar_s)
    assert (entered[0]["bar"] - her_first) % 4 == 0, (her_first, entered[0]["bar"])
    # the same notes can honestly read as their relative minor
    assert entered[0]["key"] in ("C major", "A minor")
    # no harmony claimed before she has played
    assert rows[0]["key"] is None and rows[0]["chord"] is None
    # the guitar stops at 45 s: it fades out and ends itself, well before the cap
    ending = json.loads((tmp_path / "rec" / "ending.json").read_text())
    assert ending["reason"] == realtime.END_SILENCE
    assert rows[-1]["fade_db"] < 0.0
    assert 110.0 < rows[-1]["t_s"] < 150.0


def test_after_a_restart_the_band_enters_on_her_group_end_to_end(tmp_path, monkeypatch,
                                                                plume_registry):
    """Her 18:37 take's shape: she strums in early, stops for two bars, starts again on
    plume's bar 3.  The band used to come in on bar 8, the middle of her group."""
    import json
    from conftest import fake_audio, patched_module, short_socket_path
    from plume import realtime
    from plume import roles as roles_mod

    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
    human = _take(seconds=90.0, room_s=120.0, strum_beats=4, rest=(12, 20))
    with patched_module("sounddevice", fake_audio(human, rate_x=12.0)):
        realtime.lead(prompt="lofi", seed=0, tap_start=True,
                      role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(tmp_path / "rec"), socket_path=short_socket_path("rs"),
                      status=False, max_minutes=0.5)
    rows = [json.loads(line) for line in
            (tmp_path / "rec" / "decisions.ndjson").read_text().splitlines()]
    rows = [r for r in rows if isinstance(r.get("bar"), int)]
    bar_s = rows[1]["t_s"] - rows[0]["t_s"]
    restart = round((2.0 + 20 * 0.8 - rows[0]["t_s"]) / bar_s)
    entered = next(r["bar"] for r in rows if r["entry"] != DRUMS)
    assert restart == 3 and entered == restart + 8


def test_at_real_speed_it_can_tell_she_is_playing(tmp_path, monkeypatch,
                                                  plume_registry):
    """The audit's P0.  A bar is planned a bar early, so reading the bar before
    it saw 0-1 frames: real playing looked silent, and the take faded out on her.
    Only real speed shows this, because pumping faster fills the window."""
    import json
    from conftest import fake_audio, patched_module, short_socket_path
    from plume import realtime
    from plume import roles as roles_mod

    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
    with patched_module("sounddevice", fake_audio(_take(bpm=150.0, seconds=40.0),
                                                  rate_x=1.0)):
        realtime.lead(prompt="lofi", seed=0, tap_start=True,
                      role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(tmp_path / "rec"),
                      socket_path=short_socket_path("real"),
                      status=False, max_minutes=0.6)
    rows = [json.loads(line) for line in
            (tmp_path / "rec" / "decisions.ndjson").read_text().splitlines()]
    rows = [r for r in rows if isinstance(r.get("bar"), int)]
    # drums alone for two bars, then she plays until the take is cut at 36 s
    heard = [r["playing"] for r in rows if 4 <= r["bar"] and r["t_s"] < 22.0]
    assert len(heard) >= 6 and sum(heard) >= len(heard) - 1, heard
    assert any(r["entry"] != DRUMS for r in rows), "the band never came in"


def test_the_song_moves_through_sections_and_repeats():
    names = [song_section(b)[0] for b in range(sum(n for _, n, _ in SONG_ARC))]
    assert names[0] == "verse" and names[16] == "lift" and "break" in names
    assert song_section(0) == song_section(sum(n for _, n, _ in SONG_ARC))
    assert song_section(16)[1] > song_section(0)[1]


def test_a_break_rests_the_kick_and_snare_and_a_verse_saves_the_high_pad():
    breaks = [song_section(b) for b in range(120) if song_section(b)[0] == "break"]
    assert breaks and all({"kick", "snare"} <= set(s[2]) for s in breaks)
    assert "shimmer" in song_section(0)[2] and song_section(16)[2] == ()


def test_the_ending_signals_first_and_only_then_fades():
    from plume.bandentry import OUTRO_FADE_BARS, OUTRO_SIGNAL_BARS, outro
    assert outro(149.0, 150.0, 3.0) is None
    assert outro(500.0, None, 3.0) is None
    assert outro(150.0, 150.0, 3.0) == (0, 0.0, False)
    assert outro(150.0 + 3.0 * (OUTRO_SIGNAL_BARS - 2), 150.0, 3.0)[1] == 0.0
    last = outro(150.0 + 3.0 * (OUTRO_SIGNAL_BARS + OUTRO_FADE_BARS - 1), 150.0, 3.0)
    assert last[1] == -36.0 and last[2] is True


def test_a_timed_song_signals_and_ends_itself(tmp_path, monkeypatch, plume_registry):
    import json
    from conftest import fake_audio, patched_module, short_socket_path
    from plume import realtime
    from plume import roles as roles_mod

    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
    with patched_module("sounddevice", fake_audio(_take(seconds=200.0), rate_x=12.0)):
        realtime.lead(prompt="midwest emo in F major", seed=0, tap_start=True,
                      wind_down_s=60.0, role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(tmp_path / "rec"),
                      socket_path=short_socket_path("end"), status=False, max_minutes=0.5)
    rows = [json.loads(line) for line in
            (tmp_path / "rec" / "decisions.ndjson").read_text().splitlines()]
    rows = [r for r in rows if "entry" in r]
    ending = json.loads((tmp_path / "rec" / "ending.json").read_text())
    assert ending["reason"] == realtime.END_SONG
    out = [r for r in rows if r["section"] == "outro"]
    assert out and "crash" in out[0]["roles"] and "crash" not in out[1]["roles"]
    assert all("crash" not in r["roles"] for r in rows if r["section"] != "outro")
    assert out[0]["fade_db"] == 0.0 and out[-1]["fade_db"] == -36.0


def test_named_key_is_read_from_the_prompt():
    from plume.realtime import _named_key
    assert _named_key("lofi midwest emo in F major") == "F major"
    assert _named_key("midwest emo in A minor") == "A minor"
    assert _named_key("lofi midwest emo") is None
    assert _named_key(None) is None


# ------------------------------------------- entering on her phrase, not plume's count
#
# Juniper, 2026-11-06: "the chords came in not at what felt like the end of a group of
# measures".  The band entered whenever the key settled, which was bar 8 of plume's
# count and bar 6 of hers.  Her readings arrive two bars late in the live loop, so each
# one says which bar it heard.

def _entry_bar(entry, readings, *, lag=2, key="D minor", bars=24):
    """The bar the band comes in, given which plume bars she played in."""
    for bar in range(bars):
        heard = bar - lag
        state = entry.update(bar, h(key), heard in readings, heard_bar=heard)
        if state.stage != DRUMS:
            return bar
    return None


def test_the_band_enters_on_her_four_bar_group_after_a_restart():
    """Her take of 2026-11-06 18:37: played into bar 0, silent in bar 1, restarted on 2."""
    e = BandEntry("lofi", key_bars=4, after_first_heard_bars=6, phrase_bars=4)
    played = {0} | set(range(2, 30))
    assert _entry_bar(e, played) == 10, "her bar 9, the top of her third group"


def test_a_late_first_note_sets_the_grid():
    """First heard in plume bar 2 after two quiet bars: ready on 9, her group turns on 10."""
    e = BandEntry("lofi", key_bars=4, after_first_heard_bars=5, phrase_bars=4)
    assert _entry_bar(e, set(range(2, 30))) == 10


def test_a_named_key_enters_on_her_phrase_too():
    e = BandEntry("lofi", known_key="F major", after_first_heard_bars=5, phrase_bars=4)
    assert _entry_bar(e, set(range(1, 30)), key=None) == 9


def test_once_in_the_band_ignores_later_restarts():
    e = BandEntry("lofi", key_bars=4, after_first_heard_bars=4, phrase_bars=4)
    played = set(range(0, 30)) - {10}
    assert _entry_bar(e, played) == 8
    assert e.update(13, h("D minor"), True, heard_bar=11).stage != DRUMS


def test_without_a_phrase_length_it_enters_when_ready():
    e = BandEntry("lofi", key_bars=4, after_first_heard_bars=6)
    assert _entry_bar(e, {0} | set(range(2, 30))) == 8


# ------------------------------------------- the band comes in over a few bars
#
# Juniper, 2026-11-06: it "didnt come in smooth".  The whole band arrived on one bar,
# about 27 dB over the click, with the session's ramp_bars never applied.

def test_drums_and_bass_first_then_the_chords_then_the_melody():
    e = BandEntry("lofi", min_drum_bars=4, key_bars=4, melody_delay_bars=4,
                  after_first_heard_bars=4)
    states = run(e, ["D minor"] * 10)
    first, second, third = states[4], states[5], states[6]
    assert first.stage == BAND and "bass" not in first.held_back
    assert {"pad", "keys", "lead"} <= set(first.held_back)
    assert set(second.held_back) == set(first.held_back)
    assert "pad" not in third.held_back and "keys" not in third.held_back
    assert "lead" in third.held_back
    assert states[8].stage == FULL and states[8].held_back == ()


def test_the_entry_ramps_up_from_under_her():
    from plume.bandentry import entry_ramp_db
    ramp = [entry_ramp_db(b, under_db=-8.0, ramp_bars=3) for b in range(6)]
    assert ramp[0] == -8.0 and ramp[3:] == [0.0, 0.0, 0.0]
    assert ramp[0] < ramp[1] < ramp[2] < ramp[3]
    assert entry_ramp_db(0, under_db=-8.0, ramp_bars=0) == 0.0

