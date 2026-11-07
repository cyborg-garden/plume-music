"""The band finds her within a few seconds (Juniper, 2026-11-06).

"it did not get quieter when i was playing quieter tho."  On that take (60 bpm, a
direct input) she went about 14 dB softer for four bars, and the band's first
move was heard sixteen seconds later, after she had come back up.  The patience stays,
"so that if i just make like a little mistake it doesnt assume thats the whole
direction", but it is counted in half bars and seconds, the planner reads her half a
bar sooner, and the answer is bigger and quicker.  Pure, so no device.
"""
import pytest

from plume.bandentry import (FOLLOW_GAIN_MIN_DB, FOLLOW_GAIN_RETURN_STEP_DB,
                             FOLLOW_GAIN_STEP_DB, FOLLOW_HOLD_MIN_S, FOLLOW_LEARN_BARS,
                             BandFollow, follow_hold_bars)
from plume.realtime import follow_parts, plan_lead_s

NORMAL_DB = -21.0


def bar_s(bpm):
    return 240.0 / bpm


def learned(*, trusted=True):
    f = BandFollow(trusted=trusted)
    for _ in range(FOLLOW_LEARN_BARS):
        f.bar(NORMAL_DB, 1.0 if trusted else None, playing=True, bar_s=bar_s(80))
    return f


def halves(f, bpm, levels):
    """One planner call per bar, each carrying that bar's two half-bar readings."""
    out = []
    for a, b in levels:
        whole = max(x for x in (a, b) if x is not None) if (a, b) != (None, None) else None
        out.append(f.bar(whole, 1.0 if f.trusted else None, playing=whole is not None,
                         parts=[a, b], bar_s=bar_s(bpm)))
    return out


# ------------------------------------------------------------ the hold, in time

@pytest.mark.parametrize("bpm,expected", [(60, 1.5), (80, 1.5), (120, 1.5), (160, 2.0)])
def test_the_hold_is_a_bar_and_a_half_but_never_under_three_seconds(bpm, expected):
    held = follow_hold_bars(bar_s(bpm), trusted=True)
    assert held == expected
    assert held * bar_s(bpm) >= FOLLOW_HOLD_MIN_S - 1e-9 or held == 2.0
    assert held > 1.0                     # one odd bar can never be a move


@pytest.mark.parametrize("bpm", [60, 80, 120, 160])
def test_a_room_mic_still_waits_longer(bpm):
    assert follow_hold_bars(bar_s(bpm), trusted=False) > follow_hold_bars(bar_s(bpm),
                                                                          trusted=True)


def test_three_soft_halves_move_the_band():
    f = learned()
    soft = NORMAL_DB - 14
    states = halves(f, 80, [(NORMAL_DB, soft), (soft, soft)])
    assert states[0].gain_db == 0.0
    assert states[1].moved == ("softer",) and states[1].gain_db < 0.0


def test_one_odd_bar_moves_nothing():
    f = learned()
    soft = NORMAL_DB - 14
    states = halves(f, 80, [(soft, soft)] + [(NORMAL_DB, NORMAL_DB)] * 8)
    assert all(s.gain_db == 0.0 and not s.moved for s in states)


def test_one_odd_hit_in_a_soft_passage_does_not_bring_it_back():
    f = learned()
    soft = NORMAL_DB - 14
    halves(f, 80, [(soft, soft), (soft, soft)])
    down = halves(f, 80, [(soft, soft), (NORMAL_DB + 3, soft), (soft, soft)])
    assert all("louder" not in s.moved for s in down)
    assert down[-1].gain_db < down[0].gain_db


def test_a_silent_half_is_no_evidence_either_way():
    f = learned()
    soft = NORMAL_DB - 14
    states = halves(f, 80, [(soft, None), (None, soft), (soft, NORMAL_DB)])
    assert states[-1].moved == ("softer",)


# ------------------------------------------------------------ the answer

def test_a_big_drop_gets_a_big_answer_and_quickly():
    f = learned()
    soft = NORMAL_DB - 14
    states = halves(f, 80, [(soft, soft)] * 6)
    assert states[-1].gain_db <= -8.0              # half of 14 capped at 6 was too small
    assert FOLLOW_GAIN_MIN_DB <= states[-1].gain_db
    moving = [s for s in states if s.gain_db < 0.0]
    assert moving[0].gain_db <= -FOLLOW_GAIN_STEP_DB + 1e-9   # audible in its first bar


def test_a_small_drop_still_gets_half():
    f = learned()
    states = halves(f, 80, [(NORMAL_DB - 5, NORMAL_DB - 5)] * 6)
    assert states[-1].gain_db == pytest.approx(-2.5, abs=0.1)


def test_coming_back_up_brings_the_band_back_quickly():
    f = learned()
    soft = NORMAL_DB - 14
    down = halves(f, 80, [(soft, soft)] * 6)[-1].gain_db
    back = halves(f, 80, [(NORMAL_DB, NORMAL_DB)] * 6)
    first_up = next(i for i, s in enumerate(back) if s.gain_db > down)
    assert first_up <= 1                         # as soon as the hold allows
    assert back[-1].gain_db == pytest.approx(0.0, abs=0.01)
    for a, b in zip(back, back[1:]):
        assert b.gain_db - a.gain_db <= FOLLOW_GAIN_RETURN_STEP_DB + 1e-9


# ------------------------------------------------------------ reading her sooner

def _frames(t0, t1, db, step=0.02):
    n = int(round((t1 - t0) / step))
    return [(t0 + i * step, db, False) for i in range(n)]


def test_follow_parts_reads_each_half_once_and_the_next_bars_first_half_early():
    bar = 3.0
    evs = _frames(0.0, 1.5, -20.0) + _frames(1.5, 3.0, -30.0)
    later = _frames(3.0, 4.5, -36.0) + _frames(4.5, 4.7, -10.0)
    parts, fed = follow_parts(evs, later, 0.0, bar, 2 * bar, fed_until=None,
                              heard_until=4.6, silence_rms_db=-45.0, ref_db=None)
    assert [round(p) for p in parts] == [-20, -30, -36]
    assert fed == pytest.approx(4.5)
    # Next bar: its first half was already read, so only its second half is new.
    evs2 = later + _frames(4.7, 6.0, -36.0)
    parts2, fed2 = follow_parts(evs2, [], bar, 2 * bar, 3 * bar, fed_until=fed,
                                heard_until=6.05, silence_rms_db=-45.0, ref_db=None)
    assert len(parts2) == 1 and fed2 == pytest.approx(6.0)


def test_follow_parts_waits_for_a_half_that_has_not_all_arrived():
    evs = _frames(0.0, 3.0, -20.0)
    later = _frames(3.0, 4.0, -36.0)
    parts, fed = follow_parts(evs, later, 0.0, 3.0, 6.0, fed_until=None,
                              heard_until=4.0, silence_rms_db=-45.0, ref_db=None)
    assert len(parts) == 2 and fed == pytest.approx(3.0)


def test_a_silent_half_reads_as_none():
    evs = _frames(0.0, 1.5, -20.0) + _frames(1.5, 3.0, -80.0)
    parts, _ = follow_parts(evs, [], 0.0, 3.0, 6.0, fed_until=None, heard_until=3.0,
                            silence_rms_db=-45.0, ref_db=-20.0)
    assert parts[0] is not None and parts[1] is None


# ------------------------------------------------------------ the planner's lead

def test_the_planner_runs_half_a_bar_ahead_when_it_has_time():
    lead = plan_lead_s(3.0, plan_cost_s=0.085, pace=1.0, comp_s=0.085)
    assert lead == pytest.approx(1.5 - 0.1, abs=1e-6)


def test_the_planner_keeps_a_whole_bar_when_planning_is_slow_for_the_pace():
    # The fake device runs twelve times real time: half a bar is not enough there.
    assert plan_lead_s(2.0, plan_cost_s=0.085, pace=12.0, comp_s=0.0) == 2.0
    assert plan_lead_s(2.0, plan_cost_s=None, pace=None, comp_s=0.0) == 2.0


def test_the_lead_is_never_more_than_a_bar():
    assert plan_lead_s(1.5, plan_cost_s=5.0, pace=1.0, comp_s=0.1) == 1.5


# ------------------------------------------------------------ her take, in numbers

@pytest.mark.parametrize("bpm,max_s", [(80, 7.5), (120, 5.0)])
def test_seconds_from_her_change_to_the_band_moving(bpm, max_s):
    """A sustained drop starting on a bar line, through the planner's timing: bar N
    is planned half a bar before it starts, from what has been heard by then, and its
    gain is heard from its start."""
    f, b = learned(), bar_s(bpm)
    change_bar = 10
    soft = NORMAL_DB - 14
    heard = {}
    fed = 0                                   # halves fed so far, as an index
    for n in range(4, 20):
        t_plan = (n - 0.5) * b                # half a bar ahead
        upto = int((t_plan + 1e-9) // (b / 2))    # halves complete by then
        parts = [soft if h >= 2 * change_bar else NORMAL_DB for h in range(fed, upto)]
        fed = upto
        s = f.bar(parts[-1] if parts else None, 1.0, playing=bool(parts), parts=parts,
                  bar_s=b)
        heard[n] = s.gain_db
    first = min(n for n, g in heard.items() if g < 0.0)
    assert (first - change_bar) * b <= max_s


def test_stream_pace_is_the_fastest_seen_and_waits_to_say():
    from plume.realtime import _StreamPace
    now = [0.0]
    pace = _StreamPace(clock=lambda: now[0])
    head = 0.0
    for _ in range(40):                # a second at real time, then twelve times faster
        pace.note(head)
        step = 0.1
        head += step * (1.0 if now[0] < 1.0 else 12.0)
        now[0] += step
        if now[0] < 3.0:
            assert pace.rate() is None
    assert pace.rate() == pytest.approx(12.0, rel=0.05)


# ------------------------------------------------------------ it reaches the speakers

PAIR_RIG = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4},
            {"name": "Laptop Speakers", "max_input_channels": 0, "max_output_channels": 2}]


def test_the_band_gets_quieter_in_what_the_output_device_plays(tmp_path, monkeypatch,
                                                              plume_registry):
    """The follow gain is in the samples sent to another output device, not only in the
    log: the output pair is the common setup (an interface in, built-in speakers out)."""
    import json

    import numpy as np
    from conftest import fake_audio, patched_module, short_socket_path
    from test_bandentry import _take

    from plume import realtime
    from plume import roles as roles_mod

    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only
    bpm, soft_at = 115.0, 50.0
    human = _take(bpm=bpm, seconds=130.0)
    human[int(soft_at * 48000):] *= 0.2                     # 14 dB softer from here on
    # 4x, not 12x: this reads the output by the sample against the log's times, and a
    # slower machine (a Linux container measured it) falls behind a 12x playhead, so
    # the gain lands late in the samples while the log says it is on time.
    sd = fake_audio(human, devices=PAIR_RIG, rate_x=4.0)
    with patched_module("sounddevice", sd):
        realtime.lead(prompt="midwest emo in C major", seed=0, tap_start=True, direct=True,
                      role=roles_mod.RoleParams(role="lead"), record_dir=str(tmp_path / "rec"),
                      status=False, socket_path=short_socket_path("pair"), max_minutes=1.5,
                      input_device="audio interface", input_channel=1,
                      output_device="laptop speakers")
    assert sd.opened[0]["device"] == (0, 1)               # the pair, not one device
    out = np.concatenate(sd.stream.captured).mean(axis=1).astype(np.float64)
    rows = [json.loads(x) for x in (tmp_path / "rec" / "decisions.ndjson").read_text().splitlines()]
    rows = [r for r in rows if isinstance(r.get("bar"), int) and r.get("entry") == "full"
            and r["fade_db"] == 0.0 and r["t_s"] < 120.0]

    def level(rs):
        bar = 240.0 / bpm
        seg = np.concatenate([out[int(r["t_s"] * 48000):int((r["t_s"] + bar) * 48000)]
                              for r in rs])
        return 10.0 * np.log10(np.mean(seg * seg) + 1e-20)

    steady = [r for r in rows if r["t_s"] < soft_at]
    down = [r for r in rows if r["t_s"] > soft_at and r["follow"]["gain_db"] <= -6.0]
    assert steady and down, "the take did not reach both halves"
    logged = np.mean([r["follow"]["gain_db"] for r in down])
    heard = level(down) - level(steady)
    assert heard < -4.0 and abs(heard - logged) < 2.0, (heard, logged)


def test_a_take_at_real_pace_plans_half_a_bar_ahead(tmp_path, monkeypatch, plume_registry):
    """At a pace where planning has time (here twice real time), the planner settles
    half a bar ahead, so the band hears the first half of the bar now playing."""
    import json

    from conftest import fake_audio, patched_module, short_socket_path
    from test_bandentry import _take

    from plume import realtime
    from plume import roles as roles_mod

    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only
    bpm = 115.0
    with patched_module("sounddevice", fake_audio(_take(bpm=bpm, seconds=40.0),
                                                  devices=PAIR_RIG, rate_x=2.0)):
        realtime.lead(prompt="midwest emo in C major", seed=0, tap_start=True, direct=True,
                      role=roles_mod.RoleParams(role="lead"), record_dir=str(tmp_path / "rec"),
                      status=False, socket_path=short_socket_path("half"), max_minutes=0.6,
                      input_device="audio interface", input_channel=1)
    rows = [json.loads(x) for x in (tmp_path / "rec" / "decisions.ndjson").read_text().splitlines()]
    ahead = [r["follow"]["ahead_s"] for r in rows if isinstance(r.get("bar"), int)]
    bar = 240.0 / bpm
    assert ahead[0] == pytest.approx(bar, abs=0.01)       # nothing measured yet: a bar
    assert ahead[-1] == pytest.approx(bar / 2 - 0.1, abs=0.01)


# ------------------------------------------------------------ honest about moves

def test_a_softer_turn_still_above_the_cap_is_not_reported_as_a_move():
    """Her 2026-11-06 take: loud from bar 9, the band sat at its +3 dB cap, and at bar
    37 she got a little softer but stayed far above normal.  The gain could not move,
    so the log and the recap must not say the band came down."""
    f = learned()
    loud = NORMAL_DB + 20
    for _ in range(8):
        halves(f, 80, [(loud, loud)])
    assert f.gain_db == pytest.approx(f._targets()[0])
    at_cap = f.gain_db
    states = halves(f, 80, [(loud - 5, loud - 5)] * 4)
    assert all(s.gain_db == at_cap for s in states)
    assert not any("softer" in s.moved for s in states)

