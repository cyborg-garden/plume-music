"""Hear me never sends a buzzing cable to the speakers, and a dead input is said.

Nothing here opens a device: the callback helper and the analysis-side guard are
driven block by block, as the two threads would drive them.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from plume import guitake, inputguard, realtime
from plume.inputguard import BUZZ_LINE, SILENT_LINE, InputGuard

SR = realtime.SR
BLOCK = realtime.BLOCK


def _db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def strums(seconds: float, *, peak_db: float = -6.0, every_s: float = 0.5,
           low_heavy: bool = False, seed: int = 0) -> np.ndarray:
    """Plucked-string chords with harmonics, strummed, scaled to ``peak_db``."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    out = np.zeros(n)
    strings = [82.41, 110.0, 146.83, 196.0, 246.94, 329.63]
    if low_heavy:
        strings = [82.41, 110.0, 164.81]            # a low power chord
    t_strum = 0.0
    down = True
    while t_strum < seconds:
        order = strings if down else strings[::-1]
        for i, f in enumerate(order):
            start = int((t_strum + i * 0.008) * SR)
            if start >= n:
                break
            t = np.arange(n - start) / SR
            note = np.zeros_like(t)
            for k in range(1, 13):
                if f * k > 8000:
                    break
                amp = (1.0 / k ** 2 if low_heavy else 1.0 / k) * rng.uniform(0.7, 1.0)
                note += amp * np.sin(2 * np.pi * f * k * t + rng.uniform(0, 2 * np.pi)) \
                    * np.exp(-t * (1.5 + 0.4 * k))
            out[start:] += note * np.minimum(1.0, t / 0.002)
        t_strum += every_s
        down = not down
    return (out / np.max(np.abs(out)) * _db_to_lin(peak_db)).astype(np.float32)


def plucks(seconds: float, notes, *, peak_db: float = -6.0, every_s: float = 0.25,
           decay: float = 2.0, mute_s: float | None = None, seed: int = 0) -> np.ndarray:
    """Single notes or chords picked one after another: a bass DI line, a drop-tuned
    chug (``mute_s`` cuts each one short, as a palm mute does)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    out = np.zeros(n)
    t0, i = 0.0, 0
    while t0 < seconds:
        start = int(t0 * SR)
        length = n - start if mute_s is None else min(n - start, int(mute_s * SR))
        t = np.arange(length) / SR
        chord = notes[i % len(notes)]
        for f in (chord if isinstance(chord, tuple) else (chord,)):
            for k in range(1, 16):
                if f * k > 6000:
                    break
                out[start:start + length] += (rng.uniform(0.6, 1.0) / k
                                              * np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.3))
                                              * np.exp(-t * (decay + 0.5 * k)))
        t0 += every_s
        i += 1
    return (out / np.max(np.abs(out)) * _db_to_lin(peak_db)).astype(np.float32)


def hum(seconds: float, *, level: float = 1.0, hz: float = 50.0, clip: bool = True,
        harmonics=((3, 0.3),)) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = np.sin(2 * np.pi * hz * t)
    for k, a in harmonics:
        x = x + a * np.sin(2 * np.pi * k * hz * t)
    x = x / np.max(np.abs(x)) * (1.4 if clip else 1.0) * level
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def buzz(seconds: float, *, base: float = 60.0, amps=(1.0,), level: float = 0.89,
         pulse_duty: float | None = None, offset_hz: float = 0.0) -> np.ndarray:
    """A grid buzz: harmonic k of ``base`` at ``amps[k-1]``, or a narrow pulse train."""
    t = np.arange(int(seconds * SR)) / SR
    f = base + offset_hz
    if pulse_duty is not None:
        x = ((t * f) % 1.0 < pulse_duty).astype(float)
        x -= x.mean()
    else:
        x = sum(a * np.sin(2 * np.pi * (k + 1) * f * t + 0.7 * k) for k, a in enumerate(amps))
    return (x / np.max(np.abs(x)) * level).astype(np.float32)


def quiet(seconds: float, db: float = -65.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * _db_to_lin(db)).astype(np.float32)


def run(x: np.ndarray, band: np.ndarray | None = None, *, guard: InputGuard | None = None):
    """Drive the callback and the analysis thread a block at a time.

    Returns (output, monitor gain at each block's end, guard, lines said).  The
    analysis thread sees each block after the callback that delivered it: the
    order the rings impose.
    """
    guard = guard or InputGuard(SR)
    n = (len(x) // BLOCK) * BLOCK
    out = np.zeros((n, 2), dtype=np.float32)
    gains, lines = [], []
    for i in range(0, n, BLOCK):
        o = out[i:i + BLOCK]
        if band is not None:
            o[:, 0] = band[i:i + BLOCK]
            o[:, 1] = band[i:i + BLOCK]
        realtime.mix_monitor(o, x[i:i + BLOCK], guard.slot)
        gains.append(float(guard.slot[1]))
        said = guard.feed(x[i:i + BLOCK])
        if said:
            lines.append((i / SR, said))
    return out, np.array(gains), guard, lines


# ------------------------------------------------------------------ the ceiling

def test_full_scale_input_on_a_loud_band_never_passes_the_ceiling():
    t = np.arange(SR) / SR
    band = (0.95 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    x = np.sign(np.sin(2 * np.pi * 50 * t)).astype(np.float32)     # full-scale square
    out = np.zeros((SR // BLOCK * BLOCK, 2), dtype=np.float32)
    slot = inputguard.monitor_slot()
    for i in range(0, out.shape[0], BLOCK):
        o = out[i:i + BLOCK]
        o[:, 0] = band[i:i + BLOCK]
        o[:, 1] = band[i:i + BLOCK]
        slot[0], slot[1] = 0.0, 1.0      # no duck: the ceiling alone must hold
        realtime.mix_monitor(o, x[i:i + BLOCK], slot)
    assert np.max(np.abs(out)) <= realtime.MONITOR_CEILING + 1e-6
    assert np.all(np.isfinite(out))


def test_below_the_knee_the_sum_is_untouched():
    t = np.arange(BLOCK) / SR
    band = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    x = (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    o = np.stack([band, band], axis=1)
    realtime.mix_monitor(o, x, inputguard.monitor_slot())
    np.testing.assert_allclose(o[:, 0], band + x, atol=1e-6)
    np.testing.assert_allclose(o[:, 1], band + x, atol=1e-6)


# ------------------------------------------------------------------ the duck

HUMS = {
    "50hz-clipping": dict(level=1.0, clip=True),
    "50hz-half-scale": dict(level=0.5, clip=False),
    "50hz-at--20-dbfs": dict(level=0.1, clip=False),
    "60hz-with-harmonics": dict(hz=60.0, clip=False, level=0.7,
                                harmonics=((2, 0.5), (3, 0.4), (4, 0.3))),
    "120hz-buzz-full-scale": dict(hz=120.0, clip=True,
                                  harmonics=((2, 0.5), (3, 0.33), (4, 0.25), (5, 0.2))),
    "100hz-buzz": dict(hz=100.0, clip=False, level=0.5, harmonics=((2, 0.4), (3, 0.3))),
}
BUZZES = {
    "120hz-led-weak-60": dict(base=60.0, amps=(0.3, 1.0)),
    "100hz-led-weak-50": dict(base=50.0, amps=(0.3, 1.0)),
    "180hz-led": dict(base=60.0, amps=(0.5, 0.0, 1.0)),
    "150hz-led-on-50": dict(base=50.0, amps=(0.3, 0.0, 1.0)),
    "spread-over-harmonics": dict(base=60.0, amps=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)),
    "120hz-led-at--12-dbfs": dict(base=60.0, amps=(0.3, 1.0), level=0.25),
    "pulse-10pc": dict(base=60.0, pulse_duty=0.10),
    "pulse-5pc": dict(base=50.0, pulse_duty=0.05),
    "generator-50.4hz": dict(base=50.0, amps=(1.0, 0.0, 0.3), offset_hz=0.4),
    "generator-59.6hz": dict(base=60.0, amps=(1.0, 0.4), offset_hz=-0.4),
}


@pytest.mark.parametrize("kind", [*HUMS, *BUZZES])
def test_a_hum_ducks_the_monitor_says_so_and_comes_back(kind):
    # She plays loud (-6 dBFS peaks), so a hum well under her level counts too.
    before = strums(2.0, peak_db=-6.0)
    burst = hum(2.5, **HUMS[kind]) if kind in HUMS else buzz(2.5, **BUZZES[kind])
    after = strums(3.0, peak_db=-6.0, seed=3)
    x = np.concatenate([before, burst, after])
    _, gains, guard, lines = run(x)
    onset = len(before) // BLOCK
    ducked = np.nonzero(gains[onset:] == 0.0)[0]
    assert ducked.size, "never ducked"
    # The hum test needs an 80 ms window of it before it can tell a hum from a
    # sustained low note, so the duck lands ~100 ms in.  The ceiling holds the
    # speakers at -1 dBFS meanwhile.
    assert (ducked[0] + 1) * BLOCK / SR <= 0.130
    end = (len(before) + len(burst)) // BLOCK
    assert np.all(gains[onset + ducked[0]:end - 1] == 0.0)
    back = np.nonzero(gains[end:] == 1.0)[0]
    assert back.size and back[0] * BLOCK / SR <= 0.9
    assert [line for _, line in lines] == [BUZZ_LINE]
    assert guard.fault is None


def test_a_hum_that_keeps_the_monitor_off_is_always_said():
    # A generator warbling quickly sits at the edge of the test: caught often enough
    # to hold Hear me off, too unevenly for the hop counts.  Held off is said.
    t = np.arange(4 * SR) / SR
    x = np.sin(2 * np.pi * np.cumsum(50.0 + 0.5 * np.sin(2 * np.pi * 3.0 * t)) / SR)
    _, gains, guard, lines = run(np.concatenate([quiet(0.5), (0.7 * x).astype(np.float32)]))
    assert np.sum(gains < 1.0) * BLOCK / SR > 1.0
    assert [line for _, line in lines] == [BUZZ_LINE]


def test_buzz_output_stays_under_the_ceiling_with_hear_me_on():
    t = np.arange(4 * SR) / SR
    band = (0.6 * np.sin(2 * np.pi * 196 * t)).astype(np.float32)
    x = np.concatenate([strums(1.0, peak_db=-20.0), hum(3.0)])
    out, _, _, _ = run(x, band[:len(x)])
    assert np.max(np.abs(out)) <= realtime.MONITOR_CEILING + 1e-6


PLAYING = {
    "open-chords": lambda: strums(8.0, peak_db=-6.0),
    "fast-open-chords": lambda: strums(6.0, peak_db=-6.0, every_s=0.25, seed=5),
    "low-power-chords": lambda: strums(8.0, peak_db=-6.0, low_heavy=True),
    # bass DI, including G1 (49 Hz), a hertz from 50 Hz mains, held long
    "bass-di": lambda: plucks(10.0, [41.2, 55.0, 73.42, 49.0, 61.74, 51.91, 58.27],
                              every_s=0.6, decay=1.0),
    "bass-di-held": lambda: plucks(8.0, [49.0, 55.0], every_s=2.0, decay=0.6),
    "drop-c-chugs": lambda: plucks(6.0, [(65.41, 98.0, 130.81)], every_s=0.125,
                                   mute_s=0.1, decay=6.0),
    "drop-d-palm-mutes": lambda: plucks(6.0, [(73.42, 110.0, 146.83)], every_s=0.125,
                                        mute_s=0.09, decay=8.0),
    "drop-d-ringing": lambda: plucks(6.0, [(73.42, 110.0, 146.83)], every_s=1.0, decay=1.0),
}


#: Notes near a mains line, at A440: held perfectly steady (a sine, an organ-like
#: tone with harmonics), none of them may be taken for hum.
NEAR_MAINS_HZ = (49.0, 58.27, 61.74, 98.0, 116.54, 123.47, 146.83, 196.0, 233.08, 246.94)


@pytest.mark.parametrize("organ", [False, True], ids=["sine", "organ"])
@pytest.mark.parametrize("hz", NEAR_MAINS_HZ)
def test_a_steady_note_near_mains_is_not_hum(hz, organ):
    t = np.arange(4 * SR) / SR
    amps = (1.0, 0.6, 0.4, 0.3, 0.2) if organ else (1.0,)
    x = sum(a * np.sin(2 * np.pi * (k + 1) * hz * t) for k, a in enumerate(amps))
    x = (x / np.max(np.abs(x)) * 0.7).astype(np.float32)
    _, gains, guard, lines = run(np.concatenate([quiet(0.5), x]))
    assert np.all(gains == 1.0)
    assert lines == [] and guard.fault is None


@pytest.mark.parametrize("drive", [1.0, 1.2, 1.5, 2.5], ids=["clean", "x1.2", "x1.5", "x2.5"])
@pytest.mark.parametrize("kind", list(PLAYING))
def test_loud_playing_never_ducks(kind, drive):
    # Resting first: a hard note out of silence is the biggest jump she makes.  And a
    # hot input gain clips on every note: that is hers, not a fault.
    x = np.concatenate([quiet(2.0), PLAYING[kind](), quiet(1.0)])
    x = np.clip(x * drive * 2.0, -1.0, 1.0)      # peaks at 0 dBFS and over
    _, gains, guard, lines = run(x)
    assert np.all(gains == 1.0)
    assert lines == []
    assert guard.fault is None


# ------------------------------------------------------------------ a dead input

def test_a_dead_input_is_said_after_four_bars_and_clears_when_she_plays():
    guard = InputGuard(SR)
    guard.bar_s = 2.4                       # 100 bpm: four bars is 9.6 s
    _, _, guard, lines = run(strums(3.0, peak_db=-12.0), guard=guard)
    assert lines == [] and guard.fault is None
    _, _, guard, lines = run(quiet(9.0), guard=guard)
    assert lines == [] and guard.fault is None, "said before four bars"
    _, _, guard, lines = run(quiet(1.0), guard=guard)
    assert [line for _, line in lines] == [SILENT_LINE]
    assert guard.fault == "silent"
    _, _, guard, lines = run(quiet(3.0), guard=guard)
    assert lines == [], "said only once"
    _, gains, guard, lines = run(strums(1.0, peak_db=-12.0), guard=guard)
    assert guard.fault is None and lines == []
    assert np.all(gains == 1.0), "a dead input never ducks the monitor"


def test_a_short_tempo_still_waits_eight_seconds():
    guard = InputGuard(SR)
    guard.bar_s = 1.0                       # 240 bpm: four bars is only 4 s
    run(strums(2.0, peak_db=-12.0), guard=guard)
    _, _, guard, lines = run(quiet(7.5), guard=guard)
    assert lines == []
    _, _, guard, lines = run(quiet(1.0), guard=guard)
    assert [line for _, line in lines] == [SILENT_LINE]


def test_nothing_is_said_of_silence_while_the_take_ends():
    guard = InputGuard(SR)
    run(strums(2.0, peak_db=-12.0), guard=guard)
    guard.ending = True
    _, _, guard, lines = run(quiet(20.0), guard=guard)
    assert lines == [] and guard.fault is None


def test_a_line_set_and_cleared_inside_one_feed_is_not_said():
    guard = InputGuard(SR)
    run(strums(2.0, peak_db=-12.0), guard=guard)
    x = np.concatenate([quiet(9.0), strums(1.0, peak_db=-12.0)])
    assert guard.feed(x) is None
    assert guard.fault is None


def test_a_midi_take_says_nothing_of_its_audio_input():
    guard = InputGuard(SR, notices=False)
    x = np.concatenate([strums(2.0, peak_db=-12.0), hum(2.0), quiet(12.0)])
    _, gains, guard, lines = run(x, guard=guard)
    assert lines == [] and guard.fault is None
    assert np.any(gains == 0.0), "Hear me is still kept off the speakers"


def test_silence_before_she_has_played_is_not_a_dead_input():
    _, _, guard, lines = run(quiet(10.0))
    assert guard.fault is None and lines == []


def test_a_buzz_then_silence_says_both():
    x = np.concatenate([strums(3.0, peak_db=-20.0), hum(2.6), quiet(10.0)])
    _, _, guard, lines = run(x)
    assert [line for _, line in lines] == [BUZZ_LINE, SILENT_LINE]
    assert guard.fault == "silent"


# ------------------------------------------------------------------ the page and the log

def test_the_live_page_says_the_fault_while_the_band_plays():
    def take(fault):
        return SimpleNamespace(phase="playing", row={"bar": 9, "input_fault": fault})
    assert guitake.TakeRunner._hint(take("buzz"), 0.0) == BUZZ_LINE
    assert guitake.TakeRunner._hint(take("silent"), 0.0) == SILENT_LINE
    assert guitake.TakeRunner._hint(take(None), 0.0) is None
    assert "input_fault" in guitake._ROW_FIELDS


def test_the_live_page_says_nothing_of_silence_while_the_take_stops_or_fades():
    stopping = SimpleNamespace(phase="playing", stop_asked=True,
                               row={"bar": 9, "input_fault": "silent"})
    assert guitake.TakeRunner._hint(stopping, 0.0) is None
    fading = SimpleNamespace(phase="playing", stop_asked=False,
                             row={"bar": 9, "input_fault": "silent", "fade_db": -12.0})
    assert guitake.TakeRunner._hint(fading, 0.0) is None
    fading_buzz = SimpleNamespace(phase="playing", stop_asked=False,
                                  row={"bar": 9, "input_fault": "buzz", "fade_db": -12.0})
    assert guitake.TakeRunner._hint(fading_buzz, 0.0) == BUZZ_LINE


RIG = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]


def _live_rows(tmp_path, monkeypatch, run, human, **kw):
    from conftest import FakeMido, fake_audio, patched_module, short_socket_path
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only
    sd = fake_audio(human, devices=RIG, rate_x=6.0)
    with patched_module("sounddevice", sd), patched_module("mido", FakeMido()):
        run(tmp_path, short_socket_path("guard"), **kw)
    return [json.loads(line) for line in
            (tmp_path / "rec" / realtime.REC_DECISIONS).read_text().splitlines() if line.strip()]


def _jam(tmp_path, sock, **kw):
    from plume import mind as mind_mod
    realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(), max_minutes=0.08,
                 socket_path=sock, record_dir=str(tmp_path / "rec"), monitor=True,
                 input_device="audio interface", input_channel=1, **kw)


def _lead(tmp_path, sock, **kw):
    from plume import roles as roles_mod
    realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"), status=False,
                  max_minutes=0.08, socket_path=sock, record_dir=str(tmp_path / "rec"),
                  monitor=True, input_device="audio interface", input_channel=1, **kw)


def _humming_take():
    return np.concatenate([strums(1.0, peak_db=-12.0), hum(9.0)])


@pytest.mark.parametrize("run", [_jam, _lead], ids=["jam", "lead"])
def test_a_buzzing_take_logs_input_fault(tmp_path, monkeypatch, plume_registry, run):
    rows = _live_rows(tmp_path, monkeypatch, run, _humming_take())
    assert rows
    assert all(r.get("input_fault", "missing") in (None, "buzz", "silent") for r in rows)
    assert any(r["input_fault"] == "buzz" for r in rows)


@pytest.mark.parametrize("run", [_jam, _lead], ids=["jam", "lead"])
def test_a_midi_take_never_logs_input_fault(tmp_path, monkeypatch, plume_registry, run):
    rows = _live_rows(tmp_path, monkeypatch, run, _humming_take(), midi="auto")
    assert rows
    assert all(r.get("input_fault", "missing") is None for r in rows)
