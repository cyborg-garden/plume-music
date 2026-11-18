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
from plume.inputguard import BUZZ_LINE, OVERLOAD_LINE, SILENT_LINE, InputGuard

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
    # hot input gain clips on every note: never a buzz, and never ducked, only said.
    x = np.concatenate([quiet(2.0), PLAYING[kind](), quiet(1.0)])
    x = np.clip(x * drive * 2.0, -1.0, 1.0)      # peaks at 0 dBFS and over
    _, gains, guard, lines = run(x)
    assert np.all(gains == 1.0)
    assert [line for _, line in lines] in ([], [OVERLOAD_LINE])
    assert guard.fault in (None, "overload")
    if drive >= 1.5:
        assert guard.fault == "overload", "notes flattened on the ceiling are said"
    if drive == 1.0:
        assert guard.fault is None, "peaks just touching full scale are not an overload"


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


# ------------------------------------------------------------------ an overloading input

def _hot(x: np.ndarray, drive: float = 3.0) -> np.ndarray:
    """The interface's gain set too hot: every note flattened on full scale."""
    return np.clip(x * drive, -1.0, 1.0).astype(np.float32)


def _bumpy(seconds: float, every_s: float = 4.0, *, level_db: float = -28.0,
           seed: int = 3) -> np.ndarray:
    """Quiet single notes with a burst now and then that swings rail to rail, mostly
    below 60 Hz: a knock or a crackling connection, not her playing."""
    rng = np.random.default_rng(seed)
    x = plucks(seconds, [196.0, 246.94, 293.66, 329.63], every_s=0.5).astype(np.float64)
    x *= _db_to_lin(level_db) / np.sqrt(np.mean(x ** 2))
    t = np.arange(int(0.06 * SR)) / SR
    at = 1.3
    while at + 0.1 < seconds:
        thump = 4.0 * np.sin(2 * np.pi * rng.uniform(25.0, 55.0) * t + rng.uniform(0, 6.3))
        thump = thump * np.exp(-t / 0.03) + 0.3 * rng.standard_normal(t.size) * np.exp(-t / 0.01)
        i = int(at * SR)
        x[i:i + t.size] += thump
        at += every_s * rng.uniform(0.8, 1.2)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _knocks(count: int = 8, every_s: float = 0.6) -> np.ndarray:
    """A loud count-in: knocks that hit full scale, as a knock on a pickup does."""
    rng = np.random.default_rng(9)
    x = np.zeros(int((count + 0.5) * every_s * SR))
    t = np.arange(int(0.08 * SR)) / SR
    for k in range(count):
        i = int((0.1 + k * every_s) * SR)
        x[i:i + t.size] += 6.0 * (np.sin(2 * np.pi * 110.0 * t) + rng.standard_normal(t.size)) \
            * np.exp(-t / 0.015)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


#: Every way the input can keep reaching full scale.  The player is told the same one
#: thing for all of them: what caused it is not hers to work out from a guess.
OVERLOADS = {
    "hot-strums": lambda: _hot(strums(6.0, peak_db=-3.0)),
    "20-db-over": lambda: _hot(strums(6.0, peak_db=0.0), drive=10.0),
    "hot-strums-far-apart": lambda: _hot(strums(12.0, peak_db=-3.0, every_s=2.5)),
    "hot-percussive-hits": lambda: _hot(plucks(6.0, [(82.41, 110.0, 146.83)], every_s=0.5,
                                               mute_s=0.06, decay=10.0), drive=4.0),
    "bass-20-db-hot": lambda: _hot(plucks(8.0, [41.2, 55.0, 73.42], every_s=0.6, decay=1.0),
                                   drive=10.0),
    "level-near-the-guess-line": lambda: _hot(strums(8.0, peak_db=-12.0, every_s=1.0), drive=4.2),
    "bursts-over-quiet-playing": lambda: _bumpy(16.0),
}


@pytest.mark.parametrize("kind", list(OVERLOADS))
def test_an_input_that_keeps_reaching_full_scale_gets_the_one_line(kind):
    _, gains, guard, lines = run(OVERLOADS[kind]())
    assert [line for _, line in lines] == [OVERLOAD_LINE], "said once, and never anything else"
    assert guard.fault == "overload"
    assert guard.full_hops > 0
    assert guard.full_hops >= guard.guess_clip_hops + guard.guess_bump_hops
    assert np.all(gains == 1.0), "only said: nothing is done to what she hears"


def test_the_one_line_is_true_of_a_hot_gain_and_of_bumps():
    words = OVERLOAD_LINE.lower()
    assert "gain down" in words and "cable" in words and "bumps" in words
    assert len(OVERLOAD_LINE) <= 110


def test_the_line_never_flaps_while_the_overload_goes_on():
    # Bumps ten seconds apart and a level at the guess's edge: said once, and standing
    # at every look for as long as they keep coming.
    guard = InputGuard(SR)
    x = np.concatenate([_bumpy(40.0, every_s=10.0),
                        _hot(strums(8.0, peak_db=-12.0, every_s=1.0), drive=4.2)])
    said, standing = [], []
    for i in range(0, len(x) - SR, SR):
        _, _, guard, lines = run(x[i:i + SR], guard=guard)
        said += [line for _, line in lines]
        standing.append(guard.fault)
    first = standing.index("overload")
    assert said == [OVERLOAD_LINE]
    assert set(standing[first:]) == {"overload"}


@pytest.mark.parametrize("kind", list(PLAYING))
def test_loud_playing_up_to_full_scale_unclipped_says_nothing(kind):
    x = PLAYING[kind]()
    x = (x / np.max(np.abs(x))).astype(np.float32)        # its one loudest peak at 0 dBFS
    _, _, guard, lines = run(np.concatenate([quiet(1.0), x]))
    assert lines == [] and guard.fault is None
    assert guard.full_hops <= 2, "the one peak may touch the top; nothing else does"


@pytest.mark.parametrize("stray", ["thump", "click", "hot-strum"])
def test_one_stray_thump_or_click_says_nothing(stray):
    x = strums(12.0, peak_db=-12.0, every_s=1.0).astype(np.float64)
    at = 4 * SR
    if stray == "thump":
        t = np.arange(int(0.06 * SR)) / SR
        x[at:at + t.size] += 4.0 * np.sin(2 * np.pi * 40.0 * t) * np.exp(-t / 0.03)
    elif stray == "click":
        x[at:at + 24] = 1.0
    else:
        x[at:at + SR // 2] = _hot(strums(0.5, peak_db=-3.0, every_s=1.0))
    _, _, guard, lines = run(np.clip(x, -1.0, 1.0).astype(np.float32))
    assert lines == [] and guard.fault is None
    assert guard.full_hops > 0, "it did reach full scale; it just is not said"


def test_two_bumps_are_not_said():
    x = np.concatenate([_bumpy(6.0, every_s=3.0), _bumpy(6.0, every_s=1000.0)[SR * 2:]])
    _, _, guard, lines = run(x)
    assert lines == [] and guard.fault is None
    assert guard.full_hops > 0


def test_a_loud_count_in_then_normal_playing_says_nothing():
    guard = InputGuard(SR)
    guard.counting = True
    _, _, guard, lines = run(_knocks(), guard=guard)
    assert lines == [] and guard.fault is None and guard.full_hops == 0
    guard.counting = False
    _, _, guard, lines = run(strums(8.0, peak_db=-6.0), guard=guard)
    assert lines == [] and guard.fault is None and guard.full_hops == 0


def test_the_same_knocks_outside_a_count_are_an_overload():
    _, _, guard, lines = run(_knocks())
    assert [line for _, line in lines] == [OVERLOAD_LINE]


def test_the_last_knock_still_ringing_when_the_count_lands_is_not_counted():
    guard = InputGuard(SR)
    guard.counting = True
    knocks = _knocks()
    cut = int((0.1 + 7 * 0.6) * SR) + 480          # 10 ms into the eighth knock
    run(knocks[:cut - cut % BLOCK], guard=guard)
    guard.counting = False
    _, _, guard, _ = run(knocks[cut - cut % BLOCK:], guard=guard)
    assert guard.full_hops == 0


def test_the_transport_tells_the_guard_when_it_is_counting():
    guard = InputGuard(SR)
    realtime.heed_input(guard, _knocks()[:BLOCK], counting=True)
    assert guard.counting is True
    realtime.heed_input(guard, _knocks()[:BLOCK])
    assert guard.counting is False


def test_the_overload_clears_once_it_stops_and_is_said_again_if_it_returns(monkeypatch):
    monkeypatch.setattr(inputguard, "OVERLOAD_CLEAR_S", 6.0)
    guard = InputGuard(SR)
    _, _, guard, _ = run(_hot(strums(5.0, peak_db=-3.0)), guard=guard)
    assert guard.fault == "overload"
    _, _, guard, lines = run(strums(4.0, peak_db=-12.0), guard=guard)
    assert guard.fault == "overload", "not cleared on the first clean bars"
    _, _, guard, lines = run(strums(4.0, peak_db=-12.0), guard=guard)
    assert guard.fault is None and lines == []
    _, _, guard, lines = run(_hot(strums(1.0, peak_db=-3.0, every_s=2.0)), guard=guard)
    assert guard.fault is None, "one more hot strum starts the count from nothing"
    _, _, guard, lines = run(_hot(strums(5.0, peak_db=-3.0)), guard=guard)
    assert [line for _, line in lines] == [OVERLOAD_LINE], "said again if it comes back"


_BED = None


def _clicks(seconds: float, at) -> np.ndarray:
    """Quiet strums with one flat-topped click at each of ``at`` seconds: one slot each."""
    global _BED
    if _BED is None:
        _BED = strums(4.0, peak_db=-20.0, every_s=1.0)
    x = np.tile(_BED, int(seconds // 4) + 1)[:int(seconds * SR)].copy()
    for t in at:
        i = int(t * SR)
        x[i:i + 48] = 1.0
    return x


def _heard(x: np.ndarray):
    """The guard alone, a second at a time: (guard, [(seconds, line said)])."""
    guard, lines = InputGuard(SR), []
    for i in range(0, len(x), SR):
        said = guard.feed(x[i:i + SR])
        if said:
            lines.append((i / SR, said))
    return guard, lines


def test_the_third_slot_at_full_scale_is_the_one_that_says_it():
    guard, lines = _heard(_clicks(8.0, [1.2, 3.2]))
    assert lines == [] and guard.fault is None and guard.full_hops == 2
    guard, lines = _heard(_clicks(8.0, [1.2, 3.2, 5.2]))
    assert lines == [(5.0, OVERLOAD_LINE)], "in the second the third one lands, not later"
    assert guard.full_hops == 3


def test_three_bumps_most_of_a_minute_apart_still_count_together():
    # 25 s apart: inside the minute's window, and each before the 30 s that would clear.
    guard, lines = _heard(_clicks(53.0, [1.2, 26.2, 51.2]))
    assert [line for _, line in lines] == [OVERLOAD_LINE]
    assert guard.fault == "overload"


def test_an_overload_is_said_over_a_dead_input():
    guard = InputGuard(SR)
    guard.bar_s = 2.0
    _, _, guard, _ = run(_hot(strums(5.0, peak_db=-3.0)), guard=guard)
    _, _, guard, lines = run(quiet(10.0), guard=guard)
    assert guard.silent and guard.overload, "both stand"
    assert guard.fault == "overload"


def test_a_midi_take_says_nothing_of_an_overloading_input():
    _, _, guard, lines = run(_hot(strums(6.0, peak_db=-3.0)), guard=InputGuard(SR, notices=False))
    assert lines == [] and guard.fault is None


def test_the_logs_guess_at_the_cause_is_only_a_guess_in_the_log():
    _, _, hot, _ = run(_hot(strums(8.0, peak_db=-3.0)))
    assert hot.guess_clip_hops > 0 and hot.guess_bump_hops == 0
    _, _, bumps, _ = run(_bumpy(16.0))
    assert bumps.guess_bump_hops > 0 and bumps.guess_clip_hops == 0
    _, _, heavy, _ = run(OVERLOADS["20-db-over"]())
    assert heavy.guess_clip_hops > 0 and heavy.guess_bump_hops == 0, \
        "heavy clipping leaves nothing unclipped to measure, and is still clipping"
    assert hot.fault == bumps.fault == heavy.fault == "overload", "the player is told one thing"


# ------------------------------------------------------------------ the page and the log

def test_the_live_page_says_the_fault_while_the_band_plays():
    def take(fault):
        return SimpleNamespace(phase="playing", row={"bar": 9, "input_fault": fault})
    assert guitake.TakeRunner._hint(take("buzz"), 0.0) == BUZZ_LINE
    assert guitake.TakeRunner._hint(take("silent"), 0.0) == SILENT_LINE
    assert guitake.TakeRunner._hint(take("overload"), 0.0) == OVERLOAD_LINE
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


def _live_rows(tmp_path, monkeypatch, run, human, status=None, loop_human=True, **kw):
    from conftest import (FakeMido, fake_audio, patched_module, playhead_rate,
                          short_socket_path)
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only
    sd = fake_audio(human, devices=RIG, rate_x=playhead_rate(6.0), status=status,
                    loop_human=loop_human)
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
    assert all(r.get("input_fault", "missing") in (None, "buzz", "silent", "overload")
               for r in rows)
    assert any(r["input_fault"] == "buzz" for r in rows)


@pytest.mark.parametrize("run", [_jam, _lead], ids=["jam", "lead"])
def test_a_midi_take_never_logs_input_fault(tmp_path, monkeypatch, plume_registry, run):
    rows = _live_rows(tmp_path, monkeypatch, run, _humming_take(), midi="auto")
    assert rows
    assert all(r.get("input_fault", "missing") is None for r in rows)


def test_an_overloading_take_logs_it_with_each_bars_stream_health(tmp_path, monkeypatch, plume_registry):
    from conftest import FakeCallbackFlags
    flagged = []

    def status(i):                  # an output underrun, then an input overflow, now and then
        if i % 50 == 0:
            flagged.append(i)
            return FakeCallbackFlags(4 if i % 100 == 0 else 2)
        return FakeCallbackFlags(0)
    rows = _live_rows(tmp_path, monkeypatch, _lead, _hot(strums(9.0, peak_db=-3.0)),
                      status=status)
    assert rows
    assert any(r["input_fault"] == "overload" for r in rows)
    assert all(isinstance(r.get("input_full_scale_hops"), int) for r in rows)
    full = sum(r["input_full_scale_hops"] for r in rows)
    guessed = [r["input_cause_guess"] for r in rows]
    assert all(set(g) == {"clipping_hops", "bump_hops"} for g in guessed)
    assert 0 < sum(g["clipping_hops"] + g["bump_hops"] for g in guessed) <= full
    for r in rows:
        timing = r["callback"]
        assert timing["period_ms"] == pytest.approx(1000.0 * BLOCK / SR, abs=1e-3)
        assert timing["worst_ms"] >= 0.0 and timing["worst_gap_ms"] >= 0.0
    assert any(r["callback"]["worst_ms"] > 0.0 for r in rows)
    assert all(set(r["xruns"]) == set(realtime.XRUN_NAMES) for r in rows)
    under = sum(r["xruns"]["output_underflow"] for r in rows)
    over = sum(r["xruns"]["input_overflow"] for r in rows)
    assert 0 < under <= len([i for i in flagged if i % 100 == 0])
    assert 0 < over <= len([i for i in flagged if i % 100])
    assert sum(r["xruns"]["input_underflow"] + r["xruns"]["output_overflow"]
               for r in rows) == 0
    # The take's totals, which also hold what came after the last logged bar.
    whole = json.loads((tmp_path / "rec" / realtime.REC_ENDING).read_text())["stream"]
    assert under <= whole["xruns"]["output_underflow"] <= len([i for i in flagged if i % 100 == 0])
    assert over <= whole["xruns"]["input_overflow"] <= len([i for i in flagged if i % 100])
    assert whole["xruns"]["output_underflow"] + whole["xruns"]["input_overflow"] >= len(flagged) - 1
    assert whole["input_full_scale_hops"] >= full
    assert whole["status_unreadable"] == 0
    assert whole["callback"]["worst_ms"] >= max(r["callback"]["worst_ms"] for r in rows)


def test_xruns_are_counted_from_the_flags_and_split_by_row():
    from conftest import FakeCallbackFlags
    tally = realtime.XrunTally()
    realtime.note_xruns(tally.counts, None)
    realtime.note_xruns(tally.counts, FakeCallbackFlags(0))
    realtime.note_xruns(tally.counts, FakeCallbackFlags(0x10))      # priming: not a run
    assert tally.row()["xruns"] == dict.fromkeys(realtime.XRUN_NAMES, 0)
    realtime.note_xruns(tally.counts, FakeCallbackFlags(1 | 8))
    realtime.note_xruns(tally.counts, FakeCallbackFlags(8))
    assert tally.row()["xruns"] == {"input_underflow": 1, "input_overflow": 0,
                                    "output_underflow": 0, "output_overflow": 2}
    assert tally.row()["xruns"]["output_overflow"] == 0, "each row only its own"


def test_the_takes_totals_hold_what_no_row_logged():
    from conftest import FakeCallbackFlags
    tally = realtime.XrunTally(period_s=0.004)
    realtime.note_xruns(tally.counts, FakeCallbackFlags(4))
    realtime.note_callback(tally.span, 1.000, 1.003)
    tally.row()
    realtime.note_xruns(tally.counts, FakeCallbackFlags(4 | 2))      # after the last row
    realtime.note_callback(tally.span, 1.010, 1.011)
    assert tally.totals() == {
        "xruns": {"input_underflow": 0, "input_overflow": 1, "output_underflow": 2,
                  "output_overflow": 0},
        "status_unreadable": 0,
        "callback": {"worst_ms": 3.0, "worst_gap_ms": 10.0, "period_ms": 4.0}}
    doc = realtime._ending_doc("stop", started_mono=0.0, max_minutes=1.0, bars=3,
                               stream=tally.totals())
    assert doc["stream"]["xruns"]["output_underflow"] == 2
    assert "stream" not in realtime._ending_doc("stop", started_mono=0.0, max_minutes=1.0, bars=3)


def test_each_row_holds_its_own_longest_callback_and_gap():
    tally = realtime.XrunTally(period_s=0.004)
    assert tally.row()["callback"] == {"worst_ms": 0.0, "worst_gap_ms": 0.0, "period_ms": 4.0}
    realtime.note_callback(tally.span, 10.000, 10.001)     # the first has no gap to measure
    realtime.note_callback(tally.span, 10.004, 10.0065)
    realtime.note_callback(tally.span, 10.013, 10.0135)    # late: 9 ms after the last began
    assert tally.row()["callback"] == {"worst_ms": 2.5, "worst_gap_ms": 9.0, "period_ms": 4.0}
    realtime.note_callback(tally.span, 10.017, 10.0175)
    assert tally.row()["callback"] == {"worst_ms": 0.5, "worst_gap_ms": 4.0, "period_ms": 4.0}


def test_sounddevices_own_flags_read_the_same():
    # In a process of its own: importing sounddevice starts PortAudio, which this one
    # never does, and a machine with no sound server refuses.
    import subprocess
    import sys
    code = """
import sys
try:
    import sounddevice as sd
except Exception:
    sys.exit(77)
from plume import realtime
counts = [0, 0, 0, 0, 0]
realtime.note_xruns(counts, sd.CallbackFlags())
flags = sd.CallbackFlags()
flags.input_overflow = True
flags.output_underflow = True
realtime.note_xruns(counts, flags)
print(counts)
"""
    src = str(__import__("pathlib").Path(realtime.__file__).resolve().parents[1])
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=60, env={**__import__("os").environ, "PYTHONPATH": src})
    if done.returncode == 77:
        pytest.skip("sounddevice does not load here")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[0, 1, 1, 0, 0]"


class _Surprise:
    """A status object that is nothing like the one plume expects."""

    def __init__(self, how):
        self.how = how

    def __bool__(self):
        if self.how == "bool":
            raise RuntimeError("no truth here")
        return True

    def __getattr__(self, name):
        if self.how == "missing":
            raise AttributeError(name)
        raise ValueError("not a flag")


@pytest.mark.parametrize("how", ["bool", "missing", "value", "int", "text", "object"])
def test_a_surprising_status_never_raises_in_the_callback(how):
    status = {"int": 5, "text": "flags", "object": object()}.get(how) or _Surprise(how)
    tally = realtime.XrunTally()
    realtime.note_xruns(tally.counts, status)          # must not raise
    assert tally.counts[:4] == [0, 0, 0, 0]
    # ... and the take says so at its end, so it does not read as no xruns at all.
    assert tally.totals()["status_unreadable"] == 1
    realtime.note_xruns(tally.counts, status)
    assert tally.totals()["status_unreadable"] == 2
    assert tally.row()["xruns"] == dict.fromkeys(realtime.XRUN_NAMES, 0)


def test_a_readable_status_is_never_counted_as_unreadable():
    from conftest import FakeCallbackFlags
    tally = realtime.XrunTally()
    for bits in (0, 1, 0xF):
        realtime.note_xruns(tally.counts, FakeCallbackFlags(bits))
    realtime.note_xruns(tally.counts, None)
    assert tally.totals()["status_unreadable"] == 0


def test_the_flags_are_read_by_their_public_names_only():
    import inspect
    assert "._flags" not in inspect.getsource(realtime.note_xruns)


def test_a_loud_count_in_on_a_live_take_is_never_an_overload(tmp_path, monkeypatch, plume_registry):
    # Eight knocks at full scale, then playing well under it: the take's own log.
    # Played once, not looped: a second round of knocks mid-take would be an overload.
    x = np.concatenate([quiet(0.5), _knocks(), np.tile(strums(8.0, peak_db=-9.0), 6)])

    def lead(tmp, sock, **kw):
        from plume import roles as roles_mod
        realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"),
                      status=False, max_minutes=0.1, socket_path=sock, tap_start=True,
                      record_dir=str(tmp / "rec"), input_device="audio interface",
                      input_channel=1, direct=True, **kw)
    rows = _live_rows(tmp_path, monkeypatch, lead, x, loop_human=False)
    assert len(rows) >= 2, "the count never landed, so this proves nothing"
    assert all(r["input_fault"] is None for r in rows)
    assert sum(r["input_full_scale_hops"] for r in rows) == 0
