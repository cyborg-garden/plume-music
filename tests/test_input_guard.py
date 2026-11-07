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


def hum(seconds: float, *, level: float = 1.0, hz: float = 50.0, clip: bool = True) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = np.sin(2 * np.pi * hz * t) + 0.3 * np.sin(2 * np.pi * 3 * hz * t)
    x = x / np.max(np.abs(x)) * (1.4 if clip else 1.0) * level
    return np.clip(x, -1.0, 1.0).astype(np.float32)


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

@pytest.mark.parametrize("clip", [True, False], ids=["clipping", "loud-unclipped"])
def test_a_hum_burst_ducks_the_monitor_fast_and_comes_back(clip):
    before = strums(2.0, peak_db=-20.0)
    burst = hum(2.5, level=1.0 if clip else 0.5, clip=clip)
    after = strums(3.0, peak_db=-20.0, seed=3)
    x = np.concatenate([before, burst, after])
    _, gains, guard, lines = run(x)
    onset = len(before) // BLOCK
    ducked = np.nonzero(gains[onset:] == 0.0)[0]
    assert ducked.size, "never ducked"
    # The analysis thread's low-pass needs a 20 ms window to tell a hum from a pick
    # attack (real takes clip on hard strums), so this is ~25-40 ms, not 10-20.  The
    # ceiling holds the speakers at -1 dBFS meanwhile.
    assert (ducked[0] + 1) * BLOCK / SR <= (0.030 if clip else 0.040)
    # Held for the whole burst, then back to full within ~0.5 s hold + 0.2 s ramp.
    end = (len(before) + len(burst)) // BLOCK
    assert np.all(gains[onset + ducked[0]:end - 1] == 0.0)
    back = np.nonzero(gains[end:] == 1.0)[0]
    assert back.size and back[0] * BLOCK / SR <= 0.9
    assert [line for _, line in lines] == [BUZZ_LINE]
    assert guard.fault is None


def test_buzz_output_stays_under_the_ceiling_with_hear_me_on():
    t = np.arange(4 * SR) / SR
    band = (0.6 * np.sin(2 * np.pi * 196 * t)).astype(np.float32)
    x = np.concatenate([strums(1.0, peak_db=-20.0), hum(3.0)])
    out, _, _, _ = run(x, band[:len(x)])
    assert np.max(np.abs(out)) <= realtime.MONITOR_CEILING + 1e-6


@pytest.mark.parametrize("low_heavy", [False, True], ids=["open-chords", "low-power-chords"])
@pytest.mark.parametrize("drive", [1.0, 2.5], ids=["clean", "clipping-input"])
def test_loud_strums_never_duck(low_heavy, drive):
    # Resting first: a hard strum out of silence is the biggest jump she makes.  And a
    # hot input gain clips on every strum: that is hers, not a fault.
    x = np.concatenate([quiet(2.0), strums(8.0, peak_db=-6.0, low_heavy=low_heavy),
                        quiet(1.0), strums(4.0, peak_db=-6.0, every_s=0.25,
                                           low_heavy=low_heavy, seed=5)])
    x = np.clip(x * drive, -1.0, 1.0)
    _, gains, guard, lines = run(x)
    assert np.all(gains == 1.0)
    assert lines == []
    assert guard.fault is None


# ------------------------------------------------------------------ a dead input

def test_a_dead_input_is_said_after_two_bars_and_clears_when_she_plays():
    guard = InputGuard(SR)
    guard.bar_s = 2.0                       # 120 bpm: two bars is 4 s
    _, _, guard, lines = run(strums(3.0, peak_db=-12.0), guard=guard)
    assert lines == [] and guard.fault is None
    _, _, guard, lines = run(quiet(3.5), guard=guard)
    assert lines == [] and guard.fault is None, "said before two bars"
    _, _, guard, lines = run(quiet(1.0), guard=guard)
    assert [line for _, line in lines] == [SILENT_LINE]
    assert guard.fault == "silent"
    _, _, guard, lines = run(quiet(3.0), guard=guard)
    assert lines == [], "said only once"
    _, gains, guard, lines = run(strums(1.0, peak_db=-12.0), guard=guard)
    assert guard.fault is None and lines == []
    assert np.all(gains == 1.0), "a dead input never ducks the monitor"


def test_silence_before_she_has_played_is_not_a_dead_input():
    _, _, guard, lines = run(quiet(10.0))
    assert guard.fault is None and lines == []


def test_a_buzz_then_silence_says_both():
    x = np.concatenate([strums(3.0, peak_db=-20.0), hum(2.6), quiet(6.0)])
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


def test_every_jam_decision_row_carries_input_fault(tmp_path, monkeypatch, plume_registry):
    from conftest import fake_audio, patched_module, short_socket_path

    from plume import mind as mind_mod
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only
    sd = fake_audio(None, devices=[{"name": "Audio Interface 4x4", "max_input_channels": 4,
                                    "max_output_channels": 4}], rate_x=6.0)
    with patched_module("sounddevice", sd):
        realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(), max_minutes=0.04,
                     socket_path=short_socket_path("guard"), record_dir=str(tmp_path / "rec"),
                     monitor=True, input_device="audio interface", input_channel=1)
    rows = [json.loads(line) for line in
            (tmp_path / "rec" / realtime.REC_DECISIONS).read_text().splitlines() if line.strip()]
    assert rows
    assert all(r.get("input_fault", "missing") in (None, "buzz", "silent") for r in rows)
