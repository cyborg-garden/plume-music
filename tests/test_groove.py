"""Tests for plume.groove (SPEC section 4)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from plume.groove import GrooveParams, apply_groove  # noqa: E402


def _g(swing=0.5, bias=None, hms=0.0, hvel=0.0):
    return GrooveParams(swing=swing, timing_bias_ms=bias or {},
                        humanize_ms=hms, humanize_vel=hvel)


def _rng():
    return np.random.default_rng(7)


BPM = 120.0          # beat = 0.5 s, sixteenth = 0.125 s
SIXTEENTH = 0.125


def test_straight_50_no_shift():
    t, v = apply_groove(1.25, role="hat", bpm=BPM, g=_g(0.5), rng=_rng(), velocity=0.8)
    assert t == pytest.approx(1.25 * 0.5, abs=1e-9)
    assert v == pytest.approx(0.8)


def test_swing_58_even_sixteenth_delayed():
    # delay = (0.58-0.5)*2 * sixteenth = 0.16 * 0.125 = 0.02 s
    t, _ = apply_groove(0.25, role="hat", bpm=BPM, g=_g(0.58), rng=_rng(), velocity=0.8)
    assert t == pytest.approx(0.25 * 0.5 + 0.16 * SIXTEENTH, abs=1e-9)


def test_swing_66_is_near_triplet():
    # exact triplet: even 16th lands at 1/3 of the beat
    t, _ = apply_groove(0.25, role="hat", bpm=BPM, g=_g(2.0 / 3.0), rng=_rng(),
                        velocity=0.8)
    assert t == pytest.approx((1.0 / 3.0) * 0.5, abs=1e-9)
    t66, _ = apply_groove(0.25, role="hat", bpm=BPM, g=_g(0.66), rng=_rng(),
                          velocity=0.8)
    assert t66 == pytest.approx(0.125 + 0.32 * SIXTEENTH, abs=1e-9)


def test_swing_applies_to_both_even_sixteenths():
    g = _g(0.6)
    for pos in (0.25, 0.75, 1.25, 3.75):
        t, _ = apply_groove(pos, role="hat", bpm=BPM, g=g, rng=_rng(), velocity=0.5)
        assert t > pos * 0.5


def test_swing_leaves_downbeats_and_eighths_alone():
    g = _g(0.66)
    for pos in (0.0, 0.5, 1.0, 2.5, 3.0):
        t, _ = apply_groove(pos, role="hat", bpm=BPM, g=g, rng=_rng(), velocity=0.5)
        assert t == pytest.approx(pos * 0.5, abs=1e-9)


def test_timing_bias_per_role():
    g = _g(0.5, bias={"snare": 20.0, "kick": -15.0})
    t, _ = apply_groove(1.0, role="snare", bpm=BPM, g=g, rng=_rng(), velocity=0.8)
    assert t == pytest.approx(0.5 + 0.020, abs=1e-9)
    t, _ = apply_groove(1.0, role="kick", bpm=BPM, g=g, rng=_rng(), velocity=0.8)
    assert t == pytest.approx(0.5 - 0.015, abs=1e-9)
    t, _ = apply_groove(1.0, role="hat", bpm=BPM, g=g, rng=_rng(), velocity=0.8)
    assert t == pytest.approx(0.5, abs=1e-9)  # unlisted role -> no bias


def test_downbeat_accent():
    _, v = apply_groove(0.0, role="kick", bpm=BPM, g=_g(), rng=_rng(), velocity=0.8)
    assert v == pytest.approx(0.88)
    _, v = apply_groove(4.0, role="kick", bpm=BPM, g=_g(), rng=_rng(), velocity=0.8)
    assert v == pytest.approx(0.88)  # bar 2 downbeat (global beats)
    _, v = apply_groove(1.0, role="kick", bpm=BPM, g=_g(), rng=_rng(), velocity=0.8)
    assert v == pytest.approx(0.8)   # beat 2 is not a downbeat


def test_velocity_clip_range():
    _, v = apply_groove(0.0, role="kick", bpm=BPM, g=_g(), rng=_rng(), velocity=0.99)
    assert v == 1.0  # accent clipped at 1
    _, v = apply_groove(0.5, role="hat", bpm=BPM, g=_g(), rng=_rng(), velocity=0.01)
    assert v == pytest.approx(0.05)  # floor


def test_humanize_determinism_with_seed():
    g = _g(0.57, bias={"snare": 20.0}, hms=4.0, hvel=0.08)
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    outs_a = [apply_groove(i * 0.25, role="snare", bpm=90, g=g, rng=rng_a,
                           velocity=0.7) for i in range(16)]
    outs_b = [apply_groove(i * 0.25, role="snare", bpm=90, g=g, rng=rng_b,
                           velocity=0.7) for i in range(16)]
    assert outs_a == outs_b


def test_humanize_different_seeds_differ():
    g = _g(hms=4.0, hvel=0.08)
    a = apply_groove(1.0, role="hat", bpm=BPM, g=g,
                     rng=np.random.default_rng(1), velocity=0.7)
    b = apply_groove(1.0, role="hat", bpm=BPM, g=g,
                     rng=np.random.default_rng(2), velocity=0.7)
    assert a != b


def test_humanize_jitter_is_small():
    g = _g(hms=4.0)
    rng = np.random.default_rng(0)
    ts = [apply_groove(1.0, role="hat", bpm=BPM, g=g, rng=rng, velocity=0.7)[0]
          for _ in range(200)]
    devs = np.array(ts) - 0.5
    assert abs(float(np.mean(devs))) < 0.002
    assert float(np.std(devs)) < 0.010  # SD ~4 ms
