"""Tests for plume.fx (SPEC sections 8 and 12)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from plume.fx import (  # noqa: E402
    chorus,
    correlation,
    crossfade,
    db_glide,
    delay,
    drive,
    ducker,
    equal_power_gain,
    fade_curve,
    freeverb,
    limiter,
    lofi,
    phase_for_gain,
    svf_highpass,
    svf_lowpass,
)

SR = 48000


def _noise(n=SR // 2, seed=7):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.5


def _sine(freq, n=SR // 2, amp=0.5):
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _band_energy(x, lo, hi):
    spec = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / SR)
    return spec[(freqs >= lo) & (freqs < hi)].sum()


# ---------------------------------------------------------------- filters

def test_svf_lowpass_attenuates_highs():
    x = _noise()
    y = svf_lowpass(x, SR, 500.0)
    assert y.dtype == np.float32 and y.shape == x.shape
    assert _band_energy(y, 5000, 20000) < 0.01 * _band_energy(x, 5000, 20000)
    # passband roughly intact
    assert _band_energy(y, 20, 200) > 0.5 * _band_energy(x, 20, 200)


def test_svf_highpass_attenuates_lows():
    lo = _sine(100.0)
    hi = _sine(8000.0)
    y_lo = svf_highpass(lo, SR, 2000.0)
    y_hi = svf_highpass(hi, SR, 2000.0)
    assert y_lo.dtype == np.float32
    assert np.sqrt(np.mean(y_lo.astype(np.float64) ** 2)) < 0.05 * np.sqrt(np.mean(lo.astype(np.float64) ** 2))
    assert np.sqrt(np.mean(y_hi.astype(np.float64) ** 2)) > 0.7 * np.sqrt(np.mean(hi.astype(np.float64) ** 2))


def test_svf_stereo_shape_preserved():
    x = np.stack([_sine(220.0), _sine(440.0)], axis=1)
    y = svf_lowpass(x, SR, 1000.0)
    assert y.shape == x.shape and y.dtype == np.float32


# ---------------------------------------------------------------- drive

def test_drive_bounded_and_shaped():
    x = _sine(220.0, amp=1.0)
    y = drive(x, 0.8)
    assert y.dtype == np.float32 and y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert float(np.max(np.abs(y))) <= 1.0 + 1e-6
    # saturation raises RMS relative to peak
    assert np.mean(np.abs(y.astype(np.float64))) > np.mean(np.abs(x.astype(np.float64)))


# ---------------------------------------------------------------- chorus

def test_chorus_returns_stereo():
    x = _sine(440.0)
    y = chorus(x, SR, depth=0.7)
    assert y.shape == (x.size, 2)
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))
    assert float(np.max(np.abs(y))) > 1e-3
    assert not np.array_equal(y[:, 0], y[:, 1]), "channels should decorrelate"


# ---------------------------------------------------------------- delay

def test_delay_echo_at_dotted_eighth():
    n = SR  # 1 s
    x = np.zeros(n, dtype=np.float32)
    x[0] = 1.0
    y = delay(x, SR, bpm=120.0, wet=1.0, feedback=0.0)
    assert y.dtype == np.float32 and y.shape == x.shape
    d = int(round((3 / 16) * 4 * (60.0 / 120.0) * SR))  # 18000
    tail = np.abs(y[1000:])
    assert int(np.argmax(tail)) + 1000 == d
    assert y[0] == pytest.approx(1.0, abs=1e-6)  # dry passes


def test_delay_wet_zero_is_dry():
    x = _sine(330.0)
    y = delay(x, SR, bpm=100.0, wet=0.0)
    assert np.allclose(y, x, atol=1e-6)


def test_delay_feedback_produces_multiple_echoes():
    n = 2 * SR
    x = np.zeros(n, dtype=np.float32)
    x[0] = 1.0
    y = delay(x, SR, bpm=170.0, wet=1.0, feedback=0.5)
    d = int(round((3 / 16) * 4 * (60.0 / 170.0) * SR))
    e1 = np.abs(y[d - 50:d + 50]).max()
    e2 = np.abs(y[2 * d - 50:2 * d + 50]).max()
    assert e1 > 0 and e2 > 0
    assert e2 < e1  # decaying echoes


# ---------------------------------------------------------------- freeverb

def test_freeverb_stereo_and_tail():
    n = SR  # 1 s
    x = np.zeros(n, dtype=np.float32)
    x[0] = 1.0
    y = freeverb(x, SR, wet=0.6)
    assert y.shape == (n, 2) and y.dtype == np.float32
    assert np.all(np.isfinite(y))
    # reverb tail well after the impulse
    assert np.abs(y[int(0.3 * SR):]).sum() > 1e-3
    # stereo spread: channels differ
    assert not np.array_equal(y[:, 0], y[:, 1])


def test_freeverb_wet_zero_is_dry_dual_mono():
    x = _sine(440.0)
    y = freeverb(x, SR, wet=0.0)
    assert y.shape == (x.size, 2)
    assert np.allclose(y[:, 0], x, atol=1e-6)
    assert np.allclose(y[:, 1], x, atol=1e-6)


# ---------------------------------------------------------------- ducker

def test_ducker_dips_by_expected_db():
    n = SR  # 1 s of ones
    x = np.ones(n, dtype=np.float32)
    y = ducker(x, SR, [0.5], depth=1.0, dip_db=-9.0)
    assert y.dtype == np.float32 and y.shape == x.shape
    min_db = 20 * np.log10(float(np.min(y)))
    assert min_db == pytest.approx(-9.0, abs=0.5)
    # untouched before the trigger
    assert float(np.min(y[: int(0.49 * SR)])) > 0.999
    # mostly recovered by the end (release 220 ms)
    assert float(y[-1]) > 0.9


def test_ducker_depth_scales_dip():
    x = np.ones(SR, dtype=np.float32)
    half = ducker(x, SR, [0.2], depth=0.5, dip_db=-9.0)
    full = ducker(x, SR, [0.2], depth=1.0, dip_db=-9.0)
    assert float(np.min(half)) > float(np.min(full))


def test_ducker_stereo_shape():
    x = np.ones((SR // 2, 2), dtype=np.float32)
    y = ducker(x, SR, [0.1], depth=1.0)
    assert y.shape == x.shape and y.dtype == np.float32


# ---------------------------------------------------------------- lofi

def test_lofi_amount_zero_is_identity():
    x = _sine(440.0)
    assert np.array_equal(lofi(x, SR, 0.0), x)


def test_lofi_processes_and_preserves_shape():
    x = _sine(440.0)
    y = lofi(x, SR, 0.8)
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.all(np.isfinite(y))
    assert not np.array_equal(y, x)
    xs = np.stack([x, x], axis=1)
    ys = lofi(xs, SR, 0.8)
    assert ys.shape == xs.shape and ys.dtype == np.float32


def test_lofi_darkens():
    x = _noise()
    y = lofi(x, SR, 1.0)
    assert _band_energy(y, 10000, 20000) < 0.2 * _band_energy(x, 10000, 20000)


# ---------------------------------------------------------------- limiter

def test_limiter_never_exceeds_ceiling():
    x = 4.0 * _sine(220.0, amp=1.0)
    y = limiter(x, ceiling_db=-1.0)
    ceiling = 10 ** (-1.0 / 20.0)
    assert y.dtype == np.float32
    assert float(np.max(np.abs(y))) <= ceiling + 1e-6
    hot = np.full(1000, 100.0, dtype=np.float32)
    assert float(np.max(np.abs(limiter(hot)))) <= ceiling + 1e-6


def test_limiter_transparent_for_small_signals():
    x = _sine(440.0, amp=0.05)
    y = limiter(x, ceiling_db=-1.0)
    assert np.allclose(y, x, atol=5e-4)


def test_limiter_bit_transparent_below_knee():
    """v0.1 (#4): signals below the knee (ceiling - 3 dB) pass untouched,
    so gain changes stay visible in the render instead of being squashed."""
    x = _sine(220.0, amp=0.5)                    # -6 dBFS < knee (-4 dBFS)
    y = limiter(x, ceiling_db=-1.0)
    assert np.array_equal(y, x.astype(np.float32))


def test_limiter_monotone_through_knee():
    x = np.linspace(0.0, 4.0, 4000, dtype=np.float32)
    y = limiter(x, ceiling_db=-1.0)
    assert np.all(np.diff(y.astype(np.float64)) >= -1e-9)   # monotone
    assert float(y.max()) < 10 ** (-1.0 / 20.0)             # under ceiling


# ---------------------------------------------------------------- fades (v1.4)
#
# The primitives the duet engine's gentle entrances are built from.  What
# matters is not the shape for its own sake but three properties the renderer
# leans on: constant power, exact endpoints, and segmentability — a fade is
# generated one bar at a time and must still be one smooth curve.

def test_equal_power_endpoints_and_direction():
    assert equal_power_gain(0.0, direction="in") == pytest.approx(0.0)
    assert equal_power_gain(1.0, direction="in") == pytest.approx(1.0)
    assert equal_power_gain(0.0, direction="out") == pytest.approx(1.0)
    assert equal_power_gain(1.0, direction="out") == pytest.approx(0.0)


def test_equal_power_pair_holds_power_constant():
    p = np.linspace(0.0, 1.0, 501)
    gi = equal_power_gain(p, direction="in")
    go = equal_power_gain(p, direction="out")
    assert np.allclose(gi ** 2 + go ** 2, 1.0, atol=1e-12)


def test_equal_power_is_monotone_and_clipped_outside_the_unit_interval():
    p = np.linspace(-0.5, 1.5, 401)
    gi = equal_power_gain(p, direction="in")
    assert np.all(np.diff(gi) >= -1e-12)
    assert gi[0] == pytest.approx(0.0) and gi[-1] == pytest.approx(1.0)


def test_phase_for_gain_inverts_equal_power():
    for d in ("in", "out"):
        for p in (0.0, 0.13, 0.5, 0.87, 1.0):
            g = float(equal_power_gain(p, direction=d))
            assert phase_for_gain(g, direction=d) == pytest.approx(p, abs=1e-9)


def test_fade_curve_segments_concatenate_into_one_smooth_curve():
    """Three chunks of a fade == the fade generated in one go."""
    whole = fade_curve(900, direction="in")
    parts = np.concatenate([
        fade_curve(300, direction="in", start=0.0, end=1 / 3),
        fade_curve(300, direction="in", start=1 / 3, end=2 / 3),
        fade_curve(300, direction="in", start=2 / 3, end=1.0)])
    assert np.allclose(whole, parts, atol=1e-12)
    assert np.max(np.abs(np.diff(parts))) < 3e-3      # no step at either join


def test_fade_curve_bad_direction_raises():
    with pytest.raises(ValueError):
        fade_curve(10, direction="sideways")


def test_crossfade_endpoints_are_pure():
    a = np.ones(1000)
    b = np.full(1000, 2.0)
    y = crossfade(a, b, 500)
    assert y[0] == pytest.approx(1.0)                 # all outgoing
    assert y[-1] == pytest.approx(2.0)                # all incoming
    assert np.allclose(y[500:], 2.0)                  # past the window


def test_crossfade_of_independent_noise_holds_power_flat():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(48000)
    b = rng.standard_normal(48000)
    y = crossfade(a, b, 48000)
    w = 2000
    rms = np.array([np.sqrt(np.mean(y[i:i + w] ** 2))
                    for i in range(0, 46000, w)])
    assert rms.max() / rms.min() < 1.25               # no notch, no bulge


def test_correlated_crossfade_would_bulge_without_compensation():
    """The reason ``correlation`` exists: identical material sums, not powers."""
    x = _sine(220.0, n=48000).astype(np.float64)
    naive = crossfade(x, x, 48000)
    compensated = crossfade(x, x, 48000, correlation=1.0)
    mid = slice(23000, 25000)
    assert np.max(np.abs(naive[mid])) > np.max(np.abs(x[mid])) * 1.3
    assert np.max(np.abs(compensated[mid])) == pytest.approx(
        np.max(np.abs(x[mid])), rel=1e-6)


def test_crossfade_compensation_leaves_the_window_edges_alone():
    a = np.ones(400)
    b = np.full(400, 3.0)
    for r in (0.0, 0.4, 1.0):
        y = crossfade(a, b, 200, correlation=r)
        assert y[0] == pytest.approx(1.0)
        assert np.allclose(y[200:], 3.0)


def test_crossfade_handles_stereo_and_ragged_lengths():
    a = np.ones((300, 2))
    b = np.full((500, 2), 2.0)
    y = crossfade(a, b, 200)
    assert y.shape == (500, 2)
    assert np.allclose(y[0], 1.0) and np.allclose(y[-1], 2.0)


def test_crossfade_zero_window_is_a_hard_splice():
    a = np.ones(100)
    b = np.zeros(100)
    assert np.array_equal(crossfade(a, b, 0), b)


def test_correlation_is_one_for_identical_and_zero_for_silence():
    x = _noise(4096)
    assert correlation(x, x, 4096) == pytest.approx(1.0)
    assert correlation(x, np.zeros(4096), 4096) == 0.0
    rng = np.random.default_rng(9)
    a, b = rng.standard_normal(48000), rng.standard_normal(48000)
    assert correlation(a, b, 48000) < 0.05


def test_db_glide_starts_offset_and_lands_on_unity():
    g = db_glide(1000, -6.0, 400)
    assert g[0] == pytest.approx(10 ** (-6.0 / 20.0))
    assert np.allclose(g[400:], 1.0)
    assert np.all(np.diff(g[:400]) > 0.0)             # monotone, no step


def test_db_glide_is_the_identity_without_a_move_or_a_window():
    assert np.allclose(db_glide(500, 0.0, 200), 1.0)
    assert np.allclose(db_glide(500, -9.0, 0), 1.0)
