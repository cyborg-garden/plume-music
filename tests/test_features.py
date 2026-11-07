"""Tests for plume.listen.features (SPEC-V1 section 1)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from plume.listen import HOP, WIN, Frame, FeatureExtractor  # noqa: E402

SR = 48000


def run(x, block=4096, flush=True, **kwargs):
    fe = FeatureExtractor(SR, **kwargs)
    frames = []
    for i in range(0, len(x), block):
        frames.extend(fe.process(x[i:i + block]))
    if flush:
        frames.extend(fe.flush())
    return frames


def click_train(click_samples, length, width=4, amp=1.0):
    x = np.zeros(length, dtype=np.float32)
    for s in click_samples:
        x[s:s + width] = amp
    return x


def onset_frames(frames):
    return [i for i, f in enumerate(frames) if f.onset]


#: How far apart two blockings' spectral values (flux, centroid, attack) may be.
#: The FFT's SIMD path depends on how a buffer is aligned in memory, so on some
#: builds (Linux aarch64, measured) the same frame differs in the last bit or two.
#: Every decision (onset, chroma, attack breadth) stays exact.
#: macOS has always matched exactly, so there it still has to.
EXACT_FFT = sys.platform == "darwin"
FLUX_REL = 0.0 if EXACT_FFT else 1e-12


def fft_close(want, abs_=1e-9):
    """``want`` exactly on macOS, within the FFT's last-bit rounding elsewhere."""
    return want if EXACT_FFT else pytest.approx(want, rel=FLUX_REL, abs=abs_)


def assert_frames_equal(a, b):
    assert len(a) == len(b)
    for fa, fb in zip(a, b):
        assert fa.t_s == fb.t_s
        assert fa.rms_db == fb.rms_db
        assert fa.flux == fft_close(fb.flux)
        assert fa.onset == fb.onset
        assert fa.onset_strength == fft_close(fb.onset_strength)
        assert np.array_equal(fa.chroma, fb.chroma)
        assert fa.centroid_hz == fft_close(fb.centroid_hz, abs_=0.0)
        # v1.5: attack carries ATTACK_LAG frames of band history across
        # process() calls, so it is part of the blocking-invariance contract
        # and has to be compared here or the invariance tests would not
        # actually cover it.
        assert fa.attack == fft_close(fb.attack, abs_=1e-12)
        assert fa.attack_breadth == fb.attack_breadth


# ------------------------------------------------------------------ onsets

def test_onset_timing_on_click_train():
    """Each click detected exactly once, within +/-1 frame of its true frame."""
    clicks = [12000 + k * 19200 for k in range(10)]  # every 0.4s
    x = click_train(clicks, 12000 + 10 * 19200 + SR)
    frames = run(x)
    detected = onset_frames(frames)
    assert len(detected) == len(clicks)
    for got, s in zip(detected, clicks):
        expected = s // HOP  # first frame whose window contains the click
        assert abs(got - expected) <= 1
    # onset_strength carries the flux value on onsets, 0 elsewhere
    for i, f in enumerate(frames):
        if f.onset:
            assert f.onset_strength == f.flux > 0.0
        else:
            assert f.onset_strength == 0.0


def test_onset_min_separation_60ms():
    """Two clicks 20ms apart collapse to one onset; a distant third is kept."""
    clicks = [24000, 24000 + 960, 24000 + 24000]  # 20ms pair, then +0.5s
    x = click_train(clicks, 3 * SR)
    frames = run(x)
    detected = onset_frames(frames)
    assert len(detected) == 2
    min_sep_frames = int(np.ceil(0.060 * SR / HOP))
    diffs = np.diff(detected)
    assert np.all(diffs >= min_sep_frames)


# ---------------------------------------------------------- flux / silence

def test_flux_zero_on_silence():
    frames = run(np.zeros(SR, dtype=np.float32))
    assert len(frames) > 0
    for f in frames:
        assert f.flux == 0.0
        assert not f.onset
        assert f.onset_strength == 0.0
        assert f.rms_db == -90.0
        assert np.array_equal(f.chroma, np.zeros(12, dtype=np.float32))
        assert f.centroid_hz == 0.0
        assert f.attack == 0.0
        assert f.attack_breadth == 0.0
        assert f.self_masked is False


def test_flux_zero_on_dc():
    """After the window fills (first WIN/HOP frames), DC produces zero flux."""
    frames = run(0.5 * np.ones(SR, dtype=np.float32), flush=False)
    warmup = WIN // HOP
    assert len(frames) > warmup + 10
    for f in frames[warmup:]:
        assert f.flux == fft_close(0.0)    # FFT rounding, not a change
        assert not f.onset


# ------------------------------------------------------------------ chroma

def test_chroma_a4_sine_peaks_at_pitch_class_9():
    t = np.arange(SR, dtype=np.float64) / SR
    x = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    frames = run(x, flush=False)
    mid = frames[len(frames) // 2]
    assert mid.chroma.shape == (12,)
    assert mid.chroma.dtype == np.float32
    assert int(np.argmax(mid.chroma)) == 9  # A
    # Peak-based harmonic salience: the 440Hz peak (plus any subharmonic
    # ghosts it votes for) concentrates the mass on A — require clear
    # dominance.
    assert mid.chroma[9] > 0.5
    assert abs(float(mid.chroma.sum()) - 1.0) < 1e-4  # L1-normalized
    assert 380.0 < mid.centroid_hz < 520.0


# ----------------------------------------------------------- rms_db floor

def test_rms_db_floor_and_level():
    t = np.arange(SR, dtype=np.float64) / SR
    quiet = (1e-6 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    for f in run(quiet, flush=False)[WIN // HOP:]:
        assert f.rms_db == -90.0
    loud = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    mid = run(loud, flush=False)[len(run(loud, flush=False)) // 2]
    assert -10.0 < mid.rms_db <= 0.0


# ---------------------------------------------------- streaming invariants

def _test_signal(length=2 * SR + 100):
    rng = np.random.default_rng(0)
    t = np.arange(length, dtype=np.float64) / SR
    x = 0.2 * np.sin(2 * np.pi * (220.0 + 40.0 * t) * t)
    x += 0.02 * rng.standard_normal(length)
    for s in range(6000, length - 8, 17000):
        x[s:s + 8] += 0.8
    return x.astype(np.float32)


def test_block_size_invariance_4096_vs_256():
    x = _test_signal()
    assert_frames_equal(run(x, block=4096), run(x, block=256))


def test_block_size_invariance_ragged_vs_oneshot():
    x = _test_signal()
    assert_frames_equal(run(x, block=333), run(x, block=len(x)))


def test_determinism_two_runs_identical():
    x = _test_signal()
    assert_frames_equal(run(x), run(x))


# ------------------------------------------------------------- frame maths

def test_frame_timing_and_count():
    n = 5 * WIN
    frames = run(np.zeros(n, dtype=np.float32), flush=False)
    assert len(frames) == n // HOP
    for k, f in enumerate(frames):
        assert f.t_s == (k + 1) * HOP / SR


def test_flush_emits_partial_hop_frame():
    fe = FeatureExtractor(SR)
    frames = fe.process(np.zeros(HOP + 100, dtype=np.float32))
    assert len(frames) == 1
    tail = fe.flush()
    assert len(tail) == 1
    assert tail[0].t_s == 2 * HOP / SR
    assert fe.flush() == []  # nothing pending afterwards


def test_process_rejects_non_mono():
    fe = FeatureExtractor(SR)
    with pytest.raises(ValueError):
        fe.process(np.zeros((2, HOP), dtype=np.float32))


def test_frame_is_frozen():
    frames = run(np.zeros(WIN, dtype=np.float32), flush=False)
    with pytest.raises(AttributeError):
        frames[0].flux = 1.0


# ------------------------------------------------- bass chroma (v1.4, P1)
#
# ``Frame.chroma_low`` exists to give the KEY estimator the register
# information that the flat ``chroma`` throws away.  These tests pin the two
# properties harmony.py relies on: within the bass band the LOWEST fundamental
# dominates the vector, and the vector's MAGNITUDE is the low band's share of
# the frame's pitched evidence, so material with no bass cannot vote at all.

def _midi_hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def sine_chord(midis, secs=3.0, amps=None, sr=SR):
    t = np.arange(int(secs * sr)) / sr
    amps = amps if amps is not None else [1.0] * len(midis)
    x = sum(a * np.sin(2.0 * np.pi * _midi_hz(m) * t)
            for m, a in zip(midis, amps))
    return (0.25 * x / np.max(np.abs(x))).astype(np.float32)


def steady_mean(frames, attr="chroma"):
    """Mean of `attr` over the settled second half of the frame list."""
    tail = frames[len(frames) // 2:]
    return np.mean([getattr(f, attr) for f in tail], axis=0)


def test_bass_chroma_is_dominated_by_the_lowest_fundamental():
    """THE P1 MECHANISM, made permanent.

    C2 + E3 G3 B3 D4 is a Cmaj9 whose root is the only bass note.  The flat
    chroma spreads its mass over all five pitch classes and puts barely more
    on C than on the extensions, which is why the key estimator read the
    collection as G major.  chroma_low must say plainly that the ROOT is C.
    """
    frames = run(sine_chord([36, 52, 55, 59, 62]))
    flat, low = steady_mean(frames), steady_mean(frames, "chroma_low")
    assert flat[0] < 0.35, "flat chroma should NOT already single out the root"
    assert low[0] / low.sum() > 0.6, (
        f"bass chroma not root-dominated: {np.round(low / low.sum(), 3)}")
    assert low[2] == 0.0, "D4 (293.7Hz) is above the bass band entirely"


def test_bass_chroma_is_empty_when_nothing_is_low():
    """A4 (440Hz) is a whole octave above CHROMA_BASS_MAX_MIDI."""
    frames = run(sine_chord([69]))
    for f in frames[len(frames) // 2:]:
        assert f.chroma.sum() > 0.99          # the frame IS pitched
        assert f.chroma_low.sum() == 0.0      # ...but says nothing about a root


def test_bass_chroma_magnitude_is_a_share_not_a_unit_vector():
    """It must never hand a bass-less frame the same vote as a bass note.

    The sum is the low band's tilted share of the frame's pitched evidence, so
    it is bounded by 1 and collapses toward 0 as the content leaves the band.
    """
    low_notes = steady_mean(run(sine_chord([36])), "chroma_low").sum()
    high_notes = steady_mean(run(sine_chord([76, 79, 83])), "chroma_low").sum()
    assert high_notes < 0.05 < low_notes <= 1.0, (
        f"bass share: low={low_notes:.3f} high={high_notes:.3f}")


def test_bass_chroma_is_gain_invariant():
    """Recording level must not change what the bass says about the root."""
    x = sine_chord([36, 52, 55])
    loud = steady_mean(run(x), "chroma_low")
    quiet = steady_mean(run((x * 10 ** (-30 / 20.0)).astype(np.float32)),
                        "chroma_low")
    assert loud.sum() > 0.0
    np.testing.assert_allclose(quiet, loud, atol=0.02)


def test_bass_chroma_does_not_disturb_the_flat_chroma():
    """chroma_low is additive information; chord matching must see no change."""
    flat = steady_mean(run(sine_chord([48, 52, 55])))
    assert flat[0] + flat[4] + flat[7] > 0.98      # still exactly C E G


# ------------------------------- the onset flag is the DENSITY signal (v1.5)

# ``Frame.onset`` is not the beat tracker's private business: ``mind``,
# ``roles`` and ``sim`` read ONSET DENSITY off it, and ``MindParams``'
# thresholds are calibrated to THIS picker's numbers (SPEC-V1: ~1.2 onsets/beat
# calm vs ~3.1 busy).  The v1.5 ``attack`` work added a second, banded
# detection function and the beat tracker's acoustic mode picks its own events
# off that one — but the flag below had to stay exactly where it was, because a
# banded picker over the same corpus measures 2.36 onsets/beat calm and 2.76
# busy and the separation the inverse-density coupling depends on collapses.
#
# The last time that boundary was crossed it did not look like a features bug:
# it surfaced two files away as ``tests/test_behaviors.py``'s
# ``test_busy_middle_inverse_density_with_hysteresis`` reading a calm bar as
# busy.  These two tests make the boundary itself assertable here, where the
# defect actually lives.


def test_the_onset_flag_is_picked_from_flux_and_nothing_else():
    """Reconstruct the documented picker from the frames' own ``flux`` values
    and demand it reproduce ``Frame.onset`` EXACTLY.

    The rule (see ``ONSET_DELTA`` / ``ONSET_MEDIAN_S`` / ``ONSET_MIN_SEP_S``):
    flux strictly above the trailing median of the previous ``ONSET_MEDIAN_S``
    plus ``ONSET_DELTA``, and at least ``ONSET_MIN_SEP_S`` since the last
    accepted onset.  Nothing about ``attack`` may enter it.  Anything that
    swaps the source, the threshold shape or the median window fails here
    rather than silently re-calibrating ``mind``.
    """
    from collections import deque

    from plume.listen import features as feat

    rng = np.random.default_rng(4)
    x = np.zeros(8 * SR, dtype=np.float32)
    t = np.arange(int(0.25 * SR)) / SR
    for k in range(24):                       # sparse plucks, mixed strengths
        s = int((0.3 + k * 0.31) * SR)
        vel = 0.2 + 0.6 * float(rng.uniform())
        y = (vel * np.sin(2 * np.pi * 330.0 * t) * np.exp(-t / 0.09)
             ).astype(np.float32)
        x[s:s + y.size] += y
    frames = run(x)
    assert sum(f.onset for f in frames) > 10, "fixture produced no onsets"

    med_n = max(1, int(round(feat.ONSET_MEDIAN_S * SR / feat.HOP)))
    min_sep = max(1, int(np.ceil(feat.ONSET_MIN_SEP_S * SR / feat.HOP)))
    hist: deque[float] = deque(maxlen=med_n)
    last = -(10 ** 9)
    for i, f in enumerate(frames):
        med = float(np.median(hist)) if hist else 0.0
        want = f.flux > med + feat.ONSET_DELTA and (i - last) >= min_sep
        if want:
            last = i
        hist.append(float(f.flux))
        assert f.onset is want, (
            f"frame {i} (t={f.t_s:.3f}s): onset={f.onset} but flux "
            f"{f.flux:.3f} against median+delta {med + feat.ONSET_DELTA:.3f}")
        assert f.onset_strength == (f.flux if want else 0.0)


def test_the_attack_function_never_reaches_the_onset_flag():
    """A stream with plenty of ``attack`` and no flux excursions stays quiet.

    The direct statement of the split: a slow swell has band-level rises (so
    ``attack`` is non-zero) but no flux peak above median+delta, and the
    density signal must report nothing there.  If ``Frame.onset`` were ever
    re-sourced from ``attack``, this is the first thing that would change —
    and it is exactly the "a calm bar reads busy" failure.
    """
    n = 6 * SR
    t = np.arange(n) / SR
    swell = np.minimum(t / 3.0, 1.0)               # 3s linear fade-in
    x = (0.3 * swell * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    frames = run(x)
    attack = np.asarray([f.attack for f in frames])
    # NON-VACUITY FLOOR, and it has to be a floor rather than "> 0": the point
    # is that the two functions disagree LOUDLY here, so the swell must carry
    # real, sustained attack while producing essentially no flux peaks.
    # Measured: 74% of frames non-zero, p90 = 0.044, against 1 onset.
    assert (attack > 0.0).mean() > 0.5 and float(np.percentile(attack, 90)) > 0.01, (
        f"the swell carries almost no attack ({(attack > 0.0).mean():.2f} of "
        f"frames, p90 {np.percentile(attack, 90):.4f}); this test would be "
        f"vacuous")
    n_onsets = sum(f.onset for f in frames)
    assert n_onsets <= 1, (
        f"a 3-second swell produced {n_onsets} onsets — the density signal "
        f"is reading something other than flux peaks")
