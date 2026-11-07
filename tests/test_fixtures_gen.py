"""Tests for tests/fixtures_gen.py — the v1 fixture humans (SPEC-V1 section 9).

Verifies, per kind: render is real audio (mono float32 @48k, finite, sane
peak), duration matches the truth grid, silent bars are exactly silent and
everything else is not, the advertised behaviors are measurable in the audio
itself (onset density, IOI irregularity, register, tempo drift — via scipy
spectral-flux onset detection and Welch spectra), and everything is
deterministic per seed.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from fixtures_gen import KINDS, SR, make_human, make_human_with_truth
from plume.theory import parse_key

BPM = 96.0
BARS = 24


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def detect_onsets(x: np.ndarray, *, min_sep_s: float = 0.1,
                  k: float = 1.5) -> np.ndarray:
    """Spectral-flux onset times (s). Threshold = median + k*(p90 - median)."""
    _, t, Z = signal.stft(x, fs=SR, nperseg=2048, noverlap=2048 - 512,
                          boundary=None)
    mag = np.log1p(100.0 * np.abs(Z))
    flux = np.sum(np.maximum(np.diff(mag, axis=1), 0.0), axis=0)
    thr = np.median(flux) + k * (np.percentile(flux, 90) - np.median(flux))
    dist = max(1, int(min_sep_s * SR / 512))
    peaks, _ = signal.find_peaks(flux, height=thr, distance=dist)
    return t[1:][peaks]


def bar_rms_db(audio: np.ndarray, truth: dict) -> list[float]:
    edges = list(truth["bar_times_s"]) + [audio.shape[0] / SR]
    out = []
    for b in range(truth["bars"]):
        seg = audio[int(edges[b] * SR): int(edges[b + 1] * SR)].astype(np.float64)
        out.append(20.0 * np.log10(float(np.sqrt(np.mean(seg ** 2))) + 1e-12))
    return out


def low_power_fraction(x: np.ndarray, split_hz: float = 500.0) -> float:
    f, p = signal.welch(x.astype(np.float64), fs=SR, nperseg=8192)
    return float(p[f < split_hz].sum() / p.sum())


@pytest.fixture(scope="module")
def rendered() -> dict[str, tuple[np.ndarray, dict]]:
    """Every kind rendered once with the SPEC defaults."""
    return {kind: make_human_with_truth(kind) for kind in KINDS}


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown fixture kind"):
        make_human("swing_feel")


@pytest.mark.parametrize("kind,bars", [("long_stop", 8), ("drift", 4),
                                       ("stops", 4), ("busy_middle", 4)])
def test_too_few_bars_raises(kind, bars):
    with pytest.raises(ValueError, match="needs bars >="):
        make_human(kind, bars=bars)


# ---------------------------------------------------------------------------
# every kind: format, duration, silence map, truth structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_format_and_duration(kind, rendered):
    audio, truth = rendered[kind]
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert np.all(np.isfinite(audio))
    assert 0.1 < float(np.abs(audio).max()) <= 1.0

    # duration = intro offset + the truth beat grid span, sample-exact
    end_t, _ = truth["bpm_curve"][-1]
    assert audio.shape[0] == int(round(end_t * SR))
    if kind == "rubato_intro":
        assert truth["intro_end_s"] == pytest.approx(8.0)
        assert truth["bar_times_s"][0] == pytest.approx(8.0)
    elif kind != "drift":  # constant tempo: closed-form duration
        assert audio.shape[0] / SR == pytest.approx(BARS * 4 * 60.0 / BPM,
                                                    abs=1.0 / SR * 2)


@pytest.mark.parametrize("kind", KINDS)
def test_silent_bars_exact_rest_loud(kind, rendered):
    audio, truth = rendered[kind]
    expected_silent = {"stops": 2, "long_stop": 8}.get(kind, 0)
    assert len(truth["silent_bars"]) == expected_silent
    if expected_silent:
        run = truth["silent_bars"]
        assert run == list(range(run[0], run[0] + expected_silent))  # contiguous
        assert 2 <= run[0] and run[-1] <= truth["bars"] - 3          # mid-take
    rms = bar_rms_db(audio, truth)
    silent = set(truth["silent_bars"])
    edges = list(truth["bar_times_s"]) + [audio.shape[0] / SR]
    for b in range(truth["bars"]):
        if b in silent:  # exact digital silence
            seg = audio[int(edges[b] * SR): int(edges[b + 1] * SR)]
            assert not np.any(seg), f"silent bar {b} has nonzero samples"
        else:
            assert rms[b] > -35.0, f"bar {b} unexpectedly quiet: {rms[b]:.1f} dB"


@pytest.mark.parametrize("kind", KINDS)
def test_truth_structure(kind, rendered):
    _, truth = rendered[kind]
    assert truth["bars"] == BARS
    assert truth["sr"] == SR
    assert len(truth["bar_times_s"]) == BARS
    assert len(truth["bpm_curve"]) == BARS * 4 + 1
    ts = [t for t, _ in truth["bpm_curve"]]
    assert all(b - a > 0 for a, b in zip(ts, ts[1:]))          # strictly increasing
    assert truth["bar_times_s"] == pytest.approx(ts[::4][:BARS])
    parse_key(truth["key"])                                     # parseable
    # beat times consistent with the stated bpm: dt == 60/bpm at each beat
    for (t0, bpm0), (t1, _) in zip(truth["bpm_curve"], truth["bpm_curve"][1:]):
        assert (t1 - t0) == pytest.approx(60.0 / bpm0, rel=1e-9)


# ---------------------------------------------------------------------------
# per-kind behaviors, measured in the audio
# ---------------------------------------------------------------------------

def test_steady_constant_bpm_and_grid(rendered):
    audio, truth = rendered["steady"]
    assert all(b == pytest.approx(BPM) for _, b in truth["bpm_curve"])
    # detected onsets sit on a steady pulse: IOI cv tiny, mean IOI ~ one beat
    ons = detect_onsets(audio, min_sep_s=0.15)
    ioi = np.diff(ons[(ons > 1.0) & (ons < 55.0)])
    assert np.mean(ioi) == pytest.approx(60.0 / BPM, rel=0.05)
    assert np.std(ioi) / np.mean(ioi) < 0.15


def test_drift_linear_ramp_middle_8_bars(rendered):
    audio, truth = rendered["drift"]
    bpms = np.array([b for _, b in truth["bpm_curve"]])
    s, e = (BARS - 8) // 2 * 4, (BARS - 8) // 2 * 4 + 32
    assert np.all(bpms[: s + 1] == BPM)                       # flat before
    assert np.all(bpms[e:] == pytest.approx(BPM * 1.06))      # flat after, +6%
    ramp = bpms[s:e + 1]
    assert np.all(np.diff(ramp) > 0)                          # monotonic ramp
    assert np.allclose(np.diff(ramp), np.diff(ramp)[0])       # linear
    # audible: early beats slower than late beats by the drift ratio
    bt = truth["bar_times_s"]
    ons = detect_onsets(audio, min_sep_s=0.15)
    early = np.diff(ons[(ons > bt[1]) & (ons < bt[6])])
    late = np.diff(ons[(ons > bt[18]) & (ons < bt[23])])
    assert np.mean(early) / np.mean(late) == pytest.approx(1.06, abs=0.02)


def test_busy_middle_triples_onset_density(rendered):
    audio, truth = rendered["busy_middle"]
    busy = truth["busy_bars"]
    assert len(busy) == 4 and busy == list(range(busy[0], busy[0] + 4))
    edges = list(truth["bar_times_s"]) + [audio.shape[0] / SR]
    ons = detect_onsets(audio)
    counts = [int(np.sum((ons >= edges[b]) & (ons < edges[b + 1])))
              for b in range(truth["bars"])]
    calm = [counts[b] for b in range(1, truth["bars"] - 1) if b not in busy]
    calm_mean = float(np.mean(calm))
    assert 3.0 <= calm_mean <= 6.0                       # ~1 onset/beat baseline
    for b in busy:                                       # tripled density
        assert counts[b] >= 2.5 * calm_mean, (b, counts[b], calm_mean)


def test_rubato_intro_pulse_free_then_steady(rendered):
    audio, truth = rendered["rubato_intro"]
    intro_end = truth["intro_end_s"]
    # intro has actual playing in it
    intro_seg = audio[: int(intro_end * SR)].astype(np.float64)
    assert 20.0 * np.log10(np.sqrt(np.mean(intro_seg ** 2)) + 1e-12) > -35.0
    ons = detect_onsets(audio, min_sep_s=0.15)
    intro_ons = ons[ons < intro_end - 0.5]
    assert len(intro_ons) >= 6
    ioi = np.diff(intro_ons)
    assert np.std(ioi) / np.mean(ioi) > 0.4              # genuinely pulse-free
    post = np.diff(ons[(ons > intro_end + 1.0) & (ons < intro_end + 30.0)])
    assert np.std(post) / np.mean(post) < 0.15           # then a real pulse
    assert np.mean(post) == pytest.approx(60.0 / BPM, rel=0.05)


def test_high_register_above_c5(rendered):
    hi, _ = rendered["high_register"]
    lo, _ = rendered["steady"]
    # C5 = 523 Hz: the high fixture keeps essentially no energy below 500 Hz,
    # while the ordinary fixture (bass + mid-register strums) is dominated by it
    assert low_power_fraction(hi) < 0.01
    assert low_power_fraction(lo) > 0.5
    assert len(detect_onsets(hi)) >= BARS * 3            # still actually strumming


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_deterministic_per_seed(rendered):
    audio1, truth1 = rendered["steady"]
    audio2, truth2 = make_human_with_truth("steady")
    assert np.array_equal(audio1, audio2)
    assert truth1 == truth2


def test_different_seed_different_audio():
    a = make_human("steady", bars=2, seed=1)
    b = make_human("steady", bars=2, seed=2)
    assert a.shape == b.shape
    assert not np.array_equal(a, b)


def test_make_human_matches_with_truth():
    a = make_human("stops", bars=8, seed=3)
    b, truth = make_human_with_truth("stops", bars=8, seed=3)
    assert np.array_equal(a, b)
    assert truth["silent_bars"] == [3, 4]


# ===========================================================================
# v2 joiners — humans who join a machine already playing (SPEC-V2 section 6)
# ===========================================================================
#
# The rule this file is written under: a claimed deviation that is only in the
# truth dict is not a deviation, it is a label.  Every kind's advertised
# property is therefore measured IN THE RENDERED AUDIO by plume's own onset
# detector (``plume.listen.features``) and its own harmony tracker, with the
# truth dict used only to say what the answer should be.
#
# The measurement stack, and why it has two stages:
#   1. ``feature_onsets`` — the real detector, hop 512 (10.67ms).  Its ``t_s``
#      is the END of the 42.7ms analysis window containing the attack, so it
#      reads ~11ms LATE and quantized to the hop.  Measured on ``on_grid``:
#      median -10.7ms against a grid the fixture is exactly on.  A raw
#      hop-resolution reading cannot separate a 30ms rush from a 35ms drag with
#      any margin.
#   2. ``attack_times`` — sub-hop refinement.  Around each detected frame,
#      walk back down a 1ms RMS envelope from the local peak to where it last
#      sat under 15% of that peak: the classic attack-time estimate.  Measured
#      accuracy against the fixtures' own truth: within 3-5ms on every kind.
#      That is what makes "+30ms ahead" an assertable number rather than a
#      direction.
# Both are per-DETECTION python loops over a vectorized envelope, never per
# sample.

from pathlib import Path                                        # noqa: E402

import soundfile as sf                                          # noqa: E402

from fixtures_gen import JOINER_KINDS, make_joiner              # noqa: E402
from plume.compose import PromptSpec, compose                   # noqa: E402
from plume.engine import render                                 # noqa: E402
from plume.listen.features import FeatureExtractor              # noqa: E402
from plume.listen.harmony import HarmonyTracker                 # noqa: E402
from plume.theory import parse_key as _parse_key                # noqa: E402

JBPM = 120.0
JKEY = "A minor"
JBARS = 24

_ENV_HOP = 48              # 1.0ms envelope hop for attack refinement
_ATTACK_BACK_S = 0.060     # how far back of a detection an attack may sit
_ATTACK_FWD_S = 0.006
_ATTACK_FRAC = 0.15        # envelope fraction defining "the attack started"
# Beat-matching window.  Deliberately narrower than half a beat: at 4 onsets
# per beat (``takes_over``) the 16ths sit 125ms off the beat, and a wide window
# would fold them into the beat's offset distribution.
_BEAT_TOL_S = 0.070


# ---------------------------------------------------------------------------
# joiner measurement helpers
# ---------------------------------------------------------------------------

def feature_onsets(x: np.ndarray) -> np.ndarray:
    """Onset times (s) from plume's own causal detector."""
    fe = FeatureExtractor(SR)
    frames = fe.process(np.asarray(x, dtype=np.float32)) + fe.flush()
    return np.array([f.t_s for f in frames if f.onset], dtype=np.float64)


# Memo: the joiner tests measure the same handful of buffers many times over,
# and one pass of the real detector over a 48s take costs ~0.2s.  Keyed by the
# exact bytes, so it can never return another buffer's answer.
_ATTACK_MEMO: dict[str, np.ndarray] = {}


def attack_times(x: np.ndarray) -> np.ndarray:
    """Sub-hop attack times for the detected onsets. See the note above."""
    import hashlib
    x = np.asarray(x, dtype=np.float64)
    key = hashlib.blake2b(np.ascontiguousarray(x).view(np.uint8),
                          digest_size=16).hexdigest()
    hit = _ATTACK_MEMO.get(key)
    if hit is not None:
        return hit
    out = _attack_times_uncached(x)
    _ATTACK_MEMO[key] = out
    return out


def _attack_times_uncached(x: np.ndarray) -> np.ndarray:
    n = (x.size // _ENV_HOP) * _ENV_HOP
    if n < _ENV_HOP:
        return np.zeros(0)
    env = np.sqrt(np.mean(x[:n].reshape(-1, _ENV_HOP) ** 2, axis=1))
    back = int(round(_ATTACK_BACK_S * SR / _ENV_HOP))
    fwd = int(round(_ATTACK_FWD_S * SR / _ENV_HOP))
    out = []
    for t in feature_onsets(x):
        c = int(round(t * SR / _ENV_HOP))
        a, b = max(0, c - back), min(env.size, c + fwd)
        if b - a < 3:
            continue
        w = env[a:b]
        pk = int(np.argmax(w))
        thr = _ATTACK_FRAC * w[pk]
        j = pk
        while j > 0 and w[j - 1] > thr:
            j -= 1
        out.append((a + j) * _ENV_HOP / SR)
    return np.array(out, dtype=np.float64)


def measured_offset_ms(x: np.ndarray, beats, tol: float = _BEAT_TOL_S):
    """(median signed offset in ms, n) of attacks against a machine grid.

    POSITIVE = the human is AHEAD of the machine, matching
    ``truth["mean_offset_ms"]``.  Median, not mean: a decaying pluck
    occasionally clears the flux threshold a second time one ``min_sep`` later
    (~16 spurious per 96 measured), and those land ~64ms late.  The median is
    unmoved by them; the mean is dragged ~9ms.
    """
    beats = np.asarray(beats, dtype=np.float64)
    d = []
    for t in attack_times(x):
        i = int(np.argmin(np.abs(beats - t)))
        if abs(beats[i] - t) < tol:
            d.append((beats[i] - t) * 1000.0)
    return (float(np.median(d)) if d else float("nan")), len(d)


def mean_chroma(x: np.ndarray) -> np.ndarray:
    """L1-normalized sum of the frame chroma over the whole take."""
    fe = FeatureExtractor(SR)
    frames = fe.process(np.asarray(x, dtype=np.float32)) + fe.flush()
    acc = np.zeros(12)
    for f in frames:
        acc += np.asarray(f.chroma, dtype=np.float64)
    s = acc.sum()
    return acc / s if s > 0 else acc


def tracked_key(x: np.ndarray) -> str | None:
    fe = FeatureExtractor(SR)
    ht = HarmonyTracker()
    return ht.process(fe.process(np.asarray(x, dtype=np.float32))
                      + fe.flush()).key


def key_or_relative(reported: str | None, want: str) -> bool:
    """Same key, or its relative — a pitch-class histogram cannot tell them
    apart in principle (see listen/harmony.py _KEY_RELATIVE_STICKY)."""
    if reported is None:
        return False
    a, b = _parse_key(want), _parse_key(reported)
    if (a.tonic, a.mode) == (b.tonic, b.mode):
        return True
    if a.mode == "major" and b.mode == "minor":
        return (a.tonic + 9) % 12 == b.tonic
    if a.mode in ("minor", "harmonic_minor") and b.mode == "major":
        return (a.tonic + 3) % 12 == b.tonic
    return False


def best_rotation(ref: np.ndarray, other: np.ndarray) -> int:
    """Semitones ``other`` sits above ``ref``, by chroma cross-correlation."""
    return int(np.argmax([float(np.dot(ref, np.roll(other, -r)))
                          for r in range(12)]))


def joiner_bar_density(x: np.ndarray, truth: dict) -> list[float]:
    """Detected onsets per beat, per bar."""
    ons = feature_onsets(x)
    edges = list(truth["machine_bar_times_s"]) + [x.shape[0] / SR]
    return [float(np.sum((ons >= edges[b]) & (ons < edges[b + 1]))) / 4.0
            for b in range(truth["bars"])]


def joiner_bar_rms_db(x: np.ndarray, truth: dict) -> list[float]:
    edges = list(truth["machine_bar_times_s"]) + [x.shape[0] / SR]
    out = []
    for b in range(truth["bars"]):
        seg = x[int(edges[b] * SR): int(edges[b + 1] * SR)].astype(np.float64)
        out.append(20.0 * np.log10(float(np.sqrt(np.mean(seg ** 2))) + 1e-12))
    return out


@pytest.fixture(scope="module")
def joiners() -> dict[str, tuple[np.ndarray, dict]]:
    """Every joiner kind rendered once at the SPEC-V2 defaults."""
    return {k: make_joiner(k, machine_bpm=JBPM, machine_key=JKEY, bars=JBARS)
            for k in JOINER_KINDS}


@pytest.fixture(scope="module")
def machine_audio(tmp_path_factory) -> np.ndarray:
    """REAL MATERIAL: plume's own v0 render of a 120bpm A-minor trap session.

    Built in-test via compose + engine.render, never read from ``renders/``
    (gitignored — a test that reads one silently skips itself on a clean
    clone).  This is the machine the joiner is joining.
    """
    d = tmp_path_factory.mktemp("joiner_machine")
    path = Path(d) / "machine120.wav"
    session = compose(
        PromptSpec(styles=["trap"], bpm=JBPM, key=JKEY, movements=1, mood=[],
                   name="machine120"),
        seed=7,
    )
    render(session, out_path=str(path), movement=0, bars=(0, 16))
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR
    return x.mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------

def test_joiner_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown joiner kind"):
        make_joiner("shreds")


@pytest.mark.parametrize("kind,bars", [("late_join", 11), ("drops_out", 19),
                                       ("takes_over", 7)])
def test_joiner_too_few_bars_raises(kind, bars):
    with pytest.raises(ValueError, match="needs bars >="):
        make_joiner(kind, bars=bars)


def test_joiner_bad_bpm_raises():
    with pytest.raises(ValueError, match="machine_bpm must be positive"):
        make_joiner("on_grid", machine_bpm=0.0)


# ---------------------------------------------------------------------------
# every kind: format, duration against the MACHINE's grid, truth contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", JOINER_KINDS)
def test_joiner_format_and_duration(kind, joiners):
    audio, truth = joiners[kind]
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert np.all(np.isfinite(audio))
    assert float(np.abs(audio).max()) == pytest.approx(0.85, abs=0.01)
    # the machine's grid is the reference frame, so the take is exactly as
    # long as the machine's bars — not as long as the human's playing
    assert audio.shape[0] == int(round(JBARS * 4 * 60.0 / JBPM * SR))


@pytest.mark.parametrize("kind", JOINER_KINDS)
def test_joiner_truth_contract(kind, joiners):
    audio, truth = joiners[kind]
    for k in ("onset_times_s", "mean_offset_ms", "key", "join_bar",
              "density_curve", "silent_bars", "machine_bpm"):
        assert k in truth, f"missing contract key {k!r}"
    assert truth["machine_bpm"] == pytest.approx(JBPM)
    assert truth["bars"] == JBARS
    assert len(truth["density_curve"]) == JBARS
    _parse_key(truth["key"])
    # onset count is exactly what the density curve promises
    assert len(truth["onset_times_s"]) == sum(
        int(round(4 * d)) for d in truth["density_curve"])
    # onsets sorted, inside the take, index-aligned with their grid positions
    ons = np.array(truth["onset_times_s"])
    grid = np.array(truth["grid_times_s"])
    assert ons.shape == grid.shape
    assert np.all(np.diff(ons) > 0)
    assert ons[-1] < audio.shape[0] / SR
    # join_bar is the first bar that actually plays
    play = [b for b in range(JBARS) if truth["density_curve"][b] > 0]
    assert truth["join_bar"] == play[0]
    assert sorted(truth["silent_bars"]) == [b for b in range(JBARS)
                                            if truth["density_curve"][b] == 0]


@pytest.mark.parametrize("kind", JOINER_KINDS)
def test_joiner_truth_offset_matches_its_own_onsets(kind, joiners):
    """truth["mean_offset_ms"] is the realized value, not a restatement of
    the nominal one — and it is within jitter of the nominal."""
    _, truth = joiners[kind]
    ons = np.array(truth["onset_times_s"])
    grid = np.array(truth["grid_times_s"])
    assert truth["mean_offset_ms"] == pytest.approx(
        float(np.mean(grid - ons) * 1000.0), abs=1e-6)
    assert truth["mean_offset_ms"] == pytest.approx(
        truth["nominal_offset_ms"], abs=2.0)


@pytest.mark.parametrize("kind", JOINER_KINDS)
def test_joiner_silent_bars_exact_rest_loud(kind, joiners):
    audio, truth = joiners[kind]
    expected = {"late_join": list(range(8)),
                "drops_out": list(range(8, 20))}.get(kind, [])
    assert truth["silent_bars"] == expected
    edges = list(truth["machine_bar_times_s"]) + [audio.shape[0] / SR]
    rms = joiner_bar_rms_db(audio, truth)
    for b in range(JBARS):
        seg = audio[int(edges[b] * SR): int(edges[b + 1] * SR)]
        if b in truth["silent_bars"]:
            assert not np.any(seg), f"silent bar {b} has nonzero samples"
        else:
            assert rms[b] > -35.0, f"bar {b} unexpectedly quiet: {rms[b]:.1f}dB"


# ---------------------------------------------------------------------------
# the deviations, measured in the audio
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", JOINER_KINDS)
def test_joiner_offset_is_measurable_in_the_audio(kind, joiners):
    """The headline claim: what the truth says about timing is what a real
    onset detector recovers from the rendered samples."""
    audio, truth = joiners[kind]
    got, n = measured_offset_ms(audio, truth["machine_beat_times_s"])
    assert n >= 0.7 * 4 * len([b for b in range(JBARS)
                               if truth["density_curve"][b] > 0]), \
        f"only {n} attacks matched the machine grid"
    assert got == pytest.approx(truth["mean_offset_ms"], abs=8.0), \
        f"{kind}: measured {got:+.1f}ms, truth {truth['mean_offset_ms']:+.1f}ms"


def test_rushing_is_ahead_of_the_machine(joiners):
    audio, truth = joiners["rushing"]
    assert truth["nominal_offset_ms"] == pytest.approx(30.0)
    got, _ = measured_offset_ms(audio, truth["machine_beat_times_s"])
    assert got > 20.0, f"rushing measured only {got:+.1f}ms ahead"
    base, _ = measured_offset_ms(joiners["on_grid"][0],
                                 truth["machine_beat_times_s"])
    assert got - base == pytest.approx(30.0, abs=6.0)


def test_dragging_is_behind_the_machine(joiners):
    audio, truth = joiners["dragging"]
    assert truth["nominal_offset_ms"] == pytest.approx(-35.0)
    got, _ = measured_offset_ms(audio, truth["machine_beat_times_s"])
    assert got < -25.0, f"dragging measured only {got:+.1f}ms behind"
    base, _ = measured_offset_ms(joiners["on_grid"][0],
                                 truth["machine_beat_times_s"])
    assert got - base == pytest.approx(-35.0, abs=6.0)


def test_on_grid_is_neither(joiners):
    """Guards the whole suite: if on_grid drifted, every other measurement
    above would be measuring the fixture generator, not the deviation."""
    audio, truth = joiners["on_grid"]
    assert truth["nominal_offset_ms"] == 0.0
    assert truth["silent_bars"] == []
    assert truth["join_bar"] == 0
    assert all(d == 1.0 for d in truth["density_curve"])
    got, _ = measured_offset_ms(audio, truth["machine_beat_times_s"])
    assert abs(got) < 8.0, f"on_grid measured {got:+.1f}ms off its own grid"
    assert key_or_relative(tracked_key(audio), JKEY)


def test_late_join_is_silent_then_enters_on_a_boundary(joiners):
    audio, truth = joiners["late_join"]
    assert truth["join_bar"] == 8
    bar8 = truth["machine_bar_times_s"][8]
    att = attack_times(audio)
    assert att.size, "no attacks detected at all"
    assert att.min() >= bar8 - 0.030, \
        f"played before the join bar (first attack {att.min():.3f}s)"
    assert att.min() == pytest.approx(bar8, abs=0.030), \
        "did not enter ON the bar-8 boundary"
    # and once in, it is in time
    got, _ = measured_offset_ms(audio, truth["machine_beat_times_s"])
    assert abs(got) < 8.0


def test_drops_out_plays_eight_bars_then_twelve_silent(joiners):
    audio, truth = joiners["drops_out"]
    assert truth["silent_bars"] == list(range(8, 20))
    dens = joiner_bar_density(audio, truth)
    assert all(d == 0.0 for d in dens[8:20]), "detected onsets in the dropout"
    assert np.mean(dens[:8]) > 0.8, "did not play the first 8 bars"
    assert np.mean(dens[20:]) > 0.8, "did not come back in"


def test_takes_over_density_ramps_to_four(joiners):
    audio, truth = joiners["takes_over"]
    curve = truth["density_curve"]
    assert curve[0] == 1.0
    assert max(curve) == 4.0
    assert curve[-1] == 4.0
    assert all(b >= a for a, b in zip(curve, curve[1:])), "ramp is not monotone"
    # measurable: the last bars really carry ~4 attacks per beat
    dens = joiner_bar_density(audio, truth)
    assert np.mean(dens[:4]) == pytest.approx(1.0, abs=0.4)
    assert np.mean(dens[-4:]) >= 3.5
    assert np.mean(dens[-4:]) > 2.5 * np.mean(dens[:4])


@pytest.mark.parametrize("kind,semitones", [("on_grid", 0), ("wrong_key", 1),
                                            ("far_key", 6)])
def test_joiner_key_offset_is_in_the_pitch_content(kind, semitones, joiners):
    """The key kinds are the SAME music transposed, so the audio's chroma is
    the on_grid chroma rotated by exactly the claimed interval."""
    ref = mean_chroma(joiners["on_grid"][0])
    audio, truth = joiners[kind]
    assert best_rotation(ref, mean_chroma(audio)) == semitones
    want = _parse_key(truth["key"])
    base = _parse_key(JKEY)
    assert (base.tonic + semitones) % 12 == want.tonic
    assert want.mode == base.mode
    assert key_or_relative(tracked_key(audio), truth["key"])


def test_wrong_key_is_conformably_close_far_key_is_not(joiners):
    """SPEC-V2 section 1: conform_semitones defaults to 2, so wrong_key must
    be inside that window and far_key must be outside it — otherwise behavior
    test 5 (transpose vs hold) is testing nothing."""
    def dist(k):
        a, b = _parse_key(JKEY).tonic, _parse_key(k).tonic
        return min((a - b) % 12, (b - a) % 12)
    assert dist(joiners["wrong_key"][1]["key"]) == 1
    assert dist(joiners["far_key"][1]["key"]) == 6


# ---------------------------------------------------------------------------
# REAL MATERIAL — the axis v1 shipped without
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gain_db", [0.0, -20.0, -40.0])
def test_joiner_deviation_survives_the_gain_axis(gain_db, joiners):
    """v1's sideman never entered because bar-silence was judged against an
    ABSOLUTE floor, and every fixture was peak-normalized to 0.85 so no test
    ever moved off that one level.  A joiner's deviation is a property of its
    timing and must be recoverable wherever the fader is.

    -40 dB is the bottom of the usable range, measured: at -55 dB the flux
    detector stops finding onsets at all and the measurement returns nothing.
    """
    g = 10.0 ** (gain_db / 20.0)
    for kind in ("on_grid", "rushing", "dragging"):
        audio, truth = joiners[kind]
        got, n = measured_offset_ms(audio * g, truth["machine_beat_times_s"])
        assert n > 50, f"{kind} @ {gain_db}dB: only {n} attacks found"
        assert got == pytest.approx(truth["mean_offset_ms"], abs=8.0), \
            f"{kind} @ {gain_db}dB: {got:+.1f} vs {truth['mean_offset_ms']:+.1f}"


def test_joiner_survives_a_pcm24_wav_round_trip(tmp_path, joiners):
    """Every real use of these fixtures goes through a file."""
    audio, truth = joiners["rushing"]
    p = tmp_path / "rushing.wav"
    sf.write(str(p), audio, SR, subtype="PCM_24")
    back, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == SR
    got, _ = measured_offset_ms(back.mean(axis=1),
                                truth["machine_beat_times_s"])
    assert got == pytest.approx(truth["mean_offset_ms"], abs=8.0)


def test_machine_audio_is_real_material(machine_audio):
    """Guard the guard: if the v0 render were silent or truncated, the mix
    tests below would pass by measuring the joiner alone."""
    assert machine_audio.dtype == np.float32
    assert np.all(np.isfinite(machine_audio))
    assert machine_audio.shape[0] / SR > 30.0        # 16 bars @ 120bpm + tail
    assert float(np.abs(machine_audio).max()) > 0.5
    ons = feature_onsets(machine_audio)
    assert len(ons) > 100, "the machine render has no detectable onsets"


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _matched_gain(machine: np.ndarray, human: np.ndarray, truth: dict) -> float:
    """Gain putting the human's PLAYING bars at the machine's RMS.

    Not a constant: the joiner is peak-normalized to 0.85 but is far sparser
    than a trap render, so equal peaks are ~10dB unequal in RMS.  Measured
    here: machine 0.569 RMS, drops_out's played bars 0.172 — a naive 0.7x mix
    puts the human 10dB under and buys the mix tests nothing.
    """
    bar_s = 4.0 * 60.0 / truth["machine_bpm"]
    edges = list(truth["machine_bar_times_s"]) + [
        truth["machine_bar_times_s"][-1] + bar_s]
    played = np.concatenate([
        human[int(edges[b] * SR): int(edges[b + 1] * SR)]
        for b in range(truth["bars"]) if truth["density_curve"][b] > 0])
    return _rms(machine) / _rms(played)


def test_joiner_is_audible_over_the_real_machine(machine_audio, joiners):
    """Mixed into plume's own v0 trap render, the human is not just present
    but LOAD-BEARING: the mix measurably quiets where drops_out stops.

    The bar-RMS comparison is made against the same mix with the human
    removed, so what is being measured is the human's contribution and not
    the machine's own arrangement dynamics.
    """
    audio, truth = joiners["drops_out"]
    n = min(machine_audio.shape[0], audio.shape[0])
    solo = machine_audio[:n]
    mix = solo + audio[:n] * _matched_gain(solo, audio[:n], truth)
    edges = list(truth["machine_bar_times_s"])

    def bar_db(x, b):
        s = x[int(edges[b] * SR): int(edges[b + 1] * SR)].astype(np.float64)
        return 20.0 * np.log10(float(np.sqrt(np.mean(s ** 2))) + 1e-12)

    played = [bar_db(mix, b) - bar_db(solo, b) for b in range(2, 8)]
    dropped = [bar_db(mix, b) - bar_db(solo, b) for b in range(10, 16)]
    assert np.mean(played) > 2.5, \
        f"human adds only {np.mean(played):.2f}dB while playing"
    assert np.mean(dropped) == pytest.approx(0.0, abs=0.05), \
        "the machine-only span is not machine-only"
    assert np.mean(played) - np.mean(dropped) > 2.5


def test_summed_mix_timing_reading_depends_on_the_fader(machine_audio, joiners):
    """A MEASURED NEGATIVE RESULT, asserted so it cannot be forgotten.

    SPEC-V2 section 4 says the human's beat tracking must run on the human's
    SIGNAL ONLY.  This is the experiment behind that sentence.  When a joiner
    is summed with the machine, the machine's own on-grid attacks sit inside
    ONSET_MIN_SEP_S (60ms) of the human's, so the detector emits ONE onset per
    beat somewhere between the two — and where between them depends on their
    relative level.  Measured on ``dragging`` (truth -34.7ms) over this trap
    render: -17.5ms at matched RMS, -32.0ms with the human 6dB hotter, while
    the isolated human reads -38.0ms at BOTH.

    So: an isolated reading is a property of the playing; a summed-mix reading
    is a property of the mix balance.  A tempo negotiation driven by the
    second one is partly reading its own clock back.
    """
    audio, truth = joiners["dragging"]
    n = min(machine_audio.shape[0], audio.shape[0])
    solo, hum = machine_audio[:n], audio[:n]
    beats = truth["machine_beat_times_s"]
    g = _matched_gain(solo, hum, truth)

    mix_a, _ = measured_offset_ms(solo + hum * g, beats)
    mix_b, _ = measured_offset_ms(solo + hum * (g * 2.0), beats)
    iso_a, _ = measured_offset_ms(hum * g, beats)
    iso_b, _ = measured_offset_ms(hum * (g * 2.0), beats)

    assert abs(iso_a - iso_b) <= 3.0, \
        f"isolated reading moved with the fader: {iso_a:+.1f} -> {iso_b:+.1f}"
    assert abs(mix_a - mix_b) > 8.0, (
        f"summed-mix reading barely moved with the fader "
        f"({mix_a:+.1f} -> {mix_b:+.1f}); if the detector now genuinely "
        "separates the two players, replace this test with the positive claim")
    # ...and at the quieter balance the mix is wrong about the human by more
    # than the isolated reading ever is
    assert abs(mix_a - truth["mean_offset_ms"]) > 8.0
    assert abs(iso_a - truth["mean_offset_ms"]) <= 8.0


# ---------------------------------------------------------------------------
# parameterization + determinism
# ---------------------------------------------------------------------------

def test_joiner_honours_machine_bpm_and_key():
    audio, truth = make_joiner("on_grid", machine_bpm=90.0,
                               machine_key="F major", bars=12, seed=4)
    assert truth["machine_bpm"] == pytest.approx(90.0)
    assert truth["key"] == "F major"
    assert audio.shape[0] == int(round(12 * 4 * 60.0 / 90.0 * SR))
    beats = truth["machine_beat_times_s"]
    assert len(beats) == 48
    assert beats[1] - beats[0] == pytest.approx(60.0 / 90.0)
    got, _ = measured_offset_ms(audio, beats)
    assert abs(got) < 8.0
    assert key_or_relative(tracked_key(audio), "F major")


def test_joiner_transposes_relative_to_the_machine_key():
    _, a = make_joiner("wrong_key", machine_key="F major", bars=8)
    _, b = make_joiner("far_key", machine_key="F major", bars=8)
    assert _parse_key(a["key"]).tonic == (_parse_key("F major").tonic + 1) % 12
    assert _parse_key(b["key"]).tonic == (_parse_key("F major").tonic + 6) % 12
    assert _parse_key(a["key"]).mode == "major"


@pytest.mark.parametrize("kind", JOINER_KINDS)
def test_joiner_deterministic_per_seed(kind, joiners):
    audio, truth = joiners[kind]
    again, truth2 = make_joiner(kind, machine_bpm=JBPM, machine_key=JKEY,
                                bars=JBARS)
    assert np.array_equal(audio, again)
    assert truth == truth2


def test_joiner_different_seed_different_audio():
    a, ta = make_joiner("on_grid", bars=4, seed=1)
    b, tb = make_joiner("on_grid", bars=4, seed=2)
    assert a.shape == b.shape
    assert not np.array_equal(a, b)
    assert ta["machine_beat_times_s"] == tb["machine_beat_times_s"]  # same grid


# ---------------------------------------------------------------------------
# the v1 fixtures are untouched
# ---------------------------------------------------------------------------

# sha256 of the float32 samples, recorded BEFORE the v2 joiner code was added
# to fixtures_gen.py.  Other tests depend on these bytes; this is the
# guard that "additive only" stayed additive.
_V1_DIGESTS = {
    "steady": "022f50fc06d95aa289a7da120b75d649f8df5048b52ffc16d0b62cb427e0a1f7",
    "drift": "d4f173a287430c8aa59e2b7a7cbc480f4a022770b01799302902d41c49f84d20",
    "stops": "149738a6ca65f31c81db53887a51a8471031b979797ad8e771945e4a3f2e70c7",
    "long_stop": "3fbc75bd29e9bee04373e77b68632dc0a8d58f1b11b102df368f52539359ac5a",
    "busy_middle": "b4ee1e40e4c4a34801ee89ff7f2c9a253b348dd02370f32cb64a0be7ccb52265",
    "rubato_intro": "0299a321f430e6ecd2054af8d915ef48443d6891903a91d986712f859908a2b0",
    "high_register": "674cf9525ef4ea8c0ff5216c5b4bd81ca8532049963eefa8317108fafb636d00",
}
_V1_DIGESTS_ALT = {
    "steady": "35c06841ce8e6f2feb4d66f4b3c32fe13bd43c8d9a6d2c763ae82c29e80385a6",
    "stops": "8d434cbdfcc662a1e2962359e16241dc6e7d552918e2c10f9214450e671172d0",
}


def _check_digest(got: str, want: str) -> None:
    """These digests were recorded on macOS arm64, where they are enforced.

    numpy's float math is not bit-identical across platforms (an x86 Linux runner
    renders some kinds a few ulps apart), so elsewhere a different digest says
    nothing about the code, and is skipped with that reason rather than failed.
    """
    import platform
    import sys
    if got != want and not (sys.platform == "darwin" and platform.machine() == "arm64"):
        pytest.skip("fixture digests are recorded and enforced on macOS arm64")


@pytest.mark.parametrize("kind", KINDS)
def test_v1_fixture_bytes_unchanged(kind, rendered):
    import hashlib
    audio, _ = rendered[kind]
    got = hashlib.sha256(audio.tobytes()).hexdigest()
    _check_digest(got, _V1_DIGESTS[kind])
    assert got == _V1_DIGESTS[kind], (
        f"make_human({kind!r}) changed — other tests depend on "
        "these exact samples")


@pytest.mark.parametrize("kind", sorted(_V1_DIGESTS_ALT))
def test_v1_fixture_bytes_unchanged_off_defaults(kind):
    import hashlib
    audio = make_human(kind, bpm=110, bars=12, key="F major", seed=5)
    got = hashlib.sha256(audio.tobytes()).hexdigest()
    _check_digest(got, _V1_DIGESTS_ALT[kind])
    assert got == _V1_DIGESTS_ALT[kind]


def test_v1_kinds_list_is_unchanged():
    assert KINDS == ("steady", "drift", "stops", "long_stop",
                     "busy_middle", "rubato_intro", "high_register")
    assert set(KINDS).isdisjoint(JOINER_KINDS)
