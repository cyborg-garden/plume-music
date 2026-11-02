"""Tests for plume.voices (SPEC sections 7 and 12)."""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from plume.voices import PRESETS, preset_info, render_note  # noqa: E402

SR = 48000
DRUMS = ["808kick", "hat", "openhat", "clap", "snare", "crackle"]


def _render(preset, freq_hz=440.0, seed=42, dur_s=0.5, velocity=0.8, params=None,
            glide_from_hz=None):
    return render_note(
        preset, sr=SR, dur_s=dur_s, freq_hz=freq_hz, velocity=velocity,
        rng=np.random.default_rng(seed), params=params or {},
        glide_from_hz=glide_from_hz,
    )


def _assert_valid(y):
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert y.size > 0
    assert np.all(np.isfinite(y))
    assert float(np.max(np.abs(y))) <= 1.0 + 1e-6
    assert float(np.max(np.abs(y))) > 1e-3, "silent render"


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_renders_at_440(preset):
    _assert_valid(_render(preset, freq_hz=440.0))


@pytest.mark.parametrize("preset", DRUMS)
def test_drums_render_with_freq_none(preset):
    _assert_valid(_render(preset, freq_hz=None))


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        _render("nope")


def test_determinism_same_rng_seed():
    for preset in ("hat", "clap", "pad", "crackle"):
        a = _render(preset)
        b = _render(preset)
        assert np.array_equal(a, b), preset


def test_different_seed_differs_for_noisy_voices():
    a = _render("clap", seed=1)
    b = _render("clap", seed=2)
    assert not np.array_equal(a, b)


def test_velocity_scales_level():
    loud = _render("snare", freq_hz=None, velocity=0.95)
    soft = _render("snare", freq_hz=None, velocity=0.2)
    assert float(np.max(np.abs(loud))) > float(np.max(np.abs(soft)))


def test_sub_glide_supported():
    y = _render("808sub", freq_hz=55.0, glide_from_hz=41.2)
    _assert_valid(y)
    # glide changes the waveform vs. no glide
    y2 = _render("808sub", freq_hz=55.0)
    m = min(y.size, y2.size)
    assert not np.array_equal(y[:m], y2[:m])


def test_param_override_changes_output():
    short = _render("808kick", freq_hz=None, params={"decay_ms": 100.0})
    long = _render("808kick", freq_hz=None)
    assert short.size < long.size  # natural tail follows decay_ms


def test_kick_is_low_frequency_dominant():
    y = _render("808kick", freq_hz=45.0, dur_s=0.5).astype(np.float64)
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, 1.0 / SR)
    low = spec[freqs < 200].sum()
    high = spec[freqs > 2000].sum()
    assert low > high


def test_hat_is_high_frequency_dominant():
    y = _render("hat", freq_hz=None).astype(np.float64)
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, 1.0 / SR)
    assert spec[freqs > 5000].sum() > spec[freqs < 1000].sum()


def test_openhat_longer_than_hat():
    assert _render("openhat", freq_hz=None).size > _render("hat", freq_hz=None).size


def test_gated_voices_track_duration():
    short = _render("pad", dur_s=0.25)
    long = _render("pad", dur_s=1.0)
    assert long.size > short.size


def test_preset_info_is_jsonable_and_complete():
    info = preset_info()
    json.dumps(info)  # must not raise
    assert set(info) == set(PRESETS)
    for name, entry in info.items():
        assert entry["description"]
        assert set(entry["params"]) == set(PRESETS[name])
        for pname, meta in entry["params"].items():
            assert "default" in meta
