"""The curated synths and pads (plume.synths): stereo, band-limited, moving,
velocity-aware, loudness-matched, and cheap enough for the live planner."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plume import synths  # noqa: E402
from plume.styles_lib import list_styles, load_style  # noqa: E402
from plume.voices import (PRESETS, is_stereo, preset_info, render_note,  # noqa: E402
                          render_note_stereo)

SR = 48000
NEW = sorted(synths.RENDERERS)


def _st(name, f=261.63, dur=1.0, vel=0.7, seed=3, sr=SR, params=None):
    return render_note_stereo(name, sr=sr, dur_s=dur, freq_hz=f, velocity=vel,
                              rng=np.random.default_rng(seed),
                              params=params or {}).astype(np.float64)


def _centroid(x, a, b, sr=SR):
    m = x.mean(axis=1) if x.ndim == 2 else x
    seg = m[int(a * sr):int(b * sr)] * np.hanning(int(b * sr) - int(a * sr))
    s = np.abs(np.fft.rfft(seg)) ** 2
    fr = np.fft.rfftfreq(seg.size, 1 / sr)
    return float((fr * s).sum() / s.sum())


@pytest.mark.parametrize("name", NEW)
def test_renders_valid_stereo(name):
    y = _st(name)
    assert y.ndim == 2 and y.shape[1] == 2
    assert np.isfinite(y).all()
    assert np.abs(y).max() <= 1.0
    assert np.abs(y).max() > 0.01
    assert is_stereo(name)


@pytest.mark.parametrize("name", NEW)
def test_mono_fold_is_render_note(name):
    st = _st(name)
    mono = render_note(name, sr=SR, dur_s=1.0, freq_hz=261.63, velocity=0.7,
                       rng=np.random.default_rng(3), params={})
    assert np.allclose(st.mean(axis=1), mono, atol=1e-6)


@pytest.mark.parametrize("name", NEW)
def test_deterministic(name):
    assert np.array_equal(_st(name, seed=9), _st(name, seed=9))


def test_mono_voices_come_back_dual_mono():
    y = _st("epiano")
    assert np.array_equal(y[:, 0], y[:, 1])
    assert not is_stereo("epiano")


@pytest.mark.parametrize("name", [n for n in NEW if n != "synthbass"])
def test_has_stereo_width_but_stays_mono_safe(name):
    y = _st(name, dur=1.5)
    mid = 0.5 * (y[:, 0] + y[:, 1])
    side = 0.5 * (y[:, 0] - y[:, 1])
    ratio_db = 10 * np.log10((side ** 2).sum() / (mid ** 2).sum())
    assert -20.0 < ratio_db < -1.0, ratio_db


def test_bass_is_centred():
    y = _st("synthbass", f=55.0)
    assert np.array_equal(y[:, 0], y[:, 1])


# Noise layers are switched off: the aliasing check compares a 48 kHz render
# against a 4x-oversampled one, and noise is never sample-identical.
_QUIET = {"breathpad": {"breath": 0.0}, "organ": {"click": 0.0}}


@pytest.mark.parametrize("name", NEW)
def test_band_limited_at_the_top_of_the_range(name):
    f = 130.81 if name == "synthbass" else 1046.5
    p = _QUIET.get(name, {})
    a = _st(name, f=f, dur=0.5, params=p).mean(axis=1)
    b = resample_poly(_st(name, f=f, dur=0.5, sr=4 * SR, params=p).mean(axis=1), 1, 4)
    n = min(a.size, b.size)
    a, b = a[2400:n - 2400], b[2400:n - 2400]
    w = 4096
    fr = np.fft.rfftfreq(w, 1 / SR)
    g = f / 2.0 if name == "junopoly" else f        # its sub sits an octave down
    k = np.maximum(np.round(fr / g), 1.0)
    off = np.abs(fr - k * g) > np.maximum(40.0, 0.03 * k * g)
    ex = eb = 0.0
    for i in range(0, a.size - w, w // 2):
        A = np.abs(np.fft.rfft(a[i:i + w] * np.hanning(w))) ** 2
        B = np.abs(np.fft.rfft(b[i:i + w] * np.hanning(w))) ** 2
        ex += np.maximum(A - B, 0.0)[off].sum()
        eb += B.sum()
    alias_db = 10 * np.log10(ex / eb + 1e-12)
    assert alias_db < -40.0, alias_db


@pytest.mark.parametrize("name", ["warmpad", "strings", "breathpad", "junopoly"])
def test_held_pads_move(name):
    """A held chord should not be a frozen spectrum."""
    y = _st(name, dur=3.0)
    cs = [_centroid(y, a, a + 0.25) for a in np.arange(0.6, 2.7, 0.25)]
    assert np.std(cs) / np.mean(cs) > 0.01, cs


@pytest.mark.parametrize("name", ["warmpad", "glasskeys", "junopoly", "synthlead",
                                  "synthbass", "tapekeys", "bell", "organ"])
def test_velocity_brightens(name):
    f = 55.0 if name == "synthbass" else 261.63
    soft = _st(name, f=f, vel=0.25)
    hard = _st(name, f=f, vel=1.0)
    assert _centroid(hard, 0.02, 0.2) > 1.03 * _centroid(soft, 0.02, 0.2)


def test_presets_are_listed_with_descriptions():
    info = preset_info()
    for name in NEW:
        assert name in PRESETS and info[name]["description"]
        assert set(info[name]["params"]) == set(synths.PRESETS[name])


def test_pads_hold_then_release():
    y = _st("warmpad", dur=1.0)
    rms = lambda a, b: float(np.sqrt((y[int(a * SR):int(b * SR)] ** 2).mean()))  # noqa: E731
    assert rms(0.7, 0.95) > 4 * rms(y.shape[0] / SR - 0.2, y.shape[0] / SR)


@pytest.mark.parametrize("name", sorted(list_styles()))
def test_style_variants_are_real_presets(name):
    s = load_style(name)
    for role, options in (s.get("voice_variants") or {}).items():
        assert role in s["voices"], (name, role)
        assert options, (name, role)
        for preset in options:
            assert preset in PRESETS, (name, role, preset)


def test_a_held_pad_bar_renders_well_inside_real_time():
    """The planner renders a bar ahead of time.  A four-note held chord on the
    heaviest pad must take a small fraction of the bar it fills, so slow
    machines keep up (best of three, to ride out a busy CPU)."""
    spb = 60.0 / 85.0
    best = 1e9
    for r in range(3):
        rng = np.random.default_rng(r)
        t0 = time.perf_counter()
        for m in (48, 52, 55, 59):
            render_note("warmpad", sr=SR, dur_s=3.92 * spb,
                        freq_hz=440.0 * 2 ** ((m - 69) / 12), velocity=0.5,
                        rng=rng, params={})
        best = min(best, time.perf_counter() - t0)
    assert best < 0.25 * 4 * spb, best


# ---------------------------------------------------------------------------
# in the band (DuetEngine)
# ---------------------------------------------------------------------------

def _duet_bar(style, bar, roles, bpm=100.0):
    from plume.duet import DuetEngine
    from plume.listen.beat import BeatState
    from plume.listen.harmony import HarmonyState
    from plume.mind import BarPlan
    iv = 60.0 / bpm
    eng = DuetEngine(style=style, seed=11)
    plan = BarPlan(bar=bar, state="comping", roles=roles, gain_db=0.0,
                   density=0.6, register_ceiling_midi=None, reason="test")
    beat = BeatState(t_s=0.0, bpm=bpm, confidence=0.9, locked=True, bar=bar,
                     beat_in_bar=0, phase_beats=0.0, next_beat_t_s=iv,
                     next_downbeat_t_s=4 * iv, beat_interval_s=iv)
    harm = HarmonyState(t_s=0.0, key="A minor", key_conf=0.8, chord="Am",
                        chord_conf=0.9, register_low_midi=None,
                        register_high_midi=None)
    out = eng.render_bar(plan, beat, harm)
    return out, eng.last_events


def test_a_stereo_pad_is_wide_in_the_band():
    out, events = _duet_bar("synthwave", 0, {"pad": 1.0})
    assert {e["preset"] for e in events} == {"warmpad"}
    side = out[:, 0] - out[:, 1]
    assert np.sqrt((side ** 2).mean()) > 0.05 * np.sqrt((out ** 2).mean())


def test_a_mono_voice_is_still_placed_by_the_pan_law():
    out, events = _duet_bar("lofi", 0, {"hat": 1.0})
    assert events
    assert not np.allclose(out[:, 0], out[:, 1])   # hats sit right of centre


def test_voice_variants_change_colour_between_sections():
    every = int(load_style("synthwave").get("variant_bars", 16))
    seen = [_duet_bar("synthwave", b * every, {"pad": 1.0})[1][0]["preset"]
            for b in range(3)]
    assert seen == ["warmpad", "warmpad", "strings"]
