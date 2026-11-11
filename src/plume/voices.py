"""plume.voices — the instruments (SPEC section 7).

Stateless pure render functions; the engine sums the returned buffers at
sample offsets. All output is mono float32, no NaN/inf, peak <= 1.0.
No per-sample Python loops: polyBLEP saws are computed with vectorized
mask formulas, FM via phase-array sine composition, filters via
scipy.signal.lfilter (see plume.fx). The pluck's swept filter uses
per-block biquad coefficients (64-sample blocks) with lfilter state
carried across blocks.

Conventions
-----------
- ``decay_ms`` / ``*_tau_ms`` params are exponential time constants
  (amplitude falls to 1/e in that time). Documented per preset in PRESETS.
- One-shot drums (kick, hat, clap, snare) render their natural decay tail,
  which may be longer than ``dur_s`` — the engine mixes by summation, so
  overlapping tails are correct. Gated voices (sub, pad, epiano, lead)
  hold for ``dur_s`` then release. ``crackle`` renders exactly ``dur_s``
  (the engine loops it per bar).
- ``velocity`` scales level everywhere and additionally: hat/openhat decay
  time, epiano FM index (brightness).
- ``freq_hz=None`` falls back to a per-preset default (drums ignore pitch
  except 808kick/808sub, which default to 45 Hz).
"""

from __future__ import annotations

import numpy as np

from . import samplekit, synths

from plume.fx import _svf, _svf_bandpass, svf_highpass
from scipy.signal import lfilter

__all__ = ["render_note", "render_note_stereo", "PRESETS", "preset_info",
           "is_stereo"]

_F32 = np.float32

# 808 hat oscillator bank (Hz), per TR-808 service data / Baratatronix analysis
_HAT_FREQS = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0])


# ---------------------------------------------------------------------------
# presets: defaults (all overridable via Track.params -> render_note(params=))
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "808kick": {
        "sweep_mult": 3.5,     # pitch env start = sweep_mult * f0
        "tau_ms": 30.0,        # pitch env time constant
        "decay_ms": 350.0,     # amp decay time constant
        "attack_ms": 1.0,
        "click_level": 0.3,    # noise click mix (scaled by velocity)
        "drive": 2.5,          # tanh drive
        "default_hz": 45.0,
    },
    "808sub": {
        "attack_ms": 8.0,
        "decay_ms": 900.0,
        "release_ms": 30.0,    # gate release at step end
        "drive": 1.5,
        "glide_ms": 60.0,      # exp pitch ramp when glide_from_hz given
        "default_hz": 45.0,
    },
    "hat": {
        "decay_ms": 40.0,
        "freq_scale": 1.0,     # scales the 6-square oscillator bank
        "level": 0.9,
    },
    "openhat": {
        "decay_ms": 400.0,
        "freq_scale": 1.0,
        "level": 0.8,
    },
    "clap": {
        "bp_hz": 1400.0,       # band-pass centre (1-2 kHz band)
        "bp_q": 1.5,
        "burst_spacing_ms": 11.0,
        "burst_tau_ms": 6.0,
        "tail_tau_ms": 66.0,   # ~200 ms audible tail
        "tail_level": 0.5,
    },
    "snare": {
        "tone_hz_1": 185.0,
        "tone_hz_2": 330.0,
        "tone_decay_ms": 100.0,
        "noise_decay_ms": 180.0,
        "mix": 0.5,            # noise fraction (0.5 = 50/50)
        "clip_drive": 1.5,
    },
    "pad": {
        "n_saws": 5,
        "detune_cents": 25.0,  # spread is +/- this
        "cutoff_hz": 1800.0,
        "q": 0.8,
        "attack_ms": 120.0,
        "release_ms": 300.0,
        "default_hz": 220.0,
    },
    "pluck": {
        "cutoff_start_hz": 4000.0,
        "cutoff_end_hz": 400.0,
        "filter_ms": 120.0,    # filter-env time span
        "decay_ms": 250.0,
        "q": 0.9,
        "default_hz": 440.0,
    },
    "epiano": {
        "index_start": 2.0,    # 2-op FM (1:1) index envelope
        "index_end": 0.3,
        "index_ms": 200.0,
        "tine_ratio": 14.0,    # 14:1 tine ping
        "tine_level": 0.12,
        "tine_tau_ms": 50.0,
        "decay_ms": 1200.0,
        "release_ms": 100.0,
        "default_hz": 440.0,
    },
    "lead": {
        "n_saws": 3,
        "detune_cents": 12.0,
        "sub_level": 0.3,      # square sub 1 octave down
        "glide_ms": 60.0,      # portamento when glide_from_hz given
        "attack_ms": 10.0,
        "release_ms": 80.0,
        "default_hz": 440.0,
    },
    "crackle": {
        "rate_hz": 90.0,       # mean impulses per second
        "lp_hz": 7000.0,
        "level": 0.8,
    },
    "housekick": {             # short, hard and clicky: it plays every beat
        "sweep_mult": 5.0,
        "tau_ms": 14.0,
        "decay_ms": 95.0,
        "attack_ms": 0.5,
        "click_level": 0.55,
        "drive": 3.0,
        "default_hz": 50.0,
    },
    "boomkick": {              # rounder and longer, with room for the snare
        "sweep_mult": 3.0,
        "tau_ms": 22.0,
        "decay_ms": 150.0,
        "attack_ms": 1.0,
        "click_level": 0.4,
        "drive": 3.5,
        "default_hz": 52.0,
    },
    "offhat": {                # an open hat short enough to clear the next kick
        "decay_ms": 75.0,
        "freq_scale": 1.0,
        "level": 0.85,
    },
    "roundbass": {
        "attack_ms": 6.0,
        "decay_ms": 1400.0,
        "release_ms": 45.0,
        "h2_level": 0.4,       # second and third harmonics carry it on small speakers
        "h3_level": 0.14,
        "drive": 1.3,
        "default_hz": 55.0,
    },
    "housestab": {
        "n_saws": 3,
        "detune_cents": 9.0,
        "cutoff_start_hz": 5200.0,
        "cutoff_end_hz": 800.0,
        "filter_ms": 110.0,
        "decay_ms": 170.0,
        "q": 1.1,
        "default_hz": 220.0,
    },
}

_DESCRIPTIONS = {
    "808kick": "808 kick: sine with exp pitch sweep + noise click, tanh drive. Tune freq_hz to the key's tonic (35-55 Hz).",
    "808sub": "808 sub bass: pure sine, gated by step, light drive; supports legato glide via glide_from_hz.",
    "hat": "Closed hat: 6 squares (TR-808 bank) -> bp 10k -> hp 7k, exp decay 40 ms; velocity scales level+decay.",
    "openhat": "Open hat: same oscillator bank, 400 ms decay.",
    "clap": "Clap: band-passed noise, 3 bursts spaced 11 ms + ~200 ms tail.",
    "snare": "Snare: 185+330 Hz sines + 1-8 kHz noise, 50/50, soft-clipped.",
    "pad": "Supersaw pad: 5 polyBLEP saws +/-25 cents -> SVF lowpass, slow attack/release.",
    "pluck": "Pluck: saw -> SVF lowpass with 4k->400 Hz filter envelope, 250 ms amp decay.",
    "epiano": "DX-style e-piano: 2-op FM (1:1, index 2->0.3) + 14:1 tine ping; velocity -> brightness.",
    "lead": "Lead: 3-saw supersaw + square sub -1 oct at 0.3; portamento via glide_from_hz.",
    "crackle": "Vinyl crackle bed: sparse impulses (~90/s) low-passed at 7 kHz; loops per bar.",
    "housekick": "House kick: short, hard pitch sweep and a loud click.",
    "boomkick": "Boom bap kick: rounder sweep, longer body, heavily driven.",
    "offhat": "Offbeat hat: the open hat's bank with a short decay.",
    "roundbass": "Round bass: sine with second and third harmonics, gated, lightly driven.",
    "housestab": "House stab: detuned saws through a fast closing filter.",
}

_PARAM_RANGES = {
    "sweep_mult": [1.0, 8.0], "tau_ms": [5.0, 120.0], "decay_ms": [20.0, 2000.0],
    "attack_ms": [0.1, 500.0], "click_level": [0.0, 1.0], "drive": [1.0, 6.0],
    "default_hz": [20.0, 2000.0], "release_ms": [5.0, 1000.0], "glide_ms": [10.0, 300.0],
    "freq_scale": [0.5, 2.0], "level": [0.0, 1.0], "bp_hz": [500.0, 4000.0],
    "bp_q": [0.3, 4.0], "burst_spacing_ms": [5.0, 20.0], "burst_tau_ms": [2.0, 20.0],
    "tail_tau_ms": [20.0, 200.0], "tail_level": [0.0, 1.0], "tone_hz_1": [100.0, 400.0],
    "tone_hz_2": [150.0, 600.0], "tone_decay_ms": [30.0, 300.0],
    "noise_decay_ms": [50.0, 500.0], "mix": [0.0, 1.0], "clip_drive": [1.0, 4.0],
    "n_saws": [1, 7], "detune_cents": [0.0, 50.0], "cutoff_hz": [100.0, 12000.0],
    "q": [0.5, 4.0], "cutoff_start_hz": [500.0, 12000.0], "cutoff_end_hz": [50.0, 2000.0],
    "filter_ms": [20.0, 500.0], "index_start": [0.0, 6.0], "index_end": [0.0, 2.0],
    "index_ms": [20.0, 1000.0], "tine_ratio": [1.0, 20.0], "tine_level": [0.0, 0.5],
    "tine_tau_ms": [10.0, 200.0], "sub_level": [0.0, 1.0], "rate_hz": [5.0, 500.0],
    "lp_hz": [1000.0, 16000.0], "h2_level": [0.0, 1.0], "h3_level": [0.0, 1.0],
}


def preset_info() -> dict:
    """JSON-able catalog: every preset, its params (default + range), description."""
    info: dict = {}
    for name, defaults in PRESETS.items():
        info[name] = {
            "description": _DESCRIPTIONS[name],
            "params": {
                p: {"default": v, "range": _PARAM_RANGES.get(p)}
                for p, v in defaults.items()
            },
        }
    return info


# ---------------------------------------------------------------------------
# helpers (all vectorized)
# ---------------------------------------------------------------------------

def _n_samples(sr: int, seconds: float) -> int:
    return max(1, int(round(seconds * sr)))


def _time(n: int, sr: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / sr


def _phase_from_freq(f: np.ndarray | float, n: int, sr: int, phi0: float = 0.0) -> np.ndarray:
    """Phase (radians) for constant or time-varying frequency."""
    if np.isscalar(f):
        return 2.0 * np.pi * float(f) * _time(n, sr) + phi0
    return 2.0 * np.pi * np.cumsum(np.asarray(f, dtype=np.float64)) / sr + phi0


def _attack_env(t: np.ndarray, attack_s: float) -> np.ndarray:
    return np.clip(t / max(attack_s, 1e-5), 0.0, 1.0)


def _gate_release(t: np.ndarray, dur_s: float, release_s: float) -> np.ndarray:
    """1 while gated, exponential release (tau = release_s / 3) after dur_s."""
    tau = max(release_s, 1e-4) / 3.0
    return np.where(t < dur_s, 1.0, np.exp(-(t - dur_s) / tau))


def _saw_polyblep(phase_rad: np.ndarray, freq: np.ndarray | float, sr: int) -> np.ndarray:
    """Anti-aliased sawtooth via the standard 2-sample polyBLEP correction.

    Fully vectorized: the two discontinuity-neighborhood corrections are
    applied with boolean masks.
    """
    p = np.mod(phase_rad / (2.0 * np.pi), 1.0)
    dt = np.broadcast_to(np.asarray(freq, dtype=np.float64) / sr, p.shape)
    dt = np.clip(dt, 1e-9, 0.49)
    y = 2.0 * p - 1.0
    m = p < dt
    if m.any():
        v = p[m] / dt[m]
        y[m] -= v + v - v * v - 1.0
    m = p > 1.0 - dt
    if m.any():
        v = (p[m] - 1.0) / dt[m]
        y[m] -= v * v + v + v + 1.0
    return y


def _supersaw(freq, n: int, sr: int, n_saws: int, detune_cents: float,
              rng: np.random.Generator) -> np.ndarray:
    """n detuned polyBLEP saws spread evenly across +/- detune_cents."""
    n_saws = max(1, int(n_saws))
    if n_saws == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-detune_cents, detune_cents, n_saws)
    out = np.zeros(n)
    f_arr = np.asarray(freq, dtype=np.float64)
    for c in offsets:
        fi = f_arr * (2.0 ** (c / 1200.0))
        ph = _phase_from_freq(fi if fi.ndim else float(fi), n, sr,
                              phi0=float(rng.uniform(0.0, 2.0 * np.pi)))
        out += _saw_polyblep(ph, fi, sr)
    return out / n_saws


def _swept_lowpass(x: np.ndarray, sr: int, cutoff: np.ndarray, q: float,
                   block: int = 64) -> np.ndarray:
    """SVF lowpass with time-varying cutoff: per-block biquad coefficients
    (lfilter with state carried across blocks). Loop is per 64-sample block,
    not per sample."""
    from plume.fx import _svf_coeffs
    n = x.shape[0]
    out = np.empty(n)
    zi = np.zeros(2)
    for i in range(0, n, block):
        j = min(i + block, n)
        b, a = _svf_coeffs(sr, float(cutoff[(i + j) // 2]), q, "lp")
        out[i:j], zi = lfilter(b, a, x[i:j], zi=zi)
    return out


def _glide_freq(f_target: float, glide_from: float | None, glide_s: float,
                n: int, sr: int) -> np.ndarray | float:
    """Exponential pitch ramp glide_from -> f_target over glide_s."""
    if glide_from is None or glide_from <= 0.0 or glide_from == f_target:
        return f_target
    t = _time(n, sr)
    u = np.clip(t / max(glide_s, 1e-3), 0.0, 1.0)
    return glide_from * (f_target / glide_from) ** u


def _finish(y: np.ndarray, velocity: float) -> np.ndarray:
    """Velocity scale, guarantee float32 / finite / peak <= 1."""
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = y * float(np.clip(velocity, 0.0, 1.0))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    return y.astype(_F32)


# ---------------------------------------------------------------------------
# voice renderers
# ---------------------------------------------------------------------------

def _v_808kick(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):
    f0 = float(freq_hz) if freq_hz else float(p["default_hz"])
    decay_s = p["decay_ms"] / 1000.0
    n = _n_samples(sr, max(dur_s, 4.0 * decay_s + 0.02))
    t = _time(n, sr)
    tau = p["tau_ms"] / 1000.0
    f = f0 + (p["sweep_mult"] * f0 - f0) * np.exp(-t / tau)
    body = np.sin(_phase_from_freq(f, n, sr))
    env = _attack_env(t, p["attack_ms"] / 1000.0) * np.exp(-t / decay_s)
    y = body * env
    # click: 3 ms noise burst, hp @ 2 kHz
    nc = _n_samples(sr, 0.003)
    click = np.zeros(n)
    burst = rng.standard_normal(min(nc, n))
    click[: burst.size] = burst
    click = svf_highpass(click, sr, 2000.0).astype(np.float64)
    y = y + p["click_level"] * velocity * click
    d = p["drive"]
    return np.tanh(y * d) / np.tanh(d)


def _v_808sub(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):
    f0 = float(freq_hz) if freq_hz else float(p["default_hz"])
    rel_s = p["release_ms"] / 1000.0
    n = _n_samples(sr, dur_s + 2.0 * rel_s)
    t = _time(n, sr)
    f = _glide_freq(f0, glide_from_hz, p["glide_ms"] / 1000.0, n, sr)
    y = np.sin(_phase_from_freq(f, n, sr))
    env = (_attack_env(t, p["attack_ms"] / 1000.0)
           * np.exp(-t / (p["decay_ms"] / 1000.0))
           * _gate_release(t, dur_s, rel_s))
    d = p["drive"]
    return np.tanh(y * env * d) / np.tanh(d)


def _v_hat(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz, open_=False):
    tau = (p["decay_ms"] / 1000.0) * (0.5 + 0.5 * velocity)   # velocity -> decay
    n = _n_samples(sr, max(dur_s, 5.0 * tau))
    t = _time(n, sr)
    y = np.zeros(n)
    for f in _HAT_FREQS * p["freq_scale"]:
        ph = _phase_from_freq(float(f), n, sr, phi0=float(rng.uniform(0.0, 2.0 * np.pi)))
        y += np.sign(np.sin(ph))
    y /= len(_HAT_FREQS)
    y = _svf_bandpass(y, sr, 10000.0, 1.0).astype(np.float64)
    y = svf_highpass(y, sr, 7000.0).astype(np.float64)
    y *= np.exp(-t / tau)
    peak = np.max(np.abs(y)) or 1.0
    return y / peak * p["level"]


def _v_openhat(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):
    return _v_hat(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz, open_=True)


def _v_clap(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):
    spacing = p["burst_spacing_ms"] / 1000.0
    tail_tau = p["tail_tau_ms"] / 1000.0
    tail_start = 3.0 * spacing
    n = _n_samples(sr, max(dur_s, tail_start + 5.0 * tail_tau))
    t = _time(n, sr)
    noise = rng.standard_normal(n)
    noise = _svf_bandpass(noise, sr, p["bp_hz"], p["bp_q"]).astype(np.float64)
    env = np.zeros(n)
    burst_tau = p["burst_tau_ms"] / 1000.0
    for k in range(3):
        tb = k * spacing
        seg = t - tb
        env = np.maximum(env, np.where(seg >= 0.0, np.exp(-seg / burst_tau), 0.0))
    seg = t - tail_start
    tail = np.where(seg >= 0.0, p["tail_level"] * np.exp(-seg / tail_tau), 0.0)
    env = np.maximum(env, tail)
    y = noise * env
    peak = np.max(np.abs(y)) or 1.0
    return y / peak * 0.9


def _v_snare(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):
    tone_tau = p["tone_decay_ms"] / 1000.0
    noise_tau = p["noise_decay_ms"] / 1000.0
    n = _n_samples(sr, max(dur_s, 5.0 * max(tone_tau, noise_tau)))
    t = _time(n, sr)
    tone = 0.5 * (np.sin(_phase_from_freq(p["tone_hz_1"], n, sr))
                  + np.sin(_phase_from_freq(p["tone_hz_2"], n, sr)))
    tone *= np.exp(-t / tone_tau)
    noise = rng.standard_normal(n)
    # 1-8 kHz band: centre = geometric mean, Q = centre / bandwidth
    noise = _svf_bandpass(noise, sr, 2828.0, 0.4).astype(np.float64)
    noise *= np.exp(-t / noise_tau)
    peak = np.max(np.abs(noise)) or 1.0
    noise = noise / peak
    mix = p["mix"]
    y = (1.0 - mix) * tone + mix * noise
    d = p["clip_drive"]
    return np.tanh(y * d) / np.tanh(d)


def _v_pad(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):
    f0 = float(freq_hz) if freq_hz else float(p["default_hz"])
    rel_s = p["release_ms"] / 1000.0
    n = _n_samples(sr, dur_s + 2.0 * rel_s)
    t = _time(n, sr)
    y = _supersaw(f0, n, sr, p["n_saws"], p["detune_cents"], rng)
    y = _svf(y, sr, float(p["cutoff_hz"]), float(p["q"]), "lp").astype(np.float64)
    env = _attack_env(t, p["attack_ms"] / 1000.0) * _gate_release(t, dur_s, rel_s)
    return y * env * 0.9


def _v_pluck(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):
    f0 = float(freq_hz) if freq_hz else float(p["default_hz"])
    decay_tau = p["decay_ms"] / 1000.0
    n = _n_samples(sr, max(dur_s, 4.0 * decay_tau))
    t = _time(n, sr)
    ph = _phase_from_freq(f0, n, sr, phi0=float(rng.uniform(0.0, 2.0 * np.pi)))
    y = _saw_polyblep(ph, f0, sr)
    # filter env: cutoff_start -> cutoff_end over filter_ms (exp approach)
    ftau = (p["filter_ms"] / 1000.0) / 3.0
    cutoff = p["cutoff_end_hz"] + (p["cutoff_start_hz"] - p["cutoff_end_hz"]) * np.exp(-t / ftau)
    y = _swept_lowpass(y, sr, cutoff, float(p["q"]))
    env = _attack_env(t, 0.002) * np.exp(-t / decay_tau)
    return y * env


def _v_epiano(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):
    f0 = float(freq_hz) if freq_hz else float(p["default_hz"])
    rel_s = p["release_ms"] / 1000.0
    n = _n_samples(sr, dur_s + 2.0 * rel_s)
    t = _time(n, sr)
    # velocity -> brightness (FM index)
    vscale = 0.25 + 0.75 * float(np.clip(velocity, 0.0, 1.0))
    itau = (p["index_ms"] / 1000.0) / 3.0
    index = p["index_end"] + (p["index_start"] - p["index_end"]) * np.exp(-t / itau)
    index *= vscale
    ph_c = _phase_from_freq(f0, n, sr)
    y = np.sin(ph_c + index * np.sin(ph_c))                       # 2-op FM, 1:1
    tine_tau = p["tine_tau_ms"] / 1000.0
    ph_t = _phase_from_freq(f0 * p["tine_ratio"], n, sr)
    tine = np.sin(ph_c + 1.5 * np.exp(-t / (tine_tau / 2.0)) * np.sin(ph_t))
    y = y + p["tine_level"] * tine * np.exp(-t / tine_tau)
    env = (_attack_env(t, 0.0015)
           * np.exp(-t / (p["decay_ms"] / 1000.0))
           * _gate_release(t, dur_s, rel_s))
    return y * env * 0.85


def _v_lead(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):
    f0 = float(freq_hz) if freq_hz else float(p["default_hz"])
    rel_s = p["release_ms"] / 1000.0
    n = _n_samples(sr, dur_s + 2.0 * rel_s)
    t = _time(n, sr)
    f = _glide_freq(f0, glide_from_hz, p["glide_ms"] / 1000.0, n, sr)
    y = _supersaw(f, n, sr, p["n_saws"], p["detune_cents"], rng)
    f_sub = np.asarray(f, dtype=np.float64) / 2.0
    sub = np.sign(np.sin(_phase_from_freq(f_sub if f_sub.ndim else float(f_sub), n, sr)))
    y = (y + p["sub_level"] * sub) / (1.0 + p["sub_level"])
    env = _attack_env(t, p["attack_ms"] / 1000.0) * _gate_release(t, dur_s, rel_s)
    return y * env * 0.9


def _v_crackle(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):
    n = _n_samples(sr, dur_s)
    mask = rng.random(n) < (p["rate_hz"] / sr)
    amps = rng.uniform(-1.0, 1.0, n)
    y = np.where(mask, amps * np.abs(amps), 0.0)                  # softer small pops
    y = _svf(y, sr, p["lp_hz"], 0.707, "lp").astype(np.float64)
    peak = np.max(np.abs(y)) or 1.0
    return y / peak * p["level"]


def _v_roundbass(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):
    f0 = float(freq_hz) if freq_hz else float(p["default_hz"])
    rel_s = p["release_ms"] / 1000.0
    n = _n_samples(sr, dur_s + 2.0 * rel_s)
    t = _time(n, sr)
    y = (np.sin(_phase_from_freq(f0, n, sr))
         + p["h2_level"] * np.sin(_phase_from_freq(2.0 * f0, n, sr))
         + p["h3_level"] * np.sin(_phase_from_freq(3.0 * f0, n, sr)))
    env = (_attack_env(t, p["attack_ms"] / 1000.0)
           * np.exp(-t / (p["decay_ms"] / 1000.0))
           * _gate_release(t, dur_s, rel_s))
    d = p["drive"]
    return np.tanh(y * env * d) / np.tanh(d * (1.0 + p["h2_level"] + p["h3_level"]))


def _v_housestab(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):
    f0 = float(freq_hz) if freq_hz else float(p["default_hz"])
    decay_tau = p["decay_ms"] / 1000.0
    n = _n_samples(sr, max(dur_s, 3.0 * decay_tau))
    t = _time(n, sr)
    y = _supersaw(f0, n, sr, p["n_saws"], p["detune_cents"], rng)
    ftau = (p["filter_ms"] / 1000.0) / 3.0
    cutoff = p["cutoff_end_hz"] + (p["cutoff_start_hz"] - p["cutoff_end_hz"]) * np.exp(-t / ftau)
    y = _swept_lowpass(y, sr, cutoff, float(p["q"]))
    return 0.8 * y * _attack_env(t, 0.001) * np.exp(-t / decay_tau)


def _v_kit(voice):
    def render(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):  # noqa: ARG001
        return samplekit.hit(voice, velocity, rng, sr)
    return render


def _v_tuned(voice, release_s):
    def render(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):  # noqa: ARG001
        midi = 69.0 + 12.0 * np.log2(float(freq_hz or 110.0) / 440.0)
        return samplekit.note(voice, midi, velocity, dur_s, release_s, rng, sr)
    return render


def _v_airpad(sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz):
    """Soft additive pad.  Its long release overlaps the next bar's chord, so a
    chord re-struck every bar reads as one sustained wash."""
    f0 = float(freq_hz) if freq_hz else 220.0
    attack_s, release_s = 0.35, 1.4
    n = _n_samples(sr, dur_s + 2.0 * release_s)
    t = _time(n, sr)
    y = np.zeros(n)
    for ratio, level in ((1.0, 1.0), (2.0, 0.28), (3.0, 0.1)):
        for cents in (-7.0, 7.0):
            f = f0 * ratio * 2.0 ** (cents / 1200.0)
            y += level * np.sin(_phase_from_freq(f, n, sr,
                                                 phi0=float(rng.uniform(0.0, 6.283))))
    drift = 1.0 + 0.12 * np.sin(2.0 * np.pi * 0.23 * t + float(rng.uniform(0.0, 6.283)))
    return 0.4 * y * drift * _attack_env(t, attack_s) * _gate_release(t, dur_s, release_s)


_TUNED_DESCRIPTIONS = {
    "fingerbass": "Recorded double bass, plucked very quietly.",
    "wurli": "Recorded Wurlitzer electric piano.",
    "vibes": "Recorded vibraphone, soft mallets.",
    "upright": "Recorded double bass, plucked.",
    "piano": "Recorded grand piano, close mics.",
    "airpad": "Soft additive pad that overlaps itself into a wash.",
}

_KIT_DESCRIPTIONS = {
    "kit_kick": "Recorded acoustic kick.",
    "kit_snare": "Recorded acoustic snare, centre hits; low velocity gives ghost notes.",
    "kit_rim": "Recorded snare cross-stick.",
    "kit_hat": "Recorded closed hi-hat.",
    "kit_halfhat": "Recorded half-open hi-hat.",
    "kit_openhat": "Recorded open hi-hat.",
    "kit_ride": "Recorded ride cymbal.",
    "kit_crash": "Recorded crash cymbal.",
    "kit_htom": "Recorded high tom.",
    "kit_ltom": "Recorded low tom.",
    "brush": "Recorded brush on a snare.",
    "cajon_bass": "Recorded cajon, the low tone from the middle of the face.",
    "cajon_slap": "Recorded cajon, the bright slap near the top edge.",
}

PRESETS.update({name: {} for name in (*_KIT_DESCRIPTIONS, *_TUNED_DESCRIPTIONS)})
_DESCRIPTIONS.update(_KIT_DESCRIPTIONS)
_DESCRIPTIONS.update(_TUNED_DESCRIPTIONS)

# The curated synths and pads (plume.synths) render in stereo.
PRESETS.update({name: dict(d) for name, d in synths.PRESETS.items()})
_DESCRIPTIONS.update(synths.DESCRIPTIONS)
for _k, _r in synths.PARAM_RANGES.items():
    _PARAM_RANGES.setdefault(_k, _r)

_RENDERERS = {
    **{name: _v_kit(name) for name in _KIT_DESCRIPTIONS},
    "fingerbass": _v_tuned("fingerbass", 0.08),
    "wurli": _v_tuned("wurli", 0.5),
    "vibes": _v_tuned("vibes", 1.5),
    "upright": _v_tuned("upright", 0.07),
    "piano": _v_tuned("piano", 0.2),
    "airpad": _v_airpad,
    "808kick": _v_808kick,
    "808sub": _v_808sub,
    "hat": _v_hat,
    "openhat": _v_openhat,
    "clap": _v_clap,
    "snare": _v_snare,
    "pad": _v_pad,
    "pluck": _v_pluck,
    "epiano": _v_epiano,
    "lead": _v_lead,
    "crackle": _v_crackle,
    "housekick": _v_808kick,
    "boomkick": _v_808kick,
    "offhat": _v_openhat,
    "roundbass": _v_roundbass,
    "housestab": _v_housestab,
}


def is_stereo(preset: str) -> bool:
    """True for voices that render their own stereo image."""
    return preset in synths.RENDERERS


def _params(preset: str, params: dict | None) -> dict:
    p = dict(PRESETS[preset])
    for key, val in (params or {}).items():
        if key in p:
            p[key] = type(p[key])(val) if not isinstance(val, str) else val
        else:
            p[key] = val
    return p


def render_note_stereo(preset: str, *, sr: int, dur_s: float,
                       freq_hz: float | None, velocity: float,
                       rng: np.random.Generator, params: dict,
                       glide_from_hz: float | None = None) -> np.ndarray:
    """Like :func:`render_note`, but ``(n, 2)`` float32.

    Stereo voices keep their own image; every other voice comes back as the
    same mono signal on both sides.  Folding the result to mono, ``mean`` of
    the two sides, gives exactly what ``render_note`` returns.
    """
    if preset not in synths.RENDERERS:
        y = render_note(preset, sr=sr, dur_s=dur_s, freq_hz=freq_hz,
                        velocity=velocity, rng=rng, params=params,
                        glide_from_hz=glide_from_hz)
        return np.repeat(y[:, None], 2, axis=1)
    p = _params(preset, params)
    dur_s = max(float(dur_s), 1.0 / sr)
    st = synths.render(preset, sr, dur_s, freq_hz, float(velocity), rng, p,
                       glide_from_hz)
    # The same guarantees as _finish (finite, velocity scaled, peak <= 1) in
    # one scaling pass: these buffers are long and stereo, and the live
    # planner renders several every bar.
    peak = max(float(st.max()), -float(st.min())) if st.size else 0.0
    if not np.isfinite(peak):
        st = np.nan_to_num(st, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(st))) if st.size else 0.0
    v = float(np.clip(velocity, 0.0, 1.0))
    scale = 1.0 / peak if v * peak > 1.0 else v
    out = np.empty(st.shape, dtype=_F32)
    np.multiply(st, scale, out=out, casting="same_kind")
    return out


def render_note(preset: str, *, sr: int, dur_s: float, freq_hz: float | None,
                velocity: float, rng: np.random.Generator,
                params: dict, glide_from_hz: float | None = None) -> np.ndarray:
    """Render one note/hit of ``preset`` to a mono float32 buffer.

    ``params`` overrides the preset defaults (see PRESETS). Determinism
    comes entirely from the passed-in ``rng``.
    """
    if preset in synths.RENDERERS:
        st = render_note_stereo(preset, sr=sr, dur_s=dur_s, freq_hz=freq_hz,
                                velocity=velocity, rng=rng, params=params,
                                glide_from_hz=glide_from_hz)
        return st.mean(axis=1, dtype=np.float64).astype(_F32)
    if preset not in _RENDERERS:
        raise ValueError(f"unknown preset: {preset!r} "
                         f"(have: {sorted([*_RENDERERS, *synths.RENDERERS])})")
    p = _params(preset, params)
    dur_s = max(float(dur_s), 1.0 / sr)
    y = _RENDERERS[preset](sr, dur_s, freq_hz, float(velocity), rng, p, glide_from_hz)
    return _finish(y, velocity)
