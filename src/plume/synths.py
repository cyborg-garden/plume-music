"""plume.synths — the curated synth and pad voices.

Recorded samples carry plume's acoustic sounds.  Synths and pads are the one
place where code-made sound is the real instrument, so each voice here copies
how the classic machine builds it rather than a textbook waveform:

==============  ============================================================
``warmpad``     JP-8000 style supersaw: seven band-limited saws on an uneven
                detune curve, a slow filter swell and a BBD-style chorus.
``strings``     String-ensemble pad: two saw registers through a three-phase
                ensemble chorus, the Solina recipe.
``breathpad``   Airy pad: soft sine partials with pitched, breathing noise.
``glasskeys``   DX-style electric piano: two FM pairs, a 1:1 body and a 14:1
                tine, with key scaling so high notes stay clean.
``junopoly``    Juno-style poly: one saw, a slowly swept pulse, a sub square,
                a four-pole filter and chorus I.
``synthlead``   Synthwave lead: two detuned saws, a resonant four-pole filter,
                delayed vibrato and glide.
``synthbass``   Synthwave bass: saw plus square through a snappy four-pole
                filter envelope, kept mono and centred.
``tapekeys``    Lofi keys: a mellow tine piano with tape wow and flutter.
``organ``       House organ: M1-style drawbar partials with a percussive
                upper register and key click; short notes make stabs.
``bell``        Trap bell: a 1:1 FM bell over a sine body.  Kept harmonic: an
                inharmonic ratio pulls the key tracker toward the fifth.
==============  ============================================================

Every renderer returns a stereo ``(n, 2)`` float64 buffer; plume.voices folds
it to mono for callers that want mono.  Oscillators are polyBLEP band-limited
or additive below Nyquist, the filter is a zero-delay-feedback (TPT) four-pole
ladder solved in closed form, and nothing loops per sample in Python: the only
loop is per filter block.  Each voice drifts slowly so a held chord moves.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

_TWO_PI = 2.0 * np.pi

# Adam Szabo's measured JP-8000 supersaw detune offsets (relative to the max
# detune), centre saw first.  "How to Emulate the Super Saw", 2010.
_SZABO = np.array([0.0, -0.11002313, 0.10745242, -0.06288439, 0.06216538,
                   -0.01952356, 0.01991221])


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

def _n(sr: int, seconds: float) -> int:
    return max(1, int(round(seconds * sr)))


_LN2_1200 = np.log(2.0) / 1200.0


def _cents(c):
    return np.exp(np.asarray(c, dtype=np.float64) * _LN2_1200)


def _drift(n: int, sr: int, rng: np.random.Generator, rate_hz: float) -> np.ndarray:
    """A smooth random curve in [-1, 1], changing about ``rate_hz`` times a
    second: analog oscillator drift, cheaply (a few control points, cosine
    interpolated)."""
    k = int(n / sr * rate_hz) + 3
    pts = rng.uniform(-1.0, 1.0, k)
    x = np.arange(n, dtype=np.float64) * (rate_hz / sr)
    i = np.minimum(x.astype(np.int64), k - 2)
    f = x - i
    w = 0.5 - 0.5 * np.cos(np.pi * f)
    return pts[i] * (1.0 - w) + pts[i + 1] * w


def _phase(freq, n: int, sr: int, phi0: float) -> np.ndarray:
    """Phase in cycles for a constant or per-sample frequency."""
    f = np.asarray(freq, dtype=np.float64)
    if f.ndim == 0:
        return phi0 + float(f) / sr * np.arange(n, dtype=np.float64)
    ph = np.empty(n)
    ph[0] = phi0
    np.cumsum(f[:-1] / sr, out=ph[1:])
    ph[1:] += phi0
    return ph


def _blep(p: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """polyBLEP residual for a unit step at phase 0 (phase ``p`` in cycles)."""
    out = np.zeros_like(p)
    m = p < dt
    if m.any():
        v = p[m] / dt[m]
        out[m] = v + v - v * v - 1.0
    m = p > 1.0 - dt
    if m.any():
        v = (p[m] - 1.0) / dt[m]
        out[m] = v * v + v + v + 1.0
    return out


def saw(phase_cycles: np.ndarray, freq, sr: int) -> np.ndarray:
    """Band-limited (polyBLEP) sawtooth, -1..1."""
    p = np.mod(phase_cycles, 1.0)
    f = np.asarray(freq, dtype=np.float64)
    if f.ndim == 0:
        dt = float(np.clip(f / sr, 1e-9, 0.49))
        out = 2.0 * p - 1.0
        m = p < dt
        v = p[m] / dt
        out[m] -= v + v - v * v - 1.0
        m = p > 1.0 - dt
        v = (p[m] - 1.0) / dt
        out[m] -= v * v + v + v + 1.0
        return out
    dt = np.clip(np.broadcast_to(f / sr, p.shape), 1e-9, 0.49)
    return 2.0 * p - 1.0 - _blep(p, dt)


def pulse(phase_cycles: np.ndarray, freq, sr: int, width) -> np.ndarray:
    """Band-limited pulse: the difference of two BLEP saws (zero mean)."""
    w = np.clip(np.asarray(width, dtype=np.float64), 0.05, 0.95)
    return saw(phase_cycles, freq, sr) - saw(phase_cycles + w, freq, sr)


def ladder(x: np.ndarray, sr: int, cutoff, res: float, block: int = 64,
           drive: float = 0.0) -> np.ndarray:
    """Four-pole ZDF ladder lowpass, ``res`` 0..1 (1 is near self-oscillation).

    The linear TPT ladder has a closed-form transfer function: four bilinear
    one-poles ``G`` in a loop with feedback ``k`` give ``G^4 / (1 + k G^4)``,
    which is exactly what the zero-delay-feedback solve computes.  Its
    coefficients are evaluated per block, vectorized, and run through lfilter
    with state carried across blocks.  ``drive`` adds tanh saturation at the
    input, where a transistor ladder clips.  Passband loss from feedback is
    half made up, the way most hardware does it.
    """
    x = np.asarray(x, dtype=np.float64)
    if drive > 0.0:
        a = 1.0 + 3.0 * drive
        x = np.tanh(x * a) / np.tanh(a)
    n = x.shape[0]
    k = 3.8 * float(np.clip(res, 0.0, 0.98))
    cut = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), (n,))
    starts = np.arange(0, n, block)
    mids = np.minimum(starts + block // 2, n - 1)
    fc = np.clip(cut[mids], 15.0, 0.42 * sr)
    g = np.tan(np.pi * fc / sr)
    c = (g - 1.0) / (g + 1.0)                    # one-pole pole term
    gn = (g / (1.0 + g)) ** 4                    # one-pole zero gain, ^4
    binom = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    den = np.stack([np.ones_like(c), 4 * c, 6 * c * c, 4 * c ** 3, c ** 4], axis=1)
    num = gn[:, None] * binom[None, :]
    den = den + k * num
    num = num * (1.0 + 0.5 * k)
    if np.ptp(fc) < 1e-6:                        # static cutoff: one call
        return lfilter(num[0], den[0], x)
    out = np.empty(n)
    zi = np.zeros(4)
    for bi, i in enumerate(starts):
        j = min(i + block, n)
        out[i:j], zi = lfilter(num[bi], den[bi], x[i:j], zi=zi)
    return out


def onepole_lp(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """TPT (bilinear, prewarped) one-pole lowpass."""
    g = np.tan(np.pi * min(float(cutoff_hz), 0.45 * sr) / sr)
    return lfilter([g / (1.0 + g), g / (1.0 + g)], [1.0, (g - 1.0) / (1.0 + g)],
                   x, axis=0)


def highpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Two-pole Butterworth-ish highpass (TPT SVF)."""
    g = np.tan(np.pi * min(float(cutoff_hz), 0.45 * sr) / sr)
    k = 1.41421356
    a0 = 1.0 + g * k + g * g
    b = np.array([1.0, -2.0, 1.0]) / a0
    a = np.array([1.0, 2.0 * (g * g - 1.0) / a0, (1.0 - g * k + g * g) / a0])
    return lfilter(b, a, x, axis=0)


def _bp_coeffs(sr: int, centre_hz: float, q: float):
    g = np.tan(np.pi * min(float(centre_hz), 0.45 * sr) / sr)
    k = 1.0 / q
    a0 = 1.0 + g * k + g * g
    b = np.array([g * k, 0.0, -g * k]) / a0
    a = np.array([1.0, 2.0 * (g * g - 1.0) / a0, (1.0 - g * k + g * g) / a0])
    return b, a


def bandpass(x: np.ndarray, sr: int, centre_hz: float, q: float) -> np.ndarray:
    return lfilter(*_bp_coeffs(sr, centre_hz, q), x, axis=0)


def _tone_coeffs(sr: int, lp_hz: float, dip_hz: float, dip_db: float):
    """One-pole TPT lowpass and an RBJ peaking dip, as one 3rd-order filter."""
    g = np.tan(np.pi * min(float(lp_hz), 0.45 * sr) / sr)
    bl = np.array([g, g]) / (1.0 + g)
    al = np.array([1.0, (g - 1.0) / (1.0 + g)])
    bp, ap = _peak_coeffs(sr, dip_hz, dip_db, 0.8)
    return np.polymul(bl, bp), np.polymul(al, ap)


def _delay_tap(x: np.ndarray, sr: int, delay_s: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    pos = np.arange(n, dtype=np.float64) - delay_s * sr
    pos = np.clip(pos, 0.0, n - 1.0)
    i0 = pos.astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    fr = pos - i0
    return x[i0] * (1.0 - fr) + x[i1] * fr


def juno_chorus(x: np.ndarray, sr: int, rate_hz: float, depth_ms: float,
                phase: float, mix: float = 0.5) -> np.ndarray:
    """One BBD line read with opposite modulation on each side, the Juno trick
    that makes width without moving the centre.  Mono in, stereo out."""
    n = x.shape[0]
    t = np.arange(n, dtype=np.float64) / sr
    lfo = 2.0 / np.pi * np.arcsin(np.sin(_TWO_PI * rate_hz * t + phase))  # triangle
    base = 3.5e-3
    d = depth_ms * 1e-3
    left = _delay_tap(x, sr, base + d * lfo)
    right = _delay_tap(x, sr, base - d * lfo)
    dry = (1.0 - mix)
    return np.stack([dry * x + mix * left, dry * x + mix * right], axis=1)


def ensemble(x: np.ndarray, sr: int, phase: float) -> np.ndarray:
    """Three delay lines on three-phase slow and fast LFOs (string ensemble).
    Mono in, stereo out."""
    n = x.shape[0]
    t = np.arange(n, dtype=np.float64) / sr
    taps = []
    for k in range(3):
        ph = phase + k * _TWO_PI / 3.0
        d = (5.0e-3 + 1.6e-3 * np.sin(_TWO_PI * 0.63 * t + ph)
             + 0.25e-3 * np.sin(_TWO_PI * 6.1 * t + ph * 1.7))
        taps.append(_delay_tap(x, sr, d))
    left = 0.4 * x + 0.45 * taps[0] + 0.3 * taps[1]
    right = 0.4 * x + 0.45 * taps[2] + 0.3 * taps[1]
    return np.stack([left, right], axis=1)


def adsr(n: int, sr: int, dur_s: float, a: float, d: float, s: float,
         r: float) -> np.ndarray:
    """Exponential-segment ADSR: attack to 1 in ``a`` s, decay toward ``s``
    with time constant ``d``, release with time constant ``r/3`` after the
    gate.  The release starts from wherever the envelope is, so a short note
    never clicks."""
    t = np.arange(n, dtype=np.float64) / sr
    att = 1.0 - np.exp(-t / max(a / 3.0, 1e-4))
    att = att / (1.0 - np.exp(-3.0))
    rise = np.clip(att, 0.0, 1.0)
    held = np.where(t < a, rise,
                    s + (1.0 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    gate = min(int(round(dur_s * sr)), n)
    env = held.copy()
    if gate < n:
        lvl = held[gate - 1] if gate > 0 else 0.0
        env[gate:] = lvl * np.exp(-(t[gate:] - dur_s) / max(r / 3.0, 1e-4))
    return env


def width(st: np.ndarray, amount: float) -> np.ndarray:
    """Scale the side signal: 0 is mono, 1 leaves the image as rendered."""
    mid = 0.5 * (st[:, 0] + st[:, 1])
    side = 0.5 * (st[:, 0] - st[:, 1]) * float(amount)
    return np.stack([mid + side, mid - side], axis=1)


def peak_eq(x: np.ndarray, sr: int, f_hz: float, gain_db: float,
            q: float = 0.8) -> np.ndarray:
    """RBJ peaking EQ, used as a gentle low-mid dip so chords stay clear of
    the bass and the voice in the 200-500 Hz pile-up."""
    return lfilter(*_peak_coeffs(sr, f_hz, gain_db, q), x, axis=0)


def _peak_coeffs(sr: int, f_hz: float, gain_db: float, q: float):
    A = 10.0 ** (gain_db / 40.0)
    w = _TWO_PI * f_hz / sr
    al = np.sin(w) / (2.0 * q)
    b = np.array([1 + al * A, -2 * np.cos(w), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w), 1 - al / A])
    return b / a[0], a / a[0]


def _f0(freq_hz, default: float) -> float:
    return float(freq_hz) if freq_hz else float(default)


def _key_scale(f0: float, knee_hz: float) -> float:
    """1 below ``knee_hz``, falling 6 dB per octave above: FM index and
    brightness ease off up the keyboard, like a DX7 level-scaling curve."""
    return float(min(1.0, knee_hz / max(f0, 1.0)))


def _glide(f0: float, glide_from: float | None, glide_s: float, n: int,
           sr: int):
    if not glide_from or glide_from <= 0.0 or glide_from == f0:
        return np.full(n, f0)
    t = np.arange(n, dtype=np.float64) / sr
    u = np.clip(t / max(glide_s, 1e-3), 0.0, 1.0)
    return glide_from * (f0 / glide_from) ** u


def _sine_stack(f, n, sr, partials, rng, phase_jitter=True):
    """Additive sines, each partial dropped once it would pass Nyquist."""
    f = np.asarray(f, dtype=np.float64)
    fmax = float(np.max(f)) if f.ndim else float(f)
    y = np.zeros(n)
    base = _phase(f, n, sr, 0.0)
    for ratio, level in partials:
        if ratio * fmax >= 0.45 * sr:
            continue
        ph0 = float(rng.uniform(0.0, 1.0)) if phase_jitter else 0.0
        y += level * np.sin(_TWO_PI * (ratio * base + ph0))
    return y


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "warmpad": {"detune": 0.35, "cutoff_hz": 1400.0, "env_amount": 1.6,
                "res": 0.15, "attack_ms": 450.0, "release_ms": 1200.0,
                "chorus": 0.5, "default_hz": 220.0},
    "strings": {"cutoff_hz": 3200.0, "octave_level": 0.35, "attack_ms": 380.0,
                "release_ms": 1200.0, "default_hz": 220.0},
    "breathpad": {"breath": 0.35, "cutoff_hz": 2600.0, "attack_ms": 600.0,
                  "release_ms": 1800.0, "default_hz": 220.0},
    "glasskeys": {"index": 2.2, "tine_level": 0.18, "decay_ms": 1400.0,
                  "release_ms": 220.0, "tremolo": 0.12, "default_hz": 440.0},
    "junopoly": {"cutoff_hz": 900.0, "env_amount": 2.2, "res": 0.25,
                 "sub_level": 0.35, "decay_ms": 420.0, "sustain": 0.6,
                 "release_ms": 380.0, "chorus": 0.6, "default_hz": 330.0},
    "synthlead": {"detune_cents": 9.0, "cutoff_hz": 1800.0, "env_amount": 1.8,
                  "res": 0.35, "vibrato_cents": 18.0, "glide_ms": 70.0,
                  "release_ms": 160.0, "default_hz": 440.0},
    "synthbass": {"cutoff_hz": 260.0, "env_amount": 4.0, "res": 0.3,
                  "decay_ms": 160.0, "release_ms": 50.0, "drive": 0.25,
                  "glide_ms": 40.0, "default_hz": 55.0},
    "tapekeys": {"wow_cents": 11.0, "flutter_cents": 3.0, "cutoff_hz": 3000.0,
                 "decay_ms": 1700.0, "release_ms": 300.0, "default_hz": 440.0},
    "organ": {"percussion": 0.5, "click": 0.3, "decay_ms": 3000.0,
              "release_ms": 60.0, "default_hz": 220.0},
    "bell": {"ratio": 1.0, "index": 2.4, "decay_ms": 450.0, "body": 0.6,
             "release_ms": 200.0, "default_hz": 523.25},
}

DESCRIPTIONS = {
    "warmpad": "Warm analog pad: 7-saw supersaw, slow filter swell, Juno chorus, wide.",
    "strings": "String-ensemble pad: two saw registers through a three-phase ensemble chorus.",
    "breathpad": "Airy pad: soft sine partials with pitched breathing noise, wide.",
    "glasskeys": "Glassy FM electric piano: 1:1 body and 14:1 tine pairs; velocity brightens.",
    "junopoly": "Juno-style poly: saw + swept pulse + sub through a 4-pole filter, chorus I.",
    "synthlead": "Synthwave lead: two detuned saws, resonant 4-pole filter, delayed vibrato, glide.",
    "synthbass": "Synthwave bass: saw + square, snappy 4-pole filter envelope, mono.",
    "tapekeys": "Lofi keys: mellow tine piano with tape wow and flutter.",
    "organ": "House organ: M1-style drawbars with percussive top and key click; short notes stab.",
    "bell": "Trap bell: harmonic 1:1 FM bell over a sine body, short ring.",
}

PARAM_RANGES = {
    "detune": [0.0, 1.0], "cutoff_hz": [100.0, 12000.0], "env_amount": [0.0, 6.0],
    "res": [0.0, 0.95], "attack_ms": [1.0, 3000.0], "release_ms": [5.0, 4000.0],
    "chorus": [0.0, 1.0], "default_hz": [20.0, 2000.0], "octave_level": [0.0, 1.0],
    "breath": [0.0, 1.0], "index": [0.0, 6.0], "tine_level": [0.0, 0.6],
    "decay_ms": [20.0, 6000.0], "tremolo": [0.0, 0.5], "sub_level": [0.0, 1.0],
    "sustain": [0.0, 1.0], "detune_cents": [0.0, 40.0], "vibrato_cents": [0.0, 60.0],
    "glide_ms": [5.0, 400.0], "drive": [0.0, 1.0], "wow_cents": [0.0, 40.0],
    "flutter_cents": [0.0, 15.0], "percussion": [0.0, 1.0], "click": [0.0, 1.0],
    "ratio": [0.5, 8.0], "body": [0.0, 1.0],
}

# Output trims so each voice, in its role pattern, sits at a K-weighted loudness
# matched to the voice it replaces in its role (measured offline; see the
# plan notes); glasskeys and synthbass sit ~3 dB under so a full-velocity
# note does not hit the per-note peak limit.  Applied before velocity.
_TRIM = {
    "warmpad": 1.617, "strings": 1.581, "breathpad": 2.854, "tapekeys": 1.518,
    "glasskeys": 2.6, "junopoly": 2.432, "organ": 1.437, "bell": 1.123,
    "synthlead": 1.204, "synthbass": 1.3,
}


# ---------------------------------------------------------------------------
# renderers: (sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz) -> (n, 2)
# ---------------------------------------------------------------------------

def _warmpad(sr, dur_s, freq_hz, vel, rng, p, glide_from_hz):
    f0 = _f0(freq_hz, p["default_hz"])
    rel = p["release_ms"] / 1000.0
    n = _n(sr, dur_s + rel)
    t = np.arange(n, dtype=np.float64) / sr
    # Szabo: detune grows nonlinearly with the knob; ~0.35 is the classic spot
    amt = float(np.clip(p["detune"], 0.0, 1.0))
    spread = 12.0 * amt ** 1.5 + 0.6                       # semitone fraction
    drift = _drift(n, sr, rng, 0.4)
    # Two drift directions, one per side; each saw's phase is its detune
    # ratio times the shared running phase, so only two cumsums are needed.
    base = {+1: _phase(f0 * np.exp(drift * (3.0 * np.log(2.0) / 1200.0)), n, sr, 0.0),
            -1: _phase(f0 * np.exp(drift * (-3.0 * np.log(2.0) / 1200.0)), n, sr, 0.0)}
    sides = (np.zeros(n), np.zeros(n))
    for i, off in enumerate(_SZABO):
        ratio = 2.0 ** (off * spread / 12.0)
        y = saw(ratio * base[1 if i % 2 else -1] + float(rng.uniform()),
                ratio * f0, sr)
        if i == 0:
            sides[0][:] += 0.5 * y
            sides[1][:] += 0.5 * y
        else:
            sides[i % 2][:] += 0.62 * y
    out = []
    # filter: slow swell up after the attack, a slow LFO on top; velocity opens it
    lfo = np.sin(_TWO_PI * 0.13 * t + float(rng.uniform(0.0, _TWO_PI)))
    swell = 1.0 - np.exp(-t / max(p["attack_ms"] / 1000.0 * 1.6, 0.01))
    v = 0.55 + 0.45 * float(np.clip(vel, 0.0, 1.0))
    cut = p["cutoff_hz"] * v * (0.7 + 0.3 * swell * p["env_amount"]) * (1.0 + 0.12 * lfo)
    cut = np.minimum(cut, max(f0 * 9.0, 1400.0))           # keep the top sweet
    for ch in sides:
        y = highpass(ch, sr, f0 * 0.7)                     # Szabo: HPF at f0
        out.append(ladder(y, sr, cut, p["res"], block=512))
    st = peak_eq(np.stack(out, axis=1), sr, 300.0, -2.5) * 0.42
    env = adsr(n, sr, dur_s, p["attack_ms"] / 1000.0, 1.0, 1.0, rel)
    st *= env[:, None]
    ch = p["chorus"]
    if ch > 0.0:
        mono = st.mean(axis=1)
        wet = juno_chorus(mono, sr, 0.5, 1.8, float(rng.uniform(0.0, _TWO_PI)), 0.5)
        st = (1.0 - 0.5 * ch) * st + 0.5 * ch * wet
    return st


def _strings(sr, dur_s, freq_hz, vel, rng, p, glide_from_hz):
    f0 = _f0(freq_hz, p["default_hz"])
    rel = p["release_ms"] / 1000.0
    n = _n(sr, dur_s + rel)
    drift = _drift(n, sr, rng, 0.3)
    f = f0 * _cents(2.0 * drift)
    y = saw(_phase(f, n, sr, float(rng.uniform())), f, sr)
    f8 = 2.0 * f * _cents(1.5)
    y = y + p["octave_level"] * saw(_phase(f8, n, sr, float(rng.uniform())), f8, sr)
    y = highpass(y, sr, max(140.0, f0 * 0.8))
    # gentle two-pole-ish top roll (ladder with no resonance, static)
    v = 0.6 + 0.4 * float(np.clip(vel, 0.0, 1.0))
    y = ladder(y, sr, min(p["cutoff_hz"] * v, 9000.0), 0.0)
    # a soft body bump around 1 kHz, like the cabinet filter
    y = y + 0.35 * bandpass(y, sr, 1100.0, 1.2)
    env = adsr(n, sr, dur_s, p["attack_ms"] / 1000.0, 1.0, 1.0, rel)
    y = peak_eq(y, sr, 300.0, -2.5)
    st = ensemble(y * env * 0.5, sr, float(rng.uniform(0.0, _TWO_PI)))
    return st


def _breathpad(sr, dur_s, freq_hz, vel, rng, p, glide_from_hz):
    f0 = _f0(freq_hz, p["default_hz"])
    rel = p["release_ms"] / 1000.0
    n = _n(sr, dur_s + rel)
    t = np.arange(n, dtype=np.float64) / sr
    partials = ((1.0, 1.0), (2.0, 0.32), (3.0, 0.12), (4.0, 0.06))
    sides = []
    for c in (-6.0, 6.0):
        drift = _drift(n, sr, rng, 0.25)
        f = f0 * _cents(c + 3.0 * drift)
        sides.append(_sine_stack(f, n, sr, partials, rng))
    # breath: noise through resonant bands on the 2nd and 4th harmonics,
    # swelling on a slow LFO.  The two bands are one 4th-order filter and the
    # tone filter and low-mid dip one 3rd-order filter, to keep it cheap.
    breathe = 0.65 + 0.35 * np.sin(_TWO_PI * 0.17 * t + float(rng.uniform(0.0, _TWO_PI)))
    amt = float(np.clip(p["breath"], 0.0, 1.0))
    b1, a1 = _bp_coeffs(sr, min(2.0 * f0, 9000.0), 9.0)
    b2, a2 = _bp_coeffs(sr, min(4.0 * f0, 12000.0), 7.0)
    bb = 0.9 * np.polymul(b1, a2) + 0.5 * np.polymul(b2, a1)
    ab = np.polymul(a1, a2)
    nz = lfilter(bb, ab, rng.standard_normal((n, 2)), axis=0)
    y = 0.55 * np.stack(sides, axis=1) + amt * nz * breathe[:, None]
    bt, at = _tone_coeffs(sr, p["cutoff_hz"], 300.0, -3.0)
    st = width(lfilter(bt, at, y, axis=0), 0.55)
    env = adsr(n, sr, dur_s, p["attack_ms"] / 1000.0, 1.0, 1.0, rel)
    return st * env[:, None] * 0.55


def _glasskeys(sr, dur_s, freq_hz, vel, rng, p, glide_from_hz):
    f0 = _f0(freq_hz, p["default_hz"])
    rel = p["release_ms"] / 1000.0
    dec = p["decay_ms"] / 1000.0 * _key_scale(f0, 260.0) ** 0.5
    n = _n(sr, dur_s + rel)
    t = np.arange(n, dtype=np.float64) / sr
    v = float(np.clip(vel, 0.0, 1.0))
    ks = _key_scale(f0, 520.0)
    out = []
    phi = float(rng.uniform())
    for side, cents in ((0, -2.5), (1, 2.5)):
        f = f0 * _cents(cents)
        ph = _TWO_PI * _phase(f, n, sr, phi)
        idx = p["index"] * (0.3 + 0.7 * v) * ks * (0.25 + 0.75 * np.exp(-t / 0.35))
        body = np.sin(ph + idx * np.sin(ph))
        # tine: 14:1 modulator rings briefly; dropped once its band would alias
        tine = np.zeros(n)
        if 15.0 * f < 0.45 * sr:
            ph_t = _TWO_PI * _phase(14.0 * f, n, sr, 0.0)
            tine = np.sin(ph + 1.2 * ks * np.exp(-t / 0.03) * np.sin(ph_t))
            tine = (tine - np.sin(ph)) * np.exp(-t / 0.06)
        y = body + p["tine_level"] * (0.4 + 0.6 * v) * tine
        out.append(y)
    st = np.stack(out, axis=1)
    # two-stage decay: a quick bloom, then a long ring
    amp = 0.35 * np.exp(-t / 0.18) + 0.65 * np.exp(-t / max(dec, 0.05))
    trem = 1.0 + p["tremolo"] * np.sin(_TWO_PI * 4.8 * t + float(rng.uniform(0, _TWO_PI)))
    env = amp * adsr(n, sr, dur_s, 0.002, 1.0, 1.0, rel)
    st[:, 0] *= env * trem
    st[:, 1] *= env * (2.0 - trem)                 # auto-pan tremolo, Rhodes-style
    st = highpass(peak_eq(st, sr, 320.0, -2.5), sr, 25.0)   # 1:1 FM makes DC
    return width(st, 0.6) * 0.42


def _junopoly(sr, dur_s, freq_hz, vel, rng, p, glide_from_hz):
    f0 = _f0(freq_hz, p["default_hz"])
    rel = p["release_ms"] / 1000.0
    n = _n(sr, dur_s + rel)
    t = np.arange(n, dtype=np.float64) / sr
    drift = _drift(n, sr, rng, 0.5)
    f = f0 * _cents(2.5 * drift)
    ph = _phase(f, n, sr, float(rng.uniform()))
    width = 0.5 + 0.35 * np.sin(_TWO_PI * 0.6 * t + float(rng.uniform(0.0, _TWO_PI)))
    y = 0.55 * saw(ph, f, sr) + 0.45 * pulse(ph, f, sr, width)
    sub = pulse(ph * 0.5, f * 0.5, sr, 0.5)
    y = y + p["sub_level"] * sub
    v = float(np.clip(vel, 0.0, 1.0))
    fenv = adsr(n, sr, dur_s, 0.004, p["decay_ms"] / 1000.0, 0.25, rel)
    cut = p["cutoff_hz"] * (1.0 + p["env_amount"] * (0.4 + 0.6 * v) * fenv)
    cut = cut * float(np.clip(f0 / 220.0, 0.5, 2.0)) ** 0.5    # key tracking
    cut = np.minimum(cut, 7000.0)
    y = ladder(y, sr, cut, p["res"], block=64)
    y = peak_eq(y, sr, 280.0, -3.0)
    env = adsr(n, sr, dur_s, 0.006, p["decay_ms"] / 1000.0, p["sustain"], rel)
    y = y * env * 0.42
    st = juno_chorus(y, sr, 0.5, 1.9 * p["chorus"] + 0.2,
                     float(rng.uniform(0.0, _TWO_PI)), 0.5)
    return st


def _synthlead(sr, dur_s, freq_hz, vel, rng, p, glide_from_hz):
    f0 = _f0(freq_hz, p["default_hz"])
    rel = p["release_ms"] / 1000.0
    n = _n(sr, dur_s + rel)
    t = np.arange(n, dtype=np.float64) / sr
    fg = _glide(f0, glide_from_hz, p["glide_ms"] / 1000.0, n, sr)
    # delayed vibrato: fades in after a quarter second, like a player's
    vib_amt = p["vibrato_cents"] * np.clip((t - 0.25) / 0.4, 0.0, 1.0)
    vib = vib_amt * np.sin(_TWO_PI * 5.6 * t + float(rng.uniform(0.0, _TWO_PI)))
    drift = _drift(n, sr, rng, 0.6) * 2.0
    sides = []
    for s in (-1.0, 1.0):
        f = fg * _cents(s * p["detune_cents"] * 0.5 + vib + drift)
        sides.append(saw(_phase(f, n, sr, float(rng.uniform())), f, sr))
    v = float(np.clip(vel, 0.0, 1.0))
    fenv = adsr(n, sr, dur_s, 0.01, 0.35, 0.45, rel)
    cut = p["cutoff_hz"] * (1.0 + p["env_amount"] * (0.5 + 0.5 * v) * fenv)
    cut = np.minimum(cut, 9000.0)
    mid = 0.5 * (sides[0] + sides[1])
    side = 0.5 * (sides[0] - sides[1])
    mid = ladder(mid, sr, cut, p["res"], block=64, drive=0.15)
    side = ladder(side, sr, cut, p["res"], block=64)
    env = adsr(n, sr, dur_s, 0.008, 0.4, 0.85, rel)
    st = np.stack([mid + 0.7 * side, mid - 0.7 * side], axis=1)
    return st * env[:, None] * 0.55


def _synthbass(sr, dur_s, freq_hz, vel, rng, p, glide_from_hz):
    f0 = _f0(freq_hz, p["default_hz"])
    rel = p["release_ms"] / 1000.0
    n = _n(sr, dur_s + rel)
    f = _glide(f0, glide_from_hz, p["glide_ms"] / 1000.0, n, sr)
    ph = _phase(f, n, sr, float(rng.uniform()))
    y = 0.6 * saw(ph, f, sr) + 0.4 * pulse(ph, f, sr, 0.5)
    y = y + 0.5 * np.sin(_TWO_PI * ph)                     # solid fundamental
    v = float(np.clip(vel, 0.0, 1.0))
    fenv = adsr(n, sr, dur_s, 0.002, p["decay_ms"] / 1000.0, 0.0, rel)
    cut = p["cutoff_hz"] * (1.0 + p["env_amount"] * (0.4 + 0.6 * v) * fenv)
    cut = np.minimum(cut, 6000.0)
    y = ladder(y, sr, cut, p["res"], block=32, drive=p["drive"])
    y = highpass(y, sr, 28.0)
    env = adsr(n, sr, dur_s, 0.003, 0.6, 0.8, rel)
    y = y * env * 0.62
    return np.stack([y, y], axis=1)


def _tapekeys(sr, dur_s, freq_hz, vel, rng, p, glide_from_hz):
    f0 = _f0(freq_hz, p["default_hz"])
    rel = p["release_ms"] / 1000.0
    dec = p["decay_ms"] / 1000.0 * _key_scale(f0, 260.0) ** 0.5
    n = _n(sr, dur_s + rel)
    t = np.arange(n, dtype=np.float64) / sr
    v = float(np.clip(vel, 0.0, 1.0))
    # tape: slow wow and quick flutter, the same on both sides (one tape)
    wow = p["wow_cents"] * (0.7 * np.sin(_TWO_PI * 0.55 * t + float(rng.uniform(0.0, _TWO_PI)))
                            + 0.3 * _drift(n, sr, rng, 1.2))
    flut = p["flutter_cents"] * np.sin(_TWO_PI * 7.3 * t + float(rng.uniform(0.0, _TWO_PI)))
    out = []
    phi = float(rng.uniform())
    for cents in (-4.0, 4.0):
        f = f0 * _cents(cents + wow + flut)
        ph = _TWO_PI * _phase(f, n, sr, phi)
        idx = (0.5 + 0.9 * v) * _key_scale(f0, 400.0) * np.exp(-t / 0.4)
        y = np.sin(ph + idx * np.sin(ph))                  # soft tine body
        y = y + 0.08 * np.sin(2.0 * ph) * np.exp(-t / 0.5)
        out.append(y)
    st = np.stack(out, axis=1)
    amp = 0.3 * np.exp(-t / 0.15) + 0.7 * np.exp(-t / max(dec, 0.05))
    env = amp * adsr(n, sr, dur_s, 0.004, 1.0, 1.0, rel)
    st *= env[:, None]
    st = np.tanh(st * 1.4) / np.tanh(1.4)                 # tape saturation
    st = onepole_lp(onepole_lp(st, sr, p["cutoff_hz"]), sr, p["cutoff_hz"] * 1.6)
    st = highpass(peak_eq(st, sr, 300.0, -2.0), sr, 25.0)   # 1:1 FM makes DC
    return width(st, 0.7) * 0.62


def _organ(sr, dur_s, freq_hz, vel, rng, p, glide_from_hz):
    f0 = _f0(freq_hz, p["default_hz"])
    rel = p["release_ms"] / 1000.0
    n = _n(sr, dur_s + rel)
    t = np.arange(n, dtype=np.float64) / sr
    v = float(np.clip(vel, 0.0, 1.0))
    # drawbars 8', 4', 2 2/3', 2' steady; 16' underneath
    steady = ((0.5, 0.35), (1.0, 1.0), (2.0, 0.7), (3.0, 0.45), (4.0, 0.3))
    perc = ((3.0, 0.6), (4.0, 0.45), (6.0, 0.25), (8.0, 0.12))
    pa = p["percussion"] * (0.15 + 0.85 * v)
    out = []
    for cents in (-1.5, 1.5):
        f = f0 * _cents(cents)
        # one tonewheel per pitch: percussion adds in phase with the drawbars
        body = _sine_stack(f, n, sr, steady, rng, phase_jitter=False)
        top = _sine_stack(f, n, sr, perc, rng, phase_jitter=False) * np.exp(-t / 0.12)
        y = body + pa * top
        out.append(y)
    st = np.stack(out, axis=1)
    # key click: a few ms of bright filtered noise
    m = min(n, _n(sr, 0.006))
    click = np.zeros(n)
    click[:m] = rng.standard_normal(m) * np.linspace(1.0, 0.0, m)
    click = bandpass(click, sr, min(f0 * 8.0, 6000.0), 1.5)
    st += (p["click"] * (0.4 + 0.6 * v) * click)[:, None]
    env = adsr(n, sr, dur_s, 0.003, p["decay_ms"] / 1000.0, 0.55, rel)
    st = peak_eq(width(st, 0.5), sr, 280.0, -2.5)
    return st * env[:, None] * 0.28


def _bell(sr, dur_s, freq_hz, vel, rng, p, glide_from_hz):
    f0 = _f0(freq_hz, p["default_hz"])
    rel = p["release_ms"] / 1000.0
    dec = p["decay_ms"] / 1000.0
    n = _n(sr, max(dur_s, 0.05) + max(rel, 3.0 * dec * 0.5))
    t = np.arange(n, dtype=np.float64) / sr
    v = float(np.clip(vel, 0.0, 1.0))
    ks = _key_scale(f0, 700.0)
    out = []
    phi = float(rng.uniform())
    for cents in (-3.0, 3.0):
        f = f0 * _cents(cents)
        ph = _TWO_PI * _phase(f, n, sr, phi)
        phm = _TWO_PI * _phase(p["ratio"] * f, n, sr, 0.0)
        idx = p["index"] * (0.4 + 0.6 * v) * ks * np.exp(-t / (0.35 * dec))
        bellish = np.sin(ph + idx * np.sin(phm))
        body = np.sin(ph)
        out.append((1.0 - p["body"]) * bellish * np.exp(-t / (0.6 * dec))
                   + p["body"] * body * np.exp(-t / dec))
    st = np.stack(out, axis=1)
    env = adsr(n, sr, dur_s + 3.0 * dec, 0.0015, 1.0, 1.0, rel)
    st = highpass(st, sr, 25.0)                   # 1:1 FM makes DC
    return width(st, 0.6) * env[:, None] * 0.7


RENDERERS = {
    "warmpad": _warmpad, "strings": _strings, "breathpad": _breathpad,
    "glasskeys": _glasskeys, "junopoly": _junopoly, "synthlead": _synthlead,
    "synthbass": _synthbass, "tapekeys": _tapekeys, "organ": _organ,
    "bell": _bell,
}


def render(name: str, sr: int, dur_s: float, freq_hz, velocity: float,
           rng: np.random.Generator, p: dict, glide_from_hz=None) -> np.ndarray:
    st = RENDERERS[name](sr, dur_s, freq_hz, velocity, rng, p, glide_from_hz)
    return st * _TRIM[name]
