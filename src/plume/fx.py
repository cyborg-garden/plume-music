"""plume.fx — effects DSP (SPEC section 8).

All functions are stateless (the one exception is :class:`BusTone`, a bar-by-bar
bus stage that has to carry filter and reverb state across bar lines), take/return numpy arrays, and are fully
vectorized (no per-sample Python loops; recursive filters run through
scipy.signal.lfilter).

Conventions
-----------
- float32 audio in/out, sample rate passed explicitly.
- Mono in -> mono out unless noted; ``chorus`` and ``freeverb`` return
  stereo ``(N, 2)``.  Functions that accept stereo ``(N, 2)`` input filter
  along axis 0.

Filter implementation note (SPEC allows either): the SVF low/high-pass are
the TPT/Zavalishin topology collapsed to its exact ``lfilter`` biquad
equivalent.  With ``g = tan(pi*fc/sr)`` and ``k = 1/Q`` the TPT structure's
transfer function is::

    LP(z) = g^2 (1 + z^-1)^2 / D(z)
    HP(z) =     (1 - z^-1)^2 / D(z)
    BP(z) = g k (1 - z^-2)   / D(z)          (peak gain 1)
    D(z)  = (1 + g k + g^2) + 2 (g^2 - 1) z^-1 + (1 - g k + g^2) z^-2

which we run as a single biquad via ``scipy.signal.lfilter`` — bit-exact to
the per-sample TPT recursion for constant coefficients, and C-speed.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

__all__ = [
    "svf_lowpass",
    "svf_highpass",
    "drive",
    "chorus",
    "delay",
    "freeverb",
    "ducker",
    "lofi",
    "limiter",
    "equal_power_gain",
    "fade_curve",
    "phase_for_gain",
    "correlation",
    "crossfade",
    "db_glide",
    "BusTone",
]

_F32 = np.float32


# ---------------------------------------------------------------------------
# musical fades / crossfades (SPEC-V1 section 5; v1.4 gentle entrances)
#
# Entrances and exits used to be a per-bar gain staircase that started and
# stopped on a bar edge, which reads as a hard cut on sustained material.
# These are the primitives the duet engine's envelope is built from: a
# *phase* in [0, 1] maps to an equal-power gain, so a fade can be generated
# in independent chunks (one per rendered bar) and still be sample-continuous
# across the joins, and an interrupted fade can be reversed from wherever it
# currently sits without a step (:func:`phase_for_gain`).
# ---------------------------------------------------------------------------

_IN, _OUT = "in", "out"


def _check_direction(direction: str) -> str:
    if direction not in (_IN, _OUT):
        raise ValueError(f"direction must be 'in' or 'out', got {direction!r}")
    return direction


def equal_power_gain(phase, *, direction: str = "in") -> np.ndarray:
    """Equal-power fade gain for a phase in ``[0, 1]`` (clipped).

    ``in``  -> ``sin(pi/2 * phase)``   (0 at phase 0, 1 at phase 1)
    ``out`` -> ``cos(pi/2 * phase)``   (1 at phase 0, 0 at phase 1)

    "Equal power" is the constant-energy property
    ``in(p)**2 + out(p)**2 == 1``: crossfading two uncorrelated sources with
    this pair holds the summed power flat, so a texture change is a blend
    with no level notch in the middle rather than a dip.
    """
    _check_direction(direction)
    p = np.clip(np.asarray(phase, dtype=np.float64), 0.0, 1.0)
    ang = p * (np.pi / 2.0)
    return np.sin(ang) if direction == _IN else np.cos(ang)


def phase_for_gain(gain: float, *, direction: str = "in") -> float:
    """Inverse of :func:`equal_power_gain` — the phase at which it equals ``gain``.

    Used to *reverse* a fade that is already part-way through: an exit that
    the human interrupts resumes as an entrance from the gain it had reached,
    so the bus never steps.
    """
    _check_direction(direction)
    g = float(np.clip(gain, 0.0, 1.0))
    ang = np.arcsin(g) if direction == _IN else np.arccos(g)
    return float(ang / (np.pi / 2.0))


def fade_curve(n: int, *, direction: str = "in", start: float = 0.0,
               end: float = 1.0) -> np.ndarray:
    """``n``-sample equal-power curve sweeping phase ``start`` -> ``end``.

    Half-open by design: sample ``k`` sits at ``start + (end-start)*k/n``, so
    the last sample is one step *short* of ``end`` and consecutive segments
    ``[p0,p1)``, ``[p1,p2)`` concatenate into one smooth curve with no
    repeated sample at the join.  This is what lets the duet engine generate
    a multi-bar fade one bar at a time.
    """
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    step = (float(end) - float(start)) / n
    phase = float(start) + step * np.arange(n, dtype=np.float64)
    return equal_power_gain(phase, direction=direction)


def correlation(a: np.ndarray, b: np.ndarray, n: int) -> float:
    """Normalized correlation of two buffers over their first ``n`` samples.

    0 for independent material, 1 for identical.  Used to pick the crossfade
    law: see :func:`crossfade`.  Silence on either side reads as 0.
    """
    n = int(min(n, np.asarray(a).shape[0], np.asarray(b).shape[0]))
    if n <= 0:
        return 0.0
    x = np.asarray(a, dtype=np.float64)[:n].ravel()
    y = np.asarray(b, dtype=np.float64)[:n].ravel()
    den = float(np.sqrt(float(x @ x) * float(y @ y)))
    if den <= 0.0:
        return 0.0
    return float(np.clip(float(x @ y) / den, 0.0, 1.0))


def crossfade(out_going: np.ndarray, in_coming: np.ndarray, n_xfade: int,
              *, correlation: float = 0.0) -> np.ndarray:
    """Crossfade ``out_going`` -> ``in_coming`` over ``n_xfade`` samples.

    Both buffers are treated as starting at the same instant.  Over the
    window the outgoing side follows the equal-power ``out`` curve and the
    incoming side the ``in`` curve; past the window the outgoing side is
    silent (it reaches exactly 0 at the window edge, so truncating it there
    is continuous) and the incoming side plays at full level.  Shapes may
    differ in length; the result is as long as the longer one.

    ``correlation`` selects the fade law, and it matters musically.  The
    equal-power pair holds summed *power* flat, which is right for
    independent material — but two textures that share most of their voices
    (a pad-and-bass bed that gains a hat) are strongly correlated, and
    equal-power then *adds* up to +3 dB in the middle of the window: a swell
    in the middle of every texture change.  Dividing both curves by
    ``sqrt(s^2 + c^2 + 2 r s c)`` holds the summed *amplitude* flat instead
    as ``r`` -> 1, interpolating continuously between the two laws.  The
    divisor is exactly 1 at both window edges for every ``r``, so the join
    continuity above is untouched.  ``r = 0`` is plain equal-power.

    ``n_xfade <= 0`` is a hard splice (returns ``in_coming`` unchanged).
    """
    a = np.asarray(out_going, dtype=np.float64)
    b = np.asarray(in_coming, dtype=np.float64)
    n_xfade = int(n_xfade)
    if n_xfade <= 0:
        return b.copy()
    r = float(np.clip(correlation, 0.0, 1.0))
    g_in = fade_curve(n_xfade, direction=_IN)
    g_out = fade_curve(n_xfade, direction=_OUT)
    if r > 0.0:
        norm = np.sqrt(g_in * g_in + g_out * g_out + 2.0 * r * g_in * g_out)
        g_in = g_in / norm
        g_out = g_out / norm
    ka = min(n_xfade, a.shape[0])
    kb = min(n_xfade, b.shape[0])
    out = np.zeros((max(a.shape[0], b.shape[0]),) + b.shape[1:], dtype=np.float64)
    out[: b.shape[0]] = b
    if kb:
        out[:kb] *= g_in[:kb].reshape((kb,) + (1,) * (b.ndim - 1))
    if ka:
        out[:ka] += a[:ka] * g_out[:ka].reshape((ka,) + (1,) * (a.ndim - 1))
    return out


def db_glide(n: int, delta_db: float, n_glide: int) -> np.ndarray:
    """``n``-sample gain that starts ``delta_db`` away from unity and slides to it.

    A bar rendered at a new bus gain begins at the *previous* bar's level and
    slides to its own over ``n_glide`` samples, so a per-bar gain step (the
    entry ramp, the dynamic arc) is heard as a swell rather than a jump.
    The slide is linear in dB — a constant-rate fader move — and reaches
    exactly unity at ``n_glide``, where it stays.
    """
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    n_glide = int(min(max(n_glide, 0), n))
    g = np.ones(n, dtype=np.float64)
    if n_glide <= 0 or delta_db == 0.0:
        return g
    frac = np.arange(n_glide, dtype=np.float64) / n_glide
    g[:n_glide] = 10.0 ** ((float(delta_db) * (1.0 - frac)) / 20.0)
    return g


# ---------------------------------------------------------------------------
# SVF (TPT-equivalent biquad)
# ---------------------------------------------------------------------------

def _svf_coeffs(sr: int, cutoff_hz: float, q: float, kind: str):
    """Biquad (b, a) exactly equivalent to the TPT/Zavalishin SVF output."""
    fc = float(np.clip(cutoff_hz, 10.0, 0.45 * sr))
    g = np.tan(np.pi * fc / sr)
    k = 1.0 / max(q, 1e-3)
    a0 = 1.0 + g * k + g * g
    a = np.array([1.0, 2.0 * (g * g - 1.0) / a0, (1.0 - g * k + g * g) / a0])
    if kind == "lp":
        b = np.array([g * g, 2.0 * g * g, g * g]) / a0
    elif kind == "hp":
        b = np.array([1.0, -2.0, 1.0]) / a0
    elif kind == "bp":
        b = np.array([g * k, 0.0, -g * k]) / a0
    else:  # pragma: no cover
        raise ValueError(f"unknown SVF kind: {kind}")
    return b, a


def _svf(x: np.ndarray, sr: int, cutoff_hz: float, q: float, kind: str) -> np.ndarray:
    b, a = _svf_coeffs(sr, cutoff_hz, q, kind)
    y = lfilter(b, a, np.asarray(x, dtype=np.float64), axis=0)
    return y.astype(_F32)


def svf_lowpass(x: np.ndarray, sr: int, cutoff_hz: float, q: float = 0.707) -> np.ndarray:
    """TPT SVF low-pass (lfilter-equivalent biquad, see module docstring)."""
    return _svf(x, sr, cutoff_hz, q, "lp")


def svf_highpass(x: np.ndarray, sr: int, cutoff_hz: float, q: float = 0.707) -> np.ndarray:
    """TPT SVF high-pass (lfilter-equivalent biquad, see module docstring)."""
    return _svf(x, sr, cutoff_hz, q, "hp")


def _svf_bandpass(x: np.ndarray, sr: int, cutoff_hz: float, q: float = 0.707) -> np.ndarray:
    """Unity-peak-gain band-pass. Internal (used by voices); not in SPEC's public list."""
    return _svf(x, sr, cutoff_hz, q, "bp")


# ---------------------------------------------------------------------------
# drive
# ---------------------------------------------------------------------------

def drive(x: np.ndarray, amount: float) -> np.ndarray:
    """tanh soft saturation: ``tanh(x*a)/tanh(a)`` with ``a = 1 + 3*amount``."""
    a = 1.0 + 3.0 * float(max(amount, 0.0))
    return (np.tanh(np.asarray(x, dtype=np.float64) * a) / np.tanh(a)).astype(_F32)


# ---------------------------------------------------------------------------
# modulated-delay helpers (chorus / lofi wobble)
# ---------------------------------------------------------------------------

def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        return x.mean(axis=1)
    return x


def _var_delay_tap(x: np.ndarray, sr: int, delay_s: np.ndarray) -> np.ndarray:
    """Fractional-delay tap with a time-varying delay (linear interpolation).

    Vectorized: computes read positions for every sample at once.
    """
    n = x.shape[0]
    pos = np.arange(n, dtype=np.float64) - delay_s * sr
    pos = np.clip(pos, 0.0, n - 1.0)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = pos - i0
    return x[i0] * (1.0 - frac) + x[i1] * frac


def chorus(x: np.ndarray, sr: int, depth: float = 0.5, phase_s: float = 0.0) -> np.ndarray:
    """Juno-style chorus: 2 modulated delay lines (1-10 ms), LFOs 0.5/0.8 Hz.

    Mono in -> stereo (N, 2). 50% wet per channel; the two lines' LFOs are
    quadrature-offset so L/R decorrelate.  ``phase_s`` is how far into their
    cycle the LFOs start, for a caller running this on one note of a longer part.
    """
    xm = _mono(x).astype(np.float64)
    n = xm.shape[0]
    t = np.arange(n, dtype=np.float64) / sr + float(phase_s)
    depth = float(np.clip(depth, 0.0, 1.0))
    base = 5.5e-3                      # centre of the 1-10 ms range
    amp = 4.5e-3 * depth               # stays within 1..10 ms
    d1 = base + amp * np.sin(2.0 * np.pi * 0.5 * t)
    d2 = base + amp * np.sin(2.0 * np.pi * 0.8 * t + np.pi / 2.0)
    wet = 0.5
    left = (1.0 - wet) * xm + wet * _var_delay_tap(xm, sr, d1)
    right = (1.0 - wet) * xm + wet * _var_delay_tap(xm, sr, d2)
    return np.stack([left, right], axis=1).astype(_F32)


# ---------------------------------------------------------------------------
# tempo-synced feedback delay
# ---------------------------------------------------------------------------

def _one_pole_lp_coeffs(sr: int, cutoff_hz: float):
    a1 = float(np.exp(-2.0 * np.pi * cutoff_hz / sr))
    return np.array([1.0 - a1]), np.array([1.0, -a1])


def _shift(x: np.ndarray, d: int) -> np.ndarray:
    """Delay by d samples, zero-padded, same length."""
    if d <= 0:
        return x.copy()
    out = np.zeros_like(x)
    if d < x.shape[0]:
        out[d:] = x[:-d]
    return out


def delay(x: np.ndarray, sr: int, bpm: float, wet: float,
          feedback: float = 0.45, note: str = "3/16") -> np.ndarray:
    """Tempo-synced feedback delay with a 4 kHz one-pole low-pass in the loop.

    ``note`` is a fraction of a whole note (bar of 4/4): "3/16" = dotted
    eighth. Mono in -> mono out (dry + wet echoes).

    The feedback loop is expanded as its (exact) geometric series of passes,
    truncated below -80 dB — each pass is one lfilter'd low-pass + delay, so
    everything stays vectorized.
    """
    xm = _mono(x).astype(np.float64)
    n = xm.shape[0]
    num, den = note.split("/")
    beats = (float(num) / float(den)) * 4.0
    d_samp = max(1, int(round(beats * (60.0 / bpm) * sr)))
    blp, alp = _one_pole_lp_coeffs(sr, 4000.0)

    fb = float(np.clip(feedback, 0.0, 0.98))
    if fb > 0.0:
        k_max = int(np.ceil(np.log(1e-4) / np.log(fb)))
    else:
        k_max = 1
    k_max = min(k_max, int(np.ceil(n / d_samp)) + 1)

    acc = np.zeros(n)
    cur = lfilter(blp, alp, _shift(xm, d_samp))
    g = 1.0
    for k in range(k_max):
        acc += g * cur
        if (k + 2) * d_samp >= n:
            break
        cur = lfilter(blp, alp, _shift(cur, d_samp))
        g *= fb
        if g < 1e-4:
            break
    return (xm + float(wet) * acc).astype(_F32)


# ---------------------------------------------------------------------------
# Freeverb
# ---------------------------------------------------------------------------

_FV_COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)   # @44.1k
_FV_ALLPASS = (556, 441, 341, 225)
_FV_SPREAD = 23
_FV_DAMP = 0.2          # damp 0.5 * scaledamp 0.4
_FV_FEEDBACK = 0.84     # roomsize 0.5 * 0.28 + 0.7
_FV_AP_FB = 0.5
_FV_GAIN = 0.015
_FV_WETSCALE = 3.0      # freeverb scalewet


def _fv_comb(x: np.ndarray, sr: int, d_samp: int) -> np.ndarray:
    """One Freeverb lowpass-feedback comb.

    Recursion ``O(z) = z^-D (1 - d z^-1) / (1 - d z^-1 - f(1-d) z^-D)``
    expanded as its geometric series ``sum_k f^k z^-(k+1)D LP^k`` with
    LP the unity-DC one-pole ``(1-d)/(1 - d z^-1)`` — each pass is one
    scipy.signal.lfilter call, truncated below -60 dB (inaudible;
    deterministic).  A single dense-lfilter call would be O(n*D) and
    unusably slow at render lengths.
    """
    n = x.shape[0]
    d = _FV_DAMP
    f = _FV_FEEDBACK
    blp = np.array([1.0 - d])
    alp = np.array([1.0, -d])
    k_max = int(np.ceil(np.log(1e-3) / np.log(f)))          # ~40 passes
    acc = np.zeros(n)
    cur = _shift(x, d_samp)
    g = 1.0
    for k in range(k_max):
        acc += g * cur
        if (k + 2) * d_samp >= n:
            break
        cur = _shift(lfilter(blp, alp, cur), d_samp)
        g *= f
    return acc


def _fv_allpass(x: np.ndarray, d_samp: int) -> np.ndarray:
    """Freeverb 'allpass': O = -x + (1+f) sum_k f^k z^-(k+1)D x (f=0.5)."""
    n = x.shape[0]
    f = _FV_AP_FB
    acc = np.zeros(n)
    cur = _shift(x, d_samp)
    g = 1.0
    while True:
        acc += g * cur
        g *= f
        if g < 1e-3:
            break
        cur = _shift(cur, d_samp)
        if not cur.any():
            break
    return -x + (1.0 + f) * acc


def freeverb(x: np.ndarray, sr: int, wet: float) -> np.ndarray:
    """Freeverb: 8 parallel combs + 4 series allpasses per channel.

    Mono in -> stereo (N, 2); the right channel's delays are offset by the
    classic 23-sample stereo spread. Output = dry + wet * reverb.
    """
    xm = _mono(x).astype(np.float64)
    scale = sr / 44100.0
    wet = float(max(wet, 0.0))
    outs = []
    for offset in (0, _FV_SPREAD):
        if wet <= 0.0:
            outs.append(xm)
            continue
        w = np.zeros_like(xm)
        src = xm * _FV_GAIN
        for c in _FV_COMBS:
            w += _fv_comb(src, sr, int(round((c + offset) * scale)))
        for a in _FV_ALLPASS:
            w = _fv_allpass(w, int(round((a + offset) * scale)))
        outs.append(xm + wet * _FV_WETSCALE * w)
    return np.stack(outs, axis=1).astype(_F32)


# ---------------------------------------------------------------------------
# ducker (sidechain pump)
# ---------------------------------------------------------------------------

def ducker(x: np.ndarray, sr: int, trigger_times_s, depth: float,
           dip_db: float = -9.0, attack_ms: float = 8.0,
           release_ms: float = 220.0) -> np.ndarray:
    """Kick-triggered ducking envelope (fake sidechain compressor).

    At each trigger the gain ramps down to ``dip_db`` (scaled by ``depth``)
    over ``attack_ms``, then recovers exponentially with time-constant
    ``release_ms``. Overlapping triggers take the deeper value. Shape and
    channel count are preserved. The loop below is per *trigger* (event
    rate), not per sample.
    """
    x = np.asarray(x)
    n = x.shape[0]
    t = np.arange(n, dtype=np.float64) / sr
    atk = max(attack_ms, 0.1) / 1000.0
    rel = max(release_ms, 1.0) / 1000.0
    duck = np.zeros(n)
    for tt in trigger_times_s:
        i0 = int(round(float(tt) * sr))
        if i0 >= n or i0 < 0:
            continue
        seg = t[i0:] - t[i0]
        e = np.where(seg < atk, seg / atk, np.exp(-(seg - atk) / rel))
        duck[i0:] = np.maximum(duck[i0:], e)
    dip_lin = 10.0 ** (dip_db / 20.0)
    gain = 1.0 + float(np.clip(depth, 0.0, 1.0)) * (dip_lin - 1.0) * duck
    if x.ndim == 2:
        gain = gain[:, None]
    return (x.astype(np.float64) * gain).astype(_F32)


# ---------------------------------------------------------------------------
# lofi
# ---------------------------------------------------------------------------

def lofi(x: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Lofi treatment: tape pitch wobble + darkening low-pass + bitcrush.

    - pitch wobble: 0.6 Hz sinusoidal delay modulation whose depth gives
      +/- 8 cents * amount of pitch deviation (resample-free)
    - low-pass: 10 kHz down to 5 kHz as amount goes 0 -> 1
    - 12-bit quantize when amount > 0.5

    Shape/channels preserved. amount <= 0 returns the input as float32.
    """
    x = np.asarray(x)
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return x.astype(_F32)

    def _process(ch: np.ndarray) -> np.ndarray:
        n = ch.shape[0]
        t = np.arange(n, dtype=np.float64) / sr
        cents = 8.0 * amount
        ratio_dev = 2.0 ** (cents / 1200.0) - 1.0
        a = ratio_dev / (2.0 * np.pi * 0.6)          # delay-mod amplitude, s
        d = (a + 1e-3) + a * np.sin(2.0 * np.pi * 0.6 * t)
        return _var_delay_tap(ch.astype(np.float64), sr, d)

    if x.ndim == 2:
        y = np.stack([_process(x[:, c]) for c in range(x.shape[1])], axis=1)
    else:
        y = _process(x)
    y = _svf(y, sr, 10000.0 - 5000.0 * amount, 0.707, "lp").astype(np.float64)
    if amount > 0.5:
        y = np.round(y * 2048.0) / 2048.0            # 12-bit
    return y.astype(_F32)


# ---------------------------------------------------------------------------
# limiter
# ---------------------------------------------------------------------------

def limiter(x: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """Knee-limiter: bit-transparent below the knee, tanh soft-clip above.

    The knee sits 3 dB under the ceiling ``c``. Samples with ``|x| <= knee``
    pass through untouched (so mix peaks below the knee are preserved exactly
    and gain changes are audible/measurable); above the knee the excess is
    mapped through ``tanh`` so output magnitude stays strictly below the
    ceiling — never hard-clips. Continuous with unity slope at the knee.

    v0.1 amendment: the previous full-range ``c*tanh(x/c)`` shape compressed
    every render toward the ceiling regardless of gain settings, which made
    level changes invisible; the knee restores a transparent signal path.
    """
    c = 10.0 ** (ceiling_db / 20.0)
    knee = c * 10.0 ** (-3.0 / 20.0)
    xf = np.asarray(x, dtype=np.float64)
    over = np.abs(xf) > knee
    if not over.any():
        return xf.astype(_F32)
    y = xf.copy()
    span = c - knee
    excess = np.abs(xf[over]) - knee
    y[over] = np.sign(xf[over]) * (knee + span * np.tanh(excess / span))
    return y.astype(_F32)


# ---------------------------------------------------------------------------
# live bus tone (2026-11-06): what "darker"/"brighter"/"warmer"/"dry"/"wet" do
# ---------------------------------------------------------------------------

#: The tilt shelf: a low-pass at this corner, mixed in (dark) or subtracted
#: (bright) by up to ``BUS_TONE_TILT_MIX``.  The same shape simlead always used.
BUS_TONE_TILT_HZ = 2600.0
BUS_TONE_TILT_MIX = 0.55
#: One "wet" cue adds this much reverb send (freeverb's ``wet`` units, where
#: ``plume render``'s master default is 0.1), up to the max.  "dry" takes one off.
BUS_TONE_WET_STEP = 0.12
BUS_TONE_WET_MAX = 0.36
#: A cue's whole step glides in over this long, so it is a move, not a switch.
BUS_TONE_GLIDE_S = 1.0
_BUS_TONE_IR_S = 1.5


def _glide(cur: float, target: float, n: int, per_sample: float) -> np.ndarray:
    d = float(target) - float(cur)
    steps = np.minimum(np.arange(1, n + 1, dtype=np.float64) * per_sample, abs(d))
    return float(cur) + np.sign(d) * steps


_IR_CACHE: dict[int, np.ndarray] = {}


def _bus_reverb_ir(sr: int) -> np.ndarray:
    """Freeverb's wet path as a stereo impulse response, faded at its end."""
    ir = _IR_CACHE.get(int(sr))
    if ir is None:
        n = int(_BUS_TONE_IR_S * sr)
        imp = np.zeros(n, dtype=np.float32)
        imp[0] = 1.0
        ir = freeverb(imp, sr, 1.0).astype(np.float64) - imp[:, None]
        k = n // 10
        ir[-k:] *= np.linspace(1.0, 0.0, k)[:, None]
        _IR_CACHE[int(sr)] = ir
    return ir


class BusTone:
    """A stateful tone stage for a band bus rendered one bar at a time.

    ``process(bar, tilt=, wet=)`` takes the bar's stereo audio and the cue
    state's targets: ``tilt`` in [-1, 1] (dark .. bright) and ``wet`` in
    [0, ``BUS_TONE_WET_MAX``].  Both glide toward the target at one cue step
    per ``BUS_TONE_GLIDE_S``, and are clamped, so nothing a player says can push
    it past a sane bound.  The shelf filter runs on every bar, carrying its
    state, so the moment a cue lands there is no transient; the reverb is a
    convolution with freeverb's own impulse response whose tail overlap-adds
    into the bars after, so "dry" lets what was already ringing ring out.

    Until anything moves, the bar comes back bit-exact: an uncued take sounds
    and measures exactly as it did before this stage existed.
    """

    def __init__(self, sr: int):
        self.sr = int(sr)
        self.tilt = 0.0
        self.wet = 0.0
        self._b, self._a = _svf_coeffs(self.sr, BUS_TONE_TILT_HZ, 0.707, "lp")
        self._zi: np.ndarray | None = None
        self._tail = np.zeros((0, 2), dtype=np.float64)

    def process(self, x: np.ndarray, *, tilt: float, wet: float) -> np.ndarray:
        x = np.asarray(x)
        n = x.shape[0]
        if n == 0:
            return x.astype(_F32)
        xf = x.astype(np.float64).reshape(n, -1)
        if xf.shape[1] == 1:
            xf = np.repeat(xf, 2, axis=1)
        if self._zi is None:
            self._zi = np.zeros((2, xf.shape[1]))
        lp, self._zi = lfilter(self._b, self._a, xf, axis=0, zi=self._zi)
        tilt = min(1.0, max(-1.0, float(tilt)))
        wet = min(BUS_TONE_WET_MAX, max(0.0, float(wet)))
        tc = _glide(self.tilt, tilt, n, 1.0 / (BUS_TONE_GLIDE_S * self.sr))
        wc = _glide(self.wet, wet, n, BUS_TONE_WET_STEP / (BUS_TONE_GLIDE_S * self.sr))
        self.tilt, self.wet = float(tc[-1]), float(wc[-1])
        moved = bool(np.any(tc != 0.0)) or bool(np.any(wc != 0.0)) or self._tail.shape[0] > 0
        if not moved:
            return x.astype(_F32)
        y = xf + (tc * BUS_TONE_TILT_MIX)[:, None] * (xf - lp)
        if np.any(wc != 0.0) or self._tail.shape[0]:
            from scipy.signal import fftconvolve
            acc_n = n
            parts = []
            if np.any(wc != 0.0):
                send = y.mean(axis=1) * wc
                ir = _bus_reverb_ir(self.sr)
                parts.append(np.stack([fftconvolve(send, ir[:, c]) for c in (0, 1)],
                                      axis=1))
            parts.append(self._tail)
            acc_n = max([acc_n] + [p.shape[0] for p in parts])
            acc = np.zeros((acc_n, 2))
            for p in parts:
                acc[:p.shape[0]] += p
            y = y + acc[:n]
            self._tail = acc[n:].copy()
        return y.astype(_F32)

    def flush(self) -> np.ndarray:
        """What the reverb still has ringing after the last bar."""
        tail, self._tail = self._tail, np.zeros((0, 2), dtype=np.float64)
        return tail.astype(_F32)
