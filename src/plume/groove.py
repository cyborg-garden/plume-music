"""Groove: swing, per-role timing bias, humanization. SPEC.md section 4.

One MPC-style swing knob (Roger Linn definition: delay every even-numbered
16th; 50% = straight, 66% = exact triplet) plus a per-role timing-bias table
(e.g. lofi's kick -15 ms / snare +20 ms "drunk" feel) plus small gaussian
jitter. Deterministic given the caller's np.random.Generator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# track voice families recognised for timing bias lookups
ROLES = ("kick", "snare", "hat", "clap", "bass", "pad", "pluck", "epiano", "lead")

_EPS = 1e-6


@dataclass(frozen=True)
class GrooveParams:
    swing: float                     # 0.5 straight .. 0.66 triplet
    timing_bias_ms: dict[str, float] # role -> ms (negative = early)
    humanize_ms: float               # timing jitter SD in ms (<= 10)
    humanize_vel: float              # velocity jitter SD (fraction of full scale)


def apply_groove(onset_beats: float, *, role: str, bpm: float, g: GrooveParams,
                 rng: np.random.Generator, velocity: float) -> tuple[float, float]:
    """Map a grid onset (in beats, quarter notes; bars assumed 4 beats) to
    (onset_seconds, velocity), groove-adjusted.

    Order of operations (SPEC section 4):
      1. swing: even 16ths (.25/.75 within a beat) delayed by
         (swing - 0.5) * 2 * sixteenth_duration
      2. + timing_bias_ms.get(role, 0)
      3. + N(0, humanize_ms) timing jitter
      4. velocity: +0.08 downbeat accent (integer beat 0 of the bar),
         + N(0, humanize_vel) jitter, clipped to 0.05..1.

    The returned onset may be slightly negative for early-biased roles at
    position 0; the engine clamps at render time.  onset_beats may be
    bar-local or global — both work, since swing looks at the position
    within the beat and the accent at the position within a 4-beat bar.
    """
    beat_s = 60.0 / bpm
    sixteenth_s = beat_s / 4.0
    t = onset_beats * beat_s

    frac = onset_beats % 1.0
    if abs(frac - 0.25) < _EPS or abs(frac - 0.75) < _EPS:
        t += (g.swing - 0.5) * 2.0 * sixteenth_s

    t += g.timing_bias_ms.get(role, 0.0) / 1000.0
    t += float(rng.normal(0.0, 1.0)) * (g.humanize_ms / 1000.0)

    v = velocity
    if abs(onset_beats % 4.0) < _EPS:
        v += 0.08  # downbeat accent, pre-clip
    v += float(rng.normal(0.0, 1.0)) * g.humanize_vel
    v = float(min(1.0, max(0.05, v)))
    return float(t), v
