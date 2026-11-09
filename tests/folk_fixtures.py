"""Synthetic guitar takes for the folk guess: strummed and fingerpicked, plain or coloured.

Built from the same additive steel string as test_acoustic.py (soft attack, upper
partials dying first), in standard tuning shapes rather than an open tuning.  What
varies is what the folk guess claims to hear:

* ``strum``: a down-down-up-up-down-up pattern, the down strums across six strings and
  the up strums across the top three, each raked over ``rake_ms``;
* ``pick``: an eighth-note Travis-style pattern, one string at a time, the thumb
  alternating root and fifth on the beat;
* ``sevenths``: every chord gets its major or minor seventh, the colour of midwest or
  lofi guitar rather than folk's plain triads.

Pure functions of their arguments, with one ``np.random.default_rng(seed)``.
"""
from __future__ import annotations

import numpy as np
from test_acoustic import SR, _add, _string

from plume.theory import parse_key, roman_to_chord

FOLK_NUMERALS = ("I", "IV", "I", "V")
#: (beat in bar, down?) for the strum pattern: D D U U D U.
_STRUMS = ((0.0, True), (1.0, True), (1.5, False), (2.5, False), (3.0, True), (3.5, False))


def _shape(chord, sevenths: bool) -> list[int]:
    """Six strings, low to high: root, fifth, octave, third, fifth, octave."""
    root = 40 + (chord.root - 40) % 12
    third = next(iv for iv in chord.intervals if iv in (3, 4))
    upper = root + 12
    if sevenths:
        upper = root + (11 if third == 4 else 10)
    return [root, root + 7, upper, root + 12 + third, root + 19, root + 24]


def make_take(kind: str = "strum", *, key: str = "G major", bpm: float = 96.0,
              bars: int = 12, rake_ms: tuple[float, float] = (25.0, 45.0),
              sevenths: bool = False, numerals=FOLK_NUMERALS, seed: int = 5,
              gain_db: float = -18.0) -> np.ndarray:
    if kind not in ("strum", "pick"):
        raise ValueError(f"unknown kind {kind!r}")
    rng = np.random.default_rng(seed)
    k = parse_key(key)
    beat = 60.0 / float(bpm)
    buf = np.zeros(int(round((bars * 4 * beat + 4.0) * SR)), dtype=np.float64)
    for b in range(bars):
        tones = _shape(roman_to_chord(numerals[b % len(numerals)], k), sevenths)
        t_bar = b * 4 * beat
        if kind == "strum":
            for at, down in _STRUMS:
                t0 = t_bar + at * beat + float(rng.normal(0.0, 0.008))
                rake = float(rng.uniform(*rake_ms)) / 1000.0
                strings = list(range(6)) if down else [5, 4, 3]
                for i, s in enumerate(strings):
                    _add(buf, _string(tones[s], dur_s=2.4,
                                      velocity=(0.85 if down else 0.5) * 0.95 ** i
                                      * float(rng.uniform(0.9, 1.05)),
                                      rng=rng, attack_ms=float(rng.uniform(4.0, 9.0)),
                                      tau_s=2.0), t0 + rake * i / max(1, len(strings) - 1))
        else:
            order = (0, 3, 1, 4, 0, 5, 1, 3)     # thumb on the beat, fingers between
            for e in range(8):
                t = t_bar + e * 0.5 * beat + float(rng.normal(0.0, 0.008))
                s = order[e]
                on_beat = e % 2 == 0
                _add(buf, _string(tones[s], dur_s=2.0,
                                  velocity=(0.8 if on_beat else 0.55)
                                  * float(rng.uniform(0.9, 1.05)),
                                  rng=rng, attack_ms=float(rng.uniform(4.0, 9.0)),
                                  tau_s=1.8), t)
    peak = float(np.max(np.abs(buf))) or 1.0
    return (buf * (10.0 ** (gain_db / 20.0)) / peak).astype(np.float32)


def make_tremolo(*, key: str = "D major", bpm: float = 96.0, bars: int = 12,
                 ioi_ms: float = 60.0, seed: int = 3, gain_db: float = -18.0) -> np.ndarray:
    """Tremolo picking: one string picked every ``ioi_ms``, the root of each chord.

    Midwest and post-rock, never a strum.  The picks land closer together than an onset's
    reading window, so the attack rises, dips and rises again inside it, where a strum's
    stays up through one gesture.
    """
    rng = np.random.default_rng(seed)
    k = parse_key(key)
    beat = 60.0 / float(bpm)
    end = bars * 4 * beat
    buf = np.zeros(int(round((end + 3.0) * SR)), dtype=np.float64)
    t = 0.0
    while t < end:
        chord = roman_to_chord(FOLK_NUMERALS[int(t // (4 * beat)) % len(FOLK_NUMERALS)], k)
        _add(buf, _string(62 + (chord.root - 62) % 12, dur_s=0.6,
                          velocity=float(rng.uniform(0.6, 0.8)), rng=rng, attack_ms=4.0,
                          tau_s=0.8), t)
        t += ioi_ms / 1000.0
    peak = float(np.max(np.abs(buf))) or 1.0
    return (buf * (10.0 ** (gain_db / 20.0)) / peak).astype(np.float32)
