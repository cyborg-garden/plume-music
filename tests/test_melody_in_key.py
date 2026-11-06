"""The melody never leaves the key, at the edges of its register too.

2026-11-06 take, midwest in D major: the vibes melody kept landing a C natural on
the Em7 bars of the lift.  The contour walk stepped below the register floor
(MIDI 72) and the pitch was then clamped to 72 itself, a C, whatever the key.
"""
import math
from types import SimpleNamespace

import pytest

from plume.duet import DuetEngine, _tonic_pc, choose_voicing
from plume.styles_lib import load_style
from plume.theory import parse_key

_MAJOR = (0, 2, 4, 5, 7, 9, 11)
_MINOR = (0, 2, 3, 5, 7, 8, 10)


def _lead_pitches(style, key, chords, bars=64, seed=0, density=0.8):
    eng = DuetEngine(style=style, key_fallback=key, seed=seed, sr=22050)
    k = parse_key(key)
    out = []
    for bar in range(bars):
        chord = chords[bar % len(chords)]
        voicing, _ = choose_voicing(chord, 1.0, key, key_fallback=k)
        tonic = _tonic_pc(SimpleNamespace(key=key), k)
        for ev in eng._motif_events("vibes", density, 0.6, voicing, None, bar,
                                    tonic, key):
            out.append((bar, chord, round(69 + 12 * math.log2(ev[4] / 440))))
    return out


def test_midwest_d_major_lift_stays_in_d_major():
    """The take itself: Dmaj7 Em7 A7 Dmaj7, lifted density, no C natural."""
    key = "D major"
    pcs = {(2 + s) % 12 for s in _MAJOR}
    for seed in range(8):
        for bar, chord, m in _lead_pitches("midwest", key,
                                           ["Dmaj7", "Em7", "A7", "Dmaj7"], seed=seed):
            assert m % 12 in pcs, f"seed {seed} bar {bar} {chord}: midi {m} out of {key}"


@pytest.mark.parametrize("style", ["midwest", "lofi", "jazz", "rnb", "hiphop",
                                   "house", "synthwave", "trap"])
@pytest.mark.parametrize("tonic", ["C", "D", "E", "F", "G", "A", "Bb", "B"])
@pytest.mark.parametrize("mode", ["major", "minor"])
def test_melody_stays_in_key_and_register(style, tonic, mode):
    cfg = load_style(style)
    if not cfg.get("melody", {}).get("motifs"):
        pytest.skip("style has no motif melody")
    key = f"{tonic} {mode}"
    k = parse_key(key)
    tonic_pc = _tonic_pc(SimpleNamespace(key=key), k)
    pcs = {(tonic_pc + s) % 12 for s in (_MAJOR if mode == "major" else _MINOR)}
    low, high = cfg["melody"].get("register", [72, 86])
    # Diatonic seventh chords on i/ii, IV and V: a walk that has to reach both edges.
    names = "C Db D Eb E F Gb G Ab A Bb B".split()
    root = lambda st: names[(tonic_pc + st) % 12]  # noqa: E731
    chords = ([f"{root(0)}maj7", f"{root(2)}m7", f"{root(5)}maj7", f"{root(7)}7"]
              if mode == "major" else
              [f"{root(0)}m7", f"{root(5)}m7", f"{root(8)}maj7", f"{root(7)}7"])
    for seed in range(3):
        for bar, chord, m in _lead_pitches(style, key, chords, bars=32, seed=seed):
            assert m % 12 in pcs, f"{style} {key} bar {bar} {chord}: midi {m}"
            assert low <= m <= high, f"{style} {key} bar {bar}: midi {m} outside register"
