"""The bass lines and comping figures the jazz, house, hip hop and r&b styles play.

Offline (:mod:`plume.compose`) and live (:mod:`plume.duet`) read the same figures
here, so a genre is the same genre on either path.  Beats are quarter notes from the
start of the bar.  Everything is a pure function of its arguments.
"""
from __future__ import annotations

import numpy as np

from plume.theory import Chord

#: A bass figure is (beat, length in beats, velocity, chord degree).  Degrees are
#: resolved against the chord playing, so a line never argues with its own harmony.
BASS_FIGURES: dict[str, tuple[tuple[float, float, float, int], ...]] = {
    # Off the kick: the eighth between each pair of beats, with a late octave push.
    "house_off": ((0.5, 0.4, 0.85, 1), (1.5, 0.25, 0.8, 1), (1.75, 0.2, 0.62, 8),
                  (2.5, 0.4, 0.85, 1), (3.5, 0.22, 0.72, 1), (3.75, 0.2, 0.66, 8)),
    # Locks to the boom bap kick on 1, the last sixteenth of 2 and the and of 3.
    "boom_hits": ((0.0, 1.5, 0.9, 1), (1.75, 0.7, 0.76, 1), (2.5, 1.3, 0.84, 1)),
    # A sung line: long root, then up through the chord and back down to it.
    "melodic": ((0.0, 1.4, 0.85, 1), (1.5, 0.45, 0.62, 5), (2.0, 0.9, 0.74, 3),
                (3.0, 0.45, 0.6, 7), (3.5, 0.45, 0.68, 5)),
}
WALKING = "walking"
#: Bass styles that keep their figure at any density above a whole note.
FIGURE_BASS_STYLES = frozenset({*BASS_FIGURES, WALKING})

#: A comp is (beat, length in beats, velocity).  Each list is one bar's figure.
#: Jazz sits on the triplet grid, which is where its swing comes from.
COMP_FIGURES: dict[str, tuple[tuple[tuple[float, float, float], ...], ...]] = {
    "shell_comp": (
        ((0.0, 0.6, 0.55), (5 / 3, 0.5, 0.48)),        # the Charleston
        ((2 / 3, 0.4, 0.42), (8 / 3, 0.5, 0.5)),
        ((5 / 3, 0.4, 0.5), (11 / 3, 0.3, 0.42)),
        ((0.0, 1.2, 0.5), (8 / 3, 0.4, 0.45)),
    ),
    "house_stabs": (
        ((0.75, 0.25, 0.62), (1.5, 0.25, 0.7), (2.75, 0.25, 0.62), (3.5, 0.25, 0.66)),
    ),
    "neo_comp": (
        ((0.0, 1.5, 0.58), (1.75, 0.6, 0.42), (2.5, 1.2, 0.5)),
        ((0.0, 2.4, 0.55), (2.75, 1.0, 0.46)),
    ),
}


def degree_semitones(intervals: tuple[int, ...], degree: int) -> int:
    """Semitones above the root for a chord degree: 1, 3, 5, 7 or 8 (the octave).

    A chord without that degree, such as a bare root and fifth, answers with the root.
    """
    if degree == 8:
        return 12
    wanted = {3: (3, 4), 5: (7, 6, 8), 7: (10, 11, 9)}.get(degree, ())
    for iv in wanted:
        if iv in intervals:
            return iv
    return 0


def walking_line(root_pc: int, intervals: tuple[int, ...], next_root_pc: int,
                 rng: np.random.Generator, *, low: int = 33) -> list[int]:
    """Four quarter notes for a bar: the root, two more notes, then a half step away
    from the next bar's root, so each bar leans into the one after it."""
    root = low + ((root_pc - low) % 12)
    third = degree_semitones(intervals, 3)
    fifth = degree_semitones(intervals, 5)
    seventh = degree_semitones(intervals, 7)
    shapes = []
    if third:
        shapes += [(third, fifth or 12), (2, third)]
    if seventh:
        shapes.append((seventh - 12, (fifth or 12) - 12))
    if not shapes:
        shapes.append((fifth or 12, 12))
    second, third_note = shapes[int(rng.integers(len(shapes)))]
    a, b = root + second, root + third_note
    target = low + ((next_root_pc - low) % 12)   # where the next bar's root really sits
    if target == b:                              # already there: step away and come back
        last = b + (1 if float(rng.random()) < 0.5 else -1)
    else:
        last = target + (-1 if b < target else 1)
        if last == b:                            # one step off it: come from the far side
            last = target + (1 if b < target else -1)
    return [root, a, b, last]


def rootless_voicing(chord: Chord, *, low: int = 52, shell: bool = False) -> list[int]:
    """The chord without its root, which is the bass player's note.

    ``shell`` keeps the third and seventh and adds a ninth, the small voicing a jazz
    pianist comps with.  Otherwise every tone above the root stacks up from the third,
    which is what gives 9ths and 11ths their colour.  The third is always placed in the
    same octave, so moving between chords is a small step and needs no memory.
    """
    upper = [iv for iv in chord.intervals if iv != 0]
    if not upper:
        return [low]
    third = upper[0]
    if shell:
        keep = [third]
        seventh = degree_semitones(chord.intervals, 7)
        if seventh and seventh != third:
            keep.append(seventh)
        if "b5" not in chord.quality and "dim" not in chord.quality:
            keep.append(14)
        upper = keep
    base = low + ((chord.root + third - low) % 12)
    return sorted(base + iv - third for iv in upper)
