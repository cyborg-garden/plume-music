"""Music theory: notes, keys, chords, roman numerals, voicing.

SPEC.md section 1. Pure python, no numpy needed here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NOTE_NAMES: list[str] = ["c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b"]

_LETTER_PC = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_ACC = {"": 0, "#": 1, "s": 1, "b": -1}
_PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_FLAT_PC_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Per-key preferred accidental (standard practice). Sharp-side key signatures
# spell chromatic pitch classes with sharps; flat-side keys use flats. C major
# and A minor default to flats: their chromatic chord roots come from borrowed
# flat degrees (bIII, bVI, bVII). Pitch class 6 is F# in both modes: a heard
# key has only its pitch class, and guitarists call it F# major, not Gb.  A key
# typed with a flat ("Gb major") keeps its flats (``Key.flats``).
_SHARP_MAJOR_TONICS = frozenset({2, 4, 6, 7, 9, 11})   # D, E, F#, G, A, B majors
_SHARP_MINOR_TONICS = frozenset({1, 4, 6, 8, 11})   # c#, e, f#, g#, b minors

_NOTE_RE = re.compile(r"^([a-gA-G])([#sb]?)(-?\d+)$")
_TONIC_RE = re.compile(r"^([a-gA-G])([#sb]?)$")
# glued tonic+mode single-token key forms: "Fm", "f#m", "Ebmin", "Gmaj", "Aminor"
_TONIC_MODE_RE = re.compile(r"^([a-gA-G])([#sb]?)(m|min|minor|maj|major)$", re.IGNORECASE)
_CHORD_RE = re.compile(r"^([A-Ga-g])([#b]?)(.*)$")


def note_to_midi(name: str) -> int:
    """'c4' -> 60, 'eb2' -> 39. Accepts '#', 's' (sharp) and 'b' (flat)."""
    m = _NOTE_RE.match(name.strip())
    if not m:
        raise ValueError(f"invalid note name: {name!r}")
    letter, acc, octave = m.groups()
    return (int(octave) + 1) * 12 + _LETTER_PC[letter.lower()] + _ACC[acc]


def midi_to_hz(midi: float) -> float:
    """Equal temperament, A4=440. Accepts float midi for glides."""
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


_MODES = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
}

_MODE_ALIASES = {
    "": "major",
    "maj": "major",
    "major": "major",
    "m": "minor",
    "min": "minor",
    "minor": "minor",
    "harmonic_minor": "harmonic_minor",
    "harm_minor": "harmonic_minor",
    "harmonicminor": "harmonic_minor",
    "dorian": "dorian",
    "phrygian": "phrygian",
}


@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class 0-11
    mode: str   # "major","minor","harmonic_minor","dorian","phrygian"
    #: How the tonic was written, when it was: True for "Gb", False for "F#", None
    #: for a natural or a key that was only heard.  Spelling only: Gb major and F#
    #: major are the same key, and compare equal.
    flats: bool | None = field(default=None, compare=False)

    def scale(self) -> list[int]:
        """Pitch classes, ascending from tonic."""
        return [(self.tonic + iv) % 12 for iv in _MODES[self.mode]]


_KEY_FORMAT_HINT = (
    "expected TONIC [MODE], e.g. 'F minor', 'F min', 'Fm', 'f#m', 'Ebmin', "
    "'C' (bare tonic = major), 'a dorian'; "
    "modes: major, minor, harmonic_minor, dorian, phrygian"
)


def parse_key(s: str) -> Key:
    """'C min', 'f# minor', 'Eb maj', 'a dorian', 'c harmonic minor'.

    Also accepts glued single-token forms: 'Fm', 'f#m', 'Ebmin', 'Gmaj'.
    A bare tonic ('F') is major; the 'm'/'min'/'minor' suffix forms are minor.
    """
    parts = s.strip().replace("-", "_").split()
    if not parts:
        raise ValueError(f"invalid key: {s!r} — {_KEY_FORMAT_HINT}")
    tm = _TONIC_RE.match(parts[0])
    if tm:
        letter, acc = tm.groups()
        mode_txt = "_".join(p.lower() for p in parts[1:])
    else:
        gm = _TONIC_MODE_RE.match(parts[0])
        if not gm or parts[1:]:
            raise ValueError(f"invalid key tonic: {parts[0]!r} in {s!r} — {_KEY_FORMAT_HINT}")
        letter, acc, mode_txt = gm.groups()
        mode_txt = mode_txt.lower()
    tonic = (_LETTER_PC[letter.lower()] + _ACC[acc]) % 12
    if mode_txt not in _MODE_ALIASES:
        raise ValueError(f"unknown mode: {mode_txt!r} in {s!r} — {_KEY_FORMAT_HINT}")
    flats = {"b": True, "#": False, "s": False}.get(acc)
    return Key(tonic=tonic, mode=_MODE_ALIASES[mode_txt], flats=flats)


_LETTERS = "CDEFGAB"
_ACC_NAME = {0: "", 1: "#", 11: "b"}


def _letter_name(letter: str, pc: int) -> str | None:
    """``pc`` written on ``letter`` with one accidental at most, or None."""
    acc = _ACC_NAME.get((pc - _LETTER_PC[letter.lower()]) % 12)
    return None if acc is None else letter + acc


def _default_tonic(key: Key) -> str:
    """How a key known only by its pitch class is written."""
    if key.mode == "major":
        sharp = key.tonic in _SHARP_MAJOR_TONICS
    else:
        sharp = key.tonic in _SHARP_MINOR_TONICS
    return (_PC_NAMES if sharp else _FLAT_PC_NAMES)[key.tonic]


def _spells_cleanly(tonic: str, key: Key) -> bool:
    """Does the scale on ``tonic`` take each letter once with one accidental at most?
    Harmonic minor is judged by its natural minor: its raised seventh is an
    accidental in any key."""
    mode = "minor" if key.mode == "harmonic_minor" else key.mode
    scale = Key(tonic=key.tonic, mode=mode).scale()
    start = _LETTERS.index(tonic[0])
    return all(_letter_name(_LETTERS[(start + i) % 7], pc) is not None
               for i, pc in enumerate(scale))


def tonic_name(key: Key) -> str:
    """The key's tonic as written: as typed when that is a key people write (Gb, F#,
    Cb, D# minor), else the usual name of its pitch class.  A# major would need F##
    and G##, so it is written Bb major."""
    if key.flats is not None:
        letter = _LETTERS[[_LETTER_PC[c.lower()] for c in _LETTERS].index(
            (key.tonic + (1 if key.flats else -1)) % 12)]
        typed = letter + ("b" if key.flats else "#")
        if _spells_cleanly(typed, key):
            return typed
    return _default_tonic(key)


def key_name(key: Key) -> str:
    """'Gb major', 'F# minor': the one name a key goes by on the page and in the logs."""
    return f"{tonic_name(key)} {key.mode.replace('_', ' ')}"


def key_prefers_flats(key: Key) -> bool:
    """True when chord/note spelling in this key should use flats.

    Major keys: F, Bb, Eb, Ab, Db, Gb (and C, whose chromatic chord roots are
    borrowed flat degrees) prefer flats; G, D, A, E, B, F# prefer sharps.
    Minor-family modes (minor, harmonic_minor, dorian, phrygian): d, g, c, f,
    bb, eb (and a) prefer flats; e, b, f#, c#, g# prefer sharps.  A key whose
    tonic is written with an accidental (see :func:`tonic_name`) follows it.
    """
    tonic = tonic_name(key)
    if tonic.endswith("b"):
        return True
    if tonic.endswith("#"):
        return False
    if key.mode == "major":
        return key.tonic not in _SHARP_MAJOR_TONICS
    return key.tonic not in _SHARP_MINOR_TONICS


def spell(pc: int, key: Key | None = None) -> str:
    """Spell pitch class ``pc`` (0-11) for the given key: 'Bb' in F minor,
    'A#' in B major. Without a key, sharps (the historical default).

    A note of the key's own scale takes its scale degree's letter, so each
    letter appears once: E# (not F) in F# major, Cb (not B) in Gb major.
    Notes outside the scale, and a degree that would need a double accidental,
    use the key's preferred accidental.
    """
    pc %= 12
    names = (_FLAT_PC_NAMES if key is not None and key_prefers_flats(key)
             else _PC_NAMES)
    if key is not None:
        scale = key.scale()
        if pc in scale:
            letter = _LETTERS[(_LETTERS.index(tonic_name(key)[0])
                               + scale.index(pc)) % 7]
            name = _letter_name(letter, pc)
            if name is not None:
                return name
    return names[pc]


# quality suffix -> (canonical name, intervals)
_QUALITIES: dict[str, tuple[str, tuple[int, ...]]] = {
    "": ("", (0, 4, 7)),
    "maj": ("", (0, 4, 7)),
    "m": ("m", (0, 3, 7)),
    "min": ("m", (0, 3, 7)),
    "dim": ("dim", (0, 3, 6)),
    "dim7": ("dim7", (0, 3, 6, 9)),
    "aug": ("aug", (0, 4, 8)),
    "5": ("5", (0, 7)),
    "6": ("6", (0, 4, 7, 9)),
    "m6": ("m6", (0, 3, 7, 9)),
    "min6": ("m6", (0, 3, 7, 9)),
    "7": ("7", (0, 4, 7, 10)),
    "9": ("9", (0, 4, 7, 10, 14)),
    "maj7": ("maj7", (0, 4, 7, 11)),
    "maj9": ("maj9", (0, 4, 7, 11, 14)),
    "m7": ("m7", (0, 3, 7, 10)),
    "min7": ("m7", (0, 3, 7, 10)),
    "m9": ("m9", (0, 3, 7, 10, 14)),
    "min9": ("m9", (0, 3, 7, 10, 14)),
    "m11": ("m11", (0, 3, 7, 10, 14, 17)),
    "mmaj7": ("mmaj7", (0, 3, 7, 11)),
    "m7b5": ("m7b5", (0, 3, 6, 10)),
    "sus2": ("sus2", (0, 2, 7)),
    "sus4": ("sus4", (0, 5, 7)),
    "7sus4": ("7sus4", (0, 5, 7, 10)),
    "add9": ("add9", (0, 4, 7, 14)),
    "madd9": ("madd9", (0, 3, 7, 14)),
}


@dataclass(frozen=True)
class Chord:
    root: int                    # pitch class 0-11
    quality: str                 # canonical suffix, "" = major triad
    intervals: tuple[int, ...]   # semitones from root
    sym: str                     # symbol as given / constructed


def parse_chord(sym: str) -> Chord:
    """'Cm9', 'Abmaj7', 'G7', 'Fsus2', 'Eb6', 'Dm7b5', 'C', ..."""
    s = sym.strip()
    m = _CHORD_RE.match(s)
    if not m:
        raise ValueError(f"invalid chord symbol: {sym!r}")
    letter, acc, suffix = m.groups()
    root = (_LETTER_PC[letter.lower()] + _ACC[acc]) % 12
    if suffix not in _QUALITIES:
        raise ValueError(f"unknown chord quality: {suffix!r} in {sym!r}")
    quality, intervals = _QUALITIES[suffix]
    return Chord(root=root, quality=quality, intervals=intervals, sym=s)


_ROMAN_DEGREES = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7}
_MAJOR_OFFSETS = (0, 2, 4, 5, 7, 9, 11)

# (suffix, is_major) -> canonical quality
_ROMAN_QUALITY = {
    ("", True): "",
    ("", False): "m",
    ("7", True): "7",       # dominant seventh
    ("7", False): "m7",
    ("9", True): "9",
    ("9", False): "m9",
    ("11", False): "m11",
    ("maj7", True): "maj7",
    ("maj7", False): "mmaj7",
    ("maj9", True): "maj9",
    ("6", True): "6",
    ("6", False): "m6",
    ("dim", False): "dim",
    ("dim", True): "dim",
    ("dim7", False): "dim7",
    ("dim7", True): "dim7",
    ("m7b5", False): "m7b5",
    ("m7b5", True): "m7b5",
    ("7b5", False): "m7b5",      # "ii7b5" = half-diminished
    ("7b5", True): "m7b5",
    ("sus2", True): "sus2",
    ("sus4", True): "sus4",
}


def roman_to_chord(numeral: str, key: Key) -> Chord:
    """Roman numeral in a key. Case = quality; supports 7/9/maj7 suffixes;
    borrowed bVII/iv handled; in minor, 'V' = major dominant (harmonic minor)."""
    s = numeral.strip()
    if not s:
        raise ValueError(f"invalid roman numeral: {numeral!r}")
    flat = False
    if s[0] == "b":
        flat = True
        s = s[1:]
    i = 0
    while i < len(s) and s[i] in "iIvV":
        i += 1
    roman_part, suffix = s[:i], s[i:]
    if not roman_part or roman_part.lower() not in _ROMAN_DEGREES:
        raise ValueError(f"invalid roman numeral: {numeral!r}")
    if not (roman_part.isupper() or roman_part.islower()):
        raise ValueError(f"mixed-case roman numeral: {numeral!r}")
    degree = _ROMAN_DEGREES[roman_part.lower()]
    is_major = roman_part.isupper()

    if flat:
        # borrowed flat degree: relative to the major scale, lowered a semitone
        offset = _MAJOR_OFFSETS[degree - 1] - 1
    else:
        # root from the key's own scale (natural minor for minor keys; the
        # harmonic-minor raised 7th affects chord *quality* of V, not its root)
        offset = _MODES[key.mode][degree - 1]
    root = (key.tonic + offset) % 12

    qkey = (suffix, is_major)
    if qkey not in _ROMAN_QUALITY:
        raise ValueError(f"unsupported roman suffix: {suffix!r} in {numeral!r}")
    quality = _ROMAN_QUALITY[qkey]
    _, intervals = _QUALITIES[quality if quality else ""]
    sym = spell(root, key) + quality
    return Chord(root=root, quality=quality, intervals=intervals, sym=sym)


def _nearest_with_pc(pc: int, target: int) -> int:
    """Midi note with pitch class pc nearest to target (ties resolve upward)."""
    return target + ((pc - target + 6) % 12) - 6


def voice_chord(chord: Chord, *, low_midi: int = 48, spread: bool = False,
                prev: list[int] | None = None) -> list[int]:
    """Return midi notes for a chord.

    spread=True: root in octave 2, upper structure an octave-plus above (lofi rule).
    prev given: minimal-motion voice leading (each tone placed nearest the
    previous voicing, common tones held).
    """
    if prev:
        notes: list[int] = []
        used: set[int] = set()
        for iv in chord.intervals:
            pc = (chord.root + iv) % 12
            best: int | None = None
            best_d = 10 ** 9
            for p in prev:
                cand = _nearest_with_pc(pc, p)
                d = abs(cand - p)
                if d < best_d:
                    best, best_d = cand, d
            assert best is not None
            while best < low_midi - 12:
                best += 12
            while best in used:
                best += 12
            used.add(best)
            notes.append(best)
        return sorted(notes)

    root_midi = low_midi + ((chord.root - low_midi) % 12)
    if spread:
        root_midi = 36 + ((chord.root - 36) % 12)  # octave 2
        upper_base = root_midi + 12
        notes = [root_midi] + [upper_base + iv for iv in chord.intervals[1:]]
        return notes
    return [root_midi + iv for iv in chord.intervals]
