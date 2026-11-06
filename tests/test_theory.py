"""Tests for plume.theory (SPEC section 1)."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from plume.theory import (  # noqa: E402
    NOTE_NAMES,
    Chord,
    Key,
    key_prefers_flats,
    midi_to_hz,
    note_to_midi,
    parse_chord,
    parse_key,
    roman_to_chord,
    spell,
    voice_chord,
)


# ---------------------------------------------------------------- notes

def test_note_names_constant():
    assert NOTE_NAMES == ["c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b"]


@pytest.mark.parametrize("name,midi", [
    ("c4", 60),
    ("eb2", 39),   # spec example: flats accepted
    ("ds2", 39),   # eb == ds
    ("a4", 69),
    ("cs4", 61),
    ("c#4", 61),
    ("b3", 59),
    ("bb2", 46),
    ("f#2", 42),
    ("c-1", 0),
    ("g9", 127),
])
def test_note_to_midi(name, midi):
    assert note_to_midi(name) == midi


@pytest.mark.parametrize("bad", ["", "h2", "c", "4c", "c4x", "x", "c##4"])
def test_note_to_midi_errors(bad):
    with pytest.raises(ValueError):
        note_to_midi(bad)


def test_midi_to_hz():
    assert midi_to_hz(69) == pytest.approx(440.0)
    assert midi_to_hz(57) == pytest.approx(220.0)
    assert midi_to_hz(60) == pytest.approx(261.6255653, abs=1e-4)
    # float midi for glides
    assert midi_to_hz(69.5) == pytest.approx(440.0 * 2 ** (0.5 / 12))


# ---------------------------------------------------------------- keys

@pytest.mark.parametrize("s,tonic,mode", [
    ("C min", 0, "minor"),
    ("f# minor", 6, "minor"),
    ("Eb maj", 3, "major"),
    ("a dorian", 9, "dorian"),
    ("F min", 5, "minor"),
    ("c harmonic minor", 0, "harmonic_minor"),
    ("e phrygian", 4, "phrygian"),
    ("Bb major", 10, "major"),
    # v0.1 (#6): glued single-token forms + bare tonic
    ("Fm", 5, "minor"),
    ("f#m", 6, "minor"),
    ("Ebmin", 3, "minor"),
    ("F", 5, "major"),
    ("f", 5, "major"),
    ("Gmaj", 7, "major"),
    ("Aminor", 9, "minor"),
    ("F min", 5, "minor"),
    ("bbm", 10, "minor"),
])
def test_parse_key(s, tonic, mode):
    k = parse_key(s)
    assert (k.tonic, k.mode) == (tonic, mode)


@pytest.mark.parametrize("bad", ["", "H min", "c lydian2", "xx major", "Fm minor"])
def test_parse_key_errors(bad):
    with pytest.raises(ValueError):
        parse_key(bad)


def test_parse_key_error_states_accepted_format():
    """v0.1 (#6): key errors must show the accepted formats."""
    with pytest.raises(ValueError, match="expected"):
        parse_key("Hmin")
    with pytest.raises(ValueError, match="minor"):
        parse_key("F lydian")


def test_scales():
    assert Key(0, "major").scale() == [0, 2, 4, 5, 7, 9, 11]
    assert Key(9, "minor").scale() == [9, 11, 0, 2, 4, 5, 7]
    assert Key(0, "harmonic_minor").scale() == [0, 2, 3, 5, 7, 8, 11]
    assert Key(2, "dorian").scale() == [2, 4, 5, 7, 9, 11, 0]
    assert Key(4, "phrygian").scale() == [4, 5, 7, 9, 11, 0, 2]


# ---------------------------------------------------------------- chords

@pytest.mark.parametrize("sym,root,intervals", [
    ("C", 0, (0, 4, 7)),
    ("Cm9", 0, (0, 3, 7, 10, 14)),
    ("Abmaj7", 8, (0, 4, 7, 11)),
    ("G7", 7, (0, 4, 7, 10)),
    ("Fsus2", 5, (0, 2, 7)),
    ("Fsus4", 5, (0, 5, 7)),
    ("Eb6", 3, (0, 4, 7, 9)),
    ("Dm7b5", 2, (0, 3, 6, 10)),
    ("Am7", 9, (0, 3, 7, 10)),
    ("F#m", 6, (0, 3, 7)),
    ("Bbm6", 10, (0, 3, 7, 9)),
    ("Cdim7", 0, (0, 3, 6, 9)),
    ("Gaug", 7, (0, 4, 8)),
])
def test_parse_chord(sym, root, intervals):
    ch = parse_chord(sym)
    assert ch.root == root
    assert ch.intervals == intervals
    assert ch.sym == sym


@pytest.mark.parametrize("bad", ["H7", "Cxyz", "", "7C"])
def test_parse_chord_errors(bad):
    with pytest.raises(ValueError):
        parse_chord(bad)


# ---------------------------------------------------------------- roman numerals

CMAJ = parse_key("C maj")
AMIN = parse_key("a min")
CMIN = parse_key("c min")


@pytest.mark.parametrize("numeral,key,root,intervals", [
    # major key basics
    ("I", CMAJ, 0, (0, 4, 7)),
    ("V7", CMAJ, 7, (0, 4, 7, 10)),
    ("ii7", CMAJ, 2, (0, 3, 7, 10)),
    ("Imaj7", CMAJ, 0, (0, 4, 7, 11)),
    ("IVmaj7", CMAJ, 5, (0, 4, 7, 11)),
    ("vi7", CMAJ, 9, (0, 3, 7, 10)),
    # borrowed in major
    ("bVII", CMAJ, 10, (0, 4, 7)),
    ("iv", CMAJ, 5, (0, 3, 7)),
    # minor key (natural minor degrees)
    ("i", AMIN, 9, (0, 3, 7)),
    ("VI", AMIN, 5, (0, 4, 7)),
    ("III", AMIN, 0, (0, 4, 7)),
    ("VII", AMIN, 7, (0, 4, 7)),
    ("iv", AMIN, 2, (0, 3, 7)),
    ("v", AMIN, 4, (0, 3, 7)),
    # trap staple in C minor: i VI III VII = Cm Ab Eb Bb
    ("i", CMIN, 0, (0, 3, 7)),
    ("VI", CMIN, 8, (0, 4, 7)),
    ("III", CMIN, 3, (0, 4, 7)),
    ("VII", CMIN, 10, (0, 4, 7)),
    ("bVII", CMIN, 10, (0, 4, 7)),
])
def test_roman_to_chord(numeral, key, root, intervals):
    ch = roman_to_chord(numeral, key)
    assert (ch.root, ch.intervals) == (root, intervals)


def test_minor_key_major_v_is_harmonic_minor_dominant():
    # In a minor: V = E major, contains G# (pc 8), the harmonic-minor leading tone
    ch = roman_to_chord("V", AMIN)
    assert ch.root == 4
    pcs = {(ch.root + iv) % 12 for iv in ch.intervals}
    assert 8 in pcs  # G#
    ch7 = roman_to_chord("V7", AMIN)
    assert ch7.intervals == (0, 4, 7, 10)
    assert (ch7.root + 4) % 12 == 8


@pytest.mark.parametrize("bad", ["", "VIII", "xI", "iV", "V13"])
def test_roman_errors(bad):
    with pytest.raises(ValueError):
        roman_to_chord(bad, CMAJ)


# ---------------------------------------------------------------- voicing

def test_voice_chord_default():
    notes = voice_chord(parse_chord("C"))
    assert notes == [48, 52, 55]
    assert min(notes) >= 48


def test_voice_chord_low_midi():
    notes = voice_chord(parse_chord("F"), low_midi=48)
    assert notes[0] % 12 == 5
    assert notes[0] >= 48


def test_voice_chord_spread():
    notes = voice_chord(parse_chord("Cm9"), spread=True)
    root = notes[0]
    assert 36 <= root <= 59            # octave 2-3
    assert root % 12 == 0
    assert notes[1] - root >= 12       # gap between root and upper structure
    assert len(notes) == 5


def test_voice_leading_minimal_motion():
    c = voice_chord(parse_chord("C"))
    f = voice_chord(parse_chord("F"), prev=c)
    assert len(f) == 3
    # common tone C held
    assert 48 in f
    # total motion is small (nearest inversion), much less than root position jump
    motion = sum(abs(a - b) for a, b in zip(sorted(c), sorted(f)))
    default_f = voice_chord(parse_chord("F"))
    default_motion = sum(abs(a - b) for a, b in zip(sorted(c), sorted(default_f)))
    assert motion < default_motion
    assert motion <= 6


def test_voice_leading_progression_continuity():
    key = parse_key("C maj")
    prev = None
    voicings = []
    for numeral in ["I", "vi", "ii", "V"]:
        ch = roman_to_chord(numeral, key)
        prev = voice_chord(ch, prev=prev)
        voicings.append(prev)
    # every consecutive pair: max per-voice movement stays small
    for a, b in zip(voicings, voicings[1:]):
        assert len(b) == 3
        per_voice = [abs(x - y) for x, y in zip(sorted(a), sorted(b))]
        assert max(per_voice) <= 7
    # voiced notes carry the right pitch classes
    for numeral, v in zip(["I", "vi", "ii", "V"], voicings):
        ch = roman_to_chord(numeral, key)
        want = {(ch.root + iv) % 12 for iv in ch.intervals}
        assert {n % 12 for n in v} == want


def test_voice_chord_prev_dedup_and_types():
    ch = parse_chord("C")
    v = voice_chord(ch, prev=[48, 52, 55])
    assert v == sorted(v)
    assert len(set(v)) == len(v)
    assert all(isinstance(n, int) for n in v)


def test_chord_dataclass_frozen():
    ch = parse_chord("C")
    with pytest.raises(Exception):
        ch.root = 5  # type: ignore[misc]
    assert isinstance(ch, Chord)


# ---------------------------------------------------------------- spelling

@pytest.mark.parametrize("key_str,flats", [
    # flat majors + C (borrowed black keys are flat degrees)
    ("C", True), ("F", True), ("Bb major", True), ("Eb maj", True),
    ("Ab major", True), ("Db major", True), ("Gb major", True),
    # sharp majors
    ("G", False), ("D", False), ("A", False), ("E", False), ("B", False),
    ("F# major", False),
    # flat minors + a
    ("a minor", True), ("d minor", True), ("g minor", True), ("c minor", True),
    ("Fm", True), ("Bbm", True), ("Ebmin", True),
    # sharp minors
    ("e minor", False), ("b minor", False), ("f#m", False),
    ("c# minor", False), ("g# minor", False),
])
def test_key_prefers_flats_table(key_str, flats):
    assert key_prefers_flats(parse_key(key_str)) is flats


def test_spell_uses_key_preferred_accidental():
    fm = parse_key("Fm")
    assert [spell(pc, fm) for pc in (10, 3, 8, 1)] == ["Bb", "Eb", "Ab", "Db"]
    e = parse_key("E major")
    assert (spell(6, e), spell(8, e)) == ("F#", "G#")
    assert spell(10) == "A#"  # no key: historical sharp default
    assert spell(5, fm) == "F"  # naturals unaffected


def test_roman_to_chord_spells_flats_in_flat_keys():
    fm = parse_key("Fm")
    assert roman_to_chord("i7", fm).sym == "Fm7"
    assert roman_to_chord("iv7", fm).sym == "Bbm7"
    assert roman_to_chord("VII", fm).sym == "Eb"
    assert roman_to_chord("IIImaj7", fm).sym == "Abmaj7"
    # C major: borrowed flat degrees spell flat
    assert roman_to_chord("bVII", parse_key("C")).sym == "Bb"
    # sharp keys keep sharps
    assert roman_to_chord("ii7", parse_key("E major")).sym == "F#m7"


def test_pitch_class_six_major_is_f_sharp_major_unless_typed_as_gb():
    """Guitarists call it F# major; a heard key has only its pitch class."""
    from plume.listen.harmony import _key_name
    from plume.roles import _key_name as roles_key_name
    assert _key_name(6) == "F# major"
    assert roles_key_name(6, "major") == "F# major"
    assert spell(6, Key(tonic=6, mode="major")) == "F#"
    assert spell(6, parse_key("Gb major")) == "Gb"   # typed flats are kept
    assert parse_key("Gb major") == parse_key("F# major")


@pytest.mark.parametrize("key_str,numerals,want", [
    ("F# major", "I ii iii IV V vi viidim bVII", "F# G#m A#m B C# D#m E#dim E"),
    ("Gb major", "I ii IV V", "Gb Abm Cb Db"),
    ("Eb minor", "i iv v VI VII", "Ebm Abm Bbm Cb Db"),
    ("G# minor", "i ii7b5 III iv v V VI", "G#m A#m7b5 B C#m D#m D# E"),
    ("C# minor", "i III VI VII", "C#m E A B"),
    ("Bb major", "I IV V vi bVII", "Bb Eb F Gm Ab"),
    ("A harmonic minor", "i iv V viidim", "Am Dm E G#dim"),
])
def test_chords_are_spelled_from_the_keys_own_letters(key_str, numerals, want):
    key = parse_key(key_str)
    got = " ".join(roman_to_chord(n, key).sym for n in numerals.split())
    assert got == want
    for sym in got.split():                 # every spelling still parses
        assert parse_chord(sym).root == roman_to_chord(
            numerals.split()[got.split().index(sym)], key).root


def test_chord_parsing_accepts_both_spellings():
    # internal token parsing stays spelling-agnostic
    assert parse_chord("Bbm7").root == parse_chord("A#m7").root == 10
    assert parse_chord("Eb7").root == parse_chord("D#7").root == 3
    assert note_to_midi("ab2") == note_to_midi("g#2")
