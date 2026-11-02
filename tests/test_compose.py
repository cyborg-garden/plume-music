"""Tests for plume.compose (SPEC section 6)."""
import dataclasses
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("plume.score", reason="score.py (builder D) not present yet")

from plume import notation  # noqa: E402
from plume.compose import PromptSpec, compose, make_motif, parse_prompt  # noqa: E402
from plume.score import Session, total_bars  # noqa: E402
from plume.theory import note_to_midi, parse_chord, parse_key  # noqa: E402
from plume.voices import PRESETS  # noqa: E402

FLAGSHIP = "chill trap beat over lofi ambient synthwave at 120 bpm and 3 major movements"


# -- parse_prompt ------------------------------------------------------------

def test_parse_prompt_flagship():
    spec = parse_prompt(FLAGSHIP)
    assert spec.styles == ["trap", "lofi", "synthwave"]
    assert spec.bpm == 120.0
    assert spec.movements == 3
    assert "chill" in spec.mood and "ambient" in spec.mood
    assert spec.key is None
    assert spec.name


def test_parse_prompt_key_and_movements():
    spec = parse_prompt("lofi in f# minor, two movements at 85 bpm")
    assert spec.styles == ["lofi"]
    assert spec.bpm == 85.0
    assert spec.movements == 2
    key = parse_key(spec.key)
    assert key.tonic == 6 and key.mode == "minor"


def test_parse_prompt_mood_only():
    spec = parse_prompt("dark eerie synthwave")
    assert spec.styles == ["synthwave"]
    assert "dark" in spec.mood and "eerie" in spec.mood
    assert spec.bpm is None
    assert spec.movements == 3  # default


def test_parse_prompt_free_text_never_errors():
    spec = parse_prompt("make me something nice please !!")
    assert spec.styles == []
    assert spec.bpm is None and spec.key is None
    spec = parse_prompt("")
    assert spec.name == "untitled"


def test_parse_prompt_lofi_spellings():
    assert parse_prompt("lo-fi beats").styles == ["lofi"]
    assert parse_prompt("some lo fi").styles == ["lofi"]
    assert parse_prompt("retrowave drive").styles == ["synthwave"]


# -- compose: validity -------------------------------------------------------

def _validate(session: Session) -> None:
    assert isinstance(session, Session)
    parse_key(session.key)
    assert 0.5 <= session.swing <= 0.66
    track_names = set()
    for t in session.tracks:
        assert t.voice in PRESETS, f"track {t.name} -> unknown preset {t.voice}"
        assert t.kind in ("drum", "tonal")
        assert t.patterns, f"track {t.name} has no patterns"
        for pname, pat in t.patterns.items():
            steps = notation.parse(pat)
            assert steps, f"{t.name}.{pname} = {pat!r} parsed to no steps"
        track_names.add(t.name)
    assert session.movements
    for mv in session.movements:
        assert mv.sections
        for sec in mv.sections:
            assert sec.bars % 8 == 0 and sec.bars > 0
            assert 0.0 <= sec.energy <= 1.0
            for tname, pname in sec.plays.items():
                assert tname in track_names, f"section plays unknown track {tname}"
                if pname:
                    track = next(t for t in session.tracks if t.name == tname)
                    assert pname in track.patterns, (
                        f"section {sec.name} plays {tname}.{pname} which doesn't exist")
            for sym in sec.chords:
                parse_chord(sym)
    assert total_bars(session) % 8 == 0


def test_compose_flagship_validates():
    spec = parse_prompt(FLAGSHIP)
    session = compose(spec, seed=42)
    _validate(session)
    assert session.bpm == 120.0
    assert session.style == "trap"
    assert session.seed == 42
    assert len(session.movements) == 3


def test_compose_flagship_blends_secondary_layers():
    session = compose(parse_prompt(FLAGSHIP), seed=42)
    names = {t.name for t in session.tracks}
    assert "arp" in names       # synthwave signature layer
    assert "crackle" in names   # lofi signature layer
    assert session.master["lofi"] > 0  # lofi master hint merged


def test_compose_each_style_alone_validates():
    for style in ("trap", "lofi", "synthwave"):
        session = compose(parse_prompt(f"a {style} tune"), seed=7)
        _validate(session)
        assert session.style == style
        s_lo, s_hi = {"trap": (130, 150), "lofi": (70, 90),
                      "synthwave": (80, 118)}[style]
        assert s_lo <= session.bpm <= s_hi


def test_compose_respects_explicit_key():
    session = compose(parse_prompt("trap in f minor at 140 bpm"), seed=3)
    key = parse_key(session.key)
    assert key.tonic == 5 and key.mode == "minor"


def test_compose_energy_gates_layers():
    session = compose(parse_prompt("synthwave, 3 movements"), seed=11)
    low = [s for mv in session.movements for s in mv.sections if s.energy < 0.3]
    high = [s for mv in session.movements for s in mv.sections if s.energy > 0.6]
    assert high, "expected at least one full-energy section"
    for sec in high:
        assert "snare" in sec.plays and "lead" in sec.plays
    for sec in low:
        assert "lead" not in sec.plays or sec.plays.get("lead", "") == ""


def test_compose_outro_thins_out():
    session = compose(parse_prompt(FLAGSHIP), seed=42)
    outro = session.movements[-1].sections[-1]
    assert outro.energy <= 0.25
    assert outro.fill is False
    assert set(outro.plays) <= {"pad", "crackle"}


def test_compose_default_movements_is_three():
    session = compose(parse_prompt("lofi"), seed=1)
    assert len(session.movements) == 3


# -- compose: minor-mode harmony (v0.1 #5) -----------------------------------

def _first_chords(session: Session):
    for mv in session.movements:
        for sec in mv.sections:
            if sec.chords:
                return [parse_chord(sym) for sym in sec.chords]
    return []


@pytest.mark.parametrize("style", ("lofi", "trap", "synthwave"))
@pytest.mark.parametrize("seed", (1, 2, 3, 4, 5))
def test_minor_key_session_is_diatonic_minor(style, seed):
    """HARD REQUIREMENT: a minor-key session's progression is diatonic to
    that minor key (harmonic-minor V allowed), the tonic chord is
    minor-quality on the tonic root, and bass roots match the chords."""
    from plume.theory import Key
    spec = PromptSpec(styles=[style], bpm=None, key="F minor",
                      movements=1, mood=[], name="minortest")
    session = compose(spec, seed=seed)
    key = parse_key(session.key)
    assert key.tonic == 5 and key.mode == "minor"

    natural = set(Key(key.tonic, "minor").scale())
    leading = (key.tonic + 11) % 12
    dominant = (key.tonic + 7) % 12
    chords = _first_chords(session)
    assert chords, "session has no progression"
    for ch in chords:
        pcs = {(ch.root + iv) % 12 for iv in ch.intervals}
        assert pcs <= natural | {leading}, (
            f"{style} seed {seed}: {ch.sym} not diatonic to F minor")
        if leading in pcs:
            assert ch.root == dominant, (
                f"{style} seed {seed}: raised 7th outside the dominant ({ch.sym})")
    tonic_chords = [ch for ch in chords if ch.root == key.tonic]
    assert tonic_chords, f"{style} seed {seed}: no tonic chord in progression"
    for ch in tonic_chords:
        assert 3 in ch.intervals and 4 not in ch.intervals, (
            f"{style} seed {seed}: tonic chord {ch.sym} is not minor-quality")

    # bass roots follow the chord roots (check the roots-only sparse pattern)
    bass = next((t for t in session.tracks if t.name == "bass"), None)
    assert bass is not None
    root_pcs = {ch.root for ch in chords}
    for st in notation.parse(bass.patterns["sparse"]):
        for tok in (st.cycle_alts or (st.token,)):
            if tok == "~":
                continue
            midi = note_to_midi(tok.split(":")[0])
            assert midi % 12 in root_pcs, (
                f"{style} seed {seed}: bass note {tok} is not a chord root")


def test_major_key_session_is_diatonic_major():
    """Major keys keep major pools: tonic chord is major-quality."""
    for style in ("lofi", "trap", "synthwave"):
        spec = PromptSpec(styles=[style], bpm=None, key="C major",
                          movements=1, mood=[], name="majortest")
        session = compose(spec, seed=4)
        chords = _first_chords(session)
        assert chords
        tonic_chords = [ch for ch in chords if ch.root == 0]
        for ch in tonic_chords:
            assert 4 in ch.intervals and 3 not in ch.intervals


# -- compose: determinism ----------------------------------------------------

def test_compose_deterministic_same_seed():
    spec = parse_prompt(FLAGSHIP)
    a = compose(spec, seed=42)
    b = compose(spec, seed=42)
    assert dataclasses.asdict(a) == dataclasses.asdict(b)


def test_compose_default_seed_is_stable():
    spec = parse_prompt(FLAGSHIP)
    a = compose(spec)
    b = compose(spec)
    assert a.seed == b.seed
    assert dataclasses.asdict(a) == dataclasses.asdict(b)


def test_compose_different_seed_differs():
    spec = parse_prompt(FLAGSHIP)
    a = compose(spec, seed=1)
    b = compose(spec, seed=2)
    assert dataclasses.asdict(a) != dataclasses.asdict(b)


def test_compose_output_is_json_serializable():
    import json
    session = compose(parse_prompt(FLAGSHIP), seed=42)
    json.dumps(dataclasses.asdict(session))  # no numpy scalars anywhere


# -- make_motif ---------------------------------------------------------------

def test_make_motif_shape_and_constraints():
    import numpy as np
    from plume.theory import Key, roman_to_chord
    key = Key(tonic=5, mode="minor")
    chords = [roman_to_chord(n, key) for n in ("i", "VI", "III", "VII")]
    rng = np.random.default_rng(5)
    bars = make_motif(key, chords, {"density": 0.5, "octave": 5,
                                    "rhythm_grid": "8"}, rng)
    assert len(bars) == len(chords)
    rhythms = set()
    scale_pcs = set(key.scale())
    for bar_i, bar in enumerate(bars):
        toks = bar.split()
        assert len(toks) == 8
        rhythms.add(tuple(t != "~" for t in toks))
        pitches = []
        for i, t in enumerate(toks):
            if t == "~":
                continue
            name = t.split(":")[0]
            midi = note_to_midi(name)
            pitches.append(midi)
            chord = chords[bar_i]
            chord_pcs = {(chord.root + iv) % 12 for iv in chord.intervals}
            if i in (0, 4):  # strong beats -> chord tones
                assert midi % 12 in chord_pcs
        assert pitches
        assert max(pitches) - min(pitches) <= 18  # 1.5 octaves
    assert len(rhythms) == 1  # transposition preserves the motif rhythm


def test_make_motif_deterministic():
    import numpy as np
    from plume.theory import Key, roman_to_chord
    key = Key(tonic=0, mode="minor")
    chords = [roman_to_chord(n, key) for n in ("i", "VI")]
    cfg = {"density": 0.4, "octave": 5, "rhythm_grid": "16"}
    a = make_motif(key, chords, cfg, np.random.default_rng(9))
    b = make_motif(key, chords, cfg, np.random.default_rng(9))
    assert a == b


# -- flat-key chord spelling -------------------------------------------------

@pytest.mark.parametrize("seed", (1, 2, 3))
def test_flat_key_progression_spelled_with_flats(seed):
    """--key Fm must yield Bbm7/Eb7/Abmaj7-style symbols, never A#m7/D#7."""
    spec = PromptSpec(styles=["lofi"], bpm=None, key="F minor",
                      movements=1, mood=[], name="flatspell")
    session = compose(spec, seed=seed)
    syms = [sym for mv in session.movements for sec in mv.sections
            for sym in sec.chords]
    assert syms
    assert not any("#" in sym for sym in syms), syms
    for sym in syms:
        parse_chord(sym)  # still valid chord symbols
