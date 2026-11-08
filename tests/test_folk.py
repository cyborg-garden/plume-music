"""Folk: a light acoustic band, and the guess that hears a strummed guitar.

The style half pins the bones folk was built with: plain diatonic triads with plagal
moves, root and fifth on an upright, open piano chords, brushes and a cajon.  It does
not say the band sounds good; that is for a person.  The detection half lives in
test_detect.py beside the rule it extends.
"""
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from test_genres import _band_roles, _bar, _engine, _note, _play  # noqa: E402

from plume import figures, guitake, samplekit, voices  # noqa: E402
from plume import mind as mind_mod  # noqa: E402
from plume.compose import PromptSpec, compose, parse_prompt  # noqa: E402
from plume.governor import progression_symbols  # noqa: E402
from plume.styles_lib import list_styles, load_style  # noqa: E402
from plume.theory import parse_chord, parse_key, roman_to_chord  # noqa: E402

SR = 48000


# ------------------------------------------------------------------ names

@pytest.mark.parametrize("text", ["folk", "a folk song", "folky ballad in G major",
                                  "Folk at 90", "slow folk in D major"])
def test_a_prompt_can_ask_for_folk(text):
    assert parse_prompt(text).styles == ["folk"]


@pytest.mark.parametrize("text,style", [
    ("folky jazz", "jazz"), ("folky house", "house"), ("jazz folk", "folk"),
])
def test_folky_only_colours_another_genre(text, style):
    assert parse_prompt(text).styles == [style]


def test_a_prompt_naming_an_older_style_keeps_it():
    assert parse_prompt("lofi folk").styles == ["lofi"]


@pytest.mark.parametrize("cue", ["make it folk", "go folk", "more folky"])
def test_a_cue_can_ask_for_folk(cue):
    parsed = mind_mod.parse_cue(cue)
    assert (parsed["kind"], parsed["value"]) == ("style", "folk")


def test_the_page_and_the_cli_list_it():
    assert "folk" in list_styles()
    assert guitake.canonical_style("folk") == "folk"
    assert guitake.canonical_style("Folky") == "folk"


def test_the_guides_name_it():
    root = pathlib.Path(__file__).resolve().parents[1]
    for page in ("src/plume/guide/playing.md", "README.md", "docs/PLAYING.md"):
        assert "`folk`" in (root / page).read_text(), page


# ---------------------------------------------------------------- harmony

def test_folk_harmony_is_plain_diatonic_triads():
    s = load_style("folk")
    for mode, pool in s["progressions"].items():
        key = parse_key(f"G {mode}")
        for prog in pool:
            for numeral in prog["numerals"]:
                assert len(roman_to_chord(numeral, key).intervals) == 3, numeral


def test_folk_moves_plagally_from_four_to_one():
    major = [p["numerals"] for p in load_style("folk")["progressions"]["major"]]
    def plagal(ns):
        return any(a == "IV" and b == "I" for a, b in zip(ns, ns[1:] + ns[:1]))
    assert sum(plagal(ns) for ns in major) >= 3
    assert all(set(ns) <= {"I", "ii", "IV", "V", "vi"} for ns in major)


@pytest.mark.parametrize("sym,want", [("G", [55, 62, 71]), ("C", [48, 55, 64]),
                                      ("Em", [52, 59, 67]), ("D7", [50, 57, 60, 66])])
def test_an_open_voicing_is_root_fifth_and_the_third_up_an_octave(sym, want):
    assert figures.open_voicing(parse_chord(sym)) == want


def test_folk_piano_plays_open_chords_with_their_root():
    events, _, _ = _play("folk", bars=4, roles={"pad": 1.0})
    symbols = progression_symbols("folk", "C major")
    for b, bar in enumerate(events):
        pad = sorted({e["midi"] for e in bar if e["role"] == "pad"})
        assert pad, b
        root = parse_chord(symbols[b % len(symbols)]).root
        assert pad[0] % 12 == root
        assert pad[-1] - pad[0] > 12            # open: wider than an octave


def test_folk_bass_is_root_on_one_and_fifth_on_three():
    events, _, bpm = _play("folk", bars=4, roles={"bass": 1.0})
    symbols = progression_symbols("folk", "C major")
    for b, bar in enumerate(events):
        notes = sorted((e["onset_s"] * bpm / 60.0, e["midi"]) for e in bar
                       if e["role"] == "bass")
        assert len(notes) == 2, notes
        (b1, m1), (b3, m3) = notes
        assert b1 == pytest.approx(0.0, abs=0.12) and b3 == pytest.approx(2.0, abs=0.12)
        ch = parse_chord(symbols[b % len(symbols)])
        assert round(m1) % 12 == ch.root
        assert (round(m3) - round(m1)) % 12 == 7


# ----------------------------------------------------------------- drums

def test_every_folk_voice_is_recorded():
    recorded = set(samplekit.voice_names()) | set(samplekit._tuned_manifest())
    assert set(load_style("folk")["voices"].values()) <= recorded


def test_the_brush_backbeat_is_on_two_and_four():
    events, _, bpm = _play("folk", bars=4, roles={"snare": 1.0})
    for bar in events:
        loud = [e["onset_s"] * bpm / 60.0 for e in bar
                if e["role"] == "snare" and e["velocity"] >= 0.35]
        assert sorted(round(b) for b in loud) == [1, 3]


def test_the_band_is_soft_drums_under_the_strings():
    """Folk's drums sit under the bass and piano, never on top of them."""
    engine = _engine("folk")
    loud = {}
    for role in ("kick", "snare", "bass", "pad"):
        plan, harm = _bar("folk", {role: 1.0}, 1)
        y = engine.render_bar(plan, SimpleNamespace(bpm=96.0), harm)
        loud[role] = float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)))
    assert max(loud["kick"], loud["snare"]) < min(loud["bass"], loud["pad"])


@pytest.mark.parametrize("preset", ["cajon_bass", "cajon_slap"])
def test_a_cajon_hit_is_recorded_starts_at_once_and_ends_quietly(preset):
    assert preset in samplekit.voice_names() and preset in voices.PRESETS
    y = np.abs(_note(preset, vel=0.8).astype(np.float64))
    assert np.isfinite(y).all() and y.max() > 0.1
    assert int(np.argmax(y > 0.5 * y.max())) < int(0.004 * SR)
    assert y[-200:].max() < 0.01 * y.max()


def test_the_cajon_bass_is_the_low_one():
    def centroid(preset):
        y = _note(preset, vel=0.8).astype(np.float64)[: int(0.2 * SR)]
        s = np.abs(np.fft.rfft(y))
        f = np.fft.rfftfreq(y.size, 1 / SR)
        return float((s * f).sum() / s.sum())
    assert centroid("cajon_bass") < centroid("cajon_slap")


def test_a_composed_folk_session_sits_in_its_range_and_its_key():
    s = compose(PromptSpec(styles=["folk"], key="G major", movements=1, mood=[],
                           name="t"), seed=3)
    lo, hi = load_style("folk")["bpm_range"]
    assert lo <= s.bpm <= hi
    assert s.key == "G major"
    assert _band_roles(_engine("folk")) == {"kick": 1.0, "snare": 1.0, "hat": 1.0,
                                             "bass": 1.0, "pad": 1.0, "lead": 1.0}
