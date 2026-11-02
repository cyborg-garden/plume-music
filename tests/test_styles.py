"""Tests for the style data files + loader (SPEC section 5)."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from plume import notation  # noqa: E402
from plume.styles_lib import list_styles, load_style  # noqa: E402
from plume.theory import parse_key, roman_to_chord  # noqa: E402
from plume.voices import PRESETS  # noqa: E402

STYLE_NAMES = ["lofi", "synthwave", "trap"]
BASS_STYLES = {"roots_eighths", "sub_hits", "pump_eighths"}
PAD_STYLES = {"held", "stabs", "arp16"}


def test_list_styles_contains_the_three():
    assert set(STYLE_NAMES) <= set(list_styles())


def test_load_unknown_style_raises():
    with pytest.raises(ValueError):
        load_style("bebop")


def test_load_style_alias_lofi():
    assert load_style("lo-fi")["name"] == "lofi"
    assert load_style("Lo Fi")["name"] == "lofi"


@pytest.mark.parametrize("name", STYLE_NAMES)
def test_style_schema(name):
    s = load_style(name)
    assert s["name"] == name
    lo, hi = s["bpm_range"]
    assert 40 <= lo < hi <= 220
    assert s["default_mode"] in ("major", "minor")
    assert 0.5 <= s["swing"] <= 0.66
    assert isinstance(s["timing_bias_ms"], dict)
    assert s["bass_style"] in BASS_STYLES
    assert s["pad_style"] in PAD_STYLES
    m = s["melody"]
    assert 0.0 < m["density"] <= 1.0
    assert str(m["rhythm_grid"]) in ("8", "16")
    assert set(s["master"]) <= {"reverb", "drive", "lofi", "limit_db"}


@pytest.mark.parametrize("name", STYLE_NAMES)
def test_style_progressions_parse(name):
    """v0.1: progressions are mode-keyed pools {"major": [...], "minor": [...]}."""
    s = load_style(name)
    pools = s["progressions"]
    assert set(pools) <= {"major", "minor"} and pools
    for mode, progs in pools.items():
        key = parse_key(f"C {mode}")
        assert progs, f"{name} has an empty {mode} pool"
        for prog in progs:
            assert prog["weight"] > 0
            assert len(prog["numerals"]) >= 2
            for numeral in prog["numerals"]:
                roman_to_chord(numeral, key)  # must not raise


def _minor_diatonic_pcs(tonic: int) -> set[int]:
    from plume.theory import Key
    natural = set(Key(tonic, "minor").scale())
    return natural | {(tonic + 11) % 12}     # harmonic-minor leading tone


@pytest.mark.parametrize("name", STYLE_NAMES)
def test_style_minor_pools_are_diatonic_minor(name):
    """Every minor-pool progression: diatonic to natural minor (the raised
    7th only in the dominant chord), tonic chords minor-quality."""
    from plume.theory import Key
    key = Key(tonic=5, mode="minor")         # F minor
    allowed = _minor_diatonic_pcs(key.tonic)
    natural = set(key.scale())
    leading = (key.tonic + 11) % 12
    dominant = (key.tonic + 7) % 12
    pools = load_style(name)["progressions"]
    assert "minor" in pools, f"{name} has no minor pool"
    for prog in pools["minor"]:
        for numeral in prog["numerals"]:
            ch = roman_to_chord(numeral, key)
            pcs = {(ch.root + iv) % 12 for iv in ch.intervals}
            assert pcs <= allowed, (
                f"{name} minor pool: {numeral} -> {ch.sym} not diatonic")
            if leading in pcs and leading not in natural:
                assert ch.root == dominant, (
                    f"{name}: raised 7th outside the dominant in {numeral}")
            if ch.root == key.tonic:
                assert 3 in ch.intervals and 4 not in ch.intervals, (
                    f"{name}: tonic chord {numeral} -> {ch.sym} is not minor")


@pytest.mark.parametrize("name", STYLE_NAMES)
def test_style_drum_patterns_parse(name):
    s = load_style(name)
    for role, variants in s["drum_patterns"].items():
        assert variants, f"{name}.{role} has no variants"
        for pat in variants:
            steps = notation.parse(pat)
            assert steps, f"{name}.{role} pattern {pat!r} has no hits"
    for role, fills in s["fills"].items():
        for pat in fills:
            assert notation.parse(pat)


@pytest.mark.parametrize("name", STYLE_NAMES)
def test_style_voices_reference_real_presets(name):
    s = load_style(name)
    for role, preset in s["voices"].items():
        assert preset in PRESETS, f"{name}.{role} -> unknown preset {preset!r}"


@pytest.mark.parametrize("name", STYLE_NAMES)
def test_style_fx_defaults_keys(name):
    allowed = {"lpf_hz", "hpf_hz", "drive", "chorus", "delay", "reverb", "duck"}
    s = load_style(name)
    for role, fx in s["fx_defaults"].items():
        assert set(fx) <= allowed


@pytest.mark.parametrize("name", STYLE_NAMES)
def test_style_arrangement_templates(name):
    s = load_style(name)
    templates = s["arrangement_templates"]
    assert "3" in templates
    for count, movements in templates.items():
        assert len(movements) == int(count)
        for mv in movements:
            assert mv["bars"] % 8 == 0
            assert mv["bars"] // len(mv["energy_curve"]) % 8 == 0
            for e in mv["energy_curve"]:
                assert 0.0 <= e <= 1.0


# -- genre content requirements (SPEC section 5 + research data) ------------

def test_trap_snare_on_beat_three_halftime():
    s = load_style("trap")
    for pat in s["drum_patterns"]["snare"]:
        starts = [st.start for st in notation.parse(pat)]
        assert 0.5 in starts            # step 9 of 16 = beat 3
        assert 0.25 not in starts       # no backbeat on beat 2 (half-time)


def test_trap_hat_fill_lexicon_has_triplets_and_32nd_runs():
    fills = load_style("trap")["fills"]["hat"]
    assert any("*3" in f for f in fills), "need *3 triplet bursts"
    assert any("[hh*8]" in f for f in fills), "need [hh*8] 1/32 runs"


def test_trap_is_straight_time():
    assert load_style("trap")["swing"] == 0.5


def test_lofi_drunk_feel_numbers():
    s = load_style("lofi")
    assert 0.54 <= s["swing"] <= 0.62
    assert s["timing_bias_ms"]["kick"] < 0     # kick pushed early
    assert s["timing_bias_ms"]["snare"] > 0    # snare dragged late


def test_lofi_progressions_have_sevenths():
    s = load_style("lofi")
    for pool in s["progressions"].values():
        for prog in pool:
            assert all(("7" in n) or ("9" in n) for n in prog["numerals"])


def test_trap_hat_width_and_gain():
    """v0.1 (#15): trap hats get chorus decorrelation + ~+3 dB gain."""
    s = load_style("trap")
    assert s["fx_defaults"]["hat"].get("chorus", 0) > 0
    assert s.get("gain_db", {}).get("hat", -99.0) >= -5.0


def test_synthwave_four_on_floor():
    s = load_style("synthwave")
    pat = s["drum_patterns"]["kick"][0]
    starts = [st.start for st in notation.parse(pat)]
    for beat in (0.0, 0.25, 0.5, 0.75):
        assert beat in starts


def test_synthwave_pump_bass_and_arp():
    s = load_style("synthwave")
    assert s["bass_style"] == "pump_eighths"
    assert s["fx_defaults"]["bass"].get("duck", 0) > 0   # sidechain pump
    assert "arp" in s["voices"]                          # 1/16 arp layer
    assert s["fx_defaults"]["snare"].get("reverb", 0) >= 0.4  # gated-snare wash
