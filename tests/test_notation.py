"""Tests for plume.notation (SPEC section 2)."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from plume.notation import NotationError, Step, parse, steps_for_cycle  # noqa: E402


def starts(steps):
    return [pytest.approx(s.start) for s in steps]


def tokens(steps):
    return [s.token for s in steps]


# ---------------------------------------------------------------- basics

def test_simple_seq():
    steps = parse("bd sd hh sd")
    assert tokens(steps) == ["bd", "sd", "hh", "sd"]
    assert [s.start for s in steps] == pytest.approx([0.0, 0.25, 0.5, 0.75])
    assert all(s.dur == pytest.approx(0.25) for s in steps)


def test_single_atom_fills_bar():
    (s,) = parse("bd")
    assert (s.start, s.dur) == (0.0, 1.0)


def test_rests_omitted():
    steps = parse("bd ~ sd ~")
    assert tokens(steps) == ["bd", "sd"]
    assert [s.start for s in steps] == pytest.approx([0.0, 0.5])


def test_all_rests_empty():
    assert parse("~ ~ ~ ~") == []


def test_sorted_by_start():
    steps = parse("hh*4, bd ~ sd ~")
    assert [s.start for s in steps] == sorted(s.start for s in steps)


# ---------------------------------------------------------------- nesting

def test_group_subdivides_slot():
    steps = parse("[bd sd] hh")
    assert tokens(steps) == ["bd", "sd", "hh"]
    assert [s.start for s in steps] == pytest.approx([0.0, 0.25, 0.5])
    assert [s.dur for s in steps] == pytest.approx([0.25, 0.25, 0.5])


def test_deep_nesting():
    steps = parse("[bd [sd sd]] hh")
    assert [s.start for s in steps] == pytest.approx([0.0, 0.25, 0.375, 0.5])
    assert [s.dur for s in steps] == pytest.approx([0.25, 0.125, 0.125, 0.5])


def test_nested_group_with_rest():
    steps = parse("[~ bd] sd")
    assert tokens(steps) == ["bd", "sd"]
    assert [s.start for s in steps] == pytest.approx([0.25, 0.5])


# ---------------------------------------------------------------- * repeat

def test_star_repeats_within_slot():
    steps = parse("hh*4")
    assert len(steps) == 4
    assert [s.start for s in steps] == pytest.approx([0.0, 0.25, 0.5, 0.75])


def test_star_in_seq():
    steps = parse("bd hh*2")
    assert tokens(steps) == ["bd", "hh", "hh"]
    assert [s.start for s in steps] == pytest.approx([0.0, 0.5, 0.75])


def test_star_on_group():
    steps = parse("[bd sd]*2")
    assert tokens(steps) == ["bd", "sd", "bd", "sd"]
    assert [s.start for s in steps] == pytest.approx([0.0, 0.25, 0.5, 0.75])


def test_trap_hat_roll_shapes():
    # from the style content requirements
    assert len(parse("hh*8")) == 8
    steps = parse("[hh*4] [hh*6]")
    assert len(steps) == 10
    assert steps[4].start == pytest.approx(0.5)
    assert steps[4].dur == pytest.approx(0.5 / 6)


# ---------------------------------------------------------------- @ stretch

def test_at_stretches_elem():
    steps = parse("bd@3 sd")
    assert [s.start for s in steps] == pytest.approx([0.0, 0.75])
    assert [s.dur for s in steps] == pytest.approx([0.75, 0.25])


def test_at_on_rest():
    steps = parse("~@3 sd")
    assert tokens(steps) == ["sd"]
    assert steps[0].start == pytest.approx(0.75)


def test_star_then_at():
    steps = parse("hh*2@3 sd")
    assert tokens(steps) == ["hh", "hh", "sd"]
    assert [s.start for s in steps] == pytest.approx([0.0, 0.375, 0.75])


# ---------------------------------------------------------------- <> alternation

def test_alt_first_cycle_token():
    steps = parse("<bd sd> hh")
    assert tokens(steps) == ["bd", "hh"]
    assert steps[0].cycle_alts == ("bd", "sd")
    assert steps[1].cycle_alts is None


def test_steps_for_cycle_resolution():
    steps = parse("<bd sd cp> hh")
    for cycle, want in [(0, "bd"), (1, "sd"), (2, "cp"), (3, "bd")]:
        resolved = steps_for_cycle(steps, cycle)
        assert tokens(resolved) == [want, "hh"]


def test_alt_with_rest_drops_on_that_cycle():
    steps = parse("bd <hh ~>")
    assert tokens(steps_for_cycle(steps, 0)) == ["bd", "hh"]
    assert tokens(steps_for_cycle(steps, 1)) == ["bd"]


def test_steps_for_cycle_passthrough():
    steps = parse("bd sd")
    assert steps_for_cycle(steps, 5) == steps


# ---------------------------------------------------------------- layers

def test_layers_play_simultaneously():
    steps = parse("bd*4, hh*8")
    assert len(steps) == 12
    assert sum(1 for s in steps if s.token == "bd") == 4
    assert sum(1 for s in steps if s.token == "hh") == 8
    # both layers start at 0
    assert {s.token for s in steps if s.start == 0.0} == {"bd", "hh"}


def test_three_layers():
    steps = parse("bd ~ ~ ~, ~ ~ sd ~, hh hh hh hh")
    assert len(steps) == 6


# ---------------------------------------------------------------- velocity suffix

def test_velocity_suffix_kept_in_token():
    (s,) = parse("bd:v0.5")
    assert s.token == "bd:v0.5"
    assert (s.start, s.dur) == (0.0, 1.0)


def test_velocity_in_seq_and_alt():
    steps = parse("bd:v0.5 <hh:v0.8 sd>")
    assert steps[0].token == "bd:v0.5"
    assert steps[1].cycle_alts == ("hh:v0.8", "sd")


def test_note_and_chord_tokens():
    steps = parse("c1 eb2 Cm9 Abmaj7")
    assert tokens(steps) == ["c1", "eb2", "Cm9", "Abmaj7"]


# ---------------------------------------------------------------- errors

@pytest.mark.parametrize("bad", [
    "",
    "   ",
    "[bd",
    "bd]",
    "bd>",
    "<>",
    "<bd",
    "bd*",
    "bd*x",
    "bd*0",
    "bd@0",
    "*4",
    "bd:vx",
    "bd:v2.0",
    "bd,,sd",
    "[]",
    "<[a b] c>",
    "bd !",
])
def test_errors(bad):
    with pytest.raises(NotationError):
        parse(bad)


def test_error_has_position_info():
    with pytest.raises(NotationError) as ei:
        parse("bd ]")
    assert ei.value.pos == 3
    assert "position 3" in str(ei.value)


def test_error_unclosed_bracket_position():
    with pytest.raises(NotationError) as ei:
        parse("bd [sd hh")
    assert ei.value.pos == 3


def test_notation_error_is_value_error():
    assert issubclass(NotationError, ValueError)


# ---------------------------------------------------------------- Step dataclass

def test_step_frozen_and_defaults():
    s = Step(start=0.0, dur=0.5, token="bd")
    assert s.cycle_alts is None
    with pytest.raises(Exception):
        s.token = "sd"  # type: ignore[misc]
