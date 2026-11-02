"""Form navigation + cues — SPEC-V2 sections 2 and 3.

Two halves, and the second one is the point.

**Synthetic half.**  Table-driven over hand-built ``Session`` maps and
scripted ``BarFeatures``: every rule in SPEC-V2 section 2 (wait at the
turnaround, energy chases the human, announce before a change, end together)
and section 3 (cue parsing, boundary queueing, ttl) gets a case with the
thresholds it is written against.

**Real-material half.**  v1 passed ten synthetic behavior tests and was
unusable on real music, so nothing here is trusted until it has run over
plume's own v0 renders — built in-test with ``compose`` + ``engine.render``,
in three styles, at two input gains 20 dB apart.  That half is not decoration:
it is what caught the ritardando detector declaring an ending on bar 2 of a
quiet synthwave take (the beat tracker locks at 199.5 bpm, commits an octave
re-interpretation to 99.9, and 199.5 -> 99.95 -> 99.87 is two strictly falling
tempo estimates — a "ritardando" of -50%).  The machine ended the tune and sat
there for the remaining 24 bars.  Both guards that fix it (octave-change
awareness, per-bar drop cap) are asserted below, at both gains.

Gain invariance is asserted exactly, too: the same real bar features offset by
a constant dB amount must produce a byte-identical plan sequence.  An
absolute-level rule cannot pass that test, which is precisely how v1's
silence rule shipped broken.

Self-contained: every input is synthesized in-test into ``tmp_path``.  No
reads from ``renders/`` (gitignored — a test that can silently not run is not
a test).  Deterministic: fixed seeds, no wall clock in any assertion except
the explicitly-labelled throughput measurement.
"""
from __future__ import annotations

import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plume import sim as sim_mod                                # noqa: E402
from plume.compose import PromptSpec, compose                   # noqa: E402
from plume.engine import render                                 # noqa: E402
from plume.listen.beat import BeatState, BeatTracker            # noqa: E402
from plume.listen.features import FeatureExtractor              # noqa: E402
from plume.listen.harmony import HarmonyState, HarmonyTracker   # noqa: E402
from plume.mind import (                                        # noqa: E402
    CUE_BOUNDARIES, CUE_KINDS, FORM_ACTIONS, FORM_PARAM_HELP, BarFeatures,
    Cue, FormParams, FormPlan, Mind, MindParams, flatten_sections, parse_cue,
)
from plume.score import Movement, SectionSpec, Session          # noqa: E402

SR = 48000


# ------------------------------------------------------------------ builders


def mk_beat(*, bar: int = 0, bpm: float = 120.0, conf: float = 0.9,
            locked: bool = True, octaves: int = 0) -> BeatState:
    interval = 60.0 / bpm if bpm > 0 else math.inf
    return BeatState(
        t_s=max(0, bar) * 4 * interval, bpm=bpm, confidence=conf, locked=locked,
        bar=bar, beat_in_bar=0, phase_beats=0.0,
        next_beat_t_s=interval, next_downbeat_t_s=4 * interval,
        beat_interval_s=interval, tempo_octave_changes=octaves)


HARM = HarmonyState(t_s=0.0, key="C minor", key_conf=0.8, chord="Cm",
                    chord_conf=0.7, register_low_midi=None,
                    register_high_midi=None)


def mk_bar(bar: int, *, rms: float = -20.0, dens: float = 1.0,
           silent: bool = False) -> BarFeatures:
    return BarFeatures(bar=bar, rms_db=rms, onset_density=dens, silent=silent,
                       register_low_midi=None, register_high_midi=None)


def mk_session(*specs: tuple[str, int, float], name: str = "map",
               movements: int = 1) -> Session:
    """``mk_session(("intro", 8, 0.3), ...)`` -> a v0 Session with one movement."""
    secs = [SectionSpec(name=n, bars=b, energy=e, plays={}) for n, b, e in specs]
    if movements <= 1:
        movs = [Movement(name="m0", sections=secs)]
    else:
        per = math.ceil(len(secs) / movements)
        movs = [Movement(name=f"m{i}", sections=secs[i * per:(i + 1) * per])
                for i in range(movements)]
        movs = [m for m in movs if m.sections]
    return Session(name=name, bpm=120.0, key="C minor", movements=movs)


def run_form(m: Mind, session: Session, steps: list[dict], *,
             role=None) -> list[FormPlan]:
    """Drive ``plan_form`` over scripted bars.

    Step keys: ``rms``, ``dens``, ``silent`` (BarFeatures); ``bpm``, ``conf``,
    ``locked``, ``octaves`` (BeatState); ``cue`` (a Cue); ``human=None`` for
    an unlocked tick with no completed human bar.
    """
    out = []
    for i, step in enumerate(steps):
        beat = mk_beat(bar=i, bpm=step.get("bpm", 120.0),
                       conf=step.get("conf", 0.9),
                       locked=step.get("locked", True),
                       octaves=step.get("octaves", 0))
        if step.get("human", "yes") is None:
            human = None
        else:
            human = mk_bar(i, rms=step.get("rms", -20.0),
                           dens=step.get("dens", 1.0),
                           silent=step.get("silent", False))
        out.append(m.plan_form(role, beat, HARM, human, session,
                               step.get("cue")))
    return out


def steady(n: int, **kw) -> list[dict]:
    return [dict(kw) for _ in range(n)]


# The maps.  Energies are chosen so the COMPOSED next section is not also the
# energy-jump target — otherwise "jump" and "advance" are indistinguishable.
MAP_UP = mk_session(("intro", 4, 0.30), ("verse", 4, 0.32), ("drop", 4, 0.90))
MAP_DOWN = mk_session(("hook", 4, 0.85), ("hook2", 4, 0.83), ("break", 4, 0.20))
MAP_ABC = mk_session(("intro", 4, 0.30), ("verse", 4, 0.50), ("chorus", 4, 0.85))
MAP_LONG = mk_session(("intro", 8, 0.30), ("verse", 8, 0.50),
                      ("chorus", 8, 0.85), ("break", 8, 0.20))


def announce_invariant(plans: list[FormPlan]) -> None:
    """Every MACHINE-initiated section change is preceded by an announce.

    Cue-driven changes are exempt by construction (SPEC-V2 section 2 rule 3
    says "machine-initiated"; a cue arrives with no notice — that is what
    makes it a cue).
    """
    for prev, cur in zip(plans, plans[1:]):
        if cur.section_idx == prev.section_idx or cur.action == "end":
            continue
        if "overridden by cue" in cur.reason:
            continue
        assert cur.action in ("advance", "jump"), cur
        assert prev.announce, f"unannounced change into {cur!r} after {prev!r}"
        assert prev.bars_left_in_section == 0, prev


# ================================================================== contract


def test_form_plan_contract_fields():
    """The frozen five come first, in order; the rest are defaulted extras."""
    names = [f.name for f in dataclasses.fields(FormPlan)]
    assert names[:5] == ["bar", "section_idx", "action",
                         "bars_left_in_section", "reason"]
    assert names[5:] == ["announce", "applied_cue", "dropped_cue"]
    extras = {f.name: f.default for f in dataclasses.fields(FormPlan)[5:]}
    assert extras == {"announce": False, "applied_cue": None,
                      "dropped_cue": None}
    # A spec-shaped construction (five positional args) still works.
    p = FormPlan(0, 0, "hold", 3, "why")
    assert (p.announce, p.applied_cue, p.dropped_cue) == (False, None, None)


def test_cue_contract_defaults():
    c = Cue("darker")
    assert (c.at, c.ttl_bars) == ("next-phrase", 16)
    assert [f.name for f in dataclasses.fields(Cue)] == ["text", "at",
                                                         "ttl_bars"]


def test_form_param_help_covers_every_field():
    names = {f.name for f in dataclasses.fields(FormParams)}
    assert set(FORM_PARAM_HELP) == names
    assert all(isinstance(v, str) and v for v in FORM_PARAM_HELP.values())
    assert FormParams().max_extend_bars == 16          # SPEC-V2 section 2


def test_mind_params_and_plan_bar_are_untouched():
    """v1's constructor and its defaults still work positionally."""
    m = Mind(MindParams())
    assert isinstance(m.form_params, FormParams)
    plan = m.plan_bar(mk_beat(bar=0, locked=False, conf=0.0), HARM, None)
    assert plan.state in ("waiting", "listening")
    assert Mind(MindParams(), FormParams(max_extend_bars=4)) \
        .form_params.max_extend_bars == 4


def test_flatten_reads_movements_as_one_address_space():
    s = mk_session(("a", 4, 0.1), ("b", 4, 0.2), ("c", 4, 0.3), ("d", 4, 0.4),
                   movements=2)
    flat = flatten_sections(s)
    assert [x.name for x in flat] == ["a", "b", "c", "d"]
    assert [x.idx for x in flat] == [0, 1, 2, 3]
    assert [x.movement for x in flat] == [0, 0, 1, 1]


def test_flatten_drops_unplayable_sections():
    s = mk_session(("a", 4, 0.1), ("zero", 0, 0.2), ("b", 4, 0.3))
    assert [x.name for x in flatten_sections(s)] == ["a", "b"]
    assert flatten_sections(None) == ()
    assert flatten_sections(Session(name="empty", bpm=120, key="C min")) == ()


def test_empty_session_never_raises():
    m = Mind(MindParams())
    empty = Session(name="empty", bpm=120.0, key="C minor")
    plans = run_form(m, empty, steady(6) + [dict(cue=Cue("go to the chorus"))])
    assert all(p.action == "hold" and p.section_idx == -1 for p in plans)
    assert all(p.action in FORM_ACTIONS for p in plans)


# ====================================================== rule 1: turnarounds


def test_plain_run_walks_the_composed_route_and_wraps():
    m = Mind(MindParams())
    plans = run_form(m, MAP_ABC, steady(13))
    assert [p.section_idx for p in plans] == [0] * 4 + [1] * 4 + [2] * 4 + [0]
    assert [p.action for p in plans] == (
        ["hold"] * 4 + ["advance"] + ["hold"] * 3 + ["advance"] + ["hold"] * 3
        + ["advance"])
    assert [p.bars_left_in_section for p in plans[:4]] == [3, 2, 1, 0]
    assert [p.announce for p in plans[:5]] == [False, False, False, True, False]
    announce_invariant(plans)


#: constant level, density rising into each turnaround: mid-phrase without
#: touching the energy rule (which is a LEVEL rule).
def _building_script(n: int, boundaries: set[int]) -> list[dict]:
    return [dict(rms=-20.0, dens=2.0 if i in boundaries else 1.0)
            for i in range(n)]


def test_mid_phrase_human_at_the_turnaround_extends_bounded():
    m = Mind(MindParams())
    # span 4 -> boundaries land at calls 4, 8, 12, 16, then 20 (span 20).
    plans = run_form(m, MAP_ABC, _building_script(21, {4, 8, 12, 16, 20}))
    extends = [i for i, p in enumerate(plans) if p.action == "extend"]
    assert extends == [4, 8, 12, 16]
    assert all(p.section_idx == 0 for p in plans[:20]), "never left the section"
    assert "mid-phrase at the turnaround" in plans[4].reason
    assert "16/16 used" in plans[16].reason
    # The budget is spent: the 5th turnaround advances instead, and says so.
    assert plans[20].action == "advance" and plans[20].section_idx == 1
    assert "budget is spent" in plans[20].reason
    # Total bars spent in section 0 = 4 composed + 16 extension.
    assert sum(1 for p in plans if p.section_idx == 0) == 20
    announce_invariant(plans)


def test_extension_unit_and_bound_are_parameters():
    """...and the budget is PER SECTION SPAN, not per song.

    "One more chorus" is a gesture you can make in the next section too;
    a song-long budget would mean the third turnaround of a long take can
    never be waited at.
    """
    m = Mind(MindParams(), FormParams(max_extend_bars=4, extend_unit_bars=2))
    plans = run_form(m, MAP_ABC, _building_script(13, {4, 6, 8, 10, 12}))
    assert [i for i, p in enumerate(plans) if p.action == "extend"] == [4, 6,
                                                                       12]
    assert plans[8].action == "advance"        # budget spent in section 0
    assert plans[12].section_idx == 1          # fresh budget in section 1


def test_a_resolved_human_is_not_extended_for():
    """Falling level + falling density at the boundary = they landed."""
    m = Mind(MindParams())
    script = [dict(rms=-20.0, dens=2.0) for _ in range(4)]
    script.append(dict(rms=-26.0, dens=0.5))       # resolution on the last bar
    script += steady(3, rms=-26.0, dens=0.5)
    plans = run_form(m, MAP_ABC, script)
    assert plans[4].action == "advance" and plans[4].section_idx == 1
    assert "extend" not in plans[4].reason


def test_a_silent_human_is_not_mid_phrase():
    m = Mind(MindParams())
    script = [dict(rms=-20.0, dens=1.0)] * 4 + [dict(silent=True, rms=-70.0)]
    plans = run_form(m, MAP_ABC, script)
    assert plans[4].action == "advance"


# ======================================================= rule 2: energy map


def _loud_script(n: int, quiet_bars: int, loud_db: float = -14.0,
                 quiet_db: float = -30.0) -> list[dict]:
    return [dict(rms=quiet_db if i < quiet_bars else loud_db) for i in range(n)]


def test_sustained_loud_human_jumps_to_a_higher_energy_section():
    m = Mind(MindParams())
    plans = run_form(m, MAP_UP, _loud_script(13, 6))
    jumps = [p for p in plans if p.action == "jump"]
    assert jumps, "sustained loud human never lifted the arrangement"
    j = jumps[0]
    assert j.section_idx == 2, "jumped to the hot section, not the next one"
    assert "above their baseline" in j.reason and "-> jump to section 2" in j.reason
    # It is a boundary decision, never mid-section.
    assert plans[plans.index(j) - 1].bars_left_in_section == 0
    announce_invariant(plans)


def test_sustained_quiet_human_drops_to_a_lower_energy_section():
    m = Mind(MindParams())
    script = [dict(rms=-12.0) for _ in range(6)] + [dict(rms=-30.0)] * 7
    plans = run_form(m, MAP_DOWN, script)
    jumps = [p for p in plans if p.action == "jump"]
    assert jumps and jumps[0].section_idx == 2
    assert "below their baseline" in jumps[0].reason
    announce_invariant(plans)


def test_steady_human_never_jumps():
    """The guard against a jittery energy rule (v1's on-grid case)."""
    m = Mind(MindParams())
    script = [dict(rms=-20.0 + (0.7 if i % 2 else -0.7)) for i in range(25)]
    plans = run_form(m, MAP_ABC, script)
    assert not [p for p in plans if p.action == "jump"]
    announce_invariant(plans)


def test_energy_rule_needs_sustained_evidence():
    """One loud bar is not evidence; ``energy_bars`` of them is."""
    m = Mind(MindParams())
    script = [dict(rms=-30.0)] * 3 + [dict(rms=-12.0)] + [dict(rms=-30.0)] * 9
    plans = run_form(m, MAP_UP, script)
    assert not [p for p in plans if p.action == "jump"]


def test_energy_jump_needs_a_target_and_says_so_when_there_is_none():
    flat_map = mk_session(("a", 4, 0.5), ("b", 4, 0.5), ("c", 4, 0.5))
    m = Mind(MindParams())
    plans = run_form(m, flat_map, _loud_script(13, 6))
    assert not [p for p in plans if p.action == "jump"]
    assert any("no section is hotter" in p.reason for p in plans)


def test_energy_is_relative_so_input_gain_cannot_change_the_decisions():
    """The v1 bug, as a property: shift every level, get the same plan.

    A rule written against an absolute dBFS floor cannot pass this.
    """
    script = _loud_script(21, 6)
    base = run_form(Mind(MindParams()), MAP_UP, script)
    for offset in (-40.0, -20.0, +6.0):
        shifted = [dict(s, rms=s["rms"] + offset) for s in script]
        assert run_form(Mind(MindParams()), MAP_UP, shifted) == base


def test_machine_initiates_nothing_below_form_authority_min():
    """Rule: with the form in the human's hands, the machine follows the map.

    Extensions still happen (deferring to the human is not an initiation)
    and cues still land (the third chair is explicit), but energy jumps do
    not — that is the machine claiming the form it does not have.
    """
    class Role:
        form_authority = 0.1
        clock_authority = 0.1
        role = "follow"

    m = Mind(MindParams())
    plans = run_form(m, MAP_UP, _loud_script(13, 6), role=Role())
    assert not [p for p in plans if p.action == "jump"]
    assert any("form_authority 0.10" in p.reason for p in plans)
    # ...and with authority, the same script does jump.
    m2 = Mind(MindParams())
    assert [p for p in run_form(m2, MAP_UP, _loud_script(13, 6))
            if p.action == "jump"]


def test_role_argument_is_optional_and_duck_typed():
    class Weird:
        form_authority = "not a number"

    for role in (None, Weird(), object()):
        plans = run_form(Mind(MindParams()), MAP_ABC, steady(9), role=role)
        assert [p.section_idx for p in plans][-1] == 2


# ====================================================== rule 3: announcing


def test_announce_marks_the_last_bar_of_every_section_span():
    m = Mind(MindParams())
    plans = run_form(m, MAP_LONG, steady(17))
    assert [i for i, p in enumerate(plans) if p.announce] == [7, 15]
    assert all(p.bars_left_in_section == 0 for p in plans if p.announce)


def test_extension_moves_the_announce_to_the_new_last_bar():
    m = Mind(MindParams())
    plans = run_form(m, MAP_ABC, _building_script(9, {4}))
    assert [i for i, p in enumerate(plans) if p.announce] == [3, 7]
    assert plans[4].action == "extend" and plans[4].announce is False


# ========================================================= rule 4: endings


def test_ritardando_overrides_the_form():
    m = Mind(MindParams())
    script = ([dict(bpm=120.0)] * 3
              + [dict(bpm=116.0), dict(bpm=112.0)] + [dict(bpm=108.0)] * 4)
    plans = run_form(m, MAP_ABC, script)
    assert plans[3].action != "end"
    assert plans[4].action == "end", "a real ritardando did not end the tune"
    assert "ritardando" in plans[4].reason
    assert "overrides the form" in plans[4].reason
    # Sticky: an ending is a commitment, and the form stops advancing.
    assert all(p.action == "end" for p in plans[4:])
    assert len({p.section_idx for p in plans[4:]}) == 1


def test_a_tempo_octave_re_read_is_not_a_ritardando():
    """THE REAL-MATERIAL BUG, as a unit test (see the module docstring)."""
    m = Mind(MindParams())
    script = [dict(bpm=199.5, octaves=0)] + [
        dict(bpm=b, octaves=1) for b in (99.95, 99.87, 99.93, 99.98)]
    plans = run_form(m, MAP_ABC, script)
    assert not [p for p in plans if p.action == "end"]


def test_a_single_huge_tempo_drop_is_not_a_ritardando():
    m = Mind(MindParams())
    script = [dict(bpm=b) for b in (120.0, 118.0, 60.0, 59.0, 58.0)]
    plans = run_form(m, MAP_ABC, script)
    assert not [p for p in plans if p.action == "end"]


def test_tempo_wobble_is_not_a_ritardando():
    m = Mind(MindParams())
    script = [dict(bpm=120.0 + (0.6 if i % 2 else -0.6)) for i in range(24)]
    plans = run_form(m, MAP_ABC, script)
    assert not [p for p in plans if p.action == "end"]


def test_ritard_evidence_needs_a_locked_tracker_and_a_playing_human():
    for extra in (dict(locked=False), dict(silent=True, rms=-70.0)):
        m = Mind(MindParams())
        script = [dict(bpm=b, **extra) for b in (120.0, 116.0, 112.0, 108.0)]
        assert not [p for p in run_form(m, MAP_ABC, script)
                    if p.action == "end"]


def test_loop_form_off_ends_at_the_end_of_the_map():
    m = Mind(MindParams(), FormParams(loop_form=False))
    plans = run_form(m, MAP_ABC, steady(14))
    assert plans[11].section_idx == 2
    assert plans[12].action == "end" and "loop_form is off" in plans[12].reason


# ============================================================ section 3: cues


CUE_TABLE = [
    # (text, kind, value)
    ("take it out", "end", "out"),
    ("bring it home", "end", "out"),
    ("last chorus", "end", "out"),
    ("drop to half time", "tempo", "half"),
    ("double time", "tempo", "double"),
    ("pick it up", "tempo", "faster"),
    ("slow down", "tempo", "slower"),
    ("ritardando", "tempo", "ritard"),
    ("go to the chorus", "section", "chorus"),
    ("back to the verse", "section", "verse"),
    ("hit the bridge", "section", "bridge"),
    ("straight to the drop", "section", "drop"),
    ("next section", "section", "next"),
    ("previous section", "section", "prev"),
    ("darker", "texture", "darker"),
    ("brighter", "texture", "brighter"),
    ("strip it back", "texture", "sparser"),
    ("fill it up", "texture", "busier"),
    ("more reverb", "texture", "wet"),
    ("make it trap", "style", "trap"),
    ("go lofi", "style", "lofi"),
    ("more jazzy", "style", "jazz"),
    ("louder", "energy", "up"),
    ("turn it up", "energy", "up"),
    ("bring it down", "energy", "down"),
    ("mellow", "energy", "down"),
]


@pytest.mark.parametrize("text,kind,value", CUE_TABLE)
def test_parse_cue_table(text, kind, value):
    got = parse_cue(text)
    assert got["kind"] == kind, got
    assert got["value"] == value, got
    assert got["text"] == text


def test_parse_cue_covers_every_kind():
    assert {parse_cue(t)["kind"] for t, _, _ in CUE_TABLE} == {
        "end", "tempo", "section", "texture", "style", "energy"}
    assert parse_cue("asdfgh")["kind"] == "unknown"
    assert set(CUE_KINDS) == {"energy", "section", "texture", "tempo", "end",
                              "style", "unknown"}


def test_parse_cue_extras():
    assert parse_cue("half time")["ratio"] == 0.5
    assert parse_cue("double time")["ratio"] == 2.0
    assert parse_cue("louder")["delta"] > 0
    assert parse_cue("quieter")["delta"] < 0


def test_parse_cue_is_case_and_punctuation_insensitive():
    for t in ("GO TO THE CHORUS", "Go to the chorus!!", "  go to the chorus  ",
              "go-to-the-chorus"):
        assert parse_cue(t)["kind"] == "section"
        assert parse_cue(t)["value"] == "chorus"


JUNK = ["", "   ", "asdfgh", "!!!", "12345", "\n\t", "🎺🎺🎺", None, 42, 3.5,
        object(), ["go", "to", "the", "chorus"], {"kind": "section"},
        "x" * 5000, "the the the", "-", "--", "go", "to the", b"bytes"]


@pytest.mark.parametrize("junk", JUNK)
def test_parse_cue_never_raises_on_junk(junk):
    got = parse_cue(junk)                     # must not raise
    assert got["kind"] in CUE_KINDS
    assert isinstance(got, dict) and "value" in got and "text" in got


def test_junk_mostly_reads_as_unknown_rather_than_guessing():
    kinds = [parse_cue(j)["kind"] for j in JUNK]
    assert kinds.count("unknown") >= len(JUNK) - 2, dict(zip(map(str, JUNK),
                                                            kinds))


# ------------------------------------------------------- cues in the form


def test_cue_issued_mid_phrase_applies_at_the_next_phrase_boundary():
    m = Mind(MindParams())
    steps = steady(12)
    steps[2]["cue"] = Cue("go to the chorus")          # mid-phrase (bar 3/8)
    plans = run_form(m, MAP_LONG, steps)
    assert plans[2].applied_cue is None, "applied mid-phrase"
    assert plans[3].applied_cue is None
    assert plans[4].applied_cue == "go to the chorus"  # phrase boundary
    assert plans[4].action == "jump" and plans[4].section_idx == 2
    assert plans[4].bars_left_in_section == 7
    assert "overridden by cue" in plans[4].reason
    assert all(p.section_idx == 2 for p in plans[4:11])


def test_cue_at_now_applies_immediately():
    m = Mind(MindParams())
    steps = steady(8)
    steps[2]["cue"] = Cue("go to the chorus", at="now")
    plans = run_form(m, MAP_LONG, steps)
    assert plans[2].applied_cue == "go to the chorus"
    assert plans[2].action == "jump" and plans[2].section_idx == 2


def test_cue_at_next_bar_waits_exactly_one_bar():
    m = Mind(MindParams())
    steps = steady(8)
    steps[2]["cue"] = Cue("go to the chorus", at="next-bar")
    plans = run_form(m, MAP_LONG, steps)
    assert plans[2].applied_cue is None
    assert plans[3].applied_cue == "go to the chorus"
    assert plans[3].section_idx == 2


def test_cue_at_next_section_waits_for_the_section_boundary():
    m = Mind(MindParams())
    steps = steady(14)
    steps[1]["cue"] = Cue("go to the chorus", at="next-section", ttl_bars=16)
    plans = run_form(m, MAP_LONG, steps)
    assert all(p.applied_cue is None for p in plans[:8])
    assert plans[8].applied_cue == "go to the chorus"
    assert plans[8].section_idx == 2


def test_unapplied_cue_expires_at_its_ttl():
    m = Mind(MindParams())
    steps = steady(12)
    steps[1]["cue"] = Cue("go to the chorus", at="next-section", ttl_bars=3)
    plans = run_form(m, MAP_LONG, steps)
    assert plans[4].dropped_cue == "go to the chorus"
    assert "expired unapplied" in plans[4].reason
    assert all(p.applied_cue is None for p in plans)
    assert all(p.section_idx in (0, 1) for p in plans)     # composed route only
    assert m.pending_cues() == ()


def test_ttl_does_not_drop_a_cue_that_already_applied():
    m = Mind(MindParams())
    steps = steady(12)
    steps[2]["cue"] = Cue("go to the chorus", ttl_bars=2)
    plans = run_form(m, MAP_LONG, steps)
    assert plans[4].applied_cue == "go to the chorus"
    assert all(p.dropped_cue is None for p in plans)


def test_end_cue_ends_the_tune_at_its_boundary():
    m = Mind(MindParams())
    steps = steady(12)
    steps[2]["cue"] = Cue("take it out")
    plans = run_form(m, MAP_LONG, steps)
    assert plans[3].action != "end"
    assert plans[4].action == "end" and plans[4].applied_cue == "take it out"
    assert all(p.action == "end" for p in plans[4:])


def test_energy_cue_jumps_without_waiting_for_evidence():
    m = Mind(MindParams())
    steps = steady(10)
    steps[2]["cue"] = Cue("louder")
    plans = run_form(m, MAP_UP, steps)
    assert plans[4].applied_cue == "louder"
    assert plans[4].action == "jump" and plans[4].section_idx == 2


def test_non_form_cues_are_applied_and_logged_without_moving_the_cursor():
    m = Mind(MindParams())
    steps = steady(10)
    steps[2]["cue"] = Cue("darker")
    plans = run_form(m, MAP_LONG, steps)
    assert plans[4].applied_cue == "darker"
    assert plans[4].action == "hold" and plans[4].section_idx == 0
    assert "texture:darker" in plans[4].reason


def test_unknown_cue_is_applied_as_a_no_op_and_says_so():
    m = Mind(MindParams())
    steps = steady(10)
    steps[2]["cue"] = Cue("asdfgh")
    plans = run_form(m, MAP_LONG, steps)
    assert plans[4].applied_cue == "asdfgh"
    assert "not understood" in plans[4].reason
    assert plans[4].action == "hold"


def test_section_cue_for_a_missing_section_is_ignored_with_a_reason():
    m = Mind(MindParams())
    steps = steady(10)
    steps[2]["cue"] = Cue("go to the bridge")          # MAP_LONG has none
    plans = run_form(m, MAP_LONG, steps)
    assert "no such section" in plans[4].reason
    assert plans[4].action == "hold" and plans[4].section_idx == 0


def test_cues_queue_and_all_land():
    m = Mind(MindParams())
    steps = steady(16)
    steps[1]["cue"] = Cue("darker")
    steps[2]["cue"] = Cue("go to the chorus")
    plans = run_form(m, MAP_LONG, steps)
    # Both come due at the same phrase boundary: both are reported, in issue
    # order, and both effects land.
    assert plans[4].applied_cue == "darker; go to the chorus"
    assert "texture:darker" in plans[4].reason
    assert plans[4].section_idx == 2 and plans[4].action == "jump"
    assert [p.applied_cue for p in plans].count(None) == len(plans) - 1


@pytest.mark.parametrize("at", list(CUE_BOUNDARIES) + ["junk", "", None])
def test_every_cue_boundary_value_is_safe(at):
    m = Mind(MindParams())
    steps = steady(14)
    steps[2]["cue"] = Cue("go to the chorus", at=at)
    plans = run_form(m, MAP_LONG, steps)
    assert any(p.applied_cue for p in plans), at
    assert all(p.action in FORM_ACTIONS for p in plans)


def test_cue_change_is_exempt_from_announce_but_machine_changes_are_not():
    m = Mind(MindParams())
    steps = steady(14)
    steps[2]["cue"] = Cue("go to the chorus")
    plans = run_form(m, MAP_LONG, steps)
    assert plans[3].announce is False            # no warning was possible
    assert plans[4].action == "jump"
    announce_invariant(plans)                    # still holds for the rest


# ============================================================== determinism


def _mixed_script(n: int) -> list[dict]:
    """A deterministic, varied script (no rng): levels, densities, cues."""
    steps = []
    for i in range(n):
        rms = -30.0 + 12.0 * ((i // 5) % 3) + (1.5 if i % 3 == 0 else -1.0)
        step = dict(rms=rms, dens=1.0 + 0.5 * (i % 4),
                    silent=(i % 17 == 16), bpm=120.0 - 0.4 * (i % 5))
        if i % 11 == 7:
            step["cue"] = Cue(["darker", "go to the chorus", "louder",
                               "asdfgh"][(i // 11) % 4])
        steps.append(step)
    return steps


def test_determinism_identical_runs_identical_plans():
    steps = _mixed_script(120)
    a = run_form(Mind(MindParams()), MAP_LONG, [dict(s) for s in steps])
    b = run_form(Mind(MindParams()), MAP_LONG, [dict(s) for s in steps])
    assert a == b
    assert len({p.action for p in a}) > 1, "the script exercised one path only"


def test_plan_form_never_leaves_the_contract():
    plans = run_form(Mind(MindParams()), MAP_LONG, _mixed_script(200))
    for p in plans:
        assert p.action in FORM_ACTIONS
        assert -1 <= p.section_idx < 4
        assert p.bars_left_in_section >= 0
        assert p.reason.startswith("form: ")
        assert isinstance(p.announce, bool)


def test_module_has_no_rng_no_clock_in_the_form_path():
    import inspect

    import plume.mind as mind_mod
    src = inspect.getsource(mind_mod)
    for banned in ("import random", "np.random", "default_rng", "time.time",
                   "perf_counter"):
        assert banned not in src


# ========================================================== REAL MATERIAL
#
# Everything above this line is synthetic.  v1 passed ten of those and was
# unusable.  These run the form cursor over plume's own v0 renders.


def _render(path: Path, style: str, bpm: float, key: str, *,
            bars: tuple[int, int] = (0, 32), seed: int = 7,
            gain_db: float = 0.0) -> Path:
    session = compose(
        PromptSpec(styles=[style], bpm=bpm, key=key, movements=1, mood=[],
                   name=path.stem),
        seed=seed)
    render(session, out_path=str(path), movement=0, bars=bars)
    if gain_db:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        sf.write(str(path), x * (10.0 ** (gain_db / 20.0)), sr, subtype="PCM_24")
    return path


def _human_bars(path: Path, params: MindParams):
    """Real ``BarFeatures`` + ``BeatState`` per bar boundary.

    Mirrors ``sim.simulate``'s boundary loop (and reuses its accumulator) so
    the cursor sees exactly what the live pipeline hands it — including the
    silence rule, the edge trim and the running human reference.
    """
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1).astype(np.float32)
    fe = FeatureExtractor(sr=sr)
    tracker, harmony = BeatTracker(), HarmonyTracker()
    acc = sim_mod._BarAccum()
    rows, ref, prev, last_bar = [], None, None, -2
    for f in fe.process(x):
        st = tracker.process([f])
        hs = harmony.process([f])
        acc.add(f, params.silence_rms_db)
        if st.locked and st.bar > max(last_bar, -1):
            if prev is not None and math.isfinite(prev.next_downbeat_t_s):
                t_b = prev.next_downbeat_t_s
            else:
                t_b = max(0.0, float(f.t_s)
                          - st.phase_beats * st.beat_interval_s)
            bf = sim_mod._summarize(acc, st.bar - 1, t_b, st.beat_interval_s,
                                    hs, params, ref)
            if not bf.silent:
                ref = bf.rms_db if ref is None else 0.5 * bf.rms_db + 0.5 * ref
            rows.append((bf, st, hs))
            acc.reset(t_b)
            last_bar = st.bar
        prev = st
    return rows, len(x) / sr


#: (name, style, bpm, key, gain_db).  Two gains 20 dB apart per style: the
#: axis fixtures do not have, and the one that broke v1.
REAL = [
    ("lofi80", "lofi", 80.0, "C major", 0.0),
    ("lofi80_quiet", "lofi", 80.0, "C major", -20.0),
    ("trap140", "trap", 140.0, "D minor", 0.0),
    ("trap140_quiet", "trap", 140.0, "D minor", -20.0),
    ("synth100", "synthwave", 100.0, "C minor", 0.0),
    ("synth100_quiet", "synthwave", 100.0, "C minor", -20.0),
]
REAL_NAMES = [r[0] for r in REAL]


@pytest.fixture(scope="module")
def real_bars(tmp_path_factory):
    """Real bar features per case, built once (renders are not cheap)."""
    d = tmp_path_factory.mktemp("form_real")
    params = MindParams()
    out = {}
    for name, style, bpm, key, gain in REAL:
        p = _render(d / f"{name}.wav", style, bpm, key, gain_db=gain)
        rows, dur = _human_bars(p, params)
        assert rows, f"{name}: the pipeline produced no bars at all"
        out[name] = (rows, dur)
    return out


def _run_real(rows, session=MAP_LONG, form: FormParams | None = None,
              db_offset: float = 0.0) -> list[FormPlan]:
    m = Mind(MindParams(), form)
    plans = []
    for bf, st, hs in rows:
        if db_offset:
            bf = dataclasses.replace(bf, rms_db=bf.rms_db + db_offset)
        plans.append(m.plan_form(None, st, hs, bf, session, None))
    return plans


@pytest.mark.parametrize("name", REAL_NAMES)
def test_real_material_never_false_triggers_the_ending(real_bars, name):
    """THE REGRESSION.  Quiet synthwave ended the tune on bar 2.

    The tracker locks at 199.5 bpm, re-reads the metre down an octave, and
    the naive ritard rule saw two falling estimates and committed to an
    ending it could never leave.  Nothing in the audio slowed down.
    """
    rows, _ = real_bars[name]
    plans = _run_real(rows)
    ends = [p for p in plans if p.action == "end"]
    assert not ends, f"{name}: false ending — {ends[0].reason if ends else ''}"


@pytest.mark.parametrize("name", REAL_NAMES)
def test_real_material_navigates_the_map(real_bars, name):
    """The cursor keeps moving, stays in the contract, and announces."""
    rows, _ = real_bars[name]
    plans = _run_real(rows)
    assert len(plans) == len(rows)
    for p in plans:
        assert p.action in FORM_ACTIONS
        assert 0 <= p.section_idx < 4
        assert p.bars_left_in_section >= 0
    # It advanced at least once (a 30+ bar take over 8-bar sections).
    assert len({p.section_idx for p in plans}) >= 2, f"{name}: cursor stuck"
    announce_invariant(plans)


@pytest.mark.parametrize("name", REAL_NAMES)
def test_real_material_extension_stays_bounded(real_bars, name):
    """Real bar-to-bar level noise must not extend a section forever."""
    rows, _ = real_bars[name]
    fp = FormParams()
    plans = _run_real(rows)
    runs, cur = [], 0
    for prev, p in zip(plans, plans[1:]):
        if p.section_idx == prev.section_idx:
            cur += 1
        else:
            runs.append(cur)
            cur = 0
    runs.append(cur)
    longest = max(runs) + 1
    limit = 8 + fp.max_extend_bars          # section bars + the whole budget
    assert longest <= limit, f"{name}: sat in one section for {longest} bars"


@pytest.mark.parametrize("name", REAL_NAMES)
def test_real_material_decisions_are_invariant_to_input_gain(real_bars, name):
    """Shift every real bar's level; the plan sequence must not move.

    This is the v1 failure written as a property.  ``silence_rms_db`` style
    absolute thresholds fail it by construction; the relative fast/slow EMA
    rule passes it exactly.
    """
    rows, _ = real_bars[name]
    base = _run_real(rows)
    for offset in (-30.0, -12.0, +9.0):
        assert _run_real(rows, db_offset=offset) == base, f"{name} {offset:+g}dB"


def test_real_material_two_gains_agree_in_kind(real_bars):
    """The same take 20 dB down must not navigate differently in kind.

    Not bar-identical — the front end genuinely hears less at -20 dB (fewer
    onsets, later lock, fewer bars).  But the RATE of machine-initiated
    changes must stay in the same regime, which is what an absolute-level
    rule would violate.
    """
    for loud, quiet in (("lofi80", "lofi80_quiet"),
                        ("trap140", "trap140_quiet"),
                        ("synth100", "synth100_quiet")):
        rates = []
        for name in (loud, quiet):
            rows, _ = real_bars[name]
            plans = _run_real(rows)
            changes = sum(1 for p in plans if p.action in ("advance", "jump"))
            rates.append(changes / max(1, len(plans)))
        assert abs(rates[0] - rates[1]) < 0.12, (loud, quiet, rates)
        assert all(r < 0.5 for r in rates), (loud, quiet, rates)


def test_real_material_cue_lands_on_a_boundary(real_bars):
    """A cue issued mid-phrase over real audio still waits for the phrase."""
    rows, _ = real_bars["trap140"]
    m = Mind(MindParams())
    plans, issued = [], 5
    for i, (bf, st, hs) in enumerate(rows):
        cue = Cue("go to the chorus") if i == issued else None
        plans.append(m.plan_form(None, st, hs, bf, MAP_LONG, cue))
    applied = [i for i, p in enumerate(plans) if p.applied_cue]
    assert len(applied) == 1 and applied[0] > issued
    p = plans[applied[0]]
    assert p.action == "jump" and p.section_idx == 2
    assert plans[applied[0] - 1].bars_left_in_section in (0, 3, 7)


def test_real_material_determinism(real_bars):
    for name in REAL_NAMES:
        rows, _ = real_bars[name]
        assert _run_real(rows) == _run_real(rows), name


def test_form_planning_cost_is_negligible_on_the_live_path(real_bars):
    """plan_form must not eat the beat tracker's ~180x headroom.

    Measured here as audio seconds per wall second for the CURSOR ALONE
    (feature extraction and beat tracking excluded — they are v1's cost and
    are measured in v1's suite).  Typical: ~5 us/bar, i.e. six orders of
    magnitude above real time.  The assertion is deliberately loose so a
    loaded machine cannot make it flap.
    """
    rows, dur = real_bars["trap140"]
    reps = 40
    t0 = time.perf_counter()
    for _ in range(reps):
        _run_real(rows)
    wall = (time.perf_counter() - t0) / reps
    ratio = dur / wall
    assert ratio > 2000.0, (f"form planning ran at only {ratio:,.0f}x real "
                            f"time ({wall * 1e6 / len(rows):.1f} us/bar)")
