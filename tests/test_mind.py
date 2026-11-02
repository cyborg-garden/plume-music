"""Tests for plume.mind — the negotiation state machine (SPEC-V1 section 4).

Table-driven FSM transition coverage for every arc in the spec, plus focused
tests for the ramp math, density hysteresis timing, silence-counter /
conf-floor independence, register avoidance, dynamic-arc gain following,
reason-string contract, and determinism.
"""
from __future__ import annotations

import dataclasses
import inspect
import math
import random
import re

import pytest

import plume.mind as mind_mod
from plume.listen.beat import BeatState
from plume.listen.harmony import HarmonyState
from plume.mind import STATES, BarFeatures, BarPlan, Mind, MindParams, PARAM_HELP

# ------------------------------------------------------------------ builders


def mk_beat(*, bar: int = 0, conf: float = 0.9, locked: bool = True,
            bpm: float = 96.0, t: float = 0.0) -> BeatState:
    interval = 60.0 / bpm if bpm > 0 else math.inf
    return BeatState(
        t_s=t, bpm=bpm, confidence=conf, locked=locked, bar=bar,
        beat_in_bar=0, phase_beats=0.0,
        next_beat_t_s=t + interval if bpm > 0 else math.inf,
        next_downbeat_t_s=t + 4 * interval if locked else math.inf,
        beat_interval_s=interval,
    )


def mk_harm(*, key: str | None = "A minor", key_conf: float = 0.8,
            chord: str | None = "Am", chord_conf: float = 0.7,
            lo: float | None = None, hi: float | None = None,
            t: float = 0.0) -> HarmonyState:
    return HarmonyState(t_s=t, key=key, key_conf=key_conf, chord=chord,
                        chord_conf=chord_conf, register_low_midi=lo,
                        register_high_midi=hi)


def mk_bar(bar: int, *, rms: float = -20.0, dens: float = 1.0,
           silent: bool = False, lo: float | None = None,
           hi: float | None = None) -> BarFeatures:
    return BarFeatures(bar=bar, rms_db=rms, onset_density=dens, silent=silent,
                       register_low_midi=lo, register_high_midi=hi)


def run_script(m: Mind, steps: list[dict], *, start_bar: int = 0) -> list[BarPlan]:
    """Drive a Mind through scripted bars.

    Each step dict may set: conf, locked, bpm (BeatState); rms, dens, silent,
    lo, hi (BarFeatures); or human=None for an unlocked per-second call.
    """
    plans = []
    bar = start_bar
    for step in steps:
        beat = mk_beat(bar=step.get("bar", bar), conf=step.get("conf", 0.9),
                       locked=step.get("locked", True), bpm=step.get("bpm", 96.0))
        harm = mk_harm(lo=step.get("harm_lo"), hi=step.get("harm_hi"))
        if step.get("human", "yes") is None:
            human = None
        else:
            human = mk_bar(bar, rms=step.get("rms", -20.0),
                           dens=step.get("dens", 1.0),
                           silent=step.get("silent", False),
                           lo=step.get("lo"), hi=step.get("hi"))
            bar += 1
        plans.append(m.plan_bar(beat, harm, human))
    return plans


#: params with short silence timers so the whole FSM is reachable quickly.
def small_params(**overrides) -> MindParams:
    kw = dict(min_listen_bars=4, entry_boundary_bars=4, ramp_bars=3,
              silence_thin_bars=2, silence_fade_bars=4, silence_tacet_bars=6)
    kw.update(overrides)
    return MindParams(**kw)


WARM = [{}, {}, {}, {}]                 # 4 human bars -> entry plan at bar 4
RAMP = [{}, {}]                          # remaining entering bars (k=1, 2)
TO_COMP = WARM + RAMP + [{}]             # first comping plan at bar 7


# ------------------------------------------------- contract shape (frozen)


def test_states_tuple_exact():
    assert STATES == ("waiting", "listening", "entering", "comping",
                      "thinned", "fading", "tacet")


def test_mindparams_fields_and_defaults_match_spec():
    expected = {
        "min_listen_bars": 4, "entry_conf": 0.45, "entry_boundary_bars": 4,
        "enter_under_db": -8.0, "ramp_bars": 3, "density_high": 2.5,
        "density_hysteresis": 0.8, "silence_thin_bars": 2,
        "silence_fade_bars": 6, "silence_tacet_bars": 10,
        "silence_rms_db": -45.0, "conf_floor": 0.3,
        "register_margin_st": 5.0, "energy_alpha": 0.5,
        # v1.4 commitment (see tests/test_commitment.py):
        "commit_bars": 8, "exit_conf": 0.25, "recommit_bars": 4,
        # v2.6 harmonic commitment (see tests/test_governor.py):
        "harmony_commit_bars": 4, "harmony_release_bars": 8,
    }
    fields = {f.name: f.default for f in dataclasses.fields(MindParams)}
    assert fields == expected


def test_param_help_covers_every_field():
    names = {f.name for f in dataclasses.fields(MindParams)}
    assert set(PARAM_HELP) == names
    assert all(isinstance(v, str) and v for v in PARAM_HELP.values())


def test_barfeatures_and_barplan_fields():
    assert [f.name for f in dataclasses.fields(BarFeatures)] == [
        "bar", "rms_db", "onset_density", "silent",
        "register_low_midi", "register_high_midi"]
    assert [f.name for f in dataclasses.fields(BarPlan)] == [
        "bar", "state", "roles", "gain_db", "density",
        "register_ceiling_midi", "reason",
        # v1.4 reciprocal cueing — defaulted, so the v1 keyword constructions
        # in sim/realtime/duet are untouched.  See tests/test_commitment.py.
        "announce", "announce_kind", "announce_bar"]


# ------------------------------------------------- table-driven FSM arcs

# Each case: (name, params-overrides, script, expected final state,
#             required substring of the final reason).
FSM_CASES = [
    ("waiting_no_signal", {},
     [{"human": None, "conf": 0.0, "locked": False, "bpm": 0.0, "bar": -1}],
     "waiting", "no human signal"),
    ("waiting_to_listening_on_bar", {},
     [{"conf": 0.2, "locked": False}],
     "listening", "heard 1/4 bars"),
    ("waiting_to_listening_on_pulse", {},
     [{"human": None, "conf": 0.2, "locked": False, "bpm": 90.0, "bar": -1}],
     "listening", "heard 0/4 bars"),
    ("listening_to_entering", {}, WARM, "entering", "at bar 4 boundary"),
    ("gate_blocks_short_listen", {}, WARM[:3], "listening", "heard 3/4 bars"),
    ("gate_blocks_low_conf", {},
     WARM[:3] + [{"conf": 0.4}], "listening", "conf 0.40 < entry threshold"),
    ("gate_blocks_unlocked", {},
     [{"locked": False}] * 4, "listening", "awaiting beat lock"),
    ("gate_blocks_off_boundary", {"min_listen_bars": 3},
     WARM[:3], "listening", "awaiting bar 4 boundary"),
    ("gate_waits_for_next_boundary", {"min_listen_bars": 5},
     [{}] * 5, "listening", "awaiting bar 8 boundary"),
    ("entering_ramp_continues", {}, WARM + [{}], "entering", "ramp bar 2/3"),
    ("entering_to_comping", {}, TO_COMP, "comping", "onsets/beat"),
    ("comping_to_thinned", {},
     TO_COMP + [{"silent": True}, {"silent": True}], "thinned", "human silent 2 bars"),
    ("thinned_back_to_comping", {},
     TO_COMP + [{"silent": True}, {"silent": True}, {}], "comping", "onsets/beat"),
    ("comping_to_fading_beatloss", {},
     TO_COMP + [{"conf": 0.1}], "fading", "beat lost"),
    ("fading_to_tacet_beatloss", {},
     TO_COMP + [{"conf": 0.1}, {"conf": 0.1}], "tacet", "re-armed"),
    ("beatloss_tacet_to_listening", {},
     TO_COMP + [{"conf": 0.1}, {"conf": 0.1}, {}], "listening", "re-armed after beat loss"),
    # v1.4 AMENDMENT: a SINGLE unlocked bar inside the commitment window is a
    # wobble, not an unlock — the machine holds its last sounding bar (this is
    # P0-C, the flapping fix; see tests/test_commitment.py).  A SUSTAINED
    # unlock is still an emergency and still triggers the repair, which is
    # what this case now asserts.
    ("comping_holds_one_unlocked_bar_while_committed", {},
     TO_COMP + [{"conf": 0.6, "locked": False}], "comping", "tracker wobbled"),
    ("comping_to_fading_sustained_unlock", {},
     TO_COMP + [{"conf": 0.6, "locked": False}] * 5, "fading", "unlocked"),
    ("entering_to_fading_beatloss_during_ramp", {},
     WARM + [{"conf": 0.05}], "fading", "beat lost"),
]


@pytest.mark.parametrize("name,overrides,script,state,substr",
                         [c for c in FSM_CASES], ids=[c[0] for c in FSM_CASES])
def test_fsm_arc(name, overrides, script, state, substr):
    if not isinstance(overrides, dict):
        overrides = {}
    m = Mind(small_params(**overrides))
    plans = run_script(m, script)
    assert plans[-1].state == state, plans[-1]
    assert substr in plans[-1].reason, plans[-1].reason


def test_silence_fsm_full_arc_and_rearm():
    """comping -> thinned -> fading -> tacet -> listening -> entering again."""
    m = Mind(small_params())
    plans = run_script(m, TO_COMP + [{"silent": True}] * 6)
    states = [p.state for p in plans]
    # 7 active plans, then silent bars 1..6 -> comping, thinned x2, fading x2, tacet
    assert states[-6:] == ["comping", "thinned", "thinned", "fading", "fading", "tacet"]
    assert "entry re-armed" in plans[-1].reason
    # resume: listening, then re-entry only at the next multiple-of-4 boundary
    more = run_script(m, [{}, {}], start_bar=13)   # upcoming bars 14, 15
    assert [p.state for p in more] == ["listening", "listening"]
    assert "re-armed after tacet" in more[0].reason
    reenter = run_script(m, [{}], start_bar=15)    # upcoming bar 16 boundary
    assert reenter[0].state == "entering"
    assert reenter[0].gain_db == pytest.approx(m.params.enter_under_db)
    assert "re-armed" in reenter[0].reason


def test_stops_never_reach_tacet():
    m = Mind(small_params())
    plans = run_script(m, TO_COMP + [{"silent": True}, {"silent": True}, {}, {}])
    states = [p.state for p in plans]
    assert "tacet" not in states and "fading" not in states
    assert states[-1] == "comping"        # back within one bar of resume


def test_silence_fading_gain_steps_down():
    m = Mind(small_params())
    plans = run_script(m, TO_COMP + [{"silent": True}] * 5)
    fading = [p for p in plans if p.state == "fading"]
    assert len(fading) == 2
    assert fading[1].gain_db == pytest.approx(fading[0].gain_db - 6.0)


# ------------------------------------------------- entry gate + ramp math


def test_enters_on_boundary_with_under_gain_and_sparse_roles():
    m = Mind(MindParams())
    plans = run_script(m, WARM)
    entry = plans[-1]
    assert entry.state == "entering"
    assert entry.bar == 4                       # min_listen AND boundary
    assert entry.gain_db == pytest.approx(-8.0)
    assert set(entry.roles) == {"bass", "pad"}  # sparse: no kit, no fills
    assert [p.state for p in plans[:-1]] == ["listening"] * 3


def test_ramp_math_linear_to_zero():
    m = Mind(MindParams())          # enter_under_db=-8, ramp_bars=3
    plans = run_script(m, WARM + RAMP + [{}])
    gains = [p.gain_db for p in plans[4:]]
    assert plans[4].state == "entering" and plans[5].state == "entering"
    assert plans[6].state == "comping"
    assert gains[0] == pytest.approx(-8.0 * 2 / 3)
    assert gains[1] == pytest.approx(-8.0 * 1 / 3)
    assert gains[2] == pytest.approx(0.0)       # flat human rms -> arc delta 0


def test_ramp_bars_zero_still_enters_under_then_comps():
    m = Mind(MindParams(ramp_bars=0))
    plans = run_script(m, WARM + [{}])
    assert plans[-2].state == "entering"
    assert plans[-2].gain_db == pytest.approx(-8.0)
    assert plans[-1].state == "comping"


def test_no_entry_before_min_listen_even_on_boundary():
    m = Mind(MindParams(min_listen_bars=6))
    plans = run_script(m, [{}] * 6)             # boundary at 4 passes unentered
    assert all(p.state == "listening" for p in plans[:5])
    # bars_listened=6 after 6 bars; upcoming bar 6 is not a boundary; bar 8 is
    plans2 = run_script(m, [{}, {}], start_bar=6)
    assert plans2[0].state == "listening"
    assert plans2[1].state == "entering" and plans2[1].bar == 8


def test_silent_bars_do_not_count_toward_min_listen():
    m = Mind(small_params())
    plans = run_script(m, [{"silent": True}] * 4 + WARM)
    # 4 silent bars heard nothing; entry needs 4 non-silent bars AND boundary:
    # bars_listened hits 4 at human.bar=7 -> upcoming 8 is a boundary.
    assert [p.state for p in plans[-2:]] == ["listening", "entering"]
    assert plans[-1].bar == 8


# ------------------------------------------------- density coupling


def test_inverse_density_coupling_with_one_bar_delayed_recovery():
    m = Mind(MindParams())
    run_script(m, TO_COMP)                          # comping from bar 7
    seq = [{"dens": 1.0}, {"dens": 3.0}, {"dens": 3.0},
           {"dens": 1.0}, {"dens": 1.0}, {"dens": 1.0}]
    plans = run_script(m, seq, start_bar=7)
    dens = [p.density for p in plans]
    calm, busy = dens[0], dens[1]
    assert busy < calm                              # inverse coupling
    assert dens[1] == dens[2] == busy               # stays thin while busy
    # density fell at the 4th bar; recovery is delayed EXACTLY one bar:
    assert dens[3] == busy                          # first calm bar: still thin
    assert dens[4] == calm                          # second calm bar: recovered
    assert dens[5] == calm
    assert "recovering next bar" in plans[3].reason
    assert all(p.state == "comping" for p in plans)  # coupling is a hint, not a state


def test_density_hysteresis_mid_zone_holds_thin():
    m = Mind(MindParams())                          # high=2.5, lo=1.7
    run_script(m, TO_COMP)
    plans = run_script(m, [{"dens": 3.0}] + [{"dens": 2.0}] * 4, start_bar=7)
    assert all(p.density == plans[0].density for p in plans[1:])
    assert "sparse" in plans[-1].reason


def test_density_recovery_cancelled_by_renewed_busyness():
    m = Mind(MindParams())
    run_script(m, TO_COMP)
    seq = [{"dens": 3.0}, {"dens": 1.0}, {"dens": 2.6},
           {"dens": 1.0}, {"dens": 1.0}]
    plans = run_script(m, seq, start_bar=7)
    dens = [p.density for p in plans]
    assert dens[0] == dens[1] == dens[2] == dens[3]  # still thin (delay re-armed)
    assert dens[4] > dens[0]                          # finally recovered


# ------------------------------------------------- beat-loss repair


def test_beat_loss_freezes_silence_counters():
    m = Mind(small_params())
    run_script(m, TO_COMP + [{"silent": True}])      # silent_bars == 1
    assert m._silent_bars == 1
    plans = run_script(m, [{"conf": 0.1, "silent": True},
                           {"conf": 0.1, "silent": True}], start_bar=8)
    assert [p.state for p in plans] == ["fading", "tacet"]
    assert m._silent_bars == 1                       # frozen during beat loss
    assert "silence counters frozen" in plans[0].reason


def test_beat_loss_is_emergency_not_boundary_gated():
    m = Mind(MindParams())
    # one extra comping bar so the loss lands mid-cycle (upcoming bar 9):
    plans = run_script(m, TO_COMP + [{}] + [{"conf": 0.1}])
    assert plans[-1].state == "fading"
    assert plans[-1].bar == 9 and plans[-1].bar % 4 != 0   # fired off-boundary
    plans2 = run_script(m, [{"conf": 0.1}], start_bar=9)
    assert plans2[0].state == "tacet"


def test_beat_loss_recovery_reenters_through_gate():
    m = Mind(MindParams())
    run_script(m, TO_COMP + [{"conf": 0.1}, {"conf": 0.1}])   # fading, tacet at bars 8,9
    plans = run_script(m, [{}, {}, {}], start_bar=9)          # upcoming 10, 11, 12
    assert [p.state for p in plans] == ["listening", "listening", "entering"]
    assert plans[2].bar == 12
    assert plans[2].gain_db == pytest.approx(-8.0)            # under-gain again


def test_conf_between_floor_and_entry_keeps_comping():
    m = Mind(MindParams())                                     # floor 0.3
    plans = run_script(m, TO_COMP + [{"conf": 0.4}, {"conf": 0.35}])
    assert [p.state for p in plans[-2:]] == ["comping", "comping"]


# ------------------------------------------------- register avoidance


def test_register_ceiling_from_human_bar():
    m = Mind(MindParams())
    plans = run_script(m, [{"lo": 64.0, "hi": 76.0}] * 7)
    comp = plans[-1]
    assert comp.state == "comping"
    assert comp.register_ceiling_midi == pytest.approx(64.0 - 5.0)


def test_register_ceiling_falls_back_to_harmony_and_holds():
    m = Mind(MindParams())
    run_script(m, WARM[:3])
    plans = run_script(m, [{"harm_lo": 62.0}], start_bar=3)    # harmony fallback
    entry = plans[0]
    assert entry.state == "entering"
    assert entry.register_ceiling_midi == pytest.approx(57.0)
    # register info disappears -> last known ceiling held while active
    plans2 = run_script(m, [{}], start_bar=4)
    assert plans2[0].register_ceiling_midi == pytest.approx(57.0)


def test_register_ceiling_none_when_never_observed_or_inactive():
    m = Mind(MindParams())
    plans = run_script(m, WARM)
    assert all(p.register_ceiling_midi is None for p in plans[:-1])  # listening
    assert plans[-1].register_ceiling_midi is None                    # nothing observed
    m2 = Mind(small_params())
    plans2 = run_script(m2, [{"lo": 60.0}] * 7 + [{"silent": True}] * 6)
    assert plans2[-1].state == "tacet"
    assert plans2[-1].register_ceiling_midi is None                   # inactive


# ------------------------------------------------- dynamic-arc following


def test_arc_gain_follows_ema_of_human_rms():
    m = Mind(MindParams())                    # alpha 0.5, entry ref -20
    run_script(m, TO_COMP)
    plans = run_script(m, [{"rms": -14.0}] * 3, start_bar=7)
    gains = [p.gain_db for p in plans]
    assert gains[0] == pytest.approx(3.0)     # ema -17 -> delta +3
    assert gains[1] == pytest.approx(4.5)
    assert gains[0] < gains[1] < gains[2] <= 6.0


def test_arc_gain_clamps():
    m = Mind(MindParams(energy_alpha=1.0))
    run_script(m, TO_COMP)
    loud = run_script(m, [{"rms": 0.0}], start_bar=7)
    assert loud[0].gain_db == pytest.approx(6.0)
    quiet = run_script(m, [{"rms": -40.0}], start_bar=8)
    assert quiet[0].gain_db == pytest.approx(-12.0)


def test_ramp_gain_is_pure_ramp_not_arc():
    m = Mind(MindParams())
    plans = run_script(m, WARM + [{"rms": -5.0}, {"rms": -5.0}])
    assert plans[4].gain_db == pytest.approx(-8.0 * 2 / 3)   # ramp untouched
    assert plans[5].gain_db == pytest.approx(-8.0 * 1 / 3)


# ------------------------------------------------- invariants + hygiene


def _fuzz_script(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    steps: list[dict] = []
    for _ in range(n):
        step: dict = {
            "conf": rng.random(),
            "locked": rng.random() < 0.6,
            "silent": rng.random() < 0.25,
            "dens": rng.uniform(0.0, 4.0),
            "rms": rng.uniform(-60.0, 0.0),
        }
        if rng.random() < 0.15:
            step["human"] = None
            step["bar"] = -1 if rng.random() < 0.5 else rng.randrange(0, 32)
        if rng.random() < 0.3:
            step["lo"] = rng.uniform(40.0, 90.0)
        steps.append(step)
    return steps


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_property_never_entering_or_comping_while_unlocked(seed):
    m = Mind(small_params())
    steps = _fuzz_script(seed, 400)
    bar = 0
    for step in steps:
        beat = mk_beat(bar=step.get("bar", bar), conf=step["conf"],
                       locked=step["locked"], bpm=96.0)
        harm = mk_harm(lo=step.get("lo"))
        human = None
        if step.get("human", "yes") is not None:
            human = mk_bar(bar, rms=step["rms"], dens=step["dens"],
                           silent=step["silent"], lo=step.get("lo"))
            bar += 1
        plan = m.plan_bar(beat, harm, human)
        if plan.state in ("entering", "comping"):
            # v1.4 AMENDMENT.  The v1 invariant was "never sounding a full
            # texture while unlocked".  Commitment buys _UNLOCK_GRACE_BARS
            # calls of tolerance for a WOBBLING tracker (P0-C), so the
            # invariant is now: locked, or explicitly holding the last grid
            # through a wobble and saying so in the reason.  It is still an
            # error to comp on an unlocked tracker silently.
            assert beat.locked or plan.reason.startswith("tracker wobbled"), \
                (plan, step)
        assert plan.state in STATES
        assert 0.0 <= plan.density <= 1.0
        assert all(0.0 <= v <= 1.0 for v in plan.roles.values())
        assert math.isfinite(plan.gain_db) and plan.gain_db <= 6.0
        assert isinstance(plan.reason, str) and plan.reason


def test_determinism_identical_runs_identical_plans():
    steps = _fuzz_script(seed=7, n=300)
    m1, m2 = Mind(small_params()), Mind(small_params())
    p1 = run_script(m1, [dict(s) for s in steps])
    p2 = run_script(m2, [dict(s) for s in steps])
    assert p1 == p2


def test_mind_module_is_rng_free_and_pure():
    src = inspect.getsource(mind_mod)
    assert "import random" not in src
    assert "np.random" not in src and "default_rng" not in src
    assert "time.time" not in src and "datetime" not in src


def test_roles_dict_is_a_fresh_copy_per_plan():
    m = Mind(MindParams())
    plans = run_script(m, TO_COMP)
    plans[-1].roles["kick"] = 99.0                    # caller mutation
    m2 = Mind(MindParams())
    fresh = run_script(m2, TO_COMP)[-1]
    assert fresh.roles["kick"] == 1.0                 # tables untouched


def test_inactive_states_have_empty_roles():
    m = Mind(small_params())
    plans = run_script(m, WARM[:2])
    assert all(p.roles == {} for p in plans)          # listening
    plans2 = run_script(m, WARM[2:] + [{"silent": True}] * 6, start_bar=2)
    assert plans2[-1].state == "tacet" and plans2[-1].roles == {}


def test_thinned_roles_are_a_skeleton_subset_of_comping():
    m = Mind(small_params())
    plans = run_script(m, TO_COMP + [{"silent": True}, {"silent": True}])
    comp, thin = plans[6], plans[-1]
    assert comp.state == "comping" and thin.state == "thinned"
    assert set(thin.roles) < set(comp.roles)


def test_plan_bar_numbers_are_the_upcoming_bar():
    m = Mind(MindParams())
    plans = run_script(m, WARM)
    assert [p.bar for p in plans] == [1, 2, 3, 4]
    m2 = Mind(MindParams())
    unlocked = m2.plan_bar(mk_beat(bar=-1, conf=0.0, locked=False, bpm=0.0),
                           mk_harm(), None)
    assert unlocked.bar == -1


# ------------------------------------------------- reason-string contract


def test_entry_reason_is_specific():
    m = Mind(MindParams())
    entry = run_script(m, WARM)[-1]
    assert re.search(r"entering: conf 0\.\d{2} at bar 4 boundary", entry.reason)
    assert "-8.0dB under human" in entry.reason
    assert "ramp 3 bars" in entry.reason


def test_reason_strings_name_their_evidence():
    m = Mind(small_params())
    plans = run_script(m, TO_COMP + [{"silent": True}] * 4 + [{"conf": 0.1}])
    by_state = {p.state: p.reason for p in plans}
    assert "onsets/beat" in by_state["comping"]
    assert re.search(r"human silent \d bars", by_state["thinned"])
    assert re.search(r"conf 0\.10 < floor 0\.30", by_state["fading"]) or \
        "human silent" in by_state["fading"]
    # the beat-loss fading reason specifically (last plan):
    assert "beat lost" in plans[-1].reason and "conf 0.10" in plans[-1].reason
