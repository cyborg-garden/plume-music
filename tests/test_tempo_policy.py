"""Tests for ``RoleParams.tempo_policy`` — whose clock is it (v2.2 amendment).

The governing document is ``docs/decisions/2026-08-15-tempo-authority.md``:
lead mode **holds** time by default and follows *intent*, not *error*.

This file is organised around the four claims that decision makes, because
each one is a way the change could be wrong:

1. **hold really holds.**  A rushing joiner draws zero concessions and moves
   the tempo by exactly 0.0% — not "a small amount", not "bounded", zero.
   Including the case v2.1 documented as its residual (a fixed recording,
   which cannot bend back, and against which the meet law is a bounded hunt).
2. **hold is not deaf.**  The offset is still measured, still bounded by every
   plausibility gate, and still reported — ``RoleState.human_offset_ms`` plus a
   ``tempo_hold`` event and a reason naming the number.  A silent hold and a
   broken measurement must not look the same.
3. **hold still follows INTENT.**  Explicit tempo cues (ritard / faster /
   slower / half time / double time) and the ending-ritardando detector are a
   different subsystem, and they must keep working.  This is the test that
   would have caught the plausible-but-wrong implementation of this change:
   gating the concede path *and* the cue path together, because both end up
   as "the machine's bpm moved".
4. **meet and chase still exist and are distinguishable.**  Every hold
   assertion here has a meet counterpart on the SAME input showing the meet
   law does move — otherwise "no concession" proves nothing.

MEASURED ON REAL MATERIAL.  This project's recurring failure is code that
passes synthetic fixtures and fails on real music, so the hold/meet separation
is asserted on plume's OWN v0 renders (trap 140 / D minor and lo-fi 80 / C
major, at 0 dB and -20 dB, plus a copy shifted 30 ms early) driven through the
real ``FeatureExtractor`` / ``BeatTracker`` / ``HarmonyTracker`` in the closed
loop ``simulate_lead`` actually closes, and end to end through
``simulate_lead`` itself.  The real-material harness is imported from
``test_roles`` rather than re-implemented: it carries the fader-normalisation
and swing discipline that make the gain axis mean anything (the same precedent
as ``test_fixes_v1_1`` importing ``test_beat``).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plume.listen.beat import BeatState                          # noqa: E402
from plume.listen.harmony import HarmonyState                    # noqa: E402
from plume.mind import (                                         # noqa: E402
    Cue, FormParams, MindParams, Mind, BarFeatures,
)
from plume.roles import (                                        # noqa: E402
    MAX_STEP_PCT_PER_BAR, PARAM_HELP, ROLE_TEMPO_POLICY, TEMPO_POLICIES,
    RoleNegotiator, RoleParams, RoleState, effective_bpm,
    resolve_tempo_policy,
)
from plume.sim import simulate_lead                              # noqa: E402

# The real-material harness (renders, the listening stack, the closed loop).
from test_roles import SR, SWING, Take, _render, _shift          # noqa: E402


# ------------------------------------------------------------------ builders


def mk_beat(*, bpm: float = 120.0, conf: float = 0.9, locked: bool = True,
            t: float = 0.0, bar: int = 0) -> BeatState:
    iv = 60.0 / bpm
    return BeatState(t_s=t, bpm=bpm, confidence=conf, locked=locked, bar=bar,
                     beat_in_bar=0, phase_beats=0.0, next_beat_t_s=t + iv,
                     next_downbeat_t_s=t + 4 * iv, beat_interval_s=iv)


def mk_harm(*, key: str | None = None, key_conf: float = 0.8) -> HarmonyState:
    return HarmonyState(t_s=0.0, key=key, key_conf=key_conf, chord=None,
                        chord_conf=0.0, register_low_midi=None,
                        register_high_midi=None)


def mk_bar(bar: int, *, dens: float = 1.0, silent: bool = False) -> BarFeatures:
    return BarFeatures(bar=bar, rms_db=-20.0, onset_density=dens,
                       silent=silent, register_low_midi=None,
                       register_high_midi=None)


def joiner(policy, *, role: str = "lead", offset_ms=31.0, bars: int = 8,
           bpm: float = 120.0, **kw) -> list[RoleState]:
    """A human who joins and sits ``offset_ms`` off the machine's grid.

    The offset goes through the EXPLICIT ``human_offset_ms`` path, so what is
    under test is the policy and not the phase estimator (which has its own
    tests in ``test_roles``).  ``policy`` may be ``None`` — the role's default,
    which is the case the change is actually about.
    """
    neg = RoleNegotiator(RoleParams(role=role, tempo_policy=policy, **kw),
                         composed_bpm=bpm)
    bar_s = 4.0 * 60.0 / bpm
    out = []
    for i in range(bars):
        out.append(neg.update(
            bar=i, beat=mk_beat(bpm=bpm, t=i * bar_s, bar=i),
            harmony=mk_harm(), human=mk_bar(i - 1), machine_grid_t_s=i * bar_s,
            human_offset_ms=(offset_ms(i) if callable(offset_ms)
                             else offset_ms)))
    return out


def tags(states) -> list[str]:
    return [e for s in states for e in s.events]


def concessions(states) -> list[RoleState]:
    return [s for s in states if any(e.startswith("concede")
                                     or e == "tempo_match" for e in s.events)]


def tape(bars: int, *, policy, lead_ms: float = 30.0, human_rate: float = 0.0,
         composed: float = 120.0, **kw):
    """The v2.1 closed loop: the machine's grid advances at the tempo it chose.

    ``human_rate = 0`` is A PARTNER WHO CANNOT BEND — a fixed recording — which
    is the case ``test_roles.test_a_partner_who_cannot_bend_is_the_case_that_hunts``
    documents as the meet law's residual, and the case the decision doc says
    should simply not arise under hold.
    """
    neg = RoleNegotiator(RoleParams(role="lead", tempo_policy=policy, **kw),
                         composed_bpm=composed)
    t, human_t = 0.0, -lead_ms / 1000.0
    human_bar = 4.0 * 60.0 / composed
    unit_s = 60.0 / composed / 4.0
    offs, states = [], []
    for bar in range(bars):
        off = (t - human_t) * 1000.0
        st = neg.update(bar=bar, beat=mk_beat(t=t, bar=bar), harmony=mk_harm(),
                        human=mk_bar(bar - 1), machine_grid_t_s=t,
                        human_offset_ms=off, human_offset_unit_s=unit_s)
        offs.append(off)
        states.append(st)
        t += 4.0 * 60.0 / effective_bpm(composed, st.tempo_target_pct)
        human_t += human_bar + human_rate * off / 1000.0
    return offs, states


# ----------------------------------------------------- the policy vocabulary


def test_the_three_policies_are_the_decision_docs_three():
    assert TEMPO_POLICIES == ("hold", "meet", "chase")


def test_role_defaults_map_to_the_decision_docs_table():
    """lead -> hold, follow -> chase, trade -> meet, solo_support -> meet."""
    assert ROLE_TEMPO_POLICY == {"lead": "hold", "follow": "chase",
                                 "trade": "meet", "solo_support": "meet"}
    for role, want in ROLE_TEMPO_POLICY.items():
        assert RoleParams(role=role).effective_tempo_policy == want
        assert resolve_tempo_policy(role, None) == want
        # ...and the negotiator agrees with the params it was built from.
        assert RoleNegotiator(RoleParams(role=role)).tempo_policy == want


def test_the_lead_default_really_is_the_new_one_not_the_old_one():
    """The whole change in one assertion: lead mode no longer concedes.

    ``RoleParams(role="lead")`` is what ``simulate_lead`` and live lead mode
    construct when the caller says nothing.
    """
    assert RoleParams(role="lead").tempo_policy is None      # unset
    assert RoleParams(role="lead").effective_tempo_policy == "hold"


def test_an_explicit_policy_overrides_the_roles_default():
    for role in ROLE_TEMPO_POLICY:
        for pol in TEMPO_POLICIES:
            assert RoleParams(role=role,
                              tempo_policy=pol).effective_tempo_policy == pol


@pytest.mark.parametrize("bad", ["holt", "HOLD", "follow", "", "lead", "none"])
def test_an_unknown_policy_errors_listing_the_valid_ones(bad):
    with pytest.raises(ValueError) as e:
        RoleParams(role="lead", tempo_policy=bad)
    msg = str(e.value)
    assert "tempo_policy" in msg
    for good in TEMPO_POLICIES:
        assert good in msg, f"the error must list {good!r}: {msg}"
    assert repr(bad) in msg


def test_the_param_is_documented_for_plume_roles():
    """``PARAM_HELP`` is what `plume roles` prints; an undocumented dial is a
    dial nobody can find."""
    assert "tempo_policy" in PARAM_HELP
    h = PARAM_HELP["tempo_policy"]
    for good in TEMPO_POLICIES:
        assert good in h


def test_the_policy_is_reported_on_every_state():
    for pol in TEMPO_POLICIES:
        sts = joiner(pol)
        assert {s.tempo_policy for s in sts} == {pol}


def test_an_unset_policy_follows_the_chair_but_an_explicit_one_is_a_pin():
    """A lead that yields is a follower; it should stop holding a clock it no
    longer owns.  An explicit policy is the caller's decision and never moves.
    """
    p = RoleParams(role="lead", handoff_hysteresis_bars=2, handoff_density=2.0)
    neg = RoleNegotiator(p, composed_bpm=120.0)
    assert neg.tempo_policy == "hold"
    for i in range(12):                      # a human who takes over
        neg.update(bar=i, beat=mk_beat(t=i * 2.0, bar=i), harmony=mk_harm(),
                   human=mk_bar(i - 1, dens=1.0 if i < 2 else 6.0),
                   machine_grid_t_s=i * 2.0, human_offset_ms=0.0)
    assert neg._role == "follow", "the setup never yielded; the test is vacuous"
    assert neg.tempo_policy == "chase"

    pinned = RoleNegotiator(RoleParams(role="lead", tempo_policy="hold",
                                       handoff_hysteresis_bars=2,
                                       handoff_density=2.0),
                            composed_bpm=120.0)
    for i in range(12):
        pinned.update(bar=i, beat=mk_beat(t=i * 2.0, bar=i), harmony=mk_harm(),
                      human=mk_bar(i - 1, dens=1.0 if i < 2 else 6.0),
                      machine_grid_t_s=i * 2.0, human_offset_ms=0.0)
    assert pinned._role == "follow"
    assert pinned.tempo_policy == "hold"


# ------------------------------------------------------- 1. hold really holds


@pytest.mark.parametrize("off", [31.0, -31.0, 45.0, -45.0, 12.0])
def test_a_rushing_or_dragging_joiner_draws_no_concession_under_hold(off):
    sts = joiner(None, offset_ms=off, bars=40)
    assert concessions(sts) == []
    assert all(s.tempo_offset_pct == 0.0 for s in sts)
    assert all(s.tempo_target_pct == 0.0 for s in sts)
    assert all(s.bpm == 120.0 for s in sts)


@pytest.mark.parametrize("off", [31.0, -31.0, 45.0, -45.0])
def test_NON_VACUITY_the_same_take_under_meet_really_does_concede(off):
    """The test above proves nothing unless meet moves on the same input."""
    sts = joiner("meet", offset_ms=off, bars=40)
    assert concessions(sts), f"meet never conceded to {off}ms — no separation"
    assert sts[-1].tempo_offset_pct != 0.0
    assert math.copysign(1.0, sts[-1].tempo_offset_pct) == math.copysign(1.0, off)


def test_hold_and_meet_are_told_apart_by_the_events_not_only_the_numbers():
    held = joiner(None, offset_ms=31.0, bars=6)
    met = joiner("meet", offset_ms=31.0, bars=6)
    assert "tempo_hold" in tags(held) and "concede" not in tags(held)
    assert "concede" in tags(met) and "tempo_hold" not in tags(met)


def test_a_partner_who_cannot_bend_produces_EXACTLY_zero_drift_under_hold():
    """v2.1's documented residual does not arise under the new default.

    ``test_roles.test_a_partner_who_cannot_bend_is_the_case_that_hunts`` says
    of the meet law: "against a partner who cannot bend, the tempo curve is a
    bounded hunt rather than a settle".  The decision doc's claim is stronger
    than "smaller" — under hold the number is 0.0, because nothing ever writes
    it.  Asserted as an exact equality on purpose: a tolerance here would let
    a hold that leaks a little pass as a hold.
    """
    offs_hold, hold = tape(48, policy=None)
    offs_meet, meet = tape(48, policy="meet")

    assert [s.tempo_offset_pct for s in hold] == [0.0] * 48
    assert [s.tempo_target_pct for s in hold] == [0.0] * 48
    assert {s.bpm for s in hold} == {120.0}

    # Non-vacuity: the same tape really is the case that hunts under meet.
    assert max(abs(s.tempo_offset_pct) for s in meet) > 0.5, \
        "meet did not hunt on the tape — the comparison is empty"
    assert max(abs(x) for x in offs_meet[8:]) > 0.0
    # And the machine's own grid stayed put, which is what a player leans on:
    # under hold the human's offset walks by exactly their own tempo error and
    # nothing else, so the drift is THEIRS and it is legible.
    steps = [b - a for a, b in zip(offs_hold, offs_hold[1:])]
    assert max(steps) - min(steps) < 1e-9, \
        "the machine's grid moved under hold: the offsets are not a straight ramp"


def test_hold_never_moves_the_tempo_however_extreme_the_human_is():
    """Sweep the shapes that move the meet law: a steady push, a ramp (the
    drift term), a saturating push (the cap), and an alternator."""
    shapes = {
        "steady": 45.0,
        "ramp": lambda i: -60.0 + 14.0 * i,
        "saturating": 60.0,
        "alternating": lambda i: 40.0 if i % 2 else -40.0,
    }
    for name, stim in shapes.items():
        held = joiner(None, offset_ms=stim, bars=40)
        assert all(s.tempo_offset_pct == 0.0 for s in held), name
        assert concessions(held) == [], name


# --------------------------------------------------- 2. hold is not deaf


def test_hold_still_measures_and_logs_the_offset_every_single_bar():
    sts = joiner(None, offset_ms=28.0, bars=12)
    # Bar 0 has no completed human bar in the real caller, but this harness
    # feeds one from the start, so every bar carries a measurement.
    assert all(s.human_offset_ms == pytest.approx(28.0) for s in sts)
    assert all("tempo_hold" in s.events for s in sts)
    for s in sts:
        assert s.reason.startswith("holding: human +28ms (policy=hold)"), s.reason


def test_the_logged_offset_carries_the_sign_and_the_size():
    """It is the practice-feedback signal, so the number has to be usable."""
    for off, txt in ((28.0, "+28"), (-28.0, "-28"), (0.0, "+0"), (4.0, "+4")):
        s = joiner(None, offset_ms=off, bars=3)[-1]
        assert s.human_offset_ms == pytest.approx(off)
        assert f"holding: human {txt}ms (policy=hold)" == s.reason


def test_meet_reports_the_same_measurement_hold_does():
    """The measurement is not a hold-only feature; it is the same number the
    concede rule always had, now published."""
    held = joiner(None, offset_ms=31.0, bars=6)
    met = joiner("meet", offset_ms=31.0, bars=6)
    assert [s.human_offset_ms for s in held] == [s.human_offset_ms for s in met]


def test_no_measurement_is_None_and_never_zero():
    """"No evidence" and "dead on the grid" must never be the same value.

    Alternating silence, because a bar nobody played in is the commonest way
    to have no evidence and the one a real take produces constantly.
    """
    neg = RoleNegotiator(RoleParams(role="lead"), composed_bpm=120.0)
    sts = [neg.update(bar=i, beat=mk_beat(t=i * 2.0, bar=i),
                      harmony=mk_harm(),
                      human=mk_bar(i - 1, silent=(i % 2 == 0)),
                      machine_grid_t_s=i * 2.0, human_offset_ms=30.0)
           for i in range(8)]
    assert [s.human_offset_ms for s in sts[1::2]] == [30.0] * 4
    assert all(s.human_offset_ms is None for s in sts[0::2])
    assert all("tempo_hold" not in s.events for s in sts[0::2])
    assert all("tempo_hold" in s.events for s in sts[1::2])


def test_a_silent_human_reports_no_offset_and_no_hold_line():
    sts = joiner(None, offset_ms=30.0, bars=6,
                 role="lead")
    assert all(s.human_offset_ms is not None for s in sts)
    neg = RoleNegotiator(RoleParams(role="lead"), composed_bpm=120.0)
    quiet = [neg.update(bar=i, beat=mk_beat(t=i * 2.0, bar=i),
                        harmony=mk_harm(), human=mk_bar(i - 1, silent=True),
                        machine_grid_t_s=i * 2.0, human_offset_ms=30.0)
             for i in range(6)]
    assert all(s.human_offset_ms is None for s in quiet)
    assert "tempo_hold" not in tags(quiet)


def test_hold_does_not_publish_an_offset_the_negotiator_does_not_believe():
    """The plausibility gates come FIRST; a hold is not an excuse to report a
    wrapped phase estimate as practice feedback."""
    p = RoleParams(role="lead")
    iv_ms = 60.0 / 120.0 * 1000.0
    wild = joiner(None, offset_ms=0.45 * iv_ms, bars=8)      # past the bound
    assert all(s.human_offset_ms is None for s in wild)
    assert "tempo_hold" not in tags(wild)
    ok = joiner(None, offset_ms=0.9 * p.concede_plausible_frac * iv_ms, bars=8)
    assert all(s.human_offset_ms is not None for s in ok)


def test_an_untrusted_beat_reports_nothing_under_hold_either():
    neg = RoleNegotiator(RoleParams(role="lead"), composed_bpm=120.0)
    sts = [neg.update(bar=i, beat=mk_beat(conf=0.2, t=i * 2.0, bar=i),
                      harmony=mk_harm(), human=mk_bar(i - 1),
                      machine_grid_t_s=i * 2.0, human_offset_ms=45.0)
           for i in range(8)]
    assert all(s.human_offset_ms is None for s in sts)


def test_holding_does_not_suppress_the_other_four_behaviors():
    """Only DRIFT is gated.  Yielding, taking the lead and conforming are not
    tempo decisions and must be untouched."""
    p = RoleParams(role="lead", handoff_hysteresis_bars=2, handoff_density=2.0)
    neg = RoleNegotiator(p, composed_bpm=120.0, composed_key="F minor")
    sts = [neg.update(bar=i, beat=mk_beat(t=i * 2.0, bar=i),
                      harmony=mk_harm(key="F# minor"),
                      human=mk_bar(i - 1, dens=1.0 if i < 2 else 6.0),
                      machine_grid_t_s=i * 2.0, human_offset_ms=31.0)
           for i in range(14)]
    assert "yield" in tags(sts), "the yield rule stopped working under hold"
    assert "conform" in tags(sts), "the conform rule stopped working under hold"
    assert all(s.tempo_offset_pct == 0.0 for s in sts)


# ------------------------------------------- 3. hold still follows INTENT


ARRANGEMENT = "lofi 90 bpm in A minor"


def _cue_bpms(tmp_path, text, *, policy, name, human=None, bars=16):
    res = simulate_lead(
        human, prompt=ARRANGEMENT, out_path=str(tmp_path / f"{name}.wav"),
        seed=3, bars=bars, role=RoleParams(role="lead", tempo_policy=policy),
        cues=[(4.0, Cue(text, at="next-bar"))])
    return res.machine_bpm_by_bar


@pytest.mark.parametrize("text,want", [
    ("ritard", "down"),
    ("faster", "up"),
    ("slower", "down"),
    ("half time", "down"),
    ("double time", "up"),
])
def test_an_explicit_tempo_cue_STILL_CHANGES_THE_TEMPO_under_hold(tmp_path,
                                                                 text, want):
    """THE test this change could most plausibly have broken.

    Drift-conceding and a deliberate tempo change are different subsystems,
    and the split is architectural: ``roles.py`` writes ``tempo_offset_pct``,
    while a cue is a separate multiplier folded in downstream
    (``sim._LeadCues.bar_ratio()`` -> ``m_bpm = rstate.bpm * ratio``).  An
    implementation that gated "the machine's bpm moves" rather than "the
    machine concedes to drift" would pass every hold test above and silently
    make the machine unable to end a tune.  So this is measured END TO END, on
    a real render, through the real ``simulate_lead``.
    """
    bpms = _cue_bpms(tmp_path, text, policy=None, name=f"hold_{want}_{text}")
    base = bpms[0]
    after = bpms[-1]
    assert base == pytest.approx(90.0)
    if want == "down":
        assert after < base * 0.99, f"{text!r} did not slow the machine: {bpms}"
    else:
        assert after > base * 1.01, f"{text!r} did not speed the machine: {bpms}"


@pytest.mark.parametrize("text", ["ritard", "faster", "slower", "half time",
                                  "double time"])
def test_a_cue_lands_IDENTICALLY_under_hold_and_under_meet(tmp_path, text):
    """The policy governs drift only, so with no human in the room the two
    policies must produce the same tempo curve, bar for bar."""
    a = _cue_bpms(tmp_path, text, policy=None, name=f"h_{text}")
    b = _cue_bpms(tmp_path, text, policy="meet", name=f"m_{text}")
    assert a == b, f"{text!r}: hold {a} != meet {b}"


def test_a_ritard_cue_still_lands_with_a_real_human_rushing_underneath(
        tmp_path, takes_local):
    """The case the two subsystems could actually collide in: a human who IS
    drifting, and a cue that says slow down anyway.  The machine must ignore
    their drift and honour the instruction."""
    tk = takes_local["trap140_rush"]
    wav = tmp_path / "human.wav"
    import soundfile as sf
    sf.write(str(wav), tk.x, SR)
    bpms = _cue_bpms(tmp_path, "ritard", policy=None, name="ritard_human",
                     human=str(wav), bars=16)
    assert bpms[0] == pytest.approx(90.0)
    assert bpms[-1] < 90.0 * 0.9, f"the ritard did not land: {bpms}"
    # ...and the drop is monotone, i.e. it is the ritardando and not a hunt.
    tail = bpms[4:]
    assert all(b <= a + 1e-9 for a, b in zip(tail, tail[1:])), bpms


def test_a_DETECTED_ritardando_still_ends_the_tune_under_hold():
    """``mind.Mind._form_track_tempo`` is the ending detector, and it reads
    ``BeatState.bpm`` — the human's tempo — not the negotiator's concession.
    Holding must not silence it.  Driven with a hold ``RoleState`` so the form
    authority the mind sees is exactly what lead mode now produces.
    """
    from plume.compose import PromptSpec, compose

    session = compose(PromptSpec(styles=["lofi"], bpm=90.0, key="A minor",
                                 movements=1, mood=[], name="rit"), seed=3)
    fp = FormParams()

    def run(policy):
        mind = Mind(MindParams(), fp)
        neg = RoleNegotiator(RoleParams(role="lead", tempo_policy=policy),
                             composed_bpm=90.0)
        actions = []
        bpm = 90.0
        for i in range(10):
            # A human slowing a few percent a bar: a ritardando, not a
            # tracker re-read (no octave change, each step well inside
            # ``ritard_step_max_pct``).
            if i >= 2:
                bpm *= 0.97
            beat = mk_beat(bpm=bpm, t=i * 2.6, bar=i)
            human = mk_bar(i - 1)
            rs = neg.update(bar=i, beat=beat, harmony=mk_harm(), human=human,
                            machine_grid_t_s=i * 2.6, human_offset_ms=31.0)
            fplan = mind.plan_form(rs, beat, mk_harm(), human, session)
            actions.append((fplan.action, fplan.reason))
        return actions

    held = run(None)
    assert any(a == "end" for a, _ in held), \
        f"the ritard detector went silent under hold: {held}"
    end_reason = next(r for a, r in held if a == "end")
    assert "ritardando" in end_reason
    # Non-vacuity in the other direction: meet reaches the same ending, so the
    # detector is not being carried by the policy either way.
    assert any(a == "end" for a, _ in run("meet"))


def test_the_ending_detector_and_the_concede_rule_are_separate_subsystems():
    """A structural check, so a future refactor cannot quietly re-couple them.

    ``roles.py`` must not know about cues at all: if the intent path ever
    starts running through ``tempo_offset_pct``, gating drift would gate
    intent with it, and every behavioural test above would still pass on the
    day the machine lost the ability to end a tune.
    """
    src = (Path(__file__).resolve().parents[1]
           / "src" / "plume" / "roles.py").read_text()
    body = src.split('"""', 2)[-1]           # skip the module docstring
    for token in ("parse_cue", "Cue", "ritard_bars", "_LeadCues"):
        assert token not in body, \
            f"roles.py now references {token!r}: intent and drift have merged"


# --------------------------------------------------- 4. chase, and v1 follow


def test_chase_is_the_default_for_follow_and_leaves_the_tempo_alone():
    """v1 follow semantics: the human owns the clock, and the caller's phase
    correction is what follows them — the negotiator holds no tempo of its
    own, because a concession layered on top would be a second, slower copy
    of the same correction fighting the first."""
    sts = joiner(None, role="follow", offset_ms=45.0, bars=20)
    assert {s.tempo_policy for s in sts} == {"chase"}
    assert all(s.tempo_offset_pct == 0.0 for s in sts)
    assert concessions(sts) == []
    # ...and it stays quiet: chase is not a decision the machine makes each bar.
    assert "tempo_hold" not in tags(sts)
    # The measurement is still published — the mirror works in every chair.
    assert all(s.human_offset_ms == pytest.approx(45.0) for s in sts)


def test_chase_is_unchanged_from_the_pre_v2_2_follow_behavior():
    """Before v2.2, ``role="follow"`` was inert because its clock authority
    (0.15) sits under ``CONCEDE_MIN_CLOCK``.  ``chase`` must be exactly that,
    not a new behavior wearing the old name — including the authority dials,
    which are what the caller's phase corrector actually reads.
    """
    sts = joiner(None, role="follow", offset_ms=45.0, bars=20)
    assert sts[-1].clock_authority == pytest.approx(0.15)
    assert sts[-1].form_authority == pytest.approx(0.0)
    assert all(s.reason.startswith("follow: clock 0.") for s in sts)


def test_chase_in_the_lead_chair_hands_the_clock_over_explicitly():
    """A caller who asks lead mode to chase gets a machine that writes no
    tempo of its own — the same contract as follow, said on purpose rather
    than fallen into via the authority gate."""
    sts = joiner("chase", role="lead", offset_ms=45.0, bars=20)
    assert all(s.tempo_offset_pct == 0.0 for s in sts)
    assert concessions(sts) == []
    assert sts[-1].clock_authority == pytest.approx(0.9)   # dials untouched


def test_trade_and_solo_support_still_meet():
    """Trading is a conversation between two clocks, so it entrains."""
    for role in ("trade", "solo_support"):
        sts = joiner(None, role=role, offset_ms=45.0, bars=40)
        assert {s.tempo_policy for s in sts} == {"meet"}
        assert concessions(sts), f"{role} stopped entraining"


# ---------------------------------------------------------- the CLI override


def test_the_roles_override_parses_and_reaches_the_negotiator():
    """``--roles tempo_policy=meet`` is the documented per-session escape."""
    import plume.cli as cli

    p = cli._parse_role_overrides(["tempo_policy=meet"], "lead")
    assert p.role == "lead" and p.tempo_policy == "meet"
    assert RoleNegotiator(p).tempo_policy == "meet"

    for word in ("none", "default", "auto"):
        q = cli._parse_role_overrides([f"tempo_policy={word}"], "lead")
        assert q.tempo_policy is None
        assert q.effective_tempo_policy == "hold"

    with pytest.raises(cli.CliError) as e:
        cli._parse_role_overrides(["tempo_policy=holt"], "lead")
    for good in TEMPO_POLICIES:
        assert good in str(e.value)


def test_the_param_shows_up_in_the_roles_table():
    import plume.cli as cli

    rows = {r["name"]: r for r in cli._role_rows()}
    assert "tempo_policy" in rows
    assert rows["tempo_policy"]["type"] == "str?"
    assert rows["tempo_policy"]["default"] is None


# ------------------------------------------------------------- determinism


def test_determinism():
    def run(policy):
        offs, sts = tape(24, policy=policy, human_rate=0.25)
        return [(s.bar, s.role, s.tempo_offset_pct, s.tempo_target_pct,
                 s.human_offset_ms, s.tempo_policy, s.events, s.reason)
                for s in sts]

    for pol in (None, "meet", "chase"):
        assert run(pol) == run(pol)


def test_no_bar_moves_the_tempo_past_the_step_cap_under_any_policy():
    for pol in (None, "meet", "chase"):
        for off in (45.0, -45.0, 80.0):
            sts = joiner(pol, offset_ms=off, bars=40)
            steps = [abs(b.tempo_offset_pct - a.tempo_offset_pct)
                     for a, b in zip(sts, sts[1:])]
            assert max(steps, default=0.0) <= MAX_STEP_PCT_PER_BAR + 1e-9


# ------------------------------------------------------------ real material
#
# Everything above this line is a fixture.  This project's recurring failure is
# code that passes fixtures and fails on real music, so the same claims are
# re-asserted on plume's OWN v0 renders, at two gains, through the real
# listening stack and the real closed loop.


@pytest.fixture(scope="module")
def takes_local(tmp_path_factory):
    d = tmp_path_factory.mktemp("tempo_policy_real")
    out = {}
    for style, bpm, key in (("trap", 140.0, "D minor"),
                            ("lofi", 80.0, "C major")):
        for gain in (0.0, -20.0):
            n = f"{style}{bpm:.0f}{'' if gain == 0 else '_quiet'}"
            out[n] = Take(n, _render(d / f"{n}.wav", style, bpm, key,
                                     gain_db=gain), bpm, key,
                          swing=SWING[style])
    # A real take pushed 30 ms early: the only ground truth about timing a
    # recording can be made to carry, and the input the meet law moves on.
    for stem, bpm, style in (("trap140", 140.0, "trap"),
                             ("lofi80", 80.0, "lofi")):
        for gsfx in ("", "_quiet"):
            base = out[stem + gsfx].x
            out[f"{stem}{gsfx}_rush"] = Take(f"{stem}{gsfx}_rush",
                                             _shift(base, -30.0), bpm, None,
                                             swing=SWING[style])
    return out


REAL = ("trap140", "trap140_quiet", "lofi80", "lofi80_quiet",
        "trap140_rush", "trap140_quiet_rush", "lofi80_rush",
        "lofi80_quiet_rush")


@pytest.mark.parametrize("name", REAL)
def test_a_REAL_human_moves_the_tempo_by_exactly_nothing_under_hold(takes_local,
                                                                   name):
    tk = takes_local[name]
    sts = tk.closed_loop(RoleParams(role="lead"), composed_bpm=tk.bpm,
                         measured_offset=True)
    assert len(sts) > 8, f"{name}: the loop barely ran"
    assert [s.tempo_offset_pct for s in sts] == [0.0] * len(sts), \
        f"{name}: hold leaked a concession"
    assert all(s.bpm == pytest.approx(tk.bpm) for s in sts)
    assert not [e for e in tags(sts) if e.startswith("concede")
                or e == "tempo_match"]
    # The policy tracks the chair, and on this material the chair really does
    # change (see the test below) — so what must hold is that it is never
    # ``meet``, not that it is always ``hold``.
    assert {s.tempo_policy for s in sts} <= {"hold", "chase"}
    for s in sts:
        assert s.tempo_policy == ("hold" if s.role == "lead" else "chase")


def test_a_REAL_render_that_takes_the_lead_moves_the_policy_with_the_chair(
        takes_local):
    """MEASURED, and it is why the test above cannot just assert ``hold``.

    ``test_roles`` records that plume's own lo-fi 80 render RAMPS in onset
    density and legitimately makes the machine yield the lead.  An unset
    ``tempo_policy`` follows the chair, so on that take the machine holds
    while it is leading and hands the clock over the moment it is not — which
    is the intended behavior, and this pins it so a future refactor cannot
    silently freeze the policy at construction.
    """
    tk = takes_local["lofi80"]
    sts = tk.closed_loop(RoleParams(role="lead"), composed_bpm=tk.bpm,
                         measured_offset=True)
    roles_seen = [s.role for s in sts]
    assert "follow" in roles_seen, \
        "the lo-fi render stopped yielding; this test no longer tests anything"
    assert [s.tempo_policy for s in sts] == \
        ["hold" if r == "lead" else "chase" for r in roles_seen]
    # ...and neither chair ever concedes.
    assert [s.tempo_offset_pct for s in sts] == [0.0] * len(sts)


@pytest.mark.parametrize("name", REAL)
def test_a_REAL_human_is_still_MEASURED_under_hold(takes_local, name):
    """The mirror has to work on real audio, not just on a scalar fixture."""
    tk = takes_local[name]
    sts = tk.closed_loop(RoleParams(role="lead"), composed_bpm=tk.bpm,
                         measured_offset=True)
    seen = [s for s in sts if s.human_offset_ms is not None]
    assert len(seen) >= 4, \
        f"{name}: only {len(seen)} of {len(sts)} bars produced a measurement"
    assert all(abs(s.human_offset_ms) < 200.0 for s in seen)
    held = [s for s in seen if s.tempo_policy == "hold"]
    assert held, f"{name}: never measured anything while actually leading"
    assert all("tempo_hold" in s.events for s in held)
    assert all("holding: human" in s.reason for s in held)


def test_NON_VACUITY_meet_really_does_move_on_this_real_material(takes_local):
    """If meet were inert here too, every real-material hold assertion above
    would be measuring the listening stack rather than the policy."""
    moved = []
    for name in REAL:
        tk = takes_local[name]
        sts = tk.closed_loop(RoleParams(role="lead", tempo_policy="meet"),
                             composed_bpm=tk.bpm, measured_offset=True)
        if any(s.tempo_offset_pct != 0.0 for s in sts):
            moved.append((name, max(abs(s.tempo_offset_pct) for s in sts)))
    assert moved, ("meet moved on NONE of the real takes — the hold tests "
                   "above prove nothing about the policy")


@pytest.mark.parametrize("stem", ("trap140", "lofi80"))
def test_hold_is_BIT_IDENTICAL_at_two_gains_on_real_material(takes_local, stem):
    """Same performance, 20 dB apart: the negotiation must not notice.

    The reason string carries the offset ROUNDED TO THE MILLISECOND, which is
    the resolution a human reads it at, so it is compared exactly.  The raw
    ``human_offset_ms`` is compared to a tolerance for the reason
    ``test_roles.test_the_negotiation_is_BIT_IDENTICAL_at_two_gains``
    documents and measures: ``BeatTracker``'s bpm is fader-invariant only to
    ~1e-10 relative, and it reaches the derived estimator on bars too sparse
    for the onset-phase one.
    """
    def run(name):
        tk = takes_local[name]
        return tk.closed_loop(RoleParams(role="lead"), composed_bpm=tk.bpm,
                              measured_offset=True)

    a, b = run(stem), run(stem + "_quiet")
    assert [(s.tempo_offset_pct, s.role, s.events, s.reason) for s in a] == \
           [(s.tempo_offset_pct, s.role, s.events, s.reason) for s in b]
    for x, y in zip(a, b):
        assert (x.human_offset_ms is None) == (y.human_offset_ms is None)
        if x.human_offset_ms is not None:
            assert x.human_offset_ms == pytest.approx(y.human_offset_ms,
                                                      abs=1e-3)


def test_END_TO_END_a_real_joiner_never_moves_the_machines_clock(tmp_path,
                                                                takes_local):
    """Through ``simulate_lead`` itself, which is the loop that ships.

    A real 32-bar trap render pushed 30 ms early, joining a lo-fi arrangement:
    under the new default the machine's bpm is the composed bpm on every bar,
    and under ``meet`` it is not.
    """
    import soundfile as sf

    wav = tmp_path / "rushing_human.wav"
    sf.write(str(wav), takes_local["trap140_rush"].x, SR)

    def run(policy, name):
        return simulate_lead(
            str(wav), prompt=ARRANGEMENT, out_path=str(tmp_path / f"{name}.wav"),
            seed=3, bars=24,
            role=RoleParams(role="lead", tempo_policy=policy))

    held = run(None, "e2e_hold")
    assert [r["tempo_offset_pct"] for r in held.role_states] == \
        [0.0] * len(held.role_states)
    assert all(b == pytest.approx(90.0) for b in held.machine_bpm_by_bar)
    assert any("holding: human" in r["reason"] for r in held.role_states), \
        "the take never produced a measurement, so it proves nothing"

    met = run("meet", "e2e_meet")
    assert any(r["tempo_offset_pct"] != 0.0 for r in met.role_states), \
        "meet was inert end to end — no separation between the policies"


def test_END_TO_END_the_default_take_is_a_perfect_grid(tmp_path, takes_local):
    """The product claim, in the rendered timeline rather than the log.

    Under hold every machine bar is exactly ``240 / bpm`` seconds long, so the
    grid a player leans on is real: the take's bar times are an arithmetic
    sequence to the sample.  This is the number v2.0 could not produce — it
    ended a take 2.5 beats away from the human it was following.
    """
    import soundfile as sf

    wav = tmp_path / "grid_human.wav"
    sf.write(str(wav), takes_local["trap140_rush"].x, SR)
    res = simulate_lead(str(wav), prompt=ARRANGEMENT,
                        out_path=str(tmp_path / "grid.wav"), seed=3, bars=24,
                        role=RoleParams(role="lead"))
    bar_s = 240.0 / 90.0
    for k, t in enumerate(res.bar_times_s):
        assert t == pytest.approx(k * bar_s, abs=1e-3), \
            f"bar {k} landed at {t:.4f}s, not {k * bar_s:.4f}s"
