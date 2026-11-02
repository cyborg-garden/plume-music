"""Tests for plume.roles — role negotiation (SPEC-V2 sections 0 and 1).

Two halves, and the second one is the point.

The first half is table-driven FSM coverage over hand-built ``BeatState`` /
``HarmonyState`` / ``BarFeatures``: every arc of the five behaviors, the concede
arithmetic down to the spec's own worked example, hysteresis, the ramp bound,
and determinism.

The second half (``--- real material ---``) drives the negotiator from plume's
OWN v0 renders, at two gains, through the real ``FeatureExtractor`` /
``BeatTracker`` / ``HarmonyTracker``, in a closed loop where the machine's grid
advances at the tempo the negotiator just chose.  It exists because v1 passed
all ten of its synthetic behavior tests and was completely unusable on real
music, and because writing this half found three defects that the synthetic half
cannot see even in principle:

* the ``BeatState``-derived offset carries +/-210ms of groove and drift noise on
  real audio and drove a -8% concession into the cap in the WRONG direction;
* real programmed material sits at 5.0 onsets/beat, so the spec's absolute yield
  threshold (2.0) handed the lead away on anything with hats;
* a real C-major render is reported as A minor — its relative — every frame, so
  a conform rule written on key tonics fires spuriously.

Each of those has a guard below AND a non-vacuity test that shows the guard is
load-bearing (disable it and the same input really does break).

Everything is synthesized in-test into ``tmp_path``; nothing reads ``renders/``
(those are gitignored, and a test that can silently skip itself is not a test).
"""
from __future__ import annotations

import dataclasses
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import plume.roles as roles_mod                                  # noqa: E402
from plume.compose import PromptSpec, compose                    # noqa: E402
from plume.engine import render                                  # noqa: E402
from plume.listen.beat import BeatState, BeatTracker             # noqa: E402
from plume.listen.features import FeatureExtractor               # noqa: E402
from plume.listen.harmony import HarmonyState, HarmonyTracker    # noqa: E402
from plume.mind import BarFeatures                               # noqa: E402
from plume.roles import (                                        # noqa: E402
    AUTHORITY_SLEW, MAX_STEP_PCT_PER_BAR, PARAM_HELP, ROLE_DEFAULTS, ROLES,
    RoleNegotiator, RoleParams, RoleState, collection_root, effective_bpm,
    grid_offset_ms, onset_phase_offset_ms, trade_turn,
)
from plume.sim import lead_analysis_gain                          # noqa: E402
from plume.theory import parse_key                               # noqa: E402

SR = 48000


# ------------------------------------------------------------------ builders


def mk_beat(*, bpm: float = 120.0, conf: float = 0.9, locked: bool = True,
            t: float = 0.0, next_beat: float | None = None,
            bar: int = 0) -> BeatState:
    iv = 60.0 / bpm if bpm > 0 else math.inf
    return BeatState(
        t_s=t, bpm=bpm, confidence=conf, locked=locked, bar=bar,
        beat_in_bar=0, phase_beats=0.0,
        next_beat_t_s=(t + iv) if next_beat is None else next_beat,
        next_downbeat_t_s=t + 4 * iv if locked else math.inf,
        beat_interval_s=iv,
    )


def mk_harm(*, key: str | None = None, key_conf: float = 0.8,
            t: float = 0.0) -> HarmonyState:
    return HarmonyState(t_s=t, key=key, key_conf=key_conf, chord=None,
                        chord_conf=0.0, register_low_midi=None,
                        register_high_midi=None)


def mk_bar(bar: int, *, dens: float = 1.0, silent: bool = False,
           rms: float = -20.0) -> BarFeatures:
    return BarFeatures(bar=bar, rms_db=rms, onset_density=dens, silent=silent,
                       register_low_midi=None, register_high_midi=None)


def drive(neg: RoleNegotiator, n_bars: int, *, offset_ms=0.0, dens=1.0,
          silent=False, key=None, key_conf: float = 0.8, conf: float = 0.9,
          locked: bool = True, bpm: float = 120.0,
          start_bar: int = 0) -> list[RoleState]:
    """Drive ``n_bars`` bars of a constant (or per-bar callable) stimulus.

    ``offset_ms``/``dens``/``silent``/``key`` may each be a scalar or a
    ``f(bar) -> value``.  The offset is fed through the EXPLICIT
    ``human_offset_ms`` path so the arithmetic under test is the concede rule's
    and not the phase estimator's; the derived path has its own tests.
    """
    def val(x, i):
        return x(i) if callable(x) else x

    bar_s = 4.0 * 60.0 / bpm
    out = []
    for i in range(start_bar, start_bar + n_bars):
        sil = bool(val(silent, i))
        out.append(neg.update(
            bar=i,
            beat=mk_beat(bpm=bpm, conf=val(conf, i), locked=val(locked, i),
                         t=i * bar_s, bar=i),
            harmony=mk_harm(key=val(key, i), key_conf=val(key_conf, i)),
            human=None if sil is None else mk_bar(i - 1, dens=val(dens, i),
                                                  silent=sil),
            machine_grid_t_s=i * bar_s,
            human_offset_ms=None if sil else val(offset_ms, i),
        ))
    return out


def LeadMeet(**kw) -> RoleParams:
    """``role="lead"`` under the MEET policy — what lead mode was before v2.2.

    v2.2 (``docs/decisions/2026-08-15-tempo-authority.md``) made ``"hold"`` the
    lead default: the machine is the reference and does not chase the human's
    timing ERRORS.  The concede rules did not change, they stopped being the
    default, so every test in this file that is about those rules names the
    policy they belong to.

    Written as a helper rather than sprinkled inline for one reason: "lead
    mode" and "the concede law" used to be the same sentence and are not any
    more, and a reader skimming this file should trip over that rather than
    over a keyword argument.  ``tests/test_tempo_policy.py`` owns the policy
    itself — what each one does, that the defaults are right, and that holding
    still measures, still logs, and still follows a ritardando.
    """
    kw.setdefault("tempo_policy", "meet")
    return RoleParams(role="lead", **kw)


def events_of(states, tag: str) -> list[RoleState]:
    return [s for s in states if tag in s.events]


# =========================================================== spec surface ===


def test_roles_and_defaults_are_the_spec_values():
    assert ROLES == ("lead", "follow", "trade", "solo_support")
    assert ROLE_DEFAULTS == {
        "lead": (0.9, 1.0), "follow": (0.15, 0.0),
        "trade": (0.5, 0.5), "solo_support": (0.6, 0.85),
    }
    assert set(ROLE_DEFAULTS) == set(ROLES)


def test_role_params_defaults_are_the_spec_values():
    p = RoleParams()
    assert (p.role, p.clock_authority, p.form_authority) == ("follow", None, None)
    assert (p.concede_ms, p.concede_bars, p.concede_rate, p.concede_max_pct) == \
        (25.0, 2, 0.25, 8.0)
    assert (p.handoff_silence_bars, p.handoff_density,
            p.handoff_hysteresis_bars, p.trade_bars) == (8, 2.0, 4, 4)
    assert (p.conform_semitones, p.conform_bars, p.conform_enabled) == (2, 4, True)


def test_param_help_documents_every_field_and_nothing_else():
    fields = {f.name for f in dataclasses.fields(RoleParams)}
    assert set(PARAM_HELP) == fields
    for k, v in PARAM_HELP.items():
        assert v and not v.endswith("."), k


def test_unset_authorities_take_the_role_default():
    for role, (clock, form) in ROLE_DEFAULTS.items():
        st = drive(RoleNegotiator(RoleParams(role=role)), 1)[0]
        if role == "trade":                    # bar 0 is the machine's unit
            clock, form = ROLE_DEFAULTS["lead"]
        assert (st.clock_authority, st.form_authority) == (clock, form), role
        assert st.role == role


def test_explicit_authorities_pin_the_starting_role():
    st = drive(RoleNegotiator(
        RoleParams(role="lead", clock_authority=0.6, form_authority=0.2)), 4)[-1]
    assert (st.clock_authority, st.form_authority) == (0.6, 0.2)


@pytest.mark.parametrize("kw", [
    {"role": "boss"}, {"clock_authority": 1.5}, {"form_authority": -0.1},
    {"concede_bars": 0}, {"concede_rate": 2.0}, {"concede_ms": -1.0},
    {"concede_max_pct": -1.0}, {"handoff_silence_bars": 0},
    {"handoff_density": 0.0}, {"handoff_hysteresis_bars": 0}, {"trade_bars": 0},
    {"conform_semitones": 7}, {"conform_bars": 0},
    {"concede_plausible_frac": 0.0}, {"concede_plausible_frac": 0.6},
    {"concede_spread_ms": -1.0}, {"handoff_density_ratio": 0.9},
])
def test_bad_params_are_rejected_loudly(kw):
    with pytest.raises(ValueError):
        RoleParams(**kw)


@pytest.mark.parametrize("kw", [{"composed_bpm": 0.0}, {"composed_bpm": -3.0},
                                {"composed_key": "H flat"}])
def test_bad_session_context_is_rejected_at_construction(kw):
    with pytest.raises(ValueError):
        RoleNegotiator(RoleParams(), **kw)


def test_role_state_is_frozen():
    st = drive(RoleNegotiator(RoleParams()), 1)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        st.role = "lead"


# ==================================================== offset measurement ===


@pytest.mark.parametrize("shift_ms,want", [
    (0.0, 0.0), (31.0, 31.0), (-35.0, -35.0), (120.0, 120.0), (-200.0, -200.0),
])
def test_grid_offset_sign_is_positive_when_the_human_is_ahead(shift_ms, want):
    """+ = rushing.  SPEC-V2's worked example fixes this: "human +31ms" makes
    the machine speed UP, so a human out in front must read positive."""
    iv = 0.5                                   # 120 bpm
    now = 10.0
    # A human ``shift_ms`` AHEAD has their next beat that much earlier.
    nb = now + iv - shift_ms / 1000.0
    got = grid_offset_ms(mk_beat(bpm=120.0, t=now, next_beat=nb), now)
    assert got == pytest.approx(want, abs=1e-6)


def test_grid_offset_wraps_by_the_machines_period_not_the_trackers():
    """A tracker on an octave alias must not shrink the window a real
    lay-back is measured in: wrapped by the tracker's 250ms, a genuine 200ms
    rush comes out as a 50ms DRAG."""
    now = 10.0
    nb = now + 0.5 - 0.2                       # 200ms ahead of a 120bpm grid
    beat = mk_beat(bpm=240.0, t=now, next_beat=nb)   # tracker on the octave
    assert grid_offset_ms(beat, now) == pytest.approx(-50.0, abs=1e-6)
    assert grid_offset_ms(beat, now, interval_s=0.5) == \
        pytest.approx(200.0, abs=1e-6)


@pytest.mark.parametrize("beat,grid", [
    (mk_beat(bpm=0.0, next_beat=math.inf), 1.0),
    (mk_beat(bpm=120.0), math.inf),
])
def test_no_grid_means_no_evidence_not_zero_evidence(beat, grid):
    assert grid_offset_ms(beat, grid) is None


def test_onset_phase_offset_recovers_a_known_shift():
    unit = 60.0 / 120.0 / 4.0                  # sixteenths
    base = np.arange(16) * unit
    for shift in (0.0, 0.020, -0.030):
        # human EARLY by `shift` -> onsets at t - shift -> offset +shift
        got = onset_phase_offset_ms(base - shift, 0.0, 16 * unit, unit)
        assert got == pytest.approx(shift * 1000.0, abs=1e-6)


def test_onset_phase_offset_needs_onsets():
    assert onset_phase_offset_ms([0.1, 0.2], 0.0, 1.0, 0.125) is None
    assert onset_phase_offset_ms([], 0.0, 1.0, 0.125) is None
    assert onset_phase_offset_ms([0.1, 0.2, 0.3], 0.0, 1.0, 0.0) is None


# =================================================== behavior 1: concede ===


def test_the_spec_worked_example_is_reproduced_verbatim():
    """SPEC-V2 section 1 quotes the log line; it is the contract, not prose."""
    neg = RoleNegotiator(LeadMeet(), composed_bpm=120.0)
    sts = drive(neg, 2, offset_ms=31.0)
    assert sts[1].events == ("concede",)
    assert sts[1].reason == \
        "conceding: human +31ms for 2 bars -> 120.0 -> 121.9 bpm"


def test_a_rushing_human_is_conceded_to_within_concede_bars():
    p = LeadMeet()
    sts = drive(RoleNegotiator(p, composed_bpm=120.0), 6, offset_ms=31.0)
    conc = events_of(sts, "concede")
    assert conc, "a rushing human drew no concession at all"
    assert conc[0].bar <= p.concede_bars, \
        f"first concession at bar {conc[0].bar}, concede_bars={p.concede_bars}"
    assert sts[-1].tempo_offset_pct > 0.0
    assert sts[-1].bpm > 120.0


def test_a_dragging_human_is_conceded_to_symmetrically():
    a = drive(RoleNegotiator(LeadMeet(), composed_bpm=120.0), 6,
              offset_ms=31.0)
    b = drive(RoleNegotiator(LeadMeet(), composed_bpm=120.0), 6,
              offset_ms=-31.0)
    assert events_of(b, "concede")
    # The FIRST concession is exactly mirrored.  Later ones cannot be: once the
    # tempo has moved, the beat period the next deviation is measured against
    # has moved with it, and that asymmetry is the mechanism, not a bug.
    assert b[1].tempo_target_pct == pytest.approx(-a[1].tempo_target_pct)
    assert b[-1].tempo_offset_pct < 0.0 < a[-1].tempo_offset_pct
    assert b[-1].bpm < 120.0 < a[-1].bpm
    assert b[1].reason.startswith("conceding: human -31ms for 2 bars -> 120.0 ->")


@pytest.mark.parametrize("stim", [
    0.0,                                            # dead on the grid
    lambda i: (8.0, -7.0, 6.0, -9.0)[i % 4],        # jitter, both signs
    lambda i: 12.0,                                 # a small CONSISTENT bias
    lambda i: 40.0 if i % 2 else -40.0,             # big, alternating
])
def test_an_on_grid_human_never_concedes(stim):
    """The jitter guard.  A player who is simply *there* must not make an
    unshakeable clock shakeable — that is the whole value of lead mode."""
    sts = drive(RoleNegotiator(LeadMeet(), composed_bpm=120.0),
                64, offset_ms=stim)
    assert not events_of(sts, "concede")
    assert all(s.tempo_offset_pct == 0.0 for s in sts)
    assert all(s.bpm == 120.0 for s in sts)


def test_inconsistent_same_sign_offsets_do_not_concede():
    """Two bars that merely fell on the same side are not "consistent"; their
    mean is a number neither bar ever measured.  (This is the guard that the
    real-material half proved necessary.)"""
    sts = drive(RoleNegotiator(LeadMeet()), 40,
                offset_ms=lambda i: 5.0 if i % 2 else 80.0)
    assert not events_of(sts, "concede")


def test_implausible_offsets_are_read_as_phase_error_not_timing():
    """Half a beat away is the next beat, not a human pushing."""
    p = LeadMeet()
    iv_ms = 60.0 / 120.0 * 1000.0
    sts = drive(RoleNegotiator(p, composed_bpm=120.0), 40,
                offset_ms=0.45 * iv_ms)
    assert not events_of(sts, "concede")
    # ...and the bound is where the parameter says it is.
    ok = drive(RoleNegotiator(p, composed_bpm=120.0), 4,
               offset_ms=0.9 * p.concede_plausible_frac * iv_ms)
    assert events_of(ok, "concede")


def test_a_permanently_rushing_human_saturates_at_the_cap_and_says_so():
    p = LeadMeet()
    sts = drive(RoleNegotiator(p, composed_bpm=120.0), 80, offset_ms=45.0)
    assert sts[-1].tempo_offset_pct == pytest.approx(p.concede_max_pct)
    assert events_of(sts, "concede_capped"), \
        "saturated silently — the log must say the cap is what stopped it"
    assert all(s.tempo_offset_pct <= p.concede_max_pct + 1e-9 for s in sts)
    capped = events_of(sts, "concede_capped")[0]
    assert "max 8.0%" in capped.reason and "120.0 bpm" in capped.reason


def test_no_bar_ever_moves_the_tempo_by_more_than_the_step_cap():
    """SPEC-V2 behavior 2: "ramped (no single-bar jump > 2%)"."""
    for off in (45.0, -45.0, 26.0, -80.0):
        sts = drive(RoleNegotiator(LeadMeet()), 80, offset_ms=off)
        steps = [abs(b.tempo_offset_pct - a.tempo_offset_pct)
                 for a, b in zip(sts, sts[1:])]
        assert max(steps, default=0.0) <= MAX_STEP_PCT_PER_BAR + 1e-9, \
            f"offset {off}: worst single-bar step {max(steps):.3f}%"


def test_a_concession_is_a_ramp_applied_over_the_FOLLOWING_bar():
    neg = RoleNegotiator(LeadMeet(), composed_bpm=120.0)
    sts = drive(neg, 4, offset_ms=31.0)
    ev = sts[1]
    assert ev.events == ("concede",)
    assert ev.tempo_offset_pct == 0.0, "the tempo JUMPED on the event bar"
    assert ev.tempo_target_pct == pytest.approx(1.55)
    assert sts[2].tempo_offset_pct == pytest.approx(1.55), "the ramp never landed"
    assert sts[2].bpm == pytest.approx(120.0 * 1.0155)


def _entrain(bars: int, *, lead_ms: float, human_rate: float,
             composed: float = 120.0, **kw) -> tuple[list[float], list[RoleState]]:
    """Close the loop for real: the machine's grid advances at the tempo the
    negotiator just chose, and the human absorbs ``human_rate`` of the gap too.

    ``human_rate = 0`` is a tape — a fixed recording that cannot meet anybody.
    """
    neg = RoleNegotiator(LeadMeet(**kw), composed_bpm=composed)
    t, human_t, human_bar = 0.0, -lead_ms / 1000.0, 4 * 60.0 / composed
    offs, states = [], []
    # The offset here is a PERFECT sixteenth-grid measurement — that is what
    # this harness is modelling — so it says so.  Without the subdivision the
    # negotiator has to fall back on the coarse beat-fraction bound, and a
    # phase that runs past it (a tape does exactly that) goes UNREADABLE, which
    # is the divergence this loop exists to measure rather than to hide.
    unit_s = 60.0 / composed / 4.0
    for bar in range(bars):
        off = (t - human_t) * 1000.0
        st = neg.update(bar=bar, beat=mk_beat(t=t, bar=bar), harmony=mk_harm(),
                        human=mk_bar(bar - 1), machine_grid_t_s=t,
                        human_offset_ms=off, human_offset_unit_s=unit_s)
        offs.append(off)
        states.append(st)
        t += 4 * 60.0 / effective_bpm(composed, st.tempo_target_pct)
        human_t += human_bar + human_rate * off / 1000.0
    return offs, states


def test_mutual_entrainment_settles_inside_the_tolerance():
    """SPEC-V2 section 0's actual claim: "both parties bend a little".

    It settles into the dead zone, not onto zero, and that is the design: the
    rule has a ``concede_ms`` tolerance precisely so that neither player chases
    the last few milliseconds of the other.  What must hold is that a 30ms gap
    stops being a 30ms gap and never grows into a bar line of its own.
    """
    p = RoleParams()
    offs, sts = _entrain(48, lead_ms=30.0, human_rate=0.25)
    assert offs[0] == pytest.approx(30.0)
    assert events_of(sts, "concede"), "nobody moved"
    settled = offs[8:]
    assert max(abs(x) for x in settled) <= 2.0 * p.concede_ms, \
        f"never settled: {[round(x, 1) for x in offs]}"


def test_a_partner_who_cannot_bend_is_the_case_that_hunts():
    """The honest counter-case, and the non-vacuity of the test above.

    A fixed recording is not a player: it cannot meet the machine halfway, so
    the phase->period law — an integrator — has nothing to lock onto and walks
    away instead.  What must still hold is that the WALKING is bounded: inside
    the cap, inside the per-bar ramp.  The machine bends; it never lurches, and
    it never runs the tempo away.
    """
    p = RoleParams()
    tape, sts = _entrain(48, lead_ms=30.0, human_rate=0.0)
    mutual, _ = _entrain(48, lead_ms=30.0, human_rate=0.25)
    assert max(abs(x) for x in tape[8:]) > max(abs(x) for x in mutual[8:]), \
        "a partner who bends made no difference — the test above proves nothing"
    assert all(abs(s.tempo_offset_pct) <= p.concede_max_pct + 1e-9 for s in sts)
    steps = [abs(b.tempo_offset_pct - a.tempo_offset_pct)
             for a, b in zip(sts, sts[1:])]
    assert max(steps) <= MAX_STEP_PCT_PER_BAR + 1e-9


def test_an_untrusted_beat_never_moves_the_tempo():
    """Low beat confidence is never evidence, whoever measured the offset."""
    sts = drive(RoleNegotiator(LeadMeet()), 40, offset_ms=45.0,
                conf=0.2)
    assert not events_of(sts, "concede")


def test_an_unlocked_tracker_never_moves_the_tempo_on_its_OWN_estimate():
    """The lock gate, applied to the estimator that needs it (v2.1).

    ``grid_offset_ms`` reads ``BeatState.next_beat_t_s`` and
    ``beat_interval_s``, which mean nothing before the tracker locks, so the
    DERIVED path must never vote while unlocked — that is the rule, and it is
    unchanged.  Driven with no ``human_offset_ms`` at all, so the derived path
    is the only one live.
    """
    neg = RoleNegotiator(LeadMeet(), composed_bpm=120.0)
    bar_s = 2.0
    sts = []
    for i in range(40):
        sts.append(neg.update(
            bar=i, beat=mk_beat(bpm=120.0, locked=False, t=i * bar_s, bar=i,
                                next_beat=i * bar_s + 0.545),   # 45 ms out
            harmony=mk_harm(), human=mk_bar(i - 1),
            machine_grid_t_s=i * bar_s))
    assert not events_of(sts, "concede")


def test_a_MEASURED_offset_does_not_wait_for_the_tracker_to_lock():
    """v2.1 AMENDMENT, and it is load-bearing rather than a convenience.

    ``onset_phase_offset_ms`` measures the human's own attack times against the
    MACHINE's grid.  The machine's grid is GENERATED, not detected, so it is
    exactly known from t=0 and nothing in that measurement depends on
    ``BeatState``.  Gating it on a different estimator's lock made the response
    an artefact of tracker latency: MEASURED on the section-6 joiners the
    tracker locks a bar later on ``dragging`` than on ``rushing`` (the same
    take, 35 ms later), so behavior 3 fired a bar after behavior 2 for a reason
    with nothing to do with conceding, and SPEC-V2 section 7 specifies the two
    as symmetric.

    **And on real material the gate did not delay the rule, it DELETED it.**
    Re-running the whole real-material half of this file with the lock applied
    to the measured path as well: all four shifted takes end at exactly
    +0.00%, zero concessions, zero separation between a human pushed 30 ms
    early and one dragged 35 ms late — the tracker does not hold a lock
    steadily enough on percussive v0 renders for a two-bar window ever to
    survive.  ``concede_min_conf`` still applies on both paths (the test above
    this one), so an untrusted beat is still untrusted; what changed is only
    that a measurement which never consulted the beat is no longer gated on it.
    """
    sts = drive(RoleNegotiator(LeadMeet()), 8, offset_ms=45.0,
                locked=False)
    assert events_of(sts, "concede"), \
        "a measured offset was thrown away because a different estimator " \
        "had not locked"


def test_a_silent_bar_clears_the_evidence_window():
    """A mean stitched across a gap is a mean of two performances."""
    sts = drive(RoleNegotiator(LeadMeet()), 40, offset_ms=45.0,
                silent=lambda i: i % 2 == 1)
    assert not events_of(sts, "concede")


def test_a_follower_leaves_the_tempo_alone():
    """In follow mode v1's phase correction already pulls us onto the human's
    grid; a concession on top would be a second, slower copy of it."""
    sts = drive(RoleNegotiator(RoleParams(role="follow")), 20, offset_ms=45.0)
    assert not events_of(sts, "concede")
    assert all(s.tempo_offset_pct == 0.0 for s in sts)


def test_concede_rate_and_ms_are_actually_wired_to_behavior():
    """Non-vacuity for the two dials the spec calls the mechanism."""
    fast = drive(RoleNegotiator(LeadMeet(concede_rate=0.2)), 3,
                 offset_ms=31.0)[1]
    slow = drive(RoleNegotiator(LeadMeet(concede_rate=0.1)), 3,
                 offset_ms=31.0)[1]
    assert fast.tempo_target_pct == pytest.approx(2.0 * slow.tempo_target_pct)
    deaf = drive(RoleNegotiator(LeadMeet(concede_ms=60.0)), 20,
                 offset_ms=31.0)
    assert not events_of(deaf, "concede")


# ============================================ behaviors 2 & 3: the handoff ===


def test_a_human_who_takes_over_makes_the_machine_yield():
    p = RoleParams(role="lead")
    sts = drive(RoleNegotiator(p), 16,
                dens=lambda i: 1.0 if i < 4 else 4.0)     # ramps to 4/beat
    y = events_of(sts, "yield")
    assert y, "the human took over and the machine kept leading"
    assert sts[-1].role == "follow"
    assert sts[-1].clock_authority == pytest.approx(ROLE_DEFAULTS["follow"][0])
    assert sts[-1].form_authority == pytest.approx(ROLE_DEFAULTS["follow"][1])
    assert y[0].clock_authority < ROLE_DEFAULTS["lead"][0]
    assert "onsets/beat" in y[0].reason


def test_steady_busy_material_is_not_a_takeover():
    """MEASURED: plume's own trap render is a steady 5.0 onsets/beat.  Under a
    bare absolute threshold the machine yields the lead to a hi-hat."""
    sts = drive(RoleNegotiator(RoleParams(role="lead")), 40, dens=5.0)
    assert not events_of(sts, "yield")
    assert all(s.role == "lead" for s in sts)


def test_the_same_density_yields_when_it_is_actually_NEW():
    """Non-vacuity, and the semantics in one pair: 5.0 onsets/beat is a
    takeover when the human was playing 1.0, and is just the drum pattern when
    they have been playing 5.0 all along.  Identical bars, opposite answers —
    which is exactly what an absolute threshold cannot express."""
    steady = drive(RoleNegotiator(RoleParams(role="lead")), 40, dens=5.0)
    arrived = drive(RoleNegotiator(RoleParams(role="lead")), 40,
                    dens=lambda i: 1.0 if i < 6 else 5.0)
    assert not events_of(steady, "yield")
    assert events_of(arrived, "yield")


def test_a_human_who_drops_out_makes_the_machine_take_the_lead():
    p = RoleParams(role="follow")
    sts = drive(RoleNegotiator(p), 24, silent=lambda i: i >= 3)
    t = events_of(sts, "take_lead")
    assert t, "the human laid out and nobody held the floor"
    assert t[0].bar == 3 + p.handoff_silence_bars - 1
    assert sts[-1].role == "lead"
    assert sts[-1].clock_authority == pytest.approx(ROLE_DEFAULTS["lead"][0])
    assert sts[-1].form_authority == pytest.approx(ROLE_DEFAULTS["lead"][1])


def test_the_machine_does_not_take_a_lead_it_never_had_a_grid_for():
    """"...and the machine still has a groove"."""
    sts = drive(RoleNegotiator(RoleParams(role="follow")), 24, silent=True,
                locked=False)
    assert not events_of(sts, "take_lead")


def test_hysteresis_prevents_role_flapping():
    """Alternating stimulus at the hysteresis period must not produce a role
    change per window."""
    p = RoleParams(role="lead", handoff_hysteresis_bars=4)
    sts = drive(RoleNegotiator(p), 64,
                dens=lambda i: 5.0 if (i // 4) % 2 else 0.5,
                silent=lambda i: (i // 4) % 2 == 0 and i > 20)
    changes = sum(1 for a, b in zip(sts, sts[1:]) if a.role != b.role)
    assert changes * p.handoff_hysteresis_bars <= len(sts), \
        f"{changes} role changes in {len(sts)} bars at hysteresis " \
        f"{p.handoff_hysteresis_bars}"
    for a, b in zip(sts, sts[1:]):
        assert abs(b.clock_authority - a.clock_authority) <= AUTHORITY_SLEW + 1e-9
        assert abs(b.form_authority - a.form_authority) <= AUTHORITY_SLEW + 1e-9


def test_authority_slides_rather_than_jumping():
    sts = drive(RoleNegotiator(RoleParams(role="follow")), 20, silent=True,
                locked=True, conf=0.9)
    # (the lock is what arms take-lead; see the guard test above)
    seen = [s.clock_authority for s in sts]
    assert max(abs(b - a) for a, b in zip(seen, seen[1:])) <= AUTHORITY_SLEW + 1e-9
    assert seen[-1] == pytest.approx(ROLE_DEFAULTS["lead"][0])
    assert len({round(x, 4) for x in seen}) > 2, "it jumped"


# ==================================================== behavior 4: trading ===


@pytest.mark.parametrize("trade_bars", [2, 4, 8])
def test_trade_turn_is_a_pure_function_of_the_bar(trade_bars):
    for bar in range(0, 5 * trade_bars):
        want = "machine" if (bar // trade_bars) % 2 == 0 else "human"
        assert trade_turn(bar, trade_bars) == want
    with pytest.raises(ValueError):
        trade_turn(0, 0)


def test_trading_alternates_authority_on_the_boundaries():
    p = RoleParams(role="trade", trade_bars=4)
    sts = drive(RoleNegotiator(p), 16, dens=0.5)
    assert all(s.role == "trade" for s in sts)
    for s in sts:
        want = ROLE_DEFAULTS["lead" if trade_turn(s.bar, 4) == "machine"
                             else "follow"]
        assert (s.clock_authority, s.form_authority) == want, s.bar
    # the handover is announced exactly on the boundary, once per unit
    tr = events_of(sts, "trade")
    assert [s.bar for s in tr] == [0, 4, 8, 12]
    assert "bars 4-7 are yours" in tr[1].reason


def test_the_human_wins_a_collision_and_the_machine_resumes_after():
    p = RoleParams(role="trade", trade_bars=4)
    sts = drive(RoleNegotiator(p), 8, dens=lambda i: 5.0 if i == 2 else 0.5)
    hit = events_of(sts, "trade_collision")
    assert len(hit) == 1 and hit[0].bar == 2
    assert hit[0].clock_authority == pytest.approx(ROLE_DEFAULTS["follow"][0])
    # ...a one-bar courtesy, not a turn change: bar 3 is still the machine's
    assert sts[3].clock_authority == pytest.approx(ROLE_DEFAULTS["lead"][0])
    assert "trade" not in sts[3].events, "the trade was re-announced mid-unit"


def test_a_collision_on_the_humans_own_bars_is_not_a_collision():
    sts = drive(RoleNegotiator(RoleParams(role="trade", trade_bars=4)), 8,
                dens=lambda i: 5.0 if i >= 4 else 0.5)
    assert not events_of(sts, "trade_collision")


# ================================================== behavior 5: conforming ===


def test_collection_root_is_stable_under_the_relative_ambiguity():
    """A key and its relative share a pitch collection, and a chroma histogram
    cannot tell them apart — so the conform distance must not depend on which
    one the tracker named."""
    for a, b in [("C major", "A minor"), ("F minor", "Ab major"),
                 ("D minor", "F major"), ("D dorian", "C major"),
                 ("E phrygian", "C major")]:
        assert collection_root(parse_key(a)) == collection_root(parse_key(b)), \
            f"{a} vs {b}"
    assert collection_root(parse_key("F# minor")) != \
        collection_root(parse_key("F minor"))


@pytest.mark.parametrize("mine,theirs,want", [
    ("F minor", "F# minor", 1), ("F minor", "E minor", -1),
    ("F minor", "G minor", 2), ("F minor", "Eb minor", -2),
    ("C major", "D major", 2), ("C major", "Bb major", -2),
])
def test_a_capod_human_is_met_within_conform_bars(mine, theirs, want):
    p = RoleParams(role="lead")
    sts = drive(RoleNegotiator(p, composed_key=mine), 12, key=theirs)
    ev = events_of(sts, "conform")
    assert len(ev) == 1, f"{mine} vs {theirs}: {[s.events for s in sts]}"
    assert ev[0].bar <= p.conform_bars + 1
    assert ev[0].transpose_semitones == want
    assert sts[-1].transpose_semitones == want
    assert f"transposing {want:+d}" in ev[0].reason


@pytest.mark.parametrize("theirs,st", [
    ("B minor", 6), ("D major", 6),      # a tritone of collection away
    ("B major", 3), ("F major", 3),      # three, which is still "something else"
])
def test_a_far_key_is_not_conformed_to_and_the_log_says_why(theirs, st):
    sts = drive(RoleNegotiator(RoleParams(role="lead"), composed_key="F minor"),
                20, key=theirs)
    assert not events_of(sts, "conform")
    assert all(s.transpose_semitones == 0 for s in sts)
    hold = events_of(sts, "conform_hold")
    assert len(hold) == 1, "held silently, or complained every bar"
    assert "conforming would be guessing" in hold[0].reason
    assert f"{st} st" in hold[0].reason, hold[0].reason


def test_the_relative_key_is_not_a_reason_to_transpose():
    """MEASURED: plume's own C-major lo-fi render is reported as A minor on
    every frame at both gains.  On tonics that is 3 st away and would log a
    spurious hold; on collections it is 0 st and is simply our own key."""
    sts = drive(RoleNegotiator(RoleParams(role="lead"), composed_key="C major"),
                20, key="A minor")
    assert not events_of(sts, "conform")
    assert not events_of(sts, "conform_hold")
    assert all(s.transpose_semitones == 0 for s in sts)


def test_conforming_happens_once_and_then_we_are_in_their_key():
    sts = drive(RoleNegotiator(RoleParams(role="lead"), composed_key="F minor"),
                40, key="F# minor")
    assert len(events_of(sts, "conform")) == 1
    assert not events_of(sts, "conform_hold")
    assert sts[-1].transpose_semitones == 1


def test_a_flickering_key_belief_never_conforms():
    sts = drive(RoleNegotiator(RoleParams(role="lead"), composed_key="F minor"),
                40, key=lambda i: "F# minor" if i % 2 else "E minor")
    assert not events_of(sts, "conform")


@pytest.mark.parametrize("kw,harm", [
    ({"conform_enabled": False}, {"key": "F# minor"}),
    ({}, {"key": "F# minor", "key_conf": 0.1}),
    ({}, {"key": None}),
])
def test_conforming_needs_permission_and_evidence(kw, harm):
    sts = drive(RoleNegotiator(RoleParams(role="lead", **kw),
                               composed_key="F minor"), 20, **harm)
    assert not events_of(sts, "conform")
    assert not events_of(sts, "conform_hold")


def test_without_a_composed_key_there_is_nothing_to_conform_to():
    sts = drive(RoleNegotiator(RoleParams(role="lead")), 20, key="F# minor")
    assert not events_of(sts, "conform") and not events_of(sts, "conform_hold")


# ============================================================== contract ===


def test_reason_strings_are_clean_and_self_describing():
    """The decisions log is the user-facing forensics surface."""
    scripts = [
        dict(offset_ms=45.0, dens=lambda i: 1.0 if i < 8 else 5.0,
             key="F# minor"),
        dict(offset_ms=-45.0, silent=lambda i: 10 <= i < 22, key="B minor"),
        dict(offset_ms=lambda i: (5.0, -7.0)[i % 2], dens=0.5, key="A minor"),
    ]
    for role in ROLES:
        for kw in scripts:
            for st in drive(RoleNegotiator(RoleParams(role=role),
                                           composed_key="F minor"), 40, **kw):
                r = st.reason
                assert r and "?" not in r, r
                assert "None" not in r, r
                assert "-0.0" not in r, r
                assert not r.startswith(" ") and not r.endswith(" "), repr(r)
                assert st.role in ROLES
                assert 0.0 <= st.clock_authority <= 1.0
                assert 0.0 <= st.form_authority <= 1.0
                assert st.bpm > 0.0


def test_every_event_puts_its_own_clause_in_the_reason():
    """Run under BOTH tempo policies, because they emit different events and
    an event with no clause is an event a reader of the log cannot see."""
    tag_word = {"concede": "conceding", "concede_capped": "tempo capped",
                "tempo_match": "matching tempo", "tempo_hold": "holding: human",
                "yield": "yielding", "take_lead": "taking the lead",
                "trade": "trading", "trade_collision": "trading",
                "conform": "conforming", "conform_hold": "holding key"}
    seen = set()
    for params in (LeadMeet(), RoleParams(role="lead")):
        sts = drive(RoleNegotiator(params, composed_key="F minor"),
                    60, offset_ms=45.0, dens=lambda i: 1.0 if i < 8 else 5.0,
                    key="F# minor")
        for st in sts:
            for tag in st.events:
                assert tag_word[tag] in st.reason, (tag, st.reason)
                seen.add(tag)
            if not st.events:
                assert st.reason.startswith(f"{st.role}: clock ")
    assert {"concede", "tempo_hold", "yield", "conform"} <= seen, seen


def test_bpm_and_tempo_offset_never_disagree():
    sts = drive(RoleNegotiator(LeadMeet(), composed_bpm=138.0), 40,
                offset_ms=45.0)
    for s in sts:
        assert s.bpm == pytest.approx(effective_bpm(138.0, s.tempo_offset_pct))


def test_determinism():
    def run():
        return [dataclasses.astuple(s) for s in drive(
            RoleNegotiator(RoleParams(role="lead"), composed_bpm=131.0,
                           composed_key="F minor"), 80,
            offset_ms=lambda i: 40.0 if i % 7 else -12.0,
            dens=lambda i: 0.5 + (i % 11) * 0.5,
            silent=lambda i: i % 13 == 0,
            key=lambda i: ("F# minor", "F minor", "B minor")[i % 3])]
    assert run() == run()


# ======================================================= real material ====


def _render(path: Path, style: str, bpm: float, key: str, *,
            bars: tuple[int, int] = (0, 32), seed: int = 7,
            gain_db: float = 0.0) -> np.ndarray:
    """One of plume's own v0 renders, as mono float32 at ``gain_db``."""
    import soundfile as sf
    session = compose(PromptSpec(styles=[style], bpm=bpm, key=key, movements=1,
                                 mood=[], name=path.stem), seed=seed)
    render(session, out_path=str(path), movement=0, bars=bars)
    x, _ = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if gain_db:
        x = x * (10.0 ** (gain_db / 20.0))
    return x.astype(np.float32)


def _shift(x: np.ndarray, ms: float) -> np.ndarray:
    n = int(round(abs(ms) / 1000.0 * SR))
    if ms > 0:                                  # human late (laying back)
        return np.concatenate([np.zeros(n, dtype=np.float32), x])
    if ms < 0:                                  # human early (rushing)
        return x[n:].copy()
    return x


class Take:
    """A real render, listened to once with the real listening stack.

    v2.1: the analysis chain hears the take with its FADER TAKEN OUT
    (:func:`plume.sim.lead_analysis_gain`), exactly as ``simulate_lead`` and
    live lead mode do.  ``FeatureExtractor``'s onset picker is an absolute
    threshold, so without this every "same performance, two gains" pair in
    this file is really two different analyses of two different signals, and
    the gain axis measures the detector rather than the negotiator.  With it,
    the -20 dB takes below produce BIT-IDENTICAL negotiations to the 0 dB
    ones, which is the property the whole gain axis exists to assert.
    ``self.x`` stays the take as rendered, so ``_shift`` still shifts audio.
    """

    def __init__(self, name, x, bpm, key, *, swing: float = 0.5):
        self.name, self.x, self.bpm, self.key = name, x, bpm, key
        self.swing = float(swing)
        self.dur_s = x.size / SR
        ana = (np.asarray(x, dtype=np.float64)
               * lead_analysis_gain(x)).astype(np.float32)
        fe, bt, ht = FeatureExtractor(sr=SR), BeatTracker(), HarmonyTracker()
        self.t = []          # frame end times
        self.beat = []       # BeatState AT that frame
        self.harm = []
        self.rms = []
        self.onset = []
        for i in range(0, ana.size, 4096):
            frames = fe.process(ana[i:i + 4096])
            if not frames:
                continue
            bs, hs = bt.process(frames), ht.process(frames)
            for f in frames:
                self.t.append(f.t_s)
                self.beat.append(bs)
                self.harm.append(hs)
                self.rms.append(f.rms_db)
                self.onset.append(bool(f.onset))
        self.t = np.asarray(self.t)
        self.onset_t = self.t[np.asarray(self.onset)]

    def densities(self, *, composed_bpm: float) -> list[float]:
        """Per-bar human onset density, on the machine's grid."""
        out, t, prev, bar = [], 0.0, 0.0, 0
        while t < self.dur_s and bar < 400:
            i = min(int(np.searchsorted(self.t, t)), self.t.size - 1)
            lo = int(np.searchsorted(self.t, prev))
            if bar > 0 and i > lo:
                out.append(sum(self.onset[lo:i]) / 4.0)
            prev, bar = t, bar + 1
            t += 4.0 * 60.0 / composed_bpm
        return out

    def closed_loop(self, params: RoleParams, *, composed_bpm: float,
                    composed_key: str | None = None,
                    measured_offset: bool = False) -> list[RoleState]:
        """Drive the negotiator with the machine's grid advancing at the tempo
        the negotiator itself just chose.  This is the loop ``simulate_lead``
        closes, and it is where a concede rule can run away."""
        neg = RoleNegotiator(params, composed_bpm=composed_bpm,
                             composed_key=composed_key)
        t, prev, bar, out = 0.0, 0.0, 0, []
        while t < self.dur_s and bar < 400:
            i = min(int(np.searchsorted(self.t, t)), self.t.size - 1)
            lo = int(np.searchsorted(self.t, prev))
            human = None
            if bar > 0 and i > lo:
                ons = sum(self.onset[lo:i])
                human = BarFeatures(bar=bar - 1,
                                    rms_db=float(np.mean(self.rms[lo:i])),
                                    onset_density=ons / 4.0, silent=(ons == 0),
                                    register_low_midi=None,
                                    register_high_midi=None)
            off = unit = None
            if measured_offset and human is not None:
                beat_s = 60.0 / effective_bpm(
                    composed_bpm, out[-1].tempo_offset_pct if out else 0.0)
                unit = beat_s / 4.0
                # v2.1: measured against the grid the machine is PLAYING (its
                # own swing), and the subdivision is handed to the negotiator
                # so its plausibility bound can be stated in the instrument's
                # own units instead of the beat's.
                off = onset_phase_offset_ms(self.onset_t, prev, t, unit,
                                            swing=self.swing)
            st = neg.update(bar=bar, beat=self.beat[i], harmony=self.harm[i],
                            human=human, machine_grid_t_s=t,
                            human_offset_ms=off, human_offset_unit_s=unit)
            out.append(st)
            prev = t
            t += 4.0 * 60.0 / effective_bpm(composed_bpm, st.tempo_target_pct)
            bar += 1
        return out


#: The machine's own swing per style, read from the style pack the machine
#: plays.  A swung machine measuring a swung human against a STRAIGHT grid
#: reads the groove they share as a lay-back (see onset_phase_offset_ms).
SWING = {"trap": 0.5, "lofi": 0.57, "synthwave": 0.5}


@pytest.fixture(scope="session")
def takes(tmp_path_factory):
    d = tmp_path_factory.mktemp("roles_real")
    out = {}
    for style, bpm, key in (("trap", 140.0, "D minor"),
                            ("lofi", 80.0, "C major")):
        for gain in (0.0, -20.0):
            n = f"{style}{bpm:.0f}{'' if gain == 0 else '_quiet'}"
            out[n] = Take(n, _render(d / f"{n}.wav", style, bpm, key,
                                     gain_db=gain), bpm, key,
                          swing=SWING[style])
    # The same renders shifted by a known amount, at both gains: displacing a
    # real recording is the only ground truth about timing it can be made to
    # carry, and the gain axis is the one the v1 fixtures had none of.
    for stem, bpm, style in (("trap140", 140.0, "trap"),
                             ("lofi80", 80.0, "lofi")):
        for gsfx, g in (("", 1.0), ("_quiet", 0.1)):
            base = out[stem].x if gsfx == "" else out[stem + "_quiet"].x
            for label, ms in (("rush", -30.0), ("drag", 35.0)):
                n = f"{stem}{gsfx}_{label}"
                out[n] = Take(n, _shift(base, ms), bpm, None,
                              swing=SWING[style])
    # Real renders a semitone up and a tritone away from a composed F minor.
    out["fsharp"] = Take("fsharp", _render(d / "fs.wav", "synthwave", 100.0,
                                           "F# minor", bars=(0, 24)),
                         100.0, "F# minor", swing=SWING["synthwave"])
    out["tritone"] = Take("tritone", _render(d / "tri.wav", "synthwave", 100.0,
                                             "B minor", bars=(0, 24)),
                          100.0, "B minor", swing=SWING["synthwave"])
    return out


REAL = ("trap140", "trap140_quiet", "lofi80", "lofi80_quiet")


@pytest.mark.parametrize("name", REAL)
def test_the_listening_stack_really_is_working_on_this_material(takes, name):
    """Non-vacuity for everything below: an unlocked tracker would make every
    negotiation test pass by doing nothing.

    Sampled three quarters of the way in rather than at the last frame —
    plume's arcs end on an outro that genuinely thins out, and the tracker
    letting go of a groove that stopped is the feature, not the bug.
    """
    tk = takes[name]
    j = int(0.75 * len(tk.beat))
    assert tk.beat[j].locked, f"{name}: not locked 3/4 of the way in"
    assert tk.beat[j].bpm == pytest.approx(tk.bpm, rel=0.05)
    assert tk.beat[j].confidence > RoleParams().concede_min_conf
    assert tk.harm[j].key is not None
    assert tk.harm[j].key_conf > RoleParams().conform_key_conf


@pytest.mark.parametrize("name", REAL)
def test_the_machine_does_not_run_its_tempo_away_on_real_material(takes, name):
    """THE HEADLINE GUARD.  Machine and human at the same tempo: an unshakeable
    clock must stay approximately unshakeable while a real, swung, percussive
    human plays over it.

    FAILED BEFORE the plausibility/spread gates: the ``BeatState``-derived
    offset reached -214ms (half a beat) on this exact material and produced a
    single -8% concession — the cap, in one step, in the wrong direction.
    """
    tk = takes[name]
    sts = tk.closed_loop(LeadMeet(), composed_bpm=tk.bpm)
    assert len(sts) > 12, f"{name}: only {len(sts)} bars — not a test"
    worst = max(abs(s.tempo_offset_pct) for s in sts)
    assert worst < 4.5, (
        f"{name}: tempo wandered to {worst:.2f}% of {tk.bpm} bpm on a human "
        f"playing at that tempo: "
        f"{[round(s.tempo_offset_pct, 2) for s in sts]}")
    assert not events_of(sts, "concede_capped"), f"{name}: saturated the cap"
    steps = [abs(b.tempo_offset_pct - a.tempo_offset_pct)
             for a, b in zip(sts, sts[1:])]
    assert max(steps, default=0.0) <= MAX_STEP_PCT_PER_BAR + 1e-9, name


def test_the_offset_gates_are_load_bearing_on_real_material(takes):
    """Non-vacuity for the guard above: turn the protections off and the SAME
    takes run four times as far.  If this ever stops failing, the guard proves
    nothing and the material needs replacing, not the guard relaxing.

    Stated across the corpus rather than per take, because it is a property of
    the *material*: a take whose beat estimate happens to be well-behaved (or
    that unlocks and stops voting) cannot exercise a gate against bad
    estimates, and pretending otherwise is how a vacuous test gets written.

    v2.1: the phase-return term is now one of the protections, so it is one of
    the things turned off here.  MEASURED on trap140 (identical at both gains,
    because the analysis chain is fader-invariant): 1.52% with everything on,
    2.83% with the plausibility and spread gates opened, 2.00% with only the
    phase-return term removed, **6.00% with both** — which is also why the
    ungated rule no longer reaches the 8% cap it used to. Two independent
    mechanisms now bound the runaway instead of one, and the assertion below
    says that in the units the code actually produces rather than in a cap
    number that has stopped being reachable.
    """
    def worst(name, **kw):
        tk = takes[name]
        return max(abs(s.tempo_offset_pct)
                   for s in tk.closed_loop(LeadMeet(**kw),
                                           composed_bpm=tk.bpm))

    OFF = dict(concede_plausible_frac=0.5, concede_spread_ms=1e6,
               concede_drift_ms=1e6)
    rows = {}
    for name in REAL:
        g = worst(name)
        u = worst(name, **OFF)
        rows[name] = (round(g, 2), round(u, 2))
        assert u >= g - 1e-9, f"{name}: guarding made it WORSE ({g:.2f} vs {u:.2f})"
    blew_up = [n for n, (g, u) in rows.items() if u >= 4.5 and u >= 3.0 * g]
    assert blew_up, (
        "no take ran away with every guard off — the guards prove nothing on "
        f"this corpus and the material needs replacing: {rows}")
    assert all(rows[n][0] < 4.5 for n in blew_up), rows


SHIFTED = ("trap140", "trap140_quiet", "lofi80", "lofi80_quiet")


def _final_pct(takes, stem: str, suffix: str, **kw) -> float:
    tk = takes[stem + suffix] if suffix else takes[stem]
    sts = tk.closed_loop(LeadMeet(**kw),
                         composed_bpm=takes[stem].bpm, measured_offset=True)
    return sts[-1].tempo_offset_pct


def test_pushing_a_real_human_earlier_makes_the_machine_faster(takes):
    """THE MECHANISM, end to end on real audio: same render, same machine, the
    human moved 30ms earlier versus 35ms later, and the machine ends FASTER in
    the first case than the second.  Both gains, both styles.

    Stated as a comparison rather than as an absolute sign because a groove has
    a standing offset of its own — MEASURED, the swung lo-fi render sits 20ms
    behind the grid before anybody shifts anything, and a swung player genuinely
    IS laying back.  Asking the machine to call that "on the beat" would be
    asking it to un-hear the music.  What the rule owes us is the DIFFERENCE.
    """
    rows = {}
    for stem in SHIFTED:
        rush = _final_pct(takes, stem, "_rush")
        drag = _final_pct(takes, stem, "_drag")
        rows[stem] = (round(rush, 2), round(drag, 2))
        assert rush >= drag - 1e-9, (
            f"{stem}: the human pushed EARLY and the machine ended slower "
            f"({rush:+.2f}% vs {drag:+.2f}% for the late take)")
    real = [s for s, (a, b) in rows.items() if a - b > 1.0]
    assert len(real) >= 3, (
        f"only {len(real)} of {len(rows)} takes separated at all — the rule is "
        f"inert on this corpus and the test proves nothing: {rows}")


def test_the_sign_convention_holds_end_to_end_on_real_audio(takes):
    """A rushing human really does drive the tempo UP and a dragging one DOWN,
    absolutely and not just relatively, on the take where the estimator's
    per-bar noise (IQR ~14ms) is small next to the shift."""
    assert _final_pct(takes, "trap140", "_rush") > 0.0
    assert _final_pct(takes, "trap140", "_drag") < 0.0
    assert _final_pct(takes, "trap140", "") == pytest.approx(0.0, abs=1.0)


@pytest.mark.parametrize("stem", ("trap140", "lofi80"))
@pytest.mark.parametrize("suffix", ("", "_rush", "_drag"))
def test_the_negotiation_is_BIT_IDENTICAL_at_two_gains(takes, stem, suffix):
    """THE FADER TRAP (v2.1), as a regression test.

    Everything the negotiator knows about the human comes out of
    ``FeatureExtractor``, whose onset picker is an ABSOLUTE threshold
    (``median(flux) + ONSET_DELTA``, and ``ONSET_DELTA`` is 4.0 flux units, not
    a ratio).  Flux scales with the signal, so turning a real human down
    deletes their onsets — v1's exact bug in different clothes.  MEASURED on
    ONE v0 lo-fi render at five gains with NOTHING about the playing changed:
    density 5.08 -> 4.21 -> 3.04 -> 1.94 -> 0.72 onsets/beat from 0 to -30 dB,
    four concessions at 0 dB and none below -20 dB, and the machine YIELDING
    THE LEAD at bar 12 at -6/-12/-20 dB.  Turning a player down handed them
    the band.

    The fix is ``sim.lead_analysis_gain``: the analysis chain hears the take
    with its fader taken out (one constant over the whole take, so the fader
    goes and the dynamics stay).  The property that buys is EQUALITY, not
    similarity, so that is what is asserted.

    ONE FIELD IS EXEMPT, and the exemption is measured rather than assumed.
    ``RoleState.human_offset_ms`` (v2.2) publishes the raw measurement, and on
    ``lofi80_rush`` bar 25 the two gains disagree by **3e-6 ms** — three
    picoseconds of musical time.  The cause is not the policy and not the
    concede rules: that bar's predecessor contains a single onset, so
    ``onset_phase_offset_ms`` declines (``min_onsets=3``) and the negotiator
    falls back to the DERIVED estimator, which reads ``BeatState``'s own
    ``next_beat_t_s``/``beat_interval_s`` — and ``BeatTracker``'s bpm is
    fader-invariant only to about 1e-10 relative, not exactly (MEASURED:
    80.01214816792721 vs 80.01214816819196 at the same frame).  That has
    always been true; before v2.2 nothing published a number fine-grained
    enough to see it, because every DECISION is many orders of magnitude away
    from the difference.  So the decisions are still asserted EQUAL, and the
    raw measurement is asserted equal to a tolerance five orders of magnitude
    under the estimator's own +/-10-30 ms noise — rather than rounding the
    field until the strict test passes, which would hide a real property of
    the tracker behind a display choice.
    """
    loud = takes[stem + suffix]
    quiet = takes[f"{stem}_quiet{suffix}"]
    a = loud.closed_loop(LeadMeet(), composed_bpm=loud.bpm,
                         measured_offset=True)
    b = quiet.closed_loop(LeadMeet(), composed_bpm=quiet.bpm,
                          measured_offset=True)
    assert [dataclasses.replace(s, human_offset_ms=None) for s in a] == \
           [dataclasses.replace(s, human_offset_ms=None) for s in b], (
        f"{stem}{suffix}: 20 dB of fader changed the negotiation")
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x.human_offset_ms is None) == (y.human_offset_ms is None), \
            f"{stem}{suffix} bar {x.bar}: the fader decided whether we measured"
        if x.human_offset_ms is not None:
            assert x.human_offset_ms == pytest.approx(y.human_offset_ms,
                                                      abs=1e-3), \
                f"{stem}{suffix} bar {x.bar}: the fader moved the measurement"
    assert loud.densities(composed_bpm=loud.bpm) == \
        quiet.densities(composed_bpm=quiet.bpm)


def test_the_gain_normaliser_leaves_silence_alone():
    """A silent (or near-silent) take must not be amplified into a performance,
    and a zero-length one must not divide by anything."""
    assert lead_analysis_gain(np.zeros(0, dtype=np.float32)) == 1.0
    assert lead_analysis_gain(np.zeros(48000, dtype=np.float32)) == 1.0
    assert lead_analysis_gain(np.full(48000, 1e-5, dtype=np.float32)) == 1.0
    # ...and it is exactly gain-inverting on real signal.
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(48000) * 0.2).astype(np.float32)
    g0 = lead_analysis_gain(x)
    for g in (0.1, 0.5, 2.0):
        assert lead_analysis_gain((x * g).astype(np.float32)) == \
            pytest.approx(g0 / g, rel=1e-6)


def test_a_wrapped_offset_is_refused_when_the_subdivision_is_known():
    """THE ALIASING DEFECT (v2.1).

    ``onset_phase_offset_ms`` is a circular mean over ONE SUBDIVISION, so its
    unambiguous range is +/- half a sixteenth = +/-62.5 ms at 120 bpm — while
    ``concede_plausible_frac`` is a fifth of a BEAT, +/-100 ms.  Every wrapped
    sample walked straight through the gate that exists to catch exactly this.
    MEASURED at 120 bpm: a human 70 ms ahead reads -55 ms, 90 ms ahead reads
    -35 ms, 150 ms ahead reads +25 ms — sign reversed, gate passed.

    Told the subdivision, the negotiator refuses the near-wrap band.  Without
    it, the same reading is admitted — which is the non-vacuity half.
    """
    unit_s = 60.0 / 120.0 / 4.0          # a sixteenth at 120 bpm = 125 ms
    p = LeadMeet()
    wrapped = -0.9 * (p.concede_plausible_unit_frac * unit_s * 1000.0) - 6.0

    def drive_unit(unit):
        neg = RoleNegotiator(p, composed_bpm=120.0)
        out = []
        for i in range(12):
            out.append(neg.update(
                bar=i, beat=mk_beat(bpm=120.0, t=i * 2.0, bar=i),
                harmony=mk_harm(), human=mk_bar(i - 1),
                machine_grid_t_s=i * 2.0, human_offset_ms=wrapped,
                human_offset_unit_s=unit))
        return out

    assert not events_of(drive_unit(unit_s), "concede"), (
        f"a {wrapped:+.0f}ms reading is past {p.concede_plausible_unit_frac} "
        f"of the sixteenth it was measured on — it is the next subdivision")
    assert events_of(drive_unit(None), "concede"), (
        "without the subdivision the same reading is admitted, so the bound "
        "above is what refused it and not something else")


def test_a_swung_human_over_a_swung_machine_is_not_a_lay_back():
    """A swung sixteenth is displaced 35 ms at 120 bpm — larger than
    ``concede_ms`` — so measuring a swung human against a STRAIGHT grid reads
    the groove the two players SHARE as the human laying back.  MEASURED on a
    v0 lo-fi render (swing 0.57) against a machine playing the same lo-fi style
    at the same tempo: a standing -30 ms that drove real concessions."""
    bar_s = 2.0
    unit = bar_s / 16.0

    def groove(swing, slots):
        """Onsets on ``slots`` of every beat, displaced by ``swing``."""
        n = np.array([b * 4 + s for b in range(4 * 8) for s in slots],
                     dtype=np.float64)
        return n * unit + np.where((n % 4 == 1) | (n % 4 == 3),
                                   (swing - 0.5) * 2.0 * unit, 0.0)

    def read(t, **kw):
        vals = [onset_phase_offset_ms(t, k * bar_s, (k + 1) * bar_s, unit, **kw)
                for k in range(8)]
        return float(np.mean([v for v in vals if v is not None]))

    # A shuffle at the spec's own far end (0.66 = exact triplet), with the
    # groove ON the swung subdivisions — which is what a shuffle IS.
    t = groove(0.66, (0, 1, 3))
    straight, swung = read(t), read(t, swing=0.66)
    assert abs(straight) > RoleParams().concede_ms, (
        f"the straight grid read this shuffle as only {straight:+.1f}ms out, "
        f"so it would not have conceded to it and nothing is proven")
    assert abs(swung) < 2.0, (
        f"a machine and a human sharing a groove read {swung:+.1f}ms apart")

    # ...and lo-fi's own 0.57, where the bias is smaller but still real: the
    # straight reading has to MOVE when the machine's swing is declared.
    t = groove(0.57, (0, 1, 2, 3))
    straight, swung = read(t), read(t, swing=0.57)
    assert abs(swung) < 1.0 < abs(straight - swung), (
        f"declaring the machine's swing changed nothing "
        f"({straight:+.1f} -> {swung:+.1f}ms)")


def test_swing_is_the_identity_at_one_half():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 2.0, 40))
    for k in range(2):
        a = onset_phase_offset_ms(t, k, k + 1.0, 0.125)
        b = onset_phase_offset_ms(t, k, k + 1.0, 0.125, swing=0.5)
        assert a == b


def test_the_phase_return_term_bounds_a_partner_who_cannot_bend():
    """THE HEADLINE v2.1 FIX: a phase->period law with no phase-return term
    diverges by construction, and it did — MEASURED in the rendered audio, the
    32-bar ``rushing`` joiner ended with the machine's downbeat 2.5 BEATS ahead
    of the human's, and ``dragging`` 3.7 beats behind.

    Here in the synthetic loop the same defect is a tape (``human_rate=0``):
    without the drift term the offset runs past **1.3 seconds** and is still
    growing at bar 48.  With it, the gap is bounded.  Non-vacuity is the
    comparison, so a rule that simply never conceded could not pass.
    """
    with_term, _ = _entrain(48, lead_ms=30.0, human_rate=0.0)
    without, _ = _entrain(48, lead_ms=30.0, human_rate=0.0,
                          concede_drift_ms=1e9)
    a = max(abs(x) for x in with_term[8:])
    b = max(abs(x) for x in without[8:])
    assert b > 800.0, (
        f"the un-corrected loop only reached {b:.0f}ms — it no longer diverges "
        f"on this stimulus and this test proves nothing")
    assert a < b / 3.0, (
        f"the phase-return term barely helped: {a:.0f}ms vs {b:.0f}ms")
    assert a < 250.0, f"still {a:.0f}ms out at the end of the take"


def test_what_v2_1_ADDED_on_the_live_path_is_still_free():
    """The headroom that makes live jam possible cannot silently regress.

    v1's beat tracker runs ~180x real time and that headroom is the reason a
    live session is possible at all, so everything added beside it has to stay
    negligible.  MEASURED on this machine: the de-swung phase estimator 8.2 us
    per bar (~244,000x real time; 5.3 us straight), and the fader-removal
    constant 1.1 ms for a 60-SECOND take, once, at ~52,000x.  The floors below
    are three orders of magnitude under those, so this catches an algorithmic
    regression rather than benchmarking a loaded laptop.
    """
    bar_s = 2.0
    ons = np.sort(np.random.default_rng(1).uniform(0.0, bar_s, 32))
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        onset_phase_offset_ms(ons, 0.0, bar_s, bar_s / 16.0, swing=0.57)
    per_bar = (time.perf_counter() - t0) / n
    est_x = bar_s / per_bar

    x = (np.random.default_rng(0).standard_normal(48000 * 10)
         * 0.2).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(5):
        lead_analysis_gain(x)
    gain_s = (time.perf_counter() - t0) / 5
    gain_x = 10.0 / gain_s

    print(f"\n  v2.1 additions: phase estimator {per_bar * 1e6:.1f} us/bar "
          f"-> {est_x:,.0f}x real time; analysis gain "
          f"{gain_s * 1000:.2f} ms/10s -> {gain_x:,.0f}x (once per take)")
    assert est_x > 20_000.0, f"the phase estimator fell to {est_x:,.0f}x"
    assert gain_x > 2_000.0, f"the analysis gain fell to {gain_x:,.0f}x"


def test_the_drift_term_never_fires_on_an_on_grid_human():
    """The other half: a rate law that fires on noise would make an
    unshakeable clock shakeable, which is the whole value of lead mode."""
    for stim in (0.0,
                 lambda i: (8.0, -7.0, 6.0, -9.0)[i % 4],
                 lambda i: 12.0,
                 lambda i: 40.0 if i % 2 else -40.0):
        sts = drive(RoleNegotiator(LeadMeet(), composed_bpm=120.0),
                    64, offset_ms=stim, )
        assert not events_of(sts, "tempo_match"), stim
        assert all(s.tempo_offset_pct == 0.0 for s in sts)


def _median_offset(tk) -> float:
    unit = 60.0 / tk.bpm / 4.0
    bar_s = 4 * 60.0 / tk.bpm
    vals = [onset_phase_offset_ms(tk.onset_t, k * bar_s, (k + 1) * bar_s, unit)
            for k in range(int(tk.dur_s / bar_s))]
    vals = [v for v in vals if v is not None]
    assert len(vals) > 10, f"{tk.name}: only {len(vals)} measurable bars"
    return float(np.median(vals))


@pytest.mark.parametrize("stem", SHIFTED)
@pytest.mark.parametrize("suffix,truth_ms", [("_rush", 30.0), ("_drag", -35.0)])
def test_the_onset_phase_estimator_tracks_a_known_shift(takes, stem, suffix,
                                                        truth_ms):
    """The measurement ``human_offset_ms`` wants, on real audio, at two gains
    and two grooves.

    Asserted as a DELTA from the unshifted take, because a groove has a real
    standing offset of its own — MEASURED, the swung lo-fi render sits ~20ms
    behind the grid — and a swung human genuinely IS laying back.  The
    estimator must recover the CHANGE without being asked to un-hear the swing.
    """
    got = _median_offset(takes[stem + suffix]) - _median_offset(takes[stem])
    assert got == pytest.approx(truth_ms, abs=15.0), (
        f"{stem}{suffix}: measured {got:+.1f}ms for a {truth_ms:+.1f}ms shift")


@pytest.mark.parametrize("name", [s + x for s in SHIFTED
                                  for x in ("_rush", "_drag")])
def test_the_measured_path_never_exceeds_the_cap_on_real_material(takes, name):
    tk = takes[name]
    sts = tk.closed_loop(LeadMeet(), composed_bpm=tk.bpm,
                         measured_offset=True)
    p = RoleParams()
    assert all(abs(s.tempo_offset_pct) <= p.concede_max_pct + 1e-9 for s in sts)
    steps = [abs(b.tempo_offset_pct - a.tempo_offset_pct)
             for a, b in zip(sts, sts[1:])]
    assert max(steps, default=0.0) <= MAX_STEP_PCT_PER_BAR + 1e-9


@pytest.mark.parametrize("name", ("trap140", "trap140_quiet"))
def test_real_dense_material_does_not_make_the_machine_yield_the_lead(takes,
                                                                      name):
    """MEASURED: this material runs a steady 4-8 onsets/beat, two to four times
    the spec's 2.0.  Hats are not a takeover, and a machine that hands the lead
    to a drum loop in bar 4 has no lead mode at all.

    The lo-fi renders are excluded, and the test below says why with numbers:
    their density does not sit flat, it RAMPS, and a ramp is what a takeover
    is.  (Before v2.1 only ``lofi80_quiet`` was excluded, on exactly that
    reasoning; now that the analysis chain is fader-invariant the loud and
    quiet takes are the same measurement, so the exclusion is the same too.)
    """
    tk = takes[name]
    p = RoleParams(role="lead")
    # First: the material must actually be able to trip the spec's bare
    # absolute rule, or this test guards nothing.
    dens = [h for h in tk.densities(composed_bpm=tk.bpm)]
    run = best = 0
    for x in dens:
        run = run + 1 if x > p.handoff_density else 0
        best = max(best, run)
    assert best >= p.handoff_hysteresis_bars, (
        f"{name}: densities {dens} never sustain > {p.handoff_density} for "
        f"{p.handoff_hysteresis_bars} bars, so the absolute rule would not "
        f"have fired either and this proves nothing")

    sts = tk.closed_loop(p, composed_bpm=tk.bpm)
    assert all(s.role == "lead" for s in sts), (
        f"{name}: handed the lead away to a drum loop at bar "
        f"{next(s.bar for s in sts if s.role != 'lead')} "
        f"(densities {dens})")
    assert not events_of(sts, "yield")


@pytest.mark.parametrize("name", ("lofi80", "lofi80_quiet"))
def test_the_lofi_render_really_does_ramp_and_yielding_there_is_correct(takes,
                                                                       name):
    """The exclusion above, made falsifiable rather than asserted.

    A takeover is the human getting BUSIER THAN THEY HAVE BEEN — that is the
    whole content of ``handoff_density_ratio``, and it is what separates a hat
    pattern from somebody taking the tune off you.  The trap render sits flat
    at 4-8 onsets/beat from bar 0 and never trips it; the lo-fi render's own
    arrangement steps up at its second section and MEASURED goes from ~1.5
    onsets/beat to ~4.3 — nearly a factor of three — which is the same shape as
    the ``takes_over`` fixture and a bigger jump than it.  So the machine
    yields, and that is the rule working, not failing.

    Both gains are asserted because after v2.1 they are the SAME measurement:
    the analysis chain takes the fader out, so a take 20 dB down produces the
    identical density sequence and the identical decision.  That equality is
    the point of the pair — it is what the old "only the quiet one ramps"
    exclusion was really reporting.
    """
    tk = takes[name]
    p = RoleParams(role="lead")
    dens = tk.densities(composed_bpm=tk.bpm)
    early = max(dens[:6])
    late = max(dens[8:16])
    assert late > early * p.handoff_density_ratio, (
        f"{name}: density did not ramp ({early:.2f} -> {late:.2f}), so the "
        f"yield below is not the ratio rule and this test is wrong about it")
    sts = tk.closed_loop(p, composed_bpm=tk.bpm)
    y = events_of(sts, "yield")
    assert y and sts[-1].role == "follow", \
        f"{name}: a 3x density ramp did not read as a takeover"
    assert "onsets/beat" in y[0].reason


@pytest.mark.parametrize("name", ("lofi80", "lofi80_quiet"))
def test_the_relative_key_ambiguity_does_not_move_a_real_arrangement(takes,
                                                                     name):
    """MEASURED: this C-major render is reported as A minor on every frame, at
    both gains.  On tonics that is 3 st and logs a spurious hold every take."""
    tk = takes[name]
    reported = Counter(h.key for h in tk.harm if h.key)
    assert reported, f"{name}: no key belief at all"
    top = reported.most_common(1)[0][0]
    assert parse_key(top).tonic != parse_key("C major").tonic, (
        f"{name}: reported {top!r} — the relative ambiguity this test is "
        f"about did not occur, so it proves nothing")
    sts = tk.closed_loop(RoleParams(role="lead"), composed_bpm=tk.bpm,
                         composed_key="C major")
    assert all(s.transpose_semitones == 0 for s in sts)
    assert not events_of(sts, "conform") and not events_of(sts, "conform_hold")


def test_a_real_render_a_semitone_up_is_met(takes):
    tk = takes["fsharp"]
    sts = tk.closed_loop(RoleParams(role="lead"), composed_bpm=tk.bpm,
                         composed_key="F minor")
    ev = events_of(sts, "conform")
    assert ev, f"heard {Counter(h.key for h in tk.harm).most_common(2)}"
    assert sts[-1].transpose_semitones == 1
    assert len(ev) == 1


def test_a_real_render_a_tritone_away_is_held_against(takes):
    tk = takes["tritone"]
    sts = tk.closed_loop(RoleParams(role="lead"), composed_bpm=tk.bpm,
                         composed_key="F minor")
    assert not events_of(sts, "conform")
    assert all(s.transpose_semitones == 0 for s in sts)
    assert events_of(sts, "conform_hold"), "held, but never said why"


def test_real_material_negotiation_is_deterministic(takes):
    tk = takes["trap140"]
    a = tk.closed_loop(RoleParams(role="lead"), composed_bpm=tk.bpm,
                       composed_key="D minor")
    b = tk.closed_loop(RoleParams(role="lead"), composed_bpm=tk.bpm,
                       composed_key="D minor")
    assert [dataclasses.astuple(s) for s in a] == \
        [dataclasses.astuple(s) for s in b]


def test_negotiation_is_nowhere_near_the_live_budget(takes, capsys):
    """v1's beat tracker runs ~180x real time and that headroom is why live
    jam is possible.  Role negotiation runs once per BAR, so its cost has to be
    invisible next to that; measured and printed rather than assumed."""
    tk = takes["trap140"]
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        sts = tk.closed_loop(RoleParams(role="lead"), composed_bpm=tk.bpm,
                             composed_key="D minor")
    wall = time.perf_counter() - t0
    ratio = (tk.dur_s * reps) / wall
    with capsys.disabled():
        print(f"\n  roles: {len(sts)} bars over {tk.dur_s:.1f}s audio, "
              f"{wall / reps * 1000:.2f} ms/take -> {ratio:,.0f}x real time")
    assert ratio > 2000.0, f"only {ratio:.0f}x real time"
