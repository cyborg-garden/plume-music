"""P0-C (commitment) and P1-E (reciprocal cueing) — plume.mind v1.4.

WHY THIS FILE EXISTS.  The first session with a human and an acoustic guitar
(CGCFCE, fingerpicked, built-in mic, plume's own output bleeding back in) came
back with one verdict:

    "it only plays like two random measures here and there. and then stops
     suddenly. and then starts up randomly later. [...] it actually has to be
     more opinionated in the sense of it cant chicken out. if it does really
     commit to something it has to also give the other musician a chance to
     adapt to that as well."

and one set of numbers, from plume's own status line during that take:

    tempo   167.9 -> 163.0 -> 169.2 -> 165.4 -> 80.6 -> 80.3 -> 121.0 bpm
    conf    oscillated 0.41 - 0.62 for the whole session
    state   "listening" almost the entire time, with brief excursions

0.41-0.62 straddles BOTH gates it has to clear: ``MindParams.entry_conf``
(0.45) and the beat tracker's own ``lock_conf`` (0.55).  So the machine
re-litigated its own presence at every bar boundary against a coin flip, and
the resulting behaviour — in, two bars, out, in again later — is exactly what
was reported.

Everything below is built from those numbers.  ``real_trace`` reconstructs the
session at bar granularity, and ``_UngatedMind`` restores the pre-fix logic so
every claim here is checked against a control that FAILS it: a test that only
shows the new code not flapping would pass just as happily on no input at all.
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from plume.listen.beat import BeatState
from plume.listen.harmony import HarmonyState
from plume.mind import (
    ANNOUNCE_KINDS, PARAM_HELP, BarFeatures, BarPlan, Mind, MindParams,
    _UNLOCK_GRACE_BARS,
)

# ------------------------------------------------------------------ builders


def mk_beat(*, bar: int = 0, conf: float = 0.9, locked: bool = True,
            bpm: float = 96.0) -> BeatState:
    interval = 60.0 / bpm if bpm > 0 else math.inf
    return BeatState(
        t_s=0.0, bpm=bpm, confidence=conf, locked=locked, bar=bar,
        beat_in_bar=0, phase_beats=0.0, next_beat_t_s=interval,
        next_downbeat_t_s=4 * interval if locked else math.inf,
        beat_interval_s=interval,
    )


HARM = HarmonyState(t_s=0.0, key="C major", key_conf=0.6, chord="C",
                    chord_conf=0.5, register_low_midi=48.0,
                    register_high_midi=76.0)


def run_script(m: Mind, steps: list[dict], *, start_bar: int = 0) -> list[BarPlan]:
    """Drive a Mind through scripted bars (same shape as test_mind.run_script)."""
    plans: list[BarPlan] = []
    bar = start_bar
    for step in steps:
        beat = mk_beat(bar=step.get("bar", bar), conf=step.get("conf", 0.9),
                       locked=step.get("locked", True),
                       bpm=step.get("bpm", 96.0))
        human = BarFeatures(bar=bar, rms_db=step.get("rms", -20.0),
                            onset_density=step.get("dens", 1.0),
                            silent=step.get("silent", False),
                            register_low_midi=step.get("lo"),
                            register_high_midi=step.get("hi"))
        bar += 1
        plans.append(m.plan_bar(beat, HARM, human))
    return plans


def small_params(**overrides) -> MindParams:
    """Short silence timers so the whole ladder is reachable quickly."""
    kw = dict(min_listen_bars=4, entry_boundary_bars=4, ramp_bars=3,
              silence_thin_bars=2, silence_fade_bars=4, silence_tacet_bars=6)
    kw.update(overrides)
    return MindParams(**kw)


WARM = [{}, {}, {}, {}]                  # 4 human bars -> entry plan at bar 4
TO_COMP = WARM + [{}, {}] + [{}]         # first comping plan at bar 7


def entries(plans: list[BarPlan]) -> list[int]:
    """Bars at which a listening->entering transition happened."""
    out = []
    prev = "listening"
    for p in plans:
        if p.state == "entering" and prev != "entering":
            out.append(p.bar)
        prev = p.state
    return out


# ------------------------------------------------- the reconstructed session

#: The measured confidence band, cycled bar by bar.  Endpoints are the reported
#: extremes (0.41, 0.62); the interior values straddle entry_conf 0.45 and the
#: tracker's lock_conf 0.55, which is what the status line showed it doing.
CONF_BAND = (0.52, 0.44, 0.58, 0.41, 0.62, 0.47, 0.43, 0.55)

#: The measured tempo estimates, in order.  165 vs 80 is an octave; 121 is
#: unrelated to either.  ``plan_bar`` does not read bpm — these ride along so
#: the BeatState fed to the mind is the one that was actually observed, and so
#: the fixture is not quietly a different session from the one it cites.
BPM_TRACE = (167.9, 163.0, 169.2, 165.4, 80.6, 80.3, 121.0)

#: The tracker's own lock threshold (``BeatTracker.lock_conf`` default).  The
#: reconstruction models ``locked`` as "confidence is clearing lock_conf",
#: which is the crudest reading of "confidence straddled 0.55" and reproduces
#: the reported symptom (mostly listening, brief excursions).  It is a model of
#: the SYMPTOM, not of the tracker; its job is to exercise the mind.  Fixing
#: the tracker on sustained acoustic material is P1-F and belongs to listen/.
TRACKER_LOCK_CONF = 0.55


def real_trace(n: int = 32) -> list[dict]:
    """The reconstructed session as ``run_script`` steps."""
    steps = []
    for i in range(n):
        conf = CONF_BAND[i % len(CONF_BAND)]
        steps.append({
            "conf": conf,
            "locked": conf >= TRACKER_LOCK_CONF,
            "bpm": BPM_TRACE[min(i // 5, len(BPM_TRACE) - 1)],
            "rms": -27.0,          # the measured live-mic level of that take
            "dens": 1.4,           # fingerpicked, sustained: not busy
        })
    return steps


class _UngatedMind(Mind):
    """The pre-fix (v1.2) presence logic, for non-vacuity controls.

    Two lines, both clearing v1.4 state at the top of every call:

    * ``_commit_left = 0`` — never committed, so the emergency floor is
      ``conf_floor`` again and an unlocked tracker is an emergency on its
      first bar, with no grace and no recommit lockout (an arc that ends can
      never have been "early").
    * ``_entry_pledge = None`` — an entry announced last bar is not honoured
      this bar, so the boundary confidence sample is a coin flip again.

    Nothing else is touched: same params, same trace, same everything.
    """

    def plan_bar(self, beat, harmony, human):        # noqa: D102
        self._commit_left = 0
        self._entry_pledge = None
        return super().plan_bar(beat, harmony, human)


# =====================================================================
# P0-C — commitment: one entry, not several
# =====================================================================


def test_the_real_trace_flaps_under_the_pre_fix_logic():
    """NON-VACUITY CONTROL.  This is the bug, reproduced.

    If this ever stops failing-as-designed the fixture has gone stale and
    every assertion below it is vacuous.
    """
    m = _UngatedMind(MindParams())
    plans = run_script(m, real_trace(32))
    got = entries(plans)
    assert len(got) >= 3, f"the fixture no longer reproduces the flap: {got}"
    states = [p.state for p in plans]
    # ...and it is the reported shape: in, a bar or two, straight back out.
    assert "tacet" in states and states.count("entering") <= len(got) + 2, states


def test_the_real_trace_produces_one_entry_and_no_flapping():
    """P0-C.  Same 32 bars, same confidences: in once, and stay in."""
    m = Mind(MindParams())
    plans = run_script(m, real_trace(32))
    got = entries(plans)
    assert len(got) <= 1, f"still flapping: entries at {got}"
    assert got, "commitment must not be achieved by never entering at all"
    tail = [p.state for p in plans if p.bar > got[0]]
    assert "tacet" not in tail, f"left after committing: {tail}"
    assert "fading" not in tail, f"began leaving after committing: {tail}"


def test_the_entry_lands_on_a_phrase_boundary_not_mid_phrase():
    m = Mind(MindParams())
    plans = run_script(m, real_trace(32))
    for bar in entries(plans):
        assert bar % MindParams().entry_boundary_bars == 0, bar


def test_commitment_rides_the_confidence_dips_it_used_to_die_on():
    """Every bar of the arc sat in the band that used to be a coin flip."""
    m = Mind(MindParams())
    plans = run_script(m, real_trace(32))
    after = [p for p in plans if p.bar > entries(plans)[0]]
    assert len(after) >= 16
    assert all(p.state in ("entering", "comping") for p in after), \
        [p.state for p in after]


def test_determinism_on_the_real_trace():
    a = run_script(Mind(MindParams()), real_trace(32))
    b = run_script(Mind(MindParams()), real_trace(32))
    assert a == b


# =====================================================================
# ...and it must NOT become "never stops"
# =====================================================================


def test_a_genuine_silence_still_walks_thinned_fading_tacet_inside_commit():
    """The silence ladder is exempt from commitment, start to finish.

    The human goes quiet on the bar AFTER the entry, so every rung of the
    ladder lands while the commitment is still running.
    """
    p = small_params()
    m = Mind(p)
    plans = run_script(m, WARM + [{"silent": True}] * 7)
    states = [pl.state for pl in plans]
    assert states[3] == "entering"                    # entry at bar 4
    ladder = states[4:]
    assert "thinned" in ladder and "fading" in ladder and "tacet" in ladder, ladder
    assert ladder.index("thinned") < ladder.index("fading") < ladder.index("tacet")
    tacet = [pl for pl in plans if pl.state == "tacet"][0]
    assert tacet.bar - 4 < p.commit_bars, "the ladder ran outside the window"
    assert "entry re-armed" in tacet.reason


def test_a_human_who_stops_after_many_renewals_is_still_met_with_silence():
    """Commitment RENEWS while the music runs — prove it still ends."""
    m = Mind(small_params())
    run_script(m, TO_COMP + [{}] * 20)                # renews many times
    states = [p.state for p in run_script(m, [{"silent": True}] * 6,
                                          start_bar=27)]
    assert states[-1] == "tacet", states


def test_a_sustained_unlock_still_triggers_repair_inside_commit_bars():
    """A TRUE unlock is an emergency even mid-commitment."""
    m = Mind(MindParams())
    plans = run_script(m, TO_COMP
                       + [{"conf": 0.6, "locked": False}] * (_UNLOCK_GRACE_BARS + 1))
    assert plans[-1].state == "fading", [p.state for p in plans]
    assert "tracker unlocked" in plans[-1].reason
    assert m._bars_since_entry < MindParams().commit_bars, \
        "the repair must be provably inside the commitment window"


def test_a_wobble_shorter_than_the_grace_holds_the_last_bar():
    """...but a wobble is not an unlock.  This is the flap, at bar scale."""
    m = Mind(MindParams())
    plans = run_script(m, TO_COMP
                       + [{"conf": 0.6, "locked": False}] * _UNLOCK_GRACE_BARS)
    held = plans[-_UNLOCK_GRACE_BARS:]
    assert all(p.state == "comping" for p in held), [p.state for p in held]
    assert all(p.reason.startswith("tracker wobbled") for p in held)
    # holding is not the same as pretending: it says so, and it does not
    # advance the arc's own clocks.
    assert all("holding" in p.reason for p in held)


def test_a_held_wobble_does_not_burn_the_commitment():
    """Otherwise a long glitch runs the window out and restores the bug."""
    m = Mind(MindParams())
    run_script(m, TO_COMP)
    before = m._commit_left
    run_script(m, [{"conf": 0.6, "locked": False}] * _UNLOCK_GRACE_BARS,
               start_bar=7)
    assert m._commit_left == before


def test_a_recovered_beat_resets_the_repair_so_the_next_loss_fades_first():
    """The repair is fade-then-tacet.  Before the wobble grace a recovery
    between the two was nearly impossible, so a stale stage counter never
    showed; now recoveries are the common case and a second loss must not
    cut straight to silence."""
    m = Mind(MindParams())
    plans = run_script(m, TO_COMP + [{"conf": 0.05}, {}, {"conf": 0.05}])
    assert [p.state for p in plans[-3:]] == ["fading", "comping", "fading"]


def test_confidence_below_exit_conf_still_ends_a_committed_arc():
    m = Mind(MindParams())
    p = m.params
    plans = run_script(m, TO_COMP + [{"conf": p.exit_conf - 0.05}] * 2)
    assert [pl.state for pl in plans[-2:]] == ["fading", "tacet"]
    assert f"floor {p.exit_conf:.2f}" in plans[-2].reason


# =====================================================================
# deep hysteresis: the band that keeps you in vs the band that gets you in
# =====================================================================


def test_the_staying_band_is_wider_than_the_entering_band():
    """One confidence, two answers — which is the whole point of P0-C."""
    p = MindParams()
    mid = 0.28                       # > exit_conf 0.25, < conf_floor 0.30 < entry_conf 0.45
    assert p.exit_conf < mid < p.conf_floor < p.entry_conf

    inside = Mind(MindParams())
    run_script(inside, TO_COMP)
    held = run_script(inside, [{"conf": mid}] * 4, start_bar=7)
    assert all(pl.state == "comping" for pl in held), [pl.state for pl in held]

    outside = Mind(MindParams())
    never = run_script(outside, [{"conf": mid}] * 16)
    assert all(pl.state in ("waiting", "listening") for pl in never)


def test_an_early_exit_locks_out_the_next_boundary():
    """"and then starts up randomly later" — recommit_bars."""
    m = Mind(MindParams())
    run_script(m, WARM + [{"conf": 0.05}, {"conf": 0.05}])   # fade 5, tacet 6
    plans = run_script(m, [{}] * 8, start_bar=6)             # bars 7..14
    by_bar = {pl.bar: pl for pl in plans}
    assert by_bar[8].state == "listening", by_bar[8]
    assert "sitting out" in by_bar[8].reason
    assert by_bar[12].state == "entering", by_bar[12]


def test_a_human_who_stopped_is_never_locked_out_when_they_come_back():
    """MEASURED on the ``long_stop`` fixture, and it cost a product claim.

    A long stop takes the tracker's lock with it, so the arc ends through
    beat-loss repair with the silence counters frozen mid-ladder and the
    commitment still nominally running.  Scoping the lockout on the
    commitment alone made the machine sit out the next boundary after the
    human came back — punishing them for a commitment THEY had not broken,
    and (on that fixture) never re-entering at all.  The lockout is for the
    machine bailing on a player who was still going.
    """
    m = Mind(small_params())
    plans = run_script(m, TO_COMP + [{"silent": True}] * 3
                       + [{"silent": True, "conf": 0.1}] * 2)
    assert [p.state for p in plans[-2:]] == ["fading", "tacet"]
    assert m._silent_bars > 0, "the counters must be frozen mid-ladder"
    assert m._exit_early is False
    back = run_script(m, [{}] * 4, start_bar=12)             # bars 13..16
    assert [p.state for p in back] == ["listening"] * 3 + ["entering"], back


def test_no_lockout_when_the_commitment_was_served():
    """A human who stopped and came back is met at the next boundary."""
    m = Mind(small_params())
    run_script(m, TO_COMP + [{"silent": True}] * 6)          # tacet at bar 13
    assert m._exit_early is False
    plans = run_script(m, [{}] * 3, start_bar=13)            # bars 14, 15, 16
    assert [p.state for p in plans] == ["listening", "listening", "entering"]


# =====================================================================
# P1-E — announce, so the human can adapt
# =====================================================================


def test_announce_fields_are_consistent_or_absent():
    """The three fields move together, and always name the NEXT bar."""
    for steps in (real_trace(32), TO_COMP + [{"silent": True}] * 8,
                  TO_COMP + [{"dens": 3.0}, {"dens": 3.0}] + [{"dens": 1.0}] * 4):
        m = Mind(small_params())
        for plan in run_script(m, steps):
            trio = (plan.announce, plan.announce_kind, plan.announce_bar)
            assert (all(x is None for x in trio)
                    or all(x is not None for x in trio)), plan
            if plan.announce_kind is not None:
                assert plan.announce_kind in ANNOUNCE_KINDS, plan
                assert plan.announce_bar == plan.bar + 1, plan
                assert f"announcing {plan.announce_kind} next bar" in plan.reason
                assert plan.announce in plan.reason


def test_entry_is_announced_the_bar_before_and_on_a_boundary():
    m = Mind(MindParams(min_listen_bars=3))
    plans = run_script(m, WARM)
    pickup, entry = plans[2], plans[3]
    assert pickup.state == "listening"
    assert pickup.announce_kind == "entry" and pickup.announce_bar == 4
    assert "announcing entry next bar (pickup)" in pickup.reason
    assert entry.state == "entering" and entry.bar == 4
    assert entry.bar % m.params.entry_boundary_bars == 0     # never mid-phrase
    assert "honouring the pickup announced last bar" in entry.reason


def test_an_announced_entry_is_not_chickened_out_of():
    """P1-E's reason for existing: an announcement the machine may renege on
    is worse than no announcement.  A confidence dip at the pledged bar does
    NOT cancel the entry; never being armed in the first place does."""
    dip = 0.30                          # < entry_conf 0.45, > exit_conf 0.25
    pledged = Mind(MindParams(min_listen_bars=3))
    a = run_script(pledged, [{}, {}, {}, {"conf": dip}])
    assert a[2].announce_kind == "entry"
    assert a[3].state == "entering", a[3]

    unpledged = Mind(MindParams(min_listen_bars=3))
    b = run_script(unpledged, [{}, {}, {"conf": dip}, {"conf": dip}])
    assert b[2].announce_kind is None
    assert b[3].state == "listening", b[3]


def test_every_rung_of_the_silence_ladder_is_announced_one_bar_ahead():
    m = Mind(small_params())            # thin 2, fade 4, tacet 6
    plans = run_script(m, TO_COMP + [{"silent": True}] * 6)
    by_bar = {p.bar: p for p in plans}
    for announced_at, kind, lands in ((8, "texture", "thinned"),
                                      (10, "exit", "fading"),
                                      (12, "exit", "tacet")):
        ann = by_bar[announced_at]
        assert ann.announce_kind == kind, (announced_at, ann)
        assert ann.announce_bar == announced_at + 1
        assert by_bar[announced_at + 1].state == lands, by_bar[announced_at + 1]


def test_a_plan_never_announces_the_change_it_is_already_making():
    """A second fading bar is not news; announcing it would be a lying log."""
    m = Mind(small_params())
    plans = run_script(m, TO_COMP + [{"silent": True}] * 6)
    for prev, cur in zip(plans, plans[1:]):
        if prev.announce_kind is not None:
            assert cur.state != prev.state, (prev, cur)
    assert all(p.announce is None for p in plans if p.state == "tacet")


def test_the_density_recovery_is_announced_one_bar_ahead():
    """The coupling already delays recovery by a bar; that bar is the cue."""
    m = Mind(MindParams())
    run_script(m, TO_COMP)
    seq = [{"dens": 3.0}, {"dens": 3.0}, {"dens": 1.0}, {"dens": 1.0}]
    plans = run_script(m, seq, start_bar=7)
    assert plans[2].announce_kind == "texture"
    assert plans[2].announce_bar == plans[3].bar
    assert "recovering next bar" in plans[2].reason
    assert plans[3].density > plans[2].density          # the change landed


def test_thinning_is_not_announced_but_recovery_is():
    """Asymmetric on purpose: thinning is a REACTION to the human's own
    busyness (they already know), recovery is the machine's initiative."""
    m = Mind(MindParams())
    run_script(m, TO_COMP)
    plans = run_script(m, [{"dens": 1.0}, {"dens": 3.0}], start_bar=7)
    assert plans[0].announce is None and plans[1].announce is None
    assert plans[1].density < plans[0].density          # it thinned, silently


def test_beat_loss_is_never_announced():
    """An emergency has no bar of notice, and must not claim to."""
    m = Mind(MindParams())
    plans = run_script(m, TO_COMP + [{"conf": 0.05}, {"conf": 0.05}])
    assert [p.state for p in plans[-2:]] == ["fading", "tacet"]
    assert all(p.announce is None for p in plans[-2:])


def test_an_answered_exit_announcement_is_cancelled():
    """The human plays again after the warning: the cue did its job."""
    m = Mind(small_params())
    plans = run_script(m, TO_COMP + [{"silent": True}] * 5)
    warned = plans[-1]
    assert warned.announce_kind == "exit"               # tacet was coming
    assert warned.announce_bar == warned.bar + 1
    resumed = run_script(m, [{}, {}], start_bar=12)
    assert [p.state for p in resumed] == ["comping", "comping"], resumed
    assert all(p.announce is None for p in resumed)


# =====================================================================
# hygiene
# =====================================================================


def test_new_params_are_documented_and_typed():
    fields = {f.name: f for f in dataclasses.fields(MindParams)}
    for name in ("commit_bars", "exit_conf", "recommit_bars"):
        assert name in fields, name
        assert PARAM_HELP.get(name), f"{name} is an undocumented dial"
    p = MindParams()
    assert isinstance(p.commit_bars, int) and isinstance(p.recommit_bars, int)
    assert p.exit_conf < p.conf_floor < p.entry_conf, \
        "the exit band must be strictly wider than the entry band"


def test_commit_bars_zero_restores_the_pre_fix_behavior():
    """The dial is real: turned off, the flap comes back.  A parameter that
    changes nothing at its extreme is a parameter that is not wired up."""
    m = Mind(MindParams(commit_bars=0, recommit_bars=0))
    plans = run_script(m, real_trace(32))
    assert len(entries(plans)) >= 3, entries(plans)


@pytest.mark.parametrize("commit", [1, 4, 8, 32])
def test_commitment_never_prevents_the_first_entry(commit):
    m = Mind(MindParams(commit_bars=commit))
    plans = run_script(m, WARM)
    assert plans[-1].state == "entering" and plans[-1].bar == 4
