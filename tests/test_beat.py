"""BeatTracker tests (SPEC-V1 section 2).

All musical tests drive the tracker with synthetic Frame streams constructed
directly (no audio needed): flux impulse trains with controlled onset times,
strengths, jitter, and chroma.  One audio-path test synthesizes a click-track
WAV in-test and runs it through listen.features.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from plume.listen import beat as beat_mod
from plume.listen.beat import BeatState, BeatTracker

FRAME_DT = 512.0 / 48000.0
_ZERO12 = np.zeros(12, dtype=np.float32)


@dataclass(frozen=True)
class FakeFrame:
    """Attribute-compatible stand-in for plume.listen.features.Frame."""
    t_s: float
    rms_db: float
    flux: float
    onset: bool
    onset_strength: float
    chroma: np.ndarray = field(default_factory=lambda: _ZERO12)
    centroid_hz: float = 0.0
    self_masked: bool = False


def make_frames(dur_s, onset_times, *, strengths=None, chroma_fn=None,
                masked=False):
    """Frame stream with a 3-frame flux bump per onset (peak = onset frame)."""
    n = int(round(dur_s / FRAME_DT))
    flux = np.zeros(n)
    onset_at: dict[int, float] = {}
    for j, t in enumerate(onset_times):
        i = int(round(t / FRAME_DT)) - 1          # frame END nearest t
        if 1 <= i < n - 1:
            s = strengths[j] if strengths is not None else 1.0
            flux[i - 1] = max(flux[i - 1], 0.4 * s)
            flux[i] = max(flux[i], float(s))
            flux[i + 1] = max(flux[i + 1], 0.5 * s)
            onset_at[i] = max(onset_at.get(i, 0.0), float(s))
    frames = []
    for i in range(n):
        t = (i + 1) * FRAME_DT
        chroma = chroma_fn(t) if chroma_fn is not None else _ZERO12
        frames.append(FakeFrame(
            t_s=t,
            rms_db=-20.0 if flux[i] > 0 else -90.0,
            flux=float(flux[i]),
            onset=i in onset_at,
            onset_strength=onset_at.get(i, 0.0),
            chroma=chroma,
            self_masked=masked and i in onset_at,
        ))
    return frames


def click_times(bpm, n_beats, t0=0.5):
    return [t0 + k * 60.0 / bpm for k in range(n_beats)]


def run(tracker, frames, chunk=8):
    """Feed frames in chunks; return the state after every chunk."""
    return [tracker.process(frames[k:k + chunk])
            for k in range(0, len(frames), chunk)]


def nearest_grid_err(t, bpm, t0):
    period = 60.0 / bpm
    return abs((t - t0 + period / 2) % period - period / 2)


# ---------------------------------------------------------------- basics


def test_initial_state_defaults():
    st = BeatTracker().process([])
    assert st.bpm == 0.0
    assert st.confidence == 0.0
    assert not st.locked
    assert st.bar == -1
    assert st.beat_in_bar == 0
    assert st.phase_beats == 0.0
    assert math.isinf(st.next_beat_t_s)
    assert math.isinf(st.next_downbeat_t_s)
    assert math.isinf(st.beat_interval_s)


def test_silent_stream_stays_unestimated():
    tr = BeatTracker()
    st = tr.process(make_frames(2.0, []))
    assert st.bpm == 0.0 and not st.locked
    assert math.isinf(st.next_beat_t_s)
    assert st.t_s == pytest.approx(int(round(2.0 / FRAME_DT)) * FRAME_DT)


def test_constructor_validation():
    with pytest.raises(ValueError):
        BeatTracker(tightness=0.0)
    with pytest.raises(ValueError):
        BeatTracker(bpm_range=(200.0, 60.0))


# ------------------------------------------------------------- metronome


def test_metronome_locks_fast_and_accurately():
    bpm = 120.0
    frames = make_frames(17.0, click_times(bpm, 32, t0=0.5))
    tr = BeatTracker()
    states = run(tr, frames)

    locked = [s for s in states if s.locked]
    assert locked, "never locked on a metronome"
    # lock definition needs 2 bars of high confidence on top of the first
    # tempo estimate, so "~2 bars" lands at <= 4 bars of music in practice
    assert locked[0].t_s <= 0.5 + 4 * 4 * 60.0 / bpm

    final = states[-1]
    assert final.locked
    assert final.bpm == pytest.approx(bpm, abs=2.0)
    assert final.confidence >= 0.55
    # predicted next beat within 25ms of the true grid
    assert nearest_grid_err(final.next_beat_t_s, bpm, 0.5) < 0.025
    # once locked, stays locked
    first_lock = states.index(locked[0])
    assert all(s.locked for s in states[first_lock:])
    # beat interval consistent with bpm
    assert final.beat_interval_s == pytest.approx(60.0 / final.bpm, rel=1e-9)
    assert 0.0 <= final.phase_beats < 4.0


def test_metronome_bar_counter_advances():
    frames = make_frames(20.0, click_times(120.0, 38, t0=0.5))
    states = run(BeatTracker(), frames)
    bars = [s.bar for s in states if s.locked]
    assert bars[0] == 0
    assert bars[-1] >= 2                       # kept counting after lock
    assert bars == sorted(bars)                # monotonic


# ----------------------------------------------------------------- drift


def test_tempo_drift_followed_without_half_double_jump():
    # 8 bars at 120, then +6% linearly over 16s, then hold at 127.2
    bpm0, bpm1 = 120.0, 127.2
    t_ramp0, t_ramp1 = 16.5, 32.5

    def bpm_at(t):
        if t < t_ramp0:
            return bpm0
        if t < t_ramp1:
            return bpm0 + (bpm1 - bpm0) * (t - t_ramp0) / (t_ramp1 - t_ramp0)
        return bpm1

    times, t = [], 0.5
    while t < 47.0:
        times.append(t)
        t += 60.0 / bpm_at(t)
    frames = make_frames(48.0, times)
    states = run(BeatTracker(), frames)

    locked = [s for s in states if s.locked]
    assert locked and states[-1].locked
    first_lock = states.index(locked[0])
    post = states[first_lock:]
    # never a half/double jump: bpm stays near the truth band the whole time
    assert all(105.0 <= s.bpm <= 140.0 for s in post)
    assert all(s.locked for s in post)
    # within +-4 bpm of the new tempo by 2 bars after the drift ends
    settle_t = t_ramp1 + 2 * 4 * 60.0 / bpm1
    tail = [s for s in states if s.t_s >= settle_t]
    assert tail
    assert all(s.bpm == pytest.approx(bpm1, abs=4.0) for s in tail)


# ---------------------------------------------------------------- jitter


def test_jittered_onsets_stay_locked_high_confidence():
    bpm = 120.0
    rng = np.random.default_rng(42)            # explicit seed: deterministic
    grid = np.asarray(click_times(bpm, 48, t0=0.5))
    times = grid + rng.uniform(-0.02, 0.02, size=grid.size)  # zero-mean, no bias
    frames = make_frames(25.5, list(times))
    states = run(BeatTracker(), frames)

    locked = [s for s in states if s.locked]
    assert locked
    assert locked[0].t_s <= 0.5 + 5 * 4 * 60.0 / bpm
    first_lock = states.index(locked[0])
    assert all(s.locked for s in states[first_lock:])       # never unlocks
    tail = [s for s in states if s.t_s >= 20.0]
    assert np.mean([s.confidence for s in tail]) > 0.5      # confidence high
    assert states[-1].bpm == pytest.approx(bpm, abs=3.0)


# --------------------------------------------------------------- silence


def test_silence_decays_confidence_and_unlocks():
    bpm = 120.0
    frames = make_frames(30.0, click_times(bpm, 32, t0=0.5))  # music ends 16s
    states = run(BeatTracker(), frames)

    playing = [s for s in states if s.t_s <= 16.0]
    assert playing[-1].locked and playing[-1].confidence > 0.55

    silent = [s for s in states if s.t_s > 17.0]
    unlocked = [s for s in silent if not s.locked]
    assert unlocked, "never unlocked during extended silence"
    # Bound relaxed 28.0 -> 29.0 for v1.2.  The tracker unlocks by decaying
    # confidence at SILENT_DECAY per beat after the SILENT_HOLD_BEATS ride-out,
    # so the unlock time is a function of the confidence held when the music
    # stopped — and v1.2 measures that confidence correctly instead of
    # under-reporting it (salience used to be read off the candidate's own lag
    # only, which is weak on real material; see _evaluate_tempo).  Here it is
    # 0.967 at t=16s where it used to be lower, so the same decay crosses the
    # threshold 0.2s later.  The BEHAVIOR under test — a long stop unlocks,
    # the bar counter freezes, the phase free-runs — is unchanged.
    assert unlocked[0].t_s < 29.0
    assert states[-1].confidence < 0.3
    # confidence decayed monotonically-ish: end far below the playing level
    assert states[-1].confidence < playing[-1].confidence / 2
    # bar counter freezes after unlock (never resets, never advances)
    bar_at_unlock = unlocked[0].bar
    assert states[-1].bar == bar_at_unlock
    # phase free-runs: prediction still finite and period retained
    assert math.isfinite(states[-1].next_beat_t_s)
    assert states[-1].bpm == pytest.approx(bpm, abs=3.0)


# ------------------------------------------------------------------ taps


def test_tap_count_in_hard_sets_lock_tempo_and_phase():
    tr = BeatTracker()
    for t in (0.5, 1.0, 1.5, 2.0):
        tr.tap(t)
    st = tr.process(make_frames(2.1, []))
    assert st.locked
    assert st.confidence == pytest.approx(1.0)
    assert st.bpm == pytest.approx(120.0, abs=1e-9)
    assert st.bar == 0
    # last tap = beat 0 of bar 0 (at 2.0) -> next beat 2.5, bar 1 starts 4.0
    assert st.next_beat_t_s == pytest.approx(2.5, abs=1e-9)
    assert st.next_downbeat_t_s == pytest.approx(4.0, abs=1e-9)


def test_tap_uneven_spacing_does_not_lock():
    tr = BeatTracker()
    for t in (0.5, 0.9, 1.6, 2.0):             # wildly uneven
        tr.tap(t)
    st = tr.process(make_frames(2.1, []))
    assert not st.locked
    assert st.bpm == 0.0


def test_tap_then_more_even_taps_still_locks():
    tr = BeatTracker()
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):   # 6 even taps
        tr.tap(t)
    st = tr.process(make_frames(2.6, []))
    assert st.locked and st.bpm == pytest.approx(120.0, abs=1e-6)


# ---------------------------------------------------- alpha phase correction


def _alpha_setup(tightness):
    tr = BeatTracker(tightness=tightness)
    for t in (0.0, 0.5, 1.0, 1.5):             # count-in: beat grid on 0.5s
        tr.tap(t)
    return tr


@pytest.mark.parametrize("tightness", [0.35, 0.6])
def test_alpha_correction_is_error_times_tightness(tightness):
    tr = _alpha_setup(tightness)
    t_on = 189 * FRAME_DT                      # ~2.016s, i.e. 16ms late vs beat @2.0
    st = tr.process(make_frames(2.3, [t_on]))
    err = t_on - 2.0
    assert st.next_beat_t_s == pytest.approx(2.5 + tightness * err, abs=1e-9)


def test_alpha_correction_early_onset():
    tr = _alpha_setup(0.35)
    t_on = 185 * FRAME_DT                      # ~1.973s, 27ms early vs beat @2.0
    # stop the stream before the corrected prediction is reached/emitted
    st = tr.process(make_frames(1.985, [t_on]))
    err = t_on - 2.0                           # negative: pulls prediction earlier
    assert st.next_beat_t_s == pytest.approx(2.0 + 0.35 * err, abs=1e-9)


def test_onset_outside_35pct_window_is_ignored():
    tr = _alpha_setup(0.35)
    t_on = 210 * FRAME_DT                      # ~2.24s: 240ms > 35% of 500ms
    st = tr.process(make_frames(2.3, [t_on]))
    assert st.next_beat_t_s == pytest.approx(2.5, abs=1e-9)


def test_self_masked_onsets_do_not_correct_phase():
    tr = _alpha_setup(0.35)
    t_on = 189 * FRAME_DT
    st = tr.process(make_frames(2.3, [t_on], masked=True))
    assert st.next_beat_t_s == pytest.approx(2.5, abs=1e-9)


# -------------------------------------------------------------- downbeat


def test_downbeat_from_accents_and_chroma_change():
    bpm = 120.0
    bar_s = 4 * 60.0 / bpm
    n_beats = 40
    times = click_times(bpm, n_beats, t0=0.5)
    strengths = [2.0 if k % 4 == 0 else 1.0 for k in range(n_beats)]

    def chroma_fn(t):
        bar = int((t - 0.5) // bar_s)
        c = np.zeros(12, dtype=np.float32)
        c[(bar * 3) % 12] = 1.0                # harmonic change on each downbeat
        return c

    frames = make_frames(21.0, times, strengths=strengths, chroma_fn=chroma_fn)
    states = run(BeatTracker(), frames)
    final = states[-1]
    assert final.locked
    # predicted next downbeat aligns with the true bar grid (bars start at 0.5)
    err = abs((final.next_downbeat_t_s - 0.5 + bar_s / 2) % bar_s - bar_s / 2)
    assert err < 0.06
    assert math.isfinite(final.next_downbeat_t_s)
    assert final.next_downbeat_t_s > final.t_s


# ------------------------------------------------- half/double resolution


def test_true_60bpm_not_spuriously_doubled():
    # quarter-note clicks at 60: no autocorr support at 0.5s lag, so the
    # tracker must NOT invent 120 just because 60 is outside 70-160
    frames = make_frames(20.0, click_times(60.0, 20, t0=0.5))
    final = run(BeatTracker(), frames)[-1]
    assert final.bpm == pytest.approx(60.0, abs=2.0)


def test_200bpm_clicks_resolve_to_half_toward_preferred_range():
    # clicks at 200 have exact autocorr support at the double period; the
    # half/double check resolves toward 70-160 -> 100 bpm (half-time feel)
    frames = make_frames(20.0, click_times(200.0, 64, t0=0.5))
    final = run(BeatTracker(), frames)[-1]
    assert final.bpm == pytest.approx(100.0, abs=3.0)
    assert final.locked


# ------------------------------------------- octave stickiness (v1.3)
#
# THE DEFECT THESE GUARD.  A tracker that has locked and is being played along
# with was able to change tempo OCTAVE on two evaluations — one second — of
# contrary evidence, exactly like a 4% drift correction, and to do it in
# total silence.  Measured on a 32-bar 140bpm trap render: locked at t=4.1s,
# held 139.8-140.7 for fifty seconds and thirty bar boundaries, then halved to
# 69.88 at t=54.1 and stayed there.  Confidence ROSE across the collapse
# (0.694 -> 0.736 -> 0.769), because a half-tempo grid predicts half as many
# beats and all of them are still beats the human played, so the hit rate went
# 0.875 -> 1.000.  Nothing downstream could tell.  The end-to-end guard is in
# tests/test_real_material.py (it needs a take long enough to settle); these
# pin the mechanism.


def _double_time_frames(bpm=80.0, switch_s=22.0, dur=70.0):
    """A genuine, sustained double-time feel: ``bpm`` then 2x, accented 4/4."""
    times, strengths, t, k = [], [], 0.5, 0
    while t < dur - 1.0:
        times.append(t)
        strengths.append(1.6 if k % 4 == 0 else 1.0)
        t += 60.0 / (bpm if t < switch_s else bpm * 2.0)
        k += 1
    return make_frames(dur, times, strengths=strengths)


def _settled_tracker(bpm=120.0, dur=None):
    """A tracker locked on ``bpm`` for longer than ``OCTAVE_SETTLE_S``."""
    dur = dur if dur is not None else beat_mod.OCTAVE_SETTLE_S + 14.0
    n = int(dur * bpm / 60.0) + 4
    tr = BeatTracker()
    tr.process(make_frames(dur, click_times(bpm, n, t0=0.5)))
    return tr


def test_is_octave_jump_only_fires_on_whole_octaves():
    """The gate must catch x2 / /2 (and the double octave) and nothing else.

    The PREDICATE is unchanged since v1.3; what it decides is not.  It used to
    decide whether a locked re-read faced the commitment gate at all, and its
    v1.3 rationale was that a 4/3 dotted alias — the relative real hat patterns
    produce, and the one the v1.2 full lag sweep exists to reach — must stay on
    the fast 2-evaluation path or the fix would freeze the tracker onto its own
    dotted aliases.  Live acoustic falsified that: 4/3 is the dominant COLLAPSE
    family (see ``_grid_hysteresis``), so v2.7 sends every out-of-drift re-read
    through the gate and lets SETTLEDNESS carry the v1.3 concern instead.

    All this predicate decides now is which name the change is announced under
    — ``tempo_octave_change`` vs ``tempo_grid_change`` — and which of the two
    counters moves, so that mind.py's ritard guard keeps its v1.3 meaning.
    """
    j = beat_mod._is_octave_jump
    assert j(140.0, 70.0) and j(70.0, 140.0)
    assert j(140.0, 69.0) and j(140.0, 71.5)      # inside the 6% tolerance
    assert j(160.0, 40.0)                          # two octaves down
    assert not j(140.0, 186.7)                     # 4/3 dotted alias
    assert not j(140.0, 93.3)                      # 2/3
    assert not j(140.0, 210.0)                     # 3/2
    assert not j(140.0, 144.0)                     # ordinary drift
    assert not j(140.0, 80.0)                      # not near an octave
    assert not j(0.0, 70.0) and not j(140.0, 0.0)  # no tempo yet


def test_a_settled_octave_costs_more_evaluations_than_an_unsettled_one():
    """The whole rule, on the gate itself.

    Before the octave has settled the tracker is still deciding what it is
    hearing and must be able to correct itself in two evaluations — that is
    how plume locks its own 80bpm backbeat groove, whose true tempo only
    overtakes the 160 alias 1.5 s after the lock.  After it has settled, the
    same change costs ``OCTAVE_SWITCH_EVALS``.
    """
    def evals_to_switch(settled):
        tr = _settled_tracker()
        if not settled:                       # pretend the lock just happened
            tr._lock_t = tr._t
            tr._grid_t0 = tr._t
        assert tr._locked and tr._bpm == pytest.approx(120.0, abs=3.0)
        half = tr._bpm / 2.0
        for i in range(1, 40):
            tr._adopt_bpm(half, tr._t + 0.5 * i)
            if tr._bpm == pytest.approx(half, rel=1e-9):
                return i
        return None

    assert evals_to_switch(settled=False) == 2
    assert evals_to_switch(settled=True) == beat_mod.OCTAVE_SWITCH_EVALS


def test_a_settled_octave_ignores_a_challenger_that_does_not_persist():
    """One evaluation where the grid in force wins back resets the clock.

    This is the measured difference between the collapse (6 consecutive
    evaluations, then gone) and a real correction (82 unbroken).  A challenger
    that keeps being interrupted must never accumulate its way to a switch,
    however many times it appears in total.
    """
    tr = _settled_tracker()
    half, t = tr._bpm / 2.0, tr._t
    for i in range(60):
        # 6 evaluations of the challenger, then one where the grid wins back
        t += 0.5
        tr._adopt_bpm(half if i % 7 else tr._bpm, t)
    assert tr._bpm == pytest.approx(120.0, abs=3.0)
    assert tr.events == ()
    assert tr.process([]).tempo_octave_changes == 0


def test_a_genuine_double_time_feel_still_changes_octave_and_announces_it():
    """The octave is sticky, not welded.

    A human who really does go into double time is a legitimate octave change,
    and the tracker must follow — LATE (a phrase later, by construction) but
    surely, and out loud.  Confidence drops on the change: re-reading the
    metre of a take you have been playing along with is a loss of certainty
    about what the human is doing, not the gain the mechanical hit rate on a
    new grid would otherwise report.
    """
    states = run(BeatTracker(), _double_time_frames())
    tr_states = [s for s in states if s.locked]
    assert tr_states, "never locked"
    assert tr_states[0].bpm == pytest.approx(80.0, abs=3.0)
    assert states[-1].bpm == pytest.approx(160.0, abs=4.0), (
        "a sustained, unambiguous double-time was never followed")
    # ... and it was announced, exactly once, with confidence going DOWN
    changed = [s for s in states if s.tempo_octave_changes]
    assert changed and changed[-1].tempo_octave_changes == 1


def test_the_octave_change_event_carries_the_before_and_after():
    """``BeatTracker.events`` is the forensic record — it must be readable
    without re-running the take, and it must show the confidence CUT."""
    tr = BeatTracker()
    run(tr, _double_time_frames())
    evs = [e for e in tr.events if e["event"] == "tempo_octave_change"]
    assert len(evs) == 1, [e["t_s"] for e in evs]
    e = evs[0]
    assert e["from_bpm"] == pytest.approx(80.0, abs=3.0)
    assert e["to_bpm"] == pytest.approx(160.0, abs=4.0)
    assert e["settled"] is True
    assert e["conf_after"] < e["conf_before"], (
        "a re-interpretation of the metre reported MORE certainty, not less")
    assert e["conf_after"] == pytest.approx(
        e["conf_before"] * beat_mod._OCTAVE_CONF_DROP, rel=1e-9)
    # the events view is a copy: callers cannot corrupt tracker state
    tr.events[0]["from_bpm"] = -1.0
    assert tr.events[0]["from_bpm"] == e["from_bpm"]


def test_a_settled_grid_resists_a_dotted_alias_exactly_like_an_octave():
    """v2.7: the commitment gate is about commitment, not about the ratio.

    120 -> 160 is a 4/3 dotted alias: not an octave, 33% out of the drift
    band, and on live acoustic the single most common way a settled grid gets
    re-read (22 of 54 on ``takes6/``, against 2 octaves).  Before v2.7 it took
    the ordinary two-evaluation path and switched in one second.
    """
    tr = _settled_tracker()                       # 120bpm, locked, settled
    assert tr._grid_is_settled(tr._t)
    alias = tr._bpm * 4.0 / 3.0
    assert not beat_mod._is_octave_jump(tr._bpm, alias), "not a 4/3 test"
    switched_at = None
    for i in range(1, 40):
        tr._adopt_bpm(alias, tr._t + 0.5 * i)
        if tr._bpm == pytest.approx(alias, rel=1e-9):
            switched_at = i
            break
    assert switched_at == beat_mod.OCTAVE_SWITCH_EVALS, (
        f"a settled grid yielded to a 4/3 alias in {switched_at} evaluations")


def test_a_settled_grid_ignores_a_dotted_alias_that_does_not_persist():
    """The measured difference between a collapse and a correction is
    PERSISTENCE, and it must apply to dotted aliases too: one evaluation where
    the grid in force wins back resets the clock."""
    tr = _settled_tracker()
    alias, t = tr._bpm * 4.0 / 3.0, tr._t
    for i in range(60):
        t += 0.5
        tr._adopt_bpm(alias if i % 7 else tr._bpm, t)
    assert tr._bpm == pytest.approx(120.0, abs=3.0)
    assert tr.events == ()
    assert tr.process([]).tempo_grid_changes == 0


def test_an_unsettled_grid_still_corrects_a_dotted_alias_in_two_evals():
    """NON-VACUITY, and the v1.3 concern this amendment has to answer.

    Widening the gate must not freeze the tracker onto an alias it picked in
    its first bars — that was the stated reason v1.3 left 4/3 on the fast
    path.  Settledness carries it: before the grid has settled, a dotted alias
    is still adopted in two evaluations, exactly as before.  If this test and
    the settled one above ever report the same number, the amendment is inert.
    """
    tr = _settled_tracker()
    tr._lock_t = tr._t                            # pretend the lock just happened
    tr._grid_t0 = tr._t
    assert not tr._grid_is_settled(tr._t)
    alias = tr._bpm * 4.0 / 3.0
    for i in range(1, 40):
        tr._adopt_bpm(alias, tr._t + 0.5 * i)
        if tr._bpm == pytest.approx(alias, rel=1e-9):
            assert i == 2
            break
    else:
        pytest.fail("an unsettled grid never corrected a dotted alias")
    assert 2 != beat_mod.OCTAVE_SWITCH_EVALS, "the two paths are indistinguishable"


def test_a_dotted_re_read_is_announced_as_a_grid_change_not_an_octave():
    """Both counters exist because they mean different things.

    ``tempo_octave_changes`` keeps its v1.3 meaning — mind.py's ritard guard
    and sim/cli's forensic lines are written against it — so a 4/3 re-read
    must NOT move it.  ``tempo_grid_changes`` is the "did the metre move at
    all" signal, and on live acoustic it is the only one that ever fires.
    """
    tr = _settled_tracker()
    alias, conf_before = tr._bpm * 4.0 / 3.0, tr._conf
    for i in range(1, beat_mod.OCTAVE_SWITCH_EVALS + 1):
        tr._adopt_bpm(alias, tr._t + 0.5 * i)
    st = tr.process([])
    assert st.tempo_grid_changes == 1
    assert st.tempo_octave_changes == 0, (
        "a 4/3 alias was counted as an octave re-interpretation")
    evs = [e for e in tr.events if e["event"] == "tempo_grid_change"]
    assert len(evs) == 1 and not [e for e in tr.events
                                  if e["event"] == "tempo_octave_change"]
    e = evs[0]
    assert e["octave"] is False
    assert e["ratio"] == pytest.approx(4.0 / 3.0, rel=1e-3)
    assert e["settled"] is True
    assert e["to_bpm"] == pytest.approx(alias, rel=1e-9)
    # re-reading a metre you have played along with for a phrase is a LOSS of
    # certainty, whatever the ratio was.
    assert e["conf_after"] == pytest.approx(
        e["conf_before"] * beat_mod._OCTAVE_CONF_DROP, rel=1e-9)
    assert e["conf_before"] == pytest.approx(min(max(conf_before, 0.0), 1.0))


def test_a_count_in_still_overrides_everything_about_the_octave():
    """The tap latch outranks the octave machinery in both directions: an
    alias cannot switch the octave out from under a count-in, and a fresh
    count-in re-arms the settle clock rather than being held off by it."""
    tr = _settled_tracker()
    for t in (tr._t + 0.5, tr._t + 1.0, tr._t + 1.5, tr._t + 2.0):
        tr.tap(t)                                   # a 120bpm count-in
    assert tr._grid_t0 == pytest.approx(tr._t)
    assert not tr._grid_is_settled(tr._t)
    for i in range(1, 40):                          # a relentless half-tempo alias
        tr._adopt_bpm(60.0, tr._t + 0.5 * i)
    assert tr._bpm == pytest.approx(120.0, abs=1e-6)
    assert tr.events == ()


def test_salience_is_measured_on_the_grid_in_force_not_the_top_candidate():
    """Confidence must never be credited with evidence for a REJECTED grid.

    ``_evaluate_tempo`` used to write the winning candidate's salience into
    confidence and only then ask hysteresis whether to adopt it, so while
    hysteresis was busy refusing a switch, confidence was already reporting
    the challenger's evidence.  Traced on a 140bpm trap take: at t=53.14 the
    tracker held 138.89 with salience 0.443; at t=53.64 the half-tempo rival
    won the sweep and was NOT adopted, and salience jumped to 0.872 anyway —
    the "confidence rises across the collapse" symptom, beginning BEFORE the
    collapse, on evidence for a grid nobody was playing.

    The invariant, pinned here as an ordering: salience is written AFTER the
    adoption decision, so it always describes ``self._bpm``.
    """
    seen = []
    orig = BeatTracker._adopt_bpm

    class Spy(BeatTracker):
        def _adopt_bpm(self, bpm_new, now):
            before = self._salience
            orig(self, bpm_new, now)
            seen.append((bpm_new, self._bpm, before, self._salience))

    tr = Spy()
    run(tr, make_frames(20.0, click_times(120.0, 40, t0=0.5)))
    assert seen, "no tempo evaluation ever reached adoption"
    # On every evaluation the salience READ AT ADOPTION TIME is the previous
    # evaluation's — the new value cannot have been written yet.
    assert all(b == a for _, _, b, a in seen), (
        "salience was written before the adoption decision, so it can report "
        "a candidate the tracker refused")


# ---------------------------------------------------------- determinism


def test_determinism_identical_state_sequences():
    rng = np.random.default_rng(7)
    grid = np.asarray(click_times(123.0, 40, t0=0.4))
    times = list(grid + rng.uniform(-0.015, 0.015, size=grid.size))
    frames = make_frames(21.0, times)

    s1 = run(BeatTracker(), frames, chunk=8)
    s2 = run(BeatTracker(), frames, chunk=8)
    assert s1 == s2                            # exact, field-for-field

    # blocking must not change the evolution: same final state either way
    s3 = run(BeatTracker(), frames, chunk=31)
    assert s3[-1] == s1[-1]


def test_taps_are_part_of_the_deterministic_input():
    frames = make_frames(6.0, click_times(120.0, 10, t0=0.5))

    def go():
        tr = BeatTracker()
        for t in (0.5, 1.0, 1.5, 2.0):
            tr.tap(t)
        return run(tr, frames)

    assert go() == go()


# ----------------------------------------------------------- audio path


def test_audio_path_click_track_through_features():
    features = pytest.importorskip("plume.listen.features")
    sr, bpm, dur = 48000, 120.0, 12.0
    t0 = 0.25
    audio = np.zeros(int(sr * dur), dtype=np.float32)
    click_len = int(0.025 * sr)
    t_click = np.arange(click_len) / sr
    click = (0.8 * np.sin(2 * np.pi * 1500.0 * t_click)
             * np.exp(-t_click / 0.006)).astype(np.float32)
    k = 0
    while True:
        start = int(round((t0 + k * 60.0 / bpm) * sr))
        if start + click_len > audio.size:
            break
        audio[start:start + click_len] += click
        k += 1

    fx = features.FeatureExtractor(sr)
    tr = BeatTracker()
    state = tr.process([])
    for i in range(0, audio.size, 4096):
        state = tr.process(fx.process(audio[i:i + 4096]))
    state = tr.process(fx.flush())

    assert state.locked
    assert state.bpm == pytest.approx(bpm, abs=3.0)
    assert nearest_grid_err(state.next_beat_t_s, bpm, t0) < 0.03


# ------------------------------------------------------- state invariants


def test_state_invariants_across_fixture():
    frames = make_frames(17.0, click_times(120.0, 32, t0=0.5))
    for st in run(BeatTracker(), frames):
        assert isinstance(st, BeatState)
        assert 0.0 <= st.confidence <= 1.0
        assert 0 <= st.beat_in_bar <= 3
        assert 0.0 <= st.phase_beats < 4.0
        if st.bpm > 0:
            assert st.beat_interval_s == pytest.approx(60.0 / st.bpm, rel=1e-9)
            assert st.next_beat_t_s > st.t_s - st.beat_interval_s
        else:
            assert math.isinf(st.beat_interval_s)
        if not st.locked and st.bar == -1:
            assert math.isinf(st.next_downbeat_t_s)


# ------------------------------------------- the acoustic front-end switch

# ``BeatTracker``'s envelope and onset source are a MODE, chosen once at
# construction (beat.py ACOUSTIC MODE).  These tests cover the selection
# machinery only; the musical A/B between the two front ends is
# tests/test_acoustic.py's job and the measured table is in beat.py.
#
# Why the selection deserves its own tests: the mode is read from the
# environment when the caller says nothing, because ``realtime``, ``sim`` and
# ``cli`` all construct the tracker themselves and none of them has a flag for
# it — ``PLUME_ACOUSTIC=1`` is the ONLY way a live take can reach the acoustic
# front end today.  An env hatch nobody tests is an env hatch that silently
# stops working.


def test_default_is_the_flux_front_end(monkeypatch):
    monkeypatch.delenv(beat_mod.ACOUSTIC_ENV, raising=False)
    assert BeatTracker()._acoustic is False


def test_acoustic_true_selects_the_attack_front_end():
    assert BeatTracker(acoustic=True)._acoustic is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_the_environment_can_select_acoustic_mode(monkeypatch, value):
    """The live paths construct ``BeatTracker()`` with no arguments, so this
    variable is the whole of the user-facing switch."""
    monkeypatch.setenv(beat_mod.ACOUSTIC_ENV, value)
    assert BeatTracker()._acoustic is True


@pytest.mark.parametrize("value", ["", "0", "off", "no", "false", "please"])
def test_a_non_affirmative_environment_value_stays_on_the_default(monkeypatch,
                                                                  value):
    monkeypatch.setenv(beat_mod.ACOUSTIC_ENV, value)
    assert BeatTracker()._acoustic is False


@pytest.mark.parametrize("env,explicit", [("1", False), ("", True),
                                          ("0", True), ("1", True)])
def test_an_explicit_flag_always_beats_the_environment(monkeypatch, env,
                                                       explicit):
    """So no test in the suite depends on the caller's shell."""
    monkeypatch.setenv(beat_mod.ACOUSTIC_ENV, env)
    assert BeatTracker(acoustic=explicit)._acoustic is explicit


def test_the_mode_is_fixed_for_the_life_of_a_tracker(monkeypatch):
    """Read once at construction, never per frame.

    A take has to stay a pure function of (frames, taps, mode); if the mode
    were re-read per frame, an environment change mid-session would switch the
    envelope underneath a locked grid.
    """
    monkeypatch.delenv(beat_mod.ACOUSTIC_ENV, raising=False)
    tr = BeatTracker()
    monkeypatch.setenv(beat_mod.ACOUSTIC_ENV, "1")
    tr.process(make_frames(2.0, click_times(120.0, 4, t0=0.5)))
    assert tr._acoustic is False


def test_both_modes_track_a_stub_stream_identically():
    """Frames with no ``attack`` field drive BOTH modes off ``flux``.

    This is what keeps the rest of this file — which builds ``FakeFrame``
    streams carrying no ``attack`` — meaningful under either mode, and it is
    the fallback that ``realtime``'s stubs and replayed frames depend on.
    """
    frames = make_frames(20.0, click_times(120.0, 40, t0=0.5))
    a = [(s.bpm, s.confidence, s.locked)
         for s in run(BeatTracker(acoustic=False), frames)]
    b = [(s.bpm, s.confidence, s.locked)
         for s in run(BeatTracker(acoustic=True), frames)]
    assert a == b


def test_a_count_in_is_not_gated_by_the_mode():
    """``tap()`` is the escape hatch for exactly the material the tracker
    cannot infer, so it must behave identically in both front ends."""
    out = []
    for acoustic in (False, True):
        tr = BeatTracker(acoustic=acoustic)
        for i in range(4):
            tr.tap(1.0 + i * 0.5, beat_in_bar=i % 4)
        st = tr.process([])
        out.append((st.bpm, st.locked, st.confidence, st.beat_in_bar,
                    st.next_downbeat_t_s))
    assert out[0] == out[1]
    assert out[0][0] == pytest.approx(120.0)
    assert out[0][3] == 3


# ------------------------------------------------- relock bias (v2.6)
#
# THE DEFECT.  Octave stickiness (_grid_hysteresis) is gated on
# ``self._locked``, so the moment a lock is lost the tracker has no memory of
# the metre it just spent a minute playing: the next evaluation is a fresh
# reading, and a busier texture on RESUME reads as the double.  Measured on a
# live take (takes5/decisions.ndjson): lock lost at 82 bpm, re-entry at 118;
# lost at 90, re-entry at 184.  The mind then comps on a grid at the wrong
# multiple of the human's tempo, which is the same downstream damage the v1.3
# mid-take octave collapse caused — only reached through the unlocked door.


def _texture(t0, n_beats, bpm, sub, rng, jitter=0.012):
    """Onsets ``sub`` to the beat, accented on the downbeat."""
    per = 60.0 / bpm
    ts, ss = [], []
    for k in range(int(n_beats * sub)):
        t = t0 + k * per / sub
        s = 1.15 if k % (4 * sub) == 0 else (1.0 if k % sub == 0 else 0.8)
        ts.append(t + rng.uniform(-jitter, jitter))
        ss.append(s)
    return ts, ss


def _stop_and_resume(*, bpm_before=80.0, bpm_after=80.0, gap_s=20.0,
                     sub_before=1, sub_after=2, beats_before=48,
                     beats_after=60, seed=7):
    """Play, stop long enough to unlock, resume — one frame stream.

    Returns ``(states, resume_t)``.  The default is the live failure: a
    quarter-note texture at 80 that locks, a stop, then an eighth-note texture
    AT THE SAME TEMPO, whose autocorrelation favours the double.
    """
    rng = np.random.default_rng(seed)
    t1, s1 = _texture(0.5, beats_before, bpm_before, sub_before, rng)
    resume_t = t1[-1] + gap_s
    t2, s2 = _texture(resume_t, beats_after, bpm_after, sub_after, rng)
    frames = make_frames(t2[-1] + 3.0, t1 + t2, strengths=s1 + s2)
    return run(BeatTracker(), frames), resume_t


def _at(states, t):
    return min(states, key=lambda s: abs(s.t_s - t))


def _unlocked_in_gap(states, gap_lo, gap_hi):
    return [s for s in states if gap_lo < s.t_s < gap_hi and not s.locked]


def test_a_short_stop_relocks_at_the_tempo_that_was_lost():
    """The take did not become a different piece of music because you rested.

    The resume texture is twice as dense as the one the tracker locked to, at
    the SAME tempo.  Read cold it is a 160bpm grid (pinned by the
    non-vacuity test below); read as the continuation of a take that stopped
    for five bars, it is 80.
    """
    states, resume_t = _stop_and_resume()

    before = _at(states, resume_t - 20.0)
    assert before.locked and before.bpm == pytest.approx(80.0, rel=0.04)
    assert _unlocked_in_gap(states, before.t_s, resume_t), \
        "vacuous: the tracker never unlocked, so nothing relocked"

    for probe in (resume_t + 10.0, resume_t + 20.0, resume_t + 30.0):
        st = _at(states, probe)
        assert st.bpm == pytest.approx(80.0, rel=0.04), \
            f"relocked at {st.bpm:.1f} instead of the 80 it lost, at t={probe:.1f}"
    assert _at(states, resume_t + 20.0).locked


def test_the_relock_bias_is_not_vacuous():
    """Cold, the resume texture really does read as the double.

    Without this the test above would pass on material that was never
    ambiguous, and the bias could be deleted with the suite still green.
    """
    rng = np.random.default_rng(7)
    ts, ss = _texture(0.5, 60, 80.0, 2, rng)
    states = run(BeatTracker(), make_frames(ts[-1] + 3.0, ts, strengths=ss))
    assert states[-1].bpm == pytest.approx(160.0, rel=0.04), \
        f"resume texture reads {states[-1].bpm:.1f} cold — fixture is not ambiguous"


def test_relock_bias_yields_to_a_genuinely_new_tempo():
    """A bias is not a clamp.  Count off something faster and it follows."""
    states, resume_t = _stop_and_resume(bpm_before=80.0, bpm_after=120.0,
                                        sub_after=1, beats_after=70)
    assert _at(states, resume_t - 20.0).bpm == pytest.approx(80.0, rel=0.04)
    st = _at(states, resume_t + 30.0)
    assert st.bpm == pytest.approx(120.0, rel=0.04), \
        f"held {st.bpm:.1f} against a real tempo change to 120"


def test_a_long_stop_is_a_fresh_reading():
    """The bias EXPIRES.  Past the window the tracker reads what it hears.

    This is the boundary, asserted so the window cannot silently grow to
    forever: after a break far longer than a phrase, the next thing the human
    plays is a new performance, and the tempo it reads is the one in the
    audio — here the double, exactly as the cold reading above.
    """
    states, resume_t = _stop_and_resume(gap_s=75.0)
    assert _unlocked_in_gap(states, resume_t - 75.0, resume_t)
    st = _at(states, resume_t + 30.0)
    assert st.bpm == pytest.approx(160.0, rel=0.04), \
        f"still biased at {st.bpm:.1f} long after the window should have expired"
