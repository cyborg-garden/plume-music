"""Real-time jam transport (SPEC-V1 section 7).

``jam()`` opens a sounddevice duplex stream (48 kHz, 256-sample blocks) whose
callback ONLY copies to/from lock-free ring buffers: no locks, no waiting, no
synthesis.  Its slices and sum do allocate a little, bounded and small.  Three worker roles surround it:

- **analysis thread** — drains the input ring, runs FeatureExtractor →
  BeatTracker / HarmonyTracker (with schedule-aware self-masking of the
  machine's own output before the beat tracker's phase correction);
- **planner thread** — fires at predicted bar boundaries, summarizes the
  completed human bar, calls ``Mind.plan_bar`` and ``DuetEngine.render_bar``
  and places the rendered bar into the output schedule (render-ahead >= 1
  bar, revisable until 1 beat before its downbeat); while the tracker is
  UNLOCKED it ticks the mind once a second and renders any sounding plan the
  tick carries (held wobble bars, the beat-loss fade) the way sim does —
  v1.8, because logging those plans without rendering them was half of the
  "status said listening while it audibly played" postmortem;
- **main thread** — status line (state/bpm/conf/key); Ctrl-C is a graceful
  ending: the current bar finishes, the tail flushes, records are written.

sounddevice is imported lazily inside :func:`jam` — importing this module (and
every headless CLI path, e.g. ``plume listen``/``plume sim``) never touches
the audio stack.  All musical logic lives in the sim-tested modules; this file
gets smoke tests only (no audio device in CI).
"""
from __future__ import annotations

import collections
import contextlib
import dataclasses
import json
import math
import os
import queue
import socket
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import fx as _fx
from .listen.features import HOP, RMS_FLOOR_DB, Frame  # noqa: F401 (Frame: typing)
# Bar-silence level tolerance relative to the human's running level; imported
# from sim so the two silence rules cannot drift apart again (v1.2: they had,
# three ways — sim, mind and here).
# _SOUNDING_STATES likewise: it is the shared vocabulary for "this plan makes
# sound", and v1.8's unlocked-tick rendering (below) must select exactly the
# plans sim's unlocked tick selects or the twins diverge on which bars exist.
from .sim import SILENCE_REL_DB, _SOUNDING_STATES

SR = 48000
BLOCK = 256

# Self-masking (v1, schedule-aware): while the machine's own output in a frame
# exceeds (human estimate - SELF_MASK_MARGIN_DB), onset triggers are
# suppressed so the tracker's phase correction never chases our own output.
# Full spectral masking is deferred.
SELF_MASK_MARGIN_DB = 6.0
# The machine bus must actually be audible for masking to engage — otherwise
# digital silence (-90 dB) would "mask" a quiet room forever.
SELF_MASK_ABS_FLOOR_DB = -60.0
# v1.6 — the machine BUS and the human estimate are not in the same domain
# (see SelfMasker for the measured consequence).  ``coupling_db`` is the
# speaker->room->mic path gain that turns one into the other; it is estimated
# live, starting from 0 dB (assume the worst: no attenuation, mask freely) and
# falling to whatever the room actually is.  It may creep back up this fast so
# a player who turns the speakers up is followed rather than ignored, and it is
# clamped so a pathological estimate cannot run away.
SELF_MASK_COUPLING_INIT_DB = 0.0
SELF_MASK_COUPLING_CREEP_DB_S = 2.0
SELF_MASK_COUPLING_MIN_DB = -60.0
SELF_MASK_COUPLING_MAX_DB = 6.0
# v1.8 — the REGRESSION estimator's knobs (see SelfMasker: "the coupling
# floor").  The leaky minimum of (mic - bus) needs the human to leave a gap;
# a guitar that never stops playing never gives it one, so on headphones it
# converged to ~-17 dB instead of the -60 floor and the masker kept blanking
# frames at the machine's own beat rate for the whole take (measured: the
# tracker read the machine's 96 bpm for 14.7% of a 48-bar zero-bleed take,
# 0.0% with the mask off).  The fix estimates the path gain from what the
# machine's level CHANGES actually do to the mic: over an exponentially
# weighted window, ``mic_pow ~= human_pow + g^2 * bus_pow``, so the regression
# slope of mic power on bus power IS g^2 — no gap required, because the
# machine's own silences between hits anchor the baseline.  Zero bleed reads
# slope ~0 (the mic does not move with the bus) and the estimate falls to the
# floor; a real room reads its true gain within a few seconds.
#: EWMA time constant of the regression window, seconds.  Long enough to
#: average over several bars of both tempi (decorrelating the human from the
#: bus), short enough that turning the speakers up is followed in a few bars.
#: Power is heavy-tailed — one human accent landing while the bus sits below
#: its mean drags the covariance hard — so the window errs long and the
#: estimate is smoothed again in dB (:data:`SELF_MASK_COUPLING_EST_TAU_S`)
#: before it is believed.
SELF_MASK_COUPLING_TAU_S = 6.0
#: dB-domain smoothing of the regression estimate.  The raw slope visits the
#: floor for a few hundred ms when an accent flips the covariance negative
#: (measured on the programmed corpus at -6 dB bleed); believed raw, those
#: excursions turned the mask off mid-take.
SELF_MASK_COUPLING_EST_TAU_S = 2.0
#: Evidence the regression needs before it may override the leaky minimum:
#: at least this much effective window...
SELF_MASK_COUPLING_MIN_WEIGHT_S = 1.0
#: ...and the bus must actually VARY (squared coefficient of variation of bus
#: power).  A constant pad carries no information about the path gain — the
#: slope of a flat line is noise — so the regression stands aside and the
#: leaky minimum keeps the estimate, exactly as before v1.8.
SELF_MASK_COUPLING_MIN_CV2 = 0.25
# v1.6 — how long the machine's own sound takes to come back round to the
# mic when the stream does not report its latency: output latency + the air +
# input latency.  Used only as a fallback; jam()/lead() ask the open stream.
SELF_MASK_DEFAULT_LATENCY_S = 0.030
# v1.6 — the TRANSIENT rule.  A level rule alone cannot protect a
# level-independent detector: since v1.5 BeatTracker picks its onsets off
# ``Frame.attack``, a banded LOG RATIO, so plume's own drums win the attack
# competition against a sustaining open-tuned guitar even 12 dB below it
# (measured).  So a frame is also masked when the machine's own bus RISES by
# SELF_MASK_RISE_DB over its quietest recent frame — that is our attack, and
# those are the only frames in which the tracker can mistake it for the
# human's.  The hold covers the smear of one transient, no more: masking is
# bounded to the machine's attacks instead of its whole decay.
SELF_MASK_RISE_DB = 6.0
SELF_MASK_RISE_WINDOW_S = 0.15
SELF_MASK_HOLD_S = 0.06
# v1.7 — the transient rule needs the SAME domain fix the level rule got in
# v1.6, and for a sharper reason than symmetry.  v1.6 moved the level rule into
# the mic domain and left the transient rule gated on the raw BUS, so it fired
# identically whether the machine was in the room or in the player's headphones
# — and a masked frame has its attack/flux blanked.  Blanking at the machine's
# own beat rate STAMPS the machine's tempo into the envelope the tracker
# autocorrelates: a zero does not say "unknown" to an autocorrelation, and a
# PERIODIC zero says "this is the period".  Measured on the sustained
# open-tuned acoustic fixtures with NO bleed at all (headphones, the
# recommended rig): the tracker locked to the machine's 96 bpm with the mask
# on and to the human's true 84.0 / 72.0 with it off.  The mask was not
# failing to stop plume following itself; it was teaching it to.
# So a transient may only mask when the machine is plausibly AUDIBLE in the
# mic — within this much of the human's own level.  The number is the measured
# 12 dB at which plume's drums still win the attack competition against a
# sustaining guitar, plus SELF_MASK_MARGIN_DB of headroom, and it sits in the
# middle of a 12-21 dB plateau over which every measured case is unchanged.
SELF_MASK_TRANSIENT_GATE_DB = 18.0
# Sanity bound on a stream-reported round trip.  A device that claims half a
# second would push the mask most of a bar out of alignment.
SELF_MASK_MAX_LATENCY_S = 0.25
# Trailing slop excluded from the bar-silence measure (parity with sim).
_BAR_EDGE_TRIM_S = 0.15

# v1.7 — how much of the per-frame event history the planner keeps when the
# beat tracker has NOT locked.  While locked the history is trimmed at every
# bar boundary; unlocked there are no bar boundaries, so without this it grows
# for as long as the session lasts — and "never locks" is not the exotic case,
# it is what six minutes of open-tuned acoustic guitar did.  The furthest back
# a bar summary can ever reach is two bars (t0 = next_downbeat - 2*bar_len), so
# 30 s covers two bars of anything down to 16 bpm.
EVENT_KEEP_S = 30.0
#: Only trim once the list is longer than this (~42 s of frames at HOP/SR), so
#: the ordinary case never pays for the scan on the planner's 20 ms tick.
EVENT_TRIM_AT = 4000


class RealtimeError(RuntimeError):
    """Clean, user-facing realtime failure (no device, missing sounddevice)."""


# ------------------------------------------------------------------ scheduling

def revision_deadline(next_downbeat_t_s: float, beat_interval_s: float) -> float:
    """A rendered bar is revisable until 1 beat before its downbeat.

    Returns ``next_downbeat_t_s - beat_interval_s`` (``inf`` stays ``inf``:
    an unpredicted downbeat has no deadline).
    """
    if math.isinf(next_downbeat_t_s):
        return math.inf
    return next_downbeat_t_s - beat_interval_s


# v1.8 (FIX 3) — live machine-bus calibration.  ``BarPlan.gain_db`` is
# RELATIVE TO THE HUMAN (SPEC-V1 section 4: "enter_under_db: first-entry gain
# relative to human's bar RMS"), and sim solves ``cal_lin`` to make that true.
# The live path tracked ``human_ref_db`` and never used it: the machine's
# actual level against the human was whatever the speakers happened to do.
# Live calibration re-solves per bar and SMOOTHES, because the live estimate
# of the human's level is itself live (a player who steps up to the mic mid
# take must be tracked, which sim's fixed once-per-arc factor cannot);
# the applied gain glides linearly across each bar so there is no step at a
# bar join and no zipper inside one.
LIVE_CAL_MIN_DB = -40.0     # sane clamp: never bury the machine entirely...
LIVE_CAL_MAX_DB = 6.0       # ...and never boost it into clipping territory
LIVE_CAL_ALPHA = 0.5        # per-bar smoothing toward the newly solved gain


def bus_calibration_step(
    cal_db: float | None,
    base: tuple[float, float] | None,
    rendered_ms: float,
    ref_db: float | None,
    gain_db: float,
    *,
    resolve: bool,
    alpha: float = LIVE_CAL_ALPHA,
) -> tuple[float | None, tuple[float, float] | None]:
    """One bar's step of the live calibration; returns ``(cal_db, base)``.

    Two regimes, and the split is the whole design:

    - **Full solve** (``resolve=True`` — an arc entry, sim's recalibration
      moments, or the first sounding bar a human reference exists for):
      ``rendered_ms`` is ``DuetEngine.last_bar_ms_pre_env`` (the bar BEFORE
      the bus envelope — calibrating on delivered samples inside a fade
      solves for the wrong number, sim's v1.4.1 lesson), and the solved
      target puts that bar's RMS at ``ref_db + gain_db``.  The solution is
      stored as ``base = (target, ref_db)``.
    - **Human tracking** (every other bar): the target is the stored base
      SHIFTED BY HOW MUCH THE HUMAN MOVED — ``base_db + (ref_db -
      base_ref)`` — and the bar's own rendered level is deliberately NOT
      consulted.  Re-solving against every bar would pin the machine's RMS
      to a fixed offset whatever it plays, which CANCELS the mind's own
      dynamics: a thinned or fading bar renders quiet on purpose, and a
      per-bar solve would boost it right back up (+10 dB measured on a
      thinned bar against a comping solve).  Sim holds its factor fixed
      between recal moments for the same reason; live adds only the term sim
      cannot need — the offline human's level never changes mid-take, the
      live one steps up to the mic.

    ``cal_db`` moves toward the target by ``alpha`` per bar (the glide in
    :func:`apply_calibration` smooths within the bar), and both regimes are
    clamped to ``LIVE_CAL_MIN_DB..LIVE_CAL_MAX_DB``.  No reference, or a
    silent render on a resolve bar, moves nothing: no evidence, no move.

    Pure, so the arithmetic is testable without a device or an engine.
    """
    if ref_db is None:
        return cal_db, base
    if resolve or base is None:
        if rendered_ms <= 0.0:
            return cal_db, base
        rendered_db = 10.0 * math.log10(float(rendered_ms))
        target = (float(ref_db) + float(gain_db)) - rendered_db
        target = min(LIVE_CAL_MAX_DB, max(LIVE_CAL_MIN_DB, target))
        base = (target, float(ref_db))
    else:
        target = base[0] + (float(ref_db) - base[1])
        target = min(LIVE_CAL_MAX_DB, max(LIVE_CAL_MIN_DB, target))
    new = target if cal_db is None else cal_db + float(alpha) * (target - cal_db)
    return min(LIVE_CAL_MAX_DB, max(LIVE_CAL_MIN_DB, new)), base


def apply_calibration(audio: np.ndarray, from_lin: float,
                      to_lin: float) -> np.ndarray:
    """Scale one bar by a gain gliding linearly ``from_lin`` -> ``to_lin``.

    The glide is what makes per-bar recalibration inaudible: bar N ends at
    exactly the gain bar N+1 starts from, so the bus level is continuous at
    every join (no zipper, no step).  Identity gains return the input as is.
    """
    if audio.size == 0 or (from_lin == 1.0 and to_lin == 1.0):
        return audio
    if from_lin == to_lin:
        return (audio.astype(np.float64) * float(to_lin)).astype(np.float32)
    env = np.linspace(float(from_lin), float(to_lin), audio.shape[0],
                      dtype=np.float64)
    if audio.ndim > 1:
        env = env[:, None]
    return (audio.astype(np.float64) * env).astype(np.float32)


def live_status_state(locked: bool, bar: int, sounding: bool) -> str:
    """The state word the live status line shows — the observability contract.

    v1.8, from the 2026-08-16 postmortem: the reported state may never
    disagree with the rendered audio.  The six-minute session said
    ``listening`` in every status sample while the human heard intermittent
    playing — bars that had been placed on the schedule while the tracker was
    briefly locked kept sounding after it unlocked, and (worse) the unlocked
    plans the mind intended to sound were never rendered at all, so neither
    the ear nor the readout matched the decision log.  Now that the unlocked
    path renders (see ``jam``'s planner), the readout has three honest words:

    - ``bar N`` — locked, playing on the human's grid;
    - ``playing (no grid)`` — not locked, but machine audio IS sounding
      (a held wobble bar, a beat-loss fade, or the tail of a placed bar);
    - ``listening`` — not locked and nothing of ours is sounding.

    Pure, so the mapping is testable without a device.
    """
    if locked:
        return f"bar {bar}"
    return "playing (no grid)" if sounding else "listening"


def trim_events(events: list, now_t_s: float, *,
                keep_s: float = EVENT_KEEP_S) -> list:
    """Drop per-frame events older than ``keep_s`` before ``now_t_s``.

    The planner's ``shared["events"]`` is normally trimmed at a bar boundary,
    which only exists once the tracker has LOCKED.  A session that never locks
    never reaches that line, and "never locks" is exactly what six minutes of
    open-tuned acoustic guitar did — so the list grew for the whole take.  A
    bar summary reaches back at most two bars, so anything older than
    ``keep_s`` is unreachable by construction and dropping it cannot change a
    musical decision.

    Pure, so the bound can be tested without a device, a thread or a clock.
    """
    cut = float(now_t_s) - float(keep_s)
    return [e for e in events if e[0] >= cut]


#: How long after a half bar ends its frames are trusted to have all arrived: the input
#: latency and an analysis hop, with room to spare.
FOLLOW_READ_MARGIN_S = 0.1
#: Planning may cost this many times its measured median before a bar is late.
PLAN_BUDGET_X = 4.0


def follow_parts(bar_events, later_events, w_start: float, w_end: float,
                 next_start: float, *, fed_until: float | None, heard_until: float,
                 silence_rms_db: float, ref_db: float | None):
    """Her level over each half bar not yet handed to BandFollow, oldest first.

    ``bar_events`` are the frames of the bar just finished, ``[w_start, w_end)``, and
    ``later_events`` those since, in the bar now playing, ``[w_end, next_start)``.  The
    first half of the bar now playing is read too once ``heard_until`` (the latest
    frame) is past its end: that is the half bar sooner the band hears her.  A half is
    read only once, whichever call reaches it.  None for a half she sat out, by
    :func:`bar_features_dict`'s own silence rule.  Returns ``(levels, fed_until)``.
    """
    mid, mid2 = 0.5 * (w_start + w_end), 0.5 * (w_end + next_start)
    spans = [(w_start, mid, bar_events), (mid, w_end, bar_events)]
    if heard_until >= mid2:
        spans.append((w_end, mid2, later_events))
    eps = 1e-6
    levels = []
    for t0, t1, evs in spans:
        if fed_until is not None and t0 < fed_until - eps:
            continue
        inside = [e for e in evs if t0 <= e[0] < t1]
        kw = bar_features_dict(0, inside, silence_rms_db=silence_rms_db, ref_db=ref_db,
                               beats_per_bar=2)
        levels.append(None if kw["silent"] else kw["rms_db"])
        fed_until = t1
    return levels, fed_until


def plan_lead_s(bar_s: float, *, plan_cost_s: float | None, pace: float | None,
                comp_s: float) -> float:
    """How far ahead of the playhead a lead take plans its next bar, in stream seconds.

    Half a bar, less :data:`FOLLOW_READ_MARGIN_S`, so the first half of the bar now
    playing has been heard when the next is planned and the band answers her a bar
    sooner.  Never less than planning needs: :data:`PLAN_BUDGET_X` times its measured
    cost, at the pace the stream runs (a fake device runs faster than real time), plus
    the bar's early placement.  Never more than the whole bar it always had, which is
    also the answer before anything has been measured.
    """
    bar_s = float(bar_s)
    if plan_cost_s is None or pace is None:
        return bar_s
    need = PLAN_BUDGET_X * float(plan_cost_s) * float(pace) + float(comp_s)
    return min(bar_s, max(0.5 * bar_s - FOLLOW_READ_MARGIN_S, need))


def _median_s(costs_us) -> float | None:
    """Median of the last planning costs, in seconds; None before there are any."""
    recent = sorted(costs_us[-32:])
    return recent[len(recent) // 2] * 1e-6 if recent else None


class _StreamPace:
    """Stream seconds per wall second, from the playhead: 1 on a real device.

    The fastest it has been seen running, so a stream that speeds up (the fake device
    starts at real time, then runs twelve times faster) is never taken for a slow one.
    Nothing is said until ``settle_s`` of wall time has been watched."""

    def __init__(self, *, window_s: float = 0.5, settle_s: float = 3.0, clock=None):
        self._window, self._settle = float(window_s), float(settle_s)
        self._clock = clock or time.monotonic
        self._first: float | None = None
        self._mark: tuple[float, float] | None = None
        self._fastest = 0.0

    def note(self, playhead_s: float) -> None:
        now = self._clock()
        if self._mark is None:
            self._first = now
            self._mark = (now, float(playhead_s))
            return
        dt = now - self._mark[0]
        if dt >= self._window:
            self._fastest = max(self._fastest, (float(playhead_s) - self._mark[1]) / dt)
            self._mark = (now, float(playhead_s))

    def rate(self) -> float | None:
        if self._first is None or self._clock() - self._first < self._settle:
            return None
        return self._fastest or None


def bar_features_dict(
    bar: int,
    events: list[tuple[float, float, bool]],
    *,
    silence_rms_db: float,
    ref_db: float | None = None,
    register_low_midi: float | None = None,
    register_high_midi: float | None = None,
    beats_per_bar: int = 4,
) -> dict:
    """Summarize one completed human bar as ``mind.BarFeatures`` kwargs.

    ``events`` is the list of ``(t_s, rms_db, onset)`` per analysis frame that
    fell inside the bar.  RMS combines in the power domain (mean of squared
    linear amplitudes back to dB); a bar with no frames is silent at -90 dB.
    """
    if events:
        amps = np.array([10.0 ** (e[1] / 20.0) for e in events], dtype=np.float64)
        power = float(np.mean(amps * amps))
        rms_db = 10.0 * math.log10(power) if power > 0.0 else RMS_FLOOR_DB
        rms_db = max(rms_db, RMS_FLOOR_DB)
        onsets = sum(1 for e in events if e[2])
        # Silence measure over the bar minus its trailing edge (parity with
        # sim._BarAccum.is_silent — the two MUST stay identical; `jam` and
        # `sim` are supposed to be one code path and this was the third
        # independent copy of the same broken absolute-RMS test).
        # v1.2 rule: detected onsets in the edge-trimmed core mean the human
        # is PLAYING, whatever the level (a sparse percussive bar's mean RMS
        # runs 13-25 dB under its peaks and read "silent" on every real
        # input); otherwise the bar is silent below the effective threshold,
        # which is the absolute floor raised to ref_db - SILENCE_REL_DB once
        # the human's running level is known.
        t_edge = events[-1][0] - _BAR_EDGE_TRIM_S
        core = [e for e in events if e[0] <= t_edge] or events
        core_onsets = sum(1 for e in core if e[2])
        thr = silence_rms_db
        if ref_db is not None:
            thr = max(thr, float(ref_db) - SILENCE_REL_DB)
        quiet_frac = sum(1 for e in core if e[1] < thr) / len(core)
        silent = core_onsets == 0 and (rms_db < thr or quiet_frac >= 0.8)
    else:
        rms_db = RMS_FLOOR_DB
        onsets = 0
        silent = True
    return {
        "bar": int(bar),
        "rms_db": float(rms_db),
        "onset_density": onsets / float(beats_per_bar),
        "silent": silent,
        "register_low_midi": register_low_midi,
        "register_high_midi": register_high_midi,
    }


# ------------------------------------------------------------------ ring buffer

class RingBuffer:
    """Lock-free single-producer/single-consumer ring buffer (float32).

    Safe under the GIL for one writer thread + one reader thread: the writer
    publishes by advancing ``_w`` after the data copy, the reader by advancing
    ``_r`` after its copy; each index has exactly one mutating thread.
    ``write`` never blocks — on overflow the excess is dropped and counted.
    """

    def __init__(self, capacity: int, channels: int = 1):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self._cap = int(capacity)
        self._ch = int(channels)
        shape = (self._cap,) if channels == 1 else (self._cap, channels)
        self._buf = np.zeros(shape, dtype=np.float32)
        self._w = 0          # total samples ever written (writer-owned)
        self._r = 0          # total samples ever read (reader-owned)
        self.dropped = 0     # samples dropped on overflow (writer-owned)

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def available(self) -> int:
        """Samples ready to read."""
        return self._w - self._r

    @property
    def space(self) -> int:
        """Samples writable without dropping."""
        return self._cap - (self._w - self._r)

    def write(self, x: np.ndarray) -> int:
        """Copy ``x`` in; returns samples accepted (excess dropped, counted)."""
        x = np.asarray(x, dtype=np.float32)
        if self._ch == 1 and x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        want = x.shape[0]
        n = min(want, self.space)
        if n > 0:
            i = self._w % self._cap
            first = min(n, self._cap - i)
            self._buf[i:i + first] = x[:first]
            if n > first:
                self._buf[: n - first] = x[first:n]
            self._w += n
        self.dropped += want - n
        return n

    def read(self, max_n: int | None = None) -> np.ndarray:
        """Return (a copy of) up to ``max_n`` available samples."""
        n = self.available
        if max_n is not None:
            n = min(n, max_n)
        shape = (n,) if self._ch == 1 else (n, self._ch)
        out = np.empty(shape, dtype=np.float32)
        if n > 0:
            i = self._r % self._cap
            first = min(n, self._cap - i)
            out[:first] = self._buf[i:i + first]
            if n > first:
                out[first:] = self._buf[: n - first]
            self._r += n
        return out


class TimelineBuffer:
    """Output schedule: a circular stereo buffer addressed by ABSOLUTE sample
    position.  The audio callback reads sequentially at the playhead (consumed
    samples are zeroed, so the buffer wraps clean); the planner writes rendered
    bars at future positions and may clear/rewrite a region to revise it.
    Writes clamp to ``[playhead, playhead + capacity)`` — the past is immutable.
    """

    def __init__(self, capacity: int, channels: int = 2):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._cap = int(capacity)
        self._ch = int(channels)
        self._buf = np.zeros((self._cap, self._ch), dtype=np.float32)
        self._play = 0       # absolute index of the next sample to read
        self._end = 0        # absolute end of scheduled material

    @property
    def playhead(self) -> int:
        return self._play

    @property
    def end(self) -> int:
        """Absolute end of everything scheduled so far."""
        return self._end

    def read_into(self, out: np.ndarray) -> None:
        """Fill ``out`` (n, channels) with the next n samples; advance and
        zero the consumed region."""
        n = out.shape[0]
        i = self._play % self._cap
        first = min(n, self._cap - i)
        out[:first] = self._buf[i:i + first]
        self._buf[i:i + first] = 0.0
        rest = n - first
        if rest > 0:
            out[first:] = self._buf[:rest]
            self._buf[:rest] = 0.0
        self._play += n

    def write_at(self, at: int, audio: np.ndarray, *, add: bool = True) -> int:
        """Mix (default) or overwrite ``audio`` at absolute position ``at``.

        Returns samples actually written after clamping to the writable
        window ``[playhead, playhead + capacity)``.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.repeat(audio[:, None], self._ch, axis=1)
        n = audio.shape[0]
        start = max(int(at), self._play)
        end = min(int(at) + n, self._play + self._cap)
        m = end - start
        if m <= 0:
            return 0
        seg = audio[start - int(at): start - int(at) + m]
        i = start % self._cap
        first = min(m, self._cap - i)
        if add:
            self._buf[i:i + first] += seg[:first]
            if m > first:
                self._buf[: m - first] += seg[first:]
        else:
            self._buf[i:i + first] = seg[:first]
            if m > first:
                self._buf[: m - first] = seg[first:]
        self._end = max(self._end, end)
        return m

    def clear_region(self, at: int, n: int) -> None:
        """Zero ``[at, at + n)`` (clamped to the writable window) — used to
        retract a previously placed bar before re-placing a revision."""
        start = max(int(at), self._play)
        end = min(int(at) + n, self._play + self._cap)
        m = end - start
        if m <= 0:
            return
        i = start % self._cap
        first = min(m, self._cap - i)
        self._buf[i:i + first] = 0.0
        if m > first:
            self._buf[: m - first] = 0.0


class HopRms:
    """Per-HOP RMS (dBFS) of a mono stream, emitted in lockstep with
    :class:`~plume.listen.features.FeatureExtractor` frames (both emit one
    value per completed HOP of input)."""

    def __init__(self, hop: int = HOP):
        self._hop = int(hop)
        self._buf = np.zeros(0, dtype=np.float32)

    def process(self, block: np.ndarray) -> list[float]:
        x = np.asarray(block, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if x.size:
            self._buf = np.concatenate([self._buf, x])
        n = self._buf.size // self._hop
        if n == 0:
            return []
        mat = self._buf[: n * self._hop].astype(np.float64).reshape(n, self._hop)
        self._buf = self._buf[n * self._hop:]
        power = np.mean(mat * mat, axis=1)
        floor = 10.0 ** (RMS_FLOOR_DB / 10.0)
        db = 10.0 * np.log10(np.maximum(power, floor))
        return [float(v) for v in db]


class SelfMasker:
    """Schedule-aware self-masking v1.6 (SPEC-V1 section 7).

    We know exactly what the machine played.  While the machine's own sound in
    a frame exceeds ``human_estimate - margin_db`` (and it is actually
    audible), the frame is flagged ``self_masked``, its onset trigger is
    cleared, and the flag is handed to the beat tracker, so phase correction
    never chases the machine's own output.  The human level estimate is an EMA
    over unmasked frames only.

    **v1.6 — the three reasons this was inoperative in the room.**  Juniper
    played an open-tuned acoustic into the laptop mic while plume came out of
    the laptop speakers, and the tempo estimate walked 167.9 / 163.0 / 169.2 /
    165.4 / 80.6 / 80.3 / 121.0 bpm.  That is the exact condition this class
    exists for, and all three of its moving parts were wrong:

    1. **The flag never reached the tracker.**  ``mask()`` cleared
       ``Frame.onset`` and returned ``flags``, and both call sites threw the
       flags away.  Since v1.5 :class:`~plume.listen.beat.BeatTracker` picks
       its OWN onsets off ``Frame.attack`` and only consults ``onset`` for
       duck-typed stubs, so clearing ``onset`` suppressed nothing at all on a
       real stream.  ``features.Frame.self_masked`` had been added *for this*
       and nobody ever set it.  Net effect: every phase correction the machine
       triggered in its own bleed was taken.  Now ``mask()`` sets
       ``self_masked=True`` on the frame it returns AND :func:`analysis_step`
       passes ``masked=flags`` — either alone would do; both is deliberate,
       because the previous single channel is what silently rotted.
    2. **The comparison was across two different domains.**  ``machine_rms_db``
       is the *bus*: what we sent to the output.  ``human_rms_db`` is the
       *mic*.  Between them sits a speaker, a room and a microphone, which on
       a laptop is typically 15-35 dB of attenuation.  Comparing them directly
       makes the whole feature a function of gain staging rather than of what
       is audible: with the bus hotter than the mic, EVERY frame the machine
       plays in is masked (which is also how the human's own bar features got
       hollowed out — ``shared["events"]`` reads the masked frames), and with
       it colder, none ever are.  So the bus level is now translated into the
       mic domain by ``coupling_db``, estimated live (below).
    3. **The mask was aligned to when the sound was SENT, not when it was
       HEARD.**  ``machine_ring`` is written in the audio callback, but that
       block does not leave the speakers until the output latency has elapsed
       and does not come back through the mic until the input latency has too.
       ``latency_hops`` delays the bus stream by that round trip before
       comparing.  The air (about 1 ms across a laptop) is inside the rounding.

    **The coupling estimate.**  ``mic = human + coupled machine``, so over
    frames where the machine is audible, ``f.rms_db - machine_bus_db`` is an
    upper bound on the path gain, and equals it whenever the human happens to
    be quiet.  A leaky minimum of that observation therefore converges DOWN to
    the true coupling within a second or two of the human leaving any gap, and
    creeps back up at :data:`SELF_MASK_COUPLING_CREEP_DB_S` so that turning the
    speakers up is followed.  It starts at 0 dB — assume no attenuation and
    mask freely — because the failure that costs a take is the tracker locking
    to itself, not a few over-masked frames at the top of a tune.

    **v1.8 — the coupling FLOOR, or: silence in the mic must not be required
    for the estimate to be honest.**  The leaky minimum's only real
    observations are the gaps, and a sustained open-tuned guitar never leaves
    one: measured on the zero-bleed strum fixture, the estimate sat at -5 to
    -11 dB for the entire take — indistinguishable from a real -6 dB room —
    while the truth was "headphones, no coupling at all".  With the estimate
    that high, ``m_mic`` lands at the human's own level, the transient gate is
    satisfied for most machine hits, ~11% of frames are blanked AT THE
    MACHINE'S BEAT RATE, and the autocorrelation reads the periodic holes as a
    period: the share of tempo samples reading the machine's bpm GREW with
    take length (2.9% at 20 bars, 14.7% at 48) and was 0.0% with the mask off.

    So the coupling is now estimated from what the bus's changes DO to the
    mic, which needs no gap: over an exponentially weighted window,
    ``mic_pow ~= human_pow + g^2 * bus_pow``, and since the human's playing is
    uncorrelated with the machine's hit pattern, the regression slope of mic
    power on bus power is ``g^2`` itself.  The machine's own inter-hit
    silences supply the baseline, so a human who never stops playing is no
    longer a problem — their level is the intercept, not a bias.  Zero bleed:
    the mic does not move with the bus, the slope reads ~0, and the estimate
    falls to the floor, which makes the mask converge to NEVER masking (the
    invariant the fix is for).  The regression only speaks when it has
    evidence — a full window and a bus that actually varies
    (:data:`SELF_MASK_COUPLING_MIN_CV2`) — and it can only LOWER the running
    estimate (the leaky minimum stays as the upper bound and as the whole
    story for a constant-level bus, where a slope means nothing); "no
    evidence" therefore keeps the pre-v1.8 worst-case-first behaviour rather
    than turning the mask off.

    **v1.7 — the fourth reason, which was the v1.6 fix itself.**  Once masking
    reached the tracker it started doing harm of its own, and the harm was
    invisible on the corpus v1.6 was measured against.  A masked frame has its
    ``attack``/``flux`` blanked, and the transient rule fired on any rise of
    the raw BUS — a fact about what we SENT, not about what is in the mic — so
    it fired identically on headphones, where there is nothing to mask.  The
    blanks therefore landed at the machine's own beat rate, and a periodic hole
    is not an absence of evidence to an autocorrelation: it is a period.  The
    mask was stamping the machine's tempo onto the human's envelope.

    Measured on the sustained open-tuned acoustic fixtures (the material from
    the room), machine drums at 96 bpm, tracker in acoustic mode::

        fingerpick, true 84 bpm     headphones (NO bleed)   speakers -6 dB
          mask off                       84.1  ok                96.2  BAD
          v1.6 (ungated transients)      96.2  BAD               96.2  BAD
          v1.7 (gated)                   84.1  ok                84.1  ok

        strum, true 72 bpm          headphones (NO bleed)   speakers -6 dB
          mask off                       72.0  ok                96.2  BAD
          v1.6 (ungated transients)      96.2  BAD               96.2  BAD
          v1.7 (gated)                   72.0  ok                96.2  BAD

    So v1.6 turned "the tracker can be fooled by its own output when it is in
    the room" into "the tracker is fooled by its own output even in
    headphones".  The gate (:data:`SELF_MASK_TRANSIENT_GATE_DB`) requires the
    machine's transient to be within 18 dB of the human's level IN THE MIC
    before it may mask, and on the programmed corpus it also *raises*
    confidence (0.72 -> 0.89 on headphones, 0.71 -> 0.82 on speakers) because
    it stops blanking frames that never needed blanking.

    **What this design cannot do, stated rather than hidden.**  It is not echo
    cancellation.  It works on level in whole frames, so a human note that
    lands underneath a machine note inside the same 10.7 ms frame is lost to
    the tracker along with the machine's; it cannot separate them, only decline
    to trust the frame.  A human who never stops playing no longer inflates
    the coupling estimate (v1.8's regression needs no gap), but a human who
    plays IN TIME WITH the machine defeats the regression's one assumption —
    correlated sources make the slope read high — so a same-tempo duet
    over-masks toward the pre-v1.6 behaviour rather than toward silence.  A
    single delay is a poor model of a real room's reflections.

    And the bottom-right cell of the table above is not a bug to be found
    later — it is the limit.  **Sustained, strummed, open-tuned acoustic
    guitar through laptop speakers into a laptop mic is not recoverable by
    masking at any setting**: the machine's drum transients genuinely are
    louder in that microphone than the guitar's accents, so declining to trust
    those frames leaves the tracker with too little to hold a grid, and
    trusting them locks it to the drums.  Every gate value from 6 dB to 30 dB
    was measured; none recovers it.  What recovers it is not hearing the
    speakers: headphones, which v1.7 makes genuinely free.  For speakers, the
    count-in hatch (a human counting the band in) is the designed answer —
    see the 2026-08-16 postmortem, finding 2 and point 3.
    """

    def __init__(self, margin_db: float = SELF_MASK_MARGIN_DB, *,
                 human_alpha: float = 0.95,
                 abs_floor_db: float = SELF_MASK_ABS_FLOOR_DB,
                 latency_hops: int = 0,
                 frame_dt: float = HOP / SR,
                 coupling_db: float = SELF_MASK_COUPLING_INIT_DB,
                 adapt_coupling: bool = True,
                 transients: bool = True,
                 rise_db: float = SELF_MASK_RISE_DB,
                 rise_window_s: float = SELF_MASK_RISE_WINDOW_S,
                 hold_s: float = SELF_MASK_HOLD_S,
                 transient_gate_db: float = SELF_MASK_TRANSIENT_GATE_DB,
                 blank_envelope: bool = True, enabled: bool = True):
        #: False on a direct input.  Nothing of ours can reach it, and a mask with
        #: nothing to mask only punches holes at the machine's beat rate (v1.7).
        self._enabled = bool(enabled)
        self._margin = float(margin_db)
        self._alpha = float(human_alpha)
        self._floor = float(abs_floor_db)
        self._adapt = bool(adapt_coupling)
        self._dt = float(frame_dt)
        self._creep = (SELF_MASK_COUPLING_CREEP_DB_S * self._dt)
        self._transients = bool(transients)
        self._rise = float(rise_db)
        self._gate = float(transient_gate_db)
        self._blank = bool(blank_envelope)
        self._hold_n = max(1, int(round(float(hold_s) / max(1e-9, self._dt))))
        self.human_rms_db = RMS_FLOOR_DB
        # v1.8 — the regression estimator's EWMA sums (see the class
        # docstring, "the coupling FLOOR").  One decay per frame, so the
        # estimate is chunking-invariant like everything else in here.
        self._reg_decay = math.exp(-self._dt / SELF_MASK_COUPLING_TAU_S)
        self._reg_alpha = 1.0 - math.exp(-self._dt / SELF_MASK_COUPLING_EST_TAU_S)
        self._reg_min_w = SELF_MASK_COUPLING_MIN_WEIGHT_S / self._dt
        self._reg_w = 0.0                 # effective sample count
        self._reg_b = 0.0                 # sum of bus power
        self._reg_m = 0.0                 # sum of mic power
        self._reg_bb = 0.0                # sum of bus power squared
        self._reg_bm = 0.0                # sum of bus*mic power
        self.coupling_db = float(coupling_db)   # (property: resets both halves)
        self.masked_frames = 0
        self.frames_seen = 0
        self.machine_transients = 0
        self._hold = 0
        self._bus_hist: collections.deque[float] = collections.deque(
            maxlen=max(1, int(round(float(rise_window_s)
                                    / max(1e-9, self._dt)))))
        # The bus stream, delayed by the round trip.  Pre-filled with silence
        # so the very first frames are compared against nothing rather than
        # against sound that has not been emitted yet.
        self._delay: collections.deque[float] = collections.deque(
            [RMS_FLOOR_DB] * max(0, int(latency_hops)))
        self.latency_hops = max(0, int(latency_hops))

    def set_latency_hops(self, hops: int) -> "SelfMasker":
        """Set the bus->mic round trip once the stream is open and can be
        asked.  Only valid before any frame has been masked (the transports
        call it between opening the stream and starting the analysis thread);
        afterwards it would tear a hole in the delayed stream."""
        if self.frames_seen:                            # pragma: no cover
            raise RealtimeError("latency must be set before masking starts")
        hops = max(0, int(hops))
        self.latency_hops = hops
        self._delay = collections.deque([RMS_FLOOR_DB] * hops)
        return self

    @property
    def coupling_db(self) -> float:
        """The path-gain estimate the mask judges against: the tighter of the
        leaky minimum (the v1.6 upper bound) and the smoothed regression (the
        v1.8 evidence).  Whichever knows the room is quieter wins."""
        if self._reg_smooth_db is None:
            return self._leaky_db
        return min(self._leaky_db, self._reg_smooth_db)

    @property
    def coupling_measured(self) -> bool:
        """Has the room been measured at all?  The estimate starts at the worst
        case on purpose, which is right for a mask and wrong for a report."""
        return self._reg_smooth_db is not None

    @coupling_db.setter
    def coupling_db(self, value: float) -> None:
        # Setting the estimate (tests, and __init__'s starting point) resets
        # BOTH halves: a pinned estimate must actually pin.
        self._leaky_db = float(value)
        self._reg_smooth_db: float | None = None

    def _regression_coupling_db(self) -> float | None:
        """The v1.8 estimate: slope of mic power on bus power, in dB.

        ``None`` means "no evidence" — a short window, a bus that does not
        vary (a constant pad's slope is noise, not a measurement), or a bus
        whose mean power sits at the digital-silence floor.  No evidence keeps
        the leaky minimum's worst-case-first estimate rather than clearing it:
        the regression may only ever say "the room is quieter than you
        assumed", never "assume the best".
        """
        w = self._reg_w
        if w < self._reg_min_w:
            return None
        mean_b = self._reg_b / w
        if mean_b <= 10.0 ** (SELF_MASK_COUPLING_MIN_DB / 10.0):
            return None
        var_b = self._reg_bb / w - mean_b * mean_b
        if var_b <= 0.0 or var_b < SELF_MASK_COUPLING_MIN_CV2 * mean_b * mean_b:
            return None
        cov = self._reg_bm / w - mean_b * (self._reg_m / w)
        slope = cov / var_b                      # = g^2, the power-domain gain
        if slope <= 10.0 ** (SELF_MASK_COUPLING_MIN_DB / 10.0):
            return SELF_MASK_COUPLING_MIN_DB
        return 10.0 * math.log10(slope)

    @property
    def masked_fraction(self) -> float:
        """Share of frames masked so far.  Near 1.0 means the masker has
        stopped being a mask and started being a gate — the honest signal that
        the coupling estimate never found a gap (see the class docstring)."""
        return (self.masked_frames / self.frames_seen) if self.frames_seen else 0.0

    def mask(self, frames: list, machine_rms_db: list[float]
             ) -> tuple[list, list[bool]]:
        """Return ``(frames_for_tracker, self_masked_flags)``.

        ``machine_rms_db`` must be in per-frame lockstep with ``frames``
        (``HopRms`` and ``FeatureExtractor`` emit one value each per completed
        HOP, which is what keeps it so).  Values beyond the frames given are
        *kept*, not dropped: they belong to frames that have not been extracted
        yet, and dropping them was a second, silent way to lose alignment.

        Chunking-invariant: all state carries between calls, so feeding the
        same stream in one block or in a hundred gives identical flags.
        """
        if not self._enabled:
            self.frames_seen += len(frames)
            return list(frames), [False] * len(frames)
        self._delay.extend(float(v) for v in machine_rms_db)
        out = []
        flags = []
        for f in frames:
            m_bus = self._delay.popleft() if self._delay else RMS_FLOOR_DB
            audible = m_bus > self._floor
            # (a) the LEVEL rule (SPEC-V1 section 7), now in the mic domain.
            # Judged against the coupling estimate as it stood BEFORE this
            # frame, so a frame cannot move the bar it has to clear.
            m_mic = m_bus + self.coupling_db
            loud = audible and m_mic > (self.human_rms_db - self._margin)
            # (b) the TRANSIENT rule.  A rise in the machine's own bus, held
            # for hold_s, i.e. exactly the frames our own attack lands in —
            # but only while the machine is plausibly audible IN THE MIC.
            # v1.7: that gate is the whole difference between a mask and a
            # metronome.  Ungated, the rule fired on headphones too, and a
            # masked frame is a blanked frame, so the mask punched holes in
            # the human's envelope at the MACHINE's beat rate and the
            # autocorrelation read the holes as the machine's tempo.  See
            # SELF_MASK_TRANSIENT_GATE_DB for the measurement.
            competitive = audible and m_mic > (self.human_rms_db - self._gate)
            if self._transients:
                quiet = min(self._bus_hist) if self._bus_hist else RMS_FLOOR_DB
                if competitive and (m_bus - quiet) >= self._rise:
                    self._hold = self._hold_n
                    self.machine_transients += 1
                self._bus_hist.append(m_bus)
            hit = self._hold > 0
            if hit:
                self._hold -= 1
            masked = loud or (competitive and hit)
            if self._adapt:
                if audible:
                    # The leaky minimum: an upper bound that needs the human
                    # to leave a gap, kept because it is the only estimator a
                    # constant-level bus admits (see v1.8 in the docstring).
                    obs = float(f.rms_db) - m_bus
                    self._leaky_db = min(obs, self._leaky_db + self._creep)
                    self._leaky_db = min(SELF_MASK_COUPLING_MAX_DB,
                                         max(SELF_MASK_COUPLING_MIN_DB,
                                             self._leaky_db))
                # v1.8 — the regression.  EVERY frame is a data point: the
                # bus-quiet frames are the baseline the slope is measured
                # against, which is exactly why this estimator needs no gap
                # from the HUMAN.  Power domain, one decay per frame.
                d = self._reg_decay
                b_pow = 10.0 ** (m_bus / 10.0)
                mic_pow = 10.0 ** (float(f.rms_db) / 10.0)
                self._reg_w = d * self._reg_w + 1.0
                self._reg_b = d * self._reg_b + b_pow
                self._reg_m = d * self._reg_m + mic_pow
                self._reg_bb = d * self._reg_bb + b_pow * b_pow
                self._reg_bm = d * self._reg_bm + b_pow * mic_pow
                est = self._regression_coupling_db()
                if est is not None:
                    # Smooth in dB before believing: the raw slope dips to the
                    # floor for a few hundred ms when a human accent flips the
                    # covariance negative, and an estimate that tracks those
                    # dips is a mask with a random off-switch.
                    s = self._reg_smooth_db
                    self._reg_smooth_db = (est if s is None
                                           else s + self._reg_alpha * (est - s))
                elif self._reg_smooth_db is not None:
                    # No fresh evidence: the old measurement expires UPWARD at
                    # the creep rate, so a stale "the room was quiet" cannot
                    # pin the mask off against a room that has changed.
                    self._reg_smooth_db += self._creep
                if self._reg_smooth_db is not None:
                    self._reg_smooth_db = min(
                        SELF_MASK_COUPLING_MAX_DB,
                        max(SELF_MASK_COUPLING_MIN_DB, self._reg_smooth_db))
            flags.append(masked)
            self.frames_seen += 1
            if masked:
                self.masked_frames += 1
                # ``self_masked`` is what the tracker actually reads; ``onset``
                # is cleared as well because ``shared["events"]`` (the human's
                # bar summary) still counts it.  See defect (1) above.
                # ``attack``/``flux`` are zeroed because of defect (4): the
                # tracker's ONSET intake is gated by the flag, but its tempo
                # ESTIMATE is not — ``_step`` appends every frame's envelope to
                # the autocorrelation window before it ever looks at the mask.
                # Suppressing phase while leaving period wide open is why
                # masking made no measurable difference in the closed-loop
                # repro (measured: locked bpm identical, masked or not).  A
                # masked frame means "we cannot tell what the human did here",
                # and for an autocorrelation a hole says that far better than
                # the machine's own transient does.
                #
                # v1.7 WARNING to whoever reads this next: this blanking is
                # also the sharpest edge in the class.  A zero here is only
                # "we cannot tell" as long as the zeros are NOT periodic;
                # blank on a regular grid and you have handed the
                # autocorrelation a tempo.  That is why the transient gate
                # above exists, and why loosening it is not a free knob.
                repl = {"onset": False, "onset_strength": 0.0,
                        "self_masked": True}
                if self._blank and hasattr(f, "attack"):
                    repl["attack"] = 0.0
                    repl["flux"] = 0.0
                f = dataclasses.replace(f, **repl)
            else:
                self.human_rms_db = (self._alpha * self.human_rms_db
                                     + (1.0 - self._alpha) * f.rms_db)
            out.append(f)
        return out, flags


def stream_latency_hops(stream, *, default_s: float = SELF_MASK_DEFAULT_LATENCY_S,
                        hop: int = HOP, sr: int = SR) -> int:
    """Round-trip bus->speaker->mic->buffer, in analysis hops.

    ``sounddevice.Stream.latency`` is ``(input, output)`` in seconds; their sum
    is how long after we hand PortAudio a block that block can first appear in
    the mic.  Anything the stream will not tell us (a stand-in stream in a
    test, an older binding) falls back to ``default_s``, and an implausible
    figure is clamped — a device claiming half a second would push the mask
    most of a bar out of alignment, which is worse than not masking.
    """
    lat = getattr(stream, "latency", None)
    total: float | None = None
    try:
        if isinstance(lat, (tuple, list)) and lat:
            total = sum(float(v) for v in lat)
        elif isinstance(lat, (int, float)):
            total = float(lat)
    except (TypeError, ValueError):                    # pragma: no cover
        total = None
    if total is None or not math.isfinite(total) or total < 0.0:
        total = float(default_s)
    total = min(float(SELF_MASK_MAX_LATENCY_S), total)
    return int(round(total * sr / hop))


def analysis_step(*, extractor, hop_rms, masker, tracker, harmony,
                  human_block, machine_block):
    """One turn of the analysis loop, as a function both transports call.

    It exists so the closed-loop self-masking test can drive *the code that
    runs live* rather than a re-implementation of it: the defect this round
    fixed was not in :class:`SelfMasker` at all, it was in the two copies of
    the four lines that used it, and a test of the class alone could not have
    seen it.  There is now one copy and the test is on it.

    Returns ``(frames_for_tracker, flags, beat_state, harmony_state)``.  The
    harmony tracker deliberately gets the UNMASKED frames: masking is about
    *when* things were struck, and blanking chroma would make the machine's own
    chords disappear from a key estimate that wants them.
    """
    frames = extractor.process(human_block)
    # Only a MIDI source has a clock to state; a mic's tempo is always inferred.
    apply_clock = getattr(extractor, "apply_clock", None)
    if apply_clock is not None:
        apply_clock(tracker)
    machine_db = hop_rms.process(machine_block)
    masked_frames, flags = masker.mask(frames, machine_db)
    beat_state = tracker.process(masked_frames, masked=flags)
    harm_state = harmony.process(frames)
    return masked_frames, flags, beat_state, harm_state


# ------------------------------------------------------------------ device I/O

def _resolve_midi_port(midi: str | None) -> str | None:
    """Which port ``--midi`` means, refused before anything opens or sounds."""
    if midi is None:
        return None
    from .midi import MidiError, resolve_port  # noqa: PLC0415 - optional, like sd
    try:
        return resolve_port(midi)
    except MidiError as e:
        raise RealtimeError(str(e)) from None


def _midi_listening(port: str, timeline: "TimelineBuffer",
                    recorder: "SessionRecorder | None", *, status: bool = True,
                    follow_clock: bool = False):
    """``(MidiInput, MidiFrames, MidiMasker)`` for a session playing over ``port``.

    The playhead is the stream clock the mic's frames count on, so it stamps the
    notes.  It only moves one audio block at a time, 5 ms, well inside a frame.
    """
    from .midi import MidiFrames, MidiInput, MidiMasker  # noqa: PLC0415

    def told(kind: str) -> None:
        if status:
            what = ("lost; the band hears silence until it is back"
                    if kind == "lost" else "back")
            print(f"midi: {port} {what}", file=sys.stderr, flush=True)

    def clock_told(kind: str) -> None:
        if status:
            what = ("is sending a clock; ignoring it (--midi-clock follows it)"
                    if kind == "ignored"
                    else "clock stopped; reading the tempo from your playing again")
            print(f"midi: {port} {what}", file=sys.stderr, flush=True)

    log = None if recorder is None else recorder.midi
    if log is not None:
        # First line, so a take with no notes yet still reads as a MIDI take.
        log({"t_s": 0.0, "type": "port", "port": port})
    source = MidiInput(port, clock=lambda: timeline.playhead / SR, on_change=told)
    frames = MidiFrames(source.events, sr=SR, log=log, follow_clock=follow_clock,
                        on_clock=clock_told)
    return source, frames, MidiMasker()


def _stream_input_latency_s(stream) -> float:
    """What the stream reports for its input side, for :attr:`MidiInput.offset_s`."""
    try:
        s = float(stream.latency[0])
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return 0.0
    return s if math.isfinite(s) and 0.0 <= s <= SELF_MASK_MAX_LATENCY_S else 0.0

def _import_sounddevice():
    """Lazy sounddevice import with a clean error when it is missing."""
    try:
        import sounddevice as sd  # noqa: PLC0415 — lazy by contract
    except Exception as e:  # noqa: BLE001 — any import failure is "not available"
        raise RealtimeError(
            "sounddevice is not available — live audio ('plume jam', "
            "'plume lead') needs it (pip install 'plume[play]'). Offline "
            f"twins: 'plume sim' and 'plume simlead'. ({e})"
        ) from None
    return sd


def _resolve_device(sd, device: str | None):
    """Pick a duplex-capable device; raise a clean error when there is none.

    Returns a value suitable for ``sounddevice.Stream(device=...)``:
    ``None`` (defaults) or an ``(input, output)`` index pair.
    """
    try:
        devices = sd.query_devices()
    except Exception as e:  # noqa: BLE001
        raise RealtimeError(f"audio device query failed: {e}") from None
    devices = list(devices)
    ins = [i for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
    outs = [i for i, d in enumerate(devices) if d.get("max_output_channels", 0) > 0]
    if device is not None:
        want = device.lower()
        hits = [i for i, d in enumerate(devices) if want in str(d.get("name", "")).lower()]
        if not hits:
            names = ", ".join(str(d.get("name", "?")) for d in devices) or "(none)"
            raise RealtimeError(f"no audio device matching {device!r} (devices: {names})")
        din = next((i for i in hits if i in ins), None)
        dout = next((i for i in hits if i in outs), None)
        if din is None or dout is None:
            din = din if din is not None else (ins[0] if ins else None)
            dout = dout if dout is not None else (outs[0] if outs else None)
        if din is None or dout is None:
            raise RealtimeError(
                f"device {device!r} cannot form a duplex pair "
                "(need both an input and an output device)")
        return (din, dout)
    if not ins or not outs:
        missing = "input" if not ins else "output"
        raise RealtimeError(
            f"no audio {missing} device found — live audio ('plume jam', "
            "'plume lead') needs a duplex mic+speaker setup. Try the offline "
            "twins: 'plume sim INPUT.wav' / 'plume simlead'.")
    return None  # let PortAudio use the defaults


#: What a refused pair of devices says, which the page knows it by.
OPEN_TOGETHER_FAILED = "could not open that input and output together"


@dataclasses.dataclass(frozen=True)
class InputRoute:
    """Which input a live session listens to.

    ``source`` is "direct" when the player says the band cannot reach this input (an
    instrument plugged in, or a mic with the band only in headphones), else "room".
    Every level threshold plume has was tuned on a room mic, and a direct input keeps
    them as a starting point, so its record says ``thresholds_calibrated: False``
    until someone has measured a real pickup against them.
    """
    device: object = None       # what Stream(device=) takes
    channels: int = 1           # input channels the stream must open
    guitar: int = 0             # 0-based index into indata
    voice: int | None = None    # 0-based index into indata, or no voice channel
    source: str = "room"
    device_name: str | None = None
    output_name: str | None = None      # only when the sound goes out somewhere she named

    def describe(self) -> dict:
        doc = {"source": self.source, "device": self.device_name,
               "channel": self.guitar + 1,
               "voice_channel": None if self.voice is None else self.voice + 1}
        if self.output_name is not None:
            doc["output"] = self.output_name
        if self.source == "direct":
            doc["thresholds_calibrated"] = False
        return doc


def resolve_input(sd, device: str | None = None, *, input_device: str | None = None,
                  input_channel: int | None = None, voice_channel: int | None = None,
                  direct: bool = False, output_device: str | None = None) -> InputRoute:
    """Turn the player's choice of input into a route, or say what is wrong with it.

    Channels are counted from 1, the way an interface labels its sockets.  With nothing
    asked for, this is exactly the old mono read of the system default input.

    ``output_device`` names where the sound goes, whatever the input's own outs are.  It takes
    the whole name, unlike ``device``: part of one could land on a neighbour when the named
    output has gone.  A name that finds no output is refused here, never swapped for another.
    """
    dev = _resolve_device(sd, device)
    devices = list(sd.query_devices())
    if input_device is not None:
        want = input_device.lower()
        hits = [i for i, d in enumerate(devices) if want in str(d.get("name", "")).lower()]
        if not hits:
            names = ", ".join(str(d.get("name", "?")) for d in devices) or "(none)"
            raise RealtimeError(
                f"no audio device matching {input_device!r} (devices: {names})")
        din = next((i for i in hits if devices[i].get("max_input_channels", 0) > 0), None)
        if din is None:
            raise RealtimeError(f"{devices[hits[0]].get('name')!r} has no inputs")
        dev = (din, dev[1] if isinstance(dev, (tuple, list)) else None)
    output_name = None
    if output_device is not None:
        want = output_device.lower()
        hits = [i for i, d in enumerate(devices) if want == str(d.get("name", "")).lower()]
        if not hits:
            names = ", ".join(str(d.get("name", "?")) for d in devices) or "(none)"
            raise RealtimeError(
                f"no audio output matching {output_device!r} (devices: {names})")
        dout = next((i for i in hits if devices[i].get("max_output_channels", 0) > 0), None)
        if dout is None:
            raise RealtimeError(f"{devices[hits[0]].get('name')!r} has no outputs")
        output_name = str(devices[dout].get("name", "?"))
        dev = (dev[0] if isinstance(dev, (tuple, list)) else None, dout)
    named = dev[0] if isinstance(dev, (tuple, list)) else None
    if named is None and (input_channel is not None or voice_channel is not None):
        # The system default can change under a take, and a channel number means
        # nothing without knowing whose it is.
        raise RealtimeError("choosing a channel needs a device: add --input-device NAME "
                            "(run 'plume inputs' to see them)")
    guitar = 0 if input_channel is None else int(input_channel) - 1
    voice = None if voice_channel is None else int(voice_channel) - 1
    if named is not None:
        name = str(devices[named].get("name", "?"))
        have = int(devices[named].get("max_input_channels", 0))
        for label, ch in (("--input-channel", guitar), ("--voice-channel", voice)):
            if ch is not None and not 0 <= ch < have:
                raise RealtimeError(f"{label} {ch + 1}: {name} has channels 1 to {have}")
    else:
        name = None
    if voice == guitar:
        raise RealtimeError(f"--voice-channel and --input-channel are the same channel "
                            f"({guitar + 1}): the voice needs its own")
    return InputRoute(device=dev, channels=max(guitar, -1 if voice is None else voice) + 1,
                      guitar=guitar, voice=voice, source="direct" if direct else "room",
                      device_name=name, output_name=output_name)


def input_devices(sd) -> list[dict]:
    """Every device and its channel counts.  Opens nothing.

    And, where the library says, how late it reports its output would be, and which one the
    system sends its sound to.
    """
    try:
        devices = list(sd.query_devices())
    except Exception as e:  # noqa: BLE001
        raise RealtimeError(f"audio device query failed: {e}") from None
    try:
        system_output = str(sd.query_devices(kind="output")["name"])
    except Exception:  # noqa: BLE001 - nothing depends on it, and a stream would still say
        system_output = None
    rows = []
    for d in devices:
        row = {"name": str(d.get("name", "?")), "inputs": int(d.get("max_input_channels", 0)),
               "outputs": int(d.get("max_output_channels", 0))}
        if row["outputs"] > 0:
            late = d.get("default_low_output_latency")
            if isinstance(late, (int, float)) and math.isfinite(late) and late > 0:
                row["output_ms"] = round(float(late) * 1000.0, 1)
            if row["name"] == system_output:
                row["default_output"] = True
        rows.append(row)
    return rows


#: Sessions with no ending to finish: nothing to flush, and sound to silence.
_NOTHING_TO_FINISH = ("play", "calibrate")

#: The meter holds a mic open with no session around it, so no cap and no plume stop.
MAX_METER_S = 60.0


def meter_input(sd, name: str, *, seconds: float = 5.0) -> dict:
    """Listen to every channel of one device for a few seconds; make no sound.

    An input opened without anyone asking is still a microphone open in someone's
    room, so it takes the same consent as playing.
    """
    if not 0.0 < float(seconds) <= MAX_METER_S or int(float(seconds) * SR) < 1:
        raise RealtimeError(f"--seconds has to be more than 0 and at most "
                            f"{MAX_METER_S:.0f}, not {seconds}")
    allowed, why = audio_output_allowed()
    if not allowed:
        raise RealtimeError(f"refusing to open a live input: {why}. "
                            f"Set {AUDIO_CONSENT_ENV}=1 to allow it.")
    route = resolve_input(sd, input_device=name)
    din = route.device[0]
    have = int(list(sd.query_devices())[din].get("max_input_channels", 0))
    y = sd.rec(int(float(seconds) * SR), samplerate=SR, channels=have, dtype="float32",
               device=din)
    sd.wait()

    def db(v):
        return round(max(RMS_FLOOR_DB, 20.0 * math.log10(max(float(v), 1e-9))), 1)
    rows = [{"channel": c + 1, "peak_db": db(np.abs(y[:, c]).max()),
             "rms_db": db(np.sqrt(np.mean(y[:, c] ** 2)))} for c in range(have)]
    return {"device": route.device_name, "seconds": float(seconds), "channels": rows,
            "loudest_channel": max(rows, key=lambda r: r["peak_db"])["channel"]}


#: Live audio plays out of a real human's speakers.  Twice during development an
#: automated agent started a live session and left it running,
#: still audible, with no way to stop it short of finding the PID.
#: Sound out of a machine is a physical act in someone's room, so it is opt-in:
#: an interactive terminal, or an explicit PLUME_ALLOW_AUDIO=1.
AUDIO_CONSENT_ENV = "PLUME_ALLOW_AUDIO"
#: Spellings of that variable that say no.  No one, and no page, may overrule these.
AUDIO_CONSENT_OFF = frozenset({"0", "false", "no", "off"})

#: Belt and braces for the same incident: a live session with nobody watching
#: stops on its own.  0 disables the cap (an interactive player can opt out).
DEFAULT_MAX_MINUTES = 60.0

#: Why a live session ended.  These strings are the user-visible status line
#: AND the ``reason`` in ``ending.json``, so a recording says for itself
#: whether a human ended it, another terminal did, or the cap did.
END_INTERRUPT = "interrupted (Ctrl-C / SIGINT)"
END_TERMINATED = "asked to terminate (SIGTERM)"
END_STOPPED = "stopped by 'plume stop'"
END_FORM = "the arrangement ended"
END_SILENCE = "faded out after hearing nothing from you"
END_SONG = "the song wound down as planned"
END_MAX_MINUTES = "reached the max-minutes cap"
END_WEDGED = "ended by the hard deadline (the graceful ending did not finish)"


# ------------------------------------------------- v1.3: the ending's budget
#
# Every wait on the way out is bounded, and the bounds ADD UP to a number the
# watchdog below can enforce.  Before v1.3 they did not: the ending was a 10 s
# join plus a 30 s drain plus an *unbounded* PortAudio close, and a session
# that stalled in that close could not be ended by anything short of SIGKILL
# (see :class:`HardExit` for why the signals could not reach it).

#: Planner join.  The planner loop checks its stop Event every <=20 ms, so
#: this is ~150x more patience than a healthy planner can possibly need.
PLANNER_JOIN_S = 3.0
#: Analysis join.  Same reasoning; the analysis loop's drain is bounded too.
ANALYSIS_JOIN_S = 2.0
#: Join for the cue/control socket's accept thread (it polls at 0.2 s).
CUE_CLOSE_S = 2.0
#: Absolute ceiling on "let what is already scheduled finish playing".  The
#: wait normally ends the instant the playhead reaches the end of the
#: schedule; the budget it is actually given is derived from how much audio
#: is left (see _drain_timeline), so a slow bar is never clipped and a device
#: that has stopped advancing the playhead is noticed quickly.  This ceiling
#: only exists so the total ending budget below can be a real number.
TAIL_DRAIN_MAX_S = 12.0
#: How long after the LAST bounded step the process gives itself before it
#: stops asking nicely.
HARD_EXIT_GRACE_S = 5.0
#: Total budget for a graceful ending, derived rather than declared so it can
#: never drift below the parts it has to contain.
ENDING_HARD_DEADLINE_S = (PLANNER_JOIN_S + TAIL_DRAIN_MAX_S + ANALYSIS_JOIN_S
                          + CUE_CLOSE_S + HARD_EXIT_GRACE_S)
#: Exit status when the watchdog had to end the process itself.  Distinct from
#: 0 (clean) and from 130/143 (killed by a signal) so a wrapper can tell the
#: difference between "it ended" and "it had to be ended".
EXIT_WEDGED = 75
#: How long the analysis loop keeps draining its rings after a hard stop.
ANALYSIS_DRAIN_S = 0.5


def resolve_max_minutes(max_minutes: float | None) -> float:
    """Normalize a ``--max-minutes`` value to a float; 0.0 means "no cap".

    ``None`` (the flag was not given) means :data:`DEFAULT_MAX_MINUTES` — the
    cap is opt-OUT, because the failure it exists for is the case where nobody
    is there to opt into anything.
    """
    if max_minutes is None:
        max_minutes = DEFAULT_MAX_MINUTES
    m = float(max_minutes)
    if not math.isfinite(m) or m < 0.0:
        raise ValueError(
            f"max_minutes must be a finite number >= 0 (0 disables the cap), "
            f"got {max_minutes!r}")
    return m


def session_deadline(started: float, max_minutes: float | None) -> float | None:
    """The instant a live session must end itself at, or ``None`` for no cap.

    Pure: a function of its two arguments and nothing else — ``started`` is
    supplied by the caller (a :func:`time.monotonic` reading in the live
    transports, an epoch stamp in the registry), never read in here.  That is
    what makes the cap arithmetic unit-testable without a clock or a device.
    """
    m = resolve_max_minutes(max_minutes)
    if m == 0.0:
        return None
    return float(started) + m * 60.0


def audio_output_allowed() -> tuple[bool, str]:
    """Return ``(allowed, why_not)`` for opening a live audio stream.

    Consent is granted by an interactive stdin/stdout (a human is sitting
    there) or by setting ``PLUME_ALLOW_AUDIO=1`` deliberately.
    """
    flag = os.environ.get(AUDIO_CONSENT_ENV, "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True, ""
    if flag in AUDIO_CONSENT_OFF:
        return False, f"{AUDIO_CONSENT_ENV} is set to '{flag}'"
    try:
        if sys.stdin.isatty() and sys.stdout.isatty():
            return True, ""
    except (AttributeError, ValueError):  # detached / closed streams
        pass
    return False, "not an interactive terminal"


def check_audio_consent(device: str | None = None) -> None:
    """Raise :class:`RealtimeError` unless live audio output is permitted."""
    allowed, why = audio_output_allowed()
    if not allowed:
        raise RealtimeError(
            f"refusing to play live audio: {why}. Live sessions come out of "
            "real speakers in someone's room, so they are opt-in when nothing "
            f"is attached to the terminal. Set {AUDIO_CONSENT_ENV}=1 to allow "
            "it, or use the offline twins, which never touch the audio device: "
            "'plume sim INPUT.wav' / 'plume simlead'.")


def check_device(device: str | None = None) -> None:
    """Verify live audio is possible (sounddevice importable + a duplex-capable
    device + consent); raise :class:`RealtimeError` otherwise.

    Lets the CLI fail fast on the no-device path before building the mind.

    **Order matters, and it is a v2.1 amendment (SPEC-V2 section 5).**  The
    consent guard used to run FIRST, which meant a headless box with no audio
    hardware was told "set PLUME_ALLOW_AUDIO=1" — advice that cannot work,
    because setting it only reveals the real, unfixable error one line later.
    A dead end dressed as a next step is worse than no guidance.  The
    dependency and the device are facts about the machine and are reported
    first; consent is a fact about the *session* and is only interesting once
    live audio is actually possible.  Nothing is loosened: on a box that can
    play, the guard still fires before any stream is opened.
    """
    sd = _import_sounddevice()
    _resolve_device(sd, device)
    check_audio_consent(device)


# ------------------------------------------------- v1.3: the status readout

class StatusLine:
    """The live readout, in whichever form the reader can actually consume.

    On a terminal this is one line rewritten in place with a carriage return —
    the pretty display, unchanged.  **In a pipe it is one newline-terminated
    line per update instead**, because a carriage-return display in a pipe is
    not a display at all: it is a single unbounded line with no newline
    anywhere in it, which `tail -f`, `grep --line-buffered`, a log collector
    and an agent's `readline()` all block on forever.  A six-minute session
    that "printed status the whole time" was unreadable to everything that was
    not a human staring at a terminal.

    ``close`` terminates an in-place line so whatever prints next starts on a
    line of its own; in a pipe it is a no-op, because every update already
    ended one.
    """

    def __init__(self, stream=None, *, tty: bool | None = None,
                 enabled: bool = True):
        self._stream = sys.stdout if stream is None else stream
        self.enabled = bool(enabled)
        if tty is None:
            try:
                tty = bool(self._stream.isatty())
            except (AttributeError, ValueError):   # detached/closed streams
                tty = False
        self.tty = bool(tty)
        self._open_line = False
        self.updates = 0

    def update(self, text: str) -> None:
        """Emit one status update.  Never raises: a broken pipe (the reader
        went away) disables the readout rather than killing the band."""
        if not self.enabled:
            return
        try:
            self._stream.write(("\r" + text) if self.tty else (text + "\n"))
            self._stream.flush()
        except (OSError, ValueError):
            self.enabled = False
            return
        self._open_line = self.tty
        self.updates += 1

    def close(self) -> None:
        if self._open_line and self.enabled:
            try:
                self._stream.write("\n")
                self._stream.flush()
            except (OSError, ValueError):
                self.enabled = False
        self._open_line = False


# --------------------------------------------- v1.3: records, as they happen

#: Files a take is made of.  Named here because an observer tailing a live
#: session needs to know them before the session has written anything.
REC_MANIFEST = "session.json"
REC_DECISIONS = "decisions.ndjson"
REC_CUES = "cues.ndjson"
REC_HUMAN = "human.wav"
REC_MACHINE = "machine.wav"
REC_ENDING = "ending.json"
REC_VOICE = "voice.wav"
REC_MIDI = "midi.ndjson"
#: What the device reported about latency, written once the stream is open.
REC_STREAM = "stream.json"
#: Keep's sidecar.  Plain JSON beside the take, so it outlives the page that wrote it.
REC_KEEP = "keep.json"

#: Audio is flushed to disk at most this often — the most a `kill -9` can
#: cost.  Decisions are flushed every line (they are what an observer reads).
REC_AUDIO_FLUSH_S = 1.0


def _jsonable(obj):
    """Best-effort JSON view of a params dataclass, for the manifest."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
        except (TypeError, ValueError):      # pragma: no cover - defensive
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def _refuse_a_kept_take(path) -> None:
    """Keep promises that nothing gets deleted, and a new take would overwrite this one."""
    if path is None:
        return
    try:
        with open(Path(path) / REC_KEEP, encoding="utf-8") as f:
            kept = json.load(f).get("kept") is True
    except (OSError, ValueError, AttributeError):
        return
    if kept:
        raise RealtimeError(f"{path} holds a take you kept, so nothing was recorded over it. "
                            "Give --record another folder.")


class SessionRecorder:
    """A take written to disk WHILE it happens, not after it (v1.3).

    The defect this replaces, from the first session with a human and a
    guitar: ``--record DIR`` accumulated every decision and every sample in
    memory and wrote the whole lot on a graceful exit.  So the directory sat
    empty for six minutes (nothing to watch, nothing to tail, no way to tell a
    working session from a wedged one), a session that had to be `kill -9`ed
    lost the entire take, and a forty-minute session would have held forty
    minutes of float32 in RAM waiting for an ending that might never come.

    So:

    - ``session.json`` is written at START, before the first bar, so an
      observer knows what is running the moment it is running;
    - every decision is one flushed line of ``decisions.ndjson`` — tailable
      live, and a kill costs at most the line being written;
    - every cue received is one flushed line of ``cues.ndjson``.  The file is
      created at open, so "present and empty" means no cues were sent and
      "absent" means a take from before this log existed;
    - ``human.wav``/``machine.wav`` are streamed through open ``soundfile``
      handles and flushed about once a second.  libsndfile updates the RIFF
      header on flush, so the partial files are valid, playable and
      correct-length at all times (verified, not assumed);
    - ``ending.json`` is written last and says why the take stops there.

    Every method is safe from any thread (the planner writes decisions, the
    analysis thread writes audio) and none of them raises: a recording that
    fails must never take the band down.  Failures are counted in
    :attr:`errors` and reported once at the end.
    """

    def __init__(self, rec_dir, *, sr: int = SR, subtype: str = "PCM_24",
                 flush_every_s: float = REC_AUDIO_FLUSH_S, voice: bool = False):
        self._want_voice = bool(voice)
        self._voice = None
        self.dir = Path(rec_dir)
        self.sr = int(sr)
        self.subtype = str(subtype)
        self._flush_every = float(flush_every_s)
        self._lock = threading.RLock()
        self._decisions = None            # text handle
        self._cues = None                 # text handle
        self._human = None                # soundfile handle
        self._machine = None
        self._midi = None                 # opened by the first midi() line
        self._last_flush = 0.0
        self.decisions_written = 0
        self.frames_written = 0
        self.errors: list[str] = []
        self.closed = False
        self._cue_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._cue_writer: threading.Thread | None = None
        self._cue_writer_done = False
        self._cue_writer_guard = threading.Lock()

    # -- lifecycle ------------------------------------------------------

    def open(self, manifest: dict | None = None) -> "SessionRecorder":
        """Create the directory, write the manifest, open every handle."""
        import soundfile as sf  # noqa: PLC0415 — keep module import light

        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            # A used directory: the last take's ending must not read as this one's,
            # and its keys or voice must not stand in for a take that has none.
            (self.dir / REC_ENDING).unlink(missing_ok=True)
            (self.dir / REC_KEEP).unlink(missing_ok=True)
            (self.dir / REC_STREAM).unlink(missing_ok=True)
            # _midi_listening writes a MIDI take's first line before this runs, so a
            # handle here means this take's own file.  Without one the file is the last take's.
            if self._midi is None:
                (self.dir / REC_MIDI).unlink(missing_ok=True)
            if not self._want_voice:
                (self.dir / REC_VOICE).unlink(missing_ok=True)
            if manifest is not None:
                self._write_json(REC_MANIFEST, manifest)
            try:
                self._decisions = open(self.dir / REC_DECISIONS, "w",
                                       encoding="utf-8", buffering=1)
            except OSError as e:
                self._note(f"decisions: {e}")
            try:
                self._cues = open(self.dir / REC_CUES, "w",
                                  encoding="utf-8", buffering=1)
            except OSError as e:
                self._note(f"cues: {e}")
            try:
                self._human = sf.SoundFile(
                    str(self.dir / REC_HUMAN), mode="w", samplerate=self.sr,
                    channels=1, subtype=self.subtype)
                self._machine = sf.SoundFile(
                    str(self.dir / REC_MACHINE), mode="w", samplerate=self.sr,
                    channels=1, subtype=self.subtype)
                if self._want_voice:
                    self._voice = sf.SoundFile(
                        str(self.dir / REC_VOICE), mode="w", samplerate=self.sr,
                        channels=1, subtype=self.subtype)
            except (OSError, RuntimeError, ValueError) as e:
                self._note(f"audio: {e}")
            self._last_flush = time.monotonic()
        return self

    def stream_opened(self, doc: dict) -> None:
        """Written once the device is open.  Nothing is known about it at ``open``."""
        with self._lock:
            self._write_json(REC_STREAM, doc)

    def cue_soon(self, doc: dict) -> None:
        """Log a cue from the socket thread, which must never wait on this lock.

        That thread also answers ``plume stop``, so a stalled disk here would stall
        the stop.  A small thread of our own does the waiting instead.
        """
        self._cue_queue.put(doc)
        with self._cue_writer_guard:
            if self._cue_writer is None and not self._cue_writer_done:
                self._cue_writer = threading.Thread(target=self._write_cues, daemon=True,
                                                    name="plume-cue-log")
                self._cue_writer.start()

    def _write_cues(self) -> None:
        while True:
            doc = self._cue_queue.get()
            if doc is None:
                return
            self.cue(doc)

    def _end_cue_writer(self, timeout: float) -> None:
        """Let queued cues reach the file before it closes, within reason."""
        with self._cue_writer_guard:
            self._cue_writer_done = True
            writer, self._cue_writer = self._cue_writer, None
        if writer is not None:
            self._cue_queue.put(None)
            writer.join(timeout=timeout)

    def finish(self, ending: dict | None = None) -> None:
        """Write ``ending.json`` (if given) and close every handle.  Idempotent."""
        self._end_cue_writer(timeout=1.0)
        with self._lock:
            if ending is not None and not self.closed:
                self._write_json(REC_ENDING, ending)
            self.close()

    def close(self) -> None:
        self._end_cue_writer(timeout=1.0)
        with self._lock:
            if self.closed:
                return
            self.closed = True
            for name, h in (("decisions", self._decisions), ("cues", self._cues),
                            ("human", self._human), ("machine", self._machine),
                            ("voice", self._voice), ("midi", self._midi)):
                if h is None:
                    continue
                try:
                    h.close()
                except (OSError, RuntimeError) as e:      # pragma: no cover
                    self._note(f"{name} close: {e}")
            self._decisions = self._cues = self._human = self._machine = None
            self._voice = self._midi = None

    def emergency_close(self, ending: dict | None = None,
                        *, timeout: float = 1.0) -> bool:
        """Flush and close from a watchdog, without ever blocking on the lock.

        Called off the hard-exit path, where waiting on a lock held by a thread
        that is itself stuck would defeat the whole point.  Returns whether the
        lock was taken; either way the per-line decision flushes and the
        per-second audio flushes have already made the files valid.
        """
        self._end_cue_writer(timeout=0.0)
        got = self._lock.acquire(timeout=float(timeout))
        if not got:
            return False
        try:
            self.finish(ending)
        finally:
            self._lock.release()
        return True

    # -- writes ---------------------------------------------------------

    def decision(self, doc: dict) -> None:
        """Append one decision and FLUSH it — this is the live view."""
        with self._lock:
            if self._decisions is None:
                return
            try:
                self._decisions.write(json.dumps(doc, sort_keys=True) + "\n")
                self._decisions.flush()
                self.decisions_written += 1
            except (OSError, ValueError, TypeError) as e:
                self._note(f"decision: {e}")

    def cue(self, doc: dict) -> None:
        """Append one received cue and FLUSH it.  Who asked, and when, exists
        nowhere else: the decision log only says where a cue applied."""
        with self._lock:
            if self._cues is None:
                return
            try:
                self._cues.write(json.dumps(doc, sort_keys=True) + "\n")
                self._cues.flush()
            except (OSError, ValueError, TypeError) as e:
                self._note(f"cue: {e}")

    def midi(self, doc: dict) -> None:
        """Append one MIDI event, flushed like a decision: ``plume watch`` reads it."""
        with self._lock:
            if self.closed:
                return
            try:
                if self._midi is None:
                    self.dir.mkdir(parents=True, exist_ok=True)
                    self._midi = open(self.dir / REC_MIDI, "w",
                                      encoding="utf-8", buffering=1)
                self._midi.write(json.dumps(doc, sort_keys=True) + "\n")
                self._midi.flush()
            except (OSError, ValueError, TypeError) as e:
                self._note(f"midi: {e}")

    def audio(self, human: np.ndarray, machine: np.ndarray,
              voice: np.ndarray | None = None) -> None:
        """Append one analysis chunk of each bus; flush about once a second."""
        with self._lock:
            if self._human is None or self._machine is None:
                return
            try:
                self._human.write(np.asarray(human, dtype=np.float32))
                self._machine.write(np.asarray(machine, dtype=np.float32))
                if self._voice is not None and voice is not None:
                    self._voice.write(np.asarray(voice, dtype=np.float32))
                self.frames_written += int(np.asarray(human).shape[0])
                now = time.monotonic()
                if now - self._last_flush >= self._flush_every:
                    self._human.flush()
                    self._machine.flush()
                    if self._voice is not None:
                        self._voice.flush()
                    self._last_flush = now
            except (OSError, RuntimeError, ValueError) as e:
                self._note(f"audio: {e}")

    def flush(self) -> None:
        with self._lock:
            for h in (self._decisions, self._cues, self._human, self._machine, self._voice):
                if h is None:
                    continue
                try:
                    h.flush()
                except (OSError, RuntimeError, ValueError):   # pragma: no cover
                    pass
            self._last_flush = time.monotonic()

    # -- internals ------------------------------------------------------

    def _write_json(self, name: str, doc: dict) -> None:
        try:
            tmp = self.dir / f"{name}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, sort_keys=True)
            os.replace(tmp, self.dir / name)   # observers never see half a file
        except (OSError, TypeError, ValueError) as e:
            self._note(f"{name}: {e}")

    def _note(self, msg: str) -> None:
        if len(self.errors) < 20:
            self.errors.append(str(msg))


def session_manifest(*, kind: str, style: str, role: str, seed: int,
                     session_name: str | None = None,
                     socket_path: str | None = None,
                     max_minutes: float = DEFAULT_MAX_MINUTES,
                     record_dir: str | None = None,
                     device: str | None = None, bpm: float | None = None,
                     params=None, role_params=None,
                     started_at: float | None = None,
                     input_route: "InputRoute | None" = None,
                     midi_port: str | None = None, detect: bool = False,
                     monitor: bool = False) -> dict:
    """What ``session.json`` says, written before the first bar is planned.

    Pure: everything except the pid and the clock is passed in, so the shape
    is testable without a device.
    """
    t = float(time.time() if started_at is None else started_at)
    return {
        "pid": os.getpid(),
        "kind": str(kind),
        "style": str(style or ""),
        "role": str(role or ""),
        "seed": int(seed),
        "session": session_name,
        "socket": socket_path,
        "device": device,
        "input": (input_route or InputRoute()).describe(),
        "midi_port": midi_port,
        "detect": bool(detect),
        "monitor": bool(monitor),
        "bpm": (None if bpm is None else float(bpm)),
        "sr": SR,
        "max_minutes": float(max_minutes),
        "record_dir": (None if record_dir is None else str(record_dir)),
        "started_at": round(t, 3),
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)),
        "streaming": True,
        "files": {"decisions": REC_DECISIONS, "human": REC_HUMAN,
                  "machine": REC_MACHINE, "ending": REC_ENDING},
        "stop_command": f"plume stop --pid {os.getpid()}",
        "params": _jsonable(params),
        "role_params": _jsonable(role_params),
    }


def stream_report(stream, *, block: int = BLOCK, sr: int = SR,
                  calibrated_ms: float | None = None) -> dict:
    """What can honestly be said about this stream's latency, in milliseconds.

    ``input_ms`` and ``output_ms`` are the device's own word, not a measurement, and
    ``reported_ms`` is their sum.  ``calibrated_ms`` is a round trip ``plume calibrate``
    really measured on this pair of devices, None if nobody has.
    """
    in_ms = out_ms = reported = None
    lat = getattr(stream, "latency", None)
    if isinstance(lat, (tuple, list)) and len(lat) == 2:
        try:
            pair = float(lat[0]) * 1000.0, float(lat[1]) * 1000.0
        except (TypeError, ValueError):
            pair = (float("nan"), float("nan"))
        if all(math.isfinite(v) for v in pair):
            in_ms, out_ms = round(pair[0], 3), round(pair[1], 3)
            reported = round(pair[0] + pair[1], 3)
    return {"input_ms": in_ms, "output_ms": out_ms, "reported_ms": reported,
            "block_ms": round(1000.0 * block / sr, 3),
            "calibrated_ms": None if calibrated_ms is None else round(float(calibrated_ms), 3)}


# ------------------------------------------- v1.3: the ending that cannot hang

class HardExit:
    """A watchdog that makes ``plume stop`` true: the process actually ends.

    **The measured failure.**  ``plume stop`` on a live session reported, after
    walking its whole ladder: *"pid 36655 is still running after socket,
    SIGINT, SIGTERM — it is wedged"*, and ``kill -9`` was the only thing left.
    Two things had to be true at once for that, and both were:

    1. **The ending had an unbounded step.**  ``with ... stream:`` exits
       through ``sounddevice``'s ``Stream.stop()``, which is ``Pa_StopStream``,
       which is documented to *wait until all pending audio buffers have been
       played*.  On a device that has stopped advancing (the built-in duplex
       pair, feeding back into its own mic, is the exact case) that wait does
       not end.

    2. **A blocked main thread cannot run a Python signal handler.**  CPython
       runs handlers only in the main thread's eval loop, so while the main
       thread sits in that C call nothing is delivered.  And
       :class:`GracefulSignals` had, for good reasons, replaced SIGTERM's
       *default* disposition — which the kernel applies with no interpreter
       involvement — with a Python-level handler.  So plume's own graceful
       ending had made the process harder to kill than an ordinary program:
       the one signal that would have ended it regardless was now waiting for
       an interpreter that would never get another turn.

    The fix cannot live on the main thread, cannot be a signal, and cannot be
    a `join`.  It is this: an ordinary daemon thread, armed the moment the
    ending begins, which flushes the records and calls ``os._exit`` if the
    ending has not finished by its deadline.  ``os._exit`` is the one exit that
    works from any thread, needs no interpreter cooperation from the main
    thread, and cannot be blocked by a C call in it.  It is safe to reach for
    only because the records are already on disk — which is what makes the
    streaming recorder above a prerequisite for this and not a separate
    feature.
    """

    def __init__(self, *, code: int = EXIT_WEDGED, exit_fn=None, on_exit=None,
                 stream=None, poll: float = 0.05):
        self.code = int(code)
        self._exit = exit_fn if exit_fn is not None else os._exit
        self._on_exit = on_exit
        self._stream = stream
        self._poll = float(poll)
        self._lock = threading.Lock()
        self._released = threading.Event()      # disarm / done
        self._deadline: float | None = None
        self._thread: threading.Thread | None = None
        self.reason: str | None = None
        self.fired = False

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._deadline is not None and not self._released.is_set()

    @property
    def remaining_s(self) -> float | None:
        with self._lock:
            if self._deadline is None:
                return None
            return self._deadline - time.monotonic()

    def arm(self, reason: str, *, seconds: float | None = None) -> "HardExit":
        """Start (or SHORTEN — never extend) the fuse.

        Shortening only is deliberate: a second Ctrl-C from an impatient player
        must be able to bring the deadline in, and nothing must be able to push
        it back out again.

        ``seconds`` defaults to :data:`ENDING_HARD_DEADLINE_S` read AT CALL
        TIME, not bound into the signature — a default frozen at import is a
        constant a test cannot move, and this one has to be movable or the
        watchdog can only be tested by waiting eighteen seconds for it.
        """
        if seconds is None:
            seconds = ENDING_HARD_DEADLINE_S
        with self._lock:
            if self.fired:
                return self
            new = time.monotonic() + max(0.0, float(seconds))
            if self._deadline is None or new < self._deadline:
                self._deadline = new
                self.reason = str(reason)
            if self._thread is None:
                self._released.clear()
                self._thread = threading.Thread(target=self._run,
                                                name="plume-hard-exit",
                                                daemon=True)
                self._thread.start()
        return self

    def disarm(self) -> None:
        """The ending finished on its own.  Nothing to enforce."""
        with self._lock:
            self._deadline = None
        self._released.set()

    def __enter__(self) -> "HardExit":
        return self

    def __exit__(self, *exc) -> None:
        self.disarm()

    # -- internals ------------------------------------------------------

    def _run(self) -> None:
        while not self._released.is_set():
            with self._lock:
                deadline = self._deadline
            if deadline is None:
                self._released.wait(self._poll)
                continue
            left = deadline - time.monotonic()
            if left <= 0.0:
                self._fire()
                return
            self._released.wait(min(self._poll, left))

    def _fire(self) -> None:
        with self._lock:
            if self.fired:
                return                                # pragma: no cover
            self.fired = True
            reason = self.reason
        try:
            if self._on_exit is not None:
                self._on_exit(reason)
        except Exception:                             # noqa: BLE001
            pass                                      # pragma: no cover
        self._say(
            f"plume: the graceful ending did not finish ({reason}); ending the "
            f"process anyway. The take is on disk — it has been written as it "
            f"went, so nothing after the last flush is lost.")
        try:
            self._exit(self.code)
        except SystemExit:                            # pragma: no cover
            raise
        except Exception:                             # noqa: BLE001
            pass                                      # pragma: no cover

    def _say(self, msg: str) -> None:
        stream = self._stream if self._stream is not None else sys.stderr
        try:
            stream.write(msg + "\n")
            stream.flush()      # os._exit skips buffer flushing; do it here
        except (OSError, ValueError, AttributeError):   # pragma: no cover
            pass


def _hard_exit_cleanup(recorder: "SessionRecorder | None", cues: "CueServer | None",
                       ending_doc):
    """What a hard exit does on its way out: close the take, unregister.

    Deliberately small and deliberately non-blocking.  Everything expensive
    (the decisions, the audio) is already on disk because the recorder wrote
    it as it went; this only caps the files off and removes the registry entry
    so ``plume sessions`` does not list a process that no longer exists.
    """

    def _cleanup(reason: str | None) -> None:
        if recorder is not None:
            doc = None
            try:
                doc = ending_doc()
            except Exception:                          # noqa: BLE001
                doc = None                             # pragma: no cover
            recorder.emergency_close(doc)
        if cues is not None:
            try:
                os.unlink(cues.path)
            except OSError:
                pass
        unregister_session()

    return _cleanup


def _abort_stream(stream) -> bool:
    """Stop a PortAudio stream the non-blocking way, before its blocking one.

    ``Stream.abort()`` is ``Pa_AbortStream``, which returns immediately and
    discards whatever is still queued; ``Stream.stop()`` — which the context
    manager's ``__exit__`` calls — is ``Pa_StopStream``, which waits for the
    queue to play out and is where a wedged device parks forever.  Calling
    abort first, once the tail has already been drained, leaves that wait with
    nothing to wait for.  Returns whether an abort was available and taken;
    fake streams in tests have no ``abort`` and are unaffected.
    """
    fn = getattr(stream, "abort", None)
    if not callable(fn):
        return False
    try:
        fn()
    except Exception:                                  # noqa: BLE001
        return False                                   # pragma: no cover
    return True


def _drain_timeline(timeline: "TimelineBuffer", *,
                    max_s: float | None = None) -> bool:
    """Let everything already scheduled actually play.  Bounded, always.

    Returns True when the schedule finished, False when the cap ended the
    wait — which means the playhead stopped advancing, i.e. the device is
    already not playing anything and waiting longer buys silence.

    The budget is DERIVED from what is left to play (half as long again, plus
    a second) rather than being one constant for every tempo.  A fixed cap has
    to be either long enough for a slow bar — in which case a dead device
    holds the ending open for all of it — or short enough to notice a dead
    device, in which case it truncates the last bar of a slow tune.  Deriving
    it does both: a healthy device always gets more time than it needs, and a
    stalled one is given up on in proportion to what it still owed.
    """
    end = timeline.end
    if max_s is None:
        remaining_s = max(0, end - timeline.playhead) / float(SR)
        max_s = min(TAIL_DRAIN_MAX_S, remaining_s * 1.5 + 1.0)
    until = time.monotonic() + max(0.0, float(max_s))
    while timeline.playhead < end:
        if time.monotonic() >= until:
            return False
        time.sleep(0.02)
    return True


# ------------------------------------------------------------------ jam

def jam(*, style: str, seed: int, params, device: str | None = None,
        monitor: bool = False, record_dir: str | None = None,
        tightness: float | None = None,
        role=None, lead_in: str | None = None,
        session=None, socket_path: str | None = None,
        form_params=None, from_bar: int = 0,
        max_minutes: float | None = None,
        input_mode: str | None = None, count_in: bool = False,
        latency_comp_s: float | None = None,
        input_device: str | None = None, input_channel: int | None = None,
        direct: bool = False, midi: str | None = None,
        midi_clock: bool = False) -> None:
    """Run the live sideman until it is ended (SPEC-V1 section 7).

    ``params`` is a ``plume.mind.MindParams``.  ``tightness`` overrides the
    beat tracker's phase-correction alpha (CLI ``--tightness``; additive to
    the frozen signature — the CLI contract requires the flag).

    v2 (additive, SPEC-V2 section 5): ``role`` is a
    :class:`plume.roles.RoleParams` and ``lead_in`` a compose prompt.  With
    neither — or with a plain ``follow`` role and no lead-in — this is the v1
    sideman, byte for byte in its musical decisions (since v2.5, bars are
    PLACED earlier by the resolved latency compensation; ``--latency-comp 0``
    restores v2.4 placement).  Anything else means "the machine starts", which
    is :func:`lead`, and it is dispatched there rather than reimplemented:
    one live transport is the only way the v1 default can be *guaranteed*
    unchanged (see the SPEC-V2 section 5 amendments).

    v2.2 (additive, and the reason this docstring no longer says "until
    Ctrl-C"): the session registers itself so ``plume sessions`` can name it
    and ``plume stop`` can end it from another terminal, and it ends itself
    after ``max_minutes`` (default :data:`DEFAULT_MAX_MINUTES`; 0 disables).
    All three exits — Ctrl-C, ``plume stop``, the cap — run the same graceful
    ending, and none of them touches a musical decision.

    v2.5 (additive): ``latency_comp_s`` overrides the placement compensation
    (see :mod:`plume.calibrate`); ``None`` resolves it (calibrated, else
    stream-reported), ``0`` disables it (the v2.4 behavior).  Follow mode
    only — a lead dispatch does not carry it, because in lead mode the machine
    owns the clock and the analogous correction is on the ANALYSIS side (a
    known residual, in the spec).

    ``midi`` (additive): a port to listen to instead of the mic (see
    :mod:`plume.midi`), by name or ``auto``.  The mic is still recorded to
    ``human.wav``.  ``None`` never touches the MIDI library.  ``midi_clock``: let
    that port's MIDI clock set the tempo and the bar line; otherwise it is ignored.
    """
    if _wants_lead(role, lead_in, session):
        return lead(style=style, seed=seed, params=params, role=role,
                    prompt=lead_in, session=session, device=device,
                    monitor=monitor, record_dir=record_dir,
                    tightness=tightness, socket_path=socket_path,
                    form_params=form_params, from_bar=from_bar,
                    max_minutes=max_minutes,
                    input_mode=input_mode, count_in=count_in,
                    input_device=input_device, input_channel=input_channel,
                    direct=direct,
                    **({} if midi is None else {"midi": midi,
                                                "midi_clock": midi_clock}))
    sd = _import_sounddevice()
    route = resolve_input(sd, device, input_device=input_device,
                          input_channel=input_channel, direct=direct)
    check_audio_consent(device)
    dev = route.device
    midi_port = _resolve_midi_port(midi)
    max_minutes = resolve_max_minutes(max_minutes)

    # Siblings are imported lazily so this module always imports headless.
    from .duet import DuetEngine  # noqa: PLC0415
    from .listen.beat import BeatTracker  # noqa: PLC0415
    from .listen.features import FeatureExtractor  # noqa: PLC0415
    from .listen.harmony import HarmonyTracker  # noqa: PLC0415
    from .mind import Mind  # noqa: PLC0415

    rec_path = None if record_dir is None else Path(record_dir)
    _refuse_a_kept_take(rec_path)
    recorder = None if rec_path is None else SessionRecorder(rec_path)

    extractor = FeatureExtractor(sr=SR)
    tkw: dict = {}
    if tightness is not None:
        tkw["tightness"] = tightness
    if input_mode in ("acoustic", "programmed"):
        tkw["acoustic"] = input_mode == "acoustic"
    if count_in:
        tkw["count_in"] = True
    tracker = BeatTracker(**tkw)
    harmony = HarmonyTracker()
    mind = Mind(params)
    from .governor import HarmonyGovernor  # noqa: PLC0415 — lazy, like siblings
    governor = HarmonyGovernor(style, params)
    duet = DuetEngine(style=style, seed=seed, sr=SR)

    in_ring = RingBuffer(SR * 8)                  # mic -> analysis (mono)
    machine_ring = RingBuffer(SR * 8)             # what we played (mono sum)
    timeline = TimelineBuffer(SR * 30, channels=2)

    stop = threading.Event()      # hard stop: threads drain and exit
    ending = threading.Event()    # graceful: no new bars; finish + tail

    state_lock = threading.Lock()
    shared = {
        "beat": None, "harmony": None,
        # per-frame (t_s, rms_db, onset) after self-masking, for bar summaries
        "events": [],
        "masked": 0,
        # v1.8 — absolute sample index up to which SOUNDING machine audio has
        # been scheduled.  The status line derives "is the machine sounding"
        # from this, which is the observability contract: the reported state
        # may never disagree with the rendered audio (live_status_state).
        "audible_until": 0,
    }
    masker = SelfMasker(enabled=route.source != "direct")
    midi_in = None
    if midi_port is not None:
        midi_in, extractor, masker = _midi_listening(midi_port, timeline, recorder,
                                                     follow_clock=midi_clock)
    # v1.8 (FIX 3) — the live machine-bus calibration (bus_calibration_step):
    # "db" is the smoothed applied gain (None until a human reference
    # exists), "base" the (solved_db, ref_db) pair of the last full solve,
    # "lin" the linear gain the last delivered bar ENDED at (so the next
    # bar's glide starts exactly where the previous one stopped), and
    # "prev_state" the last boundary plan state, which is what decides the
    # sim-parity resolve moments.
    cal = {"db": None, "base": None, "lin": 1.0, "prev_state": "waiting"}
    n_decisions = [0]                 # bars logged, for ending.json
    # v2.5 — placement compensation (see plume.calibrate).  Stream time is not
    # room time: a bar placed at the predicted downbeat sounds input+output
    # latency late in the room.  ``comp["samples"]`` is subtracted at both
    # placement sites; it is resolved once the stream is open (the reported
    # fallback needs it) and before the planner thread starts, so the planner
    # only ever sees the final value.
    comp = {"samples": 0}

    # v2.2 — discoverable and stoppable.  The sideman takes no cues, so its
    # socket is control-only on a private path (see control_socket_path); if
    # it cannot bind, the session still plays and 'plume stop' falls back to
    # signals.  Bookkeeping must never be able to take the band down.
    cues = None
    try:
        cues = CueServer(socket_path or control_socket_path(), kind="jam",
                         accepts_cues=False).start()
    except (RealtimeError, OSError):
        cues = None
    started_mono = time.monotonic()
    register_session(
        kind="jam", style=style, role="follow",
        socket_path=(cues.path if cues else None),
        max_minutes=max_minutes, record_dir=record_dir)
    # v1.3 — the take starts being written NOW, not at the end: the manifest
    # before the first bar, then every decision and every sample as it
    # happens.  An observer can tail the session while it plays, and a session
    # that has to be killed keeps everything up to the last flush.
    if recorder is not None:
        recorder.open(session_manifest(
            kind="jam", style=style, role="follow", seed=seed,
            socket_path=(cues.path if cues else None),
            max_minutes=max_minutes, record_dir=record_dir, device=device,
            params=params, started_at=time.time(), input_route=route,
            midi_port=midi_port))
    # Named stop_deadline, not deadline: planner_loop already has a local
    # `deadline` (the bar-revision cutoff) and the two must never be confused.
    stop_deadline = session_deadline(started_mono, max_minutes)

    def callback(indata, outdata, frames, time_info, status):  # noqa: ARG001
        # Ring copies only: no locks, no waiting, nothing outside the rings.  The
        # slices and the sum below do allocate a little; it is bounded and small.
        in_ring.write(indata[:, route.guitar])
        timeline.read_into(outdata)
        machine_ring.write(0.5 * (outdata[:, 0] + outdata[:, 1]))
        if monitor:
            outdata[:, 0] += indata[:, route.guitar]
            outdata[:, 1] += indata[:, route.guitar]

    hop_rms = HopRms()

    def _log(doc: dict) -> None:
        """One decision -> the live log, flushed (v1.3)."""
        n_decisions[0] += 1
        if recorder is not None:
            recorder.decision(doc)

    def analysis_loop():
        # v1.3: bounded on BOTH paths.  The idle path waits on the stop Event
        # (so a stop is noticed in microseconds, not on the next 5 ms tick),
        # and the busy path — which used to be a `while True` that only looked
        # at `stop` when the rings happened to run dry — now drains for a
        # bounded time after a stop and then leaves regardless.
        drain_until = None
        while True:
            if stop.is_set() and drain_until is None:
                drain_until = time.monotonic() + ANALYSIS_DRAIN_S
            if drain_until is not None and time.monotonic() >= drain_until:
                break
            n = min(in_ring.available, machine_ring.available)
            if n == 0:
                if stop.is_set():
                    break
                stop.wait(0.005)
                continue
            x = in_ring.read(n)
            m = machine_ring.read(n)
            if recorder is not None:
                recorder.audio(x, m)
            masked_frames, _flags, beat_state, harm_state = analysis_step(
                extractor=extractor, hop_rms=hop_rms, masker=masker,
                tracker=tracker, harmony=harmony,
                human_block=x, machine_block=m)
            with state_lock:
                shared["beat"] = beat_state
                shared["harmony"] = harm_state
                shared["masked"] = masker.masked_frames
                for f in masked_frames:
                    shared["events"].append((f.t_s, f.rms_db, f.onset))

    def _plan_and_place(beat_state, harm_state, human, ref_db):
        # v2.6: the governed harmony is what BOTH the mind and the renderer
        # believe — commitment must never disagree with itself mid-bar.
        harm_state = governor.govern(getattr(beat_state, "bar", None),
                                     harm_state)
        """Plan + render the upcoming bar ONCE and place it on the timeline.

        Returns ``(at_sample, n_samples, audio)`` — the audio is kept so a
        revision can MOVE the bar rather than make a second one (see
        ``_move_bar``).

        v1.8 (FIX 3): the rendered bar is calibrated against the human before
        it is placed — ``cal`` glides to put the bar's RMS at
        ``plan.gain_db`` relative to ``ref_db`` (the tracked human level) —
        and the applied gain is logged with the decision, because the
        calibration is the single number that decides how loud the machine is
        against the player and it previously left no trace anywhere.
        """
        plan = mind.plan_bar(beat_state, harm_state, human)
        doc = {
            "bar": plan.bar, "t_s": round(beat_state.t_s, 4),
            "state": plan.state, "reason": plan.reason,
            "bpm": round(beat_state.bpm, 2),
            "conf": round(beat_state.confidence, 3),
            "key": harm_state.key if harm_state else None,
            "chord": harm_state.chord if harm_state else None,
            "roles": plan.roles, "gain_db": plan.gain_db,
        }
        if math.isinf(beat_state.next_downbeat_t_s):
            doc["cal_db"] = None
            _log(doc)
            return None
        audio = duet.render_bar(plan, beat_state, harm_state)
        # Full solves happen at sim's recalibration moments — a fresh entry,
        # and the entering->comping step up to the full texture — plus the
        # first sounding bar a reference exists for; every other bar only
        # tracks the HUMAN's level (see bus_calibration_step for why).
        st = str(plan.state)
        sounding_bar = st in _SOUNDING_STATES
        resolve = sounding_bar and (
            (st == "entering" and cal["prev_state"] != "entering")
            or (st == "comping" and cal["prev_state"] == "entering")
            or cal["base"] is None)
        cal["prev_state"] = st
        cal["db"], cal["base"] = bus_calibration_step(
            cal["db"], cal["base"], duet.last_bar_ms_pre_env, ref_db,
            plan.gain_db, resolve=resolve)
        new_lin = 1.0 if cal["db"] is None else 10.0 ** (cal["db"] / 20.0)
        audio = apply_calibration(audio, cal["lin"], new_lin)
        cal["lin"] = new_lin
        doc["cal_db"] = None if cal["db"] is None else round(cal["db"], 2)
        _log(doc)
        # v2.5: early by the acoustic round trip, so the bar sounds in the
        # ROOM on the human's downbeat, not a round trip after it.
        at = int(round(beat_state.next_downbeat_t_s * SR)) - comp["samples"]
        timeline.write_at(at, audio)
        if sounding_bar:
            with state_lock:
                shared["audible_until"] = max(shared["audible_until"],
                                              at + audio.shape[0])
        return (at, audio.shape[0], audio)

    def _move_bar(placed, at):
        """Re-place the bar already rendered for this index at a new sample.

        **Why this is not another ``_plan_and_place``.**  The revision window
        exists because the *predicted downbeat* moves, not because the bar
        should be reconsidered — and re-rendering it was not free.
        ``DuetEngine.render_bar`` is a stateful stream: it advances
        ``_frac`` (the fractional-sample carry that keeps placement from
        drifting), consumes ``_tail`` (the previous bar's ring-out, which is
        mixed into this one) and rolls ``_prev_voicing`` (what voice-leading
        is measured from).  Calling it twice for one bar index advanced all
        three twice — the carry double-stepped, the previous bar's tail was
        spent into a buffer that was then thrown away, and the next bar
        voice-led from a chord that never sounded.  So a revision moves the
        samples we already have, which is exactly what "the downbeat moved by
        3 ms" means, and is idempotent per bar index however many times the
        prediction wobbles inside one window.
        """
        at_old, n, audio = placed
        timeline.clear_region(at_old, n)
        timeline.write_at(at, audio)
        return (at, n, audio)

    def planner_loop():
        planned_bar = None          # bar index the current placement is for
        # v1.7 — the highest bar index ever handed to DuetEngine.  ``placed``
        # and ``planned_bar`` are both cleared when the tracker unlocks, so
        # they cannot answer "did we already render this one?" across a lock
        # LOSS — and that is a second way into the same P2 defect the
        # revision window had.  BeatTracker's bar counter never goes
        # backwards; it FREEZES while unlocked ("bar counter freezes, phase
        # free-runs") and resumes where it stopped.  So after a dip below
        # UNLOCK_CONF and a re-lock, ``upcoming`` names a bar that has already
        # sounded, and re-rendering it double-advances DuetEngine's ``_frac``,
        # spends the previous bar's ``_tail`` twice and voice-leads from a
        # chord that never played — exactly the harm ``_move_bar`` exists to
        # prevent, arriving by the other door.  Because the counter only ever
        # rises, requiring it to pass what we already rendered costs at most
        # one bar of silence after a re-lock and cannot deadlock.
        placed = None               # (at_sample, n_samples) of that placement
        last_rendered_bar = -1
        last_unlocked_plan = -1e9
        human_ref_db = None         # EMA of non-silent human bar RMS (v1.2)
        # ``ending`` is checked here and not only on the locked path below: a
        # take that never locks must still reach the flush while the stream is
        # open, or the tail goes to a closed stream after a full join timeout.
        while not stop.is_set() and not ending.is_set():
            with state_lock:
                beat_state = shared["beat"]
                harm_state = shared["harmony"]
            if beat_state is None:
                stop.wait(0.02)
                continue
            now = timeline.playhead / SR
            if not beat_state.locked:
                if not ending.is_set() and beat_state.t_s - last_unlocked_plan >= 1.0:
                    last_unlocked_plan = beat_state.t_s
                    # v2.6: unlocked ticks hold the governed chord (bar=None
                    # never advances the progression or the streaks).
                    harm_state = governor.govern(None, harm_state)
                    plan = mind.plan_bar(beat_state, harm_state, None)
                    # v1.8 — an unlocked tick can carry a SOUNDING plan (the
                    # mind's held wobble bars and the beat-loss fade are
                    # issued here, not at a boundary), and this path used to
                    # log the plan, set planned_bar — and never render a
                    # sample.  That is the live half of the reported
                    # symptom: the status said "listening" for six minutes
                    # while the human heard intermittent playing, and the
                    # decision log said "fading" over bars that never
                    # sounded.  sim has rendered these since v1.2
                    # (non-overlapping, at the tick cadence,
                    # ``continuation=True`` so a held bar continues the
                    # entrance instead of restarting it); the live path now
                    # does exactly the same.  ``bpm > 0`` guards the frozen
                    # grid: a sounding plan can only exist after a lock, so
                    # the frozen estimate is always real on this path.
                    sounded = 0
                    if (str(plan.state) in _SOUNDING_STATES
                            and beat_state.bpm > 0.0):
                        audio = duet.render_bar(plan, beat_state, harm_state,
                                                continuation=True)
                        audio = apply_calibration(audio, cal["lin"],
                                                  cal["lin"])
                        if audio.shape[0]:
                            at = max(timeline.playhead, timeline.end)
                            sounded = timeline.write_at(at, audio)
                            if sounded:
                                with state_lock:
                                    shared["audible_until"] = max(
                                        shared["audible_until"],
                                        at + audio.shape[0])
                    _log({
                        "bar": plan.bar, "t_s": round(beat_state.t_s, 4),
                        "state": plan.state, "reason": plan.reason,
                        "bpm": round(beat_state.bpm, 2),
                        "conf": round(beat_state.confidence, 3),
                        "key": harm_state.key if harm_state else None,
                        "chord": harm_state.chord if harm_state else None,
                        "roles": plan.roles, "gain_db": plan.gain_db,
                        # The observability contract, in the log as well as
                        # the status line: whether this plan actually reached
                        # the schedule.
                        "sounding": bool(sounded),
                        "cal_db": (None if cal["db"] is None
                                   else round(cal["db"], 2)),
                    })
                planned_bar, placed = None, None
                # v1.7 — bounded while UNLOCKED, which is the case that
                # actually happened.  ``shared["events"]`` is trimmed on a bar
                # boundary, and a session that never locks has no bar
                # boundaries: Juniper's six-minute take never locked once, so
                # the list grew for the whole session (~34k tuples then, ~225k
                # for a forty-minute one) — the same accumulate-forever shape
                # SessionRecorder was written to remove, left behind in the
                # planner.  Nothing older than EVENT_KEEP_S can ever be asked
                # for: the furthest back a bar summary reaches is two bars,
                # and two bars at 16 bpm is 30 s.  The length guard keeps this
                # off the 20 ms path in the ordinary case.
                if len(shared["events"]) > EVENT_TRIM_AT:
                    with state_lock:
                        shared["events"] = trim_events(shared["events"],
                                                       beat_state.t_s)
                stop.wait(0.02)
                continue
            upcoming = beat_state.bar + 1
            deadline = revision_deadline(beat_state.next_downbeat_t_s,
                                         beat_state.beat_interval_s)
            if ending.is_set():
                break
            if planned_bar != upcoming and upcoming <= last_rendered_bar:
                # We came back from a lock loss onto a bar we have already
                # played.  Do not render it twice (see last_rendered_bar);
                # wait for the counter to move past it, which it will.
                planned_bar, placed = None, None
                stop.wait(0.02)
                continue
            if planned_bar != upcoming:
                # A new bar boundary passed: summarize the completed human bar
                # and render the upcoming one (render-ahead >= 1 bar).
                bar_len = 4.0 * beat_state.beat_interval_s
                t1 = beat_state.next_downbeat_t_s - bar_len
                t0 = t1 - bar_len
                with state_lock:
                    events = [e for e in shared["events"] if t0 <= e[0] < t1]
                    shared["events"] = [e for e in shared["events"] if e[0] >= t1]
                human = None
                if beat_state.bar >= 0:
                    kwargs = bar_features_dict(
                        beat_state.bar, events,
                        silence_rms_db=params.silence_rms_db,
                        ref_db=human_ref_db,
                        register_low_midi=(harm_state.register_low_midi
                                           if harm_state else None),
                        register_high_midi=(harm_state.register_high_midi
                                            if harm_state else None))
                    from .mind import BarFeatures  # noqa: PLC0415
                    human = BarFeatures(**kwargs)
                    if not human.silent:      # EMA of the human's own level
                        human_ref_db = (human.rms_db if human_ref_db is None
                                        else 0.5 * human.rms_db
                                        + 0.5 * human_ref_db)
                placed = _plan_and_place(beat_state, harm_state, human,
                                         human_ref_db)
                planned_bar = upcoming
                if placed is not None:
                    # Only a bar that actually reached DuetEngine counts --
                    # _plan_and_place returns None (without rendering) when
                    # there is no predicted downbeat to place it against.
                    last_rendered_bar = max(last_rendered_bar, upcoming)
            elif placed is not None and now < deadline:
                # Revisable window: MOVE the bar if the predicted downbeat
                # moved.  Never re-plan or re-render it — see _move_bar.
                at = (int(round(beat_state.next_downbeat_t_s * SR))
                      - comp["samples"])   # v2.5: mirror _plan_and_place
                if abs(at - placed[0]) > SR // 500:      # > 2 ms drift
                    placed = _move_bar(placed, at)
            stop.wait(0.02)
        # Graceful ending: let whatever is placed finish, then flush the tail.
        # The tail rings out at the gain the last bar ended at (parity with
        # sim, which scales its tail by cal_lin).
        tail = duet.flush_tail()
        if tail.shape[0]:
            timeline.write_at(timeline.end,
                              apply_calibration(tail, cal["lin"], cal["lin"]))

    stream = sd.Stream(
        samplerate=SR, blocksize=BLOCK, device=dev, channels=(route.channels, 2),
        dtype="float32", latency="low", callback=callback,
    )
    # v1.6: the mask has to line up with when the sound came BACK, not when it
    # was handed to PortAudio.  Only the open stream knows the round trip.
    masker.set_latency_hops(stream_latency_hops(stream))
    if midi_in is not None:
        midi_in.offset_s = _stream_input_latency_s(stream)
        midi_in.start()
    # v2.5: resolve the placement compensation now that the stream can report
    # its latency (the planner has not started; it sees only the final value).
    from .calibrate import (  # noqa: PLC0415 — lazy, like the other siblings
        device_names, load_roundtrip_ms, placement_comp_samples)
    _lat = getattr(stream, "latency", None)
    try:
        reported_s = (float(_lat[0]) + float(_lat[1])
                      if isinstance(_lat, (tuple, list)) and len(_lat) == 2
                      else None)
    except (TypeError, ValueError):
        reported_s = None
    _in_name, _out_name = device_names(sd, dev)
    stored_ms = load_roundtrip_ms(_in_name, _out_name, SR)
    comp["samples"] = placement_comp_samples(latency_comp_s, stored_ms,
                                             reported_s, SR)
    comp_src = ("explicit" if latency_comp_s is not None
                else "calibrated" if stored_ms is not None
                else "reported" if reported_s is not None
                and math.isfinite(reported_s) else "none")
    print(f"latency comp: {comp['samples'] * 1000 // SR} ms ({comp_src} — "
          f"'plume calibrate' measures your room)", flush=True)

    analysis = threading.Thread(target=analysis_loop, name="plume-analysis",
                                daemon=True)
    planner = threading.Thread(target=planner_loop, name="plume-planner",
                               daemon=True)

    print(f"jam: style={style} seed={seed} — {_ending_banner(cues, max_minutes)}",
          flush=True)
    status_line = StatusLine()
    why = END_INTERRUPT
    hard = HardExit(on_exit=_hard_exit_cleanup(
        recorder, cues, lambda: _ending_doc(END_WEDGED,
                                            started_mono=started_mono,
                                            max_minutes=max_minutes,
                                            bars=n_decisions[0])))
    try:
        with GracefulSignals(on_repeat=lambda _s: hard.arm(
                END_INTERRUPT, seconds=HARD_EXIT_GRACE_S)) as signals, stream:
            analysis.start()
            planner.start()
            try:
                while True:
                    reason = _live_stop_reason(cues, stop_deadline,
                                               signals=signals)
                    if reason is not None:
                        why = reason
                        break
                    with state_lock:
                        b = shared["beat"]
                        h = shared["harmony"]
                        audible_until = shared["audible_until"]
                    if b is not None:
                        # v1.8 — the observability contract: the state word
                        # is derived from what is actually on the schedule,
                        # so the readout can never say "listening" over a
                        # machine that is audibly playing (the postmortem's
                        # six minutes of exactly that).
                        sounding = timeline.playhead < audible_until
                        state_name = live_status_state(b.locked, b.bar,
                                                       sounding)
                        key = (h.key or "?") if h is not None else "?"
                        status_line.update(
                            f"[{state_name}] {b.bpm:6.1f} bpm  "
                            f"conf {b.confidence:4.2f}  {key:<10}")
                    time.sleep(0.25)
            except KeyboardInterrupt:
                why = END_INTERRUPT
            # v1.3: from here on the ending is on a clock.  Every step below is
            # individually bounded AND the whole thing is; if it still does not
            # finish, the watchdog ends the process with the take already on
            # disk.  See HardExit for the measured reason this is not optional.
            hard.arm(why)
            status_line.close()
            print(f"ending — {why}; finishing the bar...", flush=True)
            # v2.2: this drain is INSIDE `with stream` now.  It used to sit in
            # the KeyboardInterrupt handler, i.e. after the stream had already
            # closed — so the playhead never moved and the "graceful" flush was
            # a 30-second nap that played nothing.  lead() had already been
            # fixed; jam had not, and a graceful `plume stop` that drops the
            # tail is not graceful.
            ending.set()
            planner.join(timeout=PLANNER_JOIN_S)
            _drain_timeline(timeline)
            # The tail has played; take PortAudio's blocking stop out of the
            # picture before `with stream` reaches for it.
            _abort_stream(stream)
    finally:
        stop.set()
        ending.set()
        if planner.is_alive():
            planner.join(timeout=PLANNER_JOIN_S)
        if analysis.is_alive():
            analysis.join(timeout=ANALYSIS_JOIN_S)
        if midi_in is not None:
            midi_in.close()
        if cues is not None:
            cues.close()
        status_line.close()
        if recorder is not None:
            recorder.finish(_ending_doc(why, started_mono=started_mono,
                                        max_minutes=max_minutes,
                                        bars=n_decisions[0]))
        unregister_session()
        hard.disarm()

    if recorder is not None:
        print(f"recorded to {rec_path}", flush=True)
        if recorder.errors:
            print(f"recording had {len(recorder.errors)} problem(s): "
                  f"{recorder.errors[0]}", flush=True)


# v1.3 removed ``_write_records``: there is no longer a moment when a take is
# "written", because it is written continuously (see SessionRecorder).  Its
# buffer-everything-then-dump shape was the P0-A defect itself — a six-minute
# session left `--record DIR` empty until it exited, so there was nothing to
# watch while it ran and nothing at all if it had to be killed.


# =====================================================================
# v2 — the machine's own clock, the cue socket, and the lead transport
# (SPEC-V2 sections 3 and 5).
#
# Everything below this line is ADDITIVE.  ``jam``'s v1 body above is
# untouched apart from the dispatch guard at its top, which cannot fire
# for a v1 invocation (no role, no lead-in, no session).
# =====================================================================

#: Env var that overrides the live session's cue socket path.
SOCKET_ENV = "PLUME_SOCKET"

#: AF_UNIX paths are length-limited by the kernel (104 bytes on macOS, 108 on
#: Linux).  A pytest tmp_path can exceed it, and the failure mode is an
#: unhelpful OSError, so the length is checked with a message that says so.
_SOCKET_PATH_MAX = 100

#: Bytes accepted for one cue request.  A cue is a short line; anything
#: larger is a client bug and is refused rather than buffered (the socket is
#: attached to a live musical instrument, not an inbox).
_CUE_MAX_BYTES = 8192


def default_socket_path() -> str:
    """Where ``plume cue`` and a live session meet.

    ``$PLUME_SOCKET`` if set, else ``<tmpdir>/plume-live.sock``.  One path per
    machine: a second live session refuses to bind rather than stealing the
    first one's cues (see :meth:`CueServer.start`).
    """
    env = os.environ.get(SOCKET_ENV)
    if env:
        return str(env)
    return str(Path(tempfile.gettempdir()) / "plume-live.sock")


def _check_socket_path(path: str) -> str:
    p = str(path)
    if not hasattr(socket, "AF_UNIX"):      # pragma: no cover - not macOS/Linux
        raise RealtimeError(
            "cue sockets need AF_UNIX, which this platform does not have")
    if len(p.encode("utf-8")) > _SOCKET_PATH_MAX:
        raise RealtimeError(
            f"socket path is too long ({len(p)} bytes; the kernel limit is "
            f"about {_SOCKET_PATH_MAX}): {p}")
    return p


#: Who may be named as a cue's sender.  Anything else reads as "unknown".
CUE_SOURCES = ("gui", "cli", "voice")


class CueServer:
    """AF_UNIX server that queues cues for a running session (SPEC-V2 §3).

    Protocol: one JSON object per connection, newline-terminated, in and out.

        -> {"text": "go to the chorus", "at": "next-phrase", "ttl_bars": 16}
        <- {"ok": true, "queued": 1, "session": "lead"}

    The accept loop is a daemon thread that only appends to a list under a
    lock — the planner drains it at bar boundaries with :meth:`drain`.  A cue
    can therefore never block or raise into the audio path: the worst a
    malformed cue does is come back ``{"ok": false, "error": ...}`` to its own
    sender.

    **v2.2 — the same socket carries control verbs** (SPEC-V2 section 5
    amendment).  ``plume stop`` and ``plume sessions`` needed a way into a
    running session from another terminal, and this is already that way in:

        -> {"op": "stop"}      <- {"ok": true, "stopping": true, "pid": 4212}
        -> {"op": "status"}    <- {"ok": true, "pid": 4212, "bar": 37, ...}

    A verb never touches the cue queue and never blocks: ``stop`` sets
    :attr:`stopped`, which the transport's own status loop notices within a
    quarter second and turns into its normal graceful ending (finish the bar,
    flush the tail, write the records).  ``status`` is read-only and is how a
    lister proves a registry entry is a live plume session rather than a
    recycled pid.

    ``accepts_cues=False`` is the v1 sideman: it gets a control socket so it
    can be stopped, but a cue still comes back refused, because the sideman
    has no form cursor for a cue to move.  ``plume cue``'s contract at the
    default cue path is therefore unchanged.
    """

    def __init__(self, path: str | None = None, *, kind: str = "lead",
                 accepts_cues: bool = True):
        self.path = _check_socket_path(path or default_socket_path())
        self.kind = str(kind)
        self.accepts_cues = bool(accepts_cues)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._queue: list[dict] = []
        self._received = 0
        #: set by the owner each bar, so a sender learns where its cue landed
        self.bar: int | None = None
        #: set when a ``{"op": "stop"}`` arrives; the transport polls it
        self.stopped = threading.Event()
        #: extra fields a transport wants ``{"op": "status"}`` to report
        self.info: dict = {}
        #: called as ``(item, source)`` for each cue a session acted on or answered;
        #: see ``_enqueue``.  An item the looper answered also has a ``loop`` field.
        self.on_received = None
        #: a transport may answer some cues itself, at once, rather than at the
        #: next bar: ``text -> reply dict``, or None to queue it as usual.  The
        #: looper needs this because a take must start when it is asked for, and
        #: a tap-start session has no bars to drain the queue on until it has a
        #: tempo, which the take itself may be about to set.
        self.intercept = None

    # -- lifecycle ------------------------------------------------------

    def start(self) -> "CueServer":
        """Bind, listen, and start accepting.  Returns self.

        A leftover socket file from a crashed session is cleaned up, but only
        after proving nobody is listening on it: an unconditional unlink would
        let a second session silently hijack the first one's cues.
        """
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.bind(self.path)
        except OSError:
            if _socket_is_live(self.path):
                s.close()
                raise RealtimeError(
                    f"another plume session is already listening at "
                    f"{self.path} — end it, or set {SOCKET_ENV} to a "
                    f"different path") from None
            try:
                os.unlink(self.path)
            except OSError:
                pass
            try:
                s.bind(self.path)
            except OSError as e:
                s.close()
                raise RealtimeError(
                    f"cannot open the cue socket at {self.path}: {e}") from None
        # Before it listens.  Linux /tmp is shared, and this socket can stop a take.
        try:
            os.chmod(self.path, 0o600)
        except OSError as e:
            s.close()
            raise RealtimeError(
                f"cannot lock down the cue socket at {self.path}: {e}") from None
        s.listen(8)
        s.settimeout(0.2)
        self._sock = s
        self._thread = threading.Thread(target=self._serve, name="plume-cues",
                                        daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=CUE_CLOSE_S)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def __enter__(self) -> "CueServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- queue ----------------------------------------------------------

    @property
    def received(self) -> int:
        """Total cues accepted since start (monotone; survives draining)."""
        with self._lock:
            return self._received

    def drain(self) -> list[dict]:
        """Take everything queued since the last call, oldest first."""
        with self._lock:
            out, self._queue = self._queue, []
        return out

    # -- request dispatch ------------------------------------------------

    def handle(self, req: dict) -> dict:
        """Route one request: a control verb, or a cue.

        Control verbs are checked first and are the only requests that can
        arrive without ``text``; everything else is a cue and keeps the v2.0
        replies byte for byte.
        """
        op = req.get("op")
        if op is not None:
            return self._control(str(op))
        return self._enqueue(req)

    def _control(self, op: str) -> dict:
        if op == "stop":
            self.stopped.set()
            return {"ok": True, "stopping": True, "session": self.kind,
                    "pid": os.getpid(), "bar": self.bar}
        if op == "status":
            doc = {"ok": True, "session": self.kind, "pid": os.getpid(),
                   "bar": self.bar, "accepts_cues": self.accepts_cues,
                   "received": self.received}
            doc.update(self.info)
            return doc
        return {"ok": False, "error": f"unknown op {op!r} (stop, status)"}

    def _enqueue(self, req: dict) -> dict:
        if not self.accepts_cues:
            return {"ok": False,
                    "error": f"the {self.kind} session takes no cues — the v1 "
                             "sideman has no form cursor for a cue to move. "
                             "Use 'plume lead' (or 'plume jam --role lead')."}
        text = req.get("text")
        if not isinstance(text, str) or not text.strip():
            return {"ok": False, "error": "cue needs a non-empty 'text'"}
        if self.intercept is not None:
            reply = self.intercept(text)
            if reply is not None:
                # Never queued, so no row will name it.  Who asked exists only here,
                # refusals included: the sender saw the answer and the take did not.
                self._tell_log({"text": text, "at": "now", "ttl_bars": 0,
                                "loop": {"op": reply.get("op"), "ok": reply.get("ok") is True}},
                               req)
                return {**reply, "queued": 0, "session": self.kind, "bar": self.bar}
        at = req.get("at", "next-phrase")
        ttl = req.get("ttl_bars", 16)
        try:
            ttl = int(ttl)
        except (TypeError, ValueError):
            return {"ok": False,
                    "error": f"ttl_bars must be an integer, got {ttl!r}"}
        item = {"text": text, "at": str(at), "ttl_bars": ttl}
        with self._lock:
            self._queue.append(item)
            self._received += 1
            n = len(self._queue)
        self._tell_log(item, req)
        return {"ok": True, "queued": n, "session": self.kind, "bar": self.bar}

    def _tell_log(self, item: dict, req: dict) -> None:
        source = req.get("source")
        if not isinstance(source, str) or source not in CUE_SOURCES:
            source = "unknown"
        hook = self.on_received
        if hook is not None:
            try:
                hook(item, source)
            except Exception:      # noqa: BLE001 — a log must not cost the cue
                pass

    # -- accept loop ----------------------------------------------------

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                break
            with conn:
                conn.settimeout(1.0)
                try:
                    resp = self.handle(_read_json_line(conn))
                except Exception as e:      # noqa: BLE001 — never take the band down
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                try:
                    conn.sendall((json.dumps(resp, sort_keys=True) + "\n").encode())
                except OSError:
                    pass


def _read_json_line(conn: socket.socket) -> dict:
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
        if len(buf) > _CUE_MAX_BYTES:
            raise ValueError("cue is too long")
    doc = json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("cue must be a JSON object")
    return doc


def _socket_is_live(path: str) -> bool:
    """True when something is accepting connections on ``path``."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(0.25)
    try:
        s.connect(path)
    except OSError:
        return False
    finally:
        s.close()
    return True


def _socket_request(path: str, req: dict, *, timeout: float,
                    absent_msg: str) -> dict:
    """One request/reply round trip on a session socket.

    Shared by :func:`send_cue` and the control verbs so there is exactly one
    place that knows the wire format, the timeouts, and what "nothing is
    listening" looks like.
    """
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(float(timeout))
    try:
        try:
            s.connect(path)
        except (FileNotFoundError, ConnectionRefusedError):
            raise RealtimeError(absent_msg) from None
        except OSError as e:
            raise RealtimeError(f"cannot reach {path}: {e}") from None
        s.sendall((json.dumps(req, sort_keys=True) + "\n").encode("utf-8"))
        try:
            return _read_json_line(s)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            raise RealtimeError(
                f"no reply from the session at {path}: {e}") from None
    finally:
        s.close()


def send_cue(text: str, *, at: str = "next-phrase", ttl_bars: int = 16,
             socket_path: str | None = None, timeout: float = 2.0,
             source: str | None = None) -> dict:
    """Send one cue to a running session; return its reply.

    ``source`` says who is asking, for the take's cue log (see
    :data:`CUE_SOURCES`).  Left out, the log says "unknown".

    Raises :class:`RealtimeError` with a message a player can act on when
    nothing is listening — the common case, which must not look like a crash.
    """
    path = _check_socket_path(socket_path or default_socket_path())
    req = {"text": str(text), "at": str(at), "ttl_bars": int(ttl_bars)}
    if source is not None:
        req["source"] = str(source)
    resp = _socket_request(
        path, req,
        timeout=timeout,
        absent_msg=(f"no running plume session at {path} — start one with "
                    f"'plume lead' or 'plume jam --role lead' (or point "
                    f"{SOCKET_ENV} at the session's socket)"))
    if not resp.get("ok"):
        raise RealtimeError(str(resp.get("error", "the session refused the cue")))
    return resp


def send_stop(*, socket_path: str, timeout: float = 2.0) -> dict:
    """Ask the session listening at ``socket_path`` to end itself gracefully.

    The session finishes the bar it is playing, flushes the tail and writes
    its records — the Ctrl-C path, reached from another terminal.
    """
    path = _check_socket_path(socket_path)
    resp = _socket_request(
        path, {"op": "stop"}, timeout=timeout,
        absent_msg=f"nothing is listening at {path}")
    if not resp.get("ok"):
        raise RealtimeError(str(resp.get("error", "the session refused to stop")))
    return resp


def send_status(*, socket_path: str, timeout: float = 1.0) -> dict:
    """Read-only liveness probe: proves a registry entry is really plume."""
    path = _check_socket_path(socket_path)
    resp = _socket_request(
        path, {"op": "status"}, timeout=timeout,
        absent_msg=f"nothing is listening at {path}")
    if not resp.get("ok"):
        raise RealtimeError(str(resp.get("error", "no status")))
    return resp


# =====================================================================
# v2.2 — the session registry: how a running session is FOUND
# (SPEC-V2 section 5 amendment).
#
# The consent guard stops a session from starting unattended.  This stops an
# attended one from becoming unattended: a live session writes one small JSON
# file naming itself, its pid and its socket, so 'plume sessions' can say what
# is making the noise and 'plume stop' can end it from any terminal — with no
# hunting for a PID, which is exactly what an unattended session cost.
# =====================================================================

#: Env var overriding where session records live (tests, and multi-user boxes).
SESSION_DIR_ENV = "PLUME_SESSION_DIR"

#: Signals, in escalation order, once the socket has not worked.
_STOP_SIGNALS = ("SIGINT", "SIGTERM")


def session_dir() -> Path:
    """Directory holding one ``<pid>.json`` per live session."""
    env = os.environ.get(SESSION_DIR_ENV)
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "plume-sessions"


def control_socket_path(pid: int | None = None) -> str:
    """Per-process control socket, for transports that own no cue socket.

    Lead mode's stop verb rides its cue socket (one shared path per machine,
    where ``plume cue`` expects to find it).  The v1 sideman takes no cues, so
    it gets its own private path here instead of squatting on the cue socket —
    that keeps ``plume cue``'s "no running session" answer honest while still
    giving ``plume stop`` a graceful way in.
    """
    p = int(os.getpid() if pid is None else pid)
    return str(session_dir() / f"{p}.sock")


def _pid_alive(pid: int) -> bool:
    """True when a process with this pid exists and we may signal it."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:      # exists, owned by someone else
        return True
    except (OSError, ValueError, TypeError):
        return False
    return True


@dataclasses.dataclass(frozen=True)
class LiveSession:
    """One running live session, as recorded on disk.

    Times are epoch seconds — wall clock, deliberately: this is bookkeeping
    about a process, not musical logic, and it has to mean the same thing to a
    different process in a different terminal.  Nothing in here is read by the
    planner, the clock or the renderer.
    """

    pid: int
    kind: str                       # "jam" | "lead"
    style: str = ""
    role: str = ""
    session_name: str | None = None
    socket: str | None = None
    started_at: float = 0.0         # epoch seconds
    max_minutes: float = DEFAULT_MAX_MINUTES
    record_dir: str | None = None

    # -- serialization --------------------------------------------------

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, doc: dict) -> "LiveSession":
        if not isinstance(doc, dict):
            raise ValueError(f"a session record must be a JSON object, "
                             f"got {type(doc).__name__}")
        fields = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in doc.items() if k in fields}
        kw["pid"] = int(kw.get("pid", 0))
        return cls(**kw)

    # -- derived --------------------------------------------------------

    @property
    def path(self) -> Path:
        return session_dir() / f"{self.pid}.json"

    def uptime_s(self, now: float) -> float:
        """Seconds since start, given the caller's clock reading."""
        return max(0.0, float(now) - float(self.started_at))

    def deadline(self) -> float | None:
        """Epoch instant the cap will end this session at, or None."""
        return session_deadline(self.started_at, self.max_minutes)

    def remaining_s(self, now: float) -> float | None:
        d = self.deadline()
        return None if d is None else d - float(now)

    @property
    def stop_command(self) -> str:
        """The exact command that ends this session."""
        return f"plume stop --pid {self.pid}"

    def alive(self) -> bool:
        return _pid_alive(self.pid)

    def describe(self, *, now: float) -> dict:
        return {
            "pid": self.pid,
            "kind": self.kind,
            "style": self.style,
            "role": self.role,
            "session": self.session_name,
            "socket": self.socket,
            "started_at": round(float(self.started_at), 3),
            "uptime_s": round(self.uptime_s(now), 1),
            "uptime": format_uptime(self.uptime_s(now)),
            "max_minutes": float(self.max_minutes),
            "remaining_s": (None if self.remaining_s(now) is None
                            else round(self.remaining_s(now), 1)),
            "record_dir": self.record_dir,
            "stop_command": self.stop_command,
        }


def format_uptime(seconds: float) -> str:
    """``12m``, ``1h04m``, ``38s`` — short enough for a status line."""
    s = max(0, int(round(float(seconds))))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def register_session(*, kind: str, style: str = "", role: str = "",
                     session_name: str | None = None,
                     socket_path: str | None = None,
                     max_minutes: float | None = None,
                     record_dir: str | None = None,
                     started_at: float | None = None,
                     pid: int | None = None) -> LiveSession | None:
    """Write this process's session record.  Returns it, or None on failure.

    Never raises: a session that cannot write its record still has to play.
    The cost of failing is that ``plume stop`` falls back to signals, which is
    strictly better than taking the band down over bookkeeping.
    """
    rec = LiveSession(
        pid=int(os.getpid() if pid is None else pid),
        kind=str(kind), style=str(style or ""), role=str(role or ""),
        session_name=session_name, socket=socket_path,
        started_at=float(time.time() if started_at is None else started_at),
        max_minutes=resolve_max_minutes(max_minutes),
        # Absolute, because other processes read this from their own directories.
        record_dir=None if record_dir is None else os.path.abspath(record_dir),
    )
    try:
        d = session_dir()
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        tmp = d / f"{rec.pid}.json.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            os.fchmod(f.fileno(), 0o600)
            json.dump(rec.to_dict(), f, sort_keys=True)
        os.replace(tmp, rec.path)      # atomic: a lister never sees a half file
    except OSError:
        return None
    return rec


def unregister_session(pid: int | None = None) -> None:
    """Remove this process's record.  Best effort, never raises."""
    p = int(os.getpid() if pid is None else pid)
    try:
        (session_dir() / f"{p}.json").unlink()
    except OSError:
        pass


def list_sessions(*, prune: bool = True) -> list[LiveSession]:
    """Every live session on this machine, oldest first.

    Records whose process is gone are pruned silently — a crashed session must
    not leave a ghost that ``plume stop`` then reports stopping.  Unreadable
    or malformed files are pruned the same way: the registry is a cache of
    running processes, not a document anyone edits.
    """
    d = session_dir()
    try:
        files = sorted(d.glob("*.json"))
    except OSError:
        return []
    out: list[LiveSession] = []
    for f in files:
        rec = None
        try:
            # O_NONBLOCK: a FIFO here must not hang whoever is asking what is playing.
            fd = os.open(f, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "r", encoding="utf-8") as fh:
                if stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                    rec = LiveSession.from_dict(json.load(fh))
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            # Broad on purpose: this is a cache of running processes, and a
            # file in it that cannot be read is a file to drop, never an
            # exception to raise at somebody asking "what is playing?".
            rec = None
        if rec is None or rec.pid <= 0 or not _pid_alive(rec.pid):
            if prune:
                try:
                    f.unlink()
                except OSError:
                    pass
            continue
        out.append(rec)
    out.sort(key=lambda r: (r.started_at, r.pid))
    return out


def find_session(pid: int) -> LiveSession | None:
    """The live session with this pid, or None."""
    for rec in list_sessions():
        if rec.pid == int(pid):
            return rec
    return None


#: How long to wait for a graceful ending before reaching for SIGTERM.
#: Deliberately longer than the transports' own graceful ending, which is
#: bounded at ~40 s (a 10 s planner join plus a 30 s tail drain).  It costs
#: nothing when things work — the wait ends the moment the process does — and
#: a shorter value only buys the chance to destroy a recording in progress.
DEFAULT_STOP_TIMEOUT = 45.0


def stop_session(rec: LiveSession, *, timeout: float = DEFAULT_STOP_TIMEOUT,
                 poll: float = 0.1, now: float | None = None,
                 force: bool = False) -> dict:
    """End one live session, gracefully if the socket answers.

    ``timeout`` is the budget for the GRACEFUL rungs (1 and 2 below) together,
    not per rung.  It is an upper bound, never a delay: every wait ends the
    instant the process does.

    Ladder, in order, stopping as soon as the process is gone:

    1. ``{"op": "stop"}`` on the session's own socket — the graceful path.
       The session finishes its bar, flushes the tail and writes its records,
       exactly as it does for Ctrl-C.  The handshake also *proves identity*:
       only a plume session answers this socket, so a recycled pid cannot be
       mistaken for a session here.
    2. ``SIGINT`` — the same graceful path, delivered by signal, for a session
       whose socket never opened (a path too long for AF_UNIX, say).  Live
       sessions install their own handler for it (:class:`GracefulSignals`),
       which is what makes this rung work at all for a session started in the
       background by a non-interactive shell — those inherit SIGINT as
       ``SIG_IGN``.
    3. ``SIGTERM`` — the fallback.  Also handled gracefully by plume's own
       sessions, so it is only destructive for one that is genuinely wedged.

    ``SIGKILL`` is deliberately NOT in the default ladder: it would cut the
    tail.  ``force=True`` adds it as a fourth rung for the case where a human
    has decided the process must go — and since v1.3 that costs far less than
    it used to, because the take is written as it happens rather than at the
    end (see :class:`SessionRecorder`).  Without ``force`` the returned dict
    still says plainly that the session is wedged and names the command.

    **v1.3 — why this used to report a wedge that no rung could clear.**  The
    ladder was not the problem; the session was unkillable by design, in two
    layers.  ``GracefulSignals`` had replaced SIGTERM's default disposition
    (applied by the kernel, needing no interpreter) with a Python-level
    handler, and a Python-level handler only runs in the main thread's eval
    loop — which was parked in ``Pa_StopStream``, a documented "wait until all
    pending audio buffers have been played" that never returned on the device
    in question.  Worse, the handler's own second-signal path *swallowed* the
    SIGTERM (restoring ``SIG_DFL`` for a third signal that a two-rung ladder
    never sends).  So: rung 1 was noticed and could not be acted on, rung 2
    was noticed and could not be acted on, rung 3 was silently eaten.  The
    honest report at the end was the only part working correctly.

    The fix is inside the session, not here: :class:`HardExit` gives the
    ending a deadline enforced by an ordinary daemon thread that calls
    ``os._exit``, which needs nothing from the main thread.  This ladder is
    unchanged apart from ``force``.

    **A session in THIS process is never signalled.**  ``plume stop`` is
    normally a different process from the session, but plume is also a
    library: an agent that embedded a session and then asked to stop it would,
    with a naive ladder, SIGTERM itself.  Same-pid records get the socket rung
    and nothing else, and say so.

    **Known limit, stated rather than hidden:** steps 2 and 3 trust the pid.
    If the socket is dead and the operating system has recycled that pid
    between the registry write and now, the signal goes to a stranger.  The
    window is small (records are pruned on every list) and the socket path
    avoids it entirely, but it is real.
    """
    now = time.time() if now is None else float(now)
    info = {
        "pid": rec.pid, "kind": rec.kind, "style": rec.style,
        "role": rec.role, "session": rec.session_name,
        "uptime_s": round(rec.uptime_s(now), 1),
        "uptime": format_uptime(rec.uptime_s(now)),
        "record_dir": rec.record_dir,
    }
    if not rec.alive():
        unregister_session(rec.pid)
        return {**info, "stopped": True, "how": "already-gone",
                "detail": "the process had already exited"}

    deadline = time.monotonic() + max(0.0, float(timeout))
    tried: list[str] = []

    def _wait(until: float) -> bool:
        while time.monotonic() < until:
            if not rec.alive():
                return True
            time.sleep(poll)
        return not rec.alive()

    same_process = rec.pid == os.getpid()

    if rec.socket:
        try:
            send_stop(socket_path=rec.socket, timeout=2.0)
            tried.append("socket")
            if same_process:
                # We are the session.  It has been asked to end and will, but
                # waiting for our own process to exit would deadlock.
                return {**info, "stopped": True, "how": "socket",
                        "tried": tried,
                        "detail": "asked over its own socket; it is finishing "
                                  "the bar in this process"}
            if _wait(deadline):
                unregister_session(rec.pid)
                return {**info, "stopped": True, "how": "socket", "tried": tried,
                        "detail": "asked over its own socket; it finished the "
                                  "bar and flushed the tail"
                                  + (" and wrote its records" if rec.record_dir else "")}
        except RealtimeError as e:
            tried.append(f"socket-failed: {e}")

    if same_process:
        return {**info, "stopped": False, "how": "self", "tried": tried,
                "detail": (f"pid {rec.pid} is this very process and its socket "
                           f"did not answer — end it from another terminal, or "
                           f"with Ctrl-C")}

    import signal as _signal  # noqa: PLC0415 — POSIX-only, imported on use

    # SIGINT shares the graceful budget with the socket rung rather than
    # getting some fraction of it.  It IS the graceful path — the session
    # catches it and runs the same finish-the-bar/flush/write ending — and
    # that ending is internally bounded at about 40 s (a 10 s planner join
    # plus a 30 s drain).  A short fuse here does not fail safe: it lands
    # SIGTERM in the middle of a working ending and throws the recording
    # away, which is exactly what a cross-process test caught it doing.
    # SIGTERM, the genuinely destructive rung, gets a brief grace of its own.
    # A session with no ending to protect only gets that grace, since its sound
    # keeps going for as long as this waits.
    grace = min(5.0, max(0.5, float(timeout)))
    finishing = rec.kind not in _NOTHING_TO_FINISH
    if not finishing:
        deadline = time.monotonic() + grace
    waited = timeout if finishing else grace
    for name in _STOP_SIGNALS:
        sig = getattr(_signal, name, None)
        if sig is None:                       # pragma: no cover - non-POSIX
            continue
        try:
            os.kill(rec.pid, sig)
        except ProcessLookupError:
            unregister_session(rec.pid)
            return {**info, "stopped": True, "how": "already-gone",
                    "tried": tried, "detail": "the process exited on its own"}
        except (OSError, PermissionError) as e:
            tried.append(f"{name}-failed: {e}")
            continue
        tried.append(name)
        until = (deadline if name == "SIGINT"
                 else time.monotonic() + grace)
        if _wait(max(until, time.monotonic() + grace)):
            unregister_session(rec.pid)
            return {**info, "stopped": True, "how": f"signal:{name}",
                    "tried": tried,
                    "detail": (f"sent {name} (the same graceful ending as "
                               f"Ctrl-C)" + (": it finished the bar and wrote its "
                                             "records" if rec.record_dir else "")
                               if name == "SIGINT" else
                               f"sent {name} after the graceful ending did not "
                               f"finish in {waited:g}s — "
                               + ("the tail and the records for this take are "
                                  "probably lost" if rec.record_dir else
                                  "this session keeps no records, so none are lost"))}

    if force:
        sig = getattr(_signal, "SIGKILL", None)
        if sig is not None:
            try:
                os.kill(rec.pid, sig)
                tried.append("SIGKILL")
            except ProcessLookupError:
                unregister_session(rec.pid)
                return {**info, "stopped": True, "how": "already-gone",
                        "tried": tried, "forced": True,
                        "detail": "the process exited on its own"}
            except (OSError, PermissionError) as e:
                tried.append(f"SIGKILL-failed: {e}")
            else:
                if _wait(time.monotonic() + grace):
                    unregister_session(rec.pid)
                    return {**info, "stopped": True, "how": "signal:SIGKILL",
                            "tried": tried, "forced": True,
                            "detail": ((f"forced with SIGKILL after the graceful "
                                        f"ending did not finish. Since v1.3 the "
                                        f"take is written as it happens, so the "
                                        f"recording holds everything up to the "
                                        f"last flush — but the tail did not play "
                                        f"and ending.json was not written.")
                                       if rec.record_dir else
                                       "forced with SIGKILL after the graceful "
                                       "ending did not finish, so the tail did "
                                       "not play. This session keeps no records, "
                                       "so none are lost.")}

    return {**info, "stopped": False, "how": "failed", "tried": tried,
            "forced": bool(force),
            "detail": (f"pid {rec.pid} is still running after "
                       f"{', '.join(tried) or 'no attempt'} — it is wedged. "
                       + (f"Even SIGKILL did not clear it, which means the "
                          f"process is stuck in the kernel (uninterruptible "
                          f"I/O); only a reboot or the device coming back will "
                          f"free it." if force else
                          f"'plume stop --pid {rec.pid} --force' escalates to "
                          f"SIGKILL."
                          + (" Since v1.3 the take is streamed to disk as "
                             "it happens, so forcing costs the tail, not the "
                             "recording." if rec.record_dir else "")))}


@contextlib.contextmanager
def sigint_stops_this():
    """Let SIGINT interrupt a blocking wait on sound, in a process that inherited it ignored.

    A script's background job inherits ``SIG_IGN`` and CPython keeps it, so ``plume
    stop``'s SIGINT would do nothing while the sound played on.  This is for verbs with
    nothing to finish or write: :class:`GracefulSignals` is the live sessions' version.
    Restores what it found, and does nothing off the main thread.
    """
    import signal as _signal  # noqa: PLC0415

    ignored = _signal.getsignal(_signal.SIGINT) == _signal.SIG_IGN
    try:
        if ignored:
            _signal.signal(_signal.SIGINT, _signal.default_int_handler)
    except (ValueError, OSError):            # not the main thread
        ignored = False
    try:
        yield
    finally:
        if ignored:
            _signal.signal(_signal.SIGINT, _signal.SIG_IGN)


class GracefulSignals:
    """Turn SIGINT/SIGTERM into the transport's ordinary graceful ending.

    **Why this exists, measured rather than assumed.**  A live session started
    in the background by a non-interactive shell — ``plume jam &`` in a
    script, an agent's subprocess, a CI runner — inherits SIGINT set to
    ``SIG_IGN``, and CPython does not override an inherited ``SIG_IGN``.  So
    the session ignores Ctrl-C's signal entirely.  That is exactly the
    population unattended sessions came from: agent-started, backgrounded,
    unattended.  Without this, ``plume stop``'s graceful SIGINT rung is dead
    for them and the only thing that lands is SIGTERM, which kills the process
    mid-take and throws the recording away.  Verified directly: a backgrounded
    child reported ``signal.getsignal(SIGINT) == SIG_IGN`` and played straight
    through a SIGINT.

    Installing our own handler overrides the inherited disposition, so both
    signals become the same finish-the-bar/flush-the-tail/write-the-records
    ending the session already runs for a socket stop or the cap.

    A SECOND signal restores the default handler, so a user hammering Ctrl-C
    still gets out — a graceful ending must never become a trap.

    **v1.3 — restoring the default handler is not, on its own, enough**, and
    the wedged session proved it.  Restoring ``SIG_DFL`` only helps if a
    *third* signal arrives AND the main thread is in a position to be killed
    by it; it also means the second signal itself is swallowed, so ``plume
    stop``'s SIGINT-then-SIGTERM ladder had exactly one rung that could ever
    land.  ``on_repeat`` is the other half: the transports pass a callback that
    shortens :class:`HardExit`'s fuse, which needs nothing from the main
    thread at all.

    Degrades silently when it cannot install (not the main thread, or a
    platform without the signal): the session still plays, and ``plume stop``
    still has its socket rung.
    """

    #: Handled here.  SIGHUP is deliberately left alone — its default already
    #: ends a session when the terminal closes, which is the safe direction.
    SIGNALS = ("SIGINT", "SIGTERM")

    def __init__(self, on_repeat=None):
        self.event = threading.Event()
        self.reason: str | None = None
        self._on_repeat = on_repeat
        self._old: dict = {}

    def __enter__(self) -> "GracefulSignals":
        import signal as _signal  # noqa: PLC0415

        for name in self.SIGNALS:
            sig = getattr(_signal, name, None)
            if sig is None:                  # pragma: no cover - non-POSIX
                continue
            try:
                self._old[sig] = _signal.signal(sig, self._handle)
            except (ValueError, OSError):    # not the main thread; unsupported
                continue
        return self

    def _handle(self, signum, _frame) -> None:
        import signal as _signal  # noqa: PLC0415

        if self.event.is_set():
            # Already ending and asked again: get out of the way so the next
            # one is fatal.  Never trap someone who wants out.
            try:
                _signal.signal(signum, _signal.SIG_DFL)
            except (ValueError, OSError):    # pragma: no cover - defensive
                pass
            if self._on_repeat is not None:
                try:
                    self._on_repeat(signum)
                except Exception:            # noqa: BLE001 - never re-raise
                    pass                     # pragma: no cover
            return
        self.reason = (END_INTERRUPT if signum == getattr(_signal, "SIGINT", None)
                       else END_TERMINATED)
        self.event.set()

    def __exit__(self, *exc) -> None:
        import signal as _signal  # noqa: PLC0415

        for sig, old in self._old.items():
            try:
                _signal.signal(sig, old)
            except (ValueError, OSError):    # pragma: no cover - defensive
                pass
        self._old.clear()


def _live_stop_reason(cues: "CueServer | None", deadline: float | None, *,
                      signals: "GracefulSignals | None" = None,
                      now: float | None = None) -> str | None:
    """Why the transport's status loop should end, or None to keep playing.

    ``deadline`` is a :func:`time.monotonic` instant (or None), matching
    :func:`session_deadline` called with a monotonic ``started``.
    """
    if signals is not None and signals.event.is_set():
        return signals.reason or END_STOPPED
    if cues is not None and cues.stopped.is_set():
        return END_STOPPED
    if deadline is not None and (time.monotonic() if now is None else now) >= deadline:
        return END_MAX_MINUTES
    return None


def _ending_banner(cues: "CueServer | None", max_minutes: float) -> str:
    """The "how this ends" half of a live session's opening status line.

    A player who can see how the session ends before it starts never has to
    go looking for a PID.
    """
    bits = ["Ctrl-C to end", f"'plume stop' from any terminal (pid {os.getpid()})"]
    bits.append(f"self-stops after {max_minutes:g} min" if max_minutes
                else "no time cap (--max-minutes 0)")
    return " — ".join(bits)


def _ending_doc(reason: str, *, started_mono: float, max_minutes: float,
                bars: int) -> dict:
    """The ``ending.json`` payload: why a recording stops where it does."""
    return {
        "reason": reason,
        "capped": reason == END_MAX_MINUTES,
        "max_minutes": float(max_minutes),
        "elapsed_s": round(time.monotonic() - float(started_mono), 2),
        "bars": int(bars),
    }


# ------------------------------------------------------------------ clock

class MachineClock:
    """The machine's own bar grid — the clock lead mode holds (SPEC-V2 §0).

    Pure and deterministic: no wall clock, no rng.  Bars are committed one at
    a time at whatever tempo the negotiator just chose, so a concession is a
    *period* change taking effect on the next bar and never a jump inside one.
    Times are in output-stream seconds (t=0 is the first sample the stream
    plays), which is exactly the coordinate ``TimelineBuffer.write_at`` wants.
    """

    def __init__(self, bpm: float, *, t0: float = 0.0, beats_per_bar: int = 4,
                 first_bar: int = 0):
        bpm = float(bpm)
        if not math.isfinite(bpm) or bpm <= 0.0:
            raise ValueError(f"MachineClock needs bpm > 0, got {bpm}")
        if int(beats_per_bar) < 1:
            raise ValueError(f"beats_per_bar must be >= 1, got {beats_per_bar}")
        self.bpm = bpm
        self.beats_per_bar = int(beats_per_bar)
        self._t = float(t0)
        self._bar = int(first_bar)

    @property
    def bar(self) -> int:
        """Index of the next bar to be committed."""
        return self._bar

    @property
    def t_s(self) -> float:
        """Start time of that bar, in output-stream seconds."""
        return self._t

    def bar_len_s(self, bpm: float | None = None) -> float:
        b = float(self.bpm if bpm is None else bpm)
        return self.beats_per_bar * 60.0 / b

    def commit(self, bpm: float | None = None) -> tuple[int, float, float]:
        """Take the next bar at ``bpm``; return ``(bar, t_start_s, bpm)``."""
        if bpm is not None:
            bpm = float(bpm)
            if not math.isfinite(bpm) or bpm <= 0.0:
                raise ValueError(f"commit needs bpm > 0, got {bpm}")
            self.bpm = bpm
        bar, t0 = self._bar, self._t
        self._t = t0 + self.bar_len_s()
        self._bar = bar + 1
        return bar, t0, self.bpm


def machine_beat_state(bar: int, t_s: float, bpm: float, *,
                       beats_per_bar: int = 4):
    """A :class:`~plume.listen.beat.BeatState` describing the MACHINE's grid.

    Confidence 1.0 and ``locked=True`` are not a claim about the human — they
    are the literal truth about this grid, which the machine generated.  It is
    what the renderer is handed, so ``DuetEngine`` sounds the machine's tempo
    rather than the tracker's estimate of the human's.
    """
    from .listen.beat import BeatState  # noqa: PLC0415

    interval = 60.0 / float(bpm)
    return BeatState(
        t_s=float(t_s), bpm=float(bpm), confidence=1.0, locked=True,
        bar=int(bar), beat_in_bar=0, phase_beats=0.0,
        next_beat_t_s=float(t_s) + interval,
        next_downbeat_t_s=float(t_s) + beats_per_bar * interval,
        beat_interval_s=interval,
    )


def transpose_symbol(sym: str | None, semitones: int,
                     key: str | None) -> str | None:
    """Transpose a chord symbol, spelled for ``key``.  ``None``/unparseable
    passes through unchanged — a conformance transposition must never be able
    to make a bar unplayable."""
    if not sym or not int(semitones):
        return sym
    from .theory import parse_chord, parse_key, spell  # noqa: PLC0415

    try:
        ch = parse_chord(str(sym))
        k = parse_key(key) if key else None
    except ValueError:
        return sym
    return spell((ch.root + int(semitones)) % 12, k) + ch.quality


def transpose_key(key: str | None, semitones: int) -> str | None:
    """Transpose a key name ('F minor' -> 'F# minor')."""
    if not key or not int(semitones):
        return key
    from .theory import Key, parse_key, spell  # noqa: PLC0415

    try:
        k = parse_key(key)
    except ValueError:
        return key
    nk = Key(tonic=(k.tonic + int(semitones)) % 12, mode=k.mode)
    return f"{spell(nk.tonic, nk)} {nk.mode.replace('_', ' ')}"


def machine_harmony_state(t_s: float, key: str | None, chord: str | None, *,
                          transpose: int = 0):
    """The harmony the MACHINE is playing this bar (its composed key/chord,
    transposed by any accumulated conformance)."""
    from .listen.harmony import HarmonyState  # noqa: PLC0415

    k = transpose_key(key, transpose)
    c = transpose_symbol(chord, transpose, k)
    return HarmonyState(
        t_s=float(t_s), key=k, key_conf=1.0 if k else 0.0,
        chord=c, chord_conf=1.0 if c else 0.0,
        register_low_midi=None, register_high_midi=None,
    )


# ------------------------------------------------------------------ bar plan

#: A tap-started take opens on a sparse composed section, but drums you are
#: playing along to have to state the beat.
TAP_START_MIN_DENSITY = 0.55
#: Real knocks on a guitar body read 1.0-1.9 with the laptop across the room
#: (2026-11-02, projector take), under the detector's default floor of 1.5.
TAP_START_KNOCK_FLOOR = 1.0
#: How strong a quieter knock has to be for an already-confirmed count to claim
#: it as an earlier beat.  Her real first knocks read 0.6-1.0.
TAP_START_SOFT_KNOCK = 0.45
TAP_START_SOFT_FLOOR = 0.5
TAP_START_SOFT_KEPT = 4096
#: Lead mode renders hot (about -9 dBFS rms), and this take goes straight into
#: headphones with an unamplified guitar to hear over them.  It was -10, and the
#: owner heard the band as a little buried under her (2026-11-06): in a
#: direct-input page take machine.wav sat about 12 dB (LUFS) under human.wav.
#: Up 2 dB; she stays on top.  At -8 the hottest style's densest bars peak near
#: 0 dBFS, so the band is soft-limited at :data:`TAP_START_CEILING_DB`.
TAP_START_GAIN_DB = -8.0
#: The band's peak ceiling in a page take.  machine.wav is 24-bit and the output
#: has no limiter, so a peak past full scale would clip in both.  The limiter is
#: transparent 3 dB under this; at -8 dB it touches under 0.5% of samples (trap,
#: densest bars) and none in house, jazz or rnb.
TAP_START_CEILING_DB = -1.0

#: Density added on the announce bar — the fill/riser that tells the human a
#: section change is coming (SPEC-V2 §2, "announce it").
ANNOUNCE_DENSITY_BUMP = 0.15

#: Machine bus gain once it has fully yielded the clock, in dB.  Yielding is
#: not primarily a volume gesture, but a machine that has handed over the lead
#: and is still at full level is not out of the way in any sense a player hears.
YIELD_GAIN_DB = -5.0


def lead_bar_plan(bar: int, section, duet_roles, role_state, form_plan, *,
                  human=None, density_scale: float = 1.0):
    """Build the machine's :class:`~plume.mind.BarPlan` for one lead-mode bar.

    Pure.  The composed section supplies WHAT plays (its ``plays`` map, cut to
    the roles this style can voice) and HOW MUCH (its ``energy`` becomes the
    density hint); the role state supplies how far out of the way to get.

    ``state`` is ``"comping"`` — a v2 amendment.  There is no ``"leading"`` in
    ``mind.STATES`` and adding one would change a frozen v1 vocabulary that
    ``sim`` and every decision-log consumer switch on; the lead-ness of the bar
    is carried by the role/form records beside it, which is where a test can
    actually assert it.
    """
    from .mind import BarPlan  # noqa: PLC0415

    energy = float(getattr(section, "energy", 0.5)) if section is not None else 0.5
    plays = dict(getattr(section, "plays", {}) or {}) if section is not None else {}
    names = [r for r in duet_roles if r in plays] or list(duet_roles)
    # v2.1: what a sparser/busier cue does (SPEC-V2 section 3, C2).
    density = min(1.0, max(0.05, energy * float(density_scale)))
    announce = bool(getattr(form_plan, "announce", False))
    if announce:
        density = min(1.0, density + ANNOUNCE_DENSITY_BUMP)
    clock = float(getattr(role_state, "clock_authority", 1.0))
    clock = min(1.0, max(0.0, clock))
    gain_db = YIELD_GAIN_DB * (1.0 - clock)
    ceiling = None
    if human is not None and getattr(human, "register_low_midi", None) is not None:
        # Only once we are sharing the floor: while we own it, the composed
        # arrangement's own register is the point.
        if clock < 0.75:
            ceiling = float(human.register_low_midi) - 2.0
    sec_name = str(getattr(section, "name", "?")) if section is not None else "?"
    form_auth = float(getattr(role_state, "form_authority", 1.0))
    # v2.1: the opening word tells the truth about the chair the machine is in
    # (see the SPEC-V2 section 4 amendment) — "lead:" on a bar the machine has
    # yielded reads, cold, as though it were still holding the floor.
    head = "lead" if clock >= 0.5 else "yielded"
    reason = (f"{head}: {sec_name} e={energy:.2f} clock={clock:.2f} "
              f"form={form_auth:.2f}" + (" +announce" if announce else ""))
    return BarPlan(
        bar=int(bar), state="comping",
        roles={r: 1.0 for r in names},
        gain_db=round(gain_db, 3), density=round(density, 4),
        register_ceiling_midi=ceiling, reason=reason,
    )


def _named_key(prompt: str | None) -> str | None:
    """The key the player wrote in their prompt, spelled "F major", or None."""
    if not prompt:
        return None
    from .compose import parse_prompt  # noqa: PLC0415
    from .theory import parse_key  # noqa: PLC0415
    named = parse_prompt(prompt).key
    if not named:
        return None
    key = parse_key(str(named))
    return f"{str(named).split()[0]} {'major' if key.mode == 'major' else 'minor'}"


def _wants_lead(role, lead_in, session) -> bool:
    """Does this ``jam`` invocation actually mean 'the machine starts'?

    A missing role, or a bare ``follow`` role with no authority overrides and
    nothing composed to lead with, is v1 — and must stay on the v1 path.
    """
    if lead_in or session is not None:
        return True
    if role is None:
        return False
    if str(getattr(role, "role", "follow")) != "follow":
        return True
    return (getattr(role, "clock_authority", None) is not None
            or getattr(role, "form_authority", None) is not None)


def _section_at(flat, idx):
    """The flattened cursor entry at ``idx`` (or None)."""
    if not flat or idx is None or not (0 <= int(idx) < len(flat)):
        return None
    return flat[int(idx)]


def _resolve_lead_session(session, prompt, style, seed):
    """A v0 ``Session`` for the machine to lead with, from whatever was given."""
    if session is not None:
        return session
    from .compose import compose, parse_prompt  # noqa: PLC0415

    text = prompt or f"{style} jam"
    return compose(parse_prompt(text), seed=int(seed))


# ------------------------------------------------------------------ lead

def lead(*, session=None, prompt: str | None = None, style: str = "lofi",
         seed: int = 0, params=None, role=None, form_params=None,
         device: str | None = None, monitor: bool = False,
         record_dir: str | None = None, tightness: float | None = None,
         from_bar: int = 0, socket_path: str | None = None,
         status: bool = True, max_minutes: float | None = None,
         input_mode: str | None = None, count_in: bool = False,
         tap_start: bool = False, wind_down_s: float | None = None,
         input_device: str | None = None, input_channel: int | None = None,
         voice_channel: int | None = None, direct: bool = False,
         midi: str | None = None, midi_clock: bool = False,
         detect: bool = False, output_device: str | None = None) -> None:
    """The machine starts; you join (SPEC-V2 section 5).

    ``tap_start``: nothing plays until four knocks set the tempo, the drums then
    hold it, and the band joins once your key has settled (:mod:`plume.bandentry`).
    Knocks are only listened for once, so playing can never re-set the tempo.

    ``detect``: needs ``tap_start``.  No session, and the prompt is optional: whatever
    it does not name, plume works out from the playing, as a guess the rows say is one.

    Same transport as :func:`jam` — same callback, same ring buffers, same
    self-masking analysis thread — with the planner inverted: bars come from
    :class:`MachineClock` at the machine's own tempo instead of from the beat
    tracker's predicted downbeats, so the machine plays from t=0 and keeps
    playing through anything the human does (or does not do).

    Per bar the planner runs, in order: summarize the human's completed bar ->
    :meth:`plume.roles.RoleNegotiator.update` (tempo concession, role handoff,
    harmonic conformance) -> :meth:`plume.mind.Mind.plan_form` (where the
    arrangement goes, plus any cue that came due) -> :func:`lead_bar_plan` ->
    ``DuetEngine.render_bar`` at the negotiated tempo.

    **Amendment — what "the machine plays its composed arrangement" means
    here.** ``DuetEngine`` is the only bar-granular renderer in the codebase;
    ``engine.render`` renders whole sessions to a file and cannot be driven a
    bar at a time by a live clock.  So live lead takes the Session's tempo,
    key, chord cycle, style and section map (energy and ``plays``) and voices
    them through ``DuetEngine``.  That is the arrangement as a MAP, which is
    what section 2 asks for — but it is NOT a sample-exact live replay of
    ``plume render``.  ``simlead`` is the ground truth for the full
    arrangement; this is the live transport for it.

    **v2.2 — three ways this ends, one ending.**  Ctrl-C, a ``plume stop``
    over the cue socket, and the ``max_minutes`` cap all converge on the same
    graceful path (finish the bar, flush the tail, write the records), and the
    reason lands in the status line and in ``ending.json``.  The session also
    registers itself so ``plume sessions`` can name it from another terminal.
    """
    if detect and (session is not None or not tap_start):
        raise RealtimeError("--detect works with --tap-start and without a session")
    sd = _import_sounddevice()
    route = resolve_input(sd, device, input_device=input_device,
                          input_channel=input_channel, voice_channel=voice_channel,
                          direct=direct, output_device=output_device)
    check_audio_consent(device)
    dev = route.device
    midi_port = _resolve_midi_port(midi)
    # With --midi the band listens to the notes, whatever --direct says about the
    # audio input, so a level from that input decides nothing here.
    is_direct = route.source == "direct" and midi_port is None
    max_minutes = resolve_max_minutes(max_minutes)

    from .duet import DuetEngine  # noqa: PLC0415
    from .listen.beat import BeatTracker  # noqa: PLC0415
    from .listen.features import FeatureExtractor  # noqa: PLC0415
    from .listen.harmony import HarmonyTracker  # noqa: PLC0415
    from .mind import (BarFeatures, Cue, Mind,  # noqa: PLC0415
                       MindParams, flatten_sections, parse_cue)
    from .roles import (RoleNegotiator, RoleParams,  # noqa: PLC0415
                        effective_bpm, onset_phase_offset_ms)

    from .looper import DECLICK_SAMPLES, Looper  # noqa: PLC0415

    sess = _resolve_lead_session(session, prompt, style, seed)
    params = params if params is not None else MindParams()
    role = role if role is not None else RoleParams(role="lead")
    if tap_start:
        # She set the tempo on purpose: never chase, never yield the floor, and
        # the key comes from BandEntry rather than from conformance.
        role = dataclasses.replace(role, tempo_policy="hold",
                                   conform_enabled=False,
                                   handoff_density=1e9)
    style = str(getattr(sess, "style", "") or style)
    composed_bpm = float(getattr(sess, "bpm", 120.0) or 120.0)
    composed_key = getattr(sess, "key", None)

    rec_path = None if record_dir is None else Path(record_dir)
    _refuse_a_kept_take(rec_path)
    recorder = (None if rec_path is None
                else SessionRecorder(rec_path, voice=route.voice is not None))

    extractor = FeatureExtractor(sr=SR)
    tkw: dict = {}
    if tightness is not None:
        tkw["tightness"] = tightness
    if input_mode in ("acoustic", "programmed"):
        tkw["acoustic"] = input_mode == "acoustic"
    if count_in and not tap_start:
        tkw["count_in"] = True
    tracker = BeatTracker(**tkw)
    harmony = HarmonyTracker()
    entry = None
    tap_detector = None
    style_pick = None
    named_key = None
    if tap_start:
        from .bandentry import (DETECT_KEY_FALLBACK_BARS, END_AFTER_S,  # noqa: PLC0415
                                FOLLOW, BandEntry, BandFollow, GuitarEar, SilenceEnding,
                                StylePick, outro, song_section)
        guitar_ear = GuitarEar()
        # The band follows how she plays.  FOLLOW is the only mode so far: the seam a
        # "plume leads" mode would come in by.  A room mic hears the band too, so it
        # follows more cautiously (see BandFollow).
        follow = BandFollow(trusted=is_direct or midi_port is not None, mode=FOLLOW)
        from .listen.features import CountInDetector  # noqa: PLC0415
        # The ear confirms the player about a bar after a level would, so a direct input
        # waits one bar longer to give the player the same eight or so before the band.
        named_key = _named_key(prompt)
        entry = BandEntry(style, known_key=named_key,
                          **({"after_first_heard_bars": 6} if is_direct else {}),
                          **({"key_fallback_bars": DETECT_KEY_FALLBACK_BARS} if detect else {}))
        if detect:
            from .compose import parse_prompt  # noqa: PLC0415
            if not parse_prompt(prompt or "").styles:
                # Until the count lands ``style`` is only compose's placeholder.
                style_pick = StylePick(key=named_key,
                                       trust_busy=is_direct or midi_port is not None)
        # A room mic cannot hear her stop over the band, so it keeps the long wait.
        silence_ending = (SilenceEnding(end_after_s=END_AFTER_S)
                          if is_direct or midi_port is not None else SilenceEnding())
        # Stricter on evenness and speed than a tracker's count-in: with no
        # quiet start required, laughing in the room read as a count at 181 bpm.
        # Either four clear knocks or six soft ones: how hard she knocks varies by
        # a factor of two, and six even hits are too many for laughter to fake.
        tap_detector = [
            CountInDetector(sr=SR, floor=TAP_START_KNOCK_FLOOR, waiting=True,
                            even_frac=0.15, bpm_range=(45.0, 170.0)),
            CountInDetector(sr=SR, floor=TAP_START_SOFT_FLOOR, breadth=0.35,
                            taps=6, waiting=True, even_frac=0.15,
                            bpm_range=(45.0, 170.0)),
        ]
    comp_samples = [0]
    song_start = [0.0]
    mind = Mind(params, form_params) if form_params is not None else Mind(params)
    negotiator = RoleNegotiator(role, composed_bpm=composed_bpm,
                                composed_key=composed_key)
    duet = DuetEngine(style=style, key_fallback=str(composed_key or "A minor"),
                      seed=seed, sr=SR)

    def swing_of(name: str) -> float:
        # The machine's own swing, so the human's phase is measured against the
        # grid the machine IS playing rather than an idealised straight one.
        try:
            from .styles_lib import load_style as _load_style  # noqa: PLC0415
            return float(_load_style(name)["swing"])
        except Exception:                    # noqa: BLE001 - defensive
            return 0.5
    machine_swing = swing_of(style)

    def change_style(name: str) -> "np.ndarray":
        """A new engine in another style.  Returns what the old one still had ringing."""
        nonlocal duet, style, machine_swing
        old = duet
        duet = DuetEngine(style=name, key_fallback=str(composed_key or "A minor"),
                          seed=seed, sr=SR)
        style = name
        machine_swing = swing_of(name)
        entry.restyle(name)
        cues.info["style"] = name
        return old.flush_tail()
    from .sim import _LeadCues  # noqa: PLC0415
    lead_cues = _LeadCues()
    flat = flatten_sections(sess)

    in_ring = RingBuffer(SR * 8)
    machine_ring = RingBuffer(SR * 8)
    voice_ring = None if route.voice is None else RingBuffer(SR * 8)
    timeline = TimelineBuffer(SR * 30, channels=2)

    stop = threading.Event()
    ending = threading.Event()
    state_lock = threading.Lock()
    shared = {"beat": None, "harmony": None, "events": [], "onsets": [],
              "taps": None, "chroma": []}
    masker = SelfMasker(enabled=route.source != "direct")
    midi_in = None
    if midi_port is not None:
        midi_in, extractor, masker = _midi_listening(midi_port, timeline, recorder,
                                                     status=status,
                                                     follow_clock=midi_clock)
    last_decision: dict = {}            # {"d": <most recent>} for the status line
    n_decisions = [0]
    plan_us: list[float] = []          # per-bar planning cost, microseconds

    # Self-bleed is only ever warned about: see plume.looper.
    looper = Looper(SR, bleed_probe=lambda: (masker.coupling_db
                                             if masker.coupling_measured else None))
    cues = CueServer(socket_path, kind="lead")
    cues.intercept = looper.handle_cue
    cues.start()
    # v2.2 — discoverable and stoppable.  Lead already owns a socket, so the
    # stop verb rides it; only the registry entry is new.
    started_mono = time.monotonic()
    # Detect composes a placeholder, whose style and name are not what this take is.
    shown_style = "detect" if style_pick is not None else style
    sess_name = None if detect else getattr(sess, "name", None)
    cues.info = {"style": shown_style, "role": role.role, "name": sess_name}
    register_session(kind="lead", style=shown_style, role=role.role,
                     session_name=sess_name,
                     socket_path=cues.path, max_minutes=max_minutes,
                     record_dir=record_dir)
    # v1.3 — streamed from here on: manifest first, then a flushed line per bar.
    if recorder is not None:
        recorder.open(session_manifest(
            kind="lead", style=shown_style, role=role.role, seed=seed,
            session_name=sess_name, socket_path=cues.path,
            max_minutes=max_minutes, record_dir=record_dir, device=device,
            bpm=composed_bpm, params=params, role_params=role,
            started_at=time.time(), input_route=route, midi_port=midi_port,
            detect=detect, monitor=monitor))

        def log_cue(item: dict, source: str) -> None:
            # ``t_s`` is on the rows' clock: a bar written at t sounds at the
            # playhead position t - comp (see the write_at in plan_one_bar).
            doc = {"received_at": time.time(),
                   "t_s": round((timeline.playhead + comp_samples[0]) / SR, 4),
                   "bar": cues.bar, "text": item["text"], "source": source,
                   "at": item["at"], "ttl_bars": item["ttl_bars"]}
            if "loop" in item:
                doc["loop"] = item["loop"]
            recorder.cue_soon(doc)
        cues.on_received = log_cue
    stop_deadline = session_deadline(started_mono, max_minutes)

    def callback(indata, outdata, frames, time_info, status_flags):  # noqa: ARG001
        in_ring.write(indata[:, route.guitar])
        # Before machine_ring: the analysis thread reads what both of those hold, so
        # this ring always has at least that much.
        if voice_ring is not None:
            voice_ring.write(indata[:, route.voice])
        timeline.read_into(outdata)
        machine_ring.write(0.5 * (outdata[:, 0] + outdata[:, 1]))
        if monitor:
            outdata[:, 0] += indata[:, route.guitar]
            outdata[:, 1] += indata[:, route.guitar]

    hop_rms = HopRms()

    def analysis_loop():
        nonlocal harmony, tap_detector
        from .bandentry import count_start  # noqa: PLC0415
        soft_knocks: list[float] = []
        drain_until = None                  # v1.3: bounded on both paths
        while True:
            if stop.is_set() and drain_until is None:
                drain_until = time.monotonic() + ANALYSIS_DRAIN_S
            if drain_until is not None and time.monotonic() >= drain_until:
                break
            n = min(in_ring.available, machine_ring.available)
            if n == 0:
                if stop.is_set():
                    break
                stop.wait(0.005)
                continue
            x = in_ring.read(n)
            m = machine_ring.read(n)
            v = None if voice_ring is None else voice_ring.read(n)
            if recorder is not None:
                recorder.audio(x, m, v)
            looper.feed(x)
            masked_frames, _flags, beat_state, harm_state = analysis_step(
                extractor=extractor, hop_rms=hop_rms, masker=masker,
                tracker=tracker, harmony=harmony,
                human_block=x, machine_block=m)
            loop_tempo = looper.take_tempo() if tap_detector is not None else None
            if loop_tempo is not None:
                # The loop set the tempo.  As after a count: never listen for one
                # again, and the key starts from what comes next.
                tap_detector = None
                harmony = HarmonyTracker()
                harm_state = None
                with state_lock:
                    shared["events"].clear()
                    shared["onsets"].clear()
                    shared["loop_tempo"] = loop_tempo
            elif tap_detector is not None and looper.capturing:
                # A phrase being looped is not a count, however even it sounds.
                masked_frames = []
            elif tap_detector is not None:
                soft_knocks.extend(
                    f.t_s for f in masked_frames
                    if getattr(f, "attack", 0.0) >= TAP_START_SOFT_KNOCK
                    and getattr(f, "attack_breadth", 0.0) >= 0.4)
                del soft_knocks[:-TAP_START_SOFT_KEPT]
                heard = [d.process(masked_frames) for d in tap_detector]
                taps = next((h for h in heard if h), None)
                if taps:
                    taps = count_start(taps, soft_knocks)
                    tap_detector = None     # once: playing cannot re-set tempo
                    # Knocks are not harmony; her key starts from her first note.
                    harmony = HarmonyTracker()
                    harm_state = None
                    with state_lock:
                        shared["events"].clear()
                        shared["onsets"].clear()
                        shared["taps"] = taps
                else:
                    # Still waiting: no bar will ever read these frames.
                    masked_frames = []
            with state_lock:
                shared["beat"] = beat_state
                shared["harmony"] = harm_state
                for f in masked_frames:
                    if tap_start:
                        shared["chroma"].append((f.t_s, f.chroma))
                    shared["events"].append((f.t_s, f.rms_db, f.onset))
                    if f.onset:
                        shared["onsets"].append(f.t_s)

    def plan_one_bar(clock, state, bpm):
        """Plan, render and place exactly one machine bar.  Returns the next
        bar's bpm.  Split out of the loop so its cost can be measured (and so
        the loop body stays readable)."""
        held_bpm = looper.held_bpm()    # a loop is a fixed length of audio
        if held_bpm is not None:
            bpm = held_bpm
        bar, t_bar, _ = clock.commit(bpm)
        bar_len = clock.bar_len_s(bpm)
        looper.note_bar(bar, int(round(t_bar * SR)) - comp_samples[0], bpm)
        # This bar is planned a bar ahead of the playhead, so the bar just
        # before it has barely started and holds almost no frames.  Read the
        # last bar that has actually finished, at the tempo it was played: a
        # tempo change must not slide the window off that bar's own grid.  The
        # bar index handed on stays ``bar - 1``: Mind reads ``human.bar + 1`` as
        # the bar being planned, whichever bar was measured.
        starts = state["bar_starts"]
        w_end = starts[-1] if starts else t_bar - bar_len
        w_start = starts[-2] if len(starts) > 1 else w_end - bar_len
        state["bar_starts"] = [*starts[-1:], t_bar]
        # tap-start: not the knocks
        first_heard_bar = int(from_bar) + (2 if tap_start else 1)
        with state_lock:
            beat_state = shared["beat"]
            harm_state = shared["harmony"]
            evs = [e for e in shared["events"] if w_start <= e[0] < w_end]
            shared["events"] = [e for e in shared["events"] if e[0] >= w_end]
            # The bar now playing, as far as it has been heard: BandFollow reads its
            # first half early (see follow_parts).
            later_evs = list(shared["events"]) if entry is not None else []
            onsets = [t for t in shared["onsets"] if w_start <= t < w_end]
            shared["onsets"] = [t for t in shared["onsets"] if t >= w_end]
            bar_chroma = [c for t, c in shared["chroma"] if w_start <= t < w_end]
            shared["chroma"] = [e for e in shared["chroma"] if e[0] >= w_end]
        human = None
        if bar >= first_heard_bar and evs:
            kwargs = bar_features_dict(
                bar - 1, evs, silence_rms_db=params.silence_rms_db,
                ref_db=state["ref_db"],
                register_low_midi=(harm_state.register_low_midi
                                   if harm_state else None),
                register_high_midi=(harm_state.register_high_midi
                                    if harm_state else None))
            human = BarFeatures(**kwargs)
            if not human.silent:
                state["ref_db"] = (human.rms_db if state["ref_db"] is None
                                   else 0.5 * human.rms_db + 0.5 * state["ref_db"])
        offset_ms = None
        unit_s = (w_end - w_start) / 16.0
        if human is not None and not human.silent:
            # v2.1: measured against the grid the machine is PLAYING (its own
            # swing), and the subdivision is handed to the negotiator so its
            # plausibility bound can be stated in the instrument's own units.
            # Both are the same fixes sim got; without them the live path
            # would keep the aliasing defect the offline twin no longer has.
            offset_ms = onset_phase_offset_ms(onsets, w_start, w_end,
                                              unit_s, swing=machine_swing)
        mbeat = machine_beat_state(bar, t_bar, bpm)
        hb = beat_state if beat_state is not None else mbeat
        hh = harm_state if harm_state is not None else machine_harmony_state(
            t_bar, None, None)
        role_state = negotiator.update(
            bar=bar, beat=hb, harmony=hh, human=human,
            machine_grid_t_s=t_bar, human_offset_ms=offset_ms,
            human_offset_unit_s=unit_s)

        state["pending"].extend(cues.drain())
        cue = None
        if state["pending"]:
            c = state["pending"].pop(0)
            cue = Cue(text=c["text"], at=c.get("at", "next-phrase"),
                      ttl_bars=int(c.get("ttl_bars", 16)))
        # The form listens for the PLAYER slowing down, so it gets the tracker's
        # tempo, as in sim.  The machine's grid would read a ritard cue's own
        # slowing as the player ending the tune.  Only the bar number is the machine's.
        # No tracker state means no frames yet, so no human bar to read at all.
        form_beat = (mbeat if beat_state is None
                     else dataclasses.replace(beat_state, bar=bar))
        form = mind.plan_form(role_state, form_beat, hh, human, sess, cue)
        # v2.1: the renderer's half of a cue.  ``Mind.plan_form`` owns the FORM
        # consequences; tempo and texture are the renderer's and were reported
        # as "applied" while doing nothing at all (SPEC-V2 section 3, C2).
        # Live gets the tempo half — it is the machine's own clock, which this
        # loop already owns — and reports the rest honestly.
        cue_notes = lead_cues.apply(getattr(form, "applied_cue", None))
        if held_bpm is not None and any("(tempo:" in n for n in cue_notes):
            cue_notes.append(f"tempo cue NOT applied yet: a loop is holding "
                             f"{held_bpm:.1f} bpm until the last one is dropped")
        section = _section_at(flat, form.section_idx)
        sec_spec = None
        if section is not None:
            try:
                sec_spec = (sess.movements[section.movement]
                            .sections[section.section])
            except (AttributeError, IndexError, TypeError):
                sec_spec = None
        chords = list(getattr(sec_spec, "chords", None) or [])
        # The chord cursor is sim's: from the top of a section, restarted when
        # the form enters one (a re-entered one-section map included), and
        # still running through an extend.
        if (form.section_idx != state["sec_idx"]
                or str(form.action) in ("advance", "jump")):
            state["sec_bar"] = 0
        else:
            state["sec_bar"] += 1
        state["sec_idx"] = form.section_idx
        chord = chords[state["sec_bar"] % len(chords)] if chords else None
        mh = machine_harmony_state(t_bar, composed_key, chord,
                                   transpose=role_state.transpose_semitones)
        plan = lead_bar_plan(bar, sec_spec, duet.roles, role_state, form,
                             human=human, density_scale=lead_cues.density_scale)
        entered = None
        loop_db = 0.0
        if entry is not None:
            # Over drums alone the mic can hear steady pitch, which is her.  Once
            # the band is in, all that is left is whether the mic is silent.  A
            # direct input holds nothing but the player, so its level answers the whole way
            # through, and the ear can go on learning the player's notes under the band.
            if is_direct:
                playing = (bar >= first_heard_bar and human is not None
                           and not human.silent)
                if bar >= first_heard_bar:
                    guitar_ear.bar(bar_chroma)
            elif entry.band_is_in:
                playing = human is not None and not human.silent
            else:
                playing = bar >= first_heard_bar and guitar_ear.bar(bar_chroma)
            entered = entry.update(bar, harm_state, playing)
            if style_pick is not None and not style_pick.settled:
                if playing:
                    style_pick.heard_bar(len(onsets))
                if entry.band_is_in:
                    picked = style_pick.at_entry(bpm, entered.key)
                    if picked != style:
                        # Before anything of this bar is rendered, and it rings out
                        # under the new engine's first bar, so there is no gap.
                        timeline.write_at(int(round(t_bar * SR)) - comp_samples[0],
                                          change_style(picked))
                        entered = dataclasses.replace(
                            entered, chord=entry.chord_at(bar, harm_state))
            # Over a room mic with the band in, "not silent" mostly means the band.
            playing_trusted = is_direct or midi_port is not None or not entry.band_is_in
            if entry.band_is_in:
                duet.heard_notes = guitar_ear.notes()
            # The ear only runs before the band; after that its last reading is stale.
            fade_db, fade_done = silence_ending.update(
                t_bar, playing or (not is_direct and not entry.band_is_in
                                   and guitar_ear.last_bar_tonal), bar_len)
            if entered.key is not None:
                mh = machine_harmony_state(t_bar, entered.key, entered.chord)
            # A busier or sparser cue is her own word, and beats following.
            for text in str(getattr(form, "applied_cue", None) or "").split("; "):
                said = parse_cue(text) if text else {}
                if said.get("kind") == "texture":
                    follow.cue(str(said.get("value")))
            ending_now = outro(t_bar - song_start[0], wind_down_s, bar_len)
            followed = None
            if ending_now is None:
                # What she plays is only her on an input that hears nothing else, so a
                # room mic's bars are weighed by BandFollow with that in mind.
                parts = None
                if is_direct and bar >= first_heard_bar:
                    # Her level by the half bar, the newest half as soon as it is
                    # heard.  Only an input that hears nothing but her: off a room
                    # mic a half bar is mostly the band.
                    parts, state["follow_fed"] = follow_parts(
                        evs, later_evs, w_start, w_end, t_bar,
                        fed_until=state.get("follow_fed"),
                        heard_until=(later_evs[-1][0] if later_evs else w_end),
                        silence_rms_db=params.silence_rms_db, ref_db=state["ref_db"])
                followed = follow.bar(None if human is None else human.rms_db,
                                      None if human is None else human.onset_density,
                                      playing=playing and human is not None,
                                      band_in=entry.band_is_in, parts=parts,
                                      bar_s=bar_len)
            if entered.stage == "drums":
                section, density, resting = "intro", TAP_START_MIN_DENSITY, ()
            else:
                section, density, resting = song_section(entry.bars_in)
                density = follow.shape(density) * follow.density
            crash = False
            if ending_now is not None:
                out_bar, out_db, out_done = ending_now
                section, density, resting = "outro", 0.3, ("sub", "shimmer")
                crash = out_bar == 0
                fade_db = min(fade_db, out_db)
                if out_done:
                    state["ended"] = True
                    shared["end_reason"] = END_SONG
            if not crash:
                resting = (*resting, "crash")
            loop_db = fade_db
            plan = dataclasses.replace(
                plan,
                roles={r: 1.0 for r in duet.roles
                       if r not in entered.held_back and r not in resting},
                # a live "busier" or "sparser" cue scales the song's own shape
                density=float(min(1.0, max(0.05, density * lead_cues.density_scale))),
                gain_db=plan.gain_db + TAP_START_GAIN_DB + follow.gain_db + fade_db,
                reason=f"{entered.reason}; {plan.reason}"
                       + (f"; {followed.reason}" if followed is not None else ""))
        audio = duet.render_bar(plan, mbeat, mh)
        if entry is not None:
            audio = _fx.limiter(audio, TAP_START_CEILING_DB)
        timeline.write_at(int(round(t_bar * SR)) - comp_samples[0], audio)
        loop_events = looper.pop_events()
        if looper.active:
            timeline.write_at(
                int(round(t_bar * SR)) - comp_samples[0],
                looper.mix_bar(bar, (int(round((t_bar + bar_len) * SR))
                                     - int(round(t_bar * SR))), gain_db=loop_db,
                                under=audio))
        cues.bar = bar
        doc = {
            "bar": bar, "t_s": round(t_bar, 4), "state": plan.state,
            "reason": plan.reason, "bpm": round(bpm, 2),
            "roles": plan.roles, "gain_db": plan.gain_db,
            "density": plan.density, "key": mh.key, "chord": mh.chord,
            "role": role_state.role,
            "clock_authority": round(role_state.clock_authority, 4),
            "form_authority": round(role_state.form_authority, 4),
            "tempo_offset_pct": round(role_state.tempo_offset_pct, 4),
            "transpose_semitones": role_state.transpose_semitones,
            # Read every bar even under tempo_policy="hold", where it is
            # deliberately not acted on — see sim.py's note; this is the
            # practice-feedback signal.
            #
            # v1.7 — this used to be spelled ``human_offset_ms``, the same key
            # the RAW per-bar measurement uses thirteen lines below, in one
            # dict literal.  Python keeps the last one, so the negotiator's
            # reading was computed and thrown away on every bar of every take
            # and no test could see it: the key was present, the value was
            # just the other one.  They are NOT the same number — the raw
            # measurement is None on a bar the human sat out, while the
            # negotiator's is sticky (``_last_offset_ms``), which is exactly
            # the difference the "hold" policy exists to show — so both are
            # kept, under names that cannot collide.  sim writes them to two
            # different streams (``_lead_line`` the raw, the roles record the
            # sticky); live merges the streams, so live has to disambiguate.
            "role_human_offset_ms": (None if role_state.human_offset_ms is None
                                     else round(role_state.human_offset_ms, 3)),
            "tempo_policy": role_state.tempo_policy,
            "role_reason": role_state.reason,
            "role_events": list(role_state.events),
            "form": {"section_idx": form.section_idx, "action": form.action,
                     "bars_left": form.bars_left_in_section,
                     "announce": bool(form.announce),
                     "applied_cue": form.applied_cue,
                     "dropped_cue": form.dropped_cue,
                     "reason": form.reason},
            # The RAW measurement for this bar — None when the human sat it
            # out.  Same name and same meaning as sim's ``_lead_line``, which
            # is what makes a live take and a simlead take comparable.
            "human_offset_ms": (None if offset_ms is None
                                else round(offset_ms, 2)),
            "cue_effects": cue_notes or None,
        }
        if looper.active or loop_events:
            doc["loop"] = {**looper.status(), "events": loop_events,
                           "held_bpm": held_bpm}
            cues.info["loops"] = doc["loop"]
        else:
            cues.info.pop("loops", None)
        if entered is not None:
            # not the composed harmony, which nobody hears yet
            doc["key"] = entered.key
            doc["chord"] = entered.chord
            doc["entry"] = entered.stage
            doc["entry_reason"] = entered.reason
            # Heard or worked out, rather than named by the player: a page must say so.
            doc["style"] = style
            doc["style_guess"] = style_pick is not None
            doc["key_guess"] = named_key is None
            if state["tempo_from_loop"]:
                doc["tempo_from"] = "loop"
            doc["section"] = section
            doc["playing"] = playing
            doc["playing_trusted"] = playing_trusted
            doc["fade_db"] = round(fade_db, 2)
            doc["follow"] = {"mode": follow.mode, "trusted": follow.trusted,
                             "gain_db": round(follow.gain_db, 3),
                             "density": round(follow.density, 4),
                             "moved": list(followed.moved) if followed is not None else [],
                             "bars": follow.hold_bars,
                             # how far ahead this bar was planned: half a bar when
                             # the band could hear half of the bar before it
                             "ahead_s": round(float(state.get("ahead_s", bar_len)), 3)}
            if fade_done:
                state["ended"] = True
                shared["end_reason"] = END_SILENCE
        # v1.3: flushed to decisions.ndjson the moment the bar is planned, so
        # the log can be tailed while the band plays.
        last_decision["d"] = doc
        n_decisions[0] += 1
        if recorder is not None:
            recorder.decision(doc)
        if form.action == "end":
            state["ended"] = True
        return (effective_bpm(composed_bpm, role_state.tempo_target_pct)
                * lead_cues.bar_ratio())

    def planner_loop():
        nonlocal composed_bpm, negotiator
        t0 = 0.0
        tempo_from_loop = False
        if tap_start:
            from .bandentry import first_bar_time, tap_tempo  # noqa: PLC0415
            while not stop.is_set() and not ending.is_set():
                with state_lock:
                    taps = shared["taps"]
                    loop_tempo = shared.get("loop_tempo")
                if taps or loop_tempo:
                    break
                stop.wait(0.01)
            else:
                return
            if loop_tempo:
                # The loop set it: bar 0 is the top of the loop, on the grid its
                # own length makes.
                tempo_from_loop = True
                composed_bpm = loop_tempo.bpm
                next_beat = loop_tempo.next_line_s
            else:
                composed_bpm, _ = tap_tempo(taps)
                # The first knock is beat one, so the drums land on knock nine:
                # the third bar line of the count, or the next one if that has gone.
                next_beat = taps[0] + 8.0 * 60.0 / composed_bpm
            if style_pick is not None:
                picked = style_pick.at_count(composed_bpm)
                if picked != style:
                    change_style(picked)         # no bar has been rendered yet
            negotiator = RoleNegotiator(role, composed_bpm=composed_bpm,
                                        composed_key=composed_key)
            period = 60.0 / composed_bpm
            t0 = first_bar_time(next_beat, 4.0 * period,
                                timeline.playhead / SR, comp_samples[0] / SR)
            last_decision["tapped"] = composed_bpm
        song_start[0] = t0
        # --from-bar: N bars into the map and into the bar count, but the first
        # bar still sounds now.  The clock is stream time, not map position.
        clock = MachineClock(composed_bpm, t0=t0, first_bar=int(from_bar))
        state = {"ref_db": None, "pending": [], "ended": False,
                 "bar_starts": [], "sec_idx": None, "sec_bar": 0,
                 "tempo_from_loop": tempo_from_loop}
        if from_bar > 0:
            state["sec_idx"], pos = mind.seek_form(sess, int(from_bar))
            state["sec_bar"] = pos - 1  # the first plan steps onto ``pos``
        bpm = composed_bpm
        pace = _StreamPace()
        while not stop.is_set() and not ending.is_set():
            # Stay one bar ahead of the playhead: render bar N while N-1 plays.  A
            # take whose band follows her plans later, half a bar ahead when planning
            # is quick enough, so it has heard half of N-1 (see plan_lead_s).
            ahead = clock.bar_len_s(bpm)
            if entry is not None:
                ahead = plan_lead_s(ahead, plan_cost_s=_median_s(plan_us),
                                    pace=pace.rate(), comp_s=comp_samples[0] / SR)
            state["ahead_s"] = ahead
            if clock.t_s - timeline.playhead / SR > ahead:
                pace.note(timeline.playhead / SR)
                stop.wait(0.01)
                continue
            t0_wall = time.perf_counter()
            bpm = plan_one_bar(clock, state, bpm)
            plan_us.append((time.perf_counter() - t0_wall) * 1e6)
            if state["ended"]:
                ending.set()
        tail = duet.flush_tail()
        if tail.shape[0]:
            timeline.write_at(timeline.end, tail)
        if looper.active:
            timeline.write_at(int(round(clock.t_s * SR)) - comp_samples[0],
                              looper.mix_bar(clock.bar, DECLICK_SAMPLES, stop=True))

    def release_before_sound():
        if midi_in is not None:
            midi_in.close()
        cues.close()
        if recorder is not None:
            recorder.close()
        unregister_session()

    def refuse_pair(e: Exception):
        # Two devices have two clocks, and some pairs will not open together.  Nothing has
        # played yet, so this is a sentence and not a traceback.
        release_before_sound()
        raise RealtimeError(f"{OPEN_TOGETHER_FAILED}: {e}") from None

    try:
        stream = sd.Stream(
            samplerate=SR, blocksize=BLOCK, device=dev, channels=(route.channels, 2),
            dtype="float32", latency="low", callback=callback,
        )
    except Exception as e:  # noqa: BLE001 - PortAudio's error, or sounddevice's own
        if output_device is None:
            release_before_sound()
            raise
        refuse_pair(e)
    # v1.6 — see jam(): masking is aligned to the bus->speaker->mic round trip,
    # and it matters more here because the machine plays continuously.
    masker.set_latency_hops(stream_latency_hops(stream))
    if midi_in is not None:
        midi_in.offset_s = _stream_input_latency_s(stream)
        midi_in.start()
    calibrated_ms = None
    if tap_start:
        # She plays to what she hears from her own tap, so bars must sound on
        # the tapped grid: place them early by the measured round trip (jam's
        # compensation, same resolution order).  Other lead takes own their
        # clock and have no human beat to land on.
        from .calibrate import (  # noqa: PLC0415
            device_names, load_roundtrip_ms, placement_comp_samples)
        in_name, out_name = device_names(sd, dev)
        lat = getattr(stream, "latency", None)
        reported = (float(lat[0]) + float(lat[1])
                    if isinstance(lat, (tuple, list)) and len(lat) == 2 else None)
        calibrated_ms = load_roundtrip_ms(in_name, out_name, SR)
        comp_samples[0] = placement_comp_samples(None, calibrated_ms, reported, SR)
    if recorder is not None:
        recorder.stream_opened(stream_report(stream, calibrated_ms=calibrated_ms))
    # Where a bar comes back through the mic, which is what a take is cut against.
    looper.roundtrip_samples = stream_latency_hops(stream) * HOP
    if tap_start:
        looper.roundtrip_samples = comp_samples[0]
    if output_device is not None:
        # A pair can open and then refuse to run.  Starting it here says so in the same words;
        # entering the stream below starts it again, which does nothing to one already running.
        try:
            stream.start()
        except Exception as e:  # noqa: BLE001 - PortAudio's error, or sounddevice's own
            try:
                stream.close()
            except Exception:  # noqa: BLE001 - it never ran, and the refusal is what matters
                pass
            refuse_pair(e)
    analysis = threading.Thread(target=analysis_loop, name="plume-analysis",
                                daemon=True)
    planner = threading.Thread(target=planner_loop, name="plume-planner",
                               daemon=True)
    if status:
        tempo = "tempo from your knocks" if tap_start else f"{composed_bpm:.1f} bpm"
        print(f"lead: {sess_name or '?'} style={shown_style} "
              f"{tempo} role={role.role} — cues: {cues.path} "
              f"— {_ending_banner(cues, max_minutes)}", flush=True)
    why = END_FORM        # the planner setting `ending` means the map ran out
    status_line = StatusLine(enabled=bool(status))
    hard = HardExit(on_exit=_hard_exit_cleanup(
        recorder, cues, lambda: _ending_doc(END_WEDGED,
                                            started_mono=started_mono,
                                            max_minutes=max_minutes,
                                            bars=n_decisions[0])))
    try:
        with GracefulSignals(on_repeat=lambda _s: hard.arm(
                END_INTERRUPT, seconds=HARD_EXIT_GRACE_S)) as signals, stream:
            analysis.start()
            planner.start()
            try:
                while not ending.is_set():
                    reason = _live_stop_reason(cues, stop_deadline,
                                               signals=signals)
                    if reason is not None:
                        why = reason
                        break
                    for note in looper.pop_notes():
                        status_line.update(note)
                        status_line.close()     # keep it, on a line of its own
                    d = last_decision.get("d")
                    if d is not None:
                        status_line.update(
                            f"[bar {d['bar']:>4}] {d['bpm']:6.1f} bpm  "
                            f"{d['role']:<12} clock "
                            f"{d['clock_authority']:4.2f}  "
                            f"{str(d['key'] or '?'):<10}"
                            + (f" {d['entry']}" if "entry" in d else "")
                            + (f" {d['style']}?" if d.get("style_guess") else ""))
                    elif tap_start:
                        status_line.update(
                            "[waiting] press a key 4 even beats" if midi_in is not None
                            else "[waiting] knock 4 even beats")
                    time.sleep(0.25)
            except KeyboardInterrupt:
                why = END_INTERRUPT
            if why == END_FORM:
                why = shared.get("end_reason", why)
            hard.arm(why)          # v1.3: the ending is now on a clock
            status_line.close()
            if status:
                print(f"ending — {why}; finishing the bar...", flush=True)
            # Graceful ending, INSIDE the stream: let what is already
            # scheduled (the current bar plus the tail) actually play.  Once
            # the stream is closed the playhead stops moving and this wait
            # becomes a nap that plays nothing.
            ending.set()
            planner.join(timeout=PLANNER_JOIN_S)
            _drain_timeline(timeline)
            _abort_stream(stream)
    finally:
        ending.set()
        stop.set()
        if planner.is_alive():
            planner.join(timeout=PLANNER_JOIN_S)
        if analysis.is_alive():
            analysis.join(timeout=ANALYSIS_JOIN_S)
        if midi_in is not None:
            midi_in.close()
        cues.close()
        status_line.close()
        if recorder is not None:
            recorder.finish(_ending_doc(why, started_mono=started_mono,
                                        max_minutes=max_minutes,
                                        bars=n_decisions[0]))
        unregister_session()
        hard.disarm()

    if recorder is not None and status:
        print(f"recorded to {rec_path}", flush=True)
        if recorder.errors:
            print(f"recording had {len(recorder.errors)} problem(s): "
                  f"{recorder.errors[0]}", flush=True)
    if status and plan_us:
        med = sorted(plan_us)[len(plan_us) // 2]
        print(f"planning: {med:.0f} us/bar median over {len(plan_us)} bars",
              flush=True)
