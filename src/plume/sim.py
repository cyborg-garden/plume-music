"""plume.sim — the offline twin (SPEC-V1 section 6). THE verification surface.

``simulate`` streams a WAV through the identical listen -> mind -> duet code
path the real-time transport uses, deterministic and faster than real time:

read WAV (mono-sum) -> 4096-sample blocks -> FeatureExtractor -> BeatTracker
(with ``taps_s`` applied as a count-in when given) / HarmonyTracker -> at each
predicted bar boundary: summarize :class:`~plume.mind.BarFeatures` for the
completed human bar, ``Mind.plan_bar``, ``DuetEngine.render_bar`` for the
upcoming bar, place the rendered bar at its predicted start time on the
machine bus (sample-accurate) -> mix human (at ``mix_human_db``) + machine,
soft-limit at -1 dB -> PCM_24 WAV.

While the tracker is unlocked, ``Mind.plan_bar`` is ticked once per second
with ``human=None`` (per SPEC-V1 section 4) and nothing is rendered — the
machine bus stays silent until the first locked bar boundary.  Once locked,
``render_bar`` is called for EVERY boundary plan (even all-tacet ones) so the
duet engine's overlap/tail buffer keeps draining into the right bar.

Decisions NDJSON: one line per plan — ``{bar, t_s, state, reason, bpm, conf,
key, chord, roles, gain_db, register_ceiling_midi, machine_density}`` — plus
one ``{"event": ...}`` line for lock / unlock / key changes, in stream order.
``machine_density`` is the MACHINE's chosen note density (0..1); the human's
measured density is a different quantity in different units (onsets per beat)
and appears only inside ``reason``.  It was called ``density`` before v1.2,
which put both under one name on one line.

Determinism: byte-identical output (WAV + NDJSON) for identical
(input, args); no wall-clock, no global rng.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from dataclasses import replace as _replace

import numpy as np
import soundfile as sf

from plume import fx as _fx
from plume import mind as _mind_mod
from plume.duet import DuetEngine
from plume.listen.beat import BeatState, BeatTracker
from plume.listen.features import FeatureExtractor
from plume.listen.harmony import HarmonyState, HarmonyTracker
from plume.mind import BarFeatures, BarPlan, Mind, MindParams
# --- v2 (SPEC-V2 section 4).  ``roles``/``score``/``theory`` are new imports
#     for lead mode only; nothing in ``simulate`` touches them.
from plume.mind import Cue as Cue           # noqa: F401  (re-export for callers)
from plume.mind import FormParams, flatten_sections
from plume.mind import parse_cue as _parse_cue
from plume.roles import (
    CONCEDE_MIN_CLOCK as _CONCEDE_MIN_CLOCK,
)
from plume.roles import RoleNegotiator, RoleParams, onset_phase_offset_ms
from plume.styles_lib import load_style as _load_style
from plume.theory import Key as _Key
from plume.theory import parse_chord as _parse_chord
from plume.theory import parse_key as _parse_key
from plume.theory import spell as _spell

__all__ = ["SimResult", "simulate", "simulate_lead", "BLOCK",
           "InputModeDecision", "classify_input", "resolve_input_mode",
           "INPUT_MODES"]

BLOCK = 4096                 # streaming block size, samples
# Plan states in which the machine is still making sound (mirror of
# mind._ACTIVE_STATES; sim renders these even off a bar boundary).
_SOUNDING_STATES = frozenset({"entering", "comping", "thinned", "fading"})
_UNLOCKED_TICK_S = 1.0       # Mind tick cadence while unlocked
_LIMIT_DB = -1.0


@dataclass
class SimResult:
    out_path: str
    decisions_path: str
    duration_s: float
    entered_bar: int | None
    states_by_bar: list[str]
    bpm_trace: list[tuple[float, float]]        # (t_s, bpm)
    final_key: str | None
    manifest: dict = field(default_factory=dict)
    # --- v2 (SPEC-V2 section 4), additive and defaulted: ``simulate`` leaves
    #     all four empty and every v1 construction still works.  These are the
    #     "SimResult carries the same" half of the NDJSON contract — one entry
    #     per MACHINE bar, index-aligned with ``bar_times_s``.
    role_states: list[dict] = field(default_factory=list)
    form_plans: list[dict] = field(default_factory=list)
    bar_times_s: list[float] = field(default_factory=list)
    machine_bpm_by_bar: list[float] = field(default_factory=list)


# =====================================================================
# INPUT-MODE DISCRIMINATOR (v1.5.2) — which front end should hear this take?
#
# ``BeatTracker(acoustic=True)`` exists because sustained open-tuned acoustic
# guitar defeats the default flux front end (167.9 bpm against a true 84 on
# the fingerpick fixture — the live session's exact wrong number), and the
# default exists because the acoustic front end drifts on the corpus's own
# long quiet takes (see beat.py's ACOUSTIC MODE block for the full A/B).
# Nothing could SELECT between them: every entry point constructed
# ``BeatTracker()`` with defaults.  ``--input acoustic|programmed|auto`` is
# the selector; this is the measurement behind ``auto``.
#
# WHAT SEPARATES THE MATERIAL — measured 2026-08-19 on both corpora (all four
# programmed renders loud and 20 dB down, both clicks, all six fixture
# humans, all four joiners, all three sustained-acoustic fixtures loud and
# 20 dB down, plus a count-in-prepended acoustic take; the calibration table
# lives in tests/test_acoustic.py's confusion-matrix test).  Flux crest and
# RMS spread — the two properties the acoustic fixture generator is built
# around — do NOT separate: synthwave100's sustained pads sit at crest 1.75 /
# spread 7.2 dB, inside the acoustic cluster.  What separates cleanly is what
# the TRANSIENTS look like, read off the banded attack function that was
# built as an impulse-vs-note discriminator (features.ATTACK_*):
#
#   metric (first 12 s)                          programmed      acoustic
#   ------------------------------------------   -----------    ----------
#   p75 of attack_breadth at strong attacks      0.500 - 1.000  0.208-0.375
#   fraction of frames with breadth >= 0.55      0.024 - 0.065  0.002-0.013
#
# Machine transients (drum hits, clicks, synth stabs) lift half the spectrum
# or more at once; a plucked or strummed string lifts only the bands its
# harmonics land in.  Both metrics are level-independent (`attack` is a log
# ratio): the loud and -20 dB rows of every case measure identically, which
# neither flux nor RMS statistics can claim.  Thresholds sit in the measured
# gaps (0.375 -> 0.500 and 0.013 -> 0.024).  Both axes must agree to call
# the take acoustic; a split verdict keeps the programmed default and says
# so out loud — a wrong silent guess is worse than a hint.
# =====================================================================

INPUT_MODES = ("auto", "acoustic", "programmed")
INPUT_PROBE_S = 12.0            # seconds of audio the classifier reads
INPUT_PROBE_MIN_AUDIBLE = 0.10  # below this fraction of audible frames: don't guess
INPUT_SILENT_FRAME_DB = -60.0   # a frame below this is not evidence of anything
# The two calibrated axes (table above).  "Strong attack" deliberately uses
# the tracker's own picker shape (BEAT_ATTACK_MULT x median + floor, applied
# to the whole probe window rather than a trailing one — the probe is a batch
# measurement, not a causal stream).
INPUT_BREADTH_STRONG_MAX = 0.44   # acoustic at/below; programmed above
INPUT_BROAD_BREADTH = 0.55        # a frame this broadband is machine-shaped...
INPUT_BROAD_FRAC_MAX = 0.018      # ...and this many of them means programmed
_INPUT_HINT = ("input looks like sustained acoustic material in part; if this "
               "is fingerpicked/strummed acoustic guitar, pass "
               "--input acoustic")


@dataclass(frozen=True)
class InputModeDecision:
    """What ``--input auto`` decided for a take, and the numbers behind it."""
    mode: str                       # "acoustic" | "programmed"
    reason: str                     # one human-readable line
    hint: str | None                # set when programmed was kept on a split verdict
    breadth_strong: float | None    # p75 attack_breadth at strong attacks
    broad_frac: float | None        # fraction of frames with breadth >= 0.55
    audible_frac: float             # fraction of probe frames above the floor


def classify_input(audio: np.ndarray, sr: int, *,
                   probe_s: float = INPUT_PROBE_S) -> InputModeDecision:
    """Classify a mono take's first seconds: acoustic or programmed front end.

    Deterministic, gain-independent, and it reads only ``probe_s`` seconds no
    matter how long the take is.  See the block comment above for the
    calibration; the confusion matrix over both corpora is pinned as a test
    in ``tests/test_acoustic.py``.
    """
    from plume.listen.beat import (  # noqa: PLC0415 - shared constants
        BEAT_ATTACK_FLOOR, BEAT_ATTACK_MULT,
    )

    x = np.asarray(audio, dtype=np.float32).reshape(-1)[: int(probe_s * sr)]
    fe = FeatureExtractor(sr=int(sr))
    frames = []
    for i in range(0, x.shape[0], BLOCK):
        frames.extend(fe.process(x[i:i + BLOCK]))
    frames.extend(fe.flush())
    if not frames:
        return InputModeDecision(
            mode="programmed", hint=None, breadth_strong=None, broad_frac=None,
            audible_frac=0.0,
            reason="probe window is empty; keeping the programmed front end")
    rms = np.asarray([f.rms_db for f in frames], dtype=np.float64)
    att = np.asarray([f.attack for f in frames], dtype=np.float64)
    br = np.asarray([f.attack_breadth for f in frames], dtype=np.float64)
    audible = float((rms > INPUT_SILENT_FRAME_DB).mean())
    if audible < INPUT_PROBE_MIN_AUDIBLE:
        return InputModeDecision(
            mode="programmed", hint=None, breadth_strong=None, broad_frac=None,
            audible_frac=audible,
            reason=f"probe window is mostly silent ({audible:.0%} of frames "
                   f"audible); keeping the programmed front end")
    strong = att > max(BEAT_ATTACK_MULT * float(np.median(att)),
                       BEAT_ATTACK_FLOOR)
    br_strong = (float(np.percentile(br[strong], 75)) if bool(strong.any())
                 else 0.0)
    broad_frac = float((br >= INPUT_BROAD_BREADTH).mean())
    narrow = br_strong <= INPUT_BREADTH_STRONG_MAX
    sparse = broad_frac <= INPUT_BROAD_FRAC_MAX
    nums = (f"attack breadth at transients {br_strong:.3f} "
            f"(acoustic <= {INPUT_BREADTH_STRONG_MAX}), broadband-frame "
            f"share {broad_frac:.3f} (acoustic <= {INPUT_BROAD_FRAC_MAX})")
    if narrow and sparse:
        return InputModeDecision(
            mode="acoustic", hint=None, breadth_strong=br_strong,
            broad_frac=broad_frac, audible_frac=audible,
            reason=f"sustained-acoustic signature: {nums}")
    if narrow or sparse:
        return InputModeDecision(
            mode="programmed", hint=_INPUT_HINT, breadth_strong=br_strong,
            broad_frac=broad_frac, audible_frac=audible,
            reason=f"mixed signature, keeping the programmed front end: {nums}")
    return InputModeDecision(
        mode="programmed", hint=None, breadth_strong=br_strong,
        broad_frac=broad_frac, audible_frac=audible,
        reason=f"machine-transient signature: {nums}")


def resolve_input_mode(input_mode: str | None, audio: np.ndarray,
                       sr: int) -> tuple[bool, dict]:
    """``--input`` value -> (acoustic flag, manifest fields).

    ``auto`` defers first to an explicit ``PLUME_ACOUSTIC`` truthy environment
    (a user statement, exactly as documented in beat.py) and then to
    :func:`classify_input`.  Explicit ``acoustic``/``programmed`` beat both,
    matching ``BeatTracker``'s own "an explicit argument ignores the
    environment" rule.
    """
    from plume.listen.beat import _acoustic_from_env  # noqa: PLC0415

    mode = "auto" if input_mode is None else str(input_mode)
    if mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {'/'.join(INPUT_MODES)}, "
                         f"got {input_mode!r}")
    if mode == "acoustic":
        return True, {"input_mode": mode, "input_mode_resolved": "acoustic",
                      "input_mode_reason": "selected with --input acoustic",
                      "input_mode_hint": None}
    if mode == "programmed":
        return False, {"input_mode": mode, "input_mode_resolved": "programmed",
                       "input_mode_reason": "selected with --input programmed",
                       "input_mode_hint": None}
    if _acoustic_from_env():
        return True, {"input_mode": mode, "input_mode_resolved": "acoustic",
                      "input_mode_reason": "PLUME_ACOUSTIC is set in the "
                                           "environment",
                      "input_mode_hint": None}
    dec = classify_input(audio, sr)
    return dec.mode == "acoustic", {
        "input_mode": mode, "input_mode_resolved": dec.mode,
        "input_mode_reason": dec.reason, "input_mode_hint": dec.hint}


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-10)))


def _round(x: float | None, nd: int = 4):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, nd)


_BAR_EDGE_TRIM_S = 0.15    # trailing slop excluded from the silence measure
# SILENCE_REL_DB — v1.2 AMENDMENT (SPEC-V1 section 4/6).  ``silence_rms_db``
# is an ABSOLUTE dBFS floor, and an absolute floor cannot decide whether a
# human is playing.  Test fixtures are peak-normalized to 0.85 with bar mean
# RMS -15..-24 dBFS (21-30 dB of headroom over the -45 floor) and their silent
# bars are exact digital zeros, so the whole 40 dB band where real recordings
# live was never generated and the floor was unreachable by construction.  On
# a real input the same floor fires constantly, because a sparse/percussive
# bar's MEAN RMS runs 13-25 dB below its peak frames: a 120bpm click at
# -21 dBFS peak, with FOUR detected onsets per bar and the tracker locked at
# conf 0.93, measured -52.5 dB mean and was flagged silent on every bar.  The
# bar-silence test is therefore now (a) onset-guarded FIRST — detected onsets
# are level-independent evidence that the human is playing — and (b) relative
# to the human's own running level as well as the absolute floor.
SILENCE_REL_DB = 15.0
# 15, not 25: the `stops` fixture's resume-bleed bar measures -32.9 dB against
# a running reference of -17.5 (a 15.4 dB drop).  At 25 that bar stops reading
# silent and v1 behavior 3's single `thinned` bar disappears.


class _BarAccum:
    """Frame-level stats for the human bar in progress.

    Frames are kept individually so the SILENCE measure can exclude the
    trailing ``_BAR_EDGE_TRIM_S`` of the bar: a predicted boundary that runs
    a few frames late catches the first strum of a resume, and counting that
    edge bleed would disguise a musically silent bar as playing.  Onset
    density and RMS still cover the full bar.
    """

    def __init__(self) -> None:
        self.reset(0.0)

    def reset(self, t: float) -> None:
        self.t0 = t
        self.frames: list[tuple[float, float, bool]] = []   # (t_s, rms_db, onset)

    def add(self, frame, silence_rms_db: float) -> None:
        del silence_rms_db          # measured at summarize time (needs t_b)
        self.frames.append((float(frame.t_s), float(frame.rms_db),
                            bool(frame.onset)))

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def onsets(self) -> int:
        return sum(1 for _, _, o in self.frames if o)

    def rms_db(self, t0: float | None = None) -> float:
        frames = self.frames if t0 is None else [
            f for f in self.frames if f[0] > t0]
        if not frames:
            return -90.0
        pow_sum = sum((10.0 ** (r / 20.0)) ** 2 for _, r, _ in frames)
        return _db(math.sqrt(pow_sum / len(frames)))

    def core(self, t_b: float, t0: float | None = None):
        """The bar minus its trailing ``_BAR_EDGE_TRIM_S`` (see class doc)."""
        frames = self.frames if t0 is None else [
            f for f in self.frames if f[0] > t0]
        core = [(r, o) for t, r, o in frames if t <= t_b - _BAR_EDGE_TRIM_S]
        return core or [(r, o) for _, r, o in frames]

    def is_silent(self, t_b: float, silence_rms_db: float,
                  ref_db: float | None = None,
                  t0: float | None = None) -> bool:
        """Is this human bar musically silent?

        v1.2 rule (see ``SILENCE_REL_DB``), in order:

        1. **Detected onsets in the edge-trimmed core => NOT silent.**  Onsets
           are level-independent evidence of playing, and they were already
           computed and then ignored on the RMS path.  This is what makes a
           click track / kick stem / any sparse percussive input read as the
           playing it is.
        2. Otherwise the bar is silent if its RMS is under the effective
           threshold — the absolute ``silence_rms_db`` floor, raised to
           ``ref_db - SILENCE_REL_DB`` once the human's running level is
           known — or if the core is >=80% frames under that threshold (tail
           ring into a stop).
        """
        core = self.core(t_b, t0)
        if not core:
            return True
        if any(o for _, o in core):
            return False
        thr = silence_rms_db
        if ref_db is not None:
            thr = max(thr, ref_db - SILENCE_REL_DB)
        if self.rms_db(t0) < thr:
            return True
        quiet = sum(1 for r, _ in core if r < thr)
        return quiet / len(core) >= 0.8


def _summarize(acc: _BarAccum, bar: int, t_b: float, interval: float,
               hstate, params: MindParams,
               ref_db: float | None = None) -> BarFeatures:
    """One completed human bar -> BarFeatures.

    v1.2: the measured span is capped at ONE BAR (``t0`` below).  The
    accumulator is never reset before the tracker's first predicted boundary,
    so the whole pre-lock span used to be summarized as a single bar —
    measured at 11.9s / 1379 frames of real music credited as "1 bar", which
    diluted its RMS and onset density and made ``min_listen_bars`` mean "4
    bars after an internal lock event the user cannot see" instead of "4 bars
    heard".  See ``_prelock_bars`` for the bars that elapsed before it.
    """
    bar_len = interval * 4.0 if math.isfinite(interval) else None
    t0 = None
    if bar_len is not None and t_b - acc.t0 > bar_len * 1.5:
        t0 = t_b - bar_len
    span = max(t_b - (acc.t0 if t0 is None else t0), 1e-9)
    beats = max(1.0, span / interval) if math.isfinite(interval) else 4.0
    core = acc.core(t_b, t0)
    onsets = sum(1 for _, o in core if o) if t0 is not None else acc.onsets
    return BarFeatures(
        bar=bar,
        rms_db=acc.rms_db(t0),
        onset_density=onsets / beats,
        silent=acc.is_silent(t_b, params.silence_rms_db, ref_db, t0),
        register_low_midi=hstate.register_low_midi,
        register_high_midi=hstate.register_high_midi,
    )


def _prelock_bars(acc: _BarAccum, t_b: float, interval: float,
                  params: MindParams, ref_db: float | None) -> int:
    """Whole non-silent human bars that elapsed BEFORE the first boundary.

    v1.2 AMENDMENT (SPEC-V1 section 4).  ``min_listen_bars`` is specified as
    "non-silent human bars heard", but the only call site that ever passed a
    ``BarFeatures`` was the locked-boundary path, so bars heard before the
    tracker locked could never count.  Measured on a ``plume render`` lo-fi
    take: the tracker locked at t=16.7s of a 49.5s piece and the sideman
    entered at t=25.1s — 50% into the take.  Those bars were heard; they are
    now credited (via ``Mind.credit_listened_bars``, which counts them WITHOUT
    producing plans, so entry still requires a real locked boundary).
    """
    if not math.isfinite(interval) or interval <= 0.0:
        return 0
    bar_len = interval * 4.0
    n = int((t_b - acc.t0) / bar_len) - 1        # excl. the bar being summarized
    if n <= 0:
        return 0
    heard = 0
    for i in range(n):
        a, b = acc.t0 + i * bar_len, acc.t0 + (i + 1) * bar_len
        sub = _BarAccum()
        sub.reset(a)
        sub.frames = [f for f in acc.frames if a < f[0] <= b]
        if sub.frames and not sub.is_silent(b, params.silence_rms_db, ref_db):
            heard += 1
    return heard


def _plan_line(plan, t_s: float, state, hstate) -> dict:
    return {
        "bar": int(plan.bar),
        "t_s": _round(t_s),
        "state": str(plan.state),
        "reason": str(plan.reason),
        "bpm": _round(state.bpm),
        "conf": _round(state.confidence),
        "key": hstate.key,
        "chord": hstate.chord,
        "roles": {str(k): _round(v) for k, v in sorted(plan.roles.items())},
        "gain_db": _round(plan.gain_db),
        # Additive to the spec'd line keys: behavior test 7 verifies register
        # avoidance "via decisions", which needs the ceiling in the log.
        "register_ceiling_midi": _round(plan.register_ceiling_midi),
        # v1.3 AMENDMENT 24 (SPEC-V1 section 8): this was named "density", which
        # collided with the OTHER density on the same line -- the human's
        # measured onset rate, which appears inside `reason` ("human busy (7.9
        # onsets/beat > 2.5)").  Two different quantities in two different
        # units under one name on one line.  This is the MACHINE's chosen
        # note density for the bar, 0..1; the human's is onsets per beat and
        # lives only in the reason text.  BarPlan.density keeps its name --
        # the ambiguity was only ever in the log.
        "machine_density": _round(plan.density),
    }


def simulate(input_wav: str, *, style: str = "lofi", out_path: str,
             seed: int = 0, params: MindParams | None = None,
             tightness: float | None = None, mix_human_db: float = 0.0,
             taps_s: list[float] | None = None,
             bpm_hint: float | None = None,
             decisions_path: str | None = None,
             input_mode: str | None = "auto") -> SimResult:
    """Run the offline duet pipeline. See module docstring.

    ``decisions_path`` (additive deviation from the frozen signature — the
    CLI's ``--decisions`` flag needs it) defaults to ``out_path`` with the
    extension replaced by ``.ndjson``.

    ``bpm_hint`` (v1.2, additive) states a KNOWN tempo and narrows the beat
    tracker's tempo prior around it.  Unlike ``taps_s`` it makes no claim
    about phase, which is why it works on material the user did not cut on a
    downbeat — the case where a count-in deadlocks.  It is the honest answer
    for genuinely ambiguous material (half-time drum feels, where the
    beat-level flux autocorrelation is negative and a half-tempo reading is a
    legitimate perceptual answer that simply is not the notated one).

    ``input_mode`` (v1.5.2, additive) selects the beat tracker's front end:
    ``"acoustic"``, ``"programmed"``, or ``"auto"`` (the default), which
    defers to a truthy ``PLUME_ACOUSTIC`` environment and otherwise measures
    the take's first seconds with :func:`classify_input`.  The resolution and
    its reason land in ``manifest['input_mode_resolved']`` /
    ``['input_mode_reason']`` (plus ``['input_mode_hint']`` when auto kept
    the programmed default on a split verdict).  On programmed material the
    resolved tracker is exactly the pre-v1.5.2 one, so existing corpora are
    untouched byte for byte.
    """
    params = params if params is not None else MindParams()
    if decisions_path is None:
        base, _ = os.path.splitext(str(out_path))
        decisions_path = base + ".ndjson"

    human, sr = sf.read(str(input_wav), dtype="float32", always_2d=True)
    human = human.mean(axis=1).astype(np.float32)      # mono-sum

    acoustic, input_mode_fields = resolve_input_mode(input_mode, human, sr)

    fe = FeatureExtractor(sr=sr)
    hint = None if bpm_hint is None else float(bpm_hint)
    tracker = (BeatTracker(bpm_hint=hint, acoustic=acoustic)
               if tightness is None
               else BeatTracker(tightness=float(tightness), bpm_hint=hint,
                                acoustic=acoustic))
    harmony = HarmonyTracker()
    mind = Mind(params)
    from .governor import HarmonyGovernor  # noqa: PLC0415 — lazy, like siblings
    governor = HarmonyGovernor(style, params)
    duet = DuetEngine(style=style, seed=int(seed), sr=sr)

    taps = sorted(float(t) for t in (taps_s or []))
    tap_i = 0

    lines: list[dict] = []
    bpm_trace: list[tuple[float, float]] = []
    states_by_bar: dict[int, str] = {}
    entered_bar: int | None = None
    placements: list[tuple[int, np.ndarray]] = []      # (sample offset, stereo)
    next_place = 0                                     # sample cursor after last bar

    acc = _BarAccum()
    prev_state = None
    prev_key: str | None = None
    last_bar = -2                                      # sentinel: no boundary yet
    last_tick = -_UNLOCKED_TICK_S

    # Machine-bus calibration: BarPlan.gain_db is RELATIVE TO THE HUMAN'S level
    # (SPEC-V1 section 4: "enter_under_db: first-entry gain relative to human's
    # bar RMS"; the mind expects sim/realtime to calibrate the buses).  The bus
    # gain is set from the freshly rendered bar so its RMS lands at
    # human_ref + plan.gain_db, at two moments per arc: the (re-)entry bar
    # (so the machine truly enters enter_under_db below the human) and the
    # first comping bar (so the full texture breathes WITH the human instead
    # of jumping the texture delta above it).  Between those, and through
    # thinned/fading, the factor stays fixed — the mind's dynamic-arc gain and
    # the plan's role/density choices do the tracking.
    human_ref_db: float | None = None                  # EMA of non-silent bar RMS
    cal_lin = 1.0
    prev_plan_state = "waiting"
    # Observability (postmortem finding 1: "plume cannot account for its own
    # behaviour").  The calibration is the single number that decides how
    # loud the machine is against the human for the whole take, and it was
    # wrong by +3.2 dB for a release without leaving a trace anywhere.  It
    # goes in the manifest.
    cal_db_applied: float | None = None
    cal_bar: int | None = None
    cal_trace: list[dict] = []

    first_boundary = True

    def do_boundary(t_b: float, state, hstate) -> None:
        nonlocal entered_bar, next_place, human_ref_db, cal_lin
        nonlocal prev_plan_state, first_boundary, cal_db_applied, cal_bar
        # v2.6: the governed harmony is what BOTH the mind and the renderer
        # believe (see plume.governor) — parity with the live transport.
        hstate = governor.govern(getattr(state, "bar", None), hstate)
        if first_boundary:
            first_boundary = False
            pre = _prelock_bars(acc, t_b, state.beat_interval_s, params,
                                human_ref_db)
            if pre:
                mind.credit_listened_bars(pre)
        bf = _summarize(acc, state.bar - 1, t_b, state.beat_interval_s,
                        hstate, params, human_ref_db)
        if not bf.silent:
            human_ref_db = (bf.rms_db if human_ref_db is None
                            else 0.5 * bf.rms_db + 0.5 * human_ref_db)
        plan = mind.plan_bar(state, hstate, bf)
        lines.append(_plan_line(plan, t_b, state, hstate))
        states_by_bar[int(plan.bar)] = str(plan.state)
        if entered_bar is None and str(plan.state) == "entering":
            entered_bar = int(plan.bar)
        bar_audio = duet.render_bar(plan, state, hstate)
        st = str(plan.state)
        recalibrate = (
            (st == "entering" and prev_plan_state != "entering")
            or (st == "comping" and prev_plan_state == "entering"))
        if recalibrate and human_ref_db is not None and bar_audio.size:
            # v1.4.1: measure the bar's own level, NOT the delivered bar.
            # The first `entering` bar is the first bar of the gentle
            # fade-in, so the delivered samples are the bar TIMES the fade
            # envelope; solving `rendered * cal == target` on those solves
            # for the wrong number and over-boosts the whole take by the
            # envelope's RMS (+3.2 dB overall / +1.35 dB steady-state on the
            # `steady` fixture).  It used to self-correct on the first
            # comping bar, but only if `entering -> comping` ever fired, and
            # since the commitment work the mind often goes
            # `entering -> thinned` instead, so the boost simply stayed.
            # `last_bar_ms_pre_env` is the same samples before the envelope,
            # so the calibration is identical fade or no fade.
            rendered = duet.last_bar_ms_pre_env
            if rendered > 0.0:
                rendered_db = 10.0 * np.log10(rendered)
                cal_db = (human_ref_db + float(plan.gain_db)) - rendered_db
                cal_db = max(-24.0, min(24.0, cal_db))
                cal_lin = 10.0 ** (cal_db / 20.0)
                cal_db_applied, cal_bar = cal_db, int(plan.bar)
                cal_trace.append({"bar": int(plan.bar), "state": st,
                                  "cal_db": _round(cal_db),
                                  "target_db": _round(human_ref_db
                                                      + float(plan.gain_db))})
        prev_plan_state = st
        if cal_lin != 1.0 and bar_audio.size:
            bar_audio = (bar_audio * np.float32(cal_lin)).astype(np.float32)
        i0 = max(0, int(round(t_b * sr)))
        placements.append((i0, bar_audio))
        next_place = i0 + bar_audio.shape[0]
        acc.reset(t_b)

    def feed(frames) -> None:
        nonlocal prev_state, prev_key, last_bar, last_tick, tap_i, next_place
        for f in frames:
            t = float(f.t_s)
            while tap_i < len(taps) and taps[tap_i] <= t:
                tracker.tap(taps[tap_i])
                tap_i += 1
            state = tracker.process([f])
            hstate = harmony.process([f])
            acc.add(f, params.silence_rms_db)

            if prev_state is not None and state.locked != prev_state.locked:
                lines.append({"event": "lock" if state.locked else "unlock",
                              "t_s": _round(t), "bpm": _round(state.bpm),
                              "conf": _round(state.confidence)})
            # v1.3: a locked tracker may re-interpret the metre, but never
            # invisibly (see beat._grid_hysteresis).  The counter is
            # monotone, so an increment between two polls cannot be missed.
            # v2.7: keyed on the WIDER counter — see cli.py's note.
            if prev_state is not None and (state.tempo_grid_changes
                                           != prev_state.tempo_grid_changes):
                _oct = (state.tempo_octave_changes
                        != prev_state.tempo_octave_changes)
                lines.append({"event": ("tempo_octave_change" if _oct
                                        else "tempo_grid_change"),
                              "t_s": _round(t),
                              "from_bpm": _round(prev_state.bpm),
                              "to_bpm": _round(state.bpm),
                              "conf": _round(state.confidence)})
            if hstate.key != prev_key:
                lines.append({"event": "key_change", "t_s": _round(t),
                              "key": hstate.key})
                prev_key = hstate.key

            if state.locked and state.bar > max(last_bar, -1):
                if prev_state is not None and math.isfinite(
                        prev_state.next_downbeat_t_s):
                    t_b = prev_state.next_downbeat_t_s
                else:  # just locked: current bar started phase_beats ago
                    t_b = max(0.0, t - state.phase_beats * state.beat_interval_s)
                do_boundary(t_b, state, hstate)
                last_bar = state.bar
            elif not state.locked and t - last_tick >= _UNLOCKED_TICK_S:
                # v2.6: unlocked ticks hold the governed chord (bar=None
                # never advances the progression or the streaks).
                hstate = governor.govern(None, hstate)
                plan = mind.plan_bar(state, hstate, None)
                lines.append(_plan_line(plan, t, state, hstate))
                last_tick = t
                # v1.2: an unlocked tick can carry a SOUNDING plan — the
                # beat-loss repair's "fading" bar (mind._plan_beat_loss) is
                # issued here, not at a boundary.  Nothing was ever rendered
                # on this path, so a mid-take rubato made the machine stop
                # dead instead of fading out, which is the one thing the
                # repair exists to avoid.
                if str(plan.state) in _SOUNDING_STATES:
                    # v1.4.1: `continuation=True`.  These bars are KEPT and
                    # placed end to end, but they carry a repeated bar number
                    # — the beat counter freezes on unlock, so the mind's up
                    # to four held wobble bars (mind._plan_unlock_hold) all
                    # come through as `beat.bar + 1`.  Without the flag the
                    # engine reads the repeat as the live planner revising a
                    # bar and rolls the fade back for each one, so a wobble
                    # during an entrance replays the same slice of the
                    # fade-in over and over: the entrance restarts instead of
                    # continuing.  A held bar is a continuation by
                    # definition; that is the whole point of holding it.
                    bar_audio = duet.render_bar(plan, state, hstate,
                                                continuation=True)
                    if bar_audio.size:
                        if cal_lin != 1.0:
                            bar_audio = (bar_audio
                                         * np.float32(cal_lin)).astype(np.float32)
                        i0 = max(next_place, max(0, int(round(t * sr))))
                        placements.append((i0, bar_audio))
                        next_place = i0 + bar_audio.shape[0]

            prev_state = state

    n_blocks = (human.shape[0] + BLOCK - 1) // BLOCK
    last_trace_t = -1.0
    final_state = None
    final_h = None
    for b in range(n_blocks):
        frames = fe.process(human[b * BLOCK:(b + 1) * BLOCK])
        feed(frames)
        if prev_state is not None:
            final_state = prev_state
            t = final_state.t_s
            if final_state.bpm > 0.0 and (
                    not bpm_trace or bpm_trace[-1][1] != final_state.bpm
                    or t - last_trace_t >= 0.5):
                bpm_trace.append((round(t, 4), round(final_state.bpm, 4)))
                last_trace_t = t
    feed(fe.flush())
    final_state = prev_state
    final_h = harmony.process([])

    tail = duet.flush_tail()
    if tail.shape[0]:
        if cal_lin != 1.0:
            tail = (tail * np.float32(cal_lin)).astype(np.float32)
        placements.append((next_place, tail))

    # ---- mix ----------------------------------------------------------------
    machine_end = max((i0 + a.shape[0] for i0, a in placements), default=0)
    total = max(human.shape[0], machine_end)
    machine = np.zeros((total, 2), dtype=np.float64)
    for i0, a in placements:
        machine[i0:i0 + a.shape[0]] += a.astype(np.float64)
    hgain = 10.0 ** (float(mix_human_db) / 20.0)
    human_st = np.zeros((total, 2), dtype=np.float64)
    human_st[: human.shape[0]] = (human.astype(np.float64) * hgain)[:, None]
    mix = human_st + machine
    pre_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    out = _fx.limiter(mix, _LIMIT_DB)

    out_path = str(out_path)
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    sf.write(out_path, out, sr, subtype="PCM_24")

    decisions_path = str(decisions_path)
    dparent = os.path.dirname(os.path.abspath(decisions_path))
    os.makedirs(dparent, exist_ok=True)
    with open(decisions_path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, sort_keys=True) + "\n")

    # ---- result -------------------------------------------------------------
    planned = sorted(b for b in states_by_bar if b >= 0)
    states_list: list[str] = []
    if planned:
        prev = "waiting"
        for b in range(planned[0], planned[-1] + 1):
            prev = states_by_bar.get(b, prev)
            states_list.append(prev)

    def _bus_rms(x: np.ndarray) -> float:
        return _db(math.sqrt(float(np.mean(x ** 2)))) if x.size else -90.0

    final_key = final_h.key if final_h is not None else None
    # One-line diagnosis when the duet never happened: the last waiting/
    # listening reason names the gate that stayed closed (beat never locked,
    # heard-bars unmet, ...) so the user is not sent to NDJSON forensics.
    no_entry_reason: str | None = None
    if entered_bar is None:
        for line in reversed(lines):
            if line.get("state") in ("waiting", "listening"):
                no_entry_reason = str(line["reason"])
                break
        if no_entry_reason is None:
            no_entry_reason = "no bar plans were made (input too short?)"
    manifest = {
        "input": str(input_wav),
        "out": out_path,
        "decisions": decisions_path,
        "style": str(style),
        "seed": int(seed),
        "sr": int(sr),
        "tightness": tightness,
        "mix_human_db": float(mix_human_db),
        "taps_s": taps,
        "bpm_hint": hint,
        **input_mode_fields,
        "duration_s": round(total / sr, 6),
        "peak_db": _round(_db(float(np.max(np.abs(out))) if out.size else 0.0)),
        "pre_limiter_peak_db": _round(_db(pre_peak)),
        "rms_db": _round(_bus_rms(out.astype(np.float64))),
        "per_bus_rms_db": {
            "human": _round(_bus_rms(human_st)),
            "machine": _round(_bus_rms(machine)),
        },
        "entered_bar": entered_bar,
        # The machine-bus calibration actually applied, and the bar it was
        # solved on.  It is measured on the bar's own level (pre-envelope),
        # so it is the same number whether or not a fade was running when it
        # was taken — see DuetEngine.last_bar_ms_pre_env.
        "machine_cal_db": _round(cal_db_applied),
        "machine_cal_bar": cal_bar,
        "machine_cal_trace": cal_trace,
        "no_entry_reason": no_entry_reason,
        "bars_planned": len(planned),
        "final_key": final_key,
        "final_bpm": _round(final_state.bpm) if final_state is not None else None,
        "params": asdict(params),
    }
    return SimResult(
        out_path=out_path,
        decisions_path=decisions_path,
        duration_s=total / sr,
        entered_bar=entered_bar,
        states_by_bar=states_list,
        bpm_trace=bpm_trace,
        final_key=final_key,
        manifest=manifest,
    )


# =====================================================================
# v2 — ``simulate_lead``: the offline twin, INVERTED (SPEC-V2 section 4).
#
# Everything below this line is ADDITIVE.  ``simulate`` above is untouched:
# the v1 behavior matrix and its byte-identical outputs are the contract this
# module exists to protect, so lead mode gets its own entry point, its own
# helpers (all prefixed ``_lead_`` / ``_Lead``) and its own NDJSON line
# builder.  Nothing in ``simulate`` reads anything defined here.
#
# The inversion, in one paragraph.  In v1 the HUMAN owns the clock: the beat
# tracker's predicted downbeats ARE the bar boundaries, the machine waits
# until it is confident, and it enters under the human.  In v2 lead mode the
# MACHINE owns the clock from t=0: the bar grid is generated, not detected,
# and the human's WAV is mixed in at ``human_offset_s`` — they are joining us.
# Features / beat / harmony still run on the HUMAN's signal only (in sim that
# is exact: we have the isolated WAV, and SPEC-V2 section 6 measured what
# happens to a caller who reads a summed bus instead — the offset it recovers
# is a function of the fader, not of the playing).
# =====================================================================

_ENV_HOP = 48                # 1 ms envelope hop for sub-hop attack refinement
_ATTACK_BACK_S = 0.060       # how far back of a detection an attack may sit
_ATTACK_FWD_S = 0.006
_ATTACK_FRAC = 0.15          # envelope fraction defining "the attack started"
_LEAD_SR = 48000             # sample rate when there is no human WAV to read
_LEAD_SUBDIV = 16            # the machine grid the human's phase is measured on

# --- the analysis fader trap (v2.1 amendment, SPEC-V2 section 4) ------------
# The human's LEVEL must not be an input to the negotiation.  Everything the
# negotiator reads about the human comes out of ``FeatureExtractor``, whose
# onset picker is an ABSOLUTE threshold (``median(flux) + ONSET_DELTA``, and
# ``ONSET_DELTA`` is 4.0 flux units, not a ratio).  Flux scales with the
# signal, so pulling a real human's fader down deletes their onsets — which
# is v1's exact bug wearing a different hat, and it was MEASURED here on ONE
# v0 lo-fi render at five gains with nothing about the playing changed:
# density 5.08 -> 4.21 -> 3.04 -> 1.94 -> 0.72 onsets/beat from 0 to -30 dB,
# the machine conceding four times at 0 dB and YIELDING THE LEAD at bar 12 at
# -6/-12/-20 dB.  Turning a player down handed them the band.
#
# The fix is to take the fader out before anybody listens: the analysis chain
# (features -> beat / harmony / attacks) runs on the human's signal scaled so
# that the take's OWN loud passages sit at a fixed reference.  A single
# constant over the whole take removes the fader exactly while leaving every
# dynamic INSIDE the take intact — which is the distinction that matters,
# because the fader is a constant and the playing is the variation.  The MIX
# is untouched: ``mix_human_db`` is still the human's fader, and it still only
# affects what you hear.
#
# A genuinely silent (or near-silent) take is left alone rather than amplified
# into a performance: below ``_ANA_FLOOR_DB`` there is nothing there to
# normalize, and "the human is not playing" must survive this transform.
_ANA_HOP = 480               # 10 ms envelope hop for the take's own level
_ANA_PCT = 90.0              # percentile of that envelope = "their loud bits"
# _ANA_REF_DB is MEASURED, not chosen.  Onset density against input gain, on
# four takes (v0 lo-fi / v0 trap / the on_grid and takes_over joiners), from
# +12 dB down to -30 dB: the JOINERS plateau at exactly their truth (1.00 and
# 2.50 onsets/beat) everywhere below about -6 dB and gain SPURIOUS onsets above
# it (on_grid reads 1.74 at +12 dB), while v0 material keeps accumulating
# low-level flux crossings all the way up (lo-fi 2.34 -> 5.73).  Normalizing UP
# therefore invents onsets and normalizing DOWN converges on the true attacks,
# so the reference sits at the low end: at -20 dBFS the joiners land on their
# plateau and v0 trap still measures ~5.1 onsets/beat, which is the number
# amendment A5 and `handoff_density_ratio` were written against.
_ANA_REF_DB = -20.0          # where those loud bits are placed for analysis
_ANA_FLOOR_DB = -60.0        # below this the take is silence, not a quiet take
_ANA_MAX_GAIN_DB = 60.0      # never lift noise by more than this

# Lead-mode density mapping: a section's composed ``energy`` (0..1) is what
# says how busy the machine should be, since in lead mode there is no entry
# ramp and no human level to size the texture against.
_LEAD_DENSITY_MIN = 0.15
_LEAD_DENSITY_SPAN = 0.65
_LEAD_BUSY_DENSITY = 0.30    # mirror of mind._DENSITY_BUSY (inverse coupling)
_LEAD_ARC_MIN_DB = -12.0     # mirror of mind._ARC_MIN_DB / _ARC_MAX_DB
_LEAD_ARC_MAX_DB = 6.0
# Yielding (clock_authority sliding below roles.CONCEDE_MIN_CLOCK) is the
# machine getting out of the way, and it has to be AUDIBLE or SPEC-V2 behavior
# 6 is unfalsifiable.  At full yield (clock 0.0) the machine plays at 0.4x its
# density and 6 dB down; at clock 0.5 and above both factors are exactly 1.0,
# so a take that never yields is bit-identical to one built without the rule.
_LEAD_YIELD_FLOOR = 0.4
_LEAD_YIELD_DB = -6.0

#: Role palette per plan state.  Read from ``mind`` so the sideman and the
#: leader voice the same instruments; the literal is a defensive fallback, not
#: a second source of truth (``mind._ROLES`` is module-private and this file is
#: the only cross-module reader of it).
_LEAD_ROLES: dict[str, dict[str, float]] = getattr(_mind_mod, "_ROLES", None) or {
    "comping": {"kick": 1.0, "snare": 0.8, "hat": 0.7, "bass": 1.0,
                "pad": 0.7, "keys": 0.5},
    "thinned": {"kick": 0.6, "hat": 0.4},
}


def _lead_transposed_key(key_str: str, semitones: int) -> str:
    """``session.key`` moved by ``semitones``, spelled the way it spells itself.

    Round-trips through :func:`plume.theory.parse_key`, which is what the duet
    engine does with it.
    """
    k = _parse_key(str(key_str))
    nk = _Key(tonic=(k.tonic + int(semitones)) % 12, mode=k.mode)
    return f"{_spell(nk.tonic, nk)} {nk.mode.replace('_', ' ')}"


def _lead_transposed_chord(sym: str, semitones: int, key_str: str | None) -> str:
    """A composed chord symbol moved by ``semitones``, spelled for ``key_str``."""
    ch = _parse_chord(str(sym))
    key = None
    if key_str is not None:
        try:
            key = _parse_key(key_str)
        except ValueError:               # pragma: no cover - defensive
            key = None
    return _spell((ch.root + int(semitones)) % 12, key) + ch.quality


# --- what a cue actually DOES (v2.1 amendment, SPEC-V2 sections 3 and 4) ----
# THE DEFECT: every cue kind except ``energy`` reached the decision log as
# "applied" and changed nothing.  MEASURED in a machine solo (role=lead, clock
# 0.90, form 1.00 — full authority, nobody to yield to): ``--cue
# "8.0:faster@next-bar"`` logged ``(tempo:faster): applied``, the bpm stayed at
# 96.0 on every subsequent bar, and the output WAV was BYTE-IDENTICAL to the
# no-cue run.  Same for slower / double time / half time / darker / brighter /
# more space.  Worst of all, ``take it out`` logged "ending the tune here" at
# bar 5 and then played all sixteen bars to the same 40.39 s duration.  The
# decision log is the only instrument a cold user has, and it was asserting
# actions that did not occur — the exact failure mode that hid v1's four bugs.
#
# Every kind now either DOES something audible or says plainly, in the same
# log, that it was understood and not applied and why.  Nothing claims
# "applied" that a listener could not hear.
_CUE_TEMPO_STEP = 0.06       # "faster"/"slower": +/-6%, an audible push
_CUE_RITARD_PER_BAR = 0.04   # "ritard": 4% slower per bar, compounding
_CUE_RITARD_FLOOR = 0.55     # ...down to 55% of the tempo it started from
_CUE_DENSITY_STEP = 0.65     # "sparser" multiplies density by this; busier /
#: "warmer" is half a "darker": the same shelf, a gentler move.  The tone
#: itself (shelf corner, mix, reverb steps, glide) is ``fx.BusTone``'s.
_CUE_WARM_STEP = 0.5
#: Bars of ending played after an `end` cue lands: one to say it, one to land
#: on.  Short on purpose — "take it out" is an instruction to stop, and a
#: sixteen-bar outro after it is not honouring it either.
_CUE_END_BARS = 2


class _LeadCues:
    """The renderer's half of the cue contract: what each kind DOES.

    ``Mind.plan_form`` owns the *form* consequences of a cue (jump, extend,
    end) and reports them; this owns the rest, because tempo, texture and
    style are the renderer's business and only the renderer can say whether
    they landed.  Pure and deterministic — a plain accumulator over the cue
    texts ``FormPlan.applied_cue`` reports, re-parsed with the same
    :func:`plume.mind.parse_cue` that queued them.
    """

    def __init__(self) -> None:
        self.tempo_ratio = 1.0       # multiplies the machine's bpm
        self.density_scale = 1.0     # multiplies the planner's density
        self.tilt = 0.0              # -1 fully dark .. +1 fully bright
        self.wet = 0.0               # reverb send, 0 .. fx.BUS_TONE_WET_MAX
        self.ritard = False
        self._ritard_floor = 1.0

    def apply(self, applied_cue: str | None) -> list[str]:
        """Fold in the cues that landed this bar; return one note each."""
        notes: list[str] = []
        for text in str(applied_cue or "").split("; "):
            if not text:
                continue
            p = _parse_cue(text)
            kind, value = p.get("kind"), p.get("value")
            if kind == "tempo":
                notes.append(self._tempo(text, value, p.get("ratio")))
            elif kind == "texture":
                notes.append(self._texture(text, value))
            elif kind == "style":
                notes.append(
                    f"cue {text!r} (style:{value}): understood but NOT applied "
                    f"— the style pack is chosen when the take is built and "
                    f"the voices cannot be swapped mid-bar; re-run with "
                    f"--style {value}")
        return notes

    def _tempo(self, text: str, value, ratio) -> str:
        before = self.tempo_ratio
        if value == "ritard":
            self.ritard = True
            self._ritard_floor = before * _CUE_RITARD_FLOOR
            return (f"cue {text!r} (tempo:ritard): slowing "
                    f"{_CUE_RITARD_PER_BAR * 100:.0f}% a bar down to "
                    f"{self._ritard_floor:.2f}x")
        if ratio:
            self.tempo_ratio = float(ratio)
        elif value == "faster":
            self.tempo_ratio = before * (1.0 + _CUE_TEMPO_STEP)
        elif value == "slower":
            self.tempo_ratio = before * (1.0 - _CUE_TEMPO_STEP)
        self.ritard = False
        return (f"cue {text!r} (tempo:{value}): tempo x{before:.2f} -> "
                f"x{self.tempo_ratio:.2f}")

    def _texture(self, text: str, value) -> str:
        if value in ("sparser", "busier"):
            before = self.density_scale
            self.density_scale = _clamp01(
                before * (_CUE_DENSITY_STEP if value == "sparser"
                          else 1.0 / _CUE_DENSITY_STEP), 0.15, 2.0)
            return (f"cue {text!r} (texture:{value}): density x{before:.2f} -> "
                    f"x{self.density_scale:.2f}")
        if value in ("darker", "brighter", "warmer"):
            before = self.tilt
            step = {"darker": -1.0, "brighter": 1.0, "warmer": -_CUE_WARM_STEP}[value]
            self.tilt = _clamp01(before + step, -1.0, 1.0)
            if self.tilt == before:
                return (f"cue {text!r} (texture:{value}): tone already at "
                        f"{before:+.1f}, as far as it goes; nothing changed")
            return (f"cue {text!r} (texture:{value}): tone {before:+.1f} -> "
                    f"{self.tilt:+.1f}, gliding in over "
                    f"{_fx.BUS_TONE_GLIDE_S:.0f}s")
        if value in ("dry", "wet"):
            before = self.wet
            step = _fx.BUS_TONE_WET_STEP * (1.0 if value == "wet" else -1.0)
            self.wet = round(_clamp01(before + step, 0.0, _fx.BUS_TONE_WET_MAX), 6)
            if self.wet == before:
                return (f"cue {text!r} (texture:{value}): reverb already "
                        f"{'off' if before == 0.0 else f'at {before:.2f}, the most it goes'}; "
                        f"nothing changed")
            return (f"cue {text!r} (texture:{value}): reverb {before:.2f} -> "
                    f"{self.wet:.2f}, gliding in over {_fx.BUS_TONE_GLIDE_S:.0f}s")
        return (f"cue {text!r} (texture:{value}): understood but NOT applied "
                f"— no tone move is defined for {value}")

    def bar_ratio(self) -> float:
        """The tempo multiplier for the upcoming bar, advancing a ritardando."""
        if self.ritard:
            nxt = self.tempo_ratio * (1.0 - _CUE_RITARD_PER_BAR)
            self.tempo_ratio = max(nxt, self._ritard_floor)
        return self.tempo_ratio


def _clamp01(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def lead_analysis_gain(x: np.ndarray, *, ref_db: float = _ANA_REF_DB) -> float:
    """Linear scale putting ``x``'s own loud passages at ``ref_db``.

    The fader-removal constant described above.  ``1.0`` (an exact no-op) when
    the take is empty, too short to measure, or below ``_ANA_FLOOR_DB`` —
    silence must stay silence, and a zero-length human must not divide by
    anything.

    Deterministic and gain-EXACT: scaling ``x`` by ``g`` scales the measured
    percentile by exactly ``g``, so the returned factor is divided by ``g`` and
    the normalized signal is the same take at the same level whatever the fader
    was.  Vectorised; no per-sample loop.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = (x.size // _ANA_HOP) * _ANA_HOP
    if n < _ANA_HOP:
        return 1.0
    env = np.sqrt(np.mean(x[:n].reshape(-1, _ANA_HOP) ** 2, axis=1))
    loud = float(np.percentile(env, _ANA_PCT))
    if not math.isfinite(loud) or loud <= 10.0 ** (_ANA_FLOOR_DB / 20.0):
        return 1.0                        # silence, or near enough to it
    g = 10.0 ** (float(ref_db) / 20.0) / loud
    return float(min(g, 10.0 ** (_ANA_MAX_GAIN_DB / 20.0)))


class _AttackRefiner:
    """Sub-hop attack times for detected onsets — the measurement the concede
    rule actually needs.

    SPEC-V2 section 1 amendment A3 is explicit that a caller which wants the
    machine to negotiate tempo must PASS ``human_offset_ms``; differencing the
    tracker's predicted beat against the machine's grid is not a 25 ms
    instrument.  SPEC-V2 section 6 measured why the raw detector is not one
    either: ``Frame.t_s`` is the END of the analysis window containing the
    attack, so it reads ~11 ms late and quantized to the 512-sample hop.  A
    CONSTANT bias on a SIGNED threshold is not a wash — it is a direction
    preference: it turns a +30 ms rush into a +19 ms reading (under
    ``concede_ms``, therefore invisible) while turning a -35 ms lay-back into
    -46 (fires early and harder).  Lead mode would then only ever concede
    downward, which is exactly the failure amendment A3 was written about.

    So each detection is refined the way ``tests/test_fixtures_gen.py``
    measures the joiners: walk back down a 1 ms RMS envelope from the local
    peak to where it last sat under 15% of that peak.  Vectorised envelope,
    one short python loop PER DETECTION (a few hundred per take), never per
    sample.  Deterministic: a pure function of the input buffer.
    """

    def __init__(self, x: np.ndarray, sr: int):
        x = np.asarray(x, dtype=np.float64).ravel()
        n = (x.size // _ENV_HOP) * _ENV_HOP
        self._env = (np.sqrt(np.mean(x[:n].reshape(-1, _ENV_HOP) ** 2, axis=1))
                     if n >= _ENV_HOP else np.zeros(0, dtype=np.float64))
        self._sr = int(sr)
        self._back = max(1, int(round(_ATTACK_BACK_S * sr / _ENV_HOP)))
        self._fwd = max(1, int(round(_ATTACK_FWD_S * sr / _ENV_HOP)))

    def refine(self, t_det: float) -> float:
        env = self._env
        if env.size == 0:
            return float(t_det)
        c = int(round(float(t_det) * self._sr / _ENV_HOP))
        a = max(0, c - self._back)
        b = min(env.size, c + self._fwd)
        if b - a < 3:
            return float(t_det)
        w = env[a:b]
        pk = int(np.argmax(w))
        if float(w[pk]) <= 0.0:
            return float(t_det)
        thr = _ATTACK_FRAC * float(w[pk])
        j = pk
        while j > 0 and float(w[j - 1]) > thr:
            j -= 1
        return float((a + j) * _ENV_HOP / self._sr)


class _Feeder:
    """Lazily pull :class:`~plume.listen.features.Frame` s up to a given time.

    ``simulate`` is driven by the tracker (frames arrive, boundaries fall out
    of them).  ``simulate_lead`` is driven by the MACHINE's grid, so the frame
    stream has to be consumable up to an arbitrary boundary time instead.
    Same extractor, same BLOCK size, same call order — only the pull direction
    changes.
    """

    def __init__(self, x: np.ndarray, fe: FeatureExtractor):
        self._x = x
        self._fe = fe
        self._i = 0
        self._q: list = []
        self._done = False

    def _fill(self) -> None:
        while not self._q and not self._done:
            if self._i >= self._x.shape[0]:
                self._q = list(self._fe.flush())
                self._done = True
            else:
                blk = self._x[self._i:self._i + BLOCK]
                self._i += BLOCK
                self._q = list(self._fe.process(blk))

    def take_until(self, t: float) -> list:
        """Every remaining frame whose ``t_s`` is <= ``t``, in stream order."""
        out: list = []
        while True:
            self._fill()
            if not self._q or float(self._q[0].t_s) > float(t):
                return out
            out.append(self._q.pop(0))

    def drain(self) -> list:
        return self.take_until(math.inf)


class _LeadPlanner:
    """The machine's bar plan while it is holding the floor.

    v2 AMENDMENT (SPEC-V2 section 4).  Section 4 says the rest of the pipeline
    "runs identically to v1 sim", and it does — features, beat, harmony, role
    negotiation, form planning and ``DuetEngine.render_bar`` are the same code
    on the same call cadence.  ``Mind.plan_bar`` is the ONE piece that cannot
    be, because both of its central rules are inverted here:

    * Its entry gate (``min_listen_bars`` / ``entry_conf`` / a locked grid)
      exists so a machine that must earn its way in never enters on a wrong
      tempo hypothesis.  In lead mode the machine IS the tempo hypothesis, and
      SPEC-V2 behavior 1 requires machine audio in bar 0 — including with
      ``input_wav=None``, where that gate can never open at all.
    * Its silence ladder (thin -> fade -> tacet) exists so the sideman stops
      when the human stops.  SPEC-V2 behavior 7 requires the exact opposite:
      the human drops out for 12 bars and the machine KEEPS LEADING, no tacet.
      Under ``plan_bar`` a machine-solo take fades from bar 6 and goes tacet on
      bar 10 forever, which is not a bug in ``plan_bar`` — it is the correct
      sideman behavior being asked the wrong question.

    So the bar plan is built here instead, from the SECTION's composed energy
    and the role state, and the reactive rules that genuinely still apply are
    re-expressed against the same :class:`~plume.mind.MindParams` thresholds
    the sideman uses (there is no second set of numbers to keep in sync):

    * **inverse density coupling** — ``density_high`` / ``density_hysteresis``
      including the one-bar recovery delay, verbatim from
      ``Mind._update_density``;
    * **the dynamic arc** — an ``energy_alpha`` EMA of the human's bar RMS
      against their level when they joined, clamped to the same [-12, +6] dB;
    * **register avoidance** — ``register_margin_st`` below the human's low
      register, taken from the human's bar or from the harmony tracker.

    Output is an ordinary :class:`~plume.mind.BarPlan` in the v1 state
    vocabulary, so the duet engine, ``_SOUNDING_STATES`` and every log reader
    downstream are unchanged.  Pure and deterministic: no rng, no I/O, no
    wall-clock.
    """

    def __init__(self, params: MindParams):
        self.p = params
        self._ema_db: float | None = None
        self._ref_db: float | None = None
        self._ceiling: float | None = None
        self._thinned = False
        self._recovery_pending = False
        self._last_density: float | None = None

    # --- verbatim from Mind._update_density (same params, same hysteresis) --
    def _update_density(self, d: float) -> None:
        p = self.p
        lo = p.density_high - p.density_hysteresis
        if not self._thinned:
            if d > p.density_high:
                self._thinned = True
                self._recovery_pending = False
        elif d > p.density_high:
            self._recovery_pending = False
        elif d < lo:
            if self._recovery_pending:
                self._thinned = False
                self._recovery_pending = False
            else:
                self._recovery_pending = True
        else:
            self._recovery_pending = False

    def observe(self, human: BarFeatures | None, hstate) -> None:
        """Fold one completed human bar in.  Called once per bar boundary."""
        p = self.p
        reg_low = None
        if human is not None and human.register_low_midi is not None:
            reg_low = float(human.register_low_midi)
        elif getattr(hstate, "register_low_midi", None) is not None:
            reg_low = float(hstate.register_low_midi)
        if reg_low is not None:
            self._ceiling = reg_low - p.register_margin_st
        if human is None or bool(human.silent):
            return
        d = float(human.onset_density)
        self._last_density = d
        self._update_density(d)
        rms = float(human.rms_db)
        a = p.energy_alpha
        self._ema_db = (rms if self._ema_db is None
                        else a * rms + (1.0 - a) * self._ema_db)
        if self._ref_db is None:
            # The human's level when they JOINED is the arc's reference: in
            # lead mode there is no entry bar to take one at.
            self._ref_db = rms

    def plan(self, bar: int, *, energy: float, clock_authority: float,
             section_name: str, density_scale: float = 1.0) -> BarPlan:
        p = self.p
        base = _LEAD_DENSITY_MIN + _LEAD_DENSITY_SPAN * float(
            min(1.0, max(0.0, energy)))
        base *= float(density_scale)       # what a sparser/busier cue does
        state = "thinned" if self._thinned else "comping"
        density = min(base, _LEAD_BUSY_DENSITY) if self._thinned else base

        clock = float(min(1.0, max(0.0, clock_authority)))
        share = min(1.0, clock / _CONCEDE_MIN_CLOCK)
        yield_factor = _LEAD_YIELD_FLOOR + (1.0 - _LEAD_YIELD_FLOOR) * share
        yield_db = _LEAD_YIELD_DB * (1.0 - share)
        density *= yield_factor

        arc = 0.0
        if self._ema_db is not None and self._ref_db is not None:
            arc = max(_LEAD_ARC_MIN_DB,
                      min(_LEAD_ARC_MAX_DB, self._ema_db - self._ref_db))
        gain_db = arc + yield_db

        d_str = (f"{self._last_density:.1f}" if self._last_density is not None
                 else "unmeasured")
        # v2.1: the opening word tells the truth about the chair the machine is
        # sitting in.  MEASURED: 29 of 44 logged bars had `"role": "follow"`
        # while `"reason"` began `"leading: section ..."` — and `reason` is the
        # primary human-readable field, so reading the NDJSON cold the machine
        # appeared to still be leading a bar it had explicitly yielded.  The
        # word is derived from the same ``clock_authority`` the yield is.
        holding = clock >= _CONCEDE_MIN_CLOCK
        bits = [f"{'leading' if holding else 'yielded'}: section "
                f"{section_name!r} energy {energy:.2f}",
                f"density {density:.2f}"]
        if self._thinned:
            bits.append(f"sparse (human {d_str} onsets/beat > "
                        f"{p.density_high:.1f})")
        if yield_db < -0.05:
            bits.append(f"yielding {yield_db:+.1f}dB, {yield_factor:.2f}x "
                        f"density (clock {clock:.2f})")
        if abs(arc) >= 0.05:
            bits.append(f"arc {arc:+.1f}dB")
        return BarPlan(
            bar=int(bar), state=state,
            roles=dict(_LEAD_ROLES.get(state, {})),
            gain_db=float(gain_db),
            density=float(min(1.0, max(0.0, density))),
            register_ceiling_midi=self._ceiling,
            reason=", ".join(bits),
        )


def _lead_line(plan: BarPlan, t_s: float, role, fplan, *,
               machine_bpm: float, machine_key: str, machine_chord: str | None,
               beat, hstate, human: BarFeatures | None,
               human_offset_ms: float | None,
               cue_effects: list[str] | None = None) -> dict:
    """One machine bar -> one NDJSON object.

    v2 AMENDMENT (SPEC-V2 section 4).  The v1 keys keep their names and
    meanings with ONE documented exception: ``bpm``, ``key`` and ``chord``
    describe the MACHINE now, because in lead mode they are what is being
    played rather than what is being heard.  The human's readings are on the
    same line under ``human_*``, so nothing is lost and nothing is silently
    reinterpreted.  ``form`` is nested rather than emitted as a separate
    record: it is a decision about the same bar at the same boundary, and
    splitting it would make every reader join two streams to answer "what did
    the machine do in bar 12".
    """
    return {
        "bar": int(plan.bar),
        "t_s": _round(t_s),
        "state": str(plan.state),
        "reason": str(plan.reason),
        "bpm": _round(machine_bpm),
        "conf": _round(beat.confidence),
        "key": machine_key,
        "chord": machine_chord,
        "roles": {str(k): _round(v) for k, v in sorted(plan.roles.items())},
        "gain_db": _round(plan.gain_db),
        "register_ceiling_midi": _round(plan.register_ceiling_midi),
        "machine_density": _round(plan.density),
        # --- the v2 contract (SPEC-V2 section 4) ---
        "role": str(role.role),
        "clock_authority": _round(role.clock_authority),
        "form_authority": _round(role.form_authority),
        "tempo_offset_pct": _round(role.tempo_offset_pct),
        "tempo_target_pct": _round(getattr(role, "tempo_target_pct", 0.0)),
        "transpose_semitones": int(role.transpose_semitones),
        "role_events": [str(e) for e in getattr(role, "events", ())],
        "role_reason": str(role.reason),
        "form": {
            "section_idx": int(fplan.section_idx),
            "action": str(fplan.action),
            "bars_left_in_section": int(fplan.bars_left_in_section),
            "announce": bool(getattr(fplan, "announce", False)),
            "applied_cue": getattr(fplan, "applied_cue", None),
            "dropped_cue": getattr(fplan, "dropped_cue", None),
            "reason": str(fplan.reason),
        },
        # v2.1: what the RENDERER did about the cues that landed this bar —
        # the tempo/texture half of a cue's effect, which the form cursor
        # cannot report because it is not the thing that changed.
        "cue_effects": list(cue_effects) if cue_effects else None,
        # --- what the machine HEARD (the human's isolated signal) ---
        "human_bpm": _round(beat.bpm),
        "human_locked": bool(beat.locked),
        "human_key": hstate.key,
        "human_chord": hstate.chord,
        "human_silent": (None if human is None else bool(human.silent)),
        "human_density": (None if human is None
                          else _round(human.onset_density)),
        "human_rms_db": (None if human is None else _round(human.rms_db)),
        "human_offset_ms": _round(human_offset_ms),
    }


def simulate_lead(input_wav: str | None, *, session=None,
                  prompt: str | None = None, out_path: str, seed: int = 0,
                  role: RoleParams | None = None,
                  mind: MindParams | None = None,
                  human_offset_s: float = 0.0,
                  cues: "list[tuple[float, Cue]] | None" = None,
                  # --- additive, all defaulted: see the amendments below ---
                  style: str | None = None,
                  bars: int | None = None,
                  form: FormParams | None = None,
                  mix_human_db: float = 0.0,
                  decisions_path: str | None = None) -> SimResult:
    """Lead mode, offline (SPEC-V2 section 4).  The machine starts; you join.

    The machine's clock is authoritative from t=0.  Bar ``k``'s boundary is
    wherever the previous ``k`` rendered bars ended — the grid is GENERATED,
    never detected — so the take's duration is the integral of whatever tempo
    curve the concessions produced, exactly, with the fractional-sample
    remainder carried by ``DuetEngine`` (zero drift at any bpm, and a
    concession changes real rendered bar lengths rather than a number in a
    log).  The human's WAV is mixed in starting at ``human_offset_s``.

    Per machine bar, in this order: consume the human's frames up to the
    boundary -> summarize the human bar just completed -> measure their phase
    against the machine's sixteenth grid (:class:`_AttackRefiner` +
    :func:`plume.roles.onset_phase_offset_ms`) -> ``RoleNegotiator.update`` ->
    ``Mind.plan_form`` -> :class:`_LeadPlanner` -> ``DuetEngine.render_bar``
    at the machine's own tempo and its own (possibly transposed) chord.

    Beat / harmony / features run on the HUMAN's isolated signal only.  In sim
    that is exact; SPEC-V2 section 6 measured what a caller reading a summed
    bus gets instead (an offset that tracks the FADER rather than the playing),
    which is why this is the ground truth for the lead behavior matrix and
    ``jam`` inherits with a caveat.

    Deterministic: byte-identical WAV + NDJSON for identical arguments.  No
    wall-clock, no global rng.

    v2 amendments to the frozen signature, all additive and defaulted:

    ``role`` defaults to ``RoleParams(role="lead")``, not to ``RoleParams()``.
        ``RoleParams``' own default is ``follow`` (clock_authority 0.15), and a
        machine whose clock is authoritative from t=0 while it holds 15% of
        the clock is a contradiction in the first sentence of section 4.  Pass
        an explicit ``RoleParams`` for anything else, including ``trade``.
    ``style`` / ``bars`` / ``form`` / ``mix_human_db`` / ``decisions_path``
        The style defaults to the session's own; ``bars`` to the composed
        length, extended when necessary to cover the human's take (the machine
        holds the floor for at least as long as they play); ``form`` is
        :class:`~plume.mind.FormParams`; ``decisions_path`` defaults to
        ``out_path`` with a ``.ndjson`` extension, as in v1.
    **No machine-bus calibration.**
        v1 calibrates the machine bus so its RMS lands ``gain_db`` under the
        human's, because the sideman is joining THEM.  Here the human is
        joining US: the machine plays at its natural level and
        ``mix_human_db`` is the human's fader.  A mid-take recalibration would
        also step the machine's level at the join bar, which is precisely the
        disruption SPEC-V2 behavior 8 forbids.
    **One cue is delivered per bar boundary**, earliest-issued first; the rest
        wait for the next one.  ``Mind.plan_form`` takes a single cue per call
        and queues internally, so several cues issued at the same instant still
        all land — one boundary apart at worst, and their own ``at``/``ttl_bars``
        still govern when they apply.  Any cue whose time falls past the last
        boundary is reported as ``{"event": "cue_undelivered"}``.
    """
    if (session is None) == (prompt is None):
        raise ValueError(
            "simulate_lead needs exactly one of session=... or prompt=... "
            "(the machine cannot start from nothing, and two arrangements is "
            "an ambiguity, not a merge)")
    if session is None:
        from plume.compose import compose as _compose
        from plume.compose import parse_prompt as _parse_prompt
        session = _compose(_parse_prompt(str(prompt)))
    if not math.isfinite(float(human_offset_s)) or float(human_offset_s) < 0.0:
        raise ValueError(
            f"human_offset_s must be finite and >= 0, got {human_offset_s!r}")
    if float(getattr(session, "bpm", 0.0)) <= 0.0:
        raise ValueError(f"session bpm must be positive, got {session.bpm!r}")

    rp = role if role is not None else RoleParams(role="lead")
    mp = mind if mind is not None else MindParams()
    fp = form if form is not None else FormParams()
    style_name = str(style or getattr(session, "style", "") or "lofi")

    out_path = str(out_path)
    if decisions_path is None:
        base, _ = os.path.splitext(out_path)
        decisions_path = base + ".ndjson"
    decisions_path = str(decisions_path)

    # ---- the human's isolated signal, placed on the machine's timeline -----
    human_notes: list[str] = []
    if input_wav is None:
        sr = _LEAD_SR
        human = np.zeros(0, dtype=np.float32)
    else:
        human, sr = sf.read(str(input_wav), dtype="float32", always_2d=True)
        # v2.1: what was done to the human's file, said out loud.  MEASURED, a
        # 44.1 kHz human, a mono human, a 0.4 s human, a silent human and a
        # ZERO-LENGTH wav all exited 0 with identical output and nothing
        # anywhere saying "mono-summed", "the take is 0.4 s" or "the file is
        # empty" — `per_bus_rms_db.human` was the only indirect tell.
        if int(sr) != _LEAD_SR:
            human_notes.append(
                f"the human file is {int(sr)} Hz, not {_LEAD_SR}; the whole "
                f"take is rendered at {int(sr)} Hz to keep their timing exact")
        if human.shape[1] > 1:
            human_notes.append(f"mono-summed from {human.shape[1]} channels")
        human = human.mean(axis=1).astype(np.float32)          # mono-sum
        sr = int(sr)
        if human.shape[0] == 0:
            human_notes.append("the human file is EMPTY — this is a machine solo")
        elif human.shape[0] / sr < 1.0:
            human_notes.append(
                f"the human file is only {human.shape[0] / sr:.2f}s long")
    pad = int(round(float(human_offset_s) * sr))
    if pad > 0:
        human = np.concatenate(
            [np.zeros(pad, dtype=np.float32), human]).astype(np.float32)

    composed_bpm = float(session.bpm)
    composed_bar_s = 240.0 / composed_bpm
    flat = flatten_sections(session)
    if bars is not None:
        n_bars = int(bars)
        if n_bars <= 0:
            raise ValueError(f"bars must be positive, got {bars!r}")
    else:
        n_bars = max(1, sum(int(s.bars) for s in flat))
        if human.shape[0]:
            n_bars = max(n_bars,
                         int(math.ceil((human.shape[0] / sr) / composed_bar_s)))

    fe = FeatureExtractor(sr=sr)
    tracker = BeatTracker()
    harmony = HarmonyTracker()
    neg = RoleNegotiator(rp, composed_bpm=composed_bpm,
                         composed_key=str(session.key))
    mind_obj = Mind(mp, fp)
    duet = DuetEngine(style=style_name, key_fallback=str(session.key),
                      seed=int(seed), sr=sr)
    # The machine's own swing, read from the style it is playing: the human's
    # phase has to be measured against the grid the machine IS playing, not an
    # idealised straight one (see roles.onset_phase_offset_ms).
    try:
        machine_swing = float(_load_style(style_name)["swing"])
    except Exception:                    # pragma: no cover - defensive
        machine_swing = 0.5
    planner = _LeadPlanner(mp)
    # The analysis chain hears the human with their fader taken out; the MIX
    # below still hears them exactly as loud as they are (see _ANA_* above).
    ana_gain = lead_analysis_gain(human)
    human_ana = (human if ana_gain == 1.0
                 else (human.astype(np.float64) * ana_gain).astype(np.float32))
    refiner = _AttackRefiner(human_ana, sr)
    feeder = _Feeder(human_ana, fe)

    pending = sorted(list(cues or []), key=lambda c: float(c[0]))

    lines: list[dict] = []
    placements: list[tuple[int, np.ndarray]] = []
    role_states: list[dict] = []
    form_plans: list[dict] = []
    bar_times: list[float] = []
    machine_bpms: list[float] = []
    states_by_bar: list[str] = []
    bpm_trace: list[tuple[float, float]] = []

    state = tracker.process([])
    hstate = harmony.process([])
    prev_state = None
    prev_key: str | None = hstate.key
    acc = _BarAccum()
    acc.reset(0.0)
    attacks: list[float] = []
    human_ref_db: float | None = None
    cursor = 0
    prev_t_b = 0.0
    sec_idx_prev: int | None = None
    sec_bar = 0
    cues = _LeadCues()
    ending_at: int | None = None
    tone = _fx.BusTone(sr)
    machine_key = _lead_transposed_key(str(session.key), 0)

    def consume(t: float) -> None:
        nonlocal state, hstate, prev_state, prev_key
        for f in feeder.take_until(t):
            state = tracker.process([f])
            hstate = harmony.process([f])
            acc.add(f, mp.silence_rms_db)
            if bool(f.onset):
                attacks.append(refiner.refine(float(f.t_s)))
            ft = _round(float(f.t_s))
            if prev_state is not None and state.locked != prev_state.locked:
                lines.append({"event": "lock" if state.locked else "unlock",
                              "t_s": ft, "bpm": _round(state.bpm),
                              "conf": _round(state.confidence)})
            if prev_state is not None and (state.tempo_grid_changes
                                           != prev_state.tempo_grid_changes):
                _oct = (state.tempo_octave_changes
                        != prev_state.tempo_octave_changes)
                lines.append({"event": ("tempo_octave_change" if _oct
                                        else "tempo_grid_change"),
                              "t_s": ft,
                              "from_bpm": _round(prev_state.bpm),
                              "to_bpm": _round(state.bpm),
                              "conf": _round(state.confidence)})
            if hstate.key != prev_key:
                lines.append({"event": "key_change", "t_s": ft,
                              "key": hstate.key})
                prev_key = hstate.key
            prev_state = state

    for k in range(n_bars):
        t_b = cursor / sr
        consume(t_b)

        # --- the human bar that just ended, measured on the MACHINE's grid --
        bf: BarFeatures | None = None
        off: float | None = None
        off_unit_s: float | None = None
        if k > 0:
            span = max(t_b - prev_t_b, 1e-9)
            bf = _summarize(acc, k - 1, t_b, span / 4.0, hstate, mp,
                            human_ref_db)
            if not bf.silent:
                human_ref_db = (bf.rms_db if human_ref_db is None
                                else 0.5 * bf.rms_db + 0.5 * human_ref_db)
                off_unit_s = span / _LEAD_SUBDIV
                # v2.1: measured against the grid the machine is actually
                # PLAYING, swing included.  Comparing a swung human to a
                # straight grid reads the shared groove as a lay-back — see
                # ``roles.onset_phase_offset_ms``.
                off = onset_phase_offset_ms(attacks, prev_t_b, t_b, off_unit_s,
                                            swing=machine_swing)
            # Only the bar just measured is ever wanted again; keeping the whole
            # take's attacks would make a long session quadratic in onsets for
            # no benefit (the list is in time order, so this is a prefix drop).
            cut = 0
            while cut < len(attacks) and attacks[cut] < prev_t_b:
                cut += 1
            if cut:
                del attacks[:cut]

        rstate = neg.update(bar=k, beat=state, harmony=hstate, human=bf,
                            machine_grid_t_s=t_b, human_offset_ms=off,
                            human_offset_unit_s=off_unit_s)

        cue = None
        if pending and float(pending[0][0]) <= t_b:
            cue = pending.pop(0)[1]
        fplan = mind_obj.plan_form(rstate, state, hstate, bf, session, cue)
        cue_notes = cues.apply(getattr(fplan, "applied_cue", None))
        if str(fplan.action) == "end" and ending_at is None:
            # The ending is REAL: the form said end, so the take ends.  Two
            # more bars (the last one at the cue's own ritardando if one is
            # running) and the tail, rather than the previous behaviour of
            # logging "ending the tune here" and playing out the full 22 bars
            # that were left.
            ending_at = k
            cue_notes.append(
                f"ending: {_CUE_END_BARS} bar(s) and out (bar {k})")

        # --- where the arrangement is, and what it says to play -------------
        sec = flat[fplan.section_idx] if 0 <= fplan.section_idx < len(flat) else None
        if sec is None:
            energy, sec_name, chord_syms = 0.6, "", []
        else:
            spec = session.movements[sec.movement].sections[sec.section]
            energy = float(sec.energy)
            sec_name = str(sec.name)
            chord_syms = [str(c) for c in (spec.chords or [])]
            # v2 amendment: the chord cursor is sim's, because ``FormPlan``
            # exposes ``bars_left_in_section`` but not the span.  Reset when the
            # cursor ENTERS a section — a different index, or an advance/jump
            # that re-entered this one (a one-section map that wrapped) — and
            # otherwise step, so an ``extend`` keeps the cycle running, which is
            # what "cycled across bars" means for a section that got longer.
            entered = (sec.idx != sec_idx_prev
                       or str(fplan.action) in ("advance", "jump"))
            sec_idx_prev = sec.idx
            if entered:
                sec_bar = 0
            elif k > 0:
                sec_bar += 1
        machine_key = _lead_transposed_key(str(session.key),
                                           rstate.transpose_semitones)
        chord = None
        if chord_syms:
            chord = _lead_transposed_chord(chord_syms[sec_bar % len(chord_syms)],
                                           rstate.transpose_semitones,
                                           machine_key)

        # --- the machine's own clock and harmony, handed to the duet engine -
        m_bpm = float(rstate.bpm) if rstate.bpm > 0.0 else composed_bpm
        m_bpm *= cues.bar_ratio()            # what a tempo cue actually does
        spb = 60.0 / m_bpm
        m_beat = BeatState(
            t_s=t_b, bpm=m_bpm, confidence=1.0, locked=True, bar=int(k),
            beat_in_bar=0, phase_beats=0.0, next_beat_t_s=t_b + spb,
            next_downbeat_t_s=t_b + 4.0 * spb, beat_interval_s=spb)
        m_harm = HarmonyState(
            t_s=t_b, key=machine_key, key_conf=1.0, chord=chord,
            chord_conf=1.0 if chord is not None else 0.0,
            register_low_midi=None, register_high_midi=None)

        planner.observe(bf, hstate)
        plan = planner.plan(k, energy=energy,
                            clock_authority=rstate.clock_authority,
                            section_name=sec_name,
                            density_scale=cues.density_scale)
        if cue_notes:
            plan = _replace(plan, reason=plan.reason + "; " + "; ".join(cue_notes))

        # The tone cues' stage, bar by bar exactly as live runs it; bit-exact
        # pass-through until a cue moves it.
        bar_audio = tone.process(duet.render_bar(plan, m_beat, m_harm,
                                                 section=fplan.section_idx),
                                 tilt=cues.tilt, wet=cues.wet)
        placements.append((cursor, bar_audio))
        cursor += bar_audio.shape[0]

        lines.append(_lead_line(plan, t_b, rstate, fplan,
                                machine_bpm=m_bpm, machine_key=machine_key,
                                machine_chord=chord, beat=state, hstate=hstate,
                                human=bf, human_offset_ms=off,
                                cue_effects=cue_notes or None))
        role_states.append({
            "bar": int(k), "t_s": round(t_b, 6), "role": str(rstate.role),
            "clock_authority": float(rstate.clock_authority),
            "form_authority": float(rstate.form_authority),
            "tempo_offset_pct": float(rstate.tempo_offset_pct),
            "transpose_semitones": int(rstate.transpose_semitones),
            # Measured whether or not it is acted on: under tempo_policy="hold"
            # this is the whole point — the drift is read, logged and NOT
            # chased, which is what makes it legible as practice feedback
            # (roadmap v2.5, "the mirror").
            "human_offset_ms": (None if rstate.human_offset_ms is None
                                else float(rstate.human_offset_ms)),
            "tempo_policy": str(rstate.tempo_policy),
            "bpm": float(m_bpm),
            "events": [str(e) for e in getattr(rstate, "events", ())],
            "reason": str(rstate.reason),
        })
        form_plans.append({
            "bar": int(k), "section_idx": int(fplan.section_idx),
            "action": str(fplan.action),
            "bars_left_in_section": int(fplan.bars_left_in_section),
            "announce": bool(getattr(fplan, "announce", False)),
            "applied_cue": getattr(fplan, "applied_cue", None),
            "dropped_cue": getattr(fplan, "dropped_cue", None),
            "reason": str(fplan.reason),
        })
        bar_times.append(round(t_b, 6))
        machine_bpms.append(round(m_bpm, 6))
        states_by_bar.append(str(plan.state))
        bpm_trace.append((round(t_b, 4), round(m_bpm, 4)))

        acc.reset(t_b)
        prev_t_b = t_b
        if ending_at is not None and k - ending_at >= _CUE_END_BARS - 1:
            lines.append({"event": "ended", "t_s": _round(cursor / sr),
                          "bar": int(k), "from_bar": int(ending_at),
                          "bars": int(k + 1)})
            break

    consume(math.inf)
    for t_c, c in pending:
        lines.append({"event": "cue_undelivered", "t_s": _round(float(t_c)),
                      "cue": str(getattr(c, "text", c))})
    # v2.1: a cue that was DELIVERED but never reached its boundary used to
    # vanish without a trace — MEASURED, `--cue "20:go to the chorus@next-
    # section"` in a 16-bar run that sits entirely inside one 16-bar section
    # never applied, never appeared as `dropped_cue` on any bar, and had no
    # pending/dropped field in the manifest either.  `plume cue --help`
    # advertises --ttl-bars as "drop the cue if it has not applied within N
    # bars", so a drop that leaves no record is the advertised behaviour
    # happening invisibly.  Anything still queued when the take ends is
    # reported here, with the boundary it was waiting for.
    for c in mind_obj.pending_cues():
        lines.append({
            "event": "cue_never_applied",
            "cue": str(getattr(c, "text", c)),
            "at": str(getattr(c, "at", "next-phrase")),
            "reason": (f"the take ended before a {getattr(c, 'at', '?')} "
                       f"boundary came round"),
        })

    tail = duet.flush_tail()
    if tail.shape[0]:
        tail = tone.process(tail, tilt=cues.tilt, wet=cues.wet)
        placements.append((cursor, tail))
        cursor += tail.shape[0]
    ring = tone.flush()
    if ring.shape[0]:
        placements.append((cursor, ring))

    # ---- mix (no bus calibration in lead mode — see the docstring) ---------
    machine_end = max((i0 + a.shape[0] for i0, a in placements), default=0)
    total = max(human.shape[0], machine_end)
    machine = np.zeros((total, 2), dtype=np.float64)
    for i0, a in placements:
        machine[i0:i0 + a.shape[0]] += a.astype(np.float64)
    hgain = 10.0 ** (float(mix_human_db) / 20.0)
    human_st = np.zeros((total, 2), dtype=np.float64)
    human_st[: human.shape[0]] = (human.astype(np.float64) * hgain)[:, None]
    mix = human_st + machine
    pre_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    out = _fx.limiter(mix, _LIMIT_DB)

    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    sf.write(out_path, out, sr, subtype="PCM_24")

    dparent = os.path.dirname(os.path.abspath(decisions_path))
    os.makedirs(dparent, exist_ok=True)
    with open(decisions_path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, sort_keys=True) + "\n")

    def _bus_rms(x: np.ndarray) -> float:
        return _db(math.sqrt(float(np.mean(x ** 2)))) if x.size else -90.0

    events_seen: list[str] = []
    for r in role_states:
        events_seen += list(r["events"])
    manifest = {
        "mode": "lead",
        "input": None if input_wav is None else str(input_wav),
        "out": out_path,
        "decisions": decisions_path,
        "style": style_name,
        "seed": int(seed),
        "sr": int(sr),
        "session": str(getattr(session, "name", "")),
        "composed_bpm": composed_bpm,
        "composed_key": str(session.key),
        "human_offset_s": float(human_offset_s),
        # v2.1: anything notable about the human's FILE (rate, channels,
        # length), so a degenerate input is visible instead of silent.
        "human_notes": list(human_notes),
        "mix_human_db": float(mix_human_db),
        "bars": int(n_bars),
        "duration_s": round(total / sr, 6),
        "machine_duration_s": round(machine_end / sr, 6),
        "peak_db": _round(_db(float(np.max(np.abs(out))) if out.size else 0.0)),
        "pre_limiter_peak_db": _round(_db(pre_peak)),
        "rms_db": _round(_bus_rms(out.astype(np.float64))),
        "per_bus_rms_db": {
            "human": _round(_bus_rms(human_st)),
            "machine": _round(_bus_rms(machine)),
        },
        "final_role": role_states[-1]["role"] if role_states else None,
        "final_clock_authority": (role_states[-1]["clock_authority"]
                                  if role_states else None),
        "final_tempo_offset_pct": (role_states[-1]["tempo_offset_pct"]
                                   if role_states else None),
        "final_bpm": machine_bpms[-1] if machine_bpms else None,
        "transpose_semitones": (role_states[-1]["transpose_semitones"]
                                if role_states else 0),
        "role_events": sorted(set(events_seen)),
        # Every cue's fate, in the manifest, so `--json` answers "did my cue
        # land?" without anybody having to read the NDJSON (v2.1).
        "cues_applied": sorted({t for f in form_plans
                                for t in str(f["applied_cue"] or "").split("; ")
                                if t}),
        "cues_dropped": sorted({t for f in form_plans
                                for t in str(f["dropped_cue"] or "").split("; ")
                                if t}),
        "cues_never_applied": [str(getattr(c, "text", c))
                               for c in mind_obj.pending_cues()],
        "cues_undelivered": [str(getattr(c, "text", c)) for _t, c in pending],
        "machine_key": machine_key,
        "human_final_key": hstate.key,
        "human_final_bpm": _round(state.bpm),
        "role_params": asdict(rp),
        "params": asdict(mp),
        "form_params": asdict(fp),
    }
    return SimResult(
        out_path=out_path,
        decisions_path=decisions_path,
        duration_s=total / sr,
        # The machine is in from bar 0 — that is what lead mode MEANS, and
        # SPEC-V2 behavior 1 asserts it.  There is no "entering" state here.
        entered_bar=0 if states_by_bar else None,
        states_by_bar=states_by_bar,
        bpm_trace=bpm_trace,
        # v2: the MACHINE's key (transposed if it conformed) — the key the take
        # is actually in.  What was heard is ``manifest["human_final_key"]``.
        final_key=machine_key,
        manifest=manifest,
        role_states=role_states,
        form_plans=form_plans,
        bar_times_s=bar_times,
        machine_bpm_by_bar=machine_bpms,
    )
