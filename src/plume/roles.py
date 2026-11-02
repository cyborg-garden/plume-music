"""Role negotiation — who is holding time, and who is holding the form.

SPEC-V2 sections 0 and 1.  ``RoleNegotiator.update`` is called once per bar
boundary (by ``sim.simulate_lead`` offline, by ``realtime`` live) and returns a
:class:`RoleState` describing the machine's authority over the *upcoming* bar.

The v1 sideman answered one question — "how do I earn my way in behind a human
who owns the clock?".  v2 answers the harder one: **both chairs are real, and
the chair you are sitting in changes mid-song without anybody stopping.**  Two
dials, each a float in [0, 1]:

``clock_authority``
    Who owns *time*.  0.0 = the human's beat is truth and the machine
    phase-corrects hard toward it (v1).  1.0 = the machine free-runs and never
    phase-corrects — Band-in-a-Box's virtue, and RESEARCH.md's "an unshakeable
    clock is a feature": a player can lean on it.  In between is **mutual
    entrainment**, which is what human bands actually do.

``form_authority``
    Who owns *the form* — what section we are in, when it changes, what the
    harmony is.  1.0 = the machine plays its composed arrangement and the human
    fits in.  0.0 = the machine takes its harmony and sections from what it
    hears (v1).

TEMPO AUTHORITY (v2.2 amendment — this replaced conceding-by-default)
--------------------------------------------------------------------
The first paragraph below describes the ``"meet"`` policy, which SPEC-V2 made
lead mode's default and which is **no longer the default**.  See
``docs/decisions/2026-08-15-tempo-authority.md``.

A machine holding the groove must not chase the human's timing *errors*.  The
rhythm section is the REFERENCE; if it moves every time a player pushes there
is no reference, and the tempo random-walks — the human rushes, the machine
follows, the machine's new tempo becomes the thing the human is now ahead of.
MEASURED, and this is what settled it: v2.0 in lead mode met a human sitting
30 ms in front and ended a 48-bar take **2.5 beats ahead of them**, with a
decision log that read as a tidy concession and a settled tempo the whole way
down (``test_a_partner_who_cannot_bend_is_the_case_that_hunts``).  Conceding to
a partner who cannot bend is divergent by construction.  RESEARCH.md said so
before any of this was built: *"an unshakeable clock is a feature, not a
limitation; musicians adapt to a trustworthy machine."*

So ``RoleParams.tempo_policy`` is explicit, and lead mode defaults to
``"hold"`` (see :data:`TEMPO_POLICIES`, :data:`ROLE_TEMPO_POLICY`).

**Holding is not deafness, and it is not inaction.**  Three things that
"never concede" must NOT be confused with, each of which has a test:

1. The offset is still MEASURED, bounded and LOGGED every bar — ``RoleState``
   carries ``human_offset_ms`` and the reason says ``holding: human +28ms
   (policy=hold)``.  It is free (the concede rule computed it anyway), a
   silent hold is indistinguishable from a broken measurement, and it is
   exactly the number the roadmap's v2.5 "mirror" practice feedback wants:
   *you're pushing the & of 2 by 24 ms*.  A machine that follows your rushing
   destroys its own practice value — you cannot hear an error you are being
   accommodated for.
2. Form-level tempo INTENT is still followed.  Drift and a deliberate
   ritardando are different musical events handled by different subsystems,
   and the split is architectural rather than conditional: this module only
   ever writes ``tempo_offset_pct``, while cue tempo (``ritard`` / ``faster``
   / ``slower`` / ``half time`` / ``double time``) is a SEPARATE multiplier
   applied downstream — ``sim._LeadCues.bar_ratio()``, folded in as
   ``m_bpm = rstate.bpm * cues.bar_ratio()`` — and the ending ritardando
   detector is ``mind.Mind._form_track_tempo``, which drives the FORM (an
   ``end`` action), not this rule.  Nothing in ``roles.py`` is on the intent
   path, so gating drift here cannot reach it.  Verified end to end rather
   than reasoned about: ``tests/test_tempo_policy.py``.
3. The v2.1 residual — "against a partner who cannot bend the tempo curve is a
   bounded hunt rather than a settle (+1.x%)" — does not arise under ``hold``
   at all.  A fixed recording produces EXACTLY 0.0% drift, not a small one.

The five behaviors, all evaluated at bar boundaries:

Concede tempo (``tempo_policy="meet"`` only)
    The signed offset between the human's detected beats and the machine's own
    grid is measured every bar.  When it has been *consistently* on one side
    for ``concede_bars`` and its mean magnitude exceeds ``concede_ms``, the
    machine absorbs ``concede_rate`` of it as a **period** change, ramped over
    the following bar and capped at ``concede_max_pct`` from the composed
    tempo.  The human pushes, the machine picks up; the human lays back, it
    settles.  Deliberately inert on an on-grid human: jitter around zero never
    passes the same-sign test, and a small consistent bias never passes
    ``concede_ms``.  Chasing per-bar jitter would be the single most audible
    way to make an unshakeable clock shakeable.

Yield the lead
    While leading, human onset density above ``handoff_density`` for
    ``handoff_hysteresis_bars`` slides both dials down toward ``follow``.  They
    have taken over; get out of the way.

Take the lead
    While following, ``handoff_silence_bars`` of human silence (with a groove
    still under us) slides both dials up toward ``lead``.  Someone has to hold
    the floor.

Trade
    ``role="trade"`` alternates authority every ``trade_bars`` at phrase
    boundaries, with v1's human-wins collision rule.

Meet them harmonically
    A human key belief consistently within ``conform_semitones`` of ours for
    ``conform_bars`` transposes the machine's arrangement to *their* key (they
    capo'd; we are in F and they are in F#).  Further away than that, hold and
    say why — they are playing something else, and conforming would be
    guessing.

Pure and deterministic: no rng, no I/O, no wall clock; all memory lives inside
``RoleNegotiator``.  ``reason`` strings are part of the contract — they are what
makes the decisions NDJSON self-explanatory, and they are asserted verbatim by
``tests/test_roles.py``.

MEASURED ON REAL MATERIAL
-------------------------
v1 shipped ten green fixture tests and was unusable on real music, so every
threshold here was checked against plume's own v0 renders (trap 140 / D minor,
lo-fi 80 / C major, at 0 dB and -20 dB) driven through the real
``FeatureExtractor`` / ``BeatTracker`` / ``HarmonyTracker``.  Three things the
fixtures could never have shown, each of which now has a named parameter and a
regression test in ``tests/test_roles.py``:

1. **A bare offset derived from ``BeatState`` is not a 25ms instrument.**
   Differencing the tracker's predicted beat against the machine's grid
   produces samples out to +/-210ms — half a beat — because a swung groove
   biases the prediction and a sub-percent tempo mismatch walks the phase clean
   across the beat.  Unfiltered, that drove a single **-8%** concession
   straight into the cap **in the wrong direction**.  Hence
   ``concede_plausible_frac`` (an offset past a fifth of a beat is a phase
   estimate that missed, not a human pushing) and ``concede_spread_ms`` (two
   bars that merely share a sign are not "consistent"), plus a clamp of one
   concession to one bar's ramp.  With all three, the worst drift across the
   four takes falls to 3.4% and most of them do not move at all.  Inertness is
   the safe failure and it is the one those gates buy: callers that want the
   machine to actually negotiate tempo should pass ``human_offset_ms`` from
   :func:`onset_phase_offset_ms`, which on the same material recovers a known
   30ms shift to within a few ms and drives the tempo the way the human pushed.
2. **Real programmed material sits at 5.0 onsets/beat** — two and a half times
   ``handoff_density`` — so the spec's bare absolute yield gate handed the lead
   away at the first hysteresis window on anything with hats.  Hence
   ``handoff_density_ratio``: a takeover is the human getting busier than *they*
   have been, not busier than a constant.
3. **A pitch-class histogram cannot tell a key from its relative** (lo-fi in
   C major reads as A minor, every frame, at both gains), so a conform rule
   written on key tonics fires spuriously and can transpose to the wrong key.
   Hence :func:`collection_root`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .listen.beat import BeatState
from .listen.harmony import HarmonyState
from .theory import Key, parse_key, spell

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Imported lazily on purpose: ``mind.py`` is built concurrently with this
    # module and only the *annotation* is needed here.  ``BarFeatures`` is used
    # by attribute (``.silent``, ``.onset_density``), never constructed.
    from .mind import BarFeatures

ROLES = ("lead", "follow", "trade", "solo_support")

#: role -> (clock_authority, form_authority) defaults (SPEC-V2 section 1).
ROLE_DEFAULTS: dict[str, tuple[float, float]] = {
    "lead": (0.9, 1.0),
    "follow": (0.15, 0.0),
    "trade": (0.5, 0.5),
    "solo_support": (0.6, 0.85),
}

#: The three answers to "whose clock is it?" (v2.2 amendment — see
#: ``docs/decisions/2026-08-15-tempo-authority.md`` and the module docstring's
#: TEMPO AUTHORITY note).
#:
#: ``"hold"``
#:     The machine is the reference and never concedes to CLOCK-LEVEL DRIFT.
#:     The offset is still measured, still bounded and still logged every bar
#:     — it is just not answered with a tempo change.
#: ``"meet"``
#:     Mutual entrainment: the v2.0/v2.1 concede rules, with every guard
#:     (plausibility, cross-bar agreement, the one-bar ramp clamp, the
#:     unwrap-against-prediction fix, the swing term).  Unchanged; no longer
#:     the default in lead mode.
#: ``"chase"``
#:     The human owns the clock (v1 follow semantics).  The negotiator holds
#:     no tempo of its own — the caller's phase correction is what follows the
#:     human, and a concession layered on top of it would be a second, slower
#:     copy of the same correction fighting the first.
TEMPO_POLICIES = ("hold", "meet", "chase")

#: role -> default tempo policy when ``RoleParams.tempo_policy`` is unset.
#: ``lead`` holds: the rhythm section IS the reference, and a reference that
#: moves every time a player pushes is not one.
ROLE_TEMPO_POLICY: dict[str, str] = {
    "lead": "hold",
    "follow": "chase",
    "trade": "meet",
    "solo_support": "meet",
}


def resolve_tempo_policy(role: str, tempo_policy: str | None) -> str:
    """The policy in force for ``role`` — the explicit one, or the role's own.

    ``None`` means "whatever this chair implies", which is why
    :class:`RoleParams` defaults it that way rather than to a literal
    ``"hold"``: ``RoleParams``' own ``role`` default is ``follow``, and a
    follower that holds its own clock is not a follower.  Same shape as
    ``clock_authority``/``form_authority``, which have meant "None -> the
    role's default" since v2.0.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if tempo_policy is None:
        return ROLE_TEMPO_POLICY[role]
    if tempo_policy not in TEMPO_POLICIES:
        raise ValueError(
            f"tempo_policy must be one of {TEMPO_POLICIES} "
            f"(or unset for the role's default), got {tempo_policy!r}")
    return str(tempo_policy)

# --- tuning constants that are NOT user-facing parameters -------------------

#: Hard ceiling on how far ``tempo_offset_pct`` may move in a single bar, in
#: percentage points.  A concession is a *ramp*, and SPEC-V2 section 7 fixes
#: the audible bound at 2%/bar: beyond that the tempo change stops reading as
#: two players meeting and starts reading as a tape speeding up.
MAX_STEP_PCT_PER_BAR = 2.0

#: How fast an authority dial slides toward its target role's default, per bar.
#: Role changes are decisions; authority handovers are gestures.  0.25 puts a
#: full lead<->follow handover at three bars — long enough to hear, short
#: enough to be over inside a phrase.
AUTHORITY_SLEW = 0.25

#: Fraction of an OBSERVED drift rate absorbed by the phase-return term, per
#: event.  Not ``concede_rate``: that dial is about how much of a partner's
#: expressive PUSH to take on, which is a musical choice, while a drift rate is
#: a direct measurement of how far apart the two clocks' tempi are — there is
#: nothing to be tactful about, and under-correcting only means the two stay
#: apart longer.  Kept under 1.0 so a noisy rate estimate cannot ring.  Swept
#: 0.25/0.4/0.6/0.8/1.0: below 0.6 the correction is slower than the gap opens
#: (the synthetic tape's residual stays past 500 ms); 0.8 is the best measured
#: (tape 130 ms, entrainment peak 30 ms) and 1.0 gains nothing.
_DRIFT_RATE = 0.8

#: Consecutive differences the phase-return term needs before it acts.  Four,
#: rather than ``concede_bars``' two, and the extra bars are DAMPING rather
#: than caution: MEASURED on the mutual-entrainment loop, two differences let
#: the term fire while a partner who is ALSO bending is still absorbing the
#: same gap, and the two corrections rang — a limit cycle at 59-87 ms peak with
#: a period of about eleven bars, where the same loop settles at 30 ms once the
#: window outlasts a partner's own reaction.  Swept 2/3/4/5/6 against the whole
#: constraint set (both real-material sign guards, the four-take separation,
#: the entrainment settle and the tape's residual): 4 is the smallest value
#: that satisfies all of them, and 5-6 only cost tape residual.
_DRIFT_DIFFS = 4


#: The machine only concedes tempo when it is at least half responsible for the
#: clock.  Below this the v1 phase-correction already pulls the machine onto the
#: human's grid, and a concession on top of it would be a second, slower copy of
#: the same correction fighting the first.
CONCEDE_MIN_CLOCK = 0.5

#: EMA coefficient for the human's own density baseline (see ``_observe``).
#: Not a musical dial — it only has to be slower than a phrase and faster than
#: a section.
_DENSITY_BASE_ALPHA = 0.3


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _pct_str(x: float) -> str:
    """Signed percent for a reason string, without the '-0.0%' papercut."""
    return f"{x + 0.0:+.1f}" if abs(x) >= 0.05 else "+0.0"


def _ms_str(x: float) -> str:
    return f"{x + 0.0:+.0f}" if abs(x) >= 0.5 else "+0"


def effective_bpm(composed_bpm: float, tempo_offset_pct: float) -> float:
    """The tempo the machine is actually playing, given its accumulated
    concession.  ``tempo_offset_pct`` is a percentage of the *composed* tempo."""
    return float(composed_bpm) * (1.0 + float(tempo_offset_pct) / 100.0)


def trade_turn(bar: int, trade_bars: int) -> str:
    """``"machine"`` or ``"human"`` — whose bars these are, in trade mode.

    A pure function of the bar number, so trading has no state to desynchronise
    and a test can assert the whole grid without running a session.  The machine
    takes the FIRST unit: in ``lead``/``trade`` the machine is what started, so
    the human's first entry is an answer, not a cold open.
    """
    if trade_bars <= 0:
        raise ValueError(f"trade_bars must be positive, got {trade_bars}")
    if bar < 0:
        return "machine"
    return "machine" if (bar // trade_bars) % 2 == 0 else "human"


def grid_offset_ms(beat: BeatState, machine_grid_t_s: float, *,
                   interval_s: float | None = None) -> float | None:
    """Signed offset of the human's beat grid from the machine's, in ms.

    **Positive = the human is AHEAD of the machine (rushing)**, which is the
    sign convention SPEC-V2's own example fixes ("human +31ms ... 120.0 ->
    121.9 bpm": a human out in front makes the machine speed up).  Negative =
    dragging.

    ``beat.next_beat_t_s`` is a predicted *future* beat, so the raw difference
    is wrapped by one beat period into [-interval/2, +interval/2).  The wrap
    uses the MACHINE's period when the caller supplies one: we are measuring the
    human's phase against the machine's grid, and if the tracker has settled on
    an octave relative of the human's tempo, the machine's period is the window
    that keeps a real drag from wrapping into an apparent rush.

    Returns ``None`` when there is no usable grid (no tempo estimate yet, a
    non-finite prediction, or a non-finite machine grid time) — "no evidence" and
    "zero offset" must never be the same value.
    """
    interval = float(interval_s) if interval_s else float(beat.beat_interval_s)
    if not math.isfinite(interval) or interval <= 0.0:
        return None
    if not math.isfinite(float(beat.next_beat_t_s)):
        return None
    if not math.isfinite(float(machine_grid_t_s)):
        return None
    raw = float(machine_grid_t_s) - float(beat.next_beat_t_s)
    off = raw - interval * round(raw / interval)
    return off * 1000.0


def onset_phase_offset_ms(onset_times_s, t0: float, t1: float,
                          unit_s: float, *, min_onsets: int = 3,
                          swing: float = 0.5) -> float | None:
    """Signed offset of a bar's onsets from the machine's subdivision grid, ms.

    Same sign convention as :func:`grid_offset_ms`: **positive = the human is
    ahead (rushing)**.  ``unit_s`` is the machine's subdivision (a sixteenth is
    the useful choice for programmed material), ``[t0, t1)`` the bar.

    This is the estimator ``RoleNegotiator.update``'s ``human_offset_ms``
    argument wants, and it exists here because the concede rule is *specified*
    against "the human's detected beats" and a caller should not have to invent
    the measurement.  It is a circular mean of onset phase, which is what makes
    it work: every onset in the bar votes regardless of which subdivision it
    landed on, so a kick, a backbeat and a run of hats all measure the same
    displacement instead of the estimate depending on which of them the matcher
    happened to accept.

    MEASURED (32-bar v0 renders, at 0 dB and -20 dB, shifted by a known
    amount): recovers a +30ms rush as +24.0/+18.5ms and a -35ms lay-back as
    -37.0/-44.5ms on trap140, with a per-bar IQR of 9-17ms.  On swung lo-fi it
    carries a standing -20ms bias — which is the swing, and a swung human IS
    laying back — while still recovering the *change* to within 4ms.  Compare
    :func:`grid_offset_ms`, which on the same material barely separates those
    three conditions at all.

    ``swing`` (v2.1 amendment, SPEC-V2 section 1) is the MACHINE's own swing —
    :class:`plume.groove.GrooveParams`' Roger-Linn knob, 0.5 straight, 0.66
    exact triplet.  It must be passed, because the machine's grid is not a
    straight sixteenth grid whenever its style swings, and measuring a human
    against a grid the machine is not playing turns *the shared groove* into
    evidence that the human is laying back.  MEASURED, and this is what forced
    it: a v0 lo-fi render (swing 0.57) against a machine playing the same lo-fi
    style at the same tempo — two players grooving identically — read as a
    standing -30 ms lay-back and drove real concessions.  A swung sixteenth is
    displaced ``(swing-0.5)*2`` of a subdivision, i.e. 35 ms at 120 bpm, which
    is larger than ``concede_ms``: the whole rule fires on the groove.  Each
    onset is therefore de-swung by its own slot's displacement before it votes.
    At ``swing = 0.5`` this is arithmetically the identity and the result is
    bit-identical to the straight-grid version.

    **Range, and it is a hard limit of the instrument.** This is a phase
    measurement on a periodic grid, so it is only unambiguous within
    ``±unit_s/2`` (±62.5 ms against a sixteenth at 120 bpm).  A displacement
    past that wraps and comes back with the WRONG SIGN — 70 ms ahead reads
    -55 ms.  Callers must bound what they believe accordingly;
    :class:`RoleNegotiator` does it with ``concede_plausible_unit_frac``, and
    the unit has to be told to it (``human_offset_unit_s``) for that bound to
    exist at all.

    Vectorised; ``None`` when the bar has fewer than ``min_onsets`` onsets
    (a bar with two hits in it cannot tell you about the groove).
    """
    if not math.isfinite(unit_s) or unit_s <= 0.0:
        return None
    t = np.asarray(onset_times_s, dtype=np.float64).ravel()
    sel = t[(t >= float(t0)) & (t < float(t1))]
    if sel.size < int(min_onsets):
        return None
    rel = sel - float(t0)
    sw = float(swing)
    if math.isfinite(sw) and abs(sw - 0.5) > 1e-9:
        # Which subdivision each onset belongs to, then that slot's own swing
        # displacement removed.  Slot assignment is a rounding whose error is
        # under half a subdivision by construction, and the displacement is at
        # most 0.32 of one, so the assignment cannot be walked off its slot.
        slot = np.rint(rel / unit_s)
        shifted = (np.mod(slot, 4.0) == 1.0) | (np.mod(slot, 4.0) == 3.0)
        rel = rel - np.where(shifted, (sw - 0.5) * 2.0 * unit_s, 0.0)
    ang = 2.0 * np.pi * ((rel / unit_s) % 1.0)
    mean = float(np.angle(np.mean(np.exp(1j * ang))))     # (-pi, pi]
    return -(mean / (2.0 * np.pi)) * unit_s * 1000.0


@dataclass
class RoleParams:
    """Every role-negotiation threshold; defaults are the SPEC-V2 values.

    All fields are settable from the CLI (``--roles k=v``); ``PARAM_HELP``
    carries the human-readable meaning of each for ``plume roles``.

    The trailing blocks are v2.0/v2.1/v2.2 amendments (see SPEC-V2 section 1)
    and are appended rather than grouped so that the spec-frozen field order is
    intact.  The last of them, ``tempo_policy``, is the one that changed what
    lead mode DOES rather than adding a threshold to it.
    """

    role: str = "follow"                  # starting role
    clock_authority: float | None = None  # None -> role default
    form_authority: float | None = None   # None -> role default
    # --- mutual entrainment ---
    concede_ms: float = 25.0
    concede_bars: int = 2
    concede_rate: float = 0.25
    concede_max_pct: float = 8.0
    # --- role handoff ---
    handoff_silence_bars: int = 8
    handoff_density: float = 2.0
    handoff_hysteresis_bars: int = 4
    trade_bars: int = 4
    # --- harmonic meeting ---
    conform_semitones: int = 2
    conform_bars: int = 4
    conform_enabled: bool = True
    # --- v2.0 amendments (additive, appended; every one of these was put
    #     here by a measurement on real material — see the module docstring's
    #     "measured on real material" note and SPEC-V2 section 1) ---
    concede_min_conf: float = 0.45        # beat confidence required to trust an offset
    concede_plausible_frac: float = 0.2   # offsets beyond this fraction of a beat are not timing
    concede_spread_ms: float = 25.0       # the window's samples must agree within this
    conform_key_conf: float = 0.4         # key confidence required to trust a key belief
    handoff_density_ratio: float = 1.5    # ...and this much busier than they had been
    # --- v2.1 amendments (additive, appended; see SPEC-V2 section 1's v2.1
    #     block — the aliasing bound and the phase-return term) ---
    concede_plausible_unit_frac: float = 0.4  # ...and this fraction of the SUBDIVISION
    concede_drift_ms: float = 12.0        # per-bar phase drift that counts as a tempo gap
    # --- v2.2 amendment (additive, appended; see SPEC-V2 section 1's v2.2
    #     block and docs/decisions/2026-08-15-tempo-authority.md) ---
    #: ``None`` -> the role's own default (:data:`ROLE_TEMPO_POLICY`): lead
    #: holds, follow chases, trade and solo_support meet.  A literal default of
    #: ``"hold"`` would be wrong here — ``RoleParams``' own ``role`` default is
    #: ``follow``, and a follower that holds its own clock is not a follower —
    #: so this uses the same "None means the role's default" shape that
    #: ``clock_authority``/``form_authority`` have had since v2.0, and the same
    #: shape the CLI already knows how to parse (``--roles tempo_policy=meet``,
    #: ``--roles tempo_policy=none``).
    tempo_policy: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")
        for name in ("clock_authority", "form_authority"):
            v = getattr(self, name)
            if v is not None:
                v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"{name} must be in [0, 1], got {v}")
                setattr(self, name, v)
        if self.concede_ms < 0.0:
            raise ValueError(f"concede_ms must be >= 0, got {self.concede_ms}")
        if self.concede_bars < 1:
            raise ValueError(f"concede_bars must be >= 1, got {self.concede_bars}")
        if not (0.0 <= self.concede_rate <= 1.0):
            raise ValueError(f"concede_rate must be in [0, 1], got {self.concede_rate}")
        if self.concede_max_pct < 0.0:
            raise ValueError(
                f"concede_max_pct must be >= 0, got {self.concede_max_pct}")
        if self.handoff_silence_bars < 1:
            raise ValueError(
                f"handoff_silence_bars must be >= 1, got {self.handoff_silence_bars}")
        if self.handoff_density <= 0.0:
            raise ValueError(
                f"handoff_density must be > 0, got {self.handoff_density}")
        if self.handoff_hysteresis_bars < 1:
            raise ValueError("handoff_hysteresis_bars must be >= 1, got "
                             f"{self.handoff_hysteresis_bars}")
        if self.trade_bars < 1:
            raise ValueError(f"trade_bars must be >= 1, got {self.trade_bars}")
        if not (0 <= self.conform_semitones <= 6):
            raise ValueError("conform_semitones must be in [0, 6] (a tritone is "
                             f"the furthest any key can be), got {self.conform_semitones}")
        if self.conform_bars < 1:
            raise ValueError(f"conform_bars must be >= 1, got {self.conform_bars}")
        if not (0.0 < self.concede_plausible_frac <= 0.5):
            raise ValueError("concede_plausible_frac must be in (0, 0.5] (half a "
                             "beat away is the next beat), got "
                             f"{self.concede_plausible_frac}")
        if self.concede_spread_ms < 0.0:
            raise ValueError(
                f"concede_spread_ms must be >= 0, got {self.concede_spread_ms}")
        if self.handoff_density_ratio < 1.0:
            raise ValueError("handoff_density_ratio must be >= 1 (a handover is a "
                             "human getting BUSIER), got "
                             f"{self.handoff_density_ratio}")
        if not (0.0 < self.concede_plausible_unit_frac <= 0.5):
            raise ValueError(
                "concede_plausible_unit_frac must be in (0, 0.5] (half a "
                "subdivision away is the next subdivision, and a phase "
                "measurement cannot see past it), got "
                f"{self.concede_plausible_unit_frac}")
        if self.concede_drift_ms <= 0.0:
            raise ValueError(
                f"concede_drift_ms must be > 0, got {self.concede_drift_ms}")
        # Raises listing the valid values (and says that unset is legal).
        resolve_tempo_policy(self.role, self.tempo_policy)

    @property
    def effective_tempo_policy(self) -> str:
        """The policy the STARTING role implies, resolved.

        Read-only convenience for callers and logs.  The negotiator resolves
        per bar against its CURRENT role rather than this one, because a
        handoff changes the chair (see ``RoleNegotiator.tempo_policy``).
        """
        return resolve_tempo_policy(self.role, self.tempo_policy)


#: field name -> one-line meaning, for `plume roles` (kept in sync by tests).
PARAM_HELP: dict[str, str] = {
    "role": f"starting role, one of {'/'.join(ROLES)}",
    "clock_authority": "who owns time, 0=human 1=machine (unset = the role's default)",
    "form_authority": "who owns the form, 0=human 1=machine (unset = the role's default)",
    "concede_ms": "sustained human beat offset (ms) that starts a tempo concession",
    "concede_bars": "bars of consistent push/drag required before conceding",
    "concede_rate": "fraction of the human's deviation absorbed per concession",
    "concede_max_pct": "never drift further than this % from the composed tempo",
    "handoff_silence_bars": "human lays out this long while we follow -> we take the lead",
    "handoff_density": "human onsets/beat above this while we lead -> we yield",
    "handoff_hysteresis_bars": "sustain, and minimum bars between role changes (no flapping)",
    "trade_bars": "trading unit in bars ('fours')",
    "conform_semitones": "transpose to the human's key when it is within this distance",
    "conform_bars": "bars of consistent out-of-key evidence before transposing",
    "conform_enabled": "whether to meet the human harmonically at all",
    "concede_min_conf": "beat confidence required before an offset is trusted",
    "concede_plausible_frac": "offsets beyond this fraction of a beat are phase error, not timing",
    "concede_spread_ms": "the offsets in the window must agree within this to count as consistent",
    "conform_key_conf": "key confidence required before a key belief is trusted",
    "handoff_density_ratio": "yielding also needs the human this much busier than their own baseline",
    "concede_plausible_unit_frac": "offsets beyond this fraction of the measured subdivision are the next subdivision",
    "concede_drift_ms": "per-bar phase drift that means the two clocks are at different tempi",
    "tempo_policy": ("whose clock it is: hold (never chase drift) / meet "
                     "(entrain) / chase (they own it); unset = the role's "
                     "default (lead=hold follow=chase trade=meet)"),
}


@dataclass(frozen=True)
class RoleState:
    """The machine's authority over the UPCOMING bar.

    The first eight fields are the SPEC-V2 contract.  The last three are v2.0
    amendments, appended and defaulted so that every spec-shaped construction
    still works:

    ``tempo_target_pct``
        Where ``tempo_offset_pct`` is ramping TO by the end of this bar.  A
        concession is specified as "a ramp applied over the following bar", and
        a renderer cannot ramp toward a number it cannot see.
    ``bpm``
        The tempo the machine is actually playing at the START of this bar
        (``composed_bpm * (1 + tempo_offset_pct/100)``), so a log line does not
        have to re-derive it from a tempo it was never told.
    ``events``
        Machine-readable tags for what happened this bar — ``"concede"``,
        ``"concede_capped"``, ``"tempo_match"``, ``"tempo_hold"``, ``"yield"``,
        ``"take_lead"``, ``"trade"``, ``"trade_collision"``, ``"conform"``,
        ``"conform_hold"``.  ``reason`` is the contract for humans; this is the
        contract for assertions, so a behavior test never has to grep prose.

    The last two are v2.2 amendments, appended and defaulted the same way:

    ``human_offset_ms``
        The human's signed offset from the machine's grid this bar, in ms,
        **whatever the policy did with it** — positive is ahead/rushing, and
        ``None`` means no usable measurement (not "zero"; those must never be
        the same value).  Under ``tempo_policy="hold"`` this is the whole
        output of the measurement: it is what makes a hold auditable and it is
        the practice-feedback signal.  Populated identically under ``meet``,
        where it is also the input to the concede rules.
    ``tempo_policy``
        Which of :data:`TEMPO_POLICIES` was in force for this bar, so a log
        line does not have to be cross-referenced against the params to be
        read.  Defaults to ``"meet"`` — a hand-built, spec-shaped ``RoleState``
        describes the pre-v2.2 semantics.
    """

    t_s: float
    bar: int
    role: str
    clock_authority: float
    form_authority: float
    tempo_offset_pct: float          # accumulated concession from the composed tempo
    transpose_semitones: int         # accumulated harmonic conformance
    reason: str                      # human-readable, contract
    # --- v2.0 amendments (additive, appended) ---
    tempo_target_pct: float = 0.0
    bpm: float = 0.0
    events: tuple[str, ...] = ()
    # --- v2.2 amendments (additive, appended) ---
    human_offset_ms: float | None = None
    tempo_policy: str = "meet"


def _key_name(tonic_pc: int, mode: str) -> str:
    """'F# minor', 'Eb major' — spelled the way the key itself would spell it,
    and re-parseable by :func:`plume.theory.parse_key`."""
    k = Key(tonic=int(tonic_pc) % 12, mode=mode)
    return f"{spell(k.tonic, k)} {mode.replace('_', ' ')}"


def _signed_pc_distance(from_pc: int, to_pc: int) -> int:
    """Shortest signed semitone distance, in [-5, +6].

    A tritone is equidistant either way; it is canonicalised to +6 so that the
    "too far to conform" log line reads ``6 st`` rather than an arbitrary sign.
    """
    d = ((int(to_pc) - int(from_pc) + 6) % 12) - 6
    return 6 if d == -6 else d


#: mode -> semitones from the mode's tonic DOWN to the root of its diatonic
#: pitch-class collection (its relative major).  ``harmonic_minor`` is not
#: diatonic; its natural-minor collection is the closest thing it has.
_COLLECTION_OFFSET: dict[str, int] = {
    "major": 0, "minor": 3, "dorian": -2, "phrygian": -4, "harmonic_minor": 3,
}


def _consistent_window(offsets: list[float], concede_bars: int,
                       spread_ms: float) -> list[float] | None:
    """The last ``concede_bars`` offsets if they are evidence, else ``None``.

    Evidence means all on one side of the grid AND agreeing to within
    ``spread_ms``: a human sitting steadily in front of or behind the beat.
    Without the spread test, two bars 100ms apart that merely fell on the same
    side get averaged into a number neither bar ever measured, and MEASURED on
    plume's own renders that really did move the tempo.

    **Known limit, deliberate.**  A human at a slightly different TEMPO produces
    a phase error that *ramps* — 25.3ms per bar at a 1.25% difference — and a
    ramp fails the spread test, so this rule will not catch it.  That is the
    right trade: at ``concede_bars = 2`` a "ramp" is two same-signed samples,
    which noise produces half the time by chance, and admitting them was
    MEASURED to walk a matched 140bpm take to 6.0% drift (versus 1.5% with the
    spread test alone) while buying nothing stable in return — the phase->period
    law is an integrator, so against a partner who does not also bend it
    oscillates rather than locks whatever it is fed.  A genuine tempo difference
    is the beat tracker's job to report (``BeatState.bpm``), not this rule's to
    infer from phase.  What this rule is for is the other thing, the one the
    tracker cannot tell you: two players at the same tempo, one of them pushing.
    """
    tail = offsets[-concede_bars:]
    if len(tail) < concede_bars:
        return None
    if not (all(x > 0.0 for x in tail) or all(x < 0.0 for x in tail)):
        return None
    if max(tail) - min(tail) > spread_ms:
        return None
    return tail


def collection_root(key: Key) -> int:
    """Pitch class of the root of ``key``'s diatonic collection.

    v2.0 AMENDMENT (SPEC-V2 section 1, behavior 5).  The spec writes the
    conform rule as a distance between key *tonics*, but a pitch-class
    histogram — which is all ``HarmonyTracker`` has — **cannot distinguish a key
    from its relative in principle** (``harmony.py`` says so, and
    ``test_real_material`` accepts either).  So on real material the reported
    tonic is a coin flip between two answers three semitones apart, and a rule
    written on tonics is a rule written on a coin flip: F minor against a human
    playing F minor reads as 3 st away (they were called Ab major) and logs a
    spurious hold, while F# minor against a human in F minor reads as 2 st away
    (Ab major again) and would transpose us to the WRONG key.

    Distance is therefore measured between collection roots, which are stable
    under exactly that ambiguity: F minor and Ab major both root at Ab, so a
    human "in our key" is 0 st away however the tracker spelled it, and a
    capo'd human a semitone up is +1 st away however the tracker spelled THEM.
    The transposition this yields still lands the machine in the human's key
    with the machine's own mode intact, which is what conforming means.
    """
    return (key.tonic + _COLLECTION_OFFSET.get(key.mode, 0)) % 12


class RoleNegotiator:
    """The role state machine.  Pure, deterministic, rng-free.

    ``composed_bpm`` / ``composed_key`` are v2.0 amendments to the SPEC-V2
    ``__init__(self, params)`` signature — keyword-only and defaulted, so every
    spec-shaped construction still works.  They are *session context*, not
    negotiation parameters: the spec's own reason string quotes absolute tempi
    ("120.0 -> 121.9 bpm") and the conform rule is defined against "the
    machine's key", neither of which the negotiator can otherwise know.  A
    caller that omits them gets correct dial behavior and a reason string
    written against the 120 bpm / key-less default.
    """

    def __init__(self, params: RoleParams, *, composed_bpm: float = 120.0,
                 composed_key: str | None = None):
        if not math.isfinite(float(composed_bpm)) or float(composed_bpm) <= 0.0:
            raise ValueError(f"composed_bpm must be positive, got {composed_bpm}")
        if composed_key is not None:
            parse_key(composed_key)          # fail loudly, at construction
        self.params = params
        self.composed_bpm = float(composed_bpm)
        self.composed_key = composed_key

        role = params.role
        clock, form = ROLE_DEFAULTS[role]
        self._role = role
        self._clock = float(params.clock_authority
                            if params.clock_authority is not None else clock)
        self._form = float(params.form_authority
                           if params.form_authority is not None else form)
        # Explicit authority overrides pin the STARTING role's dials only; once
        # a handoff happens the new role's defaults are the target.
        self._overrides_live = (params.clock_authority is not None
                                or params.form_authority is not None)

        self._bars_since_role_change = params.handoff_hysteresis_bars  # armed at t=0
        self._dense_bars = 0             # consecutive busy human bars
        self._dens_base: float | None = None    # what "busy for them" means
        self._silent_bars = 0            # consecutive silent/absent human bars
        self._ever_locked = False        # we have had a real grid at some point

        self._offsets: list[float] = []  # per-bar signed offsets, ms (window)
        self._phase_prev: float | None = None   # last admitted phase, unwrapped
        #: the plausibility bound in force for the last admitted reading; a
        #: beat-fraction default until a caller tells us the subdivision.
        #: The plausibility bound in force for the last admitted reading.
        #: Two of them: the STEP bound (tightened to the instrument's own
        #: subdivision once a caller says what it measured against) and the
        #: PUSH bound (always the beat fraction — how far out a human can
        #: expressively sit is a fact about players, not about the grid the
        #: measurement happened to use).
        _iv = 60.0 / self.composed_bpm
        self._bound_ms = params.concede_plausible_frac * _iv * 1000.0
        self._push_bound_ms = self._bound_ms
        self._interval_s = _iv
        self._tempo_pct = 0.0            # current, ramped
        self._tempo_target_pct = 0.0     # where the ramp is heading
        #: This bar's admitted offset, measured under every policy (v2.2).
        #: ``None`` = no usable evidence; never conflated with 0.0 ms.
        self._last_offset_ms: float | None = None

        self._transpose = 0
        self._key_run_dist: int | None = None   # signed distance under observation
        self._key_run_bars = 0
        self._hold_logged = False        # "too far to conform" said once per run

        self._trade_side: str | None = None     # last announced trade turn

    # ------------------------------------------------------------------ API

    @property
    def tempo_policy(self) -> str:
        """The policy in force right now (v2.2).

        Resolved against the CURRENT role, not the starting one, so an unset
        policy follows the chair: a lead that yields becomes a follower and
        stops holding a clock it no longer owns, and takes the hold back with
        the lead.  An EXPLICIT ``tempo_policy`` is a pin and never moves — the
        same distinction ``_overrides_live`` draws for the authority dials.
        """
        return resolve_tempo_policy(self._role, self.params.tempo_policy)

    def update(self, *, bar: int, beat: BeatState, harmony: HarmonyState,
               human: "BarFeatures | None", machine_grid_t_s: float,
               human_offset_ms: float | None = None,
               human_offset_unit_s: float | None = None) -> RoleState:
        """Negotiate the upcoming bar.

        ``bar`` is the UPCOMING bar (the same convention as ``BarPlan.bar``);
        ``human`` describes the bar just completed, or ``None`` when there is no
        completed human bar (they have not joined, or the tracker is unlocked).
        ``machine_grid_t_s`` is the machine's own grid time for this boundary —
        the machine's clock, not the human's.

        ``human_offset_ms`` (v2.0 amendment, additive) lets a caller that can
        measure the offset better than a predicted-beat difference — ``sim``
        matches detected onsets against the machine's grid — supply it directly.
        It replaces the *derivation* only; ``concede_min_conf`` still applies.

        ``human_offset_unit_s`` (v2.1 amendment, additive) is the SUBDIVISION
        that measurement was made against.  A phase measurement on a periodic
        grid is only unambiguous within half its own period, and without being
        told the period the negotiator cannot know where that edge is — which
        is the whole of the aliasing defect this argument exists to close (see
        the v2.1 block in the ``_negotiate_tempo`` body).  Passing the offset
        without the unit still works and is still gated, just by the coarser
        beat-fraction bound alone.

        **The lock gate applies to the DERIVED estimator only** (v2.1).
        ``grid_offset_ms`` reads ``beat.next_beat_t_s`` and
        ``beat.beat_interval_s``, which mean nothing before the tracker locks,
        so there the gate is load-bearing.  A measured ``human_offset_ms``
        never touches ``BeatState``: it is the human's own attack times against
        the MACHINE's grid, and the machine's grid is generated rather than
        detected, so it is exactly known from t=0.  Gating that measurement on
        a different estimator's lock made the response time an artefact of
        tracker latency — MEASURED on the section-6 joiners, the tracker locks
        a bar later on ``dragging`` than on ``rushing`` (the same take, 35 ms
        later), so behavior 3 fired a bar after behavior 2 for a reason with
        nothing to do with conceding, and SPEC-V2 section 7 specifies them as
        symmetric.

        Called exactly once per bar boundary.
        """
        conf = float(beat.confidence)
        if beat.locked:
            self._ever_locked = True

        t_s = (float(machine_grid_t_s) if math.isfinite(float(machine_grid_t_s))
               else float(beat.t_s))
        events: list[str] = []
        reasons: list[str] = []
        self._last_offset_ms = None

        # 1) Consume the previous bar's ramp before anything reads the tempo.
        self._advance_ramp()

        # 2) Per-bar evidence.
        self._observe(human)

        # 3) Role transitions (discrete, hysteresis-guarded) and authority.
        self._negotiate_role(bar, human, events, reasons)

        # 4) Tempo concession.
        self._negotiate_tempo(beat, conf, human, machine_grid_t_s,
                              human_offset_ms, human_offset_unit_s,
                              events, reasons)

        # 5) Harmonic conformance.
        self._negotiate_key(harmony, events, reasons)

        reason = "; ".join(reasons) if reasons else self._steady_reason()
        return RoleState(
            t_s=t_s, bar=int(bar), role=self._role,
            clock_authority=round(self._clock, 6),
            form_authority=round(self._form, 6),
            tempo_offset_pct=round(self._tempo_pct, 6),
            transpose_semitones=int(self._transpose),
            reason=reason,
            tempo_target_pct=round(self._tempo_target_pct, 6),
            bpm=round(effective_bpm(self.composed_bpm, self._tempo_pct), 6),
            events=tuple(events),
            human_offset_ms=(None if self._last_offset_ms is None
                             else round(float(self._last_offset_ms), 6)),
            tempo_policy=self.tempo_policy,
        )

    # ------------------------------------------------------------ internals

    def _advance_ramp(self) -> None:
        """Move ``tempo_offset_pct`` toward its target, at most
        ``MAX_STEP_PCT_PER_BAR`` percentage points per bar."""
        gap = self._tempo_target_pct - self._tempo_pct
        if abs(gap) <= 1e-12:
            self._tempo_pct = self._tempo_target_pct
            return
        step = _clamp(gap, -MAX_STEP_PCT_PER_BAR, MAX_STEP_PCT_PER_BAR)
        self._tempo_pct += step

    def _ramping(self) -> bool:
        return abs(self._tempo_target_pct - self._tempo_pct) > 1e-9

    def _observe(self, human: "BarFeatures | None") -> None:
        p = self.params
        self._bars_since_role_change += 1
        if human is None or bool(human.silent):
            # An absent bar and a silent bar are the same thing to the handoff
            # rules: nobody is playing, and the floor is open.
            self._silent_bars += 1
            self._dense_bars = 0
            return
        self._silent_bars = 0
        dens = float(human.onset_density)
        if self._dens_base is None:
            self._dens_base = dens
        if dens > p.handoff_density:
            # v2.0 AMENDMENT (SPEC-V2 section 1, behavior 2).  The spec's yield
            # rule is a bare absolute threshold, and MEASURED on plume's own
            # 140bpm trap render the human sits at a steady 5.0 onsets/beat —
            # two and a half times the default — so in lead mode against any
            # real programmed material the machine handed the lead away at the
            # first hysteresis window and never took it back.  Hats are not a
            # takeover.  A takeover is the human getting BUSIER than they have
            # been, so the absolute gate now carries a relative one beside it.
            # The `takes_over` fixture (density RAMPS from ~1 to 4/beat) passes
            # both, which is the behavior the spec's own test asks for.
            if dens > self._dens_base * p.handoff_density_ratio:
                self._dense_bars += 1
            else:
                self._dense_bars = 0
            # Baseline frozen during a burst: it is what they were doing BEFORE
            # this, and an EMA that eats the burst normalises the takeover away
            # before the hysteresis window can close.
        else:
            self._dense_bars = 0
            self._dens_base += _DENSITY_BASE_ALPHA * (dens - self._dens_base)

    # --- authority -------------------------------------------------------

    def _targets(self, bar: int) -> tuple[float, float]:
        if self._role == "trade":
            side = trade_turn(bar, self.params.trade_bars)
            return ROLE_DEFAULTS["lead" if side == "machine" else "follow"]
        base = ROLE_DEFAULTS[self._role]
        if self._overrides_live:
            p = self.params
            return (p.clock_authority if p.clock_authority is not None else base[0],
                    p.form_authority if p.form_authority is not None else base[1])
        return base

    def _slide(self, targets: tuple[float, float]) -> None:
        for attr, want in (("_clock", targets[0]), ("_form", targets[1])):
            have = getattr(self, attr)
            gap = float(want) - have
            step = _clamp(gap, -AUTHORITY_SLEW, AUTHORITY_SLEW)
            setattr(self, attr, _clamp(have + step, 0.0, 1.0))

    def _set_role(self, role: str) -> None:
        self._role = role
        self._overrides_live = False      # a handoff retires the starting pin
        self._bars_since_role_change = 0

    def _negotiate_role(self, bar: int, human: "BarFeatures | None",
                        events: list[str], reasons: list[str]) -> None:
        p = self.params

        if self._role == "trade":
            self._trade(bar, human, events, reasons)
            return

        hyst_ok = self._bars_since_role_change >= p.handoff_hysteresis_bars
        before = self._clock

        if (self._role == "follow" and hyst_ok
                and self._silent_bars >= p.handoff_silence_bars
                and self._ever_locked):
            # Take the lead.  ``_ever_locked`` is the "machine still has a
            # groove" gate: without a grid we ever agreed on, holding the floor
            # would be starting a different tune, not continuing this one.
            n = self._silent_bars
            self._set_role("lead")
            self._slide(self._targets(bar))
            events.append("take_lead")
            reasons.append(
                f"taking the lead: human out {n} bars "
                f"(>= {p.handoff_silence_bars}) -> follow -> lead, "
                f"clock {before:.2f} -> {self._clock:.2f}")
            return

        if (self._role == "lead" and hyst_ok
                and self._dense_bars >= p.handoff_hysteresis_bars):
            d = float(human.onset_density) if human is not None else 0.0
            n = self._dense_bars
            self._set_role("follow")
            self._slide(self._targets(bar))
            events.append("yield")
            reasons.append(
                f"yielding: human {d:.1f} onsets/beat for {n} bars "
                f"(> {p.handoff_density:.1f}) -> lead -> follow, "
                f"clock {before:.2f} -> {self._clock:.2f}")
            return

        self._slide(self._targets(bar))

    def _trade(self, bar: int, human: "BarFeatures | None",
               events: list[str], reasons: list[str]) -> None:
        """Alternate authority every ``trade_bars``, human wins collisions.

        v2.0 AMENDMENT (SPEC-V2 section 1, behavior 4).  The spec says the
        machine backs off "within a beat" when the human plays over its bars.
        ``RoleState`` is a per-bar contract and this negotiator only ever sees
        completed bars, so what is decided here is "they played over my last
        bar, so this one is theirs"; the sub-bar reaction is the renderer's job
        (it has the authority number a full bar before it needs it).  The dials
        are set, not slid: a three-bar slew would smear a four-bar trade into
        mush, and the whole point of trading fours is that the handover lands on
        the boundary.
        """
        p = self.params
        side = trade_turn(bar, p.trade_bars)
        collided = (side == "machine" and human is not None
                    and not bool(human.silent)
                    and float(human.onset_density) > p.handoff_density)
        if collided:
            self._clock, self._form = ROLE_DEFAULTS["follow"]
            events.append("trade_collision")
            reasons.append(
                f"trading: human played over my bar "
                f"({float(human.onset_density):.1f} onsets/beat > "
                f"{p.handoff_density:.1f}) — backing off to clock "
                f"{self._clock:.2f}")
            # NOT a turn change: the backoff is a one-bar courtesy, so the turn
            # counter is untouched and the machine simply resumes its own bars
            # next bar without re-announcing the trade.
            return
        self._clock, self._form = self._targets(bar)
        if side != self._trade_side:
            self._trade_side = side
            events.append("trade")
            lo = (bar // p.trade_bars) * p.trade_bars
            whose = "mine" if side == "machine" else "yours"
            reasons.append(
                f"trading: bars {lo}-{lo + p.trade_bars - 1} are {whose} "
                f"(clock {self._clock:.2f} form {self._form:.2f})")

    # --- tempo -----------------------------------------------------------

    def _expected_drift_ms(self, beat: BeatState, bar_s: float) -> float:
        """How much phase the two clocks are EXPECTED to open this bar, in ms.

        v2.1 amendment.  Unwrapping a phase measurement against the previous
        reading alone assumes the phase barely moved, and in lead mode that is
        exactly false the moment the machine concedes: a 2% concession at
        140 bpm opens 34 ms of phase per bar against a half-range of 53 ms, so
        the true branch and the wrapped one are nearly EQUIDISTANT and the
        estimator's own ~20 ms of per-bar noise decides which one is picked.
        MEASURED on the trap140 rush take: bar 8's reading of -22.9 ms (truth
        about -2 ms) unwrapped to **+82 ms**, one milliseconds-wide coin flip,
        and the machine read a divergence as a fresh push and ran to +5.98%,
        five beats out.

        The prediction removes the coin flip, and it costs nothing new: the
        machine already knows its OWN bar length exactly, and ``BeatState.bpm``
        is the tracker's estimate of the human's — which amendment A4 already
        names as the right authority on a tempo *difference*.  It is used only
        to say where to look, never as the response: a wrong prediction inside
        half a subdivision changes nothing, and the guards below keep a wildly
        wrong one (unlocked, or an octave relative) out entirely.
        """
        if not beat.locked:
            return 0.0
        bpm = float(beat.bpm)
        if not math.isfinite(bpm) or bpm <= 0.0:
            return 0.0
        human_bar_s = 4.0 * 60.0 / bpm
        if not (0.75 <= human_bar_s / bar_s <= 1.25):
            return 0.0                   # an octave relative, not a tempo gap
        return (bar_s - human_bar_s) * 1000.0

    def _clear_offsets(self) -> None:
        """Forget the evidence window AND the phase it was unwrapped against.

        Unwrapping is a statement about continuity, so it may only ever be
        applied across bars that were admitted back to back.  A cleared window
        with a stale ``_phase_prev`` would unwrap the first reading of a NEW
        run against a phase from before the gap, which is the same error as
        averaging across a silence.
        """
        self._offsets.clear()
        self._phase_prev = None

    def _negotiate_tempo(self, beat: BeatState, conf: float,
                         human: "BarFeatures | None", machine_grid_t_s: float,
                         human_offset_ms: float | None,
                         human_offset_unit_s: float | None,
                         events: list[str], reasons: list[str]) -> None:
        p = self.params
        interval_s = 60.0 / effective_bpm(self.composed_bpm, self._tempo_pct)
        bar_s = 4.0 * interval_s
        measured = (human_offset_ms is not None
                    and math.isfinite(float(human_offset_ms)))
        # See ``update``'s docstring for why ``beat.locked`` gates only the
        # derived estimator (v2.1).
        usable = (conf >= p.concede_min_conf
                  and (measured or beat.locked)
                  and human is not None and not bool(human.silent))
        off: float | None = None
        if usable:
            if measured:
                off = float(human_offset_ms)
            else:
                off = grid_offset_ms(beat, machine_grid_t_s, interval_s=interval_s)

        unit_s = None
        if (human_offset_unit_s is not None and measured
                and math.isfinite(float(human_offset_unit_s))
                and float(human_offset_unit_s) > 0.0):
            unit_s = float(human_offset_unit_s)

        # ---- the plausibility bound, and what it is a bound ON (v2.1) ------
        #
        # v2.0 measured that a bare offset is not a 25 ms instrument: on
        # plume's own 140 bpm trap render the BeatState-derived difference
        # reaches +/-210 ms — half a beat — because a swung groove biases the
        # prediction and a fraction of a percent of tempo mismatch walks the
        # phase across the beat.  Unfiltered it produced a single -8% step into
        # the cap in the WRONG DIRECTION.  Hence a bound.
        #
        # v2.1: the bound has to be stated in the INSTRUMENT's units.
        # ``onset_phase_offset_ms`` is a circular mean over ONE SUBDIVISION, so
        # it is only unambiguous within +/- half a sixteenth (+/-62.5 ms at 120
        # bpm) — while ``concede_plausible_frac`` of a BEAT is +/-100 ms, so
        # every wrapped sample sailed straight through the gate that exists to
        # catch exactly this.  MEASURED at 120 bpm: a human 70 ms ahead reads
        # -55 ms, 90 ms ahead reads -35 ms, 150 ms ahead reads +25 ms — sign
        # reversed, gate passed — and end to end that made the machine's
        # response DIRECTION a coin flip once the human's tempo differed by
        # more than ~5% (4 of 12 simulate_lead runs went the wrong way; the
        # same 132 bpm human gave 116.5 bpm at one seed and 127.0 at another).
        bound_ms = p.concede_plausible_frac * interval_s * 1000.0
        if unit_s is not None:
            bound_ms = min(bound_ms,
                           p.concede_plausible_unit_frac * unit_s * 1000.0)

        if off is not None and math.isfinite(off):
            if unit_s is not None and self._phase_prev is not None:
                # UNWRAP against a PREDICTION, then bound the step.  The
                # phase two clocks are apart is a real, continuous quantity and
                # it can legitimately run to several subdivisions once they are
                # at different tempi — bounding its magnitude would blind the
                # machine at exactly the moment it is coming apart, which is
                # how the divergence below became unrecoverable.  What must
                # stay bounded is the STEP: a phase that moved further than the
                # instrument can follow in one bar has not been followed, and
                # continuing to unwrap from it would be inventing a history.
                u_ms = unit_s * 1000.0
                predicted = self._phase_prev + self._expected_drift_ms(
                    beat, 4.0 * interval_s)
                cand = off + u_ms * round((predicted - off) / u_ms)
                # The step bound is the instrument's OWN half-range, not
                # ``concede_plausible_unit_frac``.  MEASURED, and this is the
                # subtle one: the drift the machine has to correct is itself
                # tens of ms per bar (a 2% concession at 140 bpm opens 34 ms of
                # phase every bar), so a step bound tighter than the half-range
                # rejects exactly the readings that describe the divergence —
                # trap140's drag take sat at -2.0% for TWELVE bars with the
                # correction starved, ending 3.2 beats out.  Nearest-branch
                # unwrapping already bounds the step by construction; what a
                # tighter test would buy is nothing, and what it costs is the
                # phase-return term.
                off = cand if abs(cand - predicted) <= 0.5 * u_ms else None
            elif abs(off) > bound_ms:
                # No continuity to lean on, so the raw reading has to stand on
                # its own: past the bound it is the next subdivision, not a
                # human pushing.
                off = None
        if off is None or not math.isfinite(off):
            # Unusable evidence CLEARS the window rather than skipping it: a
            # mean stitched across a silence, an unlock or a blown phase
            # estimate is a mean of two different performances.
            self._clear_offsets()
            return

        self._phase_prev = off
        self._bound_ms = bound_ms
        self._interval_s = interval_s
        self._push_bound_ms = p.concede_plausible_frac * interval_s * 1000.0
        if unit_s is not None:
            self._push_bound_ms = min(
                self._push_bound_ms,
                p.concede_plausible_unit_frac * unit_s * 1000.0)
        self._offsets.append(off)
        # THE MEASUREMENT IS THE PRODUCT, not just the concede rule's input
        # (v2.2).  Recorded here — after every plausibility gate, so what is
        # published is a reading the negotiator itself believes — and BEFORE
        # any policy or authority gate, so a machine that is not going to act
        # on it still reports it.  See the module docstring: a silent hold is
        # indistinguishable from a broken measurement, and this number is the
        # v2.5 practice-feedback signal.
        self._last_offset_ms = off
        # One more sample than the push rule needs: the DRIFT rule below is a
        # law about differences, so it needs ``concede_bars`` of them.
        keep = max(p.concede_bars, _DRIFT_DIFFS + 1)
        if len(self._offsets) > keep:
            del self._offsets[:-keep]

        # ---- whose clock is it? (v2.2 amendment; see TEMPO AUTHORITY) ------
        policy = self.tempo_policy
        if policy == "hold":
            # The machine is the reference.  Say so, every bar, with the
            # number — that is the difference between holding and being deaf,
            # and it is the only way a listener can tell the two apart from
            # the log.  Nothing below this line runs, so ``_tempo_target_pct``
            # never moves and a fixed recording produces EXACTLY 0.0% drift
            # rather than the bounded hunt v2.1 documented.
            events.append("tempo_hold")
            reasons.append(f"holding: human {_ms_str(off)}ms (policy=hold)")
            return
        if policy == "chase":
            # The human owns the clock: the caller's phase correction follows
            # them, and this negotiator holds no tempo of its own.  Silent on
            # purpose — under ``chase`` the machine is not making a decision
            # every bar, and this path is byte-identical to what ``follow``
            # already did through ``CONCEDE_MIN_CLOCK`` before v2.2.
            return

        if self._clock < CONCEDE_MIN_CLOCK:
            return
        if self._ramping():
            # One concession at a time: the previous ramp has not landed, so
            # the offsets in the window were measured against a grid that is
            # still moving.
            return

        if self._negotiate_drift(bar_s, events, reasons):
            return
        self._negotiate_push(interval_s, events, reasons)

    def _negotiate_drift(self, bar_s: float,
                         events: list[str], reasons: list[str]) -> bool:
        """The PHASE-RETURN term (v2.1 amendment, SPEC-V2 section 1 behavior 1).

        THE DEFECT THIS FIXES.  The spec's rule is a phase->period law with no
        phase-return term, and such a law diverges by construction: it changes
        the machine's *tempo* in response to a *phase* error, so once the tempo
        has moved the phase keeps opening and nothing ever closes it again.
        MEASURED IN THE RENDERED AUDIO (machine-bus low-band phase-locked bar
        tracker, agreeing with the decision log to <=50 ms): on the 32-bar
        ``rushing`` joiner the machine ended +3.26% with its downbeat 1.24 s —
        0.62 bar, two and a half BEATS — ahead of the human's; ``dragging``
        ended -3.42% and 3.7 beats behind.  The loop never recovered, because
        past about half a subdivision of drift per bar the offset reading
        alternates sign and the spread test kills every window.

        The missing term is the one every phase-locked loop has: when the gap
        is OPENING at a steady rate, the two clocks are at different tempi, and
        the size of the tempo difference is exactly that rate.  ``_DRIFT_RATE``
        of it is absorbed — same ramp bound, same cap as a push — which is a
        first-order loop that settles instead of an integrator that walks.

        Deliberately narrow, because amendment A4 measured what admitting bare
        ramps costs.  It fires only when

        * ``_DRIFT_DIFFS`` consecutive differences share a sign (a rate, not
          noise) and agree within ``concede_spread_ms`` (a *steady* rate),
        * their mean exceeds ``concede_drift_ms`` per bar, and
        * the gap is OPENING — ``sign(drift) == sign(phase)`` — so the machine
          never fights its own correction while a push it already conceded to
          is still closing, which is what makes SPEC-V2 section 0's "both
          parties bend a little" survive this rule.

        MEASURED, end to end, with everything above in place (the numbers the
        defect report gave, re-run):

        ==========================  ==========  ==========
        take                        before      after
        ==========================  ==========  ==========
        trap140 rush/drag worst     3.7 beats   1.6 beats
        synthetic tape (48 bars)    1330 ms     130 ms
        mutual entrainment, peak    40 ms       30 ms
        ==========================  ==========  ==========

        Returns True when it acted.
        """
        p = self.params
        if len(self._offsets) < _DRIFT_DIFFS + 1:
            return False
        diffs = [b - a for a, b in zip(self._offsets, self._offsets[1:])]
        if not (all(d > 0.0 for d in diffs) or all(d < 0.0 for d in diffs)):
            return False                        # not a ramp, just noise
        # The rate is measured END TO END rather than as a mean of the
        # differences, and NOT spread-tested the way ``_consistent_window``
        # tests a push.  A push must be steady to be a push; a drift only has
        # to be there.  MEASURED on the ``dragging`` joiner: the machine really
        # was coming apart at 69 ms/bar and the window read
        # +20/+110/+191/+211/+301 — unmistakably a ramp — but the per-bar
        # DIFFERENCES were +90/+81/+20/+90, a spread of 70 ms against a 25 ms
        # tolerance, because the estimator's own per-bar noise is +/-30 ms.
        # The spread test threw away every window and the machine sat at -3.47%
        # for eighteen bars, ending 3.8 beats out.  Over the whole window that
        # same noise divides by the number of bars instead of accumulating.
        rate = (self._offsets[-1] - self._offsets[0]) / len(diffs)
        if abs(rate) <= p.concede_drift_ms:
            return False
        phase = self._offsets[-1]
        if phase * rate <= 0.0:
            return False                        # the gap is CLOSING, not opening

        old_bpm = effective_bpm(self.composed_bpm, self._tempo_pct)
        # The human gaining ``rate`` ms of lead per bar IS a tempo difference of
        # ``rate/bar_s``; the sign follows the same convention as everywhere
        # else (+ = the human is ahead, so the machine must speed up).
        delta_pct = _DRIFT_RATE * (rate / 1000.0) / bar_s * 100.0
        delta_pct = _clamp(delta_pct, -MAX_STEP_PCT_PER_BAR,
                           MAX_STEP_PCT_PER_BAR)
        new_target = _clamp(self._tempo_target_pct + delta_pct,
                            -p.concede_max_pct, p.concede_max_pct)
        self._offsets.clear()
        if abs(new_target - self._tempo_target_pct) < 1e-9:
            events.append("concede_capped")
            reasons.append(
                f"tempo capped: drifting {_ms_str(rate)}ms/bar but already "
                f"{_pct_str(self._tempo_pct)}% from {self.composed_bpm:.1f} bpm "
                f"(max {p.concede_max_pct:.1f}%)")
            return True
        self._tempo_target_pct = new_target
        new_bpm = effective_bpm(self.composed_bpm, new_target)
        events.append("concede")
        events.append("tempo_match")
        reasons.append(
            f"matching tempo: we are coming apart {_ms_str(rate)}ms per bar "
            f"({_ms_str(phase)}ms out) -> {old_bpm:.1f} -> {new_bpm:.1f} bpm")
        return True

    def _negotiate_push(self, interval_s: float,
                        events: list[str], reasons: list[str]) -> None:
        """The spec's own rule: meet a human who is sitting in front of the beat."""
        p = self.params
        window = _consistent_window(self._offsets, p.concede_bars,
                                    p.concede_spread_ms)
        if window is None:
            return                       # jitter around the grid, not a push
        mean = sum(window) / len(window)
        if abs(mean) <= p.concede_ms:
            return
        rate = 0.0
        if len(self._offsets) >= 2:
            diffs = [b - a for a, b in zip(self._offsets, self._offsets[1:])]
            rate = sum(diffs) / len(diffs)
        moving = abs(rate) > p.concede_drift_ms
        if moving and abs(mean) > self._push_bound_ms:
            # v2.1: where this law ends and ``_negotiate_drift`` begins.
            #
            # An expressive push is tens of milliseconds (RESEARCH.md); a fifth
            # of a beat is already the outside of it.  A phase further out than
            # that WHILE STILL MOVING is two clocks coming apart, and answering
            # it with the push law is how the loop overshot — MEASURED on
            # trap140's rush take, six bars after conceding +1.86% to a human
            # 30 ms ahead the drifted phase read -42 ms, this law read that as
            # a lay-back, and the machine went DOWN to -1.13% against a human
            # who never stopped rushing.
            #
            # Both halves of the condition are load-bearing and each was
            # measured the hard way.  Suppressing on MOVEMENT alone is wrong
            # because a human who starts pushing IS an opening gap — that is
            # the shape of the thing this law exists for — and doing it rang
            # the mutual-entrainment loop at 87 ms where it settles at 30.
            # Suppressing on MAGNITUDE alone is wrong because a large offset
            # that is NOT moving is a standing offset that the drift law cannot
            # see (there is no rate to measure), and leaving it to nobody
            # deadlocked that same loop at a permanent -85 ms.
            return

        n = len(window)
        old_bpm = effective_bpm(self.composed_bpm, self._tempo_pct)
        # SPEC-V2's worked example fixes the units: the deviation is a fraction
        # of the machine's BEAT period, and `concede_rate` of that fraction is
        # applied as a tempo scale.  (+31ms at 120bpm -> 0.031/0.5 = 6.2%,
        # x0.25 = 1.55% -> 121.86 -> "121.9 bpm".)
        delta_pct = p.concede_rate * (mean / 1000.0) / interval_s * 100.0
        # v2.0 AMENDMENT (SPEC-V2 section 1, behavior 1): a single concession is
        # clamped to one bar's ramp.  The spec says the concession is "a ramp
        # applied over the FOLLOWING BAR" and section 7 bounds a bar at 2%; a
        # target that needs three bars of ramping contradicts both, so the
        # clamp is the spec's own arithmetic rather than an addition to it.
        # MEASURED: without it, one noisy pair of derived offsets (+68, +66ms,
        # from a neighbourhood reading +208, -86, +197) set a 3.9% target off
        # two bars of evidence and walked a matched 140bpm take to +5.8%.
        # Nothing is lost: the human who really is that far out is still met,
        # one bar at a time, and the log says so every time.
        delta_pct = _clamp(delta_pct, -MAX_STEP_PCT_PER_BAR,
                           MAX_STEP_PCT_PER_BAR)
        want = self._tempo_target_pct + delta_pct
        new_target = _clamp(want, -p.concede_max_pct, p.concede_max_pct)
        self._offsets.clear()

        if abs(new_target - self._tempo_target_pct) < 1e-9:
            events.append("concede_capped")
            reasons.append(
                f"tempo capped: human {_ms_str(mean)}ms for {n} bars but "
                f"already {_pct_str(self._tempo_pct)}% from "
                f"{self.composed_bpm:.1f} bpm "
                f"(max {p.concede_max_pct:.1f}%)")
            return

        self._tempo_target_pct = new_target
        new_bpm = effective_bpm(self.composed_bpm, new_target)
        events.append("concede")
        reasons.append(
            f"conceding: human {_ms_str(mean)}ms for {n} bars -> "
            f"{old_bpm:.1f} -> {new_bpm:.1f} bpm")

    # --- harmony ---------------------------------------------------------

    def _negotiate_key(self, harmony: HarmonyState,
                       events: list[str], reasons: list[str]) -> None:
        p = self.params
        if not p.conform_enabled or self.composed_key is None:
            return
        if harmony.key is None or float(harmony.key_conf) < p.conform_key_conf:
            self._key_run_dist = None
            self._key_run_bars = 0
            self._hold_logged = False
            return
        try:
            human_key = parse_key(harmony.key)
        except ValueError:               # pragma: no cover - defensive
            return
        mine = parse_key(self.composed_key)
        my_pc = (mine.tonic + self._transpose) % 12
        # Collection roots, not tonics — see :func:`collection_root`.
        d = _signed_pc_distance(
            (collection_root(mine) + self._transpose) % 12,
            collection_root(human_key))

        if d == 0:
            self._key_run_dist = None
            self._key_run_bars = 0
            self._hold_logged = False
            return
        if d != self._key_run_dist:
            self._key_run_dist = d
            self._key_run_bars = 1
            self._hold_logged = False
        else:
            self._key_run_bars += 1
        if self._key_run_bars < p.conform_bars:
            return

        my_name = _key_name(my_pc, mine.mode)
        their_name = _key_name(human_key.tonic, human_key.mode)
        if abs(d) <= p.conform_semitones:
            self._transpose += d
            self._key_run_dist = None
            self._key_run_bars = 0
            self._hold_logged = False
            events.append("conform")
            reasons.append(
                f"conforming: human in {their_name}, {d:+d} st from {my_name} "
                f"for {p.conform_bars} bars -> transposing {d:+d}")
            return
        if not self._hold_logged:
            self._hold_logged = True
            events.append("conform_hold")
            reasons.append(
                f"holding key: human in {their_name}, {abs(d)} st from "
                f"{my_name} for {self._key_run_bars} bars "
                f"(> {p.conform_semitones}) — they are playing something "
                f"else, conforming would be guessing")

    # --- reporting -------------------------------------------------------

    def _steady_reason(self) -> str:
        bits = [f"{self._role}: clock {self._clock:.2f} form {self._form:.2f}"]
        if abs(self._tempo_pct) >= 0.05 or self._ramping():
            bpm = effective_bpm(self.composed_bpm, self._tempo_pct)
            bits.append(f"tempo {_pct_str(self._tempo_pct)}% ({bpm:.1f} bpm)")
        if self._transpose:
            bits.append(f"transposed {self._transpose:+d} st")
        return ", ".join(bits)
