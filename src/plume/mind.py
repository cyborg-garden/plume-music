"""The negotiation state machine — the sideman ethic as code (SPEC-V1 section 4).

``Mind.plan_bar`` is called once per predicted bar boundary (or once per second
while the beat tracker is unlocked, with ``human=None``) and returns a
:class:`BarPlan` for the *upcoming* bar.  It encodes the negotiation behaviors
from the jazz research (RESEARCH.md interaction spec / the original research sweep
jazz-negotiation recommendations):

Enter like a sideman (enter-late gate)
    Silent until ``min_listen_bars`` non-silent human bars have been heard AND
    the beat tracker is locked AND beat confidence >= ``entry_conf`` AND the
    upcoming bar is a multiple of ``entry_boundary_bars``.  Entering mid-phrase
    or on a wrong tempo hypothesis is the single most jarring machine behavior.

Enter under, ramp up
    First entry at ``enter_under_db`` (relative to the human's running level;
    sim/realtime calibrate the buses), sparse texture (bass + pad, no fills),
    gain ramping linearly to 0 over ``ramp_bars`` bars.  State is "entering"
    while the ramp runs, then "comping".

Inverse density coupling (hysteresis + one-bar delay)
    Human onset density above ``density_high`` thins the machine's pattern
    density; recovery requires density below ``density_high -
    density_hysteresis`` AND is delayed exactly one extra bar so it reads as a
    musical answer, not a mirror.  This modulates ``BarPlan.density`` (the
    pattern-density hint), not the role set — texture stays put.

Silence 3-stage FSM (with re-arm)
    Consecutive silent human bars (``silent`` flag or bar RMS below
    ``silence_rms_db``): >= ``silence_thin_bars`` -> "thinned" (groove
    skeleton); >= ``silence_fade_bars`` -> "fading" (gain steps down 6 dB per
    bar); >= ``silence_tacet_bars`` -> "tacet" and the entry logic re-arms.
    The human resuming resets the counter: thinned/fading recover to comping
    on the very next plan; after tacet the machine goes back through the
    entry gate (boundary + lock + confidence — ``min_listen_bars`` is a
    cumulative session total and stays satisfied, so re-entry happens at the
    next boundary, under-gain and ramp again).

Beat-loss repair (independent of the silence FSM)
    Beat confidence below ``conf_floor`` — or the tracker unlocking — while
    the machine is sounding is an emergency: "fading" immediately (no boundary
    wait), then "tacet" and re-arm.  Silence counters are frozen during
    beat-loss (a rubato human is not a silent human).  Fighting a lost beat is
    the most audible failure mode; laying out and re-entering at the next
    confident boundary is the repair.

Register avoidance
    ``register_ceiling_midi`` = human ``register_low_midi`` (per-bar features,
    falling back to the harmony tracker's rolling estimate) minus
    ``register_margin_st``.  The last known ceiling is held while register
    information is momentarily absent (e.g. a silent bar); ``None`` only when
    nothing was ever observed or the machine is not sounding.

Dynamic-arc following
    ``gain_db`` tracks an EMA (``energy_alpha``) of the human's bar RMS as a
    delta from the level at entry, clamped to [-12, +6] dB.  The arc moves the
    gain, not the layer set — texture changes happen only through the state
    machine, and machine-initiated texture changes (entry) only at
    ``entry_boundary_bars`` boundaries.  Responses to the human's withdrawal
    (silence stages) and emergencies (beat-loss, tacet) are exempt from the
    boundary rule by design: they are reactions, not initiations.

Commitment (v1.4 AMENDMENT — the flapping fix; SPEC-V1 section 4)
    THE DEFECT, measured in the first session with a human and an acoustic
    guitar: the entry and exit gates were re-evaluated every bar against a
    confidence that sat ON both thresholds (the whole take oscillated
    0.41-0.62, straddling ``entry_conf`` 0.45 and the tracker's ``lock_conf``
    0.55), so the machine armed, played a bar or two, disarmed on the next
    dip, and re-armed later.  The player's words: "it only plays like two
    random measures here and there, and then stops suddenly, and then starts
    up randomly later ... it cant chicken out."

    A musician decides whether to be in the band once per section, not once
    per three seconds.  So entering is now a COMMITMENT:

    * ``commit_bars`` — for this many bars from the entry bar (inclusive) the
      machine does not leave for ORDINARY reasons.  A confidence dip is
      ordinary.  A wobbling tracker is ordinary.
    * ``exit_conf`` — while committed, the emergency confidence floor is
      ``exit_conf`` (0.25), not ``conf_floor`` (0.3).  The band of confidences
      that KEEPS you in is deliberately far wider than the band that gets you
      in (0.45).  Deep hysteresis, in the same spirit as the tracker's own
      lock/unlock asymmetry.
    * ``_UNLOCK_GRACE_BARS`` — while committed, an unlocked tracker is a
      wobble for this many consecutive plan calls and the machine HOLDS its
      last sounding bar; only a SUSTAINED unlock is a true unlock and triggers
      beat-loss repair.  Outside the commitment window an unlock is an
      emergency on the first bar, exactly as before.
    * ``recommit_bars`` — if the arc ended EARLY (the commitment was still
      running when the machine went tacet), it may not re-enter until this
      many bars after the exit gesture began.  Bailing out and jumping
      straight back in is the other half of "starts up randomly later".

    Two things this must NOT become, both proven in ``tests/test_commitment.py``:
    it must not become "never stops" — the silence ladder
    (thinned/fading/tacet) is fully exempt from commitment, because a human
    who genuinely stops is not an ordinary dip; and a TRUE unlock still
    triggers beat-loss repair inside ``commit_bars``.

Reciprocal cueing — announce before you commit (v1.4)
    "if it does really commit to something it has to also give the other
    musician a chance to adapt to that as well."  RESEARCH.md's anticipation
    protocol, and SPEC-V2 section 2's "Announce it" rule — which existed only
    for v2 FORM changes (``FormPlan.announce``) and not for the thing the
    player actually noticed, the machine appearing and vanishing.

    A change the human needs to know about is signalled in the bar BEFORE it
    lands: ``BarPlan.announce`` /
    ``announce_kind`` / ``announce_bar`` carry the intent, and the renderer
    voices it (a pickup, a fill, a riser, a soft swell).  This module states
    the intent and logs it; it renders nothing.

    Announced: ``"entry"`` (the machine PLEDGES the entry a bar ahead and then
    honours the pledge through a confidence dip — that is what "it cant
    chicken out" means), ``"exit"`` (the fade and the tacet of the silence
    ladder), ``"texture"`` (thinning to the skeleton, and the density
    coupling's one-bar-delayed recovery).

    NOT announced, and honestly so: beat-loss repair (an emergency has no
    bar of notice by definition) and the onset of density thinning (a
    reaction to the human's own busyness — they already know).  An announced
    exit that the human answers by playing again is CANCELLED, not honoured;
    that is the cue working, not a broken promise.

Pure and deterministic: no rng, no I/O, no wall-clock; all memory lives inside
``Mind``.  ``reason`` strings are part of the contract — they make the
decisions NDJSON self-explanatory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .listen.beat import BeatState
from .listen.harmony import HarmonyState

if TYPE_CHECKING:                       # pragma: no cover - typing only
    from .score import Session

STATES = ("waiting", "listening", "entering", "comping", "thinned", "fading", "tacet")

# States in which the machine is (still) sounding.
_ACTIVE_STATES = frozenset({"entering", "comping", "thinned", "fading"})

# Role levels per state (role -> level 0..1; absent = tacet for that role).
# Entering is the sideman's first touch: roots and a pad, no fills, no kit.
# Thinned is the groove skeleton the research prescribes (kick+hat).
_ROLES: dict[str, dict[str, float]] = {
    "entering": {"bass": 0.8, "pad": 0.6},
    "comping": {"kick": 1.0, "snare": 0.8, "hat": 0.7, "bass": 1.0, "pad": 0.7, "keys": 0.5},
    "thinned": {"kick": 0.6, "hat": 0.4},
    "fading": {"pad": 0.5, "bass": 0.3},
}

# Pattern-density hints per state; comping drops to the busy value while the
# inverse-density coupling is engaged.
_DENSITY: dict[str, float] = {
    "entering": 0.3,
    "comping": 0.8,
    "thinned": 0.25,
    "fading": 0.15,
}
_DENSITY_BUSY = 0.3

_FADE_STEP_DB = 6.0     # per-bar gain drop while fading
_GAIN_FLOOR_DB = -60.0
# v1.2: non-silent human bars that must have been measured ON the locked grid
# before entry can arm.  Two, because the FIRST such bar is the one during
# which the lock happened — it is only partly inside the grid the machine
# would play on.  Bars credited from before the lock (see
# Mind.credit_listened_bars) prove the human has been playing and satisfy
# min_listen_bars, but they cannot vouch for the grid.
_MIN_BARS_ON_GRID = 2
_ARC_MIN_DB = -12.0     # dynamic-arc clamp around the entry reference
_ARC_MAX_DB = 6.0
# v1.4: consecutive plan calls for which an UNLOCKED tracker is treated as a
# wobble rather than a true unlock, WHILE COMMITTED.  Not a MindParams field
# on purpose — the CLI surface is the negotiation thresholds a player would
# actually dial, and this is the width of a glitch, not a musical preference.
# FOUR, and the number is forced by the tracker, not chosen: a round trip
# through an unlock costs UNLOCK_BEATS/4 = 1 bar to drop out plus LOCK_BEATS/4
# = 2 bars of confident beats to re-lock, so any grace shorter than 3 bars can
# never observe a recovery — it would guarantee the bail-out it exists to
# prevent.  Four is that minimum plus one bar of margin, and it is one phrase,
# which is also how long a player will busk through a patch they cannot hear.
_UNLOCK_GRACE_BARS = 4
# The gesture the renderer should use to voice each announcement.  This module
# does not render; it names the intent (P1-E / RESEARCH.md reciprocal cueing).
_ANNOUNCE_GESTURE: dict[str, str] = {
    "entry": "pickup", "exit": "fill", "texture": "swell",
}
#: element of this tuple, or None — ``BarPlan.announce_kind``.
ANNOUNCE_KINDS = ("entry", "exit", "texture")


@dataclass
class MindParams:
    """Every negotiation threshold; defaults are the research values.

    All fields are settable via the CLI (``--mind k=v``); ``PARAM_HELP`` below
    carries the human-readable meaning of each field for ``plume mind``.
    """

    min_listen_bars: int = 4
    # entry_conf 0.55 -> 0.45 — v1.2 AMENDMENT (SPEC-V1 section 4).  0.55 was
    # EXACTLY BeatTracker's lock_conf, and this gate samples confidence
    # instantaneously at the bar boundary with no hysteresis of its own, while
    # the tracker's lock has plenty (8 beats >= 0.55 to lock, 4 beats < 0.3 to
    # unlock) — so the tracker stays locked while confidence oscillates around
    # 0.5 and the mind's re-check is a coin flip, ANDed with a 1-in-4 boundary
    # test.  Measured on a real render at a perfectly ordinary live-mic level
    # (RMS -27 dBFS): boundary confidences 0.56, 0.55, 0.50, 0.51, ..., 0.55 —
    # the two samples that cleared 0.55 landed on bars 0 and 9, neither a
    # multiple of 4, so ``entered_bar`` was None over a 49.5s take with the
    # reason "locked but conf 0.55 < entry threshold 0.55".  A test asserts
    # this stays strictly below the tracker's lock threshold.
    entry_conf: float = 0.45          # beat confidence required to arm entry
    entry_boundary_bars: int = 4      # enter only at multiples of this
    enter_under_db: float = -8.0      # first-entry gain relative to human's bar RMS
    ramp_bars: int = 3                # bars to ramp from enter_under_db to 0
    density_high: float = 2.5         # human onsets/beat that triggers thinning
    density_hysteresis: float = 0.8   # come back when density < high - hysteresis (one-bar delay)
    silence_thin_bars: int = 2        # human silent this long -> thin to skeleton
    silence_fade_bars: int = 6        # -> begin fading
    silence_tacet_bars: int = 10      # -> fully out, re-arm entry
    silence_rms_db: float = -45.0     # "silent" bar threshold
    conf_floor: float = 0.3           # beat conf below this -> fading/tacet (beat-loss repair)
    register_margin_st: float = 5.0   # keep voicings this many semitones away from human register
    energy_alpha: float = 0.5         # EMA for following the human's dynamic arc
    # --- commitment (v1.4; see the module docstring) --------------------
    commit_bars: int = 8              # once in, stay in for at least this many bars
    exit_conf: float = 0.25           # emergency conf floor WHILE committed (deep hysteresis)
    recommit_bars: int = 4            # after an early exit, sit out at least this long
    # --- harmonic commitment (v2.6; see plume.governor) -----------------
    harmony_commit_bars: int = 4      # stable-key bars before committing to key + progression (0 = off)
    harmony_release_bars: int = 8     # sustained different-key bars before modulating


#: field name -> one-line meaning, for `plume mind` (kept in sync by tests).
PARAM_HELP: dict[str, str] = {
    "min_listen_bars": "non-silent human bars to hear before entry is possible",
    "entry_conf": "beat confidence required to arm entry",
    "entry_boundary_bars": "enter (and re-enter) only at bar multiples of this",
    "enter_under_db": "first-entry gain in dB relative to the human's level",
    "ramp_bars": "bars to ramp the entry gain from enter_under_db up to 0",
    "density_high": "human onsets/beat above this thins the machine's density",
    "density_hysteresis": "recover when density < density_high - this (plus a one-bar delay)",
    "silence_thin_bars": "consecutive silent human bars before thinning to a skeleton",
    "silence_fade_bars": "consecutive silent human bars before fading out",
    "silence_tacet_bars": "consecutive silent human bars before tacet + entry re-arm",
    "silence_rms_db": "bar RMS (dBFS) below which a human bar counts as silent",
    "conf_floor": "beat confidence below this triggers beat-loss repair (fade/tacet)",
    "register_margin_st": "semitones to keep voicings below the human's low register",
    "energy_alpha": "EMA coefficient for following the human's dynamic arc",
    "commit_bars": "bars the machine stays in for once it enters (no ordinary exit)",
    "exit_conf": "beat confidence below this ends a COMMITTED arc (deep hysteresis)",
    "recommit_bars": "bars to sit out after an exit that broke the commitment",
    "harmony_commit_bars": "stable-key bars before committing to that key + the style progression (0 = follow tracker chords)",
    "harmony_release_bars": "consecutive different-key bars before modulating a committed key",
}


@dataclass(frozen=True)
class BarFeatures:
    """Summary of one completed human bar (built by sim/realtime)."""

    bar: int
    rms_db: float
    onset_density: float               # onsets per beat
    silent: bool
    register_low_midi: float | None
    register_high_midi: float | None


@dataclass(frozen=True)
class BarPlan:
    """The mind's decision for the UPCOMING bar.

    The first seven fields are the frozen v1 contract.  The three
    ``announce*`` fields are v1.4 (P1-E / RESEARCH.md reciprocal cueing) and
    are defaulted, so every existing keyword construction still works.  They
    describe a change landing on the bar AFTER ``bar`` — the machine telling
    the human what it is about to do, so they can adapt to it.  The renderer
    voices it; this object only states the intent.
    """

    bar: int
    state: str                          # element of STATES
    roles: dict[str, float]             # role -> level 0..1 (0/absent = tacet)
    gain_db: float                      # machine bus gain for the bar
    density: float                      # 0..1 pattern-density hint
    register_ceiling_midi: float | None # voice below this; None = free
    reason: str                         # human-readable; part of the contract
    announce: str | None = None         # what is coming, e.g. "coming in at bar 12"
    announce_kind: str | None = None    # element of ANNOUNCE_KINDS
    announce_bar: int | None = None     # the bar it refers to (always bar + 1)


# =====================================================================
# v2 — form navigation and cues (SPEC-V2 sections 2 and 3).
#
# Everything below this line is ADDITIVE.  ``plan_bar`` and ``MindParams``
# are untouched: the v1 behavior matrix must keep passing byte-identically,
# so the form cursor gets its own parameter object (``FormParams``), its own
# state block on ``Mind`` (every attribute prefixed ``_form_``) and its own
# entry point (``Mind.plan_form``).
# =====================================================================

FORM_ACTIONS = ("advance", "hold", "extend", "jump", "end")

#: Boundaries a cue may name (``Cue.at``).
CUE_BOUNDARIES = ("now", "next-bar", "next-phrase", "next-section")

#: Cue kinds ``parse_cue`` can return.  "unknown" is the v2 amendment: the
#: spec lists six kinds, but ``parse_cue`` must not raise on junk and
#: mapping junk onto a real kind is worse than admitting it.
CUE_KINDS = ("energy", "section", "texture", "tempo", "end", "style", "unknown")


@dataclass
class FormParams:
    """Thresholds for the arrangement cursor (SPEC-V2 section 2).

    Separate from :class:`MindParams` on purpose — ``MindParams``' field set
    and defaults are frozen by the v1 tests and by ``plume mind``.
    """

    # --- turnaround waiting -------------------------------------------
    max_extend_bars: int = 16       # total bars a section may be stretched by
    extend_unit_bars: int = 4       # one "phrase unit" of extension
    phrase_bars: int = 4            # bars per phrase (cue boundaries too)
    phrase_rise_db: float = 3.0     # bar-RMS rise that reads as "still building"
    phrase_rise_density: float = 0.15   # ...or this FRACTIONAL rise in onsets/beat
    # --- energy chasing -----------------------------------------------
    energy_bars: int = 2            # consecutive bars of evidence before a jump
    energy_up_db: float = 3.0       # human above their own baseline by this -> up
    energy_down_db: float = 3.0     # ...below -> down
    energy_margin: float = 0.05     # a target section must differ by this much
    energy_fast_alpha: float = 0.5  # EMA of the human's recent level
    energy_slow_alpha: float = 0.1  # EMA standing in for their session baseline
    # --- ending -------------------------------------------------------
    ritard_bars: int = 2            # consecutive slowing bars that read as a ritard
    ritard_pct: float = 4.0         # ...totalling at least this much tempo drop
    ritard_step_max_pct: float = 10.0   # a bigger single-bar drop is the TRACKER
    # --- authority ----------------------------------------------------
    form_authority_min: float = 0.5  # below this the machine initiates no changes
    loop_form: bool = True          # past the last section, wrap (map, not script)


#: field name -> one-line meaning (kept in sync by tests, like PARAM_HELP).
FORM_PARAM_HELP: dict[str, str] = {
    "max_extend_bars": "total bars a section may be extended by while waiting",
    "extend_unit_bars": "bars added per turnaround extension (one phrase unit)",
    "phrase_bars": "bars per phrase — the boundary cues wait for",
    "phrase_rise_db": "human bar-RMS rise that reads as mid-phrase (still building)",
    "phrase_rise_density": "fractional onsets/beat rise that also reads as mid-phrase",
    "energy_bars": "consecutive bars of energy evidence before a section jump",
    "energy_up_db": "human above their own baseline by this -> a hotter section",
    "energy_down_db": "human below their own baseline by this -> drop to a calmer one",
    "energy_margin": "minimum section-energy difference for a jump target to count",
    "energy_fast_alpha": "EMA coefficient for the human's recent level",
    "energy_slow_alpha": "EMA coefficient for the human's session baseline",
    "ritard_bars": "consecutive slowing bars that count as a ritardando",
    "ritard_pct": "total tempo drop (%) across those bars",
    "ritard_step_max_pct": "a bigger single-bar drop is a tracker re-read, not a ritard",
    "form_authority_min": "form_authority below this: machine initiates no changes",
    "loop_form": "wrap to the first section past the end instead of ending",
}


@dataclass(frozen=True)
class FormPlan:
    """The cursor's decision for the UPCOMING bar (SPEC-V2 section 2).

    The first five fields are the frozen contract.  ``announce``,
    ``applied_cue`` and ``dropped_cue`` are v2 AMENDMENT 1 (see SPEC-V2
    section 2): they are defaulted, so every spec-shaped construction still
    works, and they exist because the spec's own rules are otherwise
    unobservable — "announce it" is a claim about the bar BEFORE a change,
    and a cue that applied (or expired unapplied) has to reach the decision
    log as data, not as prose inside ``reason``.
    """

    bar: int
    section_idx: int                # index into the flattened section list
    action: str                     # element of FORM_ACTIONS
    bars_left_in_section: int       # bars after the upcoming one; 0 = last bar
    reason: str                     # human-readable; part of the contract
    announce: bool = False          # play a fill/riser: a change may land next bar
    applied_cue: str | None = None  # text of the cue applied at this boundary
    dropped_cue: str | None = None  # text of a cue that expired unapplied


@dataclass(frozen=True)
class Cue:
    """A terse instruction from the third chair (SPEC-V2 section 3)."""

    text: str
    at: str = "next-phrase"         # element of CUE_BOUNDARIES
    ttl_bars: int = 16


@dataclass
class _QueuedCue:
    cue: Cue
    parsed: dict
    issued: int                     # internal bar counter at which it arrived


# ------------------------------------------------------------ cue parsing

_DIRECTIONAL = ("go to", "back to", "hit the", "into the", "take it to",
                "jump to", "to the", "onto the", "move to", "switch to",
                "straight to", "cut to")

#: section names that are also ordinary words ("drop it down", "break it up",
#: "build it") — these need a directional phrase to read as a section cue.
_AMBIGUOUS_SECTIONS = frozenset({"drop", "break", "build", "head", "hook"})

_SECTION_NAMES = ("intro", "verse", "pre chorus", "prechorus", "chorus",
                  "bridge", "breakdown", "drop", "break", "build", "hook",
                  "solo", "outro", "head", "refrain", "vamp", "coda",
                  "turnaround", "tag")

#: Generic song-section words -> the role in the arc they name, for when the
#: composed map has no section by that name (v2.1; see
#: ``Mind._form_section_target``).  Deliberately small: only words whose
#: meaning really is positional.  Anything else is a name, and a name that is
#: not in the map is reported as not being in the map.
_GENERIC_SECTION_ROLES: dict[str, str] = {
    "intro": "first", "head": "first",
    "outro": "last", "coda": "last", "tag": "last",
    "chorus": "hottest", "hook": "hottest", "drop": "hottest",
    "refrain": "hottest", "peak": "hottest", "outchorus": "last",
    "breakdown": "coldest", "break": "coldest",
    "verse": "middle_coldest",
}

_STYLE_NAMES = {
    "trap": "trap", "lofi": "lofi", "lo fi": "lofi", "boom bap": "hiphop",
    "hip hop": "hiphop", "hiphop": "hiphop",
    "synthwave": "synthwave", "house": "house", "techno": "techno",
    "jazz": "jazz", "jazzy": "jazz", "funk": "funk", "funky": "funk",
    "rock": "rock", "ambient": "ambient", "bossa": "bossa",
    "reggae": "reggae", "disco": "disco", "dnb": "dnb",
    "drum and bass": "dnb", "neo soul": "rnb", "rnb": "rnb", "r b": "rnb",
    "soul": "soul", "gospel": "gospel",
}

_END_PHRASES = ("take it out", "take us out", "take it home", "bring it home",
                "wrap it up", "wrap it", "end it", "the end", "ending",
                "last chorus", "final chorus", "last time", "one more and out",
                "let it end", "cut it off", "finish it", "stop it", "stop",
                "finish", "land it", "out chorus")

_TEMPO_PHRASES = (
    ("half", ("half time", "halftime", "half speed", "half tempo")),
    ("double", ("double time", "doubletime", "double speed", "double up",
                "double tempo")),
    ("ritard", ("ritard", "ritardando", "rit", "slow to a stop",
                "slow it to a stop")),
    ("faster", ("faster", "speed up", "speed it up", "pick it up",
                "push the tempo", "push it along", "up the tempo")),
    ("slower", ("slower", "slow down", "slow it down", "pull the tempo back",
                "ease off the tempo", "back off the tempo")),
)

_TEXTURE_PHRASES = (
    ("darker", ("darker", "dark", "murkier", "moodier", "gloomier")),
    ("brighter", ("brighter", "bright", "shinier", "crisper", "open it up")),
    ("warmer", ("warmer", "warm", "rounder", "softer tone")),
    ("sparser", ("sparser", "sparse", "strip it back", "strip it down",
                 "thin it out", "thinner", "less busy", "pare it back",
                 "simplify", "space")),
    ("busier", ("busier", "busy", "fill it up", "denser", "dense",
                "more notes", "ornament")),
    ("dry", ("dry", "drier", "less reverb", "kill the reverb")),
    ("wet", ("wet", "wetter", "more reverb", "washier", "more space")),
)

_ENERGY_UP = ("louder", "harder", "bigger", "lift it", "lift", "build it",
              "more energy", "turn it up", "turn up", "go off", "hype",
              "amp it up", "bring it up", "raise it", "hotter", "heavier",
              "more")
_ENERGY_DOWN = ("quieter", "softer", "calmer", "mellow", "mellower", "chill",
                "chiller", "back off", "pull back", "drop it down",
                "bring it down", "take it down", "cool it", "smaller",
                "lighter", "less")

_ENERGY_DELTA = 0.2


def _cue_words(text: object) -> str:
    """Normalize any input to lowercase words; never raises."""
    try:
        raw = "" if text is None else str(text)
    except Exception:                       # pragma: no cover - defensive
        return ""
    low = raw.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", low)
    return " ".join(cleaned.split())


def _has(words: str, phrases) -> str | None:
    """First phrase in ``phrases`` present in ``words`` on word boundaries."""
    for p in phrases:
        if re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", words):
            return p
    return None


def parse_cue(text: str) -> dict:
    """Rule-based cue parsing (SPEC-V2 section 3).

    Returns ``{"kind": ..., "value": ..., "text": <original>}`` plus a few
    kind-specific extras (``delta`` for energy, ``ratio`` for tempo).
    ``kind`` is an element of :data:`CUE_KINDS`; ``value`` is always a short
    canonical token (or ``None`` for ``"unknown"``).

    v2 AMENDMENT 2 (SPEC-V2 section 3): junk in, ``kind="unknown"`` out.
    This function NEVER raises — not on ``None``, not on a non-string, not
    on empty text.  A live cue arriving over a socket mid-song must not be
    able to take the band down, and guessing a kind for text we did not
    understand is the worse failure: it would change the music for no
    reason the player could trace.
    """
    raw = "" if text is None else str(text)
    w = _cue_words(raw)

    def out(kind: str, value, **extra) -> dict:
        d = {"kind": kind, "value": value, "text": raw}
        d.update(extra)
        return d

    if not w:
        return out("unknown", None)

    # Order matters: "drop to half time" is a TEMPO cue, not a section jump
    # to "drop" nor an energy drop; "last chorus" is an ENDING, not a jump.
    if _has(w, _END_PHRASES):
        return out("end", "out")

    for value, phrases in _TEMPO_PHRASES:
        if _has(w, phrases):
            ratio = {"half": 0.5, "double": 2.0}.get(value)
            return out("tempo", value, ratio=ratio)

    if _has(w, ("next section", "next part")):
        return out("section", "next")
    if _has(w, ("previous section", "last section", "back a section",
               "previous part")):
        return out("section", "prev")
    name = _has(w, _SECTION_NAMES)
    directional = _has(w, _DIRECTIONAL)
    if name is not None and (directional is not None
                             or name not in _AMBIGUOUS_SECTIONS):
        return out("section", "prechorus" if name == "pre chorus" else name)
    if directional is not None:
        # v2.1 AMENDMENT (SPEC-V2 section 3).  THE DEFECT: the parser knew
        # nineteen generic song-section names and v0 `compose` never emits one
        # of them — it names sections `settle-a`, `drift-b-outro`,
        # `establish-a`, `peak-b`, `neon-rise-a`, `apex-b`.  The parser's
        # vocabulary and the composer's had an EMPTY INTERSECTION, so the whole
        # section-cue kind was dead on every session plume can actually
        # produce: "go to the arc-c" parsed as `unknown` and logged "not
        # understood", while `plume cue`'s own help example, "go to the
        # chorus", parsed cleanly and then named a section no plume session
        # has.  That is the v1 lesson exactly — green on hand-named fixtures,
        # unusable on what plume itself makes.
        #
        # So a directional phrase now CAPTURES whatever follows it as a section
        # name, rather than requiring the name to be on a list.  "go to the"
        # is an unambiguous statement that a section is being named; what the
        # user calls their sections is their business, and whether that name
        # exists is the map's business to answer (and to log when it does not).
        tail = w.split(directional, 1)[1].strip()
        for lead in ("the ", "a ", "that "):
            if tail.startswith(lead):
                tail = tail[len(lead):]
        tail = tail.split(" section")[0].split(" part")[0].strip()
        if tail:
            return out("section", tail)

    style = _has(w, tuple(_STYLE_NAMES))
    if style is not None:
        return out("style", _STYLE_NAMES[style])

    for value, phrases in _TEXTURE_PHRASES:
        if _has(w, phrases):
            return out("texture", value)

    if _has(w, _ENERGY_UP):
        return out("energy", "up", delta=_ENERGY_DELTA)
    if _has(w, _ENERGY_DOWN):
        return out("energy", "down", delta=-_ENERGY_DELTA)

    return out("unknown", None)


_FILLER_WORDS = ("please", "now", "a bit", "a little")


def cue_is_exact(text: object) -> bool:
    """Is this text a known cue and nothing more?

    ``parse_cue`` is built to find a cue inside anything a player might say, which is
    right for acting on it and wrong for repeating it.  A section name counts here only
    if it is one of the generic ones, because a map's own names can be anything.
    """
    w = _cue_words(text)
    for filler in _FILLER_WORDS:
        w = w.removeprefix(filler + " ").removesuffix(" " + filler)
    known = {*_END_PHRASES, *_ENERGY_UP, *_ENERGY_DOWN, *_STYLE_NAMES,
             "next section", "next part", "previous section", "last section",
             "back a section", "previous part"}
    for _, phrases in (*_TEMPO_PHRASES, *_TEXTURE_PHRASES):
        known.update(phrases)
    if w in known:
        return True
    names = {*_SECTION_NAMES, *_GENERIC_SECTION_ROLES}
    if w in names - _AMBIGUOUS_SECTIONS:
        return True
    for directional in _DIRECTIONAL:
        if w.startswith(directional + " "):
            tail = w[len(directional) + 1:]
            for lead in ("the ", "a ", "that "):
                tail = tail.removeprefix(lead)
            return tail.removesuffix(" section").removesuffix(" part") in names
    return False


# --------------------------------------------------------- the form map

@dataclass(frozen=True)
class _Sec:
    """One playable section of the flattened arrangement."""

    idx: int
    movement: int
    section: int
    name: str
    bars: int
    energy: float


def flatten_sections(session: "Session | None") -> tuple[_Sec, ...]:
    """Flatten ``session.movements`` into the cursor's address space.

    Sections with no bars are dropped — they are unplayable, and leaving
    them in makes "advance to the next section" a step that consumes no
    time (an infinite loop waiting to happen).  Duck-typed on purpose:
    ``plan_form`` accepts anything shaped like a v0 ``Session``.
    """
    out: list[_Sec] = []
    movements = getattr(session, "movements", None) or []
    for mi, mv in enumerate(movements):
        for si, sec in enumerate(getattr(mv, "sections", None) or []):
            try:
                bars = int(getattr(sec, "bars", 0))
            except (TypeError, ValueError):
                bars = 0
            if bars <= 0:
                continue
            try:
                energy = float(getattr(sec, "energy", 0.5))
            except (TypeError, ValueError):
                energy = 0.5
            out.append(_Sec(idx=len(out), movement=mi, section=si,
                            name=str(getattr(sec, "name", "")),
                            bars=bars, energy=energy))
    return tuple(out)


def _form_authority_of(role: object) -> float:
    """``role.form_authority`` if it is readable, else 1.0.

    v2 AMENDMENT 3 (SPEC-V2 section 2): ``plan_form``'s ``role`` argument is
    typed ``RoleState`` in the spec, but ``roles.py`` is a separate module
    built alongside this one, so this file does not import it — the argument
    is duck-typed and ``None`` is legal.  Being called at all implies the
    machine is navigating a form, so an unreadable/absent role means full
    form authority rather than none.
    """
    if role is None:
        return 1.0
    try:
        return float(getattr(role, "form_authority", 1.0))
    except (TypeError, ValueError):
        return 1.0


class Mind:
    """The negotiation state machine.  Pure, deterministic, rng-free."""

    def __init__(self, params: MindParams, form: FormParams | None = None):
        """``form`` (v2, additive keyword) parameterizes ``plan_form`` only.

        Existing ``Mind(params)`` call sites are unaffected: nothing below
        the v2 block is read by ``plan_bar``.
        """
        self.params = params
        self.form_params = form if form is not None else FormParams()
        # --- v2 form cursor (SPEC-V2 section 2); plan_bar never reads these.
        self._form_calls = 0              # bar boundaries seen by plan_form
        self._form_idx = 0                # current section (flattened index)
        self._form_pos = 0                # bar-in-section of the upcoming bar
        self._form_span = 0               # bars in the current section span
        self._form_extended = 0           # extension bars added to this span
        self._form_ending = False         # the ending suite owns the form now
        self._form_mid_phrase = False     # human still building at the last bar
        self._form_fast_db: float | None = None   # human's recent level
        self._form_slow_db: float | None = None   # ...vs their baseline
        self._form_energy_run = 0         # signed consecutive energy evidence
        self._form_prev_rms: float | None = None
        self._form_prev_dens: float | None = None
        self._form_bpm_hist: list[float] = []
        self._form_last_oct: int | None = None    # tempo re-reads seen so far
        self._form_cues: list[_QueuedCue] = []
        self._heard_any = False           # any human signal ever observed
        self._entered = False             # machine currently in an active arc
        self._bars_listened = 0           # cumulative non-silent human bars heard
        self._bars_on_grid = 0            # ...of which were measured while locked
        self._bars_since_entry = 0        # 0 on the entry bar itself
        self._silent_bars = 0             # consecutive silent human bars (frozen during beat-loss)
        self._density_thinned = False     # inverse density coupling engaged
        self._density_recovery_pending = False   # one-bar recovery delay armed
        self._last_density_seen: float | None = None
        self._ema_rms_db: float | None = None    # EMA of human bar RMS (non-silent bars)
        self._entry_ref_db: float | None = None  # human level reference at entry
        self._last_gain_db = 0.0
        self._beatloss_stage = 0          # 0 none, 1 faded one bar
        self._rearm_from: str | None = None      # why the entry logic re-armed
        self._last_ceiling: float | None = None  # last known register ceiling
        # --- v1.4 commitment (see the module docstring) -------------------
        self._commit_left = 0             # committed bars remaining in this arc
        self._unlocked_run = 0            # consecutive plan calls with no lock
        self._last_active_state = "comping"   # last SOUNDING state issued
        self._fade_began_bar: int | None = None   # bar the exit gesture started
        self._exit_bar: int | None = None     # ...of the arc that actually ended
        self._exit_early = False          # ...and whether it broke a commitment
        self._entry_pledge: int | None = None  # bar we ANNOUNCED we'd enter at

    # ------------------------------------------------------------------ API

    def credit_listened_bars(self, n: int) -> None:
        """Credit ``n`` non-silent human bars heard OUTSIDE a plan boundary.

        v1.2 AMENDMENT (SPEC-V1 section 4).  ``min_listen_bars`` is specified
        as "non-silent human bars heard", but ``plan_bar`` only ever receives
        a ``BarFeatures`` at a LOCKED bar boundary, so bars the human played
        before the tracker locked were heard and then forgotten — the gate
        meant "4 bars after lock", and entry landed 50% into a real take.  The
        caller (sim / realtime) credits those bars here.  This deliberately
        does NOT plan, sound, or arm anything: entry still requires
        ``beat.locked`` and a real ``entry_boundary_bars`` boundary.
        """
        n = int(n)
        if n <= 0:
            return
        self._bars_listened += n
        self._heard_any = True

    def plan_bar(self, beat: BeatState, harmony: HarmonyState,
                 human: BarFeatures | None) -> BarPlan:
        p = self.params
        conf = float(beat.confidence)

        # The upcoming bar this plan is for.  With a completed human bar the
        # grid is unambiguous; unlocked calls fall back to the (frozen) beat
        # counter, or -1 before any lock.
        if human is not None:
            upcoming = int(human.bar) + 1
        elif beat.bar >= 0:
            upcoming = int(beat.bar) + 1
        else:
            upcoming = -1

        h_silent: bool | None = None
        if human is not None:
            # v1.2 AMENDMENT: ``BarFeatures.silent`` is the contract.  This
            # line used to OR in a SECOND, independent absolute-RMS test —
            # ``float(human.rms_db) <= p.silence_rms_db`` — which the feature
            # extractor's onset guard could not reach, so it silently
            # overrode the guarded decision.  (Proven: monkeypatching
            # ``sim._BarAccum.is_silent`` to always return False changed
            # nothing at all; the mind was re-deriving silence from one
            # scalar it cannot contextualize.)  Only the layer holding the
            # frame data — per-frame RMS, onsets, the bar edge trim, the
            # human's running level — can decide this.
            h_silent = bool(human.silent)
            if not h_silent:
                self._heard_any = True
        if beat.bpm > 0.0 or conf > 0.0:
            self._heard_any = True

        # --- emergency: beat-loss repair (silence counters frozen) ---------
        #
        # v1.4 COMMITMENT.  While committed the machine tolerates the ordinary:
        # the floor drops from conf_floor to exit_conf, and an unlocked tracker
        # is a wobble for _UNLOCK_GRACE_BARS calls before it counts as a true
        # unlock.  Outside the commitment window this block is byte-for-byte
        # the v1.2 behaviour.
        if self._entered:
            committed = self._commit_left > 0
            self._unlocked_run = self._unlocked_run + 1 if not beat.locked else 0
            floor = p.exit_conf if committed else p.conf_floor
            true_unlock = (not beat.locked
                           and (not committed
                                or self._unlocked_run > _UNLOCK_GRACE_BARS))
            if conf < floor or true_unlock:
                return self._plan_beat_loss(upcoming, conf, floor)
            if not beat.locked:
                # Committed, inside the grace window: hold the last sounding
                # bar on the last known grid rather than bail on a wobble.
                return self._plan_unlock_hold(upcoming, conf)
        else:
            self._beatloss_stage = 0
            self._unlocked_run = 0

        # --- per-bar bookkeeping (only when the beat is trustworthy) -------
        if human is not None:
            if h_silent:
                self._silent_bars += 1
            else:
                self._silent_bars = 0
                self._bars_listened += 1
                if beat.locked:
                    self._bars_on_grid += 1
                self._last_density_seen = float(human.onset_density)
                self._update_density(float(human.onset_density))
                self._update_ema(float(human.rms_db))
        ceiling = self._update_ceiling(human, harmony)

        if self._entered:
            # Commitment is spent in BARS ON THE GRID.  A held wobble bar
            # (_plan_unlock_hold) and a beat-loss fade both return above this
            # line, so neither burns it: a tracker glitch must not be able to
            # run the commitment out and put the machine back on a hair
            # trigger.  Unlocked ticks arrive with human=None once a second
            # and are not bars at all.
            if human is not None and self._commit_left > 0:
                self._commit_left -= 1
            # The beat came back and the machine is playing a normal bar, so
            # the repair is over: a LATER loss must fade before it lays out,
            # not jump straight to tacet on a stale stage counter.  This
            # mattered little when a recovery was nearly impossible; with the
            # wobble grace above, recoveries are the common case.
            self._beatloss_stage = 0
            return self._plan_entered(upcoming, conf, ceiling)
        return self._plan_not_entered(beat, upcoming, conf, ceiling, human)

    # ----------------------------------------------------------- internals

    def _update_density(self, d: float) -> None:
        p = self.params
        lo = p.density_high - p.density_hysteresis
        if not self._density_thinned:
            if d > p.density_high:
                self._density_thinned = True
                self._density_recovery_pending = False
        elif d > p.density_high:
            self._density_recovery_pending = False       # re-confirmed busy
        elif d < lo:
            if self._density_recovery_pending:           # second calm bar: recover
                self._density_thinned = False
                self._density_recovery_pending = False
            else:                                        # first calm bar: wait one
                self._density_recovery_pending = True
        else:
            self._density_recovery_pending = False       # mid-zone: hold thin

    def _update_ema(self, rms_db: float) -> None:
        a = self.params.energy_alpha
        if self._ema_rms_db is None:
            self._ema_rms_db = rms_db
        else:
            self._ema_rms_db = a * rms_db + (1.0 - a) * self._ema_rms_db

    def _update_ceiling(self, human: BarFeatures | None,
                        harmony: HarmonyState) -> float | None:
        reg_low = None
        if human is not None and human.register_low_midi is not None:
            reg_low = float(human.register_low_midi)
        elif harmony.register_low_midi is not None:
            reg_low = float(harmony.register_low_midi)
        if reg_low is not None:
            self._last_ceiling = reg_low - self.params.register_margin_st
        return self._last_ceiling

    @staticmethod
    def _db_str(x: float) -> str:
        """Signed dB for a reason string, without the '-0.0dB' papercut."""
        return f"{x + 0.0:+.1f}" if abs(x) >= 0.05 else "+0.0"

    def _arc_gain(self) -> float:
        if self._ema_rms_db is None or self._entry_ref_db is None:
            return 0.0
        delta = self._ema_rms_db - self._entry_ref_db
        return max(_ARC_MIN_DB, min(_ARC_MAX_DB, delta))

    def _plan(self, bar: int, state: str, gain_db: float, density: float,
              ceiling: float | None, reason: str,
              announce: tuple[str, str] | None = None) -> BarPlan:
        """Build one :class:`BarPlan` and record the bookkeeping it implies.

        ``announce`` is ``(kind, what)`` for the bar AFTER ``bar`` — see the
        module docstring.  The reason string carries it in plain words because
        the decisions NDJSON is how a player finds out what the machine
        thought it was doing.
        """
        a_text = a_kind = None
        a_bar = None
        if announce is not None:
            a_kind, what = announce
            a_bar = bar + 1
            a_text = f"{what} at bar {a_bar}"
            gesture = _ANNOUNCE_GESTURE.get(a_kind, "cue")
            reason = (f"{reason}; announcing {a_kind} next bar "
                      f"({gesture}): {a_text}")
        if state in _ACTIVE_STATES:
            self._last_active_state = state
            if state != "fading":
                self._fade_began_bar = None      # the leaving was called off
            elif self._fade_began_bar is None:
                self._fade_began_bar = bar       # the exit gesture starts here
        roles = dict(_ROLES.get(state, {}))
        return BarPlan(bar=bar, state=state, roles=roles, gain_db=gain_db,
                       density=density,
                       register_ceiling_midi=ceiling if state in _ACTIVE_STATES else None,
                       reason=reason, announce=a_text, announce_kind=a_kind,
                       announce_bar=a_bar)

    def _end_arc(self, upcoming: int) -> None:
        """Record an arc ending, for ``recommit_bars`` (v1.4).

        The exit is dated from the bar the exit GESTURE began (the first
        ``fading`` bar), not from the tacet — a listener hears the leaving
        start when the gain starts dropping.

        It counts as EARLY, and so arms the lockout, only when BOTH hold:

        * the commitment was still running when the arc ended, and
        * **the human was still playing when the machine left**
          (``_silent_bars == 0``).

        The second condition is the one that matters and it was learned the
        expensive way: without it, a long stop that ran the silence ladder out
        and took the tracker's lock with it (counters frozen at 4 silent bars,
        confidence 0.21) armed a lockout, and when the human came back the
        machine sat out the next boundary for a "commitment" the human had
        broken, not it.  ``recommit_bars`` is a penalty for the machine
        bailing on a player who was still going — that is the "starts up
        randomly later" half of the complaint.  A player who stopped and came
        back gets met at the next boundary, every time.
        """
        self._exit_bar = (self._fade_began_bar if self._fade_began_bar is not None
                          else upcoming)
        self._exit_early = self._commit_left > 0 and self._silent_bars == 0
        self._commit_left = 0
        self._fade_began_bar = None
        self._entry_pledge = None
        self._entered = False

    def _plan_unlock_hold(self, upcoming: int, conf: float) -> BarPlan:
        """Committed + tracker wobbling: hold the last sounding bar (v1.4).

        Deliberately does NOT advance the entry ramp or the silence counters —
        it is the same "missing data, not a miss" stance the beat tracker
        takes on a silent span.  ``_bars_since_entry`` is untouched, so a
        wobble during the ramp does not skip the ramp.
        """
        state = (self._last_active_state
                 if self._last_active_state in _ACTIVE_STATES else "comping")
        reason = (f"tracker wobbled (unlocked, conf {conf:.2f}) — committed, "
                  f"holding {state} on the last grid "
                  f"({self._unlocked_run}/{_UNLOCK_GRACE_BARS} grace bars used, "
                  f"{self._commit_left} committed bars left)")
        return self._plan(upcoming, state, self._last_gain_db,
                          _DENSITY.get(state, 0.0), self._last_ceiling, reason)

    def _plan_beat_loss(self, upcoming: int, conf: float,
                        floor: float) -> BarPlan:
        """``floor`` is exit_conf while committed, conf_floor otherwise."""
        if conf < floor:
            cause = f"conf {conf:.2f} < floor {floor:.2f}"
        else:
            cause = f"tracker unlocked (conf {conf:.2f})"
        if self._beatloss_stage == 0:
            self._beatloss_stage = 1
            gain = max(self._last_gain_db - _FADE_STEP_DB, _GAIN_FLOOR_DB)
            self._last_gain_db = gain
            reason = (f"beat lost ({cause}): fading out, "
                      f"silence counters frozen at {self._silent_bars}")
            return self._plan(upcoming, "fading", gain, _DENSITY["fading"],
                              self._last_ceiling, reason)
        self._end_arc(upcoming)
        self._beatloss_stage = 0
        self._rearm_from = "beat loss"
        reason = f"beat lost ({cause}): tacet — entry re-armed, awaiting confident relock"
        return self._plan(upcoming, "tacet", 0.0, 0.0, None, reason)

    def _silence_announce(self) -> tuple[str, str] | None:
        """The silence ladder's next rung, one bar ahead (v1.4 / P1-E).

        The ladder is a pure timer on ``_silent_bars``, so if the human stays
        silent the machine already knows what the next bar holds.  Announcing
        it is what turns "it stopped suddenly" into a gesture the human can
        answer — and answering it (playing again) resets ``_silent_bars`` and
        cancels the announced change, which is the cue doing its job.

        Only a CHANGE of rung is announced — a second fading bar is not news,
        and announcing what is already happening is the kind of log line that
        makes a decisions file useless.
        """
        nxt = self._silence_rung(self._silent_bars + 1)
        return nxt if nxt != self._silence_rung(self._silent_bars) else None

    def _silence_rung(self, n: int) -> tuple[str, str] | None:
        """Which rung of the silence ladder ``n`` consecutive silent bars is.

        Highest rung first: leaving outranks thinning.
        """
        p = self.params
        if n >= p.silence_tacet_bars:
            return ("exit", "laying out")
        if n >= p.silence_fade_bars:
            return ("exit", "fading out")
        if n >= p.silence_thin_bars:
            return ("texture", "thinning to the skeleton")
        return None

    def _plan_entered(self, upcoming: int, conf: float,
                      ceiling: float | None) -> BarPlan:
        p = self.params
        self._bars_since_entry += 1
        k = self._bars_since_entry
        ramp = max(1, p.ramp_bars)

        # Silence FSM (timer-driven; exempt from the boundary rule AND from
        # commitment — these are responses to the human's withdrawal, not
        # machine initiations, and a human who genuinely stops must still be
        # met with thinning and eventually silence.  v1.4: each rung of the
        # ladder is ANNOUNCED on the bar before it lands, so the human can
        # answer it (playing again cancels it) instead of being dropped.
        if self._silent_bars >= p.silence_tacet_bars:
            self._rearm_from = "tacet (silence)"
            reason = (f"tacet: human silent {self._silent_bars} bars "
                      f"(>= {p.silence_tacet_bars}) — laying out, entry re-armed")
            self._end_arc(upcoming)
            return self._plan(upcoming, "tacet", 0.0, 0.0, None, reason)
        if self._silent_bars >= p.silence_fade_bars:
            gain = max(self._last_gain_db - _FADE_STEP_DB, _GAIN_FLOOR_DB)
            self._last_gain_db = gain
            reason = (f"fading: human silent {self._silent_bars} bars "
                      f"(fade at {p.silence_fade_bars}, tacet at {p.silence_tacet_bars}), "
                      f"gain {self._db_str(gain)}dB")
            return self._plan(upcoming, "fading", gain, _DENSITY["fading"],
                              ceiling, reason,
                              announce=self._silence_announce())
        if self._silent_bars >= p.silence_thin_bars:
            reason = (f"thinned to skeleton: human silent {self._silent_bars} bars "
                      f"(>= {p.silence_thin_bars})")
            return self._plan(upcoming, "thinned", self._last_gain_db,
                              _DENSITY["thinned"], ceiling, reason,
                              announce=self._silence_announce())

        # Entry ramp still running?
        if k < ramp:
            gain = p.enter_under_db * (1.0 - k / ramp)
            self._last_gain_db = gain
            reason = (f"entering: ramp bar {k + 1}/{ramp}, "
                      f"gain {self._db_str(gain)}dB")
            return self._plan(upcoming, "entering", gain, _DENSITY["entering"],
                              ceiling, reason,
                              announce=self._silence_announce())

        # Comping (dynamic-arc gain; inverse density coupling on the hint).
        gain = self._arc_gain()
        self._last_gain_db = gain
        # v1.4: the commitment RENEWS while the music is going.  Eight bars of
        # commitment fixes nothing in a six-minute take if bar nine puts the
        # machine back on a hair trigger — a player decides whether to be in
        # the band once per SECTION.  So a healthy comping bar whose evidence
        # would still justify entering (conf >= entry_conf) buys another
        # commit_bars.  Renewal deliberately happens ONLY here: a thinned,
        # fading or ramping bar does not re-up, so the silence ladder always
        # runs out to tacet and the arc can still end.
        recommit_note = ""
        if self._commit_left <= 0 and conf >= p.entry_conf:
            self._commit_left = max(0, int(p.commit_bars))
            recommit_note = f", re-committed for {p.commit_bars} bars"

        d = self._last_density_seen
        d_str = f"{d:.1f}" if d is not None else "unmeasured"
        announce = self._silence_announce()
        if self._density_thinned:
            density = _DENSITY_BUSY
            lo = p.density_high - p.density_hysteresis
            if self._density_recovery_pending:
                reason = (f"comping sparse: human density fell to {d_str} "
                          f"(< {lo:.1f}) — recovering next bar")
                # The recovery already carries a deliberate one-bar delay, so
                # the mind knows a texture change lands next bar and can say
                # so.  (The THINNING that got us here fires the same bar it is
                # decided and is not announced: it is a reaction to the human's
                # own busyness, not something they need warning of.)
                if announce is None:
                    announce = ("texture", "filling back in")
            else:
                reason = (f"comping sparse: human busy ({d_str} onsets/beat "
                          f"> {p.density_high:.1f})")
        else:
            density = _DENSITY["comping"]
            if d is not None:
                reason = (f"comping: human {d_str} onsets/beat, "
                          f"density {density:.2f}, gain {self._db_str(gain)}dB")
            else:
                reason = (f"comping: human density unmeasured, "
                          f"density {density:.2f}, gain {self._db_str(gain)}dB")
        return self._plan(upcoming, "comping", gain, density,
                          ceiling, reason + recommit_note, announce=announce)

    def _plan_not_entered(self, beat: BeatState, upcoming: int, conf: float,
                          ceiling: float | None,
                          human: BarFeatures | None) -> BarPlan:
        p = self.params
        on_boundary = upcoming >= 0 and upcoming % p.entry_boundary_bars == 0
        # v1.2: bars must also have been measured ON the locked grid (see
        # _MIN_BARS_ON_GRID).  Entering on a lock threshold the confidence has
        # only just crossed is exactly the "entered on a wrong tempo
        # hypothesis" failure the enter-late gate exists to prevent.
        armed = (self._bars_listened >= p.min_listen_bars
                 and self._bars_on_grid >= _MIN_BARS_ON_GRID
                 and beat.locked and conf >= p.entry_conf)

        # v1.4: a pledge made last bar is HONOURED through a confidence dip.
        # This is the entry half of "it cant chicken out" — the machine that
        # announces a pickup and then does not come in is worse than one that
        # never announced, and on the real 0.41-0.62 trace the boundary sample
        # is a coin flip.  Only a tracker that is no longer locked, or the
        # boundary having already gone by, cancels it.
        if not beat.locked or (self._entry_pledge is not None
                               and upcoming > self._entry_pledge):
            self._entry_pledge = None
        pledged = (self._entry_pledge is not None
                   and upcoming == self._entry_pledge
                   and beat.locked and conf >= p.exit_conf
                   and self._bars_on_grid >= _MIN_BARS_ON_GRID)

        if (armed and on_boundary and not self._locked_out(upcoming)) or pledged:
            # Enter: under-gain, sparse, ramp begins.
            self._entered = True
            self._commit_left = max(0, int(p.commit_bars) - 1)   # this bar is the first
            self._exit_bar = None
            self._exit_early = False
            self._unlocked_run = 0
            was_pledged = self._entry_pledge == upcoming
            self._entry_pledge = None
            self._beatloss_stage = 0
            self._bars_since_entry = 0
            self._silent_bars = 0
            self._density_thinned = False
            self._density_recovery_pending = False
            if self._ema_rms_db is not None:
                self._entry_ref_db = self._ema_rms_db
            elif human is not None:
                self._entry_ref_db = float(human.rms_db)
            else:
                self._entry_ref_db = 0.0
            ramp = max(1, p.ramp_bars)
            gain = p.enter_under_db
            self._last_gain_db = gain
            rearm_note = f" (re-armed after {self._rearm_from})" if self._rearm_from else ""
            self._rearm_from = None
            pledge_note = (" — honouring the pickup announced last bar"
                           if was_pledged else "")
            reason = (f"entering: conf {conf:.2f} at bar {upcoming} boundary, "
                      f"{p.enter_under_db:+.1f}dB under human, ramp {ramp} bars, "
                      f"committed for {p.commit_bars} bars"
                      f"{pledge_note}{rearm_note}")
            return self._plan(upcoming, "entering", gain, _DENSITY["entering"],
                              ceiling, reason)

        if not self._heard_any:
            return self._plan(upcoming, "waiting", 0.0, 0.0, None,
                              "waiting: no human signal detected yet")

        # v1.4: pledge the entry a bar ahead so it can be ANNOUNCED, and so
        # the boundary sample stops being a coin flip.  A pledge is only made
        # when every gate is already open and the NEXT bar is a real entry
        # boundary — never mid-phrase, and never during the recommit lockout.
        announce = None
        nxt = upcoming + 1
        if (armed and upcoming >= 0 and not on_boundary
                and nxt % p.entry_boundary_bars == 0
                and not self._locked_out(nxt)):
            self._entry_pledge = nxt
            announce = ("entry", "coming in")

        # Listening: say exactly which gate is still closed.
        if self._bars_listened < p.min_listen_bars:
            sub = (f"heard {self._bars_listened}/{p.min_listen_bars} bars "
                   f"(conf {conf:.2f}, {'locked' if beat.locked else 'unlocked'})")
        elif not beat.locked:
            sub = f"awaiting beat lock (conf {conf:.2f})"
        elif self._bars_on_grid < _MIN_BARS_ON_GRID:
            sub = (f"heard {self._bars_listened} bars, "
                   f"{self._bars_on_grid}/{_MIN_BARS_ON_GRID} of them on the "
                   f"locked grid (conf {conf:.2f})")
        elif self._locked_out(upcoming) and armed and on_boundary:
            sub = (f"armed (conf {conf:.2f}) but sitting out "
                   f"{p.recommit_bars} bars after breaking a commitment at "
                   f"bar {self._exit_bar}")
        elif conf < p.entry_conf:
            sub = f"locked but conf {conf:.2f} < entry threshold {p.entry_conf:.2f}"
        else:
            if upcoming >= 0:
                nb = ((upcoming // p.entry_boundary_bars) + 1) * p.entry_boundary_bars
            else:
                nb = p.entry_boundary_bars
            sub = f"armed (conf {conf:.2f}), awaiting bar {nb} boundary"
        if self._rearm_from:
            sub = f"re-armed after {self._rearm_from}; {sub}"
        return self._plan(upcoming, "listening", 0.0, 0.0, None,
                          f"listening: {sub}", announce=announce)

    def _locked_out(self, bar: int) -> bool:
        """Is ``bar`` inside the ``recommit_bars`` lockout? (v1.4)

        Armed only by an arc that ended EARLY — see :meth:`_end_arc`.  Dated
        from the bar the exit gesture began, because that is when a listener
        heard the machine start to leave; measuring from the tacet would
        double-count the fade.
        """
        p = self.params
        if not self._exit_early or self._exit_bar is None or bar < 0:
            return False
        return bar - self._exit_bar < max(0, int(p.recommit_bars))

    # ------------------------------------------ v2: form navigation + cues

    def plan_form(self, role, beat: BeatState, harmony: HarmonyState,
                  human: BarFeatures | None, session: "Session",
                  cue: "Cue | None" = None) -> FormPlan:
        """Where the arrangement goes next (SPEC-V2 section 2).

        Called once per bar boundary, like :meth:`plan_bar`, and returns the
        decision for the UPCOMING bar.  The v0 ``Session`` is read as a MAP:
        the cursor may extend a section, jump to any other section, or wrap —
        composed order is the default route, not a script.

        Deterministic and side-effect free apart from the cursor's own memory
        (same discipline as ``plan_bar``): no rng, no I/O, no wall clock.  Two
        minds fed the same sequence return equal plans.

        ``role`` is duck-typed (see :func:`_form_authority_of`) and may be
        ``None``; ``cue`` is queued on arrival and applied at its boundary.
        """
        fp = self.form_params
        flat = flatten_sections(session)
        n_bar = self._form_calls
        self._form_calls += 1
        first = n_bar == 0

        if human is not None:
            bar = int(human.bar) + 1
        elif beat.bar >= 0:
            bar = int(beat.bar) + 1
        else:
            bar = -1

        if cue is not None:
            self._form_cues.append(_QueuedCue(
                cue=cue, parsed=parse_cue(getattr(cue, "text", "")),
                issued=n_bar))

        self._form_track_human(human)
        ritard = self._form_track_tempo(beat, human)

        if not flat:
            self._form_cues.clear()
            return FormPlan(bar=bar, section_idx=-1, action="hold",
                            bars_left_in_section=0,
                            reason="form: session has no playable sections")

        if self._form_idx >= len(flat):        # the map changed under us
            self._form_enter(flat, 0)
        cur = flat[self._form_idx]

        # --- first boundary: take up position, change nothing --------------
        if first or self._form_span <= 0:
            if self._form_span <= 0:             # keeps a seek_form position
                self._form_enter(flat, self._form_idx)
            cur = flat[self._form_idx]
            left = self._form_span - 1 - self._form_pos
            return FormPlan(
                bar=bar, section_idx=self._form_idx, action="hold",
                bars_left_in_section=max(0, left),
                announce=left <= 0,
                reason=(f"form: start of section {self._form_idx} "
                        f"{cur.name!r} ({cur.bars} bars, "
                        f"energy {cur.energy:.2f})"))

        # --- the ending suite outranks the form (SPEC-V2 rule 4) -----------
        if self._form_ending or ritard is not None:
            newly = not self._form_ending
            self._form_ending = True
            # An ending outranks queued instructions too: "go to the chorus"
            # cannot land after the band has committed to stopping.
            self._form_cues.clear()
            self._form_pos = min(self._form_pos + 1, max(0, self._form_span - 1))
            reason = (f"form: {ritard} — ending suite overrides the form"
                      if newly else
                      f"form: ending — section {self._form_idx} {cur.name!r} "
                      f"held for the final hit")
            return FormPlan(bar=bar, section_idx=self._form_idx, action="end",
                            bars_left_in_section=max(
                                0, self._form_span - 1 - self._form_pos),
                            reason=reason)

        authority = _form_authority_of(role)
        leads = authority >= fp.form_authority_min

        # --- resolve the position of the upcoming bar ----------------------
        self._form_pos += 1
        action = "hold"
        if self._form_pos >= self._form_span:
            action, reason = self._form_boundary(flat, leads, authority)
        else:
            cur = flat[self._form_idx]
            reason = (f"section {self._form_idx} {cur.name!r} bar "
                      f"{self._form_pos + 1}/{self._form_span} "
                      f"(energy {cur.energy:.2f})")

        # --- cues, evaluated against the RESOLVED position -----------------
        applied, dropped, cue_reason, cue_action = self._form_apply_cues(
            flat, n_bar)
        if cue_action is not None:
            reason = f"{reason}; overridden by {cue_reason}"
            action = cue_action
        elif cue_reason:
            reason = f"{reason}; {cue_reason}"
        reason = f"form: {reason}"

        cur = flat[self._form_idx]
        left = max(0, self._form_span - 1 - self._form_pos)
        return FormPlan(bar=bar, section_idx=self._form_idx, action=action,
                        bars_left_in_section=left,
                        announce=(left == 0 and action != "end"),
                        applied_cue=applied, dropped_cue=dropped,
                        reason=reason)

    # ------------------------------------------------ form internals (v2)

    def seek_form(self, session: "Session", bars: int) -> tuple[int, int]:
        """Place the cursor ``bars`` bars into the composed route, for the first
        :meth:`plan_form` to take up.  Past the end of the map it holds the last
        bar, where the form's own rules decide what happens next.

        Returns ``(section_idx, bar_in_section)``.
        """
        flat = flatten_sections(session)
        if not flat:
            return 0, 0
        left = max(0, int(bars))
        for sec in flat:
            if left < sec.bars:
                break
            left -= sec.bars
        else:
            left = sec.bars - 1
        self._form_enter(flat, sec.idx)
        self._form_pos = left
        return sec.idx, left

    def _form_enter(self, flat: tuple[_Sec, ...], idx: int) -> None:
        """Place the cursor at the first bar of ``flat[idx]``."""
        idx = max(0, min(int(idx), len(flat) - 1))
        self._form_idx = idx
        self._form_pos = 0
        self._form_span = flat[idx].bars
        self._form_extended = 0
        self._form_energy_run = 0

    def _form_track_human(self, human: BarFeatures | None) -> None:
        """Fold one completed human bar into the cursor's energy memory.

        v2 AMENDMENT 4 (SPEC-V2 section 2).  The spec says "sustained human
        RMS above the section's energy", but those are not comparable
        quantities: ``SectionSpec.energy`` is 0..1 and ``BarFeatures.rms_db``
        is dBFS, which moves 1:1 with the input gain.  Any fixed mapping
        between them is the v1 bug again — a rule that fires constantly on a
        hot input and never on a quiet one.  So energy is chased RELATIVELY:
        a fast EMA of the human's own level against a slow one standing in
        for their session baseline.  ``delta`` is a change in the human's
        playing, is invariant to input gain by construction, and is what
        "they lifted" actually means.
        """
        fp = self.form_params
        if human is None or bool(human.silent):
            self._form_mid_phrase = False
            if human is not None:
                self._form_energy_run = 0
            return
        rms = float(human.rms_db)
        dens = float(human.onset_density)

        rising = False
        if self._form_prev_rms is not None:
            rising = rms - self._form_prev_rms >= fp.phrase_rise_db
        if not rising and self._form_prev_dens is not None:
            floor = max(self._form_prev_dens, 0.25)
            rising = dens - self._form_prev_dens >= floor * fp.phrase_rise_density
        self._form_mid_phrase = rising
        self._form_prev_rms = rms
        self._form_prev_dens = dens

        a, b = fp.energy_fast_alpha, fp.energy_slow_alpha
        if self._form_fast_db is None:
            self._form_fast_db = rms
            self._form_slow_db = rms
        else:
            self._form_fast_db = a * rms + (1.0 - a) * self._form_fast_db
            self._form_slow_db = b * rms + (1.0 - b) * self._form_slow_db
        d = self._form_fast_db - (self._form_slow_db or 0.0)
        run = self._form_energy_run
        if d >= fp.energy_up_db:
            self._form_energy_run = run + 1 if run > 0 else 1
        elif d <= -fp.energy_down_db:
            self._form_energy_run = run - 1 if run < 0 else -1
        else:
            self._form_energy_run = 0

    def _form_energy_delta_db(self) -> float:
        if self._form_fast_db is None or self._form_slow_db is None:
            return 0.0
        return self._form_fast_db - self._form_slow_db

    def _form_track_tempo(self, beat: BeatState,
                          human: BarFeatures | None) -> str | None:
        """Ritardando detection: monotone slowing across ``ritard_bars``.

        v2 AMENDMENT 5 (SPEC-V2 section 2).  The spec defers this to "v1's
        ending suite", which does not exist in the v1 codebase — there is no
        ritard detector anywhere in ``listen/`` or ``mind.py``.  This is the
        minimal honest version: consecutive bar-boundary tempo estimates,
        each strictly lower than the last, totalling ``ritard_pct``, measured
        only while the tracker is locked and the human is playing (a silent
        bar's tempo estimate is not evidence about the human's intent).  The
        spec's other ending trigger — "the human plays the head again" — is
        NOT implemented: it needs melodic matching against the composed
        theme, which nothing in v0/v1 provides.

        TWO GUARDS, BOTH FOUND ON REAL MATERIAL (see tests/test_form.py):
        a synthwave render 20 dB down locks at 199.5 bpm and then commits an
        octave re-interpretation to 99.9 — which is 199.5 -> 99.95 -> 99.87,
        two strictly falling estimates totalling -50%, and the naive rule
        declared a ritardando on bar 2 and ended the tune for the remaining
        24 bars.  So: any increment of ``BeatState.tempo_octave_changes``
        clears the history (a re-read is not a player slowing down), and a
        single-bar drop over ``ritard_step_max_pct`` is disqualifying (a
        human ritard is a few percent per bar; a tracker jump is tens).
        """
        fp = self.form_params
        oct_changes = int(getattr(beat, "tempo_octave_changes", 0) or 0)
        reread = (self._form_last_oct is not None
                  and oct_changes != self._form_last_oct)
        self._form_last_oct = oct_changes
        if (reread or not beat.locked or float(beat.bpm) <= 0.0
                or human is None or bool(human.silent)):
            self._form_bpm_hist = []
            return None
        hist = self._form_bpm_hist
        hist.append(float(beat.bpm))
        keep = max(2, int(fp.ritard_bars) + 1)
        if len(hist) > keep:
            del hist[:-keep]
        if len(hist) < keep:
            return None
        for a, b in zip(hist, hist[1:]):
            step = (a - b) / a * 100.0
            if step <= 0.0 or step > fp.ritard_step_max_pct:
                return None
        drop = (hist[0] - hist[-1]) / hist[0] * 100.0
        if drop < fp.ritard_pct:
            return None
        self._form_bpm_hist = []
        return (f"ritardando {hist[0]:.1f} -> {hist[-1]:.1f} bpm over "
                f"{len(hist) - 1} bars (-{drop:.1f}% >= {fp.ritard_pct:.1f}%)")

    def _form_boundary(self, flat: tuple[_Sec, ...], leads: bool,
                       authority: float) -> tuple[str, str]:
        """Decide what happens at an exhausted section (rules 1, 2, 3)."""
        fp = self.form_params
        cur = flat[self._form_idx]

        # 1. Wait at the turnaround — the "one more chorus" gesture.
        if self._form_mid_phrase:
            unit = max(1, int(fp.extend_unit_bars))
            if self._form_extended + unit <= fp.max_extend_bars:
                self._form_extended += unit
                self._form_span += unit
                return "extend", (
                    f"extending {cur.name!r} by {unit} bars — human still "
                    f"mid-phrase at the turnaround "
                    f"({self._form_extended}/{fp.max_extend_bars} used)")
            over = (f"would extend {cur.name!r} but the budget is spent "
                    f"({self._form_extended}/{fp.max_extend_bars}); ")
        else:
            over = ""

        # 2. Energy chases the human.
        if leads and abs(self._form_energy_run) >= fp.energy_bars:
            direction = 1 if self._form_energy_run > 0 else -1
            run = abs(self._form_energy_run)
            target = self._form_energy_target(flat, direction)
            d = self._form_energy_delta_db()
            if target is not None:
                tgt = flat[target]
                self._form_enter(flat, target)   # resets the evidence run
                word = "above" if direction > 0 else "below"
                return "jump", (
                    f"{over}human {d:+.1f}dB {word} their baseline for "
                    f"{run} bars -> jump to section "
                    f"{target} {tgt.name!r} (energy {tgt.energy:.2f} vs "
                    f"{cur.energy:.2f})")
            over = (f"{over}human {d:+.1f}dB off baseline but no section is "
                    f"{'hotter' if direction > 0 else 'calmer'} than "
                    f"{cur.energy:.2f}; ")

        # 3. Otherwise take the composed route.
        old = self._form_idx
        nxt = self._form_idx + 1
        if nxt >= len(flat):
            if not fp.loop_form:
                self._form_ending = True
                return "end", (f"{over}last section {cur.name!r} complete "
                               f"and loop_form is off — ending")
            nxt = 0
        tgt = flat[nxt]
        self._form_enter(flat, nxt)
        if not leads:
            note = (f" (form_authority {authority:.2f} — following the "
                    f"composed route, initiating nothing)")
        else:
            note = ""
        return "advance", (f"{over}section {old} {cur.name!r} complete -> "
                           f"section {nxt} {tgt.name!r} ({tgt.bars} bars, "
                           f"energy {tgt.energy:.2f}){note}")

    def _form_energy_target(self, flat: tuple[_Sec, ...],
                            direction: int) -> int | None:
        """Nearest section (forward, then wrapping) at the wanted energy."""
        fp = self.form_params
        cur = flat[self._form_idx]
        order = (list(range(self._form_idx + 1, len(flat)))
                 + list(range(0, self._form_idx)))
        for i in order:
            e = flat[i].energy
            if direction > 0 and e >= cur.energy + fp.energy_margin:
                return i
            if direction < 0 and e <= cur.energy - fp.energy_margin:
                return i
        return None

    def _form_section_target(self, flat: tuple[_Sec, ...],
                             value: str) -> int | None:
        """Resolve a section cue's value to a flattened index.

        Three passes, in order, and the log line says which one hit.

        1. ``next`` / ``prev``.
        2. A literal name match against the map, punctuation-insensitive on
           BOTH sides — v2.1: ``want`` used to be compared raw against a
           normalized name, so ``"arc c"`` (which is what ``parse_cue`` gets
           after punctuation stripping) never matched the section actually
           called ``arc-c``.
        3. **v2.1: generic song-section words resolve by ROLE IN THE ARC.**
           `plume cue`'s own documented example is "go to the chorus" and no
           plume session has ever contained a section called that.  Rather
           than let the headline example be permanently dead, the handful of
           words that name a *position* in a song rather than a section of
           this one are resolved against the map's shape: an intro is the
           first span, an outro the last, a chorus/drop/hook the hottest, a
           breakdown the coldest, a verse the coldest span that is neither
           end.  This is a guess, so it is a LAST resort and the caller's log
           line names the section it actually chose.
        """
        if value == "next":
            return (self._form_idx + 1) % len(flat)
        if value == "prev":
            return (self._form_idx - 1) % len(flat)

        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", str(s).lower())

        want = norm(value)
        if not want:
            return None
        order = (list(range(self._form_idx + 1, len(flat)))
                 + [self._form_idx] + list(range(0, self._form_idx)))
        for i in order:
            if want in norm(flat[i].name):
                return i

        role = _GENERIC_SECTION_ROLES.get(want)
        if role is None or not flat:
            return None
        if role == "first":
            return 0
        if role == "last":
            return len(flat) - 1
        if role == "hottest":
            return max(range(len(flat)), key=lambda i: (flat[i].energy, -i))
        if role == "coldest":
            return min(range(len(flat)), key=lambda i: (flat[i].energy, i))
        if role == "middle_coldest":          # "verse"
            inner = list(range(1, len(flat) - 1)) or list(range(len(flat)))
            return min(inner, key=lambda i: (flat[i].energy, i))
        return None                           # pragma: no cover - defensive

    def _form_cue_due(self, q: _QueuedCue, n_bar: int) -> bool:
        """Is this queued cue due at the (already resolved) upcoming bar?"""
        fp = self.form_params
        at = str(getattr(q.cue, "at", "next-phrase") or "next-phrase")
        if at not in CUE_BOUNDARIES:
            at = "next-phrase"
        if at == "now":
            return True
        if at == "next-bar":
            return n_bar >= q.issued + 1
        if at == "next-section":
            return self._form_pos == 0 and n_bar > q.issued
        phrase = max(1, int(fp.phrase_bars))
        return self._form_pos % phrase == 0

    def _form_apply_cues(self, flat: tuple[_Sec, ...],
                         n_bar: int) -> tuple[str | None, str | None,
                                              str, str | None]:
        """Apply every due cue; expire the stale ones.

        Returns ``(applied_text, dropped_text, reason, action_override)``.
        Cue-driven section changes are the third chair speaking, so they are
        exempt from the announce rule (rule 3 covers MACHINE-initiated
        changes; a cue arrives with no notice by definition).

        Several cues can come due at one boundary; their texts are joined
        with "; " in ``applied_cue`` rather than silently dropping all but
        one, and each lands in ``reason`` with its own effect.
        """
        applied_all: list[str] = []
        dropped_all: list[str] = []
        notes: list[str] = []
        action: str | None = None
        keep: list[_QueuedCue] = []
        for q in self._form_cues:
            text = str(getattr(q.cue, "text", ""))
            kind = q.parsed.get("kind")
            value = q.parsed.get("value")
            if self._form_cue_due(q, n_bar):
                applied_all.append(text)
                if kind == "end":
                    self._form_ending = True
                    action = "end"
                    notes.append(f"cue {text!r} (end): ending the tune here")
                elif kind == "section":
                    target = self._form_section_target(flat, value)
                    if target is None:
                        notes.append(f"cue {text!r} (section:{value}): no such "
                                     f"section in the map — ignored")
                    elif target == self._form_idx and self._form_pos == 0:
                        notes.append(f"cue {text!r} (section:{value}): already "
                                     f"at section {target}")
                    else:
                        tgt = flat[target]
                        self._form_enter(flat, target)
                        action = "jump"
                        notes.append(f"cue {text!r} (section:{value}) -> "
                                     f"section {target} {tgt.name!r}")
                elif kind == "energy":
                    direction = 1 if value == "up" else -1
                    target = self._form_energy_target(flat, direction)
                    if target is None:
                        notes.append(f"cue {text!r} (energy:{value}): no "
                                     f"{'hotter' if direction > 0 else 'calmer'}"
                                     f" section in the map — ignored")
                    else:
                        tgt = flat[target]
                        self._form_enter(flat, target)
                        action = "jump"
                        notes.append(f"cue {text!r} (energy:{value}) -> section "
                                     f"{target} {tgt.name!r} "
                                     f"(energy {tgt.energy:.2f})")
                elif kind == "unknown":
                    notes.append(f"cue {text!r}: not understood — no change")
                else:
                    # v2.1: NOT "applied".  The form cursor is not the thing a
                    # tempo/texture/style cue acts on, and saying "applied"
                    # here was a claim this object cannot make — MEASURED, a
                    # machine solo logged `(tempo:faster): applied` and then
                    # rendered a byte-identical WAV, which is precisely the
                    # kind of lying log line that hid v1's four bugs.  What is
                    # true is that the cue was understood and handed on; who
                    # honours it, and whether they did, is theirs to report.
                    notes.append(f"cue {text!r} ({kind}:{value}): understood, "
                                 f"form unchanged — for the renderer")
                continue
            if n_bar - q.issued >= max(0, int(getattr(q.cue, "ttl_bars", 0))):
                dropped_all.append(text)
                notes.append(f"cue {text!r} expired unapplied after "
                             f"{n_bar - q.issued} bars "
                             f"(ttl {getattr(q.cue, 'ttl_bars', 0)}, "
                             f"at {getattr(q.cue, 'at', '?')})")
                continue
            keep.append(q)
        self._form_cues = keep
        return ("; ".join(applied_all) or None,
                "; ".join(dropped_all) or None,
                "; ".join(notes), action)

    def pending_cues(self) -> tuple[Cue, ...]:
        """Cues queued but not yet applied (introspection for the log/CLI)."""
        return tuple(q.cue for q in self._form_cues)
