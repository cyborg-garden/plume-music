"""Causal tempo / phase / downbeat tracking with prediction (SPEC-V1 section 2).

The ADAM findings, encoded:

- **Predict, don't react.**  The tracker maintains a predicted next-beat time
  and free-runs on the current period between onsets.  Detected onsets within
  +/-35% of a beat correct the prediction only partially
  (``next_beat += tightness * error`` — alpha-correction, never a snap).
- **Tempo != phase.**  The period comes from autocorrelation of the onset
  envelope over a trailing ~12s window (re-estimated every ~0.5s) with period
  hysteresis: an estimate differing by more than ``drift_pct`` percent must
  persist for 2 evaluations before the bpm switches, except an exact
  half/double of the current period, which is resolved immediately toward the
  70-160 bpm range preferring the onset-density fit.  Period changes never
  move the phase.
- **Once SETTLED, the GRID is sticky** (v1.3, widened in v2.7).  A drift-band
  adjustment is tracking and keeps the 2-evaluation ``drift_pct`` hysteresis
  above.  Anything outside that band is a re-interpretation of the metre: once
  the tracker has been locked on one grid for ``OCTAVE_SETTLE_S`` it yields
  only to ``OCTAVE_SWITCH_EVALS`` consecutive winning evaluations.  v1.3
  applied this to whole-octave jumps only; v2.7 applies it to EVERY
  out-of-drift re-read, because on live acoustic the octave is 4% of them and
  the 4/3 and 3/2 dotted aliases are the rest.  Either way a locked re-read is
  announced — a ``tempo_octave_change`` (octaves) or ``tempo_grid_change``
  (everything else) on ``BeatTracker.events``, and an increment of
  ``BeatState.tempo_grid_changes`` (plus ``tempo_octave_changes`` for octaves),
  with a confidence cut on the settled path.  See ``_grid_hysteresis``.
- **Downbeat by accent + harmonic change.**  The 4 beat-phase hypotheses are
  scored over a rolling 2-bar window by onset-strength accents and by
  chroma-change alignment (harmonic change lands on downbeats); the best
  hypothesis must lead for a full bar to commit, and overriding a committed
  downbeat needs 2 bars of contrary evidence.
- **Confidence** is autocorrelation peak salience times the fraction of the
  last 8 predicted beats matched by onsets, smoothed with an EMA (0.8).
  Rubato (events present but off-grid) decays it toward 0; below 0.3 for a
  full bar the tracker unlocks (bar counter freezes, phase free-runs).
  **Silence is not rubato**: beat spans with no audible input are missing
  data, not misses — confidence holds through a ~2-bar break
  (``SILENT_HOLD_BEATS``), then decays, so a short stop survives (v1 behavior
  3) while a long stop still unlocks (behavior 4).
- ``tap()`` is the count-in escape hatch: >=4 roughly even taps hard-set the
  period (median IOI) and bar phase (the last tap is beat ``beat_in_bar`` of
  bar 0, default 0) with confidence 1; afterwards the tracker only corrects
  drift.  ``features.CountInDetector`` reaches it from a live microphone —
  and ``BeatTracker(count_in=True)`` (or ``PLUME_COUNT_IN=1``) arms that
  detector INSIDE the tracker, so any entry point that constructs a tracker
  can offer the hatch without owning the wiring (v1.5.2; the detector shipped
  in v1.5.1 with the honest caveat that nothing constructed it).
- **The envelope is ``Frame.flux`` BY DEFAULT** and ``Frame.attack`` under the
  opt-in ``acoustic`` mode (v1.5.1).  The banded, level-independent attack
  function is what makes sustained open-tuned acoustic guitar trackable at
  all, and it is what makes programmed material measurably worse; the two
  could not be reconciled, so they are a switch.  ``BeatTracker(acoustic=True)``
  or ``PLUME_ACOUSTIC=1`` selects it.  See ``ACOUSTIC MODE`` below for the
  A/B that forced the default back, and ``features.ATTACK_*`` for the
  detection function itself.

Deterministic: pure function of the frame stream + taps + the mode chosen at
construction.  No rng, no clock.
"""
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # runtime-decoupled: frames are duck-typed (attribute access)
    from .features import Frame

# --- tuning constants (internal; the frozen constructor surface is below) ---
# ACF_WINDOW_S 6.0 -> 12.0 — v1.2 AMENDMENT (SPEC-V1 section 2).  The BAR is
# measurably the cleanest periodicity in real material (normalized ACF at the
# bar lag +0.807 vs +0.191 at the true beat lag, measured on plume's own lo-fi
# render), and it is the term that tells the true beat (whose multiples all
# land on-grid) apart from its 3/2 and 3/4 aliases (whose multiples do not).
# At 6s the bar lag of anything under 120bpm exceeded the search window
# entirely, and a 6s window spans only two bars at 80bpm.
ACF_WINDOW_S = 12.0       # trailing onset-envelope window for autocorrelation
EVAL_PERIOD_S = 0.5       # tempo re-estimation cadence
ONSET_TOL_FRAC = 0.35     # phase correction only from onsets within this beat fraction
UNLOCK_CONF = 0.3         # confidence below this for >= 1 bar -> unlock
# PREF_BPM_LO/HI: the spec's "resolved toward 70-160" band.  As of the v1.2
# amendment this is expressed as a continuous log-normal prior
# (_PRIOR_CENTER_BPM / _PRIOR_SIGMA_OCT below) rather than a binary flag; the
# constants are kept because they name the band the prior is centred on.
PREF_BPM_LO = 70.0
PREF_BPM_HI = 160.0
CONF_EMA = 0.8            # confidence smoothing (per predicted beat)
HIT_WINDOW_BEATS = 8      # predicted-beat hit rate window
LOCK_BEATS = 8            # 2 bars (4/4) of conf >= lock_conf -> locked
UNLOCK_BEATS = 4          # 1 bar of conf < UNLOCK_CONF -> unlocked
DOWNBEAT_WINDOW = 8       # rolling 2 bars of per-beat downbeat evidence
DOWNBEAT_MARGIN = 0.15    # best hypothesis must lead 2nd by this relative margin
DOWNBEAT_COMMIT_BEATS = 4     # 1 bar of agreement commits the downbeat
DOWNBEAT_OVERRIDE_BEATS = 8   # 2 bars of contrary evidence overrides a commit
TAP_MIN = 4               # taps needed for a count-in
TAP_EVEN_FRAC = 0.25      # every tap IOI within this fraction of the median
# Silence handling (behavior 3 of the v1 matrix): a stop is not rubato.  A beat
# span with no audible input is missing data, not a miss — confidence HOLDS for
# up to SILENT_HOLD_BEATS consecutive silent beats (a musical 2-bar break),
# then decays multiplicatively so a long stop still unlocks (behavior 4).
SILENT_SPAN_RMS_DB = -60.0    # max frame rms below this => the beat span was silent
SILENT_HOLD_BEATS = 8         # 2 bars of silence ride-out before decay starts
SILENT_DECAY = 0.9            # per-beat confidence decay after the hold window
_DEFAULT_FRAME_DT = 512.0 / 48000.0
_MIN_EVAL_ONSETS = 4      # autocorrelation needs events to mean anything
# Tempo candidate scoring — v1.2 AMENDMENT (SPEC-V1 section 2).  See
# _evaluate_tempo's docstring for the measurements behind each constant.
_COMB_WEIGHTS = (1.0, 0.6, 0.35, 0.8)    # rectified ACF comb over beat multiples
_PRIOR_CENTER_BPM = 110.0  # log-normal tempo prior centre ...
_PRIOR_SIGMA_OCT = 0.9     # ... and width in octaves.  Wide on purpose: the
# prior exists to break ties the ACF genuinely cannot, not to overrule
# evidence.  At 0.55 it penalised a true 60bpm by 3.3x against its own double
# and invented 120 out of a clean 60bpm click train — the exact failure the
# spec's "a true 65 still beats a fabricated 130 with no ACF support" warns
# about.  At 0.9 the 60-vs-120 comb ratio (1.84) survives it.
_NEST_BAND_S = (1.2, 5.0)  # where to look for the supra-beat (bar) lag
_NEST_TOL = 0.06           # relative tolerance on the bar/beat ratio
_NEST_BONUS = 1.25         # ratio ~= 4 (beats nest 4 to a bar)
_NEST_NEUTRAL = 1.0        # ratio ~= 2 or 8: consistent, uninformative
_NEST_PENALTY = 0.6        # anything else: the candidate is not a beat
_CONTINUITY_BONUS = 1.12   # candidate within drift_pct of the incumbent
_HINT_SIGMA_OCT = 0.14     # prior width (octaves) when a bpm hint is given
_TAP_CONF_FLOOR = 0.75    # confidence floor while a count-in is anchored
# Phase re-acquisition (v1.2): _phase_correct ignores onsets beyond
# +/-ONSET_TOL_FRAC of a beat, so a grid that lands a half beat off is
# mathematically unreachable by correction — the tracker spins forever with a
# perfect tempo and a zero hit rate.  After this many consecutive audible
# predicted beats with no onset anywhere near them, the phase is re-seeded
# from the most recent onset.
REACQUIRE_MISSES = 4
# --- octave stickiness (v1.3 AMENDMENT, SPEC-V1 section 2) ------------------
# A LOCKED tracker that has been on one octave for a while is being played
# along with.  A within-octave period adjustment is tracking; an OCTAVE change
# is a re-interpretation of the metre, and it must not happen quietly.  Before
# v1.3 both went through the same 2-evaluation hysteresis — one second — so a
# 50-second-old 140bpm lock could halve at bar 20 of a trap take and stay
# there, silently, with confidence RISING across the switch.
#
# WHAT SEPARATES A CORRECTION FROM A COLLAPSE — measured, not assumed.  Three
# cases, all with the challenger exactly one octave from the grid in force:
#
#   case                       when            challenger score / grid in force
#   -------------------------- --------------- ---------------------------------
#   groove80 (truth 80 @ 160)  1.5 s post-lock  1.01-1.08, 82 evals unbroken
#   busy80   (truth 80 @ 160)  ~1 s post-lock   1.10-1.17, 82 evals unbroken
#   trap collapse (truth 140)  50 s post-lock   1.04-1.07, 6 evals then gone
#
# The SCORE MARGIN does not separate them at all — the collapse's 1.053 sits
# inside the legitimate correction's 1.01-1.08 band, so any margin gate that
# blocks the collapse also blocks a real correction.  A margin gate was built,
# measured, and deleted.  Two things do separate them, and both are used:
# HOW LONG the tracker has held the current octave (1.5 s vs 50 s) and HOW
# PERSISTENT the challenger is (82 consecutive evaluations vs 6).
_OCTAVE_TOL_OCT = 0.085    # |log2(new/old)| this close to a whole octave
OCTAVE_SETTLE_S = 8.0      # locked AND on one octave this long => settled
OCTAVE_SWITCH_EVALS = 16   # a settled octave yields only to this many
                           # consecutive winning evaluations (~8s, a phrase);
                           # an unsettled one still yields in 2 (unchanged).
_OCTAVE_CONF_DROP = 0.5    # re-reading a SETTLED metre is a loss of certainty

# --- relock bias (v2.6) -----------------------------------------------------
# Octave stickiness above is gated on ``self._locked``, so a lock LOSS threw
# away everything the tracker knew about the metre: the next evaluation was a
# cold reading.  Measured on a live acoustic take
# (``takes5/decisions.ndjson``): lock lost while comping at 82 bpm, re-entry
# at 118; lost at 90, re-entry at 184.  The mind then comps on a grid at the
# wrong multiple of the human's tempo — the same downstream damage as the v1.3
# mid-take collapse, reached through the unlocked door.
#
# Sized from the measurement, not from taste.  Instrumenting the sweep across
# a relock (quarter-note texture at 80, a stop, an eighth-note texture at the
# SAME 80) the double-tempo alias beats the truth by 1.99x at the first
# evaluation after the resume and by 1.06x once the ACF window has filled.
# That 1.99x is what the bias has to overcome.  At ``_RELOCK_SIGMA_OCT`` 0.35
# an octave-away rival is handicapped 59x at full strength and 55x -> 40x
# across the ten seconds that fixture spends between resume and relock —
# far more than enough, and the width is set by the OTHER end of the trade:
# 0.5 was tried first and cost test_acoustic's P2 guard (see RELOCK_REF_BEATS
# for the mechanism), while 0.35 restores it and improves its off-grid
# fraction 0.11 -> 0.06.  A genuinely new tempo is handicapped too (a real
# count-off at 120 against a lost 80: 4.04x) and still wins, because the
# evidence for it has no 80-periodicity in it at all.  That case is a test,
# not an argument: test_relock_bias_yields_to_a_genuinely_new_tempo.
RELOCK_BREAK_BARS = 8.0    # a rest of up to this many bars is a break INSIDE
# one performance; past it, the next thing played is a new performance and the
# tracker reads it cold.  Measured in bars of the tempo that was lost, so the
# window is a phrase at any tempo rather than a magic number of seconds.
RELOCK_HOLD_BARS = 32.0    # once the human plays again the bias fades over
# this many bars of PLAYING, so it can never become a permanent clamp — but it
# is a section of a tune, not a phrase, because the live failure is long.
# Swept on the full 944s takes5 acoustic take with tests/relock_replay.py,
# scoring how far the reported tempo sits from the tempo held at the last lock
# while UNLOCKED — which is the number the mind actually enters on, since
# MindParams.entry_conf gates on confidence and not on lock:
#
#     bias off   median wander 0.279 oct   within +/-11%  35.8%
#     8 bars     ................ 0.255 ...............   34.9%
#     16 bars    ................ 0.229 ...............   36.1%
#     32 bars    ................ 0.217 ...............   38.9%
#     64 bars    ................ 0.206 ...............   39.9%
#
# 32 takes most of the available improvement; 64 buys one more point for a
# memory that outlives two minutes of playing.  HONEST SCOPE: the same replay
# shows the two lock LOSSES in that take relocking at 0.88x and 1.22x of the
# lost tempo either way, and the locked fraction unmoved at 13.7% — this bias
# steadies the tempo the mind reads between locks, it does not make sustained
# acoustic guitar lockable.  That is a separate defect.
_RELOCK_SIGMA_OCT = 0.35   # log-normal width, in octaves, around the lost tempo
RELOCK_REF_BEATS = 32      # the remembered tempo is the MEDIAN over this many
# CONFIDENT locked beats (8 bars in 4/4), never the reading at the instant of
# the loss.  By that instant the tempo is usually already gone: on the quiet
# 32-bar lo-fi render the tracker holds 80.11 locked at conf 0.74 for forty
# seconds, then slides 127.84 (conf 0.55) -> 181.45 (conf 0.20) over four
# seconds and only THEN drops the lock.  A bias anchored on 181.45 pinned the
# outro at 167.91 against a truth of 80 — it turned a wobble the tracker used
# to recover from into a permanent collapse.  The median over the last eight
# bars of CONFIDENT playing reads 80, because the collapse never earned
# confidence.  Pinned by test_acoustic's P2 guard, which caught this.
# ===========================================================================
# ACOUSTIC MODE — v1.5.1 (SPEC-V1 section 2).  OPT-IN, AND WHY.
# ===========================================================================
# v1.5 replaced the tracker's envelope and onset source outright: instead of
# ``Frame.flux`` + ``Frame.onset`` it read ``Frame.attack`` (features.ATTACK_*)
# and picked its own events off it.  That change is real and it works — on
# sustained open-tuned acoustic guitar, which is the material that produced
# Juniper's live take (167.9/163.0/169.2/165.4/80.6/80.3/121.0 bpm at
# confidence 0.41-0.62, and "it only plays like two random measures here and
# there").  It also REGRESSED THE PROGRAMMED CORPUS, which is what every other
# test and both flagship demos are built on, so it is now a switch and the
# corpus keeps the default.
#
# THE MEASURED A/B (2026-08-19, both corpora, same harness, one run each; the
# script is a plain FeatureExtractor + BeatTracker over the whole take).
# "bpm" is the MEDIAN tracked tempo after the case's settle time, "conf" its
# median confidence, "lock" the fraction of that window spent locked, "t_lock"
# the first lock.  Two numbers per cell: DEFAULT(flux) / ACOUSTIC(attack).
#
#   case              on/s    bpm  flux/attack    conf      lock       t_lock
#   ----------------  -----  ------------------  -------  ---------  ---------
#   click120           2.00   120.0 /  120.0     .95/.97  1.00/1.00   6.1/ 6.6
#   click120_quiet     2.00   120.0 /  120.0     .94/.97  1.00/1.00   6.1/ 6.6
#   click60            1.00    60.0 /   60.0     .92/.94  1.00/1.00  22.1/24.1
#   lofi80            10.04    80.0 /   79.8     .78/.73  1.00/1.00  14.3/14.9
#   trap140           10.99   138.9 /  139.7     .73/.85  1.00/1.00   3.8/ 6.6
#   synth100          11.63   100.0 /  100.1     .83/.82  1.00/1.00   4.2/ 4.6
#   lofi80_q16         2.81    80.0 /   79.8     .72/.73  0.48/1.00  35.3/14.9
#   lofi80_q32         2.61    80.0 /   80.0     .71/.74  0.56/1.00  36.1/15.6
#   trap140_q32        8.15   138.9 /  140.1     .71/.83  1.00/1.00   3.8/ 7.2
#   synth100_q32       7.01   100.0 /  100.0     .80/.75  0.76/1.00   4.6/ 3.7
#   fx_steady          1.88    96.2 /   96.2     .95/.93  1.00/1.00   7.6/ 7.6
#   fx_drift           1.92    99.6 /   99.6     .95/.91  1.00/1.00   7.6/ 7.6
#   fx_stops           1.70    96.2 /   96.2     .95/.92  1.00/1.00   7.6/ 7.6
#   fx_busy_middle     2.35    96.2 /   96.2     .91/.91  1.00/1.00   7.6/ 7.6
#   fx_high_register   1.73    96.2 /   96.2     .95/.95  1.00/1.00   7.6/ 7.0
#   fx_rubato_intro    1.79    96.2 /   96.2     .95/.92  1.00/1.00  19.3/22.2
#   jn_on_grid         2.34   120.0 /  120.0     .96/.95  1.00/1.00   6.1/ 7.0
#   jn_rushing         2.25   120.0 /  120.0     .96/.95  1.00/1.00   6.0/ 6.5
#   jn_dragging        2.25   120.0 /  120.0     .96/.95  1.00/1.00   6.1/ 6.1
#   jn_takes_over      5.03   120.0 /  120.0     .92/.91  1.00/1.00   6.1/ 7.0
#   ac_fingerpick      3.14   167.9 /   84.1     .83/.80  1.00/1.00   4.7/ 9.4
#   ac_strum           1.12    72.0 /   72.0     .54/.82  0.99/1.00  24.3/12.4
#   ac_drone           0.80   never locks either way; conf .11/.28, no lock
#
# The ``on/s`` column is single because it is IDENTICAL in both modes: the mode
# switches the TRACKER's envelope and onset source, and ``Frame.onset`` — the
# density signal ``mind`` and ``roles`` are calibrated against — is picked
# from ``flux`` either way (features.py has the measurement that forced that
# split).  The suffix ``_q16`` / ``_q32`` means the same render 20 dB down at
# 16 and 32 bars.
#
# WHAT ACOUSTIC MODE BUYS.  ``ac_fingerpick`` is the whole reason it exists:
# the default reads 167.9 bpm against a true 84 — the live session's exact
# number, held for the entire take at confidence 0.83, i.e. confidently wrong.
# Acoustic mode reads 84.1.  ``ac_strum``'s confidence comes off the sideman's
# gates (median 0.54, with 100% of the settled take inside [0.40, 0.65],
# against 0.82 and 0%).  It also fixes the GAIN axis on programmed material,
# which is the one place its level-independence helps the corpus: ``lofi80``
# 20 dB down locks in 14.9s and holds 100% of the take instead of 35.3s and
# 48%.
#
# WHAT IT COSTS, AND WHY THE CORPUS KEEPS THE DEFAULT.  The decisive number is
# not in the table above, because the table reports medians and the fault is in
# the TAIL.  On the corpus's own quiet case — ``lofi80_quiet``, a 32-bar lo-fi
# render at -20 dBFS, the exact material tests/test_real_material.py runs —
# the FINAL tracked tempo is:
#
#       default (flux)      80.36 bpm   (truth 80.0)
#       acoustic (attack)   92.98 bpm   (truth 80.0, ratio 1.162)
#
# 92.98 is not an octave, a dotted alias or a subdivision: it is an UNRELATED
# tempo, arrived at over the thinning outro of a long quiet take, and it is
# what the status line and the sim manifest's ``final_bpm`` report to the user.
# That single number is why acoustic mode is not the default — a front end
# that ends a flagship render on a fabricated tempo is worse than one that is
# slow to lock.  (Neither mode is immune to long quiet outros: the same
# measurement on ``synth100`` at 32 bars and -20 dB ends at 144.2 on flux and
# 65.0 on attack.  Only ``lofi80_quiet`` is in the corpus, and only flux keeps
# it.)  Smaller costs, all accepted: ``trap140`` takes 6.6s to lock instead of
# 3.8s, ``click60`` 24.1s instead of 22.1s, ``fx_rubato_intro`` 22.2s instead
# of 19.3s, ``synth100`` gains two brief excursions to dotted aliases
# (133.2 = 4/3, 66.7 = 2/3) that it does not have on flux, and the fixture
# humans lose 0.02-0.04 of confidence while staying far above the 0.55 lock
# gate.
#
# ===========================================================================
# 2026-08-28 (v2.7): THE DECISIVE NUMBER ABOVE NO LONGER HOLDS.  Read the
# paragraph before this one as the state of the world on 2026-08-19; it was
# measured correctly and it has since been fixed out from under.
#
# ``lofi80_quiet`` ending on 92.98 was amendment 26's blind spot, not a
# property of the attack front end: 92.98 / 80.0 is 1.162, and the outro's 4/3
# alias took the ordinary two-evaluation path because it was not an octave.
# Amendment 48 sends every out-of-drift re-read of a SETTLED grid through the
# commitment gate, and the same case now ends at 80.36 in both modes.  (At
# FULL gain the old routing was worse still and nobody had measured it: attack
# ended that render on 181.5 bpm.)  Re-measured, both routings, both modes —
# median bpm after 20 s / END bpm / median conf / locked % / t_lock:
#
#   case            v1.3 attack              v2.7 attack              v2.7 flux
#   lofi80          79.8/ 79.8/.73/100/14.8  79.8/ 79.8/.73/100/14.8  80.0/ 80.1/.78/100/14.3
#   lofi80_q20      79.8/ 79.8/.73/100/14.8  79.8/ 79.8/.73/100/14.8  80.0/ 79.8/.72/ 48/35.3
#   trap140        139.7/139.5/.85/100/ 6.7 139.7/139.5/.85/100/ 6.7 138.9/138.9/.73/100/ 3.8
#   synth100       100.1/100.1/.82/100/ 4.6 100.1/100.1/.83/100/ 4.6 100.0/100.0/.83/100/ 4.2
#   lofi80_32       80.0/181.5/.74/100/15.6  79.9/ 80.4/.74/100/15.6  80.0/ 80.4/.77/100/13.7
#   lofi80_32_q20   80.0/ 93.0/.74/100/15.6  79.9/ 80.4/.74/100/15.6  80.0/ 80.4/.71/ 59/36.1
#
# WHAT IS LEFT OF THE REASON FLUX IS THE DEFAULT: trap140 locking at 6.7s
# instead of 3.8s, and ~0.05 of median confidence on lofi80.  That is all.  On
# the quiet cases attack now wins outright (100% locked at ~15s against 48-59%
# at ~35s).  v2.7 does NOT change the default — that is a decision about every
# test in this file and both flagship demos, and it is not this amendment's to
# make — but the case for the current default is one late lock, and a future
# reader should know that rather than inherit the 92.98 above as still true.
# ===========================================================================
#
# The A/B is pinned in tests/test_acoustic.py: the acoustic-material tests run
# with ``acoustic=True``, the NON-REGRESSION half runs on the DEFAULT and now
# covers the corpus's own material at the corpus's own gain AND length, and
# ``test_the_mode_flag_is_load_bearing_in_both_directions`` asserts each mode
# fails where the other succeeds — so neither half can quietly become a no-op.
#
# Selecting it: ``BeatTracker(acoustic=True)``, or ``PLUME_ACOUSTIC=1`` in the
# environment for the live paths (``realtime``/``cli``) that construct the
# tracker themselves and have no flag for it yet.  ``acoustic=False`` always
# wins over the environment, so a test never depends on the caller's shell.
#
# What the flag does NOT change: ``features`` computes ``Frame.attack`` and
# ``Frame.attack_breadth`` either way (they are additive fields), and
# ``features.CountInDetector`` reads them either way.  THE COUNT-IN HATCH IS
# NOT GATED — it is the pragmatic escape for exactly the material that defeats
# the tracker, and it must work whatever envelope the tracker is running.
ACOUSTIC_ENV = "PLUME_ACOUSTIC"
# v1.5.2 — the same escape hatch for the COUNT-IN (see ``tap()`` and
# ``BeatTracker.__init__``'s ``count_in``): the live paths construct
# ``BeatTracker()`` with defaults and had no flag to arm the detector with,
# exactly the situation ACOUSTIC_ENV already solves for the front end.
COUNT_IN_ENV = "PLUME_COUNT_IN"


def _env_flag(name: str) -> bool:
    """Truthy environment flag, read once per construction, never per frame —
    a tracker's behaviour is fixed for its whole life and a take stays a pure
    function of (frames, taps, modes chosen at construction)."""
    v = os.environ.get(name, "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _acoustic_from_env() -> bool:
    """Is acoustic mode requested by the environment?  Default: no."""
    return _env_flag(ACOUSTIC_ENV)


# --- the tracker's own onset picker (acoustic mode only) --------------------
# In acoustic mode the tracker picks its own events off ``Frame.attack``
# instead of reading ``Frame.onset``.  features.py explains why ``Frame.onset``
# could not simply be switched over even there: ``mind`` and ``roles`` read
# onset DENSITY off the same flag and their thresholds are calibrated to the
# flux picker's numbers.  What the tracker needs is different in kind — not
# "how much is happening" but "where are the attacks" — and on sustained
# acoustic material the flux picker cannot answer it.  Measured on a strummed
# open-tuned fixture with one strum per beat: the flux picker finds the right
# NUMBER of events (0.93 per beat against a true 1.0) but the wrong ones, so
# only 5 of 8 predicted beats were matched (hit rate 0.625) and confidence sat
# at 0.53 — inside MindParams.entry_conf (0.45) and lock_conf (0.55) for 80%
# of the take.  With the attack picker the hit rate is 1.00 and confidence is
# 0.83.  That single number is the "it plays two measures and stops" complaint.
#
# Thresholds are RELATIVE (a multiple of the trailing median) because ``attack``
# is a log ratio: the same constants hold at any input gain, which is what
# lets the same tracker work on a -21 dBFS click and the same click 20 dB down.
BEAT_ATTACK_MULT = 2.6        # multiple of the trailing median
BEAT_ATTACK_FLOOR = 0.010     # absolute floor, log-ratio units
BEAT_ATTACK_MEDIAN_S = 0.35   # trailing median window
BEAT_ATTACK_MIN_SEP_S = 0.060 # minimum separation between picked attacks
# smoothing kernel for the flux envelope before autocorrelation (~75ms):
# keeps period salience meaningful under natural onset jitter.
_ACF_SMOOTH = np.hanning(9)[1:-1]
_ACF_SMOOTH = _ACF_SMOOTH / _ACF_SMOOTH.sum()


def _is_octave_jump(bpm_old: float, bpm_new: float) -> bool:
    """Is ``bpm_new`` a whole-octave relative (x2, /2, x4, ...) of ``bpm_old``?

    Measured in octaves rather than as a ratio test against 0.5/2.0 so that
    the tolerance is symmetric in both directions and covers the double
    octave.  ``_OCTAVE_TOL_OCT`` = 0.085 octaves is about +/-6%, comfortably
    wider than the ~3% drift band an in-octave correction lives in.
    """
    if bpm_old <= 0.0 or bpm_new <= 0.0:
        return False
    d = abs(math.log2(bpm_new / bpm_old))
    return round(d) >= 1 and abs(d - round(d)) <= _OCTAVE_TOL_OCT


@dataclass(frozen=True)
class BeatState:
    t_s: float
    bpm: float                 # current period hypothesis (0.0 until first estimate)
    confidence: float          # 0..1
    locked: bool               # confidence above lock threshold for >= 2 consecutive bars
    bar: int                   # 0-based bar counter since lock (-1 before lock)
    beat_in_bar: int           # 0..3 (meter fixed 4/4 in v1)
    phase_beats: float         # continuous position in bar [0, 4)
    next_beat_t_s: float       # PREDICTED next beat time (math.inf before estimate)
    next_downbeat_t_s: float   # PREDICTED next bar start (math.inf before lock)
    beat_interval_s: float     # 60/bpm (inf before estimate)
    # v1.3 (additive, defaulted — every existing construction is keyword-only).
    # MONOTONE COUNT of committed octave re-interpretations of the metre.  A
    # counter, not an edge flag: downstream polls the state at whatever cadence
    # it likes, and an increment between two polls is unmissable where a
    # self-clearing boolean would be dropped.  ``BeatTracker.events`` carries
    # the full record (t_s, from_bpm, to_bpm, confidence before/after).
    tempo_octave_changes: int = 0
    # v2.7 (additive, defaulted).  MONOTONE COUNT of committed re-reads of the
    # metre of ANY ratio -- the octave ones above plus every dotted/odd alias
    # (4/3, 3/2, 9/8, 5/4).  On live acoustic those are 96% of the total, so
    # ``tempo_octave_changes`` alone is not a usable "did the grid move" signal.
    tempo_grid_changes: int = 0


class BeatTracker:
    """Causal beat tracker over :class:`~plume.listen.features.Frame` streams."""

    def __init__(self, *, tightness: float = 0.35,
                 bpm_range: tuple[float, float] = (60.0, 200.0),
                 lock_conf: float = 0.55, drift_pct: float = 3.0,
                 bpm_hint: float | None = None,
                 acoustic: bool | None = None,
                 count_in: bool | None = None):
        """``bpm_hint`` (v1.2, additive) states a KNOWN tempo without claiming
        a phase.  It narrows the log-normal tempo prior around the hint
        (``_HINT_SIGMA_OCT``, about +/-10%), so an octave or dotted alias
        cannot win, while the sweep still tracks real drift.  Unlike ``tap()``
        it needs no phase guess — which is exactly why taps deadlock on
        pre-recorded material the user did not cut on a downbeat.

        ``acoustic`` (v1.5.1, additive) selects the sustained-acoustic front
        end: the envelope and the onset source become ``Frame.attack`` instead
        of ``Frame.flux``/``Frame.onset``.  ``None`` (the default) reads
        ``PLUME_ACOUSTIC`` from the environment and is OFF unless it is set;
        an explicit ``True``/``False`` ignores the environment entirely.  See
        the ACOUSTIC MODE block above for the corpus A/B that put the default
        here.

        ``count_in`` (v1.5.2, additive) arms a
        :class:`~plume.listen.features.CountInDetector` inside the tracker:
        four prominent, broadband, evenly spaced knocks in the frame stream
        reach ``tap()`` with ``beat_in_bar`` 0..3, so the last count is beat 4
        of the bar and the band comes in on the next downbeat.  ``None`` (the
        default) reads ``PLUME_COUNT_IN`` from the environment and is OFF
        unless set — the same escape the live paths use for ``acoustic``,
        because they construct the tracker with defaults.  The detector reads
        ``Frame.attack``/``attack_breadth`` whatever front end is selected
        (the hatch must work in either mode), and duck-typed frame stubs
        without those fields simply never trigger it.
        """
        if not (0.0 < tightness <= 1.0):
            raise ValueError(f"tightness must be in (0, 1], got {tightness}")
        if not (0.0 < bpm_range[0] < bpm_range[1]):
            raise ValueError(f"bad bpm_range {bpm_range}")
        if bpm_hint is not None and not (0.0 < float(bpm_hint) < 1000.0):
            raise ValueError(f"bpm_hint must be positive, got {bpm_hint}")
        self._tightness = float(tightness)
        self._bpm_lo, self._bpm_hi = float(bpm_range[0]), float(bpm_range[1])
        self._lock_conf = float(lock_conf)
        self._drift_pct = float(drift_pct)
        self._bpm_hint = None if bpm_hint is None else float(bpm_hint)
        self._acoustic = (_acoustic_from_env() if acoustic is None
                          else bool(acoustic))
        self._count_in = (_env_flag(COUNT_IN_ENV) if count_in is None
                          else bool(count_in))
        # The detector is only imported (and only exists) when armed, so the
        # tracker stays runtime-decoupled from ``features`` for every other
        # construction — test stubs keep working with no ``attack`` fields.
        self._countin = None
        if self._count_in:
            from .features import CountInDetector  # noqa: PLC0415
            self._countin = CountInDetector()
        self._prior_center = (_PRIOR_CENTER_BPM if self._bpm_hint is None
                              else self._bpm_hint)
        self._prior_sigma = (_PRIOR_SIGMA_OCT if self._bpm_hint is None
                             else _HINT_SIGMA_OCT)

        # stream bookkeeping
        self._t = 0.0                       # time of last processed frame
        self._t_first: float | None = None
        self._frame_dt: float | None = None
        self._prev_frame_t: float | None = None
        self._frames_since_eval = 0

        # flux envelope + onsets over the trailing ACF window
        self._flux: list[float] = []
        self._onsets: deque[tuple[float, float]] = deque()   # (t_s, strength)
        # ... and the last onset time OUTSIDE that window: the deque is
        # trimmed to ACF_WINDOW_S, so it cannot answer "how long has the
        # human been silent" for any span longer than 12s.
        self._last_onset_t: float | None = None

        # tempo (period) hypothesis
        self._bpm = 0.0
        self._interval = math.inf
        self._salience = 0.0
        self._pending_bpm: float | None = None
        self._pending_n = 0
        # octave stickiness (v1.3): a SEPARATE pending slot, so the slow
        # octave clock and the fast within-octave clock cannot reset each
        # other — sharing one slot let an octave candidate inherit an
        # unrelated jump's accumulated count.
        self._oct_pending_bpm: float | None = None
        self._oct_pending_n = 0
        self._octave_changes = 0
        self._grid_changes = 0
        self._events: list[dict] = []
        # when the CURRENT grid was adopted (any change outside the drift
        # band restarts it) and when the tracker last locked — together these
        # decide whether the octave has settled.  See _octave_is_settled.
        self._grid_t0: float | None = None
        self._lock_t: float | None = None
        # relock bias (v2.6): the metre the tracker was playing when the lock
        # was lost, when it was lost, and when audible material came back.
        self._relock_bpm: float | None = None
        self._relock_t: float | None = None
        self._relock_resume_t: float | None = None
        self._relock_hist: deque[float] = deque(maxlen=RELOCK_REF_BEATS)

        # phase (prediction)
        self._next_beat = math.inf
        self._last_beat_t = math.inf        # most recently emitted predicted beat
        self._beat_count = -1               # index of most recently emitted beat
        self._next_hit = False              # early-onset match for the upcoming beat
        self._next_accent = 0.0
        # per-emitted-beat evidence: {"idx", "hit", "accent", "chroma_change"}
        self._records: deque[dict] = deque(maxlen=2 * DOWNBEAT_WINDOW)

        # chroma accumulation between beats (for downbeat harmonic-change votes)
        self._chroma_sum = np.zeros(12, dtype=np.float64)
        self._chroma_n = 0
        self._prev_beat_chroma: np.ndarray | None = None

        # confidence / lock
        self._conf = 0.0
        self._conf_seeded = False
        self._span_max_rms_db = -math.inf   # loudest frame since the last beat
        self._silent_beats = 0              # consecutive silent-span beats
        self._above = 0
        self._below = 0
        self._locked = False
        self._bar = -1
        self._bib = 0

        # downbeat hypothesis (anchor a: beats with idx % 4 == a are beat 0)
        self._anchor: int | None = None     # committed
        self._cand = -1
        self._cand_n = 0
        self._override = -1
        self._override_n = 0

        # the tracker's own attack picker (BEAT_ATTACK_* above), acoustic mode
        # only.  Kept here and not in FeatureExtractor so that Frame.onset —
        # which mind and roles read as density — is untouched.
        self._att_hist: deque[float] = deque()
        self._att_med_n = 0
        self._att_last_t = -math.inf
        self._has_attack = False        # acoustic AND the stream carries it

        # taps
        self._taps: list[float] = []
        # count-in anchor: after tap() hard-sets the tempo, the tracker "only
        # corrects drift" (SPEC-V1 section 2) — ACF re-estimates outside the
        # drift band are ignored until the tracker genuinely unlocks, so a
        # subdivision alias in the audio cannot overthrow the human's count-in.
        self._tap_anchored = False
        self._tap_bpm: float | None = None   # the tapped tempo itself

    # ---------------------------------------------------------------- events

    @property
    def events(self) -> tuple[dict, ...]:
        """Explicit, ordered record of things the tracker DID to itself.

        v1.3 (additive).  Today it carries ``tempo_octave_change`` only.  The
        rule it exists to enforce: a tracker that has locked and is being
        played along with may re-interpret the metre, but it may never do so
        invisibly — the decision log, the sideman, and the human debugging a
        take all need to see the moment the grid stopped meaning what it meant.
        """
        return tuple(dict(e) for e in self._events)

    # ------------------------------------------------------------------ taps

    def tap(self, t_s: float, *, beat_in_bar: int = 0) -> None:
        """Count-in tap.  >=4 roughly even taps hard-set tempo + bar phase.

        ``beat_in_bar`` (v1.5, additive, default 0 = the pre-v1.5 behaviour
        exactly) states which beat of the bar THIS tap is.  It exists because
        the natural count is "1 2 3 4" and then the band plays: with the
        default the fourth count becomes beat 1 of bar 0, so the downbeat the
        human is counting towards lands a FULL BAR after they stopped counting.
        Passing ``beat_in_bar=3`` on the last tap makes the next beat the
        downbeat, which is what counting a band in means.  See
        ``features.CountInDetector`` for the microphone path.

        **v1.2 AMENDMENT (SPEC-V1 section 2) — the TAP LATCH.**  The spec says
        a count-in hard-sets tempo and phase with confidence 1 and that the
        tracker "afterwards only corrects drift".  Only the PERIOD was
        protected: ``_evaluate_tempo`` overwrote ``_salience`` unconditionally
        about 10ms later (traced: ``TAP t=3.00 conf=1.000 sal=1.000`` then
        ``eval t=3.01 SAL 1.000->0.430``), confidence then EMA-decayed
        1.00 -> 0.59 over 11s, and confidence is exactly what gates lock and
        what the sideman's entry gate reads.  Worse, the one-bar unlock cleared
        ``_tap_anchored`` permanently, so a count-in that dipped once was gone
        for good and the ACF alias took over with nothing left to veto it.
        Three latches now hold while a count-in is anchored: salience is not
        overwritten, confidence is floored at ``_TAP_CONF_FLOOR``, and unlock
        no longer discards the anchor (only a new ``tap()`` replaces it).
        """
        t_s = float(t_s)
        beat_in_bar = int(beat_in_bar) % 4
        if self._taps and t_s <= self._taps[-1]:
            return                                    # non-monotonic tap: ignore
        self._taps.append(t_s)
        if len(self._taps) < TAP_MIN:
            return
        iois = np.diff(np.asarray(self._taps[-8:], dtype=np.float64))
        med = float(np.median(iois))
        if not (0.2 <= med <= 2.0):                   # 30..300 bpm sanity
            return
        if not bool(np.all(np.abs(iois - med) <= TAP_EVEN_FRAC * med)):
            return                                    # not roughly even: no lock
        # hard-set period + phase: last tap = beat ``beat_in_bar`` of bar 0
        # (default 0 — the pre-v1.5 behaviour)
        self._bpm = 60.0 / med
        self._interval = med
        self._salience = 1.0
        self._conf = 1.0
        self._conf_seeded = True
        self._pending_bpm = None
        self._pending_n = 0
        self._oct_pending_bpm = None
        self._oct_pending_n = 0
        self._grid_t0 = t_s                 # v1.3: a count-in IS a new grid
        self._lock_t = t_s
        self._last_beat_t = t_s
        self._next_beat = t_s + med
        self._beat_count = 0
        # _bib is (beat_count - anchor) % 4, and beat_count is 0 here, so the
        # anchor that makes THIS tap land on beat ``beat_in_bar`` is -bib % 4.
        self._anchor = (-beat_in_bar) % 4
        self._cand, self._cand_n = self._anchor, DOWNBEAT_COMMIT_BEATS
        self._override, self._override_n = -1, 0
        self._bib = beat_in_bar
        self._bar = 0
        self._locked = True
        self._above = LOCK_BEATS
        self._below = 0
        self._next_hit = False
        self._next_accent = 0.0
        self._records.clear()
        self._records.append(
            {"idx": 0, "hit": True, "accent": 1.0, "chroma_change": 0.0})
        self._tap_anchored = True
        self._tap_bpm = self._bpm
        # v2.6: a fresh count-in IS the human saying what the tempo is now.
        # It supersedes whatever grid a previous lock lost.
        self._relock_bpm = None
        self._relock_t = None
        self._relock_resume_t = None
        self._t = max(self._t, t_s)

    def _decay_salience(self) -> None:
        """Evidence decay when an evaluation finds nothing.  No-op under a
        count-in latch (see tap())."""
        if not self._tap_anchored:
            self._salience *= 0.8

    # --------------------------------------------------------------- process

    def process(self, frames: list[Frame],
                masked: list[bool] | None = None) -> BeatState:
        """Consume frames; return the tracker state after the last one.

        ``masked`` (v1.5, additive, optional) is a per-frame "the machine is
        drowning the human out here" flag, in lockstep with ``frames`` — the
        second value ``realtime.SelfMasker.mask()`` already returns and
        currently discards.  It exists because of ACOUSTIC MODE: the masker
        suppresses the tracker's onsets by clearing ``Frame.onset``, which
        reaches the default path but NOT the acoustic one, where the tracker
        picks its own onsets off ``Frame.attack``.  Under acoustic mode,
        passing neither this nor ``Frame.self_masked`` leaves the tracker
        taking phase corrections from the machine's own bleed — which does not
        move the grid (the machine plays on the grid the tracker predicted) but
        does inflate the predicted-beat hit rate, i.e. it makes the tracker
        confident by listening to itself.  On the default path the flag is
        merely redundant with the cleared ``onset``.
        """
        for i, f in enumerate(frames):
            self._step(f, bool(masked[i]) if masked is not None
                       and i < len(masked) else False)
        return self._state()

    # ---------------------------------------------------------------- steps

    def _attack_onset(self, t: float, a: float) -> tuple[bool, float]:
        """Causal peak-pick on ``Frame.attack`` (BEAT_ATTACK_* above).

        Threshold = ``BEAT_ATTACK_MULT`` x the trailing median + an absolute
        floor, with a minimum separation.  Relative rather than absolute
        because ``attack`` is a log ratio and therefore gain-independent; the
        floor only exists to stop a near-silent stream, whose median is ~0,
        from calling its own noise an attack.
        """
        # sized once the true frame rate is known (frame 2), not from the
        # 48k/512 default — a stream at another rate would otherwise keep a
        # median window of the wrong duration for the whole session.
        if not self._att_med_n and self._frame_dt is not None:
            self._att_med_n = max(
                1, int(round(BEAT_ATTACK_MEDIAN_S / self._frame_dt)))
            self._att_hist = deque(self._att_hist, maxlen=self._att_med_n)
        med = float(np.median(self._att_hist)) if self._att_hist else 0.0
        thr = BEAT_ATTACK_MULT * med + BEAT_ATTACK_FLOOR
        hit = a > thr and (t - self._att_last_t) >= BEAT_ATTACK_MIN_SEP_S
        if hit:
            self._att_last_t = t
        self._att_hist.append(a)
        return hit, (a if hit else 0.0)

    def _step(self, f, masked_flag: bool = False) -> None:
        t = float(f.t_s)
        if self._t_first is None:
            self._t_first = t
        if self._frame_dt is None and self._prev_frame_t is not None:
            dt = t - self._prev_frame_t
            if 0.0 < dt < 0.1:
                self._frame_dt = dt
        self._prev_frame_t = t
        self._t = t

        # v1.5.2: armed count-in (see __init__).  The detector confirms a
        # count-in on the frame carrying its last knock and returns all its
        # tap times at once; feeding them to tap() with beat_in_bar 0..3 makes
        # the last count beat 4, so the next beat is the downbeat — exactly
        # the wiring features.CountInDetector's docstring specifies.
        if self._countin is not None:
            for i, t_tap in enumerate(self._countin.process((f,))):
                self.tap(t_tap, beat_in_bar=i % 4)

        # onset-envelope window.  DEFAULT: ``Frame.flux``, exactly as v1.4.
        # ACOUSTIC MODE: ``Frame.attack`` (the banded, level-independent attack
        # function — see features.ATTACK_*), because on sustained material flux
        # ranks the SUBDIVISION lag above the beat lag and the tracker then
        # locks an octave high with high confidence.  Measured on a fingerpicked
        # open-tuned fixture: beat-lag ACF minus 8th-lag ACF is -0.030 on flux
        # and +0.060 on attack.  ``flux`` is also the fallback in acoustic mode
        # for duck-typed frame stubs that predate the field.
        attack = getattr(f, "attack", None) if self._acoustic else None
        self._has_attack = attack is not None
        env = float(f.flux) if attack is None else float(attack)
        self._flux.append(env)
        n_keep = int(round(ACF_WINDOW_S / (self._frame_dt or _DEFAULT_FRAME_DT)))
        if len(self._flux) > n_keep:
            del self._flux[: len(self._flux) - n_keep]

        # loudest frame within the current beat span (silence detection)
        rms_db = getattr(f, "rms_db", None)
        if rms_db is not None:
            self._span_max_rms_db = max(self._span_max_rms_db, float(rms_db))

        # chroma accumulation between beats
        chroma = np.asarray(f.chroma, dtype=np.float64)
        if chroma.shape == (12,) and float(chroma.sum()) > 1e-9:
            self._chroma_sum += chroma
            self._chroma_n += 1

        # onsets: bookkeeping + partial phase correction.
        #
        # In ACOUSTIC MODE, when the stream carries ``attack``, the tracker
        # picks its own events off it (see BEAT_ATTACK_* / _attack_onset).
        # Otherwise — the default, and any duck-typed frame stub — it reads
        # ``Frame.onset`` exactly as v1.4 did.
        masked = bool(getattr(f, "self_masked", False)) or bool(masked_flag)
        if self._has_attack:
            hit, strength = self._attack_onset(t, float(env))
        else:
            hit, strength = bool(f.onset), float(f.onset_strength)
        if hit and not masked:
            self._onsets.append((t, strength))
            self._last_onset_t = t
            self._phase_correct(t, strength)
        while self._onsets and self._onsets[0][0] < t - ACF_WINDOW_S:
            self._onsets.popleft()

        # emit predicted beats the stream has passed
        while math.isfinite(self._next_beat) and t >= self._next_beat:
            self._emit_beat()

        # periodic tempo re-estimation
        self._frames_since_eval += 1
        dt = self._frame_dt or _DEFAULT_FRAME_DT
        if self._frames_since_eval >= max(1, int(round(EVAL_PERIOD_S / dt))):
            self._frames_since_eval = 0
            self._evaluate_tempo(t, dt)

    # phase: alpha-correction from onsets near a predicted beat ---------------

    def _phase_correct(self, t: float, strength: float) -> None:
        if not math.isfinite(self._interval) or not math.isfinite(self._next_beat):
            return
        # nearest predicted grid point: the upcoming beat or the one just past
        cands = [self._next_beat]
        if math.isfinite(self._last_beat_t):
            cands.append(self._last_beat_t)
        nearest = min(cands, key=lambda b: abs(t - b))
        err = t - nearest
        if abs(err) > ONSET_TOL_FRAC * self._interval:
            return
        self._next_beat += self._tightness * err          # never snap
        if nearest == cands[0]:                           # early onset: upcoming beat
            self._next_hit = True
            self._next_accent = max(self._next_accent, strength)
        elif self._records:                               # late onset: beat just past
            rec = self._records[-1]
            rec["hit"] = True
            rec["accent"] = max(rec["accent"], strength)

    # beats: emission, confidence, lock, downbeat ----------------------------

    def _emit_beat(self) -> None:
        t_beat = self._next_beat
        self._last_beat_t = t_beat
        self._next_beat = t_beat + self._interval
        self._beat_count += 1

        # chroma snapshot for the span since the previous beat.  A change in
        # this span means the harmony moved AT the previous beat (its onset
        # introduced the new chord), so the vote lands on that beat's record.
        if self._chroma_n > 0:
            mean = self._chroma_sum / self._chroma_n
            s = float(mean.sum())
            if s > 1e-12:
                mean = mean / s
            if self._prev_beat_chroma is not None and self._records:
                change = 0.5 * float(np.abs(mean - self._prev_beat_chroma).sum())
                rec = self._records[-1]
                rec["chroma_change"] = max(rec["chroma_change"], change)
            self._prev_beat_chroma = mean
        self._chroma_sum[:] = 0.0
        self._chroma_n = 0

        # Silence gating: was the span leading into this beat audible at all?
        silent_span = self._span_max_rms_db < SILENT_SPAN_RMS_DB
        self._span_max_rms_db = -math.inf
        self._silent_beats = self._silent_beats + 1 if silent_span else 0

        # confidence: salience x hit rate over the last 8 completed beats.
        # Silent beats are missing data, not misses: they are excluded from the
        # hit-rate window, and while the input is silent the confidence holds
        # for SILENT_HOLD_BEATS (a ride-out) before decaying toward 0.
        if silent_span:
            if self._silent_beats > SILENT_HOLD_BEATS:
                self._conf *= SILENT_DECAY
        elif self._records:
            recent = [r for r in list(self._records)[-HIT_WINDOW_BEATS:]
                      if not r.get("silent")]
            if recent:
                hit_rate = sum(1 for r in recent if r["hit"]) / len(recent)
                instant = max(0.0, min(1.0, self._salience)) * hit_rate
                if self._conf_seeded:
                    self._conf = CONF_EMA * self._conf + (1.0 - CONF_EMA) * instant
                else:
                    self._conf = instant
                    self._conf_seeded = True

        if self._tap_anchored:
            self._conf = max(self._conf, _TAP_CONF_FLOOR)

        self._records.append({
            "idx": self._beat_count,
            "hit": self._next_hit,
            "accent": self._next_accent,
            "chroma_change": 0.0,
            "silent": silent_span,
        })
        self._next_hit = False
        self._next_accent = 0.0
        self._maybe_reacquire_phase()

        self._update_downbeat()

        # advance musical position
        anchor = self._anchor if self._anchor is not None else 0
        self._bib = (self._beat_count - anchor) % 4

        # lock / unlock bookkeeping
        self._above = self._above + 1 if self._conf >= self._lock_conf else 0
        # v2.6: the tempo actually being PLAYED, sampled only while the
        # tracker is both locked and confident — the relock bias reads the
        # median of this, never the reading at the moment the lock broke.
        if self._locked and self._conf >= self._lock_conf and self._bpm > 0.0:
            self._relock_hist.append(self._bpm)
        self._below = self._below + 1 if self._conf < UNLOCK_CONF else 0
        just_locked = False
        if not self._locked and self._above >= LOCK_BEATS:
            self._locked = True
            just_locked = True
            self._lock_t = self._t          # v1.3: octave settle reference
            if self._anchor is None:
                # no committed downbeat evidence yet: anchor at the lock beat
                self._anchor = self._beat_count % 4
                self._bib = 0
            if self._bar < 0:
                self._bar = 0
            # v2.6: the relock bias has done its job (or was never armed).
            self._relock_bpm = None
            self._relock_t = None
            self._relock_resume_t = None
        elif self._locked and self._below >= UNLOCK_BEATS:
            self._locked = False           # bar counter freezes, phase free-runs
            self._above = 0
            # v2.6: remember the metre that WAS being played — the median
            # over the last RELOCK_REF_BEATS confident locked beats, not the
            # reading at this instant, which is routinely already collapsing.
            # Without this memory the tracker re-reads the next thing it hears
            # from scratch and a busier texture at the same tempo relocks on
            # the double.
            self._relock_bpm = (float(np.median(self._relock_hist))
                                if self._relock_hist else None)
            self._relock_t = self._t
            self._relock_resume_t = None
            self._relock_hist.clear()      # the next lock builds its own
            # v1.2: the count-in anchor SURVIVES an unlock.  Clearing it here
            # meant one dip below UNLOCK_CONF destroyed the human's count-in
            # permanently, with no path back short of tapping again — which is
            # precisely how "taps set bpm 80 conf 1.0, then the tracker drifts
            # back off" happened.  Only a new tap() replaces the anchor.
        if self._locked and not just_locked and self._bib == 0:
            self._bar += 1

    def _maybe_reacquire_phase(self) -> None:
        """Re-seed the beat phase when correction provably cannot reach it.

        v1.2 AMENDMENT (SPEC-V1 section 2).  ``_phase_correct`` only accepts
        onsets within +/-``ONSET_TOL_FRAC`` of a predicted beat, so a grid that
        lands (say) a half beat off receives ZERO corrections forever: the
        tempo stays perfect, the hit rate stays at zero, and confidence decays
        to nothing with no path back.  Reproduced deterministically by tapping
        a correct 80bpm count-in offset 0.375s against an 80bpm click (ended
        conf 0.001, unlocked), and hit unprompted on the repo's own
        ``rubato_intro`` fixture.  When ``REACQUIRE_MISSES`` consecutive
        AUDIBLE predicted beats have gone unmatched while onsets are in fact
        arriving, the phase is re-seeded from the most recent onset.  Only
        while unlocked and not tap-anchored: a locked or counted-in grid is
        authoritative and must never be jumped.
        """
        if self._locked or self._tap_anchored:
            return
        if not math.isfinite(self._interval) or not self._onsets:
            return
        recent = [rec for rec in list(self._records)[-REACQUIRE_MISSES:]
                  if not rec.get("silent")]
        if len(recent) < REACQUIRE_MISSES or any(rec["hit"] for rec in recent):
            return
        base = self._onsets[-1][0]
        if base <= self._t - ACF_WINDOW_S:
            return
        k = max(0, int(math.ceil((self._t - base) / self._interval - 1e-9)))
        self._next_beat = base + k * self._interval
        if self._next_beat <= self._t:
            self._next_beat += self._interval
        self._last_beat_t = self._next_beat - self._interval
        self._records.clear()

    def _update_downbeat(self) -> None:
        recent = list(self._records)[-DOWNBEAT_WINDOW:]
        if len(recent) < DOWNBEAT_COMMIT_BEATS:
            return
        max_acc = max(r["accent"] for r in recent)
        scores = np.zeros(4)
        for r in recent:
            acc = r["accent"] / max_acc if max_acc > 1e-12 else 0.0
            scores[r["idx"] % 4] += acc + r["chroma_change"]
        order = np.argsort(scores)
        best = int(order[-1])
        margin_ok = scores[best] > scores[int(order[-2])] + \
            DOWNBEAT_MARGIN * max(scores[best], 1e-12)
        if self._anchor is None:
            if margin_ok and best == self._cand:
                self._cand_n += 1
            elif margin_ok:
                self._cand, self._cand_n = best, 1
            else:
                self._cand_n = 0
            if self._cand_n >= DOWNBEAT_COMMIT_BEATS:
                self._anchor = self._cand
        elif margin_ok and best != self._anchor:
            if best == self._override:
                self._override_n += 1
            else:
                self._override, self._override_n = best, 1
            if self._override_n >= DOWNBEAT_OVERRIDE_BEATS:
                self._anchor = self._override      # 2 bars of contrary evidence
                self._override, self._override_n = -1, 0
        else:
            self._override_n = 0

    # tempo: autocorrelation + half/double resolution + hysteresis -----------

    def _evaluate_tempo(self, now: float, dt: float) -> None:
        """Autocorrelation tempo estimate by FULL LAG SWEEP with a metric prior.

        **v1.2 AMENDMENT (SPEC-V1 section 2).**  The previous version proposed
        candidates only from at most two ACF peaks and their OCTAVE relatives
        (``for mult in (1.0, 0.5, 2.0)``).  On real programmed music the
        strongest ACF lag is usually a DOTTED (3-unit) grouping — a 2/3, 3/4,
        4/3 or 3/2 relative of the beat, not an octave one — because hat
        patterns group sixteenths 2+1.  Measured: on a fresh 140bpm trap render
        the ACF peak sat at 4/3 of the true beat and the shipped candidate pool
        was {90.7, 181.5, 362.9} — 140bpm was **not reachable at any
        evaluation**.  On a fresh 80bpm lo-fi render the peak sat at 2/3 and
        the truth was reachable at 1 of 4 evaluations.  No amount of
        re-weighting rescues a candidate that is never proposed, so every
        integer lag in the bpm range is now scored directly (~47 lags per
        evaluation at 60-200bpm; measured cost 18ms -> 46ms for 50s of audio,
        0.04% -> 0.09% of realtime).  Also corrected in passing: the lo-fi
        alias is exactly 4/3 and 2x, NOT 80*sqrt(2), and the ACF-peak ratio is
        identical at swing 0.50 / 0.57 / 0.62 — swing is not involved.

        Each candidate lag is scored by four independent terms:

        - **Rectified ACF comb** over lag multiples m=1..4 (``_COMB_WEIGHTS``,
          negatives clipped to 0): a true beat lag has support at its multiples
          even when its own lag anti-correlates (backbeats).  The m=4 (bar)
          weight is now near-primary, because the bar is the cleanest cue in
          real material and it is what separates the true beat from its 3/2 and
          3/4 aliases.
        - **Log-normal tempo prior** around ``_PRIOR_CENTER_BPM``.  This
          replaces the old binary ``_BAND_PREF_OUT`` flag, which could neither
          rank two in-band candidates (80 vs 106.7 vs 160 are all inside
          70-160) nor rescue an out-of-pool truth.
        - **Metric nesting**: the strongest supra-beat lag in ``_NEST_BAND_S``
          should be 4x the candidate beat.  This is what kills the 3/2 alias,
          which the prior alone actively *prefers* (120 over 80).
        - **Continuity**: a small bonus for staying within ``drift_pct`` of the
          incumbent.  The old scoring was memoryless, so a persistent alias
          could flip a 47-second-old correct lock on the final evaluation.

        The old ``density_fit`` term is DELETED: ``_DENSITY_SUBDIVS`` was
        closed under x2 for {1,2}, {2,4} and {3,6}, so ``density_fit(f)`` was
        EXACTLY equal to ``density_fit(2f)`` for onsets/beat in
        {1.5, 2, 2.2, 4, 4.4, 5, 6, 6.7} — including the 6.7 onsets/beat the
        lo-fi mix actually measures.  It contributed nothing to the one
        decision it existed to make.

        Salience is sqrt-compressed evidence: humanized envelopes cap the
        normalized ACF around 0.35-0.5 even when rock-steady, which the linear
        product with hit-rate kept below lock threshold forever.
        """
        # RELOCK BIAS (v2.6): while unlocked shortly after losing a lock, the
        # sweep leans toward the metre it lost.  A log-normal in octaves, not
        # an octave test, because the aliases a relock actually picks are not
        # all octaves — the live take went 82 -> 118, a 3/2.
        #
        # Read FIRST, above every early return in this function.  The rest of
        # it gives up on silence (flat flux, too few onsets) — and silence is
        # exactly when the bias has to expire.  Computing the weight where it
        # is used left the memory alive through any stop long enough to have
        # ended the take, which the long-stop test now pins.
        relock_w = self._relock_weight(now)
        relock_ref = self._relock_bpm

        n = len(self._flux)
        if n < int(round(1.0 / dt)):
            return
        buf = np.asarray(self._flux, dtype=np.float64)
        if float(buf.max(initial=0.0)) <= 1e-9 or len(self._onsets) < 2:
            self._decay_salience()             # no events / flat flux: decay
            return
        if len(self._onsets) < _MIN_EVAL_ONSETS:
            return                             # too sparse to (re-)estimate
        env = np.convolve(buf, _ACF_SMOOTH, mode="same")
        x = env - env.mean()
        r = np.correlate(x, x, mode="full")[n - 1:]
        r0 = float(r[0])
        if r0 <= 1e-12:
            self._decay_salience()
            return

        lag_lo = max(2, int(math.floor(60.0 / (self._bpm_hi * dt))))
        lag_in_hi = min(int(math.ceil(60.0 / (self._bpm_lo * dt))),
                        (n - 1) // 2, len(r) - 2)
        if lag_in_hi <= lag_lo:
            return

        def refine(k: int) -> float:
            denom = r[k - 1] - 2.0 * r[k] + r[k + 1]
            if denom < -1e-12:
                d = float(np.clip(0.5 * (r[k - 1] - r[k + 1]) / denom,
                                  -0.5, 0.5))
                return k + d
            return float(k)

        def score_at(lag: float) -> float:
            if lag < 1.0 or lag > len(r) - 2:
                return -1.0
            i = int(math.floor(lag))
            fr = lag - i
            return float((r[i] * (1.0 - fr) + r[i + 1] * fr) / r0)

        def comb(lag: float) -> float:
            tot = 0.0
            for m, w in enumerate(_COMB_WEIGHTS, start=1):
                s = score_at(lag * m)
                if s < -0.5:            # multiple beyond the window: stop
                    break
                tot += w * max(s, 0.0)
            return tot / sum(_COMB_WEIGHTS)

        # metric nesting reference: the strongest supra-beat (bar-level) lag
        nest_lo = max(lag_lo + 1, int(round(_NEST_BAND_S[0] / dt)))
        nest_hi = min(int(round(_NEST_BAND_S[1] / dt)), (n - 1) // 2, len(r) - 2)
        bar_lag = None
        if nest_hi > nest_lo:
            kb = int(np.argmax(r[nest_lo:nest_hi + 1])) + nest_lo
            # The supra-beat peak only counts as a BAR if it outranks its own
            # half.  Without this gate a plain click train — whose ACF simply
            # decays monotonically across its comb of peaks — hands the
            # 4-grouping bonus to the DOUBLE-tempo candidate, and a true 60bpm
            # click reads as 120.  A genuine bar stands above the half-bar
            # (measured: trap's bar lag +0.67 vs its half-bar +0.09); ACF
            # decay does not (60bpm click: 2s +0.80 vs 1s +0.89).
            if r[kb] > 0.0 and score_at(refine(kb)) > score_at(refine(kb) / 2.0):
                bar_lag = refine(kb)

        def nesting(lag: float) -> float:
            if bar_lag is None or lag <= 0.0:
                return _NEST_NEUTRAL
            ratio = bar_lag / lag
            for target, mult in ((4.0, _NEST_BONUS), (2.0, _NEST_NEUTRAL),
                                 (8.0, _NEST_NEUTRAL)):
                if abs(ratio - target) <= _NEST_TOL * target:
                    return mult
            return _NEST_PENALTY

        def prior(bpm: float) -> float:
            z = math.log2(bpm / self._prior_center) / self._prior_sigma
            return math.exp(-0.5 * z * z)

        def relock(bpm: float) -> float:
            if relock_w <= 0.0 or not relock_ref:
                return 1.0
            z = math.log2(bpm / relock_ref) / _RELOCK_SIGMA_OCT
            return math.exp(-0.5 * relock_w * z * z)

        def evidence(lag: float) -> float:
            """Periodicity a grid of this period EXPLAINS — for CONFIDENCE,
            never for ranking.

            v1.2 took the MAX normalized ACF over the beat lag and its metric
            multiples, on the argument that a grid predicting the bar correctly
            is well evidenced even when the beat-level ACF is weak (the lo-fi
            mix autocorrelates +0.19 at the beat and +0.81 at the bar).

            STILL THE MAX, and ``_COMB_WEIGHTS`` is used here only for its
            LENGTH (the number of metric multiples to look at) — no weight in
            it is applied.  SPEC-V1's v1.3 section records this as a *known
            residual, not fixed*: "the evidence term is `max` over a
            candidate's metric multiples, so a grid and its half share most of
            their support by construction ... salience cannot discriminate f
            from f/2 and is not asked to; the discrimination lives in the
            comb/prior/nesting ranking and in this stickiness rule."

            A previous revision of this docstring announced a v1.5 change to a
            weighted comb, with a measured "0.74 -> 0.40 on that fixture".
            **That change is not in this function and there is no v1.5 section
            in SPEC-V1 for it.**  The docstring was corrected rather than the
            code because the confidence complaint it claimed to fix is in fact
            fixed by the OTHER half of v1.5 — the ``attack`` envelope — and
            that is measurable: on the strummed open-tuned fixture the shipped
            tracker reads median confidence 0.82 with 11% of the take inside
            [0.40, 0.65], and on the pulse-free ``drone`` fixture it reads 0.32
            against ``MindParams.entry_conf`` 0.45 and never locks.  Both are
            pinned in tests/test_acoustic.py.

            Switching to the comb remains a live candidate and is UNMEASURED
            over the corpus; it must not be adopted on the strength of the
            deleted paragraph's number.  Ranking is untouched either way: this
            function is read only after ``_adopt_bpm``.
            """
            return max(max(score_at(lag * m), 0.0)
                       for m in range(1, len(_COMB_WEIGHTS) + 1))

        # FULL LAG SWEEP over the bpm range (see docstring): every candidate
        # beat period is scored on its own evidence, so a dotted/triplet ACF
        # peak can no longer hide the truth from the pool.
        best_total = 0.0
        best_bpm = 0.0
        best_lag = 0.0
        for k in range(lag_lo, lag_in_hi + 1):
            lag_star = refine(k)
            bpm_c = 60.0 / (lag_star * dt)
            if not (self._bpm_lo * 0.98 <= bpm_c <= self._bpm_hi * 1.02):
                continue
            c = comb(lag_star)
            if c <= 0.0:
                continue
            total = c * prior(bpm_c) * nesting(lag_star) * relock(bpm_c)
            if self._bpm > 0.0 and \
                    abs(bpm_c - self._bpm) / self._bpm * 100.0 <= self._drift_pct:
                total *= _CONTINUITY_BONUS
            if total > best_total:
                best_total = total
                best_bpm = bpm_c
                best_lag = lag_star
        if best_bpm <= 0.0:
            return
        self._adopt_bpm(best_bpm, now)
        if not self._tap_anchored:
            # TAP LATCH (see tap()): while a count-in holds, the ACF may still
            # correct DRIFT but must not overwrite the count-in's evidence.
            #
            # v1.3 AMENDMENT (SPEC-V1 section 2) — salience describes THE GRID
            # IN FORCE, not the top-scoring candidate.  It is read AFTER
            # _adopt_bpm precisely so hysteresis has already had its say.
            # Before v1.3 the salience of whichever candidate won the sweep was
            # written straight into confidence even when hysteresis refused to
            # adopt it, so confidence reported the evidence for a grid the
            # tracker was not playing.  Traced on a 140bpm trap take: at
            # t=53.14 the tracker held 138.89 with salience 0.443; at t=53.64
            # the half-tempo rival won the sweep but was NOT adopted, and
            # salience jumped 0.443 -> 0.872 anyway.  That is the mechanism
            # behind "confidence RISES across a tempo collapse" — the rise
            # begins BEFORE the collapse, on evidence for a grid nobody was
            # playing yet.
            lag_now = ((60.0 / (self._bpm * dt)) if self._bpm > 0.0
                       else best_lag)
            self._salience = math.sqrt(max(0.0, min(1.0, evidence(lag_now))))

    def _adopt_bpm(self, bpm_new: float, now: float) -> None:
        if self._bpm <= 0.0:                       # first estimate
            self._set_bpm(bpm_new)
            base = self._onsets[-1][0] if self._onsets else now
            k = max(1, int(math.ceil((now - base) / self._interval - 1e-9)))
            self._next_beat = base + k * self._interval
            if self._next_beat <= now:
                self._next_beat += self._interval
            return
        diff_pct = abs(bpm_new - self._bpm) / self._bpm * 100.0
        if diff_pct <= self._drift_pct:            # tracking, not switching
            if self._tap_bpm is not None and self._tap_anchored:
                # v1.2: "only corrects drift" was implemented as an UNBOUNDED
                # random walk — every evaluation could move the tempo up to
                # drift_pct with no memory of the tapped value, measured as
                # 80.00 tapped -> 78.33 after 30s.  The correction band is
                # relative to the COUNT-IN, not to the last accepted step.
                lo = self._tap_bpm * (1.0 - self._drift_pct / 100.0)
                hi = self._tap_bpm * (1.0 + self._drift_pct / 100.0)
                bpm_new = min(max(bpm_new, lo), hi)
            self._set_bpm(bpm_new)
            self._pending_bpm, self._pending_n = None, 0
            self._oct_pending_bpm, self._oct_pending_n = None, 0
            return
        if self._tap_anchored:
            # Count-in authority: only drift-band corrections are accepted
            # while the tapped lock holds; alias jumps are discarded outright.
            self._pending_bpm, self._pending_n = None, 0
            self._oct_pending_bpm, self._oct_pending_n = None, 0
            return
        if self._locked:
            # v2.7 AMENDMENT (SPEC-V1 section 2).  EVERY out-of-drift re-read
            # of a LOCKED grid goes through the commitment gate, not only the
            # octave ones.  v1.3 routed octaves here and let every other alias
            # -- 4/3, 3/2, 9/8, 5/4 -- through the ordinary 2-evaluation path,
            # reasoning that a dotted alias is what the full lag sweep exists
            # to REACH.  Live acoustic material says the sweep reaches it and
            # then keeps reaching, forever: on ``takes6/`` (370 s of
            # fingerstyle guitar, no count-in) the tracker re-read the grid 54
            # times -- 8.8 a minute -- and only TWO of those were octaves.  The
            # dominant family was 4/3 (x22), then 3/2 (x13), 9/8 (x8), 5/4
            # (x6).  The guard that exists to stop the metre changing under a
            # take was blind to 96% of the cases it exists to stop.
            #
            # The v1.3 reasoning is answered by SETTLEDNESS rather than by
            # ratio: an UNSETTLED grid still yields in 2 evaluations, so the
            # sweep can still correct a dotted alias it picked in its first
            # bars -- which is the only place that correction was ever wanted.
            self._grid_hysteresis(bpm_new, now)
            return
        self._oct_pending_bpm, self._oct_pending_n = None, 0
        # NOTE: the pre-v1.1 immediate "resolve exact half/double toward
        # 70-160" fast path lived here; it could flap-adopt a transient alias
        # the moment the true tempo jittered out of the candidate pool.  The
        # half/double preference (toward 70-160, by onset-density fit) is now
        # applied in _evaluate_tempo's candidate scoring, so an in-band
        # winner arrives persistently and passes through the same 2-eval
        # hysteresis as any other switch.
        self._period_hysteresis(bpm_new)

    def _period_hysteresis(self, bpm_new: float) -> None:
        """The ordinary rule: a big jump must persist >= 2 evaluations.

        This governs the UNLOCKED tracker.  It is a named method rather than
        four lines inline because the v2.7 amendment removed its other caller
        — a locked, out-of-drift re-read — and ``tests/grid_guard_replay.py``
        has to be able to put that caller back to measure what changed.  A
        measurement of "before" that re-implements "before" is measuring its
        own re-implementation.
        """
        if self._pending_bpm is not None and \
                abs(bpm_new - self._pending_bpm) / self._pending_bpm * 100.0 \
                <= self._drift_pct:
            self._pending_n += 1
            if self._pending_n >= 2:
                self._set_bpm(bpm_new)
                self._pending_bpm, self._pending_n = None, 0
        else:
            self._pending_bpm, self._pending_n = bpm_new, 1

    # -------------------------------------------------- relock bias (v2.6)

    def _relock_weight(self, now: float) -> float:
        """How hard the tempo sweep leans on the metre that was just lost.

        1.0 while the human is still playing right after a lock loss, fading
        to 0.0 over ``RELOCK_HOLD_BARS`` of playing; 0.0 outright once they
        have been SILENT for longer than ``RELOCK_BREAK_BARS``, and 0.0
        whenever the tracker is locked (v1.3's octave stickiness is in charge
        there) or anchored to a count-in (``tap()`` is).

        **Silence ends a take; playing does not.**  This distinction is the
        whole rule, and the live evidence is what forced it.  A first cut
        decayed the bias on one clock running from the unlock, which was
        wrong in both directions: it burned its window on the rest, and it
        expired long before the tracker got anywhere.  On
        ``takes5/decisions.ndjson`` the tracker loses lock at t=495 holding
        104 bpm and does not lock again until t=594 — and across those
        ninety-nine seconds the human never stops playing (the mind logs
        ``listening`` with onsets throughout), while the reported tempo
        wanders 59.8 -> 118.4 -> 187.7 -> 91.5.  A guitarist does not silently
        double the tempo mid-tune, so as long as the playing is continuous the
        tracker keeps its opinion.  Put the instrument down for a phrase and
        the next thing played is a new performance, read cold.

        The silence test reads ``_last_onset_t`` and not ``_onsets``: that
        deque is trimmed to ``ACF_WINDOW_S``, so asking it about a rest longer
        than twelve seconds silently returns the wrong answer.
        """
        if self._locked or self._tap_anchored:
            return 0.0
        if self._relock_bpm is None or self._relock_t is None:
            return 0.0
        bar_s = 4.0 * 60.0 / self._relock_bpm
        if self._last_onset_t is None or \
                (now - self._last_onset_t) > RELOCK_BREAK_BARS * bar_s:
            self._relock_bpm = None                # the rest ended the take
            return 0.0
        if self._relock_resume_t is None:
            for t_on, _strength in self._onsets:
                if t_on > self._relock_t:
                    self._relock_resume_t = t_on
                    break
            if self._relock_resume_t is None:
                return 1.0        # nothing new heard yet since the loss
        return max(0.0, 1.0 - (now - self._relock_resume_t)
                   / (RELOCK_HOLD_BARS * bar_s))

    # ------------------------------ grid stickiness (v1.3, widened v2.7)

    def _grid_is_settled(self, now: float) -> bool:
        """Has the tracker committed to this GRID, or is it still reading?

        Settled = LOCKED, and ``OCTAVE_SETTLE_S`` has passed since the later of
        (a) the lock and (b) the last change of grid outside the drift band.
        This is the distinction the measurements in the constants block are
        about: a tracker one bar past its lock is still deciding what it is
        hearing, and correcting the grid there is the feature working
        (measured on plume's own backbeat groove, whose true 80bpm only
        overtakes the 160 alias 1.5s after the lock, and on a busy-from-bar-one
        human at the same tempo).  A tracker fifty seconds into a stable take
        is being played along with, and changing the grid under it is the bug.

        v2.7 note: this predicate never mentioned octaves in its logic, only in
        its name.  It is the COMMITMENT clock, and it now gates re-reads of any
        ratio.
        """
        if not self._locked:
            return False
        t0 = max(self._grid_t0 if self._grid_t0 is not None else -math.inf,
                 self._lock_t if self._lock_t is not None else -math.inf)
        return math.isfinite(t0) and (now - t0) >= OCTAVE_SETTLE_S

    def _grid_hysteresis(self, bpm_new: float, now: float) -> None:
        """A LOCKED grid is re-read only on SUSTAINED evidence, and loudly.

        **v1.3 AMENDMENT (SPEC-V1 section 2).**  Before this, an octave change
        and a 4% period correction were governed by the same rule — "differs by
        more than drift_pct AND persists >= 2 evaluations", i.e. one second.
        Measured on a 32-bar 140bpm trap render: the tracker locked at t=4.1s,
        held 139.8-140.7 for fifty seconds across thirty bar boundaries, and
        then at t=53.6 a half-tempo candidate won two consecutive evaluations
        and the tracked tempo dropped to 69.88 for the rest of the take.  No
        unlock, no event, nothing downstream could see it — and because a
        half-tempo grid predicts HALF AS MANY beats, all of which are a subset
        of the true beats, the predicted-beat hit rate went 0.875 -> 1.000 and
        confidence ROSE (0.694 -> 0.736 -> 0.769) across the collapse.  The
        mind kept comping on a grid at half the human's tempo, which halves
        every onset-density reading it takes and silently breaks the
        inverse-density coupling.

        Musically: a drummer does not halve the tempo at bar 20 because the
        texture thinned.  So the requirement scales with commitment —

        - a SETTLED grid (``_grid_is_settled``) yields only to
          ``OCTAVE_SWITCH_EVALS`` consecutive winning evaluations, about eight
          seconds, a phrase.  One evaluation where the grid in force wins back
          resets the clock, which is exactly what kills the collapse: its
          challenger managed 6 evaluations and then vanished, while both
          measured legitimate corrections ran 82 unbroken.
        - an UNSETTLED one keeps the ordinary 2-evaluation hysteresis, so the
          tracker can still fix a grid it picked in its first bars.

        Either way, a locked re-read is now OBSERVABLE: an event on
        ``BeatTracker.events`` and an increment of ``BeatState``'s
        ``tempo_grid_changes``.  The confidence cut applies only on the
        settled path — re-reading a metre you have played along with for a
        phrase is a loss of certainty, while refining a grid you picked two
        bars ago is the tracker doing its job, and halving confidence there
        would only delay the sideman's entry for being right.

        **v2.7 AMENDMENT (SPEC-V1 section 2) — the ratio was never the
        signal.**  v1.3 reached this method only for whole-octave jumps; every
        other alias took the ordinary 2-evaluation path on the reasoning that
        a dotted alias is what the full lag sweep exists to REACH.  Three live
        acoustic takes falsified that.  On ``takes6/`` — 370 s of fingerstyle
        guitar with no count-in — the locked grid was re-read 54 times (8.8 a
        minute), of which the octave, the only family v1.3 guarded, was TWO.
        The distribution was 4/3 x22, 3/2 x13, 9/8 x8, 5/4 x6, octave x2.  A
        guard covering 4% of its own failure mode is not a guard.

        The v1.3 reasoning survives, relocated: what makes an early correction
        legitimate is that the tracker has not committed yet, not that the
        correction happens to be an octave.  An unsettled grid still yields in
        2 evaluations at any ratio, so the sweep can still climb off a dotted
        alias in its first bars — see
        ``test_an_unsettled_grid_still_corrects_a_dotted_alias_in_two_evals``.

        Honest scope: this stops the metre CHANGING under a settled take.  It
        does not make the first reading right, and it cannot: measured
        independently of the tracker, ``takes6/human.wav`` has an ACF salience
        of only 0.06-0.19 per window against 0.4+ for a clear beat, and single
        30 s windows read 55, 58, 122, 134, 63, 160, 165 against a true 111.7.
        A 12-second window genuinely cannot separate 111 from its 4/3 and 3/2
        relatives.  Longer-horizon integration is the separate defect.

        NOT applied before lock at all.  An unlocked tracker is still deciding
        what it is hearing, and the grid it settles on there is a legitimate
        first reading — including the deliberately-accepted half-time reading
        of trap material, whose beat-level flux ACF is negative (an
        anti-correlated backbeat) while its bar level is strongly positive.
        """
        settled = self._grid_is_settled(now)
        need = OCTAVE_SWITCH_EVALS if settled else 2
        same = (self._oct_pending_bpm is not None
                and abs(bpm_new - self._oct_pending_bpm) / self._oct_pending_bpm
                * 100.0 <= self._drift_pct)
        if same:
            self._oct_pending_n += 1
        else:
            self._oct_pending_bpm, self._oct_pending_n = bpm_new, 1
        if self._oct_pending_n < need:
            return
        old_bpm, old_conf = self._bpm, self._conf
        # Computed against the OLD tempo, before _set_bpm moves it.
        is_octave = _is_octave_jump(old_bpm, bpm_new)
        self._set_bpm(bpm_new)
        self._grid_changes += 1
        if is_octave:
            # v2.7: ``tempo_octave_changes`` keeps its v1.3 meaning exactly --
            # whole-octave re-interpretations only -- because mind.py's ritard
            # guard and sim/cli's forensic lines are written against it.  The
            # wider count is the new, additive ``tempo_grid_changes``.
            self._octave_changes += 1
        self._pending_bpm, self._pending_n = None, 0
        self._oct_pending_bpm, self._oct_pending_n = None, 0
        if settled:
            # Without this the hit rate alone would push confidence UP through
            # the change, because a coarser grid predicts fewer beats and every
            # one of them still lands on a beat the human played.
            self._conf *= _OCTAVE_CONF_DROP
            if self._tap_anchored:
                self._conf = max(self._conf, _TAP_CONF_FLOOR)
        self._events.append({
            "event": ("tempo_octave_change" if is_octave
                      else "tempo_grid_change"),
            "t_s": float(now),
            "from_bpm": float(old_bpm),
            "to_bpm": float(self._bpm),
            "octave": bool(is_octave),
            # The alias family, so a log can be counted by ratio without
            # re-deriving it: 4/3 and 3/2 are what live acoustic produces.
            "ratio": float(self._bpm / old_bpm) if old_bpm > 0.0 else 0.0,
            "settled": bool(settled),
            "conf_before": float(min(max(old_conf, 0.0), 1.0)),
            "conf_after": float(min(max(self._conf, 0.0), 1.0)),
        })

    def _set_bpm(self, bpm: float) -> None:
        """Change the period hypothesis.  Phase (next_beat time) is untouched."""
        old = self._bpm
        self._bpm = float(bpm)
        self._interval = 60.0 / self._bpm
        if old <= 0.0 or abs(self._bpm - old) / old * 100.0 > self._drift_pct:
            # A NEW GRID, not a drift correction — the settle clock restarts.
            # Only jumps outside the drift band count: otherwise the ordinary
            # per-evaluation tracking of a live tempo would reset the clock on
            # every evaluation and nothing would ever settle.
            self._grid_t0 = self._t

    # ---------------------------------------------------------------- state

    def _state(self) -> BeatState:
        interval = self._interval
        bpm = self._bpm
        if math.isfinite(self._next_beat) and math.isfinite(interval):
            frac = 1.0 - (self._next_beat - self._t) / interval
            frac = min(max(frac, 0.0), 1.0 - 1e-9)
            phase = (self._bib + frac) % 4.0
        else:
            phase = float(self._bib)
        if self._locked and math.isfinite(self._next_beat):
            nb_bib = (self._bib + 1) % 4
            next_downbeat = self._next_beat + interval * ((4 - nb_bib) % 4)
        else:
            next_downbeat = math.inf
        return BeatState(
            t_s=self._t,
            bpm=bpm,
            confidence=float(min(max(self._conf, 0.0), 1.0)),
            locked=self._locked,
            bar=self._bar if self._locked or self._bar >= 0 else -1,
            beat_in_bar=int(self._bib),
            phase_beats=float(phase),
            next_beat_t_s=self._next_beat,
            next_downbeat_t_s=next_downbeat,
            beat_interval_s=interval,
            tempo_octave_changes=int(self._octave_changes),
            tempo_grid_changes=int(self._grid_changes),
        )
