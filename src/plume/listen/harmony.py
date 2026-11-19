"""Causal key/chord belief tracking (SPEC-V1 section 3).

Key belief
    A 12-bin pitch-class histogram of incoming chroma with exponential decay
    (half-life ``key_halflife_s``) is Pearson-correlated against all 24 key
    profiles (12 major + 12 minor; Krumhansl-Kessler with a share of corpus
    profiles, and for audio as each would sound with its partials: see
    _KEY_CORPUS_SHARE).  Each key's score is that correlation plus
    ``_KEY_ROOT_WEIGHT`` times the share of the music's chords that sit on its
    tonic (_KEY_ROOT_WEIGHT).  The reported key is
    an *incumbent* with switch hysteresis: a challenger key must beat the
    incumbent's score by ``_KEY_SWITCH_MARGIN`` sustained for
    ``_KEY_SWITCH_SUSTAIN_S`` seconds before the belief flips.  ``key`` is
    ``None`` until enough voiced mass has accumulated and the best correlation
    clears ``_KEY_CONF_FLOOR``; ``key_conf`` is the incumbent's Krumhansl-Kessler
    correlation clipped to [0, 1] and discounted while evidence is thin.
    ``key_best`` / ``key_lead`` report the best score right now, before any
    hysteresis, and how far it leads the belief, in units of the margin the
    belief's hysteresis would ask of it.

Chord belief
    Per-frame cosine scores of the chroma against 24 binary maj/min triad
    templates plus an N (silence) state are turned into likelihoods
    (``exp(beta * score)``) and smoothed by an online 25-state HMM with
    self-transition ``chord_selftrans`` (remaining probability spread
    uniformly) and fixed-lag smoothing (lag ``_HMM_LAG`` frames: the reported
    posterior is the forward-backward posterior of the frame ``_HMM_LAG``
    frames ago).  A chord symbol is emitted only when the winning posterior
    exceeds 0.5 on two consecutive (lagged) frames; between qualifying
    candidates the previously emitted chord is held.  The N state emits
    ``chord=None``.  Symbols are spelled for the believed key via
    ``theory.spell`` ("Bbm" in F minor, "A#m" with no flat-preferring key).

Register estimate — THE APPROXIMATION, documented per spec
    ``Frame`` carries only a 12-bin chroma and a spectral centroid — not the
    full spectrum — so true per-pitch octave information is gone.  We
    approximate the spectral distribution of active pitch content as a
    Gaussian in semitone space centered on an estimated fundamental
    ``centroid_hz / 2`` (harmonic-rich tones place the spectral centroid
    above the fundamental; /2 is a crude but monotone proxy).  Each candidate
    MIDI note m in [24, 108] receives weight
    ``chroma[m % 12] * exp(-((m - mu)^2) / (2 * sigma^2))`` with
    ``sigma = _REG_SIGMA_ST`` semitones, accumulated into a rolling histogram
    with exponential decay (half-life ``_REG_HALFLIFE_S``).
    ``register_low_midi`` / ``register_high_midi`` are the weighted p10/p90 of
    that histogram.  Both are ``None`` while silent (no voiced frame within
    the trailing ``_REG_SILENCE_S`` seconds, or no accumulated mass).  This is
    an octave-placement *estimate*: it is reliable for ordering (low content
    vs high content) and coarse placement, not for exact pitch bounds.

Determinism: pure function of the input frame stream.  No rng, no wall-clock.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..theory import Key, parse_key, spell

if TYPE_CHECKING:  # runtime-independent: frames are duck-typed
    from .features import Frame

# ---------------------------------------------------------------- constants

# Krumhansl-Kessler probe-tone profiles (tonic first).
_KK_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KK_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
# v1.6 AMENDMENT: what each key is matched against.  Two changes, each for a
# kind of wrong key the probe-tone profiles alone gave at band entry.
#
# _CORPUS_MAJOR / _CORPUS_MINOR / _KEY_CORPUS_SHARE.  Corpus profiles
# (Albrecht and Shanahan, 2013: how long each pitch class sounds in pieces of
# known key), tonic first, added to the probe-tone ones at this share (both
# z-scored).  The probe-tone minor profile weights its b6 nearly as high as
# its fifth, so a maj7 home chord (C E G B) fits E minor about as well as C
# major; the corpus profiles keep the b6 and the chromatic notes low.  Alone
# they lean to major on minor material, so they are a share.
#
# _KEY_PARTIAL_DECAY.  Audio chroma is the sound's, partials included: every
# note also puts energy a twelfth above (its 3rd and 6th partials) and a major
# third above (its 5th).  Against bare profiles that leans C major material
# toward G major, and a ringing low string's 5th partial puts a major third
# under a minor melody.  So for audio each profile is matched as the chroma its
# scale would sound as: convolved with partials 1-8 at amplitude
# _KEY_PARTIAL_DECAY ** (n - 1), as Gomez (2006) builds key templates for
# audio.  Lower than Gomez's 0.6 because not every instrument is rich in
# partials.  MIDI input has none, so a tracker made with ``partials=False``
# matches the bare profiles (_PROFILES_BARE).
#
# Confidence (key_conf) and the floors that gate it keep the probe-tone
# profiles alone (_TONAL_PROFILES): they were set on that scale.  The margins
# below that compare two keys are on the scale of the matched profiles.
_CORPUS_MAJOR = np.array(
    [0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081]
)
_CORPUS_MINOR = np.array(
    [0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052]
)
_KEY_CORPUS_SHARE = 0.5
_KEY_PARTIAL_DECAY = 0.4
# _KEY_ROOT_WEIGHT — v1.6.  Which pitch class the chords sit on.  A pitch-class
# histogram cannot tell a key from its neighbour when the music uses notes of
# both: I bVII IV I in D (D C G D) holds exactly G major's notes, and so does a
# I bVII vamp but for one, so both read as the IV's key however the profiles
# are built.  What names D is that the chords sit on D.  So every voiced frame
# is read as a triad (cosine against the 24 major and minor triads, root
# counted _ROOT_TEMPLATE_WEIGHT times, on the same bass-weighted chroma the key
# histogram takes, sharpened by _HMM_BETA as the chord reader is), the reading's
# weight is added to its root, and that root histogram decays with the key's
# half-life.  Its share on a key's tonic, times this weight, is added to the
# key's correlation.  The root term counts a major and a minor triad on one root
# alike: whether a frame's third is major or minor is the partials' and the
# profiles' question (a low string's 5th partial is a major third under any
# melody).  It moves nothing between keys on one tonic, and between a key and
# its relative only as far as the music dwells on one of their tonics, which is
# what a listener hears as home.  Measured on generated strummed I bVII IV I in
# four voicings and 12 keys: the band came in on I in 48 of 48 takes at 0.3, and
# in 31 of 48 with no root term (none of 12 for one voicing).  0.4 came in on
# I no more often, and over 36 I bVII vamps moved the key after the band was in
# 6 times against 4 at 0.3.
_KEY_ROOT_WEIGHT = 0.3
_ROOT_TEMPLATE_WEIGHT = 2.0

_KEY_SWITCH_MARGIN = 0.10     # challenger must beat incumbent score by this...
_KEY_SWITCH_SUSTAIN_S = 3.0   # ...sustained this long, to flip the belief
# v1.6: the margin was 0.05 on correlation alone.  The root term swings with
# the chord of the moment, most in a two-chord vamp (I bVII leans to bVII's key
# for a bar, then back), and at 0.05 those swings walked beliefs off the key
# after the band was in: over the 270 generated benchmark takes, key changes
# after entry were 16 at 0.05 and 12 at 0.10.  Moves of two or more accidentals
# leave a gap far above either (0.9-1.0 for I IV V I up a whole step), and were
# followed equally fast at both.
# _KEY_RELATIVE_STICKY — v1.2 AMENDMENT (SPEC-V1 section 3).  A key and its
# RELATIVE (C minor / Eb major) contain exactly the same pitch collection, so
# a pitch-class histogram cannot distinguish them in principle — only metric
# position and bass register can, and neither is available to this tracker.
# On loop-based modal material the raw distribution actively favours the
# relative: renders/trap is i-bIII-bVI-bVII (Cm/Eb/Ab/Bb), in which Eb appears
# in 3 of 4 chords and C in only 2, so C minor lost to Eb major by +0.32 to
# +0.38 correlation on all three minor-key renders — far above
# _KEY_SWITCH_MARGIN, so the ordinary hysteresis never engaged.  (Measured:
# swapping in Temperley Kostka-Payne or Albrecht-Shanahan profiles answers
# Eb major too, and additionally breaks C-major lofi into F major — it is not
# a profile-set bug.)  Since correlation cannot decide, the belief established
# from earlier evidence is KEPT: a relative challenger must clear a margin
# that a same-collection reading cannot reach.  Non-relative modulations are
# unaffected and still switch on the ordinary margin.
_KEY_RELATIVE_STICKY = 0.60
# _KEY_NEIGHBOR_STICKY — v1.4 AMENDMENT (SPEC-V1 section 3).  The relative
# stickiness above is the extreme case of a general fact: two keys whose
# diatonic collections differ by ONE accidental share 6 of 7 pitch classes, so
# a pitch-class histogram separates them by very little and a small correlation
# edge is not evidence of a modulation.  Measured on renders/flagship.wav
# (C minor throughout): the belief walked C minor -> Bb major (t=33.7s) ->
# Eb major (t=44.0s) and finished on Eb — Bb major is NOT the relative, so
# _KEY_RELATIVE_STICKY never applied, and the ordinary 0.05 margin was cleared
# by a passing bVII-heavy phrase.  Keys at collection distance 1 now need a
# margin between the ordinary one and the relative one; distance >= 2 (a real
# modulation, or any change of tonal centre worth following) is unaffected.
#
# The value is bounded from above by a measurement, not chosen for taste: on
# material that is unambiguously and entirely in one key, the correlation gap
# to its one-accidental neighbour TOPS OUT at a ceiling (with the probe-tone
# profiles 0.187, for E major over G# minor on a saturated E-A-B loop).  A
# margin at or above that would make the rule absolute — a genuine modulation
# to a neighbouring key could never be followed at all — so the margin is set
# just under the ceiling: strongly sticky, still passable by a real key
# change.  At 0.15 the A-minor fixtures stopped drifting to F major and
# flagship stopped visiting Bb major, while a full modulation still landed.
# The v1.6 score (profiles and root term) separates neighbouring keys further,
# so the same stickiness needs a larger margin: below 0.26 the subdominant
# passage in test_harmony moves the key within its 35 s.  The ceiling, measured
# on generated strummed takes as the new key's lead over the old once a move
# has settled: up a fifth 0.40-0.55 (I IV V I), 0.30-0.42 (I V vi IV), 0.45-0.54
# (i iv v i), followed in 14-36 s (one I V vi IV take of four went to the new
# key's relative minor instead, as its vi chord dwelt).  That is slower than
# with the probe-tone profiles at 0.15, which followed the same moves in 5-22 s.
# Up a fourth the lead is smaller, -0.02 to 0.12 major and 0.12 to 0.31 minor
# (every partial leans a note toward its dominant), so a move to the
# subdominant is often not followed; it was not at 0.15 either.  An I IV vamp
# moved up a fifth is the old key's V I and leads by -0.01 to 0.11: not
# followed (at 0.15 it was, after 42-90 s).
_KEY_NEIGHBOR_STICKY = 0.26
_KEY_CONF_FLOOR = 0.3         # below this correlation, key reports None
_KEY_MASS_FLOOR = 5.0         # min accumulated voiced chroma mass (guards std)
# _KEY_MIN_EVIDENCE_S — v1.2 AMENDMENT (SPEC-V1 section 3).  The only gate on
# naming a key used to be _KEY_MASS_FLOOR = 5.0, and because frame chroma is
# L1-normalised every voiced frame contributes EXACTLY 1.0 mass — so the
# "mass" floor was really a floor of 5 FRAMES = 53ms, and the first
# ``key_change`` event fired at t=0.064s on every input, before any musical
# evidence existed.  _KEY_CONF_FLOOR = 0.3 is no second guard either: a
# histogram containing a SINGLE pitch class already correlates r=0.684 with
# that pitch's KK major profile.  Measured on plume's own renders, the key
# first declared at 53ms was wrong on 3 of 4 (lofi C major -> "C minor", trap
# C minor -> "F# minor", flagship C minor -> "C major") and was held for
# 5-11s.  Key belief now requires _KEY_MIN_EVIDENCE_S of decayed VOICED TIME,
# which is sample-rate independent and says what it means.  Measured sweep:
# 1.0-2.5s makes all four renders first-declare correctly; above ~4s the
# relative-major ambiguity (below) takes over again.
_KEY_MIN_EVIDENCE_S = 2.0

# ---- v1.4 AMENDMENTS to the key belief (SPEC-V1 section 3) ----------------
#
# _KEY_BASS_WEIGHT.  The key histogram is accumulated from
# ``chroma + _KEY_BASS_WEIGHT * chroma_low`` rather than from ``chroma``
# alone.  ``chroma`` is flat across registers — a bass root counts exactly as
# much as an upper extension — but the bass note is the strongest single piece
# of key evidence a musician has.  Minimal repro (pure sines): C2 + E3 G3 B3 D4
# (a Cmaj9) named chord "C" correctly and key "G MAJOR", and doubling the C2
# loudly recovered "C major" — i.e. the estimator was reading pitch-class MASS,
# not root evidence.  ``Frame.chroma_low`` (features.CHROMA_BASS_*) is a
# register-tilted bass-band chroma whose sum is the low band's SHARE of the
# frame's pitched evidence, so this term is self-scaling: a frame with a real
# bass note contributes a lot of root evidence and a frame with nothing below
# C4 contributes almost none.  The weight is greater than 1 BECAUSE the term
# is a share and not a unit vector: measured medians run 0.006 (no bass) to
# 0.13 (bass fixture).  Measured safe band on the whole probe corpus (the
# Cmaj9 repro, pure tones, plume's four renders, the pluck fixtures): 1.0 to
# 3.0 all give the same, correct answer everywhere; 2.0 is the middle of it.
# Above ~4 the renders' relative-major pull starts dragging the incumbent's
# own correlation toward _KEY_CONF_FLOOR and synthwave begins dropping its
# key.  Chord matching does not use this term at all.
_KEY_BASS_WEIGHT = 2.0
#
# _KEY_DUTY_FLOOR.  Unpitched material used to get a confident key: a bare
# click track (30ms of 1kHz per beat) named a key at conf ~0.65 and the sideman
# comped on it, because "voiced" only asks that a frame have SOME chroma mass
# and the decayed evidence clock eventually crossed _KEY_MIN_EVIDENCE_S no
# matter how sparse the voicing was.  What separates a click from music is not
# the spectrum of one frame — a 1kHz click really is a tone — but SUSTAIN.
# Measured voiced-frame fraction: click track 0.14, every render and fixture
# 1.00.  A key may therefore only be ACQUIRED while the decayed voiced-time
# fraction clears this floor.  It is not a retention condition: once a key is
# established a long rest must not erase it (the ``long_stop`` fixture is 8
# silent bars), so retention keeps using the absolute evidence floor alone.
_KEY_DUTY_FLOOR = 0.5
#
# _KEY_COMMIT_EVIDENCE_S.  Key belief used to be snap-then-sticky: the first
# key_change fired at exactly t=2.112s in EVERY run — the instant the evidence
# clock crossed _KEY_MIN_EVIDENCE_S, on every input regardless of what was in
# it — and the belief was then asserted at full confidence (0.64-0.89) on the
# strength of ~2s of audio.  ``key_conf`` is now DISCOUNTED by
# ``min(1, evidence_s / _KEY_COMMIT_EVIDENCE_S)``, so an early reading reports
# what it actually is: a guess.  Because a key is only reported once its
# confidence clears _KEY_CONF_FLOOR, that also makes the first declaration a
# function of the material rather than a constant — measured first
# ``key_change`` times moved from 2.112s on EVERYTHING to 2.8-3.9s spread by
# how strongly each input states its key.
#
# NOT changed with it: the switch hysteresis.  An earlier version of this
# amendment also let a belief below the commitment threshold be re-pointed at
# the running argmax without margin or sustain, on the reasoning that an
# uncommitted belief is not worth defending.  Measured, that is strictly
# worse: a 16-bar C-minor synthwave render states C minor for its first ~0.7s
# of evidence and then spends the take on its bVII and bIII, so free
# re-evaluation walked the belief to Bb major and left it there, and the same
# happened on trap.  The correlation ordering early in a loop-based tune is
# not a better estimate than the incumbent — it is just noisier.  Re-evaluation
# stays the province of the ordinary margin-plus-sustain hysteresis at all
# times; what the ramp changes is how loudly an unsettled belief speaks.
_KEY_COMMIT_EVIDENCE_S = 8.0

_HMM_LAG = 3                  # fixed-lag smoothing, frames
_HMM_BETA = 8.0               # cosine-score -> likelihood sharpness
_S_N_VOICED = 0.1             # N-state score while voiced (silence scores 1)
_EMIT_POSTERIOR = 0.5         # emit when winner's posterior > this ...
_EMIT_CONSECUTIVE = 2         # ... this many lagged frames in a row

_REG_HALFLIFE_S = 5.0         # register histogram decay half-life
_REG_SIGMA_ST = 5.0           # Gaussian octave-placement width, semitones
_REG_SILENCE_S = 2.0          # no voiced frame this long -> register None
_REG_GRID = np.arange(24, 109)            # candidate MIDI notes
_REG_GRID_PC = _REG_GRID % 12

_VOICED_CHROMA_EPS = 1e-6
_VOICED_RMS_DB = -80.0


def _key_profiles(major: np.ndarray, minor: np.ndarray) -> np.ndarray:
    """(24, 12) row-z-scored key profiles; row k<12 = major tonic k, else minor."""
    rows = [np.roll(major, t) for t in range(12)]
    rows += [np.roll(minor, t) for t in range(12)]
    m = np.array(rows, dtype=np.float64)
    m -= m.mean(axis=1, keepdims=True)
    m /= m.std(axis=1, keepdims=True)
    return m


def _z(p: np.ndarray) -> np.ndarray:
    return (p - p.mean()) / p.std()


def _key_template(probe: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """One mode's template, tonic first: probe-tone plus a share of corpus."""
    return _z(_z(probe) + _KEY_CORPUS_SHARE * _z(corpus))


def _as_heard(profile: np.ndarray) -> np.ndarray:
    """The chroma a scale weighted by ``profile`` sounds as, partials included."""
    partials = np.zeros(12)
    for n in range(1, 9):
        partials[int(round(12 * np.log2(n))) % 12] += _KEY_PARTIAL_DECAY ** (n - 1)
    weight = profile - profile.min()
    return sum(weight[pc] * np.roll(partials, pc) for pc in range(12))


def _root_triads() -> np.ndarray:
    """(24, 12) L2-normalized triads with the root counted _ROOT_TEMPLATE_WEIGHT
    times; row r<12 = major root r, else minor.  See _KEY_ROOT_WEIGHT."""
    t = (_triad_templates() > 0).astype(np.float64)
    for r in range(24):
        t[r, r % 12] = _ROOT_TEMPLATE_WEIGHT
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def _triad_templates() -> np.ndarray:
    """(24, 12) L2-normalized binary triads; row r<12 = major root r, else minor."""
    t = np.zeros((24, 12))
    for r in range(12):
        for iv in (0, 4, 7):
            t[r, (r + iv) % 12] = 1.0
        for iv in (0, 3, 7):
            t[12 + r, (r + iv) % 12] = 1.0
    return t / np.linalg.norm(t, axis=1, keepdims=True)


# How tonal the evidence is (key_conf): the probe-tone profiles alone.
_TONAL_PROFILES = _key_profiles(_KK_MAJOR, _KK_MINOR)
# Which key it is (see _KEY_CORPUS_SHARE): for MIDI as written, for audio as
# it sounds (_KEY_PARTIAL_DECAY).
_PROFILES_BARE = _key_profiles(_key_template(_KK_MAJOR, _CORPUS_MAJOR),
                               _key_template(_KK_MINOR, _CORPUS_MINOR))
_PROFILES = _key_profiles(_as_heard(_key_template(_KK_MAJOR, _CORPUS_MAJOR)),
                          _as_heard(_key_template(_KK_MINOR, _CORPUS_MINOR)))
_TEMPLATES = _triad_templates()
_ROOT_TRIADS = _root_triads()
_N_STATES = 25  # 24 triads + N
_N_IDX = 24


@dataclass(frozen=True)
class HarmonyState:
    t_s: float
    key: str | None            # e.g. "F minor" (parse_key-compatible)
    key_conf: float            # 0..1
    chord: str | None          # triad symbol spelled per key ("Bbm", "Ab")
    chord_conf: float
    register_low_midi: float | None
    register_high_midi: float | None
    #: What the evidence says right now, before hysteresis: the best-fitting key
    #: and how far its fit leads the runner-up's.  The belief above is ``key``;
    #: these let a caller about to commit check that belief against the present.
    key_best: str | None = None
    key_lead: float = 0.0


_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)


def _collections() -> np.ndarray:
    """(24, 12) bool: the diatonic pitch-class collection of each key index.

    A natural-minor key has exactly its relative major's collection, so
    relative pairs come out identical — which is the point: collection
    distance is the single quantity both stickiness rules are written in.
    """
    out = np.zeros((24, 12), dtype=bool)
    for idx in range(24):
        tonic = idx % 12
        root = tonic if idx < 12 else (tonic + 3) % 12
        for d in _MAJOR_SCALE:
            out[idx, (root + d) % 12] = True
    return out


_COLLECTIONS = _collections()
# pairwise symmetric-difference size, in pitch classes
_COLL_DIST = (_COLLECTIONS[:, None, :] != _COLLECTIONS[None, :, :]).sum(axis=2)


def _switch_margin(inc: int, challenger: int) -> float:
    """Correlation margin a challenger must clear to unseat the incumbent.

    Scaled by how much of the incumbent's pitch collection the challenger
    shares: identical collection (the relative) is undecidable from a
    pitch-class histogram in principle, one accidental apart is nearly so, and
    anything further is an ordinary modulation.
    """
    dist = int(_COLL_DIST[inc, challenger])
    if dist == 0:                       # relative major/minor: same 7 pcs
        return _KEY_RELATIVE_STICKY
    if dist <= 2:                        # one accidental apart: 6 of 7 shared
        return _KEY_NEIGHBOR_STICKY
    return _KEY_SWITCH_MARGIN


def _key_index_to_obj(idx: int) -> Key:
    return Key(tonic=idx % 12, mode="major" if idx < 12 else "minor")


def _key_name(idx: int) -> str:
    k = _key_index_to_obj(idx)
    return f"{spell(k.tonic, k)} {k.mode}"


def _chord_name(idx: int, key: Key | None) -> str:
    root, minor = idx % 12, idx >= 12
    return spell(root, key) + ("m" if minor else "")


class HarmonyTracker:
    def __init__(self, *, key_halflife_s: float = 15.0, chord_selftrans: float = 0.9,
                 partials: bool = True):
        """``partials``: the chroma comes from sound, partials and all (audio);
        False for chroma built from the notes alone (MIDI).  See
        _KEY_PARTIAL_DECAY."""
        self._profiles = _PROFILES if partials else _PROFILES_BARE
        self._key_hl = float(key_halflife_s)
        off = (1.0 - float(chord_selftrans)) / (_N_STATES - 1)
        self._trans = np.full((_N_STATES, _N_STATES), off)
        np.fill_diagonal(self._trans, float(chord_selftrans))

        self._last_t: float | None = None
        # key belief
        self._hist = np.zeros(12)
        self._evidence_s = 0.0                 # decayed voiced time (see above)
        self._elapsed_s = 0.0                  # decayed total time (duty gate)
        self._key_idx: int | None = None       # incumbent, 0..23
        self._chal_idx: int | None = None
        self._chal_time = 0.0
        self._key_conf = 0.0
        self._roots = np.zeros(12)             # where the chords sit (see above)
        self._best_idx: int | None = None
        self._best_lead = 0.0
        # chord HMM
        self._alpha = np.full(_N_STATES, 1.0 / _N_STATES)
        self._lik_buf: deque[np.ndarray] = deque(maxlen=_HMM_LAG + 1)
        self._alpha_buf: deque[np.ndarray] = deque(maxlen=_HMM_LAG + 1)
        self._gamma: np.ndarray | None = None  # latest lagged posterior
        self._pending_idx: int | None = None
        self._pending_count = 0
        self._emit_idx: int | None = None      # emitted state (24 -> None chord)
        # register
        self._reg_hist = np.zeros(_REG_GRID.shape[0])
        self._last_voiced_t: float | None = None
        self._state = HarmonyState(
            t_s=0.0, key=None, key_conf=0.0, chord=None, chord_conf=0.0,
            register_low_midi=None, register_high_midi=None,
        )

    # ------------------------------------------------------------------ api

    def process(self, frames: Sequence["Frame"]) -> HarmonyState:
        for f in frames:
            self._step(f)
        return self._state

    def adopt(self, key: str) -> None:
        """Take ``key`` as the belief from here on, as the band has committed to it.

        The band may come in on the key the present evidence favours rather than
        on this tracker's belief, which can still be holding an early reading
        (see bandentry ENTRY_BEST_LEAD).  Left alone, that stale belief would
        go on disagreeing with the band's key and, after the governor's release
        bars, pull the band back to it.  Adopting the band's key leaves the
        evidence untouched; only the incumbent changes, so it still has to
        hold against challengers by the ordinary margins.
        """
        k = parse_key(key)
        self._key_idx = k.tonic + (0 if k.mode == "major" else 12)
        self._chal_idx, self._chal_time = None, 0.0

    # ------------------------------------------------------------- internals

    def _step(self, f: "Frame") -> None:
        t = float(f.t_s)
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t

        chroma = np.asarray(f.chroma, dtype=np.float64)
        # duck-typed: frames predating v1.4 carry no bass chroma, in which case
        # the key histogram falls back to the flat chroma alone.
        low = getattr(f, "chroma_low", None)
        chroma_low = (np.zeros(12) if low is None
                      else np.asarray(low, dtype=np.float64))
        voiced = chroma.sum() > _VOICED_CHROMA_EPS and f.rms_db > _VOICED_RMS_DB
        if voiced:
            self._last_voiced_t = t

        self._update_key(chroma, chroma_low, voiced, dt)
        self._update_chord(chroma, voiced)
        self._update_register(chroma, voiced, float(f.centroid_hz), dt)

        key_obj = self._key_obj()
        reg_lo, reg_hi = self._register(t)
        self._state = HarmonyState(
            t_s=t,
            key=None if key_obj is None else _key_name(self._key_idx),  # type: ignore[arg-type]
            key_conf=self._key_conf if self._key_idx is not None else 0.0,
            chord=self._chord_symbol(key_obj),
            chord_conf=self._chord_conf(),
            register_low_midi=reg_lo,
            register_high_midi=reg_hi,
            key_best=None if self._best_idx is None else _key_name(self._best_idx),
            key_lead=self._best_lead,
        )

    # ---- key

    def _update_key(self, chroma: np.ndarray, chroma_low: np.ndarray,
                    voiced: bool, dt: float) -> None:
        if dt > 0.0:
            decay = 0.5 ** (dt / self._key_hl)
            self._hist *= decay
            self._roots *= decay
            self._evidence_s *= decay
            self._elapsed_s *= decay
        self._elapsed_s += dt          # voiced or not: the duty denominator
        if voiced:
            heard = chroma + _KEY_BASS_WEIGHT * chroma_low
            self._hist += heard
            self._evidence_s += dt
            # The triad this frame sounds like, credited to its root.
            fit = _HMM_BETA * (_ROOT_TRIADS @ heard) / np.linalg.norm(heard)
            p = np.exp(fit - fit.max())
            p /= p.sum()
            self._roots += p[:12] + p[12:]

        mass = self._hist.sum()
        std = self._hist.std()
        if self._evidence_s < _KEY_MIN_EVIDENCE_S \
                or mass < _KEY_MASS_FLOOR or std <= 0.0:
            if self._key_idx is not None:
                self._key_conf = 0.0
            self._best_idx, self._best_lead = None, 0.0
            return
        hz = (self._hist - self._hist.mean()) / std
        corr = self._profiles @ hz / 12.0     # Pearson r vs each of 24 keys
        # Each key scored by its correlation and the share of chords on its tonic.
        score = corr + _KEY_ROOT_WEIGHT * np.tile(self._roots / self._roots.sum(), 2)
        # How tonal: the probe-tone fit, the scale every floor here was set on.
        tonal = _TONAL_PROFILES @ hz / 12.0

        # Confidence is the incumbent's correlation DISCOUNTED by how much
        # evidence stands behind it, so a fresh reading cannot announce itself
        # with the same authority as a settled one (_KEY_COMMIT_EVIDENCE_S).
        settled = min(1.0, self._evidence_s / _KEY_COMMIT_EVIDENCE_S)

        def conf_of(idx: int) -> float:
            return float(np.clip(tonal[idx], 0.0, 1.0) * settled)

        best = int(np.argmax(score))
        inc = self._key_idx
        # How far the best key now leads the belief, in units of the margin the
        # belief's hysteresis would ask of it (bandentry ENTRY_BEST_LEAD).
        self._best_idx = best
        self._best_lead = (0.0 if inc is None or best == inc
                           else float(score[best] - score[inc]) / _switch_margin(inc, best))
        if self._key_idx is None:
            # ACQUISITION.  Needs sustained tonal content, not merely enough
            # accumulated frames: see _KEY_DUTY_FLOOR.
            duty = (self._evidence_s / self._elapsed_s
                    if self._elapsed_s > 0.0 else 0.0)
            if tonal[best] >= _KEY_CONF_FLOOR and duty >= _KEY_DUTY_FLOOR:
                self._key_idx = best
                self._chal_idx, self._chal_time = None, 0.0
                self._key_conf = conf_of(best)
            else:
                self._key_conf = 0.0
            return

        inc = self._key_idx
        self._key_conf = conf_of(inc)
        others = np.delete(np.arange(24), inc)
        cb = int(others[np.argmax(score[others])])
        margin = _switch_margin(inc, cb)
        if score[cb] > score[inc] + margin:
            if cb == self._chal_idx:
                self._chal_time += dt
            else:
                self._chal_idx, self._chal_time = cb, 0.0
            if self._chal_time >= _KEY_SWITCH_SUSTAIN_S:
                self._key_idx = cb
                self._key_conf = conf_of(cb)
                self._chal_idx, self._chal_time = None, 0.0
        else:
            self._chal_idx, self._chal_time = None, 0.0

    def _key_obj(self) -> Key | None:
        if self._key_idx is None or self._key_conf < _KEY_CONF_FLOOR:
            return None
        return _key_index_to_obj(self._key_idx)

    # ---- chord

    def _update_chord(self, chroma: np.ndarray, voiced: bool) -> None:
        scores = np.zeros(_N_STATES)
        if voiced:
            norm = np.linalg.norm(chroma)
            scores[:24] = _TEMPLATES @ chroma / norm
            scores[_N_IDX] = _S_N_VOICED
        else:
            scores[_N_IDX] = 1.0
        lik = np.exp(_HMM_BETA * scores)

        self._alpha = (self._trans.T @ self._alpha) * lik
        self._alpha /= self._alpha.sum()
        self._lik_buf.append(lik)
        self._alpha_buf.append(self._alpha.copy())
        if len(self._lik_buf) <= _HMM_LAG:
            return

        # backward pass over the lag window: posterior of the oldest frame
        beta = np.ones(_N_STATES)
        for k in range(len(self._lik_buf) - 1, 0, -1):
            beta = self._trans @ (self._lik_buf[k] * beta)
            beta /= beta.sum()
        gamma = self._alpha_buf[0] * beta
        gamma /= gamma.sum()
        self._gamma = gamma

        cand = int(np.argmax(gamma))
        if gamma[cand] > _EMIT_POSTERIOR:
            if cand == self._pending_idx:
                self._pending_count += 1
            else:
                self._pending_idx, self._pending_count = cand, 1
            if self._pending_count >= _EMIT_CONSECUTIVE:
                self._emit_idx = cand
        else:
            self._pending_idx, self._pending_count = None, 0

    def _chord_symbol(self, key: Key | None) -> str | None:
        if self._emit_idx is None or self._emit_idx == _N_IDX:
            return None
        return _chord_name(self._emit_idx, key)

    def _chord_conf(self) -> float:
        """Posterior of the reported chord; 0.0 while no chord is reported.

        The N (silence) state deliberately reports 0.0 even though its own
        posterior may be ~1: ``chord_conf`` is confidence IN the emitted
        chord symbol, and ``chord=None`` carries none.
        """
        if self._gamma is None or self._emit_idx is None \
                or self._emit_idx == _N_IDX:
            return 0.0
        return float(self._gamma[self._emit_idx])

    # ---- register

    def _update_register(
        self, chroma: np.ndarray, voiced: bool, centroid_hz: float, dt: float
    ) -> None:
        if dt > 0.0:
            self._reg_hist *= 0.5 ** (dt / _REG_HALFLIFE_S)
        if voiced and centroid_hz > 0.0:
            f0 = centroid_hz / 2.0
            mu = 69.0 + 12.0 * np.log2(f0 / 440.0)
            w = chroma[_REG_GRID_PC] * np.exp(
                -0.5 * ((_REG_GRID - mu) / _REG_SIGMA_ST) ** 2
            )
            self._reg_hist += w

    def _register(self, t: float) -> tuple[float | None, float | None]:
        if self._last_voiced_t is None or t - self._last_voiced_t > _REG_SILENCE_S:
            return None, None
        total = self._reg_hist.sum()
        if total <= 1e-9:
            return None, None
        cum = np.cumsum(self._reg_hist)
        lo = float(_REG_GRID[np.searchsorted(cum, 0.10 * total, side="left")])
        hi = float(_REG_GRID[np.searchsorted(cum, 0.90 * total, side="left")])
        return lo, hi
