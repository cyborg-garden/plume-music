"""Causal frame features (SPEC-V1 section 1).

One :class:`FeatureExtractor` instance consumes an arbitrary-blocked mono
float32 stream and emits one :class:`Frame` per completed hop.  Frame ``k``
describes the trailing ``WIN``-sample window ending at stream sample
``(k + 1) * HOP`` (zero-padded on the left near stream start), so
``t_s = (k + 1) * HOP / sr`` — the time of the frame END.  Everything is a
pure function of the input stream: no rng, no wall-clock, and framing depends
only on stream content, never on how it was blocked.

v1.5 adds ``Frame.attack`` — a banded, level-independent ATTACK detection
function — and :class:`CountInDetector`, which reads it.  ``flux`` and the
``onset`` flag picked from it are unchanged and still drive ``mind``'s onset
density AND, by default, the beat tracker.  ``attack`` is what the tracker
listens to under its OPT-IN acoustic mode (``beat.ACOUSTIC_ENV`` /
``BeatTracker(acoustic=True)``); ``CountInDetector`` reads it unconditionally,
because the count-in hatch must work in either mode.  See the ATTACK_* block
below for what the function is, and ``beat.py``'s ACOUSTIC MODE block for the
corpus A/B that made it opt-in rather than the default.
"""
from __future__ import annotations

import dataclasses
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

HOP = 512          # 10.67ms at 48k
WIN = 2048

# CHROMA_WIN — v1.2 AMENDMENT (SPEC-V1 section 1).  The flux/onset/RMS path
# needs a SHORT window for temporal precision (WIN = 2048 = 42.7ms), but the
# same window cannot resolve the register real music actually lives in: at
# 48k a 2048-point Hann main lobe is 4 bins = 93.8Hz wide, while A3 (220.0Hz)
# and C4 (261.6Hz) are only 41.6Hz apart.  They MERGE into one lobe whose
# interpolated centroid lands near 243Hz -> pitch class B, a note in neither.
# Because the merge depends on the partials' relative phase, the reported
# pitch classes flicker frame to frame (measured on a steady A3+C4 dyad:
# B, (A,C), B, B, (A,C), (G#,C), ...).  Sparse widely-spaced pluck fixtures
# never trigger this; every real mix does, because that is exactly where the
# bass note and the chord root sit.  Chroma therefore analyses its own longer
# trailing window (4096 = 85.3ms, 11.7Hz bins) over the SAME hop and the same
# frame timing, so nothing else in the contract moves.
#
# KNOWN LIMIT, measured 2026-08-14 (v1.4, pre-existing — not introduced by the
# bass-chroma work below).  The same merge returns one octave lower and 4096 is
# not enough to stop it there: a 4096-point Hann main lobe is 4 bins = 46.9Hz
# wide, while C2-E2 are 17.0Hz apart and E2-G2 are 15.6Hz apart.  A CLOSE triad
# voiced in the C2 octave therefore reads as noise — measured on pure
# C2+E2+G2 sines the chroma is F#:0.22 G#:0.14 D:0.13 B:0.12 with NO C at all,
# and the key comes out "B minor".  Resolving 15.6Hz needs CHROMA_WIN = 16384
# (341ms), which would smear chord changes badly, so this is documented rather
# than fixed: bass parts are voiced in octaves and fifths, not close triads,
# and a bass note against a chord an octave up (C2 + C3 E3 G3) resolves
# correctly at 4096 — measured chroma C:0.62 E:0.20 G:0.19.
CHROMA_WIN = 4096
_MAX_WIN = max(WIN, CHROMA_WIN)

RMS_FLOOR_DB = -90.0
_RMS_FLOOR_LIN = 10.0 ** (RMS_FLOOR_DB / 20.0)
_SILENT_MAG_EPS = 1e-10

# Onset picker defaults (adaptive threshold = median of trailing window + delta).
# ONSET_DELTA tuned 2.0 -> 4.0 for v1: at 2.0 the picker double-counted plucked
# strums (echoes/decay wobble), reading calm strumming at ~2.2 onsets/beat vs a
# true 1.0 — which sat right at MindParams.density_high and latched the inverse
# density coupling on calm material.  At 4.0 the fixture measurements are
# ~1.2/beat calm vs ~3.1/beat busy (truth 1.0 / 3.0): clean separation.
#
# v1.5 leaves this picker EXACTLY as it was.  It drives ``Frame.onset``, which
# is what ``mind`` and ``roles`` read as onset density AND what the beat
# tracker reads by default; only under the tracker's opt-in acoustic mode does
# it pick its own events off ``Frame.attack`` instead.  The note above
# ATTACK_BANDS explains why the two could not be merged even there.
ONSET_DELTA = 4.0          # flux units above trailing median
ONSET_MEDIAN_S = 0.35      # trailing median window
ONSET_MIN_SEP_S = 0.060    # minimum separation between onsets

# ---------------------------------------------------------------------------
# ATTACK_* — v1.5 AMENDMENT (SPEC-V1 section 1).  THE SUSTAINED-ACOUSTIC FIX.
# ---------------------------------------------------------------------------
# Everything in this module before v1.5 was tuned on programmed renders and on
# ``pluck`` fixtures: material with sharp broadband transients and short decay.
# The first session with a real human — a fingerpicked acoustic guitar in
# CGCFCE (an open tuning: every string rings, decays overlap for seconds,
# note-to-note transients are soft flesh attacks) through a laptop mic — put
# the tracker at 167.9/163.0/169.2/165.4/80.6/80.3/121.0 bpm with confidence
# pinned at 0.41-0.62.  Reproduced here on a synthesized open-tuned fixture
# (tests/test_acoustic.py) and measured; two separate defects, both in ``flux``:
#
# 1. ``flux`` IS A SUM OVER 1025 BINS, so its floor is bin-count noise.  Under
#    heavy sustain, overlapping decays beat against each other and every bin
#    wobbles a little in both directions; the half-wave rectification keeps only
#    the up-wobbles, and 1025 of them add up.  Measured on the fingerpicked
#    fixture: median flux 2.50, p90 5.06 — a p90/median CONTRAST of 2.02, i.e.
#    a real attack is only twice the sustain noise.  On the same material the
#    banded detector below measures a contrast of 4.46.  The absolute
#    ``ONSET_DELTA`` threshold then has no good value: 4.0 flux units is most
#    of a soft attack on this material and nothing at all on a loud render
#    (this is also the mechanism behind the already-documented "20 dB down,
#    onsets thin from ~10/s to ~2.8/s" limit in tests/test_real_material.py).
#
# 2. ``flux`` DESTROYS ACCENT, and accent is what tells a beat from its
#    subdivision.  A fingerpicked pattern plays a loud bass string ON the beat
#    and a quiet upper string on the "and"; the human hears 84 bpm.  Because
#    log1p-of-magnitude summed over all bins is dominated by the ringing
#    already present, the loud and quiet events land at nearly the same flux
#    value, and the flux envelope's autocorrelation ranks the 8th-note lag
#    ABOVE the beat lag (+0.755 vs +0.725).  That is a 2x tempo error with a
#    positive margin — the tracker was not confused, it was confidently wrong,
#    and it reported 167.9 bpm at confidence 0.83.
#
# ``Frame.attack`` is the replacement detection function.  Under the beat
# tracker's OPT-IN acoustic mode it is what the tracker reads — both for its
# autocorrelation envelope and for its own onset picker (beat.py
# ``BEAT_ATTACK_*``); by default the tracker stays on ``flux``/``onset``,
# because the same properties that make ``attack`` right for sustained guitar
# make it measurably worse on the programmed corpus (beat.py ACOUSTIC MODE has
# the A/B).  ``Frame.onset`` comes from ``flux`` in BOTH modes; see the note
# above ONSET_DELTA and the one below.  Three changes, each measured:
#
# - **BANDED.**  The rFFT is grouped into ``ATTACK_BANDS`` log-spaced bands
#   over [ATTACK_FMIN_HZ, ATTACK_FMAX_HZ] and magnitudes are SUMMED INSIDE a
#   band before compression.  Summing first averages the beating away instead
#   of rectifying it, which is the whole of defect 1: the p90/median contrast
#   on the fingerpicked fixture goes from 2.02 (flux) to 10.8 (attack, over
#   frames where there is any attack at all).
# - **A TRUE LOG RATIO, AND THIS IS THE TERM THAT FIXES THE OCTAVE.**
#   ``log1p(|X|)`` is only logarithmic where |X| >> 1, so at ordinary live-mic
#   levels it degenerates to a LINEAR function of magnitude — which is both why
#   an absolute threshold could never hold at two gains AND why accent was
#   destroyed, since a linear difference is dominated by whatever is already
#   ringing.  ``log1p(ATTACK_LAMBDA * |X|)`` with lambda 1000 puts every level
#   from about -110 dBFS up into the logarithmic region, so a band difference
#   is a pure log RATIO: multiplying the whole input by any constant leaves
#   ``attack`` unchanged.  Measured on the fingerpicked fixture, beat-lag ACF
#   minus 8th-lag ACF is -0.030 on flux and +0.060 on attack — the sign flip
#   that turns 167.9 bpm into 84.0.
# - **MAX-FILTERED PREVIOUS FRAME, AT A LAG** (the SuperFlux trick, Böck &
#   Widmer 2013).  The reference is not the previous frame's band but the
#   MAXIMUM over that band and its two neighbours ``ATTACK_LAG`` hops earlier,
#   which discriminates an attack from a partial merely drifting or beating
#   between bands.
#
#   HONEST ACCOUNTING, because the first version of this comment got it wrong:
#   the max filter is NOT what restores the accent.  That claim came from a
#   prototype that used plain ``log1p``, where the banded detector scored
#   -0.004 without the filter and +0.053 with it.  Re-measured on the SHIPPED
#   function, ATTACK_MAXFILT=0 scores +0.068 and ATTACK_MAXFILT=1 scores
#   +0.060 — the filter is very slightly WORSE on that metric, and the log
#   ratio above is doing the work.  It is kept because it pays elsewhere, over
#   the whole corpus: synth100 locks at 4.6s instead of 10.0s and stays locked
#   89% of the take instead of 75%, the fixture humans lock at 7.6s instead of
#   9.3s, jn_takes_over at 7.0s instead of 9.1s — and, the reason that matters
#   most here, the pulse-free ``drone`` fixture reads confidence 0.284 with the
#   filter and 0.356 without it, i.e. the filter keeps genuinely ambiguous
#   material further below the sideman's entry gate.  It costs the acoustic
#   fixtures about 2.5s of time-to-lock.  ATTACK_MAXFILT=2 and 3 were also
#   measured and are worse on everything.
#
# ``flux``, ``onset`` and ``onset_strength`` are BIT-IDENTICAL to v1.4 — same
# definitions, same values, same picker.  ``flux`` itself is read nowhere
# outside this module; the ``onset`` flag picked from it is read by ``mind``,
# ``roles``, ``sim`` and ``realtime``, and leaving it alone is what keeps this
# change out of their calibration.
ATTACK_BANDS = 24          # log-spaced bands (measured: 12 too coarse, 40 no
                           # better than 24 and 1.7x the cost)
ATTACK_FMIN_HZ = 50.0
ATTACK_FMAX_HZ = 10000.0
ATTACK_LAMBDA = 1000.0     # log1p(lambda*|X|): logarithmic from ~-110 dBFS up
ATTACK_LAG = 2             # hops back for the reference frame (21ms)
ATTACK_MAXFILT = 1         # +/- this many bands in the reference max-filter
ATTACK_BREADTH_EPS = 0.02  # a band "rose" if it rose by this many nats

# WHY ``Frame.onset`` IS STILL PICKED FROM ``flux`` AND NOT FROM ``attack``.
# Measured, and the measurement changed the design.  An attack-driven picker is
# strictly better at FINDING soft attacks — that is the entire point — but
# ``Frame.onset`` is not only the tracker's input: ``mind``, ``roles`` and
# ``sim`` read ONSET DENSITY off it, and ``MindParams.density_high`` was
# calibrated against this picker's numbers (SPEC-V1: ~1.2 onsets/beat calm vs
# ~3.1 busy).  A banded picker over the same corpus measures 2.36 onsets/beat
# on the CALM fixture and 2.76 on the BUSY one — the separation the coupling
# depends on collapses from 1.2-vs-3.1 to 2.36-vs-2.76, because a ringing
# pluck's decay ripples produce genuine band-level rises that are 10x weaker
# than the attack (measured: median 0.08 of the preceding attack) but still
# well above any threshold that also passes a soft acoustic attack.  Sweeping
# ATTACK_BANDS x ATTACK_LAG x ATTACK_MAXFILT x threshold (48 combinations) never
# brought the worst-case density ratio below 2.6x.  Retuning ``density_high``
# would fix it, and mind.py is not mine to change.
#
# So the two consumers are split, which is defensible on its own terms: ``mind``
# asks "how much is happening" (a broadband event count) and the tracker asks
# "where are the attacks" (a grid).  ``Frame.onset``/``onset_strength``/``flux``
# are therefore BIT-IDENTICAL to v1.4, and ``BeatTracker`` runs its own picker
# over ``Frame.attack`` — see beat.py ``_attack_onset`` and ``BEAT_ATTACK_*``.

# SPEC-V1 wrote "rectangular pitch-class mapping of rFFT bins 60-5000Hz", but
# at WIN=2048/48k the bins are 23.4Hz wide: below ~400Hz a semitone is narrower
# than a bin, so nearest-pitch-class assignment of fundamentals is quantization
# noise (a pure A3 at 220Hz straddles bins whose centers round to Ab and Bb —
# neither is A), and rectangular mapping credits every well-resolved upper
# partial to ITS OWN pitch class (a pluck's 5th harmonics map an Am triad's
# A/C/E onto C#/E/G# — exactly the C#m triad, which then outscores the played
# chord at 0.99 confidence).  Deviation, replacing the earlier 200Hz-floor
# bandaid: chroma is a PEAK-BASED HARMONIC PITCH SALIENCE.  Spectral local
# maxima in [CHROMA_FMIN_HZ, CHROMA_FMAX_HZ] are located with sub-bin
# precision (parabolic interpolation — accurate to a few Hz for isolated
# partials, well under a semitone even at 110Hz), and each peak votes for
# every candidate fundamental that would place it as harmonic h = 1..
# CHROMA_HARMONICS (within CHROMA_TOL_ST semitones), weighted by the peak
# magnitude times CHROMA_HARMONIC_DECAY ** (h - 1).  Upper partials therefore
# vote for the note that produced them, and low fundamentals are carried by
# their well-resolved harmonics.  Vectorized across frames and peaks;
# validated against pure tones, pluck triads, and the fixture keys (v1
# behavior test 10).
CHROMA_FMIN_HZ = 50.0
CHROMA_FMAX_HZ = 5000.0
CHROMA_F0_MIN_MIDI = 33        # A1, 55Hz — lowest counted fundamental
CHROMA_F0_MAX_MIDI = 96        # C7, ~2093Hz
CHROMA_HARMONICS = 6
CHROMA_HARMONIC_DECAY = 0.75
CHROMA_TOL_ST = 0.35           # peak-to-harmonic match tolerance, semitones

# CHROMA_REQUIRE_F0 — v1.2 AMENDMENT.  CHROMA_TOL_ST was intended to reject
# implausible harmonic hypotheses, but 12*log2(h) is within 0.35 semitones of
# an integer for EVERY h in 1..9 (h=3: 0.020, h=5: 0.137, h=6: 0.020,
# h=7: 0.312), so the tolerance only ever tests the peak's own
# frequency-measurement error and NEVER rejects a subharmonic hypothesis.
# Nothing required the candidate fundamental's own partial to be present, so a
# SINGLE spectral peak at pitch class p deposited {p: 0.661, p-7: 0.243,
# p-4: 0.096} — the minor triad a perfect fifth below the note played.  A pure
# 440Hz A read as "Dm" at 0.999 confidence, and that is the surviving
# mechanism behind out-of-key machine bass under the human's chord.  The fix
# is the standard corroboration condition for subharmonic-summation salience:
# a candidate fundamental only scores if some peak matched it at h == 1, i.e.
# its own fundamental is visible.  This costs nothing legitimate because
# CHROMA_F0_MIN_MIDI (55Hz) already sits above CHROMA_FMIN_HZ (50Hz), so every
# legal candidate's fundamental is inside the analysed band.
CHROMA_REQUIRE_F0 = True

# CHROMA_BASS_* — v1.4 AMENDMENT (SPEC-V1 sections 1 and 3).  ``chroma`` is
# flat across registers: it is a pitch-class histogram, so a bass root and an
# upper extension of the same voicing contribute equally.  That is right for
# CHORD matching (a triad template has no register) and wrong for KEY, where
# the bass note is the strongest single piece of evidence a listener has —
# measured minimal repro (pure sines): C2 + E3 G3 B3 D4 (a Cmaj9) reads chord
# "C" correctly but key "G major", because the collection {C,D,E,G,B} scores
# higher against the G-major profile (23.67) than the C-major one (22.28) once
# the root's register is thrown away.  Doubling the C2 loudly fixed it, which
# is exactly the diagnosis: the estimator was responding to pitch-class MASS,
# not to root evidence.
#
# ``Frame.chroma_low`` is therefore a second, REGISTER-TILTED chroma built from
# the same corroborated fundamentals but counting only those whose f0 lands in
# the bass band [CHROMA_BASS_MIN_MIDI, CHROMA_BASS_MAX_MIDI] (41.2-261.6 Hz,
# the "roughly 40-260Hz" band), each weighted by
# ``0.5 ** ((midi - CHROMA_BASS_MIN_MIDI) / CHROMA_BASS_HALFLIFE_ST)`` so that
# LOWER fundamentals count for more within the band rather than the band being
# a hard rectangular gate.  At halflife 5 semitones a C2 root outweighs an E3
# by 9x and a B3 by 20x, so the lowest sounding note dominates the vector
# without the higher band members vanishing.  The tilt matters more than the
# band edges: measured on the Cmaj9 repro, a shallow halflife of 8 leaves the
# other three in-band voices (E3, G3, B3) enough weight that C wins by only
# 0.094 correlation, while at 5 the same fix holds across a 3x range of
# harmony._KEY_BASS_WEIGHT instead of being a knife edge.
#
# It is deliberately NOT L1-normalized on its own.  It is divided by the
# frame's FULL vote mass, so ``chroma_low.sum()`` is the low-band's tilted
# SHARE of the frame's pitched evidence, in [0, 1] — a magnitude as well as a
# direction.  That makes the vector self-gating, which is the whole reason for
# the choice: a frame with a real bass note carries a large share and speaks
# loudly about the root, while a frame with nothing down there carries
# essentially none and cannot inject noise.  Measured medians: pluck fixtures
# with an 808 bass 0.127, plume's own renders 0.040-0.056, the ``high_register``
# fixture (voiced entirely above C5, no bass by construction) 0.006, a 1 kHz
# click track 0.000.  An L1-normalized version would have handed that click and
# that high-register strum the same unit of "root evidence" as a bass note.
# Nothing here touches ``chroma``; chord matching is unchanged.
CHROMA_BASS_MIN_MIDI = 28      # E1, 41.2Hz
CHROMA_BASS_MAX_MIDI = 60      # C4, 261.6Hz
CHROMA_BASS_HALFLIFE_ST = 5.0  # within-band tilt: weight halves every 5 st up


@dataclass(frozen=True)
class Frame:
    t_s: float             # time of frame END relative to stream start
    rms_db: float          # frame RMS in dBFS (floor -90)
    flux: float            # half-wave-rectified log-magnitude spectral flux, >= 0
    onset: bool            # causal peak-pick on flux (median+delta, min separation)
    onset_strength: float  # flux value when onset else 0
    chroma: np.ndarray     # shape (12,) float32, L1-normalized (zeros if silent)
    centroid_hz: float     # spectral centroid (0 if silent)
    # v1.4: register-tilted bass-band chroma (see CHROMA_BASS_* above).
    # shape (12,) float32; sums to the low band's share of the frame's pitched
    # evidence (0..1), all-zero when no bass fundamental is corroborated.
    # Defaulted so that duck-typed frame stubs predating v1.4 still work.
    chroma_low: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(12, dtype=np.float32)
    )
    # v1.5: banded, level-independent ATTACK detection function (see ATTACK_*
    # above).  >= 0.  Read by ``CountInDetector`` always, and by the beat
    # tracker only under its opt-in acoustic mode; ``mind`` and ``roles`` never
    # read it (see the note above ATTACK_BANDS for why the two are split).
    # Defaulted so duck-typed frame stubs predating v1.5 still work — they fall
    # back to ``flux``/``onset`` in BeatTracker.
    attack: float = 0.0
    # v1.5: fraction of the ATTACK_BANDS bands that rose in this frame, 0..1.
    # Impulse-vs-note discriminator for ``CountInDetector`` (see _attack).
    attack_breadth: float = 0.0
    # v1.5: set by whoever knows the machine is drowning the human out (the
    # realtime SelfMasker).  ``BeatTracker`` already honoured this attribute
    # duck-typed; it is a real field now because in ACOUSTIC MODE the tracker
    # picks its OWN onsets off ``attack``, so clearing ``onset`` alone no
    # longer suppresses that path — an onset picked from a frame the machine is
    # filling is the machine correcting its own phase toward itself.  See
    # beat.py ``BeatTracker.process(..., masked=)``.
    self_masked: bool = False


class FeatureExtractor:
    """Streaming causal feature extraction at HOP/WIN over an internal buffer."""

    def __init__(
        self,
        sr: int = 48000,
        *,
        onset_delta: float = ONSET_DELTA,
        onset_median_s: float = ONSET_MEDIAN_S,
        onset_min_sep_s: float = ONSET_MIN_SEP_S,
    ):
        if sr <= 0:
            raise ValueError(f"sr must be positive, got {sr}")
        self._sr = int(sr)
        self._onset_delta = float(onset_delta)
        self._med_frames = max(1, int(round(onset_median_s * self._sr / HOP)))
        self._min_sep_frames = max(1, int(math.ceil(onset_min_sep_s * self._sr / HOP)))

        # Stream buffer.  Invariant between calls: holds the trailing
        # (_MAX_WIN - HOP + r) stream samples, r in [0, HOP) = samples past the
        # last emitted frame boundary.  Initial zeros provide the left padding
        # for early (partially covered) windows.  The buffer is sized for the
        # LONGEST analysis window (chroma); the short WIN view is its tail, so
        # frame timing and the short-window features are unchanged.
        self._buf = np.zeros(_MAX_WIN - HOP, dtype=np.float32)
        self._frame_idx = 0                    # next frame index to emit
        self._last_onset_frame = -(10 ** 9)
        self._flux_hist: deque[float] = deque(maxlen=self._med_frames)

        self._win_fn = np.hanning(WIN)         # float64
        n_bins = WIN // 2 + 1
        self._prev_logmag = np.zeros(n_bins, dtype=np.float64)
        self._freqs = np.fft.rfftfreq(WIN, 1.0 / self._sr)

        # --- banded attack detection function (ATTACK_* above) -------------
        # Band edges as rFFT bin slices.  Every band is forced at least one bin
        # wide, so a sample rate whose bins are coarser than a low band still
        # yields ATTACK_BANDS usable bands rather than empty ones.
        edges = np.geomspace(min(ATTACK_FMIN_HZ, self._sr * 0.4),
                             min(ATTACK_FMAX_HZ, self._sr * 0.45),
                             ATTACK_BANDS + 1)
        idx = np.clip(np.searchsorted(self._freqs, edges), 1, n_bins - 1)
        for i in range(1, idx.size):           # keep the slices non-empty
            idx[i] = max(idx[i], idx[i - 1] + 1)
        idx = np.clip(idx, 1, n_bins)
        self._band_lo = idx[:-1].astype(np.int64)
        self._band_hi = idx[1:].astype(np.int64)
        self._n_bands = int(self._band_lo.size)
        # trailing band-log history: the ATTACK_LAG frames before this block,
        # so the reference frame is available across process() boundaries and
        # the result is independent of how the stream was blocked.
        #
        # Seeded from the FIRST frame rather than with zeros.  A zero seed
        # makes frame 0's reference "digital silence", so its attack value is
        # the whole log1p(ATTACK_LAMBDA*|X|) magnitude — measured at ~13
        # against a real attack of ~0.07, i.e. 200x.  One such sample sits
        # inside the tracker's 12-second autocorrelation window and dominates
        # its mean and variance for the entire window: measured on plume's own
        # renders, the zero seed pushed time-to-lock from 3.8s to 16.3s
        # (trap140) and 4.2s to 18.9s (synth100) — a delay of almost exactly
        # ACF_WINDOW_S, which is what identified it.
        self._band_hist: np.ndarray | None = None

        # Peak-based harmonic pitch salience over the LONGER chroma window.
        self._cwin_fn = np.hanning(CHROMA_WIN)
        self._cfreqs = np.fft.rfftfreq(CHROMA_WIN, 1.0 / self._sr)
        self._bin_hz = self._sr / CHROMA_WIN
        self._chroma_bin_ok = (self._cfreqs >= CHROMA_FMIN_HZ) & \
            (self._cfreqs <= min(CHROMA_FMAX_HZ, self._cfreqs[-1]))
        self._harm_w = CHROMA_HARMONIC_DECAY ** np.arange(CHROMA_HARMONICS)
        # candidate-midi offset per harmonic: midi(f/h) = midi(f) - 12*log2(h)
        self._harm_off = 12.0 * np.log2(np.arange(1, CHROMA_HARMONICS + 1))

    # ------------------------------------------------------------------ API

    def process(self, block: np.ndarray) -> list[Frame]:
        """Feed an arbitrary-length mono block; return newly completed frames."""
        x = np.asarray(block)
        if x.ndim != 1:
            raise ValueError(f"expected mono 1-D block, got shape {x.shape}")
        x = x.astype(np.float32, copy=False)
        if x.size:
            self._buf = np.concatenate([self._buf, x])
        n_new = (self._buf.size - (_MAX_WIN - HOP)) // HOP
        if n_new <= 0:
            return []
        long_w = np.lib.stride_tricks.sliding_window_view(
            self._buf, _MAX_WIN)[: n_new * HOP : HOP]
        # the short (WIN) window is the TAIL of the long one: same frame end,
        # so t_s / rms / flux / onset are bit-identical to the WIN-only path.
        frames = self._compute(long_w[:, _MAX_WIN - WIN:], long_w)
        self._buf = self._buf[n_new * HOP:]
        return frames

    def flush(self) -> list[Frame]:
        """Zero-pad any pending partial hop and emit its frame (if any)."""
        r = self._buf.size - (_MAX_WIN - HOP)
        if r <= 0:
            return []
        return self.process(np.zeros(HOP - r, dtype=np.float32))

    # ------------------------------------------------------------- internals

    def _chroma(self, mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Peak-based harmonic pitch salience -> (n, 12) raw chroma.

        Vectorized over frames AND peaks: local spectral maxima are located
        with parabolic (sub-bin) frequency interpolation, then each peak
        votes, with geometrically decaying weight, for every candidate
        fundamental placing it as one of the first CHROMA_HARMONICS partials.

        Returns ``(chroma, chroma_low)``: the same votes accumulated twice,
        the second time keeping only candidate fundamentals inside the bass
        band and tilting them by register (see CHROMA_BASS_* above).  Both are
        raw (un-normalized); the caller L1-normalizes each independently.
        """
        n = mag.shape[0]
        out = np.zeros((n, 12))
        out_low = np.zeros((n, 12))
        interior = mag[:, 1:-1]
        is_peak = (interior > mag[:, :-2]) & (interior >= mag[:, 2:]) & \
            (interior > _SILENT_MAG_EPS) & self._chroma_bin_ok[None, 1:-1]
        fi, ki = np.nonzero(is_peak)          # frame index, bin index - 1
        if fi.size == 0:
            return out, out_low
        k = ki + 1
        a, b, c = mag[fi, k - 1], mag[fi, k], mag[fi, k + 1]
        denom = a - 2.0 * b + c
        delta = np.where(np.abs(denom) > 1e-12,
                         0.5 * (a - c) / np.where(denom != 0.0, denom, 1.0),
                         0.0)
        delta = np.clip(delta, -0.5, 0.5)
        freq = (k + delta) * self._bin_hz
        peak_midi = 69.0 + 12.0 * np.log2(freq / 440.0)
        # (H, P) candidate fundamentals: peak as harmonic h of candidate
        cand = peak_midi[None, :] - self._harm_off[:, None]
        cand_round = np.round(cand)
        valid = (np.abs(cand - cand_round) <= CHROMA_TOL_ST) & \
            (cand_round >= CHROMA_F0_MIN_MIDI) & \
            (cand_round <= CHROMA_F0_MAX_MIDI)
        cm = cand_round.astype(np.int64)
        if CHROMA_REQUIRE_F0:
            # Harmonic corroboration (see CHROMA_REQUIRE_F0 above): keep a
            # candidate fundamental only where SOME peak of the same frame
            # matched it at h == 1 (row 0 of `cand` is the peak itself).
            span = CHROMA_F0_MAX_MIDI + 1
            key = np.where(valid, fi[None, :] * span + cm, -1)
            seen = np.zeros(n * span, dtype=bool)
            root = key[0][key[0] >= 0]
            if root.size:
                seen[root] = True
            valid = valid & (key >= 0) & seen[np.where(key >= 0, key, 0)]
        weights = self._harm_w[:, None] * b[None, :]
        pc = np.mod(cm, 12)
        flat_idx = (fi[None, :] * 12 + pc)[valid]
        np.add.at(out.reshape(-1), flat_idx, weights[valid])
        # register-tilted bass-band copy of the same votes (CHROMA_BASS_*)
        low_valid = valid & (cm >= CHROMA_BASS_MIN_MIDI) & \
            (cm <= CHROMA_BASS_MAX_MIDI)
        if low_valid.any():
            tilt = 0.5 ** ((cm - CHROMA_BASS_MIN_MIDI) / CHROMA_BASS_HALFLIFE_ST)
            np.add.at(out_low.reshape(-1), (fi[None, :] * 12 + pc)[low_valid],
                      (weights * tilt)[low_valid])
        return out, out_low

    def _attack(self, mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Banded, level-independent attack detection function (ATTACK_* above).

        ``mag`` is (n, n_bins).  Returns (n,) >= 0.  Vectorized over frames and
        bands; the only loop is over the ``ATTACK_BANDS`` band slices, whose
        widths differ so they cannot be one reduction.

        Carries ``ATTACK_LAG`` frames of band history across calls, so the
        result for frame k depends only on the stream, never on how it was
        blocked into ``process()`` calls.
        """
        n = mag.shape[0]
        band = np.empty((n, self._n_bands), dtype=np.float64)
        for b in range(self._n_bands):
            band[:, b] = mag[:, self._band_lo[b]:self._band_hi[b]].sum(axis=1)
        band = np.log1p(ATTACK_LAMBDA * band)
        if self._band_hist is None:                 # see __init__: first-frame seed
            self._band_hist = np.repeat(band[:1], ATTACK_LAG, axis=0)
        # reference = the frame ATTACK_LAG hops back, max-filtered over +/-
        # ATTACK_MAXFILT bands (SuperFlux: rejects a partial that merely slid
        # or beat between bands, keeps a genuine new attack).
        stacked = np.vstack((self._band_hist, band))
        prev = stacked[:n]                              # (n, n_bands)
        if ATTACK_MAXFILT > 0:
            m = ATTACK_MAXFILT
            pad = np.pad(prev, ((0, 0), (m, m)), mode="edge")
            prev = np.maximum.reduce(
                [pad[:, i:i + self._n_bands] for i in range(2 * m + 1)])
        self._band_hist = stacked[-ATTACK_LAG:].copy()
        rise = np.maximum(band - prev, 0.0)
        # Stream start: the first WIN/HOP windows are LEFT-PADDED WITH ZEROS
        # (see __init__), so their band levels are artificially low and the
        # frames after them read as a large rise that is an artifact of the
        # padding scrolling out, not of anything the human played.  Measured on
        # a take opened mid-sustain: frames 1 and 2 report 0.90 and 0.80 where
        # a real attack on the same material is 0.049 — an 18x phantom onset,
        # and it lands inside the tracker's 12-second autocorrelation window.
        # A frame is only comparable once both it AND its reference are fully
        # covered by real stream.
        n_warm = WIN // HOP + ATTACK_LAG
        if self._frame_idx < n_warm:
            k = min(n, n_warm - self._frame_idx)
            rise[:k] = 0.0
        # BREADTH: the fraction of bands that rose at all.  A knock on a guitar
        # body, a rim click or a hand clap is an impulse and lifts nearly every
        # band at once; a plucked or bowed note lifts the bands its harmonics
        # land in and leaves the rest alone.  That is the one property which
        # separates "the human is counting the band in" from "the human is
        # playing", and it is free here — the per-band rises are already
        # computed.  See ``CountInDetector``.
        breadth = (rise > ATTACK_BREADTH_EPS).mean(axis=1)
        return rise.sum(axis=1) / self._n_bands, breadth

    def _compute(self, mat: np.ndarray, cmat: np.ndarray) -> list[Frame]:
        """Compute features for consecutive frame windows.

        ``mat`` is (n, WIN) — the short window driving RMS / flux / onset /
        centroid.  ``cmat`` is (n, CHROMA_WIN) — the longer trailing window
        driving chroma only (see CHROMA_WIN).  Both end at the same sample.
        All reductions are per-row, so results are bitwise independent of how
        the stream was blocked into process() calls.
        """
        n = mat.shape[0]
        matf = mat.astype(np.float64)

        rms = np.sqrt(np.mean(matf * matf, axis=1))
        rms_db = np.where(
            rms > _RMS_FLOOR_LIN,
            20.0 * np.log10(np.maximum(rms, _RMS_FLOOR_LIN)),
            RMS_FLOOR_DB,
        )
        rms_db = np.maximum(rms_db, RMS_FLOOR_DB)

        mag = np.abs(np.fft.rfft(matf * self._win_fn, axis=1))
        logmag = np.log1p(mag)

        prev = np.vstack((self._prev_logmag[None, :], logmag[:-1, :]))
        diff = logmag - prev
        flux = np.where(diff > 0.0, diff, 0.0).sum(axis=1)
        self._prev_logmag = logmag[-1].copy()

        attack, breadth = self._attack(mag)

        cmag = np.abs(np.fft.rfft(cmat.astype(np.float64) * self._cwin_fn,
                                  axis=1))
        chroma_raw, chroma_low_raw = self._chroma(cmag)
        chroma_sum = chroma_raw.sum(axis=1)
        chroma_low_sum = chroma_low_raw.sum(axis=1)

        den = mag.sum(axis=1)
        num = (mag * self._freqs).sum(axis=1)
        centroid = np.where(
            den > _SILENT_MAG_EPS, num / np.where(den > 0.0, den, 1.0), 0.0
        )

        out: list[Frame] = []
        for i in range(n):
            fidx = self._frame_idx
            f = float(flux[i])
            a = float(attack[i])
            # RELATIVE threshold on the attack function (ATTACK_* above): a
            # multiple of the trailing median plus an absolute floor.  Because
            # ``attack`` is a log ratio this is gain-independent — the same two
            # constants hold for a -21 dBFS click and the same click 20 dB down,
            # which an absolute offset on ``flux`` provably could not do.
            med = float(np.median(self._flux_hist)) if self._flux_hist else 0.0
            threshold = med + self._onset_delta
            onset = f > threshold and (
                fidx - self._last_onset_frame >= self._min_sep_frames
            )
            if onset:
                self._last_onset_frame = fidx
            self._flux_hist.append(f)

            if chroma_sum[i] > _SILENT_MAG_EPS:
                chroma = (chroma_raw[i] / chroma_sum[i]).astype(np.float32)
            else:
                chroma = np.zeros(12, dtype=np.float32)
            # scaled by the FULL vote mass, not its own: chroma_low.sum() is
            # the low band's SHARE of pitched evidence (see CHROMA_BASS_*)
            if chroma_low_sum[i] > _SILENT_MAG_EPS \
                    and chroma_sum[i] > _SILENT_MAG_EPS:
                chroma_low = (chroma_low_raw[i] / chroma_sum[i]).astype(np.float32)
            else:
                chroma_low = np.zeros(12, dtype=np.float32)

            out.append(
                Frame(
                    t_s=(fidx + 1) * HOP / self._sr,
                    rms_db=float(rms_db[i]),
                    flux=f,
                    onset=bool(onset),
                    onset_strength=f if onset else 0.0,
                    chroma=chroma,
                    centroid_hz=float(centroid[i]),
                    chroma_low=chroma_low,
                    attack=a,
                    attack_breadth=float(breadth[i]),
                )
            )
            self._frame_idx += 1
        return out


# ===========================================================================
# COUNT-IN (v1.5, SPEC-V1 section 2 — "tap() is the count-in escape hatch")
# ===========================================================================

# A count-in event must be this many times the trailing median attack.  High on
# purpose: the point is to be unmistakable, not sensitive.
COUNTIN_MULT = 5.0
# Absolute floor, log-ratio units.  Set from the measurement and NOT from the
# first fixture, which is the interesting part: knocks over DIGITAL SILENCE
# reach an attack of 7.2-9.4 (a rise from true zero is an arbitrarily large log
# ratio), so a fixture built that way justified a floor of 6.5 and made the
# detector look far better than it is.  The same knocks over a -54 dBFS room
# bed — an ordinary quiet room on a laptop mic — reach only 1.5-3.6, which
# overlaps the loudest broadband frames in real material (fixture strums peak
# at 5.7, a 1kHz click at 7.8).  So the floor cannot be the discriminator; it
# is only a sanity gate, and breadth plus the quiet-start rule do the work.
# Measured at 1.5: 12/12 count-ins found across a 7.5x range of knock level,
# 2 false positives over 14 negative takes.  At 2.5 the false positives go UP
# (5) and the true positives collapse to 6/12 — fewer candidates means more
# apparent quiet, so a too-high floor is worse on BOTH axes.
COUNTIN_FLOOR = 1.5
COUNTIN_MEDIAN_S = 2.0        # trailing median window — longer than the beat
                              # picker's, because a count-in is judged against
                              # the whole texture, not the last third of a second
COUNTIN_BREADTH = 0.55        # fraction of bands that must rise together
COUNTIN_MIN_SEP_S = 0.15      # 400 bpm ceiling; also debounces one knock
# A count-in STARTS FROM QUIET.  Measured necessity: prominence + breadth +
# evenness alone fire on any percussive material, because a click track is a
# broadband impulse train and is therefore a count-in by every other test —
# 17 spurious confirmations on a 120bpm click, 24 on the calm strum fixture,
# 6 on a trap render.  Requiring the FIRST count to be preceded by this much
# time with no candidate at all takes every one of those to zero, because a
# groove's candidates never stop for 0.9s, and it matches what counting a band
# in actually is: something you do when the band is not already playing.
# Only the first count is gated — once a count-in is under way its own
# spacing may legitimately exceed this.
#
# 1.4s, not 0.9s, because of the ONE case 0.9 still let through: a 60bpm click
# track has a 1.0s inter-click gap, so at 0.9 every click qualified as a fresh
# start and the detector re-confirmed a "count-in" 15 times across the take.
# 1.4s is slower than any groove's event spacing and still admits a count-in
# down to COUNTIN_BPM_RANGE's 40bpm floor (a 1.5s spacing), because only the
# FIRST count is gated.
COUNTIN_GAP_S = 1.4
COUNTIN_TAPS = 4              # taps that make a count-in
COUNTIN_EVEN_FRAC = 0.22      # every IOI within this fraction of the median
COUNTIN_BPM_RANGE = (40.0, 200.0)


class CountInDetector:
    """Detect a human COUNTING THE BAND IN through a microphone.

    THE PROBLEM THIS SOLVES.  ``BeatTracker`` infers tempo from the audio, and
    on sustained open-tuned acoustic material that inference is sometimes
    genuinely underdetermined — a drone with irregular swells has no tempo to
    find, and the tracker correctly reports low confidence and stays out.  The
    escape hatch is the one every human band already uses: somebody counts.
    ``BeatTracker.tap()`` has always accepted that; nothing could reach it from
    a live microphone.  This can.

    WIRED LIVE IN TWO PLACES: ``BeatTracker(count_in=True)`` feeds ``tap()``
    from it, and ``plume lead --tap-start`` waits on one before anything plays.

    WHY IT DOES NOT FIRE WHILE SOMEBODY IS PLAYING — the whole design problem.
    Steady strumming is a train of evenly-spaced strong attacks, so prominence
    and evenness alone would confirm a "count-in" on every bar of ordinary
    playing and hard-latch the tracker's tempo and phase at confidence 1.0.
    The discriminator is ``Frame.attack_breadth``: a count-in is knocked,
    clapped or clicked, and an impulse lifts nearly every band at once, while a
    plucked or strummed note lifts the bands its harmonics land in and leaves
    the rest alone.  A candidate must therefore be

      * PROMINENT   — ``attack`` above ``COUNTIN_MULT`` x the trailing 2s
                      median and above ``COUNTIN_FLOOR``;
      * BROADBAND   — ``attack_breadth`` >= ``COUNTIN_BREADTH``;
      * EVEN        — ``COUNTIN_TAPS`` of them whose inter-onset intervals are
                      all within ``COUNTIN_EVEN_FRAC`` of their median, at a
                      sane tempo.

    Every candidate goes in the buffer, so anything strong and broadband
    happening BETWEEN two counts breaks the evenness test rather than being
    ignored — a count-in has to be clean to be accepted, which is the correct
    bias for something that overrides the tracker outright.

    USAGE — realtime.py owns the wiring, and this is the whole of it::

        count_in = CountInDetector(sr=sr)
        ...
        for i, t in enumerate(count_in.process(frames)):
            tracker.tap(t, beat_in_bar=i % 4)

    ``process`` returns the tap times of a count-in confirmed by this block, in
    order, and an empty list otherwise.  Times are ``Frame.t_s`` — stream
    seconds — which is the clock ``BeatTracker`` already runs on, so they go
    to ``tap()`` unmodified.  Passing ``beat_in_bar`` makes the LAST count
    beat 4 of the bar, so the band comes in on the next beat (see
    ``BeatTracker.tap``).  The buffer clears on confirmation, so a second
    count-in needs a second set of taps.

    A keyboard or OSC count-in needs no part of this class: call
    ``BeatTracker.tap(t_s, beat_in_bar=...)`` directly with a stream-clock
    timestamp.

    Deterministic: a pure function of the frame stream.  No rng, no clock.
    """

    def __init__(self, sr: int = 48000, *, mult: float = COUNTIN_MULT,
                 floor: float = COUNTIN_FLOOR,
                 breadth: float = COUNTIN_BREADTH,
                 taps: int = COUNTIN_TAPS,
                 even_frac: float = COUNTIN_EVEN_FRAC,
                 bpm_range: tuple[float, float] = COUNTIN_BPM_RANGE,
                 waiting: bool = False):
        """``waiting``: nothing is playing yet and nothing happens until a count
        arrives (``lead --tap-start``), so a false count costs little and a missed
        one leaves her knocking at silence.  Any four even knocks count, with no
        quiet start required, because a real room never gave one."""
        if sr <= 0:
            raise ValueError(f"sr must be positive, got {sr}")
        if taps < 2:
            raise ValueError(f"taps must be >= 2, got {taps}")
        if not (0.0 < bpm_range[0] < bpm_range[1]):
            raise ValueError(f"bad bpm_range {bpm_range}")
        self._mult = float(mult)
        self._floor = float(floor)
        self._breadth = float(breadth)
        self._taps = int(taps)
        self._even = float(even_frac)
        self._lo, self._hi = float(bpm_range[0]), float(bpm_range[1])
        self._waiting = bool(waiting)
        n = max(1, int(round(COUNTIN_MEDIAN_S * sr / HOP)))
        self._hist: deque[float] = deque(maxlen=n)
        self._cands: list[float] = []
        self._last_t = -math.inf        # last candidate, accepted or not
        self._t0: float | None = None   # first frame seen (the quiet reference)

    @property
    def pending(self) -> tuple[float, ...]:
        """Candidate tap times not yet forming a count-in (diagnostics)."""
        return tuple(self._cands)

    def reset(self) -> None:
        """Forget pending candidates (the level history is kept)."""
        self._cands.clear()

    def process(self, frames) -> list[float]:
        """Feed frames; return a confirmed count-in's tap times, else ``[]``."""
        out: list[float] = []
        for f in frames:
            a = float(getattr(f, "attack", 0.0))
            b = float(getattr(f, "attack_breadth", 0.0))
            t = float(f.t_s)
            if self._t0 is None:
                self._t0 = t
            # judged against the level BEFORE this frame, so a count-in cannot
            # raise the bar it has to clear
            med = float(np.median(self._hist)) if self._hist else 0.0
            self._hist.append(a)
            if a < max(self._mult * med, self._floor) or b < self._breadth:
                continue
            if t - self._last_t < COUNTIN_MIN_SEP_S:
                continue
            # every candidate advances the quiet reference, including one that
            # is refused as a starter — otherwise a continuous groove would
            # accumulate "quiet" between its own events and eventually arm.
            since = t - (self._last_t if math.isfinite(self._last_t)
                         else self._t0)
            self._last_t = t
            if not self._waiting and not self._cands and since < COUNTIN_GAP_S:
                continue                    # a count-in starts from quiet
            self._cands.append(t)
            if self._waiting:
                self._cands = self._cands[-self._taps:]
            if len(self._cands) < self._taps:
                continue
            # NO SLIDING WINDOW.  The count-in must be the FIRST ``taps``
            # candidates after the quiet start; if they are not even, the
            # attempt is abandoned and a fresh quiet start is required.  A
            # sliding window instead re-tests every consecutive four events for
            # the rest of the take, which is a great many chances for ordinary
            # playing to line up by accident — and the cost of it lining up is
            # a count-in LATCH: tempo and phase hard-set at confidence 1.0,
            # which the tracker then defends against the audio.
            taps = list(self._cands)
            if not self._waiting:
                self._cands = []
            iois = np.diff(np.asarray(taps, dtype=np.float64))
            m = float(np.median(iois))
            if not (60.0 / self._hi <= m <= 60.0 / self._lo):
                continue
            if not bool(np.all(np.abs(iois - m) <= self._even * m)):
                continue
            out = taps
            break
        return out
