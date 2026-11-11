"""Sustained open-tuned acoustic guitar — the material v1 was never tuned on.

WHY THIS FILE EXISTS.  Everything in ``tests/fixtures_gen.py`` and every
programmed render in ``tests/test_real_material.py`` shares one property:
SHARP TRANSIENTS.  ``pluck`` notes have a 2ms attack and a 250ms decay, 808s
and hats are impulses, click tracks are impulses.  The first session with a
real human was none of those — a fingerpicked acoustic guitar in CGCFCE (an
open tuning: three C's, everything rings, decays overlap for seconds, and a
flesh-plucked note has no broadband click at all), captured on a laptop mic
with plume's own output bleeding back in.  plume's status line over that take:

    167.9 -> 163.0 -> 169.2 -> 165.4 -> 80.6 -> 80.3 -> 121.0 bpm
    confidence 0.41-0.62 for the whole session

and the human's verdict was "it only plays like two random measures here and
there, and then stops suddenly".  Those numbers are not noise.  165 and 80 are
an octave apart, and confidence sitting at 0.41-0.62 straddles BOTH gates the
sideman keys off (``MindParams.entry_conf`` 0.45 and ``BeatTracker`` 's
``lock_conf`` 0.55), so the machine armed, played a bar, disarmed, and re-armed
later — for the whole take.

This suite synthesizes that material and holds the tracker to it.  The fixture
is deliberately built to have the three properties the existing corpus lacks:

* **soft attacks** — 11-22ms raised-cosine, so the transient is spread over one
  and a half analysis hops instead of landing inside one;
* **overlapping sustain** — 2.4-3.2s decays at 72-84bpm, so three to four notes
  are always ringing under the one being played;
* **accent as the only metre cue** — no percussion, no note-length cue.  The
  fingerpicked pattern puts a loud bass string on the beat and a quiet upper
  string on the "and"; the strummed one alternates a full six-string down-strum
  with a quiet three-string up-strum.  If accent does not survive the sustain,
  the only thing left is the subdivision, and the tracker locks an octave high.

TWO MODES, AND WHICH HALF OF THIS FILE TESTS WHICH.  The fix for the material
above — the banded, level-independent ``Frame.attack`` envelope — measurably
COSTS the programmed corpus, so it is opt-in (``BeatTracker(acoustic=True)`` /
``PLUME_ACOUSTIC=1``) and the corpus keeps the default.  ``beat.py``'s ACOUSTIC
MODE block carries the full measured A/B.  Accordingly:

* the ACOUSTIC-MATERIAL tests below construct ``BeatTracker(acoustic=True)``;
* the NON-REGRESSION tests construct ``BeatTracker(acoustic=False)`` — the
  shipping default — and cover the programmed corpus at ITS OWN gain and
  length, including the exact ``lofi80_quiet`` case (32 bars at -20 dBFS) that
  ``tests/test_real_material.py`` runs, because an earlier revision of this
  file guarded only 16-bar renders and a -20 dB 32-bar drift to 93.0 bpm
  shipped straight through it;
* both are passed explicitly rather than left to default, so no test in this
  file depends on whether ``PLUME_ACOUSTIC`` is set in the caller's shell;
* ``test_the_mode_flag_is_load_bearing_in_both_directions`` proves neither
  half is passing for free.

THE COUNT-IN HATCH IS NOT GATED and is tested in BOTH modes: it is the
pragmatic escape for exactly the material that defeats the tracker, so it has
to work whichever envelope the tracker is running.

Determinism: every fixture is a pure function of (kind, bpm, bars, seed) — one
``np.random.default_rng``, no wall clock, no globals.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # fixtures_gen

from plume.listen import features as feat                       # noqa: E402
from plume.listen.beat import BeatTracker                       # noqa: E402
from plume.listen.features import (                             # noqa: E402
    CountInDetector, FeatureExtractor,
)
from plume.mind import MindParams                               # noqa: E402
from plume.theory import midi_to_hz                             # noqa: E402

SR = 48000
BLOCK = 4800

# CGCFCE, low to high — the tuning Juniper actually played.  Three C's, an open
# G, an F and an E: every string rings and nothing damps anything else.
CGCFCE = (36, 43, 48, 53, 60, 64)

# fret shapes over the open tuning.  Only one or two strings move; the C's keep
# droning, which is what an open tuning is for and is exactly what makes the
# harmony — and the transients — mush together.
_SHAPES = (
    {},                              # all open
    {3: 2, 4: 2},
    {1: 2, 3: 2},
    {0: 3, 3: 2, 4: 3},
)

ACOUSTIC_KINDS = ("fingerpick", "strum", "drone")
# the tempo each kind is generated at by default, and whether it HAS one
ACOUSTIC_TRUTH = {"fingerpick": 84.0, "strum": 72.0, "drone": 0.0}


# --------------------------------------------------------------- the fixture

def _string(midi: float, *, dur_s: float, velocity: float,
            rng: np.random.Generator, attack_ms: float = 14.0,
            tau_s: float = 2.2, n_harm: int = 14) -> np.ndarray:
    """One steel string, plucked and left ringing.

    Additive rather than a ``voices.py`` preset on purpose: ``pluck`` is a saw
    through a filter envelope with a 2ms attack, i.e. the sharp-transient
    material the tracker was already tuned on.  What is needed here is the
    opposite, and it needs three things ``pluck`` cannot express — a settable
    soft attack, a multi-second decay, and PARTIALS THAT DIE FIRST (decay
    tau/h^0.8), which is why a fingerpicked note is bright for 40ms and then
    two seconds of near-sine drone.  That last property is the whole
    difficulty: the only broadband part of the event is over before the onset
    detector's hop has finished.
    """
    n = max(1, int(round(dur_s * SR)))
    t = np.arange(n, dtype=np.float64) / SR
    f0 = float(midi_to_hz(midi))
    h = np.arange(1, n_harm + 1, dtype=np.float64)
    freqs = f0 * h * np.sqrt(1.0 + 1.2e-4 * h * h)      # string inharmonicity
    keep = freqs < 0.45 * SR
    freqs, h = freqs[keep], h[keep]
    amps = 1.0 / h ** 1.15
    taus = tau_s / h ** 0.8
    phi = rng.uniform(0.0, 2.0 * np.pi, size=h.size)
    y = (amps[:, None] * np.exp(-t[None, :] / taus[:, None])
         * np.sin(2.0 * np.pi * freqs[:, None] * t[None, :]
                  + phi[:, None])).sum(axis=0)
    na = min(n, max(2, int(round(attack_ms / 1000.0 * SR))))
    u = np.linspace(0.0, 1.0, na, endpoint=False)
    y[:na] *= 0.5 * (1.0 - np.cos(np.pi * u))
    return y * float(velocity) / float(amps.sum())


def _add(buf: np.ndarray, y: np.ndarray, t_s: float) -> None:
    s0 = int(round(t_s * SR))
    if s0 < 0:
        y, s0 = y[-s0:], 0
    n = min(y.shape[0], buf.shape[0] - s0)
    if n > 0:
        buf[s0:s0 + n] += y[:n]


def make_acoustic(kind: str, *, bpm: float = 84.0, bars: int = 20,
                  seed: int = 7, gain_db: float = -20.0
                  ) -> tuple[np.ndarray, dict]:
    """Sustained open-tuned acoustic guitar, mono float32 @ 48k.

    Kinds
    -----
    - ``fingerpick`` : 8th-note Travis-ish pattern, a LOUD bass string on the
      beat and a quiet upper string on the "and".  The metre is carried by that
      accent alone, so a detector that flattens accent reads the 8ths as the
      beat and reports 2x the tempo.  This is the 167.9-for-a-true-84 case.
    - ``strum``      : one strum per beat, full six-string down-strum on 1 and
      3, quiet three-string up-strum on 2 and 4, raked over 30-48ms, ringing
      3.2s.
    - ``drone``      : GENUINELY PULSE-FREE.  Irregular swells with 60-240ms
      attacks at random 0.55-2.4s spacing.  There is no tempo to find, and the
      correct behaviour is to say so — this is the fixture that tests whether
      an honestly uncertain tracker reads LOW rather than 0.5.

    ``gain_db`` peak-normalizes to an ordinary live-mic level (-20 dBFS by
    default) rather than the 0.85 the other fixture families use.

    Returns ``(audio, truth)``; ``truth["bpm"]`` is 0.0 for ``drone``.
    """
    if kind not in ACOUSTIC_KINDS:
        raise ValueError(f"unknown kind {kind!r} (have: {list(ACOUSTIC_KINDS)})")
    if bars < 1:
        raise ValueError(f"bars must be >= 1, got {bars}")
    rng = np.random.default_rng(seed)
    beat = 60.0 / float(bpm)
    n_beats = bars * 4
    buf = np.zeros(int(round((n_beats * beat + 4.0) * SR)), dtype=np.float64)
    events: list[float] = []

    def shape(b: int) -> list[int]:
        s = _SHAPES[(b // 2) % len(_SHAPES)]
        return [m + s.get(i, 0) for i, m in enumerate(CGCFCE)]

    if kind == "fingerpick":
        order = [0, 4, 2, 5, 1, 4, 2, 3]          # string index per 8th
        for b in range(bars):
            tones = shape(b)
            for k in range(8):
                t = (b * 4 + k * 0.5) * beat + float(rng.normal(0.0, 0.009))
                si = order[k]
                on_beat = k % 2 == 0
                _add(buf, _string(
                    tones[si], dur_s=2.6,
                    velocity=(0.85 if on_beat else 0.55)
                    * float(rng.uniform(0.85, 1.1)),
                    rng=rng, attack_ms=float(rng.uniform(11.0, 18.0)),
                    tau_s=2.4 if si < 2 else 1.8), t)
                events.append(t)

    elif kind == "strum":
        for b in range(bars):
            tones = shape(b)
            for f in (0.0, 1.0, 2.0, 3.0):
                down = (f % 2.0) == 0.0
                t0 = (b * 4 + f) * beat + float(rng.normal(0.0, 0.012))
                rake = float(rng.uniform(0.030, 0.048))
                order = list(range(6)) if down else list(range(5, 2, -1))
                for i, si in enumerate(order):
                    _add(buf, _string(
                        tones[si], dur_s=3.2,
                        velocity=(0.85 if down else 0.42) * (0.96 ** i)
                        * float(rng.uniform(0.9, 1.05)),
                        rng=rng, attack_ms=float(rng.uniform(13.0, 22.0)),
                        tau_s=3.0 if si < 3 else 2.2),
                        t0 + rake * i / max(1, len(order) - 1))
                events.append(t0)

    else:                                          # drone
        t, i = 0.5, 0
        while t < n_beats * beat:
            tones = shape(i // 3)
            _add(buf, _string(
                tones[int(rng.integers(0, 6))], dur_s=4.0,
                velocity=float(rng.uniform(0.5, 0.9)), rng=rng,
                attack_ms=float(rng.uniform(60.0, 240.0)), tau_s=3.4), t)
            events.append(t)
            t += float(rng.uniform(0.55, 2.4))
            i += 1

    peak = float(np.max(np.abs(buf))) or 1.0
    buf *= (10.0 ** (gain_db / 20.0)) / peak
    truth = {
        "kind": kind, "bars": bars, "sr": SR,
        "bpm": 0.0 if kind == "drone" else float(bpm),
        "beat_times_s": [i * beat for i in range(n_beats)],
        "event_times_s": sorted(events),
    }
    return buf.astype(np.float32), truth


def knock(rng: np.random.Generator, *, level: float = 0.5,
          dur_s: float = 0.05) -> np.ndarray:
    """A knock on the guitar body: broadband impulse + a short 180Hz ring.

    This is what a human uses to count a band in when their hands are already
    on the instrument, and it is the signal ``CountInDetector`` looks for.
    """
    n = int(dur_s * SR)
    t = np.arange(n) / SR
    y = rng.standard_normal(n) * np.exp(-t / 0.004)
    y += 0.6 * np.sin(2.0 * np.pi * 180.0 * t) * np.exp(-t / 0.020)
    return (level * y / max(float(np.max(np.abs(y))), 1e-9)).astype(np.float32)


def prepend_count_in(audio: np.ndarray, bpm: float, *, taps: int = 4,
                     level: float = 0.5, lead_s: float = 2.0,
                     room_db: float = -54.0,
                     seed: int = 3) -> tuple[np.ndarray, list[float]]:
    """``taps`` knocks at ``bpm`` over a room-noise bed, then ``audio``.

    ``room_db`` matters and is not decoration.  An earlier version of this
    helper put the knocks over DIGITAL SILENCE, which made the detector look
    far better than it is: a rise from true zero is an arbitrarily large log
    ratio, so any threshold passes.  A real room has a floor, the knock's rise
    above it is finite, and that finite number is what the detector has to
    work with.  -54 dBFS is an ordinary quiet room on a laptop mic.
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    n = int((lead_s + taps * beat) * SR)
    pre = (rng.standard_normal(n) * (10.0 ** (room_db / 20.0))
           ).astype(np.float32)
    times = []
    for i in range(taps):
        t = lead_s + i * beat
        k = knock(rng, level=level)
        s0 = int(t * SR)
        pre[s0:s0 + k.size] += k
        times.append(t)
    return np.concatenate([pre, audio]), times


# ------------------------------------------------------------------ harness

class Trace:
    """One take, run through the real FeatureExtractor + BeatTracker.

    ``acoustic`` selects the tracker's front end and is passed EXPLICITLY
    (never left as ``None``), so a stray ``PLUME_ACOUSTIC`` in the caller's
    shell cannot change what this suite measures.  ``False`` is the shipping
    default and is what the non-regression half runs on; ``True`` is the
    sustained-acoustic mode the material at the top of this file needs.
    """

    def __init__(self, audio, *, acoustic=False, count_in=False, tracker=None,
                 block=BLOCK):
        self.fx = FeatureExtractor(sr=SR)
        self.bt = (tracker if tracker is not None
                   else BeatTracker(acoustic=bool(acoustic)))
        self.det = CountInDetector(sr=SR) if count_in else None
        self.taps: list[float] = []
        self.t, self.bpm, self.conf, self.locked = [], [], [], []
        self.frames = []
        for i in range(0, len(audio), block):
            fs = self.fx.process(audio[i:i + block])
            if not fs:
                continue
            self.frames.extend(fs)
            if self.det is not None:
                for j, tt in enumerate(self.det.process(fs)):
                    self.bt.tap(tt, beat_in_bar=j % 4)
                    self.taps.append(tt)
            st = self.bt.process(fs)
            self.t.append(st.t_s)
            self.bpm.append(st.bpm)
            self.conf.append(st.confidence)
            self.locked.append(bool(st.locked))
        self.t = np.asarray(self.t)
        self.bpm = np.asarray(self.bpm)
        self.conf = np.asarray(self.conf)
        self.state = st

    def after(self, t_s):
        """Samples from ``t_s`` on (the 'has it settled and does it hold' view)."""
        m = self.t >= t_s
        return self.bpm[m], self.conf[m], np.asarray(self.locked)[m]


@pytest.fixture(scope="module")
def fingerpick():
    return make_acoustic("fingerpick", bpm=84.0, bars=20)


@pytest.fixture(scope="module")
def strum():
    return make_acoustic("strum", bpm=72.0, bars=20)


@pytest.fixture(scope="module")
def drone():
    return make_acoustic("drone", bpm=84.0, bars=20)


# ============================================================ tempo stability

# The take is 61s; the tracker is allowed this long to settle.  Measured
# time-to-lock in ACOUSTIC MODE — which is what these tests run in — is 9.4s
# (fingerpick) and 12.4s (strum), so 24s is a real bound and not a rubber
# stamp.  (On the default front end the same fixtures lock at 4.7s and 24.3s,
# but at the wrong tempo and the wrong confidence respectively; see the A/B in
# beat.py.)
SETTLE_S = 24.0


def test_fingerpick_locks_the_true_tempo_not_its_octave(fingerpick):
    """84bpm, and NOT the 168 the shipped tracker reported at confidence 0.83.

    This is the regression guard for the live take's 167.9/163.0/169.2/165.4.
    The fixture plays 8th notes with the accent on the beat; ``flux`` ranks the
    8th-note lag ABOVE the beat lag (measured +0.755 vs +0.725), so the default
    envelope has positive evidence for the wrong answer — which is why this
    test runs in ACOUSTIC MODE and why
    ``test_the_mode_flag_is_load_bearing_in_both_directions`` pins the default
    still reading 167.9 here.
    """
    audio, truth = fingerpick
    tr = Trace(audio, acoustic=True)
    bpm, _conf, locked = tr.after(SETTLE_S)
    assert locked.mean() > 0.9, "should be locked once settled"
    err = np.abs(np.log2(bpm / truth["bpm"]))
    assert np.median(err) < 0.02, (
        f"median tempo {np.median(bpm):.1f} vs truth {truth['bpm']}")
    # no sample anywhere near the octave, in either direction
    assert not np.any(np.abs(err - 1.0) < 0.15), (
        f"octave excursion: {sorted(set(np.round(bpm[np.abs(err - 1.0) < 0.15], 1)))}")


def test_fingerpick_holds_the_grid_for_the_whole_take(fingerpick):
    """No octave jump and no unrelated tempo, for every sample after settling.

    "Unrelated" is the third failure mode in the live trace: 121.0 bpm is
    neither 165 nor 80.  Anything more than 4% off the truth and not an octave
    of it is that failure.
    """
    audio, truth = fingerpick
    tr = Trace(audio, acoustic=True)
    bpm, _c, _l = tr.after(SETTLE_S)
    within = np.abs(bpm - truth["bpm"]) / truth["bpm"] <= 0.04
    assert within.all(), (
        f"{(~within).sum()}/{within.size} samples off-grid: "
        f"{sorted(set(np.round(bpm[~within], 1)))}")
    # The tracker's FIRST reading is still 167.9 — a fingerpicked 8th-note
    # pattern really does look like 168 for the first couple of bars, and the
    # v1.3 octave hysteresis is what lets an UNSETTLED octave be corrected
    # (measured: it re-reads 167.9 -> 83.9 at t=9.5s).  What must not happen is
    # a re-decision once the grid has settled, which is the whole complaint.
    late = [e for e in tr.bt.events if e["t_s"] >= SETTLE_S]
    assert late == [], f"grid re-decided after settling: {late}"


def test_strum_confidence_clears_both_gates(strum):
    """The take that produced "two random measures here and there".

    The complaint is not that the tempo was wrong — it was roughly right — but
    that confidence sat at 0.41-0.62, i.e. ON both ``MindParams.entry_conf``
    (0.45) and ``lock_conf`` (0.55), so the sideman armed and disarmed on every
    re-evaluation.  Measured on the DEFAULT front end: median 0.544, with 100%
    of the settled take inside [0.40, 0.65].  In ACOUSTIC MODE: median 0.823,
    0%.  (Both re-measured 2026-08-19; the "0.52 / 79% / 0.83 / 7%" an earlier
    revision quoted was from a build where the mode was not yet a switch.)
    """
    audio, truth = strum
    tr = Trace(audio, acoustic=True)
    bpm, conf, locked = tr.after(SETTLE_S)
    p = MindParams()
    assert np.median(np.abs(bpm - truth["bpm"]) / truth["bpm"]) <= 0.04
    assert np.median(conf) > 0.70, f"median confidence {np.median(conf):.3f}"
    assert np.median(conf) > p.entry_conf + 0.15
    straddle = float(np.mean((conf >= 0.40) & (conf <= 0.65)))
    assert straddle < 0.30, f"{straddle:.2f} of the take straddles the gates"
    assert locked.mean() > 0.85


def test_ambiguous_material_reads_low_not_mid(drone):
    """A tracker that is honestly uncertain must read LOW, because the gates.

    ``drone`` has no pulse by construction — irregular swells with 60-240ms
    attacks.  There is no right tempo, so the only correct behaviours are a low
    confidence and no lock.  Reading 0.5 here is worse than reading 0.0: 0.5
    is the value that makes the sideman flap.
    """
    audio, truth = drone
    assert truth["bpm"] == 0.0
    tr = Trace(audio, acoustic=True)
    p = MindParams()
    assert not any(tr.locked), "must never lock on pulse-free material"
    assert np.median(tr.conf) < p.entry_conf, (
        f"median confidence {np.median(tr.conf):.3f} >= entry gate "
        f"{p.entry_conf}")
    assert np.percentile(tr.conf, 90) < p.entry_conf + 0.05


@pytest.mark.parametrize("kind,bpm", [("fingerpick", 84.0), ("strum", 72.0)])
def test_lock_survives_a_full_take_without_re_deciding(kind, bpm):
    """Once settled, the grid must not be re-decided — a musician does not
    re-decide whether they are in the band every three seconds."""
    audio, truth = make_acoustic(kind, bpm=bpm, bars=20)
    tr = Trace(audio, acoustic=True)
    _b, _c, locked = tr.after(SETTLE_S)
    # count lock -> unlock transitions after settling
    flips = int(np.sum(locked[1:] != locked[:-1]))
    assert flips <= 1, f"{flips} lock/unlock flips after settling"


# ================================================== the attack feature itself

def test_attack_is_gain_invariant(fingerpick):
    """``attack`` is a log RATIO, so scaling the input must not move it.

    This is the property that lets the tracker's threshold be relative, and it
    is what the old absolute ``ONSET_DELTA`` on ``flux`` could not have: the
    same material 20 dB down thinned flux onsets from ~10/s to ~2.8/s (a
    documented v1.2 limit).  ``log1p(|X|)`` is only logarithmic where |X| >> 1;
    ``log1p(ATTACK_LAMBDA*|X|)`` is logarithmic from about -110 dBFS up.
    """
    audio, _ = fingerpick
    a = np.asarray([f.attack for f in _frames(audio)])
    b = np.asarray([f.attack for f in _frames(audio * 0.1)])   # -20 dB
    m = a > 0.02                                   # where there is an attack
    assert m.sum() > 100
    rel = np.abs(b[m] - a[m]) / a[m]
    assert float(np.median(rel)) < 0.02, f"median drift {np.median(rel):.4f}"


def test_attack_does_not_depend_on_blocking(fingerpick):
    """Framing must be a pure function of the stream, not of how it arrived.

    ``attack`` carries ``ATTACK_LAG`` frames of band history across
    ``process()`` calls precisely so this holds.
    """
    audio, _ = fingerpick
    audio = audio[: 20 * SR]
    a = [f.attack for f in _frames(audio, block=4800)]
    b = [f.attack for f in _frames(audio, block=1111)]
    n = min(len(a), len(b))
    assert n > 500
    np.testing.assert_allclose(a[:n], b[:n], rtol=0, atol=1e-12)


def test_attack_has_no_startup_spike(fingerpick):
    """The band history is seeded from the first frame, not from silence.

    Seeded with zeros, frame 0's reference is digital silence and its attack
    value is the entire log-magnitude — measured at ~200x a real attack.  One
    such sample inside the tracker's 12s autocorrelation window delayed lock by
    almost exactly ACF_WINDOW_S on plume's own renders (trap 3.8s -> 16.3s).
    """
    audio, _ = fingerpick
    a_full = np.asarray([f.attack for f in _frames(audio)])
    assert a_full[0] == 0.0, "frame 0 must reference itself, not silence"

    # The load-bearing case: a stream that OPENS MID-SUSTAIN.  There is no
    # transient at t=0, so an honest detector reports nothing there.  Seeded
    # with zeros it instead reports the whole log-magnitude of whatever is
    # ringing — the largest value in the take, sitting inside the tracker's
    # 12-second autocorrelation window.
    mid = audio[int(31.4 * SR):]                # lands between two events
    a_mid = np.asarray([f.attack for f in _frames(mid)])
    typical = float(np.percentile(a_mid[a_mid > 0.02], 50))
    assert typical < 1.0                        # a real attack here is ~0.1
    assert float(a_mid[:4].max()) < 2.0 * typical, (
        f"startup spike {a_mid[:4].max():.3f} against a typical attack of "
        f"{typical:.3f}")


def test_attack_and_flux_are_independent_features(fingerpick):
    """``flux``/``onset``/``onset_strength`` are unchanged by the v1.5 work.

    ``mind`` and ``roles`` read onset DENSITY off ``Frame.onset`` and their
    thresholds are calibrated to the flux picker; the tracker reads ``attack``.
    The two must not have been quietly merged.
    """
    audio, _ = fingerpick
    fr = _frames(audio)
    for f in fr:
        if f.onset:
            assert f.onset_strength == f.flux
        else:
            assert f.onset_strength == 0.0
    assert any(f.attack > 0.0 for f in fr)


def _frames(audio, block=BLOCK):
    fx = FeatureExtractor(sr=SR)
    out = []
    for i in range(0, len(audio), block):
        out.extend(fx.process(audio[i:i + block]))
    return out


# ======================================================== count-in (the hatch)

# The hatch must work in EITHER front end — that is the whole point of it, and
# it is why ``CountInDetector`` reads ``Frame.attack``/``attack_breadth``
# unconditionally while the TRACKER's envelope is a switch.  Every count-in
# test below is therefore run in both modes.
MODES = [pytest.param(False, id="default"), pytest.param(True, id="acoustic")]


@pytest.mark.parametrize("acoustic", MODES)
def test_count_in_sets_tempo_and_downbeat_from_four_taps(acoustic):
    """Four taps hard-set the grid; ``beat_in_bar`` puts the band on the next
    beat rather than a bar later."""
    bt = BeatTracker(acoustic=acoustic)
    bpm, beat = 96.0, 60.0 / 96.0
    for i in range(4):
        bt.tap(1.0 + i * beat, beat_in_bar=i % 4)
    st = bt.process([])
    assert st.bpm == pytest.approx(bpm, rel=1e-9)
    assert st.locked and st.confidence == pytest.approx(1.0)
    assert st.beat_in_bar == 3, "the 4th count is beat 4 of the bar"
    # ... so the very next beat is the downbeat
    assert st.next_downbeat_t_s == pytest.approx(1.0 + 4 * beat, abs=1e-9)


@pytest.mark.parametrize("acoustic", MODES)
def test_count_in_default_keeps_pre_v15_phase(acoustic):
    """``beat_in_bar`` defaults to 0 — the behaviour every existing tap test
    was written against."""
    bt = BeatTracker(acoustic=acoustic)
    beat = 0.5
    for i in range(4):
        bt.tap(1.0 + i * beat)
    st = bt.process([])
    assert st.beat_in_bar == 0
    assert st.bpm == pytest.approx(120.0)


@pytest.mark.parametrize("acoustic", MODES)
def test_count_in_detected_through_a_microphone(drone, acoustic):
    """The whole point: material with NO inferable tempo, counted in by hand.

    ``drone`` is pulse-free, so the tracker cannot and must not guess.  Four
    knocks at 84bpm ahead of it must set the grid exactly and hold it.
    """
    audio, _ = drone
    au, taps = prepend_count_in(audio, 84.0, level=0.5)
    tr = Trace(au, acoustic=acoustic, count_in=True)
    assert len(tr.taps) == 4, f"count-in not detected (got {tr.taps})"
    err = np.abs(np.asarray(tr.taps) - np.asarray(taps))
    assert err.max() < 0.035, f"tap timing error {err.max()*1000:.1f}ms"
    assert tr.state.locked
    assert tr.state.bpm == pytest.approx(84.0, rel=0.04)
    assert tr.state.confidence >= 0.70


@pytest.mark.parametrize("acoustic", MODES)
def test_a_count_in_holds_for_the_whole_take_against_the_audio(acoustic):
    """AND IT HOLDS.  A hatch that sets the grid and then loses it is not a
    hatch — that is the "taps set bpm 80 conf 1.0, then the tracker drifts back
    off" failure the v1.2 tap latch exists for, and it has to stay fixed in
    BOTH front ends because the ACF alias underneath is still there either way.

    ``fingerpick`` is the adversarial case on purpose: counted in at its true
    84bpm, its own 8th-note subdivision (168) is exactly what the default
    envelope prefers, so the audio is actively arguing for an octave the whole
    take.  The count-in must win every sample of it, and the tapped tempo must
    stay inside the ``drift_pct`` band around the COUNT-IN rather than random
    walking away from it.
    """
    audio, _ = make_acoustic("fingerpick", bpm=84.0, bars=16)
    au, taps = prepend_count_in(audio, 84.0, level=0.5)
    tr = Trace(au, acoustic=acoustic, count_in=True)
    assert len(tr.taps) == 4
    after = tr.t >= taps[-1] + 2.0
    assert after.sum() > 100
    bpm, conf, locked = tr.bpm[after], tr.conf[after], np.asarray(tr.locked)[after]
    assert np.all(np.abs(bpm - 84.0) / 84.0 <= 0.031), (
        f"count-in tempo drifted to {sorted(set(np.round(bpm, 1)))}")
    assert locked.all(), "the count-in lock did not hold"
    assert conf.min() >= 0.70, f"confidence fell to {conf.min():.3f} under a tap"


@pytest.mark.parametrize("acoustic", MODES)
@pytest.mark.parametrize("level", [0.6, 0.3, 0.15])
def test_count_in_works_across_knock_levels(level, acoustic):
    """A 4x range of how hard the human knocks."""
    audio, _ = make_acoustic("fingerpick", bpm=84.0, bars=6)
    au, taps = prepend_count_in(audio, 84.0, level=level)
    tr = Trace(au, acoustic=acoustic, count_in=True)
    assert len(tr.taps) == 4
    assert np.max(np.abs(np.asarray(tr.taps) - np.asarray(taps))) < 0.035


@pytest.mark.parametrize("acoustic", MODES)
@pytest.mark.parametrize("kind,bpm", [("fingerpick", 84.0), ("strum", 72.0),
                                      ("drone", 84.0)])
def test_count_in_does_not_fire_on_ordinary_playing(kind, bpm, acoustic):
    """The failure that would matter most: a false count-in HARD-SETS tempo and
    phase at confidence 1.0 and latches.  Steady strumming is a train of even,
    strong attacks and passes every test except breadth and the quiet start.
    """
    audio, _ = make_acoustic(kind, bpm=bpm, bars=16)
    tr = Trace(audio, acoustic=acoustic, count_in=True)
    assert tr.taps == [], f"false count-in at {tr.taps}"


@pytest.mark.parametrize("acoustic", MODES)
def test_count_in_does_not_fire_on_a_click_track(acoustic):
    """A click train is broadband and perfectly even — a count-in by every test
    except that it never stops.  Measured: 17 false confirmations before the
    quiet-start rule, 0 after."""
    spb = 0.5
    x = np.zeros(int(40 * SR), dtype=np.float32)
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    w = (np.exp(-t * 90.0) * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
    for b in range(80):
        i = int(round(b * spb * SR))
        x[i:i + n] += 0.09 * w
    tr = Trace(x, acoustic=acoustic, count_in=True)
    assert tr.taps == [], f"false count-in at {tr.taps}"


def test_count_in_detector_is_deterministic(drone):
    audio, _ = drone
    au, _ = prepend_count_in(audio, 84.0)
    a = Trace(au, count_in=True).taps
    b = Trace(au, count_in=True).taps
    assert a == b and len(a) == 4


def test_count_in_rejects_uneven_knocks():
    """Four knocks that are not a count-in must not become one."""
    audio, _ = make_acoustic("drone", bpm=84.0, bars=4)
    rng = np.random.default_rng(11)
    pre = np.zeros(int(4.0 * SR), dtype=np.float32)
    for t in (1.5, 2.0, 2.9, 3.1):                 # deliberately uneven
        k = knock(rng, level=0.5)
        s0 = int(t * SR)
        pre[s0:s0 + k.size] += k
    tr = Trace(np.concatenate([pre, audio]), count_in=True)
    assert tr.taps == []


# ================================================== fixture-generator contract

def test_fixture_is_deterministic():
    a, ta = make_acoustic("fingerpick", bpm=84.0, bars=6, seed=5)
    b, tb = make_acoustic("fingerpick", bpm=84.0, bars=6, seed=5)
    np.testing.assert_array_equal(a, b)
    assert ta == tb


def test_fixture_seeds_differ():
    a, _ = make_acoustic("strum", bpm=72.0, bars=6, seed=1)
    b, _ = make_acoustic("strum", bpm=72.0, bars=6, seed=2)
    assert not np.array_equal(a, b)


def test_fixture_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        make_acoustic("shred")


def test_fixture_really_is_sustained_and_soft(fingerpick):
    """Guard the guard: if the fixture ever stops being hard, the suite stops
    testing anything.  Two measurable properties define this material.
    """
    audio, truth = fingerpick
    fr = _frames(audio)
    rms = np.asarray([f.rms_db for f in fr])
    # SUSTAIN: no frame between events ever falls near silence — something is
    # always ringing.  (A pluck fixture drops 20+ dB between notes.)
    assert float(rms.max() - np.percentile(rms, 5)) < 12.0, (
        "fixture has percussive gaps; it is no longer sustained material")
    # SOFT: the flux contrast that made the old detector fail.  p90/median of
    # flux on the programmed corpus is 4-10; on this material it is ~2.
    flux = np.asarray([f.flux for f in fr])
    assert float(np.percentile(flux, 90) / np.median(flux)) < 3.0, (
        "fixture transients have become sharp; it no longer reproduces the bug")


# ============================================================ NON-REGRESSION

# THE PROGRAMMED CORPUS, ON THE SHIPPING DEFAULT.  Everything from here down
# runs ``acoustic=False``: this half exists to prove the sustained-acoustic
# work above costs the corpus nothing, and it can only do that on the front end
# the corpus actually ships with.
#
# The full DEFAULT-vs-ACOUSTIC table (22 cases, measured 2026-08-19) lives in
# ``beat.py``'s ACOUSTIC MODE block rather than being duplicated here.  The
# short version: the two front ends are within noise of each other on every
# programmed case except the GAIN axis, where acoustic mode is much better
# (lofi80 20 dB down: locks in 14.9s and holds 100%, against 35.3s and 48%),
# and except the TAIL of long quiet takes, where it is much worse — and worse
# in the way that matters, which is what the next paragraph is about.
#
# WHY THIS SECTION WAS REWRITTEN (the P2 the last round left behind).  The
# previous version of these tests built only 16-BAR renders and asserted lock
# fractions, and a real drift shipped straight through it: on ``lofi80_quiet``
# — the corpus's own case, a 32-bar lo-fi render at -20 dBFS — the tracker
# ended the take reporting 93.0 bpm against a truth of 80.0.  A ratio of 1.162
# is not an octave and not a dotted alias; it is an unrelated tempo, and it is
# the number the status line and the sim manifest's ``final_bpm`` show the
# human.  It was invisible here for three separate reasons, all now fixed:
#
#   1. the renders were 16 bars, and the fault is in the outro of a 32-bar arc;
#   2. nothing asserted the FINAL tempo, only the median over the settled
#      window — and a median is exactly the statistic a late drift survives;
#   3. the guard ran in whichever mode happened to be the default, so it could
#      not say which front end it was protecting.
#
# ``test_nonregression_the_corpus_quiet_case_holds_its_tempo_to_the_last_bar``
# below is the replacement, and it covers the same material at the same gain
# and the same length as ``tests/test_real_material.py``'s ``lofi80_quiet``.
#
# ONSETS/S IS IDENTICAL IN BOTH MODES — that is the point of leaving
# ``Frame.onset`` on ``flux`` (see features.py).  ``mind`` and ``roles`` read
# onset density and see literally the same numbers they saw before.

def _click(bpm, bars=17, peak_db=-21.0):
    spb = 60.0 / bpm
    x = np.zeros(int(bars * 4 * spb * SR), dtype=np.float32)
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    w = (np.exp(-t * 90.0) * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
    amp = 10.0 ** (peak_db / 20.0)
    for b in range(bars * 4):
        i = int(round(b * spb * SR))
        x[i:i + n] += (amp * (1.0 if b % 4 == 0 else 0.6) * w).astype(np.float32)
    return x


@pytest.mark.parametrize("bpm,bars", [(120.0, 17), (60.0, 13)])
def test_nonregression_click_track(bpm, bars):
    tr = Trace(_click(bpm, bars=bars), acoustic=False)
    bpms, conf, locked = tr.after(28.0)
    assert np.median(np.abs(bpms - bpm) / bpm) < 0.02
    assert locked.mean() > 0.8
    assert np.median(conf) > 0.75


@pytest.mark.parametrize("bpm", [120.0, 60.0])
def test_nonregression_click_is_gain_independent(bpm):
    """The same click 20 dB down must track identically.

    A click is an impulse train, so its onsets survive a 20 dB drop even on the
    default front end's absolute ``flux`` threshold — which is exactly why it
    is the WRONG case to conclude the gain axis is safe from.  Sustained
    programmed material at the same drop is not safe on the default; see
    ``test_nonregression_renders_20db_down`` and the corpus-length guard below.
    """
    loud = Trace(_click(bpm, bars=17, peak_db=-21.0), acoustic=False)
    quiet = Trace(_click(bpm, bars=17, peak_db=-41.0), acoustic=False)
    assert quiet.state.bpm == pytest.approx(loud.state.bpm, rel=0.02)
    assert abs(np.median(quiet.conf) - np.median(loud.conf)) < 0.08


@pytest.mark.parametrize("kind", ["steady", "stops", "busy_middle",
                                  "high_register"])
def test_nonregression_fixture_humans(kind):
    """``tests/fixtures_gen.py`` material must be unaffected."""
    from fixtures_gen import make_human
    audio = make_human(kind, bpm=96, bars=24)
    tr = Trace(audio, acoustic=False)
    bpms, conf, locked = tr.after(20.0)
    assert np.median(np.abs(bpms - 96.0) / 96.0) < 0.02, (
        f"{kind}: median bpm {np.median(bpms):.1f}")
    assert locked.mean() > 0.85
    assert np.median(conf) > 0.80


def test_nonregression_rubato_intro_still_waits_then_locks():
    from fixtures_gen import make_human
    tr = Trace(make_human("rubato_intro", bpm=96, bars=24), acoustic=False)
    early = tr.conf[tr.t < 7.0]
    assert not any(np.asarray(tr.locked)[tr.t < 8.0]), \
        "locked during the pulse-free intro"
    assert float(np.median(early)) < 0.55
    bpms, _c, locked = tr.after(30.0)
    assert np.median(np.abs(bpms - 96.0) / 96.0) < 0.02
    assert locked.mean() > 0.9


@pytest.mark.parametrize("kind", ["on_grid", "rushing", "dragging",
                                  "takes_over"])
def test_nonregression_joiners(kind):
    """v2 joiners.  ``takes_over`` is the one that improved: its density ramps
    to 4 onsets/beat and the old envelope followed it to 161bpm against a true
    120."""
    from fixtures_gen import make_joiner
    audio, truth = make_joiner(kind, machine_bpm=120.0, bars=16)
    tr = Trace(audio, acoustic=False)
    bpms, _c, locked = tr.after(14.0)
    assert np.median(np.abs(bpms - 120.0) / 120.0) < 0.03, (
        f"{kind}: median bpm {np.median(bpms):.1f}")
    assert locked.mean() > 0.85


#: synthwave's voices and levels on the day the ``--input auto`` discriminator
#: was calibrated (2026-08-19, the table in sim.py).  The confusion matrix
#: below IS that calibration, so its synthwave row keeps this material.  Every
#: other test in this file hears the style as it plays today.  The style named
#: no offline ``gain_db`` that day (compose's defaults set every track), and
#: the offline render is what the table heard, so the block it has since
#: gained for the new voices comes out too.
_SYNTHWAVE_CALIBRATED = {
    "voices": {"bass": "808sub", "pad": "pad", "arp": "pluck", "lead": "lead"},
    "role_gain_db": {"arp": -5.0, "lead": -2.0, "bass": None},
}


def _calibrated_synthwave(name: str) -> dict:
    """Today's synthwave style with the calibration day's palette put back."""
    from plume.styles_lib import load_style
    style = load_style(name)
    if name == "synthwave":
        style["voices"].update(_SYNTHWAVE_CALIBRATED["voices"])
        style.pop("gain_db", None)
        for role, db in _SYNTHWAVE_CALIBRATED["role_gain_db"].items():
            if db is None:
                style["role_gain_db"].pop(role, None)
            else:
                style["role_gain_db"][role] = db
    return style


@pytest.fixture(scope="module")
def renders(tmp_path_factory):
    """plume's own programmed material — the metric-alias axis the fixture
    humans do not have.  Built once; this is the expensive part of the file.

    ``lofi80_32`` is deliberately the SAME LENGTH as the corpus's own
    ``lofi80_quiet`` (a full 32-bar arc, so the take has the thinning outro
    that 16 bars never reaches).  It costs one extra render and it is the only
    case in this file that would have caught the 93.0 bpm drift.

    ``synthwave100_calibrated`` is synthwave on the palette the discriminator
    was calibrated on (``_SYNTHWAVE_CALIBRATED``).  It costs one more render,
    and it is what keeps a change of palette from quietly swapping the
    material under the confusion matrix.
    """
    import importlib
    from plume.compose import PromptSpec, compose
    from plume.engine import render
    import soundfile as sf
    # the module, not the function plume re-exports under the same name
    compose_mod = importlib.import_module("plume.compose")
    d = tmp_path_factory.mktemp("acoustic_renders")
    out = {}
    for style, bpm, key, bars, name in (
        ("lofi", 80.0, "C major", (0, 16), "lofi80"),
        ("trap", 140.0, "D minor", (0, 16), "trap140"),
        ("synthwave", 100.0, "C minor", (0, 16), "synthwave100"),
        ("synthwave", 100.0, "C minor", (0, 16), "synthwave100_calibrated"),
        ("lofi", 80.0, "C major", (0, 32), "lofi80_32"),
    ):
        p = d / f"{name}.wav"
        with pytest.MonkeyPatch.context() as mp:
            if name.endswith("_calibrated"):
                mp.setattr(compose_mod, "load_style", _calibrated_synthwave)
            session = compose(PromptSpec(styles=[style], bpm=bpm, key=key,
                                         movements=1, mood=[], name=p.stem), seed=7)
        render(session, out_path=str(p), movement=0, bars=bars)
        x, _ = sf.read(str(p), dtype="float32", always_2d=True)
        out[name] = (x.mean(axis=1).astype(np.float32), bpm)
    return out


@pytest.mark.parametrize("name", ["lofi80", "trap140", "synthwave100",
                                  "lofi80_32"])
def test_nonregression_programmed_renders(renders, name):
    """The dotted/octave alias corpus.  These are the cases the ACF comb, the
    metric-nesting term and the octave stickiness exist for, and the acoustic
    front end must not have cost any of them."""
    audio, bpm = renders[name]
    tr = Trace(audio, acoustic=False)
    bpms, conf, locked = tr.after(20.0)
    assert np.median(np.abs(bpms - bpm) / bpm) < 0.02, (
        f"{name}: median bpm {np.median(bpms):.1f} vs {bpm}")
    assert locked.mean() > 0.6, f"{name}: locked {locked.mean():.2f}"
    assert np.median(conf) > 0.60


def test_nonregression_the_corpus_quiet_case_holds_its_tempo_to_the_last_bar(
        renders):
    """THE P2 GUARD.  ``lofi80_quiet``, on the DEFAULT front end, to the end.

    Same material, same gain and same length as
    ``tests/test_real_material.py::test_tracker_locks_at_the_true_tempo``'s
    ``lofi80_quiet`` case: a 32-bar lo-fi render at -20 dBFS.  That test reads
    the sim manifest's ``final_bpm``; this one reads the tracker directly, so
    the two fail independently rather than through one shared pipeline.

    THE FINAL SAMPLE IS ASSERTED, not just the median, and that is the whole
    point of the test.  The drift this replaces (93.0 bpm against a truth of
    80.0, ratio 1.162 — an unrelated tempo, not an octave) happened over the
    outro, so it moved the last reading by 16% while barely moving the median
    at all.  A guard that only looks at the median cannot see it.

    Off-grid samples are bounded rather than forbidden: at -20 dB the default
    envelope's absolute flux threshold really does thin onsets (~10/s to
    ~2.6/s, a documented v1.2 limit) and the tracker really does wander a
    little over the outro.  Measured on the default: 11% of settled samples
    more than 4% off, final 80.36, median 80.03, locked 56% of the settled
    window.  The bounds below are those numbers with margin.
    """
    audio, bpm = renders["lofi80_32"]
    tr = Trace(audio * 0.1, acoustic=False)
    bpms, conf, locked = tr.after(20.0)
    assert bpms.size > 200, "take too short to be this test"
    assert tr.state.bpm == pytest.approx(bpm, rel=0.04), (
        f"the take ENDED on {tr.state.bpm:.2f} bpm against a truth of {bpm} "
        f"(ratio {tr.state.bpm / bpm:.3f})")
    assert np.median(np.abs(bpms - bpm) / bpm) < 0.02, (
        f"median bpm {np.median(bpms):.2f}")
    off = np.abs(bpms - bpm) / bpm > 0.04
    assert off.mean() < 0.25, (
        f"{off.mean():.2f} of the settled take is off-grid: "
        f"{sorted(set(np.round(bpms[off], 1)))}")
    assert locked.mean() > 0.40, f"locked {locked.mean():.2f} of the take"


@pytest.mark.parametrize("name", ["lofi80", "trap140", "synthwave100"])
def test_nonregression_renders_20db_down(renders, name):
    """The gain axis, on the DEFAULT front end — the honest version.

    A 20 dB drop is where the default's ABSOLUTE flux threshold binds: onsets
    thin from ~10/s to ~2.8/s, and on ``lofi80`` the tracker takes 35.3s to
    lock and holds only 48% of the settled window (against 14.3s and 100% at
    full level).  That is a documented v1.2 limit, and asserting a high lock
    fraction here would be asserting something false.

    What must NOT regress at any gain is the TEMPO, which is the musical fault
    — a slow lock costs the sideman runway, a wrong tempo makes it play the
    wrong music.  The lock fraction is floored well under the measurement so it
    still catches a collapse.  The strong version of this claim lives in
    ``test_the_gain_axis_is_what_acoustic_mode_buys`` below.
    """
    audio, bpm = renders[name]
    tr = Trace(audio * 0.1, acoustic=False)
    bpms, _conf, locked = tr.after(20.0)
    assert np.median(np.abs(bpms - bpm) / bpm) < 0.03, (
        f"{name}: median bpm {np.median(bpms):.2f} vs {bpm}")
    assert tr.state.bpm == pytest.approx(bpm, rel=0.05), (
        f"{name}: ended on {tr.state.bpm:.2f} bpm vs {bpm}")
    assert locked.mean() > 0.35, f"{name}: locked {locked.mean():.2f}"


@pytest.mark.parametrize("name", ["lofi80", "trap140", "synthwave100"])
def test_the_gain_axis_is_what_acoustic_mode_buys(renders, name):
    """The other side of the A/B: 20 dB down, ACOUSTIC mode does much better.

    This is the one place the sustained-acoustic front end helps the programmed
    corpus, and it is worth pinning because it is the reason a live take at an
    ordinary mic level should reach for ``PLUME_ACOUSTIC=1`` rather than
    fighting the gain.  ``attack`` is a log RATIO, so its picker's threshold is
    relative and a 20 dB drop moves nothing: measured lock fraction 1.00 on all
    three (against 0.48 / 1.00 / 0.76 on the default), and ``lofi80`` locks at
    14.9s instead of 35.3s.
    """
    audio, bpm = renders[name]
    tr = Trace(audio * 0.1, acoustic=True)
    bpms, _conf, locked = tr.after(20.0)
    assert np.median(np.abs(bpms - bpm) / bpm) < 0.03, (
        f"{name}: median bpm {np.median(bpms):.2f} vs {bpm}")
    assert locked.mean() > 0.90, f"{name}: locked {locked.mean():.2f}"


# ======================================================== the switch itself

def test_the_default_front_end_is_the_corpus_one(monkeypatch):
    """``BeatTracker()`` with nothing said must be the flux front end.

    Stated as its own test because everything in the NON-REGRESSION section
    passes ``acoustic=False`` explicitly (so the suite cannot be changed by the
    caller's shell), and that explicitness would otherwise hide the default
    silently flipping.  ``tests/test_beat.py`` covers the selection machinery
    in detail; this is the one-line statement of which side ships.
    """
    from plume.listen import beat as beat_mod
    monkeypatch.delenv(beat_mod.ACOUSTIC_ENV, raising=False)
    assert BeatTracker()._acoustic is False


def test_the_mode_flag_is_load_bearing_in_both_directions(fingerpick, renders):
    """NON-VACUITY.  Each mode must FAIL where the other succeeds.

    Two halves of this file make opposite claims — the acoustic tests say the
    ``attack`` front end is necessary, the non-regression tests say the
    ``flux`` one is.  If either mode quietly became the other, or if the flag
    stopped being read, both halves would keep passing and the suite would be
    testing one front end twice.  So each direction is pinned by the case that
    the OTHER mode gets wrong:

    * ``fingerpick`` on the DEFAULT reads its own 8th-note subdivision, 167.9
      bpm against a true 84 — the live session's exact number, at confidence
      0.83.  That is why acoustic mode exists.
    * ``trap140`` in ACOUSTIC mode locks at 6.7 s where the DEFAULT locks at
      3.8 s — 76% later, three seconds of a sideman standing there.  That is
      what is left of the reason acoustic mode is not the default.

    **v2.7 re-measurement, and it moved.**  Until v2.7 this half pinned a much
    harder failure: ``lofi80_32`` in acoustic mode ended the take on 92.98 bpm
    against a true 80 at -20 dBFS (and on 181.5 at full gain) — an unrelated
    tempo, not an octave.  SPEC-V1 amendment 48 fixed exactly that: the outro
    thins, a 4/3 alias out-scores the grid, and before v2.7 nothing stopped a
    settled grid yielding to a non-octave alias in one second.  Both now end on
    80.4.  This test instructed a re-run of the A/B rather than a relaxation
    when that happened, so here it is (median bpm after 20 s / END bpm /
    median conf / locked % / time to first lock):

        case            v1.3 attack              v2.7 attack              v2.7 flux
        lofi80          79.8/ 79.8/.73/100/14.8  79.8/ 79.8/.73/100/14.8  80.0/ 80.1/.78/100/14.3
        lofi80_q20      79.8/ 79.8/.73/100/14.8  79.8/ 79.8/.73/100/14.8  80.0/ 79.8/.72/ 48/35.3
        trap140        139.7/139.5/.85/100/ 6.7 139.7/139.5/.85/100/ 6.7 138.9/138.9/.73/100/ 3.8
        synth100       100.1/100.1/.82/100/ 4.6 100.1/100.1/.83/100/ 4.6 100.0/100.0/.83/100/ 4.2
        lofi80_32       80.0/181.5/.74/100/15.6  79.9/ 80.4/.74/100/15.6  80.0/ 80.4/.77/100/13.7
        lofi80_32_q20   80.0/ 93.0/.74/100/15.6  79.9/ 80.4/.74/100/15.6  80.0/ 80.4/.71/ 59/36.1

    **What that table says, said plainly: the strong case for flux as the
    default is gone.**  On the quiet cases the acoustic front end now wins
    outright (100% locked at ~15 s against 48-59% at ~35 s), it carries trap140
    at higher confidence, and its one remaining measured deficit is the late
    lock pinned below plus ~0.05 of median confidence on ``lofi80``.  Changing
    which mode ships as the default is a decision about every test in this
    suite and both flagship demos, so v2.7 does NOT make it — it records the
    measurement and leaves the call.  If the late lock is ever fixed too, this
    test has nothing left to pin and the default should change rather than the
    test being weakened.
    """
    audio, truth = fingerpick
    default = Trace(audio, acoustic=False)
    bpm, _c, _l = default.after(SETTLE_S)
    octaves = np.abs(np.log2(np.median(bpm) / truth["bpm"]))
    assert abs(octaves - 1.0) < 0.15, (
        f"the DEFAULT front end no longer octave-locks the fingerpicked "
        f"fixture (median {np.median(bpm):.1f} vs truth {truth['bpm']}); the "
        f"acoustic-mode tests above may now be passing for free")

    def first_lock(tr):
        lk = np.asarray(tr.locked)
        assert lk.any(), "never locked"
        return float(tr.t[int(np.argmax(lk))])

    trap, _tbpm = renders["trap140"]
    t_flux = first_lock(Trace(trap, acoustic=False))
    t_attack = first_lock(Trace(trap, acoustic=True))
    assert t_attack > t_flux * 1.35, (
        f"ACOUSTIC mode no longer locks trap140 materially later than the "
        f"default ({t_attack:.1f}s vs {t_flux:.1f}s); that late lock is the "
        f"LAST measured reason the default is flux — re-run the A/B in the "
        f"docstring above, and if flux has no advantage left the honest fix "
        f"is to change the default, not to weaken this test")


def test_frame_contract_is_additive():
    """v1.5 adds fields; it must not have moved or removed any."""
    import dataclasses
    names = [f.name for f in dataclasses.fields(feat.Frame)]
    assert names[:8] == ["t_s", "rms_db", "flux", "onset", "onset_strength",
                         "chroma", "centroid_hz", "chroma_low"]
    for extra in ("attack", "attack_breadth", "self_masked"):
        assert extra in names


@pytest.mark.parametrize("acoustic", MODES)
def test_tracker_falls_back_to_flux_for_stubs(acoustic):
    """Duck-typed frame stubs predating ``attack`` must still drive the
    tracker — which is what keeps ``tests/test_beat.py`` meaningful.

    Run in ACOUSTIC mode too, because that is the only mode where the fallback
    is load-bearing: there the tracker WANTS ``Frame.attack``, and a stub that
    has none must send it back to ``flux``/``onset`` rather than to a stream of
    zeros that never locks.
    """
    class Stub:
        def __init__(self, t, flux, onset):
            self.t_s = t
            self.rms_db = -20.0
            self.flux = flux
            self.onset = onset
            self.onset_strength = flux if onset else 0.0
            self.chroma = np.zeros(12, dtype=np.float32)

    bt = BeatTracker(acoustic=acoustic)
    dt = 512.0 / 48000.0
    frames = []
    for i in range(int(30.0 / dt)):
        t = (i + 1) * dt
        on = abs((t % 0.5)) < dt or abs((t % 0.5) - 0.5) < dt
        frames.append(Stub(t, 20.0 if on else 0.5, on))
    st = bt.process(frames)
    assert st.bpm == pytest.approx(120.0, rel=0.03)
    assert st.locked


def test_masked_frames_do_not_correct_phase():
    """In ACOUSTIC mode the tracker picks its OWN onsets off ``attack``, so
    clearing ``Frame.onset`` no longer suppresses them.  Both the
    ``self_masked`` field and the ``masked=`` argument must.

    Deliberately run with ``acoustic=True``: on the default path the masking
    flags are merely redundant with the already-cleared ``onset``, so a
    default-mode version of this test passes whether or not the tracker honours
    them at all.  This is the one configuration where it proves something.
    """
    audio, _ = make_acoustic("fingerpick", bpm=84.0, bars=8)
    fx = FeatureExtractor(sr=SR)
    frames = []
    for i in range(0, len(audio), BLOCK):
        frames.extend(fx.process(audio[i:i + BLOCK]))

    open_bt = BeatTracker(acoustic=True)
    open_bt.process(frames)
    assert len(open_bt._onsets) > 0, (
        "the attack picker found no onsets at all; the masking assertions "
        "below would then be vacuous")

    arg_bt = BeatTracker(acoustic=True)
    arg_bt.process(frames, masked=[True] * len(frames))
    assert len(arg_bt._onsets) == 0, "masked= did not suppress the attack picker"

    import dataclasses
    field_bt = BeatTracker(acoustic=True)
    field_bt.process([dataclasses.replace(f, self_masked=True) for f in frames])
    assert len(field_bt._onsets) == 0, "self_masked did not suppress it"


# ===================================================== v1.5.2: --input auto
#
# The discriminator behind ``--input auto`` (sim.classify_input).  Acoustic
# mode existed and nothing could select it: every entry point constructed
# ``BeatTracker()`` with defaults, so the fingerpick fixture read 167.9 bpm —
# the live session's exact wrong number — with no reachable fix.  ``auto``
# has to earn the right to pick silently, so its confusion matrix over BOTH
# corpora is pinned here: every programmed render (loud and 20 dB down),
# every joiner, every fixture human, both clicks, and every sustained-
# acoustic fixture (loud and 20 dB down, plus a count-in-prepended take).
# If a new corpus case ever lands on the wrong side, the honest fix is in
# sim.INPUT_BREADTH_STRONG_MAX / INPUT_BROAD_FRAC_MAX with a re-measured
# calibration table — not a per-case exemption here.

def _matrix_cases(renders, fingerpick, strum, drone):
    """{name: (audio, wanted_mode)} over both corpora, both gains."""
    from fixtures_gen import make_human, make_joiner
    cases = {}
    for name in ("lofi80", "trap140", "synthwave100", "lofi80_32"):
        # synthwave on the palette this table was measured on; today's palette
        # has its own test below.
        audio, _bpm = renders["synthwave100_calibrated" if name == "synthwave100"
                              else name]
        cases[name] = (audio, "programmed")
        cases[f"{name}_q20"] = ((audio * 0.1).astype(np.float32), "programmed")
    for kind in ("steady", "stops", "busy_middle", "high_register",
                 "rubato_intro", "drift"):
        cases[f"h_{kind}"] = (make_human(kind, bpm=96, bars=24), "programmed")
    for kind in ("on_grid", "rushing", "dragging", "takes_over"):
        a, _t = make_joiner(kind, machine_bpm=120.0, bars=16)
        cases[f"jn_{kind}"] = (a, "programmed")
    cases["click120"] = (_click(120.0, bars=17), "programmed")
    cases["click60"] = (_click(60.0, bars=13), "programmed")
    for name, (audio, _truth) in (("ac_fingerpick", fingerpick),
                                  ("ac_strum", strum), ("ac_drone", drone)):
        cases[name] = (audio, "acoustic")
        cases[f"{name}_q20"] = ((audio * 0.1).astype(np.float32), "acoustic")
    fp8, _ = make_acoustic("fingerpick", bpm=84.0, bars=8)
    au, _taps = prepend_count_in(fp8, 84.0)
    cases["ac_fp_countin"] = (au, "acoustic")
    return cases


def test_auto_discriminator_confusion_matrix(renders, fingerpick, strum,
                                             drone):
    """``--input auto`` must be RIGHT on every case of both corpora.

    This is the measurement that licenses auto to pick without a hint.  The
    quiet (``_q20``) rows double as the gain-invariance claim: ``attack`` is
    a log ratio, so neither metric may move with level.
    """
    from plume.sim import classify_input
    cases = _matrix_cases(renders, fingerpick, strum, drone)
    rows, wrong = [], []
    for name, (audio, want) in cases.items():
        dec = classify_input(audio, SR)
        ok = dec.mode == want
        rows.append(f"  {'ok ' if ok else 'BAD'} {name:<20} want={want:<10} "
                    f"got={dec.mode:<10} {dec.reason}")
        if not ok:
            wrong.append(name)
    assert not wrong, ("auto misclassified " + ", ".join(wrong)
                       + " — full matrix:\n" + "\n".join(rows))


def test_auto_hears_todays_synthwave_as_sustained_a_known_miss(renders):
    """A MISS, PINNED SO IT IS NOT FORGOTTEN.  This asserts the wrong answer.

    The synth palette (plume.synths) gave synthwave a supersaw pad, a Juno
    arp, a ladder bass and lead in place of ``pad``, ``pluck``, ``808sub`` and
    ``lead``.  Its render is still programmed material, but its transients
    are softer, and the discriminator now reads it as sustained acoustic:

      metric (first 12 s)                      calibrated    today
      p75 of attack_breadth at strong attacks  0.500         0.417  (<= 0.44)
      fraction of frames with breadth >= 0.55  0.024         0.006  (<= 0.018)

    Today's numbers sit inside the acoustic cluster on the second axis and in
    the calibration gap on the first, 0.042 from the strummed fixture.  No
    threshold separates them with room to spare, and moving one toward real
    guitars to fit a synth pad would risk the 167.9 bpm fault the
    discriminator exists to prevent.  The fix is a feature that tells a
    machine pad from a string, which is the follow lane's work.  Both front
    ends still track this render (the tests above use today's palette).

    When this fails, auto has changed its mind about today's synthwave.  If
    it now says ``programmed``, delete this test and drop the calibrated
    render from ``_matrix_cases``, so the matrix reads the live style again.
    """
    from plume.sim import classify_input
    cal = classify_input(renders["synthwave100_calibrated"][0], SR)
    assert cal.mode == "programmed"
    assert cal.breadth_strong == pytest.approx(0.500, abs=0.011), (
        "the calibrated render is not the material the table was measured on")
    assert cal.broad_frac == pytest.approx(0.024, abs=0.002)
    now = classify_input(renders["synthwave100"][0], SR)
    assert now.mode == "acoustic", (
        f"auto now reads today's synthwave as {now.mode} ({now.reason}); "
        "see this test's docstring for what to change")


def test_classifier_reads_only_the_probe_window(fingerpick):
    """A six-minute take costs what a 12-second take costs, by construction."""
    from plume.sim import INPUT_PROBE_S, classify_input
    audio, _ = fingerpick
    whole = classify_input(audio, SR)
    probe = classify_input(audio[: int(INPUT_PROBE_S * SR)], SR)
    assert whole == probe


def test_classifier_refuses_to_guess_on_silence():
    from plume.sim import classify_input
    dec = classify_input(np.zeros(int(6.0 * SR), dtype=np.float32), SR)
    assert dec.mode == "programmed"
    assert "silent" in dec.reason or "empty" in dec.reason
    assert dec.hint is None


def test_classifier_split_verdict_keeps_programmed_and_hints():
    """A take that is only PARTLY machine-shaped must not be guessed at.

    A sustained drone with a slow broadband knock riding on it (someone
    tapping the guitar body under ringing strings): the transient-breadth
    axis reads acoustic (the strong attacks are still string swells, measured
    br_strong 0.208) while the broadband-frame share reads programmed (the
    knocks lift it to ~0.027, above the 0.018 gap).  The contract for a
    split verdict is programmed + a hint — a wrong silent guess is worse
    than a hint.
    """
    from plume.sim import classify_input
    audio, _ = make_acoustic("drone", bpm=84.0, bars=8)
    rng = np.random.default_rng(9)
    x = audio.copy()
    t = 0.7
    while t < 12.0:                                  # a knock every 1.4s
        k = knock(rng, level=0.5)
        s0 = int(t * SR)
        x[s0:s0 + k.size] += k
        t += 1.4
    dec = classify_input(x, SR)
    assert dec.mode == "programmed"
    assert dec.hint is not None and "--input acoustic" in dec.hint


def test_resolve_input_mode_explicit_beats_env_and_auto_defers(monkeypatch):
    from plume.listen import beat as beat_mod
    from plume.sim import resolve_input_mode
    silence = np.zeros(SR, dtype=np.float32)
    monkeypatch.setenv(beat_mod.ACOUSTIC_ENV, "1")
    ac, fields = resolve_input_mode("auto", silence, SR)
    assert ac is True and "PLUME_ACOUSTIC" in fields["input_mode_reason"]
    ac, _f = resolve_input_mode("programmed", silence, SR)
    assert ac is False, "--input programmed must beat the environment"
    monkeypatch.delenv(beat_mod.ACOUSTIC_ENV)
    ac, fields = resolve_input_mode("auto", silence, SR)
    assert ac is False
    with pytest.raises(ValueError, match="input_mode"):
        resolve_input_mode("acousticc", silence, SR)


# ============================================ v1.5.2: the count-in, ARMED
#
# ``CountInDetector`` shipped in v1.5.1 with the honest caveat that nothing
# constructed it.  ``BeatTracker(count_in=True)`` (or ``PLUME_COUNT_IN=1``,
# for the live paths that construct with defaults) now arms it inside the
# tracker; these tests hold the WIRING — the detector itself is held by the
# MODES-parametrized count-in tests above via the external Trace wiring.

def test_tracker_arms_its_own_count_in_detector():
    """``BeatTracker(count_in=True)`` alone — no external wiring — latches
    the counted tempo AND the counted downbeat, on the DEFAULT front end
    (where the fingerpick fixture otherwise octave-locks at 167.9)."""
    audio, _ = make_acoustic("fingerpick", bpm=84.0, bars=8)
    au, taps = prepend_count_in(audio, 84.0)
    tr = Trace(au, tracker=BeatTracker(acoustic=False, count_in=True))
    bt = tr.bt
    assert bt._tap_anchored, "the armed detector never reached tap()"
    assert len(bt._taps) == 4
    assert np.allclose(bt._taps, taps, atol=0.06)
    assert tr.state.bpm == pytest.approx(84.0, rel=0.035)
    # downbeat semantics: the four taps got beat_in_bar 0..3, so the last
    # count is beat 4 of the bar and the NEXT beat is the downbeat — the
    # anchor tap() derives for beat_in_bar=3 is (-3) % 4.
    assert bt._anchor == (-3) % 4


def test_count_in_env_flag_arms_it_and_explicit_false_wins(monkeypatch):
    from plume.listen import beat as beat_mod
    monkeypatch.delenv(beat_mod.COUNT_IN_ENV, raising=False)
    assert BeatTracker()._countin is None, "count-in must be OFF by default"
    monkeypatch.setenv(beat_mod.COUNT_IN_ENV, "1")
    assert BeatTracker()._countin is not None
    assert BeatTracker(count_in=False)._countin is None, \
        "an explicit False must ignore the environment"


@pytest.mark.parametrize("acoustic", MODES)
def test_armed_count_in_does_not_disturb_stub_streams(acoustic):
    """Duck-typed frames without ``attack`` fields must run through an ARMED
    tracker exactly as they do through an unarmed one."""
    class Stub:
        def __init__(self, t, flux, onset):
            self.t_s = t
            self.rms_db = -20.0
            self.flux = flux
            self.onset = onset
            self.onset_strength = flux if onset else 0.0
            self.chroma = np.zeros(12, dtype=np.float32)

    dt = 512.0 / 48000.0
    frames = []
    for i in range(int(30.0 / dt)):
        t = (i + 1) * dt
        on = abs(t % 0.5) < dt or abs((t % 0.5) - 0.5) < dt
        frames.append(Stub(t, 20.0 if on else 0.5, on))
    bt = BeatTracker(acoustic=acoustic, count_in=True)
    st = bt.process(frames)
    assert st.bpm == pytest.approx(120.0, rel=0.03) and st.locked
    assert bt._taps == [], "a stub stream must never fire the count-in"


@pytest.mark.parametrize("acoustic", MODES)
def test_armed_count_in_holds_48_bars_worst_sample(acoustic):
    """THE LONG-TAKE GUARD (method lesson: >=48 bars, worst window — two
    rounds of defects survived 16-bar fixtures and median assertions).

    48 bars at 84 bpm is ~137s, a real take's length.  After the latch, EVERY
    sample must stay inside the count-in's drift band — the tap anchor clamps
    corrections to +/-3% of the tapped tempo, so a single sample outside
    ~3.5% of truth means the anchor let go somewhere in the take.  The
    default front end would read 167.9 here without the count-in, so this
    also proves the latch holds against a persistent octave alias for a full
    take, in both modes.
    """
    audio, _ = make_acoustic("fingerpick", bpm=84.0, bars=48)
    au, taps = prepend_count_in(audio, 84.0)
    tr = Trace(au, tracker=BeatTracker(acoustic=acoustic, count_in=True))
    assert tr.bt._tap_anchored
    m = tr.t >= taps[-1] + 2.0
    assert m.sum() > 400, "take too short to be this test"
    bpms = tr.bpm[m]
    worst = float(np.max(np.abs(bpms - 84.0) / 84.0))
    assert worst < 0.035, (
        f"worst post-latch sample is {worst:.1%} off the counted 84 "
        f"({sorted(set(np.round(bpms[np.abs(bpms - 84.0) / 84.0 > 0.035], 1)))})")
    assert np.asarray(tr.locked)[m].all(), "the latch unlocked mid-take"
    assert float(tr.conf[m].min()) >= 0.70, (
        f"confidence fell to {tr.conf[m].min():.2f} under the tap floor")
    assert tr.state.tempo_octave_changes == 0


def test_sim_taps_still_work_in_acoustic_mode(tmp_path):
    """``plume sim --taps`` + ``--input acoustic`` together: the tap latch
    must survive the front-end switch (the count-in hatch is NOT gated on
    the mode — beat.py's contract)."""
    import soundfile as sf
    from plume.sim import simulate
    audio, _ = make_acoustic("fingerpick", bpm=84.0, bars=16)
    wav = tmp_path / "fp.wav"
    sf.write(str(wav), audio, SR)
    spb = 60.0 / 84.0
    taps = [0.5 + i * spb for i in range(4)]
    r = simulate(str(wav), out_path=str(tmp_path / "duet.wav"),
                 taps_s=taps, input_mode="acoustic")
    m = r.manifest
    assert m["input_mode_resolved"] == "acoustic"
    assert m["final_bpm"] == pytest.approx(84.0, rel=0.035)
    late = [(t, b) for t, b in r.bpm_trace if t > taps[-1] + 1.0]
    assert late, "no trace after the taps"
    worst = max(abs(b - 84.0) / 84.0 for _t, b in late)
    assert worst < 0.035, f"tap latch drifted {worst:.1%} in acoustic mode"
