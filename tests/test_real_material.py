"""The real-material suite — the test gap that let v1 ship broken.

WHY THIS FILE EXISTS.  v1 passed all ten synthetic-fixture behavior tests and
its musical QA, and then failed completely on real music: the sideman never
entered, the silence ladder walked to tacet while the human played
continuously, the beat tracker locked to dotted and octave aliases, and the
harmony tracker named a key from 53ms of audio.  Every one of those was
invisible to ``tests/fixtures_gen.py`` material, for reasons that are
measurable rather than mysterious:

* **No input-gain axis.**  Fixtures are peak-normalized to 0.85, so every bar
  measures -15..-24 dBFS — 21 to 30 dB above the -45 dBFS silence floor, which
  is therefore unreachable by construction.  Silent bars are exact digital
  zeros at -63 dB.  Nothing between -24 and -63 dB is ever generated, and that
  is the 40 dB band where real recordings live.
* **No crest-factor axis.**  ``pluck`` notes decay over 0.4s and ``808sub``
  notes run 1.75 beats, so fixture bars are continuously filled.  Real
  percussive material has a 5-10% duty cycle and a 13-25 dB peak-to-mean gap —
  and that gap alone pushes a loudly-playing bar under an absolute floor.
* **No metric-alias axis.**  Fixture onsets are sparse and on-grid, so the
  autocorrelation peak IS the beat.  Real programmed music groups sixteenths
  2+1, which puts the strongest ACF lag on a DOTTED (3-unit) relative of the
  beat — a tempo the octave-only candidate generator could not even propose.
* **No low-register axis.**  Fixture partials are widely spaced.  Real mixes
  stack a bass note and a chord root inside one FFT lobe.

So this suite runs the FULL sim pipeline over plume's own v0 renders — built
in-test with ``compose`` + ``engine.render``, plus a bare click track — and
asserts that the sideman actually works on them.  Everything here is a
regression guard for a defect that really shipped; see SPEC-V1 "v1.2
amendments" for the algorithm changes each one covers.

Determinism: renders and sims are built once per session from fixed seeds and
reused; no wall clock, no global rng.

Self-containment: every input here is SYNTHESIZED IN-TEST into ``tmp_path``.
Nothing reads ``renders/`` — those files are gitignored (``*.wav``), so a test
that read one skipped itself on any clean clone, which is how ``bpm_hint``
shipped with zero coverage.  A test that can silently not run is not a test.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plume import sim as sim_mod                                # noqa: E402
from plume.listen import beat as beat_mod                       # noqa: E402
from plume.compose import PromptSpec, compose                   # noqa: E402
from plume.engine import render                                 # noqa: E402
from plume.listen.beat import BeatTracker                       # noqa: E402
from plume.listen.features import FeatureExtractor              # noqa: E402
from plume.listen.harmony import HarmonyTracker                 # noqa: E402
from plume.mind import MindParams                               # noqa: E402
from plume.theory import parse_key                              # noqa: E402
from plume.voices import render_note                            # noqa: E402

SR = 48000


# --------------------------------------------------------------- material


def _click(path: Path, bpm: float, bars: int = 17, peak_db: float = -21.0) -> Path:
    """A bare click track: the minimal input a human could possibly give.

    Deliberately quiet-ish and very sparse (30ms of 1kHz per beat, ~5% duty
    cycle) — the crest factor the fixtures never produce.
    """
    spb = 60.0 / bpm
    x = np.zeros(int(bars * 4 * spb * SR), dtype=np.float32)
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    w = (np.exp(-t * 90.0) * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
    amp = 10.0 ** (peak_db / 20.0)
    for b in range(bars * 4):
        i = int(round(b * spb * SR))
        x[i:i + n] += (amp * (1.0 if b % 4 == 0 else 0.6) * w).astype(np.float32)
    sf.write(str(path), x, SR, subtype="PCM_24")
    return path


def _render(path: Path, style: str, bpm: float, key: str, *,
            bars: tuple[int, int] = (0, 16), seed: int = 7,
            gain_db: float = 0.0) -> Path:
    session = compose(
        PromptSpec(styles=[style], bpm=bpm, key=key, movements=1, mood=[],
                   name=path.stem),
        seed=seed,
    )
    render(session, out_path=str(path), movement=0, bars=bars)
    if gain_db:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        sf.write(str(path), x * (10.0 ** (gain_db / 20.0)), sr, subtype="PCM_24")
    return path


class Material:
    """One real-material case plus its ground truth and its sim result."""

    def __init__(self, name, path, bpm, key, *, allow_half=False,
                 allow_double=False):
        self.name = name
        self.path = str(path)
        self.bpm = bpm
        self.key = key
        self.allow_half = allow_half
        self.allow_double = allow_double
        self.result = None
        self.lines = None

    @property
    def truth_pcs(self) -> set[int]:
        """Pitch classes of the true key, plus those of its relative."""
        k = parse_key(self.key)
        return set(k.scale())

    def key_ok(self, reported: str | None) -> bool:
        """True key or its relative major/minor (same pitch collection).

        Relative keys are accepted because a pitch-class distribution cannot
        distinguish them in principle — only metric position and bass register
        can, and the harmony tracker sees neither.  See SPEC-V1 section 3.
        """
        if reported is None:
            return False
        want, got = parse_key(self.key), parse_key(reported)
        if (want.tonic, want.mode) == (got.tonic, got.mode):
            return True
        if want.mode == "major" and got.mode == "minor":
            return (want.tonic + 9) % 12 == got.tonic
        if want.mode == "minor" and got.mode == "major":
            return (want.tonic + 3) % 12 == got.tonic
        return False


@pytest.fixture(scope="session")
def material(tmp_path_factory):
    """Build the corpus once, run the full sim over each, return them keyed."""
    d = tmp_path_factory.mktemp("real_material")
    cases = [
        # A bare click track: sparse, percussive, well below full scale.
        Material("click120", _click(d / "click120.wav", 120.0), 120.0, None),
        # The same click 20 dB quieter — an ordinary live-mic level, and the
        # point at which the old absolute silence floor swallowed everything.
        Material("click120_quiet",
                 _click(d / "click120q.wav", 120.0, peak_db=-41.0),
                 120.0, None),
        # plume's own v0 material: swung lo-fi, trap, synthwave.
        Material("lofi80", _render(d / "lofi80.wav", "lofi", 80.0, "C major"),
                 80.0, "C major"),
        Material("trap140", _render(d / "trap140.wav", "trap", 140.0, "D minor"),
                 140.0, "D minor"),
        Material("synth100",
                 _render(d / "synth100.wav", "synthwave", 100.0, "C minor"),
                 100.0, "C minor"),
        # A real render at a quiet recording level: the gain axis the fixture
        # family has none of.
        # A real render at a quiet recording level, over the full 32-bar arc:
        # 20 dB down the onset detector's ABSOLUTE flux threshold
        # (features.ONSET_DELTA) is a much larger margin relative to the
        # material, so onsets thin from ~10/s to ~2.8/s and the tracker needs
        # noticeably longer to build confidence.  That front-end scaling is a
        # known v1.2 limit, documented in SPEC-V1; the take is sized so the
        # sideman still has runway, which is what this case is here to prove.
        Material("lofi80_quiet",
                 _render(d / "lofi80q.wav", "lofi", 80.0, "C major",
                         bars=(0, 32), gain_db=-20.0),
                 80.0, "C major"),
    ]
    out = {}
    for m in cases:
        m.result = sim_mod.simulate(
            m.path, style="lofi", out_path=str(d / f"duet_{m.name}.wav"),
            seed=0)
        m.lines = [json.loads(ln) for ln
                   in Path(m.result.decisions_path).read_text().splitlines()]
        out[m.name] = m
    return out


ALL = ("click120", "click120_quiet", "lofi80", "trap140", "synth100",
       "lofi80_quiet")
TONAL = ("lofi80", "trap140", "synth100", "lofi80_quiet")


# ------------------------------------------------------- the beat tracker


def _octave_ok(got: float, truth: float, m: Material) -> bool:
    if abs(got - truth) / truth < 0.045:
        return True
    if m.allow_double and abs(got - 2 * truth) / truth < 0.09:
        return True
    if m.allow_half and abs(got - 0.5 * truth) / truth < 0.03:
        return True
    return False


@pytest.mark.parametrize("name", ALL)
def test_tracker_locks_at_the_true_tempo(material, name):
    """No octave, dotted or subdivision alias on any real input.

    FAILED BEFORE v1.2: the candidate generator only proposed a peak and its
    OCTAVE relatives, while real hat patterns put the strongest ACF lag on a
    DOTTED (4/3, 3/2, 2/3) relative — so the true tempo was not merely
    outranked, it was never a candidate.  Measured: 80bpm lo-fi -> 160,
    140bpm trap -> 185.6 (= 140 * 4/3).
    """
    m = material[name]
    bpm = m.result.manifest["final_bpm"]
    assert bpm, f"{name}: no tempo estimate at all"
    assert _octave_ok(bpm, m.bpm, m), (
        f"{name}: tracked {bpm:.1f} bpm, truth {m.bpm} "
        f"(ratio {bpm / m.bpm:.3f})")


@pytest.mark.parametrize("name", ALL)
def test_tracker_locks_with_runway_left_to_play(material, name):
    """A lock must leave at least a couple of phrases of take to play over.

    Stated as runway rather than as a fraction of the take because the honest
    limit here is level-dependent: the onset picker's flux threshold is
    ABSOLUTE (features.ONSET_DELTA), so 20 dB down the same margin thins
    onsets from ~10/s to ~2.8/s and confidence takes longer to build.  That is
    a documented v1.2 limit (SPEC-V1 section 1); what must never regress is
    the tracker locking too late to be any use.
    """
    m = material[name]
    locks = [ln for ln in m.lines if ln.get("event") == "lock"]
    assert locks, f"{name}: never locked"
    bar_s = 4.0 * 60.0 / m.bpm
    runway = m.result.duration_s - locks[0]["t_s"]
    assert runway > 8 * bar_s, (
        f"{name}: locked at {locks[0]['t_s']:.1f}s of "
        f"{m.result.duration_s:.1f}s — only {runway / bar_s:.1f} bars left")


@pytest.mark.parametrize("name", ("click120", "click120_quiet", "lofi80",
                                  "trap140", "synth100"))
def test_tracker_locks_inside_the_first_third_at_normal_level(material, name):
    """At any ordinary recording level the lock is not merely usable, it is
    early.  (Excludes the deliberately -20 dB case; see the test above.)"""
    m = material[name]
    locks = [ln for ln in m.lines if ln.get("event") == "lock"]
    assert locks[0]["t_s"] < 0.34 * m.result.duration_s, (
        f"{name}: first lock at {locks[0]['t_s']:.1f}s of "
        f"{m.result.duration_s:.1f}s")


def _half_time_trap(bars: int = 16, bpm: float = 140.0, seed: int = 3
                    ) -> np.ndarray:
    """A half-time trap drum feel, built from plume's own voices.

    Kick on 1, snare on 3, hats on the OFFBEAT eighths, 808 sub under the
    kick.  Nothing lands on beats 2 and 4, so the flux autocorrelation at the
    beat lag is anti-correlated while the two-beat lag is strongly positive —
    which is exactly what makes a half-tempo reading a legitimate perceptual
    answer rather than a tracker bug (SPEC-V1 amendment 19).  Peak-normalised
    to 0.7; a pure function of (bars, bpm, seed).
    """
    rng = np.random.default_rng(seed)
    spb = 60.0 / bpm
    buf = np.zeros(int(round(bars * 4 * spb * SR)) + SR, dtype=np.float64)

    def add(y: np.ndarray, t_s: float) -> None:
        i = int(round(t_s * SR))
        n = min(y.size, buf.size - i)
        if n > 0:
            buf[i:i + n] += y[:n]

    def note(preset, dur_s, vel, hz=None):
        return render_note(preset, sr=SR, dur_s=dur_s, freq_hz=hz,
                           velocity=vel, rng=rng, params={})

    for b in range(bars):
        t0 = b * 4 * spb
        add(note("808kick", 0.5, 1.0), t0)
        add(note("snare", 0.4, 0.95), t0 + 2.0 * spb)
        for k in range(4):                       # offbeat eighths only
            add(note("hat", 0.05, 0.28), t0 + (k + 0.5) * spb)
        add(note("808sub", 1.6 * spb, 0.9, hz=65.41), t0)     # C2
    peak = float(np.max(np.abs(buf)))
    return (buf / peak * 0.7).astype(np.float32) if peak > 0 else \
        buf.astype(np.float32)


def _track_bpm(x: np.ndarray, hint: float | None):
    fe, tr = FeatureExtractor(sr=SR), BeatTracker(bpm_hint=hint)
    st = None
    for i in range(0, x.size, 4096):
        st = tr.process(fe.process(x[i:i + 4096]))
    return st


# ------------------------------ the tempo octave, mid-performance (v1.3)


@pytest.fixture(scope="session")
def long_take(tmp_path_factory):
    """A 24-bar trap render plus its sim — the shortest take that SETTLES.

    The 16-bar cases in ``material`` are over almost as soon as the tracker
    locks, so they cannot exercise the distinction v1.3 turns on: a tracker
    one bar past its lock is still reading the metre, while one forty seconds
    into a stable take is being played along with.  This take runs 44 s, and
    on it the pre-v1.3 rule really does halve the tempo mid-performance (see
    the non-vacuity test below, which reproduces it on this exact input).
    """
    d = tmp_path_factory.mktemp("long_take")
    wav = _render(d / "trap140_long.wav", "trap", 140.0, "D minor",
                  bars=(0, 24))
    res = sim_mod.simulate(str(wav), style="lofi",
                           out_path=str(d / "duet_long.wav"), seed=0)
    lines = [json.loads(ln) for ln
             in Path(res.decisions_path).read_text().splitlines()]
    return {"dir": d, "path": str(wav), "res": res, "lines": lines}


def _settled_bpms(lines, settle_s=None):
    """(t_s, bpm) at every bar boundary once the tempo has SETTLED.

    Deliberately not "every boundary after the lock": the boundary that
    coincides with the lock still reports whatever the tracker was reading on
    its way in (on this take, 184.6 — the 4/3 dotted alias real hat patterns
    produce), and that first reading is not what the guard is about.  The
    guard is about the tempo the tracker committed to and then played along
    with, so the window starts ``OCTAVE_SETTLE_S`` after the lock — exactly
    the boundary at which ``beat._grid_is_settled`` starts returning True.
    """
    lock = next((ln for ln in lines if ln.get("event") == "lock"), None)
    assert lock is not None, "never locked"
    t0 = lock["t_s"] + (beat_mod.OCTAVE_SETTLE_S if settle_s is None
                        else settle_s)
    return [(ln["t_s"], ln["bpm"]) for ln in lines
            if ln.get("bar", -1) >= 0 and ln.get("bpm") and ln["t_s"] >= t0]


def test_a_settled_tempo_never_changes_octave_mid_performance(long_take):
    """P1: the tracked tempo HALVED at bar 20 of a stable take and stayed.

    Measured before v1.3, on renders of this exact shape: locked at ~140,
    held it for fifty seconds and thirty bar boundaries, then dropped to
    69.88 and never came back — with NO unlock event, NO octave event, and
    confidence RISING across the switch (0.694 -> 0.736 -> 0.769), because a
    half-tempo grid predicts half as many beats and every one of them is
    still a beat the human played.  Every onset-density reading the mind took
    from that point was halved, which silently disables inverse-density
    coupling, and nothing downstream could see why.

    Musically: a drummer does not halve the tempo at bar 20 because the
    texture thinned.
    """
    lines = long_take["lines"]
    trace = _settled_bpms(lines)
    assert len(trace) >= 8, f"take too short to be a test: {len(trace)} bars"
    ref = trace[0][1]
    worst_t, worst = max(trace, key=lambda tb: abs(tb[1] - ref))
    assert abs(worst - ref) <= 0.05 * ref, (
        f"tempo moved {ref:.2f} -> {worst:.2f} bpm at t={worst_t:.1f}s "
        f"(x{worst / ref:.3f}) after the lock: "
        f"{[(round(t, 1), round(b, 1)) for t, b in trace]}")


def test_no_octave_change_happens_without_an_event_downstream_can_see(long_take):
    """If the octave DOES move, the decision log must say so.

    A silent re-interpretation is the part of this defect that made it
    invisible: `unlock` never fired, so a consumer watching lock state saw a
    perfectly healthy tracker on a grid at half the human's tempo.
    """
    lines = long_take["lines"]
    trace = _settled_bpms(lines)
    events = [ln for ln in lines if ln.get("event") == "tempo_octave_change"]
    ref = trace[0][1]
    moved = [(t, b) for t, b in trace if abs(math.log2(b / ref)) > 0.5]
    assert len(events) >= len(
        {round(math.log2(b / ref)) for _, b in moved}), (
        f"the octave moved to {moved} with {len(events)} logged events")
    for e in events:                       # and the line is self-describing
        assert e["from_bpm"] and e["to_bpm"] and e["t_s"] is not None


def test_a_committed_octave_change_reaches_the_decisions_log(long_take,
                                                             monkeypatch):
    """The `tempo_octave_change` line, end to end through ``sim``.

    The guards above assert the octave does NOT move on this take, so on their
    own they leave the logging path unexercised — and an event nobody ever
    emits is exactly as invisible as the silent switch it replaces.  Here the
    persistence requirement is dropped to the pre-v1.3 two evaluations (the
    SETTLE window is left alone, so the change still counts as a
    re-interpretation) and the resulting collapse must show up as a line with
    both tempi and a confidence that FELL.
    """
    monkeypatch.setattr(beat_mod, "OCTAVE_SWITCH_EVALS", 2)
    d = long_take["dir"]
    res = sim_mod.simulate(long_take["path"], style="lofi",
                           out_path=str(d / "duet_long_loose.wav"), seed=0)
    lines = [json.loads(ln) for ln
             in Path(res.decisions_path).read_text().splitlines()]
    evs = [ln for ln in lines if ln.get("event") == "tempo_octave_change"]
    assert evs, "the octave halved and the decisions log never said so"
    e = evs[0]
    assert abs(math.log2(e["to_bpm"] / e["from_bpm"])) == pytest.approx(
        1.0, abs=0.1), e
    conf_before = [ln["conf"] for ln in lines
                   if ln.get("conf") is not None and ln.get("t_s", 0) < e["t_s"]]
    assert e["conf"] < conf_before[-1], (
        f"confidence ROSE across a re-interpretation of the metre: "
        f"{conf_before[-1]} -> {e['conf']}")


def test_the_mid_take_octave_guard_is_not_vacuous(long_take, monkeypatch):
    """The fixture must actually be able to collapse.

    Runs the SAME render through the SAME sim with the pre-v1.3 rule restored
    (nothing ever settles, so an octave costs the same 2 evaluations as a 4%
    drift correction).  If this does not halve the tempo, the guard above is
    passing for free and the render needs replacing — not the guard relaxing.
    """
    settle_s = beat_mod.OCTAVE_SETTLE_S       # read BEFORE disabling it
    monkeypatch.setattr(beat_mod, "OCTAVE_SETTLE_S", 1e9)
    monkeypatch.setattr(beat_mod, "OCTAVE_SWITCH_EVALS", 2)
    d = long_take["dir"]
    res = sim_mod.simulate(long_take["path"], style="lofi",
                           out_path=str(d / "duet_long_legacy.wav"), seed=0)
    lines = [json.loads(ln) for ln
             in Path(res.decisions_path).read_text().splitlines()]
    trace = _settled_bpms(lines, settle_s)
    ref = trace[0][1]
    halved = [(t, b) for t, b in trace if abs(b - ref / 2.0) < 0.05 * ref]
    assert halved, (
        "the pre-v1.3 rule did NOT collapse on this take, so the octave "
        f"guard proves nothing: {[(round(t, 1), round(b, 1)) for t, b in trace]}")


def test_bpm_hint_recovers_the_notated_tempo_on_half_time_material():
    """``--bpm-hint`` is the documented escape for genuinely ambiguous meter.

    Half-time drum feels (beat-level flux ACF negative, bar-level strongly
    positive) make a half-tempo reading a legitimate perceptual answer that is
    simply not the notated one.  The hint narrows the tempo prior around a
    known bpm WITHOUT claiming a phase — which is exactly why it works where a
    count-in tap deadlocks.

    Both halves are asserted, because only the pair is meaningful: the
    unhinted read must actually BE half-time (otherwise the hint is a no-op
    and the test proves nothing), and the hint must recover the notated 140.

    Self-contained since v1.3.  This used to read ``renders/trap.wav``, which
    is gitignored — so on a clean clone it called ``pytest.skip`` and
    ``bpm_hint``, a whole feature, had zero coverage.
    """
    x = _half_time_trap()
    free, hinted = _track_bpm(x, None), _track_bpm(x, 140.0)
    assert free.bpm == pytest.approx(70.0, rel=0.05), (
        f"material is not half-time-ambiguous any more (free read "
        f"{free.bpm:.1f} bpm) — the hint below would prove nothing")
    assert hinted.locked, "hinted track never locked"
    assert hinted.bpm == pytest.approx(140.0, rel=0.05), (
        f"bpm_hint=140 gave {hinted.bpm:.1f} bpm")


# ------------------------------------------------------------ the harmony


@pytest.mark.parametrize("name", TONAL)
def test_key_belief_matches_the_true_key(material, name):
    m = material[name]
    assert m.key_ok(m.result.final_key), (
        f"{name}: reported {m.result.final_key!r}, truth {m.key!r} "
        f"(relative accepted)")


@pytest.mark.parametrize("name", TONAL)
def test_no_key_is_declared_before_there_is_evidence(material, name):
    """Key belief needs seconds of audio, not milliseconds.

    FAILED BEFORE v1.2: chroma is L1-normalised, so every voiced frame adds
    exactly 1.0 to the histogram and the "mass floor" of 5.0 was really a floor
    of FIVE FRAMES.  A ``key_change`` event fired at t=0.064s on every input,
    naming a key from 53ms — and it was the WRONG key on 3 of 4 renders, held
    for 5-11 seconds.
    """
    m = material[name]
    changes = [ln for ln in m.lines
               if ln.get("event") == "key_change" and ln.get("key")]
    assert changes, f"{name}: never named a key"
    assert changes[0]["t_s"] >= 1.0, (
        f"{name}: first key {changes[0]['key']!r} declared at "
        f"{changes[0]['t_s']:.3f}s — before any musical evidence")
    assert m.key_ok(changes[0]["key"]), (
        f"{name}: FIRST key declared was {changes[0]['key']!r}, "
        f"truth {m.key!r}")


@pytest.mark.parametrize("name", TONAL)
def test_no_out_of_key_chord_belief(material, name):
    """The machine voices from the chord belief, so an out-of-key chord belief
    IS an out-of-key bass note under the human.

    FAILED BEFORE v1.2: a single spectral peak deposited a phantom minor triad
    a perfect FIFTH BELOW itself (a pure 440Hz A read as "Dm" at 0.999
    confidence), because the harmonic tolerance could not reject a subharmonic
    hypothesis — ``12*log2(h)`` is within 0.35 semitones of an integer for
    every h in 1..9.  Allowance: at most one chord root outside the key's
    collection, since real progressions do use borrowed chords.
    """
    m = material[name]
    pcs = m.truth_pcs
    from plume.theory import parse_chord
    bad = []
    for ln in m.lines:
        sym = ln.get("chord")
        if not sym:
            continue
        try:
            root = parse_chord(sym).root
        except ValueError:                       # pragma: no cover - defensive
            continue
        if root % 12 not in pcs:
            bad.append(sym)
    assert len(set(bad)) <= 1, (
        f"{name}: chord beliefs outside {m.key}: {sorted(set(bad))}")


@pytest.mark.parametrize("hz,pc", [(220.0, 9), (440.0, 9), (261.63, 0),
                                   (329.63, 4), (110.0, 9)])
def test_pure_tone_reads_as_exactly_one_pitch_class(hz, pc):
    """THE MINIMAL REPRO, made permanent.

    A pure tone must deposit its own pitch class and nothing else, and must
    never produce a chord rooted a perfect fifth below it.  Before v1.2 a
    440Hz A produced chroma {A: 0.661, D: 0.243, F: 0.096} and the harmony
    tracker reported chord "Dm" at 0.999 confidence.
    """
    t = np.arange(3 * SR) / SR
    x = (0.2 * np.sin(2.0 * np.pi * hz * t)).astype(np.float32)
    frames = FeatureExtractor(sr=SR).process(x)
    state = HarmonyTracker().process(frames)
    chroma = np.mean([f.chroma for f in frames[len(frames) // 2:]], axis=0)
    assert chroma[pc] > 0.95, (
        f"{hz}Hz: only {chroma[pc]:.3f} of the chroma mass on its own pitch "
        f"class; spill = {np.round(chroma, 3).tolist()}")
    if state.chord is not None:                  # a lone tone is not a triad
        from plume.theory import parse_chord
        assert parse_chord(state.chord).root % 12 != (pc - 7) % 12, (
            f"{hz}Hz reported {state.chord!r} — the minor triad a fifth below")


def test_two_low_notes_a_minor_third_apart_are_resolved():
    """A3 + C4 must read as {A, C} and never as the phantom between them.

    FAILED BEFORE v1.2: at WIN=2048/48k a Hann main lobe is 93.8Hz wide while
    A3 and C4 are 41.6Hz apart, so they merged into one lobe whose interpolated
    centroid landed near 243Hz -> pitch class B, flickering with the partials'
    relative phase.  This is the register real music puts its bass and chord
    roots in, and no fixture ever exercised it.
    """
    t = np.arange(3 * SR) / SR
    x = (0.15 * np.sin(2 * np.pi * 220.0 * t)
         + 0.15 * np.sin(2 * np.pi * 261.63 * t)).astype(np.float32)
    frames = FeatureExtractor(sr=SR).process(x)[100:]
    assert frames
    for f in frames[::7]:
        mass = float(f.chroma[9] + f.chroma[0])          # A + C
        assert mass > 0.95, f"only {mass:.3f} on {{A, C}}: {np.round(f.chroma, 3)}"
        assert f.chroma[11] < 0.02 and f.chroma[8] < 0.02   # no B, no G#


# ------------------------------------------------- the sideman, end to end


@pytest.mark.parametrize("name", ALL)
def test_the_heard_bars_counter_advances(material, name):
    """P0: on real input the listening counter used to stay at 0 FOREVER.

    Even on an ideal 120bpm click where the tracker locked at conf 0.94 and
    the bar counter advanced 0..16, EVERY reason line read
    ``listening: heard 0/4 bars (conf 0.94, locked)`` — because the bar's MEAN
    RMS sat under the absolute silence floor, so every bar the human played
    was recorded as silence and the entry gate could never arm.  The signature
    to guard is therefore: the LAST thing the listening state ever said must
    not still be "heard 0/N".
    """
    m = material[name]
    listening = [ln for ln in m.lines if ln.get("state") == "listening"]
    assert listening, f"{name}: no listening lines at all"
    last = str(listening[-1]["reason"])
    assert not re.search(r"heard 0/", last), (
        f"{name}: the heard-bars counter never advanced — {last!r}")
    counts = [int(mt.group(1)) for ln in listening
              for mt in [re.search(r"heard (\d+)", str(ln["reason"]))] if mt]
    assert counts and max(counts) > 0


@pytest.mark.parametrize("name", ALL)
def test_the_machine_enters(material, name):
    """THE HEADLINE FEATURE.  The duet must actually happen."""
    m = material[name]
    assert m.result.entered_bar is not None, (
        f"{name}: never entered — {m.result.manifest.get('no_entry_reason')}")
    assert "entering" in m.result.states_by_bar
    assert "comping" in m.result.states_by_bar, (
        f"{name}: entered but never reached comping: {m.result.states_by_bar}")


@pytest.mark.parametrize("name", ALL)
def test_the_machine_enters_like_a_sideman_not_immediately(material, name):
    """Late, but not absurdly late: after the listen gate, inside the take."""
    m = material[name]
    p = MindParams()
    assert m.result.entered_bar >= p.min_listen_bars
    assert m.result.entered_bar % p.entry_boundary_bars == 0
    entered = [ln for ln in m.lines if ln.get("state") == "entering"]
    assert entered[0]["t_s"] < 0.6 * m.result.duration_s, (
        f"{name}: entered at {entered[0]['t_s']:.1f}s of "
        f"{m.result.duration_s:.1f}s — the take is half over")


@pytest.mark.parametrize("name", ALL)
def test_the_silence_ladder_never_fires_while_the_human_plays(material, name):
    """P1: the silence detector was INVERTED on real input.

    On a CONTINUOUS click (bus RMS -31 dB, floor -45 dB) the log walked the
    whole ladder — ``thinned: human silent 2 bars`` -> ``fading: ... 6 bars``
    -> ``tacet: ... 10 bars`` — while the human played every beat of every bar.
    """
    m = material[name]
    # The final bar is excluded: plume's 32-bar arcs end on an "outro" section
    # that genuinely thins out, and responding to that with `fading` is the
    # feature working, not the bug.  Everything before it is continuous
    # playing and must draw no silence response at all.
    body = m.result.states_by_bar[:-1]
    ladder = [s for s in body if s in ("thinned", "fading", "tacet")]
    assert not ladder, (
        f"{name}: silence ladder fired on continuous playing: "
        f"{m.result.states_by_bar}")
    assert "tacet" not in m.result.states_by_bar
    silent_claims = [ln for ln in m.lines
                     if "silent" in str(ln.get("reason", ""))]
    assert len(silent_claims) <= 1, (
        f"{name}: claimed the human was silent while they played: "
        f"{[ln['reason'] for ln in silent_claims][:3]}")


@pytest.mark.parametrize("name", ALL)
def test_the_machine_is_audible_and_under_the_human(material, name):
    """Entering means sound on the machine bus, below the human's level."""
    m = material[name]
    bus = m.result.manifest["per_bus_rms_db"]
    assert bus["machine"] > -60.0, f"{name}: machine bus is silent ({bus})"
    if m.key is None:
        # A bare click has a ~5% duty cycle, so its MEAN bus RMS sits ~25 dB
        # below its peaks and a "machine under human" comparison in mean RMS
        # is meaningless for it.  (It is also where the entry calibration's
        # +/-24 dB clamp binds.)  Level balance is asserted on the sustained
        # material below.
        return
    assert bus["machine"] < bus["human"] + 6.0, (
        f"{name}: machine is louder than the human ({bus})")


@pytest.mark.parametrize("name", ALL)
def test_no_entry_failure_is_explained_in_one_line(material, name):
    """When the duet does happen, no_entry_reason must be absent — and when it
    does not, the manifest must say which gate stayed closed rather than
    sending the user to NDJSON forensics."""
    m = material[name]
    if m.result.entered_bar is None:             # pragma: no cover - guard
        assert m.result.manifest["no_entry_reason"]
    else:
        assert m.result.manifest["no_entry_reason"] is None


def test_entry_threshold_stays_below_the_tracker_lock_threshold():
    """Guards the tie that made entry a coin flip.

    ``MindParams.entry_conf`` was EXACTLY ``BeatTracker``'s ``lock_conf``
    (0.55 == 0.55), and the mind samples confidence instantaneously at the bar
    boundary while the tracker's lock carries hysteresis — so the tracker
    stayed locked while confidence oscillated around 0.5 and the mind's
    re-check almost never coincided with an entry boundary.
    """
    assert MindParams().entry_conf < BeatTracker()._lock_conf


# ------------------------------------------------------- silence, for real


def test_a_real_stop_is_still_detected(material):
    """The relative silence rule must not make the detector deaf.

    The v1.2 rule is onset-guarded and level-relative; this asserts it still
    hears an actual stop, so the fix cannot be "call nothing silent".
    """
    m = material["lofi80"]
    x, sr = sf.read(m.path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    half = x.size // 2
    stopped = x.copy()
    stopped[half:half + int(12.0 * sr)] = 0.0          # a long, real stop
    d = Path(m.result.out_path).parent
    sf.write(str(d / "stopped.wav"), stopped, sr, subtype="PCM_24")
    res = sim_mod.simulate(str(d / "stopped.wav"), style="lofi",
                           out_path=str(d / "duet_stopped.wav"), seed=0)
    assert res.entered_bar is not None
    assert any(s in ("thinned", "fading", "tacet") for s in res.states_by_bar), (
        f"a 12-second stop produced no silence response: {res.states_by_bar}")


def test_a_sounding_plan_off_a_bar_boundary_actually_renders(material,
                                                             monkeypatch):
    """The beat-loss repair's `fading` bar must make sound.

    FAILED BEFORE v1.2: `fading` is issued by `mind._plan_beat_loss`, which on
    a mid-take rubato fires from the UNLOCKED tick path — and that path logged
    the plan and rendered nothing, so the machine stopped dead instead of
    fading out, which is the one outcome the repair exists to avoid.  The
    tracker is forced to report unlocked for a window here because the natural
    race (conf_floor == UNLOCK_CONF) usually lands the first repair bar on a
    boundary instead.
    """
    import dataclasses

    from plume.listen.beat import BeatTracker

    m = material["lofi80"]
    orig = BeatTracker._state
    entered = [ln for ln in m.lines if ln.get("state") == "entering"][0]
    bar_s = 4.0 * 60.0 / m.bpm
    lo = entered["t_s"] + 4 * bar_s          # well into the comping arc
    hi = lo + 10.0

    def unlocked_window(self):
        st = orig(self)
        if lo <= st.t_s <= hi and st.locked:
            return dataclasses.replace(st, locked=False)
        return st

    monkeypatch.setattr(BeatTracker, "_state", unlocked_window)
    d = Path(m.result.out_path).parent
    res = sim_mod.simulate(m.path, style="lofi",
                           out_path=str(d / "duet_unlocked.wav"), seed=0)
    lines = [json.loads(ln) for ln
             in Path(res.decisions_path).read_text().splitlines()]
    sounding = [ln for ln in lines
                if ln.get("state") in ("fading", "entering", "comping",
                                       "thinned")
                and lo <= ln.get("t_s", -1) <= hi]
    assert sounding, "no sounding plan was issued during the unlocked window"
    mix, sr = sf.read(str(d / "duet_unlocked.wav"), dtype="float64",
                      always_2d=True)
    mix = mix.mean(axis=1)
    ln0 = sounding[0]
    a = int(ln0["t_s"] * sr)
    seg = mix[a:a + int(1.5 * sr)]
    rms = 20.0 * math.log10(max(float(np.sqrt(np.mean(seg ** 2))), 1e-9))
    assert rms > -40.0, (
        f"a {ln0['state']!r} plan at t={ln0['t_s']:.2f}s produced silence "
        f"({rms:.1f} dB)")


def test_determinism_on_real_material(material):
    """Same input, same args -> byte-identical decisions and audio."""
    m = material["trap140"]
    d = Path(m.result.out_path).parent
    a = sim_mod.simulate(m.path, style="lofi", out_path=str(d / "det_a.wav"),
                         seed=0)
    b = sim_mod.simulate(m.path, style="lofi", out_path=str(d / "det_b.wav"),
                         seed=0)
    assert a.states_by_bar == b.states_by_bar
    assert a.bpm_trace == b.bpm_trace
    assert a.entered_bar == b.entered_bar
    assert (Path(a.decisions_path).read_bytes()
            == Path(b.decisions_path).read_bytes())
    assert (Path(a.out_path).read_bytes() == Path(b.out_path).read_bytes())


def test_frame_features_are_independent_of_blocking_on_real_audio(material):
    """The longer chroma window must not have broken block-invariance."""
    m = material["synth100"]
    x, sr = sf.read(m.path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)[: 20 * SR]

    def run(chunk):
        fe = FeatureExtractor(sr=sr)
        out = []
        for i in range(0, x.size, chunk):
            out += fe.process(x[i:i + chunk])
        return out + fe.flush()

    a, b = run(4096), run(997)
    assert len(a) == len(b)
    for fa, fb in zip(a, b):
        assert fa.t_s == fb.t_s
        assert fa.rms_db == fb.rms_db
        assert fa.flux == fb.flux
        assert fa.onset == fb.onset
        assert np.array_equal(fa.chroma, fb.chroma)


def test_bar_silence_is_relative_not_absolute():
    """A loud bar and the SAME bar 30 dB quieter must agree about silence.

    This is the axis ``tests/fixtures_gen.py`` has none of: every fixture is
    peak-normalized, so the absolute floor was 21-30 dB away and unreachable.
    """
    acc_loud = sim_mod._BarAccum()
    acc_loud.reset(0.0)
    acc_quiet = sim_mod._BarAccum()
    acc_quiet.reset(0.0)
    for i in range(200):                          # a sparse percussive bar
        t = 0.01 * (i + 1)
        hit = i % 50 == 0
        loud = -12.0 if hit else -52.0
        acc_loud.frames.append((t, loud, hit))
        acc_quiet.frames.append((t, loud - 30.0, hit))
    p = MindParams()
    assert not acc_loud.is_silent(2.0, p.silence_rms_db, -20.0)
    assert not acc_quiet.is_silent(2.0, p.silence_rms_db, -50.0), (
        "a bar with onsets read as silent purely because it was recorded quiet")


def test_onsets_veto_silence_regardless_of_level():
    """Detected onsets are level-independent evidence that the human plays."""
    acc = sim_mod._BarAccum()
    acc.reset(0.0)
    for i in range(200):
        acc.frames.append((0.01 * (i + 1), -80.0, i % 50 == 0))
    assert not acc.is_silent(2.0, -45.0, None)
    acc.frames = [(t, r, False) for t, r, _ in acc.frames]
    assert acc.is_silent(2.0, -45.0, None), "no onsets and -80 dB is silence"


def test_mind_trusts_the_bar_features_silent_flag():
    """The mind must not re-derive silence from the RMS scalar.

    FAILED BEFORE v1.2: ``plan_bar`` OR-ed in its own absolute-RMS test, a
    second independent copy of the gate that the feature extractor's onset
    guard could not reach — so fixing ``sim`` alone changed nothing at all.
    This is the exact case ``tests/test_mind.py``'s ``mk_bar`` (default
    ``rms=-20.0``) never constructs.
    """
    from plume.listen.beat import BeatState
    from plume.listen.harmony import HarmonyState
    from plume.mind import BarFeatures, Mind

    mind = Mind(MindParams())
    beat = BeatState(t_s=1.0, bpm=120.0, confidence=0.9, locked=True, bar=1,
                     beat_in_bar=0, phase_beats=0.0, next_beat_t_s=1.5,
                     next_downbeat_t_s=2.0, beat_interval_s=0.5)
    harm = HarmonyState(t_s=1.0, key="C major", key_conf=0.8, chord="C",
                        chord_conf=0.8, register_low_midi=60.0,
                        register_high_midi=72.0)
    bar = BarFeatures(bar=0, rms_db=-52.0, onset_density=1.0, silent=False,
                      register_low_midi=60.0, register_high_midi=72.0)
    mind.plan_bar(beat, harm, bar)
    assert mind._bars_listened == 1, (
        "a bar the extractor called NOT silent was counted as silence "
        "because its RMS was under an absolute floor")


def test_bars_heard_before_the_lock_are_credited(material):
    """``min_listen_bars`` means bars HEARD, not bars after an internal lock.

    FAILED BEFORE v1.2: the only call site that ever passed a ``BarFeatures``
    was the locked-boundary path, and the accumulator was never reset before
    the first boundary — so 11.9 seconds of real music (about 5 bars) was
    credited as exactly ONE bar, and entry landed halfway through the take.
    """
    m = material["lofi80"]
    entered = [ln for ln in m.lines if ln.get("state") == "entering"][0]
    locks = [ln for ln in m.lines if ln.get("event") == "lock"]
    bar_s = 4.0 * 60.0 / m.bpm
    assert entered["t_s"] - locks[0]["t_s"] < 8 * bar_s, (
        "entry waited a full listen window AFTER the lock, on top of "
        "everything heard before it")


@pytest.mark.parametrize("name", ALL)
def test_reason_strings_are_clean(material, name):
    """The decisions log is the user-facing forensics surface."""
    m = material[name]
    for ln in m.lines:
        reason = str(ln.get("reason", ""))
        assert "?" not in reason, f"{name}: placeholder in reason {reason!r}"
        assert "-0.0dB" not in reason, f"{name}: negative-zero gain {reason!r}"
        assert "None" not in reason, f"{name}: unformatted None in {reason!r}"
        if ln.get("chord") is None and "chord_conf" in ln:
            assert not ln["chord_conf"]
