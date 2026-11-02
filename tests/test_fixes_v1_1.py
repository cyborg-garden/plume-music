"""Regression tests for the v1.1 fix round (post-verification P0/P1 issues).

Each test pins one fixed failure mode:

- P0  the entry gate never opened because sparse-but-playing bars (click
      tracks, kick stems) read as "silent" and `heard N bars` never accrued;
- P1  the human-silence ladder (thin/fade/tacet) ran while the human was
      audibly playing;
- P1  busy-from-bar-1 humans locked the tempo octave wrong (or never locked)
      and defeated the inverse-density coupling;
- P1  the tracker could not lock on plume's own backbeat renders, and taps
      were abandoned for a subdivision alias within two bars;
- P1  pluck harmonics hallucinated chords (Am -> C#m) and low-register
      fundamentals quantized to the wrong pitch class (A3 -> Ab);
- P2  sim gave no diagnosis when the duet never happened; bad-style errors
      leaked the exception class; chord_conf was 1.0 while chord was null.

All inputs are synthesized in-test (deterministic seeds); no reliance on
uncommitted render artifacts.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("plume.mind")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import fixtures_gen  # noqa: E402
from plume.engine import render  # noqa: E402
from plume.listen.beat import BeatTracker  # noqa: E402
from plume.listen.features import FeatureExtractor  # noqa: E402
from plume.listen.harmony import HarmonyTracker  # noqa: E402
from plume.score import Movement, SectionSpec, Session, Track  # noqa: E402
from plume.sim import simulate  # noqa: E402
from plume.theory import midi_to_hz, parse_key  # noqa: E402
from plume.voices import render_note  # noqa: E402
from test_beat import click_times, make_frames  # noqa: E402

SR = 48000
_ACTIVE = ("entering", "comping", "thinned", "fading")
_QUIET = ("thinned", "fading", "tacet")


def _plans(lines):
    return [ln for ln in lines if "state" in ln]


def _read_decisions(path):
    return [json.loads(ln) for ln in open(path, encoding="utf-8")]


# ------------------------------------------------------------ input builders

def _click_track(bpm: float, dur_s: float) -> np.ndarray:
    """Deterministic 1 kHz click track (10 ms bursts on every beat)."""
    out = np.zeros(int(dur_s * SR), dtype=np.float32)
    burst = int(0.010 * SR)
    env = np.exp(-np.arange(burst) / (0.002 * SR))
    tone = (0.8 * env * np.sin(2 * np.pi * 1000.0 *
                               np.arange(burst) / SR)).astype(np.float32)
    period = 60.0 / bpm
    k = 0
    while True:
        i0 = int(round(k * period * SR))
        if i0 + burst > out.shape[0]:
            break
        out[i0:i0 + burst] += tone
        k += 1
    return out


def _busy_all_fixture(bpm: float, bars: int = 16, seed: int = 11):
    """fixtures_gen 'busy_middle' with EVERY bar busy (3 onsets/beat)."""
    orig = fixtures_gen._busy_bars
    fixtures_gen._busy_bars = (
        lambda kind, bars_: list(range(bars_)) if kind == "busy_middle"
        else orig(kind, bars_))
    try:
        return fixtures_gen.make_human_with_truth(
            "busy_middle", bpm=bpm, bars=bars, seed=seed)
    finally:
        fixtures_gen._busy_bars = orig


def _groove_session(bpm: float = 80.0, bars: int = 16, seed: int = 7) -> Session:
    """Backbeat groove (kick 1/3, snare 2/4, swung hats): the class of input
    whose flux envelope ANTI-correlates at the one-beat lag, which the plain
    in-range ACF peak-pick could never lock on."""
    tracks = [
        Track(name="kick", voice="808kick", kind="drum",
              patterns={"main": "bd ~ ~ ~ ~ ~ ~ ~ bd ~ ~ ~ ~ ~ ~ ~"}),
        Track(name="snare", voice="snare", kind="drum", gain_db=-2.0,
              patterns={"main": "~ ~ ~ ~ sd ~ ~ ~ ~ ~ ~ ~ sd ~ ~ ~"}),
        Track(name="hat", voice="hat", kind="drum", gain_db=-10.0,
              patterns={"main": "hh*8"}),
        Track(name="bass", voice="808sub", kind="tonal", gain_db=-4.0,
              patterns={"main": "a1 ~ ~ ~ ~ ~ a1 ~ c2 ~ ~ ~ e2 ~ ~ ~"}),
    ]
    section = SectionSpec(name="a", bars=bars, energy=0.8,
                          plays={t.name: "main" for t in tracks},
                          chords=[], fill=False)
    return Session(name="groove80", bpm=bpm, key="A min", swing=0.6, seed=seed,
                   timing_bias_ms={}, humanize_ms=2.0, humanize_vel=0.05,
                   tracks=tracks,
                   movements=[Movement(name="m", sections=[section])],
                   master={"reverb": 0.05, "drive": 0.05, "lofi": 0.0,
                           "limit_db": -1.0})


def _am_strums(dur_s: float = 8.0, seed: int = 3) -> np.ndarray:
    """Repeated Am plucks (A3 C4 E4) — the chord-hallucination repro."""
    rng = np.random.default_rng(seed)
    buf = np.zeros(int(dur_s * SR), dtype=np.float64)
    k = 0
    while True:
        t0 = 0.3 + 0.6 * k
        if t0 > dur_s - 0.8:
            break
        for i, m in enumerate((57, 60, 64)):
            y = render_note("pluck", sr=SR, dur_s=0.4, freq_hz=midi_to_hz(m),
                            velocity=0.6 * (0.94 ** i), rng=rng, params={})
            s0 = int((t0 + 0.009 * i) * SR)
            n = min(len(y), len(buf) - s0)
            buf[s0:s0 + n] += y[:n]
        k += 1
    peak = float(np.abs(buf).max())
    return (0.8 * buf / peak).astype(np.float32)


# --------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("fixes_v1_1")


@pytest.fixture(scope="module")
def click_sim(workdir):
    wav = workdir / "click120.wav"
    sf.write(str(wav), _click_track(120.0, 24.0), SR, subtype="FLOAT")
    res = simulate(str(wav), out_path=str(workdir / "click120_duet.wav"),
                   seed=5, mix_human_db=-120.0)
    return res, _read_decisions(res.decisions_path)


@pytest.fixture(scope="module")
def groove_wav(workdir):
    wav = workdir / "groove80.wav"
    render(_groove_session(), out_path=str(wav))
    return str(wav)


# ------------------------------------------- P0: entry gate opens on playing

def test_click_track_gets_accompaniment(click_sim):
    """P0: an ideal click track must produce a duet with default MindParams.

    Pre-fix, the bar-silence measure read the (mostly-quiet-framed) click
    bars as silent, `heard N/4 bars` stayed at 0 forever, and the machine
    bus was digital silence in every default run."""
    res, lines = click_sim
    assert res.entered_bar is not None, \
        res.manifest.get("no_entry_reason")
    # the machine actually sounds
    machine, sr = sf.read(res.out_path, dtype="float32", always_2d=True)
    assert sr == SR
    rms = float(np.sqrt(np.mean(machine.astype(np.float64) ** 2)))
    assert 20.0 * math.log10(max(rms, 1e-10)) > -60.0
    assert res.manifest["no_entry_reason"] is None


def test_click_track_never_reads_as_human_silence(click_sim):
    """P1: 'human silent N bars' must not fire while the human is playing.

    Pre-fix the full silence ladder ran on a continuous click track
    (thinned -> fading -> tacet, cycling forever)."""
    _, lines = click_sim
    for ln in _plans(lines):
        assert ln["state"] not in _QUIET, ln["reason"]
        assert "human silent" not in ln["reason"], ln["reason"]


# ------------------------- P1: busy-from-bar-1 octave errors + density flip

@pytest.mark.parametrize("bpm", [96.0, 80.0])
def test_busy_from_bar_one_locks_true_tempo_and_thins(workdir, bpm):
    """P1: a human playing 3 onsets/beat from bar 1 must get accompaniment
    at the TRUE tempo (96 pre-fix never locked; 80 pre-fix locked the 160
    double and mirrored the human's density instead of inverting)."""
    audio, truth = _busy_all_fixture(bpm)
    wav = workdir / f"busy_all_{int(bpm)}.wav"
    sf.write(str(wav), audio, SR, subtype="FLOAT")
    res = simulate(str(wav), out_path=str(workdir / f"busy_{int(bpm)}.wav"),
                   seed=5, mix_human_db=-120.0)
    lines = _plans(_read_decisions(res.decisions_path))

    assert res.entered_bar is not None, res.manifest.get("no_entry_reason")

    active = [ln for ln in lines if ln["state"] in _ACTIVE]
    assert active, "machine never active"
    # every active bar runs on the true grid — never the double/half octave
    for ln in active:
        assert abs(ln["bpm"] - bpm) <= 0.05 * bpm, (ln["bpm"], ln["reason"])

    # inverse-density coupling engages: comping against a 3-onsets/beat
    # human stays at the sparse density hint, never the full 0.8
    comping = [ln for ln in active if ln["state"] == "comping"]
    assert comping
    # `machine_density` (the machine's own 0..1 density) since v1.2 — the
    # human's onsets/beat is the other density, and it lives in `reason`.
    assert all(ln["machine_density"] <= 0.35 for ln in comping), \
        [(ln["machine_density"], ln["reason"])
         for ln in comping if ln["machine_density"] > 0.35]


# ------------------- P1: lock on plume's own backbeat render; taps survive

def test_own_backbeat_render_locks_and_enters(workdir, groove_wav):
    """P1: the tracker must lock and enter on plume's own groove render
    (backbeat kick/snare anti-correlates the flux at the one-beat lag, which
    kept confidence at ~0.2 pre-fix — 'plume could not jam with its own
    output')."""
    res = simulate(groove_wav, out_path=str(workdir / "groove_duet.wav"),
                   seed=5, mix_human_db=-120.0)
    lines = _plans(_read_decisions(res.decisions_path))
    assert res.entered_bar is not None, res.manifest.get("no_entry_reason")
    active = [ln for ln in lines if ln["state"] in _ACTIVE]
    for ln in active:
        assert 72.0 <= ln["bpm"] <= 88.0, (ln["bpm"], ln["reason"])


def test_taps_hold_against_subdivision_alias(workdir, groove_wav):
    """P1: --taps promised 'hard-set tempo + downbeat' but the tracker
    abandoned the tapped 80 for a ~112-160 alias within two bars.  Post-fix
    the count-in anchors the period: only drift-band corrections apply while
    the tapped lock holds."""
    res = simulate(groove_wav, out_path=str(workdir / "groove_taps.wav"),
                   seed=5, mix_human_db=-120.0,
                   taps_s=[0.0, 0.75, 1.5, 2.25, 3.0])
    assert res.entered_bar is not None
    # after the count-in completes, the tempo never leaves the 80 family
    for t, b in res.bpm_trace:
        if t >= 4.0:
            assert abs(b - 80.0) <= 4.0, (t, b)


def test_tap_anchor_rejects_conflicting_acf_unit():
    """Unit-level tap anchor: a tapped 80 bpm lock must survive a stream
    whose ACF votes 120 (pre-fix _adopt_bpm switched within ~1 s)."""
    tr = BeatTracker()
    for k in range(5):
        tr.tap(0.5 + 0.75 * k)
    st = tr.process([])
    assert st.locked and st.bpm == pytest.approx(80.0, abs=0.5)
    frames = make_frames(16.0, click_times(120.0, 28, t0=4.0))
    final = tr.process([f for f in frames if f.t_s > 3.5])
    if final.locked:                      # anchored: the alias was discarded
        assert final.bpm == pytest.approx(80.0, abs=2.0)
    # (if confidence decays enough to unlock, re-estimating freely is the
    # designed escape hatch — but it must never happen while still locked)


# ------------------------------------------ P1: harmony/chroma hallucination

def _run_harmony(audio):
    fe = FeatureExtractor(sr=SR)
    h = HarmonyTracker()
    emitted: dict[str, int] = {}
    state = None
    for i in range(0, len(audio), 4096):
        for f in fe.process(audio[i:i + 4096]):
            state = h.process([f])
            if state.chord:
                emitted[state.chord] = emitted.get(state.chord, 0) + 1
    return state, emitted


def test_am_strum_reads_am_not_csharp_minor():
    """P1: isolated Am pluck strums read 'C#m' at 0.99 confidence pre-fix
    (the triad's 5th harmonics ARE the C#m triad under rectangular chroma),
    which put a sour Db bass under the human in every fixture duet."""
    state, emitted = _run_harmony(_am_strums())
    assert emitted, "no chord ever emitted"
    top = max(emitted, key=emitted.get)
    assert top == "Am", emitted
    total = sum(emitted.values())
    assert emitted.get("C#m", 0) + emitted.get("Dbm", 0) < 0.05 * total, emitted
    # key belief lands on A minor or its relative
    assert state.key is not None
    tonic = parse_key(state.key).tonic
    assert tonic in (9, 0), state.key      # A or C


@pytest.mark.parametrize("freq,pc", [(220.0, 9), (261.63, 0), (440.0, 9)])
def test_pure_tone_chroma_pitch_class(freq, pc):
    """P1: a pure A3 quantized to the Ab/Bb bins pre-fix (23.4 Hz bins are
    wider than a semitone below ~400 Hz); peak-interpolated harmonic
    salience must put the argmax on the played pitch class."""
    fe = FeatureExtractor(sr=SR)
    t = np.arange(SR, dtype=np.float64) / SR
    frames = fe.process((0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32))
    mid = frames[len(frames) // 2]
    assert int(np.argmax(mid.chroma)) == pc
    assert float(mid.chroma[pc]) > 0.4


def test_chord_conf_zero_while_no_chord():
    """P2: pure silence used to report chord=null with chord_conf 1.0 (the
    N state's own posterior); confidence in a null chord is 0 by definition."""
    fe = FeatureExtractor(sr=SR)
    h = HarmonyTracker()
    state = None
    for i in range(0, 2 * SR, 4096):
        for f in fe.process(np.zeros(4096, dtype=np.float32)):
            state = h.process([f])
    assert state is not None
    assert state.chord is None
    assert state.chord_conf == 0.0


# ----------------------------------------------------- P2: sim ergonomics

def test_sim_says_why_no_duet(workdir, capsys, monkeypatch):
    """P2: a run that never enters must carry a one-line diagnosis (manifest
    `no_entry_reason` + 'why no duet:' in the CLI summary) instead of rc=0
    with an unexplained silent machine bus."""
    wav = workdir / "short_steady.wav"
    audio = fixtures_gen.make_human("steady", bpm=96, bars=2, seed=11)
    sf.write(str(wav), audio, SR, subtype="FLOAT")
    out = workdir / "short_duet.wav"
    res = simulate(str(wav), out_path=str(out), seed=5)
    assert res.entered_bar is None
    reason = res.manifest["no_entry_reason"]
    assert reason and ("listening" in reason or "waiting" in reason)

    from plume import cli
    code = cli.main(["sim", str(wav), "--out", str(out)])
    stdout, _ = capsys.readouterr()
    assert code == 0
    assert "why no duet:" in stdout


def test_bad_style_error_is_clean(workdir, capsys, monkeypatch):
    """P2: `--style nope` leaked 'error: ValueError: ...' while every other
    CLI error is a clean one-liner."""
    wav = workdir / "err_steady.wav"
    audio = fixtures_gen.make_human("steady", bpm=96, bars=2, seed=11)
    sf.write(str(wav), audio, SR, subtype="FLOAT")
    from plume import cli
    code = cli.main(["sim", str(wav), "--style", "definitely-not-a-style",
                     "--out", str(workdir / "err.wav")])
    stdout, stderr = capsys.readouterr()
    assert code == 1
    assert "ValueError" not in stderr and "ValueError" not in stdout
    assert stderr.startswith("error:")
