"""Tests for plume.sim (SPEC-V1 section 6).

The whole sim pipeline requires ``plume.mind`` (builder D, concurrent) —
``sim`` imports it at module level per the frozen contract.  Until the mind
lands, every test here is skipped with a clear reason; duet-only coverage
lives in tests/test_duet.py and runs standalone.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip(
    "plume.mind",
    reason="plume.mind not built yet (builder D); sim integrates against it")

from plume.duet import DuetEngine  # noqa: E402
from plume.mind import MindParams, STATES  # noqa: E402
from plume.sim import _SOUNDING_STATES, BLOCK, SimResult, simulate  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fixtures_gen import make_human_with_truth  # noqa: E402

SR = 48000
_DECISION_KEYS = {"bar", "t_s", "state", "reason", "bpm", "conf", "key",
                  "chord", "roles", "gain_db"}


@pytest.fixture(scope="module")
def steady_wav(tmp_path_factory):
    d = tmp_path_factory.mktemp("simfx")
    audio, truth = make_human_with_truth("steady", bpm=96, bars=16,
                                         key="A minor", seed=1)
    path = d / "steady.wav"
    sf.write(str(path), audio, SR, subtype="FLOAT")
    return str(path), truth


@pytest.fixture(scope="module")
def steady_run(steady_wav, tmp_path_factory):
    wav, truth = steady_wav
    d = tmp_path_factory.mktemp("simout")
    res = simulate(wav, style="lofi", out_path=str(d / "duet.wav"), seed=5)
    return res, truth


# ---------------------------------------------------------------------------
# pipeline smoke
# ---------------------------------------------------------------------------

def test_simresult_fields_and_files(steady_run):
    res, truth = steady_run
    assert isinstance(res, SimResult)
    assert pathlib.Path(res.out_path).is_file()
    assert pathlib.Path(res.decisions_path).is_file()
    data, sr = sf.read(res.out_path, always_2d=True)
    assert sr == SR and data.shape[1] == 2
    assert res.duration_s == pytest.approx(data.shape[0] / SR, abs=1e-6)
    # output covers at least the human take
    human_s = truth["bar_times_s"][-1]
    assert res.duration_s >= human_s
    # limiter ceiling respected
    assert float(np.max(np.abs(data))) <= 10.0 ** (-1.0 / 20.0) + 1e-4


def test_states_and_traces(steady_run):
    res, truth = steady_run
    assert res.states_by_bar, "locked steady input must produce bar plans"
    assert set(res.states_by_bar) <= set(STATES)
    assert res.bpm_trace and all(t >= 0 and b > 0 for t, b in res.bpm_trace)
    # tempo settles near truth (±3 bpm, allowing half/double none)
    late = [b for t, b in res.bpm_trace if t > truth["bar_times_s"][6]]
    assert late and abs(late[-1] - truth["bpm_base"]) <= 3.0
    # A key belief exists by the end of the take.  NOTE: asserting the
    # *correct* tonic (A or relative C) is behavior test 10's job and
    # currently fails upstream of sim: listen/features.py maps rFFT bins to
    # pitch classes rectangularly, and at WIN=2048/48k (23.4 Hz bins) the
    # fixture's dominant 110 Hz sub-bass lands between bins (93.75/117.2 Hz
    # -> F#/Bb), tritone-skewing the chroma histogram.  Fix belongs in
    # features.py (chroma fmin above the unresolvable register, or
    # interpolated bin weighting), owned by the listen builder.
    assert res.final_key is not None


def test_entry_is_late_and_on_boundary(steady_run):
    res, _ = steady_run
    p = MindParams()
    assert res.entered_bar is not None
    assert res.entered_bar >= p.min_listen_bars
    assert res.entered_bar % p.entry_boundary_bars == 0


def test_decisions_ndjson_schema(steady_run):
    res, _ = steady_run
    lines = [json.loads(ln) for ln in
             pathlib.Path(res.decisions_path).read_text().splitlines()]
    assert lines
    bar_lines = [ln for ln in lines if "event" not in ln]
    event_lines = [ln for ln in lines if "event" in ln]
    assert bar_lines and event_lines
    for ln in bar_lines:
        assert _DECISION_KEYS <= set(ln)
        assert ln["state"] in STATES
        assert isinstance(ln["roles"], dict)
        assert isinstance(ln["reason"], str) and ln["reason"]
    assert any(ln["event"] == "lock" for ln in event_lines)
    assert any(ln["event"] == "key_change" for ln in event_lines)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_byte_identical_reruns(steady_wav, tmp_path):
    wav, _ = steady_wav
    a = simulate(wav, style="lofi", out_path=str(tmp_path / "a.wav"), seed=5)
    b = simulate(wav, style="lofi", out_path=str(tmp_path / "b.wav"), seed=5)
    assert pathlib.Path(a.out_path).read_bytes() == \
        pathlib.Path(b.out_path).read_bytes()
    assert pathlib.Path(a.decisions_path).read_text() == \
        pathlib.Path(b.decisions_path).read_text()
    assert a.states_by_bar == b.states_by_bar
    assert a.bpm_trace == b.bpm_trace


# ---------------------------------------------------------------------------
# taps count-in + mix level
# ---------------------------------------------------------------------------

def test_taps_countin_locks_immediately(tmp_path):
    audio, truth = make_human_with_truth("steady", bpm=96, bars=10,
                                         key="A minor", seed=2)
    wav = tmp_path / "h.wav"
    sf.write(str(wav), audio, SR, subtype="FLOAT")
    beat = 60.0 / 96.0
    taps = [k * beat for k in range(4)]
    res = simulate(str(wav), out_path=str(tmp_path / "o.wav"), seed=0,
                   taps_s=taps)
    p = MindParams()
    # locked from the count-in: bpm correct essentially immediately
    early = [b for t, b in res.bpm_trace if t < 4.0]
    assert early and all(abs(b - 96.0) <= 3.0 for b in early)
    assert res.entered_bar is not None
    assert res.entered_bar <= p.min_listen_bars + p.entry_boundary_bars


def test_mix_human_db_scales_human_bus(steady_wav, tmp_path):
    wav, _ = steady_wav
    ref = simulate(wav, out_path=str(tmp_path / "r.wav"), seed=5)
    quiet = simulate(wav, out_path=str(tmp_path / "q.wav"), seed=5,
                     mix_human_db=-40.0)
    h0 = ref.manifest["per_bus_rms_db"]["human"]
    h1 = quiet.manifest["per_bus_rms_db"]["human"]
    assert h0 - h1 == pytest.approx(40.0, abs=0.5)
    # machine bus unaffected by the human mix gain
    assert quiet.manifest["per_bus_rms_db"]["machine"] == \
        ref.manifest["per_bus_rms_db"]["machine"]


def test_manifest_contents(steady_run):
    res, _ = steady_run
    m = res.manifest
    for k in ("peak_db", "rms_db", "per_bus_rms_db", "params", "style",
              "seed", "sr", "duration_s", "entered_bar",
              "machine_cal_db", "machine_cal_bar", "machine_cal_trace"):
        assert k in m
    assert m["style"] == "lofi" and m["sr"] == SR
    assert set(m["per_bus_rms_db"]) == {"human", "machine"}
    assert m["params"]["min_listen_bars"] == MindParams().min_listen_bars
    assert BLOCK == 4096


# ---------------------------------------------------------------------------
# machine-bus calibration — measured OUTSIDE the fade
# ---------------------------------------------------------------------------
#
# ``BarPlan.gain_db`` is relative to the human, so sim solves one number that
# sets how loud the machine is against them for the whole take.  It used to
# solve that on the DELIVERED entry bar — which, since the gentle-fade work,
# sits inside the two-bar fade-in — so it solved `delivered * cal == target`
# when it meant `own_level * cal == target`, and over-boosted the take by the
# fade envelope's RMS (+3.2 dB overall, +1.35 dB steady-state on this
# fixture).  It self-corrected on the first `comping` bar, but only if
# `entering -> comping` ever fired, and the committed mind often goes
# `entering -> thinned` instead, so the boost simply stayed.
#
# The invariant below is the fix stated as a property: the calibration is a
# function of the human's level and the bar's OWN level, and the fade is not
# in it.  Every solve is compared, so the one taken inside the fade — the
# entry bar, the one that was wrong — is covered directly.


class _NoFadeEngine(DuetEngine):
    """The style's engine with only the fade lengths zeroed.

    Everything else — texture crossfade window, glide, announce levels — is
    left at the style's values, so the two runs differ ONLY by the envelope
    whose influence on the calibration is being denied.  (Zeroing the
    crossfade too would change the raw render, and the comparison would be
    measuring that instead.)
    """

    def __init__(self, **kw):
        super().__init__(fade_in_bars=0.0, fade_out_bars=0.0, **kw)


def _run(wav, out, monkeypatch=None, **kw):
    if monkeypatch is not None:
        monkeypatch.setattr("plume.sim.DuetEngine", _NoFadeEngine)
    return simulate(str(wav), style="lofi", out_path=str(out), seed=5, **kw)


def test_the_bus_calibration_is_identical_whether_or_not_a_fade_runs(
        steady_wav, tmp_path, monkeypatch):
    """THE headline invariant — exactly equal, not merely close."""
    wav, _ = steady_wav
    faded = _run(wav, tmp_path / "faded.wav")
    flat = _run(wav, tmp_path / "flat.wav", monkeypatch=monkeypatch)

    assert faded.manifest["machine_cal_trace"], \
        "nothing calibrated — the invariant would hold vacuously"
    assert faded.manifest["machine_cal_trace"] == \
        flat.manifest["machine_cal_trace"], (
            "the calibration moved with the fade: "
            f"{faded.manifest['machine_cal_trace']} vs "
            f"{flat.manifest['machine_cal_trace']}")
    assert faded.manifest["machine_cal_db"] == flat.manifest["machine_cal_db"]
    assert faded.manifest["machine_cal_bar"] == flat.manifest["machine_cal_bar"]


def test_the_entry_bar_really_is_calibrated_inside_its_fade(
        steady_wav, tmp_path, monkeypatch):
    """Guards the invariant above from passing vacuously, twice over.

    The first solve must land on an ``entering`` bar — the bar that is inside
    the fade-in; if calibration only ever happened on a later ``comping`` bar
    there would be no fade for it to be invariant to — and the two runs must
    actually deliver different audio, or equal calibrations prove nothing.
    """
    wav, _ = steady_wav
    faded = _run(wav, tmp_path / "faded.wav")
    trace = faded.manifest["machine_cal_trace"]
    assert trace[0]["state"] == "entering", \
        f"the first calibration was taken on a {trace[0]['state']} bar"
    assert trace[0]["bar"] == faded.entered_bar

    flat = _run(wav, tmp_path / "flat.wav", monkeypatch=monkeypatch)
    assert pathlib.Path(faded.out_path).read_bytes() != \
        pathlib.Path(flat.out_path).read_bytes(), \
        "the fade changed nothing, so nothing is being controlled for"


@pytest.fixture(scope="module")
def thinning_wav(tmp_path_factory):
    """A take on which the mind goes ``entering -> thinned``, never ``comping``.

    This is the shape that made the old bug permanent rather than transient:
    the calibration was solved once, on the entry bar, INSIDE the fade — and
    the ``comping`` bar that used to re-solve it never came.
    """
    d = tmp_path_factory.mktemp("simthin")
    audio, truth = make_human_with_truth("long_stop", bpm=96, bars=20,
                                         key="A minor", seed=3)
    path = d / "long_stop.wav"
    sf.write(str(path), audio, SR, subtype="FLOAT")
    return str(path), truth


def test_the_calibration_survives_an_entrance_that_never_reaches_comping(
        thinning_wav, tmp_path, monkeypatch):
    """The bug's actual shape, end to end.

    One solve, on an ``entering`` bar, with a two-bar fade-in running — and
    no second solve to rescue it.  The old code was over-boosted for the
    whole take here; the number must now be the un-faded one.
    """
    wav, _ = thinning_wav
    faded = _run(wav, tmp_path / "faded.wav")
    trace = faded.manifest["machine_cal_trace"]
    assert [t["state"] for t in trace] == ["entering"], (
        "this fixture is supposed to enter and never reach comping, so the "
        f"entry solve is the only one — got {trace}")

    flat = _run(wav, tmp_path / "flat.wav", monkeypatch=monkeypatch)
    assert trace == flat.manifest["machine_cal_trace"], (
        "the one and only calibration moved with the fade: "
        f"{trace} vs {flat.manifest['machine_cal_trace']}")
    assert faded.manifest["per_bus_rms_db"]["machine"] != \
        flat.manifest["per_bus_rms_db"]["machine"], \
        "the fade changed nothing, so nothing is being controlled for"


def test_sim_renders_an_off_boundary_sounding_bar_as_a_continuation(
        thinning_wav, tmp_path, monkeypatch):
    """The mind's held bars arrive off the bar grid, with the counter frozen.

    While the tracker is unlocked the mind is ticked at 1 Hz and can still
    return a SOUNDING plan (beat-loss repair, and up to four held wobble bars
    while committed).  The beat counter freezes on unlock, so every one of
    them carries a bar number the engine has already seen — which is also how
    the LIVE planner signals "throw the last render away and redo this bar".
    They want opposite things from the fade envelope, so sim has to say which
    it is, and a held bar is a continuation: rolling the envelope back for it
    replays the same slice of the fade-in and the entrance restarts.
    """
    wav, _ = thinning_wav
    calls: list[tuple[int, str, bool]] = []

    class _Rec(DuetEngine):
        def render_bar(self, plan, beat, harmony, *, continuation=False):
            calls.append((int(getattr(plan, "bar", 0)), str(plan.state),
                          bool(continuation)))
            return super().render_bar(plan, beat, harmony,
                                      continuation=continuation)

    monkeypatch.setattr("plume.sim.DuetEngine", _Rec)
    simulate(str(wav), style="lofi", out_path=str(tmp_path / "o.wav"), seed=5)

    held = [c for c in calls if c[2]]
    assert held, ("no off-boundary sounding bar was rendered on this fixture "
                  "— the continuation contract is not being exercised")
    bars = [b for b, _, _ in calls]
    for bar, state, _ in held:
        assert state in _SOUNDING_STATES, \
            f"bar {bar} was continued but is {state}, which makes no sound"
        assert bars.count(bar) > 1, (
            f"bar {bar} was flagged as a continuation but never repeats — "
            "the flag is being set on bars that do not need it")
    assert not any(c[2] for c in calls if c[1] not in _SOUNDING_STATES)


def test_the_entry_lands_at_its_specified_db_under_the_human(
        steady_wav, tmp_path, monkeypatch):
    """The other half: invariance is worthless if the level is wrong.

    Measured on the machine bus alone (the human's fader taken to silence —
    it does not touch the mind, which reads the input WAV) with the fade off,
    so the entry bar is delivered at the level it was calibrated to.
    """
    wav, _ = steady_wav
    res = _run(wav, tmp_path / "m.wav", monkeypatch=monkeypatch,
               mix_human_db=-200.0)
    machine, sr = sf.read(res.out_path, always_2d=True)
    human, _ = sf.read(str(wav), always_2d=True)

    lines = [json.loads(ln) for ln in
             pathlib.Path(res.decisions_path).read_text().splitlines()]
    entering = [ln for ln in lines
                if "event" not in ln and ln["state"] == "entering"]
    assert entering, "the machine never entered on this fixture"
    ln = entering[0]

    bar_n = int(round(4.0 * (60.0 / float(ln["bpm"])) * sr))
    i0 = int(round(float(ln["t_s"]) * sr))
    m_bar = machine[i0:i0 + bar_n].mean(axis=1)
    h_bar = human[i0:i0 + bar_n].mean(axis=1)

    def _rms_db(x):
        return 20.0 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-12))

    under = _rms_db(m_bar) - _rms_db(h_bar)
    assert under == pytest.approx(float(ln["gain_db"]), abs=1.0), (
        f"the entry sits {under:.2f} dB under the human, but the plan asked "
        f"for {ln['gain_db']:.2f} dB")
