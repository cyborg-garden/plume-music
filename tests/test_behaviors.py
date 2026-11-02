"""SPEC-V1 section 9 behavior matrix — the product claims, one test each.

Every test runs ``plume.sim.simulate`` over a fixtures_gen human (seed 11,
defaults: 96 bpm, 24 bars, A minor) and asserts on the decisions log, the
SimResult, and the machine bus audio (rendered with the human mixed at
-120 dB — ``mix_human_db`` only affects the final mix, never the analysis, so
the decisions are identical to a default run and the WAV *is* the machine bus).

Timing slack: predicted bar boundaries sit within a few tens of ms of the
truth grid, so bar-count bounds get ``_SLACK_S`` of tolerance when converted
to seconds.
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

from plume.listen.beat import BeatTracker  # noqa: E402,F401  (import check)
from plume.mind import MindParams  # noqa: E402
from plume.sim import simulate  # noqa: E402
from plume.theory import parse_key  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fixtures_gen import make_human_with_truth  # noqa: E402

SR = 48000
SEED_FIXTURE = 11
SEED_MACHINE = 5
_SLACK_S = 0.2
_P = MindParams()          # every threshold referenced below is the default

_ACTIVE = ("entering", "comping", "thinned", "fading")


# ------------------------------------------------------------------ caches

@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("behaviors")


@pytest.fixture(scope="module")
def fixture_cache(workdir):
    cache: dict = {}

    def get(kind: str):
        if kind not in cache:
            audio, truth = make_human_with_truth(kind, seed=SEED_FIXTURE)
            path = workdir / f"{kind}.wav"
            sf.write(str(path), audio, SR, subtype="FLOAT")
            cache[kind] = (str(path), audio, truth)
        return cache[kind]

    return get


@pytest.fixture(scope="module")
def sim_cache(workdir, fixture_cache):
    """Machine-bus renders (human at -120 dB) + parsed decision logs."""
    cache: dict = {}

    def get(kind: str, **kw):
        key = (kind, tuple((k, tuple(v) if isinstance(v, list) else v)
                           for k, v in sorted(kw.items())))
        if key not in cache:
            wav, _, _ = fixture_cache(kind)
            tag = f"{kind}_{abs(hash(key)) % 10**8}"
            res = simulate(wav, out_path=str(workdir / f"{tag}.wav"),
                           seed=SEED_MACHINE, mix_human_db=-120.0, **kw)
            machine, sr = sf.read(res.out_path, dtype="float32",
                                  always_2d=True)
            assert sr == SR
            lines = [json.loads(ln) for ln in
                     open(res.decisions_path, encoding="utf-8")]
            cache[key] = (res, lines, machine.mean(axis=1))
        return cache[key]

    return get


# ------------------------------------------------------------------ helpers

def _plans(lines):
    return [ln for ln in lines if "state" in ln]


def _events(lines):
    return [ln for ln in lines if "event" in ln]


def _first(lines, state):
    for i, ln in enumerate(_plans(lines)):
        if ln["state"] == state:
            return i, ln
    return None, None


def _rms_db(x: np.ndarray) -> float:
    p = float(np.mean(np.asarray(x, np.float64) ** 2))
    return 10.0 * math.log10(max(p, 1e-18))


def _centroid_hz(x: np.ndarray) -> float:
    mag = np.abs(np.fft.rfft(np.asarray(x, np.float64)))
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    return float((mag * freqs).sum() / max(mag.sum(), 1e-12))


def _low_frac(x: np.ndarray, fcut_hz: float) -> float:
    """Fraction of spectral power below ``fcut_hz``."""
    mag2 = np.abs(np.fft.rfft(np.asarray(x, np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    return float(mag2[freqs < fcut_hz].sum() / max(mag2.sum(), 1e-12))


def _bar_dur(line) -> float:
    return 4.0 * 60.0 / float(line["bpm"])


def _lock_t(lines) -> float:
    return next(e["t_s"] for e in _events(lines) if e["event"] == "lock")


def _truth_bpm(truth, t: float) -> float:
    ts = [p[0] for p in truth["bpm_curve"]]
    bs = [p[1] for p in truth["bpm_curve"]]
    return float(np.interp(t, ts, bs))


# ---------------------------------------------------------------- 1. steady

def test_steady_enters_late_on_boundary_under_and_in_tempo(fixture_cache,
                                                           sim_cache):
    _, audio, truth = fixture_cache("steady")
    res, lines, machine = sim_cache("steady")

    # enters at a bar >= min_listen_bars, ON an entry_boundary multiple
    assert res.entered_bar is not None
    assert res.entered_bar >= _P.min_listen_bars
    assert res.entered_bar % _P.entry_boundary_bars == 0

    # in state "entering"
    _, entry = _first(lines, "entering")
    assert entry is not None and entry["bar"] == res.entered_bar

    # first-bar machine RMS >= 6 dB below the human bar RMS
    i0 = int(entry["t_s"] * SR)
    i1 = i0 + int(_bar_dur(entry) * SR)
    assert _rms_db(machine[i0:i1]) <= _rms_db(audio[i0:i1]) - 6.0

    # bpm within +/-3 of truth after lock
    lock_t = _lock_t(lines)
    post = [(t, b) for t, b in res.bpm_trace if t >= lock_t]
    assert post
    assert all(abs(b - truth["bpm_base"]) <= 3.0 for _, b in post)


# ----------------------------------------------------------------- 2. drift

def test_drift_followed_without_half_double_jumps(fixture_cache, sim_cache):
    _, _, truth = fixture_cache("drift")
    res, lines, _ = sim_cache("drift")

    bars = truth["bar_times_s"]
    n = truth["bars"]
    ramp_end_bar = max(0, (n - 8) // 2) + 8            # ramp spans middle 8 bars
    bpm_final = truth["bpm_base"] * 1.06

    # within +/-4 bpm by 2 bars after the drift ends
    t_check = bars[ramp_end_bar + 2] if ramp_end_bar + 2 < n else bars[-1]
    settled = [(t, b) for t, b in res.bpm_trace if t >= t_check]
    assert settled
    assert all(abs(b - bpm_final) <= 4.0 for _, b in settled)

    # no spurious half/double: bpm never leaves [truth*0.9, truth*1.15]
    for t, b in res.bpm_trace:
        tb = _truth_bpm(truth, t)
        assert 0.9 * tb <= b <= 1.15 * tb, (t, b, tb)


# ----------------------------------------------------------------- 3. stops

def test_stops_thin_then_recover_never_tacet(fixture_cache, sim_cache):
    _, _, truth = fixture_cache("stops")
    _, lines, _ = sim_cache("stops")

    bar_s = 60.0 / truth["bpm_base"] * 4.0
    stop_start = truth["bar_times_s"][truth["silent_bars"][0]]
    resume = truth["bar_times_s"][truth["silent_bars"][-1] + 1]

    # thinned within silence_thin_bars+1 of the stop
    i_thin, thin = _first(lines, "thinned")
    assert thin is not None, "machine never thinned during the stop"
    assert thin["t_s"] <= stop_start + (_P.silence_thin_bars + 1) * bar_s + _SLACK_S

    # comping again within 2 bars of the resume
    later = [ln for ln in _plans(lines)[i_thin + 1:] if ln["state"] == "comping"]
    assert later, "machine never comped again after the stop"
    assert later[0]["t_s"] <= resume + 2.0 * bar_s + _SLACK_S

    # never advances to tacet
    assert all(ln["state"] != "tacet" for ln in _plans(lines))


# ------------------------------------------------------------- 4. long_stop

def test_long_stop_fades_to_tacet_then_reenters_under(fixture_cache,
                                                      sim_cache):
    _, _, truth = fixture_cache("long_stop")
    _, lines, _ = sim_cache("long_stop")

    plans = _plans(lines)
    i_fade, fade = _first(lines, "fading")
    i_tacet, tacet = _first(lines, "tacet")
    assert fade is not None, "never reached fading"
    assert tacet is not None, "never reached tacet"
    assert i_fade < i_tacet, "fading must precede tacet"

    # re-enters after the resume: only at a boundary, entering, under-gain again
    resume = truth["bar_times_s"][truth["silent_bars"][-1] + 1]
    reentries = [ln for ln in plans[i_tacet + 1:]
                 if ln["state"] == "entering" and ln["t_s"] > resume]
    assert reentries, "machine never re-entered after the long stop"
    re = reentries[0]
    assert re["bar"] % _P.entry_boundary_bars == 0
    assert abs(re["gain_db"] - _P.enter_under_db) < 1e-6

    # nothing active between tacet and the re-entry
    for ln in plans[i_tacet + 1:]:
        if ln is re:
            break
        assert ln["state"] not in _ACTIVE


# ----------------------------------------------------------- 5. busy_middle

def test_busy_middle_inverse_density_with_hysteresis(fixture_cache, sim_cache):
    _, _, truth = fixture_cache("busy_middle")
    _, lines, _ = sim_cache("busy_middle")

    bar_s = 60.0 / truth["bpm_base"] * 4.0
    busy_t0 = truth["bar_times_s"][truth["busy_bars"][0]]
    busy_t1 = truth["bar_times_s"][truth["busy_bars"][-1]] + bar_s

    comping = [ln for ln in _plans(lines) if ln["state"] == "comping"]
    assert comping

    def overlap(ln):
        t0, t1 = ln["t_s"] - _bar_dur(ln), ln["t_s"]
        return max(0.0, min(t1, busy_t1) - max(t0, busy_t0))

    # busy-informed: the summarized human bar lies (mostly) in the busy span;
    # calm: no overlap at all, with a 2-bar guard after it for the hysteresis
    busy_plans = [ln for ln in comping if overlap(ln) > 0.5 * _bar_dur(ln)]
    calm_plans = [ln for ln in comping
                  if ln["t_s"] - _bar_dur(ln) >= busy_t1 + 2.0 * bar_s
                  or ln["t_s"] <= busy_t0]
    assert busy_plans and calm_plans

    # inverse coupling: the MACHINE's density during the busy bars < its
    # density in calm bars.  (The log key is `machine_density` since v1.2 —
    # the human's density is a different quantity in different units and
    # appears only in the reason text.)
    assert max(ln["machine_density"] for ln in busy_plans) \
        < min(ln["machine_density"] for ln in calm_plans)

    # hysteresis: recovery delayed >= 1 bar after density falls.  The reason
    # strings are contract: the last "human busy" plan must be followed by at
    # least one still-sparse "recovering next bar" plan before a normal
    # comping plan appears.
    idx_busy = [i for i, ln in enumerate(comping) if "human busy" in ln["reason"]]
    idx_wait = [i for i, ln in enumerate(comping)
                if "recovering next bar" in ln["reason"]]
    idx_normal = [i for i, ln in enumerate(comping)
                  if ln["reason"].startswith("comping: ")]
    last_busy = max(idx_busy)
    recovered_after = [i for i in idx_normal if i > last_busy]
    assert recovered_after, "density never recovered after the busy stretch"
    waited = [i for i in idx_wait if last_busy < i < recovered_after[0]]
    assert waited, "recovery was not delayed by the one-bar hysteresis"


# ---------------------------------------------------------- 6. rubato_intro

def test_rubato_intro_machine_silent_until_lock(sim_cache):
    _, lines, machine = sim_cache("rubato_intro")

    lock_t = _lock_t(lines)
    pre = machine[: int(lock_t * SR)]
    assert pre.size and float(np.max(np.abs(pre))) < 1e-4  # digital silence

    for ln in _plans(lines):
        if ln["t_s"] < lock_t:
            assert ln["state"] in ("waiting", "listening")
    _, entry = _first(lines, "entering")
    assert entry is not None and entry["t_s"] >= lock_t


# --------------------------------------------------------- 7. high_register

def test_high_register_ceiling_respected(fixture_cache, sim_cache):
    _, audio_hi, _ = fixture_cache("high_register")
    _, lines_hi, machine_hi = sim_cache("high_register")
    _, lines_st, machine_st = sim_cache("steady")

    # plans carry a ceiling below the human's register_low - margin
    # (high_register strums are voiced from midi 76 up)
    active_hi = [ln for ln in _plans(lines_hi) if ln["state"] in _ACTIVE]
    assert active_hi
    for ln in active_hi:
        assert ln["register_ceiling_midi"] is not None
        assert ln["register_ceiling_midi"] <= 76.0 - _P.register_margin_st + 2.0

    # the ceiling tracks the human: steady's low-voiced human forces a lower
    # ceiling than the high-register human leaves
    def mean_ceiling(lines):
        cs = [ln["register_ceiling_midi"] for ln in _plans(lines)
              if ln["state"] in _ACTIVE and ln["register_ceiling_midi"] is not None]
        return float(np.mean(cs))

    assert mean_ceiling(lines_st) < mean_ceiling(lines_hi)

    # rendered audio respects it: the human's register floor is C5 (523 Hz)
    # and essentially all machine power stays below it, while the human has
    # essentially none there — the machine voices under the human
    _, entry = _first(lines_hi, "entering")
    seg = slice(int(entry["t_s"] * SR), None)
    assert _low_frac(machine_hi[seg], 523.0) > 0.9
    assert _low_frac(audio_hi[seg], 523.0) < 0.05

    # and vs the steady fixture: steady's lower ceiling pushes the machine's
    # tonal weight further down-spectrum than the high-register run's
    _, entry_st = _first(lines_st, "entering")
    seg_st = slice(int(entry_st["t_s"] * SR), None)
    assert _low_frac(machine_st[seg_st], 80.0) > _low_frac(machine_hi[seg], 80.0)


# ---------------------------------------------------------- 8. determinism

def test_determinism_byte_identical(workdir, fixture_cache):
    wav, _, _ = fixture_cache("steady")
    outs = []
    for i in (1, 2):
        res = simulate(wav, out_path=str(workdir / f"det{i}.wav"),
                       seed=SEED_MACHINE)
        outs.append((pathlib.Path(res.out_path).read_bytes(),
                     pathlib.Path(res.decisions_path).read_bytes()))
    assert outs[0][0] == outs[1][0], "duet WAVs differ between runs"
    assert outs[0][1] == outs[1][1], "decision logs differ between runs"


# ------------------------------------------------------------------ 9. taps

def test_taps_count_in_locks_immediately(fixture_cache, sim_cache):
    _, _, truth = fixture_cache("steady")
    taps = [t for t, _ in truth["bpm_curve"][1:5]]     # 4 taps ending on a downbeat
    res, lines, _ = sim_cache("steady", taps_s=taps)

    assert res.entered_bar is not None
    assert res.entered_bar <= _P.min_listen_bars + _P.entry_boundary_bars

    # phase error at the first machine downbeat < 30 ms vs the truth grid
    _, entry = _first(lines, "entering")
    err = min(abs(entry["t_s"] - bt) for bt in truth["bar_times_s"])
    assert err < 0.030, f"first machine downbeat {err * 1000:.1f} ms off grid"


# ----------------------------------------------------------- 10. listen-only

def test_listen_only_confidence_rises_and_key_correct(fixture_cache, capsys):
    from plume.cli import main as cli_main

    wav, _, truth = fixture_cache("steady")
    rc = cli_main(["listen", wav, "--json"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)

    lock_t = next(e["t_s"] for e in doc["events"] if e["event"] == "lock")
    pre = [s["conf"] for s in doc["states"] if s["t_s"] <= lock_t + 0.5]
    assert len(pre) >= 3
    # monotonically increasing to lock (tiny numerical dips tolerated)
    assert all(b >= a - 0.02 for a, b in zip(pre, pre[1:]))
    assert pre[-1] > pre[0]

    final_key = doc["states"][-1]["key"]
    assert final_key is not None
    tonic = parse_key(final_key).tonic % 12
    truth_key = parse_key(truth["key"])
    rel = (truth_key.tonic + (3 if truth_key.mode == "minor" else 9)) % 12
    assert tonic in {truth_key.tonic % 12, rel}, final_key
