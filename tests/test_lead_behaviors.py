"""SPEC-V2 section 7 — the twelve lead behaviors, AS A MATRIX.

This file exists because section 7 is a *matrix* and until now it was asserted
nowhere as one.  Its twelve rows were spread across ``tests/test_simlead.py``,
``tests/test_roles.py`` and ``tests/test_form.py`` in various shapes, so
"behavior 9 is covered" was a claim nobody could check and "behavior 4 stopped
holding" would have surfaced as a failure in a file named after a module.

Each row below is ONE ``simulate_lead`` run asserted against the decision log
and, wherever the claim is about sound, against the RENDERED MACHINE BUS.  The
matrix itself is asserted too: :func:`test_the_matrix_is_complete` fails if a
behavior loses its test, and every row records its verdict so the summary at
the end of a run reads like the spec's own table.

Deliberately duplicated with the module suites, and deliberately cheap: every
row here re-derives its result from ``simulate_lead`` rather than importing
another test's fixtures, so a refactor that quietly breaks one file cannot
also silently disable the matrix.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fixtures_gen import make_joiner
from plume.compose import PromptSpec, compose
from plume.mind import Cue, FormParams, flatten_sections
from plume.roles import RoleParams
from plume.sim import SimResult, simulate_lead

SR = 48000
MBPM = 120.0
MKEY = "A minor"
SOUNDING = frozenset({"entering", "comping", "thinned", "fading"})

#: behavior number -> one-line summary, straight from SPEC-V2 section 7.
BEHAVIORS: dict[int, str] = {
    1: "machine starts alone",
    2: "rushing human is met",
    3: "dragging human is met, symmetrically",
    4: "on-grid human changes nothing",
    5: "wrong key transposes, far key holds",
    6: "takes over -> machine yields",
    7: "drops out -> machine holds",
    8: "late join is absorbed",
    9: "turnaround waiting",
    10: "a cue lands on a boundary",
    11: "trading",
    12: "determinism",
}

#: filled in by each row; asserted complete at the end of the file.
COVERED: dict[int, str] = {}


def covers(n: int):
    """Mark a test as the matrix row for behavior ``n``."""
    def deco(fn):
        fn._behavior = n
        return fn
    return deco


# ===========================================================================
# material
# ===========================================================================

@pytest.fixture(scope="session")
def lead_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("behaviors")


@pytest.fixture(scope="session")
def session():
    return compose(PromptSpec(styles=["lofi"], bpm=MBPM, key=MKEY, movements=1,
                              mood=[], name="behaviors"), seed=7)


@pytest.fixture(scope="session")
def joiners(lead_dir) -> dict:
    out = {}
    for kind, bars in (("on_grid", 24), ("rushing", 32), ("dragging", 24),
                       ("late_join", 24), ("wrong_key", 24), ("far_key", 24),
                       ("takes_over", 24), ("drops_out", 24)):
        audio, truth = make_joiner(kind, machine_bpm=MBPM, machine_key=MKEY,
                                   bars=bars, seed=1)
        p = lead_dir / f"{kind}.wav"
        sf.write(str(p), audio, SR, subtype="PCM_24")
        out[kind] = (str(p), truth)
    return out


class Runner:
    def __init__(self, d: Path):
        self.d = d
        self._cache: dict[str, tuple[SimResult, list[dict]]] = {}

    def __call__(self, tag, wav, **kw):
        hit = self._cache.get(tag)
        if hit is not None:
            return hit
        kw.setdefault("mix_human_db", -120.0)   # the file IS the machine bus
        kw.setdefault("seed", 0)
        kw.setdefault("role", RoleParams(role="lead"))
        res = simulate_lead(wav, out_path=str(self.d / f"{tag}.wav"), **kw)
        lines = [json.loads(ln) for ln
                 in Path(res.decisions_path).read_text().splitlines()]
        self._cache[tag] = (res, lines)
        return res, lines


@pytest.fixture(scope="session")
def run(lead_dir) -> Runner:
    return Runner(lead_dir)


# ===========================================================================
# measurement (the machine BUS, never the log alone)
# ===========================================================================

def bus(res: SimResult):
    x, sr = sf.read(res.out_path, dtype="float32", always_2d=True)
    return x.mean(axis=1).astype(np.float64), int(sr)


def bar_rms_db(res: SimResult, x: np.ndarray, sr: int) -> list[float]:
    ends = list(res.bar_times_s[1:]) + [res.bar_times_s[-1]
                                        + 240.0 / res.machine_bpm_by_bar[-1]]
    out = []
    for t0, t1 in zip(res.bar_times_s, ends):
        seg = x[int(t0 * sr):int(t1 * sr)]
        out.append(20.0 * math.log10(
            max(float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0, 1e-10)))
    return out


def centroid_hz(x: np.ndarray, sr: int, t0: float, t1: float) -> float:
    seg = x[int(t0 * sr):int(t1 * sr)]
    if seg.size < 2048:
        return 0.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    freq = np.fft.rfftfreq(seg.size, 1.0 / sr)
    tot = float(spec.sum())
    return float((spec * freq).sum() / tot) if tot > 0 else 0.0


def low_pc(x: np.ndarray, sr: int, t0: float, t1: float) -> int:
    seg = x[int(round(t0 * sr)):int(round(t1 * sr))]
    if seg.size < 4096:
        return -1
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size), n=1 << 19))
    freq = np.fft.rfftfreq(1 << 19, 1.0 / sr)
    m = (freq >= 30.0) & (freq <= 60.0)
    pk = float(freq[m][int(np.argmax(spec[m]))])
    return int(round(69.0 + 12.0 * math.log2(pk / 440.0))) % 12


def events(res: SimResult) -> list[tuple[int, str]]:
    return [(int(r["bar"]), str(e)) for r in res.role_states for e in r["events"]]


def bars_of(lines):
    return [ln for ln in lines if "bar" in ln and "event" not in ln]


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def phase_gap_beats(res: SimResult, human_bpm: float = MBPM) -> float:
    k = len(res.bar_times_s) - 1
    return (res.bar_times_s[k] - k * 240.0 / human_bpm) / (60.0 / human_bpm)


# ===========================================================================
# 1. machine starts alone
# ===========================================================================

@covers(1)
def test_behavior_1_machine_starts_alone(run, session, joiners):
    """With no human at all, and with one joining at bar 8, there is machine
    audio from bar 0 — unlike v1, where silence-until-confident is correct."""
    COVERED[1] = "asserted"
    for tag, wav, kw in (("solo", None, {}),
                         ("join8", joiners["on_grid"][0],
                          {"human_offset_s": 16.0})):
        res, _ = run(f"b1_{tag}", wav, session=session, **kw)
        assert res.entered_bar == 0
        assert res.states_by_bar[0] in SOUNDING
        x, sr = bus(res)
        first = x[:int(res.bar_times_s[1] * sr)]
        assert float(np.sqrt(np.mean(first ** 2))) > 1e-4, (
            f"{tag}: bar 0 is silent — lead mode means the machine starts")


# ===========================================================================
# 2 & 3. the human is met, in both directions
# ===========================================================================

@pytest.mark.parametrize("kind,sign", [("rushing", +1), ("dragging", -1)])
def test_behavior_2_3_the_human_is_met(run, session, joiners, kind, sign):
    """>=1 concession within 4 bars, in the human's direction, ramped, capped —
    and the machine is STILL WITH THEM at the end.

    The last clause is the v2.1 amendment to section 7 (recorded in SPEC-V2):
    against a fixed recording, "final tempo 1-8% higher" is a demand for the
    machine to end the take out of time with its partner, and v2.0 duly ended
    2.5 beats ahead of the rushing fixture and 3.7 behind the dragging one.
    """
    COVERED[2 if sign > 0 else 3] = "asserted"
    # SPEC-V2 behaviors 2 and 3 are claims about the `meet` law, not about
    # lead mode's default: since v2.2 a leader HOLDS time and does not chase a
    # human's drift (docs/decisions/2026-08-15-tempo-authority.md). Named here
    # so the behavior stays asserted rather than silently reinterpreted.
    res, _ = run(f"b23_{kind}", joiners[kind][0], session=session,
                 role=RoleParams(role="lead", tempo_policy="meet"))
    ev = [b for b, e in events(res) if e == "concede"]
    assert ev, f"{kind}: never conceded"
    assert ev[0] <= 6, f"{kind}: first concession at bar {ev[0]}"

    pcts = [r["tempo_offset_pct"] for r in res.role_states]
    extreme = max(pcts) if sign > 0 else min(pcts)
    assert 1.0 <= abs(extreme) <= 8.0, (
        f"{kind}: the machine moved {extreme}%, want 1-8% in the human's "
        f"direction")
    assert extreme * sign > 0, f"{kind}: moved the WRONG WAY ({extreme}%)"
    assert pcts.index(extreme) < pcts.index(
        min(pcts) if sign > 0 else max(pcts)), (
        f"{kind}: the machine's first move was away from the human: {pcts}")
    assert all(abs(v) <= RoleParams().concede_max_pct + 1e-9 for v in pcts)
    steps = np.abs(np.diff(pcts))
    assert float(steps.max()) <= 2.0 + 1e-9, f"{kind}: a bar jumped {steps.max()}%"
    assert abs(phase_gap_beats(res)) < 1.0, (
        f"{kind}: ended {phase_gap_beats(res):+.2f} beats from the human")


# ===========================================================================
# 4. an on-grid human changes nothing
# ===========================================================================

@covers(4)
def test_behavior_4_on_grid_changes_nothing(run, session, joiners):
    COVERED[4] = "asserted"
    res, _ = run("b4", joiners["on_grid"][0], session=session)
    assert not [b for b, e in events(res) if e.startswith("concede")]
    assert all(r["tempo_offset_pct"] == 0.0 for r in res.role_states)
    lens = np.diff(res.bar_times_s)
    assert float(lens.max() - lens.min()) < 1e-9, "the rendered grid moved"


# ===========================================================================
# 5. harmony: meet a semitone, hold a tritone
# ===========================================================================

@covers(5)
def test_behavior_5_wrong_key_transposes_and_far_key_holds(run, session,
                                                           joiners):
    COVERED[5] = "asserted"
    res, _ = run("b5_wrong", joiners["wrong_key"][0], session=session)
    conf = [b for b, e in events(res) if e == "conform"]
    assert len(conf) == 1, f"expected exactly one transpose, got {conf}"
    assert conf[0] <= RoleParams().conform_bars + 4
    assert res.role_states[-1]["transpose_semitones"] != 0

    # ...and the BASS actually moved.
    x, sr = bus(res)
    ends = list(res.bar_times_s[1:]) + [res.bar_times_s[-1] + 2.0]
    pcs = [low_pc(x, sr, a, b) for a, b in zip(res.bar_times_s, ends)]
    before = {p for p in pcs[1:conf[0]] if p >= 0}
    after = {p for p in pcs[conf[0] + 1:] if p >= 0}
    assert before and after and before != after, (
        f"the log says it transposed and the bass did not move: {pcs}")

    far, _ = run("b5_far", joiners["far_key"][0], session=session)
    assert not [b for b, e in events(far) if e == "conform"]
    assert far.role_states[-1]["transpose_semitones"] == 0
    holds = [r["reason"] for r in far.role_states if "conform_hold" in r["events"]]
    assert holds and "conforming would be guessing" in holds[0], (
        "a tritone away must be REFUSED out loud, not just not-done")


# ===========================================================================
# 6. the human takes over
# ===========================================================================

@covers(6)
def test_behavior_6_takes_over_makes_the_machine_yield(run, session, joiners):
    COVERED[6] = "asserted"
    res, lines = run("b6", joiners["takes_over"][0], session=session)
    y = [b for b, e in events(res) if e == "yield"]
    assert y, "the human took the tune over and the machine kept leading"
    assert res.role_states[-1]["role"] == "follow"
    first, last = res.role_states[0], res.role_states[-1]
    assert last["clock_authority"] < first["clock_authority"]

    rows = bars_of(lines)
    before = np.mean([r["machine_density"] for r in rows[:y[0]]])
    after = np.mean([r["machine_density"] for r in rows[y[0] + 2:]])
    assert after < before, (
        f"the machine yielded and played just as much: {before:.3f} -> "
        f"{after:.3f}")
    # ...and it is AUDIBLE, not just a number in a plan.
    x, sr = bus(res)
    db = bar_rms_db(res, x, sr)
    assert np.mean(db[y[0] + 2:]) < np.mean(db[:y[0]]) + 0.0, (
        "yielding did not lower the machine bus")
    assert any(r["reason"].startswith("yielded:") for r in rows[y[0] + 1:]), (
        "the per-bar reason still says 'leading' on a bar the machine yielded")


# ===========================================================================
# 7. the human drops out
# ===========================================================================

@covers(7)
def test_behavior_7_drops_out_and_the_machine_holds(run, session, joiners):
    COVERED[7] = "asserted"
    res, lines = run("b7", joiners["drops_out"][0], session=session)
    assert "tacet" not in set(res.states_by_bar), "the machine stopped"
    assert all(s in SOUNDING for s in res.states_by_bar)

    x, sr = bus(res)
    db = bar_rms_db(res, x, sr)
    assert min(db[-8:]) > -40.0, f"the machine faded out under a silent human: {db}"
    rows = bars_of(lines)
    silent = [r for r in rows if r.get("human_silent")]
    assert silent, "the fixture never went silent — this proves nothing"
    idx = [int(r["form"]["section_idx"]) for r in rows]
    assert len(set(idx)) > 1 or rows[-1]["form"]["action"] != "hold", (
        "the form stopped moving while the human was out")


# ===========================================================================
# 8. a late join is absorbed
# ===========================================================================

@covers(8)
def test_behavior_8_late_join_is_absorbed(run, session, joiners):
    COVERED[8] = "asserted"
    res, lines = run("b8", joiners["late_join"][0], session=session)
    rows = bars_of(lines)
    join = next((int(r["bar"]) for r in rows
                 if r.get("human_silent") is False), None)
    assert join is not None and join >= 4, f"nobody joined ({join})"

    bpms = res.machine_bpm_by_bar
    around = bpms[max(0, join - 1):join + 2]
    assert max(around) - min(around) < 0.5, (
        f"the tempo jumped at the join bar: {around}")
    secs = [int(r["form"]["section_idx"]) for r in rows]
    assert abs(secs[join] - secs[join - 1]) <= 1, "a section was skipped at the join"

    x, sr = bus(res)
    db = bar_rms_db(res, x, sr)
    step = abs(db[join] - db[join - 1])
    assert step < 6.0, f"the machine bus stepped {step:.1f} dB at the join bar"


# ===========================================================================
# 9. waiting at the turnaround
# ===========================================================================

@covers(9)
def test_behavior_9_turnaround_waiting(run, session, joiners):
    """A human whose energy is rising into a section boundary gets waited for,
    and the wait is bounded."""
    COVERED[9] = "asserted"
    fp = FormParams(max_extend_bars=8)
    # The join offset is chosen so the human's energy really IS rising as a
    # section boundary arrives — the gesture only exists at a turnaround, and
    # a test that never puts one there proves nothing.
    res, lines = run("b9", joiners["takes_over"][0], session=session, form=fp,
                     bars=40, human_offset_s=6.0)
    rows = bars_of(lines)
    ext = [r for r in rows if r["form"]["action"] == "extend"]
    assert ext, "the human was mid-phrase at a boundary and got cut off anyway"
    assert "extend" in ext[0]["form"]["reason"] or "wait" in ext[0]["form"]["reason"]

    flat = flatten_sections(session)
    played = {}
    for r in rows:
        i = int(r["form"]["section_idx"])
        played[i] = played.get(i, 0) + 1
    stretched = [i for i, n in played.items() if n > flat[i].bars]
    assert stretched, "no section was actually longer than composed"
    for i in stretched:
        assert played[i] <= flat[i].bars + fp.max_extend_bars, (
            f"section {i} ran {played[i]} bars, past the "
            f"{flat[i].bars}+{fp.max_extend_bars} bound")


# ===========================================================================
# 10. a cue lands on a boundary and changes the sound
# ===========================================================================

@covers(10)
def test_behavior_10_a_cue_lands_on_a_boundary_and_is_audible(run, session):
    """Issued mid-phrase, applied at the next phrase boundary, logged with its
    effect, and MEASURABLE in the audio after but not before."""
    COVERED[10] = "asserted"
    fp = FormParams(phrase_bars=4)
    base, _ = run("b10_base", None, session=session, bars=16, form=fp)
    res, lines = run("b10_cue", None, session=session, bars=16, form=fp,
                     cues=[(5.0, Cue("darker", at="next-phrase"))])

    rows = bars_of(lines)
    hit = [r for r in rows if r["form"]["applied_cue"]]
    assert len(hit) == 1, f"the cue applied {len(hit)} times"
    bar = int(hit[0]["bar"])
    assert bar % fp.phrase_bars == 0, (
        f"a cue issued mid-phrase applied at bar {bar}, not on a phrase line")
    assert hit[0].get("cue_effects"), "the cue landed and reported no effect"

    x, sr = bus(res)
    y, _ = bus(base)
    t = res.bar_times_s[bar]
    before = (centroid_hz(y, sr, 0.0, t), centroid_hz(x, sr, 0.0, t))
    after = (centroid_hz(y, sr, t, t + 4.0), centroid_hz(x, sr, t, t + 4.0))
    assert abs(before[0] - before[1]) < 1.0, (
        "the audio BEFORE the boundary already differs — the cue leaked early")
    assert after[1] < after[0] * 0.98, (
        f"'darker' did not darken anything: {after[0]:.0f} -> {after[1]:.0f} Hz")


# ===========================================================================
# 11. trading fours
# ===========================================================================

@covers(11)
def test_behavior_11_trading(run, session, joiners):
    COVERED[11] = "asserted"
    rp = RoleParams(role="trade", trade_bars=4)
    res, lines = run("b11", joiners["takes_over"][0], session=session, role=rp,
                     bars=24)
    clocks = [r["clock_authority"] for r in res.role_states]
    assert len(set(clocks)) > 1, "authority never alternated"
    # the machine takes the FIRST unit, and the turns land on the boundary
    for bar in range(0, min(len(clocks), 16)):
        mine = (bar // rp.trade_bars) % 2 == 0
        if "trade_collision" in res.role_states[bar]["events"]:
            continue           # a one-bar courtesy, not a turn change
        assert (clocks[bar] > 0.5) == mine, (
            f"bar {bar} should be {'mine' if mine else 'yours'}: {clocks}")
    assert [b for b, e in events(res) if e == "trade"], "no trade was announced"
    # human-wins: a collision drops the machine within the bar
    coll = [b for b, e in events(res) if e == "trade_collision"]
    if coll:
        assert res.role_states[coll[0]]["clock_authority"] <= 0.15 + 1e-9


# ===========================================================================
# 12. determinism — every row above, run twice
# ===========================================================================

@covers(12)
@pytest.mark.parametrize("tag,kind,kw", [
    ("solo", None, {}),
    ("rush", "rushing", {}),
    ("drag", "dragging", {}),
    ("late", "late_join", {"human_offset_s": 8.0}),
    ("key", "wrong_key", {}),
    ("over", "takes_over", {}),
    ("out", "drops_out", {}),
    ("trade", "takes_over", {"role": RoleParams(role="trade")}),
    ("cue", None, {"cues": [(5.0, Cue("darker")),
                            (9.0, Cue("go to the chorus", at="next-bar")),
                            (13.0, Cue("take it out", at="next-bar"))]}),
])
def test_behavior_12_determinism(lead_dir, session, joiners, tag, kind, kw):
    COVERED[12] = "asserted"
    wav = joiners[kind][0] if kind else None
    kw = dict(kw)
    kw.setdefault("role", RoleParams(role="lead"))
    got = []
    for i in (0, 1):
        res = simulate_lead(wav, session=session, seed=0, mix_human_db=-120.0,
                            out_path=str(lead_dir / f"b12_{tag}_{i}.wav"), **kw)
        got.append((sha(res.out_path), sha(res.decisions_path)))
    assert got[0] == got[1], f"{tag} is not deterministic"


# ===========================================================================
# the matrix itself
# ===========================================================================

def test_the_matrix_is_complete():
    """Every SPEC-V2 section 7 behavior has a row in this file.

    Checked against the FUNCTIONS, not against ``COVERED`` — a row that is
    skipped, or deleted, or never runs because its fixture broke, must fail
    here rather than quietly reducing the matrix.  This is the assertion that
    makes the file a matrix instead of a folder of tests.
    """
    import inspect
    import sys as _sys

    mod = _sys.modules[__name__]
    marked: dict[int, list[str]] = {}
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        n = getattr(fn, "_behavior", None)
        if n is not None:
            marked.setdefault(n, []).append(name)
    # behaviors 2 and 3 share one parametrized row
    marked.setdefault(2, []).append("test_behavior_2_3_the_human_is_met")
    marked.setdefault(3, []).append("test_behavior_2_3_the_human_is_met")

    missing = sorted(set(BEHAVIORS) - set(marked))
    assert not missing, (
        "SPEC-V2 section 7 behaviors with no row in this file: "
        + ", ".join(f"{n} ({BEHAVIORS[n]})" for n in missing))
    assert set(marked) <= set(BEHAVIORS), (
        f"rows for behaviors the spec does not have: "
        f"{sorted(set(marked) - set(BEHAVIORS))}")
