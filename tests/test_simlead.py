"""``plume.sim.simulate_lead`` — the offline twin, inverted (SPEC-V2 section 4).

WHAT THIS FILE IS FOR.  v1 passed all ten of its synthetic-fixture behavior
tests and was then completely unusable on real music, because four independent
bugs hid behind fixtures that happened to be continuously-filled sustained
plucks.  So every claim here is asserted in at least one of two places that a
fixture cannot fake:

* **the RENDERED MACHINE BUS**, not the decision log.  "A concession changed
  the tempo" is asserted by measuring the spacing of the machine's own attacks
  in the output WAV; "a transpose moved the bass" is asserted by measuring the
  low-band spectral peak of each rendered bar and converting it to a pitch
  class.  A number in an NDJSON line is a claim; a bar that is 34 ms shorter
  is the thing itself.
* **REAL MATERIAL**, at more than one gain.  The human in the real-material
  cases is one of plume's own v0 renders (``compose`` + ``engine.render``),
  driven against a DIFFERENT composed session at a different tempo and in a
  different key — the actual v2 use case — at 0 dB and at -20 dB.

Self-containment: every input is synthesized in-test into ``tmp_path``.
Nothing reads ``renders/`` (those are gitignored, so a test that read one
would silently skip itself on a clean clone).

Determinism: material and sim runs are built once per session from fixed seeds
and memoized; no wall clock, no global rng.  The one timing assertion
(``test_v2_analysis_cost_on_the_live_path``) has three orders of magnitude of
headroom over its floor, so it cannot flake on a loaded machine — it exists to
catch an algorithmic regression, not to benchmark.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fixtures_gen import make_joiner
from plume import sim as sim_mod
from plume.compose import PromptSpec, compose
from plume.engine import render
from plume.listen.features import FeatureExtractor
from plume.mind import Cue, MindParams, flatten_sections
from plume.roles import RoleParams
from plume.sim import SimResult, simulate, simulate_lead
from plume.theory import parse_chord, parse_key

SR = 48000


def _MEET(**overrides) -> RoleParams:
    """A leader that concedes tempo — the ``meet`` law, named explicitly.

    Since v2.2 ``role="lead"`` defaults to ``tempo_policy="hold"``: the machine
    is the reference and does NOT chase a human's timing drift (see
    docs/decisions/2026-08-15-tempo-authority.md).  Every test below that
    asserts a concession is a test of ``meet``, not of lead mode's default, so
    it says so here rather than relying on a default that has since changed.
    """
    return RoleParams(role="lead", tempo_policy="meet", **overrides)
MBPM = 120.0                 # the machine's composed tempo
MKEY = "A minor"             # ...and its composed key
SOUNDING = frozenset({"entering", "comping", "thinned", "fading"})


# ===========================================================================
# material + memoized runs
# ===========================================================================

@pytest.fixture(scope="session")
def lead_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("simlead")


@pytest.fixture(scope="session")
def machine_session():
    """The machine's composed arrangement: 4 x 8 bars, lofi, 120bpm, A minor."""
    return compose(PromptSpec(styles=["lofi"], bpm=MBPM, key=MKEY, movements=1,
                              mood=[], name="lead-demo"), seed=7)


@pytest.fixture(scope="session")
def joiners(lead_dir) -> dict:
    """Every SPEC-V2 section 6 joiner, written to disk with its truth dict."""
    out = {}
    specs = [("on_grid", 24), ("rushing", 32), ("dragging", 24),
             ("late_join", 24), ("wrong_key", 24), ("far_key", 24),
             ("takes_over", 24), ("drops_out", 24)]
    for kind, bars in specs:
        audio, truth = make_joiner(kind, machine_bpm=MBPM, machine_key=MKEY,
                                   bars=bars, seed=1)
        p = lead_dir / f"joiner_{kind}.wav"
        sf.write(str(p), audio, SR, subtype="PCM_24")
        out[kind] = (str(p), truth)
    return out


@pytest.fixture(scope="session")
def real_human(lead_dir) -> dict:
    """A REAL v0 render as the human: trap, 140bpm, D minor, at two gains.

    Deliberately at a different tempo AND a different key from the machine's
    session — that is the v2 case, not a click track that already agrees with
    us.  The -20 dB copy is the gain axis the v1 fixture family has none of.
    """
    src = lead_dir / "human_trap.wav"
    hs = compose(PromptSpec(styles=["trap"], bpm=140.0, key="D minor",
                            movements=1, mood=[], name="human"), seed=7)
    render(hs, out_path=str(src), movement=0, bars=(0, 16))
    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    out = {}
    for gain in (0.0, -20.0):
        p = lead_dir / f"human_trap_{int(gain)}db.wav"
        sf.write(str(p), x * (10.0 ** (gain / 20.0)), sr, subtype="PCM_24")
        out[gain] = str(p)
    return out


@pytest.fixture(scope="session")
def other_session():
    """A composed session that agrees with the real human about nothing."""
    return compose(PromptSpec(styles=["lofi"], bpm=96.0, key="F minor",
                              movements=1, mood=[], name="machine"), seed=3)


class Runner:
    """Memoized ``simulate_lead`` calls (each costs ~1-3 s of rendering)."""

    def __init__(self, d: Path):
        self.d = d
        self._cache: dict[str, tuple[SimResult, list[dict]]] = {}
        self._n = 0

    def __call__(self, tag: str, wav, **kw) -> tuple[SimResult, list[dict]]:
        hit = self._cache.get(tag)
        if hit is not None:
            return hit
        self._n += 1
        res = simulate_lead(wav, out_path=str(self.d / f"{tag}.wav"), **kw)
        lines = [json.loads(ln) for ln
                 in Path(res.decisions_path).read_text().splitlines()]
        self._cache[tag] = (res, lines)
        return res, lines


@pytest.fixture(scope="session")
def run(lead_dir) -> Runner:
    return Runner(lead_dir)


def bars_of(lines: list[dict]) -> list[dict]:
    """The per-bar plan lines, in order (events filtered out)."""
    return [ln for ln in lines if "bar" in ln and "event" not in ln]


def events_of(res: SimResult) -> list[tuple[int, str]]:
    return [(int(r["bar"]), str(e)) for r in res.role_states for e in r["events"]]


# ===========================================================================
# audio measurement — the machine bus, measured, never assumed
# ===========================================================================

def machine_bus(res: SimResult) -> tuple[np.ndarray, int]:
    """The rendered output as mono.

    Every run whose audio is measured here is made with
    ``mix_human_db=-120``, which puts the human 1e-6 down: the file IS the
    machine bus, and the analysis path is untouched by it (features / beat /
    harmony run on the isolated human BEFORE the mix, which is the whole point
    of SPEC-V2 section 6's fader experiment).
    """
    x, sr = sf.read(res.out_path, dtype="float32", always_2d=True)
    return x.mean(axis=1).astype(np.float64), int(sr)


def attack_times(x: np.ndarray, sr: int) -> np.ndarray:
    """Sub-hop attack times of plume's own detected onsets.

    Same two-stage measurement ``tests/test_fixtures_gen.py`` uses on the
    joiners: the real detector (hop 512, so ~11 ms late and quantized), then a
    1 ms-envelope walk back to the foot of the attack.  Reuses the refiner
    ``sim`` itself uses so the test cannot pass on a measurement the shipping
    code does not have.
    """
    fe = FeatureExtractor(sr)
    frames = fe.process(x.astype(np.float32)) + fe.flush()
    ref = sim_mod._AttackRefiner(x, sr)
    return np.array([ref.refine(f.t_s) for f in frames if f.onset],
                    dtype=np.float64)


def bar_aligned_attacks(res: SimResult, x: np.ndarray, sr: int) -> np.ndarray:
    """For each machine bar boundary, the nearest detected attack."""
    att = attack_times(x, sr)
    assert att.size >= len(res.bar_times_s), "machine bus has too few attacks"
    return np.array([att[int(np.argmin(np.abs(att - t)))]
                     for t in res.bar_times_s], dtype=np.float64)


def low_band_pc(x: np.ndarray, sr: int, t0: float, t1: float) -> int:
    """Pitch class of the strongest 30-60 Hz partial in ``[t0, t1)``.

    That band is where ``duet._low_octave_hz`` puts BOTH pitched things the
    machine plays down there — the bass root and the kick, which v0 tunes to
    the key's tonic — so a transposition has to move it or it did not happen.
    """
    a, b = int(round(t0 * sr)), int(round(t1 * sr))
    seg = x[a:b]
    if seg.size < 4096:
        return -1
    seg = seg * np.hanning(seg.size)
    n = 1 << 19
    spec = np.abs(np.fft.rfft(seg, n=n))
    freq = np.fft.rfftfreq(n, 1.0 / sr)
    m = (freq >= 30.0) & (freq <= 60.0)
    pk = float(freq[m][int(np.argmax(spec[m]))])
    return int(round(69.0 + 12.0 * math.log2(pk / 440.0))) % 12


def bar_pcs(res: SimResult, x: np.ndarray, sr: int) -> list[int]:
    ends = list(res.bar_times_s[1:]) + [res.bar_times_s[-1]
                                        + 240.0 / res.machine_bpm_by_bar[-1]]
    return [low_band_pc(x, sr, t0, t1)
            for t0, t1 in zip(res.bar_times_s, ends)]


def rms_db(x: np.ndarray) -> float:
    return 20.0 * math.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-10))


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ===========================================================================
# 1. the contract: signature, defaults, and v1 left alone
# ===========================================================================

def test_needs_exactly_one_of_session_or_prompt(lead_dir, machine_session):
    with pytest.raises(ValueError, match="exactly one"):
        simulate_lead(None, out_path=str(lead_dir / "x.wav"))
    with pytest.raises(ValueError, match="exactly one"):
        simulate_lead(None, session=machine_session, prompt="trap at 120",
                      out_path=str(lead_dir / "x.wav"))


def test_rejects_a_negative_or_non_finite_human_offset(lead_dir, machine_session):
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="human_offset_s"):
            simulate_lead(None, session=machine_session,
                          out_path=str(lead_dir / "x.wav"), human_offset_s=bad)


def test_bars_must_be_positive_when_it_is_given(lead_dir, machine_session):
    for bad in (0, -8):
        with pytest.raises(ValueError, match="bars must be positive"):
            simulate_lead(None, session=machine_session, bars=bad,
                          out_path=str(lead_dir / "x.wav"))


def test_bars_defaults_to_the_composed_length_and_covers_the_human(
        run, lead_dir, machine_session, joiners):
    composed = sum(s.bars for m in machine_session.movements
                   for s in m.sections)
    solo, _ = run("solo", None, session=machine_session, seed=0,
                  mix_human_db=-120.0)
    assert len(solo.bar_times_s) == composed

    # a human who outlasts the arrangement: the machine holds the floor for at
    # least as long as they play, rather than stopping under them.
    wav, _ = joiners["on_grid"]
    long_join = simulate_lead(wav, session=machine_session, seed=0,
                              human_offset_s=40.0,
                              out_path=str(lead_dir / "long_join.wav"))
    x, sr = sf.read(wav, always_2d=True)
    need = 40.0 + x.shape[0] / sr
    assert long_join.bar_times_s[-1] >= need - 240.0 / MBPM
    assert len(long_join.bar_times_s) > composed

    short = simulate_lead(None, session=machine_session, seed=0, bars=5,
                          out_path=str(lead_dir / "short.wav"))
    assert len(short.bar_times_s) == 5


def test_the_machine_plays_its_composed_chord_cycle(run, machine_session):
    """Lead mode means the machine plays ITS arrangement — the section's
    ``chords`` list, cycled across the section's bars and restarted when the
    cursor enters a new section."""
    res, lines = run("solo", None, session=machine_session, seed=0,
                     mix_human_db=-120.0)
    rows = bars_of(lines)
    flat = [s for m in machine_session.movements for s in m.sections
            if s.bars > 0]
    pos = {}
    for row, fp in zip(rows, res.form_plans):
        idx = fp["section_idx"]
        pos[idx] = 0 if idx not in pos or fp["action"] in ("advance", "jump") \
            else pos[idx] + 1
        want = flat[idx].chords[pos[idx] % len(flat[idx].chords)]
        assert row["chord"] == want, (
            f"bar {row['bar']}: played {row['chord']}, composed {want}")
    # ...and the rendered low end follows it
    x, sr = machine_bus(res)
    pcs = bar_pcs(res, x, sr)
    for k, row in enumerate(rows):
        allowed = {parse_key(row["key"]).tonic, parse_chord(row["chord"]).root}
        assert pcs[k] in allowed


def test_runs_from_a_bare_prompt_with_no_human(lead_dir):
    """SPEC-V2 section 8: ``simlead`` works from a prompt, machine solo."""
    res = simulate_lead(None, prompt="chill lofi, 2 movements",
                        out_path=str(lead_dir / "prompt_solo.wav"))
    assert res.manifest["mode"] == "lead"
    assert len(res.states_by_bar) >= 8
    assert set(res.states_by_bar) <= SOUNDING


def test_role_defaults_to_lead_not_to_roleparams_default(run, machine_session):
    """v2 amendment: ``RoleParams()`` starts in ``follow`` (clock 0.15), which
    contradicts "the machine's clock is authoritative from t=0"."""
    res, _ = run("defaults", None, session=machine_session, seed=0)
    assert res.role_states[0]["role"] == "lead"
    assert res.role_states[0]["clock_authority"] == pytest.approx(0.9)
    assert res.manifest["role_params"]["role"] == "lead"


def test_simulate_is_untouched_and_leaves_the_v2_fields_empty(lead_dir, joiners):
    """The v1 twin must keep behaving exactly as v1, including its SimResult."""
    wav, _ = joiners["on_grid"]
    res = simulate(wav, style="lofi", out_path=str(lead_dir / "v1.wav"), seed=0)
    assert res.role_states == [] and res.form_plans == []
    assert res.bar_times_s == [] and res.machine_bpm_by_bar == []
    assert "mode" not in res.manifest          # lead-only manifest key


def test_ndjson_and_simresult_carry_the_v2_contract(run, machine_session, joiners):
    wav, _ = joiners["on_grid"]
    res, lines = run("contract", wav, session=machine_session, seed=0)
    rows = bars_of(lines)
    assert len(rows) == len(res.role_states) == len(res.form_plans)
    assert len(rows) == len(res.bar_times_s) == len(res.machine_bpm_by_bar)
    for row, rs, fp in zip(rows, res.role_states, res.form_plans):
        # SPEC-V2 section 4: the four named keys plus the form record.
        for key in ("role", "clock_authority", "tempo_offset_pct",
                    "transpose_semitones", "form"):
            assert key in row, f"NDJSON line is missing {key!r}"
        assert row["role"] == rs["role"]
        assert row["clock_authority"] == pytest.approx(rs["clock_authority"])
        assert row["tempo_offset_pct"] == pytest.approx(rs["tempo_offset_pct"])
        assert row["transpose_semitones"] == rs["transpose_semitones"]
        assert row["form"]["action"] == fp["action"]
        assert row["form"]["section_idx"] == fp["section_idx"]
        # ...and the human's own readings are still on the line, under their
        # own names (the v2 amendment: bpm/key/chord describe the MACHINE now).
        for key in ("human_bpm", "human_key", "human_silent", "human_offset_ms"):
            assert key in row
    assert res.manifest["composed_bpm"] == MBPM
    assert res.manifest["composed_key"] == MKEY


def test_ndjson_is_sorted_key_json_one_object_per_line(run, machine_session):
    _, lines = run("defaults", None, session=machine_session, seed=0)
    text = Path(
        run.d.joinpath("defaults.ndjson")).read_text().splitlines()
    assert text and all(json.loads(ln) is not None for ln in text)
    assert len(text) == len(lines)


# ===========================================================================
# 2. the machine starts (SPEC-V2 behavior 1) — and keeps going (behavior 7)
# ===========================================================================

def test_machine_plays_from_bar_zero_with_no_human_at_all(run, machine_session):
    """Behavior 1.  Under ``Mind.plan_bar`` this take is silent forever: the
    entry gate never opens without a human, and the silence ladder would go
    tacet at bar 10 if it did.  Lead mode is the inversion."""
    res, lines = run("solo", None, session=machine_session, seed=0,
                     mix_human_db=-120.0)
    assert res.entered_bar == 0
    assert set(res.states_by_bar) <= SOUNDING
    assert "tacet" not in res.states_by_bar
    assert res.manifest["input"] is None

    x, sr = machine_bus(res)
    bar0 = x[: int(res.bar_times_s[1] * sr)]
    assert rms_db(bar0) > -40.0, "bar 0 of a machine-solo take is silent"
    # ...and it is still playing at the end, not fading into nothing.
    last = x[int(res.bar_times_s[-1] * sr):int(res.duration_s * sr)]
    assert rms_db(last) > -40.0
    # every bar audible, not just the ends
    for k in range(len(res.bar_times_s) - 1):
        seg = x[int(res.bar_times_s[k] * sr):int(res.bar_times_s[k + 1] * sr)]
        assert rms_db(seg) > -45.0, f"machine bar {k} is silent"

    assert all(ln["state"] in SOUNDING for ln in bars_of(lines))


def test_machine_plays_from_bar_zero_when_the_human_joins_at_bar_8(
        run, machine_session, joiners):
    """Behavior 1, the other half: the human's WAV starts 16 s (8 bars) in."""
    wav, _ = joiners["on_grid"]
    res, _ = run("join8", wav, session=machine_session, seed=0,
                 human_offset_s=16.0, mix_human_db=-120.0)
    x, sr = machine_bus(res)
    for k in range(8):
        seg = x[int(res.bar_times_s[k] * sr):int(res.bar_times_s[k + 1] * sr)]
        assert rms_db(seg) > -45.0, f"machine bar {k} (before the human) silent"
    assert res.entered_bar == 0


def test_drops_out_and_the_machine_holds_the_floor(run, machine_session, joiners):
    """Behavior 7.  The human plays 8 bars, stops for 12, and the machine must
    keep leading — no tacet, and the form keeps moving."""
    wav, truth = joiners["drops_out"]
    res, _ = run("drops_out", wav, session=machine_session, seed=0,
                 mix_human_db=-120.0)
    assert "tacet" not in res.states_by_bar
    x, sr = machine_bus(res)
    for k in truth["silent_bars"]:
        if k + 1 >= len(res.bar_times_s):
            continue
        seg = x[int(res.bar_times_s[k] * sr):int(res.bar_times_s[k + 1] * sr)]
        assert rms_db(seg) > -45.0, f"machine went quiet in the human's bar {k}"
    # the form advanced through the silence rather than freezing
    assert len({f["section_idx"] for f in res.form_plans}) >= 2
    assert "advance" in {f["action"] for f in res.form_plans}


def test_the_inversion_is_real_v1_winds_down_where_lead_mode_holds(
        run, lead_dir, machine_session, joiners):
    """The measured contrast behind ``_LeadPlanner``'s docstring.

    Same WAV, same style, same seed.  v1's sideman is RIGHT to wind down when
    the human drops out — that is what a sideman does — and it does: the plan
    walks entering -> thinned -> fading and the machine stops making decisions
    when the take runs out under it.  Lead mode must do the opposite on the
    identical input.  Without this test, behavior 7 is only asserting that a
    planner that cannot emit ``tacet`` did not emit ``tacet``.
    """
    wav, _ = joiners["drops_out"]
    v1 = simulate(wav, style="lofi", out_path=str(lead_dir / "v1_drops.wav"),
                  seed=0)
    assert {"thinned", "fading"} & set(v1.states_by_bar), (
        "v1 did not wind down — the contrast this test rests on has changed")
    assert v1.states_by_bar[-1] in ("thinned", "fading", "tacet")

    v2, _ = run("drops_out", wav, session=machine_session, seed=0,
                mix_human_db=-120.0)
    assert v2.states_by_bar[-1] == "comping"
    assert len(v2.states_by_bar) > len(v1.states_by_bar)


# ===========================================================================
# 3. the human joins (SPEC-V2 behavior 8) — the grid must not move
# ===========================================================================

def test_a_human_joining_at_bar_8_does_not_disturb_the_grid(
        run, machine_session, joiners):
    """Behavior 8.  The machine-solo take is the control: the same session with
    nobody there.  A human arriving on the grid, in key, must leave the bar
    grid and the form byte-for-byte where they were."""
    wav, _ = joiners["on_grid"]
    solo, _ = run("solo", None, session=machine_session, seed=0,
                  mix_human_db=-120.0)
    join, _ = run("join8", wav, session=machine_session, seed=0,
                  human_offset_s=16.0, mix_human_db=-120.0)

    assert len(join.bar_times_s) == len(solo.bar_times_s)
    delta = np.abs(np.array(join.bar_times_s) - np.array(solo.bar_times_s))
    assert float(delta.max()) == 0.0, "the human moved the machine's grid"
    assert join.machine_bpm_by_bar == solo.machine_bpm_by_bar
    assert ([f["action"] for f in join.form_plans]
            == [f["action"] for f in solo.form_plans]), "the human skipped a section"
    assert [e for _, e in events_of(join) if e.startswith("concede")] == []
    # bar lengths are constant across the join bar in particular
    lens = np.diff(join.bar_times_s)
    assert float(lens.max() - lens.min()) < 1e-9


def test_late_join_fixture_is_absorbed_without_a_tempo_jump(
        run, machine_session, joiners):
    wav, truth = joiners["late_join"]
    res, _ = run("late_join", wav, session=machine_session, seed=0)
    jb = int(truth["join_bar"])
    # `tempo_hold` is emitted every bar the offset is measured and deliberately
    # not acted on (v2.2 default). It is a reading, not a tempo event — this
    # test's claim is that nothing MOVED the tempo, which hold satisfies.
    assert [e for _, e in events_of(res) if e != "tempo_hold"] == []
    assert res.role_states[-1]["tempo_offset_pct"] == pytest.approx(0.0)
    assert res.form_plans[jb]["action"] in ("hold", "advance")


# ===========================================================================
# 4. tempo concession (behaviors 2, 3, 4) — measured in the rendered audio
# ===========================================================================

def test_on_grid_human_changes_nothing(run, machine_session, joiners):
    """Behavior 4, the guard against a jittery concede rule."""
    wav, _ = joiners["on_grid"]
    res, _ = run("on_grid", wav, session=machine_session, seed=0,
                 mix_human_db=-120.0)
    assert [e for _, e in events_of(res) if e.startswith("concede")] == []
    assert all(p == 0.0 for p in
               (r["tempo_offset_pct"] for r in res.role_states))
    lens = np.diff(run("on_grid", wav, session=machine_session, seed=0,
                       mix_human_db=-120.0)[0].bar_times_s)
    assert float(lens.max() - lens.min()) < 1e-9


def _phase_gap_beats(res, human_bpm: float) -> float:
    """How far the machine's LAST bar line has drifted from the human's, in beats.

    The human here is a fixture at a fixed tempo, so their bar ``k`` is at
    ``k * 240/bpm`` exactly.  This is the number SPEC-V2's concede rule is
    really about and the one v2.0 got catastrophically wrong.
    """
    bt = res.bar_times_s
    k = len(bt) - 1
    return (bt[k] - k * 240.0 / human_bpm) / (60.0 / human_bpm)


def test_rushing_human_is_met_and_the_rendered_bars_really_get_shorter(
        run, machine_session, joiners):
    """Behavior 2, asserted in the AUDIO.

    The decision log saying "conceding" is a claim.  What is measured here is
    the spacing of the machine's own attacks in the rendered bus: the first
    bars are 2.000 s apart and the later ones shorter, which is the tempo
    change, not a log line about one.

    **v2.1 AMENDMENT to SPEC-V2 section 7 behavior 2** (recorded in SPEC-V2).
    The spec asks for "final machine tempo higher than composed by 1-8%", and
    against THIS fixture that is a demand for the machine to end the take out
    of time with the human.  ``rushing`` is the same recording at the same
    tempo, displaced 30 ms; it cannot bend.  So a machine that permanently
    keeps the 1-8% it conceded is a machine walking away from its partner, and
    MEASURED, v2.0 did exactly that: it ended +3.26% with its downbeat 1.24 s —
    two and a half BEATS — ahead of the human's.  What behavior 2 is really
    claiming is (a) the machine met them, promptly, in the right direction, by
    an audible-but-musical amount, and (b) it was still with them at the end.
    Both are asserted here, and (b) is the one v2.0 could not pass.

    KNOWN LIMIT, asserted honestly rather than hidden: closing the phase the
    concession itself opened takes a counter-excursion, so against a partner
    who cannot bend the tempo curve is a bounded hunt rather than a settle —
    MEASURED here, +1.5% then -3.9%, inside the 8% cap, ramped, ending 0.85
    beats from the human where v2.0 ended 2.5 beats away.  The direction of the
    FIRST move is the part that is a claim about the human, and that is what is
    asserted.
    """
    wav, truth = joiners["rushing"]
    assert truth["mean_offset_ms"] > 20.0          # the fixture really rushes
    res, _ = run("rushing", wav, session=machine_session, seed=0,
                 mix_human_db=-120.0, role=_MEET())

    ev = [(b, e) for b, e in events_of(res) if e == "concede"]
    assert ev, "no concession against a human 30 ms in front for the whole take"
    assert ev[0][0] <= 6, f"first concession at bar {ev[0][0]}, too late"

    pcts = [r["tempo_offset_pct"] for r in res.role_states]
    peak = max(pcts)
    assert 1.0 <= peak <= 8.0, (
        f"the machine never came up to meet them (peak {peak}%, want 1-8%)")
    assert pcts.index(peak) < pcts.index(min(pcts)), (
        "the machine's FIRST move against a human in front of the beat was "
        f"downwards ({pcts})")
    assert all(abs(x) <= 8.0 for x in pcts)

    # ramped: SPEC-V2 section 7 bounds a single bar at 2%
    step = np.abs(np.diff(pcts))
    assert float(step.max()) <= 2.0 + 1e-9

    # ...and it is STILL WITH THEM at the end.  This is the phase-return term.
    gap = _phase_gap_beats(res, MBPM)
    assert abs(gap) < 1.0, (
        f"the machine ended {gap:+.2f} beats away from the human it spent the "
        f"whole take meeting")

    # --- the audio -------------------------------------------------------
    x, sr = machine_bus(res)
    aligned = bar_aligned_attacks(res, x, sr)
    off = np.abs(aligned - np.array(res.bar_times_s)) * 1000.0
    assert float(np.median(off)) < 20.0, (
        "the machine's attacks do not sit on the grid the log claims")
    spacing = np.diff(aligned)
    first = float(np.median(spacing[:4]))
    fastest = float(np.min(spacing))
    assert fastest < first, "bars never got shorter in the rendered audio"
    measured_pct = (first / fastest - 1.0) * 100.0
    assert measured_pct == pytest.approx(peak, abs=0.8), (
        f"audio says {measured_pct:.2f}% faster at its fastest, log says {peak:.2f}%")


def test_dragging_human_is_met_symmetrically(run, machine_session, joiners):
    """Behavior 3.  The sign matters: a lay-back must SLOW the machine.

    Symmetric with behavior 2 above, including the v2.1 amendment: the
    concession is measured at its extreme (a fixed recording cannot come back
    to meet the machine, so a permanent offset would just be the machine
    leaving) and the take must still END with the human.
    """
    wav, truth = joiners["dragging"]
    assert truth["mean_offset_ms"] < -20.0
    res, _ = run("dragging", wav, session=machine_session, seed=0,
                 mix_human_db=-120.0, role=_MEET())
    assert [b for b, e in events_of(res) if e == "concede"]
    pcts = [r["tempo_offset_pct"] for r in res.role_states]
    trough = min(pcts)
    assert -8.0 <= trough <= -1.0, (
        f"the machine never settled back to meet them (trough {trough}%)")
    assert pcts.index(trough) < pcts.index(max(pcts)), (
        f"the machine's FIRST move against a human behind the beat was "
        f"upwards ({pcts})")
    gap = _phase_gap_beats(res, MBPM)
    assert abs(gap) < 1.0, f"ended {gap:+.2f} beats away from the human"

    x, sr = machine_bus(res)
    aligned = bar_aligned_attacks(res, x, sr)
    spacing = np.diff(aligned)
    assert float(np.max(spacing)) > float(np.median(spacing[:4])), (
        "bars never got LONGER against a human laying back")


@pytest.mark.parametrize("kind,sign", [("rushing", +1), ("dragging", -1)])
def test_the_machine_ends_the_take_WITH_the_human(run, machine_session,
                                                  joiners, kind, sign):
    """THE v2.0 DEFECT, as its own regression test.

    A phase->period law with no phase-return term diverges by construction, and
    it did: MEASURED on these exact fixtures, v2.0 ended the ``rushing`` take
    2.5 beats ahead of the human and the ``dragging`` take 3.7 beats behind.
    Nothing in the decision log said so — the log showed a tidy concession and
    a settled tempo — which is why this is asserted against the bar grid the
    audio was actually rendered on.

    Non-vacuity: the same take with conceding switched off (``concede_ms``
    unreachable) must sit at exactly zero, so a machine that simply never moved
    could not pass this by accident.
    """
    wav, _ = joiners[kind]
    res, _ = run(f"{kind}_gap", wav, session=machine_session, seed=0,
                 mix_human_db=-120.0, role=_MEET())
    gap = _phase_gap_beats(res, MBPM)
    assert abs(gap) < 1.0, f"{kind}: ended {gap:+.2f} beats out"
    assert any(e == "concede" for _, e in events_of(res)), \
        f"{kind}: never conceded, so the gap proves nothing"

    inert, _ = run(f"{kind}_inert", wav, session=machine_session, seed=0,
                   mix_human_db=-120.0,
                   role=_MEET(concede_ms=10_000.0,
                              concede_drift_ms=10_000.0))
    assert not any(e == "concede" for _, e in events_of(inert))
    assert abs(_phase_gap_beats(inert, MBPM)) < 1e-6


def test_a_concession_never_exceeds_the_cap(run, machine_session, joiners):
    wav, _ = joiners["rushing"]
    res, _ = run("rushing_cap", wav, session=machine_session, seed=0,
                 role=_MEET(concede_max_pct=1.0, concede_ms=5.0))
    assert all(abs(r["tempo_offset_pct"]) <= 1.0 + 1e-9
               for r in res.role_states)
    bpms = np.array(res.machine_bpm_by_bar)
    assert float(bpms.max()) <= MBPM * 1.01 + 1e-6
    assert float(bpms.min()) >= MBPM * 0.99 - 1e-6


def test_total_duration_matches_the_integrated_tempo_curve(
        run, machine_session, joiners):
    """The bar grid is ACCUMULATED, not recomputed — no drift, no double count.

    A 32-bar take against a permanently-rushing human, with the machine's tempo
    moving underneath it.  Summing 240/bpm over the bars must land on the
    rendered end of the last bar to within a millisecond, or bar boundaries and
    rendered audio have quietly diverged.
    """
    wav, _ = joiners["rushing"]
    res, _ = run("rushing", wav, session=machine_session, seed=0,
                 mix_human_db=-120.0, role=_MEET())
    assert len(res.machine_bpm_by_bar) == 32
    assert len(set(res.machine_bpm_by_bar)) > 1, "the tempo never moved"

    integrated = float(np.sum(240.0 / np.array(res.machine_bpm_by_bar)))
    rendered_end = (res.bar_times_s[-1] + 240.0 / res.machine_bpm_by_bar[-1])
    assert rendered_end == pytest.approx(integrated, abs=0.002), (
        f"grid drifted {abs(rendered_end - integrated) * 1000:.2f} ms over 32 bars")

    # every boundary, not just the last: a compensating pair of errors would
    # pass the total and fail here.
    want = np.concatenate([[0.0], np.cumsum(240.0 / np.array(
        res.machine_bpm_by_bar))[:-1]])
    assert float(np.max(np.abs(want - np.array(res.bar_times_s)))) < 0.002

    # and the WAV is at least as long as the machine's last bar
    x, sr = machine_bus(res)
    assert x.size / sr >= rendered_end - 1e-3


# ===========================================================================
# 5. harmonic meeting (behavior 5) — measured in the rendered bass
# ===========================================================================

def test_wrong_key_transposes_the_rendered_bass_pitch_classes(
        run, machine_session, joiners):
    """Behavior 5.  A human a semitone up must move the machine's LOW END, not
    just an integer in the log."""
    wav, truth = joiners["wrong_key"]
    res, lines = run("wrong_key", wav, session=machine_session, seed=0,
                     mix_human_db=-120.0)

    conform = [b for b, e in events_of(res) if e == "conform"]
    assert len(conform) == 1, f"expected exactly one transpose, got {conform}"
    at = conform[0]
    assert at <= 4 + 2 + 2, f"conformed at bar {at}, later than conform_bars+2"
    assert res.manifest["transpose_semitones"] == 1
    assert parse_key(res.final_key).tonic == (parse_key(MKEY).tonic + 1) % 12
    assert parse_key(res.final_key).tonic == parse_key(truth["key"]).tonic

    rows = bars_of(lines)
    x, sr = machine_bus(res)
    pcs = bar_pcs(res, x, sr)

    # 1. every bar's rendered low end is one of the two pitch classes the
    #    machine is actually playing down there: the chord root, or the kick
    #    (v0 tunes it to the key's tonic).
    for k, row in enumerate(rows):
        allowed = {parse_key(row["key"]).tonic}
        if row["chord"]:
            allowed.add(parse_chord(row["chord"]).root)
        assert pcs[k] in allowed, (
            f"bar {k}: rendered low end pc {pcs[k]}, machine playing {allowed}")

    # 2. the same position in the chord cycle, before and after the transpose,
    #    is exactly one semitone apart in the AUDIO.
    tonic_pre = [k for k in range(at) if rows[k]["chord"] == "Am7"]
    tonic_post = [k for k in range(at, len(rows)) if rows[k]["chord"] == "Bbm7"]
    assert tonic_pre and tonic_post
    assert {pcs[k] for k in tonic_pre} == {parse_key(MKEY).tonic}
    assert {pcs[k] for k in tonic_post} == {(parse_key(MKEY).tonic + 1) % 12}


def test_far_key_holds_and_says_why(run, machine_session, joiners):
    """Behavior 5's other half: a tritone away is not a capo, it is a different
    tune.  Conforming would be guessing, so the machine holds — and logs it."""
    wav, _ = joiners["far_key"]
    res, lines = run("far_key", wav, session=machine_session, seed=0,
                     mix_human_db=-120.0)
    assert [b for b, e in events_of(res) if e == "conform"] == []
    assert [b for b, e in events_of(res) if e == "conform_hold"]
    assert res.manifest["transpose_semitones"] == 0
    assert parse_key(res.final_key).tonic == parse_key(MKEY).tonic
    held = [r["reason"] for r in res.role_states if "conform_hold" in r["events"]]
    assert any("holding key" in h and "conforming would be guessing" in h
               for h in held)

    # the control for the transpose test: the rendered low end never moves off
    # the untransposed harmony.
    rows = bars_of(lines)
    x, sr = machine_bus(res)
    pcs = bar_pcs(res, x, sr)
    tonic_bars = [k for k in range(len(rows)) if rows[k]["chord"] == "Am7"]
    assert tonic_bars
    assert {pcs[k] for k in tonic_bars} == {parse_key(MKEY).tonic}


# ===========================================================================
# 6. role handoff (behaviors 6, 11)
# ===========================================================================

def test_takes_over_makes_the_machine_yield_and_thin(
        run, machine_session, joiners):
    """Behavior 6.  ``clock_authority`` falls, the role becomes ``follow``, and
    the machine's own density drops — in the plan AND in the bus level."""
    wav, _ = joiners["takes_over"]
    res, lines = run("takes_over", wav, session=machine_session, seed=0,
                     mix_human_db=-120.0)
    yields = [b for b, e in events_of(res) if e == "yield"]
    assert yields, "the machine never got out of the way"
    at = yields[0]
    assert res.role_states[-1]["role"] == "follow"
    assert (res.role_states[-1]["clock_authority"]
            < res.role_states[0]["clock_authority"])

    rows = bars_of(lines)
    before = np.median([r["machine_density"] for r in rows[:at]])
    after = np.median([r["machine_density"] for r in rows[at + 3:]])
    assert after < before, f"machine density {before} -> {after}, did not drop"

    x, sr = machine_bus(res)
    def seg(k):
        return x[int(res.bar_times_s[k] * sr):int(res.bar_times_s[k + 1] * sr)]
    early = np.median([rms_db(seg(k)) for k in range(2, min(at, 6))])
    late = np.median([rms_db(seg(k)) for k in range(at + 4, len(rows) - 1)])
    assert late < early, f"machine bus {early:.1f} -> {late:.1f} dB, did not back off"


def test_trade_alternates_authority_on_the_trade_unit(
        run, machine_session, joiners):
    """Behavior 11.  Authority alternates every ``trade_bars`` at boundaries."""
    wav, _ = joiners["takes_over"]
    res, _ = run("trade", wav, session=machine_session, seed=0,
                 role=RoleParams(role="trade", trade_bars=4))
    clock = [r["clock_authority"] for r in res.role_states]
    # the machine takes the first four; the human's four follow.
    assert all(c == pytest.approx(0.9) for c in clock[:4])
    assert all(c == pytest.approx(0.15) for c in clock[4:8])
    assert [e for _, e in events_of(res) if e == "trade"]
    # a collision is a one-bar backoff, not a turn change
    assert [e for _, e in events_of(res) if e == "trade_collision"]


# ===========================================================================
# 7. cues (behavior 10)
# ===========================================================================

def test_cues_are_delivered_applied_and_reported(run, machine_session):
    res, lines = run("cues", None, session=machine_session, seed=0,
                     cues=[(5.0, Cue("darker")),
                           (5.0, Cue("go to the chorus")),
                           (10_000.0, Cue("take it out"))],
                     mix_human_db=-120.0)
    applied = [f for f in res.form_plans if f["applied_cue"]]
    assert applied, "no cue ever applied"
    # both cues issued at the same instant land (joined, not dropped)
    texts = " ".join(f["applied_cue"] for f in applied)
    assert "darker" in texts and "go to the chorus" in texts
    # a cue issued past the last boundary is reported, not silently swallowed
    undelivered = [ln for ln in lines if ln.get("event") == "cue_undelivered"]
    assert len(undelivered) == 1
    assert undelivered[0]["cue"] == "take it out"
    # every applied cue reaches the NDJSON, not only the SimResult
    rows = bars_of(lines)
    assert any(r["form"]["applied_cue"] for r in rows)


def test_a_cue_that_names_no_section_is_a_no_op_with_a_reason(
        run, machine_session):
    """A name that is genuinely not in this map is refused, and says so."""
    res, _ = run("cues_missing", None, session=machine_session, seed=0,
                 cues=[(5.0, Cue("go to the tuba solo")),
                       (10_000.0, Cue("take it out"))],
                 mix_human_db=-120.0)
    applied = [f for f in res.form_plans if f["applied_cue"]][0]
    assert "no such section" in applied["reason"]
    assert applied["action"] in ("hold", "advance", "extend")


def test_the_documented_cue_example_moves_a_REAL_composed_session(
        run, machine_session):
    """`plume cue "go to the chorus"` — the help's own example — on a session
    plume actually composed.

    THE DEFECT (v2.0): ``parse_cue`` knew nineteen generic section names and
    ``compose`` emits none of them, so the two vocabularies had an EMPTY
    intersection and the whole section-cue kind was dead on real material.
    MEASURED: the documented example logged "no such section in the map —
    ignored", and naming the session's real section instead ("go to drift-a")
    logged "not understood — no change".  Both directions now work, and the
    log names the section it landed on rather than the word the user typed.
    """
    flat = flatten_sections(machine_session)
    hottest = max(range(len(flat)), key=lambda i: flat[i].energy)

    res, _ = run("cues_generic", None, session=machine_session, seed=0,
                 cues=[(5.0, Cue("go to the chorus", at="next-bar"))],
                 mix_human_db=-120.0)
    hit = [f for f in res.form_plans if f["applied_cue"]][0]
    assert hit["action"] == "jump" and hit["section_idx"] == hottest
    assert flat[hottest].name in hit["reason"], (
        "the log must name the section it CHOSE, not the word that was typed")

    # ...and the session's own name for a section works too.
    target = 2
    res2, _ = run("cues_named", None, session=machine_session, seed=0,
                  cues=[(5.0, Cue(f"go to {flat[target].name}", at="next-bar"))],
                  mix_human_db=-120.0)
    hit2 = [f for f in res2.form_plans if f["applied_cue"]][0]
    assert hit2["action"] == "jump" and hit2["section_idx"] == target
    assert "not understood" not in hit2["reason"]


def _solo(run, session, tag, cues=None, bars=16):
    return run(tag, None, session=session, seed=0, bars=bars, cues=cues,
               mix_human_db=-120.0)


def test_every_cue_kind_either_changes_the_audio_or_says_it_did_not(
        run, machine_session, lead_dir):
    """THE ERGONOMICS P0, as a matrix.

    MEASURED before v2.1, in a MACHINE SOLO with full authority (role=lead,
    clock 0.90, form 1.00 — nothing to yield to, nothing to explain a no-op):
    ``--cue "8.0:faster@next-bar"`` logged ``(tempo:faster): applied``, the bpm
    stayed at 96.0 on every subsequent bar, and the WAV was BYTE-IDENTICAL to
    the no-cue run.  The same for slower / double time / half time / darker /
    brighter / more space, and ``take it out`` logged "ending the tune here"
    and then played every remaining bar.  Of 18 cue phrases brute-forced, only
    the energy family had any effect at all.

    The decision log is the only instrument a cold user has.  So: every cue
    either lands in the audio, or the log says in that same bar that it did
    NOT and why.  Never "applied" for something a listener cannot hear.
    """
    base, _ = _solo(run, machine_session, "cuefx_base")
    ref = sha(base.out_path)

    def effects(lines):
        return [e for r in bars_of(lines) for e in (r.get("cue_effects") or [])]

    for i, text in enumerate(("faster", "slower", "double time", "half time",
                              "darker", "brighter", "sparser", "busier")):
        res, lines = _solo(run, machine_session, f"cuefx_{i}",
                           cues=[(4.0, Cue(text, at="next-bar"))])
        assert sha(res.out_path) != ref, (
            f"cue {text!r} produced a byte-identical take — it did nothing")
        assert effects(lines), f"cue {text!r} changed the audio and said nothing"

    # ...and the honest no-op: understood, not applied, and it SAYS so.
    res, lines = _solo(run, machine_session, "cuefx_style",
                       cues=[(4.0, Cue("make it jazz", at="next-bar"))])
    notes = effects(lines)
    assert sha(res.out_path) == ref, "the style cue changed the audio after all"
    assert notes and "NOT applied" in notes[0], (
        f"a cue that did nothing must say so, got {notes!r}")


def test_a_tempo_cue_moves_the_bpm_by_the_amount_it_logged(run, machine_session):
    for text, want in (("double time", 2.0), ("half time", 0.5)):
        res, _ = _solo(run, machine_session, f"cuebpm_{want}",
                       cues=[(4.0, Cue(text, at="next-bar"))])
        bpms = res.machine_bpm_by_bar
        assert bpms[0] == pytest.approx(MBPM)
        assert bpms[-1] == pytest.approx(MBPM * want, rel=1e-6)
        # ...and the BAR GRID follows, because a bar's length IS the tempo
        # (the voice tail past the last bar line is not part of that sum).
        assert res.bar_times_s[-1] == pytest.approx(
            sum(240.0 / b for b in bpms[:-1]), abs=0.01)


def test_take_it_out_actually_ends_the_take(run, machine_session):
    """P1: `take it out` reached the decision log and nowhere else.

    MEASURED before v2.1: cued at bar 10, ``FormPlan.action`` was ``"end"``
    for the next 22 bars, the machine bus sat at -5..-7 dB per bar, and the
    take ran its full 64.65 s.  ``_LeadPlanner`` never read ``FormPlan.action``
    at all, so an ending was a log line with no music in it.
    """
    full, _ = _solo(run, machine_session, "end_full", bars=16)
    ended, end_lines = _solo(run, machine_session, "end_cue", bars=16,
                             cues=[(4.0, Cue("take it out", at="next-bar"))])
    assert len(ended.bar_times_s) < len(full.bar_times_s), \
        "the take did not end — same number of bars as the uncued run"
    assert ended.duration_s < full.duration_s * 0.75
    ev = [ln for ln in end_lines if ln.get("event") == "ended"]
    assert len(ev) == 1 and ev[0]["bars"] == len(ended.bar_times_s)
    # the ending is not a truncation: the last bar is fully rendered and the
    # voice tail is still there.
    x, sr = sf.read(ended.out_path, dtype="float32", always_2d=True)
    assert x.shape[0] / sr >= ended.bar_times_s[-1]


def test_a_ritardando_cue_really_slows_down_bar_by_bar(run, machine_session):
    res, _ = _solo(run, machine_session, "cue_rit", bars=16,
                   cues=[(4.0, Cue("ritardando", at="next-bar"))])
    bpms = res.machine_bpm_by_bar
    tail = bpms[4:]
    assert all(b <= a + 1e-9 for a, b in zip(tail, tail[1:])), \
        f"a ritardando that does not decrease: {bpms}"
    assert tail[-1] < tail[0] * 0.9


# ===========================================================================
# 8. determinism (behavior 12)
# ===========================================================================

@pytest.mark.parametrize("tag,kind,kw", [
    ("det_solo", None, {}),
    ("det_rush", "rushing", {}),
    ("det_join", "on_grid", {"human_offset_s": 16.0}),
    ("det_trade", "takes_over", {"role": RoleParams(role="trade")}),
    ("det_cue", None, {"cues": [(5.0, Cue("darker")),
                                (9.0, Cue("go to the chorus", at="next-bar"))]}),
])
def test_determinism_byte_identical(lead_dir, machine_session, joiners,
                                    tag, kind, kw):
    wav = joiners[kind][0] if kind else None
    outs = []
    for i in (0, 1):
        res = simulate_lead(wav, session=machine_session, seed=0,
                            out_path=str(lead_dir / f"{tag}_{i}.wav"), **kw)
        outs.append((sha(res.out_path), sha(res.decisions_path)))
    assert outs[0] == outs[1], f"{tag} is not deterministic"


def test_a_different_seed_gives_different_audio(lead_dir, machine_session):
    a = simulate_lead(None, session=machine_session, seed=0,
                      out_path=str(lead_dir / "seed0.wav"))
    b = simulate_lead(None, session=machine_session, seed=1,
                      out_path=str(lead_dir / "seed1.wav"))
    assert sha(a.out_path) != sha(b.out_path)
    # ...but the decisions are identical: the seed is the renderer's, and the
    # negotiation must not be a function of it.
    assert sha(a.decisions_path) == sha(b.decisions_path)


# ===========================================================================
# 9. REAL MATERIAL — a v0 render as the human, against a different session
# ===========================================================================

@pytest.fixture(scope="session")
def real_runs(run, real_human, other_session) -> dict:
    out = {}
    for gain, wav in real_human.items():
        out[gain] = run(f"real{int(gain)}", wav, session=other_session, seed=0,
                        mix_human_db=-120.0)
    return out


@pytest.mark.parametrize("gain", [0.0, -20.0])
def test_real_material_the_machine_leads_all_the_way_through(real_runs, gain):
    """A REAL v0 render (trap, 140 bpm, D minor) joining a machine playing a
    different composed session (lofi, 96 bpm, F minor), at two gains.

    This is the case v1 could not survive: percussive real audio whose bar mean
    RMS sits far under an absolute floor.  In lead mode the failure mode would
    be the machine reading the human as absent and either stopping or handing
    over the lead — so what is asserted is that it plays every bar and keeps
    its clock.
    """
    res, lines = real_runs[gain]
    rows = bars_of(lines)
    assert len(rows) >= 32
    assert "tacet" not in res.states_by_bar
    assert res.role_states[-1]["role"] == "lead"
    assert res.role_states[-1]["clock_authority"] == pytest.approx(0.9)

    x, sr = machine_bus(res)
    for k in range(len(res.bar_times_s) - 1):
        seg = x[int(res.bar_times_s[k] * sr):int(res.bar_times_s[k + 1] * sr)]
        assert rms_db(seg) > -45.0, f"machine bar {k} silent at {gain} dB"


@pytest.mark.parametrize("gain", [0.0, -20.0])
def test_real_material_the_machine_holds_its_own_tempo_and_key(real_runs, gain):
    """A 140 bpm human is not a reason to leave 96 bpm, and D minor is 3
    semitones from F minor's collection — past ``conform_semitones``, so the
    machine must HOLD and say so (SPEC-V2 section 1, amendment A6)."""
    res, _ = real_runs[gain]
    assert abs(res.role_states[-1]["tempo_offset_pct"]) <= 8.0
    assert res.manifest["transpose_semitones"] == 0
    assert parse_key(res.final_key).tonic == parse_key("F minor").tonic
    bpms = np.array(res.machine_bpm_by_bar)
    assert float(bpms.min()) > 96.0 * 0.92 and float(bpms.max()) < 96.0 * 1.08


def test_real_material_the_machine_bus_does_not_chase_the_input_gain(real_runs):
    """The v1 bug, restated for lead mode.  Dropping the human 20 dB must not
    move the machine's own level: in lead mode the machine is the reference and
    ``mix_human_db`` is the human's fader (v2 amendment: no bus calibration)."""
    hot, _ = real_runs[0.0]
    quiet, _ = real_runs[-20.0]
    a = hot.manifest["per_bus_rms_db"]["machine"]
    b = quiet.manifest["per_bus_rms_db"]["machine"]
    assert abs(a - b) < 3.0, f"machine bus moved {a:.1f} -> {b:.1f} dB with the input gain"


def test_real_material_grid_is_exact_at_both_gains(real_runs):
    for gain in (0.0, -20.0):
        res, _ = real_runs[gain]
        want = np.concatenate([[0.0], np.cumsum(
            240.0 / np.array(res.machine_bpm_by_bar))[:-1]])
        assert float(np.max(np.abs(want - np.array(res.bar_times_s)))) < 0.002


def test_real_material_is_deterministic(lead_dir, real_human, other_session):
    wav = real_human[0.0]
    outs = []
    for i in (0, 1):
        res = simulate_lead(wav, session=other_session, seed=0,
                            out_path=str(lead_dir / f"realdet{i}.wav"))
        outs.append((sha(res.out_path), sha(res.decisions_path)))
    assert outs[0] == outs[1]


def test_real_material_mixes_the_human_in_at_the_offset(
        lead_dir, real_human, other_session):
    """``human_offset_s`` is a real placement in the output, not bookkeeping."""
    res = simulate_lead(real_human[0.0], session=other_session, seed=0,
                        human_offset_s=8.0, mix_human_db=12.0,
                        out_path=str(lead_dir / "real_off.wav"))
    x, sr = sf.read(res.out_path, dtype="float32", always_2d=True)
    mono = x.mean(axis=1)
    before = rms_db(mono[int(4.0 * sr):int(7.5 * sr)])
    after = rms_db(mono[int(9.0 * sr):int(12.5 * sr)])
    assert after > before + 4.0, (
        f"the human is not audibly present after the offset ({before:.1f} -> "
        f"{after:.1f} dB)")
    assert res.manifest["human_offset_s"] == 8.0


# ===========================================================================
# 10. cost on the live path
# ===========================================================================

def test_v2_analysis_cost_on_the_live_path(real_human, other_session):
    """v1's beat tracker runs ~180x real time and that headroom is why live jam
    is possible.  Everything ``simulate_lead`` ADDS to the per-bar analysis —
    the sub-hop attack refinement, the phase measurement, the role negotiator,
    the form cursor and the lead planner — is timed here against the audio it
    covers.  MEASURED: ~1.6 ms for 80 s of audio, i.e. ~49,000x real time; the
    floor below is 100x, three orders of magnitude of slack, so this test fails
    on an algorithmic regression and not on a busy machine.
    """
    from plume.listen.beat import BeatTracker
    from plume.listen.harmony import HarmonyTracker
    from plume.mind import BarFeatures, FormParams, Mind
    from plume.roles import RoleNegotiator, onset_phase_offset_ms

    x, sr = sf.read(real_human[0.0], dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    fe = FeatureExtractor(sr)
    frames = fe.process(x) + fe.flush()
    bt, ht = BeatTracker(), HarmonyTracker()
    beat, harm = bt.process([]), ht.process([])
    for f in frames:
        beat, harm = bt.process([f]), ht.process([f])
    onsets = [f.t_s for f in frames if f.onset]

    neg = RoleNegotiator(RoleParams(role="lead"), composed_bpm=96.0,
                         composed_key="F minor")
    mind = Mind(MindParams(), FormParams())
    planner = sim_mod._LeadPlanner(MindParams())
    refiner = sim_mod._AttackRefiner(x, sr)
    bar_s, n_bars = 240.0 / 96.0, 32

    t0 = time.perf_counter()
    for k in range(n_bars):
        att = [refiner.refine(t) for t in onsets
               if k * bar_s <= t < (k + 1) * bar_s]
        off = onset_phase_offset_ms(att, k * bar_s, (k + 1) * bar_s, bar_s / 16.0)
        bf = BarFeatures(bar=k - 1, rms_db=-20.0, onset_density=3.0,
                         silent=False, register_low_midi=40.0,
                         register_high_midi=70.0)
        rs = neg.update(bar=k, beat=beat, harmony=harm, human=bf,
                        machine_grid_t_s=k * bar_s, human_offset_ms=off)
        fp = mind.plan_form(rs, beat, harm, bf, other_session)
        planner.observe(bf, harm)
        planner.plan(k, energy=0.5, clock_authority=rs.clock_authority,
                     section_name=str(fp.section_idx))
    wall = time.perf_counter() - t0

    covered = n_bars * bar_s
    ratio = covered / max(wall, 1e-9)
    assert ratio > 100.0, (
        f"v2 per-bar analysis runs at only {ratio:.0f}x real time "
        f"({wall * 1000:.2f} ms for {covered:.0f}s of audio)")
