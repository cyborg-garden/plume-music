"""Tests for plume.duet (SPEC-V1 section 5).

Standalone by design: ``BarPlan`` is duck-typed by the duet engine, so when
``plume.mind`` (builder D, concurrent) is not present yet we substitute a
field-identical stand-in dataclass and every test here still runs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from plume.duet import CHORD_CONF_FULL, DuetEngine, choose_voicing
from plume.listen.beat import BeatState
from plume.listen.harmony import HarmonyState
from plume.theory import parse_key

try:  # integration: use the real contract type once builder D lands
    from plume.mind import BarPlan
except ImportError:  # pragma: no cover - stand-in matches SPEC-V1 section 4
    @dataclass(frozen=True)
    class BarPlan:  # type: ignore[no-redef]
        bar: int
        state: str
        roles: dict[str, float]
        gain_db: float
        density: float
        register_ceiling_midi: float | None
        reason: str


SR = 48000


def make_plan(bar=0, state="comping", roles=None, gain_db=0.0, density=0.5,
              ceiling=None, reason="test"):
    return BarPlan(bar=bar, state=state,
                   roles=roles if roles is not None else {"pad": 1.0},
                   gain_db=gain_db, density=density,
                   register_ceiling_midi=ceiling, reason=reason)


def make_beat(bpm=96.0, t_s=0.0, bar=0):
    iv = 60.0 / bpm
    return BeatState(t_s=t_s, bpm=bpm, confidence=0.9, locked=True, bar=bar,
                     beat_in_bar=0, phase_beats=0.0, next_beat_t_s=t_s + iv,
                     next_downbeat_t_s=t_s + 4 * iv, beat_interval_s=iv)


def make_harmony(key="A minor", chord="Am", chord_conf=0.9,
                 reg_lo=None, reg_hi=None):
    return HarmonyState(t_s=0.0, key=key, key_conf=0.8, chord=chord,
                        chord_conf=chord_conf, register_low_midi=reg_lo,
                        register_high_midi=reg_hi)


def engine(seed=7, style="lofi", **kw):
    return DuetEngine(style=style, seed=seed, **kw)


# ---------------------------------------------------------------------------
# bar-length exactness
# ---------------------------------------------------------------------------

def test_bar_length_zero_drift_at_awkward_bpm():
    """97.3 bpm over 32 bars: rounded bar lengths accumulate zero drift."""
    bpm = 97.3
    eng = engine()
    beat = make_beat(bpm)
    lens = [eng.render_bar(make_plan(bar=b, roles={}), beat,
                           make_harmony()).shape[0] for b in range(32)]
    ideal = 4.0 * (60.0 / bpm) * SR
    assert abs(sum(lens) - 32 * ideal) < 0.5          # zero cumulative drift
    for n in lens:
        assert abs(n - ideal) <= 1.0                  # each bar within 1 sample


@pytest.mark.parametrize("bpm", [60.0, 84.5, 121.7, 199.9])
def test_bar_length_zero_drift_various_bpm(bpm):
    eng = engine()
    beat = make_beat(bpm)
    lens = [eng.render_bar(make_plan(bar=b, roles={}), beat,
                           make_harmony()).shape[0] for b in range(16)]
    assert abs(sum(lens) - 16 * 4.0 * (60.0 / bpm) * SR) < 0.5


def test_render_bar_rejects_zero_bpm():
    eng = engine()
    bad = BeatState(t_s=0.0, bpm=0.0, confidence=0.0, locked=False, bar=-1,
                    beat_in_bar=0, phase_beats=0.0, next_beat_t_s=math.inf,
                    next_downbeat_t_s=math.inf, beat_interval_s=math.inf)
    with pytest.raises(ValueError):
        eng.render_bar(make_plan(), bad, make_harmony())


def test_output_is_stereo_float32():
    out = engine().render_bar(make_plan(), make_beat(), make_harmony())
    assert out.dtype == np.float32 and out.ndim == 2 and out.shape[1] == 2
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# overlap/tail buffer: no clicks at bar joins
# ---------------------------------------------------------------------------

def test_bar_joins_match_continuous_render():
    """Joined per-bar outputs == one continuous render of the same events."""
    bpm = 96.0
    beat = make_beat(bpm)
    harm = make_harmony(chord="Fm", key="F minor")
    plans = [make_plan(bar=b, roles={"pad": 1.0, "bass": 1.0}, density=0.2)
             for b in range(2)]

    eng = engine(seed=11)
    joined = np.concatenate(
        [eng.render_bar(p, beat, harm) for p in plans] + [eng.flush_tail()])

    twin = engine(seed=11)                    # same seed => same raw bars
    raws, lens = [], []
    for p in plans:
        n = twin._next_len(bpm)
        raws.append(twin._render_raw(p, beat, harm, n))
        lens.append(n)
    total = max(lens[0] + raws[1].shape[0], raws[0].shape[0])
    cont = np.zeros((total, 2))
    cont[: raws[0].shape[0]] += raws[0]
    cont[lens[0]: lens[0] + raws[1].shape[0]] += raws[1]
    ref = cont.astype(np.float32)

    m = min(joined.shape[0], ref.shape[0])
    diff = joined[:m] - ref[:m]
    assert float(np.sqrt(np.mean(diff ** 2))) < 1e-7
    # anything past the shorter buffer is silence on the longer one
    longer = joined if joined.shape[0] >= ref.shape[0] else ref
    assert float(np.max(np.abs(longer[m:]), initial=0.0)) < 1e-7


def _entry_plans():
    """tacet, then two entering bars — the shape of a real entrance."""
    return [make_plan(bar=0, state="tacet", roles={}, density=0.0),
            make_plan(bar=1, state="entering",
                      roles={"pad": 1.0, "bass": 1.0}, density=0.2),
            make_plan(bar=2, state="entering",
                      roles={"pad": 1.0, "bass": 1.0}, density=0.2)]


def _run(eng, plans, beat, harm):
    return np.concatenate([eng.render_bar(p, beat, harm) for p in plans]
                          + [eng.flush_tail()])


def test_entry_bars_are_the_raw_render_times_the_equal_power_curve():
    """v1.4: the entrance envelope, verified against the curve independently.

    The pre-v1.4 guarantee was "joined bars == one continuous render".  With
    a fade running the guarantee generalizes to "== continuous render * the
    equal-power fade", and this pins the exact curve: any per-bar staircase,
    any restart of the phase at a bar edge, or any envelope applied to the
    retained tail instead of the output would break it.
    """
    bpm, fade_bars = 96.0, 2.0
    beat, harm = make_beat(bpm), make_harmony(chord="Fm", key="F minor")
    plans = _entry_plans()

    faded = _run(engine(seed=11, fade_in_bars=fade_bars), plans, beat, harm)
    raw = _run(engine(seed=11, fade_in_bars=0.0, texture_xfade_beats=0.0),
               plans, beat, harm)

    n_bar = int(round(4.0 * (60.0 / bpm) * SR))       # bar 0 is silent/tacet
    m = min(faded.shape[0], raw.shape[0])
    t = np.arange(m, dtype=np.float64) - n_bar        # 0 at the entry downbeat
    phase = np.clip(t / (fade_bars * n_bar), 0.0, 1.0)
    env = np.sin(phase * (np.pi / 2.0))[:, None]

    diff = faded[:m] - raw[:m] * env
    assert float(np.max(np.abs(diff))) < 1e-6, "entry is not the equal-power fade"
    assert float(np.max(np.abs(faded[:n_bar]))) == 0.0   # the tacet bar stays silent


def test_exit_bars_are_the_raw_render_times_the_equal_power_curve():
    bpm, fade_bars = 96.0, 2.0
    beat, harm = make_beat(bpm), make_harmony(chord="Fm", key="F minor")
    roles = {"pad": 1.0, "bass": 1.0}
    plans = [make_plan(bar=0, state="comping", roles=roles, density=0.2),
             make_plan(bar=1, state="fading", roles=roles, density=0.2),
             make_plan(bar=2, state="fading", roles=roles, density=0.2)]

    faded = _run(engine(seed=11, fade_out_bars=fade_bars), plans, beat, harm)
    raw = _run(engine(seed=11, fade_out_bars=0.0, texture_xfade_beats=0.0),
               plans, beat, harm)

    n_bar = int(round(4.0 * (60.0 / bpm) * SR))
    m = min(faded.shape[0], raw.shape[0])
    t = np.arange(m, dtype=np.float64) - n_bar        # 0 where the exit starts
    phase = np.clip(t / (fade_bars * n_bar), 0.0, 1.0)
    env = np.cos(phase * (np.pi / 2.0))[:, None]

    diff = faded[:m] - raw[:m] * env
    assert float(np.max(np.abs(diff))) < 1e-6, "exit is not the equal-power fade"


def test_a_bar_with_no_fade_or_texture_change_is_untouched():
    """The v1.3 contract: nothing happening means nothing applied."""
    bpm = 96.0
    beat, harm = make_beat(bpm), make_harmony()
    plans = [make_plan(bar=b, state="comping",
                       roles={"pad": 1.0, "bass": 1.0}, density=0.2)
             for b in range(3)]
    a = _run(engine(seed=11), plans, beat, harm)
    b = _run(engine(seed=11, fade_in_bars=0.0, fade_out_bars=0.0,
                    texture_xfade_beats=0.0), plans, beat, harm)
    assert a.tobytes() == b.tobytes()


def test_flush_tail_returns_leftover_then_empty():
    eng = engine()
    out = eng.render_bar(make_plan(roles={"pad": 1.0, "bass": 1.0}),
                         make_beat(96.0), make_harmony())
    assert float(np.max(np.abs(out))) > 0.0
    tail = eng.flush_tail()
    assert tail.shape[0] > 0 and float(np.max(np.abs(tail))) > 0.0
    assert eng.flush_tail().shape[0] == 0


# ---------------------------------------------------------------------------
# graceful harmonic degradation tiers
# ---------------------------------------------------------------------------

def test_degradation_tier_triad():
    notes, tier = choose_voicing("Am", 0.9, "A minor",
                                 key_fallback=parse_key("C major"))
    assert tier == "triad"
    assert {n % 12 for n in notes} == {9, 0, 4}        # A, C, E


def test_degradation_tier_root_fifth():
    notes, tier = choose_voicing("Am", CHORD_CONF_FULL - 0.1, "A minor",
                                 key_fallback=parse_key("C major"))
    assert tier == "root_fifth"
    assert len(notes) == 2 and notes[1] - notes[0] == 7
    assert notes[0] % 12 == 9


def test_degradation_tier_key_pedal():
    notes, tier = choose_voicing(None, 0.0, "F minor",
                                 key_fallback=parse_key("C major"))
    assert tier == "key_pedal"
    assert len(notes) == 1 and notes[0] % 12 == 5      # F


def test_degradation_tier_fallback_pedal():
    notes, tier = choose_voicing(None, 0.0, None,
                                 key_fallback=parse_key("D minor"))
    assert tier == "fallback_pedal"
    assert len(notes) == 1 and notes[0] % 12 == 2      # D


def test_no_belief_still_renders_audio():
    silent_harmony = HarmonyState(t_s=0.0, key=None, key_conf=0.0, chord=None,
                                  chord_conf=0.0, register_low_midi=None,
                                  register_high_midi=None)
    eng = engine(key_fallback="F minor")
    out = eng.render_bar(make_plan(roles={"pad": 1.0}), make_beat(),
                         silent_harmony)
    assert float(np.max(np.abs(out))) > 0.0
    assert all(e["tier"] == "fallback_pedal" for e in eng.last_events)
    assert all(e["midi"] % 12 == 5 for e in eng.last_events)   # F pedal


# ---------------------------------------------------------------------------
# register ceiling
# ---------------------------------------------------------------------------

def test_register_ceiling_honored_in_voicings():
    notes, _ = choose_voicing("Am", 0.9, "A minor",
                              key_fallback=parse_key("C major"),
                              ceiling_midi=60.0)
    assert notes and max(notes) <= 60.0


def test_register_ceiling_honored_in_rendered_events():
    eng = engine()
    eng.render_bar(make_plan(roles={"pad": 1.0, "lead": 1.0}, ceiling=60.0),
                   make_beat(), make_harmony())
    tonal = [e for e in eng.last_events if e["role"] in ("pad", "lead")]
    assert tonal
    assert all(e["midi"] <= 60 for e in tonal)


def test_no_ceiling_allows_higher_voicings():
    hi = engine()
    hi.render_bar(make_plan(roles={"lead": 1.0}, ceiling=None),
                  make_beat(), make_harmony())
    lo = engine()
    lo.render_bar(make_plan(roles={"lead": 1.0}, ceiling=50.0),
                  make_beat(), make_harmony())
    assert max(e["midi"] for e in hi.last_events) > \
        max(e["midi"] for e in lo.last_events)


# ---------------------------------------------------------------------------
# density coupling + roles
# ---------------------------------------------------------------------------

def test_low_density_picks_sparser_patterns():
    roles = {"kick": 1.0, "hat": 1.0, "bass": 1.0, "pad": 1.0}
    sparse = engine()
    sparse.render_bar(make_plan(density=0.1, roles=roles), make_beat(),
                      make_harmony())
    busy = engine()
    busy.render_bar(make_plan(density=0.95, roles=roles), make_beat(),
                    make_harmony())
    assert len(sparse.last_events) < len(busy.last_events)


def test_density_monotone_per_role():
    for role in ("hat", "bass"):
        counts = []
        for d in (0.1, 0.5, 0.95):
            eng = engine()
            eng.render_bar(make_plan(density=d, roles={role: 1.0}),
                           make_beat(), make_harmony())
            counts.append(len(eng.last_events))
        assert counts[0] <= counts[1] <= counts[2]
        assert counts[0] < counts[2]


def test_roles_come_from_style():
    eng = engine()
    assert set(eng.roles) >= {"kick", "snare", "hat", "bass", "pad"}


def test_zero_level_and_unknown_roles_are_tacet():
    eng = engine()
    out = eng.render_bar(
        make_plan(roles={"kick": 0.0, "theremin": 1.0}), make_beat(),
        make_harmony())
    assert eng.last_events == []
    assert float(np.max(np.abs(out))) == 0.0


def test_role_level_scales_output():
    full = engine()
    a = full.render_bar(make_plan(roles={"pad": 1.0}), make_beat(),
                        make_harmony())
    half = engine()
    b = half.render_bar(make_plan(roles={"pad": 0.5}), make_beat(),
                        make_harmony())
    ra = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    rb = float(np.sqrt(np.mean(b.astype(np.float64) ** 2)))
    assert rb == pytest.approx(ra * 0.5, rel=1e-3)


def test_gain_db_applied():
    a = engine().render_bar(make_plan(gain_db=0.0), make_beat(),
                            make_harmony())
    b = engine().render_bar(make_plan(gain_db=-12.0), make_beat(),
                            make_harmony())
    ra = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    rb = float(np.sqrt(np.mean(b.astype(np.float64) ** 2)))
    assert 20.0 * math.log10(ra / rb) == pytest.approx(12.0, abs=0.1)


@pytest.mark.parametrize("style", ["lofi", "trap", "synthwave"])
def test_all_bundled_styles_render(style):
    eng = DuetEngine(style=style, seed=3)
    out = eng.render_bar(
        make_plan(roles={"kick": 1.0, "bass": 1.0, "pad": 1.0}, density=0.8),
        make_beat(120.0), make_harmony())
    assert float(np.max(np.abs(out))) > 0.0
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_byte_identical_across_engines():
    def run():
        eng = engine(seed=42)
        beat = make_beat(97.3)
        harm = make_harmony(chord="Fm", key="F minor")
        outs = [eng.render_bar(
            make_plan(bar=b, roles={"kick": 1.0, "hat": 0.7, "bass": 1.0,
                                    "pad": 0.8}, density=0.6),
            beat, harm) for b in range(3)]
        outs.append(eng.flush_tail())
        return np.concatenate(outs)

    a, b = run(), run()
    assert a.shape == b.shape
    assert a.tobytes() == b.tobytes()


def test_different_seeds_differ():
    p = make_plan(roles={"kick": 1.0, "hat": 1.0})
    a = DuetEngine(style="lofi", seed=1).render_bar(p, make_beat(),
                                                    make_harmony())
    b = DuetEngine(style="lofi", seed=2).render_bar(p, make_beat(),
                                                    make_harmony())
    assert a.tobytes() != b.tobytes()
