"""Gentle entrances, exits and texture changes (P1-D).

The first session with a human and an acoustic guitar produced the verdict:

    "it should have a softer fade in, music is gentle and cross faded"

Entry used to ramp a gain over whole bars and start/stop on a bar edge, so an
entrance read as a hard cut on sustained material.  These tests measure the
RENDERED machine bus — real ``DuetEngine`` output, not synthetic buffers —
and pin the four things that verdict asks for:

1. a fade-in spread smoothly across its FULL musical duration (a one-bar
   step must not be able to pass), and a fade-out that actually decays,
2. no sample-level discontinuity at any bar join, entrances and exits
   included, and a tail that rings on past the last bar,
3. texture changes that blend rather than splice, with no level notch,
4. fade length that comes from the style JSON, so it is data.

The measurement trick used throughout: render a plan sequence twice from the
same seed — once normally and once with the fades switched off — and divide.
The engine is deterministic, so the ratio of the two buses IS the envelope,
recovered from real audio without the test needing to know the curve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from plume.duet import FADE_DEFAULTS, DuetEngine
from plume.listen.beat import BeatState
from plume.listen.harmony import HarmonyState
from plume.styles_lib import load_style

try:  # the real contract once the mind's v1.4 fields land
    from plume.mind import BarPlan
except ImportError:  # pragma: no cover - stand-in matches SPEC-V1 section 4
    @dataclass
    class BarPlan:  # type: ignore[no-redef]
        bar: int
        state: str
        roles: dict
        gain_db: float
        density: float
        register_ceiling_midi: float | None
        reason: str
        announce: str | None = None
        announce_kind: str | None = None
        announce_bar: int | None = None


SR = 48000
BPM = 96.0

# A sustained, un-percussive texture: pad and bass only, low density.  This is
# the material the complaint came from (fingerpicked open-tuned acoustic), and
# it is also the only material on which a join discontinuity is unambiguous —
# a kick on the downbeat is a legitimate step of its own.
GENTLE = {"pad": 1.0, "bass": 1.0}
GENTLE_PLUS = {"pad": 1.0, "bass": 1.0, "hat": 0.7}

_NO_FADES = {"fade_in_bars": 0.0, "fade_out_bars": 0.0,
             "texture_xfade_beats": 0.0}


def plan(bar, state="comping", roles=None, gain_db=0.0, density=0.2,
         ceiling=None, announce=None, announce_kind=None):
    return BarPlan(bar=bar, state=state,
                   roles=GENTLE if roles is None else roles,
                   gain_db=gain_db, density=density,
                   register_ceiling_midi=ceiling, reason="test",
                   announce=announce, announce_kind=announce_kind)


def beat(bpm=BPM):
    iv = 60.0 / bpm
    return BeatState(t_s=0.0, bpm=bpm, confidence=0.9, locked=True, bar=0,
                     beat_in_bar=0, phase_beats=0.0, next_beat_t_s=iv,
                     next_downbeat_t_s=4 * iv, beat_interval_s=iv)


def harmony(key="A minor", chord="Am"):
    return HarmonyState(t_s=0.0, key=key, key_conf=0.8, chord=chord,
                        chord_conf=0.9, register_low_midi=None,
                        register_high_midi=None)


def render(plans, *, style="lofi", seed=5, bpm=BPM, flush=True, **kw):
    """Render a plan sequence -> (engine, bars, joined bus, join offsets)."""
    eng = DuetEngine(style=style, seed=seed, sr=SR, **kw)
    h = beat(bpm)
    hm = harmony()
    bars = [eng.render_bar(p, h, hm) for p in plans]
    tail = eng.flush_tail() if flush else np.zeros((0, 2), dtype=np.float32)
    edges = np.cumsum([b.shape[0] for b in bars])
    return eng, bars, np.concatenate(bars + [tail]), edges


def mono(x):
    return x.astype(np.float64).mean(axis=1)


def wrms(x, w):
    """Non-overlapping windowed RMS -> (centre index, rms) pairs."""
    x = np.asarray(x, dtype=np.float64)
    return [(i + w // 2, float(np.sqrt(np.mean(x[i:i + w] ** 2))))
            for i in range(0, x.shape[0] - w, w)]


def envelope(plans, *, floor=1e-4, w=2048, **kw):
    """Recover the applied bus envelope from two real renders.

    Same seed, same plans, fades on vs fades off: the quotient of the two
    windowed-RMS curves is the envelope the engine applied.  Windows where
    the un-faded reference is near silent are dropped (the quotient there is
    noise, not envelope).
    """
    _, _, faded, edges = render(plans, **kw)
    _, _, flat, _ = render(plans, **{**kw, **_NO_FADES})
    n = min(faded.shape[0], flat.shape[0])
    a, b = mono(faded[:n]), mono(flat[:n])
    out = []
    for (i, rb), (_, ra) in zip(wrms(b, w), wrms(a, w)):
        if rb > floor:
            out.append((i, ra / rb))
    return out, edges


# ---------------------------------------------------------------------------
# 1. the fade-in is spread across its whole musical duration
# ---------------------------------------------------------------------------

def test_fade_in_is_monotonic_over_its_full_musical_duration():
    """A two-bar fade rises all the way through bar two, not just bar one."""
    plans = [plan(0, "tacet", roles={}),
             plan(1, "entering"), plan(2, "entering"),
             plan(3, "comping"), plan(4, "comping")]
    env, edges = envelope(plans, fade_in_bars=2.0)
    entry = edges[0]                       # first sounding sample
    during = [(i, g) for i, g in env if entry <= i <= edges[2]]
    assert len(during) > 20, "not enough measurable windows inside the fade"

    vals = [g for _, g in during]
    assert all(a <= b + 2e-2 for a, b in zip(vals, vals[1:])), \
        f"fade-in is not monotonic: {[round(v, 3) for v in vals]}"
    assert vals[0] < 0.15, f"fade-in starts at {vals[0]:.3f}, not from silence"
    assert vals[-1] == pytest.approx(1.0, abs=0.02), \
        f"fade-in ends at {vals[-1]:.3f}, not full level"


def test_fade_in_midpoint_is_well_below_full_level():
    """The one-bar-step impostor test.

    A fade that finishes inside bar one would sit at full level for the whole
    second half of a two-bar fade.  Measured at the musical MIDPOINT of the
    declared duration, an honest fade must still be meaningfully down.
    """
    plans = [plan(0, "tacet", roles={}),
             plan(1, "entering"), plan(2, "entering"), plan(3, "comping")]
    env, edges = envelope(plans, fade_in_bars=2.0)
    mid = (edges[0] + edges[2]) / 2.0      # halfway through the two-bar fade
    at_mid = min(env, key=lambda kv: abs(kv[0] - mid))[1]
    end = [g for i, g in env if i >= edges[2]]
    assert at_mid < 0.85, \
        f"gain at the fade midpoint is {at_mid:.3f} — a one-bar step would pass"
    assert at_mid == pytest.approx(math.sin(math.pi / 4), abs=0.08), \
        f"midpoint {at_mid:.3f} is not the equal-power half-way gain"
    assert end and end[0] > at_mid + 0.15, "the fade did not keep rising"


def test_fade_in_is_a_curve_not_a_per_bar_staircase():
    """Inside a single bar the gain must move; a per-bar step would be flat."""
    plans = [plan(0, "tacet", roles={}),
             plan(1, "entering"), plan(2, "entering"), plan(3, "comping")]
    env, edges = envelope(plans, fade_in_bars=2.0)
    inside = [g for i, g in env if edges[0] < i < edges[1]]
    assert len(inside) > 8
    assert inside[-1] - inside[0] > 0.25, \
        "gain barely moved within the first fade bar — that is a staircase"


# ---------------------------------------------------------------------------
# 2. no discontinuity at bar joins — entrances and exits included
# ---------------------------------------------------------------------------

def _join_step(bus, edge, look=1024):
    """Step at a bar join, in units of the local sample-to-sample motion.

    A splice shows up as one sample-to-sample jump far larger than the
    signal's own slope around it; a continuous join does not.
    """
    d = np.abs(np.diff(bus[edge - look:edge + look]))
    step = abs(float(bus[edge] - bus[edge - 1]))
    local = float(np.percentile(d, 99.0))
    return step, local


@pytest.mark.parametrize("fade_in_bars", [1.0, 2.0, 3.0])
def test_entry_bar_joins_are_continuous(fade_in_bars):
    plans = [plan(0, "tacet", roles={}), plan(1, "entering"),
             plan(2, "entering"), plan(3, "entering"), plan(4, "comping")]
    _, _, bus, edges = render(plans, fade_in_bars=fade_in_bars)
    m = mono(bus)
    for k, e in enumerate(edges[:-1]):
        step, local = _join_step(m, int(e))
        assert step <= max(local * 1.5, 1e-6), (
            f"discontinuity at the join after bar {k}: step {step:.3e} vs "
            f"local motion {local:.3e}")


def test_exit_bar_joins_are_continuous_and_the_tail_rings_on():
    plans = [plan(b, "comping") for b in range(3)]
    plans += [plan(3, "fading", roles={"pad": 0.5, "bass": 0.3}, gain_db=-6.0)]
    eng, bars, bus, edges = render(plans, fade_out_bars=2.0, flush=False)
    tail = eng.flush_tail()

    m = mono(bus)
    for k, e in enumerate(edges[:-1]):
        step, local = _join_step(m, int(e))
        assert step <= max(local * 1.5, 1e-6), \
            f"discontinuity at the join after bar {k} of the exit"

    assert tail.shape[0] > 0, "nothing rang past the last bar — it was truncated"
    assert float(np.max(np.abs(tail))) > 0.0, "the tail is silent"
    joined = np.concatenate([bus, tail])
    step, local = _join_step(mono(joined), int(edges[-1]))
    assert step <= max(local * 1.5, 1e-6), "step into the ring-out tail"


def _rms(x):
    return float(np.sqrt(np.mean(mono(x) ** 2))) if x.shape[0] else 0.0


def test_fade_out_actually_decays_across_its_musical_duration():
    plans = [plan(b, "comping") for b in range(3)]
    plans += [plan(3, "fading", roles={"pad": 0.5, "bass": 0.3}),
              plan(4, "fading", roles={"pad": 0.5, "bass": 0.3})]
    _, bars, _, _ = render(plans, fade_out_bars=2.0, flush=False)

    first, second = bars[3], bars[4]
    h = first.shape[0] // 2
    assert _rms(first[h:]) < _rms(first[:h]) * 0.95, \
        "the fade-out bar did not decay within itself"
    assert _rms(second) < _rms(first) * 0.75, \
        "the fade-out did not keep decaying into the second bar"
    q = second.shape[0] // 4
    assert _rms(second[-q:]) < _rms(second[:q]) * 0.5, \
        "the fade-out did not reach silence by the end of its duration"


def test_fade_out_envelope_falls_smoothly_over_its_full_duration():
    """The exit twin of the fade-in test, measured the same way.

    Roles, density and gain are held constant across the exit so no texture
    crossfade or gain glide fires: the quotient against the un-faded render
    is then purely the fade-out envelope.
    """
    plans = [plan(0, "comping"), plan(1, "comping"),
             plan(2, "fading"), plan(3, "fading")]
    env, edges = envelope(plans, fade_out_bars=2.0)
    during = [(i, g) for i, g in env if edges[1] <= i <= edges[3]]
    assert len(during) > 20

    vals = [g for _, g in during]
    assert all(a >= b - 2e-2 for a, b in zip(vals, vals[1:])), \
        f"fade-out is not monotonic: {[round(v, 3) for v in vals]}"
    assert vals[0] == pytest.approx(1.0, abs=0.05), "fade-out did not start at full"
    mid = min(during, key=lambda kv: abs(kv[0] - (edges[1] + edges[3]) / 2))[1]
    assert mid == pytest.approx(math.cos(math.pi / 4), abs=0.1), \
        f"fade-out midpoint {mid:.3f} is not the equal-power half-way gain"
    assert vals[-1] < 0.2, f"fade-out ended at {vals[-1]:.3f}, not near silence"


# ---------------------------------------------------------------------------
# 3. texture changes blend rather than splice
# ---------------------------------------------------------------------------

def test_texture_change_triggers_a_crossfade_of_the_declared_length():
    plans = [plan(0, roles=GENTLE), plan(1, roles=GENTLE),
             plan(2, roles=GENTLE_PLUS)]
    eng, bars, _, _ = render(plans)
    beats = load_style("lofi")["fades"]["texture_xfade_beats"]
    assert eng.last_xfade_samples == pytest.approx(
        beats * (60.0 / BPM) * SR, rel=1e-3)


def test_steady_texture_never_pays_for_a_crossfade():
    """The second render only happens when the texture actually moves."""
    plans = [plan(b, roles=GENTLE) for b in range(4)]
    eng, _, _, _ = render(plans)
    assert eng.last_xfade_samples == 0


def test_texture_crossfade_has_no_level_notch():
    """Across the blend the bus never dips below either texture's own level."""
    a_only = [plan(b, roles=GENTLE) for b in range(3)]
    changed = [plan(0, roles=GENTLE), plan(1, roles=GENTLE),
               plan(2, roles=GENTLE_PLUS)]
    eng, blend_bars, _, _ = render(changed)
    _, a_bars, _, _ = render(a_only)
    _, b_bars, _, _ = render(changed, **_NO_FADES)   # hard splice, same audio

    w = eng.last_xfade_samples
    assert w > 0
    step = w // 8
    blend = np.array([r for _, r in wrms(mono(blend_bars[2][:w]), step)])
    old = np.array([r for _, r in wrms(mono(a_bars[2][:w]), step)])
    new = np.array([r for _, r in wrms(mono(b_bars[2][:w]), step)])
    floor = np.minimum(old, new)
    assert np.all(blend >= floor * 0.75), (
        "level notch inside the crossfade: "
        f"{np.round(blend / np.maximum(floor, 1e-9), 3)}")
    # ...and no swell either: equal-power on correlated textures would bulge
    assert np.all(blend <= np.maximum(old, new) * 1.35), "crossfade swells"


def test_texture_crossfade_join_is_continuous():
    plans = [plan(0, roles=GENTLE), plan(1, roles=GENTLE),
             plan(2, roles=GENTLE_PLUS), plan(3, roles=GENTLE_PLUS)]
    _, _, bus, edges = render(plans)
    m = mono(bus)
    step, local = _join_step(m, int(edges[1]))
    assert step <= max(local * 1.5, 1e-6), \
        "the texture change spliced instead of blending"


def test_gain_step_between_bars_is_glided_not_stepped():
    """A per-bar gain_db move (the entry ramp, the dynamic arc) is a fader move.

    The bar after a +9 dB move must START at the level it left off at and
    arrive at its own level by the end of the glide window — not jump.
    """
    plans = [plan(0, roles=GENTLE, gain_db=-9.0),
             plan(1, roles=GENTLE, gain_db=0.0),
             plan(2, roles=GENTLE, gain_db=0.0)]
    env, edges = envelope(plans, w=1024)
    glide_n = load_style("lofi")["fades"]["texture_xfade_beats"] * (60.0 / BPM) * SR

    at_join = [g for i, g in env if edges[0] <= i < edges[0] + 4096]
    inside = [g for i, g in env if edges[0] <= i < edges[0] + glide_n]
    after = [g for i, g in env if edges[0] + glide_n <= i < edges[1]]

    assert at_join, "no measurable window at the join"
    assert at_join[0] == pytest.approx(10.0 ** (-9.0 / 20.0), abs=0.06), (
        f"the bar starts at {at_join[0]:.3f}, not at the level the previous "
        "bar left off at — that is a step, not a glide")
    assert all(a <= b + 2e-2 for a, b in zip(inside, inside[1:])), \
        "the glide is not monotonic"
    assert after and all(g == pytest.approx(1.0, abs=0.03) for g in after), \
        "the glide never arrived at the bar's own level"

    # ...and with the glide window collapsed, the same move DOES step.
    hard, edges_h = envelope(plans, w=1024, texture_xfade_beats=0.0)
    at_join_hard = [g for i, g in hard if edges_h[0] <= i < edges_h[0] + 4096]
    assert at_join_hard[0] == pytest.approx(1.0, abs=0.03), \
        "the un-glided reference did not step — nothing is being measured"


# ---------------------------------------------------------------------------
# 4. fade length is style data, not code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", ["lofi", "synthwave", "trap"])
def test_fade_lengths_come_from_the_style_json(style):
    data = load_style(style)["fades"]
    eng = DuetEngine(style=style, seed=1, sr=SR)
    assert eng.fade_in_bars == data["fade_in_bars"]
    assert eng.fade_out_bars == data["fade_out_bars"]
    assert eng.texture_xfade_beats == data["texture_xfade_beats"]
    assert eng.announce_gain_db == data["announce_gain_db"]


def test_gentle_styles_fade_longer_than_punchy_ones():
    lofi = load_style("lofi")["fades"]
    trap = load_style("trap")["fades"]
    assert lofi["fade_in_bars"] > trap["fade_in_bars"]
    assert lofi["fade_out_bars"] > trap["fade_out_bars"]
    assert lofi["texture_xfade_beats"] > trap["texture_xfade_beats"]


def test_style_fade_length_changes_the_rendered_bus():
    """Not just a number on the object: lofi is measurably quieter entering."""
    plans = [plan(0, "tacet", roles={}), plan(1, "entering"),
             plan(2, "entering")]

    def entry_gain(style):
        env, edges = envelope(plans, style=style)
        during = [g for i, g in env if edges[0] <= i <= edges[1]]
        return float(np.mean(during))

    assert entry_gain("lofi") < entry_gain("trap") - 0.2, \
        "the style's fade length did not reach the audio"


def test_missing_fades_block_falls_back_to_defaults(monkeypatch):
    import plume.duet as duet_mod

    real = duet_mod.load_style

    def stripped(name):
        s = real(name)
        s.pop("fades", None)
        return s

    monkeypatch.setattr(duet_mod, "load_style", stripped)
    eng = DuetEngine(style="lofi", seed=1, sr=SR)
    assert eng.fade_in_bars == FADE_DEFAULTS["fade_in_bars"]
    assert eng.announce_beats == FADE_DEFAULTS["announce_beats"]


def test_constructor_overrides_beat_the_style():
    eng = DuetEngine(style="lofi", seed=1, sr=SR, fade_in_bars=0.25)
    assert eng.fade_in_bars == 0.25
    assert eng.fade_out_bars == load_style("lofi")["fades"]["fade_out_bars"]


# ---------------------------------------------------------------------------
# 5. the reciprocal cue: announcing an entry before it lands
# ---------------------------------------------------------------------------

def test_announced_entry_renders_a_quiet_pickup_in_the_bar_before():
    silent = plan(0, "listening", roles={})
    announced = plan(0, "listening", roles={},
                     announce="coming in at bar 1", announce_kind="entry")
    eng_q, bars_q, _, _ = render([silent], flush=False)
    eng_a, bars_a, _, _ = render([announced], flush=False)

    assert float(np.max(np.abs(bars_q[0]))) == 0.0, "control bar is not silent"
    peak = float(np.max(np.abs(bars_a[0])))
    assert peak > 0.0, "the announced entry made no sound at all"
    assert eng_a.last_announce is not None
    assert eng_a.last_announce["kind"] == "entry"


def test_announce_pickup_is_a_courtesy_not_a_fanfare():
    """Audible, but far under the band it is announcing."""
    announced = plan(0, "listening", roles={},
                     announce="coming in", announce_kind="entry")
    band = plan(1, "comping", roles=GENTLE)
    _, bars, _, _ = render([announced, band], flush=False)
    cue = float(np.sqrt(np.mean(mono(bars[0]) ** 2)))
    full = float(np.sqrt(np.mean(mono(bars[1]) ** 2)))
    assert cue > 0.0
    quiet_db = 20.0 * math.log10(cue / full)
    assert -45.0 < quiet_db < -12.0, \
        f"pickup sits {quiet_db:.1f} dB under the band — not a courtesy cue"


def test_announce_pickup_starts_and_ends_at_silence():
    """It leaves no ringing tail for the entry bar's own fade-in to chop."""
    announced = plan(0, "listening", roles={},
                     announce="coming in", announce_kind="entry")
    eng, bars, _, _ = render([announced], flush=False)
    b = mono(bars[0])
    assert abs(float(b[0])) == 0.0
    assert abs(float(b[-1])) < 1e-6, "the pickup is still ringing at the bar edge"
    assert eng.flush_tail().shape[0] == 0 or \
        float(np.max(np.abs(eng.flush_tail()), initial=0.0)) < 1e-6


def test_announced_entry_then_entry_has_no_step_at_the_downbeat():
    plans = [plan(0, "listening", roles={}, announce="coming in",
                  announce_kind="entry"),
             plan(1, "entering"), plan(2, "entering")]
    _, _, bus, edges = render(plans)
    step, local = _join_step(mono(bus), int(edges[0]))
    assert step <= max(local * 1.5, 1e-9), \
        "the announced entry stepped at the downbeat it announced"


def test_an_entry_cue_is_not_rendered_over_a_bar_that_already_sounds():
    """The pickup means "I am about to come in" — from nothing.

    On a bar the machine is already playing there is nothing to come in
    from, so the entry gesture is suppressed.  (A TEXTURE announcement on a
    sounding bar is the opposite case and IS voiced — see section 5b.)
    """
    p = plan(0, "comping", roles=GENTLE, announce="coming in",
             announce_kind="entry")
    eng, _, _, _ = render([p], flush=False)
    assert eng.last_announce is None


def test_an_unknown_announce_kind_renders_nothing():
    p = plan(0, "listening", roles={}, announce="something",
             announce_kind="weather")
    eng, bars, _, _ = render([p], flush=False)
    assert float(np.max(np.abs(bars[0]))) == 0.0
    assert eng.last_announce is None


# ---------------------------------------------------------------------------
# 5b. announcing a TEXTURE change — from inside the music
# ---------------------------------------------------------------------------
#
# mind.py emits ``announce_kind == "texture"`` (thinning to the skeleton, and
# the density coupling's one-bar-delayed recovery) with a reason that promises
# a gesture: "announcing texture next bar (swell)".  The renderer voiced only
# "entry", so the promised cue actually arrived as a texture crossfade — and a
# crossfade is a BLEND, something you notice after it has happened, not
# something you can play towards.  The requirement was explicit: "if it does
# really commit to something it has to also give the other musician a chance
# to adapt to that as well."  So the gesture has to be heard coming, and it
# has to stay a courtesy.  Both halves are measured here on real renders.

_TEXTURE_ANNOUNCE = dict(announce="thinning to the skeleton at bar 3",
                         announce_kind="texture")


def _rms1(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _db(a, b):
    return 20.0 * math.log10(max(a, 1e-30) / max(b, 1e-30))


def _plain_and_cued(**kw):
    """Bar 2 rendered twice, identical but for the texture announcement.

    The gesture is placed after every other event in the bar and draws its
    rng after them, so the two renders' events are sample-identical and the
    DIFFERENCE of the two bars is exactly the cue — measured, not modelled.
    """
    lead = [plan(0, "comping"), plan(1, "comping")]
    _, plain, _, _ = render(lead + [plan(2, "comping")], flush=False, **kw)
    eng, cued, _, _ = render(lead + [plan(2, "comping", **_TEXTURE_ANNOUNCE)],
                             flush=False, **kw)
    return eng, plain, cued


def test_an_announced_texture_change_is_voiced_at_all():
    """The half of the contract that was simply missing."""
    eng, plain, cued = _plain_and_cued()
    assert eng.last_announce is not None, \
        "the mind promised a swell and the renderer stayed silent"
    assert eng.last_announce["kind"] == "texture"
    assert not np.array_equal(plain[2], cued[2]), \
        "the announcement changed nothing in the rendered bus"


def test_the_texture_swell_is_audible_against_the_bar_it_shares():
    """It has to be heard THROUGH the band, not merely be non-zero.

    Unlike the entry pickup — which plays into silence and only has to exist
    — this cue shares its bar with the whole texture.  Measured inside the
    gesture's own window: the cue against the band over the same samples.
    Under the band (a courtesy) but not buried (a cue).
    """
    eng, plain, cued = _plain_and_cued()
    a = eng.last_announce
    i0 = int(round(a["onset_s"] * SR))
    i1 = min(i0 + int(round(a["dur_s"] * SR)), cued[2].shape[0])
    cue = mono(cued[2])[i0:i1] - mono(plain[2])[i0:i1]
    band = mono(plain[2])[i0:i1]
    assert i1 - i0 > SR // 4, "the cue window is too short to mean anything"
    under_db = _db(_rms1(cue), _rms1(band))
    assert -30.0 < under_db < -8.0, (
        f"the swell sits {under_db:.1f} dB under the bar it shares — "
        "inaudible below -30, a fanfare above -8")


def test_the_texture_swell_is_small_beside_the_bars_around_it():
    """A courtesy, not a fanfare — bounded, on the real bus."""
    eng, plain, cued = _plain_and_cued()
    lift_db = _db(_rms(cued[2]), _rms(plain[2]))
    assert 0.0 < lift_db < 0.75, \
        f"the announced bar is {lift_db:.2f} dB louder than the same bar " \
        "without the cue"
    peak_db = _db(float(np.max(np.abs(cued[2]))),
                  float(np.max(np.abs(plain[2]))))
    assert peak_db < 1.5, f"the cue lifted the bar's peak by {peak_db:.2f} dB"
    # ...and against a NEIGHBOURING bar, not only against itself
    assert float(np.max(np.abs(cued[2]))) < \
        float(np.max(np.abs(plain[1]))) * 1.3, \
        "the announced bar sticks out of the bars around it"


def test_the_texture_swell_leaves_no_click_at_either_bar_join():
    """It is shaped to zero at both ends, inside the bar it announces."""
    plans = [plan(0, "comping"), plan(1, "comping"),
             plan(2, "comping", **_TEXTURE_ANNOUNCE),
             plan(3, "comping", roles={"pad": 0.6, "bass": 0.3},
                  density=0.25)]
    _, _, bus, edges = render(plans)
    m = mono(bus)
    for k in (0, 1, 2):
        step, local = _join_step(m, int(edges[k]))
        assert step <= max(local * 1.5, 1e-6), \
            f"discontinuity at the join after bar {k} of the announced change"


def test_the_texture_swell_ends_at_exactly_the_bar_edge():
    eng, _, cued = _plain_and_cued()
    assert abs(float(mono(cued[2])[-1] - mono(_plain_and_cued()[1][2])[-1])) \
        < 1e-9, "the swell is still ringing at the bar edge it announces"


def test_the_texture_swell_stays_under_the_register_ceiling():
    """It speaks from ABOVE the voicing — but never over the human.

    The ceiling is the machine's promise to stay out of the human's
    register; delivering a courtesy by breaking it would be no courtesy.
    """
    p = plan(0, "comping", ceiling=60.0, **_TEXTURE_ANNOUNCE)
    eng, _, _, _ = render([p], flush=False)
    assert eng.last_announce is not None
    assert eng.last_announce["midi"] <= 60


def test_the_texture_swell_is_a_different_gesture_from_the_entry_pickup():
    """Otherwise "I'm coming in" and "I'm changing" are the same sound."""
    entry = plan(0, "listening", roles={}, announce="coming in",
                 announce_kind="entry")
    e_eng, _, _, _ = render([entry], flush=False)
    t_eng, _, _, _ = render([plan(0, "comping", **_TEXTURE_ANNOUNCE)],
                            flush=False)
    e, t = e_eng.last_announce, t_eng.last_announce
    assert t["midi"] > e["midi"] + 6, \
        f"both cues speak from the same register ({e['midi']}, {t['midi']})"
    assert t["voice_role"] != e["voice_role"], \
        f"both cues use the same voice ({e['voice_role']})"


def _cue_and_band(style="lofi", **plan_kw):
    """The cue signal and the band it shares, over the cue's own window."""
    lead = [plan(0, "comping", **plan_kw), plan(1, "comping", **plan_kw)]
    _, plain, _, _ = render(lead + [plan(2, "comping", **plan_kw)],
                            style=style, flush=False)
    eng, cued, _, _ = render(
        lead + [plan(2, "comping", **{**plan_kw, **_TEXTURE_ANNOUNCE})],
        style=style, flush=False)
    a = eng.last_announce
    i0 = int(round(a["onset_s"] * SR))
    i1 = min(i0 + int(round(a["dur_s"] * SR)), cued[2].shape[0])
    return (mono(cued[2])[i0:i1] - mono(plain[2])[i0:i1],
            mono(plain[2])[i0:i1])


def test_the_texture_swell_rises_instead_of_decaying():
    """It is a SWELL — the thing mind.py's reason text promises by name.

    A rising window over one note swells a sustaining voice and does the
    opposite to a plucked one: it deletes the attack and leaves the decay.
    Measured on lofi (whose `lead` is a pluck) the single-note version peaked
    at 10% of the gesture and fell monotonically from there — a diminuendo
    announcing itself as a crescendo.  So the gesture re-articulates, and
    this is the assertion that keeps it honest on any preset.
    """
    cue, _ = _cue_and_band()
    n = cue.shape[0] // 12
    chunks = [_rms1(cue[k * n:(k + 1) * n]) for k in range(12)]
    assert int(np.argmax(chunks)) >= 6, \
        f"the swell peaks in its first half: {[round(c, 5) for c in chunks]}"
    assert _rms1(cue[cue.shape[0] // 2:]) > _rms1(cue[:cue.shape[0] // 2]), \
        "the second half of the swell is quieter than the first"


def test_the_texture_swell_level_is_measured_against_the_band_not_assumed():
    """Turn the band down 12 dB; the cue must follow it down, not stay put.

    A fixed offset on the announcing role's mix gain cannot do this, and that
    is exactly how the cue ended up 48.5 dB under a lofi bus it was supposed
    to sit 18 dB under.
    """
    loud = _db(*[_rms1(x) for x in _cue_and_band(gain_db=0.0)])
    quiet = _db(*[_rms1(x) for x in _cue_and_band(gain_db=-12.0)])
    assert loud == pytest.approx(quiet, abs=0.5), (
        f"the cue sits {loud:.1f} dB under a loud band and {quiet:.1f} dB "
        "under a quiet one — its level is assumed, not measured")


@pytest.mark.parametrize("style", ["lofi", "synthwave", "trap"])
def test_every_bundled_style_can_voice_a_texture_announcement(style):
    eng, _, _, _ = render([plan(0, "comping", **_TEXTURE_ANNOUNCE)],
                          style=style, flush=False)
    assert eng.last_announce is not None, \
        f"{style} has no voice for the swell it promises"


@pytest.mark.parametrize("style", ["lofi", "synthwave", "trap"])
def test_the_texture_swell_is_audible_and_small_in_every_style(style):
    """The two halves of the courtesy, on every bundled bus.

    Role names do not carry timbre — lofi's `lead` is a pluck, synthwave's a
    supersaw, trap's `pad` a pluck — so a cue that is right on one style is
    not evidence about the others.
    """
    cue, band = _cue_and_band(style=style)
    under_db = _db(_rms1(cue), _rms1(band))
    assert -30.0 < under_db < -8.0, \
        f"{style}: the swell sits {under_db:.1f} dB under the bar it shares"
    assert _rms1(cue[cue.shape[0] // 2:]) > _rms1(cue[:cue.shape[0] // 2]), \
        f"{style}: the swell does not rise"


def test_the_texture_swell_is_deterministic():
    p = [plan(0, "comping", **_TEXTURE_ANNOUNCE)]
    a = render(p, flush=False)[1][0]
    b = render(p, flush=False)[1][0]
    assert a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# 7. the bar's own level — measured OUTSIDE the fade
# ---------------------------------------------------------------------------
#
# sim calibrates the machine bus off the first `entering` bar so the machine
# lands `gain_db` under the human.  Since the gentle-fade work, that bar sits
# INSIDE the two-bar fade-in, so the delivered samples are the bar TIMES the
# envelope and solving on them over-boosted the whole take (+3.2 dB overall,
# +1.35 dB steady-state on the `steady` fixture).  It self-corrected only if
# `entering -> comping` fired, and the committed mind often goes
# `entering -> thinned`, so the boost simply stayed.

def _pre_env_levels(plans, **kw):
    eng = DuetEngine(style="lofi", seed=5, sr=SR, **kw)
    h, hm = beat(), harmony()
    out = []
    for p in plans:
        eng.render_bar(p, h, hm)
        out.append(eng.last_bar_ms_pre_env)
    return out


def test_the_bars_own_level_is_identical_whether_or_not_a_fade_runs():
    """THE calibration invariant — exactly equal, not merely close."""
    plans = [plan(0, "tacet", roles={}), plan(1, "entering"),
             plan(2, "entering"), plan(3, "comping"), plan(4, "comping")]
    faded = _pre_env_levels(plans, fade_in_bars=3.0)
    flat = _pre_env_levels(plans, fade_in_bars=0.0)
    assert faded == flat, \
        f"the bar's own level moved with the fade: {faded} vs {flat}"


def test_the_delivered_entry_bar_really_is_attenuated_by_the_fade():
    """Guards the invariant above from passing vacuously.

    Also states the size of the error the old calibration made: it solved
    `delivered * cal == target` when it meant `own_level * cal == target`,
    so it over-boosted by exactly this much.
    """
    eng = DuetEngine(style="lofi", seed=5, sr=SR, fade_in_bars=3.0)
    h, hm = beat(), harmony()
    eng.render_bar(plan(0, "tacet", roles={}), h, hm)
    delivered = eng.render_bar(plan(1, "entering"), h, hm)
    own = eng.last_bar_ms_pre_env
    got = float(np.mean(delivered.astype(np.float64) ** 2))
    assert own > 0.0 and got > 0.0
    err_db = 10.0 * math.log10(own / got)
    assert err_db > 3.0, (
        f"the entry bar is only {err_db:.2f} dB down inside its fade — "
        "there would have been nothing to over-boost")


def test_the_bars_own_level_follows_the_plans_gain():
    """It is a real level, not a constant that happens to be invariant."""
    a = _pre_env_levels([plan(0, "comping", gain_db=0.0)])[0]
    b = _pre_env_levels([plan(0, "comping", gain_db=-6.0)])[0]
    assert 10.0 * math.log10(a / b) == pytest.approx(6.0, abs=0.2)


def test_the_bars_own_level_is_the_delivered_bar_when_nothing_is_fading():
    """With no fade and no glide the two are the same measurement."""
    eng = DuetEngine(style="lofi", seed=5, sr=SR)
    h, hm = beat(), harmony()
    eng.render_bar(plan(0, "comping"), h, hm)
    out = eng.render_bar(plan(1, "comping"), h, hm)
    assert eng.last_env_gain_rms == 1.0
    assert eng.last_bar_ms_pre_env == pytest.approx(
        float(np.mean(out.astype(np.float64) ** 2)), rel=1e-12)


# ---------------------------------------------------------------------------
# 8. a held wobble bar is a CONTINUATION, not a new entrance
# ---------------------------------------------------------------------------
#
# While committed, the mind answers a wobbling tracker with up to four
# SOUNDING hold bars (mind._plan_unlock_hold).  They arrive on the unlocked
# 1 Hz tick, and the beat counter FREEZES on unlock — so every one of them
# carries the same bar number, and sim keeps and places all of them.  The
# engine's other repeated-bar caller is the live planner revising a bar it is
# about to throw away.  The two want opposite things from the envelope, so the
# caller has to say which it is.

_HELD = 4


def _held(*, continuation, fade_in_bars=6.0):
    """Two real bars, then ``_HELD`` hold bars all carrying bar number 2."""
    eng = DuetEngine(style="lofi", seed=5, sr=SR, fade_in_bars=fade_in_bars)
    h, hm = beat(), harmony()
    eng.render_bar(plan(0, "tacet", roles={}), h, hm)
    eng.render_bar(plan(1, "entering"), h, hm)
    phases, gains, bars = [], [], []
    for _ in range(_HELD):
        bars.append(eng.render_bar(plan(2, "comping"), h, hm,
                                   continuation=continuation))
        phases.append(round(eng._env_phase, 12))
        gains.append(eng.last_env_gain_rms)
    return phases, gains, bars


def test_held_wobble_bars_keep_the_fade_moving():
    phases, gains, _ = _held(continuation=True)
    assert phases == sorted(phases) and len(set(phases)) == _HELD, \
        f"the fade stalled across the held bars: {phases}"
    assert gains == sorted(gains) and len(set(gains)) == _HELD, \
        f"the envelope did not keep rising across the held bars: {gains}"


def test_a_held_wobble_bar_does_not_restart_the_entrance():
    """The audible version of the same thing.

    Without the flag a wobble mid-entrance replayed the FIRST slice of the
    fade-in once per held bar, so the entrance restarted up to four times —
    which is what "starts up randomly, then stops" sounds like from the
    room.  A continued fade only ever gets louder, bar over bar.
    """
    _, _, bars = _held(continuation=True)
    levels = [_rms(b) for b in bars]
    assert all(a < b for a, b in zip(levels, levels[1:])), \
        f"the entrance restarted instead of continuing: {levels}"

    _, _, rolled = _held(continuation=False)
    rolled_levels = [_rms(b) for b in rolled]
    assert max(rolled_levels) - min(rolled_levels) < \
        (max(levels) - min(levels)) * 0.5, \
        "the rolled-back reference did not stall — nothing is being measured"


def test_without_the_flag_a_repeated_bar_still_rolls_the_fade_back():
    """The live planner's revision contract, unchanged and still the default."""
    phases, gains, _ = _held(continuation=False)
    assert len(set(phases)) == 1, f"the revision advanced the fade: {phases}"
    assert len(set(gains)) == 1


def test_held_wobble_bars_join_without_a_click():
    _, _, bars = _held(continuation=True)
    bus = mono(np.concatenate(bars))
    for k, e in enumerate(np.cumsum([b.shape[0] for b in bars])[:-1]):
        step, local = _join_step(bus, int(e))
        assert step <= max(local * 1.5, 1e-6), \
            f"discontinuity at the join after held bar {k}"


def test_held_wobble_bars_are_deterministic():
    a = _held(continuation=True)
    b = _held(continuation=True)
    assert a[0] == b[0]
    assert [x.tobytes() for x in a[2]] == [x.tobytes() for x in b[2]]


def test_plans_without_announce_fields_still_render():
    """Degrades gracefully against a BarPlan that predates the v1.4 fields."""
    @dataclass
    class OldPlan:
        bar: int
        state: str
        roles: dict = field(default_factory=lambda: dict(GENTLE))
        gain_db: float = 0.0
        density: float = 0.2
        register_ceiling_midi: float | None = None
        reason: str = "old"

    eng = DuetEngine(style="lofi", seed=3, sr=SR)
    out = eng.render_bar(OldPlan(0, "comping"), beat(), harmony())
    assert float(np.max(np.abs(out))) > 0.0
    assert eng.last_announce is None


# ---------------------------------------------------------------------------
# 6. the envelope survives the live planner's habits
# ---------------------------------------------------------------------------

def test_rerendering_the_same_bar_does_not_advance_the_fade():
    """The live planner re-renders a bar when the predicted downbeat moves.

    Without a rollback the envelope would advance once per *render* instead
    of once per *bar*, so a revised bar would jump ahead in the fade and the
    entrance would arrive early and stepped.  (Only the envelope is rolled
    back here — the voice-tail carry is not, see the note in the report.)
    """
    def phases(repeats):
        eng = DuetEngine(style="lofi", seed=5, sr=SR, fade_in_bars=3.0)
        h, hm = beat(), harmony()
        eng.render_bar(plan(0, "tacet", roles={}), h, hm)
        out = []
        for bar in (1, 2):
            for _ in range(repeats):
                eng.render_bar(plan(bar, "entering"), h, hm)
            out.append(round(eng._env_phase, 9))
        return out

    assert phases(1) == phases(3), \
        "a revision of the same bar advanced the envelope a second time"


def test_rerendered_bar_keeps_the_same_envelope_gain():
    eng = DuetEngine(style="lofi", seed=5, sr=SR, fade_in_bars=3.0)
    h, hm = beat(), harmony()
    eng.render_bar(plan(0, "tacet", roles={}), h, hm)
    eng.render_bar(plan(1, "entering"), h, hm)
    once = eng.last_env_gain_rms
    eng.render_bar(plan(1, "entering"), h, hm)
    assert eng.last_env_gain_rms == pytest.approx(once, rel=1e-12)


def test_fade_length_is_musical_so_it_follows_the_tempo():
    """Two bars of fade is two bars at any tempo, not a fixed number of seconds."""
    plans = [plan(0, "tacet", roles={}), plan(1, "entering"),
             plan(2, "entering"), plan(3, "comping")]
    for bpm in (72.0, 132.0):
        env, edges = envelope(plans, bpm=bpm, fade_in_bars=2.0)
        end = [g for i, g in env if i >= edges[2]]
        mid = min(env, key=lambda kv: abs(kv[0] - (edges[0] + edges[2]) / 2))[1]
        assert end and end[0] == pytest.approx(1.0, abs=0.03), \
            f"fade did not complete in two bars at {bpm} bpm"
        assert mid < 0.85, f"fade completed too early at {bpm} bpm"


def test_interrupted_exit_reverses_without_a_step():
    """The human comes back mid-fade-out: resume from where the gain got to."""
    plans = [plan(0, "comping"), plan(1, "comping"),
             plan(2, "fading", roles={"pad": 0.5, "bass": 0.3}),
             plan(3, "comping"), plan(4, "comping")]
    _, _, bus, edges = render(plans, fade_out_bars=3.0, fade_in_bars=3.0)
    m = mono(bus)
    for k in (1, 2, 3):
        step, local = _join_step(m, int(edges[k]))
        assert step <= max(local * 1.5, 1e-6), \
            f"step at join {k} when the exit reversed"


def test_no_fade_running_is_bit_identical_to_the_raw_render():
    """The contract that keeps every pre-v1.4 test honest."""
    plans = [plan(b, "comping") for b in range(3)]
    _, _, with_fades, _ = render(plans)
    _, _, without, _ = render(plans, **_NO_FADES)
    assert with_fades.tobytes() == without.tobytes()
