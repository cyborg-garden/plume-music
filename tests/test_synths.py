"""The curated synths and pads (plume.synths): stereo, alias-suppressed, moving,
velocity-aware, loudness-matched, and cheap enough for the live planner."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plume import synths  # noqa: E402
from plume.styles_lib import list_styles, load_style  # noqa: E402
from plume.voices import (PRESETS, is_stereo, preset_info, render_note,  # noqa: E402
                          render_note_stereo)

SR = 48000
NEW = sorted(synths.RENDERERS)


def _st(name, f=261.63, dur=1.0, vel=0.7, seed=3, sr=SR, params=None):
    return render_note_stereo(name, sr=sr, dur_s=dur, freq_hz=f, velocity=vel,
                              rng=np.random.default_rng(seed),
                              params=params or {}).astype(np.float64)


def _centroid(x, a, b, sr=SR):
    m = x.mean(axis=1) if x.ndim == 2 else x
    seg = m[int(a * sr):int(b * sr)] * np.hanning(int(b * sr) - int(a * sr))
    s = np.abs(np.fft.rfft(seg)) ** 2
    fr = np.fft.rfftfreq(seg.size, 1 / sr)
    return float((fr * s).sum() / s.sum())


@pytest.mark.parametrize("name", NEW)
def test_renders_valid_stereo(name):
    y = _st(name)
    assert y.ndim == 2 and y.shape[1] == 2
    assert np.isfinite(y).all()
    assert np.abs(y).max() <= 1.0
    assert np.abs(y).max() > 0.01
    assert is_stereo(name)


@pytest.mark.parametrize("name", NEW)
def test_mono_fold_is_render_note(name):
    st = _st(name)
    mono = render_note(name, sr=SR, dur_s=1.0, freq_hz=261.63, velocity=0.7,
                       rng=np.random.default_rng(3), params={})
    assert np.allclose(st.mean(axis=1), mono, atol=1e-6)


@pytest.mark.parametrize("name", NEW)
def test_deterministic(name):
    assert np.array_equal(_st(name, seed=9), _st(name, seed=9))


def test_mono_voices_come_back_dual_mono():
    y = _st("epiano")
    assert np.array_equal(y[:, 0], y[:, 1])
    assert not is_stereo("epiano")


@pytest.mark.parametrize("name", [n for n in NEW if n != "synthbass"])
def test_has_stereo_width_but_stays_mono_safe(name):
    y = _st(name, dur=1.5)
    mid = 0.5 * (y[:, 0] + y[:, 1])
    side = 0.5 * (y[:, 0] - y[:, 1])
    ratio_db = 10 * np.log10((side ** 2).sum() / (mid ** 2).sum())
    assert -20.0 < ratio_db < -1.0, ratio_db


def test_bass_is_centred():
    y = _st("synthbass", f=55.0)
    assert np.array_equal(y[:, 0], y[:, 1])


# Noise layers are switched off: the aliasing check compares a 48 kHz render
# against a 4x-oversampled one, and noise is never sample-identical.
_QUIET = {"breathpad": {"breath": 0.0}, "organ": {"click": 0.0}}


@pytest.mark.parametrize("name", NEW)
def test_band_limited_at_the_top_of_the_range(name):
    f = 130.81 if name == "synthbass" else 1046.5
    p = _QUIET.get(name, {})
    a = _st(name, f=f, dur=0.5, params=p).mean(axis=1)
    b = resample_poly(_st(name, f=f, dur=0.5, sr=4 * SR, params=p).mean(axis=1), 1, 4)
    n = min(a.size, b.size)
    a, b = a[2400:n - 2400], b[2400:n - 2400]
    w = 4096
    fr = np.fft.rfftfreq(w, 1 / SR)
    g = f / 2.0 if name == "junopoly" else f        # its sub sits an octave down
    k = np.maximum(np.round(fr / g), 1.0)
    off = np.abs(fr - k * g) > np.maximum(40.0, 0.03 * k * g)
    ex = eb = 0.0
    for i in range(0, a.size - w, w // 2):
        A = np.abs(np.fft.rfft(a[i:i + w] * np.hanning(w))) ** 2
        B = np.abs(np.fft.rfft(b[i:i + w] * np.hanning(w))) ** 2
        ex += np.maximum(A - B, 0.0)[off].sum()
        eb += B.sum()
    alias_db = 10 * np.log10(ex / eb + 1e-12)
    assert alias_db < -40.0, alias_db


@pytest.mark.parametrize("name", ["warmpad", "strings", "breathpad", "junopoly"])
def test_held_pads_move(name):
    """A held chord should not be a frozen spectrum."""
    y = _st(name, dur=3.0)
    cs = [_centroid(y, a, a + 0.25) for a in np.arange(0.6, 2.7, 0.25)]
    assert np.std(cs) / np.mean(cs) > 0.01, cs


@pytest.mark.parametrize("name", ["warmpad", "glasskeys", "junopoly", "synthlead",
                                  "synthbass", "tapekeys", "bell", "organ"])
def test_velocity_brightens(name):
    f = 55.0 if name == "synthbass" else 261.63
    soft = _st(name, f=f, vel=0.25)
    hard = _st(name, f=f, vel=1.0)
    assert _centroid(hard, 0.02, 0.2) > 1.03 * _centroid(soft, 0.02, 0.2)


def test_presets_are_listed_with_descriptions():
    info = preset_info()
    for name in NEW:
        assert name in PRESETS and info[name]["description"]
        assert set(info[name]["params"]) == set(synths.PRESETS[name])


def test_pads_hold_then_release():
    y = _st("warmpad", dur=1.0)
    rms = lambda a, b: float(np.sqrt((y[int(a * SR):int(b * SR)] ** 2).mean()))  # noqa: E731
    assert rms(0.7, 0.95) > 4 * rms(y.shape[0] / SR - 0.2, y.shape[0] / SR)


@pytest.mark.parametrize("name", sorted(list_styles()))
def test_style_variants_are_real_presets(name):
    s = load_style(name)
    for role, options in (s.get("voice_variants") or {}).items():
        assert role in s["voices"], (name, role)
        assert options, (name, role)
        for preset in options:
            assert preset in PRESETS, (name, role, preset)


def test_a_held_pad_bar_renders_well_inside_real_time():
    """The planner renders a bar ahead of time.  A four-note held chord on the
    heaviest pad must take a small fraction of the bar it fills, so slow
    machines keep up (best of three, to ride out a busy CPU)."""
    spb = 60.0 / 85.0
    best = 1e9
    for r in range(3):
        rng = np.random.default_rng(r)
        t0 = time.perf_counter()
        for m in (48, 52, 55, 59):
            render_note("warmpad", sr=SR, dur_s=3.92 * spb,
                        freq_hz=440.0 * 2 ** ((m - 69) / 12), velocity=0.5,
                        rng=rng, params={})
        best = min(best, time.perf_counter() - t0)
    assert best < 0.25 * 4 * spb, best


@pytest.mark.parametrize("name", NEW)
@pytest.mark.parametrize("dur", [0.08, 1.0])
def test_a_note_ends_at_silence_not_with_a_click(name, dur):
    """The buffer ends while the release is still audible; cutting it there
    is a tick at the end of every note.  The last sample must be silent and
    the last 2 ms nearly so."""
    y = _st(name, dur=dur, vel=0.9)
    peak = np.abs(y).max()
    assert np.abs(y[-1]).max() < 1e-4 * peak, name
    tail = np.abs(y[-int(0.002 * SR):]).max()
    assert tail < 0.01 * peak, (name, tail / peak)


@pytest.mark.parametrize("name", NEW)
def test_darker_and_brighter_move_every_new_voice(name):
    """The live tone cues run on the bus (plume.fx.BusTone), after the voices.
    Each new voice must still carry a top end they can take off and add."""
    from plume import fx
    f = 55.0 if name == "synthbass" else 261.63
    notes = [_st(name, f=f * 2 ** (k / 12), dur=0.45, seed=k) for k in (0, 4, 7, 12, 7, 4)]
    y = np.concatenate([np.pad(x, ((0, max(0, SR // 2 - x.shape[0])), (0, 0)))[: SR // 2]
                        for x in notes]).astype(np.float32)
    bars = [y[i:i + SR] for i in range(0, y.shape[0] - SR + 1, SR)]

    def top(x):
        m = np.asarray(x, dtype=np.float64).mean(axis=1)
        s = np.abs(np.fft.rfft(m)) ** 2
        return s[np.fft.rfftfreq(m.size, 1 / SR) >= 4000.0].sum()
    plain = top(np.concatenate(bars[1:]))
    for tilt, check in ((-1.0, lambda r: r < 0.5), (1.0, lambda r: r > 1.5)):
        tone = fx.BusTone(SR)
        out = np.concatenate([tone.process(b, tilt=tilt, wet=0.0) for b in bars][1:])
        assert plain > 0.0 and check(top(out) / plain), (name, tilt, top(out) / plain)


@pytest.mark.parametrize("name", NEW)
def test_section_energy_brightens_every_new_voice_offline(name):
    """Offline compose dims a tonal voice in low-energy sections through its
    brightness params (engine._energy_params).  The FM keys, the bell and the
    organ have no cutoff, so they need their own brightness scaled too."""
    from types import SimpleNamespace

    from plume.engine import _energy_params
    tr = SimpleNamespace(voice=name, kind="tonal", params={})
    f = 55.0 if name == "synthbass" else 261.63

    def upper(energy, k):            # share of the note above k x f0
        y = _st(name, f=f, dur=1.0, params=_energy_params(tr, energy))
        m = y.mean(axis=1)[: int(0.6 * SR)]
        s = np.abs(np.fft.rfft(m)) ** 2
        return s[np.fft.rfftfreq(m.size, 1 / SR) > k * f].sum() / s.sum()
    gain = max(upper(1.0, k) / upper(0.0, k) for k in (2.5, 6.0))
    assert gain > 1.1, (name, gain)


#: Each converted style's voices before plume.synths, to price the new ones.
_OLD_PALETTE = {
    "lofi": {"pad": "epiano", "lead": "pluck"},
    "synthwave": {"bass": "808sub", "pad": "pad", "arp": "pluck", "lead": "lead"},
    "midwest": {"pad": "airpad", "shimmer": "airpad"},
    "rnb": {"pad": "airpad"},
    "house": {"pad": "housestab", "lead": "pluck"},
    "hiphop": {"pad": "piano"},
}


def _bar_seconds(style, voices, bars, bpm=115.0):
    import copy

    from plume.duet import DuetEngine
    from plume.listen.beat import BeatState
    from plume.listen.harmony import HarmonyState
    from plume.mind import BarPlan
    eng = DuetEngine(style=style, seed=3)
    if voices is not None:
        eng._style = copy.deepcopy(eng._style)
        eng._style["voices"].update(voices)
        eng._style.pop("voice_variants", None)
    roles = {r: 1.0 for r in eng._style["voices"] if r not in ("sub", "tick", "foot")}
    iv = 60.0 / bpm
    t0 = time.perf_counter()
    for b in bars:
        plan = BarPlan(bar=b, state="comping", roles=roles, gain_db=0.0,
                       density=0.7, register_ceiling_midi=None, reason="cost")
        beat = BeatState(t_s=0.0, bpm=bpm, confidence=0.9, locked=True, bar=b,
                         beat_in_bar=0, phase_beats=0.0, next_beat_t_s=iv,
                         next_downbeat_t_s=4 * iv, beat_interval_s=iv)
        harm = HarmonyState(t_s=0.0, key="C major", key_conf=0.8,
                            chord=["C", "Am", "F", "G"][b % 4], chord_conf=0.9,
                            register_low_midi=None, register_high_midi=None)
        eng.render_bar(plan, beat, harm)
    return time.perf_counter() - t0


@pytest.mark.parametrize("style", sorted(_OLD_PALETTE))
def test_the_new_palette_costs_the_planner_about_what_the_old_one_did(style):
    """A live take at a 12x fake playhead fell behind by whole bars when these
    voices cost twice the old ones: the "darker" live test heard silence where
    the band should have been.  So the price is relative, not absolute: a
    style's bars, every voice variant included, must render within 1.4x the
    same bars on its old voices.  Interleaved and best of five, so a busy CPU
    slows both alike."""
    every = int(load_style(style).get("variant_bars", 16))
    bars = [k * every + 1 for k in range(3)]      # one bar of each variant
    new, old = 1e9, 1e9
    for _ in range(5):
        new = min(new, _bar_seconds(style, None, bars))
        old = min(old, _bar_seconds(style, _OLD_PALETTE[style], bars))
    assert new < 1.4 * old, (style, round(new / old, 2))


#: The role gains those old voices played at, where the style changed them.
_OLD_ROLE_DB = {
    "lofi": {"lead": -2.0},
    "synthwave": {"arp": -5.0, "lead": -2.0, "bass": -1.0},
    "house": {"pad": -1.0, "lead": -2.0},
}
_K1 = ([1.53512485958697, -2.69169618940638, 1.19839281085285],
       [1.0, -1.69065929318241, 0.73248077421585])
_K2 = ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])


def _role_lufs(style, role, preset, role_db=None, bars=8, bpm=100.0):
    """K-weighted loudness of one role alone, on ``preset``, in the style's
    own pattern for that role."""
    import copy

    from scipy.signal import lfilter

    from plume.duet import DuetEngine
    from plume.listen.beat import BeatState
    from plume.listen.harmony import HarmonyState
    from plume.mind import BarPlan
    eng = DuetEngine(style=style, seed=5)
    eng._style = copy.deepcopy(eng._style)
    eng._style["voices"][role] = preset
    eng._style.pop("voice_variants", None)
    if role_db is not None:
        eng._style["role_gain_db"][role] = role_db
        eng._style.pop("voice_gain_db", None)
    iv = 60.0 / bpm
    out = []
    for b in range(bars):
        plan = BarPlan(bar=b, state="comping", roles={role: 1.0}, gain_db=0.0,
                       density=0.7, register_ceiling_midi=None, reason="level")
        beat = BeatState(t_s=0.0, bpm=bpm, confidence=0.9, locked=True, bar=b,
                         beat_in_bar=0, phase_beats=0.0, next_beat_t_s=iv,
                         next_downbeat_t_s=4 * iv, beat_interval_s=iv)
        harm = HarmonyState(t_s=0.0, key="C major", key_conf=0.8,
                            chord=["C", "Am", "F", "G"][b % 4], chord_conf=0.9,
                            register_low_midi=None, register_high_midi=None)
        out.append(eng.render_bar(plan, beat, harm)[: int(4 * iv * SR)])
    z = lfilter(*_K2, lfilter(*_K1, np.concatenate(out), axis=0), axis=0)
    return -0.691 + 10.0 * np.log10(float(np.sum(np.mean(z ** 2, axis=0))) + 1e-20)


@pytest.mark.parametrize("style,role", [(s, r) for s in sorted(_OLD_PALETTE)
                                        for r in sorted(_OLD_PALETTE[s])])
def test_every_new_voice_and_variant_plays_at_the_old_voices_level(style, role):
    """A variant must not jump: before voice_gain_db, hiphop's tape keys came
    in 6.9 dB over the piano, and midwest's strings 4.8 dB under the pad they
    took over from.  Each voice a role can play, at its own gain, must sit
    within 1.5 dB of the old voice at the old gain."""
    s = load_style(style)
    old_db = _OLD_ROLE_DB.get(style, {}).get(role, s["role_gain_db"].get(role))
    ref = _role_lufs(style, role, _OLD_PALETTE[style][role], role_db=old_db)
    for preset in dict.fromkeys((s.get("voice_variants") or {}).get(role, [s["voices"][role]])):
        got = _role_lufs(style, role, preset)
        assert abs(got - ref) < 1.5, (style, role, preset, round(got - ref, 1))


#: The track gains those old voices rendered at offline (``plume render``):
#: the style's ``gain_db``, or compose's default for the role where the style
#: named none.
_OLD_OFFLINE_DB = {
    "lofi": {"pad": -7.0, "lead": -6.0},
    "synthwave": {"bass": -1.0, "pad": -7.0, "arp": -9.0, "lead": -6.0},
    "midwest": {"pad": -7.0, "shimmer": -6.0},
    "rnb": {"pad": -14.0},
    "house": {"pad": -5.0, "lead": -6.0},
}


#: Two different parts to measure a role on: (seed, key, bpm).
_OFFLINE_PARTS = ((7, "A minor", 100.0), (11, "C major", 120.0))

#: Where a new voice is held off the old level on purpose, by how much.
#: synthwave's bass sits 2.7 dB under the 808 it replaced (it was 4.1 under).
#: The 808 was a loud sine with no attack to speak of.  The ladder bass has a
#: pluck and overtones, and its track drive adds more the harder it is pushed.
#: Any closer to the old loudness and plume stops hearing its own render:
#: from a track gain of +2 dB the key comes back G minor for C minor
#: (tests/test_real_material.py, synth100), and from about +3 dB the tempo
#: goes to the dotted alias, 66.6 for 100 bpm (tests/test_acoustic.py,
#: synthwave100).  The gain is +1.
_OFFLINE_HELD_DB = {("synthwave", "bass"): -2.7}


def _offline_role_lufs(tmp_path, style, role, part, preset=None, gain_db=None):
    """K-weighted loudness of one role alone through compose and the offline
    engine: the composed pattern, the track's own gain and fx, the sections'
    energy.  The role is made to play through the middle eight bars, so a
    lead that waits for the peak is measured too.  The master is left clear
    and turned down, so the limiter never touches the level."""
    import soundfile as sf
    from scipy.signal import lfilter

    from plume import engine
    from plume.compose import PromptSpec, compose
    seed, key, bpm = part
    s = compose(PromptSpec(styles=[style], bpm=bpm, key=key, movements=1),
                seed=seed)
    tr = next(t for t in s.tracks if t.name == role)
    if preset is not None:
        tr.voice, tr.gain_db = preset, gain_db
    s.tracks = [tr]
    for sec in s.movements[0].sections:
        sec.plays = {role: "main"}
    s.master = {"gain_db": -12.0, "limit_db": 0.0}
    out = tmp_path / f"{seed}-{tr.voice}.wav"
    man = engine.render(s, out_path=str(out), bars=(12, 20))
    assert man["limiter_gain_reduction_db"] == 0.0
    x, _ = sf.read(str(out))
    z = lfilter(*_K2, lfilter(*_K1, x, axis=0), axis=0)
    return -0.691 + 10.0 * np.log10(float(np.sum(np.mean(z ** 2, axis=0))) + 1e-20)


@pytest.mark.parametrize("style,role", [(s, r) for s in sorted(_OLD_OFFLINE_DB)
                                        for r in sorted(_OLD_OFFLINE_DB[s])])
def test_every_new_voice_renders_offline_at_the_old_voices_level(tmp_path, style, role):
    """role_gain_db and voice_gain_db only reach the live band.  ``plume
    render`` takes a track's level from the style's ``gain_db``, and with that
    left alone the new voices came out at the old gains: synthwave's bass
    4.2 dB under the 808 it replaced, house's organ 4.5 dB over the stab.  So
    the promise holds offline too, in the style's own composed part, against
    the old voice at the old gain.  One gain cannot follow every melody (the
    bell against the pluck moves 3 dB from one part to the next), so it is
    held on the average of two parts, within 1 dB, with neither part more
    than 2.5 dB off.  ``_OFFLINE_HELD_DB`` names the one voice kept off the
    old level, and it is held to that offset the same way."""
    old = _OLD_PALETTE[style][role]
    assert load_style(style)["voices"][role] != old
    want = _OFFLINE_HELD_DB.get((style, role), 0.0)
    off = []
    for part in _OFFLINE_PARTS:
        ref = _offline_role_lufs(tmp_path, style, role, part, preset=old,
                                 gain_db=_OLD_OFFLINE_DB[style][role])
        off.append(_offline_role_lufs(tmp_path, style, role, part) - ref - want)
    shown = (style, role, want, [round(float(d), 1) for d in off])
    assert abs(float(np.mean(off))) < 1.0, shown
    assert max(abs(d) for d in off) < 2.5, shown


# ---------------------------------------------------------------------------
# in the band (DuetEngine)
# ---------------------------------------------------------------------------

def _duet_bar(style, bar, roles, bpm=100.0, section=None):
    from plume.duet import DuetEngine
    from plume.listen.beat import BeatState
    from plume.listen.harmony import HarmonyState
    from plume.mind import BarPlan
    iv = 60.0 / bpm
    eng = DuetEngine(style=style, seed=11)
    plan = BarPlan(bar=bar, state="comping", roles=roles, gain_db=0.0,
                   density=0.6, register_ceiling_midi=None, reason="test")
    beat = BeatState(t_s=0.0, bpm=bpm, confidence=0.9, locked=True, bar=bar,
                     beat_in_bar=0, phase_beats=0.0, next_beat_t_s=iv,
                     next_downbeat_t_s=4 * iv, beat_interval_s=iv)
    harm = HarmonyState(t_s=0.0, key="A minor", key_conf=0.8, chord="Am",
                        chord_conf=0.9, register_low_midi=None,
                        register_high_midi=None)
    out = eng.render_bar(plan, beat, harm, section=section)
    return out, eng.last_events


def test_a_stereo_pad_is_wide_in_the_band():
    out, events = _duet_bar("synthwave", 0, {"pad": 1.0})
    assert {e["preset"] for e in events} == {"warmpad"}
    side = out[:, 0] - out[:, 1]
    assert np.sqrt((side ** 2).mean()) > 0.05 * np.sqrt((out ** 2).mean())


def test_a_mono_voice_is_still_placed_by_the_pan_law():
    out, events = _duet_bar("lofi", 0, {"hat": 1.0})
    assert events
    assert not np.allclose(out[:, 0], out[:, 1])   # hats sit right of centre


def test_voice_variants_change_colour_between_sections():
    every = int(load_style("synthwave").get("variant_bars", 16))
    seen = [_duet_bar("synthwave", b * every, {"pad": 1.0})[1][0]["preset"]
            for b in range(3)]
    assert seen == ["warmpad", "warmpad", "strings"]


def test_with_a_form_running_variants_change_with_its_sections():
    """In lead mode the form's sections start wherever the band came in, so a
    bar count would change the pad's voice a bar or more inside a section.
    Given the section, the voice follows it: bar 40 of section 0 is still the
    first voice, bar 5 of section 2 is already the third."""
    opts = load_style("synthwave")["voice_variants"]["pad"]
    assert _duet_bar("synthwave", 40, {"pad": 1.0}, section=0)[1][0]["preset"] == opts[0]
    assert _duet_bar("synthwave", 5, {"pad": 1.0}, section=2)[1][0]["preset"] == opts[2]
    assert _duet_bar("synthwave", 5, {"pad": 1.0}, section=5)[1][0]["preset"] == opts[5 % 3]


def test_a_form_with_no_sections_falls_back_to_the_bar_count():
    """The form answers -1 for a session with nothing to play.  That is no
    section at all, so the voice still moves on the bar count instead of
    holding the first one for the whole take."""
    opts = load_style("synthwave")["voice_variants"]["pad"]
    assert opts[2] != opts[0]
    assert _duet_bar("synthwave", 40, {"pad": 1.0}, section=-1)[1][0]["preset"] == opts[2]
    assert _duet_bar("synthwave", 0, {"pad": 1.0}, section=-1)[1][0]["preset"] == opts[0]


def test_the_offline_lead_gives_the_band_the_forms_sections(tmp_path, monkeypatch):
    """The whole path, not the engine alone: a composed take whose sections are
    8 bars long.  A count of 16 bars would hold the first pad through bar 31;
    following the form, the third section (bars 16 to 23) is the third voice
    and the fourth is back on the first."""
    from plume import sim
    from plume.compose import PromptSpec, compose
    from plume.duet import DuetEngine
    from plume.mind import flatten_sections

    session = compose(PromptSpec(styles=["synthwave"], bpm=110.0, key="A minor",
                                 movements=1, mood=[], name="variants"), seed=7)
    assert [sec.bars for sec in flatten_sections(session)] == [8, 8, 8, 8]
    heard: dict[int, tuple] = {}

    class Listening(DuetEngine):
        def render_bar(self, plan, beat, harmony, **kw):
            out = super().render_bar(plan, beat, harmony, **kw)
            heard[int(plan.bar)] = (kw.get("section"), {
                e["preset"] for e in self.last_events if e["role"] == "pad"})
            return out

    monkeypatch.setattr(sim, "DuetEngine", Listening)
    res = sim.simulate_lead(None, session=session, seed=0,
                            out_path=str(tmp_path / "lead.wav"))
    opts = load_style("synthwave")["voice_variants"]["pad"]
    assert len(res.form_plans) == 32
    for fp in res.form_plans:
        section, presets = heard[fp["bar"]]
        assert section == fp["section_idx"] == fp["bar"] // 8
        assert presets <= {opts[section % len(opts)]}, (fp["bar"], presets)
    assert heard[16][1] == heard[23][1] == {"strings"}
    assert heard[8][1] == heard[24][1] == {"warmpad"}


@pytest.mark.parametrize("name", NEW)
def test_no_dc_offset(name):
    """1:1 FM makes a DC term; a voice must not carry it into the mix."""
    y = _st(name, dur=1.0)
    assert abs(float(y.mean())) < 0.01 * float(np.sqrt((y ** 2).mean()))
