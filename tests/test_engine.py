"""Tests for plume.score + plume.engine (SPEC sections 3, 9, 12).

The fixture session is hand-constructed (not via compose) so these tests
do not depend on builder C's modules.
"""
import json
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from plume.engine import NoteEvent, compile_events, render  # noqa: E402
from plume.score import (  # noqa: E402
    Movement,
    SectionSpec,
    Session,
    Track,
    load,
    save,
    total_bars,
)

BPM = 120.0
BAR_S = 240.0 / BPM  # 2.0 s


def make_session(seed: int = 42, bars: int = 4) -> Session:
    tracks = [
        Track(name="kick", voice="808kick", kind="drum",
              patterns={"main": "bd ~ ~ ~ bd ~ ~ ~ bd ~ ~ ~ bd ~ ~ ~"}),
        Track(name="hat", voice="hat", kind="drum", gain_db=-8.0,
              patterns={"main": "hh*8", "fill": "hh*16"}),
        Track(name="bass", voice="808sub", kind="tonal", gain_db=-3.0,
              patterns={"main": "c1 c1 ~ ~ eb1 ~ g1 ~"}),
        Track(name="pad", voice="pad", kind="tonal", gain_db=-6.0,
              fx={"duck": 0.5, "reverb": 0.3, "lpf_hz": 4000.0},
              patterns={"main": "<Cm7 Abmaj7 Eb Bb>"}),
    ]
    section = SectionSpec(
        name="a", bars=bars, energy=0.8,
        plays={"kick": "main", "hat": "main", "bass": "main", "pad": "main"},
        chords=[], fill=True)
    return Session(
        name="fixture", bpm=BPM, key="C min", swing=0.54, seed=seed,
        timing_bias_ms={"snare": 2.0}, humanize_ms=2.0, humanize_vel=0.05,
        tracks=tracks, movements=[Movement(name="m1", sections=[section])],
        master={"reverb": 0.1, "drive": 0.1, "lofi": 0.0, "limit_db": -1.0})


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    d = tmp_path_factory.mktemp("engine")
    session = make_session()
    manifest = render(session, out_path=str(d / "take.wav"),
                      stems_dir=str(d / "stems"))
    return d, manifest


# ---------------------------------------------------------------------------
# score.py
# ---------------------------------------------------------------------------

def test_score_save_load_roundtrip(tmp_path):
    s = make_session()
    p = tmp_path / "fixture.plume.json"
    save(s, p)
    doc = json.loads(p.read_text())
    assert doc["plume_version"] == 0
    s2 = load(p)
    assert s2 == s


def test_score_load_tolerant_of_missing_optionals(tmp_path):
    doc = {
        "name": "mini", "bpm": 100, "key": "F min",
        "tracks": [{"name": "k", "voice": "808kick"}],
        "movements": [{"name": "m", "sections": [
            {"name": "a", "bars": 8, "energy": 0.5, "plays": {"k": "main"}}]}],
    }
    p = tmp_path / "mini.json"
    p.write_text(json.dumps(doc))
    s = load(p)
    assert s.swing == 0.5 and s.seed == 0 and s.humanize_ms == 4.0
    assert s.tracks[0].kind == "drum" and s.tracks[0].gain_db == 0.0
    assert s.movements[0].sections[0].fill is True
    assert s.master == {}


def test_total_bars():
    assert total_bars(make_session(bars=4)) == 4
    assert total_bars(make_session(bars=2)) == 2


# ---------------------------------------------------------------------------
# compile_events
# ---------------------------------------------------------------------------

def test_compile_event_counts_and_roles():
    events = compile_events(make_session())
    by_track = {}
    for ev in events:
        by_track.setdefault(ev.track, []).append(ev)
    assert len(by_track["kick"]) == 16          # 4-on-floor x 4 bars
    # hat: 8 per bar x 3 bars + 16-hit fill in the last bar
    assert len(by_track["hat"]) == 40
    assert len(by_track["bass"]) == 16          # 4 notes per bar
    # pad chords via <>: Cm7(4) + Abmaj7(4) + Eb(3) + Bb(3)
    assert len(by_track["pad"]) == 14
    assert {ev.role for ev in by_track["kick"]} == {"kick"}
    assert {ev.role for ev in by_track["bass"]} == {"bass"}
    assert all(isinstance(ev, NoteEvent) for ev in events)
    assert events == sorted(events, key=lambda e: e.t_s)


def test_compile_kick_tuned_to_tonic():
    events = compile_events(make_session())
    kick_hz = {ev.freq_hz for ev in events if ev.role == "kick"}
    assert len(kick_hz) == 1
    # C tonic in the low octave: C1 = 32.70 Hz (nearest to the 35-55 band)
    assert abs(kick_hz.pop() - 32.70) < 0.1
    hat_hz = {ev.freq_hz for ev in events if ev.track == "hat"}
    assert hat_hz == {None}


def test_compile_bass_glide_on_adjacent_steps():
    events = [ev for ev in compile_events(make_session()) if ev.track == "bass"]
    events.sort(key=lambda e: e.t_s)
    per_bar = 4
    for bar in range(4):
        first, second = events[bar * per_bar], events[bar * per_bar + 1]
        assert first.glide_from_hz is None or bar > 0  # bar starts may chain
        # steps 1+2 are adjacent eighths -> legato glide on the second
        assert second.glide_from_hz is not None
        assert abs(second.glide_from_hz - first.freq_hz) < 1e-6
        # step 3 (eb1 after a rest) must not glide
        third = events[bar * per_bar + 2]
        assert third.glide_from_hz is None


def test_compile_velocities_and_timing():
    events = compile_events(make_session())
    assert all(0.05 <= ev.velocity <= 1.0 for ev in events)
    kicks = sorted(ev.t_s for ev in events if ev.role == "kick")
    for t in kicks:
        nominal = round(t / (BAR_S / 4.0)) * (BAR_S / 4.0)
        assert abs(t - nominal) < 0.03          # only humanize jitter (2 ms SD)
    assert all(ev.t_s > -0.03 for ev in events)
    assert max(ev.t_s for ev in events) < 4 * BAR_S


def test_compile_determinism():
    a = compile_events(make_session(seed=9))
    b = compile_events(make_session(seed=9))
    assert a == b
    c = compile_events(make_session(seed=10))
    assert a != c


def test_compile_missing_pattern_raises():
    s = make_session()
    s.movements[0].sections[0].plays["bass"] = "nope"
    with pytest.raises(ValueError, match="nope"):
        compile_events(s)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def test_render_wav_and_manifest(rendered):
    d, manifest = rendered
    out = d / "take.wav"
    assert out.exists()
    expected = 4 * BAR_S + 1.5
    assert abs(manifest["duration_s"] - expected) <= 0.05
    data, sr = sf.read(str(out), always_2d=True)
    assert sr == 48000
    assert data.shape[1] == 2
    assert abs(data.shape[0] / sr - expected) <= 0.05
    # non-silent, no clipping (limiter ceiling is -1 dB)
    assert float(np.sqrt(np.mean(data ** 2))) > 1e-4
    assert float(np.max(np.abs(data))) < 0.999
    assert manifest["bars"] == 4
    assert manifest["events"] == 86             # 16+40+16+14
    assert manifest["peak_db"] < 0.0
    assert manifest["rms_db"] < manifest["peak_db"]
    assert set(manifest["per_track_rms_db"]) == {"kick", "hat", "bass", "pad"}
    assert all(v < 0.0 for v in manifest["per_track_rms_db"].values())


def test_render_stems(rendered):
    d, _ = rendered
    for name in ("kick", "hat", "bass", "pad"):
        stem = d / "stems" / f"{name}.wav"
        assert stem.exists()
        data, sr = sf.read(str(stem), always_2d=True)
        assert sr == 48000 and data.shape[1] == 2
        assert float(np.max(np.abs(data))) > 1e-4   # every track played


def test_render_manifest_reports_limiter_gr(rendered):
    """v0.1 (#4): the manifest exposes how hard the limiter worked."""
    _, manifest = rendered
    assert "limiter_gain_reduction_db" in manifest
    assert manifest["limiter_gain_reduction_db"] >= 0.0


def test_master_gain_db_is_a_real_gain_stage(tmp_path):
    """v0.1 (#4): no hidden normalization — gain changes move the levels."""
    def rms_at(gain_db, name):
        s = make_session(seed=7, bars=2)
        s.master = {"gain_db": gain_db, "reverb": 0.0, "drive": 0.0,
                    "lofi": 0.0, "limit_db": -1.0}
        return render(s, out_path=str(tmp_path / name))

    lo = rms_at(-18.0, "lo.wav")
    hi = rms_at(-12.0, "hi.wav")
    # both far below the limiter knee -> exactly linear
    assert lo["limiter_gain_reduction_db"] == 0.0
    assert hi["limiter_gain_reduction_db"] == 0.0
    assert abs((hi["rms_db"] - lo["rms_db"]) - 6.0) < 0.1
    assert abs((hi["peak_db"] - lo["peak_db"]) - 6.0) < 0.1
    # a hot mix engages the limiter but still never clips
    hot = rms_at(+12.0, "hot.wav")
    assert hot["limiter_gain_reduction_db"] > 0.0
    assert hot["peak_db"] < 0.0
    data, _ = sf.read(str(tmp_path / "hot.wav"), always_2d=True)
    assert float(np.max(np.abs(data))) < 0.999


def test_stems_have_headroom(rendered):
    """v0.1 (#12): stems are written with -3 dBFS headroom."""
    d, _ = rendered
    ceiling = 10.0 ** (-3.0 / 20.0)
    for name in ("kick", "hat", "bass", "pad"):
        data, _ = sf.read(str(d / "stems" / f"{name}.wav"), always_2d=True)
        assert float(np.max(np.abs(data))) <= ceiling + 1e-3


def test_compile_section_energy_gain():
    """v0.1 (#14): section energy maps to a -3..0 dB pre-master gain."""
    lo = make_session(seed=7, bars=2)
    lo.movements[0].sections[0].energy = 0.0
    hi = make_session(seed=7, bars=2)
    hi.movements[0].sections[0].energy = 1.0
    lo_gains = {ev.gain for ev in compile_events(lo)}
    hi_gains = {ev.gain for ev in compile_events(hi)}
    assert len(lo_gains) == 1 and len(hi_gains) == 1
    assert abs(lo_gains.pop() - 10.0 ** (-3.0 / 20.0)) < 1e-9
    assert abs(hi_gains.pop() - 1.0) < 1e-9


def test_render_same_seed_identical_bytes(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    render(make_session(seed=7, bars=2), out_path=str(a))
    render(make_session(seed=7, bars=2), out_path=str(b))
    assert a.read_bytes() == b.read_bytes()


def test_render_different_seed_different_bytes(tmp_path):
    a = tmp_path / "a.wav"
    c = tmp_path / "c.wav"
    render(make_session(seed=7, bars=2), out_path=str(a))
    render(make_session(seed=8, bars=2), out_path=str(c))
    assert a.read_bytes() != c.read_bytes()


def test_render_movement_and_bars_window(tmp_path):
    out = tmp_path / "win.wav"
    manifest = render(make_session(), out_path=str(out),
                      movement=0, bars=(1, 3))
    assert manifest["bars"] == 2
    assert abs(manifest["duration_s"] - (2 * BAR_S + 1.5)) <= 0.05
    info = sf.info(str(out))
    assert abs(info.frames / info.samplerate - (2 * BAR_S + 1.5)) <= 0.05


def test_render_bad_windows_raise(tmp_path):
    s = make_session()
    with pytest.raises(ValueError):
        render(s, out_path=str(tmp_path / "x.wav"), movement=5)
    with pytest.raises(ValueError):
        render(s, out_path=str(tmp_path / "x.wav"), bars=(3, 3))
    with pytest.raises(ValueError):
        render(s, out_path=str(tmp_path / "x.wav"), bars=(0, 99))
