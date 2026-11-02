"""Tests for plume.analyze (SPEC sections 10, 12).

Renders the same hand-built fixture session as test_engine (no dependence
on builder C's compose) and checks the extracted features.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from plume.analyze import analyze  # noqa: E402
from plume.engine import render  # noqa: E402
from plume.score import Movement, SectionSpec, Session, Track  # noqa: E402

BPM = 120.0
BARS = 4
EXPECTED_DUR = BARS * 240.0 / BPM + 1.5  # 9.5 s


def make_session(seed: int = 5) -> Session:
    tracks = [
        Track(name="kick", voice="808kick", kind="drum",
              patterns={"main": "bd ~ ~ ~ bd ~ ~ ~ bd ~ ~ ~ bd ~ ~ ~"}),
        Track(name="hat", voice="hat", kind="drum", gain_db=-8.0,
              patterns={"main": "hh*8"}),
        Track(name="bass", voice="808sub", kind="tonal", gain_db=-3.0,
              patterns={"main": "c1 c1 ~ ~ eb1 ~ g1 ~"}),
        Track(name="pad", voice="pad", kind="tonal", gain_db=-6.0,
              fx={"duck": 0.5, "reverb": 0.3, "lpf_hz": 4000.0},
              patterns={"main": "<Cm7 Abmaj7 Eb Bb>"}),
    ]
    section = SectionSpec(
        name="a", bars=BARS, energy=0.8,
        plays={"kick": "main", "hat": "main", "bass": "main", "pad": "main"},
        chords=[], fill=False)
    return Session(
        name="ana-fixture", bpm=BPM, key="C min", swing=0.54, seed=seed,
        humanize_ms=2.0, humanize_vel=0.05,
        tracks=tracks, movements=[Movement(name="m1", sections=[section])],
        master={"reverb": 0.1, "drive": 0.1, "lofi": 0.0, "limit_db": -1.0})


@pytest.fixture(scope="module")
def wav(tmp_path_factory):
    d = tmp_path_factory.mktemp("analyze")
    path = d / "fixture.wav"
    render(make_session(), out_path=str(path))
    return str(path)


@pytest.fixture(scope="module")
def features(wav):
    return analyze(wav)


def test_tempo_estimate_close_to_session_bpm(features):
    assert abs(features["tempo_bpm_estimate"] - BPM) <= 3.0
    assert 0.0 <= features["tempo_confidence"] <= 1.0
    assert features["tempo_confidence"] > 0.05


def test_basic_shape_and_levels(features):
    assert features["sample_rate"] == 48000
    assert features["channels"] == 2
    assert abs(features["duration_s"] - EXPECTED_DUR) <= 0.05
    assert -60.0 < features["rms_db"] < 0.0
    assert features["rms_db"] <= features["peak_db"] < 0.0
    assert features["lufs_approx"] < 0.0
    assert features["clipped_sample_count"] == 0


def test_silence_and_width(features):
    assert 0.0 <= features["silence_ratio"] < 0.6
    assert 0.0 <= features["stereo_width"] <= 1.0


def test_spectral_features(features):
    assert 20.0 < features["spectral_centroid_hz_mean"] < 20000.0
    bands = features["band_energy_db"]
    assert set(bands) == {"sub", "low", "mid", "high", "air"}
    # 808 kick + sub bass: low end carries more energy than the air band
    assert bands["sub"] > bands["air"]
    assert bands["low"] > bands["air"]


def test_onsets(features):
    assert features["onset_count"] > 10
    expected_density = features["onset_count"] / features["duration_s"]
    assert abs(features["onset_density_per_s"] - expected_density) < 1e-9


def test_bar_rms_with_and_without_hint(wav, features):
    assert features["bar_rms_db"] == []          # no hint -> empty
    hinted = analyze(wav, bpm_hint=BPM)
    # 9.5 s at 2 s bars -> 4 full bars + the 1.5 s tail segment
    assert len(hinted["bar_rms_db"]) == 5
    assert all(isinstance(v, float) for v in hinted["bar_rms_db"])
    # the tail is quieter than the first full bar
    assert hinted["bar_rms_db"][-1] < hinted["bar_rms_db"][0]
    # v0.1: the hint biases the tempo estimate toward its metrical level
    assert abs(hinted["tempo_bpm_estimate"] - BPM) <= 4.0


def make_trap_session(seed: int = 3) -> Session:
    """Half-time trap at 140: snare on beat 3 only, 16th hats, sparse 808s."""
    tracks = [
        Track(name="kick", voice="808kick", kind="drum",
              patterns={"main": "bd ~ ~ ~ ~ ~ bd bd ~ ~ bd ~ ~ bd ~ ~"}),
        Track(name="snare", voice="snare", kind="drum", gain_db=-2.0,
              patterns={"main": "~ ~ ~ ~ ~ ~ ~ ~ sd ~ ~ ~ ~ ~ ~ ~"}),
        Track(name="hat", voice="hat", kind="drum", gain_db=-6.0,
              patterns={"main": ("hh hh:v0.55 hh:v0.75 hh:v0.55 " * 4).strip()}),
        Track(name="bass", voice="808sub", kind="tonal", gain_db=-3.0,
              patterns={"main": "f1@3 ~ ~ ~ f1@2 ~ ~ ~ ab1@2 ~ ~ ~ c1@2 ~ ~ ~"}),
    ]
    section = SectionSpec(name="a", bars=8, energy=0.8,
                          plays={t.name: "main" for t in tracks},
                          chords=[], fill=False)
    return Session(name="trap-fixture", bpm=140.0, key="F min", swing=0.5,
                   seed=seed, humanize_ms=2.0, humanize_vel=0.05,
                   tracks=tracks, movements=[Movement(name="m", sections=[section])],
                   master={"reverb": 0.05, "drive": 0.1, "lofi": 0.0,
                           "limit_db": -1.0})


@pytest.fixture(scope="module")
def trap_wav(tmp_path_factory):
    d = tmp_path_factory.mktemp("analyze-trap")
    path = d / "trap140.wav"
    render(make_trap_session(), out_path=str(path))
    return str(path)


def test_trap_halftime_hint_recovers_tempo(trap_wav):
    """v0.1 (#8): bpm_hint biases the estimate to the hinted metrical level."""
    hinted = analyze(trap_wav, bpm_hint=140.0)
    assert abs(hinted["tempo_bpm_estimate"] - 140.0) <= 4.0


def test_trap_halftime_candidates_contain_tempo_family(trap_wav):
    """v0.1 (#8): without a hint, 140 (or 70/280) appears in tempo_candidates."""
    plain = analyze(trap_wav)
    fam = (70.0, 140.0, 280.0)
    assert any(min(abs(c["bpm"] - m) for m in fam) <= 6.0
               for c in plain["tempo_candidates"]), plain["tempo_candidates"]


def test_tempo_candidates_present(features):
    cands = features["tempo_candidates"]
    assert cands and all(set(c) == {"bpm", "confidence"} for c in cands)
    assert all(c["bpm"] > 0 and 0.0 <= c["confidence"] <= 1.0 for c in cands)
    # candidates include an octave family around the leader
    lead = cands[0]["bpm"]
    rels = {round(lead / 2, 2), round(lead * 2, 2)}
    assert any(any(abs(c["bpm"] - r) / r < 0.05 for r in rels) for c in cands[1:])
    # the true tempo (or an octave relative of it) is in the list
    assert any(min(abs(c["bpm"] - m) for m in (BPM / 2, BPM, BPM * 2)) <= 5.0
               for c in cands)
