"""Tests for the v1 CLI additions (SPEC-V1 section 8): sim / jam / listen / mind.

Additive file — tests/test_cli.py (v0) is untouched.  Builders run
concurrently: anything needing plume.mind / plume.sim skips until those
modules exist; listen and the help/error contracts run now.  ``plume listen``
must work fully headless — several tests poison ``sounddevice`` in
sys.modules to prove the audio stack is never touched.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

HAVE_MIND = importlib.util.find_spec("plume.mind") is not None
HAVE_SIM = (HAVE_MIND
            and importlib.util.find_spec("plume.sim") is not None
            and importlib.util.find_spec("plume.duet") is not None)


def run_cli(args, cwd, env_extra=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("PLUME_SESSION", None)
    env.pop("PLUME_DEBUG", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "plume", *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=300,
    )


def run_main(argv, capsys, monkeypatch):
    """In-process CLI invocation -> (exit_code, stdout, stderr)."""
    monkeypatch.delenv("PLUME_DEBUG", raising=False)
    from plume import cli
    code = cli.main(argv)
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture(scope="module")
def steady_wav(tmp_path_factory):
    """A short deterministic 'human guitarist' WAV (8 steady bars)."""
    import soundfile as sf
    from fixtures_gen import SR, make_human

    audio = make_human("steady", bpm=110, bars=8, key="A minor", seed=1)
    path = tmp_path_factory.mktemp("listen") / "steady.wav"
    sf.write(str(path), audio, SR, subtype="PCM_24")
    return str(path)


# ---------------------------------------------------------------- help surface

def test_help_lists_v1_commands(tmp_path):
    r = run_cli(["--help"], cwd=tmp_path)
    assert r.returncode == 0
    for cmd in ("sim", "jam", "listen", "mind"):
        assert cmd in r.stdout


def test_mind_help_has_full_params_table(tmp_path):
    r = run_cli(["mind", "--help"], cwd=tmp_path)
    assert r.returncode == 0
    for name in ("min_listen_bars", "entry_conf", "entry_boundary_bars",
                 "enter_under_db", "ramp_bars", "density_high",
                 "density_hysteresis", "silence_thin_bars",
                 "silence_fade_bars", "silence_tacet_bars", "silence_rms_db",
                 "conf_floor", "register_margin_st", "energy_alpha"):
        assert name in r.stdout
    assert "--mind" in r.stdout or "K=V" in r.stdout


def test_sim_help_has_mind_table_and_flags(tmp_path):
    r = run_cli(["sim", "--help"], cwd=tmp_path)
    assert r.returncode == 0
    for name in ("min_listen_bars", "enter_under_db", "energy_alpha"):
        assert name in r.stdout          # full MindParams table in the epilog
    for flag in ("--style", "--out", "--decisions", "--seed", "--tightness",
                 "--mind", "--taps", "--mix-human-db", "--json"):
        assert flag in r.stdout


def test_jam_help_flags(tmp_path):
    r = run_cli(["jam", "--help"], cwd=tmp_path)
    assert r.returncode == 0
    for flag in ("--style", "--seed", "--tightness", "--mind", "--record",
                 "--device"):
        assert flag in r.stdout
    assert "plume mind" in r.stdout     # points at the parameter table


# ---------------------------------------------------------------- plume listen

def test_listen_streams_ndjson(steady_wav, capsys, monkeypatch):
    code, out, _ = run_main(["listen", steady_wav], capsys, monkeypatch)
    assert code == 0
    lines = [json.loads(ln) for ln in out.strip().splitlines()]
    states = [d for d in lines if "event" not in d]
    assert len(states) >= 20             # ~17.5s of audio at ~0.5s cadence
    for d in states:
        for k in ("t_s", "bpm", "conf", "locked", "bar", "beat_in_bar",
                  "phase_beats", "next_beat_t_s", "next_downbeat_t_s",
                  "key", "key_conf", "chord", "chord_conf"):
            assert k in d, f"missing {k}"
    ts = [d["t_s"] for d in states]
    assert ts == sorted(ts)
    # a steady strummer must lock, and locking must appear as an event too
    assert states[-1]["locked"] is True
    events = [d for d in lines if "event" in d]
    assert any(e["event"] == "lock" for e in events)
    # confidence reaches the lock threshold monotonically enough to lock
    assert states[-1]["conf"] > states[0]["conf"]


def test_listen_json_single_document(steady_wav, capsys, monkeypatch):
    code, out, _ = run_main(["listen", steady_wav, "--json"], capsys, monkeypatch)
    assert code == 0
    doc = json.loads(out)                # exactly one JSON document
    assert doc["path"] == steady_wav
    assert doc["sr"] == 48000
    assert doc["duration_s"] > 15.0
    assert isinstance(doc["states"], list) and doc["states"]
    assert isinstance(doc["events"], list)
    assert any(e["event"] == "lock" for e in doc["events"])


def test_listen_is_headless(steady_wav, capsys, monkeypatch):
    """listen must never import sounddevice (poisoned import would raise)."""
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    code, out, _ = run_main(["listen", steady_wav], capsys, monkeypatch)
    assert code == 0
    assert out.strip()


def test_listen_missing_file_error_contract(tmp_path, capsys, monkeypatch):
    code, out, _ = run_main(
        ["listen", str(tmp_path / "nope.wav"), "--json"], capsys, monkeypatch)
    assert code == 1
    doc = json.loads(out)
    assert "error" in doc and "not found" in doc["error"]


def test_listen_garbage_file_error(tmp_path, capsys, monkeypatch):
    bad = tmp_path / "junk.wav"
    bad.write_bytes(b"this is not audio")
    code, out, _ = run_main(["listen", str(bad), "--json"], capsys, monkeypatch)
    assert code == 1
    assert "error" in json.loads(out)


def test_listen_needs_no_session(steady_wav, tmp_path, capsys, monkeypatch):
    """listen must not attempt session resolution (empty cwd, no env)."""
    monkeypatch.chdir(tmp_path)
    code, _, _ = run_main(["listen", steady_wav], capsys, monkeypatch)
    assert code == 0


# ---------------------------------------------------------------- plume jam

def test_jam_without_sounddevice_is_clean(capsys, monkeypatch):
    """The no-device path fails fast and clean — before mind/duet import."""
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    code, out, err = run_main(["jam"], capsys, monkeypatch)
    assert code == 1
    assert "error:" in err
    assert "sounddevice" in err
    assert "Traceback" not in err


def test_jam_without_sounddevice_json(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    code, out, _ = run_main(["jam", "--json"], capsys, monkeypatch)
    assert code == 1
    doc = json.loads(out)
    assert "sounddevice" in doc["error"]


def test_jam_no_devices_is_clean(capsys, monkeypatch):
    import types
    fake = types.SimpleNamespace(query_devices=lambda: [])
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    code, _, err = run_main(["jam"], capsys, monkeypatch)
    assert code == 1
    assert "no audio" in err


# ---------------------------------------------------------------- plume mind

@pytest.mark.skipif(not HAVE_MIND, reason="plume.mind not built yet")
class TestMindCommand:
    def test_lists_every_field_with_defaults(self, capsys, monkeypatch):
        import dataclasses

        import plume.mind as mind_mod
        code, out, _ = run_main(["mind", "--json"], capsys, monkeypatch)
        assert code == 0
        doc = json.loads(out)
        rows = {r["name"]: r for r in doc["params"]}
        defaults = mind_mod.MindParams()
        fields = dataclasses.fields(mind_mod.MindParams)
        assert set(rows) == {f.name for f in fields}
        for f in fields:
            assert rows[f.name]["default"] == getattr(defaults, f.name)
            assert rows[f.name]["meaning"]  # every field documented

    def test_human_output_is_a_table(self, capsys, monkeypatch):
        code, out, _ = run_main(["mind"], capsys, monkeypatch)
        assert code == 0
        assert "min_listen_bars" in out
        assert "--mind" in out


# ------------------------------------------------------- --mind / --taps plumbing

def test_sim_bad_taps_errors_before_pipeline(steady_wav, capsys, monkeypatch):
    code, out, _ = run_main(
        ["sim", steady_wav, "--taps", "0.5,banana", "--json"],
        capsys, monkeypatch)
    assert code == 1
    assert "--taps" in json.loads(out)["error"]


def test_sim_decreasing_taps_error(steady_wav, capsys, monkeypatch):
    code, out, _ = run_main(
        ["sim", steady_wav, "--taps", "2.0,1.0", "--json"], capsys, monkeypatch)
    assert code == 1
    assert "increasing" in json.loads(out)["error"]


def test_sim_missing_input_error(tmp_path, capsys, monkeypatch):
    code, out, _ = run_main(
        ["sim", str(tmp_path / "ghost.wav"), "--json"], capsys, monkeypatch)
    assert code == 1
    assert "not found" in json.loads(out)["error"]


@pytest.mark.skipif(not HAVE_MIND, reason="plume.mind not built yet")
class TestMindOverrides:
    def test_unknown_key_lists_valid_names(self, steady_wav, capsys, monkeypatch):
        code, out, _ = run_main(
            ["sim", steady_wav, "--mind", "nope=1", "--json"],
            capsys, monkeypatch)
        assert code == 1
        err = json.loads(out)["error"]
        assert "nope" in err
        assert "min_listen_bars" in err      # valid names listed

    def test_typo_gets_a_suggestion(self, steady_wav, capsys, monkeypatch):
        code, out, _ = run_main(
            ["sim", steady_wav, "--mind", "entry_covf=0.5", "--json"],
            capsys, monkeypatch)
        assert code == 1
        assert "entry_conf" in json.loads(out)["error"]

    def test_missing_equals_errors(self, steady_wav, capsys, monkeypatch):
        code, out, _ = run_main(
            ["sim", steady_wav, "--mind", "entry_conf", "--json"],
            capsys, monkeypatch)
        assert code == 1
        assert "K=V" in json.loads(out)["error"]

    def test_int_field_rejects_fraction(self, steady_wav, capsys, monkeypatch):
        code, out, _ = run_main(
            ["sim", steady_wav, "--mind", "min_listen_bars=2.5", "--json"],
            capsys, monkeypatch)
        assert code == 1
        assert "integer" in json.loads(out)["error"]

    def test_parse_produces_typed_params(self):
        from plume.cli import _parse_mind_overrides
        p = _parse_mind_overrides(
            ["min_listen_bars=2", "enter_under_db=-10", "entry_conf=0.7"])
        assert p.min_listen_bars == 2 and isinstance(p.min_listen_bars, int)
        assert p.enter_under_db == -10.0
        assert p.entry_conf == 0.7
        assert p.silence_tacet_bars == 10     # untouched fields keep defaults


# ------------------------------------------------------- sim (integration)

@pytest.mark.skipif(not HAVE_SIM, reason="plume.sim/duet not built yet")
class TestSimEndToEnd:
    def test_sim_renders_and_reports(self, steady_wav, tmp_path, capsys,
                                     monkeypatch):
        out_wav = tmp_path / "duet.wav"
        code, out, _ = run_main(
            ["sim", steady_wav, "--out", str(out_wav), "--seed", "3", "--json"],
            capsys, monkeypatch)
        assert code == 0
        doc = json.loads(out)
        assert pathlib.Path(doc["out_path"]).exists()
        assert pathlib.Path(doc["decisions_path"]).exists()
        for k in ("duration_s", "entered_bar", "states_by_bar", "bpm_trace",
                  "final_key", "manifest"):
            assert k in doc
        with open(doc["decisions_path"], encoding="utf-8") as f:
            for line in f:
                json.loads(line)             # NDJSON: every line parses

    def test_sim_honors_decisions_path(self, steady_wav, tmp_path, capsys,
                                       monkeypatch):
        out_wav = tmp_path / "d2.wav"
        dec = tmp_path / "custom" / "d.ndjson"
        dec.parent.mkdir()
        code, out, _ = run_main(
            ["sim", steady_wav, "--out", str(out_wav),
             "--decisions", str(dec), "--json"],
            capsys, monkeypatch)
        assert code == 0
        assert json.loads(out)["decisions_path"] == str(dec)
        assert dec.exists()
