"""Tests for plume.cli (SPEC section 11) + packaging surface.

Builders C and D run concurrently: the full make -> render -> analyze
round-trip is skipped until plume.{score,styles_lib,groove,compose,engine,
analyze} all exist. Everything else here runs against a hand-written session
file and modules that already exist (notation, voices).
"""
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

_FULL_MODULES = ("score", "styles_lib", "groove", "compose", "engine", "analyze")
HAVE_FULL = all(
    importlib.util.find_spec(f"plume.{m}") is not None for m in _FULL_MODULES
)
HAVE_STYLES = importlib.util.find_spec("plume.styles_lib") is not None


def run_cli(args, cwd, env_extra=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("PLUME_SESSION", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "plume", *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=300,
    )


def session_doc(name="demo"):
    return {
        "plume_version": 0,
        "name": name,
        "bpm": 100.0,
        "key": "F min",
        "swing": 0.5,
        "seed": 0,
        "style": "lofi",
        "timing_bias_ms": {},
        "humanize_ms": 4.0,
        "humanize_vel": 0.08,
        "tracks": [
            {"name": "kick", "voice": "808kick", "kind": "drum",
             "gain_db": 0.0, "pan": 0.0, "params": {}, "fx": {},
             "patterns": {"main": "bd ~ bd ~"}},
            {"name": "pad", "voice": "pad", "kind": "tonal",
             "gain_db": 0.0, "pan": 0.0, "params": {}, "fx": {"lpf_hz": 1800},
             "patterns": {}},
        ],
        "movements": [
            {"name": "m1", "sections": [
                {"name": "a", "bars": 8, "energy": 0.5,
                 "plays": {"kick": "main"}, "chords": ["Fm7"], "fill": True},
            ]},
        ],
        "master": {"reverb": 0.1, "drive": 0.0, "lofi": 0.0, "limit_db": -1.0},
    }


def write_session(dirpath, name="demo"):
    p = pathlib.Path(dirpath) / f"{name}.plume.json"
    p.write_text(json.dumps(session_doc(name), sort_keys=True, indent=2) + "\n")
    return p


def canonical_hash(doc):
    canon = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- packaging

def test_package_version_and_lazy_dir():
    import plume
    assert plume.__version__ == "0.1.0"
    for name in ("Session", "load", "save", "compose", "render", "analyze"):
        assert name in dir(plume)
    with pytest.raises(AttributeError):
        plume.nope_not_a_thing


def test_help_runs(tmp_path):
    r = run_cli(["--help"], cwd=tmp_path)
    assert r.returncode == 0
    for cmd in ("make", "new", "show", "set", "track", "pattern",
                "arrange", "render", "analyze", "play", "voices",
                "styles", "doctor"):
        assert cmd in r.stdout


def test_no_args_is_usage_error(tmp_path):
    r = run_cli([], cwd=tmp_path)
    assert r.returncode != 0


# ---------------------------------------------------------------- resolution

def test_no_session_errors_json(tmp_path):
    r = run_cli(["set", "bpm", "120", "--json"], cwd=tmp_path)
    assert r.returncode != 0
    payload = json.loads(r.stdout)
    assert "error" in payload


def test_multiple_sessions_error_lists_candidates(tmp_path):
    write_session(tmp_path, "one")
    write_session(tmp_path, "two")
    r = run_cli(["set", "bpm", "120", "--json"], cwd=tmp_path)
    assert r.returncode != 0
    err = json.loads(r.stdout)["error"]
    assert "one.plume.json" in err and "two.plume.json" in err


def test_single_session_in_cwd_resolves(tmp_path):
    p = write_session(tmp_path)
    r = run_cli(["set", "bpm", "120", "--json"], cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(p.read_text())
    assert doc["bpm"] == 120


def test_env_var_resolution(tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    p = write_session(other)
    work = tmp_path / "work"
    work.mkdir()
    r = run_cli(["set", "bpm", "90", "--json"], cwd=work,
                env_extra={"PLUME_SESSION": str(p)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(p.read_text())["bpm"] == 90


def test_explicit_session_beats_env(tmp_path):
    p = write_session(tmp_path)
    r = run_cli(["set", "bpm", "85", "--session", str(p), "--json"], cwd=tmp_path,
                env_extra={"PLUME_SESSION": str(tmp_path / "missing.plume.json")})
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(p.read_text())["bpm"] == 85


# ---------------------------------------------------------------- set / hash

def test_set_prints_state_hash_matching_file(tmp_path):
    p = write_session(tmp_path)
    r = run_cli(["set", "bpm", "120", "--json"], cwd=tmp_path)
    payload = json.loads(r.stdout)
    h = payload["hash"]
    assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)
    assert h == canonical_hash(json.loads(p.read_text()))


def test_hash_is_deterministic(tmp_path):
    write_session(tmp_path)
    h1 = json.loads(run_cli(["set", "bpm", "111", "--json"], cwd=tmp_path).stdout)["hash"]
    h2 = json.loads(run_cli(["set", "bpm", "111", "--json"], cwd=tmp_path).stdout)["hash"]
    assert h1 == h2
    h3 = json.loads(run_cli(["set", "bpm", "112", "--json"], cwd=tmp_path).stdout)["hash"]
    assert h3 != h1


def test_set_dotted_paths(tmp_path):
    p = write_session(tmp_path)
    assert run_cli(["set", "master.reverb", "0.3", "--json"], cwd=tmp_path).returncode == 0
    assert run_cli(["set", "tracks.pad.fx.lpf_hz", "900", "--json"], cwd=tmp_path).returncode == 0
    assert run_cli(["set", "tracks.kick.fx.drive", "0.4", "--json"], cwd=tmp_path).returncode == 0
    doc = json.loads(p.read_text())
    assert doc["master"]["reverb"] == 0.3
    pad = next(t for t in doc["tracks"] if t["name"] == "pad")
    kick = next(t for t in doc["tracks"] if t["name"] == "kick")
    assert pad["fx"]["lpf_hz"] == 900
    assert kick["fx"]["drive"] == 0.4


def test_set_string_value(tmp_path):
    p = write_session(tmp_path)
    r = run_cli(["set", "key", "A min", "--json"], cwd=tmp_path)
    assert r.returncode == 0
    assert json.loads(p.read_text())["key"] == "A min"


def test_set_unknown_top_key_errors(tmp_path):
    p = write_session(tmp_path)
    before = p.read_text()
    r = run_cli(["set", "bmp", "120", "--json"], cwd=tmp_path)
    assert r.returncode != 0
    assert "error" in json.loads(r.stdout)
    assert p.read_text() == before


def test_json_mode_stdout_is_pure_json(tmp_path):
    write_session(tmp_path)
    r = run_cli(["set", "swing", "0.57", "--json"], cwd=tmp_path)
    json.loads(r.stdout)  # whole stdout is exactly one JSON document


def test_set_human_mode_prints_hash(tmp_path):
    p = write_session(tmp_path)
    r = run_cli(["set", "bpm", "95"], cwd=tmp_path)
    assert r.returncode == 0
    assert canonical_hash(json.loads(p.read_text())) in r.stdout


# ---------------------------------------------------------------- pattern

def test_pattern_set_reports_step_count(tmp_path):
    p = write_session(tmp_path)
    r = run_cli(["pattern", "set", "kick", "alt", "bd ~ [bd bd] sd", "--json"],
                cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    payload = json.loads(r.stdout)
    assert payload["steps"] == 4  # rests omitted
    doc = json.loads(p.read_text())
    kick = next(t for t in doc["tracks"] if t["name"] == "kick")
    assert kick["patterns"]["alt"] == "bd ~ [bd bd] sd"


def test_pattern_set_invalid_rejected_file_unchanged(tmp_path):
    p = write_session(tmp_path)
    before = p.read_text()
    r = run_cli(["pattern", "set", "kick", "bad", "[bd", "--json"], cwd=tmp_path)
    assert r.returncode != 0
    assert "error" in json.loads(r.stdout)
    assert p.read_text() == before


def test_pattern_set_unknown_track(tmp_path):
    write_session(tmp_path)
    r = run_cli(["pattern", "set", "ghost", "x", "bd", "--json"], cwd=tmp_path)
    assert r.returncode != 0


def test_pattern_get(tmp_path):
    write_session(tmp_path)
    r = run_cli(["pattern", "get", "kick", "main", "--json"], cwd=tmp_path)
    assert json.loads(r.stdout)["pattern"] == "bd ~ bd ~"
    r = run_cli(["pattern", "get", "kick", "--json"], cwd=tmp_path)
    assert json.loads(r.stdout)["patterns"] == {"main": "bd ~ bd ~"}
    r = run_cli(["pattern", "get", "kick", "nope", "--json"], cwd=tmp_path)
    assert r.returncode != 0


# ---------------------------------------------------------------- track

def test_track_add_list_rm(tmp_path):
    p = write_session(tmp_path)
    r = run_cli(["track", "add", "lead1", "--voice", "lead", "--gain-db", "-3",
                 "--pan", "0.2", "--json"], cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    t = json.loads(r.stdout)["track"]
    assert t["kind"] == "tonal" and t["gain_db"] == -3.0 and t["pan"] == 0.2

    r = run_cli(["track", "list", "--json"], cwd=tmp_path)
    names = [t["name"] for t in json.loads(r.stdout)["tracks"]]
    assert names == ["kick", "pad", "lead1"]

    r = run_cli(["track", "rm", "lead1", "--json"], cwd=tmp_path)
    assert r.returncode == 0
    doc = json.loads(p.read_text())
    assert all(t["name"] != "lead1" for t in doc["tracks"])


def test_track_add_infers_drum_kind(tmp_path):
    write_session(tmp_path)
    r = run_cli(["track", "add", "hats", "--voice", "hat", "--json"], cwd=tmp_path)
    assert json.loads(r.stdout)["track"]["kind"] == "drum"


def test_track_add_rejects_unknown_voice_and_duplicate(tmp_path):
    write_session(tmp_path)
    r = run_cli(["track", "add", "x", "--voice", "kazoo", "--json"], cwd=tmp_path)
    assert r.returncode != 0
    r = run_cli(["track", "add", "kick", "--voice", "808kick", "--json"], cwd=tmp_path)
    assert r.returncode != 0


def test_track_rm_unknown_errors(tmp_path):
    write_session(tmp_path)
    r = run_cli(["track", "rm", "ghost", "--json"], cwd=tmp_path)
    assert r.returncode != 0


# ---------------------------------------------------------------- arrange / show

def test_arrange_show(tmp_path):
    write_session(tmp_path)
    r = run_cli(["arrange", "show", "--json"], cwd=tmp_path)
    assert r.returncode == 0
    movs = json.loads(r.stdout)["movements"]
    assert movs[0]["name"] == "m1" and movs[0]["sections"][0]["bars"] == 8


def test_arrange_set(tmp_path):
    p = write_session(tmp_path)
    spec = [{"name": "m1", "sections": [
        {"name": "intro", "bars": 8, "energy": 0.2, "plays": {"kick": "main"}},
        {"name": "drop", "bars": 16, "energy": 0.9, "plays": {"kick": "main"}},
    ]}]
    r = run_cli(["arrange", "set", "--spec-json", json.dumps(spec), "--json"],
                cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(p.read_text())
    secs = doc["movements"][0]["sections"]
    assert [s["bars"] for s in secs] == [8, 16]
    assert secs[0]["chords"] == [] and secs[0]["fill"] is True  # defaults filled


def test_arrange_set_invalid_spec(tmp_path):
    write_session(tmp_path)
    r = run_cli(["arrange", "set", "--spec-json",
                 '[{"name": "m1", "sections": [{"name": "a"}]}]', "--json"],
                cwd=tmp_path)
    assert r.returncode != 0


def test_show_summary(tmp_path):
    write_session(tmp_path)
    r = run_cli(["show", "--json"], cwd=tmp_path)
    assert r.returncode == 0
    s = json.loads(r.stdout)["summary"]
    assert s["name"] == "demo" and s["bpm"] == 100.0
    assert s["progression"] == ["Fm7"]
    assert [t["name"] for t in s["tracks"]] == ["kick", "pad"]
    r = run_cli(["show"], cwd=tmp_path)
    assert "demo" in r.stdout and "F min" in r.stdout


# ---------------------------------------------------------------- catalogs / doctor

def test_voices_catalog(tmp_path):
    r = run_cli(["voices", "--json"], cwd=tmp_path)
    assert r.returncode == 0
    info = json.loads(r.stdout)
    assert "808kick" in info and "pad" in info


@pytest.mark.skipif(not HAVE_STYLES, reason="styles_lib not present yet (builder C)")
def test_styles_catalog(tmp_path):
    r = run_cli(["styles", "--json"], cwd=tmp_path)
    assert r.returncode == 0
    names = json.loads(r.stdout)["styles"]
    for s in ("trap", "lofi", "synthwave"):
        assert s in names


def test_doctor(tmp_path):
    r = run_cli(["doctor", "--json"], cwd=tmp_path)
    payload = json.loads(r.stdout)
    checks = {c["name"]: c for c in payload["checks"]}
    for dep in ("numpy", "scipy", "soundfile"):
        assert checks[dep]["status"] == "ok"
    # sounddevice absence must be a warning, never an error
    assert checks["sounddevice"]["status"] in ("ok", "warn")
    assert checks["audio_device"]["status"] in ("ok", "warn")
    if HAVE_FULL:
        assert r.returncode == 0 and payload["ok"] is True


# ---------------------------------------------------------------- notation (v0.1 #1)

def test_notation_command(tmp_path):
    r = run_cli(["notation"], cwd=tmp_path)
    assert r.returncode == 0
    for needle in ("GRAMMAR", "one bar", "[hh*8]", "velocity", "<", "@"):
        assert needle in r.stdout
    rj = run_cli(["notation", "--json"], cwd=tmp_path)
    doc = json.loads(rj.stdout)
    assert set(doc) == {"grammar", "tokens", "examples"}
    assert doc["examples"] and all("pattern" in e and "description" in e
                                   for e in doc["examples"])


def test_help_points_at_notation(tmp_path):
    r = run_cli(["--help"], cwd=tmp_path)
    assert "plume notation" in r.stdout
    r = run_cli(["pattern", "set", "--help"], cwd=tmp_path)
    assert "mini-notation quick reference" in r.stdout
    assert "plume notation" in r.stdout


def test_subcommand_help_lists_choices(tmp_path):
    for cmd, subs in (("track", ("add", "rm", "list")),
                      ("pattern", ("set", "get")),
                      ("arrange", ("show", "set"))):
        r = run_cli([cmd, "--help"], cwd=tmp_path)
        assert r.returncode == 0
        for s in subs:
            assert s in r.stdout


# ---------------------------------------------------------------- set validation (v0.1 #2)

@pytest.mark.parametrize("args", [
    ["set", "master.bogus", "1"],
    ["set", "master.gain", "0.55"],
    ["set", "tracks.pad.fx.bogus", "1"],
    ["set", "tracks.pad.params.nope", "1"],
    ["set", "tracks.pad.wat", "1"],
    ["set", "master", "notadict"],
    ["set", "tracks.pad", "scalar"],
    ["set", "movements.0.sections.0.enrgy", "0.5"],
    ["set", "bpm.nested", "1"],
    ["set", "bpm", "fast"],
])
def test_set_rejects_bad_paths_file_unchanged(tmp_path, args):
    p = write_session(tmp_path)
    before = p.read_text()
    r = run_cli([*args, "--json"], cwd=tmp_path)
    assert r.returncode != 0, f"{args} unexpectedly succeeded"
    assert "error" in json.loads(r.stdout)
    assert p.read_text() == before


def test_set_error_lists_valid_keys_and_did_you_mean(tmp_path):
    write_session(tmp_path)
    err = json.loads(run_cli(["set", "master.gain", "0.5", "--json"],
                             cwd=tmp_path).stdout)["error"]
    assert "gain_db" in err and "reverb" in err          # valid keys listed
    assert "did you mean" in err
    err = json.loads(run_cli(["set", "tracks.pad.fx.lpf", "900", "--json"],
                             cwd=tmp_path).stdout)["error"]
    assert "lpf_hz" in err


def test_set_valid_nested_paths(tmp_path):
    p = write_session(tmp_path)
    assert run_cli(["set", "master.gain_db", "-3", "--json"],
                   cwd=tmp_path).returncode == 0
    assert run_cli(["set", "tracks.pad.params.cutoff_hz", "900", "--json"],
                   cwd=tmp_path).returncode == 0
    assert run_cli(["set", "movements.0.sections.0.energy", "0.7", "--json"],
                   cwd=tmp_path).returncode == 0
    assert run_cli(["set", "movements.0.sections.0.plays.pad", '""', "--json"],
                   cwd=tmp_path).returncode == 0
    doc = json.loads(p.read_text())
    assert doc["master"]["gain_db"] == -3
    pad = next(t for t in doc["tracks"] if t["name"] == "pad")
    assert pad["params"]["cutoff_hz"] == 900
    assert doc["movements"][0]["sections"][0]["energy"] == 0.7


def test_set_validates_key_and_pattern_values(tmp_path):
    write_session(tmp_path)
    r = run_cli(["set", "key", "H min", "--json"], cwd=tmp_path)
    assert r.returncode != 0
    assert "expected" in json.loads(r.stdout)["error"]
    r = run_cli(["set", "tracks.kick.patterns.alt", "[bd", "--json"], cwd=tmp_path)
    assert r.returncode != 0
    r = run_cli(["set", "tracks.kick.patterns.alt", "bd ~ bd bd", "--json"],
                cwd=tmp_path)
    assert r.returncode == 0
    # plays referencing a nonexistent pattern is rejected
    r = run_cli(["set", "movements.0.sections.0.plays.kick", "ghost", "--json"],
                cwd=tmp_path)
    assert r.returncode != 0


# ---------------------------------------------------------------- get / unset (v0.1 #3, #11)

def test_get_reads_values(tmp_path):
    write_session(tmp_path)
    assert json.loads(run_cli(["get", "bpm", "--json"], cwd=tmp_path).stdout)["value"] == 100.0
    assert json.loads(run_cli(["get", "master.reverb", "--json"],
                              cwd=tmp_path).stdout)["value"] == 0.1
    assert json.loads(run_cli(["get", "tracks.pad.fx.lpf_hz", "--json"],
                              cwd=tmp_path).stdout)["value"] == 1800
    # whole document when no path given
    doc = json.loads(run_cli(["get", "--json"], cwd=tmp_path).stdout)["value"]
    assert doc["master"]["reverb"] == 0.1
    # human mode prints plain JSON of the value
    r = run_cli(["get", "bpm"], cwd=tmp_path)
    assert r.stdout.strip() == "100.0"


def test_get_unknown_key_lists_siblings(tmp_path):
    write_session(tmp_path)
    r = run_cli(["get", "master.revrb", "--json"], cwd=tmp_path)
    assert r.returncode != 0
    err = json.loads(r.stdout)["error"]
    assert "reverb" in err


def test_get_list_enumerates_settable_paths(tmp_path):
    write_session(tmp_path)
    payload = json.loads(run_cli(["get", "--list", "--json"], cwd=tmp_path).stdout)
    paths = {row["path"] for row in payload["paths"]}
    for expect in ("bpm", "master.reverb", "master.gain_db",
                   "tracks.pad.fx.lpf_hz", "tracks.kick.patterns.main",
                   "movements.0.sections.0.energy"):
        assert expect in paths, f"missing {expect}"
    # prefix filter
    payload = json.loads(run_cli(["get", "master.", "--list", "--json"],
                                 cwd=tmp_path).stdout)
    assert all(row["path"].startswith("master.") for row in payload["paths"])


def test_show_json_includes_full_document(tmp_path):
    write_session(tmp_path)
    payload = json.loads(run_cli(["show", "--json"], cwd=tmp_path).stdout)
    assert payload["document"]["master"]["reverb"] == 0.1
    assert payload["document"]["tracks"][0]["patterns"]["main"] == "bd ~ bd ~"


def test_unset_removes_optional_leaves(tmp_path):
    p = write_session(tmp_path)
    assert run_cli(["unset", "tracks.pad.fx.lpf_hz", "--json"],
                   cwd=tmp_path).returncode == 0
    assert run_cli(["unset", "master.reverb", "--json"],
                   cwd=tmp_path).returncode == 0
    doc = json.loads(p.read_text())
    pad = next(t for t in doc["tracks"] if t["name"] == "pad")
    assert "lpf_hz" not in pad["fx"]
    assert "reverb" not in doc["master"]


@pytest.mark.parametrize("path", ["tracks", "movements", "bpm", "tracks.pad",
                                  "movements.0.sections.0.energy"])
def test_unset_refuses_structural_keys(tmp_path, path):
    p = write_session(tmp_path)
    before = p.read_text()
    r = run_cli(["unset", path, "--json"], cwd=tmp_path)
    assert r.returncode != 0
    assert p.read_text() == before


def test_unset_missing_key_errors(tmp_path):
    write_session(tmp_path)
    r = run_cli(["unset", "tracks.kick.fx.lpf_hz", "--json"], cwd=tmp_path)
    assert r.returncode != 0


# ---------------------------------------------------------------- session name shorthand

def test_session_flag_resolves_bare_name(tmp_path):
    write_session(tmp_path, "verify1")
    write_session(tmp_path, "other")  # two candidates: cwd resolution would fail
    r = run_cli(["show", "--session", "verify1", "--json"], cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout)["path"].endswith("verify1.plume.json")
    # explicit full path still wins
    r = run_cli(["show", "--session", "verify1.plume.json", "--json"], cwd=tmp_path)
    assert r.returncode == 0


def test_session_flag_missing_mentions_both_forms(tmp_path):
    r = run_cli(["show", "--session", "nope", "--json"], cwd=tmp_path)
    assert r.returncode != 0
    err = json.loads(r.stdout)["error"]
    assert "nope" in err and "nope.plume.json" in err


# ---------------------------------------------------------------- get defaults

def test_get_returns_schema_default_after_unset(tmp_path):
    write_session(tmp_path)
    assert run_cli(["unset", "master.limit_db"], cwd=tmp_path).returncode == 0
    r = run_cli(["get", "master.limit_db"], cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "(default)" in r.stdout and "-1.0" in r.stdout
    payload = json.loads(run_cli(["get", "master.limit_db", "--json"],
                                 cwd=tmp_path).stdout)
    assert payload == {"path": "master.limit_db", "value": -1.0, "default": True}


def test_get_default_for_absent_master_key(tmp_path):
    write_session(tmp_path)  # master has no gain_db; get --list shows it as null
    payload = json.loads(run_cli(["get", "master.gain_db", "--json"],
                                 cwd=tmp_path).stdout)
    assert payload == {"path": "master.gain_db", "value": 0.0, "default": True}


def test_get_default_unknown_keys_still_error(tmp_path):
    write_session(tmp_path)
    r = run_cli(["get", "master.bogus", "--json"], cwd=tmp_path)
    assert r.returncode != 0
    r = run_cli(["get", "tracks.nosuch.gain_db", "--json"], cwd=tmp_path)
    assert r.returncode != 0
    r = run_cli(["get", "tracks.kick.patterns.nosuch", "--json"], cwd=tmp_path)
    assert r.returncode != 0


def test_get_default_present_value_not_flagged(tmp_path):
    write_session(tmp_path)
    payload = json.loads(run_cli(["get", "master.reverb", "--json"],
                                 cwd=tmp_path).stdout)
    assert payload["value"] == 0.1
    assert "default" not in payload


# ---------------------------------------------------------------- arrange @file (v0.1 #7)

def test_arrange_set_from_file(tmp_path):
    p = write_session(tmp_path)
    spec = [{"name": "m1", "sections": [
        {"name": "solo", "bars": 8, "energy": 0.4, "plays": {"kick": "main"}}]}]
    spec_path = tmp_path / "arr.json"
    spec_path.write_text(json.dumps(spec))
    r = run_cli(["arrange", "set", "--spec-json", f"@{spec_path}", "--json"],
                cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(p.read_text())
    assert doc["movements"][0]["sections"][0]["name"] == "solo"
    r = run_cli(["arrange", "set", "--spec-json", "@/nope/missing.json", "--json"],
                cwd=tmp_path)
    assert r.returncode != 0


def test_arrange_show_roundtrips_into_set(tmp_path):
    write_session(tmp_path)
    shown = json.loads(run_cli(["arrange", "show", "--json"], cwd=tmp_path).stdout)
    r = run_cli(["arrange", "set", "--spec-json", json.dumps(shown), "--json"],
                cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    again = json.loads(run_cli(["arrange", "show", "--json"], cwd=tmp_path).stdout)
    assert again == shown


def test_arrange_set_help_documents_schema(tmp_path):
    r = run_cli(["arrange", "set", "--help"], cwd=tmp_path)
    assert "@FILE.json" in r.stdout
    assert "sections" in r.stdout and "energy" in r.stdout and "plays" in r.stdout


# ---------------------------------------------------------------- errors (v0.1 #6)

def test_corrupted_session_clean_error(tmp_path):
    p = tmp_path / "broken.plume.json"
    doc = session_doc("broken")
    doc["tracks"] = "garbage"                    # corrupt: not a list of dicts
    p.write_text(json.dumps(doc))
    r = run_cli(["render", "--json"], cwd=tmp_path)
    assert r.returncode != 0
    err = json.loads(r.stdout)["error"]
    assert "session file invalid" in err
    assert "broken.plume.json" in err
    assert "Traceback" not in r.stdout + r.stderr


# ---------------------------------------------------------------- unused tracks (v0.1 #10)

def test_unused_tracks_flagged(tmp_path):
    write_session(tmp_path)                      # pad is never played
    payload = json.loads(run_cli(["show", "--json"], cwd=tmp_path).stdout)
    flags = {t["name"]: t["unused"] for t in payload["summary"]["tracks"]}
    assert flags == {"kick": False, "pad": True}
    r = run_cli(["show"], cwd=tmp_path)
    assert "pad(pad) (unused)" in r.stdout
    assert "kick(808kick) (unused)" not in r.stdout


# ---------------------------------------------------------------- analyze hint (v0.1 #13)

def test_analyze_defaults_bpm_hint_from_session(tmp_path):
    import numpy as np
    import soundfile as sf
    write_session(tmp_path)                      # session bpm = 100
    wav = tmp_path / "noise.wav"
    rng = np.random.default_rng(0)
    sf.write(str(wav), 0.1 * rng.standard_normal(48000 * 2), 48000)
    payload = json.loads(run_cli(["analyze", str(wav), "--json"],
                                 cwd=tmp_path).stdout)
    assert payload["bpm_hint_used"] == 100.0
    assert payload["bpm_hint_source"] == "session"
    assert payload["bar_rms_db"]                 # hint enables per-bar loudness
    payload = json.loads(run_cli(["analyze", str(wav), "--bpm-hint", "60",
                                  "--json"], cwd=tmp_path).stdout)
    assert payload["bpm_hint_used"] == 60.0      # explicit beats session
    assert payload["bpm_hint_source"] == "explicit"


def test_analyze_without_session_has_no_hint(tmp_path):
    import numpy as np
    import soundfile as sf
    wav = tmp_path / "noise.wav"
    rng = np.random.default_rng(0)
    sf.write(str(wav), 0.1 * rng.standard_normal(48000), 48000)
    payload = json.loads(run_cli(["analyze", str(wav), "--json"],
                                 cwd=tmp_path).stdout)
    assert payload["bpm_hint_used"] is None
    assert payload["bar_rms_db"] == []


# ---------------------------------------------------------------- key forms (v0.1 #6)

@pytest.mark.skipif(not HAVE_FULL, reason="compose modules missing")
def test_new_accepts_short_key_forms(tmp_path):
    r = run_cli(["new", "shortkey", "--style", "lofi", "--key", "Fm",
                 "--seed", "1", "--json"], cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    summary = json.loads(r.stdout)["summary"]
    assert summary["key"] == "Fm"
    # minor-key harmony: the tonic chord is F-minor quality
    assert any(c.startswith("Fm") for c in summary["progression"]), summary


@pytest.mark.skipif(not HAVE_FULL, reason="compose modules missing")
def test_new_flat_key_progression_uses_flats(tmp_path):
    r = run_cli(["new", "flats", "--style", "lofi", "--key", "Fm",
                 "--seed", "1", "--json"], cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    prog = json.loads(r.stdout)["summary"]["progression"]
    assert prog
    assert not any("#" in sym for sym in prog), prog


@pytest.mark.skipif(not HAVE_FULL, reason="compose modules missing")
def test_new_session_flag_sets_output_path(tmp_path):
    outdir = tmp_path / "out"
    outdir.mkdir()
    target = outdir / "custom.plume.json"
    r = run_cli(["new", "routed", "--style", "lofi", "--seed", "1",
                 "--session", str(target), "--json"], cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout)["path"] == str(target)
    assert target.exists()
    assert not (tmp_path / "routed.plume.json").exists()


@pytest.mark.skipif(not HAVE_FULL, reason="compose modules missing")
def test_make_session_flag_sets_output_path(tmp_path):
    target = tmp_path / "made.plume.json"
    r = run_cli(["make", "chill lofi", "--seed", "1",
                 "--session", str(target), "--json"], cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout)["path"] == str(target)
    assert target.exists()


@pytest.mark.skipif(not HAVE_FULL, reason="compose modules missing")
def test_new_session_flag_missing_dir_errors(tmp_path):
    target = tmp_path / "missing" / "x.plume.json"
    r = run_cli(["new", "faildir", "--style", "lofi",
                 "--session", str(target), "--json"], cwd=tmp_path)
    assert r.returncode != 0
    assert "directory does not exist" in json.loads(r.stdout)["error"]
    assert not target.exists()


@pytest.mark.skipif(not HAVE_FULL, reason="compose modules missing")
def test_make_new_help_documents_session_flag(tmp_path):
    for cmd in ("make", "new"):
        r = run_cli([cmd, "--help"], cwd=tmp_path)
        assert r.returncode == 0
        assert "--session PATH writes the new session file" in r.stdout


@pytest.mark.skipif(not HAVE_FULL, reason="compose modules missing")
def test_new_bad_key_states_accepted_format(tmp_path):
    r = run_cli(["new", "badkey", "--style", "lofi", "--key", "Hmin", "--json"],
                cwd=tmp_path)
    assert r.returncode != 0
    err = json.loads(r.stdout)["error"]
    assert "expected" in err and "minor" in err


# ---------------------------------------------------------------- round trip

@pytest.mark.skipif(
    not HAVE_FULL,
    reason="builders C/D modules missing; round-trip runs at integration",
)
def test_make_render_analyze_roundtrip(tmp_path):
    r = run_cli(["make", "lofi at 80 bpm, 2 movements", "--name", "rt",
                 "--seed", "7", "--json"], cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    made = json.loads(r.stdout)
    spath = pathlib.Path(made["path"])
    assert spath.exists()
    assert len(made["hash"]) == 12
    assert made["summary"]["movements"], "composed session has movements"

    r = run_cli(["render", "--out", "rt.wav", "--session", str(spath), "--json"],
                cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    manifest = json.loads(r.stdout)
    wav = tmp_path / "rt.wav"
    assert wav.exists()
    assert manifest["duration_s"] > 0

    r = run_cli(["analyze", str(wav), "--json"], cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    info = json.loads(r.stdout)
    assert info["duration_s"] > 0
    assert info["sample_rate"] == 48000
