"""Tests for the v2 CLI + live transport (SPEC-V2 section 5).

Additive file — ``tests/test_cli.py`` (v0) and ``tests/test_cli_v1.py`` (v1)
are untouched, and several tests here exist purely to assert that they still
describe reality: ``plume jam`` without the new flags must call exactly the v1
signature, and every v0/v1 subcommand must still be on the parser with its
flags.

WHAT IS ACTUALLY EXERCISED HERE.  v1's lesson was that ten green tests over
synthetic fixtures proved nothing, because the fixtures were continuously
filled sustained plucks and real percussive audio is not.  A CLI is mostly
argument plumbing, so most of this file is contract tests — but the parts
that touch audio (the live lead planner's per-bar decision chain) are run
against plume's OWN v0 renders at two gains, in a loop where the machine's
grid advances at the tempo the negotiator just chose.  That harness is
``_lead_planner_pass``; it is the offline stand-in for ``realtime.lead``'s
planner thread, running the identical calls in the identical order.

Builders run concurrently: anything needing ``plume.sim.simulate_lead``
(builder D) skips until it lands, and the CLI's wiring to it is tested
separately with a recording stub so the argument contract is covered either
way.  A test that can silently not run is not a test — so the stub test is
NOT skipped, ever.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from plume import cli                                           # noqa: E402
from plume import realtime                                      # noqa: E402
from plume import roles as roles_mod                            # noqa: E402
from plume.compose import PromptSpec, compose                   # noqa: E402
from plume.engine import render                                 # noqa: E402

SR = 48000

HAVE_SIMLEAD = (importlib.util.find_spec("plume.sim") is not None
                and hasattr(__import__("plume.sim", fromlist=["x"]),
                            "simulate_lead"))


# ------------------------------------------------------------------ helpers

def run_main(argv, capsys, monkeypatch):
    """In-process CLI invocation -> (exit_code, stdout, stderr)."""
    monkeypatch.delenv("PLUME_DEBUG", raising=False)
    code = cli.main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def sock_path(name: str) -> str:
    """A short-enough AF_UNIX path.

    pytest's ``tmp_path`` routinely exceeds the kernel's ~104-byte sun_path
    limit on macOS, and the resulting OSError looks nothing like "your path is
    too long", so sockets live directly in the temp dir under a short name.
    """
    p = Path(tempfile.gettempdir()) / f"plume-t{os.getpid()}-{name}.sock"
    try:
        p.unlink()
    except OSError:
        pass
    return str(p)


class _FakeSd:
    """Minimal sounddevice stand-in: enough for _resolve_device only."""

    def __init__(self, devices):
        self._devices = list(devices)

    def query_devices(self):
        return list(self._devices)


_DUPLEX = [{"name": "Built-in", "max_input_channels": 2,
            "max_output_channels": 2}]


class _FakeStreamObj:
    """A sounddevice.Stream stand-in that pumps the callback off a real WAV.

    The 'mic' is real material and the machine's output is captured, so the
    live transport can be exercised with no audio hardware.  ``rate_x`` runs
    the pump faster than real time; the playhead is what the planner paces
    against, so this makes the test proportionally quicker without changing a
    single decision.
    """

    def __init__(self, human, blocksize, callback, rate_x=10.0):
        self._human = np.asarray(human, dtype=np.float32)
        self._n = int(blocksize)
        self._cb = callback
        self._rate = float(rate_x)
        self._stop = threading.Event()
        self._thread = None
        self.captured: list[np.ndarray] = []
        self.blocks = 0

    def _pump(self):
        pos = 0
        period = self._n / float(SR) / self._rate
        t_next = time.perf_counter()
        # A hard cap: a wedged planner must fail the test, not hang the suite.
        max_blocks = int(2.5 * (self._human.shape[0] / self._n) + 4000)
        while not self._stop.is_set() and self.blocks < max_blocks:
            indata = np.zeros((self._n, 1), dtype=np.float32)
            take = self._human[pos:pos + self._n]
            if take.size:
                indata[:take.size, 0] = take
            pos += self._n
            outdata = np.zeros((self._n, 2), dtype=np.float32)
            self._cb(indata, outdata, self._n, None, None)
            self.captured.append(outdata.copy())
            self.blocks += 1
            t_next += period
            slack = t_next - time.perf_counter()
            if slack > 0:
                time.sleep(slack)

    def __enter__(self):
        self._thread = threading.Thread(
            target=self._pump, name="fake-stream", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5.0)
        return False


class _FakeSdStream(_FakeSd):
    """A whole fake ``sounddevice`` module: devices plus a pumping Stream."""

    def __init__(self, devices, human, rate_x=10.0):
        super().__init__(devices)
        self._human = human
        self._rate = rate_x
        self.stream = None

    def Stream(self, *, blocksize, callback, **kw):  # noqa: N802 - sd's name
        self.stream = _FakeStreamObj(self._human, blocksize, callback,
                                     rate_x=self._rate)
        return self.stream


class _patched_module:
    """``sys.modules`` swap that works outside a monkeypatch fixture."""

    def __init__(self, name, mod):
        self._name, self._mod = name, mod
        self._old = None
        self._had = False

    def __enter__(self):
        self._had = self._name in sys.modules
        self._old = sys.modules.get(self._name)
        sys.modules[self._name] = self._mod
        return self._mod

    def __exit__(self, *exc):
        if self._had:
            sys.modules[self._name] = self._old
        else:
            sys.modules.pop(self._name, None)
        return False


# ------------------------------------------------------------------ material

def _render_wav(path: Path, style: str, bpm: float, key: str, *,
                bars=(0, 16), seed: int = 7, gain_db: float = 0.0) -> Path:
    """One of plume's OWN v0 renders — real material, not a fixture."""
    session = compose(
        PromptSpec(styles=[style], bpm=bpm, key=key, movements=1, mood=[],
                   name=path.stem),
        seed=seed)
    render(session, out_path=str(path), movement=0, bars=bars)
    if gain_db:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        sf.write(str(path), x * (10.0 ** (gain_db / 20.0)), sr,
                 subtype="PCM_24")
    return path


@pytest.fixture(scope="module")
def material(tmp_path_factory):
    """Real v0 renders at two gains, plus the sessions that made them."""
    d = tmp_path_factory.mktemp("cli_v2_material")
    out = {}
    for name, style, bpm, key in (("trap140", "trap", 140.0, "D minor"),
                                  ("lofi80", "lofi", 80.0, "C major")):
        sess = compose(PromptSpec(styles=[style], bpm=bpm, key=key,
                                  movements=1, mood=[], name=name), seed=7)
        loud = _render_wav(d / f"{name}.wav", style, bpm, key)
        quiet = _render_wav(d / f"{name}_q.wav", style, bpm, key, gain_db=-20.0)
        out[name] = {"session": sess, "loud": loud, "quiet": quiet,
                     "bpm": bpm, "key": key, "style": style}
    return out


@pytest.fixture(scope="module")
def trap_pass(material):
    """One full planner pass over the trap render, shared by the suite."""
    m = material["trap140"]
    return _lead_planner_pass(m["session"], m["loud"])


def _lead_planner_pass(session, human_wav, *, role_params=None, seed=0,
                       max_bars=24):
    """Offline twin of ``realtime.lead``'s planner thread.

    Runs the SAME per-bar chain in the SAME order — bar summary ->
    RoleNegotiator.update -> Mind.plan_form -> lead_bar_plan ->
    DuetEngine.render_bar — driven by a MachineClock whose next bar starts at
    the tempo the negotiator just chose.  No audio device, no threads, so it
    is deterministic and can assert on real material.

    Returns ``(decisions, machine_bars, audio_s, cost)`` where ``cost`` is
    ``{"decide_s", "render_s"}`` — the two are reported separately because
    only the first is NEW on the live path.  ``DuetEngine.render_bar`` is v1
    code that ``jam`` already pays per bar; what v2 adds is the decision
    chain, and that is the number that must not eat the headroom.
    """
    from plume.duet import DuetEngine
    from plume.listen.beat import BeatTracker
    from plume.listen.features import FeatureExtractor
    from plume.listen.harmony import HarmonyTracker
    from plume.mind import BarFeatures, Mind, MindParams, flatten_sections

    x, sr = sf.read(str(human_wav), dtype="float32", always_2d=True)
    mono = x.mean(axis=1).astype(np.float32)
    assert sr == SR

    params = MindParams()
    rp = role_params if role_params is not None else roles_mod.RoleParams(
        role="lead")
    composed_bpm = float(session.bpm)
    composed_key = session.key
    extractor = FeatureExtractor(sr=SR)
    tracker = BeatTracker()
    harmony = HarmonyTracker()
    mind = Mind(params)
    neg = roles_mod.RoleNegotiator(rp, composed_bpm=composed_bpm,
                                  composed_key=composed_key)
    duet = DuetEngine(style=session.style or "lofi",
                      key_fallback=str(composed_key or "A minor"), seed=seed,
                      sr=SR)
    flat = flatten_sections(session)

    clock = realtime.MachineClock(composed_bpm)
    bpm = composed_bpm
    events: list[tuple[float, float, bool]] = []
    onsets: list[float] = []
    beat_state = harm_state = None
    ref_db = None
    decisions = []
    bars = []
    decide_wall = 0.0
    render_wall = 0.0

    block = 4096
    pos = 0
    while pos < mono.shape[0] and len(bars) < max_bars:
        chunk = mono[pos:pos + block]
        pos += block
        frames = extractor.process(chunk)
        beat_state = tracker.process(frames)
        harm_state = harmony.process(frames)
        for f in frames:
            events.append((f.t_s, f.rms_db, f.onset))
            if f.onset:
                onsets.append(f.t_s)
        now = pos / float(SR)
        # Render bar N once the stream has reached its start (the live loop
        # renders one bar AHEAD; offline "ahead" is just "as soon as due").
        while clock.t_s <= now and len(bars) < max_bars:
            t0 = time.perf_counter()
            bar, t_bar, _ = clock.commit(bpm)
            bar_len = clock.bar_len_s(bpm)
            evs = [e for e in events if t_bar - bar_len <= e[0] < t_bar]
            events = [e for e in events if e[0] >= t_bar]
            ons = [t for t in onsets if t_bar - bar_len <= t < t_bar]
            onsets = [t for t in onsets if t >= t_bar]
            human = None
            if bar > 0 and evs:
                human = BarFeatures(**realtime.bar_features_dict(
                    bar - 1, evs, silence_rms_db=params.silence_rms_db,
                    ref_db=ref_db,
                    register_low_midi=(harm_state.register_low_midi
                                       if harm_state else None),
                    register_high_midi=(harm_state.register_high_midi
                                        if harm_state else None)))
                if not human.silent:
                    ref_db = (human.rms_db if ref_db is None
                              else 0.5 * human.rms_db + 0.5 * ref_db)
            offset_ms = None
            if human is not None and not human.silent:
                offset_ms = roles_mod.onset_phase_offset_ms(
                    ons, t_bar - bar_len, t_bar, bar_len / 16.0)
            mbeat = realtime.machine_beat_state(bar, t_bar, bpm)
            hb = beat_state if beat_state is not None else mbeat
            hh = (harm_state if harm_state is not None
                  else realtime.machine_harmony_state(t_bar, None, None))
            rs = neg.update(bar=bar, beat=hb, harmony=hh, human=human,
                            machine_grid_t_s=t_bar, human_offset_ms=offset_ms)
            form = mind.plan_form(rs, mbeat, hh, human, session, None)
            sec = realtime._section_at(flat, form.section_idx)
            spec = None
            if sec is not None:
                spec = session.movements[sec.movement].sections[sec.section]
            chords = list(getattr(spec, "chords", None) or [])
            mh = realtime.machine_harmony_state(
                t_bar, composed_key,
                chords[bar % len(chords)] if chords else None,
                transpose=rs.transpose_semitones)
            plan = realtime.lead_bar_plan(bar, spec, duet.roles, rs, form,
                                          human=human)
            t1 = time.perf_counter()
            decide_wall += t1 - t0
            audio = duet.render_bar(plan, mbeat, mh)
            render_wall += time.perf_counter() - t1
            bars.append(audio)
            decisions.append({
                "bar": bar, "t_s": t_bar, "bpm": bpm, "role": rs.role,
                "clock_authority": rs.clock_authority,
                "form_authority": rs.form_authority,
                "tempo_offset_pct": rs.tempo_offset_pct,
                "transpose_semitones": rs.transpose_semitones,
                "role_events": list(rs.events), "role_reason": rs.reason,
                "action": form.action, "section_idx": form.section_idx,
                "announce": bool(form.announce), "density": plan.density,
                "gain_db": plan.gain_db, "roles": dict(plan.roles),
                "state": plan.state, "peak": float(np.abs(audio).max()),
            })
            bpm = roles_mod.effective_bpm(composed_bpm, rs.tempo_target_pct)
    audio_s = sum(b.shape[0] for b in bars) / float(SR)
    return (decisions, bars, audio_s,
            {"decide_s": decide_wall, "render_s": render_wall})


# ==================================================================== verbs

class TestVerbsExist:
    """Every v2 verb is on the parser, and every v0/v1 verb still is."""

    def test_top_level_help_lists_v2_verbs(self, capsys, monkeypatch):
        with pytest.raises(SystemExit) as ei:
            run_main(["--help"], capsys, monkeypatch)
        assert ei.value.code == 0
        out = capsys.readouterr().out
        for verb in ("lead", "simlead", "cue", "roles"):
            assert verb in out

    def test_v0_v1_verbs_are_all_still_there(self):
        parser = cli._build_parser()
        sub = next(a for a in parser._actions
                   if isinstance(a, __import__("argparse")._SubParsersAction))
        for verb in ("make", "new", "show", "set", "get", "unset", "notation",
                     "track", "pattern", "arrange", "render", "analyze",
                     "play", "voices", "styles", "doctor", "sim", "jam",
                     "listen", "mind"):
            assert verb in sub.choices, f"v0/v1 verb {verb} disappeared"

    @pytest.mark.parametrize("verb", ["lead", "simlead", "cue", "roles"])
    def test_verb_help_works(self, verb, capsys, monkeypatch):
        with pytest.raises(SystemExit) as ei:
            run_main([verb, "--help"], capsys, monkeypatch)
        assert ei.value.code == 0
        assert capsys.readouterr().out.strip()

    @pytest.mark.parametrize("verb", ["lead", "simlead", "cue", "roles"])
    def test_verb_accepts_json_flag(self, verb):
        """--json is on every v2 verb (the --json discipline is universal)."""
        parser = cli._build_parser()
        sub = next(a for a in parser._actions
                   if isinstance(a, __import__("argparse")._SubParsersAction))
        flags = {o for act in sub.choices[verb]._actions
                 for o in act.option_strings}
        assert "--json" in flags


# ==================================================================== roles

class TestRolesCommand:
    def test_documents_every_roleparams_field(self, capsys, monkeypatch):
        code, out, _ = run_main(["roles"], capsys, monkeypatch)
        assert code == 0
        for f in dataclasses.fields(roles_mod.RoleParams):
            assert f.name in out, f"'plume roles' does not document {f.name}"

    def test_json_documents_every_field_with_a_meaning(self, capsys,
                                                       monkeypatch):
        code, out, _ = run_main(["roles", "--json"], capsys, monkeypatch)
        assert code == 0
        doc = json.loads(out)
        names = [r["name"] for r in doc["params"]]
        assert names == [f.name for f in dataclasses.fields(
            roles_mod.RoleParams)]
        for r in doc["params"]:
            assert r["meaning"].strip(), f"{r['name']} has no meaning text"
            assert r["type"]

    def test_json_is_exactly_one_document(self, capsys, monkeypatch):
        code, out, err = run_main(["roles", "--json"], capsys, monkeypatch)
        assert code == 0
        assert err == ""
        json.loads(out)                       # one document, nothing else
        assert out.count("\n{") == 0

    def test_defaults_match_the_dataclass(self, capsys, monkeypatch):
        """The drift guard: v1 shipped a hand-written MindParams table that
        disagreed with the code.  Every printed default must BE the default."""
        code, out, _ = run_main(["roles", "--json"], capsys, monkeypatch)
        assert code == 0
        doc = json.loads(out)
        defaults = roles_mod.RoleParams()
        for r in doc["params"]:
            assert r["default"] == getattr(defaults, r["name"]), r["name"]

    def test_role_defaults_table_is_present(self, capsys, monkeypatch):
        code, out, _ = run_main(["roles", "--json"], capsys, monkeypatch)
        doc = json.loads(out)
        assert set(doc["roles"]) == set(roles_mod.ROLES)
        for name, pair in doc["roles"].items():
            ca, fa = roles_mod.ROLE_DEFAULTS[name]
            assert pair == {"clock_authority": ca, "form_authority": fa}

    def test_help_epilog_carries_the_derived_table(self, capsys, monkeypatch):
        with pytest.raises(SystemExit):
            run_main(["roles", "--help"], capsys, monkeypatch)
        out = capsys.readouterr().out
        for f in dataclasses.fields(roles_mod.RoleParams):
            assert f.name in out
        assert "--roles" in out

    @pytest.mark.parametrize("verb", ["simlead", "lead", "jam"])
    def test_roles_table_reaches_the_verbs_that_take_it(self, verb, capsys,
                                                        monkeypatch):
        with pytest.raises(SystemExit):
            run_main([verb, "--help"], capsys, monkeypatch)
        out = capsys.readouterr().out
        assert "--roles" in out and "--role" in out
        if verb == "simlead":                 # the full table is in its epilog
            for name in ("concede_ms", "trade_bars", "conform_semitones"):
                assert name in out
        else:                                 # ...or a pointer to it
            assert "plume roles" in out

    def test_table_survives_a_broken_roles_module(self, monkeypatch):
        """`--help` must not die while a sibling module is half-written."""
        monkeypatch.setattr(cli, "_role_rows",
                            lambda: (_ for _ in ()).throw(RuntimeError("nope")))
        text = cli._roles_table()
        assert "plume roles" in text and "--roles" in text


# ============================================================ --roles k=v

class TestRolesOverrides:
    def test_unknown_key_has_did_you_mean(self, capsys, monkeypatch):
        code, out, err = run_main(["simlead", "--prompt", "x",
                                   "--roles", "concede_m=5"], capsys,
                                  monkeypatch)
        assert code == 1
        assert "unknown RoleParams field" in err
        assert "did you mean 'concede_ms'" in err
        assert "Traceback" not in err

    def test_unknown_key_json(self, capsys, monkeypatch):
        code, out, _ = run_main(["simlead", "--prompt", "x", "--json",
                                 "--roles", "trade_bar=8"], capsys, monkeypatch)
        assert code == 1
        doc = json.loads(out)
        assert "trade_bar" in doc["error"] and "trade_bars" in doc["error"]

    def test_unknown_role_has_did_you_mean(self, capsys, monkeypatch):
        code, _, err = run_main(["simlead", "--prompt", "x", "--role", "leed"],
                                capsys, monkeypatch)
        assert code == 1
        assert "unknown role 'leed'" in err and "did you mean 'lead'" in err

    def test_every_valid_key_is_accepted(self):
        """Every field the table advertises can actually be set."""
        defaults = roles_mod.RoleParams()
        for f in dataclasses.fields(roles_mod.RoleParams):
            cur = getattr(defaults, f.name)
            if f.name == "role":
                v = "trade"
            elif f.name in ("clock_authority", "form_authority"):
                v = "0.5"
            elif f.name == "tempo_policy":
                # str-typed with a None default (None = the role's own default,
                # same shape as the two authorities above), so neither the
                # float() fallback nor the bool/int branches apply.
                v = "meet"
            elif isinstance(cur, bool):
                v = "false"
            elif isinstance(cur, int):
                v = str(max(1, int(cur)))
            else:
                v = str(float(cur))
            p = cli._parse_role_overrides([f"{f.name}={v}"])
            assert p is not None

    def test_missing_equals_is_a_clean_error(self):
        with pytest.raises(cli.CliError) as ei:
            cli._parse_role_overrides(["concede_ms"])
        assert "K=V" in str(ei.value)

    def test_int_field_rejects_a_fraction(self):
        with pytest.raises(cli.CliError) as ei:
            cli._parse_role_overrides(["concede_bars=1.5"])
        assert "integer" in str(ei.value)

    def test_float_field_rejects_words(self):
        with pytest.raises(cli.CliError) as ei:
            cli._parse_role_overrides(["concede_ms=fast"])
        assert "number" in str(ei.value)

    def test_bool_field_takes_true_false(self):
        assert cli._parse_role_overrides(
            ["conform_enabled=false"]).conform_enabled is False
        assert cli._parse_role_overrides(
            ["conform_enabled=1"]).conform_enabled is True
        with pytest.raises(cli.CliError):
            cli._parse_role_overrides(["conform_enabled=maybe"])

    def test_optional_authority_accepts_none(self):
        p = cli._parse_role_overrides(["clock_authority=none"])
        assert p.clock_authority is None
        p = cli._parse_role_overrides(["clock_authority=0.25"])
        assert p.clock_authority == pytest.approx(0.25)

    def test_out_of_range_authority_is_a_clean_error(self):
        with pytest.raises(cli.CliError) as ei:
            cli._parse_role_overrides(["clock_authority=1.5"])
        assert "[0, 1]" in str(ei.value)

    def test_dataclass_validation_surfaces_cleanly(self):
        """RoleParams.__post_init__ raises ValueError; the CLI must not."""
        for bad in ("concede_bars=0", "trade_bars=0", "conform_semitones=9",
                    "handoff_density=0", "handoff_density_ratio=0.5"):
            with pytest.raises(cli.CliError):
                cli._parse_role_overrides([bad])

    def test_roles_role_beats_the_role_flag(self):
        p = cli._parse_role_overrides(["role=trade"], "lead")
        assert p.role == "trade"

    def test_role_flag_used_when_table_is_silent(self):
        assert cli._parse_role_overrides(["concede_ms=10"], "lead").role == "lead"
        assert cli._parse_role_overrides(None, "solo_support").role == "solo_support"

    def test_no_flags_is_the_dataclass_default(self):
        assert cli._parse_role_overrides(None) == roles_mod.RoleParams()


# ==================================================================== cues

class TestCueArgParsing:
    def test_plain(self):
        assert cli._parse_cue_arg("8.0:darker") == (8.0, "darker",
                                                    "next-phrase")

    def test_boundary_suffix(self):
        t, text, at = cli._parse_cue_arg("16:go to the chorus@next-section")
        assert (t, text, at) == (16.0, "go to the chorus", "next-section")

    def test_at_sign_that_is_not_a_boundary_stays_text(self):
        t, text, at = cli._parse_cue_arg("4:email me @home")
        assert text == "email me @home" and at == "next-phrase"

    def test_colon_inside_the_text_survives(self):
        t, text, at = cli._parse_cue_arg("2:note: go darker")
        assert (t, text) == (2.0, "note: go darker")

    @pytest.mark.parametrize("bad", ["darker", "abc:darker", "-1:darker",
                                     "5:", "5:   "])
    def test_bad_forms_are_clean_errors(self, bad):
        with pytest.raises(cli.CliError):
            cli._parse_cue_arg(bad)

    def test_boundaries_come_from_mind(self):
        from plume.mind import CUE_BOUNDARIES
        assert cli._cue_boundaries() == tuple(CUE_BOUNDARIES)

    @pytest.mark.parametrize("bad", ["4:darker@whenever", "4:darker@soon",
                                     "8:go to the chorus@nextsection"])
    def test_an_unrecognised_boundary_is_an_error(self, bad):
        """v2.1: it used to be SWALLOWED.

        MEASURED: ``--cue "4:darker@whenever"`` exited 0 and logged
        ``cue 'darker@whenever' (texture:darker): applied`` — the bogus
        boundary was absorbed into the cue TEXT and the cue quietly ran at the
        default next-phrase.  ``plume cue --at`` has always validated this;
        the two surfaces now agree.
        """
        with pytest.raises(cli.CliError) as ei:
            cli._parse_cue_arg(bad)
        msg = str(ei.value)
        assert "next-phrase" in msg and "boundary" in msg

    @pytest.mark.parametrize("ok", ["4:email me @home",
                                    "4:tell @ the turnaround",
                                    "4:mail bob@example.com about it"])
    def test_prose_containing_an_at_sign_is_still_prose(self, ok):
        t, text, at = cli._parse_cue_arg(ok)
        assert at == "next-phrase" and "@" in text


class TestHumanOffsetIsBounded:
    """v2.1: ``--human-offset`` pads the human with silence, and unbounded
    padding fills the disk.

    MEASURED: ``plume simlead h.wav --bars 8 --human-offset 9999`` exited 0,
    reported ``duration_s: 10029.3`` and wrote a 2,888,438,444-byte (2.89 GB,
    2h47m) WAV — from a request for about twenty seconds of music.
    """

    def _args(self, off):
        return SimpleNamespace(human_offset=off)

    @pytest.mark.parametrize("bad", [9999.0, 3600.1, float("inf"), -1.0,
                                     float("nan")])
    def test_absurd_offsets_are_refused(self, bad):
        with pytest.raises(cli.CliError) as ei:
            cli._check_human_offset(self._args(bad))
        assert "--human-offset" in str(ei.value)

    @pytest.mark.parametrize("ok", [0.0, 8.0, 120.0, 3600.0])
    def test_sane_offsets_pass(self, ok):
        cli._check_human_offset(self._args(ok))

    def test_the_error_names_the_size_the_number_implies(self):
        with pytest.raises(cli.CliError) as ei:
            cli._check_human_offset(self._args(9999.0))
        assert "GB" in str(ei.value)


class TestPromptFeedback:
    """v2.1: the prompt path had no feedback channel at all."""

    def test_a_tempo_without_the_word_bpm_is_read(self):
        """Both lead-mode help texts documented examples that hit this bug:
        `plume roles` printed ``--prompt 'chill trap at 120'`` and
        `plume lead --help` documented ``'chill trap at 120, 3 movements'``,
        and BOTH composed at 140."""
        from plume.compose import parse_prompt
        for text in ("chill trap at 120", "chill trap at 120, 3 movements",
                     "lofi @96", "dark synthwave around 100", "trap tempo 150"):
            assert parse_prompt(text).bpm is not None, text
        assert parse_prompt("chill trap at 120").bpm == 120.0

    @pytest.mark.parametrize("text,want", [
        ("at 3 movements", None),      # a count, not a tempo
        ("trap at 4 bars", None),      # a length, not a tempo
        ("lofi in 2024", None),        # out of the tempo range
    ])
    def test_a_number_that_is_not_a_tempo_is_not_read_as_one(self, text, want):
        from plume.compose import parse_prompt
        assert parse_prompt(text).bpm == want

    def test_a_prompt_that_parses_to_nothing_says_so(self, capsys, monkeypatch,
                                                    tmp_path, stub_simlead):
        code, out, err = run_main(
            ["simlead", "--prompt", "asdfjkl qqq zzz",
             "--out", str(tmp_path / "d.wav")], capsys, monkeypatch)
        assert code == 0
        assert "nothing in" in err and "understood" in err

    def test_a_good_prompt_reports_what_it_took(self, capsys, monkeypatch,
                                               tmp_path, stub_simlead):
        code, out, err = run_main(
            ["simlead", "--prompt", "chill trap at 120 bpm, 3 movements",
             "--out", str(tmp_path / "d.wav")], capsys, monkeypatch)
        assert code == 0
        assert "bpm 120" in err and "trap" in err

    def test_json_carries_the_report_and_stderr_stays_clean(
            self, capsys, monkeypatch, tmp_path, stub_simlead):
        code, out, err = run_main(
            ["simlead", "--prompt", "asdfjkl", "--out", str(tmp_path / "d.wav"),
             "--json"], capsys, monkeypatch)
        assert code == 0 and err == ""
        doc = json.loads(out)
        assert doc["prompt"]["understood_any"] is False
        assert "asdfjkl" in doc["prompt"]["ignored"]

    def test_an_empty_prompt_says_the_prompt_is_empty(self, capsys, monkeypatch,
                                                     tmp_path):
        code, out, err = run_main(
            ["simlead", "--prompt", "", "--out", str(tmp_path / "d.wav")],
            capsys, monkeypatch)
        assert code == 1
        assert "--prompt is empty" in err
        assert "session" not in err.split("empty")[0]


class TestCueSocket:
    """The socket path, with no live session anywhere near it."""

    def test_roundtrip(self):
        path = sock_path("rt")
        with realtime.CueServer(path, kind="lead") as srv:
            resp = realtime.send_cue("darker", socket_path=path)
            assert resp["ok"] is True
            assert resp["session"] == "lead"
            queued = srv.drain()
        assert queued == [{"text": "darker", "at": "next-phrase",
                           "ttl_bars": 16}]

    def test_drain_is_destructive_and_ordered(self):
        path = sock_path("order")
        with realtime.CueServer(path) as srv:
            for t in ("one", "two", "three"):
                realtime.send_cue(t, socket_path=path)
            got = [c["text"] for c in srv.drain()]
            assert got == ["one", "two", "three"]
            assert srv.drain() == []
            assert srv.received == 3          # monotone, survives draining

    def test_at_and_ttl_are_carried(self):
        path = sock_path("attl")
        with realtime.CueServer(path) as srv:
            realtime.send_cue("take it out", at="now", ttl_bars=4,
                              socket_path=path)
            c = srv.drain()[0]
        assert c == {"text": "take it out", "at": "now", "ttl_bars": 4}

    def test_bar_is_reported_back(self):
        path = sock_path("bar")
        with realtime.CueServer(path) as srv:
            srv.bar = 12
            assert realtime.send_cue("darker", socket_path=path)["bar"] == 12

    def test_no_listener_is_a_clean_error(self):
        path = sock_path("dead")
        with pytest.raises(realtime.RealtimeError) as ei:
            realtime.send_cue("darker", socket_path=path)
        msg = str(ei.value)
        assert "no running plume session" in msg and path in msg

    def test_junk_does_not_take_the_server_down(self):
        """A cue arriving mid-song must not be able to end the session."""
        path = sock_path("junk")
        with realtime.CueServer(path) as srv:
            for payload in (b"not json\n", b"[1,2,3]\n", b"{}\n",
                            b'{"text": ""}\n', b'{"text": 5}\n',
                            b'{"text": "x", "ttl_bars": "soon"}\n'):
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(2.0)
                s.connect(path)
                s.sendall(payload)
                reply = json.loads(s.recv(4096).decode().splitlines()[0])
                s.close()
                assert reply["ok"] is False
                assert reply["error"]
            # still alive and still serving
            assert realtime.send_cue("darker", socket_path=path)["ok"] is True
            assert [c["text"] for c in srv.drain()] == ["darker"]

    def test_oversized_cue_is_refused_not_buffered(self):
        path = sock_path("big")
        with realtime.CueServer(path) as srv:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(path)
            try:
                # The server hangs up mid-stream once the cue passes the cap,
                # so the write itself may fail — which is the point.
                s.sendall(b'{"text": "' + b"x" * 200000 + b'"}\n')
            except OSError:
                pass
            s.close()
            assert srv.drain() == []          # never buffered
            # ...and the session is still taking cues.
            assert realtime.send_cue("darker", socket_path=path)["ok"] is True

    def test_second_session_refuses_to_steal_the_socket(self):
        path = sock_path("dup")
        with realtime.CueServer(path):
            with pytest.raises(realtime.RealtimeError) as ei:
                realtime.CueServer(path).start()
            assert "already listening" in str(ei.value)

    def test_stale_socket_file_is_reclaimed(self):
        """A crashed session leaves the file behind; the next one must start."""
        path = sock_path("stale")
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(path)
        s.close()                       # bound-then-closed: file, no listener
        assert Path(path).exists()
        with realtime.CueServer(path):
            assert realtime.send_cue("darker", socket_path=path)["ok"] is True

    def test_close_removes_the_socket_file(self):
        path = sock_path("cleanup")
        srv = realtime.CueServer(path).start()
        assert Path(path).exists()
        srv.close()
        assert not Path(path).exists()

    def test_default_path_follows_the_env(self, monkeypatch):
        monkeypatch.setenv("PLUME_SOCKET", "/tmp/plume-env-test.sock")
        assert realtime.default_socket_path() == "/tmp/plume-env-test.sock"
        monkeypatch.delenv("PLUME_SOCKET")
        assert realtime.default_socket_path().endswith("plume-live.sock")

    def test_absurd_path_length_says_so(self):
        long = "/tmp/" + "x" * 200 + ".sock"
        with pytest.raises(realtime.RealtimeError) as ei:
            realtime.send_cue("x", socket_path=long)
        assert "too long" in str(ei.value)


class TestCueCommand:
    def test_end_to_end_over_the_socket(self, capsys, monkeypatch):
        path = sock_path("cli")
        with realtime.CueServer(path, kind="lead") as srv:
            code, out, _ = run_main(
                ["cue", "go to the chorus", "--at", "next-section",
                 "--socket", path, "--json"], capsys, monkeypatch)
            queued = srv.drain()
        assert code == 0
        doc = json.loads(out)
        assert doc["sent"] is True
        assert doc["session"] == "lead"
        assert doc["parsed"]["kind"] == "section"
        assert queued == [{"text": "go to the chorus", "at": "next-section",
                           "ttl_bars": 16}]

    def test_human_output_names_the_kind(self, capsys, monkeypatch):
        path = sock_path("cli2")
        with realtime.CueServer(path):
            code, out, _ = run_main(["cue", "darker", "--socket", path],
                                    capsys, monkeypatch)
        assert code == 0
        assert "darker" in out and "next-phrase" in out

    def test_unparseable_cue_is_still_delivered(self, capsys, monkeypatch):
        """parse_cue never raises and junk is an explicit no-op, so the CLI
        must not pre-emptively refuse it either."""
        path = sock_path("cli3")
        with realtime.CueServer(path) as srv:
            code, out, _ = run_main(
                ["cue", "asdfgh qwerty", "--socket", path, "--json"],
                capsys, monkeypatch)
            queued = srv.drain()
        assert code == 0
        assert json.loads(out)["parsed"]["kind"] == "unknown"
        assert len(queued) == 1

    def test_no_session_is_a_clean_error(self, capsys, monkeypatch):
        path = sock_path("cli4")
        code, out, err = run_main(["cue", "darker", "--socket", path],
                                  capsys, monkeypatch)
        assert code == 1
        assert "no running plume session" in err
        assert "Traceback" not in err

    def test_no_session_json(self, capsys, monkeypatch):
        path = sock_path("cli5")
        code, out, err = run_main(["cue", "darker", "--socket", path, "--json"],
                                  capsys, monkeypatch)
        assert code == 1
        assert err == ""
        assert "no running plume session" in json.loads(out)["error"]

    def test_bad_boundary_has_did_you_mean(self, capsys, monkeypatch):
        code, _, err = run_main(["cue", "darker", "--at", "next-phase"],
                                capsys, monkeypatch)
        assert code == 1
        assert "next-phrase" in err

    def test_bad_ttl_is_rejected_before_sending(self, capsys, monkeypatch):
        code, _, err = run_main(["cue", "darker", "--ttl-bars", "0"],
                                capsys, monkeypatch)
        assert code == 1
        assert "ttl-bars" in err


# ================================================================= simlead

class _StubResult:
    """Stand-in SimResult so the CLI's wiring is covered whether or not
    builder D's simulate_lead has landed."""

    def __init__(self, out_path, decisions_path):
        self.out_path = str(out_path)
        self.decisions_path = str(decisions_path)
        self.duration_s = 12.0
        self.entered_bar = 0
        self.states_by_bar = ["comping"] * 6
        self.bpm_trace = [(i * 0.5, 120.0) for i in range(200)]
        self.final_key = "D minor"
        self.manifest = {"tempo_offset_pct": 1.5}


@pytest.fixture
def stub_simlead(monkeypatch, tmp_path):
    """Record what the CLI passes to simulate_lead."""
    import plume.sim as sim_mod

    seen = {}

    def fake(input_wav, **kw):
        seen["input"] = input_wav
        seen.update(kw)
        out = Path(kw["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        # Mirror the real simulate_lead: honour decisions_path when given.
        dec = Path(kw.get("decisions_path") or out.with_suffix(".ndjson"))
        dec.write_text('{"bar": 0}\n')
        return _StubResult(out, dec)

    monkeypatch.setattr(sim_mod, "simulate_lead", fake, raising=False)
    return seen


class TestSimleadWiring:
    def test_prompt_path(self, stub_simlead, tmp_path, capsys, monkeypatch):
        out = tmp_path / "duet.wav"
        code, txt, _ = run_main(
            ["simlead", "--prompt", "chill trap at 120", "--out", str(out)],
            capsys, monkeypatch)
        assert code == 0
        assert stub_simlead["input"] is None
        assert stub_simlead["prompt"] == "chill trap at 120"
        assert stub_simlead["session"] is None
        assert stub_simlead["seed"] == 0
        assert stub_simlead["human_offset_s"] == 0.0
        assert stub_simlead["role"].role == "lead"       # the machine starts
        assert stub_simlead["mind"] is None
        assert stub_simlead["cues"] is None
        assert str(out) in txt

    def test_session_path_and_human_wav(self, stub_simlead, tmp_path, capsys,
                                        monkeypatch, material):
        sess_path = tmp_path / "s.plume.json"
        from plume import score
        score.save(material["trap140"]["session"], sess_path)
        code, _, err = run_main(
            ["simlead", str(material["trap140"]["loud"]),
             "--session", str(sess_path), "--out", str(tmp_path / "d.wav"),
             "--human-offset", "8.0", "--seed", "3",
             "--role", "trade", "--roles", "trade_bars=8",
             "--mind", "min_listen_bars=2",
             "--cue", "8.0:darker", "--cue", "16:go to the chorus@next-section"],
            capsys, monkeypatch)
        assert code == 0, err
        assert stub_simlead["input"] == str(material["trap140"]["loud"])
        assert stub_simlead["prompt"] is None
        assert stub_simlead["session"].bpm == 140.0
        assert stub_simlead["human_offset_s"] == 8.0
        assert stub_simlead["seed"] == 3
        assert stub_simlead["role"].role == "trade"
        assert stub_simlead["role"].trade_bars == 8
        assert stub_simlead["mind"].min_listen_bars == 2
        cues = stub_simlead["cues"]
        assert [(t, c.text, c.at) for t, c in cues] == [
            (8.0, "darker", "next-phrase"),
            (16.0, "go to the chorus", "next-section")]

    def test_json_is_exactly_one_document(self, stub_simlead, tmp_path, capsys,
                                          monkeypatch):
        code, out, err = run_main(
            ["simlead", "--prompt", "x", "--out", str(tmp_path / "d.wav"),
             "--json"], capsys, monkeypatch)
        assert code == 0
        assert err == ""
        doc = json.loads(out)
        assert doc["final_key"] == "D minor"

    def test_json_decimates_the_trace_unless_asked(self, stub_simlead,
                                                   tmp_path, capsys,
                                                   monkeypatch):
        code, out, _ = run_main(
            ["simlead", "--prompt", "x", "--out", str(tmp_path / "a.wav"),
             "--json"], capsys, monkeypatch)
        doc = json.loads(out)
        assert len(doc["bpm_trace"]) <= cli._TRACE_POINTS
        assert doc["bpm_trace_decimated"] is True
        assert doc["bpm_trace_points_total"] == 200
        code, out, _ = run_main(
            ["simlead", "--prompt", "x", "--out", str(tmp_path / "b.wav"),
             "--json", "--full-trace"], capsys, monkeypatch)
        assert len(json.loads(out)["bpm_trace"]) == 200

    def test_decisions_flag_is_honoured(self, stub_simlead, tmp_path,
                                        capsys, monkeypatch):
        want = tmp_path / "elsewhere.ndjson"
        code, out, _ = run_main(
            ["simlead", "--prompt", "x", "--out", str(tmp_path / "d.wav"),
             "--decisions", str(want), "--json"], capsys, monkeypatch)
        assert code == 0
        assert want.exists()
        assert json.loads(out)["decisions_path"] == str(want)

    def test_decisions_flag_relocates_when_sim_cannot_place_it(
            self, monkeypatch, tmp_path, capsys):
        """Against the SPEC-V2 signature (no decisions_path parameter) the CLI
        still honours --decisions, by moving the file sim wrote."""
        import plume.sim as sim_mod

        def spec_shaped(input_wav, *, session=None, prompt=None, out_path,
                        seed=0, role=None, mind=None, human_offset_s=0.0,
                        cues=None):
            out = Path(out_path)
            out.write_bytes(b"")
            dec = out.with_suffix(".ndjson")
            dec.write_text('{"bar": 0}\n')
            return _StubResult(out, dec)

        monkeypatch.setattr(sim_mod, "simulate_lead", spec_shaped,
                            raising=False)
        want = tmp_path / "moved.ndjson"
        code, out, err = run_main(
            ["simlead", "--prompt", "x", "--out", str(tmp_path / "d.wav"),
             "--decisions", str(want), "--json"], capsys, monkeypatch)
        assert code == 0, err
        assert want.exists()
        assert json.loads(out)["decisions_path"] == str(want)

    def test_optional_kwargs_only_go_to_a_sim_that_takes_them(
            self, stub_simlead, tmp_path, capsys, monkeypatch):
        code, _, err = run_main(
            ["simlead", "--prompt", "x", "--out", str(tmp_path / "d.wav"),
             "--bars", "8", "--mix-human-db", "-6", "--style", "trap"],
            capsys, monkeypatch)
        assert code == 0, err
        assert stub_simlead["bars"] == 8
        assert stub_simlead["mix_human_db"] == -6.0
        assert stub_simlead["style"] == "trap"

    def test_accepted_kwargs_reads_a_real_signature(self):
        def f(a, *, b=1, c=2):
            return a, b, c

        assert cli._accepted_kwargs(f) == {"a", "b", "c"}
        assert cli._accepted_kwargs(lambda a, **kw: None) is None


class TestSimleadErrors:
    def test_missing_input_file(self, capsys, monkeypatch, tmp_path):
        code, _, err = run_main(
            ["simlead", str(tmp_path / "nope.wav"), "--prompt", "x"],
            capsys, monkeypatch)
        assert code == 1
        assert "file not found" in err

    def test_prompt_and_session_together(self, capsys, monkeypatch, tmp_path,
                                         material):
        from plume import score
        p = tmp_path / "s.plume.json"
        score.save(material["lofi80"]["session"], p)
        code, _, err = run_main(
            ["simlead", "--prompt", "x", "--session", str(p)],
            capsys, monkeypatch)
        assert code == 1
        assert "not both" in err

    def test_no_prompt_and_no_session(self, capsys, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PLUME_SESSION", raising=False)
        code, _, err = run_main(["simlead"], capsys, monkeypatch)
        assert code == 1
        assert "no session found" in err

    def test_bad_cue_arg(self, capsys, monkeypatch):
        code, _, err = run_main(["simlead", "--prompt", "x", "--cue", "darker"],
                                capsys, monkeypatch)
        assert code == 1
        assert "T:TEXT" in err

    def test_missing_simulate_lead_says_so(self, capsys, monkeypatch):
        import plume.sim as sim_mod
        monkeypatch.delattr(sim_mod, "simulate_lead", raising=False)
        code, _, err = run_main(["simlead", "--prompt", "x"], capsys,
                                monkeypatch)
        assert code == 1
        assert "simulate_lead" in err
        assert "Traceback" not in err


@pytest.mark.skipif(not HAVE_SIMLEAD,
                    reason="plume.sim.simulate_lead not built yet (builder D)")
class TestSimleadEndToEnd:
    """The real thing, on real material, through the real CLI."""

    def test_machine_solo_from_a_bare_prompt(self, tmp_path, capsys,
                                             monkeypatch):
        out = tmp_path / "solo.wav"
        code, txt, err = run_main(
            ["simlead", "--prompt", "chill trap at 120", "--out", str(out),
             "--json"], capsys, monkeypatch)
        assert code == 0, err
        doc = json.loads(txt)
        assert Path(doc["out_path"]).exists()
        x, sr = sf.read(doc["out_path"], dtype="float32", always_2d=True)
        assert sr == SR
        assert x.shape[0] > SR                      # more than a second
        assert float(np.abs(x).max()) > 0.01        # the machine actually plays

    def test_human_joins_a_real_render(self, material, tmp_path, capsys,
                                       monkeypatch):
        out = tmp_path / "joined.wav"
        code, txt, err = run_main(
            ["simlead", str(material["trap140"]["loud"]),
             "--prompt", "trap at 140", "--human-offset", "4.0",
             "--out", str(out), "--json"], capsys, monkeypatch)
        assert code == 0, err
        doc = json.loads(txt)
        assert Path(doc["out_path"]).exists()
        lines = [json.loads(ln) for ln
                 in Path(doc["decisions_path"]).read_text().splitlines() if ln]
        assert lines, "no decisions were logged"

    def test_deterministic(self, material, tmp_path, capsys, monkeypatch):
        outs = []
        for i in range(2):
            o = tmp_path / f"det{i}.wav"
            code, _, err = run_main(
                ["simlead", str(material["lofi80"]["loud"]),
                 "--prompt", "lofi at 80", "--out", str(o), "--json"],
                capsys, monkeypatch)
            assert code == 0, err
            outs.append(o.read_bytes())
        assert outs[0] == outs[1]

    def test_quiet_input_still_produces_machine_audio(self, material, tmp_path,
                                                      capsys, monkeypatch):
        """v1's bug in one line: a -20 dB human must not silence the machine —
        and in LEAD mode the machine owns the floor, so it must not even
        hesitate."""
        o = tmp_path / "quiet.wav"
        code, txt, err = run_main(
            ["simlead", str(material["trap140"]["quiet"]),
             "--prompt", "trap at 140", "--out", str(o), "--json"],
            capsys, monkeypatch)
        assert code == 0, err
        x, _ = sf.read(json.loads(txt)["out_path"], dtype="float32",
                       always_2d=True)
        assert float(np.abs(x).max()) > 0.01


# ==================================================================== lead

class TestLeadNoDevice:
    def test_without_sounddevice_is_clean(self, capsys, monkeypatch):
        monkeypatch.setitem(sys.modules, "sounddevice", None)
        code, out, err = run_main(["lead", "chill trap at 120"], capsys,
                                  monkeypatch)
        assert code == 1
        assert "error:" in err
        assert "sounddevice" in err
        assert "Traceback" not in err

    def test_without_sounddevice_json(self, capsys, monkeypatch):
        monkeypatch.setitem(sys.modules, "sounddevice", None)
        code, out, err = run_main(["lead", "chill trap at 120", "--json"],
                                  capsys, monkeypatch)
        assert code == 1
        assert err == ""
        doc = json.loads(out)
        assert "sounddevice" in doc["error"]
        assert len(out.strip().splitlines()) >= 1
        json.loads(out)                     # exactly one document

    def test_no_devices_at_all_is_clean(self, capsys, monkeypatch):
        monkeypatch.setitem(sys.modules, "sounddevice", _FakeSd([]))
        code, _, err = run_main(["lead", "x"], capsys, monkeypatch)
        assert code == 1
        assert "no audio" in err

    def test_named_device_missing_lists_devices(self, capsys, monkeypatch):
        monkeypatch.setitem(sys.modules, "sounddevice", _FakeSd(_DUPLEX))
        code, _, err = run_main(["lead", "x", "--device", "UMC404"], capsys,
                                monkeypatch)
        assert code == 1
        assert "UMC404" in err and "Built-in" in err

    def test_device_check_happens_before_composing(self, capsys, monkeypatch):
        """The no-device path must not spend a second composing first."""
        monkeypatch.setitem(sys.modules, "sounddevice", None)
        import plume.compose as compose_mod
        monkeypatch.setattr(compose_mod, "compose", lambda *a, **k:
                            pytest.fail("composed before checking the device"))
        code, _, err = run_main(["lead", "chill trap at 120"], capsys,
                                monkeypatch)
        assert code == 1

    def test_error_message_names_the_offline_twin(self, capsys, monkeypatch):
        monkeypatch.setitem(sys.modules, "sounddevice", None)
        code, _, err = run_main(["lead", "x"], capsys, monkeypatch)
        assert "simlead" in err or "plume sim" in err


class TestLeadWiring:
    """What the CLI hands realtime.lead, with the transport stubbed out."""

    @pytest.fixture
    def stub_lead(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(realtime, "check_device", lambda dev=None: None)
        monkeypatch.setattr(realtime, "lead", lambda **kw: seen.update(kw))
        return seen

    def test_prompt_and_defaults(self, stub_lead, capsys, monkeypatch):
        code, out, err = run_main(["lead", "chill trap at 120"], capsys,
                                  monkeypatch)
        assert code == 0, err
        assert stub_lead["prompt"] == "chill trap at 120"
        assert stub_lead["session"] is None
        assert stub_lead["role"].role == "lead"
        assert stub_lead["from_bar"] == 0
        assert stub_lead["seed"] == 0
        assert "lead ended" in out

    def test_session_and_from_bar(self, stub_lead, tmp_path, capsys,
                                  monkeypatch, material):
        from plume import score
        p = tmp_path / "s.plume.json"
        score.save(material["trap140"]["session"], p)
        code, _, err = run_main(
            ["lead", "--session", str(p), "--from-bar", "8", "--role", "trade",
             "--roles", "trade_bars=8", "--record", str(tmp_path / "rec"),
             "--socket", "/tmp/plume-x.sock"], capsys, monkeypatch)
        assert code == 0, err
        assert stub_lead["prompt"] is None
        assert stub_lead["session"].bpm == 140.0
        assert stub_lead["from_bar"] == 8
        assert stub_lead["role"].role == "trade"
        assert stub_lead["role"].trade_bars == 8
        assert stub_lead["socket_path"] == "/tmp/plume-x.sock"

    def test_json_reports_the_socket(self, stub_lead, capsys, monkeypatch):
        code, out, _ = run_main(["lead", "x", "--json"], capsys, monkeypatch)
        assert code == 0
        doc = json.loads(out)
        assert doc["ended"] is True
        assert doc["socket"].endswith(".sock")

    def test_negative_from_bar_is_a_clean_error(self, stub_lead, capsys,
                                                monkeypatch):
        code, _, err = run_main(["lead", "x", "--from-bar", "-1"], capsys,
                                monkeypatch)
        assert code == 1
        assert "from-bar" in err

    def test_prompt_and_session_together(self, stub_lead, tmp_path, capsys,
                                         monkeypatch, material):
        from plume import score
        p = tmp_path / "s.plume.json"
        score.save(material["lofi80"]["session"], p)
        code, _, err = run_main(["lead", "x", "--session", str(p)], capsys,
                                monkeypatch)
        assert code == 1
        assert "not both" in err


# ============================================================ jam: unchanged

class TestJamBackwardsCompatible:
    def test_bare_jam_passes_exactly_the_v1_signature(self, capsys,
                                                      monkeypatch):
        """No --role, no --lead-in => not one extra kwarg reaches realtime."""
        seen = {}
        monkeypatch.setattr(realtime, "check_device", lambda dev=None: None)
        monkeypatch.setattr(realtime, "jam", lambda **kw: seen.update(kw))
        code, _, err = run_main(["jam"], capsys, monkeypatch)
        assert code == 0, err
        assert set(seen) == {"style", "seed", "params", "device", "monitor",
                             "record_dir", "tightness"}

    def test_v1_flags_still_present(self, capsys, monkeypatch):
        with pytest.raises(SystemExit):
            run_main(["jam", "--help"], capsys, monkeypatch)
        out = capsys.readouterr().out
        for flag in ("--style", "--seed", "--tightness", "--mind", "--record",
                     "--device", "--monitor"):
            assert flag in out

    def test_role_flag_reaches_realtime(self, capsys, monkeypatch):
        seen = {}
        monkeypatch.setattr(realtime, "check_device", lambda dev=None: None)
        monkeypatch.setattr(realtime, "jam", lambda **kw: seen.update(kw))
        code, _, err = run_main(["jam", "--role", "lead"], capsys, monkeypatch)
        assert code == 0, err
        assert seen["role"].role == "lead"

    def test_lead_in_reaches_realtime(self, capsys, monkeypatch):
        seen = {}
        monkeypatch.setattr(realtime, "check_device", lambda dev=None: None)
        monkeypatch.setattr(realtime, "jam", lambda **kw: seen.update(kw))
        code, _, _ = run_main(["jam", "--lead-in", "trap at 140"], capsys,
                              monkeypatch)
        assert code == 0
        assert seen["lead_in"] == "trap at 140"

    def test_bad_role_is_rejected_before_the_device_opens(self, capsys,
                                                          monkeypatch):
        monkeypatch.setattr(realtime, "check_device", lambda dev=None: None)
        monkeypatch.setattr(realtime, "jam", lambda **kw:
                            pytest.fail("jam ran with a bad role"))
        code, _, err = run_main(["jam", "--role", "leed"], capsys, monkeypatch)
        assert code == 1
        assert "unknown role" in err

    @pytest.mark.parametrize("role,lead_in,session,want", [
        (None, None, None, False),                      # bare v1 jam
        (roles_mod.RoleParams(), None, None, False),    # explicit follow
        (roles_mod.RoleParams(role="lead"), None, None, True),
        (roles_mod.RoleParams(role="trade"), None, None, True),
        (roles_mod.RoleParams(clock_authority=0.8), None, None, True),
        (None, "trap at 140", None, True),
        (None, None, object(), True),
    ])
    def test_wants_lead_truth_table(self, role, lead_in, session, want):
        assert realtime._wants_lead(role, lead_in, session) is want

    def test_jam_dispatches_to_lead_only_when_asked(self, monkeypatch):
        called = {}
        monkeypatch.setattr(realtime, "lead", lambda **kw: called.setdefault(
            "kw", kw))
        monkeypatch.setitem(sys.modules, "sounddevice", None)
        # follow + nothing else: falls through to the v1 body, which dies on
        # the missing device rather than dispatching.
        with pytest.raises(realtime.RealtimeError):
            realtime.jam(style="lofi", seed=0, params=None)
        assert "kw" not in called
        realtime.jam(style="lofi", seed=0, params=None,
                     role=roles_mod.RoleParams(role="lead"))
        assert called["kw"]["role"].role == "lead"


# ============================================================ machine clock

class TestMachineClock:
    def test_bar_times_at_a_fixed_tempo(self):
        c = realtime.MachineClock(120.0)
        got = [c.commit(120.0)[:2] for _ in range(4)]
        assert got == [(0, 0.0), (1, 2.0), (2, 4.0), (3, 6.0)]

    def test_tempo_change_takes_effect_on_the_next_bar(self):
        c = realtime.MachineClock(120.0)
        c.commit(120.0)                      # bar 0 at 120 -> 2.0s long
        bar, t, bpm = c.commit(240.0)        # bar 1 starts at 2.0, is 1.0s
        assert (bar, t, bpm) == (1, 2.0, 240.0)
        assert c.t_s == pytest.approx(3.0)

    def test_bar_len_tracks_bpm(self):
        c = realtime.MachineClock(140.0)
        assert c.bar_len_s() == pytest.approx(4 * 60.0 / 140.0)
        assert c.bar_len_s(70.0) == pytest.approx(4 * 60.0 / 70.0)

    def test_t0_offset(self):
        c = realtime.MachineClock(120.0, t0=5.0)
        assert c.commit()[1] == 5.0
        assert c.t_s == pytest.approx(7.0)

    def test_is_pure(self):
        """Two clocks fed the same tempi produce identical grids — no wall
        clock anywhere on the live path's timing."""
        seq = [120.0, 121.0, 121.9, 121.9, 118.0]
        a = realtime.MachineClock(120.0)
        b = realtime.MachineClock(120.0)
        assert [a.commit(x) for x in seq] == [b.commit(x) for x in seq]

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_bad_bpm_raises(self, bad):
        with pytest.raises(ValueError):
            realtime.MachineClock(bad)
        c = realtime.MachineClock(120.0)
        with pytest.raises(ValueError):
            c.commit(bad)

    def test_machine_beat_state_is_a_real_grid(self):
        from plume.listen.beat import BeatState
        bs = realtime.machine_beat_state(3, 6.0, 120.0)
        assert isinstance(bs, BeatState)
        assert bs.locked is True and bs.confidence == 1.0
        assert bs.bar == 3
        assert bs.beat_interval_s == pytest.approx(0.5)
        assert bs.next_downbeat_t_s == pytest.approx(8.0)


class TestTransposeHelpers:
    def test_chord_and_key_move_together(self):
        assert realtime.transpose_key("F minor", 1) == "F# minor"
        assert realtime.transpose_symbol("Fm7", 1, "F# minor") == "F#m7"

    def test_zero_is_identity(self):
        assert realtime.transpose_symbol("Fm7", 0, "F minor") == "Fm7"
        assert realtime.transpose_key("F minor", 0) == "F minor"

    def test_junk_passes_through(self):
        """A conformance transposition must never make a bar unplayable."""
        assert realtime.transpose_symbol(None, 2, "C major") is None
        assert realtime.transpose_symbol("???", 2, "C major") == "???"
        assert realtime.transpose_key(None, 2) is None

    def test_machine_harmony_state_applies_the_transpose(self):
        h = realtime.machine_harmony_state(1.0, "F minor", "Fm7", transpose=1)
        assert h.key == "F# minor" and h.chord == "F#m7"
        assert h.key_conf == 1.0


# ============================================================== bar plans

class TestLeadBarPlan:
    class _Sec:
        def __init__(self, name="drop", energy=0.9, plays=None):
            self.name = name
            self.energy = energy
            self.plays = plays if plays is not None else {}

    class _Form:
        def __init__(self, announce=False):
            self.announce = announce

    def _role(self, clock=0.9, form=1.0):
        return roles_mod.RoleState(
            t_s=0.0, bar=0, role="lead", clock_authority=clock,
            form_authority=form, tempo_offset_pct=0.0, transpose_semitones=0,
            reason="")

    def test_energy_becomes_density(self):
        p = realtime.lead_bar_plan(2, self._Sec(energy=0.4), ("kick", "bass"),
                                   self._role(), self._Form())
        assert p.density == pytest.approx(0.4)
        assert p.bar == 2

    def test_plays_selects_the_roles(self):
        sec = self._Sec(plays={"kick": "main", "bass": "main"})
        p = realtime.lead_bar_plan(0, sec, ("kick", "snare", "bass"),
                                   self._role(), self._Form())
        assert set(p.roles) == {"kick", "bass"}

    def test_no_overlap_falls_back_to_the_whole_style(self):
        """A section whose track names are not the style's roles must still
        make sound — silence is the failure mode this build exists to kill."""
        sec = self._Sec(plays={"guitar": "main"})
        p = realtime.lead_bar_plan(0, sec, ("kick", "bass"), self._role(),
                                   self._Form())
        assert set(p.roles) == {"kick", "bass"}

    def test_machine_always_sounds_in_lead_mode(self):
        for energy in (0.0, 0.05, 0.5, 1.0):
            p = realtime.lead_bar_plan(0, self._Sec(energy=energy), ("kick",),
                                       self._role(), self._Form())
            assert p.roles and max(p.roles.values()) > 0.0
            assert p.state in __import__("plume.mind", fromlist=["x"]).STATES

    def test_announce_bumps_density(self):
        base = realtime.lead_bar_plan(0, self._Sec(energy=0.5), ("kick",),
                                      self._role(), self._Form(False))
        ann = realtime.lead_bar_plan(0, self._Sec(energy=0.5), ("kick",),
                                     self._role(), self._Form(True))
        assert ann.density > base.density
        assert "announce" in ann.reason

    def test_yielding_pulls_the_machine_down(self):
        held = realtime.lead_bar_plan(0, self._Sec(), ("kick",),
                                      self._role(clock=1.0), self._Form())
        gone = realtime.lead_bar_plan(0, self._Sec(), ("kick",),
                                      self._role(clock=0.0), self._Form())
        assert held.gain_db == pytest.approx(0.0)
        assert gone.gain_db < held.gain_db

    def test_register_ceiling_only_once_sharing(self):
        from plume.mind import BarFeatures
        human = BarFeatures(bar=0, rms_db=-20.0, onset_density=2.0,
                            silent=False, register_low_midi=60.0,
                            register_high_midi=72.0)
        owning = realtime.lead_bar_plan(0, self._Sec(), ("pad",),
                                        self._role(clock=0.9), self._Form(),
                                        human=human)
        sharing = realtime.lead_bar_plan(0, self._Sec(), ("pad",),
                                         self._role(clock=0.3), self._Form(),
                                         human=human)
        assert owning.register_ceiling_midi is None
        assert sharing.register_ceiling_midi == pytest.approx(58.0)

    def test_pure(self):
        args = (1, self._Sec(), ("kick", "bass"), self._role(), self._Form())
        assert realtime.lead_bar_plan(*args) == realtime.lead_bar_plan(*args)


# ======================================================== REAL MATERIAL

class TestLeadPlannerOnRealMaterial:
    """The live lead planner's per-bar chain, on plume's own v0 renders.

    Not a fixture-shaped test: the human here is a real percussive mix with a
    5-10% duty cycle at two gains 20 dB apart, which is exactly the material
    that made every v1 bar read as silent.
    """

    def test_machine_plays_from_bar_zero(self, trap_pass):
        decisions, bars, _, _ = trap_pass
        assert len(bars) >= 8
        assert decisions[0]["bar"] == 0
        # Lead mode's whole point: no silence-until-confident, no tacet.
        assert all(d["peak"] > 0.0 for d in decisions), (
            "the machine went silent in lead mode")

    def test_machine_never_tacets(self, trap_pass):
        decisions, _, _, _ = trap_pass
        assert all(d["roles"] for d in decisions)
        assert all(max(d["roles"].values()) > 0.0 for d in decisions)

    def test_grid_is_monotone_and_at_the_composed_tempo(self, trap_pass,
                                                        material):
        decisions, _, _, _ = trap_pass
        ts = [d["t_s"] for d in decisions]
        assert ts == sorted(ts)
        composed = float(material["trap140"]["session"].bpm)
        for d in decisions:
            drift = abs(d["bpm"] - composed) / composed * 100.0
            assert drift <= 8.0 + 1e-6, f"bar {d['bar']} drifted {drift:.1f}%"

    def test_tempo_never_jumps_more_than_two_percent_in_a_bar(self, trap_pass):
        decisions, _, _, _ = trap_pass
        for a, b in zip(decisions, decisions[1:]):
            step = abs(b["bpm"] - a["bpm"]) / a["bpm"] * 100.0
            assert step <= 2.0 + 1e-6, f"bar {b['bar']} jumped {step:.2f}%"

    def test_form_actions_are_from_the_vocabulary(self, trap_pass):
        from plume.mind import FORM_ACTIONS
        decisions, _, _, _ = trap_pass
        assert all(d["action"] in FORM_ACTIONS for d in decisions)

    def test_role_states_are_well_formed(self, trap_pass):
        decisions, _, _, _ = trap_pass
        for d in decisions:
            assert d["role"] in roles_mod.ROLES
            assert 0.0 <= d["clock_authority"] <= 1.0
            assert 0.0 <= d["form_authority"] <= 1.0
            assert d["role_reason"]

    def test_gain_invariance_of_the_machines_own_behavior(self, material):
        """The machine's clock and form must not depend on the human's fader.

        v1's defining bug was a rule that fired on a hot input and never on a
        quiet one.  In lead mode the machine owns the grid, so a 20 dB change
        in the human's level must not move a single bar boundary.
        """
        m = material["trap140"]
        loud, _, _, _ = _lead_planner_pass(m["session"], m["loud"], max_bars=12)
        quiet, _, _, _ = _lead_planner_pass(m["session"], m["quiet"],
                                            max_bars=12)
        assert len(loud) == len(quiet)
        assert [d["bar"] for d in loud] == [d["bar"] for d in quiet]
        assert all(d["peak"] > 0.0 for d in quiet)

    def test_runs_on_swung_lofi_too(self, material):
        m = material["lofi80"]
        decisions, bars, _, _ = _lead_planner_pass(m["session"], m["loud"],
                                                   max_bars=10)
        assert len(bars) >= 6
        assert all(d["peak"] > 0.0 for d in decisions)

    def test_trade_role_alternates_authority(self, material):
        m = material["trap140"]
        rp = roles_mod.RoleParams(role="trade", trade_bars=2)
        decisions, _, _, _ = _lead_planner_pass(m["session"], m["loud"],
                                                role_params=rp, max_bars=12)
        auth = [d["clock_authority"] for d in decisions]
        assert len(set(auth)) > 1, f"authority never moved in trade: {auth}"

    def test_deterministic(self, material):
        m = material["lofi80"]
        a, _, _, _ = _lead_planner_pass(m["session"], m["loud"], max_bars=8)
        b, _, _, _ = _lead_planner_pass(m["session"], m["loud"], max_bars=8)
        strip = lambda ds: [{k: v for k, v in d.items()} for d in ds]  # noqa: E731
        assert strip(a) == strip(b)

    def test_lead_runs_end_to_end_over_a_fake_stream(self, material, tmp_path):
        """``realtime.lead`` itself — threads, timeline, cue socket and all —
        driven by a fake duplex device whose 'mic' is a REAL v0 render.

        There is no audio device in CI, so this is the only coverage the live
        transport can have; without it the planner thread's wiring would be
        asserted precisely nowhere.  ``loop_form=False`` is what lets it stop
        on its own: the end of the map becomes an ``end`` action.
        """
        from plume.mind import FormParams

        m = material["lofi80"]
        human, sr = sf.read(str(m["loud"]), dtype="float32", always_2d=True)
        fake = _FakeSdStream(_DUPLEX, human.mean(axis=1).astype(np.float32),
                             rate_x=12.0)
        rec = tmp_path / "rec"
        path = sock_path("live")
        with _patched_module("sounddevice", fake):
            realtime.lead(session=m["session"], seed=0,
                          role=roles_mod.RoleParams(role="lead"),
                          form_params=FormParams(loop_form=False),
                          record_dir=str(rec), socket_path=path, status=False)

        assert fake.stream is not None and fake.stream.blocks > 0
        # The machine actually made sound, from the very first block on.
        out = np.concatenate(fake.stream.captured, axis=0)
        assert float(np.abs(out).max()) > 1e-4, "the machine never sounded"
        first_bar = int(4 * 60.0 / float(m["session"].bpm) * SR)
        assert float(np.abs(out[:first_bar]).max()) > 1e-4, (
            "lead mode was silent through its own first bar")
        # ...and the records exist and describe lead mode.
        lines = [json.loads(ln) for ln
                 in (rec / "decisions.ndjson").read_text().splitlines() if ln]
        assert len(lines) >= 4
        assert lines[0]["bar"] == 0
        for d in lines:
            assert d["role"] in roles_mod.ROLES
            assert "form" in d and d["form"]["action"]
            assert d["roles"]
        assert (rec / "human.wav").exists() and (rec / "machine.wav").exists()
        # the socket is cleaned up when the session ends
        assert not Path(path).exists()

    def test_lead_takes_a_live_cue_over_the_socket(self, material, tmp_path):
        """A cue sent by ``plume cue`` while the band is playing reaches the
        planner and lands in the decision log."""
        from plume.mind import FormParams

        m = material["trap140"]
        human, _ = sf.read(str(m["loud"]), dtype="float32", always_2d=True)
        fake = _FakeSdStream(_DUPLEX, human.mean(axis=1).astype(np.float32),
                             rate_x=8.0)
        path = sock_path("livecue")
        sent = threading.Event()

        def cue_when_ready():
            for _ in range(400):
                if Path(path).exists():
                    try:
                        realtime.send_cue("go to the drop", at="next-bar",
                                          socket_path=path)
                        sent.set()
                        return
                    except realtime.RealtimeError:
                        pass
                time.sleep(0.01)

        t = threading.Thread(target=cue_when_ready, daemon=True)
        rec = tmp_path / "rec2"
        with _patched_module("sounddevice", fake):
            t.start()
            realtime.lead(session=m["session"], seed=0,
                          form_params=FormParams(loop_form=False),
                          record_dir=str(rec), socket_path=path, status=False)
        t.join(timeout=5.0)
        assert sent.is_set(), "the cue never reached the running session"
        lines = [json.loads(ln) for ln
                 in (rec / "decisions.ndjson").read_text().splitlines() if ln]
        applied = [d for d in lines if d["form"]["applied_cue"]]
        dropped = [d for d in lines if d["form"]["dropped_cue"]]
        assert applied or dropped, (
            "a cue was accepted by the socket but never reached the form log")

    def test_planning_cost_leaves_the_live_headroom_intact(self, trap_pass,
                                                           capsys):
        """v1's beat tracker runs ~180x real time and that headroom is why a
        live jam is possible.  What v2 ADDS to the live path is the per-bar
        decision chain (role negotiation -> form navigation -> bar plan);
        ``DuetEngine.render_bar`` beside it is v1 code that ``jam`` already
        pays for every bar.  Both are measured; the floor is on the addition.
        """
        decisions, bars, audio_s, cost = trap_pass
        assert cost["decide_s"] > 0.0
        decide_x = audio_s / cost["decide_s"]
        render_x = audio_s / cost["render_s"]
        total_x = audio_s / (cost["decide_s"] + cost["render_s"])
        with capsys.disabled():
            print(f"\n  lead planner over {len(bars)} bars / {audio_s:.1f} "
                  f"audio-s:\n"
                  f"    decide (v2, NEW): {cost['decide_s'] * 1e3:.1f} ms "
                  f"= {decide_x:.0f}x real time "
                  f"({cost['decide_s'] / len(bars) * 1e6:.0f} us/bar)\n"
                  f"    render (v1 duet): {cost['render_s'] * 1e3:.1f} ms "
                  f"= {render_x:.0f}x real time\n"
                  f"    combined:         {total_x:.0f}x real time")
        # The addition must be nowhere near the budget: 1000x means the whole
        # negotiation costs under a millisecond per bar.
        assert decide_x > 1000.0, (
            f"v2's per-bar decisions run at only {decide_x:.0f}x real time")
        # And the whole planner still has to fit a bar inside a bar, with
        # room for the analysis thread beside it.
        assert total_x > 10.0, f"lead planning is only {total_x:.0f}x real time"


# ================================== v1.5.2: --input / --count-in wiring
#
# Acoustic mode and the count-in existed and NOTHING could select them: cli,
# sim and realtime all constructed BeatTracker() with defaults.  These tests
# hold the selection open from every entry point.  The discriminator behind
# ``--input auto`` (and its confusion matrix over both corpora) is held in
# tests/test_acoustic.py; here it is the plumbing.

def _acoustic_wav(tmp_path, bars=4) -> str:
    """A small sustained-acoustic WAV (the material the P1 is about)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_acoustic import make_acoustic
    audio, _ = make_acoustic("fingerpick", bpm=84.0, bars=bars)
    p = tmp_path / "fingerpick.wav"
    sf.write(str(p), audio, SR)
    return str(p)


class _RecordingTracker:
    """Subclass factory: a BeatTracker that records its construction kwargs."""

    @staticmethod
    def make(calls):
        from plume.listen.beat import BeatTracker

        class Rec(BeatTracker):
            def __init__(self, **kw):
                calls.append(dict(kw))
                super().__init__(**kw)
        return Rec


class TestInputModeHelpIsOneTable:
    def _sub(self):
        parser = cli._build_parser()
        import argparse as ap
        return next(a for a in parser._actions
                    if isinstance(a, ap._SubParsersAction)).choices

    def test_the_four_verbs_share_one_input_flag_verbatim(self):
        """sim/jam/listen/lead --input: same choices, same help, and both are
        DERIVED from cli._INPUT_MODE_TABLE — no second copy to drift."""
        subs = self._sub()
        seen = {}
        for verb in ("sim", "jam", "listen", "lead"):
            act = next(a for a in subs[verb]._actions
                       if "--input" in getattr(a, "option_strings", ()))
            seen[verb] = (act.help, tuple(act.choices), act.dest)
        assert len(set(seen.values())) == 1, f"--input drifted: {seen}"
        help_text, choices, dest = seen["sim"]
        assert help_text == cli._input_mode_help()
        assert choices == cli._input_mode_choices() == ("auto", "acoustic",
                                                        "programmed")
        assert dest == "input_mode"

    def test_count_in_is_on_the_live_verbs_only(self):
        subs = self._sub()
        for verb in ("jam", "lead"):
            act = next(a for a in subs[verb]._actions
                       if "--count-in" in getattr(a, "option_strings", ()))
            assert act.help == cli._COUNT_IN_HELP
        for verb in ("sim", "listen"):
            assert not any("--count-in" in getattr(a, "option_strings", ())
                           for a in subs[verb]._actions), (
                f"{verb} grew --count-in; it has --taps / no live taps")


class TestInputModeReachesEveryEntryPoint:
    """Construct-and-inspect: the tracker each verb builds is in the mode the
    flag named.  This is the exact defect class the round is about — the mode
    existed, the constructors never heard about it."""

    def test_sim_constructs_an_acoustic_tracker(self, tmp_path, capsys,
                                                monkeypatch):
        import plume.sim as sim_mod
        calls = []
        monkeypatch.setattr(sim_mod, "BeatTracker",
                            _RecordingTracker.make(calls))
        wav = _acoustic_wav(tmp_path)
        code, out, err = run_main(
            ["sim", wav, "--out", str(tmp_path / "d.wav"),
             "--input", "acoustic", "--json"], capsys, monkeypatch)
        assert code == 0, err
        assert calls and calls[-1].get("acoustic") is True
        doc = json.loads(out)
        assert doc["manifest"]["input_mode_resolved"] == "acoustic"
        assert "--input acoustic" in doc["manifest"]["input_mode_reason"]

    def test_sim_auto_probes_the_file(self, tmp_path, capsys, monkeypatch):
        """Bare `plume sim take.wav` on sustained-acoustic material now
        selects the acoustic front end by measurement — the P1's point."""
        import plume.sim as sim_mod
        calls = []
        monkeypatch.setattr(sim_mod, "BeatTracker",
                            _RecordingTracker.make(calls))
        monkeypatch.delenv("PLUME_ACOUSTIC", raising=False)
        wav = _acoustic_wav(tmp_path)
        code, out, err = run_main(
            ["sim", wav, "--out", str(tmp_path / "d.wav"), "--json"],
            capsys, monkeypatch)
        assert code == 0, err
        assert calls and calls[-1].get("acoustic") is True
        doc = json.loads(out)
        assert doc["manifest"]["input_mode"] == "auto"
        assert doc["manifest"]["input_mode_resolved"] == "acoustic"
        assert "signature" in doc["manifest"]["input_mode_reason"]

    def test_sim_programmed_beats_the_environment(self, tmp_path, capsys,
                                                  monkeypatch):
        import plume.sim as sim_mod
        calls = []
        monkeypatch.setattr(sim_mod, "BeatTracker",
                            _RecordingTracker.make(calls))
        monkeypatch.setenv("PLUME_ACOUSTIC", "1")
        wav = _acoustic_wav(tmp_path)
        code, _out, err = run_main(
            ["sim", wav, "--out", str(tmp_path / "d.wav"),
             "--input", "programmed", "--json"], capsys, monkeypatch)
        assert code == 0, err
        assert calls and calls[-1].get("acoustic") is False

    def test_listen_constructs_an_acoustic_tracker(self, tmp_path, capsys,
                                                   monkeypatch):
        import plume.listen.beat as beat_mod
        calls = []
        monkeypatch.setattr(beat_mod, "BeatTracker",
                            _RecordingTracker.make(calls))
        wav = _acoustic_wav(tmp_path)
        code, out, err = run_main(
            ["listen", wav, "--input", "acoustic", "--json"],
            capsys, monkeypatch)
        assert code == 0, err
        assert calls and calls[-1].get("acoustic") is True
        doc = json.loads(out)
        assert doc["input_mode"]["input_mode_resolved"] == "acoustic"
        # ...and the selection is the first event on the NDJSON stream too.
        assert doc["events"][0]["event"] == "input_mode"
        assert doc["events"][0]["mode"] == "acoustic"

    def test_listen_auto_probes_the_file(self, tmp_path, capsys, monkeypatch):
        import plume.listen.beat as beat_mod
        calls = []
        monkeypatch.setattr(beat_mod, "BeatTracker",
                            _RecordingTracker.make(calls))
        monkeypatch.delenv("PLUME_ACOUSTIC", raising=False)
        wav = _acoustic_wav(tmp_path)
        code, out, err = run_main(["listen", wav, "--json"], capsys,
                                  monkeypatch)
        assert code == 0, err
        assert calls and calls[-1].get("acoustic") is True
        assert json.loads(out)["input_mode"]["input_mode_resolved"] == \
            "acoustic"


class TestLiveInputWiring:
    """jam/lead: kwargs when the transport has grown them, the environment
    hatch until then, and NOTHING when no flag was given."""

    V1_KEYS = {"style", "seed", "params", "device", "monitor", "record_dir",
               "tightness"}

    def _stub_jam_v1(self, seen):
        """A stub with realtime.jam's EXACT current signature — no
        input_mode, no count_in, no **kw — that builds the tracker the real
        transport would build."""
        def jam(*, style, seed, params, device=None, monitor=False,
                record_dir=None, tightness=None):
            from plume.listen.beat import BeatTracker
            bt = BeatTracker()
            seen["acoustic"] = bt._acoustic
            seen["count_in_armed"] = bt._countin is not None
        return jam

    def test_bare_jam_touches_nothing(self, capsys, monkeypatch):
        monkeypatch.delenv("PLUME_ACOUSTIC", raising=False)
        monkeypatch.delenv("PLUME_COUNT_IN", raising=False)
        seen = {}
        monkeypatch.setattr(realtime, "check_device", lambda dev=None: None)
        monkeypatch.setattr(realtime, "jam", self._stub_jam_v1(seen))
        code, _, err = run_main(["jam"], capsys, monkeypatch)
        assert code == 0, err
        assert seen == {"acoustic": False, "count_in_armed": False}

    def test_jam_acoustic_and_count_in_reach_the_tracker_via_env(
            self, capsys, monkeypatch):
        """The transport has not grown the kwargs yet: the CLI must still
        make both reachable, through the construction-site environment
        hatch — and restore the environment afterwards."""
        monkeypatch.delenv("PLUME_ACOUSTIC", raising=False)
        monkeypatch.delenv("PLUME_COUNT_IN", raising=False)
        seen = {}
        monkeypatch.setattr(realtime, "check_device", lambda dev=None: None)
        monkeypatch.setattr(realtime, "jam", self._stub_jam_v1(seen))
        code, _, err = run_main(["jam", "--input", "acoustic", "--count-in"],
                                capsys, monkeypatch)
        assert code == 0, err
        assert seen == {"acoustic": True, "count_in_armed": True}
        assert "PLUME_ACOUSTIC" not in os.environ
        assert "PLUME_COUNT_IN" not in os.environ

    def test_jam_programmed_beats_the_environment(self, capsys, monkeypatch):
        monkeypatch.setenv("PLUME_ACOUSTIC", "1")
        seen = {}
        monkeypatch.setattr(realtime, "check_device", lambda dev=None: None)
        monkeypatch.setattr(realtime, "jam", self._stub_jam_v1(seen))
        code, _, err = run_main(["jam", "--input", "programmed"], capsys,
                                monkeypatch)
        assert code == 0, err
        assert seen["acoustic"] is False
        assert os.environ["PLUME_ACOUSTIC"] == "1", "environment not restored"

    def test_jam_auto_on_a_live_device_hints_not_guesses(self, capsys,
                                                         monkeypatch):
        """auto cannot probe a device that is not open: keep programmed and
        SAY SO (a wrong silent guess is worse than a hint)."""
        monkeypatch.delenv("PLUME_ACOUSTIC", raising=False)
        seen = {}
        monkeypatch.setattr(realtime, "check_device", lambda dev=None: None)
        monkeypatch.setattr(realtime, "jam", self._stub_jam_v1(seen))
        code, _, err = run_main(["jam", "--input", "auto"], capsys,
                                monkeypatch)
        assert code == 0
        assert seen["acoustic"] is False
        assert "--input acoustic" in err, "the hint never reached stderr"

    def test_jam_passes_kwargs_once_the_transport_accepts_them(self, capsys,
                                                               monkeypatch):
        """The forward interface (specified for the realtime integrator):
        jam(input_mode=..., count_in=...).  A transport that accepts **kw is
        the 'has grown them' case; the CLI must prefer kwargs over the
        environment hatch."""
        seen = {}
        monkeypatch.setattr(realtime, "check_device", lambda dev=None: None)
        monkeypatch.setattr(realtime, "jam", lambda **kw: seen.update(kw))
        code, _, err = run_main(["jam", "--input", "acoustic", "--count-in"],
                                capsys, monkeypatch)
        assert code == 0, err
        assert seen["input_mode"] == "acoustic"
        assert seen["count_in"] is True

    def test_bare_jam_still_passes_exactly_the_v1_signature(self, capsys,
                                                            monkeypatch):
        """The original v1 contract, restated beside the new flags: with
        neither given, not one extra kwarg reaches realtime."""
        seen = {}
        monkeypatch.setattr(realtime, "check_device", lambda dev=None: None)
        monkeypatch.setattr(realtime, "jam", lambda **kw: seen.update(kw))
        code, _, err = run_main(["jam"], capsys, monkeypatch)
        assert code == 0, err
        assert set(seen) == self.V1_KEYS

    def test_lead_wiring_kwargs_and_env(self, capsys, monkeypatch):
        monkeypatch.delenv("PLUME_ACOUSTIC", raising=False)
        monkeypatch.delenv("PLUME_COUNT_IN", raising=False)
        # kwargs path (a transport that accepts them)
        seen = {}
        monkeypatch.setattr(realtime, "check_device", lambda dev=None: None)
        monkeypatch.setattr(realtime, "lead", lambda **kw: seen.update(kw))
        code, _, err = run_main(
            ["lead", "x", "--input", "acoustic", "--count-in"], capsys,
            monkeypatch)
        assert code == 0, err
        assert seen["input_mode"] == "acoustic" and seen["count_in"] is True
        # env-hatch path (the transport's exact current signature)
        seen2 = {}

        def lead(*, session=None, prompt=None, style="lofi", seed=0,
                 params=None, role=None, form_params=None, device=None,
                 monitor=False, record_dir=None, tightness=None, from_bar=0,
                 socket_path=None, status=True, max_minutes=None):
            from plume.listen.beat import BeatTracker
            bt = BeatTracker()
            seen2["acoustic"] = bt._acoustic
            seen2["count_in_armed"] = bt._countin is not None
        monkeypatch.setattr(realtime, "lead", lead)
        code, _, err = run_main(
            ["lead", "x", "--input", "acoustic", "--count-in"], capsys,
            monkeypatch)
        assert code == 0, err
        assert seen2 == {"acoustic": True, "count_in_armed": True}
        assert "PLUME_ACOUSTIC" not in os.environ
        assert "PLUME_COUNT_IN" not in os.environ


class TestDoctorInputModeLine:
    def test_doctor_probe_names_the_mode_and_why(self, tmp_path, capsys,
                                                 monkeypatch):
        wav = _acoustic_wav(tmp_path)
        code, out, err = run_main(["doctor", "--probe", wav, "--json"],
                                  capsys, monkeypatch)
        doc = json.loads(out)
        checks = {c["name"]: c for c in doc["checks"]}
        assert "input_mode" in checks
        line = checks["input_mode"]["detail"]
        assert line.startswith("acoustic")
        assert "signature" in line and wav in line

    def test_doctor_without_probe_states_the_live_rule(self, capsys,
                                                       monkeypatch):
        monkeypatch.delenv("PLUME_ACOUSTIC", raising=False)
        code, out, err = run_main(["doctor", "--json"], capsys, monkeypatch)
        doc = json.loads(out)
        checks = {c["name"]: c for c in doc["checks"]}
        detail = checks["input_mode"]["detail"]
        assert "--input acoustic" in detail and "--probe" in detail

    def test_doctor_probe_missing_file_is_an_error_line(self, tmp_path,
                                                        capsys, monkeypatch):
        code, out, err = run_main(
            ["doctor", "--probe", str(tmp_path / "nope.wav"), "--json"],
            capsys, monkeypatch)
        doc = json.loads(out)
        checks = {c["name"]: c for c in doc["checks"]}
        assert checks["input_mode"]["status"] == "error"
        assert code == 1


class TestCountInThroughTheFakeDevice:
    def test_count_in_sets_the_tempo_live_and_holds(self, tmp_path,
                                                    monkeypatch, capsys):
        """END TO END, from the CLI: `plume jam --count-in` over the fake
        duplex device, with a real count-in (four knocks at 84) followed by
        the fingerpicked acoustic fixture — the exact material whose live
        session read 167.9 bpm.

        The CLI reaches the real realtime.jam (only check_device is stubbed:
        there is no consent gate to bypass because no real device ever
        opens), realtime constructs BeatTracker() with defaults, and the
        environment hatch arms the count-in.  The decision log must show the
        counted tempo — 84, not 167.9 — appearing early (the latch, not the
        tracker's own ~5s octave-wrong lock) and never leaving the tap band
        afterwards.
        """
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from conftest import fake_audio, patched_module, wait_until
        from test_acoustic import make_acoustic, prepend_count_in

        monkeypatch.delenv("PLUME_ACOUSTIC", raising=False)
        monkeypatch.delenv("PLUME_COUNT_IN", raising=False)
        monkeypatch.setenv(realtime.SESSION_DIR_ENV,
                           str(tmp_path / "registry"))
        audio, _ = make_acoustic("fingerpick", bpm=84.0, bars=16)
        human, taps = prepend_count_in(audio, 84.0)
        rec = tmp_path / "rec"
        sock = sock_path("countin")
        fake = fake_audio(human, rate_x=12.0)
        monkeypatch.setattr(realtime, "check_device", lambda dev=None: None)

        def stop_soon():
            if wait_until(lambda: Path(sock).exists(), timeout=30.0):
                time.sleep(3.0)                # ~1s warmup + ~24s of take
                try:
                    realtime.send_stop(socket_path=sock)
                except realtime.RealtimeError:
                    pass
        t = threading.Thread(target=stop_soon, daemon=True)
        with patched_module("sounddevice", fake):
            t.start()
            code, _, err = run_main(
                ["jam", "--count-in", "--record", str(rec),
                 "--socket", sock, "--max-minutes", "0"],
                capsys, monkeypatch)
        t.join(timeout=10.0)
        assert code == 0, err
        lines = [json.loads(ln) for ln in
                 (rec / "decisions.ndjson").read_text().splitlines() if ln]
        plans = [d for d in lines if "bar" in d and "bpm" in d]
        assert plans, "no decisions were logged at all"
        in_band = [d for d in plans if abs(d["bpm"] - 84.0) / 84.0 < 0.035]
        assert in_band, (
            "the counted 84 bpm never appeared; the live count-in hatch is "
            f"not armed (tempi seen: {sorted({d['bpm'] for d in plans})})")
        t0 = in_band[0]["t_s"]
        assert t0 < 6.5, (
            f"84 bpm only appeared at t={t0:.1f}s — that is a tracker lock, "
            f"not the count-in latch (taps end at {taps[-1]:.1f}s)")
        after = [d for d in plans if d["t_s"] >= t0]
        off = [d["bpm"] for d in after if abs(d["bpm"] - 84.0) / 84.0 >= 0.035]
        assert not off, f"the latch let go mid-take: {off}"
        # the environment hatch was cleaned up after the session
        assert "PLUME_COUNT_IN" not in os.environ
