"""A live session must be findable and stoppable without hunting for a PID.

Twice during development an automated agent started a live plume session on a
human's machine and left it running, still audible,
stoppable only by finding the process.  ``tests/test_audio_consent.py`` guards
the *start* of that story: nothing opens the speakers unattended.  This file
guards the rest of it:

- a running session says what it is (``plume sessions``),
- it can be ended from any other terminal (``plume stop``),
- and if nobody ever does, it ends itself (``--max-minutes``).

WHAT IS ACTUALLY EXERCISED HERE.  This project's recurring failure is code
that passes synthetic fixtures and dies on real material, so the registry and
the arithmetic get unit tests *and* the whole lifecycle is run end to end
against plume's OWN v0 renders (``engine.render``) at two gains, through the
real ``realtime.lead`` transport — real threads, real cue socket, real
planner, real DuetEngine — with only the audio *device* faked.  A `plume stop`
that works on a mock socket and not on a running band would be exactly the
failure this repo keeps making.

No test here opens a real audio device.
"""
from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from plume import cli                                           # noqa: E402
from plume import mind as mind_mod                              # noqa: E402
from plume import realtime                                      # noqa: E402
from plume import roles as roles_mod                            # noqa: E402
from plume.compose import PromptSpec, compose                   # noqa: E402
from plume.engine import render                                 # noqa: E402

SR = realtime.SR
ENV = realtime.AUDIO_CONSENT_ENV


# ================================================================ helpers

@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Every test gets its own registry — never the developer's real one.

    Without this a test run would list (and could stop) a session the human
    is actually playing.  That would be the very incident this file exists
    for, caused by its own tests.
    """
    d = tmp_path / "reg"
    d.mkdir()
    monkeypatch.setenv(realtime.SESSION_DIR_ENV, str(d))
    return d


def short_sock(name: str) -> str:
    """An AF_UNIX path short enough for the kernel's ~104-byte sun_path.

    pytest's tmp_path routinely blows that limit and the resulting OSError
    looks nothing like "your path is too long", so sockets live directly in
    the temp dir under a short name.
    """
    p = Path(tempfile.gettempdir()) / f"plume-L{os.getpid()}-{name}.sock"
    try:
        p.unlink()
    except OSError:
        pass
    return str(p)


def run_cli(argv, capsys):
    """In-process CLI invocation -> (exit_code, stdout, stderr)."""
    code = cli.main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def run_cli_json(argv, capsys):
    code, out, err = run_cli([*argv, "--json"], capsys)
    return code, json.loads(out), err


DEAD_PID = 2 ** 22          # far above any real pid; never alive


def write_record(reg: Path, pid: int, **kw) -> realtime.LiveSession:
    """Put a record straight on disk, as a running session would."""
    rec = realtime.LiveSession(pid=pid, kind=kw.pop("kind", "lead"), **kw)
    (reg / f"{pid}.json").write_text(json.dumps(rec.to_dict(), sort_keys=True))
    return rec


# ---- a fake audio device: a real v0 render in, the machine's output out ----
#
# v1.3: this harness used to live here, privately, and a near-identical second
# copy lived in test_cli_v2.py.  It now lives in tests/conftest.py, with the
# knobs the P0-B investigation needed (a `stop()` that blocks the way
# Pa_StopStream does, and an `abort()` that can be made to block too), so
# every file and every future round can exercise the live path headlessly.
# That missing shared surface is what let a whole class of live-only defects
# through in the first place.

from conftest import (  # noqa: E402
    fake_audio,
    patched_module,
    wait_until,
)


@pytest.fixture(scope="module")
def material(tmp_path_factory):
    """plume's OWN v0 renders at two gains, plus the session that made them.

    Real material, not a fixture: the governing lesson of this repo is that
    synthetic input hides the defects that matter.
    """
    d = tmp_path_factory.mktemp("lifecycle_material")
    sess = compose(PromptSpec(styles=["lofi"], bpm=80.0, key="C major",
                              movements=1, mood=[], name="lifecycle"), seed=7)
    out = {"session": sess}
    for tag, gain_db in (("loud", 0.0), ("quiet", -20.0)):
        p = d / f"lifecycle_{tag}.wav"
        render(sess, out_path=str(p), movement=0, bars=(0, 8))
        if gain_db:
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            sf.write(str(p), x * (10.0 ** (gain_db / 20.0)), sr,
                     subtype="PCM_24")
        x, _ = sf.read(str(p), dtype="float32", always_2d=True)
        out[tag] = x.mean(axis=1).astype(np.float32)
        out[f"{tag}_path"] = p
    return out


def run_lead(human, *, session, sock, rec=None, max_minutes=None,
             rate_x=12.0, fake=None, **kw):
    """Run the real lead transport over the fake device.  Returns the fake."""
    fake = fake if fake is not None else fake_audio(human, rate_x=rate_x)
    with patched_module("sounddevice", fake):
        realtime.lead(session=session, seed=0,
                      role=roles_mod.RoleParams(role="lead"),
                      record_dir=(None if rec is None else str(rec)),
                      socket_path=sock, status=False,
                      max_minutes=max_minutes, **kw)
    return fake


def run_jam(human, *, sock, rec=None, max_minutes=None, rate_x=12.0,
            fake=None, style="lofi", **kw):
    """Run the real v1 SIDEMAN transport over the fake device.

    v1.3: ``jam`` is the transport the first session with a human and a guitar
    actually used, and it had no live coverage at all — every end-to-end test
    ran ``lead``.  Both of the P0 defects were in code the two share, but the
    one that was reported was reported about ``jam``.
    """
    fake = fake if fake is not None else fake_audio(human, rate_x=rate_x)
    with patched_module("sounddevice", fake):
        realtime.jam(style=style, seed=0, params=mind_mod.MindParams(),
                     record_dir=(None if rec is None else str(rec)),
                     socket_path=sock, max_minutes=max_minutes, **kw)
    return fake


def stop_when_running(sock, *, after=0.4, timeout=30.0):
    """Background thread that waits for the socket, then asks it to stop."""
    done = {}

    def run():
        if not wait_until(lambda: Path(sock).exists(), timeout=timeout):
            return
        time.sleep(after)                 # let it actually play some bars
        try:
            done.update(realtime.send_stop(socket_path=sock))
        except realtime.RealtimeError as e:
            done["error"] = str(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, done


# ================================================ 1. the cap's arithmetic

class TestDurationCap:
    """Pure arithmetic — no clock, no device, no threads."""

    def test_default_is_the_module_constant(self):
        assert realtime.session_deadline(0.0, None) == pytest.approx(
            realtime.DEFAULT_MAX_MINUTES * 60.0)

    def test_zero_disables_the_cap(self):
        """An interactive human explicitly opting out of the belt and braces."""
        assert realtime.session_deadline(1000.0, 0) is None
        assert realtime.session_deadline(1000.0, 0.0) is None
        assert realtime.resolve_max_minutes(0) == 0.0

    def test_deadline_is_start_plus_minutes(self):
        assert realtime.session_deadline(100.0, 2.5) == pytest.approx(250.0)
        assert realtime.session_deadline(-10.0, 1.0) == pytest.approx(50.0)

    def test_is_a_pure_function_of_its_arguments(self):
        """No wall clock in here: same inputs, same answer, minutes apart."""
        a = realtime.session_deadline(12345.0, 7.0)
        time.sleep(0.01)
        b = realtime.session_deadline(12345.0, 7.0)
        assert a == b == pytest.approx(12345.0 + 420.0)

    @pytest.mark.parametrize("bad", [-1, -0.001, float("nan"), float("inf")])
    def test_nonsense_caps_are_refused(self, bad):
        with pytest.raises(ValueError):
            realtime.resolve_max_minutes(bad)

    def test_cli_rejects_a_negative_cap(self, capsys):
        code, _out, err = run_cli(["jam", "--max-minutes", "-5"], capsys)
        assert code == 1
        assert "--max-minutes" in err

    def test_stop_reason_fires_exactly_at_the_deadline(self):
        """The predicate the transports poll, tested without a transport."""
        assert realtime._live_stop_reason(None, 100.0, now=99.999) is None
        assert (realtime._live_stop_reason(None, 100.0, now=100.0)
                == realtime.END_MAX_MINUTES)
        assert realtime._live_stop_reason(None, None, now=1e12) is None

    def test_a_stop_request_beats_the_clock(self):
        srv = realtime.CueServer(short_sock("pred"), kind="lead")
        assert realtime._live_stop_reason(srv, None, now=0.0) is None
        srv.stopped.set()
        assert (realtime._live_stop_reason(srv, None, now=0.0)
                == realtime.END_STOPPED)


# ==================================================== 2. the registry

class TestRegistry:
    def test_write_read_roundtrip(self, isolated_registry):
        rec = realtime.register_session(kind="lead", style="trap", role="lead",
                                        session_name="demo", socket_path="/x.sock",
                                        max_minutes=30.0)
        assert rec is not None and rec.pid == os.getpid()
        assert rec.path.exists()
        [back] = realtime.list_sessions()
        assert back == rec
        assert back.style == "trap" and back.max_minutes == 30.0
        realtime.unregister_session()
        assert realtime.list_sessions() == []

    def test_missing_registry_dir_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv(realtime.SESSION_DIR_ENV, str(tmp_path / "nope"))
        assert realtime.list_sessions() == []

    def test_a_dead_pid_is_pruned_silently(self, isolated_registry):
        write_record(isolated_registry, DEAD_PID, style="lofi")
        assert (isolated_registry / f"{DEAD_PID}.json").exists()
        assert realtime.list_sessions() == []
        assert not (isolated_registry / f"{DEAD_PID}.json").exists()

    def test_a_live_pid_survives_the_prune(self, isolated_registry):
        write_record(isolated_registry, os.getpid(), style="lofi")
        write_record(isolated_registry, DEAD_PID, style="lofi")
        [alive] = realtime.list_sessions()
        assert alive.pid == os.getpid()

    def test_garbage_and_empty_files_are_pruned_not_raised(self,
                                                           isolated_registry):
        (isolated_registry / "999999.json").write_text("{not json")
        (isolated_registry / "888888.json").write_text("")
        (isolated_registry / "777777.json").write_text('["a list"]')
        assert realtime.list_sessions() == []
        assert list(isolated_registry.glob("*.json")) == []

    def test_records_are_written_atomically(self, isolated_registry):
        """A lister must never catch a half-written record."""
        realtime.register_session(kind="jam")
        assert list(isolated_registry.glob("*.tmp")) == []

    def test_unregister_is_idempotent(self, isolated_registry):
        realtime.unregister_session()
        realtime.unregister_session(DEAD_PID)      # never raises

    def test_registration_failure_does_not_raise(self, tmp_path, monkeypatch):
        """Bookkeeping must never be able to take the band down."""
        blocked = tmp_path / "blocked"
        blocked.write_text("i am a file, not a directory")
        monkeypatch.setenv(realtime.SESSION_DIR_ENV, str(blocked))
        assert realtime.register_session(kind="jam") is None
        assert realtime.list_sessions() == []

    def test_uptime_and_remaining_are_functions_of_the_given_now(self):
        rec = realtime.LiveSession(pid=1, kind="lead", started_at=1000.0,
                                   max_minutes=10.0)
        assert rec.uptime_s(now=1720.0) == pytest.approx(720.0)
        assert rec.remaining_s(now=1720.0) == pytest.approx(-120.0)
        assert rec.uptime_s(now=0.0) == 0.0          # clamped, never negative

    def test_uncapped_record_has_no_deadline(self):
        rec = realtime.LiveSession(pid=1, kind="lead", max_minutes=0.0)
        assert rec.deadline() is None
        assert rec.remaining_s(now=1e9) is None

    @pytest.mark.parametrize("secs,text", [(0, "0s"), (38, "38s"), (59, "59s"),
                                           (60, "1m"), (725, "12m"),
                                           (3600, "1h00m"), (3900, "1h05m")])
    def test_uptime_formatting(self, secs, text):
        assert realtime.format_uptime(secs) == text

    def test_stop_command_names_the_pid(self):
        rec = realtime.LiveSession(pid=4212, kind="lead")
        assert rec.stop_command == "plume stop --pid 4212"


# ================================================ 3. the stop ladder

class TestStopSession:
    def test_stopping_an_already_dead_session_is_clean(self, isolated_registry):
        rec = write_record(isolated_registry, DEAD_PID, socket=None)
        out = realtime.stop_session(rec, timeout=0.5)
        assert out["stopped"] is True
        assert out["how"] == "already-gone"
        assert not (isolated_registry / f"{DEAD_PID}.json").exists()

    def test_the_socket_verb_reaches_a_live_server(self):
        """{"op": "stop"} sets the flag the transports poll — no audio here."""
        path = short_sock("verb")
        with realtime.CueServer(path, kind="lead") as srv:
            assert not srv.stopped.is_set()
            resp = realtime.send_stop(socket_path=path)
            assert resp["ok"] is True and resp["stopping"] is True
            assert resp["pid"] == os.getpid()
            for _ in range(100):
                if srv.stopped.is_set():
                    break
                time.sleep(0.01)
            assert srv.stopped.is_set()

    def test_the_status_verb_reports_the_session(self):
        path = short_sock("status")
        with realtime.CueServer(path, kind="jam", accepts_cues=False) as srv:
            srv.bar = 12
            doc = realtime.send_status(socket_path=path)
            assert doc["session"] == "jam" and doc["bar"] == 12
            assert doc["accepts_cues"] is False
            assert doc["pid"] == os.getpid()

    def test_a_control_verb_never_touches_the_cue_queue(self):
        path = short_sock("nocue")
        with realtime.CueServer(path, kind="lead") as srv:
            realtime.send_stop(socket_path=path)
            realtime.send_status(socket_path=path)
            assert srv.drain() == []
            assert srv.received == 0

    def test_cues_still_work_exactly_as_before(self):
        """The control verbs are additive: v2.0's cue replies are unchanged."""
        path = short_sock("cuestill")
        with realtime.CueServer(path, kind="lead") as srv:
            resp = realtime.send_cue("go to the chorus", socket_path=path)
            assert resp["ok"] is True and resp["queued"] == 1
            assert srv.drain() == [{"text": "go to the chorus",
                                    "at": "next-phrase", "ttl_bars": 16}]

    def test_the_sideman_refuses_cues_but_accepts_stop(self):
        """A control socket is not a cue socket."""
        path = short_sock("sideman")
        with realtime.CueServer(path, kind="jam", accepts_cues=False) as srv:
            with pytest.raises(realtime.RealtimeError) as exc:
                realtime.send_cue("darker", socket_path=path)
            assert "no form cursor" in str(exc.value)
            realtime.send_stop(socket_path=path)
            assert srv.stopped.is_set()

    def test_an_unknown_op_is_refused_not_crashed(self):
        """A bad verb comes back to its sender; the band plays on."""
        path = short_sock("badop")
        with realtime.CueServer(path, kind="lead") as srv:
            resp = realtime._socket_request(path, {"op": "explode"},
                                            timeout=2.0, absent_msg="unused")
            assert resp["ok"] is False
            assert "explode" in resp["error"]
            assert not srv.stopped.is_set()
            # and the server is still serving
            assert realtime.send_status(socket_path=path)["ok"] is True

    def test_send_stop_says_plainly_when_nothing_is_listening(self):
        with pytest.raises(realtime.RealtimeError) as exc:
            realtime.send_stop(socket_path=short_sock("absent"))
        assert "nothing is listening" in str(exc.value)

    def test_a_dead_socket_falls_back_to_signals(self, isolated_registry,
                                                 monkeypatch):
        """A session whose socket never opened is still stoppable.

        The registry says it is running and the socket answers nothing, so the
        ladder must move on to SIGINT rather than give up at rung 1.
        """
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        rec = write_record(isolated_registry, DEAD_PID,
                           socket=short_sock("ghost"))
        out = realtime.stop_session(rec, timeout=0.5)
        assert out["stopped"] is True
        assert any("socket-failed" in t for t in out["tried"]), out["tried"]
        assert not (isolated_registry / f"{DEAD_PID}.json").exists()

    def test_stop_never_signals_its_own_process(self, isolated_registry,
                                                monkeypatch):
        """plume is a library too: `stop` must not SIGTERM the caller.

        Without this guard an embedded session asked to stop itself would
        walk the ladder straight onto its own pid.
        """
        killed = []      # signal 0 is the liveness probe, not a kill
        monkeypatch.setattr(os, "kill",
                            lambda pid, sig: sig and killed.append(sig))
        path = short_sock("selfstop")
        with realtime.CueServer(path, kind="lead") as srv:
            rec = write_record(isolated_registry, os.getpid(), socket=path)
            out = realtime.stop_session(rec, timeout=0.2, poll=0.02)
            assert out["stopped"] is True and out["how"] == "socket"
            assert srv.stopped.is_set()
        assert killed == [], "stop signalled its own process"

    def test_self_stop_without_a_socket_says_so_rather_than_signalling(
            self, isolated_registry, monkeypatch):
        killed = []      # signal 0 is the liveness probe, not a kill
        monkeypatch.setattr(os, "kill",
                            lambda pid, sig: sig and killed.append(sig))
        rec = write_record(isolated_registry, os.getpid(), socket=None)
        out = realtime.stop_session(rec, timeout=0.2, poll=0.02)
        assert out["stopped"] is False and out["how"] == "self"
        assert "another terminal" in out["detail"]
        assert killed == []

    def test_the_default_ladder_never_reaches_for_sigkill(self):
        """SIGKILL cuts the tail, so it is never automatic — only --force."""
        assert "SIGKILL" not in realtime._STOP_SIGNALS
        assert realtime._STOP_SIGNALS == ("SIGINT", "SIGTERM")

    def test_force_adds_a_sigkill_rung_and_says_it_forced(
            self, isolated_registry, monkeypatch):
        """The deliberate escalation.

        It costs far less than it used to: since v1.3 the take is streamed to
        disk as it happens, so a forced stop loses the tail and ending.json,
        not the recording.  That is the whole reason it can be offered at all.
        """
        sent = []
        monkeypatch.setattr(os, "kill",
                            lambda pid, sig: sig and sent.append(sig))
        # alive until SIGKILL lands, then gone
        monkeypatch.setattr(realtime, "_pid_alive",
                            lambda pid: signal.SIGKILL not in sent)
        rec = write_record(isolated_registry, 4242, socket=None)
        out = realtime.stop_session(rec, timeout=0.3, poll=0.02, force=True)
        assert out["stopped"] is True
        assert out["how"] == "signal:SIGKILL"
        assert out["forced"] is True
        assert out["tried"] == ["SIGINT", "SIGTERM", "SIGKILL"]
        assert "the tail did not play" in out["detail"]

    def test_without_force_nothing_sends_sigkill(self, isolated_registry,
                                                 monkeypatch):
        sent = []
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        monkeypatch.setattr(os, "kill",
                            lambda pid, sig: sig and sent.append(sig))
        rec = write_record(isolated_registry, 12346, socket=None)
        out = realtime.stop_session(rec, timeout=0.2, poll=0.02)
        assert signal.SIGKILL not in sent
        assert out["stopped"] is False

    def test_sigint_gets_the_whole_graceful_budget_not_a_fraction(
            self, isolated_registry, monkeypatch):
        """Regression: SIGTERM used to land while the flush was still running.

        A cross-process check caught this — `plume stop` on a session with a
        dead socket sent SIGINT, waited a *fraction* of --timeout, decided it
        had failed, and SIGTERMed a process that was in the middle of writing
        its recording.  The take was destroyed by the thing whose whole job is
        not to destroy takes.  SIGINT is the graceful path and must get the
        same patience as the socket rung.
        """
        sent, t_sent = [], {}
        # The session takes 4s to finish its ending, then exits.
        monkeypatch.setattr(os, "kill", lambda pid, sig: (
            sig and (sent.append(sig), t_sent.setdefault(sig, time.monotonic()))))
        monkeypatch.setattr(
            realtime, "_pid_alive",
            lambda pid: (2 not in t_sent      # SIGINT == 2
                         or time.monotonic() - t_sent[2] < 4.0))
        rec = write_record(isolated_registry, 4242, socket=None)
        out = realtime.stop_session(rec, timeout=20.0, poll=0.05)
        assert out["stopped"] is True
        assert out["how"] == "signal:SIGINT", out
        assert sent == [2], f"SIGTERM landed on a working ending: {sent}"

    def test_the_default_budget_outlasts_the_transports_own_ending(self):
        """The graceful ending is bounded — and this budget contains it.

        v1.3 made the bound real (it used to end in an UNBOUNDED PortAudio
        close), so this can now compare against the transport's own number
        instead of a comment's estimate of it.
        """
        assert realtime.DEFAULT_STOP_TIMEOUT > realtime.ENDING_HARD_DEADLINE_S
        assert realtime.DEFAULT_STOP_TIMEOUT > 40.0

    def test_a_wedged_session_is_reported_not_silently_left(self,
                                                            isolated_registry,
                                                            monkeypatch):
        """Signals land but the process never dies: say so, name the command.

        v1.3 changed WHICH command it names.  ``kill -9`` was the honest
        advice when a forced kill destroyed the take; now that the take is on
        disk already, the thing to reach for is ``plume stop --force``, which
        does the same escalation and still unregisters the session.
        """
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        monkeypatch.setattr(os, "kill", lambda pid, sig: None)
        rec = write_record(isolated_registry, 12345, socket=None)
        out = realtime.stop_session(rec, timeout=0.2, poll=0.02)
        assert out["stopped"] is False and out["how"] == "failed"
        assert "--pid 12345 --force" in out["detail"]
        assert out["tried"] == ["SIGINT", "SIGTERM"]

    def test_force_that_also_fails_says_what_that_means(self,
                                                        isolated_registry,
                                                        monkeypatch):
        """Even SIGKILL bouncing is a real state, and it means something
        specific (uninterruptible kernel I/O) — say that, do not repeat the
        advice that just failed."""
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        monkeypatch.setattr(os, "kill", lambda pid, sig: None)
        rec = write_record(isolated_registry, 12347, socket=None)
        out = realtime.stop_session(rec, timeout=0.2, poll=0.02, force=True)
        assert out["stopped"] is False and out["forced"] is True
        assert "uninterruptible" in out["detail"]
        assert "--force" not in out["detail"], "it re-suggested what just failed"


# ================================ 3b. signals become the graceful ending

def _wait_for(event, seconds=1.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if event.is_set():
            return True
        time.sleep(0.01)
    return event.is_set()


class TestGracefulSignals:
    """The rung that a backgrounded session used to have no answer for."""

    def test_sigint_sets_the_ending_flag_instead_of_raising(self):
        with realtime.GracefulSignals() as g:
            assert not g.event.is_set()
            os.kill(os.getpid(), signal.SIGINT)
            assert _wait_for(g.event)
            assert g.reason == realtime.END_INTERRUPT

    def test_sigterm_is_graceful_too_and_says_which_signal(self):
        with realtime.GracefulSignals() as g:
            os.kill(os.getpid(), signal.SIGTERM)
            assert _wait_for(g.event)
            assert g.reason == realtime.END_TERMINATED

    def test_handlers_are_restored_on_exit(self):
        before = {s: signal.getsignal(getattr(signal, s))
                  for s in realtime.GracefulSignals.SIGNALS}
        with realtime.GracefulSignals():
            during = {s: signal.getsignal(getattr(signal, s))
                      for s in realtime.GracefulSignals.SIGNALS}
        after = {s: signal.getsignal(getattr(signal, s))
                 for s in realtime.GracefulSignals.SIGNALS}
        assert during != before, "no handler was installed"
        assert after == before, "the session did not put the handlers back"

    def test_an_inherited_SIG_IGN_is_overridden(self):
        """The actual defect.

        A live session started in the background by a non-interactive shell
        inherits SIGINT as SIG_IGN, and CPython does not override an inherited
        SIG_IGN — measured directly, not assumed.  Those sessions ignored
        Ctrl-C's signal entirely, so `plume stop`'s graceful rung was dead for
        exactly the agent-started, backgrounded, unattended sessions the whole
        feature exists for, and the only thing that landed was a SIGTERM that
        destroyed the take.
        """
        old = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
            with realtime.GracefulSignals() as g:
                assert signal.getsignal(signal.SIGINT) is not signal.SIG_IGN
                os.kill(os.getpid(), signal.SIGINT)
                assert _wait_for(g.event), "SIGINT was still being ignored"
            assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
        finally:
            signal.signal(signal.SIGINT, old)

    def test_a_second_signal_stands_aside_so_a_user_can_always_get_out(self):
        """A graceful ending must never become a trap."""
        with realtime.GracefulSignals() as g:
            os.kill(os.getpid(), signal.SIGTERM)
            assert _wait_for(g.event)
            g._handle(signal.SIGTERM, None)      # the impatient second press
            assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
            signal.signal(signal.SIGTERM, g._handle)   # put it back for __exit__

    def test_installing_off_the_main_thread_degrades_quietly(self):
        """signal.signal() only works on the main thread; a session must play."""
        ok = []

        def worker():
            with realtime.GracefulSignals() as g:
                ok.append(not g.event.is_set())

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5.0)
        assert ok == [True]

    def test_the_predicate_prefers_a_signal_over_the_clock(self):
        g = realtime.GracefulSignals()
        assert realtime._live_stop_reason(None, None, signals=g, now=0.0) is None
        g.reason = realtime.END_TERMINATED
        g.event.set()
        assert (realtime._live_stop_reason(None, 1e12, signals=g, now=0.0)
                == realtime.END_TERMINATED)


# ============================================ 4. `plume stop` / `sessions`

class TestStopCli:
    def test_nothing_running_is_a_clean_message_and_exit_zero(self, capsys):
        """Quiet is the healthy answer, not a failure — wrappers depend on it."""
        code, out, err = run_cli(["stop"], capsys)
        assert code == 0, err
        assert "nothing to stop" in out
        assert "no live plume sessions" in out
        assert err == ""

    def test_nothing_running_json_shape(self, capsys):
        code, doc, _ = run_cli_json(["stop"], capsys)
        assert code == 0
        assert doc == {"stopped": [], "count": 0, "failed": 0}

    def test_stop_json_shape_for_a_real_stop(self, capsys, isolated_registry,
                                             monkeypatch):
        # Listed as running, but the signal finds nothing: the ladder's
        # "it exited on its own" rung, which is fast and touches no process.
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        write_record(isolated_registry, DEAD_PID, kind="lead", style="trap",
                     role="lead", started_at=time.time() - 725.0)
        code, doc, _ = run_cli_json(["stop", "--all"], capsys)
        assert code == 0
        assert set(doc) == {"stopped", "count", "failed"}
        assert doc["count"] == 1 and doc["failed"] == 0
        [row] = doc["stopped"]
        for key in ("pid", "kind", "style", "role", "session", "uptime",
                    "uptime_s", "stopped", "how", "detail"):
            assert key in row, key
        assert row["pid"] == DEAD_PID and row["kind"] == "lead"
        assert row["uptime"] == "12m"

    def test_human_line_names_kind_pid_and_uptime(self, capsys,
                                                  isolated_registry,
                                                  monkeypatch):
        """'stopped: lead session, pid N, running 12m' — the whole point."""
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        write_record(isolated_registry, DEAD_PID, kind="lead",
                     started_at=time.time() - 725.0)
        code, out, _ = run_cli(["stop"], capsys)
        assert code == 0
        assert f"stopped: lead session, pid {DEAD_PID}, running 12m" in out

    def test_several_sessions_without_all_refuses_and_lists_them(
            self, capsys, isolated_registry, monkeypatch):
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        write_record(isolated_registry, 101, kind="lead", style="trap")
        write_record(isolated_registry, 102, kind="jam", style="lofi")
        code, _out, err = run_cli(["stop"], capsys)
        assert code == 1
        assert "2 live sessions" in err
        assert "pid 101" in err and "pid 102" in err
        assert "--all" in err

    def test_all_stops_every_session(self, capsys, isolated_registry,
                                     monkeypatch):
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        write_record(isolated_registry, DEAD_PID, kind="lead")
        write_record(isolated_registry, DEAD_PID + 1, kind="jam")
        code, doc, _ = run_cli_json(["stop", "--all"], capsys)
        assert code == 0 and doc["count"] == 2

    def test_pid_targets_exactly_one(self, capsys, isolated_registry,
                                     monkeypatch):
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: pid == 101)
        write_record(isolated_registry, 101, kind="lead")
        monkeypatch.setattr(realtime, "stop_session",
                            lambda rec, **kw: {"pid": rec.pid, "kind": rec.kind,
                                               "style": "", "role": "",
                                               "session": None, "uptime": "1s",
                                               "uptime_s": 1.0,
                                               "record_dir": None,
                                               "stopped": True, "how": "socket",
                                               "detail": "ok"})
        code, doc, _ = run_cli_json(["stop", "--pid", "101"], capsys)
        assert code == 0 and doc["stopped"][0]["pid"] == 101

    def test_an_unknown_pid_errors_and_says_what_is_running(
            self, capsys, isolated_registry, monkeypatch):
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: pid == 101)
        write_record(isolated_registry, 101, kind="lead")
        code, _out, err = run_cli(["stop", "--pid", "999"], capsys)
        assert code == 1
        assert "999" in err and "101" in err
        assert "plume sessions" in err

    def test_a_session_that_refuses_to_die_exits_nonzero(
            self, capsys, isolated_registry, monkeypatch):
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        write_record(isolated_registry, 101, kind="lead")
        monkeypatch.setattr(realtime, "stop_session",
                            lambda rec, **kw: {"pid": rec.pid, "kind": "lead",
                                               "style": "", "role": "",
                                               "session": None, "uptime": "2h00m",
                                               "uptime_s": 7200.0,
                                               "record_dir": None,
                                               "stopped": False, "how": "failed",
                                               "detail": "wedged; kill -9 101"})
        code, out, _ = run_cli(["stop"], capsys)
        assert code == 1
        assert "COULD NOT STOP" in out and "kill -9" in out

    # -- v1.7: the escalation the wedge message names ---------------------

    def test_the_cli_accepts_the_force_flag_its_own_message_advises(
            self, capsys, isolated_registry, monkeypatch):
        """``stop_session`` grew ``force=`` and the wedged detail line tells
        the player to run ``plume stop --pid N --force`` — but the parser had
        no such flag, so following plume's own advice printed
        ``unrecognized arguments: --force`` and the take stayed wedged.  The
        library half of P0-3 shipped without the half a human can reach.
        """
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        write_record(isolated_registry, 101, kind="lead")
        seen = {}

        def spy(rec, **kw):
            seen.update(kw)
            return {"pid": rec.pid, "kind": "lead", "style": "", "role": "",
                    "session": None, "uptime": "1s", "uptime_s": 1.0,
                    "record_dir": None, "stopped": True,
                    "how": "signal:SIGKILL", "forced": True, "detail": "ok"}

        monkeypatch.setattr(realtime, "stop_session", spy)
        code, doc, err = run_cli_json(["stop", "--force"], capsys)
        assert code == 0, err
        assert seen.get("force") is True, (
            f"--force did not reach stop_session (got {seen!r})")
        assert doc["stopped"][0]["forced"] is True

    def test_without_the_flag_the_cli_never_forces(self, capsys,
                                                   isolated_registry,
                                                   monkeypatch):
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        write_record(isolated_registry, 101, kind="lead")
        seen = {}

        def spy(rec, **kw):
            seen.update(kw)
            return {"pid": rec.pid, "kind": "lead", "style": "", "role": "",
                    "session": None, "uptime": "1s", "uptime_s": 1.0,
                    "record_dir": None, "stopped": True, "how": "socket",
                    "detail": "ok"}

        monkeypatch.setattr(realtime, "stop_session", spy)
        code, _doc, err = run_cli_json(["stop"], capsys)
        assert code == 0, err
        assert seen.get("force") is False, (
            "the default ladder must never reach for SIGKILL")

    def test_the_advice_in_the_wedge_message_is_a_command_that_parses(
            self, isolated_registry, monkeypatch):
        """The link between the two halves, so they cannot drift apart again.

        Take the command plume ACTUALLY prints to a player whose session is
        wedged, lift the flags out of it, and hand them to the real parser.
        This is the test that would have failed the day the detail line
        started advertising ``--force``.
        """
        import re
        import shlex

        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        monkeypatch.setattr(os, "kill", lambda pid, sig: None)
        rec = write_record(isolated_registry, 12346, socket=None)
        out = realtime.stop_session(rec, timeout=0.2, poll=0.02)
        assert out["stopped"] is False

        m = re.search(r"'(plume stop [^']+)'", out["detail"])
        assert m, f"the wedge message names no command: {out['detail']!r}"
        argv = shlex.split(m.group(1))[1:]           # drop the 'plume'
        assert "--force" in argv

        from plume.cli import _build_parser
        args = _build_parser().parse_args(argv)      # must not SystemExit
        assert args.pid == 12346 and args.force is True


class TestSessionsCli:
    def test_nothing_running_says_so_and_exits_zero(self, capsys):
        code, out, err = run_cli(["sessions"], capsys)
        assert code == 0, err
        assert "no live plume sessions are running" in out

    def test_json_shape_when_empty(self, capsys):
        code, doc, _ = run_cli_json(["sessions"], capsys)
        assert code == 0
        assert doc["count"] == 0 and doc["sessions"] == []
        assert doc["registry"]

    def test_json_shape_lists_every_field_a_human_needs(
            self, capsys, isolated_registry, monkeypatch):
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        write_record(isolated_registry, 4212, kind="lead", style="trap",
                     role="lead", session_name="demo", socket="/tmp/x.sock",
                     started_at=time.time() - 725.0, max_minutes=60.0,
                     record_dir="takes/")
        code, doc, _ = run_cli_json(["sessions"], capsys)
        assert code == 0 and doc["count"] == 1
        [row] = doc["sessions"]
        for key in ("pid", "kind", "style", "role", "session", "socket",
                    "started_at", "uptime", "uptime_s", "max_minutes",
                    "remaining_s", "record_dir", "stop_command"):
            assert key in row, key
        assert row["pid"] == 4212 and row["kind"] == "lead"
        assert row["uptime"] == "12m"
        assert row["stop_command"] == "plume stop --pid 4212"
        assert row["remaining_s"] == pytest.approx(2875.0, abs=2.0)

    def test_human_output_shows_pid_role_style_uptime_and_the_stop_command(
            self, capsys, isolated_registry, monkeypatch):
        """A human who hears music must be able to find out what is making it."""
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        write_record(isolated_registry, 4212, kind="lead", style="trap",
                     role="lead", session_name="demo",
                     started_at=time.time() - 725.0)
        code, out, _ = run_cli(["sessions"], capsys)
        assert code == 0
        header, *_rest = out.splitlines()
        for col in ("pid", "kind", "role", "style", "up", "ends in", "session"):
            assert col in header
        assert "4212" in out and "lead" in out and "trap" in out
        assert "12m" in out and "demo" in out
        assert "plume stop --pid 4212" in out

    def test_an_uncapped_session_shows_no_deadline(self, capsys,
                                                   isolated_registry,
                                                   monkeypatch):
        monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
        write_record(isolated_registry, 4212, kind="jam", max_minutes=0.0)
        code, doc, _ = run_cli_json(["sessions"], capsys)
        assert doc["sessions"][0]["remaining_s"] is None
        code, out, _ = run_cli(["sessions"], capsys)
        assert code == 0 and " -" in out

    def test_help_table_is_derived_from_the_dataclass(self, capsys):
        """Hand-written tables have drifted here before; this one cannot."""
        import dataclasses

        with pytest.raises(SystemExit):
            run_cli(["sessions", "--help"], capsys)
        out = capsys.readouterr().out
        for f in dataclasses.fields(realtime.LiveSession):
            assert f.name in out, f.name
        assert str(realtime.DEFAULT_MAX_MINUTES) in out

    def test_max_minutes_help_quotes_the_real_default(self, capsys):
        for verb in ("jam", "lead"):
            with pytest.raises(SystemExit):
                run_cli([verb, "--help"], capsys)
            out = capsys.readouterr().out
            assert "--max-minutes" in out
            assert f"{float(realtime.DEFAULT_MAX_MINUTES):g}" in out


# ======================================= 5. the whole thing, on real audio

@pytest.mark.usefixtures("fake_device_consent")
class TestLiveLifecycleOnRealMaterial:
    """``realtime.lead`` for real — real threads, real socket, real planner,
    real DuetEngine — over plume's own v0 renders.  Only the device is fake.
    """

    def test_a_running_session_registers_and_deregisters(self, material,
                                                         isolated_registry,
                                                         tmp_path):
        seen: list[list] = []
        sock = short_sock("reg1")

        def watch():
            for _ in range(2000):
                rows = realtime.list_sessions()
                if rows:
                    seen.append(rows)
                    realtime.send_stop(socket_path=sock)
                    return
                time.sleep(0.01)

        t = threading.Thread(target=watch, daemon=True)
        rec = tmp_path / "rec"
        t.start()
        run_lead(material["loud"], session=material["session"], sock=sock,
                 rec=rec, max_minutes=0)
        t.join(timeout=5.0)

        assert seen, "a running lead session was never listed by list_sessions"
        [live] = seen[0]
        assert live.pid == os.getpid()
        assert live.kind == "lead"
        assert live.socket == sock
        assert live.style == "lofi"
        # ...and it cleans up after itself.
        assert realtime.list_sessions() == []
        assert not Path(sock).exists()

    def test_plume_stop_ends_a_real_session_gracefully(self, material,
                                                       isolated_registry,
                                                       tmp_path):
        """The unattended-session fix, end to end: another terminal ends the band, and the
        band still finishes its bar, flushes its tail and writes its records.
        """
        sock = short_sock("stop1")
        rec = tmp_path / "rec"
        stopped: dict = {}

        def stopper():
            for _ in range(3000):
                rows = realtime.list_sessions()
                if rows:
                    time.sleep(0.4)          # let it actually play some bars
                    stopped.update(realtime.stop_session(rows[0], timeout=20.0))
                    return
                time.sleep(0.01)

        t = threading.Thread(target=stopper, daemon=True)
        t.start()
        fake = run_lead(material["loud"], session=material["session"],
                        sock=sock, rec=rec, max_minutes=0)
        t.join(timeout=30.0)

        assert stopped, "the stopper never found the running session"
        assert stopped["how"] == "socket", stopped
        assert stopped["kind"] == "lead"
        # It really made sound before it was stopped...
        out = np.concatenate(fake.stream.captured, axis=0)
        assert float(np.abs(out).max()) > 1e-4, "the machine never sounded"
        # ...and the graceful ending happened: records exist and say why.
        lines = [json.loads(ln) for ln
                 in (rec / "decisions.ndjson").read_text().splitlines() if ln]
        assert len(lines) >= 2
        assert (rec / "human.wav").exists() and (rec / "machine.wav").exists()
        doc = json.loads((rec / "ending.json").read_text())
        assert doc["reason"] == realtime.END_STOPPED
        assert doc["capped"] is False
        assert doc["bars"] == len(lines)

    def test_the_cap_ends_a_real_session_by_itself(self, material,
                                                   isolated_registry, tmp_path):
        """Nobody stops it.  It stops anyway, and the record says the cap did.

        This is the belt-and-braces half of the unattended-session incident: an agent that
        walks away cannot leave music playing.
        """
        sock = short_sock("cap1")
        rec = tmp_path / "rec"
        t0 = time.monotonic()
        fake = run_lead(material["loud"], session=material["session"],
                        sock=sock, rec=rec, max_minutes=3.0 / 60.0)  # 3 seconds
        elapsed = time.monotonic() - t0

        assert elapsed < 60.0, f"the cap did not end the session ({elapsed:.0f}s)"
        assert fake.stream.blocks > 0
        out = np.concatenate(fake.stream.captured, axis=0)
        assert float(np.abs(out).max()) > 1e-4, "it never played before capping"
        doc = json.loads((rec / "ending.json").read_text())
        assert doc["reason"] == realtime.END_MAX_MINUTES
        assert doc["capped"] is True
        assert doc["max_minutes"] == pytest.approx(3.0 / 60.0)
        assert doc["elapsed_s"] >= 3.0
        assert realtime.list_sessions() == []       # deregistered on the way out

    def test_the_cap_is_invariant_to_the_human_s_fader(self, material,
                                                       isolated_registry,
                                                       tmp_path):
        """A quiet player must not get a different session length.

        The same defect class as v1's: a rule that fires on a hot input and
        never on a quiet one.  The lifecycle must be deaf to level.
        """
        endings = []
        for tag in ("loud", "quiet"):
            rec = tmp_path / f"rec_{tag}"
            run_lead(material[tag], session=material["session"],
                     sock=short_sock(f"gain_{tag}"), rec=rec,
                     max_minutes=3.0 / 60.0)
            endings.append(json.loads((rec / "ending.json").read_text()))
        assert endings[0]["reason"] == endings[1]["reason"]
        assert endings[0]["capped"] is endings[1]["capped"] is True

    def test_an_uncapped_session_keeps_playing_until_it_is_stopped(
            self, material, isolated_registry, tmp_path):
        """--max-minutes 0 really does disable the cap."""
        sock = short_sock("nocap")
        rec = tmp_path / "rec"

        def stopper():
            for _ in range(3000):
                if Path(sock).exists():
                    time.sleep(1.5)      # far longer than any cap under test
                    try:
                        realtime.send_stop(socket_path=sock)
                        return
                    except realtime.RealtimeError:
                        pass
                time.sleep(0.01)

        t = threading.Thread(target=stopper, daemon=True)
        t.start()
        run_lead(material["loud"], session=material["session"], sock=sock,
                 rec=rec, max_minutes=0)
        t.join(timeout=30.0)
        doc = json.loads((rec / "ending.json").read_text())
        assert doc["reason"] == realtime.END_STOPPED
        assert doc["max_minutes"] == 0.0
        assert doc["elapsed_s"] > 1.0


# ================================ 6. the unattended-session regression, still green

def _run_cli_subprocess(args, env_extra=None):
    """Run the CLI in a genuinely non-interactive subprocess (pipes, no tty)."""
    env = dict(os.environ)
    env.pop(ENV, None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "plume", *args],
                          capture_output=True, text=True, timeout=180, env=env)


@pytest.mark.parametrize("verb,args", [
    ("jam", ["jam"]),
    ("lead", ["lead", "chill lofi at 96 bpm"]),
    ("jam-uncapped", ["jam", "--max-minutes", "0"]),
    ("lead-uncapped", ["lead", "chill lofi at 96 bpm", "--max-minutes", "0"]),
])
def test_a_live_verb_from_a_pipe_still_refuses(verb, args):
    """The unattended-session regression.  Nothing added here may open a door around it.

    In particular ``--max-minutes 0`` — the legitimate "I am here, do not
    time me out" flag — must not double as a way for a non-interactive agent
    to start an unbounded session.  The device check runs first (v2.1: a
    headless box must not be told to set a flag that cannot help it), so on a
    machine with no audio hardware the refusal is the device error; both are
    refusals and neither opens a stream.
    """
    proc = _run_cli_subprocess(args)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert ("refusing to play live audio" in out
            or "no audio" in out
            or "sounddevice" in out), out


def test_a_refused_session_leaves_no_registry_entry(tmp_path):
    """A session that never started must never be listed as running."""
    reg = tmp_path / "reg"
    proc = _run_cli_subprocess(
        ["jam"], {realtime.SESSION_DIR_ENV: str(reg)})
    assert proc.returncode != 0
    assert not reg.exists() or list(reg.glob("*.json")) == []


def test_stop_and_sessions_work_from_a_pipe(tmp_path):
    """Discoverability must not itself need a tty — an agent has to be able
    to check what it left running, and clean it up."""
    reg = tmp_path / "reg"
    reg.mkdir(exist_ok=True)
    for verb in ("sessions", "stop"):
        proc = _run_cli_subprocess(
            [verb, "--json"], {realtime.SESSION_DIR_ENV: str(reg)})
        assert proc.returncode == 0, proc.stdout + proc.stderr
        json.loads(proc.stdout)


# =====================================================================
# 7. v1.3 — the first session with a human and a guitar
#
# Six minutes of playing produced: an empty --record directory, a session
# that `plume stop` could not end (socket, SIGINT and SIGTERM all reported
# and all ineffective; `kill -9` lost the take), and a status line that was
# one unbounded unterminated line in the pipe it was being read through.
#
# Every test below runs the REAL transport — real threads, real socket, real
# planner, real DuetEngine — over the fake device in tests/conftest.py.  None
# of them opens an audio device.
# =====================================================================

def _ndjson(path):
    """Read a decisions log the way a live observer must: line at a time,
    stopping at a line the writer has not finished."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.endswith("\n"):
                break
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


@pytest.mark.usefixtures("fake_device_consent")
class TestRecordsStreamWhileThePlayingHappens:
    """P0-A.  ``--record takes/live1`` was EMPTY for a six-minute session and
    everything was written on a graceful exit — so there was nothing to watch
    while it ran, and nothing at all when it had to be killed."""

    def test_the_manifest_is_on_disk_before_the_first_bar_is_played(
            self, material, isolated_registry, tmp_path):
        """An observer must be able to tell what is running, immediately."""
        sock = short_sock("manifest")
        rec = tmp_path / "rec"
        seen = {}

        def watch():
            if wait_until(lambda: (rec / "session.json").exists(), timeout=20.0):
                seen["doc"] = json.loads((rec / "session.json").read_text())
                seen["bars_at_the_time"] = len(
                    _ndjson(rec / "decisions.ndjson")
                    if (rec / "decisions.ndjson").exists() else [])
            try:
                realtime.send_stop(socket_path=sock)
            except realtime.RealtimeError:
                pass

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        run_lead(material["loud"], session=material["session"], sock=sock,
                 rec=rec, max_minutes=0)
        t.join(timeout=10.0)

        assert "doc" in seen, "session.json never appeared while it was playing"
        doc = seen["doc"]
        assert doc["pid"] == os.getpid()
        assert doc["kind"] == "lead" and doc["style"] == "lofi"
        assert doc["socket"] == sock
        assert doc["record_dir"] == str(rec)
        assert doc["streaming"] is True
        assert doc["params"], "the manifest does not say what it is running with"

    def test_decisions_are_readable_while_the_session_is_still_playing(
            self, material, isolated_registry, tmp_path):
        """The whole point: `tail -f decisions.ndjson` shows a live session."""
        sock = short_sock("streamdec")
        rec = tmp_path / "rec"
        mid = {}

        def watch():
            log = rec / "decisions.ndjson"
            wait_until(lambda: log.exists() and len(_ndjson(log)) >= 2,
                       timeout=20.0)
            mid["rows"] = _ndjson(log) if log.exists() else []
            try:
                realtime.send_stop(socket_path=sock)
            except realtime.RealtimeError:
                pass

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        run_lead(material["loud"], session=material["session"], sock=sock,
                 rec=rec, max_minutes=0)
        t.join(timeout=10.0)

        assert mid.get("rows"), "decisions.ndjson was still empty mid-session"
        assert len(mid["rows"]) >= 2
        # every line an observer saw mid-flight is a complete, valid record
        for row in mid["rows"]:
            assert isinstance(row["bar"], int)
            assert row["form"]["action"]
        # ...and the finished log is a superset of what was already visible.
        final = _ndjson(rec / "decisions.ndjson")
        assert len(final) >= len(mid["rows"])
        assert [r["bar"] for r in final[:len(mid["rows"])]] == [
            r["bar"] for r in mid["rows"]]

    def test_the_wavs_grow_while_the_session_is_still_playing(
            self, material, isolated_registry, tmp_path):
        """40 minutes of float32 must not sit in RAM waiting for an ending."""
        sock = short_sock("streamwav")
        rec = tmp_path / "rec"
        frames = []

        def watch():
            wav = rec / "human.wav"
            wait_until(lambda: wav.exists() and sf.info(str(wav)).frames > 0,
                       timeout=20.0)
            for _ in range(3):
                try:
                    frames.append(sf.info(str(wav)).frames)
                except (RuntimeError, OSError):      # pragma: no cover
                    pass
                time.sleep(0.35)
            try:
                realtime.send_stop(socket_path=sock)
            except realtime.RealtimeError:
                pass

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        run_lead(material["loud"], session=material["session"], sock=sock,
                 rec=rec, max_minutes=0)
        t.join(timeout=10.0)

        assert len(frames) >= 2, "human.wav never became readable mid-session"
        assert frames[-1] > frames[0], f"the wav did not grow: {frames}"
        # and it was a VALID wav at every point, not a header waiting for a close
        y, sr = sf.read(str(rec / "human.wav"), dtype="float32")
        assert sr == SR and y.shape[0] >= frames[-1]

    def test_a_take_that_is_killed_keeps_everything_up_to_the_last_flush(
            self, material, isolated_registry, tmp_path):
        """The `kill -9` case, as far as a test can honestly go: read the take
        at an arbitrary mid-session instant and require it to be complete and
        parseable with nothing closed."""
        sock = short_sock("killmid")
        rec = tmp_path / "rec"
        snapshot = {}

        def watch():
            log = rec / "decisions.ndjson"
            wait_until(lambda: log.exists() and len(_ndjson(log)) >= 3,
                       timeout=20.0)
            snapshot["rows"] = _ndjson(log)
            try:
                snapshot["audio"] = sf.read(str(rec / "human.wav"),
                                            dtype="float32")[0].shape[0]
            except (RuntimeError, OSError):          # pragma: no cover
                snapshot["audio"] = 0
            try:
                realtime.send_stop(socket_path=sock)
            except realtime.RealtimeError:
                pass

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        run_lead(material["loud"], session=material["session"], sock=sock,
                 rec=rec, max_minutes=0)
        t.join(timeout=10.0)

        assert len(snapshot.get("rows", [])) >= 3
        assert snapshot["audio"] > 0, "a killed take would have had no audio"

    def test_the_sideman_streams_its_take_too(self, material,
                                              isolated_registry, tmp_path):
        """`jam` is the transport the human actually played into, and it had
        no end-to-end coverage at all — every live test ran `lead`."""
        sock = short_sock("jamstream")
        rec = tmp_path / "rec"
        seen = {}

        def watch():
            wait_until(lambda: (rec / "session.json").exists(), timeout=20.0)
            seen["manifest"] = (rec / "session.json").exists()
            time.sleep(0.5)
            try:
                realtime.send_stop(socket_path=sock)
            except realtime.RealtimeError:
                pass

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        fake = run_jam(material["loud"], sock=sock, rec=rec, max_minutes=0)
        t.join(timeout=10.0)

        assert seen.get("manifest"), "jam wrote nothing until it exited"
        doc = json.loads((rec / "session.json").read_text())
        assert doc["kind"] == "jam" and doc["role"] == "follow"
        assert (rec / "human.wav").exists() and (rec / "machine.wav").exists()
        assert sf.info(str(rec / "human.wav")).frames > 0
        end = json.loads((rec / "ending.json").read_text())
        assert end["reason"] == realtime.END_STOPPED
        assert fake.stream.blocks > 0


@pytest.mark.usefixtures("fake_device_consent")
class TestStopActuallyStops:
    """P0-B.  `plume stop` reported, honestly: *"pid 36655 is still running
    after socket, SIGINT, SIGTERM — it is wedged"*, and `kill -9` was the only
    thing left.  The honest report stays; the wedge does not."""

    def test_a_blocking_portaudio_stop_no_longer_wedges_the_ending(
            self, material, isolated_registry, tmp_path):
        """THE ROOT CAUSE, reproduced without an audio device.

        ``sounddevice``'s ``Stream.stop()`` is ``Pa_StopStream``, documented to
        *wait until all pending audio buffers have been played* — and the
        transports left the stream through ``with ... stream:``, whose
        ``__exit__`` calls exactly that.  On the built-in duplex pair feeding
        back into its own mic, that wait did not end.  The fix is to reach for
        ``Pa_AbortStream`` first, once the tail has already been drained.
        """
        sock = short_sock("blockstop")
        fake = fake_audio(material["loud"], rate_x=12.0, block_on_stop=True,
                          block_timeout=30.0)
        t, _ = stop_when_running(sock, after=0.4)
        t0 = time.monotonic()
        run_lead(material["loud"], session=material["session"], sock=sock,
                 max_minutes=0, fake=fake)
        elapsed = time.monotonic() - t0
        t.join(timeout=10.0)

        assert fake.stream.aborted is True, (
            "the ending walked into the blocking stop without aborting first")
        assert fake.stream.stop_blocked_s < 1.0, (
            f"stop blocked for {fake.stream.stop_blocked_s:.1f}s")
        assert elapsed < realtime.ENDING_HARD_DEADLINE_S, (
            f"the ending took {elapsed:.1f}s")

    def test_the_watchdog_ends_a_process_the_device_will_not_release(
            self, material, isolated_registry, tmp_path, monkeypatch):
        """And when even the abort does not come back, the process still goes.

        This is the case no amount of politeness inside the process can fix,
        and the reason the last resort is a thread calling ``os._exit``: the
        main thread is inside a C call, so no Python signal handler will ever
        run again in it.  ``os._exit`` is safe to reach for only because the
        take is already on disk.
        """
        monkeypatch.setattr(realtime, "ENDING_HARD_DEADLINE_S", 1.0)
        sock = short_sock("watchdog")
        rec = tmp_path / "rec"
        fake = fake_audio(material["loud"], rate_x=12.0, block_on_stop=True,
                          block_on_abort=True, block_timeout=30.0)
        exits = []
        real_hard_exit = realtime.HardExit
        made = []

        def _factory(**kw):
            def _fake_exit(code):
                exits.append(code)
                # os._exit would have ended the process here; let the blocked
                # stop() return so the test can finish and assert.
                fake.stream.release()

            kw["exit_fn"] = _fake_exit
            kw["stream"] = io.StringIO()
            h = real_hard_exit(**kw)
            made.append(h)
            return h

        monkeypatch.setattr(realtime, "HardExit", _factory)
        t, _ = stop_when_running(sock, after=0.4)
        t0 = time.monotonic()
        run_lead(material["loud"], session=material["session"], sock=sock,
                 rec=rec, max_minutes=0, fake=fake)
        elapsed = time.monotonic() - t0
        t.join(timeout=10.0)

        assert exits == [realtime.EXIT_WEDGED], (
            f"the watchdog never ended the wedged process (exits={exits})")
        assert made and made[0].fired is True
        assert elapsed < 20.0, f"the wedge held for {elapsed:.1f}s"
        # ...and it closed the take on its way out, from the watchdog thread.
        assert (rec / "ending.json").exists()
        assert json.loads((rec / "ending.json").read_text())["reason"] == (
            realtime.END_WEDGED)
        assert _ndjson(rec / "decisions.ndjson"), "the take was lost after all"

    def test_a_registry_entry_never_outlives_a_hard_exit(
            self, material, isolated_registry, tmp_path, monkeypatch):
        """A pid that no longer exists must not still be listed as playing."""
        monkeypatch.setattr(realtime, "ENDING_HARD_DEADLINE_S", 1.0)
        sock = short_sock("wdreg")
        fake = fake_audio(material["loud"], rate_x=12.0, block_on_stop=True,
                          block_on_abort=True, block_timeout=30.0)
        listed = {}
        real_hard_exit = realtime.HardExit

        def _factory(**kw):
            kw["exit_fn"] = lambda _c: (
                listed.setdefault("after", realtime.list_sessions()),
                fake.stream.release())
            kw["stream"] = io.StringIO()
            return real_hard_exit(**kw)

        monkeypatch.setattr(realtime, "HardExit", _factory)
        t, _ = stop_when_running(sock, after=0.4)
        run_lead(material["loud"], session=material["session"], sock=sock,
                 max_minutes=0, fake=fake)
        t.join(timeout=10.0)
        assert "after" in listed, "the watchdog never fired"
        assert listed["after"] == [], "a dead session was still registered"

    def test_every_worker_loop_leaves_within_the_bounded_ending(
            self, material, isolated_registry):
        """No thread may need an unbounded wait to notice a stop.

        The analysis loop used to be `while True`, checking its stop Event
        only when the ring buffers happened to run dry; the planner slept on
        `time.sleep` rather than the Event.
        """
        sock = short_sock("threads")
        t, done = stop_when_running(sock, after=0.4)
        t0 = time.monotonic()
        run_lead(material["loud"], session=material["session"], sock=sock,
                 max_minutes=0)
        elapsed = time.monotonic() - t0
        t.join(timeout=10.0)

        assert done.get("ok") is True, done
        assert elapsed < realtime.ENDING_HARD_DEADLINE_S, (
            f"the ending needed {elapsed:.1f}s")
        alive = [th.name for th in threading.enumerate()
                 if th.name in ("plume-analysis", "plume-planner", "plume-cues")]
        assert alive == [], f"worker threads outlived the session: {alive}"

    def test_the_sideman_stops_the_same_way(self, material, isolated_registry,
                                            tmp_path):
        """`jam` had the same ending and the same bug; it gets the same test."""
        sock = short_sock("jamstop")
        rec = tmp_path / "rec"
        t, done = stop_when_running(sock, after=0.4)
        t0 = time.monotonic()
        run_jam(material["loud"], sock=sock, rec=rec, max_minutes=0)
        elapsed = time.monotonic() - t0
        t.join(timeout=10.0)
        assert done.get("ok") is True, done
        assert elapsed < realtime.ENDING_HARD_DEADLINE_S
        assert json.loads((rec / "ending.json").read_text())["reason"] == (
            realtime.END_STOPPED)

    def test_a_second_signal_shortens_the_deadline_instead_of_being_eaten(self):
        """The rung that was silently doing nothing.

        ``GracefulSignals`` restored SIG_DFL on a second signal, so the second
        signal itself was SWALLOWED and only a third would have been fatal —
        but `plume stop`'s ladder is two rungs long, and SIGTERM was always the
        second one.  A user hammering Ctrl-C hit the same wall.  Now the second
        signal also pulls the hard deadline in, which needs nothing from the
        main thread.
        """
        pulled = []
        g = realtime.GracefulSignals(on_repeat=pulled.append)
        with g:
            os.kill(os.getpid(), signal.SIGTERM)
            assert _wait_for(g.event)
            assert pulled == [], "the FIRST signal must be graceful"
            g._handle(signal.SIGTERM, None)      # the impatient second press
            assert pulled == [signal.SIGTERM]
            signal.signal(signal.SIGTERM, g._handle)   # restore for __exit__

    def test_the_transports_arm_the_watchdog_only_for_the_ending(
            self, material, isolated_registry, monkeypatch):
        """A fuse burning during normal play would end healthy sessions."""
        sock = short_sock("armwhen")
        armed_at = []
        real_hard_exit = realtime.HardExit

        def _factory(**kw):
            h = real_hard_exit(**kw)
            real_arm = h.arm

            def arm(reason, **akw):
                armed_at.append(reason)
                return real_arm(reason, **akw)

            h.arm = arm
            return h

        monkeypatch.setattr(realtime, "HardExit", _factory)
        t, _ = stop_when_running(sock, after=0.5)
        run_lead(material["loud"], session=material["session"], sock=sock,
                 max_minutes=0)
        t.join(timeout=10.0)
        assert armed_at == [realtime.END_STOPPED], armed_at


@pytest.mark.usefixtures("fake_device_consent")
class TestPipedStatusIsReadable:
    """P2-G, on the live transport rather than the class."""

    def test_a_piped_session_emits_newline_terminated_status_lines(
            self, material, isolated_registry, capsys):
        """pytest's captured stdout is not a tty — which is exactly the case
        that produced one unbounded line with no newline in it."""
        sock = short_sock("pipestatus")
        t, _ = stop_when_running(sock, after=1.2)
        fake = fake_audio(material["loud"], rate_x=12.0)
        with patched_module("sounddevice", fake):
            realtime.lead(session=material["session"], seed=0,
                          role=roles_mod.RoleParams(role="lead"),
                          socket_path=sock, status=True, max_minutes=0)
        t.join(timeout=10.0)
        out = capsys.readouterr().out
        assert "\r" not in out, "carriage returns went into a pipe"
        status = [ln for ln in out.splitlines() if ln.startswith("[bar ")]
        assert len(status) >= 2, f"only {len(status)} status lines: {out!r}"
        assert out.endswith("\n")
        assert any(ln.startswith("ending — ") for ln in out.splitlines())


# ------------------------------------- 7b. the wedge, across a real process

_WEDGED_SCRIPT = """
import os, signal, sys, time
sys.path.insert(0, {src!r})
from plume import realtime

# Exactly the state the wedged session was in: BOTH signals have Python-level
# dispositions that do nothing (GracefulSignals had already consumed its
# first signal and swallowed the second), and the main thread is not going to
# reach any of its own cleanup.
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)

guard = realtime.HardExit()
print("ready", flush=True)
guard.arm("stopped by 'plume stop'", seconds={fuse})
while True:
    time.sleep(0.05)
"""


def test_a_fake_session_that_ignores_sigint_still_dies_within_the_deadline(
        tmp_path):
    """The cross-process proof, because the in-process one cannot be honest.

    A test cannot let ``os._exit`` run inside pytest, so every in-process
    check above stubs it.  This one does not stub anything: a real child
    process ignores SIGINT and SIGTERM the way the wedged session effectively
    did, the parent sends both and watches them do nothing, and the child ends
    itself anyway — with the exit status that says a watchdog had to do it.
    """
    script = tmp_path / "wedged_session.py"
    script.write_text(_WEDGED_SCRIPT.format(src=str(SRC), fuse=2.0))
    proc = subprocess.Popen([sys.executable, str(script)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    try:
        assert proc.stdout.readline().strip() == "ready"
        proc.send_signal(signal.SIGINT)
        time.sleep(0.3)
        assert proc.poll() is None, "SIGINT was supposed to be ignored here"
        proc.send_signal(signal.SIGTERM)
        time.sleep(0.3)
        assert proc.poll() is None, "SIGTERM was supposed to be ignored here"
        code = proc.wait(timeout=20.0)
    finally:
        if proc.poll() is None:                      # pragma: no cover
            proc.kill()
            proc.wait(timeout=5.0)
    assert code == realtime.EXIT_WEDGED, (
        f"the process did not end itself (exit {code}); "
        f"stderr: {proc.stderr.read()}")


_KILL9_SCRIPT = """
import sys, time
sys.path.insert(0, {src!r})
sys.path.insert(0, {tests!r})
import numpy as np
from conftest import fake_audio, patched_module
from plume import realtime
from plume import mind as mind_mod

rng = np.random.default_rng(4)
sr = realtime.SR
t = np.arange(int(40.0 * sr)) / sr
# something with transients, so bars actually get planned and logged
env = (np.sin(2 * np.pi * 2.0 * t) > 0.7).astype(np.float32)
human = (env * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32) * 0.3

fake = fake_audio(human, rate_x=12.0)
with patched_module("sounddevice", fake):
    realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(),
                 record_dir={rec!r}, socket_path={sock!r}, max_minutes=0)
"""


@pytest.mark.usefixtures("fake_device_consent")
def test_a_take_survives_a_real_kill_9(tmp_path, isolated_registry):
    """The postmortem's actual loss, reproduced with a real SIGKILL.

    "``plume stop`` could not stop it ... so it took ``kill -9``, and the take
    was lost."  Every other streaming test above stops the session politely
    and then reads the files; none of them proves the thing that was lost.
    This one starts a REAL child process running the REAL sideman transport
    over the fake device, waits until it is playing, sends SIGKILL — which no
    handler, watchdog or ``finally`` can intercept — and then requires the
    take on disk to be complete and readable with nothing having closed it.

    SIGKILL is safe to use here precisely because the device is fake: no
    stream is ever opened on real hardware.
    """
    rec = tmp_path / "take"
    sock = short_sock("kill9")
    script = tmp_path / "kill9_session.py"
    script.write_text(_KILL9_SCRIPT.format(
        src=str(SRC), tests=str(Path(__file__).parent), rec=str(rec),
        sock=sock))
    # The child registers itself, so it must be pointed at the SAME isolated
    # registry as the parent — a SIGKILLed session cannot unregister itself,
    # and a stale record in the developer's real registry is exactly the kind
    # of leak `plume stop` would later act on.
    env = dict(os.environ)
    env[realtime.SESSION_DIR_ENV] = str(isolated_registry)
    proc = subprocess.Popen([sys.executable, str(script)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
    try:
        log = rec / realtime.REC_DECISIONS
        playing = wait_until(
            lambda: log.exists() and len(_ndjson(log)) >= 3, timeout=60.0)
        assert playing, (
            "the child never wrote three decisions; stderr so far is not "
            "readable without ending it, so this is a setup failure")
        # the manifest must already be there — an observer has to be able to
        # tell what is running BEFORE it is over
        manifest = json.loads((rec / realtime.REC_MANIFEST).read_text())
        assert manifest["kind"] == "jam"
        assert manifest["pid"] == proc.pid

        rows_before = _ndjson(log)
        frames_before = sf.info(str(rec / realtime.REC_HUMAN)).frames
        os.kill(proc.pid, signal.SIGKILL)
        assert proc.wait(timeout=20.0) == -signal.SIGKILL
    finally:
        if proc.poll() is None:                       # pragma: no cover
            proc.kill()
            proc.wait(timeout=5.0)

    # Nothing ran on the way out. What is on disk is what streaming bought.
    rows_after = _ndjson(log)
    assert len(rows_after) >= len(rows_before) >= 3, (
        "decisions written before the kill did not survive it")
    assert [r["bar"] for r in rows_after[:len(rows_before)]] == \
           [r["bar"] for r in rows_before]
    for r in rows_after:
        assert isinstance(r["bar"], int) and r["state"]

    y, sr = sf.read(str(rec / realtime.REC_HUMAN), dtype="float32")
    assert sr == SR
    assert y.shape[0] >= frames_before > 0, (
        "human.wav was not a valid, correct-length file at the instant of "
        "the kill — the RIFF header never got its flush")
    m, _sr = sf.read(str(rec / realtime.REC_MACHINE), dtype="float32")
    assert m.shape[0] > 0

    # ending.json is the one thing a kill legitimately costs, and its absence
    # is how a reader tells a killed take from a finished one.
    assert not (rec / realtime.REC_ENDING).exists()


def test_the_hard_exit_says_on_stderr_that_it_had_to_do_it(tmp_path):
    """Silence here would look exactly like a crash.  It must say what it did
    and that the take survived — and say it on stderr, flushed, because
    ``os._exit`` flushes nothing."""
    script = tmp_path / "wedged_session2.py"
    script.write_text(_WEDGED_SCRIPT.format(src=str(SRC), fuse=0.5))
    proc = subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True, timeout=30.0)
    assert proc.returncode == realtime.EXIT_WEDGED
    assert "did not finish" in proc.stderr
    assert "on disk" in proc.stderr


# =================== 8. the room: plume hearing itself, live ================
#
# ``tests/test_realtime.py`` proves the masking BEHAVIOUR in a closed loop.
# This section proves the WIRING, in the real transport, on the fake device —
# which is where the defect actually was.  For four versions the masker was
# constructed, called, and had everything it produced discarded, and no test
# anywhere could see it because no test had a mic that could hear the
# speakers.  The fake device has one now (``bleed_db``).

class _SpyMasker(realtime.SelfMasker):
    """A SelfMasker that remembers every instance the transport made."""

    made: list = []

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        _SpyMasker.made.append(self)


@pytest.fixture
def spy_masker(monkeypatch):
    _SpyMasker.made = []
    monkeypatch.setattr(realtime, "SelfMasker", _SpyMasker)
    return _SpyMasker


@pytest.mark.usefixtures("fake_device_consent")
class TestTheTransportActuallyMasksItself:
    """The mic hears the speakers; the transport must act on that."""

    def test_the_sideman_masks_its_own_output_out_of_the_tracker(
            self, material, isolated_registry, spy_masker):
        sock = short_sock("mask-jam")
        fake = fake_audio(material["loud"], rate_x=14.0, bleed_db=-6.0,
                          bleed_delay_s=0.020)
        stop_when_running(sock, after=5.0)
        run_jam(material["loud"], sock=sock, max_minutes=0, fake=fake)

        assert spy_masker.made, "jam did not build a SelfMasker at all"
        m = spy_masker.made[0]
        assert m.frames_seen > 100, "the analysis loop never ran"
        assert m.masked_frames > 0, (
            "the machine was in the mic for the whole session and not one "
            "frame was masked — the mask is not in the signal path")
        assert m.masked_fraction < 0.9, (
            f"masked {m.masked_fraction:.0%} of the stream: that is a gate, "
            f"not a mask")

    def test_the_lead_transport_masks_too(self, material, isolated_registry,
                                          spy_masker):
        sock = short_sock("mask-lead")
        fake = fake_audio(material["loud"], rate_x=10.0, bleed_db=-6.0)
        stop_when_running(sock, after=1.6)
        run_lead(material["loud"], session=material["session"], sock=sock,
                 max_minutes=0, fake=fake)
        assert spy_masker.made
        m = spy_masker.made[0]
        assert m.frames_seen > 100
        assert m.masked_frames > 0

    def test_the_masker_takes_its_latency_from_the_open_stream(
            self, material, isolated_registry, spy_masker):
        # Defect 3: a mask aligned to when the block was HANDED to PortAudio
        # is aligned to the wrong moment.  Only the open stream knows the
        # round trip, so the transport has to ask it — after opening it.
        sock = short_sock("mask-lat")
        fake = fake_audio(material["loud"], rate_x=14.0, bleed_db=-6.0,
                          latency=(0.025, 0.030))
        stop_when_running(sock, after=1.5)
        run_jam(material["loud"], sock=sock, max_minutes=0, fake=fake)
        m = spy_masker.made[0]
        assert m.latency_hops == realtime.stream_latency_hops(fake.stream)
        assert m.latency_hops == 5          # 55 ms at 48k/512

    def test_a_session_with_the_room_in_the_mic_still_records_a_take(
            self, material, isolated_registry, tmp_path):
        """The end-to-end sanity check: bleed must not break the session."""
        sock = short_sock("mask-rec")
        rec = tmp_path / "take"
        fake = fake_audio(material["loud"], rate_x=10.0, bleed_db=-6.0)
        stop_when_running(sock, after=1.6)
        run_jam(material["loud"], sock=sock, rec=rec, max_minutes=0, fake=fake)
        lines = [json.loads(x) for x in
                 (rec / realtime.REC_DECISIONS).read_text().splitlines() if x]
        assert lines, "no decisions were logged at all"
        assert (rec / realtime.REC_HUMAN).exists()
        assert (rec / realtime.REC_MACHINE).exists()

    def test_headphones_are_still_the_no_op_they_should_be(
            self, material, isolated_registry, spy_masker):
        """No bleed, nothing to mask.  The mask must cost nothing here — it is
        the recommended configuration (RESEARCH.md) and the common one."""
        sock = short_sock("mask-hp")
        fake = fake_audio(material["loud"], rate_x=14.0)     # bleed_db=None
        stop_when_running(sock, after=5.0)
        run_jam(material["loud"], sock=sock, max_minutes=0, fake=fake)
        m = spy_masker.made[0]
        assert m.frames_seen > 100
        # the machine bus is still loud, so the mask CAN fire; what must not
        # happen is it swallowing the stream
        assert m.masked_fraction < 0.5, m.masked_fraction


def _planner_source() -> str:
    """``jam``'s ``planner_loop`` as text, read from disk by name.

    Not ``inspect.getsource``: that resolves a live code object's line number
    against whatever the file says now, so editing the module while a process
    has it imported makes it hand back a different function entirely.
    """
    import ast
    import textwrap

    tree = ast.parse(Path(realtime.__file__).read_text(encoding="utf-8"))
    jam = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "jam")
    planner = next(n for n in ast.walk(jam)
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "planner_loop")
    lines = Path(realtime.__file__).read_text(encoding="utf-8").splitlines()
    return textwrap.dedent(
        "\n".join(lines[planner.lineno - 1: planner.end_lineno]))


@pytest.mark.usefixtures("fake_device_consent")
class TestABarIsRenderedOnce:
    """P2: a revised bar must not be a second bar.

    ``DuetEngine.render_bar`` is a stateful stream — it advances ``_frac``
    (the fractional-sample carry), consumes ``_tail`` (the previous bar's
    ring-out) and rolls ``_prev_voicing``.  The planner's revision window used
    to call ``_plan_and_place`` a second time for the same bar whenever the
    predicted downbeat moved more than 2 ms, which advanced all three twice
    for one bar of music.  A revision now moves the samples that already
    exist.
    """

    def test_the_planner_never_renders_one_bar_index_twice(
            self, material, isolated_registry, monkeypatch):
        from plume import duet as duet_mod

        seen: list[int] = []
        held: list[bool] = []
        real = duet_mod.DuetEngine.render_bar

        def spy(self, plan, beat, harm, *, continuation=False):
            # A bar can lose the lock after it was placed.  The bars the mind then
            # holds carry its index on purpose and sound after it, so only a render
            # that is not a continuation can be the same bar a second time.  The
            # boundary path has to be locked and the held path must not be, or a
            # render could dodge this count by calling itself a continuation.
            if continuation:
                held.append(bool(beat.locked))
            else:
                assert beat.locked, "a boundary bar was rendered without a lock"
                seen.append(int(plan.bar))
            return real(self, plan, beat, harm, continuation=continuation)

        monkeypatch.setattr(duet_mod.DuetEngine, "render_bar", spy)
        sock = short_sock("once")
        fake = fake_audio(material["loud"], rate_x=14.0, bleed_db=-6.0)
        stop_when_running(sock, after=5.0)
        run_jam(material["loud"], sock=sock, max_minutes=0, fake=fake)
        assert len(seen) >= 2, (
            f"only {len(seen)} bar(s) were rendered — this test would pass "
            f"vacuously; the session never got going")
        assert len(seen) == len(set(seen)), (
            f"bars rendered more than once: "
            f"{sorted(b for b in set(seen) if seen.count(b) > 1)}")
        assert not any(held), "a locked bar was rendered as a held continuation"

    def test_a_bar_is_not_re_rendered_after_the_tracker_re_locks(self):
        """The second door into the same defect, stated deterministically.

        The threaded test above only catches this when the timing happens to
        produce a lock loss — it did, once, in a full-suite run, and passed
        five times in a row in isolation, which is the worst possible way to
        learn about a defect.  So the rule is stated directly here instead.

        ``BeatTracker``'s bar counter never goes backwards: it FREEZES while
        unlocked and resumes where it stopped.  ``planned_bar`` and ``placed``
        are both cleared on unlock, so on re-lock the planner is looking at a
        bar index it has already rendered with no memory that it did — and
        rendering it again double-advances ``DuetEngine._frac``, spends the
        previous bar's ``_tail`` twice and voice-leads from a chord that never
        sounded.  ``last_rendered_bar`` is what remembers across the gap.
        """
        planner = _planner_source()
        assert "last_rendered_bar" in planner, (
            "the planner has no memory of what it rendered across a lock "
            "loss — a re-lock will render an already-played bar again")
        # the guard must be a >= comparison against the upcoming bar, and it
        # must come BEFORE the plan/render branch
        guard = planner.index("upcoming <= last_rendered_bar")
        render = planner.index("_plan_and_place(")
        assert guard < render, (
            "the already-rendered check runs after the render it is supposed "
            "to prevent")

    def test_the_re_lock_guard_cannot_deadlock(self):
        """It skips only while the counter is behind, and the counter rises.

        Stated because "just don't render it" is the kind of guard that can
        silently mean "never play again": if ``beat.bar`` could reset to 0 on
        re-lock, this would mute the session permanently.  It cannot — the
        only assignments in ``beat.py`` are ``+= 1``, a first-lock ``0`` and
        ``tap()``'s count-in — and that is what makes the guard safe.
        """
        src = (Path(realtime.__file__).parent / "listen" / "beat.py").read_text()
        assigns = [ln.strip() for ln in src.splitlines()
                   if ln.strip().startswith("self._bar ")
                   and "=" in ln and "==" not in ln]
        assert assigns, "beat.py no longer has a bar counter to reason about"
        for ln in assigns:
            assert ("+= 1" in ln or "= 0" in ln or "= -1" in ln), (
                f"beat.py's bar counter now moves in a way the planner's "
                f"re-lock guard was not designed for: {ln!r}")

    def test_a_revision_moves_the_bar_it_does_not_remake_it(self):
        """The unit-level statement, without threads: the timeline ends up
        holding exactly one copy of the bar, at the corrected sample."""
        tl = realtime.TimelineBuffer(realtime.SR * 30, channels=2)
        audio = np.ones((realtime.SR // 4, 2), dtype=np.float32)
        placed = (1000, audio.shape[0], audio)
        tl.write_at(placed[0], audio)

        # what _move_bar does, with the same clear-then-write contract
        tl.clear_region(placed[0], placed[1])
        tl.write_at(1200, audio)

        out = np.zeros((realtime.SR // 2, 2), dtype=np.float32)
        tl.read_into(out)
        loud = np.flatnonzero(np.abs(out[:, 0]) > 0.5)
        assert loud.size == audio.shape[0], (
            "the old placement was not cleared, or the new one overlapped it")
        assert int(loud[0]) == 1200
