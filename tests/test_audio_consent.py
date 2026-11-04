"""No process may play out of a human's speakers without consent.

Twice during development an automated agent started a live session and left it
running, still audible, stoppable only by finding the
PID.  Every path that opens the audio device is guarded here: an interactive
terminal, or an explicit ``PLUME_ALLOW_AUDIO=1``.  Nothing else.

These tests must never be relaxed to "make CI pass" — CI is exactly the
non-interactive case they exist to block.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import types

import numpy as np
import pytest
import soundfile as sf

from plume import realtime


ENV = realtime.AUDIO_CONSENT_ENV


def test_non_interactive_is_refused_by_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    allowed, why = realtime.audio_output_allowed()
    assert allowed is False
    assert why


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_explicit_opt_in_is_honoured(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    allowed, _ = realtime.audio_output_allowed()
    assert allowed is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_explicit_opt_out_beats_a_tty(monkeypatch, value):
    """An explicit no wins even from an interactive terminal."""
    monkeypatch.setenv(ENV, value)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    allowed, why = realtime.audio_output_allowed()
    assert allowed is False
    assert value in why


def test_check_audio_consent_names_the_offline_alternative(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(realtime.RealtimeError) as exc:
        realtime.check_audio_consent()
    msg = str(exc.value)
    assert ENV in msg
    assert "plume sim" in msg          # the user is told what to do instead


def test_engine_play_is_guarded(monkeypatch):
    """`plume play` opens the device too — it must go through the same gate."""
    from plume import engine

    monkeypatch.setenv(ENV, "0")
    with pytest.raises(realtime.RealtimeError):
        engine.play(object())          # guard must fire before anything else


def _run(args, env_extra=None):
    """Run the CLI in a genuinely non-interactive subprocess (pipes, no tty)."""
    import os

    env = dict(os.environ)
    env.pop(ENV, None)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "plume", *args],
        capture_output=True, text=True, timeout=120, env=env,
    )


@pytest.mark.parametrize("verb", ["jam", "lead"])
def test_live_verbs_refuse_from_a_pipe(verb):
    """The unattended-session regression, end to end: no tty, so no sound, and a clean exit.

    The invariant is **no audio comes out** and the exit is clean.  Which of
    the two refusals the user sees depends on the box: since v2.1 the device
    check runs first (a machine with no audio hardware must not be told to set
    a consent flag that cannot help it), so on a headless box the refusal is
    the device error instead.  Both are refusals; neither opens a stream.  The
    consent gate itself is asserted directly above and in
    tests/test_realtime.py::TestDeviceErrors — this test must not be the one
    that gets relaxed, and it is not being relaxed: it still requires a
    non-zero exit and a recognised refusal, never a session.
    """
    args = [verb] if verb == "jam" else [verb, "chill lofi at 96 bpm"]
    proc = _run(args)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert ("refusing to play live audio" in out
            or "no audio" in out
            or "sounddevice" in out), out


@pytest.mark.parametrize("transport", ["jam", "lead"])
@pytest.mark.parametrize("flag", [None, "0"])
def test_the_transports_themselves_refuse_without_consent(transport, flag, tmp_path, monkeypatch,
                                                          plume_registry):
    """Every caller gets the gate, not only the ones that remembered the CLI's."""
    from conftest import fake_audio, patched_module
    from plume import mind, roles
    if flag is None:
        monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
    else:
        monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, flag)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    sd = fake_audio(None)
    with patched_module("sounddevice", sd), pytest.raises(realtime.RealtimeError, match="refusing"):
        if transport == "jam":
            realtime.jam(style="lofi", seed=0, params=mind.MindParams(), max_minutes=0.01,
                         record_dir=str(tmp_path / "rec"))
        else:
            realtime.lead(prompt="lofi", seed=0, role=roles.RoleParams(role="lead"),
                          max_minutes=0.01, record_dir=str(tmp_path / "rec"), status=False)
    assert sd.streams == []


def test_a_play_can_be_found_while_it_sounds(tmp_path, monkeypatch, plume_registry):
    """`plume stop` must not say nothing is running while plume is making sound."""
    from conftest import patched_module
    from plume import engine
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only
    seen = []

    class FakeSd:
        def play(self, data, sr):
            pass

        def wait(self):
            seen.append([s.kind for s in realtime.list_sessions()])

    monkeypatch.setattr(engine, "render",
                        lambda session, out_path, **kw: sf.write(out_path, np.zeros(10), 48000))
    with patched_module("sounddevice", FakeSd()):
        engine.play(object())
    assert seen == [["play"]]
    assert realtime.list_sessions() == []


def test_a_calibration_can_be_found_while_it_sounds(monkeypatch, plume_registry):
    from plume import calibrate, cli
    seen = []
    monkeypatch.setattr(realtime, "check_device", lambda device=None: None)

    def run(device=None):
        seen.append([s.kind for s in realtime.list_sessions()])
        return {"ok": False, "error": "no"}
    monkeypatch.setattr(calibrate, "run", run)
    cli.cmd_calibrate(types.SimpleNamespace(device=None))
    assert seen == [["calibrate"]]
    assert realtime.list_sessions() == []


# ---- plume stop silences a sound started by a script, fast ----

_IGNORING_SIGINT = """
import signal, sys, time, types
signal.signal(signal.SIGINT, signal.SIG_IGN)     # what a script's background job inherits
fake = types.ModuleType("sounddevice")
fake.play = lambda data, sr: None
def wait():
    print("sounding", flush=True)
    time.sleep(120)
fake.wait = wait
sys.modules["sounddevice"] = fake                # never a real device
import numpy as np, soundfile as sf
from plume import calibrate, cli, engine, realtime
realtime.check_audio_consent = lambda: None
realtime.check_device = lambda device=None: None
engine.render = lambda session, out_path, **kw: sf.write(out_path, np.zeros(10), 48000)
calibrate.run = lambda device=None: wait()
{call}
"""


@pytest.mark.parametrize("call,kind", [
    ("engine.play(object())", "play"),
    ("cli.cmd_calibrate(types.SimpleNamespace(device=None))", "calibrate"),
])
def test_stop_silences_a_session_that_inherited_sigint_ignored(call, kind, plume_registry,
                                                               monkeypatch):
    monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
    child = subprocess.Popen([sys.executable, "-c", _IGNORING_SIGINT.format(call=call)],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                             preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
    try:
        assert child.stdout.readline().strip() == "sounding"
        rec = realtime.list_sessions()[0]
        assert rec.kind == kind
        threading.Thread(target=child.wait, daemon=True).start()   # no zombie to wait on
        began = time.monotonic()
        out = realtime.stop_session(rec, poll=0.02)
        took = time.monotonic() - began
        assert out["stopped"] and out["how"] == "signal:SIGINT", out
        assert took < 3.0, f"sound went on for {took:.1f} s after plume stop"
        assert "records" not in out["detail"] and "lost" not in out["detail"]
        assert child.wait(timeout=5.0) != -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5.0)


_SLOW_BEFORE_SOUND = """
import signal, sys, time, types
signal.signal(signal.SIGINT, signal.SIG_IGN)     # what a script's background job inherits
fake = types.ModuleType("sounddevice")
def sound(*a, **kw):
    print("sound started", flush=True)
fake.play = sound
fake.wait = lambda: None
sys.modules["sounddevice"] = fake                # never a real device
import numpy as np, soundfile as sf
from plume import calibrate, cli, engine, realtime
realtime.check_audio_consent = lambda: None
def slow(*a, **kw):
    print("preparing", flush=True)
    time.sleep(2.0)
def render(session, out_path, **kw):
    slow()
    sf.write(out_path, np.zeros(10), 48000)
engine.render = render
realtime.check_device = {check}
calibrate.run = lambda device=None: sound() or {{"ok": False}}
{call}
"""


@pytest.mark.parametrize("call,check,kind", [
    ("engine.play(object())", "lambda device=None: None", "play"),
    ("cli.cmd_calibrate(types.SimpleNamespace(device=None))", "slow", "calibrate"),
])
def test_a_stop_while_getting_ready_means_no_sound_ever_starts(call, check, kind, plume_registry,
                                                               monkeypatch):
    monkeypatch.delenv(realtime.AUDIO_CONSENT_ENV, raising=False)
    child = subprocess.Popen(
        [sys.executable, "-c", _SLOW_BEFORE_SOUND.format(call=call, check=check)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
    try:
        assert child.stdout.readline().strip() == "preparing"
        found = realtime.list_sessions()
        assert [r.kind for r in found] == [kind], "plume stop cannot see it yet"
        threading.Thread(target=child.wait, daemon=True).start()   # no zombie to wait on
        out = realtime.stop_session(found[0], poll=0.02)
        assert out["stopped"] and out["how"] == "signal:SIGINT", out
        assert child.stdout.read() == "", "sound started after plume stop"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5.0)


def test_a_stuck_play_gets_a_short_fuse_and_loses_no_records(monkeypatch, plume_registry):
    """Even if SIGINT goes unheard, sound must not run on for the graceful budget."""
    sent = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sig and sent.append(sig))
    monkeypatch.setattr(realtime, "_pid_alive",
                        lambda pid: signal.SIGTERM not in sent)
    rec = realtime.LiveSession(pid=4242, kind="play", style="", role="", session_name=None,
                               socket=None, started_at=time.time(), max_minutes=None,
                               record_dir=None)
    began = time.monotonic()
    out = realtime.stop_session(rec, poll=0.02)
    assert time.monotonic() - began < 6.0
    assert sent == [signal.SIGINT, signal.SIGTERM]
    assert "no records" in out["detail"] and "probably lost" not in out["detail"]


@pytest.mark.parametrize("record_dir,says", [(None, "no records"), ("takes/x", "probably lost")])
def test_a_take_with_no_records_does_not_report_losing_them(record_dir, says, monkeypatch,
                                                            plume_registry):
    sent = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sig and sent.append(sig))
    monkeypatch.setattr(realtime, "_pid_alive",
                        lambda pid: signal.SIGTERM not in sent)
    rec = realtime.LiveSession(pid=4242, kind="jam", style="", role="", session_name=None,
                               socket=None, started_at=time.time(), max_minutes=None,
                               record_dir=record_dir)
    assert says in realtime.stop_session(rec, timeout=0.2, poll=0.02)["detail"]


def test_a_stopped_play_exits_without_a_traceback(tmp_path, monkeypatch, capsys):
    from plume import cli, engine

    def interrupted(*a, **kw):
        raise KeyboardInterrupt
    monkeypatch.setattr(engine, "play", interrupted)
    monkeypatch.setattr(cli, "resolve_session", lambda s: tmp_path / "s.json")
    monkeypatch.setattr(cli, "_load_session", lambda p: object())
    assert cli.main(["play", "--session", "s"]) == 130
    assert "stopped" in capsys.readouterr().out
