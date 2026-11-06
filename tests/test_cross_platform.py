"""macOS, Linux and Windows: the places they disagree, tested on whichever one runs this.

Most of plume is numpy and files, and runs anywhere.  What does not is the hardening:
the cue socket (AF_UNIX), private files (``O_NOFOLLOW``, ``fchmod``), child processes
(sessions, SIGINT, SIGHUP) and asking whether a pid is alive.  Each of those has a
Windows path in :mod:`plume.oscompat` or :mod:`plume.realtime`, and each Windows path
is driven here on every OS, by forcing it, so a Mac or Linux run still proves the
logic Windows will take.  CI runs it on Windows for real.
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import socket
import stat
import sys
import tempfile
import time
from pathlib import Path

import pytest

from plume import gui, oscompat, realtime, voice

WINDOWS = sys.platform == "win32"
POSIX = not WINDOWS


def _short(name: str) -> str:
    p = Path(tempfile.gettempdir()) / f"plume-X{os.getpid()}-{name}.sock"
    try:
        p.unlink()
    except OSError:
        pass
    return str(p)


def _can_symlink(tmp_path) -> bool:
    try:
        (tmp_path / ".probe-link").symlink_to(tmp_path)
    except (OSError, NotImplementedError):
        return False
    (tmp_path / ".probe-link").unlink()
    return True


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    d = tmp_path / "reg"
    d.mkdir()
    monkeypatch.setenv(realtime.SESSION_DIR_ENV, str(d))
    monkeypatch.delenv(realtime.TRANSPORT_ENV, raising=False)
    return d


# ================================================================ private files

def test_open_private_refuses_a_symlink_even_without_o_nofollow(tmp_path, monkeypatch):
    """Windows has no O_NOFOLLOW; the lstat check stands in for it."""
    if not _can_symlink(tmp_path):
        pytest.skip("this account cannot make symlinks")
    target = tmp_path / "precious"
    target.write_text("hers")
    (tmp_path / "link").symlink_to(target)
    monkeypatch.setattr(oscompat, "O_NOFOLLOW", 0)
    with pytest.raises(OSError):
        oscompat.open_private(tmp_path / "link", os.O_WRONLY | os.O_TRUNC)
    assert target.read_text() == "hers"


def test_open_private_still_opens_a_plain_file_without_o_nofollow(tmp_path, monkeypatch):
    monkeypatch.setattr(oscompat, "O_NOFOLLOW", 0)
    fd = oscompat.open_private(tmp_path / "new", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(fd, b"ok")
    os.close(fd)
    assert (tmp_path / "new").read_bytes() == b"ok"


def test_open_private_always_asks_for_binary(monkeypatch, tmp_path):
    """On Windows a WAV opened in text mode loses bytes; O_BINARY is never left out."""
    seen = []
    real = os.open

    def spy(path, flags, *args, **kw):
        seen.append(flags)
        return real(path, flags & ~0x8000, *args, **kw)

    monkeypatch.setattr(oscompat, "O_BINARY", 0x8000)
    monkeypatch.setattr(oscompat.os, "open", spy)
    os.close(oscompat.open_private(tmp_path / "f", os.O_WRONLY | os.O_CREAT))
    assert seen and seen[0] & 0x8000


def test_a_wav_read_through_open_private_comes_back_byte_for_byte(tmp_path):
    raw = bytes(range(256)) * 64          # every byte, CR LF and 0x1A included
    (tmp_path / "x.wav").write_bytes(raw)
    fd = oscompat.open_private(tmp_path / "x.wav", oscompat.read_flags())
    with os.fdopen(fd, "rb") as f:
        assert f.read() == raw


def test_fchmod_private_is_skipped_on_windows(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(oscompat, "IS_WINDOWS", True)
    monkeypatch.setattr(oscompat.os, "fchmod", lambda *a: called.append(a), raising=False)
    monkeypatch.setattr(oscompat.os, "chmod", lambda *a: called.append(a))
    oscompat.fchmod_private(3)
    oscompat.chmod_private(tmp_path, 0o700)
    assert called == []


@pytest.mark.skipif(WINDOWS, reason="POSIX file modes")
def test_fchmod_private_sets_the_mode_on_posix(tmp_path):
    fd = os.open(tmp_path / "f", os.O_WRONLY | os.O_CREAT, 0o666)
    try:
        oscompat.fchmod_private(fd, 0o600)
    finally:
        os.close(fd)
    assert stat.S_IMODE(os.stat(tmp_path / "f").st_mode) == 0o600


# ================================================================ is it alive?

class _FakeKernel32:
    """Enough of kernel32 for ``_pid_alive_windows``: a process table and an error."""

    def __init__(self, table, error=0):
        self.table, self.error, self.closed = table, error, []

    def OpenProcess(self, access, inherit, pid):  # noqa: N802 - the Windows name
        return pid if pid in self.table else 0

    def GetExitCodeProcess(self, handle, ref):  # noqa: N802
        ref._obj.value = self.table[handle]
        return 1

    def CloseHandle(self, handle):  # noqa: N802
        self.closed.append(handle)
        return 1


def test_windows_pid_probe_reads_the_exit_code():
    k = _FakeKernel32({10: 259, 11: 0})          # 259 is STILL_ACTIVE
    api = (k, lambda: k.error)
    assert oscompat._pid_alive_windows(10, api) is True
    assert oscompat._pid_alive_windows(11, api) is False
    assert k.closed == [10, 11]


def test_windows_pid_probe_counts_someone_elses_process_as_alive():
    k = _FakeKernel32({}, error=5)                 # ERROR_ACCESS_DENIED
    assert oscompat._pid_alive_windows(12, (k, lambda: k.error)) is True
    k.error = 87                                   # ERROR_INVALID_PARAMETER: no such pid
    assert oscompat._pid_alive_windows(12, (k, lambda: k.error)) is False


def test_asking_if_alive_never_signals_on_windows(monkeypatch):
    """os.kill(pid, 0) on Windows is CTRL_C_EVENT: the probe would interrupt the take."""
    monkeypatch.setattr(oscompat, "IS_WINDOWS", True)
    monkeypatch.setattr(oscompat.os, "kill", lambda *a: pytest.fail("signalled a pid to probe it"))
    monkeypatch.setattr(oscompat, "_pid_alive_windows", lambda pid: pid == 77)
    assert oscompat.pid_alive(77) is True
    assert oscompat.pid_alive(78) is False
    assert realtime._pid_alive(77) is True
    assert gui._pid_alive(77) is True


@pytest.mark.parametrize("pid", [0, -1, None, "12", True, 1.5])
def test_nonsense_pids_are_not_alive(pid):
    assert oscompat.pid_alive(pid) is False


def test_this_process_is_alive_and_a_huge_pid_is_not():
    assert oscompat.pid_alive(os.getpid()) is True
    assert oscompat.pid_alive(2 ** 22) is False


# ================================================================ child processes

def test_children_get_their_own_session_or_process_group(monkeypatch):
    monkeypatch.setattr(oscompat, "IS_WINDOWS", False)
    assert oscompat.detached_kwargs() == {"start_new_session": True}
    assert oscompat.interrupt_signal() == signal.SIGINT
    monkeypatch.setattr(oscompat, "IS_WINDOWS", True)
    assert oscompat.detached_kwargs() == {"creationflags": 0x200}   # CREATE_NEW_PROCESS_GROUP
    assert oscompat.interrupt_signal() == 1                           # CTRL_BREAK_EVENT


def test_closing_signals_are_only_ones_this_platform_has():
    names = oscompat.closing_signals()
    assert "SIGTERM" in names
    assert all(hasattr(signal, n) for n in names)
    assert ("SIGHUP" in names) == hasattr(signal, "SIGHUP")


def test_graceful_signals_lists_only_real_signals():
    assert all(hasattr(signal, n) for n in realtime.GracefulSignals.SIGNALS)
    assert {"SIGINT", "SIGTERM"} <= set(realtime.GracefulSignals.SIGNALS)


def test_sigbreak_ends_a_session_like_ctrl_c():
    g = realtime.GracefulSignals()
    sigbreak = getattr(signal, "SIGBREAK", None)
    if sigbreak is None:
        pytest.skip("SIGBREAK is Windows only")
    g._handle(sigbreak, None)
    assert g.reason == realtime.END_INTERRUPT


# ================================================================ the stop ladder on Windows

def _record(reg, pid, **kw):
    rec = realtime.LiveSession(pid=pid, kind=kw.pop("kind", "lead"), **kw)
    (reg / f"{pid}.json").write_text(json.dumps(rec.to_dict(), sort_keys=True))
    return rec


def test_windows_ladder_asks_with_ctrl_break_and_never_terminates_unasked(
        isolated_registry, monkeypatch):
    """TerminateProcess is SIGKILL: no handler, no tail.  It waits for --force."""
    monkeypatch.setattr(oscompat, "IS_WINDOWS", True)
    monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
    sent = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append(sig))
    rec = _record(isolated_registry, 4242, socket=None)
    out = realtime.stop_session(rec, timeout=0.2, poll=0.02)
    assert sent == [1]                                   # CTRL_BREAK_EVENT only
    assert out["tried"] == ["CTRL_BREAK_EVENT"]
    assert out["stopped"] is False and "--force" in out["detail"]
    assert "TerminateProcess" in out["detail"]


def test_windows_ladder_ctrl_break_gets_the_whole_graceful_budget(isolated_registry, monkeypatch):
    monkeypatch.setattr(oscompat, "IS_WINDOWS", True)
    t_sent = {}
    monkeypatch.setattr(os, "kill", lambda pid, sig: t_sent.setdefault(sig, time.monotonic()))
    monkeypatch.setattr(realtime, "_pid_alive",
                        lambda pid: 1 not in t_sent or time.monotonic() - t_sent[1] < 1.0)
    rec = _record(isolated_registry, 4243, socket=None)
    out = realtime.stop_session(rec, timeout=10.0, poll=0.02)
    assert out["stopped"] is True and out["how"] == "signal:CTRL_BREAK_EVENT"
    assert list(t_sent) == [1]


def test_windows_force_terminates_and_says_so(isolated_registry, monkeypatch):
    monkeypatch.setattr(oscompat, "IS_WINDOWS", True)
    sent = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append(sig))
    monkeypatch.setattr(realtime, "_pid_alive", lambda pid: signal.SIGTERM not in sent)
    rec = _record(isolated_registry, 4244, socket=None)
    out = realtime.stop_session(rec, timeout=0.2, poll=0.02, force=True)
    assert sent == [1, signal.SIGTERM]
    assert out["stopped"] is True and out["forced"] is True
    assert out["how"] == "signal:TerminateProcess"
    assert out["tried"] == ["CTRL_BREAK_EVENT", "TerminateProcess"]


def test_windows_ladder_survives_a_session_in_another_console(isolated_registry, monkeypatch):
    """CTRL_BREAK only reaches a group on this console; elsewhere it fails, and says so."""
    monkeypatch.setattr(oscompat, "IS_WINDOWS", True)
    monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)

    def refuse(pid, sig):
        raise OSError(87, "The parameter is incorrect")
    monkeypatch.setattr(os, "kill", refuse)
    rec = _record(isolated_registry, 4245, socket=None)
    out = realtime.stop_session(rec, timeout=0.2, poll=0.02)
    assert out["stopped"] is False
    assert out["tried"][0].startswith("CTRL_BREAK_EVENT-failed")


def test_windows_silences_a_stuck_play_but_never_terminates_a_take(isolated_registry, monkeypatch):
    """Play keeps no records, so ending it costs nothing; a take waits for --force."""
    monkeypatch.setattr(oscompat, "IS_WINDOWS", True)
    sent = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append(sig))
    monkeypatch.setattr(realtime, "_pid_alive", lambda pid: signal.SIGTERM not in sent)
    play = _record(isolated_registry, 4246, kind="play", socket=None)
    out = realtime.stop_session(play, timeout=30.0, poll=0.02)
    assert sent == [1, signal.SIGTERM] and out["stopped"] is True
    assert "no records" in out["detail"]
    sent.clear()
    monkeypatch.setattr(realtime, "_pid_alive", lambda pid: True)
    take = _record(isolated_registry, 4247, kind="lead", record_dir="takes/x", socket=None)
    out = realtime.stop_session(take, timeout=0.2, poll=0.02)
    assert sent == [1] and out["stopped"] is False


@pytest.mark.skipif(WINDOWS, reason="the POSIX ladder")
def test_the_posix_ladder_is_unchanged(monkeypatch):
    monkeypatch.setattr(oscompat, "IS_WINDOWS", False)
    ladder, kill_name, kill_sig = realtime._stop_ladder()
    assert ladder == ("SIGINT", "SIGTERM")
    assert kill_name == "SIGKILL" and kill_sig == signal.SIGKILL


# ================================================================ the cue transport

def test_the_default_transport_follows_the_platform(monkeypatch):
    assert realtime.default_transport() == ("unix" if hasattr(socket, "AF_UNIX") else "tcp")
    monkeypatch.delattr(socket, "AF_UNIX", raising=False)
    assert realtime.default_transport() == "tcp"


def test_the_transport_env_is_checked(monkeypatch):
    monkeypatch.setenv(realtime.TRANSPORT_ENV, "tcp")
    assert realtime.default_transport() == "tcp"
    monkeypatch.setenv(realtime.TRANSPORT_ENV, "carrier-pigeon")
    with pytest.raises(realtime.RealtimeError, match="must be one of"):
        realtime.default_transport()


class TestTcpTransport:
    """The Windows transport, forced, so it is proven on every OS."""

    def test_cue_status_and_stop_round_trip(self):
        path = _short("rt")
        with realtime.CueServer(path, kind="lead", transport="tcp") as srv:
            assert realtime.send_cue("darker", socket_path=path, source="cli")["ok"] is True
            assert realtime.send_status(socket_path=path)["session"] == "lead"
            assert realtime.send_stop(socket_path=path)["stopping"] is True
            assert srv.drain() == [{"text": "darker", "at": "next-phrase", "ttl_bars": 16}]
            assert srv.stopped.is_set()

    def test_it_listens_on_loopback_only(self):
        path = _short("lo")
        with realtime.CueServer(path, transport="tcp") as srv:
            host, port = srv._sock.getsockname()[:2]
            assert host == "127.0.0.1"
            assert realtime._read_endpoint(path)[0] == port

    @pytest.mark.skipif(WINDOWS, reason="POSIX file modes; Windows relies on the folder's ACL")
    def test_the_endpoint_file_is_private(self):
        path = _short("mode")
        with realtime.CueServer(path, transport="tcp"):
            assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    @pytest.mark.parametrize("auth", [None, "", "0" * 64, 7])
    def test_a_request_without_the_token_is_refused_and_does_nothing(self, auth):
        """A loopback port is reachable by anything on the machine; the token is the lock."""
        path = _short("auth")
        with realtime.CueServer(path, transport="tcp") as srv:
            port, _token = realtime._read_endpoint(path)
            for req in ({"text": "darker"}, {"op": "stop"}):
                if auth is not None:
                    req["auth"] = auth
                with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
                    s.sendall((json.dumps(req) + "\n").encode())
                    reply = json.loads(s.recv(4096).decode().splitlines()[0])
                assert reply == {"error": "not authorised for this session", "ok": False}
            assert srv.drain() == [] and not srv.stopped.is_set()

    def test_junk_does_not_take_the_server_down(self):
        path = _short("junk")
        with realtime.CueServer(path, transport="tcp") as srv:
            port, _ = realtime._read_endpoint(path)
            for payload in (b"not json\n", b"[1,2,3]\n", b"\n"):
                with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
                    s.sendall(payload)
                    assert json.loads(s.recv(4096).decode().splitlines()[0])["ok"] is False
            assert realtime.send_cue("darker", socket_path=path)["ok"] is True
            assert [c["text"] for c in srv.drain()] == ["darker"]

    def test_a_second_session_refuses_to_steal_it(self):
        path = _short("dup")
        with realtime.CueServer(path, transport="tcp"):
            with pytest.raises(realtime.RealtimeError, match="already listening"):
                realtime.CueServer(path, transport="tcp").start()
            assert realtime.send_status(socket_path=path)["ok"] is True

    def test_a_crashed_sessions_file_is_reclaimed(self):
        path = _short("stale")
        dead = socket.socket()
        dead.bind(("127.0.0.1", 0))
        port = dead.getsockname()[1]
        dead.close()                                  # nothing listens there now
        Path(path).write_text(json.dumps({"port": port, "token": "ab" * 32, "pid": 1}))
        with realtime.CueServer(path, transport="tcp"):
            assert realtime.send_cue("darker", socket_path=path)["ok"] is True

    def test_a_port_now_used_by_something_else_is_not_mistaken_for_a_session(self):
        """After a crash the port can be anyone's.  Only a reply with our token counts."""
        path = _short("squat")
        squatter = socket.socket()
        squatter.bind(("127.0.0.1", 0))
        squatter.listen(1)
        try:
            Path(path).write_text(json.dumps({"port": squatter.getsockname()[1],
                                              "token": "cd" * 32, "pid": 1}))
            assert realtime._socket_is_live(path) is False
        finally:
            squatter.close()

    def test_close_removes_its_own_file_only(self):
        path = _short("close")
        srv = realtime.CueServer(path, transport="tcp").start()
        assert Path(path).exists()
        srv.close()
        assert not Path(path).exists()
        Path(path).write_text(json.dumps({"port": 9, "token": "ef" * 32}))
        realtime.CueServer(path, transport="tcp").close()   # never started: not its file
        assert Path(path).exists()
        Path(path).unlink()

    def test_a_long_path_is_fine_because_no_kernel_sees_it(self, tmp_path):
        path = str(tmp_path / ("x" * 120) / "plume.sock")
        Path(path).parent.mkdir()
        with realtime.CueServer(path, transport="tcp"):
            assert realtime.send_cue("darker", socket_path=path)["ok"] is True

    def test_nothing_there_is_the_same_clean_error(self):
        path = _short("absent")
        with pytest.raises(realtime.RealtimeError, match="no running plume session"):
            realtime.send_cue("darker", socket_path=path)

    def test_the_env_switches_a_session_to_tcp(self, monkeypatch):
        monkeypatch.setenv(realtime.TRANSPORT_ENV, "tcp")
        path = _short("env")
        with realtime.CueServer(path) as srv:
            assert srv.transport == "tcp"
            assert realtime._is_endpoint_file(path)
            assert realtime.send_cue("darker", socket_path=path)["ok"] is True

    def test_plume_stop_ends_a_tcp_session_over_its_endpoint(self, isolated_registry, monkeypatch):
        killed = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: sig and killed.append(sig))
        path = _short("stop")
        with realtime.CueServer(path, kind="lead", transport="tcp") as srv:
            rec = _record(isolated_registry, os.getpid(), socket=path)
            out = realtime.stop_session(rec, timeout=0.2, poll=0.02)
            assert out["stopped"] is True and out["how"] == "socket"
            assert srv.stopped.is_set()
        assert killed == []


def test_a_machine_without_unix_sockets_still_takes_cues(monkeypatch):
    """What Windows gets by default: no AF_UNIX anywhere, and the cue still lands."""
    path = _short("nounix")
    monkeypatch.delattr(socket, "AF_UNIX", raising=False)
    with realtime.CueServer(path, kind="lead") as srv:
        assert srv.transport == "tcp"
        assert realtime.send_cue("busier", socket_path=path)["ok"] is True
        assert [c["text"] for c in srv.drain()] == ["busier"]


# ================================================================ devices

#: One microphone and one pair of speakers, as PortAudio lists them on Windows:
#: once per host API, MME's names cut to 31 characters.
_WINDOWS_LISTING = [
    {"name": "Microsoft Sound Mapper - Input", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Microphone (USB Audio Interfac", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Microsoft Sound Mapper - Output", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Speakers (USB Audio Interface)", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Primary Sound Capture Driver", "hostapi": 1, "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Microphone (USB Audio Interface)", "hostapi": 1, "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Microphone (USB Audio Interface)", "hostapi": 2, "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Speakers (USB Audio Interface)", "hostapi": 2, "max_input_channels": 0, "max_output_channels": 2},
]


class _Default:
    hostapi = 0


class _WindowsSd:
    default = _Default()

    def query_devices(self, device=None, kind=None):
        if kind == "output":
            return _WINDOWS_LISTING[2]
        return list(_WINDOWS_LISTING)


def test_one_host_api_keeps_each_device_once_with_its_real_index():
    kept = realtime.one_host_api(_WINDOWS_LISTING, 0)
    assert [i for i, _ in kept] == [0, 1, 2, 3]
    names = [d["name"] for _, d in kept]
    assert len(names) == len(set(names))


def test_without_host_apis_nothing_is_dropped():
    mac = [{"name": "Built-in Microphone", "max_input_channels": 1, "max_output_channels": 0}]
    assert realtime.one_host_api(mac, None) == [(0, mac[0])]
    assert realtime.one_host_api(mac, 0) == [(0, mac[0])]    # none say which: keep all


def test_on_windows_a_device_is_listed_and_opened_once(monkeypatch):
    monkeypatch.setattr(oscompat, "IS_WINDOWS", True)
    sd = _WindowsSd()
    rows = realtime.input_devices(sd)
    assert [r["name"] for r in rows].count("Speakers (USB Audio Interface)") == 1
    route = realtime.resolve_input(sd, input_device="USB Audio")
    assert route.device[0] == 1                       # the default host API's row
    assert realtime._resolve_device(sd, "USB Audio") == (1, 3)   # both ends in one host API


def test_off_windows_the_listing_is_left_alone(monkeypatch):
    monkeypatch.setattr(oscompat, "IS_WINDOWS", False)
    assert len(realtime.input_devices(_WindowsSd())) == len(_WINDOWS_LISTING)


def test_windows_default_device_names_are_not_cards():
    from plume import guitake
    rows = [{"name": d["name"], "inputs": d["max_input_channels"],
             "outputs": d["max_output_channels"]} for d in _WINDOWS_LISTING[:4]]
    outs = [o["name"] for o in guitake.describe_outputs(rows)]
    assert outs == ["Speakers (USB Audio Interface)"]


# ================================================================ voice

@pytest.mark.parametrize("system,machine,want", [
    ("darwin", "arm64", ("mlx", "faster")),
    ("darwin", "x86_64", ("faster",)),
    ("linux", "x86_64", ("faster",)),
    ("linux", "aarch64", ("faster",)),
    ("win32", "AMD64", ("faster",)),
])
def test_voice_picks_a_backend_that_runs_here(system, machine, want):
    assert voice.backends_for(system, machine) == want


def test_voice_falls_back_to_faster_whisper_on_apple_silicon_without_mlx():
    have = {"faster_whisper"}
    found = voice.choose_backend("darwin", "arm64", lambda m: object() if m in have else None)
    assert found == "faster"


@pytest.mark.parametrize("system,machine,package", [
    ("darwin", "arm64", "mlx-whisper"), ("linux", "x86_64", "faster-whisper"),
    ("win32", "AMD64", "faster-whisper")])
def test_voice_without_a_backend_names_the_one_to_install(system, machine, package):
    with pytest.raises(RuntimeError, match=package):
        voice.choose_backend(system, machine, lambda m: None)


def test_faster_whisper_is_driven_without_downloading_anything(monkeypatch):
    """The faster-whisper call shape, against a stand-in: no model, no network."""
    import types
    made = []

    class Model:
        def __init__(self, name, **kw):
            made.append((name, kw))

        def transcribe(self, audio, **kw):
            assert kw["language"] == "en" and kw["condition_on_previous_text"] is False
            return iter([types.SimpleNamespace(text=" plume,"), types.SimpleNamespace(text=" stop")]), None

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = Model
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr(voice, "_FASTER_MODEL", None)
    import numpy as np
    assert voice._whisper_faster(np.zeros(16000, dtype=np.float32)) == " plume, stop"
    voice._whisper_faster(np.zeros(16000, dtype=np.float32))
    assert [m[0] for m in made] == ["large-v3-turbo"]       # loaded once


def test_the_band_is_ended_even_when_the_interrupt_cannot_be_sent(tmp_path):
    """Windows with no console to send CTRL_BREAK through: wait, then end it."""
    class NoConsole:
        def __init__(self, cmd, **kw):
            self.code, self.killed = None, False

        def poll(self):
            return self.code

        def send_signal(self, sig):
            raise OSError(6, "The handle is invalid")

        def wait(self, timeout=None):
            if self.code is None:
                import subprocess
                raise subprocess.TimeoutExpired("plume", timeout)
            return self.code

        def kill(self):
            self.code, self.killed = -9, True

    band = voice.Band(prompt="p", base=tmp_path, write=lambda s: None, popen=NoConsole)
    band.start()
    proc = band._proc
    import unittest.mock
    with unittest.mock.patch.object(voice, "STOP_GRACE_S", 0.01):
        band.stop()
    assert proc.killed


# ================================================================ the gui's folders

def test_the_gui_does_not_refuse_a_machine_without_unix_sockets(monkeypatch, tmp_path, capsys):
    """It used to say "needs macOS or Linux".  The cue transport now has a fallback."""
    monkeypatch.delattr(socket, "AF_UNIX", raising=False)
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))
    served = []
    gui.serve(takes=tmp_path / "takes", open_browser=False, wait=served.append)
    assert len(served) == 1
    assert "http://127.0.0.1:" in capsys.readouterr().out


def test_without_folder_handles_a_take_folder_is_rechecked_by_path(tmp_path, monkeypatch):
    """Windows cannot open a folder by handle: the path is re-checked instead."""
    monkeypatch.setattr(oscompat, "DIR_FDS", False)
    folder = tmp_path / "take"
    folder.mkdir()
    (folder / "human.wav").write_bytes(b"RIFF")
    where = gui._open_folder(folder)
    assert where == folder
    fd = gui._open_in(where, "human.wav", oscompat.read_flags())
    with os.fdopen(fd, "rb") as f:
        assert f.read() == b"RIFF"
    gui._close_folder(where)
    if _can_symlink(tmp_path):
        (tmp_path / "link").symlink_to(folder, target_is_directory=True)
        with pytest.raises(OSError):
            gui._open_folder(tmp_path / "link")
        (folder / "elsewhere.wav").symlink_to(folder / "human.wav")
        with pytest.raises(OSError):
            gui._open_in(where, "elsewhere.wav", oscompat.read_flags())


def test_without_folder_handles_keep_still_writes_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(oscompat, "DIR_FDS", False)
    folder = tmp_path / "take"
    folder.mkdir()
    fd = gui._open_in(folder, ".k.tmp", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(fd, b"{}")
    os.close(fd)
    gui._replace_in(folder, ".k.tmp", "keep.json")
    assert (folder / "keep.json").read_bytes() == b"{}"
    assert not (folder / ".k.tmp").exists()


def test_ctypes_byref_has_the_attribute_the_fake_kernel_uses():
    """Guards the fake above: a byref object exposes what it points at as ``_obj``."""
    v = ctypes.c_ulong(0)
    ctypes.byref(v)._obj.value = 3
    assert v.value == 3


# ================================================================ CI shards

def _collected(shard: str | None) -> set[str]:
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "PLUME_TEST_SHARD"}
    if shard is not None:
        env["PLUME_TEST_SHARD"] = shard
    here = Path(__file__).resolve().parent
    done = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
                           "--collect-only", str(here / "test_theory.py")],
                          capture_output=True, text=True, env=env, cwd=here.parent)
    return {line for line in done.stdout.splitlines() if "::" in line}


def test_ci_shards_split_the_suite_with_nothing_lost_or_doubled():
    whole = _collected(None)
    parts = [_collected(f"{k}/3") for k in (1, 2, 3)]
    assert whole and all(parts)
    assert set().union(*parts) == whole
    assert sum(len(p) for p in parts) == len(whole)
