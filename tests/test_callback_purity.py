"""The audio callback stays pure: ring copies only.

It runs on the audio device's own thread.  Waiting on anything there is an audible
glitch, and the fake device would not notice: it has no deadline.  So this watches
what the callback actually enters, on the real transports, and reads the names in
every piece of plume code it reaches, which also sees a branch this run did not take.
Every C function it calls has to be on a short list, and its Python has a line budget,
which counts work the same way on a loaded machine.

What this cannot see: the time spent inside a C function on the list, and numpy
arithmetic, which is not a function call.  Only a real device has a deadline.
"""
from __future__ import annotations

import dis
import os
import re
import socket
import sys
import threading
import time

import numpy as np
import pytest
from conftest import FakeMido, fake_audio, patched_module, short_socket_path

import plume
from plume import mind as mind_mod
from plume import realtime
from plume import roles as roles_mod

PLUME_DIR = os.path.dirname(plume.__file__)
NUMPY_DIR = os.path.dirname(np.__file__)
THIS_FILE = os.path.abspath(__file__)
RIG = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]

#: A name in callback code that means waiting, or reaching outside the rings.
FORBIDDEN_NAME = re.compile(
    r"lock|acquire|release|sleep|wait|join|looper|midi|masker|recorder|duet|mind|shared|cue",
    re.IGNORECASE)
#: C functions that wait.
FORBIDDEN_CALL = {"sleep", "acquire", "release", "wait", "join", "acquire_lock", "select"}
#: The C functions a callback may call: its rings, and nothing else.
ALLOWED_CALLS = {"asarray", "min", "setprofile"}
#: The plume files whose code may run in the callback: the transport, and the input sound
#: plume.inputfx (the effect on the "Hear me" monitoring switch), held to the same rules.
CALLBACK_FILES = {"realtime.py", "inputfx.py"}
#: The most lines of Python one callback may run.  A loop goes past this at once.
MAX_LINES_PER_BLOCK = 120
#: Her sound's own lines, counted apart: a fixed amount of work per buffer, whatever its
#: length (test_inputfx checks that the count does not grow with the buffer), so a
#: per-sample loop would still go past this at once.
MAX_SOUND_LINES_PER_BLOCK = 160
SOUND_FILE = os.sep + "inputfx.py"
WITH_OPS = {"BEFORE_WITH", "SETUP_WITH", "BEFORE_ASYNC_WITH", "LOAD_SPECIAL"}


class Watcher:
    def __init__(self):
        self.entered = {}          # code object -> True
        self.calls = set()
        self.blocks = 0
        self.lines = 0
        self.most_lines = 0
        self.sound_lines = 0
        self.most_sound_lines = 0

    def wrap(self, callback):
        def watched(*args):
            self.lines = 0
            self.sound_lines = 0
            sys.settrace(self._trace)
            sys.setprofile(self._profile)
            try:
                callback(*args)
            finally:
                sys.setprofile(None)
                sys.settrace(None)
            self.most_lines = max(self.most_lines, self.lines)
            self.most_sound_lines = max(self.most_sound_lines, self.sound_lines)
            self.blocks += 1
        return watched

    def _trace(self, frame, event, arg):
        if event == "line":
            if frame.f_code.co_filename.endswith(SOUND_FILE):
                self.sound_lines += 1
            else:
                self.lines += 1
        return self._trace

    def _profile(self, frame, event, arg):
        if event == "call":
            self.entered[frame.f_code] = True
        elif event == "c_call":
            self.calls.add(getattr(arg, "__name__", str(arg)))
            owner = getattr(arg, "__self__", None)
            if "lock" in type(owner).__name__.lower():
                self.calls.add("acquire")

    def problems(self) -> list[str]:
        out = sorted(f"called {c}" for c in self.calls & FORBIDDEN_CALL)
        out += sorted(f"called {c}, which is not on the list" for c in self.calls - ALLOWED_CALLS)
        if self.most_lines > MAX_LINES_PER_BLOCK:
            out.append(f"ran {self.most_lines} lines in one block")
        if self.most_sound_lines > MAX_SOUND_LINES_PER_BLOCK:
            out.append(f"ran {self.most_sound_lines} lines of her sound in one block")
        for code in self.entered:
            where = os.path.abspath(code.co_filename)
            if where == THIS_FILE or where.startswith(NUMPY_DIR):
                continue
            if not where.startswith(PLUME_DIR):
                out.append(f"entered {where}:{code.co_name}")
                continue
            if os.path.basename(where) not in CALLBACK_FILES:
                out.append(f"entered {os.path.basename(where)}:{code.co_name}")
            names = (*code.co_names, *code.co_freevars, *code.co_varnames)
            out += [f"{code.co_name} names {n}" for n in names if FORBIDDEN_NAME.search(n)]
            if any(i.opname in WITH_OPS for i in dis.get_instructions(code)):
                out.append(f"{code.co_name} uses a with block")
        return out

    def names_entered(self) -> set[str]:
        return {c.co_name for c in self.entered}


def watched_device(**kw):
    sd = fake_audio(None, devices=RIG, rate_x=6.0, **kw)
    watcher = Watcher()
    original = sd.Stream

    def Stream(*, blocksize, callback, **opened):  # noqa: N802 - sounddevice's name
        return original(blocksize=blocksize, callback=watcher.wrap(callback), **opened)
    sd.Stream = Stream
    return sd, watcher


@pytest.fixture
def env(tmp_path, monkeypatch, plume_registry):
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only
    return tmp_path


def run_lead(tmp_path, sock, **kw):
    realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"), status=False,
                  max_minutes=0.04, socket_path=sock, record_dir=str(tmp_path / "rec"),
                  tap_start=True, **kw)


def run_jam(tmp_path, sock, **kw):
    realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(), max_minutes=0.04,
                 socket_path=sock, record_dir=str(tmp_path / "rec"), **kw)


@pytest.mark.parametrize("midi", [None, "auto"], ids=["mic", "midi"])
@pytest.mark.parametrize("run,extra", [
    (run_lead, {"voice_channel": 2}),
    (run_jam, {}),
], ids=["lead", "jam"])
def test_the_callback_only_copies_rings(env, run, extra, midi):
    sd, watcher = watched_device()
    kw = {"midi": midi} if midi else {}
    with patched_module("sounddevice", sd), patched_module("mido", FakeMido()):
        run(env, short_socket_path("pure"), monitor=True, input_device="audio interface",
            input_channel=1, **extra, **kw)
    assert watcher.blocks > 50, "the callback barely ran, so this proves nothing"
    assert {"callback", "write", "read_into"} <= watcher.names_entered()
    assert watcher.problems() == []


@pytest.mark.parametrize("sound", ["clean", "dreamy", "lofi", "crunch"])
@pytest.mark.parametrize("run,extra", [
    (run_lead, {"voice_channel": 2}),
    (run_jam, {}),
], ids=["lead", "jam"])
def test_her_sound_on_hear_me_keeps_the_callback_pure(env, run, extra, sound):
    sd, watcher = watched_device()
    with patched_module("sounddevice", sd), patched_module("mido", FakeMido()):
        run(env, short_socket_path("snd"), monitor=True, input_device="audio interface",
            input_channel=1, sound=sound, **extra)
    assert watcher.blocks > 50, "the callback barely ran, so this proves nothing"
    assert "process" in watcher.names_entered(), "the sound never ran"
    assert watcher.problems() == []


def test_a_detect_take_with_a_monitor_leaves_the_callback_alone(env):
    sd, watcher = watched_device()
    with patched_module("sounddevice", sd):
        realtime.lead(prompt=None, detect=True, tap_start=True, seed=0, status=False,
                      role=roles_mod.RoleParams(role="lead"), max_minutes=0.04,
                      socket_path=short_socket_path("detp"), record_dir=str(env / "rec"),
                      monitor=True, input_device="audio interface", input_channel=1, direct=True)
    assert (env / "rec" / "stream.json").exists()
    assert watcher.blocks > 50, "the callback barely ran, so this proves nothing"
    assert watcher.problems() == []


class Dirty:
    """Stands in for the things a callback must not touch."""

    def __init__(self):
        self.lock = threading.Lock()

    def feed(self):
        return 1


def _dirty_callbacks():
    dirty = Dirty()
    held = threading.Lock()
    event = threading.Event()
    event.set()
    devnull = os.open(os.devnull, os.O_WRONLY)

    def takes_a_lock_by_call(*_):
        got = held.acquire(blocking=False)
        if got:
            held.release()

    def takes_a_lock_by_with(*_):
        with held:
            pass

    def takes_a_lock_with_an_innocent_name(*_):
        gate = held
        with gate:
            pass

    def sleeps(*_):
        time.sleep(0.0001)

    def waits(*_):
        event.wait(0.0)

    def reaches_the_looper(*_):
        looper = dirty
        looper.feed()

    def reaches_midi(*_):
        midi_in = dirty
        return midi_in

    def writes_to_a_descriptor(*_):
        os.write(devnull, b"x")

    def spins_on_the_clock(*_):
        until = time.perf_counter() + 0.002
        while time.perf_counter() < until:
            pass

    def spins_on_nothing(*_):
        total = 0
        for i in range(2000):
            total += i

    def allocates_a_lot(*_):
        np.ones(1000).sum()

    def asks_the_network_stack(*_):
        socket.gethostname()

    def reads_the_cue_state(*_):
        cues = dirty
        return cues.lock

    return [takes_a_lock_by_call, takes_a_lock_by_with, takes_a_lock_with_an_innocent_name,
            sleeps, waits, reaches_the_looper, reaches_midi, writes_to_a_descriptor,
            spins_on_the_clock, spins_on_nothing, allocates_a_lot, asks_the_network_stack,
            reads_the_cue_state]


@pytest.mark.parametrize("callback", _dirty_callbacks(), ids=lambda c: c.__name__)
def test_the_watcher_itself_notices(callback, monkeypatch):
    """A guard that has never failed proves nothing, so each of these has to fail it."""
    monkeypatch.setattr(sys.modules[__name__], "THIS_FILE", "")
    monkeypatch.setattr(sys.modules[__name__], "PLUME_DIR", os.path.dirname(os.path.abspath(__file__)))
    watcher = Watcher()
    watcher.wrap(callback)()
    assert watcher.problems()
