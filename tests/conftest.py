"""Shared test harness — above all, a FAKE AUDIO DEVICE.

WHY THIS FILE EXISTS.  plume's live transports (``realtime.jam`` and
``realtime.lead``) are the only part of the system with no offline twin: `sim`
and `simlead` prove the *music*, but the threads, the ring buffers, the
schedule, the socket, the recorder and the ending all live in code that only
runs when an audio device is open.  There is no audio device in CI, and there
must never be one in a test — an agent left a live session playing in a
human's room, and `tests/test_audio_consent.py` exists because of it.

So every live-path test drives the real transport through
:class:`FakeAudioStream`: a duplex stream stand-in pumped by an ordinary
thread out of a numpy buffer, with no CoreAudio, no PortAudio and no sound.
Two copies of roughly this harness had already grown privately inside
`test_session_lifecycle.py` and `test_cli_v2.py`; the round that found the
records-are-buffered and the stop-does-not-stop defects found them precisely
because the live path was under-tested, so the harness moves here, gets the
knobs those defects needed, and is available to every file and every future
round.

The knobs that matter, and the failures they reproduce:

- ``rate_x`` — pump faster than real time.  The planner paces itself against
  the playhead, so this shortens a test without changing a single decision.
- ``block_on_stop`` — ``stop()`` blocks, the way ``Pa_StopStream`` is
  documented to ("waits until all pending audio buffers have been played") and
  the way it actually did on the machine where `plume stop` failed.  This is
  the wedge, reproducible on a laptop with no audio hardware at all.
- ``block_on_abort`` — even ``Pa_AbortStream`` does not come back.  The
  pathological case that no amount of politeness inside the process can fix,
  and the one the hard-exit watchdog exists for.
- ``bleed_db`` / ``bleed_delay_s`` — **the laptop.**  What the machine plays
  comes back into the microphone, quieter and later: speakers in the same room
  as the mic, which is how Juniper actually played (an acoustic guitar into the
  built-in mic while plume came out of the built-in speakers) and the condition
  every self-masking defect lives in.  Without this the fake device is an
  isolation booth and the beat tracker can never hear itself, so the bug that
  cost a whole session is untestable by construction.
- ``latency`` — the stand-in reports a round trip the way ``sounddevice`` does,
  so ``realtime.stream_latency_hops`` aligns the mask to it.
- ``human_channel`` / ``inputs`` / ``bleed_channels`` — **the audio interface.**
  ``human`` arrives on one input channel, ``inputs`` puts other material on other
  channels (a voice mic, or a loud decoy on the channel plume must not read), and
  ``bleed_channels`` says which inputs can hear the room.  A guitar plugged in is a
  channel the band cannot reach.
- ``hold`` — a callable; the device waits while it is true.  Which lets a test
  that reads the output by the sample keep up on a loaded machine, where a fast
  playhead outruns the planner (tests/test_looper_live.py).

Nothing here is autouse.  Files that do not ask for it are unaffected.
"""
from __future__ import annotations

import collections
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plume import realtime  # noqa: E402

SR = realtime.SR

def _rate_scale() -> float:
    try:
        scale = float(os.environ.get("PLUME_TEST_RATE_SCALE", "1") or 1)
    except ValueError:
        return 1.0
    return min(1.0, max(0.05, scale))


#: The longest ``hold`` can pause the fake device: a wedge must still fail the test.
HOLD_MAX_S = 30.0
#: A device table that satisfies ``_resolve_device``: one duplex device.
DUPLEX_DEVICES = [{"name": "Fake Duplex", "max_input_channels": 2,
                   "max_output_channels": 2}]


class FakeAudioStream:
    """A ``sounddevice.Stream`` stand-in: a numpy buffer in, captured out.

    Implements the whole surface plume uses — the context manager protocol
    (which is how the transports open it), plus ``start``/``stop``/``close``/
    ``abort``, so a test can assert which of those the ending actually
    reached.  The callback is driven from a daemon thread in exactly the
    shape PortAudio drives it: ``(indata, outdata, frames, time, status)``,
    fixed blocksize, ``outdata`` pre-zeroed and read back after.
    """

    def __init__(self, human=None, *, blocksize: int = 256, callback=None,
                 rate_x: float = 12.0, channels=(1, 2), warmup_s: float = 1.0,
                 block_on_stop: bool = False, block_on_abort: bool = False,
                 block_timeout: float = 30.0, capture: bool = True,
                 bleed_db: float | None = None, bleed_delay_s: float = 0.020,
                 latency: tuple[float, float] | None = None,
                 loop_human: bool = True, human_channel: int = 0,
                 inputs: dict | None = None,
                 bleed_channels: tuple[int, ...] = (0,), hold=None):
        self._human_ch = int(human_channel)
        self._inputs = {int(k): np.asarray(v, dtype=np.float32)
                        for k, v in (inputs or {}).items()}
        self._bleed_channels = tuple(int(c) for c in bleed_channels)
        self._human = (np.zeros(0, dtype=np.float32) if human is None
                       else np.asarray(human, dtype=np.float32))
        self._loop_human = bool(loop_human)
        #: while this says so, the device waits, as if the test had paused time.
        self._hold = hold
        #: speaker -> room -> mic path.  None = headphones (perfect isolation).
        self._bleed = (None if bleed_db is None
                       else float(10.0 ** (float(bleed_db) / 20.0)))
        #: the delay is rounded to whole callback blocks (5.3 ms at 256/48k),
        #: which is plenty of resolution for a 20 ms room and keeps the pump a
        #: deque of blocks rather than an index arithmetic puzzle.
        self._bleed_blocks = max(
            1, int(round(float(bleed_delay_s) * SR / max(1, int(blocksize)))))
        self._echo: collections.deque = collections.deque(
            maxlen=self._bleed_blocks)
        #: what ``sounddevice`` calls ``Stream.latency``: (input, output), s.
        self.latency = ((0.010, 0.010) if latency is None
                        else (float(latency[0]), float(latency[1])))
        self._n = int(blocksize)
        self._cb = callback
        # A shared CI runner falls behind a fast playhead that a laptop keeps up
        # with, and the failure reads as a musical one.  PLUME_TEST_RATE_SCALE
        # slows every fake device by the same factor (never speeds one up).
        self._rate = float(rate_x) * _rate_scale()
        self._warmup_s = float(warmup_s)
        self._in_ch = int(channels[0]) if isinstance(channels, (tuple, list)) else 1
        self._out_ch = int(channels[1]) if isinstance(channels, (tuple, list)) else 2
        self._capture = bool(capture)
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        #: released by ``abort`` (or the timeout) to unblock a blocking stop
        self._release = threading.Event()
        self._block_on_stop = bool(block_on_stop)
        self._block_on_abort = bool(block_on_abort)
        self._block_timeout = float(block_timeout)

        self.captured: list[np.ndarray] = []
        self.blocks = 0
        self.started = False
        self.aborted = False
        self.stopped = False
        self.closed = False
        #: how long the caller spent inside ``stop`` (the wedge, measured)
        self.stop_blocked_s = 0.0

    # -- the pump -------------------------------------------------------

    def _pump(self):
        pos = 0
        real_period = self._n / float(SR)
        fast_period = real_period / max(1e-6, self._rate)
        t_start = time.perf_counter()
        t_next = t_start
        # Hard cap: a wedged planner must fail the test, not hang the suite.
        span = max(1, self._human.shape[0])
        max_blocks = int(8.0 * (span / self._n) + 40000)
        while not self._stop_evt.is_set() and self.blocks < max_blocks:
            # WARM-UP, and why it is not optional.  `rate_x` accelerates the
            # DEVICE, not the planner: at 12x the playhead eats twelve seconds
            # of schedule per wall second, and if the planner has not placed
            # bar 0 by the time the playhead passes it, `write_at` clamps the
            # whole bar into the past and the machine plays silence.  On an
            # idle machine the planner wins that race easily; under load
            # (several test runs at once) it does not, and the failure looks
            # exactly like a musical regression — "the machine never sounded"
            # — while being nothing but CPU contention.  A test that lies
            # about WHICH thing broke is worse than a slow one, so the first
            # second runs at real time and lets the planner get its bar ahead.
            period = (real_period
                      if time.perf_counter() - t_start < self._warmup_s
                      else fast_period)
            if self._hold is not None and self._hold():
                give_up = time.perf_counter() + HOLD_MAX_S
                while (self._hold() and not self._stop_evt.is_set()
                       and time.perf_counter() < give_up):
                    time.sleep(0.001)
                t_next = time.perf_counter()
            indata = np.zeros((self._n, self._in_ch), dtype=np.float32)
            if self._human.size:
                take = self._human[pos % span:][:self._n]
                if take.size:
                    indata[:take.size, self._human_ch] = take
            for ch, sig in self._inputs.items():
                take = sig[pos % max(1, sig.size):][:self._n]
                indata[:take.size, ch] = take
            # THE ROOM.  What plume played a moment ago is in the mic now,
            # quieter.  Without this the fake device is an isolation booth and
            # self-masking cannot be tested at all — which is exactly how a
            # mask that reached nothing shipped.
            if self._bleed is not None and len(self._echo) == self._echo.maxlen:
                for ch in self._bleed_channels:
                    if ch < self._in_ch:
                        indata[:, ch] += self._echo[0] * self._bleed
            pos += self._n
            outdata = np.zeros((self._n, self._out_ch), dtype=np.float32)
            if self._cb is not None:
                self._cb(indata, outdata, self._n, None, None)
            if self._bleed is not None:
                self._echo.append(outdata.mean(axis=1).astype(np.float32))
            if self._capture:
                self.captured.append(outdata.copy())
            self.blocks += 1
            t_next += period
            slack = t_next - time.perf_counter()
            if slack > 0:
                time.sleep(slack)

    # -- the sounddevice.Stream surface ---------------------------------

    def start(self):
        if self._thread is None:
            self.started = True
            self._thread = threading.Thread(target=self._pump,
                                            name="fake-audio-stream",
                                            daemon=True)
            self._thread.start()
        return self

    def stop(self, ignore_errors=True):  # noqa: ARG002 - sounddevice's signature
        """``Pa_StopStream``: waits for pending buffers.  Optionally forever."""
        t0 = time.monotonic()
        if self._block_on_stop and not self._release.is_set():
            self._release.wait(self._block_timeout)
        self.stop_blocked_s = time.monotonic() - t0
        self.stopped = True
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def abort(self, ignore_errors=True):  # noqa: ARG002 - sounddevice's signature
        """``Pa_AbortStream``: returns at once, discarding what is queued."""
        self.aborted = True
        if not self._block_on_abort:
            self._release.set()
        self._stop_evt.set()

    def close(self, ignore_errors=True):  # noqa: ARG002
        self.closed = True
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def release(self) -> None:
        """Let a blocked ``stop`` return — what a test does after asserting
        that the process WOULD have exited (we cannot really ``os._exit``
        inside pytest)."""
        self._release.set()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        self.close()
        return False

    # -- assertions helpers ---------------------------------------------

    @property
    def output(self) -> np.ndarray:
        if not self.captured:
            return np.zeros((0, self._out_ch), dtype=np.float32)
        return np.concatenate(self.captured, axis=0)

    def sounded(self, floor: float = 1e-4) -> bool:
        out = self.output
        return bool(out.size) and float(np.abs(out).max()) > floor


class FakeAudioDevice:
    """A whole stand-in ``sounddevice`` module: a device table and a Stream."""

    def __init__(self, human=None, *, devices=None, **stream_kw):
        self._human = human
        self._devices = list(DUPLEX_DEVICES if devices is None else devices)
        self._stream_kw = dict(stream_kw)
        self.streams: list[FakeAudioStream] = []
        #: what each ``Stream(...)`` was opened with: device and channels
        self.opened: list[dict] = []
        self.recorded: list[dict] = []

    def rec(self, frames, *, samplerate, channels, dtype, device):  # noqa: ARG002
        """``sounddevice.rec``: what ``plume inputs --meter`` reads, no pump."""
        self.recorded.append({"device": device, "channels": channels})
        out = np.zeros((int(frames), int(channels)), dtype=np.float32)
        kw = self._stream_kw
        sources = dict(kw.get("inputs") or {})
        if self._human is not None:
            sources[kw.get("human_channel", 0)] = self._human
        for ch, sig in sources.items():
            sig = np.asarray(sig, dtype=np.float32)[:int(frames)]
            out[:sig.size, int(ch)] = sig
        return out

    def wait(self):
        return None

    def query_devices(self):
        return list(self._devices)

    def Stream(self, *, blocksize, callback, **kw):  # noqa: N802 - sd's name
        self.opened.append({"device": kw.get("device"),
                            "channels": kw.get("channels", (1, 2))})
        s = FakeAudioStream(self._human, blocksize=blocksize, callback=callback,
                            channels=kw.get("channels", (1, 2)),
                            **self._stream_kw)
        self.streams.append(s)
        return s

    @property
    def stream(self) -> FakeAudioStream | None:
        return self.streams[-1] if self.streams else None


class patched_module:  # noqa: N801 - used as a context manager
    """``sys.modules`` swap that works outside a monkeypatch fixture."""

    def __init__(self, name, mod):
        self._name, self._mod, self._old, self._had = name, mod, None, False

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


def fake_audio(human=None, **kw) -> FakeAudioDevice:
    """Build a fake ``sounddevice`` module for ``patched_module``."""
    return FakeAudioDevice(human, **kw)


def short_socket_path(name: str) -> str:
    """An AF_UNIX path short enough for the kernel's ~104-byte ``sun_path``.

    pytest's ``tmp_path`` routinely blows that limit and the resulting OSError
    looks nothing like "your path is too long", so sockets live directly in
    the temp dir under a short name.
    """
    p = Path(tempfile.gettempdir()) / f"plume-t{os.getpid()}-{name}.sock"
    try:
        p.unlink()
    except OSError:
        pass
    return str(p)


# ------------------------------------------------------------------ other platforms
#
# Each mark says exactly which POSIX feature a test needs, so a Windows run skips
# only that and the reason reads as a fact.  The Windows-side logic each one
# guards is tested on every OS in tests/test_cross_platform.py.

WINDOWS = sys.platform == "win32"
#: POSIX file modes: Windows' os.chmod only toggles read-only.
needs_posix_modes = pytest.mark.skipif(WINDOWS, reason="POSIX file modes")
#: SIGINT/SIGTERM/SIGKILL sent with os.kill: on Windows that is TerminateProcess.
needs_posix_signals = pytest.mark.skipif(WINDOWS, reason="POSIX signals")
needs_fifo = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs here")
#: An endless device file: POSIX has one, Windows does not.
DEV_ZERO = os.path.join(os.sep, "dev", "zero")
needs_dev_zero = pytest.mark.skipif(not os.path.exists(DEV_ZERO), reason="no endless device file here")
needs_unix_sockets = pytest.mark.skipif(not hasattr(__import__("socket"), "AF_UNIX"),
                                        reason="no AF_UNIX here")


def _can_symlink() -> bool:
    probe = Path(tempfile.mkdtemp(prefix="plume-link-"))
    try:
        (probe / "l").symlink_to(probe, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        return False
    finally:
        try:
            (probe / "l").unlink()
        except OSError:
            pass
        probe.rmdir()


#: Windows makes symlinks only for administrators or in developer mode.
needs_symlinks = pytest.mark.skipif(not _can_symlink(), reason="cannot make symlinks here")


def _has_dir_fds() -> bool:
    from plume import oscompat
    return oscompat.DIR_FDS


#: A folder opened by handle, so a swap of its name after the check changes nothing.
needs_dir_fds = pytest.mark.skipif(not _has_dir_fds(), reason="no dir_fd support here")


def signal_self(sig) -> None:
    """Deliver ``sig`` to this process, the way another process would.

    On POSIX that is ``os.kill``.  On Windows ``os.kill`` on our own pid would be
    TerminateProcess, ending the test run, so the handler is invoked by raise_signal.
    """
    import signal as _signal
    if WINDOWS:
        _signal.raise_signal(sig)
    else:
        os.kill(os.getpid(), sig)


def closing_signal_params() -> list:
    """SIGTERM, plus SIGHUP or SIGBREAK where this platform has them."""
    import signal as _signal
    from plume import oscompat
    return [getattr(_signal, n) for n in oscompat.closing_signals()]


def wait_until(predicate, *, timeout: float = 5.0, poll: float = 0.01) -> bool:
    """Poll ``predicate`` until it is true or the timeout runs out."""
    until = time.monotonic() + float(timeout)
    while time.monotonic() < until:
        if predicate():
            return True
        time.sleep(poll)
    return bool(predicate())


# ------------------------------------------------------------------ CI shards
#
# The whole suite takes most of an hour on one core, and CI runs it on three
# operating systems.  ``PLUME_TEST_SHARD=K/N`` keeps the tests whose node id hashes
# to K of N, so N jobs together run every test exactly once.  Unset: everything.

SHARD_ENV = "PLUME_TEST_SHARD"


def _shard() -> tuple[int, int] | None:
    raw = os.environ.get(SHARD_ENV, "").strip()
    if not raw:
        return None
    k, _, n = raw.partition("/")
    k, n = int(k), int(n)
    if not 1 <= k <= n:
        raise pytest.UsageError(f"{SHARD_ENV} must be K/N with 1 <= K <= N, got {raw!r}")
    return k, n


def pytest_collection_modifyitems(config, items):
    shard = _shard()
    if shard is None:
        return
    import zlib
    k, n = shard
    kept, dropped = [], []
    for item in items:
        # The node id, with / separators on every OS, so a test lands in the same shard.
        nodeid = item.nodeid.replace("\\", "/")
        (kept if zlib.crc32(nodeid.encode()) % n == k - 1 else dropped).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = kept


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def fake_device_consent(monkeypatch):
    """The transports refuse to open a device without consent, and tests are no
    exception.  Only for tests that run on the FAKE device.  A child process inherits
    it, so a test using this must not start a real ``plume`` verb."""
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")


@pytest.fixture
def plume_registry(tmp_path, monkeypatch):
    """A private session registry — never the developer's real one.

    Without it a test run could list, and stop, a session a human is actually
    playing: the very incident these tests exist for, caused by its own tests.
    """
    d = tmp_path / "plume-sessions"
    d.mkdir()
    monkeypatch.setenv(realtime.SESSION_DIR_ENV, str(d))
    return d


@pytest.fixture
def fake_device():
    """Factory fixture: ``fake_device(human, rate_x=8.0)`` -> FakeAudioDevice."""
    return fake_audio


# ------------------------------------------------------------- a fake MIDI port
#
# Same rule as the audio device: no test opens a real port.  ``FakeMido`` stands
# in for the whole ``mido`` module through ``patched_module("mido", FakeMido())``.

class FakeMidiMessage:
    """The attributes plume reads off a ``mido.Message``."""

    def __init__(self, type, **kw):  # noqa: A002 - mido's name
        self.type = type
        self.channel = kw.get("channel", 0)
        for name, value in kw.items():
            setattr(self, name, value)


class FakeMidiPort:
    def __init__(self, name, callback, *, fail_close=False):
        self.name, self.callback, self.closed = name, callback, False
        self._fail_close = fail_close

    def send(self, msg):
        """What the library's thread does when a message arrives."""
        if not self.closed:
            self.callback(msg)

    def close(self):
        self.closed = True
        if self._fail_close:
            raise OSError("device is gone")


class FakeMido:
    """A stand-in ``mido`` module: named ports that can be unplugged."""

    def __init__(self, names=("Fake Keys",), *, fail_open=False, fail_list=False,
                 fail_close=False):
        self.names = list(names)
        self.fail_open, self.fail_list = fail_open, fail_list
        self.fail_close = fail_close
        self.ports: list[FakeMidiPort] = []

    def get_input_names(self):
        if self.fail_list:
            raise OSError("backend trouble")
        return list(self.names)

    def open_input(self, name, callback=None):
        if self.fail_open or name not in self.names:
            raise OSError(f"cannot open {name}")
        port = FakeMidiPort(name, callback, fail_close=self.fail_close)
        self.ports.append(port)
        return port

    @property
    def port(self) -> FakeMidiPort | None:
        return self.ports[-1] if self.ports else None
