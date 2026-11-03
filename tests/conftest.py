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
                 loop_human: bool = True):
        self._human = (np.zeros(0, dtype=np.float32) if human is None
                       else np.asarray(human, dtype=np.float32))
        self._loop_human = bool(loop_human)
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
        self._rate = float(rate_x)
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
            indata = np.zeros((self._n, self._in_ch), dtype=np.float32)
            if self._human.size:
                take = self._human[pos % span:][:self._n]
                if take.size:
                    indata[:take.size, 0] = take
            # THE ROOM.  What plume played a moment ago is in the mic now,
            # quieter.  Without this the fake device is an isolation booth and
            # self-masking cannot be tested at all — which is exactly how a
            # mask that reached nothing shipped.
            if self._bleed is not None and len(self._echo) == self._echo.maxlen:
                indata[:, 0] += self._echo[0] * self._bleed
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

    def query_devices(self):
        return list(self._devices)

    def Stream(self, *, blocksize, callback, **kw):  # noqa: N802 - sd's name
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


def wait_until(predicate, *, timeout: float = 5.0, poll: float = 0.01) -> bool:
    """Poll ``predicate`` until it is true or the timeout runs out."""
    until = time.monotonic() + float(timeout)
    while time.monotonic() < until:
        if predicate():
            return True
        time.sleep(poll)
    return bool(predicate())


# ------------------------------------------------------------------ fixtures

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
