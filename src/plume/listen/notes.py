"""Note transcription for the key reading: a small pitch model over short windows.

The model is Spotify's basic-pitch (ICASSP 2022, Apache-2.0; the licence ships
beside the weights in ``models/``), run as ONNX through ``onnxruntime``.  It is
optional: ``pip install plume-music[notes]``.  Without it :func:`note_listener`
returns None and the key is read from the chroma histogram alone.  Setting
``PLUME_NOTES=0`` does the same with the extra installed.

Streaming.  The model takes 2 s windows of 22.05 kHz audio and returns, per
11.6 ms frame, a probability for each of the 88 piano keys that the note is
sounding.  :class:`NoteListener` keeps the last window of input, and every
``hop_s`` of new audio runs the model once over the window that ends now.  It
keeps the frames of the newest ``hop_s`` except the last ``_RIGHT_MARGIN``
frames, which have no audio after them yet and are taken from the next window
instead; so every frame is read with at least that much context on both sides
and appears exactly once.  A frame is therefore available
``hop_s + _RIGHT_MARGIN`` frames after it sounded, at most about 1.2 s, plus
the model's own run time.

Level.  The model scales each window to its own loudest part, so it reports
mains hum or room noise as notes whenever nothing louder is in the window.
Each frame therefore carries the level of the input under it, and the reader
of the frames decides which levels are playing (see harmony.py).

Where it runs.  Never in the audio callback.  With ``background=True`` (the live
path) the model runs on a thread of its own: the analysis thread only copies
the window across, so beat and tap frames never wait behind the model.  That
thread holds one window at most; if the model has not finished the last window
when the next is due, the waiting one is replaced, and the frames it would
have given are read from the newer window, which still covers them.  With
``background=False`` the model runs inside :meth:`NoteListener.process`, which
makes the frames a pure function of the input: for tests and for replays run
faster than real time.

Cost, measured: 30 to 75 ms of one core per window with the resampling, so one
window a second is 3 to 8 % of one core, on one thread (the session is held to
one); handing a window to the background thread costs the caller under 0.1 ms.
Memory: peak resident about 140 MB for a tracker and feature extractor with the
listener against about 105 MB without it, so about 40 MB for the runtime's
import, the 230 kB model's session and its working buffers, and at most two
windows of audio.  Both are fixed: the window never grows and at most one runs
per ``hop_s`` of input.  The model's thread shares the interpreter lock: a probe
thread's 0.5 ms sleeps overran by about 1.8 ms at the 99th percentile with it
running, against about 1 ms without.  The audio callback never runs Python that
waits on it.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from math import gcd
from pathlib import Path

import numpy as np

#: The model's input: 2 s at 22.05 kHz less one FFT hop.
MODEL_SR = 22050
MODEL_HOP = 256
MODEL_N = MODEL_SR * 2 - MODEL_HOP
MODEL_FRAMES = 172
#: MIDI number of the model's lowest key (A0).
MIDI_LOW = 21
N_KEYS = 88
FRAME_S = MODEL_HOP / MODEL_SR

#: Frames at the end of each window that are not kept, because the model has not
#: heard what follows them: half the overlap basic-pitch's own file reader trims.
_RIGHT_MARGIN = 15
#: Frames the level under a frame is averaged over (about 60 ms): longer than one
#: cycle of the lowest bass notes, so a steady note has a steady level.
_LEVEL_FRAMES = 5

MODEL_PATH = Path(__file__).resolve().parent / "models" / "basic_pitch_nmp.onnx"


_SESSIONS: dict = {}


def _session(path: Path):
    """One onnxruntime session per model file, shared: a take makes a new listener
    when its count lands, and loading the model again would stall that frame."""
    if path not in _SESSIONS:
        _SESSIONS[path] = _new_session(path)
    return _SESSIONS[path]


def _new_session(path: Path):
    import onnxruntime as ort  # noqa: PLC0415 - optional extra

    opts = ort.SessionOptions()
    # One thread: it shares the machine with the audio thread and a player.
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def notes_available() -> bool:
    try:
        import onnxruntime  # noqa: F401, PLC0415
    except Exception:                                     # noqa: BLE001
        return False
    return MODEL_PATH.exists()


class _Reader:
    """Runs the model over one window and keeps the frames not yet given out."""

    def __init__(self, sr: int, session):
        self._sr = sr
        g = gcd(sr, MODEL_SR)
        self._up, self._down = MODEL_SR // g, sr // g
        self._sess = session
        self._in_name = session.get_inputs()[0].name
        names = [o.name for o in session.get_outputs()]
        # basic-pitch's export: :0 contour, :1 note, :2 onset
        self._note_name = next(n for n in names if n.endswith(":1"))
        self._done_frame = 0      # model frames (on the stream clock) already returned

    def read(self, buf: np.ndarray, seen: int) -> list[tuple[float, np.ndarray, float]]:
        """``buf``: the newest input, ending at input sample ``seen``."""
        from scipy.signal import resample_poly  # noqa: PLC0415

        y = resample_poly(buf, self._up, self._down).astype(np.float32)
        end = seen * MODEL_SR / self._sr              # model samples since the start
        if y.size >= MODEL_N:
            win = y[-MODEL_N:]
        else:
            win = np.concatenate([np.zeros(MODEL_N - y.size, np.float32), y])
        start = end - MODEL_N                         # may be negative early on
        out = self._sess.run([self._note_name], {self._in_name: win[None, :, None]})[0][0]
        # the input's level under each frame, dB re full scale
        sq = np.zeros(out.shape[0] * MODEL_HOP)
        sq[:win.size] = win.astype(np.float64) ** 2
        ms = sq.reshape(out.shape[0], MODEL_HOP).mean(axis=1)
        ms = np.convolve(ms, np.full(_LEVEL_FRAMES, 1.0 / _LEVEL_FRAMES), mode="same")
        level = 10.0 * np.log10(ms + 1e-12)
        f0 = start / MODEL_HOP                        # stream frame index of out[0]
        first = max(self._done_frame, int(np.ceil(f0)))
        last = int(np.floor(f0)) + out.shape[0] - _RIGHT_MARGIN
        frames = []
        for fi in range(first, last):
            k = int(round(fi - f0))
            if 0 <= k < out.shape[0] and fi >= 0:
                frames.append(((fi + 0.5) * FRAME_S, out[k].astype(np.float32),
                               float(level[k])))
        self._done_frame = max(self._done_frame, last)
        return frames


def _work(jobs: queue.Queue, done: deque, reader: _Reader) -> None:
    """The background thread: one window at a time until told to stop (None)."""
    while True:
        job = jobs.get()
        try:
            if job is None:
                return
            done.append(reader.read(*job))
        finally:
            jobs.task_done()


class NoteListener:
    """Audio blocks in, per-frame note probabilities out.

    :meth:`process` takes mono float blocks at ``sr`` and returns the frames that
    became available, as ``(t_s, probs, level_db)``: ``t_s`` is the frame's centre
    on the input's own clock, ``probs`` the 88 key probabilities, ``level_db`` the
    input's level under the frame.  With ``background=True`` the model runs on its
    own thread and a frame is returned by the first call after it is ready.
    """

    def __init__(self, sr: int = 48000, *, hop_s: float = 1.0, session=None,
                 background: bool = False):
        self._sr = int(sr)
        reader = _Reader(self._sr, session if session is not None else _session(MODEL_PATH))
        # input kept: one model window plus a little, at the input's rate
        self._keep = int(np.ceil((MODEL_N + 64) * self._sr / MODEL_SR))
        self._hop_in = int(round(hop_s * self._sr))
        self._buf = np.zeros(0, dtype=np.float32)
        self._seen = 0            # input samples received
        self._since = 0           # input samples since the last window ran
        self.replaced = 0         # windows the model was too late for
        self._reader = None if background else reader
        self._jobs: queue.Queue | None = None
        if background:
            self._jobs = queue.Queue(maxsize=1)
            self._done: deque = deque()
            # The thread holds the reader and the two queues, never this listener, so
            # dropping the listener (a take makes a new one when its count lands) is
            # enough to end it: see close().
            threading.Thread(target=_work, args=(self._jobs, self._done, reader),
                             name="plume-notes", daemon=True).start()

    def process(self, block) -> list[tuple[float, np.ndarray, float]]:
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        if x.size:
            self._buf = np.concatenate([self._buf, x])[-self._keep:]
            self._seen += x.size
            self._since += x.size
        due = self._since >= self._hop_in
        if due:
            self._since = 0
        if self._jobs is None:
            return self._reader.read(self._buf, self._seen) if due else []
        if due:
            job = (self._buf, self._seen)             # _buf is never written in place
            try:
                self._jobs.put_nowait(job)
            except queue.Full:
                # The model is still on an earlier window: this one replaces the
                # one waiting, and covers its frames.
                try:
                    self._jobs.get_nowait()
                    self._jobs.task_done()
                    self.replaced += 1
                except queue.Empty:
                    pass
                try:
                    self._jobs.put_nowait(job)
                except queue.Full:
                    self.replaced += 1
        frames: list = []
        while self._done:
            frames.extend(self._done.popleft())
        return frames

    def wait(self) -> None:
        """Block until the background thread has read every window handed to it."""
        if self._jobs is not None:
            self._jobs.join()

    def close(self) -> None:
        """End the background thread, if there is one."""
        jobs, self._jobs = self._jobs, None
        if jobs is None:
            return
        try:
            jobs.get_nowait()                         # a window still waiting: dropped
            jobs.task_done()
        except queue.Empty:
            pass
        jobs.put(None)

    def __del__(self):
        try:
            self.close()
        except Exception:                                 # noqa: BLE001
            pass


def note_listener(sr: int = 48000, **kw) -> NoteListener | None:
    """A listener, or None when the optional model runtime is not installed."""
    if not notes_available():
        return None
    try:
        return NoteListener(sr, **kw)
    except Exception:                                     # noqa: BLE001
        return None
