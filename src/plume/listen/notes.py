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
``hop_s + _RIGHT_MARGIN`` frames after it sounded, at most about 1.2 s.

Cost, measured: 40 to 50 ms of one core per window with the resampling, so one
window a second is about 5 % of one core, on one thread (the session is held to
one).  Memory: about 70 MB over a process that has numpy and scipy loaded (the
runtime's import, the 230 kB model's session and its working buffers), and 2 s
of audio.  Both are fixed: the window never grows and at most one runs per
``hop_s`` of input.  It runs on the analysis thread, never in the audio
callback.
"""

from __future__ import annotations

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


class NoteListener:
    """Audio blocks in, per-frame note probabilities out.

    :meth:`process` takes mono float blocks at ``sr`` and returns the frames that
    became available, as ``(t_s, probs)``: ``t_s`` is the frame's centre on the
    input's own clock, ``probs`` the 88 key probabilities.
    """

    def __init__(self, sr: int = 48000, *, hop_s: float = 1.0, session=None):
        self._sr = int(sr)
        g = gcd(self._sr, MODEL_SR)
        self._up, self._down = MODEL_SR // g, self._sr // g
        self._sess = session if session is not None else _session(MODEL_PATH)
        self._in_name = self._sess.get_inputs()[0].name
        names = [o.name for o in self._sess.get_outputs()]
        # basic-pitch's export: :0 contour, :1 note, :2 onset
        self._note_name = next(n for n in names if n.endswith(":1"))
        # input kept: one model window plus a little, at the input's rate
        self._keep = int(np.ceil((MODEL_N + 64) * self._sr / MODEL_SR))
        self._hop_in = int(round(hop_s * self._sr))
        self._buf = np.zeros(0, dtype=np.float32)
        self._seen = 0            # input samples received
        self._since = 0           # input samples since the last window ran
        self._done_frame = 0      # model frames (on the stream clock) already returned

    def process(self, block) -> list[tuple[float, np.ndarray]]:
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return []
        self._buf = np.concatenate([self._buf, x])[-self._keep:]
        self._seen += x.size
        self._since += x.size
        if self._since < self._hop_in:
            return []
        self._since = 0
        return self._run()

    def _run(self) -> list[tuple[float, np.ndarray]]:
        from scipy.signal import resample_poly  # noqa: PLC0415

        y = resample_poly(self._buf, self._up, self._down).astype(np.float32)
        end = self._seen * MODEL_SR / self._sr        # model samples since the start
        if y.size >= MODEL_N:
            win = y[-MODEL_N:]
        else:
            win = np.concatenate([np.zeros(MODEL_N - y.size, np.float32), y])
        start = end - MODEL_N                         # may be negative early on
        out = self._sess.run([self._note_name], {self._in_name: win[None, :, None]})[0][0]
        f0 = start / MODEL_HOP                        # stream frame index of out[0]
        first = max(self._done_frame, int(np.ceil(f0)))
        last = int(np.floor(f0)) + out.shape[0] - _RIGHT_MARGIN
        frames = []
        for fi in range(first, last):
            k = int(round(fi - f0))
            if 0 <= k < out.shape[0] and fi >= 0:
                frames.append(((fi + 0.5) * FRAME_S, out[k].astype(np.float32)))
        self._done_frame = max(self._done_frame, last)
        return frames


def note_listener(sr: int = 48000, **kw) -> NoteListener | None:
    """A listener, or None when the optional model runtime is not installed."""
    if not notes_available():
        return None
    try:
        return NoteListener(sr, **kw)
    except Exception:                                     # noqa: BLE001
        return None
