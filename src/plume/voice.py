"""Talk to a running take: "plume, stop", "plume, restart", "plume, busier".

It reads the session's own recording, so it shares the mic without opening it, and
transcribes on this machine; no audio leaves the Mac.  Commands need the wake word,
because a speech model fed a band hallucinates sentences, and because sung lyrics
must never end a take.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL = "mlx-community/whisper-large-v3-turbo"
WINDOW_S = 4.0
HOP_S = 2.0
COOLDOWN_S = 6.0          # a window hears one spoken command more than once
STOP_GRACE_S = 12.0
MIN_RMS_DB = -50.0
_STT_SR = 16000
AUDIO_CONSENT_ENV = "PLUME_ALLOW_AUDIO"

#: What the speech model tends to write for "plume".
_WAKE = r"(?:plume|plumes|plum|plumb|bloom|blume|flume)"
_COMMANDS = (
    ("restart", r"restart|start over|start again|again|reset"),
    ("stop", r"stop|end it|that's it|finish"),
    ("busier", r"busier|more|build|bigger|louder"),
    ("sparser", r"sparser|less|chill|calm|simpler|quieter"),
)
_CUES = {"busier": "busier", "sparser": "sparser"}


def command_in(text: str) -> str | None:
    """The command following the wake word in ``text``, or None."""
    t = re.sub(r"[^a-z' ]+", " ", str(text or "").lower())
    for name, words in _COMMANDS:
        if re.search(rf"\b{_WAKE}\s+(?:please\s+)?(?:{words})\b", t):
            return name
    return None


def next_take_dir(base) -> Path:
    """The first unused takesN directory under ``base``."""
    base = Path(base)
    used = [int(m.group(1)) for p in base.glob("takes*")
            if (m := re.fullmatch(r"takes(\d+)", p.name))]
    return base / f"takes{max(used, default=0) + 1}"


def _require_whisper() -> None:
    """Fail before any band starts, not after."""
    if importlib.util.find_spec("mlx_whisper") is None:
        raise RuntimeError("plume voice needs mlx-whisper: pip install -e '.[voice]'")


def _whisper(audio: np.ndarray) -> str:
    import mlx_whisper  # noqa: PLC0415 — optional, heavy, and only needed here
    return str(mlx_whisper.transcribe(
        audio, path_or_hf_repo=MODEL, language="en",
        condition_on_previous_text=False, no_speech_threshold=0.5)["text"])


class VoiceLoop:
    """Poll a take's recording; act on a heard command.  The transcriber and the
    actions are passed in, so all of this is testable without a model or a band."""

    def __init__(self, record_dir, *, act, transcribe=_whisper, clock=time.monotonic):
        self.record_dir = Path(record_dir)
        self._act = act
        self._transcribe = transcribe
        self._clock = clock
        self._read_to = 0
        self._last_fired: dict[str, float] = {}

    def retarget(self, record_dir) -> None:
        self.record_dir = Path(record_dir)
        self._read_to = 0

    def tail(self) -> np.ndarray | None:
        """The newest window, once a hop's worth of new audio has arrived."""
        try:
            with sf.SoundFile(str(self.record_dir / "human.wav")) as f:
                sr, frames = f.samplerate, f.frames
                if frames - self._read_to < int(HOP_S * sr):
                    return None
                start = max(0, frames - int(WINDOW_S * sr))
                f.seek(start)
                y = f.read(frames - start, dtype="float32", always_2d=True).mean(axis=1)
        except (OSError, RuntimeError):
            return None
        self._read_to = frames
        n = int(y.size * _STT_SR / sr)
        return np.interp(np.arange(n) * sr / _STT_SR, np.arange(y.size), y).astype(np.float32)

    def step(self) -> str | None:
        audio = self.tail()
        if audio is None or audio.size == 0:
            return None
        if 20.0 * np.log10(float(np.sqrt(np.mean(audio ** 2))) + 1e-9) < MIN_RMS_DB:
            return None
        name = command_in(self._transcribe(audio))
        if name is None or self._clock() - self._last_fired.get(name, -1e9) < COOLDOWN_S:
            return None
        self._last_fired[name] = self._clock()
        self._act(name)
        return name


def faded_out(record_dir) -> bool:
    """Did this take end itself for want of a player?"""
    from .realtime import END_SILENCE  # noqa: PLC0415
    try:
        doc = json.loads((Path(record_dir) / "ending.json").read_text())
    except (OSError, ValueError):
        return False
    return doc.get("reason") == END_SILENCE


class Band:
    """The one take this process owns.  Stopping it signals that process directly,
    so it can neither miss its own band nor stop someone else's."""

    def __init__(self, *, prompt: str, base, extra_args=(), write=print, popen=subprocess.Popen):
        self._prompt, self._base, self._extra = prompt, base, tuple(extra_args)
        self._write, self._popen = write, popen
        self._proc = None
        self.dir: Path | None = None
        self.log = Path(tempfile.gettempdir()) / "plume-voice-band.log"

    def start(self) -> Path:
        self.stop()
        self.dir = next_take_dir(self._base)
        # Claimed now: a restart before the band has written anything must not
        # record the next take over this one.
        self.dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        # Whoever started this process was asked or set it themselves (see cmd_voice);
        # the child has no terminal of its own to be asked through.
        env[AUDIO_CONSENT_ENV] = "1"
        with self.log.open("a") as log:
            self._proc = self._popen(
                [sys.executable, "-m", "plume", "lead", "--tap-start", self._prompt,
                 "--input", "acoustic", "--record", f"{self.dir}/",
                 "--seed", str(int(time.time()) % 100000), *self._extra],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log, env=env,
                # Its own process group: a terminal's Ctrl-C must reach only us, or the
                # band gets two interrupts, and a second one cuts its ending short.
                start_new_session=True)
        self._write(json.dumps({"take": str(self.dir)}))
        return self.dir

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        if not self.alive():
            return
        self._proc.send_signal(signal.SIGINT)     # its graceful ending: finish the bar
        try:
            self._proc.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=STOP_GRACE_S)


def run(*, prompt: str, base=".", extra_args=(), poll_s: float = 0.25, write=print,
        transcribe=None, popen=subprocess.Popen) -> None:
    """Start a tap-start take and keep it voice-controlled until it is over.

    A take that faded for want of a player is followed by a new one.  Any other
    ending, or any failure here, leaves no band playing with nobody in control.
    """
    if transcribe is None:
        _require_whisper()
        transcribe = _whisper
    band = Band(prompt=prompt, base=base, extra_args=extra_args, write=write, popen=popen)
    finished = {"done": False}

    def act(name: str) -> None:
        write(json.dumps({"heard": name}))
        if name == "stop":
            band.stop()
            finished["done"] = True
        elif name == "restart":
            loop.retarget(band.start())
        else:
            sent = subprocess.run([sys.executable, "-m", "plume", "cue", _CUES[name]],
                                  capture_output=True, check=False)
            if sent.returncode != 0:
                write(json.dumps({"cue_failed": name}))

    ends = (signal.SIGTERM, signal.SIGHUP)   # SIGHUP: the terminal closed; the band,
                                             # in its own session, won't get one

    def terminated(signum, _frame):
        for sig in ends:           # a second one must not interrupt stopping the band
            signal.signal(sig, signal.SIG_IGN)
        raise SystemExit(128 + signum)     # through the finally below

    if threading.current_thread() is threading.main_thread():
        for sig in ends:
            signal.signal(sig, terminated)
    try:
        loop = VoiceLoop(band.start(), act=act, transcribe=transcribe)
        while not finished["done"]:
            loop.step()
            if finished["done"] or band.alive():
                time.sleep(poll_s)
            elif faded_out(band.dir):
                loop.retarget(band.start())
            elif (band.dir / "ending.json").exists():
                write(json.dumps({"ended": True}))
                return
            else:
                raise RuntimeError(f"the band stopped unexpectedly; see {band.log}")
    finally:
        band.stop()
