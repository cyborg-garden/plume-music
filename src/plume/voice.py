"""Talk to a running take: "plume, stop", "plume, restart", "plume, busier".

It reads the session's own recording, so it shares the mic without opening it, and
transcribes on this machine; no audio leaves the computer.  Commands need the wake word,
because a speech model fed a band hallucinates sentences, and because sung lyrics
must never end a take.

Without a voice channel it reads ``human.wav``, which is the mic with the band in it.
With ``--voice-channel`` it reads ``voice.wav``, and on a direct input it has to: there
``human.wav`` holds only the guitar.  Every window leaves a line in ``voice.ndjson``
saying what it heard, so a take with no commands can say why.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
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
from scipy.signal import resample_poly

from . import oscompat
from .looper import _LOOP_WORD

#: The same model, whisper-large-v3-turbo, packaged for each backend.  mlx-whisper
#: runs it on Apple Silicon's GPU; faster-whisper (CTranslate2) runs it everywhere
#: else, on a CUDA GPU if there is one and on the CPU if not.
MODELS = {"mlx": "mlx-community/whisper-large-v3-turbo", "faster": "large-v3-turbo"}
#: The packages behind each backend, by import name and by pip name.
_BACKEND_MODULES = {"mlx": "mlx_whisper", "faster": "faster_whisper"}
_BACKEND_PACKAGES = {"mlx": "mlx-whisper", "faster": "faster-whisper"}
MODEL = MODELS["mlx"]
WINDOW_S = 4.0
HOP_S = 2.0
COOLDOWN_S = 6.0          # a window hears one spoken command more than once
STOP_GRACE_S = 12.0
MIN_RMS_DB = -50.0
_STT_SR = 16000
AUDIO_CONSENT_ENV = "PLUME_ALLOW_AUDIO"

#: What the speech model tends to write for "plume".
_WAKE = r"(?:plume|plumes|plum|plumb|bloom|blume|flume)"
_COMMANDS = (             # stop first: when one window holds two commands, it wins
    ("stop", r"stops?|stopping|stopped|end it|that's it|thats it|finish|finished"
             # These are ordinary words too, so nothing may follow them: "enough bass".
             r"|(?:end|end this|end the take|end this take|end the song|quit|halt|done|enough"
             r"|that's enough|we're done|we are done|we're finished|i'm done|i am done|shut up"
             r"|shut it off|turn it off|kill it|wrap it up)"
             r"(?=(?:[\s,]*(?:now|please|here))*\s*(?:[,|]|$))"),
    ("restart", r"restart|start over|start again|again|reset"),
    ("busier", r"busier|busy|more|build|bigger|louder"),
    ("sparser", r"sparser|sparse|less|chill|calm|simpler|quieter"),
)
_CUES = {"busier": "busier", "sparser": "sparser"}
#: Stop and restart words with no room for an object of their own: "end it" has one.
_TAKES_NO_OBJECT = {"end it", "that's it", "thats it", "start over", "start again", "again",
                    "done", "enough", "that's enough", "we're done", "we are done",
                    "we're finished", "i'm done", "i am done", "shut up", "shut it off",
                    "turn it off", "kill it", "wrap it up"}
#: Only "plume" itself, not what the model mishears, because "the flowers bloom and the
#: rain will stop" must not count.
_REAL_WAKE = r"plumes?"
#: In the same clause as "stop", these mean it isn't an order: "don't stop", "until i stop".
_NOT_AN_ORDER = re.compile(
    r"\b(?:don't|dont|do not|not|never|non|can't|cant|cannot|won't|wont|shouldn't|shouldnt"
    r"|couldn't|couldnt|wouldn't|wouldnt|didn't|didnt|doesn't|doesnt|when|if|until|till|til"
    r"|before|after|once|while|unless|why|did)\b")
#: Asking nicely, which _NOT_AN_ORDER would otherwise read as "don't".
_POLITE = re.compile(r"\b(?:why (?:don't|dont) (?:you|we)|(?:can't|cant|won't|wont) you|if you could)\b")
#: "stop, plume", "stop it plume", "stop the band now, plume"
_STOP_THEN_WAKE = re.compile(
    r"\bstops?(?:[\s,|]+(?:it|that|this|everything|playing|right|now|please"
    r"|the\s+(?:music|band|song|drums)))*"
    rf"[\s,|]+{_REAL_WAKE}\b")
_WORDS_BEFORE_STOP = 8
_DETERMINER = r"the|a|an|my|our|that|this|these|those|all|of"
#: Words that join things up or aim elsewhere, so they can't be part of a loop's name.
_NOT_A_NAME = (rf"{_DETERMINER}|and|or|but|then|with|plus|not|too|stops?|everything|it"
               r"|band|music|songs?")   # "stop the band loop" is a stop that lost a word
#: What the band is called.  Only right before the word "loop" is it a loop's name.
_THE_BAND = r"band|drums?|music|song"
#: A loop as the verb's direct object: "the loop", "my old bass loop", "loop 1", "looping".
_LOOP_OBJECT = (
    rf"\s*(?:(?:{_DETERMINER})\s+)+"
    rf"(?:(?!(?:{_NOT_A_NAME})\b)(?!(?:{_THE_BAND})\s+(?!{_LOOP_WORD}))[a-z']+\s+){{0,2}}"
    rf"{_LOOP_WORD}|\s*{_LOOP_WORD}")
#: Or the whole next sentence: "Plume, stop. The loop."
_LOOP_AIM = rf"(?:{_LOOP_OBJECT}|[\s,]*\|[\s,|]*the\s+loop\b)"
_ENDING = (r"(?:[\s,|]*(?:now|please|right\s+now|it's\s+too\s+(?:loud|much)"
           r"|and\s+keep\s+playing|keep\s+going)\b)*[\s,|]*")
_SAID_AGAIN = rf"\b{_WAKE}[\s|,]+(?:please[\s|,]+)?(?:stops?|finish|restart|reset)\b"
_ABOUT_THE_LOOP = re.compile(rf"{_LOOP_AIM}{_ENDING}(?:{_SAID_AGAIN}{_LOOP_AIM}{_ENDING})*")


def _stop_said_loosely(t: str) -> bool:
    """"stop" with words before it, or the wake word after it.  Only "stop" gets this:
    "plume, in the end" and "plume, louder again" are not commands."""
    def an_order(start: int) -> bool:
        return not _NOT_AN_ORDER.search(_POLITE.sub(" ", re.split(r"[,|]", t[:start])[-1]))

    for m in re.finditer(r"\bstops?\b", t):
        near = re.findall(r"[a-z']+", t[:m.start()])[-_WORDS_BEFORE_STOP:]
        if (any(re.fullmatch(_REAL_WAKE, w) for w in near) and an_order(m.start())
                and not _ABOUT_THE_LOOP.fullmatch(t[m.end():])):
            return True
    return any(an_order(m.start()) for m in _STOP_THEN_WAKE.finditer(t))


def command_in(text: str) -> str | None:
    """The command following the wake word in ``text``, or None.

    "stop" and "restart" leave the take alone only if the verb's direct object is the
    loop and everything after it is a harmless ending (see ``_ENDING``) or the same
    thing said again.  Any other words mean the command stands, because a stop that
    fails is worse than one that ends a take by accident.  The other commands stand
    back from any sentence naming a loop, and a stop beats them wherever it falls.
    """
    said = str(text or "").lower().replace("’", "'")
    # | ends a sentence.  Commas stay: a loop after one is not the verb's object.
    t = re.sub(r"[^a-z'|, ]+", " ", re.sub(r"[.!?]+", "|", said))
    for name, words in _COMMANDS:
        # The model puts a full stop anywhere, so one may sit between plume and the word.
        for m in re.finditer(rf"\b{_WAKE}[\s|,]+(?:please[\s|,]+)?({words})\b", t):
            rest = t[m.end():]
            if m[1] not in _TAKES_NO_OBJECT and _ABOUT_THE_LOOP.fullmatch(rest):
                continue
            if name in ("stop", "restart") or not re.search(_LOOP_WORD, rest.split("|")[0]):
                return name
        if name == "stop" and _stop_said_loosely(t):
            return name
    return None


def next_take_dir(base) -> Path:
    """The first unused takesN directory under ``base``."""
    base = Path(base)
    used = [int(m.group(1)) for p in base.glob("takes*")
            if (m := re.fullmatch(r"takes(\d+)", p.name))]
    return base / f"takes{max(used, default=0) + 1}"


def _apple_silicon(system: str | None = None, machine: str | None = None) -> bool:
    system = sys.platform if system is None else system
    machine = platform.machine() if machine is None else machine
    return system == "darwin" and machine.lower() in ("arm64", "aarch64")


def backends_for(system: str | None = None, machine: str | None = None) -> tuple[str, ...]:
    """The speech backends that can run here, best first.

    mlx only runs on Apple Silicon, where it is the faster of the two.
    """
    return ("mlx", "faster") if _apple_silicon(system, machine) else ("faster",)


def choose_backend(system: str | None = None, machine: str | None = None,
                   find_spec=None) -> str:
    """The first installed backend that can run here.  Raises if there is none.

    Fails before any band starts, not after.
    """
    find_spec = importlib.util.find_spec if find_spec is None else find_spec
    wanted = backends_for(system, machine)
    for name in wanted:
        if find_spec(_BACKEND_MODULES[name]) is not None:
            return name
    raise RuntimeError(f"plume voice needs {_BACKEND_PACKAGES[wanted[0]]}: "
                       "pip install -e '.[voice]'")


def _require_whisper() -> str:
    """Fail before any band starts, not after.  Returns the backend to use."""
    return choose_backend()


def _whisper_mlx(audio: np.ndarray) -> str:
    import mlx_whisper  # noqa: PLC0415 — optional, heavy, and only needed here
    return str(mlx_whisper.transcribe(
        audio, path_or_hf_repo=MODELS["mlx"], language="en",
        condition_on_previous_text=False, no_speech_threshold=0.5)["text"])


_FASTER_MODEL = None
_FASTER_ON_CPU = False


def _whisper_faster(audio: np.ndarray) -> str:
    """Transcribe with faster-whisper, on the GPU if it works and the CPU if not.

    ``device="auto"`` picks CUDA on any machine with an NVIDIA driver, even one
    missing the CUDA libraries, and then loading or transcribing fails ("cublas64_12.dll
    not found").  So a failure off the CPU is retried once on the CPU, and the CPU is
    kept from then on.  Segments are lazy, so the failure can come while reading them.
    """
    global _FASTER_MODEL, _FASTER_ON_CPU
    from faster_whisper import WhisperModel  # noqa: PLC0415 — optional and heavy

    def run() -> str:
        global _FASTER_MODEL
        if _FASTER_MODEL is None:
            _FASTER_MODEL = (WhisperModel(MODELS["faster"], device="cpu", compute_type="int8")
                             if _FASTER_ON_CPU else
                             WhisperModel(MODELS["faster"], device="auto",
                                          compute_type="default"))
        segments, _info = _FASTER_MODEL.transcribe(
            audio, language="en", condition_on_previous_text=False,
            no_speech_threshold=0.5, beam_size=1)
        return "".join(seg.text for seg in segments)

    try:
        return run()
    except (RuntimeError, OSError):
        if _FASTER_ON_CPU:
            raise
        _FASTER_MODEL, _FASTER_ON_CPU = None, True
        return run()


_TRANSCRIBERS = {"mlx": _whisper_mlx, "faster": _whisper_faster}


_CHOSEN: str | None = None


def _whisper(audio: np.ndarray) -> str:
    """Transcribe with whichever backend runs here, chosen once."""
    global _CHOSEN
    if _CHOSEN is None:
        _CHOSEN = choose_backend()
    return _TRANSCRIBERS[_CHOSEN](audio)


def _to_stt(y: np.ndarray, sr: int) -> np.ndarray:
    """To the model's rate, low-passed: anything above its 8 kHz would otherwise fold
    down over the voice, and the band's hats and crash live up there."""
    g = math.gcd(int(sr), _STT_SR)
    return resample_poly(y, _STT_SR // g, int(sr) // g).astype(np.float32)


def _rms_db(audio: np.ndarray) -> float:
    return 20.0 * np.log10(float(np.sqrt(np.mean(audio ** 2))) + 1e-9)


class VoiceLoop:
    """Poll a take's recording; act on a heard command.  The transcriber and the
    actions are passed in, so all of this is testable without a model or a band."""

    def __init__(self, record_dir, *, act, transcribe=_whisper, clock=time.monotonic):
        self.record_dir = Path(record_dir)
        self._act = act
        self._transcribe = transcribe
        self._clock = clock
        self._read_to = 0
        self._missed_s = 0.0
        self._source = "human.wav"
        self._last_fired: dict[str, float] = {}

    def retarget(self, record_dir) -> None:
        self.record_dir = Path(record_dir)
        self._read_to = 0
        self._missed_s = 0.0

    def tail(self) -> np.ndarray | None:
        """The newest window, once a hop's worth of new audio has arrived."""
        try:
            self._source = ("voice.wav" if (self.record_dir / "voice.wav").exists()
                            else "human.wav")
            with sf.SoundFile(str(self.record_dir / self._source)) as f:
                sr, frames = f.samplerate, f.frames
                if frames - self._read_to < int(HOP_S * sr):
                    return None
                start = max(0, frames - int(WINDOW_S * sr))
                # Audio that slid past while the model was busy: nobody ever heard it.
                self._missed_s = (max(0, start - self._read_to) / sr
                                  if self._read_to else 0.0)
                f.seek(start)
                y = f.read(frames - start, dtype="float32", always_2d=True).mean(axis=1)
        except (OSError, RuntimeError):
            return None
        self._read_to = frames
        return _to_stt(y, sr)

    def _log(self, **row) -> None:
        """One line per window.  Best effort: a log must not end a take."""
        try:
            with (self.record_dir / "voice.ndjson").open("a") as f:
                f.write(json.dumps({"source": self._source, "missed_s": round(self._missed_s, 2),
                                    **row}) + "\n")
        except OSError:
            pass

    def step(self) -> str | None:
        audio = self.tail()
        if audio is None or audio.size == 0:
            return None
        level = round(_rms_db(audio), 1)
        if level < MIN_RMS_DB:
            self._log(rms_db=level, text=None, command=None, skipped="quiet")
            return None
        began = time.monotonic()
        text = str(self._transcribe(audio))
        name = command_in(text)
        cooling = (name is not None
                   and self._clock() - self._last_fired.get(name, -1e9) < COOLDOWN_S)
        self._log(rms_db=level, text=text, command=name, cooling=cooling,
                  took_s=round(time.monotonic() - began, 2))
        if name is None or cooling:
            return None
        self._last_fired[name] = self._clock()
        self._act(name)
        return name


def replay(path, *, transcribe=None, write=print) -> None:
    """What the model hears in a recording that already exists, window by window.  No
    device and no band: this is how to ask what happened in a take that is over."""
    if transcribe is None:
        _require_whisper()
        transcribe = _whisper
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    size, hop = int(WINDOW_S * sr), int(HOP_S * sr)
    for start in range(0, max(1, y.size - size + 1), hop):
        audio = _to_stt(y[start:start + size], sr)
        level = round(_rms_db(audio), 1)
        text = None if level < MIN_RMS_DB else str(transcribe(audio))
        write(json.dumps({"t_s": round(start / sr, 2), "rms_db": level, "text": text,
                          "command": command_in(text)}))


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
        env.update(oscompat.own_group_env())      # it leads its group: see detached_kwargs
        with self.log.open("a") as log:
            self._proc = self._popen(
                # After "--" a prompt is only ever a prompt.  Before the flags, one
                # spelled "--max-minutes=0" would turn off the session cap.
                [sys.executable, "-m", "plume", "lead", "--tap-start",
                 "--input", "acoustic", "--record", f"{self.dir}/",
                 "--seed", str(int(time.time()) % 100000), *self._extra, "--", self._prompt],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log, env=env,
                # Its own process group: a terminal's Ctrl-C must reach only us, or the
                # band gets two interrupts, and a second one cuts its ending short.
                **oscompat.detached_kwargs())
        self._write(json.dumps({"take": str(self.dir)}))
        return self.dir

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        if not self.alive():
            return
        try:
            oscompat.interrupt(self._proc)        # its graceful ending: finish the bar
        except OSError:
            pass                                  # no console to send it through: wait, then end it
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
        backend = _require_whisper()
        # The first call downloads and loads the model, which can take minutes.  Inside
        # the take, that is minutes of band with nobody listening.
        write(json.dumps({"loading": MODELS[backend]}))
        _whisper(np.zeros(_STT_SR, dtype=np.float32))
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
            sent = subprocess.run([sys.executable, "-m", "plume", "cue", _CUES[name], "--source", "voice"],
                                  capture_output=True, check=False)
            if sent.returncode != 0:
                write(json.dumps({"cue_failed": name}))

    # SIGHUP: the terminal closed; the band, in its own session, won't get one.
    # SIGBREAK is Windows' Ctrl-Break and closing console.  Each where it exists.
    ends = tuple(getattr(signal, name) for name in oscompat.closing_signals())

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
