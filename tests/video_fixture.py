"""Made-up playing, talking and tiny videos for the video tests.

Everything is synthetic and seeded.  The videos are a few seconds of a 32-pixel
black square, written by ffmpeg itself, so no file is fetched and none is kept.
"""
from __future__ import annotations

import subprocess

import numpy as np
import pytest
from scipy import signal

from plume import takevideo

FFMPEG = takevideo.find_ffmpeg()
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg (the imageio-ffmpeg package, or one on the PATH)")

_PITCHES = (110.0, 123.5, 146.8, 164.8, 196.0, 220.0, 246.9, 293.7, 329.6, 392.0, 440.0)


def playing(seconds: float, rate: int, seed: int = 1) -> np.ndarray:
    """Plucked notes at uneven times: something like a hand on strings."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * rate))
    at = 0.2
    while at < seconds - 0.5:
        length = int(min(1.2, seconds - at) * rate)
        t = np.arange(length) / rate
        pitch = rng.choice(_PITCHES) * rng.choice((1.0, 2.0))
        note = sum(np.sin(2 * np.pi * pitch * k * t) / k for k in (1, 2, 3, 4))
        note *= np.exp(-t / rng.uniform(0.15, 0.4)) * np.minimum(1.0, t / 0.003)
        begin = int(at * rate)
        out[begin:begin + length] += note * rng.uniform(0.1, 0.25)
        at += rng.uniform(0.12, 0.6)
    return out


def talking(seconds: float, rate: int, seed: int = 2, level: float = 0.15) -> np.ndarray:
    """Bursts of voice-shaped noise: syllables, with gaps."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    sos = signal.butter(2, [200.0, min(2500.0, rate * 0.45)], btype="bandpass", fs=rate, output="sos")
    noise = signal.sosfilt(sos, rng.normal(0.0, 1.0, n))
    gate = np.zeros(n)
    at = 0.1
    while at < seconds:
        length = rng.uniform(0.08, 0.3)
        gate[int(at * rate):int((at + length) * rate)] = 1.0
        at += length + rng.uniform(0.05, 0.4)
    smooth = np.ones(int(0.01 * rate)) / int(0.01 * rate)
    return noise * np.convolve(gate, smooth, mode="same") * level


def heard_by_camera(sound: np.ndarray, rate: int, level: float = 0.4) -> np.ndarray:
    """The same playing as a camera across the room has it: duller, quieter, with the room in it."""
    sos = signal.butter(1, 1800.0, btype="lowpass", fs=rate, output="sos")
    dull = signal.sosfilt(sos, sound)
    echo = np.zeros_like(dull)
    late = int(0.023 * rate)
    echo[late:] = dull[:-late] * 0.3
    return (dull + echo) * level


def placed(sound: np.ndarray, start: int, length: int) -> np.ndarray:
    """``sound`` on a clock ``length`` samples long, its first sample at ``start`` (which may be negative)."""
    out = np.zeros(length)
    lo, hi = max(0, start), min(length, start + len(sound))
    if hi > lo:
        out[lo:hi] = sound[lo - start:hi - start]
    return out


def make_video(path, *, seconds: float, sound: np.ndarray | None = None, rate: int = 48000, tags=()):
    """A tiny video.  ``sound`` is its mono sound track, or None for a video with none."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
           "-i", f"color=c=black:s=32x32:r=5:d={seconds}"]
    data = None
    if sound is not None:
        cmd += ["-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0"]
        data = (np.clip(sound, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    # mpeg4 and aac: the two encoders every ffmpeg has.
    cmd += ["-c:v", "mpeg4", "-q:v", "20"]
    if sound is not None:
        cmd += ["-c:a", "aac", "-b:a", "96k"]
    for tag in tags:
        cmd += ["-metadata", tag]
    cmd += ["-t", str(seconds), "-f", "mp4", str(path)]
    subprocess.run(cmd, input=data, check=True, timeout=120, stdin=None if data is not None else subprocess.DEVNULL)
    return path


def picture_md5(path) -> str:
    """The picture's packets, hashed as they are: the same before and after a copy."""
    done = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:v:0",
                           "-c", "copy", "-f", "md5", "-"], capture_output=True, check=True, timeout=120,
                          stdin=subprocess.DEVNULL)
    return done.stdout.decode().strip()


def float_sound(path, rate: int = 48000) -> np.ndarray:
    """A video's sound as floats that are not clipped on the way out, so a peak over full scale shows.

    Every channel, one after another, as decoded: mixing two down to one in floats adds
    them, which would read as a peak the file does not have.
    """
    done = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:a:0",
                           "-ar", str(rate), "-f", "f32le", "-"], capture_output=True, check=True,
                          timeout=120, stdin=subprocess.DEVNULL)
    return np.frombuffer(done.stdout, dtype="<f4")
