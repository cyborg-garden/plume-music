"""Recorded one-shots behind the ``kit_*`` voices.

Quiet and hard hits on a real drum differ in tone as well as level, which is a
large part of why a synthesized kit sounds stiff.  So velocity picks a recorded
layer here, and ``voices._finish`` still scales its level.  Files load once, on
first use: a planning thread cannot afford to read disk on every hit.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

_SAMPLES = Path(__file__).resolve().parent / "samples"
_KIT = _SAMPLES / "kit"
_TUNED = _SAMPLES / "tuned"


@lru_cache(maxsize=1)
def _manifest() -> dict:
    return json.loads((_KIT / "manifest.json").read_text())


@lru_cache(maxsize=1)
def _tuned_manifest() -> dict:
    return json.loads((_TUNED / "manifest.json").read_text())


@lru_cache(maxsize=None)
def _load(name: str, sr: int, folder: Path = _KIT) -> np.ndarray:
    y, file_sr = sf.read(str(folder / name), dtype="float32")
    if file_sr != sr:
        n = int(round(y.shape[0] * sr / file_sr))
        y = np.interp(np.arange(n) * file_sr / sr, np.arange(y.shape[0]), y)
    y = np.asarray(y, dtype=np.float32)
    y.setflags(write=False)
    return y


def voice_names() -> list[str]:
    return sorted(_manifest())


def hit(voice: str, velocity: float, rng: np.random.Generator, sr: int) -> np.ndarray:
    """A recorded hit for ``velocity``, drawn from ``rng`` so takes stay
    reproducible.  A neighbouring layer is sometimes used, so a repeated hit at
    one velocity does not repeat the same recording."""
    layers = _manifest()[voice]
    v = float(np.clip(velocity, 0.0, 1.0))
    i = int(round(v * (len(layers) - 1)))
    if len(layers) > 2:
        i = int(np.clip(i + rng.integers(-1, 2), 0, len(layers) - 1))
    names = layers[i]
    return _load(names[int(rng.integers(len(names)))], sr).copy()


def note(voice: str, midi: float, velocity: float, dur_s: float, release_s: float,
         rng: np.random.Generator, sr: int) -> np.ndarray:
    """A recorded note at ``midi``, repitched from the nearest sample.

    A note rings for ``dur_s``, then fades over ``release_s``; a sample that
    decays sooner just ends.
    """
    entries = _tuned_manifest()[voice]
    layers = sorted({e["layer"] for e in entries})
    layer = layers[int(round(float(np.clip(velocity, 0.0, 1.0)) * (len(layers) - 1)))]
    pool = [e for e in entries if e["layer"] == layer]
    nearest = min(abs(e["midi"] - midi) for e in pool)
    picks = [e for e in pool if abs(e["midi"] - midi) == nearest]
    e = picks[int(rng.integers(len(picks)))]
    y = _load(e["file"], sr, _TUNED)
    ratio = 2.0 ** ((midi - e["midi"]) / 12.0)
    n = min(int(y.shape[0] / ratio), int((dur_s + release_s) * sr))
    out = np.interp(np.arange(n) * ratio, np.arange(y.shape[0]), y)
    fade_at = int(dur_s * sr)
    if fade_at < n:
        out[fade_at:] *= np.linspace(1.0, 0.0, n - fade_at) ** 2
    return out.astype(np.float32)
