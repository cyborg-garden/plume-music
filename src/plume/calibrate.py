"""Acoustic round-trip calibration and placement compensation (v2.5).

WHY THIS EXISTS.  The live transports place bars at the tracker's predicted
downbeat in STREAM time (``at = next_downbeat_t_s * SR``).  Stream time is the
audio callback's sample counter: an input sample at index N was captured at the
mic INPUT-LATENCY earlier, and an output sample at index N leaves the speaker
OUTPUT-LATENCY later.  So a bar placed exactly on the human's stream-time
downbeat sounds, in the room, input + output latency AFTER the human's beat —
measured on the development MacBook: 112 ms, over a fifth of a beat at 120 bpm,
and dead stable (four clicks within 0.2 ms of each other).  The self-masker
already compensates this round trip for HEARING (so the tracker cannot lock to
the machine's own output); nothing compensated it for PLAYING.

The compensation is one subtraction at the placement sites, and the number it
subtracts comes from here, resolved in this order:

1. an explicit ``--latency-comp MS`` (0 disables — the v2.4 behavior);
2. a stored acoustic measurement for this device pair (``plume calibrate``);
3. the open stream's reported input+output latency — an estimate PortAudio
   makes without listening to the room (it read 141 ms where the room said
   112), kept because being ~30 ms wrong beats being 112 ms wrong;
4. nothing (0) — offline transports, or a stream that reports no latency.

``plume calibrate`` plays a short click train through the speakers and
matched-filters the microphone: the lag from "written to the output stream" to
"back in the input stream" IS the round trip the placement must subtract, by
construction — same stream, same clocks, no model of the hardware needed.  The
result is stored per (input device, output device, sample rate), because the
number is a fact about a hardware pair, not about the machine.

Measurement is pure (`click_train`, `measure_offsets`, `summarize`) so the
whole path is testable without a device; only :func:`run` opens a stream, and
its caller gates it behind the same device + consent checks as every other
sound-making verb.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

#: Where measurements persist.  A calibration is per hardware pair and should
#: outlive the shell and the session, so it lives in the home directory, not
#: the session tmpdir; the env hatch exists for tests and odd deployments.
CALIBRATION_ENV = "PLUME_CALIBRATION_FILE"
DEFAULT_CALIBRATION_PATH = "~/.plume/calibration.json"

CLICK_S = 0.005            # one click: 5 ms of windowed noise (broadband)
CLICK_GAIN = 0.8
CLICK_SPACING_S = 0.5      # clicks 500 ms apart
CLICK_COUNT = 8
LEAD_IN_S = 0.5            # stream settles before the first click
CLICK_SEED = 3             # clicks are deterministic; only the room varies
#: One click's search window.  Shorter than the spacing so click k's window
#: can never contain click k+1; longer than any plausible domestic round trip.
SEARCH_S = 0.45
#: A window whose peak matched-filter response is below this fraction of the
#: click's self-correlation never heard the click (muted speakers, mic across
#: the room) and is skipped rather than reported as a garbage lag.
DETECT_FRAC = 0.02
#: Placement compensation ceiling for a number somebody MEASURED or asked for.
#: A laptop mic with AirPods output measured 317 ms by tapping along (2026-11-02),
#: so a quarter second silently left every bar 67 ms late.  0.4 s still stops a
#: broken measurement moving a bar inside jam's revision window (one beat before
#: the downbeat) at anything up to 150 bpm; a 317 ms pair is only safe to about
#: 190 bpm in jam, where a bar can move once it has started.
MAX_COMP_S = 0.4
#: Ceiling for the stream's own guess, which nobody measured.  That same pair
#: reported 548 ms against the real 317, so the guess keeps the tighter bound.
MAX_REPORTED_COMP_S = 0.25
#: Per-click spread above this means the clicks disagree about the round trip
#: (echoes, a resampling stack) and the median deserves a warning, not trust.
SPREAD_WARN_MS = 5.0


# --------------------------------------------------------------- pure parts

def click_train(sr: int, *, clicks: int = CLICK_COUNT,
                spacing_s: float = CLICK_SPACING_S,
                lead_in_s: float = LEAD_IN_S) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(stereo out, the mono click, mark indices) — deterministic."""
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")
    n_click = max(1, int(round(CLICK_S * sr)))
    rng = np.random.default_rng(CLICK_SEED)
    click = (rng.standard_normal(n_click) * np.hanning(n_click)).astype(np.float32)
    click /= max(1e-12, float(np.max(np.abs(click))))
    click *= CLICK_GAIN
    gap = int(round(spacing_s * sr))
    total = int(round(lead_in_s * sr)) + gap * clicks + n_click
    out = np.zeros((total, 2), dtype=np.float32)
    marks: list[int] = []
    for k in range(clicks):
        i = int(round(lead_in_s * sr)) + k * gap
        out[i:i + n_click, 0] += click
        out[i:i + n_click, 1] += click
        marks.append(i)
    return out, click, marks


def measure_offsets(rec: np.ndarray, click: np.ndarray, marks: list[int],
                    sr: int, *, search_s: float = SEARCH_S) -> list[int]:
    """Per-click round trip in samples, via matched filter after each mark.

    Clicks the microphone never heard are skipped, not reported: a lag from a
    silent window is the argmax of noise, and one such garbage value would
    poison the median more than a missing one.
    """
    rec = np.asarray(rec, dtype=np.float32)
    if rec.ndim > 1:
        rec = rec.mean(axis=1)
    win = int(round(search_s * sr))
    floor = DETECT_FRAC * float(np.dot(click, click))
    lags: list[int] = []
    for i in marks:
        seg = rec[i:i + win]
        if seg.shape[0] < click.shape[0]:
            continue
        cc = np.correlate(seg, click, "valid")
        peak = int(np.argmax(np.abs(cc)))
        if abs(float(cc[peak])) < floor:
            continue
        lags.append(peak)
    return lags


def summarize(lags: list[int], sr: int, *, clicks: int = CLICK_COUNT) -> dict:
    """Round-trip summary: median + spread in ms, and how many clicks made it."""
    if not lags:
        return {"ok": False, "heard": 0, "clicks": clicks,
                "roundtrip_ms": None, "spread_ms": None}
    ms = sorted(float(l) / sr * 1000.0 for l in lags)
    median = ms[len(ms) // 2] if len(ms) % 2 else 0.5 * (ms[len(ms) // 2 - 1]
                                                         + ms[len(ms) // 2])
    return {"ok": True, "heard": len(ms), "clicks": clicks,
            "roundtrip_ms": round(median, 1),
            "spread_ms": round(ms[-1] - ms[0], 1)}


def placement_comp_samples(explicit_s: float | None, stored_ms: float | None,
                           reported_s: float | None, sr: int) -> int:
    """Samples to subtract at the placement sites (the resolution order in the
    module docstring).  Never negative, clamped to :data:`MAX_COMP_S`."""
    if explicit_s is not None:
        s = float(explicit_s)
    elif stored_ms is not None:
        s = float(stored_ms) / 1000.0
    elif reported_s is not None and math.isfinite(reported_s):
        s = min(float(reported_s), MAX_REPORTED_COMP_S)
    else:
        s = 0.0
    # NaN would sail through min/max (Python keeps the first argument) and
    # blow up at int(); any non-finite request means "no compensation".
    if not math.isfinite(s):
        s = 0.0
    s = min(max(s, 0.0), MAX_COMP_S)
    return int(round(s * sr))


# ------------------------------------------------------------------ storage

def calibration_path() -> Path:
    return Path(os.environ.get(CALIBRATION_ENV, "")
                or DEFAULT_CALIBRATION_PATH).expanduser()


def _key(input_name: str, output_name: str, sr: int) -> str:
    return f"{input_name}␟{output_name}␟{int(sr)}"


def load_roundtrip_ms(input_name: str, output_name: str, sr: int) -> float | None:
    """The stored measurement for this device pair, or None."""
    path = calibration_path()
    try:
        doc = json.loads(path.read_text())
        entry = doc.get("entries", {}).get(_key(input_name, output_name, sr))
        ms = None if entry is None else float(entry["roundtrip_ms"])
        # json accepts the nonstandard NaN literal; a NaN measurement is a
        # broken measurement, and None routes to the reported fallback.
        return ms if ms is not None and math.isfinite(ms) else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        # AttributeError: valid JSON of the wrong SHAPE ([1,2], "x", null,
        # {"entries": [1]}) — as fail-soft as invalid JSON, per the module
        # rule that a bad file must never take a live session down.
        return None


def save_roundtrip(input_name: str, output_name: str, sr: int,
                   summary: dict) -> Path:
    """Persist one measurement; other device pairs' entries survive."""
    path = calibration_path()
    try:
        doc = json.loads(path.read_text())
        if not isinstance(doc, dict) or not isinstance(doc.get("entries"), dict):
            raise ValueError("no entries")
    except (OSError, ValueError):
        doc = {"entries": {}}
    doc["entries"][_key(input_name, output_name, sr)] = {
        "input": input_name, "output": output_name, "sr": int(sr),
        "roundtrip_ms": summary["roundtrip_ms"],
        "spread_ms": summary["spread_ms"],
        "heard": summary["heard"], "clicks": summary["clicks"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def device_names(sd, dev) -> tuple[str, str]:
    """(input name, output name) for a Stream ``device=`` value, best effort.

    ``dev`` is what ``_resolve_device`` produced: None (system defaults) or an
    (input, output) index pair.  Name lookup failing must never take the
    feature down — the names are only a storage key — so any trouble collapses
    to "default".
    """
    din, dout = (dev if isinstance(dev, (tuple, list)) else (None, None))
    names = []
    for d, kind in ((din, "input"), (dout, "output")):
        try:
            q = sd.query_devices(d, kind) if d is not None else \
                sd.query_devices(kind=kind)
            names.append(str(q.get("name", "default")) or "default")
        except Exception:  # noqa: BLE001 — key only, never fatal
            names.append("default")
    return names[0], names[1]


# ------------------------------------------------------------------- device

def run(device: str | None = None, *, sr: int = 48000) -> dict:
    """Play the click train, record the mic, measure.  Blocking, ~5 s.

    The caller is responsible for the device + audio-consent gate
    (``realtime.check_device``), exactly as for jam/lead/play.
    """
    from . import realtime  # noqa: PLC0415 — lazy: this module stays headless
    sd = realtime._import_sounddevice()
    dev = realtime._resolve_device(sd, device)
    out, click, marks = click_train(sr)
    rec = np.zeros(out.shape[0], dtype=np.float32)
    pos = [0]
    done = __import__("threading").Event()

    def callback(indata, outdata, frames, time_info, status):  # noqa: ARG001
        n = pos[0]
        m = min(frames, out.shape[0] - n)
        outdata[:m] = out[n:n + m]
        if m < frames:
            outdata[m:] = 0.0
        rec[n:n + m] = indata[:m, 0]
        pos[0] += frames
        if pos[0] >= out.shape[0]:
            done.set()
            raise sd.CallbackStop()

    stream = sd.Stream(samplerate=sr, blocksize=256, device=dev,
                       channels=(1, 2), dtype="float32", latency="low",
                       callback=callback)
    reported = None
    with stream:
        lat = getattr(stream, "latency", None)
        if isinstance(lat, (tuple, list)) and len(lat) == 2:
            try:
                reported = float(lat[0]) + float(lat[1])
            except (TypeError, ValueError):
                reported = None
        done.wait(timeout=out.shape[0] / sr + 5.0)

    summary = summarize(measure_offsets(rec, click, marks, sr), sr)
    in_name, out_name = device_names(sd, dev)
    summary.update({"input": in_name, "output": out_name, "sr": sr,
                    "reported_ms": None if reported is None
                    else round(reported * 1000.0, 1)})
    return summary
