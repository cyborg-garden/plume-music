"""``plume watch``: a live feed of a recording session, for whoever sits outside
time (SPEC-LIVE-API section 1).

It reads what a session already streams to disk, never the audio device, so it
cannot disturb the band and can be attached or killed at any moment.  One NDJSON
line per group of bars: what the band is doing, and what the player is doing as
far as a mic can tell.

Hearing the player through a speaker in the same room is the hard part: the mic
hears the band too.  The part of the mic that matches what the band played, at the
best-fitting delay, is removed, but in a real room that takes off only a few dB.  So
over a room mic with the band in, the ``you_*`` fields mostly describe the band.  They
mean what they say on a direct input, or over drums alone.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .listen.features import FeatureExtractor

_NOTES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_FLOOR_DB = -90.0
#: How far the player must move from their own recent level to count as a change.
_LEVEL_CHANGE_DB = 5.0
_BUSY_CHANGE = 0.6            # onsets per beat
_STOPPED_DROP_DB = 9.0
_TRAIL = 0.6                  # weight of history in the player's running level
_MAX_ROOM_DELAY_S = 0.7       # output latency + air + input latency, generously


def _db(power: float) -> float:
    return _FLOOR_DB if power <= 1e-9 else max(_FLOOR_DB, 10.0 * math.log10(power))


def _bleed_power(mic: np.ndarray, band: np.ndarray, sr: int) -> float:
    """Power of the part of ``mic`` that is ``band`` arriving late."""
    n = min(mic.size, band.size)
    energy = float(np.sum(band[:n] ** 2))
    if n == 0 or energy <= 1e-9:
        return 0.0
    size = 2 * n
    xcorr = np.fft.irfft(np.fft.rfft(mic[:n], size) * np.conj(np.fft.rfft(band[:n], size)),
                         size)[:int(_MAX_ROOM_DELAY_S * sr)]
    gain = float(np.max(np.abs(xcorr))) / energy
    return gain * gain * energy / n


class Listener:
    def __init__(self, record_dir, *, every: int = 4):
        self._dir = Path(record_dir)
        self._every = max(1, int(every))
        self._lines_read = 0
        self._rows: list[dict] = []
        self._extractor: FeatureExtractor | None = None
        self._fed = 0
        self._level: float | None = None
        self._busy: float | None = None
        self._stopped = False
        self._section: str | None = None
        self._stage: str | None = None

    def _new_rows(self) -> None:
        path = self._dir / "decisions.ndjson"
        if not path.exists():
            return
        lines = path.read_text().splitlines()
        # A line still being written has no newline yet: json refuses it, and it
        # is read again on the next poll.
        for line in lines[self._lines_read:]:
            try:
                row = json.loads(line)
            except ValueError:
                break
            self._lines_read += 1
            if (isinstance(row, dict) and isinstance(row.get("bar"), int)
                    and "t_s" in row and "bpm" in row):
                self._rows.append(row)

    @staticmethod
    def _read(path: Path, start: int, stop: int) -> np.ndarray | None:
        try:
            with sf.SoundFile(str(path)) as f:
                if f.frames < stop:
                    return None
                f.seek(start)
                y = f.read(stop - start, dtype="float32", always_2d=True)
        except (OSError, RuntimeError):
            return None
        return y.mean(axis=1) if y.shape[0] == stop - start else None

    def poll(self) -> list[dict]:
        """Every group of bars that has finished since the last poll."""
        self._new_rows()
        out = []
        while len(self._rows) >= self._every:
            doc = self._group(self._rows[:self._every])
            if doc is None:
                break
            del self._rows[:self._every]
            out.append(doc)
        return out

    def _group(self, rows: list[dict]) -> dict | None:
        try:
            with sf.SoundFile(str(self._dir / "human.wav")) as f:
                sr = f.samplerate
        except (OSError, RuntimeError):      # not written yet, or its header isn't
            return None
        bar_s = 240.0 / float(rows[-1]["bpm"])
        start = int(float(rows[0]["t_s"]) * sr)
        stop = int((float(rows[-1]["t_s"]) + bar_s) * sr)
        human = self._read(self._dir / "human.wav", start, stop)
        band = self._read(self._dir / "machine.wav", start, stop)
        if human is None or band is None:
            return None
        if self._extractor is None:
            self._extractor = FeatureExtractor(sr=sr)
        frames = self._extractor.process(human)
        beats = len(rows) * 4.0
        onsets = sum(1 for f in frames if f.onset)
        loud = [f.chroma for f in frames if f.rms_db > -55.0 and np.sum(f.chroma) > 0]
        notes: list[str] = []
        if loud:
            chroma = np.mean(loud, axis=0)
            notes = [_NOTES[i] for i in np.argsort(chroma)[::-1][:3] if chroma[i] > 0.08]
        heard, played = float(np.mean(human ** 2)), float(np.mean(band ** 2))
        heard_db, band_db = _db(heard), _db(played)
        you_db = _db(max(heard - _bleed_power(human, band, sr), 0.0))
        busy = onsets / beats
        last = rows[-1]
        changes = self._changes(last, you_db, busy)
        return {
            "bar": int(last["bar"]), "t_s": round(float(last["t_s"]), 2),
            "bpm": float(last["bpm"]), "section": last.get("section"),
            "stage": last.get("entry"), "key": last.get("key"),
            "chord": last.get("chord"), "you_db": round(you_db, 1),
            "band_db": round(band_db, 1), "mic_db": round(heard_db, 1),
            "you_busy": round(busy, 2), "you_notes": notes, "changes": changes,
        }

    def _changes(self, row: dict, you_db: float, busy: float) -> list[str]:
        out = []
        section, stage = row.get("section"), row.get("entry")
        if section != self._section and section is not None:
            out.append(f"section:{section}")
        if stage != self._stage and stage is not None:
            out.append(f"stage:{stage}")
        self._section, self._stage = section, stage
        if self._level is None:
            self._level, self._busy = you_db, busy
            return out
        drop = self._level - you_db
        if not self._stopped and drop >= _STOPPED_DROP_DB:
            self._stopped = True
            out.append("you:stopped")
        elif self._stopped and drop < _LEVEL_CHANGE_DB:
            self._stopped = False
            out.append("you:back")
        elif not self._stopped:
            if you_db - self._level >= _LEVEL_CHANGE_DB:
                out.append("you:louder")
            elif drop >= _LEVEL_CHANGE_DB:
                out.append("you:quieter")
            if busy - self._busy >= _BUSY_CHANGE:
                out.append("you:busier")
            elif self._busy - busy >= _BUSY_CHANGE:
                out.append("you:sparser")
        if not self._stopped:      # a rest must not drag down what "playing" means
            self._level = _TRAIL * self._level + (1.0 - _TRAIL) * you_db
            self._busy = _TRAIL * self._busy + (1.0 - _TRAIL) * busy
        return out


def follow(record_dir, *, every: int = 4, changes_only: bool = False,
           poll_s: float = 0.5, write=print) -> None:
    """Stream until the session has written its ending and the feed has caught up."""
    listener = Listener(record_dir, every=every)
    ending = Path(record_dir) / "ending.json"
    while True:
        done = ending.exists()
        for doc in listener.poll():
            if doc["changes"] or not changes_only:
                write(json.dumps(doc))
        if done:
            write(json.dumps({"ended": True}))
            return
        time.sleep(poll_s)
