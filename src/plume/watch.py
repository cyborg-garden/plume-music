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

So every line says which of them to believe: ``you_trusted`` names the fields that
describe the player, ``you_why`` says why, and ``input`` says what was listening.  Without
them a reader would report notes from a silent player, reading fields that are the
band's all along.  ``you:*`` entries in ``changes`` come from the same fields and
deserve the same doubt.

A session played over MIDI has none of that doubt.  Its ``midi.ndjson`` says which
notes were played and when, so the ``you_*`` fields come from there, and
``you_source`` says which of the two a line is reading.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .listen.features import ONSET_MIN_SEP_S, FeatureExtractor
from .midi import DRUM_CHANNEL, velocity_db
from .mind import MindParams

_NOTES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_FLOOR_DB = -90.0
#: How far the player must move from their own recent level to count as a change.
_LEVEL_CHANGE_DB = 5.0
_BUSY_CHANGE = 0.6            # onsets per beat
_STOPPED_DROP_DB = 9.0
_TRAIL = 0.6                  # weight of history in the player's running level
_MAX_ROOM_DELAY_S = 0.7       # output latency + air + input latency, generously
#: Below this the band is not sounding, so nothing in the mic can be the band's.
_BAND_AUDIBLE_DB = -60.0
#: Share of a declared direct input's power that matches the band before it is no
#: longer believed to be direct.  Uncalibrated: a guess until a real rig measures it.
_DIRECT_BLEED_SHARE = 0.25
_YOU_FIELDS = ("you_db", "you_busy", "you_notes")
#: A frame quieter than the level a session calls silence has no notes in it.
_NOTES_FLOOR_DB = MindParams.silence_rms_db
#: Chroma sums to 1, so hiss puts about 1/12 (0.083) on every note and up to 0.10 on
#: its luckiest.  A real note stands above that, even in a scale run (0.125).
_NOTE_SHARE = 0.12


def you_trust(source: str, stage: str | None, band_db: float,
              bleed_share: float) -> tuple[list[str], str]:
    """(which ``you_*`` fields describe the player, and why)."""
    if band_db < _BAND_AUDIBLE_DB:
        return list(_YOU_FIELDS), "the band is not sounding, so the input is all you"
    if source == "direct":
        if bleed_share >= _DIRECT_BLEED_SHARE:
            return [], ("declared direct, but the band is in this input: check for "
                        "loopback routing or a mic on this channel")
        return list(_YOU_FIELDS), "direct input: the band does not reach it"
    if stage == "drums":
        return ["you_notes"], ("room mic over drums alone: drums barely favour a note, "
                               "but level and onsets include them")
    return [], "room mic with the band in: these fields mostly describe the band"


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
        self._source: str | None = None
        self._midi_lines_read = 0
        self._midi: list[dict] | None = None     # None: not a MIDI session
        self._held: dict[tuple, tuple[float, float]] = {}

    def _input_source(self) -> str:
        """What the take says is listening; a take that does not say is a room mic,
        the assumption that cannot mislead."""
        if self._source is not None:
            return self._source
        try:
            doc = json.loads((self._dir / "session.json").read_text())
        except (OSError, ValueError):
            return "room"          # not written yet, or from before this existed
        source = doc.get("input", {}).get("source") if isinstance(doc, dict) else None
        self._source = "direct" if source == "direct" else "room"
        return self._source

    def _new_midi(self) -> None:
        path = self._dir / "midi.ndjson"
        if not path.exists():
            return
        if self._midi is None:
            self._midi = []
        for line in path.read_text().splitlines()[self._midi_lines_read:]:
            try:
                doc = json.loads(line)
            except ValueError:
                break
            self._midi_lines_read += 1
            if isinstance(doc, dict) and "t_s" in doc:
                self._midi.append(doc)

    def _midi_reading(self, start_s: float, stop_s: float, beats: float
                      ) -> tuple[float, float, list[str], list[str]]:
        """(you_db, you_busy, you_notes, midi changes) for this stretch of the take."""
        now = [d for d in self._midi if d["t_s"] < stop_s]
        self._midi = [d for d in self._midi if d["t_s"] >= stop_s]
        weight = np.zeros(12)
        levels: list[float] = []
        changes: list[str] = []
        carried: list[float] = []        # struck earlier, still heard in this stretch
        onsets, last_onset = 0, -math.inf

        def sounded(key, since, until):
            if key[0] != DRUM_CHANNEL:
                weight[key[1] % 12] += max(0.0, until - max(since, start_s))

        for doc in now:
            t, kind = float(doc["t_s"]), doc.get("type")
            key = (doc.get("channel", 0), doc.get("note"))
            if kind == "on":
                if key in self._held:
                    sounded(key, self._held[key][0], t)
                self._held[key] = (t, float(doc.get("velocity", 0)))
                early = t < start_s               # only what it was still sounding
                if not early:
                    levels.append(velocity_db(self._held[key][1]))
                if key[0] == DRUM_CHANNEL:
                    self._held.pop(key)
                if not early and t - last_onset >= ONSET_MIN_SEP_S:  # a chord is one onset
                    onsets += 1
                    last_onset = t
            elif kind == "off" and key in self._held:
                since, velocity = self._held.pop(key)
                sounded(key, since, t)
                if t > start_s:
                    carried.append(velocity_db(velocity))
            elif kind == "lost":
                changes.append("midi:lost")
            elif kind == "back":
                changes.append("midi:back")
        for key, (since, velocity) in self._held.items():
            sounded(key, since, stop_s)
            carried.append(velocity_db(velocity))
        if not levels:                           # a long chord is still playing
            levels = carried
        notes: list[str] = []
        total = float(weight.sum())
        if total > 0.0:
            share = weight / total
            notes = [_NOTES[i] for i in np.argsort(share)[::-1][:3] if share[i] > 0.08]
        you_db = float(np.mean(levels)) if levels else _FLOOR_DB
        return you_db, onsets / beats, notes, changes

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
        self._new_midi()
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
        loud = [f.chroma for f in frames if f.rms_db > _NOTES_FLOOR_DB and np.sum(f.chroma) > 0]
        notes: list[str] = []
        if loud:
            chroma = np.mean(loud, axis=0)
            notes = [_NOTES[i] for i in np.argsort(chroma)[::-1][:3] if chroma[i] > _NOTE_SHARE]
        heard, played = float(np.mean(human ** 2)), float(np.mean(band ** 2))
        heard_db, band_db = _db(heard), _db(played)
        bleed = _bleed_power(human, band, sr)
        source = self._input_source()
        last = rows[-1]
        trusted, why = you_trust(source, last.get("entry"), band_db,
                                 bleed / heard if heard > 1e-9 else 0.0)
        # A direct input has nothing to subtract, and over a real instrument the
        # correlation would only ever remove some of the player.
        you_db = heard_db if source == "direct" else _db(max(heard - bleed, 0.0))
        busy = onsets / beats
        you_source = "mic"
        midi_changes: list[str] = []
        if self._midi is not None:
            # Notes, not sound: nothing the band does can reach these.
            you_source = source = "midi"
            trusted, why = list(_YOU_FIELDS), "MIDI: these are the notes you played"
            you_db, busy, notes, midi_changes = self._midi_reading(
                start / sr, stop / sr, beats)
        changes = self._changes(last, you_db, busy) + midi_changes
        return {
            "bar": int(last["bar"]), "t_s": round(float(last["t_s"]), 2),
            "bpm": float(last["bpm"]), "section": last.get("section"),
            "stage": last.get("entry"), "key": last.get("key"),
            "chord": last.get("chord"), "you_db": round(you_db, 1),
            "band_db": round(band_db, 1), "mic_db": round(heard_db, 1),
            "you_busy": round(busy, 2), "you_notes": notes, "you_source": you_source,
            "input": source, "you_trusted": trusted, "you_why": why, "changes": changes,
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
