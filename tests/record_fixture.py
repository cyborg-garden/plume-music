"""Synthetic take folders, in the real shape of a tap-start take.

``renders/*.ndjson`` are sim rows with no ``entry`` or ``fade_db``, so The Record
needs its own fixture.  The field list follows the row ``realtime.lead`` writes,
and the rows carry the mic-derived free text a real row can carry, on purpose.  A
fixture of clean strings proves nothing about an honesty filter.

No audio is involved, and nothing here came from a real take.  Run as a script to
write a shelf of takes for looking at the page::

    python tests/record_fixture.py OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plume import realtime  # noqa: E402
from plume.bandentry import song_section  # noqa: E402

DRUM_ROLES = ("kick", "snare", "hat", "sub")
BAND_ROLES = (*DRUM_ROLES, "bass", "pad", "keys", "shimmer", "crash")
FULL_ROLES = (*BAND_ROLES, "lead", "arp")

#: Text that must never reach the page.  Each is shaped like what the engine writes.
MIC_FORM_REASON = "extending 'verse' by 4 bars — human still mid-phrase (4/16 used)"
MIC_FORM_REASON_2 = "human +4.2 dB above their baseline -> jump"
MIC_ROLE_REASON = "human +3.1 dB above their baseline for 8 bars -> jump"
MIC_ROLE_EVENT = "human dragging 37.125 ms"
MIC_REASON = "following the player's phrasing, they sound tired tonight"
MIC_ENTRY_REASON = "drums: heard 3 bars, no settled key"
#: Sentinel measurements: these exact digits must not appear in any output.
OFFSET_MS = 37.125
ROLE_OFFSET_MS = 41.625
#: Fields of session.json that say where and on what.  None may leave the bridge.
SECRET_RECORD_DIR = "/nowhere/placeholder-record-dir"
SECRET_DEVICE = "placeholder-device-name"
SECRET_PROMPT = "placeholder-prompt-text"

POISON = (MIC_FORM_REASON, "mid-phrase", MIC_FORM_REASON_2, "baseline", MIC_ROLE_REASON,
          MIC_ROLE_EVENT, "dragging 37", MIC_REASON, "sound tired", MIC_ENTRY_REASON,
          "no settled key", str(OFFSET_MS), str(ROLE_OFFSET_MS), "37.13", "41.63",
          SECRET_RECORD_DIR, SECRET_DEVICE, SECRET_PROMPT, "understood, form unchanged",
          "for the renderer", "NOT applied", "human")


def make_row(bar: int, *, bpm: float = 53.0, entry: str | None = "full", section: str | None = "verse",
             key: str | None = "F major", chord: str | None = "Fmaj7", density: float = 0.45,
             roles=FULL_ROLES, fade_db: float = 0.0, playing: bool = True, action: str = "hold",
             applied: str | None = None, dropped: str | None = None, effects=None, note: str = "",
             t_s: float | None = None) -> dict:
    """One row in the shape ``realtime.lead`` writes, mic-derived text included."""
    bar_s = 240.0 / bpm
    doc = {
        "bar": bar, "t_s": round(4.5 + bar * bar_s if t_s is None else t_s, 4),
        "state": "comping", "reason": MIC_REASON, "bpm": bpm,
        "roles": {r: 1.0 for r in roles}, "gain_db": -10.0 + fade_db, "density": density,
        "key": key, "chord": chord, "role": "lead", "clock_authority": 1.0,
        "form_authority": 1.0, "tempo_offset_pct": 0.0, "transpose_semitones": 0,
        "role_human_offset_ms": ROLE_OFFSET_MS, "tempo_policy": "hold",
        "role_reason": MIC_ROLE_REASON, "role_events": [MIC_ROLE_EVENT],
        "form": {"section_idx": bar % 3, "action": action, "bars_left": 3 - bar % 4,
                 "announce": False, "applied_cue": applied, "dropped_cue": dropped,
                 "reason": f"{MIC_FORM_REASON}; {MIC_FORM_REASON_2}" + (f"; {note}" if note else "")},
        "human_offset_ms": OFFSET_MS, "cue_effects": effects,
    }
    if entry is not None:
        doc.update(entry=entry, entry_reason=MIC_ENTRY_REASON, section=section,
                   playing=playing, fade_db=fade_db)
    return doc


def song(bars: int, *, bpm: float = 53.0, key: str = "F major", heard_bar: int | None = 3,
         band_bar: int | None = 8, outro_bar: int | None = None,
         cues: dict | None = None) -> list[dict]:
    """A whole take: drums, band, full band, the real song arc, then maybe an outro.

    ``cues`` maps a bar to extra ``make_row`` arguments.
    """
    rows = []
    for bar in range(bars):
        kw = dict((cues or {}).get(bar, {}))
        if band_bar is None or bar < band_bar:
            kw.setdefault("entry", "drums")
            kw.setdefault("section", "intro")
            kw.setdefault("key", None)
            kw.setdefault("chord", None)
            kw.setdefault("density", 0.55)
            kw.setdefault("roles", DRUM_ROLES)
            kw.setdefault("playing", heard_bar is not None and bar >= heard_bar)
        else:
            stage = "band" if bar < band_bar + 4 else "full"
            section, density, resting = song_section(bar - band_bar)
            if outro_bar is not None and bar >= outro_bar:
                section, density, resting = "outro", 0.3, ("sub", "shimmer")
            roles = BAND_ROLES if stage == "band" else FULL_ROLES
            kw.setdefault("entry", stage)
            kw.setdefault("section", section)
            kw.setdefault("key", key)
            kw.setdefault("density", density)
            kw.setdefault("roles", tuple(r for r in roles if r not in resting))
        rows.append(make_row(bar, bpm=bpm, **kw))
    return rows


def landed(text: str = "busier", before: float = 1.0, after: float = 1.43) -> dict:
    return {"applied": text, "effects": [f"cue {text!r} (texture:{text}): density "
                                         f"x{before:.2f} -> x{after:.2f}"]}


def ending_doc(reason=realtime.END_STOPPED, *, elapsed_s: float = 240.0, bars: int = 40) -> dict:
    return {"reason": reason, "capped": reason == realtime.END_MAX_MINUTES,
            "max_minutes": 60.0, "elapsed_s": elapsed_s, "bars": bars}


def cue_line(text: str, bar: int | None, t_s: float, source: str = "gui") -> dict:
    return {"received_at": 1800000000.0, "t_s": t_s, "bar": bar, "text": text, "source": source,
            "at": "next-phrase", "ttl_bars": 16}


def write_take(folder, rows=(), *, ending=None, cues=None, seed: int = 7, started_at: float = 1800000000.0,
               wav_bytes: bytes | None = None) -> Path:
    """Write a take folder.  ``cues=None`` leaves no cue log, as an older take has."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / realtime.REC_MANIFEST).write_text(json.dumps({
        "kind": "lead", "seed": seed, "started_at": started_at, "bpm": 120.0,
        "record_dir": SECRET_RECORD_DIR, "device": SECRET_DEVICE, "name": SECRET_PROMPT,
        "socket": SECRET_RECORD_DIR + "/sock"}))
    (folder / realtime.REC_DECISIONS).write_text("".join(json.dumps(r) + "\n" for r in rows))
    if cues is not None:
        (folder / realtime.REC_CUES).write_text("".join(json.dumps(c) + "\n" for c in cues))
    if ending is not None:
        (folder / realtime.REC_ENDING).write_text(json.dumps(ending))
    if wav_bytes is not None:
        (folder / realtime.REC_HUMAN).write_bytes(wav_bytes)
        (folder / realtime.REC_MACHINE).write_bytes(wav_bytes)
    return folder


def write_shelf(root) -> list[Path]:
    """A varied shelf, to look at.  Every number and name is invented."""
    root = Path(root)
    day = 86400.0
    start = 1800000000.0
    bar_s = 240.0 / 53.0
    made = []
    # A long take with cues that landed, one that did nothing, and a name.
    made.append(write_take(
        root / "take-13",
        song(53, bpm=53.0, band_bar=9, outro_bar=None, cues={
            20: landed("busier"), 28: landed("sparser", 1.43, 1.00),
            36: {"applied": "louder"}, 44: {"applied": "wrap it up", "action": "end"}})[:45],
        ending=ending_doc(realtime.END_FORM, elapsed_s=209.0, bars=45),
        cues=[cue_line("busier", 17, 4.5 + 16.2 * bar_s), cue_line("sparser", 26, 4.5 + 25.4 * bar_s),
              cue_line("louder", 34, 4.5 + 33.5 * bar_s, "cli"), cue_line("wrap it up", 42, 4.5 + 41.2 * bar_s)],
        seed=113, started_at=start))
    (made[-1] / "keep.json").write_text(json.dumps({"kept": True, "name": "slow one, late"}))
    # A minor take that ran its full length and wound down.
    made.append(write_take(
        root / "take-12",
        song(64, bpm=72.0, key="A minor", band_bar=8, outro_bar=54, cues={
            22: landed("sparser", 1.0, 0.7), 40: landed("darker") | {
                "effects": ["cue 'darker' (texture:darker): tone +0.0 -> -1.0 from here on"]}}),
        ending=ending_doc(realtime.END_SONG, elapsed_s=225.0, bars=64),
        cues=[cue_line("sparser", 20, 4.5 + 19.4 * 240 / 72), cue_line("darker", 37, 4.5 + 36.5 * 240 / 72, "voice")],
        seed=212, started_at=start - day))
    # A quick take stopped early, from before the cue log existed.
    made.append(write_take(
        root / "take-11",
        song(24, bpm=96.0, key="G major", band_bar=8, heard_bar=2, cues={12: landed("busier")}),
        ending=ending_doc(realtime.END_INTERRUPT, elapsed_s=72.0, bars=24),
        seed=311, started_at=start - 2 * day))
    # Nobody came in.
    made.append(write_take(
        root / "take-10", song(9, bpm=61.0, heard_bar=None, band_bar=None),
        ending=ending_doc(realtime.END_SILENCE, elapsed_s=38.0, bars=9), cues=[],
        seed=410, started_at=start - 3 * day))
    # Killed: no ending record.
    made.append(write_take(
        root / "take-9", song(30, bpm=84.0, key="D minor", band_bar=12, heard_bar=9),
        cues=[], seed=509, started_at=start - 4 * day))
    # Stopped during the count.  No rows, and a mic recording in a real one.
    made.append(write_take(
        root / "take-8", (), ending=ending_doc(realtime.END_STOPPED, elapsed_s=11.0, bars=0),
        cues=[], seed=608, started_at=start - 5 * day))
    return made


if __name__ == "__main__":
    for path in write_shelf(sys.argv[1]):
        print(path.name)
