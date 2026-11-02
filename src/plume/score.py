"""plume.score — the session document (SPEC section 3).

The JSON layout mirrors the dataclasses 1:1 (``dataclasses.asdict``) plus a
top-level ``"plume_version": 0``. This file IS the API's core noun —
everything the CLI does is a read-modify-write of it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PLUME_VERSION = 0

__all__ = [
    "Track", "SectionSpec", "Movement", "Session",
    "save", "load", "total_bars", "PLUME_VERSION",
]


@dataclass
class Track:
    name: str
    voice: str                        # preset name, see voices.PRESETS
    kind: str                         # "drum" | "tonal"
    gain_db: float = 0.0
    pan: float = 0.0
    params: dict[str, float | str] = field(default_factory=dict)
    fx: dict[str, float] = field(default_factory=dict)
        # optional keys: lpf_hz, hpf_hz, drive, chorus (0-1), delay (0-1 wet),
        # reverb (0-1 send), duck (0-1 depth, sidechained from the kick track)
    patterns: dict[str, str] = field(default_factory=dict)  # name -> mini-notation


@dataclass
class SectionSpec:
    name: str
    bars: int
    energy: float                     # 0..1, gates layers & brightness
    plays: dict[str, str]             # track name -> pattern name ("" / absent = tacet)
    chords: list[str] = field(default_factory=list)  # cycled across bars (one per bar)
    fill: bool = True                 # fill in last bar of the section


@dataclass
class Movement:
    name: str
    sections: list[SectionSpec]


@dataclass
class Session:
    name: str
    bpm: float
    key: str                          # e.g. "F min" — parse with theory.parse_key
    swing: float = 0.5                # 0.5 straight .. 0.66 triplet
    seed: int = 0
    style: str = ""                   # primary style name
    timing_bias_ms: dict[str, float] = field(default_factory=dict)  # role -> ms
    humanize_ms: float = 4.0          # timing jitter SD, <=10
    humanize_vel: float = 0.08        # velocity jitter SD (fraction)
    tracks: list[Track] = field(default_factory=list)
    movements: list[Movement] = field(default_factory=list)
    master: dict[str, float] = field(default_factory=dict)
        # reverb (0-1), drive (0-1), lofi (0-1), limit_db (-1.0 ceiling)


def save(session: Session, path: str | Path) -> None:
    """Write the session as JSON: sorted keys, indent=2, version field."""
    d = asdict(session)
    d["plume_version"] = PLUME_VERSION
    Path(path).write_text(json.dumps(d, sort_keys=True, indent=2) + "\n",
                          encoding="utf-8")


_DRUM_VOICES = {"808kick", "hat", "openhat", "clap", "snare", "crackle"}


def _track_from_dict(d: dict) -> Track:
    voice = str(d.get("voice", ""))
    kind = d.get("kind") or ("drum" if voice in _DRUM_VOICES else "tonal")
    return Track(
        name=str(d.get("name", "")),
        voice=voice,
        kind=str(kind),
        gain_db=float(d.get("gain_db", 0.0)),
        pan=float(d.get("pan", 0.0)),
        params=dict(d.get("params") or {}),
        fx={k: float(v) for k, v in (d.get("fx") or {}).items()},
        patterns={k: str(v) for k, v in (d.get("patterns") or {}).items()},
    )


def _section_from_dict(d: dict) -> SectionSpec:
    return SectionSpec(
        name=str(d.get("name", "")),
        bars=int(d.get("bars", 0)),
        energy=float(d.get("energy", 0.5)),
        plays={k: str(v) for k, v in (d.get("plays") or {}).items()},
        chords=[str(c) for c in (d.get("chords") or [])],
        fill=bool(d.get("fill", True)),
    )


def _movement_from_dict(d: dict) -> Movement:
    return Movement(
        name=str(d.get("name", "")),
        sections=[_section_from_dict(s) for s in (d.get("sections") or [])],
    )


def _session_from_dict(d: dict) -> Session:
    for key in ("tracks", "movements"):
        if d.get(key) is not None and not isinstance(d[key], list):
            raise ValueError(f"{key!r} must be a list of objects, "
                             f"got {type(d[key]).__name__}")
    for i, t in enumerate(d.get("tracks") or []):
        if not isinstance(t, dict):
            raise ValueError(f"tracks[{i}] must be an object, got {type(t).__name__}")
    for i, m in enumerate(d.get("movements") or []):
        if not isinstance(m, dict):
            raise ValueError(f"movements[{i}] must be an object, got {type(m).__name__}")
    return Session(
        name=str(d.get("name", "untitled")),
        bpm=float(d.get("bpm", 90.0)),
        key=str(d.get("key", "C min")),
        swing=float(d.get("swing", 0.5)),
        seed=int(d.get("seed", 0)),
        style=str(d.get("style", "")),
        timing_bias_ms={k: float(v) for k, v in (d.get("timing_bias_ms") or {}).items()},
        humanize_ms=float(d.get("humanize_ms", 4.0)),
        humanize_vel=float(d.get("humanize_vel", 0.08)),
        tracks=[_track_from_dict(t) for t in (d.get("tracks") or [])],
        movements=[_movement_from_dict(m) for m in (d.get("movements") or [])],
        master={k: float(v) for k, v in (d.get("master") or {}).items()},
    )


def load(path: str | Path) -> Session:
    """Load a session JSON; tolerant of missing optional fields."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"not a session document: {path}")
    return _session_from_dict(d)


def total_bars(session: Session) -> int:
    return sum(sec.bars for mv in session.movements for sec in mv.sections)
