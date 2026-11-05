"""plume CLI — SPEC section 11.

``plume <command> [args] [--session PATH] [--json]``

Session resolution order: explicit ``--session`` (a bare name also tries
``<name>.plume.json``), else ``PLUME_SESSION`` env var, else exactly one
``*.plume.json`` in the cwd, else an error listing candidates.

Every command supports ``--json`` (machine output: exactly one JSON document
on stdout, nothing else). Errors in ``--json`` mode: nonzero exit and
``{"error": ...}`` on stdout. Every mutating command prints the new state
hash — sha256 of the canonical JSON (sorted keys, compact separators), first
12 hex chars — so an LLM can verify effects.

stdlib-only module (argparse); plume modules are imported lazily inside the
command handlers that need them.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import json
import math
import os
import re
import sys
from pathlib import Path

SESSION_SUFFIX = ".plume.json"

# Top-level Session fields (score.Session). `set` refuses unknown top-level
# keys so typos ("bmp") fail loudly instead of writing junk.
_SESSION_TOP_KEYS = {
    "name", "bpm", "key", "swing", "seed", "style", "timing_bias_ms",
    "humanize_ms", "humanize_vel", "tracks", "movements", "master",
    "plume_version",
}

_STR_TOP_KEYS = {"name", "key", "style"}
_NUM_TOP_KEYS = {"bpm", "swing", "humanize_ms", "humanize_vel"}

# session document schema for validated `set` (v0.1 amendment: nested paths
# are checked against these key sets instead of silently writing junk)
_MASTER_KEYS = {"gain_db", "reverb", "drive", "lofi", "limit_db"}
_FX_KEYS = {"lpf_hz", "hpf_hz", "drive", "chorus", "delay", "reverb", "duck"}
_TRACK_FIELD_KEYS = {"gain_db", "pan", "voice", "kind", "fx", "params", "patterns"}
_MOVEMENT_KEYS = {"name", "sections"}
_SECTION_KEYS = {"name", "bars", "energy", "plays", "chords", "fill"}
_EXTRA_PARAM_KEYS = {"voicing"}   # engine hints that are not voice params

# Presets whose tracks default to kind="drum" when --kind is not given.
_DRUM_PRESETS = {"808kick", "hat", "openhat", "clap", "snare", "crackle",
                 "housekick", "boomkick", "offhat", "brush"}


class CliError(Exception):
    """User-facing CLI error: message only, no traceback."""


# ---------------------------------------------------------------- helpers

def _canonical_json(doc) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def state_hash(doc) -> str:
    """sha256 of canonical JSON, first 12 hex chars."""
    return hashlib.sha256(_canonical_json(doc).encode("utf-8")).hexdigest()[:12]


def _read_doc(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except FileNotFoundError:
        raise CliError(f"session file not found: {path}") from None
    except json.JSONDecodeError as e:
        raise CliError(f"session file invalid: not valid JSON — {e} ({path})") from None
    if not isinstance(doc, dict):
        raise CliError(f"session file invalid: not a JSON object ({path})")
    return doc


def _write_doc(path: Path, doc: dict) -> None:
    doc.setdefault("plume_version", 0)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, sort_keys=True, indent=2)
        f.write("\n")


def resolve_session(explicit: str | None) -> Path:
    """SPEC session resolution: --session, then PLUME_SESSION, then cwd."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        # convenience: `--session verify1` finds `verify1.plume.json`
        if not str(explicit).endswith(SESSION_SUFFIX):
            alt = Path(str(explicit) + SESSION_SUFFIX)
            if alt.exists():
                return alt
            raise CliError(f"session file not found: tried {p} and {alt}")
        raise CliError(f"session file not found: {p}")
    env = os.environ.get("PLUME_SESSION")
    if env:
        p = Path(env)
        if not p.exists():
            raise CliError(f"PLUME_SESSION points at a missing file: {p}")
        return p
    candidates = sorted(Path.cwd().glob("*" + SESSION_SUFFIX))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise CliError(
            "no session found: pass --session PATH, set PLUME_SESSION, or "
            "create one with 'plume make'/'plume new'"
        )
    names = ", ".join(c.name for c in candidates)
    raise CliError(
        f"multiple session files in cwd — pass --session PATH. candidates: {names}"
    )


def _load_session(path: Path):
    """score.load with a clean error instead of a raw traceback (v0.1)."""
    score = _import("plume.score")
    try:
        return score.load(path)
    except CliError:
        raise
    except Exception as e:  # noqa: BLE001 — corrupted files raise all sorts
        why = str(e) if isinstance(e, (ValueError, KeyError)) else f"{type(e).__name__}: {e}"
        raise CliError(f"session file invalid: {why} ({path})") from None


def _parse_value(raw: str):
    """VALUE for `set`: JSON literal when it parses, bare string otherwise."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _import(modname: str):
    try:
        return importlib.import_module(modname)
    except ImportError as e:
        raise CliError(f"cannot import {modname}: {e}") from None


def _safe_filename(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    return out or "session"


def _find_track(doc: dict, name: str) -> dict:
    for t in doc.get("tracks", []):
        if t.get("name") == name:
            return t
    known = ", ".join(t.get("name", "?") for t in doc.get("tracks", [])) or "(none)"
    raise CliError(f"no track named {name!r} (tracks: {known})")


# ---------------------------------------------------------------- validated set

def _dym(key: str, valid) -> str:
    m = difflib.get_close_matches(str(key), sorted(str(v) for v in valid), n=1)
    return f" — did you mean {m[0]!r}?" if m else ""


def _bad_key(key: str, valid, ctx: str):
    raise CliError(
        f"unknown key {key!r} in {ctx} (valid: {', '.join(sorted(valid))})"
        + _dym(key, valid)
    )


def _need_num(dotted: str, value, *, integer: bool = False, lo=None, hi=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CliError(f"{dotted} must be a number, got {json.dumps(value)}")
    if integer and not isinstance(value, int):
        raise CliError(f"{dotted} must be an integer, got {json.dumps(value)}")
    if lo is not None and value < lo or hi is not None and value > hi:
        raise CliError(f"{dotted} must be in range {lo}..{hi}, got {value}")


def _need_str(dotted: str, value):
    if not isinstance(value, str):
        raise CliError(f"{dotted} must be a string, got {json.dumps(value)}")


def _need_bool(dotted: str, value):
    if not isinstance(value, bool):
        raise CliError(f"{dotted} must be true or false, got {json.dumps(value)}")


def _need_dict(dotted: str, value, valid_keys, ctx: str):
    if not isinstance(value, dict):
        raise CliError(
            f"refusing to replace object {dotted} with a non-object value "
            f"({json.dumps(value)}); set a nested key instead "
            f"(valid: {', '.join(sorted(valid_keys))})"
        )
    for k in value:
        if k not in valid_keys:
            _bad_key(k, valid_keys, ctx)


def _validate_pattern(dotted: str, value):
    _need_str(dotted, value)
    notation = _import("plume.notation")
    try:
        notation.parse(value)
    except notation.NotationError as e:
        raise CliError(f"bad pattern for {dotted}: {e} "
                       "(see 'plume notation' for the grammar)") from None


def _validate_chord(dotted: str, value):
    _need_str(dotted, value)
    theory = _import("plume.theory")
    try:
        theory.parse_chord(value)
    except ValueError as e:
        raise CliError(f"{dotted}: {e} (chord symbols like Cm7, Abmaj7, G7, Fsus2)") from None


def _resolve_list_item(node: list, key: str, ctx: str):
    """List element by int index or (for tracks) by element name."""
    try:
        idx = int(key)
    except ValueError:
        for item in node:
            if isinstance(item, dict) and item.get("name") == key:
                return item
        names = ", ".join(str(i.get("name", "?")) for i in node if isinstance(i, dict))
        raise CliError(f"no element named {key!r} in {ctx} "
                       f"(have: {names or '(none)'})") from None
    if not -len(node) <= idx < len(node):
        raise CliError(f"index {idx} out of range in {ctx} (0..{len(node) - 1})")
    return node[idx]


def _set_track_path(doc: dict, keys: list[str], value, dotted: str) -> None:
    if len(keys) == 1:
        raise CliError("cannot replace 'tracks' wholesale — use "
                       "'plume track add/rm' and per-track paths like "
                       "tracks.NAME.gain_db")
    tracks = doc.get("tracks", [])
    if not isinstance(tracks, list):
        raise CliError("session file invalid: 'tracks' is not a list")
    track = _resolve_list_item(tracks, keys[1], "tracks")
    if not isinstance(track, dict):
        raise CliError(f"session file invalid: track {keys[1]!r} is not an object")
    tname = track.get("name", keys[1])
    if len(keys) == 2:
        raise CliError(
            f"set a field on track {tname!r} instead of replacing it "
            f"(valid: {', '.join(sorted(_TRACK_FIELD_KEYS))}); "
            "to remove it use 'plume track rm'"
        )
    k2 = keys[2]
    if k2 == "name":
        raise CliError("renaming tracks via 'set' is not supported "
                       "(section plays reference track names); "
                       "add a new track and remove the old one instead")
    if k2 not in _TRACK_FIELD_KEYS:
        _bad_key(k2, _TRACK_FIELD_KEYS, f"tracks.{tname}")
    if k2 in ("gain_db", "pan"):
        if len(keys) != 3:
            raise CliError(f"{dotted}: path too deep — tracks.{tname}.{k2} is a number")
        _need_num(dotted, value)
        track[k2] = value
        return
    if k2 == "voice":
        if len(keys) != 3:
            raise CliError(f"{dotted}: path too deep — tracks.{tname}.voice is a string")
        voices = _import("plume.voices")
        if value not in voices.PRESETS:
            raise CliError(f"unknown voice {value!r} "
                           f"(presets: {', '.join(sorted(voices.PRESETS))})"
                           + _dym(str(value), voices.PRESETS))
        track[k2] = value
        return
    if k2 == "kind":
        if len(keys) != 3 or value not in ("drum", "tonal"):
            raise CliError(f"{dotted} must be \"drum\" or \"tonal\", "
                           f"got {json.dumps(value)}")
        track[k2] = value
        return
    # containers: fx / params / patterns
    if len(keys) > 4:
        raise CliError(f"{dotted}: path too deep — "
                       f"tracks.{tname}.{k2} holds scalar values one level down")
    if k2 == "fx":
        if len(keys) == 3:
            _need_dict(dotted, value, _FX_KEYS, f"tracks.{tname}.fx")
            for k, v in value.items():
                _need_num(f"{dotted}.{k}", v)
            track["fx"] = value
            return
        if keys[3] not in _FX_KEYS:
            _bad_key(keys[3], _FX_KEYS, f"tracks.{tname}.fx")
        _need_num(dotted, value)
        track.setdefault("fx", {})[keys[3]] = value
        return
    if k2 == "params":
        voices = _import("plume.voices")
        pnames = set(voices.PRESETS.get(track.get("voice", ""), {})) | _EXTRA_PARAM_KEYS
        ctx = f"tracks.{tname}.params (voice {track.get('voice')!r})"
        if len(keys) == 3:
            _need_dict(dotted, value, pnames, ctx)
            track["params"] = value
            return
        if keys[3] not in pnames:
            _bad_key(keys[3], pnames, ctx)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise CliError(f"{dotted} must be a number or string, got {json.dumps(value)}")
        track.setdefault("params", {})[keys[3]] = value
        return
    # patterns
    if len(keys) == 3:
        if not isinstance(value, dict):
            raise CliError(f"refusing to replace object {dotted} with a non-object "
                           f"value; set tracks.{tname}.patterns.NAME instead")
        for k, v in value.items():
            _validate_pattern(f"{dotted}.{k}", v)
        track["patterns"] = value
        return
    _validate_pattern(dotted, value)
    track.setdefault("patterns", {})[keys[3]] = value


def _set_movement_path(doc: dict, keys: list[str], value, dotted: str) -> None:
    if len(keys) == 1:
        raise CliError("cannot replace 'movements' wholesale via 'set' — "
                       "use 'plume arrange set --spec-json ...'")
    movements = doc.get("movements", [])
    if not isinstance(movements, list):
        raise CliError("session file invalid: 'movements' is not a list")
    try:
        midx = int(keys[1])
    except ValueError:
        raise CliError(f"movements are addressed by index: movements.0.name, "
                       f"movements.0.sections.1.energy (got {keys[1]!r})") from None
    if not 0 <= midx < len(movements):
        raise CliError(f"movement index {midx} out of range (0..{len(movements) - 1})")
    mv = movements[midx]
    if len(keys) == 2:
        raise CliError(f"set a field on movements.{midx} instead of replacing it "
                       f"(valid: {', '.join(sorted(_MOVEMENT_KEYS))}) — "
                       "or use 'plume arrange set'")
    if keys[2] not in _MOVEMENT_KEYS:
        _bad_key(keys[2], _MOVEMENT_KEYS, f"movements.{midx}")
    if keys[2] == "name":
        if len(keys) != 3:
            raise CliError(f"{dotted}: path too deep")
        _need_str(dotted, value)
        mv["name"] = value
        return
    # sections
    if len(keys) == 3:
        raise CliError("cannot replace a sections list via 'set' — "
                       "use 'plume arrange set --spec-json ...'")
    sections = mv.get("sections", [])
    try:
        sidx = int(keys[3])
    except ValueError:
        raise CliError(f"sections are addressed by index "
                       f"(movements.{midx}.sections.0...), got {keys[3]!r}") from None
    if not 0 <= sidx < len(sections):
        raise CliError(f"section index {sidx} out of range (0..{len(sections) - 1})")
    sec = sections[sidx]
    ctx = f"movements.{midx}.sections.{sidx}"
    if len(keys) == 4:
        raise CliError(f"set a field on {ctx} instead of replacing it "
                       f"(valid: {', '.join(sorted(_SECTION_KEYS))})")
    k4 = keys[4]
    if k4 not in _SECTION_KEYS:
        _bad_key(k4, _SECTION_KEYS, ctx)
    if k4 == "name":
        if len(keys) != 5:
            raise CliError(f"{dotted}: path too deep")
        _need_str(dotted, value)
        sec["name"] = value
    elif k4 == "bars":
        if len(keys) != 5:
            raise CliError(f"{dotted}: path too deep")
        _need_num(dotted, value, integer=True, lo=1)
        sec["bars"] = value
    elif k4 == "energy":
        if len(keys) != 5:
            raise CliError(f"{dotted}: path too deep")
        _need_num(dotted, value, lo=0.0, hi=1.0)
        sec["energy"] = value
    elif k4 == "fill":
        if len(keys) != 5:
            raise CliError(f"{dotted}: path too deep")
        _need_bool(dotted, value)
        sec["fill"] = value
    elif k4 == "plays":
        def _check_play(tname: str, pname, path: str):
            track = _find_track(doc, tname)
            _need_str(path, pname)
            if pname and pname not in track.get("patterns", {}):
                known = ", ".join(sorted(track.get("patterns", {}))) or "(none)"
                raise CliError(f"{path}: track {tname!r} has no pattern "
                               f"{pname!r} (patterns: {known})")
        if len(keys) == 5:
            if not isinstance(value, dict):
                raise CliError(f"refusing to replace object {dotted} with a "
                               f"non-object value; set {ctx}.plays.TRACK instead")
            for tname, pname in value.items():
                _check_play(tname, pname, f"{dotted}.{tname}")
            sec["plays"] = value
        elif len(keys) == 6:
            _check_play(keys[5], value, dotted)
            sec.setdefault("plays", {})[keys[5]] = value
        else:
            raise CliError(f"{dotted}: path too deep")
    else:  # chords
        if len(keys) == 5:
            if not isinstance(value, list):
                raise CliError(f"{dotted} must be a list of chord symbols, "
                               f"got {json.dumps(value)}")
            for i, sym in enumerate(value):
                _validate_chord(f"{dotted}.{i}", sym)
            sec["chords"] = value
        elif len(keys) == 6:
            chords = sec.get("chords", [])
            try:
                cidx = int(keys[5])
            except ValueError:
                raise CliError(f"chords are addressed by index, got {keys[5]!r}") from None
            if not 0 <= cidx < len(chords):
                raise CliError(f"chord index {cidx} out of range (0..{len(chords) - 1})")
            _validate_chord(dotted, value)
            chords[cidx] = value
        else:
            raise CliError(f"{dotted}: path too deep")


def set_dotted(doc: dict, dotted: str, value) -> None:
    """Validated write of a dotted path against the session document schema.

    v0.1 amendment: unknown keys, type-destroying writes (replacing an
    object with a scalar) and out-of-schema values are rejected with an
    error that lists the valid keys at that level.
    """
    keys = dotted.split(".")
    if not keys or any(k == "" for k in keys):
        raise CliError(f"bad key path: {dotted!r}")
    k0 = keys[0]
    if k0 not in _SESSION_TOP_KEYS:
        known = sorted(_SESSION_TOP_KEYS - {"plume_version"})
        raise CliError(f"unknown session field {k0!r} "
                       f"(known: {', '.join(known)})" + _dym(k0, known))
    if k0 == "plume_version":
        raise CliError("plume_version is managed automatically and cannot be set")
    if k0 in _STR_TOP_KEYS:
        if len(keys) != 1:
            raise CliError(f"{dotted}: path too deep — {k0} is a string")
        _need_str(dotted, value)
        if k0 == "key":
            theory = _import("plume.theory")
            try:
                theory.parse_key(value)
            except ValueError as e:
                raise CliError(str(e)) from None
        doc[k0] = value
        return
    if k0 in _NUM_TOP_KEYS:
        if len(keys) != 1:
            raise CliError(f"{dotted}: path too deep — {k0} is a number")
        _need_num(dotted, value)
        doc[k0] = value
        return
    if k0 == "seed":
        if len(keys) != 1:
            raise CliError(f"{dotted}: path too deep — seed is an integer")
        _need_num(dotted, value, integer=True)
        doc[k0] = value
        return
    if k0 == "master":
        if len(keys) == 1:
            _need_dict(dotted, value, _MASTER_KEYS, "master")
            for k, v in value.items():
                _need_num(f"master.{k}", v)
            doc["master"] = value
            return
        if len(keys) != 2:
            raise CliError(f"{dotted}: path too deep — master.* values are numbers")
        if keys[1] not in _MASTER_KEYS:
            _bad_key(keys[1], _MASTER_KEYS, "master")
        _need_num(dotted, value)
        doc.setdefault("master", {})[keys[1]] = value
        return
    if k0 == "timing_bias_ms":
        if len(keys) == 1:
            if not isinstance(value, dict):
                raise CliError(f"refusing to replace object {dotted} with a "
                               "non-object value; set timing_bias_ms.ROLE instead")
            for k, v in value.items():
                _need_num(f"{dotted}.{k}", v)
            doc["timing_bias_ms"] = value
            return
        if len(keys) != 2:
            raise CliError(f"{dotted}: path too deep — timing_bias_ms.ROLE is a number")
        _need_num(dotted, value)
        doc.setdefault("timing_bias_ms", {})[keys[1]] = value
        return
    if k0 == "tracks":
        _set_track_path(doc, keys, value, dotted)
        return
    # movements
    _set_movement_path(doc, keys, value, dotted)


# ---------------------------------------------------------------- get / unset

def _descend(doc: dict, dotted: str):
    """Read-only descent of a dotted path (never creates keys)."""
    node = doc
    walked: list[str] = []
    for k in dotted.split("."):
        if k == "":
            raise CliError(f"bad key path: {dotted!r}")
        ctx = ".".join(walked) or "session"
        if isinstance(node, list):
            node = _resolve_list_item(node, k, ctx)
        elif isinstance(node, dict):
            if k not in node:
                valid = sorted(node)
                raise CliError(f"no key {k!r} in {ctx} "
                               f"(has: {', '.join(valid) or '(empty)'})"
                               + _dym(k, valid))
            node = node[k]
        else:
            raise CliError(f"cannot descend into {ctx!r}: not an object or list")
        walked.append(k)
    return node


# Schema defaults for `get` on known keys absent from the document.
# master.* mirrors the engine's defaults; top-level scalars mirror score.Session.
_MASTER_DEFAULTS = {"gain_db": 0.0, "reverb": 0.0, "drive": 0.0,
                    "lofi": 0.0, "limit_db": -1.0}
_TOP_DEFAULTS = {"swing": 0.5, "seed": 0, "style": "",
                 "humanize_ms": 4.0, "humanize_vel": 0.08}


def _schema_default(doc: dict, dotted: str) -> tuple[bool, object]:
    """(True, default) when ``dotted`` is a known-schema path with a default
    the engine would use; (False, None) otherwise (unknown keys keep erroring).
    """
    keys = dotted.split(".")
    k0 = keys[0]
    if len(keys) == 1 and k0 in _TOP_DEFAULTS:
        return True, _TOP_DEFAULTS[k0]
    if k0 == "master" and len(keys) == 2 and keys[1] in _MASTER_DEFAULTS:
        return True, _MASTER_DEFAULTS[keys[1]]
    if k0 == "timing_bias_ms" and len(keys) == 2:
        return True, 0.0
    if k0 == "tracks" and len(keys) in (3, 4):
        try:
            track = _resolve_list_item(doc.get("tracks") or [], keys[1], "tracks")
        except CliError:
            return False, None
        if not isinstance(track, dict):
            return False, None
        k2 = keys[2]
        if len(keys) == 3:
            if k2 in ("gain_db", "pan"):
                return True, 0.0
            if k2 == "kind":
                return True, ("drum" if track.get("voice") in _DRUM_PRESETS
                              else "tonal")
            if k2 in ("fx", "params", "patterns"):
                return True, {}
            return False, None
        if k2 == "fx" and keys[3] in _FX_KEYS:
            return True, None  # absent fx = effect off; no numeric default
        if k2 == "params":
            voices = _import("plume.voices")
            pdefs = voices.PRESETS.get(track.get("voice", ""), {})
            if keys[3] in pdefs:
                return True, pdefs[keys[3]]
        return False, None
    if (k0 == "movements" and len(keys) == 5 and keys[2] == "sections"
            and keys[4] in ("fill", "chords")):
        try:
            _descend(doc, ".".join(keys[:4]))  # the section itself must exist
        except CliError:
            return False, None
        return True, (True if keys[4] == "fill" else [])
    return False, None


def _settable_paths(doc: dict) -> list[tuple[str, object]]:
    """All settable dotted paths with their current values (get --list)."""
    out: list[tuple[str, object]] = []
    for k in ("name", "bpm", "key", "swing", "seed", "style",
              "humanize_ms", "humanize_vel"):
        if k in doc:
            out.append((k, doc[k]))
    for k, v in sorted((doc.get("timing_bias_ms") or {}).items()):
        out.append((f"timing_bias_ms.{k}", v))
    for k in sorted(_MASTER_KEYS):
        out.append((f"master.{k}", (doc.get("master") or {}).get(k)))
    for t in doc.get("tracks", []):
        name = t.get("name", "?")
        for k in ("voice", "kind", "gain_db", "pan"):
            out.append((f"tracks.{name}.{k}", t.get(k)))
        for k, v in sorted((t.get("fx") or {}).items()):
            out.append((f"tracks.{name}.fx.{k}", v))
        for k, v in sorted((t.get("params") or {}).items()):
            out.append((f"tracks.{name}.params.{k}", v))
        for k, v in sorted((t.get("patterns") or {}).items()):
            out.append((f"tracks.{name}.patterns.{k}", v))
    for i, m in enumerate(doc.get("movements", [])):
        out.append((f"movements.{i}.name", m.get("name")))
        for j, s in enumerate(m.get("sections", [])):
            base = f"movements.{i}.sections.{j}"
            for k in ("name", "bars", "energy", "fill"):
                out.append((f"{base}.{k}", s.get(k)))
            for k, v in sorted((s.get("plays") or {}).items()):
                out.append((f"{base}.plays.{k}", v))
            out.append((f"{base}.chords", s.get("chords")))
    return out


# ---------------------------------------------------------------- summaries

def _played_track_names(doc: dict) -> set[str]:
    played: set[str] = set()
    for m in doc.get("movements", []):
        for s in m.get("sections", []):
            for tname, pname in (s.get("plays") or {}).items():
                if pname:
                    played.add(tname)
    return played


def _progression(doc: dict) -> list[str]:
    for m in doc.get("movements", []):
        for s in m.get("sections", []):
            if s.get("chords"):
                return list(s["chords"])
    return []


def _summary_dict(doc: dict) -> dict:
    played = _played_track_names(doc)
    return {
        "name": doc.get("name"),
        "key": doc.get("key"),
        "bpm": doc.get("bpm"),
        "style": doc.get("style"),
        "swing": doc.get("swing"),
        "seed": doc.get("seed"),
        "progression": _progression(doc),
        "tracks": [
            {"name": t.get("name"), "voice": t.get("voice"), "kind": t.get("kind"),
             "unused": t.get("name") not in played}
            for t in doc.get("tracks", [])
        ],
        "movements": [
            {
                "name": m.get("name"),
                "bars": sum(int(s.get("bars", 0)) for s in m.get("sections", [])),
                "sections": [
                    {
                        "name": s.get("name"),
                        "bars": s.get("bars"),
                        "energy": s.get("energy"),
                    }
                    for s in m.get("sections", [])
                ],
            }
            for m in doc.get("movements", [])
        ],
    }


def _movements_lines(doc: dict, only: int | None = None) -> list[str]:
    lines = []
    for i, m in enumerate(doc.get("movements", [])):
        if only is not None and i != only:
            continue
        total = sum(int(s.get("bars", 0)) for s in m.get("sections", []))
        lines.append(f"  [{i}] {m.get('name')}: {total} bars")
        for s in m.get("sections", []):
            lines.append(
                f"      {str(s.get('name')):<14} {int(s.get('bars', 0)):>3} bars"
                f"   energy {s.get('energy')}"
            )
    return lines


def _human_summary(doc: dict, path: Path | None = None, only_movement: int | None = None) -> str:
    lines = []
    head = f"session: {doc.get('name')}"
    if path is not None:
        head += f"  ({path})"
    lines.append(head)
    lines.append(
        f"key: {doc.get('key')}   bpm: {doc.get('bpm')}   "
        f"style: {doc.get('style') or '-'}   swing: {doc.get('swing')}   "
        f"seed: {doc.get('seed')}"
    )
    prog = _progression(doc)
    if prog:
        lines.append("progression: " + " ".join(str(c) for c in prog))
    tracks = doc.get("tracks", [])
    if tracks:
        played = _played_track_names(doc)
        lines.append(
            "tracks: " + ", ".join(
                f"{t.get('name')}({t.get('voice')})"
                + ("" if t.get("name") in played else " (unused)")
                for t in tracks
            )
        )
    if doc.get("movements"):
        lines.append("movements:")
        lines.extend(_movements_lines(doc, only=only_movement))
    return "\n".join(lines)


# ---------------------------------------------------------------- notation doc

_NOTATION_GRAMMAR = """\
pattern  := layer ("," layer)*          # layers play simultaneously
layer    := seq
seq      := elem+                       # whitespace separated; the bar is
                                        # divided equally among the elems
elem     := atom | group | elem "*" INT | elem "@" INT | alt
group    := "[" seq "]"                 # subdivides its slot
alt      := "<" atom+ ">"               # per-bar alternation (bar 1 -> first,
                                        # bar 2 -> second, ... cycling)
atom     := TOKEN | "~"                 # "~" = rest
TOKEN    := /[A-Za-z#][A-Za-z0-9#b.]*/  # bd, hh, c4, eb2, Cm9, Abmaj7 ...
suffix   := ":v" FLOAT                  # per-step velocity 0..1, e.g. bd:v0.5"""

_NOTATION_TOKENS = {
    "one pattern = one bar": (
        "A pattern string describes exactly one bar (one cycle); it repeats "
        "every bar the section plays it."
    ),
    "drum tracks": (
        "Any token is a trigger for the track's voice — 'x', 'bd', 'hh', 'sd' "
        "all just mean 'hit'. Token spelling is convention only."
    ),
    "tonal tracks — notes": (
        "Note names with octave: c4 (middle C = midi 60), eb2, f#3. "
        "'#' or 's' = sharp, 'b' = flat."
    ),
    "tonal tracks — chords": (
        "Chord symbols: C, Cm, Cm7, Cm9, Abmaj7, G7, Fsus2, Fsus4, Eb6, "
        "Dm7b5, Caug, Cdim7, Cadd9... Chords are auto-voiced; a bass (mono) "
        "voice takes the root only."
    ),
    "~": "Rest — occupies its slot in silence.",
    "[ ]": "Group: subdivides one slot. '[hh hh hh]' = a triplet in that slot.",
    "*": "Repeat within the slot: 'hh*8' = 8 hits squeezed into one slot.",
    "@": "Stretch across slots: 'c2@3' holds c2 for 3 slots' duration.",
    "< >": "Alternate per bar: '<Cm7 Fm7>' plays Cm7 in odd bars, Fm7 in even.",
    ",": "Layer separator: both sides play at once in the same bar.",
    ":v": "Velocity suffix on any token: 'sd:v0.35' = quiet ghost hit (0..1).",
}

_NOTATION_EXAMPLES = [
    {"pattern": "bd ~ ~ ~ bd ~ ~ ~ bd ~ ~ ~ bd ~ ~ ~",
     "description": "four-on-the-floor kick on a 16-slot grid"},
    {"pattern": "~ ~ ~ ~ sd ~ ~ ~ ~ ~ ~ ~ sd ~ ~ ~",
     "description": "backbeat snare (beats 2 and 4)"},
    {"pattern": "hh hh:v0.55 hh:v0.75 hh:v0.55 hh hh:v0.55 hh:v0.75 hh:v0.55",
     "description": "8th-note hats with velocity accents"},
    {"pattern": "[hh*4] [hh*4] [hh*3] [hh*8]",
     "description": "trap hat fill: 16ths, a triplet burst, then a 1/32 run"},
    {"pattern": "c1@3 ~ eb1 f1@2 ~",
     "description": "bass line with sustains: c1 held 3 slots, f1 held 2"},
    {"pattern": "<Cm7 Abmaj7 Eb Bb>",
     "description": "one chord per bar, cycling over 4 bars"},
    {"pattern": "bd ~ bd ~, ~ hh ~ hh",
     "description": "two layers at once: kick pattern + off-beat hats"},
    {"pattern": "bd ~ ~ <~ bd> ~ ~ bd ~",
     "description": "per-bar variation: the 4th slot hits only in even bars"},
]

_NOTATION_COMPACT = """\
mini-notation quick reference (one pattern = one bar):
  tokens: drum tracks: any token = hit (bd, hh, x...);
          tonal tracks: notes (c4, eb2, f#3) or chords (Cm7, Abmaj7, G7)
  ~ rest | [a b] subdivide slot | tok*4 repeat in slot | tok@3 stretch 3 slots
  <a b> alternate per bar | "," parallel layers | tok:v0.5 velocity 0..1
Full reference: plume notation"""


def cmd_notation(args):
    doc = {
        "grammar": _NOTATION_GRAMMAR,
        "tokens": _NOTATION_TOKENS,
        "examples": _NOTATION_EXAMPLES,
    }
    lines = ["plume mini-notation (Strudel subset) — one pattern string = one bar",
             "", "GRAMMAR", "", _NOTATION_GRAMMAR, "", "SEMANTICS", ""]
    for k, v in _NOTATION_TOKENS.items():
        lines.append(f"  {k}")
        lines.append(f"      {v}")
    lines += ["", "EXAMPLES", ""]
    for ex in _NOTATION_EXAMPLES:
        lines.append(f"  {ex['pattern']}")
        lines.append(f"      {ex['description']}")
    lines += ["", "Try one: plume pattern set TRACK NAME \"...\" "
              "(it validates and reports the step count)"]
    return doc, "\n".join(lines), 0


# ---------------------------------------------------------------- commands
# Each handler returns (payload_for_json, human_text, exit_code).

def _write_session_and_summarize(session, name_override: str | None, seed,
                                 out: str | None = None):
    score = _import("plume.score")
    if name_override:
        session.name = name_override
    if out:
        path = Path(out)
        parent = path.parent
        if not parent.is_dir():
            raise CliError(f"cannot write session: directory does not exist: {parent}")
    else:
        path = Path.cwd() / (_safe_filename(session.name) + SESSION_SUFFIX)
    score.save(session, path)
    doc = _read_doc(path)
    h = state_hash(doc)
    payload = {"path": str(path), "hash": h, "summary": _summary_dict(doc)}
    human = _human_summary(doc, path=path) + f"\nhash: {h}"
    return payload, human, 0


def cmd_make(args):
    compose_mod = _import("plume.compose")
    spec = compose_mod.parse_prompt(args.prompt)
    if args.name:
        spec.name = args.name
    session = compose_mod.compose(spec, seed=args.seed)
    rep, note = prompt_report(args.prompt)
    if note and not getattr(args, "json", False):
        print(note, file=sys.stderr)
    payload, human, code = _write_session_and_summarize(
        session, args.name, args.seed, out=getattr(args, "session", None))
    if rep is not None and isinstance(payload, dict):
        payload["prompt"] = {"text": args.prompt, **rep}
    return payload, human, code


def cmd_new(args):
    compose_mod = _import("plume.compose")
    styles_lib = _import("plume.styles_lib")
    known = styles_lib.list_styles()
    if args.style not in known:
        raise CliError(f"unknown style {args.style!r} (styles: {', '.join(known)})")
    if args.key is not None:
        theory = _import("plume.theory")
        try:
            theory.parse_key(args.key)
        except ValueError as e:
            raise CliError(str(e)) from None
    spec = compose_mod.PromptSpec(
        styles=[args.style],
        bpm=args.bpm,
        key=args.key,
        movements=args.movements,
        mood=[],
        name=args.name,
    )
    session = compose_mod.compose(spec, seed=args.seed)
    return _write_session_and_summarize(session, args.name, args.seed,
                                        out=getattr(args, "session", None))


def cmd_show(args):
    path = resolve_session(args.session)
    doc = _read_doc(path)
    payload = {"path": str(path), "hash": state_hash(doc),
               "summary": _summary_dict(doc), "document": doc}
    human = _human_summary(doc, path=path, only_movement=args.movement)
    if args.events:
        engine = _import("plume.engine")
        session = _load_session(path)
        events = engine.compile_events(session)
        per_track: dict[str, int] = {}
        for ev in events:
            per_track[ev.track] = per_track.get(ev.track, 0) + 1
        stats = {
            "count": len(events),
            "per_track": per_track,
            "t_first_s": round(min((ev.t_s for ev in events), default=0.0), 4),
            "t_last_s": round(max((ev.t_s for ev in events), default=0.0), 4),
        }
        payload["events"] = stats
        human += (
            f"\nevents: {stats['count']}"
            f" (t {stats['t_first_s']}s..{stats['t_last_s']}s)\n"
            + "\n".join(f"  {k}: {v}" for k, v in sorted(per_track.items()))
        )
    return payload, human, 0


def cmd_set(args):
    path = resolve_session(args.session)
    doc = _read_doc(path)
    value = _parse_value(args.value)
    set_dotted(doc, args.key, value)
    _write_doc(path, doc)
    h = state_hash(doc)
    payload = {"hash": h, "key": args.key, "value": value}
    human = f"{args.key} = {json.dumps(value)}\nhash: {h}"
    return payload, human, 0


def cmd_get(args):
    path = resolve_session(args.session)
    doc = _read_doc(path)
    if args.list:
        prefix = args.path or ""
        rows = [(p, v) for p, v in _settable_paths(doc) if p.startswith(prefix)]
        if not rows and prefix:
            raise CliError(f"no settable paths under {prefix!r} "
                           "(try 'plume get --list' for everything)")
        payload = {"paths": [{"path": p, "value": v} for p, v in rows]}
        human = "\n".join(f"{p} = {json.dumps(v)}" for p, v in rows) or "(none)"
        return payload, human, 0
    if not args.path:
        return {"path": "", "value": doc}, json.dumps(doc, indent=2, sort_keys=True), 0
    try:
        value = _descend(doc, args.path)
    except CliError:
        known, default = _schema_default(doc, args.path)
        if not known:
            raise
        return ({"path": args.path, "value": default, "default": True},
                json.dumps(default, indent=2, sort_keys=True) + "  (default)", 0)
    return ({"path": args.path, "value": value},
            json.dumps(value, indent=2, sort_keys=True), 0)


def cmd_unset(args):
    path = resolve_session(args.session)
    doc = _read_doc(path)
    keys = args.path.split(".")
    if any(k == "" for k in keys):
        raise CliError(f"bad key path: {args.path!r}")
    removable = False
    if len(keys) == 2 and keys[0] in ("master", "timing_bias_ms"):
        removable = True
        parent = doc.get(keys[0]) or {}
        leaf = keys[1]
    elif (len(keys) == 4 and keys[0] == "tracks"
          and keys[2] in ("fx", "params", "patterns")):
        removable = True
        track = _resolve_list_item(doc.get("tracks", []), keys[1], "tracks")
        parent = track.get(keys[2]) or {}
        leaf = keys[3]
    if not removable:
        raise CliError(
            f"cannot unset {args.path!r}: only optional leaf keys are removable "
            "(master.*, timing_bias_ms.*, tracks.NAME.fx.*, tracks.NAME.params.*, "
            "tracks.NAME.patterns.*). Structural keys need 'plume track rm' / "
            "'plume arrange set'."
        )
    if not isinstance(parent, dict) or leaf not in parent:
        raise CliError(f"nothing to unset: {args.path!r} is not present")
    del parent[leaf]
    _write_doc(path, doc)
    h = state_hash(doc)
    return ({"hash": h, "removed": args.path},
            f"removed {args.path}\nhash: {h}", 0)


def cmd_track_add(args):
    path = resolve_session(args.session)
    doc = _read_doc(path)
    if any(t.get("name") == args.name for t in doc.get("tracks", [])):
        raise CliError(f"track {args.name!r} already exists")
    voices = _import("plume.voices")
    if args.voice not in voices.PRESETS:
        raise CliError(
            f"unknown voice {args.voice!r} (presets: {', '.join(sorted(voices.PRESETS))})"
        )
    kind = args.kind or ("drum" if args.voice in _DRUM_PRESETS else "tonal")
    track = {
        "name": args.name,
        "voice": args.voice,
        "kind": kind,
        "gain_db": args.gain_db,
        "pan": args.pan,
        "params": {},
        "fx": {},
        "patterns": {},
    }
    doc.setdefault("tracks", []).append(track)
    _write_doc(path, doc)
    h = state_hash(doc)
    payload = {"hash": h, "track": track}
    human = (
        f"added track {args.name} (voice {args.voice}, kind {kind}, "
        f"gain {args.gain_db} dB, pan {args.pan})\nhash: {h}"
    )
    return payload, human, 0


def cmd_track_rm(args):
    path = resolve_session(args.session)
    doc = _read_doc(path)
    _find_track(doc, args.name)  # raises if absent
    doc["tracks"] = [t for t in doc.get("tracks", []) if t.get("name") != args.name]
    _write_doc(path, doc)
    h = state_hash(doc)
    return {"hash": h, "removed": args.name}, f"removed track {args.name}\nhash: {h}", 0


def cmd_track_list(args):
    path = resolve_session(args.session)
    doc = _read_doc(path)
    tracks = [
        {
            "name": t.get("name"),
            "voice": t.get("voice"),
            "kind": t.get("kind"),
            "gain_db": t.get("gain_db", 0.0),
            "pan": t.get("pan", 0.0),
            "patterns": sorted(t.get("patterns", {})),
        }
        for t in doc.get("tracks", [])
    ]
    lines = [
        f"{t['name']:<12} {t['voice']:<10} {t['kind']:<6} "
        f"gain {t['gain_db']:>6} dB  pan {t['pan']:>5}  "
        f"patterns: {', '.join(t['patterns']) or '-'}"
        for t in tracks
    ] or ["(no tracks)"]
    return {"tracks": tracks}, "\n".join(lines), 0


def cmd_pattern_set(args):
    notation = _import("plume.notation")
    path = resolve_session(args.session)
    doc = _read_doc(path)
    track = _find_track(doc, args.track)
    try:
        steps = notation.parse(args.pattern)
    except notation.NotationError as e:
        raise CliError(f"bad pattern: {e} — run 'plume notation' for the grammar") from None
    track.setdefault("patterns", {})[args.name] = args.pattern
    _write_doc(path, doc)
    h = state_hash(doc)
    n = len(steps)
    payload = {"hash": h, "track": args.track, "name": args.name,
               "pattern": args.pattern, "steps": n}
    human = (
        f"tracks.{args.track}.patterns.{args.name} = {args.pattern!r} "
        f"({n} step{'s' if n != 1 else ''})\nhash: {h}"
    )
    return payload, human, 0


def cmd_pattern_get(args):
    path = resolve_session(args.session)
    doc = _read_doc(path)
    track = _find_track(doc, args.track)
    patterns = track.get("patterns", {})
    if args.name is not None:
        if args.name not in patterns:
            known = ", ".join(sorted(patterns)) or "(none)"
            raise CliError(
                f"track {args.track!r} has no pattern {args.name!r} (patterns: {known})"
            )
        payload = {"track": args.track, "name": args.name, "pattern": patterns[args.name]}
        return payload, patterns[args.name], 0
    payload = {"track": args.track, "patterns": patterns}
    lines = [f"{k}: {v}" for k, v in sorted(patterns.items())] or ["(no patterns)"]
    return payload, "\n".join(lines), 0


def cmd_arrange_show(args):
    path = resolve_session(args.session)
    doc = _read_doc(path)
    payload = {"movements": doc.get("movements", [])}
    lines = _movements_lines(doc) or ["(no movements)"]
    return payload, "\n".join(lines), 0


def cmd_arrange_set(args):
    path = resolve_session(args.session)
    doc = _read_doc(path)
    raw = args.spec_json
    if raw.startswith("@"):
        spec_path = Path(raw[1:])
        try:
            raw = spec_path.read_text(encoding="utf-8")
        except OSError as e:
            raise CliError(f"cannot read spec file {spec_path}: {e}") from None
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as e:
        raise CliError(f"--spec-json is not valid JSON: {e}") from None
    if isinstance(spec, dict) and "movements" in spec:
        movements = spec["movements"]
    else:
        movements = spec
    if not isinstance(movements, list):
        raise CliError("--spec-json must be a list of movements (or {\"movements\": [...]})"
                       " — see 'plume arrange set --help' for the schema")
    for i, m in enumerate(movements):
        if not isinstance(m, dict) or "name" not in m or not isinstance(m.get("sections"), list):
            raise CliError(f"movement {i} must be an object with 'name' and 'sections'"
                           " — see 'plume arrange set --help' for the schema")
        for j, s in enumerate(m["sections"]):
            missing = [k for k in ("name", "bars", "energy", "plays") if k not in s]
            if missing:
                raise CliError(
                    f"movement {i} section {j} missing keys: {', '.join(missing)}"
                    " — see 'plume arrange set --help' for the schema"
                )
            s.setdefault("chords", [])
            s.setdefault("fill", True)
    doc["movements"] = movements
    _write_doc(path, doc)
    h = state_hash(doc)
    human = "arrangement updated\n" + "\n".join(_movements_lines(doc)) + f"\nhash: {h}"
    return {"hash": h, "movements": movements}, human, 0


def cmd_render(args):
    engine = _import("plume.engine")
    path = resolve_session(args.session)
    session = _load_session(path)
    out = args.out
    if out is None:
        stem = path.name
        if stem.endswith(SESSION_SUFFIX):
            stem = stem[: -len(SESSION_SUFFIX)]
        out = str(path.parent / (stem + ".wav"))
    bars = tuple(args.bars) if args.bars else None
    manifest = engine.render(
        session,
        out_path=out,
        movement=args.movement,
        bars=bars,
        stems_dir=args.stems,
    )
    lines = [f"{k}: {manifest[k]}" for k in
             ("out", "duration_s", "bars", "events", "peak_db", "rms_db",
              "limiter_gain_reduction_db")
             if k in manifest]
    return dict(manifest), "\n".join(lines) or str(manifest), 0


def cmd_analyze(args):
    analyze_mod = _import("plume.analyze")
    if not Path(args.path).exists():
        raise CliError(f"file not found: {args.path}")
    bpm_hint = args.bpm_hint
    hint_source = "explicit" if bpm_hint is not None else None
    if bpm_hint is None:
        # default the hint to the session bpm when a session resolves
        # (precedence: explicit --bpm-hint > session bpm > none)
        try:
            sdoc = _read_doc(resolve_session(args.session))
            if isinstance(sdoc.get("bpm"), (int, float)):
                bpm_hint = float(sdoc["bpm"])
                hint_source = "session"
        except CliError:
            pass
    info = analyze_mod.analyze(args.path, bpm_hint=bpm_hint)
    info = dict(info)
    info["bpm_hint_used"] = bpm_hint
    info["bpm_hint_source"] = hint_source
    lines = []
    for k in sorted(info):
        v = info[k]
        if isinstance(v, float):
            v = round(v, 4)
        lines.append(f"{k}: {v}")
    return info, "\n".join(lines), 0


def cmd_play(args):
    engine = _import("plume.engine")
    path = resolve_session(args.session)
    session = _load_session(path)
    kwargs = {}
    if args.movement is not None:
        kwargs["movement"] = args.movement
    try:
        engine.play(session, **kwargs)
    except KeyboardInterrupt:      # Ctrl-C or plume stop: the stream has closed
        return {"played": str(path), "stopped": True}, f"stopped {path}", 130
    return {"played": str(path)}, f"played {path}", 0


def cmd_voices(args):
    voices = _import("plume.voices")
    info = voices.preset_info()
    lines = []
    for name in sorted(info):
        entry = info[name]
        desc = entry.get("description", "") if isinstance(entry, dict) else ""
        lines.append(f"{name:<10} {desc}")
    return dict(info), "\n".join(lines), 0


def cmd_styles(args):
    styles_lib = _import("plume.styles_lib")
    names = list(styles_lib.list_styles())
    return {"styles": names}, "\n".join(names) or "(no styles)", 0


def cmd_doctor(args):
    checks: list[dict] = []

    def add(name, status, detail):
        checks.append({"name": name, "status": status, "detail": detail})

    py_ok = sys.version_info >= (3, 11)
    add("python", "ok" if py_ok else "error",
        ".".join(str(v) for v in sys.version_info[:3]))

    import plume as _plume
    add("plume", "ok", _plume.__version__)

    for mod in ("numpy", "scipy", "soundfile"):
        try:
            m = importlib.import_module(mod)
            add(mod, "ok", getattr(m, "__version__", "?"))
        except Exception as e:  # noqa: BLE001
            add(mod, "error", f"import failed: {e}")

    # sounddevice is optional: absence is a warning, not an error.
    try:
        sd = importlib.import_module("sounddevice")
        add("sounddevice", "ok", getattr(sd, "__version__", "?"))
        try:
            devices = sd.query_devices()
            n_out = sum(1 for d in devices if d.get("max_output_channels", 0) > 0)
            if n_out:
                add("audio_device", "ok", f"{n_out} output device(s)")
            else:
                add("audio_device", "warn", "no audio output device found")
        except Exception as e:  # noqa: BLE001
            add("audio_device", "warn", f"device query failed: {e}")
    except Exception:  # noqa: BLE001
        add("sounddevice", "warn",
            "not installed (optional; pip install 'plume[play]')")
        add("audio_device", "warn", "unavailable without sounddevice")

    for mod in ("theory", "notation", "score", "styles_lib", "groove",
                "compose", "voices", "fx", "engine", "analyze"):
        try:
            importlib.import_module(f"plume.{mod}")
            add(f"plume.{mod}", "ok", "importable")
        except Exception as e:  # noqa: BLE001
            add(f"plume.{mod}", "error", f"import failed: {e}")

    # v1.5.2: which beat-tracker front end a session would get, and why.
    # With --probe FILE.wav the file's first seconds are measured (the same
    # discriminator `--input auto` uses on sim/listen); without it, the line
    # states the live-session rule, which cannot probe a device that is not
    # open yet.
    probe = getattr(args, "probe", None)
    if probe:
        if not Path(probe).exists():
            add("input_mode", "error", f"cannot probe: file not found: {probe}")
        else:
            try:
                sf = importlib.import_module("soundfile")
                sim_mod = _import("plume.sim")
                audio, psr = sf.read(str(probe), dtype="float32",
                                     always_2d=True)
                dec = sim_mod.classify_input(
                    audio.mean(axis=1), int(psr))
                detail = f"{dec.mode} — {dec.reason} (probed {probe})"
                if dec.hint:
                    detail += f"; hint: {dec.hint}"
                add("input_mode", "ok", detail)
            except Exception as e:  # noqa: BLE001 - bad/corrupt files
                add("input_mode", "warn", f"cannot probe {probe}: {e}")
    else:
        try:
            beat_mod = _import("plume.listen.beat")
            env_acoustic = beat_mod._acoustic_from_env()
        except Exception:  # noqa: BLE001  # pragma: no cover - defensive
            env_acoustic = False
        if env_acoustic:
            add("input_mode", "ok",
                "acoustic — PLUME_ACOUSTIC is set, so live sessions and "
                "'--input auto' select the acoustic front end")
        else:
            add("input_mode", "ok",
                "auto — offline verbs (sim/listen) probe the file's first "
                "seconds; live verbs (jam/lead) use 'programmed' unless "
                "--input acoustic is passed (a device cannot be probed "
                "before it opens). Probe a take with "
                "'plume doctor --probe TAKE.wav'")

    ok = not any(c["status"] == "error" for c in checks)
    payload = {"ok": ok, "checks": checks}
    mark = {"ok": "ok  ", "warn": "warn", "error": "ERR "}
    human = "\n".join(f"{mark[c['status']]}  {c['name']:<16} {c['detail']}" for c in checks)
    human += "\n" + ("all good" if ok else "problems found")
    return payload, human, 0 if ok else 1


# ---------------------------------------------------------------- v1: live mode
# sim / jam / listen / mind (SPEC-V1 section 8). All additive; none of these
# needs a session file. plume.{sim,realtime,mind,listen.*} import lazily so
# cli.py stays importable while those modules are being built.

# Meanings for every MindParams field (SPEC-V1 section 4). Only the PROSE
# lives here — every name, type and default in the help epilogs is read from
# the dataclass at runtime by `_mind_table()` below, exactly as `plume mind`
# does. A hand-written mirror of the defaults drifted once already (the table
# claimed entry_conf 0.55 long after the v1.2 amendment moved it to 0.45,
# while `plume mind` printed the true 0.45 — SPEC-V1 "v1.2 amendments"), so
# there is now no second copy of a number to go stale.
_MIND_PARAM_DOCS = {
    "min_listen_bars": "bars of listening required before the first entry",
    "entry_conf": "beat confidence required to arm entry",
    "entry_boundary_bars": "enter/change texture only at multiples of this many bars",
    "enter_under_db": "first-entry machine gain relative to the human's bar RMS (dB)",
    "ramp_bars": "bars to ramp from enter_under_db up to 0 dB",
    "density_high": "human onsets/beat that triggers thinning",
    "density_hysteresis": "come back when density < density_high - this (one-bar delay)",
    "silence_thin_bars": "human silent this many bars -> thin to skeleton",
    "silence_fade_bars": "human silent this many bars -> begin fading out",
    "silence_tacet_bars": "human silent this many bars -> fully out (tacet), entry re-armed",
    "silence_rms_db": "bar RMS below this counts as silent (dBFS)",
    "conf_floor": "beat confidence below this -> fade/tacet (beat-loss repair)",
    "register_margin_st": "keep voicings this many semitones clear of the human's register",
    "energy_alpha": "EMA coefficient for following the human's dynamic arc",
    "commit_bars": "once in, stay in at least this many bars (no ordinary exit)",
    "exit_conf": "beat confidence below this ends a COMMITTED arc (deep hysteresis)",
    "recommit_bars": "bars to sit out after an exit that broke the commitment",
    "harmony_commit_bars": "stable-key bars before committing to that key + the style progression (0 = follow tracker chords)",
    "harmony_release_bars": "consecutive different-key bars before modulating a committed key",
}

_MIND_TABLE_HEAD = "MindParams — every negotiation threshold (defaults = research values):"
_MIND_TABLE_FOOT = (
    "Set any field with --mind K=V (repeatable), e.g. --mind min_listen_bars=2 "
    "--mind enter_under_db=-10. Unknown keys error with the valid names.")


def _mind_rows() -> list[dict]:
    """[{name, type, default, meaning}] read from the real MindParams.

    Single source of truth for `plume mind`, `plume sim --help` and
    `plume mind --help`. Import failures are the caller's problem: the help
    epilogs degrade to a pointer (see `_mind_table`), `plume mind` raises.
    """
    import dataclasses

    defaults = _import("plume.mind").MindParams()
    rows = []
    for f in dataclasses.fields(type(defaults)):
        val = getattr(defaults, f.name)
        rows.append({"name": f.name, "type": type(val).__name__,
                     "default": val,
                     "meaning": _MIND_PARAM_DOCS.get(f.name, "")})
    return rows


def _mind_table() -> str:
    """The full parameter table for the --help epilogs.

    Built from the dataclass so a default can never disagree with the code.
    plume.mind imports lazily everywhere else in this file, so a table that
    cannot be built must not take `--help` down with it — it degrades to a
    one-line pointer instead.
    """
    try:
        rows = _mind_rows()
    except Exception:      # noqa: BLE001  # pragma: no cover - defensive
        # Broad on purpose: `--help` must work even while plume.mind is half
        # written. `plume mind` still raises the real error, so nothing is
        # hidden — it is just not hidden inside `plume sim --help`.
        return ("MindParams — run 'plume mind' for the full parameter table.\n\n"
                + _MIND_TABLE_FOOT)
    lines = [_MIND_TABLE_HEAD]
    for r in rows:
        lines.append(f"  {r['name']:<20} {r['type']:<6} {r['default']!s:<7}"
                     f"{r['meaning']}")
    return "\n".join(lines) + "\n\n" + _MIND_TABLE_FOOT


# ---- --input (front-end selection) — ONE table, four verbs ---------------
#
# sim / jam / listen / lead all take --input, and their help MUST be the same
# string built from this table: the v1 lesson about hand-written copies
# drifting (see _MIND_PARAM_DOCS above) applies to flags too, so there is no
# second copy of a mode name or its meaning anywhere.  test_cli_v2 asserts
# the four parsers share it verbatim.
_INPUT_MODE_TABLE = (
    ("auto", "measure the take's first seconds and pick (offline verbs probe "
             "the file; a live device cannot be probed before it opens, so "
             "live verbs keep 'programmed' unless told otherwise)"),
    ("acoustic", "sustained-material front end — fingerpicked/strummed "
                 "acoustic guitar, anything that rings between notes"),
    ("programmed", "the default flux front end the programmed corpus ships "
                   "with — drum machines, renders, clicks"),
)

_COUNT_IN_HELP = ("arm count-in tap detection: knock/clap/click 4 even counts "
                  "and the band hard-sets tempo + downbeat on your next beat "
                  "(the last count is beat 4 of the bar)")


def _input_mode_choices() -> tuple[str, ...]:
    return tuple(name for name, _ in _INPUT_MODE_TABLE)


def _input_mode_help() -> str:
    """The one --input help string (see _INPUT_MODE_TABLE)."""
    return ("beat-tracker front end: "
            + "; ".join(f"'{name}' = {doc}" for name, doc in _INPUT_MODE_TABLE)
            + ". Default: auto.")


def _add_midi_flag(p) -> None:
    p.add_argument("--midi", default=None, metavar="PORT",
                   help="listen to this MIDI port instead of the mic: part of its "
                        "name, or 'auto' if there is only one (run 'plume midi'; "
                        "needs the midi extra)")
    p.add_argument("--midi-clock", dest="midi_clock", action="store_true", default=False,
                   help="let that port's MIDI clock set the tempo and the bar line, "
                        "over what you play; without this a clock is ignored")


def _midi_args(args) -> dict:
    """What --midi and --midi-clock ask of a live verb, only if they were given."""
    if getattr(args, "midi_clock", False) and not getattr(args, "midi", None):
        raise CliError("--midi-clock needs --midi: a clock comes from a MIDI port")
    out = {}
    if getattr(args, "midi", None):
        out["midi"] = args.midi
    if getattr(args, "midi_clock", False):
        out["midi_clock"] = True
    return out


def _add_input_mode_flag(p) -> None:
    """Attach --input to a verb's parser.  dest is input_mode (sim/listen
    already use the positional name 'input' for the WAV); default None means
    'not given', which the live verbs need to keep the bare-invocation
    signature exactly v1 (and which every verb treats as 'auto')."""
    p.add_argument("--input", dest="input_mode", default=None,
                   choices=_input_mode_choices(), metavar="MODE",
                   help=_input_mode_help())


def _temp_env(overrides: dict):
    """Context manager: set/remove environment variables, restore on exit.

    The live verbs use it to reach BeatTracker construction sites inside
    realtime, which reads PLUME_ACOUSTIC / PLUME_COUNT_IN at construction
    (the escape hatch beat.py documents for paths with no flag of their own).
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        saved = {k: os.environ.get(k) for k in overrides}
        try:
            for k, v in overrides.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            yield
        finally:
            for k, old in saved.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old
    return _ctx()


def _live_input_extras(realtime_fn, input_mode: str | None,
                       count_in: bool) -> tuple[dict, dict, str | None]:
    """(extra kwargs, env overrides, stderr note) for jam/lead input wiring.

    Prefers passing ``input_mode`` / ``count_in`` straight to the transport
    when its signature has grown them (checked with _accepted_kwargs, so a
    future realtime picks them up with no CLI change); until then it falls
    back to the environment hatch that BeatTracker construction sites read.
    With neither flag given, everything is empty and the bare invocation
    reaches realtime with exactly the v1 signature.
    """
    extra: dict = {}
    env: dict = {}
    note: str | None = None
    if input_mode is None and not count_in:
        return extra, env, note
    beat_mod = _import("plume.listen.beat")
    acc = _accepted_kwargs(realtime_fn)

    def accepts(name: str) -> bool:
        return acc is None or name in acc

    if input_mode is not None:
        if accepts("input_mode"):
            extra["input_mode"] = input_mode
        elif input_mode == "acoustic":
            env[beat_mod.ACOUSTIC_ENV] = "1"
        elif input_mode == "programmed":
            env[beat_mod.ACOUSTIC_ENV] = "0"
        if input_mode == "auto" and not beat_mod._acoustic_from_env():
            # auto and no explicit environment choice: a live device cannot
            # be measured before it opens (the transport has no live probe
            # yet either), so the guess would be blind — keep the programmed
            # default and SAY SO (a wrong silent guess is worse than a
            # hint).  This holds on BOTH wiring paths: kwargs pass "auto"
            # through and realtime leaves the front end at its default too.
            # A truthy PLUME_ACOUSTIC is an explicit user statement and auto
            # defers to it, exactly as sim.resolve_input_mode does offline.
            note = ("input auto: a live device cannot be probed before it "
                    "opens — using the programmed front end. Pass "
                    "--input acoustic for sustained acoustic material "
                    "(fingerpicked/strummed guitar).")
    if count_in:
        if accepts("count_in"):
            extra["count_in"] = True
        else:
            env[beat_mod.COUNT_IN_ENV] = "1"
    return extra, env, note


def _parse_taps(raw: str) -> list[float]:
    """--taps "0.5,1.0,1.5,2.0" -> [0.5, 1.0, 1.5, 2.0] (validated)."""
    try:
        taps = [float(t) for t in raw.replace(",", " ").split()]
    except ValueError:
        raise CliError(
            f"--taps must be comma-separated times in seconds, "
            f"e.g. \"0.5,1.0,1.5,2.0\" (got {raw!r})") from None
    if not taps:
        raise CliError("--taps is empty — give tap times in seconds")
    if any(t < 0 for t in taps) or sorted(taps) != taps:
        raise CliError("--taps times must be non-negative and increasing")
    return taps


def _parse_mind_overrides(pairs: list[str]):
    """--mind K=V pairs -> a plume.mind.MindParams (typed, validated)."""
    import dataclasses

    mind_mod = _import("plume.mind")
    defaults = mind_mod.MindParams()
    valid = [f.name for f in dataclasses.fields(mind_mod.MindParams)]
    kwargs = {}
    for raw in pairs:
        if "=" not in raw:
            raise CliError(f"--mind expects K=V, got {raw!r} "
                           "(run 'plume mind' for the parameter table)")
        k, v = raw.split("=", 1)
        k = k.strip()
        if k not in valid:
            raise CliError(f"unknown MindParams field {k!r} "
                           f"(valid: {', '.join(valid)})" + _dym(k, valid))
        cur = getattr(defaults, k)
        try:
            if isinstance(cur, bool):
                if v.strip().lower() not in ("0", "1", "true", "false"):
                    raise ValueError
                val = v.strip().lower() in ("1", "true")
            elif isinstance(cur, int):
                fv = float(v)
                if fv != int(fv):
                    raise ValueError
                val = int(fv)
            else:
                val = float(v)
        except ValueError:
            kind = ("an integer" if isinstance(cur, int)
                    and not isinstance(cur, bool) else "a number")
            raise CliError(f"--mind {k}: expected {kind}, got {v!r}") from None
        kwargs[k] = val
    return mind_mod.MindParams(**kwargs)


def _finite(x: float | None):
    """inf/nan -> None so state lines stay valid JSON."""
    if x is None:
        return None
    x = float(x)
    return x if (x == x and abs(x) != float("inf")) else None


def _rle(states: list[str]) -> str:
    """Run-length summary: ['a','a','b'] -> 'a×2 b'."""
    out = []
    for s in states:
        if out and out[-1][0] == s:
            out[-1][1] += 1
        else:
            out.append([s, 1])
    return " ".join(s if n == 1 else f"{s}×{n}" for s, n in out)


_TRACE_POINTS = 32          # --json bpm_trace samples unless --full-trace


def _decimate(seq, n: int):
    """Evenly sample at most ``n`` items, always keeping the first and last."""
    seq = list(seq)
    if len(seq) <= n or n < 2:
        return seq
    idx = sorted({round(i * (len(seq) - 1) / (n - 1)) for i in range(n)})
    return [seq[i] for i in idx]


def cmd_sim(args):
    if not Path(args.input).exists():
        raise CliError(f"file not found: {args.input}")
    taps = _parse_taps(args.taps) if args.taps else None
    params = _parse_mind_overrides(args.mind) if args.mind else None
    import dataclasses

    sim_mod = _import("plume.sim")
    try:
        result = sim_mod.simulate(
            args.input,
            style=args.style,
            out_path=args.out,
            seed=args.seed,
            params=params,
            tightness=args.tightness,
            mix_human_db=args.mix_human_db,
            taps_s=taps,
            bpm_hint=args.sim_bpm_hint,
            input_mode=getattr(args, "input_mode", None) or "auto",
        )
    except ValueError as e:
        # bad style / bad argument: clean one-line CLI error, no class name
        raise CliError(str(e)) from None
    decisions_path = result.decisions_path
    if args.decisions and str(args.decisions) != str(result.decisions_path):
        # simulate()'s frozen signature has no decisions-path parameter; the
        # CLI honors --decisions by relocating the file it wrote.
        try:
            os.replace(result.decisions_path, args.decisions)
        except OSError as e:
            raise CliError(f"cannot write decisions to {args.decisions}: {e}") from None
        decisions_path = args.decisions
    payload = dataclasses.asdict(result)
    payload["decisions_path"] = str(decisions_path)
    # v1.2: --json used to emit ~580 (t_s, bpm) pairs before the manifest, so
    # the interesting fields scrolled off the top of the terminal.  The trace
    # is decimated to at most _TRACE_POINTS evenly spaced samples (first and
    # last always kept, so the settled tempo is still visible) unless
    # --full-trace is given; the full trace is always in the NDJSON.
    if not args.full_trace:
        payload["bpm_trace"] = _decimate(result.bpm_trace, _TRACE_POINTS)
        payload["bpm_trace_decimated"] = (
            len(result.bpm_trace) > len(payload["bpm_trace"]))
        payload["bpm_trace_points_total"] = len(result.bpm_trace)
    lines = [
        f"out: {result.out_path}",
        f"decisions: {decisions_path}",
        f"duration_s: {round(result.duration_s, 3)}",
        f"entered_bar: {result.entered_bar}",
        f"final_key: {result.final_key}",
        f"states: {_rle(result.states_by_bar) or '(none)'}",
    ]
    # v1.5.2: which front end heard the take, and why — the selection exists
    # precisely because it used to be invisible and unreachable.
    if result.manifest.get("input_mode_resolved"):
        lines.append(f"input: {result.manifest['input_mode_resolved']} "
                     f"({result.manifest.get('input_mode_reason', '')})")
    if result.manifest.get("input_mode_hint"):
        lines.append(f"hint: {result.manifest['input_mode_hint']}")
    if result.entered_bar is None and result.manifest.get("no_entry_reason"):
        lines.append(f"why no duet: {result.manifest['no_entry_reason']}")
    return payload, "\n".join(lines), 0


def cmd_jam(args):
    realtime = _import("plume.realtime")
    # A malformed flag is a fact about the command line, not about the
    # machine, so it is reported before anything probes the hardware.
    max_minutes = _parse_max_minutes(getattr(args, "max_minutes", None))
    midi_extra = _midi_args(args)
    try:
        # Fail fast (and clean) on the no-device path before building the mind.
        realtime.check_device(args.device)
    except realtime.RealtimeError as e:
        raise CliError(str(e)) from None
    mind_mod = _import("plume.mind")
    params = (_parse_mind_overrides(args.mind) if args.mind
              else mind_mod.MindParams())
    # v2 (additive): --role / --roles / --lead-in are passed only when the user
    # actually gave one, so a bare `plume jam` calls exactly the v1 signature.
    extra = {}
    role_given = getattr(args, "role", None) or getattr(args, "roles", None)
    lead_in = getattr(args, "lead_in", None)
    if role_given:
        extra["role"] = _parse_role_overrides(getattr(args, "roles", None),
                                              getattr(args, "role", None))
    if lead_in:
        extra["lead_in"] = lead_in
    if getattr(args, "socket", None):
        extra["socket_path"] = args.socket
    # Only when the flag was actually given: a bare `plume jam` must still
    # reach realtime.jam with exactly the v1 signature (the cap is applied
    # inside realtime.jam, which defaults it to DEFAULT_MAX_MINUTES, so
    # "not passed" and "passed the default" behave identically).
    if max_minutes is not None:
        extra["max_minutes"] = max_minutes
    # v1.5.2: --input / --count-in.  Passed as kwargs when the transport has
    # grown them, else via the environment hatch BeatTracker construction
    # sites read; with neither flag, nothing here changes at all.
    in_extra, in_env, in_note = _live_input_extras(
        realtime.jam, getattr(args, "input_mode", None),
        bool(getattr(args, "count_in", False)))
    extra.update(in_extra)
    # v2.5: only when the flag was given — a bare `plume jam` still reaches
    # realtime.jam without the kwarg, and resolution happens there.
    if getattr(args, "latency_comp", None) is not None:
        extra["latency_comp_s"] = args.latency_comp / 1000.0
    extra.update(midi_extra)
    if in_note and not getattr(args, "json", False):
        print(in_note, file=sys.stderr)
    try:
        with _temp_env(in_env):
            realtime.jam(
                style=args.style,
                seed=args.seed,
                params=params,
                device=args.device,
                monitor=args.monitor,
                record_dir=args.record,
                tightness=args.tightness,
                **extra,
                **_route_kwargs(args),
            )
    except (realtime.RealtimeError, ValueError) as e:
        # RealtimeError: no device / stream trouble; ValueError: bad style.
        # Both surface as a clean one-line CLI error without the class name.
        raise CliError(str(e)) from None
    payload = {"ended": True, "record_dir": args.record}
    human = "jam ended" + (f" — recorded to {args.record}" if args.record else "")
    return payload, human, 0


def cmd_calibrate(args):
    realtime = _import("plume.realtime")
    with realtime.sigint_stops_this():
        realtime.register_session(kind="calibrate")   # so `plume stop` can find it
        try:
            realtime.check_device(args.device)
            cal = _import("plume.calibrate")
            res = cal.run(device=args.device)
        except KeyboardInterrupt:  # Ctrl-C or plume stop: the stream has closed
            return {"ok": False, "stopped": True}, "calibrate: stopped. Nothing saved.", 130
        except realtime.RealtimeError as e:
            raise CliError(str(e)) from None
        finally:
            realtime.unregister_session()
    if not res["ok"]:
        return ({"ok": False, **res},
                "calibrate: none of the clicks came back through the "
                "microphone — speakers muted, volume at zero, or the mic "
                "cannot hear them. Nothing saved.", 1)
    # The measurement cost a click train; an unwritable store must not
    # discard it. Report the number either way, and how to use it by hand.
    try:
        path = cal.save_roundtrip(res["input"], res["output"], res["sr"], res)
        save_err = None
    except OSError as e:
        path, save_err = None, e
    payload = {**res, "saved_to": None if path is None else str(path)}
    lines = [
        f"round trip: {res['roundtrip_ms']} ms "
        f"({res['heard']}/{res['clicks']} clicks, spread {res['spread_ms']} ms)",
        f"devices: {res['input']} -> {res['output']}",
    ]
    if path is not None:
        lines.append(f"saved: {path} — live sessions on this pair now place "
                     f"bars {res['roundtrip_ms']} ms early")
    else:
        lines.append(f"NOT saved ({save_err}) — pass it by hand: "
                     f"plume jam --latency-comp {res['roundtrip_ms']}")
    if res["spread_ms"] is not None and res["spread_ms"] > cal.SPREAD_WARN_MS:
        lines.append(f"warning: clicks disagree by {res['spread_ms']} ms — "
                     "an echoey room or a resampling stack; re-run, and "
                     "treat the number with suspicion")
    if res.get("reported_ms") is not None:
        lines.append(f"(the stream itself claimed {res['reported_ms']} ms)")
    return payload, "\n".join(lines), 0


def cmd_listen(args):
    if not Path(args.input).exists():
        raise CliError(f"file not found: {args.input}")
    listen_features = _import("plume.listen.features")
    listen_beat = _import("plume.listen.beat")
    listen_harmony = _import("plume.listen.harmony")
    sf = _import("soundfile")
    np = _import("numpy")

    try:
        audio, sr = sf.read(args.input, dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001 — bad/corrupt files raise all sorts
        raise CliError(f"cannot read {args.input}: {e}") from None
    mono = audio.mean(axis=1).astype(np.float32)

    # v1.5.2: front-end selection (one rule with sim's — sim.resolve_input_mode
    # is the single implementation of what --input means offline).
    sim_mod = _import("plume.sim")
    try:
        acoustic, mode_fields = sim_mod.resolve_input_mode(
            getattr(args, "input_mode", None) or "auto", mono, int(sr))
    except ValueError as e:
        raise CliError(str(e)) from None

    extractor = listen_features.FeatureExtractor(sr=int(sr))
    tracker = listen_beat.BeatTracker(acoustic=acoustic)
    harmony = listen_harmony.HarmonyTracker()

    states: list[dict] = []
    events: list[dict] = []
    stream_lines: list[str] = []
    # The selection is an event like any other tracker fact: first line of the
    # stream, so a human tailing the output knows which front end is hearing
    # the take before the first state arrives.
    mode_ev = {"event": "input_mode",
               "mode": mode_fields["input_mode_resolved"],
               "reason": mode_fields["input_mode_reason"]}
    events.append(mode_ev)
    stream_lines.append(json.dumps(mode_ev, sort_keys=True))
    if mode_fields.get("input_mode_hint"):
        print(f"hint: {mode_fields['input_mode_hint']}", file=sys.stderr)
    prev_locked = False
    prev_key = None
    prev_octaves = 0
    prev_grids = 0
    next_emit = 0.5

    def _snapshot(beat, harm):
        return {
            "t_s": round(beat.t_s, 4),
            "bpm": round(beat.bpm, 2),
            "conf": round(beat.confidence, 4),
            "locked": beat.locked,
            "bar": beat.bar,
            "beat_in_bar": beat.beat_in_bar,
            "phase_beats": round(beat.phase_beats, 4),
            "next_beat_t_s": _finite(beat.next_beat_t_s),
            "next_downbeat_t_s": _finite(beat.next_downbeat_t_s),
            "key": harm.key,
            "key_conf": round(harm.key_conf, 4),
            "chord": harm.chord,
            "chord_conf": round(harm.chord_conf, 4),
        }

    def _consume(frames, final=False):
        nonlocal prev_locked, prev_key, prev_octaves, prev_grids, next_emit
        if not frames and not final:
            return
        prev_bpm = states[-1]["bpm"] if states else 0.0
        beat = tracker.process(frames)
        harm = harmony.process(frames)
        if beat.locked != prev_locked:
            ev = {"event": "lock" if beat.locked else "unlock",
                  "t_s": round(beat.t_s, 4)}
            events.append(ev)
            stream_lines.append(json.dumps(ev, sort_keys=True))
            prev_locked = beat.locked
        # v1.3: a locked tracker may re-interpret the metre, but never
        # invisibly (see listen.beat._grid_hysteresis).  `plume listen` is
        # where a human looks when a take tracked wrong, so the change has to
        # be here and not only in the sim decisions log.
        # v2.7: key off the WIDER counter.  On live acoustic the octave is
        # 4% of the re-reads, so a log keyed on tempo_octave_changes alone
        # showed 2 of the 54 the take actually contained.
        if beat.tempo_grid_changes != prev_grids:
            octave = beat.tempo_octave_changes != prev_octaves
            ev = {"event": ("tempo_octave_change" if octave
                            else "tempo_grid_change"),
                  "t_s": round(beat.t_s, 4),
                  "from_bpm": round(prev_bpm, 2), "to_bpm": round(beat.bpm, 2),
                  "conf": round(beat.confidence, 4)}
            events.append(ev)
            stream_lines.append(json.dumps(ev, sort_keys=True))
            prev_octaves = beat.tempo_octave_changes
            prev_grids = beat.tempo_grid_changes
        if harm.key != prev_key:
            ev = {"event": "key_change", "t_s": round(beat.t_s, 4),
                  "from": prev_key, "to": harm.key}
            events.append(ev)
            stream_lines.append(json.dumps(ev, sort_keys=True))
            prev_key = harm.key
        if beat.t_s >= next_emit or final:
            snap = _snapshot(beat, harm)
            states.append(snap)
            stream_lines.append(json.dumps(snap, sort_keys=True))
            while next_emit <= beat.t_s:
                next_emit += 0.5

    block = 4096
    for i in range(0, mono.shape[0], block):
        _consume(extractor.process(mono[i:i + block]))
    _consume(extractor.flush(), final=True)

    payload = {"path": args.input, "sr": int(sr),
               "duration_s": round(mono.shape[0] / float(sr), 4),
               "input_mode": mode_fields,
               "states": states, "events": events}
    return payload, "\n".join(stream_lines), 0


def cmd_mind(args):
    rows = _mind_rows()
    payload = {"params": rows,
               "usage": ("override with --mind K=V (repeatable) on "
                         "'plume sim' / 'plume jam' / 'plume simlead' / "
                         "'plume lead'")}
    lines = ["MindParams — every negotiation threshold "
             "(--mind K=V on sim/jam/simlead/lead overrides any of them):", ""]
    for r in rows:
        lines.append(f"  {r['name']:<20} {r['type']:<6} {r['default']!s:>7}   "
                     f"{r['meaning']}")
    lines += ["", "example: plume sim in.wav --mind min_listen_bars=2 "
              "--mind enter_under_db=-10"]
    return payload, "\n".join(lines), 0


# =====================================================================
# v2 — lead mode (SPEC-V2 section 5): roles / simlead / lead / cue.
#
# Everything below is ADDITIVE.  No v0/v1 command, flag, default, output
# line or error string above this point is touched; `jam` gains two
# optional flags whose absence is exactly the v1 invocation.
#
# plume.roles / plume.sim / plume.realtime are imported LAZILY inside the
# command bodies (and defensively inside the help-table builders), so this
# module stays importable while its siblings are still being built.
# =====================================================================

_ROLES_TABLE_HEAD = ("RoleParams — who holds time and form, and what makes "
                     "that change (defaults = SPEC-V2 values):")
_ROLES_TABLE_FOOT = (
    "Set any field with --roles K=V (repeatable), e.g. --roles concede_ms=18 "
    "--roles trade_bars=8. --role R is shorthand for --roles role=R. "
    "clock_authority/form_authority accept 'none' to mean the role's default. "
    "Unknown keys error with the valid names.\n"
    "NOTE the table's 'role' default is RoleParams' own ('follow'). The VERBS "
    "override it: 'plume simlead' and 'plume lead' default to --role lead — "
    "they are named for the chair the machine sits in. 'plume jam' passes no "
    "role at all, which is the v1 sideman.")


def _role_rows() -> list[dict]:
    """[{name, type, default, meaning}] read from the real RoleParams.

    Single source of truth for `plume roles`, `plume simlead --help`,
    `plume lead --help` and `plume jam --help` — v1 learned this the hard way
    with a hand-written MindParams table that drifted from the defaults, so
    the table is DERIVED from ``dataclasses.fields`` and the meanings come
    from ``roles.PARAM_HELP`` (whose coverage `plume roles` asserts).
    """
    import dataclasses

    roles_mod = _import("plume.roles")
    defaults = roles_mod.RoleParams()
    help_map = dict(getattr(roles_mod, "PARAM_HELP", {}))
    rows = []
    for f in dataclasses.fields(type(defaults)):
        val = getattr(defaults, f.name)
        rows.append({"name": f.name,
                     "type": _type_name(f.type),
                     "default": val,
                     "meaning": help_map.get(f.name, "")})
    return rows


def _type_name(ftype) -> str:
    """Readable type for the table: 'float', 'int', 'bool', 'str', 'float?'."""
    t = (ftype if isinstance(ftype, str)
         else getattr(ftype, "__name__", str(ftype)))
    t = t.replace(" ", "")
    optional = t.endswith("|None") or t.startswith("Optional[")
    base = (t.removesuffix("|None").removeprefix("Optional[").removesuffix("]")
            or "str")
    return base + ("?" if optional else "")


def _roles_table() -> str:
    """The full RoleParams table for the --help epilogs.

    Degrades to a one-line pointer rather than taking `--help` down while
    plume.roles is half written (same discipline as ``_mind_table``).
    """
    try:
        rows = _role_rows()
    except Exception:      # noqa: BLE001  # pragma: no cover - defensive
        return ("RoleParams — run 'plume roles' for the full parameter "
                "table.\n\n" + _ROLES_TABLE_FOOT)
    lines = [_ROLES_TABLE_HEAD]
    for r in rows:
        lines.append(f"  {r['name']:<24} {r['type']:<7} "
                     f"{r['default']!s:<7}{r['meaning']}")
    return "\n".join(lines) + "\n\n" + _ROLES_TABLE_FOOT


_ROLE_NONE_WORDS = ("none", "null", "default", "auto", "")


def _coerce_role_value(name: str, type_name: str, raw: str):
    """One --roles K=V value, typed from the dataclass's own annotation."""
    optional = type_name.endswith("?")
    base = type_name.removesuffix("?")
    txt = raw.strip()
    if optional and txt.lower() in _ROLE_NONE_WORDS:
        return None
    if base == "str":
        return txt
    if base == "bool":
        if txt.lower() not in ("0", "1", "true", "false", "yes", "no"):
            raise CliError(f"--roles {name}: expected true or false, got {raw!r}")
        return txt.lower() in ("1", "true", "yes")
    try:
        fv = float(txt)
    except ValueError:
        kind = "an integer" if base == "int" else "a number"
        extra = " (or 'none')" if optional else ""
        raise CliError(f"--roles {name}: expected {kind}{extra}, "
                       f"got {raw!r}") from None
    if base == "int":
        if fv != int(fv):
            raise CliError(f"--roles {name}: expected an integer, got {raw!r}")
        return int(fv)
    return fv


def _parse_role_overrides(pairs: list[str] | None, role: str | None = None):
    """--role R + --roles K=V pairs -> a plume.roles.RoleParams (validated).

    ``--roles role=...`` wins over ``--role`` — the explicit table is the more
    specific statement, and silently ignoring one of two flags that name the
    same field is exactly the kind of drift this file avoids elsewhere.
    """
    roles_mod = _import("plume.roles")
    rows = {r["name"]: r for r in _role_rows()}
    valid = list(rows)
    kwargs = {}
    for raw in pairs or []:
        if "=" not in raw:
            raise CliError(f"--roles expects K=V, got {raw!r} "
                           "(run 'plume roles' for the parameter table)")
        k, v = raw.split("=", 1)
        k = k.strip()
        if k not in rows:
            raise CliError(f"unknown RoleParams field {k!r} "
                           f"(valid: {', '.join(valid)})" + _dym(k, valid))
        kwargs[k] = _coerce_role_value(k, rows[k]["type"], v)
    if role is not None and "role" not in kwargs:
        kwargs["role"] = role
    want = kwargs.get("role")
    if want is not None and want not in roles_mod.ROLES:
        raise CliError(f"unknown role {want!r} "
                       f"(valid: {', '.join(roles_mod.ROLES)})"
                       + _dym(want, roles_mod.ROLES))
    try:
        return roles_mod.RoleParams(**kwargs)
    except ValueError as e:
        raise CliError(str(e)) from None


def _parse_cue_arg(raw: str) -> tuple[float, str, str]:
    """``--cue "8.0:darker"`` -> ``(8.0, "darker", "next-phrase")``.

    ``T:TEXT@BOUNDARY`` sets ``Cue.at``; the ``@`` suffix is only read as a
    boundary when it actually names one, so a cue whose text contains an '@'
    is still just text.
    """
    if ":" not in str(raw):
        raise CliError(f"--cue expects T:TEXT (seconds, then the cue), "
                       f"got {raw!r} — e.g. --cue \"8.0:darker\"")
    t_raw, text = str(raw).split(":", 1)
    try:
        t = float(t_raw)
    except ValueError:
        raise CliError(f"--cue time must be seconds, got {t_raw!r} "
                       f"in {raw!r}") from None
    if t < 0.0:
        raise CliError(f"--cue time must be >= 0, got {t}")
    at = "next-phrase"
    if "@" in text:
        head, _, tail = text.rpartition("@")
        if head and tail in _cue_boundaries():
            text, at = head, tail
        elif head and not head[-1].isspace() and _looks_like_a_boundary(tail):
            # v2.1: an unrecognised boundary is an ERROR, not a default.
            # MEASURED: `--cue "4:darker@whenever"` exited 0 and logged
            # `cue 'darker@whenever' (texture:darker): applied` — the bogus
            # boundary was absorbed into the cue TEXT and the cue silently ran
            # at the default.  `plume cue --at` has always validated this; the
            # two surfaces now agree.  A tail that could not be meant as a
            # boundary (it has a space in it, or it is an email address) is
            # still just text, which is what the rpartition rule is for.
            raise CliError(
                f"--cue boundary {tail!r} is not one of "
                f"{', '.join(_cue_boundaries())} (in {raw!r})")
    text = text.strip()
    if not text:
        raise CliError(f"--cue has no text: {raw!r}")
    return t, text, at


#: Hard ceiling on ``--human-offset``, in seconds (one hour).
MAX_HUMAN_OFFSET_S = 3600.0


def _check_human_offset(args) -> None:
    """``--human-offset`` is padding, and unbounded padding fills the disk.

    MEASURED: ``plume simlead human.wav --bars 8 --human-offset 9999`` exited
    0, reported ``duration_s: 10029.3`` and wrote a **2,888,438,444-byte**
    (2.89 GB, 2h47m) WAV — from a request for eight bars, about twenty
    seconds, of music.  One typo'd flag.  The offset is also checked against
    the take the machine is actually going to play, because an offset past the
    end of it means the human is never heard at all, which is a mistake rather
    than a plan and is worth saying out loud.
    """
    off = float(getattr(args, "human_offset", 0.0) or 0.0)
    if not math.isfinite(off) or off < 0.0:
        raise CliError(f"--human-offset must be a finite number of seconds "
                       f">= 0, got {off!r}")
    if off > MAX_HUMAN_OFFSET_S:
        raise CliError(
            f"--human-offset {off:g}s is longer than {MAX_HUMAN_OFFSET_S:g}s "
            f"(one hour). The offset pads the human's audio with silence, so "
            f"the take — and the WAV on disk — is at least that long: "
            f"{off:g}s at 48kHz stereo 24-bit is about "
            f"{off * 48000 * 6 / 1e9:.1f} GB. Did you mean bars, not seconds?")


def _looks_like_a_boundary(tail: str) -> bool:
    """Was this ``@suffix`` MEANT as a boundary?

    A single bare word with no spaces, at the end of a cue, in the position the
    help documents a boundary in — and (checked by the caller) with no space
    before the ``@``.  ``"darker@whenever"`` is a boundary someone got wrong;
    ``"email me @home"`` is prose that happens to contain an ``@``, and prose
    must stay prose.
    """
    t = str(tail).strip()
    return bool(t) and " " not in t and len(t) <= 24


def _cue_boundaries() -> tuple[str, ...]:
    try:
        return tuple(_import("plume.mind").CUE_BOUNDARIES)
    except Exception:      # noqa: BLE001 — mind may still be building
        return ("now", "next-bar", "next-phrase", "next-section")


def _accepted_kwargs(fn) -> set[str] | None:
    """Keyword names ``fn`` accepts, or None when it accepts anything (**kw).

    Lets an optional argument be offered to a sibling that may or may not have
    grown it yet, without a try/except TypeError that would also swallow a
    real TypeError from inside the callee.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):      # pragma: no cover - builtins etc.
        return None
    names = set()
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        if p.kind in (inspect.Parameter.KEYWORD_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD):
            names.add(p.name)
    return names


def _session_or_prompt(args, *, what: str):
    """(session_obj, prompt) for the v2 verbs: exactly one source, resolved.

    An EMPTY ``--prompt`` is rejected here rather than falling through to
    session auto-discovery: MEASURED, ``--prompt ""`` used to error with
    "multiple session files in cwd — pass --session PATH", which never
    mentions the prompt and points at the wrong problem entirely.
    """
    prompt = getattr(args, "prompt", None)
    explicit = getattr(args, "session", None)
    if prompt is not None and not str(prompt).strip():
        raise CliError(
            f"{what}: --prompt is empty. Describe the music you want "
            f"(e.g. --prompt 'chill lofi at 96 bpm'), or drop --prompt to use "
            f"a composed session file.")
    if prompt and explicit:
        raise CliError(f"{what}: give either --prompt or --session, not both")
    if prompt:
        return None, prompt
    if getattr(args, "detect", False):
        # Before discovery: a session file lying in this folder is not this take's.
        return None, None
    return _load_session(resolve_session(explicit)), None


def prompt_report(prompt: str | None) -> tuple[dict | None, str | None]:
    """``(machine-readable, one-line note)`` about what a prompt was understood
    to say, or ``(None, None)`` when there is no prompt.

    v2.1: the prompt path had no feedback channel at all, so a typo and a
    perfect prompt looked identical from outside (see
    :func:`plume.compose.describe_prompt`).  The note is printed on stderr —
    it is commentary on the request, not part of the result — and the dict
    rides along in ``--json``.
    """
    if not prompt:
        return None, None
    try:
        rep = _import("plume.compose").describe_prompt(str(prompt))
    except Exception:                    # noqa: BLE001 - never block a render
        return None, None
    bits = ", ".join(f"{k} {v}" for k, v in rep["understood"].items())
    if not rep["understood_any"]:
        note = (f"note: nothing in {prompt!r} was understood — composing with "
                f"the defaults. Try 'chill lofi at 96 bpm, 3 movements'.")
    elif rep["ignored"]:
        note = (f"note: understood {bits}; ignored "
                f"{', '.join(repr(w) for w in rep['ignored'])}")
    else:
        note = f"note: understood {bits}"
    return rep, note


def cmd_roles(args):
    rows = _role_rows()
    roles_mod = _import("plume.roles")
    payload = {
        "params": rows,
        "roles": {name: {"clock_authority": ca, "form_authority": fa}
                  for name, (ca, fa) in roles_mod.ROLE_DEFAULTS.items()},
        "usage": ("override with --roles K=V (repeatable) on 'plume simlead' / "
                  "'plume lead' / 'plume jam'; --role R is shorthand for "
                  "--roles role=R"),
    }
    lines = [_ROLES_TABLE_HEAD, ""]
    for r in rows:
        lines.append(f"  {r['name']:<24} {r['type']:<7} {r['default']!s:>7}   "
                     f"{r['meaning']}")
    lines += ["", "role defaults (clock_authority, form_authority) — "
                  "0 = the human owns it, 1 = the machine does:"]
    for name, (ca, fa) in roles_mod.ROLE_DEFAULTS.items():
        lines.append(f"  {name:<14} clock {ca:<5} form {fa}")
    lines += ["", "example: plume simlead take.wav --prompt 'chill trap at 120' "
              "--role lead --roles concede_ms=18"]
    return payload, "\n".join(lines), 0


def cmd_simlead(args):
    import dataclasses

    if args.input is not None and not Path(args.input).exists():
        raise CliError(f"file not found: {args.input}")
    session, prompt = _session_or_prompt(args, what="simlead")
    _check_human_offset(args)
    role = _parse_role_overrides(args.roles, args.role)
    mind_params = _parse_mind_overrides(args.mind) if args.mind else None
    cues = None
    if args.cue:
        mind_mod = _import("plume.mind")
        cues = []
        for raw in args.cue:
            t, text, at = _parse_cue_arg(raw)
            cues.append((t, mind_mod.Cue(text=text, at=at)))

    sim_mod = _import("plume.sim")
    if not hasattr(sim_mod, "simulate_lead"):
        raise CliError(
            "plume.sim has no simulate_lead() yet — 'plume simlead' needs the "
            "SPEC-V2 section 4 offline twin. 'plume sim' (the v1 sideman) "
            "works now.")
    kwargs = {
        "session": session, "prompt": prompt, "out_path": args.out,
        "seed": args.seed, "role": role, "mind": mind_params,
        "human_offset_s": args.human_offset, "cues": cues,
    }
    # The SPEC-V2 section 4 signature is the frozen part; sim's additive
    # keywords are offered only when it actually takes them, so this file
    # keeps working against both the spec signature and the built one.
    optional = {"style": args.style, "bars": args.bars,
                "mix_human_db": args.mix_human_db,
                "decisions_path": args.decisions}
    accepted = _accepted_kwargs(sim_mod.simulate_lead)
    for k, v in optional.items():
        if v is not None and (accepted is None or k in accepted):
            kwargs[k] = v
    relocate = (args.decisions is not None and "decisions_path" not in kwargs)
    try:
        result = sim_mod.simulate_lead(args.input, **kwargs)
    except ValueError as e:
        raise CliError(str(e)) from None

    decisions_path = getattr(result, "decisions_path", None)
    if relocate and decisions_path and str(args.decisions) != str(decisions_path):
        try:
            os.replace(decisions_path, args.decisions)
        except OSError as e:
            raise CliError(
                f"cannot write decisions to {args.decisions}: {e}") from None
        decisions_path = args.decisions

    if dataclasses.is_dataclass(result):
        payload = dataclasses.asdict(result)
    elif isinstance(result, dict):
        payload = dict(result)
    else:                       # any object with attributes: report them all
        payload = {k: v for k, v in vars(result).items()
                   if not k.startswith("_")}
    if decisions_path is not None:
        payload["decisions_path"] = str(decisions_path)
    trace = payload.get("bpm_trace")
    if trace and not args.full_trace:
        payload["bpm_trace"] = _decimate(trace, _TRACE_POINTS)
        payload["bpm_trace_decimated"] = len(trace) > len(payload["bpm_trace"])
        payload["bpm_trace_points_total"] = len(trace)
    lines = [f"out: {payload.get('out_path', args.out)}"]
    if decisions_path is not None:
        lines.append(f"decisions: {decisions_path}")
    if payload.get("duration_s") is not None:
        lines.append(f"duration_s: {round(float(payload['duration_s']), 3)}")
    if payload.get("final_key") is not None:
        lines.append(f"final_key: {payload['final_key']}")
    states = payload.get("states_by_bar")
    if states:
        lines.append(f"states: {_rle(list(states))}")
    for key in ("roles_by_bar", "role_by_bar"):
        if payload.get(key):
            lines.append(f"roles: {_rle([str(r) for r in payload[key]])}")
            break
    man = payload.get("manifest") or {}
    for key in ("tempo_offset_pct", "transpose_semitones", "concessions"):
        if isinstance(man, dict) and man.get(key) is not None:
            lines.append(f"{key}: {man[key]}")
    for n in (man.get("human_notes") or []):
        lines.append(f"note: {n}")
    rep, note = prompt_report(prompt)
    if rep is not None:
        payload["prompt"] = {"text": prompt, **rep}
        # With --json the note is IN the document; printing it on stderr too
        # would make the JSON surface two channels for one fact.
        if not getattr(args, "json", False):
            print(note, file=sys.stderr)
    return payload, "\n".join(lines), 0


def cmd_lead(args):
    realtime = _import("plume.realtime")
    # Argument errors before hardware probing (see cmd_jam).
    if args.detect:
        if not args.tap_start:
            raise CliError("--detect needs --tap-start: plume listens for what it detects")
        if getattr(args, "session", None):
            raise CliError("--detect plays from what it hears, not from a session: "
                           "drop --session or --detect")
        if args.style:
            raise CliError("--style is not read with --detect: "
                           "name the style in the prompt, or leave it out and plume will guess")
    max_minutes = _parse_max_minutes(getattr(args, "max_minutes", None))
    midi_extra = _midi_args(args)
    try:
        # Fail fast (and clean) on the no-device path, before composing.
        realtime.check_device(args.device)
    except realtime.RealtimeError as e:
        raise CliError(str(e)) from None
    if args.wind_down is not None and (
            not 0 < args.wind_down < float("inf") or not args.tap_start):
        raise CliError("--wind-down needs --tap-start and a time above 0")
    session, prompt = _session_or_prompt(args, what="lead")
    _rep, _note = prompt_report(prompt)
    if _note and not getattr(args, "json", False):
        print(_note, file=sys.stderr)
    role = _parse_role_overrides(args.roles, args.role)
    mind_mod = _import("plume.mind")
    params = (_parse_mind_overrides(args.mind) if args.mind
              else mind_mod.MindParams())
    if args.from_bar < 0:
        raise CliError(f"--from-bar must be >= 0, got {args.from_bar}")
    # v1.5.2: --input / --count-in, same wiring as cmd_jam.
    in_extra, in_env, in_note = _live_input_extras(
        realtime.lead, getattr(args, "input_mode", None),
        bool(getattr(args, "count_in", False)))
    if in_note and not getattr(args, "json", False):
        print(in_note, file=sys.stderr)
    try:
        with _temp_env(in_env):
            realtime.lead(
                max_minutes=max_minutes,
                session=session,
                prompt=prompt,
                style=args.style or "lofi",
                seed=args.seed,
                params=params,
                role=role,
                device=args.device,
                monitor=args.monitor,
                record_dir=args.record,
                tightness=args.tightness,
                from_bar=args.from_bar,
                socket_path=args.socket,
                **({"tap_start": True} if args.tap_start else {}),
                **({"detect": True} if args.detect else {}),
                **({} if args.wind_down is None else {"wind_down_s": args.wind_down}),
                **midi_extra,
                **in_extra,
                **_route_kwargs(args),
            )
    except (realtime.RealtimeError, ValueError) as e:
        raise CliError(str(e)) from None
    socket_path = args.socket or realtime.default_socket_path()
    payload = {"ended": True, "record_dir": args.record, "socket": socket_path}
    human = "lead ended" + (f" — recorded to {args.record}" if args.record else "")
    return payload, human, 0


def cmd_cue(args):
    realtime = _import("plume.realtime")
    boundaries = _cue_boundaries()
    if args.at not in boundaries:
        raise CliError(f"unknown --at {args.at!r} "
                       f"(valid: {', '.join(boundaries)})"
                       + _dym(args.at, boundaries))
    if args.ttl_bars < 1:
        raise CliError(f"--ttl-bars must be >= 1, got {args.ttl_bars}")
    parsed = None
    looped = _import("plume.looper").parse_loop_cue(args.text)
    try:
        if looped is None:
            parsed = _import("plume.mind").parse_cue(args.text)
    except CliError:
        pass
    try:
        resp = realtime.send_cue(args.text, at=args.at, ttl_bars=args.ttl_bars,
                                 socket_path=args.socket, source=args.source)
    except realtime.RealtimeError as e:
        raise CliError(str(e)) from None
    socket_path = args.socket or realtime.default_socket_path()
    payload = {"sent": True, "text": args.text, "at": args.at,
               "ttl_bars": args.ttl_bars, "socket": socket_path,
               "session": resp.get("session"), "queued": resp.get("queued"),
               "bar": resp.get("bar")}
    if parsed is not None:
        payload["parsed"] = parsed
    if looped is not None:
        # A loop cue is answered at once, by the looper, not queued for a bar.
        payload["loop"] = {k: v for k, v in resp.items()
                           if k not in ("ok", "queued", "session", "bar")}
        return payload, f"loop: {resp.get('said', 'done')}", 0
    kind = (parsed or {}).get("kind", "?")
    where = f" at bar {resp['bar']}" if resp.get("bar") is not None else ""
    human = (f"cued {args.text!r} ({kind}) for {args.at}{where} — "
             f"{resp.get('queued', '?')} queued")
    return payload, human, 0


# ------------------------------------------------- v2.2: sessions / stop
#
# The consent guard keeps a session from STARTING unattended.  These two keep
# a started one from becoming unattended: whatever is making noise can be
# named, and ended, from any terminal, without hunting for a PID.

_SESSION_COLUMNS = (
    # (header, record key, width, formatter)
    ("pid", "pid", 7, str),
    ("kind", "kind", 5, str),
    ("role", "role", 9, lambda v: v or "-"),
    ("style", "style", 11, lambda v: v or "-"),
    ("up", "uptime", 7, str),
    ("ends in", "remaining_s", 8,
     lambda v: "-" if v is None else _short_secs(v)),
    ("session", "session", 0, lambda v: v or "-"),
)


def _short_secs(seconds: float) -> str:
    realtime = _import("plume.realtime")
    return realtime.format_uptime(max(0.0, float(seconds)))


def _sessions_table(rows: list[dict]) -> str:
    """Render live sessions as a fixed-width table (columns above)."""
    head = "  ".join(h.ljust(w) if w else h for h, _, w, _ in _SESSION_COLUMNS)
    lines = [head.rstrip(), "  ".join(
        ("-" * len(h)).ljust(w) if w else "-" * len(h)
        for h, _, w, _ in _SESSION_COLUMNS).rstrip()]
    for r in rows:
        cells = []
        for _h, key, w, fmt in _SESSION_COLUMNS:
            text = str(fmt(r.get(key)))
            cells.append(text.ljust(w) if w else text)
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def _live_sessions(realtime) -> list:
    try:
        return realtime.list_sessions()
    except OSError as e:                       # pragma: no cover - defensive
        raise CliError(f"cannot read the session registry: {e}") from None


def cmd_voice(args):
    voice = _import("plume.voice")
    realtime = _import("plume.realtime")
    def write(line):
        print(line, flush=True)
    if args.wind_down is not None and not 0 < args.wind_down < float("inf"):
        raise CliError(f"--wind-down must be a time above 0, got {args.wind_down}")
    if args.direct and args.voice_channel is None:
        raise CliError("on a direct input the recording holds only the guitar, so nothing "
                       "you say would reach the model: add --voice-channel N with a mic "
                       "on that channel")
    extra = [] if args.wind_down is None else ["--wind-down", str(args.wind_down)]
    for flag, value in (("--device", args.device), ("--input-device", args.input_device),
                        ("--input-channel", args.input_channel),
                        ("--voice-channel", args.voice_channel)):
        if value is not None:
            extra += [flag, str(value)]
    if args.direct:
        extra.append("--direct")
    midi_args = _midi_args(args)
    try:
        realtime.check_audio_consent()
        # Here, not in the band: its error only reaches the band log, and only after
        # the model has loaded, which can be a download of minutes.
        realtime.resolve_input(realtime._import_sounddevice(), args.device,
                               input_device=args.input_device,
                               input_channel=args.input_channel,
                               voice_channel=args.voice_channel, direct=args.direct)
        if "midi" in midi_args:
            extra += ["--midi", realtime._resolve_midi_port(midi_args["midi"])]
        if "midi_clock" in midi_args:
            extra.append("--midi-clock")
        voice.run(prompt=args.prompt, base=args.base, extra_args=extra, write=write)
    except KeyboardInterrupt:
        pass                       # voice.run has already stopped the band
    except (realtime.RealtimeError, RuntimeError) as e:
        raise CliError(str(e)) from None
    return {"ended": True}, "", 0


def cmd_voice_replay(args):
    voice = _import("plume.voice")
    def write(line):
        print(line, flush=True)
    try:
        voice.replay(args.path, write=write)
    except (RuntimeError, OSError) as e:
        raise CliError(str(e)) from None
    return {"replayed": args.path}, "", 0


def cmd_inputs(args):
    realtime = _import("plume.realtime")
    try:
        sd = realtime._import_sounddevice()
        if args.meter is None:
            devices = realtime.input_devices(sd)
            rows = [f"{d['inputs']:>3} in {d['outputs']:>3} out  {d['name']}" for d in devices]
            return {"devices": devices}, "\n".join(rows), 0
        doc = realtime.meter_input(sd, args.meter, seconds=args.seconds)
    except realtime.RealtimeError as e:
        raise CliError(str(e)) from None
    rows = [f"channel {c['channel']}: peak {c['peak_db']:>6} dB  rms {c['rms_db']:>6} dB"
            for c in doc["channels"]]
    rows.append(f"loudest: channel {doc['loudest_channel']}")
    return doc, "\n".join(rows), 0


def _route_kwargs(args) -> dict:
    """The input route, only as far as it was asked for: a bare invocation must reach
    the transport with the signature it always had."""
    extra = {}
    for name in ("input_device", "input_channel", "voice_channel"):
        if getattr(args, name, None) is not None:
            extra[name] = getattr(args, name)
    if getattr(args, "direct", False):
        extra["direct"] = True
    return extra


def _add_route_flags(p, *, voice: bool = True) -> None:
    p.add_argument("--input-device", dest="input_device", default=None, metavar="NAME",
                   help="listen on this device and leave the output alone (name substring; "
                        "'plume inputs' lists them)")
    p.add_argument("--input-channel", dest="input_channel", type=int, default=None,
                   metavar="N", help="the instrument's channel on that device, counted from "
                                     "1 as the interface labels them (default: 1)")
    if voice:
        p.add_argument("--voice-channel", dest="voice_channel", type=int, default=None,
                       metavar="N", help="a mic on this channel of the same device, recorded "
                                         "to voice.wav; what 'plume voice' listens to")
    p.add_argument("--direct", action="store_true", default=False,
                   help="the band cannot reach this input (instrument plugged in, or "
                        "headphones): turns off self-masking and trusts its level. Thresholds "
                        "are from a room mic and uncalibrated here")


def cmd_midi(args):  # noqa: ARG001
    midi = _import("plume.midi")
    try:
        ports = midi.list_ports()
    except midi.MidiError as e:
        raise CliError(str(e)) from None
    human = "\n".join(ports) if ports else "no MIDI input ports found"
    return {"ports": ports}, human, 0


def cmd_watch(args):
    watch = _import("plume.watch")
    def write(line):
        print(line, flush=True)
    try:
        watch.follow(args.record_dir, every=args.every,
                     changes_only=args.changes_only, write=write)
    except KeyboardInterrupt:      # Ctrl-C is how you leave a watch, not a failure
        pass
    return {"ended": True}, "", 0


def cmd_gui(args):
    gui = _import("plume.gui")
    try:
        if args.action == "url":
            url = gui.fresh_url()
            return {"url": url}, url, 0
        gui.serve(take=args.take, takes=args.takes, open_browser=not args.no_open)
    except gui.GuiError as e:
        raise CliError(str(e)) from None
    return {"ended": True}, "plume gui closed.", 0


def cmd_sessions(args):
    import time as _time  # noqa: PLC0415

    realtime = _import("plume.realtime")
    now = _time.time()
    recs = _live_sessions(realtime)
    rows = [r.describe(now=now) for r in recs]
    payload = {"sessions": rows, "count": len(rows),
               "registry": str(realtime.session_dir())}
    if not rows:
        return payload, ("no live plume sessions are running "
                         f"(registry: {realtime.session_dir()})"), 0
    tail = ("\nstop one:  plume stop --pid PID"
            "\nstop all:  plume stop --all")
    if len(rows) == 1:
        tail = f"\nstop it:   {rows[0]['stop_command']}"
    return payload, _sessions_table(rows) + "\n" + tail, 0


def cmd_stop(args):
    import time as _time  # noqa: PLC0415

    realtime = _import("plume.realtime")
    now = _time.time()
    recs = _live_sessions(realtime)
    if args.pid is not None:
        targets = [r for r in recs if r.pid == args.pid]
        if not targets:
            known = ", ".join(str(r.pid) for r in recs) or "none"
            raise CliError(f"no live plume session with pid {args.pid} "
                           f"(running: {known}) — 'plume sessions' lists them")
    elif not recs:
        targets = []
    elif len(recs) == 1 or args.all:
        targets = list(recs)
    else:
        listing = "\n".join(
            f"  pid {r.pid:<7} {r.kind:<5} {r.style or '-':<10} "
            f"up {realtime.format_uptime(r.uptime_s(now))}" for r in recs)
        raise CliError(
            f"{len(recs)} live sessions are running — say which:\n{listing}\n"
            "  plume stop --pid PID   (one)\n"
            "  plume stop --all       (all of them)")

    if not targets:
        # Nothing to stop is the normal, healthy answer to "is anything
        # playing?" — it exits 0 and says so.  A non-zero exit here would make
        # every wrapper script treat quiet as a failure.
        return ({"stopped": [], "count": 0, "failed": 0},
                "nothing to stop — no live plume sessions are running", 0)

    timeout = (realtime.DEFAULT_STOP_TIMEOUT if args.timeout is None
               else args.timeout)
    if timeout < 0:
        raise CliError(f"--timeout must be >= 0, got {args.timeout}")
    # --force adds SIGKILL as a last rung. It is not the default because it
    # cuts the tail; it is offered because the wedge message names it, and a
    # message that names a flag the parser rejects is worse than no advice.
    force = bool(getattr(args, "force", False))
    results = [realtime.stop_session(r, timeout=timeout, force=force)
               for r in targets]
    failed = [r for r in results if not r.get("stopped")]
    payload = {"stopped": results, "count": len(results), "failed": len(failed)}
    lines = []
    for r in results:
        what = f"{r['kind']} session, pid {r['pid']}, running {r['uptime']}"
        lines.append(("stopped: " if r.get("stopped") else "COULD NOT STOP: ")
                     + what)
        lines.append(f"         {r['detail']}")
        if r.get("record_dir") and r.get("stopped"):
            lines.append(f"         records: {r['record_dir']}")
    return payload, "\n".join(lines), (1 if failed else 0)


# ------------------------------------------- v2.2 help: the record schema

# Meaning-only, as with _MIND_PARAM_DOCS: every field name and default in the
# printed table is read from realtime.LiveSession at runtime, so a registry
# field can never be documented as something it is not.  Hand-written tables
# have drifted in this repo before (SPEC-V1 "v1.2 amendments").
_SESSION_FIELD_DOCS = {
    "pid": "process id — the one number 'plume stop --pid' needs",
    "kind": "jam (the sideman) or lead (the machine starts)",
    "style": "style the session is playing",
    "role": "role it started in (see 'plume roles')",
    "session_name": "session/composition name, when it came from one",
    "socket": "its control socket; 'plume stop' asks here first",
    "started_at": "epoch seconds — uptime is measured from this",
    "max_minutes": "self-stop cap in minutes (0 = no cap)",
    "record_dir": "where human.wav/machine.wav/decisions land, if recording",
}

_SESSIONS_EPILOG_BODY = """\
Every live session writes one JSON record while it runs, so a session started
in one terminal can be found from another. Records whose process is gone are
pruned on every read, silently — the registry is a cache of running
processes, not a document anyone edits.

Location: $PLUME_SESSION_DIR, else <tmpdir>/plume-sessions/<pid>.json

"""

_SESSIONS_EPILOG_FOOT = (
    "--json emits {\"sessions\": [...], \"count\": N, \"registry\": PATH} with "
    "every field above plus uptime/uptime_s/remaining_s/stop_command.\n"
    "Nothing running is not an error: both verbs exit 0 and say so.")


def _session_record_table() -> str:
    """The LiveSession field table for --help, read from the dataclass."""
    try:
        import dataclasses  # noqa: PLC0415

        rec = _import("plume.realtime").LiveSession
        rows = []
        for f in dataclasses.fields(rec):
            default = f.default if f.default is not dataclasses.MISSING else "-"
            rows.append((f.name, str(f.type), default))
    except Exception:      # noqa: BLE001  # pragma: no cover - defensive
        # --help must survive a half-written plume.realtime, same discipline
        # as _mind_table: degrade to a pointer, never take help down.
        return _SESSIONS_EPILOG_BODY + _SESSIONS_EPILOG_FOOT
    lines = ["Each record (fields read from realtime.LiveSession):"]
    for name, type_name, default in rows:
        lines.append(f"  {name:<14} {type_name:<16} {default!s:<10}"
                     f"{_SESSION_FIELD_DOCS.get(name, '')}")
    return (_SESSIONS_EPILOG_BODY + "\n".join(lines) + "\n\n"
            + _SESSIONS_EPILOG_FOOT)


_STOP_EPILOG = """\
Ends running live sessions from any terminal — no PID hunting.

  plume stop                 the one running session (errors if several)
  plume stop --pid 4212      that one
  plume stop --all           all of them
  plume sessions             what is running, and the command to stop it

How a session is stopped, in order, stopping as soon as it is gone:

  1. {"op": "stop"} on its own socket — the graceful path. It finishes the
     bar it is playing, flushes the tail, and writes its records, exactly as
     Ctrl-C does. The handshake also proves the pid really is plume.
  2. SIGINT — the same graceful ending, for a session whose socket never
     opened. It shares rung 1's budget rather than getting a fraction of it:
     a short fuse here does not fail safe, it lands SIGTERM in the middle of
     a working flush and throws the recording away.
  3. SIGTERM — the hard fallback, after --timeout has actually run out. The
     take's tail and records are probably lost, and the output says so.

plume play and plume calibrate have no ending to finish, so they get about 5
seconds, not --timeout, before SIGTERM, and they keep no records to lose.

SIGKILL is deliberately not in the default ladder: it would cut the tail. If
SIGTERM does not land, the output says so and names the command to run:

  plume stop --pid 4212 --force    add SIGKILL as a fourth rung

Forcing costs the tail and ending.json, not the recording — since v1.3 the
take is streamed to disk as it happens, so everything up to the last flush
survives.

Nothing running exits 0 with a plain message; a session that refuses to die
exits 1.
"""

_MAX_MINUTES_HELP = (
    "end the session after N minutes (default: {default:g}; "
    "0 disables the cap). A live session with nobody watching stops itself: "
    "the reason lands in the status line and in ending.json.")


def _max_minutes_help() -> str:
    try:
        default = _import("plume.realtime").DEFAULT_MAX_MINUTES
    except Exception:      # noqa: BLE001  # pragma: no cover - defensive
        return _MAX_MINUTES_HELP.replace("{default:g}", "60")
    return _MAX_MINUTES_HELP.format(default=float(default))


def _parse_max_minutes(raw):
    """--max-minutes -> float, or None when the flag was not given."""
    if raw is None:
        return None
    try:
        realtime = _import("plume.realtime")
        return realtime.resolve_max_minutes(raw)
    except ValueError as e:
        raise CliError(f"--max-minutes: {e}") from None


# ---------------------------------------------------------------- parser

_ARRANGE_SET_EPILOG = """\
--spec-json takes inline JSON or @FILE.json (contents read from disk).

Schema — a list of movements, or {"movements": [...]}:

  [{"name": "m1",
    "sections": [
      {"name": "intro", "bars": 8, "energy": 0.2,       # required
       "plays": {"kick": "main", "pad": "sparse"},      # required: track -> pattern
       "chords": ["Fm7", "Bbm7", "Eb7", "Abmaj7"],      # optional (default [])
       "fill": true}                                    # optional (default true)
    ]}]

Worked example:
  plume arrange set --spec-json '[{"name": "m1", "sections": [
    {"name": "intro", "bars": 8, "energy": 0.2, "plays": {"kick": "main"}},
    {"name": "drop", "bars": 16, "energy": 0.9, "plays": {"kick": "main"}}]}]'

'plume arrange show --json' outputs {"movements": [...]} in exactly this
shape — it round-trips straight back into 'arrange set --spec-json'."""

_ARRANGE_SHOW_EPILOG = """\
--json output ({"movements": [...]}) round-trips into
'plume arrange set --spec-json'."""

_RENDER_EPILOG = """\
Signal path: per-track fx -> sum -> master.gain_db -> reverb/drive/lofi ->
limiter at master.limit_db. The manifest reports peak_db, rms_db and
limiter_gain_reduction_db (>0 means the limiter engaged — turn gains or
master.gain_db down for a cleaner mix).
Stems (--stems DIR) are post-fx, pre-master, written with -3 dBFS headroom."""

_ANALYZE_EPILOG = """\
bpm hint precedence: explicit --bpm-hint > the resolved session's bpm > none.
The hint enables bar_rms_db (per-bar loudness) and biases the tempo estimate
toward the hinted metrical level; tempo_candidates lists ranked alternatives
including half/double-time relatives."""


_SIM_EPILOG_BODY = """\
The offline twin of 'plume jam': streams INPUT.wav through the identical
listen -> mind -> duet pipeline, deterministic and faster than real time.
Pipeline: mono-sum -> features -> beat/harmony tracking -> per-bar negotiation
(Mind) -> bar renders (DuetEngine) placed sample-accurately -> mix with the
human at --mix-human-db, soft-limited at -1 dB -> PCM_24 WAV. A decisions
NDJSON (one line per bar: state, reason, bpm, conf, key, chord, roles,
gain_db, machine_density; plus lock/unlock/key-change events) lands next to
the mix (or at --decisions). Byte-identical for identical (input, args).

machine_density is the density the MACHINE chose for that bar (0..1); the
human's measured density appears only inside 'reason' text, in onsets/beat.

--taps "0.5,1.0,1.5,2.0" simulates a count-in: >=4 roughly even taps hard-set
tempo + downbeat phase before the audio starts.

--json prints the SimResult manifest (paths, entered_bar, states_by_bar,
bpm_trace, final_key, peaks/rms).

"""

_JAM_EPILOG = """\
Live duplex jam (mic in, machine out) at 48 kHz / 256-sample blocks. The
audio callback only copies ring buffers; analysis (beat/key tracking),
planning (Mind) and bar rendering (DuetEngine) run on worker threads with
>= 1 bar of render-ahead, revisable until 1 beat before each downbeat.
Machine output is self-masked out of the beat tracker's phase correction.

Ctrl-C ends gracefully: the current bar finishes and the tail rings out.
--record DIR writes human.wav + machine.wav + decisions.ndjson.
--monitor mixes the mic into the output (headphones recommended).

Needs sounddevice + a duplex-capable audio device; without one it errors
cleanly — use 'plume sim' for the offline twin. Run 'plume mind' for the
full --mind K=V parameter table."""

_LISTEN_EPILOG = """\
Headless listening test: no synthesis, no audio device, no session file.
Streams INPUT.wav through the causal feature/beat/harmony trackers and emits
an NDJSON line roughly every 0.5 s:

  {"t_s":..., "bpm":..., "conf":..., "locked":..., "bar":...,
   "beat_in_bar":..., "phase_beats":..., "next_beat_t_s":...,
   "next_downbeat_t_s":..., "key":..., "key_conf":..., "chord":...,
   "chord_conf":...}

plus event lines ({"event": "lock"|"unlock", ...} and {"event":
"key_change", "from":..., "to":...}) as they happen. Unpredicted times
(inf) serialize as null. --json wraps the same data in one document:
{"path", "sr", "duration_s", "states": [...], "events": [...]}."""

_MIND_EPILOG_BODY = """\
Prints every MindParams field with its default and meaning. Override any of
them per run with --mind K=V (repeatable) on 'plume sim' or 'plume jam';
unknown keys error with the valid names.

"""

# ------------------------------------------------------------- v2 epilogs

_ROLES_EPILOG_BODY = """\
Prints every RoleParams field with its default and meaning, plus the
(clock_authority, form_authority) pair each role starts from. Override any
field per run with --roles K=V (repeatable) on 'plume simlead', 'plume lead'
or 'plume jam'; unknown keys error with the valid names.

Authority is a dial, not a mode: clock_authority 0 = the human's beat is
truth, 1 = the machine's clock free-runs; form_authority 0 = the machine
takes the form from what it hears, 1 = it plays its composed arrangement.
The v1 sideman is (0, 0); a backing track is (1, 1); everything interesting
is in between, and the dials MOVE during the session.

"""

_SIMLEAD_EPILOG_BODY = """\
The offline twin of 'plume lead', inverted from 'plume sim': the MACHINE's
clock is authoritative and starts at t=0 from --session (or from --prompt via
v0 compose), and HUMAN.wav is mixed in starting at --human-offset — they are
joining us. Omit HUMAN.wav entirely for a machine solo.

Deterministic: byte-identical for identical (input, args). The decisions
NDJSON gains role, clock_authority, tempo_offset_pct, transpose_semitones and
form records alongside v1's per-bar lines.

--cue "8.0:darker" queues a cue at 8.0s (repeatable). Append @BOUNDARY to
pick when it lands: "8.0:go to the chorus@next-section". Boundaries: now,
next-bar, next-phrase (default), next-section.

--json prints the SimResult manifest (paths, states, bpm_trace, final_key).

"""

_LEAD_EPILOG = """\
Live lead mode (SPEC-V2 section 5): the machine starts from PROMPT (or
--session) and holds the floor; you join it. Same duplex transport as 'plume
jam' — 48 kHz, 256-sample blocks, a callback that only copies ring buffers —
but the planner runs off the MACHINE's clock, so bars exist from t=0 and keep
coming through anything you do or don't play.

Each bar: your completed bar -> role negotiation (concede tempo, yield or
take the lead, meet you harmonically) -> form navigation (wait at the
turnaround, chase your energy, apply queued cues) -> render at the negotiated
tempo. --role trade alternates authority every trade_bars.

Cues arrive live on a local socket — see 'plume cue'. --record DIR writes
human.wav + machine.wav + decisions.ndjson. Ctrl-C ends gracefully.

--tap-start --detect needs no PROMPT at all: whatever the prompt does not name,
plume works out as you play. The key comes from what it hears, and the style
from your tempo, then your key and how busy you are. Both are guesses.

Needs sounddevice + a duplex-capable audio device; without one it errors
cleanly — use 'plume simlead' for the offline twin. Run 'plume roles' for the
full --roles K=V parameter table.

NOTE: live lead voices the session's tempo, key, chord cycle, style and
section map through the bar-granular duet renderer. It is the arrangement as
a map, not a sample-exact live replay of 'plume render'; 'plume simlead' is
the ground truth for the full arrangement."""

_CUE_EPILOG = """\
Send one cue to a running 'plume lead' / 'plume jam --role lead' session over
a local AF_UNIX socket ($PLUME_SOCKET, else <tmpdir>/plume-live.sock; --socket
overrides). The session queues it and applies it at --at; it lands in the
decision log with its effect.

  plume cue "darker"
  plume cue "go to the chorus" --at next-section
  plume cue "take it out" --at now

Cue kinds are rule-parsed: energy / section / texture / tempo / end / style,
plus "unknown" for text we did not understand (applied as an explicit no-op
and logged as such — guessing would change the music for no traceable
reason). --json reports the parse alongside the session's reply.

A cue asking for a loop goes to the looper instead, which answers at once, and
refuses by name what it cannot follow. See docs/PLAYING.md, "Making a loop":

  plume cue "make a loop of this"
  plume cue "make a 4 bar loop called bass"
  plume cue "that's the loop"
  plume cue "mute the bass loop"
  plume cue "stop the loop"
  plume cue "swap the bass loop for the keys loop"
  plume cue "drop the bass loop"
  plume cue "list loops"

With nothing listening this errors cleanly and changes nothing."""

_JAM_V2_EPILOG = """\

v2: --role R (or --roles K=V) and --lead-in "prompt" make the machine START
and you join — the same transport as 'plume lead'. Without them this is the
v1 sideman, unchanged. Run 'plume roles' for the parameter table.

The cue socket ('plume cue', --socket) belongs to lead mode: the v1 sideman
has no form cursor for a cue to move, so a plain 'plume jam' opens no socket."""


def _build_parser() -> argparse.ArgumentParser:
    # Global options accepted both before and after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--session", default=argparse.SUPPRESS,
                        help="path to the session file (*.plume.json)")
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="machine output: exactly one JSON document on stdout")

    parser = argparse.ArgumentParser(
        prog="plume",
        description="plume — an LLM-native music partner (prompt -> session -> WAV -> analysis)",
        epilog=("Patterns use mini-notation — run 'plume notation' for the full "
                "grammar, token semantics and examples."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--session", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", default=False, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    _MAKE_NEW_EPILOG = ("--session PATH writes the new session file to PATH "
                        "(its directory must exist) instead of "
                        "./<name>.plume.json.")
    p = sub.add_parser("make", parents=[common],
                       help="compose a session from a free-text prompt",
                       epilog=_MAKE_NEW_EPILOG,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", help="free text, e.g. 'chill trap over lofi at 120 bpm'")
    p.add_argument("--name", default=None, help="session name (default: derived from the prompt)")
    p.add_argument("--seed", type=int, default=None,
                   help="integer rng seed (default: stable hash of the prompt)")
    p.set_defaults(func=cmd_make)

    p = sub.add_parser("new", parents=[common],
                       help="create a session from a style",
                       epilog=_MAKE_NEW_EPILOG,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="session name")
    p.add_argument("--style", required=True,
                   help="style name (run 'plume styles' for the list)")
    p.add_argument("--bpm", type=float, default=None,
                   help="tempo in bpm (default: style range midpoint)")
    p.add_argument("--key", default=None,
                   help="key, e.g. 'F minor', 'Fm', 'f#m', 'C' (default: rng pick)")
    p.add_argument("--movements", type=int, default=3,
                   help="number of movements (default: 3)")
    p.add_argument("--seed", type=int, default=None,
                   help="integer rng seed for reproducibility")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("show", parents=[common], help="session summary / event stats")
    p.add_argument("--events", action="store_true",
                   help="also compile the session and print event statistics")
    p.add_argument("--movement", type=int, default=None,
                   help="limit the movement table to this 0-based index")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser(
        "set", parents=[common],
        help="set a value by dotted path (bpm, master.gain_db, tracks.pad.fx.lpf_hz)",
        epilog=("Paths are validated against the session schema; unknown keys "
                "are rejected with the valid options. Run 'plume get --list' "
                "to enumerate every settable path with its current value."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("key", help="dotted path, e.g. bpm | master.gain_db | tracks.pad.fx.lpf_hz")
    p.add_argument("value", help="JSON literal when it parses (0.3, true), else a string")
    p.set_defaults(func=cmd_set)

    p = sub.add_parser(
        "get", parents=[common],
        help="read a value by dotted path (JSON out); --list enumerates settable paths",
    )
    p.add_argument("path", nargs="?", default=None,
                   help="dotted path (omit for the whole session document)")
    p.add_argument("--list", action="store_true",
                   help="list settable paths (optionally under PATH) with current values")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser(
        "unset", parents=[common],
        help="remove an optional key (master.*, tracks.NAME.fx.*, ...)",
    )
    p.add_argument("path", help="dotted path of the key to remove")
    p.set_defaults(func=cmd_unset)

    p = sub.add_parser("notation", parents=[common],
                       help="pattern mini-notation reference (grammar + examples)")
    p.set_defaults(func=cmd_notation)

    p = sub.add_parser("track", parents=[common], help="manage tracks")
    tsub = p.add_subparsers(dest="track_command", required=True)
    tp = tsub.add_parser("add", parents=[common], help="add a track")
    tp.add_argument("name", help="track name (referenced by section plays)")
    tp.add_argument("--voice", required=True,
                    help="voice preset (run 'plume voices' for the catalog)")
    tp.add_argument("--kind", choices=["drum", "tonal"], default=None,
                    help="token interpretation (default: inferred from the voice)")
    tp.add_argument("--gain-db", dest="gain_db", type=float, default=0.0,
                    help="track gain in dB (default: 0)")
    tp.add_argument("--pan", type=float, default=0.0,
                    help="stereo position -1 (left) .. +1 (right) (default: 0)")
    tp.set_defaults(func=cmd_track_add)
    tp = tsub.add_parser("rm", parents=[common], help="remove a track")
    tp.add_argument("name", help="track name to remove")
    tp.set_defaults(func=cmd_track_rm)
    tp = tsub.add_parser("list", parents=[common], help="list tracks")
    tp.set_defaults(func=cmd_track_list)

    p = sub.add_parser("pattern", parents=[common], help="get/set track patterns")
    psub = p.add_subparsers(dest="pattern_command", required=True)
    pp = psub.add_parser(
        "set", parents=[common], help="set a pattern (validates the notation)",
        epilog=_NOTATION_COMPACT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pp.add_argument("track", help="track name")
    pp.add_argument("name", help="pattern name (sections reference it via plays)")
    pp.add_argument("pattern", help="mini-notation string (one bar)")
    pp.set_defaults(func=cmd_pattern_set)
    pp = psub.add_parser("get", parents=[common], help="print a track's pattern(s)")
    pp.add_argument("track", help="track name")
    pp.add_argument("name", nargs="?", default=None,
                    help="pattern name (omit to list all patterns on the track)")
    pp.set_defaults(func=cmd_pattern_get)

    p = sub.add_parser("arrange", parents=[common], help="show/set movements & sections")
    asub = p.add_subparsers(dest="arrange_command", required=True)
    ap = asub.add_parser(
        "show", parents=[common], help="print the movement/section table",
        epilog=_ARRANGE_SHOW_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.set_defaults(func=cmd_arrange_show)
    ap = asub.add_parser(
        "set", parents=[common], help="replace the arrangement from a JSON spec",
        epilog=_ARRANGE_SET_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--spec-json", dest="spec_json", required=True,
                    help="movements spec: inline JSON or @FILE.json (schema below)")
    ap.set_defaults(func=cmd_arrange_set)

    p = sub.add_parser(
        "render", parents=[common], help="render the session to WAV",
        epilog=_RENDER_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", default=None,
                   help="output WAV path (default: <session>.wav next to the session)")
    p.add_argument("--movement", type=int, default=None,
                   help="render only this 0-based movement")
    p.add_argument("--bars", nargs=2, type=int, default=None, metavar=("A", "B"),
                   help="render only bars A..B (0-based, half-open) within the scope")
    p.add_argument("--stems", default=None, metavar="DIR",
                   help="also write per-track stems (post-fx, pre-master, -3 dB headroom)")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser(
        "analyze", parents=[common], help="analyze a WAV file",
        epilog=_ANALYZE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", help="WAV file to analyze")
    p.add_argument("--bpm-hint", dest="bpm_hint", type=float, default=None,
                   help="expected bpm (default: the resolved session's bpm, if any)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("play", parents=[common], help="render and play (needs sounddevice)")
    p.add_argument("--movement", type=int, default=None,
                   help="play only this 0-based movement")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("voices", parents=[common], help="voice preset catalog")
    p.set_defaults(func=cmd_voices)

    p = sub.add_parser("styles", parents=[common], help="style catalog")
    p.set_defaults(func=cmd_styles)

    p = sub.add_parser("doctor", parents=[common], help="environment check")
    p.add_argument("--probe", default=None, metavar="TAKE.wav",
                   help="also report which --input mode this file would "
                        "select (the same measurement 'auto' makes) and why")
    p.set_defaults(func=cmd_doctor)

    # ---- v1 live mode (SPEC-V1 section 8): sim / jam / listen / mind ----
    p = sub.add_parser(
        "sim", parents=[common],
        help="offline duet: stream a WAV through the live pipeline (deterministic)",
        epilog=_SIM_EPILOG_BODY + _mind_table(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", metavar="INPUT.wav",
                   help="the human's audio (WAV; mono-summed)")
    p.add_argument("--style", default="lofi",
                   help="style name (default: lofi; run 'plume styles')")
    p.add_argument("--out", default="duet.wav",
                   help="output mix WAV (default: duet.wav)")
    p.add_argument("--decisions", default=None, metavar="d.ndjson",
                   help="decisions NDJSON path (default: derived from --out)")
    p.add_argument("--seed", type=int, default=0,
                   help="integer rng seed for the duet engine (default: 0)")
    p.add_argument("--tightness", type=float, default=None,
                   help="beat-tracker phase-correction alpha in (0, 1] "
                        "(default: tracker default 0.35)")
    p.add_argument("--mind", action="append", default=None, metavar="K=V",
                   help="override a MindParams field (repeatable; table below "
                        "and via 'plume mind')")
    p.add_argument("--taps", default=None, metavar='"T1,T2,..."',
                   help="count-in tap times in seconds (>=4 taps hard-set "
                        "tempo + downbeat)")
    p.add_argument("--mix-human-db", dest="mix_human_db", type=float, default=0.0,
                   help="human level in the final mix in dB (default: 0)")
    p.add_argument("--bpm-hint", dest="sim_bpm_hint", type=float, default=None,
                   metavar="BPM",
                   help="known tempo: narrows the tracker's tempo prior around "
                        "it (no phase claim, unlike --taps)")
    p.add_argument("--full-trace", action="store_true", default=False,
                   help="with --json, emit every bpm_trace point instead of a "
                        "decimated sample")
    _add_input_mode_flag(p)
    p.set_defaults(func=cmd_sim)

    p = sub.add_parser(
        "jam", parents=[common],
        help="real-time duet over mic/speaker (needs sounddevice)",
        epilog=_JAM_EPILOG + _JAM_V2_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--style", default="lofi",
                   help="style name (default: lofi; run 'plume styles')")
    p.add_argument("--seed", type=int, default=0,
                   help="integer rng seed for the duet engine (default: 0)")
    p.add_argument("--tightness", type=float, default=None,
                   help="beat-tracker phase-correction alpha in (0, 1]")
    p.add_argument("--mind", action="append", default=None, metavar="K=V",
                   help="override a MindParams field (repeatable; "
                        "run 'plume mind' for the table)")
    p.add_argument("--record", default=None, metavar="DIR",
                   help="write human.wav + machine.wav + decisions.ndjson here")
    p.add_argument("--device", default=None, metavar="NAME",
                   help="audio device name substring (default: system default)")
    p.add_argument("--monitor", action="store_true", default=False,
                   help="mix the mic into the output (headphones recommended)")
    _add_route_flags(p, voice=False)
    # v2 (additive): absent = the v1 sideman, byte for byte.
    p.add_argument("--role", default=None, metavar="R",
                   help="starting role (run 'plume roles'); default follow = "
                        "v1 behavior")
    p.add_argument("--roles", action="append", default=None, metavar="K=V",
                   help="override a RoleParams field (repeatable; table via "
                        "'plume roles')")
    p.add_argument("--lead-in", dest="lead_in", default=None, metavar="PROMPT",
                   help="the machine starts, composing from this prompt, and "
                        "you join it")
    p.add_argument("--socket", default=None, metavar="PATH",
                   help="cue socket path (default: $PLUME_SOCKET or "
                        "<tmpdir>/plume-live.sock)")
    p.add_argument("--max-minutes", dest="max_minutes", type=float,
                   default=None, metavar="N", help=_max_minutes_help())
    _add_input_mode_flag(p)
    p.add_argument("--count-in", dest="count_in", action="store_true",
                   default=False, help=_COUNT_IN_HELP)
    p.add_argument("--latency-comp", dest="latency_comp", type=float,
                   default=None, metavar="MS",
                   help="place bars this many ms early to cancel the "
                        "speaker->mic round trip (0 disables; default: the "
                        "'plume calibrate' measurement for this device pair, "
                        "else the stream's reported latency). Follow mode "
                        "only.")
    _add_midi_flag(p)
    p.set_defaults(func=cmd_jam)

    p = sub.add_parser(
        "calibrate", parents=[common],
        help="measure the speaker->mic round trip with a click train; live "
             "bars then place early by it (plays sound; needs sounddevice)")
    p.add_argument("--device", default=None, metavar="NAME",
                   help="audio device name substring (default: system default)")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser(
        "listen", parents=[common],
        help="stream a WAV through the trackers; NDJSON states, no synthesis",
        epilog=_LISTEN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", metavar="INPUT.wav", help="WAV file to listen to")
    _add_input_mode_flag(p)
    p.set_defaults(func=cmd_listen)

    p = sub.add_parser(
        "mind", parents=[common],
        help="print every MindParams threshold: name, default, meaning",
        epilog=_MIND_EPILOG_BODY + _mind_table(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=cmd_mind)

    # ---- v2 lead mode (SPEC-V2 section 5): roles / simlead / lead / cue ----
    p = sub.add_parser(
        "roles", parents=[common],
        help="print every RoleParams threshold: name, default, meaning",
        epilog=_ROLES_EPILOG_BODY + _roles_table(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=cmd_roles)

    p = sub.add_parser(
        "simlead", parents=[common],
        help="offline lead: the machine starts, a WAV joins it (deterministic)",
        epilog=_SIMLEAD_EPILOG_BODY + _roles_table(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", metavar="HUMAN.wav", nargs="?", default=None,
                   help="the human's audio (WAV; omit for a machine solo)")
    p.add_argument("--prompt", default=None,
                   help="compose the machine's session from this free text "
                        "(alternative to --session)")
    p.add_argument("--out", default="duet.wav",
                   help="output mix WAV (default: duet.wav)")
    p.add_argument("--decisions", default=None, metavar="d.ndjson",
                   help="decisions NDJSON path (default: derived from --out)")
    p.add_argument("--human-offset", dest="human_offset", type=float,
                   default=0.0, metavar="S",
                   help="seconds into the machine's take that the human joins "
                        "(default: 0)")
    p.add_argument("--seed", type=int, default=0,
                   help="integer rng seed (default: 0)")
    p.add_argument("--style", default=None,
                   help="style for --prompt sessions that name none "
                        "(default: the prompt's / the session's)")
    p.add_argument("--bars", type=int, default=None, metavar="N",
                   help="length of the machine's take in bars "
                        "(default: the session's own length)")
    p.add_argument("--mix-human-db", dest="mix_human_db", type=float,
                   default=None, metavar="DB",
                   help="human level in the final mix in dB (default: 0)")
    p.add_argument("--role", default="lead", metavar="R",
                   help="starting role: lead|follow|trade|solo_support "
                        "(default: lead)")
    p.add_argument("--roles", action="append", default=None, metavar="K=V",
                   help="override a RoleParams field (repeatable; table below)")
    p.add_argument("--mind", action="append", default=None, metavar="K=V",
                   help="override a MindParams field (repeatable; table via "
                        "'plume mind')")
    p.add_argument("--cue", action="append", default=None, metavar='"T:TEXT"',
                   help="queue a cue at T seconds (repeatable), e.g. "
                        '"8.0:darker" or "16:go to the chorus@next-section"')
    p.add_argument("--full-trace", action="store_true", default=False,
                   help="with --json, emit every bpm_trace point instead of a "
                        "decimated sample")
    p.set_defaults(func=cmd_simlead)

    p = sub.add_parser(
        "lead", parents=[common],
        help="live lead: the machine starts, you join (needs sounddevice)",
        epilog=_LEAD_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("prompt", nargs="?", default=None, metavar="PROMPT",
                   help="free text to compose from, e.g. 'chill trap at 120, "
                        "3 movements' (alternative to --session)")
    p.add_argument("--from-bar", dest="from_bar", type=int, default=0,
                   metavar="N",
                   help="start N bars into the session's map (default: 0)")
    p.add_argument("--role", default="lead", metavar="R",
                   help="starting role: lead|follow|trade|solo_support "
                        "(default: lead)")
    p.add_argument("--roles", action="append", default=None, metavar="K=V",
                   help="override a RoleParams field (repeatable; run "
                        "'plume roles')")
    p.add_argument("--mind", action="append", default=None, metavar="K=V",
                   help="override a MindParams field (repeatable; run "
                        "'plume mind')")
    p.add_argument("--style", default=None,
                   help="not read when a prompt or session is given, which is every "
                        "take: name the style in the prompt")
    p.add_argument("--seed", type=int, default=0,
                   help="integer rng seed (default: 0)")
    p.add_argument("--tightness", type=float, default=None,
                   help="beat-tracker phase-correction alpha in (0, 1]")
    p.add_argument("--record", default=None, metavar="DIR",
                   help="write human.wav + machine.wav + decisions.ndjson here")
    p.add_argument("--device", default=None, metavar="NAME",
                   help="audio device name substring (default: system default)")
    p.add_argument("--monitor", action="store_true", default=False,
                   help="mix the mic into the output (headphones recommended)")
    p.add_argument("--socket", default=None, metavar="PATH",
                   help="cue socket path (default: $PLUME_SOCKET or "
                        "<tmpdir>/plume-live.sock)")
    p.add_argument("--max-minutes", dest="max_minutes", type=float,
                   default=None, metavar="N", help=_max_minutes_help())
    _add_input_mode_flag(p)
    p.add_argument("--count-in", dest="count_in", action="store_true",
                   default=False, help=_COUNT_IN_HELP)
    _add_route_flags(p)
    p.add_argument("--wind-down", dest="wind_down", type=float, default=None,
                   metavar="SECONDS",
                   help="with --tap-start: this long after the drums start, signal "
                        "the ending with a crash, then fade and end")
    p.add_argument("--tap-start", dest="tap_start", action="store_true",
                   default=False,
                   help="wait silently for 4 knocks, start drums at that tempo "
                        "and hold it; bass and chords join in your key once it "
                        "has settled, then the melody")
    p.add_argument("--detect", action="store_true", default=False,
                   help="with --tap-start: no session, and PROMPT is optional. What "
                        "it does not name, plume guesses from your playing")
    _add_midi_flag(p)
    p.set_defaults(func=cmd_lead)

    p = sub.add_parser(
        "cue", parents=[common],
        help="send a cue to a running lead/jam session (local socket)",
        epilog=_CUE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("text", help='the cue, e.g. "darker", "go to the chorus"')
    p.add_argument("--at", default="next-phrase",
                   help="boundary to apply it at: now | next-bar | "
                        "next-phrase (default) | next-section")
    p.add_argument("--ttl-bars", dest="ttl_bars", type=int, default=16,
                   metavar="N",
                   help="drop the cue if it has not applied within N bars "
                        "(default: 16)")
    p.add_argument("--socket", default=None, metavar="PATH",
                   help="cue socket path (default: $PLUME_SOCKET or "
                        "<tmpdir>/plume-live.sock)")
    p.add_argument("--source", default="cli", choices=("cli", "gui", "voice"),
                   help="who is asking, for the take's cues.ndjson (default: cli)")
    p.set_defaults(func=cmd_cue)

    # ---- v2.2: finding and ending a live session (SPEC-V2 section 5) ----
    p = sub.add_parser(
        "voice", parents=[common],
        help="play a tap-start take you can talk to: 'plume, stop', 'plume, "
             "restart', 'plume, busier', 'plume, sparser' (plays sound; speech "
             "is transcribed on this machine; needs mlx-whisper)",
    )
    p.add_argument("prompt", help="what to play, e.g. 'lofi midwest emo in F major'")
    p.add_argument("--base", default=".", help="where takesN directories go")
    p.add_argument("--wind-down", dest="wind_down", type=float, default=None,
                   metavar="SECONDS", help="signal the ending this long after the "
                                           "drums start, then fade and end")
    p.add_argument("--device", default=None, metavar="NAME",
                   help="audio device name substring, both directions")
    _add_route_flags(p)
    _add_midi_flag(p)
    p.set_defaults(func=cmd_voice)

    p = sub.add_parser(
        "voice-replay", parents=[common],
        help="what the speech model hears in a recording, window by window, and whether "
             "it would count as a command (no device, no sound; needs mlx-whisper)",
    )
    p.add_argument("path", help="a WAV, e.g. a take's human.wav or voice.wav")
    p.set_defaults(func=cmd_voice_replay)

    p = sub.add_parser(
        "inputs", parents=[common],
        help="list audio devices and their channels; --meter also listens for a few "
             "seconds and says which channel is loudest (plays nothing)",
    )
    p.add_argument("--meter", default=None, metavar="NAME",
                   help="device name substring: open its inputs and show each channel's level")
    p.add_argument("--seconds", type=float, default=5.0, metavar="S",
                   help="how long --meter listens, up to 60 (default: 5)")
    p.set_defaults(func=cmd_inputs)

    p = sub.add_parser(
        "midi", parents=[common],
        help="list MIDI input ports, for --midi (opens none of them)",
    )
    p.set_defaults(func=cmd_midi)

    p = sub.add_parser(
        "watch", parents=[common],
        help="stream a live session as NDJSON: what the band and the player are "
             "doing, read from its --record directory (opens no audio device)",
    )
    p.add_argument("record_dir", help="the session's --record directory")
    p.add_argument("--every", type=int, default=4, metavar="N",
                   help="one line per N bars (default: 4)")
    p.add_argument("--changes-only", dest="changes_only", action="store_true",
                   default=False, help="only lines where something changed")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser(
        "gui", parents=[common],
        help="start a take and read your takes in a local page: what plume did, a roast, "
             "and Keep. It opens no device and starts nothing until you click Start",
    )
    p.add_argument("action", nargs="?", choices=("url",), default=None,
                   help="'url' prints a new one-time link to the running page")
    p.add_argument("--take", default=None, metavar="DIR",
                   help="open this take folder's Record first")
    p.add_argument("--takes", default=None, metavar="DIR",
                   help="folder of take folders for the shelf (default: ~/Music/Plume)")
    p.add_argument("--no-open", dest="no_open", action="store_true", default=False,
                   help="print the one-time link instead of opening a browser")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser(
        "sessions", parents=[common],
        help="list running live sessions: pid, role, style, uptime, how to stop",
        epilog=_session_record_table(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=cmd_sessions)

    p = sub.add_parser(
        "stop", parents=[common],
        help="stop running live sessions (jam/lead) from any terminal",
        epilog=_STOP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pid", type=int, default=None, metavar="N",
                   help="stop just this session ('plume sessions' lists pids)")
    p.add_argument("--all", action="store_true", default=False,
                   help="stop every running live session")
    p.add_argument("--timeout", type=float, default=None, metavar="SECONDS",
                   help="budget for the graceful ending before SIGTERM "
                        "(default: 45 — longer than the transport's own "
                        "~40s ending, so a working flush is never cut short). "
                        "An upper bound, not a delay.")
    p.add_argument("--force", action="store_true", default=False,
                   help="escalate to SIGKILL if the graceful ending does not "
                        "finish. Costs the tail and ending.json, not the "
                        "recording: since v1.3 the take is written as it "
                        "happens. Only for a session that is genuinely wedged "
                        "— 'plume stop' says so when it is.")
    p.set_defaults(func=cmd_stop)

    return parser


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    as_json = bool(getattr(args, "json", False))
    try:
        payload, human, code = args.func(args)
    except CliError as e:
        if as_json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — CLI boundary: report, don't traceback
        if os.environ.get("PLUME_DEBUG"):
            raise
        # v2.1: the human text is PROSE.  A Python class name leaking into it
        # ("error: LibsndfileError: Error opening 'x.wav': Format not
        # recognised.") is the one error in the surface that reads like a
        # traceback, and it says nothing the message did not.  The class name
        # is still in --json, where a machine can use it.
        msg = str(e).strip() or type(e).__name__
        if as_json:
            print(json.dumps({"error": msg, "error_type": type(e).__name__}))
        else:
            print(f"error: {msg}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(human)
    return code


if __name__ == "__main__":
    sys.exit(main())
