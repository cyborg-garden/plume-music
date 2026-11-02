"""Style data loader. SPEC.md section 5.

Style files live in ``src/plume/styles/*.json`` (shipped as package data).
Each file is a declarative description of a genre: bpm range, swing,
timing-bias table, weighted progression pool, drum grids, fill lexicon,
bass/pad styles, melody config, role->preset voice map, fx defaults,
master hints, and arrangement templates.
"""
from __future__ import annotations

import json
from pathlib import Path

_STYLES_DIR = Path(__file__).resolve().parent / "styles"

# keys every style file must carry (light schema check at load time)
_REQUIRED_KEYS = (
    "name", "bpm_range", "default_mode", "swing", "timing_bias_ms",
    "progressions", "drum_patterns", "fills", "bass_style", "pad_style",
    "melody", "voices", "fx_defaults", "master", "arrangement_templates",
)


def _normalize(name: str) -> str:
    """'Lo-Fi' / 'lo fi' / 'synth-wave' -> 'lofi' / 'synthwave'."""
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def list_styles() -> list[str]:
    """Names of all bundled styles, sorted."""
    return sorted(p.stem for p in _STYLES_DIR.glob("*.json"))


def load_style(name: str) -> dict:
    """Load a style by name from package data. Raises ValueError for unknown.

    Returns a fresh dict on every call (callers may mutate their copy).
    """
    path = _STYLES_DIR / f"{_normalize(name)}.json"
    if not path.is_file():
        available = ", ".join(list_styles()) or "(none)"
        raise ValueError(f"unknown style {name!r}; available: {available}")
    with open(path, encoding="utf-8") as f:
        style = json.load(f)
    missing = [k for k in _REQUIRED_KEYS if k not in style]
    if missing:
        raise ValueError(f"style {name!r} is missing keys: {missing}")
    return style
