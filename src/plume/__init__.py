"""plume — an LLM-native music partner.

Convenience imports are lazy (PEP 562) so that ``import plume`` stays cheap
and the package imports even in partial checkouts.
"""
from __future__ import annotations

import importlib

__version__ = "0.1.0"

# public name -> (module, attribute)
_LAZY: dict[str, tuple[str, str]] = {
    "Session": ("plume.score", "Session"),
    "load": ("plume.score", "load"),
    "save": ("plume.score", "save"),
    "compose": ("plume.compose", "compose"),
    "render": ("plume.engine", "render"),
    "analyze": ("plume.analyze", "analyze"),
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # cache for next access
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
