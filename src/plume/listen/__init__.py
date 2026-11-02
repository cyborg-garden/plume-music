"""plume.listen — causal listening: frame features, beat tracking, harmony.

Sibling modules (beat, harmony) are built concurrently; their re-exports are
tolerant so this package imports cleanly while siblings are absent or broken.
"""
from .features import HOP, WIN, Frame, FeatureExtractor

__all__ = ["HOP", "WIN", "Frame", "FeatureExtractor"]

try:
    from .beat import BeatState, BeatTracker  # noqa: F401

    __all__ += ["BeatState", "BeatTracker"]
except Exception:  # pragma: no cover - sibling not built yet
    pass

try:
    from .harmony import HarmonyState, HarmonyTracker  # noqa: F401

    __all__ += ["HarmonyState", "HarmonyTracker"]
except Exception:  # pragma: no cover - sibling not built yet
    pass
