"""Replay key detection over a RECORDED take: what the tracker heard, what the band took.

Not a test, and it must never become one: it reads a take's ``human.wav``, which is
gitignored (SPEC-V1 amendment 21: no test may read a gitignored artifact).  It is the
measurement behind the entry-key rule in ``bandentry.py``, kept runnable so it can be
checked instead of believed.  Run with::

    PYTHONPATH=src .venv/bin/python tests/key_replay.py <take dir> [--ahead SECONDS]

It needs ``human.wav`` and ``decisions.ndjson`` (for the bar grid and whether she was
playing each bar).  The listening is the live path's own: FeatureExtractor frames into
a HarmonyTracker started at the first bar, read ``--ahead`` seconds before each bar
(the planner runs ahead of the bar it plans; default 0), then BandEntry with the
detect-mode settings of a direct input.

Printed per bar: the tracker's key and its top three correlations, the key and chord
the band would play, and the key the take logged.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from plume.bandentry import DETECT_KEY_FALLBACK_BARS, BandEntry  # noqa: E402
from plume.listen import harmony as H                             # noqa: E402
from plume.listen.features import FeatureExtractor                # noqa: E402
from plume.mind import MindParams                                 # noqa: E402

BLOCK = 4096
LIVE_LAG_BARS = 2


def frames_of(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    fe = FeatureExtractor(sr=sr)
    frames = []
    for i in range(0, len(x), BLOCK):
        frames.extend(fe.process(x[i:i + BLOCK]))
    return frames


def top3(tracker):
    hist = tracker._hist
    if hist.std() <= 0.0:
        return ""
    corr = H._PROFILES @ ((hist - hist.mean()) / hist.std()) / 12.0
    return " ".join(f"{H._key_name(int(k))}={corr[k]:.2f}"
                    for k in np.argsort(corr)[::-1][:3])


def replay(take, ahead=0.0, style=None):
    rows = [json.loads(line) for line in open(os.path.join(take, "decisions.ndjson"))]
    rows = [r for r in rows if isinstance(r.get("bar"), int)]
    frames = frames_of(os.path.join(take, "human.wav"))
    tracker = H.HarmonyTracker()
    # As the live loop builds it for a direct input in detect mode.
    entry = BandEntry(style or rows[0].get("style") or "midwest",
                      after_first_heard_bars=6, key_fallback_bars=DETECT_KEY_FALLBACK_BARS,
                      phrase_bars=MindParams().entry_boundary_bars)
    i = 0
    while i < len(frames) and frames[i].t_s < rows[0]["t_s"]:
        i += 1
    out = []
    for r in rows:
        batch = []
        while i < len(frames) and frames[i].t_s < r["t_s"] - ahead:
            batch.append(frames[i])
            i += 1
        h = tracker.process(batch)
        # The logged ``playing`` is the live loop's reading, of the bar its window
        # read: two bars before the bar it planned.
        e = entry.update(r["bar"], h, bool(r.get("playing")), heard_bar=r["bar"] - LIVE_LAG_BARS)
        out.append((r["bar"], h.key, top3(tracker), e.key, e.chord, r.get("key")))
    return out


def main(argv):
    args = argv[1:]
    ahead = 0.0
    if "--ahead" in args:
        j = args.index("--ahead")
        ahead = float(args[j + 1])
        del args[j:j + 2]
    if not args:
        sys.exit(__doc__)
    out = replay(os.path.expanduser(args[0]), ahead=ahead)
    print(f"{'bar':>3}  {'tracker':<10} {'top three':<48} {'band':<10} {'chord':<8} logged")
    for bar, hk, top, ek, chord, logged in out:
        print(f"{bar:>3}  {hk or '-':<10} {top:<48} {ek or '-':<10} {chord or '-':<8} {logged or '-'}")
    first = next(((bar, ek) for bar, _hk, _t, ek, _c, _l in out if ek), None)
    if first is not None:
        print(f"band enters at bar {first[0]} in {first[1]}")


if __name__ == "__main__":
    main(sys.argv)
