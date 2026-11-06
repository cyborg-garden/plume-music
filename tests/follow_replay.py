"""Replay BandFollow over a RECORDED take: her level per bar against the band's gain.

Not a test, and it must never become one: it reads a take's ``human.wav``, which is
gitignored (SPEC-V1 amendment 21: no test may read a gitignored artifact).  It is the
measurement behind the follow timings in ``bandentry.py``, kept runnable so they can be
checked instead of believed.  Run with::

    PYTHONPATH=src .venv/bin/python tests/follow_replay.py ~/Music/Plume/<take>/

It needs ``human.wav`` and ``decisions.ndjson`` (for the bar grid, when the band came
in, and the gain the take actually logged).  The listening is the live path's own:
FeatureExtractor frames, ``bar_features_dict`` per bar, and, where this code has
them, the half-bar readings the planner feeds BandFollow (``follow_parts``).  Each bar
N is planned from bar N-2, and from the first half of bar N-1 when the planner runs
half a bar ahead, exactly as live.

Printed per bar: her level over that bar, the gain the take logged, and the gain this
code would give.  A bar's gain is heard from that bar's start.  Then, for every move
the replay makes, how many seconds after her change it was heard.
"""
import inspect
import json
import math
import os
import sys

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from plume import realtime                                       # noqa: E402
from plume.bandentry import BandFollow                           # noqa: E402
from plume.listen.features import FeatureExtractor               # noqa: E402

BLOCK = 4096
SILENCE_RMS_DB = -45.0


def frames_of(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    fe = FeatureExtractor(sr=sr)
    events = []
    for i in range(0, len(x), BLOCK):
        for f in fe.process(x[i:i + BLOCK]):
            events.append((f.t_s, f.rms_db, f.onset))
    return events, x, sr


def bar_db(x, sr, t0, t1):
    seg = x[int(t0 * sr):int(t1 * sr)].astype(np.float64)
    return 20.0 * math.log10(math.sqrt(float(np.mean(seg * seg))) + 1e-12) if seg.size else -90.0


def replay(take):
    rows = [json.loads(line) for line in open(os.path.join(take, "decisions.ndjson"))]
    rows = [r for r in rows if isinstance(r.get("bar"), int)]
    events, x, sr = frames_of(os.path.join(take, "human.wav"))
    starts = [r["t_s"] for r in rows]
    bar_s = starts[1] - starts[0]
    starts.append(starts[-1] + bar_s)
    halves = "parts" in inspect.signature(BandFollow.bar).parameters
    follow = BandFollow(trusted=bool(rows[0].get("follow", {}).get("trusted", True)))
    ref_db, fed = None, None
    out = []
    for i, r in enumerate(rows):
        n = r["bar"]
        band_in = r.get("entry") not in (None, "drums")
        human, parts = None, None
        if i >= 2:
            w0, w1 = starts[i - 2], starts[i - 1]
            evs = [e for e in events if w0 <= e[0] < w1]
            if evs:
                kw = realtime.bar_features_dict(n - 1, evs, silence_rms_db=SILENCE_RMS_DB,
                                                ref_db=ref_db)
                human = kw
                if not kw["silent"]:
                    ref_db = (kw["rms_db"] if ref_db is None
                              else 0.5 * kw["rms_db"] + 0.5 * ref_db)
            if halves:
                later = [e for e in events if w1 <= e[0] < starts[i]]
                parts, fed = realtime.follow_parts(
                    evs, later, w0, w1, starts[i], fed_until=fed,
                    heard_until=starts[i - 1] + 0.5 * bar_s + 0.1,
                    silence_rms_db=SILENCE_RMS_DB, ref_db=ref_db)
        playing = human is not None and not human["silent"]
        kw = {"playing": playing, "band_in": band_in}
        if halves:
            kw.update(parts=parts, bar_s=bar_s)
        s = follow.bar(None if human is None else human["rms_db"],
                       None if human is None else human["onset_density"], **kw)
        out.append((n, r["t_s"], bar_db(x, sr, starts[i], starts[i + 1]),
                    r.get("follow", {}).get("gain_db"), s.gain_db, s.moved))
    return out, bar_s


def main(argv):
    if len(argv) < 2:
        sys.exit("usage: follow_replay.py <take dir>")
    take = os.path.expanduser(argv[1])
    out, bar_s = replay(take)
    print(f"take {take}  bar {bar_s:.2f} s ({240.0 / bar_s:.1f} bpm)")
    print(f"{'bar':>3} {'t_s':>7} {'her dB':>7} {'logged':>7} {'replay':>7}  moved")
    for n, t, her, logged, g, moved in out:
        print(f"{n:>3} {t:7.1f} {her:7.1f} {logged if logged is not None else float('nan'):7.1f}"
              f" {g:7.1f}  {' '.join(moved)}")
    first = next((t for _n, t, _her, _logged, g, _moved in out if g != 0.0), None)
    if first is not None:
        print(f"first band move heard at t={first:.1f} s")


if __name__ == "__main__":
    main(sys.argv)
