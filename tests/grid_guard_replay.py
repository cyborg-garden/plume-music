"""Re-measure the v2.7 settled-grid guard against a RECORDED take.

Not a test — the measurement behind the "54 re-reads, only 2 of them octaves"
paragraph in ``listen/beat._grid_hysteresis``, kept runnable so the numbers in
that comment can be checked instead of believed.  Run with::

    .venv/bin/python tests/grid_guard_replay.py takes6/
    .venv/bin/python tests/grid_guard_replay.py takes6/ takes7/ takes5/

It cannot be a test and must never become one, for the same reason
``relock_replay.py`` cannot: it reads a jam recording, and ``*.wav`` is
gitignored (SPEC-V1 amendment 21 — a test that silently does not run is not a
test).  Point it at any ``plume jam --record`` directory; it needs only
``human.wav``.

It is OFFLINE.  It opens no audio device and writes no audio: the take is read
from disk and pushed through the same FeatureExtractor + BeatTracker the live
path uses.

THE A/B.  ``v1.3`` restores the pre-amendment routing by subclassing the
tracker and sending non-octave re-reads back to ``_period_hysteresis`` — the
real method, not a copy of it, which is why that method has a name.  ``v2.7``
is the shipped tracker, untouched.

WHAT IT COUNTS.  A *re-read* is any change of the reported tempo, while
LOCKED, that lands outside the tracker's drift band — i.e. the tracker
re-interpreting the metre under a human who is being played along with.  It is
counted off the tempo trajectory rather than off ``BeatTracker.events`` on
purpose: the v1.3 arm does not emit events for non-octave changes, and an A/B
whose two arms are measured by different instruments is not an A/B.
"""
import math
import os
import sys
from collections import Counter

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from plume.listen import beat as beat_mod                       # noqa: E402
from plume.listen.beat import BeatTracker                       # noqa: E402
from plume.listen.features import FeatureExtractor              # noqa: E402

BLOCK = 4096

# The alias families a beat tracker actually confuses, as ratios >= 1.  The
# tolerance is 2% — tight enough that a 4/3 (1.3333) is never scored as a
# 5/4 (1.25) or a 3/2, and loose enough to survive the tracker's own drift.
FAMILIES = ((2.0, "2/1"), (1.5, "3/2"), (4.0 / 3.0, "4/3"),
            (1.25, "5/4"), (9.0 / 8.0, "9/8"), (4.0, "4/1"))
FAM_TOL = 0.035


def classify(old_bpm, new_bpm):
    """Name the alias family of a re-read, direction-folded ('4/3' covers 3/4)."""
    r = new_bpm / old_bpm
    if r < 1.0:
        r = 1.0 / r
    for value, name in FAMILIES:
        if abs(r - value) / value <= FAM_TOL:
            return name
    return "other"


class LegacyTracker(BeatTracker):
    """The v1.3 routing: only WHOLE OCTAVES face the commitment gate."""

    def _grid_hysteresis(self, bpm_new, now):
        if beat_mod._is_octave_jump(self._bpm, bpm_new):
            return super()._grid_hysteresis(bpm_new, now)
        self._oct_pending_bpm, self._oct_pending_n = None, 0
        self._period_hysteresis(bpm_new)


def replay(x, sr, cls, *, acoustic=True):
    fe, tr = FeatureExtractor(sr), cls(acoustic=acoustic)
    traj = []
    for i in range(0, len(x), BLOCK):
        frames = fe.process(x[i:i + BLOCK])
        if frames:
            st = tr.process(frames)
            traj.append((st.t_s, st.bpm, st.locked, st.confidence))
    return traj, tr


def rereads(traj, drift_pct=3.0):
    """Every locked, out-of-drift change of the reported tempo."""
    out, prev = [], None
    for t, bpm, locked, _conf in traj:
        if not locked or bpm <= 0.0:
            prev = bpm if bpm > 0.0 else prev
            continue
        if prev is not None and prev > 0.0:
            if abs(bpm - prev) / prev * 100.0 > drift_pct:
                out.append((t, prev, bpm, classify(prev, bpm)))
        prev = bpm
    return out


def truth_bpm(traj_bpms):
    """The take's tempo, recovered the way the 08-27 measurement recovered it:
    the MEDIAN of the tracker's own readings, folded into one octave.  Single
    windows are unreliable on this material (55, 58, 122, 134, 63, 160, 165 on
    takes6); the median across all of them read 111.7, Juniper's actual tempo.
    """
    folded = []
    for b in traj_bpms:
        if b <= 0.0:
            continue
        while b < 90.0:
            b *= 2.0
        while b >= 180.0:
            b /= 2.0
        folded.append(b)
    return float(np.median(folded)) if folded else float("nan")


def on_truth(traj, truth, tol_pct=5.0):
    """Fraction of the take whose REPORTED tempo is within tol of the truth.

    Octave-folded, because a grid at 2x the truth is a different defect (and
    one v1.3 already guards); this asks whether the tracker is on a DOTTED
    alias of the tune or on the tune.  Measured over every frame, not only
    locked ones: ``MindParams.entry_conf`` gates the sideman on CONFIDENCE,
    so the mind comps on this tempo whether or not the tracker calls it a lock.
    """
    hits = tot = 0
    for _t, bpm, _lk, _cf in traj:
        if bpm <= 0.0:
            continue
        tot += 1
        d = abs(math.log2(bpm / truth))
        d = abs(d - round(d))                      # fold to the nearest octave
        if d <= math.log2(1.0 + tol_pct / 100.0):
            hits += 1
    return hits / max(tot, 1)


def summarise(name, traj, truth, drift_pct=3.0):
    dur = traj[-1][0] if traj else 0.0
    rr = rereads(traj, drift_pct)
    locked = [s for s in traj if s[2]]
    fam = Counter(r[3] for r in rr)
    bpms = [s[1] for s in locked if s[1] > 0.0]
    conf = [s[3] for s in traj]
    print(f"  {name:<6} re-reads {len(rr):>4}"
          f"  ({len(rr) / max(dur, 1e-9) * 60.0:>5.1f}/min)"
          f"   locked {len(locked) / max(len(traj), 1) * 100:>5.1f}%"
          f"   median conf {np.median(conf) if conf else float('nan'):>5.2f}"
          f"   median bpm {np.median(bpms) if bpms else float('nan'):>6.1f}"
          f"   on-truth {on_truth(traj, truth) * 100:>5.1f}%")
    if fam:
        order = sorted(fam.items(), key=lambda kv: -kv[1])
        print("         families: "
              + "  ".join(f"{k} x{v}" for k, v in order))
    return len(rr), on_truth(traj, truth)


def main(dirs):
    for d in dirs:
        path = os.path.join(d, "human.wav")
        if not os.path.exists(path):
            print(f"{d}: no human.wav — skipped")
            continue
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        ta = replay(x, sr, LegacyTracker)[0]
        tb = replay(x, sr, BeatTracker)[0]
        truth = truth_bpm([s[1] for s in ta] + [s[1] for s in tb])
        print(f"\n{d}  ({len(x) / sr:.0f} s @ {sr} Hz)   "
              f"truth (octave-folded median) {truth:.1f} bpm")
        a, oa = summarise("v1.3", ta, truth)
        b, ob = summarise("v2.7", tb, truth)
        if a:
            print(f"         => re-reads {a} -> {b} "
                  f"({(a - b) / a * 100:+.0f}%)   "
                  f"on-truth {oa * 100:.1f}% -> {ob * 100:.1f}% "
                  f"({(ob - oa) * 100:+.1f} pts)")


if __name__ == "__main__":
    args = sys.argv[1:] or ["takes6/"]
    main(args)
