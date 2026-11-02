"""Re-measure the relock bias against a RECORDED take.

Not a test — the measurement behind ``RELOCK_HOLD_BARS`` and the "honest
scope" paragraph in ``listen/beat.py``, kept runnable so the numbers in that
comment can be checked instead of believed.  Run with::

    .venv/bin/python tests/relock_replay.py takes5/          # A/B, default sweep
    .venv/bin/python tests/relock_replay.py takes5/ 8 16 32  # explicit sweep

It cannot be a test and must never become one: it reads a jam recording, and
``*.wav`` is gitignored (SPEC-V1 amendment 21 — no test may read a gitignored
artifact, because a test that silently does not run is not a test).  Point it
at any ``plume jam --record`` directory; it needs only ``human.wav``.

What it reports, per configuration:

- **locked %** — how much of the take the tracker held a lock at all.
- **unlocked wander** — while UNLOCKED, how far (in octaves) the reported
  tempo sits from the tempo held at the last lock.  This is the number that
  matters downstream: ``MindParams.entry_conf`` gates the sideman's entry on
  CONFIDENCE, not on lock, so the mind enters and comps on the tempo shown
  here whether or not the tracker is locked.
- **episodes** — every loss -> relock pair, with the ratio between them.

The 2026-08-27 run over ``takes5/`` (944 s of open-tuned acoustic guitar) is
what SPEC-V1 amendment 47 records: wander 0.279 -> 0.200 octaves from bias off
to 32 bars, the two relocks unmoved at 0.88x and 1.22x, and locked % unmoved
at 13.7.
"""
import math
import os
import sys

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from plume.listen import beat as beat_mod                       # noqa: E402
from plume.listen.beat import BeatTracker                       # noqa: E402
from plume.listen.features import FeatureExtractor              # noqa: E402

BLOCK = 4096


def replay(x, sr, *, hold, bias=True, acoustic=True):
    """One pass of the real listening stack.  Restores the constants it moves."""
    keep = (beat_mod.RELOCK_BREAK_BARS, beat_mod.RELOCK_HOLD_BARS)
    beat_mod.RELOCK_BREAK_BARS = 0.0 if not bias else keep[0]
    beat_mod.RELOCK_HOLD_BARS = hold
    try:
        fe, tr = FeatureExtractor(sr), BeatTracker(acoustic=acoustic)
        traj = []
        for i in range(0, len(x), BLOCK):
            frames = fe.process(x[i:i + BLOCK])
            if frames:
                st = tr.process(frames)
                traj.append((st.t_s, st.bpm, st.locked))
        return traj
    finally:
        beat_mod.RELOCK_BREAK_BARS, beat_mod.RELOCK_HOLD_BARS = keep


def score(traj):
    """(locked_frac, wander_median_oct, within_frac, [(loss_t, loss_bpm, relock_t, relock_bpm)])."""
    eps, dev = [], []
    pend = ref = None
    prev_locked, prev_bpm = False, 0.0
    for t, bpm, locked in traj:
        if prev_locked and not locked:
            pend, ref = (t, prev_bpm), prev_bpm
        elif not prev_locked and locked and pend is not None:
            eps.append((pend[0], pend[1], t, bpm))
            pend = None
        if not locked and ref and bpm > 0.0:
            dev.append(abs(math.log2(bpm / ref)))
        prev_locked, prev_bpm = locked, bpm
    dev = np.asarray(dev) if dev else np.zeros(1)
    locked_frac = sum(1 for _, _, l in traj if l) / max(1, len(traj))
    return locked_frac, float(np.median(dev)), float((dev <= 0.15).mean()), eps


def main(argv):
    take = argv[1] if len(argv) > 1 else "takes5/"
    holds = [float(a) for a in argv[2:]] or [8.0, 16.0, 32.0, 64.0]
    path = os.path.join(take, "human.wav") if os.path.isdir(take) else take
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    print(f"{path} — {len(x) / sr:.1f}s @ {sr}\n")
    print(f"{'config':>14} | {'locked':>7} | {'wander':>7} | {'within±11%':>10} | episodes")
    for name, kw in [("bias off", dict(hold=holds[0], bias=False))] + \
                    [(f"hold {h:g} bars", dict(hold=h)) for h in holds]:
        lf, med, within, eps = score(replay(x, sr, **kw))
        print(f"{name:>14} | {lf * 100:6.1f}% | {med:6.3f}o | {within * 100:9.1f}% | {len(eps)}")
        for lt, lb, rt, rb in eps:
            print(f"{'':>14}   loss {lt:6.1f}s {lb:6.1f} -> relock {rt:6.1f}s {rb:6.1f}"
                  f"  (gap {rt - lt:5.1f}s, ratio {rb / lb:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
