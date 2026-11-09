"""Replay her chords (plume.herchords) over RECORDED takes: does the band's chord fit hers?

Not a test, and it must never become one: it reads a take's ``human.wav``, which is
gitignored (SPEC-V1 amendment 21: no test may read a gitignored artifact).  It is the
measurement behind the thresholds in ``herchords.py``, kept runnable so they can be
checked instead of believed.  Run with::

    PYTHONPATH=src .venv/bin/python tests/chords_replay.py ~/Music/Plume/<take>/ [...]

It needs ``human.wav`` and ``decisions.ndjson``.  The listening is the live path's
own: FeatureExtractor chroma, bar N planned from bar N-2, ``playing`` as the take
logged it, the key, style and template chord as the take logged them.

The score is the roadmap's: on each bar she played (above -45 dB) with the band in,
the share of her bar's pitch energy on the band's chord tones (root to seventh),
from a plain FFT chroma independent of the one the follower listens with.  Four
notes of twelve hold 0.33 by chance.  "before" scores the chord the take played,
"after" the chord this code would have played.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from plume import herchords as hc                                # noqa: E402
from plume.listen.features import FeatureExtractor               # noqa: E402
from plume.theory import parse_chord                             # noqa: E402

BLOCK = 4096
PLAYED_DB = -45.0


def fft_chroma(x, sr):
    """A plain spectrum folded to pitch classes, 70 Hz to 2 kHz."""
    n = 8192
    acc = np.zeros(12)
    for i in range(0, len(x) - n, n // 2):
        mag = np.abs(np.fft.rfft(x[i:i + n] * np.hanning(n)))
        f = np.fft.rfftfreq(n, 1 / sr)
        m = (f > 70) & (f < 2000)
        pc = np.round(69 + 12 * np.log2(f[m] / 440)).astype(int) % 12
        np.add.at(acc, pc, mag[m] ** 2)
    return acc / (acc.sum() + 1e-12)


def tones4(sym):
    ch = parse_chord(sym)
    return [(ch.root + i) % 12 for i in ch.intervals[:4]]


def replay(take, *, verbose=False):
    rows = [json.loads(line) for line in open(os.path.join(take, "decisions.ndjson"))]
    rows = [r for r in rows if isinstance(r.get("bar"), int)]
    x, sr = sf.read(os.path.join(take, "human.wav"), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    fe = FeatureExtractor(sr=sr)
    frames = []
    for i in range(0, len(x), BLOCK):
        frames.extend((f.t_s, f.chroma, f.chroma_low) for f in fe.process(x[i:i + BLOCK]))
    by_bar = {r["bar"]: r for r in rows}
    starts = {r["bar"]: r["t_s"] for r in rows}
    follower = hc.ChordFollower()
    out = []
    for r in rows:
        bar = r["bar"]
        heard = bar - 2
        if heard in starts and heard + 1 in starts:
            w0, w1 = starts[heard], starts[heard + 1]
            bar_frames = [(c, lo) for t, c, lo in frames if w0 <= t < w1]
            follower.hear(heard, hc.clean_bar(bar_frames), playing=bool(r.get("playing")))
        if not r.get("key") or not r.get("chord") or bar + 1 not in by_bar:
            continue
        d = follower.decide(bar, r["key"], r.get("style") or "lofi", r["chord"])
        seg = x[int(r["t_s"] * sr):int(by_bar[bar + 1]["t_s"] * sr)]
        if seg.size < 4096:
            continue
        rms = 20 * np.log10(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-12)
        if rms < PLAYED_DB:
            continue
        c = fft_chroma(seg, sr)
        before = float(c[tones4(r["chord"])].sum())
        after = float(c[tones4(d["chord"])].sum())
        out.append((bar, r["chord"], d["chord"], d["source"], before, after))
        if verbose:
            print(f"  bar {bar:3d} {r['chord']:>8} {before:.2f} -> {d['chord']:>8} "
                  f"{after:.2f}  {d['source']:<9} {d['reason']}")
    return out


def main(argv):
    verbose = "-v" in argv
    takes = [a for a in argv if a != "-v"]
    allb = []
    for take in takes:
        res = replay(take, verbose=verbose)
        if not res:
            continue
        b = np.mean([r[4] for r in res])
        a = np.mean([r[5] for r in res])
        hers = sum(r[3] != "template" for r in res)
        print(f"{os.path.basename(os.path.normpath(take))}: {len(res)} bars, "
              f"fit {b:.3f} -> {a:.3f}, followed her on {hers}")
        allb.extend(res)
    if not allb:
        print("no bars with the band in and her playing")
        return
    b = np.array([r[4] for r in allb])
    a = np.array([r[5] for r in allb])
    hers = np.array([r[3] != "template" for r in allb])
    print(f"ALL: {len(allb)} bars, mean fit {b.mean():.3f} -> {a.mean():.3f}; "
          f"followed her on {hers.sum()} bars ({hers.mean():.0%})")
    if hers.any():
        print(f"  on the bars it followed: {b[hers].mean():.3f} -> {a[hers].mean():.3f}; "
              f"worst after {a[hers].min():.2f} (template there {b[hers].min():.2f} worst); "
              f"worse than the template on {(a[hers] < b[hers] - 1e-9).sum()} bars")


if __name__ == "__main__":
    main(sys.argv[1:])
