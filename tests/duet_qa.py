"""Regenerate renders/duet_demo.* and the 6 duet QA takes, and re-run the QA.

Not a test — a reproducible generator for the committed demo artifacts.
Run with:  .venv/bin/python tests/duet_qa.py

Reproduces the committed layout:
  renders/duet_demo.{wav,ndjson,json,qa.json}
  renders/duet_<kind>_<style>.{wav,ndjson,sim.json,qa.json}
  renders/qa/<name>.machine.{wav,ndjson}
  renders/duet_qa_report.json

REPRODUCIBLE FROM A CLEAN CLONE (v1.3).  Every input this script needs is
SYNTHESIZED here — see ``make_flagship_input`` — because ``*.wav`` is
gitignored and no WAV is tracked.  Before v1.3 the flagship block of the
committed ``renders/duet_qa_report.json`` was built from
``renders/duet_demo_input.wav``, a file that existed on exactly one laptop:
the committed numbers could be read but never checked or regenerated.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import fixtures_gen as fg                                       # noqa: E402
from plume import sim as sim_mod                                # noqa: E402
from plume.listen.features import FeatureExtractor              # noqa: E402
from plume.theory import parse_chord, parse_key                 # noqa: E402

R = os.path.join(REPO, "renders")
QA = os.path.join(R, "qa")
os.makedirs(QA, exist_ok=True)
PC = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
SEED = 5
SR = 48000

# --- the flagship demo input (SPEC-V1 section 10) ---------------------------
# "a 24-bar steady strummer in F minor" — 60.000 s of mono 48k.  These four
# numbers ARE the spec sentence; keep them together so the artifact and the
# definition of done cannot drift apart.
FLAGSHIP = {"kind": "steady", "bpm": 96, "bars": 24, "key": "F minor",
            "seed": 1}


def make_flagship_input(path: str | None = None) -> str:
    """Synthesize the flagship demo human and write it as PCM_24 mono.

    Pure function of ``FLAGSHIP`` (fixtures_gen is itself a pure function of
    kind/bpm/bars/key/seed), so a clean clone regenerates the same input and
    can therefore actually verify ``renders/duet_qa_report.json``.  The WAV
    stays out of git — it is 8 MB of derivable data.

    HONEST CAVEAT: this is NOT byte-identical to the pre-v1.3
    ``renders/duet_demo_input.wav``.  That file's length (2 880 000 samples),
    sample rate, channel count and 0.85 peak match this generator exactly and
    its chroma matches F minor to three decimals, so it was certainly made
    this way — but fixtures_gen has moved since, and the original seed was
    never written down.  Which is the whole point: an artifact with no
    generator cannot be reproduced, only guessed at.  The committed QA numbers
    therefore changed when this landed; they are now checkable.
    """
    path = path or os.path.join(R, "duet_demo_input.wav")
    x = fg.make_human(FLAGSHIP["kind"], bpm=FLAGSHIP["bpm"],
                      bars=FLAGSHIP["bars"], key=FLAGSHIP["key"],
                      seed=FLAGSHIP["seed"])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, x, SR, subtype="PCM_24")
    return path


def _db(x):
    return float(20.0 * np.log10(max(float(x), 1e-12)))


def _mono(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    return x.mean(axis=1), sr


def _bass_pc(seg, sr, hi_hz=400.0):
    """Pitch class the machine is putting UNDER the human, for one bar.

    Measured with the same harmonic pitch salience the machine itself listens
    with (FeatureExtractor chroma), over a low-passed copy of the machine bus.
    A raw low-band spectral argmax is NOT good enough here: the 808 sub has a
    60ms exponential glide and the kick is a broadband pitched thud, so the
    strongest single bin can sit half a semitone off the note actually played
    (measured 40.28Hz on a bar whose note is Bb1 -- between Eb1 and E1, and
    neither).  Chroma integrates the whole bar and its partials instead.
    """
    if seg.size < 8192:
        return None, -120.0, 0.0
    n = 1 << int(np.ceil(np.log2(seg.size)))
    spec = np.fft.rfft(seg, n)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    spec[f > hi_hz] = 0.0
    low = np.fft.irfft(spec, n)[: seg.size].astype(np.float32)
    frames = FeatureExtractor(sr=int(sr)).process(low)
    voiced = [fr.chroma for fr in frames if fr.chroma.sum() > 0]
    lvl = _db(np.sqrt(np.mean(seg ** 2)))
    if not voiced:
        return None, -120.0, 0.0
    ch = np.mean(voiced, axis=0)
    return int(np.argmax(ch)), round(lvl, 1), round(float(ch.max()), 3)


# --- QA gate constants (each one is a claim; see the comments) --------------
# A bar whose HUMAN level is under this is a stop, not a balance measurement:
# machine-minus-human is meaningless when the human is digital silence.  The
# `stops` fixture writes exact zeros, which measured -120 dB and produced
# "later_comping_delta_db" entries of +62.0 and +224.2 dB.
_HUMAN_PLAYING_DB = -60.0
# Mean |machine - human| across the settled comping bars.  Entering 8 dB under
# and converging to within 4 dB is the SPEC-V1 section 4 arc (enter_under_db
# -8.0, ramp_bars 3).
_CONVERGE_DB = 4.0
# bass_roots vacuity floor.  Scoring only the bars whose low-register chroma
# has a clear winner is right (see 4b), but with NO floor on how many that is,
# a take with ONE decisive bar scored 1/1 and passed.  Two floors:
#
#   * _MIN_DECISIVE_BARS = 4 — the PRINCIPLED one, and the one that actually
#     binds.  fixtures_gen cycles a 4-chord progression (`chords[b % 4]`), so
#     one full harmonic cycle is the smallest sample in which a wrong root
#     could show up at all.  The number comes from the fixture's harmony, not
#     from the observed pass/fail line.
#
#   * _MIN_DECISIVE_FRAC = 0.20 — a COLLAPSE DETECTOR for long takes, not a
#     quality bar, and deliberately set low.  Being honest about why: the
#     synthwave takes legitimately measure only ~0.35-0.41 decisive (their
#     sub-bass is sustained and glides, so a bar's low-register chroma often
#     has no clear winner), and the fraction MOVES WITH THE TRACKER — while
#     the listen stack was being changed under this script, duet_steady_
#     synthwave was measured at 5/17 = 0.294 and then back at 6/17 = 0.353
#     from the same fixture.  A fraction floor set anywhere near passing
#     material would therefore fail on tracker work that is not a regression
#     in the bass at all.  That the bass measurement is this weak on
#     sustained sub voices is itself the finding, which is why
#     `decisive_frac` is now reported on every take instead of being implied.
#     At 17 bars this floor needs 4 bars — i.e. it is non-binding here and
#     only starts to bite past ~20 bars, which is exactly its job.
_MIN_DECISIVE_BARS = 4
_MIN_DECISIVE_FRAC = 0.20


def run_one(name, human_wav, style, truth):
    out = os.path.join(R, f"{name}.wav")
    res = sim_mod.simulate(human_wav, style=style, out_path=out, seed=SEED)
    # machine-only bus: same run with the human muted in the mix
    mout = os.path.join(QA, f"{name}.machine.wav")
    mres = sim_mod.simulate(human_wav, style=style, out_path=mout, seed=SEED,
                            mix_human_db=-200.0,
                            decisions_path=os.path.join(QA, f"{name}.machine.ndjson"))
    lines = [json.loads(ln) for ln in
             open(res.decisions_path, encoding="utf-8").read().splitlines()]

    mach, sr = _mono(mout)
    hum, _ = _mono(human_wav)
    bpm = res.manifest["final_bpm"] or truth["bpm"]
    bar_s = 4.0 * 60.0 / bpm

    entering = [ln for ln in lines if ln.get("state") == "entering"]
    entry_t = entering[0]["t_s"] if entering else None
    checks = {}

    # 1. silent before entry
    if entry_t:
        pre = mach[: int(entry_t * sr)]
        frames = pre[: (pre.size // 480) * 480].reshape(-1, 480) if pre.size >= 480 \
            else np.zeros((1, 480))
        checks["silent_before_entry"] = {
            "pass": bool(_db(np.sqrt(np.mean(pre ** 2))) < -60.0),
            "machine_pre_rms_db": round(_db(np.sqrt(np.mean(pre ** 2))), 1),
            "machine_pre_max_10ms_frame_db":
                round(float(np.max([_db(np.sqrt(np.mean(f ** 2))) for f in frames])), 1),
        }
        # 2. entry lands on a bar boundary
        idx = np.argmax(np.abs(mach) > 10 ** (-60 / 20.0))
        first_audible = round(float(idx) / sr, 2)
        near = round(entry_t / bar_s) * bar_s
        checks["entry_on_boundary"] = {
            "pass": bool(abs(first_audible - entry_t) < 0.12),
            "first_audible_s": first_audible,
            "planned_entry_t_s": round(entry_t, 4),
            "nearest_truth_bar_s": round(near, 3),
            "err_ms": round(abs(first_audible - entry_t) * 1000.0, 1),
        }

        # 3. enters under the human, then converges
        def bar_db(sig, k):
            a, b = int((entry_t + k * bar_s) * sr), int((entry_t + (k + 1) * bar_s) * sr)
            s = sig[a:b]
            return round(_db(np.sqrt(np.mean(s ** 2))), 1) if s.size else -120.0

        m0, h0 = bar_db(mach, 0), bar_db(hum, 0)
        # Convergence is measured only over bars where the human is PLAYING.
        # A stop bar is human digital silence, so machine-minus-human there
        # is a division-by-silence artefact, not a balance: the `stops` takes
        # reported +62.0 and +224.2 dB.  Excluding them is what lets this be
        # gated at all — before v1.3 `converges` was computed, was False on
        # both `stops` takes, and was then left out of `pass`, so a metric
        # nobody could fail sat in the report looking like a check.
        n_later = int((len(mach) / sr - entry_t) / bar_s)
        later, skipped = [], []
        for k in range(3, n_later):
            h = bar_db(hum, k)
            if h <= _HUMAN_PLAYING_DB:
                skipped.append({"bar_after_entry": k, "human_bar_db": h})
                continue
            later.append(round(bar_db(mach, k) - h, 1))
        converges = bool(later and abs(np.mean(later)) < _CONVERGE_DB)
        checks["enter_under"] = {
            "pass": bool(m0 - h0 < -4.0) and converges,
            "machine_entry_bar_db": m0, "human_entry_bar_db": h0,
            "delta_db": round(m0 - h0, 1),
            "later_comping_delta_db": later,
            "later_mean_delta_db": round(float(np.mean(later)), 2) if later else None,
            "bars_skipped_human_silent": skipped,
            "converges": converges,
        }

        # 4. bass roots: in key, and matching the machine's own chord belief
        # 4a. every CHORD BELIEF the machine voices from must be in key.
        #     This is the defect that put a Db bass under the human's Am: the
        #     belief itself was hallucinated (HarmonyTracker labelled the
        #     human's Am bars "Dbm"), and the machine played what it believed.
        kobj = parse_key(truth["key"])
        pcs = set(kobj.scale())
        beliefs, bad_beliefs = [], []
        for ln in lines:
            if ln.get("state") not in ("entering", "comping", "thinned"):
                continue
            sym = ln.get("chord")
            if not sym:
                continue
            try:
                root = parse_chord(sym).root % 12
            except ValueError:
                continue
            beliefs.append(sym)
            if root not in pcs:
                bad_beliefs.append({"bar": ln["bar"], "chord": sym})
        checks["chord_belief_in_key"] = {
            "pass": not bad_beliefs,
            "bars_checked": len(beliefs),
            "out_of_key": bad_beliefs,
            "distinct": sorted(set(beliefs)),
        }

        # 4b. and the pitch class actually SOUNDING in the bass register must
        #     be in key, on every bar where the measurement is decisive.  Bars
        #     whose low-register chroma has no clear winner are reported but
        #     not scored: on the `drift` fixtures the machine's bar spans a
        #     human chord change, so two bass notes and their tails share the
        #     bar and no single pitch class is "the" note.
        decisive_min = 0.30
        detail, in_key, decisive, match = [], 0, 0, 0
        soft = []          # non-decisive measurements, surfaced not dropped
        for ln in lines:
            if ln.get("state") not in ("entering", "comping", "thinned"):
                continue
            t0 = ln["t_s"]
            a, b = int(t0 * sr), int((t0 + bar_s) * sr)
            pc, lvl, conf = _bass_pc(mach[a:b], sr)
            if pc is None:
                continue
            ok = pc in pcs
            belief = ln.get("chord")
            broot = None
            if belief:
                try:
                    broot = parse_chord(belief).root % 12
                except ValueError:
                    broot = None
            if conf >= decisive_min:
                decisive += 1
                in_key += ok
                match += (broot is not None and broot == pc)
            elif not ok:
                soft.append({"bar": ln["bar"], "pc": PC[pc],
                             "chroma_conf": conf, "level_db": lvl,
                             "belief_chord": belief})
            detail.append({"bar": ln["bar"], "pc": PC[pc], "level_db": lvl,
                           "chroma_conf": conf, "decisive": conf >= decisive_min,
                           "belief_chord": belief, "in_key": bool(ok)})
        frac = decisive / len(detail) if detail else 0.0
        enough = (decisive >= _MIN_DECISIVE_BARS
                  and frac >= _MIN_DECISIVE_FRAC)
        checks["bass_roots"] = {
            # Two independent conditions, both required: the sample is big
            # enough to mean something (`enough`) AND every bar in it is in
            # key.  `decisive and in_key == decisive` alone passed on a
            # sample of one.
            "pass": bool(enough and in_key == decisive),
            "bars_checked": len(detail), "bars_decisive": decisive,
            "decisive_frac": round(frac, 3),
            "enough_decisive": bool(enough),
            "min_decisive_bars": _MIN_DECISIVE_BARS,
            "min_decisive_frac": _MIN_DECISIVE_FRAC,
            "in_key": in_key, "match_belief_root": match,
            # Bars too smeared to score, whose (weak) winner is nevertheless
            # out of key.  NOT gated — below the salience floor the winner is
            # a coin flip between two overlapping bass tails, and on `drift`
            # the machine's bar genuinely spans a human chord change.  But
            # they are reported, because "we dropped 11 of 17 bars and 3 of
            # them measured a Bb in A minor" is a fact a reader of this
            # report is entitled to, and a rise here is the early warning
            # that the decisive bars are about to go out of key too.
            "non_decisive_out_of_key": soft,
            "detail": detail,
        }

    qa = {"name": name, "checks": checks,
          "pass_all": all(c.get("pass") for c in checks.values()) and bool(checks)}
    with open(os.path.join(R, f"{name}.qa.json"), "w", encoding="utf-8") as fh:
        json.dump(qa, fh, indent=1, sort_keys=True)
    return res, qa, mres


def _print_take(name, res, qa):
    """One line per take. Prints the non-decisive bass measurements too —
    they used to be computed and then silently dropped."""
    br = qa["checks"].get("bass_roots", {})
    cb = qa["checks"].get("chord_belief_in_key", {})
    eu = qa["checks"].get("enter_under", {})
    soft = br.get("non_decisive_out_of_key", [])
    print(f"{name:26s} entered={res.entered_bar} key={res.final_key} "
          f"bpm={res.manifest['final_bpm']} "
          f"belief_in_key={cb.get('pass')} "
          f"bass_in_key={br.get('in_key')}/{br.get('bars_decisive')} "
          f"(dec {br.get('decisive_frac')} of {br.get('bars_checked')}) "
          f"converges={eu.get('converges')} "
          f"pass_all={qa['pass_all']}")
    if soft:
        print(f"{'':26s}   non-decisive & out of key: "
              + ", ".join(f"bar {s['bar']} {s['pc']} (conf {s['chroma_conf']})"
                          for s in soft))


def main():
    import dataclasses
    report = {"duets": []}

    # --- flagship demo ------------------------------------------------------
    # The flagship demo input is the 24-bar steady strummer in F minor
    # (SPEC-V1 section 10), synthesized here rather than read off disk.
    demo_in = make_flagship_input()
    res, qa, _ = run_one("duet_demo", demo_in, "lofi",
                         {"bpm": float(FLAGSHIP["bpm"]), "key": FLAGSHIP["key"]})
    with open(os.path.join(R, "duet_demo.json"), "w", encoding="utf-8") as fh:
        json.dump(dataclasses.asdict(res), fh, indent=1, sort_keys=True, default=str)
    report["flagship"] = {
        "input": dict(FLAGSHIP),
        "input_from": ("tests/duet_qa.py: make_flagship_input() -> "
                       "renders/duet_demo_input.wav (gitignored, regenerate "
                       "with: .venv/bin/python tests/duet_qa.py)"),
        "cmd": "plume sim renders/duet_demo_input.wav --style lofi --seed 5",
        "entered_bar": res.entered_bar, "final_key": res.final_key,
        "final_bpm": res.manifest["final_bpm"], "pass_all": qa["pass_all"],
        "checks": {k: v.get("pass") for k, v in qa["checks"].items()},
    }
    report["duets"].append(qa)
    _print_take("duet_demo", res, qa)

    # --- the 6 QA takes -----------------------------------------------------
    for kind in ("steady", "drift", "stops"):
        x, truth = fg.make_human_with_truth(kind, bpm=96, bars=24,
                                            key="A minor", seed=1)
        hp = os.path.join(QA, f"fixture_{kind}.wav")
        sf.write(hp, x, SR, subtype="PCM_24")
        truth = dict(truth) if isinstance(truth, dict) else {}
        truth.setdefault("bpm", 96.0)
        truth.setdefault("key", "A minor")
        for style in ("lofi", "synthwave"):
            name = f"duet_{kind}_{style}"
            res, qa, _ = run_one(name, hp, style, truth)
            with open(os.path.join(R, f"{name}.sim.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(res.manifest, fh, indent=1, sort_keys=True, default=str)
            report["duets"].append(qa)
            _print_take(name, res, qa)

    report["all_pass"] = all(d["pass_all"] for d in report["duets"])
    with open(os.path.join(R, "duet_qa_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True, default=str)
    print("all_pass:", report["all_pass"])


if __name__ == "__main__":
    main()
