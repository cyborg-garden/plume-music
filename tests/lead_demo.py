"""Regenerate the v2 flagship pair: renders/lead_demo.* and renders/trade_demo.*

Not a test — a reproducible generator for the committed demo artifacts, in the
same shape as ``tests/duet_qa.py``.  Run with::

    .venv/bin/python tests/lead_demo.py

Produces, per demo:

    renders/<name>.wav        the mix (machine + human)
    renders/<name>.ndjson     the decision log, one object per machine bar
    renders/<name>.json       the manifest + a measured summary

SPEC-V2 section 8 asks for exactly these two:

``lead_demo``
    "machine leads a 120bpm trap session, human joins at bar 8 rushing" — and
    since v2.2 the machine HOLDS rather than concedes, because a rhythm section
    that chases a push stops being a reference and walks away from its partner
    (docs/decisions/2026-08-15-tempo-authority.md).
``meet_demo``
    the identical take pinned to ``tempo_policy="meet"``, i.e. SPEC-V2's
    original sentence, kept so the two laws can be heard side by side.
``trade_demo``
    the same machine, ``role="trade"``, against a human who plays over the
    machine's bars so the human-wins collision rule is exercised.

REPRODUCIBLE FROM A CLEAN CLONE.  Every input is SYNTHESIZED here —
``*.wav`` is gitignored and no WAV is tracked, so a demo that depended on an
audio file on one laptop would be a number nobody else could check.  The human
is ``tests/fixtures_gen.make_joiner``, a pure function of kind/bpm/key/bars/
seed; the machine's arrangement is v0 ``compose``, a pure function of its
prompt spec and seed.  Same inputs, byte-identical outputs.

The summary block is MEASURED off the rendered file, not copied out of the
decision log, so a demo whose log and audio disagree fails here rather than
shipping.
"""
import json
import math
import os
import sys

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

from fixtures_gen import make_joiner                            # noqa: E402
from plume.compose import PromptSpec, compose                   # noqa: E402
from plume.mind import Cue                                      # noqa: E402
from plume.roles import RoleParams                              # noqa: E402
from plume.sim import simulate_lead                             # noqa: E402

R = os.path.join(REPO, "renders")
os.makedirs(R, exist_ok=True)
SR = 48000

#: The two demos, as data.  These numbers ARE SPEC-V2 section 8's sentence;
#: keeping them together is what stops the artifact and the definition of done
#: from drifting apart.
DEMOS = {
    "lead_demo": {
        "prompt": PromptSpec(styles=["trap"], bpm=120.0, key="A minor",
                             movements=1, mood=[], name="lead-demo"),
        "joiner": "rushing", "join_bar": 8, "bars": 32, "human_bars": 32,
        "role": RoleParams(role="lead"),
        "cues": [],
        "what": ("the machine leads a 120bpm trap session; the human joins at "
                 "bar 8 pushing 30ms in front, and the machine HOLDS the "
                 "tempo (v2.2 default) — the reference does not move, the "
                 "push stays audible, and the human can come back to it"),
    },
    # The same take under the old law, kept as a demo because conceding is
    # still a real capability — and because hearing them back to back is the
    # clearest statement of what the dial does.
    "meet_demo": {
        "prompt": PromptSpec(styles=["trap"], bpm=120.0, key="A minor",
                             movements=1, mood=[], name="meet-demo"),
        "joiner": "rushing", "join_bar": 8, "bars": 32, "human_bars": 32,
        "role": RoleParams(role="lead", tempo_policy="meet"),
        "cues": [],
        "what": ("the identical take with tempo_policy=meet: the machine "
                 "concedes toward the rushing human instead of holding"),
    },
    "trade_demo": {
        "prompt": PromptSpec(styles=["trap"], bpm=120.0, key="A minor",
                             movements=1, mood=[], name="trade-demo"),
        "joiner": "takes_over", "join_bar": 0, "bars": 32, "human_bars": 24,
        "role": RoleParams(role="trade", trade_bars=4),
        "cues": [(40.0, Cue("take it out", at="next-phrase"))],
        "what": ("trading fours: authority alternates every 4 bars at the "
                 "boundary, the human playing over the machine's bars makes "
                 "it back off, and a cue takes the tune out"),
    },
}
SEED = 5


def _mono(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(axis=1).astype(np.float64), int(sr)


def _bar_rms_db(res, x, sr):
    ends = list(res.bar_times_s[1:]) + [
        res.bar_times_s[-1] + 240.0 / res.machine_bpm_by_bar[-1]]
    out = []
    for t0, t1 in zip(res.bar_times_s, ends):
        seg = x[int(t0 * sr):int(t1 * sr)]
        v = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
        out.append(round(20.0 * math.log10(max(v, 1e-10)), 2))
    return out


def build(name: str, spec: dict) -> dict:
    session = compose(spec["prompt"], seed=SEED)
    audio, truth = make_joiner(spec["joiner"], machine_bpm=session.bpm,
                               machine_key=str(session.key),
                               bars=spec["human_bars"], seed=1)
    human_path = os.path.join(R, f"{name}_input.wav")
    sf.write(human_path, audio, SR, subtype="PCM_24")

    join_s = spec["join_bar"] * 240.0 / session.bpm
    res = simulate_lead(
        human_path, session=session, seed=SEED,
        out_path=os.path.join(R, f"{name}.wav"),
        decisions_path=os.path.join(R, f"{name}.ndjson"),
        role=spec["role"], bars=spec["bars"], human_offset_s=join_s,
        cues=spec["cues"] or None)

    x, sr = _mono(res.out_path)
    bar_db = _bar_rms_db(res, x, sr)
    ev = [(int(r["bar"]), e) for r in res.role_states for e in r["events"]]
    pcts = [r["tempo_offset_pct"] for r in res.role_states]
    k = len(res.bar_times_s) - 1
    gap_beats = ((res.bar_times_s[k] - k * 240.0 / session.bpm)
                 / (60.0 / session.bpm))

    doc = {
        "what": spec["what"],
        "cmd": (f"plume simlead renders/{name}_input.wav "
                f"--role {spec['role'].role} --seed {SEED} "
                f"--bars {spec['bars']} --human-offset {join_s:.3f} "
                f"--out renders/{name}.wav"),
        "input_from": (f"tests/lead_demo.py: make_joiner("
                       f"{spec['joiner']!r}, machine_bpm={session.bpm}, "
                       f"machine_key={str(session.key)!r}, "
                       f"bars={spec['human_bars']}, seed=1) -> "
                       f"renders/{name}_input.wav (gitignored; regenerate "
                       f"with .venv/bin/python tests/lead_demo.py)"),
        "human_truth": {k2: truth[k2] for k2 in
                        ("kind", "bpm", "key", "mean_offset_ms", "bars")
                        if k2 in truth},
        "manifest": res.manifest,
        "measured": {
            "bars_rendered": len(res.bar_times_s),
            "duration_s": round(res.duration_s, 4),
            "machine_bus_rms_db_by_bar": bar_db,
            "bpm_by_bar": [round(b, 3) for b in res.machine_bpm_by_bar],
            "tempo_offset_pct_extremes": [round(min(pcts), 4),
                                          round(max(pcts), 4)],
            "role_events": ev,
            "roles_by_bar": [r["role"] for r in res.role_states],
            "final_phase_gap_beats": round(gap_beats, 4),
            "states_by_bar": list(res.states_by_bar),
        },
    }
    with open(os.path.join(R, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True, default=str)
    return doc


def main():
    for name, spec in DEMOS.items():
        doc = build(name, spec)
        m = doc["measured"]
        print(f"{name}: {m['bars_rendered']} bars, "
              f"{m['duration_s']:.2f}s, tempo "
              f"{m['tempo_offset_pct_extremes'][0]:+.2f}..."
              f"{m['tempo_offset_pct_extremes'][1]:+.2f}%, "
              f"ends {m['final_phase_gap_beats']:+.2f} beats from the human")
        print(f"  events: {m['role_events']}")
        print(f"  -> renders/{name}.wav / .ndjson / .json")


if __name__ == "__main__":
    main()
