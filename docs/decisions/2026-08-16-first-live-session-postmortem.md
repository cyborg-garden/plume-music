# Postmortem: the first session with a real guitar

**Date:** 2026-08-15/16 · **Player:** Juniper, acoustic guitar in CGCFCE
(open tuning: heavy sustain, ringing open strings, weak note-to-note
transients) · **Rig:** laptop mic in, laptop speakers out for the first third,
**headphones for the last two thirds** · **Duration:** ~6 minutes ·
**Artifacts:** none — see finding 1.

## What the human reported

> "it only plays like two random measures here and there. and then stops
> suddenly. and then starts up randomly later."

Also asked for: softer, crossfaded entrances ("music is gentle"); and that the
machine be **more opinionated** — "it cant chicken out. if it does really
commit to something it has to also give the other musician a chance to adapt to
that as well."

## Finding 1 (P0) — plume cannot account for its own behaviour

`--record DIR` wrote **nothing** during the session (the directory was empty at
the end) because records are buffered and only written on graceful exit. Then
`plume stop` could not stop it — socket, SIGINT and SIGTERM were all ignored —
so it took `kill -9`, and the take was lost.

The only surviving evidence is 1080 samples of the stdout status line. **Every
single one reads `listening`.** The strings `entering`, `comping`, `thinned`,
`fading`, `tacet` appear zero times. By its own account the machine never
played a note, while the human in the room heard it play repeatedly.

So either the status line does not reflect what the planner actually rendered,
or audio reached the speakers from outside the state machine. **We cannot tell
which, and that is the finding.** Until records stream, every diagnosis of live
behaviour is speculation — including the two confident ones below that turned
out to be wrong.

## Finding 2 (P0) — the tracker cannot hold a grid on this material

Tempo estimate by third of the session:

| | bpm range | distinct values (rounded) | mean confidence |
|---|---|---|---|
| first (speakers) | 70.8 – 200.9 | 71, 74, 80, 92, 104, 111, 116, 118 … | 0.30 |
| middle (speakers) | 66.2 – 167.9 | 66, 78, 95, 109, 115, 120, 151 … | 0.44 |
| **last (headphones)** | **80.3 – 187.5** | **80, 85, 121, 126, 141, 154, 165 …** | **0.41** |

It never settled, and it was **no better on headphones**. Confidence sat around
0.41 — below the 0.55 lock threshold, above nothing useful. Sustained,
drone-heavy, open-tuned acoustic guitar through a laptop mic is simply outside
what an onset-flux tracker tuned on programmed material can follow.

## Two diagnoses I got wrong, and why

- **"Self-masking is broken, so plume was tracking its own output."** Verification
  did find the self-masking inoperative, and it would matter on speakers — but
  the human had headphones on for two thirds of the session and the tempo was
  just as unstable there. Real bug, wrong culprit.
- **"It was flapping in and out of playing."** The mind-side commitment work is
  good and its evidence is sound *on a reconstructed trace*. But this session
  never entered at all, so nothing flapped. A fix for a real defect, credited to
  the wrong symptom.

Both errors have the same shape: **inferring live behaviour from a system whose
live reporting is broken**, then stating the inference with more confidence than
the evidence carried. The user corrected both from the room.

## What actually follows

1. **Observability first.** Stream records; write a manifest at start; make the
   status line and the planner read one source of truth, and add a test that
   fails if the reported state ever disagrees with what was rendered.
2. **Then re-run the session and look**, rather than reasoning about it.
3. **The count-in is not a workaround.** Given finding 2, asking a human to
   count the band in is the correct design for this material — it is what
   musicians do — and it converts the hardest DSP problem in the project into a
   solved interaction problem.
4. Keep the commitment, fades and announce work: independently justified, and
   all three answer things the player actually asked for.
