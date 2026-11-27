# Research Synthesis

> Historical research notes and build decisions, not current behaviour. For what Plume does today, read the README and `docs/PLAYING.md`.

Ten parallel research agents, 2026-08-13. This file is the distilled version: what
we learned and what we decided because of it. The raw findings are not published.

## The five findings that shape the architecture

**1. Predict, don't react.** Human ensemble asynchrony is ~30ms — too tight for
any react-to-what-you-hear loop. Humans do it with *anticipation*: extrapolate
the beat, schedule against the prediction, phase-correct gently (ADAM model;
ReaLJam's "anticipation protocol" rediscovered this for machines). So the
engine schedules events against a predicted beat grid, ahead of time, with
partial phase correction (gain α ≈ 0.25–0.5, exposed as a `tightness` knob).
This also makes decision latency — even an LLM in the loop — musically
invisible: commit the next bar early, revise until the deadline.

**2. Systematic micro-timing, not jitter.** Groove research (Friberg &
Sundström, Nelias 2022) killed the "random humanization" myth: swing ratio is
tempo-dependent (~3.5:1 slow → 1:1 fast, short eighth pinned ~100ms), soloists
delay *downbeats* ~30ms while locking offbeats, hats ride a few ms ahead.
Deterministic per-role, per-beat-position offset tables; random jitter ≤10ms
SD only. Groove = one MPC-style swing param (50–66%) + a timing-bias table.

**3. The systems that felt alive listened and had intent.** Voyager (the
option to *not* play reads as intent), Continuator (mirroring the player's own
material = "it plays like me"), Somax2 (continuous co-improvisation from
multi-layer listening). The ones that felt mechanical (Band-in-a-Box, Lyria
RealTime) never listen. But B-i-a-B's lesson counts too: an unshakeable clock
is a feature. Behavior states (comp / lay out / lead / trade) must be explicit
and user-controllable — ReaLJam's user study showed control over *when the
machine plays* is essential to enjoyment.

**4. The feedback loop is the whole game for LLM operators.** Every
LLM-drives-a-DAW experiment failed the same way (Hurliman, Dec 2025: "one-shot,
or human tightly in the loop") until render + analyze endpoints existed. So
`render` and `analyze` are first-class from day one, the session is a
dumpable/restorable JSON score, every mutation is idempotent, and patterns are
Strudel-style mini-notation strings — terse, diffable, and already in LLM
training corpora (the 4-tool Strudel MCP proves pattern-as-text beats
fine-grained note tools).

**5. Pure-Python DSP is fast enough — measured, on this machine.** 16 voices
(2 osc + envelope + filter) + master saturation in NumPy/scipy, 256-sample
blocks: **124µs per block against a 5,805µs budget (47× headroom)**. The rule
that keeps it true: zero per-sample Python loops — recursive filters via
`scipy.signal.lfilter` / vectorized TPT SVF, coefficients updated per block.

## Decisions (with the losing side noted)

The Winner column is what was built. Where the first plan lost to something
simpler, the first plan sits in the Over column.

| Decision | Winner | Over | Why |
|---|---|---|---|
| Engine | Pure NumPy/scipy block DSP, 48kHz, 256-sample blocks, float32 | SuperCollider scsynth + Supriya | 47× measured headroom; one codebase for offline & live; no server process to supervise. scsynth remains the swap-in behind `Engine` interface if we outgrow it. |
| Language | Python 3.11+ | TypeScript | The whole MIR ecosystem (BeatNet, PESTO, librosa) is Python; DSP benchmark confirms perf. |
| API shape | CLI + session-as-JSON-file on disk. A live take adds a local cue socket and one-line-per-event log files (`decisions.ndjson`, `cues.ndjson`) | Daemon-first; a JSON-RPC daemon and an MCP facade (first plan, not built) | Stateless CLI over a declarative score file is the most LLM-legible surface and needs no process management for prompted mode. A socket that takes plain-word cues and a log an agent can tail cover live mode. |
| Patterns | Strudel-ish mini-notation subset (`bd ~ [hh hh] sd`, `*n`, `<a b>`, `@n`, `,`) | JSON note arrays | LLMs one-shot mini-notation; note arrays invite off-by-one bugs. |
| Sounds | 808 family + subtractive/FM synth voices, from the canonical recipes (research has exact envelope/filter numbers), plus a small set of recorded samples from open libraries. MIDI out (mido/rtmidi) is a second output for the live band | MIDI to plugin hosts as the only sound | No outside program needed; deterministic renders. |
| Live beat tracking (v1) | Hand-rolled causal tracker in NumPy behind a `BeatTracker` interface: tempo from autocorrelation of the onset envelope, a predicted next beat with partial phase correction, downbeat from accents and harmonic change; a count-in sets bar phase | BeatNet (first plan, not used), madmom (NC-licensed models, broken on py≥3.10), Essentia (AGPL, offline), librosa (offline-only) | Runs on NumPy alone with no model to download, and every threshold is exposed. |
| Live pitch/key (v1) | Hand-rolled chroma → 24-triad HMM + decaying Krumhansl histogram for chords/key (no off-the-shelf option is real-time). The optional `notes` extra adds the basic-pitch note model as a second reading of the key | PESTO streaming for melody (first plan, not used), CREPE (heavy, dominated), chordino/autochord/crema (batch-only) | |
| Audio I/O (v1) | sounddevice duplex 48kHz/256, dumb callback + ring buffers, 4-thread model (callback / planner / analysis / cues) | rtmixer (not needed) | The callback only copies buffers; planning and listening run on their own threads. |
| Mic bleed (v1) | Schedule-aware self-masking (we know what we played + the round trip `plume calibrate` measures with clicks) > speech-tuned AEC; headphones or a plugged-in instrument recommended | WebRTC/speex AEC alone | AEC is speech-tuned, underperforms on music. |
| License | AGPL-3.0-or-later (Juniper's call 2026-08-13) | MIT (the day-one default) | Copyleft, network-clause protected. Side benefit: GPLv3 tools (pedalboard, DawDreamer, surge) are now license-compatible for direct integration; only madmom's CC-NC models stay off-limits. |

## The interaction spec (live mode, encoded from the jazz research)

Twenty negotiation behaviors mapped to audio-detectable triggers — the full
list was in the original research sweep (jazz-negotiation → recommendations). Headlines:

- **Enter like a sideman**: silent until beat+key confidence stable ≥2 bars
  (min 4 bars of listening), enter only at a 4/8-bar boundary, 6–10dB under
  the human's RMS, sparse, ramp over 2–4 bars.
- **Tempo ≠ phase**: per-beat phase nudge (α·error); re-estimate tempo only on
  sustained drift (>2–4% over 4+ beats). Role-dependent stubbornness: α high
  when following, low when holding groove.
- **Inverse density coupling** with hysteresis + one-bar delay; response-rate
  cap (~1 in 2–3 phrases); register avoidance; dynamic-arc matching at
  boundaries only.
- **Silence state machine**: human silent >2 bars → thin to skeleton; 4–8
  bars → fade out, re-arm entry; no pulse ever established → ambient pads, no
  grid.
- **Ending detection suite**: ritardando-following, sustained-chord hold,
  head-recall detection, tag-loop handling. (Endings are where machine
  partners embarrass themselves.)
- **Graceful degradation**: chord confidence high → full voicings; low →
  roots/fifths; none → pedal tones. Beat lost → non-metric ambience, re-enter
  on next confident downbeat. *Playing neutral beats playing wrong.*
- **Every threshold is an API parameter**, and each behavior gets a fixture:
  synthetic "human" WAV exercising the trigger → render → assert the response.
  The interaction spec doubles as the integration-test matrix.

## The style playbook (prompted mode)

Per-genre data — exact progressions, drum grids, swing values, arrangement
templates — was in the original research sweep (generative-structure). Encoded as JSON
style files: trap (130–150bpm half-time, i-VI-III-VII family, hat-roll fill
lexicon on a mixed grid, straight swing), lofi (70–90bpm, ii7-V7-Imaj7 with
7ths/9ths mandatory, swing 54–62%, kick −15ms/snare +20ms "drunk" bias), synthwave
(80–118bpm, i-VI-VII lifts, four-on-floor, gated snare). Melody = motif +
transformation (transpose/invert/fragment/sequence under chord-tone
constraints), never note-by-note sampling. Arrangement = energy scalar (0–1)
per 8-bar-multiple section gating layers/density/cutoff — "three movements" is
three energy arcs.

## Live-mode defaults

Onset/pitch tracking tuned for plucked strings with open-string
drones; chroma/key detection over chord templates (sus/add-heavy, modally
ambiguous harmony — never assume standard tuning); tempo follower must tolerate
rubato and lay out during free sections; default comping = texture and
sub-bass, patient, minor/modal; additive layering UX; a post-hardcore energy preset;
**every parameter exposed**.

