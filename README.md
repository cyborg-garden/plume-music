# plume

> the visible trace of something else happening — smoke off a fire, spray off
> a wave. a plume never starts on its own. it rises off what you're already doing.

**Play music with your machine.** Plume is a musical partner: hand it a prompt
("chill trap beat over lofi ambient synthwave at 120 bpm and 3 major
movements") and it writes an arrangement you can read, revise, and render —
deterministically, to the sample. Live mode (v1) listens through the
microphone, catches your beat, and enters like a sideman: late, underneath,
on the phrase boundary.

Built LLM-first: the CLI is the instrument, the session is a JSON score, every
command speaks `--json`, and `render` + `analyze` close the loop so an agent
can hear its own work headlessly — no speakers, no DAW, no GPU.

## Quickstart

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e .
# or: uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .

plume make "chill trap beat over lofi ambient synthwave at 120 bpm and 3 major movements"
plume render --out take1.wav          # deterministic: same seed = same bytes
plume analyze take1.wav --json        # tempo, key bands, energy arc, per-track RMS
plume play                            # hear it (needs an audio device)
```

Then start negotiating:

```bash
plume notation                        # the pattern mini-language, fully documented
plume pattern set drums main "bd ~ [~ bd] ~, ~ ~ sd ~, [hh hh hh]*2 [hh*8]"
plume set swing 0.58                  # MPC swing, 0.5 straight .. 0.66 triplet
plume set tracks.pad.fx.lpf_hz 1200
plume render && plume analyze take1.wav --json
```

Every mutating command returns a state hash; `plume show --json` dumps the
whole score; `plume get`/`set`/`unset` address any field by dotted path.
Styles: `trap`, `lofi`, `synthwave` (each a data file — new genres need zero
code). Voices: 808 kick/sub (key-tuned, glides), hats, clap, snare, supersaw
pad, pluck, FM e-piano, lead, vinyl crackle — pure NumPy DSP, 48kHz.

## Play with it

[docs/PLAYING.md](docs/PLAYING.md) covers playing a real instrument with plume:
counting it in, talking to it, running it with a coding agent, and setting up for
recording or filming.

## Live mode (v1 — the sideman)

One pipeline, two transports. `sim` streams a WAV through the identical
listen → mind → duet code path offline (deterministic, faster than real time);
`jam` is the same thing live through your mic and speakers.

```bash
plume sim you_playing.wav --style lofi --out duet.wav   # offline duet
cat duet.ndjson                        # the decision log: every bar, with reasons
plume sim you_playing.wav --json      # full manifest (entry bar, key, levels)
plume listen you_playing.wav          # analysis only: beat/key/chord stream
plume mind                            # every negotiation threshold + meaning
plume jam --style lofi                # live (needs an audio device / sounddevice)
plume jam --record takes/             # also writes human.wav + machine.wav + log
plume calibrate                       # measure your room's speaker->mic round trip
```

The machine enters only after it has heard you for `min_listen_bars` with a
locked beat, on a phrase boundary, 8 dB under your level — then ramps in,
thins out when you get busy, lays out when you stop, and re-enters the same
way. Every threshold is a `--mind k=v` override; every decision lands in the
NDJSON log with a human-readable reason. A worked example lives in
`renders/duet_demo.*` (input, duet, decision log, manifest).

**Timing is compensated for your room.** Stream time is not room time: audio
placed on the tracked downbeat leaves the speakers an input+output latency
later — 112 ms on the development laptop, a fifth of a beat at 120 bpm.
`plume calibrate` plays a short click train and matched-filters the mic to
measure the true round trip once per device pair; the sideman (follow-mode
`jam`) then places bars that much early, so the machine lands **in the room**
on your beat. (Lead mode owns the clock — its correction is analysis-side,
and still on the list.)
Uncalibrated sessions fall back to the stream's reported latency (roughly
right), and `--latency-comp MS` overrides either (`0` disables).

## Lead mode (v2 — the band)

v1 gave the machine one chair: you own the clock, it earns its way in. v2 adds
the other chair, and the ability to swap mid-song. **The machine starts — from
a prompt or a session you already wrote — and you join *it*.**

```bash
plume simlead you_joining.wav --prompt "chill trap at 120 bpm"   # offline twin
plume simlead --prompt "dark lofi at 88 bpm"                     # machine solo
plume simlead you.wav --session my.plume.json --role trade       # trade fours
plume simlead you.wav --prompt "..." --cue "16:go to the chorus@next-section"
plume roles                            # every role threshold + meaning
plume lead "chill trap at 120 bpm"     # live: the machine starts, you join
plume cue "take it out"                # to a running lead session
```

Authority is two dials, not a mode: `clock_authority` (who owns time) and
`form_authority` (who owns the form), each a float, each moving *during* the
session — that movement is the negotiation. Push the beat and it concedes
tempo and comes with you; lay back and it settles; get busy and it yields the
lead; stop and it holds the floor; play a semitone away and it transposes to
meet you. Everything lands in the decision log with a reason, and the flagship
pair in `renders/lead_demo.*` and `renders/trade_demo.*` (mix, decision log,
manifest) is regenerated from scratch by
`.venv/bin/python tests/lead_demo.py`.

Two things worth knowing, because they are what make it usable rather than
demoable:

- **The negotiation listens to your playing, not your fader.** The analysis
  chain takes your level out before anything reads it, so the same take at
  0 dB and at −20 dB produces bit-identical decisions. (It did not, once:
  turning a player down 20 dB used to make the machine hand them the band.)
- **A cue either changes the music or says it did not.** Every cue kind reports
  its actual effect per bar in `cue_effects`; the ones the renderer cannot
  honour say "understood but NOT applied", and why.

## Live sessions are findable, stoppable, and time-bounded

Live mode makes sound in a real room. Twice during development an automated
agent started a session and left it running, still audible, stoppable only by finding the PID. So a live session is opt-in to
start, and it can always be found and ended without hunting a process.

```bash
plume sessions                         # what is running: pid, role, style, uptime
plume stop                             # end it (finishes the bar, flushes the tail)
plume stop --all                       # end every live session
plume stop --pid 4212                  # end that one
plume jam --max-minutes 20             # or lead: self-stops after 20 minutes
plume lead "..." --max-minutes 0       # no cap (an interactive human opting out)
```

Four things hold the line, and none of them is a convention anyone has to
remember:

- **Nothing opens the speakers unattended.** An interactive terminal, or an
  explicit `PLUME_ALLOW_AUDIO=1`. Nothing else. On a machine with no audio
  hardware you get the device error instead — a box that cannot play must not
  be told to set a consent flag that cannot help it.
- **Every live session registers itself** — one JSON file per pid under
  `$PLUME_SESSION_DIR` (default `<tmpdir>/plume-sessions/`), so `sessions`
  and `stop` work across terminals and outlive the shell that started them.
  Records whose process is gone are pruned silently.
- **`plume stop` is graceful first.** It asks over the session's own socket —
  the same channel `plume cue` uses — and the session finishes its bar,
  flushes its tail and writes its records, exactly as Ctrl-C does. Then
  SIGINT, then SIGTERM. Never SIGKILL: that would cut the tail and throw the
  records away. If nothing is running it says so and exits 0.
- **Signals are graceful too, including in the background.** A session
  started in the background by a non-interactive shell (`plume jam &` in a
  script, an agent's subprocess, a CI job) inherits SIGINT *ignored* — so it
  used to play straight through Ctrl-C's signal, and the only thing that could
  end it was a SIGTERM that destroyed the take. Live sessions now handle
  SIGINT and SIGTERM themselves and end the same graceful way. A second
  Ctrl-C still gets you out.
- **A session with nobody watching ends itself** after `--max-minutes`
  (default 60; `0` disables). It ends the same graceful way, and it says why —
  in the status line and in `ending.json` beside the recording, so a take that
  stops because a cap fired never looks like one somebody ended on purpose.

## The papers behind the play

- **[DREAM.md](DREAM.md)** — why this exists; the sideman ethic
- **[RESEARCH.md](RESEARCH.md)** — the evidence base (entrainment science, 40
  years of interactive music systems, groove microtiming, LLM-tool failure
  modes), distilled from a 10-agent research sweep
- **[SPEC.md](SPEC.md)** — architecture, score model, notation grammar, API

## Roadmap

- **v0 — the rhythm section** ✅ prompted mode end-to-end
- **v1 — the sideman** ✅ live listening: causal beat/key tracking,
  predict-don't-react clock, the negotiation state machine
- **v2 — the band** ✅ lead mode and role negotiation: both chairs are real,
  and the chair changes mid-song (this)
- **The smart looper** — "make a loop of this": play it by mic or MIDI, and
  plume trims it, fits it to the tempo and keeps it going. See DREAM.md
- **v3 — the studio** — record the human into the session, mix stems, bounce
- **v∞ — the room** — multiple agents in different chairs; forkable jams

License: AGPL-3.0-or-later. Bundled samples carry their own licences and credits: see
[src/plume/samples/README.md](src/plume/samples/README.md).
