# plume

> the visible trace of something else happening: smoke off a fire, spray off
> a wave. a plume never starts on its own. it rises off what you're already doing.

**Play music with your machine.** Give plume a prompt ("chill trap beat over lofi
ambient synthwave at 120 bpm and 3 major movements") and it writes an arrangement you
can read, change and render, the same to the sample every time. Live, it listens through
a mic or an instrument, catches your beat, and comes in like a sideman: late, underneath,
on the phrase boundary.

It is built for language models as much as for people. The CLI is the instrument, a
session is a JSON score, every command speaks `--json`, and `render` plus `analyze` let
an agent hear its own work with no speakers, no DAW and no GPU.

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
Styles: `trap`, `lofi`, `midwest`, `synthwave`, `jazz`, `house`, `hiphop` and
`rnb`, each a data file; a prompt can also say `boom bap`, `r&b` or `neo soul`.
Voices: 808 kick/sub (key-tuned, glides), hats, clap, snare, supersaw pad, pluck,
FM e-piano, lead, vinyl crackle, plus a house kick and stab, a boom bap kick and a
round bass, all pure NumPy DSP at 48kHz. Beside them is a small recorded kit with a
brush on a snare, and recorded double bass, grand piano, electric piano and vibraphone,
all from open sample libraries (see `src/plume/samples/README.md`).

## Play with it

`plume gui` opens a local page with one Start button. It opens no device and starts
nothing until you click Start. The first press is the setup, one line at a time: play
something so plume can find your channel and check your level, say whether headphones are
on, then count, and the band comes in. Later presses go straight to the count. A Guide
link next to Settings has short pages on all of this, including how to drive a take from a
terminal or a coding agent.

[docs/PLAYING.md](docs/PLAYING.md) has the long version: counting in, talking to plume,
running it with a coding agent, and recording or filming. The page has only ever started
simulated takes, and only one Mac and one audio interface have been tried with plume at
all, so other computers and other gear are untested. The test suite runs on macOS, Linux
and Windows; live sound on Linux and Windows has not been tried yet. [docs/DIRECT-INPUT-TEST.md](docs/DIRECT-INPUT-TEST.md)
says how to try it on real hardware.

## Live mode (v1: the sideman)

One pipeline, two transports. `sim` streams a WAV through the same
listen → mind → duet code offline, deterministic and faster than real time.
`jam` does the same thing live through your mic and speakers.

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

The machine comes in only after it has heard you for `min_listen_bars` with a
locked beat, on a phrase boundary, 8 dB under your level. Then it ramps in, thins
out when you get busy, lays out when you stop, and comes back the same way. Every
threshold is a `--mind k=v` override, and every decision lands in the NDJSON log
with a reason a person can read. A worked example is in `renders/duet_demo.*`
(input, duet, decision log, manifest).

**Timing is corrected for your room.** Audio placed on the tracked downbeat leaves
the speakers one input-plus-output latency later: 112 ms on the development laptop,
a fifth of a beat at 120 bpm. `plume calibrate` plays a short click train and listens
for it to measure the real round trip, once per device pair. The sideman
(follow-mode `jam`) then places bars that much early, so the machine lands on your
beat in the room. Lead mode owns the clock, so its correction belongs in the analysis,
and that is still to do. Uncalibrated sessions use the latency the stream reports,
which is roughly right, and `--latency-comp MS` overrides either (`0` turns it off).

## Lead mode (v2: the band)

In v1 the machine has one chair: you own the clock, and it earns its way in. v2 adds
the other chair, and the two can swap mid-song. **The machine starts, from a prompt
or a session you already wrote, and you join *it*.**

```bash
plume simlead you_joining.wav --prompt "chill trap at 120 bpm"   # offline twin
plume simlead --prompt "dark lofi at 88 bpm"                     # machine solo
plume simlead you.wav --session my.plume.json --role trade       # trade fours
plume simlead you.wav --prompt "..." --cue "16:go to the chorus@next-section"
plume roles                            # every role threshold + meaning
plume lead "chill trap at 120 bpm"     # live: the machine starts, you join
plume cue "take it out"                # to a running lead session
```

Who leads is two dials: `clock_authority` (who owns time) and `form_authority`
(who owns the form). Each is a number that moves during the session, and that
movement is the negotiation. Push the beat and it gives up tempo and comes with you.
Lay back and it settles. Get busy and it hands you the lead. Stop and it holds the
floor. Play a semitone away and it transposes to meet you. Everything lands in the
decision log with a reason. `renders/lead_demo.*` and `renders/trade_demo.*` (mix,
decision log, manifest) are rebuilt from scratch by `.venv/bin/python tests/lead_demo.py`.

- **The negotiation listens to your playing, not your fader.** The analysis takes
  your level out before anything reads it, so the same take at 0 dB and at −20 dB
  makes exactly the same decisions.
- **A cue either changes the music or says it did not.** Every cue reports its real
  effect per bar in `cue_effects`. A cue the renderer cannot honour says "understood
  but NOT applied", and why.

## Live sessions are findable, stoppable, and time-bounded

Live mode makes sound in a real room, and an automated agent can start a session and
walk away from it. So a live session needs an explicit yes to start, and it can always
be found and ended without hunting for a process.

```bash
plume sessions                         # what is running: pid, role, style, uptime
plume stop                             # end it (finishes the bar, flushes the tail)
plume stop --all                       # end every live session
plume stop --pid 4212                  # end that one
plume gui                              # a page with one Start: it starts nothing until you click
plume gui --take takes6/               # read a take: what plume did, a roast, Keep
plume gui url                          # a new link to the running page
plume jam --max-minutes 20             # or lead: self-stops after 20 minutes
plume lead "..." --max-minutes 0       # no cap (an interactive human opting out)
```

None of these rules is a convention anyone has to remember:

- **Nothing opens the speakers unattended.** It takes an interactive terminal, or an
  explicit `PLUME_ALLOW_AUDIO=1`. On a machine with no audio hardware you get the
  device error instead, since a consent flag could not help there.
- **Every live session registers itself,** as one JSON file per pid under
  `$PLUME_SESSION_DIR` (default `<tmpdir>/plume-sessions/`), so `sessions` and `stop`
  work from any terminal and outlive the shell that started them. Records whose
  process is gone are pruned silently.
- **`plume stop` is graceful first.** It asks over the session's own socket, the one
  `plume cue` uses, and the session finishes its bar, flushes its tail and writes its
  records, exactly as Ctrl-C does. Then SIGINT, then SIGTERM. Never SIGKILL, which
  would cut the tail and lose the records. If nothing is running it says so and exits 0.
  On Windows the signal rung is Ctrl-Break, and ending a take outright waits for
  `--force`.
- **Signals are graceful in the background too.** A session started in the background
  by a non-interactive shell (`plume jam &` in a script, an agent's subprocess, a CI job)
  inherits SIGINT ignored. Live sessions handle SIGINT and SIGTERM themselves, so they
  still end the graceful way and keep the take. A second Ctrl-C still gets you out.
- **A session with nobody watching ends itself** after `--max-minutes` (default 60;
  `0` turns it off). It ends the same graceful way and says why, in the status line and
  in `ending.json` beside the recording, so a take stopped by the cap never looks like
  one somebody ended on purpose.

## The papers behind the play

- **[DREAM.md](DREAM.md):** why this exists, and the sideman ethic
- **[RESEARCH.md](RESEARCH.md):** the evidence base (entrainment science, 40 years of
  interactive music systems, groove microtiming, LLM-tool failure modes), distilled
  from a 10-agent research sweep
- **[SPEC.md](SPEC.md):** architecture, score model, notation grammar, API

## Roadmap

- **v0, the rhythm section** ✅ prompted mode end to end
- **v1, the sideman** ✅ live listening: causal beat and key tracking, a clock that
  predicts instead of reacting, the negotiation state machine
- **v2, the band** ✅ lead mode and role negotiation: both chairs are real, and the
  chair changes mid-song
- **The smart looper:** "make a loop of this". Play it by mic or MIDI, and plume trims
  it, fits it to the tempo and keeps it going. See DREAM.md
- **v3, the studio:** record the human into the session, mix stems, bounce
- **v∞, the room:** several agents in different chairs, and jams anyone can fork

License: AGPL-3.0-or-later. Bundled samples carry their own licences and credits: see
[src/plume/samples/README.md](src/plume/samples/README.md).
