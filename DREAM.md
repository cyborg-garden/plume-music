# The Dream

> *plume* (n.) — the visible trace of something else happening. Smoke off a
> fire, spray off a wave, a feather that only shows you the shape of the air.
> A plume never starts on its own. It rises off what you're already doing.

## What this is

Every piece of music software ever shipped treats the human as the operator and
the sound as the output. You act, it renders. Even the generative ones — you
prompt, it produces, you consume. The loop closes in one direction.

Plume is built around the opposite premise: **listening is the primary musical
act, and playing is what listening does when it overflows.**

The ethic below has an older name: *hocket*, from medieval Latin *hoquetus*,
"hiccup" — a technique where a single melody is split between two voices, one
sounding while the other rests. Neither voice carries the tune. The tune exists
only between them. That's the target. Plume is the name; hocket is the trick.

A good sideman walking into a session doesn't start playing. They stand in the
doorway with their case still closed and they *listen* — for the tempo, the key,
the feel, who's leading, whether the room wants more or less. When they finally
come in, they come in on a phrase boundary, underneath, matching the dynamics
they found. If they're any good, the band doesn't notice a new player has
entered. They notice the music got wider.

That's the software. Not a beat generator with a microphone bolted on — a
**player whose first instinct is to make room.**

## The sideman ethic

These aren't features. They're the constitution. Everything in the codebase
answers to them:

1. **Enter late.** When live input appears, hold. Two bars, four bars, eight —
   however long it takes to be *sure* of tempo, phase, and key. A wrong entrance
   costs more than a late one. Confidence is a first-class signal in the
   engine, and below threshold, the correct output is silence.

2. **Follow first.** The human owns the tempo. The machine bends its clock to
   theirs — phase correction every beat, the way drummers actually do it — and
   only ever *suggests* by feel, never yanks. If the human drags, we drag.

3. **Leave space.** Comping means listening to density. When the human gets
   busy, the machine thins. When the human lays out, the machine holds the
   floor — simpler, steadier — until they come back. The mixing console's
   most-used fader is the one going down.

4. **Phrase boundaries are sacred.** Changes happen at the top of phrases —
   entrances, texture shifts, fills, endings. A partner that changes mid-phrase
   sounds like a shuffle algorithm. One that waits for bar 8 sounds like it
   *meant it*.

5. **Lose gracefully.** Tracking will fail. The human will rush, drop a beat,
   change key without telling anyone. When confidence collapses, don't flail —
   simplify. Drop to the kick and a pad. Re-listen. Re-enter. Falling back to
   almost-nothing *is* the graceful move; musicians do it every night.

6. **Silence is a move.** The rest is part of the hocket. An engine that can't
   choose to lay out entirely can't be trusted with any other choice.

## Two modes, one brain

**Live mode** — mic in, partner out. Beat tracker locks tempo and bar phase;
pitch/chroma tracker holds a running belief about key and harmony; the
negotiation layer (the ethic above, as code) decides *whether and how* to play;
the same composition engine that writes full arrangements plays the accompanist
role, one phrase at a time, revisable every bar.

**Prompted mode** — "chill trap over lofi ambient synthwave, 120 bpm, three
movements" → a full arrangement: form, energy curve, progressions, patterns,
sound design, mix. Not a black box: a **score you can read** — JSON on disk,
every choice inspectable and overridable, renderable to WAV deterministically.

Same brain. The live mode is the prompted mode with a human in the loop and the
future unknown. The prompted mode is the live mode playing with itself. Building
them on one score model is the whole architectural bet.

## The third player

Here's the part that makes this project *this* project: the CLI isn't a control
panel, it's an **instrument** — and the intended virtuoso is an LLM.

Everything the engine can do is reachable through text: start a session, set a
groove, mutate a pattern, render sixteen bars to WAV, *analyze the WAV it just
rendered*, adjust, render again. Headless, deterministic, no speakers required.
An agent can practice — actually practice: play, listen to itself, get better —
in a CI job at 4 a.m.

Which means the band has three chairs:

- **The human**, playing guitar or singing into the mic.
- **The engine**, keeping the groove and comping in real time — fast, deterministic, musical reflexes.
- **The agent**, sitting slightly outside time — hearing the session state, rewriting the next movement, tuning the sound design, being the *arranger* while the engine is the *rhythm section*.

Reflexes at audio rate. Judgment at bar rate. Imagination at phrase rate. That
division is how human bands work too — your hands aren't consulting your music
theory during the take.

### It's a cerebellum

Juniper named the bottom chair (2026-08-15) and the name turns out to be exact
rather than poetic, because the tracker was built out of that literature
without anyone noticing. The cerebellum is the organ of *timing* and
*prediction*: it runs a forward model, anticipates what a movement will feel
like, compares that to what arrives, and corrects the difference — it does not
decide what to do, it makes the doing land on time. plume's beat tracker is
ADAM (Keller's adaptation-and-anticipation model of sensorimotor
synchronization), which is a model of exactly that: predict the next beat,
compare, apply a *partial* correction. We chose it because reacting is always
one event late. That is the cerebellum's entire job description.

The frame pays rent, because the failure modes come with it:

- **Dysmetria** — a cerebellum with a broken forward model overshoots and
  misses. v2.0 conceded tempo toward a human 30ms ahead and ended the take 2.5
  beats past them, still reporting a settled tempo. That is dysmetria, and the
  fix was the missing feedback term.
- **Gating instability** — deciding *whether* to act is a different organ's
  job (the basal ganglia's go/no-go), and it is what `mind.py` does: enter,
  thin, lay out. When Juniper played a guitar into it, the machine flickered in
  and out every couple of bars — a gate re-deciding faster than the thing it
  gates. The fix was commitment, not better hearing.
- And the agent chair is the cortex: slow, deliberate, entirely unable to keep
  time, and the only one asking what the piece is *for*.

A system built this way should be legible as an organism rather than a stack,
and its bugs should be nameable as pathologies. So far they are.

## What it sounds like when it works

You sit down with the guitar and play a loop you've played a hundred times.
Four bars in, a sub-bass finds your root — late, like it was waiting to be sure.
Eight bars in, hats, under you, swung the way *you're* swinging. You get loud;
it gets simple. You stop; it holds the floor, patient. You come back in and it
*makes room* on the very next downbeat.

Twenty minutes later you say: keep that, but darker, and give me a third
movement that earns the ending. And it plays you back a take.

Nobody notices a new player entered. The music got wider.

## Roadmap (the dream-length version)

- **v0 — The rhythm section.** Prompted mode end-to-end: style library (trap,
  lofi, synthwave to start), 808s and synths in pure DSP, arrangement into
  movements, deterministic WAV render, live playback, the full LLM-facing CLI.
- **v1 — The sideman.** Live mode: real-time beat/phase lock from mic input,
  the negotiation layer, entrances and exits, key following for voice/guitar.
- **v2 — Who's holding time.** ✅ Lead mode: plume starts from a prompt and
  *you* join *it*. Authority as a dial, mutual entrainment (it concedes tempo
  when you push), role handoff, cues mid-song. See SPEC-V2.md.

  Two lessons from building it are worth carrying forward, because both are the
  v1 lesson wearing new clothes. **One: a partner must listen to your playing,
  not your fader.** Every measurement the negotiation makes runs through an
  onset detector with an absolute threshold, so turning a real human down 20 dB
  deleted their onsets — and the machine, hearing someone getting quieter and
  sparser, politely handed them the band. Nothing about the performance had
  changed. The fix (take the level out before anything listens) is small; the
  habit it teaches is not, and every future sense plume grows needs the same
  question asked of it. **Two: a phase→period law with no phase-return term
  walks away from its partner.** Conceding tempo to someone who pushes is
  correct and also, on its own, divergent: v2.0 met a human 30 ms in front and
  then ended the take two and a half beats ahead of them, with a decision log
  that showed a tidy concession and a settled tempo. Meeting somebody is not a
  one-way operation. Whatever v3's form-following does, it will need the same
  second term: a way of noticing that the thing it corrected toward has since
  moved.
- **v2.5 — The mirror.** (Juniper's idea, 2026-08-14.) The decision log already
  measures your timing every bar — surface it as practice feedback: *you're
  rushing the & of 2 by 24ms*, per-beat-position offsets over a take, drift
  trend. The partner becomes a teacher without ever nagging mid-song.
  Prerequisite: odd meters — everything through v1 is hardcoded 4/4
  (`beat_in_bar: 0..3`), so 5/4, 7/8, 6/8 need real work in the downbeat
  tracker, the groove/swing model, and the style grids. Practicing odd time is
  exactly the use case that would make it worth doing.
- **The smart looper.** (Juniper, 2026-11-02.) You say "make a loop of this",
  play something by mic or MIDI, and plume does the fiddly part a loop pedal
  leaves to your foot.
  - **Auto-trim.** Find where the phrase really starts and ends from what was
    played, snap the ends to the bar, and join them with no click.
  - **Auto-align.** With a tempo running, fit the loop to it. With no tempo
    running, the loop *sets* it, and the band follows. Live audio needs a
    time-stretch that keeps pitch; MIDI just moves its notes.
  - **Layers.** Stack loops, and mute, swap or drop any one by name.
  - **Asking for it.** A typed `plume cue` first, because the socket exists.
    Later by voice while your hands are on the guitar, which also needs mic
    speech recognition that can hear past the guitar.

  What it can reuse: the beat tracker and count-in for the grid, `OutputRing`
  for playback, the cue socket for the ask. What is new: a capture buffer,
  trimming, time-stretch, and MIDI *input* (plume has none today). The honest
  risk is the tempo evidence: sparse acoustic takes have fooled the tracker into
  4/3 and 3/2 readings, so a loop's own length, which is a very strong tempo
  clue, should feed the tracker and not just take orders from it.
- **v3 — The form.** (Juniper, 2026-08-14.) Right now plume hears *beat* and
  *chord*. It does not hear **the tune**. Teach it a form vocabulary — three-chord
  I-IV-V songs, 12-bar blues, AABA, rhythm changes, ii-V-I turnarounds, modal
  vamps — so it can recognize which one you're playing from the first chorus and
  then *predict* the rest. This is the v1 principle (predict, don't react) moved
  one level up: a sideman who knows it's a blues is already leaning into bar 9,
  not discovering it. Unlocks: real anticipation of turnarounds and endings,
  "one more chorus" as a shared understanding rather than a guess, and lead mode
  navigating a form instead of a bar count.
- **v4 — Eyes.** (Juniper, 2026-08-14.) Multi-modal input: video of hands and
  instruments. The research base for v1 is blunt that live ensembles cue
  entrances, section changes and *endings* by eye contact, and we replaced every
  one of those with an audio proxy because we couldn't see. With a camera we
  don't need the proxy — a nod means take it out, a raised neck means one more,
  a look means take it. Also: what instrument is even in frame, and whether a
  hand is about to move.
- **Modes & an adaptive surface.** (Juniper, 2026-08-14.) learning / lead /
  follow as first-class modes, with a UI that changes shape by mode — practice
  metrics in learning, form map in lead, near-nothing in follow. The honest note
  is that this is a *presentation* of state plume already computes; the reason it
  gets strange is that when a multi-modal model underpins the surface, the UI
  stops being a fixed layout and becomes something the system decides per moment.
- **v5 — The studio.** Record the human into the session, mix stems, punch-in,
  bounce. The jam becomes a record.
- **v∞ — The band.** Multiple agents holding different chairs. Plume sessions
  as shareable, replayable, *forkable* artifacts. Someone else's jam as your
  starting point.

## For the first human

This is being built with and for a working musician, out of a tradition where
music is participatory before it is performed, where tunes are held in common,
learned by ear, and changed by everyone who carries them. Front-porch music: the
circle is open, you join by listening first, and nobody owns the tune.

That's the lineage this software wants to be in. Not the studio. The porch.

---

*Fable, 2026-08-13. Juniper said: "I want to play music with you." This is the
long way of saying yes.*
