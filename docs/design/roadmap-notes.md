# Roadmap notes

> Historical notes, not current behaviour. For what Plume does today, read the README and `docs/PLAYING.md`.

These are the long-form notes behind the roadmap in [DREAM.md](../../DREAM.md),
as they were written at the time.

- **v0 — The rhythm section.** Prompted mode end-to-end: style library (trap,
  lofi, synthwave to start), 808s and synths in pure DSP, arrangement into
  movements, deterministic WAV render, live playback, the full LLM-facing CLI.
- **v1 — The sideman.** Live mode: real-time beat/phase lock from mic input,
  the negotiation layer, entrances and exits, key following for voice/guitar.
- **v2 — Who's holding time.** ✅ Lead mode: plume starts from a prompt and
  *you* join *it*. Authority as a dial, mutual entrainment (it concedes tempo
  when you push), role handoff, cues mid-song. See docs/design/SPEC-V2.md.

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
- **Scoring to picture.** (Juniper, 2026-11-05.) The first time plume played
  with a film instead of a person. A three-minute short came with a cue sheet
  in plain words: where the voices start, a scene change, a dark turn and the
  moment it lifts, a line that has to stand out, then a wordless ending that
  wants a band and drums. Plume rendered a cue-locked sketch. Notes on it came
  back in plain words, about a harsh transition, the key, and an entry that
  should land a second earlier after a pause, and each one became an exact
  change in minutes. The stems were good enough to use.

  How it was done, with plume as it is:
  - **One session per scene, tempo solved from the clock.** Each scene's tempo
    is `bars × 240 / seconds`, so its bar lines land on the cue to the sample.
    Scene tempos ran from 60 to 123 bpm.
  - **A script did the timeline.** It laid each render at its timecode, with a
    fade per boundary, a deliberate silence before the last scene, and a fade
    to nothing before the picture ends. It stacked the stems on the same
    timeline, and laid the mix over the film's own sound, turned down under
    dialogue.
  - **A guide part where the human goes.** Plume has no guitar voice, so a
    quiet part plays the open strings of the guitarist's tuning. Its stem is
    muted once the real guitar is recorded.

  What plume would need for this to be a mode rather than a script:
  - **A timeline.** Sections pinned to timecodes, tempo per section, and hit
    points. Boundaries as real choices: cut, crossfade, silence, tail.
    Something shaped like `plume score cues.json --picture film.mp4`.
  - **Phrases per bar.** `< >` takes single tokens only, so a melody that
    changes every bar needed one section per bar.
  - **Transpose.** "Half a step down" meant rewriting every note token by
    script. It should be one field.
  - **Concert pitch.** The capo arithmetic and the sound in the room disagreed
    by a semitone. Measuring the render showed F at A440, so the guitar was
    sounding E. Better: let plume hear thirty seconds of the guitar and set
    the key itself, the same wish as "detect" in live mode.
  - **Automation inside a section.** Filter, level and reverb curves. Every
    scene was static, so any change in feel needed a new scene.
  - **Quiet-safe lofi.** Master `lofi` above 0.5 adds a 12-bit crush, which
    is audible on quiet material under dialogue. Crush wants its own control.
  - **A room bed.** The film's own ambience had been removed, so the script
    made air and dust from filtered noise. Plume's crackle loops each bar.
  - **Dialogue-aware mixing.** Read the film's speech level and sit under it,
    with a separate level for scenes without words.
  - **Credits travel with stems.** The Wurlitzer samples are CC-BY. A stem
    used in a film carries that credit, so `render` should list the credits
    each render needs.

  Why it belongs here: the sideman ethic transfers. The film is the player
  who owns the clock, and the notes in words are the cues. What plume
  listens to is cut points and spoken lines instead of a downbeat, and the
  first instinct is still to make room.
- **The conductor.** (Juniper, 2026-11-06.) Scoring the film was the first
  time the agent driving plume was asked for a whole chain of musical work in
  one breath. The next step is to make that the normal way to play:

  > Pull that sample, listen to this melody I play, loop it and fix my tempo
  > mistakes, smooth the joins where you had to cut, then set the samples
  > under it in a synthwave arrangement.

  Each verb in that sentence is a plume command or close to one: fetch a
  sample, `listen`, the smart looper's trim and align, a crossfade at the
  cut, `make` in a style. The new part is an agent holding the whole
  sentence, running it, and playing back what it heard and what it changed,
  so the human can say "no, keep my rush in bar 3" and it does. Everything
  it changes is logged with a reason, as the decision log already does.

  Where this sits: much of AI music generates the finished track from a
  prompt. Plume keeps the human making the music, on real instruments and
  MIDI, and the machine does the handling around it: tidying, looping,
  arranging, effects. The working phrase is **the analog effects wing of AI
  music**. That frame has rules: sources must be ones the player has rights
  to, and every sample's licence and credit travels with the take.
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
