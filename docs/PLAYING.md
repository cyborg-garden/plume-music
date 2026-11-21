# Playing with plume

Plume plays a band around you through your computer's speakers or your headphones while
you play a real instrument into a mic or an audio interface. You count it in, it holds your
tempo, and the band joins once it hears you start.

The page `plume gui` opens has a Guide next to Settings with the short version of this
document. This is the long one.

To hear what this sounds like, listen to
[`examples/first-keeper.m4a`](examples/first-keeper.m4a): one take on a guitar in
FACGCE tuning at 53 bpm, recorded by a laptop mic across the room from a projector's
speakers.

## Your first take

```bash
.venv/bin/pip install -e '.[play,voice]'   # voice: mlx-whisper on Apple Silicon, faster-whisper elsewhere
plume calibrate        # plays clicks; skip this on Bluetooth (see below)
plume lead --tap-start "lofi midwest emo in F major" --input acoustic --record takes1/
```

Nothing plays until you count. Knock on the guitar body or clap eight even beats, and
the drums land where the ninth would be. Plume needs four clear hits or six soft ones
to catch the count. Mouth clicks are usually too quiet, and if your first few knocks
are faint it still finds them once the later ones confirm the tempo.

The drums stay alone until you play. The bass and chords come in about eight bars after
you start, and the melody four bars after that. If you stop, the take fades out and ends
by itself. On a direct input or MIDI it is over about 20 seconds after you stop, from about
100 bpm up, and about 23 from 80 bpm. Slower bars can't end any sooner without a rest in the
song dipping the band: about 27 seconds at 66 bpm, about 30 at 60, about 40 at 45.
A room mic can't tell you from the band, so it keeps waiting for 30 seconds of silence at the
mic, and a band coming out of speakers can keep it from ever hearing that: stop those takes
yourself. Until you have played a bar, every input waits the 30 seconds,
so you have time to get your hands back on the strings after the count. Play during the
fade and the band comes straight back. On a direct input a rest under 6 seconds never dips the
band. A longer one can: from about 6 to 11 seconds depending on the tempo and where in the bar
you stop, later at very slow tempos. It can take two or three bars to come back up.

Once it is in, the band keeps listening for the whole take, measured against how you
normally play rather than against any fixed level. Your normal is where you have been
playing when you are not in a softer or louder passage, and it moves with you. Play softer for a couple of bars and it
comes down to meet you: quieter, fewer notes, the drums played lighter, and once you have
held it through a phrase the lighter parts sitting out, though the kick, snare and bass
keep going. Come back
up and so does the band. Playing fewer notes at your usual level only thins the band's own
patterns. One quiet bar, or one loud one, moves nothing. On an input that hears only you, if you have been playing well
below full scale, a knock on the instrument's body or a crackle from the cable that hits
full scale does not make the band swell: it lifts only as far as it would have with
those bars left out, and the `held_back` field in the decision row says when. That
covers the odd bump and crackle, not a cable that crackles all the time: clip in most of
the last half minute, or play near full scale yourself, and it is all read as your
playing. A room mic and a MIDI keyboard are followed as they always were. Answer yourself soft and loud a
few bars at a time and the band settles rather than pumping, with the song's parts and
lifts as written. A softer passage is followed for as long as it lasts, and when you come
back to your normal the band comes back to its own level with you, not past it, and a
louder chorus is lifted with you for as long as it lasts. If you open softly for a few bars
and then play at your real level, that becomes your normal within about 16 bars; open
softly for as long as a verse and what follows is a chorus. If you open loudly, or stay
louder or softer for well over a minute or two, it becomes your normal after about 64 bars. The song's own shape is only a
default: a lift you play softly under is played as a verse, and a verse you drive is played
as a lift, always from a phrase line. The `yield` field in each decision row says when that
happened. On a room mic the song keeps its own shape, since the mic hears the band too. Chords follow
what you play when the band can tell what it is: a triad, a power chord or a two-string
shape is enough, with no seventh needed. A loop is followed once you have played it a few
times, even when a bar of it doesn't read cleanly, and a new loop takes over once you have
played it twice. When the band can't tell, it plays the song's own chords. A room mic can't
tell your chords from the band's, so there it always plays the song's own.

Add `--detect` and you need no prompt at all: `plume lead --tap-start --detect`. Plume works
out whatever the prompt doesn't name as you play. It listens for the key, and after 16 bars
with no settled key it takes its latest reading. If it has no reading at all, as with playing
that has no tonal centre, the band keeps waiting on drums until it gets one. It picks a
style from your tempo when you count, then once more as the band comes in, from how you
have played over the bars it heard. Your tempo says which styles it could be, each style's
own range, and five things heard on a direct input pick among them: whether your eighths
swing, how many strokes you play to a beat, how many of them fall off the beat, how even the
gaps between them are, and how many of them are strummed chords. All five come from when and
how you strike, not from how loud you are, so the input gain barely moves them until your
level nears the input's noise, where the quietest strokes go unheard. Each style
has a band for the ones that tell it, a style loses for what falls outside its bands, and the
best score wins (`plume/playstyle.py` has the bands). Swung eighths are `jazz`, and nothing is
jazz without them. A busy even pulse is `synthwave`, other busy straight playing is
`midwest`, stabs off the beat are `house`, and a few notes with room are `trap`, `lofi`,
`hiphop` or `rnb`. `folk` needs a strum, straight and on the beat: an attack that lasts while
the pick crosses the strings of a chord, so a very quick strum is not heard as one, and
nothing is folk without one. When two styles fit about equally, or none fits, as with a
plain note or chord on each beat, your tempo and key decide, as they do on a room mic, which
can't tell your notes from the drums. With `--midi` your notes and when you play them count
the same way, all but the strum. These are guesses, and the bands were set on generated
parts, not on real players: name the style you want when it is wrong. Each take's
`decisions.ndjson` gives the answer and the next best style with
their scores and everything heard, in `style_why`. So far this has only run on a simulated device, never with a real guitar.

To stop from another terminal, run `plume stop` (`plume play` and `plume calibrate` too), or `plume sessions` to see what is
running. A take also stops by itself after 60 minutes; `--max-minutes N` changes that.

### Choices worth making before you start

**Name your key.** Write it in the prompt, as above. Plume can guess a key, but a
guess took 14 bars to settle on the right one in testing, and with the band audible in
the mic it is partly hearing itself. A named key never moves.

**Set a length.** `--wind-down 150` plays for 2:30 from the first drum hit, then signals
the ending: a crash cymbal, and the kit thinning out for four bars. Then it fades over
six bars and ends, so you can wind down with it. At 52 bpm, the signal and fade add about
45 seconds.

**Pick a tempo with your hands.** Plume plays exactly what you count. Below 70 bpm the
`midwest` style plays a swung breakbeat with the snare on 2 and 4. At 70 and above it
plays half-time trap with the snare on 3. The other styles are `lofi`, `trap`, `synthwave`,
`jazz`, `house`, `hiphop`, `rnb` and `folk` (4/4 only for now), and a new take with a new `--seed` gets a new melody. A
prompt can say `boom bap` for `hiphop`, or `r&b` or `neo soul` for `rnb`. Plume plays what you
count in any style, so a house take at 90 is slow house. Each style's usual range is `lofi`
70 to 90, `rnb` 60 to 90, `hiphop` 80 to 100, `house` 118 to 130, `folk` 60 to 132 and `jazz` 70 to 220.

The band you play with is drier than a `plume render` of the same style. `plume lead`, `plume
jam` and `plume sim` use each part's filters and no reverb, delay or whole-mix drive or tape.
So that the newest styles stay themselves, `house`'s bass and stabs still pump with the kick,
`hiphop`'s kick, snare and bass are saturated, and `rnb`'s keys keep their chorus, at the level
they had without it. That chorus runs per note, where a render runs one over the whole part, so
overlapping notes do not share it. `hiphop`'s tape wobble and every style's reverb are only in
`plume render`.

## Talking to it

`plume voice` starts the same take and also listens for spoken commands. Whisper turns
the speech into text on this computer, and nothing leaves it.

```bash
plume voice "lofi midwest emo in F major" --base . --wind-down 150
```

| Say | What happens |
|---|---|
| "plume, stop" (or "end", "finish", "quit", "done", "enough", "that's it", "shut up", "turn it off") | ends the take |
| "plume, restart" | ends the take and waits for a new count |
| "plume, busier" | more hats, keys and melody from the next phrase |
| "plume, sparser" | less of them |

Every command needs "plume" or what the speech model writes for it, such as "bloom" or "plum",
so most lyrics can't end your take, though one with "bloom, stop" in it can. The word "stop"
can have words before it ("plume, could you stop") or "plume" after it ("stop, plume"), and it beats a louder or
quieter said in the same breath. "Plume, don't stop" and "plume, wait until I stop" do nothing.
Every other word, the other stop words included, has to come right after "plume".

Save "plume" for commands. When a stop is in doubt it stops, so "plume, stop rushing", "plume,
stop the hats", "plume, that's it, keep going", "plume, I'm done, you take it" and "when plume
stops I'll come in" all end the take. Some plain ways to say stop are still missed, and so is
a stop with more than 8 words before it.

It reacts a few seconds late, because it listens in four-second windows. It has not yet
worked on a live voice. The path from a transcript to a change in the running take is
tested, but the one real session it ran in never caught a spoken command, and nothing
recorded what the model had heard. Speak up and keep a keyboard nearby.

Each take now keeps `voice.ndjson`, with one line per window: how loud it was, what the
model wrote, and whether that counted as a command. `plume voice-replay takes/human.wav`
asks the same question of a recording that already exists, with no band and no sound.

A second mic on its own channel keeps your voice apart from the band, and on a plugged-in
guitar it is the only way plume can hear you: add `--voice-channel N` (see
[`DIRECT-INPUT-TEST.md`](DIRECT-INPUT-TEST.md)).

## With a coding agent as the brains

An agent is far too slow to play notes, but fast enough to decide what the next section
should do. Plume plays bar by bar on its own, and the agent steers from outside. A take
started from the page's Start takes cues the same way, so an agent can steer that too.

```bash
# start the take. The variable is deliberate: without a terminal attached, plume
# refuses to make sound unless told to, so an agent can't start it by accident.
PLUME_ALLOW_AUDIO=1 plume lead --tap-start "lofi midwest emo in F major" --record takes/ &

# read what is happening: one JSON line every 8 bars when something changes
plume watch takes/ --every 8 --changes-only

# steer. It lands at the next phrase
plume cue "busier"
plume cue "sparser"

plume stop
```

A `watch` line looks like this:

```json
{"bar": 31, "section": "lift", "stage": "full", "key": "F major", "chord": "Fmaj7",
 "you_db": -24.7, "band_db": -25.1, "mic_db": -24.4, "you_busy": 5.38,
 "you_notes": ["A", "F", "C"], "you_source": "mic",
 "changes": ["section:lift", "you:busier"], "playing": true, "playing_trusted": true, "fade_db": 0.0}
```

`playing` is what the engine decided about you, and what it fades the band by.
`playing_trusted` says whether to believe it: over a room mic with the band in, it mostly
describes the band. `fade_db` goes below 0 while the band is fading out.

Trust `section`, `stage`, `key` and `chord`, which report what the band is doing. The
`you_*` fields come with their own verdict. Every line has `input` (`room`, `direct`, `headphones` or `midi`),
`you_trusted` (the names of the `you_*` fields that describe the player) and `you_why`.

| Situation | `you_trusted` |
|---|---|
| Room mic, band in | none |
| Room mic, drums alone | `you_notes` only |
| `--direct`, `--midi`, or the band not sounding | all three |
| `--direct`, but the band turns up in the input | none, and `you_why` says to check routing |
| `--band-in-headphones` (a mic, band only in headphones) | all three |
| `--band-in-headphones`, and the take has heard the band in the mic | as a room mic from then on; the line has `band_in_mic: true` |

Report nothing from a field that isn't listed. `you:*` entries in `changes` come from the
same fields, so they get the same doubt. On a room mic those fields measure the mic with
plume's sound subtracted, and in a real room that removes only a few dB. A reader who skips
this check will report notes from a player who is sitting silent.

A line's `you_source` says where its `you_*` fields came from: `mic`, or `midi` on a
`--midi` take. A `midi` line can be trusted with the band in: `you_notes` and `you_busy`
are what you played, and `you_db` is your velocity expressed as a level, not a measured
sound. A pulled cable shows up in `changes` as `midi:lost` and then `midi:back`.

Agents running tests should use the fake audio device in `tests/conftest.py`. An agent
that runs a live verb to check its work plays music out loud in whatever room the machine
is in.

## Making a loop

Ask for a loop with a typed `plume cue`, then play. Plume finds where your phrase starts
and ends, snaps it to the bar, joins the ends so they can't click, and plays it back.

```bash
plume cue "make a loop of this"            # then play, then stop
plume cue "make a 4 bar loop called bass"  # with the band in, closes on the bar line
plume cue "that's the loop"                # close a take by hand
plume cue "cancel the loop"
plume cue "mute the bass loop"
plume cue "unmute the bass loop"
plume cue "mute loop 1"                    # a loop you did not name
plume cue "stop the loop"                  # mutes them all, or closes a take in progress
plume cue "stop loop 1"                    # mutes only that one
plume cue "swap the bass loop for the keys loop"
plume cue "drop the bass loop"
plume cue "mute all loops"                 # unmute and drop take "all" as well
plume cue "list loops"
```

**This has only been run against a fake audio device. Nobody has played into it yet.**
Whether a loop sounds right, and how loud it should sit against the band, still needs a
person at the instrument. It runs on `plume lead`, but not on `plume jam`. `plume voice`
starts a `plume lead --tap-start` take, so a cue should reach that too, but nobody has run it.

**Getting ready is fine.** A rustle after the cue does not start or end the take, even just
before your first note. That needs the rustle to be short and well under your playing:
anything louder counts as your first note. Loud or sudden room noise can become a loop, and these
thresholds are not yet tuned on a real room. A single sound of any length waits 8 seconds for more, so
say "that's the loop" to keep one note sooner. One limit: with the band running and the bars
named, a knock that rings counts as your first note, and the take closes that many bars later.

**A named length closes on its bar line,** with the band running, even if you never stop
sounding: held chords and strums with no gap in them close that many bars after the first
sound. The first sound has to be clearly louder (10 dB) than the mic was when you gave the
cue, or loud in itself (about -25 dBFS), so steady hiss below that can't start it in tests. Noise that
is loud, or starts after the cue (a fan switched on, even at -50 dBFS), can start it and become the loop.
Low rumble can start it even when quiet: in tests, rumble at -60 to -40 dBFS with one bar named closed the take one bar
after the cue, before anyone played.
The cost: a quiet chord already ringing at the cue doesn't start it, and neither may playing
less than 10 dB above a loud room. Then the take waits for you to stop, say "that's the loop", or reach the 64 second limit.
At the limit it looks at the sound itself. With a bar grid running, a held note or chord has
narrow pitched peaks, so it is kept. Quiet steady white hiss has none, so in tests (up to -30 dBFS) it makes no loop, though
hiss at -24 dBFS and louder did, and so did noise with more low end at -30 dBFS. When nothing is kept, `plume cue "list loops"` says there was no clear start above the room: play louder, or say
"that's the loop" before the limit. A quiet melody gets the same message. Low rumble and a
steady hum can become a loop, since a hum has those peaks too and a loud fridge can read as a
held note. These thresholds are not yet tuned on a real room. Saying "that's the loop" usually keeps
what is there, noise included, but it can't save everything: sound below about -46 dBFS gets "heard nothing to loop", and with nothing
running a steady sound is refused as one sound. How loud a real laptop mic's room noise is has not been
measured. With nothing running there is no bar line, so a held chord keeps the take open the
same way, and at the limit it is refused even with the bars named, because a chord shows no reliable
tempo. A chord of several notes can be told to name the bars, or that it fits no bar count; naming them doesn't help. Count the band in first, then make the loop. The most bars one take can hold is 45. A loop can't be called "both", "every", "each" or
"all", since `mute both loops` means every loop, however many there are.

**Use headphones or a direct input.** A loop is the raw mic. Over speakers, the band and any
earlier loops are in that mic and end up in the new loop, and the take may not hear you
stop. Plume says so in `plume cue "list loops"` once it has heard the band in the room, but
does not fix it. A take from before the band comes in gets no warning, since there is nothing
to measure yet.

**With the band running,** the loop keeps your place in the bar, so a pickup stays a
pickup. It comes in at the top of its own cycle, which can be a few bars after you stop.
From then on the band holds that tempo until the last loop is dropped. It won't chase you,
and a cue like "faster" waits. A mute, swap or drop lands on the next bar plume hasn't planned
yet, which is usually the one after the one you hear.

**With nothing running** (`plume lead --tap-start`, before you count), the loop sets the
tempo and the band follows it. It needs at least four notes, and the fewer there are the
more evenly they have to sit. If you land on the next "one"
to finish, that gives it the phrase length directly. Doubled and halved tempos fit the same
notes equally well, so it picks the one nearer 100 bpm, which can be wrong. Saying the bars
("make a 4 bar loop") removes the guess. A take with no clear beat is usually refused, and
plume goes back to waiting for your count. "Usually" because this is odds, not proof: in
simulation at most about 1 random take in 20 still got a tempo, so laughter could start the
band, and about a third of steady 4-note takes with loose timing were refused.

**What it won't do yet:**

- It has no time-stretch that keeps pitch. A loop is fitted by padding or trimming its end,
  and only within 2%. Past that it is refused, and the reason is in `plume cue "list loops"`.
- A loop is cut from audio, never from MIDI notes. On a `--midi` take it reads the mic, so
  the keyboard's sound has to reach the mic, and with `--midi` plume can't warn you about
  the band in that mic.
- The last note is cut with a 20 ms fade rather than ringing over the top of the loop.
- The band plays at the level it always has, and loops give way to it: where the two
  would add up past full scale, the loops dip for a few milliseconds. The band's own peaks
  already go past full scale with no loops at all, and this doesn't change that. Within
  about 5 ms of those peaks the loops don't dip, since dipping on the peak itself saves
  nothing. Near a peak the loops can add some clipping: on a simulated device, 2.84% of
  samples went over full scale where the band alone had 2.65%. Nothing here was measured
  on real speakers.
- It can't be asked for by voice. A spoken "plume, stop" or "plume, restart" always acts on
  the take unless all of this is true, because a stop that fails is worse than one that ends
  a take by accident:
  - What you are stopping or restarting is the loop ("stop the loop", "stop this loop", "stop
    the bass loop", "stop the whole loop"), or the next sentence is "the loop" ("Plume,
    stop. The loop.").
  - Everything after it is one of: "now", "please", "right now", "it's too loud", "it's too
    much", "and keep playing", "keep going". Saying "stop" or "restart" the same way again is fine too.

  When all of that is true, nothing happens to the take and you type the cue instead. These
  do nothing: "plume, stop the loop", "plume, stop the loop right now", "plume, stop the loop,
  it's too loud", "plume, stop the loop. plume, stop the loop."

  Anything else ends the take, even if you meant only the loop: "plume, stop the loop and the
  band", "plume, stop the loop then the band", "plume, stop the loop, the band is fine",
  "plume, stop the loop and the bass", "plume, stop the loop too, please", "Plume, stop. Great
  loop.", "plume, stop great loop", "plume, stop, the loop", "plume, stop the band loop".

  The trade-offs: if the speech model cuts you off at "plume, stop the" or "plume, stop the
  loop and", the take ends, and "plume, stop that great loop" with no punctuation reads as a
  loop's name and does nothing. None of this was tried on real speech, only on the words the
  model might write.
- On a `--tap-start` take the silence ending still applies: stop playing and the take fades
  out, loops and all, about 20 seconds later on a direct input or MIDI. A room mic waits 30
  seconds before the fade starts, so it ends later than that, about 45 at 115 bpm.

Loops live in memory and end with the take. Only `--record` writes audio, into the same
`human.wav` and `machine.wav` as always, and `.gitignore` keeps those out of git.

## Recording

### Which sound goes where

| Setup | Delay measured | Hears you clearly? | Good for |
|---|---|---|---|
| Laptop mic in, projector or wired speakers out | 359 ms (projector over HDMI) | Over drums only | Filming: the room hears everything |
| Laptop mic in, AirPods out | about 317 ms | Yes, almost nothing leaks into the mic | Clean takes and late nights |
| AirPods mic in | | No | Don't use it |
| Instrument plugged in, headphones on the same interface | Not yet measured | Should be best | See below |

Measure your own setup, because reported numbers are wrong. macOS reported 141 ms for the
projector against a real 359, and 548 ms for the AirPods pair against a real 317.

**Speakers:** run `plume calibrate`. It plays clicks and listens for them, and saves the
result per device pair. If your clicks disagree by more than a few milliseconds, the room
is noisy or you made a noise during the test, so run it again. A Bluetooth speaker works
the same way, and calibration is what makes it usable.

**Bluetooth headphones:** calibration can't work, because the mic can't hear the clicks.
Plume falls back on the system's guess, capped at 250 ms. You will be playing to what you
hear rather than to your own knocks, so this affects where the drums first land more than
how the take feels.

**Never use the AirPods mic.** Opening it switches them to a phone-call codec: audio cuts
off at 12 kHz, a clap takes 50 ms to rise when it should take 1 to 3, and the output is
likely to stutter. Set your input to the laptop mic first. On a Mac that is System Settings,
or `SwitchAudioSource -t input -s "MacBook Pro Microphone"`.

### Plug the instrument in if you can

A guitar through an audio interface gives plume you alone. With only a room mic it has to
pick you out from a band of about the same volume, and it can only manage that while the
drums play alone, since drums barely favour any note. After the band joins, it can't tell
whether you changed chords or stopped playing.

```bash
plume inputs                                        # what's there, and how many channels
plume inputs --meter "audio interface"              # play: which channel moves?
plume lead --tap-start "lofi midwest emo in F major" \
  --device "audio interface" --input-channel 1 --direct --record takes/
```

Use part of the name `plume inputs` printed. `--device` names the interface for both directions, which keeps one clock and the lowest
delay. To keep your output elsewhere, use `--input-device` instead, which leaves it on the
system default, or add `--output-device NAME` to `plume lead` to name it, using its whole name. Channels count from 1,
as printed on the interface. `--direct` tells plume the band can't reach this input, which
turns off self-masking and lets the level alone say whether you are playing, even with the
band in.

For a mic with the band only in your headphones, add `--band-in-headphones` instead. The band
can't reach the mic then, so plume believes what it hears once the band is in: it follows your
chords and hears you stop. It is still a mic, which hears the room too, so before the band comes
in plume waits for notes, the same as on any mic, and with the band in it still goes by notes
rather than a level, so a noisy room doesn't keep the band from hearing you stop. If the band
can be heard in the room at all, leave it off, and answer No when setup asks. With `--midi` the
band listens to the keys, so the flag changes nothing there.

Plume checks this for itself, in `plume lead` with or without a count and in `plume jam`. For
the whole take it measures how much of the mic is the band's own sound, as it would arrive through
a speaker and a room: it compares the two between 300 Hz and 4 kHz, the range a small speaker
plays, and allows for the speaker's and the room's colouring rather than looking for an exact
copy. A player's notes sit on a few frequencies, so the band is found under playing too. If the
band is clearly there for about eight seconds, because the headphones came out or the band was
on speakers after all, the take stops going by what the mic hears and treats it as a room mic
for the rest of the take. It doesn't switch back, and a fade already under way carries on. The
log row where it decides carries `band_in_mic` with the bar, the share it measured and the
reason, the live page says so in one line, and Stop is then how the take ends. This is also what
covers headphones being unplugged where the list of devices doesn't change. From then on the
take masks its own sound in the mic as it would for a room mic from the start, so it still
hears where your notes land.

This setup hasn't been played yet: it has only run against a fake device. The thresholds
were tuned for a room mic and will likely need adjusting. Run
[`DIRECT-INPUT-TEST.md`](DIRECT-INPUT-TEST.md), which takes about ten minutes. `plume
calibrate` can't help here, because the input can't hear the clicks, so plume uses the delay
the system reports.

A mic close to the amp or the guitar, with the speakers further away, is the cheaper step
in the same direction.

### A second reading of the key

Plume names your key from how much sound there is on each of the twelve notes. Overtones
count there too, so a major key can come out as the key a fifth up, or as the minor key on
its third. With the optional `notes` extra it also reads the notes themselves, with a
small pitch model, and goes by both readings together.

```bash
.venv/bin/pip install -e '.[play,notes]'
```

Nothing else changes: the same commands, and no flag. It works on a mic and on a direct
input. With `--midi` your keys are the notes already, and the model is not used.

The model is 230 kB and comes with plume; what the extra installs is `onnxruntime`. It runs
beside the take on a thread of its own, one two-second window each second, and takes 3 to
8% of one processor core and about 40 MB of memory.

`PLUME_NOTES=0` switches it off for a take, and the key comes from the first reading alone.
`PLUME_NOTES=inline` is for replaying a recording faster than real time: it runs the model
in step with the audio instead of beside it.

The notes never turn a minor key into the major key with the same notes (A minor into C
major, say): between those two the first reading, which hears your low notes, decides,
unless the notes say minor. So far this has only been measured on rendered parts and on a public set of guitar
recordings, through a fake device.

When the band is ready to come in but two keys are about level, it waits for one more of
your phrases and then comes in on the key it hears by then. It waits like this once at
most, with or without the extra.

### MIDI

A keyboard or pad controller can tell plume exactly what you played, instead of plume
working it out from a mic. It needs the optional `midi` extra. So far this has only run
against a simulated port, never a real keyboard, so treat what follows as how it is
meant to behave.

```bash
.venv/bin/pip install -e '.[play,midi]'
plume midi                                  # lists your ports; opens none of them
plume lead --tap-start "lofi at 90 bpm" --midi auto --record takes/
```

`--midi` takes part of a port's name, or `auto` when there is only one. It works with
`jam`, `lead` and `voice`. Your keyboard's own sound still has to reach your ears, since
plume doesn't play your notes back.

What changes:

- Your notes, timing and how hard you played go to the same beat and harmony code the
  mic feeds. The band's sound can't get mixed up with yours, so what plume hears stays
  yours after the band joins, and it can tell when you stop.
- To count in with `--tap-start`, press any key four even beats. Any velocity counts.
- The sustain pedal holds notes. Channel 10 counts as drums: hits with no pitch.
- A MIDI clock is ignored unless you add `--midi-clock`, because keyboards and DAWs often
  send one you aren't playing to. Plume prints a line the first time it ignores one.
- With `--midi-clock`, the clock sets the tempo and the bar line, and it beats what you
  play: if you play at another tempo or off its beat, the band stays with the clock. A
  Start message makes the next tick beat one. Without Start, the first tick plume hears
  is beat one, which may not be yours. `jam` follows the clock, tempo changes included.
  `--tap-start` still waits for your count and then holds it.
- When the clock ends, plume goes back to reading the tempo from your playing, which took
  about 10 seconds in a simulated take. A clock has ended when a Stop message arrives or
  a second goes by with no ticks. Continue picks up in the same place in the bar. After a
  cable pull or a silent gap, the next tick is beat one again.
- The mic stays open and is still saved to `human.wav`, so `plume voice` still hears you.

If the cable comes out, the session keeps going. Held notes are let go, plume prints a
line, and the band hears silence until the port returns, when it reconnects by itself. A
port that hasn't opened after a second counts the same way, so a stuck driver can't hold up
a take or stop you ending it. A driver that won't list its ports within 5 seconds gets the
session refused before it starts. All of that was tested with a simulated driver, never a
real one.
Under `--tap-start`, that ends the take the way any silence does: about 20 seconds later once
you have played a bar. Until then the fade waits 30 seconds first, so it takes longer. `--record`
adds `midi.ndjson`, every note with its time.

Things to watch for:

- Even single notes at a slow tempo can read as double time, because nothing marks which
  ones are the beat. Chords on the beat fix this, and so does counting in.
- With no real keyboard tested yet, the timing between your keys and the band is the part
  to check first. It relies on the input latency the audio system reports, and reported
  latencies have been wrong before. If the band feels late or early against your keys,
  say so; the fix belongs in that one number.
- A port is matched by its exact name when reconnecting. A system that renames a port
  when it returns won't reconnect, so restart the take.

### Sending the band to Ableton

Plume can send its band out as MIDI, so the instruments in Ableton Live, or any other
DAW or synth, play it. Plume still picks the tempo, key, style and every note; the other
program only makes the sound, and can record it. It needs the same `midi` extra. So far
this has only run against a simulated port, never a real copy of Ableton.

```bash
plume lead --tap-start "lofi at 90 bpm" --midi-out virtual               # plume's band and MIDI
plume lead --tap-start "lofi at 90 bpm" --midi-out virtual --midi-only   # Ableton is the band
```

`--midi-out` takes part of a port's name, or `virtual` to make a new port called `plume`
while the take runs. It works with `jam` and `lead`. Each part has its own channel:

| Channel | Part |
| --- | --- |
| 1 | bass |
| 2 | pad and keys |
| 3 | lead |
| 4 | arp |
| 5 | shimmer |
| 6 | sub (a tuned 808) |
| 7 | any other part with notes |
| 10 | drums, on General MIDI notes: kick 36, rim 37, snare 38, clap 39, closed hat 42, pedal hat 44, open hat 46, crash 49 |

`--midi-only` keeps plume's own band silent, so you hear only Ableton's. Your own sound
on Hear me, and loops, still come from plume, and a loop still makes room for the band
as if plume were playing it. With `--record`, `machine.wav` then holds no band, but the
take's folder still has it: any take with `--midi-out` and `--record` writes `band.mid`
when it ends, a standard MIDI file of every note sent. It has one track per part, on the
channels above, and the take's tempo, and its time 0 is the start of `machine.wav`, so
dragged into a DAW at the take's start each note lands where plume's own sound of it
would have. Use `--midi-only` with `--midi`, a direct input or headphones: on a room
mic plume can't take Ableton's band out of what it hears, so it would hear the band as
you.

`--midi-out-offset` goes from -500 to 500 ms. Notes whose sound plume had to drop,
because their bar came too late, aren't sent either.

When a take ends, however it ends, plume lets go of every held note and sends All Notes
Off on each channel it used. A take also starts by sending All Notes Off on all of its
channels, so if plume was ever killed outright and left a note hanging, the next take
clears it.

Don't send the band to the port plume listens to with `--midi`: it would come straight
back in as your playing. Plume warns when both name the same port. The same loop happens
if Ableton sends its MIDI out to the IAC port plume listens on, so keep Ableton's MIDI
outputs off that port.

**macOS.** Either use `--midi-out virtual`, or turn on the IAC Driver once: open Audio
MIDI Setup, choose Window > Show MIDI Studio, open IAC Driver and tick "Device is
online". Then use `--midi-out IAC`. The IAC port is there even when plume isn't running,
so Ableton keeps its settings between takes; a virtual port only exists during a take.

**Windows.** Windows can't make a port by itself. Install loopMIDI, add a port called
`plume`, and use `--midi-out plume`.

**Linux.** `--midi-out virtual` works with ALSA, for synths that run there.

**In Ableton Live 12** (start the take first if you use `--midi-out virtual`, so the
port exists):

1. Live > Settings (Options > Settings on Windows) > Link, Tempo & MIDI: on the input
   port (`plume`, or IAC Driver), turn Track on. Leave Sync and Remote off.
2. Make one MIDI track per part. Set MIDI From to the port and the part's channel, for
   example Ch. 1 for bass and Ch. 10 for drums. Set Monitor to In.
3. Put an instrument on each track. A Drum Rack on channel 10 already has kick on C1
   (36), snare on D1 (38) and hats on F#1 (42).
4. Arm the tracks to record what plume plays into clips or the arrangement.

**Timing.** Plume sends each note when its own sound of that note would reach your
speakers. Ableton then adds its own output latency, so with `--midi-only` the band can
sound a little late. Ableton shows that delay as Overall Latency in Settings > Audio.
Send the notes that much earlier with a negative offset, for example
`--midi-out-offset -12` for 12 ms. Ableton's tempo doesn't follow plume yet, so turn
off anything in Ableton that syncs to its own tempo, or set its tempo to plume's.

What isn't sent yet: fades and the bus volume (only how hard each note is played), tone
cues such as darker or wetter, the vinyl crackle, and a tempo clock.

### Filming

Point a phone at yourself and start recording before you count. With speakers, the phone
hears the whole band live, so there is nothing to sync afterwards. That is the simplest
setup, and the audio will be as good as the phone's mic and your room.

If you want a cleaner track, or you are on headphones, every `--record` directory holds
two files to work with:

- `human.wav` is the mic. On speakers, this is already a recording of the room with
  everything in it, and for the example above the player preferred it to every mix.
- `machine.wav` is the band alone. The two are not aligned: to line them up, drop the first
  part of `human.wav` by your measured delay. A take from a direct input has Together under its picture on the page, which lines them up by the delay the device reported and mixes them as it plays.

To swap audio into a phone video, line it up on your count, since knocks are easy to see
in a waveform. On headphones, turn phone Bluetooth off so the AirPods stay with the Mac.
Phone videos can carry your location, so strip the metadata before posting one:
`ffmpeg -i in.mov -map_metadata -1 -c copy out.mov`.

## Starting from the page

`plume gui` opens a page in your browser with one Start. The page opens no device and
starts nothing until a person clicks Start. That click is the consent plume otherwise asks
a terminal for. It runs `plume lead --tap-start --detect` for you, in a new timestamped
folder under `~/Music/Plume` (or `--takes DIR`). A used folder is never recorded over.

### Which device it picks

With no typing, it picks the device setup heard you on, or else the only audio interface
it can find, using the same device for sound in and out. With two interfaces, setup
listens to both and keeps the one that heard you. With none, it uses the built-in mic.
Settings, next to Takes, changes any of this, and one small line under Start says what
it will use.

Plume keeps no list of makers. A device with inputs and outputs counts as an interface
unless its name says it is software (BlackHole, a Zoom or Teams driver), the computer's
own (built-in, internal, a microphone array, a laptop, a display or webcam), or a mic or
something wireless. Those all play as room mics. This is a guess from a name, and a laptop
chip with a name plume does not know can pass as an interface, because its mic does hear
you. When the guess is wrong, a take that hears the band in a direct input says so on the
first view and tells you to try another device.

One small choice sits under Start: Style, on Let plume guess until you pick one. A picked
style is passed to the take as the prompt, the same as naming it in a terminal, so the band
plays it from the first bar and the live guess never runs. The pick is saved in
`setup.json` with the rest of setup (Set up again keeps it), and picking Let plume guess
goes back to guessing. A style plume does not have is refused in plain words, and the
choice goes back to what was saved.

Key, in Settings, says Detect until you type over it. The screens say what was chosen and
what was guessed: "F major (a guess) · Folk, plume's guess", or "F major · Folk, chosen".
Each take records it too: `style_chosen` in `session.json` and on each row of
`decisions.ndjson`, beside `style_guess`. An older take that says neither shows just the
style.

### During a take

After setup the page says to knock or clap 8 even beats. On an interface it says to play
them on muted strings, since a clap can't reach a pickup. Once the band is in, the first
view is what plume settled on, a small sign that it hears you, and Stop.

Everything else is in one Details menu: the bar, the tempo, the stage, the notes it hears
from you and your level, the delay, and the Busier and Sparser cues. A field plume can't
trust over the band is hidden rather than guessed at. On a room mic that is most of them
once the band is in, and the sign then says it hears the room. That is all it knows, so it
makes no claim about you.

On a direct input or MIDI, when you have played and then go quiet for more than a bar, one
line says "Stop playing and the band will finish on its own." During the fade the big line
says the band is finishing, and that playing brings it back. A room mic may never hear you
stop, so nothing promises that there. Instead, for a few bars once the band is in, one line
says to press Stop when you are done, and Details keeps saying it.

When the take ends, by fading, by Stop or by a crash, the page lands on its Record with a
New take button, or says in plain words what went wrong.

Stop before the band has come in and no take is kept, so trying a wrong channel leaves
nothing on the shelf. If plume heard nothing, the page goes to Settings and says so. If
you played for more than a minute and the count never caught, the recording is kept.

### Hear me

Hear me plays your input back to you. It lives in Settings and is off until you say
otherwise, on any input, including a built-in mic. Through speakers it would feed back,
and plume cannot tell headphones from speakers, so the bridge refuses Hear me without a
yes about headphones saved for that device. Ticking the box is that yes, saved when you
tick it. Start never carries one, so a page left open cannot bring it back, and on a room
mic a change in what is plugged in takes the answer away.

Hear me stays out of `human.wav` and `machine.wav` on every input. Details shows the delay
your device reports, which can be less than the real one, since nobody has measured it. A
Direct or monitor button on your interface has no delay. Using both doubles the sound.

### One take at a time

Only one take runs at a time, including one started from a terminal: the page then offers
to stop it instead of offering Start. Reloading finds the running take again. Closing the
page doesn't stop a take that fades by itself. A take still waiting for a count, or on a
room mic that can't hear you stop, ends about 3 minutes after the last page stopped looking
at it. `plume stop` works as usual, and Ctrl-C on `plume gui` stops the take it started.

### The first press

The first press of Start is the setup, one line at a time, in the same place Start was:

1. "Play something." Plume listens for a few seconds on a short child that plays nothing, keeps the loudest channel and checks the level with no questions. If it is hot or barely there it says what to turn, and you can play again or start anyway. If nothing arrives it names the device, and if nothing arrives or it is barely there it offers any other device to try. With nothing plugged in it says to plug something in.
2. "Headphones on?" No means speakers, including filming with a camera in the room, and Hear me stays off. On a mic, the same answer also saves whether the band is only in your headphones: yes means the mic is believed once the band is in. That is saved as its own answer, for that mic, channel and output. Ticking Hear me in Settings never saves it, and an answer saved before plume asked this doesn't count, so you are asked once: "Is the band only in your headphones?"
3. If yes, "Can you hear yourself?" Yes means something else is already sending the sound, such as a monitor button or an instrument you can hear in the room, so Hear me stays off. No asks you to check the volume first, and "Still nothing" turns Hear me on.
4. If you said headphones, an output that reports a large delay, or looks wireless (which it says is a guess), gets one line suggesting wired headphones.
5. Then the count, with no click, and the band coming in is the end of the setup.

After that, Start goes straight to the count. The answers stay in `setup.json` in plume's
state folder (`~/.plume`, 0600), by device ID, with no device names. A different device, or
a change in what is plugged in, asks again for only what it needs. Plugging headphones in
when it asks is fine. If the device it heard you on goes away during setup, it says so
rather than moving the take to another one. Set up again, in Settings, forgets all of it.

No device stays open between steps. Not now, the wordmark, Takes, Back, a reload or
closing the tab all leave at any step, call off a listen in progress and leave nothing
running, and only a click ever starts a listen. If a closing tab cannot get that word out,
the short child still ends by itself within about 20 seconds. A failed step offers Start
anyway, with Hear me off unless you said otherwise. The level and delay numbers are
guesses, since no real interface has been measured against them.

All of this has only run on fake devices and a fake process, in tests and in headless
Chromium with the sound muted. It has never started a real take, so follow
[the direct input test](DIRECT-INPUT-TEST.md#7-the-start-button-5-minutes) the first time.
Only one Mac and one interface have been tried with plume at all, so other computers and
other gear are untested. The names it uses to recognise software and laptop devices on
Linux and Windows come from what those systems usually list, not from a real machine. On
Windows each device is listed once, from the default sound system (MME), so names can be
cut short and the delay is longer than it needs to be. Nobody has played through plume on
Windows yet. There, closing the window a take runs in gives plume about five seconds
before Windows ends it, too short for its usual ending, so the tail is cut. The recording
up to then is safe. End a take with Ctrl-C or `plume stop` instead.

### Another output

Output in Settings sends the band somewhere other than your input's own outs, so people in the room can hear it while your guitar still records through your interface. It starts on Same as input, or System output on a room mic, and either of those is exactly what plume did before there was a choice. The choice is saved per input in `setup.json`, by ID. Choosing opens nothing: sound still starts only from Start. An output that has gone is refused in plain words at Start and is never swapped for another. A Direct or monitor button on your interface only reaches that interface's own outs, so your guitar reaches the other output only through plume.

So on a direct input Settings offers one more tick, off by default: Send my instrument to this output. It arrives slightly late, and it is only for an instrument plugged in, since a mic on that channel would howl. Plume cannot tell an instrument from a mic, so the tick is your word for it, and it is for that output and that channel only. Change either and it goes off. For Hear me, plume treats another output as speakers, so a yes about headphones does not reach it: Hear me saved for your headphones stays with your interface's outs, and comes back when you do. A mic is never sent through another output at all, and setup does not ask "Headphones on?" or "Can you hear yourself?" while one is chosen, since those are about your own output. A mic is asked one thing there: "Is the band only in your headphones? If you can hear it in the room at all, say No.", with the output named beside it. Yes means the band can't reach the mic, so plume believes what it hears once the band is in. The answer is for that mic, channel and output: change any of them, or what is plugged in, and it is asked again. It never turns Hear me on.

Wireless speakers arrive late, which makes playing along hard. Plume cannot tell wireless from a name, so it says this for any other output.

The choice belongs to one input, and with more than one interface plugged in setup can hear you on a different input from the one Settings was showing. It then asks whether to send the band to that output from the new input too, before anything plays. Say no and that input keeps its own. If Settings showed no output and the new input has one saved from before, it asks about that one the same way, and a no clears it. If an answer does not save, the question stays until one does.

**Not yet tried on real gear.** All of this has only run against a fake device. Three things only real gear can answer:

- Two devices have separate clocks. Over a take they can drift apart, and plume does nothing about that. Nobody has measured how far, or what it sounds like.
- Some pairs will not open together, or will not open at 48 kHz. Plume then says so on Settings rather than crashing, and offers another output. Which pairs refuse is not known.
- `stream.json` records what the stream reports for each direction and their sum, and Together lines you up on that sum. With two devices the audio library may hold extra sound between them that it does not report, so Together could sit a little off on these takes. A `plume calibrate` figure is kept per pair, so one from another pair is never used here. Compare Together with what you heard, and note any gap.

## Reading a take

`plume gui --take takes6/` opens The Record in your browser: a strip of what plume did and
when, a plain sentence for each moment, and a roast. It opens no device by itself, and it
plays a recording only when you press Play. `plume gui --takes DIR` shows a shelf of take
folders, each with its own carved stone.

Keep writes a small `keep.json` beside the take and deletes nothing, and `--record`
refuses a folder holding a kept take. A name that fails to save says so on the next screen
you land on, then clears.

Show files opens the take's folder in your file manager and saves `mix.wav` there first:
`human.wav` and `machine.wav` as one file, the player moved earlier by the round trip in
`stream.json` (the measured one if there is one, else the one the device reported) and
balanced the way Together plays them. The file ends with a note of what it was made from
(the mixing rule, the delay used, each recording and `stream.json` by size and time, and a
fingerprint of the mixed sound),
and it is made again whenever any of that differs. Recording again into the same folder
removes it.

Plume only ever replaces or removes a `mix.wav` that is still exactly the file it wrote.
The note at the end carries a fingerprint of the sound, and plume checks the whole file
against it. If you have changed the file in any way, edited it, made it louder, trimmed it,
tagged it, or saved it from another program, it is yours: plume leaves it where it is and
tells you. The same goes for a `mix.wav` plume never made, and for one of plume's that got
damaged. Move or rename it and plume will save a new one.

The page says nothing when the mix is plain. Otherwise it says one line:

- no mix was saved: a take with one recording, one still being recorded, one whose two
  recordings have different sample rates, one too long for a WAV, or a folder plume cannot
  write in;
- no new mix was saved and an older one of plume's is still in the folder, so the `mix.wav`
  there is out of date;
- the folder already has a `mix.wav` plume can't replace, so it was left: one of yours, one
  of plume's that another program has since changed, or one that is locked. Move or rename
  it to get a new one;
- plume's own older mix is open in another program and could not be updated (Windows, where
  an open file cannot be replaced);
- the take was recorded from a mic, which may have heard the band, so the band could be in
  the mix twice. Only a take from a plugged-in instrument is free of that line today.

If the file manager does not open, the page says where the folder is.

### A video of your take

Add a video takes a video you filmed while you played, which heard you and not the band in
your headphones, and saves it again as `video.mp4` in the take's folder with the band and
your recording in its sound. Choose the file or drop it on the take's page.

- Plume finds where the take begins by matching `human.wav` against the video's own
  sound. The video can be longer than the take, shorter, or start part way through it.
- Your recording goes in at that place and the band's the round trip later, balanced the
  way Together and `mix.wav` are. With **Keep the video's sound** on, the camera's sound
  stays under them at its own level for the whole video. Off, only the band and your
  recording are heard.
- The picture is copied as it is. The sound is AAC. Tags the camera wrote, such as where
  and on what it was filmed, are left out.
- If the match is not clear, or the video has no sound, nothing is saved yet. The page
  shows a slider to put the take in place, with a few seconds to listen to first when the
  video has sound.
- Only plume's own `video.mp4` is ever replaced. One of yours is left, and the page says so.
- The video is copied into the take's folder while plume works, so the disk needs room for
  it twice. The copy is removed when the work ends, is cancelled, or fails. One left by a
  plume that closed unexpectedly is removed the next time plume starts or lists your takes,
  and never while any plume still holds it, however long it waits for a nudge.

This needs ffmpeg. Installing plume brings one. Without it, plume uses one on your PATH,
and with neither the page says so.

Checked on made-up takes and tiny made-up videos. The whole path has not yet been run in
a real browser with a real video.

This page has only been checked on made-up take folders, in headless Chromium with the
sound muted. Play has not been pressed in a real browser, and the page has not been tried
on a take from a real mic or interface, or in Firefox or Safari.

Once the band is in on a room mic plume cannot hear you, so the page says so and makes no claim about your playing. On a direct input or a `--midi` take it can still hear you, so the page drops that line, but it still makes no claim about your playing, because neither has been tried on real hardware. Takes recorded before `cues.ndjson` existed can only say "A cue landed", not who asked or when.

Every cue that reaches a session goes in `cues.ndjson`, including what the looper answered, and refusals. The page reports the looper's answer and nothing more, even for a take that stopped before its first bar: a take it agreed to can still end with no loop. A cue that changes your own sound (`sound dreamy`, with Hear me on) is answered at once and logged the same way, and the page says what your sound became, or that plume has no such sound. MIDI notes do not send cues; a cue sent during a `--midi` take is logged like any other. None of this has been run on a real interface, keyboard or speakers, only on the simulated device.

## When it goes wrong

| What you notice | Likely cause | Try |
|---|---|---|
| No drums after counting | Knocks too quiet or uneven | Knock harder, closer to the mic, or clap |
| Drums start by themselves | Something in the room looked like a count | Restart, and stay quiet until you count |
| Tempo feels double or half | You counted a different pulse than you meant | Count the beat you want to feel; plume won't correct it |
| Band never joins | Plume hasn't heard steady notes | Play sustained notes over the drums |
| Doesn't fade when you stop | The band is in, and a room mic can't tell you stopped | Say "plume, stop" or run `plume stop`; a direct input or `--midi` fixes it |
| A spoken command does nothing | Look at `voice.ndjson` in the take | `DIRECT-INPUT-TEST.md` says what each line means |
| `plume voice` quits at once on an NVIDIA machine | The GPU libraries are only half there | Set `PLUME_WHISPER_DEVICE=cpu` |
| `live MIDI needs mido` | The `midi` extra isn't installed | `pip install -e '.[midi]'` |
| `more than one MIDI port` | `--midi auto` won't guess | Run `plume midi` and name one |
| Band sticks to the wrong tempo over MIDI | `--midi-clock` is on and your gear's clock isn't what you're playing | Drop `--midi-clock` |
| No sound from the projector | HDMI audio dropped or output changed | Check the output device, or use a speaker and calibrate |
