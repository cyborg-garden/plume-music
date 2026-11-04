# Playing with plume

Plume plays a band around you through your Mac's speakers while you play a real
instrument into its mic. You count it in, it holds your tempo, and the band joins
once it hears you start.

To hear what this sounds like, listen to
[`examples/first-keeper.m4a`](examples/first-keeper.m4a): one take on a guitar in
FACGCE tuning at 53 bpm, recorded by a laptop mic across the room from a projector's
speakers.

## Your first take

```bash
.venv/bin/pip install -e '.[play,voice]'
plume calibrate        # plays clicks; skip this on Bluetooth (see below)
plume lead --tap-start "lofi midwest emo in F major" --input acoustic --record takes1/
```

Nothing plays until you count. Knock on the guitar body or clap eight even beats, and
the drums land where the ninth would be. Plume needs four clear hits or six soft ones
to catch the count. Mouth clicks are usually too quiet, and if your first few knocks
are faint it still finds them once the later ones confirm the tempo.

The drums stay alone until you play. The bass and chords come in about eight bars after
you start, and the melody four bars after that. If you play
nothing for 30 seconds, the take fades out and ends.

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
plays half-time trap with the snare on 3. The other styles are `lofi`, `trap` and
`synthwave`, and a new take with a new `--seed` gets a new melody.

## Talking to it

`plume voice` starts the same take and also listens for spoken commands. Speech is
transcribed on your Mac with Whisper, and nothing leaves the machine.

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
stops I'll come in" all end the take. Some plain ways to say stop are still missed, and one with
more than 8 words before "stop" is too, so keep a keyboard nearby. None of this has met real speech. It reacts a few
seconds late, because it listens in four-second windows. This has not yet been proven
on a live voice: from a transcript to a change in the running take is tested, but the
one real session it ran in never caught a spoken command, and nothing recorded what the
model had heard. Speak up and keep a keyboard nearby.

Each take now keeps `voice.ndjson`, with one line per window: how loud it was, what the
model wrote, and whether that counted as a command. `plume voice-replay takes/human.wav`
asks the same question of a recording that already exists, with no band and no sound.

A second mic on its own channel keeps your voice apart from the band, and on a plugged-in
guitar it is the only way plume can hear you: add `--voice-channel N` (see
[`DIRECT-INPUT-TEST.md`](DIRECT-INPUT-TEST.md)).

## With a coding agent as the brains

An agent is far too slow to play notes, but it is fast enough to decide what the next
section should do. Plume's bar-by-bar playing stays local, and the agent steers from
outside it.

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
 "changes": ["section:lift", "you:busier"]}
```

Trust `section`, `stage`, `key` and `chord`, which report what the band is doing. The
`you_*` fields come with their own verdict. Every line has `input` (`room`, `direct` or `midi`),
`you_trusted` (the names of the `you_*` fields that describe the player) and `you_why`.

| Situation | `you_trusted` |
|---|---|
| Room mic, band in | none |
| Room mic, drums alone | `you_notes` only |
| `--direct`, `--midi`, or the band not sounding | all three |
| `--direct`, but the band turns up in the input | none, and `you_why` says to check routing |

Report nothing from a field that isn't listed. `you:*` entries in `changes` come from the
same fields, so they get the same doubt. On a room mic those fields measure the mic with
plume's sound subtracted, and that subtraction removes only a few dB in a real room, so a reader who skips this check will report notes
from the player while the player sits silent.

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
- On a `--tap-start` take the silence ending still applies: 30 seconds of you not playing
  fades the take out, loops and all.

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
likely to stutter. Set your input to the laptop mic first, in System Settings or with
`SwitchAudioSource -t input -s "MacBook Pro Microphone"`.

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
delay. To keep your output elsewhere, use `--input-device` instead. Channels count from 1,
as printed on the interface. `--direct` tells plume the band can't reach this input, which
turns off self-masking and lets the level alone say whether you are playing, even with the
band in. It also applies to a mic when the band is only in your headphones.

This setup hasn't been played yet: it has only run against a fake device. The thresholds
were tuned for a room mic and will likely need adjusting. Run
[`DIRECT-INPUT-TEST.md`](DIRECT-INPUT-TEST.md), which takes about ten minutes. `plume
calibrate` can't help here, because the input can't hear the clicks, so plume uses the delay
the system reports.

A mic close to the amp or the guitar, with the speakers further away, is the cheaper step
in the same direction.

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
Under `--tap-start`, 30 seconds of that ends the take the way any silence does. `--record`
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

### Filming

Point a phone at yourself and start recording before you count. With speakers, the phone
hears the whole band live, so there is nothing to sync afterwards. That is the simplest
setup, and the audio will be as good as the phone's mic and your room.

If you want a cleaner track, or you are on headphones, every `--record` directory holds
two files to work with:

- `human.wav` is the mic. On speakers, this is already a recording of the room with
  everything in it, and for the example above the player preferred it to every mix.
- `machine.wav` is the band alone. The two are not aligned: to line them up, drop the first
  part of `human.wav` by your measured delay. There is no mix command yet.

To swap audio into a phone video, line it up on your count, since knocks are easy to see
in a waveform. On headphones, turn phone Bluetooth off so the AirPods stay with the Mac.
Phone videos can carry your location, so strip the metadata before posting one:
`ffmpeg -i in.mov -map_metadata -1 -c copy out.mov`.

## Reading a take

`plume gui --take takes6/` opens The Record in your browser: a strip of what plume did and when, a plain sentence for each moment, and a roast. It opens no audio device, and it plays a recording only when you click Listen. `plume gui --takes DIR` shows a shelf of take folders, each with its own carved stone. Keep writes a small `keep.json` beside the take and deletes nothing, and `--record` refuses a folder holding a kept take. A name that fails to save says so on the next screen you land on, then clears. This page has only been checked on made-up take folders, in headless Chromium with the sound muted. Listen has not been pressed in a real browser, and it has not been tried on a take from a real mic or interface, or in Firefox or Safari.

Once the band is in on a room mic plume cannot hear you, so the page says so and makes no claim about your playing. On a direct input or a `--midi` take it can still hear you, so the page drops that line, but it still makes no claim about your playing, because neither has been tried on real hardware. Takes recorded before `cues.ndjson` existed can only say "A cue landed", not who asked or when.

Every cue that reaches a session goes in `cues.ndjson`, including what the looper answered, and refusals. The page reports the looper's answer and nothing more, even for a take that stopped before its first bar: a take it agreed to can still end with no loop. MIDI notes do not send cues; a cue sent during a `--midi` take is logged like any other. None of this has been run on a real interface, keyboard or speakers, only on the simulated device.

## When it goes wrong

| What you notice | Likely cause | Try |
|---|---|---|
| No drums after counting | Knocks too quiet or uneven | Knock harder, closer to the mic, or clap |
| Drums start by themselves | Something in the room looked like a count | Restart, and stay quiet until you count |
| Tempo feels double or half | You counted a different pulse than you meant | Count the beat you want to feel; plume won't correct it |
| Band never joins | Plume hasn't heard steady notes | Play sustained notes over the drums |
| Doesn't fade when you stop | The band is in, and a room mic can't tell you stopped | Say "plume, stop" or run `plume stop`; a direct input or `--midi` fixes it |
| A spoken command does nothing | Look at `voice.ndjson` in the take | `DIRECT-INPUT-TEST.md` says what each line means |
| `live MIDI needs mido` | The `midi` extra isn't installed | `pip install -e '.[midi]'` |
| `more than one MIDI port` | `--midi auto` won't guess | Run `plume midi` and name one |
| Band sticks to the wrong tempo over MIDI | `--midi-clock` is on and your gear's clock isn't what you're playing | Drop `--midi-clock` |
| No sound from the projector | HDMI audio dropped or output changed | Check the output device, or use a speaker and calibrate |
