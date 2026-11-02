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

To stop from another terminal, run `plume stop`, or `plume sessions` to see what is
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
| "plume, stop" | ends the take |
| "plume, restart" | ends the take and waits for a new count |
| "plume, busier" | more hats, keys and melody from the next phrase |
| "plume, sparser" | less of them |

Every command needs the word "plume", so a lyric can't end your take. It reacts a few
seconds late, because it listens in four-second windows. This has not yet been proven
on a live voice over a loud band: the recognition is tested, but the one real session
it ran in never caught a spoken command, so speak up and keep a keyboard nearby.

## With a coding agent as the brains

An agent is far too slow to play notes, but it is fast enough to decide what the next
section should do. Plume's bar-by-bar playing stays local, and the agent steers from
outside it.

```bash
# start the take. The variable is deliberate: without a terminal attached, plume
# refuses to make sound unless told to, so an agent can't start it by accident.
PLUME_ALLOW_AUDIO=1 plume lead --tap-start "lofi midwest emo in F major" --record take/ &

# read what is happening: one JSON line every 8 bars when something changes
plume watch take/ --every 8 --changes-only

# steer. It lands at the next phrase
plume cue "busier"
plume cue "sparser"

plume stop
```

A `watch` line looks like this:

```json
{"bar": 31, "section": "lift", "stage": "full", "key": "F major", "chord": "Fmaj7",
 "you_db": -24.7, "band_db": -25.1, "mic_db": -24.4, "you_busy": 5.38,
 "you_notes": ["A", "F", "C"], "changes": ["section:lift", "you:busier"]}
```

Trust `section`, `stage`, `key` and `chord`, which report what the band is doing. Don't
trust the `you_*` fields once the band is in on a room mic. They measure the mic with
plume's sound subtracted, and that subtraction removes only a few dB in a real room. In one
test an agent reported notes the player was "playing" while the player sat silent.
They are only reliable over a direct input, or over the drums alone.

Agents running tests should use the fake audio device in `tests/conftest.py`. An agent
that runs a live verb to check its work plays music out loud in whatever room the machine
is in.

## Recording

### Which sound goes where

| Setup | Delay measured | Hears you clearly? | Good for |
|---|---|---|---|
| Laptop mic in, projector or wired speakers out | 359 ms (projector over HDMI) | Over drums only | Filming: the room hears everything |
| Laptop mic in, AirPods out | about 317 ms | Yes, almost nothing leaks into the mic | Clean takes and late nights |
| AirPods mic in | | No | Don't use it |
| Instrument plugged in, anything out | Not yet measured | Should be best | See below |

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

Set the interface as your system input and leave the output where you want it. Avoid
`--device` for this, since it applies one name to both directions. This setup hasn't been
tested yet. The silence ending, the `you_*` fields and following your changes should all
start working, but the thresholds were tuned for a room mic and will likely need adjusting.

A mic close to the amp or the guitar, with the speakers further away, is the cheaper step
in the same direction.

### MIDI

Plume has no live MIDI input yet, so a keyboard has to reach it as sound: its speaker,
or a line into an interface. The planned version would read notes directly, which removes
the guesswork about what you played. It is part of the looper on the roadmap in
[`DREAM.md`](../DREAM.md).

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

## When it goes wrong

| What you notice | Likely cause | Try |
|---|---|---|
| No drums after counting | Knocks too quiet or uneven | Knock harder, closer to the mic, or clap |
| Drums start by themselves | Something in the room looked like a count | Restart, and stay quiet until you count |
| Tempo feels double or half | You counted a different pulse than you meant | Count the beat you want to feel; plume won't correct it |
| Band never joins | Plume hasn't heard steady notes | Play sustained notes over the drums |
| Doesn't fade when you stop | The band is in, and it can't tell you stopped | Say "plume, stop" or run `plume stop` |
| No sound from the projector | HDMI audio dropped or output changed | Check the output device, or use a speaker and calibrate |
