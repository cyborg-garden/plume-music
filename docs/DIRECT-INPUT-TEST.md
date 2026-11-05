# Direct input test

About fifteen minutes, with a guitar, an interface and headphones. Everything here has only
run against a fake device in the tests, so this is the first time any of it meets a real
pickup. Plume makes sound during steps 3, 5 and 7, and you are the one who starts it.

## 1. Set up (1 minute)

```bash
.venv/bin/pip install -e '.[play,voice]'
```

- Plug the guitar into the interface and set that input to instrument (INST), if it has
  the switch. Some small interfaces only take a guitar on one socket, so check the front.
- Plug headphones into the interface. Turn the interface's Direct button on so you can hear
  yourself. It has no latency.
- Set the interface to 48 kHz (on a Mac, in Audio MIDI Setup).
- Leave plume's `--monitor` off, or you'll hear yourself twice. If the interface has no such
  button, use `--monitor` instead: it follows `--input-channel` and stays out of both
  recordings, but it adds the delay your device reports and has no limiter, so start quiet.

## 2. Find the channel (1 minute)

```bash
plume inputs
plume inputs --meter "audio interface" --seconds 8     # play hard for the whole 8 seconds
plume inputs --meter "audio interface" --seconds 8     # now hands off, guitar volume up
```

Use part of the name `plume inputs` printed. The meter plays nothing.

- **Playing:** `loudest` should be your socket's number, with its peak around -18 to -6 dB.
  At 0 dB, turn the gain down.
- **Hands off:** that channel's rms should be below about -50 dB. Plume's silence floor is
  -45 dB, so if hum sits above it, plume hears someone playing even when nobody is.

Write down both rms numbers. They are the first real data for these thresholds.

## 3. Play a take (4 minutes)

Use your channel number. In a second terminal, start the watch from step 4 first.

Give every try its own directory: `takes-di1/`, then `takes-di2/`, and change the watch to
match. A watch started on a used directory finds the old `ending.json` and exits straight
away. Keep the `takes` at the front, so git ignores the recording.

```bash
plume lead --tap-start "lofi midwest emo in F major" \
  --device "audio interface" --input-channel 1 --direct --record takes-di1/
```

| Do this | Look for |
|---|---|
| Mute the strings and chuck them 8 times, evenly | Drums land where the ninth would be. A knock on the body may not reach a pickup |
| Play over the drums | Bass and chords join about 8 bars in, melody 4 bars after |
| Keep playing a minute, then stop and mute the strings | About 20 seconds later (from about 100 bpm up; a few seconds more below that) the band has faded out and the take has ended |
| On another try, in a new directory: stop, then come back in during the fade | The band comes straight back |

If the count never catches, that is the first thing to report. The knock detector has only
heard synthetic chucks through a pickup.

## 4. Read the feed (during step 3)

```bash
plume watch takes-di1/ --every 4
```

- While you play: `"input": "direct"`, `you_trusted` lists all three fields, and `you_notes`
  matches what you are playing.
- After you stop: `you_notes` is `[]`, `you_db` falls to about your hands-off number, and
  `changes` says `you:stopped`, with the band still sounding.
- If `you_why` mentions loopback and you have none on, the check is wrong, so report it. Its
  threshold is a guess.

Afterwards:

```bash
cat takes-di1/ending.json                 # want: "faded out after hearing nothing from you"
open takes-di1/human.wav                  # want: only guitar, no band at all (xdg-open on Linux)
```

Any band in `human.wav` means the routing is wrong, and `--direct` is claiming something
false.

## 5. Voice, on its own channel (3 minutes)

Put a mic on a second input (phantom power on for a condenser), and sit close to it. Find
its channel with the meter.

```bash
plume voice "lofi midwest emo in F major" --base di2 \
  --device "audio interface" --input-channel 1 --voice-channel 2 --direct
```

Wait for the model to load. A `{"take": ...}` line means the band process has started. Count
in, play until the band is in, then say "plume, busier", wait a phrase, and say "plume, stop".
Keep a second terminal open with `plume stop`.

```bash
cat di2/takes1/voice.ndjson
grep cue_effects di2/takes1/decisions.ndjson | grep -v 'cue_effects": null'
```

| What `voice.ndjson` shows | What it means |
|---|---|
| `"skipped": "quiet"` while you spoke | The mic is off, on another channel or too low |
| `text` empty while you spoke | The model dropped it as not speech. Note the `rms_db` |
| `text` has the phrase with "plume" misheard | The wake word list needs that spelling. Report it exactly |
| `command` set, nothing changed | `{"cue_failed": ...}` on screen, or no `cue_effects`. That would be a bug |
| `missed_s` above 0 | The model is falling behind and audio went unheard |

Then repeat with the old setup, to see whether the band was what buried you:

```bash
plume voice "lofi midwest emo in F major" --base di3
```

## 6. The old take (1 minute, no sound)

If the take where nothing registered still exists:

```bash
plume voice-replay takes1/human.wav | grep -i -E 'plum|bloom|flume|stop|busier|sparser'
```

Empty output means the model never wrote anything close, and the band is the likely cause.

## 7. The Start button (5 minutes)

**Untested.** The page's Start and its setup have only ever run a fake process and a fake
meter on fake devices, so this is the first time they meet a real interface. Only one Mac and
one interface have ever been tried with plume at all. The first press of Start listens to your
input for a few seconds and plays nothing, but once setup is done the band comes in with no
further click. Turn the headphones down first.

```bash
plume gui
```

Set up again in Settings makes this a first run, if you have been here before.

| Do this | Look for |
|---|---|
| Read the first screen, without clicking | One Start. The small line under it names what it will use, and Settings sits next to Takes |
| Click Start with your hands on the strings, and play | "Play something." The next line says it heard you on your interface, on the channel you are plugged into. Note what level it asked you to change, if any, against the -18 to -6 dB in step 2 |
| With nothing plugged in or the gain down, click Set up again, then Start | "Nothing came in" names the device, and Play again, Start anyway and any other device are offered. Nothing else is open |
| Say Yes to headphones. With the Direct button off, say No to hearing yourself | "Check your headphone volume." Then "Still nothing" goes on with Hear me on |
| Do it once more through speakers, saying No to headphones | Hear me stays off and is not offered |
| Click Not now on any step | Back on Start, and nothing keeps running. `plume sessions` shows no take |
| Finish setup with headphones | The folder is new, under `~/Music/Plume`. "Play 8 even beats. Muted strings work well." and the small sign says Hearing you when you play. You hear yourself |
| Chuck muted strings 8 times | Drums land where the ninth would be, and the first view shows only a key line, the sign and Stop |
| Open Details and play | Bar, tempo and stage, and your notes and level follow you. Key and style show as a guess once the band is in. Note whether the guess is any good |
| Stop playing and wait | "Stop playing and the band will finish on its own." then the band says it is finishing, and the take ends by itself |
| Reload the page during a take | The same take comes back, and there is still only one band |
| Click Stop, or let it fade | The page lands on that take's Record. `human.wav` holds only guitar, with no band |
| Click Start again | Straight to the count, with no questions |
| With the page open, unplug the interface, then click Start | Plume asks you to play something again rather than starting on the wrong device. Note what it picked, because that take can make sound |

To try speakers in the room, open Settings and pick them under Output. This is the only part of plume that opens two devices, and it has not been tried on real gear.

| Do this | Look for |
|---|---|
| Pick the speakers, leave the tick off, click Start and play | The band comes out of the speakers. Your guitar comes only through your interface's own monitoring. `session.json` names the output under `input` |
| Tick Send my instrument to this output, with an instrument plugged in, and start again | Your guitar comes out of the speakers too, slightly late. Note how late it feels, against the delay in Details |
| Move to another channel in Settings | The tick shows off until you say so again |
| Unplug the speakers, then click Start | Settings says it could not find that output, and nothing plays anywhere else |
| Try a pair that may refuse, such as a display's speakers | Either it plays, or Settings says the two would not open together. Note which |
| Play ten minutes | Note whether the band and your guitar drift apart. Plume does nothing about this |
| Open the take and play Together | Note whether you sit where you played. `stream.json` holds what the two devices reported, and they may not report everything |

Note the delay in Details, and whether playing through Hear me feels late. That number is what
the device reports, which nobody has measured. If you can't stop it from the page, `plume stop`
in a terminal always works.

## What to send back

- Both rms numbers from step 2, and your interface's name.
- Whether the count caught, and on what you hit.
- `takes-di1/ending.json`, and whether the band came in and faded on time.
- Five lines of `plume watch` while playing and five after stopping.
- The `text` column of `voice.ndjson` from step 5, and from the room mic in `di3`.
- From step 7: what setup said about your level and whether it was right, the channel and
  device it heard you on, the delay Details showed, how it felt to play through, and the key
  and style plume guessed against what you played. Say how long it took from the first press
  to the band coming in.
- If you tried another output: which devices, whether they opened, any drift over a long take,
  and `stream.json` next to how far off Together sounded.

## What only you can answer

- Whether a loop behaves in a real room. Try a take with a fridge or amp hum running, and one
  with low rumble (traffic, a fan, a boiler), each left alone until the 64 second limit. Note
  whether a loop was made. Also try rumble with "1 bar" named: in tests it closed the take one bar after the cue.
  In tests hum and rumble can become a loop, and these thresholds
  are not yet tuned on a real room.
- Whether a real pickup's level fits the -45 dB floor and the 15 dB relative rule.
- Whether drums land on your count with no calibration. Plume uses the delay the system
  reports here, and on other setups that number has been wrong by over 200 ms.
- Whether Whisper hears a close mic in a real room.
- Whether a separate input and output device hold together. Two clocks can drift, which is
  why the commands above use one device for both. Output in Settings is how to find out.
