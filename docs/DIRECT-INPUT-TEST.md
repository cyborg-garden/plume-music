# Direct input test

About ten minutes, with a guitar, an interface and headphones. Everything here has only
run against a fake device in the tests, so this is the first time any of it meets a real
pickup. Plume makes sound during steps 3 and 5, and you are the one who starts it.

## 1. Set up (1 minute)

```bash
.venv/bin/pip install -e '.[play,voice]'
```

- Plug the guitar into the interface and set that input to instrument (INST), if it has
  the switch. Some small interfaces only take a guitar on one socket, so check the front.
- Plug headphones into the interface. Turn direct monitoring on so you can hear yourself.
- Set the interface to 48 kHz in Audio MIDI Setup.
- Leave plume's `--monitor` off, or you'll hear yourself twice.

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
| Keep playing a minute, then stop and mute the strings | About 20 seconds later (from about 75 bpm up) the band has faded out and the take has ended |
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
open takes-di1/human.wav                  # want: only guitar, no band at all
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

## What to send back

- Both rms numbers from step 2, and your interface's name.
- Whether the count caught, and on what you hit.
- `takes-di1/ending.json`, and whether the band came in and faded on time.
- Five lines of `plume watch` while playing and five after stopping.
- The `text` column of `voice.ndjson` from step 5, and from the room mic in `di3`.

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
  why the commands above use one device for both.
