# Plume in Logic Pro and GarageBand

Plume sends its band out as MIDI, one channel per part. Logic Pro puts its own
instrument on each part and can record the notes.

**Status: untested in Logic.** No step on this page has been run in Logic Pro or
GarageBand. Each claim about them carries a mark:

- `[untested]` nobody has tried it in the app.
- `[tested: <app version>]` someone ran it and it worked.

Claims about plume itself (flags, channels, `band.mid`) are covered by plume's tests
against a simulated port. They carry no mark.

The test sheet is [logic-checklist.md](logic-checklist.md). What waits until after v1
is in [logic-later.md](logic-later.md).

## What v1 does and does not do

- Plume plays the band. Logic makes the sound, on eight separate tracks.
- **Plume has no record control and no arm control in Logic.** You press record, you
  press stop, and you arm the tracks, by hand.
- **This cannot match the Ableton setup in v1.** With Ableton Live, plume can start
  and stop recording and arm the tracks for you. Logic has no documented way for
  another program to do that short of acting as a control surface, and plume does not
  do that in v1.
- Logic's tempo does not follow plume. Plume sends no clock and no transport messages.
- GarageBand does not take the band live. It takes the band's MIDI file after the take.

## What you need

- Logic Pro on a Mac. `[untested]` Which versions work is not known.
- Plume with the `midi` extra: `pip install -e '.[midi]'`.
- A way for you to play that is not a room mic, if Logic is the only band you hear:
  a direct input, a MIDI keyboard, or headphones. On a room mic plume hears Logic's
  band as your playing.

## The tracks

Eight software-instrument tracks. Names and channels are the same as the Ableton
template.

| Track name | MIDI channel | Part | A stock instrument to start with `[untested]` |
| --- | --- | --- | --- |
| `Plume Bass` | 1 | bass | a bass patch |
| `Plume Keys` | 2 | pad and keys | an electric piano |
| `Plume Lead` | 3 | lead | a lead synth |
| `Plume Arp` | 4 | arp | a plucked synth |
| `Plume Shimmer` | 5 | shimmer | a soft pad or bell |
| `Plume Sub` | 6 | sub (a tuned 808) | a sine or 808 bass |
| `Plume Other` | 7 | any other part with notes | a piano |
| `Plume Drums` | 10 | drums | a General MIDI drum kit |

Drums arrive on General MIDI notes: kick 36, rim 37, snare 38, clap 39, closed hat 42,
pedal hat 44, open hat 46, crash 49. `[untested]` Logic's stock acoustic and
electronic drum kits put those drums on those notes.

## The port

Plume can send to one of two kinds of port.

- **Plume's own port.** `--midi-out virtual` makes a port called `plume`. It exists
  only while a take runs.
- **The IAC Driver.** A port macOS keeps open all the time. Turn it on once: open
  Audio MIDI Setup, choose Window > Show MIDI Studio, open IAC Driver and tick
  "Device is online". Then use `--midi-out IAC`.

`[untested]` Which one Logic needs is the main open question. If a track set to the
`plume` port loses that setting when the port goes away between takes, use the IAC
Driver. If Logic's tracks listen to every port, either works. With the IAC Driver,
keep Logic's own MIDI output off that port, or the notes loop.

## Set up the project once

Every step here is `[untested]`.

1. Make a new empty project with eight software-instrument tracks.
2. Name each track and give it an instrument, from the table above.
3. Open each track's inspector. Set **MIDI In Channel** to the track's channel. If
   the inspector has **MIDI In Port**, leave it on All, or pick the port you use.
4. Show the record-enable button if it is hidden: Track > Configure Track Header >
   Record Enable.
5. Record-enable all eight tracks. Logic sends incoming notes only to the selected
   track unless tracks are record-enabled. With all eight enabled, each track plays
   only its own channel.
6. If every track plays every part, look in File > Project Settings > Recording for
   a setting that splits incoming MIDI by channel across record-enabled tracks, and
   turn it on.
7. Save it: File > Save as Template. Name it `Plume`.

`[untested]` It is not known whether the template keeps the MIDI In Channel and
record-enable settings. If it does not keep record-enable, step 5 is a step for every
session.

## Play a take

1. Open the `Plume` template in Logic. Check that all eight tracks are record-enabled.
2. Set Logic's tempo to the tempo you ask plume for. This only makes the bar lines
   match; the notes are right either way.
3. Press record in Logic.
4. Start plume:

   ```bash
   plume lead --tap-start "lofi at 90 bpm" --midi-out virtual --midi-only --record
   ```

   `--midi-only` keeps plume's own band silent, so you hear Logic's. Leave it off to
   hear both. Use `--midi-out IAC` if you chose the IAC Driver.
5. Play. When you finish, stop plume, then press stop in Logic.

`[untested]` If Logic does not see the `plume` port when it appears after Logic is
open, start plume first, then open the project.

When a take ends, plume lets go of every held note and sends All Notes Off on each
channel it used.

## Timing

Plume sends each note when its own sound of that note would reach your speakers.
Logic then adds its own output delay, so with `--midi-only` the band can sound a
little late. Send the notes earlier with a negative offset, for example
`--midi-out-offset -12` for 12 ms. `[untested]` Logic shows its delay under
Logic Pro > Settings > Audio, as the resulting latency.

## After the take: band.mid

A take with `--midi-out` and `--record` writes `band.mid` in the take's folder. It
holds every note the band sent, one track per part, on the channels above, with the
take's tempo. Its time 0 is the start of the take's audio files.

The tracks inside `band.mid` are named for the parts (`bass`, `pad and keys`,
`drums`), not `Plume Bass`.

`[untested]` In Logic: File > Import > MIDI File, or drag `band.mid` into the tracks
area at bar 1. Logic may ask whether to import the file's tempo. Say yes to keep the
bar lines close to the take.

## GarageBand

**Unverified until checked in GarageBand.**

GarageBand has no setting for which MIDI channel a track listens to `[untested]`, so
it cannot keep the parts apart while you play. Use it after the take:

1. Play the take with plume's own band, and with `--midi-out virtual --record` so
   plume writes `band.mid`.
2. Open GarageBand with an empty project.
3. After the take, drag `band.mid` in. Drop it in the empty tracks area at the start
   of the project.
4. `[untested]` GarageBand makes one track per part and picks an instrument for
   each. Change any instrument you like.
5. Drag `human.wav` from the same folder onto a new audio track at the start, to
   hear yourself with the band.

`[untested]` Whether GarageBand splits `band.mid` into separate tracks, and whether it
keeps the file's tempo, are not known.

## What is not sent

Fades and the bus volume (only how hard each note is played), tone cues such as
darker or wetter, the vinyl crackle, a tempo clock, and transport messages.
