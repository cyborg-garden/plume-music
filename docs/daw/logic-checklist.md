# Logic test sheet

One hour in Logic Pro. The answers decide what [logic.md](logic.md) says and how the
`Plume` template is built. Nothing here has been run yet.

Write each answer next to its question. "Yes", "no", or what you saw.

**Before you start**

- Logic Pro version: ______  macOS version: ______
- Plume has the `midi` extra: `pip install -e '.[midi]'`.
- Headphones on, or a direct input, so plume does not hear Logic's band as you.
- Leave the IAC Driver off for now. Part C turns it on.

The take command used all through:

```bash
plume lead --tap-start "lofi at 90 bpm" --midi-out virtual --midi-only --record
```

## A. Build the project (about 20 minutes)

1. New empty project. Eight software-instrument tracks.
2. Name and set them:

   | Track name | MIDI In Channel | Instrument |
   | --- | --- | --- |
   | `Plume Bass` | 1 | any stock bass |
   | `Plume Keys` | 2 | any stock electric piano |
   | `Plume Lead` | 3 | any stock lead synth |
   | `Plume Arp` | 4 | any stock pluck |
   | `Plume Shimmer` | 5 | any stock soft pad |
   | `Plume Sub` | 6 | any stock sine or 808 bass |
   | `Plume Other` | 7 | any stock piano |
   | `Plume Drums` | 10 | a stock General MIDI drum kit |

3. Record-enable all eight tracks. If the button is hidden: Track > Configure Track
   Header > Record Enable.

Answer:

- A1. Where is the channel setting, and what is it called? ______
- A2. Is there a MIDI In Port setting on the track? What does it list? ______
- A3. Which stock instruments did you pick? (These go in the template.) ______

## B. First take (about 10 minutes)

Start the take command. Play for a minute. Stop.

- B1. Does Logic see the `plume` port when it appears with Logic already open? ______
- B2. Does each track play only its own part? ______
- B3. If every track played everything: is there a setting under File > Project
  Settings > Recording that splits incoming MIDI by channel? Its exact name: ______
  Did turning it on fix it? ______
- B4. With only one track record-enabled, do the other seven still sound? ______
- B5. Drums: do kick, snare, hats, clap and crash land on the right drums? ______
- B6. Press record in Logic, then run a take. Does each track get its own region with
  only its own notes? ______
- B7. Does the band sound late against you? Roughly how much, and what latency does
  Logic Pro > Settings > Audio show? ______

## C. Does it survive? (about 15 minutes)

1. File > Save as Template, named `Plume`. Quit Logic.
2. Open Logic. New project from the `Plume` template.

- C1. Did MIDI In Channel persist on all eight tracks? ______
- C2. Did record-enable persist on all eight tracks? ______
- C3. Did the instruments and track names persist? ______

3. With plume not running, and the project open, start the take command.

- C4. Do the tracks play, though the `plume` port appeared only when the take
  started? ______
- C5. Stop the take and start another without touching Logic. Still playing? ______
- C6. If C4 or C5 is no: turn on the IAC Driver (Audio MIDI Setup > Window > Show
  MIDI Studio > IAC Driver > "Device is online"), swap `--midi-out virtual` for
  `--midi-out IAC`, and try C4 and C5 again. Does IAC fix it? ______
- C7. If A2 found a MIDI In Port setting: set one track to the `plume` port, stop the
  take, start a new one. Does the track still listen, or did the setting reset? ______

## D. One look at MMC (about 5 minutes)

For a later note only. Plume v1 sends no MMC.

- D1. Does File > Project Settings > Synchronization > MIDI still have "Listen to MIDI
  Machine Control (MMC) Input"? Exact wording: ______
- D2. Only if you have a tool that sends MMC play and stop to hand: with the setting
  on, does Logic obey them? Does it obey MMC record? ______ Skip this if not.

## E. Afterwards (about 10 minutes)

- E1. Drag the take's `band.mid` into a new Logic project at bar 1. One track per
  part? Does Logic offer to import the tempo? ______
- E2. GarageBand: drag the same `band.mid` into an empty project. One track per part,
  or one merged track? Which instruments did it pick? Does the tempo follow? ______
- E3. Keep the saved template. It goes into the repo after a check for paths and
  names from this Mac inside the package.

## What the answers change

| Answer | Result |
| --- | --- |
| C1 and C2 yes | The template works as opened. The recipe drops its per-session steps. |
| C2 no | The recipe keeps "record-enable all eight tracks" as a step for every session. |
| C4 and C5 yes | Plume's own port stays the default on Mac. |
| C4 or C5 no, C6 yes | The IAC Driver becomes the default for Logic, with one setup step. |
| B3 needed | The recipe names that setting. |
| D1, D2 | Recorded in [logic-later.md](logic-later.md). No change to v1. |
| E2 merged track | The GarageBand section says so and stays "listen back only". |
