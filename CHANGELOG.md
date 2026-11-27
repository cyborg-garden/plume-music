# Changelog

What changes between releases, newest first. Each entry is a plain sentence about
what a player or an agent sees.

## Unreleased

### Added

- **A local page.** `plume gui` opens a page with one Start. The first press is the
  setup: it finds the input channel, checks the level and asks about headphones.
  The page has a live view, Settings with an output choice, and a Guide.
- **Take pages.** A take opens as a page with a chart of what the band did, a
  playhead, a saved mix, Show files, and a video with the band laid in.
- **Live input.** Plume listens to a plugged-in instrument or a MIDI instrument as
  well as a room mic. `plume inputs` and `plume midi` list what is there.
- **A looper.** Play a phrase, ask for a loop, and plume keeps it going.
- **Hear me.** The player's own input in the mix, with a choice of sounds.
- **Styles.** `jazz`, `house`, `hiphop`, `rnb` and `folk` join `trap`, `lofi`,
  `midwest` and `synthwave`.
- **Voices.** Recorded bass, piano, electric piano, vibraphone, brush and cajon.
  Ten stereo synths and pads, with a palette for each style.
- **Style and key by ear.** `--detect` guesses the style from the playing and may
  change its mind once. The key is read from chroma and from notes, and plume
  waits when it is unsure. A plain click and a plain beat hold the start until
  both are known.
- **Say what you play.** The player can name the instrument.
- **The band follows.** In tap-start takes the band follows the player's level
  and busyness. It follows held or looped chords. The take page shows when it
  followed.
- **MIDI out.** The live band plays through a DAW as MIDI. A take keeps its band
  as `band.mid`.
- **A named output.** `plume lead` sends the sound to the output you name.
- **Linux and Windows.** The suite runs on macOS, Linux and Windows.

### Changed

- The page keeps two modes, day and night.
- Direct and MIDI takes end about 20 seconds after the player stops.
- The band sits higher under the player, and the harmony stays clear of the drums.
- Tone cues change a live take.

### Fixed

- The band enters on the player's phrase, in the right key, and swells in.
- An ordinary rest does not dip the band.
- A soft player is found within seconds.
- Knocks and crackle stay out of the player's level.
- Room chatter is not read as a cue.
- A fast song is not called jazz for being fast, and "house" in a prompt means
  the genre.
- The melody stays in key at the edges of its range.
- Each bar is read whole, even when the listener runs late.
- A refused or missing device leaves no listener and no empty take behind.
- A saved video stays private and does not get stuck.
- An overloaded input says so, and the take logs the stream's health.
- Hear me is capped and ducks on a buzzing input.

## 0.1.0 - 2026-11-03

First public release. Prompted mode, live listening, lead mode and role
negotiation, driven from the command line.
