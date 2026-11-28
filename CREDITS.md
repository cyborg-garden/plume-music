# Credits

Plume is AGPL-3.0-or-later (see [LICENSE](LICENSE)). It ships a few files made by
other people. Each keeps its own licence.

## Samples

The recorded drum kit and the pitched instruments come from open sample libraries.
[src/plume/samples/README.md](src/plume/samples/README.md) lists each file, its
source and how it is prepared.

| Sound | Source | Licence |
|---|---|---|
| Drum kit | Virtuosity Drums, Versilian Studios | CC0-1.0 |
| Brush hits | Frankensnare, Karoryfer Samples | CC0-1.0 |
| Cajon, vibraphone, grand piano | Versilian Community Sample Library | CC0-1.0 |
| Finger bass | Sneakybass, Karoryfer Samples | CC0-1.0 |
| Upright bass | Double bass, D. Smolken | CC0-1.0 |
| Electric piano | Wurlitzer EP200, Greg Sullivan | CC-BY-3.0 |

The electric piano's licence asks for this credit:

> "Wurlitzer EP200 Electric Piano" samples by Greg Sullivan,
> https://creativecommons.org/licenses/by/3.0/, trimmed and normalised for plume.

If you publish audio that uses the `wurli` voice, put the line above in its
description.

CC0 text: https://creativecommons.org/publicdomain/zero/1.0/

## Note model

`src/plume/listen/models/basic_pitch_nmp.onnx` is the note model of Spotify's
basic-pitch 0.4.0, unmodified. Apache-2.0. The licence and notice sit beside it:
[LICENSE-basic-pitch](src/plume/listen/models/LICENSE-basic-pitch) and
[NOTICE-basic-pitch](src/plume/listen/models/NOTICE-basic-pitch).

> Basic Pitch. Copyright 2022 Spotify AB.

## Fonts

The local page uses two fonts, both under the SIL Open Font License 1.1. The
licence texts sit beside the font files in `src/plume/gui_static/fonts/`.

- DM Mono. Copyright 2020 The DM Mono Project Authors.
- Schibsted Grotesk. Copyright 2023 The Schibsted-Grotesk Project Authors.

## Libraries

Plume runs on NumPy, SciPy, soundfile and imageio-ffmpeg. The optional extras add
sounddevice, mido, python-rtmidi, onnxruntime, mlx-whisper and faster-whisper.
`pyproject.toml` is the list that counts.

## Ideas

[docs/design/RESEARCH.md](docs/design/RESEARCH.md) names the research and the
earlier music systems plume learns from.
