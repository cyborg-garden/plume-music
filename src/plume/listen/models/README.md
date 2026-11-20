# Models

`basic_pitch_nmp.onnx` is the note model of Spotify's basic-pitch 0.4.0
(`basic_pitch/saved_models/icassp_2022/nmp.onnx` in the PyPI wheel), unmodified.
Apache-2.0: see `LICENSE-basic-pitch` and `NOTICE-basic-pitch`. The notice keeps the
lines that pertain to the model file; the upstream notice's other lines credit code
libraries that are not redistributed here.

SHA-256: `2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec`

It is read by `plume/listen/notes.py` when the optional `notes` extra is installed.
