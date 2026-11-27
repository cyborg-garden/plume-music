# Contributing

Plume is a live bandmate. Most rules here protect two things: nobody's speakers go
off by surprise, and nobody's private details land in a public repo.

## Set up

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[play,midi,notes]' pytest "pytest-rerunfailures==16.7"
```

Plain `python -m venv` and `pip` work the same way. In a Windows shell the
interpreter is `.venv\Scripts\python`.

## Tests never make sound

- A test never opens a real audio device and never plays sound.
- Live paths run on the fake device in `tests/conftest.py`.
- Do not set `PLUME_ALLOW_AUDIO` to run tests. The suite runs with it unset.
- A test never reads a recording that git ignores. Tests make every input they
  need, with `tests/fixtures_gen.py` or by rendering into `tmp_path`.
- No audio is tracked. If a test wants a WAV, write a generator.

Run the tests that cover what you touch:

```bash
.venv/bin/python -m pytest -q tests/test_folk.py
```

The whole suite takes hours on one core. CI runs it in eight shards on macOS, Linux
and Windows. To run one shard locally:

```bash
PLUME_TEST_SHARD=1/8 .venv/bin/python -m pytest -q
```

In PowerShell, set the variable first:

```powershell
$env:PLUME_TEST_SHARD = "1/8"
.venv\Scripts\python -m pytest -q
```

## One musical change per pull request

A change to how the band plays is its own pull request, with nothing else in it. A
person plays with it before it merges. Tests show the code does what it says. Only
playing shows whether it feels right.

Refactors, docs and tooling stay out of musical pull requests, and musical changes
stay out of theirs.

## Keep private things private

This repo is public and a check reads pull requests for private details. Keep
these out of code, tests, docs, commit messages and pull request text:

- real names, handles and contact details
- real account, server or channel ids
- home paths and machine names. Write `~/Music/Plume/<take>/` or `takes1/`.
- recordings and anything copied out of a real take. `session.json` holds real
  device names, so do not paste it.
- a particular product as the assumed setup. Write "a laptop mic" or "an audio
  interface". Name a product only where the name is the point, such as a device
  list plume reads.

`sanitize.config.json` says what the check allows.

## Security fixes

If you find a security problem, report it in private (see [SECURITY.md](SECURITY.md)).

When you fix one, the pull request and commit say what the change does, in neutral
words. They do not list what was open or how it could be used. That detail goes in
the private report.

## Commits and pull requests

- Subject: `type(scope): imperative summary`, lowercase, 72 characters at most.
  For example `fix(follow): keep knocks and crackle out of her level`.
- Body: what changes and why. Leave out how.
- Keep a pull request small enough to read in one sitting.

## Docs

- Present tense. Plain words. Short sentences.
- A doc says what is true now. History lives in [CHANGELOG.md](CHANGELOG.md) and
  in `docs/decisions/`.
- Say only what has been tried. If nobody has played it on real gear, say so.
- In `docs/PLAYING.md` and `docs/DIRECT-INPUT-TEST.md`, every `--record` and
  `plume watch` example names a folder that starts with `takes`. Git ignores
  those folders, and a test holds the docs to it.
- `docs/design/` holds the specs each version was built against. They are history,
  so leave them as they are.

## Credits

If you add a sample, a model or a font, add its source and licence to
[CREDITS.md](CREDITS.md) and to the README beside the file.
