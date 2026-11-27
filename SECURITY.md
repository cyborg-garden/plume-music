# Security

Plume runs on your own machine. It opens a local page, a local socket for cues, and
your audio and MIDI devices. It sends nothing to a server of ours, because there is
none.

## Report a problem

Report a security problem in private. On the repository page, open the **Security**
tab and choose **Report a vulnerability**. That opens a private thread with the
maintainers.

Do not open a public issue or pull request for a security problem.

Useful things to include:

- what you did, step by step
- what happened, and what you expected
- your operating system and plume version (`pip show plume-music`)

Leave out recordings, home paths and device serial numbers unless we ask.

## What to expect

We answer as soon as we can. A confirmed problem is fixed first and described
afterwards. The changelog names the fix once a release carries it.

## Supported versions

Fixes land on `main` and in the newest release. Older releases get no fixes.
