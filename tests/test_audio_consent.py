"""No process may play out of a human's speakers without consent.

Twice during development an automated agent started a live session and left it
running, still audible, stoppable only by finding the
PID.  Every path that opens the audio device is guarded here: an interactive
terminal, or an explicit ``PLUME_ALLOW_AUDIO=1``.  Nothing else.

These tests must never be relaxed to "make CI pass" — CI is exactly the
non-interactive case they exist to block.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from plume import realtime


ENV = realtime.AUDIO_CONSENT_ENV


def test_non_interactive_is_refused_by_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    allowed, why = realtime.audio_output_allowed()
    assert allowed is False
    assert why


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_explicit_opt_in_is_honoured(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    allowed, _ = realtime.audio_output_allowed()
    assert allowed is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_explicit_opt_out_beats_a_tty(monkeypatch, value):
    """An explicit no wins even from an interactive terminal."""
    monkeypatch.setenv(ENV, value)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    allowed, why = realtime.audio_output_allowed()
    assert allowed is False
    assert value in why


def test_check_audio_consent_names_the_offline_alternative(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(realtime.RealtimeError) as exc:
        realtime.check_audio_consent()
    msg = str(exc.value)
    assert ENV in msg
    assert "plume sim" in msg          # the user is told what to do instead


def test_engine_play_is_guarded(monkeypatch):
    """`plume play` opens the device too — it must go through the same gate."""
    from plume import engine

    monkeypatch.setenv(ENV, "0")
    with pytest.raises(realtime.RealtimeError):
        engine.play(object())          # guard must fire before anything else


def _run(args, env_extra=None):
    """Run the CLI in a genuinely non-interactive subprocess (pipes, no tty)."""
    import os

    env = dict(os.environ)
    env.pop(ENV, None)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "plume", *args],
        capture_output=True, text=True, timeout=120, env=env,
    )


@pytest.mark.parametrize("verb", ["jam", "lead"])
def test_live_verbs_refuse_from_a_pipe(verb):
    """The unattended-session regression, end to end: no tty, so no sound, and a clean exit.

    The invariant is **no audio comes out** and the exit is clean.  Which of
    the two refusals the user sees depends on the box: since v2.1 the device
    check runs first (a machine with no audio hardware must not be told to set
    a consent flag that cannot help it), so on a headless box the refusal is
    the device error instead.  Both are refusals; neither opens a stream.  The
    consent gate itself is asserted directly above and in
    tests/test_realtime.py::TestDeviceErrors — this test must not be the one
    that gets relaxed, and it is not being relaxed: it still requires a
    non-zero exit and a recognised refusal, never a session.
    """
    args = [verb] if verb == "jam" else [verb, "chill lofi at 96 bpm"]
    proc = _run(args)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert ("refusing to play live audio" in out
            or "no audio" in out
            or "sounddevice" in out), out
