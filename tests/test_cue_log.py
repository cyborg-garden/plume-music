"""The cue log: who asked for a cue, and when (plume GUI scope, section 6).

Nothing else in a take folder can say so.  ``decisions.ndjson`` records where a
cue applied, never when it arrived or who sent it, so The Record could not say
"You asked at bar 30.  It landed 13.6 seconds later."

Everything here runs on the fake device.  No sound, no microphone.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from conftest import fake_audio, patched_module, short_socket_path, wait_until
from plume import looper as looper_mod
from plume import realtime, record
from plume import roles as roles_mod
from test_bandentry import _take


def _lines(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln]


# ---------------------------------------------------------------- the recorder

def test_a_recording_has_a_cue_log_from_the_start_even_with_no_cues(tmp_path):
    """Present and empty means "no cues".  Absent means an older take, which is
    why the file is created at open and not on the first cue."""
    rec = realtime.SessionRecorder(tmp_path / "t").open({"kind": "lead"})
    rec.finish()
    assert (tmp_path / "t" / realtime.REC_CUES).read_text() == ""


def test_a_cue_line_is_flushed_when_it_is_written(tmp_path):
    rec = realtime.SessionRecorder(tmp_path / "t").open({"kind": "lead"})
    rec.cue({"text": "busier"})
    # read before finish(): a killed session must not lose it
    assert _lines(tmp_path / "t" / realtime.REC_CUES) == [{"text": "busier"}]
    rec.finish()


def test_a_cue_after_the_recorder_closed_is_ignored_not_raised(tmp_path):
    rec = realtime.SessionRecorder(tmp_path / "t").open({"kind": "lead"})
    rec.finish()
    rec.cue({"text": "too late"})
    assert _lines(tmp_path / "t" / realtime.REC_CUES) == []


def test_a_cue_before_the_recorder_opened_is_ignored_not_raised(tmp_path):
    realtime.SessionRecorder(tmp_path / "t").cue({"text": "too early"})
    assert not (tmp_path / "t").exists()


# ------------------------------------------------------------------- the socket

@pytest.fixture
def server():
    srv = realtime.CueServer(short_socket_path("cuelog"), kind="lead").start()
    yield srv
    srv.close()


def test_the_hook_hears_each_accepted_cue_and_its_source(server):
    heard = []
    server.on_received = lambda item, source: heard.append((item["text"], source))
    realtime.send_cue("busier", socket_path=server.path, source="gui")
    realtime.send_cue("sparser", socket_path=server.path, source="voice")
    realtime.send_cue("darker", socket_path=server.path, source="cli")
    realtime.send_cue("louder", socket_path=server.path)
    assert heard == [("busier", "gui"), ("sparser", "voice"), ("darker", "cli"),
                     ("louder", "unknown")]


@pytest.mark.parametrize("claimed", ["root", "", 7, None, ["gui"], "GUI"])
def test_a_source_outside_the_list_reads_as_unknown(server, claimed):
    heard = []
    server.on_received = lambda item, source: heard.append(source)
    assert server.handle({"text": "busier", "source": claimed})["ok"] is True
    assert heard == ["unknown"]


def test_a_source_does_not_change_the_reply_or_the_queue(server):
    plain = server.handle({"text": "busier"})
    tagged = server.handle({"text": "busier", "source": "gui"})
    plain.pop("queued"), tagged.pop("queued")
    assert plain == tagged
    assert server.drain() == [{"text": "busier", "at": "next-phrase", "ttl_bars": 16}] * 2


def test_a_refused_cue_is_not_logged(server):
    heard = []
    server.on_received = lambda item, source: heard.append(item)
    assert server.handle({"text": "   "})["ok"] is False
    assert server.handle({"text": "busier", "ttl_bars": "soon"})["ok"] is False
    assert heard == []


def test_a_failing_hook_cannot_lose_the_cue(server):
    def boom(item, source):
        raise OSError("disk full")
    server.on_received = boom
    assert server.handle({"text": "busier"})["ok"] is True
    assert [c["text"] for c in server.drain()] == ["busier"]


# ------------------------------------------------------------------- the looper

@pytest.fixture
def looped(server):
    server.intercept = looper_mod.Looper(realtime.SR).handle_cue
    heard = []
    server.on_received = lambda item, source: heard.append((item, source))
    return server, heard


def test_a_cue_the_looper_answers_still_reaches_the_log(looped):
    """It answers at once and never queues, so no row can ever name this cue."""
    server, heard = looped
    reply = realtime.send_cue("make a 4 bar loop called bass", socket_path=server.path,
                              source="gui")
    assert reply["ok"] is True and reply["queued"] == 0
    assert server.drain() == []
    assert heard == [({"text": "make a 4 bar loop called bass", "at": "now", "ttl_bars": 0,
                       "loop": {"op": "record", "ok": True}}, "gui")]


def test_a_cue_the_looper_refuses_is_logged_as_refused(looped):
    """The sender saw the refusal and the take would have no record of it."""
    server, heard = looped
    assert server.handle({"text": "stop the loop", "source": "voice"})["ok"] is False
    assert heard[0][0]["loop"] == {"op": "stop", "ok": False}
    assert heard[0][1] == "voice"


def test_a_cue_for_the_band_is_logged_once_with_no_looper_answer(looped):
    server, heard = looped
    assert server.handle({"text": "busier", "source": "cli"})["queued"] == 1
    assert heard == [({"text": "busier", "at": "next-phrase", "ttl_bars": 16}, "cli")]


def test_a_failing_hook_cannot_lose_the_loopers_answer(server):
    def boom(item, source):
        raise OSError("disk full")
    server.intercept = looper_mod.Looper(realtime.SR).handle_cue
    server.on_received = boom
    assert server.handle({"text": "make a loop"})["ok"] is True


def test_the_looper_answer_does_not_depend_on_the_log(looped):
    """Byte for byte what a sender got before there was a log to tell."""
    server, _ = looped
    with_log = server.handle({"text": "list loops"})
    server.on_received = None
    assert server.handle({"text": "list loops"}) == with_log


# ------------------------------------------------------------------ the callers

def test_plume_cue_says_it_came_from_the_command_line(server, capsys):
    from plume import cli
    heard = []
    server.on_received = lambda item, source: heard.append(source)
    assert cli.main(["cue", "busier", "--socket", server.path]) == 0
    assert cli.main(["cue", "sparser", "--socket", server.path,
                     "--source", "voice"]) == 0
    assert heard == ["cli", "voice"]


def test_the_voice_loop_marks_its_cues_as_voice():
    source = Path(realtime.__file__).with_name("voice.py").read_text()
    assert '"cue", _CUES[name], "--source", "voice"' in source


# -------------------------------------------------------- a take on the fake device

def _lead_with_cues(tmp_path, monkeypatch, *, record: bool):
    """A tap-start take, with one cue sent once the drums are planned."""
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
    fake = fake_audio(_take(seconds=90.0, room_s=120.0), rate_x=12.0)
    sock = short_socket_path("cuelive")
    rec_dir = tmp_path / "rec"
    sent = {}

    def cue_once_bars_are_planned():
        def in_bars():
            try:
                return realtime.send_status(socket_path=sock).get("bar") is not None
            except realtime.RealtimeError:
                return False
        if wait_until(in_bars, timeout=30.0):
            time.sleep(0.1)
            sent["reply"] = realtime.send_cue("busier", socket_path=sock, source="gui")

    t = threading.Thread(target=cue_once_bars_are_planned, daemon=True)
    with patched_module("sounddevice", fake):
        t.start()
        realtime.lead(prompt="lofi", seed=0, tap_start=True,
                      role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(rec_dir) if record else None,
                      socket_path=sock, status=False, max_minutes=0.5)
    t.join(timeout=5.0)
    return rec_dir, sent


def test_a_live_cue_is_logged_on_the_same_clock_as_the_rows(tmp_path, monkeypatch,
                                                            plume_registry):
    rec_dir, sent = _lead_with_cues(tmp_path, monkeypatch, record=True)
    assert sent["reply"]["ok"] is True
    logged = _lines(rec_dir / realtime.REC_CUES)
    assert len(logged) == 1
    cue = logged[0]
    assert cue["text"] == "busier" and cue["source"] == "gui"
    assert cue["at"] == "next-phrase" and cue["ttl_bars"] == 16
    assert isinstance(cue["bar"], int) and isinstance(cue["received_at"], float)

    rows = [r for r in _lines(rec_dir / realtime.REC_DECISIONS)
            if isinstance(r.get("bar"), int)]
    landed = [r for r in rows if r["form"]["applied_cue"]]
    assert landed, "the cue never reached a phrase line in this take"
    # It arrived while the bar it names was planned, and landed afterwards.
    assert landed[0]["bar"] > cue["bar"]
    wait = landed[0]["t_s"] - cue["t_s"]
    bar_s = 240.0 / rows[0]["bpm"]
    # next-phrase is at most four bars off, plus the bar planned ahead of the
    # playhead.  A different clock would land nowhere near this window.
    assert 0.0 < wait <= 6.0 * bar_s
    # Rows are the near future: this cue's own bar was planned before it sounded.
    row = next(r for r in rows if r["bar"] == cue["bar"])
    assert row["t_s"] - cue["t_s"] < 2.0 * bar_s


def test_without_record_no_cue_log_is_written_and_the_cue_still_works(
        tmp_path, monkeypatch, plume_registry):
    rec_dir, sent = _lead_with_cues(tmp_path, monkeypatch, record=False)
    assert sent["reply"]["ok"] is True
    assert not rec_dir.exists()
    assert not list(tmp_path.rglob(realtime.REC_CUES))


# ------------------------------------------------- the other doors into a take

def test_loop_and_band_cues_share_one_log_on_a_real_take(tmp_path, monkeypatch, plume_registry):
    """Through the real ``plume cue``, the real socket and the real looper."""
    from plume import cli

    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
    fake = fake_audio(_take(seconds=90.0, room_s=120.0), rate_x=12.0)
    sock = short_socket_path("cueloop")
    rec_dir = tmp_path / "rec"
    codes = {}

    def status():
        try:
            return realtime.send_status(socket_path=sock)
        except realtime.RealtimeError:
            return None

    def cues():
        if wait_until(status, timeout=30.0):
            codes["early"] = cli.main(["cue", "list loops", "--socket", sock])
        if wait_until(lambda: (status() or {}).get("bar") is not None, timeout=30.0):
            codes["refused"] = cli.main(["cue", "cancel the loop", "--socket", sock,
                                         "--source", "gui"])
            codes["band"] = cli.main(["cue", "busier", "--socket", sock, "--source", "voice"])

    t = threading.Thread(target=cues, daemon=True)
    with patched_module("sounddevice", fake):
        t.start()
        realtime.lead(prompt="lofi", seed=0, tap_start=True,
                      role=roles_mod.RoleParams(role="lead"), record_dir=str(rec_dir),
                      socket_path=sock, status=False, max_minutes=0.5)
    t.join(timeout=5.0)

    # The CLI turns a refused cue into a non-zero exit; the point is that it was logged.
    assert codes["early"] == 0 and codes["band"] == 0 and codes["refused"] != 0
    logged = _lines(rec_dir / realtime.REC_CUES)
    assert [(c["text"], c["source"], c.get("loop")) for c in logged] == [
        ("list loops", "cli", {"op": "list", "ok": True}),
        ("cancel the loop", "gui", {"op": "cancel", "ok": False}),
        ("busier", "voice", None)]
    assert logged[0]["bar"] is None and isinstance(logged[1]["bar"], int)

    said = [e["sentence"] for e in record.read_take(rec_dir)["events"]]
    assert any("Another terminal asked for the list of loops before the first bar" in s for s in said)
    assert any("You asked to cancel a loop take" in s and "said no" in s for s in said)
    # A counted take says it was counted.
    rows = _lines(rec_dir / realtime.REC_DECISIONS)
    assert not any("tempo_from" in r for r in rows)


def test_a_cue_during_a_midi_take_reaches_the_log(tmp_path, monkeypatch, plume_registry):
    """MIDI notes send no cues of their own, so the door to prove is a take listening
    to a keyboard.  It must log like any other, and the page must know the route."""
    from conftest import FakeMido
    from plume.compose import PromptSpec, compose

    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
    sock = short_socket_path("cuemidi")
    rec_dir = tmp_path / "rec"
    fake, mido = fake_audio(None, rate_x=8.0), FakeMido()
    sent = {}

    def cue_then_stop():
        def in_bars():
            try:
                return realtime.send_status(socket_path=sock).get("bar") is not None
            except realtime.RealtimeError:
                return False
        if wait_until(in_bars, timeout=30.0):
            sent["band"] = realtime.send_cue("busier", socket_path=sock, source="cli")
            sent["loop"] = realtime.send_cue("make a loop", socket_path=sock, source="gui")
            realtime.send_stop(socket_path=sock)

    session = compose(PromptSpec(styles=["lofi"], bpm=120.0, key="C major", movements=1,
                                 mood=[], name="cue-midi"), seed=7)
    t = threading.Thread(target=cue_then_stop, daemon=True)
    with patched_module("sounddevice", fake), patched_module("mido", mido):
        t.start()
        realtime.lead(session=session, seed=0, role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(rec_dir), socket_path=sock, status=False,
                      max_minutes=0.5, midi="auto")
    t.join(timeout=5.0)

    assert sent["band"]["ok"] and sent["loop"]["ok"]
    logged = _lines(rec_dir / realtime.REC_CUES)
    assert [(c["text"], c["source"]) for c in logged] == [("busier", "cli"), ("make a loop", "gui")]
    assert logged[1]["loop"] == {"op": "record", "ok": True}
    assert record.read_take(rec_dir)["route"] == "midi"


def test_a_spoken_cue_reaches_the_log_as_voice(tmp_path, monkeypatch, plume_registry):
    """voice.run, the real ``plume cue`` in a subprocess, and the real lead."""
    from test_voice import ROUTE, _voice_session

    rec, lines, _ = _voice_session(tmp_path, monkeypatch, said="busier", said_at=55.0, extra=ROUTE)
    assert '{"heard": "busier"}' in lines
    logged = _lines(rec / realtime.REC_CUES)
    assert [(c["text"], c["source"]) for c in logged if c["text"] == "busier"] == [("busier", "voice")]
    assert isinstance(logged[0]["t_s"], float)
    assert record.read_take(rec)["route"] == "direct"
