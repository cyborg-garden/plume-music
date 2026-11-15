"""A take's files: the saved mix, and showing the folder.

Nothing here opens an audio device, plays a sound or opens a file manager window.
The takes are a few thousand synthetic samples, and the launcher is a list that
remembers what it was asked for.  The real launcher refuses to run under pytest,
and that refusal is tested too.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from conftest import needs_symlinks
from plume import gui, oscompat, realtime
from test_gui_bridge import Client

RATE = 48000
BAND = 10 ** (3 / 20)


def make_take(folder, *, human=None, machine=None, stream=None, source="direct"):
    folder.mkdir(parents=True)
    if human is not None:
        sf.write(folder / "human.wav", human, RATE, subtype="PCM_24")
    if machine is not None:
        sf.write(folder / "machine.wav", machine, RATE, subtype="PCM_24")
    (folder / "session.json").write_text(json.dumps({"seed": 1, "input": {"source": source}}))
    (folder / "decisions.ndjson").write_text("")
    if stream is not None:
        (folder / "stream.json").write_text(json.dumps(stream))
    return folder


def clicks(human_at=580, machine_at=100, n=4800, human_level=0.5, machine_level=0.25):
    human = np.zeros(n)
    human[human_at] = human_level
    machine = np.zeros(n)
    machine[machine_at] = machine_level
    return human, machine


def read_mix(folder):
    pcm, rate = sf.read(folder / "mix.wav", dtype="int16")
    assert rate == RATE and pcm.ndim == 1
    return pcm


@pytest.fixture
def root(tmp_path):
    return tmp_path / "library"


@pytest.fixture
def shown():
    return []


@pytest.fixture
def server(root, shown, tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))
    root.mkdir()
    srv = gui.GuiServer(roots=[root], log=lambda line: None,
                        revealer=lambda folder, select: shown.append((folder, select))).start()
    yield srv
    srv.close()


@pytest.fixture
def client(server):
    return Client(server).login()


def take_id(client, folder):
    _, out = client.get("/api/takes")
    return next(t["id"] for t in json.loads(out)["takes"] if t["folder"] == folder)


# ------------------------------------------------------------------ the saved mix

def test_the_mix_lines_the_player_up_with_the_band(root, server, client):
    human, machine = clicks()                       # she played at 100, and it came back 10 ms late
    folder = make_take(root / "t", human=human, machine=machine,
                       stream={"calibrated_ms": 10.0, "reported_ms": 99.0})
    assert server.library.write_mix(take_id(client, "t")) == "mix.wav"
    pcm = read_mix(folder)
    assert len(pcm) == 4800
    assert int(np.argmax(abs(pcm))) == 100
    assert pcm[100] / 32767 == pytest.approx((0.5 + 0.25 * BAND) * 0.7, abs=1e-3)
    assert np.count_nonzero(pcm) == 1


def test_the_reported_delay_is_used_only_when_none_was_measured(root, server, client):
    human, machine = clicks(human_at=100 + 960)
    folder = make_take(root / "t", human=human, machine=machine,
                       stream={"calibrated_ms": None, "reported_ms": 20.0})
    server.library.write_mix(take_id(client, "t"))
    assert np.flatnonzero(read_mix(folder)).tolist() == [100]


def test_the_mix_is_as_long_as_whichever_track_ends_last(root, server, client):
    human = np.zeros(9600)
    human[9000] = 0.5
    machine = np.zeros(4800)
    machine[100] = 0.25
    folder = make_take(root / "t", human=human, machine=machine, stream={"calibrated_ms": 10.0})
    server.library.write_mix(take_id(client, "t"))
    pcm = read_mix(folder)
    assert len(pcm) == 9600 - 480
    assert np.flatnonzero(pcm).tolist() == [100, 9000 - 480]


def test_the_mix_is_what_the_page_plays(root, server, client):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine, stream={"calibrated_ms": 10.0})
    tid = take_id(client, "t")
    server.library.write_mix(tid)
    resp, played = client.get(f"/api/take/{tid}/audio/together")
    assert resp.status == 200
    saved = (folder / "mix.wav").read_bytes()
    # The same samples.  The file then says what it was made from, after them.
    assert saved[44:44 + 4800 * 2] == played[44:] and len(played) == 44 + 4800 * 2
    assert saved[:4] == b"RIFF" and int.from_bytes(saved[4:8], "little") == len(saved) - 8


def test_two_full_scale_peaks_at_once_do_not_clip(root, server, client):
    rng = np.random.default_rng(7)
    loud = np.sign(rng.standard_normal(4800)) * 0.999
    folder = make_take(root / "t", human=loud, machine=loud, stream={"calibrated_ms": 0.0})
    server.library.write_mix(take_id(client, "t"))
    pcm = read_mix(folder)
    assert 0.9 * 32767 < abs(pcm).max() < 32767


def test_with_no_stream_file_the_tracks_are_mixed_as_they_lie(root, server, client):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine)
    assert server.library.write_mix(take_id(client, "t")) == "mix.wav"
    pcm = read_mix(folder)
    assert len(pcm) == 4800
    assert np.flatnonzero(pcm).tolist() == [100, 580]


@pytest.mark.parametrize("missing", ["machine", "human"])
def test_a_take_with_one_track_gets_no_mix(root, server, client, missing):
    human, machine = clicks()
    tracks = {"human": human, "machine": machine}
    tracks[missing] = None
    folder = make_take(root / "t", **tracks, stream={"calibrated_ms": 10.0})
    before = sorted(os.listdir(folder))
    assert server.library.write_mix(take_id(client, "t")) is None
    assert sorted(os.listdir(folder)) == before


def test_a_room_mic_take_gets_a_mix_too(root, server, client):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine,
                       stream={"calibrated_ms": 10.0}, source="room")
    assert server.library.write_mix(take_id(client, "t")) == "mix.wav"
    assert np.flatnonzero(read_mix(folder)).tolist() == [100]


def test_tracks_at_two_rates_get_no_mix(root, server, client):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine)
    sf.write(folder / "machine.wav", machine, 44100, subtype="PCM_24")
    assert server.library.write_mix(take_id(client, "t")) is None
    assert not (folder / "mix.wav").exists()


def test_a_take_still_being_recorded_is_left_alone(root, server, client, monkeypatch):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine, stream={"calibrated_ms": 10.0})
    tid = take_id(client, "t")
    before = sorted(os.listdir(folder))
    monkeypatch.setattr(gui, "_live_folders", lambda: {str(folder.resolve())})
    assert server.library.write_mix(tid) is None
    assert sorted(os.listdir(folder)) == before


def test_a_mix_that_is_up_to_date_is_not_written_again(root, server, client):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine, stream={"calibrated_ms": 10.0})
    tid = take_id(client, "t")
    server.library.write_mix(tid)
    first = os.stat(folder / "mix.wav")
    assert server.library.write_mix(tid) == "mix.wav"
    again = os.stat(folder / "mix.wav")
    assert (again.st_ino, again.st_mtime_ns) == (first.st_ino, first.st_mtime_ns)


def test_a_newer_recording_or_delay_makes_a_new_mix(root, server, client):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine, stream={"calibrated_ms": 10.0})
    tid = take_id(client, "t")
    server.library.write_mix(tid)
    later = os.stat(folder / "mix.wav").st_mtime_ns + 5_000_000_000
    (folder / "stream.json").write_text(json.dumps({"calibrated_ms": 0.0}))
    os.utime(folder / "stream.json", ns=(later, later))
    server.library.write_mix(tid)
    assert np.flatnonzero(read_mix(folder)).tolist() == [100, 580]


def test_the_mix_arrives_whole_under_its_own_name_or_not_at_all(root, server, client, monkeypatch):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine, stream={"calibrated_ms": 10.0})
    tid = take_id(client, "t")
    seen = []
    real = gui._mixed_pcm

    def watched(*args):
        seen.append(sorted(os.listdir(folder)))
        return real(*args)

    monkeypatch.setattr(gui, "_mixed_pcm", watched)
    server.library.write_mix(tid)
    # While it was being written it had another name, and afterwards that name is gone.
    assert seen and all("mix.wav" not in names for names in seen)
    assert [n for n in os.listdir(folder) if n.endswith(".tmp")] == []
    whole = (folder / "mix.wav").read_bytes()

    def broken(*args):
        raise OSError("disk full")

    monkeypatch.setattr(gui, "_mixed_pcm", broken)
    later = os.stat(folder / "mix.wav").st_mtime_ns + 5_000_000_000
    os.utime(folder / "human.wav", ns=(later, later))
    with pytest.raises(OSError):
        server.library.write_mix(tid)
    assert (folder / "mix.wav").read_bytes() == whole
    assert [n for n in os.listdir(folder) if n.endswith(".tmp")] == []


@needs_symlinks
def test_a_planted_mix_symlink_is_replaced_and_its_target_untouched(root, server, client, tmp_path):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine, stream={"calibrated_ms": 10.0})
    target = tmp_path / "elsewhere.txt"
    target.write_text("mine")
    (folder / "mix.wav").symlink_to(target)
    server.library.write_mix(take_id(client, "t"))
    assert target.read_text() == "mine"
    assert not (folder / "mix.wav").is_symlink() and len(read_mix(folder)) == 4800


def test_the_recordings_are_not_changed_by_a_mix(root, server, client):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine, stream={"calibrated_ms": 10.0})
    before = {n: (folder / n).read_bytes() for n in os.listdir(folder)}
    server.library.write_mix(take_id(client, "t"))
    assert {n: (folder / n).read_bytes() for n in before} == before
    assert sorted(set(os.listdir(folder)) - set(before)) == ["mix.wav"]


# ------------------------------------------------------------------ showing the folder

def test_show_files_writes_the_mix_and_shows_the_takes_own_folder(root, shown, client):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine, stream={"calibrated_ms": 10.0})
    resp, out = client.post("/api/reveal", {"take": take_id(client, "t")})
    assert resp.status == 200 and json.loads(out) == {"shown": True, "mix": True, "band_twice": False}
    assert shown == [(folder.resolve(), "mix.wav")]
    assert len(read_mix(folder)) == 4800


def test_show_files_still_shows_a_take_that_can_have_no_mix(root, shown, client):
    human, _ = clicks()
    folder = make_take(root / "t", human=human)
    resp, out = client.post("/api/reveal", {"take": take_id(client, "t")})
    assert resp.status == 200 and json.loads(out) == {"shown": True, "mix": False, "band_twice": False}
    assert shown == [(folder.resolve(), None)]


def test_a_mix_that_cannot_be_written_does_not_stop_the_folder_showing(root, shown, client, monkeypatch):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine)

    def broken(*args):
        raise OSError("read-only")

    monkeypatch.setattr(gui, "_mixed_pcm", broken)
    resp, out = client.post("/api/reveal", {"take": take_id(client, "t")})
    assert resp.status == 200 and json.loads(out) == {"shown": True, "mix": False, "band_twice": False}
    assert shown == [(folder.resolve(), None)]


def test_a_request_cannot_name_a_folder(root, shown, client, tmp_path):
    human, machine = clicks()
    make_take(root / "t", human=human, machine=machine)
    outside = make_take(tmp_path / "outside" / "private", human=human, machine=machine)
    tid = take_id(client, "t")
    outside_id = gui.Library._id_of(outside.resolve())
    crafted = [
        ({"take": tid, "path": str(outside)}, 400),
        ({"take": tid, "folder": str(outside)}, 400),
        ({"take": tid, "select": "../../outside/private/human.wav"}, 400),
        ({"take": str(outside)}, 404),
        ({"take": "../../outside/private"}, 404),
        ({"take": f"{tid}/../../outside/private"}, 404),
        ({"take": outside_id}, 404),
        ({"take": "0" * 16}, 404),
        ({"take": [tid]}, 400),
        ({"path": str(outside)}, 400),
        ({}, 400),
    ]
    for body, status in crafted:
        resp, out = client.post("/api/reveal", body)
        assert resp.status == status, body
        assert str(outside).encode() not in out
    assert shown == []
    assert not (outside / "mix.wav").exists()


@needs_symlinks
def test_a_take_folder_swapped_for_a_symlink_is_not_shown(root, shown, client, tmp_path):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine)
    outside = make_take(tmp_path / "outside" / "private", human=human, machine=machine)
    tid = take_id(client, "t")
    folder.rename(root / "moved")
    folder.symlink_to(outside)
    assert client.post("/api/reveal", {"take": tid})[0].status == 404
    assert shown == [] and not (outside / "mix.wav").exists()


def test_show_files_is_guarded_like_every_other_post(root, shown, server, client):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine)
    body = {"take": take_id(client, "t")}
    assert client.post("/api/reveal", body, Origin=None)[0].status == 403
    assert client.post("/api/reveal", body, Origin="http://evil.example")[0].status == 403
    assert client.post("/api/reveal", body, **{"X-Plume-Token": "0" * 64})[0].status == 403
    assert client.post("/api/reveal", body, **{"Content-Type": "text/plain"})[0].status == 415
    assert client.post("/api/reveal", body, **{"Sec-Fetch-Site": "cross-site"})[0].status == 403
    assert client.post("/api/reveal", body, host="evil.example")[0].status == 403
    stranger = Client(server)
    stranger.csrf = client.csrf
    assert stranger.post("/api/reveal", body)[0].status == 401
    assert client.get("/api/reveal")[0].status == 404
    assert shown == [] and not (folder / "mix.wav").exists()


def test_a_file_manager_that_will_not_start_is_said_plainly(root, tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))
    human, machine = clicks()
    make_take(root / "t", human=human, machine=machine)

    def none_here(folder, select):
        raise FileNotFoundError("no such launcher")

    srv = gui.GuiServer(roots=[root], log=lambda line: None, revealer=none_here).start()
    try:
        client = Client(srv).login()
        resp, out = client.post("/api/reveal", {"take": take_id(client, "t")})
    finally:
        srv.close()
    assert resp.status == 500 and json.loads(out) == {"error": "could not open the folder"}


def test_the_real_launcher_does_not_run_under_pytest(root, tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))
    human, machine = clicks()
    make_take(root / "t", human=human, machine=machine)
    started = []
    monkeypatch.setattr(oscompat, "_launch", started.append)
    srv = gui.GuiServer(roots=[root], log=lambda line: None).start()
    try:
        client = Client(srv).login()
        resp, _ = client.post("/api/reveal", {"take": take_id(client, "t")})
    finally:
        srv.close()
    assert resp.status == 500 and started == []


# ------------------------------------------------------------------ what a mix is made from

def _made(folder):
    st = os.stat(folder / "mix.wav")
    return st.st_ino, st.st_mtime_ns


def _mixed_take(root, server, client, **kw):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine, stream={"calibrated_ms": 10.0}, **kw)
    tid = take_id(client, "t")
    assert server.library.write_mix(tid) == "mix.wav"
    return folder, tid


@pytest.mark.parametrize("track, at", [("human", 1580), ("machine", 1100)])
def test_a_recording_put_back_with_an_older_date_still_makes_a_new_mix(root, server, client, track, at):
    folder, tid = _mixed_take(root, server, client)
    other = np.zeros(4800)
    other[at] = 0.5
    old = os.stat(folder / f"{track}.wav").st_mtime_ns - 3_600_000_000_000
    sf.write(folder / f"{track}.wav", other, RATE, subtype="PCM_24")
    os.utime(folder / f"{track}.wav", ns=(old, old))          # same length, and older than the mix
    server.library.write_mix(tid)
    assert 1100 in np.flatnonzero(read_mix(folder)).tolist()


def test_a_delay_put_back_with_an_older_date_still_makes_a_new_mix(root, server, client):
    folder, tid = _mixed_take(root, server, client)
    old = os.stat(folder / "stream.json").st_mtime_ns - 3_600_000_000_000
    (folder / "stream.json").write_text(json.dumps({"calibrated_ms": 20.0}))
    os.utime(folder / "stream.json", ns=(old, old))
    server.library.write_mix(tid)
    # With 20 ms taken off, her click falls before the start, and only the band's is left.
    assert read_mix(folder)[100] / 32767 == pytest.approx(0.25 * BAND * 0.7, abs=1e-3)


def test_a_change_to_how_the_two_are_balanced_makes_a_new_mix(root, server, client, monkeypatch):
    folder, tid = _mixed_take(root, server, client)
    before = _made(folder)
    monkeypatch.setattr(gui, "_TOGETHER_BAND_DB", 6.0)
    server.library.write_mix(tid)
    assert _made(folder) != before
    assert read_mix(folder)[100] / 32767 == pytest.approx((0.5 + 0.25 * 10 ** (6 / 20)) * 0.7, abs=1e-3)


def test_a_new_version_of_the_rule_makes_a_new_mix(root, server, client, monkeypatch):
    folder, tid = _mixed_take(root, server, client)
    before = _made(folder)
    monkeypatch.setattr(gui, "_MIX_RULE", gui._MIX_RULE + 1)
    server.library.write_mix(tid)
    assert _made(folder) != before


@pytest.mark.parametrize("damage", ["cut short", "added to", "another file"])
def test_a_mix_that_is_not_the_whole_file_is_made_again(root, server, client, damage):
    folder, tid = _mixed_take(root, server, client)
    whole = (folder / "mix.wav").read_bytes()
    when = os.stat(folder / "mix.wav").st_mtime_ns + 5_000_000_000
    if damage == "cut short":
        (folder / "mix.wav").write_bytes(whole[:-1])
    elif damage == "added to":
        (folder / "mix.wav").write_bytes(whole + b"\0\0")
    else:
        sf.write(folder / "mix.wav", np.zeros(4800), RATE, subtype="PCM_16")
    os.utime(folder / "mix.wav", ns=(when, when))          # newer than everything it is made from
    server.library.write_mix(tid)
    assert (folder / "mix.wav").read_bytes() == whole


def test_the_mix_can_be_read_by_anything_that_reads_a_wav(root, server, client):
    import wave
    folder, _ = _mixed_take(root, server, client)
    with wave.open(str(folder / "mix.wav")) as w:
        assert (w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()) == (4800, RATE, 1, 2)
    assert sf.info(folder / "mix.wav").frames == 4800


@pytest.mark.skipif(oscompat.IS_WINDOWS, reason="modes are POSIX")
@pytest.mark.parametrize("mode", [0o644, 0o640, 0o600])
def test_the_mix_can_be_read_by_whoever_can_read_the_recordings(root, server, client, mode):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine)
    os.chmod(folder / "human.wav", mode)
    server.library.write_mix(take_id(client, "t"))
    assert stat.S_IMODE(os.stat(folder / "mix.wav").st_mode) == mode


def test_a_mix_too_long_for_a_wav_is_not_made_and_the_folder_still_opens(root, server, shown, client, monkeypatch):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine)
    monkeypatch.setattr(gui, "_WAV_MAX_BYTES", 44 + 4800 * 2 - 1)
    tid = take_id(client, "t")
    assert server.library.write_mix(tid) is None
    resp, out = client.post("/api/reveal", {"take": tid})
    assert resp.status == 200 and json.loads(out)["mix"] is False
    assert shown == [(folder.resolve(), None)] and not (folder / "mix.wav").exists()


def test_a_header_that_cannot_hold_the_length_does_not_stop_the_folder_opening(root, shown, client, monkeypatch):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine)

    def too_big(*args, **kw):
        raise OverflowError("int too big to convert")

    monkeypatch.setattr(gui, "_wav_header", too_big)
    resp, out = client.post("/api/reveal", {"take": take_id(client, "t")})
    assert resp.status == 200 and json.loads(out)["mix"] is False
    assert shown == [(folder.resolve(), None)]
    assert sorted(n for n in os.listdir(folder) if "mix" in n) == []


def test_a_mix_left_half_written_by_a_killed_run_is_cleared_away(root, server, client):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine)
    stale = folder / ".mix.wav.0123456789abcdef.tmp"
    stale.write_bytes(b"half")
    long_ago = time.time_ns() - 3_600_000_000_000
    os.utime(stale, ns=(long_ago, long_ago))
    fresh = folder / ".mix.wav.fedcba9876543210.tmp"          # another run may be writing this now
    fresh.write_bytes(b"half")
    mine = folder / ".mix.wav.notes.tmp"                        # not a name plume makes
    mine.write_bytes(b"keep")
    os.utime(mine, ns=(long_ago, long_ago))
    server.library.write_mix(take_id(client, "t"))
    assert not stale.exists() and fresh.exists() and mine.exists()


def test_two_clicks_at_once_make_one_mix(root, server, client, monkeypatch):
    human, machine = clicks()
    make_take(root / "t", human=human, machine=machine)
    tid = take_id(client, "t")
    real, guard = gui._mixed_pcm, threading.Lock()
    state = {"now": 0, "most": 0, "writes": 0}

    def slow(*args):
        with guard:
            state["now"] += 1
            state["writes"] += 1
            state["most"] = max(state["most"], state["now"])
        time.sleep(0.2)
        with guard:
            state["now"] -= 1
        return real(*args)

    monkeypatch.setattr(gui, "_mixed_pcm", slow)
    go = threading.Barrier(2)
    answers = []

    def click():
        go.wait(5)
        answers.append(server.library.write_mix(tid))

    threads = [threading.Thread(target=click) for _ in range(2)]
    [t.start() for t in threads]
    [t.join(10) for t in threads]
    assert answers == ["mix.wav", "mix.wav"]
    assert state["most"] == 1 and state["writes"] == 1


def test_one_take_being_mixed_does_not_hold_up_another(root, server, client, monkeypatch):
    human, machine = clicks()
    make_take(root / "a", human=human, machine=machine)
    make_take(root / "b", human=human, machine=machine)
    ids = [take_id(client, "a"), take_id(client, "b")]
    real, both = gui._mixed_pcm, threading.Barrier(2)

    def together(*args):
        both.wait(5)          # only passes if the two takes are being mixed at the same moment
        return real(*args)

    monkeypatch.setattr(gui, "_mixed_pcm", together)
    answers, errors = [], []

    def click(tid):
        try:
            answers.append(server.library.write_mix(tid))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=click, args=(tid,)) for tid in ids]
    [t.start() for t in threads]
    [t.join(15) for t in threads]
    assert errors == [] and answers == ["mix.wav", "mix.wav"]


def test_recording_again_into_a_folder_takes_the_old_mix_away(root, server, client):
    folder, _ = _mixed_take(root, server, client)
    assert (folder / "mix.wav").exists()
    recorder = realtime.SessionRecorder(folder, sr=RATE).open({"seed": 2})
    try:
        assert not (folder / "mix.wav").exists()
    finally:
        recorder.close()


# ------------------------------------------------------------------ a mic that may have heard the band

@pytest.mark.parametrize("audio, twice", [
    ({"source": "direct"}, False),
    ({"source": "room"}, True),
    ({"source": "room", "isolated": True}, False),
    ({"source": "room", "isolated": "yes"}, True),
    ({"source": "room", "isolated": 1}, True),
    ({}, True),
    (None, True),
    ("room", True),
])
def test_show_files_says_when_the_band_may_be_in_the_mix_twice(root, shown, client, audio, twice):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine)
    (folder / "session.json").write_text(json.dumps({"seed": 1, "input": audio}))
    resp, out = client.post("/api/reveal", {"take": take_id(client, "t")})
    assert resp.status == 200 and json.loads(out) == {"shown": True, "mix": True, "band_twice": twice}


def test_no_mix_is_never_said_to_hold_the_band_twice(root, shown, client):
    human, _ = clicks()
    make_take(root / "t", human=human, source="room")
    _, out = client.post("/api/reveal", {"take": take_id(client, "t")})
    assert json.loads(out) == {"shown": True, "mix": False, "band_twice": False}


# ------------------------------------------------------------------ per platform

WIN = {"SystemRoot": "C:\\Windows"}


@pytest.mark.parametrize("platform, folder, select, want", [
    ("darwin", "/music/take", None, ["/usr/bin/open", "-R", "/music/take"]),
    ("darwin", "/music/take", "mix.wav", ["/usr/bin/open", "-R", "/music/take/mix.wav"]),
    # A folder named like an app is shown, never started.
    ("darwin", "/music/My Band.app", None, ["/usr/bin/open", "-R", "/music/My Band.app"]),
    ("darwin", "/music/loop.workflow", None, ["/usr/bin/open", "-R", "/music/loop.workflow"]),
    ("linux", "/music/take", None, ["xdg-open", "/music/take"]),
    ("linux", "/music/take", "mix.wav", ["xdg-open", "/music/take"]),
    ("freebsd14", "/music/take", None, ["xdg-open", "/music/take"]),
    ("win32", "C:\\Music\\take", None, '"C:\\Windows\\explorer.exe" "C:\\Music\\take"'),
    ("win32", "C:\\Music\\take", "mix.wav", '"C:\\Windows\\explorer.exe" /select,"C:\\Music\\take\\mix.wav"'),
    # The file manager there splits on commas and spaces unless the path is quoted.
    ("win32", "C:\\Users\\a b\\Music\\Plume\\one, two", "mix.wav",
     '"C:\\Windows\\explorer.exe" /select,"C:\\Users\\a b\\Music\\Plume\\one, two\\mix.wav"'),
    ("win32", "C:\\Users\\a b\\Music\\Plume\\one, two\\", None,
     '"C:\\Windows\\explorer.exe" "C:\\Users\\a b\\Music\\Plume\\one, two"'),
])
def test_each_system_is_asked_in_its_own_words(platform, folder, select, want):
    assert oscompat.reveal_command(folder, select, platform=platform, environ=WIN) == want


def test_the_windows_file_manager_is_named_in_full_wherever_windows_lives():
    assert oscompat.reveal_command("D:\\t", platform="win32", environ={"SystemRoot": "E:\\WinNT"}) \
        == '"E:\\WinNT\\explorer.exe" "D:\\t"'
    assert oscompat.reveal_command("D:\\t", platform="win32", environ={}) == '"C:\\Windows\\explorer.exe" "D:\\t"'


def test_a_windows_path_that_could_end_its_own_quotes_is_refused():
    with pytest.raises(ValueError):
        oscompat.reveal_command('C:\\take" /root,C:\\', platform="win32", environ=WIN)


@pytest.mark.parametrize("select", ["../other", "a/b", "a\\b", "..", ""])
def test_only_a_plain_file_name_can_be_picked_out(select):
    with pytest.raises(ValueError):
        oscompat.reveal_command("/music/take", select, platform="darwin")


def test_reveal_hands_one_command_to_the_launcher(tmp_path):
    ran = []
    oscompat.reveal(tmp_path, None, run=ran.append)
    assert ran == [oscompat.reveal_command(tmp_path)]


class FakeProcess:
    def __init__(self, code=0, hangs=False):
        self.code, self.hangs, self.waits = code, hangs, []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.hangs and timeout is not None:
            raise subprocess.TimeoutExpired("file manager", timeout)
        return self.code


class FakePopen:
    def __init__(self, proc):
        self.proc, self.calls = proc, []

    def __call__(self, *args, **kw):
        self.calls.append((args, kw))
        return self.proc


def test_the_launcher_starts_one_program_and_never_a_shell():
    popen = FakePopen(FakeProcess())
    argv = ["/usr/bin/open", "-R", "/music/a take; rm -rf $HOME/mix.wav"]
    oscompat._launch(argv, popen=popen, wait=True)
    (args, kw), = popen.calls
    assert args == (argv,)
    assert not kw.get("shell") and "executable" not in kw
    assert kw["stdin"] == kw["stdout"] == kw["stderr"] == subprocess.DEVNULL


def test_a_file_manager_that_starts_and_then_fails_is_a_failure():
    with pytest.raises(OSError):
        oscompat._launch(["xdg-open", "/music/take"], popen=FakePopen(FakeProcess(code=3)), wait=True)


def test_a_file_manager_that_stays_open_is_not_waited_for():
    proc = FakeProcess(hangs=True)
    began = time.monotonic()
    oscompat._launch(["xdg-open", "/music/take"], popen=FakePopen(proc), wait=True)
    assert time.monotonic() - began < 2
    assert proc.waits[0] is not None and proc.waits[0] <= 3
    wait_until_reaped = time.monotonic() + 5
    while len(proc.waits) < 2 and time.monotonic() < wait_until_reaped:
        time.sleep(0.01)
    assert proc.waits[1:] == [None], "it is still reaped when it ends"


def test_where_the_exit_code_means_nothing_it_is_not_read():
    proc = FakeProcess(code=1)
    oscompat._launch('"C:\\Windows\\explorer.exe" "C:\\t"', popen=FakePopen(proc), wait=False)
    assert all(timeout is None for timeout in proc.waits)


def test_the_exit_code_is_read_everywhere_but_windows():
    assert oscompat.LAUNCH_EXIT_MEANS_SOMETHING is (not oscompat.IS_WINDOWS)


def test_a_failed_file_manager_reaches_the_page_as_a_failure(root, tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))
    human, machine = clicks()
    make_take(root / "t", human=human, machine=machine)
    popen = FakePopen(FakeProcess(code=1))

    def revealer(folder, select):
        oscompat.reveal(folder, select, run=lambda command: oscompat._launch(command, popen=popen, wait=True))

    srv = gui.GuiServer(roots=[root], log=lambda line: None, revealer=revealer).start()
    try:
        client = Client(srv).login()
        resp, out = client.post("/api/reveal", {"take": take_id(client, "t")})
    finally:
        srv.close()
    assert len(popen.calls) == 1
    assert resp.status == 500 and json.loads(out) == {"error": "could not open the folder"}
