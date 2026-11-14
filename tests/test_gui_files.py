"""A take's files: the saved mix, and showing the folder.

Nothing here opens an audio device, plays a sound or opens a file manager window.
The takes are a few thousand synthetic samples, and the launcher is a list that
remembers what it was asked for.  The real launcher refuses to run under pytest,
and that refusal is tested too.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import soundfile as sf

from conftest import needs_symlinks
from plume import gui, oscompat
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
    assert (folder / "mix.wav").read_bytes() == played


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
    assert resp.status == 200 and json.loads(out) == {"shown": True, "mix": True}
    assert shown == [(folder.resolve(), "mix.wav")]
    assert len(read_mix(folder)) == 4800


def test_show_files_still_shows_a_take_that_can_have_no_mix(root, shown, client):
    human, _ = clicks()
    folder = make_take(root / "t", human=human)
    resp, out = client.post("/api/reveal", {"take": take_id(client, "t")})
    assert resp.status == 200 and json.loads(out) == {"shown": True, "mix": False}
    assert shown == [(folder.resolve(), None)]


def test_a_mix_that_cannot_be_written_does_not_stop_the_folder_showing(root, shown, client, monkeypatch):
    human, machine = clicks()
    folder = make_take(root / "t", human=human, machine=machine)

    def broken(*args):
        raise OSError("read-only")

    monkeypatch.setattr(gui, "_mixed_pcm", broken)
    resp, out = client.post("/api/reveal", {"take": take_id(client, "t")})
    assert resp.status == 200 and json.loads(out) == {"shown": True, "mix": False}
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


# ------------------------------------------------------------------ per platform

@pytest.mark.parametrize("platform, folder, select, want", [
    ("darwin", "/music/take", None, ["open", "/music/take"]),
    ("darwin", "/music/take", "mix.wav", ["open", "-R", "/music/take/mix.wav"]),
    ("linux", "/music/take", None, ["xdg-open", "/music/take"]),
    ("linux", "/music/take", "mix.wav", ["xdg-open", "/music/take"]),
    ("freebsd14", "/music/take", None, ["xdg-open", "/music/take"]),
    ("win32", "C:\\Music\\take", None, ["explorer", "C:\\Music\\take"]),
    ("win32", "C:\\Music\\take", "mix.wav", ["explorer", "/select,C:\\Music\\take\\mix.wav"]),
])
def test_each_system_is_asked_in_its_own_words(platform, folder, select, want):
    assert oscompat.reveal_command(folder, select, platform=platform) == want


@pytest.mark.parametrize("select", ["../other", "a/b", "a\\b", "..", ""])
def test_only_a_plain_file_name_can_be_picked_out(select):
    with pytest.raises(ValueError):
        oscompat.reveal_command("/music/take", select, platform="darwin")


def test_reveal_hands_one_command_to_the_launcher(tmp_path):
    ran = []
    oscompat.reveal(tmp_path, None, run=ran.append)
    assert ran == [oscompat.reveal_command(tmp_path)]
    assert ran[0][-1] == str(tmp_path)
