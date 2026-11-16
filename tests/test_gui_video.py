"""A video dropped on a take: taken in, lined up, and saved with the band in it.

Nothing here opens an audio device, plays a sound or opens a file manager window.
The takes are a few seconds of synthetic playing, the videos a 32-pixel square made
by ffmpeg itself, and the launcher is a list that remembers what it was asked for.
Nothing waits on a clock: a test that needs the work held part way holds it with an
event of its own.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import threading

import numpy as np
import pytest
import soundfile as sf

from conftest import needs_symlinks
from plume import gui, takevideo
from test_gui_bridge import Client
from test_gui_files import make_take, take_id
from video_fixture import (FFMPEG, heard_by_camera, make_video, needs_ffmpeg, picture_md5, placed, playing,
                           talking)

RATE = 48000
TAKE_S, VIDEO_S = 8.0, 14.0
START = 3 * RATE + 12345                # where her recording begins in the video: 3.257 s
BAND_AHEAD_MS = 250.0
CLICK_AT = RATE                         # the band's one loud click, a second into its recording
TOLERANCE_S = 0.005
PILOT_HZ = 2500.0                       # a hum only the camera heard
WAIT_S = 120.0                          # only ever reached by a failure


def pilot(n):
    return 0.05 * np.sin(2 * np.pi * PILOT_HZ * np.arange(n) / RATE)


def band():
    machine = np.zeros(int(TAKE_S * RATE))
    machine[CLICK_AT] = 0.9
    return machine


@pytest.fixture(scope="module")
def human():
    return playing(TAKE_S, RATE, seed=31) * 0.5


@pytest.fixture(scope="module")
def film(tmp_path_factory, human):
    """The video of that take: her playing across the room, talking before and after, a hum."""
    if FFMPEG is None:
        pytest.skip("needs ffmpeg")
    n = int(VIDEO_S * RATE)
    talk = talking(VIDEO_S, RATE, 32)
    talk[START:START + len(human)] *= 0.3
    sound = placed(heard_by_camera(human, RATE), START, n) + talk + pilot(n)
    path = make_video(tmp_path_factory.mktemp("film") / "in.mp4", seconds=VIDEO_S, sound=sound,
                      tags=("title=secret-title", "location=+12.3456-045.6789/", "comment=made on a secret-device"))
    return path.read_bytes()


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


@pytest.fixture
def take(root, human):
    return make_take(root / "t", human=human, machine=band(),
                     stream={"calibrated_ms": BAND_AHEAD_MS, "reported_ms": 999.0})


def upload(client, tid, data, *, sound=True, **headers):
    send = {"Origin": client.origin, "Content-Type": "application/octet-stream",
            "X-Plume-Token": client.csrf, "Sec-Fetch-Site": "same-origin"}
    send.update(headers)
    resp, out = client.request("POST", f"/api/video/{tid}?sound={1 if sound else 0}", headers=send, body=data)
    return resp.status, json.loads(out)


def status(client, tid):
    resp, out = client.get(f"/api/take/{tid}/video")
    assert resp.status == 200
    return json.loads(out)


def finished(server, client, tid):
    assert server.library.video_wait(tid, WAIT_S), "the work never ended"
    return status(client, tid)


def temps(folder):
    return sorted(name for name in os.listdir(folder) if name.endswith(".tmp"))


def heard(folder, name="video.mp4"):
    return takevideo.decode(FFMPEG, folder / name, rate=RATE)


def level(sound, lo_s, hi_s):
    return float(np.sqrt(np.mean(sound[int(lo_s * RATE):int(hi_s * RATE)] ** 2)))


def pilot_level(sound):
    """How much of the camera's hum is in a sound: its size at that one pitch."""
    t = np.arange(len(sound)) / RATE
    return float(abs(np.sum(sound * np.exp(-2j * np.pi * PILOT_HZ * t))) / len(sound) * 2)


# ------------------------------------------------------------------ what is offered

def test_a_take_with_both_recordings_can_have_a_video(client, take):
    doc = status(client, take_id(client, "t"))
    assert doc["state"] == "idle" and doc["can"] is True and doc["theirs"] is False
    assert doc["ready"] is (FFMPEG is not None)
    assert doc["max_bytes"] == gui._VIDEO_MAX_BYTES


def test_a_take_with_one_recording_cannot(root, client, human):
    make_take(root / "solo", human=human)
    assert status(client, take_id(client, "solo"))["can"] is False


def test_a_take_still_being_recorded_cannot(client, take, monkeypatch):
    tid = take_id(client, "t")
    monkeypatch.setattr(gui, "_live_folders", lambda: {str(take.resolve())})
    assert status(client, tid)["can"] is False
    code, doc = upload(client, tid, b"x" * 64)
    assert code == 409 and doc["why"] == "no_take" and temps(take) == []


def test_without_ffmpeg_the_page_is_told_and_nothing_is_taken_in(root, take, shown, tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))
    srv = gui.GuiServer(roots=[root], log=lambda line: None, ffmpeg=lambda: None).start()
    try:
        client = Client(srv).login()
        tid = take_id(client, "t")
        assert status(client, tid)["ready"] is False
        code, doc = upload(client, tid, b"x" * 64)
        assert code == 409 and doc["why"] == "no_tool"
        assert temps(take) == [] and not (take / "video.mp4").exists()
    finally:
        srv.close()


def test_the_ffmpeg_is_looked_for_once(root, take, tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))
    asked = []
    srv = gui.GuiServer(roots=[root], log=lambda line: None, ffmpeg=lambda: asked.append(1)).start()
    try:
        client = Client(srv).login()
        for _ in range(3):
            status(client, take_id(client, "t"))
    finally:
        srv.close()
    assert asked == [1]


# ---------------------------------------------------------------- the whole way

@needs_ffmpeg
def test_a_dropped_video_comes_back_with_the_band_where_she_heard_it(server, client, take, film):
    tid = take_id(client, "t")
    code, doc = upload(client, tid, film)
    assert code == 202 and doc["state"] in ("listening", "making", "done")
    doc = finished(server, client, tid)
    assert doc["state"] == "done" and doc["file"] == "video.mp4" and doc["have"] is True
    assert doc["start_s"] == pytest.approx(START / RATE, abs=TOLERANCE_S)
    assert doc["confidence"] >= takevideo.CONFIDENT
    out = heard(take)
    assert len(out) / RATE == pytest.approx(VIDEO_S, abs=0.3)
    # The band's click, a second into its recording, lands the round trip after her start.
    want = START / RATE + BAND_AHEAD_MS / 1000.0 + CLICK_AT / RATE
    assert int(np.argmax(np.abs(out))) / RATE == pytest.approx(want, abs=TOLERANCE_S)
    assert temps(take) == []
    assert takevideo.is_own_video(take / "video.mp4")


@needs_ffmpeg
def test_the_picture_is_copied_and_nothing_the_camera_wrote_comes_along(server, client, take, film, tmp_path):
    tid = take_id(client, "t")
    upload(client, tid, film)
    assert finished(server, client, tid)["state"] == "done"
    (tmp_path / "in.mp4").write_bytes(film)
    assert picture_md5(take / "video.mp4") == picture_md5(tmp_path / "in.mp4")
    made = (take / "video.mp4").read_bytes()
    # The title and the maker's note as text, and the place as the box an mp4 keeps it in.
    for secret in (b"secret-title", b"loci", b"secret-device"):
        assert secret in film and secret not in made
    probe = takevideo.probe(FFMPEG, take / "video.mp4")
    assert probe.has_audio and probe.duration_s == pytest.approx(VIDEO_S, abs=0.3)


@needs_ffmpeg
def test_keeping_the_videos_sound_keeps_her_talking_before_and_after(server, client, take, film):
    tid = take_id(client, "t")
    upload(client, tid, film, sound=True)
    finished(server, client, tid)
    out = heard(take)
    assert level(out, 0.5, 2.5) > 0.01 and level(out, 12.0, 13.5) > 0.01
    assert pilot_level(out) == pytest.approx(0.05, rel=0.3)


@needs_ffmpeg
def test_without_the_videos_sound_only_the_band_and_her_recording_are_heard(server, client, take, film, human):
    tid = take_id(client, "t")
    upload(client, tid, film, sound=False)
    assert finished(server, client, tid)["state"] == "done"
    out = heard(take)
    assert level(out, 0.5, 2.5) < 0.001 and level(out, 12.0, 13.5) < 0.001
    assert pilot_level(out) < 0.002
    # Her clean recording is there, where the video has her playing ...
    lo = START + int(0.25 * RATE)
    clean = human[int(0.25 * RATE):int(0.25 * RATE) + RATE // 2]
    lags = np.arange(-240, 241)
    fit = [float(np.dot(out[lo + lag:lo + lag + len(clean)], clean)) for lag in lags]
    assert abs(int(lags[int(np.argmax(fit))])) / RATE <= TOLERANCE_S
    assert max(fit) > 0.3 * float(np.dot(clean, clean)) * 0.7
    # ... and the band's click still where she heard it.
    want = START / RATE + BAND_AHEAD_MS / 1000.0 + CLICK_AT / RATE
    assert int(np.argmax(np.abs(out))) / RATE == pytest.approx(want, abs=TOLERANCE_S)


@needs_ffmpeg
def test_a_take_that_does_not_say_its_delay_puts_the_band_with_the_player(server, client, root, film, human):
    folder = make_take(root / "jam", human=human, machine=band())           # no stream.json
    tid = take_id(client, "jam")
    upload(client, tid, film, sound=False)
    doc = finished(server, client, tid)
    assert doc["state"] == "done"
    out = heard(folder)
    assert int(np.argmax(np.abs(out))) / RATE == pytest.approx((START + CLICK_AT) / RATE, abs=TOLERANCE_S)


@needs_ffmpeg
def test_a_video_that_began_after_the_take_did(server, client, root, human, tmp_path):
    late = 2 * RATE                                  # the camera started two seconds into the take
    n = int(5.0 * RATE)
    sound = placed(heard_by_camera(human, RATE), -late, n) + talking(5.0, RATE, 33, 0.05)
    data = make_video(tmp_path / "late.mp4", seconds=5.0, sound=sound).read_bytes()
    machine = np.zeros(int(TAKE_S * RATE))
    machine[3 * RATE] = 0.9                          # three seconds in: one second into the video
    folder = make_take(root / "late", human=human, machine=machine)
    tid = take_id(client, "late")
    upload(client, tid, data, sound=False)
    doc = finished(server, client, tid)
    assert doc["state"] == "done" and doc["start_s"] == pytest.approx(-2.0, abs=TOLERANCE_S)
    out = heard(folder)
    assert len(out) / RATE == pytest.approx(5.0, abs=0.3)
    assert int(np.argmax(np.abs(out))) / RATE == pytest.approx(1.0, abs=TOLERANCE_S)


@needs_ffmpeg
def test_show_video_shows_the_file_in_its_folder(server, client, take, film, shown):
    tid = take_id(client, "t")
    assert client.post("/api/video/show", {"take": tid})[0].status == 404       # nothing to show yet
    upload(client, tid, film)
    finished(server, client, tid)
    resp, out = client.post("/api/video/show", {"take": tid})
    assert resp.status == 200 and json.loads(out) == {"shown": True}
    assert shown == [(take.resolve(), "video.mp4")]
    assert not (take / "mix.wav").exists()           # showing a video makes nothing else


@needs_ffmpeg
def test_a_second_video_replaces_plumes_own_first(server, client, take, film):
    tid = take_id(client, "t")
    upload(client, tid, film, sound=True)
    finished(server, client, tid)
    first = (take / "video.mp4").read_bytes()
    assert status(client, tid)["theirs"] is False
    upload(client, tid, film, sound=False)
    assert finished(server, client, tid)["state"] == "done"
    assert (take / "video.mp4").read_bytes() != first and temps(take) == []


# ------------------------------------------------------------- when it is unsure

@pytest.fixture
def strange(tmp_path):
    """A video of something else: nothing in its sound is the take."""
    sound = talking(6.0, RATE, 41) + heard_by_camera(playing(6.0, RATE, seed=42), RATE)
    return make_video(tmp_path / "strange.mp4", seconds=6.0, sound=sound).read_bytes()


@needs_ffmpeg
def test_an_unsure_match_saves_nothing_and_offers_a_nudge(server, client, take, strange):
    tid = take_id(client, "t")
    upload(client, tid, strange)
    doc = finished(server, client, tid)
    assert doc["state"] == "unsure" and doc["why"] == "unsure" and doc["listen"] is True
    assert doc["confidence"] < takevideo.CONFIDENT
    assert doc["min_s"] <= doc["start_s"] <= doc["max_s"] and doc["step_s"] == 0.01
    assert not (take / "video.mp4").exists()
    assert len(temps(take)) == 1                     # the video waits here for her answer


@needs_ffmpeg
def test_a_nudged_place_is_the_place_used(server, client, take, strange):
    tid = take_id(client, "t")
    upload(client, tid, strange)
    finished(server, client, tid)
    resp, out = client.post("/api/video/place", {"take": tid, "start_s": 1.5, "sound": False})
    assert resp.status == 202
    doc = finished(server, client, tid)
    assert doc["state"] == "done" and doc["start_s"] == 1.5
    out = heard(take)
    want = 1.5 + BAND_AHEAD_MS / 1000.0 + CLICK_AT / RATE
    assert int(np.argmax(np.abs(out))) / RATE == pytest.approx(want, abs=TOLERANCE_S)
    assert temps(take) == []


@needs_ffmpeg
def test_a_preview_is_a_few_seconds_of_both_and_writes_nothing(server, client, take, strange):
    tid = take_id(client, "t")
    upload(client, tid, strange)
    finished(server, client, tid)
    before = sorted(os.listdir(take))
    resp, out = client.get(f"/api/take/{tid}/video/preview?at=0.0&k={client.csrf}",
                           headers={"X-Plume-Token": None})          # as an audio element asks
    assert resp.status == 200 and resp.getheader("Content-Type") == "audio/wav"
    assert out[:4] == b"RIFF" and len(out) == 44 + 2 * int(gui._VIDEO_PREVIEW_S * RATE)
    pcm = np.frombuffer(out[44:], dtype="<i2") / 32768.0
    assert level(pcm, 0.0, 6.0) > 0.01               # the video's own sound is in it
    # ... and so is the band's click: the preview starts a second before she first plays.
    first = takevideo.first_sound_s(takevideo.decode(FFMPEG, open(take / "human.wav", "rb")))

    def click_in_preview(at):
        from_s = max(0.0, at + first - gui._VIDEO_PREVIEW_LEAD_S)
        return at + BAND_AHEAD_MS / 1000.0 + CLICK_AT / RATE - from_s

    assert int(np.argmax(np.abs(pcm))) / RATE == pytest.approx(click_in_preview(0.0), abs=TOLERANCE_S)
    assert sorted(os.listdir(take)) == before
    # Nudged to a place before the video began, the click comes that much sooner.
    _, out = client.get(f"/api/take/{tid}/video/preview?at=-0.5&k={client.csrf}")
    pcm = np.frombuffer(out[44:], dtype="<i2") / 32768.0
    assert click_in_preview(-0.5) == pytest.approx(click_in_preview(0.0) - 0.5)
    assert int(np.argmax(np.abs(pcm))) / RATE == pytest.approx(click_in_preview(-0.5), abs=TOLERANCE_S)


@needs_ffmpeg
@pytest.mark.parametrize("at", ["", "nan", "inf", "1e9", "-1e9", "x", "1.5&at=2"])
def test_a_preview_asked_for_badly_is_refused(server, client, take, strange, at):
    tid = take_id(client, "t")
    upload(client, tid, strange)
    finished(server, client, tid)
    assert client.get(f"/api/take/{tid}/video/preview?at={at}")[0].status == 400


def test_a_preview_or_a_place_with_nothing_waiting_is_refused(client, take):
    tid = take_id(client, "t")
    assert client.get(f"/api/take/{tid}/video/preview?at=0")[0].status == 404
    assert client.post("/api/video/place", {"take": tid, "start_s": 0.0, "sound": True})[0].status == 409


@needs_ffmpeg
@pytest.mark.parametrize("change", [{"take": None}, {"take": 5}, {"start_s": "1"}, {"start_s": True},
                                    {"start_s": None}, {"sound": "yes"}, {"sound": None}, {"start_s": 1e9},
                                    {"start_s": -1e9}, {"path": "/etc/passwd"}, {"name": "../x.mp4"}])
def test_a_place_asked_for_badly_is_refused(server, client, take, strange, change):
    tid = take_id(client, "t")
    upload(client, tid, strange)
    finished(server, client, tid)
    body = {"take": tid, "start_s": 1.0, "sound": True, **change}
    body = {k: v for k, v in body.items() if v is not None}
    assert client.post("/api/video/place", body)[0].status == 400
    assert status(client, tid)["state"] == "unsure" and not (take / "video.mp4").exists()


@needs_ffmpeg
@pytest.mark.parametrize("sound", [None, np.zeros(int(4.0 * RATE))])
def test_a_video_with_no_usable_sound_is_placed_by_hand(server, client, take, tmp_path, sound):
    data = make_video(tmp_path / "mute.mp4", seconds=4.0, sound=sound).read_bytes()
    tid = take_id(client, "t")
    upload(client, tid, data)
    doc = finished(server, client, tid)
    assert doc["state"] == "unsure" and doc["why"] == "silent" and doc["listen"] is False
    assert doc["start_s"] == 0.0 and doc["min_s"] < 0 < doc["max_s"]
    assert doc["max_s"] == pytest.approx(4.0, abs=0.3) and doc["min_s"] == pytest.approx(-TAKE_S, abs=0.3)
    client.post("/api/video/place", {"take": tid, "start_s": 0.5, "sound": True})
    assert finished(server, client, tid)["state"] == "done"
    out = heard(take)
    want = 0.5 + BAND_AHEAD_MS / 1000.0 + CLICK_AT / RATE
    assert int(np.argmax(np.abs(out))) / RATE == pytest.approx(want, abs=TOLERANCE_S)


# ------------------------------------------------------------------- whose file

@needs_ffmpeg
def test_the_players_own_video_is_never_replaced(server, client, take, film):
    (take / "video.mp4").write_bytes(b"her own film")
    tid = take_id(client, "t")
    assert status(client, tid)["theirs"] is True
    code, doc = upload(client, tid, film)
    assert code == 409 and doc["why"] == "theirs"
    assert (take / "video.mp4").read_bytes() == b"her own film" and temps(take) == []


@needs_ffmpeg
def test_a_video_of_hers_that_arrives_while_plume_works_is_kept(server, client, take, film, monkeypatch):
    real = takevideo.write_video

    def and_then_she_saves_one(*a, **kw):
        real(*a, **kw)
        (take / "video.mp4").write_bytes(b"her own film")

    monkeypatch.setattr(takevideo, "write_video", and_then_she_saves_one)
    tid = take_id(client, "t")
    upload(client, tid, film)
    doc = finished(server, client, tid)
    assert doc["state"] == "failed" and doc["why"] == "theirs"
    assert (take / "video.mp4").read_bytes() == b"her own film" and temps(take) == []


@needs_symlinks
@needs_ffmpeg
def test_a_symlink_where_the_video_goes_is_not_written_through(server, client, take, film, tmp_path):
    target = tmp_path / "elsewhere.mp4"
    target.write_bytes(b"not plume's")
    os.symlink(target, take / "video.mp4")
    tid = take_id(client, "t")
    code, doc = upload(client, tid, film)
    assert code == 409 and doc["why"] == "theirs"
    assert target.read_bytes() == b"not plume's" and (take / "video.mp4").is_symlink()


@needs_symlinks
def test_a_recording_that_is_a_symlink_is_not_a_take_to_film(root, client, human, tmp_path):
    folder = make_take(root / "linked", human=human)
    real = tmp_path / "machine.wav"
    sf.write(real, band(), RATE)
    os.symlink(real, folder / "machine.wav")
    assert status(client, take_id(client, "linked"))["can"] is False


# ------------------------------------------------------- what is left behind

@needs_ffmpeg
@pytest.mark.parametrize("data", [b"not a film at all\n" * 100, b"\x00" * 4096, b"RIFF" + bytes(64)])
def test_what_is_not_a_video_is_refused_kindly_and_nothing_is_left(server, client, take, data):
    tid = take_id(client, "t")
    code, _ = upload(client, tid, data)
    assert code == 202
    doc = finished(server, client, tid)
    assert doc["state"] == "failed" and doc["why"] == "not_video"
    assert temps(take) == [] and not (take / "video.mp4").exists()


@needs_ffmpeg
def test_a_sound_file_is_not_a_video(server, client, take):
    tid = take_id(client, "t")
    upload(client, tid, (take / "human.wav").read_bytes())
    assert finished(server, client, tid)["why"] == "not_video"
    assert temps(take) == []


def test_a_video_too_big_is_refused_before_a_byte_is_kept(client, take, monkeypatch):
    monkeypatch.setattr(gui, "_VIDEO_MAX_BYTES", 1000)
    tid = take_id(client, "t")
    code, doc = upload(client, tid, b"x" * 1001)
    assert code == 413 and doc["why"] == "too_big"
    assert temps(take) == [] and status(client, tid)["state"] == "idle"


def test_a_video_the_disk_has_no_room_for_is_refused(client, take, monkeypatch):
    monkeypatch.setattr(gui, "_free_bytes", lambda folder: 100)
    tid = take_id(client, "t")
    code, doc = upload(client, tid, b"x" * 500)
    assert code == 507 and doc["why"] == "no_room" and temps(take) == []


def test_a_refusal_waits_for_the_whole_video_so_the_page_hears_why(server, client, take):
    """A browser that is still sending cannot read an answer: it sees the line drop
    instead, and the player would be told nothing she can use."""
    (take / "video.mp4").write_bytes(b"her own film")
    tid = take_id(client, "t")
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=30)
    head = (f"POST /api/video/{tid}?sound=1 HTTP/1.1\r\nHost: {client.host}\r\nCookie: {client.cookie}\r\n"
            f"X-Plume-Token: {client.csrf}\r\nOrigin: {client.origin}\r\n"
            "Content-Type: application/octet-stream\r\nContent-Length: 3000000\r\n\r\n")
    sock.sendall(head.encode())
    sock.settimeout(0.0)
    with pytest.raises(BlockingIOError):
        sock.recv(1)                                 # nothing said yet: the video is still coming
    sock.settimeout(30)
    for _ in range(30):
        sock.sendall(b"x" * 100000)                  # never refused part way
    answer = b""
    while b"theirs" not in answer:
        got = sock.recv(65536)
        assert got, answer
        answer += got
    sock.close()
    assert answer.startswith(b"HTTP/1.1 409") and temps(take) == []


@pytest.mark.parametrize("length", [None, "", "x", "-5", "0"])
def test_an_upload_must_say_how_long_it_is(server, client, take, length):
    tid = take_id(client, "t")
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    conn.putrequest("POST", f"/api/video/{tid}?sound=1", skip_host=True, skip_accept_encoding=True)
    for name, value in (("Host", client.host), ("Cookie", client.cookie), ("X-Plume-Token", client.csrf),
                        ("Origin", client.origin), ("Content-Type", "application/octet-stream")):
        conn.putheader(name, value)
    if length is not None:
        conn.putheader("Content-Length", length)
    conn.endheaders()
    assert conn.getresponse().status in (400, 411)
    conn.close()
    assert temps(take) == []


def test_an_upload_that_stops_short_leaves_nothing(server, client, take):
    tid = take_id(client, "t")
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    head = (f"POST /api/video/{tid}?sound=1 HTTP/1.1\r\nHost: {client.host}\r\nCookie: {client.cookie}\r\n"
            f"X-Plume-Token: {client.csrf}\r\nOrigin: {client.origin}\r\n"
            "Content-Type: application/octet-stream\r\nContent-Length: 100000\r\n\r\n")
    sock.sendall(head.encode() + b"x" * 5000)
    sock.shutdown(socket.SHUT_WR)                    # the tab closed part way through
    sock.recv(65536)                                 # the bridge has seen the end by the time it answers or hangs up
    sock.close()
    assert server.library.video_wait(tid, WAIT_S)
    assert temps(take) == [] and status(client, tid)["state"] == "idle"


def test_leftovers_from_a_run_that_died_are_swept_and_fresh_ones_are_not(server, client, take, monkeypatch):
    old_in, old_out = take / (".video-in." + "a" * 16 + ".tmp"), take / (".video.mp4." + "b" * 16 + ".tmp")
    fresh, other = take / (".video-in." + "c" * 16 + ".tmp"), take / "her-notes.tmp"
    for path in (old_in, old_out, fresh, other):
        path.write_bytes(b"x")
    long_ago = 1_000_000_000
    for path in (old_in, old_out, other):
        os.utime(path, (long_ago, long_ago))
    monkeypatch.setattr(gui, "_free_bytes", lambda folder: 0)     # the sweep comes first, whatever follows
    upload(client, take_id(client, "t"), b"x" * 10)
    assert temps(take) == sorted([fresh.name, other.name])


@needs_ffmpeg
def test_the_tool_failing_leaves_the_take_as_it_was(server, client, take, film, monkeypatch):
    def broken(*a, **kw):
        raise takevideo.VideoError("failed")

    monkeypatch.setattr(takevideo, "write_video", broken)
    before = sorted(os.listdir(take))
    tid = take_id(client, "t")
    upload(client, tid, film)
    doc = finished(server, client, tid)
    assert doc["state"] == "failed" and doc["why"] == "failed"
    assert sorted(os.listdir(take)) == before


@needs_ffmpeg
def test_anything_unexpected_is_a_plain_failure_not_a_stuck_job(server, client, take, film, monkeypatch):
    def broken(*a, **kw):
        raise ZeroDivisionError

    monkeypatch.setattr(takevideo, "align", broken)
    tid = take_id(client, "t")
    upload(client, tid, film)
    doc = finished(server, client, tid)
    assert doc["state"] == "failed" and doc["why"] == "failed" and temps(take) == []
    assert upload(client, tid, film)[0] == 202       # and she can try again
    finished(server, client, tid)


# ------------------------------------------------------- one at a time, and stopping

@pytest.fixture
def held(monkeypatch):
    """The work, stopped as it starts to read the video, until ``held.go`` is set."""
    class Held:
        reached, go = threading.Event(), threading.Event()

    real = takevideo.probe

    def probe(*a, **kw):
        Held.reached.set()
        assert Held.go.wait(WAIT_S)
        return real(*a, **kw)

    monkeypatch.setattr(takevideo, "probe", probe)
    yield Held
    Held.go.set()


@needs_ffmpeg
def test_two_videos_for_one_take_do_not_collide(server, client, take, film, held):
    tid = take_id(client, "t")
    assert upload(client, tid, film)[0] == 202
    assert held.reached.wait(WAIT_S)
    assert status(client, tid)["state"] == "listening"
    code, doc = upload(client, tid, film)
    assert code == 409 and doc["why"] == "busy"
    assert len(temps(take)) == 1                     # the first one's, untouched
    held.go.set()
    assert finished(server, client, tid)["state"] == "done"
    assert temps(take) == []


@needs_ffmpeg
def test_another_take_is_not_held_up(server, client, root, take, film, human, held):
    make_take(root / "u", human=human, machine=band())
    assert upload(client, take_id(client, "t"), film)[0] == 202
    assert held.reached.wait(WAIT_S)
    assert upload(client, take_id(client, "u"), film)[0] == 202


@needs_ffmpeg
def test_cancel_stops_the_work_and_leaves_nothing(server, client, take, film, held):
    tid = take_id(client, "t")
    upload(client, tid, film)
    assert held.reached.wait(WAIT_S)
    resp, out = client.post("/api/video/cancel", {"take": tid})
    assert resp.status == 200
    held.go.set()
    doc = finished(server, client, tid)
    assert doc["state"] == "idle"
    assert temps(take) == [] and not (take / "video.mp4").exists()


@needs_ffmpeg
def test_cancel_while_it_waits_for_a_nudge_removes_the_video_it_kept(server, client, take, strange):
    tid = take_id(client, "t")
    upload(client, tid, strange)
    assert finished(server, client, tid)["state"] == "unsure"
    client.post("/api/video/cancel", {"take": tid})
    assert status(client, tid)["state"] == "idle" and temps(take) == []


def test_cancel_with_nothing_running_is_fine(client, take):
    tid = take_id(client, "t")
    resp, out = client.post("/api/video/cancel", {"take": tid})
    assert resp.status == 200 and json.loads(out)["state"] == "idle"
    assert client.post("/api/video/cancel", {"take": "0" * 16})[0].status == 404
    assert client.post("/api/video/cancel", {"take": tid, "more": 1})[0].status == 400


@needs_ffmpeg
def test_closing_the_bridge_stops_the_work_and_leaves_nothing(root, take, film, held, tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))
    srv = gui.GuiServer(roots=[root], log=lambda line: None).start()
    client = Client(srv).login()
    tid = take_id(client, "t")
    upload(client, tid, film)
    assert held.reached.wait(WAIT_S)
    held.go.set()
    srv.close()
    assert srv.library.video_wait(tid, WAIT_S)
    assert temps(take) == []


@needs_ffmpeg
def test_closing_the_bridge_removes_a_video_that_was_waiting_for_a_nudge(root, take, strange, tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))
    srv = gui.GuiServer(roots=[root], log=lambda line: None).start()
    client = Client(srv).login()
    tid = take_id(client, "t")
    upload(client, tid, strange)
    assert finished(srv, client, tid)["state"] == "unsure" and len(temps(take)) == 1
    srv.close()
    assert temps(take) == []


# ------------------------------------------------------------ the request guards

GUARDS = [({"Origin": None}, 403), ({"Origin": "http://127.0.0.1:1"}, 403),
          ({"Sec-Fetch-Site": "cross-site"}, 403), ({"Sec-Fetch-Site": "same-site"}, 403),
          ({"X-Plume-Token": None}, 403), ({"X-Plume-Token": "0" * 64}, 403),
          ({"Cookie": None}, 401), ({"Cookie": "plume_1=" + "0" * 64}, 401)]


@pytest.mark.parametrize("headers,code", GUARDS + [({"Content-Type": "application/json"}, 415),
                                                    ({"Content-Type": "multipart/form-data"}, 415),
                                                    ({"Content-Type": "video/mp4"}, 415)])
def test_an_upload_passes_every_check_a_post_does_before_a_byte_is_kept(client, take, headers, code):
    tid = take_id(client, "t")
    status_code, doc = upload(client, tid, b"x" * 2000, **headers)
    assert status_code == code and "error" in doc
    assert temps(take) == [] and not (take / "video.mp4").exists()


def test_an_upload_from_the_wrong_host_is_refused(client, take):
    tid = take_id(client, "t")
    send = {"Origin": client.origin, "Content-Type": "application/octet-stream", "X-Plume-Token": client.csrf}
    resp, _ = client.request("POST", f"/api/video/{tid}?sound=1", headers=send, body=b"x" * 100,
                             host="evil.example")
    assert resp.status == 403 and temps(take) == []


@pytest.mark.parametrize("op", ["video/place", "video/cancel", "video/show"])
@pytest.mark.parametrize("headers,code", GUARDS + [({"Content-Type": "text/plain"}, 415)])
def test_the_other_video_posts_pass_the_same_checks(client, take, op, headers, code):
    resp, _ = client.post(f"/api/{op}", {"take": take_id(client, "t")}, **headers)
    assert resp.status == code


@pytest.mark.parametrize("path", ["/api/take/{id}/video", "/api/take/{id}/video/preview?at=0"])
def test_reading_about_a_video_needs_the_session_and_the_page_key(client, take, path):
    path = path.replace("{id}", take_id(client, "t"))
    assert client.get(path, headers={"X-Plume-Token": None})[0].status == 401
    assert client.get(path, headers={"Cookie": None})[0].status == 401
    assert client.get(path, headers={"Sec-Fetch-Site": "cross-site"})[0].status == 403
    assert client.get(path, host="evil.example")[0].status == 403


def test_only_the_preview_takes_the_page_key_in_its_address(client, take):
    tid = take_id(client, "t")
    assert client.get(f"/api/take/{tid}/video?k={client.csrf}", headers={"X-Plume-Token": None})[0].status == 401


# ------------------------------------------------------------------ path safety

@pytest.mark.parametrize("name", ["0" * 16, "..", "..%2f..%2fetc", "%2e%2e", "t", "", "a" * 64,
                                  "../../../../etc/passwd", "0" * 16 + "/x"])
def test_only_a_take_the_library_listed_can_be_named(client, take, root, name):
    code, _ = upload(client, name, b"x" * 100)
    assert code == 404
    assert client.get(f"/api/take/{name}/video")[0].status == 404
    assert client.post("/api/video/show", {"take": name})[0].status in (400, 404)
    assert temps(take) == [] and sorted(os.listdir(root)) == ["t"]


@pytest.mark.parametrize("query", ["", "?sound=2", "?sound=1&sound=0", "?sound=1&name=../../x.mp4",
                                   "?name=x.mp4", "?sound=yes"])
def test_an_upload_says_only_whether_to_keep_the_sound(client, take, query):
    tid = take_id(client, "t")
    send = {"Origin": client.origin, "Content-Type": "application/octet-stream", "X-Plume-Token": client.csrf}
    resp, _ = client.request("POST", f"/api/video/{tid}{query}", headers=send, body=b"x" * 100)
    assert resp.status == 400 and temps(take) == []


@needs_ffmpeg
def test_the_only_names_ever_written_are_plumes_own(server, client, take, film):
    before = set(os.listdir(take))
    tid = take_id(client, "t")
    upload(client, tid, film)
    finished(server, client, tid)
    assert set(os.listdir(take)) - before == {"video.mp4"}
    assert sorted(os.listdir(take.parent)) == ["t"]
