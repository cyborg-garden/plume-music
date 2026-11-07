"""A guitar on one channel of a multi-input interface, through the fake device.

Everything here is a stand-in: no interface, no gain staging, no real pickup.  It
proves the routing and the logic.  How a real guitar reads against these thresholds
is what docs/DIRECT-INPUT-TEST.md is for.
"""
import json
import threading

import numpy as np
import pytest
from conftest import needs_laptop_speed  # noqa: E402
import soundfile as sf

from conftest import fake_audio, patched_module, short_socket_path
from plume import cli, realtime, record
from plume import mind as mind_mod
from plume import roles as roles_mod
from plume.realtime import RealtimeError, resolve_input
from plume.watch import Listener

SR = realtime.SR

RIG = [
    {"name": "Laptop Mic", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "Laptop Speakers", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4},
    {"name": "Monitor Only", "max_input_channels": 0, "max_output_channels": 2},
]


def rig(**kw):
    return fake_audio(None, devices=RIG, **kw)


def rows_of(rec):
    rows = [json.loads(line) for line in (rec / "decisions.ndjson").read_text().splitlines()]
    return [r for r in rows if isinstance(r.get("bar"), int)]


# ------------------------------------------------------------------ routing

def test_no_flags_is_exactly_what_it_always_was():
    route = resolve_input(rig())
    assert (route.device, route.channels, route.guitar, route.voice) == (None, 1, 0, None)
    assert route.source == "room"


def test_input_device_leaves_the_output_on_the_system_default():
    assert resolve_input(rig(), input_device="audio interface").device == (2, None)


def test_channels_are_counted_the_way_the_interface_labels_them():
    route = resolve_input(rig(), input_device="audio interface", input_channel=2)
    assert route.guitar == 1 and route.channels == 2


def test_a_voice_channel_widens_the_stream_to_reach_it():
    route = resolve_input(rig(), input_device="audio interface", input_channel=2, voice_channel=4)
    assert (route.guitar, route.voice, route.channels) == (1, 3, 4)


def test_device_names_both_directions_and_input_device_overrides_one():
    assert resolve_input(rig(), device="audio interface").device == (2, 2)
    assert resolve_input(rig(), device="monitor only",
                         input_device="audio interface").device == (2, 3)


def test_input_channel_works_with_device_alone():
    assert resolve_input(rig(), device="audio interface", input_channel=3).guitar == 2


@pytest.mark.parametrize("kw,says", [
    ({"input_device": "audio interface", "input_channel": 5}, "4"),
    ({"input_device": "audio interface", "voice_channel": 9}, "4"),
    ({"input_device": "audio interface", "input_channel": 0}, "1"),
    ({"input_device": "audio interface", "input_channel": 2, "voice_channel": 2}, "same"),
    ({"input_channel": 2}, "--input-device"),
    ({"voice_channel": 2}, "--input-device"),
    ({"input_device": "nope"}, "Audio Interface"),
    ({"input_device": "monitor"}, "no inputs"),
])
def test_a_bad_route_says_what_is_wrong(kw, says):
    with pytest.raises(RealtimeError, match=says):
        resolve_input(rig(), **kw)


# ------------------------------------------------------------ the transports

def _guitar_and_decoy(seconds=8.0):
    t = np.arange(int(seconds * SR)) / SR
    guitar = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    decoy = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    return guitar, decoy


def _tone_level(path, freq):
    y, sr = sf.read(str(path), dtype="float32")
    spec = np.abs(np.fft.rfft(y[:sr * 2]))
    return spec[int(freq * 2)]


@pytest.fixture
def consent(tmp_path, monkeypatch, plume_registry):
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")


def test_lead_records_the_chosen_channel_and_not_channel_one(tmp_path, consent):
    guitar, decoy = _guitar_and_decoy()
    fake = fake_audio(guitar, devices=RIG, human_channel=1, inputs={0: decoy})
    with patched_module("sounddevice", fake):
        realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(tmp_path / "rec"), status=False, max_minutes=0.05,
                      socket_path=short_socket_path("route"),
                      input_device="audio interface", input_channel=2)
    assert fake.opened[0] == {"device": (2, None), "channels": (2, 2)}
    human = tmp_path / "rec" / "human.wav"
    assert _tone_level(human, 220.0) > 10.0 * _tone_level(human, 1000.0)


def test_jam_records_the_chosen_channel_too(tmp_path, consent):
    from plume import mind as mind_mod
    guitar, decoy = _guitar_and_decoy()
    fake = fake_audio(guitar, devices=RIG, human_channel=2, inputs={0: decoy})
    with patched_module("sounddevice", fake):
        realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(),
                     record_dir=str(tmp_path / "rec"), max_minutes=0.05,
                     input_device="audio interface", input_channel=3, direct=True)
    assert fake.opened[0] == {"device": (2, None), "channels": (3, 2)}
    human = tmp_path / "rec" / "human.wav"
    assert _tone_level(human, 220.0) > 10.0 * _tone_level(human, 1000.0)
    assert json.loads((tmp_path / "rec" / "session.json").read_text())["input"]["source"] == "direct"


def test_monitor_mixes_in_the_chosen_channel(tmp_path, consent):
    guitar, _ = _guitar_and_decoy()
    fake = fake_audio(guitar, devices=RIG, human_channel=1, capture=True)
    with patched_module("sounddevice", fake):
        realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"),
                      status=False, max_minutes=0.03, monitor=True, tap_start=True,
                      socket_path=short_socket_path("mon"),
                      input_device="audio interface", input_channel=2)
    # nothing is played before a count, so all that can come out is the guitar
    assert np.abs(fake.stream.output).max() > 0.1


def test_monitor_follows_the_channel_and_stays_out_of_both_recordings(tmp_path, consent):
    """What she hears of herself is not what the take keeps.  No count, so the band never
    starts: anything at all in machine.wav would be her guitar leaking into it."""
    guitar, decoy = _guitar_and_decoy()
    fake = fake_audio(guitar, devices=RIG, human_channel=1, inputs={0: decoy}, capture=True)
    with patched_module("sounddevice", fake):
        realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"), status=False,
                      max_minutes=0.05, monitor=True, tap_start=True, direct=True,
                      record_dir=str(tmp_path / "rec"), socket_path=short_socket_path("monl"),
                      input_device="audio interface", input_channel=2)
    out = fake.stream.output[:, 0]
    spectrum = np.abs(np.fft.rfft(out[:SR * 2]))
    assert spectrum[int(220.0 * 2)] > 100.0 * spectrum[int(1000.0 * 2)], "the wrong channel"
    # unchanged by the monitor, and the same in both ears
    assert np.abs(out).max() == pytest.approx(0.2, abs=0.01)
    assert np.array_equal(fake.stream.output[:, 0], fake.stream.output[:, 1])
    machine, _ = sf.read(str(tmp_path / "rec" / "machine.wav"), dtype="float32")
    assert machine.size and np.abs(machine).max() == 0.0
    human, _ = sf.read(str(tmp_path / "rec" / "human.wav"), dtype="float32")
    assert np.abs(human).max() == pytest.approx(0.2, abs=0.01), "doubled, or the decoy"
    assert _tone_level(tmp_path / "rec" / "human.wav", 220.0) > 100.0 * _tone_level(
        tmp_path / "rec" / "human.wav", 1000.0)
    assert json.loads((tmp_path / "rec" / "session.json").read_text())["monitor"] is True


# ---------------------------------------------- what the device says about latency

def _stream_doc(tmp_path, *, latency=(0.012, 0.021), tap_start=True):
    fake = fake_audio(np.zeros(SR, dtype=np.float32), devices=RIG, latency=latency)
    with patched_module("sounddevice", fake):
        realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"), status=False,
                      max_minutes=0.01, tap_start=tap_start, direct=True,
                      record_dir=str(tmp_path / "rec"), socket_path=short_socket_path("strm"),
                      input_device="audio interface", input_channel=1)
    return json.loads((tmp_path / "rec" / "stream.json").read_text())


def test_stream_json_holds_what_the_device_reported(tmp_path, consent):
    assert _stream_doc(tmp_path) == {
        "input_ms": 12.0, "output_ms": 21.0, "reported_ms": 33.0,
        "block_ms": pytest.approx(5.333, abs=0.001), "calibrated_ms": None}


def test_stream_json_is_left_with_no_temporary_beside_it(tmp_path, consent):
    _stream_doc(tmp_path)
    assert not list((tmp_path / "rec").glob("*.tmp"))


def test_stream_json_carries_a_real_calibration_when_there_is_one(tmp_path, consent):
    from plume import calibrate
    # The fake device's table can't be asked about one device, so it files under "default".
    calibrate.save_roundtrip("default", "default", SR,
                            {"roundtrip_ms": 41.5, "spread_ms": 0.5, "heard": 8, "clicks": 8})
    assert _stream_doc(tmp_path)["calibrated_ms"] == 41.5


def test_a_used_folder_does_not_keep_the_last_takes_stream(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "stream.json").write_text("{}")
    recorder = realtime.SessionRecorder(rec).open({})
    recorder.close()
    assert not (rec / "stream.json").exists()


@pytest.mark.parametrize("latency", [None, "low", (0.01,), ("x", "y"), (float("nan"), 0.01)])
def test_a_stream_that_reports_nonsense_says_nothing(latency):
    class Stream:
        pass
    stream = Stream()
    stream.latency = latency
    doc = realtime.stream_report(stream)
    assert doc["input_ms"] is None and doc["output_ms"] is None and doc["reported_ms"] is None
    assert doc["block_ms"] > 0.0
    json.dumps(doc, allow_nan=False)


def test_a_voice_channel_is_recorded_beside_the_guitar(tmp_path, consent):
    guitar, voice = _guitar_and_decoy()
    fake = fake_audio(guitar, devices=RIG, human_channel=0, inputs={1: voice})
    with patched_module("sounddevice", fake):
        realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(tmp_path / "rec"), status=False, max_minutes=0.05,
                      socket_path=short_socket_path("vch"), direct=True,
                      input_device="audio interface", input_channel=1, voice_channel=2)
    rec = tmp_path / "rec"
    assert _tone_level(rec / "voice.wav", 1000.0) > 10.0 * _tone_level(rec / "voice.wav", 220.0)
    assert _tone_level(rec / "human.wav", 220.0) > 10.0 * _tone_level(rec / "human.wav", 1000.0)
    doc = json.loads((rec / "session.json").read_text())["input"]
    assert doc["source"] == "direct" and doc["channel"] == 1 and doc["voice_channel"] == 2
    assert doc["thresholds_calibrated"] is False


def test_no_voice_channel_means_no_voice_file(tmp_path, consent):
    with patched_module("sounddevice", fake_audio(_guitar_and_decoy()[0])):
        realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(tmp_path / "rec"), status=False, max_minutes=0.03,
                      socket_path=short_socket_path("nov"))
    assert not (tmp_path / "rec" / "voice.wav").exists()
    assert json.loads((tmp_path / "rec" / "session.json").read_text())["input"]["source"] == "room"


# ------------------------------------------------------- what direct changes

def test_a_masker_for_a_direct_input_never_touches_a_frame():
    from plume.listen.features import FeatureExtractor
    x = np.random.default_rng(0).standard_normal(SR // 2).astype(np.float32) * 0.1
    frames = FeatureExtractor(sr=SR).process(x)
    masker = realtime.SelfMasker(enabled=False)
    out, flags = masker.mask(frames, [-6.0] * len(frames))
    assert not any(flags) and masker.masked_frames == 0
    assert all(a is b for a, b in zip(out, frames))


def _direct_take(*, stops_at=45.0, total=135.0):
    from test_bandentry import _take
    return _take(seconds=stops_at, room_s=total - stops_at)


def _play(tmp_path, *, human, name, **kw):
    fake = fake_audio(human, devices=RIG, rate_x=12.0, **kw.pop("fake", {}))
    with patched_module("sounddevice", fake):
        realtime.lead(prompt="midwest emo in C major", seed=0, tap_start=True,
                      role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(tmp_path / name), status=False,
                      socket_path=short_socket_path(name), max_minutes=0.7, **kw)
    return tmp_path / name


@needs_laptop_speed
def test_direct_hears_her_stop_with_the_band_in_and_fades(tmp_path, consent):
    # the room is loud, and the only input that can hear it is the voice mic
    rec = _play(tmp_path, human=_direct_take(), name="direct", direct=True,
                input_device="audio interface", input_channel=1, voice_channel=4,
                fake={"bleed_db": -6.0, "bleed_channels": (3,)})
    ending = json.loads((rec / "ending.json").read_text())
    assert ending["reason"] == realtime.END_SILENCE
    rows = rows_of(rec)
    assert all(r["playing_trusted"] for r in rows)
    in_band = [r for r in rows if r["entry"] != "drums"]
    assert in_band and any(r["playing"] for r in in_band)
    assert any(not r["playing"] for r in in_band)
    assert rows[-1]["fade_db"] < 0.0


def test_hum_on_an_idle_pickup_is_not_her_playing(tmp_path, consent):
    """A single coil hums whether or not anyone is touching it.  -57 dB is a guess at a
    bad case, and a real pickup is one of the things the checklist measures."""
    human = _direct_take()
    t = np.arange(human.size) / SR
    human = human + (2e-3 * np.sqrt(2) * np.sin(2 * np.pi * 60.0 * t)).astype(np.float32)
    rec = _play(tmp_path, human=human, name="hum", direct=True,
                input_device="audio interface", input_channel=1)
    assert json.loads((rec / "ending.json").read_text())["reason"] == realtime.END_SILENCE


def test_a_room_mic_with_the_band_in_says_its_playing_flag_cannot_be_believed(tmp_path, consent):
    rec = _play(tmp_path, human=_direct_take(), name="room",
                fake={"bleed_db": -6.0})
    rows = rows_of(rec)
    assert all(r["playing_trusted"] for r in rows if r["entry"] == "drums")
    in_band = [r for r in rows if r["entry"] != "drums"]
    assert in_band and not any(r["playing_trusted"] for r in in_band)


def test_direct_keeps_learning_her_notes_after_the_band_is_in(tmp_path, consent, monkeypatch):
    """Before, the ear stopped at the band's entrance because the mic was then mostly
    the band.  On a direct input it is only ever the player."""
    from plume import duet as duet_mod
    seen = []

    class Spy(duet_mod.DuetEngine):
        def render_bar(self, plan, *a, **kw):
            seen.append(tuple(self.heard_notes))
            return super().render_bar(plan, *a, **kw)
    monkeypatch.setattr(duet_mod, "DuetEngine", Spy)
    # C major until the band is well in, then the player moves somewhere it was not
    human = _direct_take()
    t = np.arange(human.size - 30 * SR) / SR
    moved = sum(np.sin(2 * np.pi * f * t) for f in (138.59, 174.61, 207.65, 277.18)) * 0.03
    human = human.copy()
    human[30 * SR:45 * SR] = moved[:15 * SR].astype(np.float32)
    _play(tmp_path, human=human, name="ear", direct=True,
          input_device="audio interface", input_channel=1)
    rows = rows_of(tmp_path / "ear")
    first_in = next(i for i, r in enumerate(rows) if r["entry"] != "drums")
    assert seen[first_in] and len(set(seen[first_in:])) > 1


# ------------------------------------------------------------- the JSON feed

BAR = 2.0


def _take_dir(path, *, source=None, playing=0.2, bleed=0.0, bars=16, stage="full"):
    path.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    human, band, rows = [], [], []
    for b in range(bars):
        t = np.arange(int(BAR * SR)) / SR
        m = 0.1 * (np.sin(2 * np.pi * 110.0 * t) + np.sin(2 * np.pi * 277.0 * t))
        amp = playing(b) if callable(playing) else playing
        g = amp * sum(np.sin(2 * np.pi * f * t) for f in (349.2, 440.0, 523.3)) * np.exp(-t * 3.0)
        human.append(g + bleed * m + rng.standard_normal(t.size) * 1e-5)
        band.append(m)
        rows.append({"bar": b, "t_s": b * BAR, "bpm": 120.0, "section": "verse", "entry": stage})
    sf.write(str(path / "human.wav"), np.concatenate(human).astype("float32"), SR)
    sf.write(str(path / "machine.wav"), np.concatenate(band).astype("float32"), SR)
    (path / "decisions.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if source is not None:
        (path / "session.json").write_text(json.dumps({"input": {"source": source}}))
    return path


ALL_YOU = ["you_db", "you_busy", "you_notes"]


def test_a_silent_player_on_a_room_mic_is_not_believed(tmp_path):
    """A silent player's mic still holds notes, because they are the band's, and the
    line has to say so, or a reader would report them as the player's."""
    doc = Listener(_take_dir(tmp_path, source="room", playing=0.0, bleed=0.5), every=4).poll()[0]
    assert doc["input"] == "room"
    assert doc["you_notes"], "the mic does hear pitches: that is the trap"
    assert doc["you_trusted"] == []
    assert "band" in doc["you_why"]


def test_a_silent_player_on_a_direct_input_reads_as_silent(tmp_path):
    played = lambda b: 0.0 if b >= 8 else 0.2  # noqa: E731
    docs = Listener(_take_dir(tmp_path, source="direct", playing=played), every=4).poll()
    assert docs[0]["input"] == "direct" and docs[0]["you_trusted"] == ALL_YOU
    assert docs[-1]["you_notes"] == [] and docs[-1]["you_db"] < -80.0
    assert docs[-1]["you_trusted"] == ALL_YOU
    assert "you:stopped" in [c for d in docs for c in d["changes"]]


def test_a_declared_direct_input_with_the_band_in_it_loses_its_trust(tmp_path):
    """Loopback on, or a mic plugged in by mistake.  It is measured, not assumed."""
    doc = Listener(_take_dir(tmp_path, source="direct", playing=0.0, bleed=0.5), every=4).poll()[0]
    assert doc["input"] == "direct" and doc["you_trusted"] == []
    assert "loopback" in doc["you_why"]


def test_room_mic_over_drums_alone_believes_only_the_notes(tmp_path):
    doc = Listener(_take_dir(tmp_path, source="room", stage="drums", bleed=0.1), every=4).poll()[0]
    assert doc["you_trusted"] == ["you_notes"]


def test_a_take_with_no_input_record_is_treated_as_a_room_mic(tmp_path):
    doc = Listener(_take_dir(tmp_path, bleed=0.1), every=4).poll()[0]
    assert doc["input"] == "room" and doc["you_trusted"] == []


# ---------------------------------------------------------------- plume inputs

def test_inputs_lists_devices_without_opening_anything(capsys, monkeypatch):
    from plume import cli
    fake = rig()
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "0")
    with patched_module("sounddevice", fake):
        assert cli.main(["inputs", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert {"name": "Audio Interface 4x4", "inputs": 4, "outputs": 4} in doc["devices"]
    assert all(d["inputs"] > 0 or d["outputs"] > 0 for d in doc["devices"])
    assert fake.recorded == [] and fake.opened == []


def test_inputs_meter_says_which_channel_has_the_guitar(capsys, monkeypatch):
    from plume import cli
    guitar, _ = _guitar_and_decoy(seconds=2.0)
    fake = fake_audio(guitar, devices=RIG, human_channel=2)
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")
    with patched_module("sounddevice", fake):
        assert cli.main(["inputs", "--meter", "audio interface", "--seconds", "1", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert fake.recorded == [{"device": 2, "channels": 4}]
    assert doc["device"] == "Audio Interface 4x4" and doc["loudest_channel"] == 3
    assert len(doc["channels"]) == 4
    assert doc["channels"][2]["peak_db"] > -20.0 > doc["channels"][0]["peak_db"]


def test_the_meter_needs_consent_because_it_opens_a_real_input(capsys, monkeypatch):
    from plume import cli
    fake = rig()
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "0")
    with patched_module("sounddevice", fake):
        assert cli.main(["inputs", "--meter", "audio interface"]) != 0
    assert fake.recorded == []


def test_lead_flags_reach_the_transport(monkeypatch):
    from plume import cli
    seen = {}

    def lead(**kw):
        seen.update(kw)
    monkeypatch.setattr(realtime, "lead", lead)
    monkeypatch.setattr(realtime, "check_device", lambda device=None: None)
    assert cli.main(["lead", "--tap-start", "lofi", "--input-device", "audio interface",
                     "--input-channel", "2", "--voice-channel", "3", "--direct"]) == 0
    assert (seen["input_device"], seen["input_channel"]) == ("audio interface", 2)
    assert seen["voice_channel"] == 3 and seen["direct"] is True


def test_bare_lead_passes_no_routing_at_all(monkeypatch):
    from plume import cli
    seen = {}

    def lead(**kw):
        seen.update(kw)
    monkeypatch.setattr(realtime, "lead", lead)
    monkeypatch.setattr(realtime, "check_device", lambda device=None: None)
    assert cli.main(["lead", "--tap-start", "lofi"]) == 0
    assert not {"input_device", "input_channel", "voice_channel", "direct"} & set(seen)


def test_jam_flags_reach_the_transport(monkeypatch):
    from plume import cli
    seen = {}

    def jam(**kw):
        seen.update(kw)
    monkeypatch.setattr(realtime, "jam", jam)
    monkeypatch.setattr(realtime, "check_device", lambda device=None: None)
    assert cli.main(["jam", "--input-device", "audio interface", "--input-channel", "2",
                     "--direct"]) == 0
    assert (seen["input_device"], seen["input_channel"]) == ("audio interface", 2)
    assert seen["direct"] is True and "voice_channel" not in seen


def test_bare_jam_passes_no_routing_at_all(monkeypatch):
    from plume import cli
    seen = {}

    def jam(**kw):
        seen.update(kw)
    monkeypatch.setattr(realtime, "jam", jam)
    monkeypatch.setattr(realtime, "check_device", lambda device=None: None)
    assert cli.main(["jam"]) == 0
    assert not {"input_device", "input_channel", "voice_channel", "direct"} & set(seen)


# ---------------------------------------------------- the transports' wiring

@pytest.fixture
def maskers(monkeypatch):
    """Every masker a transport builds, so a test can ask what it really did."""
    made = []

    class Spy(realtime.SelfMasker):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)
    monkeypatch.setattr(realtime, "SelfMasker", Spy)
    return made


def _bleeding_lead(tmp_path, name, **kw):
    """Long enough for the band to be sounding into the guitar's own channel."""
    return _play(tmp_path, human=_direct_take(), name=name, input_device="audio interface",
                 fake={"bleed_db": -6.0, "bleed_channels": (0,)}, **kw)


def test_lead_on_a_direct_input_masks_nothing_even_with_the_band_in_it(tmp_path, consent,
                                                                       maskers):
    _bleeding_lead(tmp_path, "m-direct", direct=True, input_channel=1)
    assert len(maskers) == 1 and maskers[0].frames_seen > 0
    assert maskers[0].masked_frames == 0


def test_lead_on_a_room_mic_still_masks(tmp_path, consent, maskers):
    """The same take without --direct, or the test above proves nothing."""
    _bleeding_lead(tmp_path, "m-room")
    assert len(maskers) == 1 and maskers[0].masked_frames > 0


def _short_jam(tmp_path, name, **kw):
    from plume import mind as mind_mod
    with patched_module("sounddevice", fake_audio(_guitar_and_decoy(seconds=30.0)[0],
                                                  devices=RIG, rate_x=12.0)):
        realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(),
                     record_dir=str(tmp_path / name), max_minutes=0.05,
                     input_device="audio interface", **kw)


def test_jam_turns_masking_off_only_on_a_direct_input(tmp_path, consent, maskers):
    _short_jam(tmp_path, "j-direct", direct=True, input_channel=1)
    _short_jam(tmp_path, "j-room", input_channel=1)
    assert [m._enabled for m in maskers] == [False, True]


def test_a_direct_input_gets_one_bar_longer_before_the_band(tmp_path, consent, monkeypatch):
    """The ear confirms the player a bar after a level would, and direct goes by level."""
    from plume import bandentry
    made = []

    class Spy(bandentry.BandEntry):
        def __init__(self, *a, **kw):
            made.append(kw.get("after_first_heard_bars"))
            super().__init__(*a, **kw)
    monkeypatch.setattr(bandentry, "BandEntry", Spy)
    silence = np.zeros(SR)
    for name, kw in (("e-direct", {"direct": True}), ("e-room", {})):
        with patched_module("sounddevice", fake_audio(silence, devices=RIG)):
            realtime.lead(prompt="lofi", seed=0, tap_start=True,
                          role=roles_mod.RoleParams(role="lead"), status=False,
                          socket_path=short_socket_path(name), max_minutes=0.01, **kw)
    assert made == [6, None]


def _knocks_then(sound, *, seconds=135.0):
    """The count, then nothing of the player's but ``sound``, looped to fill the take."""
    from test_bandentry import _take
    x = _take(seconds=8.0, room_s=seconds - 8.0)
    t = np.arange(x.size) / SR
    tail = sound(t).astype(np.float32)
    tail[:int(8.0 * SR)] = 0.0
    return x + tail


def test_a_quiet_ringing_tone_does_not_hold_off_the_fade_on_direct(tmp_path, consent):
    """Over drums a room mic trusts the ear: a tonal bar means the player.  On a direct input
    level decides, or an amp's buzz keeps a take nobody is playing open for ever."""
    buzz = lambda t: 10 ** (-50 / 20) * np.sqrt(2) * np.sin(2 * np.pi * 220.0 * t)  # noqa: E731
    rec = _play(tmp_path, human=_knocks_then(buzz), name="buzz", direct=True,
                input_device="audio interface", input_channel=1)
    assert json.loads((rec / "ending.json").read_text())["reason"] == realtime.END_SILENCE
    # On time: the fake device loops, so a stuck take would still end, much later.
    assert rows_of(rec)[-1]["t_s"] < 120.0


# ------------------------------------------------- a pickup's noise is not notes

def _hum_over_band(path, level_db, *, hum_hz=60.0, seconds=32.0):
    """Hands off a direct input: mains hum and hiss, with the band loud in machine.wav."""
    path.mkdir(exist_ok=True)
    rng = np.random.default_rng(2)
    t = np.arange(int(seconds * SR)) / SR
    amp = 10 ** (level_db / 20)
    human = (amp * 0.8 * np.sqrt(2) * np.sin(2 * np.pi * hum_hz * t)
             + amp * 0.6 * rng.standard_normal(t.size))
    band = 0.1 * (np.sin(2 * np.pi * 110.0 * t) + np.sin(2 * np.pi * 277.0 * t))
    sf.write(str(path / "human.wav"), human.astype("float32"), SR)
    sf.write(str(path / "machine.wav"), band.astype("float32"), SR)
    rows = [{"bar": b, "t_s": b * BAR, "bpm": 120.0, "section": "verse", "entry": "full"}
            for b in range(int(seconds / BAR))]
    (path / "decisions.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "session.json").write_text(json.dumps({"input": {"source": "direct"}}))
    return path


@pytest.mark.parametrize("level_db", [-52.0, -48.0, -46.0])
def test_hum_below_the_silence_floor_names_no_notes(tmp_path, level_db):
    doc = Listener(_hum_over_band(tmp_path, level_db), every=4).poll()[0]
    assert doc["you_db"] < -45.0
    assert doc["you_notes"] == []


@pytest.mark.parametrize("level_db", [-52.0, -48.0, -46.0])
def test_a_clear_buzz_below_the_silence_floor_names_no_notes(tmp_path, level_db):
    """A tone clears the share test, so only the floor can keep this one out."""
    doc = Listener(_hum_over_band(tmp_path, level_db, hum_hz=220.0), every=4).poll()[0]
    assert doc["you_db"] < -45.0
    assert doc["you_notes"] == []


def test_hiss_above_the_floor_names_no_notes(tmp_path):
    """Chroma sums to 1, so flat noise puts 1/12 on every note: that is not a pitch."""
    path = tmp_path / "hiss"
    _hum_over_band(path, -30.0)
    rng = np.random.default_rng(3)
    y, _ = sf.read(str(path / "human.wav"), dtype="float32")
    sf.write(str(path / "human.wav"),
             (rng.standard_normal(y.size) * 10 ** (-30 / 20)).astype("float32"), SR)
    assert Listener(path, every=4).poll()[0]["you_notes"] == []
    assert all(d["you_notes"] == [] for d in Listener(path, every=1).poll())


def test_quiet_playing_above_the_floor_still_names_its_notes(tmp_path):
    doc = Listener(_take_dir(tmp_path, source="direct", playing=0.02), every=4).poll()[0]
    assert doc["you_db"] > -45.0
    assert set(doc["you_notes"]) == {"F", "A", "C"}


def test_a_chord_change_each_bar_still_names_the_notes(tmp_path):
    """Four bars of moving chords spread the average: that must not read as noise."""
    path = tmp_path / "moving"
    path.mkdir()
    chords = [(261.6, 329.6, 392.0), (220.0, 261.6, 329.6), (174.6, 220.0, 261.6),
              (196.0, 246.9, 293.7)]
    t = np.arange(int(BAR * SR)) / SR
    human = np.concatenate([0.1 * sum(np.sin(2 * np.pi * f * t) for f in c) * np.exp(-t * 2)
                            for c in chords])
    sf.write(str(path / "human.wav"), human.astype("float32"), SR)
    sf.write(str(path / "machine.wav"), np.zeros(human.size, dtype="float32"), SR)
    rows = [{"bar": b, "t_s": b * BAR, "bpm": 120.0, "section": "verse", "entry": "full"}
            for b in range(4)]
    (path / "decisions.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert Listener(path, every=4).poll()[0]["you_notes"]


def test_direct_reports_all_of_what_it_hears_and_a_room_mic_does_not(tmp_path):
    """Subtracting the band's match off a direct input would only ever remove the player."""
    d = Listener(_take_dir(tmp_path / "d", source="direct", playing=0.02, bleed=0.3),
                 every=4).poll()[0]
    assert d["you_db"] == d["mic_db"]
    r = Listener(_take_dir(tmp_path / "r", source="room", playing=0.02, bleed=0.3),
                 every=4).poll()[0]
    assert r["you_db"] < r["mic_db"]


# ------------------------------------------------------------- the same directory

def test_a_new_take_in_a_used_directory_clears_the_old_ending(tmp_path):
    """Otherwise the last take's ending reads as this one's, and a watch exits on it."""
    (tmp_path / realtime.REC_ENDING).write_text('{"reason": "the old one"}')
    rec = realtime.SessionRecorder(tmp_path).open({})
    assert not (tmp_path / realtime.REC_ENDING).exists()
    rec.finish({"reason": "this one"})
    assert json.loads((tmp_path / realtime.REC_ENDING).read_text())["reason"] == "this one"


@pytest.mark.parametrize("doc", ["DIRECT-INPUT-TEST.md", "PLAYING.md"])
def test_the_docs_record_where_git_will_not_pick_it_up(doc):
    """Only ``takes*/`` is ignored, and session.json holds the real device name."""
    import re
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "docs" / doc).read_text()
    named = re.findall(r"--record (\S+)", text) + re.findall(r"plume watch (\S+)", text)
    assert named and all(n.startswith("takes") for n in named), named


@pytest.mark.parametrize("seconds", ["0", "-1", "nan", "inf", "86400"])
def test_the_meter_refuses_a_length_it_could_not_honestly_hold(capsys, monkeypatch, seconds):
    """An open mic with no session around it: no cap, no registry, no plume stop."""
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only
    fake = rig()
    with patched_module("sounddevice", fake):
        assert cli.main(["inputs", "--meter", "audio interface", "--seconds", seconds]) != 0
    assert "--seconds" in capsys.readouterr().err
    assert fake.recorded == []


def test_jam_monitor_mixes_in_the_chosen_channel(tmp_path, consent):
    """Nothing else in the band sits on these frequencies."""
    t = np.arange(4 * 48000) / 48000
    guitar = (0.4 * np.sin(2 * np.pi * 3137.0 * t)).astype(np.float32)
    decoy = (0.4 * np.sin(2 * np.pi * 4409.0 * t)).astype(np.float32)
    fake = fake_audio(guitar, devices=RIG, human_channel=1, inputs={0: decoy})
    with patched_module("sounddevice", fake):
        realtime.jam(style="lofi", params=mind_mod.MindParams(), seed=0,
                     max_minutes=0.03, monitor=True, socket_path=short_socket_path("jmon"),
                     input_device="audio interface", input_channel=2)
    out = fake.stream.output.mean(axis=1)[:2 * 48000]
    t = np.arange(out.size) / 48000

    def level(hz):
        return abs(np.dot(out, np.exp(-2j * np.pi * hz * t))) / out.size
    assert level(3137.0) > 0.1
    assert level(4409.0) < 0.1 * level(3137.0)


def test_a_jam_that_means_lead_takes_its_input_route_along(monkeypatch):
    """The dispatch keeps every route flag: losing one would quietly read the wrong input."""
    seen = {}
    monkeypatch.setattr(realtime, "lead", lambda **kw: seen.update(kw))
    realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(),
                 role=roles_mod.RoleParams(role="lead"), input_device="audio interface",
                 input_channel=3, direct=True, monitor=True)
    assert (seen["input_device"], seen["input_channel"], seen["direct"],
            seen["monitor"]) == ("audio interface", 3, True, True)


# ------------------------------------------------------------------ another output

def test_output_device_changes_the_output_and_only_the_output():
    route = resolve_input(rig(), device="audio interface", input_channel=2, direct=True, output_device="laptop speakers")
    assert route.device == (2, 1) and route.guitar == 1 and route.source == "direct"
    assert route.device_name == "Audio Interface 4x4" and route.output_name == "Laptop Speakers"


def test_output_device_with_an_input_device_or_none():
    assert resolve_input(rig(), input_device="laptop mic", output_device="laptop speakers").device == (0, 1)
    assert resolve_input(rig(), output_device="laptop speakers").device == (None, 1)


def test_part_of_an_output_name_is_not_enough_so_one_that_went_cannot_become_its_neighbour():
    """Room Speakers has gone and Room Speakers 2 is there.  That is a refusal, never a swap."""
    table = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4},
             {"name": "Room Speakers 2", "max_input_channels": 0, "max_output_channels": 2}]
    with pytest.raises(RealtimeError, match="no audio output matching"):
        resolve_input(fake_audio(None, devices=table), device="audio interface", output_device="Room Speakers")
    assert resolve_input(fake_audio(None, devices=table), output_device="room speakers 2").device == (None, 1)


def test_no_output_asked_for_is_as_it_was():
    assert resolve_input(rig()).device is None
    assert resolve_input(rig(), device="audio interface").device == (2, 2)
    assert "output" not in resolve_input(rig(), device="audio interface").describe()


@pytest.mark.parametrize("name,words", [("no such thing", "no audio output matching"), ("laptop mic", "has no outputs")])
def test_output_device_is_refused_and_never_swapped(name, words):
    with pytest.raises(RealtimeError, match=words):
        resolve_input(rig(), device="audio interface", output_device=name)


def test_only_the_lead_command_takes_the_output_flag():
    args = cli._build_parser().parse_args(["lead", "--output-device", "speakers"])
    assert args.output_device == "speakers"
    assert cli._route_kwargs(args) == {"output_device": "speakers"}
    assert "output_device" not in cli._route_kwargs(cli._build_parser().parse_args(["lead"]))
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(["jam", "--output-device", "speakers"])


def _two_device_take(tmp_path, *, name="rec", **kw):
    device = fake_audio(kw.pop("human", None), devices=RIG, latency=(0.012, 0.150))
    with patched_module("sounddevice", device):
        realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"), status=False,
                      max_minutes=0.01, tap_start=True, direct=True, record_dir=str(tmp_path / name),
                      socket_path=short_socket_path("twodev"), device="audio interface",
                      input_channel=1, output_device="laptop speakers", **kw)
    return device, tmp_path / name


def test_the_stream_opens_on_the_pair_and_the_take_says_so(tmp_path, consent):
    device, rec = _two_device_take(tmp_path)
    assert device.opened[0]["device"] == (2, 1)
    manifest = json.loads((rec / "session.json").read_text())
    assert manifest["input"]["output"] == "Laptop Speakers" and manifest["input"]["device"] == "Audio Interface 4x4"


def test_stream_json_holds_both_devices_numbers_and_together_lines_up_on_them(tmp_path, consent):
    """As far as a fake can: it reports what it is told.  Real gear may add a delay it does not report."""
    _, rec = _two_device_take(tmp_path)
    doc = json.loads((rec / "stream.json").read_text())
    assert doc["input_ms"] == 12.0 and doc["output_ms"] == 150.0 and doc["reported_ms"] == 162.0
    assert doc["calibrated_ms"] is None
    assert record.playing_delay_frames(rec, SR) == round(0.162 * SR)


def test_a_calibration_of_other_devices_is_not_used_for_this_pair(tmp_path, consent):
    from plume import calibrate
    calibrate.save_roundtrip("Audio Interface 4x4", "Audio Interface 4x4", SR,
                             {"roundtrip_ms": 41.5, "spread_ms": 0.5, "heard": 8, "clicks": 8})
    _, rec = _two_device_take(tmp_path)
    assert json.loads((rec / "stream.json").read_text())["calibrated_ms"] is None


def test_the_monitor_mixes_into_the_chosen_output(tmp_path, consent):
    """The opt-in is the existing Hear me mix, with the same stream on the other output."""
    guitar = (0.2 * np.sin(2 * np.pi * 220.0 * np.arange(SR * 2) / SR)).astype(np.float32)
    device, _ = _two_device_take(tmp_path, human=guitar, monitor=True)
    assert device.opened[0]["device"] == (2, 1)
    assert np.abs(device.stream.output).max() > 0.1


class _RefusingAudio:
    """A device table whose pair will not open, as a real pair of clocks may not."""

    def __init__(self):
        self._inner = rig()

    def query_devices(self, *a, **kw):
        return self._inner.query_devices(*a, **kw)

    def Stream(self, **kw):  # noqa: N802 - sd's name
        raise RuntimeError("Error opening Stream: Illegal combination of I/O devices [PaErrorCode -9993]")


def test_a_pair_that_will_not_open_is_a_sentence_and_not_a_traceback(tmp_path, consent, capsys):
    with patched_module("sounddevice", _RefusingAudio()):
        code = cli.main(["lead", "--tap-start", "--detect", f"--record={tmp_path / 'rec'}/", "--device=audio interface",
                         "--direct", "--output-device=laptop speakers", f"--socket={short_socket_path('refuse')}"])
    err = capsys.readouterr().err
    assert code != 0 and realtime.OPEN_TOGETHER_FAILED in err and "Traceback" not in err
    assert realtime.list_sessions(prune=False) == [], "nothing left behind claiming to be a take"


class _WontStart:
    """The pair opens, and then will not run, which PortAudio reports from a different call."""

    def __init__(self):
        self._inner = rig()
        self.stream = None

    def query_devices(self, *a, **kw):
        return self._inner.query_devices(*a, **kw)

    def Stream(self, **kw):  # noqa: N802 - sd's name
        self.stream = self._inner.Stream(**kw)
        self.stream.start = self._refuse
        return self.stream

    @staticmethod
    def _refuse():
        raise RuntimeError("Error starting stream: Unanticipated host error [PaErrorCode -9999]")


def test_a_pair_that_opens_and_will_not_start_is_the_same_sentence(tmp_path, consent, capsys):
    audio = _WontStart()
    with patched_module("sounddevice", audio):
        code = cli.main(["lead", "--tap-start", "--detect", f"--record={tmp_path / 'rec'}/", "--device=audio interface",
                         "--direct", "--output-device=laptop speakers", f"--socket={short_socket_path('nostart')}"])
    err = capsys.readouterr().err
    assert code != 0 and realtime.OPEN_TOGETHER_FAILED in err and "Traceback" not in err
    assert audio.stream.closed is True, "the device is let go"
    assert realtime.list_sessions(prune=False) == [], "nothing left behind claiming to be a take"


def test_one_device_failing_to_start_is_as_it_was(tmp_path, consent):
    audio = _WontStart()
    with patched_module("sounddevice", audio):
        with pytest.raises(RuntimeError, match="Error starting stream"):
            realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"), status=False,
                          max_minutes=0.01, tap_start=True, direct=True, record_dir=str(tmp_path / "rec"),
                          socket_path=short_socket_path("onestart"), device="audio interface", input_channel=1)


def test_one_device_refusing_to_open_is_as_it_was(tmp_path, consent):
    with patched_module("sounddevice", _RefusingAudio()):
        with pytest.raises(RuntimeError, match="Illegal combination"):
            realtime.lead(prompt="lofi", seed=0, role=roles_mod.RoleParams(role="lead"), status=False,
                          max_minutes=0.01, tap_start=True, direct=True, record_dir=str(tmp_path / "rec"),
                          socket_path=short_socket_path("oneref"), device="audio interface", input_channel=1)
    assert "plume-cues" not in [th.name for th in threading.enumerate()], "the listener outlived the refusal"
