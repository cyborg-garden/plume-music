"""The band sent as MIDI (plume.midiout), with a fake mido: no real port, no sound."""
from __future__ import annotations

import random
import statistics

import numpy as np
import pytest
from conftest import fake_audio, patched_module, short_socket_path

from plume import cli, midiout, realtime
from plume import mind as mind_mod
from plume.midiout import (DRUM_CHANNEL, BandMidi, Schedule, StreamClock, bar_notes,
                           resolve_output)

SR = 48000
BLOCK = 256


class Msg:
    def __init__(self, type, **kw):  # noqa: A002 - mido's own name
        self.type = type
        self.__dict__.update(kw)

    def as_tuple(self):
        if self.type == "control_change":
            return ("cc", self.channel, self.control, self.value)
        return (self.type[5:], self.channel, self.note, self.velocity)


class FakeOut:
    def __init__(self, name, virtual):
        self.name, self.virtual = name, virtual
        self.sent: list[tuple[float, tuple]] = []
        self.closed = False
        self.now = lambda: 0.0

    def send(self, msg):
        assert not self.closed
        self.sent.append((self.now(), msg.as_tuple()))

    def close(self):
        self.closed = True


class FakeMidoOut:
    """A stand-in mido with output ports.  Opening one makes no port anywhere."""
    Message = Msg

    def __init__(self, names=("IAC Driver Bus 1",), *, no_virtual=False):
        self.names = list(names)
        self.no_virtual = no_virtual
        self.ports: list[FakeOut] = []

    def get_output_names(self):
        return list(self.names)

    def get_input_names(self):
        return []

    def open_output(self, name, virtual=False):
        if virtual and self.no_virtual:
            raise NotImplementedError("virtual ports are not supported")
        if not virtual and name not in self.names:
            raise OSError(f"cannot open {name}")
        port = FakeOut(name, virtual)
        self.ports.append(port)
        return port


#: One bar at 120 bpm, as DuetEngine.last_events has it.
BAR = [
    {"role": "kick", "onset_s": 0.0, "dur_s": 0.3, "midi": 33, "velocity": 0.9, "level": 1.0},
    {"role": "hat", "onset_s": 0.25, "dur_s": 0.05, "midi": None, "velocity": 0.5,
     "level": 1.0},
    {"role": "bass", "onset_s": 0.0, "dur_s": 0.45, "midi": 45, "velocity": 0.8,
     "level": 0.5},
    {"role": "pad", "onset_s": 0.0, "dur_s": 2.0, "midi": 60, "velocity": 0.6, "level": 1.0},
    {"role": "crackle", "onset_s": 0.0, "dur_s": 2.0, "midi": None, "velocity": 0.7,
     "level": 1.0},
    {"role": "announce", "voice_role": "pad", "onset_s": 1.5, "dur_s": 0.5, "midi": 72,
     "freq_hz": 523.25},
]


def test_a_known_bar_becomes_notes_on_each_parts_channel():
    notes = sorted(bar_notes(BAR, 1000, SR))
    assert notes == [
        (1000, 1000 + round(0.05 * SR), DRUM_CHANNEL, 36, 114),   # kick: GM, not its pitch
        (1000, 1000 + round(0.45 * SR), 0, 45, 51),               # bass: level lowers it
        (1000, 1000 + round(2.0 * SR), 1, 60, 76),                # pad
        (1000 + 12000, 1000 + 12000 + round(0.05 * SR), DRUM_CHANNEL, 42, 64),   # hat
        (1000 + 72000, 1000 + 72000 + round(0.5 * SR), 1, 72, 102),   # the cue, as the pad
    ]                                                             # crackle: not sent


def test_unknown_parts_go_to_the_spare_channel_and_unknown_drums_nowhere():
    notes = bar_notes([{"role": "organ", "onset_s": 0.0, "dur_s": 1.0, "midi": 50},
                       {"role": "cowbell", "onset_s": 0.0, "dur_s": 1.0, "midi": None}],
                      0, SR)
    assert [(n[2], n[3]) for n in notes] == [(midiout.OTHER_CHANNEL, 50)]


def test_due_messages_come_in_time_order_offs_first_on_a_shared_sample():
    s = Schedule()
    s.add([(0, 100, 0, 40, 90), (100, 200, 0, 40, 80), (50, 60, 9, 36, 100)])
    out = s.due(1_000)
    assert out == [("on", 0, 40, 90), ("on", 9, 36, 100), ("off", 9, 36, 0),
                   ("off", 0, 40, 0), ("on", 0, 40, 80), ("off", 0, 40, 0)]


def test_a_note_struck_again_while_sounding_is_ended_once():
    s = Schedule()
    s.add([(0, 300, 1, 60, 90), (100, 200, 1, 60, 70)])
    out = s.due(1_000)
    assert out == [("on", 1, 60, 90), ("off", 1, 60, 0), ("on", 1, 60, 70),
                   ("off", 1, 60, 0)]
    assert s.sounding == 0


def paired(msgs) -> bool:
    held: dict = {}
    for kind, ch, note, _v in msgs:
        if kind == "on":
            if held.get((ch, note)):
                return False
            held[(ch, note)] = True
        elif kind == "off":
            if not held.get((ch, note)):
                return False
            held[(ch, note)] = False
    return not any(held.values())


def test_every_on_has_one_off_over_random_bars():
    rng = random.Random(7)
    s = Schedule()
    sent = []
    for bar in range(40):
        notes = []
        for _ in range(rng.randint(0, 12)):
            on = bar * 1000 + rng.randint(0, 999)
            notes.append((on, on + rng.randint(1, 3000), rng.choice([0, 1, 9]),
                          rng.choice([36, 38, 60, 62]), rng.randint(1, 127)))
        s.add(notes, key=bar)
        if rng.random() < 0.3:
            s.move(bar, rng.randint(-50, 50))
        sent += s.due(bar * 1000 + rng.randint(0, 999))
    sent += [m for m in s.release() if m[0] != "cc"]
    assert paired(sent)


def test_release_ends_what_sounds_and_forgets_what_waits():
    s = Schedule()
    s.add([(0, 10_000, 0, 45, 90), (0, 10_000, 9, 36, 100), (5_000, 6_000, 2, 70, 90)])
    s.due(100)
    out = s.release()
    assert out == [("off", 0, 45, 0), ("off", 9, 36, 0),
                   ("cc", 0, 64, 0), ("cc", 0, 123, 0), ("cc", 9, 64, 0), ("cc", 9, 123, 0)]
    assert len(s) == 0 and s.due(1e9) == []


def test_a_moved_bar_moves_again_by_its_new_place():
    s = Schedule()
    s.add([(1000, 2000, 0, 40, 90)], key=1000)
    s.move(1000, 100, new_key=1100)
    s.move(1100, -50, new_key=1050)
    assert s.next_sample() == 1050
    assert s.due(1049) == [] and s.due(1050) == [("on", 0, 40, 90)]


class Rig:
    """A BandMidi on a fake port and a fake clock, with a playhead the test moves."""

    def __init__(self, *, only=False, offset_s=0.0, latency_s=0.0):
        self.mido = FakeMidoOut()
        self.t = 0.0
        self.playhead = 0
        self.band = BandMidi("IAC Driver Bus 1", only=only, offset_s=offset_s, sr=SR,
                             block=BLOCK, clock=lambda: self.t)
        with patched_module("mido", self.mido):
            self.band.open()
        self.port = self.mido.ports[0]
        self.port.now = lambda: self.t
        self.band.start(lambda: self.playhead, latency_s=latency_s, thread=False)

    def run(self, until_s, *, cb_late=lambda: 0.0, wake=lambda: 0.001):
        """Callbacks every block (``cb_late`` after their ideal time) and the sender
        waking every ``wake()`` seconds, in time order."""
        k = self.playhead // BLOCK
        next_cb = k * BLOCK / SR + cb_late()
        next_wake = self.t
        while self.t < until_s:
            if next_cb <= next_wake:
                self.t = next_cb
                self.playhead += BLOCK
                k += 1
                next_cb = max(self.t, k * BLOCK / SR + cb_late())
            else:
                self.t = next_wake
                self.band.pump()
                next_wake = self.t + wake()

    def sent(self):
        return [m for _t, m in self.port.sent]


def test_a_bar_is_sent_in_order_and_on_time():
    rig = Rig()
    rig.band.bar(SR // 2, BAR)          # the bar starts half a second in
    rig.run(3.0)
    sent = rig.sent()
    assert sent[:3] == [("on", 9, 36, 114), ("on", 0, 45, 51), ("on", 1, 60, 76)]
    assert paired(sent)
    times = {m: t for t, m in rig.port.sent}
    assert times[("on", 9, 36, 114)] == pytest.approx(0.5, abs=0.0015)
    assert times[("on", 9, 42, 64)] == pytest.approx(0.75, abs=0.0015)


def test_output_latency_and_the_offset_move_the_send_time():
    for latency, offset, want in ((0.02, 0.0, 0.52), (0.02, -0.015, 0.505),
                                  (0.0, 0.01, 0.51)):
        rig = Rig(latency_s=latency, offset_s=offset)
        rig.band.bar(SR // 2, BAR[:1])
        rig.run(1.0)
        assert rig.port.sent[0][0] == pytest.approx(want, abs=0.0015)


def test_stop_ends_every_sounding_note_and_says_all_notes_off():
    rig = Rig()
    rig.band.bar(0, BAR)
    rig.band.bar(2 * SR, BAR)           # queued, never reached
    rig.run(0.6)                        # the pad still sounds; the bass has ended
    rig.band.close()
    sent = rig.sent()
    assert paired([m for m in sent if m[0] != "cc"])
    assert ("cc", 1, 123, 0) in sent and ("cc", 9, 123, 0) in sent and ("cc", 0, 64, 0) in sent
    assert sent.count(("on", 9, 36, 114)) == 1       # the second bar never started
    assert rig.port.closed
    rig.band.close()                                 # twice is harmless


def test_midi_only_silences_the_band_and_otherwise_hands_back_the_same_audio():
    audio = np.ones((64, 2), dtype=np.float32)
    assert Rig(only=True).band.heard(audio).max() == 0.0
    assert Rig().band.heard(audio) is audio


def test_a_port_that_fails_mid_take_costs_the_notes_not_the_take():
    rig = Rig()
    rig.port.send = lambda _m: (_ for _ in ()).throw(OSError("unplugged"))
    rig.band.bar(0, BAR)
    rig.run(0.2)
    rig.band.close()
    assert rig.band.errors and "unplugged" in rig.band.errors[0]


def test_send_jitter_against_a_fake_clock():
    """Callbacks up to 1 ms late, one in fifty 4 ms late; the sender wakes every
    1 ms +- 0.3 ms.  The send error is what the stream clock gets wrong plus the wake
    granularity."""
    rng = random.Random(3)
    rig = Rig(latency_s=0.01)
    bars = 8
    for b in range(bars):
        rig.band.bar(b * 2 * SR + SR // 4, BAR)
    rig.run(bars * 2.0 + 1.0,
            cb_late=lambda: 0.004 if rng.random() < 0.02 else rng.uniform(0.0, 0.001),
            wake=lambda: 0.001 + rng.uniform(-0.0003, 0.0003))
    ideal = sorted(on / SR + 0.01 for b in range(bars)
                   for on, *_ in bar_notes(BAR, b * 2 * SR + SR // 4, SR))
    ons = sorted(t for t, m in rig.port.sent if m[0] == "on")
    assert len(ons) == len(ideal)
    err_ms = [1000.0 * (t - i) for t, i in zip(ons, ideal)]
    print(f"midi out send error: mean {statistics.mean(err_ms):.2f} ms, "
          f"max {max(err_ms):.2f} ms, min {min(err_ms):.2f} ms")
    assert min(err_ms) > -0.5          # never early by more than the clock guess
    assert max(err_ms) < 2.0
    assert statistics.mean(err_ms) < 1.0


def test_the_stream_clock_takes_the_earliest_sighting():
    c = StreamClock(SR, BLOCK)
    c.observe(0, 0.0)
    assert not c.ready
    c.observe(BLOCK, 0.0102)      # seen 4.9 ms after that callback
    c.observe(2 * BLOCK, 0.0109)  # seen 0.1 ms after
    assert c.time_of(BLOCK) == pytest.approx(0.0109, abs=1e-9)


class TestPorts:
    def test_virtual_needs_no_listing(self):
        with patched_module("mido", None):
            assert resolve_output("virtual") == ("plume", True)

    def test_part_of_a_name_finds_the_port(self):
        with patched_module("mido", FakeMidoOut(["IAC Driver Bus 1", "Other"])):
            assert resolve_output("iac") == ("IAC Driver Bus 1", False)

    @pytest.mark.parametrize("names,want,says", [
        (["IAC Driver Bus 1"], "nope", "no MIDI output port matching nope"),
        (["plume A", "plume B"], "plume", "more than one"),
    ])
    def test_a_bad_name_says_what_there_is(self, names, want, says):
        with patched_module("mido", FakeMidoOut(names)):
            with pytest.raises(midiout.MidiError, match=says):
                resolve_output(want)

    def test_no_virtual_ports_says_what_to_do(self):
        band = BandMidi("plume", virtual=True)
        with patched_module("mido", FakeMidoOut(no_virtual=True)):
            with pytest.raises(midiout.MidiError, match="loopMIDI"):
                band.open()

    def test_a_virtual_port_is_asked_for_by_name(self):
        mido = FakeMidoOut()
        with patched_module("mido", mido):
            BandMidi("plume", virtual=True).open()
        assert (mido.ports[0].name, mido.ports[0].virtual) == ("plume", True)


@pytest.mark.usefixtures("fake_device_consent")
class TestLive:
    def test_off_never_touches_the_module(self, plume_registry):
        """With no --midi-out a take runs as it always has: nothing imports midiout."""
        sd = fake_audio(None, rate_x=12.0)
        with patched_module("sounddevice", sd), patched_module("plume.midiout", None):
            realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(),
                         socket_path=short_socket_path("mo-off"), max_minutes=0.02)
        assert sd.streams

    def test_only_without_a_port_is_refused_before_a_stream_opens(self, plume_registry):
        sd = fake_audio(None)
        with patched_module("sounddevice", sd):
            with pytest.raises(realtime.RealtimeError, match="--midi-out"):
                realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(),
                             socket_path=short_socket_path("mo-only"), midi_only=True)
        assert not sd.streams

    def test_a_bad_port_is_refused_before_a_stream_opens(self, plume_registry):
        sd = fake_audio(None)
        with patched_module("sounddevice", sd), patched_module("mido", FakeMidoOut()):
            with pytest.raises(realtime.RealtimeError, match="nope"):
                realtime.lead(prompt="lofi", seed=0, status=False,
                              socket_path=short_socket_path("mo-bad"), midi_out="nope")
        assert not sd.streams

    @pytest.mark.parametrize("only", [False, True], ids=["both", "midi-only"])
    def test_a_lead_take_sends_its_band_and_ends_every_note(self, only, plume_registry):
        """The machine plays from its first bar, so notes go out with no input at all.
        With --midi-only the device hears none of the band."""
        sd, mido = fake_audio(None, rate_x=6.0), FakeMidoOut()
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            realtime.lead(prompt="lofi at 120 bpm", seed=0, status=False, max_minutes=0.04,
                          socket_path=short_socket_path("mo-lead"), midi_out="iac",
                          midi_only=only)
        port = mido.ports[0]
        sent = [m for _t, m in port.sent]
        assert port.closed and port.name == "IAC Driver Bus 1"
        assert sum(m[0] == "on" for m in sent) >= 8, sent[:5]
        assert {m[1] for m in sent if m[0] == "on"} >= {0, DRUM_CHANNEL}
        assert paired([m for m in sent if m[0] != "cc"])
        assert ("cc", DRUM_CHANNEL, 123, 0) in sent
        assert sd.stream.sounded() is not only

    def test_jam_hands_the_port_to_lead(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(realtime, "lead", lambda **kw: seen.update(kw))
        realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(), lead_in="lofi",
                     midi_out="virtual", midi_only=True, midi_out_offset_s=-0.01)
        assert (seen["midi_out"], seen["midi_only"], seen["midi_out_offset_s"]) == (
            "virtual", True, -0.01)
        seen.clear()
        realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(), lead_in="lofi")
        assert "midi_out" not in seen


class TestCli:
    @pytest.mark.parametrize("verb,extra", [("jam", []), ("lead", ["lofi at 90 bpm"])])
    def test_flags_reach_the_take_only_when_given(self, verb, extra, monkeypatch):
        seen = []
        monkeypatch.setattr(realtime, "check_device", lambda *_a, **_k: None)
        monkeypatch.setattr(realtime, verb, lambda **kw: seen.append(kw))
        assert cli.main([verb, *extra, "--midi-out", "virtual", "--midi-only",
                         "--midi-out-offset", "-12"]) == 0
        assert cli.main([verb, *extra]) == 0
        assert seen[0]["midi_out"] == "virtual" and seen[0]["midi_only"] is True
        assert seen[0]["midi_out_offset_s"] == pytest.approx(-0.012)
        assert not {"midi_out", "midi_only", "midi_out_offset_s"} & set(seen[1])

    @pytest.mark.parametrize("flags", [["--midi-only"], ["--midi-out-offset", "5"]])
    def test_options_with_no_port_are_refused(self, flags, monkeypatch, capsys):
        monkeypatch.setattr(realtime, "check_device", lambda *_a, **_k: None)
        monkeypatch.setattr(realtime, "jam", lambda **_k: None)
        assert cli.main(["jam", *flags]) != 0
        assert "--midi-out" in capsys.readouterr().err
