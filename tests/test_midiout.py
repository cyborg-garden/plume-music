"""The band sent as MIDI (plume.midiout), with a fake mido: no real port, no sound."""
from __future__ import annotations

import os
import random
import statistics
import threading
import time

import numpy as np
import pytest
from conftest import fake_audio, patched_module, short_socket_path

from plume import cli, midiout, realtime
from plume import mind as mind_mod
from plume.midiout import (DRUM_CHANNEL, BandFile, BandMidi, Schedule, StreamClock, bar_notes,
                           resolve_output)

SR = 48000
BLOCK = 256


class Msg:
    def __init__(self, type, **kw):  # noqa: A002 - mido's own name
        self.type = type
        self.__dict__.update(kw)

    def as_tuple(self):
        return as_tuple(self)


def as_tuple(msg):
    """A message as ``(kind, channel, a, b)``: the fake's, or a real mido one."""
    if msg.type == "control_change":
        return ("cc", msg.channel, msg.control, msg.value)
    return (msg.type[5:], msg.channel, msg.note, msg.velocity)


class FakeOut:
    def __init__(self, name, virtual):
        self.name, self.virtual = name, virtual
        self.sent: list[tuple[float, tuple]] = []
        self.closed = False
        self.now = lambda: 0.0

    def send(self, msg):
        assert not self.closed
        self.sent.append((self.now(), as_tuple(msg)))

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


def test_an_earlier_notes_off_never_cuts_a_restruck_note_short():
    s = Schedule()
    s.add([(0, 1000, 0, 60, 100)], key="A")
    s.add([(500, 2000, 0, 60, 90)], key="B")
    assert s.due(1500) == [("on", 0, 60, 100), ("off", 0, 60, 0), ("on", 0, 60, 90)]
    assert s.due(2000) == [("off", 0, 60, 0)]


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

    def __init__(self, *, only=False, offset_s=0.0, latency_s=0.0, file=None):
        self.mido = FakeMidoOut()
        self.t = 0.0
        self.playhead = 0
        self.band = BandMidi("IAC Driver Bus 1", only=only, offset_s=offset_s, sr=SR,
                             block=BLOCK, clock=lambda: self.t, file=file)
        with patched_module("mido", self.mido):
            self.band.open()
        self.port = self.mido.ports[0]
        self.port.now = lambda: self.t
        self.band.start(lambda: self.playhead, latency_s=latency_s, thread=False)
        #: what start sent to clear the channels, before any note
        self.cleared = [m for _t, m in self.port.sent]
        self.port.sent.clear()

    def run(self, until_s, *, cb_late=lambda: 0.0, wake=lambda: 0.001, burst=1,
            prefill=0):
        """The device's callbacks and the sender's looks, in time order.

        Each period of ``burst`` blocks the device runs that many callbacks back to
        back, ``cb_late()`` after the period starts.  ``prefill`` more run at once
        at the very start, filling its buffer ahead of that.  The reported output
        latency holds for the steady callbacks, so block j is heard
        ``(j - prefill) * BLOCK / SR`` plus that latency: the clock the tests
        measure against.  The sender looks every ``wake()`` seconds."""
        period = burst * BLOCK / SR
        events = [(i * 1e-5, "cb") for i in range(prefill)]
        h = 0
        while h * period < until_s:
            t0 = h * period + cb_late()
            events += [(t0 + j * 1e-5, "cb") for j in range(burst)]
            h += 1
        w = self.t
        while w < until_s:
            events.append((w, "wake"))
            w += wake()
        for t, kind in sorted(events):
            self.t = t
            if kind == "cb":
                self.playhead += BLOCK
            else:
                self.band.pump()

    def timed(self):
        return list(self.port.sent)

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
        assert rig.timed()[0][0] == pytest.approx(want, abs=0.0015)


def test_a_take_starts_by_clearing_every_channel_it_can_play_on():
    rig = Rig()
    for ch in (0, 1, 2, 3, 4, 5, 6, DRUM_CHANNEL):
        assert ("cc", ch, 64, 0) in rig.cleared and ("cc", ch, 123, 0) in rig.cleared
    assert all(m[0] == "cc" for m in rig.cleared)


def test_stop_ends_every_sounding_note_and_says_all_notes_off():
    rig = Rig()
    rig.band.bar(0, BAR)
    rig.band.bar(2 * SR, BAR)           # queued, never reached
    rig.run(0.6)                        # the pad still sounds; the bass has ended
    before = len(rig.sent())
    rig.band.close()
    sent = rig.sent()
    assert sent[before:] == [("off", 1, 60, 0), ("cc", 0, 64, 0), ("cc", 0, 123, 0),
                             ("cc", 1, 64, 0), ("cc", 1, 123, 0), ("cc", 9, 64, 0),
                             ("cc", 9, 123, 0)]
    assert paired([m for m in sent if m[0] != "cc"])
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


def test_a_moved_bar_is_sent_where_it_moved_to():
    rig = Rig()
    rig.band.bar(SR // 2, BAR[:1])
    rig.band.move(SR // 2, SR // 2 + SR // 10)
    rig.band.move(SR // 2 + SR // 10, SR // 2 + SR // 5)    # moved again, by its new place
    rig.run(1.0)
    assert rig.timed()[0] == (pytest.approx(0.7, abs=0.0015), ("on", 9, 36, 114))


def test_notes_already_behind_the_playhead_are_not_sent_late():
    rig = Rig()
    rig.run(0.5)
    rig.band.bar(SR // 4, BAR)          # half of this bar's audio is already gone
    rig.run(3.0)
    ons = [m for m in rig.sent() if m[0] == "on"]
    assert ons == [("on", 1, 72, 102)]  # only the cue at 1.5 s into the bar


def test_an_off_that_fails_is_kept_until_it_is_sent():
    rig = Rig()
    real = rig.port.send
    failed = []

    def once(msg):
        if msg.type == "note_off" and not failed:
            failed.append(msg.as_tuple())
            raise OSError("busy")
        real(msg)
    rig.port.send = once
    rig.band.bar(0, BAR[:1])
    rig.run(1.0)
    assert failed == [("off", 9, 36, 0)]
    assert rig.band._schedule.sounding == 1
    rig.band.close()
    assert ("off", 9, 36, 0) in rig.sent()


def stalled_band(stall_s):
    """A band whose sender is held inside its first note-on for ``stall_s``, with
    two more note-ons behind it in the same batch.  The port keeps any message
    already inside the driver, even once closed."""
    mido = FakeMidoOut()
    band = BandMidi("IAC Driver Bus 1", sr=SR, block=BLOCK, poll_s=0.001)
    with patched_module("mido", mido):
        band.open()
    port = mido.ports[0]
    held = threading.Event()
    real = port.send

    def slow(msg):
        if msg.type == "note_on" and not held.is_set():
            held.set()
            time.sleep(stall_s)
        real(msg)
    port.send = slow
    port.close = lambda: None
    playhead = [0]
    band.start(lambda: playhead[0])
    port.sent.clear()                   # what start sent to clear the channels
    playhead[0] = BLOCK
    time.sleep(0.003)
    playhead[0] = 2 * BLOCK
    band.bar(2 * BLOCK, [{"role": r, "onset_s": 0.0, "dur_s": 5.0, "midi": n}
                         for r, n in (("bass", 40), ("pad", 60), ("lead", 72))])
    assert held.wait(2.0)
    return band, port


def test_no_note_on_follows_the_final_all_notes_off_when_a_send_stalls():
    band, port = stalled_band(0.6)
    band.close(timeout=1.0)             # the sender is still stuck: release waits for it
    sent = [m for _t, m in port.sent]
    # Only the note-on already inside the driver went out, the rest of its batch did
    # not since close had begun, and the release came after it.
    ons = [i for i, m in enumerate(sent) if m[0] == "on"]
    assert [sent[i] for i in ons] == [("on", 0, 40, 102)]
    assert sent[ons[0] + 1:] == [("off", 0, 40, 0), ("off", 1, 60, 0), ("off", 2, 72, 0),
                                 ("cc", 0, 64, 0), ("cc", 0, 123, 0), ("cc", 1, 64, 0),
                                 ("cc", 1, 123, 0), ("cc", 2, 64, 0), ("cc", 2, 123, 0)]


def test_a_stall_longer_than_close_waits_never_puts_a_note_after_the_release():
    band, port = stalled_band(0.5)
    band.close(timeout=0.1)             # gives up on the sender, and on the release
    time.sleep(0.7)                     # the stuck note-on lands now
    sent = [m for _t, m in port.sent]
    ccs = [i for i, m in enumerate(sent) if m[0] == "cc"]
    assert not ccs or not [m for m in sent[ccs[-1]:] if m[0] == "on"]


class Wedged(FakeOut):
    def send(self, msg):
        threading.Event().wait()        # a driver that never returns


def wedged_band():
    band = BandMidi("IAC Driver Bus 1", sr=SR, block=BLOCK)
    with patched_module("mido", FakeMidoOut()):
        band.open()
    band._handle = Wedged("IAC Driver Bus 1", False)
    band.bar(0, BAR[:1])
    band._schedule.due(10)              # a note sounding, so release has work to do
    return band


def test_close_is_bounded_when_the_driver_never_returns():
    band = wedged_band()
    closing = threading.Thread(target=band.close, kwargs={"timeout": 0.1}, daemon=True)
    closing.start()
    closing.join(1.0)
    assert not closing.is_alive(), "close waited on the driver"
    assert band.errors


def test_a_hard_exit_gets_out_past_a_stuck_driver_with_the_take_closed_first(monkeypatch):
    done = []
    monkeypatch.setattr(realtime, "unregister_session",
                        lambda *_a: done.append(time.monotonic()))
    band = wedged_band()
    exited = threading.Event()
    armed = time.monotonic()
    hard = realtime.HardExit(exit_fn=lambda _c: exited.set(),
                             on_exit=realtime._hard_exit_cleanup(None, None, dict,
                                                                 band_midi=band),
                             stream=open(os.devnull, "w"))
    hard.arm("test", seconds=0.0)
    assert exited.wait(3.0), "the watchdog never exited"
    assert len(done) == 1 and done[0] - armed < 0.5, "the take waited on the driver"


def test_a_hard_exit_lets_go_of_the_notes(monkeypatch):
    monkeypatch.setattr(realtime, "unregister_session", lambda *_a: None)
    rig = Rig()
    rig.band.bar(0, BAR[:1])
    rig.run(0.01)
    realtime._hard_exit_cleanup(None, None, dict, band_midi=rig.band)("test")
    assert rig.sent()[-2:] == [("cc", 9, 64, 0), ("cc", 9, 123, 0)]
    assert rig.port.closed


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
    err_ms = send_errors_ms(rig, ideal)
    assert min(err_ms) > -0.5          # never early by more than the clock guess
    assert max(err_ms) < 2.0
    assert statistics.mean(err_ms) < 1.0


def send_errors_ms(rig, ideal, label="one block per callback"):
    ons = sorted(t for t, m in rig.timed() if m[0] == "on")
    assert len(ons) == len(ideal)
    err_ms = [1000.0 * (t - i) for t, i in zip(ons, ideal)]
    print(f"midi out send error, {label}: mean {statistics.mean(err_ms):.2f} ms, "
          f"max {max(err_ms):.2f} ms, min {min(err_ms):.2f} ms")
    return err_ms


@pytest.mark.parametrize("burst,prefill", [(1, 0), (2, 0), (4, 0), (2, 4), (1, 3)])
def test_send_timing_when_the_device_runs_callbacks_in_bursts(burst, prefill):
    """A device that runs several callbacks back to back each period, and fills
    its buffer at start-up, hears block j from j blocks after the first one, like
    any other.  The clock must not take the end of a burst for its start."""
    rng = random.Random(5)
    rig = Rig(latency_s=0.01)
    bars = 4
    for b in range(bars):
        rig.band.bar(b * 2 * SR + SR // 4, BAR)
    rig.run(bars * 2.0 + 1.0, burst=burst, prefill=prefill,
            cb_late=lambda: rng.uniform(0.0, 0.0005))
    ideal = sorted((on - prefill * BLOCK) / SR + 0.01 for b in range(bars)
                   for on, *_ in bar_notes(BAR, b * 2 * SR + SR // 4, SR))
    err_ms = send_errors_ms(rig, ideal, f"{burst} per burst, {prefill} at start")
    assert min(err_ms) > -0.5
    assert max(err_ms) < 2.0


def test_the_stream_clock_dates_a_move_by_where_it_moved_from():
    c = StreamClock(SR, BLOCK)
    c.observe(0, 0.0)
    assert not c.ready
    c.observe(BLOCK, 0.0102)            # block 0 went at or before 10.2 ms
    c.observe(2 * BLOCK, 0.0109)        # 0.7 ms later: the same burst, ignored
    assert c.time_of(0) == pytest.approx(0.0102, abs=1e-9)
    c.observe(3 * BLOCK, 0.0102 + 2 * BLOCK / SR + 0.0001)   # its own period
    assert c.time_of(0) == pytest.approx(0.0102, abs=1e-9)


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


def clicks(bpm: float, seconds: float) -> np.ndarray:
    """A 1 kHz click on every beat: enough for the sideman to lock."""
    out = np.zeros(int(seconds * SR), dtype=np.float32)
    n = int(0.010 * SR)
    tone = (0.8 * np.exp(-np.arange(n) / (0.002 * SR))
            * np.sin(2 * np.pi * 1000.0 * np.arange(n) / SR)).astype(np.float32)
    for k in range(int(seconds * bpm / 60.0) - 1):
        i0 = int(round(k * 60.0 / bpm * SR))
        out[i0:i0 + n] += tone
    return out


@pytest.mark.usefixtures("fake_device_consent")
class TestLive:
    @pytest.mark.parametrize("verb", ["jam", "lead"])
    def test_off_never_touches_the_module(self, verb, plume_registry):
        """With no --midi-out a take runs as it always has: nothing imports midiout."""
        sd = fake_audio(None, rate_x=12.0)
        sock = short_socket_path("mo-off")
        with patched_module("sounddevice", sd), patched_module("plume.midiout", None):
            if verb == "jam":
                realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(),
                             socket_path=sock, max_minutes=0.02)
            else:
                realtime.lead(prompt="lofi at 120 bpm", seed=0, status=False,
                              socket_path=sock, max_minutes=0.02)
        # A lead plays from its first bar; a sideman with no one to follow stays quiet.
        assert sd.stream.sounded() is (verb == "lead")

    def test_a_locked_jam_sends_its_bars_and_lets_go_at_the_end(self, plume_registry):
        sd, mido = fake_audio(clicks(120.0, 60.0), rate_x=12.0), FakeMidoOut()
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            realtime.jam(style="lofi", seed=0, params=mind_mod.MindParams(),
                         max_minutes=0.05, socket_path=short_socket_path("mo-jam"),
                         midi_out="iac")
        port = mido.ports[0]
        sent = [m for _t, m in port.sent]
        assert sum(m[0] == "on" for m in sent) >= 4, "the sideman never sent a bar"
        assert paired([m for m in sent if m[0] != "cc"])
        assert port.closed and sent[-1][2] == 123

    def test_a_stream_that_will_not_open_still_closes_the_port(self, plume_registry):
        sd, mido = fake_audio(None), FakeMidoOut()

        def refuse(**_kw):
            raise OSError("device busy")
        sd.Stream = refuse
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            with pytest.raises(OSError, match="busy"):
                realtime.lead(prompt="lofi", seed=0, status=False,
                              socket_path=short_socket_path("mo-busy"), midi_out="iac")
        assert mido.ports[0].closed

    def test_the_same_port_both_ways_is_warned_about(self, capsys):
        realtime._warn_midi_loop("IAC Driver Bus 1", BandMidi("IAC Driver Bus 1"))
        assert "both IAC Driver Bus 1" in capsys.readouterr().err
        realtime._warn_midi_loop("Keys", BandMidi("IAC Driver Bus 1"))
        realtime._warn_midi_loop(None, BandMidi("IAC Driver Bus 1"))
        assert capsys.readouterr().err == ""

    def test_an_offset_past_half_a_second_is_refused(self):
        with pytest.raises(realtime.RealtimeError, match="500"):
            realtime._open_band_midi("virtual", offset_s=0.6)

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
    def test_a_lead_take_sends_its_band_and_ends_every_note(self, only, plume_registry,
                                                           tmp_path):
        """The machine plays from its first bar, so notes go out with no input at all.
        With --midi-only the device hears none of the band.  Either way the take's
        folder keeps what was sent as band.mid."""
        sd, mido = fake_audio(None, rate_x=6.0), RealFileMido()
        rec = tmp_path / "take"
        with patched_module("sounddevice", sd), patched_module("mido", mido):
            realtime.lead(prompt="lofi at 120 bpm", seed=0, status=False, max_minutes=0.04,
                          socket_path=short_socket_path("mo-lead"), midi_out="iac",
                          midi_only=only, record_dir=str(rec))
        port = mido.ports[0]
        sent = [m for _t, m in port.sent]
        assert port.closed and port.name == "IAC Driver Bus 1"
        assert sum(m[0] == "on" for m in sent) >= 8, sent[:5]
        assert {m[1] for m in sent if m[0] == "on"} >= {0, DRUM_CHANNEL}
        assert paired([m for m in sent if m[0] != "cc"])
        assert ("cc", DRUM_CHANNEL, 123, 0) in sent
        assert sd.stream.sounded() is not only
        bpm, tracks = read_band(rec / realtime.REC_BAND)
        assert bpm == pytest.approx(120.0, abs=0.5)
        assert sorted(n[1:] for notes in tracks.values() for n in notes
                      if n[1] == "on") == sorted(m for m in sent if m[0] == "on")
        for notes in tracks.values():
            assert paired([n[1:] for n in notes]) and notes[-1][1] == "off"

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

    def test_a_huge_offset_is_refused(self, monkeypatch, capsys):
        monkeypatch.setattr(realtime, "check_device", lambda *_a, **_k: None)
        monkeypatch.setattr(realtime, "jam", lambda **_k: None)
        assert cli.main(["jam", "--midi-out", "virtual", "--midi-out-offset", "-600"]) != 0
        assert "-500 and 500" in capsys.readouterr().err

    @pytest.mark.parametrize("flags", [["--midi-only"], ["--midi-out-offset", "5"]])
    def test_options_with_no_port_are_refused(self, flags, monkeypatch, capsys):
        monkeypatch.setattr(realtime, "check_device", lambda *_a, **_k: None)
        monkeypatch.setattr(realtime, "jam", lambda **_k: None)
        assert cli.main(["jam", *flags]) != 0
        assert "--midi-out" in capsys.readouterr().err


# ------------------------------------------------ band.mid: what was sent, kept

def read_band(path):
    """A written band.mid read back with mido: its tempo and, per track name, the notes
    as ``(seconds, kind, channel, note, velocity)``."""
    mido = pytest.importorskip("mido")
    f = mido.MidiFile(str(path))
    assert f.type == 1
    tempos = [m.tempo for m in f.tracks[0] if m.type == "set_tempo"]
    assert len(tempos) == 1
    tracks = {}
    for tr in f.tracks[1:]:
        t, name, notes = 0, None, []
        for m in tr:
            t += m.time
            if m.type == "track_name":
                name = m.name
            elif m.type in ("note_on", "note_off"):
                kind = "on" if m.type == "note_on" and m.velocity else "off"
                notes.append((mido.tick2second(t, f.ticks_per_beat, tempos[0]), kind,
                              m.channel, m.note, m.velocity))
        assert tr[-1].type == "end_of_track"
        tracks[name] = notes
    return mido.tempo2bpm(tempos[0]), tracks


def test_a_take_keeps_what_it_sent_as_a_midi_file_at_the_audios_time(tmp_path):
    path = tmp_path / "band.mid"
    rig = Rig(latency_s=0.02, file=BandFile(path, sr=SR))
    rig.band.bar(SR // 2, BAR, bpm=100.0)           # the bar's audio starts at 0.5 s
    rig.band.bar(4 * SR, BAR, bpm=100.0)            # queued, never reached
    rig.run(0.8)                                    # the pad still sounds at the end
    rig.playhead = int(0.8 * SR)
    rig.band.close()
    bpm, tracks = read_band(path)
    assert bpm == pytest.approx(100.0)
    assert set(tracks) == {"bass", "pad and keys", "drums"}
    sent = [m for m in rig.sent() if m[0] != "cc"]
    kept = sorted((t, kind, ch, n, v) for notes in tracks.values()
                  for t, kind, ch, n, v in notes)
    # Every note the port was sent is in the file, and on the channel it was sent on.
    assert sorted(m for m in sent if m[0] == "on") == sorted(
        (k, ch, n, v) for _t, k, ch, n, v in kept if k == "on")
    assert all(ch == {"bass": 0, "pad and keys": 1, "drums": DRUM_CHANNEL}[name]
               for name, notes in tracks.items() for _t, _k, ch, _n, _v in notes)
    tick = 60.0 / 100.0 / midiout.TICKS_PER_BEAT
    at = {(k, ch, n): t for t, k, ch, n, _v in kept}
    # At the time plume's own sound of each note starts, not when the port was sent it.
    assert at[("on", 9, 36)] == pytest.approx(0.5, abs=tick)
    assert at[("off", 9, 36)] == pytest.approx(0.55, abs=tick)
    assert at[("on", 9, 42)] == pytest.approx(0.75, abs=tick)
    assert at[("on", 1, 60)] == pytest.approx(0.5, abs=tick)
    assert at[("off", 1, 60)] == pytest.approx(0.8, abs=tick)   # let go at the end
    for notes in tracks.values():
        assert paired([(k, ch, n, v) for _t, k, ch, n, v in notes])
        assert notes[-1][1] == "off"
    assert not list(tmp_path.glob("*.tmp"))


def test_the_files_tempo_is_the_takes_and_its_times_do_not_drift(tmp_path):
    f = BandFile(tmp_path / "band.mid", sr=SR)
    for bpm in (88.0, 90.0, 90.0, 91.0, 240.0):     # one wild bar does not move it
        f.tempo(bpm)
    f.tempo(float("nan"))
    one_hour = 3600 * SR
    f.add(one_hour, "on", 0, 40, 90)
    f.add(one_hour + SR, "off", 0, 40, 0)
    assert f.write()
    bpm, tracks = read_band(f.path)
    assert bpm == pytest.approx(90.0)
    tick = 60.0 / 90.0 / midiout.TICKS_PER_BEAT
    assert [round(t / tick) * tick for t, *_ in tracks["bass"]] == pytest.approx(
        [3600.0, 3601.0], abs=tick)
    g = BandFile(tmp_path / "none.mid", sr=SR)
    g.add(0, "on", 9, 36, 100)
    assert g.write()
    assert read_band(g.path)[0] == pytest.approx(midiout.DEFAULT_FILE_BPM)


def test_the_file_is_bounded_and_still_ends_every_note(tmp_path):
    f = BandFile(tmp_path / "band.mid", sr=SR, max_messages=6)
    for i in range(20):
        f.add(i * 100, "on", 1, 60 + i % 3, 80)
        f.add(i * 100 + 50, "off", 1, 60 + i % 3, 0)
    f.add(5000, "on", 1, 72, 80)                     # past the cap: dropped
    f.add(4000, "off", 0, 33, 0)                     # an off for no note: not kept
    assert f.dropped == 18
    f.add(5000, "cc", 1, 123, 0)                     # not a note: not kept
    assert len(f) == 6
    g = BandFile(tmp_path / "held.mid", sr=SR, max_messages=4)
    g.add(0, "on", 0, 40, 90)
    g.add(10, "on", 1, 60, 90)
    g.add(20, "on", 2, 70, 90)                       # no room: the two held still end
    g.end(SR)
    assert len(g) == 4 and g.dropped == 1
    assert g.write()
    _bpm, tracks = read_band(g.path)
    assert [n[1:] for notes in tracks.values() for n in notes] == [
        ("on", 0, 40, 90), ("off", 0, 40, 0), ("on", 1, 60, 90), ("off", 1, 60, 0)]
    assert f.write() and paired([n[1:] for notes in read_band(f.path)[1].values()
                                 for n in notes])


def test_a_file_that_cannot_be_written_costs_the_file_not_the_take(tmp_path):
    (tmp_path / "taken").write_text("not a folder")
    f = BandFile(tmp_path / "taken" / "band.mid", sr=SR)
    f.add(0, "on", 0, 40, 90)
    assert f.write() is False and f.errors
    assert f.write() is False                        # once is all it tries


def test_a_band_that_never_started_writes_no_file(tmp_path):
    band = BandMidi("IAC Driver Bus 1", sr=SR, block=BLOCK,
                    file=BandFile(tmp_path / "band.mid", sr=SR))
    with patched_module("mido", FakeMidoOut()):
        band.open()
    band.close()
    assert not (tmp_path / "band.mid").exists()


def test_a_hard_exit_keeps_the_file_past_a_stuck_driver(monkeypatch, tmp_path):
    monkeypatch.setattr(realtime, "unregister_session", lambda *_a: None)
    band = wedged_band()
    band.file = BandFile(tmp_path / "band.mid", sr=SR)
    band._playhead = lambda: SR
    band.file.add(0, "on", 9, 36, 114)               # sounding when the driver stuck
    began = time.monotonic()
    realtime._hard_exit_cleanup(None, None, dict, band_midi=band)("test")
    assert time.monotonic() - began < 2.0
    _bpm, tracks = read_band(tmp_path / "band.mid")
    assert [n[1:] for n in tracks["drums"]] == [("on", 9, 36, 114), ("off", 9, 36, 0)]


def test_a_new_take_in_the_folder_drops_the_last_takes_band(tmp_path):
    (tmp_path / realtime.REC_BAND).write_bytes(b"MThd old")
    rec = realtime.SessionRecorder(tmp_path).open({})
    rec.close()
    assert not (tmp_path / realtime.REC_BAND).exists()


class RealFileMido(FakeMidoOut):
    """The fake ports, with mido's own messages and file classes, so a take run inside
    it can write its band.mid."""

    def __init__(self):
        super().__init__()
        mido = pytest.importorskip("mido")
        for name in ("Message", "MetaMessage", "MidiFile", "MidiTrack", "bpm2tempo"):
            setattr(self, name, getattr(mido, name))

