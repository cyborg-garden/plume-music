"""``plume watch``: a feed read from a recording directory, no device."""
import json

import numpy as np
import soundfile as sf

from plume.watch import Listener, follow

SR = 48000
BPM = 120.0
BAR = 2.0


def session(tmp_path, *, playing, bleed=0.1, bars=24):
    """playing(bar) -> guitar amplitude.  The band is always on, bleeding into the mic."""
    rng = np.random.default_rng(1)
    human, band, rows = [], [], []
    for b in range(bars):
        t = np.arange(int(BAR * SR)) / SR
        m = 0.1 * np.sin(2 * np.pi * 110.0 * t)
        g = playing(b) * sum(np.sin(2 * np.pi * f * t) for f in (349.2, 440.0, 523.3)) \
            * np.exp(-t * 3.0)
        human.append(g + bleed * m + rng.standard_normal(t.size) * 1e-4)
        band.append(m)
        rows.append({"bar": b, "t_s": b * BAR, "bpm": BPM, "key": "F major", "chord": "Fmaj7",
                     "section": "verse" if b < 12 else "lift", "entry": "full"})
    sf.write(str(tmp_path / "human.wav"), np.concatenate(human).astype("float32"), SR)
    sf.write(str(tmp_path / "machine.wav"), np.concatenate(band).astype("float32"), SR)
    (tmp_path / "decisions.ndjson").write_text(
        '{"kind": "manifest"}\n' + "".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def test_one_line_per_group_with_the_band_and_the_player(tmp_path):
    docs = Listener(session(tmp_path, playing=lambda b: 0.2), every=4).poll()
    assert [d["bar"] for d in docs] == list(range(3, 24, 4))
    assert docs[0]["section"] == "verse" and docs[0]["key"] == "F major"
    assert set(docs[0]["you_notes"]) & {"F", "A", "C"}


def test_it_says_when_the_player_stops_and_returns_despite_the_bleed(tmp_path):
    played = lambda b: 0.0 if 12 <= b < 20 else 0.2  # noqa: E731
    docs = Listener(session(tmp_path, playing=played), every=4).poll()
    changes = [c for d in docs for c in d["changes"]]
    assert "you:stopped" in changes and "you:back" in changes
    assert changes.index("you:stopped") < changes.index("you:back")


def test_it_passes_on_what_the_engine_decided_about_the_player(tmp_path):
    """The engine's word, not a second opinion: it is what fades the band."""
    d = session(tmp_path, playing=lambda b: 0.2, bars=1)
    row = json.loads((d / "decisions.ndjson").read_text().splitlines()[1])
    row.update(playing=False, playing_trusted=True, fade_db=-3.0)
    (d / "decisions.ndjson").write_text('{"kind": "manifest"}\n' + json.dumps(row) + "\n")
    doc = Listener(d, every=1).poll()[0]
    assert doc["playing"] is False and doc["playing_trusted"] is True and doc["fade_db"] == -3.0


def test_it_says_when_the_section_changes(tmp_path):
    docs = Listener(session(tmp_path, playing=lambda b: 0.2), every=4).poll()
    assert any("section:lift" in d["changes"] for d in docs)


def test_audio_that_has_not_arrived_waits_and_is_not_lost(tmp_path):
    d = session(tmp_path, playing=lambda b: 0.2)
    with (d / "decisions.ndjson").open("a") as f:
        f.write(json.dumps({"bar": 24, "t_s": 48.0, "bpm": BPM}) + "\n")
    listener = Listener(d, every=1)
    first = listener.poll()
    assert len(first) == 24 and listener.poll() == []


def test_a_half_written_line_is_read_on_the_next_poll(tmp_path):
    d = session(tmp_path, playing=lambda b: 0.2)
    text = (d / "decisions.ndjson").read_text()
    (d / "decisions.ndjson").write_text(text[:-20])
    listener = Listener(d, every=1)
    assert len(listener.poll()) == 23
    (d / "decisions.ndjson").write_text(text)
    assert len(listener.poll()) == 1


def test_follow_ends_when_the_session_does(tmp_path):
    d = session(tmp_path, playing=lambda b: 0.2)
    (d / "ending.json").write_text("{}")
    lines = []
    follow(d, changes_only=True, write=lines.append, poll_s=0.0)
    assert json.loads(lines[-1]) == {"ended": True}
    assert all(json.loads(x).get("changes") for x in lines[:-1])


def test_a_recording_not_written_yet_waits_and_junk_rows_are_skipped(tmp_path):
    (tmp_path / "decisions.ndjson").write_text(
        '[1, 2]\n' + json.dumps({"bar": 0, "t_s": 0.0, "bpm": BPM}) + "\n")
    assert Listener(tmp_path, every=1).poll() == []


def test_ctrl_c_leaves_the_watch_quietly(monkeypatch):
    from argparse import Namespace

    from plume import cli, watch

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(watch, "follow", interrupted)
    args = Namespace(record_dir="takes", every=4, changes_only=False)
    assert cli.cmd_watch(args) == ({"ended": True}, "", 0)
