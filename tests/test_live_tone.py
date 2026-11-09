"""Tone cues change the LIVE band, not only the offline twin (2026-11-06).

THE DEFECT: "darker"/"brighter" in a tap-start or page take set
``_LeadCues.tilt``, which only ``plume simlead`` read.  ``realtime.lead`` never
touched it, yet the decision log said ``tone +0.0 -> -1.0 from here on``.
"warmer", "dry" and "wet" said "NOT applied" everywhere.  Now one stateful bus
stage, :class:`plume.fx.BusTone`, runs bar by bar in both the live loop and
simlead: gradual, bounded, and bit-exact pass-through until a cue moves it.
"""
import json
import threading

import numpy as np
import pytest
from conftest import needs_laptop_speed  # noqa: E402

from plume import fx
from plume.sim import _LeadCues

SR = 48000


def _noise_bars(n_bars=8, bar_s=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n = int(bar_s * SR)
    return [(rng.standard_normal((n, 2)) * 0.1).astype(np.float32) for _ in range(n_bars)]


def _hf_share(x, lo_hz=4000.0):
    m = np.asarray(x, dtype=np.float64).mean(axis=1)
    spec = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(m.shape[0], 1.0 / SR)
    return float(spec[f >= lo_hz].sum() / max(spec.sum(), 1e-20))


# ------------------------------------------------------------ the bus stage

def test_a_neutral_bus_is_bit_exact_pass_through():
    tone = fx.BusTone(SR)
    for bar in _noise_bars():
        out = tone.process(bar, tilt=0.0, wet=0.0)
        assert out.dtype == np.float32 and np.array_equal(out, bar)
    assert tone.flush().shape[0] == 0


def _hf_energy(x, lo_hz=4000.0):
    m = np.asarray(x, dtype=np.float64).mean(axis=1)
    spec = np.abs(np.fft.rfft(m)) ** 2
    return float(spec[np.fft.rfftfreq(m.shape[0], 1.0 / SR) >= lo_hz].sum())


def test_darker_takes_the_top_off_and_brighter_adds_it():
    for tilt in (-1.0, 1.0):
        tone = fx.BusTone(SR)
        bars = _noise_bars()
        outs = [tone.process(b, tilt=tilt, wet=0.0) for b in bars]
        late_in, late_out = np.concatenate(bars[4:]), np.concatenate(outs[4:])
        ratio = _hf_energy(late_out) / _hf_energy(late_in)
        assert ratio < 0.5 if tilt < 0 else ratio > 1.5, (tilt, ratio)


def test_the_tone_moves_gradually_not_in_one_step():
    tone = fx.BusTone(SR)
    bar = _noise_bars(1, bar_s=2.0)[0]
    out = tone.process(bar, tilt=-1.0, wet=0.0)
    # the first 20 ms is almost the dry signal; the end of the glide is not
    head = np.abs(out[: int(0.02 * SR)] - bar[: int(0.02 * SR)]).max()
    tail = np.abs(out[-int(0.02 * SR):] - bar[-int(0.02 * SR):]).max()
    assert head < 0.1 * tail
    assert tone.tilt == pytest.approx(-1.0)        # ...and it does arrive


def test_it_is_bounded_however_hard_it_is_pushed():
    tone = fx.BusTone(SR)
    for bar in _noise_bars():
        tone.process(bar, tilt=-50.0, wet=50.0)
    assert tone.tilt == pytest.approx(-1.0)
    assert tone.wet == pytest.approx(fx.BUS_TONE_WET_MAX)


def test_bar_by_bar_is_one_continuous_stream():
    """No step at a bar line: per-bar output equals processing the whole run."""
    bars = _noise_bars()
    a = fx.BusTone(SR)
    pieces = [a.process(b, tilt=-1.0, wet=0.2) for b in bars]
    b = fx.BusTone(SR)
    whole = b.process(np.concatenate(bars), tilt=-1.0, wet=0.2)
    np.testing.assert_allclose(np.concatenate(pieces), whole, atol=1e-5)
    np.testing.assert_allclose(a.flush(), b.flush(), atol=1e-5)


def test_wet_leaves_a_tail_and_dry_lets_it_ring_out():
    tone = fx.BusTone(SR)
    click = np.zeros((SR // 2, 2), dtype=np.float32)
    click[100] = 0.8
    tone.process(click, tilt=0.0, wet=fx.BUS_TONE_WET_MAX)   # glide up
    tone.process(click, tilt=0.0, wet=fx.BUS_TONE_WET_MAX)
    silence = np.zeros_like(click)
    after = tone.process(silence, tilt=0.0, wet=0.0)
    assert np.abs(after).max() > 1e-4, "no reverb tail at all"
    # going dry fades the send, so new sound is drier, and the old tail decays
    tail = tone.flush()
    assert tail.shape[0] > 0 and np.abs(tail[-100:]).max() < np.abs(after).max()


# ------------------------------------------------------------ the log

@pytest.mark.parametrize("cue,field,sign", [
    ("darker", "tilt", -1), ("brighter", "tilt", 1), ("warmer", "tilt", -1),
    ("wet", "wet", 1), ("more reverb", "wet", 1)])
def test_tone_cues_move_the_state_and_say_so(cue, field, sign):
    c = _LeadCues()
    notes = c.apply(cue)
    assert len(notes) == 1 and "NOT applied" not in notes[0]
    assert np.sign(getattr(c, field)) == sign


def test_warmer_is_a_gentler_move_than_darker():
    w, d = _LeadCues(), _LeadCues()
    w.apply("warmer")
    d.apply("darker")
    assert d.tilt < w.tilt < 0.0


def test_a_cue_at_its_limit_says_nothing_changed():
    c = _LeadCues()
    note = c.apply("dry")[0]
    assert "already" in note and c.wet == 0.0
    c.apply("darker")
    note = c.apply("darker")[0]
    assert "already" in note and c.tilt == -1.0
    for _ in range(10):
        c.apply("wet")
    assert c.wet == pytest.approx(fx.BUS_TONE_WET_MAX)
    assert "already" in c.apply("wet")[0]


def test_no_tone_cue_claims_the_others_do_not_land():
    """The old note said "'sparser'/'busier'/'darker'/'brighter' do land" from
    warmer/dry/wet; every texture cue lands now, so no note may say otherwise."""
    for cue in ("darker", "brighter", "warmer", "dry", "wet", "sparser", "busier"):
        for note in _LeadCues().apply(cue):
            assert "NOT applied" not in note, (cue, note)


# ------------------------------------------------------------ a live take

RIG = [{"name": "Audio Interface 4x4", "max_input_channels": 4, "max_output_channels": 4}]


@pytest.fixture
def consent(tmp_path, monkeypatch, plume_registry):
    from plume import realtime
    monkeypatch.setenv("PLUME_CALIBRATION_FILE", str(tmp_path / "none.json"))
    monkeypatch.setenv(realtime.AUDIO_CONSENT_ENV, "1")   # the FAKE device only


def _live_take(tmp_path, tag, cue=None):
    from conftest import fake_audio, patched_module, short_socket_path, wait_until
    from test_bandentry import _take

    from plume import realtime
    from plume import roles as roles_mod

    rec = tmp_path / tag
    sock = short_socket_path(tag)
    human = _take(bpm=115.0, seconds=80.0)

    def send_later():
        def band_in():
            try:
                st = realtime.send_status(socket_path=sock)
            except realtime.RealtimeError:
                return False
            return (st.get("bar") or 0) >= 14
        if wait_until(band_in, timeout=60.0):
            realtime.send_cue(cue, socket_path=sock, source="cli")

    fake = fake_audio(human, devices=RIG, rate_x=12.0)
    t = threading.Thread(target=send_later, daemon=True) if cue else None
    with patched_module("sounddevice", fake):
        if t:
            t.start()
        realtime.lead(prompt="midwest emo in C major", seed=0, tap_start=True,
                      direct=True, role=roles_mod.RoleParams(role="lead"),
                      record_dir=str(rec), status=False, socket_path=sock,
                      max_minutes=1.0, input_device="audio interface",
                      input_channel=1)
    if t:
        t.join(timeout=5.0)
    rows = [json.loads(x) for x in (rec / "decisions.ndjson").read_text().splitlines()]
    return fake.stream.output, [r for r in rows if isinstance(r.get("bar"), int)]


@needs_laptop_speed
def test_darker_in_a_live_take_darkens_what_the_band_plays(tmp_path, consent):
    base_out, _ = _live_take(tmp_path, "base")
    out, rows = _live_take(tmp_path, "dark", cue="darker")
    hit = [r for r in rows if r["form"]["applied_cue"] == "darker"]
    assert hit, "the cue never landed"
    assert any("tone" in e for e in hit[0]["cue_effects"])
    t = float(hit[0]["t_s"])

    def win(x, a, b):
        return x[int(a * SR):int(b * SR)]
    before, after = (t - 6.0, t - 0.5), (t + 1.5, t + 7.0)
    assert win(out, *after).shape[0] > SR, "the take ended before it could be heard"
    dark = _hf_share(win(out, *after)) / _hf_share(win(out, *before))
    plain = _hf_share(win(base_out, *after)) / _hf_share(win(base_out, *before))
    assert dark < plain * 0.75, (
        f"'darker' did not darken the live band: top-end share x{dark:.2f} "
        f"vs x{plain:.2f} uncued")
