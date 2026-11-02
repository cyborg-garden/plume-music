"""Placement compensation + acoustic calibration (v2.5) — headless.

The measurement chain is pure (click train in, lags out), so the whole
feature's arithmetic is tested without a device, per the project rule that
live behavior must be testable offline.
"""
import inspect
import json

import numpy as np
import pytest

from plume import calibrate as cal


SR = 48000


def _echo(out, click, marks, lag, gain=0.3):
    """A synthetic mic capture: each played click comes back ``lag`` samples
    later at ``gain``."""
    rec = np.zeros(out.shape[0], dtype=np.float32)
    for i in marks:
        j = i + lag
        rec[j:j + click.shape[0]] += gain * click[: max(0, rec.shape[0] - j)]
    return rec


def test_click_train_shape_and_determinism():
    out, click, marks = cal.click_train(SR)
    out2, click2, marks2 = cal.click_train(SR)
    assert out.shape[1] == 2 and out.dtype == np.float32
    assert len(marks) == cal.CLICK_COUNT
    assert np.array_equal(out, out2) and marks == marks2
    assert np.array_equal(click, click2)
    spacing = np.diff(marks)
    assert np.all(spacing == spacing[0])


def test_measure_offsets_recovers_exact_lag():
    out, click, marks = cal.click_train(SR)
    lag = int(0.1124 * SR)  # the development MacBook's real number
    rec = _echo(out, click, marks, lag)
    lags = cal.measure_offsets(rec, click, marks, SR)
    assert lags == [lag] * len(marks)


def test_measure_offsets_skips_unheard_clicks():
    out, click, marks = cal.click_train(SR)
    lag = int(0.05 * SR)
    rec = _echo(out, click, marks[:3], lag)  # only 3 of 8 come back
    lags = cal.measure_offsets(rec, click, marks, SR)
    assert lags == [lag] * 3


def test_measure_offsets_rejects_noise_floor():
    out, click, marks = cal.click_train(SR)
    rng = np.random.default_rng(9)
    rec = (1e-4 * rng.standard_normal(out.shape[0])).astype(np.float32)
    assert cal.measure_offsets(rec, click, marks, SR) == []


def test_summarize_median_spread_and_empty():
    s = cal.summarize([4800, 4810, 4790, 5000], SR)
    assert s["ok"] and s["heard"] == 4
    assert s["roundtrip_ms"] == pytest.approx(100.1, abs=0.1)
    assert s["spread_ms"] == pytest.approx((5000 - 4790) / SR * 1000, abs=0.1)
    empty = cal.summarize([], SR)
    assert not empty["ok"] and empty["roundtrip_ms"] is None


def test_placement_comp_resolution_order():
    f = cal.placement_comp_samples
    # explicit beats stored beats reported beats nothing
    assert f(0.050, 112.4, 0.141, SR) == int(round(0.050 * SR))
    assert f(None, 112.4, 0.141, SR) == int(round(0.1124 * SR))
    assert f(None, None, 0.141, SR) == int(round(0.141 * SR))
    assert f(None, None, None, SR) == 0
    # explicit zero DISABLES — it must not fall through to the others
    assert f(0.0, 112.4, 0.141, SR) == 0


def test_placement_comp_clamps():
    f = cal.placement_comp_samples
    assert f(-0.05, None, None, SR) == 0                      # never negative
    assert f(9.0, None, None, SR) == int(cal.MAX_COMP_S * SR)  # ceiling
    assert f(None, None, float("nan"), SR) == 0               # NaN reported
    assert f(None, None, float("inf"), SR) == 0


def test_measured_comp_passes_the_reported_ceiling():
    """A measured 317 ms must be honoured; the stream's unmeasured guess must not
    reach the same height (that pair reported 548 ms)."""
    f = cal.placement_comp_samples
    assert f(None, 317.0, None, SR) == int(round(0.317 * SR))
    assert f(0.317, None, None, SR) == int(round(0.317 * SR))
    assert f(None, None, 0.548, SR) == int(cal.MAX_REPORTED_COMP_S * SR)
    assert cal.MAX_REPORTED_COMP_S < cal.MAX_COMP_S  # non-vacuity


def test_store_roundtrip_and_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv(cal.CALIBRATION_ENV, str(tmp_path / "cal.json"))
    assert cal.load_roundtrip_ms("mic", "spk", SR) is None
    cal.save_roundtrip("mic", "spk", SR,
                       {"roundtrip_ms": 112.4, "spread_ms": 0.2,
                        "heard": 8, "clicks": 8})
    cal.save_roundtrip("mic", "hdmi", SR,
                       {"roundtrip_ms": 300.0, "spread_ms": 1.0,
                        "heard": 8, "clicks": 8})
    # each pair keeps its own number; sr is part of the key
    assert cal.load_roundtrip_ms("mic", "spk", SR) == 112.4
    assert cal.load_roundtrip_ms("mic", "hdmi", SR) == 300.0
    assert cal.load_roundtrip_ms("mic", "spk", 44100) is None


def test_store_survives_corrupt_file(tmp_path, monkeypatch):
    path = tmp_path / "cal.json"
    monkeypatch.setenv(cal.CALIBRATION_ENV, str(path))
    path.write_text("not json {")
    assert cal.load_roundtrip_ms("mic", "spk", SR) is None    # load fails soft
    cal.save_roundtrip("mic", "spk", SR,
                       {"roundtrip_ms": 90.0, "spread_ms": 0.1,
                        "heard": 8, "clicks": 8})              # save recovers
    assert cal.load_roundtrip_ms("mic", "spk", SR) == 90.0
    json.loads(path.read_text())                               # valid again


def test_jam_signature_grew_the_kwarg_additively():
    from plume import realtime
    params = inspect.signature(realtime.jam).parameters
    p = params["latency_comp_s"]
    assert p.default is None and p.kind is inspect.Parameter.KEYWORD_ONLY


def test_placement_comp_rejects_nan_everywhere():
    # NaN survives Python's min/max (the first argument wins), so an
    # unguarded clamp would raise at int() AFTER the device is open.
    f = cal.placement_comp_samples
    assert f(float("nan"), None, None, SR) == 0
    assert f(None, float("nan"), None, SR) == 0
    assert f(None, None, float("nan"), SR) == 0


@pytest.mark.parametrize("body", ["[1, 2, 3]", "\"x\"", "null",
                                  '{"entries": [1]}'])
def test_store_survives_wrong_shape_json(tmp_path, monkeypatch, body):
    # Valid JSON of the wrong SHAPE must be as fail-soft as invalid JSON:
    # load_roundtrip_ms runs inside jam startup, and save must self-heal.
    path = tmp_path / "cal.json"
    monkeypatch.setenv(cal.CALIBRATION_ENV, str(path))
    path.write_text(body)
    assert cal.load_roundtrip_ms("mic", "spk", SR) is None
    cal.save_roundtrip("mic", "spk", SR,
                       {"roundtrip_ms": 90.0, "spread_ms": 0.1,
                        "heard": 8, "clicks": 8})
    assert cal.load_roundtrip_ms("mic", "spk", SR) == 90.0


def test_load_rejects_stored_nan(tmp_path, monkeypatch):
    # json.loads accepts the nonstandard NaN literal; a NaN measurement must
    # route to the reported fallback (None), not to the placement math.
    path = tmp_path / "cal.json"
    monkeypatch.setenv(cal.CALIBRATION_ENV, str(path))
    key = cal._key("mic", "spk", SR)
    path.write_text(json.dumps(
        {"entries": {key: {"roundtrip_ms": float("nan")}}}))
    assert cal.load_roundtrip_ms("mic", "spk", SR) is None


def test_calibrate_reports_measurement_even_when_save_fails(monkeypatch):
    # The measurement cost a click train; an unwritable store must not
    # discard it.
    import argparse
    from plume import cli
    realtime = __import__("plume.realtime", fromlist=["x"])
    monkeypatch.setattr(realtime, "check_device", lambda device=None: None)
    ok = {"ok": True, "heard": 8, "clicks": 8, "roundtrip_ms": 112.4,
          "spread_ms": 0.2, "input": "mic", "output": "spk", "sr": SR,
          "reported_ms": 141.3}
    monkeypatch.setattr(cal, "run", lambda device=None: ok)

    def boom(*a, **k):
        raise OSError(13, "Permission denied")
    monkeypatch.setattr(cal, "save_roundtrip", boom)
    payload, human, code = cli.cmd_calibrate(
        argparse.Namespace(device=None, json=False))
    assert code == 0 and payload["saved_to"] is None
    assert "112.4" in human and "NOT saved" in human
    assert "--latency-comp 112.4" in human


def test_placement_sites_actually_subtract_comp():
    # Non-vacuity floor: every other test here passes with the placement
    # subtraction reverted.  The two sites (plan + revision-move) must both
    # subtract the shared comp, or the whole feature is a no-op that still
    # prints its banner.
    from plume import realtime
    src = inspect.getsource(realtime)
    assert src.count('- comp["samples"]') == 2
