"""packaging/check_wheels.py: a pin with no wheel for a covered system is caught.

Offline: the file lists are the shapes PyPI gives for these releases.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cw():
    spec = importlib.util.spec_from_file_location("check_wheels", ROOT / "packaging" / "check_wheels.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_wheels"] = mod  # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("check_wheels", None)


# onnxruntime 1.32.0: macOS wheels for Apple Silicon on macOS 14 only.
ONNX = [f"onnxruntime-1.32.0-cp{v}-cp{v}-{p}.whl"
        for v in ("311", "312", "313")
        for p in ("macosx_14_0_arm64", "manylinux_2_28_x86_64", "win_amd64")]
# python-rtmidi 1.5.8: nothing newer than cp312.
RTMIDI = [f"python_rtmidi-1.5.8-cp{v}-cp{v}-{p}.whl"
          for v in ("311", "312")
          for p in ("macosx_11_0_arm64", "macosx_10_9_x86_64", "manylinux_2_28_x86_64", "win_amd64")]


def test_covered_systems_pass(cw):
    pins = [cw.Pin("onnxruntime", "1.32.0"), cw.Pin("python-rtmidi", "1.5.8")]
    files = {("onnxruntime", "1.32.0"): ONNX, ("python-rtmidi", "1.5.8"): RTMIDI}
    assert cw.gaps(pins, files) == []


def test_intel_mac_and_macos_13_are_gaps(cw):
    intel = cw.System("Intel Mac", "macos", "x86_64", (14, 0))
    old = cw.System("macOS 13", "macos", "arm64", (13, 0))
    out = cw.gaps([cw.Pin("onnxruntime", "1.32.0")], {("onnxruntime", "1.32.0"): ONNX},
                  pythons=("3.12",), systems=(intel, old))
    assert out == ["onnxruntime==1.32.0: no wheel for Python 3.12 on Intel Mac",
                   "onnxruntime==1.32.0: no wheel for Python 3.12 on macOS 13"]


def test_python_313_is_a_gap_for_rtmidi(cw):
    out = cw.gaps([cw.Pin("python-rtmidi", "1.5.8")], {("python-rtmidi", "1.5.8"): RTMIDI},
                  pythons=("3.13",))
    assert len(out) == len(cw.SYSTEMS)
    assert all("Python 3.13" in line for line in out)


def test_pure_python_and_abi3_wheels_count(cw):
    win = cw.SYSTEMS[2]
    assert cw.has_wheel(["mido-1.3.3-py3-none-any.whl"], "3.12", win)
    assert cw.has_wheel(["protobuf-7.36.2-cp310-abi3-win_amd64.whl"], "3.12", win)
    assert not cw.has_wheel(["x-1.0-cp310-cp310-win_amd64.whl"], "3.12", win)
    assert not cw.has_wheel(["x-1.0.tar.gz"], "3.12", win)


def test_legacy_manylinux_tags(cw):
    linux = cw.SYSTEMS[1]
    assert cw.has_wheel(["cffi-2.1.1-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"],
                        "3.11", linux)
    assert not cw.has_wheel(["x-1-cp311-cp311-manylinux_2_34_x86_64.whl"], "3.11", linux)


def test_markers_pick_one_pin_per_python(cw):
    pins = cw.read_pins('numpy==2.4.6 ; python_version < "3.12"\nnumpy==2.5.5 ; python_version >= "3.12"\n')
    assert [p.version for p in pins if p.applies("3.11")] == ["2.4.6"]
    assert [p.version for p in pins if p.applies("3.12")] == ["2.5.5"]


def test_unreadable_marker_is_refused(cw):
    with pytest.raises(ValueError):
        cw.read_pins('numpy==2.5.5 ; sys_platform == "darwin"\n')


def test_header_lists_what_the_check_covers(cw):
    """The file's claims and the check's targets are the same list."""
    text = (ROOT / "packaging" / "constraints.txt").read_text(encoding="utf-8")
    header = text[: text.index("\n\n# Direct dependencies.")]
    assert "Python " + " and ".join(cw.PYTHONS) in header
    for system in cw.SYSTEMS:
        assert system.label in header
    pins = cw.read_pins(text)  # every line parses
    assert {p.name for p in pins} >= {"numpy", "onnxruntime", "python-rtmidi"}
