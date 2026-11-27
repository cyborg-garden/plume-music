"""Check that every pin in constraints.txt has a wheel for every system it claims.

    python packaging/check_wheels.py            # asks PyPI, exits 1 on a gap

An install with the pins only works where each pinned version ships a wheel
(or a pure-Python one).  A CI install proves that for the runners it has; this
proves it for every system and Python the constraints file says it covers,
including the ones no runner is, such as an older macOS.  Standard library
only, so it runs before anything is installed.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CONSTRAINTS = Path(__file__).with_name("constraints.txt")

PYTHONS = ("3.11", "3.12")


@dataclass(frozen=True)
class System:
    label: str  # as written in the constraints.txt header
    kind: str  # "macos", "linux" or "windows"
    arch: str
    floor: tuple[int, int]  # oldest macOS, or oldest glibc on Linux


SYSTEMS = (
    System("macOS 14 or newer on Apple Silicon", "macos", "arm64", (14, 0)),
    System("Linux x86_64 with glibc 2.28 or newer", "linux", "x86_64", (2, 28)),
    System("Windows x64", "windows", "amd64", (0, 0)),
)


@dataclass(frozen=True)
class Pin:
    name: str
    version: str
    python_lt: str | None = None  # from `; python_version < "X"`
    python_ge: str | None = None  # from `; python_version >= "X"`

    def applies(self, python: str) -> bool:
        v = _ver(python)
        if self.python_lt and not v < _ver(self.python_lt):
            return False
        if self.python_ge and not v >= _ver(self.python_ge):
            return False
        return True


def _ver(s: str) -> tuple[int, ...]:
    return tuple(int(p) for p in s.split("."))


_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)\s*(?:;\s*(.*))?$")
_MARKER = re.compile(r'^python_version\s*(<|>=)\s*"([0-9.]+)"$')


def read_pins(text: str) -> list[Pin]:
    """The pins in a constraints file.  Markers other than python_version are refused."""
    pins = []
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        m = _PIN.match(line)
        if not m:
            raise ValueError(f"not a name==version pin: {raw!r}")
        name, version, marker = m.groups()
        lt = ge = None
        if marker:
            mm = _MARKER.match(marker.strip())
            if not mm:
                raise ValueError(f"marker this check cannot read: {raw!r}")
            lt, ge = (mm.group(2), None) if mm.group(1) == "<" else (None, mm.group(2))
        pins.append(Pin(name, version, lt, ge))
    return pins


def _python_ok(tag: str, abi: str, python: str) -> bool:
    major, minor = _ver(python)[:2]
    for t in tag.split("."):
        m = re.fullmatch(r"(cp|py)(\d)(\d*)", t)
        if not m or int(m.group(2)) != major:
            continue
        want = int(m.group(3)) if m.group(3) else None
        if m.group(1) == "py" and (want is None or want <= minor):
            return True
        if m.group(1) == "cp" and want is not None:
            abis = abi.split(".")
            if want == minor or (want <= minor and "abi3" in abis):
                return True
    return False


def _platform_ok(tags: str, system: System) -> bool:
    for t in tags.split("."):
        if t == "any":
            return True
        if system.kind == "windows" and t == f"win_{system.arch}":
            return True
        m = re.fullmatch(r"macosx_(\d+)_(\d+)_(\w+)", t)
        if system.kind == "macos" and m:
            arch_ok = m.group(3) in (system.arch, "universal2")
            if arch_ok and (int(m.group(1)), int(m.group(2))) <= system.floor:
                return True
        if system.kind == "linux":
            legacy = {"manylinux1": (2, 5), "manylinux2010": (2, 12), "manylinux2014": (2, 17)}
            m = re.fullmatch(r"manylinux_(\d+)_(\d+)_(\w+)", t)
            if m and m.group(3) == system.arch:
                if (int(m.group(1)), int(m.group(2))) <= system.floor:
                    return True
            for prefix, glibc in legacy.items():
                if t == f"{prefix}_{system.arch}" and glibc <= system.floor:
                    return True
    return False


def has_wheel(filenames: list[str], python: str, system: System) -> bool:
    """True if one of a release's files installs on this Python and system."""
    for f in filenames:
        if not f.endswith(".whl"):
            continue
        parts = f[:-4].split("-")
        if len(parts) < 5:
            continue
        py, abi, plat = parts[-3:]
        if _python_ok(py, abi, python) and _platform_ok(plat, system):
            return True
    return False


def gaps(pins: list[Pin], files: dict[tuple[str, str], list[str]],
         pythons=PYTHONS, systems=SYSTEMS) -> list[str]:
    """Every (pin, Python, system) with no wheel, as readable lines."""
    out = []
    for python in pythons:
        for pin in pins:
            if not pin.applies(python):
                continue
            for system in systems:
                if not has_wheel(files[(pin.name, pin.version)], python, system):
                    out.append(f"{pin.name}=={pin.version}: no wheel for "
                               f"Python {python} on {system.label}")
    return out


def pypi_files(name: str, version: str) -> list[str]:
    req = urllib.request.Request(
        f"https://pypi.org/pypi/{name}/{version}/json",
        headers={"User-Agent": "plume-constraints-check", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return [u["filename"] for u in json.load(r)["urls"]]


def main() -> int:
    pins = read_pins(CONSTRAINTS.read_text(encoding="utf-8"))
    files = {(p.name, p.version): pypi_files(p.name, p.version) for p in pins}
    bad = gaps(pins, files)
    print("\n".join(bad) or f"every pin has a wheel for Python {', '.join(PYTHONS)} on "
          + "; ".join(s.label for s in SYSTEMS))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
