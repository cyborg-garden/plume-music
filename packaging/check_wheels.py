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

# The libraries Plume asks for by name.  Each needs exactly one pin on every
# covered system and Python; what they bring with them may be pinned on some
# systems only.
DIRECT = ("numpy", "scipy", "soundfile", "imageio-ffmpeg",
          "sounddevice", "mido", "python-rtmidi", "onnxruntime")


@dataclass(frozen=True)
class System:
    label: str  # as written in the constraints.txt header
    kind: str  # "macos", "linux" or "windows"
    arch: str  # as wheel tags spell it
    floor: tuple[int, int]  # oldest macOS, or oldest glibc on Linux
    sys_platform: str  # what pip's markers see on this system
    platform_machine: str


SYSTEMS = (
    System("macOS 14 or newer on Apple Silicon", "macos", "arm64", (14, 0), "darwin", "arm64"),
    System("macOS 13 or newer on an Intel Mac", "macos", "x86_64", (13, 0), "darwin", "x86_64"),
    System("Linux x86_64 with glibc 2.28 or newer", "linux", "x86_64", (2, 28), "linux", "x86_64"),
    System("Windows x64", "windows", "amd64", (0, 0), "win32", "AMD64"),
)


@dataclass(frozen=True)
class Pin:
    name: str
    version: str
    # From the marker: (variable, operator, value) clauses, all of which must hold.
    when: tuple[tuple[str, str, str], ...] = ()

    def applies(self, python: str, system: System) -> bool:
        for var, op, value in self.when:
            if var == "python_version":
                ok = _ver(python) < _ver(value) if op == "<" else _ver(python) >= _ver(value)
            else:
                have = getattr(system, var)
                ok = (have == value) if op == "==" else (have != value)
            if not ok:
                return False
        return True


def _ver(s: str) -> tuple[int, ...]:
    return tuple(int(p) for p in s.split("."))


_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)\s*(?:;\s*(.*))?$")
_CLAUSE = re.compile(r'^(python_version)\s*(<|>=)\s*"([0-9.]+)"$'
                     r'|^(sys_platform|platform_machine)\s*(==|!=)\s*"([A-Za-z0-9_]+)"$')


def _clauses(marker: str, raw: str) -> tuple[tuple[str, str, str], ...]:
    out = []
    for part in marker.split(" and "):
        m = _CLAUSE.match(part.strip())
        if not m:
            raise ValueError(f"marker this check cannot read: {raw!r}")
        g = m.groups()
        out.append(g[:3] if g[0] else g[3:])
    return tuple(out)


def read_pins(text: str) -> list[Pin]:
    """The pins in a constraints file.

    A marker may join python_version (< or >=), sys_platform and
    platform_machine (== or !=) clauses with `and`.  Anything else is refused,
    so a pin is never counted for a system it does not reach.
    """
    pins = []
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        m = _PIN.match(line)
        if not m:
            raise ValueError(f"not a name==version pin: {raw!r}")
        name, version, marker = m.groups()
        pins.append(Pin(name, version, _clauses(marker, raw) if marker else ()))
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
            for system in systems:
                if not pin.applies(python, system):
                    continue
                if not has_wheel(files[(pin.name, pin.version)], python, system):
                    out.append(f"{pin.name}=={pin.version}: no wheel for "
                               f"Python {python} on {system.label}")
    return out


def pin_counts(pins: list[Pin], pythons=PYTHONS, systems=SYSTEMS,
               direct=DIRECT) -> list[str]:
    """Every (name, Python, system) with two pins, or a direct library with none."""
    out = []
    names = sorted({p.name for p in pins} | set(direct))
    for python in pythons:
        for system in systems:
            for name in names:
                n = sum(1 for p in pins if p.name == name and p.applies(python, system))
                if n > 1:
                    out.append(f"{name}: {n} pins for Python {python} on {system.label}")
                elif n == 0 and name in direct:
                    out.append(f"{name}: no pin for Python {python} on {system.label}")
    return out


def pypi_files(name: str, version: str) -> list[str]:
    req = urllib.request.Request(
        f"https://pypi.org/pypi/{name}/{version}/json",
        headers={"User-Agent": "plume-constraints-check", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return [u["filename"] for u in json.load(r)["urls"]]


def main() -> int:
    pins = read_pins(CONSTRAINTS.read_text(encoding="utf-8"))
    counted = pin_counts(pins)
    if counted:
        print("\n".join(counted))
        return 1
    files = {(p.name, p.version): pypi_files(p.name, p.version) for p in pins}
    bad = gaps(pins, files)
    print("\n".join(bad) or f"every pin has a wheel for Python {', '.join(PYTHONS)} on "
          + "; ".join(s.label for s in SYSTEMS))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
