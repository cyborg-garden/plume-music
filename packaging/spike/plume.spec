# -*- mode: python ; coding: utf-8 -*-
# Spike: a one-folder frozen Plume.  Throwaway: not for main.
#
#   pyinstaller packaging/spike/plume.spec --noconfirm
#
# PLUME_SPIKE_FFMPEG=0 leaves imageio-ffmpeg (and its ffmpeg binary) out, to measure
# the size without it.
import ctypes.util
import os
import subprocess
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

HERE = Path(SPECPATH)
SRC = HERE.parent.parent / "src"
PKG = SRC / "plume"
WITH_FFMPEG = os.environ.get("PLUME_SPIKE_FFMPEG", "1") != "0"

# plume imports its own modules by name (cli._import, plume.__getattr__), which the
# analysis cannot follow.  Take every one.
plume_modules = sorted(collect_submodules("plume"))
third_party = ["numpy", "scipy", "soundfile", "sounddevice", "mido", "mido.backends.rtmidi",
               "rtmidi", "onnxruntime"]
if WITH_FFMPEG:
    third_party.append("imageio_ffmpeg")
# A machine with no wheel for one of these still builds, and the self-test then says
# which import fails.
import importlib.util
missing = [m for m in third_party if importlib.util.find_spec(m.split(".")[0]) is None]
print("spike: not installed, left out of the build:", missing)

# The self-test reads this list back and imports each name inside the bundle.
listing = Path(workpath) / "spike_modules.txt"
listing.parent.mkdir(parents=True, exist_ok=True)
listing.write_text("\n".join([*plume_modules, *third_party]) + "\n", encoding="utf-8")

datas = [(str(listing), "spike")]
for folder in ("styles", "samples", "gui_static", "guide", "listen/models"):
    datas.append((str(PKG / folder), f"plume/{folder}"))

binaries = []
if sys.platform.startswith("linux"):
    # The sounddevice wheel carries PortAudio on macOS and Windows only.
    soname = ctypes.util.find_library("portaudio")
    found = None
    if soname:
        for line in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout.splitlines():
            if soname in line and "=>" in line:
                found = line.split("=>")[1].strip()
                break
    if not found:
        raise SystemExit("libportaudio2 is not installed: the Linux bundle would have no PortAudio")
    binaries.append((found, "."))

a = Analysis(
    [str(HERE / "plume_app.py")],
    pathex=[str(SRC), str(HERE)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[*plume_modules, *[m for m in third_party if m not in missing], "spike_selftest"],
    runtime_hooks=[str(HERE / "rth_portaudio.py")],
    # pytest: plume refuses to start a real child when pytest is loaded.  The rest is
    # weight nothing here uses.
    excludes=["pytest", "_pytest", "tkinter", "setuptools", "pip", "wheel",
              *([] if WITH_FFMPEG else ["imageio_ffmpeg"])],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="plume",
    console=True,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Plume", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Plume.app",
        bundle_identifier="garden.cyborg.plume",
        info_plist={
            "CFBundleDisplayName": "Plume",
            "CFBundleShortVersionString": "0.1.0",
            "NSMicrophoneUsageDescription": "Plume listens to your playing so the band can follow you.",
            "NSHighResolutionCapable": True,
        },
    )
