"""Runtime hook: let sounddevice find the PortAudio the bundle carries on Linux.

sounddevice asks ``ctypes.util.find_library('portaudio')``, which looks at the
system's library cache and never inside a frozen folder.
"""
import ctypes.util
import os
import sys

if sys.platform.startswith("linux"):
    _bundled = os.path.join(getattr(sys, "_MEIPASS", ""), "libportaudio.so.2")
    if os.path.exists(_bundled):
        _find = ctypes.util.find_library
        ctypes.util.find_library = lambda name: _bundled if name == "portaudio" else _find(name)
