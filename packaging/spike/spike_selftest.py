"""Spike self-test: runs inside the frozen bundle.  Throwaway: not for main.

    plume --spike-selftest [--ci] [--expect-ffmpeg=0|1] [--out=result.json]

Opens no audio stream and makes no sound.  ``--ci`` adds the checks that touch the
audio and MIDI drivers to list what is there (a runner has nothing to list); without
it those are skipped.  The take is a simulated one: ``plume simlead`` on a rendered WAV.
"""
from __future__ import annotations

import http.client
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

RESULTS: list[dict] = []


def check(name):
    def wrap(fn):
        fn.check_name = name
        return fn
    return wrap


def _run_check(fn, *args) -> None:
    began = time.perf_counter()
    try:
        detail = fn(*args)
        status = "skip" if isinstance(detail, str) and detail.startswith("skip") else "pass"
    except Exception as e:  # noqa: BLE001 - every failure is a result
        detail = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
        status = "fail"
    RESULTS.append({"check": fn.check_name, "status": status,
                    "seconds": round(time.perf_counter() - began, 3), "detail": detail})


def _child(*args: str):
    from plume import guitake

    return [*guitake.PLUME_CHILD, *args]


def _popen_like_start(argv, log_path: Path):
    """The page's own way of starting a take (guitake.default_spawner), minus the
    consent to open an audio device: this child must not have it."""
    from plume import oscompat

    env = {k: v for k, v in os.environ.items() if k != "PLUME_ALLOW_AUDIO"}
    with open(log_path, "wb") as log:
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log,
                                env={**env, **oscompat.own_group_env()}, **oscompat.detached_kwargs())


# ------------------------------------------------------------------------ the checks

@check("frozen")
def frozen():
    if not getattr(sys, "frozen", False):
        raise RuntimeError("not a frozen build")
    return {"executable": Path(sys.executable).name, "python": sys.version.split()[0],
            "platform": sys.platform, "machine": __import__("platform").machine()}


@check("import every module")
def imports():
    listing = Path(sys._MEIPASS) / "spike" / "spike_modules.txt"  # noqa: SLF001
    names = [n for n in listing.read_text(encoding="utf-8").split() if n]
    failed = {}
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            failed[name] = f"{type(e).__name__}: {e}"
    if failed:
        raise RuntimeError(json.dumps(failed))
    return {"imported": len(names)}


@check("lazy names on the package (plume/__init__.py)")
def lazy_names():
    import plume

    return {name: type(getattr(plume, name)).__name__ for name in plume._LAZY}  # noqa: SLF001


@check("doctor's imports (cli.py)")
def doctor_imports():
    seen = {}
    for mod in ("numpy", "scipy", "soundfile", "sounddevice"):
        seen[mod] = getattr(importlib.import_module(mod), "__version__", "?")
    for mod in ("theory", "notation", "score", "styles_lib", "groove", "compose", "voices", "fx",
                "engine", "analyze"):
        importlib.import_module(f"plume.{mod}")
    return seen


@check("midi library by name (midi.py)")
def midi_import(ci: bool):
    from plume import midi

    mido = midi._import_mido()  # noqa: SLF001
    importlib.import_module("mido.backends.rtmidi")
    import rtmidi

    out = {"mido": str(getattr(mido, "version_info", "?")), "rtmidi": getattr(rtmidi, "__version__", "?")}
    if ci:
        out["apis"] = [rtmidi.get_api_name(a) for a in rtmidi.get_compiled_api()]
        try:
            out["input_ports"] = mido.get_input_names()
        except Exception as e:  # noqa: BLE001 - a runner may have no MIDI system at all
            out["input_ports_error"] = f"{type(e).__name__}: {e}"
    return out


@check("data files")
def data_files():
    from plume import gui, samplekit, styles_lib
    from plume.listen import notes

    styles = sorted(p.name for p in styles_lib._STYLES_DIR.glob("*.json"))  # noqa: SLF001
    samples = [p for p in samplekit._SAMPLES.rglob("*") if p.is_file()]  # noqa: SLF001
    missing = [n for n in gui.STATIC_FILES if not (gui._STATIC_DIR / n).is_file()]  # noqa: SLF001
    pages = gui.guide_pages()
    if not styles or not samples or missing or len(pages) != len(gui.GUIDE_PAGES) or not notes.MODEL_PATH.is_file():
        raise RuntimeError(f"styles={len(styles)} samples={len(samples)} missing_static={missing} "
                           f"guide={len(pages)} model={notes.MODEL_PATH.is_file()}")
    return {"styles": len(styles), "samples": len(samples), "static": len(gui.STATIC_FILES),
            "guide": len(pages), "model_bytes": notes.MODEL_PATH.stat().st_size}


@check("pitch model runs (onnxruntime)")
def onnx():
    import numpy as np

    from plume.listen import notes

    if not notes.notes_available():
        raise RuntimeError("notes_available() is False")
    session = notes._session(notes.MODEL_PATH)  # noqa: SLF001
    feed = {}
    for item in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in item.shape]
        feed[item.name] = np.zeros(shape, dtype=np.float32)
    outs = session.run(None, feed)
    return {"outputs": [list(o.shape) for o in outs]}


@check("soundfile writes and reads a WAV")
def soundfile_roundtrip(tmp: Path):
    import numpy as np
    import soundfile as sf

    path = tmp / "roundtrip.wav"
    sf.write(str(path), np.zeros((4800, 2), dtype="float32"), 48000, subtype="PCM_16")
    data, rate = sf.read(str(path))
    if data.shape != (4800, 2) or rate != 48000:
        raise RuntimeError(f"{data.shape} at {rate}")
    return {"libsndfile": sf.__libsndfile_version__}


@check("audio library lists devices (no stream opened)")
def devices(ci: bool):
    if not ci:
        return "skip: only with --ci"
    import sounddevice as sd

    listed = sd.query_devices()
    return {"portaudio": sd.get_portaudio_version()[1], "devices": len(listed),
            "host_apis": [a["name"] for a in sd.query_hostapis()]}


@check("ffmpeg")
def ffmpeg(expect: bool | None):
    from plume import takevideo

    try:
        import imageio_ffmpeg  # noqa: F401
        packaged = True
    except ImportError:
        packaged = False
    exe = takevideo.find_ffmpeg()
    inside = bool(exe) and Path(exe).resolve().is_relative_to(Path(sys._MEIPASS).resolve())  # noqa: SLF001
    path, os.environ["PATH"] = os.environ.get("PATH", ""), ""
    try:
        bare = takevideo.find_ffmpeg()
    finally:
        os.environ["PATH"] = path
    out = {"imageio_ffmpeg_importable": packaged, "found": bool(exe), "inside_bundle": inside,
           "found_with_an_empty_path": bool(bare)}
    if exe:
        done = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=30)
        out["version"] = done.stdout.splitlines()[0] if done.stdout else f"exit {done.returncode}"
        if done.returncode != 0:
            raise RuntimeError(f"ffmpeg -version exited {done.returncode}: {done.stderr[:200]}")
    if expect is True and not inside:
        raise RuntimeError(f"expected the bundle's own ffmpeg: {out}")
    if expect is False and packaged:
        raise RuntimeError("imageio_ffmpeg is in a build that should not have it")
    return out


@check("child launch: the page's argv starts this program")
def child_launch():
    from plume import oscompat

    done = subprocess.run(_child("--spike-child-report"), stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=120, env={**os.environ, **oscompat.own_group_env()},
                          **oscompat.detached_kwargs())
    if done.returncode != 0:
        raise RuntimeError(f"exit {done.returncode}: {done.stderr[-400:]}")
    doc = json.loads(done.stdout)
    same = os.path.samefile(doc["executable"], sys.executable)
    if not (doc["frozen"] and same and doc["parent"] == os.getpid() and doc["leads_own_group"]):
        raise RuntimeError(f"{doc} (this pid {os.getpid()})")
    doc["executable"] = Path(doc["executable"]).name
    doc["parent_is_this_process"] = True
    del doc["parent"]
    return doc


@check("child launch: a plume command through that argv")
def child_command():
    done = subprocess.run(_child("styles", "--json"), stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=120)
    if done.returncode != 0:
        raise RuntimeError(f"exit {done.returncode}: {done.stderr[-400:]}")
    json.loads(done.stdout)
    return {"stdout_bytes": len(done.stdout)}


@check("child launch: 'inputs --json' (what the page asks at load)")
def child_inputs(ci: bool):
    if not ci:
        return "skip: only with --ci"
    from plume import guitake, oscompat

    done = subprocess.run(_child("inputs", "--json"), stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=guitake.LIST_TIMEOUT_S * 6, **oscompat.detached_kwargs())
    doc = json.loads(done.stdout)
    return {"exit": done.returncode, "keys": sorted(doc), "devices": len(doc.get("devices") or []),
            "error": str(doc.get("error", ""))[:200]}


@check("child stop: the interrupt the page sends reaches the child")
def child_interrupt(tmp: Path):
    from plume import oscompat

    marker = tmp / "interrupted.txt"
    ready = tmp / "ready.txt"
    proc = _popen_like_start(_child("--spike-wait", str(ready), str(marker)), tmp / "wait.log")
    try:
        deadline = time.monotonic() + 60
        while not ready.exists():
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"child never got ready (exit {proc.poll()}): "
                                   f"{(tmp / 'wait.log').read_text(errors='replace')[-300:]}")
            time.sleep(0.05)
        oscompat.interrupt(proc)
        code = proc.wait(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
    if not marker.exists():
        raise RuntimeError(f"child exited {code} without handling the interrupt")
    return {"exit": code, "handled": marker.read_text()}


@check("simulated take from the bundle (make, render, simlead as detached children)")
def simulated_take(tmp: Path):
    import soundfile as sf

    work = tmp / "take"
    work.mkdir()
    session, human, mix = work / "s.plume.json", work / "human.wav", work / "mix.wav"
    steps = (
        ("make", _child("make", "slow folk at 90", f"--session={session}", "--json")),
        ("render", _child("render", f"--session={session}", f"--out={human}", "--bars", "0", "8", "--json")),
        ("simlead", _child("simlead", str(human), "--prompt", "chill trap at 90", "--bars", "8",
                           f"--out={mix}", "--json")),
    )
    timing = {}
    for name, argv in steps:
        began = time.perf_counter()
        proc = _popen_like_start(argv, work / f"{name}.log")
        code = proc.wait(timeout=600)
        timing[name] = round(time.perf_counter() - began, 2)
        if code != 0:
            raise RuntimeError(f"{name} exited {code}: "
                               f"{(work / f'{name}.log').read_text(errors='replace')[-600:]}")
    decisions = mix.with_suffix(".ndjson")
    with sf.SoundFile(str(mix)) as f:
        frames, rate = f.frames, f.samplerate
    lines = sum(1 for _ in open(decisions, encoding="utf-8"))
    if frames <= 0 or lines <= 0:
        raise RuntimeError(f"frames={frames} decisions={lines}")
    return {"seconds_of_audio": round(frames / rate, 2), "decision_lines": lines, "step_seconds": timing}


@check("reverse lookup of 127.0.0.1 (http.server does this at bind; timed, never fails)")
def reverse_lookup():
    import socket

    began = time.perf_counter()
    name = socket.getfqdn("127.0.0.1")
    return {"seconds": round(time.perf_counter() - began, 2), "is_localhost": name == "localhost"}


def _get(port: int, path: str, headers: dict | None = None, *, host: str | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    try:
        conn.putrequest("GET", path, skip_host=True)
        conn.putheader("Host", host or f"127.0.0.1:{port}")
        for k, v in (headers or {}).items():
            conn.putheader(k, v)
        conn.endheaders()
        r = conn.getresponse()
        return r.status, dict(r.getheaders()), r.read()
    finally:
        conn.close()


@check("the page's server, in this process (no take started, no device listed)")
def gui_http(tmp: Path):
    os.environ["PLUME_GUI_STATE_DIR"] = str(tmp / "state")
    from plume import gui, guitake

    def no_spawn(argv, log_path):  # noqa: ARG001
        raise guitake.TakeError("the self-test starts no take")

    library = tmp / "library"
    library.mkdir()
    stages = {}
    began = time.perf_counter()
    server = gui.GuiServer(roots=[library], write_state=False, log=lambda line: None,
                           spawner=no_spawn, lister=lambda: []).start()
    stages["start"] = round(time.perf_counter() - began, 2)
    try:
        began = time.perf_counter()
        port = server.port
        path = server.first_url().split(str(port), 1)[1]
        status, headers, _ = _get(port, path, {"Sec-Fetch-Site": "none"})
        cookie = headers.get("Set-Cookie", "")
        where = headers.get("Location", "")
        if status != 303 or "HttpOnly" not in cookie or "SameSite=Strict" not in cookie or "#k=" not in where:
            raise RuntimeError(f"auth: {status} cookie flags or redirect wrong")
        jar = {"Cookie": cookie.split(";", 1)[0]}
        key = where.split("#k=", 1)[1]
        seen = {"auth": status}
        seen["token_twice"] = _get(port, path, {"Sec-Fetch-Site": "none"})[0]
        seen["page"] = _get(port, "/", jar)[0]
        statics = {name: _get(port, f"/static/{name}", jar)[0] for name in gui.STATIC_FILES}
        seen["static_ok"] = sum(1 for s in statics.values() if s == 200)
        seen["guide"] = _get(port, "/api/guide", {**jar, "X-Plume-Token": key})[0]
        seen["guide_no_page_key"] = _get(port, "/api/guide", jar)[0]
        seen["guide_no_cookie"] = _get(port, "/api/guide", {"X-Plume-Token": key})[0]
        seen["wrong_host"] = _get(port, "/", jar, host=f"localhost:{port}")[0]
        seen["cross_site"] = _get(port, "/api/guide", {**jar, "X-Plume-Token": key,
                                                       "Sec-Fetch-Site": "cross-site"})[0]
        seen["wrong_origin"] = _get(port, "/api/guide", {**jar, "X-Plume-Token": key,
                                                         "Origin": "http://example.invalid"})[0]
        stages["requests"] = round(time.perf_counter() - began, 2)
    finally:
        began = time.perf_counter()
        server.close()
        stages["close"] = round(time.perf_counter() - began, 2)
    want = {"auth": 303, "token_twice": 403, "page": 200, "static_ok": len(gui.STATIC_FILES), "guide": 200,
            "guide_no_page_key": 401, "guide_no_cookie": 401, "wrong_host": 403, "cross_site": 403,
            "wrong_origin": 403}
    if seen != want:
        raise RuntimeError(f"got {seen}, want {want}")
    return {**seen, "stage_seconds": stages}


@check("cold start: a fresh process until the page answers (three runs, seconds)")
def page_ready_time():
    runs = []
    for _ in range(3):
        began = time.perf_counter()
        done = subprocess.run(_child("--spike-page-ready"), stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=300)
        runs.append(round(time.perf_counter() - began, 2))
        if done.returncode != 0 or json.loads(done.stdout).get("page") != 200:
            raise RuntimeError(f"exit {done.returncode}: {done.stdout[-200:]} {done.stderr[-400:]}")
    return {"runs": runs, "inside_child": json.loads(done.stdout)}


# ------------------------------------------------------------------------ child verbs

def _child_page_ready() -> int:
    """What the app does before a browser can show anything: import the page's server,
    bind it, and answer the first two requests.  No take, no device list."""
    began = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="plume-spike-") as name:
        os.environ["PLUME_GUI_STATE_DIR"] = str(Path(name) / "state")
        from plume import gui, guitake

        def no_spawn(argv, log_path):  # noqa: ARG001
            raise guitake.TakeError("the self-test starts no take")

        imported = time.perf_counter()
        server = gui.GuiServer(roots=[Path(name)], write_state=False, log=lambda line: None,
                               spawner=no_spawn, lister=lambda: []).start()
        try:
            bound = time.perf_counter()
            path = server.first_url().split(str(server.port), 1)[1]
            _, headers, _ = _get(server.port, path, {"Sec-Fetch-Site": "none"})
            status = _get(server.port, "/", {"Cookie": headers.get("Set-Cookie", "").split(";", 1)[0]})[0]
            answered = time.perf_counter()
        finally:
            server.close()
    print(json.dumps({"page": status, "import_s": round(imported - began, 2),
                      "bind_s": round(bound - imported, 2), "answer_s": round(answered - bound, 2)}))
    return 0


def _child_report() -> int:
    from plume import oscompat

    print(json.dumps({"frozen": bool(getattr(sys, "frozen", False)), "executable": sys.executable,
                      "parent": os.getppid(), "leads_own_group": oscompat.leads_own_group(),
                      "argv_after_launcher": sys.argv[1:]}))
    return 0


def _child_wait(ready: str, marker: str) -> int:
    def handled(signum, frame):  # noqa: ARG001
        Path(marker).write_text(signal.Signals(signum).name)
        raise SystemExit(0)

    for name in ("SIGINT", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handled)
    Path(ready).write_text("ready")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        time.sleep(0.05)
    return 3


# ------------------------------------------------------------------------ entry

def run(argv: list[str]) -> int:
    verb = argv[0]
    if verb == "--spike-child-report":
        return _child_report()
    if verb == "--spike-page-ready":
        return _child_page_ready()
    if verb == "--spike-wait":
        return _child_wait(argv[1], argv[2])
    if verb != "--spike-selftest":
        print(f"unknown spike verb {verb}", file=sys.stderr)
        return 2
    ci = "--ci" in argv
    if os.environ.get("PLUME_SPIKE_TRACE"):
        import faulthandler

        faulthandler.dump_traceback_later(float(os.environ["PLUME_SPIKE_TRACE"]), repeat=True)
    expect = None
    out = None
    for a in argv[1:]:
        if a.startswith("--expect-ffmpeg="):
            expect = a.endswith("=1")
        if a.startswith("--out="):
            out = a.split("=", 1)[1]
    began = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="plume-spike-") as name:
        tmp = Path(name)
        _run_check(frozen)
        _run_check(imports)
        _run_check(lazy_names)
        _run_check(doctor_imports)
        _run_check(midi_import, ci)
        _run_check(data_files)
        _run_check(onnx)
        _run_check(soundfile_roundtrip, tmp)
        _run_check(devices, ci)
        _run_check(ffmpeg, expect)
        _run_check(child_launch)
        _run_check(child_command)
        _run_check(child_inputs, ci)
        _run_check(child_interrupt, tmp)
        _run_check(simulated_take, tmp)
        _run_check(gui_http, tmp)
        _run_check(page_ready_time)
        _run_check(reverse_lookup)
    failed = [r["check"] for r in RESULTS if r["status"] == "fail"]
    doc = {"ok": not failed, "failed": failed, "seconds": round(time.perf_counter() - began, 2),
           "results": RESULTS}
    text = json.dumps(doc, indent=1, default=str)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    print(text)
    return 1 if failed else 0
