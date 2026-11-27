"""Spike: the page inside a pywebview window, checked from outside.  Throwaway: not for main.

    python packaging/spike/spike_webview.py          (or, frozen:  plume --spike-webview)

Opens a real window, so it is for a CI runner or a person who asked for it.  The page's
server runs in this process with a made-up take, a refused Start and a fake file manager.
No audio device is opened here; the one sound is played muted, at volume zero.

The window gets no ``js_api``: the page talks to the server over HTTP and nothing else.
The checks drive the page with ``evaluate_js``, which is this harness, not the product.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

RESULT: dict = {"checks": {}, "requests": []}


def _make_take(folder: Path) -> None:
    import numpy as np
    import soundfile as sf

    from plume.realtime import REC_HUMAN, REC_MACHINE

    folder.mkdir(parents=True)
    t = np.arange(48000 * 3) / 48000.0
    for name, hz in ((REC_HUMAN, 220.0), (REC_MACHINE, 330.0)):
        sf.write(str(folder / name), (0.05 * np.sin(2 * np.pi * hz * t)).astype("float32"), 48000,
                 subtype="PCM_16")


def _first_id(doc):
    if isinstance(doc, dict):
        if isinstance(doc.get("id"), str):
            return doc["id"]
        doc = list(doc.values())
    if isinstance(doc, list):
        for item in doc:
            found = _first_id(item)
            if found:
                return found
    return None


class _Page:
    def __init__(self, window):
        self.window = window

    def now(self, code: str):
        return self.window.evaluate_js(code)

    def later(self, expr: str, wait: float = 30.0):
        """The value of an expression that may be a promise, as JSON."""
        self.now("window.__spike = undefined; (async () => { try { window.__spike = JSON.stringify(await (" + expr +
                 ")); } catch (e) { window.__spike = JSON.stringify({error: String(e)}); } })(); 0")
        until = time.monotonic() + wait
        while time.monotonic() < until:
            got = self.now("window.__spike === undefined ? null : window.__spike")
            if got is not None:
                return json.loads(got)
            time.sleep(0.1)
        return {"error": "timed out"}


def _checks(window, server, folder: Path, revealed: list) -> None:
    checks = RESULT["checks"]
    page = _Page(window)
    try:
        until = time.monotonic() + 60
        while time.monotonic() < until:
            try:
                if page.now("document.readyState === 'complete' && location.pathname === '/' "
                            "&& !!sessionStorage.getItem('plume-key')"):
                    break
            except Exception:  # noqa: BLE001 - the window is still coming up
                pass
            time.sleep(0.2)
        login = page.now("""({
            path: location.pathname, query: location.search, hash_has_key: /k=/.test(location.hash),
            cookie_visible_to_script: /plume_/.test(document.cookie),
            page_key_length: (sessionStorage.getItem('plume-key') || '').length,
            page_drawn: !!document.getElementById('player') && document.body.innerText.length > 0,
            agent: navigator.userAgent })""")
        login["pass"] = (login.get("path") == "/" and not login.get("query") and not login.get("hash_has_key")
                         and not login.get("cookie_visible_to_script") and login.get("page_key_length") == 64
                         and bool(login.get("page_drawn")))
        checks["token link to HttpOnly cookie login"] = login

        key = "sessionStorage.getItem('plume-key')"
        takes = page.later("fetch('/api/takes', {credentials: 'same-origin', headers: {'X-Plume-Token': " + key +
                           "}}).then(r => r.json().then(d => ({status: r.status, doc: d})))")
        take_id = _first_id(takes.get("doc"))
        checks["api GET with cookie and page key"] = {"status": takes.get("status"), "take_listed": bool(take_id),
                                                      "pass": takes.get("status") == 200 and bool(take_id)}
        if not take_id:
            return
        tid = json.dumps(take_id)

        shown = page.later("fetch('/api/reveal', {method: 'POST', credentials: 'same-origin', headers: "
                           "{'Content-Type': 'application/json', 'X-Plume-Token': " + key + "}, "
                           "body: JSON.stringify({take: " + tid + "})})"
                           ".then(r => r.json().then(d => ({status: r.status, doc: d})))")
        mix = folder / "mix.wav"
        checks["saving mix.wav (the page asks, the server writes)"] = {
            "status": shown.get("status"), "doc": shown.get("doc"), "mix_on_disk": mix.is_file(),
            "pass": shown.get("status") == 200 and mix.is_file()}
        checks["reveal-folder (the page asks, the server opens the file manager)"] = {
            "file_manager_asked": len(revealed), "pass": shown.get("status") == 200 and len(revealed) == 1,
            "note": "the file manager here is a fake; the real one is the server's, not the window's"}

        upload = page.later("fetch('/api/video/' + encodeURIComponent(" + tid + ") + '?sound=1', {method: 'POST', "
                            "credentials: 'same-origin', body: new Blob([new Uint8Array(4096)]), headers: "
                            "{'Content-Type': 'application/octet-stream', 'X-Plume-Token': " + key + "}})"
                            ".then(r => r.json().catch(() => null).then(d => ({status: r.status, doc: d})))")
        box = page.now("(() => { const b = document.getElementById('video-file'); "
                       "return b ? {type: b.type, accept: b.accept} : null; })()")
        checks["file upload ('Add a video')"] = {
            "upload_status": upload.get("status"), "upload_doc": upload.get("doc"), "file_input": box,
            # 401, 403 and 415 are the server's own refusals of the request's shape.
            "pass": upload.get("status") not in (None, 401, 403, 415) and bool(box),
            "note": "the bytes reach the server from inside the window; the file chooser itself "
                    "was not opened, which needs a hand"}
        page.later("fetch('/api/video/cancel', {method: 'POST', credentials: 'same-origin', headers: "
                   "{'Content-Type': 'application/json', 'X-Plume-Token': " + key + "}, "
                   "body: JSON.stringify({take: " + tid + "})}).then(r => r.status)")

        url = "'/api/take/' + encodeURIComponent(" + tid + ") + '/audio/together?k=' + " + key
        loaded = page.later("new Promise((done) => { const a = new Audio(" + url + "); a.muted = true; a.volume = 0; "
                            "a.preload = 'auto'; window.__audio = a; "
                            "a.addEventListener('canplay', () => done({duration: a.duration, ready: a.readyState})); "
                            "a.addEventListener('error', () => done({error: a.error && a.error.code})); a.load(); })")
        played = page.later("(async () => { const a = window.__audio; a.muted = true; a.volume = 0; "
                            "try { await a.play(); } catch (e) { return {play_refused: String(e)}; } "
                            "await new Promise(r => setTimeout(r, 1200)); const at = a.currentTime; a.pause(); "
                            "return {moved_s: at}; })()")
        checks["audio playback (muted)"] = {
            "loaded": loaded, "played": played,
            "pass": isinstance(loaded.get("duration"), (int, float)) and loaded["duration"] > 0
                    and isinstance(played.get("moved_s"), (int, float)) and played["moved_s"] > 0,
            "note": "play() here has no click behind it; in the app a person clicks Listen"}

        bridge = page.now("({pywebview: typeof window.pywebview, api_names: window.pywebview && window.pywebview.api "
                          "? Object.keys(window.pywebview.api) : null})")
        checks["no JS bridge"] = {**(bridge or {}), "pass": not (bridge or {}).get("api_names"),
                                  "note": "no js_api is passed; what the toolkit injects by itself is listed"}

        seen = RESULT["requests"]
        posts = [r for r in seen if r["method"] == "POST"]
        gets = [r for r in seen if r["method"] == "GET"]
        checks["Host / Origin / Sec-Fetch-Site checks"] = {
            "requests": len(seen), "refused_by_these_checks": sum(1 for r in seen if r["refused"]),
            "posts_with_origin": f"{sum(1 for r in posts if r['origin'])}/{len(posts)}",
            "sec_fetch_site_values_on_get": sorted({str(r["site"]) for r in gets}),
            "sec_fetch_site_values_on_post": sorted({str(r["site"]) for r in posts}),
            "pass": bool(posts) and all(r["origin"] for r in posts) and not any(r["refused"] for r in seen)}
    except Exception as e:  # noqa: BLE001
        RESULT["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
    finally:
        window.destroy()


def run() -> int:
    import webview

    with tempfile.TemporaryDirectory(prefix="plume-spike-") as name:
        tmp = Path(name)
        os.environ["PLUME_GUI_STATE_DIR"] = str(tmp / "state")
        from plume import gui, guitake

        def no_spawn(argv, log_path):  # noqa: ARG001
            raise guitake.TakeError("the spike starts no take")

        revealed: list = []
        folder = tmp / "library" / "2026-01-01-000000"
        _make_take(folder)

        inner = gui._Handler._check_common  # noqa: SLF001

        def recorded(self, *, allow_none: bool = False):
            row = {"method": self.command, "route": getattr(self, "route", "?"),
                   "origin": bool(self.headers.get("Origin")), "site": self.headers.get("Sec-Fetch-Site"),
                   "refused": False}
            RESULT["requests"].append(row)
            try:
                inner(self, allow_none=allow_none)
            except Exception:
                row["refused"] = True
                raise

        gui._Handler._check_common = recorded  # noqa: SLF001
        server = gui.GuiServer(roots=[tmp / "library"], write_state=False, log=lambda line: None,
                               spawner=no_spawn, lister=lambda: [],
                               revealer=lambda where, select: revealed.append(select)).start()
        try:
            RESULT["pywebview"] = getattr(webview, "__version__", "?")
            RESULT["frozen"] = bool(getattr(sys, "frozen", False))
            window = webview.create_window("Plume", server.first_url(), width=1100, height=760)
            # A wedged window must not hold a runner for its whole time limit.
            watchdog = threading.Timer(180.0, lambda: (RESULT.setdefault("error", "timed out"), os._exit(3)))
            watchdog.daemon = True
            watchdog.start()
            began = time.perf_counter()
            webview.start(_checks, (window, server, folder, revealed))
            RESULT["window_seconds"] = round(time.perf_counter() - began, 2)
        finally:
            server.close()
    checks = RESULT["checks"]
    RESULT["ok"] = bool(checks) and all(c.get("pass") for c in checks.values()) and "error" not in RESULT
    RESULT["requests"] = RESULT["requests"][:80]
    print(json.dumps(RESULT, indent=1, default=str))
    return 0 if RESULT["ok"] else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.exit(run())
