"""Spike: the page inside a pywebview window, checked from outside.  Throwaway: not for main.

    python packaging/spike/spike_webview.py          (or, frozen:  plume --spike-webview)

Opens a real window, so it is for a CI runner or a person who asked for it.  The page's
server runs in this process with a made-up take, a refused Start and a fake file manager.
No audio device is opened here; the one sound is played muted, at volume zero.

The window gets no ``js_api``: the page talks to the server over HTTP and nothing else.
The checks are a script the page loads from its own server, for this spike only.
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

RESULT: dict = {"checks": {}, "requests": [], "responses": []}


def _make_take(folder: Path) -> None:
    import numpy as np
    import soundfile as sf

    from plume.realtime import REC_HUMAN, REC_MACHINE

    folder.mkdir(parents=True)
    t = np.arange(48000 * 3) / 48000.0
    for name, hz in ((REC_HUMAN, 220.0), (REC_MACHINE, 330.0)):
        sf.write(str(folder / name), (0.05 * np.sin(2 * np.pi * hz * t)).astype("float32"), 48000,
                 subtype="PCM_16")


#: The checks, as a script the page loads from its own server.  The page's content policy
#: allows scripts from itself only, and refuses eval, which is how a window toolkit runs
#: script from outside.  So the checks ride in as one more static file, for this spike only,
#: and report back in the query of an ordinary request.
PAGE_SCRIPT = r"""
(async () => {
  const out = {};
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const key = () => { try { return sessionStorage.getItem('plume-key') || ''; } catch (e) { return ''; } };
  const get = (path) => fetch(path, { credentials: 'same-origin', headers: { 'X-Plume-Token': key() } });
  const post = (path, body, type) => fetch(path, {
    method: 'POST', credentials: 'same-origin', body: body,
    headers: { 'Content-Type': type || 'application/json', 'X-Plume-Token': key() } });
  const answer = async (r) => ({ status: r.status, doc: await r.json().catch(() => null) });
  const firstId = (doc) => {
    if (doc && typeof doc === 'object') {
      if (typeof doc.id === 'string') return doc.id;
      for (const v of Object.values(doc)) { const f = firstId(v); if (f) return f; }
    }
    return null;
  };
  try {
    for (let i = 0; i < 300 && !key(); i++) await sleep(100);
    out.login = {
      path: location.pathname, query: location.search, hash_has_key: /k=/.test(location.hash),
      cookie_visible_to_script: /plume_/.test(document.cookie), page_key_length: key().length,
      page_drawn: !!document.getElementById('player') && document.body.innerText.length > 0,
      agent: navigator.userAgent };
    const takes = await answer(await get('/api/takes'));
    const id = firstId(takes.doc);
    out.takes = { status: takes.status, take_listed: !!id };
    if (id) {
      const where = encodeURIComponent(id);
      out.reveal = await answer(await post('/api/reveal', JSON.stringify({ take: id })));
      const box = document.getElementById('video-file');
      out.upload = await answer(await post('/api/video/' + where + '?sound=1',
                                           new Blob([new Uint8Array(4096)]), 'application/octet-stream'));
      out.upload.file_input = box ? { type: box.type, accept: box.accept } : null;
      await post('/api/video/cancel', JSON.stringify({ take: id })).catch(() => null);
      out.audio = {};
      for (const track of ['together', 'human']) {
        const a = new Audio('/api/take/' + where + '/audio/' + track + '?k=' + key());
        a.muted = true; a.volume = 0; a.preload = 'auto';
        const got = await Promise.race([
          new Promise((done) => {
            a.addEventListener('canplay', () => done({ duration: a.duration, ready: a.readyState }));
            a.addEventListener('error', () => done({ error: a.error && a.error.code,
                                                     message: a.error && a.error.message }));
            a.load();
          }),
          sleep(20000).then(() => ({ error: 'timed out' }))]);
        try {
          await a.play();
          await sleep(1200);
          got.moved_s = a.currentTime;
          a.pause();
        } catch (e) { got.play_refused = String(e); }
        out.audio[track] = got;
      }
    }
    out.bridge = { pywebview: typeof window.pywebview,
                   api_names: window.pywebview && window.pywebview.api ? Object.keys(window.pywebview.api) : null };
  } catch (e) { out.error = String((e && e.stack) || e); }
  await get('/api/guide?spike=' + encodeURIComponent(JSON.stringify(out))).catch(() => null);
})();
"""


def _judge(out: dict, folder: Path, revealed: list) -> None:
    checks = RESULT["checks"]
    login = dict(out.get("login") or {})
    login["pass"] = (login.get("path") == "/" and not login.get("query") and not login.get("hash_has_key")
                     and not login.get("cookie_visible_to_script") and login.get("page_key_length") == 64
                     and bool(login.get("page_drawn")))
    checks["token link to HttpOnly cookie login"] = login
    takes = out.get("takes") or {}
    checks["api GET with cookie and page key"] = {**takes, "pass": takes.get("status") == 200
                                                  and bool(takes.get("take_listed"))}
    shown = out.get("reveal") or {}
    mix = folder / "mix.wav"
    checks["saving mix.wav (the page asks, the server writes)"] = {
        "status": shown.get("status"), "doc": shown.get("doc"), "mix_on_disk": mix.is_file(),
        "pass": shown.get("status") == 200 and mix.is_file()}
    checks["reveal-folder (the page asks, the server opens the file manager)"] = {
        "file_manager_asked": len(revealed), "pass": shown.get("status") == 200 and len(revealed) == 1,
        "note": "the file manager here is a fake; the real one is the server's, not the window's"}
    upload = out.get("upload") or {}
    checks["file upload ('Add a video')"] = {
        **upload,
        # 401, 403 and 415 are the server's own refusals of the request's shape.
        "pass": upload.get("status") not in (None, 401, 403, 415) and bool(upload.get("file_input")),
        "note": "the bytes reach the server from inside the window; the file chooser itself "
                "was not opened, which needs a hand"}
    audio = out.get("audio") or {}

    def plays(got) -> bool:
        return (isinstance(got, dict) and isinstance(got.get("duration"), (int, float)) and got["duration"] > 0
                and isinstance(got.get("moved_s"), (int, float)) and got["moved_s"] > 0)

    checks["audio playback (muted)"] = {
        **audio,
        "audio_requests": [r for r in RESULT["responses"] if "/audio/" in r["path"]],
        "pass": bool(audio) and all(plays(got) for got in audio.values()),
        "note": "play() here has no click behind it; in the app a person clicks Listen"}
    bridge = out.get("bridge") or {}
    checks["no JS bridge"] = {**bridge, "pass": bool(bridge) and not bridge.get("api_names"),
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
    if out.get("error"):
        RESULT["error"] = out["error"]


def _checks(window, folder: Path, revealed: list) -> None:
    try:
        until = time.monotonic() + 150
        while "page" not in RESULT and time.monotonic() < until:
            time.sleep(0.2)
        if "page" not in RESULT:
            RESULT["error"] = "the page's script never reported"
            try:
                RESULT["url_at_timeout"] = str(window.get_current_url()).split("?")[0]
            except Exception as e:  # noqa: BLE001
                RESULT["url_at_timeout"] = f"{type(e).__name__}: {e}"
        _judge(RESULT.pop("page", {}), folder, revealed)
    except Exception as e:  # noqa: BLE001
        RESULT["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
    finally:
        window.destroy()


def _static_with_script(tmp: Path) -> Path:
    """A copy of the page's files with the checks added as one more script."""
    import shutil

    from plume import gui

    copy = tmp / "gui_static"
    shutil.copytree(gui._STATIC_DIR, copy)  # noqa: SLF001
    index = copy / "index.html"
    text = index.read_text(encoding="utf-8")
    index.write_text(text.replace("</body>", '<script src="/static/spike.js"></script>\n</body>'), encoding="utf-8")
    (copy / "spike.js").write_text(PAGE_SCRIPT, encoding="utf-8")
    return copy


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
            if "spike=" in self.path:
                from urllib.parse import parse_qs, urlsplit

                RESULT["page"] = json.loads(parse_qs(urlsplit(self.path).query)["spike"][0])
                return inner(self, allow_none=allow_none)
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
        sent = gui._Handler.send_response  # noqa: SLF001

        def send_response(self, code, message=None):
            path = self.path.split("?", 1)[0]
            RESULT["responses"].append({
                "method": self.command, "path": "/".join(p if len(p) < 24 else "{id}" for p in path.split("/")),
                "status": code, "cookie": bool(self.headers.get("Cookie")),
                "page_key_in_query": "k=" in self.path, "range": self.headers.get("Range"),
                "site": self.headers.get("Sec-Fetch-Site"), "dest": self.headers.get("Sec-Fetch-Dest")})
            return sent(self, code, message)

        gui._Handler.send_response = send_response  # noqa: SLF001
        gui._STATIC_DIR = _static_with_script(tmp)  # noqa: SLF001
        gui.STATIC_FILES["spike.js"] = "text/javascript; charset=utf-8"
        server = gui.GuiServer(roots=[tmp / "library"], write_state=False, log=lambda line: None,
                               spawner=no_spawn, lister=lambda: [],
                               revealer=lambda where, select: revealed.append(select)).start()
        try:
            RESULT["pywebview"] = getattr(webview, "__version__", "?")
            RESULT["frozen"] = bool(getattr(sys, "frozen", False))
            window = webview.create_window("Plume", server.first_url(), width=1100, height=760)
            # A wedged window must not hold a runner for its whole time limit.
            watchdog = threading.Timer(240.0, lambda: (RESULT.setdefault("error", "timed out"), os._exit(3)))
            watchdog.daemon = True
            watchdog.start()
            began = time.perf_counter()
            # PLUME_SPIKE_PRIVATE=0: a window whose cookies and storage are kept on disk.
            private = os.environ.get("PLUME_SPIKE_PRIVATE", "1") != "0"
            RESULT["private_mode"] = private
            webview.start(_checks, (window, folder, revealed), private_mode=private,
                          storage_path=None if private else str(tmp / "webview-storage"))
            RESULT["window_seconds"] = round(time.perf_counter() - began, 2)
        finally:
            server.close()
    checks = RESULT["checks"]
    RESULT["ok"] = bool(checks) and all(c.get("pass") for c in checks.values()) and "error" not in RESULT
    del RESULT["requests"], RESULT["responses"]
    print(json.dumps(RESULT, indent=1, default=str))
    return 0 if RESULT["ok"] else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.exit(run())
