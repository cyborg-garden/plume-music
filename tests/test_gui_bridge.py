"""The bridge: what a website, or another program, can and cannot do to it.

HTTP lets any page in any browser try to reach this, unlike the unix socket, and
even v0.1 serves recordings of a room.  So every check here applies to every
route.  It uses a real server on a real loopback port, with synthetic folders.
No audio device is opened and nothing is played: the "WAV" is a few bytes.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import signal
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from plume import gui, realtime, record
from record_fixture import POISON, cue_line, ending_doc, landed, song, write_take

WAV = bytes(range(256)) * 4


class Client:
    """Plain http.client, so no header is added that a test did not ask for."""

    def __init__(self, server):
        self.server = server
        self.host = f"127.0.0.1:{server.port}"
        self.origin = f"http://{self.host}"
        self.cookie = None
        self.csrf = None

    def request(self, method, path, *, headers=None, body=None, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=5)
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        send = {"Host": self.host if host is None else host}
        if self.cookie:
            send["Cookie"] = self.cookie
        if self.csrf:
            send["X-Plume-Token"] = self.csrf
        send.update(headers or {})
        for name, value in send.items():
            if value is not None:
                conn.putheader(name, value)
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        if data is not None:
            conn.putheader("Content-Length", str(len(data)))
        conn.endheaders(data)
        resp = conn.getresponse()
        out = resp.read()
        conn.close()
        return resp, out

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def login(self):
        resp, _ = self.get(f"/auth?token={self.server.mint_token()}",
                           headers={"Sec-Fetch-Site": "none"})
        assert resp.status == 303
        self.cookie = resp.getheader("Set-Cookie").split(";", 1)[0]
        # The page key comes in the fragment, the one part of a URL no server receives.
        self.csrf = re.fullmatch(r"/#k=([0-9a-f]{64})", resp.getheader("Location")).group(1)
        resp, out = self.get("/api/session", headers={"Sec-Fetch-Site": "same-origin"})
        assert resp.status == 200
        assert json.loads(out)["csrf"] == self.csrf
        return self

    def post(self, path, body, **headers):
        send = {"Origin": self.origin, "Content-Type": "application/json",
                "X-Plume-Token": self.csrf, "Sec-Fetch-Site": "same-origin"}
        send.update(headers)
        return self.request("POST", path, headers=send, body=body)


@pytest.fixture
def takes(tmp_path):
    root = tmp_path / "library"
    rows = song(40, cues={20: landed("busier")})
    write_take(root / "one", rows, ending=ending_doc(), cues=[cue_line("busier", 18, rows[18]["t_s"])],
               wav_bytes=WAV)
    write_take(root / "two", (), ending=ending_doc(bars=0))
    (root / "not-a-take").mkdir()
    return root


@pytest.fixture
def server(takes, tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state"))
    logged = []
    srv = gui.GuiServer(roots=[takes], log=logged.append).start()
    srv.logged = logged
    yield srv
    srv.close()


@pytest.fixture
def anon(server):
    return Client(server)


@pytest.fixture
def client(server):
    return Client(server).login()


def take_id(client, folder="one"):
    _, out = client.get("/api/takes")
    return next(t["id"] for t in json.loads(out)["takes"] if t["folder"] == folder)


# ---------------------------------------------------------------- where it listens

def test_it_binds_loopback_on_a_random_port(server):
    host, port = server.httpd.server_address[:2]
    assert host == "127.0.0.1" and port != 0


def test_two_servers_get_different_ports(takes, server):
    other = gui.GuiServer(roots=[takes], log=lambda line: None, write_state=False).start()
    try:
        assert other.port != server.port
    finally:
        other.close()


# --------------------------------------------------------------- DNS rebinding: Host

@pytest.mark.parametrize("host", ["evil.example.com", "evil.example.com:80", "localhost",
                                  "127.0.0.1", "127.0.0.1:1", "0.0.0.0", "[::1]", "127.0.0.1.evil.example"])
def test_a_wrong_host_is_refused_even_with_a_good_cookie(client, host):
    for path in ("/", "/api/takes", "/static/app.js", f"/auth?token={client.server.mint_token()}"):
        resp, out = client.get(path, host=host)
        assert resp.status == 403, path
        assert b"mic" not in out


def test_a_missing_host_is_refused(client):
    assert client.get("/api/takes", headers={"Host": None})[0].status == 403


def test_two_host_headers_are_refused(client):
    conn = http.client.HTTPConnection("127.0.0.1", client.server.port, timeout=5)
    conn.putrequest("GET", "/api/takes", skip_host=True, skip_accept_encoding=True)
    conn.putheader("Host", client.host)
    conn.putheader("Host", "evil.example.com")
    conn.putheader("Cookie", client.cookie)
    conn.endheaders()
    assert conn.getresponse().status == 403
    conn.close()


def test_an_absolute_request_target_is_refused(client):
    assert client.get(f"http://evil.example.com/api/takes")[0].status in (400, 403, 404)


# -------------------------------------------------------------------- Origin

@pytest.mark.parametrize("origin", ["http://evil.example.com", "null", "http://localhost",
                                    "http://127.0.0.1", "https://127.0.0.1"])
def test_a_wrong_origin_is_refused_on_any_method(client, origin):
    assert client.get("/api/takes", headers={"Origin": origin})[0].status == 403
    tid = take_id(client)
    assert client.post("/api/keep", {"take": tid, "kept": True}, Origin=origin)[0].status == 403
    assert record.read_keep(client.server.library.path_of(tid))["kept"] is False


def test_a_post_with_no_origin_is_refused(client):
    tid = take_id(client)
    assert client.post("/api/keep", {"take": tid, "kept": True}, Origin=None)[0].status == 403


def test_a_same_origin_get_with_no_origin_is_fine(client):
    assert client.get("/api/takes", headers={"Sec-Fetch-Site": "same-origin"})[0].status == 200


def test_no_cors_headers_ever(client):
    tid = take_id(client)
    for resp, _ in (client.get("/api/takes", headers={"Origin": client.origin}),
                    client.get("/api/takes", headers={"Origin": "http://evil.example.com"}),
                    client.post("/api/keep", {"take": tid, "kept": False}),
                    client.request("OPTIONS", "/api/keep", headers={"Origin": "http://evil.example.com"})):
        assert not [h for h, _ in resp.getheaders() if h.lower().startswith("access-control-")]


def test_a_preflight_is_not_answered_with_permission(client):
    assert client.request("OPTIONS", "/api/keep")[0].status == 405


# ------------------------------------------------------------- Sec-Fetch-Site

@pytest.mark.parametrize("site", ["cross-site", "same-site", "none"])
def test_fetch_metadata_refuses_anything_not_same_origin(client, site):
    tid = take_id(client)
    assert client.post("/api/keep", {"take": tid, "kept": True}, **{"Sec-Fetch-Site": site})[0].status == 403
    assert client.get("/api/takes", headers={"Sec-Fetch-Site": site})[0].status == 403
    assert client.get(f"/api/take/{tid}/audio/human", headers={"Sec-Fetch-Site": site})[0].status == 403


@pytest.mark.parametrize("site", ["cross-site", "same-site"])
def test_the_token_link_and_page_refuse_a_cross_site_visit(client, site):
    assert client.get("/", headers={"Sec-Fetch-Site": site})[0].status == 403
    token = client.server.mint_token()
    assert client.get(f"/auth?token={token}", headers={"Sec-Fetch-Site": site})[0].status == 403


def test_the_token_link_works_from_a_terminal(anon):
    resp, _ = anon.get(f"/auth?token={anon.server.mint_token()}", headers={"Sec-Fetch-Site": "none"})
    assert resp.status == 303


# ------------------------------------------------------------- token and cookie

@pytest.mark.parametrize("path", ["/", "/static/app.js", "/api/session", "/api/takes"])
def test_nothing_is_served_without_the_cookie(anon, path):
    assert anon.get(path)[0].status == 401


def test_recordings_are_not_served_without_the_cookie(client, anon):
    tid = take_id(client)
    resp, out = anon.get(f"/api/take/{tid}/audio/human")
    assert resp.status == 401 and WAV not in out


def test_the_session_cookie_is_strict_httponly_and_named_for_the_port(anon):
    resp, _ = anon.get(f"/auth?token={anon.server.mint_token()}")
    cookie = resp.getheader("Set-Cookie")
    assert cookie.startswith(f"plume_{anon.server.port}=")
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Path=/" in cookie


def test_the_redirect_strips_the_token(anon):
    resp, _ = anon.get(f"/auth?token={anon.server.mint_token()}")
    assert re.fullmatch(r"/#k=[0-9a-f]{64}", resp.getheader("Location"))


def test_a_token_works_once(anon):
    token = anon.server.mint_token()
    assert anon.get(f"/auth?token={token}")[0].status == 303
    resp, out = anon.get(f"/auth?token={token}")
    assert resp.status == 403 and "plume gui url" in out.decode()


def test_a_used_link_with_a_live_cookie_just_goes_home(client):
    token = client.server.mint_token()
    assert client.get(f"/auth?token={token}")[0].status == 303
    assert client.get(f"/auth?token={token}")[0].status == 303


@pytest.mark.parametrize("token", ["", "nope", "a.b", "." * 40, "0" * 32 + "." + "0" * 64, "%C3%A9.%C3%A9"])
def test_a_made_up_token_is_refused(anon, token):
    assert anon.get(f"/auth?token={token}")[0].status == 403


def test_a_made_up_cookie_is_refused(anon):
    anon.cookie = f"plume_{anon.server.port}=" + "0" * 64
    assert anon.get("/api/takes")[0].status == 401


def test_a_cookie_for_another_port_is_refused(client):
    value = client.cookie.split("=", 1)[1]
    client.cookie = f"plume_1={value}"
    assert client.get("/api/takes")[0].status == 401


def test_a_post_needs_the_csrf_header(client):
    tid = take_id(client)
    body = {"take": tid, "kept": True}
    assert client.post("/api/keep", body, **{"X-Plume-Token": None})[0].status == 403
    assert client.post("/api/keep", body, **{"X-Plume-Token": "0" * 64})[0].status == 403
    assert record.read_keep(client.server.library.path_of(tid))["kept"] is False


@pytest.mark.parametrize("ctype", [None, "text/plain", "application/x-www-form-urlencoded",
                                   "multipart/form-data", "application/jsonp"])
def test_a_post_needs_a_json_content_type(client, ctype):
    tid = take_id(client)
    assert client.post("/api/keep", {"take": tid, "kept": True}, **{"Content-Type": ctype})[0].status == 415


def test_the_csrf_value_is_not_the_cookie(client):
    assert client.csrf != client.cookie.split("=", 1)[1]


def test_the_session_api_needs_the_cookie_so_another_site_cannot_read_csrf(anon):
    assert anon.get("/api/session")[0].status == 401


# ------------------------------------------------------- a way back in: plume gui url

def test_a_fresh_url_comes_from_the_state_file_alone(server, anon):
    url = gui.fresh_url()
    assert url.startswith(f"http://127.0.0.1:{server.port}/auth?token=")
    assert anon.get(url.split(str(server.port), 1)[1])[0].status == 303
    assert gui.fresh_url() != url


def test_the_state_file_is_only_hers(server):
    path = gui.state_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_a_state_dir_others_could_read_is_tightened(tmp_path, monkeypatch, takes):
    loose = tmp_path / "loose"
    loose.mkdir(mode=0o755)
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(loose))
    srv = gui.GuiServer(roots=[takes], log=lambda line: None).start()
    try:
        assert stat.S_IMODE(loose.stat().st_mode) == 0o700
    finally:
        srv.close()


def test_a_symlink_at_the_state_path_is_not_followed(tmp_path, monkeypatch, takes):
    state = tmp_path / "s"
    state.mkdir(mode=0o700)
    target = tmp_path / "elsewhere"
    target.write_text("untouched")
    (state / "gui.json").symlink_to(target)
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(state))
    with pytest.raises(OSError):
        gui.GuiServer(roots=[takes], log=lambda line: None).start()
    assert target.read_text() == "untouched"


def test_the_state_file_goes_away_with_the_server(takes, tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "x"))
    srv = gui.GuiServer(roots=[takes], log=lambda line: None).start()
    path = gui.state_path()
    assert path.exists()
    srv.close()
    assert not path.exists()
    with pytest.raises(gui.GuiError):
        gui.fresh_url()


def test_no_running_bridge_is_said_plainly(tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "empty"))
    with pytest.raises(gui.GuiError, match="plume gui"):
        gui.fresh_url()


def test_a_dead_bridge_is_not_given_a_link(tmp_path, monkeypatch):
    state = tmp_path / "dead"
    state.mkdir(mode=0o700)
    # A process that has already exited, so this pid is known to be gone.
    proc = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                          capture_output=True, text=True, check=True)
    (state / "gui.json").write_text(json.dumps({"pid": int(proc.stdout), "port": 1, "key": "00"}))
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(state))
    with pytest.raises(gui.GuiError):
        gui.fresh_url()


# -------------------------------------------------------------------- logging

def test_no_token_or_cookie_reaches_the_log(takes, tmp_path, monkeypatch, capfd):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "state2"))
    logged = []
    srv = gui.GuiServer(roots=[takes], log=logged.append).start()
    try:
        client = Client(srv)
        token = srv.mint_token()
        client.get(f"/auth?token={token}")                    # good
        client.get(f"/auth?token={token}")                    # already used
        client.get("/auth?token=SECRETWRONGTOKEN")            # bad
        client.get("/nope/SECRETPATH?token=SECRETQUERY")      # not found
        client.get("/SECRETPATH", host="evil.example.com")    # refused
        client.request("BREW", "/SECRETPATH")                 # unknown method
        client.login()
        # A malformed request line, which the stdlib would log whole.
        import socket
        with socket.create_connection(("127.0.0.1", srv.port), timeout=5) as raw:
            raw.sendall(b"GET /auth?token=SECRETRAW\r\n\r\n")
            raw.recv(1024)
            raw2 = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
            raw2.sendall(b"NONSENSE SECRETLINE\r\n\r\n")
            raw2.recv(1024)
            raw2.close()
    finally:
        srv.close()
    captured = capfd.readouterr()
    everything = "\n".join(logged) + captured.err + captured.out
    assert logged, "the log sink was never called"
    assert token not in everything
    assert client.cookie.split("=", 1)[1] not in everything
    assert client.csrf not in everything
    assert "SECRET" not in everything
    assert "token=" not in everything


def test_log_lines_are_method_route_status_only(server, client):
    client.get(f"/api/take/{take_id(client)}")
    assert all(re.fullmatch(r"[A-Z?]+ [^ ?]+ \d{3}", line) for line in server.logged), server.logged
    assert any(line.startswith("GET /api/take/{id} ") for line in server.logged)


# ------------------------------------------------------------- response headers

CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; font-src 'self'; "
       "connect-src 'self'; media-src 'self'; base-uri 'none'; form-action 'none'; "
       "frame-ancestors 'none'")


@pytest.mark.parametrize("path,status", [("/", 200), ("/api/takes", 200), ("/static/app.js", 200),
                                         ("/nope", 404), ("/static/nope.js", 404),
                                         ("/api/take/deadbeef", 404)])
def test_security_headers_on_every_response(client, path, status):
    resp, _ = client.get(path)
    assert resp.status == status
    assert resp.getheader("Content-Security-Policy") == CSP
    assert resp.getheader("X-Content-Type-Options") == "nosniff"
    assert resp.getheader("Referrer-Policy") == "no-referrer"
    assert resp.getheader("Cache-Control") == "no-store"


@pytest.mark.parametrize("make", [
    lambda c: c.get("/", host="evil.example.com"),
    lambda c: c.request("BREW", "/"),
    lambda c: c.request("GET", "/api/takes", headers={"Sec-Fetch-Site": "cross-site"}),
])
def test_security_headers_on_refusals_too(client, make):
    resp, _ = make(client)
    assert resp.getheader("Content-Security-Policy") == CSP
    assert resp.getheader("Cache-Control") == "no-store"


def test_the_server_header_gives_nothing_away(client):
    assert "Python" not in (client.get("/")[0].getheader("Server") or "")


def test_errors_do_not_echo_the_request(client):
    _, out = client.get("/<script>alert(1)</script>")
    assert b"<script>" not in out


# ---------------------------------------------------------------- the route table

def test_the_route_table_is_exactly_what_the_page_needs():
    assert gui.GET_ROUTES == ("/", "/auth", "/static/{name}", "/api/session", "/api/takes",
                              "/api/take/{id}", "/api/take/{id}/audio/{track}",
                              "/api/home", "/api/live", "/api/setup")
    assert sorted(gui.POST_OPS) == ["cue", "keep", "rename", "setup/answer", "setup/cancel", "setup/listen",
                                    "setup/reset", "start", "stop"]


@pytest.mark.parametrize("op", ["calibrate", "play", "lead", "jam", "voice", "delete", "record",
                                "meter", "listen"])
def test_only_start_can_start_anything(client, op):
    assert client.post(f"/api/{op}", {"take": "x"})[0].status == 404
    assert client.get(f"/api/{op}")[0].status == 404


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "TRACE", "CONNECT", "BREW"])
def test_other_methods_are_refused(client, method):
    assert client.request(method, "/api/takes")[0].status == 405


def test_the_bridge_cannot_reach_a_sound_device_and_starts_nothing_itself():
    """The child and its consent live in guitake.py, which test_gui_take.py pins."""
    source = Path(gui.__file__).read_text() + Path(record.__file__).read_text()
    for banned in ("sounddevice", "subprocess", "os.system", "Popen", "os.exec", "os.spawn",
                   "realtime.lead", "realtime.jam", "send_cue", "send_stop"):
        assert banned not in source, banned


def test_importing_the_gui_does_not_import_sounddevice():
    code = "import sys, plume.gui; assert 'sounddevice' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                   env={**os.environ, "PYTHONPATH": str(Path(gui.__file__).parents[1])})


# ------------------------------------------------------------------- the library

def test_the_shelf_lists_takes_and_skips_other_folders(client):
    _, out = client.get("/api/takes")
    assert sorted(t["folder"] for t in json.loads(out)["takes"]) == ["one", "two"]


def test_a_take_id_is_not_a_path(client):
    for t in json.loads(client.get("/api/takes")[1])["takes"]:
        assert re.fullmatch(r"[0-9a-f]{16}", t["id"])


@pytest.mark.parametrize("bad", ["../one", "..%2Fone", "%2e%2e", "one", "..", "/etc/passwd",
                                 "0" * 16, "A" * 16, "0" * 500])
def test_a_client_cannot_name_a_path(client, bad):
    assert client.get(f"/api/take/{bad}")[0].status == 404
    assert client.get(f"/api/take/{bad}/audio/human")[0].status == 404
    assert client.post("/api/keep", {"take": bad, "kept": True})[0].status == 404


@pytest.mark.parametrize("track", ["../../secret", "session.json", "HUMAN", "human.wav", "..%2f"])
def test_only_the_two_recordings_can_be_fetched(client, track):
    assert client.get(f"/api/take/{take_id(client)}/audio/{track}")[0].status == 404


@pytest.mark.parametrize("name", ["../gui.py", "..%2fgui.py", "%2e%2e/gui.py", "nope.js", "app.js/",
                                  "fonts/../app.js", "fonts/../../gui.py"])
def test_static_cannot_escape(client, name):
    assert client.get(f"/static/{name}")[0].status == 404


def test_every_response_about_a_take_is_clean(client):
    tid = take_id(client)
    one = json.loads(client.get(f"/api/take/{tid}")[1])
    assert one.pop("audio") == ["human", "machine"]     # track names, not claims
    shelf = json.loads(client.get("/api/takes")[1])
    for doc in (one, shelf):
        out = json.dumps(doc)
        for bad in POISON:
            assert bad not in out, bad


def test_a_take_response_has_the_record(client):
    doc = json.loads(client.get(f"/api/take/{take_id(client)}")[1])
    assert doc["bars"] == 40 and doc["roast"] and doc["events"][-1]["tier"] == "end"
    assert doc["audio"] == ["human", "machine"]
    assert "A cue" in json.dumps(doc) or "You asked" in json.dumps(doc)


def test_a_take_response_with_nan_stays_valid_json(client, takes):
    (takes / "one" / "ending.json").write_text('{"reason": "x", "elapsed_s": NaN}')
    out = client.get(f"/api/take/{take_id(client)}")[1].decode()
    json.loads(out, parse_constant=lambda c: pytest.fail(f"{c} in the response"))


def test_a_new_take_appears_without_a_restart(client, takes):
    write_take(takes / "three", song(3))
    assert "three" in [t["folder"] for t in json.loads(client.get("/api/takes")[1])["takes"]]


def test_the_shelf_notices_a_take_that_changed(client, takes):
    assert next(t for t in json.loads(client.get("/api/takes")[1])["takes"]
                if t["folder"] == "two")["bars"] == 0
    write_take(takes / "two", song(5), ending=ending_doc())
    assert next(t for t in json.loads(client.get("/api/takes")[1])["takes"]
                if t["folder"] == "two")["bars"] == 5


def test_a_symlinked_folder_in_the_library_is_not_followed(client, takes, tmp_path):
    outside = write_take(tmp_path / "outside", song(3))
    (takes / "link").symlink_to(outside, target_is_directory=True)
    # A followed link shows up under its target's name, so check for that as well.
    folders = [t["folder"] for t in json.loads(client.get("/api/takes")[1])["takes"]]
    assert "link" not in folders and "outside" not in folders


def test_an_absent_library_is_an_empty_shelf(tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "s3"))
    srv = gui.GuiServer(roots=[tmp_path / "nothing-here"], log=lambda line: None).start()
    try:
        client = Client(srv).login()
        assert json.loads(client.get("/api/takes")[1])["takes"] == []
    finally:
        srv.close()


def test_a_pinned_take_opens_first_and_sits_on_the_shelf(tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "s4"))
    pinned = write_take(tmp_path / "pinned", song(5))
    srv = gui.GuiServer(roots=[], pinned=pinned, log=lambda line: None).start()
    try:
        client = Client(srv)
        resp, _ = client.get(f"/auth?token={srv.mint_token()}")
        client.cookie = resp.getheader("Set-Cookie").split(";", 1)[0]
        client.csrf = resp.getheader("Location").split("#k=")[1]
        session = json.loads(client.get("/api/session")[1])
        takes = json.loads(client.get("/api/takes")[1])["takes"]
        assert [t["id"] for t in takes] == [session["open_take"]]
    finally:
        srv.close()


# ----------------------------------------------------------------- Keep and rename

def test_keep_writes_a_sidecar_and_undo_clears_the_flag(client):
    tid = take_id(client)
    path = client.server.library.path_of(tid)
    resp, out = client.post("/api/keep", {"take": tid, "kept": True})
    assert resp.status == 200 and json.loads(out)["kept"] is True
    assert record.read_keep(path) == {"kept": True, "name": ""}
    resp, _ = client.post("/api/keep", {"take": tid, "kept": False})
    assert record.read_keep(path)["kept"] is False


def test_rename_keeps_the_flag(client):
    tid = take_id(client)
    client.post("/api/keep", {"take": tid, "kept": True})
    resp, out = client.post("/api/rename", {"take": tid, "name": "  slow one\n\u0007 "})
    assert resp.status == 200 and json.loads(out)["name"] == "slow one"
    assert record.read_keep(client.server.library.path_of(tid)) == {"kept": True, "name": "slow one"}


def test_a_long_name_is_cut_not_refused(client):
    tid = take_id(client)
    _, out = client.post("/api/rename", {"take": tid, "name": "x" * 500})
    assert len(json.loads(out)["name"]) == record.MAX_NAME_CHARS


def test_keep_deletes_and_changes_nothing_else(client):
    tid = take_id(client)
    folder = client.server.library.path_of(tid)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    client.post("/api/keep", {"take": tid, "kept": True})
    after = {p.name: p.read_bytes() for p in folder.iterdir()}
    assert set(after) - set(before) == {record.REC_KEEP}
    assert all(after[name] == before[name] for name in before)


@pytest.mark.parametrize("path,body", [
    ("/api/keep", {"take": "x"}), ("/api/keep", {"kept": True}), ("/api/keep", []),
    ("/api/keep", {"take": 7, "kept": True}), ("/api/rename", {"name": "x"}),
    ("/api/rename", {"take": "x", "name": 7}), ("/api/keep", b"not json"), ("/api/keep", b"")])
def test_a_bad_body_is_refused(client, path, body):
    if isinstance(body, dict) and body.get("take") == "x":
        body = {**body, "take": take_id(client)}
    assert client.post(path, body)[0].status == 400


def test_a_kept_flag_must_be_a_real_boolean(client):
    assert client.post("/api/keep", {"take": take_id(client), "kept": "yes"})[0].status == 400


def test_an_oversized_body_is_refused(client):
    resp, _ = client.post("/api/rename", {"take": take_id(client), "name": "x" * 100000})
    assert resp.status == 413


def test_a_read_only_folder_is_reported_not_crashed(client):
    tid = take_id(client)
    folder = client.server.library.path_of(tid)
    folder.chmod(0o500)
    try:
        resp, out = client.post("/api/keep", {"take": tid, "kept": True})
        assert resp.status == 500 and str(folder) not in out.decode()
    finally:
        folder.chmod(0o700)


# ------------------------------------------------------- audio: the range handler

def audio(client, tid=None, track="human", **headers):
    tid = tid or take_id(client)
    return client.get(f"/api/take/{tid}/audio/{track}", headers=headers or None)


def test_the_whole_file_is_served_with_range_support(client):
    resp, out = audio(client)
    assert resp.status == 200 and out == WAV
    assert resp.getheader("Accept-Ranges") == "bytes"
    assert resp.getheader("Content-Type") == "audio/wav"
    assert resp.getheader("Content-Length") == str(len(WAV))


@pytest.mark.parametrize("spec,start,end", [("bytes=0-9", 0, 9), ("bytes=10-19", 10, 19),
                                            ("bytes=1000-", 1000, 1023), ("bytes=-24", 1000, 1023),
                                            ("bytes=1000-99999", 1000, 1023)])
def test_a_range_is_served_exactly(client, spec, start, end):
    resp, out = audio(client, Range=spec)
    assert resp.status == 206 and out == WAV[start:end + 1]
    assert resp.getheader("Content-Range") == f"bytes {start}-{end}/{len(WAV)}"
    assert resp.getheader("Content-Length") == str(end - start + 1)


@pytest.mark.parametrize("spec", ["bytes=5000-", "bytes=1024-", "bytes=20-10", "bytes=-0",
                                  "bytes=x-y", "items=0-1", "bytes=", "bytes=-", "bytes=0-1,4-5",
                                  "bytes=99999999999999999999-"])
def test_an_unsatisfiable_or_unsupported_range_gets_416(client, spec):
    resp, out = audio(client, Range=spec)
    assert resp.status == 416 and out == b""
    assert resp.getheader("Content-Range") == f"bytes */{len(WAV)}"


def test_a_take_with_no_recording_is_a_404(client):
    assert audio(client, take_id(client, "two"))[0].status == 404


def test_a_symlinked_recording_is_not_served(client, takes, tmp_path):
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"secret")
    (takes / "one" / "human.wav").unlink()
    (takes / "one" / "human.wav").symlink_to(secret)
    resp, out = audio(client)
    assert resp.status == 404 and b"secret" not in out


# ------------------------------------------------------------------ the page itself

STATIC = Path(gui.__file__).with_name("gui_static")
TEXT_FILES = [p for p in STATIC.rglob("*") if p.suffix in (".html", ".js", ".css")]


def test_there_are_static_files_to_check():
    assert {"index.html", "app.js", "stone.js", "theme.js", "plume.css"} <= {p.name for p in TEXT_FILES}


def test_static_files_name_no_remote_url():
    for path in TEXT_FILES:
        found = re.findall(r"""(?:https?:)?//[a-zA-Z0-9-]+\.[a-zA-Z]""", path.read_text())
        assert not found, f"{path.name}: {found}"
        assert "@import" not in path.read_text()


def test_every_static_file_is_on_the_servers_list_and_vice_versa():
    on_disk = {p.relative_to(STATIC).as_posix() for p in STATIC.rglob("*")
               if p.is_file() and p.suffix != ".txt"}
    assert on_disk == set(gui.STATIC_FILES)


def test_nothing_inline_for_the_csp_to_refuse():
    html = (STATIC / "index.html").read_text()
    assert "<style" not in html and " style=" not in html
    assert not re.search(r"<script(?![^>]*\bsrc=)", html), "inline script"
    assert not re.search(r"\son[a-z]+=", html), "inline handler"
    assert "data:" not in "".join(p.read_text() for p in TEXT_FILES)


def test_night_is_never_chosen_by_the_system():
    for path in TEXT_FILES:
        assert "prefers-color-scheme" not in path.read_text(), path.name


def test_the_page_cannot_reach_the_mic_or_start_sound_by_itself():
    source = "".join(p.read_text() for p in TEXT_FILES)
    for banned in ("getUserMedia", "AudioContext", "autoplay", "new Audio", "createMediaStream",
                   "EventSource", "WebSocket", "innerHTML", "outerHTML", "insertAdjacentHTML",
                   "document.write", "eval("):
        assert banned not in source, banned
    assert 'preload="none"' in (STATIC / "index.html").read_text()


def test_the_page_has_no_demo_code():
    source = "".join(p.read_text() for p in TEXT_FILES)
    for banned in ("Sim(", "seedTakes", "demo", "Demo", "Again", "plumeChecks"):
        assert banned not in source, banned


def test_the_static_files_ship_with_the_package():
    text = (Path(gui.__file__).parents[2] / "pyproject.toml").read_text()
    assert '"gui_static/*"' in text and '"gui_static/fonts/*"' in text


# ------------------------------------------------------------------- the command

def test_plume_gui_refuses_a_folder_that_is_not_a_take(tmp_path, capsys):
    from plume import cli
    assert cli.main(["gui", "--take", str(tmp_path / "nope")]) == 1
    assert "not a take" in capsys.readouterr().err
    assert cli.main(["gui", "--take", str(tmp_path)]) == 1


def test_plume_gui_url_says_plainly_when_nothing_is_running(tmp_path, monkeypatch, capsys):
    from plume import cli
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "none"))
    assert cli.main(["gui", "url"]) == 1
    assert "plume gui" in capsys.readouterr().err


def test_plume_gui_url_prints_a_link_that_works(server, anon, capsys):
    from plume import cli
    assert cli.main(["gui", "url"]) == 0
    url = capsys.readouterr().out.strip()
    assert anon.get(url.split(str(server.port), 1)[1])[0].status == 303


def test_the_real_command_serves_then_quits_cleanly_on_ctrl_c(tmp_path):
    """The whole command, as its own process, with a private home."""
    take = write_take(tmp_path / "take", song(5), ending=ending_doc())
    env = {**os.environ, "HOME": str(tmp_path), "PLUME_GUI_STATE_DIR": str(tmp_path / "state"),
           "PYTHONPATH": str(Path(gui.__file__).parents[1])}
    env.pop("PLUME_ALLOW_AUDIO", None)
    proc = subprocess.Popen([sys.executable, "-m", "plume", "gui", "--take", str(take), "--no-open"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
    try:
        url = proc.stdout.readline().strip()
        found = re.fullmatch(r"http://127\.0\.0\.1:(\d+)(/auth\?token=\S+)", url)
        assert found, url
        conn = http.client.HTTPConnection("127.0.0.1", int(found.group(1)), timeout=5)
        conn.putrequest("GET", found.group(2), skip_host=True, skip_accept_encoding=True)
        conn.putheader("Host", f"127.0.0.1:{found.group(1)}")
        conn.endheaders()
        assert conn.getresponse().status == 303
        conn.close()
    finally:
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=15)
    assert proc.returncode == 0
    assert "token=" not in out + err, "the one-time link must only be printed once"
    assert not (tmp_path / "state" / "gui.json").exists()


def test_the_bridge_itself_never_handles_consent_to_play():
    """Only the Start route's spawner may, and test_gui_take.py pins that one by its syntax."""
    source = Path(gui.__file__).read_text()
    assert "PLUME_ALLOW_AUDIO" not in source and "check_audio_consent" not in source
    assert "AUDIO_CONSENT" not in source


# ------------------------------------------------------- the cookie is not the key

def test_the_cookie_alone_reads_nothing_private(client, anon):
    """Cookies ignore ports, so any server on 127.0.0.1 can capture this one."""
    tid = take_id(client)
    anon.cookie = client.cookie
    for path in ("/api/session", "/api/takes", f"/api/take/{tid}", f"/api/take/{tid}/audio/human"):
        assert anon.get(path)[0].status == 401, path
    assert anon.get(f"/api/take/{tid}/audio/human?k={'0' * 64}")[0].status == 401
    # The page itself holds nothing private, and has to load to read its fragment.
    assert anon.get("/")[0].status == 200


def test_audio_accepts_the_key_in_the_query_for_the_audio_element(client, anon):
    tid = take_id(client)
    anon.cookie = client.cookie
    resp, out = anon.get(f"/api/take/{tid}/audio/human?k={client.csrf}")
    assert resp.status == 200 and out == WAV
    assert client.csrf not in "".join(client.server.logged)


def test_a_key_in_the_query_opens_only_audio(client, anon):
    anon.cookie = client.cookie
    assert anon.get(f"/api/takes?k={client.csrf}")[0].status == 401


def test_the_key_alone_is_not_enough_either(client, anon):
    anon.csrf = client.csrf
    assert anon.get("/api/takes")[0].status == 401


# ------------------------------------------------------------- a folder can change

def test_a_planted_tmp_symlink_cannot_aim_keep_at_another_file(client, takes, tmp_path, monkeypatch):
    """Even one that guessed the temporary name, which is what the O_EXCL is for."""
    tid = take_id(client)
    outside = tmp_path / "precious"
    outside.write_text("original")
    monkeypatch.setattr(gui.secrets, "token_hex", lambda n: "guessed")
    planted = takes / "one" / f".{record.REC_KEEP}.guessed.tmp"
    planted.symlink_to(outside)
    assert client.post("/api/keep", {"take": tid, "kept": True})[0].status != 200
    assert outside.read_text() == "original" and planted.is_symlink()
    assert not (takes / "one" / record.REC_KEEP).exists()


def test_keeps_temporary_file_has_a_new_name_each_time(client, takes, monkeypatch):
    tid = take_id(client)
    names = []
    real = os.open

    def spy(path, flags, *args, **kw):
        if isinstance(path, str) and path.endswith(".tmp"):
            names.append(path)
        return real(path, flags, *args, **kw)

    monkeypatch.setattr(gui.os, "open", spy)
    for kept in (True, False):
        assert client.post("/api/keep", {"take": tid, "kept": kept})[0].status == 200
    assert len(set(names)) == 2
    assert not [p.name for p in (takes / "one").glob(".*")]


def test_a_symlink_on_the_way_to_a_library_folder_is_not_a_real_folder(tmp_path):
    real = tmp_path / "real"
    (real / "take").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert gui._still_real((real / "take").resolve()) is True
    assert gui._still_real(link / "take") is False


def test_a_symlink_at_a_state_file_is_not_read(tmp_path):
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({"pid": 1, "port": 1, "key": "00"}))
    assert gui._read_state(target) is not None
    link = tmp_path / "gui.json"
    link.symlink_to(target)
    assert gui._read_state(link) is None


def test_a_recording_is_opened_in_the_folder_it_was_checked_in(client, takes, tmp_path, monkeypatch):
    """The folder's name is looked at once.  Swapping it after the check changes nothing."""
    tid = take_id(client)
    real_open_folder = gui._open_folder
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / realtime.REC_HUMAN).write_bytes(b"OUTSIDE")

    def swap_after_opening(path):
        fd = real_open_folder(path)
        path.rename(tmp_path / "moved-away")
        path.symlink_to(outside, target_is_directory=True)
        return fd

    monkeypatch.setattr(gui, "_open_folder", swap_after_opening)
    resp, out = client.get(f"/api/take/{tid}/audio/human?k={client.csrf}")
    assert resp.status == 200 and out == WAV


def test_a_planted_cookie_of_the_same_name_cannot_lock_anyone_out(client):
    name, value = client.cookie.split("=", 1)
    for cookie in (f"{name}=junk; {name}={value}", f"{name}={value}; {name}=junk"):
        assert client.get("/api/takes", headers={"Cookie": cookie})[0].status == 200
    assert client.get("/api/takes", headers={"Cookie": f"{name}=junk"})[0].status == 401


def test_a_stale_file_naming_a_reused_pid_does_not_beat_a_live_bridge(server):
    """A kill leaves the state files behind, and the pid can belong to anything later."""
    state = gui._read_state(gui.state_path())
    assert state[1] == server.port
    gui._write_state({"pid": os.getpid(), "port": 1, "key": state[0].hex()}, gui.state_path())
    assert f":{server.port}/" in gui.fresh_url()


def test_keep_replaces_a_symlinked_keep_file_without_touching_its_target(client, takes, tmp_path):
    tid = take_id(client)
    outside = tmp_path / "precious"
    outside.write_text("original")
    (takes / "one" / record.REC_KEEP).symlink_to(outside)
    assert client.post("/api/keep", {"take": tid, "kept": True})[0].status == 200
    assert outside.read_text() == "original" and not (takes / "one" / record.REC_KEEP).is_symlink()


def _swap_for_symlink(folder, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "human.wav").write_bytes(b"OUTSIDE")
    (outside / "session.json").write_text("{}")
    folder.rename(tmp_path / "moved-away")
    folder.symlink_to(outside)
    return outside


def test_a_listed_folder_swapped_for_a_symlink_is_gone(client, takes, tmp_path):
    tid = take_id(client)
    outside = _swap_for_symlink(takes / "one", tmp_path)
    assert client.get(f"/api/take/{tid}/audio/human")[0].status == 404
    assert client.get(f"/api/take/{tid}")[0].status == 404
    assert client.post("/api/keep", {"take": tid, "kept": True})[0].status == 404
    assert not (outside / record.REC_KEEP).exists()


def test_a_pinned_folder_swapped_for_a_symlink_is_gone(tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "s9"))
    pinned = write_take(tmp_path / "pinned", song(5), wav_bytes=WAV).resolve()
    srv = gui.GuiServer(roots=[], pinned=pinned, log=lambda line: None).start()
    try:
        client = Client(srv).login()
        tid = srv.library.pinned_id()
        assert client.get(f"/api/take/{tid}/audio/human")[0].status == 200
        outside = _swap_for_symlink(pinned, tmp_path)
        assert client.get(f"/api/take/{tid}/audio/human")[0].status == 404
        assert client.post("/api/keep", {"take": tid, "kept": True})[0].status == 404
        assert not (outside / record.REC_KEEP).exists()
    finally:
        srv.close()


def test_one_poisoned_take_does_not_cost_the_shelf(client, takes):
    (takes / "one" / record.REC_DECISIONS).write_text(json.dumps({"bar": 1, "t_s": 10 ** 400}) + "\n")
    (takes / "one" / realtime.REC_MANIFEST).write_text(json.dumps({"started_at": 10 ** 400}))
    resp, out = client.get("/api/takes")
    assert resp.status == 200 and {t["folder"] for t in json.loads(out)["takes"]} == {"one", "two"}


def test_an_unexpected_failure_is_an_answer_not_a_dropped_connection(client, monkeypatch):
    def boom(self):
        raise ZeroDivisionError

    monkeypatch.setattr(gui.Library, "shelf", boom)
    resp, out = client.get("/api/takes")
    assert resp.status == 500 and "ZeroDivision" not in out.decode()


@pytest.mark.parametrize("name", [record.REC_KEEP, record.REC_DECISIONS, realtime.REC_MANIFEST,
                                  realtime.REC_ENDING, realtime.REC_CUES])
def test_a_fifo_answers_instead_of_hanging(client, takes, name):
    path = takes / "one" / name
    path.unlink(missing_ok=True)
    os.mkfifo(path)
    assert client.get("/api/takes")[0].status == 200


def test_a_fifo_keep_file_does_not_wedge_the_library(client, takes):
    tid = take_id(client)
    good = take_id(client, "two")
    path = takes / "one" / record.REC_KEEP
    os.mkfifo(path)
    assert client.post("/api/keep", {"take": tid, "kept": True})[0].status == 200
    assert client.get(f"/api/take/{good}")[0].status == 200


def test_a_stalled_keep_write_does_not_hold_the_lock_readers_need(client, takes):
    good = take_id(client, "two")
    with client.server.library._keep_lock:
        assert client.get(f"/api/take/{good}")[0].status == 200


def test_a_fifo_named_as_a_recording_is_refused_at_once(client, takes):
    tid = take_id(client)
    path = takes / "one" / realtime.REC_HUMAN
    path.unlink()
    os.mkfifo(path)
    assert client.get(f"/api/take/{tid}/audio/human")[0].status == 404


def test_a_non_utf8_folder_name_does_not_break_the_id():
    assert re.fullmatch(r"[0-9a-f]{16}", gui.Library._id_of(Path("/x/\udcff")))


def test_a_folder_being_recorded_is_marked_live(client, takes, monkeypatch):
    tid = take_id(client)
    (takes / "one" / realtime.REC_ENDING).unlink()
    assert "was killed" in json.dumps(json.loads(client.get(f"/api/take/{tid}")[1])["events"])

    class Session:
        record_dir = str(takes / "one")

    monkeypatch.setattr(gui, "list_sessions", lambda prune=True: [Session()])
    doc = json.loads(client.get(f"/api/take/{tid}")[1])
    assert doc["live"] is True and "was killed" not in json.dumps(doc["events"])
    shelf = json.loads(client.get("/api/takes")[1])["takes"]
    assert any(t["folder"] == "one" for t in shelf)


# --------------------------------------------------------------- another way back in

def test_a_url_still_works_when_the_newest_bridge_has_gone(takes, tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "two-bridges"))
    first = gui.GuiServer(roots=[takes], log=lambda line: None).start()
    second = gui.GuiServer(roots=[takes], log=lambda line: None).start()
    second.close()
    try:
        assert f":{first.port}/" in gui.fresh_url()
    finally:
        first.close()
    with pytest.raises(gui.GuiError):
        gui.fresh_url()


def test_a_dead_bridges_file_does_not_mint_a_link(takes, tmp_path, monkeypatch):
    monkeypatch.setenv(gui.STATE_DIR_ENV, str(tmp_path / "dead"))
    srv = gui.GuiServer(roots=[takes], log=lambda line: None).start()
    srv.close()
    gui._write_state({"pid": 999999, "port": 1, "key": "00"}, gui._own_state_path(1))
    monkeypatch.setattr(gui, "_pid_alive", lambda pid: False)
    with pytest.raises(gui.GuiError):
        gui.fresh_url()


def test_each_bridges_own_file_is_only_hers(server):
    mode = stat.S_IMODE(os.stat(gui._own_state_path(server.port)).st_mode)
    assert mode == 0o600


# ---------------------------------------------------------------- old-protocol forms

def _raw(server, data: bytes) -> bytes:
    import socket as socketlib

    with socketlib.create_connection(("127.0.0.1", server.port), timeout=3) as sock:
        sock.sendall(data)
        out = b""
        try:
            while chunk := sock.recv(65536):
                out += chunk
        except TimeoutError:
            pass
    return out


@pytest.mark.parametrize("request_bytes", [
    b"GET /api/takes\r\n\r\n", b"GET /api/takes HTTP/9.9\r\n\r\n", b"NONSENSE\r\n\r\n"])
def test_every_answer_has_a_status_line_and_headers(client, request_bytes):
    with_host = request_bytes.replace(
        b"\r\n\r\n", f"\r\nHost: {client.host}\r\nCookie: {client.cookie}\r\nX-Plume-Token: {client.csrf}\r\n\r\n".encode())
    out = _raw(client.server, with_host)
    assert out.startswith(b"HTTP/1."), out[:60]
    assert b"Content-Security-Policy:" in out and b"nosniff" in out
    assert b'"takes"' not in out


# ---------------------------------------------------------- together: both as one

def _together_take(folder, *, human, machine, rate=48000, stream=None, source="direct"):
    import soundfile as sf
    folder.mkdir(parents=True)
    sf.write(folder / "human.wav", human, rate, subtype="PCM_24")
    sf.write(folder / "machine.wav", machine, rate, subtype="PCM_24")
    (folder / "session.json").write_text(json.dumps({"seed": 1, "input": {"source": source}}))
    (folder / "decisions.ndjson").write_text("")
    if stream is not None:
        (folder / "stream.json").write_text(json.dumps(stream))
    return folder


def _mix(tmp_path, source="direct", stream={"reported_ms": 10.0}):
    from plume import gui
    human = np.zeros(4800); human[480 + 100] = 0.5          # she played at 100, heard 10 ms late
    machine = np.zeros(4800); machine[100] = 0.25
    folder = _together_take(tmp_path / "takes" / "t1", human=human, machine=machine, stream=stream, source=source)
    return gui, human, machine, folder


def test_together_lines_the_player_up_and_leaves_room_for_both(tmp_path):
    import soundfile as sf
    from plume import gui, record
    gui_mod, _, _, folder = _mix(tmp_path)
    assert record.playing_delay_frames(folder, 48000) == 480
    with sf.SoundFile(folder / "human.wav") as h, sf.SoundFile(folder / "machine.wav") as m:
        pcm = np.frombuffer(gui._mixed_pcm(h, m, 480, 0, 200), dtype="<i2") / 32767.0
    assert pcm[100] == pytest.approx((0.5 + 0.25) * 0.7, abs=1e-3)
    assert abs(pcm[:100]).max() == 0


@pytest.mark.parametrize("stream, want", [
    ({"calibrated_ms": 20.0, "reported_ms": 10.0}, 960), ({"calibrated_ms": None, "reported_ms": 10.0}, 480),
    ({"reported_ms": "lots"}, 0), ({"reported_ms": True}, 0), ({"reported_ms": -5}, 0), ({"reported_ms": 90000}, 0), (None, 0)])
def test_the_delay_is_the_measured_one_then_the_reported_one_then_none(tmp_path, stream, want):
    from plume import record
    folder = tmp_path / "t"; folder.mkdir()
    if stream is not None:
        (folder / "stream.json").write_text(json.dumps(stream))
    assert record.playing_delay_frames(folder, 48000) == want


def test_a_mix_past_the_end_of_a_file_is_silence_not_a_short_read(tmp_path):
    import soundfile as sf
    from plume import gui
    _, _, _, folder = _mix(tmp_path)
    with sf.SoundFile(folder / "human.wav") as h, sf.SoundFile(folder / "machine.wav") as m:
        assert len(gui._mixed_pcm(h, m, 480, 4700, 300)) == 600


def test_the_mix_header_says_what_follows():
    import io, wave
    from plume import gui
    with wave.open(io.BytesIO(gui._wav_header(10, 48000) + b"\0" * 20)) as w:
        assert (w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()) == (10, 48000, 1, 2)


def _two_track_take(takes, name, source):
    human = np.zeros(4800); human[580] = 0.5
    machine = np.zeros(4800); machine[100] = 0.25
    return _together_take(takes / name, human=human, machine=machine,
                          stream={"reported_ms": 10.0}, source=source)


def test_together_is_served_whole_and_by_range(takes, client):
    _two_track_take(takes, "direct-take", "direct")
    tid = take_id(client, "direct-take")
    assert json.loads(client.get(f"/api/take/{tid}")[1])["together"] is True
    resp, whole = client.get(f"/api/take/{tid}/audio/together")
    assert resp.status == 200 and resp.getheader("Content-Type") == "audio/wav"
    assert int(resp.getheader("Content-Length")) == len(whole) == 44 + 4800 * 2
    pcm = np.frombuffer(whole[44:], dtype="<i2")
    assert int(np.argmax(abs(pcm))) == 100
    assert pcm[100] / 32767 == pytest.approx(0.75 * 0.7, abs=1e-3)
    # Odd byte edges and the header boundary come back byte for byte.
    for first, last in ((0, 43), (10, 60), (45, 244), (100, 100), (len(whole) - 3, len(whole) - 1)):
        part, got = client.get(f"/api/take/{tid}/audio/together", headers={"Range": f"bytes={first}-{last}"})
        assert part.status == 206 and got == whole[first:last + 1]
    assert client.get(f"/api/take/{tid}/audio/together",
                      headers={"Range": f"bytes={len(whole)}-"})[0].status == 416


def test_a_room_mic_take_has_no_together_since_it_may_hold_the_band_already(takes, client):
    _two_track_take(takes, "room-take", "room")
    tid = take_id(client, "room-take")
    assert json.loads(client.get(f"/api/take/{tid}")[1])["together"] is False
    assert client.get(f"/api/take/{tid}/audio/together")[0].status == 404


def test_together_needs_the_page_key(takes, anon, client):
    _two_track_take(takes, "direct-take", "direct")
    tid = take_id(client, "direct-take")
    assert anon.get(f"/api/take/{tid}/audio/together")[0].status in (401, 403)
