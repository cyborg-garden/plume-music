"""Spike entry point for a frozen Plume.  Throwaway: not for main.

The page starts its children as ``sys.executable -P -m plume ...``.  In a frozen
build ``sys.executable`` is this program, which has no ``-P`` or ``-m``.  The hack
is the few lines in :func:`_own_argv`: drop them, and the rest is a plume command.
With no command at all, this is the app: it runs ``plume gui``.
"""
import sys


def _own_argv(argv):
    while argv[:1] == ["-P"]:
        argv = argv[1:]
    if argv[:2] == ["-m", "plume"]:
        argv = argv[2:]
    return argv


def _bind_without_reverse_lookup():
    """http.server looks up the name of 127.0.0.1 when it binds.  Inside a macOS app
    bundle that lookup was measured at 35 seconds.  The page's server never uses the name."""
    import http.server
    import socketserver

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    http.server.HTTPServer.server_bind = server_bind


def main():
    argv = _own_argv(sys.argv[1:])
    _bind_without_reverse_lookup()
    if argv[:1] == ["--spike-webview"]:
        import spike_webview

        return spike_webview.run()
    if argv[:1] and argv[0].startswith("--spike-"):
        import spike_selftest

        return spike_selftest.run(argv)
    sys.argv = [sys.argv[0], *(argv or ["gui"])]
    from plume.cli import main as plume_main

    return plume_main()


if __name__ == "__main__":
    sys.exit(main())
