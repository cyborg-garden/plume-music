"""The few places macOS, Linux and Windows disagree, in one file.

plume hardens its private files and its child processes with POSIX tools:
``O_NOFOLLOW`` so a planted symlink is never written through, ``O_NONBLOCK`` so a
FIFO can never hang a reader, ``fchmod`` so a file is 0600 before a byte lands in
it, and process sessions so a terminal's Ctrl-C reaches the band once.  Windows has
none of these, and two of the obvious stand-ins are traps:

- ``os.kill(pid, 0)`` is the POSIX "is it alive?" probe.  On Windows signal 0 is
  ``CTRL_C_EVENT``, and every other number is ``TerminateProcess``.  Asking whether a
  take is alive would interrupt it, or end it without its tail.
- ``os.open`` opens in *text* mode on Windows unless asked for ``O_BINARY``.  A WAV
  read that way has its line endings rewritten and stops at the first 0x1A byte.

So every caller goes through here, and on POSIX everything below is exactly what the
callers did before: the same flags, the same calls, in the same order.
"""
from __future__ import annotations

import errno
import os
import signal
import stat
import subprocess
import sys

#: True on Windows, where the POSIX hardening below has to be replaced, not skipped.
IS_WINDOWS = sys.platform == "win32"

#: Each flag, or 0 where the platform has no such thing.  ``O_BINARY`` is the one
#: that only Windows has, and the one Windows cannot do without.
O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
O_BINARY = getattr(os, "O_BINARY", 0)

#: Opening a folder by handle and working inside it (``dir_fd``) is what keeps a
#: swapped folder name from redirecting a write.  Windows cannot open a folder with
#: ``os.open`` at all, so callers fall back to re-checking the path.  (``os.replace``
#: takes ``src_dir_fd`` wherever ``os.rename`` does, but is not listed itself.)
DIR_FDS = (hasattr(os, "O_DIRECTORY") and os.open in os.supports_dir_fd
           and os.rename in os.supports_dir_fd and os.unlink in os.supports_dir_fd)


def open_private(path, flags: int, mode: int = 0o600, *, dir_fd: int | None = None) -> int:
    """``os.open`` with ``O_NOFOLLOW`` and ``O_BINARY``, on every platform.

    Where ``O_NOFOLLOW`` exists this is that flag and nothing more.  Where it does
    not (Windows), a symlink at ``path`` is refused by ``lstat`` first.  That leaves a
    window a POSIX open does not have, but creating a symlink on Windows needs an
    administrator or developer mode, and refusing the plain case is what matters.
    """
    flags |= O_BINARY
    if O_NOFOLLOW:
        flags |= O_NOFOLLOW
    else:
        try:
            st = os.lstat(path) if dir_fd is None else os.lstat(path, dir_fd=dir_fd)
        except FileNotFoundError:
            st = None
        if st is not None and stat.S_ISLNK(st.st_mode):
            raise OSError(errno.ELOOP, "refusing to follow a symlink", str(path))
    if dir_fd is None:
        return os.open(path, flags, mode)
    return os.open(path, flags, mode, dir_fd=dir_fd)


def read_flags() -> int:
    """For reading a file that must be a real file: no symlink, no waiting on a FIFO."""
    return os.O_RDONLY | O_NONBLOCK


def fchmod_private(fd: int, mode: int = 0o600) -> None:
    """``fchmod``, where it means what it says.

    Windows (3.13+) has an ``os.fchmod`` that only toggles the read-only bit.  There a
    file's privacy comes from its folder's access list instead: plume writes these
    under the user's home or the user's own temp folder, both private to that user.
    """
    if not IS_WINDOWS and hasattr(os, "fchmod"):
        os.fchmod(fd, mode)


def chmod_private(path, mode: int) -> None:
    """``chmod``, on POSIX.  On Windows it is a no-op, for the reason above."""
    if not IS_WINDOWS:
        os.chmod(path, mode)


# ------------------------------------------------------------------ processes

#: ``STILL_ACTIVE``: what ``GetExitCodeProcess`` reports for a running process.
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _kernel32():                                  # pragma: no cover - Windows only
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k.OpenProcess.restype = wintypes.HANDLE
    k.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    k.GetExitCodeProcess.restype = wintypes.BOOL
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    k.CloseHandle.restype = wintypes.BOOL
    return k, ctypes.get_last_error


def _pid_alive_windows(pid: int, api=None) -> bool:
    """Ask Windows whether ``pid`` is running, without sending it anything.

    ``api`` is ``(kernel32, get_last_error)``; tests pass a stand-in.
    """
    import ctypes  # noqa: PLC0415

    kernel32, last_error = _kernel32() if api is None else api
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        # Access denied means it exists and belongs to someone else, like EPERM.
        return last_error() == _ERROR_ACCESS_DENIED
    try:
        code = ctypes.c_ulong(0)
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _pid_alive_posix(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:      # exists, owned by someone else
        return True
    except (OSError, ValueError, TypeError, OverflowError):
        return False
    return True


def pid_alive(pid) -> bool:
    """True when a process with this pid exists.  Sends it nothing, on any platform."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if IS_WINDOWS:
        try:
            return _pid_alive_windows(pid)
        except (OSError, ValueError, OverflowError, AttributeError):
            return False
    return _pid_alive_posix(pid)


def detached_kwargs() -> dict:
    """``Popen`` options that keep a terminal's Ctrl-C from reaching the child too.

    POSIX: a session of its own.  Windows: a process group of its own, which is also
    what lets :func:`interrupt` reach that child, and only that child.
    """
    if IS_WINDOWS:
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)}
    return {"start_new_session": True}


def interrupt_signal() -> int:
    """The signal that asks a child started with :func:`detached_kwargs` to finish.

    SIGINT on POSIX.  On Windows it is ``CTRL_BREAK_EVENT``, the one console event
    that can be aimed at a single process group; Python delivers it as ``SIGBREAK``,
    which plume's sessions handle the same way as SIGINT.
    """
    if IS_WINDOWS:
        return int(getattr(signal, "CTRL_BREAK_EVENT", 1))
    return int(signal.SIGINT)


def interrupt(proc) -> None:
    """Ask a child to finish gracefully: Ctrl-C's ending, sent from here."""
    proc.send_signal(interrupt_signal())


def closing_signals() -> tuple[str, ...]:
    """The names of signals that mean "the terminal or the system wants this gone".

    SIGTERM everywhere it can be caught, SIGHUP where terminals send it (POSIX), and
    SIGBREAK where Windows sends Ctrl-Break and a closing console.
    """
    return tuple(n for n in ("SIGTERM", "SIGHUP", "SIGBREAK") if hasattr(signal, n))


#: Set by a parent that started this process with :func:`detached_kwargs`, to its
#: own pid.  On Windows a console event can only be aimed at one process when that
#: process leads its own group, and there is no API to ask whether it does.
OWN_GROUP_ENV = "PLUME_OWN_GROUP"

# Read once and removed, so nothing this process starts inherits a claim that is
# only true of this process.
_OWN_GROUP_MARK = os.environ.pop(OWN_GROUP_ENV, None)


def own_group_env() -> dict:
    """The env entry a parent adds when it starts a child with :func:`detached_kwargs`."""
    return {OWN_GROUP_ENV: str(os.getpid())}


def leads_own_group(mark: str | None = None, parent: int | None = None) -> bool:
    """True when this process was started as the leader of its own process group.

    Only when the mark names our actual parent.  A copy inherited by a grandchild, or
    an interpreter started through a launcher that is itself the group leader, has a
    different parent and is not a leader, so it says False.  False is the safe answer:
    it costs ``plume stop`` its Ctrl-Break rung, never a stray Ctrl-Break.
    """
    mark = _OWN_GROUP_MARK if mark is None else mark
    parent = os.getppid() if parent is None else parent
    try:
        return mark is not None and int(mark) == int(parent) and int(parent) > 0
    except (TypeError, ValueError):
        return False


def replace(src, dst, *, attempts: int = 5, delay: float = 0.02, **kw) -> None:
    """``os.replace``, retried briefly on Windows when a reader holds ``dst`` open.

    Windows refuses to replace a file another process has open without
    ``FILE_SHARE_DELETE``, which is how Python opens files.  The reader lets go in
    milliseconds, so a short retry keeps the update instead of dropping it.  POSIX
    replaces regardless, so there this is ``os.replace`` exactly.
    """
    if not IS_WINDOWS:
        os.replace(src, dst, **kw)
        return
    import time  # noqa: PLC0415

    for i in range(max(1, int(attempts))):
        try:
            os.replace(src, dst, **kw)
            return
        except PermissionError:
            if i + 1 >= attempts:
                raise
            time.sleep(delay * (i + 1))
