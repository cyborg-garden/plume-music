"""How plume runs itself as a child, from an install and from a frozen bundle.

One place builds the command line, so every child gets the same protection:
only this plume runs, whatever folder the parent was started from.

Installed (a venv, pip, an editable checkout): ``python -P -m plume``.  Without
``-P``, ``-m`` puts the working folder first on the import path, and a
``plume.py`` there would run with the consent the parent was given.

Frozen (a one-folder bundle: ``sys.frozen`` is set and ``sys.executable`` is the
bundle's own binary): that binary, followed by plain CLI arguments.  It has no
``-P`` to pass, so the binary's entry point calls :func:`main` first.  That
drops the working folder from the import path, then runs the CLI when there are
arguments, and loads the window only when there are none.

One-file builds are out of scope.  They unpack to a new folder on each start,
and a child would unpack a second copy.

The process-group side of a child is unchanged and lives in
:mod:`plume.oscompat`: ``detached_kwargs`` and ``own_group_env``.  The argv
built here is the process the parent starts directly, with no launcher between,
so the parent-pid mark stays true of it.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable

#: What macOS adds to a double-clicked app's arguments on some versions.  Not a CLI word.
_FINDER_ARG = "-psn_"


def is_frozen() -> bool:
    """True inside a frozen bundle, where there is no ``python`` to run ``-m plume`` with."""
    return bool(getattr(sys, "frozen", False))


def plume_argv() -> tuple[str, ...]:
    """The start of the command line that runs this plume as a child.

    Read at call time, not import time, so a test can stand in a frozen layout.
    """
    if is_frozen():
        return (sys.executable,)
    return (sys.executable, "-P", "-m", "plume")


def cli_args(argv: list[str]) -> list[str]:
    """The arguments that are plume's own, without what a desktop adds on a double-click."""
    return [a for a in argv if not a.startswith(_FINDER_ARG)]


def _bundle_roots() -> tuple[str, ...]:
    """The folders a frozen bundle's own code lives in."""
    roots = [os.path.dirname(os.path.abspath(sys.executable))]
    unpacked = getattr(sys, "_MEIPASS", None)
    if unpacked:
        roots.append(os.path.abspath(unpacked))
    return tuple(os.path.normcase(os.path.realpath(r)) for r in roots)


def _inside(path: str, roots: tuple[str, ...]) -> bool:
    for root in roots:
        try:
            if os.path.commonpath([path, root]) == root:
                return True
        except ValueError:          # different drives
            continue
    return False


def drop_working_folder(path: list[str] | None = None) -> None:
    """Take the working folder off the import path, in place: what ``-P`` does for an install.

    An entry that is the bundle's own folder stays, even when the bundle is also the
    working folder: that code is the app.
    """
    path = sys.path if path is None else path
    try:
        cwd = os.path.normcase(os.path.realpath(os.getcwd()))
    except OSError:
        cwd = None
    roots = _bundle_roots() if is_frozen() else ()
    kept = []
    for entry in path:
        if not isinstance(entry, str):
            kept.append(entry)
            continue
        if entry == "":
            continue
        resolved = os.path.normcase(os.path.realpath(entry))
        if cwd is not None and resolved == cwd and not _inside(resolved, roots):
            continue
        kept.append(entry)
    path[:] = kept


def main(argv: list[str] | None = None, window: Callable[[], int | None] | None = None) -> int:
    """The entry point of a frozen binary: the CLI when there are arguments, else the window.

    ``window`` is called only when there are no arguments, so it is the one place window
    code may be imported: pass a function that imports inside itself.  A child started
    with :func:`plume_argv` always has arguments, and never loads a window.
    """
    args = cli_args(sys.argv[1:] if argv is None else list(argv))
    drop_working_folder()
    if args or window is None:
        from .cli import main as cli_main
        return int(cli_main(args) or 0)
    return int(window() or 0)
