"""Mini-notation parser (Strudel subset). SPEC.md section 2.

One pattern string = one bar = one cycle.

    pattern  := layer ("," layer)*
    layer    := seq
    seq      := elem+
    elem     := atom | group | elem "*" INT | elem "@" INT | alt
    group    := "[" seq "]"
    alt      := "<" atom+ ">"
    atom     := TOKEN | "~"
    TOKEN    := /[A-Za-z#][A-Za-z0-9#b.]*/  (optional ":v<float>" velocity suffix)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace


class NotationError(ValueError):
    """Pattern syntax error, with 0-based character position in .pos."""

    def __init__(self, msg: str, pos: int):
        super().__init__(f"{msg} (at position {pos})")
        self.pos = pos


@dataclass(frozen=True)
class Step:
    start: float      # 0..1 within the bar
    dur: float        # fraction of bar
    token: str
    cycle_alts: tuple[str, ...] | None = None  # set when token came from <...>; token = alts[0]


_TOKEN_RE = re.compile(r"[A-Za-z#][A-Za-z0-9#b.]*")
_VEL_RE = re.compile(r":v(\d+(?:\.\d+)?|\.\d+)")
_INT_RE = re.compile(r"\d+")

# AST node shapes:
#   ("atom", token)            token includes any :v suffix; "~" = rest
#   ("alt", (tok, tok, ...))   per-cycle alternation
#   ("rep", node, n)           node repeated n times within its slot
#   ("group", elems)           elems = list of (node, weight)


class _Parser:
    def __init__(self, s: str):
        self.s = s
        self.i = 0

    def _skip_ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def _peek(self) -> str | None:
        return self.s[self.i] if self.i < len(self.s) else None

    def parse_pattern(self) -> list[list[tuple]]:
        layers = [self.parse_seq()]
        self._skip_ws()
        while self._peek() == ",":
            self.i += 1
            layers.append(self.parse_seq())
            self._skip_ws()
        if self.i < len(self.s):
            raise NotationError(f"unexpected character {self.s[self.i]!r}", self.i)
        return layers

    def parse_seq(self) -> list[tuple]:
        elems: list[tuple] = []
        while True:
            self._skip_ws()
            c = self._peek()
            if c is None or c in ",]>":
                break
            elems.append(self.parse_elem())
        if not elems:
            raise NotationError("empty sequence", self.i)
        return elems

    def parse_elem(self) -> tuple:
        node = self.parse_primary()
        weight = 1
        while self._peek() in ("*", "@"):
            op = self.s[self.i]
            self.i += 1
            m = _INT_RE.match(self.s, self.i)
            if not m:
                raise NotationError(f"expected integer after {op!r}", self.i)
            n = int(m.group())
            self.i = m.end()
            if n < 1:
                raise NotationError(f"{op!r} count must be >= 1", self.i - len(m.group()))
            if op == "*":
                node = ("rep", node, n)
            else:
                weight = n
        return (node, weight)

    def parse_primary(self) -> tuple:
        c = self._peek()
        if c is None:
            raise NotationError("unexpected end of pattern", self.i)
        if c == "[":
            open_pos = self.i
            self.i += 1
            elems = self.parse_seq()
            if self._peek() != "]":
                raise NotationError("unclosed '['", open_pos)
            self.i += 1
            return ("group", elems)
        if c == "<":
            open_pos = self.i
            self.i += 1
            alts: list[str] = []
            while True:
                self._skip_ws()
                nxt = self._peek()
                if nxt == ">":
                    break
                if nxt is None:
                    raise NotationError("unclosed '<'", open_pos)
                if nxt in "[]<,*@":
                    raise NotationError(
                        f"expected token or '~' inside <...>, got {nxt!r}", self.i)
                alts.append(self.parse_atom_token())
            if not alts:
                raise NotationError("empty <...> alternation", open_pos)
            self.i += 1  # consume '>'
            return ("alt", tuple(alts))
        if c == "~":
            self.i += 1
            return ("atom", "~")
        m = _TOKEN_RE.match(self.s, self.i)
        if m:
            return ("atom", self._finish_token(m))
        raise NotationError(f"unexpected character {c!r}", self.i)

    def parse_atom_token(self) -> str:
        """A single atom inside <...>: TOKEN(+:v) or '~'."""
        if self._peek() == "~":
            self.i += 1
            return "~"
        m = _TOKEN_RE.match(self.s, self.i)
        if not m:
            raise NotationError(
                f"expected token or '~', got {self._peek()!r}", self.i)
        return self._finish_token(m)

    def _finish_token(self, m: re.Match) -> str:
        self.i = m.end()
        text = m.group()
        if self._peek() == ":":
            vm = _VEL_RE.match(self.s, self.i)
            if not vm:
                raise NotationError("expected ':v<float>' velocity suffix", self.i)
            v = float(vm.group(1))
            if not 0.0 <= v <= 1.0:
                raise NotationError(f"velocity {v} out of range 0..1", self.i)
            text += vm.group()
            self.i = vm.end()
        return text


def _expand_seq(elems: list[tuple], start: float, dur: float, out: list[Step]) -> None:
    total = sum(w for _, w in elems)
    t = start
    for node, w in elems:
        span = dur * (w / total)
        _expand_node(node, t, span, out)
        t += span


def _expand_node(node: tuple, start: float, dur: float, out: list[Step]) -> None:
    kind = node[0]
    if kind == "atom":
        if node[1] != "~":
            out.append(Step(start=start, dur=dur, token=node[1]))
    elif kind == "alt":
        alts = node[1]
        out.append(Step(start=start, dur=dur, token=alts[0], cycle_alts=alts))
    elif kind == "rep":
        _, inner, n = node
        sub = dur / n
        for k in range(n):
            _expand_node(inner, start + k * sub, sub, out)
    elif kind == "group":
        _expand_seq(node[1], start, dur, out)
    else:  # pragma: no cover
        raise AssertionError(f"unknown node kind {kind!r}")


def parse(pattern: str) -> list[Step]:
    """Parse one bar of mini-notation into Steps sorted by start; rests omitted.

    Steps from <...> alternations are always included (token = first
    alternative) — resolve per bar with steps_for_cycle().
    """
    layers = _Parser(pattern).parse_pattern()
    out: list[Step] = []
    for elems in layers:
        _expand_seq(elems, 0.0, 1.0, out)
    out.sort(key=lambda st: (st.start, st.dur, st.token))
    return out


def steps_for_cycle(steps: list[Step], cycle: int) -> list[Step]:
    """Resolve <...> alternations for bar number `cycle` (0-based)."""
    res: list[Step] = []
    for st in steps:
        if st.cycle_alts:
            tok = st.cycle_alts[cycle % len(st.cycle_alts)]
            if tok == "~":
                continue
            res.append(replace(st, token=tok))
        else:
            res.append(st)
    return res
