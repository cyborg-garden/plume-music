/* Where each hairline's click target goes. Pure, so it can be tested without a browser.

   A hairline never leaves its true bar, but its target may sit beside it. Events on one
   bar share one line and split the space around it, so none of them hides another. */

const SLICE = 12, EDGE = 24;

/* xs: each event's true x, in order. Returns { left, width } for each. */
export function hitBoxes(xs) {
  const groups = [];
  xs.forEach(function (x, i) {
    const last = groups[groups.length - 1];
    if (last && Math.abs(last.x - x) < 0.5) last.items.push(i);
    else groups.push({ x: x, items: [i] });
  });
  const out = new Array(xs.length);
  groups.forEach(function (g, k) {
    const want = Math.max(EDGE, SLICE * g.items.length) / 2;
    const lo = k ? (groups[k - 1].x + g.x) / 2 : g.x - want;
    const hi = k < groups.length - 1 ? (groups[k + 1].x + g.x) / 2 : g.x + want;
    const left = Math.max(lo, g.x - want), right = Math.min(hi, g.x + want);
    const each = (right - left) / g.items.length;
    g.items.forEach(function (i, n) { out[i] = { left: left + n * each, width: each }; });
  });
  return out;
}

/* The pieces of a hairline at x, from top to bottom, that do not cross a label.
   boxes: { left, right, top, bottom } in the same coordinates. */
export function lineSpans(top, bottom, x, boxes) {
  let spans = [[top, bottom]];
  boxes.forEach(function (box) {
    if (x < box.left || x > box.right) return;
    const next = [];
    spans.forEach(function (span) {
      if (box.top > span[0]) next.push([span[0], Math.min(span[1], box.top)]);
      if (box.bottom < span[1]) next.push([Math.max(span[0], box.bottom), span[1]]);
    });
    spans = next;
  });
  return spans;
}

/* The next event for an arrow key, or -1 if this key does not move. */
export function nextIndex(key, index, count) {
  if (key === 'ArrowRight' || key === 'ArrowDown') return Math.min(count - 1, index + 1);
  if (key === 'ArrowLeft' || key === 'ArrowUp') return Math.max(0, index - 1);
  if (key === 'Home') return 0;
  if (key === 'End') return count - 1;
  return -1;
}
