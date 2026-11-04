/* One carved stone per take, lit from one side.

   A stone is the lower envelope of flat planes, which makes it a true convex block:
   straight edges, hard creases, a flat base it stands on, and a hewn top rather than
   a tip. Take facts set the big shapes. The seed sets the rest, and that part is
   only identity.
     height = length, lean = tempo, deep cut = where the band came in,
     the cut's slope = major or minor, one chip in the left edge per cue that landed.

   Everything is painted in slices of about 5 ms, so nothing here blocks the page. */

export function rng(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hex(s) {
  s = (s || '').trim();
  if (s[0] !== '#') return [128, 128, 128];
  return [parseInt(s.substr(1, 2), 16), parseInt(s.substr(3, 2), 16), parseInt(s.substr(5, 2), 16)];
}
export function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
export function tok(name) { return hex(cssVar(name)); }

const NOISE = (function () { const r = rng(991), a = new Float32Array(4096); for (let i = 0; i < 4096; i++) a[i] = r() - 0.5; return a; })();
function vnoise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y);
  let fx = x - xi, fy = y - yi;
  const g = function (a, b) { return NOISE[(Math.imul(a, 73856) ^ Math.imul(b, 19349)) & 4095]; };
  fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
  return (g(xi, yi) * (1 - fx) + g(xi + 1, yi) * fx) * (1 - fy) + (g(xi, yi + 1) * (1 - fx) + g(xi + 1, yi + 1) * fx) * fy;
}

/* The block, in units where 1 is nearly the canvas height. y is up, y = 0 is the ground.
   Each plane is z = a*x + b*y + c, and z is the lowest of them, so z = 0 draws the outline.
   Each edge plane also has a tool direction for the chisel marks on its face. */
export function blockPlanes(p, r) {
  const jit = function (amount) { return (r() - 0.5) * amount; };
  const h = p.height, w = 0.34 + 0.24 * h;       /* a tall stone is broader, never a spike */
  const planes = [];
  function edge(deg, along, slope) {
    /* An edge at this outward angle, through the point (along.x, along.y). */
    const a = deg * Math.PI / 180, nx = Math.cos(a), ny = Math.sin(a);
    planes.push({ a: -nx * slope, b: -ny * slope, c: (along[0] * nx + along[1] * ny) * slope, dir: deg + 90 + jit(30) });
  }
  const left = -w / 2, right = w / 2;
  /* The base stays flat and wide. The sides draw in a little as they rise. */
  edge(-90, [0, 0], 1.6);
  edge(4 + jit(6), [right + jit(0.02), h * 0.3], 0.62 + r() * 0.4);
  edge(176 + jit(6), [left + jit(0.02), h * 0.3], 0.62 + r() * 0.4);
  /* A flat hewn top, tilted one way, with a broad shoulder cut off each corner. */
  const tilt = (r() < 0.5 ? -1 : 1) * (5 + r() * 9);
  edge(90 + tilt, [jit(0.04), h], 0.7 + r() * 0.3);
  edge(38 + jit(14), [right - 0.2 * w + jit(0.03), h * (0.86 + jit(0.05))], 0.7 + r() * 0.4);
  edge(142 + jit(14), [left + 0.2 * w + jit(0.03), h * (0.86 + jit(0.05))], 0.7 + r() * 0.4);
  /* The front: a few broad faces, tilted differently, so light breaks across creases. */
  const T = 0.075 + 0.02 * h, faces = 3 + Math.floor(r() * 2);
  for (let i = 0; i < faces; i++) {
    const a = (i / faces) * Math.PI * 2 + r() * 1.4, tiltBy = 0.1 + r() * 0.2;
    const cx = jit(w * 0.5), cy = h * (0.25 + r() * 0.5);
    const ga = Math.cos(a) * tiltBy, gb = Math.sin(a) * tiltBy;
    let c = T + 0.012 * i - ga * cx - gb * cy;
    /* No face may cut below the ground across the base, or the stone hovers. */
    const atBase = Math.min(ga * left, ga * right) + c;
    if (atBase < T * 0.5) c += T * 0.5 - atBase;
    planes.push({ a: ga, b: gb, c: c, dir: 20 + r() * 140 });
  }
  return { planes: planes, w: w, h: h, T: T };
}

function* stoneGen(cv, p) {
  const W = cv.width, H = cv.height, u = H * 0.93, ax = W * 0.5 - p.lean * p.height * u * 0.42, ay = H * 0.965;
  const r = rng(p.seed * 13 + 5);
  const block = blockPlanes(p, r), planes = block.planes, n = planes.length;

  /* Precompute in pixels: x and y here are canvas pixels, with the lean as a shear. */
  const toUnit = function (x, y) { const yy = (ay - y) / u; return [(x - ax) / u - p.lean * yy, yy]; };

  /* The deep cut: a V-shaped groove chiselled across the front from the left edge, with
     one steep wall and one long one. It notches the outline and dies out before the
     far side, so the stone stays one stone. */
  let cut = null;
  if (p.cut !== null && p.cut !== undefined) {
    const ang = (p.minor ? -1 : 1) * 0.2, cy = p.cut * block.h;
    cut = { x: -block.w * 0.62, y: cy, ca: Math.cos(ang), sa: Math.sin(ang), L: block.w * (0.96 + r() * 0.1), D: 0.034, steep: 0.009, wide: 0.04 };
  }
  function groove(x, y) {
    if (!cut) return 0;
    const dx = x - cut.x, dy = y - cut.y, along = dx * cut.ca + dy * cut.sa, t = dy * cut.ca - dx * cut.sa;
    if (along <= 0 || along >= cut.L) return 0;
    let e = Math.min(1, (cut.L - along) / (cut.L * 0.3)); e = e * e * (3 - 2 * e);
    const side = t > 0 ? t / cut.steep : -t / cut.wide;
    return side >= 1 ? 0 : cut.D * e * (1 - side);
  }

  function solid(x, y) {
    let z = Infinity, id = 0;
    for (let k = 0; k < n; k++) { const P = planes[k], v = P.a * x + P.b * y + P.c; if (v < z) { z = v; id = k; } }
    return [z, id];
  }

  /* One angular chip out of the left edge per cue that landed. */
  const chips = [];
  for (let i = 0; i < p.nicks; i++) {
    const y = block.h * (0.14 + 0.66 * (i + 0.3 + r() * 0.4) / p.nicks);
    let x = -block.w;
    while (x < 0 && solid(x, y)[0] <= 0) x += 0.002;
    /* Many chips get smaller, so they stay chips and do not become a row of teeth. */
    chips.push({ x: x, y: y, r: (0.021 + r() * 0.012) * Math.min(1, Math.sqrt(3 / p.nicks)) });
  }

  /* The surface. */
  const z = new Float32Array(W * H).fill(-1), ids = new Int16Array(W * H).fill(-1);
  let minX = W, maxX = 0, minY = H;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const q = toUnit(x, y), s = solid(q[0], q[1]);
      let v = s[0], id = s[1];
      const g = groove(q[0], q[1]);
      if (g > 0) { v -= g; id = 40; }
      for (let c = 0; c < chips.length; c++) {
        const C = chips[c], d = (Math.abs(q[0] - C.x) + Math.abs(q[1] - C.y)) - C.r;
        const zc = d * 0.9;
        if (zc < v) { v = zc; id = 50 + c; }
      }
      if (v <= 0) continue;
      /* Chisel marks: short parallel strokes, a different way on each face. */
      const P = planes[Math.min(id, n - 1)], a = (P.dir || 0) * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
      const along = q[0] * ca + q[1] * sa, across = q[1] * ca - q[0] * sa;
      const mark = vnoise(along * 9 + id * 7, across * 95) * 0.0034 + vnoise(q[0] * 34, q[1] * 34) * 0.0036 + vnoise(q[0] * 140, q[1] * 140) * 0.0012;
      const i = y * W + x;
      z[i] = (v + mark * Math.min(1, v * 22)) * u;
      ids[i] = id;
      if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y;
    }
    yield;
  }

  const ctx = cv.getContext('2d');
  if (!ctx) return;
  const img = ctx.createImageData(W, H), px = img.data, hi = tok('--mass-hi'), mid = tok('--mass-mid'), lo = tok('--mass-lo');
  /* Raked light from the upper right. */
  const lx = 0.72, ly = -0.42, lz = 0.55;
  const Z = function (a, c) { if (a < 0 || c < 0 || a >= W || c >= H) return -1; return z[c * W + a]; };
  const step = Math.max(1.2, u / 240);
  /* The edge turns away from the light, so a pale face still has a silhouette. */
  const rim = parseFloat(cssVar('--rim')), rimDepth = u * 0.012;
  for (let y = minY; y < H; y++) {
    for (let x = minX; x <= maxX; x++) {
      const id = y * W + x, zz = z[id];
      if (zz < 0) continue;
      let e0 = Z(x + 1, y), w0 = Z(x - 1, y), s0 = Z(x, y + 1), n0 = Z(x, y - 1);
      if (e0 < 0) e0 = zz; if (w0 < 0) w0 = zz; if (s0 < 0) s0 = zz; if (n0 < 0) n0 = zz;
      const nx = (w0 - e0) * 0.5, ny = (n0 - s0) * 0.5, nl = Math.sqrt(nx * nx + ny * ny + 1);
      const diff = Math.max(0, (nx * lx + ny * ly + lz) / nl);
      /* Cast shadows, which is what makes the cut read as depth. */
      let sh = 1;
      for (let k = 1; k <= 16; k++) {
        const hh = Z(Math.round(x + 0.86 * step * k), Math.round(y - 0.5 * step * k));
        const over = hh - zz - k * step * 0.62;
        if (over > 0) sh = Math.min(sh, 1 - 0.72 * Math.min(1, over / (u * 0.01)));
      }
      /* A bright line where two faces meet, as on any cut edge. */
      const me = ids[id];
      const crease = (ids[id + 1] !== me && ids[id + 1] >= 0) || (y + 1 < H && ids[id + W] !== me && ids[id + W] >= 0);
      const fine = NOISE[(Math.imul(x, 2654435) ^ Math.imul(y, 40503)) & 4095] * 0.04;
      let tone = 0.05 + 0.1 * sh + Math.pow(diff, 1.5) * sh * 1.02 + fine + (crease && sh > 0.6 ? 0.07 : 0);
      tone = Math.max(0, Math.min(0.96, tone));
      if (rim < 1 && zz < rimDepth) tone *= rim + (1 - rim) * (zz / rimDepth);
      const A = tone < 0.5 ? lo : mid, B = tone < 0.5 ? mid : hi, f = tone < 0.5 ? tone * 2 : (tone - 0.5) * 2;
      const q = id * 4;
      px[q] = A[0] + (B[0] - A[0]) * f; px[q + 1] = A[1] + (B[1] - A[1]) * f; px[q + 2] = A[2] + (B[2] - A[2]) * f;
      px[q + 3] = 255 * Math.min(1, zz / 0.9);
    }
    yield;
  }

  /* It stands on something: a soft shadow thrown away from the light. */
  ctx.clearRect(0, 0, W, H);
  const off = document.createElement('canvas'); off.width = W; off.height = H;
  const offCtx = off.getContext('2d');
  offCtx.putImageData(img, 0, 0);
  const sw = block.w * u * 0.86, sx = ax - sw * 0.16, sy = ay + H * 0.002;
  ctx.save();
  ctx.translate(sx, sy); ctx.scale(1, 0.05);
  const glow = ctx.createRadialGradient(0, 0, sw * 0.2, 0, 0, sw);
  const inkRgb = lo.join(',');
  glow.addColorStop(0, 'rgba(' + inkRgb + ',.55)'); glow.addColorStop(1, 'rgba(' + inkRgb + ',0)');
  ctx.fillStyle = glow;
  ctx.fillRect(-sw, -sw, sw * 2, sw * 2);
  ctx.restore();
  ctx.drawImage(off, 0, 0);
}

const jobs = { q: [], on: false };
function pump() {
  const t = performance.now();
  while (jobs.q.length && performance.now() - t < 5) {
    if (jobs.q[0].next().done) jobs.q.shift();
  }
  if (jobs.q.length) setTimeout(pump, 0); else jobs.on = false;
}
export function clearStones() { jobs.q.length = 0; }
/* keep: leave the old painting up until the new one replaces it. */
export function queueStone(cv, portrait, keep) {
  const ctx = cv.getContext('2d');
  if (ctx && !keep) ctx.clearRect(0, 0, cv.width, cv.height);
  jobs.q.push(stoneGen(cv, portrait));
  if (!jobs.on) { jobs.on = true; setTimeout(pump, 0); }
}
/* True once every queued stone is painted. */
export function stonesIdle() { return jobs.q.length === 0; }
