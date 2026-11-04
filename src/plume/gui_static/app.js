/* The Record and the shelf. Everything shown comes from the bridge, already filtered
   and already in sentences. Words go in with textContent only. */
import { clearStones, cssVar, queueStone, rng, stonesIdle, tok } from '/static/stone.js';
import { hitBoxes, lineSpans, nextIndex } from '/static/layout.js';
import { lostWords, pinnedRoute } from '/static/session.js';

const $ = function (id) { return document.getElementById(id); };
const S = { csrf: '', take: null, selected: null, roastOn: true, screen: 'loading', nav: 0,
  shelfKey: null, shelfDone: false, shelfTakes: [], lastTake: null, shelfScroll: 0, cameByLink: false,
  nameFor: null, nameSent: '', keepTurn: {}, nameTurn: {}, moving: false, failed: null, newNames: {} };

function load(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function save(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private window */ } }

/* The page key. The redirect hands it over in the fragment, which no server receives.
   It lives in this tab only, because the cookie alone reaches every server on 127.0.0.1. */
function takePageKey() {
  const found = /^#k=([0-9a-f]{64})$/.exec(location.hash);
  if (found) {
    history.replaceState(null, '', location.pathname);
    S.cameByLink = true;
    try { sessionStorage.setItem('plume-key', found[1]); } catch (e) { /* kept in memory */ }
    return found[1];
  }
  try { return sessionStorage.getItem('plume-key') || ''; } catch (e) { return ''; }
}

async function api(path) {
  const r = await fetch(path, { credentials: 'same-origin', headers: { 'X-Plume-Token': S.csrf } });
  if (!r.ok) { const e = new Error('http ' + r.status); e.status = r.status; throw e; }
  return r.json();
}
async function post(op, body) {
  const r = await fetch('/api/' + op, {
    method: 'POST', credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', 'X-Plume-Token': S.csrf },
    body: JSON.stringify(body)
  });
  if (!r.ok) { const e = new Error('http ' + r.status); e.status = r.status; throw e; }
  return r.json();
}

function fmtTime(s) { const m = Math.floor(s / 60), r = Math.floor(s % 60); return m + ':' + (r < 10 ? '0' : '') + r; }
function fmtSize(n) { return n >= 1048576 ? Math.round(n / 1048576) + ' MB' : n >= 1024 ? Math.round(n / 1024) + ' KB' : n + ' B'; }
function fmtDate(epoch) {
  if (!epoch) return '';
  const d = new Date(epoch * 1000), p = function (v) { return (v < 10 ? '0' : '') + v; };
  return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate());
}
function metaOf(t) {
  const bits = [];
  if (t.key) bits.push(t.key);
  if (t.bars && t.bpm !== null) bits.push(t.bpm + ' bpm');
  bits.push(fmtTime(t.seconds));
  return bits;
}

function setScreen(name) {
  S.screen = name;
  document.body.setAttribute('data-screen', name);
  ['record', 'shelf', 'problem'].forEach(function (s) { $(s).hidden = s !== name; });
  if (name !== 'record') stopAudio();
  window.scrollTo(0, 0);
}
/* A moment's wait says so. Anything quicker says nothing, so it does not flicker. */
let busyTimer = 0;
function busy(on) {
  clearTimeout(busyTimer);
  if (on) busyTimer = setTimeout(function () { $('busy').hidden = false; }, 150);
  else $('busy').hidden = true;
}
function problem(err) {
  busy(false);
  setScreen('problem');
  const lost = err && err.status === 401 ? lostWords(!!S.csrf) : null;
  $('problem-title').textContent = lost ? lost.title : 'That did not load.';
  $('problem-text').textContent = lost
    ? lost.text
    : 'The take may have moved. Is plume gui still running in your terminal? If it is not, plume stop does not need it, and neither do your takes.';
}

/* ------------------------------------------------------------------ the record */

function paintLine(take, cssW, cssH) {
  const cv = $('terrain'), dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = Math.round(cssW * dpr); cv.height = Math.round(cssH * dpr);
  const ctx = cv.getContext('2d'), points = take.band, n = points.length;
  const cs = getComputedStyle(document.body), x0 = parseFloat(cs.paddingLeft) + 6, x1 = cssW - parseFloat(cs.paddingRight) - 6;
  const t0 = n ? points[0].t_s : 0, t1 = n ? points[n - 1].t_s : 0, span = t1 > t0 ? t1 - t0 : 1;
  const xAt = function (t) { return x0 + (x1 - x0) * Math.max(0, Math.min(1, (t - t0) / span)); };
  const ground = cssH - 8, scale = Math.min(cssH * 0.34, 130);
  const yOf = function (point) { return ground - point.level * scale; };
  if (!ctx || !n) return { xAt: xAt, yAt: function () { return ground - 40; }, peak: ground - 40, x0: x0 };
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const trace = function () {
    ctx.beginPath();
    ctx.moveTo(0, yOf(points[0]));
    points.forEach(function (point, i) {
      const x = xAt(point.t_s), y = yOf(point);
      if (i === 0) { ctx.lineTo(x, y); return; }
      const px = xAt(points[i - 1].t_s), py = yOf(points[i - 1]);
      ctx.quadraticCurveTo((x + px) / 2, py, (x * 0.75 + px * 0.25), (py + y) / 2);
      ctx.lineTo(x, y);
    });
    ctx.lineTo(cssW, yOf(points[n - 1]));
  };
  trace(); ctx.lineTo(cssW, cssH); ctx.lineTo(0, cssH); ctx.closePath();
  ctx.fillStyle = cssVar('--land'); ctx.fill();
  trace(); ctx.strokeStyle = cssVar('--ridge'); ctx.lineWidth = 1; ctx.stroke();
  let peak = cssH;
  points.forEach(function (point) { peak = Math.min(peak, yOf(point)); });
  const yAt = function (t) {
    let best = points[0];
    points.forEach(function (point) { if (point.t_s <= t) best = point; });
    return yOf(best);
  };
  return { xAt: xAt, yAt: yAt, peak: peak, x0: x0 };
}

function drawSheet() {
  const take = S.take, r = $('sheet').getBoundingClientRect(), g = paintLine(take, r.width, r.height);
  const wrap = $('events'), list = $('ev-list');
  const had = document.activeElement && document.activeElement.id;
  wrap.textContent = ''; list.textContent = '';
  const phone = window.innerWidth < 600, empty = !take.bars;
  $('sheet-empty').hidden = !empty;
  $('sheet-empty').textContent = take.unread ? 'This log is too big to draw here.'
    : take.live ? 'Nothing has played yet.' : 'Nothing played in this take.';
  /* On a phone the list is the only control, so it is always open. */
  if (phone) $('ev-more').open = true;
  const hearY = g.peak - 18, top = { plan: hearY - 16, cue: hearY - 40, end: hearY - 64 };
  const evs = take.events.map(function (e) { return Object.assign({ x: g.xAt(e.t_s) }, e); });
  /* A label never moves a line off its true bar. Crowded labels give way, by rank. */
  const rank = { end: 0, cue: 1, plan: 2 }, placed = [], showLabel = {};
  evs.slice().sort(function (a, b) { return rank[a.tier] - rank[b.tier] || a.bar - b.bar; }).forEach(function (e) {
    showLabel[e.id] = placed.every(function (x) { return Math.abs(x - e.x) >= 15; });
    if (showLabel[e.id]) placed.push(e.x);
  });
  /* The hearing line: drawn up to the bar the band came in, and no further. */
  const hl = $('hear-line'), tick = $('hear-tick'), yes = $('hear-yes'), no = $('hear-no');
  const tap = !empty && take.tap_start;
  [hl, tick, yes, no].forEach(function (el) { el.hidden = !tap; });
  if (tap) {
    /* Only a room mic loses the player to the band. On direct input or MIDI the line runs on. */
    const hearsOn = take.route === 'direct' || take.route === 'midi';
    const end = take.band_t_s !== null && !hearsOn ? take.band_t_s : take.end_t_s, xEnd = g.xAt(end), xStart = g.xAt(take.start_t_s);
    hl.style.left = xStart + 'px'; hl.style.top = hearY + 'px'; hl.style.width = Math.max(0, xEnd - xStart) + 'px';
    tick.style.left = xEnd + 'px'; tick.style.top = (hearY - 4) + 'px';
    yes.textContent = 'plume can hear';
    no.textContent = "can't hear you from here on";
    yes.style.left = (xStart - 3) + 'px'; yes.style.top = (hearY + 2) + 'px';
    no.style.left = (xEnd + 5) + 'px'; no.style.top = (hearY + 2) + 'px';
    yes.hidden = (xEnd - xStart) < 120;
    no.hidden = take.band_t_s === null || hearsOn;
    /* Never let the label run off a narrow screen. */
    if (!no.hidden && xEnd + 5 + no.offsetWidth > r.width - 4) no.style.left = Math.max(0, r.width - 4 - no.offsetWidth) + 'px';
  }
  /* A hairline stops short of a label's words instead of being drawn through them. */
  const labels = [yes, no].filter(function (el) { return !el.hidden; }).map(function (el) {
    const box = el.getBoundingClientRect();
    return { left: box.left - r.left, right: box.right - r.left, top: box.top - r.top, bottom: box.bottom - r.top };
  });
  const boxes = hitBoxes(evs.map(function (e) { return e.x; }));
  if (!empty) evs.forEach(function (e, i) {
    const box = boxes[i], lineAt = (e.x - box.left) + 'px';
    const base = g.yAt(e.t_s), t = Math.max(2, top[e.tier]);
    const b = document.createElement('button');
    b.type = 'button'; b.id = e.id; b.className = 'hl' + (e.dashed ? ' dash' : '') + (phone ? ' static' : '');
    /* One tab stop for the whole strip. Arrow keys move along it. */
    b.tabIndex = !phone && e.id === S.selected ? 0 : -1;
    if (phone) b.setAttribute('aria-hidden', 'true');
    b.setAttribute('aria-pressed', e.id === S.selected ? 'true' : 'false');
    b.setAttribute('aria-label', e.sentence);
    const height = Math.max(6, base - t);
    lineSpans(t, t + height, e.x, labels).forEach(function (span) {
      const ln = document.createElement('span'); ln.className = 'ln'; ln.style.left = lineAt;
      ln.style.top = (span[0] - t) + 'px'; ln.style.height = (span[1] - span[0]) + 'px';
      b.appendChild(ln);
    });
    if (showLabel[e.id]) {
      const lab = document.createElement('span'); lab.className = 'lab'; lab.style.left = lineAt;
      const num = document.createElement('b'); num.textContent = e.bar;
      lab.appendChild(num); lab.appendChild(document.createTextNode(e.label));
      b.appendChild(lab);
    }
    b.style.left = box.left + 'px'; b.style.width = Math.max(1, box.width) + 'px';
    b.style.top = t + 'px'; b.style.height = height + 'px';
    b.addEventListener('click', function () { selectEvent(e.id); });
    wrap.appendChild(b);
  });
  evs.forEach(function (e) {
    const li = document.createElement('li'), b = document.createElement('button');
    b.type = 'button'; b.id = 'evl-' + e.id; b.setAttribute('aria-pressed', e.id === S.selected ? 'true' : 'false');
    const num = document.createElement('b'); num.textContent = 'bar ' + e.bar;
    b.appendChild(num); b.appendChild(document.createTextNode(e.label));
    b.addEventListener('click', function () {
      selectEvent(e.id);
      $('ev-sentence').scrollIntoView({ block: 'nearest' });
    });
    li.appendChild(b); list.appendChild(li);
  });
  /* A resize rebuilds the marks. Nobody's place on them should be lost for that. */
  if (had && $(had)) $(had).focus({ preventScroll: true });
}
function showRow(row) {
  $('raw-row').textContent = row
    ? JSON.stringify(row, null, 1) + '\n\nOnly fields about what plume did. The full log is decisions.ndjson in the take folder.'
    : S.take && S.take.unread ? 'No row. The log is too big to read here.'
    : S.take && S.take.live ? 'No row yet.' : 'No row. Nothing played.';
}
function selectEvent(id) {
  S.selected = id;
  const e = S.take.events.find(function (x) { return x.id === id; });
  if (!e) return;
  const phone = window.innerWidth < 600;
  document.querySelectorAll('.hl, #ev-list button').forEach(function (b) {
    b.setAttribute('aria-pressed', (b.id === id || b.id === 'evl-' + id) ? 'true' : 'false');
    if (b.classList.contains('hl')) b.tabIndex = !phone && b.id === id ? 0 : -1;
  });
  $('ev-sentence').textContent = e.sentence;
  $('ev-note').textContent = e.note || '';
  showRow(e.row);
}
function drawRoast() {
  const ul = $('roast'); ul.textContent = '';
  $('roast-toggle').textContent = S.roastOn ? 'Turn roast off' : 'Turn roast on';
  const say = function (text) { const li = document.createElement('li'); li.textContent = text; ul.appendChild(li); };
  if (!S.roastOn) { say('Roast off. The log is still there.'); return; }
  if (!S.take.roast.length) { say(S.take.roast_empty); return; }
  S.take.roast.forEach(function (line, i) {
    const li = document.createElement('li');
    li.appendChild(document.createTextNode(line.text));
    const b = document.createElement('button');
    b.type = 'button'; b.id = 'roast-' + i; b.textContent = 'bar ' + line.bar;
    b.addEventListener('click', function () {
      if (line.event) selectEvent(line.event);
      else { $('ev-sentence').textContent = 'Bar ' + line.bar + '. Nothing changed here, but this is the row.'; $('ev-note').textContent = ''; }
      showRow(line.row);
      $('raw').open = true;
    });
    li.appendChild(b); ul.appendChild(li);
  });
}
function titleOf(t) { return t.name || t.folder; }
function refreshHead() {
  const t = S.take;
  $('rec-title').textContent = titleOf(t);
  const bits = metaOf(t);
  if (t.name) bits.unshift(t.folder);
  if (t.bytes) bits.push(fmtSize(t.bytes));
  $('rec-meta').textContent = bits.join('   ');
  const k = $('keep');
  k.classList.toggle('undo', t.kept);
  k.textContent = t.kept ? 'Undo' : 'Keep';
  $('kept-status').hidden = !t.kept;
  $('keep-panel').hidden = !t.kept;
  if (t.kept) {
    const box = $('take-name');
    if (S.nameFor !== t.id || document.activeElement !== box) {
      box.value = t.name || ''; S.nameFor = t.id; S.nameSent = box.value;
    }
    $('keep-path').textContent = t.where;
  } else S.nameFor = null;
}
function stopAudio() {
  const p = $('player');
  p.pause();
  p.removeAttribute('src');
  p.load();
  p.hidden = true;
}
/* Sound starts only from this click. Nothing else in the page sets a source. */
function drawListen() {
  const t = S.take, wrap = $('listen-buttons');
  stopAudio();
  wrap.textContent = '';
  $('listen').hidden = !t.audio.length;
  const names = { human: t.recorded_from === 'input' ? 'Your input' : 'Your mic', machine: 'The band' };
  t.audio.forEach(function (track) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'quiet'; b.id = 'listen-' + track; b.textContent = names[track];
    b.addEventListener('click', function () {
      const p = $('player');
      /* An audio element cannot set a header, so the page key goes in the query. */
      p.src = '/api/take/' + encodeURIComponent(t.id) + '/audio/' + track + '?k=' + S.csrf;
      p.hidden = false;
      p.play().catch(function () { /* the controls are there */ });
    });
    wrap.appendChild(b);
  });
}
function paintPortrait() {
  /* Clearing the queue leaves any unfinished shelf stones blank, so remember that. */
  if (!stonesIdle()) S.shelfDone = false;
  clearStones(); queueStone($('portrait'), S.take.portrait);
}

/* Screens live in the address, so Back and reload work. Only the last request to start counts. */
function routedTake() {
  const found = /^#\/take\/([0-9a-f]{16})$/.exec(location.hash);
  return found ? found[1] : null;
}
function showRoute() {
  const id = routedTake();
  if (id) openRecord(id); else openShelf();
}
function goTake(id) {
  S.lastTake = id; S.shelfScroll = window.scrollY;
  history.pushState(null, '', '#/take/' + id);
  openRecord(id);
}
function goShelf() {
  if (location.hash) history.pushState(null, '', location.pathname);
  openShelf();
}

async function openRecord(id) {
  const turn = ++S.nav;
  busy(true);
  S.moving = true;
  saveName();
  let take;
  try { take = await api('/api/take/' + encodeURIComponent(id)); } catch (err) { if (turn === S.nav) { S.moving = false; problem(err); } return; }
  if (turn !== S.nav) return;
  busy(false);
  leaveName();
  S.take = take;
  setScreen('record');
  arrived(id);
  refreshHead();
  const first = take.events.find(function (e) { return e.tier === 'cue'; })
    || take.events.find(function (e) { return e.label === 'bass and chords'; }) || take.events[0];
  S.selected = first.id;
  drawSheet(); selectEvent(S.selected); drawRoast(); drawListen(); paintPortrait();
  $('record').focus({ preventScroll: true });
}

/* A response belongs to the take it was sent for, and only the newest of its kind for
   that take counts. By the time one comes back, the page may be showing another take. */
function ticket(turns, id) { turns[id] = (turns[id] || 0) + 1; return turns[id]; }
/* A failed save outlives the screen it failed on, or it could vanish unread: it stays
   through one change of screen, in the words that fit where it now shows. */
function showing(id) { return !S.moving && S.screen === 'record' && S.take && S.take.id === id; }
function keepProblem(id, title, here, away) {
  S.failed = { id: id, here: here, away: away + title + '.', moves: 0 };
  $('keep-error').hidden = false;
  $('keep-error').textContent = showing(id) ? here : S.failed.away;
}
function clearProblem() {
  S.failed = null;
  $('keep-error').hidden = true;
}
function arrived(id) {
  S.moving = false;
  if (!S.failed || S.failed.moves > 0) { clearProblem(); return; }
  S.failed.moves += 1;
  $('keep-error').textContent = S.failed.id === id ? S.failed.here : S.failed.away;
}
async function setKept(kept) {
  const id = S.take.id, title = titleOf(S.take), turn = ticket(S.keepTurn, id), names = S.nameTurn[id];
  clearProblem();
  try {
    const doc = await post('keep', { take: id, kept: kept });
    if (turn !== S.keepTurn[id] || !S.take || S.take.id !== id) return;
    S.take.kept = doc.kept;
    /* A rename sent since then knows the name better than this answer does. */
    if (names === S.nameTurn[id]) S.take.name = doc.name;
    refreshHead();
    if (kept) $('take-name').focus();
  } catch (err) {
    if (turn !== S.keepTurn[id]) return;
    keepProblem(id, title, err.status === 401 ? lostWords(!!S.csrf).text : 'That was not saved. Can plume write in this take\'s folder?', 'Keep was not saved for ');
  }
}
async function rename(id, name) {
  const title = titleOf(S.take), turn = ticket(S.nameTurn, id);
  clearProblem();
  try {
    const doc = await post('rename', { take: id, name: name });
    if (turn === S.nameTurn[id]) renamedOnShelf(id, doc.name);
    if (turn !== S.nameTurn[id] || !S.take || S.take.id !== id) return;
    S.take.name = doc.name;
    refreshHead();
  } catch (err) {
    if (turn !== S.nameTurn[id]) return;
    keepProblem(id, title, 'That name was not saved.', 'The name was not saved for ');
  }
}
/* The name box holds words for one take: S.nameFor. A change event can fire after the
   page has moved on, so nothing here asks which take is showing now. */
function saveName() {
  const box = $('take-name');
  if (!S.nameFor || box.value === S.nameSent) return;
  S.nameSent = box.value;
  rename(S.nameFor, box.value);
}
function leaveName() {
  saveName();
  S.nameFor = null;
  $('take-name').value = '';
}

/* ------------------------------------------------------------------- the shelf */

function shelfName(t) { return titleOf(t) + (t.kept ? ', kept' : ''); }
function buildShelf(doc) {
  const g = $('shelf-grid');
  g.textContent = '';
  clearStones();
  $('shelf-empty').hidden = doc.takes.length > 0;
  $('shelf-how').hidden = doc.takes.length === 0;
  doc.takes.forEach(function (t) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'shelf-take'; b.id = 'shelf-' + t.id;
    const c = document.createElement('canvas'); c.width = 320; c.height = 360;
    const name = document.createElement('span'); name.className = 't'; name.id = 'shelf-name-' + t.id;
    name.textContent = shelfName(t);
    const m = document.createElement('span'); m.className = 'm';
    m.textContent = [fmtDate(t.started_at)].concat(metaOf(t), [t.bytes ? fmtSize(t.bytes) : '']).filter(Boolean).join('  ');
    b.appendChild(c); b.appendChild(name); b.appendChild(m);
    b.addEventListener('click', function () { goTake(t.id); });
    g.appendChild(b);
    queueStone(c, t.portrait);
  });
  S.shelfTakes = doc.takes;
  S.shelfDone = true;
}
/* A rename can be answered after the shelf was asked for, so that answer knows better. */
function renamedOnShelf(id, name) {
  S.newNames[id] = name;
  const t = S.shelfTakes.find(function (x) { return x.id === id; });
  const shown = $('shelf-name-' + id);
  if (t) t.name = name;
  if (!t || !shown) return;
  shown.textContent = shelfName(t);
  S.shelfKey = JSON.stringify(S.shelfTakes) + currentTheme();
}
async function openShelf() {
  const turn = ++S.nav;
  busy(true);
  S.moving = true;
  saveName();
  S.newNames = {};
  let doc;
  try { doc = await api('/api/takes'); } catch (err) { if (turn === S.nav) { S.moving = false; problem(err); } return; }
  if (turn !== S.nav) return;
  doc.takes.forEach(function (t) { if (t.id in S.newNames) t.name = S.newNames[t.id]; });
  busy(false);
  leaveName();
  setScreen('shelf');
  arrived(null);
  /* Coming back to the same shelf repaints nothing and returns to where you were. */
  const key = JSON.stringify(doc.takes) + currentTheme();
  if (key !== S.shelfKey || !S.shelfDone) { buildShelf(doc); S.shelfKey = key; }
  const back = S.lastTake && $('shelf-' + S.lastTake);
  if (back) { window.scrollTo(0, S.shelfScroll); back.focus({ preventScroll: true }); }
  else $('shelf').focus({ preventScroll: true });
}
/* A new light changes the paint and nothing else: not the focus, not the scroll. */
function repaintShelf() {
  clearStones();
  S.shelfTakes.forEach(function (t) {
    const b = $('shelf-' + t.id), c = b && b.querySelector('canvas');
    if (c) queueStone(c, t.portrait, true);
  });
  S.shelfDone = true;
  S.shelfKey = JSON.stringify(S.shelfTakes) + currentTheme();
}

/* --------------------------------------------------------------- light and grain */

function paintGrain() {
  const c = document.createElement('canvas'); c.width = c.height = 220;
  const x = c.getContext('2d');
  if (!x) return;
  const im = x.createImageData(220, 220), ink = tok('--grain-ink'), r = rng(44), st = parseFloat(cssVar('--grain-strength')) || 1;
  for (let i = 0; i < im.data.length; i += 4) {
    im.data[i] = ink[0]; im.data[i + 1] = ink[1]; im.data[i + 2] = ink[2];
    im.data[i + 3] = (r() * 0.065 * st + (r() < 0.012 ? 0.04 * st : 0)) * 255;
  }
  x.putImageData(im, 0, 0);
  const g = $('grain'), w = Math.max(1, window.innerWidth), h = Math.max(1, window.innerHeight);
  g.width = w; g.height = h;
  const gx = g.getContext('2d');
  if (!gx) return;
  gx.clearRect(0, 0, w, h);
  gx.fillStyle = gx.createPattern(c, 'repeat');
  gx.fillRect(0, 0, w, h);
}
function currentTheme() { return document.documentElement.getAttribute('data-theme') || 'day'; }
function reflectTheme() {
  const cur = currentTheme();
  ['day', 'night'].forEach(function (n) { $('theme-' + n).setAttribute('aria-pressed', n === cur ? 'true' : 'false'); });
  paintGrain();
  if (S.screen === 'record' && S.take) { drawSheet(); selectEvent(S.selected); paintPortrait(); }
  if (S.screen === 'shelf') repaintShelf();
}
function setTheme(name) {
  if (name === 'day') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', name);
  save('plume-theme', name);
  reflectTheme();
}

/* ---------------------------------------------------------------------- wiring */

async function init() {
  S.roastOn = load('plume-roast') !== 'off';
  $('wordmark').addEventListener('click', goShelf);
  $('nav-takes').addEventListener('click', goShelf);
  window.addEventListener('popstate', showRoute);
  $('events').addEventListener('keydown', function (ev) {
    const list = S.take ? S.take.events : [];
    const at = list.findIndex(function (e) { return e.id === S.selected; });
    const to = nextIndex(ev.key, at, list.length);
    if (to < 0) return;
    ev.preventDefault();
    selectEvent(list[to].id);
    if ($(list[to].id)) $(list[to].id).focus();
  });
  ['day', 'night'].forEach(function (n) { $('theme-' + n).addEventListener('click', function () { setTheme(n); }); });
  $('keep').addEventListener('click', function () { setKept(!S.take.kept); });
  $('take-name').addEventListener('change', saveName);
  $('roast-toggle').addEventListener('click', function () {
    S.roastOn = !S.roastOn;
    save('plume-roast', S.roastOn ? 'on' : 'off');
    drawRoast();
  });
  let rt = 0;
  window.addEventListener('resize', function () {
    clearTimeout(rt);
    rt = setTimeout(function () {
      paintGrain();
      if (S.screen === 'record' && S.take) { drawSheet(); selectEvent(S.selected); }
    }, 120);
  });
  /* Labels are measured in the real font, once it is in. */
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () {
    if (S.screen === 'record' && S.take) { drawSheet(); selectEvent(S.selected); }
  });
  S.csrf = takePageKey();
  reflectTheme();
  let session;
  try { session = await api('/api/session'); } catch (err) { problem(err); return; }
  const pinned = pinnedRoute(S.cameByLink, session.open_take, location.hash);
  if (pinned) history.replaceState(null, '', pinned);
  showRoute();
}
window.plumeStonesIdle = stonesIdle;
init();
