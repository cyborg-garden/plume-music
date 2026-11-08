/* Home, Settings, the Guide, the live view, the Record and the shelf. Everything shown comes from the bridge,
   already filtered and already in sentences. Words go in with textContent only.

   Nothing here starts anything by itself: every POST that opens a device follows a click on
   Start or on a choice it led to, and nothing here gives those buttons focus, so a stray key
   cannot press them. */
import { clearStones, cssVar, queueStone, rng, stonesIdle, tok } from '/static/stone.js';
import { hitBoxes, lineSpans, nextIndex } from '/static/layout.js';
import { lostWords, pinnedRoute, routeOf } from '/static/session.js';
import { pageTitle, parseGuide, renderGuide } from '/static/guide.js';
import { BAND_IN_INPUT_LINE, DIRECT_LINE, FADING_LINE, NO_ENDING, ROOM_END_LINE, ROOM_OUTPUT_LINE, SEND_LINE, WIRELESS_LINE, askingLine, bandInInput, countWords, delayLine, endLine, endedNote, isFading, isRunning, keyLine, landing, meterShare, movedOutput, routeLine, sendDelayLine, settledLine, setupTrouble, setupView, signView, stageWords, watchEnding, youView } from '/static/live.js';

const $ = function (id) { return document.getElementById(id); };
const S = { csrf: '', take: null, selected: null, roastOn: true, screen: 'loading', nav: 0,
  shelfKey: null, shelfDone: false, shelfTakes: [], lastTake: null, shelfScroll: 0, cameByLink: false,
  nameFor: null, nameSent: '', keepTurn: {}, nameTurn: {}, moving: false, failed: null, newNames: {},
  live: { state: 'idle' }, liveTurn: 0, following: false, starting: false, stopSent: false, stopPending: false,
  trackButtons: {}, track: null, audioOn: false, seekTo: null,
  home: null, homeTurn: 0, homeNote: '', settingsNote: '', form: { device: null, channel: null },
  setup: null, setupTurn: 0, listening: false, listenSlow: false, answering: false, ending: NO_ENDING,
  guide: null };

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
/* The answer as well as the status: a refused Start says why, and what is running instead. */
async function send(op, body, more) {
  const r = await fetch('/api/' + op, {
    ...more, method: 'POST', credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', 'X-Plume-Token': S.csrf },
    body: JSON.stringify(body)
  });
  let doc = null;
  try { doc = await r.json(); } catch (e) { /* the status still says what happened */ }
  return { ok: r.ok, status: r.status, doc: doc };
}
async function post(op, body) {
  const answer = await send(op, body);
  if (!answer.ok) { const e = new Error('http ' + answer.status); e.status = answer.status; throw e; }
  return answer.doc;
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
  if (t.key || t.style) bits.push(settledLine(t));
  if (t.bars && t.bpm !== null) bits.push(t.bpm + ' bpm');
  bits.push(fmtTime(t.seconds));
  return bits;
}

function setScreen(name) {
  if (S.screen === 'setup' && name !== 'setup') leaveSetup();
  if (S.screen === 'settings' && name !== 'settings') S.settingsNote = '';
  S.screen = name;
  document.body.setAttribute('data-screen', name);
  ['home', 'settings', 'guide', 'setup', 'live', 'record', 'shelf', 'problem'].forEach(function (s) { $(s).hidden = s !== name; });
  if (name !== 'record') stopAudio();
  window.scrollTo(0, 0);
  reflectNav();
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
  drawReplay();
  const ended = endedNote(S.live, t.id);
  $('rec-note').hidden = !ended;
  $('rec-note').textContent = ended;
  $('keep-panel').hidden = !t.kept;
  if (t.kept) {
    const box = $('take-name');
    if (S.nameFor !== t.id || document.activeElement !== box) {
      box.value = t.name || ''; S.nameFor = t.id; S.nameSent = box.value;
    }
    $('keep-path').textContent = t.where;
  } else S.nameFor = null;
}
/* The next thing anyone wants after a take. Only on the one that just ended, so reading an old
   one is not an offer to start. */
function drawReplay() {
  $('rec-again').hidden = !(S.screen === 'record' && S.take && S.live.state === 'done' && S.live.take === S.take.id);
}
function stopAudio() {
  const p = $('player');
  p.pause();
  p.removeAttribute('src');
  p.load();
  S.audioOn = false; S.seekTo = null;
  $('listen-seek').value = '0';
  $('listen-time').textContent = '';
  $('listen-play').textContent = 'Play';
}
/* Sound starts only from this click. Nothing else in the page sets a source. */
function trackUrl(track) {
  /* An audio element cannot set a header, so the page key goes in the query. */
  return '/api/take/' + encodeURIComponent(S.take.id) + '/audio/' + track + '?k=' + S.csrf;
}
function fmtClock(s) {
  const n = Math.max(0, Math.floor(s || 0));
  return Math.floor(n / 60) + ':' + String(n % 60).padStart(2, '0');
}
function drawPlayer() {
  const p = $('player'), known = isFinite(p.duration) && p.duration > 0;
  $('listen-play').textContent = S.audioOn ? 'Pause' : 'Play';
  if (known) $('listen-seek').value = String(Math.round(1000 * p.currentTime / p.duration));
  $('listen-time').textContent = known ? fmtClock(p.currentTime) + ' / ' + fmtClock(p.duration) : '';
  Object.keys(S.trackButtons).forEach(function (track) {
    S.trackButtons[track].setAttribute('aria-pressed', track === S.track ? 'true' : 'false');
  });
}
/* Changing what you hear keeps your place, so the three can be compared at one moment. */
function playTrack(track) {
  const p = $('player'), at = p.getAttribute('src') ? p.currentTime : 0;
  S.track = track;
  S.seekTo = at || null;
  p.src = trackUrl(track);
  S.audioOn = true;
  p.play().catch(function () { S.audioOn = false; drawPlayer(); });
  drawPlayer();
}
function drawListen() {
  const t = S.take, wrap = $('listen-buttons');
  stopAudio();
  wrap.textContent = '';
  const tracks = (t.together ? ['together'] : []).concat(t.audio);
  $('listen').hidden = !tracks.length;
  const names = { together: 'Together', human: t.recorded_from === 'input' ? 'Your input' : 'Your mic', machine: 'The band' };
  S.track = tracks[0] || null;
  S.trackButtons = {};
  tracks.forEach(function (track) {
    const b = document.createElement('button');
    b.type = 'button'; b.id = 'listen-' + track; b.textContent = names[track];
    S.trackButtons[track] = b;
    b.addEventListener('click', function () { playTrack(track); });
    wrap.appendChild(b);
  });
  drawPlayer();
}
function paintPortrait() {
  /* Clearing the queue leaves any unfinished shelf stones blank, so remember that. */
  if (!stonesIdle()) S.shelfDone = false;
  clearStones(); queueStone($('portrait'), S.take.portrait);
}

/* Screens live in the address, so Back and reload work. Only the last request to start counts. */
function showRoute() {
  const route = routeOf(location.hash);
  if (route.name === 'take') openRecord(route.id);
  else if (route.name === 'takes') openShelf();
  else if (route.name === 'live') openLive();
  else if (route.name === 'settings') openSettings();
  else if (route.name === 'guide') openGuide(route.page);
  else openHome();
}
function goTake(id) {
  S.lastTake = id; S.shelfScroll = window.scrollY;
  history.pushState(null, '', '#/take/' + id);
  openRecord(id);
}
function goShelf() {
  S.homeNote = '';
  if (location.hash !== '#/takes') history.pushState(null, '', '#/takes');
  openShelf();
}
function goHome() {
  if (routeOf(location.hash).name !== 'home') history.pushState(null, '', '#/');
  openHome();
}
function goSettings() {
  S.settingsNote = '';
  if (location.hash !== '#/settings') history.pushState(null, '', '#/settings');
  openSettings();
}
function goGuide(page) {
  const where = page ? '#/guide/' + page : '#/guide';
  if (location.hash !== where) history.pushState(null, '', where);
  openGuide(page);
}
function goLive() {
  if (location.hash !== '#/live') history.pushState(null, '', '#/live');
  openLive();
}
/* A screen with nothing to fetch. Leaving a Record still saves what was typed there. */
function arriveAt(name) {
  S.nav += 1;
  busy(false);
  S.moving = true;
  saveName();
  leaveName();
  setScreen(name);
  arrived(null);
  $(name).focus({ preventScroll: true });
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

/* ---------------------------------------------------------------- home and settings */

function chosenDevice() {
  if (!S.home) return null;
  const id = S.form.device || S.home.device;
  return S.home.devices.find(function (d) { return d.id === id; }) || null;
}
/* What the bridge has saved, as of its last answer. The page keeps no tick of its own: one left
   here would outlast the gear it was true of. */
function hearOn() {
  if (!S.home) return false;
  const device = chosenDevice();
  return !!(device ? device.hear_me : S.home.hear_me && S.home.hear_me.default);
}
function channelNow() {
  if (S.form.channel !== null) return S.form.channel;
  const device = chosenDevice();
  return (device && device.channel) || 1;
}
function drawOptions(select, options, value) {
  select.textContent = '';
  options.forEach(function (o) {
    const el = document.createElement('option');
    el.value = o.value; el.textContent = o.label;
    select.appendChild(el);
  });
  select.value = value;
}
/* Where the chosen input's sound goes, as saved, or null for its own. */
function outputSaved() {
  const device = chosenDevice();
  return (device ? device : S.home || {}).output || null;
}
function outputName() {
  const saved = outputSaved();
  if (!saved) return '';
  const found = S.home.outputs.find(function (o) { return o.id === saved; });
  return found ? found.name : 'an output that has gone';
}
function outputProblem() {
  const device = chosenDevice();
  return (device ? device : S.home || {}).output_problem || '';
}
function drawOutput() {
  const doc = S.home, device = chosenDevice(), saved = outputSaved();
  const direct = !!device && device.mode === 'direct';
  /* Sent to her own outs, this is the same as not choosing, so it is not offered twice. */
  const others = doc.outputs.filter(function (o) { return !(direct && o.id === device.id); });
  const options = [{ value: '', label: direct ? 'Same as input' : 'System output' }]
    .concat(others.map(function (o) { return { value: o.id, label: o.name }; }));
  if (saved && !others.some(function (o) { return o.id === saved; })) options.push({ value: saved, label: 'Not plugged in' });
  drawOptions($('set-output'), options, saved || '');
  $('set-output').hidden = $('output-label').hidden = options.length < 2;
  const problem = outputProblem(), note = $('output-note');
  note.hidden = !problem && !saved;
  note.textContent = problem || WIRELESS_LINE;
  note.classList.toggle('warn', !!problem);
}
function drawSettings() {
  const doc = S.home;
  if (!doc) return;
  /* A device that went away falls back to the default rather than to nothing. */
  if (S.form.device && !doc.devices.some(function (d) { return d.id === S.form.device; })) { S.form.device = null; S.form.channel = null; }
  const device = chosenDevice(), most = device ? device.inputs : 1;
  if (S.form.channel > most) S.form.channel = null;
  const devices = doc.devices.map(function (d) {
    return { value: d.id, label: d.name + (d.mode === 'room' ? ' (room mic)' : '') };
  });
  if (!doc.device) devices.unshift({ value: '', label: 'System default (room mic)' });
  drawOptions($('set-device'), devices, device ? device.id : '');
  const channels = [];
  for (let n = 1; n <= most; n++) channels.push({ value: String(n), label: String(n) });
  drawOptions($('set-channel'), channels, String(channelNow()));
  $('set-channel').hidden = $('channel-label').hidden = most < 2;
  const styles = $('style-list');
  styles.textContent = '';
  doc.styles.forEach(function (name) { const o = document.createElement('option'); o.value = name; styles.appendChild(o); });
  $('set-folder').textContent = doc.folder;
  drawOutput();
  drawHear();
  drawSummary();
}
function drawHear() {
  /* Any input, once a person says headphones are on: ticking this says so, to the bridge, there
     and then. Plume cannot tell headphones from speakers, and the bridge refuses it without that yes. */
  const device = chosenDevice(), on = hearOn(), elsewhere = !!outputSaved();
  const direct = !!device && device.mode === 'direct', gone = !!outputProblem();
  $('hear-row').hidden = !S.home || elsewhere;
  $('hear-words').hidden = !S.home || !on || elsewhere;
  $('hear-me').checked = on;
  $('hear-direct').hidden = !direct;
  $('hear-direct').textContent = DIRECT_LINE;
  $('hear-delay').textContent = delayLine(device && device.reported_ms);
  drawSound();
  /* On another output a yes about headphones means nothing. An instrument takes its own opt-in. */
  const offer = elsewhere && direct && !gone, sends = offer && device.send_channel === channelNow();
  $('send-row').hidden = !offer;
  $('send-me').checked = sends;
  $('send-words').hidden = !offer;
  $('send-words').textContent = SEND_LINE;
  $('room-output').hidden = !(elsewhere && !direct);
  $('room-output').textContent = ROOM_OUTPUT_LINE;
}
/* Her sound on Hear me. Kept by this browser, and sent with Start only while Hear me is on. */
function soundNow() { return load('plume-sound') || 'off'; }
function drawSound() {
  const sounds = (S.home && S.home.sounds) || [];
  $('sound-more').hidden = !sounds.length;
  const now = sounds.some(function (o) { return o.id === soundNow(); }) ? soundNow() : 'off';
  drawOptions($('set-sound'), sounds.map(function (o) { return { value: o.id, label: o.label }; }), now);
}
async function sendHear(on) {
  const box = $('hear-me'), note = $('hear-note'), asked = chosenDevice(), body = { step: 'hear_me', answer: on ? 'on' : 'off' };
  if (asked) body.device = asked.id;
  box.disabled = true;
  note.hidden = true;
  /* A look at the bridge asked before this answer knows less than it does. */
  S.homeTurn += 1;
  let answer = null;
  try { answer = await send('setup/answer', body); } catch (err) { /* said below */ }
  box.disabled = false;
  if (answer && answer.status === 401) { problem(answer); return; }
  if (answer && answer.ok) {
    const device = asked && S.home.devices.find(function (d) { return d.id === asked.id; });
    if (device) device.hear_me = answer.doc.hear_me;
    else if (S.home.hear_me) S.home.hear_me.default = answer.doc.hear_me;
  } else {
    note.hidden = false;
    note.textContent = (answer && answer.doc && answer.doc.error) || 'That did not reach plume gui.';
  }
  drawHear();
}
/* The output and the opt-in are saved as they change, so a reload or a new page finds them. */
async function sendOutput(value) {
  const select = $('set-output'), asked = chosenDevice(), body = { step: 'output', answer: value || 'same' };
  if (asked) body.device = asked.id;
  select.disabled = true;
  $('hear-note').hidden = true;
  S.homeTurn += 1;
  let answer = null;
  try { answer = await send('setup/answer', body); } catch (err) { /* said below */ }
  select.disabled = false;
  if (settingsAnswerLeft(answer)) return;
  if (answer && answer.ok) {
    const holder = asked ? S.home.devices.find(function (d) { return d.id === asked.id; }) : S.home;
    if (holder) { holder.output = answer.doc.output || null; holder.output_problem = null; holder.send_channel = null; }
  } else showSettingsNote((answer && answer.doc && answer.doc.error) || 'That did not reach plume gui.');
  drawOutput();
  drawHear();
  drawSummary();
}
async function sendSend(on) {
  const box = $('send-me'), asked = chosenDevice(), channel = channelNow();
  const body = { step: 'send', answer: on ? 'on' : 'off', channel: channel };
  if (asked) body.device = asked.id;
  box.disabled = true;
  S.homeTurn += 1;
  let answer = null;
  try { answer = await send('setup/answer', body); } catch (err) { /* said below */ }
  box.disabled = false;
  if (settingsAnswerLeft(answer)) return;
  if (answer && answer.ok) {
    const device = asked && S.home.devices.find(function (d) { return d.id === asked.id; });
    if (device) device.send_channel = on ? channel : null;
  } else showSettingsNote((answer && answer.doc && answer.doc.error) || 'That did not reach plume gui.');
  drawHear();
}
/* A lost page key, or a take started since: then there is nothing here to redraw. */
function settingsAnswerLeft(answer) {
  if (answer && answer.status === 401) { problem(answer); return true; }
  if (answer && answer.status === 409 && answer.doc && answer.doc.live) { applyLive(answer.doc.live); return true; }
  return false;
}
function showSettingsNote(words) {
  S.settingsNote = words;
  drawSettingsNote();
}
function drawSettingsNote() {
  let note = S.settingsNote || (S.home && S.home.problem) || '';
  /* The Output choice already says this, right where it can be fixed. */
  if (note === outputProblem()) note = '';
  $('settings-note').hidden = !note;
  $('settings-note').textContent = note;
}
function drawSummary() {
  if (S.home) $('home-gear').textContent = routeLine(chosenDevice(), channelNow(), $('set-key').value.trim(), $('set-style').value.trim(), outputName());
}
/* A terminal's take is the only thing that takes Start's place. */
function drawHomeState() {
  const there = S.live.state === 'elsewhere';
  $('start').hidden = there;
  $('home-elsewhere').hidden = !there;
  $('home-gear').hidden = there;
  if (there) {
    $('elsewhere-text').textContent = S.live.message;
    $('elsewhere-stop').disabled = !!S.live.stopping;
    $('elsewhere-stop').textContent = S.live.stopping ? 'Stopping' : 'Stop it';
  }
  const note = there ? '' : S.homeNote || (S.home && S.home.problem) || '';
  $('home-note').hidden = !note;
  $('home-note').textContent = note;
}
async function loadHome() {
  const turn = ++S.homeTurn;
  let doc;
  /* Start does not need this: with nothing chosen the bridge makes the same choice itself. */
  try { doc = await api('/api/home'); } catch (err) { if (turn === S.homeTurn && err.status === 401) problem(err); return; }
  if (turn !== S.homeTurn) return;
  S.home = doc;
  drawSettings();
  drawHomeState();
  drawSettingsNote();
}
/* Plume's own stone. These are the numbers of the first take it was carved from. */
const HOME_STONE = { seed: 12034, height: 0.8137, lean: 0.2, cut: 0.2326, minor: false, nicks: 0 };
function paintHomeStone() {
  if (!stonesIdle()) S.shelfDone = false;
  clearStones(); queueStone($('home-stone'), HOME_STONE, true);
}
function openHome() {
  if (isRunning(S.live)) { history.replaceState(null, '', '#/live'); openLive(); return; }
  /* Setup is only ever reached by a click, never by reload, Back or Forward. */
  if (location.hash === '#/setup') history.replaceState(null, '', '#/');
  arriveAt('home');
  drawSettings();
  drawHomeState();
  loadHome();
  paintHomeStone();
}
/* Only ever by asking: nothing here opens a device, and a take that is running keeps what it was
   started with, so while one runs this is the take's screen instead. */
function openSettings() {
  if (isRunning(S.live)) { history.replaceState(null, '', '#/live'); openLive(); return; }
  arriveAt('settings');
  drawSettings();
  drawSettingsNote();
  /* A device plugged in since the page opened shows up here. */
  loadHome();
}
/* ---------------------------------------------------------------------- the guide */

/* The words come from the bridge once, and each page is drawn from them the same way every
   time. A link inside a page only ever goes to one of the page's own screens. */
async function openGuide(page) {
  arriveAt('guide');
  const turn = S.nav;
  if (!S.guide) {
    let doc;
    try { doc = await api('/api/guide'); } catch (err) {
      if (turn !== S.nav) return;
      if (err.status === 401) { problem(err); return; }
      $('guide-pages').textContent = '';
      $('guide-next').textContent = '';
      $('guide-body').textContent = 'The guide did not load. Is plume gui still running in your terminal?';
      return;
    }
    if (turn !== S.nav) return;
    S.guide = doc.pages.map(function (p) { const blocks = parseGuide(p.text); return { id: p.id, blocks: blocks, title: pageTitle(blocks) }; });
  }
  drawGuide(page);
}
function followGuideLink(href) {
  history.pushState(null, '', href);
  showRoute();
}
function guideLink(page, words) {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'textlink'; b.textContent = words;
  b.addEventListener('click', function () { goGuide(page.id); });
  return b;
}
function drawGuide(id) {
  const pages = S.guide, at = Math.max(0, pages.findIndex(function (p) { return p.id === id; })), page = pages[at];
  const nav = $('guide-pages');
  nav.textContent = '';
  pages.forEach(function (p) {
    const b = guideLink(p, p.title);
    if (p === page) b.setAttribute('aria-current', 'page');
    nav.appendChild(b);
  });
  renderGuide($('guide-body'), page.blocks, followGuideLink);
  const next = $('guide-next');
  next.textContent = '';
  if (pages[at + 1]) next.appendChild(guideLink(pages[at + 1], 'Next: ' + pages[at + 1].title));
}

/* Only what was decided here, so what setup learned is not overruled by a page that never
   changed it, and the device setup just heard, so the bridge does not choose again. It checks
   every field again, and a device that has gone is refused rather than swapped for another. */
function startBody(heard) {
  const body = {}, key = $('set-key').value.trim(), style = $('set-style').value.trim();
  if (key) body.key = key;
  if (style) body.style = style;
  const shown = S.home ? chosenDevice() : null, changed = S.form.channel !== null;
  const device = S.form.device ? shown && shown.id : heard || (changed && shown ? shown.id : null);
  if (device) body.device = device;
  /* The rows here are from before setup heard her, so only a channel she changed is sent. */
  if (changed && shown && shown.id === device) body.channel = S.form.channel;
  if (hearOn() && soundNow() !== 'off') body.sound = soundNow();
  return body;
}
/* inSettings: the message is about something chosen there, so it is said where it can be fixed. */
function backHome(message, inSettings) {
  S.following = false;
  S.homeNote = inSettings ? '' : message || '';
  S.settingsNote = inSettings ? message || '' : '';
  history.replaceState(null, '', inSettings ? '#/settings' : '#/');
  if (inSettings) openSettings();
  else openHome();
}
/* The page goes to the live view straight away. Listing devices can take seconds, and Stop
   has to be there during them. */
async function startTake(heard) {
  if (S.starting || isRunning(S.live)) return;
  S.starting = true;
  S.stopSent = false; S.stopPending = false;
  S.liveTurn += 1;
  S.homeNote = '';
  S.following = true;
  S.live = { state: 'starting' };
  S.ending = NO_ENDING;
  $('cue-note').textContent = '';
  /* Back from a take goes home, not to a step that is over. */
  if (location.hash === '#/setup') history.replaceState(null, '', '#/live');
  goLive();
  let answer = null;
  try { answer = await send('start', startBody(heard)); } catch (err) { /* said below */ }
  S.starting = false;
  S.liveTurn += 1;
  if (answer && answer.status === 401) { problem(answer); return; }
  if (answer && answer.ok) applyLive(answer.doc);
  else if (answer && answer.status === 409 && answer.doc && answer.doc.live) applyLive(answer.doc.live);
  else {
    /* Nothing started, as far as this page knows. The next look at the bridge corrects that. */
    S.live = { state: 'idle' };
    if (!answer) backHome('Plume could not reach plume gui. Is it still running in your terminal?');
    else if (answer.status === 400 && answer.doc && answer.doc.error) backHome(answer.doc.error, true);
    else backHome('The take did not start.');
    return;
  }
  if (S.stopPending && isRunning(S.live)) { S.stopPending = false; S.stopSent = false; stopTake(); }
  pollSoon();
}

/* ---------------------------------------------------------------- the first press */

/* A press of Start asks the bridge how far this device has got. With everything known that is
   the count, and the band. The click is the consent for the short listen, and for nothing else:
   a reload or Back never comes here. */
async function pressStart() {
  if (S.starting || isRunning(S.live) || S.screen === 'setup') return;
  const turn = ++S.setupTurn;
  S.homeNote = '';
  S.setup = null;
  S.listening = S.answering = false;
  const seen = chosenDevice();
  S.setupShown = S.home ? { device: seen ? seen.id : null, output: outputSaved(), direct: !!seen && seen.mode === 'direct' } : null;
  if (location.hash !== '#/setup') history.pushState(null, '', '#/setup');
  arriveAt('setup');
  drawSetup();
  let doc;
  try {
    doc = await api('/api/setup' + (S.form.device ? '?device=' + encodeURIComponent(S.form.device) : ''));
  } catch (err) {
    if (turn !== S.setupTurn) return;
    if (err.status === 401) problem(err);
    else startTake();
    return;
  }
  if (turn !== S.setupTurn) return;
  if (!doc.done && doc.step === 'listen') { S.setup = doc; setupListen(); return; }
  followSetup(doc);
}
function followSetup(doc) {
  S.listening = S.answering = false;
  if (askAboutOutput(doc)) return;
  S.setup = doc;
  if (doc.done) startTake(doc.device);
  else drawSetup();
}
/* Nothing plays while this is asked: she is told where the band would go first. */
function askAboutOutput(doc) {
  const moved = movedOutput(S.setupShown, doc, S.home && S.home.outputs);
  if (!moved) return false;
  S.setup = Object.assign(moved, { then: doc });
  drawSetup();
  return true;
}
/* What home said is settled once the answer is saved, for this input, and not before. The next
   press of Start reads what home knows, so it has to hear the answer too, or it would ask again
   after each take. An answer that did not save keeps the question, because Start anyway would
   start on the input she was heard on and leave out what she just said. */
async function answerMoved(answer) {
  const asked = S.setup, then = asked.then, turn = S.setupTurn;
  const wanted = answer === 'yes' ? asked.output : asked.otherwise;
  const settled = { device: then.device, output: wanted, direct: then.mode === 'direct' };
  if (wanted === (then.output || null)) { S.setupShown = settled; loadHome(); followSetup(then); return; }
  S.answering = true;
  Array.from($('setup-choices').children).forEach(setOff);
  let reply = null;
  try { reply = await send('setup/answer', { step: 'output', answer: wanted || 'same', device: then.device }); } catch (err) { /* said below */ }
  if (reply && reply.ok) loadHome();
  if (turn !== S.setupTurn) return;
  S.answering = false;
  if (reply && reply.ok) { S.setupShown = settled; followSetup(then.step === 'listen' ? then : reply.doc); return; }
  if (reply && reply.status === 400 && reply.doc && reply.doc.error) { backHome(reply.doc.error, true); return; }
  if (reply && (reply.status === 401 || (reply.status === 409 && reply.doc && reply.doc.live))) { settleSetup(reply); return; }
  S.setup = Object.assign({}, asked, { note: (reply && reply.doc && reply.doc.error) || 'That did not reach plume gui. Try again.' });
  drawSetup();
}
function pickedName() {
  const device = S.form.device && chosenDevice();
  return device ? device.name : '';
}
function drawSetup() {
  const known = S.home && !S.form.device ? S.home.setup : null;
  const view = S.setup || S.listening
    ? setupView(S.setup, S.listening, pickedName(), S.listenSlow)
    : { line: askingLine(known), more: '', note: null, choices: [], others: [], listening: false };
  $('setup-say').textContent = view.line;
  $('setup-more').hidden = !view.more;
  $('setup-more').textContent = view.more;
  $('setup-note').hidden = !view.note;
  $('setup-note').textContent = view.note || '';
  $('setup-listening').hidden = !view.listening;
  /* Time going by, not a level: the child only reports when it has finished. */
  if (view.listening) $('setup-listening-fill').style.animationDuration = ((S.setup && S.setup.seconds) || 4) + 's';
  const choices = $('setup-choices');
  choices.textContent = '';
  view.choices.forEach(function (choice, i) {
    const b = document.createElement('button');
    b.type = 'button'; b.id = 'setup-' + choice.answer; b.className = i === 0 ? 'solid' : 'quiet';
    b.textContent = choice.label;
    b.addEventListener('click', function (ev) { if (!(ev && ev.detail > 1)) chooseSetup(choice.answer); });
    choices.appendChild(b);
  });
  const others = $('setup-others');
  others.textContent = '';
  view.others.forEach(function (device) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'textlink';
    b.textContent = 'Try ' + device.name;
    b.addEventListener('click', function (ev) {
      if (ev && ev.detail > 1) return;
      S.form.device = device.id; S.form.channel = null;
      setupListen();
    });
    others.appendChild(b);
  });
}
function chooseSetup(answer) {
  if (S.answering || S.listening) return;
  if (S.setup.step === 'moved') answerMoved(answer);
  else if (answer === 'again') setupListen();
  else if (answer === 'start') startTake();
  else setupAnswer(S.setup.step, answer);
}
let slowTimer = 0;
async function setupListen() {
  const turn = S.setupTurn;
  S.listening = true;
  S.listenSlow = false;
  drawSetup();
  clearTimeout(slowTimer);
  slowTimer = setTimeout(function () {
    if (turn === S.setupTurn && S.listening) { S.listenSlow = true; drawSetup(); }
  }, (((S.setup && S.setup.seconds) || 4) + 3) * 1000);
  let answer = null;
  try { answer = await send('setup/listen', S.form.device ? { device: S.form.device } : {}); } catch (err) { /* said below */ }
  if (turn !== S.setupTurn) return;
  clearTimeout(slowTimer);
  S.listening = false;
  /* It has just heard her, which knows better than a channel set before she played. */
  if (answer && answer.ok && answer.doc && answer.doc.heard && answer.doc.heard.channel) S.form.channel = null;
  settleSetup(answer);
}
function setOff(b) { b.disabled = true; }
async function setupAnswer(step, value) {
  const turn = S.setupTurn, body = { step: step, answer: value };
  if (S.setup.device) body.device = S.setup.device;
  S.answering = true;
  Array.from($('setup-choices').children).forEach(setOff);
  let answer = null;
  try { answer = await send('setup/answer', body); } catch (err) { /* said below */ }
  if (turn !== S.setupTurn) return;
  S.answering = false;
  settleSetup(answer);
}
function settleSetup(answer) {
  if (answer && answer.status === 401) { problem(answer); return; }
  if (answer && answer.ok) { followSetup(answer.doc); return; }
  if (answer && answer.status === 409 && answer.doc && answer.doc.live) { applyLive(answer.doc.live); return; }
  /* A device that has gone is no reason to ask for it again. */
  if (answer && answer.status === 400) S.form.device = null;
  followSetup(setupTrouble(answer && answer.doc && answer.doc.error));
}
/* Every way off this screen comes through here. A listen in progress is called off, and an
   answer still on its way lands on nothing. */
function leaveSetup() {
  S.setupTurn += 1;
  clearTimeout(slowTimer);
  if (S.listening) send('setup/cancel', {}).catch(function () { /* it ends by itself in seconds */ });
  S.listening = S.answering = false;
}
async function resetSetup() {
  const note = $('set-again-note');
  note.textContent = '';
  let answer = null;
  try { answer = await send('setup/reset', {}); } catch (err) { /* said below */ }
  if (answer && answer.status === 401) { problem(answer); return; }
  if (!answer || !answer.ok) { note.textContent = 'That did not reach plume gui.'; return; }
  S.form.channel = null;
  if (S.home) {
    S.home.devices.forEach(function (d) { d.hear_me = false; d.output = null; d.output_problem = null; d.send_channel = null; });
    S.home.output = null; S.home.output_problem = null;
    if (S.home.hear_me) S.home.hear_me.default = false;
  }
  note.textContent = 'Start will ask again.';
  drawHear();
  loadHome();
}

/* ------------------------------------------------------------------- a live take */

function openLive() {
  if (!isRunning(S.live)) { land(S.live); return; }
  S.following = true;
  arriveAt('live');
  drawLive();
}
function land(doc) {
  const to = landing(doc);
  S.following = false;
  if (to.take) {
    history.replaceState(null, '', '#/take/' + to.take);
    openRecord(to.take);
  } else backHome(to.message, to.settings);
}
function liveWords(doc, row) {
  if (doc.state === 'starting') return 'Opening the device.';
  if (doc.state === 'waiting') return countWords(doc.mode);
  if (doc.state === 'stopping') return 'Stopping.';
  if (isFading(row)) return FADING_LINE;
  return keyLine(row, doc);
}
function drawLive() {
  const doc = S.live, row = doc.row || null, you = youView(row, doc);
  const stopping = doc.state === 'stopping' || S.stopSent;
  $('live-say').textContent = liveWords(doc, row);
  $('live-stop').disabled = stopping;
  $('live-stop').textContent = stopping ? 'Stopping' : 'Stop';
  const inInput = bandInInput(row);
  const hint = S.liveLost ? 'Lost touch with plume gui. If it has closed, plume stop still ends the take.' : doc.message || doc.hint || (inInput ? BAND_IN_INPUT_LINE : '');
  $('live-hint').hidden = !hint;
  $('live-hint').textContent = hint;
  const hasLevel = you.level !== null, sign = signView(you, row, doc);
  $('hearing').hidden = $('meter').hidden = !sign;
  $('hearing-word').textContent = sign ? sign.word : '';
  $('meter-fill').style.width = (sign ? Math.round(meterShare(sign.level) * 100) : 0) + '%';
  $('live-facts').hidden = !row;
  if (row) {
    $('live-bar').textContent = row.bar;
    $('live-tempo').textContent = Math.round(row.bpm) + ' bpm';
    $('live-stage').textContent = stageWords(row.stage);
    $('live-notes-box').hidden = !you.notes;
    $('live-notes').textContent = you.notes && you.notes.length ? you.notes.join(' ') : 'None';
    $('live-level-box').hidden = !hasLevel;
    $('live-level').textContent = hasLevel ? Math.round(you.level) + ' dB' : '';
  }
  $('live-why').hidden = !you.why || inInput;
  $('live-why').textContent = you.why;
  /* Only once a stream has said so, and only when this take is monitoring. */
  const ms = doc.hear_me && doc.stream ? doc.stream.reported_ms : null;
  $('live-delay').hidden = typeof ms !== 'number';
  $('live-delay').textContent = typeof ms !== 'number' ? '' : doc.output ? sendDelayLine(ms) : delayLine(ms);
  /* That button only reaches the interface's own outs, so it says nothing about other speakers. */
  $('live-direct').hidden = !doc.hear_me || doc.mode !== 'direct' || !!doc.output;
  $('live-direct').textContent = DIRECT_LINE;
  $('live-cues').hidden = doc.state !== 'playing';
  /* Told once she has gone quiet, where the engine can tell. A room mic can't, so it is told
     once the band is in, for a few bars. */
  S.ending = watchEnding(S.ending, row);
  const ends = stopping ? '' : endLine(S.ending, row);
  $('live-end').hidden = !ends;
  $('live-end').textContent = ends;
  $('live-how').hidden = !row || row.input !== 'room';
  $('live-how').textContent = ROOM_END_LINE;
  /* Empty Details are not offered. */
  $('live-details').hidden = !row && $('live-delay').hidden && $('live-direct').hidden;
}
/* A click sends one request. The bridge signals a take once however often it is asked, and
   a second interrupt would cut the band's ending short. */
async function stopTake(ev) {
  /* Stop opens where an answer was, and a double click there is not a Stop. */
  if (S.stopSent || (ev && ev.detail > 1)) return;
  S.stopSent = true;
  if (S.starting) S.stopPending = true;
  if (S.screen === 'live') drawLive();
  try {
    const doc = await post('stop', {});
    if (!S.starting) applyLive(doc);
  } catch (err) {
    S.stopSent = false;
    if (err.status === 401) { problem(err); return; }
    S.liveLost = true;
    if (S.screen === 'live') drawLive();
    else if (S.screen === 'home') { S.homeNote = 'That did not reach plume gui. plume stop in a terminal still ends the take.'; drawHomeState(); }
  }
}
async function sendCue(cue) {
  const note = $('cue-note');
  note.textContent = '';
  let answer = null;
  try { answer = await send('cue', { cue: cue }); } catch (err) { /* said below */ }
  if (answer && answer.status === 401) { problem(answer); return; }
  if (answer && answer.ok) note.textContent = 'Asked for ' + cue + '.';
  else note.textContent = answer && answer.doc && answer.doc.error ? answer.doc.error : 'The band did not answer that cue.';
}
/* Everything the bridge says about the take comes through here. */
function applyLive(doc) {
  S.live = doc;
  if (isRunning(doc)) S.following = true;
  else {
    S.stopSent = false;
    S.ending = NO_ENDING;
    /* Stopped before the band came in: there is no take to go back to. */
    if (doc.state === 'idle' && S.screen !== 'live') S.following = false;
  }
  reflectNav();
  if (isRunning(doc)) {
    if (S.screen === 'home' || S.screen === 'setup' || S.screen === 'settings') goLive();
    else if (S.screen === 'live') drawLive();
    return;
  }
  if (S.screen === 'live') { land(doc); return; }
  /* A take started from a terminal: Start gives way to it, so setup does too. */
  if (S.screen === 'setup' && doc.state === 'elsewhere') { backHome(''); return; }
  if (S.screen === 'home') drawHomeState();
}
/* A person reading the shelf during a take stays there, with a way back to it. */
function reflectNav() {
  const nav = $('nav-live'), running = isRunning(S.live);
  nav.hidden = S.screen === 'live' || !(running || S.following);
  nav.textContent = running ? 'Playing' : 'Last take';
  $('nav-settings').hidden = running;
  drawReplay();
}
/* Asking is also what tells the bridge a page is still here, so this runs on every screen
   and never overlaps itself. */
let pollTimer = 0;
function pollSoon() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(pollLive, isRunning(S.live) ? 1000 : 3000);
}
async function pollLive() {
  const turn = S.liveTurn;
  try {
    const doc = await api('/api/live');
    /* Asked before a Start, this can describe the take before. */
    if (turn === S.liveTurn && !S.starting) {
      S.liveLost = false;
      applyLive(doc);
    }
  } catch (err) {
    if (err.status === 401 && (S.screen === 'live' || S.screen === 'home' || S.screen === 'settings')) problem(err);
    else if (err.status !== 401) {
      S.liveLost = true;
      if (S.screen === 'live') drawLive();
    }
  }
  pollSoon();
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
  if (S.screen === 'home') paintHomeStone();
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
  $('wordmark').addEventListener('click', goHome);
  $('listen-play').addEventListener('click', function () {
  const p = $('player');
  if (!S.track) return;
  if (S.audioOn) { p.pause(); S.audioOn = false; drawPlayer(); return; }
  if (!p.getAttribute('src')) { playTrack(S.track); return; }
  S.audioOn = true;
  p.play().catch(function () { S.audioOn = false; drawPlayer(); });
  drawPlayer();
});
$('listen-seek').addEventListener('input', function () {
  const p = $('player'), to = Number($('listen-seek').value) / 1000;
  if (isFinite(p.duration) && p.duration > 0) p.currentTime = to * p.duration;
});
$('player').addEventListener('loadedmetadata', function () {
  const p = $('player');
  if (S.seekTo !== null) { p.currentTime = S.seekTo; S.seekTo = null; }
  drawPlayer();
});
$('player').addEventListener('timeupdate', drawPlayer);
$('player').addEventListener('ended', function () {
  const p = $('player');
  S.audioOn = false; p.currentTime = 0; drawPlayer();
});
$('nav-takes').addEventListener('click', goShelf);
  $('nav-settings').addEventListener('click', goSettings);
  $('settings-done').addEventListener('click', goHome);
  $('nav-guide').addEventListener('click', function () { goGuide(null); });
  $('settings-terminal').addEventListener('click', function () { goGuide('terminal'); });
  $('nav-live').addEventListener('click', goLive);
  $('start').addEventListener('click', pressStart);
  $('set-sound').addEventListener('change', function () { save('plume-sound', $('set-sound').value); });
  $('rec-again').addEventListener('click', function (ev) { if (!(ev && ev.detail > 1)) pressStart(); });
  $('setup-leave').addEventListener('click', goHome);
  $('set-again').addEventListener('click', resetSetup);
  $('live-stop').addEventListener('click', stopTake);
  $('elsewhere-stop').addEventListener('click', function () { stopTake(); });
  $('cue-busier').addEventListener('click', function () { sendCue('busier'); });
  $('cue-sparser').addEventListener('click', function () { sendCue('sparser'); });
  $('hear-me').addEventListener('change', function () { sendHear($('hear-me').checked); });
  $('send-me').addEventListener('change', function () { sendSend($('send-me').checked); });
  $('set-output').addEventListener('change', function () { S.settingsNote = ''; drawSettingsNote(); sendOutput($('set-output').value); });
  $('set-device').addEventListener('change', function () {
    S.form.device = $('set-device').value || null;
    S.form.channel = null;
    S.settingsNote = '';
    drawSettingsNote();
    drawSettings();
  });
  $('set-channel').addEventListener('change', function () {
    S.form.channel = parseInt($('set-channel').value, 10) || 1;
    S.settingsNote = '';
    drawSettingsNote();
    drawSummary();
    /* The opt-in was for one channel, so another shows unticked until she says so. */
    drawHear();
  });
  /* Typing a fix answers the complaint about what was there. */
  ['set-key', 'set-style'].forEach(function (id) {
    $(id).addEventListener('input', function () { S.homeNote = ''; S.settingsNote = ''; drawSummary(); drawHomeState(); drawSettingsNote(); });
  });
  window.addEventListener('popstate', showRoute);
  /* A reload or a closed tab leaves no screen behind to come through leaveSetup. keepalive lets
     this outlast the page. If it never arrives the child still ends by itself within seconds. */
  window.addEventListener('pagehide', function () {
    if (S.listening) send('setup/cancel', {}, { keepalive: true }).catch(function () { /* see above */ });
  });
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
  /* A reload finds the take it left, and never starts a second one. */
  try { S.live = await api('/api/live'); } catch (err) { if (err.status === 401) { problem(err); return; } }
  const pinned = pinnedRoute(S.cameByLink, session.open_take, location.hash);
  if (isRunning(S.live)) history.replaceState(null, '', '#/live');
  else if (pinned) history.replaceState(null, '', pinned);
  showRoute();
  pollSoon();
}
window.plumeStonesIdle = stonesIdle;
init();
