/* Just enough of a browser to run the real page script under Node, with a stand-in
   for the bridge. Nothing here can make sound: the audio element only counts calls.

   The focus rule matters most. As in a browser, moving focus off an edited text box
   fires its change event, and setting its value from code clears the edit. */

const RECT = { left: 0, top: 0, width: 1280, height: 360, right: 1280, bottom: 360 };

class Classes {
  constructor() { this.names = new Set(); }
  toggle(name, on) { if (on) this.names.add(name); else this.names.delete(name); }
  contains(name) { return this.names.has(name); }
}

class El {
  constructor(page, tag) {
    Object.assign(this, { page, tag, children: [], attrs: {}, style: {}, classList: new Classes(), listeners: {},
      hidden: false, offsetWidth: 100, _id: '', _text: '', _value: '', edited: false,
      plays: 0, pauses: 0, loads: 0, type: '', tabIndex: 0, open: false, width: 0, height: 0 });
  }
  get id() { return this._id; }
  set id(v) { this._id = v; this.page.byId.set(v, this); }
  get className() { return [...this.classList.names].join(' '); }
  set className(v) { this.classList.names = new Set(v.split(' ').filter(Boolean)); }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(''); }
  set textContent(v) {
    this.children.forEach((c) => c.forget());
    this.children = [];
    this._text = String(v);
  }
  forget() {
    if (this._id && this.page.byId.get(this._id) === this) this.page.byId.delete(this._id);
    this.children.forEach((c) => c.forget());
  }
  get value() { return this._value; }
  set value(v) { this._value = String(v); this.edited = false; }
  get src() { return this.attrs.src || ''; }
  set src(v) { this.attrs.src = String(v); }
  appendChild(c) { this.children.push(c); return c; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  fire(type, ev) { (this.listeners[type] || []).forEach((fn) => fn.call(this, ev || { preventDefault() {} })); }
  click() { this.fire('click'); }
  focus() {
    const was = this.page.activeElement;
    if (was === this) return;
    this.page.activeElement = this;
    if (was && was.edited) { was.edited = false; was.fire('change'); }
  }
  blur() { if (this.page.activeElement === this) this.page.body.focus(); }
  getBoundingClientRect() { return RECT; }
  scrollIntoView() {}
  querySelector(sel) { return this.walk().find((e) => e.tag === sel) || null; }
  walk() { return this.children.flatMap((c) => [c, ...c.walk()]); }
  getContext() { return null; }
  /* The audio element. It keeps count, and that is all it does. */
  play() { this.plays += 1; return Promise.resolve(); }
  pause() { this.pauses += 1; }
  load() { this.loads += 1; }
}

/* A store that remembers every value it was ever given, so a key briefly kept shows. */
function store() {
  const map = new Map(), written = [];
  return { written, map,
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => { map.set(k, String(v)); written.push([k, String(v)]); },
    removeItem: (k) => map.delete(k) };
}

export async function start(html, { takes, docs, address = '#k=' + 'ab'.repeat(32), pinned = null, stored = null }) {
  const page = { byId: new Map(), activeElement: null, calls: [], held: [], failNext: {}, saved: {} };
  page.body = new El(page, 'body');
  page.root = new El(page, 'html');
  page.activeElement = page.body;
  for (const found of html.matchAll(/<([a-z0-9]+)\b([^>]*)\bid="([^"]+)"([^>]*)>/g)) {
    const el = new El(page, found[1]);
    el.id = found[3];
    el.hidden = /\bhidden\b/.test(found[2] + found[4]);
  }
  const location = { hash: '', pathname: '/' };
  const setAddress = (url) => { const i = url.indexOf('#'); location.hash = i < 0 ? '' : url.slice(i); };
  setAddress(address);
  const back = [location.hash], stateLog = [];
  let at = 0;
  const windowListeners = {};
  const win = { innerWidth: 1280, innerHeight: 900, scrollY: 0, devicePixelRatio: 1, scrollTo() {},
    addEventListener(type, fn) { (windowListeners[type] = windowListeners[type] || []).push(fn); } };
  const history = {
    pushState(_s, _t, url) { setAddress(url); back.length = at + 1; back.push(location.hash); at += 1; stateLog.push(url); },
    replaceState(_s, _t, url) { setAddress(url); back[at] = location.hash; stateLog.push(url); },
    go(delta) { at += delta; location.hash = back[at]; (windowListeners.popstate || []).forEach((fn) => fn({})); }
  };
  page.local = store();
  page.session = store();
  if (stored) page.session.map.set('plume-key', stored);
  const reply = (status, doc) => ({ ok: status < 400, status, json: async () => doc });
  async function bridge(path, opts = {}) {
    const method = opts.method || 'GET', body = opts.body ? JSON.parse(opts.body) : null;
    const call = { method, path, body, key: opts.headers && opts.headers['X-Plume-Token'] };
    page.calls.push(call);
    const op = path.replace('/api/', '');
    const held = page.held.find((h) => h.op === op && !h.taken);
    if (held) { held.taken = true; await held.until; }
    if (page.failNext[op]) { page.failNext[op] -= 1; return reply(500, { error: 'no' }); }
    if (path === '/api/session') return reply(200, { csrf: 'ab'.repeat(32), open_take: pinned });
    if (path === '/api/takes') return reply(200, { takes: takes.map((t) => Object.assign({}, t, page.saved[t.id])) });
    if (path.startsWith('/api/take/')) {
      const id = path.split('/')[3], doc = docs[id];
      return doc ? reply(200, Object.assign({}, doc, page.saved[id])) : reply(404, { error: 'no' });
    }
    const was = Object.assign({}, docs[body.take], page.saved[body.take]);
    const doc = { kept: 'kept' in body ? body.kept : was.kept, name: 'name' in body ? body.name : was.name };
    page.saved[body.take] = doc;
    return reply(200, doc);
  }
  Object.assign(globalThis, { window: win, location, history, fetch: bridge,
    getComputedStyle: () => ({ paddingLeft: '40px', paddingRight: '40px', getPropertyValue: () => '#808080' }),
    document: {
      body: page.body, documentElement: page.root,
      get activeElement() { return page.activeElement; },
      getElementById: (id) => page.byId.get(id) || null,
      createElement: (tag) => new El(page, tag),
      createTextNode: (text) => { const node = new El(page, '#text'); node.textContent = text; return node; },
      querySelectorAll: () => ['events', 'ev-list'].flatMap((id) => page.byId.get(id).walk()).filter((e) => e.tag === 'button')
    } });
  for (const [name, value] of [['localStorage', page.local], ['sessionStorage', page.session]]) {
    Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
  }
  page.$ = (id) => page.byId.get(id);
  page.location = location;
  page.stateLog = stateLog;
  page.history = history;
  page.settle = async () => { for (let i = 0; i < 8; i++) await new Promise((r) => setTimeout(r, 1)); };
  /* The next POST or GET for this op waits until the returned function is called. */
  page.hold = (op) => { let open; const until = new Promise((r) => { open = r; }); page.held.push({ op, until }); return open; };
  /* What the page shows of a take, opened the way a click on the shelf opens it. */
  page.open = async (id) => { page.$('shelf-' + id).click(); await page.settle(); };
  page.shelf = async () => { page.$('nav-takes').click(); await page.settle(); };
  /* Typing leaves the box focused and edited, with no change event yet. */
  page.type = (text) => { const box = page.$('take-name'); box.focus(); box._value = text; box.edited = true; };
  page.posts = (op) => page.calls.filter((c) => c.path === '/api/' + op).map((c) => c.body);
  await import('./app.mjs');
  await page.settle();
  return page;
}
