/* The guide's small Markdown: headings, paragraphs, lists, code, inline code, links, bold and
   italic. Nothing else is understood, and nothing is ever HTML: every word reaches the page
   through textContent, so a tag in a guide file shows as the characters it is.

   parseGuide is pure, so the same text always gives the same tree, and it can be tested
   without a browser. renderGuide only builds elements from that tree. */

/* Links may only go to the page's own screens. Anything else shows as its words. */
const SAFE_LINK = /^#\/(?:guide\/[a-z0-9-]{1,40}|settings|takes)?$/;

export function safeHref(href) {
  return SAFE_LINK.test(href) ? href : null;
}

/* One line of words into text, code, bold, italic and links. An unmatched mark stays a mark. */
export function parseInline(text) {
  const out = [];
  let plain = '';
  const flush = function () { if (plain) { out.push({ t: 'text', s: plain }); plain = ''; } };
  let i = 0;
  while (i < text.length) {
    const c = text[i];
    if (c === '`') {
      const end = text.indexOf('`', i + 1);
      if (end > i + 1) { flush(); out.push({ t: 'code', s: text.slice(i + 1, end) }); i = end + 1; continue; }
    } else if (c === '*' && text[i + 1] === '*') {
      const end = text.indexOf('**', i + 2);
      if (end > i + 2) { flush(); out.push({ t: 'b', c: parseInline(text.slice(i + 2, end)) }); i = end + 2; continue; }
    } else if (c === '*') {
      const end = text.indexOf('*', i + 1);
      if (end > i + 1 && text[i + 1] !== ' ') { flush(); out.push({ t: 'i', c: parseInline(text.slice(i + 1, end)) }); i = end + 1; continue; }
    } else if (c === '[') {
      const close = text.indexOf('](', i + 1), stop = close < 0 ? -1 : text.indexOf(')', close + 2);
      if (close > i + 1 && stop > close + 2) {
        flush();
        const label = parseInline(text.slice(i + 1, close)), href = safeHref(text.slice(close + 2, stop));
        out.push(href ? { t: 'a', href: href, c: label } : { t: 'span', c: label });
        i = stop + 1;
        continue;
      }
    }
    plain += c;
    i += 1;
  }
  flush();
  return out;
}

const HEADING = /^(#{1,3}) +(.*)$/, BULLET = /^[-*] +(.*)$/, NUMBER = /^\d{1,3}\. +(.*)$/, FENCE = /^```/;

/* The whole page into blocks. A list item's wrapped lines are indented; a blank line ends a block. */
export function parseGuide(text) {
  const lines = String(text).replace(/\r\n?/g, '\n').split('\n');
  const blocks = [];
  let para = null, list = null, i = 0;
  const end = function () {
    if (para) blocks.push({ t: 'p', inl: parseInline(para.join(' ')) });
    if (list) blocks.push({ t: list.kind, items: list.items.map(function (words) { return parseInline(words.join(' ')); }) });
    para = null; list = null;
  };
  while (i < lines.length) {
    const line = lines[i], trimmed = line.trim();
    let found;
    if (FENCE.test(trimmed)) {
      end();
      const code = [];
      i += 1;
      while (i < lines.length && !FENCE.test(lines[i].trim())) { code.push(lines[i]); i += 1; }
      blocks.push({ t: 'pre', s: code.join('\n') });
      i += 1;
      continue;
    }
    if (!trimmed) { end(); i += 1; continue; }
    if ((found = HEADING.exec(trimmed))) {
      end();
      blocks.push({ t: 'h', level: found[1].length, inl: parseInline(found[2].trim()) });
    } else if ((found = BULLET.exec(trimmed)) || (found = NUMBER.exec(trimmed))) {
      const kind = BULLET.test(trimmed) ? 'ul' : 'ol';
      if (para || (list && list.kind !== kind)) end();
      if (!list) list = { kind: kind, items: [] };
      list.items.push([found[1].trim()]);
    } else if (list && /^\s/.test(line)) {
      list.items[list.items.length - 1].push(trimmed);
    } else {
      if (list) end();
      if (!para) para = [];
      para.push(trimmed);
    }
    i += 1;
  }
  end();
  return blocks;
}

/* A page's title is its first heading, as plain words. */
export function pageTitle(blocks) {
  const h = blocks.find(function (b) { return b.t === 'h'; });
  return h ? plainText(h.inl) : '';
}
export function plainText(inl) {
  return inl.map(function (n) { return n.s !== undefined ? n.s : plainText(n.c); }).join('');
}

const TAGS = { text: null, code: 'code', b: 'strong', i: 'em', a: 'a', span: 'span' };

function inlineInto(el, inl, onLink) {
  inl.forEach(function (n) {
    if (n.t === 'text') { el.appendChild(document.createTextNode(n.s)); return; }
    const child = document.createElement(TAGS[n.t]);
    if (n.t === 'code') child.textContent = n.s;
    else inlineInto(child, n.c, onLink);
    if (n.t === 'a') {
      child.setAttribute('href', n.href);
      child.addEventListener('click', function (ev) { ev.preventDefault(); onLink(n.href); });
    }
    el.appendChild(child);
  });
}

/* Builds the page from its tree. The first heading is the page's own h1. */
export function renderGuide(el, blocks, onLink) {
  el.textContent = '';
  blocks.forEach(function (b) {
    let node;
    if (b.t === 'h') { node = document.createElement('h' + b.level); inlineInto(node, b.inl, onLink); }
    else if (b.t === 'p') { node = document.createElement('p'); inlineInto(node, b.inl, onLink); }
    else if (b.t === 'pre') {
      node = document.createElement('pre');
      const code = document.createElement('code');
      code.textContent = b.s;
      node.appendChild(code);
    } else {
      node = document.createElement(b.t);
      b.items.forEach(function (item) { const li = document.createElement('li'); inlineInto(li, item, onLink); node.appendChild(li); });
    }
    el.appendChild(node);
  });
}
