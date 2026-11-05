/* What the page does about its link and its key. Pure, so it can be tested without a browser. */

/* Which screen an address means. Anything else is home. */
export function routeOf(hash) {
  const found = /^#\/take\/([0-9a-f]{16})$/.exec(hash);
  if (found) return { name: 'take', id: found[1] };
  if (hash === '#/takes') return { name: 'takes' };
  if (hash === '#/live') return { name: 'live' };
  return { name: 'home' };
}

/* Where a pinned take sends the page. Only a link just opened goes there: after that
   the address already says where the player is, and an empty one means home. */
export function pinnedRoute(cameByLink, pinned, address) {
  return cameByLink && pinned && !address ? '#/take/' + pinned : null;
}

/* The bridge refused this page. hasKey says whether this tab holds a key at all. */
export function lostWords(hasKey) {
  return hasKey ? {
    title: 'This page\'s link has stopped working.',
    text: 'plume gui restarted, or a newer link was opened. Run plume gui url in a terminal and open the new link.'
  } : {
    title: 'This tab has no key.',
    text: 'The key stays in the tab that opened the link, so a second tab cannot use it. Go back to that tab, or run plume gui url in a terminal and open the new link.'
  };
}
