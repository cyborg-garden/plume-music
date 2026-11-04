/* What the page does about its link and its key. Pure, so it can be tested without a browser. */

/* Where a pinned take sends the page. Only a link just opened goes there: after that
   the address already says where the player is, and an empty one means the shelf. */
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
