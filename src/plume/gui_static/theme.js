/* Runs before the first paint, so a saved Night does not flash Day.
   Night is a deliberate choice: the system's dark setting never picks it. */
(function () {
  'use strict';
  var saved = null;
  try { saved = localStorage.getItem('plume-theme'); } catch (e) { /* private window */ }
  if (saved === 'night') document.documentElement.setAttribute('data-theme', 'night');
})();
