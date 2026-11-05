/* What the page says about a take in progress. Pure, so it can be tested without a browser.

   The words stay honest about what plume knows: a detected key or style is a guess, and a
   field the bridge does not trust as being the player is not shown at all. */

const RUNNING = ['starting', 'waiting', 'playing', 'stopping'];
const STAGES = { drums: 'Drums', band: 'Bass and chords', full: 'Full band' };

export function isRunning(doc) { return !!doc && RUNNING.includes(doc.state); }

/* Where a take that is over sends a page that was following it. */
export function landing(doc) {
  if (doc.state === 'done') return { take: doc.take };
  return { home: true, message: doc.state === 'failed' ? doc.message : '' };
}

/* The note a Record shows for the take that just ended, and only for that one. */
export function endedNote(doc, id) {
  return doc && doc.state === 'done' && doc.take === id && doc.note ? doc.note : '';
}

export function stageWords(stage) { return STAGES[stage] || ''; }

/* "F major, midwest (a guess)". A key or style the player typed is never a guess. */
export function keyLine(row, doc) {
  const key = (row && row.key) || doc.key, style = (row && row.style) || doc.style;
  const keyGuess = !!(key && row && row.key_guess && !doc.key);
  const styleGuess = !!(style && row && row.style_guess && !doc.style);
  const line = [key || 'Listening for the key', style].filter(Boolean).join(', ');
  if (keyGuess && styleGuess) return line + ' (a guess)';
  if (keyGuess) return line + ' (the key is a guess)';
  if (styleGuess) return line + ' (the style is a guess)';
  return line;
}

/* What can be said about the player right now. null means plume can't say, so don't show it.
   Before any row the only sound in the recording is the player, so its level is theirs. */
export function youView(row, doc) {
  if (!row) return { notes: null, level: doc.level_db === undefined ? null : doc.level_db, why: '' };
  const trusted = row.you_trusted || [];
  const notes = trusted.includes('you_notes') && row.you_notes ? row.you_notes : null;
  let level = null;
  if (trusted.includes('you_db')) level = typeof doc.level_db === 'number' ? doc.level_db : row.you_db;
  if (notes && level !== null && level !== undefined) return { notes, level, why: '' };
  return { notes, level: level === undefined ? null : level,
    why: row.input === 'room' ? 'Plume can\'t tell over the band on a room mic.'
      : 'Plume hears the band on this input, so it can\'t tell.' };
}

/* How far along the meter a level sits, from -60 dB to 0 dB. */
export function meterShare(db) {
  if (typeof db !== 'number') return 0;
  return Math.max(0, Math.min(1, (db + 60) / 60));
}

/* The device's own number, called what it is. */
export function delayLine(ms) {
  if (typeof ms !== 'number') return 'Use headphones. Plume shows the delay your device reports once a take starts.';
  return 'Use headphones. Plume adds about ' + Math.round(ms) + ' ms, as your device reports it. This is not a measured round trip.';
}

/* The menu's one closed line: what Start will use, without opening it. */
export function routeLine(device, channel, key, style) {
  const bits = device
    ? [device.name, device.mode === 'direct' ? 'channel ' + channel : 'room mic']
    : ['System default', 'room mic'];
  if (key) bits.push(key);
  if (style) bits.push(style);
  return bits.join(', ');
}
