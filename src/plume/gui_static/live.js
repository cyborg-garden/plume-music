/* What the page says about a take in progress. Pure, so it can be tested without a browser.

   The words stay honest about what plume knows: a detected key or style is a guess, and a
   field the bridge does not trust as being the player is not shown at all. */

const RUNNING = ['starting', 'waiting', 'playing', 'stopping'];
const STAGES = { drums: 'Drums', band: 'Bass and chords', full: 'Full band' };

export function isRunning(doc) { return !!doc && RUNNING.includes(doc.state); }

/* Where a take that is over sends a page that was following it. */
export function landing(doc) {
  if (doc.state === 'done') return { take: doc.take };
  return { home: true, message: doc.state === 'elsewhere' ? '' : doc.message || '', settings: !!doc.settings };
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
  return guessLine(key || 'Listening for the key', style, keyGuess, styleGuess);
}

function guessLine(key, style, keyGuess, styleGuess) {
  const line = [key, style].filter(Boolean).join(', ');
  if (keyGuess && styleGuess) return line + ' (a guess)';
  if (keyGuess) return line + ' (the key is a guess)';
  if (styleGuess) return line + ' (the style is a guess)';
  return line;
}

/* The same words for a take that is over, so the screens she keeps say it too. */
export function settledLine(take) {
  return guessLine(take.key, take.style, !!(take.key && take.key_guess), !!(take.style && take.style_guess));
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

/* The small sign. -45 dB is where plume itself calls a signal silence. */
export function hearingWord(db) { return db >= -45 ? 'Hearing you' : 'Listening'; }

/* What the sign can honestly show. A room mic with the band in cannot pick her out, but the mic
   is still live, so it says that much and nothing about her. null means no sign. */
export function signView(you, row, doc) {
  if (you.level !== null) return { word: hearingWord(you.level), level: you.level };
  if (!row || row.input !== 'room' || typeof doc.level_db !== 'number') return null;
  return { word: doc.level_db >= -45 ? 'Hearing the room' : 'Listening', level: doc.level_db };
}

/* A direct input that hears the band is the check on a wrong guess about what a device is, so it
   is said where it will be read. */
export function bandInInput(row) {
  return !!row && row.input === 'direct' && !(row.you_trusted || []).includes('you_db');
}
export const BAND_IN_INPUT_LINE = 'Plume hears the band on this input. Press Stop and try another device in Settings.';

/* The device's own number, called what it is. */
export function delayLine(ms) {
  if (typeof ms !== 'number') return 'Use headphones. Plume shows the delay your device reports once a take starts.';
  return 'Use headphones. Hear me adds about ' + Math.round(ms) + ' ms, by your device\'s own count, and it could be more.';
}

/* Said where it can be read while it is happening: on home, and during the take. */
export const DIRECT_LINE = 'A Direct or monitor button on your interface has no delay. Using both doubles the sound.';

/* Another output: what Settings says about it, and what the take says while it runs. */
export const WIRELESS_LINE = 'Wireless speakers arrive late, which makes playing along hard.';
export const SEND_LINE = 'It arrives slightly late. Only for an instrument plugged in: a mic on this channel would howl.';
export const ROOM_OUTPUT_LINE = 'Hear me stays off here: a room mic through speakers would howl.';
export function sendDelayLine(ms) {
  return 'Your instrument reaches that output about ' + Math.round(ms) + ' ms late, by your device\'s own count, and it could be more.';
}

/* A clap can't reach a pickup, and only a room mic hears the room. */
export function countWords(mode) {
  return mode === 'direct' ? 'Play 8 even beats. Muted strings work well.' : 'Knock or clap 8 even beats.';
}

/* Home's one small line: what Start will use. output: the name of another output, if any. */
export function routeLine(device, channel, key, style, output) {
  const bits = device
    ? [device.name, device.mode === 'direct' ? 'channel ' + channel : 'room mic']
    : ['System default', 'room mic'];
  if (output) bits.push('out to ' + output);
  if (key) bits.push(key);
  if (style) bits.push(style);
  return bits.join(', ');
}

/* ------------------------------------------------------------------ how a take ends */

const END_LINE = 'Stop playing and the band will finish on its own.';
export const FADING_LINE = 'The band is finishing. Play and it comes back.';
export const ROOM_END_LINE = 'A room mic may not hear you stop, so press Stop when you are done.';
export const NO_ENDING = { played: false, quietFrom: null, roomFrom: null };
/* Long enough to be read during a take, and then it gets out of the way. Details keeps it. */
const ROOM_END_BARS = 8;

export function isFading(row) { return !!row && row.fade_db < 0; }

/* What the engine has said about her playing, as a page polling it goes along. Only where the
   engine trusts that word: over a room mic with the band in, "playing" mostly means the band. */
export function watchEnding(seen, row) {
  /* The moment the engine stops trusting a room mic is when Stop becomes how it ends. */
  const lost = !!row && row.input === 'room' && row.playing_trusted === false;
  const roomFrom = seen.roomFrom === null && lost ? row.bar : seen.roomFrom;
  if (!row || !row.playing_trusted) return { played: seen.played, quietFrom: null, roomFrom };
  if (row.playing) return { played: true, quietFrom: null, roomFrom };
  return { played: seen.played, quietFrom: seen.quietFrom === null ? row.bar : seen.quietFrom, roomFrom };
}

/* Said once she has played and then gone quiet for more than a bar, which a rest can be too,
   so it stays calm. Drums waiting for her first note are not the end of anything. */
export function endLine(seen, row) {
  if (row && seen.roomFrom !== null && !isFading(row)) return row.bar - seen.roomFrom < ROOM_END_BARS ? ROOM_END_LINE : '';
  if (!row || !seen.played || seen.quietFrom === null || isFading(row)) return '';
  return row.bar - seen.quietFrom >= 1 ? END_LINE : '';
}

/* ---------------------------------------------------------------------- the first press */

const TRY_OR_START = [{ answer: 'again', label: 'Play again' }, { answer: 'start', label: 'Start anyway' }];
const LOST_PLACE = 'Plume lost its place. You can still start.';
const ASKING = 'This computer may be asking to use the microphone.';

/* The line while the bridge is being asked, from what it last said, so nothing flickers. */
export function askingLine(known) {
  if (known && known.done) return 'Opening the device.';
  if (known && known.step === 'listen') return known.line;
  return 'One moment.';
}

/* Somewhere to go when a step could not be done. A failed step never keeps her here. */
export function setupTrouble(words) {
  return { step: 'listen', line: words || LOST_PLACE, note: null, choices: TRY_OR_START, others: [] };
}

function heardNote(doc) {
  if (!doc.heard || doc.heard.result !== 'ok' || !doc.device_name) return null;
  return 'Heard you on ' + doc.device_name + (doc.mode === 'direct' ? ', channel ' + doc.channel : '') + '.';
}

/* Home showed one place for the band, and setup has just heard her on an input that would send it
   somewhere else. Home may have named an output, or named none while the new input has its own
   saved. shown: the input and output home was showing at the press of Start. Null when they
   agree, and unless she was heard: a device that heard nothing has not moved anything.
   output is what a Yes leaves saved on the new input, and otherwise what a No leaves. */
export function movedOutput(shown, doc, outputs) {
  if (!shown || !doc.device || doc.device === shown.device) return null;
  if (!doc.heard || !['ok', 'hot', 'low'].includes(doc.heard.result)) return null;
  const direct = doc.mode === 'direct', saved = doc.output || null;
  const output = shown.output || saved;
  if (!output) return null;
  if (shown.output
    ? saved === shown.output || (direct && !saved && doc.device === shown.output)
    : shown.direct && saved === shown.device) return null;
  const otherwise = shown.output ? saved : null;
  const nameOf = function (id) {
    const found = (outputs || []).find(function (o) { return o.id === id; });
    return found ? found.name : '';
  };
  const instead = otherwise ? nameOf(otherwise) || 'the output saved for it' : direct ? doc.device_name : 'the system output';
  return {
    step: 'moved', done: false, note: null, others: [], output: output, otherwise: otherwise,
    line: 'Plume heard you on ' + doc.device_name + '. Send the band to ' + (nameOf(output) || 'the output you chose')
      + '? If not, it comes out of ' + instead + '.',
    choices: [{ answer: 'yes', label: 'Yes' }, { answer: 'no', label: 'No' }]
  };
}

/* The big line is the first sentence, and the rest sits under it in smaller words. */
function splitLine(line) {
  const found = /^(.*?[.?!])\s+(\S.*)$/.exec(line);
  return found ? { line: found[1], more: found[2] } : { line: line, more: '' };
}

/* One line and what can be pressed. picked: a device she chose by name, if any.
   slow: listening has gone on past the point a device should have opened. */
export function setupView(doc, listening, picked, slow) {
  if (listening) {
    return { line: doc && doc.step === 'listen' && !(doc.choices || []).length ? doc.line : 'Play something.',
      more: '', note: slow ? ASKING : picked ? 'Listening on ' + picked + '.' : null,
      choices: [], others: [], listening: true };
  }
  /* The bridge wants step 1 again and nobody has clicked for it yet. */
  const choices = doc.choices.length || doc.step !== 'listen' ? doc.choices : TRY_OR_START;
  return Object.assign(splitLine(doc.line), { note: doc.note || heardNote(doc), choices: choices, others: doc.others || [], listening: false });
}
