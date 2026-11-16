/* A video for a take: what the page says and shows at each point. Pure, so it can be tested
   without a browser. The bridge says where the work has got to; every word is chosen here. */

export const ADD_VIDEO = 'Add a video';
export const KEEP_SOUND = 'Keep the video\'s sound';
export const SHOW_VIDEO = 'Show video';
export const DROP_HINT = 'Drop your video anywhere here.';

const WORKING = {
  receiving: 'Bringing your video in.',
  listening: 'Finding where you start playing.',
  making: 'Laying the band in.'
};
const FAILED = {
  theirs: 'This take\'s folder already has a video.mp4 that plume didn\'t make. Move or rename it, then add your video again.',
  in_use: 'video.mp4 is open in another program. Close it there, then add your video again.',
  not_video: 'Plume couldn\'t read that as a video. Try another file?',
  too_big: 'That video is too big for plume to bring in.',
  too_long: 'That video is too long for plume to bring in.',
  no_room: 'There isn\'t enough room on this disk for the video.',
  busy: 'Plume is still working on a video for this take.',
  no_tool: 'Adding a video needs ffmpeg, and plume couldn\'t find it. Install ffmpeg, then open plume again.',
  lost: 'This page\'s link has stopped working. Run plume gui url in a terminal and open the new link.'
};
const FAILED_PLAINLY = 'That didn\'t work, and your take is untouched. Try again?';
const UNSURE = 'Plume isn\'t sure where you start playing. Listen, and slide until you and the band sit together.';
const SILENT = 'This video has no sound to line up with. Slide to the moment you start playing, then save it and watch.';
const DONE = 'Your video is ready, with the band in it. It\'s saved as video.mp4 in this take\'s folder.';

export function isWorking(doc) { return !!doc && doc.state in WORKING; }

/* A place in the video, for under the slider: "0:51.24", or "2.00 s before the video starts". */
export function placeWords(seconds) {
  if (seconds < 0) return (-seconds).toFixed(2) + ' s before the video starts';
  const whole = Math.floor(seconds / 60), rest = seconds - whole * 60;
  return whole + ':' + (rest < 10 ? '0' : '') + rest.toFixed(2);
}

/* What to show for a take's video. doc is the bridge's answer, or null before one has come.
   dropping is true while a file is held over the page; again, just after a second video was
   offered while the first is still being worked on. */
export function videoView(doc, dropping, again) {
  const view = { shown: false, say: '', warn: false, add: false, bar: null, nudge: false, listen: false,
    show: false, stop: false };
  if (!doc || doc.can !== true) return view;
  view.shown = true;
  const state = doc.state || 'idle';
  if (state in WORKING) {
    /* again: she offered another video while this one is under way. */
    view.say = again ? FAILED.busy : WORKING[state];
    /* A share when the bridge knows one; otherwise a bar that only says something is happening. */
    view.bar = typeof doc.progress === 'number' ? Math.max(0, Math.min(1, doc.progress)) : 'busy';
    view.stop = true;
    return view;
  }
  if (state === 'unsure') {
    view.say = doc.why === 'silent' ? SILENT : UNSURE;
    view.nudge = true;
    view.listen = doc.listen === true;
    view.stop = true;
    return view;
  }
  view.add = doc.ready === true;
  view.show = doc.have === true;
  if (doc.ready !== true) { view.say = FAILED.no_tool; return view; }
  if (state === 'failed') { view.say = FAILED[doc.why] || FAILED_PLAINLY; view.warn = true; }
  else if (state === 'done') view.say = DONE;
  else if (dropping) view.say = DROP_HINT;
  return view;
}
