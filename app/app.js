// Triptych player with character/word highlighting synced to audio, paragraph practice
// with live and time-warped pitch-contour overlay.

const $ = (s) => document.querySelector(s);
const audio = $('#audio');

const state = {
  seg: null,            // segment JSON
  aligned: [],          // chars with start times, sorted by start
  charEls: [],          // index -> <span class=ch>
  pyEls: [],            // index -> <span class=ps>
  wordOf: [],           // char index -> word {s,e}
  activeIdx: -1,
  selectedPara: -1,
  loopRange: null,      // [t0, t1]
  refContour: null,     // {times, semitones}
  liveST: [],           // live semitone samples [{t, st}]
  liveF0: [],
  recorder: null,
  recChunks: [],
  myAudioUrl: null,
  compare: null,
};

// ---------- segment list ----------
async function loadList() {
  const segs = await (await fetch('/api/segments')).json();
  const ul = $('#segment-list');
  ul.innerHTML = '';
  let lastDate = null;
  for (const s of segs) {
    if (s.date !== lastDate) {
      const g = document.createElement('li');
      g.className = 'date-group';
      g.textContent = s.date;
      ul.appendChild(g);
      lastDate = s.date;
    }
    const li = document.createElement('li');
    li.dataset.guid = s.guid;
    const dur = s.duration_s ? fmt(s.duration_s) : '';
    const usable = s.status === 'ready' || s.status === 'enriched';
    li.innerHTML = `<div class="date"><span>${dur}</span>${s.qc_flags.length ? '<span class="qc">⚠ ' + s.qc_flags.length + '</span>' : ''}<span>${s.status}</span></div>
                    <div class="headline">${esc(s.headline)}</div>
                    ${!usable && s.error ? `<div class="why">${esc(s.error)}</div>` : ''}`;
    if (!usable) { li.classList.add('unusable'); li.title = s.error || s.status; }
    else li.onclick = () => openSegment(s.guid);
    ul.appendChild(li);
  }
}

// ---------- open a segment ----------
async function openSegment(guid) {
  const r = await fetch(`/api/segments/${guid}`);
  if (!r.ok) return;
  state.seg = await r.json();
  document.querySelectorAll('#segment-list li').forEach((li) => li.classList.toggle('active', li.dataset.guid === guid));
  $('#empty').classList.add('hidden');
  $('#player').classList.remove('hidden');
  renderHeader();
  renderRows();
  audio.src = `/api/segments/${guid}/audio`;
  audio.playbackRate = parseFloat($('#rate').value);
  selectPara(-1);
  state.compare = null;
  drawContour();
}

function renderHeader() {
  const s = state.seg;
  $('#seg-date').textContent = s.broadcast_date;
  $('#seg-headline').textContent = s.headline;
  $('#seg-tags').innerHTML = (s.cctv_tags || '').split(/\s+/).filter(Boolean).map((t) => `<span>${esc(t)}</span>`).join('');
  $('#seg-qc').textContent = s.qc_flags.length ? '⚠ ' + s.qc_flags.join(' · ') : '';
}

function renderRows() {
  const s = state.seg;
  const rows = $('#rows');
  rows.innerHTML = '';
  state.charEls = [];
  state.pyEls = [];
  state.wordOf = new Array(s.chars.length).fill(null);
  for (const w of s.words) for (let i = w.s; i < w.e; i++) state.wordOf[i] = w;

  state.aligned = s.chars.filter((c) => c.start != null).sort((a, b) => a.start - b.start);

  const byPara = new Map();
  for (const c of s.chars) {
    if (!byPara.has(c.p)) byPara.set(c.p, []);
    byPara.get(c.p).push(c);
  }

  for (const [p, chars] of byPara) {
    const row = document.createElement('div');
    row.className = 'row';
    row.dataset.p = p;
    const zh = document.createElement('div'); zh.className = 'zh';
    const py = document.createElement('div'); py.className = 'py';
    const en = document.createElement('div'); en.className = 'en';

    for (const c of chars) {
      const el = document.createElement('span');
      el.className = 'ch' + (c.start == null ? ' inert' : '');
      el.textContent = c.ch;
      el.dataset.i = c.i;
      if (c.start != null) el.onclick = (e) => { e.stopPropagation(); seek(c.start); };
      zh.appendChild(el);
      state.charEls[c.i] = el;

      const ps = document.createElement('span');
      ps.className = 'ps';
      ps.textContent = c.py || (c.start == null ? c.ch : '');
      if (!c.py && c.start == null) ps.style.color = 'var(--muted)';
      const w = state.wordOf[c.i];
      if (w && c.i === w.e - 1) ps.classList.add('sp');
      ps.dataset.i = c.i;
      if (c.start != null) ps.onclick = (e) => { e.stopPropagation(); seek(c.start); };
      py.appendChild(ps);
      state.pyEls[c.i] = ps;
    }
    const english = s.english && s.english[p];
    en.textContent = english || '(no translation)';
    if (!english) en.classList.add('none');

    for (const cell of [zh, py, en]) {
      cell.onclick = () => selectPara(p);
      row.appendChild(cell);
    }
    rows.appendChild(row);
  }
}

// ---------- playback + highlighting ----------
function seek(t) {
  audio.currentTime = t;
  if (audio.paused) audio.play();
}

function paraRange(p) {
  const cs = state.aligned.filter((c) => c.p === p);
  if (!cs.length) return null;
  return [cs[0].start, cs[cs.length - 1].end];
}

async function selectPara(p) {
  document.querySelectorAll('.row').forEach((r) => r.classList.toggle('selected', +r.dataset.p === p));
  state.selectedPara = p;
  state.loopRange = p >= 0 ? paraRange(p) : null;
  const label = $('#selected-para');
  const rec = $('#btn-record'), ref = $('#btn-play-ref');
  if (!state.loopRange) {
    label.textContent = 'no paragraph selected';
    rec.disabled = ref.disabled = true;
    $('#practice-status').textContent = 'select a paragraph to practise';
    state.refContour = null;
    drawContour();
    return;
  }
  const [t0, t1] = state.loopRange;
  label.textContent = `¶${p + 1}  ${fmt(t0)} – ${fmt(t1)}`;
  rec.disabled = ref.disabled = false;
  $('#practice-status').textContent = 'reference loaded · press record, speak the paragraph, press stop';
  state.compare = null;
  state.liveST = [];
  state.refContour = await (await fetch(`/api/segments/${state.seg.guid}/pitch?t0=${t0}&t1=${t1}`)).json();
  drawContour();
}

function findActive(t) {
  // binary search: last aligned char with start <= t
  const a = state.aligned;
  let lo = 0, hi = a.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid].start <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  if (ans < 0) return -1;
  // if we're past its end and before the next start, keep it (gap) unless the gap is long
  if (t > a[ans].end + 0.35) return -1;
  return a[ans].i;
}

function setActive(i) {
  if (i === state.activeIdx) return;
  document.querySelectorAll('.active, .word').forEach((el) => el.classList.remove('active', 'word'));
  state.activeIdx = i;
  if (i < 0) return;
  state.charEls[i]?.classList.add('active');
  state.pyEls[i]?.classList.add('active');
  if ($('#show-words').checked) {
    const w = state.wordOf[i];
    if (w) for (let k = w.s; k < w.e; k++) {
      if (k !== i) { state.charEls[k]?.classList.add('word'); state.pyEls[k]?.classList.add('word'); }
    }
  }
  const el = state.charEls[i];
  if (el) {
    const r = el.getBoundingClientRect();
    if (r.top < 120 || r.bottom > window.innerHeight - 220) el.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }
}

function tick() {
  if (state.seg) {
    const t = audio.currentTime;
    if (state.loopRange && $('#loop').checked && !audio.paused && t >= state.loopRange[1]) {
      audio.currentTime = state.loopRange[0];
    }
    setActive(findActive(t));
    $('#time').textContent = `${fmt(t)} / ${fmt(audio.duration || 0)}`;
    $('#btn-play').textContent = audio.paused ? '▶' : '❚❚';
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

$('#btn-play').onclick = () => (audio.paused ? audio.play() : audio.pause());
$('#rate').onchange = () => (audio.playbackRate = parseFloat($('#rate').value));
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); $('#btn-play').click(); }
  if (e.code === 'ArrowLeft') audio.currentTime = Math.max(0, audio.currentTime - 3);
  if (e.code === 'ArrowRight') audio.currentTime += 3;
  if (e.code === 'KeyR' && !$('#btn-record').disabled) $('#btn-record').click();
});

// ---------- practice: record + live pitch ----------
let ac = null, worklet = null, micStream = null;

async function ensureMic() {
  if (micStream) return;
  micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  ac = new AudioContext();
  await ac.audioWorklet.addModule('pitch-worklet.js');
  const src = ac.createMediaStreamSource(micStream);
  worklet = new AudioWorkletNode(ac, 'pitch-processor');
  src.connect(worklet);
  worklet.port.onmessage = (e) => {
    if (!state.recorder || state.recorder.state !== 'recording') return;
    const { t, f0 } = e.data;
    if (!state.recT0) state.recT0 = t;
    state.liveF0.push({ t: t - state.recT0, f0 });
    drawContour();
  };
}

$('#btn-record').onclick = async () => {
  const btn = $('#btn-record');
  if (state.recorder && state.recorder.state === 'recording') {
    state.recorder.stop();
    return;
  }
  await ensureMic();
  if (ac.state === 'suspended') await ac.resume();
  state.recChunks = [];
  state.liveF0 = [];
  state.recT0 = 0;
  state.compare = null;
  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus'
             : MediaRecorder.isTypeSupported('audio/mp4') ? 'audio/mp4' : '';
  state.recorder = new MediaRecorder(micStream, mime ? { mimeType: mime } : undefined);
  state.recorder.ondataavailable = (e) => e.data.size && state.recChunks.push(e.data);
  state.recorder.onstop = onRecorded;
  state.recorder.start(100);
  btn.textContent = '■ stop';
  btn.classList.add('recording');
  $('#practice-status').textContent = 'recording…';
  $('#practice-score').textContent = '';
};

async function onRecorded() {
  const btn = $('#btn-record');
  btn.textContent = '● record';
  btn.classList.remove('recording');
  const blob = new Blob(state.recChunks, { type: state.recorder.mimeType });
  if (state.myAudioUrl) URL.revokeObjectURL(state.myAudioUrl);
  state.myAudioUrl = URL.createObjectURL(blob);
  $('#btn-play-mine').disabled = false;
  $('#practice-status').textContent = 'aligning your contour to the reference…';
  const [t0, t1] = state.loopRange;
  const fd = new FormData();
  fd.append('audio', blob, 'rec.webm');
  fd.append('t0', t0); fd.append('t1', t1); fd.append('para', state.selectedPara);
  const r = await fetch(`/api/segments/${state.seg.guid}/compare`, { method: 'POST', body: fd });
  if (!r.ok) { $('#practice-status').textContent = 'comparison failed: ' + r.status; return; }
  state.compare = await r.json();
  const d = state.compare.mean_abs_semitone_diff;
  $('#practice-status').textContent = `done · your median ${state.compare.usr.median_hz} Hz vs reference ${state.compare.ref.median_hz} Hz`;
  $('#practice-score').textContent = d == null ? '' : `mean |Δ| ${d.toFixed(2)} st`;
  drawContour();
}

$('#btn-play-ref').onclick = () => {
  const [t0, t1] = state.loopRange;
  audio.currentTime = t0;
  audio.play();
  const stop = () => { if (audio.currentTime >= t1) { audio.pause(); audio.removeEventListener('timeupdate', stop); } };
  audio.addEventListener('timeupdate', stop);
};
$('#btn-play-mine').onclick = () => { if (state.myAudioUrl) new Audio(state.myAudioUrl).play(); };

// ---------- contour drawing ----------
function drawContour() {
  const cv = $('#contour');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#111317'; ctx.fillRect(0, 0, W, H);

  const ref = state.refContour;
  const ST_MIN = -12, ST_MAX = 12;
  const y = (st) => H - ((Math.max(ST_MIN, Math.min(ST_MAX, st)) - ST_MIN) / (ST_MAX - ST_MIN)) * (H - 20) - 10;

  // grid: one line per 2 semitones, zero emphasised
  for (let st = ST_MIN; st <= ST_MAX; st += 2) {
    ctx.strokeStyle = st === 0 ? '#3a4150' : '#1f242c';
    ctx.lineWidth = st === 0 ? 2 : 1;
    ctx.beginPath(); ctx.moveTo(0, y(st)); ctx.lineTo(W, y(st)); ctx.stroke();
  }
  if (!ref) return;

  const T0 = ref.times[0], T1 = ref.times[ref.times.length - 1];
  const x = (t) => ((t - T0) / Math.max(1e-6, T1 - T0)) * W;

  // character boundaries for the selected paragraph
  ctx.strokeStyle = '#2a3140'; ctx.lineWidth = 1;
  for (const c of state.aligned) if (c.p === state.selectedPara) {
    ctx.beginPath(); ctx.moveTo(x(c.start), 0); ctx.lineTo(x(c.start), H); ctx.stroke();
  }
  ctx.fillStyle = '#6b7280'; ctx.font = '18px sans-serif'; ctx.textAlign = 'center';
  for (const c of state.aligned) if (c.p === state.selectedPara) ctx.fillText(c.ch, x((c.start + c.end) / 2), 22);

  const plot = (times, st, color, width) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round';
    let pen = false;
    ctx.beginPath();
    for (let i = 0; i < times.length; i++) {
      const v = st[i];
      if (v == null || Number.isNaN(v)) { pen = false; continue; }
      const px = x(times[i]), py = y(v);
      if (!pen) { ctx.moveTo(px, py); pen = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
  };

  plot(ref.times, ref.semitones, '#6fb6ff', 4);

  // live: own time axis stretched over the panel, semitones relative to running median
  if (state.liveF0.length && !state.compare) {
    const voiced = state.liveF0.filter((p) => p.f0 > 0).map((p) => p.f0);
    if (voiced.length > 5) {
      const med = voiced.slice().sort((a, b) => a - b)[voiced.length >> 1];
      const dur = Math.max(T1 - T0, state.liveF0[state.liveF0.length - 1].t);
      const times = state.liveF0.map((p) => T0 + p.t * ((T1 - T0) / dur));
      const st = state.liveF0.map((p) => (p.f0 > 0 ? 12 * Math.log2(p.f0 / med) : null));
      plot(times, st, '#9aa0a6', 2.5);
    }
  }

  if (state.compare) plot(ref.times, state.compare.usr_on_ref, '#ff6b57', 3.5);

  // playhead
  if (!audio.paused || audio.currentTime > 0) {
    const t = audio.currentTime;
    if (t >= T0 && t <= T1) {
      ctx.strokeStyle = '#e8b84a'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x(t), 0); ctx.lineTo(x(t), H); ctx.stroke();
    }
  }
}
audio.addEventListener('timeupdate', drawContour);

// ---------- utils ----------
function fmt(s) {
  s = Math.max(0, s || 0);
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${m}:${r.toString().padStart(2, '0')}`;
}
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

loadList();
