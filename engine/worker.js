// Little Parables voice engine: the Web Worker (a module worker).
// It fetches the engine files, keeps them on the device, copies a voice from a sample and turns text into speech.
// Everything runs on this device. No audio and no text leave the page.
//
// The model code is derived from KevinAHM/pocket-tts-web inference-worker.js (Apache-2.0, see licenses/Apache-2.0.txt).
// Model: Kyutai Pocket TTS english_2026-04, ONNX int8 export by KevinAHM (CC BY 4.0, see licenses/CC-BY-4.0-model.txt).
// One model file is modified: model/flow_lm_main_int8_dyn.onnx accepts an attention cache of any length
// (see tools/make-dyn-model.py). Runtime: onnxruntime-web 1.18.0 (MIT), one thread, unshared memory.
//
// The page talks to this worker through src/voice-engine.js. Every request carries a number (rid) and gets exactly
// one answer with the same number. Messages without a number are news for the page (progress, audio pieces).

const CACHE = 'lp-voice-engine-v1';      // The name of the Cache Storage box. Change it only when the layout changes.
const PART = 4 * 1024 * 1024;            // Big files are kept in pieces of this size; the model files are also fetched that way, so a broken download resumes.
const SR = 24000;                        // The model's sample rate.
const FRAME = 1920;                      // Samples in one model frame (80 ms).
const MAX_FRAMES = 500;                  // The most frames one group of sentences may take (40 seconds).
const CHUNK_GAP = 0.3;                   // Seconds of quiet between two groups of sentences.
const TEMPERATURE = 0.7;
const SLACK = 64;                        // Spare slots kept free in the attention cache.
const VOICES_KEPT = 3;                   // Ready-to-speak voices held in memory at once.

let BASE = '';                           // The engine folder, with a closing slash.
let manifest = null;                     // The list of engine files: { version, bytes, files: [{ path, size, sha256 }] }.
let ort = null, meta = null, tok = null, bos = null;
const sess = {};
const voices = new Map();                // voice id -> { state, used }
let stTensor = null;

/* ---------- small helpers ---------- */

/* An error with a short code the page understands. */
function fail(code, message) { const e = new Error(message || code); e.code = code; return e; }
/* Which code fits an unknown error: running out of memory has many spellings. */
function codeOf(err) {
  if (err && err.code) return err.code;
  const s = String((err && (err.message || err)) || '');
  if (/out of memory|memory\.grow|bad_alloc|allocat|Maximum memory|OOM/i.test(s)) return 'memory';
  return 'engine';
}
const news = (m, transfer) => postMessage(m, transfer || []);
/* Let waiting messages (such as "stop") in. A message channel is used because timers are slowed in hidden tabs. */
const breath = (() => {
  const ch = new MessageChannel(); let wake = null;
  ch.port1.onmessage = () => { const w = wake; wake = null; if (w) w(); };
  return () => new Promise((r) => { wake = r; ch.port2.postMessage(0); });
})();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
/* True when this browser can run the fast (SIMD) kind of WebAssembly. The bytes are a tiny test program. */
function hasSimd() {
  try { return WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11])); }
  catch (e) { return false; }
}

/* ---------- the engine files: fetch once, keep in Cache Storage, resume after a break ---------- */

let cache = null, have = new Set();
async function openCache() {
  try { cache = self.caches ? await caches.open(CACHE) : null; } catch (e) { cache = null; }
  have = new Set();
  if (cache) try { for (const r of await cache.keys()) have.add(r.url); } catch (e) {}
}
const marker = () => BASE + '__lp/complete';
const partKey = (f, i) => BASE + '__lp/' + f.sha256.slice(0, 16) + '/' + f.path + '?part=' + i;
const partCount = (f) => Math.max(1, Math.ceil(f.size / PART));
const isKept = (f) => { for (let i = 0; i < partCount(f); i++) if (!have.has(partKey(f, i))) return false; return true; };

/* Progress, counted per file so nothing is counted twice. */
const prog = { got: new Map(), total: 0, last: 0 };
function progress(f, bytes, force) {
  prog.got.set(f.path, bytes);
  const now = performance.now();
  if (!force && now - prog.last < 120) return;
  prog.last = now;
  let got = 0; for (const v of prog.got.values()) got += v;
  news({ type: 'progress', got: Math.min(got, prog.total), total: prog.total });
}

async function sha256(bytes) {
  if (!(self.crypto && crypto.subtle)) return null;   // Not offered on plain http: the size check still runs.
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  let s = ''; for (const b of d) s += b.toString(16).padStart(2, '0');
  return s;
}

/* Fetch bytes from..to of a file (or the whole file). The stream is read piece by piece so progress can be shown,
   and it is given up when nothing arrives for 25 seconds. Answers { bytes, whole }: whole is true when the server
   sent the complete file although only a part was asked for. */
async function fetchRange(f, from, to, onBytes) {
  const all = from === 0 && to === f.size;
  const ctl = new AbortController();
  let timer = setTimeout(() => ctl.abort(), 25000);
  try {
    const res = await fetch(BASE + f.path, { headers: all ? {} : { Range: `bytes=${from}-${to - 1}` }, signal: ctl.signal, cache: 'no-cache' });
    if (!res.ok) throw fail('network', `HTTP ${res.status} for ${f.path}`);
    const whole = all || res.status !== 206;
    const want = whole ? f.size : to - from;
    const out = new Uint8Array(want); let n = 0;
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      clearTimeout(timer); timer = setTimeout(() => ctl.abort(), 25000);
      if (n + value.length > want) throw fail('network', `${f.path}: more bytes than expected`);
      out.set(value, n); n += value.length;
      onBytes(whole ? n : from + n);
    }
    if (n !== want) throw fail('network', `${f.path}: got ${n} bytes, expected ${want}`);
    return { bytes: out, whole };
  } catch (e) {
    throw e.code ? e : fail('network', `${f.path}: ${e && e.message || e}`);
  } finally { clearTimeout(timer); }
}
/* The same, tried up to four times with a growing wait in between. */
async function fetchRangeRetry(f, from, to, onBytes) {
  let last = null;
  for (let i = 0; i < 4; i++) {
    try { return await fetchRange(f, from, to, onBytes); } catch (e) { last = e; }
    await sleep([800, 2500, 6000, 0][i]);
  }
  throw last;
}
async function keep(key, bytes) {
  if (!cache) return;
  try { await cache.put(key, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } })); have.add(key); }
  catch (e) { news({ type: 'note', code: 'storage', text: String(e && e.message || e) }); cache = null; }   // Full or blocked: carry on without keeping.
}
async function kept(key, want) {
  if (!cache || !have.has(key)) return null;
  try {
    const r = await cache.match(key); if (!r) return null;
    const b = new Uint8Array(await r.arrayBuffer());
    return b.length === want ? b : null;
  } catch (e) { return null; }
}

/* One engine file as bytes: from the device when it is there, else from the network (and then kept).
   `wanted` false means "only make sure it is kept": a file that is already there is not read.
   Only the model files (.onnx) are asked for in pieces. A web server may send scripts and WebAssembly squeezed
   (gzip), and a piece of a squeezed file cannot be used, so those are fetched whole. */
async function getFile(f, wanted = true) {
  const n = partCount(f);
  if (!wanted && isKept(f)) { progress(f, f.size); return null; }
  const out = new Uint8Array(f.size);
  let fresh = false, inPieces = n > 1 && /\.onnx$/.test(f.path);
  const keepWhole = async (bytes) => {   // The whole file came at once. Keep it in pieces all the same.
    out.set(bytes, 0);
    for (let k = 0; k < n; k++) await keep(partKey(f, k), bytes.subarray(k * PART, Math.min(f.size, (k + 1) * PART)));
  };
  for (let i = 0; i < n; i++) {
    const from = i * PART, to = Math.min(f.size, from + PART);
    let part = await kept(partKey(f, i), to - from);
    if (!part) {
      fresh = true;
      if (!inPieces) { await keepWhole((await fetchRangeRetry(f, 0, f.size, (b) => progress(f, b))).bytes); break; }
      let r;
      try { r = await fetchRangeRetry(f, from, to, (b) => progress(f, b)); }
      catch (e) {   // Pieces did not work (a server that handles ranges badly, or no connection): one try for the whole file.
        await keepWhole((await fetchRange(f, 0, f.size, (b) => progress(f, b))).bytes); break;
      }
      if (r.whole) { await keepWhole(r.bytes); break; }   // The server does not do ranges at all.
      part = r.bytes;
      await keep(partKey(f, i), part);
    }
    out.set(part, from);
    progress(f, to);
  }
  if (fresh) {
    const h = await sha256(out);
    if (h && h !== f.sha256) {
      if (cache) for (let i = 0; i < n; i++) { try { await cache.delete(partKey(f, i)); } catch (e) {} have.delete(partKey(f, i)); }
      throw fail('network', `${f.path}: the file arrived damaged`);
    }
  }
  progress(f, f.size, true);
  return out;
}
const fileOf = (path) => { const f = manifest.files.find((x) => x.path === path); if (!f) throw fail('engine', 'not in the manifest: ' + path); return f; };

/* The list of files. The fresh one from the site wins; without a connection the one kept on the device is used. */
async function readManifest() {
  let fresh = null, old = null;
  try { const r = await fetch(BASE + 'manifest.json', { cache: 'no-cache' }); if (r.ok) fresh = await r.json(); } catch (e) {}
  if (cache) try { const r = await cache.match(marker()); if (r) old = await r.json(); } catch (e) {}
  manifest = fresh || old;
  if (!manifest || !Array.isArray(manifest.files)) throw fail('network', 'the list of engine files could not be read');
  return { complete: !!old && old.version === manifest.version };
}
/* Write down that every file is here, and throw away pieces of older versions. */
async function markComplete() {
  if (!cache) return false;
  if (!manifest.files.every(isKept)) return false;
  try {
    await cache.put(marker(), new Response(JSON.stringify(manifest), { headers: { 'content-type': 'application/json' } }));
    const good = new Set([marker()]);
    for (const f of manifest.files) for (let i = 0; i < partCount(f); i++) good.add(partKey(f, i));
    for (const r of await cache.keys()) if (!good.has(r.url)) await cache.delete(r);
    for (const name of await caches.keys()) if (name.startsWith('lp-voice-engine-') && name !== CACHE) await caches.delete(name);
    return true;
  } catch (e) { return false; }
}

/* Import a script from its bytes. When the browser will not import from memory, the plain address is used. */
async function importBytes(bytes, path) {
  const url = URL.createObjectURL(new Blob([bytes], { type: 'text/javascript' }));
  try { return await import(url); }
  catch (e) { return await import(BASE + path); }
  finally { URL.revokeObjectURL(url); }
}

/* ---------- starting the engine ---------- */

const T = (dtype, data, dims) => new ort.Tensor(dtype, data, dims);

async function startRuntime() {
  if (ort) return;
  if (typeof WebAssembly !== 'object') throw fail('unsupported', 'no WebAssembly');
  if (!hasSimd()) throw fail('simd', 'no WebAssembly SIMD');
  const mod = await importBytes(await getFile(fileOf('ort/ort.wasm.min.mjs')), 'ort/ort.wasm.min.mjs');
  const o = mod.default || mod;
  const wasm = await getFile(fileOf('ort/ort-wasm-simd.wasm'));
  o.env.wasm.wasmPaths = { 'ort-wasm-simd.wasm': URL.createObjectURL(new Blob([wasm], { type: 'application/wasm' })) };
  o.env.wasm.simd = true;
  o.env.wasm.numThreads = 1;
  o.env.wasm.proxy = false;
  ort = o;
  meta = JSON.parse(new TextDecoder().decode(await getFile(fileOf('model/bundle.json'))));
}
async function openSession(key, path) {
  const bytes = await getFile(fileOf(path));
  news({ type: 'stage', stage: 'start', part: key });
  sess[key] = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
}

function parseNpy(u8) {
  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const major = view.getUint8(6);
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const off = major === 1 ? 10 : 12;
  const text = new TextDecoder().decode(u8.subarray(off, off + headerLen));
  const shape = text.match(/\(\s*([0-9,\s]+)\)/)[1].split(',').map((x) => x.trim()).filter(Boolean).map(Number);
  return { data: new Float32Array(u8.slice(off + headerLen).buffer), shape };
}

/* mode "speak": every file is fetched and kept, and the four parts that speak are started.
   mode "encoder": only the part that listens to a voice sample is started (in a worker that is closed afterwards). */
async function load(m) {
  const t0 = performance.now();
  BASE = m.base;
  await openCache();
  const { complete } = await readManifest();
  const speak = m.mode !== 'encoder';
  // Two files are not needed to start speaking: the encoder (used when a voice is copied) and the storyteller's
  // voice. They are fetched now only when there is somewhere to keep them.
  const need = speak
    ? manifest.files.filter((f) => cache || !/mimi_encoder|storyteller\.voice/.test(f.path))
    : manifest.files.filter((f) => /^ort\/|bundle\.json$|mimi_encoder/.test(f.path));
  prog.got.clear(); prog.total = need.reduce((s, f) => s + f.size, 0); prog.last = 0;
  news({ type: 'manifest', version: manifest.version, total: prog.total, complete });
  await startRuntime();
  if (!speak) {
    await openSession('mimi_encoder', 'model/mimi_encoder_int8.onnx');
    return { ms: Math.round(performance.now() - t0) };
  }
  await openSession('text_conditioner', 'model/text_conditioner_int8.onnx');
  await openSession('flow_lm_main', 'model/flow_lm_main_int8_dyn.onnx');
  await openSession('flow_lm_flow', 'model/flow_lm_flow_int8.onnx');
  await openSession('mimi_decoder', 'model/mimi_decoder_int8.onnx');
  const sp = await importBytes(await getFile(fileOf('sentencepiece.js')), 'sentencepiece.js');
  tok = new sp.SentencePieceProcessor();
  const tb = await getFile(fileOf('model/' + meta.tokenizer_file));
  let b64 = ''; for (let i = 0; i < tb.length; i += 0x8000) b64 += String.fromCharCode.apply(null, tb.subarray(i, i + 0x8000));
  await tok.loadFromB64StringModel(btoa(b64));
  bos = parseNpy(await getFile(fileOf('model/' + meta.bos_before_voice_file)));
  stTensor = { s: T('float32', new Float32Array([0]), [1, 1]), t: T('float32', new Float32Array([1]), [1, 1]) };
  if (cache) for (const f of need) await getFile(f, false);   // The files not needed right now (see above).
  const cached = await markComplete();
  let got = 0; for (const v of prog.got.values()) got += v;
  return { ms: Math.round(performance.now() - t0), total: prog.total, got, cached, version: manifest.version, wasm: wasmBytes() };
}

/* How much memory the engine holds (the WebAssembly heap), for the test page. */
const wasmMemories = [];
{
  const Orig = WebAssembly.Memory;
  const note = (m) => { if (m && !wasmMemories.includes(m)) wasmMemories.push(m); };
  for (const name of ['instantiate', 'instantiateStreaming']) {
    const orig = WebAssembly[name];
    if (typeof orig !== 'function') continue;
    WebAssembly[name] = async function (...a) {
      const r = await orig.apply(this, a);
      try { for (const v of Object.values((r.instance || r).exports)) if (v instanceof Orig) note(v); } catch (e) {}
      return r;
    };
  }
}
function wasmBytes() { let n = 0; for (const m of wasmMemories) { try { n += m.buffer.byteLength; } catch (e) {} } return n; }

/* ---------- the model's memory between steps ---------- */

function filled(shape, dtype, fill) {
  const size = shape.reduce((a, b) => a * b, 1);
  if (dtype === 'int64') return new BigInt64Array(size);
  if (dtype === 'bool') return new Uint8Array(size).fill(fill === 'ones' ? 1 : 0);
  const d = new Float32Array(size);
  if (fill === 'nan') d.fill(NaN); else if (fill === 'ones') d.fill(1);
  return d;
}
function initState(list, cacheLen) {
  const s = {};
  for (const e of list) {
    const shape = (cacheLen && e.shape.length === 5) ? [e.shape[0], e.shape[1], cacheLen, e.shape[3], e.shape[4]] : e.shape;
    s[e.input_name] = T(e.dtype, filled(shape, e.dtype, e.fill), shape);
  }
  return s;
}
// The model keeps what it has heard so far in a cache. The stock export passes a fixed 1000-slot cache (49 MB) in
// and out on every step. Here the cache is only as long as needed, which makes each step much cheaper.
function usedSlots(state) {
  for (const e of meta.flow_lm_state_manifest) if (e.key === 'step') return Number(state[e.input_name].data[0]);
  return 0;
}
function ensureRoom(state, need) {
  for (const e of meta.flow_lm_state_manifest) {
    if (e.shape.length !== 5) continue;
    const t = state[e.input_name];
    const [a, b, n, c, d] = t.dims;
    if (n >= need) continue;
    const row = c * d, slab = n * row, newSlab = need * row;
    const out = new Float32Array(a * b * newSlab).fill(NaN);
    for (let i = 0; i < a * b; i++) out.set(t.data.subarray(i * slab, (i + 1) * slab), i * newSlab);
    state[e.input_name] = T('float32', out, [a, b, need, c, d]);
  }
}
function updateState(state, result, list) { for (const e of list) state[e.input_name] = result[e.output_name]; }

/* ---------- voices ---------- */

/* Turn the small form of a voice (what the encoder heard: [1, frames, 1024] numbers) into the ready-to-speak
   state: one pass through the language model. */
async function useVoice(id, data, shape) {
  let d = data, dims = shape.slice();
  if (meta.insert_bos_before_voice && bos) {
    const c = new Float32Array(bos.data.length + d.length);
    c.set(bos.data, 0); c.set(d, bos.data.length);
    d = c; dims = [1, dims[1] + bos.shape[1], dims[2]];
  }
  const state = initState(meta.flow_lm_state_manifest, dims[1] + 2);
  const r = await sess.flow_lm_main.run({
    sequence: T('float32', new Float32Array(0), [1, 0, meta.latent_dim]),
    text_embeddings: T('float32', d, dims),
    ...state,
  });
  updateState(state, r, meta.flow_lm_state_manifest);
  voices.set(id, { state, used: performance.now() });
  while (voices.size > VOICES_KEPT) {   // Forget the voice that was used longest ago, and tell the page.
    let old = null; for (const [k, v] of voices) if (k !== id && (!old || v.used < voices.get(old).used)) old = k;
    if (!old) break;
    voices.delete(old); news({ type: 'voice_dropped', id: old });
  }
}
/* The storyteller's voice comes with the engine: a small header ("LPV1", frames, width) and then the numbers. */
async function useStoryteller() {
  const u8 = await getFile(fileOf('storyteller.voice'));
  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== 'LPV1') throw fail('engine', 'storyteller.voice is not a voice file');
  const frames = view.getUint32(4, true), dim = view.getUint32(8, true);
  await useVoice('storyteller', new Float32Array(u8.slice(16, 16 + frames * dim * 4).buffer), [1, frames, dim]);
}

/* ---------- a voice sample: tidy it, then encode it ---------- */

/* Change the sample rate with a windowed-sinc filter (16 zero crossings each side). */
function resample(x, from, to) {
  if (from === to) return x;
  const step = from / to, n = Math.floor(x.length / step), out = new Float32Array(n);
  const cut = Math.min(1, to / from) * 0.94, half = 16 / cut;
  const RES = 256, table = new Float32Array(Math.ceil(half * RES) + 2);   // The filter, sampled finely, read with rounding.
  for (let i = 0; i < table.length; i++) {
    const t = i / RES, a = Math.PI * t * cut;
    table[i] = t >= half ? 0 : (t === 0 ? 1 : Math.sin(a) / a) * (0.5 + 0.5 * Math.cos(Math.PI * t / half)) * cut;
  }
  for (let i = 0; i < n; i++) {
    const c = i * step, lo = Math.max(0, Math.ceil(c - half)), hi = Math.min(x.length - 1, Math.floor(c + half));
    let s = 0;
    for (let k = lo; k <= hi; k++) s += x[k] * table[Math.round(Math.abs(k - c) * RES)];
    out[i] = s;
  }
  return out;
}

/* Make the sample ready for the encoder: 24 kHz, at most maxSec, quiet start removed, ended at a pause, a steady
   level, and 0.3 s of silence at the end. Refuses a sample that is too short ("short") or too quiet ("quiet"),
   because those give a random voice. */
function tidySample(pcm, rate, maxSec) {
  let x = resample(pcm, rate, SR);
  const W = 480, win = (a, w) => { let e = 0; for (let k = 0; k < W; k++) e += a[w * W + k] ** 2; return Math.sqrt(e / W); };
  let peak = 0; for (let i = 0; i < x.length; i++) { const v = Math.abs(x[i]); if (v > peak) peak = v; }
  if (x.length < 3 * SR) throw fail('short', 'the sample is shorter than 3 seconds');
  if (peak < 0.01) throw fail('quiet', 'the sample is silent');
  const g = Math.min(0.9 / peak, 8);   // A gentle level fix for quiet phone recordings.
  // The quiet before the first word carries nothing: start 0.1 s before the first loud moment.
  let nW = Math.floor(x.length / W), first = 0;
  while (first < nW && win(x, first) * g <= 0.02) first++;
  x = x.subarray(Math.max(0, first * W - Math.floor(0.1 * SR)));
  if (x.length > maxSec * SR) x = x.subarray(0, Math.floor(maxSec * SR));
  nW = Math.floor(x.length / W);
  const rms = new Float32Array(nW);
  let loud = 0;
  for (let w = 0; w < nW; w++) { rms[w] = win(x, w); if (rms[w] * g > 0.02) loud++; }
  const speech = loud * W / SR;
  if (speech < 3) throw fail(x.length < 5 * SR ? 'short' : 'quiet', `only ${speech.toFixed(1)} s of voice was heard`);
  // End the sample at a pause. A sample cut in the middle of a word makes the copied voice swallow or change
  // the first word of what it reads (measured: "Pip the rabbit" came out as "the rabbit").
  const sorted = Array.from(rms).sort((a, b) => a - b);
  const quiet = Math.max(0.004, 0.12 * (sorted[Math.floor(nW * 0.8)] || 0));
  let end = x.length, endedAtPause = false, run = 0;
  for (let w = nW - 1; w >= Math.max(0, nW - Math.floor(4 * SR / W)); w--) {   // Look back up to 4 seconds.
    if (rms[w] < quiet) { run++; if (run >= 8) { end = (w + 6) * W; endedAtPause = true; break; } } else run = 0;
  }
  const a = new Float32Array(end + Math.floor(0.3 * SR));
  for (let i = 0; i < end; i++) a[i] = x[i] * g;
  const fade = Math.min(end, Math.floor(0.03 * SR));
  for (let i = 0; i < fade; i++) a[end - 1 - i] *= i / fade;
  return { audio: a, seconds: a.length / SR, speech, peak, endedAtPause };
}

// The copied voice, small form: the Mimi encoder output, [1, frames, 1024] float32 (12.5 frames per second of sample).
// The sample is encoded in 3-second pieces with 1 second of lead-in. Encoding it whole needs over 1 GB of memory
// for a 12-second sample; in pieces it needs about 0.2 GB and gives the same result (cosine 0.9999 per frame).
const PIECE = 37, LEAD = 13;
async function encode(m) {
  const t0 = performance.now();
  const tidy = tidySample(m.pcm, m.rate, m.maxSec || 20);
  const audio = tidy.audio, n = Math.floor(audio.length / FRAME);
  const out = new Float32Array(n * 1024);
  for (let i = 0; i < n;) {
    const j = Math.min(n, i + PIECE), s = Math.max(0, i - LEAD);
    const piece = audio.slice(s * FRAME, j * FRAME);
    const r = await sess.mimi_encoder.run({ audio: T('float32', piece, [1, 1, piece.length]) });
    const emb = r[sess.mimi_encoder.outputNames[0]].data;
    out.set(emb.subarray((i - s) * 1024, (j - s) * 1024), i * 1024);
    news({ type: 'encode_progress', done: j, total: n });
    i = j;
    await breath();
  }
  return {
    data: out, shape: [1, n, 1024],
    info: { seconds: +tidy.seconds.toFixed(2), speech: +tidy.speech.toFixed(1), peak: +tidy.peak.toFixed(3), endedAtPause: tidy.endedAtPause, ms: Math.round(performance.now() - t0), wasm: wasmBytes() },
    transfer: [out.buffer],
  };
}

/* ---------- text: sentences, and groups of sentences the model reads in one go ---------- */

/* The sentences of a text as { at, len } in the text as it was given, so the page can light the right words. */
function sentences(text) {
  const out = [];
  for (const m of text.matchAll(/[^.!?…]*[.!?…]+["'”’)\]]*|[^.!?…]+$/g)) {
    const lead = m[0].length - m[0].trimStart().length, body = m[0].trim();
    if (!/[0-9A-Za-zÀ-ɏ]/.test(body)) {   // Stray punctuation belongs to the sentence before it.
      if (out.length && body) out[out.length - 1].len = m.index + lead + body.length - out[out.length - 1].at;
      continue;
    }
    out.push({ at: m.index + lead, len: body.length });
  }
  return out;
}
/* What the model is given for one stretch of text: one line, plain quotes, a capital first, a full stop last. */
function forModel(s) {
  let p = s.trim().replace(/\s+/g, ' ').replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/[–—]/g, ',').replace(/…/g, '...');
  if (!p) return '';
  if (!/[A-Z]/.test(p[0])) p = p[0].toUpperCase() + p.slice(1);
  if (/[0-9A-Za-z]/.test(p[p.length - 1])) p += '.';
  return p;
}
const tokens = (s) => tok.encodeIds(forModel(s)).length;
/* Groups of whole sentences, each at most as long as the model can take at once (50 tokens). A single sentence
   that is longer is cut after a comma where possible, else between words. */
function groups(text) {
  const max = Number(meta.max_token_per_chunk || 50);
  const parts = [];
  for (const s of sentences(text)) {
    const body = text.slice(s.at, s.at + s.len);
    if (tokens(body) <= max) { parts.push(s); continue; }
    let from = 0, lastGood = -1, lastComma = -1;
    const cuts = [...body.matchAll(/\s+/g)].map((m) => ({ at: m.index, next: m.index + m[0].length, comma: /[,;:]$/.test(body.slice(0, m.index)) }));
    for (let i = 0; i <= cuts.length; i++) {
      const stop = i < cuts.length ? cuts[i].at : body.length;
      if (tokens(body.slice(from, stop)) > max && lastGood >= 0) {
        const c = cuts[lastComma >= 0 ? lastComma : lastGood];
        parts.push({ at: s.at + from, len: c.at - from });
        from = c.next; lastGood = -1; lastComma = -1;
        i = cuts.indexOf(c);
        continue;
      }
      if (i < cuts.length) { lastGood = i; if (cuts[i].comma) lastComma = i; }
    }
    if (from < body.length) parts.push({ at: s.at + from, len: body.length - from });
  }
  const out = [];
  for (const p of parts) {
    const g = out[out.length - 1];
    if (g && tokens(text.slice(g.at, p.at + p.len)) <= max) { g.len = p.at + p.len - g.at; g.sents.push(p); }
    else out.push({ at: p.at, len: p.len, sents: [p] });
  }
  return out;
}

/* ---------- pacing: the model's silences, tidied as the sound comes out ----------
   The model starts every group with 0.1 to 0.6 s of silence and sometimes pauses for a second between sentences.
   This lets the silence before the first sound go, shortens any pause to at most MAX_PAUSE, and keeps only a
   short tail. It also writes down where the pauses are, which tells the page where each sentence begins. */
const WIN = 240, QUIET = 0.008, LEAD_KEEP = 0.06, MAX_PAUSE = 0.55, PAUSE_HEAD = 0.3, TAIL_KEEP = 0.12, PAUSE_NOTE = 0.18;
function shaper() {
  let carry = new Float32Array(0), held = [], heldN = 0, started = false, outN = 0;
  const pauses = [];
  const join = (list, n) => { const o = new Float32Array(n); let k = 0; for (const p of list) { o.set(p, k); k += p.length; } return o; };
  const take = (list) => {   // Loud sound again: let the held silence out, shortened if it was long.
    let quietPart = join(held, heldN); held = []; heldN = 0;
    if (!started) { quietPart = quietPart.subarray(Math.max(0, quietPart.length - Math.floor(LEAD_KEEP * SR))); started = true; }
    else {
      if (quietPart.length > MAX_PAUSE * SR) {
        const head = Math.floor(PAUSE_HEAD * SR), tail = Math.floor((MAX_PAUSE - PAUSE_HEAD) * SR);
        quietPart = join([quietPart.subarray(0, head), quietPart.subarray(quietPart.length - tail)], head + tail);
      }
      if (quietPart.length >= PAUSE_NOTE * SR) pauses.push({ at: outN / SR, dur: quietPart.length / SR });
    }
    list.push(quietPart); outN += quietPart.length;
  };
  return {
    pauses,
    get started() { return started; },
    feed(pcm) {
      const x = carry.length ? join([carry, pcm], carry.length + pcm.length) : pcm;
      const nW = Math.floor(x.length / WIN), out = [];
      for (let w = 0; w < nW; w++) {
        const seg = x.subarray(w * WIN, (w + 1) * WIN);
        let e = 0; for (let k = 0; k < WIN; k++) e += seg[k] * seg[k];
        if (Math.sqrt(e / WIN) < QUIET) { held.push(seg); heldN += WIN; continue; }
        if (heldN || !started) take(out);
        out.push(seg); outN += WIN;
      }
      carry = x.slice(nW * WIN);
      let n = 0; for (const p of out) n += p.length;
      return join(out, n);
    },
    end() {
      if (!started) { held = []; heldN = 0; return new Float32Array(0); }
      const tail = join(held, heldN).subarray(0, Math.floor(TAIL_KEEP * SR));
      held = []; heldN = 0; outN += tail.length;
      return tail.slice();
    },
  };
}

/* ---------- speaking ---------- */

function gauss(n, std) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    a[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * std;
  }
  return a;
}

const stopped = new Set();   // Jobs the page asked to stop.
let line = Promise.resolve();   // One piece of model work at a time: the model cannot do two things at once.

/* Read `text` in voice `voice`, from group number `from` on. The sound goes to the page in pieces as it is made. */
async function generate(m) {
  const job = m.job;
  if (!voices.has(m.voice)) {
    if (m.voice === 'storyteller') await useStoryteller();
    else throw fail('novoice', 'the voice is not loaded');
  }
  const voice = voices.get(m.voice); voice.used = performance.now();
  const list = groups(String(m.text || ''));
  news({ type: 'chunks', job, list });
  const L = meta.latent_dim, std = Math.sqrt(TEMPERATURE);
  const emptySeq = T('float32', new Float32Array(0), [1, 0, L]);
  const emptyText = T('float32', new Float32Array(0), [1, 0, meta.conditioning_dim]);
  const tStart = performance.now();
  let frames = 0, loopMs = 0, firstMs = null, outSamples = 0, firstAt = null, firstSamples = 0, peakWasm = wasmBytes();
  const halted = () => stopped.has(job);

  for (let ci = Math.max(0, m.from | 0); ci < list.length && !halted(); ci++) {
    const body = forModel(m.text.slice(list[ci].at, list[ci].at + list[ci].len));
    const afterEos = meta.model_recommended_frames_after_eos != null ? Number(meta.model_recommended_frames_after_eos) : (body.split(/\s+/).length <= 4 ? 3 : 1);
    const state = { ...voice.state };
    let mimi = initState(meta.mimi_state_manifest);
    const ids = tok.encodeIds(body);
    let te = (await sess.text_conditioner.run({ token_ids: T('int64', BigInt64Array.from(ids.map((x) => BigInt(x))), [1, ids.length]) }))[sess.text_conditioner.outputNames[0]];
    if (te.dims.length === 2) te = T('float32', new Float32Array(te.data), [1, te.dims[0], te.dims[1]]);
    ensureRoom(state, usedSlots(state) + ids.length + SLACK);
    const cond = await sess.flow_lm_main.run({ sequence: emptySeq, text_embeddings: te, ...state });
    updateState(state, cond, meta.flow_lm_state_manifest);

    const shape = shaper(), lat = [];
    let decoded = 0, eosStep = null, sent = 0, pieces = 0;
    const tLoop = performance.now();
    const send = (pcm) => {
      if (!pcm.length) return;
      if (firstMs == null) firstMs = performance.now() - tStart;
      sent += pcm.length; outSamples += pcm.length;
      // How fast speech is being made, in seconds of sound per second of work. The page uses it to decide how far
      // to buffer ahead. rate counts the sound as the model makes it. outRate counts what the listener gets (after
      // the silences are tidied), from the first piece of this job on; it is 0 until a second piece exists.
      const now = performance.now();
      const rate = (frames * FRAME / SR) / Math.max(0.001, (loopMs + now - tLoop) / 1000);
      const outRate = firstAt == null ? 0 : ((outSamples - firstSamples) / SR) / Math.max(0.001, (now - firstAt) / 1000);
      if (firstAt == null) { firstAt = now; firstSamples = outSamples; }
      news({ type: 'audio', job, chunk: ci, pcm, rate, outRate }, [pcm.buffer]);
    };
    let cur = T('float32', new Float32Array(L).fill(NaN), [1, 1, L]);
    for (let step = 0; step < MAX_FRAMES; step++) {
      await breath();
      if (halted()) break;
      if (step % 16 === 0) ensureRoom(state, usedSlots(state) + 16 + SLACK);
      const ar = await sess.flow_lm_main.run({ sequence: cur, text_embeddings: emptyText, ...state });
      if (ar.eos_logit.data[0] > -4.0 && eosStep == null) eosStep = step;
      const last = (eosStep != null && step >= eosStep + afterEos) || step === MAX_FRAMES - 1;
      const x = gauss(L, std);
      const fr = await sess.flow_lm_flow.run({ c: ar.conditioning, s: stTensor.s, t: stTensor.t, x: T('float32', x, [1, L]) });
      const dir = fr.flow_dir.data;
      for (let i = 0; i < L; i++) x[i] += dir[i];
      lat.push(new Float32Array(x)); frames++;
      cur = T('float32', x, [1, 1, L]);
      updateState(state, ar, meta.flow_lm_state_manifest);

      // Decode in short pieces (a quarter of a second) until the first sound is out, so it comes quickly; then one
      // more short piece and a half-second one, so the reading has something in hand; then a second at a time.
      const pending = lat.length - decoded;
      const size = !shape.started || pieces < 1 ? 3 : pieces < 2 ? 6 : 12;
      const n = last ? pending : pending >= size ? size : 0;
      if (n > 0) {
        const d = new Float32Array(n * L);
        for (let f = 0; f < n; f++) d.set(lat[decoded + f], f * L);
        const dr = await sess.mimi_decoder.run({ latent: T('float32', d, [1, n, L]), ...mimi });
        for (const e of meta.mimi_state_manifest) mimi[e.input_name] = dr[e.output_name];
        decoded += n;
        const was = shape.started;
        send(shape.feed(new Float32Array(dr[sess.mimi_decoder.outputNames[0]].data)));
        if (was) pieces++;
        peakWasm = Math.max(peakWasm, wasmBytes());
      }
      if (last) break;
    }
    loopMs += performance.now() - tLoop;
    if (halted()) break;
    send(shape.end());
    const speech = sent;
    if (ci < list.length - 1) send(new Float32Array(Math.floor(CHUNK_GAP * SR)));
    news({ type: 'chunk_done', job, chunk: ci, speech, samples: sent, pauses: shape.pauses });
  }
  const was = halted(); stopped.delete(job);
  const totalMs = performance.now() - tStart, made = frames * FRAME / SR;
  return {
    stopped: was,
    metrics: {
      chunks: list.length, frames, madeSec: +made.toFixed(2), outSec: +(outSamples / SR).toFixed(2), totalMs: Math.round(totalMs),
      firstMs: firstMs == null ? null : Math.round(firstMs), speed: +(made / Math.max(0.001, totalMs / 1000)).toFixed(2),
      loopSpeed: +(made / Math.max(0.001, loopMs / 1000)).toFixed(2), outSpeed: +((outSamples / SR) / Math.max(0.001, totalMs / 1000)).toFixed(2),
      wasm: Math.max(peakWasm, wasmBytes()),
    },
  };
}

/* ---------- messages ---------- */

const handlers = {
  load,
  encode,
  generate: (m) => { const p = line.then(() => generate(m)); line = p.catch(() => {}); return p; },
  voice: (m) => { const p = line.then(async () => { const t0 = performance.now(); await useVoice(m.id, m.data, m.shape); return { ms: Math.round(performance.now() - t0) }; }); line = p.catch(() => {}); return p; },
  drop_voice: async (m) => { voices.delete(m.id); return {}; },
  memory: async () => ({ wasm: wasmBytes(), voices: [...voices.keys()] }),
};
self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'stop') { stopped.add(m.job); return; }
  try {
    const h = handlers[m.type];
    if (!h) throw fail('engine', 'unknown request ' + m.type);
    const r = await h(m);
    const transfer = r.transfer || []; delete r.transfer;
    postMessage({ rid: m.rid, ok: true, ...r }, transfer);
  } catch (err) {
    postMessage({ rid: m.rid, ok: false, code: codeOf(err), message: String(err && (err.message || err)).slice(0, 400) });
  }
};
postMessage({ type: 'up' });
