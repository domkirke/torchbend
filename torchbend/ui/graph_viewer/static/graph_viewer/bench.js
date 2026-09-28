// ─── the bench, shared by the graph editor and play mode ─────────────────────
// One store per model, in localStorage, that both pages read when they open and
// write on every change: the entries of every input (expressions, files with
// their crop / resample settings, prompts, reference recordings, scalars), which
// one is selected, which are in the batch, the batch mode, and play's enabled
// optional arguments. Each page keeps its own in-memory entry objects and
// converts through the neutral form below, so the two stay in step on every
// switch without either knowing the other's internals.
//
//   store = { v: 2,
//             sets:     { input: [ savedEntry, ... ] },
//             selected: { input: entryId },
//             batch:    { on: bool, mode: "pad" | "loop" | "stack" | "sequential" },
//             optional: [ input, ... ] }
//
//   savedEntry = { id, type: "expr" | "file" | "mode" | "scalar", value, label,
//                  mode?, inBatch?, meta?, mime?, dataUrl?, tooBig? }
//
// A page holds entries only for the method it shows, so a save merges: inputs
// the page does not show keep what the store already had for them.
(function () {
  const PREFIX = "tb_inputs_";
  const OLD_PLAY_PREFIX = "tb_play_inputs_";
  const MAX_FILE_BYTES = 4 * 1024 * 1024;      // per file, to respect the quota
  // the processing a media widget applies to a loaded file (crop, resample…)
  const MEDIA_KEYS = [
    "imgTargetW", "imgTargetH", "imgChannels", "imgMode",
    "cropX", "cropY", "cropW", "cropH",
    "cropStart", "cropEnd", "targetSR", "normalize", "toMono", "monoByShape",
  ];
  const BATCH_MODES = ["pad", "loop", "stack", "sequential"];

  function key(model) { return PREFIX + (model || ""); }

  function _read(k) {
    try { return JSON.parse(localStorage.getItem(k) || "null"); } catch (_) { return null; }
  }

  function empty() {
    return { v: 2, sets: {}, selected: {}, batch: { on: false, mode: "pad" }, optional: [] };
  }

  // The store for `model`, migrating the two formats that came before it: the
  // editor's own (expressions under `value`, no ids, no batch) and play mode's
  // per-method store (whose entries fill any input the editor's did not have).
  function load(model) {
    const cur = _read(key(model));
    if (cur && cur.v === 2) {
      const s = Object.assign(empty(), cur);
      s.batch = Object.assign({ on: false, mode: "pad" }, cur.batch || {});
      return s;
    }
    const s = empty();
    let found = false;
    if (cur && cur.sets) {
      found = true;
      for (const [name, arr] of Object.entries(cur.sets)) {
        s.sets[name] = (arr || []).map((e, i) => Object.assign({ id: "m" + i + "_" + name }, e));
      }
    }
    const oldPlay = _read(OLD_PLAY_PREFIX + (model || ""));
    if (oldPlay && oldPlay.byFn) {
      for (const fnStore of Object.values(oldPlay.byFn)) {
        for (const [name, arr] of Object.entries((fnStore && fnStore.sets) || {})) {
          if (s.sets[name] && s.sets[name].length) continue;
          if (!arr || !arr.length) continue;
          found = true;
          s.sets[name] = arr.map((e, i) => Object.assign({ id: "p" + i + "_" + name }, e));
        }
        (fnStore.optional || []).forEach(n => { if (!s.optional.includes(n)) s.optional.push(n); });
      }
    }
    // the editor's batch switch lived in its own keys before
    try {
      if (localStorage.getItem("tb_graph_batch_on") === "1") s.batch.on = true;
      const m = localStorage.getItem("tb_graph_batch_mode") || localStorage.getItem("tb_play_batch_mode");
      if (BATCH_MODES.includes(m)) s.batch.mode = m;
    } catch (_) {}
    return found ? s : (cur ? s : null);
  }

  function _fileToDataUrl(f) {
    return new Promise((res, rej) => {
      const r = new FileReader();
      r.onload = () => res(r.result);
      r.onerror = rej;
      r.readAsDataURL(f);
    });
  }

  async function dataUrlToFile(dataUrl, name, mime) {
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    return new File([blob], name || "input", { type: mime || blob.type });
  }

  // A file's data URL, computed once and kept on the File itself, so a save
  // made on the way out of the page (which cannot wait for a FileReader) still
  // carries every file that was ever loaded.
  async function ensureDataUrl(file) {
    if (!file || file._tbDataUrl !== undefined) return;
    if (file.size > MAX_FILE_BYTES) { file._tbDataUrl = null; return; }
    try { file._tbDataUrl = await _fileToDataUrl(file); } catch (_) { file._tbDataUrl = null; }
  }

  // neutral entry → what the store holds (synchronous: files use their cached URL)
  function toSaved(n) {
    const out = { id: n.id, type: n.type, value: n.value == null ? "" : n.value, label: n.label || "" };
    if (n.mode) out.mode = n.mode;
    if (n.inBatch === false) out.inBatch = false;
    const meta = {};
    MEDIA_KEYS.forEach(k => { if (n.meta && n.meta[k] !== undefined) meta[k] = n.meta[k]; });
    if (Object.keys(meta).length) out.meta = meta;
    if (n.file) {
      out.mime = n.file.type;
      out.label = out.label || n.file.name;
      if (n.file._tbDataUrl) out.dataUrl = n.file._tbDataUrl;
      else out.tooBig = true;
    }
    return out;
  }

  // what the store holds → neutral entry (asynchronous: files are rebuilt)
  async function fromSaved(s) {
    const n = { id: s.id, type: s.type || "expr", value: s.value == null ? "" : s.value,
                label: s.label || "", mode: s.mode || null, inBatch: s.inBatch !== false,
                meta: Object.assign({}, s.meta || {}), file: null, lost: false };
    // the editor's older format kept an expression under `expr`
    if (n.type === "expr" && !n.value && s.expr) n.value = s.expr;
    if (s.dataUrl) {
      try {
        n.file = await dataUrlToFile(s.dataUrl, s.label, s.mime);
        n.file._tbDataUrl = s.dataUrl;
      } catch (_) { n.lost = true; }
    } else if (s.tooBig || s.type === "file") {
      n.lost = true;       // a file too large to keep: it has to be loaded again
    }
    return n;
  }

  // Write `part` (sets / selected for the inputs this page shows, plus the
  // shared settings) into the store, keeping every other input as it was.
  function save(model, part, names) {
    const s = load(model) || empty();
    const shown = new Set(names || Object.keys(part.sets || {}));
    for (const name of shown) {
      if (part.sets && name in part.sets) s.sets[name] = part.sets[name];
      if (part.selected && part.selected[name] != null) s.selected[name] = part.selected[name];
      else if (part.selected) delete s.selected[name];
    }
    if (part.batch) s.batch = Object.assign({}, s.batch, part.batch);
    if (part.optional) {
      const keep = (s.optional || []).filter(n => !shown.has(n));
      s.optional = keep.concat(part.optional);
    }
    s.v = 2;
    try {
      localStorage.setItem(key(model), JSON.stringify(s));
      return true;
    } catch (_) {
      return false;        // over the quota: the page says so
    }
  }

  window.TBBench = {
    key, load, save, empty, toSaved, fromSaved, ensureDataUrl, dataUrlToFile,
    MEDIA_KEYS, BATCH_MODES, MAX_FILE_BYTES,
  };
})();
