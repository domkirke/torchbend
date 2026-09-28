/* TorchBend — Play mode
 *
 * A lightweight performance UI over a compiled (scripted or eager) bended model.
 * It exposes only the defined macros + an input bench, and runs either in
 * realtime (forward on every macro change) or offline (forward on demand).
 */
(function () {
  "use strict";

  // The same page serves generate mode (see generate.js): the input bench and
  // the compile are shared, nothing runs on its own, and every included entry
  // of a placeholder is one of the inputs generation iterates over.
  const PAGE_MODE = window.PLAY_PAGE_MODE || "play";

  // ── state ────────────────────────────────────────────────────────────────
  const state = {
    fn: window.PLAY_DEFAULT_FN || "forward",
    device: "cpu",
    scripted: false,
    mode: null,            // active runtime: "scripted" | "eager"
    macros: [],
    placeholders: [],
    inputs: {},            // name -> [{id, kind:'expr'|'file'|'scalar', value|file, valid}]
    optionalOn: new Set(),  // optional placeholders the user asked for explicitly
    // Which entry runs when batch is off, per placeholder. Like the entries and
    // the batch settings, shared with the graph editor through the bench store
    // (bench.js) — both pages read it on arrival and write it on every change.
    selected: {},
    batchOn: false,         // off by default, as in the editor (the setting is shared)
    batchSupported: true,   // the interface can say a method takes one input at a time
    runMode: "realtime",   // "realtime" | "offline"
    // how several entries on one placeholder become a run — see the server's
    // _PLAY_BATCH_MODES: stack | pad | loop | sequential
    batchMode: "pad",
    adderOpen: new Set(),   // placeholders whose "add an input" block is expanded
    availParams: [],        // every BendingParameter the editor knows
    derivedOpen: localStorage.getItem("tb_play_derived") === "1",
    macroEdit: false,       // show each macro's attachments and their ranges
    bindings: [],           // active bendings, with their parameters
    bendableNodes: [],      // nodes a new bending can attach to
    aliases: {},            // {alias: [node,…]} — what #alias searches against
    callbacks: null,        // available bending callback types (fetched once)
    // Operations the *interface* declares (see /api/callbacks/) — distinct from
    // `callbacks` above, which is the bending vocabulary.
    ifaceCallbacks: [],
    inputModes: {},         // placeholder -> declared alternative input mode
    runTarget: "",          // "" = the compiled graph; otherwise a callback name
    promotable: [],         // bendable callback params no macro drives yet
    orient: localStorage.getItem("tb_play_orient") || "columns",  // "columns" | "rows"
    folded: _loadFolded(),              // {inputs, output}: panes reduced to their header
    compiled: false,
    compiling: true,       // the page compiles on load and runs straight after
    queued: false,         // a run is scheduled but has not started yet
    running: false,
    abort: null,           // AbortController of the run in flight
    lastRunMs: 0,          // observed forward-pass duration, paces realtime restarts
    resample: false,       // ask the next run for a fresh draw of the inputs
    // bumped once per forward pass: the rendered WAVs a view caches belong to
    // one run, and this is what tells the views to drop them
    audioEpoch: 0,
  };

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, txt) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  };
  const genId = () => Date.now().toString(36) + Math.random().toString(36).slice(2);

  function toast(msg, isError) {
    const t = $("play-toast");
    t.textContent = msg;
    t.classList.toggle("error", !!isError);
    t.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => t.classList.remove("show"), isError ? 6000 : 2500);
  }

  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw _httpError(data, r);
    return data;
  }

  // ── errors ───────────────────────────────────────────────────────────────────
  // A failure in play mode is a failure in the model's own forward, so it gets
  // the same treatment as in the editor: the server locates the line in user
  // code and TBTraceError (errors.js) shows it with its source. Carry the whole
  // payload on the Error so whoever catches it can decide.
  function _httpError(data, r) {
    const err = new Error((data && data.error) || `HTTP ${r.status}`);
    err.payload = data || null;
    return err;
  }

  function showTraceError(title, data) {
    TBTraceError.show(title, data, { onCopyPath: () => toast("Path copied") });
  }

  // Located failures earn the panel; a bare message stays a toast — a panel for
  // "HTTP 500" would only be in the way.
  function reportError(title, e) {
    const data = e && e.payload;
    if (window.TBTraceError && TBTraceError.isLocated(data)) {
      showTraceError(title, data);
      return `${data.error_type || "Error"}: ${data.error}`;
    }
    toast(e.message, true);
    return e.message;
  }

  // ── compile ────────────────────────────────────────────────────────────────
  async function compile() {
    state.compiled = false;
    state.compiling = true;
    if (state.abort) { state.abort.abort(); state.abort = null; }
    state.running = false;
    _syncRunUI();
    $("play-status").textContent = "Compiling…";
    $("play-mode-badge").textContent = "";
    $("play-mode-badge").className = "play-mode-badge";
    try {
      const data = await postJSON("/api/play/compile/", {
        fn: state.fn, device: state.device, scripted: state.scripted,
      });
      state.mode = data.mode;
      // the server moved the model (or refused to): show where it really is
      if (data.device) { state.device = data.device; if ($("play-device")) $("play-device").value = data.device; }
      // carry pinned slider ranges across recompiles (they're computed from initial value)
      const prevMacros = {};
      state.macros.forEach(m => { prevMacros[m.name] = m; });
      state.macros = data.macros || [];
      state.macros.forEach(m => {
        const prev = prevMacros[m.name];
        if (prev && m.min == null && prev._sliderMin != null) {
          m._sliderMin = prev._sliderMin;
          m._sliderMax = prev._sliderMax;
        }
      });
      // the method's inputs are about to be swapped for the new one's: write
      // the bench first — the debounced save may not have caught the last edit
      _savePlayBenchNow();
      state.placeholders = data.placeholders || [];
      state.inputModes = data.input_modes || {};
      state.batchSupported = data.batch_supported !== false;
      _syncBatchUI();
      await pruneInputsToPlaceholders();
      state.availParams = data.available_params || state.availParams;
      state.promotable = data.promotable_params || state.promotable;
      state.bindings = data.bindings || [];
      state.bendableNodes = data.bendable_nodes || state.bendableNodes;
      state.aliases = data.aliases || state.aliases;
      state.compiled = true;
      renderMacros();
      renderBendings();
      renderInputs();
      updateModeBadge(data);
      // hand "busy" over to the run that is about to be scheduled — clearing it
      // before that would flash the empty state between compile and first run
      state.compiling = false;
      _syncRunUI();
      const note = data.scripting_error
        ? ` · scripting fell back to eager (${shorten(data.scripting_error)})`
        : "";
      $("play-status").textContent =
        `Ready · ${data.mode} runtime on ${data.device} · compiled in ${data.compile_ms} ms${note}`;
      document.dispatchEvent(new CustomEvent("tb:compiled", { detail: data }));
      if (state.runMode === "realtime") scheduleRun();
    } catch (e) {
      state.compiling = false;
      $("play-status").textContent = "Compile failed: " + reportError("Compile failed", e);
      _syncRunUI();          // keeps Generate disabled, with a reason in its title
      loadDevices();         // a refused device switch: back to where the model is
    }
  }

  function shorten(s) { return s.length > 80 ? s.slice(0, 77) + "…" : s; }

  function updateModeBadge(data) {
    const b = $("play-mode-badge");
    b.textContent = data.mode === "scripted" ? "scripted" : "eager";
    b.className = "play-mode-badge " + (data.mode === "scripted" ? "scripted" : "eager");
    b.title = data.mode === "scripted"
      ? "TorchScript runtime (experimental) — fastest, but macro changes may not propagate"
      : "Eager runtime — evaluates the bended graph live; macro changes apply immediately";
  }

  // ── macros ──────────────────────────────────────────────────────────────────
  function renderMacros() {
    const host = $("play-macros");
    host.innerHTML = "";
    $("play-macro-count").textContent = String(state.macros.length);
    $("play-no-macros").style.display = state.macros.length ? "none" : "block";
    _syncMacroPicker();
    renderDerived();

    state.macros.forEach((m) => {
      const card = el("div", "play-macro");
      card.dataset.macro = m.name;
      _makeMacroDropTarget(card, m);
      const head = el("div", "play-macro-head");
      head.appendChild(el("span", "play-macro-name", m.name));
      const tags = el("span", "play-macro-tags");
      tags.appendChild(el("span", "play-tag", m.param_type));
      // A normalised macro reads 0…1 whatever it drives. The range that spans
      // belongs to each attachment, so with several links disagreeing there is
      // no single answer to show — say so rather than pick one.
      if (m.target_range) {
        const [lo, hi] = m.target_range;
        const t = el("span", "play-tag", `→ ${fmt(lo)} … ${fmt(hi)}`);
        t.title = m.n_links > 1
          ? `All ${m.n_links} of this macro's links map 0…1 onto this range.`
          : "The range this macro's 0…1 maps onto, applied by parameter arithmetic";
        tags.appendChild(t);
      } else if (m.n_links > 1) {
        const t = el("span", "play-tag", `→ ${m.n_links} ranges`);
        t.title = "This macro drives several parameters, each over its own range — "
                + "open the editor's binding to see or change them.";
        tags.appendChild(t);
      }
      if (m.drives_weight)
        tags.appendChild(el("span", "play-tag " + (m.fast ? "fast" : "slow"),
          m.fast ? "weight" : "weight ⟳"));
      head.appendChild(tags);
      card.appendChild(head);

      const row = el("div", "play-macro-row");
      const valLabel = el("span", "play-macro-val");
      let trailing = null;

      if (m.param_type === "bool") {
        const cb = el("input");
        cb.type = "checkbox";
        cb.checked = !!m.value;
        cb.className = "play-macro-toggle";
        cb.addEventListener("change", () => setMacro(m, cb.checked ? 1 : 0));
        row.appendChild(cb);
        valLabel.textContent = cb.checked ? "true" : "false";
      } else {
        const hasRange = m.min != null && m.max != null;
        const slider = el("input");
        slider.type = "range";
        slider.className = "play-macro-slider";
        if (!hasRange && m._sliderMin == null) {
          m._sliderMin = m.value - Math.abs(m.value || 1) * 4 - 1;
          m._sliderMax = m.value + Math.abs(m.value || 1) * 4 + 1;
          if (m.param_type === "int") {
            m._sliderMin = Math.floor(m._sliderMin);
            m._sliderMax = Math.ceil(m._sliderMax);
          }
        }
        slider.min = hasRange ? m.min : m._sliderMin;
        slider.max = hasRange ? m.max : m._sliderMax;
        slider.step = m.param_type === "int" ? 1 : (slider.max - slider.min) / 1000 || 0.001;
        slider.value = m.value;
        const num = el("input");
        num.type = "number";
        num.className = "play-macro-num";
        num.value = m.value;
        if (m.param_type === "int") num.step = 1;
        // the value the target actually receives, shown alongside the macro's own
        const mapped = (v) => {
          if (!m.target_range) return "";
          const [lo, hi] = m.target_range;
          return fmt(lo + Number(v) * (hi - lo));
        };
        const mapLabel = el("span", "play-macro-mapped", mapped(m.value));
        const show = (v) => { valLabel.textContent = fmt(v); mapLabel.textContent = mapped(v); };
        slider.addEventListener("input", () => {
          num.value = slider.value;
          show(slider.value);
          setMacro(m, slider.value);
        });
        num.addEventListener("change", () => {
          slider.value = num.value;
          show(num.value);
          setMacro(m, num.value);
        });
        row.appendChild(slider);
        row.appendChild(num);
        valLabel.textContent = fmt(m.value);
        trailing = mapLabel;      // appended after valLabel, below
      }
      row.appendChild(valLabel);
      if (trailing) row.appendChild(trailing);
      card.appendChild(row);
      if (state.macroEdit) card.appendChild(renderMacroEdit(m));
      host.appendChild(card);
    });
  }

  function fmt(v) {
    const n = Number(v);
    return Number.isInteger(n) ? String(n) : n.toFixed(4);
  }

  // ── macros from the editor ───────────────────────────────────────────────────
  // The panel is built at compile time, but BendingParameters keep being created
  // next door in the graph editor. This re-reads them live — and the picker lists
  // the editor's whole catalogue, so one that did not come across on its own can
  // be pulled in by name.
  async function refreshMacros(body) {
    const data = body
      ? await postJSON("/api/play/macros/", body)
      : await (await fetch("/api/play/macros/")).json();
    if (data.error) throw new Error(data.error);
    const prev = {};
    state.macros.forEach((m) => { prev[m.name] = m; });
    state.macros = data.macros || [];
    // keep the slider ranges we derived for unbounded macros
    state.macros.forEach((m) => {
      const p = prev[m.name];
      if (p && m.min == null && p._sliderMin != null) {
        m._sliderMin = p._sliderMin;
        m._sliderMax = p._sliderMax;
      }
    });
    state.availParams = data.available || [];
    state.promotable = data.promotable || [];
    if (data.bindings) state.bindings = data.bindings;
    renderMacros();
    renderBendings();
    return data;
  }

  function _flashMacro(name) {
    const card = document.querySelector(`.play-macro[data-macro="${CSS.escape(name)}"]`);
    if (!card) return false;
    card.scrollIntoView({ block: "nearest", behavior: "smooth" });
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1200);
    return true;
  }

  // Two things can become a macro, and the picker groups them accordingly:
  // BendingParameters the editor already has, and bendable callback parameters
  // that no macro drives yet — those get promoted (a BendingParameter is created
  // and linked) on the way in, which is the only way they can be moved at all.
  function _syncMacroPicker() {
    const sel = $("play-add-macro");
    if (!sel) return;
    const shown = new Set(state.macros.map((m) => m.name));
    const total = state.availParams.length + state.promotable.length;
    sel.innerHTML = "";
    const head = el("option", null, total ? `＋ add macro (${total})` : "nothing to add");
    head.value = "";
    sel.appendChild(head);

    if (state.availParams.length) {
      const g = document.createElement("optgroup");
      g.label = "editor macros";
      state.availParams.forEach((a) => {
        const here = shown.has(a.name);
        const range = (a.min != null && a.max != null) ? ` [${a.min}…${a.max}]` : "";
        const o = el("option", null, `${here ? "✓ " : ""}${a.name} · ${a.param_type}${range}`);
        o.value = "have:" + a.name;
        g.appendChild(o);
      });
      sel.appendChild(g);
    }

    if (state.promotable.length) {
      const g = document.createElement("optgroup");
      g.label = "promote a bending parameter";
      state.promotable.forEach((pp) => {
        const o = el("option", null,
          `${pp.node}.${pp.label} · ${pp.callback_type} (${pp.param_type})`);
        o.value = `promote:${pp.binding}:${pp.param}`;
        o.title = `Create a BendingParameter for ${pp.callback_type}.${pp.param} on `
                + `'${pp.node}' and drive it from here`;
        g.appendChild(o);
      });
      sel.appendChild(g);
    }
    sel.disabled = !total;
  }

  async function addMacroFromEditor(name) {
    // already on screen: point at it rather than pretending to add it twice
    if (state.macros.some((m) => m.name === name)) {
      if (_flashMacro(name)) { toast(`'${name}' is already here`); return; }
    }
    try {
      await refreshMacros({ name });
      _flashMacro(name);
      toast(`Imported macro '${name}'`);
      if (state.runMode === "realtime") scheduleRun();
    } catch (e) {
      reportError(`Could not import '${name}'`, e);
    }
  }

  async function promoteToMacro(binding, param) {
    try {
      const data = await refreshMacros({ binding, param });
      const name = data.promoted;
      if (name) {
        _flashMacro(name);
        toast(`Promoted to macro '${name}'`);
      }
      // the binding now reads that parameter from a BendingParameter, so the
      // runtime the server invalidated has to be rebuilt before the next run
      await compile();
    } catch (e) {
      reportError("Could not promote that parameter", e);
    }
  }

  // ── bendings ─────────────────────────────────────────────────────────────────
  // A bending's parameters are playable exactly as they are. A macro is for
  // binding several of them together, or for giving one a named 0…1 control —
  // not a toll to pay before anything can move.

  const _paramDesc = (b, name) => ((b.descriptor || {}).params || {})[name] || {};

  function renderBendings() {
    const host = $("play-bendings");
    if (!host) return;
    host.innerHTML = "";
    $("play-bend-count").textContent = String(state.bindings.length);
    if (!state.bindings.length) {
      const empty = el("div", "play-empty");
      empty.innerHTML = "No bendings on this method.<br>Use <b>＋ bend</b> to attach one.";
      host.appendChild(empty);
      return;
    }
    state.bindings.forEach((b) => host.appendChild(renderBinding(b)));
  }

  function renderBinding(b) {
    const card = el("div", "play-bind");
    const head = el("div", "play-bind-head");
    head.appendChild(el("span", "play-bind-type", b.callback_type));
    head.appendChild(el("span", "play-bind-node", b.node));
    const del = el("button", "play-mini-btn danger", "×");
    del.title = "Remove this bending";
    del.addEventListener("click", () => removeBinding(b));
    head.appendChild(del);
    card.appendChild(head);

    const params = (b.descriptor || {}).params || {};
    let shown = 0;
    Object.keys(params).forEach((name) => {
      const pd = params[name];
      if (pd.visible === false) return;
      card.appendChild(renderBindParam(b, name, pd));
      shown += 1;
    });
    if (!shown) card.appendChild(el("div", "play-bind-none", "no adjustable parameters"));
    return card;
  }

  function renderBindParam(b, name, pd) {
    const row = el("div", "play-bind-param");
    row.dataset.binding = b.id;
    row.dataset.param = name;
    const macroName = (b.bp_links || {})[name];

    const lbl = el("span", "play-bind-label", pd.label || name);
    lbl.title = `${b.callback_type}.${name} on '${b.node}'`;
    row.appendChild(lbl);

    const value = b.params[name];
    if (macroName) {
      // driven by a macro: show which, and leave the driving to it
      const chip = el("span", "play-bind-macro", "⊕ " + macroName);
      chip.title = `Driven by macro '${macroName}' — move it above, or unlink to `
                 + `take this parameter back`;
      row.appendChild(chip);
      row.appendChild(el("span", "play-bind-val", fmt(value)));
      const unlink = el("button", "play-mini-btn", "⊘");
      unlink.title = "Unlink from the macro and drive this parameter directly";
      unlink.addEventListener("click", () => unlinkParam(b, name));
      row.appendChild(unlink);
    } else if (pd.type === "bool") {
      const cb = el("input");
      cb.type = "checkbox";
      cb.className = "play-macro-toggle";
      cb.checked = !!value;
      const val = el("span", "play-bind-val", cb.checked ? "true" : "false");
      cb.addEventListener("change", () => {
        val.textContent = cb.checked ? "true" : "false";
        setBindParam(b, name, cb.checked ? 1 : 0);
      });
      row.appendChild(cb);
      row.appendChild(val);
    } else {
      const isInt = pd.type === "int";
      const [rlo, rhi] = pd.range || [null, null];
      const basis = Math.max(Math.abs(Number(value) || 0), 1);
      const lo = rlo != null ? rlo : -basis * 4;
      const hi = rhi != null ? rhi : basis * 4;
      const slider = el("input");
      slider.type = "range";
      slider.className = "play-macro-slider";
      slider.min = lo; slider.max = hi;
      slider.step = isInt ? 1 : ((hi - lo) / 1000 || 0.001);
      slider.value = value;
      const num = el("input");
      num.type = "number";
      num.className = "play-macro-num";
      num.value = fmt(value);
      if (isInt) num.step = 1;
      const push = (v) => {
        num.value = fmt(v);
        setBindParam(b, name, isInt ? Math.round(Number(v)) : Number(v));
      };
      slider.addEventListener("input", () => push(slider.value));
      num.addEventListener("change", () => { slider.value = num.value; push(num.value); });
      row.appendChild(slider);
      row.appendChild(num);

      // only a plain float parameter can be joined: a macro is normalised, and
      // a callback cannot take the same macro twice
      if (pd.type === "float" || pd.type == null) _makeJoinable(row, b, name);
    }
    return row;
  }

  // ── drag a parameter ─────────────────────────────────────────────────────────
  // Two gestures, one drag:
  //   parameter → macro      attach it to that macro
  //   parameter → parameter  make a new macro driving both
  // The row itself is the drag handle (a grip alone is too small to find), with
  // the sliders guarded so grabbing one still moves it instead of starting a
  // drag.
  let _dragSrc = null;

  function _clearDropTargets() {
    document.querySelectorAll(".drop-target").forEach((r) => r.classList.remove("drop-target"));
  }

  function _makeJoinable(row, b, name) {
    const grip = el("span", "play-bind-grip", "⠿");
    grip.title = "Drag onto a macro to attach this parameter to it, "
               + "or onto another bending's parameter to make a macro for both";
    row.appendChild(grip);

    row.draggable = true;
    row.dataset.joinable = "1";
    // A pointer down on a slider must move the slider, not begin a drag. The
    // document-level pointerup in init() re-arms every row afterwards.
    row.querySelectorAll("input").forEach((inp) => {
      inp.addEventListener("pointerdown", () => { row.draggable = false; });
    });

    row.addEventListener("dragstart", (e) => {
      _dragSrc = { binding: b.id, param: name, label: `${b.node}.${name}` };
      row.classList.add("dragging");
      try { e.dataTransfer.setData("text/plain", _dragSrc.label); } catch (err) { /* ok */ }
      e.dataTransfer.effectAllowed = "link";
    });
    row.addEventListener("dragend", () => {
      row.classList.remove("dragging");
      _clearDropTargets();
      _dragSrc = null;
    });

    // …and a drop target for another parameter
    const accepts = () => _dragSrc && _dragSrc.binding !== b.id;
    row.addEventListener("dragover", (e) => {
      if (!accepts()) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = "link";
      row.classList.add("drop-target");
    });
    row.addEventListener("dragleave", () => row.classList.remove("drop-target"));
    row.addEventListener("drop", (e) => {
      row.classList.remove("drop-target");
      if (!accepts()) return;
      e.preventDefault();
      joinParams(_dragSrc, { binding: b.id, param: name, label: `${b.node}.${name}` });
    });
  }

  // a macro card accepts a parameter dropped on it: attach, don't create
  function _makeMacroDropTarget(card, m) {
    card.addEventListener("dragover", (e) => {
      if (!_dragSrc) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = "link";
      card.classList.add("drop-target");
    });
    card.addEventListener("dragleave", () => card.classList.remove("drop-target"));
    card.addEventListener("drop", (e) => {
      card.classList.remove("drop-target");
      if (!_dragSrc) return;
      e.preventDefault();
      attachToMacro(_dragSrc, m.name);
    });
  }

  // parameter → macro: the parameter joins what the macro is already doing, so
  // it takes the macro's current position rather than keeping its own value.
  async function attachToMacro(src, macroName) {
    try {
      const data = await postJSON(
        `/api/bending/${encodeURIComponent(src.binding)}/link/`,
        { param_name: src.param, bp_name: macroName });
      if (data.bindings) state.bindings = data.bindings;
      await refreshMacros();
      renderBendings();
      _flashMacro(macroName);
      toast(`${src.label} now follows '${macroName}'`);
      await compile();
    } catch (e) {
      reportError(`Could not attach ${src.label} to '${macroName}'`, e);
    }
  }

  async function joinParams(src, dst) {
    try {
      // src first: the server keeps the first target's value and moves the rest
      const data = await postJSON("/api/play/join/", {
        targets: [{ binding: src.binding, param: src.param },
                  { binding: dst.binding, param: dst.param }],
      });
      state.macros = data.macros || state.macros;
      state.availParams = data.available || state.availParams;
      state.promotable = data.promotable || state.promotable;
      state.bindings = data.bindings || state.bindings;
      renderMacros();
      renderBendings();
      _flashMacro(data.macro);
      toast(`${dst.label} now follows ${src.label} — macro '${data.macro}'`);
      await compile();     // the bending topology changed under the runtime
    } catch (e) {
      reportError("Could not join those parameters", e);
    }
  }

  // ── writing to a bending ─────────────────────────────────────────────────────
  const _bindTimers = {};
  function setBindParam(b, name, value) {
    b.params[name] = value;
    const key = b.id + "|" + name;
    clearTimeout(_bindTimers[key]);
    _bindTimers[key] = setTimeout(async () => {
      try {
        const data = await postJSON2(`/api/bending/${encodeURIComponent(b.id)}/`,
                                     { [name]: value }, "PATCH");
        if (data.bindings) state.bindings = data.bindings;
        if (state.runMode === "realtime") scheduleRun();
      } catch (e) {
        reportError(`Could not set ${b.callback_type}.${name}`, e);
      }
    }, 30);
  }

  async function postJSON2(url, body, method) {
    const r = await fetch(url, {
      method: method || "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw _httpError(data, r);
    return data;
  }

  async function unlinkParam(b, name) {
    try {
      const data = await postJSON(`/api/bending/${encodeURIComponent(b.id)}/unlink/`,
                                  { param_name: name });
      if (data.bindings) state.bindings = data.bindings;
      await refreshMacros();
      renderBendings();
      toast(`${b.node}.${name} is on its own again`);
      await compile();
    } catch (e) {
      reportError(`Could not unlink ${name}`, e);
    }
  }

  async function removeBinding(b) {
    if (!confirm(`Remove the ${b.callback_type} bending on '${b.node}'?`)) return;
    try {
      const data = await postJSON2(`/api/bending/${encodeURIComponent(b.id)}/`, {}, "DELETE");
      if (data.bindings) state.bindings = data.bindings;
      await compile();
      toast(`Removed ${b.callback_type} on '${b.node}'`);
    } catch (e) {
      reportError("Could not remove that bending", e);
    }
  }

  // ── add a bending ────────────────────────────────────────────────────────────
  // Attaching a bending used to mean going back to the editor, finding the node
  // in the graph and using its menu. Here it is a node and a callback type.
  async function openAddBend() {
    if (!state.callbacks) {
      try {
        const r = await fetch("/api/bending/callbacks/");
        state.callbacks = (await r.json()).callbacks || [];
      } catch (e) { reportError("Could not read the callback list", e); return; }
    }
    // The editor's activation browser, verbatim: same markup, same styles, same
    // search, same stars and saved searches. Picking a row here means "bend
    // this", so the callback choice rides in its footer.
    const browser = TBSearch.openBrowser({
      title: "＋ bend a node",
      nodes: state.bendableNodes,
      ctx: { aliases: state.aliases, tags: TBSearch.loadTags() },
      onPick: (n) => { chosen = n.name; syncGo(); },
      onClose: () => { if (foot.parentNode) foot.parentNode.removeChild(foot); },
    });
    if (!browser) { reportError("Node browser unavailable", new Error("no #act-modal")); return; }

    let chosen = null;
    const panel = document.getElementById("act-modal-panel");
    const foot = el("div", "act-modal-foot");
    const cbSel = el("select", "play-modal-select");
    state.callbacks.forEach((c) => {
      const name = c.name || c;
      const o = el("option", null, name);
      o.value = name;
      cbSel.appendChild(o);
    });
    const go = el("button", "play-run-btn", "Bend");
    const syncGo = () => {
      go.disabled = !chosen;
      go.textContent = chosen ? `Bend ${chosen}` : "Bend";
      go.title = chosen ? `Attach ${cbSel.value} to '${chosen}'` : "Pick a node first";
    };
    go.addEventListener("click", async () => {
      go.disabled = true;
      go.textContent = "bending…";
      const node = chosen, cb = cbSel.value;
      browser.close();
      try {
        await postJSON("/api/bending/", { fn: state.fn, node, callback_type: cb, params: {} });
        toast(`${cb} on '${node}'`);
        await compile();
      } catch (e) {
        reportError("Could not add that bending", e);
      }
    });
    cbSel.addEventListener("change", syncGo);
    foot.appendChild(el("span", "act-modal-foot-label", "callback"));
    foot.appendChild(cbSel);
    foot.appendChild(go);
    panel.appendChild(foot);
    syncGo();
  }

  // ── macro edit mode ──────────────────────────────────────────────────────────
  // A macro reads 0…1; each attachment decides what that spans. Edit mode is
  // where those ranges live, so the performing view stays uncluttered.
  async function renameMacro(m, newName) {
    const name = (newName || "").trim();
    if (!name || name === m.name) return false;
    try {
      await postJSON2(`/api/bending_params/${encodeURIComponent(m.name)}/`,
                      { name }, "PATCH");
      await refreshMacros();
      renderBendings();
      _flashMacro(name);
      toast(`Renamed to '${name}'`);
      await compile();     // the driven callbacks' generated forward changed
      return true;
    } catch (e) {
      reportError(`Could not rename '${m.name}'`, e);
      return false;
    }
  }

  async function deleteMacro(m) {
    if (!confirm(`Delete macro "${m.name}"? Everything it drives goes back to `
                 + `the value it is on now.`)) return;
    try {
      await postJSON2(`/api/bending_params/${encodeURIComponent(m.name)}/`, {}, "DELETE");
      await refreshMacros();
      renderBendings();
      toast(`Deleted '${m.name}'`);
      await compile();
    } catch (e) {
      reportError(`Could not delete '${m.name}'`, e);
    }
  }

  function renderMacroEdit(m) {
    const box = el("div", "play-macro-edit");

    // name row — a macro's name is how you find it again; it should be
    // changeable where you are performing, not only back in the editor
    const nameRow = el("div", "play-macro-edit-row");
    nameRow.appendChild(el("span", "play-macro-edit-arrow", "name"));
    const nameInp = el("input");
    nameInp.type = "text";
    nameInp.className = "play-macro-name-inp";
    nameInp.value = m.name;
    nameInp.spellcheck = false;
    nameInp.title = "Becomes an argument of the generated forward — letters, "
                  + "digits and underscores only";
    const commitName = async () => {
      if (!(await renameMacro(m, nameInp.value))) nameInp.value = m.name;
    };
    nameInp.addEventListener("change", commitName);
    nameInp.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); nameInp.blur(); }
      if (e.key === "Escape") { nameInp.value = m.name; nameInp.blur(); }
    });
    nameRow.appendChild(nameInp);
    const del = el("button", "play-mini-btn danger", "×");
    del.title = "Delete this macro";
    del.addEventListener("click", () => deleteMacro(m));
    nameRow.appendChild(del);
    box.appendChild(nameRow);

    if (!(m.links || []).length) {
      box.appendChild(el("div", "play-macro-edit-none",
        "not attached to anything — drag a bending parameter onto this card"));
      return box;
    }
    m.links.forEach((link) => {
      const row = el("div", "play-macro-edit-row");
      const where = el("span", "play-macro-edit-target", `${link.node}.${link.param}`);
      where.title = `${link.callback_type} on '${link.node}'`;
      row.appendChild(where);
      if (!link.range) {
        row.appendChild(el("span", "play-macro-edit-direct", "drives directly"));
        box.appendChild(row);
        return;
      }
      row.appendChild(el("span", "play-macro-edit-arrow", "0…1→"));
      const mk = (idx, title) => {
        const i = el("input");
        i.type = "text";
        i.className = "play-macro-edit-inp";
        i.value = link.range[idx];
        i.title = title;
        return i;
      };
      const loInp = mk(0, "value the macro sends at 0");
      const hiInp = mk(1, "value the macro sends at 1");
      const commit = async () => {
        const lo = parseFloat(loInp.value), hi = parseFloat(hiInp.value);
        if (!isFinite(lo) || !isFinite(hi) || lo === hi) {
          loInp.value = link.range[0]; hiInp.value = link.range[1];
          return;
        }
        try {
          const data = await postJSON(
            `/api/bending/${encodeURIComponent(link.binding)}/link/`,
            { param_name: link.param, bp_name: m.name, range_min: lo, range_max: hi });
          if (data.bindings) state.bindings = data.bindings;
          await refreshMacros();
          if (state.runMode === "realtime") scheduleRun();
        } catch (e) {
          reportError(`Could not re-map ${link.node}.${link.param}`, e);
        }
      };
      loInp.addEventListener("change", commit);
      hiInp.addEventListener("change", commit);
      row.appendChild(loInp);
      row.appendChild(hiInp);
      box.appendChild(row);
    });
    return box;
  }

  function toggleMacroEdit() {
    state.macroEdit = !state.macroEdit;
    $("play-macro-edit").classList.toggle("active", state.macroEdit);
    renderMacros();
  }

  // ── derived values ───────────────────────────────────────────────────────────
  // A macro reads 0…1 and every attachment maps that onto its own range, so a
  // single macro is worth a different number at each target. This panel is all
  // of them at once, and it follows a drag without a round trip: the value is
  // just lo + v·(hi − lo), the same arithmetic the server applies.
  function _derivedValue(m, link) {
    if (!link.range) return m.value;                 // drives its target directly
    const [lo, hi] = link.range;
    return lo + Number(m.value) * (hi - lo);
  }

  function renderDerived() {
    const panel = $("play-derived");
    const body = $("play-derived-body");
    if (!panel || !body) return;
    const withLinks = state.macros.filter((m) => (m.links || []).length);
    const total = withLinks.reduce((n, m) => n + m.links.length, 0);
    panel.style.display = total ? "" : "none";
    $("play-derived-count").textContent = String(total);
    $("play-derived-caret").textContent = state.derivedOpen ? "▾" : "▸";
    body.style.display = state.derivedOpen ? "" : "none";
    if (!total || !state.derivedOpen) { body.innerHTML = ""; return; }

    body.innerHTML = "";
    withLinks.forEach((m) => {
      const group = el("div", "play-derived-group");
      group.appendChild(el("div", "play-derived-macro", m.name));
      m.links.forEach((link) => {
        const row = el("div", "play-derived-row");
        row.dataset.macro = m.name;
        row.dataset.link = link.binding + "|" + link.param;
        const where = el("span", "play-derived-target", `${link.node}.${link.param}`);
        where.title = `${link.callback_type} on '${link.node}'`;
        row.appendChild(where);
        row.appendChild(el("span", "play-derived-range",
          link.range ? `${fmt(link.range[0])}…${fmt(link.range[1])}` : "direct"));
        row.appendChild(el("span", "play-derived-value", fmt(_derivedValue(m, link))));
        group.appendChild(row);
      });
      body.appendChild(group);
    });
  }

  // follow a macro while it is being dragged, without re-rendering the panel
  function updateDerived(m) {
    if (!state.derivedOpen) return;
    (m.links || []).forEach((link) => {
      const row = document.querySelector(
        `.play-derived-row[data-macro="${CSS.escape(m.name)}"]`
        + `[data-link="${CSS.escape(link.binding + "|" + link.param)}"]`);
      if (!row) return;
      const cell = row.querySelector(".play-derived-value");
      if (cell) cell.textContent = fmt(_derivedValue(m, link));
    });
  }

  function toggleDerived() {
    state.derivedOpen = !state.derivedOpen;
    localStorage.setItem("tb_play_derived", state.derivedOpen ? "1" : "0");
    renderDerived();
  }

  let _macroTimer = null;
  function setMacro(m, value) {
    m.value = value;
    updateDerived(m);
    clearTimeout(_macroTimer);
    _macroTimer = setTimeout(async () => {
      try {
        await postJSON("/api/play/macro/", { name: m.name, value: Number(value) });
        if (state.runMode === "realtime") scheduleRun();
      } catch (e) {
        reportError(`Macro '${m.name}' failed`, e);
      }
    }, 30);
  }

  // ── input bench ──────────────────────────────────────────────────────────────
  // The bench starts empty: nothing is guessed, nothing is drawn for you. Every
  // input is added explicitly — imported from the editor's bench, dropped as a
  // file, or written as an expression — so what the model runs on is always
  // something you put there.
  //
  // Optional arguments (kwargs with a default in the traced signature) are kept
  // out of the way entirely: leaving one out is not "missing input", it is the
  // model running on the very default it was traced with. Add one from the
  // header picker to get a widget matching its type.
  const SCALAR_TYPES = new Set(["bool", "int", "float", "str"]);

  const _isScalar = (p) => SCALAR_TYPES.has(p.arg_type);
  const _phShape = (name) => {
    const p = state.placeholders.find((x) => x.name === name);
    return p ? p.shape : null;
  };

  // ── persistence ──────────────────────────────────────────────────────────────
  // The bench is a setup, not a scratch pad: leaving for the editor and coming
  // back must not throw it away. Mirror it to localStorage per model and method,
  // the way the editor mirrors its own, and restore on compile.
  const PLAY_STORE_PREFIX = "tb_play_inputs_";
  const _MAX_PERSIST_BYTES = 4 * 1024 * 1024;      // 4 MB / file, as in the editor
  // widget settings worth carrying across: a restored file keeps its crop, its
  // target size, its sample rate — re-deriving them would undo the user's work
  const _MEDIA_KEYS = [
    "imgTargetW", "imgTargetH", "imgChannels", "imgMode",
    "cropX", "cropY", "cropW", "cropH",
    "cropStart", "cropEnd", "targetSR", "normalize", "toMono", "monoByShape",
  ];

  function _playStoreKey() { return PLAY_STORE_PREFIX + (window.PLAY_MODEL || ""); }

  function _fileToDataUrl(f) {
    return new Promise((res, rej) => {
      const r = new FileReader();
      r.onload = () => res(r.result);
      r.onerror = rej;
      r.readAsDataURL(f);
    });
  }

  async function _serializeEntry(e) {
    // mode entries are not carried across play sessions (they never were); one
    // holding a loaded file must not fall through to the file branch below and
    // come back as a plain waveform for a placeholder that expects an encoding
    if (e.kind === "mode") return null;
    if (e.kind === "scalar") return { type: "scalar", value: e.value };
    if (e.kind === "expr")
      return (e.value && e.value.trim()) ? { type: "expr", value: e.value } : null;
    const f = e.file;
    if (!f) return null;
    const meta = {};
    _MEDIA_KEYS.forEach((k) => { if (e[k] !== undefined) meta[k] = e[k]; });
    // the *source* file is stored, not the processed one: the settings above
    // reproduce the processing, and storing both would double the quota cost
    if (f.size > _MAX_PERSIST_BYTES)
      return { type: "file", label: e.label, tooBig: true, meta };
    try {
      return { type: "file", label: e.label || f.name, mime: f.type,
               dataUrl: await _fileToDataUrl(f), meta };
    } catch (err) {
      return { type: "file", label: e.label, tooBig: true, meta };
    }
  }

  let _persistTimer = null;
  function persistInputs() {
    clearTimeout(_persistTimer);
    _persistTimer = setTimeout(_doPersistInputs, 250);
  }

  async function _doPersistInputs() {
    const files = [];
    Object.values(state.inputs).forEach((arr) => (arr || []).forEach((e) => { if (e.file) files.push(e.file); }));
    await Promise.all(files.map((f) => TBBench.ensureDataUrl(f)));
    _savePlayBenchNow();
  }

  // entry as play holds it ⇄ the bench's neutral form (bench.js)
  function _playToNeutral(e) {
    const meta = {};
    TBBench.MEDIA_KEYS.forEach((k) => { if (e[k] !== undefined) meta[k] = e[k]; });
    return { id: e.id, type: e.kind, value: e.value, label: e.label, mode: e.mode || null,
             inBatch: e.inBatch, meta, file: e.file || null };
  }

  async function _playFromNeutral(n, p) {
    let kind = n.type, value = n.value;
    // the editor has no scalar widget: its expression for a plain argument
    // comes back as that argument's value
    if (_isScalar(p) && kind === "expr") {
      kind = "scalar";
      const t = p.arg_type;
      const raw = String(value).trim();
      value = t === "bool" ? /^(1|true|yes|on)$/i.test(raw)
            : (t === "int" || t === "float") ? Number(raw)
            : raw;
    }
    if (!_isScalar(p) && kind === "scalar") { kind = "expr"; value = String(value); }
    const e = { id: n.id, kind, value, label: n.label, valid: true };
    if (n.inBatch === false) e.inBatch = false;
    if (kind === "mode") e.mode = n.mode;
    if (n.file) {
      e.file = n.file;
      if (!e.label) e.label = n.file.name;
    }
    Object.assign(e, n.meta || {});
    // re-derive the processed file (crop, resample…) before the first run
    if (kind === "file" && e.file) {
      try {
        if (TBInputs.isImageFile(e.file) && window.TBImage) {
          await TBImage.init(e, p.shape); await TBImage.process(e);
        } else if (window.TBAudio && TBAudio.isAudioFile(e.file)) {
          await TBAudio.init(e, p.shape); await TBAudio.process(e);
        }
      } catch (_) { /* the raw file still works */ }
    }
    return e;
  }

  function _savePlayBenchNow() {
    const sets = {}, selected = {};
    for (const p of state.placeholders) {
      sets[p.name] = (state.inputs[p.name] || []).map((e) => TBBench.toSaved(_playToNeutral(e)));
      if (state.selected[p.name]) selected[p.name] = state.selected[p.name];
    }
    const ok = TBBench.save(window.PLAY_MODEL, {
      sets, selected,
      batch: { on: state.batchOn, mode: state.batchMode },
      optional: [...state.optionalOn],
    }, state.placeholders.map((p) => p.name));
    if (ok) { _persistWarned = false; return; }
    // The bench still works — it just will not survive leaving the page, and
    // finding that out on the way back is worse than hearing it now.
    if (!_persistWarned) {
      _persistWarned = true;
      toast("Inputs are too large to remember between visits — "
            + "they will need reloading if you leave this page", true);
    }
  }
  let _persistWarned = false;

  function _loadPlayInputs() {
    try {
      const raw = localStorage.getItem(_playStoreKey());
      const store = raw ? JSON.parse(raw) : null;
      return (store && store.byFn && store.byFn[state.fn]) || null;
    } catch (err) { return null; }
  }

  async function _entryFromSaved(saved, p) {
    if (saved.type === "scalar") return { id: genId(), kind: "scalar", value: saved.value };
    if (saved.type === "expr" && saved.value && saved.value.trim())
      return { id: genId(), kind: "expr", value: saved.value, valid: true };
    if (saved.type !== "file" || !saved.dataUrl) return null;   // tooBig: reload it here
    try {
      const file = await _dataUrlToFile(saved.dataUrl, saved.label, saved.mime);
      const e = { id: genId(), kind: "file", file, label: saved.label || file.name, valid: true };
      Object.assign(e, saved.meta || {});
      // Re-derive the processed file straight away. The widget would do it when
      // it renders, but the first run is scheduled the moment compile lands —
      // without this it would go out with the raw, uncropped file.
      if (TBInputs.isImageFile(file) && window.TBImage) {
        await TBImage.init(e, p.shape); await TBImage.process(e);
      } else if (window.TBAudio && TBAudio.isAudioFile(file)) {
        await TBAudio.init(e, p.shape); await TBAudio.process(e);
      }
      return e;
    } catch (err) { return null; }
  }

  // Inputs shared by the editor (graph mode), persisted to localStorage.
  function _loadSharedInputs() {
    try {
      const raw = localStorage.getItem("tb_inputs_" + (window.PLAY_MODEL || ""));
      return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
  }

  function _sharedFor(name) {
    const arr = state.shared && state.shared.sets && state.shared.sets[name];
    return (arr && arr.length) ? arr : null;
  }

  async function _dataUrlToFile(dataUrl, name, mime) {
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    return new File([blob], name || "input", { type: mime || blob.type });
  }

  // Build play-mode entries for a placeholder from the editor's stored entries.
  async function _entriesFromShared(savedArr) {
    const out = [];
    for (const s of savedArr) {
      if (s.type === "expr") {
        // `expr` is the editor's own key — tolerate it in case an older/foreign
        // payload made it into the store; drop entries with nothing in them.
        const value = (s.value != null ? s.value : s.expr) || "";
        if (!value.trim()) continue;
        out.push({ id: genId(), kind: "expr", value, label: s.label, valid: true });
      } else if (s.type === "file" && s.dataUrl) {
        try {
          const file = await _dataUrlToFile(s.dataUrl, s.label, s.mime);
          out.push({ id: genId(), kind: "file", file, label: s.label || file.name, valid: true });
        } catch (e) { /* skip unreadable file */ }
      }
      // s.tooBig files are intentionally skipped (re-load them in play mode)
    }
    return out;
  }

  async function importFromEditor(name) {
    const saved = _sharedFor(name);
    if (!saved) { toast(`Nothing stored for '${name}' in the editor`); return; }
    const entries = await _entriesFromShared(saved);
    if (!entries.length) {
      toast(`The editor's entries for '${name}' could not be imported ` +
            `(files above 4 MB are not carried over)`, true);
      return;
    }
    state.inputs[name] = (state.inputs[name] || []).concat(entries);
    state.adderOpen.delete(name);
    renderInputs();
    inputChanged();
  }

  async function importAllFromEditor() {
    let n = 0;
    for (const p of state.placeholders) {
      if (_isScalar(p)) continue;                 // a scalar has its own widget
      const saved = _sharedFor(p.name);
      if (!saved) continue;
      const entries = await _entriesFromShared(saved);
      if (!entries.length) continue;
      if (p.optional) state.optionalOn.add(p.name);
      state.inputs[p.name] = (state.inputs[p.name] || []).concat(entries);
      n += entries.length;
    }
    renderInputs();
    if (n) { toast(`Imported ${n} input${n > 1 ? "s" : ""} from the editor`); inputChanged(); }
    else toast("The editor's bench has nothing to import");
  }

  // ── entry / readiness helpers ────────────────────────────────────────────────
  function _entryHasValue(e) {
    if (e.kind === "file") return !!(e.croppedFile || e.file);
    if (e.kind === "mode" && e.file) return true;    // a loaded recording
    if (e.kind === "scalar") return e.value != null && String(e.value).trim() !== "";
    return !!(e.value && e.value.trim());
  }

  function _hasValue(name) {
    const filler = _inputModeClaimed()[name];
    // generate mode sends the checked entries only: an input whose entries are
    // all unchecked has nothing to give, whatever their kind
    if (PAGE_MODE === "generate") {
      const has = (n) => _runEntries(n).some((e) => e.kind === "mode"
        ? !!(e.file || (e.value || "").trim()) : _entryHasValue(e));
      return has(name) || (!!filler && has(filler));
    }
    // A prompt in an input mode is a value, and so is one that another
    // placeholder's mode fills on its behalf.
    if (_hasInputModeValue(name)) return true;
    if (filler && _hasInputModeValue(filler)) return true;
    return (state.inputs[name] || []).some(_entryHasValue);
  }

  // Required placeholders still waiting for a value — the run is held until
  // every one of them has one.
  function _missingRequired() {
    // A callback brings its own arguments; the bench feeds the compiled graph,
    // which is not what Generate is pointed at right now.
    if (state.runTarget) return [];
    return state.placeholders.filter((p) => !p.optional && !_hasValue(p.name))
                             .map((p) => p.name);
  }

  function _visiblePlaceholders() {
    // A declared input mode is the interface saying "this is the way in", so the
    // placeholder is offered even when the signature calls it optional —
    // GPT-2 marks every argument of `forward` optional, and the bench would
    // otherwise start empty with the prompt nowhere to be found. Whatever an
    // active mode fills comes along too, shown as driven rather than editable.
    const claimed = _inputModeClaimed();
    return state.placeholders.filter((p) =>
      !p.optional || state.optionalOn.has(p.name)
      || _inputModeFor(p.name) || claimed[p.name]);
  }

  async function pruneInputsToPlaceholders() {
    const names = new Set(state.placeholders.map((p) => p.name));
    Object.keys(state.inputs).forEach((k) => { if (!names.has(k)) delete state.inputs[k]; });
    [...state.optionalOn].forEach((n) => { if (!names.has(n)) state.optionalOn.delete(n); });
    [...state.adderOpen].forEach((n) => { if (!names.has(n)) state.adderOpen.delete(n); });

    // Fill the gaps from the last visit. Only placeholders with nothing on them
    // are touched, so this restores after a trip to the editor or a method
    // switch without ever overwriting what is already on the bench.
    const saved = TBBench.load(window.PLAY_MODEL);
    if (saved) {
      (saved.optional || []).forEach((n) => { if (names.has(n)) state.optionalOn.add(n); });
      let lost = 0;
      for (const p of state.placeholders) {
        if ((state.inputs[p.name] || []).length) continue;
        const arr = (saved.sets || {})[p.name];
        if (!arr || !arr.length) continue;
        const entries = [];
        for (const sv of arr) {
          const n = await TBBench.fromSaved(sv);
          if (n.lost) { lost += 1; continue; }
          entries.push(await _playFromNeutral(n, p));
        }
        if (entries.length) state.inputs[p.name] = entries;
        const sel = (saved.selected || {})[p.name];
        if (sel && entries.some((e) => e.id === sel)) state.selected[p.name] = sel;
      }
      // a file too big to store is not silently forgotten — say it needs reloading
      if (lost) toast(`${lost} file input${lost > 1 ? "s" : ""} could not be restored `
                      + `(too large to keep) — load ${lost > 1 ? "them" : "it"} again`);
    }
    state.placeholders.forEach((p) => { if (!state.inputs[p.name]) state.inputs[p.name] = []; });
  }

  // ── rendering ────────────────────────────────────────────────────────────────
  function renderInputs() {
    const host = $("play-inputs");
    host.innerHTML = "";
    _syncOptionalPicker();
    _syncImportBtn();
    _syncBatchUI();

    persistInputs();       // every structural change lands here
    const visible = _visiblePlaceholders();
    $("play-input-count").textContent = String(visible.length);
    if (!visible.length) {
      const empty = el("div", "play-empty");
      empty.innerHTML = state.placeholders.length
        ? "Every argument of this method is optional.<br>Add one from <b>＋ optional input</b> above, or run on the traced defaults."
        : "This method takes no inputs.";
      host.appendChild(empty);
    } else {
      visible.forEach((p) => host.appendChild(renderInputCard(p)));
    }
    _syncRunUI();
  }

  function _defaultLabel(p) {
    if (p.arg_default === undefined) return "";
    // the value comes from a Python signature, so name a null the way Python does
    return p.arg_default === null ? "None" : String(p.arg_default);
  }

  // ── input modes ──────────────────────────────────────────────────────────────
  // The interface's own way into a placeholder — GPT-2's `input_ids` as a
  // prompt, tokenized server-side. One prompt decides the sequence length, so
  // the mask and the positions come from the same call rather than from
  // whatever the bench happens to hold for them.
  // Which way in a placeholder is showing. Only a display preference — the
  // prompts themselves are entries in `state.inputs`, exactly like expressions
  // and files, so they get the same list, the same add button and the same
  // trip through the run.
  const _inputModeShown = {};   // placeholder -> mode name, or "" for expr

  function _isIntDtype(p) {
    return /^(int|uint|long|short)/.test(p.dtype || "");
  }

  function _exprSuggestion(p) {
    const dims = (p.shape || []).join(", ");
    if (_isIntDtype(p)) return `torch.randint(0, 10, (${dims}))`;
    if (p.dtype === "bool") return `torch.ones(${dims}, dtype=torch.bool)`;
    return `torch.randn(${dims})`;
  }

  function _inputModeFor(name) {
    return (state.inputModes || {})[name] || null;
  }

  // Same reading as the editor's: dtype and the declared mode decide before
  // the shape gets a say, so token ids do not come up as audio.
  function _placeholderKind(p) {
    return TBInputs.guessType(p.shape, {
      dtype: p.dtype, name: p.name, hasMode: !!_inputModeFor(p.name),
    });
  }

  function _modeEntries(name) {
    return (state.inputs[name] || []).filter((e) => e.kind === "mode");
  }

  function _inputModeShownFor(p) {
    const spec = _inputModeFor(p.name);
    if (!spec) return "";
    if (_inputModeShown[p.name] === undefined) {
      const arr = state.inputs[p.name] || [];
      _inputModeShown[p.name] = arr.length && arr.every((e) => e.kind === "mode")
        ? spec.type : "";
    }
    return _inputModeShown[p.name];
  }

  function _inputModeClaimed() {
    const claimed = {};
    Object.entries(state.inputModes || {}).forEach(([name, spec]) => {
      if (!_modeEntries(name).length) return;
      (spec.fills || [name]).forEach((f) => { if (f !== name) claimed[f] = name; });
    });
    return claimed;
  }

  function _hasInputModeValue(name) {
    return _modeEntries(name).some((e) => e.file || (e.value || "").trim());
  }

  function addModeEntry(p, spec, text) {
    state.inputs[p.name] = state.inputs[p.name] || [];
    state.inputs[p.name].push({ id: genId(), kind: "mode", mode: spec.type,
                                value: text,
                                label: text.length > 28 ? text.slice(0, 28) + "…" : text,
                                valid: true });
    state.adderOpen.delete(p.name);
    renderInputs();
    inputChanged();
  }

  function _renderInputModeBar(p) {
    const spec = _inputModeFor(p.name);
    if (!spec) return null;
    const shown = _inputModeShownFor(p);

    const bar = el("div", "play-input-mode-bar");
    [["expr", ""], [spec.label || spec.type, spec.type]].forEach(([label, mode]) => {
      const btn = el("button", "play-input-mode-btn" + (shown === mode ? " active" : ""), label);
      btn.title = mode
        ? `Feed ${p.name} as ${spec.type} — the interface encodes it`
          + ((spec.fills || []).length > 1 ? `, filling ${spec.fills.join(", ")}` : "")
        : "Feed it as an expression or a file";
      btn.addEventListener("click", () => {
        _inputModeShown[p.name] = mode;
        renderInputs();
      });
      bar.appendChild(btn);
    });
    return bar;
  }

  function _renderModeAdder(p, spec) {
    const wrap = el("div", "play-input-mode");
    if ((spec.fills || []).length > 1) {
      wrap.appendChild(el("div", "play-input-mode-note",
        "also fills " + spec.fills.filter((f) => f !== p.name).join(", ")));
    }
    const row = el("div", "play-input-entry");
    const ta = el("textarea", "play-input-mode-text");
    ta.placeholder = spec.placeholder || "";
    if (!_modeEntries(p.name).length) ta.value = spec.default || "";
    const add = el("button", "play-mini-btn", "+ add");
    add.title = "Add this prompt as an entry (Enter)";
    const commit = () => {
      const v = ta.value.trim();
      if (!v) return;
      addModeEntry(p, spec, v);
    };
    add.addEventListener("click", commit);
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); commit(); }
    });
    row.appendChild(ta);
    row.appendChild(add);
    wrap.appendChild(row);
    // an audio mode also takes a recording loaded from this computer (see the
    // editor's _buildModeAdder)
    if (spec.type === "audio") {
      const fileIn = document.createElement("input");
      fileIn.type = "file";
      fileIn.accept = "audio/*,.wav,.flac,.ogg,.aiff,.aif,.mp3";
      fileIn.style.display = "none";
      const pick = el("button", "play-mini-btn", "\u{1F4C1} load file…");
      pick.title = "Load a recording from this computer";
      pick.addEventListener("click", () => fileIn.click());
      fileIn.addEventListener("change", () => {
        const f = fileIn.files && fileIn.files[0];
        if (!f) return;
        state.inputs[p.name] = state.inputs[p.name] || [];
        state.inputs[p.name].push({ id: genId(), kind: "mode", mode: spec.type,
                                    value: "", file: f, label: f.name, valid: true });
        state.adderOpen.delete(p.name);
        fileIn.value = "";
        renderInputs();
        inputChanged();
      });
      wrap.appendChild(pick);
      wrap.appendChild(fileIn);
    }
    return wrap;
  }

  function renderInputCard(p) {
    const card = el("div", "play-input" + (p.optional ? " play-input-optional" : ""));

    const head = el("div", "play-input-head");
    head.appendChild(el("span", "play-input-name", p.name));
    const kind = _isScalar(p) ? p.arg_type : _placeholderKind(p);
    head.appendChild(el("span", "play-input-kind kind-" + kind, kind));
    if (p.shape) head.appendChild(el("span", "play-input-shape", "[" + p.shape.join(", ") + "]"));
    if (p.optional) {
      const tag = el("span", "play-input-opt-tag", "optional");
      tag.title = `Defaults to ${_defaultLabel(p)} — remove this input to run on that default`;
      head.appendChild(tag);
    }
    if (p.used === false) {
      const tag = el("span", "play-input-baked-tag", "baked");
      tag.title = "This argument has no users left in the traced graph — its value "
                + "was folded in at trace time (control flow, or an ATen-level "
                + "trace). Changing it here does nothing until the method is "
                + "re-traced on the value you want.";
      head.appendChild(tag);
    }
    const _claimedBy = _inputModeClaimed()[p.name];
    if (_claimedBy) {
      card.classList.add("play-input-claimed");
      card.appendChild(head);
      card.appendChild(el("div", "play-input-mode-note", `filled from ${_claimedBy}`));
      return card;
    }
    if (false) {   // one bench with the editor now: nothing to import
      const imp = el("button", "play-mini-btn", "⇩ editor");
      imp.title = "Import this input from the graph editor's bench";
      imp.addEventListener("click", () => importFromEditor(p.name));
      head.appendChild(imp);
    }
    if (p.optional) {
      const off = el("button", "play-mini-btn danger", "×");
      off.title = "Drop this input — the model falls back to its default";
      off.addEventListener("click", () => {
        state.optionalOn.delete(p.name);
        state.inputs[p.name] = [];
        renderInputs();
        inputChanged();
      });
      head.appendChild(off);
    }
    card.appendChild(head);

    const _modeBar = _renderInputModeBar(p);
    if (_modeBar) card.appendChild(_modeBar);

    if (_isScalar(p)) {
      card.appendChild(renderScalarBody(p));
      return card;
    }

    const entries = state.inputs[p.name] || [];
    entries.forEach((entry) => card.appendChild(renderInputEntry(p, entry)));

    // The add block is three controls tall — worth the room on an empty card,
    // pure clutter on one that already has what it needs. Once there is an
    // entry it folds into a single button, which is also where "+ batch item"
    // lives now.
    if (!entries.length || state.adderOpen.has(p.name)) {
      // whichever way in is on show gets the adder; the entries above are the
      // same list either way, so prompts and expressions sit together
      const shownMode = _inputModeShownFor(p);
      card.appendChild(shownMode ? _renderModeAdder(p, _inputModeFor(p.name))
                                 : renderAdder(p));
      if (entries.length) {
        const hide = el("button", "play-mini-btn play-adder-toggle", "− done");
        hide.addEventListener("click", () => {
          state.adderOpen.delete(p.name);
          renderInputs();
        });
        card.appendChild(hide);
      }
    } else {
      const show = el("button", "play-mini-btn play-adder-toggle", "＋ add entry");
      show.title = "Add another entry — entries are stacked along the batch dimension";
      show.addEventListener("click", () => {
        state.adderOpen.add(p.name);
        renderInputs();
      });
      card.appendChild(show);
    }
    return card;
  }

  function _scalarSeed(p) {
    if (p.arg_default !== undefined && p.arg_default !== null) return p.arg_default;
    if (p.arg_type === "bool") return false;
    if (p.arg_type === "str") return "";
    return 0;
  }

  // A scalar argument (a temperature, a flag, a length) gets the widget its type
  // asks for rather than a tensor expression — the server passes it through as a
  // plain Python value.
  function renderScalarBody(p) {
    const arr = state.inputs[p.name];
    if (!arr.length) arr.push({ id: genId(), kind: "scalar", value: _scalarSeed(p) });
    const entry = arr[0];
    const row = el("div", "play-input-entry");

    if (p.arg_type === "bool") {
      const cb = el("input");
      cb.type = "checkbox";
      cb.className = "play-macro-toggle";
      cb.checked = !!entry.value && entry.value !== "false";
      const lbl = el("span", "play-macro-val", cb.checked ? "true" : "false");
      cb.addEventListener("change", () => {
        entry.value = cb.checked;
        lbl.textContent = cb.checked ? "true" : "false";
        inputChanged();
      });
      row.appendChild(cb);
      row.appendChild(lbl);
    } else if (p.arg_type === "str") {
      const ti = el("input");
      ti.type = "text";
      ti.className = "play-input-expr";
      ti.value = entry.value == null ? "" : entry.value;
      ti.spellcheck = false;
      ti.addEventListener("input", () => { entry.value = ti.value; inputChanged(); });
      row.appendChild(ti);
    } else {
      const num = el("input");
      num.type = "number";
      num.className = "play-macro-num";
      if (p.arg_type === "int") num.step = 1;
      num.value = entry.value;
      num.addEventListener("input", () => { entry.value = num.value; inputChanged(); });
      row.appendChild(num);
    }

    if (p.arg_default !== undefined && p.arg_default !== null) {
      const rst = el("button", "play-mini-btn", "default");
      rst.title = `Back to ${_defaultLabel(p)}`;
      rst.addEventListener("click", () => {
        entry.value = p.arg_default;
        renderInputs();
        inputChanged();
      });
      row.appendChild(rst);
    }
    return row;
  }

  // The add block: the same choice the editor's input panel offers — a media
  // drop zone typed by the placeholder's shape, an expression, or a quick fill.
  function renderAdder(p) {
    const kind = _placeholderKind(p);
    const wrap = el("div", "play-input-adder");

    const zone = el("div", "upload-area play-drop");
    const fileInput = el("input");
    fileInput.type = "file";
    fileInput.className = "upload-file-input";
    if (kind === "image") fileInput.accept = "image/*";
    else if (kind === "audio") fileInput.accept = "audio/*";
    const hint = el("div", "upload-hint");
    hint.appendChild(el("span", "upload-icon",
      kind === "image" ? "🖼" : kind === "audio" ? "♪" : "＋"));
    hint.appendChild(el("span", null,
      kind === "image" ? "drop image or click to add"
        : kind === "audio" ? "drop audio or click to add"
          : "drop a file or click to add"));
    zone.appendChild(hint);
    zone.appendChild(fileInput);
    zone.addEventListener("click", () => fileInput.click());
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag-over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("drag-over");
      if (e.dataTransfer.files[0]) addFileEntry(p, e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files[0]) { addFileEntry(p, fileInput.files[0]); fileInput.value = ""; }
    });
    wrap.appendChild(zone);

    wrap.appendChild(el("div", "input-or-sep", "or expression"));

    const row = el("div", "play-input-entry");
    const ta = el("textarea", "play-input-expr");
    ta.rows = 1;
    ta.spellcheck = false;
    ta.placeholder = p.shape && p.shape.length
      ? _exprSuggestion(p)
      : "torch expression, e.g. torch.randn(1, 3, 64, 64)";
    const add = el("button", "play-mini-btn", "+ add");
    add.title = "Add this expression as an entry (Enter)";
    const commitExpr = () => {
      const v = ta.value.trim();
      if (!v) return;
      addExprEntry(p, v);
      ta.value = "";
    };
    add.addEventListener("click", commitExpr);
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); commitExpr(); }
    });
    row.appendChild(ta);
    row.appendChild(add);
    wrap.appendChild(row);

    const chips = el("div", "play-input-chips");
    const chip = (text, title, expr) => {
      const b = el("button", "play-mini-btn", text);
      b.title = title;
      b.addEventListener("click", () => addExprEntry(p, expr));
      chips.appendChild(b);
    };
    if (p.shape && p.shape.length) {
      const dims = p.shape.join(", ");
      // an integer input takes randint, not randn: offering randn there is
      // offering something that fails in the embedding lookup
      if (_isIntDtype(p)) {
        chip("randint", `${_exprSuggestion(p)} — integers of the traced shape`,
             _exprSuggestion(p));
      } else {
        chip("randn", `torch.randn(${dims}) — noise of the traced shape`, `torch.randn(${dims})`);
      }
      chip("zeros", `torch.zeros(${dims})`, `torch.zeros(${dims})`);
    }
    if (p.default != null && String(p.default).trim())
      chip("traced", "The value this method was traced with", String(p.default));
    if (chips.children.length) wrap.appendChild(chips);

    return wrap;
  }

  function addExprEntry(p, expr) {
    state.inputs[p.name].push({ id: genId(), kind: "expr", value: expr, valid: true });
    state.adderOpen.delete(p.name);     // one ask, one entry — fold it back up
    renderInputs();
    inputChanged();
  }

  function addFileEntry(p, file) {
    state.inputs[p.name].push({ id: genId(), kind: "file", file, label: file.name, valid: true });
    state.adderOpen.delete(p.name);
    renderInputs();
    inputChanged();
  }

  function renderInputEntry(p, entry) {
    const name = p.name;
    const col = el("div", "play-input-col");
    const row = el("div", "play-input-entry");
    const modeSpec = entry.kind === "mode" ? _inputModeFor(name) : null;
    if (entry.kind === "file") {
      row.appendChild(el("span", "play-file-chip", "📄 " + entry.label));
    } else if (entry.kind === "mode" && entry.file) {
      // a recording loaded for an audio mode: the interface encodes the file
      row.appendChild(el("span", "play-file-chip", "\u{1F3B5} " + entry.label));
    } else if (entry.kind === "mode") {
      // typed text for a mode (a prompt, a path): the mode's own hint, not an expression's
      const ta = el("textarea", "play-input-expr");
      ta.rows = 1;
      ta.value = entry.value;
      ta.spellcheck = false;
      ta.placeholder = (modeSpec && modeSpec.placeholder) || "";
      ta.addEventListener("input", () => {
        entry.value = ta.value;
        entry.label = ta.value.length > 28 ? ta.value.slice(0, 28) + "…" : ta.value;
        inputChanged();
      });
      row.appendChild(ta);
    } else {
      const ta = el("textarea", "play-input-expr");
      ta.rows = 1;
      ta.value = entry.value;
      ta.spellcheck = false;
      ta.placeholder = "torch expression, e.g. torch.randn(1, 3, 64, 64)";
      ta.addEventListener("input", () => { entry.value = ta.value; inputChanged(); });
      row.appendChild(ta);
    }
    const fileBtn = el("label", "play-mini-btn", "file");
    fileBtn.title = "Replace with a file";
    const fileInput = el("input");
    fileInput.type = "file";
    fileInput.style.display = "none";
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length && entry.kind === "mode") {
        // stays an entry of the mode: the file is the interface's to encode,
        // not a plain waveform for the placeholder
        entry.file = fileInput.files[0];
        entry.value = "";
        entry.label = fileInput.files[0].name;
        if (entry._previewUrl) { URL.revokeObjectURL(entry._previewUrl); delete entry._previewUrl; }
        renderInputs();
        inputChanged();
      } else if (fileInput.files.length) {
        entry.kind = "file";
        entry.file = fileInput.files[0];
        entry.label = fileInput.files[0].name;
        // reset the derived media fields so the widget re-derives from the new file
        ["_img", "thumbUrl", "croppedFile", "srcW", "srcH", "imgChannels",
         "imgTargetW", "imgTargetH", "imgMode", "cropX", "cropY", "cropW", "cropH",
         "audioBuffer", "totalSamples", "sampleRate", "numChannels", "targetSR",
         "normalize", "toMono", "cropStart", "cropEnd", "_wfPeaks", "_wfPeaksLen"]
          .forEach((k) => delete entry[k]);
        renderInputs();
        inputChanged();
      }
    });
    fileBtn.appendChild(fileInput);
    // a text mode (a prompt) has no file to take; an audio mode takes recordings
    if (entry.kind !== "mode" || (modeSpec && modeSpec.type === "audio")) {
      if (entry.kind === "mode") fileInput.accept = "audio/*,.wav,.flac,.ogg,.aiff,.aif,.mp3";
      row.appendChild(fileBtn);
    }

    // batch on: in or out of the batch; off: which one runs
    if (!_isScalar(p) && (state.inputs[name] || []).length > 1) {
      const inRun = _runEntries(name).includes(entry);
      const multi = _multiEntry();
      const pick = el("button", "play-mini-btn" + (inRun ? " active" : ""),
                      multi ? (inRun ? "\u2611" : "\u2610") : (inRun ? "\u25CF" : "\u25CB"));
      pick.title = multi
        ? (inRun ? "In the batch — click to leave it out" : "Left out — click to put it in the batch")
        : "Run this entry";
      pick.addEventListener("click", () => {
        if (multi) entry.inBatch = !inRun;
        else state.selected[name] = entry.id;
        renderInputs();
        inputChanged();
      });
      row.insertBefore(pick, row.firstChild);
    }

    const del = el("button", "play-mini-btn danger", "✕");
    del.title = "Remove this entry";
    del.addEventListener("click", () => {
      const arr = state.inputs[name];
      const idx = arr.findIndex((e) => e.id === entry.id);
      if (idx >= 0) arr.splice(idx, 1);
      renderInputs();
      inputChanged();
    });
    row.appendChild(del);
    col.appendChild(row);

    // a recording loaded for an audio mode: just listen to it -- the interface
    // prepares it itself, so the bench's crop/resample widget would not apply
    if (entry.kind === "mode" && entry.file) {
      if (!entry._previewUrl) entry._previewUrl = URL.createObjectURL(entry.file);
      const player = el("audio", "play-mode-audio");
      player.controls = true;
      player.preload = "metadata";
      player.src = entry._previewUrl;
      col.appendChild(player);
    }

    // rich media widgets (preview + crop / resize / resample), same as the editor
    if (entry.kind === "file" && entry.file) {
      if (TBInputs.isImageFile(entry.file) && window.TBImage) {
        const widgetHost = el("div", "play-img-widget");
        col.appendChild(widgetHost);
        TBImage.widget(entry, widgetHost, { shape: _phShape(name), onChange: inputChanged })
          .catch(() => {});
      } else if (window.TBAudio && TBAudio.isAudioFile(entry.file)) {
        const widgetHost = el("div", "play-audio-widget");
        col.appendChild(widgetHost);
        TBAudio.widget(entry, widgetHost, { shape: _phShape(name), onChange: inputChanged })
          .catch(() => {});
      }
    }
    return col;
  }

  // ── header controls ──────────────────────────────────────────────────────────
  function _syncOptionalPicker() {
    const sel = $("play-add-optional");
    if (!sel) return;
    const off = state.placeholders.filter((p) => p.optional && !state.optionalOn.has(p.name));
    sel.innerHTML = "";
    const head = el("option", null, `＋ optional input (${off.length})`);
    head.value = "";
    sel.appendChild(head);
    off.forEach((p) => {
      const suffix = (p.arg_default === undefined || p.arg_default === null)
        ? "" : ` = ${_defaultLabel(p)}`;
      const baked = p.used === false ? "  (baked)" : "";
      const o = el("option", null, `${p.name} · ${p.arg_type}${suffix}${baked}`);
      o.value = p.name;
      sel.appendChild(o);
    });
    sel.style.display = off.length ? "" : "none";
  }

  function _syncImportBtn() {
    const btn = $("play-import-inputs");
    if (!btn) return;
    btn.style.display = "none";   // one bench with the editor: nothing to import
    return;
    const any = state.placeholders.some((p) => !_isScalar(p) && _sharedFor(p.name));
    btn.style.display = any ? "" : "none";
  }

  const BATCH_MODE_HELP = {
    pad: "Batch: zero-pad the shorter entries up to the largest, then stack. "
       + "Entries that already agree are untouched, which is why this is the "
       + "default.",
    loop: "Batch: tile the shorter entries up to the largest, then stack. For "
        + "audio this repeats the material instead of trailing off into silence.",
    stack: "Batch: strict stacking — the entries must already agree on every "
         + "dimension but the first, or the run is an error.",
    sequential: "No batching: one forward pass per entry, outputs gathered "
              + "afterwards. Slower, but the only thing that works when the "
              + "shapes cannot be reconciled (different ranks) or when the model "
              + "hard-codes its batch size.",
  };

  // The most entries any one placeholder holds — that is the batch size, or the
  // number of passes in sequential mode (shorter placeholders reuse their last).
  function _maxEntries() {
    return _visiblePlaceholders().reduce((n, p) => _isScalar(p) ? n
      : Math.max(n, _runEntries(p.name).filter(_entryHasValue).length), 0);
  }

  function _syncBatchUI() {
    const sel = $("play-batch-mode");
    const hint = $("play-batch-hint");
    if ($("play-batch-on")) $("play-batch-on").checked = state.batchOn;
    if (sel) {
      sel.disabled = !state.batchOn;
      // a method that takes one input at a time can still run several, one
      // after another — but not stacked
      Array.from(sel.options).forEach((o) => {
        o.disabled = !state.batchSupported && o.value !== "sequential";
      });
    }
    if (sel) {
      sel.value = state.batchMode;
      sel.title = BATCH_MODE_HELP[state.batchMode] || "";
    }
    if (!hint) return;
    if (state.batchOn && !state.batchSupported && state.batchMode !== "sequential") {
      hint.textContent = "this method takes one input at a time — pick sequential to run several";
      hint.title = "Its interface says it does not batch; the selected entry runs.";
      return;
    }
    const n = _maxEntries();
    if (n <= 1) { hint.textContent = ""; hint.title = ""; return; }
    const sequential = state.batchMode === "sequential";
    hint.textContent = sequential ? `${n} entries ⇒ ${n} runs`
                                  : `${n} entries ⇒ batch of ${n}`;
    hint.title = sequential
      ? "One forward pass per entry; same-shaped outputs are gathered into one "
        + "batch afterwards, mismatched ones are shown per run."
      : "Entries are stacked along dim 0 — the output comes back batched.";
  }

  // The entries of a placeholder that go into the run. Batch on: every one
  // left checked. Off: the selected one (or the first, if none is).
  // Several entries go in: batch on, and either the method batches or they run
  // one after another (sequential is not batching).
  function _multiEntry() {
    if (PAGE_MODE === "generate") return true;
    return state.batchOn && (state.batchSupported || state.batchMode === "sequential");
  }

  function _runEntries(name) {
    const arr = state.inputs[name] || [];
    if (_multiEntry()) return arr.filter((e) => e.inBatch !== false);
    const sel = arr.find((e) => e.id === state.selected[name]);
    return sel ? [sel] : arr.slice(0, 1);
  }

  function buildRunForm() {
    const form = new FormData();
    // batch on: the checked entries of a placeholder are stacked along the batch
    // dim (or run one after another, in sequential mode)
    form.append("batch", _multiEntry() ? "1" : "0");
    form.append("batch_mode", state.batchMode);
    // Input expressions are held fixed across runs (so a macro move changes only
    // the macro); this asks the server for a fresh draw, once.
    if (state.resample) { form.append("resample", "true"); state.resample = false; }
    const claimed = _inputModeClaimed();
    _visiblePlaceholders().forEach((p) => {
      if (claimed[p.name]) return;          // another placeholder's mode fills it
      _runEntries(p.name).forEach((e) => {
        if (e.kind === "file" && (e.croppedFile || e.file)) form.append(p.name, e.croppedFile || e.file);
        else if (e.kind === "scalar") form.append(p.name, String(e.value));
        else if (e.kind === "mode") {
          // an audio mode may hold a loaded file rather than typed text
          if (!e.file && !(e.value || "").trim()) return;
          form.append(p.name, e.file || e.value);
          form.append("__mode__" + p.name, e.mode);
        } else if (e.value && e.value.trim()) form.append(p.name, e.value);
      });
    });
    return form;
  }

  // ── generate mode's form ───────────────────────────────────────────────────
  // Every included entry of every placeholder, each with a label the files it
  // produces are named after. Labels go in the order the server reads the
  // values: uploads first for plain inputs, typed text first for input modes.
  function _entryLabel(e) {
    if (e.file) return (e.file.name || "file").replace(/\.[^.]+$/, "");
    const v = String(e.kind === "scalar" ? e.value : (e.value || "")).trim();
    return v.length > 32 ? v.slice(0, 32) : v;
  }

  function buildGenerateForm() {
    const form = new FormData();
    const claimed = _inputModeClaimed();
    _visiblePlaceholders().forEach((p) => {
      if (claimed[p.name]) return;
      let entries = _runEntries(p.name).filter(_entryHasValue);
      // a placeholder read through its input mode reads every value that way,
      // so an expression beside a prompt would be taken for a prompt
      const modeEntries = entries.filter((e) => e.kind === "mode");
      if (modeEntries.length) entries = modeEntries;
      const files = [], others = [];
      let modeName = null;
      entries.forEach((e) => {
        if (e.kind === "mode") {
          if (!e.file && !(e.value || "").trim()) return;
          form.append(p.name, e.file || e.value);
          modeName = e.mode;
          (e.file ? files : others).push(_entryLabel(e));
        } else if (e.kind === "file" && (e.croppedFile || e.file)) {
          form.append(p.name, e.croppedFile || e.file);
          files.push(_entryLabel(e));
        } else if (e.kind === "scalar") {
          form.append(p.name, String(e.value));
          others.push(_entryLabel(e));
        } else if (e.value && e.value.trim()) {
          form.append(p.name, e.value);
          others.push(_entryLabel(e));
        }
      });
      if (modeName) form.append("__mode__" + p.name, modeName);
      form.append("__labels__" + p.name,
                  JSON.stringify(modeName ? others.concat(files) : files.concat(others)));
    });
    return form;
  }

  // One node of the method, chosen in the editor's activation browser (the
  // same one "＋ bend" opens). Resolves to its name, or null if closed.
  function pickNode(title) {
    return new Promise((resolve) => {
      let done = false;
      const finish = (name) => { if (!done) { done = true; resolve(name); } };
      const browser = TBSearch.openBrowser({
        title,
        nodes: state.bendableNodes,
        ctx: { aliases: state.aliases, tags: TBSearch.loadTags() },
        onPick: (n) => { finish(n.name); if (browser) browser.close(); },
        onClose: () => finish(null),
      });
      if (!browser) finish(null);
    });
  }

  // what generate.js needs from the page it shares with play mode
  window.TBPlay = {
    pickNode,
    state, buildGenerateForm, toast, reportError, postJSON,
    visiblePlaceholders: () => _visiblePlaceholders(),
    entries: (name) => _runEntries(name).filter(_entryHasValue),
    missingRequired: () => _missingRequired(),
  };

  // re-run after an input change while in realtime mode
  let _inputRunTimer = null;
  function inputChanged() {
    _syncRunUI();          // Generate follows what the bench actually holds
    persistInputs();       // …and the bench outlives this visit
    document.dispatchEvent(new CustomEvent("tb:inputs"));
    if (state.runMode !== "realtime") return;
    clearTimeout(_inputRunTimer);
    _inputRunTimer = setTimeout(scheduleRun, 250);
  }

  // ── run ──────────────────────────────────────────────────────────────────────
  // The Generate button and the output pane both reflect the run state, so a slow
  // model never looks idle (or stuck) while it is working.
  function _syncRunUI() {
    // busy covers the whole "you will get an output shortly" span: compiling on
    // entry counts, since the first run is fired the moment it lands.
    const busy = state.running || state.compiling || state.queued;
    const hasOutput = $("play-output").children.length > 0;

    const missing = state.compiled ? _missingRequired() : [];

    const btn = $("play-run");
    // A callback runs through the interface, so it is ready before (and
    // regardless of) the compiled runtime.
    const cbSpec = _cbSpec();
    const ready = state.compiled || !!cbSpec;
    btn.disabled = !ready || state.running || missing.length > 0;
    btn.textContent = state.running ? "⏳ running…" : "▶ Generate";
    btn.title = !ready            ? "Compile the model first"
              : state.running     ? "Running — the result will replace the output below"
              : missing.length    ? "Waiting for an input on: " + missing.join(", ")
              : cbSpec            ? `Run ${cbSpec.label || cbSpec.name} once`
              : "Run the model once";

    const pane = $("play-output-pane");
    pane.classList.toggle("is-busy", busy);
    pane.classList.toggle("has-output", hasOutput);
    $("play-busy").classList.toggle("on", busy);
    // With an output already on screen, busy means "this one is stale" — dim it
    // and spin in the header. With nothing yet, the pane itself is the spinner:
    // never tell the user to press Generate while the model is already running.
    $("play-output-busy").style.display = (busy && !hasOutput) ? "" : "none";
    const emptyEl = $("play-output-empty");
    emptyEl.style.display = (!busy && !hasOutput) ? "" : "none";
    // The bench starts empty, so "nothing here yet" usually means "nothing to run
    // on yet" — name the inputs that are still missing instead of the Generate
    // button the user cannot press.
    emptyEl.textContent = missing.length
      ? `Add an input on ${missing.join(", ")} to run the model.`
      : "Run the model to see its output.";
  }

  let _runTimer = null;
  // ── interface callbacks ──────────────────────────────────────────────────────
  // Not everything a model does is in the graph. An interface can declare whole
  // operations that run Python around it — GPT-2 samples a token at a time
  // through the traced forward — which cannot be compiled and so have no place
  // in the runtime play mode builds. They are offered here as an alternative
  // target for Generate: the macros still drive the same BendingParameters on
  // the same module, so moving one changes what comes back.
  function _cbSpec() {
    return (state.ifaceCallbacks || []).find((c) => c.name === state.runTarget) || null;
  }

  async function loadInterfaceCallbacks() {
    try {
      const r = await fetch("/api/callbacks/");
      const d = await r.json();
      state.ifaceCallbacks = (d && d.callbacks) || [];
    } catch (e) {
      state.ifaceCallbacks = [];
    }
    const sel = $("play-run-target");
    if (!sel) return;
    if (!state.ifaceCallbacks.length) { sel.style.display = "none"; return; }
    sel.style.display = "";
    sel.innerHTML = "";
    const graph = el("option", null, "run: graph");
    graph.value = "";
    sel.appendChild(graph);
    state.ifaceCallbacks.forEach((c) => {
      const o = el("option", null, "run: " + (c.label || c.name));
      o.value = c.name;
      sel.appendChild(o);
    });
    sel.value = state.runTarget || "";
    sel.addEventListener("change", () => setRunTarget(sel.value));
  }

  function setRunTarget(name) {
    state.runTarget = name || "";
    const spec = _cbSpec();
    // These run whole Python loops — GPT-2's generate is about a second — so
    // firing one on every macro nudge is not playable. Hand back the choice
    // rather than making it silently.
    const wasRealtime = state.runMode === "realtime";
    if (spec && wasRealtime) setRunMode("offline");
    // Whatever the status said belonged to the previous target — a graph run
    // that failed on an empty bench, most often — and reads as this target's
    // problem if left standing.
    if (spec) {
      $("play-status").textContent = wasRealtime
        ? `Generate runs ${spec.label || spec.name} — switched to offline, since it runs a `
          + `Python loop per press. Move a macro, then press Generate.`
        : `Generate runs ${spec.label || spec.name}. Macros still drive the same bendings.`;
    } else {
      $("play-status").textContent = state.compiled
        ? "Generate runs the compiled graph."
        : "Compiling…";
    }
    renderCallbackArgs();
    _syncRunUI();
  }

  function renderCallbackArgs() {
    const host = $("play-cb-args");
    if (!host) return;
    const spec = _cbSpec();
    if (!spec) { host.style.display = "none"; host.innerHTML = ""; _cbReaders = {}; return; }
    host.style.display = "";
    host.innerHTML = "";
    _cbReaders = {};

    if (spec.doc) host.appendChild(el("div", "play-cb-doc", spec.doc));
    (spec.args || []).forEach((arg) => {
      const { row, read } = _playCallbackField(arg);
      host.appendChild(row);
      _cbReaders[arg.name] = read;
    });
  }

  let _cbReaders = {};

  function _playCallbackField(arg) {
    const row = el("div", "play-cb-arg");
    const label = el("label", "play-cb-label", arg.name);
    if (arg.optional) label.appendChild(el("span", "play-cb-optional", "  optional"));
    row.appendChild(label);

    if (arg.type === "text") {
      const ta = el("textarea", "play-cb-text");
      ta.value = arg.default == null ? "" : String(arg.default);
      if (arg.placeholder) ta.placeholder = arg.placeholder;
      row.appendChild(ta);
      return { row, read: () => ta.value };
    }
    if (arg.type === "bool") {
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = !!arg.default;
      label.prepend(cb);
      return { row, read: () => cb.checked };
    }
    if (arg.type === "choice") {
      const sel = el("select", "play-cb-input");
      (arg.choices || []).forEach((c) => {
        const o = el("option", null, String(c));
        o.value = String(c);
        if (String(c) === String(arg.default)) o.selected = true;
        sel.appendChild(o);
      });
      row.appendChild(sel);
      return { row, read: () => sel.value };
    }
    if (arg.type === "int" || arg.type === "float") {
      const wrap = el("div", "play-cb-row");
      const num = document.createElement("input");
      num.type = "number";
      num.className = "play-cb-input";
      num.value = arg.default == null ? "" : String(arg.default);
      num.step = arg.step != null ? arg.step : (arg.type === "int" ? 1 : 0.01);
      if (arg.range) {
        num.min = arg.range[0]; num.max = arg.range[1];
        const rng = document.createElement("input");
        rng.type = "range";
        rng.className = "play-cb-range";
        rng.min = arg.range[0]; rng.max = arg.range[1]; rng.step = num.step;
        rng.value = num.value === "" ? arg.range[0] : num.value;
        rng.addEventListener("input", () => { num.value = rng.value; });
        num.addEventListener("input", () => { rng.value = num.value; });
        wrap.appendChild(rng);
      }
      wrap.appendChild(num);
      row.appendChild(wrap);
      return { row, read: () => num.value };
    }
    const inp = document.createElement("input");
    inp.type = "text";
    inp.className = "play-cb-input";
    inp.value = arg.default == null ? "" : String(arg.default);
    row.appendChild(inp);
    return { row, read: () => inp.value };
  }

  async function runCallbackOnce(spec, signal) {
    const payload = {};
    Object.entries(_cbReaders).forEach(([k, read]) => { payload[k] = read(); });
    const r = await fetch(`/api/callbacks/${encodeURIComponent(spec.name)}/run/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok || data.error) throw _httpError(data, r);
    return data;
  }

  function renderCallbackOutput(data) {
    const host = $("play-output");
    host.innerHTML = "";
    const res = (data && data.result) || {};
    const card = el("div", "play-out");
    const head = el("div", "play-out-head");
    head.appendChild(el("span", "play-out-label", data.name || "result"));
    head.appendChild(el("span", "play-out-shape", res.medium || res.kind || res.view || ""));
    card.appendChild(head);

    if (res.kind === "text" || res.kind === "repr") {
      (res.items || []).forEach((t) => card.appendChild(el("div", "play-out-text", t)));
    } else {
      // Came back through the same view system as an activation, so the same
      // renderer draws it — TBViews.render takes (payload, container).
      const holder = el("div");
      card.appendChild(holder);
      try {
        if (window.TBViews) TBViews.render(res, holder);
        else holder.textContent = JSON.stringify(res).slice(0, 400);
      } catch (e) { holder.textContent = "could not render: " + (e.message || e); }
    }
    host.appendChild(card);
    _syncRunUI();
  }

  // Drop a stale "Run failed …" once something has actually succeeded.
  function _clearRunError() {
    const el0 = $("play-status");
    if (el0 && /^Run failed/.test(el0.textContent || "")) {
      const spec = _cbSpec();
      el0.textContent = spec
        ? `Generate runs ${spec.label || spec.name}. Macros still drive the same bendings.`
        : "Generate runs the compiled graph.";
    }
  }

  function scheduleRun() {
    if (PAGE_MODE === "generate") return;     // generate mode runs jobs, not passes
    if (!state.compiled && !state.runTarget) return;
    clearTimeout(_runTimer);
    // Held rather than failed: with an empty bench there is simply nothing to run
    // on, and _syncRunUI says which input is still missing. Dropping the pending
    // timer first means removing the last entry cancels the run instead of
    // letting it fire into the guard.
    if (_missingRequired().length) { state.queued = false; _syncRunUI(); return; }
    // A macro / input moved: whatever is in flight is already stale, so drop it
    // and start over rather than queueing behind it. Pace the restarts by how
    // long the model actually takes — aborting only closes the connection, the
    // server still finishes the pass, so a drag must not leave a pile of them.
    const wait = Math.min(Math.max(state.lastRunMs || 0, 30), 500);
    state.queued = true;
    _syncRunUI();
    _runTimer = setTimeout(runOnce, wait);
  }

  async function runOnce() {
    state.queued = false;
    // A declared callback runs through the interface, not the compiled runtime,
    // so it does not wait on compilation.
    const cbSpec = _cbSpec();
    if (!state.compiled && !cbSpec) { _syncRunUI(); return; }
    const missing = _missingRequired();
    if (missing.length) {
      toast("Add an input on " + missing.join(", ") + " first");
      _syncRunUI();
      return;
    }
    if (state.abort) state.abort.abort();       // supersede the run in flight
    const ctl = new AbortController();
    state.abort = ctl;
    state.running = true;
    _syncRunUI();
    try {
      if (cbSpec) {
        const data = await runCallbackOnce(cbSpec, ctl.signal);
        renderCallbackOutput(data);
        _clearRunError();
        if (data.run_ms != null) state.lastRunMs = data.run_ms;
        $("play-perf").textContent =
          (data.run_ms != null ? data.run_ms.toFixed(1) + " ms" : "") + " · interface";
        return;
      }
      const r = await fetch("/api/play/run/", { method: "POST", body: buildRunForm(),
                                                signal: ctl.signal });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw _httpError(data, r);
      state.audioEpoch += 1;          // this run's audio supersedes the last one's
      renderOutputs(data.outputs || []);
      // A run that worked clears whatever the last failure left standing —
      // otherwise the first auto-run against an empty bench keeps accusing the
      // page long after the inputs are fine.
      _clearRunError();
      if (data.run_ms != null) state.lastRunMs = data.run_ms;
      const runs = data.runs > 1 ? ` · ${data.runs} runs` : "";
      $("play-perf").textContent =
        `${data.run_ms != null ? data.run_ms.toFixed(1) + " ms" : ""}${runs}`
        + ` · ${data.mode} · ${data.device}`;
    } catch (e) {
      if (e.name === "AbortError") return;      // a newer run owns the output
      const msg = reportError("Run failed", e);
      $("play-perf").textContent = "error";
      // a failing model would otherwise re-fire on every macro nudge, stacking
      // panels — drop back to manual until the user asks for another run
      if (state.runMode === "realtime") {
        setRunMode("offline");
        $("play-status").textContent =
          `Run failed (${shorten(msg)}) — switched to offline; fix it and press Generate.`;
      } else {
        $("play-status").textContent = `Run failed: ${shorten(msg)}`;
      }
    } finally {
      if (state.abort === ctl) {                // only the latest run clears the UI
        state.abort = null;
        state.running = false;
        _syncRunUI();
      }
    }
  }

  // ── output rendering ─────────────────────────────────────────────────────────
  function renderOutputs(outputs) {
    const host = $("play-output");
    host.innerHTML = "";
    outputs.forEach((o, i) => host.appendChild(renderOneOutput(o, i)));
    _syncRunUI();   // owns the empty / busy placeholders
  }

  function renderOneOutput(o, idx) {
    const card = el("div", "play-out");
    const head = el("div", "play-out-head");
    head.appendChild(el("span", "play-out-label", o.label || `output[${idx}]`));
    head.appendChild(el("span", "play-out-shape", o.shape ? "[" + o.shape.join(", ") + "]" : ""));
    card.appendChild(head);

    // view picker (from server _view_meta); switching POSTs the choice then re-runs
    const meta = o._view_meta;
    if (meta && meta.compatible && meta.compatible.length > 1) {
      const node = o.node || o.label;
      card.appendChild(TBViews.renderPicker(meta, async (view, options) => {
        try {
          await postJSON(`/api/views/${encodeURIComponent(state.fn)}/${encodeURIComponent(node)}/`,
                         { view, options });
          runOnce();   // re-run so the output re-serializes with the chosen view
        } catch (e) { reportError("View change failed", e); }
      }));
    }

    const vizBox = el("div", "play-viz");
    TBViews.render(o, vizBox, {
      audioFetch: o.is_audio_compatible ? ((sel) => _fetchRunAudio(idx, sel)) : null,
      // keep the chosen batch/channel view mode across runs (cards are recreated each run)
      stateKey: "play:" + state.fn + ":" + (o.node || o.label || idx),
      // …but not the audio it rendered: this run's output is different audio
      audioEpoch: state.audioEpoch,
    });
    card.appendChild(vizBox);
    return card;
  }

  async function _fetchRunAudio(idx, sel) {
    const form = buildRunForm();
    // render the batch / channel the user is looking at, not always the first
    if (sel) { form.append("batch_idx", String(sel.batch)); form.append("channel", String(sel.channel)); }
    const r = await fetch(`/api/play/run_audio/${idx}/`, { method: "POST", body: form });
    if (!r.ok) {
      const data = await r.json().catch(() => ({}));
      const err = _httpError(data, r);
      reportError("Audio render failed", err);
      throw err;
    }
    return r.blob();
  }

  // ── wiring ──────────────────────────────────────────────────────────────────
  // One device per model, shared with the graph editor: the dropdown starts on
  // wherever the model already is, and greys out what the interface says the
  // method will not run on (its _device_compat_).
  let _deviceCompat = {};
  async function loadDevices() {
    try {
      const r = await fetch("/api/play/devices/");
      const data = await r.json();
      _deviceCompat = data.compat || {};
      const sel = $("play-device");
      sel.innerHTML = "";
      (data.devices || [{ value: "cpu", label: "CPU" }]).forEach((d) => {
        const opt = el("option", null, d.label);
        opt.value = d.value;
        sel.appendChild(opt);
      });
      sel.value = data.current || "cpu";
      state.device = sel.value;
      _applyDeviceCompat();
    } catch (e) { /* cpu only */ }
  }

  function _applyDeviceCompat() {
    const sel = $("play-device");
    if (!sel) return;
    const compat = _deviceCompat[state.fn] || {};
    Array.from(sel.options).forEach((opt) => {
      opt.disabled = compat[opt.value] === false;
      opt.title = opt.disabled ? `${state.fn} is not expected to work on ${opt.value}` : "";
    });
  }

  function setRunMode(mode) {
    state.runMode = mode;
    $("play-mode-realtime").classList.toggle("active", mode === "realtime");
    $("play-mode-offline").classList.toggle("active", mode === "offline");
    $("play-run").style.display = mode === "offline" ? "" : "none";
    _syncRunUI();
    if (mode === "realtime") scheduleRun();
  }

  function applyOrient() {
    const rows = state.orient === "rows";
    $("play-layout").classList.toggle("orient-rows", rows);
    const btn = $("play-orient");
    btn.textContent = rows ? "⬍ rows" : "⬌ columns";
    applyFold();
  }

  // ── foldable panes ─────────────────────────────────────────────────────────
  // The input bench and the output can each be folded down to their header, to
  // give the rest of the page their room. Remembered per browser.
  const _FOLD_PANES = { inputs: "play-inputs-pane", output: "play-output-pane" };

  function _loadFolded() {
    try {
      const saved = JSON.parse(localStorage.getItem("tb_play_folded") || "{}");
      return { inputs: !!saved.inputs, output: !!saved.output };
    } catch (_) {
      return { inputs: false, output: false };
    }
  }

  function applyFold() {
    const f = state.folded;
    for (const [key, id] of Object.entries(_FOLD_PANES)) {
      const pane = $(id);
      if (!pane) continue;
      pane.classList.toggle("folded", !!f[key]);
      const btn = pane.querySelector(".play-fold-btn");
      if (btn) {
        btn.textContent = f[key] ? "▸" : "▾";
        btn.title = f[key] ? "Unfold" : "Fold";
      }
    }
    // the grid tracks follow: a folded pane takes only its header's room, and
    // the space it frees goes to the panes still open
    const layout = $("play-layout");
    // generate mode's panes: bench · sweep · export (the output cannot fold)
    const gen = PAGE_MODE === "generate";
    if (state.orient === "rows") {
      layout.style.gridTemplateColumns = "";
      layout.style.gridTemplateRows = gen ? [
        f.inputs ? "auto" : "minmax(90px, 26%)", "minmax(140px, 1fr)", "minmax(140px, 1fr)",
      ].join(" ") : [
        f.inputs ? "auto" : "minmax(90px, 26%)",
        f.output ? "1fr" : "minmax(90px, 22%)",
        f.output ? "auto" : "1fr",
      ].join(" ");
    } else {
      layout.style.gridTemplateRows = "";
      layout.style.gridTemplateColumns = gen ? [
        f.inputs ? "34px" : "minmax(240px, 1fr)", "minmax(320px, 1.4fr)", "minmax(300px, 1.2fr)",
      ].join(" ") : [
        f.inputs ? "34px" : "minmax(240px, 1fr)",
        "minmax(220px, 0.9fr)",
        f.output ? "34px" : "minmax(300px, 1.6fr)",
      ].join(" ");
    }
  }

  function toggleFold(key) {
    state.folded[key] = !state.folded[key];
    try { localStorage.setItem("tb_play_folded", JSON.stringify(state.folded)); } catch (_) {}
    applyFold();
  }

  function toggleOrient() {
    state.orient = state.orient === "rows" ? "columns" : "rows";
    localStorage.setItem("tb_play_orient", state.orient);
    applyOrient();
  }

  function init() {
    // method buttons
    document.querySelectorAll("#play-method-selector .method-btn").forEach((b) => {
      b.addEventListener("click", () => {
        document.querySelectorAll("#play-method-selector .method-btn")
          .forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        state.fn = b.dataset.fn;
        _applyDeviceCompat();
        compile();
      });
    });
    $("play-device").addEventListener("change", (e) => { state.device = e.target.value; compile(); });
    $("play-scripted").addEventListener("change", (e) => { state.scripted = e.target.checked; compile(); });
    $("play-recompile").addEventListener("click", compile);
    $("play-orient").addEventListener("click", toggleOrient);
    for (const [key, id] of Object.entries(_FOLD_PANES)) {
      const header = $(id) && $(id).querySelector(".play-pane-header");
      if (!header) continue;
      // the caret folds and unfolds; on a folded pane the title (or, in
      // columns mode, anywhere on the strip) unfolds it too
      header.addEventListener("click", (e) => {
        const onCaret = e.target.closest(".play-fold-btn");
        const folded = state.folded[key];
        const onTitle = e.target.closest(".play-pane-title");
        const onStrip = folded && state.orient !== "rows" && e.target === header;
        if (onCaret || (folded && (onTitle || onStrip))) toggleFold(key);
      });
    }
    $("play-run").addEventListener("click", runOnce);
    $("play-resample").addEventListener("click", () => {
      state.resample = true;
      runOnce();
    });
    $("play-import-inputs").addEventListener("click", importAllFromEditor);
    // a slider grab disarmed its row's draggable; give it back on release
    document.addEventListener("pointerup", () => {
      document.querySelectorAll('.play-bind-param[data-joinable="1"]')
        .forEach((r) => { r.draggable = true; });
    });
    $("play-derived-head").addEventListener("click", toggleDerived);
    $("play-macro-edit").addEventListener("click", toggleMacroEdit);
    $("play-add-bend").addEventListener("click", openAddBend);
    $("play-refresh-macros").addEventListener("click", async () => {
      const before = state.macros.length;
      try {
        await refreshMacros();
        const added = state.macros.length - before;
        toast(added > 0 ? `${added} new macro${added > 1 ? "s" : ""} from the editor`
                        : "Macros are up to date");
      } catch (e) { reportError("Could not read the editor's macros", e); }
    });
    $("play-add-macro").addEventListener("change", (e) => {
      const v = e.target.value;
      e.target.selectedIndex = 0;
      if (!v) return;
      if (v.startsWith("promote:")) {
        const rest = v.slice("promote:".length);
        const cut = rest.indexOf(":");
        promoteToMacro(rest.slice(0, cut), rest.slice(cut + 1));
      } else {
        addMacroFromEditor(v.slice("have:".length));
      }
    });
    $("play-batch-mode").addEventListener("change", (e) => {
      state.batchMode = e.target.value;
      _syncBatchUI();
      renderInputs();
      inputChanged();
    });
    $("play-batch-on").addEventListener("change", (e) => {
      state.batchOn = e.target.checked;
      _syncBatchUI();
      renderInputs();
      inputChanged();
    });
    // the bench's shared settings, before the first compile runs anything
    const _bench = TBBench.load(window.PLAY_MODEL);
    if (_bench && _bench.batch) {
      state.batchOn = !!_bench.batch.on;
      if (TBBench.BATCH_MODES.includes(_bench.batch.mode)) state.batchMode = _bench.batch.mode;
    }
    _syncBatchUI();

    // leaving for the editor: write the bench now (the debounced save may not
    // have run); coming back from the browser cache: reload to read it
    window.addEventListener("pagehide", () => { try { _savePlayBenchNow(); } catch (_) {} });
    window.addEventListener("pageshow", (e) => { if (e.persisted) location.reload(); });
    $("play-add-optional").addEventListener("change", (e) => {
      const name = e.target.value;
      e.target.selectedIndex = 0;
      if (!name) return;
      state.optionalOn.add(name);
      renderInputs();
    });
    $("play-mode-realtime").addEventListener("click", () => setRunMode("realtime"));
    $("play-mode-offline").addEventListener("click", () => setRunMode("offline"));

    setRunMode("realtime");
    _syncBatchUI();        // restore the persisted choice before the first render
    applyOrient();
    loadInterfaceCallbacks();
    loadDevices().then(compile);

    // restore the module's device when leaving play mode
    window.addEventListener("pagehide", () => {
      try {
        navigator.sendBeacon("/api/play/release/");
      } catch (e) { /* ignore */ }
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
