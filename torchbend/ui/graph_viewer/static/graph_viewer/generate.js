/* TorchBend — Generate mode
 *
 * Offline batch generation over combinations of bending values. Runs on play
 * mode's page (play.js keeps the input bench and the compile; see TBPlay):
 * this file adds the sweep designer and the export panel.
 *
 *   config = { fn,
 *              params:  [{key, mode: fixed|values|range|random, value, values,
 *                         min, max, n, scale: lin|log, group}],
 *              inputs:  "each" | "zip" | "first",       how bench entries vary
 *              seed:    {base, repeats},
 *              combine: {mode: product|zip|random, n, seed},
 *              export:  {dir, name, type, tensors, input_as_original,
 *                        split_batch, stop_on_error,
 *                        targets: [{node, format}]} }   outputs or activations
 *
 * The server expands it (generation.py) — every change asks it for the plan,
 * so the run count and the first file names are always in view.
 */
(function () {
  "use strict";
  if (window.PLAY_PAGE_MODE !== "generate") return;

  const P = () => window.TBPlay;
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, txt) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  };

  const SAMPLE_HELP = {
    fixed:  "One value for every run",
    values: "Exactly these values (comma separated)",
    range:  "n evenly spaced values from min to max, both included",
    random: "n values drawn uniformly between min and max (reproducible: the plan seed)",
  };
  const COMBINE_HELP = {
    product: "Every combination of the groups",
    zip:     "Groups advance together — run i takes the i-th value of each (shorter ones cycle)",
    random:  "n distinct combinations drawn from the product",
  };
  const INPUTS_HELP = {
    each:  "Every included bench entry is its own axis, combined with the rest",
    zip:   "Bench inputs advance together: entry i of each placeholder",
    first: "Only the first included entry of each input",
  };

  const g = {
    fn: null,
    params: [],            // sweepable, from the server
    outputs: [],           // what the method returns: [{index, node, label, format, …}]
    nodeInfo: {},          // node -> its description as a target (outputs and picked ones)
    cfg: null,
    plan: null,
    job: null,             // last status
    poll: null,
    planTimer: null,
    planSeq: 0,
  };

  // ── config, remembered per model and method ──────────────────────────────
  function _key() { return "tb_generate_" + (window.PLAY_MODEL || "") + "_" + (g.fn || ""); }

  function _defaults() {
    return {
      params: [], inputs: "each",
      seed: { base: "", repeats: 1 },
      combine: { mode: "product", n: 20, seed: 0 },
      export: { dir: "", name: "{input}_{index:04d}", type: "{fn}",
                tensors: false, input_as_original: false, split_batch: true, targets: null,
                stop_on_error: true },
    };
  }

  function _load() {
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(_key()) || "null"); } catch (_) {}
    const d = _defaults();
    if (!saved) return d;
    const out = Object.assign(d, saved, {
      seed: Object.assign(d.seed, saved.seed || {}),
      combine: Object.assign(d.combine, saved.combine || {}),
      export: Object.assign(d.export, saved.export || {}),
      params: Array.isArray(saved.params) ? saved.params : [],
    });
    if (!Array.isArray(out.export.targets)) out.export.targets = null;   // set once outputs are known
    delete out.export.outputs;             // the switches this replaced
    return out;
  }

  function _save() {
    try { localStorage.setItem(_key(), JSON.stringify(g.cfg)); } catch (_) {}
  }

  function changed() {
    _save();
    schedulePlan();
  }

  function serverConfig() {
    const c = g.cfg;
    return {
      fn: g.fn,
      params: c.params,
      inputs: c.inputs,
      seed: { base: c.seed.base === "" ? null : Number(c.seed.base),
              repeats: Number(c.seed.repeats) || 1 },
      combine: { mode: c.combine.mode, n: Number(c.combine.n) || 1,
                 seed: Number(c.combine.seed) || 0 },
      export: Object.assign({}, c.export, {
        dir: (c.export.dir || "").trim() || undefined,
        targets: c.export.targets || [],
      }),
    };
  }

  // ── parameters ───────────────────────────────────────────────────────────
  async function loadParams() {
    try {
      const r = await fetch("/api/generate/params/?fn=" + encodeURIComponent(g.fn));
      const data = await r.json();
      if (data.error) throw new Error(data.error);
      g.params = data.params || [];
      g.outputs = data.outputs || [];
      g.outputs.forEach((o) => { g.nodeInfo[o.node] = o; });
      _defaultTargets();
      if (data.running) attach(data.running);
    } catch (e) {
      g.params = [];
      P().reportError("Could not list what can vary", e);
    }
  }

  function _param(key) { return g.params.find((p) => p.key === key); }

  function _newSpec(p) {
    const num = (v, d) => (v == null || v === "" || isNaN(Number(v)) ? d : Number(v));
    const value = num(p.value, 0);
    let lo = num(p.min, null), hi = num(p.max, null);
    if (lo == null || hi == null) {
      const span = Math.abs(value) || 1;
      lo = lo == null ? value - span : lo;
      hi = hi == null ? value + span : hi;
    }
    return { key: p.key, mode: p.type === "bool" ? "range" : "range", value,
             values: String(value), min: lo, max: hi, n: p.type === "bool" ? 2 : 5,
             scale: "lin", group: "" };
  }

  function renderAddPicker() {
    const sel = $("gen-add-param");
    sel.innerHTML = "";
    sel.appendChild(new Option("＋ parameter", ""));
    const used = new Set(g.cfg.params.map((s) => s.key));
    const groups = {};
    g.params.forEach((p) => { (groups[p.group_label] = groups[p.group_label] || []).push(p); });
    Object.entries(groups).forEach(([label, items]) => {
      const og = document.createElement("optgroup");
      og.label = label;
      items.forEach((p) => {
        const o = new Option(p.label + (used.has(p.key) ? "  ✓" : ""), p.key);
        o.disabled = used.has(p.key);
        og.appendChild(o);
      });
      sel.appendChild(og);
    });
    sel.disabled = !g.params.length;
    sel.title = g.params.length ? "A macro, or a parameter of a bending no macro drives"
      : "Nothing to vary for this method yet: add a bending in the editor or play mode";
  }

  function renderParams() {
    const box = $("gen-params");
    box.innerHTML = "";
    // drop what no longer exists (a bending removed since)
    g.cfg.params = g.cfg.params.filter((s) => _param(s.key));
    $("gen-param-count").textContent = g.cfg.params.length;
    $("gen-no-params").style.display = g.cfg.params.length ? "none" : "";
    const groups = [...new Set(g.cfg.params.map((s) => s.group).filter(Boolean))];
    g.cfg.params.forEach((spec) => box.appendChild(renderParam(spec, groups)));
    renderAddPicker();
  }

  function _field(label, input, title) {
    const w = el("label", "gen-field");
    w.appendChild(el("span", "gen-field-label", label));
    w.appendChild(input);
    if (title) w.title = title;
    return w;
  }

  function _input(value, type, onChange, attrs) {
    const i = el("input", "gen-input");
    i.type = type || "text";
    i.value = value == null ? "" : value;
    Object.assign(i, attrs || {});
    i.addEventListener("change", () => onChange(i.type === "checkbox" ? i.checked : i.value));
    if (i.type === "checkbox") i.checked = !!value;
    return i;
  }

  function _select(options, value, onChange, help) {
    const s = el("select", "play-mini-select");
    options.forEach((o) => s.appendChild(new Option(o, o)));
    s.value = value;
    if (help) s.title = help[value] || "";
    s.addEventListener("change", () => { if (help) s.title = help[s.value] || ""; onChange(s.value); });
    return s;
  }

  function renderParam(spec, groups) {
    const p = _param(spec.key);
    const card = el("div", "gen-param");
    const head = el("div", "gen-param-head");
    head.appendChild(el("span", "gen-param-label", p.label));
    head.appendChild(el("span", "gen-type", p.type));
    const count = el("span", "gen-count");
    count.dataset.key = spec.key;
    head.appendChild(count);
    const rm = el("button", "play-mini-btn gen-remove", "✕");
    rm.title = "Stop varying this (it keeps its current value)";
    rm.addEventListener("click", () => {
      g.cfg.params = g.cfg.params.filter((s) => s !== spec);
      renderParams();
      changed();
    });
    head.appendChild(rm);
    card.appendChild(head);

    const row = el("div", "gen-param-row");
    const modes = ["fixed", "values", "range", "random"];
    row.appendChild(_select(modes, spec.mode, (v) => { spec.mode = v; renderParams(); changed(); },
                            SAMPLE_HELP));
    const set = (k) => (v) => { spec[k] = v; changed(); };
    const numAttrs = p.type === "int" ? { step: 1 } : { step: "any" };
    if (spec.mode === "fixed") {
      row.appendChild(p.type === "bool"
        ? _field("value", _input(!!spec.value, "checkbox", set("value")))
        : _field("value", _input(spec.value, "number", set("value"), numAttrs)));
    } else if (spec.mode === "values") {
      row.appendChild(_field("values", _input(spec.values, "text", set("values"),
        { placeholder: p.type === "bool" ? "0, 1" : "0.1, 0.5, 2", className: "gen-input gen-wide" })));
    } else if (p.type !== "bool") {
      row.appendChild(_field("min", _input(spec.min, "number", set("min"), numAttrs)));
      row.appendChild(_field("max", _input(spec.max, "number", set("max"), numAttrs)));
      row.appendChild(_field("n", _input(spec.n, "number", set("n"), { min: 1, step: 1 })));
      if (spec.mode === "range" && p.type === "float") {
        row.appendChild(_select(["lin", "log"], spec.scale || "lin", set("scale")));
      }
    } else if (spec.mode === "random") {
      row.appendChild(_field("n", _input(spec.n, "number", set("n"), { min: 1, step: 1 })));
    }
    if (p.choices && p.choices.length) {
      row.appendChild(el("span", "gen-hint", "choices: " + p.choices.join(", ")));
    }
    card.appendChild(row);

    const grow = el("div", "gen-param-row");
    const gi = _input(spec.group, "text", (v) => { spec.group = v.trim(); renderParams(); changed(); },
                      { placeholder: "own", className: "gen-input gen-group" });
    const listId = "gen-groups-list";
    gi.setAttribute("list", listId);
    grow.appendChild(_field("group", gi,
      "Parameters in the same group move together (value i of each) instead of being combined"));
    let dl = $(listId);
    if (!dl) { dl = el("datalist"); dl.id = listId; document.body.appendChild(dl); }
    dl.innerHTML = "";
    groups.forEach((name) => dl.appendChild(new Option(name, name)));
    grow.appendChild(el("span", "gen-hint", "now " + _fmt(p.value)));
    card.appendChild(grow);
    return card;
  }

  function _fmt(v) {
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toPrecision(4).replace(/\.?0+$/, "");
    return String(v);
  }

  // ── inputs, seed, combination ────────────────────────────────────────────
  function renderCombine() {
    const box = $("gen-combine");
    box.innerHTML = "";
    const c = g.cfg;

    box.appendChild(el("div", "play-subhead", "Inputs"));
    const r1 = el("div", "gen-param-row");
    r1.appendChild(_select(["each", "zip", "first"], c.inputs,
                           (v) => { c.inputs = v; changed(); }, INPUTS_HELP));
    const inputsInfo = el("span", "gen-hint");
    inputsInfo.id = "gen-inputs-info";
    r1.appendChild(inputsInfo);
    box.appendChild(r1);

    box.appendChild(el("div", "play-subhead", "Seed"));
    const r2 = el("div", "gen-param-row");
    r2.appendChild(_field("base", _input(c.seed.base, "number",
      (v) => { c.seed.base = v; changed(); }, { placeholder: "none", step: 1 }),
      "Seed each run with this (plus its repeat): runs become reproducible, and each is computed in full. Empty: no seeding, and unchanged parts of the model are reused from the previous run."));
    r2.appendChild(_field("repeats", _input(c.seed.repeats, "number",
      (v) => { c.seed.repeats = v; changed(); }, { min: 1, step: 1 }),
      "Run every combination this many times, with seeds base, base+1, …"));
    box.appendChild(r2);

    box.appendChild(el("div", "play-subhead", "Combine"));
    const r3 = el("div", "gen-param-row");
    r3.appendChild(_select(["product", "zip", "random"], c.combine.mode,
      (v) => { c.combine.mode = v; renderCombine(); changed(); }, COMBINE_HELP));
    if (c.combine.mode === "random") {
      r3.appendChild(_field("n", _input(c.combine.n, "number",
        (v) => { c.combine.n = v; changed(); }, { min: 1, step: 1 }), "How many combinations"));
    }
    r3.appendChild(_field("plan seed", _input(c.combine.seed, "number",
      (v) => { c.combine.seed = v; changed(); }, { step: 1 }),
      "Seeds the random values and the random combinations, so the same plan comes back"));
    box.appendChild(r3);
    _syncInputsInfo();
  }

  function _syncInputsInfo() {
    const info = $("gen-inputs-info");
    if (!info || !g.plan) return;
    const parts = Object.entries(g.plan.inputs || {}).map(([n, labels]) => `${n}: ${labels.length}`);
    info.textContent = parts.length ? parts.join(" · ") : "no inputs on the bench";
  }

  // ── export ───────────────────────────────────────────────────────────────
  function renderExport() {
    const box = $("gen-export");
    box.innerHTML = "";
    const x = g.cfg.export;
    const set = (k) => (v) => { x[k] = v; changed(); };

    box.appendChild(_field("folder", _input(x.dir, "text", set("dir"),
      { placeholder: `generations/${window.PLAY_MODEL || "model"}/${g.fn}/<date-time>`,
        className: "gen-input gen-wide" }),
      "Where the files go, on the machine running torchbend. Relative to where it was started."));
    const name = _input(x.name, "text", set("name"), { className: "gen-input gen-wide" });
    name.id = "gen-name";
    box.appendChild(_field("file name", name,
      "A template: {field} is replaced by its value, {field:.2f} formats it, / makes folders"));
    const chips = el("div", "gen-chips");
    chips.id = "gen-fields";
    box.appendChild(chips);
    box.appendChild(_field("type", _input(x.type, "text", set("type"), { className: "gen-input gen-wide" }),
      "What the audio explorer colours generations by — a template like the file name"));

    box.appendChild(renderTargets());

    const flags = el("div", "gen-flags");
    [["tensors", "also save tensors (.pt)"],
     ["input_as_original", "input as original",
      "Audio explorer: group the generations made from the same audio input, and "
      + "compare them to that recording. Off (or no audio input): all the "
      + "generations are mapped in one space, against their mean."],
     ["split_batch", "one file per batch item"],
     ["stop_on_error", "stop at the first error"]].forEach(([k, label, title]) => {
      const w = el("label", "gen-flag");
      if (title) w.title = title;
      w.appendChild(_input(x[k], "checkbox", set(k)));
      w.appendChild(document.createTextNode(" " + label));
      flags.appendChild(w);
    });
    box.appendChild(flags);
    _syncFields();
  }

  // What gets written for each run: one output to start with -- the audio one,
  // when the method declares one -- and whatever else ＋ adds: the other
  // outputs, or any activation of the graph. Only these are computed.
  function _defaultTargets() {
    const x = g.cfg.export;
    if (Array.isArray(x.targets) || !g.outputs.length) return;
    const first = g.outputs.find((o) => o.audio) || g.outputs[0];
    x.targets = [{ node: first.node, format: first.format }];
    _save();
  }

  const FORMAT_HELP = {
    auto: "wav for audio, png for images, pt for anything else",
    wav: "Audio file, at the rate the model declares (or infers)",
    png: "Image, min-max normalised",
    pt: "The tensor itself (torch.save)",
  };

  async function _info(node) {
    if (g.nodeInfo[node]) return g.nodeInfo[node];
    const r = await fetch(`/api/generate/target/?fn=${encodeURIComponent(g.fn)}&node=${encodeURIComponent(node)}`);
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    g.nodeInfo[node] = data;
    return data;
  }

  async function addTarget(node) {
    try {
      const info = await _info(node);
      const x = g.cfg.export;
      x.targets = x.targets || [];
      if (x.targets.some((t) => t.node === node)) return;
      x.targets.push({ node, format: info.format });
      renderExport();
      changed();
      _syncStart();
    } catch (e) {
      P().reportError("Cannot export that node", e);
    }
  }

  function renderTargets() {
    const box = el("div", "gen-outputs");
    const head = el("div", "gen-param-row");
    head.appendChild(el("span", "play-subhead", "Export"));
    const targets = g.cfg.export.targets || [];
    const add = el("select", "play-mini-select");
    add.appendChild(new Option("＋", ""));
    g.outputs.filter((o) => !targets.some((t) => t.node === o.node))
      .forEach((o) => add.appendChild(new Option("output " + o.label, "node:" + o.node)));
    add.appendChild(new Option("an activation…", "pick"));
    add.title = "Export another output of the method, or any activation of its graph";
    add.addEventListener("change", async () => {
      const v = add.value;
      add.selectedIndex = 0;
      if (v === "pick") {
        const node = await P().pickNode("＋ export an activation");
        if (node) addTarget(node);
      } else if (v) {
        addTarget(v.slice("node:".length));
      }
    });
    head.appendChild(add);
    box.appendChild(head);
    if (!targets.length) {
      box.appendChild(el("div", "gen-hint", "nothing to export yet — ＋ adds an output or an activation"));
    }
    targets.forEach((t, i) => {
      const info = g.nodeInfo[t.node];
      if (!info) { _info(t.node).then(renderExport).catch(() => {}); }
      const isOutput = g.outputs.some((o) => o.node === t.node);
      const row = el("div", "gen-output");
      const name = el("span", "gen-output-label", info ? info.label : t.node);
      name.title = `${isOutput ? "output" : "activation"} ${t.node}`
        + (info && info.shape.length ? ` · shape [${info.shape.join(", ")}]` : "")
        + (info && info.audio ? " · declared audio" : "")
        + (info ? ` · {output} in a file name: ${info.field}` : "");
      row.appendChild(name);
      if (!isOutput) row.appendChild(el("span", "gen-type", "activation"));
      if (info && info.shape.length) row.appendChild(el("span", "gen-hint", `[${info.shape.join("×")}]`));
      const fmt = _select(["auto", "wav", "png", "pt"], t.format || "auto",
                          (v) => { t.format = v; changed(); }, FORMAT_HELP);
      fmt.classList.add("gen-output-format");
      row.appendChild(fmt);
      const rm = el("button", "play-mini-btn gen-remove", "✕");
      rm.title = "Do not export this";
      rm.addEventListener("click", () => {
        targets.splice(i, 1);
        renderExport();
        changed();
        _syncStart();
      });
      row.appendChild(rm);
      box.appendChild(row);
    });
    return box;
  }

  // A field dropped into the name always stands apart: "_" between it and
  // whatever it touches, unless that already is a separator (or a folder /).
  const _SEP = /[_\/\-. ]/;
  function _insertField(text, at, end, field) {
    const before = text.slice(0, at), after = text.slice(end);
    const left = before && !_SEP.test(before[before.length - 1]) ? "_" : "";
    const right = after && !_SEP.test(after[0]) ? "_" : "";
    const block = left + "{" + field + "}" + right;
    return { text: before + block + after, caret: (before + block).length };
  }

  function _syncFields() {
    const chips = $("gen-fields");
    if (!chips || !g.plan) return;
    chips.innerHTML = "";
    (g.plan.fields || []).forEach((f) => {
      const c = el("button", "gen-chip", "{" + f + "}");
      c.title = "Insert into the file name";
      c.addEventListener("click", () => {
        const input = $("gen-name");
        const at = input.selectionStart ?? input.value.length;
        const end = input.selectionEnd ?? at;
        const { text, caret } = _insertField(input.value, at, end, f);
        input.value = text;
        input.setSelectionRange(caret, caret);
        g.cfg.export.name = input.value;
        changed();
        input.focus();
      });
      chips.appendChild(c);
    });
  }

  // ── plan ─────────────────────────────────────────────────────────────────
  function schedulePlan() {
    clearTimeout(g.planTimer);
    g.planTimer = setTimeout(plan, 250);
  }

  function _form() {
    const form = P().buildGenerateForm();
    form.append("config", JSON.stringify(serverConfig()));
    return form;
  }

  async function plan() {
    if (!g.fn || !P().state.compiled) return;
    const seq = ++g.planSeq;
    let data;
    try {
      const r = await fetch("/api/generate/plan/", { method: "POST", body: _form() });
      data = await r.json();
    } catch (e) {
      data = { error: String(e) };
    }
    if (seq !== g.planSeq) return;           // a newer plan is on its way
    const count = $("gen-run-count");
    const preview = $("gen-preview");
    preview.innerHTML = "";
    if (data.error) {
      g.plan = null;
      count.textContent = "";
      preview.appendChild(el("div", "gen-error", data.error));
      _syncStart();
      return;
    }
    g.plan = data;
    count.textContent = `${data.runs} run${data.runs === 1 ? "" : "s"}`;
    document.querySelectorAll(".gen-count").forEach((c) => {
      const axis = (data.axes || []).find((a) => a.key === c.dataset.key);
      c.textContent = axis ? `${axis.n} value${axis.n === 1 ? "" : "s"}` : "";
      c.title = axis ? axis.values.join(", ") + (axis.n > axis.values.length ? ", …" : "") : "";
    });
    if (data.missing) preview.appendChild(el("div", "gen-error", data.missing));
    preview.appendChild(el("div", "play-subhead", "First files"));
    if (data.name_error) preview.appendChild(el("div", "gen-error", data.name_error));
    const list = el("div", "gen-names");
    (data.names || []).forEach((n) => list.appendChild(el("div", "gen-name", n)));
    if (data.runs > (data.names || []).length) list.appendChild(el("div", "gen-hint", "…"));
    preview.appendChild(list);
    _syncInputsInfo();
    _syncFields();
    _syncStart();
  }

  // ── the job ──────────────────────────────────────────────────────────────
  function _running() { return g.job && g.job.state === "running"; }

  function _syncStart() {
    const btn = $("gen-start");
    const missing = P().missingRequired();
    const noOutput = !(g.cfg.export.targets || []).length;
    const blocked = !g.plan || !g.plan.runs || g.plan.missing || missing.length || noOutput
      || _running();
    btn.disabled = !!blocked;
    btn.textContent = g.plan && g.plan.runs ? `▶ Generate ${g.plan.runs}` : "▶ Generate";
    btn.title = noOutput ? "Add an output (or an activation) to export"
      : missing.length ? "Missing input: " + missing.join(", ")
      : g.plan && g.plan.missing ? g.plan.missing
      : _running() ? "A generation is running" : "Render every run of the plan to files";
    $("gen-cancel").style.display = _running() ? "" : "none";
  }

  async function start() {
    try {
      const r = await fetch("/api/generate/start/", { method: "POST", body: _form() });
      const data = await r.json();
      if (!r.ok || data.error) throw Object.assign(new Error(data.error || r.statusText), { payload: data });
      g.job = data;
      renderProgress();
      attach(data.id);
    } catch (e) {
      P().reportError("Could not start the generation", e);
    }
  }

  function attach(id) {
    clearInterval(g.poll);
    const tick = async () => {
      try {
        const r = await fetch("/api/generate/status/?id=" + encodeURIComponent(id));
        g.job = await r.json();
      } catch (_) { return; }
      renderProgress();
      _syncStart();
      if (!_running()) clearInterval(g.poll);
    };
    tick();
    g.poll = setInterval(tick, 700);
  }

  async function cancel() {
    if (!g.job) return;
    try {
      await P().postJSON("/api/generate/cancel/", { id: g.job.id });
    } catch (e) { P().reportError("Could not stop the generation", e); }
  }

  function renderProgress() {
    const box = $("gen-progress");
    const j = g.job;
    if (!j || j.state === "none") { box.style.display = "none"; return; }
    box.style.display = "";
    box.innerHTML = "";
    const pct = j.total ? Math.round(100 * j.done / j.total) : 0;
    const bar = el("div", "gen-bar");
    const fill = el("div", "gen-bar-fill");
    fill.style.width = pct + "%";
    bar.appendChild(fill);
    box.appendChild(bar);
    const eta = j.eta != null && j.state === "running" ? ` · ~${Math.ceil(j.eta)} s left` : "";
    const words = { running: "running", done: "done", cancelled: "stopped", error: "failed" };
    box.appendChild(el("div", "gen-status",
      `${words[j.state] || j.state} · ${j.done}/${j.total} runs · ${j.files} files · ${j.elapsed} s${eta}`));
    const dir = el("div", "gen-dir", j.dir);
    dir.title = "Output folder (with manifest.json)";
    box.appendChild(dir);
    if (j.message) box.appendChild(el("div", "gen-error", j.message));
    (j.errors || []).slice(-3).forEach((e) => {
      if (e.error !== j.message) box.appendChild(el("div", "gen-error",
        (e.index != null ? `run ${e.index}: ` : "") + e.error));
    });
    if (j.recent && j.recent.length) {
      const list = el("div", "gen-names");
      j.recent.forEach((f) => list.appendChild(el("div", "gen-name", f)));
      box.appendChild(list);
    }
    if (!_running() && j.files) {
      const ex = el("button", "play-mini-btn", "◎ open in audio explorer");
      ex.title = "Map these generations by similarity, in a new tab (computed in the background; it takes a while)";
      ex.addEventListener("click", () => explore(j.dir, false));
      const re = el("button", "play-mini-btn", "↻ recompute");
      re.title = "Restart the explorer and ignore its caches";
      re.addEventListener("click", () => explore(j.dir, true));
      const row = el("div", "gen-param-row");
      row.appendChild(ex);
      row.appendChild(re);
      box.appendChild(row);
    }
  }

  async function explore(dir, recompute) {
    // opened now, while this is still a click, or the browser blocks it
    const tab = window.open("about:blank", "_blank");
    try {
      const data = await P().postJSON("/api/generate/explore/", { dir, restart: recompute, recompute });
      if (data.error) throw new Error(data.error);
      const url = "/generate/explore/?dir=" + encodeURIComponent(dir);
      if (tab) tab.location = url; else window.open(url, "_blank");
    } catch (e) {
      if (tab) tab.close();
      P().reportError("Could not start the audio explorer", e);
    }
  }

  // ── wiring ───────────────────────────────────────────────────────────────
  async function onCompiled(e) {
    const fn = (e.detail && e.detail.fn) || P().state.fn;
    if (fn !== g.fn) {
      g.fn = fn;
      g.cfg = _load();
      g.nodeInfo = {};             // node names repeat across methods
    }
    await loadParams();
    renderParams();
    renderCombine();
    renderExport();
    plan();
  }

  function init() {
    document.addEventListener("tb:compiled", onCompiled);
    document.addEventListener("tb:inputs", schedulePlan);
    $("gen-add-param").addEventListener("change", (e) => {
      const p = _param(e.target.value);
      e.target.selectedIndex = 0;
      if (!p) return;
      g.cfg.params.push(_newSpec(p));
      renderParams();
      changed();
    });
    $("gen-start").addEventListener("click", start);
    $("gen-cancel").addEventListener("click", cancel);
    // a job started before this page was opened: follow it
    fetch("/api/generate/status/").then((r) => r.json()).then((j) => {
      if (j && j.state && j.state !== "none") {
        g.job = j;
        renderProgress();
        if (j.state === "running") attach(j.id);
      }
    }).catch(() => {});
  }

  document.addEventListener("DOMContentLoaded", init);
})();
