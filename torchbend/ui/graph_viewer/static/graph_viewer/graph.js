"use strict";

// ─── colour palette per op type ──────────────────────────────────────────────
const OP_COLORS = {
    placeholder:   "#4A90D9",
    call_module:   "#5CB85C",
    call_function: "#F0AD4E",
    call_method:   "#9B59B6",
    get_attr:      "#45B39D",
    output:        "#E74C3C",
};

const DEFAULT_COLOR = "#555";

// ─── state ───────────────────────────────────────────────────────────────────
let cy = null;
let currentFn = INITIAL_FN;
let currentGraphData = null;
let _recurrenceMap = {};  // target → count of nodes sharing that target
let _navIdx = -1;   // arrow-key navigation index (module-scope so tap handler can reset it)
// The sidebar activation search doubles as a navigation scope: while a query is
// active, arrow navigation walks only its matches and the graph shows which
// nodes those are. null means "no query" — navigation covers everything.
let _searchMatchSet = null;
let _searchQuery = "";
// assigned from inside the navigation block below
let _navResetForScope = () => {};
let _navGoToNode = null;
const collapsedSections = {};
// inputSets[name] = [{id, type:'file'|'expr', file?, expr?, label, valid}]
const inputSets = {};
// selectedIds[name] = id of currently selected input (null if none)
const selectedIds = {};
let currentVizData = null;
let currentVizNode = null;   // full node data of what's currently shown in viz section
let currentVizActId = null;  // effective node id for activation fetches (output node → its feeder)
let currentSelectedNode = null;
let _vizSectionCleanup = null;
let _pruneUnreachable = true;

// ─── hand-placed nodes ────────────────────────────────────────────────────────
// Dagre lays the graph out, but a graph anyone has spent time reading has been
// nudged into shape by hand, and losing that on every re-render (a retrace, a
// bending, a fold toggled) makes the arranging pointless. Positions are kept
// per method and travel with the rest of the client state.
let _nodePositions = {};        // fn -> {nodeId: {x, y}}

function _savedPositions(fn) {
    return _nodePositions[fn || currentFn] || null;
}

function _rememberPosition(fn, id, pos) {
    const key = fn || currentFn;
    if (!key) return;
    (_nodePositions[key] || (_nodePositions[key] = {}))[id] = { x: pos.x, y: pos.y };
    _scheduleClientStatePush();
    _syncResetLayoutBtn();
}

function _forgetPositions(fn) {
    delete _nodePositions[fn || currentFn];
    _scheduleClientStatePush();
    _syncResetLayoutBtn();
}

// Put back whatever was moved by hand. Applied after the layout runs and before
// the module panes are drawn, so a pane wraps the node where the user left it.
function _applySavedPositions() {
    const saved = _savedPositions(currentFn);
    if (!saved) return 0;
    let n = 0;
    cy.batch(() => {
        Object.entries(saved).forEach(([id, pos]) => {
            const el = cy.$id(id);
            if (el.length && !el.data("is_compound")) { el.position(pos); n++; }
        });
    });
    return n;
}

function _syncResetLayoutBtn() {
    const btn = document.getElementById("reset-layout-btn");
    if (!btn) return;
    const n = Object.keys(_savedPositions(currentFn) || {}).length;
    btn.style.display = n ? "" : "none";
    btn.title = `Put the ${n} hand-placed node${n === 1 ? "" : "s"} back where the layout wants them`;
}

// ─── graph simplification ─────────────────────────────────────────────────────
// An ATen trace is mostly bookkeeping: symbolic sizes, tuple destructuring and
// runs of views.  These switches decide how much of it the canvas shows.  They
// are presentational only — the trace, and every node's bendability, are
// untouched; folded-away nodes stay reachable through the details panel.
let _simplify = { shape_calc: true, unpack: true, collapse: true };
let _expandedChains = new Set();

// ─── alias nodes ──────────────────────────────────────────────────────────────
// A node consumed from all over the graph (an attention mask, a shared weight)
// throws one long edge per consumer and drags the layout apart.  Rather than
// draw those edges, we draw the node again next to each distant consumer.  The
// copies are canvas-only: the graph data, and everything keyed off it, still
// sees one node.
const _ALIAS_MIN_FANOUT = 6;    // below this a node is not worth duplicating
const _ALIAS_MIN_SPAN   = 25;   // ranks apart before an edge counts as "long"
let _aliasEnabled = true;
let _aliasOf      = new Map();  // alias element id → the node it stands for
let _aliasesFor   = new Map();  // node id → its alias element ids
let _aliasPlan    = { nodes: [], reroute: new Map() };
let _logicalAdj   = { up: new Map(), down: new Map() };

// The id of the node an element stands for — itself, unless it is a copy.
function _realId(id) { return _aliasOf.get(id) || id; }

// Every element drawn for one logical node: the node plus its copies.
function _elesFor(id) {
    let col = cy.$id(id);
    (_aliasesFor.get(id) || []).forEach(a => { col = col.union(cy.$id(a)); });
    return col;
}

function _planAliases(data) {
    _aliasOf = new Map();
    _aliasesFor = new Map();
    const plan = { nodes: [], reroute: new Map() };
    if (!_aliasEnabled) return plan;

    const order = {}, byId = {};
    data.nodes.forEach(n => { if (!n.is_compound) { order[n.id] = n.order; byId[n.id] = n; } });

    const outgoing = new Map();
    data.edges.forEach(e => {
        if (!outgoing.has(e.source)) outgoing.set(e.source, []);
        outgoing.get(e.source).push(e);
    });

    outgoing.forEach((edges, src) => {
        if (edges.length < _ALIAS_MIN_FANOUT || !byId[src]) return;
        const far = edges.filter(e =>
            Math.abs((order[e.target] ?? 0) - (order[src] ?? 0)) > _ALIAS_MIN_SPAN);
        // one distant consumer is just one edge — leave the node where it is
        if (far.length < 2) return;
        far.forEach((e, i) => {
            const aliasId = `${src}#a${i}`;
            const d = { ...byId[src], id: aliasId, is_alias: true, alias_of: src,
                        alias_consumer: e.target,
                        label: "⇢ " + (byId[src].chain_label || byId[src].label || src) };
            delete d.parent;
            plan.nodes.push(d);
            plan.reroute.set(e.id, aliasId);
            _aliasOf.set(aliasId, src);
            if (!_aliasesFor.has(src)) _aliasesFor.set(src, []);
            _aliasesFor.get(src).push(aliasId);
        });
    });
    return plan;
}

// Ancestry has to be read off the data, not the canvas: once an edge is routed
// through a copy, cytoscape's own predecessors()/successors() no longer see it.
function _buildLogicalAdj(data) {
    const up = new Map(), down = new Map();
    (data.edges || []).forEach(e => {
        if (!down.has(e.source)) down.set(e.source, []);
        down.get(e.source).push({ to: e.target, edge: e.id });
        if (!up.has(e.target)) up.set(e.target, []);
        up.get(e.target).push({ to: e.source, edge: e.id });
    });
    _logicalAdj = { up, down };
}

function _logicalClosure(startId, dir) {
    const adj = _logicalAdj[dir] || new Map();
    const nodes = new Set(), edges = new Set(), stack = [startId];
    while (stack.length) {
        const cur = stack.pop();
        (adj.get(cur) || []).forEach(({ to, edge }) => {
            edges.add(edge);
            if (!nodes.has(to)) { nodes.add(to); stack.push(to); }
        });
    }
    return { nodes, edges };
}

function _paintClosure(closure, nodeClass, edgeClass) {
    closure.nodes.forEach(id => _elesFor(id).removeClass("dimmed").addClass(nodeClass));
    closure.edges.forEach(id => cy.$id(id).removeClass("dimmed").addClass(edgeClass));
}

// ─── depth mode ───────────────────────────────────────────────────────────────
// A graph of every ATen operation answers "what runs"; a graph of modules
// answers "what is this model". Depth mode shows the second, folding anything
// deeper than the chosen level into the module that contains it, and letting
// you go inside one to see its own graph.
let _moduleDepth = "auto";   // "auto" | 0 (off) | 1..4
let _scope = "";          // the module the view is inside

function _setScope(path) {
    _scope = path || "";
    loadGraph(currentFn);
}

function _renderScopeBar() {
    const bar = document.getElementById("graph-scope-bar");
    if (!bar) return;
    const active = _moduleDepth === "auto"
        ? !!(currentGraphData && (currentGraphData.module_depth || currentGraphData.scope))
        : !!_moduleDepth;
    if (!active) { bar.style.display = "none"; bar.innerHTML = ""; return; }
    bar.style.display = "";
    bar.innerHTML = "";

    const crumb = (label, path, current) => {
        const b = document.createElement("button");
        b.className = "scope-crumb" + (current ? " current" : "");
        b.textContent = label;
        if (!current) {
            b.title = path ? `Go up to ${path}` : "Go back to the whole model";
            b.addEventListener("click", () => _setScope(path));
        }
        bar.appendChild(b);
    };

    const parts = _scope ? _scope.split(".") : [];
    crumb(currentModelName || "model", "", parts.length === 0);
    parts.forEach((part, i) => {
        const sep = document.createElement("span");
        sep.className = "scope-sep";
        sep.textContent = "›";
        bar.appendChild(sep);
        crumb(part, parts.slice(0, i + 1).join("."), i === parts.length - 1);
    });

    const hint = document.createElement("span");
    hint.className = "scope-hint";
    const chosen = currentGraphData ? currentGraphData.module_depth : null;
    hint.textContent = (_moduleDepth === "auto" && chosen != null
        ? `auto depth ${chosen} · ` : "") + "double-click a module to go inside";
    bar.appendChild(hint);
}

function _graphQuery() {
    const p = new URLSearchParams({
        prune:      _pruneUnreachable   ? 1 : 0,
        shape_calc: _simplify.shape_calc ? 1 : 0,
        unpack:     _simplify.unpack     ? 1 : 0,
        collapse:   _simplify.collapse   ? 1 : 0,
    });
    if (_expandedChains.size) p.set("expand", [..._expandedChains].join(","));
    p.set("depth", String(_moduleDepth));
    if (_scope) p.set("scope", _scope);
    return p.toString();
}

// ─── the whole trace, not just what is drawn ──────────────────────────────────
// Depth mode folds a module's contents into one node, and the simplification
// toggles hide whole classes of node. Those nodes still exist, and someone
// looking for one by name has no way of knowing which fold is hiding it. So the
// search reads this index — every node in the trace — and a match that is not
// currently on screen offers to go and open the view that contains it.
let _nodeIndex    = [];     // full node list for the model+method below
let _nodeIndexKey = null;   // which model and method it was fetched for
let _pendingSelect = null;  // node to select once the next graph render lands
let _actSearchingNow = false;  // is the browser actually filtering anything?

// The index belongs to one model's one method. Keying it on the method alone
// would survive a model switch — every model has a `forward` — and the search
// would then be looking through the previous model's nodes.
function _nodeIndexKeyFor(fn) { return `${currentModelName || ""}\u0000${fn}`; }

// Anything that re-traces changes the node set out from under the index.
function _invalidateNodeIndex() { _nodeIndex = []; _nodeIndexKey = null; }

// Resolves to true when it actually fetched — the caller only needs to redraw
// the list on the load that brought the index in.
function _fetchNodeIndex(fn) {
    const key = _nodeIndexKeyFor(fn);
    if (_nodeIndexKey === key && _nodeIndex.length) return Promise.resolve(false);
    return fetch(`/api/node_index/${fn}/`)
        .then(r => (r.ok ? r.json() : { nodes: [] }))
        .then(d => { _nodeIndex = d.nodes || []; _nodeIndexKey = key; return true; })
        .catch(() => { _invalidateNodeIndex(); return false; });
}

// A node shown on behalf of something else says so, rather than appearing under
// a bare internal name nobody asked for.
function _vizLabelFor(nodeName) {
    const d = currentVizNode;
    if (!d || nodeName === d.id) return nodeName;
    if (d.is_module_group) return `${d.group_label || d.module_path} → ${nodeName}`;
    if (d.op === "output") return `output ← ${nodeName}`;
    return nodeName;
}

// Which real node's activation stands for this one.
//
// A module group is a stand-in: it has no tensor, and asking the server for
// `__group__decoder.layers` gets "no activation for ...". What it produces is
// the member read from outside it, which the serializer names in
// `output_nodes`. Everything else stands for itself.
function _vizNodeFor(d) {
    if (!d) return null;
    if (d.is_module_group) {
        const outs = d.output_nodes || [];
        return outs.length ? outs[outs.length - 1] : null;
    }
    return d.id;
}

// The method's real placeholders, whatever the view is showing.
//
// The inputs belong to the method, not to the part of it you happen to be
// looking at, so the payload carries them separately from the drawn nodes:
// scoping into a submodule or folding one away must not change what the bench
// offers. The older paths stay as fallbacks for a payload without the list --
// deriving the inputs from what is drawn once concluded, inside a module, that
// every configured input was stale and deleted it.
function _modelPlaceholders(data) {
    const listed = (data && data.placeholders) || [];
    if (listed.length) return listed;
    const indexed = (_nodeIndexKey === _nodeIndexKeyFor(currentFn))
        ? _nodeIndex.filter(n => n.op === "placeholder") : [];
    if (_scope) return indexed;
    const drawn = ((data && data.nodes) || [])
        .filter(n => n.op === "placeholder" && !n.is_boundary);
    return drawn.length ? drawn : indexed;
}

// Nodes in the trace that the current view does not draw.
function _offviewNodes(data) {
    if (!data || !_nodeIndex.length) return [];
    const drawn = new Set();
    (data.nodes || []).forEach(n => {
        drawn.add(n.id);
        // a collapsed chain stands for its members, a merged producer for the
        // getitems it swallowed — neither is really missing
        (n.members || []).forEach(m => drawn.add(m.name));
        (n.merged  || []).forEach(m => drawn.add(m.name));
    });
    return _nodeIndex.filter(n => !drawn.has(n.id))
                     .map(n => Object.assign({}, n, { _offview: true }));
}

// Where a node lives, in words — for the row hint and the confirmation.
function _offviewWhere(n) {
    return n.module_path ? `inside ${n.module_path}` : "at the top level";
}

// What the view has to change for `n` to be on screen. Each entry is something
// the user is told about before it happens: this reaches into their view.
function _offviewPlan(n) {
    const scope = n.module_path || "";
    const steps = [];
    if (scope !== _scope) {
        steps.push({ text: scope ? `go inside ${scope}` : "go back to the whole model",
                     apply: () => { _scope = scope; } });
    }
    if (n.kind === "shape_calc" && _simplify.shape_calc) {
        steps.push({ text: "show shape arithmetic",
                     apply: () => { _simplify.shape_calc = false; } });
    }
    if (n.kind === "layout" && _simplify.collapse) {
        steps.push({ text: "expand collapsed chains",
                     apply: () => { _simplify.collapse = false; } });
    }
    if (n.kind === "unpack" && _simplify.unpack) {
        steps.push({ text: "separate merged outputs",
                     apply: () => { _simplify.unpack = false; } });
    }
    // Nothing above explains it and pruning is on: the node has no path to an
    // output — an argument the trace never used, most often — and only showing
    // unreachable nodes will bring it back.
    if (!steps.length && _pruneUnreachable) {
        steps.push({ text: "show nodes that reach no output",
                     apply: () => {
                         _pruneUnreachable = false;
                         const btn = document.getElementById("prune-btn");
                         if (btn) btn.classList.remove("active");
                     } });
    }
    return { scope, steps };
}

// Go to a node the current view does not draw, after asking.
function _goToOffviewNode(n) {
    const plan = _offviewPlan(n);
    const what = plan.steps.length
        ? plan.steps.map(st => "  · " + st.text).join("\n")
        : "  · reload the view";
    const ok = confirm(
        `'${n.label}' is ${_offviewWhere(n)}, which this view does not show.\n\n`
        + `To get there the view will:\n${what}\n\nGo there?`);
    if (!ok) return false;
    plan.steps.forEach(st => st.apply());
    _pendingSelect = n.id;
    _syncSimplifyCheckboxes();
    loadGraph(currentFn);
    return true;
}

// The options panel has to agree with what a jump just changed under it.
function _syncSimplifyCheckboxes() {
    Object.entries(_simplify).forEach(([key, val]) => {
        const el = document.getElementById("gopt-simplify-" + key);
        if (el) el.checked = val;
    });
}

// ─── node info overlay state ──────────────────────────────────────────────────
let _goptNodeInfoMode  = "hover";   // "none" | "hover" | "permanent"
let _goptNodeInfoShape = true;

// ─── bending state ────────────────────────────────────────────────────────────
let _bendingBindings      = [];        // active bindings cached from server
let _bendingParams        = [];        // BendingParameter objects cached from server
let _syncBatchEnabled     = false;
let _isBatchSyncing       = false;
let _availableCallbacks   = null;      // callback descriptors, fetched once
let _bendingUpdateMode    = "auto";    // "auto" | "live" | "manual"
let _bendingAutoThreshMs  = 500;       // threshold for auto→manual switch
let _bendingLastMs        = null;      // last activation-capture duration
let _bendingDebounceTimer = null;      // debounce for slider updates
const _BEND_DEBOUNCE_MS   = 120;
// node being targeted by the open bend dialog
let _bendDialogNode       = null;

// ─── in-flight computation tracking ───────────────────────────────────────────
// Activations are recomputed on every macro / param move, and on a slow model
// that takes a while: without a signal the UI just shows stale numbers. Every
// refresh wave goes through here — it aborts the wave still in flight (a newer
// macro value makes its result obsolete anyway), marks the affected cards as
// waiting, and lights a global indicator until everything settles.
let _computeAbort = null;      // AbortController of the current wave
let _computeRuns  = 0;         // number of requests in flight

function _computeSignal() {
    if (_computeAbort) _computeAbort.abort();
    _computeAbort = new AbortController();
    return _computeAbort.signal;
}
function _isAbort(err) {
    return !!err && (err.name === "AbortError" || err.code === 20);
}
function _syncComputeIndicator() {
    const el = document.getElementById("compute-indicator");
    if (el) el.classList.toggle("on", _computeRuns > 0);
}
function _computeBegin() { _computeRuns++; _syncComputeIndicator(); }
function _computeEnd()   { _computeRuns = Math.max(0, _computeRuns - 1); _syncComputeIndicator(); }

// mark a card / panel as "waiting for a value" (CSS dims it and shows a spinner)
function _setBusy(el, on) { if (el) el.classList.toggle("is-computing", !!on); }
function _setPinsBusy(pins, on) {
    pins.forEach(p => _setBusy(document.getElementById(`pin-card-${p.id}`), on));
}

// One refresh wave: detail panel + weight pins + activation pins. Cancels the
// previous wave, so a slider drag never leaves a stale response to land last.
async function _refreshLiveViews() {
    const signal = _computeSignal();
    const t0 = Date.now();
    _computeBegin();
    try {
        const refreshes = [];
        if (currentVizNode) {
            if (currentVizNode.op === "get_attr")
                refreshes.push(fetchWeight(currentFn, currentVizNode.id, signal));
            else if (currentVizActId && hasAnyInput())
                refreshes.push(fetchActivation(currentFn, currentVizActId, signal));
        }
        refreshes.push(_refreshWeightPins(currentFn, signal));
        if (hasAnyInput()) refreshes.push(_refreshActivationPins(currentFn, signal));
        await Promise.all(refreshes);
    } catch (err) {
        if (!_isAbort(err)) throw err;
    } finally {
        _computeEnd();
    }
    return Date.now() - t0;
}

// ─── per-model input persistence ──────────────────────────────────────────────
// perModelInputs[modelName] = { sets: {...}, selected: {...} }  (expr-only entries)
const perModelInputs = {};
let currentModelName = (() => {
    const m = (typeof INITIAL_MODELS !== "undefined" ? INITIAL_MODELS : []).find(m => m.current);
    return m ? m.name : null;
})();

function _saveModelInputs(name) {
    if (!name) return;
    _saveBenchNow(name);        // the shared bench, complete, for this model
    // persist only expr-type entries (File objects can't survive a model switch anyway)
    const sets = {};
    Object.entries(inputSets).forEach(([k, arr]) => {
        const kept = arr.filter(e => e.type === "expr");
        if (kept.length) sets[k] = kept.map(e => ({ ...e }));
    });
    perModelInputs[name] = { sets, selected: { ...selectedIds } };
}

function _restoreModelInputs(name) {
    Object.keys(inputSets).forEach(k => delete inputSets[k]);
    Object.keys(selectedIds).forEach(k => delete selectedIds[k]);
    const saved = name && perModelInputs[name];
    if (!saved) return;
    Object.assign(inputSets, saved.sets || {});
    Object.assign(selectedIds, saved.selected || {});
}

// ─── shared input store (for play mode) ───────────────────────────────────────
// Mirror the current inputs to localStorage so the play page can reuse them —
// expressions verbatim, and loaded files as data URLs (capped, to respect the
// localStorage quota). Written on every input change; read by play.js.
const INPUT_STORE_PREFIX = "tb_inputs_";
const _MAX_PERSIST_FILE_BYTES = 4 * 1024 * 1024;   // 4 MB / file
let _persistInputsTimer = null;

// A snapshot of the shared store taken *before* client-state restore can rewrite
// it: server client-state carries only `expr` entries, so a loaded sound (a
// `file` entry) survives just here. `_restoreSharedInputs` reads this back so a
// graph → play → graph round trip does not drop the file.
let _sharedInputSnapshot = null;

async function _dataUrlToFile(dataUrl, name, mime) {
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    return new File([blob], name || "input", { type: mime || blob.type });
}

// Refill placeholder slots from the shared store that client-state could not
// carry — `file` entries, and `mode` entries. Runs after client-state is applied
// and before loadGraph, so `_applyDefaultInputs` sees the slot already filled.
async function _restoreSharedInputs(name) {
    const store = _sharedInputSnapshot;
    _sharedInputSnapshot = null;
    if (!name || !store) return;
    const hadAny = hasAnyInput();
    Object.keys(inputSets).forEach(k => delete inputSets[k]);
    Object.keys(selectedIds).forEach(k => delete selectedIds[k]);
    let lost = 0;
    for (const [key, arr] of Object.entries(store.sets || {})) {
        const entries = [];
        for (const saved of (arr || [])) {
            const n = await TBBench.fromSaved(saved);
            if (n.lost) { lost += 1; continue; }
            const e = _graphFromNeutral(n);
            // re-derive the processed file (crop, resample…) from its settings
            // before anything runs on it
            try {
                if (e.file && window.TBImage && TBInputs.isImageFile(e.file)) {
                    await TBImage.init(e, null); await TBImage.process(e);
                } else if (e.file && window.TBAudio && TBAudio.isAudioFile(e.file) && e.type === "file") {
                    await TBAudio.init(e, null); await TBAudio.process(e);
                }
            } catch (_) { /* the raw file still works */ }
            entries.push(e);
        }
        inputSets[key] = entries;
    }
    for (const [key, id] of Object.entries(store.selected || {})) {
        if ((inputSets[key] || []).some(e => e.id === id)) selectedIds[key] = id;
    }
    for (const [key, arr] of Object.entries(inputSets)) {
        if (!selectedIds[key] && arr.length) selectedIds[key] = arr[0].id;
    }
    _benchBatch.on = !!(store.batch && store.batch.on);
    _benchBatch.mode = (store.batch && store.batch.mode) || "pad";
    _syncBenchBatchBar();
    if (lost) showToast("warn", `${lost} loaded file${lost > 1 ? "s" : ""} could not be carried over (too large to keep) — load ${lost > 1 ? "them" : "it"} again`);
    _runAfterBatch(hadAny);
    _notifyInputsRestored();
}

function _fileToDataUrl(f) {
    return new Promise((res, rej) => {
        const r = new FileReader();
        r.onload  = () => res(r.result);
        r.onerror = rej;
        r.readAsDataURL(f);
    });
}

function _persistInputsForPlay(name) {
    if (!name) return;
    clearTimeout(_persistInputsTimer);
    _persistInputsTimer = setTimeout(() => _doPersistInputsForPlay(name), 200);
}

async function _doPersistInputsForPlay(name) {
    // files first get their data URL (kept on the File), so any later save —
    // including the one made on the way out of the page — carries them
    const files = [];
    Object.values(inputSets).forEach(arr => arr.forEach(e => { if (e.file) files.push(e.file); }));
    await Promise.all(files.map(f => TBBench.ensureDataUrl(f)));
    _saveBenchNow(name);
}

// entry as the graph holds it ⇄ the bench's neutral form (bench.js)
function _graphToNeutral(e) {
    const meta = {};
    TBBench.MEDIA_KEYS.forEach(k => { if (e[k] !== undefined) meta[k] = e[k]; });
    return { id: e.id, type: e.type, value: e.type === "expr" ? e.expr : (e.value || ""),
             label: e.label, mode: e.mode || null, inBatch: e.inBatch, meta, file: e.file || null };
}

function _graphFromNeutral(n) {
    // a scalar (play's widget for a plain argument) is an expression here
    const type = n.type === "scalar" ? "expr" : n.type;
    const e = { id: n.id, type, label: n.label, valid: true };
    if (n.inBatch === false) e.inBatch = false;
    if (type === "expr") e.expr = String(n.value);
    if (type === "mode") { e.mode = n.mode; e.value = n.value || ""; }
    if (n.file) e.file = n.file;
    Object.assign(e, n.meta || {});
    if (!e.label) e.label = type === "expr" ? (e.expr.length > 45 ? e.expr.slice(0, 45) + "…" : e.expr)
                                            : (n.file ? n.file.name : String(n.value));
    return e;
}

let _benchSaveWarned = false;
function _saveBenchNow(name) {
    if (!name) return;
    const sets = {};
    for (const [k, arr] of Object.entries(inputSets))
        sets[k] = arr.filter(e => e.valid).map(e => TBBench.toSaved(_graphToNeutral(e)));
    const ok = TBBench.save(name, {
        sets, selected: { ...selectedIds },
        batch: { on: _benchBatch.on, mode: _benchBatch.mode },
    }, Object.keys(inputSets));
    if (!ok && !_benchSaveWarned) {
        _benchSaveWarned = true;
        showToast("warn", "Inputs are too large to keep between pages — play mode will not see them all");
    }
}

// ─── per-model pin persistence ────────────────────────────────────────────────
const perModelPins = {};

function _saveModelPins(name) {
    if (!name) return;
    perModelPins[name] = { pinPages, currentPinPage };
}

function _persistPinPages() {
    if (!currentModelName) return;
    try {
        const toSave = pinPages.map(page =>
            page.map(({ data, originalData, ...rest }) => rest)
        );
        localStorage.setItem(`tb_pin_pages_${currentModelName}`, JSON.stringify({ pages: toSave, current: currentPinPage }));
    } catch (_) {}
    _scheduleClientStatePush();
}

function _loadPinPages(name) {
    if (!name) return;
    _loadVizStatesLocal(name);         // the cards' view states come with them
    try {
        const raw = localStorage.getItem(`tb_pin_pages_${name}`);
        if (!raw) return;
        const saved = JSON.parse(raw);
        const restored = (saved.pages || []).map(page =>
            page.map(pin => ({ ...pin, data: null, originalData: null }))
        );
        if (restored.length > 0) {
            pinPages = restored;
            currentPinPage = Math.min(saved.current || 0, pinPages.length - 1);
        }
    } catch (_) {}
}

// A dashboard restored in a new session: the server has forgotten the views its
// cards were set to (they last a session, or need sync), the pins have not.
// Tell it again, once per model, before the cards fetch their activations.
let _pinViewsRestoredFor = null;
async function _restorePinViews() {
    if (_pinViewsRestoredFor === currentModelName) return;
    // not before the dashboard itself is back: an empty one proves nothing
    if (!pinPages.some(page => page.length)) return;
    _pinViewsRestoredFor = currentModelName;
    const posts = [];
    pinPages.forEach(page => page.forEach(pin => {
        if (!pin.view || !pin.view.view || !pin.label) return;
        posts.push(fetch(`/api/views/${encodeURIComponent(currentFn)}/${encodeURIComponent(pin.label)}/`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(pin.view),
        }).catch(() => {}));
    }));
    await Promise.all(posts);
}

async function _refetchNullPins() {
    const fn = currentFn;
    if (!fn) return;
    await _restorePinViews();
    const hasNull = pinPages.some(page =>
        page.some(p => p.data === null && !p.phNodeId && p.type !== "bending" && p.type !== "bp")
    );
    if (!hasNull) return;
    _refreshWeightPins(fn);
    if (hasAnyInput()) _refreshActivationPins(fn);
}

function _restoreModelPins(name) {
    if (name && perModelPins[name]) {
        pinPages = perModelPins[name].pinPages;
        currentPinPage = perModelPins[name].currentPinPage;
    } else {
        pinPages = [[]];
        currentPinPage = 0;
        _loadPinPages(name);
    }
    renderPinTabs();
    renderCurrentPage();
    _updatePinCount();
}

// ─── placeholder input observer registry ─────────────────────────────────────
const _phInputObservers = {};

function _notifyPhInputObservers(nodeId) {
    (_phInputObservers[nodeId] || []).forEach(cb => cb());
}

// ─── input state helpers ──────────────────────────────────────────────────────
function genId() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function hasAnyInput() {
    return Object.values(inputSets).some(arr => arr.some(i => i.valid));
}

// ─── bench batch (as in play mode) ────────────────────────────────────────────
// Off: each input runs on its one selected entry. On: every entry left checked
// is stacked along the batch dimension, reconciled server-side as play mode
// does (pad / loop / stack). An entry's `inBatch === false` keeps it on the
// bench without putting it in the batch.
// Shared with play mode through the bench store (bench.js): read when the page
// restores its inputs, written with them.
const _benchBatch = { on: false, mode: "pad" };

// Whether this page stacks entries. Play's `sequential` runs one pass per
// entry; the graph draws one pass, so there it runs the selected entry.
function _batchActive() {
    return _benchBatch.on && _benchBatch.mode !== "sequential" && _batchSupported();
}

// The interface can say a method takes one input at a time (its
// `_batch_compat_`): the shared batch setting then does not apply here.
function _batchSupported() {
    return !currentGraphData || currentGraphData.batch_supported !== false;
}

function _batchEntries(name) {
    return (inputSets[name] || []).filter(i => i.valid && i.inBatch !== false);
}

function _syncBenchBatchBar() {
    const on = document.getElementById("bench-batch-on");
    const mode = document.getElementById("bench-batch-mode");
    const note = document.getElementById("bench-batch-note");
    const supported = _batchSupported();
    if (on) {
        on.checked = _benchBatch.on && supported;
        on.disabled = !supported;
        on.parentElement.title = supported
            ? "Batch: every checked entry of an input is stacked along the batch dimension (as in play mode). Off: the selected entry alone."
            : `${currentFn} takes one input at a time (its interface says so): the selected entry runs`;
    }
    if (mode) { mode.value = _benchBatch.mode; mode.disabled = !_benchBatch.on || !supported; }
    if (note && !supported) {
        note.textContent = "this method takes one input at a time";
    } else if (note) {
        const n = Math.max(0, ...Object.keys(inputSets).map(k => _batchEntries(k).length));
        note.textContent = !_benchBatch.on ? ""
            : _benchBatch.mode === "sequential" ? "sequential: the graph runs the selected entry"
            : (n > 1 ? `batch of ${n}` : "one entry each");
    }
}

// A batch change is an input change: show it on what is on display now.
function _benchBatchChanged() {
    _persistInputsForPlay(currentModelName);    // shared with play mode
    _syncBenchBatchBar();
    if (currentGraphData && !document.getElementById("input-panel").classList.contains("hidden"))
        buildInputPanel(currentGraphData);
    if (hasAnyInput()) _refreshLiveViews();
}

function buildActivationForm(nodes) {
    const form = new FormData();
    if (_batchActive()) {
        form.append("batch", "1");
        form.append("batch_mode", _benchBatch.mode);
        for (const name of Object.keys(inputSets)) {
            const entries = _batchEntries(name);
            if (!entries.length) continue;
            // An input takes either its mode's entries (prompts, recordings —
            // the interface encodes them, all together) or plain ones
            // (expressions, files); the server cannot mix the two readings.
            const modes = entries.filter(e => e.type === "mode");
            if (modes.length) {
                modes.forEach(e => form.append(name, e.file || e.value));
                form.append("__mode__" + name, modes[0].mode);
            } else {
                entries.forEach(e => form.append(name, e.type === "file" ? (e.croppedFile || e.file) : e.expr));
                // files and typed values reach the server in separate lists; this
                // is how it puts them back in the bench's order (batch index i =
                // the i-th row)
                form.append("__order__" + name, entries.map(e => e.type === "file" ? "f" : "e").join(""));
            }
        }
        if (nodes && nodes.length) form.append("nodes", JSON.stringify(nodes));
        return form;
    }
    for (const [name, arr] of Object.entries(inputSets)) {
        const selId = selectedIds[name];
        const inp = arr.find(i => i.id === selId && i.valid) || arr.find(i => i.valid);
        if (!inp) continue;
        if (inp.type === 'file') {
            form.append(name, inp.croppedFile || inp.file);
        } else if (inp.type === 'mode') {
            // the value itself, plus a marker saying how the server should read
            // it — an interface encoder rather than an expression. An audio
            // mode may hold a loaded file instead of typed text; the server
            // decodes it and hands the encoder (waveform, rate).
            form.append(name, inp.file || inp.value);
            form.append('__mode__' + name, inp.mode);
        } else {
            form.append(name, inp.expr);
        }
    }
    if (nodes && nodes.length) form.append('nodes', JSON.stringify(nodes));
    return form;
}

function _notifyInputsRestored() {
    // re-render visible input UI after server restore.
    // Guard: if currentFn is null the new model's graph hasn't loaded yet —
    // skip the panel rebuild to avoid rebuilding with stale previous-model graph data.
    if (currentSelectedNode) renderPlaceholderInputsSidebar(currentSelectedNode);
    updateRetraceBtn();
    if (currentFn && currentGraphData && !document.getElementById("input-panel").classList.contains("hidden"))
        buildInputPanel(currentGraphData);
}

// `opts.select`: should this entry become the one the model runs on? True for
// anything the user adds by hand — typing a prompt and then watching the graph
// keep running on the seeded default is not a choice anyone made. False for the
// seeded defaults themselves, which only claim the slot if it is still empty.
// `opts.noRun`: part of a batch (seeding defaults, restoring a bench) — the
// caller retraces once when the whole batch is in. Retracing on the first
// entry would send a form holding that entry alone, and a model with several
// inputs fails the trace on the ones not there yet ("missing argument").
function setInputEntry(name, entry, opts) {
    const select = !opts || opts.select !== false;
    if (entry.file) TBBench.ensureDataUrl(entry.file);   // ready for the next save
    const hadAny = hasAnyInput();
    if (!inputSets[name]) inputSets[name] = [];
    const idx = inputSets[name].findIndex(i => i.id === entry.id);
    const isNew = idx < 0;
    if (idx >= 0) {
        inputSets[name][idx] = entry;
    } else {
        inputSets[name].push(entry);
        if (entry.valid && (select || !selectedIds[name])) selectedIds[name] = entry.id;
    }
    if (currentSelectedNode === name) renderPlaceholderInputsSidebar(name);
    _notifyPhInputObservers(name);
    updateRetraceBtn();
    _persistInputsForPlay(currentModelName);
    _scheduleClientStatePush();
    if (isNew && currentGraphData && !document.getElementById("input-panel").classList.contains("hidden"))
        buildInputPanel(currentGraphData);
    // Re-run when the input the model uses changes: the first valid input added
    // (so activations appear immediately), and any later one the user picks in
    // its place — otherwise the graph still shows the previous input's numbers.
    const nowSelected = selectedIds[name] === entry.id;
    if (opts && opts.noRun) return;
    if (entry.valid && currentFn && (!hadAny || (isNew && select && nowSelected)))
        retrace(currentFn);
}

// After a batch seeded with `noRun`: the one run the first entry would have
// triggered, now that every entry is in.
function _runAfterBatch(hadAnyBefore) {
    if (!hadAnyBefore && hasAnyInput() && currentFn) retrace(currentFn);
}

function removeInputEntry(name, id) {
    if (!inputSets[name]) return;
    inputSets[name] = inputSets[name].filter(i => i.id !== id);
    if (selectedIds[name] === id) {
        const next = inputSets[name].find(i => i.valid);
        selectedIds[name] = next ? next.id : null;
    }
    if (currentSelectedNode === name) renderPlaceholderInputsSidebar(name);
    if (currentGraphData && !document.getElementById("input-panel").classList.contains("hidden")) {
        buildInputPanel(currentGraphData);
    }
    _notifyPhInputObservers(name);
    updateRetraceBtn();
    _persistInputsForPlay(currentModelName);
    _scheduleClientStatePush();
}

function selectInput(name, id) {
    selectedIds[name] = id;
    if (currentSelectedNode === name) renderPlaceholderInputsSidebar(name);
    _notifyPhInputObservers(name);
    _persistInputsForPlay(currentModelName);
    _scheduleClientStatePush();
}

function addFileInput(nodeId, file, opts) {
    const entry = { id: genId(), type: 'file', file, label: file.name, valid: true };
    setInputEntry(nodeId, entry, opts);
    if (currentGraphData && !document.getElementById("input-panel").classList.contains("hidden")) {
        buildInputPanel(currentGraphData);
    }
    if (_isAudioFile(file))  _initAudioCrop(nodeId, entry);
    if (_isImageFile(file))  _initImagePreview(nodeId, entry);
}

function _isAudioFile(file) { return TBAudio.isAudioFile(file); }

function _isImageFile(file) { return TBInputs.isImageFile(file); }

// ─── audio crop ───────────────────────────────────────────────────────────────
// The decode / crop / resample machinery lives in inputs.js (TBAudio) so the
// editor and play mode process a dropped file identically.
async function _initAudioCrop(nodeId, entry) {
    try {
        const node = currentGraphData && currentGraphData.nodes.find(n => n.id === nodeId);
        await TBAudio.init(entry, node && node.shape);
        await _applyCrop(entry);
        if (currentGraphData && !document.getElementById("input-panel").classList.contains("hidden"))
            buildInputPanel(currentGraphData);
    } catch (e) { /* unsupported format, skip crop UI */ }
}

async function _applyCrop(entry) {
    await TBAudio.process(entry);
    _persistInputsForPlay(currentModelName);
}

// ─── image preview + resize ───────────────────────────────────────────────────
function _detectImageChannels(img) {
    const sw = Math.min(img.naturalWidth, 64), sh = Math.min(img.naturalHeight, 64);
    const sc = document.createElement('canvas');
    sc.width = sw; sc.height = sh;
    const sCtx = sc.getContext('2d');
    sCtx.drawImage(img, 0, 0, sw, sh);
    const px = sCtx.getImageData(0, 0, sw, sh).data;
    let hasColor = false, hasAlpha = false;
    for (let i = 0; i < px.length; i += 4) {
        if (px[i] !== px[i + 1] || px[i] !== px[i + 2]) hasColor = true;
        if (px[i + 3] < 255) hasAlpha = true;
        if (hasColor && hasAlpha) break;
    }
    return hasAlpha ? 4 : (hasColor ? 3 : 1);
}

function _initImagePreview(nodeId, entry) {
    const url = URL.createObjectURL(entry.file);
    const img = new Image();
    img.onload = () => {
        entry.srcW = img.naturalWidth;
        entry.srcH = img.naturalHeight;
        entry.thumbUrl = url;

        // Auto-detect actual channel count from pixel data
        const detectedChannels = _detectImageChannels(img);

        // default target size from node shape: shape = [B, C, H, W]
        const node = currentGraphData && currentGraphData.nodes.find(n => n.id === nodeId);
        const sh = node && node.shape;
        const nodeChannels = (sh && sh.length >= 3) ? sh[sh.length - 3] : null;
        // Prefer node shape channels if available; fall back to auto-detected
        entry.imgChannels = nodeChannels || detectedChannels;
        entry.imgTargetW  = (sh && sh.length >= 1) ? sh[sh.length - 1] : img.naturalWidth;
        entry.imgTargetH  = (sh && sh.length >= 2) ? sh[sh.length - 2] : img.naturalHeight;

        // crop region defaults: full image
        entry.cropX = 0; entry.cropY = 0;
        entry.cropW = img.naturalWidth; entry.cropH = img.naturalHeight;

        _applyImageProcess(entry).then(() => {
            if (currentGraphData && !document.getElementById("input-panel").classList.contains("hidden"))
                buildInputPanel(currentGraphData);
        });
    };
    img.onerror = () => URL.revokeObjectURL(url);
    img.src = url;
}

async function _applyImageProcess(entry) {
    if (!entry || !entry.file) return;
    // Shared processor (crop/resize mode, target size, channels) lives in inputs.js
    await window.TBImage.process(entry);
    _persistInputsForPlay(currentModelName);
}

// ─── Cytoscape init ───────────────────────────────────────────────────────────
function initCytoscape() {
    cy = cytoscape({
        container: document.getElementById("cy"),
        elements: [],
        style: [
            {
                selector: "node",
                style: {
                    // a collapsed run reads as `view→transpose` on the canvas,
                    // but its `label` stays its real node name — that is what
                    // favourites, lists and the activation rows key on
                    "label": (ele) => ele.data("chain_label") || ele.data("label"),
                    "text-valign": "center",
                    "text-halign": "center",
                    "background-color": (ele) => OP_COLORS[ele.data("op")] || DEFAULT_COLOR,
                    "color": "#fff",  // node label stays white — sits on coloured background
                    "font-size": "12px",
                    "font-family": "'SF Mono', 'Fira Code', 'Consolas', monospace",
                    "padding": "12px",
                    "width": "label",
                    "height": "label",
                    "shape": "roundrectangle",
                    "border-width": 0,
                    "min-width": "76px",
                    "min-height": "36px",
                    "text-wrap": "wrap",
                    "text-max-width": "170px",
                    "transition-property": "opacity, border-width, border-color",
                    "transition-duration": "150ms",
                    "z-index": 10,
                },
            },
            // ── packed loops (`tb.loop`) ──────────────────────────────────────
            // One node standing for many iterations of a body the graph does
            // not draw: coloured apart, and badged with how many and which, so
            // it reads as a loop and not as one more op. Its carry outputs are
            // outlined in the same colour. Placed before the bending/search
            // rules, so their borders still win on top of it.
            {
                selector: "node[?is_loop]",
                style: {
                    "label": (ele) => `${ele.data("chain_label") || ele.data("label")}\n${ele.data("loop_badge")}`,
                    "background-color": "#E8833A",
                    "border-width": 4,
                    "border-style": "double",
                    "border-color": "#9c4a12",
                    "font-size": "11px",
                    "text-max-width": "240px",
                },
            },
            // ── annotated values (`mark(description=…, **meta)`) ──────────────
            // A ✎ line under the label says what the value is; the full text
            // is on hover, in the details panel and in the sidebar's process.
            {
                selector: "node[annot]",
                style: {
                    "label": (ele) => _labelWithBadges(ele),
                    "text-max-width": "240px",
                },
            },
            {
                selector: "node[annot][!is_loop]",
                style: { "border-width": 2, "border-color": "#7a5cc4" },
            },
            {
                selector: "node[?is_loop_carry]",
                style: {
                    "label": (ele) => `${ele.data("chain_label") || ele.data("label")}\n${ele.data("loop_badge")}`,
                    "border-width": 2,
                    "border-style": "dashed",
                    "border-color": "#E8833A",
                    "font-size": "10px",
                },
            },
            // search scope: matches lift, the rest recede — the query is a
            // navigation scope, so it has to be visible on the graph itself
            {
                selector: "node.search-match",
                style: {
                    "border-width": 3,
                    "border-color": "#2c7be5",
                    "z-index": 30,
                },
            },
            {
                selector: "node.search-dim",
                style: { "opacity": 0.25 },
            },
            {
                selector: "edge.search-dim",
                style: { "opacity": 0.12 },
            },
            // bending indicator: gold border
            {
                selector: "node[?has_bending]",
                style: {
                    "border-width": 3,
                    "border-color": "#FFD700",
                    "border-style": "solid",
                },
            },
            // macro indicator: accent-blue dashed border (controlled by BendingParameter)
            {
                selector: "node[?has_macro]",
                style: {
                    "border-width": 3,
                    "border-color": "#2c7be5",
                    "border-style": "dashed",
                },
            },
            // both bent and macro: purple solid border
            {
                selector: "node[?has_bending][?has_macro]",
                style: {
                    "border-width": 4,
                    "border-color": "#a855f7",
                    "border-style": "solid",
                },
            },
            // ── arrow-key navigation: prev / next neighbours ───────────────
            {
                selector: ".nav-prev",
                style: {
                    "border-width":  2,
                    "border-color":  "#74c7ec",
                    "border-style":  "dashed",
                    "z-index":       50,
                    "opacity":       1,
                },
            },
            {
                selector: ".nav-next",
                style: {
                    "border-width":  2,
                    "border-color":  "#a6e3a1",
                    "border-style":  "dashed",
                    "z-index":       50,
                    "opacity":       1,
                },
            },
            // ── arrow-key navigation: current node ────────────────────────
            {
                selector: ".nav-current",
                style: {
                    "border-width":  3,
                    "border-color":  "#000000",
                    "border-style":  "solid",
                    "z-index":       100,
                    "opacity":       1,
                },
            },
            // ── show-bent mode ─────────────────────────────────────────────
            {
                selector: ".show-bent-dim",
                style: { "opacity": 0.07 },
            },
            {
                selector: ".show-bent-focus",
                style: {
                    "border-width":  5,
                    "border-color":  "#FFD700",
                    "border-style":  "solid",
                    "background-color": "#1a1600",
                    "color":         "#FFD700",
                    "font-size":     "12px",
                    "z-index":       100,
                    "opacity":       1,
                },
            },
            {
                selector: ".show-bent-dim-edge",
                style: { "opacity": 0.04 },
            },
            {
                selector: ".show-bent-focus-edge",
                style: {
                    "line-color":          "#b8860b",
                    "target-arrow-color":  "#b8860b",
                    "opacity":             0.75,
                    "width":               2,
                },
            },
            {
                selector: "edge",
                style: {
                    "width": 1.5,
                    "line-color": "#b0b0be",
                    "target-arrow-color": "#b0b0be",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                    "opacity": 0.9,
                    "transition-property": "opacity, line-color, target-arrow-color, width",
                    "transition-duration": "150ms",
                },
            },
            // ── the model's inputs, reached through what the scope dropped ────
            // Dashed, because the value does get there but not untouched: the
            // ops between them and the block are outside the view.
            {
                selector: "edge[?is_indirect]",
                style: {
                    "line-style": "dashed",
                    "line-dash-pattern": [4, 4],
                    "opacity": 0.55,
                    "width": 1,
                },
            },
            // ── alias copies ──────────────────────────────────────────────────
            // Same colour as the original so it reads as the same value, but
            // smaller and dashed so it reads as a reference, not a second node.
            {
                selector: "node[?is_alias]",
                style: {
                    "font-size": "10px",
                    "padding": "6px",
                    "min-width": "48px",
                    "min-height": "24px",
                    "border-width": 1.5,
                    "border-color": "#ffffff",
                    "border-style": "dotted",
                    "background-opacity": 0.6,
                    "shape": "roundrectangle",
                    "z-index": 5,
                },
            },
            // ── ports: what feeds this block, and what reads it ───────────────
            // Outside the scope, so drawn as an outline rather than a solid —
            // present for context, not part of what you are looking at.
            {
                selector: "node[?is_boundary]",
                style: {
                    "background-opacity": 0.12,
                    "border-width": 1.5,
                    "border-style": "dashed",
                    "border-color": "#8a94a6",
                    "color": "#8a94a6",
                    "font-size": "10px",
                    "shape": "round-rectangle",
                },
            },
            // ── the model's inputs, kept inside a submodule view ──────────────
            // They belong to the method, not to the block being shown, so they
            // keep the placeholder colour and say so with a dashed outline.
            {
                selector: "node[?is_global_input]",
                style: {
                    "border-width": 1.5,
                    "border-style": "dashed",
                    "border-color": "#8a94a6",
                },
            },
            // ── folded modules (depth mode) ───────────────────────────────────
            // Bigger and squarer than an operation, because it stands for many
            // of them; the count says how many.
            {
                selector: "node[?is_module_group]",
                style: {
                    "label": (ele) => `${ele.data("group_label")}\n${ele.data("n_members")} nodes`,
                    "text-wrap": "wrap",
                    "background-color": "#5CB85C",
                    "background-opacity": 0.85,
                    "shape": "round-rectangle",
                    "border-width": 2,
                    "border-color": "#3d8b3d",
                    "font-size": "13px",
                    "padding": "16px",
                    "min-width": "110px",
                    "min-height": "54px",
                },
            },
            // a folded module whose inside was annotated: say what it produces
            {
                selector: "node[?is_module_group][annot]",
                style: {
                    "label": (ele) => {
                        const annot = ele.data("annot");
                        return `${ele.data("group_label")}\n${ele.data("n_members")} nodes\n✎ `
                            + (annot.length > 42 ? annot.slice(0, 41) + "…" : annot);
                    },
                    "border-color": "#7a5cc4",
                    "border-width": 3,
                },
            },
            // ── collapsed layout chains ───────────────────────────────────────
            // A run of pure reshaping drawn as one node.  Dashed to say "there
            // is more inside"; the label spells out the run (view→transpose).
            {
                selector: "node[?is_chain]",
                style: {
                    "border-width": 2,
                    "border-color": "#8a94a6",
                    "border-style": "dashed",
                    "background-opacity": 0.55,
                    "font-size": "10px",
                    "shape": "round-octagon",
                },
            },
            {
                selector: "node[?is_chain].chain-expanded",
                style: { "border-style": "solid", "border-color": "#4a9eff" },
            },
            // ── module pane nodes (added post-layout, not Cytoscape compound parents) ──
            {
                selector: "node[?is_compound]",
                style: {
                    "width": 80,
                    "height": 40,
                    "background-color": "#b8c8d8",
                    "background-opacity": 0.10,
                    "border-width": 1,
                    "border-color": "#b0bcc8",
                    "border-style": "dashed",
                    "label": "data(label)",
                    "text-valign": "top",
                    "text-halign": "center",
                    "font-size": "9px",
                    "color": "#9aaabb",
                    "font-family": "'SF Mono', 'Fira Code', 'Consolas', monospace",
                    "shape": "roundrectangle",
                    "text-margin-y": 12,
                    "z-index": 0,
                    "cursor": "default",
                    "events": "no",
                    "transition-property": "background-opacity, border-width, border-color, color",
                    "transition-duration": "350ms",
                },
            },
            {
                selector: "node[?is_compound].compound-highlight",
                style: {
                    "background-opacity": 0.28,
                    "border-width": 2.5,
                    "border-color": "#4a9eff",
                    "border-style": "solid",
                    "color": "#4a9eff",
                },
            },
            // ── nodes with no shape data (only active when inputs are loaded) ─
            {
                selector: ".no-shape",
                style: { "opacity": 0.45 },
            },
            // ── highlight classes ──────────────────────────────────────────
            {
                selector: ".dimmed",
                style: { "opacity": 0.2 },
            },
            {
                selector: ".selected",
                style: {
                    "border-width": 4,
                    "border-color": "#1c1c1e",
                    "border-style": "solid",
                    "opacity": 1,
                },
            },
            {
                selector: ".ancestor-node",
                style: {
                    "border-width": 3,
                    "border-color": "#3498DB",
                    "border-style": "solid",
                    "opacity": 1,
                },
            },
            {
                selector: ".descendant-node",
                style: {
                    "border-width": 3,
                    "border-color": "#27AE60",
                    "border-style": "solid",
                    "opacity": 1,
                },
            },
            {
                selector: ".ancestor-edge",
                style: {
                    "line-color": "#3498DB",
                    "target-arrow-color": "#3498DB",
                    "width": 2.5,
                    "opacity": 1,
                },
            },
            {
                selector: ".descendant-edge",
                style: {
                    "line-color": "#27AE60",
                    "target-arrow-color": "#27AE60",
                    "width": 2.5,
                    "opacity": 1,
                },
            },
        ],
        layout: { name: "preset" },
        wheelSensitivity: 0.3,
    });

    function _navClear() {
        _navIdx = -1;
        ["graph-nav-panel", "graph-nav-code"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = "none";
        });
        const panel = document.getElementById("graph-nav-panel");
        if (panel) panel.classList.remove("code-open");
        _clearNavLocators();
    }

    cy.on("tap", "node", (evt) => {
        const node = evt.target;
        if (node.data("is_compound")) return;
        _navClear();
        onNodeClick(node);
    });
    // `dragfree` rather than `position`: the latter fires for every animation
    // frame of the layout itself, which would save dagre's output as if it were
    // the user's own arrangement.
    cy.on("dragfree", "node", (evt) => {
        const node = evt.target;
        if (node.data("is_compound")) return;
        _rememberPosition(currentFn, node.id(), node.position());
        addModulePanes();      // the panes are bounding boxes; they have moved
    });
    cy.on("dbltap", "node", (evt) => {
        const node = evt.target;
        // a folded module opens into its own graph
        if (node.data("is_module_group")) {
            _setScope(node.data("module_path"));
            return;
        }
        // a copy is a signpost: double-tap to go where it points
        if (!node.data("is_alias")) return;
        const orig = cy.$id(node.data("alias_of"));
        if (orig.length) cy.animate({ center: { eles: orig }, zoom: cy.zoom() }, { duration: 300 });
    });
    cy.on("tap", (evt) => {
        if (evt.target === cy) {
            clearSelection();
            _navClear();
        }
    });
    cy.on("cxttap", "node", (evt) => {
        const oe = evt.originalEvent;
        if (oe) oe.preventDefault();
        let menuEvt;
        if (oe && oe.clientX != null) {
            menuEvt = oe;
        } else {
            const rp = evt.target.renderedPosition();
            const bb = cy.container().getBoundingClientRect();
            menuEvt = { clientX: bb.left + rp.x, clientY: bb.top + rp.y };
        }
        _showNodeContextMenu(menuEvt, evt.target.data());
    });
    cy.container().addEventListener("contextmenu", evt => evt.preventDefault());
}

// ─── load & render ────────────────────────────────────────────────────────────
function loadGraph(fn) {
    showLoading(true);
    currentFn = fn;

    document.querySelectorAll(".method-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.fn === fn);
    });
    document.getElementById("method-label").textContent = fn;
    applyDeviceCompat(fn);

    fetch(`/api/graph/${fn}/?${_graphQuery()}`)
        .then((r) => {
            if (!r.ok) return r.json().then((d) => { { const _e = new Error(d.error || r.statusText); _e.traceback = d.traceback || ""; throw _e; } });
            return r.json();
        })
        .then((data) => {
            renderGraph(data);
            showLoading(false);
            _refetchNullPins();
            // The index is what the search looks through; it outlives any one
            // view, so it is fetched alongside rather than blocking the render.
            _fetchNodeIndex(fn).then(fetched => {
                if (!fetched) return;
                renderActivationList(currentGraphData);
                // the bench falls back to the index when the payload carries no
                // placeholder list, so it is only now that it can be complete
                if (!document.getElementById("input-panel").classList.contains("hidden"))
                    buildInputPanel(currentGraphData);
            });
            if (_pendingSelect) {
                const want = _pendingSelect;
                _pendingSelect = null;
                const el = cy.$id(want);
                if (el && el.length) {
                    cy.animate({ center: { eles: el }, zoom: cy.zoom() }, { duration: 250 });
                    onNodeClick(el);
                } else {
                    showToast("warn", `'${want}' is still not drawn in this `
                        + `view — try turning off the simplification toggles.`);
                }
            }
            // Re-apply bendings to the freshly-rendered graph. The graph data
            // already carries server-side has_bending, but this also reconciles
            // the binding list, pin cards and macro styling from the live cache
            // — fixing the reload case where bendings rendered before the graph.
            _fetchBendings();
        })
        .catch((err) => {
            console.error("graph load error:", err);
            const el = document.getElementById("loading");
            el.style.display = "flex";
            el.style.flexDirection = "column";
            el.style.gap = "8px";
            el.innerHTML = `<span style="color:#E74C3C;font-size:14px">⚠ Error loading graph</span><span style="font-size:11px;color:#aaa">${err.message}</span>`;
        });
}

function _applyDefaultInputs(data) {
    const placeholders = _modelPlaceholders(data);
    const phKeys = new Set(placeholders.map(ph => ph.id));
    // Nothing to compare against yet (the index has not landed and the view
    // draws no placeholders): leave the configured inputs alone rather than
    // deleting them on no evidence.
    if (!phKeys.size) return;

    // Flush stale inputSets entries whose keys don't belong to this model's placeholders.
    // This cleans up any cross-model contamination from saved state.
    Object.keys(inputSets).forEach(k => {
        if (!phKeys.has(k)) {
            delete inputSets[k];
            delete selectedIds[k];
        }
    });

    if (!data.default_inputs || !Object.keys(data.default_inputs).length) return;
    const hadAny = hasAnyInput();
    placeholders.forEach(ph => {
        const key = ph.id;  // node data uses 'id', not 'name'
        const expr = data.default_inputs[key];
        if (!expr) return;
        if (inputSets[key] && inputSets[key].length > 0) return;
        setInputEntry(key, {
            id: genId(), type: "expr", expr,
            label: expr.length > 45 ? expr.slice(0, 45) + "…" : expr,
            valid: true,
        }, { select: false, noRun: true });
    });
    _runAfterBatch(hadAny);
}

function _computeRecurrenceMap(data) {
    const counts = {};
    if (!data || !data.nodes) return counts;
    data.nodes.forEach(n => {
        if (!n.target || n.is_compound) return;
        counts[n.target] = (counts[n.target] || 0) + 1;
    });
    return counts;
}

function renderGraph(data) {
    currentGraphData = data;
    _loadSnapshots();
    // trust the payload over local state: a scope the server could not honour
    // (a module that vanished on retrace) must not leave the bar lying
    if (data.scope !== undefined) _scope = data.scope || "";
    _renderScopeBar();
    _recurrenceMap = _computeRecurrenceMap(data);
    _updateNavScopeDisplay();
    const elements = [];

    _buildLogicalAdj(data);
    _aliasPlan = _planAliases(data);

    data.nodes.forEach((n) => {
        if (n.is_compound) return;  // pane nodes added post-layout via addModulePanes()
        const d = { ...n };
        delete d.parent;            // don't use Cytoscape compound parent-child (breaks dagre)
        // what mark() said about it (or about what is folded inside it)
        const annotated = (data.annotation_targets || {})[n.id];
        if (annotated && annotated.length)
            d.annot = annotated.map(k => _annotTitle((data.annotations || {})[k])).filter(Boolean).join(" · ") || "annotated";
        elements.push({ data: d });
    });
    _aliasPlan.nodes.forEach((d) => elements.push({ data: d }));

    const _OP_SHORT = { call_module: "mod", call_function: "fn", call_method: "meth", get_attr: "attr", placeholder: "ph" };
    data.edges.forEach((e) => {
        const arg = e.label || "";
        const shortOp = _OP_SHORT[e.target_op] || e.target_op || "";
        const fn = e.target_fn || "";
        const argPart = arg === "" ? "" : (isNaN(arg) ? `${arg}=` : `[${arg}]`);
        const opPart = [shortOp, fn].filter(Boolean).join(":");
        // which output of the producer this is, when several were merged into it
        const outPart = e.src_index == null ? "" : `out[${e.src_index}]`;
        const label = [outPart, argPart, opPart].filter(Boolean).join(" ");
        const source = _aliasPlan.reroute.get(e.id) || e.source;
        elements.push({ data: { id: e.id, source, target: e.target, label } });
    });

    cy.elements().remove();
    cy.add(elements);

    // Re-apply edge label style if checkbox is checked
    const edgeLabelCb = document.getElementById("gopt-edge-labels");
    if (edgeLabelCb && edgeLabelCb.checked) {
        cy.style().selector("edge").style({ "label": "data(label)", "font-size": "8px", "color": "#6e6e73", "text-background-opacity": 0, "text-margin-y": -6 }).update();
    }

    applyFilters();
    runLayout();
    updateStats(data);
    renderProcessList(data);
    renderActivationList(data);

    _applyDefaultInputs(data);
    _syncBenchBatchBar();         // this method may not batch

    // an expand was requested to reach a specific member — select it now that
    // the reloaded graph actually contains it
    if (_pendingSelectNode) {
        const want = _pendingSelectNode;
        _pendingSelectNode = null;
        const el = cy.$id(want);
        if (el.length) {
            cy.animate({ center: { eles: el }, zoom: cy.zoom() }, { duration: 250 });
            onNodeClick(el);
        }
    }

    if (!document.getElementById("input-panel").classList.contains("hidden")) {
        buildInputPanel(data);
    }
}

function runLayout() {
    const rankDir = document.getElementById("layout-select").value;
    // remove pane nodes before dagre runs — they'd confuse the layout
    cy.nodes('[?is_compound]').remove();
    cy.resize();

    // When permanent labels are on, add extra inter-node space to prevent pills from
    // overlapping neighbours.  Pills are max ~220px wide and ~45px tall.
    const permanent = _goptNodeInfoMode === "permanent";
    const nodeSep = permanent
        ? (rankDir === "LR" ? 85  : 210)  // TB: pills are wide, push nodes apart horizontally
        : 85;
    const rankSep = permanent
        ? (rankDir === "LR" ? 270 : 180)  // LR: pills extend right; TB: pills extend below
        : 135;

    let layout;
    try {
        layout = cy.layout({
            name: "dagre",
            rankDir,
            nodeSep,
            rankSep,
            edgeSep: 20,
            padding: 70,
            animate: false,  // positions must be final at layoutstop for boundingBox() to work
            fit: true,
        });
    } catch (e) {
        layout = cy.layout({
            name: "breadthfirst",
            directed: true,
            spacingFactor: 1.4,
            padding: 30,
            animate: false,
            fit: true,
        });
    }
    layout.run();
    _applySavedPositions();
    addModulePanes();
    cy.animate({ fit: { padding: 50 } }, { duration: 300, easing: "ease-out", complete: _updateNodeInfoOverlay });
    // re-apply show-bent highlighting after each layout (new graph data)
    if (_showBentMode) _applyShowBent();
}

// ─── module panes (post-layout bounding-box nodes) ────────────────────────────
function buildModuleGroups(data, maxDepth) {
    const compounds = {};
    const compoundParentMap = {};  // compound_id → parent compound_id

    data.nodes.forEach(n => {
        if (n.is_compound) {
            compounds[n.id] = compounds[n.id] || { label: n.label, module_path: n.module_path || "", directRealNodes: [], childCompounds: [] };
            compounds[n.id].label = n.label;
            compounds[n.id].module_path = n.module_path || "";
            if (n.parent) compoundParentMap[n.id] = n.parent;
        }
    });

    // Wire child-compound lists
    Object.entries(compoundParentMap).forEach(([childId, parentId]) => {
        if (compounds[parentId]) compounds[parentId].childCompounds.push(childId);
    });

    // Assign each real node to its direct parent compound
    data.nodes.forEach(n => {
        if (n.is_compound) return;
        if (n.parent && compounds[n.parent]) {
            compounds[n.parent].directRealNodes.push(n.id);
        }
    });

    // A copy is drawn beside its consumer, so it belongs to the consumer's pane;
    // putting it in the original's would stretch that pane across the graph.
    const parentOf = {};
    data.nodes.forEach(n => { if (!n.is_compound) parentOf[n.id] = n.parent; });
    _aliasPlan.nodes.forEach(a => {
        const pane = parentOf[a.alias_consumer];
        if (pane && compounds[pane]) compounds[pane].directRealNodes.push(a.id);
    });

    // Recursively collect all descendant real-node ids for each compound
    function allRealNodes(cid) {
        const c = compounds[cid];
        if (!c) return [];
        const ids = [...c.directRealNodes];
        c.childCompounds.forEach(childId => ids.push(...allRealNodes(childId)));
        return ids;
    }

    const groups = {};
    Object.keys(compounds).forEach(id => {
        if (maxDepth != null) {
            const depth = (compounds[id].module_path || "").split(".").length;
            if (depth > maxDepth) return;
        }
        const realNodes = allRealNodes(id);
        if (realNodes.length > 0) groups[id] = { label: compounds[id].label, realNodes };
    });
    return groups;
}

function getModuleDepth() {
    const sel = document.getElementById("depth-select");
    if (!sel) return null;
    const v = parseInt(sel.value, 10);
    return v === 0 ? null : v;
}

function addModulePanes() {
    if (!cy || !currentGraphData) return;
    // Idempotent: this runs after a layout, and again whenever a node is
    // dragged. Re-adding a pane whose id is already on the canvas is an error
    // in cytoscape, so clear them first rather than assume a clean slate.
    cy.nodes('[?is_compound]').remove();
    const groups = buildModuleGroups(currentGraphData, getModuleDepth());
    const PAD = 40;
    const rankDir = document.getElementById("layout-select").value;
    const isLR = rankDir === 'LR';

    const nodeOp = {};
    currentGraphData.nodes.forEach(n => { nodeOp[n.id] = n.op; });

    // ATTR_GAP: distance perpendicular to flow from consumer's near edge to weight node centre
    // ATTR_SPACING: stacking distance when several weight nodes share a consumer
    const ATTR_GAP = 80;
    const ATTR_SPACING = 28;

    Object.values(groups).forEach(group => {
        // "Weight" nodes = get_attr + any node whose ONLY predecessors are get_attr nodes
        // (e.g. aten.t.default for weight transpose).  Both kinds live at rank 0/1 in dagre
        // and would otherwise stretch the pane across multiple sibling modules.
        const attrSet = new Set(group.realNodes.filter(id => nodeOp[id] === 'get_attr'));
        const weightIds = group.realNodes.filter(id => {
            if (attrSet.has(id)) return true;
            const preds = cy.$id(id).incomers('node');
            return preds.length > 0 && preds.every(p => attrSet.has(p.data('id')));
        });
        if (weightIds.length === 0) return;

        // The "real ops" are everything else — used to find the consuming node
        const opSet = new Set(group.realNodes.filter(id => !weightIds.includes(id)));
        if (opSet.size === 0) return;

        // Map consumer → weight nodes feeding into it
        const consumerToWeights = {};
        weightIds.forEach(id => {
            const cyNode = cy.$id(id);
            if (!cyNode.length) return;
            let consumer = null;
            cyNode.outgoers('node').forEach(s => {
                if (!consumer && opSet.has(s.data('id'))) consumer = s;
            });
            if (!consumer || !consumer.length) return;
            const cid = consumer.data('id');
            consumerToWeights[cid] = consumerToWeights[cid] || [];
            consumerToWeights[cid].push(id);
        });

        Object.entries(consumerToWeights).forEach(([consumerId, ids]) => {
            const consumer = cy.$id(consumerId);
            if (!consumer.length) return;
            const cpos = consumer.position();
            const cbb  = consumer.boundingBox({});
            ids.forEach((id, i) => {
                const n = cy.$id(id);
                if (!n.length) return;
                const offset = (i - (ids.length - 1) / 2) * ATTR_SPACING;
                // Place PERPENDICULAR to the flow direction so weight nodes don't
                // land in the gap between sequential modules and inflate their panes.
                if (isLR) {
                    // Flow is left→right; stack weight nodes ABOVE the consumer
                    n.position({ x: cpos.x + offset, y: cbb.y1 - ATTR_GAP });
                } else {
                    // Flow is top→bottom; stack weight nodes LEFT of the consumer
                    n.position({ x: cbb.x1 - ATTR_GAP, y: cpos.y + offset });
                }
            });
        });
    });

    // Add pane nodes sorted largest-first (parents before children) so parent panes
    // render behind child panes in Cytoscape's paint order.
    const entries = Object.entries(groups).sort((a, b) => b[1].realNodes.length - a[1].realNodes.length);

    entries.forEach(([parentId, group]) => {
        const children = group.realNodes.map(id => cy.$id(id)).filter(n => n.length > 0);
        if (children.length === 0) return;

        let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
        children.forEach(n => {
            const bb = n.boundingBox({});
            x1 = Math.min(x1, bb.x1); y1 = Math.min(y1, bb.y1);
            x2 = Math.max(x2, bb.x2); y2 = Math.max(y2, bb.y2);
        });

        cy.add({
            data: { id: parentId, label: group.label, is_compound: true, op: "module" },
            position: { x: (x1 + x2) / 2, y: (y1 + y2) / 2 },
            style: {
                width:  Math.max(80, (x2 - x1) + PAD * 2),
                height: Math.max(40, (y2 - y1) + PAD * 2),
            },
        });
        cy.$id(parentId).lock().ungrabify();
    });

    updateNodeOpacity();
}

function _highlightModulePane(paneId) {
    const pane = cy.$id(paneId);
    if (!pane.length) return;
    pane.addClass("compound-highlight");
    cy.animate({ fit: { eles: pane, padding: 36 } }, { duration: 250 });
    clearTimeout(pane._hlTimer);
    pane._hlTimer = setTimeout(() => pane.removeClass("compound-highlight"), 900);
}

// ─── stats ────────────────────────────────────────────────────────────────────
function updateStats(data) {
    const real = data.nodes.filter(n => !n.is_compound);
    document.getElementById("stat-nodes").textContent = real.length;
    document.getElementById("stat-edges").textContent = data.edges.length;
    document.getElementById("stat-bent").textContent = real.filter(n => n.has_bending).length;
    _updateHiddenStats(data);
}

// What simplification removed, and the switch that brings each part back —
// otherwise a node's absence is indistinguishable from it never being traced.
function _updateHiddenStats(data) {
    const el = document.getElementById("graph-hidden");
    if (!el) return;
    const h = data.hidden_counts || {};
    const parts = [];
    if (h.shape_calc) parts.push({ key: "shape_calc", txt: `${h.shape_calc} shape` });
    if (h.unpack)     parts.push({ key: "unpack",     txt: `${h.unpack} unpack` });
    if (h.chains)     parts.push({ key: "collapse",   txt: `${h.layout + h.chains} in ${h.chains} chains` });
    if (!parts.length) { el.style.display = "none"; el.innerHTML = ""; return; }
    el.style.display = "";
    el.innerHTML = "folded: " + parts
        .map(p => `<button class="hidden-chip" data-key="${p.key}" title="Show these again">${p.txt}</button>`)
        .join(" · ");
    el.querySelectorAll(".hidden-chip").forEach(btn => {
        btn.addEventListener("click", () => _setSimplify(btn.dataset.key, false));
    });
}

function _setSimplify(key, on) {
    _simplify[key] = on;
    const cb = document.getElementById("gopt-simplify-" + key);
    if (cb) cb.checked = on;
    loadGraph(currentFn);
}

// ─── filters ──────────────────────────────────────────────────────────────────
function applyFilters() {
    if (!cy) return;
    const visible = new Set(
        Array.from(document.querySelectorAll("#op-filters input:checked")).map((i) => i.value)
    );
    cy.nodes().forEach((node) => {
        if (node.data("is_compound")) return;   // always show module panes
        node.style("display", visible.has(node.data("op")) ? "element" : "none");
    });
    // hide edges whose endpoints are hidden
    cy.edges().forEach((edge) => {
        const srcVisible = visible.has(edge.source().data("op"));
        const tgtVisible = visible.has(edge.target().data("op"));
        edge.style("display", srcVisible && tgtVisible ? "element" : "none");
    });
}

// ─── selection / highlight ────────────────────────────────────────────────────
function onNodeClick(node) {
    if (node.data("is_compound")) return;
    // clicking a copy selects what it stands for
    const id = _realId(node.data("id"));
    const real = cy.$id(id);
    const self = real.length ? real : node;

    clearHighlights();

    cy.elements().addClass("dimmed");

    // the node and every copy of it
    _elesFor(id).removeClass("dimmed").addClass("selected");

    // ancestry comes off the data, so it stays whole across rerouted edges
    _paintClosure(_logicalClosure(id, "up"),   "ancestor-node",   "ancestor-edge");
    _paintClosure(_logicalClosure(id, "down"), "descendant-node", "descendant-edge");

    // unhide edges directly connected to selected node
    _elesFor(id).connectedEdges().removeClass("dimmed");

    showDetails(self.data());

    const d = self.data();
    currentVizNode = d;
    if (d.op === "get_attr") {
        currentVizActId = null;
        fetchWeight(currentFn, d.id);
    } else if (d.op === "output") {
        // output is a sink; show the activation of the node that feeds into it
        const src = _elesFor(id).incomers("node").first();
        const srcId = src.length ? _vizNodeFor(src.data()) : null;
        if (src.length && src.data("op") === "get_attr") {
            currentVizActId = null;
            fetchWeight(currentFn, src.data("id"));
        } else if (srcId && hasAnyInput()) {
            currentVizActId = srcId;
            fetchActivation(currentFn, srcId);
        } else {
            currentVizActId = null;
            clearViz();
        }
    } else if (d.is_module_group) {
        // A folded module has no tensor of its own; what it produces is the
        // member anything outside it reads. Show that.
        const outId = _vizNodeFor(d);
        if (outId && hasAnyInput()) {
            currentVizActId = outId;
            fetchActivation(currentFn, outId);
        } else {
            currentVizActId = null;
            clearViz();
        }
    } else if (d.op === "placeholder") {
        currentVizActId = null;
        showPlaceholderVizPanel(d.id, d.label, d.shape);
    } else if (hasAnyInput()) {
        currentVizActId = d.id;
        fetchActivation(currentFn, d.id);
    } else {
        currentVizActId = null;
        clearViz();
    }
}

function clearHighlights() {
    if (!cy) return;
    cy.elements().removeClass(
        "dimmed selected ancestor-node descendant-node ancestor-edge descendant-edge nav-current nav-prev nav-next"
    );
}

function clearSelection() {
    clearHighlights();
    _closeNodeCtxMenu();
    hideDetails();
    clearViz();
}

// ─── details panel ────────────────────────────────────────────────────────────
// ─── annotations (mark(description=…, **meta)) ───────────────────────────────
// Everything a model's marks say about its values, as the serializer sends it:
// `annotations` by node, `annotation_targets` by drawn element. Read in graph
// order, they are the model's own account of its process.

function _annotTitle(a) {
    if (!a) return "";
    const meta = a.meta || {};
    if (meta.title) return String(meta.title);
    if (a.alias) return "#" + a.alias;
    const d = a.description || "";
    return d.length > 34 ? d.slice(0, 33) + "…" : d;
}

function _annotationsFor(id, data) {
    const g = data || currentGraphData || {};
    return ((g.annotation_targets || {})[id] || []).map(k => (g.annotations || {})[k]).filter(Boolean);
}

function _labelWithBadges(ele) {
    const lines = [ele.data("chain_label") || ele.data("label")];
    if (ele.data("loop_badge")) lines.push(ele.data("loop_badge"));
    const annot = ele.data("annot");
    if (annot) lines.push("✎ " + (annot.length > 42 ? annot.slice(0, 41) + "…" : annot));
    return lines.join("\n");
}

function _escAnnot(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function _fmtMetaValue(v) {
    if (v == null) return "—";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(+v.toPrecision(5));
    if (typeof v === "object") return JSON.stringify(v);
    return String(v);
}

// one annotation, as DOM: aliases, description, metadata
function _annotBlock(a, showNode) {
    const box = document.createElement("div");
    box.className = "annot-block";
    const head = document.createElement("div");
    head.className = "annot-head";
    (a.aliases || []).forEach(al => {
        const chip = document.createElement("span");
        chip.className = "annot-alias";
        chip.textContent = "#" + al;
        head.appendChild(chip);
    });
    if (a.meta && a.meta.title) {
        const t = document.createElement("span");
        t.className = "annot-title";
        t.textContent = a.meta.title;
        head.appendChild(t);
    }
    if (showNode) {
        const n = document.createElement("span");
        n.className = "annot-node";
        n.textContent = a.node;
        head.appendChild(n);
    }
    if (head.childNodes.length) box.appendChild(head);
    if (a.description) {
        const p = document.createElement("div");
        p.className = "annot-desc";
        p.textContent = a.description;
        box.appendChild(p);
    }
    const meta = Object.entries(a.meta || {}).filter(([k]) => k !== "title");
    if (meta.length) {
        const table = document.createElement("table");
        table.className = "annot-meta";
        meta.forEach(([k, v]) => {
            const tr = document.createElement("tr");
            const tk = document.createElement("td");
            tk.className = "annot-meta-key";
            tk.textContent = k;
            const tv = document.createElement("td");
            tv.textContent = _fmtMetaValue(v);
            tr.appendChild(tk);
            tr.appendChild(tv);
            table.appendChild(tr);
        });
        box.appendChild(table);
    }
    return box;
}

function renderDetailAnnotations(id) {
    const section = document.getElementById("detail-annotation-section");
    if (!section) return;
    section.innerHTML = "";
    const list = _annotationsFor(id);
    section.style.display = list.length ? "" : "none";
    if (!list.length) return;
    const title = document.createElement("div");
    title.className = "detail-section-title";
    title.textContent = list.length > 1 ? `annotations (${list.length})` : "annotation";
    section.appendChild(title);
    // on a module box, say which node inside each one is about
    list.forEach(a => section.appendChild(_annotBlock(a, a.node !== id)));
}

// The sidebar's account of the process: every annotated value, in the order
// the graph computes it (or by `step=` when the marks give one). Click to go.
function renderProcessList(data) {
    const section = document.getElementById("process-section");
    const host = document.getElementById("process-list");
    if (!section || !host) return;
    host.innerHTML = "";
    const all = Object.values((data && data.annotations) || {})
        .filter(a => a.description || a.alias || Object.keys(a.meta || {}).length);
    section.style.display = all.length ? "" : "none";
    if (!all.length) return;
    const hasSteps = all.some(a => a.meta && a.meta.step != null);
    const sorted = all.slice().sort((a, b) => hasSteps
        ? ((a.meta.step ?? 1e9) - (b.meta.step ?? 1e9)) || (a.order - b.order)
        : (a.order - b.order));
    sorted.forEach((a, i) => {
        const row = document.createElement("div");
        row.className = "process-step" + (a.drawn ? "" : " process-step-hidden");
        const num = document.createElement("span");
        num.className = "process-num";
        num.textContent = a.meta && a.meta.step != null ? a.meta.step : i + 1;
        row.appendChild(num);
        const body = document.createElement("div");
        body.className = "process-body";
        const t = document.createElement("div");
        t.className = "process-title";
        t.textContent = _annotTitle(a) || a.node;
        body.appendChild(t);
        if (a.description && _annotTitle(a) !== a.description) {
            const dsc = document.createElement("div");
            dsc.className = "process-desc";
            dsc.textContent = a.description;
            body.appendChild(dsc);
        }
        row.appendChild(body);
        row.title = (a.description ? a.description + "\n\n" : "")
            + (a.drawn ? `click: go to ${a.drawn}` : `${a.node} is not drawn in this view (pruned or hidden)`);
        if (a.drawn) row.addEventListener("click", () => {
            const el = cy.$id(a.drawn);
            if (!el.length) return;
            cy.animate({ center: { eles: el }, zoom: Math.max(cy.zoom(), 1.2) }, { duration: 300 });
            onNodeClick(el);
        });
        host.appendChild(row);
    });
}

function showDetails(data) {
    document.getElementById("details-empty").style.display = "none";
    document.getElementById("details-content").style.display = "block";

    // header
    const badge = document.getElementById("detail-op-badge");
    badge.textContent = data.op;
    badge.style.background = OP_COLORS[data.op] || DEFAULT_COLOR;

    document.getElementById("detail-name").textContent = data.label;

    // fields
    document.getElementById("detail-target").textContent = data.target || "—";

    const shapeEl = document.getElementById("detail-shape");
    if (data.shape && data.shape.length > 0) {
        shapeEl.textContent = `[${data.shape.join(", ")}]`;
    } else {
        shapeEl.textContent = "—";
    }

    // source location
    const srcRow = document.getElementById("source-row");
    const srcEl  = document.getElementById("detail-source");
    const _srcOps = ["call_function", "call_module", "call_method", "get_attr"];
    if (_srcOps.includes(data.op)) {
        if (data.source_file) {
            const base = data.source_file.replace(/.*[/\\]/, "");
            srcEl.textContent = `${base}:${data.source_line}`;
            srcEl.title = `${data.source_file}:${data.source_line}${data.source_fn ? " (" + data.source_fn + ")" : ""} — click to view`;
        } else {
            srcEl.textContent = data.target || "view source";
            srcEl.title = "click to view source";
        }
        srcEl.style.cursor = "pointer";
        srcEl.onclick = () => openSourceModal(currentFn, data.id);
        srcRow.style.display = "";
    } else {
        srcRow.style.display = "none";
        srcEl.onclick = null;
        srcEl.style.cursor = "";
    }

    // bending callbacks (read-only static display in table)
    const bendRow = document.getElementById("bending-row");
    if (data.has_bending && data.bending_callbacks && data.bending_callbacks.length > 0) {
        bendRow.style.display = "";
        document.getElementById("detail-bending").textContent =
            data.bending_callbacks.join("\n");
    } else {
        bendRow.style.display = "none";
    }

    // interactive bending ops section
    renderBendingOpsSection(data);

    // tags / aliases
    renderDetailTags(data.id || data.label);
    renderDetailAnnotations(data.id || data.label);

    // what this node stands in for, if anything
    renderNodeContents(data);

    // args (inputs)
    renderArgs(data.args || []);
    renderConsumers(data.id);

    // configured inputs (for placeholder nodes)
    currentSelectedNode = data.id;
    const phSec = document.getElementById("placeholder-inputs-section");
    if (phSec) phSec.style.display = "none";
    if (data.op === "placeholder") renderPlaceholderInputsSidebar(data.id);
}

// Mirrors node_kinds.op_name on the server: 'aten::sym_size.int' -> 'sym_size'.
function _opName(target) {
    if (!target) return "";
    if (target.includes("::")) return target.split("::")[1].split(".")[0];
    return target.split(".").pop();
}

// A folded node hides real, individually bendable nodes behind it.  The details
// panel is where they stay reachable: a chain lists its members, an unpacked
// producer lists the outputs that were merged into it.
function renderNodeContents(data) {
    const section = document.getElementById("detail-contents-section");
    const list = document.getElementById("detail-contents");
    const title = document.getElementById("detail-contents-title");
    if (!section || !list) return;

    const rows = [];
    if (data.is_chain && data.members) {
        title.textContent = `collapsed chain · ${data.members.length} nodes`;
        data.members.forEach((m, i) => rows.push({
            name: m.name,
            op: _opName(m.target),
            shape: m.shape,
            step: i + 1,
        }));
    } else if (data.merged && data.merged.length) {
        title.textContent = `merged outputs · ${data.merged.length}`;
        data.merged.forEach(m => rows.push({
            name: m.name,
            op: `[${m.index}]`,
            shape: m.shape,
        }));
    } else if (data.is_module_group && (data.output_nodes || []).length) {
        // What the folded module produces: the members anything outside reads.
        // Named rather than implied, because a module with two results has two
        // and the panel can only be showing one of them.
        const outs = data.output_nodes;
        title.textContent = outs.length > 1
            ? `outputs of ${data.group_label} · ${outs.length}`
            : `output of ${data.group_label}`;
        outs.forEach(name => {
            const n = (_nodeIndex || []).find(x => x.id === name);
            rows.push({ name, op: n ? (n.op_name || n.op) : "", shape: n ? n.shape : null });
        });
    }

    if (!rows.length) { section.style.display = "none"; list.innerHTML = ""; return; }
    section.style.display = "";
    list.innerHTML = "";

    rows.forEach((r) => {
        const div = document.createElement("div");
        div.className = "content-item";
        const a = document.createElement("a");
        a.href = "#";
        a.className = "node-link";
        a.textContent = r.name;
        a.title = "Select this node — it stays bendable on its own";
        a.addEventListener("click", (e) => {
            e.preventDefault();
            _selectFoldedNode(r.name, data);
        });
        div.appendChild(a);
        const meta = document.createElement("span");
        meta.className = "content-meta";
        meta.textContent = [r.op, r.shape ? `[${r.shape.join(", ")}]` : null]
            .filter(Boolean).join(" ");
        div.appendChild(meta);
        list.appendChild(div);
    });

    if (data.is_chain) {
        const btn = document.createElement("button");
        btn.className = "gopt-mini";
        btn.style.marginTop = "6px";
        btn.textContent = "expand in graph";
        btn.addEventListener("click", () => _expandChain(data.id));
        list.appendChild(btn);
    }
}

// Clicking a folded member: if it is drawn, go to it; otherwise open the chain
// that swallowed it and select it once the graph comes back.
function _selectFoldedNode(name, parentData) {
    const el = cy.$id(name);
    if (el.length) {
        cy.animate({ center: { eles: el }, zoom: cy.zoom() }, { duration: 250 });
        onNodeClick(el);
        return;
    }
    if (parentData && parentData.is_chain) {
        _expandChain(parentData.id, name);
    }
}

function _expandChain(chainId, selectAfter) {
    _expandedChains.add(chainId);
    _pendingSelectNode = selectAfter || null;
    loadGraph(currentFn);
}

function _collapseChainOf(nodeData) {
    // The server stamps every member of an opened chain with its chain id, so a
    // member in the middle of the run can fold the whole thing back up.
    if (!nodeData || !nodeData.in_chain) return;
    _expandedChains.delete(nodeData.in_chain);
    loadGraph(currentFn);
}

let _pendingSelectNode = null;

function renderDetailTags(nodeId) {
    const section = document.getElementById("detail-tags-section");
    if (!section) return;
    section.innerHTML = "";

    // backend aliases this node belongs to (read-only)
    const backendAliases = [];
    if (currentGraphData && currentGraphData.aliases) {
        Object.entries(currentGraphData.aliases).forEach(([name, members]) => {
            if (members.includes(nodeId)) backendAliases.push(name);
        });
    }

    function rebuild() {
        section.innerHTML = "";
        const userTags = _getNodeTags(nodeId);
        const allAliases = [...backendAliases]; // read-only backend

        if (allAliases.length === 0 && userTags.length === 0) {
            // still show + button even when empty
        }

        const header = document.createElement("div");
        header.className = "detail-tags-header";
        const title = document.createElement("span");
        title.className = "detail-tags-title";
        title.textContent = "aliases";
        header.appendChild(title);
        section.appendChild(header);

        const pillsRow = document.createElement("div");
        pillsRow.className = "detail-tags-pills";

        // backend alias pills (read-only, different color)
        allAliases.forEach(alias => {
            const pill = document.createElement("span");
            pill.className = "detail-tag-pill detail-tag-backend";
            pill.textContent = "#" + alias;
            pill.title = "Built-in alias (from mark_tensor)";
            pillsRow.appendChild(pill);
        });

        // user tag pills (removable)
        userTags.forEach(tag => {
            const pill = document.createElement("span");
            pill.className = "detail-tag-pill detail-tag-user";
            pill.textContent = "#" + tag;
            const del = document.createElement("span");
            del.className = "detail-tag-del";
            del.textContent = "×";
            del.title = "Remove alias";
            del.addEventListener("click", () => {
                _removeNodeTag(nodeId, tag);
                rebuild();
                if (_actModalData && _actModalData._rebuildAliasChips) {
                    _actModalData._rebuildAliasChips();
                    _filterActModal();
                }
            });
            pill.appendChild(del);
            pillsRow.appendChild(pill);
        });

        // + button → inline input
        let _inputVisible = false;
        const addBtn = document.createElement("button");
        addBtn.className = "detail-tag-add-btn";
        addBtn.textContent = "+";
        addBtn.title = "Add alias";

        const inp = document.createElement("input");
        inp.type = "text";
        inp.className = "detail-tag-input";
        inp.placeholder = "alias name…";
        inp.style.display = "none";

        function commitTag() {
            const val = inp.value.trim().replace(/^#/, "");
            if (val) {
                _addNodeTag(nodeId, val);
                rebuild();
                if (_actModalData && _actModalData._rebuildAliasChips) {
                    _actModalData._rebuildAliasChips();
                    _filterActModal();
                }
            } else {
                inp.style.display = "none";
                addBtn.style.display = "";
                _inputVisible = false;
            }
        }

        addBtn.addEventListener("click", () => {
            _inputVisible = true;
            addBtn.style.display = "none";
            inp.style.display = "";
            inp.value = "";
            requestAnimationFrame(() => inp.focus());
        });
        inp.addEventListener("keydown", e => {
            if (e.key === "Enter") { e.preventDefault(); commitTag(); }
            if (e.key === "Escape") { inp.style.display = "none"; addBtn.style.display = ""; _inputVisible = false; }
        });
        inp.addEventListener("blur", commitTag);

        pillsRow.appendChild(addBtn);
        pillsRow.appendChild(inp);
        section.appendChild(pillsRow);
    }

    rebuild();
}

function renderPlaceholderInputsSidebar(name) {
    let section = document.getElementById("placeholder-inputs-section");
    if (!section) {
        section = document.createElement("div");
        section.id = "placeholder-inputs-section";
        const args = document.getElementById("detail-args");
        args.parentNode.insertBefore(section, args.nextSibling);
    }

    const arr = inputSets[name] || [];
    if (arr.length === 0) { section.style.display = "none"; return; }

    const selId = selectedIds[name];
    section.style.display = "block";
    section.innerHTML = "";

    const title = document.createElement("div");
    title.className = "detail-section-title";
    title.style.marginTop = "12px";
    title.textContent = "configured inputs";
    section.appendChild(title);

    const list = document.createElement("div");
    list.className = "ph-input-list";
    arr.forEach(inp => {
        const row = document.createElement("label");
        row.className = "ph-input-row" + (inp.id === selId ? " ph-input-selected" : "");
        row.title = inp.label;

        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = `ph-sel-${name}`;
        radio.value = inp.id;
        radio.checked = inp.id === selId;
        radio.addEventListener("change", () => selectInput(name, inp.id));

        const lbl = document.createElement("span");
        lbl.className = "ph-input-label" + (inp.valid ? "" : " ph-input-invalid");
        lbl.textContent = inp.label;

        row.appendChild(radio);
        row.appendChild(lbl);
        list.appendChild(row);
    });
    section.appendChild(list);
}

function renderArgs(args) {
    const container = document.getElementById("detail-args");
    container.innerHTML = "";

    if (args.length === 0) {
        container.innerHTML = '<span class="no-args">—</span>';
        return;
    }

    args.forEach((arg) => {
        const div = document.createElement("div");
        div.className = "arg-item";

        if (arg.type === "node") {
            const a = document.createElement("a");
            a.href = "#";
            a.className = "node-link";
            a.textContent = arg.name;
            a.dataset.node = arg.name;
            a.addEventListener("click", (e) => {
                e.preventDefault();
                const target = cy.$id(arg.name);
                if (target.length) {
                    cy.animate({ center: { eles: target }, zoom: cy.zoom() }, { duration: 250 });
                    onNodeClick(target);
                }
            });
            div.appendChild(a);
        } else if (arg.type === "list") {
            const items = arg.items
                .map((i) => (i.type === "node" ? `<a href="#" class="node-link" data-node="${i.name}">${i.name}</a>` : escapeHtml(i.value)))
                .join(", ");
            div.innerHTML = `[${items}]`;
            div.querySelectorAll(".node-link").forEach((a) => {
                a.addEventListener("click", (e) => {
                    e.preventDefault();
                    const target = cy.$id(a.dataset.node);
                    if (target.length) {
                        cy.animate({ center: { eles: target }, zoom: cy.zoom() }, { duration: 250 });
                        onNodeClick(target);
                    }
                });
            });
        } else {
            div.className += " arg-value";
            div.textContent = arg.value;
        }

        container.appendChild(div);
    });
}

// Which nodes read this one. The args section says where a value came from;
// without this the panel never says where it goes, which is the other half of
// reading a graph.
function renderConsumers(nodeId) {
    let section = document.getElementById("detail-users-section");
    if (!section) {
        section = document.createElement("div");
        section.id = "detail-users-section";
        section.className = "detail-section";
        const args = document.getElementById("detail-args");
        // sits directly under the args block it mirrors
        args.parentNode.insertBefore(section, args.nextSibling);
    }
    section.innerHTML = "";
    if (!currentGraphData || !nodeId) { section.style.display = "none"; return; }

    const users = currentGraphData.edges
        .filter(e => e.source === nodeId)
        .map(e => currentGraphData.nodes.find(n => n.id === e.target))
        .filter(Boolean);
    if (!users.length) { section.style.display = "none"; return; }
    section.style.display = "";

    const title = document.createElement("div");
    title.className = "detail-section-title";
    title.textContent = `used by (${users.length})`;
    section.appendChild(title);

    users.forEach((u) => {
        const div = document.createElement("div");
        div.className = "arg-item";
        const a = document.createElement("a");
        a.href = "#";
        a.className = "node-link";
        a.textContent = u.label || u.id;
        a.title = `${u.op}${u.shape && u.shape.length ? "  [" + u.shape.join("×") + "]" : ""}`;
        a.addEventListener("click", (e) => {
            e.preventDefault();
            if (_navGoToNode) _navGoToNode(u.id, u.label || u.id);
            else {
                const t = cy.$id(u.id);
                if (t.length) { cy.animate({ center: { eles: t }, zoom: cy.zoom() }, { duration: 250 }); onNodeClick(t); }
            }
        });
        div.appendChild(a);
        if (u.shape && u.shape.length) {
            const sh = document.createElement("span");
            sh.className = "act-list-shape";
            sh.textContent = ` [${u.shape.join("×")}]`;
            div.appendChild(sh);
        }
        section.appendChild(div);
    });
}

function hideDetails() {
    document.getElementById("details-empty").style.display = "flex";
    document.getElementById("details-content").style.display = "none";
    renderActivationList(currentGraphData);
}

// ─── search as a navigation scope ────────────────────────────────────────────
// A query in the activation panel narrows what the arrow keys walk, and the
// graph marks the matches so the scope is visible rather than merely felt.
function _setSearchScope(matchSet, query) {
    const changed = (_searchQuery !== (query || ""));
    _searchMatchSet = (matchSet && matchSet.size) ? matchSet : null;
    _searchQuery = query || "";
    _scopeChanged(changed);
}

// Is this node inside the active scope? Always true when nothing scopes it.
function _inSearchScope(label) {
    const sc = _effectiveScope();
    if (sc.set) return sc.set.has(label);
    if (sc.filter) {
        const n = currentGraphData && currentGraphData.nodes.find(x => (x.label || x.id) === label);
        return n ? !!sc.filter(n) : true;
    }
    return true;
}

// Release every narrowing the activation browser applies.  The scope names only
// one of them, but clearing just that one would leave the graph still dimmed
// with nothing on screen explaining why.
function _clearBrowserFilters() {
    const box = document.getElementById("act-modal-search");
    if (box) box.value = "";
    const modSel = document.getElementById("act-module-select");
    if (modSel) modSel.value = "";
    _actActiveModule = null;
    _actActiveAlias  = null;
    _activeListId    = null;
    _actBentOnly = _actMacroOnly = _actFavOnly = false;
    _actDynFilters = [];
    _actActivePresets.clear();
    _actActiveOps = new Set(_actModalNodes.map(n => n.op));

    const panel = document.getElementById("act-filter-panel");
    if (panel) {
        panel.querySelectorAll(".act-filter-chip").forEach(c => c.classList.add("active"));
        panel.querySelectorAll(".act-filter-chip:not([data-op])").forEach(c => c.classList.remove("active"));
        panel.querySelectorAll(".act-filter-dyn .act-filter-row:not(:first-child)").forEach(r => r.remove());
    }
    document.querySelectorAll(".act-list-chip").forEach(c => c.classList.remove("active"));

    if (typeof _filterActModal === "function") _filterActModal();
    _scopeChanged(true);
}

// Drop whatever is scoping the graph, whichever of the three it is.
function _clearScope() {
    const sc = _effectiveScope();
    if (sc.kind === "selection") { _actSelection.clear(); _refreshSelectionVisual(); _updateSelectionBar(); }
    else if (sc.kind === "list") {
        _activeListId = null; _navScope.mode = "all"; _navScope.listId = null;
        document.querySelectorAll(".act-list-chip").forEach(c => c.classList.remove("active"));
        if (typeof _filterActModal === "function") _filterActModal();
    }
    else if (sc.kind === "browser") { _clearBrowserFilters(); return; }
    else if (sc.kind === "search") {
        const inp = document.querySelector(".act-search-input");
        if (inp) { inp.value = ""; inp.dispatchEvent(new Event("input")); return; }
        _searchMatchSet = null; _searchQuery = "";
    } else { _navScope.mode = "all"; _navScope.listId = null; }
    document.querySelectorAll(".nav-scope-btn").forEach(b =>
        b.classList.toggle("active", b.dataset.scope === _navScope.mode));
    _scopeChanged(true);
}

function _makeCollapsibleSection(id, title, parent, opts = {}) {
    const defaultCollapsed = opts.defaultCollapsed ?? false;
    const color = opts.color ?? null;
    const isCollapsed = id in collapsedSections ? collapsedSections[id] : defaultCollapsed;

    const sec = document.createElement("div");
    sec.className = "collapsible-sec" + (isCollapsed ? " sec-collapsed" : "") + (color ? " collapsible-sec--sub" : "");
    if (color) sec.dataset.secId = id;

    const hd = document.createElement("div");
    hd.className = "collapsible-hd";
    if (color) hd.style.borderLeft = `3px solid ${color}`;

    const arrow = document.createElement("span");
    arrow.className = "collapsible-arrow";
    arrow.textContent = "▾";

    const lbl = document.createElement("span");
    lbl.className = "collapsible-title";
    lbl.textContent = title;

    hd.appendChild(arrow);
    hd.appendChild(lbl);
    hd.addEventListener("click", () => {
        const collapsed = sec.classList.toggle("sec-collapsed");
        collapsedSections[id] = collapsed;
    });

    const bd = document.createElement("div");
    bd.className = "collapsible-bd";

    sec.appendChild(hd);
    sec.appendChild(bd);
    parent.appendChild(sec);
    return bd;
}

function renderActivationList(data) {
    const container = document.getElementById("details-empty");
    container.innerHTML = "";

    if (!data || !data.nodes || data.nodes.length === 0) {
        container.innerHTML = `<div class="empty-hint"><div class="empty-icon">⬡</div><div>click a node<br>to inspect it</div></div>`;
        return;
    }

    // ── search bar ───────────────────────────────────────────────────────────
    const searchBar = document.createElement("div");
    searchBar.className = "act-search-bar";

    const searchInp = document.createElement("input");
    searchInp.type = "text";
    searchInp.className = "act-search-input";
    searchInp.placeholder = "name · #alias · mod: · shape: 2x?x512 · op: …";
    searchInp.setAttribute("spellcheck", "false");

    const expandBtn = document.createElement("button");
    expandBtn.className = "act-expand-btn";
    expandBtn.title = "Activation browser  [Tab]";
    expandBtn.textContent = "⊞";

    searchBar.appendChild(searchInp);
    searchBar.appendChild(expandBtn);
    container.appendChild(searchBar);

    const offview = _offviewNodes(data);
    const offviewSec = document.createElement("div");
    offviewSec.className = "act-offview-sec";
    offviewSec.style.display = "none";
    container.appendChild(offviewSec);

    // Matches that live outside the drawn graph. They cannot be clicked into
    // place the way a drawn node can, so each says where it is and opens that
    // view instead.  With nothing typed the section says how many there are, so
    // a folded module never reads as a part of the model that has gone missing.
    function _renderOffviewMatches(sec, matches, query) {
        sec.innerHTML = "";
        if (!matches.length) {
            if (query || !offview.length) { sec.style.display = "none"; return; }
            sec.style.display = "";
            const hint = document.createElement("div");
            hint.className = "act-offview-more";
            hint.textContent = `${offview.length} more node`
                + (offview.length === 1 ? "" : "s")
                + ` elsewhere in the model — search, or open the browser [Tab]`;
            hint.title = "In the trace, but folded away or hidden by this view";
            hint.addEventListener("click", () => openActModal(data));
            sec.appendChild(hint);
            return;
        }
        sec.style.display = "";

        const head = document.createElement("div");
        head.className = "act-offview-head";
        head.textContent = `elsewhere in the model · ${matches.length}`;
        head.title = "In the trace, but folded away or hidden by this view";
        sec.appendChild(head);

        matches.slice(0, 40).forEach(n => {
            const row = document.createElement("div");
            row.className = "act-offview-row";
            row.dataset.nodeName = n.label;

            const top = document.createElement("div");
            top.className = "act-list-row act-offview-top";

            const badge = document.createElement("span");
            badge.className = "act-list-badge";
            badge.style.background = OP_COLORS[n.op] || DEFAULT_COLOR;
            badge.textContent = (n.op || "").replace("call_", "").replace("_", "\u200b");

            const name = document.createElement("span");
            name.className = "act-list-name";
            name.textContent = n.label;

            const shape = document.createElement("span");
            shape.className = "act-list-shape";
            shape.textContent = n.shape && n.shape.length ? `[${n.shape.join("×")}]` : "";

            top.appendChild(badge);
            top.appendChild(name);
            top.appendChild(shape);

            // The module it lives in gets its own line: it is usually the longer
            // of the two, and squeezing it beside the name truncates both.
            const where = document.createElement("div");
            where.className = "act-offview-where";
            where.textContent = "↗ " + (n.module_path || "top level");

            row.appendChild(top);
            row.appendChild(where);
            row.title = `${n.label} — ${_offviewWhere(n)}. Click to go there.`;
            row.addEventListener("click", () => _goToOffviewNode(n));
            sec.appendChild(row);
        });

        if (matches.length > 40) {
            const more = document.createElement("div");
            more.className = "act-offview-more";
            more.textContent = `+ ${matches.length - 40} more — open the browser [Tab]`;
            more.addEventListener("click", () => openActModal(data));
            sec.appendChild(more);
        }
    }

    // live highlight — matches are accented, non-matches stay visible.
    // Uses the same advanced search grammar as the modal: name regex, field
    // filters (shape:, op:, target:, src:, tag:), #alias, and shape patterns
    // such as "2x?x128", "45x...x4", "2|4×512", or a bare dim like "512".
    function _applyActFilter() {
        const q = searchInp.value.trim();
        let matchSet = null;   // null = no query; Set = explicit match set
        const terms = q ? _parseActSearch(q) : null;
        if (q) {
            matchSet = new Set();
            data.nodes.forEach(n => {
                if (n.is_compound) return;
                if (_matchActSearch(n, terms, data.aliases)) matchSet.add(n.label);
            });
        }

        let firstMatch = null;
        container.querySelectorAll(".act-list-row").forEach(row => {
            const matches = matchSet ? matchSet.has(row.dataset.nodeName || "") : false;
            row.classList.toggle("act-list-match", matches);
            if (matches && !firstMatch) firstMatch = row;
        });
        if (firstMatch) firstMatch.scrollIntoView({ block: "nearest" });

        // A folded module is still part of the model. Anything the query finds
        // in the trace but not on screen goes in its own section, so it is
        // clear these are somewhere else rather than missing.
        _renderOffviewMatches(offviewSec, q ? offview.filter(
            n => _matchActSearch(n, terms, data.aliases)) : [], q);

        // the query is also a navigation scope, and the graph says so.
        // Only what is drawn can scope the canvas.
        _setSearchScope(matchSet, q);
    }
    _renderOffviewMatches(offviewSec, [], "");
    searchInp.addEventListener("input", () => { _applyActFilter(); _updateAliasSuggest(searchInp); });
    searchInp.addEventListener("keydown", e => _aliasSuggestKeydown(e, searchInp));
    searchInp.addEventListener("blur", () => {
        setTimeout(() => { const l = document.getElementById("alias-suggest-list"); if (l) l.style.display = "none"; }, 120);
    });
    expandBtn.addEventListener("click", () => openActModal(data));

    // ── activations section ──────────────────────────────────────────────────
    const actBd = _makeCollapsibleSection("sidebar-activations", "activations", container, { defaultCollapsed: true });

    data.nodes.filter(n => !n.is_compound).forEach((n) => {
        const row = document.createElement("div");
        row.className = "act-list-row" + (n.has_bending ? " act-list-bent" : "");
        row.dataset.nodeName = n.label;
        row.dataset.nodeOp   = n.op;

        const badge = document.createElement("span");
        badge.className = "act-list-badge";
        badge.style.background = OP_COLORS[n.op] || DEFAULT_COLOR;
        badge.textContent = n.op.replace("call_", "").replace("_", "​");

        const name = document.createElement("span");
        name.className = "act-list-name";
        name.textContent = n.label;

        const shape = document.createElement("span");
        shape.className = "act-list-shape";
        shape.textContent = n.shape && n.shape.length ? `[${n.shape.join("×")}]` : "";

        row.appendChild(badge);
        row.appendChild(name);
        row.appendChild(shape);

        // bend straight from the list — no detour through the node menu
        if (_SRC_OPS.has(n.op)) {
            const bendBtn = document.createElement("button");
            bendBtn.className = "act-list-bend-btn";
            bendBtn.textContent = "⚡";
            bendBtn.title = `Bend ${n.label}`;
            bendBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                _openBendDialog(n);
            });
            row.appendChild(bendBtn);
        }

        row.addEventListener("click", () => {
            const cyNode = cy.$id(n.id);
            if (cyNode.length) {
                cy.animate({ center: { eles: cyNode }, zoom: cy.zoom() }, { duration: 200 });
                onNodeClick(cyNode);
            }
        });

        actBd.appendChild(row);
    });

    // Macro (BendingParameter) rows in sidebar
    if (_bendingParams.length > 0) {
        const macroSep = document.createElement("div");
        macroSep.className = "act-list-macro-sep";
        macroSep.textContent = "macro";
        actBd.appendChild(macroSep);

        _bendingParams.forEach(bp => {
            const bpRow = document.createElement("div");
            bpRow.className = "act-list-row act-list-macro-row";

            const badge = document.createElement("span");
            badge.className = "act-list-badge act-list-macro-badge";
            badge.textContent = "⊕";

            const name = document.createElement("span");
            name.className = "act-list-name";
            name.textContent = bp.name;

            const val = document.createElement("span");
            val.className = "act-list-shape";
            val.textContent = _bpFmt(bp, bp.value);

            bpRow.appendChild(badge);
            bpRow.appendChild(name);
            bpRow.appendChild(val);
            bpRow.addEventListener("click", () => addBendingParamPin(bp.name));
            actBd.appendChild(bpRow);
        });
    }

    // ── modules section (below activations) ──────────────────────────────────
    const moduleNodes = data.nodes
        .filter(n => n.is_compound)
        .sort((a, b) => {
            const da = (a.module_path || "").split(".").length;
            const db = (b.module_path || "").split(".").length;
            return da !== db ? da - db : (a.module_path || "").localeCompare(b.module_path || "");
        });

    if (moduleNodes.length > 0) {
        const modBd = _makeCollapsibleSection("sidebar-modules", "modules", container, { defaultCollapsed: true });
        moduleNodes.forEach(n => {
            const depth = Math.min((n.module_path || "").split(".").length - 1, 4);
            const row = document.createElement("div");
            row.className = `mod-list-row mod-list-depth-${depth}`;
            const dot = document.createElement("span");
            dot.className = "mod-list-dot";
            const lbl = document.createElement("span");
            lbl.className = "mod-list-label";
            lbl.textContent = n.label;
            lbl.title = n.module_path || n.label;
            row.appendChild(dot);
            row.appendChild(lbl);
            row.addEventListener("click", () => _highlightModulePane(n.id));
            modBd.appendChild(row);
        });
    }
}

// ─── activation modal ─────────────────────────────────────────────────────────
// ─── activation modal search + filter ────────────────────────────────────────
function _parseActSearch(q) { return TBSearch.parse(q); }

function _shapeStr(n) { return TBSearch.shapeStr(n); }
function _isShapePattern(tok) { return TBSearch.isShapePattern(tok); }
function matchShapeQuery(shape, query) { return TBSearch.matchShapeQuery(shape, query); }
function _parseNumericCmp(raw) { return TBSearch.parseNumericCmp(raw); }
function _testNumericCmp(cmp, actual) { return TBSearch.testNumericCmp(cmp, actual); }
function _matchField(n, field, re, rawValue) {
    return TBSearch.match(n, [{ field, re, raw: rawValue }], _searchCtx());
}

// What TBSearch needs to know that a bare node object does not carry.
function _searchCtx(aliases) {
    return {
        aliases: aliases || (_actModalData && _actModalData.aliases)
                 || (currentGraphData && currentGraphData.aliases) || null,
        tags: _nodeTags,
        recurrence: _recurrenceMap,
    };
}

function _matchActSearch(n, terms, aliases) {
    return TBSearch.match(n, terms, _searchCtx(aliases));
}

// return all alias/tag names a node belongs to (backend + user)
function _nodeAliases(n, data) {
    return TBSearch.nodeAliases(n, _searchCtx(data && data.aliases));
}

// Top-level filter function — reads module-level filter state
function _filterActModal() {
    const q     = (document.getElementById("act-modal-search") || {}).value || "";
    const terms = _parseActSearch(q.trim());

    // build alias lookup set for active alias (backend aliases + user tags)
    let _aliasNodeSet = null;
    if (_actActiveAlias) {
        _aliasNodeSet = new Set();
        if (_actModalData && _actModalData.aliases) {
            (_actModalData.aliases[_actActiveAlias] || []).forEach(id => _aliasNodeSet.add(id));
        }
        // include nodes tagged by user with this name
        _actModalNodes.forEach(n => {
            const nodeId = n.id || n.label;
            if (_getNodeTags(nodeId).includes(_actActiveAlias)) _aliasNodeSet.add(nodeId);
        });
    }

    // Is the browser actually being asked something? What is dimmed on the
    // graph hangs on this: with nothing filtering, nothing should be.
    _actSearchingNow = terms.length > 0 || _actDynFilters.length > 0
        || _actActivePresets.size > 0 || _actBentOnly || _actMacroOnly
        || _actFavOnly || _actActiveAlias !== null || _actActiveModule !== null
        || !!_activeListId;

    function _nodeMatches(n) {
        // op chips always AND
        if (!_actActiveOps.has(n.op)) return false;

        // alias filter always AND
        if (_aliasNodeSet !== null && !_aliasNodeSet.has(n.id || n.label)) return false;

        // module scope always AND — picking a module narrows what you are looking
        // at rather than adding another thing to look for
        if (_actActiveModule !== null) {
            if (_actActiveModule === "\u0000none") {
                if (n.module_path) return false;
            } else if (!TBSearch.inModule(n.module_path, _actActiveModule)) {
                return false;
            }
        }

        // active list filter always AND
        if (_activeListId && _nodeLists[_activeListId]) {
            if (!_nodeLists[_activeListId].nodes.includes(n.label)) return false;
        }

        // collect per-filter results (combined with AND/OR mode)
        const checks = [];
        if (_actBentOnly)  checks.push(!!n.has_bending);
        if (_actMacroOnly) checks.push(!!n.has_macro);
        if (_actFavOnly)   checks.push(_isFav(n.label));
        _actDynFilters.forEach(f => { if (f.re || (f.field === "shape" && f.raw)) checks.push(_matchField(n, f.field, f.re, f.raw)); });
        if (terms.length > 0) checks.push(_matchActSearch(n, terms));
        if (_actActivePresets.size > 0) {
            const allData = _actModalData || {};
            const presetTerms = [..._actActivePresets].flatMap(q => _parseActSearch(q));
            checks.push(_matchActSearch(n, presetTerms, allData.aliases));
        }

        if (checks.length === 0) return true;
        return _actFilterMode === "all" ? checks.every(Boolean) : checks.some(Boolean);
    }

    _actFiltered = _actModalNodes.filter(_nodeMatches);
    // starred nodes float to the top
    _actFiltered.sort((a, b) => (_isFav(b.label) ? 1 : 0) - (_isFav(a.label) ? 1 : 0));
    _refreshRowListBtns();
    _publishBrowserScope(q);

    const list = document.getElementById("act-modal-list");
    if (list) {
        if (_actGrouped) {
            // Grouped mode: op-type sections, then sub-groups by target/function name
            list.querySelectorAll(".act-modal-row").forEach(r => r.style.display = "none");
            list.querySelectorAll(".act-group-header, .act-group-subheader").forEach(h => h.remove());
            const byOp = {};
            _actFiltered.forEach(n => { (byOp[n.op] = byOp[n.op] || []).push(n); });
            const opOrder = ["placeholder", "get_attr", "call_module", "call_function", "call_method", "output"];
            const ops = [...new Set([...opOrder.filter(o => byOp[o]), ...Object.keys(byOp).filter(o => !opOrder.includes(o))])];

            // On first render (null), start every group and sub-group collapsed.
            if (_actGroupCollapsed === null) _actGroupCollapsed = new Set(ops);
            if (_actSubCollapsed   === null) _actSubCollapsed   = new Set();

            function _rowHidden(op, subKey) {
                if (_actGroupCollapsed.has(op)) return true;
                if (subKey && _actSubCollapsed.has(`${op}:${subKey}`)) return true;
                return false;
            }

            function showRow(n, op, subKey) {
                const row = list.querySelector(`.act-modal-row[data-name="${CSS.escape(n.label)}"]`);
                if (row) {
                    row.style.display = _rowHidden(op, subKey) ? "none" : "";
                    list.appendChild(row);
                }
            }

            function makeHdr(text, op) {
                const h = document.createElement("div");
                h.className = "act-group-header";
                h.dataset.opGroup = op;
                const collapsed = _actGroupCollapsed.has(op);

                const arrow = document.createElement("span");
                arrow.className = "act-group-arrow";
                arrow.textContent = collapsed ? "▸" : "▾";

                h.appendChild(arrow);
                h.appendChild(document.createTextNode(" " + text));
                h.style.background = (OP_COLORS[op] || DEFAULT_COLOR) + "22";
                h.style.borderLeft = `3px solid ${OP_COLORS[op] || DEFAULT_COLOR}`;
                h.addEventListener("click", () => {
                    if (_actGroupCollapsed.has(op)) _actGroupCollapsed.delete(op);
                    else _actGroupCollapsed.add(op);
                    _filterActModal();
                });
                return h;
            }

            function makeSubHdr(text, op, subKey) {
                const h = document.createElement("div");
                h.className = "act-group-subheader";
                h.style.display = _actGroupCollapsed.has(op) ? "none" : "";
                const sk = `${op}:${subKey}`;
                const subCollapsed = _actSubCollapsed.has(sk);

                const arrow = document.createElement("span");
                arrow.className = "act-group-arrow";
                arrow.textContent = subCollapsed ? "▸" : "▾";

                h.appendChild(arrow);
                h.appendChild(document.createTextNode(" " + text));
                h.addEventListener("click", () => {
                    if (_actSubCollapsed.has(sk)) _actSubCollapsed.delete(sk);
                    else _actSubCollapsed.add(sk);
                    _filterActModal();
                });
                return h;
            }

            ops.forEach(op => {
                const group = byOp[op];
                if (!group || !group.length) return;

                // For placeholder/output, annotate header with shape info
                const isIO = op === "placeholder" || op === "output";
                const shapeInfo = isIO
                    ? group.map(n => n.shape && n.shape.length ? `[${n.shape.join("×")}]` : "?").join("  ")
                    : "";
                const label = `${op.replace("call_", "")} (${group.length})${shapeInfo ? "  " + shapeInfo : ""}`;
                list.appendChild(makeHdr(label, op));

                // Sub-group by target/function prefix
                const subKey = (n) => {
                    if (!n.target) return "";
                    if (op === "call_module")   return (n.target + "").split(".")[0];
                    if (op === "get_attr")       return (n.target + "").split(".")[0];
                    if (op === "call_function")  return (n.target + "").split(".").pop().replace(/\d+$/, "");
                    if (op === "call_method")    return n.target;
                    return "";
                };

                const bySub = {};
                group.forEach(n => { const k = subKey(n); (bySub[k] = bySub[k] || []).push(n); });
                const subKeys = Object.keys(bySub).sort();
                const hasMultipleSubs = subKeys.length > 1 || (subKeys.length === 1 && subKeys[0] !== "");

                // Seed sub-collapsed the first time each sub-key is encountered.
                if (hasMultipleSubs) {
                    subKeys.forEach(k => {
                        if (k) {
                            const sk = `${op}:${k}`;
                            if (!_actSubSeen.has(sk)) {
                                _actSubSeen.add(sk);
                                _actSubCollapsed.add(sk);
                            }
                        }
                    });
                }

                subKeys.forEach(key => {
                    if (hasMultipleSubs && key) list.appendChild(makeSubHdr(key, op, key));
                    bySub[key].forEach(n => showRow(n, op, hasMultipleSubs ? key : ""));
                });
            });
        } else {
            // Flat mode: remove all group chrome, restore topo order, show/hide
            list.querySelectorAll(".act-group-header, .act-group-subheader").forEach(h => h.remove());
            const filteredSet = new Set(_actFiltered.map(n => n.label));
            // Re-append in _actFiltered (topo) order so rows land in the right position
            _actFiltered.forEach(n => {
                const row = list.querySelector(`.act-modal-row[data-name="${CSS.escape(n.label)}"]`);
                if (row) { row.style.display = ""; list.appendChild(row); }
            });
            // Hide rows outside the filtered set
            list.querySelectorAll(".act-modal-row").forEach(row => {
                if (!filteredSet.has(row.dataset.name)) row.style.display = "none";
            });
        }
        _filterActModalBpRows(q);
    }

    if (_actModalMode === "detail") {
        _actDetailIdx = Math.min(_actDetailIdx, Math.max(0, _actFiltered.length - 1));
        _showActDetail(_actDetailIdx);
    }
}

function _showAddFilterMenu(anchor, dynSection) {
    document.querySelectorAll(".act-field-dropdown").forEach(el => el.remove());
    const fields = [
        { key: "name",   label: "name"   },
        { key: "target", label: "target" },
        { key: "shape",  label: "shape"  },
        { key: "src",    label: "source" },
        { key: "tag",    label: "alias"  },
        { key: "mod",    label: "module" },
    ];
    const dd = document.createElement("div");
    dd.className = "act-field-dropdown";
    fields.forEach(({ key, label }) => {
        const btn = document.createElement("button");
        btn.className = "act-field-dropdown-item";
        btn.textContent = label;
        btn.addEventListener("click", ev => {
            ev.stopPropagation();
            _actAddDynFilter(key, dynSection);
            dd.remove();
        });
        dd.appendChild(btn);
    });
    const rect = anchor.getBoundingClientRect();
    dd.style.cssText = `position:fixed;z-index:3300;top:${rect.bottom + 4}px;left:${rect.left}px`;
    document.body.appendChild(dd);
    const dismiss = ev => { if (!dd.contains(ev.target)) { dd.remove(); document.removeEventListener("click", dismiss); } };
    setTimeout(() => document.addEventListener("click", dismiss), 0);
}

function _actAddDynFilter(field, dynSection, initialValue) {
    const id = genId();
    let re = null;
    if (initialValue) try { re = new RegExp(initialValue, "i"); } catch (_) {}
    const filterObj = { id, field, re };
    _actDynFilters.push(filterObj);

    const row = document.createElement("div");
    row.className = "act-filter-row";

    const lbl = document.createElement("span");
    lbl.className = "act-filter-field";
    lbl.textContent = field;

    const inp = document.createElement("input");
    inp.type = "text";
    inp.className = "act-filter-text-input";
    inp.placeholder = "regexp…";
    inp.setAttribute("spellcheck", "false");
    if (initialValue) inp.value = initialValue;
    inp.addEventListener("input", () => {
        const val = inp.value.trim();
        filterObj.raw = val;
        try { filterObj.re = val ? new RegExp(val, "i") : null; } catch (_) { filterObj.re = null; }
        _filterActModal();
    });

    const removeBtn = document.createElement("button");
    removeBtn.className = "act-filter-remove-btn";
    removeBtn.title = "Remove filter";
    removeBtn.textContent = "×";
    removeBtn.addEventListener("click", () => {
        const idx = _actDynFilters.indexOf(filterObj);
        if (idx >= 0) _actDynFilters.splice(idx, 1);
        row.remove();
        _filterActModal();
    });

    row.appendChild(lbl);
    row.appendChild(inp);
    row.appendChild(removeBtn);
    dynSection.appendChild(row);
    inp.focus();
}

// ─── activation modal state ───────────────────────────────────────────────────
let _actModalNodes   = [];
let _actFiltered     = [];
let _actDetailIdx    = 0;
let _actModalMode    = "list";
let _actModalData    = null;
let _actDetailTab    = "viz"; // "viz" | "code"

// ── favourites (persisted in localStorage) ───────────────────────────────────
const _FAV_KEY = "torchbend_fav_nodes";
function _loadFavs() { try { return new Set(JSON.parse(localStorage.getItem(_FAV_KEY) || "[]")); } catch { return new Set(); } }
function _saveFavs() { localStorage.setItem(_FAV_KEY, JSON.stringify([..._favNodes])); _scheduleClientStatePush(); }
let _favNodes = _loadFavs();

function _isFav(label) { return _favNodes.has(label); }
function _toggleFav(label) {
    if (_favNodes.has(label)) _favNodes.delete(label); else _favNodes.add(label);
    _saveFavs();
}

// filter state (lifted so _filterActModal can be top-level)
let _actActiveOps   = new Set();
let _actBentOnly    = false;
let _actMacroOnly   = false;
let _actFavOnly     = false;
let _actGrouped          = false;
let _actGroupCollapsed   = null;   // null = uninitialised; entries are op names
let _actSubCollapsed     = null;   // null = uninitialised; entries are "op:subKey"
let _actSubSeen          = new Set(); // all sub-keys ever encountered; used to seed only once
let _actSelection   = new Set();   // selected node labels

// The rows as they actually appear: grouping reorders them and collapsing hides
// them, and a range selection has to follow that, not the flat filtered list.
function _visibleActRows() {
    const list = document.getElementById("act-modal-list");
    if (!list) return [];
    return [...list.querySelectorAll(".act-modal-row")]
        .filter(r => r.style.display !== "none" && r.offsetParent !== null);
}
// Anchor for shift-range selection, held as a node label rather than an index:
// grouping reorders and hides rows, so a position in _actFiltered is not where
// the row actually is on screen.
let _actLastSelLabel = null;
let _actFilterMode  = "all"; // "all" | "any"
let _actDynFilters  = [];    // [{id, field, re}]
let _actActiveAlias = null;  // alias name currently filtering, or null
let _actActiveModule = null; // module path currently filtering (with its submodules), or null
// What the activation browser is currently showing, whenever that is narrower
// than everything.  The browser already intersects every control it has — ops,
// module, aliases, lists, presets, the search box — so publishing its result is
// what lets all of them scope the canvas and the arrow keys, instead of only
// lists and the sidebar search doing so.
let _browserScope = null;    // {set: Set<label>, label: string} | null
let _actActivePresets = new Set(); // active preset query strings

// ─── node lists (client-side, localStorage-backed) ────────────────────────────
let _nodeLists = (() => {
    try { return JSON.parse(localStorage.getItem("tb_act_node_lists") || "{}"); } catch(_) { return {}; }
})();
let _activeListId   = null;   // currently active list filter (id string or null)
let _actListRebuildFn = null; // set by openActModal to rebuild the list chips row

function _saveNodeLists() {
    try { localStorage.setItem("tb_act_node_lists", JSON.stringify(_nodeLists)); } catch(_) {}
    _updateNavScopeDisplay();
}

// ─── navigation scope ─────────────────────────────────────────────────────────
let _navScope = { mode: "all", listId: null };

// ─── the one scope ───────────────────────────────────────────────────────────
// Selecting nodes, activating a list and typing a search were three separate
// notions of "what I am working on", and only navigation buttons fed the arrow
// keys. They are one thing now: whichever is active scopes navigation *and* is
// drawn on the graph, and the sidebar says which.
//
// Priority runs most-explicit first: an explicit selection beats a list, which
// beats a search, which beats the scope buttons.
// Hand the browser's current result set to the graph.  Only when it is a
// strict subset: with nothing filtering, nothing should be dimmed.
function _publishBrowserScope(query) {
    // "Narrowed" has to mean a filter is doing something, not merely that the
    // listing is shorter than the node set: the set holds the whole trace, most
    // of which any one view leaves undrawn.
    const narrowed = _actSearchingNow
        && _actModalNodes.length > 0 && _actFiltered.length < _actModalNodes.length;
    if (!narrowed) {
        const had = _browserScope !== null;
        _browserScope = null;
        if (had) _scopeChanged(true);
        return;
    }
    _browserScope = {
        set: new Set(_actFiltered.map(n => n.label)),
        label: _browserScopeLabel(query),
    };
    _scopeChanged(true);
}

// Name the narrowing after whichever control the user most likely just touched.
function _browserScopeLabel(query) {
    const q = (query || "").trim();
    if (q) return q;
    if (_actActiveModule) {
        return _actActiveModule === "\u0000none" ? "no module" : "mod: " + _actActiveModule;
    }
    if (_activeListId && _nodeLists[_activeListId]) return _nodeLists[_activeListId].name;
    if (_actActiveAlias) return "#" + _actActiveAlias;
    if (_actBentOnly)  return "bended";
    if (_actMacroOnly) return "macro";
    if (_actFavOnly)   return "starred";
    return "filters";
}

function _effectiveScope() {
    if (_actSelection && _actSelection.size) {
        return { kind: "selection", label: `${_actSelection.size} node${_actSelection.size > 1 ? "s" : ""}`,
                 set: new Set(_actSelection) };
    }
    // Everything the browser filters by, already intersected — so a query still
    // narrows an active list instead of being masked by it.  The sidebar search
    // is the one narrowing the browser does not know about, and navigation
    // already honours it, so fold it in here too rather than let the canvas and
    // the arrow keys disagree.
    if (_browserScope) {
        const set = _searchMatchSet
            ? new Set([..._browserScope.set].filter(l => _searchMatchSet.has(l)))
            : _browserScope.set;
        const label = _searchMatchSet && _searchQuery
            ? `${_browserScope.label} + ${_searchQuery}` : _browserScope.label;
        return { kind: "browser", label, set };
    }
    // reached only before the browser has ever been opened
    if (_activeListId && _nodeLists[_activeListId]) {
        const l = _nodeLists[_activeListId];
        return { kind: "list", label: l.name, set: new Set(l.nodes) };
    }
    if (_searchMatchSet) {
        return { kind: "search", label: _searchQuery, set: _searchMatchSet };
    }
    if (_navScope.mode === "favs")
        return { kind: "favs", label: "starred", filter: n => _isFav(n.label) };
    if (_navScope.mode === "bended")
        return { kind: "bended", label: "bended", filter: n => !!n.has_bending };
    if (_navScope.mode === "list") {
        const l = _navScope.listId && _nodeLists[_navScope.listId];
        return l ? { kind: "list", label: l.name, set: new Set(l.nodes) }
                 : { kind: "list", label: "no list picked", set: new Set() };
    }
    return { kind: "all", label: "all nodes" };
}

function _computeNavNodes(base) {
    const sc = _effectiveScope();
    if (sc.set) return base.filter(n => sc.set.has(n.label));
    if (sc.filter) return base.filter(sc.filter);
    return base;
}

// Paint the scope on the graph: what is in it lifts, the rest recedes.
function _applyScopeHighlight() {
    if (!cy) return;
    const sc = _effectiveScope();
    const inScope = sc.set ? (n => sc.set.has(n.data("label") || n.id()))
                  : sc.filter ? (n => sc.filter({ label: n.data("label") || n.id(),
                                                  has_bending: n.data("has_bending") }))
                  : null;
    cy.batch(() => {
        cy.nodes().removeClass("search-match search-dim");
        if (!inScope) return;
        cy.nodes().forEach(n => n.addClass(inScope(n) ? "search-match" : "search-dim"));
    });
}

// Everything that can change the scope funnels through here.
function _scopeChanged(resetNav) {
    _applyScopeHighlight();
    if (resetNav !== false) _navResetForScope();
    _updateNavScopeDisplay();
}

function _updateNavScopeDisplay() {
    const countEl  = document.getElementById("nav-scope-count");
    const listRow  = document.getElementById("nav-scope-list-row");
    const listSel  = document.getElementById("nav-scope-list-sel");

    if (listRow) {
        listRow.style.display = _navScope.mode === "list" ? "" : "none";
        if (_navScope.mode === "list" && listSel) {
            const prev = listSel.value;
            listSel.innerHTML = '<option value="">— pick list —</option>';
            Object.entries(_nodeLists).forEach(([lid, lst]) => {
                const opt = document.createElement("option");
                opt.value = lid;
                opt.textContent = `${lst.name} (${lst.nodes.length})`;
                if (lid === (prev || _navScope.listId)) opt.selected = true;
                listSel.appendChild(opt);
            });
        }
    }

    // name whatever is scoping the graph right now
    const sc = _effectiveScope();
    const kindEl  = document.getElementById("sel-scope-kind");
    const labelEl = document.getElementById("sel-scope-label");
    const clearEl = document.getElementById("sel-scope-clear");
    const sumEl   = document.getElementById("sel-scope-summary");
    if (sumEl) {
        const KIND = { selection: "selected", list: "list", search: "search",
                       browser: "filtered", favs: "starred", bended: "bended", all: "" };
        sumEl.dataset.kind = sc.kind;
        if (kindEl)  kindEl.textContent = KIND[sc.kind] || sc.kind;
        if (labelEl) {
            labelEl.textContent = sc.label || "";
            labelEl.title = (sc.kind === "search" || sc.kind === "browser")
                ? `${sc.set ? sc.set.size : 0} nodes — ${sc.label}` : (sc.label || "");
        }
        if (clearEl) {
            clearEl.style.display = sc.kind === "all" ? "none" : "";
            clearEl.textContent = "\u2715 clear";
        }
        sumEl.classList.toggle("is-all", sc.kind === "all");

        // Anything but "all" means part of the graph is dimmed. Say so loudly,
        // on the whole section: a graph that went grey with no obvious reason
        // reads as broken, not as filtered.
        const section = sumEl.closest(".sidebar-section");
        if (section) {
            const active = sc.kind !== "all";
            const was = section.classList.contains("scope-active");
            section.classList.toggle("scope-active", active);
            section.dataset.scopeKind = sc.kind;
            if (active && !was) {
                section.classList.remove("scope-pulse");
                void section.offsetWidth;            // restart the animation
                section.classList.add("scope-pulse");
            }
            let note = document.getElementById("sel-scope-note");
            if (!note) {
                note = document.createElement("div");
                note.id = "sel-scope-note";
                note.className = "sel-scope-note";
                sumEl.insertAdjacentElement("afterend", note);
            }
            if (active && currentGraphData) {
                const real = currentGraphData.nodes.filter(n => !n.is_compound);
                const total = real.length;
                // a scope is either a set of names or a predicate (starred, bent)
                const inFocus = sc.set ? sc.set.size
                              : (typeof sc.filter === "function" ? real.filter(sc.filter).length : 0);
                note.textContent = `${inFocus} of ${total} nodes in focus — the rest of the graph is dimmed`;
                note.style.display = "";
            } else {
                note.style.display = "none";
            }
        }
    }

    if (!countEl || !currentGraphData) return;
    const skipWeights = document.getElementById("gopt-nav-skip-weights");
    const skip = skipWeights ? skipWeights.checked : true;
    const base = currentGraphData.nodes.filter(n => !n.is_compound && !(skip && n.op === "get_attr"));
    const nodes = _computeNavNodes(base);
    countEl.textContent = nodes.length;
}
function _genListId() { return "lst_" + Date.now().toString(36) + Math.random().toString(36).slice(2,6); }
function _createList(name) {
    const lid = _genListId();
    _nodeLists[lid] = { name, nodes: [] };
    _saveNodeLists();
    return lid;
}
function _deleteList(lid) {
    delete _nodeLists[lid];
    if (_activeListId === lid) _activeListId = null;
    _saveNodeLists();
}
function _addToList(lid, labels) {
    if (!_nodeLists[lid]) return;
    const s = new Set(_nodeLists[lid].nodes);
    labels.forEach(l => s.add(l));
    _nodeLists[lid].nodes = [...s];
    _saveNodeLists();
}
function _removeFromList(lid, labels) {
    if (!_nodeLists[lid]) return;
    const s = new Set(labels);
    _nodeLists[lid].nodes = _nodeLists[lid].nodes.filter(l => !s.has(l));
    _saveNodeLists();
}
function _refreshRowListBtns() {
    document.querySelectorAll("#act-modal-list .act-list-btn").forEach(btn => {
        if (typeof btn._update === "function") btn._update();
    });
}
function _showListPicker(triggerEl, labels, onDone) {
    document.getElementById("_act-list-picker")?.remove();
    if (!labels.length) return;
    const menu = document.createElement("div");
    menu.id = "_act-list-picker";
    menu.className = "ctx-menu";
    menu.style.zIndex = "3300";
    const listIds = Object.keys(_nodeLists);
    listIds.forEach(lid => {
        const item = document.createElement("div");
        item.className = "ctx-menu-item";
        item.textContent = "→ " + _nodeLists[lid].name;
        item.addEventListener("click", () => { menu.remove(); _addToList(lid, labels); onDone?.(lid); });
        menu.appendChild(item);
    });
    if (listIds.length) { const sep = document.createElement("div"); sep.className = "ctx-menu-sep"; menu.appendChild(sep); }
    const newItem = document.createElement("div");
    newItem.className = "ctx-menu-item";
    newItem.textContent = "+ new list…";
    newItem.addEventListener("click", () => {
        menu.remove();
        const name = prompt("List name:");
        if (!name || !name.trim()) return;
        const lid = _createList(name.trim());
        _addToList(lid, labels);
        onDone?.(lid);
    });
    menu.appendChild(newItem);
    document.body.appendChild(menu);
    const rect = triggerEl.getBoundingClientRect();
    menu.style.left = rect.left + "px";
    menu.style.top  = (rect.bottom + 2) + "px";
    requestAnimationFrame(() => {
        const r = menu.getBoundingClientRect();
        if (r.right  > window.innerWidth)  menu.style.left = (window.innerWidth  - r.width  - 8) + "px";
        if (r.bottom > window.innerHeight) menu.style.top  = (rect.top - r.height - 2) + "px";
    });
    setTimeout(() => document.addEventListener("mousedown", e => {
        if (!menu.contains(e.target)) menu.remove();
    }, { once: true }), 0);
}

// ─── node tags (client-side, localStorage-backed) ─────────────────────────────
const _TAG_KEY = "tb_act_tags";
let _nodeTags = (() => {
    try { return JSON.parse(localStorage.getItem(_TAG_KEY) || "{}"); } catch (_) { return {}; }
})();
function _saveTags() {
    try { localStorage.setItem(_TAG_KEY, JSON.stringify(_nodeTags)); } catch (_) {}
    _scheduleClientStatePush();
}
function _getNodeTags(nodeId) { return _nodeTags[nodeId] || []; }

function _getAllAliases() {
    const counts = {};
    if (currentGraphData && currentGraphData.aliases) {
        Object.entries(currentGraphData.aliases).forEach(([name, members]) => {
            counts[name] = (counts[name] || 0) + members.length;
        });
    }
    Object.values(_nodeTags).forEach(tags => tags.forEach(t => {
        if (!(t in counts)) counts[t] = 0;
        counts[t]++;
    }));
    return Object.entries(counts).sort((a, b) => a[0].localeCompare(b[0]));
}

let _aliasSuggestHi = -1;

function _updateAliasSuggest(inp) {
    const list = document.getElementById("alias-suggest-list");
    if (!list) return;
    const val = inp.value;
    const hashIdx = val.lastIndexOf("#");
    if (hashIdx === -1) { list.style.display = "none"; _aliasSuggestHi = -1; return; }

    const fragment = val.slice(hashIdx + 1).toLowerCase();
    const all = _getAllAliases().filter(([name]) => name.toLowerCase().includes(fragment));

    list.innerHTML = "";
    _aliasSuggestHi = -1;

    if (!all.length) {
        const empty = document.createElement("div");
        empty.className = "alias-suggest-item alias-suggest-empty";
        empty.textContent = fragment ? "no matching aliases" : "no aliases defined yet";
        list.appendChild(empty);
    } else {
        all.forEach(([name, count]) => {
            const item = document.createElement("div");
            item.className = "alias-suggest-item";
            item.dataset.alias = name;
            const hash = document.createElement("span");
            hash.className = "alias-suggest-hash";
            hash.textContent = "#";
            const label = document.createElement("span");
            label.textContent = name;
            const cnt = document.createElement("span");
            cnt.className = "alias-suggest-count";
            cnt.textContent = count ? `${count} node${count !== 1 ? "s" : ""}` : "";
            item.appendChild(hash);
            item.appendChild(label);
            item.appendChild(cnt);
            item.addEventListener("mousedown", e => {
                e.preventDefault();
                inp.value = val.slice(0, hashIdx) + "#" + name;
                list.style.display = "none";
                _aliasSuggestHi = -1;
                _filterActModal();
                inp.focus();
            });
            list.appendChild(item);
        });
    }

    // Position below the input using fixed coords
    const rect = inp.getBoundingClientRect();
    list.style.top  = (rect.bottom + 4) + "px";
    list.style.left = rect.left + "px";
    list.style.width = rect.width + "px";
    list.style.display = "";
}

function _aliasSuggestKeydown(e, inp) {
    const list = document.getElementById("alias-suggest-list");
    if (!list || list.style.display === "none") return false;
    const items = [...list.querySelectorAll(".alias-suggest-item:not(.alias-suggest-empty)")];
    if (e.key === "ArrowDown") {
        e.preventDefault();
        _aliasSuggestHi = Math.min(_aliasSuggestHi + 1, items.length - 1);
    } else if (e.key === "ArrowUp") {
        e.preventDefault();
        _aliasSuggestHi = Math.max(_aliasSuggestHi - 1, -1);
    } else if (e.key === "Enter" && _aliasSuggestHi >= 0) {
        e.preventDefault();
        items[_aliasSuggestHi].dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
        return true;
    } else if (e.key === "Escape") {
        list.style.display = "none"; _aliasSuggestHi = -1; return true;
    } else { return false; }
    items.forEach((it, i) => it.classList.toggle("active", i === _aliasSuggestHi));
    if (_aliasSuggestHi >= 0) items[_aliasSuggestHi].scrollIntoView({ block: "nearest" });
    return true;
}
function _addNodeTag(nodeId, tag) {
    const tags = _getNodeTags(nodeId);
    if (!tags.includes(tag)) { _nodeTags[nodeId] = [...tags, tag]; _saveTags(); }
}
function _removeNodeTag(nodeId, tag) {
    _nodeTags[nodeId] = _getNodeTags(nodeId).filter(t => t !== tag);
    if (!_nodeTags[nodeId].length) delete _nodeTags[nodeId];
    _saveTags();
}

// ─── bookmark persistence ─────────────────────────────────────────────────────
const _ACT_BM_KEY = "tb_act_bookmarks";
function _loadBookmarks() {
    try { return JSON.parse(localStorage.getItem(_ACT_BM_KEY) || "[]"); } catch (_) { return []; }
}
function _saveBookmarks(bms) {
    try { localStorage.setItem(_ACT_BM_KEY, JSON.stringify(bms)); } catch (_) {}
    _scheduleClientStatePush();
}

// ── server-side client state sync (pins, favs, tags, bookmarks) ──────────────
let _clientStatePushTimer = null;

function _scheduleClientStatePush() {
    clearTimeout(_clientStatePushTimer);
    _clientStatePushTimer = setTimeout(_pushClientState, 1500);
}

function _serializeInputSetsSync() {
    const sets = {};
    Object.entries(inputSets).forEach(([k, arr]) => {
        const kept = arr.filter(e => e.type === "expr").map(e => ({
            id: e.id, type: "expr", expr: e.expr, label: e.label, valid: e.valid,
        }));
        if (kept.length) sets[k] = kept;
    });
    return { sets, selected: { ...selectedIds } };
}

// Saved state must be read before it can be written back. Without this, any
// push that happens before the fetch lands — a pin render, an input restore —
// writes the empty in-memory state over what was on disk, and the model comes
// back next time with everything forgotten.
let _clientStateLoaded = false;

// The cards' view states (batch / channel mode, index, audio toggle) without
// what only means something in this page: cached audio URLs, the player.
const _VIZ_CACHE_KEYS = ["__audio", "__player", "__audioFp"];
function _vizStatesForSave() {
    const out = {};
    Object.entries((window.TBViews && TBViews._states) || {}).forEach(([k, st]) => {
        if (!st || typeof st !== "object") return;
        const clean = {};
        Object.entries(st).forEach(([f, v]) => { if (!_VIZ_CACHE_KEYS.includes(f)) clean[f] = v; });
        out[k] = clean;
    });
    return out;
}

// Restore saved view states. States saved before the caches were left out
// carry a `__player` that JSON turned into {} -- a player with no audio, which
// breaks playback -- so they are cleaned on the way in too.
function _applyVizStates(saved) {
    if (!saved || !window.TBViews) return;
    TBViews._states = TBViews._states || {};
    Object.entries(saved).forEach(([k, st]) => {
        if (!st || typeof st !== "object") return;
        const clean = Object.assign({}, st);
        _VIZ_CACHE_KEYS.forEach(f => delete clean[f]);
        TBViews._states[k] = clean;
    });
}

function _saveVizStatesLocal(name) {
    if (!name) return;
    try { localStorage.setItem(`tb_viz_states_${name}`, JSON.stringify(_vizStatesForSave())); } catch (_) {}
}

function _loadVizStatesLocal(name) {
    if (!name) return;
    try { _applyVizStates(JSON.parse(localStorage.getItem(`tb_viz_states_${name}`) || "null")); } catch (_) {}
}

function _pushClientState() {
    if (!currentModelName || !_clientStateLoaded) return;
    // the view states persist with or without sync, like the pins they belong to
    _saveVizStatesLocal(currentModelName);
    const toSave = pinPages.map(page =>
        page.map(({ data, originalData, ...rest }) => rest)
    );
    const state = {
        pins:      { pages: toSave, current: currentPinPage },
        favs:      [..._favNodes],
        tags:      _nodeTags,
        bookmarks: _loadBookmarks(),
        vizStates: _vizStatesForSave(),
        inputs:    _serializeInputSetsSync(),
        positions: _nodePositions,
    };
    fetch(`/api/client-state/?name=${encodeURIComponent(currentModelName)}`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(state),
    }).catch(() => {});
}

async function _fetchAndApplyClientState(name) {
    if (!name) return;
    // The shared bench (bench.js) is the inputs' source of truth: it holds what
    // play mode last did too, where the server's copy is this page's own and
    // expressions only. Read it before anything below can rewrite it.
    _sharedInputSnapshot = TBBench.load(name);
    _clientStateLoaded = false;      // a different model's state is not this one's
    // this browser's copy first; the server's (with sync) overrides it below
    _loadVizStatesLocal(name);
    try {
        const r = await fetch(`/api/client-state/?name=${encodeURIComponent(name)}`);
        if (!r.ok) return;
        const st = await r.json();
        if (!st || !Object.keys(st).length) return;
        if (st.pins) {
            const restored = (st.pins.pages || []).map(page =>
                page.map(pin => ({ ...pin, data: null, originalData: null }))
            );
            if (restored.length > 0) {
                pinPages = restored;
                currentPinPage = Math.min(st.pins.current || 0, pinPages.length - 1);
                try { localStorage.setItem(`tb_pin_pages_${name}`, JSON.stringify(st.pins)); } catch (_) {}
            }
        }
        if (st.favs) {
            _favNodes = new Set(st.favs);
            try { localStorage.setItem(_FAV_KEY, JSON.stringify(st.favs)); } catch (_) {}
        }
        if (st.tags) {
            _nodeTags = st.tags;
            try { localStorage.setItem(_TAG_KEY, JSON.stringify(st.tags)); } catch (_) {}
        }
        if (st.bookmarks) {
            try { localStorage.setItem(_ACT_BM_KEY, JSON.stringify(st.bookmarks)); } catch (_) {}
            _renderBookmarkBar();
        }
        if (st.vizStates) _applyVizStates(st.vizStates);
        if (st.positions) {
            _nodePositions = st.positions;
            // the graph may already be on screen when this lands
            if (cy && currentFn && _applySavedPositions()) addModulePanes();
            _syncResetLayoutBtn();
        }
        if (st.inputs && st.inputs.sets && !_sharedInputSnapshot) {
            Object.keys(inputSets).forEach(k => delete inputSets[k]);
            Object.keys(selectedIds).forEach(k => delete selectedIds[k]);
            Object.assign(inputSets, st.inputs.sets || {});
            Object.assign(selectedIds, st.inputs.selected || {});
            // mirror to localStorage so play mode sees the restored inputs — go through
            // the shared-store writer, which emits the play-mode entry format ({value},
            // files as data URLs); dumping st.inputs verbatim would keep the editor's
            // {expr} key and play mode would read empty expressions.
            _persistInputsForPlay(name);
            _notifyInputsRestored();
        }
        renderPinTabs();
        renderCurrentPage();
        _updatePinCount();
    } catch (_) {
        // a state we could not read is still a state we must not overwrite
        // blindly — but refusing to save anything ever is worse, so allow
        // writes and let the next successful read reconcile
    } finally {
        _clientStateLoaded = true;
    }
}
function _currentSearchState() {
    return {
        q:          (document.getElementById("act-modal-search") || {}).value || "",
        activeOps:  [..._actActiveOps],
        bentOnly:   _actBentOnly,
        macroOnly:  _actMacroOnly,
        favOnly:    _actFavOnly,
        filterMode: _actFilterMode,
        module:     _actActiveModule,
        dynFilters: _actDynFilters.map(f => ({ field: f.field, value: f.re ? f.re.source : "" })),
    };
}

function _renderBookmarkBar() {
    const bar = document.getElementById("act-bookmark-bar");
    if (!bar) return;
    const bms = _loadBookmarks();
    bar.innerHTML = "";
    if (bms.length === 0) { bar.style.display = "none"; return; }
    bar.style.display = "";

    bms.forEach(bm => {
        const chip = document.createElement("button");
        chip.className = "act-bm-chip";
        chip.textContent = bm.name;
        chip.title = `Restore: ${bm.name}`;
        chip.addEventListener("click", () => _restoreBookmark(bm));

        const del = document.createElement("span");
        del.className = "act-bm-del";
        del.textContent = "×";
        del.title = "Delete bookmark";
        del.addEventListener("click", e => {
            e.stopPropagation();
            _saveBookmarks(_loadBookmarks().filter(b => b.id !== bm.id));
            _renderBookmarkBar();
        });
        chip.appendChild(del);
        bar.appendChild(chip);
    });
}

function _restoreBookmark(bm) {
    const s = bm.state;
    document.getElementById("act-modal-search").value = s.q || "";
    _actActiveOps  = new Set(s.activeOps || []);
    _actBentOnly   = !!s.bentOnly;
    _actMacroOnly  = !!s.macroOnly;
    _actFavOnly    = !!s.favOnly;
    _actFilterMode = s.filterMode || "all";
    _actActiveModule = s.module || null;
    _actDynFilters = [];

    // rebuild filter panel to reflect restored state
    const panel = document.getElementById("act-filter-panel");
    // update op chips
    panel.querySelectorAll(".act-filter-chip[data-op]").forEach(chip => {
        const active = _actActiveOps.has(chip.dataset.op);
        chip.classList.toggle("active", active);
    });
    // update bent / macro / fav chips
    panel.querySelectorAll(".act-filter-chip:not([data-op])").forEach(chip => {
        if (chip.textContent === "macro")  chip.classList.toggle("active", _actMacroOnly);
        else if (chip.textContent === "★ fav") chip.classList.toggle("active", _actFavOnly);
        else chip.classList.toggle("active", _actBentOnly);
    });
    // update mode radios
    panel.querySelectorAll("input[type=radio]").forEach(r => {
        if (r.value === _actFilterMode) r.checked = true;
    });
    // update module scope
    const modSel = panel.querySelector("#act-module-select");
    if (modSel) modSel.value = _actActiveModule || "";
    // restore dynamic filters: remove existing dyn rows, re-add
    const dynSection = panel.querySelector(".act-filter-dyn");
    if (dynSection) {
        dynSection.querySelectorAll(".act-filter-row:not(:first-child)").forEach(r => r.remove());
        (s.dynFilters || []).forEach(f => _actAddDynFilter(f.field, dynSection, f.value));
    }

    _filterActModal();
}

// Pin a set of nodes at once: the browser's current results by default, or a
// given list (the multi-selection). Weights, placeholders and activations each
// come from where they live; a module stands for the nodes it produces.
function _pinAllFiltered(pageIdx, nodeList) {
    while (pinPages.length <= pageIdx) pinPages.push([]);
    const fn = currentFn;
    const nodes = _expandModulePins(nodeList || _actFiltered);
    if (nodes.length === 0) return;

    // placeholders: add immediately
    nodes.filter(n => n.op === "placeholder").forEach(n => {
        pinPages[pageIdx].push({ id: genId(), label: n.label, shape: null, data: null, phNodeId: n.id, nodeOp: "placeholder" });
    });

    // weights (get_attr): fetch individually
    const weightNodes = nodes.filter(n => n.op === "get_attr");
    // activations: batch fetch
    const actNodes = nodes.filter(n => n.op !== "placeholder" && n.op !== "get_attr");

    const promises = [];

    weightNodes.forEach(n => {
        promises.push(
            fetch(`/api/weights/${fn}/${n.label}/`).then(r => r.json())
                .then(data => ({ n, data })).catch(() => ({ n, data: null }))
        );
    });

    if (actNodes.length > 0 && hasAnyInput()) {
        const batchLabels = actNodes.flatMap(n => [n.label, n.label + "_bended"]);
        promises.push(
            fetch(`/api/activate/${fn}/`, { method: "POST", body: buildActivationForm(batchLabels) })
                .then(r => r.json())
                .then(acts => actNodes.map(n => ({ n, data: acts[n.label] || null })))
                .catch(() => actNodes.map(n => ({ n, data: null })))
        );
    } else {
        actNodes.forEach(n => promises.push(Promise.resolve({ n, data: null })));
    }

    Promise.all(promises).then(results => {
        results.flat().forEach(({ n, data }) => {
            pinPages[pageIdx].push({ id: genId(), label: n.label, shape: data ? data.shape : null, data, phNodeId: null, nodeOp: n.op });
        });
        renderPinTabs();
        if (pageIdx === currentPinPage) renderCurrentPage();
        _updatePinCount();
        showToast("info", `Pinned ${nodes.length} nodes → page ${pageIdx + 1}`);
    });

    // add placeholders immediately
    renderPinTabs();
    if (pageIdx === currentPinPage) renderCurrentPage();
    _updatePinCount();
}

function openActModal(data) {
    if (!data || !data.nodes) return;
    _actModalData = data;

    const modal       = document.getElementById("act-modal");
    const list        = document.getElementById("act-modal-list");
    const filterPanel = document.getElementById("act-filter-panel");

    document.getElementById("act-modal-search").value = "";
    filterPanel.innerHTML = "";
    list.innerHTML = "";

    // The browser is the whole trace, not only what this view draws: a node
    // inside a folded module is still a node someone came here to find, so it
    // is listed like any other and says where it lives.  The two halves come
    // from different serialisations, so neither `order` is comparable with the
    // other — the index is the one topological order that covers both.
    const _idxRank = new Map(_nodeIndex.map((n, i) => [n.id, i]));
    const _actRank = (n) => {
        if (_idxRank.has(n.id)) return _idxRank.get(n.id);
        // a folded module is not a traced node — sit it just before the member
        // the outside reads, which is where it stands in for
        for (const o of (n.output_nodes || []))
            if (_idxRank.has(o)) return _idxRank.get(o) - 0.5;
        return Number.MAX_SAFE_INTEGER;
    };
    _actModalNodes  = data.nodes.filter(n => !n.is_compound)
        .concat(_offviewNodes(data))
        .sort((a, b) => _actRank(a) - _actRank(b));
    _actDynFilters  = [];
    _actBentOnly    = false;
    _actMacroOnly   = false;
    _actFavOnly     = false;
    _actFilterMode  = "all";
    _actActiveModule = null;
    _actActivePresets.clear();
    // The selection is a scope, not a property of the pane being open — toggling
    // the browser must not throw it away. Keep it, dropping only what this node
    // set no longer contains (a different method, or another model).
    const _known = new Set(_actModalNodes.map(n => n.label));
    [..._actSelection].forEach(l => { if (!_known.has(l)) _actSelection.delete(l); });
    if (_actLastSelLabel && !_known.has(_actLastSelLabel)) _actLastSelLabel = null;

    const ops = [...new Set(_actModalNodes.map(n => n.op))];
    _actActiveOps   = new Set(ops);

    // ── filter panel ─────────────────────────────────────────────────────────
    function _fRow(labelText, contentEl) {
        const row = document.createElement("div");
        row.className = "act-filter-row";
        const lbl = document.createElement("span");
        lbl.className = "act-filter-field";
        lbl.textContent = labelText;
        row.appendChild(lbl);
        row.appendChild(contentEl);
        return row;
    }

    // op row (always AND with results)
    const opChips = document.createElement("div");
    opChips.className = "act-filter-chips";
    ops.forEach(op => {
        const chip = document.createElement("button");
        chip.className = "act-filter-chip active";
        chip.dataset.op = op;
        chip.style.setProperty("--chip-color", OP_COLORS[op] || DEFAULT_COLOR);
        chip.textContent = op.replace("call_", "");
        chip.addEventListener("click", () => {
            const isSolo = _actActiveOps.size === 1 && _actActiveOps.has(op);
            if (isSolo) {
                // Already solo'd — restore all ops
                ops.forEach(o => _actActiveOps.add(o));
                opChips.querySelectorAll(".act-filter-chip").forEach(c => c.classList.add("active"));
            } else {
                // Solo this op
                _actActiveOps.clear();
                _actActiveOps.add(op);
                opChips.querySelectorAll(".act-filter-chip").forEach(c => {
                    c.classList.toggle("active", c.dataset.op === op);
                });
            }
            _filterActModal();
        });
        opChips.appendChild(chip);
    });
    filterPanel.appendChild(_fRow("op", opChips));

    // ── module scope ──────────────────────────────────────────────────────────
    // A chip row cannot carry a real model's module tree (GPT-2 has 151 paths,
    // five deep), so this is a select: indented by depth, counted, and scoping
    // to a module keeps everything beneath it.
    const modules = TBSearch.moduleIndex(data);
    if (modules.length) {
        const sel = document.createElement("select");
        sel.className = "act-filter-select";
        sel.id = "act-module-select";
        sel.title = "Show only nodes in this module (submodules included)";

        const total = _actModalNodes.length;
        const optAll = document.createElement("option");
        optAll.value = "";
        optAll.textContent = `all modules (${total})`;
        sel.appendChild(optAll);

        // how many nodes each module holds once its submodules are counted in
        const deep = new Map();
        modules.forEach(m => {
            deep.set(m.path, _actModalNodes.filter(
                n => TBSearch.inModule(n.module_path, m.path)).length);
        });

        modules.forEach(m => {
            const count = deep.get(m.path) || 0;
            if (!count) return;                        // an empty module is noise
            const opt = document.createElement("option");
            opt.value = m.path;
            const leaf = m.path.split(".").pop();
            opt.textContent = `${"\u00a0\u00a0".repeat(m.depth - 1)}${leaf} (${count})`;
            opt.title = `${m.path} — ${count} nodes`;
            sel.appendChild(opt);
        });

        const orphans = _actModalNodes.filter(n => !n.module_path).length;
        if (orphans) {
            const opt = document.createElement("option");
            opt.value = "\u0000none";
            opt.textContent = `(no module) (${orphans})`;
            sel.appendChild(opt);
        }

        sel.value = _actActiveModule || "";
        sel.addEventListener("change", () => {
            _actActiveModule = sel.value || null;
            _filterActModal();
        });
        filterPanel.appendChild(_fRow("module", sel));
    }

    // separator — dynamic filters below this are combined with AND/OR mode
    const sep = document.createElement("div");
    sep.className = "act-filter-sep";
    filterPanel.appendChild(sep);

    // dynamic section (bent + user-added field rows live here)
    const dynSection = document.createElement("div");
    dynSection.className = "act-filter-dyn";
    filterPanel.appendChild(dynSection);

    const bentChips = document.createElement("div");
    bentChips.className = "act-filter-chips";
    const bentChip = document.createElement("button");
    bentChip.className = "act-filter-chip";
    bentChip.textContent = "bent only";
    bentChip.addEventListener("click", () => {
        _actBentOnly = !_actBentOnly;
        bentChip.classList.toggle("active", _actBentOnly);
        _filterActModal();
    });
    bentChips.appendChild(bentChip);
    const macroChip = document.createElement("button");
    macroChip.className = "act-filter-chip";
    macroChip.textContent = "macro";
    macroChip.style.setProperty("--chip-color", "#2c7be5");
    macroChip.addEventListener("click", () => {
        _actMacroOnly = !_actMacroOnly;
        macroChip.classList.toggle("active", _actMacroOnly);
        _filterActModal();
    });
    bentChips.appendChild(macroChip);
    const favChip = document.createElement("button");
    favChip.className = "act-filter-chip";
    favChip.textContent = "★ fav";
    favChip.style.setProperty("--chip-color", "#e6a817");
    favChip.addEventListener("click", () => {
        _actFavOnly = !_actFavOnly;
        favChip.classList.toggle("active", _actFavOnly);
        _filterActModal();
    });
    bentChips.appendChild(favChip);
    dynSection.appendChild(_fRow("bent", bentChips));

    // ── preset "interesting nodes" chips ─────────────────────────────────────
    const PRESETS = [
        { label: "modules",     query: "op:call_module",  color: "#7c6af7",
          tip: "Show only submodule calls (op:call_module)" },
        { label: "non-trivial", query: "trivial:no",      color: "#2c7be5",
          tip: "Exclude trivial ops: reshape, cast, clone… (trivial:no)" },
        { label: "shape ∆",    query: "change:yes",      color: "#e6a817",
          tip: "Nodes whose output shape differs from all parent shapes (change:yes)" },
        { label: "junction",   query: "in:>1",           color: "#3dc9b4",
          tip: "Nodes with more than one input — fan-in > 1 (in:>1)" },
        { label: "recurrent",  query: "op:call_module recur:>1", color: "#c94040",
          tip: "Module weights used more than once — weight-tied or truly recurrent layers (op:call_module recur:>1)" },
        { label: "has alias",  query: "has_alias:yes",   color: "#89dceb",
          tip: "Nodes that belong to at least one alias or user tag (has_alias:yes)" },
    ];

    function _showChipDelta(chip, delta) {
        const old = chip.querySelector(".preset-chip-delta");
        if (old) old.remove();
        if (delta === 0) return;
        const badge = document.createElement("span");
        badge.className = "preset-chip-delta";
        badge.textContent = delta > 0 ? `+${delta}` : `${delta}`;
        badge.style.color = delta > 0 ? "var(--act-match-color, #3dc9b4)" : "#c94040";
        chip.appendChild(badge);
        // fade out after 2 s
        setTimeout(() => badge.classList.add("preset-chip-delta-fade"), 1800);
        setTimeout(() => { if (badge.parentNode) badge.remove(); }, 2300);
    }

    const presetChips = document.createElement("div");
    presetChips.className = "act-filter-chips";
    PRESETS.forEach(({ label, query, color, tip }) => {
        const chip = document.createElement("button");
        chip.className = "act-filter-chip";
        chip.textContent = label;
        chip.title = tip;
        if (color) chip.style.setProperty("--chip-color", color);
        chip.addEventListener("click", () => {
            const before = _actFiltered.length;
            if (_actActivePresets.has(query)) {
                _actActivePresets.delete(query);
                chip.classList.remove("active");
            } else {
                _actActivePresets.add(query);
                chip.classList.add("active");
            }
            _filterActModal();
            _showChipDelta(chip, _actFiltered.length - before);
        });
        presetChips.appendChild(chip);
    });
    dynSection.appendChild(_fRow("preset", presetChips));

    // ── aliases + user-tag filter ─────────────────────────────────────────────
    _actActiveAlias = null;
    const aliasChips = document.createElement("div");
    aliasChips.className = "act-filter-chips";
    let _aliasChipEls = {};

    function _buildAliasChips() {
        aliasChips.innerHTML = "";
        _aliasChipEls = {};
        // collect all alias names: backend + unique user-tag names
        const names = new Set(Object.keys(data.aliases || {}));
        _actModalNodes.forEach(n => {
            _getNodeTags(n.id || n.label).forEach(t => names.add(t));
        });
        if (names.size === 0) return;
        names.forEach(aliasName => {
            const chip = document.createElement("button");
            chip.className = "act-filter-chip" + (_actActiveAlias === aliasName ? " active" : "");
            chip.textContent = "#" + aliasName;
            chip.title = `Filter by alias/tag: #${aliasName}`;
            chip.addEventListener("click", () => {
                if (_actActiveAlias === aliasName) {
                    _actActiveAlias = null;
                    chip.classList.remove("active");
                } else {
                    if (_actActiveAlias && _aliasChipEls[_actActiveAlias]) {
                        _aliasChipEls[_actActiveAlias].classList.remove("active");
                    }
                    _actActiveAlias = aliasName;
                    chip.classList.add("active");
                }
                _filterActModal();
            });
            _aliasChipEls[aliasName] = chip;
            aliasChips.appendChild(chip);
        });
    }
    _buildAliasChips();
    // expose so tag add/remove can trigger refresh
    data._rebuildAliasChips = _buildAliasChips;
    dynSection.appendChild(_fRow("aliases", aliasChips));

    // ── lists ─────────────────────────────────────────────────────────────────
    const listsChips = document.createElement("div");
    listsChips.className = "act-filter-chips";
    let _listChipEls = {};

    const clearListBtn = document.createElement("button");
    clearListBtn.className = "act-clear-list-btn";
    clearListBtn.textContent = "clear list";
    clearListBtn.title = "Remove all nodes from this list";
    clearListBtn.style.display = "none";
    clearListBtn.addEventListener("click", () => {
        if (!_activeListId || !_nodeLists[_activeListId]) return;
        if (!confirm(`Clear all nodes from "${_nodeLists[_activeListId].name}"?`)) return;
        _nodeLists[_activeListId].nodes = [];
        _saveNodeLists();
        _filterActModal();
        _refreshRowListBtns();
        _rebuildListChips();
    });

    function _updateListChipCounts() {
        Object.entries(_listChipEls).forEach(([lid, chip]) => {
            const cnt = chip.querySelector(".act-list-chip-count");
            if (cnt && _nodeLists[lid]) cnt.textContent = _nodeLists[lid].nodes.length;
        });
        clearListBtn.style.display = _activeListId ? "" : "none";
    }

    function _makeListChip(lid, lst) {
        const chip = document.createElement("button");
        chip.className = "act-filter-chip act-list-chip" + (_activeListId === lid ? " active" : "");
        const nameSpan = document.createElement("span");
        nameSpan.className = "act-list-chip-name";
        nameSpan.textContent = lst.name;
        const countSpan = document.createElement("span");
        countSpan.className = "act-list-chip-count";
        countSpan.textContent = lst.nodes.length;
        const delSpan = document.createElement("span");
        delSpan.className = "act-list-chip-del";
        delSpan.textContent = "×";
        delSpan.title = "Delete list";
        delSpan.addEventListener("click", e => {
            e.stopPropagation();
            if (!confirm(`Delete list "${lst.name}"?`)) return;
            _deleteList(lid);
            _rebuildListChips();
            _filterActModal();
            _refreshRowListBtns();
        });
        chip.appendChild(nameSpan);
        chip.appendChild(countSpan);
        chip.appendChild(delSpan);
        chip.addEventListener("click", () => {
            if (_activeListId === lid) {
                _activeListId = null;
                chip.classList.remove("active");
            } else {
                Object.values(_listChipEls).forEach(c => c.classList.remove("active"));
                _activeListId = lid;
                chip.classList.add("active");
            }
            _filterActModal();
            _refreshRowListBtns();
            clearListBtn.style.display = _activeListId ? "" : "none";
            // the list now scopes navigation and lights up on the graph
            _scopeChanged(true);
        });
        return chip;
    }

    function _rebuildListChips() {
        listsChips.innerHTML = "";
        _listChipEls = {};
        Object.entries(_nodeLists).forEach(([lid, lst]) => {
            const chip = _makeListChip(lid, lst);
            _listChipEls[lid] = chip;
            listsChips.appendChild(chip);
        });
        const addChip = document.createElement("button");
        addChip.className = "act-filter-chip act-list-new-btn";
        addChip.title = "Create new list";
        addChip.textContent = "+";
        addChip.addEventListener("click", () => {
            const name = prompt("List name:");
            if (!name || !name.trim()) return;
            _createList(name.trim());
            _rebuildListChips();
        });
        listsChips.appendChild(addChip);
        clearListBtn.style.display = _activeListId ? "" : "none";
    }
    _rebuildListChips();
    _actListRebuildFn = _rebuildListChips;

    const listsRowWrap = document.createElement("div");
    listsRowWrap.className = "act-lists-row-wrap";
    listsRowWrap.appendChild(listsChips);
    listsRowWrap.appendChild(clearListBtn);
    dynSection.appendChild(_fRow("lists", listsRowWrap));

    // Wire header "add filtered to list" button
    const addToListHeaderBtn = document.getElementById("act-add-to-list-btn");
    if (addToListHeaderBtn) {
        addToListHeaderBtn.onclick = (e) => {
            const labels = _actFiltered.map(n => n.label);
            if (!labels.length) return;
            _showListPicker(addToListHeaderBtn, labels, () => {
                _rebuildListChips();
                _refreshRowListBtns();
            });
        };
    }

    // footer: AND/OR mode + add filter button
    const footer = document.createElement("div");
    footer.className = "act-filter-footer";

    const modeWrap = document.createElement("div");
    modeWrap.className = "act-filter-mode";
    const modeLabel = document.createElement("span");
    modeLabel.className = "act-filter-mode-text";
    modeLabel.textContent = "match";
    modeWrap.appendChild(modeLabel);
    const modeId = "afm_" + genId();
    ["all", "any"].forEach(m => {
        const lbl = document.createElement("label");
        lbl.className = "act-filter-mode-opt";
        const radio = document.createElement("input");
        radio.type = "radio"; radio.name = modeId; radio.value = m;
        radio.checked = m === "all";
        radio.addEventListener("change", () => { _actFilterMode = m; _filterActModal(); });
        lbl.appendChild(radio);
        lbl.appendChild(document.createTextNode(" " + m));
        modeWrap.appendChild(lbl);
    });
    footer.appendChild(modeWrap);

    const addBtn = document.createElement("button");
    addBtn.className = "act-add-filter-btn";
    addBtn.textContent = "+ field";
    addBtn.addEventListener("click", e => { e.stopPropagation(); _showAddFilterMenu(addBtn, dynSection); });
    footer.appendChild(addBtn);

    const bmBtn = document.createElement("button");
    bmBtn.className = "act-add-filter-btn";
    bmBtn.textContent = "⊕ save";
    bmBtn.title = "Bookmark current search";
    bmBtn.addEventListener("click", () => {
        const name = prompt("Bookmark name:", document.getElementById("act-modal-search").value || "search");
        if (!name) return;
        const bms = _loadBookmarks();
        bms.push({ id: genId(), name, state: _currentSearchState() });
        _saveBookmarks(bms);
        _renderBookmarkBar();
    });
    footer.appendChild(bmBtn);
    filterPanel.appendChild(footer);

    // filter toggle
    document.getElementById("act-filter-toggle").onclick = () => {
        const hidden = filterPanel.classList.toggle("hidden");
        document.getElementById("act-filter-toggle").classList.toggle("active", !hidden);
    };

    // group/tree sort toggle
    _actGrouped = false;
    _actGroupCollapsed = null;
    _actSubCollapsed   = null;
    _actSubSeen.clear();
    const sortBtn = document.getElementById("act-sort-btn");
    sortBtn.classList.remove("active");
    sortBtn.onclick = () => {
        _actGrouped = !_actGrouped;
        if (_actGrouped) { _actGroupCollapsed = null; _actSubCollapsed = null; _actSubSeen.clear(); }
        sortBtn.classList.toggle("active", _actGrouped);
        _filterActModal();
    };

    // search bar input + alias suggestions
    const _searchInp = document.getElementById("act-modal-search");
    _searchInp.oninput = () => { _filterActModal(); _updateAliasSuggest(_searchInp); };
    _searchInp.addEventListener("keydown", e => _aliasSuggestKeydown(e, _searchInp));
    _searchInp.addEventListener("blur", () => {
        setTimeout(() => {
            const list = document.getElementById("alias-suggest-list");
            if (list) list.style.display = "none";
        }, 120);
    });
    _actFiltered = _actModalNodes.slice();
    _actSearchingNow = false;

    // ── list rows ─────────────────────────────────────────────────────────────
    _actModalNodes.forEach(n => {
        const row = document.createElement("div");
        row.className = "act-modal-row" + (n._offview ? " act-modal-row-offview" : "");
        row.dataset.name = n.label;
        row.dataset.op   = n.op;
        if (n._offview) {
            row.title = `${_offviewWhere(n)} — not drawn in this view. `
                      + `Click to go there.`;
        }

        // — main line —
        const main = document.createElement("div");
        main.className = "act-modal-row-main";

        const badge = document.createElement("span");
        badge.className = "act-modal-badge";
        badge.style.background = OP_COLORS[n.op] || DEFAULT_COLOR;
        badge.textContent = n.op.replace("call_", "").replace("_", "​");

        const name = document.createElement("span");
        name.className = "act-modal-name";
        name.textContent = n.label;

        const shape = document.createElement("span");
        shape.className = "act-modal-shape";
        shape.textContent = n.shape && n.shape.length ? `[${n.shape.join("×")}]` : "";

        const starBtn = document.createElement("button");
        starBtn.className = "act-modal-star-btn" + (_isFav(n.label) ? " starred" : "");
        starBtn.title = "Add to favourites";
        starBtn.textContent = "★";
        starBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            _toggleFav(n.label);
            starBtn.classList.toggle("starred", _isFav(n.label));
            _filterActModal(); // re-sort
        });

        const pinBtn = document.createElement("button");
        pinBtn.className = "act-modal-pin-btn";
        pinBtn.title = "Pin to dashboard";
        pinBtn.textContent = "⊕";
        pinBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            _showActModalPagePicker(e, n, pinBtn);
        });

        const detailBtn = document.createElement("button");
        detailBtn.className = "act-modal-detail-btn";
        detailBtn.title = "Open detail view";
        detailBtn.textContent = "⊡";
        detailBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            const fi = _actFiltered.indexOf(n);
            _actDetailIdx = fi >= 0 ? fi : 0;
            _setActModalMode("detail");
        });

        const saveBtn = document.createElement("button");
        saveBtn.className = "act-modal-detail-btn";
        saveBtn.title = "Save activation";
        saveBtn.textContent = "⬇";
        saveBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            _showActSaveMenu(e, n.label);
        });

        const listBtn = document.createElement("button");
        listBtn.className = "act-modal-detail-btn act-list-btn";
        listBtn._update = function() {
            if (_activeListId && _nodeLists[_activeListId]) {
                const inList = _nodeLists[_activeListId].nodes.includes(n.label);
                listBtn.textContent = inList ? "−" : "+";
                listBtn.title = inList ? "Remove from list" : "Add to active list";
                listBtn.classList.toggle("act-list-btn-in", inList);
            } else {
                listBtn.textContent = "⊞";
                listBtn.title = "Add to list";
                listBtn.classList.remove("act-list-btn-in");
            }
        };
        listBtn._update();
        listBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            if (_activeListId && _nodeLists[_activeListId]) {
                const inList = _nodeLists[_activeListId].nodes.includes(n.label);
                if (inList) {
                    _removeFromList(_activeListId, [n.label]);
                } else {
                    _addToList(_activeListId, [n.label]);
                }
                listBtn._update();
                _updateListChipCounts();
                if (inList) _filterActModal(); // re-filter only on removal
            } else {
                _showListPicker(listBtn, [n.label], () => {
                    _rebuildListChips();
                    _refreshRowListBtns();
                });
            }
        });

        main.appendChild(badge);
        main.appendChild(name);
        main.appendChild(shape);
        main.appendChild(starBtn);
        main.appendChild(pinBtn);
        main.appendChild(detailBtn);
        main.appendChild(saveBtn);
        main.appendChild(listBtn);
        row.appendChild(main);

        // — sub line: target + source —
        const hasSub = n.target || n.source_file || n._offview;
        if (hasSub) {
            const sub = document.createElement("div");
            sub.className = "act-modal-row-sub";

            if (n._offview) {
                const where = document.createElement("span");
                where.className = "act-offview-where";
                where.textContent = "↗ " + (n.module_path || "top level");
                where.title = `${_offviewWhere(n)} — click the row to go there`;
                sub.appendChild(where);
            }

            if (n.target) {
                const tgt = document.createElement("span");
                tgt.className = "act-row-target";
                tgt.textContent = n.target;
                tgt.title = n.target;
                sub.appendChild(tgt);
            }

            if (n.source_file) {
                const base = n.source_file.replace(/.*[/\\]/, "");
                const src = document.createElement("span");
                src.className = "act-row-src";
                src.textContent = `${base}:${n.source_line}` + (n.source_fn ? ` · ${n.source_fn}` : "");
                src.title = `${n.source_file}:${n.source_line}` + (n.source_fn ? ` (${n.source_fn})` : "");
                sub.appendChild(src);
            }

            row.appendChild(sub);
        }

        row.addEventListener("click", (e) => {
            if (e.ctrlKey || e.metaKey) {
                // Toggle this node in/out of selection
                e.preventDefault();
                if (_actSelection.has(n.label)) _actSelection.delete(n.label);
                else _actSelection.add(n.label);
                row.classList.toggle("act-selected", _actSelection.has(n.label));
                _actLastSelLabel = n.label;
                _updateSelectionBar();
            } else if (e.shiftKey && _actLastSelLabel) {
                // Range selection follows what is on screen. In grouped mode the
                // rows are reordered into op sections and collapsed ones are
                // hidden, so walking _actFiltered would sweep up everything
                // lying between the two rows in the *ungrouped* order — nodes
                // the user can neither see nor has any reason to select.
                e.preventDefault();
                const rows = _visibleActRows();
                const from = rows.findIndex(r => r.dataset.name === _actLastSelLabel);
                const to   = rows.findIndex(r => r.dataset.name === n.label);
                if (from >= 0 && to >= 0) {
                    const lo = Math.min(from, to), hi = Math.max(from, to);
                    rows.slice(lo, hi + 1).forEach(r => _actSelection.add(r.dataset.name));
                } else {
                    // the anchor scrolled out of the filter or its group closed
                    _actSelection.add(n.label);
                }
                _actLastSelLabel = n.label;
                _refreshSelectionVisual();
                _updateSelectionBar();
            } else {
                // Normal click: navigate to graph (existing behaviour)
                if (n._offview) { _goToOffviewNode(n); return; }
                const cyNode = cy.$id(n.id);
                if (cyNode.length) {
                    cy.animate({ center: { eles: cyNode }, zoom: cy.zoom() }, { duration: 200 });
                    _navIdx = -1;
                    ["graph-nav-panel", "graph-nav-code"].forEach(id => {
                        const el = document.getElementById(id);
                        if (el) el.style.display = "none";
                    });
                    const _navPanel = document.getElementById("graph-nav-panel");
                    if (_navPanel) _navPanel.classList.remove("code-open");
                    onNodeClick(cyNode);
                }
            }
        });

        list.appendChild(row);
    });

    // ── BendingParameter "macro" rows ────────────────────────────────────────
    _renderActModalBpRows(list);

    // mode toggle buttons
    document.getElementById("act-mode-list").onclick   = () => _setActModalMode("list");
    document.getElementById("act-mode-detail").onclick = () => _setActModalMode("detail");
    document.getElementById("act-mode-bend").onclick   = () => _setActModalMode("bend");

    // detail content tabs
    document.getElementById("act-tab-viz").onclick  = () => _switchActDetailTab("viz");
    document.getElementById("act-tab-code").onclick = () => _switchActDetailTab("code");

    // detail nav
    document.getElementById("act-detail-prev").onclick = () => { _actDetailIdx = Math.max(0, _actDetailIdx - 1); _showActDetail(_actDetailIdx); };
    document.getElementById("act-detail-next").onclick = () => { _actDetailIdx = Math.min(_actFiltered.length - 1, _actDetailIdx + 1); _showActDetail(_actDetailIdx); };

    document.getElementById("act-detail-pin-btn").onclick = (e) => {
        const node = _actFiltered[_actDetailIdx];
        if (node) _showActModalPagePicker(e, node, document.getElementById("act-detail-pin-btn"));
    };

    document.getElementById("act-detail-save-btn").addEventListener("click", (e) => {
        const node = _actFiltered[_actDetailIdx];
        if (!node) return;
        _showActSaveMenu(e, node.label);
    });

    // pin-all button (in header, wired per-open)
    document.getElementById("act-pin-all-btn").onclick = (e) => {
        if (_actFiltered.length === 0) return;
        _showActModalPagePicker(e, null, document.getElementById("act-pin-all-btn"), true);
    };

    _setActModalMode("list");
    _renderBookmarkBar();
    // the rows exist now, so a surviving selection can be shown on them again
    _refreshSelectionVisual();
    _updateSelectionBar();
    modal.style.display = "flex";
}

function _setActModalMode(mode) {
    _actModalMode = mode;
    const listEl    = document.getElementById("act-modal-list");
    const detailEl  = document.getElementById("act-modal-detail");
    const bendEl    = document.getElementById("act-modal-bend-view");
    const searchBar = document.getElementById("act-modal-search-bar");
    const filterEl  = document.getElementById("act-filter-panel");
    const btnList   = document.getElementById("act-mode-list");
    const btnDetail = document.getElementById("act-mode-detail");
    const btnBend   = document.getElementById("act-mode-bend");

    const isBend   = mode === "bend";
    const isDetail = mode === "detail";
    const isList   = !isDetail && !isBend;
    listEl.style.display    = isList ? "" : "none";
    detailEl.style.display  = isDetail ? "" : "none";
    bendEl.style.display    = isBend   ? "flex" : "none";
    searchBar.style.display = isList ? "" : "none";
    filterEl.style.display  = isList ? "" : "none";
    // Hide selection bar when leaving list mode
    const selBar = document.getElementById("act-selection-bar");
    if (selBar) selBar.style.display = (isList && _actSelection.size) ? "flex" : "none";
    btnList.classList.toggle("active",   mode === "list");
    btnDetail.classList.toggle("active", isDetail);
    btnBend.classList.toggle("active",   isBend);

    if (isDetail) _showActDetail(_actDetailIdx);
    if (isBend)   _renderBendActModalView();
}

const _SRC_OPS = new Set(["call_function", "call_module", "call_method", "get_attr"]);

function _switchActDetailTab(tab) {
    _actDetailTab = tab;
    document.getElementById("act-tab-viz").classList.toggle("active", tab === "viz");
    document.getElementById("act-tab-code").classList.toggle("active", tab === "code");
    const vizEl  = document.getElementById("act-detail-viz");
    const codeEl = document.getElementById("act-detail-code");
    vizEl.style.display  = tab === "viz"  ? "" : "none";
    codeEl.style.display = tab === "code" ? "" : "none";
    // Lazy-load code if the panel is empty for the current node
    const node = _actFiltered[_actDetailIdx];
    if (tab === "code" && node && !codeEl.dataset.loadedFor) {
        _loadActDetailCode(node, _actDetailIdx, codeEl);
    }
}

function _getAudioSampleRate() {
    for (const arr of Object.values(inputSets)) {
        for (const inp of arr) {
            if (inp.sampleRate && inp.sampleRate > 0) return inp.sampleRate;
        }
    }
    return 22050;
}

let _actSaveMenu = null;
function _showActSaveMenu(triggerEvt, nodeLabel) {
    if (_actSaveMenu) { _actSaveMenu.remove(); _actSaveMenu = null; }
    const fn = currentFn;
    const menu = document.createElement("div");
    menu.className = "ctx-menu";
    menu.style.zIndex = "3200";

    const formats = [
        { label: "⬇ tensor (.pt)",  fmt: "tensor" },
        { label: "⬇ image (.png)",  fmt: "image"  },
        { label: "⬇ audio (.wav)",  fmt: "audio"  },
    ];

    formats.forEach(({ label, fmt }) => {
        const item = document.createElement("div");
        item.className = "ctx-menu-item";
        item.textContent = label;
        item.addEventListener("click", async () => {
            menu.remove(); _actSaveMenu = null;
            const form = buildActivationForm();
            form.append("format", fmt);
            if (fmt === "audio") form.append("sample_rate", String(_getAudioSampleRate()));
            try {
                const r = await fetch(`/api/activate_save/${fn}/${encodeURIComponent(nodeLabel)}/`, {
                    method: "POST", body: form,
                });
                if (!r.ok) { const d = await r.json(); throw new Error(d.error, { cause: d.traceback }); }
                const blob = await r.blob();
                const ext  = fmt === "tensor" ? "pt" : fmt === "image" ? "png" : "wav";
                const url  = URL.createObjectURL(blob);
                const a = Object.assign(document.createElement("a"), { href: url, download: `${nodeLabel}.${ext}` });
                document.body.appendChild(a); a.click(); document.body.removeChild(a);
                setTimeout(() => URL.revokeObjectURL(url), 1000);
                showToast("ok", `Saved ${nodeLabel}.${ext}`);
            } catch (err) { showToast("error", err.message, err.traceback || err.cause); }
        });
        menu.appendChild(item);
    });

    document.body.appendChild(menu);
    _actSaveMenu = menu;
    const btn = triggerEvt.currentTarget || triggerEvt.target;
    const rect = btn ? btn.getBoundingClientRect() : { left: triggerEvt.clientX, bottom: triggerEvt.clientY };
    menu.style.left = rect.left + "px";
    menu.style.top  = (rect.bottom + 4) + "px";
    requestAnimationFrame(() => {
        const r = menu.getBoundingClientRect();
        if (r.right  > window.innerWidth)  menu.style.left = (window.innerWidth  - r.width  - 8) + "px";
        if (r.bottom > window.innerHeight) menu.style.top  = (rect.top - r.height - 4) + "px";
    });
    const dismiss = (e) => { if (!menu.contains(e.target)) { menu.remove(); _actSaveMenu = null; } };
    setTimeout(() => document.addEventListener("mousedown", dismiss, { once: true }), 0);
}

// ── multi-selection helpers ───────────────────────────────────────────────────

function _updateSelectionBar() {
    const bar   = document.getElementById("act-selection-bar");
    const count = document.getElementById("act-sel-count");
    if (!bar) return;
    const n = _actSelection.size;
    bar.style.display = n === 0 ? "none" : "flex";
    if (count) count.textContent = `${n} selected`;
    // a selection is one of the things that scopes the graph
    _scopeChanged(true);
}

function _refreshSelectionVisual() {
    document.querySelectorAll("#act-modal-list .act-modal-row").forEach(row => {
        row.classList.toggle("act-selected", _actSelection.has(row.dataset.name));
    });
}

function _clearActSelection() {
    _actSelection.clear();
    _actLastSelLabel = null;
    _refreshSelectionVisual();
    _updateSelectionBar();
}

function _initActSelectionBar() {
    const selAll = document.getElementById("act-sel-all");
    const selPin = document.getElementById("act-sel-pin");
    const selStar = document.getElementById("act-sel-star");
    const selDetail = document.getElementById("act-sel-detail");
    const selClear = document.getElementById("act-sel-clear");
    if (!selAll) return;

    selAll.addEventListener("click", () => {
        // select all currently visible rows
        document.querySelectorAll("#act-modal-list .act-modal-row").forEach(row => {
            if (row.style.display !== "none") _actSelection.add(row.dataset.name);
        });
        _refreshSelectionVisual();
        _updateSelectionBar();
    });

    selPin.addEventListener("click", (e) => {
        if (!_actSelection.size) return;
        const nodes = _actFiltered.filter(n => _actSelection.has(n.label));
        if (nodes.length === 1) {
            _showActModalPagePicker(e, nodes[0], selPin);
        } else {
            // Pin all selected to current page
            _pinAllFiltered(currentPinPage, nodes);
            showToast("ok", `Pinned ${nodes.length} nodes`);
            _clearActSelection();
        }
    });

    selStar.addEventListener("click", () => {
        const labels = [..._actSelection];
        const allStarred = labels.every(l => _isFav(l));
        labels.forEach(l => {
            if (allStarred) _favNodes.delete(l);
            else _favNodes.add(l);
        });
        _saveFavs();
        _refreshSelectionVisual();
        // Update star buttons
        document.querySelectorAll("#act-modal-list .act-modal-star-btn").forEach(btn => {
            const row = btn.closest(".act-modal-row");
            if (row) btn.classList.toggle("starred", _isFav(row.dataset.name));
        });
        _filterActModal();
    });

    selDetail.addEventListener("click", () => {
        if (!_actSelection.size) return;
        // Filter the modal to only the selected nodes, switch to detail view
        const selectedNodes = _actFiltered.filter(n => _actSelection.has(n.label));
        if (!selectedNodes.length) return;
        // Temporarily override _actFiltered so detail view cycles through selection
        const saved = _actFiltered.slice();
        _actFiltered = selectedNodes;
        _actDetailIdx = 0;
        _setActModalMode("detail");
        // Restore on close/switch
        const restoreFn = () => { _actFiltered = saved; };
        document.getElementById("act-mode-list").addEventListener("click", restoreFn, { once: true });
    });

    selClear.addEventListener("click", _clearActSelection);

    const selAddList = document.getElementById("act-sel-addlist");
    if (selAddList) {
        selAddList.addEventListener("click", () => {
            if (!_actSelection.size) return;
            _showListPicker(selAddList, [..._actSelection], () => {
                _actListRebuildFn?.();
                _refreshRowListBtns();
            });
        });
    }
}


function _loadActDetailCode(node, capturedIdx, container) {
    if (!_SRC_OPS.has(node.op)) {
        container.innerHTML = `<span class="viz-loading">no source for ${node.op} nodes</span>`;
        container.dataset.loadedFor = String(capturedIdx);
        return;
    }
    container.innerHTML = `<span class="viz-loading">loading source…</span>`;
    container.dataset.loadedFor = "";
    fetch(`/api/node_source/${encodeURIComponent(currentFn)}/${encodeURIComponent(node.id)}/`)
        .then(r => r.json())
        .then(d => {
            if (_actDetailIdx !== capturedIdx) return;
            container.innerHTML = "";
            if (d.error) { container.innerHTML = `<span class="viz-error">${d.error}</span>`; return; }
            _renderSourceBlock(d, container);
            container.dataset.loadedFor = String(capturedIdx);
        })
        .catch(e => {
            if (_actDetailIdx !== capturedIdx) return;
            container.innerHTML = `<span class="viz-error">${e.message}</span>`;
        });
}

function _showActDetail(idx) {
    _actDetailIdx = idx;
    const node = _actFiltered[idx];
    const counter  = document.getElementById("act-detail-counter");
    const badgeEl  = document.getElementById("act-detail-badge");
    const nameEl   = document.getElementById("act-detail-name");
    const prevBtn  = document.getElementById("act-detail-prev");
    const nextBtn  = document.getElementById("act-detail-next");
    const propsEl  = document.getElementById("act-detail-props");
    const vizEl      = document.getElementById("act-detail-viz");
    const codeEl     = document.getElementById("act-detail-code");
    const pickerEl   = document.getElementById("act-detail-view-picker");

    counter.textContent = _actFiltered.length
        ? `${idx + 1} / ${_actFiltered.length}`
        : "— / —";
    prevBtn.disabled = idx <= 0;
    nextBtn.disabled = idx >= _actFiltered.length - 1;

    if (!node) { propsEl.innerHTML = ""; vizEl.innerHTML = ""; codeEl.innerHTML = ""; return; }

    badgeEl.textContent = node.op.replace("call_", "").replace("_", "​");
    badgeEl.style.background = OP_COLORS[node.op] || DEFAULT_COLOR;
    nameEl.textContent = node.label;

    // ── tabs: enable code tab only for nodes that have source ─────────────────
    const codeTabBtn = document.getElementById("act-tab-code");
    const hasSource = _SRC_OPS.has(node.op);
    codeTabBtn.disabled = !hasSource;
    // if current tab is code but this node can't show code, fall back to viz
    if (_actDetailTab === "code" && !hasSource) {
        _actDetailTab = "viz";
        document.getElementById("act-tab-viz").classList.add("active");
        codeTabBtn.classList.remove("active");
    }
    document.getElementById("act-tab-viz").classList.toggle("active", _actDetailTab === "viz");
    codeTabBtn.classList.toggle("active", _actDetailTab === "code");

    // ── panel visibility ──────────────────────────────────────────────────────
    vizEl.style.display  = _actDetailTab === "viz"  ? "" : "none";
    codeEl.style.display = _actDetailTab === "code" ? "" : "none";

    // reset code panel so it reloads for the new node
    codeEl.innerHTML = "";
    delete codeEl.dataset.loadedFor;

    // ── properties table ──────────────────────────────────────────────────────
    _renderNodePropsInto(node, propsEl);

    // ── viz ───────────────────────────────────────────────────────────────────
    if (pickerEl) pickerEl.innerHTML = "";
    _purgePlotlyIn(vizEl);
    vizEl.innerHTML = `<span class="viz-loading">loading…</span>`;

    const fn = currentFn;
    const fetchData = (node.op === "get_attr")
        ? fetch(`/api/weights/${fn}/${node.label}/`).then(r => r.json())
        : (hasAnyInput()
            ? fetch(`/api/activate/${fn}/`, { method: "POST", body: buildActivationForm([node.label, node.label + "_bended"]) })
                .then(r => r.json())
                .then(acts => { if (acts.error) throw new Error(acts.error); return acts[node.label] || null; })
            : Promise.resolve(null));

    const capturedIdx = idx;
    fetchData.then(data => {
        if (_actDetailIdx !== capturedIdx) return;
        _purgePlotlyIn(vizEl);
        vizEl.innerHTML = "";
        if (data) {
            // update shape in props if we now know it
            if (data.shape && data.shape.length) {
                const shEl = propsEl.querySelector(".adp-shape-val");
                if (shEl) shEl.textContent = `[${data.shape.join(" × ")}]`;
            }
            // audio play button
            if (data.is_audio_compatible && hasAnyInput()) {
                const audioBar = document.createElement("div");
                audioBar.className = "act-detail-audio-bar";
                const playBtn = document.createElement("button");
                playBtn.className = "act-detail-audio-btn";
                playBtn.textContent = "▶ play audio";
                playBtn.addEventListener("click", () => {
                    playBtn.textContent = "…";
                    playBtn.disabled = true;
                    fetch(`/api/activate_audio/${fn}/${node.label}/`, {
                        method: "POST", body: buildActivationForm(),
                    })
                    .then(r => r.ok ? r.blob() : r.json().then(d => { throw new Error(d.error); }))
                    .then(blob => {
                        const existing = audioBar.querySelector("audio");
                        if (existing) existing.remove();
                        const audio = document.createElement("audio");
                        audio.controls = true;
                        audio.className = "viz-audio-player";
                        audio.src = URL.createObjectURL(blob);
                        audioBar.appendChild(audio);
                        audio.play().catch(() => {});
                        playBtn.textContent = "▶ play audio";
                        playBtn.disabled = false;
                    })
                    .catch(e => {
                        playBtn.textContent = "▶ play audio";
                        playBtn.disabled = false;
                        showToast("error", "Audio failed: " + e.message);
                    });
                });
                const dlBtn2 = document.createElement("button");
                dlBtn2.className = "act-detail-audio-btn";
                dlBtn2.textContent = "⬇ save";
                dlBtn2.addEventListener("click", () => _downloadActivationAudio(node.label));
                audioBar.appendChild(playBtn);
                audioBar.appendChild(dlBtn2);
                vizEl.appendChild(audioBar);
            }
            // view picker (same as right sidebar — lets user switch e.g. spectrogram ↔ waveform)
            if (pickerEl) {
                pickerEl.innerHTML = "";
                const vmeta = data._view_meta;
                if (vmeta && window.TBViews && vmeta.compatible && vmeta.compatible.length > 1) {
                    pickerEl.appendChild(TBViews.renderPicker(vmeta, async (view, options) => {
                        try {
                            await fetch(`/api/views/${encodeURIComponent(fn)}/${encodeURIComponent(node.label)}/`, {
                                method: "POST", headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ view, options }),
                            });
                            _showActDetail(capturedIdx);
                        } catch (e) { showToast("error", e.message); }
                    }));
                }
            }
            _renderIntoPlotly(data, vizEl, { stateKey: fn + "/" + node.label,
                                             audioNode: node.label });
        } else {
            vizEl.innerHTML = `<span class="viz-loading">no data — add inputs first</span>`;
        }
    }).catch(err => {
        if (_actDetailIdx !== capturedIdx) return;
        vizEl.innerHTML = `<span class="viz-error">${err.message}</span>`;
    });

    // ── code (lazy: load now if code tab is active) ───────────────────────────
    if (_actDetailTab === "code") {
        _loadActDetailCode(node, idx, codeEl);
    }
}

function _renderNodePropsInto(node, container) {
    container.innerHTML = "";

    function row(label, valueEl) {
        const tr = document.createElement("tr");
        const td1 = document.createElement("td");
        td1.className = "adp-label";
        td1.textContent = label;
        const td2 = document.createElement("td");
        td2.className = "adp-value";
        if (typeof valueEl === "string") td2.textContent = valueEl;
        else td2.appendChild(valueEl);
        tr.appendChild(td1); tr.appendChild(td2);
        return tr;
    }

    const table = document.createElement("table");
    table.className = "adp-table";

    // target
    table.appendChild(row("target", node.target || "—"));

    // shape
    const shapeSpan = document.createElement("span");
    shapeSpan.className = "adp-shape-val";
    shapeSpan.textContent = node.shape && node.shape.length ? `[${node.shape.join(" × ")}]` : "—";
    table.appendChild(row("shape", shapeSpan));

    // source (clickable → open source modal)
    if (["call_function", "call_module", "call_method", "get_attr"].includes(node.op)) {
        const src = document.createElement("span");
        if (node.source_file) {
            const base = node.source_file.replace(/.*[/\\]/, "");
            src.textContent = `${base}:${node.source_line}`;
            src.title = `${node.source_file}:${node.source_line}` + (node.source_fn ? ` (${node.source_fn})` : "") + " — click to view";
        } else {
            src.textContent = node.target || "view source";
            src.title = "click to view source";
        }
        src.className = "adp-src";
        src.style.cursor = "pointer";
        src.addEventListener("click", () => openSourceModal(currentFn, node.id));
        table.appendChild(row("source", src));
    }

    container.appendChild(table);

    // bending
    if (node.has_bending && node.bending_callbacks && node.bending_callbacks.length) {
        const pre = document.createElement("pre");
        pre.className = "adp-bending";
        pre.textContent = node.bending_callbacks.join("\n");
        table.appendChild(row("bending", pre));
    }

    // args / inputs
    if (node.args && node.args.length) {
        const title = document.createElement("div");
        title.className = "adp-section-title";
        title.textContent = "inputs";
        container.appendChild(title);

        const argList = document.createElement("div");
        argList.className = "adp-args";
        node.args.forEach(arg => {
            const div = document.createElement("div");
            div.className = "adp-arg";
            if (arg.type === "node") {
                const a = document.createElement("a");
                a.href = "#"; a.className = "node-link adp-node-link";
                a.textContent = arg.name;
                a.addEventListener("click", e => {
                    e.preventDefault();
                    // Show in detail view if the node is in the filtered list
                    const idx = _actFiltered.findIndex(n => n.label === arg.name || n.id === arg.name);
                    if (idx >= 0) {
                        _actDetailIdx = idx;
                        _setActModalMode("detail");
                    } else {
                        const target = cy.$id(arg.name);
                        if (target.length) {
                            cy.animate({ center: { eles: target }, zoom: cy.zoom() }, { duration: 200 });
                            _navIdx = -1;
                            onNodeClick(target);
                            document.getElementById("act-modal").style.display = "none";
                        }
                    }
                });
                div.appendChild(a);
            } else if (arg.type === "list") {
                const items = arg.items.map(i => {
                    if (i.type === "node") {
                        const a = document.createElement("a");
                        a.href = "#"; a.className = "node-link adp-node-link";
                        a.textContent = i.name;
                        a.addEventListener("click", e => {
                            e.preventDefault();
                            const idx = _actFiltered.findIndex(n => n.label === i.name || n.id === i.name);
                            if (idx >= 0) { _actDetailIdx = idx; _setActModalMode("detail"); }
                            else {
                                const t = cy.$id(i.name);
                                if (t.length) { cy.animate({ center: { eles: t }, zoom: cy.zoom() }, { duration: 200 }); _navIdx = -1; onNodeClick(t); }
                                document.getElementById("act-modal").style.display = "none";
                            }
                        });
                        return a;
                    }
                    const s = document.createElement("span"); s.textContent = i.value; return s;
                });
                div.textContent = "[";
                items.forEach((el, i) => { div.appendChild(el); if (i < items.length - 1) { const c = document.createElement("span"); c.textContent = ", "; div.appendChild(c); } });
                div.appendChild(Object.assign(document.createElement("span"), { textContent: "]" }));
            } else {
                div.className += " adp-arg-val";
                div.textContent = arg.value;
            }
            argList.appendChild(div);
        });
        container.appendChild(argList);
    }
}

// pinAll=true → call _pinAllFiltered instead of _pinNodeToPage
function _showActModalPagePicker(e, node, anchorBtn, pinAll) {
    document.querySelectorAll(".pin-page-picker").forEach(el => el.remove());

    const picker = document.createElement("div");
    picker.className = "pin-page-picker";

    function doPin(idx) {
        if (pinAll) _pinAllFiltered(idx);
        else        _pinNodeToPage(node, idx);
        picker.remove();
        anchorBtn.classList.add("act-modal-pin-done");
        setTimeout(() => anchorBtn.classList.remove("act-modal-pin-done"), 1200);
    }

    pinPages.forEach((_, idx) => {
        const item = document.createElement("button");
        item.className = "pin-page-picker-item";
        item.textContent = `Page ${idx + 1}`;
        if (idx === currentPinPage) item.classList.add("current");
        item.addEventListener("click", ev => { ev.stopPropagation(); doPin(idx); });
        picker.appendChild(item);
    });

    const addItem = document.createElement("button");
    addItem.className = "pin-page-picker-item pin-page-picker-add";
    addItem.textContent = "+ new page";
    addItem.addEventListener("click", ev => {
        ev.stopPropagation();
        const newIdx = pinPages.length;
        pinPages.push([]);
        renderPinTabs();
        doPin(newIdx);
    });
    picker.appendChild(addItem);

    // position near button
    const rect = anchorBtn.getBoundingClientRect();
    picker.style.cssText = `position:fixed;z-index:3200;top:${rect.bottom + 4}px;right:${window.innerWidth - rect.right}px`;

    document.body.appendChild(picker);
    const dismiss = (ev) => { if (!picker.contains(ev.target)) { picker.remove(); document.removeEventListener("click", dismiss); } };
    setTimeout(() => document.addEventListener("click", dismiss), 0);
}

// A module has no tensor of its own. Asking the server for `__group__decoder`
// or `__mod__decoder` gets "no activation for ...", which is what put an empty
// card on the board. What a module produces are the members anything outside it
// reads — the payload names them — so those are what pinning it pins.
//
// Returns null for anything that is not a module, so callers can tell "not a
// module" from "a module that produces nothing this view can reach".
function _moduleOutputNodes(node) {
    if (!node || !(node.is_module_group || node.is_compound || node.op === "module"))
        return null;
    const drawn = (currentGraphData && currentGraphData.nodes) || [];
    return (node.output_nodes || []).map(name =>
        drawn.find(n => n.id === name)
        || (_nodeIndex || []).find(n => n.id === name)
        // in the trace but in neither list (a fold this view does not carry):
        // the name is what every pin path is keyed by, and it is enough
        || { id: name, label: name, op: "call_function", shape: null });
}

// Replace any module in a list with the nodes it produces, keeping the rest.
function _expandModulePins(nodes) {
    const out = [];
    const seen = new Set();
    (nodes || []).forEach(n => {
        (_moduleOutputNodes(n) || [n]).forEach(t => {
            if (t && !seen.has(t.label)) { seen.add(t.label); out.push(t); }
        });
    });
    return out;
}

function _pinNodeToPage(node, pageIdx) {
    while (pinPages.length <= pageIdx) pinPages.push([]);

    const moduleOuts = _moduleOutputNodes(node);
    if (moduleOuts) {
        if (!moduleOuts.length) {
            showToast("warn", `'${node.label}' produces nothing to pin`);
            return;
        }
        moduleOuts.forEach(n => _pinNodeToPage(n, pageIdx));
        return;
    }

    // Output node is a sink — pin its source feeder instead.
    if (node.op === "output") {
        const srcEdge = currentGraphData && currentGraphData.edges.find(e => e.target === node.id);
        const srcNode = srcEdge && currentGraphData.nodes.find(n => n.id === srcEdge.source);
        if (srcNode) { _pinNodeToPage(srcNode, pageIdx); }
        return;
    }

    function _commit(pin) {
        pinPages[pageIdx].push(pin);
        renderPinTabs();
        if (pageIdx === currentPinPage) renderCurrentPage();
        _updatePinCount();
        showToast("info", `Pinned "${node.label}" → page ${pageIdx + 1}`);
    }

    if (node.op === "placeholder") {
        _commit({ id: genId(), label: node.label, shape: null, data: null, phNodeId: node.id, nodeOp: "placeholder" });
        return;
    }

    const fn = currentFn;
    const fetchData = (node.op === "get_attr")
        ? fetch(`/api/weights/${fn}/${node.label}/`).then(r => r.json())
        : (hasAnyInput()
            ? fetch(`/api/activate/${fn}/`, { method: "POST", body: buildActivationForm([node.label, node.label + "_bended"]) })
                .then(r => r.json())
                .then(acts => { if (acts.error) throw new Error(acts.error); return acts[node.label] || null; })
            : Promise.resolve(null));

    fetchData
        .then(data => _commit({ id: genId(), label: node.label, shape: data ? data.shape : null, data, phNodeId: null, nodeOp: node.op }))
        .catch(err => showToast("error", `Could not pin: ${err.message}`, err.traceback));
}

// ─── prism.js lazy loader ─────────────────────────────────────────────────────
// Only core + Python grammar — we do line numbers / highlighting ourselves.
let _prismLoaded = false;
const _PRISM_BASE = "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0";
function _loadPrism(cb) {
    if (_prismLoaded) { cb(); return; }
    const cssHref = `${_PRISM_BASE}/themes/prism-tomorrow.min.css`;
    if (!document.querySelector(`link[href="${cssHref}"]`)) {
        const l = document.createElement("link");
        l.rel = "stylesheet"; l.href = cssHref;
        document.head.appendChild(l);
    }
    const urls = [
        `${_PRISM_BASE}/prism.min.js`,
        `${_PRISM_BASE}/components/prism-python.min.js`,
    ];
    let i = 0;
    function next() {
        if (i >= urls.length) { _prismLoaded = true; cb(); return; }
        if (document.querySelector(`script[src="${urls[i]}"]`)) { i++; next(); return; }
        const s = document.createElement("script");
        s.src = urls[i++];
        s.onload = next; s.onerror = next;
        document.head.appendChild(s);
    }
    next();
}

// ─── source code modal ────────────────────────────────────────────────────────
function openSourceModal(fn, nodeId) {
    let modal = document.getElementById("source-modal");
    if (!modal) {
        modal = document.createElement("div");
        modal.id = "source-modal";
        modal.style.display = "none";
        modal.innerHTML = `
            <div id="source-modal-backdrop"></div>
            <div id="source-modal-panel">
                <div id="source-modal-header">
                    <div id="source-modal-titles">
                        <span id="source-modal-name"></span>
                        <span id="source-modal-filepath"></span>
                    </div>
                    <button id="source-modal-close">×</button>
                </div>
                <div id="source-modal-body"></div>
            </div>`;
        document.body.appendChild(modal);
        document.getElementById("source-modal-backdrop").addEventListener("click", () => {
            modal.style.display = "none";
        });
        document.getElementById("source-modal-close").addEventListener("click", () => {
            modal.style.display = "none";
        });
    }

    document.getElementById("source-modal-name").textContent = nodeId;
    document.getElementById("source-modal-filepath").textContent = "";

    const body = document.getElementById("source-modal-body");
    body.innerHTML = `<span class="viz-loading">…</span>`;
    modal.style.display = "flex";

    fetch(`/api/node_source/${encodeURIComponent(fn)}/${encodeURIComponent(nodeId)}/`)
        .then(r => r.json())
        .then(d => {
            if (d.error) { body.innerHTML = `<span class="viz-error">${d.error}</span>`; return; }
            const nameEl = document.getElementById("source-modal-name");
            const pathEl = document.getElementById("source-modal-filepath");
            // Show node name + any aliases (e.g. "stem_weight  #encoder")
            const aliasNames = _nodeAliases({id: nodeId}, currentGraphData);
            const aliasStr = aliasNames.length ? "  " + aliasNames.map(a => "#" + a).join(" ") : "";
            nameEl.textContent = nodeId + aliasStr;
            if (d.file) {
                const lineStr = d.highlight_line ? `:${d.highlight_line}` : "";
                pathEl.textContent = d.file + lineStr;
                pathEl.title = d.file + lineStr;
            }
            _renderSourceBlock(d, body);
        })
        .catch(e => { body.innerHTML = `<span class="viz-error">${e.message}</span>`; });
}

// ─── source code display ──────────────────────────────────────────────────────
// We post-process Prism's output: highlight the whole code block first, then
// split by newline and wrap each line in a flex div with a line-number gutter.
// This avoids all alignment issues from Prism's absolute-positioned plugins.
function _renderSourceBlock(d, container) {
    container.innerHTML = "";
    if (!d || !d.lines || d.lines.length === 0) return;

    const code = d.lines.map(l => l.text).join("\n");
    const hl = d.highlight_line || 0;
    const startNo = (d.lines[0] && d.lines[0].no) || 1;

    // Scratch element — syntax-highlight the raw code
    const scratch = document.createElement("code");
    scratch.className = "language-python";
    scratch.textContent = code;

    const pre = document.createElement("pre");
    pre.className = "source-prism-pre";
    pre.appendChild(scratch);
    // Must be in DOM for Prism to work in some configurations
    container.appendChild(pre);

    _loadPrism(() => {
        Prism.highlightElement(scratch);

        // Split the highlighted HTML by newlines into per-line fragments.
        // Prism doesn't normally produce multi-line spans for Python, but we
        // defensively handle unclosed tags by carrying them across lines.
        const lineHtmls = scratch.innerHTML.split("\n");
        scratch.innerHTML = "";   // clear and rebuild

        lineHtmls.forEach((lineHtml, i) => {
            const lineNo = startNo + i;
            const isHl = (lineNo === hl);

            const lineDiv = document.createElement("div");
            lineDiv.className = "code-line" + (isHl ? " code-line-hl" : "");

            const gutter = document.createElement("span");
            gutter.className = "code-lineno";
            gutter.textContent = String(lineNo);

            const text = document.createElement("span");
            text.className = "code-text";
            text.innerHTML = lineHtml;

            lineDiv.appendChild(gutter);
            lineDiv.appendChild(text);
            scratch.appendChild(lineDiv);
        });

        if (hl) {
            const hlEl = pre.querySelector(".code-line-hl");
            if (hlEl) setTimeout(() => hlEl.scrollIntoView({ block: "center", behavior: "instant" }), 0);
        }
    });
}

// ─── helpers ──────────────────────────────────────────────────────────────────
function showLoading(on) {
    document.getElementById("loading").style.display = on ? "flex" : "none";
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

// ─── tensor visualisation ─────────────────────────────────────────────────────
// Viridis-inspired colormap [0,1] → [r,g,b]
function heatColor(t) {
    t = Math.max(0, Math.min(1, t));
    const stops = [
        [68,  1,  84], [59, 82, 139], [33, 145, 140],
        [94, 201,  98], [253, 231,  37],
    ];
    const s = t * (stops.length - 1);
    const i = Math.min(Math.floor(s), stops.length - 2);
    const f = s - i;
    return stops[i].map((v, k) => Math.round(v + f * (stops[i + 1][k] - v)));
}

function drawScalar(val, wrap) {
    const d = document.createElement("div");
    d.className = "viz-scalar";
    d.textContent = Number.isFinite(val) ? val.toFixed(5) : String(val);
    wrap.appendChild(d);
}

function drawBar(vals, wrap, maxW) {
    const W = maxW || wrap.offsetWidth || 206, H = Math.round(W * 0.27);
    const canvas = document.createElement("canvas");
    canvas.width = W; canvas.height = H;
    canvas.className = "viz-bar-canvas";
    canvas.style.width  = W + "px";
    canvas.style.height = H + "px";
    const ctx = canvas.getContext("2d");
    const n = vals.length;
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const rng = mx - mn || 1;
    ctx.fillStyle = "#e8e8ed";
    ctx.fillRect(0, 0, W, H);
    // zero line
    const zero = H - ((0 - mn) / rng) * H;
    ctx.strokeStyle = "#b0b0be"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(0, zero); ctx.lineTo(W, zero); ctx.stroke();
    const bw = W / n;
    vals.forEach((v, i) => {
        const h = Math.abs((v / rng) * H);
        const y = v >= 0 ? zero - h : zero;
        ctx.fillStyle = v >= 0 ? "#2c7be5" : "#E74C3C";
        ctx.fillRect(i * bw, y, Math.max(1, bw - 0.5), h);
    });
    wrap.appendChild(canvas);
}

function drawLine(vals, wrap, maxW, maxH) {
    const W = maxW || wrap.offsetWidth || 206;
    const H = maxH || Math.round(W * 0.27);
    const canvas = document.createElement("canvas");
    canvas.width = W; canvas.height = H;
    canvas.className = "viz-bar-canvas";
    canvas.style.width  = W + "px";
    canvas.style.height = H + "px";
    const ctx = canvas.getContext("2d");
    const n = vals.length;
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const rng = mx - mn || 1;
    ctx.fillStyle = "#e8e8ed";
    ctx.fillRect(0, 0, W, H);
    const zero = H - ((0 - mn) / rng) * H;
    ctx.strokeStyle = "#b0b0be"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(0, zero); ctx.lineTo(W, zero); ctx.stroke();
    // fill under curve
    ctx.beginPath();
    ctx.moveTo(0, zero);
    vals.forEach((v, i) => { ctx.lineTo((i / (n - 1)) * W, H - ((v - mn) / rng) * H); });
    ctx.lineTo(W, zero);
    ctx.closePath();
    ctx.fillStyle = "rgba(44,123,229,0.25)";
    ctx.fill();
    // line
    ctx.beginPath();
    vals.forEach((v, i) => {
        const x = (i / (n - 1)) * W, y = H - ((v - mn) / rng) * H;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#2c7be5"; ctx.lineWidth = 1.5;
    ctx.stroke();
    wrap.appendChild(canvas);
}

function drawLineset(channels, wrap, maxW, totalBatches, maxH, dim0Label) {
    const outer = document.createElement("div");
    outer.className = "lineset-wrap";

    // batch/filter/… selector
    const ctrl = document.createElement("div");
    ctrl.className = "lineset-ctrl";
    const lbl = document.createElement("span");
    lbl.className = "lineset-ch-label";
    lbl.textContent = dim0Label || "batch";
    const inp = document.createElement("input");
    inp.type = "number";
    inp.className = "lineset-ch-input";
    inp.min = 0;
    inp.max = (totalBatches || channels.length) - 1;
    inp.value = 0;
    const count = document.createElement("span");
    count.className = "lineset-ch-count";
    count.textContent = `/ ${(totalBatches || channels.length) - 1}`;
    ctrl.appendChild(lbl);
    ctrl.appendChild(inp);
    ctrl.appendChild(count);
    outer.appendChild(ctrl);

    const canvasWrap = document.createElement("div");
    outer.appendChild(canvasWrap);

    // reserve ~28 px for the control row + gap so the canvas doesn't overflow
    const innerH = maxH ? Math.max(40, maxH - 28) : null;

    function render(ch) {
        canvasWrap.innerHTML = "";
        const idx = Math.max(0, Math.min(ch, channels.length - 1));
        drawLine(channels[idx], canvasWrap, maxW, innerH);
    }
    render(0);

    inp.addEventListener("change", () => render(parseInt(inp.value, 10) || 0));
    inp.addEventListener("input",  () => render(parseInt(inp.value, 10) || 0));

    wrap.appendChild(outer);
}

function drawHeatmap(rows, wrap, maxW, maxH) {
    maxW = maxW || 206;
    maxH = maxH || maxW * 3;
    const fh = rows.length, fw = rows[0].length;
    const src = document.createElement("canvas");
    src.width = fw; src.height = fh;
    const ctx = src.getContext("2d");
    const img = ctx.createImageData(fw, fh);
    for (let r = 0; r < fh; r++) {
        for (let c = 0; c < fw; c++) {
            const [R, G, B] = heatColor(rows[r][c]);
            const idx = (r * fw + c) * 4;
            img.data[idx] = R; img.data[idx+1] = G;
            img.data[idx+2] = B; img.data[idx+3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0);
    const scale = Math.min(4, maxW / fw, maxH / fh);
    const out = document.createElement("canvas");
    out.width = Math.round(fw * scale);
    out.height = Math.round(fh * scale);
    out.style.imageRendering = "pixelated";
    const outCtx = out.getContext("2d");
    outCtx.imageSmoothingEnabled = false;
    outCtx.drawImage(src, 0, 0, out.width, out.height);
    wrap.appendChild(out);
}

function drawRGB(chw, wrap, maxW, maxH) {
    maxW = maxW || 206;
    const H = chw[0].length, W = chw[0][0].length;
    const src = document.createElement("canvas");
    src.width = W; src.height = H;
    const ctx = src.getContext("2d");
    const img = ctx.createImageData(W, H);
    for (let r = 0; r < H; r++) {
        for (let c = 0; c < W; c++) {
            const idx = (r * W + c) * 4;
            img.data[idx]   = Math.round(chw[0][r][c] * 255);
            img.data[idx+1] = Math.round(chw[1][r][c] * 255);
            img.data[idx+2] = Math.round(chw[2][r][c] * 255);
            img.data[idx+3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0);
    const scale = Math.min(4, maxW / W, (maxH || maxW * 1.5) / H);
    const out = document.createElement("canvas");
    out.width = Math.round(W * scale); out.height = Math.round(H * scale);
    out.style.imageRendering = "pixelated";
    const outCtx = out.getContext("2d");
    outCtx.imageSmoothingEnabled = false;
    outCtx.drawImage(src, 0, 0, out.width, out.height);
    wrap.appendChild(out);
}

function _normRows2d(rows) {
    let mn = Infinity, mx = -Infinity;
    rows.forEach(r => { if (r) r.forEach(v => { if (v < mn) mn = v; if (v > mx) mx = v; }); });
    const rng = mx - mn || 1;
    return rows.map(r => (r || []).map(v => (v - mn) / rng));
}

// 3D temporal [B, C, T]:
//   small (sidebar): show first-batch channels as heatmap rows (spectral view)
//   expanded (modal): batch + channel selectors + line plot
function drawTemporal3d(vizData, wrap, maxW, maxH) {
    if (!maxW || maxW <= 300) {
        drawHeatmap(_normRows2d(vizData.data[0] || [[]]), wrap, maxW || 206, maxH);
        return;
    }

    const outer = document.createElement("div");
    outer.className = "lineset-wrap";

    const ctrl = document.createElement("div");
    ctrl.className = "lineset-ctrl";

    function makeSelector(labelText, maxVal) {
        const lbl = document.createElement("span");
        lbl.className = "lineset-ch-label";
        lbl.textContent = labelText;
        const inp = document.createElement("input");
        inp.type = "number";
        inp.className = "lineset-ch-input";
        inp.min = 0; inp.max = maxVal; inp.value = 0;
        const cnt = document.createElement("span");
        cnt.className = "lineset-ch-count";
        cnt.textContent = `/ ${maxVal}`;
        ctrl.appendChild(lbl); ctrl.appendChild(inp); ctrl.appendChild(cnt);
        return inp;
    }

    const batchInp = makeSelector((vizData.dim_labels && vizData.dim_labels[0]) || "batch", vizData.n_batches - 1);
    const chInp    = makeSelector((vizData.dim_labels && vizData.dim_labels[1]) || "ch",    vizData.n_channels - 1);
    outer.appendChild(ctrl);

    const canvasWrap = document.createElement("div");
    outer.appendChild(canvasWrap);

    // reserve ~28 px for the control row + gap so the canvas doesn't overflow
    const innerH = maxH ? Math.max(40, maxH - 28) : null;

    function render() {
        canvasWrap.innerHTML = "";
        const b  = Math.max(0, Math.min(parseInt(batchInp.value, 10) || 0, vizData.data.length - 1));
        const ch = Math.max(0, Math.min(parseInt(chInp.value,  10) || 0, (vizData.data[b] || []).length - 1));
        drawLine((vizData.data[b] || [[]])[ch] || [], canvasWrap, maxW, innerH);
    }
    render();
    batchInp.addEventListener("change", render); batchInp.addEventListener("input", render);
    chInp.addEventListener("change",   render); chInp.addEventListener("input",  render);
    wrap.appendChild(outer);
}

function drawGrid(slices, wrap, maxW, maxH) {
    const n = slices.length;
    maxW = maxW || 206;
    const cols = Math.min(n, maxW > 400 ? 8 : 4);
    const rows = Math.ceil(n / cols);
    const gap = 3;
    const cellW = Math.floor((maxW - (cols - 1) * gap) / cols);
    const cellH = maxH ? Math.floor((maxH - (rows - 1) * gap) / rows) : undefined;
    slices.forEach((slice) => drawHeatmap(slice, wrap, cellW, cellH));
}

function _renderImageTile(chw, imageType, tileW, tileH) {
    const H = chw[0].length, W = chw[0][0].length;
    const src = document.createElement('canvas');
    src.width = W; src.height = H;
    const ctx = src.getContext('2d');
    const imgD = ctx.createImageData(W, H);
    const clamp = v => Math.round(Math.max(0, Math.min(1, v)) * 255);
    for (let r = 0; r < H; r++) {
        for (let c = 0; c < W; c++) {
            const idx = (r * W + c) * 4;
            if (imageType === 'gray') {
                const v = clamp(chw[0][r][c]);
                imgD.data[idx]=v; imgD.data[idx+1]=v; imgD.data[idx+2]=v; imgD.data[idx+3]=255;
            } else if (imageType === 'rgb') {
                imgD.data[idx]   = clamp(chw[0][r][c]);
                imgD.data[idx+1] = clamp(chw[1][r][c]);
                imgD.data[idx+2] = clamp(chw[2][r][c]);
                imgD.data[idx+3] = 255;
            } else {
                imgD.data[idx]   = clamp(chw[0][r][c]);
                imgD.data[idx+1] = clamp(chw[1][r][c]);
                imgD.data[idx+2] = clamp(chw[2][r][c]);
                imgD.data[idx+3] = clamp(chw[3][r][c]);
            }
        }
    }
    ctx.putImageData(imgD, 0, 0);
    // Scale to fill frame width, preserve natural aspect ratio; nearest-neighbour for crisp pixels
    const out = document.createElement('canvas');
    out.width = tileW; out.height = Math.max(1, Math.round(tileW * H / W));
    out.style.imageRendering = 'pixelated';
    const outCtx = out.getContext('2d');
    outCtx.imageSmoothingEnabled = false;
    outCtx.drawImage(src, 0, 0, out.width, out.height);
    return out;
}

function _renderGrayscaleTile(hw, tileW, tileH) {
    const H = hw.length, W = hw[0].length;
    const src = document.createElement('canvas');
    src.width = W; src.height = H;
    const ctx = src.getContext('2d');
    const imgD = ctx.createImageData(W, H);
    for (let r = 0; r < H; r++) {
        for (let c = 0; c < W; c++) {
            const idx = (r * W + c) * 4;
            const v = Math.round(Math.max(0, Math.min(1, hw[r][c])) * 255);
            imgD.data[idx] = v; imgD.data[idx+1] = v; imgD.data[idx+2] = v; imgD.data[idx+3] = 255;
        }
    }
    ctx.putImageData(imgD, 0, 0);
    // Force each tile to exactly tileW × tileH — uniform block grid regardless of spatial shape
    const out = document.createElement('canvas');
    out.width = tileW; out.height = tileH;
    out.style.imageRendering = 'pixelated';
    const outCtx = out.getContext('2d');
    outCtx.imageSmoothingEnabled = false;
    outCtx.drawImage(src, 0, 0, out.width, out.height);
    return out;
}

function drawImageBatch(vizData, wrap, maxW, maxH) {
    maxW = maxW || 206;

    // ── image_batch: tile every batch item as a proper gray/rgb/rgba image ────
    if (vizData.image_type) {
        const batchData = vizData.data;
        const nShow = batchData.length;
        const imageType = vizData.image_type;
        if (nShow === 0) return;
        const gap = 2;
        const minTile = Math.round(56 * _pinTileZoom);
        const cols = Math.min(nShow, Math.max(1, Math.floor((maxW + gap) / (minTile + gap))));
        const tileW = Math.max(minTile, Math.floor((maxW - (cols - 1) * gap) / cols));
        const grid = document.createElement('div');
        grid.className = 'image-batch-grid';
        grid.style.cssText = `display:flex;flex-wrap:wrap;gap:${gap}px;`;
        for (let b = 0; b < nShow; b++) {
            grid.appendChild(_renderImageTile(batchData[b], imageType, tileW, tileW));
        }
        wrap.appendChild(grid);
        return;
    }

    // ── grid_4d: tile channels, navigate batches ──────────────────────────────
    const batchData = vizData.data;   // [n_show][C][H][W]
    const nShow    = batchData.length;
    const nBatches = vizData.n_batches;
    const C      = batchData[0].length;
    // Use shown_channels (actual data sent) for pagination — n_channels is the full tensor
    // size which may exceed what the backend serialized, creating phantom empty pages.
    const totalC = vizData.shown_channels || vizData.n_channels || C;
    if (nShow === 0 || C === 0) return;

    const outer = document.createElement('div');
    outer.className = 'lineset-wrap';

    // ── controls row ──────────────────────────────────────────────────────────
    const ctrl = document.createElement('div');
    ctrl.className = 'lineset-ctrl';

    const _dim0Lbl = (vizData.dim_labels && vizData.dim_labels[0]) || 'batch';
    let batchInp = null;
    if (nShow > 1) {
        const lbl = document.createElement('span'); lbl.className = 'lineset-ch-label'; lbl.textContent = _dim0Lbl;
        const inp = document.createElement('input');
        inp.type = 'number'; inp.className = 'lineset-ch-input';
        inp.min = 0; inp.max = nShow - 1; inp.value = 0;
        const cnt = document.createElement('span'); cnt.className = 'lineset-ch-count';
        cnt.textContent = `/ ${nBatches - 1}`;
        ctrl.appendChild(lbl); ctrl.appendChild(inp); ctrl.appendChild(cnt);
        batchInp = inp;
    }
    outer.appendChild(ctrl);

    // ── tile geometry ─────────────────────────────────────────────────────────
    const gap = 2;
    const minTile = Math.round(56 * _pinTileZoom);
    const cols = Math.max(1, Math.floor((maxW + gap) / (minTile + gap)));
    const tileW = Math.max(minTile, Math.floor((maxW - (cols - 1) * gap) / cols));
    const ctrlH = nShow > 1 ? 28 : 0;
    // Self-sizing (maxH=null): size to exactly fit all channels in one page.
    // Fixed height (maxH given): compute rows from available height and paginate.
    const nRowsForAll = Math.max(1, Math.ceil(totalC / cols));
    const availH = maxH
        ? Math.max(minTile, maxH - ctrlH - gap)
        : nRowsForAll * (tileW + gap) - gap;
    const rowsPerPage = Math.max(1, Math.floor((availH + gap) / (tileW + gap)));
    const tilesPerPage = cols * rowsPerPage;
    const nPages = Math.ceil(totalC / tilesPerPage);

    // ── page spinner (only when channels exceed one page) ─────────────────────
    let pageInp = null, pageRangeLbl = null;
    if (nPages > 1) {
        const spacer = document.createElement('span'); spacer.style.flex = '1';
        ctrl.appendChild(spacer);
        const lbl = document.createElement('span'); lbl.className = 'lineset-ch-label'; lbl.textContent = 'pg';
        const inp = document.createElement('input');
        inp.type = 'number'; inp.className = 'lineset-ch-input';
        inp.min = 0; inp.max = nPages - 1; inp.value = 0;
        const cnt = document.createElement('span'); cnt.className = 'lineset-ch-count';
        ctrl.appendChild(lbl); ctrl.appendChild(inp); ctrl.appendChild(cnt);
        pageInp = inp; pageRangeLbl = cnt;
    }

    const gridWrap = document.createElement('div');
    outer.appendChild(gridWrap);

    function render() {
        gridWrap.innerHTML = '';
        const b  = batchInp ? Math.max(0, Math.min(parseInt(batchInp.value) || 0, nShow  - 1)) : 0;
        const pg = pageInp  ? Math.max(0, Math.min(parseInt(pageInp.value)  || 0, nPages - 1)) : 0;
        const start = pg * tilesPerPage;
        const end   = Math.min(start + tilesPerPage, totalC);
        if (pageRangeLbl) pageRangeLbl.textContent = `ch ${start}–${end - 1} / ${totalC - 1}`;
        const grid = document.createElement('div');
        grid.style.cssText = `display:flex;flex-wrap:wrap;gap:${gap}px;`;
        for (let c = start; c < end; c++) {
            if (c < C) grid.appendChild(_renderGrayscaleTile(batchData[b][c], tileW, tileW));
        }
        // Pad last row so the grid is always a complete rectangle
        const rendered = end - start;
        const remainder = rendered % cols;
        if (remainder > 0) {
            for (let i = 0; i < cols - remainder; i++) {
                const ph = document.createElement('div');
                ph.style.cssText = `width:${tileW}px;height:${tileW}px;flex-shrink:0;`;
                grid.appendChild(ph);
            }
        }
        gridWrap.appendChild(grid);
    }

    if (batchInp) {
        batchInp.classList.add('batch-sync-inp');
        function _onBatchChange() {
            render();
            if (_syncBatchEnabled && !_isBatchSyncing) {
                _isBatchSyncing = true;
                const val = parseInt(batchInp.value) || 0;
                document.querySelectorAll('.batch-sync-inp').forEach(inp => {
                    if (inp === batchInp) return;
                    inp.value = Math.min(val, parseInt(inp.max) || 0);
                    inp.dispatchEvent(new Event('input'));
                });
                _isBatchSyncing = false;
            }
        }
        batchInp.addEventListener('input', _onBatchChange);
        batchInp.addEventListener('change', _onBatchChange);
    }
    if (pageInp)  { pageInp.addEventListener('input',  render); pageInp.addEventListener('change',  render); }
    render();
    wrap.appendChild(outer);
}

// ─── Plotly-based rich rendering for pin-dashboard cards ──────────────────────
function _plotlyLayout() {
    return {
        margin: { l: 32, r: 6, t: 6, b: 24 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(245,245,247,0.6)',
        font: { family: "'SF Mono','Fira Code','Consolas',monospace", size: 10, color: '#6e6e73' },
        xaxis: { tickfont: { size: 9 }, gridcolor: '#e0e0e5', zerolinecolor: '#c8c8ce', linecolor: '#d1d1d6' },
        yaxis: { tickfont: { size: 9 }, gridcolor: '#e0e0e5', zerolinecolor: '#c8c8ce', linecolor: '#d1d1d6' },
        showlegend: false,
        hovermode: 'x unified',
    };
}

// Creates a full-size child div inside `wrap` and registers it as wrap._plotlyEl.
// Prevents Plotly's aspect-ratio logic from being centered by justify-content.
function _matPlotDiv(wrap) {
    const d = document.createElement('div');
    d.style.cssText = 'width:100%;height:100%;min-width:0;min-height:0';
    wrap.appendChild(d);
    wrap._plotlyEl = d;
    return d;
}

function _matLayout() {
    return {
        margin: { l: 6, r: 6, t: 6, b: 6 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor:  'rgba(0,0,0,0)',
        font: { family: "'SF Mono','Fira Code','Consolas',monospace", size: 9, color: '#6e6e73' },
        xaxis: { showticklabels: false, showgrid: false, zeroline: false, ticks: '' },
        yaxis: { showticklabels: false, showgrid: false, zeroline: false, ticks: '', autorange: 'reversed' },
        showlegend: false,
    };
}
const _plyCfg = {
    responsive: true,
    displayModeBar: 'hover',
    modeBarButtonsToRemove: ['sendDataToCloud', 'editInChartStudio', 'toImage', 'lasso2d', 'select2d'],
    displaylogo: false,
    scrollZoom: true,
};

function _purgePlotlyIn(wrap) {
    if (typeof Plotly === 'undefined') return;
    [wrap, wrap._plotlyEl, ...wrap.querySelectorAll('.js-plotly-plot')].filter(Boolean).forEach(el => {
        try { Plotly.purge(el); } catch (_) {}
    });
}

function _relayoutPlotlyIn(wrap) {
    // Re-render canvas-based grids (grid_4d / image_batch) on resize, using actual card height
    if (wrap._lastData && (wrap._lastData.kind === 'grid_4d' || wrap._lastData.kind === 'image_batch')) {
        const h = wrap.offsetHeight;
        _purgePlotlyIn(wrap);
        wrap.innerHTML = '';
        drawImageBatch(wrap._lastData, wrap, (wrap.offsetWidth || 300) - 4, h > 0 ? h : null);
        return;
    }
    if (typeof Plotly === 'undefined') return;
    [wrap, wrap._plotlyEl, ...wrap.querySelectorAll('.js-plotly-plot')].filter(Boolean).forEach(el => {
        if (!el.layout) return;
        try {
            if (Plotly.Plots && Plotly.Plots.resize) Plotly.Plots.resize(el);
            else Plotly.relayout(el, { autosize: true });
        } catch (_) {}
    });
}

function _renderIntoPlotly(data, wrap, opts = {}) {
    _purgePlotlyIn(wrap);
    wrap.innerHTML = '';
    wrap._lastData = data;
    wrap._lastOpts = opts;

    if (!data || (data.error && !data.data)) {
        if (data && data.error) wrap.innerHTML = `<span class="viz-error">${data.error}</span>`;
        return;
    }

    // modular node-view payloads (activations) are rendered by the shared TBViews
    // renderer; the legacy kind-based path below stays for weights.
    if (data.view && window.TBViews) {
        TBViews.render(data, wrap, { original: opts.originalData || null, detailed: opts.detailed,
                                     imgZoom: opts.imgZoom, stateKey: opts.stateKey,
                                     audioFetch: data.is_audio_compatible
                                         ? _activationAudioFetch(opts.audioNode) : null });
        return;
    }

    if (typeof Plotly === 'undefined') {
        _renderInto(data, wrap, Math.max(80, (wrap.offsetWidth || 330) - 8), null);
        return;
    }

    const kind = data.kind;

    if (kind === 'scalar') { drawScalar(data.data, wrap); return; }

    if (kind === 'bar') {
        const orig = opts.originalData && opts.originalData.kind === 'bar' ? opts.originalData : null;
        const vals = data.data;
        const traces = [];
        if (orig) {
            traces.push({
                y: orig.data, x: orig.data.map((_, i) => i), type: 'bar', name: 'orig',
                marker: { color: 'rgba(44,123,229,0.55)' },
                hovertemplate: '[%{x}] %{y:.4f}<extra>orig</extra>',
            });
        }
        traces.push({
            y: vals, x: vals.map((_, i) => i), type: 'bar', name: orig ? 'bent' : '',
            marker: { color: orig ? vals.map(() => 'rgba(255,215,0,0.85)') : vals.map(v => v >= 0 ? '#2c7be5' : '#E74C3C') },
            hovertemplate: '[%{x}] %{y:.4f}<extra></extra>',
        });
        Plotly.newPlot(wrap, traces, { ..._plotlyLayout(), bargap: 0.05, barmode: 'overlay' }, _plyCfg);
        return;
    }

    if (kind === 'line') {
        const orig = opts.originalData && opts.originalData.kind === 'line' ? opts.originalData : null;
        const traces = [];
        if (orig) {
            traces.push({
                y: orig.data, type: 'scatter', mode: 'lines', name: 'orig',
                line: { color: '#4A90D9', width: 1.5 },
                fill: 'tozeroy', fillcolor: 'rgba(74,144,217,0.15)',
                hovertemplate: '[%{x}] %{y:.4f}<extra>orig</extra>',
            });
        }
        traces.push({
            y: data.data, type: 'scatter', mode: 'lines', name: orig ? 'bent' : '',
            line: { color: orig ? '#FFD700' : '#2c7be5', width: 1.5 },
            fill: 'tozeroy', fillcolor: orig ? 'rgba(255,215,0,0.18)' : 'rgba(44,123,229,0.18)',
            hovertemplate: '[%{x}] %{y:.4f}<extra></extra>',
        });
        Plotly.newPlot(wrap, traces, _plotlyLayout(), _plyCfg);
        return;
    }

    if (kind === 'lineset') {
        const n = data.data.length;
        let selectedCh = 0, isUnfolded = false;

        // Compute global y-range once so all channels share the same axis scale.
        let globalMin = Infinity, globalMax = -Infinity;
        for (let i = 0; i < n; i++) {
            for (const v of data.data[i]) {
                if (v < globalMin) globalMin = v;
                if (v > globalMax) globalMax = v;
            }
        }
        const _yPad = Math.max(1e-6, (globalMax - globalMin) * 0.05);
        const _yRange = [globalMin - _yPad, globalMax + _yPad];

        const outer = document.createElement('div');
        outer.style.cssText = 'display:flex;flex-direction:column;gap:4px;width:100%;height:100%';

        const _linesetDim0 = (data.dim_labels && data.dim_labels[0]) || 'batch';
        const ctrl = document.createElement('div');
        ctrl.className = 'lineset-ctrl';
        const lbl = document.createElement('span'); lbl.className = 'lineset-ch-label'; lbl.textContent = _linesetDim0;
        const inp = document.createElement('input');
        inp.type = 'number'; inp.className = 'lineset-ch-input';
        inp.min = 0; inp.max = n - 1; inp.value = 0;
        const cnt = document.createElement('span'); cnt.className = 'lineset-ch-count';
        cnt.textContent = `/ ${n - 1}`;
        ctrl.appendChild(lbl); ctrl.appendChild(inp); ctrl.appendChild(cnt);
        if (n > 1) {
            const spacer = document.createElement('span'); spacer.style.flex = '1';
            ctrl.appendChild(spacer);
            const foldBtn = document.createElement('button');
            foldBtn.className = 'lineset-fold-btn';
            foldBtn.textContent = '⊞'; foldBtn.title = `Expand all ${_linesetDim0}s`;
            foldBtn.addEventListener('click', () => setUnfolded(!isUnfolded));
            ctrl.appendChild(foldBtn);
        }

        const contentWrap = document.createElement('div');
        contentWrap.style.cssText = 'flex:1;min-height:0;overflow:hidden;position:relative';

        const plotDiv = document.createElement('div');
        plotDiv.style.cssText = 'width:100%;height:100%';

        const foldCol = document.createElement('div');
        foldCol.className = 'lineset-fold-col';
        foldCol.style.display = 'none';

        contentWrap.appendChild(plotDiv);
        contentWrap.appendChild(foldCol);
        outer.appendChild(ctrl);
        outer.appendChild(contentWrap);
        wrap.appendChild(outer);
        wrap._plotlyEl = plotDiv;

        const MAX_BG = 12, MAX_UNFOLD = 32;

        function _linesetLayout() {
            const base = _plotlyLayout();
            const h = contentWrap.offsetHeight;
            return {
                ...base,
                ...(h > 0 ? { height: h, autosize: false } : {}),
                yaxis: { ...base.yaxis, range: _yRange },
            };
        }

        const _lsOrig = opts.originalData && opts.originalData.kind === 'lineset' ? opts.originalData : null;
        function buildTraces(ch) {
            const traces = [];
            if (n > 1) {
                const step = Math.max(1, Math.ceil(n / MAX_BG));
                // Collect background channels sorted far-to-near so closer ones render on top
                const bgChs = [];
                for (let i = 0; i < n; i += step) { if (i !== ch) bgChs.push(i); }
                bgChs.sort((a, b) => Math.abs(b - ch) - Math.abs(a - ch));
                for (const i of bgChs) {
                    const dist = Math.abs(i - ch) / Math.max(n - 1, 1);
                    const alpha = _lsOrig
                        ? Math.max(0.05, 0.32 * (1 - dist * 0.75))
                        : Math.max(0.05, 0.22 * (1 - dist * 0.75));
                    const w = Math.max(0.4, 1.1 * (1 - dist * 0.65));
                    traces.push({ y: data.data[i], type: 'scatter', mode: 'lines',
                        line: { color: _lsOrig ? `rgba(255,215,0,${alpha.toFixed(2)})` : `rgba(44,123,229,${alpha.toFixed(2)})`, width: w },
                        hoverinfo: 'skip', showlegend: false });
                }
            }
            const safeOrig = _lsOrig ? _lsOrig.data[Math.max(0, Math.min(ch, _lsOrig.data.length - 1))] : null;
            if (safeOrig) {
                traces.push({ y: safeOrig, type: 'scatter', mode: 'lines', name: 'orig',
                    line: { color: '#4A90D9', width: 1.5 },
                    fill: 'tozeroy', fillcolor: 'rgba(74,144,217,0.15)',
                    hovertemplate: `${_linesetDim0} ${ch} · [%{x}] %{y:.4f}<extra>orig</extra>`,
                    showlegend: false });
            }
            traces.push({ y: data.data[Math.max(0, Math.min(ch, n - 1))],
                type: 'scatter', mode: 'lines',
                line: { color: _lsOrig ? '#FFD700' : '#2c7be5', width: 1.8 },
                fill: 'tozeroy', fillcolor: _lsOrig ? 'rgba(255,215,0,0.18)' : 'rgba(44,123,229,0.18)',
                hovertemplate: `${_linesetDim0} ${ch} · [%{x}] %{y:.4f}<extra></extra>`,
                showlegend: false });
            return traces;
        }

        function renderFolded(ch) {
            Plotly.react(plotDiv, buildTraces(ch), _linesetLayout(), _plyCfg);
        }

        function renderUnfolded() {
            foldCol.innerHTML = '';
            const availW = Math.max(80, (contentWrap.offsetWidth || 280) - 40);
            const rowH = 52;
            const step = n > MAX_UNFOLD ? Math.ceil(n / MAX_UNFOLD) : 1;
            let shown = 0;
            for (let ch = 0; ch < n && shown < MAX_UNFOLD; ch += step, shown++) {
                const row = document.createElement('div');
                row.className = 'lineset-fold-row' + (ch === selectedCh ? ' lineset-fold-selected' : '');
                const rowLabel = document.createElement('span');
                rowLabel.className = 'lineset-fold-label'; rowLabel.textContent = String(ch);
                const cw = document.createElement('div');
                cw.style.cssText = 'flex:1;min-width:0;overflow:hidden';
                drawLine(data.data[ch], cw, availW, rowH - 10);
                row.appendChild(rowLabel); row.appendChild(cw);
                const capturedCh = ch;
                row.addEventListener('click', () => {
                    selectedCh = capturedCh; inp.value = capturedCh;
                    foldCol.querySelectorAll('.lineset-fold-row').forEach(r => r.classList.remove('lineset-fold-selected'));
                    row.classList.add('lineset-fold-selected');
                });
                foldCol.appendChild(row);
            }
        }

        function setUnfolded(val) {
            isUnfolded = val;
            plotDiv.style.display = val ? 'none' : '';
            foldCol.style.display = val ? '' : 'none';
            contentWrap.style.overflowY = val ? 'auto' : 'hidden';
            const fb = ctrl.querySelector('.lineset-fold-btn');
            if (fb) { fb.textContent = val ? '⊟' : '⊞'; fb.title = val ? 'Fold channels' : 'Expand all channels'; }
            if (val) renderUnfolded(); else requestAnimationFrame(() => renderFolded(selectedCh));
        }

        requestAnimationFrame(() => renderFolded(0));
        inp.addEventListener('input',  () => { selectedCh = parseInt(inp.value) || 0; if (!isUnfolded) renderFolded(selectedCh); });
        inp.addEventListener('change', () => { selectedCh = parseInt(inp.value) || 0; if (!isUnfolded) renderFolded(selectedCh); });
        return;
    }

    if (kind === 'temporal_3d') {
        const outer = document.createElement('div');
        outer.style.cssText = 'display:flex;flex-direction:column;gap:4px;width:100%;height:100%';
        const ctrl = document.createElement('div');
        ctrl.className = 'lineset-ctrl';
        function mkSel(labelText, max) {
            const l = document.createElement('span'); l.className = 'lineset-ch-label'; l.textContent = labelText;
            const i = document.createElement('input');
            i.type = 'number'; i.className = 'lineset-ch-input'; i.min = 0; i.max = max; i.value = 0;
            const c = document.createElement('span'); c.className = 'lineset-ch-count'; c.textContent = `/ ${max}`;
            ctrl.appendChild(l); ctrl.appendChild(i); ctrl.appendChild(c);
            return i;
        }
        const bInp = mkSel((data.dim_labels && data.dim_labels[0]) || 'batch', data.n_batches - 1);
        const cInp = mkSel((data.dim_labels && data.dim_labels[1]) || 'ch',    data.n_channels - 1);
        const plotDiv = document.createElement('div');
        plotDiv.style.cssText = 'flex:1;min-height:0;width:100%';
        outer.appendChild(ctrl); outer.appendChild(plotDiv);
        wrap.appendChild(outer);
        wrap._plotlyEl = plotDiv;
        const T3D_MAX_BG = 8;
        const _t3dOrig = opts.originalData && opts.originalData.kind === 'temporal_3d' ? opts.originalData : null;
        function render() {
            const b    = Math.max(0, Math.min(parseInt(bInp.value) || 0, data.data.length - 1));
            const ch   = Math.max(0, Math.min(parseInt(cInp.value) || 0, (data.data[b] || []).length - 1));
            const chans = data.data[b] || [];
            const nc   = chans.length;
            const traces = [];
            if (nc > 1) {
                const step = Math.max(1, Math.ceil(nc / T3D_MAX_BG));
                for (let i = 0; i < nc; i += step) {
                    if (i === ch) continue;
                    traces.push({ y: chans[i], type: 'scatter', mode: 'lines',
                        line: { color: _t3dOrig ? 'rgba(255,215,0,0.10)' : 'rgba(44,123,229,0.12)', width: 0.8 },
                        hoverinfo: 'skip', showlegend: false });
                }
            }
            const origChans = _t3dOrig ? (_t3dOrig.data[b] || []) : null;
            if (origChans && origChans[ch]) {
                traces.push({ y: origChans[ch], type: 'scatter', mode: 'lines', name: 'orig',
                    line: { color: '#4A90D9', width: 1.5 },
                    fill: 'tozeroy', fillcolor: 'rgba(74,144,217,0.15)',
                    hovertemplate: `ch ${ch} · [%{x}] %{y:.4f}<extra>orig</extra>`,
                    showlegend: false });
            }
            traces.push({ y: chans[ch] || [], type: 'scatter', mode: 'lines',
                line: { color: _t3dOrig ? '#FFD700' : '#2c7be5', width: 1.8 },
                fill: 'tozeroy', fillcolor: _t3dOrig ? 'rgba(255,215,0,0.18)' : 'rgba(44,123,229,0.18)',
                hovertemplate: `ch ${ch} · [%{x}] %{y:.4f}<extra></extra>`,
                showlegend: false });
            Plotly.react(plotDiv, traces, _plotlyLayout(), _plyCfg);
        }
        requestAnimationFrame(render);
        bInp.addEventListener('input', render); bInp.addEventListener('change', render);
        cInp.addEventListener('input', render); cInp.addEventListener('change', render);
        return;
    }

    if (kind === 'heatmap') {
        const hasOrig = !!(opts.originalData && opts.originalData.kind === 'heatmap');
        const outer = document.createElement('div');
        outer.style.cssText = 'display:flex;flex-direction:column;width:100%;height:100%';
        let showOrig = false;
        const plotDiv = document.createElement('div');
        plotDiv.style.cssText = 'flex:1;min-height:0;width:100%';
        if (hasOrig) {
            const ctrl = document.createElement('div');
            ctrl.className = 'lineset-ctrl';
            const spacer = document.createElement('span'); spacer.style.flex = '1';
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'lineset-fold-btn viz-orig-toggle';
            toggleBtn.textContent = 'bent'; toggleBtn.title = 'Toggle bent / orig';
            ctrl.appendChild(spacer); ctrl.appendChild(toggleBtn);
            outer.appendChild(ctrl);
            toggleBtn.addEventListener('click', () => {
                showOrig = !showOrig;
                toggleBtn.textContent = showOrig ? 'orig' : 'bent';
                toggleBtn.classList.toggle('viz-orig-toggle--orig', showOrig);
                Plotly.react(plotDiv, [{
                    z: showOrig ? opts.originalData.data : data.data,
                    type: 'heatmap', colorscale: 'Viridis', showscale: false,
                    hovertemplate: '(%{x},%{y}): %{z:.4f}<extra></extra>',
                }], _matLayout(), _plyCfg);
            });
        }
        outer.appendChild(plotDiv);
        wrap.appendChild(outer);
        wrap._plotlyEl = plotDiv;
        requestAnimationFrame(() => Plotly.newPlot(plotDiv, [{
            z: data.data, type: 'heatmap', colorscale: 'Viridis', showscale: false,
            hovertemplate: '(%{x},%{y}): %{z:.4f}<extra></extra>',
        }], _matLayout(), _plyCfg));
        return;
    }

    if (kind === 'rgb') {
        const _chwToZ = (chw) => {
            const H = chw[0].length, W = chw[0][0].length;
            const z = [];
            for (let r = 0; r < H; r++) {
                const row = [];
                for (let c = 0; c < W; c++) {
                    row.push([
                        Math.round(Math.max(0, Math.min(1, chw[0][r][c])) * 255),
                        Math.round(Math.max(0, Math.min(1, chw[1][r][c])) * 255),
                        Math.round(Math.max(0, Math.min(1, chw[2][r][c])) * 255),
                    ]);
                }
                z.push(row);
            }
            return z;
        };
        const hasOrig = !!(opts.originalData && opts.originalData.kind === 'rgb');
        const outer = document.createElement('div');
        outer.style.cssText = 'display:flex;flex-direction:column;width:100%;height:100%';
        let showOrig = false;
        const zBent = _chwToZ(data.data);
        const plotDiv = document.createElement('div');
        plotDiv.style.cssText = 'flex:1;min-height:0;width:100%';
        if (hasOrig) {
            const ctrl = document.createElement('div');
            ctrl.className = 'lineset-ctrl';
            const spacer = document.createElement('span'); spacer.style.flex = '1';
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'lineset-fold-btn viz-orig-toggle';
            toggleBtn.textContent = 'bent'; toggleBtn.title = 'Toggle bent / orig';
            ctrl.appendChild(spacer); ctrl.appendChild(toggleBtn);
            outer.appendChild(ctrl);
            toggleBtn.addEventListener('click', () => {
                showOrig = !showOrig;
                toggleBtn.textContent = showOrig ? 'orig' : 'bent';
                toggleBtn.classList.toggle('viz-orig-toggle--orig', showOrig);
                Plotly.react(plotDiv, [{
                    z: showOrig ? _chwToZ(opts.originalData.data) : zBent,
                    type: 'image', colormodel: 'rgb',
                    hovertemplate: '(%{x},%{y})<extra></extra>',
                }], _matLayout(), _plyCfg);
            });
        }
        outer.appendChild(plotDiv);
        wrap.appendChild(outer);
        wrap._plotlyEl = plotDiv;
        requestAnimationFrame(() => Plotly.newPlot(plotDiv, [{
            z: zBent, type: 'image', colormodel: 'rgb',
            hovertemplate: '(%{x},%{y})<extra></extra>',
        }], _matLayout(), _plyCfg));
        return;
    }

    if (kind === 'grid') {
        // N × H × W — one heatmap per channel, with channel spinner
        const slices = data.data;
        const n = slices.length;
        const hasOrig = !!(opts.originalData && opts.originalData.kind === 'grid');
        let showOrig = false;

        const outer = document.createElement('div');
        outer.style.cssText = 'display:flex;flex-direction:column;width:100%;height:100%';

        function _mkOrigToggle(ctrl) {
            const spacer = document.createElement('span'); spacer.style.flex = '1';
            ctrl.appendChild(spacer);
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'lineset-fold-btn viz-orig-toggle';
            toggleBtn.textContent = 'bent'; toggleBtn.title = 'Toggle bent / orig';
            ctrl.appendChild(toggleBtn);
            return toggleBtn;
        }

        let plotDiv;
        if (n > 1) {
            const ctrl = document.createElement('div');
            ctrl.className = 'lineset-ctrl';
            const lbl = document.createElement('span'); lbl.className = 'lineset-ch-label'; lbl.textContent = 'ch';
            const inp = document.createElement('input');
            inp.type = 'number'; inp.className = 'lineset-ch-input';
            inp.min = 0; inp.max = n - 1; inp.value = 0;
            const cnt = document.createElement('span'); cnt.className = 'lineset-ch-count';
            cnt.textContent = `/ ${n - 1}`;
            ctrl.appendChild(lbl); ctrl.appendChild(inp); ctrl.appendChild(cnt);
            if (hasOrig) {
                const tb = _mkOrigToggle(ctrl);
                tb.addEventListener('click', () => {
                    showOrig = !showOrig;
                    tb.textContent = showOrig ? 'orig' : 'bent';
                    tb.classList.toggle('viz-orig-toggle--orig', showOrig);
                    renderGrid();
                });
            }
            outer.appendChild(ctrl);

            plotDiv = document.createElement('div');
            plotDiv.style.cssText = 'flex:1;min-height:0;width:100%';
            outer.appendChild(plotDiv);

            function renderGrid() {
                const ch = Math.max(0, Math.min(parseInt(inp.value) || 0, n - 1));
                const activeSlices = showOrig ? opts.originalData.data : slices;
                Plotly.react(plotDiv, [{
                    z: activeSlices[ch], type: 'heatmap', colorscale: 'Viridis', showscale: false,
                    hovertemplate: '(%{x},%{y}): %{z:.4f}<extra></extra>',
                }], _matLayout(), _plyCfg);
            }
            requestAnimationFrame(renderGrid);
            inp.addEventListener('input',  renderGrid);
            inp.addEventListener('change', renderGrid);
        } else {
            plotDiv = document.createElement('div');
            plotDiv.style.cssText = 'width:100%;height:100%';
            if (hasOrig) {
                const ctrl = document.createElement('div');
                ctrl.className = 'lineset-ctrl';
                const tb = _mkOrigToggle(ctrl);
                outer.appendChild(ctrl);
                tb.addEventListener('click', () => {
                    showOrig = !showOrig;
                    tb.textContent = showOrig ? 'orig' : 'bent';
                    tb.classList.toggle('viz-orig-toggle--orig', showOrig);
                    Plotly.react(plotDiv, [{
                        z: showOrig ? opts.originalData.data[0] : slices[0],
                        type: 'heatmap', colorscale: 'Viridis', showscale: false,
                        hovertemplate: '(%{x},%{y}): %{z:.4f}<extra></extra>',
                    }], _matLayout(), _plyCfg);
                });
            }
            outer.appendChild(plotDiv);
            requestAnimationFrame(() => Plotly.newPlot(plotDiv, [{
                z: slices[0], type: 'heatmap', colorscale: 'Viridis', showscale: false,
                hovertemplate: '(%{x},%{y}): %{z:.4f}<extra></extra>',
            }], _matLayout(), _plyCfg));
        }

        wrap.appendChild(outer);
        wrap._plotlyEl = plotDiv;
        return;
    }

    if (kind === 'grid_4d' || kind === 'image_batch') {
        drawImageBatch(data, wrap, (wrap.offsetWidth || 300) - 4, null);
        return;
    }

    wrap.innerHTML = `<span class="viz-error">unsupported kind: ${kind}</span>`;
}

function _renderInto(data, wrap, maxW, maxH) {
    wrap.innerHTML = "";
    if (data.view && window.TBViews) {   // modular node-view payload (activations)
        TBViews.render(data, wrap, {});
        return;
    }
    if (data.error && !data.data) {
        wrap.innerHTML = `<span class="viz-error">${data.error}</span>`;
        return;
    }
    const kind = data.kind;
    if (kind === "scalar")      return drawScalar(data.data, wrap);
    if (kind === "bar")         return drawBar(data.data, wrap, maxW);
    if (kind === "line")        return drawLine(data.data, wrap, maxW, maxH);
    if (kind === "lineset")     return drawLineset(data.data, wrap, maxW, data.n_batches, maxH, data.dim_labels && data.dim_labels[0]);
    if (kind === "temporal_3d") return drawTemporal3d(data, wrap, maxW, maxH);
    if (kind === "heatmap")     return drawHeatmap(data.data, wrap, maxW, maxH);
    if (kind === "rgb")         return drawRGB(data.data, wrap, maxW, maxH);
    if (kind === "grid")        return drawGrid(data.data, wrap, maxW, maxH);
    if (kind === "grid_4d")    return drawImageBatch(data, wrap, maxW, maxH);
    if (kind === "image_batch") return drawImageBatch(data, wrap, maxW, maxH);
    wrap.innerHTML = `<span class="viz-error">unsupported kind: ${kind}</span>`;
}

function _updateAudioBtn(data, label) {
    let btn = document.getElementById("viz-audio-btn");
    if (!btn) {
        btn = document.createElement("button");
        btn.id = "viz-audio-btn";
        btn.className = "viz-audio-btn";
        btn.title = "Play as audio";
        btn.textContent = "▶";
        const headerBtns = document.querySelector(".viz-header-btns");
        if (headerBtns) headerBtns.insertBefore(btn, headerBtns.firstChild);
    }
    let dlBtn = document.getElementById("viz-audio-dl-btn");
    if (!dlBtn) {
        dlBtn = document.createElement("button");
        dlBtn.id = "viz-audio-dl-btn";
        dlBtn.className = "viz-audio-btn";
        dlBtn.title = "Download as WAV";
        dlBtn.textContent = "⬇";
        const headerBtns = document.querySelector(".viz-header-btns");
        if (headerBtns) headerBtns.insertBefore(dlBtn, headerBtns.firstChild);
    }
    if (data && data.is_audio_compatible && label && hasAnyInput()) {
        btn.style.display = "";
        btn.onclick = () => _playActivationAudio(label);
        dlBtn.style.display = "";
        dlBtn.onclick = () => _downloadActivationAudio(label);
    } else {
        btn.style.display = "none";
        btn.onclick = null;
        dlBtn.style.display = "none";
        dlBtn.onclick = null;
    }
}

// Rendering an activation as sound is a server round trip — the tensor lives
// there, and the clip has to be the batch and channel the card is showing. This
// is the same contract play mode hands its outputs, so TBViews gives a pinned
// activation the same playable strips: one per batch, and one per channel when
// the channel axis is listed.
function _activationAudioFetch(nodeName) {
    if (!nodeName || !currentFn || !hasAnyInput()) return null;
    return async (sel) => {
        const form = buildActivationForm();
        if (sel) {
            form.append("batch_idx", String(sel.batch));
            form.append("channel", String(sel.channel));
        }
        const r = await fetch(
            `/api/activate_audio/${encodeURIComponent(currentFn)}/${encodeURIComponent(nodeName)}/`,
            { method: "POST", body: form });
        if (!r.ok) {
            const d = await r.json().catch(() => ({}));
            const err = new Error(d.error || r.statusText);
            err.traceback = d.traceback || "";
            throw err;
        }
        return r.blob();
    };
}

let _audioEl = null;
function _playActivationAudio(nodeName) {
    const btn = document.getElementById("viz-audio-btn");
    if (btn) { btn.textContent = "…"; btn.disabled = true; }

    fetch(`/api/activate_audio/${currentFn}/${nodeName}/`, {
        method: "POST", body: buildActivationForm(),
    })
    .then(r => {
        if (!r.ok) return r.json().then(d => { { const _e = new Error(d.error || r.statusText); _e.traceback = d.traceback || ""; throw _e; } });
        return r.blob();
    })
    .then(blob => {
        // release the previous clip's object URL, otherwise every play leaks one
        if (_audioEl) { _audioEl.pause(); URL.revokeObjectURL(_audioEl.src); _audioEl.remove(); }
        _audioEl = document.createElement("audio");
        _audioEl.controls = true;
        _audioEl.className = "viz-audio-player";
        _audioEl.src = URL.createObjectURL(blob);
        const section = document.getElementById("viz-section");
        section.appendChild(_audioEl);
        _audioEl.play().catch(() => {});
        if (btn) { btn.textContent = "▶"; btn.disabled = false; }
    })
    .catch(e => {
        showToast("error", "Audio playback failed: " + e.message);
        if (btn) { btn.textContent = "▶"; btn.disabled = false; }
    });
}

function _downloadActivationAudio(nodeName) {
    fetch(`/api/activate_audio/${currentFn}/${nodeName}/`, {
        method: "POST", body: buildActivationForm(),
    })
    .then(r => {
        if (!r.ok) return r.json().then(d => { { const _e = new Error(d.error || r.statusText); _e.traceback = d.traceback || ""; throw _e; } });
        return r.blob();
    })
    .then(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = nodeName + ".wav";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    })
    .catch(e => showToast("error", "Download failed: " + e.message));
}

function renderTensorViz(data, label, originalData = null) {
    if (_vizSectionCleanup) { _vizSectionCleanup(); _vizSectionCleanup = null; }
    currentVizData = { data, label, originalData };
    // snapshots are of activations: fetchActivation mounts the bar again after
    const snapHost = document.getElementById("viz-snap");
    if (snapHost) snapHost.innerHTML = "";
    _panelSnapNode = null;
    document.getElementById("viz-section").classList.remove("viz-frozen");

    // remove any lingering audio player
    if (_audioEl) { _audioEl.pause(); _audioEl.remove(); _audioEl = null; }

    const section = document.getElementById("viz-section");
    const wrap = document.getElementById("viz-canvas-wrap");

    section.style.display = "block";
    document.getElementById("viz-label").textContent = _vizLabelFor(label) || "data";
    document.getElementById("viz-shape").textContent = data.shape && data.shape.length
        ? `[${data.shape.join(" × ")}]` : "";

    _updateAudioBtn(data, label);

    // modular node-view picker (activation payloads carry _view_meta)
    let pickerHost = document.getElementById("viz-view-picker");
    if (!pickerHost) {
        pickerHost = document.createElement("div");
        pickerHost.id = "viz-view-picker";
        wrap.parentNode.insertBefore(pickerHost, wrap);
    }
    pickerHost.innerHTML = "";
    const vmeta = data._view_meta;
    if (vmeta && window.TBViews && vmeta.compatible && vmeta.compatible.length > 1) {
        pickerHost.appendChild(TBViews.renderPicker(vmeta, async (view, options) => {
            try {
                await fetch(`/api/views/${encodeURIComponent(currentFn)}/${encodeURIComponent(label)}/`, {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ view, options }),
                });
                fetchActivation(currentFn, label);   // re-fetch + re-render with the chosen view
            } catch (e) { showToast("error", e.message); }
        }));
    }

    _purgePlotlyIn(wrap);
    wrap.innerHTML = "";
    if (data.view && window.TBViews) {
        // modular view payload — lightweight canvas preview in the detail panel
        // (expand modal & pins use the richer Plotly render instead)
        wrap.style.height = "";
        wrap.style.display = "";
        TBViews.render(data, wrap, { original: originalData || null, detailed: false,
                                     stateKey: currentFn + "/" + label,
                                     audioFetch: data.is_audio_compatible
                                         ? _activationAudioFetch(label) : null });
    } else if (typeof Plotly !== "undefined") {
        wrap.style.height = "160px";
        wrap.style.display = "block";
        _renderIntoPlotly(data, wrap, Object.assign(originalData ? { originalData } : {},
                                                    { stateKey: currentFn + "/" + label,
                                                      audioNode: label }));
    } else {
        wrap.style.height = "";
        wrap.style.display = "";
        _renderInto(data, wrap, 206);
    }
}

// ─── activation snapshots ────────────────────────────────────────────────────
// A view of an activation (the panel, the expand modal, a dashboard card) can
// save what it shows under a name. The copy lives on the server, whole -- it is
// what activation interpolation works from. Recalling a snapshot puts it back
// *into the graph*: it becomes a Snapshot bending on the node, so the node takes
// the saved value and everything computed after it follows (its `mix` slider,
// in the bendings list, crossfades back to the live value). "live" takes it off;
// deleting a snapshot takes it off every node it was recalled onto.
let _snapshots = [];          // [{name, fn, node, shape, …}] — the server's list
let _panelSnapNode = null;    // {fn, node} the panel's snapshot bar is for
const _snapBars = new Set();  // every bar on screen, to redraw when the list changes

async function _loadSnapshots() {
    try {
        const r = await fetch("/api/snapshots/");
        _snapshots = (await r.json()).snapshots || [];
    } catch (_) { _snapshots = []; }
    _syncSnapBars();
}

function _snapshotsOf(fn, node) {
    return _snapshots.filter(s => s.fn === fn && s.node === node);
}

async function _fetchSnapshot(name) {
    const r = await fetch(`/api/snapshots/${encodeURIComponent(name)}/`);
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    return d;
}

// A name that is safe anywhere a snapshot may end up (a file, a key): letters,
// digits, "-" and "_", accents dropped, at most 40 characters.
function _safeSnapName(text) {
    return String(text || "").normalize("NFKD").replace(/[\u0300-\u036f]/g, "")
        .replace(/[^A-Za-z0-9_-]+/g, "_").replace(/_+/g, "_").replace(/^_+|_+$/g, "")
        .slice(0, 40).replace(/_+$/, "");
}

// What the graph runs on, as a name: the first input (in graph order) that is
// a file -- its name without the extension -- or a text, as typed.
function _firstInputLabel() {
    const order = currentGraphData
        ? currentGraphData.nodes.filter(n => n.op === "placeholder").map(n => n.id) : [];
    const names = order.concat(Object.keys(inputSets).filter(n => !order.includes(n)));
    for (const name of names) {
        const arr = inputSets[name] || [];
        const used = _batchActive() ? _batchEntries(name)[0]
            : (arr.find(i => i.id === selectedIds[name] && i.valid) || arr.find(i => i.valid));
        if (!used) continue;
        if (used.file && used.file.name) return used.file.name.replace(/\.[^.]+$/, "");
        if (used.type === "mode" && (used.value || "").trim()) return used.value.trim();
    }
    return null;
}

// After the first input when it is a file or a text ("rain", then "rain_2"…),
// else after the node's alias or name ("speech_1"…).
function _defaultSnapName(node) {
    const taken = new Set(_snapshots.map(s => s.name));
    const input = _safeSnapName(_firstInputLabel());
    if (input) {
        if (!taken.has(input)) return input;
        let k = 2;
        while (taken.has(`${input}_${k}`)) k++;
        return `${input}_${k}`;
    }
    const aliases = currentGraphData ? _nodeAliases({ id: node, label: node }, currentGraphData) : [];
    const base = _safeSnapName(aliases.length ? aliases[0] : node) || "snapshot";
    let k = 1;
    while (taken.has(`${base}_${k}`)) k++;
    return `${base}_${k}`;
}

async function _saveSnapshot(fn, node) {
    if (!hasAnyInput()) { showToast("info", "Add an input first: a snapshot is of what the graph computes"); return null; }
    const name = (window.prompt(`Save a snapshot of '${node}' as:`, _defaultSnapName(node)) || "").trim();
    if (!name) return null;
    const post = async (overwrite) => {
        const form = buildActivationForm([node, node + "_bended"]);
        form.append("fn", fn);
        form.append("node", node);
        form.append("name", name);
        if (overwrite) form.append("overwrite", "true");
        const r = await fetch("/api/snapshots/", { method: "POST", body: form });
        return [r, await r.json()];
    };
    try {
        let [r, d] = await post(false);
        if (r.status === 409 && d.exists) {
            if (!window.confirm(`There is already a snapshot named '${name}'. Replace it?`)) return null;
            [r, d] = await post(true);
        }
        if (d.error) throw new Error(d.error);
        showToast("info", `Saved snapshot '${name}'`);
        await _loadSnapshots();
        return name;
    } catch (e) {
        showToast("error", "Snapshot failed: " + e.message);
        return null;
    }
}

// The snapshot recalled onto a node, read from the bindings (the server's state).
function _recalledOn(fn, node) {
    const b = _bendingBindings.find(b => b.snapshot && b.fn === fn
        && (b.nodes || [b.node]).includes(node));
    return b ? b.snapshot : null;
}

// A change of bendings came back from the server: take it in, and refresh every
// view -- what comes after a recalled node is computed anew from it.
async function _afterSnapshotBindings(d) {
    _syncBendingState(d);
    _refreshBentNodeStyles();
    _refreshAllBendingUI();
    _syncSnapBars();
    _syncRecalledMarks();
    await _refreshLiveViews();
    _refreshModalIfOpen();
}

async function _postSnapshotJSON(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                                 body: JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok || d.error) throw Object.assign(new Error(d.error || r.statusText), { traceback: d.traceback });
    return d;
}

async function _recallSnapshot(fn, node, name) {
    try {
        const d = await _postSnapshotJSON(`/api/snapshots/${encodeURIComponent(name)}/recall/`, { fn, node });
        showToast("info", `'${name}' recalled into ${node} — what follows is computed from it`);
        await _afterSnapshotBindings(d);
    } catch (e) {
        showToast("error", "Recall failed: " + e.message, e.traceback);
        _syncSnapBars();
    }
}

async function _releaseSnapshot(fn, node) {
    try {
        const d = await _postSnapshotJSON("/api/snapshots/release/", { fn, node });
        await _afterSnapshotBindings(d);
    } catch (e) {
        showToast("error", e.message, e.traceback);
        _syncSnapBars();
    }
}

async function _deleteSnapshot(name) {
    if (!window.confirm(`Delete snapshot '${name}'?`)) return;
    let d;
    try {
        const r = await fetch(`/api/snapshots/${encodeURIComponent(name)}/`, { method: "DELETE" });
        d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || r.statusText);
    } catch (e) { showToast("error", e.message); return; }
    await _loadSnapshots();
    // recalled somewhere: those nodes follow the graph again
    if ((d.released || []).length) await _afterSnapshotBindings(d);
}

// The same controls for every view of a node: they act on the graph, not on
// the view.
function _snapCtx(fn, node) {
    return {
        fn, node,
        get: () => _recalledOn(fn, node),
        set: (name) => name ? _recallSnapshot(fn, node, name) : _releaseSnapshot(fn, node),
    };
}

// save / recall / let go / delete, for one view (see _snapCtx).
function _snapBar(ctx) {
    const bar = document.createElement("div");
    bar.className = "snap-bar";
    // a dashboard card drags by its header: the bar's controls must not start one
    bar.addEventListener("mousedown", (e) => e.stopPropagation());
    const render = () => {
        bar.innerHTML = "";
        const recalled = ctx.get();
        const save = document.createElement("button");
        save.className = "snap-rec-btn";
        save.innerHTML = '<span class="snap-rec-dot"></span>rec';
        save.title = "Save a snapshot of this activation — a named copy that can be recalled into the graph, and that interpolation can use";
        save.addEventListener("click", async (e) => { e.stopPropagation(); await _saveSnapshot(ctx.fn, ctx.node); });
        bar.appendChild(save);
        const snaps = _snapshotsOf(ctx.fn, ctx.node);
        const sel = document.createElement("select");
        sel.className = "snap-select";
        sel.appendChild(new Option(snaps.length ? "live" : "live (no snapshot yet)", ""));
        snaps.forEach(sn => sel.appendChild(new Option("❄ " + sn.name, sn.name)));
        sel.value = recalled || "";
        sel.disabled = !snaps.length && !recalled;
        sel.title = recalled
            ? `'${recalled}' is recalled into the graph here: this node takes its value, and everything after it follows. Pick "live" to take it off.`
            : "Recall a snapshot into the graph: this node takes the saved value, and everything after it follows";
        sel.addEventListener("click", (e) => e.stopPropagation());
        sel.addEventListener("change", () => ctx.set(sel.value || null));
        bar.appendChild(sel);
        if (recalled) {
            const del = document.createElement("button");
            del.className = "snap-del-btn";
            del.textContent = "✕ delete";
            del.title = `Delete snapshot '${recalled}' — this node follows the graph again`;
            del.addEventListener("click", (e) => { e.stopPropagation(); _deleteSnapshot(recalled); });
            bar.appendChild(del);
        }
        const status = document.createElement("span");
        status.className = "snap-status";
        status.textContent = recalled ? "recalled — what follows uses it" : "following the graph";
        bar.appendChild(status);
        bar.classList.toggle("snap-frozen", !!recalled);
    };
    bar._render = render;
    render();
    _snapBars.add(bar);
    return bar;
}

function _syncSnapBars() {
    for (const bar of [..._snapBars]) {
        if (bar.isConnected) bar._seen = true;
        else if (bar._seen) { _snapBars.delete(bar); continue; }   // its view is gone
        bar._render();
    }
}

// Mark the views of recalled nodes: the panel, and the dashboard's cards.
function _syncRecalledMarks() {
    const panel = document.getElementById("viz-section");
    const onPanel = _panelSnapNode && _recalledOn(_panelSnapNode.fn, _panelSnapNode.node);
    if (panel) panel.classList.toggle("viz-frozen", !!onPanel);
    pinPages.forEach(page => page.forEach(pin => {
        const card = document.getElementById(`pin-card-${pin.id}`);
        if (card) card.classList.toggle("pin-frozen", !!(pin.label && _recalledOn(currentFn, pin.label)));
    }));
}

// the panel's bar, for the node the panel shows
function _mountPanelSnap(fn, node) {
    const host = document.getElementById("viz-snap");
    if (!host) return;
    host.innerHTML = "";
    _panelSnapNode = { fn, node };
    host.appendChild(_snapBar(_snapCtx(fn, node)));
    const recalled = _recalledOn(fn, node);
    document.getElementById("viz-section").classList.toggle("viz-frozen", !!recalled);
    if (recalled) document.getElementById("viz-label").textContent += `  ❄ ${recalled}`;
}

function _refreshModalIfOpen() {
    const modal = document.getElementById("viz-modal");
    if (modal && modal.style.display !== "none") openExpandModal();
}

function openExpandModal() {
    if (!currentVizData) return;
    const { data, label } = currentVizData;

    document.getElementById("viz-modal-label").textContent = label || "data";
    document.getElementById("viz-modal-shape").textContent = data.shape && data.shape.length
        ? `[${data.shape.join(" × ")}]` : "";

    const modalWrap = document.getElementById("viz-modal-wrap");
    _purgePlotlyIn(modalWrap);
    modalWrap.innerHTML = "";
    _renderIntoPlotly(data, modalWrap, { detailed: true, audioNode: label,
                                        stateKey: currentFn + "/" + (currentVizData.label || "") });   // richer interactive plots when expanded

    // the same snapshot controls as the panel it expands: they act on the panel
    const snapHost = document.getElementById("viz-modal-snap");
    if (snapHost) {
        snapHost.innerHTML = "";
        if (_panelSnapNode && _panelSnapNode.node === label) {
            const { fn, node } = _panelSnapNode;
            snapHost.appendChild(_snapBar(_snapCtx(fn, node)));
            const recalled = _recalledOn(fn, node);
            if (recalled) document.getElementById("viz-modal-label").textContent += `  ❄ ${recalled}`;
        }
    }

    document.getElementById("viz-modal").style.display = "flex";
}

function clearViz() {
    if (_vizSectionCleanup) { _vizSectionCleanup(); _vizSectionCleanup = null; }
    currentVizNode = null;
    currentVizActId = null;
    currentVizData = null;
    if (_audioEl) { _audioEl.pause(); _audioEl.remove(); _audioEl = null; }
    _updateAudioBtn(null, null);
    const section = document.getElementById("viz-section");
    section.style.display = "none";
    const wrap = document.getElementById("viz-canvas-wrap");
    _purgePlotlyIn(wrap);
    wrap.innerHTML = "";
    wrap.style.height = "";
    wrap.style.display = "";
}

// Fetch weight for a get_attr node and render it
function fetchWeight(fn, nodeName, signal) {
    const wrap = document.getElementById("viz-canvas-wrap");
    const section = document.getElementById("viz-section");
    section.style.display = "block";
    document.getElementById("viz-label").textContent = _vizLabelFor(nodeName);
    document.getElementById("viz-shape").textContent = "";
    wrap.innerHTML = `<span class="viz-loading">loading…</span>`;

    const hasBending = _bendingBindings.some(b =>
        b.node === nodeName || b.node.replace(/\./g, '_') === nodeName
    );
    const bentP = fetch(`/api/weights/${fn}/${nodeName}/`, { signal }).then(r => r.json());
    const origP = hasBending
        ? fetch(`/api/weights/${fn}/${nodeName}/?original=true`, { signal }).then(r => r.json())
        : Promise.resolve(null);

    return Promise.all([bentP, origP])
        .then(([bentData, origData]) => renderTensorViz(bentData, nodeName, origData))
        .catch(e => {
            // an aborted wave was superseded — the newer one owns the panel now
            if (!_isAbort(e)) wrap.innerHTML = `<span class="viz-error">${e.message}</span>`;
        });
}

// Fetch activation for the given node using current inputs
// Every panel fetch is numbered; only the latest may draw. A slower, older one
// -- a refresh still downloading a snapshot that was just deleted, or a live
// value that lands after a recall -- would otherwise draw over the newer state.
let _panelFetchSeq = 0;

async function fetchActivation(fn, nodeName, signal) {
    const seq = ++_panelFetchSeq;
    if (!hasAnyInput()) return;

    const wrap = document.getElementById("viz-canvas-wrap");
    const section = document.getElementById("viz-section");
    section.style.display = "block";
    document.getElementById("viz-label").textContent = _vizLabelFor(nodeName);
    document.getElementById("viz-shape").textContent = "";
    wrap.innerHTML = `<span class="viz-loading">running…</span>`;

    // Show original-vs-bent overlay whenever any bending is active
    const hasAnyActiveBendings = _bendingBindings.some(b => !b.vis_muted);

    try {
        const bentResp = await fetch(`/api/activate/${fn}/`, {
            method: "POST", body: buildActivationForm([nodeName, nodeName + "_bended"]), signal,
        });
        const bentActs = await bentResp.json();
        if (seq !== _panelFetchSeq) return;              // superseded
        if (bentActs.error) {
            // located the same way as a trace failure: running the model goes
            // through the very same user code
            if (bentActs.location || bentActs.last_node)
                showTraceError("Activation failed", bentActs);
            throw new Error(bentActs.error);
        }
        // Prefer the post-bending activation (nodeName_bended) over the pre-bending one
        const directlyBent = !!(bentActs[nodeName + "_bended"]);
        const bentData = bentActs[nodeName + "_bended"] || bentActs[nodeName];
        if (!bentData) throw new Error(`no activation for ${nodeName}`);

        // Only show original-vs-bent overlay when this node is directly bent
        const origData = directlyBent ? (bentActs[nodeName] || null) : null;

        renderTensorViz(bentData, nodeName, origData);
        _mountPanelSnap(fn, nodeName);
    } catch (e) {
        // an aborted wave was superseded — the newer one owns the panel now
        if (!_isAbort(e)) wrap.innerHTML = `<span class="viz-error">${e.message}</span>`;
    }
}

// Fetch result of an expression for a given node and render into vizWrap.
// Caches fetched data on vizWrap._lastData for re-render on resize.
function _fetchExprViz(nodeId, expr, vizWrap) {
    vizWrap.innerHTML = `<span class="viz-loading">…</span>`;
    fetch(`/api/eval_expr/${currentFn}/${nodeId}/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ expr }),
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        vizWrap._lastData = data;
        _renderIntoPlotly(data, vizWrap);
    })
    .catch(e => { vizWrap.innerHTML = `<span class="viz-error">${e.message}</span>`; });
}

function _rerenderVizWrap(vizWrap) {
    if (!vizWrap._lastData) return;
    // keep the options the wrap was first rendered with — dropping them lost the
    // node name behind the audio clips, and the strips went silent on a redraw
    _renderIntoPlotly(vizWrap._lastData, vizWrap, vizWrap._lastOpts || {});
}

function _rerenderAllGridCards() {
    document.querySelectorAll(".pin-viz-wrap").forEach(wrap => {
        if (!wrap._lastData) return;
        const d = wrap._lastData;
        // new modular image / channel-grid views: re-render via TBViews with the zoom
        if (d.view === 'image' || d.view === 'channel_grid') {
            const opts = Object.assign({}, wrap._lastOpts || {}, { detailed: true, imgZoom: _pinTileZoom });
            _purgePlotlyIn(wrap);
            wrap.innerHTML = '';
            _renderIntoPlotly(d, wrap, opts);
            return;
        }
        // legacy kind-based grids
        if (d.kind !== 'grid_4d' && d.kind !== 'image_batch') return;
        const h = wrap.offsetHeight;
        _purgePlotlyIn(wrap);
        wrap.innerHTML = '';
        drawImageBatch(d, wrap, (wrap.offsetWidth || 300) - 4, h > 0 ? h : null);
    });
}

// Build a mini input-management block for a single placeholder node.
// vizWrap is the element where tensor viz should be rendered on selection.
function _buildPhInputControls(nodeId, vizWrap) {
    const ctrl = document.createElement("div");
    ctrl.className = "pin-ph-ctrl";

    const listContainer = document.createElement("div");
    listContainer.className = "pin-ph-list";
    ctrl.appendChild(listContainer);

    ctrl.appendChild(buildExprInput(nodeId, "expression…"));

    // retrace button row
    const footer = document.createElement("div");
    footer.className = "pin-ph-footer";
    const retraceBtn = document.createElement("button");
    retraceBtn.className = "pin-ph-retrace-btn";
    retraceBtn.textContent = "↺ retrace";
    retraceBtn.disabled = !hasAnyInput();
    retraceBtn.addEventListener("click", () => { if (currentFn) retrace(currentFn); });
    footer.appendChild(retraceBtn);
    ctrl.appendChild(footer);

    let lastSelId = null;

    function refreshList() {
        listContainer.innerHTML = "";
        const arr = inputSets[nodeId] || [];
        const selId = selectedIds[nodeId];
        retraceBtn.disabled = !hasAnyInput();

        arr.forEach(inp => {
            const row = document.createElement("div");
            row.className = "pin-ph-row" + (inp.id === selId ? " pin-ph-selected" : "");

            const nameSpan = document.createElement("span");
            nameSpan.className = "pin-ph-name" + (inp.valid ? "" : " pin-ph-invalid");
            nameSpan.textContent = inp.label;
            nameSpan.title = inp.label;

            const selBtn = document.createElement("button");
            selBtn.className = "pin-ph-btn" + (inp.id === selId ? " active" : "");
            selBtn.textContent = inp.id === selId ? "●" : "○";
            selBtn.title = "Use this input";
            selBtn.addEventListener("click", () => {
                selectInput(nodeId, inp.id);
                if (vizWrap && inp.type === "expr") _fetchExprViz(nodeId, inp.expr, vizWrap);
            });

            const delBtn = document.createElement("button");
            delBtn.className = "pin-ph-btn";
            delBtn.textContent = "×";
            delBtn.title = "Remove";
            delBtn.addEventListener("click", () => removeInputEntry(nodeId, inp.id));

            row.appendChild(nameSpan);
            row.appendChild(selBtn);
            row.appendChild(delBtn);
            listContainer.appendChild(row);
        });
        // auto-fetch viz when selection changes (e.g. after new entry is auto-selected)
        if (vizWrap && selId && selId !== lastSelId) {
            lastSelId = selId;
            const selEntry = arr.find(e => e.id === selId);
            if (selEntry && selEntry.type === "expr") _fetchExprViz(nodeId, selEntry.expr, vizWrap);
        }
    }

    refreshList();

    if (!_phInputObservers[nodeId]) _phInputObservers[nodeId] = [];
    _phInputObservers[nodeId].push(refreshList);
    ctrl._cleanupObserver = () => {
        _phInputObservers[nodeId] = (_phInputObservers[nodeId] || []).filter(cb => cb !== refreshList);
    };

    return ctrl;
}

// Show the viz section with input management for a placeholder node
function showPlaceholderVizPanel(nodeId, nodeLabel, shape) {
    if (_vizSectionCleanup) { _vizSectionCleanup(); _vizSectionCleanup = null; }

    const section = document.getElementById("viz-section");
    const wrap = document.getElementById("viz-canvas-wrap");
    section.style.display = "block";
    document.getElementById("viz-label").textContent = nodeLabel;
    document.getElementById("viz-shape").textContent =
        shape && shape.length ? `[${shape.join(" × ")}]` : "";

    wrap.innerHTML = "";

    const panel = document.createElement("div");
    panel.className = "ph-panel";

    const vizPreview = document.createElement("div");
    vizPreview.className = "ph-panel-viz";

    const ctrl = _buildPhInputControls(nodeId, vizPreview);
    panel.appendChild(ctrl);
    panel.appendChild(vizPreview);
    wrap.appendChild(panel);

    currentVizData = { data: null, label: nodeLabel, phNodeId: nodeId };

    _vizSectionCleanup = () => {
        if (ctrl._cleanupObserver) ctrl._cleanupObserver();
    };
}

// ─── retrace ──────────────────────────────────────────────────────────────────
function updateNodeOpacity() {
    if (!cy) return;
    const active = hasAnyInput();
    // get_attr nodes are weights — their shape is always known, so never dim them.
    // Folded modules stand for many nodes rather than producing one tensor, so a
    // missing shape on the group says nothing about whether it was traced. Same
    // for a packed loop node: it returns its whole carry as a list, so it never
    // has a single shape, traced or not -- its getitem outputs carry the shapes.
    cy.nodes('[!has_shape][op != "get_attr"]:not([?is_compound]):not([?is_module_group]):not([?is_loop])')
      .toggleClass('no-shape', active);
}

function updateRetraceBtn() {
    const active = hasAnyInput();
    const btn = document.getElementById("retrace-btn");
    if (btn) btn.disabled = !active;

    const topBtn = document.getElementById("topbar-retrace-btn");
    if (topBtn) topBtn.disabled = !active;

    const inputGrp = document.getElementById("input-group");
    if (inputGrp) inputGrp.classList.toggle("no-input", !active);

    const inputPanelBtn = document.getElementById("input-panel-btn");
    if (inputPanelBtn) {
        const isOpen = inputPanelBtn.classList.contains("active");
        inputPanelBtn.textContent = active ? "+ inputs" : "⚠ no input";
        if (isOpen) inputPanelBtn.classList.add("active");
    }

    updateNodeOpacity();
}

// The cards a retrace refreshes: live activations and weights on the current page.
function _retracePins() {
    return (pinPages[currentPinPage] || []).filter(p =>
        !p.phNodeId && p.type !== "bending" && p.type !== "bp" && p.label);
}

async function retrace(fn) {
    if (!hasAnyInput()) return;
    const buttons = ["retrace-btn", "pin-retrace-btn"]
        .map(id => document.getElementById(id)).filter(Boolean);
    buttons.forEach(b => { b.disabled = true; b.textContent = "↺ …"; });
    showLoading(true);
    // Pending from the moment the retrace is asked for, not from when the cards
    // start fetching again: tracing and running the model is the long part, and
    // the dashboard hides the graph's own loading overlay.
    const pending = _retracePins();
    const panel = document.getElementById("viz-section");
    _setPinsBusy(pending, true);
    if (currentVizNode) _setBusy(panel, true);
    _computeBegin();
    try {
        const r = await fetch(`/api/retrace/${fn}/?${_graphQuery()}`, { method: "POST", body: buildActivationForm() });
        const data = await r.json();
        // a trace failure carries the failing line and the last node traced —
        // show it, rather than only echoing the exception message
        if (data.error) {
            showTraceError("Retrace failed", data);
            const e = new Error(data.error);
            e.shown = true;
            throw e;
        }
        _invalidateNodeIndex();
        renderGraph(data);
        showLoading(false);
        _fetchNodeIndex(fn).then(fetched => {
            if (fetched) renderActivationList(currentGraphData);
        });
        showToast("info", "Graph retraced — shapes updated");
        // the cards stay pending until their new values are in
        const refreshes = [_refreshActivationPins(fn), _refreshWeightPins(fn)];
        if (currentVizNode) {
            if (currentVizNode.op === "get_attr")
                refreshes.push(fetchWeight(fn, currentVizNode.id));
            else if (currentVizActId && hasAnyInput())
                refreshes.push(fetchActivation(fn, currentVizActId));
        }
        await Promise.allSettled(refreshes);
    } catch (err) {
        showLoading(false);
        // the panel (with the failing line) is already up for a server-side
        // trace error; this covers transport failures
        if (!err.shown) showToast("error", "Retrace failed: " + err.message, err.traceback);
    } finally {
        _setPinsBusy(pending, false);
        _setBusy(panel, false);
        _computeEnd();
        buttons.forEach(b => {
            b.textContent = "↺ retrace";
            b.disabled = b.id === "retrace-btn" ? !hasAnyInput() : false;
        });
    }
}

// ─── trace / run error panel ──────────────────────────────────────────────────
// The panel itself lives in errors.js (TBTraceError) so the editor and play mode
// report a failure identically; this binds it to the editor's toast + error log.
function showTraceError(title, data) {
    TBTraceError.show(title, data, {
        onCopyPath: () => showToast("info", "Path copied"),
        log: _addToErrorLog,
    });
}

// ─── node context menu ────────────────────────────────────────────────────────
// Unused by the live UI (see `_showNodeContextMenu` near the bottom of this
// file, wired to the canvas's `cxttap` handler) but left as found.
let _nodeCtxMenu = null;

function _closeNodeCtxMenu() {
    if (_nodeCtxMenu) { _nodeCtxMenu.remove(); _nodeCtxMenu = null; }
}

function _showNodeCtxMenu(cyNode) {
    _closeNodeCtxMenu();
    const d = cyNode.data();
    const pos = cyNode.renderedPosition();
    const cyRect = document.getElementById("cy").getBoundingClientRect();
    const x = cyRect.left + pos.x;
    const y = cyRect.top + pos.y;

    const menu = document.createElement("div");
    menu.className = "node-ctx-menu";

    const tagBtn = document.createElement("button");
    tagBtn.className = "node-ctx-btn";
    tagBtn.textContent = "# add alias";
    tagBtn.addEventListener("click", () => {
        _closeNodeCtxMenu();
        const addBtn = document.querySelector("#detail-tags-section .detail-tag-add-btn");
        if (addBtn) addBtn.click();
    });

    const pinBtn = document.createElement("button");
    pinBtn.className = "node-ctx-btn";
    pinBtn.textContent = "⊕ pin";
    pinBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const nodeData = (currentGraphData && currentGraphData.nodes.find(n => n.id === d.id)) || d;
        const savedRect = pinBtn.getBoundingClientRect();
        _closeNodeCtxMenu();
        _showActModalPagePicker(e, nodeData, { getBoundingClientRect: () => savedRect });
    });

    const expandBtn = document.createElement("button");
    expandBtn.className = "node-ctx-btn";
    expandBtn.textContent = "⊞ expand";
    expandBtn.addEventListener("click", () => {
        _closeNodeCtxMenu();
        openExpandModal();
    });

    // Bend actions first (yellow group)
    if (["call_function", "call_module", "call_method", "get_attr"].includes(d.op)) {
        const bendBtn = document.createElement("button");
        bendBtn.className = "node-ctx-btn ctx-bend";
        bendBtn.textContent = "⚡ bend";
        bendBtn.addEventListener("click", () => {
            _closeNodeCtxMenu();
            const nodeData = (currentGraphData && currentGraphData.nodes.find(n => n.id === d.id)) || d;
            _openBendDialog(nodeData);
        });
        menu.appendChild(bendBtn);

        if (_bendingBindings.length > 0) {
            const applyBtn = document.createElement("button");
            applyBtn.className = "node-ctx-btn ctx-bend";
            applyBtn.textContent = "⚡ apply existing…";
            applyBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                _closeNodeCtxMenu();
                _showApplyExistingMenu(e, d);
            });
            menu.appendChild(applyBtn);
        }
        // Separator after bend group
        const sep = document.createElement("div");
        sep.className = "node-ctx-bend-sep";
        menu.appendChild(sep);
    }

    menu.appendChild(expandBtn);
    menu.appendChild(pinBtn);

    if (["call_function", "call_module", "call_method", "get_attr"].includes(d.op)) {
        const srcBtn = document.createElement("button");
        srcBtn.className = "node-ctx-btn";
        srcBtn.textContent = "{ } source";
        srcBtn.addEventListener("click", () => {
            _closeNodeCtxMenu();
            openSourceModal(currentFn, d.id);
        });
        menu.appendChild(srcBtn);
    }

    menu.appendChild(tagBtn);
    document.body.appendChild(menu);
    _nodeCtxMenu = menu;

    // clamp to viewport after append so dimensions are known
    const mw = menu.offsetWidth || 140;
    const mh = menu.offsetHeight || 64;
    const left = Math.min(x + 10, window.innerWidth  - mw - 8);
    const top  = Math.max(8, Math.min(y - mh / 2, window.innerHeight - mh - 8));
    menu.style.left = left + "px";
    menu.style.top  = top  + "px";

    const dismiss = (ev) => {
        if (!menu.contains(ev.target)) {
            _closeNodeCtxMenu();
            document.removeEventListener("mousedown", dismiss);
        }
    };
    setTimeout(() => document.addEventListener("mousedown", dismiss), 0);
}

// ─── pin dashboard ────────────────────────────────────────────────────────────
let pinPages = [[]];          // array of pages, each page is an array of pin objects
let currentPinPage = 0;
let _pinCtxMenu = null;
let _lastCardDragEndTime = 0;
let _pinTileZoom = 1.0;       // zoom factor for grid_4d / image_batch tiles

function _closePinCtxMenu() {
    if (_pinCtxMenu) { _pinCtxMenu.remove(); _pinCtxMenu = null; }
}

let _pinCtxTab = "act";   // persists tab selection across opens

function _openPinCtxMenu(clientX, clientY) {
    _closePinCtxMenu();
    const nodes = _actModalNodes.length
        ? _actModalNodes
        : (currentGraphData ? currentGraphData.nodes.filter(n => !n.is_compound) : []);

    const menu = document.createElement("div");
    menu.className = "pin-ctx-menu";
    _pinCtxMenu = menu;

    // Clamp to viewport
    const menuW = 310, menuH = 440;
    menu.style.left = Math.min(clientX, window.innerWidth  - menuW - 10) + "px";
    menu.style.top  = Math.min(clientY, window.innerHeight - menuH - 10) + "px";

    // ── tab bar ───────────────────────────────────────────────────────────────
    const TABS = [
        { id: "act",   icon: "◈", label: "act"   },
        { id: "alias", icon: "#", label: "alias"  },
        { id: "bend",  icon: "~", label: "bend"   },
        { id: "macro", icon: "⊙", label: "macro"  },
    ];
    const tabBar = document.createElement("div");
    tabBar.className = "pin-ctx-tabbar";
    const tabBtns = {};
    TABS.forEach(t => {
        const btn = document.createElement("button");
        btn.className = "pin-ctx-tab" + (t.id === _pinCtxTab ? " active" : "");
        btn.innerHTML = `<span class="pin-ctx-tab-icon">${t.icon}</span><span class="pin-ctx-tab-label">${t.label}</span>`;
        btn.addEventListener("click", () => {
            _pinCtxTab = t.id;
            Object.values(tabBtns).forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            searchInp.value = "";
            renderList("");
            searchInp.focus();
        });
        tabBtns[t.id] = btn;
        tabBar.appendChild(btn);
    });
    menu.appendChild(tabBar);

    // ── search ────────────────────────────────────────────────────────────────
    const searchRow = document.createElement("div");
    searchRow.className = "pin-ctx-search-row";
    const searchInp = document.createElement("input");
    searchInp.className = "pin-ctx-search";
    searchInp.spellcheck = false;
    searchRow.appendChild(searchInp);
    menu.appendChild(searchRow);

    // ── scrollable list ───────────────────────────────────────────────────────
    const list = document.createElement("div");
    list.className = "pin-ctx-list";
    menu.appendChild(list);

    let _hi = -1;

    function _sep(txt) {
        const s = document.createElement("div");
        s.className = "pin-ctx-sep";
        s.textContent = txt;
        return s;
    }

    function _makeNodeRow(n) {
        const row = document.createElement("div");
        row.className = "pin-ctx-row";
        const dot = document.createElement("span");
        dot.className = "pin-ctx-dot";
        dot.style.background = OP_COLORS[n.op] || DEFAULT_COLOR;
        const name = document.createElement("span");
        name.className = "pin-ctx-name";
        name.textContent = n.label;
        const shp = document.createElement("span");
        shp.className = "pin-ctx-shape";
        shp.textContent = n.shape && n.shape.length ? `[${n.shape.join("×")}]` : "";
        row.append(dot, name, shp);
        row.addEventListener("mouseenter", () => {
            list.querySelectorAll(".pin-ctx-row").forEach(r => r.classList.remove("hi"));
            row.classList.add("hi");
            _hi = [...list.children].indexOf(row);
        });
        row.addEventListener("click", () => { _pinNodeToPage(n, currentPinPage); _closePinCtxMenu(); });
        return row;
    }

    function _makeAliasGroupRow(aliasName, memberNodes) {
        const row = document.createElement("div");
        row.className = "pin-ctx-row pin-ctx-row-alias";
        const badge = document.createElement("span");
        badge.className = "pin-ctx-dot pin-ctx-alias-badge";
        badge.textContent = "#";
        const name = document.createElement("span");
        name.className = "pin-ctx-name";
        name.textContent = aliasName;
        const cnt = document.createElement("span");
        cnt.className = "pin-ctx-shape";
        cnt.textContent = memberNodes.length + " nodes";
        row.append(badge, name, cnt);
        row.addEventListener("mouseenter", () => {
            list.querySelectorAll(".pin-ctx-row").forEach(r => r.classList.remove("hi"));
            row.classList.add("hi");
            _hi = [...list.children].indexOf(row);
        });
        row.addEventListener("click", () => {
            memberNodes.forEach(n => _pinNodeToPage(n, currentPinPage));
            _closePinCtxMenu();
        });
        return row;
    }

    function _makeBendingRow(b) {
        const row = document.createElement("div");
        row.className = "pin-ctx-row pin-ctx-row-bend";
        const badge = document.createElement("span");
        badge.className = "pin-ctx-dot pin-ctx-bend-badge";
        badge.textContent = "~";
        const name = document.createElement("span");
        name.className = "pin-ctx-name";
        name.textContent = b.callback_type || b.id;
        name.appendChild(_bendInfoIcon(b));
        const info = document.createElement("span");
        info.className = "pin-ctx-shape";
        info.textContent = (b.nodes || [b.node]).slice(0, 2).join(", ");
        row.append(badge, name, info);
        row.addEventListener("mouseenter", () => {
            list.querySelectorAll(".pin-ctx-row").forEach(r => r.classList.remove("hi"));
            row.classList.add("hi");
            _hi = [...list.children].indexOf(row);
        });
        row.addEventListener("click", () => { addBendingPin(b.id); _closePinCtxMenu(); });
        return row;
    }

    function _makeBpRow(bp) {
        const row = document.createElement("div");
        row.className = "pin-ctx-row pin-ctx-row-bp";
        const badge = document.createElement("span");
        badge.className = "pin-ctx-dot pin-ctx-macro-badge";
        badge.textContent = "⊙";
        const name = document.createElement("span");
        name.className = "pin-ctx-name";
        name.textContent = bp.name;
        const val = document.createElement("span");
        val.className = "pin-ctx-shape";
        val.textContent = _bpFmt(bp, bp.value);
        row.append(badge, name, val);
        row.addEventListener("mouseenter", () => {
            list.querySelectorAll(".pin-ctx-row").forEach(r => r.classList.remove("hi"));
            row.classList.add("hi");
            _hi = [...list.children].indexOf(row);
        });
        row.addEventListener("click", () => { addBendingParamPin(bp.name); _closePinCtxMenu(); });
        return row;
    }

    function renderList(q) {
        list.innerHTML = "";
        _hi = -1;
        const lq = q.toLowerCase();

        if (_pinCtxTab === "act") {
            searchInp.placeholder = "search activations…";
            let filtered;
            if (q.startsWith("#")) {
                const pattern = q.slice(1);
                let re = null;
                try { re = pattern ? new RegExp(pattern, "i") : null; } catch (_) {}
                const aliasSet = new Set();
                if (currentGraphData && currentGraphData.aliases)
                    Object.entries(currentGraphData.aliases).forEach(([n, members]) => {
                        if (!re || re.test(n)) members.forEach(id => aliasSet.add(id));
                    });
                nodes.forEach(n => { if (_getNodeTags(n.id || n.label).some(t => !re || re.test(t))) aliasSet.add(n.id || n.label); });
                filtered = nodes.filter(n => aliasSet.has(n.id || n.label)).slice(0, 80);
            } else {
                filtered = nodes.filter(n =>
                    !q || n.label.toLowerCase().includes(lq) || (n.target || "").toLowerCase().includes(lq)
                ).slice(0, 80);
            }
            const pinnedLabels = new Set((pinPages[currentPinPage] || []).map(p => p.label));
            const favs = filtered.filter(n => _isFav(n.label) && !pinnedLabels.has(n.label));
            if (favs.length) {
                list.appendChild(_sep("★ favourites"));
                favs.forEach(n => list.appendChild(_makeNodeRow(n)));
                list.appendChild(_sep("all"));
            }
            filtered.forEach(n => list.appendChild(_makeNodeRow(n)));

        } else if (_pinCtxTab === "alias") {
            searchInp.placeholder = "search aliases…";
            const nodeById = Object.fromEntries(nodes.map(n => [n.id || n.label, n]));
            // backend aliases
            const aliases = (currentGraphData && currentGraphData.aliases) || {};
            Object.entries(aliases).forEach(([aliasName, memberIds]) => {
                if (q && !aliasName.toLowerCase().includes(lq)) return;
                const memberNodes = memberIds.map(id => nodeById[id]).filter(Boolean);
                if (memberNodes.length) list.appendChild(_makeAliasGroupRow(aliasName, memberNodes));
            });
            // user tags (each unique tag as a group)
            const tagGroups = {};
            Object.entries(_nodeTags || {}).forEach(([nodeId, tags]) => {
                (tags || []).forEach(t => {
                    if (q && !t.toLowerCase().includes(lq)) return;
                    if (!tagGroups[t]) tagGroups[t] = [];
                    const n = nodeById[nodeId];
                    if (n) tagGroups[t].push(n);
                });
            });
            if (Object.keys(tagGroups).length) {
                if (Object.keys(aliases).length) list.appendChild(_sep("tags"));
                Object.entries(tagGroups).forEach(([tag, ns]) => list.appendChild(_makeAliasGroupRow(tag, ns)));
            }
            if (!list.children.length) {
                const empty = document.createElement("div");
                empty.className = "pin-ctx-sep";
                empty.textContent = "no aliases or tags defined";
                list.appendChild(empty);
            }

        } else if (_pinCtxTab === "bend") {
            searchInp.placeholder = "search bendings…";
            const filtered = _bendingBindings.filter(b =>
                !q || (b.callback_type || "").toLowerCase().includes(lq) ||
                      (b.nodes || [b.node]).some(n => (n || "").toLowerCase().includes(lq))
            );
            if (!filtered.length) {
                const empty = document.createElement("div");
                empty.className = "pin-ctx-sep";
                empty.textContent = "no active bendings";
                list.appendChild(empty);
            }
            filtered.forEach(b => list.appendChild(_makeBendingRow(b)));

        } else if (_pinCtxTab === "macro") {
            searchInp.placeholder = "search macros…";
            const filtered = _bendingParams.filter(bp => !q || bp.name.toLowerCase().includes(lq));
            if (!filtered.length) {
                const empty = document.createElement("div");
                empty.className = "pin-ctx-sep";
                empty.textContent = "no macros defined";
                list.appendChild(empty);
            }
            filtered.forEach(bp => list.appendChild(_makeBpRow(bp)));
        }
    }

    renderList("");

    searchInp.addEventListener("input", () => {
        renderList(searchInp.value.trim());
        if (_pinCtxTab === "act") _updateAliasSuggest(searchInp);
    });
    searchInp.addEventListener("blur", () => {
        setTimeout(() => { const l = document.getElementById("alias-suggest-list"); if (l) l.style.display = "none"; }, 120);
    });

    searchInp.addEventListener("keydown", (e) => {
        if (_pinCtxTab === "act" && _aliasSuggestKeydown(e, searchInp)) return;
        const rows = [...list.querySelectorAll(".pin-ctx-row")];
        if (e.key === "ArrowDown") {
            e.preventDefault();
            _hi = Math.min(_hi + 1, rows.length - 1);
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            _hi = Math.max(_hi - 1, 0);
        } else if (e.key === "Enter") {
            e.preventDefault();
            const target = _hi >= 0 ? rows[_hi] : (rows.length === 1 ? rows[0] : null);
            if (target) target.click();
            return;
        } else if (e.key === "Escape") {
            e.preventDefault();
            _closePinCtxMenu();
            return;
        }
        rows.forEach((r, i) => r.classList.toggle("hi", i === _hi));
        if (rows[_hi]) rows[_hi].scrollIntoView({ block: "nearest" });
    });

    // dismiss on click outside
    setTimeout(() => {
        const outside = (e) => {
            if (!menu.contains(e.target)) {
                _closePinCtxMenu();
                document.removeEventListener("mousedown", outside);
            }
        };
        document.addEventListener("mousedown", outside);
    }, 0);

    document.getElementById("pin-dashboard").appendChild(menu);
    requestAnimationFrame(() => searchInp.focus());
}

function _cardGridPos(pageLocalIdx) {
    const items = document.getElementById("pin-items");
    const pw = (items ? items.offsetWidth : window.innerWidth) - 40;
    const cardW = 340, cardH = 300, gap = 16, pad = 20;
    const cols = Math.max(1, Math.floor(pw / (cardW + gap)));
    const col = pageLocalIdx % cols;
    const row = Math.floor(pageLocalIdx / cols);
    return { x: pad + col * (cardW + gap), y: pad + row * (cardH + gap) };
}

function renderPinTabs() {
    const bar = document.getElementById("pin-page-tabs");
    if (!bar) return;
    bar.innerHTML = "";
    _persistPinPages();
    pinPages.forEach((page, i) => {
        const tab = document.createElement("div");
        tab.className = "pin-tab" + (i === currentPinPage ? " active" : "");
        tab.dataset.pageIdx = i;
        tab.textContent = `${i + 1} · ${page.length}`;
        tab.addEventListener("click", () => switchPinPage(i));
        bar.appendChild(tab);
    });
    const addBtn = document.createElement("div");
    addBtn.className = "pin-tab-add";
    addBtn.textContent = "+";
    addBtn.title = "New page";
    addBtn.addEventListener("click", addPinPage);
    bar.appendChild(addBtn);
}

// Update the sentinel div so #pin-items scrolls to cover all cards.
function _updatePinScrollArea() {
    const container = document.getElementById("pin-items");
    if (!container) return;
    let maxR = 0, maxB = 0;
    container.querySelectorAll(".pin-card, .pin-card-bending").forEach(card => {
        maxR = Math.max(maxR, card.offsetLeft + card.offsetWidth);
        maxB = Math.max(maxB, card.offsetTop  + card.offsetHeight);
    });
    let s = container.querySelector(".pin-scroll-sentinel");
    if (!s) {
        s = document.createElement("div");
        s.className = "pin-scroll-sentinel";
        s.style.cssText = "position:absolute;pointer-events:none;width:1px;height:1px;";
        container.appendChild(s);
    }
    s.style.left = (maxR + 24) + "px";
    s.style.top  = (maxB + 24) + "px";
}

function _watchCardSize(card, pin) {
    let settled = false;
    // Skip the first observation (initial layout), then save on subsequent resizes
    const ro = new ResizeObserver(() => {
        if (!settled) { settled = true; return; }
        clearTimeout(card._szTimer);
        card._szTimer = setTimeout(() => {
            const h = card.offsetHeight;
            if (h > 0) pin.h = h;
            _updatePinScrollArea();
        }, 120);
    });
    ro.observe(card);
    card._sizeObserver = ro;
}

function renderCurrentPage() {
    const container = document.getElementById("pin-items");
    if (!container) return;
    const page = pinPages[currentPinPage] || [];
    const pageIds = new Set(page.map(p => p.id));

    // Remove cards that are no longer in the current page
    container.querySelectorAll(".pin-card, .pin-card-bending").forEach(card => {
        const cardPinId = card.id.replace("pin-card-", "");
        if (!pageIds.has(cardPinId)) {
            card.querySelectorAll(".pin-ph-ctrl").forEach(ctrl => {
                if (ctrl._cleanupObserver) ctrl._cleanupObserver();
            });
            if (card._sizeObserver) card._sizeObserver.disconnect();
            card.remove();
        }
    });

    // Add cards that don't exist yet; reposition existing auto-placed cards in case
    // they were laid out while the dashboard was hidden (offsetWidth=0).
    page.forEach((pin, idx) => {
        const existing = document.getElementById(`pin-card-${pin.id}`);
        if (existing) {
            if (pin.x == null) {
                const pos = _cardGridPos(idx);
                existing.style.left = pos.x + "px";
                existing.style.top  = pos.y + "px";
            }
            return;
        }
        const card = pin.type === "bending" ? _makeBendingPinCard(pin)
                   : pin.type === "bp"      ? _makeBpPinCard(pin)
                   : _makePinCard(pin);
        const pos = _cardGridPos(idx);
        card.style.left = (pin.x != null ? pin.x : pos.x) + "px";
        card.style.top  = (pin.y != null ? pin.y : pos.y) + "px";
        const isSelfSizing = pin.data && (pin.data.kind === "grid_4d" || pin.data.kind === "image_batch"
                                          || pin.data.view === "image" || pin.data.view === "channel_grid");
        if (pin.h != null && !isSelfSizing) card.style.height = pin.h + "px";
        container.appendChild(card);
        _watchCardSize(card, pin);
    });
    _updatePinScrollArea();
}

function switchPinPage(idx) {
    if (idx < 0 || idx >= pinPages.length) return;
    currentPinPage = idx;
    renderPinTabs();
    renderCurrentPage();
    // refreshes are scoped to the visible page, so bring this one up to date
    if (currentFn) {
        _refreshWeightPins(currentFn);
        if (hasAnyInput()) _refreshActivationPins(currentFn);
    }
}

function addPinPage() {
    // Ask whether to copy the current page or start fresh
    const dashboard = document.getElementById("pin-dashboard");
    const addBtn = dashboard && dashboard.querySelector(".pin-tab-add");
    const anchor = addBtn || dashboard;
    if (!anchor) { _doAddPinPage(false); return; }

    const pop = document.createElement("div");
    pop.className = "pin-new-page-pop";
    const rect = anchor.getBoundingClientRect();
    pop.style.cssText = `position:fixed;left:${rect.left}px;top:${rect.bottom + 4}px;z-index:9000`;

    function makeOpt(label, icon, fn) {
        const btn = document.createElement("button");
        btn.className = "pin-new-page-opt";
        btn.innerHTML = `<span>${icon}</span> ${label}`;
        btn.addEventListener("click", () => { pop.remove(); fn(); });
        return btn;
    }
    pop.appendChild(makeOpt("Empty page", "☐", () => _doAddPinPage(false)));
    pop.appendChild(makeOpt("Copy current page", "⧉", () => _doAddPinPage(true)));
    document.body.appendChild(pop);
    const dismiss = (e) => { if (!pop.contains(e.target) && e.target !== addBtn) { pop.remove(); document.removeEventListener("mousedown", dismiss); } };
    setTimeout(() => document.addEventListener("mousedown", dismiss), 0);
}

function _doAddPinPage(copyCurrentPage) {
    const newPage = copyCurrentPage
        ? pinPages[currentPinPage].map(p => ({ ...p, id: genId(), data: null, originalData: null }))
        : [];
    pinPages.push(newPage);
    currentPinPage = pinPages.length - 1;
    renderPinTabs();
    renderCurrentPage();
    if (copyCurrentPage) _refetchNullPins();
}

function movePinToPage(pinId, targetPage) {
    const srcPage = pinPages[currentPinPage];
    const idx = srcPage.findIndex(p => p.id === pinId);
    if (idx < 0) return;
    const [pin] = srcPage.splice(idx, 1);
    if (!pinPages[targetPage]) pinPages[targetPage] = [];
    pinPages[targetPage].push(pin);
    renderPinTabs();
    renderCurrentPage();
    _updatePinCount();
}

function _makePinCard(pin) {
    // a card saved when a recall only froze the view: it follows the graph now
    if (pin.snapshot) delete pin.snapshot;
    const card = document.createElement("div");
    card.className = "pin-card" + (pin.phNodeId ? " pin-card-ph" : "");
    card.id = `pin-card-${pin.id}`;
    if (pin.collapsed) card.classList.add("pin-card--collapsed");

    // header
    const hdr = document.createElement("div");
    hdr.className = "pin-card-header";

    // alias tags (backend aliases + user tags for this node) — they ride in the
    // title rather than a row of their own: a node is usually recognised by the
    // alias it was marked with, and a collapsed card hides everything but the
    // header, which is exactly when the name alone says least.
    const nodeObj = { id: pin.label, label: pin.label };
    const aliasNames = _nodeAliases(nodeObj, currentGraphData);

    const lbl = document.createElement("span");
    lbl.className = "pin-card-label";
    lbl.textContent = pin.label;
    lbl.title = aliasNames.length
        ? `${pin.label}\naliases: ${aliasNames.map(a => "#" + a).join(", ")}`
        : pin.label;

    const aliasRow = document.createElement("span");
    aliasRow.className = "pin-card-aliases pin-card-aliases-inline";
    aliasNames.forEach(a => {
        const chip = document.createElement("span");
        chip.className = "pin-card-alias-chip";
        chip.textContent = "#" + a;
        chip.title = `Alias '${a}'`;
        aliasRow.appendChild(chip);
    });

    const shp = document.createElement("span");
    shp.className = "pin-card-shape";
    shp.textContent = pin.shape && pin.shape.length ? `[${pin.shape.join(" × ")}]` : "";

    const closeBtn = document.createElement("button");
    closeBtn.className = "pin-card-close";
    closeBtn.textContent = "×";
    closeBtn.title = "Remove";

    const showInGraphBtn = document.createElement("button");
    showInGraphBtn.className = "pin-card-show-btn";
    showInGraphBtn.textContent = "⊙";
    showInGraphBtn.title = "Show in graph";
    showInGraphBtn.addEventListener("click", () => {
        togglePinDashboard(false);
        const cyNode = cy.$id(pin.label);
        if (cyNode && cyNode.length) {
            cy.animate({ center: { eles: cyNode }, zoom: Math.max(cy.zoom(), 1.5) }, { duration: 300 });
            onNodeClick(cyNode);
        }
    });

    const pinSaveBtn = document.createElement("button");
    pinSaveBtn.className = "pin-card-show-btn";
    pinSaveBtn.textContent = "⬇";
    pinSaveBtn.title = "Save activation";
    pinSaveBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        _showActSaveMenu(e, pin.label);
    });

    // A B×C×T activation comes up as channel plots — the shape it has, but not
    // necessarily the thing it is. When it is long enough to be audible, the
    // switch to the audio view (and its per-channel play strips) is offered
    // here, rather than the view dropdown being the only way to find it.
    const listenBtn = document.createElement("button");
    listenBtn.className = "pin-card-show-btn pin-card-listen-btn";
    listenBtn.textContent = "♪";
    listenBtn.title = "Listen — show this activation as audio";
    listenBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        _setPinView(pin, "audio");
    });
    _syncPinListenBtn(pin, listenBtn);

    const bendable = ["call_function", "call_module", "call_method", "get_attr"];
    if (pin.nodeOp && bendable.includes(pin.nodeOp)) {
        const bendBtn = document.createElement("button");
        bendBtn.className = "pin-card-bend-btn";
        bendBtn.textContent = "⚡";
        bendBtn.title = "Bend this activation";
        bendBtn.addEventListener("click", () => {
            const nodeData = (currentGraphData && currentGraphData.nodes.find(n => n.id === pin.label))
                || { id: pin.label, op: pin.nodeOp };
            _openBendDialog(nodeData);
        });
        hdr.appendChild(lbl);
        if (aliasNames.length) hdr.appendChild(aliasRow);
        hdr.appendChild(shp); hdr.appendChild(showInGraphBtn); hdr.appendChild(listenBtn); hdr.appendChild(pinSaveBtn); hdr.appendChild(bendBtn); hdr.appendChild(closeBtn);
    } else {
        hdr.appendChild(lbl);
        if (aliasNames.length) hdr.appendChild(aliasRow);
        hdr.appendChild(shp); hdr.appendChild(showInGraphBtn); hdr.appendChild(listenBtn); hdr.appendChild(pinSaveBtn); hdr.appendChild(closeBtn);
    }
    card.appendChild(hdr);
    // an activation: it can be saved, and a snapshot recalled into the graph
    // at it -- on a row of its own, above the view properties
    if (!pin.phNodeId && pin.nodeOp !== "get_attr" && pin.nodeOp !== "placeholder") {
        const snapRow = document.createElement("div");
        snapRow.className = "pin-snap-row";
        snapRow.appendChild(_snapBar(_snapCtx(currentFn, pin.label)));
        card.appendChild(snapRow);
        if (_recalledOn(currentFn, pin.label)) card.classList.add("pin-frozen");
    }

    // drag by header — pointer-events disabled during drag so elementFromPoint
    // can detect page tabs underneath for drag-to-page
    let dragging = false, ox = 0, oy = 0;
    hdr.addEventListener("mousedown", (e) => {
        if (e.target.tagName === "BUTTON") return;
        if (e.detail >= 2) {
            pin.collapsed = card.classList.toggle("pin-card--collapsed");
            e.preventDefault();
            return;
        }
        const r = card.getBoundingClientRect();
        ox = e.clientX - r.left; oy = e.clientY - r.top;
        dragging = true;
        card.style.pointerEvents = "none";
        card.style.zIndex = "9999";
        e.preventDefault();
    });
    document.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const cr = document.getElementById("pin-items").getBoundingClientRect();
        card.style.left = (e.clientX - ox - cr.left) + "px";
        card.style.top  = Math.max(0, e.clientY - oy - cr.top) + "px";
        const under = document.elementFromPoint(e.clientX, e.clientY);
        document.querySelectorAll(".pin-tab").forEach(t => t.classList.remove("drop-target"));
        if (under && under.classList.contains("pin-tab")) {
            const tp = parseInt(under.dataset.pageIdx, 10);
            if (tp !== currentPinPage) under.classList.add("drop-target");
        }
    });
    document.addEventListener("mouseup", (e) => {
        if (!dragging) return;
        dragging = false;
        _lastCardDragEndTime = Date.now();
        card.style.pointerEvents = "";
        card.style.zIndex = "";
        pin.x = parseFloat(card.style.left);
        pin.y = parseFloat(card.style.top);
        _updatePinScrollArea();
        document.querySelectorAll(".pin-tab").forEach(t => t.classList.remove("drop-target"));
        const under = document.elementFromPoint(e.clientX, e.clientY);
        if (under && under.classList.contains("pin-tab")) {
            const tp = parseInt(under.dataset.pageIdx, 10);
            if (!isNaN(tp) && tp !== currentPinPage) {
                movePinToPage(pin.id, tp);
                return;
            }
        }
    });

    const vizWrap = document.createElement("div");
    vizWrap.className = "pin-viz-wrap";

    const ro = new ResizeObserver(() => {
        clearTimeout(card._rzTimer);
        card._rzTimer = setTimeout(() => _relayoutPlotlyIn(vizWrap), 150);
    });
    ro.observe(card);

    if (pin.phNodeId) {
        // placeholder pin: input management controls + live Plotly viz
        const ctrl = _buildPhInputControls(pin.phNodeId, vizWrap);
        card.appendChild(ctrl);
        card.appendChild(vizWrap);

        closeBtn.addEventListener("click", () => {
            ro.disconnect();
            _purgePlotlyIn(vizWrap);
            if (ctrl._cleanupObserver) ctrl._cleanupObserver();
            removePin(pin.id);
        });
    } else {
        // regular tensor snapshot pin
        card.appendChild(vizWrap);
        _renderPinViewPicker(pin, card);
        requestAnimationFrame(() => _renderIntoPlotly(
            pin.data, vizWrap,
            Object.assign({ detailed: true, imgZoom: _pinTileZoom, stateKey: "pin/" + pin.id,
                            audioNode: pin.label },
                          pin.originalData ? { originalData: pin.originalData } : {})
        ));

        closeBtn.addEventListener("click", () => {
            ro.disconnect();
            _purgePlotlyIn(vizWrap);
            removePin(pin.id);
        });
    }

    _makePinBendingSection(pin, card);
    return card;
}

function _makeBendingPinCard(pin) {
    const b = _bendingBindings.find(x => x.id === pin.bindingId) || pin.bindingSnapshot || {};
    const card = document.createElement("div");
    card.className = "pin-card pin-card-bending";
    card.id = `pin-card-${pin.id}`;
    card.dataset.bid = pin.bindingId;
    if (pin.collapsed) card.classList.add("pin-card--collapsed");

    // header
    const hdr = document.createElement("div");
    hdr.className = "pin-card-header";

    const lbl = document.createElement("span");
    lbl.className = "pin-card-label";
    lbl.innerHTML = `<span class="pin-cb-icon">⚡</span> ${b.callback_type || pin.label}`;
    lbl.title = b.id || "";
    lbl.appendChild(_bendInfoIcon(b));

    const idSpan = document.createElement("span");
    idSpan.className = "pin-card-shape pin-cb-id";
    idSpan.textContent = b.id ? `[${b.id}]` : "";

    const closeBtn = document.createElement("button");
    closeBtn.className = "pin-card-close";
    closeBtn.textContent = "×";
    closeBtn.title = "Remove";
    closeBtn.addEventListener("click", () => removePin(pin.id));

    hdr.appendChild(lbl); hdr.appendChild(idSpan); hdr.appendChild(closeBtn);
    card.appendChild(hdr);

    // drag support
    let dragging = false, ox = 0, oy = 0;
    hdr.addEventListener("mousedown", (e) => {
        if (e.target.tagName === "BUTTON") return;
        if (e.detail >= 2) {
            pin.collapsed = card.classList.toggle("pin-card--collapsed");
            e.preventDefault();
            return;
        }
        const r = card.getBoundingClientRect();
        ox = e.clientX - r.left; oy = e.clientY - r.top;
        dragging = true;
        card.style.pointerEvents = "none";
        card.style.zIndex = "9999";
        e.preventDefault();
    });
    document.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const cr = document.getElementById("pin-items").getBoundingClientRect();
        card.style.left = (e.clientX - ox - cr.left) + "px";
        card.style.top  = Math.max(0, e.clientY - oy - cr.top) + "px";
        const under = document.elementFromPoint(e.clientX, e.clientY);
        document.querySelectorAll(".pin-tab").forEach(t => t.classList.remove("drop-target"));
        if (under && under.classList.contains("pin-tab")) {
            const tp = parseInt(under.dataset.pageIdx, 10);
            if (tp !== currentPinPage) under.classList.add("drop-target");
        }
    });
    document.addEventListener("mouseup", (e) => {
        if (!dragging) return;
        dragging = false;
        _lastCardDragEndTime = Date.now();
        card.style.pointerEvents = "";
        card.style.zIndex = "";
        pin.x = parseFloat(card.style.left);
        pin.y = parseFloat(card.style.top);
        document.querySelectorAll(".pin-tab").forEach(t => t.classList.remove("drop-target"));
        const under = document.elementFromPoint(e.clientX, e.clientY);
        if (under && under.classList.contains("pin-tab")) {
            const tp = parseInt(under.dataset.pageIdx, 10);
            if (!isNaN(tp) && tp !== currentPinPage) { movePinToPage(pin.id, tp); }
        }
    });

    // body
    const body = document.createElement("div");
    body.className = "pin-cb-body";

    // target info
    const targetRow = document.createElement("div");
    targetRow.className = "pin-cb-target";
    const nodes = b.nodes || (b.node ? [b.node] : []);
    const nodeList = nodes.length ? nodes.join(", ") : (b.node || "—");
    targetRow.innerHTML =
        `<span class="pin-cb-target-label">node</span><span class="pin-cb-target-val">${nodeList}</span>` +
        `<span class="pin-cb-target-label">fn</span><span class="pin-cb-target-val">${b.fn || "—"}</span>`;
    body.appendChild(targetRow);

    // param controls
    if (b.descriptor && Object.keys(b.descriptor.params || {}).length) {
        const paramWrap = document.createElement("div");
        paramWrap.className = "pin-cb-params";
        _buildParamControls(paramWrap, b, "pbc");
        body.appendChild(paramWrap);
    } else {
        const noParams = document.createElement("div");
        noParams.className = "pin-cb-no-params";
        noParams.textContent = "no parameters";
        body.appendChild(noParams);
    }

    card.appendChild(body);
    return card;
}

function _refreshBendingPinCards() {
    const page = pinPages[currentPinPage] || [];
    page.forEach(pin => {
        if (pin.type !== "bending") return;
        // Update the snapshot so sliders don't reset
        const live = _bendingBindings.find(x => x.id === pin.bindingId);
        if (live) pin.bindingSnapshot = live;
        const card = document.getElementById(`pin-card-${pin.id}`);
        if (!card) return;
        // Update timing notes in-place
        const note = _bendingLastMs != null ? `last update: ${_bendingLastMs}ms` : "";
        card.querySelectorAll(".pbc-timing-note").forEach(el => { el.textContent = note; });
        // If binding was removed, mark card stale
        if (!live) {
            const lbl = card.querySelector(".pin-card-label");
            if (lbl && !lbl.dataset.stale) {
                lbl.dataset.stale = "1";
                lbl.style.opacity = "0.4";
            }
        }
    });
}

function _makePinBendingSection(pin, card) {
    // If there was an expanded section, undo its height contribution before replacing
    const existing = card.querySelector(".pin-bending-section");
    if (existing) {
        const oldBody = existing.querySelector(".pin-bending-body.expanded");
        if (oldBody && card.style.height && card._bendBodyH) {
            card.style.height = Math.max(240, parseFloat(card.style.height) - card._bendBodyH) + "px";
            card._bendBodyH = 0;
        }
        existing.remove();
    }

    const pinNode = pin.label;
    const bindings = _bendingBindings.filter(b =>
        b.node === pinNode || b.node.replace(/\./g, '_') === pinNode
    );
    if (bindings.length === 0) return;

    const wasExpanded = card._pinBendingExpanded || false;

    const section = document.createElement("div");
    section.className = "pin-bending-section";

    const toggle = document.createElement("button");
    toggle.className = "pin-bending-toggle";
    const arrow = document.createElement("span");
    arrow.className = "pin-bending-arrow";
    arrow.textContent = wasExpanded ? "▴" : "▾";
    toggle.appendChild(document.createTextNode("⚡ bending "));
    const cnt = document.createElement("span");
    cnt.className = "pin-bending-count";
    cnt.textContent = String(bindings.length);
    toggle.appendChild(cnt);
    toggle.appendChild(document.createTextNode(" "));
    toggle.appendChild(arrow);

    const body = document.createElement("div");
    body.className = "pin-bending-body";

    toggle.addEventListener("click", () => {
        const isExpanded = body.classList.contains("expanded");
        if (!isExpanded) {
            body.classList.add("expanded");
            arrow.textContent = "▴";
            card._pinBendingExpanded = true;
            requestAnimationFrame(() => {
                const bodyH = body.getBoundingClientRect().height;
                card._bendBodyH = bodyH;
                if (card.style.height)
                    card.style.height = (parseFloat(card.style.height) + bodyH) + "px";
            });
        } else {
            const bodyH = card._bendBodyH || body.getBoundingClientRect().height;
            body.classList.remove("expanded");
            arrow.textContent = "▾";
            card._pinBendingExpanded = false;
            if (card.style.height)
                card.style.height = Math.max(240, parseFloat(card.style.height) - bodyH) + "px";
            card._bendBodyH = 0;
        }
    });

    bindings.forEach(b => {
        const row = document.createElement("div");
        row.className = "pin-bop-row";
        row.dataset.bid = b.id;

        const rowHdr = document.createElement("div");
        rowHdr.className = "pin-bop-row-header";

        const nameSpan = document.createElement("span");
        nameSpan.className = "pin-bop-name";
        nameSpan.textContent = b.callback_type || b.callback || "bend";
        nameSpan.appendChild(_bendInfoIcon(b));

        const eyeBtn = document.createElement("button");
        eyeBtn.className = "pin-bop-eye" + (b.vis_muted ? " muted" : "");
        eyeBtn.title = b.vis_muted ? "Unmute for viz" : "Mute for viz";
        eyeBtn.textContent = b.vis_muted ? "○" : "●";
        eyeBtn.addEventListener("click", async () => {
            const live = _bendingBindings.find(x => x.id === b.id);
            const nowMuted = live ? live.vis_muted : b.vis_muted;
            const r = await fetch(`/api/bending/${b.id}/`, {
                method: "PATCH",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({vis_muted: !nowMuted}),
            });
            const d = await r.json();
            if (!r.ok) { showToast("error", d.error, d.traceback); return; }
            _syncBendingState(d);
            _applyVisMutedInPlace();
            _refreshSinglePinActivation(pin);
        });

        const removeBtn = document.createElement("button");
        removeBtn.className = "pin-bop-remove";
        removeBtn.textContent = "×";
        removeBtn.title = "Remove bending";
        removeBtn.addEventListener("click", () => _removeBending(b.id));

        rowHdr.appendChild(nameSpan);
        rowHdr.appendChild(eyeBtn);
        rowHdr.appendChild(removeBtn);

        const paramWrap = document.createElement("div");
        paramWrap.className = "pin-bop-params";
        _buildParamControls(paramWrap, b, "pbc");

        row.appendChild(rowHdr);
        row.appendChild(paramWrap);
        body.appendChild(row);
    });

    section.appendChild(toggle);
    section.appendChild(body);
    card.appendChild(section);

    // Restore expanded state and adjust card height
    if (wasExpanded) {
        body.classList.add("expanded");
        arrow.textContent = "▴";
        requestAnimationFrame(() => {
            const bodyH = body.getBoundingClientRect().height;
            card._bendBodyH = bodyH;
            if (card.style.height)
                card.style.height = (parseFloat(card.style.height) + bodyH) + "px";
        });
    }
}

function _refreshPinCardBendingSections() {
    (pinPages[currentPinPage] || []).forEach(pin => {
        const card = document.getElementById(`pin-card-${pin.id}`);
        if (card) _makePinBendingSection(pin, card);
    });
}

// (Re)build the modular view picker on an activation pin card from pin.data._view_meta.
// Show the "listen" button only when there is something to switch *to*: the
// tensor is long enough to be audio, the audio view accepts its shape, and it
// is not already the one being drawn.
function _syncPinListenBtn(pin, btn) {
    const d = pin && pin.data;
    const meta = d && d._view_meta;
    const compatible = (meta && meta.compatible || []).some(c => c.name === "audio");
    const show = !!(d && d.is_audio_compatible && d.view !== "audio" && compatible);
    btn.style.display = show ? "" : "none";
}

// Change one pin's view, the same way its picker does.
async function _setPinView(pin, view, options) {
    pin.view = { view, options: options || {} };
    _persistPinPages();
    try {
        await fetch(`/api/views/${encodeURIComponent(currentFn)}/${encodeURIComponent(pin.label)}/`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ view, options: options || {} }),
        });
        await _refreshSinglePinActivation(pin);
    } catch (e) { showToast("error", e.message); }
}

function _renderPinViewPicker(pin, card) {
    if (!window.TBViews || !pin || !pin.data || !pin.data._view_meta) return;
    const meta = pin.data._view_meta;
    if (!meta.compatible || meta.compatible.length <= 1) return;
    const vizWrap = card.querySelector(".pin-viz-wrap");
    let host = card.querySelector(".pin-view-picker");
    if (!host) {
        host = document.createElement("div");
        host.className = "pin-view-picker";
        card.insertBefore(host, vizWrap);
    }
    host.innerHTML = "";
    host.appendChild(TBViews.renderPicker(meta, async (view, options) => {
        // kept on the pin too: the server's copy lasts a session (or needs
        // sync), the pin's travels with the dashboard
        pin.view = { view, options: options || {} };
        _persistPinPages();
        try {
            await fetch(`/api/views/${encodeURIComponent(currentFn)}/${encodeURIComponent(pin.label)}/`, {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ view, options }),
            });
            _refreshSinglePinActivation(pin);
        } catch (e) { showToast("error", e.message); }
    }));
}

async function _refreshSinglePinActivation(pin) {
    if (!hasAnyInput() || pin.phNodeId || pin.type === "bending" || pin.type === "bp" || pin.nodeOp === "get_attr") return;
    const card = document.getElementById(`pin-card-${pin.id}`);
    if (!card) return;
    const vizWrap = card.querySelector(".pin-viz-wrap");
    if (!vizWrap) return;

    const hasAnyActiveBendings = _bendingBindings.some(b => !b.vis_muted);
    try {
        const bentResp = await fetch(`/api/activate/${currentFn}/`, {
            method: "POST", body: buildActivationForm([pin.label, pin.label + "_bended"]),
        });
        const bentActs = await bentResp.json();
        const directlyBent = !!(bentActs[pin.label + "_bended"]);
        const bentData = bentActs[pin.label + "_bended"] || bentActs[pin.label];
        if (bentActs.error || !bentData) return;
        pin.data  = bentData;
        pin.shape = pin.data.shape || pin.shape;

        let origData = null;
        if (hasAnyActiveBendings) {
            if (directlyBent) {
                origData = bentActs[pin.label] || null;
            } else {
                try {
                    const origForm = buildActivationForm([pin.label]);
                    origForm.append("original", "true");
                    const origResp = await fetch(`/api/activate/${currentFn}/`, {
                        method: "POST", body: origForm,
                    });
                    const origActs = await origResp.json();
                    if (!origActs.error && origActs[pin.label]) origData = origActs[pin.label];
                } catch (_) {}
            }
        }
        pin.originalData = origData;

        _renderPinViewPicker(pin, card);
        const lb = card.querySelector(".pin-card-listen-btn");
        if (lb) _syncPinListenBtn(pin, lb);
        _purgePlotlyIn(vizWrap);
        vizWrap.innerHTML = "";
        requestAnimationFrame(() => _renderIntoPlotly(
            pin.data, vizWrap,
            Object.assign({ detailed: true, imgZoom: _pinTileZoom, stateKey: "pin/" + pin.id,
                            audioNode: pin.label },
                          pin.originalData ? { originalData: pin.originalData } : {})
        ));
    } catch (_) {}
}

function addPin(label, data, phNodeId) {
    const id = genId();
    const nodeOp = !phNodeId && currentVizNode ? currentVizNode.op : null;
    const originalData = !phNodeId && currentVizData ? (currentVizData.originalData || null) : null;
    const pin = { id, label, shape: data ? data.shape : null, data: data || null, originalData, phNodeId: phNodeId || null, nodeOp };
    pinPages[currentPinPage].push(pin);
    renderPinTabs();
    renderCurrentPage();
    _updatePinCount();
    showToast("info", `Pinned "${label}"`);
}

function addBendingPin(bindingId) {
    const b = _bendingBindings.find(x => x.id === bindingId);
    if (!b) { showToast("error", "Bending not found"); return; }
    // Don't duplicate
    for (const page of pinPages) {
        if (page.some(p => p.type === "bending" && p.bindingId === bindingId)) {
            showToast("info", "Already pinned"); return;
        }
    }
    const pin = { id: genId(), type: "bending", bindingId, label: b.callback_type, bindingSnapshot: b };
    pinPages[currentPinPage].push(pin);
    renderPinTabs();
    renderCurrentPage();
    _updatePinCount();
    showToast("info", `Pinned bending "${b.callback_type}"`);
}

function removePin(id) {
    for (let p = 0; p < pinPages.length; p++) {
        const idx = pinPages[p].findIndex(pin => pin.id === id);
        if (idx >= 0) {
            pinPages[p].splice(idx, 1);
            renderPinTabs();
            renderCurrentPage();
            _updatePinCount();
            return;
        }
    }
}

async function _refreshActivationPins(fn, signal) {
    await _restorePinViews();          // a new session: the cards' views first
    const nodeIds = new Set();
    const busyPins = [];
    // Only the current page: cards on other pages are not in the DOM, so their
    // result was computed, serialized and parsed only to be thrown away. Pages
    // refresh when you switch to them (see switchPinPage).
    (pinPages[currentPinPage] || []).forEach(pin => {
        if (!pin.phNodeId && pin.type !== "bending" && pin.type !== "bp" && pin.nodeOp !== "get_attr" && pin.label) {
            nodeIds.add(pin.label);
            busyPins.push(pin);
        }
    });
    if (nodeIds.size === 0 || !hasAnyInput()) return;

    _setPinsBusy(busyPins, true);
    try {
        const allLabels = [...nodeIds].flatMap(l => [l, l + "_bended"]);
        const bentForm = buildActivationForm(allLabels);
        const acts = await fetch(`/api/activate/${fn}/`, { method: "POST", body: bentForm, signal })
            .then(r => r.json());

        if (acts.error) return;
        pinPages.forEach(page => {
            page.forEach(pin => {
                const directlyBentPin = !!(acts[pin.label + "_bended"]);
                const bentPinData = acts[pin.label + "_bended"] || acts[pin.label];
                if (!pin.phNodeId && pin.type !== "bending" && pin.type !== "bp" && nodeIds.has(pin.label) && bentPinData) {
                    pin.data  = bentPinData;
                    pin.shape = bentPinData.shape || pin.shape;
                    // Only show original overlay for directly bent pins
                    pin.originalData = directlyBentPin ? (acts[pin.label] || null) : null;

                    const card = document.getElementById(`pin-card-${pin.id}`);
                    if (card) {
                        const shpEl = card.querySelector(".pin-card-shape");
                        if (shpEl) shpEl.textContent = pin.shape && pin.shape.length
                            ? `[${pin.shape.join(" × ")}]` : "";
                        const vizWrap = card.querySelector(".pin-viz-wrap");
                        if (vizWrap) {
                            _purgePlotlyIn(vizWrap);
                            vizWrap.innerHTML = "";
                            requestAnimationFrame(() => _renderIntoPlotly(
                                pin.data, vizWrap,
                                Object.assign({ stateKey: "pin/" + pin.id, audioNode: pin.label },
                                              pin.originalData ? { originalData: pin.originalData } : {})
                            ));
                        }
                    }
                }
            });
        });
    } catch (err) {
        if (_isAbort(err)) throw err;   // wave superseded: let the caller know
    } finally {
        _setPinsBusy(busyPins, false);
    }
}

async function _refreshWeightPins(fn, signal) {
    const weightPins = [];
    // current page only — see _refreshActivationPins
    (pinPages[currentPinPage] || []).forEach(pin => {
        if (pin.nodeOp === "get_attr" && pin.label && pin.type !== "bending" && pin.type !== "bp") weightPins.push(pin);
    });
    if (weightPins.length === 0) return;
    _setPinsBusy(weightPins, true);
    await Promise.all(weightPins.map(pin => {
        const hasBending = _bendingBindings.some(b =>
            b.node === pin.label || b.node.replace(/\./g, '_') === pin.label
        );
        const bentP = fetch(`/api/weights/${fn}/${pin.label}/`, { signal }).then(r => r.json());
        const origP = hasBending
            ? fetch(`/api/weights/${fn}/${pin.label}/?original=true`, { signal }).then(r => r.json())
            : Promise.resolve(null);
        return Promise.all([bentP, origP])
            .then(([data, origData]) => {
                if (data.error) return;
                pin.data = data;
                pin.originalData = origData;
                pin.shape = data.shape || pin.shape;
                const card = document.getElementById(`pin-card-${pin.id}`);
                if (!card) return;
                const shpEl = card.querySelector(".pin-card-shape");
                if (shpEl) shpEl.textContent = pin.shape && pin.shape.length ? `[${pin.shape.join(" × ")}]` : "";
                const vizWrap = card.querySelector(".pin-viz-wrap");
                if (vizWrap) {
                    _purgePlotlyIn(vizWrap);
                    vizWrap.innerHTML = "";
                    requestAnimationFrame(() => _renderIntoPlotly(
                        pin.data, vizWrap,
                        Object.assign({ audioNode: pin.label },
                                      origData ? { originalData: origData } : {})
                    ));
                }
            })
            .catch(err => { if (_isAbort(err)) throw err; })
            .finally(() => _setBusy(document.getElementById(`pin-card-${pin.id}`), false));
    }));
}

function clearPins() {
    pinPages = [[]];
    currentPinPage = 0;
    renderPinTabs();
    renderCurrentPage();
    _updatePinCount();
}

function _updatePinCount() {
    const el = document.getElementById("pin-count");
    if (el) el.textContent = pinPages.reduce((s, p) => s + p.length, 0);
}

function togglePinDashboard(force) {
    const panel = document.getElementById("pin-dashboard");
    if (!panel) return;
    const show = force !== undefined ? Boolean(force) : panel.classList.contains("hidden");
    if (!show) _closePinCtxMenu();
    panel.classList.toggle("hidden", !show);
    if (show) {
        renderPinTabs();
        renderCurrentPage();
    }
}

// ─── input panel ─────────────────────────────────────────────────────────────
// A placeholder's type is a question about the node, not only its shape: an
// integer tensor is never audio, and an input the interface knows how to fill
// is whatever the interface says it is.
function guessInputType(node) {
    if (!node) return "numeric";
    if (Array.isArray(node)) return TBInputs.guessType(node);   // bare shape
    return TBInputs.guessType(node.shape, {
        dtype: node.dtype,
        name: node.label || node.id,
        hasMode: !!_inputModeFor(node.id || node.label),
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// ─── interface options ────────────────────────────────────────────────────────
// Settings the interface owns, as opposed to per-run arguments (callbacks) or
// per-placeholder feeds (input modes). Some only take effect the next time the
// model is traced; those say so, and offer the retrace rather than pretending
// the change already landed.
let _ifaceOptions = null;
let _ifaceStale = false;

async function loadInterfaceOptions() {
    const section = document.getElementById("iface-options-section");
    try {
        const r = await fetch("/api/options/");
        const d = await r.json();
        _ifaceOptions = (d && d.options) || [];
        const nameEl = document.getElementById("iface-options-name");
        if (nameEl) nameEl.textContent = d.interface || "";
    } catch (e) {
        _ifaceOptions = [];
    }
    if (!section) return;
    section.style.display = _ifaceOptions.length ? "" : "none";
    _ifaceStale = false;
    _syncIfaceStale();
    if (_ifaceOptions.length) renderInterfaceOptions();
}

function _syncIfaceStale() {
    const el = document.getElementById("iface-options-stale");
    if (el) el.style.display = _ifaceStale ? "" : "none";
}

function renderInterfaceOptions() {
    const host = document.getElementById("iface-options");
    if (!host || !_ifaceOptions) return;
    host.innerHTML = "";
    _ifaceOptions.forEach(opt => host.appendChild(_ifaceOptionRow(opt)));
}

function _ifaceOptionRow(opt) {
    const wrap = document.createElement("div");
    wrap.className = "iface-opt";

    const label = document.createElement("label");
    label.className = "iface-opt-label";
    label.textContent = opt.label || opt.name;
    wrap.appendChild(label);

    const row = document.createElement("div");
    row.className = "iface-opt-row";
    let read;

    if (opt.type === "choice") {
        const sel = document.createElement("select");
        sel.className = "iface-opt-select";
        (opt.choices || []).forEach(c => {
            const o = document.createElement("option");
            o.value = String(c); o.textContent = String(c);
            if (String(c) === String(opt.value)) o.selected = true;
            sel.appendChild(o);
        });
        row.appendChild(sel);
        read = () => sel.value;
        sel.addEventListener("change", () => _commitIfaceOption(opt, read(), wrap));
    } else if (opt.type === "bool") {
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = !!opt.value;
        row.appendChild(cb);
        read = () => cb.checked;
        cb.addEventListener("change", () => _commitIfaceOption(opt, read(), wrap));
    } else {
        const inp = document.createElement("input");
        inp.type = (opt.type === "int" || opt.type === "float") ? "number" : "text";
        inp.className = "iface-opt-input";
        inp.value = opt.value == null ? "" : String(opt.value);
        if (opt.range) { inp.min = opt.range[0]; inp.max = opt.range[1]; }
        if (opt.type === "int") inp.step = 1;
        row.appendChild(inp);
        read = () => inp.value;
        // commit on blur / Enter rather than per keystroke: a setter can be
        // expensive, and a half-typed number is not a value anyone meant
        inp.addEventListener("change", () => _commitIfaceOption(opt, read(), wrap));
    }
    wrap.appendChild(row);

    if (opt.doc) {
        const doc = document.createElement("div");
        doc.className = "iface-opt-doc";
        doc.textContent = opt.doc;
        wrap.appendChild(doc);
    }
    return wrap;
}

async function _commitIfaceOption(opt, value, wrap) {
    const old = wrap.querySelector(".iface-opt-err");
    if (old) old.remove();
    try {
        const r = await fetch(`/api/options/${encodeURIComponent(opt.name)}/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ value }),
        });
        const d = await r.json();
        if (!r.ok || d.error) throw Object.assign(new Error(d.error || r.statusText),
                                                  { traceback: d.traceback || "" });
        // show what the interface actually holds — a setter may have clamped it
        opt.value = d.value;
        const field = wrap.querySelector("select, input");
        if (field) {
            if (field.type === "checkbox") field.checked = !!d.value;
            else field.value = String(d.value);
        }
        if (d.needs === "retrace") { _ifaceStale = true; _syncIfaceStale(); }
        showToast("info", `${opt.label || opt.name} → ${d.value}`);
    } catch (err) {
        const box = document.createElement("div");
        box.className = "iface-opt-err";
        box.textContent = err.message || String(err);
        wrap.appendChild(box);
        // put the widget back to what the interface still holds
        const field = wrap.querySelector("select, input");
        if (field && opt.value != null) {
            if (field.type === "checkbox") field.checked = !!opt.value;
            else field.value = String(opt.value);
        }
    }
}


// ─── interface callbacks ──────────────────────────────────────────────────────
// A traced graph is not everything a model does.  An interface can also carry
// whole operations that run Python around the graph — GPT-2's generate loops
// over the traced forward, sampling a token at a time — which cannot be traced
// and so have no node to click.  The interface declares them in `_callbacks_`,
// saying what each argument is and what comes back, and this panel is built
// from that declaration rather than from anything guessed here.
let _callbackSpecs = null;

async function loadCallbacks() {
    const btn = document.getElementById("callback-panel-btn");
    try {
        const r = await fetch("/api/callbacks/");
        const d = await r.json();
        _callbackSpecs = (d && d.callbacks) || [];
        const label = document.getElementById("callback-interface");
        if (label) label.textContent = d.interface || "";
        // A bare BendedModule declares nothing; offering an empty panel would
        // only raise the question of what belongs in it.
        if (btn) btn.style.display = _callbackSpecs.length ? "" : "none";
        if (_callbackSpecs.length) buildCallbackPanel();
    } catch (e) {
        _callbackSpecs = [];
        if (btn) btn.style.display = "none";
    }
}

function buildCallbackPanel() {
    const host = document.getElementById("callback-items");
    if (!host || !_callbackSpecs) return;
    host.innerHTML = "";
    _callbackSpecs.forEach(spec => host.appendChild(_callbackCard(spec)));
}

function _callbackCard(spec) {
    const card = document.createElement("div");
    card.className = "cb-card";

    const head = document.createElement("div");
    head.className = "cb-card-head";
    const name = document.createElement("span");
    name.className = "cb-name";
    name.textContent = spec.label || spec.name;
    head.appendChild(name);
    if (spec.returns) {
        const ret = document.createElement("span");
        ret.className = "cb-returns";
        ret.textContent = "→ " + spec.returns;
        head.appendChild(ret);
    }
    card.appendChild(head);

    if (spec.doc) {
        const doc = document.createElement("div");
        doc.className = "cb-doc";
        doc.textContent = spec.doc;
        card.appendChild(doc);
    }

    const fields = {};
    (spec.args || []).forEach(arg => {
        const { row, read } = _callbackField(arg);
        card.appendChild(row);
        fields[arg.name] = read;
    });

    const run = document.createElement("button");
    run.className = "cb-run";
    run.textContent = "▷ run " + (spec.label || spec.name);
    card.appendChild(run);

    const result = document.createElement("div");
    result.className = "cb-result";
    card.appendChild(result);

    run.addEventListener("click", async () => {
        const payload = {};
        Object.entries(fields).forEach(([k, read]) => { payload[k] = read(); });
        run.disabled = true;
        run.textContent = "running…";
        result.innerHTML = "";
        try {
            const r = await fetch(`/api/callbacks/${encodeURIComponent(spec.name)}/run/`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const d = await r.json();
            if (!r.ok || d.error) throw Object.assign(new Error(d.error || r.statusText),
                                                      { traceback: d.traceback || "" });
            _renderCallbackResult(result, d);
        } catch (err) {
            const box = document.createElement("div");
            box.className = "cb-error";
            box.textContent = err.message || String(err);
            result.appendChild(box);
            if (err.traceback) showToast("error", err.message, err.traceback);
        } finally {
            run.disabled = false;
            run.textContent = "▷ run " + (spec.label || spec.name);
        }
    });

    return card;
}

// One widget per declared argument.  Returns the row and a reader for its value.
function _callbackField(arg) {
    const row = document.createElement("div");
    row.className = "cb-arg";

    const label = document.createElement("label");
    label.className = "cb-arg-label";
    label.textContent = arg.name;
    if (arg.optional) {
        const opt = document.createElement("span");
        opt.className = "cb-optional";
        opt.textContent = "  optional";
        label.appendChild(opt);
    }
    row.appendChild(label);

    if (arg.type === "text") {
        const ta = document.createElement("textarea");
        ta.className = "cb-text";
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
        const sel = document.createElement("select");
        sel.className = "cb-input";
        (arg.choices || []).forEach(c => {
            const o = document.createElement("option");
            o.value = String(c); o.textContent = String(c);
            if (String(c) === String(arg.default)) o.selected = true;
            sel.appendChild(o);
        });
        row.appendChild(sel);
        return { row, read: () => sel.value };
    }

    if (arg.type === "int" || arg.type === "float") {
        const wrap = document.createElement("div");
        wrap.className = "cb-row";
        const num = document.createElement("input");
        num.type = "number";
        num.className = "cb-input";
        num.value = arg.default == null ? "" : String(arg.default);
        if (arg.range) { num.min = arg.range[0]; num.max = arg.range[1]; }
        num.step = arg.step != null ? arg.step : (arg.type === "int" ? 1 : 0.01);
        // A declared range is what makes a slider meaningful; without one a
        // number box is the honest widget.
        if (arg.range) {
            const rng = document.createElement("input");
            rng.type = "range";
            rng.className = "cb-range";
            rng.min = arg.range[0]; rng.max = arg.range[1]; rng.step = num.step;
            rng.value = num.value === "" ? arg.range[0] : num.value;
            rng.addEventListener("input", () => { num.value = rng.value; });
            num.addEventListener("input", () => { rng.value = num.value; });
            wrap.appendChild(rng);
            num.style.maxWidth = "72px";
        }
        wrap.appendChild(num);
        row.appendChild(wrap);
        return { row, read: () => num.value };
    }

    const inp = document.createElement("input");
    inp.type = "text";
    inp.className = "cb-input";
    inp.value = arg.default == null ? "" : String(arg.default);
    row.appendChild(inp);
    return { row, read: () => inp.value };
}

function _renderCallbackResult(host, d) {
    host.innerHTML = "";
    const meta = document.createElement("div");
    meta.className = "cb-result-meta";
    const res = d.result || {};
    const n = (res.items || []).length;
    // the declared medium, then how it is being drawn — "result" says nothing
    const what = res.medium || res.kind || res.view || "result";
    meta.innerHTML = `<span>${what}${n > 1 ? ` · ${n} items` : ""}</span>`
                   + `<span>${d.run_ms != null ? d.run_ms + " ms" : ""}</span>`;
    host.appendChild(meta);

    if (res.kind === "text" || res.kind === "repr") {
        (res.items || []).forEach(t => {
            const box = document.createElement("div");
            box.className = "cb-text-out";
            box.textContent = t;
            host.appendChild(box);
        });
        return;
    }
    // Everything else came back through the same view system as an activation,
    // so hand it to the same renderer — picker included, so a waveform can be
    // switched to a spectrogram here exactly as it can on a node.
    _renderMediaPayload(host, res, () => _rerunCallback(d.name));
}

// Shared by the editor panel and play mode: draw a view payload, with its
// picker when it has more than one way to be drawn.
function _renderMediaPayload(host, res, onViewChange) {
    const m = res._view_meta;
    if (m && m.compatible && m.compatible.length > 1 && window.TBViews) {
        host.appendChild(TBViews.renderPicker(m, async (view, options) => {
            try {
                await fetch(`/api/views/${encodeURIComponent(res._view_fn || "")}/`
                            + `${encodeURIComponent(res._view_node || "")}/`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ view, options }),
                });
                if (onViewChange) onViewChange();
            } catch (e) { showToast("error", "View change failed"); }
        }));
    }
    const holder = document.createElement("div");
    host.appendChild(holder);
    try {
        // TBViews.render takes (payload, container) — in that order
        if (window.TBViews) TBViews.render(res, holder);
        else holder.textContent = JSON.stringify(res).slice(0, 400);
    } catch (e) {
        holder.textContent = "could not render: " + (e.message || e);
    }
}

function _rerunCallback(name) {
    const cards = [...document.querySelectorAll("#callback-items .cb-card")];
    const card = cards.find(c => (c.querySelector(".cb-name") || {}).textContent
                                 === (name || "").trim()) || cards[0];
    const btn = card && card.querySelector(".cb-run");
    if (btn) btn.click();
}

function toggleCallbackPanel(force) {
    const panel = document.getElementById("callback-panel");
    const btn = document.getElementById("callback-panel-btn");
    if (!panel) return;
    const open = force != null ? force : panel.classList.contains("hidden");
    panel.classList.toggle("hidden", !open);
    if (btn) btn.classList.toggle("active", open);
}


// ─── input modes ──────────────────────────────────────────────────────────────
// Some placeholders have a way in that only the interface knows: GPT-2's
// `input_ids` is a prompt, and turning prose into token ids is the tokenizer's
// job. When the interface declares one, the bench offers it beside the usual
// expression.
//
// A prompt is an *entry*, exactly like an expression or a dropped file — same
// list, same add button, same selection, same persistence. Holding it in a
// state map beside `inputSets` instead is what made it invisible to everything
// driven by that: no choices to pick between, no add step, and no retrace, so
// the value never reached the graph.
// Which display mode a placeholder is showing is a UI preference, and only that.
const _inputModeShown = {};      // placeholder -> mode name, or "" for expr

function _inputModes() {
    return (currentGraphData && currentGraphData.input_modes) || {};
}

function _inputModeFor(name) {
    return _inputModes()[name] || null;
}

// Placeholders filled by whichever mode entry is *selected* — an unselected
// prompt sitting in the list decides nothing.
function _inputModeClaimed() {
    const claimed = {};
    Object.entries(_inputModes()).forEach(([name, spec]) => {
        const entry = _selectedEntry(name);
        if (!entry || entry.type !== "mode") return;
        (spec.fills || [name]).forEach(f => { if (f !== name) claimed[f] = name; });
    });
    return claimed;
}

function _selectedEntry(name) {
    const arr = inputSets[name] || [];
    return arr.find(i => i.id === selectedIds[name] && i.valid) || arr.find(i => i.valid) || null;
}

function _inputModeShownFor(node) {
    const spec = _inputModeFor(node.id);
    if (!spec) return "";
    if (_inputModeShown[node.id] === undefined) {
        // open on the mode when the only entries are prompts, so the pane comes
        // back the way it was left
        const arr = inputSets[node.id] || [];
        _inputModeShown[node.id] = arr.length && arr.every(e => e.type === "mode")
            ? spec.type : "";
    }
    return _inputModeShown[node.id];
}

function _renderInputModeBar(node) {
    const spec = _inputModeFor(node.id);
    if (!spec) return null;
    const shown = _inputModeShownFor(node);

    const bar = document.createElement("div");
    bar.className = "input-mode-bar";
    [["expr", ""], [spec.label || spec.type, spec.type]].forEach(([label, mode]) => {
        const btn = document.createElement("button");
        btn.className = "input-mode-btn" + (shown === mode ? " active" : "");
        btn.textContent = label;
        btn.title = mode
            ? `Feed ${node.id} as ${spec.type} — the interface encodes it`
              + ((spec.fills || []).length > 1
                 ? `, filling ${spec.fills.join(", ")}` : "")
            : "Feed it as an expression or a file";
        btn.addEventListener("click", () => {
            _inputModeShown[node.id] = mode;
            if (currentGraphData) buildInputPanel(currentGraphData);
        });
        bar.appendChild(btn);
    });
    return bar;
}

// The prompt equivalent of buildExprInput: type, then add it to the list.
function _buildModeAdder(node, spec) {
    const wrap = document.createElement("div");
    wrap.className = "input-mode-adder";

    if ((spec.fills || []).length > 1) {
        const note = document.createElement("div");
        note.className = "input-mode-note";
        note.textContent = "also fills " + spec.fills.filter(f => f !== node.id).join(", ");
        wrap.appendChild(note);
    }

    const ta = document.createElement("textarea");
    ta.className = "input-mode-text";
    ta.placeholder = spec.placeholder || "";
    // prefill only while nothing has been added yet, so the box is a suggestion
    // the first time and an empty field for the second prompt
    if (!(inputSets[node.id] || []).some(e => e.type === "mode")) {
        ta.value = spec.default || "";
    }
    wrap.appendChild(ta);

    const addBtn = document.createElement("button");
    addBtn.className = "expr-add-btn input-mode-add";
    addBtn.textContent = "+ add";
    addBtn.disabled = !ta.value.trim();
    wrap.appendChild(addBtn);

    const commit = () => {
        const text = ta.value.trim();
        if (!text) return;
        setInputEntry(node.id, {
            id: genId(), type: "mode", mode: spec.type, value: text,
            label: text.length > 32 ? text.slice(0, 32) + "…" : text,
            valid: true,
        });
        ta.value = "";
        addBtn.disabled = true;
    };
    ta.addEventListener("input", () => { addBtn.disabled = !ta.value.trim(); });
    ta.addEventListener("keydown", (e) => {
        // Enter adds, Shift+Enter keeps the newline — a prompt may want one
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); commit(); }
    });
    addBtn.addEventListener("click", commit);

    // An audio mode takes a recording: typed as a path above, or loaded here
    // from this computer. The file is sent as is and prepared by the
    // interface's own encoder, not the bench's generic waveform loader.
    if (spec.type === "audio") {
        const fileIn = document.createElement("input");
        fileIn.type = "file";
        fileIn.accept = "audio/*,.wav,.flac,.ogg,.aiff,.aif,.mp3";
        fileIn.style.display = "none";
        const pick = document.createElement("button");
        pick.className = "expr-add-btn input-mode-file";
        pick.textContent = "\u{1F4C1} load file…";
        pick.title = "Load a recording from this computer";
        pick.addEventListener("click", () => fileIn.click());
        fileIn.addEventListener("change", () => {
            const f = fileIn.files && fileIn.files[0];
            if (!f) return;
            setInputEntry(node.id, {
                id: genId(), type: "mode", mode: spec.type, value: "", file: f,
                label: f.name, valid: true,
            });
            fileIn.value = "";
        });
        wrap.appendChild(pick);
        wrap.appendChild(fileIn);
    }
    return wrap;
}


function buildInputPanel(data) {
    const container = document.getElementById("input-items");
    container.innerHTML = "";
    _syncBenchBatchBar();

    const placeholders = _modelPlaceholders(data);
    if (placeholders.length === 0) {
        container.innerHTML = '<div class="no-inputs">No placeholder inputs found.</div>';
        return;
    }
    // Say so when these are the model's inputs rather than this view's, so the
    // bench does not look like it is describing the module you are inside.
    if (_scope) {
        const note = document.createElement("div");
        note.className = "input-scope-note";
        note.textContent = `inputs of the whole model — you are inside ${_scope}`;
        container.appendChild(note);
    }

    placeholders.forEach((node) => {
        const type = guessInputType(node);
        const shapeStr = node.shape ? node.shape.join(" × ") : "?";
        const arr = inputSets[node.id] || [];
        const selId = selectedIds[node.id];

        const item = document.createElement("div");
        item.className = "input-item";

        const header = document.createElement("div");
        header.className = "input-item-header";
        header.innerHTML = `<span class="input-item-name">${node.label}</span><span class="input-type-badge input-type-${type}">${type}</span>`;
        item.appendChild(header);

        const shapeDiv = document.createElement("div");
        shapeDiv.className = "input-item-shape";
        shapeDiv.textContent = `[${shapeStr}]`;
        item.appendChild(shapeDiv);

        const modeBar = _renderInputModeBar(node);
        if (modeBar) item.appendChild(modeBar);

        // Driven by another placeholder's mode: editing it here would do
        // nothing, because the encoder fills it on every run.
        const claimedBy = _inputModeClaimed()[node.id];
        if (claimedBy) {
            item.classList.add("input-item-claimed");
            const note = document.createElement("div");
            note.className = "input-mode-note";
            note.textContent = `filled from ${claimedBy}`;
            item.appendChild(note);
            container.appendChild(item);
            return;
        }

        if (arr.length > 0) {
            const list = document.createElement("div");
            list.className = "input-configured-list";
            arr.forEach(inp => {
                const rowWrap = document.createElement("div");

                const row = document.createElement("div");
                const inUse = _batchActive() ? (inp.valid && inp.inBatch !== false) : inp.id === selId;
                row.className = "input-configured-row" + (inUse ? " input-configured-selected" : "");

                const nameSpan = document.createElement("span");
                nameSpan.className = "input-configured-name" + (inp.valid ? "" : " input-configured-invalid");
                nameSpan.textContent = inp.label;
                nameSpan.title = inp.label;

                const selBtn = document.createElement("button");
                selBtn.className = "input-cfg-btn" + (inUse ? " active" : "");
                if (_batchActive()) {
                    // batch: every checked entry goes in; this one in or out
                    selBtn.title = inUse ? "In the batch — click to leave it out" : "Left out — click to put it in the batch";
                    selBtn.textContent = inUse ? "\u2611" : "\u2610";
                    selBtn.addEventListener("click", () => {
                        inp.inBatch = !inUse;
                        _benchBatchChanged();
                    });
                } else {
                    selBtn.title = "Use this input";
                    selBtn.textContent = inp.id === selId ? "●" : "○";
                    selBtn.addEventListener("click", () => { selectInput(node.id, inp.id); buildInputPanel(data); });
                }

                const delBtn = document.createElement("button");
                delBtn.className = "input-cfg-btn";
                delBtn.title = "Remove";
                delBtn.textContent = "×";
                delBtn.addEventListener("click", () => removeInputEntry(node.id, inp.id));

                row.appendChild(nameSpan);

                // expression entries get an inline edit button
                if (inp.type === 'expr') {
                    const editBtn = document.createElement("button");
                    editBtn.className = "input-cfg-btn";
                    editBtn.title = "Edit expression";
                    editBtn.textContent = "✎";
                    editBtn.addEventListener("click", () => {
                        // swap row into edit mode
                        row.innerHTML = "";

                        const editInp = document.createElement("input");
                        editInp.type = "text";
                        editInp.className = "expr-edit-input";
                        editInp.value = inp.expr;
                        editInp.spellcheck = false;

                        const editStatus = document.createElement("span");
                        editStatus.className = "expr-status";

                        const saveBtn = document.createElement("button");
                        saveBtn.className = "input-cfg-btn input-cfg-save";
                        saveBtn.textContent = "✓";
                        saveBtn.title = "Save (Enter)";
                        saveBtn.disabled = true;

                        const cancelBtn = document.createElement("button");
                        cancelBtn.className = "input-cfg-btn";
                        cancelBtn.textContent = "✗";
                        cancelBtn.title = "Cancel (Escape)";

                        row.appendChild(editInp);
                        row.appendChild(editStatus);
                        row.appendChild(saveBtn);
                        row.appendChild(cancelBtn);
                        editInp.focus();
                        editInp.select();

                        let _validExpr = null;
                        let _debounce  = null;

                        editInp.addEventListener("input", () => {
                            const expr = editInp.value.trim();
                            clearTimeout(_debounce);
                            _validExpr = null;
                            saveBtn.disabled = true;
                            if (!expr) { editStatus.textContent = ""; editStatus.className = "expr-status"; return; }
                            editStatus.textContent = "…";
                            editStatus.className = "expr-status expr-pending";
                            _debounce = setTimeout(() => {
                                evalExprPreview(node.id, expr, editStatus, (ok) => {
                                    if (ok) { _validExpr = expr; saveBtn.disabled = false; }
                                });
                            }, 400);
                        });

                        function doSave() {
                            if (!_validExpr) return;
                            inp.expr  = _validExpr;
                            inp.label = _validExpr.length > 35 ? _validExpr.slice(0, 35) + "…" : _validExpr;
                            inp.valid = true;
                            buildInputPanel(data);
                        }

                        saveBtn.addEventListener("click", doSave);
                        cancelBtn.addEventListener("click", () => buildInputPanel(data));
                        editInp.addEventListener("keydown", (e) => {
                            if (e.key === "Enter")  { e.preventDefault(); doSave(); }
                            if (e.key === "Escape") { e.preventDefault(); buildInputPanel(data); }
                        });
                    });
                    row.appendChild(editBtn);
                }

                row.appendChild(selBtn);
                row.appendChild(delBtn);
                rowWrap.appendChild(row);

                // image preview/crop pane for image file entries — shared widget (inputs.js)
                if (inp.type === 'file' && inp.file && _isImageFile(inp.file) && window.TBImage) {
                    const imgHost = document.createElement("div");
                    rowWrap.appendChild(imgHost);
                    TBImage.widget(inp, imgHost, {
                        shape: node.shape,
                        onChange: () => _persistInputsForPlay(currentModelName),
                    }).catch(() => {});
                }

                // a recording loaded for an audio input mode: a plain player to
                // hear it. Not the bench's crop/resample widget — the interface
                // prepares the file itself, and would ignore those settings
                if (inp.type === 'mode' && inp.file) {
                    if (!inp._previewUrl) inp._previewUrl = URL.createObjectURL(inp.file);
                    const player = document.createElement("audio");
                    player.className = "input-mode-audio";
                    player.controls = true;
                    player.preload = "metadata";
                    player.src = inp._previewUrl;
                    rowWrap.appendChild(player);
                }

                // audio pane for audio file entries — shared widget (inputs.js)
                if (inp.type === 'file' && inp.file && window.TBAudio && _isAudioFile(inp.file)) {
                    const audioHost = document.createElement("div");
                    rowWrap.appendChild(audioHost);
                    TBAudio.widget(inp, audioHost, {
                        shape: node.shape,
                        onChange: () => _persistInputsForPlay(currentModelName),
                    }).catch(() => {});
                }

                list.appendChild(rowWrap);
            });
            item.appendChild(list);
        }

        // whichever way in is on show gets the adder; the entries above are
        // the same list either way, so a prompt and an expression sit together
        const shownMode = _inputModeShownFor(node);
        if (shownMode) {
            item.appendChild(_buildModeAdder(node, _inputModeFor(node.id)));
        } else if (type === "image") {
            item.appendChild(buildImageInput(node));
        } else if (type === "audio") {
            item.appendChild(buildAudioInput(node));
        } else {
            item.appendChild(buildNumericInput(node));
        }

        container.appendChild(item);
    });
}

// Shared expression-input widget — used by all three input types.
function buildExprInput(nodeId, placeholder) {
    const wrap = document.createElement("div");
    wrap.className = "numeric-input-wrap";

    const inp = document.createElement("input");
    inp.type = "text";
    inp.className = "numeric-input expr-input";
    inp.placeholder = placeholder;
    inp.spellcheck = false;

    const status = document.createElement("div");
    status.className = "expr-status";

    const addBtn = document.createElement("button");
    addBtn.className = "expr-add-btn";
    addBtn.textContent = "+ add";
    addBtn.style.display = "none";

    let pendingExpr = null;
    let debounceTimer = null;

    inp.addEventListener("input", () => {
        const expr = inp.value.trim();
        clearTimeout(debounceTimer);
        pendingExpr = null;
        addBtn.style.display = "none";
        if (!expr) { status.textContent = ""; status.className = "expr-status"; return; }
        status.textContent = "…";
        status.className = "expr-status expr-pending";
        debounceTimer = setTimeout(() => evalExprPreview(nodeId, expr, status, (valid) => {
            if (valid) { pendingExpr = expr; addBtn.style.display = "inline-block"; }
        }), 500);
    });

    addBtn.addEventListener("click", () => {
        if (!pendingExpr) return;
        const entry = {
            id: genId(), type: "expr", expr: pendingExpr,
            label: pendingExpr.length > 35 ? pendingExpr.slice(0, 35) + "…" : pendingExpr,
            valid: true,
        };
        setInputEntry(nodeId, entry);
        inp.value = ""; status.textContent = ""; status.className = "expr-status";
        addBtn.style.display = "none"; pendingExpr = null;
    });

    wrap.appendChild(inp);
    wrap.appendChild(status);
    wrap.appendChild(addBtn);
    return wrap;
}

function _mediaDropZone(node, accept, icon, label, mimePrefix) {
    const zone = document.createElement("div");
    zone.className = "upload-area";

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = accept;
    fileInput.className = "upload-file-input";

    const hint = document.createElement("div");
    hint.className = "upload-hint";
    hint.innerHTML = `<span class="upload-icon">${icon}</span><span>${label}</span>`;

    zone.appendChild(hint);
    zone.appendChild(fileInput);
    zone.addEventListener("click", () => fileInput.click());
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag-over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
    zone.addEventListener("drop", (e) => {
        e.preventDefault(); zone.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith(mimePrefix)) addFileInput(node.id, file);
    });
    fileInput.addEventListener("change", () => {
        if (fileInput.files[0]) { addFileInput(node.id, fileInput.files[0]); fileInput.value = ""; }
    });
    return zone;
}

function _mediaWithExpr(node, accept, icon, dropLabel, mimePrefix, exprPlaceholder) {
    const container = document.createElement("div");
    container.className = "media-with-expr";
    container.appendChild(_mediaDropZone(node, accept, icon, dropLabel, mimePrefix));

    const sep = document.createElement("div");
    sep.className = "input-or-sep";
    sep.textContent = "or expression";
    container.appendChild(sep);

    container.appendChild(buildExprInput(node.id, exprPlaceholder));
    return container;
}

// `torch.randn` is the wrong suggestion for an integer input: it looks right
// and then fails inside the embedding lookup. Suggest something the declared
// dtype will accept.
function _exprSuggestion(node, fallback) {
    if (!node.shape) return fallback;
    const dims = node.shape.join(", ");
    const dtype = node.dtype || "";
    if (/^(int|uint|long|short)/.test(dtype)) {
        // a small high bound is safe for any vocabulary worth the name; the
        // preview reports it immediately if it is not
        return `torch.randint(0, 10, (${dims}))`;
    }
    if (dtype === "bool") return `torch.ones(${dims}, dtype=torch.bool)`;
    return `torch.randn(${dims})`;
}

function buildImageInput(node) {
    return _mediaWithExpr(node, "image/*", "🖼", "drop image or click to add", "image/",
                          _exprSuggestion(node, "torch.zeros(1, 3, 64, 64)"));
}

function buildAudioInput(node) {
    return _mediaWithExpr(node, "audio/*", "♪", "drop audio or click to add", "audio/",
                          _exprSuggestion(node, "torch.zeros(1, 1, 8000)"));
}

function buildNumericInput(node) {
    return buildExprInput(node.id, _exprSuggestion(node, "torch.zeros(1)"));
}

function evalExprPreview(nodeId, expr, statusEl, callback) {
    fetch(`/api/eval_expr/${currentFn}/${nodeId}/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ expr }),
    })
        .then((r) => r.json())
        .then((data) => {
            if (data.error) {
                statusEl.textContent = "✗ " + data.error;
                statusEl.className = "expr-status expr-error";
                if (callback) callback(false);
            } else {
                const shape = data.shape && data.shape.length
                    ? "[" + data.shape.join(" × ") + "]" : "ok";
                statusEl.textContent = "✓ " + shape;
                statusEl.className = "expr-status expr-ok";
                if (callback) callback(true);
            }
        })
        .catch(() => {
            statusEl.textContent = "✗ server error";
            statusEl.className = "expr-status expr-error";
            if (callback) callback(false);
        });
}

// ─── the bench as a large window ─────────────────────────────────────────────
// Same panel, same builder; ⤢ just gives it the middle of the screen and lays
// the inputs out as a grid of cards. Esc or the backdrop brings it back.
function setInputPanelExpanded(on) {
    const panel = document.getElementById("input-panel");
    const back = document.getElementById("input-panel-backdrop");
    const btn = document.getElementById("expand-input-panel");
    if (!panel) return;
    if (on && panel.classList.contains("hidden")) toggleInputPanel();
    panel.classList.toggle("expanded", !!on);
    if (back) back.classList.toggle("visible", !!on);
    if (btn) {
        btn.textContent = on ? "\u2921" : "\u2922";
        btn.title = on ? "Back to the side panel (Esc)" : "Open the inputs in a large window (Esc to return)";
    }
    try { localStorage.setItem("tb_inputs_expanded", on ? "1" : "0"); } catch (_) {}
}

function toggleInputPanel() {
    const panel = document.getElementById("input-panel");
    const btn = document.getElementById("input-panel-btn");
    const isOpen = !panel.classList.contains("hidden");

    if (isOpen) {
        panel.classList.add("hidden");
        btn.classList.remove("active");
    } else {
        panel.classList.remove("hidden");
        btn.classList.add("active");
        if (currentGraphData) buildInputPanel(currentGraphData);
    }
}

// ─── device switch ──────────────────────────────────────────────────────────
// Mirrors play mode's device dropdown (see play.js `loadDevices`), applied to
// the module the editor itself explores rather than a compiled play runtime.
// `_deviceCompat[fn][device]` is false when the active interface's
// `_device_compat_` says `fn` is not expected to work on `device`; those
// options are shown, disabled, rather than left out, so it is clear the
// device exists and *why* it is not offered here.
let _deviceCompat = {};

async function loadDevices() {
    try {
        const r = await fetch("/api/devices/");
        const data = await r.json();
        _deviceCompat = data.compat || {};
        const sel = document.getElementById("graph-device");
        sel.innerHTML = "";
        (data.devices || [{ value: "cpu", label: "CPU" }]).forEach((d) => {
            const opt = document.createElement("option");
            opt.value = d.value;
            opt.textContent = d.label;
            sel.appendChild(opt);
        });
        sel.value = data.current || "cpu";
        sel.dataset.applied = sel.value;
        applyDeviceCompat(currentFn);
    } catch (e) { /* cpu only */ }
}

function applyDeviceCompat(fn) {
    const sel = document.getElementById("graph-device");
    if (!sel) return;
    const compat = _deviceCompat[fn] || {};
    Array.from(sel.options).forEach((opt) => {
        const ok = compat[opt.value] !== false;
        opt.disabled = !ok;
        opt.title = ok ? "" : `${fn} is not expected to work on ${opt.value}`;
    });
    // the active device just became unsupported for the newly selected method
    // (switching fn, not switching device) — say so rather than silently fail
    // on the next request
    if (compat[sel.value] === false) {
        showToast("warn", `${fn} is not expected to work on ${sel.value} — switch device or expect it to fail`);
    }
}

async function setDevice(device) {
    const sel = document.getElementById("graph-device");
    const prev = sel.dataset.applied || "cpu";
    try {
        const r = await fetch("/api/devices/set/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ device, fn: currentFn }),
        });
        const data = await r.json();
        if (!r.ok || data.error) throw new Error(data.error || r.statusText);
        sel.dataset.applied = device;
        showToast("info", `Now running on ${device}`);
        loadGraph(currentFn);   // cached activations were invalidated server-side
    } catch (e) {
        showToast("error", `Could not switch device: ${e.message}`);
        sel.value = prev;   // the switch did not happen — say so in the control too
    }
}

// ─── bending info tooltip ─────────────────────────────────────────────────────
// A small ⓘ next to a bending wherever one is shown (details panel, activations
// window, dashboard cards). Hovering it lists everything the bending is: its
// target, each parameter's current value — and, for a linked one, the macro
// driving it and the range it maps to — and the options it was built with.
// Read from the live binding at hover time, so values follow the sliders.
function _fmtBendVal(v) {
    if (v === null || v === undefined) return "none";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4)));
    if (typeof v === "boolean") return v ? "on" : "off";
    if (Array.isArray(v)) return "[" + v.map(_fmtBendVal).join(", ") + "]";
    return String(v);
}

let _bendInfoAnchor = null;

function _hideBendInfo() {
    _bendInfoAnchor = null;
    const tip = document.getElementById("bend-info-tip");
    if (tip) tip.style.display = "none";
}

// Rows re-render as sliders move: an icon replaced under the pointer never gets
// its mouseleave, and the tooltip would stay up for good. Drop it as soon as
// its anchor is gone or the pointer is no longer on it.
document.addEventListener("mousemove", (e) => {
    if (_bendInfoAnchor && (!_bendInfoAnchor.isConnected || !_bendInfoAnchor.contains(e.target)))
        _hideBendInfo();
}, true);

function _showBendInfo(anchor, id, fallback) {
    const b = (_bendingBindings || []).find(x => x.id === id) || fallback;
    if (!b) return;
    let tip = document.getElementById("bend-info-tip");
    if (!tip) {
        tip = document.createElement("div");
        tip.id = "bend-info-tip";
        document.body.appendChild(tip);
    }
    _bendInfoAnchor = anchor;
    tip.innerHTML = "";
    const add = (cls, text) => {
        const d = document.createElement("div");
        d.className = cls;
        d.textContent = text;
        tip.appendChild(d);
        return d;
    };
    add("bit-title", (b.name ? b.name + " — " : "") + (b.callback_type || "bending"));
    add("bit-target", `${b.fn || currentFn || ""}: ${(b.nodes || [b.node]).filter(Boolean).join(", ")}`);
    if (b.vis_muted) add("bit-flag", "preview muted");

    const rows = (title, entries) => {
        if (!entries.length) return;
        add("bit-section", title);
        const table = document.createElement("div");
        table.className = "bit-table";
        entries.forEach(([k, v, link]) => {
            const kEl = document.createElement("span");
            kEl.className = "bit-k";
            kEl.textContent = k;
            const vEl = document.createElement("span");
            vEl.className = "bit-v";
            vEl.textContent = v;
            if (link) {
                const l = document.createElement("span");
                l.className = "bit-link";
                l.textContent = "  \u2190 " + link;
                vEl.appendChild(l);
            }
            table.appendChild(kEl);
            table.appendChild(vEl);
        });
        tip.appendChild(table);
    };
    const links = b.bp_links || {}, maps = b.bp_maps || {};
    rows("parameters", Object.entries(b.params || {}).map(([k, v]) => {
        const macro = links[k];
        const range = maps[k] ? ` [${_fmtBendVal(maps[k][0])}, ${_fmtBendVal(maps[k][1])}]` : "";
        return [k, _fmtBendVal(v), macro ? macro + range : null];
    }));
    rows("init options", Object.entries(b.init_params || {}).map(([k, v]) => [k, _fmtBendVal(v), null]));

    tip.style.display = "block";
    const r = anchor.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = r.right + 8, top = r.top - 4;
    if (left + tw > window.innerWidth - 8) left = Math.max(8, r.left - tw - 8);
    if (top + th > window.innerHeight - 8) top = Math.max(8, window.innerHeight - th - 8);
    tip.style.left = left + "px";
    tip.style.top = top + "px";
}

function _bendInfoIcon(b) {
    const icon = document.createElement("span");
    icon.className = "bend-info-icon";
    icon.textContent = "\u24D8";
    icon.setAttribute("aria-label", "bending parameters");
    icon.addEventListener("mouseenter", () => _showBendInfo(icon, b && b.id, b));
    icon.addEventListener("mouseleave", _hideBendInfo);
    // hovering is the whole point; a click must not also trigger the row it sits in
    icon.addEventListener("click", (e) => e.stopPropagation());
    return icon;
}

// ─── method buttons (dynamic, rebuilt on model switch) ────────────────────────
function renderMethodButtons(methods) {
    const selector = document.getElementById("method-selector");
    selector.innerHTML = "";
    methods.forEach((m) => {
        const btn = document.createElement("button");
        btn.className = "method-btn";
        btn.dataset.fn = m;
        btn.textContent = m;
        btn.addEventListener("click", () => loadGraph(btn.dataset.fn));
        selector.appendChild(btn);
    });
}

// ─── model picker ─────────────────────────────────────────────────────────────
let _pickerOpen = false;

function renderModelPicker(models) {
    const popup = document.getElementById("model-picker-popup");
    popup.innerHTML = "";

    // hide arrow when there is only one model — picker still works but is cosmetically quiet
    const arrow = document.getElementById("picker-arrow");
    if (arrow) arrow.style.display = models.length > 1 ? "inline" : "none";

    models.forEach((m) => {
        const row = document.createElement("div");
        row.className = "model-row" + (m.current ? " model-row-current" : "");

        const name = document.createElement("span");
        name.className = "model-row-name";
        name.textContent = m.name;

        const badges = document.createElement("span");
        badges.className = "model-row-badges";

        const statusBadge = document.createElement("span");
        statusBadge.className = "model-badge model-badge-" + m.status;
        statusBadge.textContent = m.status;
        badges.appendChild(statusBadge);

        if (m.param_mb != null) {
            const memBadge = document.createElement("span");
            memBadge.className = "model-badge model-badge-mem";
            memBadge.textContent = m.param_mb + " MB";
            badges.appendChild(memBadge);
        }

        row.appendChild(name);
        row.appendChild(badges);

        row.addEventListener("click", () => {
            toggleModelPicker(false);
            if (!m.current) selectModel(m.name);
        });

        popup.appendChild(row);
    });
}

function toggleModelPicker(force) {
    _pickerOpen = (force !== undefined) ? Boolean(force) : !_pickerOpen;
    const popup = document.getElementById("model-picker-popup");
    const btn   = document.getElementById("model-picker-btn");
    popup.classList.toggle("open", _pickerOpen);
    btn.classList.toggle("active", _pickerOpen);
}

function selectModel(name) {
    _saveModelInputs(currentModelName);
    _saveModelPins(currentModelName);
    clearTimeout(_clientStatePushTimer);
    _pushClientState();   // flush current model state before switching
    document.getElementById("module-name").textContent = name + "…";
    showLoading(true);

    fetch(`/api/models/select/${encodeURIComponent(name)}/`, { method: "POST" })
        .then((r) => r.json())
        .then((data) => {
            if (data.error) {
                showToast("error", data.error);
                // restore previous module name
                fetch("/api/models/")
                    .then((r) => r.json())
                    .then((d) => {
                        const cur = (d.models || []).find((m) => m.current);
                        if (cur) document.getElementById("module-name").textContent = cur.name;
                    })
                    .catch(() => {});
                showLoading(false);
                return;
            }
            currentModelName = name;
            // A scope is a path into *this* model's module tree. Carrying it to
            // the next model lands on a module that does not exist there and
            // draws an empty graph, with no obvious way back.
            _scope = "";
            toggleCallbackPanel(false);
            loadCallbacks();
            loadInterfaceOptions();
            loadDevices();   // a different model may be a different interface
            _restoreModelInputs(name);
            _restoreModelPins(name);
            _fetchBendings();
            document.getElementById("module-name").textContent = data.module_type || data.name;
            renderMethodButtons(data.methods || []);
            renderModelPicker(data.models || []);
            // reset selection state
            currentFn = null;
            clearSelection();
            // Fetch saved state first so _applyDefaultInputs only fills missing keys.
            // This prevents a race where stale saved inputs overwrite correct defaults.
            const firstMethod = data.methods && data.methods.length > 0 ? data.methods[0] : null;
            _fetchAndApplyClientState(name)
                .then(() => _restoreSharedInputs(name))
                .then(() => {
                    if (firstMethod) {
                        loadGraph(firstMethod);
                    } else {
                        showLoading(false);
                        showToast("warn", `'${name}' has no traced methods`);
                    }
                });
        })
        .catch((err) => {
            showToast("error", "Switch failed: " + err.message, err.traceback);
            showLoading(false);
        });
}

// ─── toast ────────────────────────────────────────────────────────────────────
// ── error log ─────────────────────────────────────────────────────────────────
const _errorLog = [];

function _addToErrorLog(msg, tb) {
    const entry = { ts: new Date(), msg, tb: tb || "" };
    _errorLog.push(entry);
    _renderErrorLogEntry(entry);
    // Update badge
    const badge = document.getElementById("error-log-badge");
    const count = document.getElementById("error-log-count");
    if (badge) { badge.style.display = ""; if (count) count.textContent = _errorLog.length; }
}

function _renderErrorLogEntry(entry) {
    const body = document.getElementById("error-log-body");
    if (!body) return;
    const empty = body.querySelector(".error-log-empty");
    if (empty) empty.remove();

    const div = document.createElement("div");
    div.className = "error-log-entry";

    const hdr = document.createElement("div");
    hdr.className = "error-log-entry-header";
    const ts = document.createElement("span");
    ts.className = "error-log-ts";
    ts.textContent = entry.ts.toLocaleTimeString();
    const msgEl = document.createElement("span");
    msgEl.className = "error-log-msg";
    msgEl.textContent = entry.msg;
    const toggle = document.createElement("span");
    toggle.className = "error-log-toggle";
    toggle.textContent = entry.tb ? "▸" : "";
    hdr.appendChild(ts); hdr.appendChild(msgEl); hdr.appendChild(toggle);

    const tbEl = document.createElement("pre");
    tbEl.className = "error-log-tb" + (entry.tb ? " hidden" : " hidden");
    tbEl.textContent = entry.tb || "";

    if (entry.tb) {
        hdr.addEventListener("click", () => {
            const open = tbEl.classList.toggle("hidden");
            toggle.textContent = open ? "▸" : "▾";
        });
    }

    div.appendChild(hdr);
    if (entry.tb) div.appendChild(tbEl);
    body.appendChild(div);
    body.scrollTop = body.scrollHeight;
}

function _initErrorLog() {
    const badge  = document.getElementById("error-log-badge");
    const panel  = document.getElementById("error-log-panel");
    const body   = document.getElementById("error-log-body");
    if (!badge || !panel || !body) return;

    // Show empty state
    const empty = document.createElement("div");
    empty.className = "error-log-empty"; empty.textContent = "No errors yet";
    body.appendChild(empty);

    badge.addEventListener("click", () => { panel.style.display = panel.style.display === "none" ? "flex" : "none"; });
    document.getElementById("error-log-close").addEventListener("click",  () => { panel.style.display = "none"; });
    document.getElementById("error-log-clear").addEventListener("click",  () => {
        _errorLog.length = 0;
        body.innerHTML = "";
        const e = document.createElement("div"); e.className = "error-log-empty"; e.textContent = "No errors yet";
        body.appendChild(e);
        badge.style.display = "none";
        const count = document.getElementById("error-log-count");
        if (count) count.textContent = "0";
    });
    document.getElementById("error-log-copy").addEventListener("click", () => {
        const text = _errorLog.map((e, i) =>
            `[${i + 1}] ${e.ts.toISOString()}  ${e.msg}${e.tb ? "\n" + e.tb : ""}`
        ).join("\n\n---\n\n");
        navigator.clipboard.writeText(text).then(
            () => showToast("ok", "Error log copied"),
            () => showToast("warn", "Could not copy — check clipboard permissions")
        );
    });
}

function showToast(type, msg, traceback) {
    // For errors: also add to the persistent log
    if ((type === "error" || type === "warn") && msg) {
        _addToErrorLog(msg, traceback || "");
    }
    const toast = document.getElementById("toast");
    if (!toast) return;
    toast.className = "toast toast-" + type;
    toast.textContent = msg;
    toast.style.display = "block";
    clearTimeout(toast._hideTimer);
    toast._hideTimer = setTimeout(() => { toast.style.display = "none"; }, 4500);
}

// ─── show-bent mode ───────────────────────────────────────────────────────────

let _showBentMode = false;
const _LOCATOR_MARGIN = 28;   // px from viewport edge where locators sit

function _toggleShowBent() {
    _showBentMode = !_showBentMode;
    document.getElementById("show-bent-btn").classList.toggle("active", _showBentMode);
    _applyShowBent();
}

function _applyShowBent() {
    if (!cy || !currentGraphData) return;

    const bentLocators = document.getElementById("bent-locators");

    if (!_showBentMode) {
        cy.elements().removeClass(
            "show-bent-dim show-bent-focus show-bent-dim-edge show-bent-focus-edge"
        );
        if (bentLocators) { bentLocators.innerHTML = ""; bentLocators.style.display = "none"; }
        return;
    }

    const bentIds = new Set(
        currentGraphData.nodes
            .filter(n => (n.has_bending || n.has_macro) && !n.is_compound)
            .map(n => n.id)
    );

    if (bentIds.size === 0) {
        showToast("warn", "No bent nodes in this graph");
        _showBentMode = false;
        document.getElementById("show-bent-btn").classList.remove("active");
        return;
    }

    cy.nodes().forEach(node => {
        if (node.data("is_compound")) return;
        if (bentIds.has(node.id())) {
            node.addClass("show-bent-focus").removeClass("show-bent-dim");
        } else {
            node.addClass("show-bent-dim").removeClass("show-bent-focus");
        }
    });
    cy.edges().forEach(edge => {
        const touches = bentIds.has(edge.source().id()) || bentIds.has(edge.target().id());
        if (touches) {
            edge.addClass("show-bent-focus-edge").removeClass("show-bent-dim-edge");
        } else {
            edge.addClass("show-bent-dim-edge").removeClass("show-bent-focus-edge");
        }
    });

    _updateBentLocators();
}

function _updateNodeInfoOverlay() {
    const overlay = document.getElementById("node-info-overlay");
    if (!overlay || !cy) return;
    overlay.innerHTML = "";
    if (_goptNodeInfoMode !== "permanent") return;
    const layoutDir = (document.getElementById("layout-select") || {}).value || "TB";
    cy.nodes().forEach(node => {
        if (node.data("is_compound")) return;
        if (node.data("is_alias")) return;   // the original already carries the pill
        if (node.style("display") === "none") return;
        const d = node.data();
        const parts = [];
        if (d.op)     parts.push(`<span class="pill-op">${d.op}</span>`);
        if (d.target) parts.push(`<span class="pill-target">${d.target}</span>`);
        if (_goptNodeInfoShape && d.shape && d.shape.length)
            parts.push(`<span class="pill-shape">[${d.shape.join("×")}]</span>`);
        if (!parts.length) return;
        const bb = node.renderedBoundingBox();
        const pill = document.createElement("div");
        pill.className = "node-info-pill";
        pill.innerHTML = parts.join("");
        if (layoutDir === "LR") {
            // left-to-right: pill to the RIGHT — uses horizontal inter-layer space
            pill.style.left     = (bb.x2 + 6) + "px";
            pill.style.top      = ((bb.y1 + bb.y2) / 2) + "px";
            pill.style.transform = "translateY(-50%)";
        } else {
            // top-to-bottom: pill BELOW the node — uses vertical inter-layer space
            pill.style.left     = ((bb.x1 + bb.x2) / 2) + "px";
            pill.style.top      = (bb.y2 + 4) + "px";
            pill.style.transform = "translateX(-50%)";
        }
        overlay.appendChild(pill);
    });
}

function _updateBentLocators() {
    const overlay = document.getElementById("bent-locators");
    if (!overlay || !cy || !currentGraphData) return;

    if (!_showBentMode) { overlay.style.display = "none"; return; }
    overlay.style.display = "";
    overlay.innerHTML = "";

    const W = overlay.offsetWidth  || overlay.clientWidth  || 800;
    const H = overlay.offsetHeight || overlay.clientHeight || 600;
    const M = _LOCATOR_MARGIN;   // margin from edge
    const cx = W / 2, cy_c = H / 2;

    const bentNodes = currentGraphData.nodes.filter(n => (n.has_bending || n.has_macro) && !n.is_compound);

    bentNodes.forEach(n => {
        const cyNode = cy.getElementById(n.id);
        if (!cyNode || !cyNode.length) return;

        const rp = cyNode.renderedPosition();
        // Is the node centre within the visible "safe" area?
        if (rp.x >= M && rp.x <= W - M && rp.y >= M && rp.y <= H - M) return;

        // Direction from viewport centre toward the node
        const dx = rp.x - cx, dy = rp.y - cy_c;
        const angle = Math.atan2(dy, dx);   // 0 = right, PI/2 = down

        // Edge intersection of the ray (cx,cy_c)→(dx,dy) with the margin rectangle
        const pt = _rayRectIntersect(cx, cy_c, dx, dy, M, W - M, M, H - M);

        // Arrow element — clip-path triangle points "up" (–π/2); rotate to point outward
        const arrow = document.createElement("div");
        arrow.className = "bent-locator";
        arrow.style.left = Math.round(pt.x) + "px";
        arrow.style.top  = Math.round(pt.y) + "px";
        arrow.style.transform =
            `translate(-50%,-50%) rotate(${(angle + Math.PI / 2).toFixed(4)}rad)`;
        arrow.style.pointerEvents = "all";
        arrow.title = n.id;
        arrow.addEventListener("click", () => {
            cy.animate({ fit: { eles: cyNode, padding: 120 }, duration: 350 });
            onNodeClick(cyNode);
        });

        // Label — pulled back 20 px toward the centre so it sits just inside the arrow
        const lx = pt.x - Math.cos(angle) * 20;
        const ly = pt.y - Math.sin(angle) * 20;
        const label = document.createElement("div");
        label.className = "bent-locator-label";
        label.textContent = n.id;
        label.style.left = Math.round(lx) + "px";
        label.style.top  = Math.round(ly) + "px";

        overlay.appendChild(arrow);
        overlay.appendChild(label);
    });
}

/** Smallest-t intersection of ray from (rx,ry) with direction (dx,dy) and axis-aligned rectangle. */
function _rayRectIntersect(rx, ry, dx, dy, xMin, xMax, yMin, yMax) {
    let t = Infinity;
    if (dx > 1e-6)  t = Math.min(t, (xMax - rx) / dx);
    if (dx < -1e-6) t = Math.min(t, (xMin - rx) / dx);
    if (dy > 1e-6)  t = Math.min(t, (yMax - ry) / dy);
    if (dy < -1e-6) t = Math.min(t, (yMin - ry) / dy);
    if (!isFinite(t)) t = 0;
    return { x: rx + t * dx, y: ry + t * dy };
}

// ─── nav prev/next locators ──────────────────────────────────────────────────

// Return how sourceId appears in targetNode.args: "[0]", "[2][1]", or null.
function _findArgIdx(args, sourceId) {
    if (!args) return null;
    for (let i = 0; i < args.length; i++) {
        const a = args[i];
        if (a.type === "node" && a.name === sourceId) return `[${i}]`;
        if (a.type === "list" && a.items) {
            for (let j = 0; j < a.items.length; j++) {
                if (a.items[j].type === "node" && a.items[j].name === sourceId) return `[${i}][${j}]`;
            }
        }
    }
    return null;
}

function _buildNavHint(el, nodeData, which, edgeInfo) {
    el.innerHTML = "";
    const nameEl = document.createElement("div");
    nameEl.className = `nav-hint-node nav-hint-node-${which}`;
    nameEl.textContent = nodeData.label || nodeData.id;
    el.appendChild(nameEl);
    if (nodeData.shape && nodeData.shape.length) {
        const shapeEl = document.createElement("div");
        shapeEl.className = "nav-hint-shape";
        shapeEl.textContent = `[${nodeData.shape.join(" × ")}]`;
        el.appendChild(shapeEl);
    }
    if (edgeInfo) {
        const edgeEl = document.createElement("div");
        edgeEl.className = "nav-hint-edge";
        const opSpan = document.createElement("span");
        opSpan.className = "nav-hint-edge-op";
        opSpan.textContent = edgeInfo.op;
        const sepSpan = document.createTextNode(" ");
        const argSpan = document.createElement("span");
        argSpan.className = "nav-hint-edge-arg";
        argSpan.textContent = edgeInfo.arg;
        edgeEl.appendChild(opSpan);
        edgeEl.appendChild(sepSpan);
        edgeEl.appendChild(argSpan);
        el.appendChild(edgeEl);
    }
    el.style.display = "block";
}

function _navPositionPanel(cyNode) {
    const panel = document.getElementById("graph-nav-panel");
    if (!panel || panel.classList.contains("code-open") || !cy) return;
    if (!cyNode || cyNode.empty()) return;

    const cyEl  = cy.container();
    const cyRect = cyEl.getBoundingClientRect();
    const bb     = cyNode.renderedBoundingBox({ includeLabels: false });

    const nodeCx     = cyRect.left + (bb.x1 + bb.x2) / 2;
    const nodeBottom = cyRect.top  + bb.y2;
    const nodeTop    = cyRect.top  + bb.y1;

    const sidebarFolded = document.getElementById("sidebar")?.classList.contains("pane-folded");
    const detailsFolded = document.getElementById("details-panel")?.classList.contains("pane-folded");
    const areaLeft  = (sidebarFolded ? 0 : 175) + 6;
    const areaRight = window.innerWidth - (detailsFolded ? 0 : 230) - 6;

    const pw = panel.offsetWidth  || 300;
    const ph = panel.offsetHeight || 140;
    const GAP = 14;

    let left = nodeCx - pw / 2;
    let top;
    // Prefer below node; flip above if too close to bottom
    if (nodeBottom + ph + GAP < window.innerHeight - 8) {
        top = nodeBottom + GAP;
    } else {
        top = nodeTop - ph - GAP;
    }

    left = Math.max(areaLeft, Math.min(areaRight - pw, left));
    top  = Math.max(4, Math.min(window.innerHeight - ph - 4, top));

    panel.style.left      = Math.round(left) + "px";
    panel.style.top       = Math.round(top)  + "px";
    panel.style.bottom    = "auto";
    panel.style.transform = "none";
}

function _navResetPanelPosition() {
    const panel = document.getElementById("graph-nav-panel");
    if (!panel) return;
    panel.style.left = panel.style.top = panel.style.bottom = panel.style.transform = "";
}

function _updateNavLocators(prevNodeData, nextNodeData) {
    const overlay = document.getElementById("nav-locators");
    if (!overlay || !cy) return;

    cy.nodes().removeClass("nav-prev nav-next");
    overlay.innerHTML = "";

    if (!prevNodeData && !nextNodeData) {
        overlay.style.display = "none";
        return;
    }
    overlay.style.display = "";

    const W   = overlay.offsetWidth  || overlay.clientWidth  || 800;
    const H   = overlay.offsetHeight || overlay.clientHeight || 600;
    const M   = _LOCATOR_MARGIN;
    const dir = (document.getElementById("layout-select") || {}).value || "TB";

    // Sidebar (175px left) and details panel (230px right) sit on top of the canvas.
    // Chips must be clamped to the actual visible graph area.
    const sidebarFolded = document.getElementById("sidebar")?.classList.contains("pane-folded");
    const detailsFolded = document.getElementById("details-panel")?.classList.contains("pane-folded");
    const L = (sidebarFolded ? 0 : 175) + 4;          // min chip left
    const R = W - (detailsFolded ? 0 : 230) - 4;      // max chip right (exclusive — subtract chip width before clamping)
    const cx  = (L + R) / 2, cy_c = H / 2;

    function _makeChip(isPrev, label) {
        const arrowGlyph = isPrev
            ? (dir === "LR" ? "←" : "↑")
            : (dir === "LR" ? "→" : "↓");
        const chip = document.createElement("div");
        chip.className = "nav-chip " + (isPrev ? "nav-chip-prev" : "nav-chip-next");
        const arrowSpan = document.createElement("span");
        arrowSpan.className = "nav-chip-arrow";
        arrowSpan.textContent = arrowGlyph;
        const labelSpan = document.createElement("span");
        labelSpan.className = "nav-chip-label";
        labelSpan.textContent = label;
        chip.appendChild(arrowSpan);
        chip.appendChild(labelSpan);
        return chip;
    }

    // Append chip, read its rendered size, then set a clamped absolute position.
    // idealLeft / idealTop are where the chip's top-left corner *wants* to be.
    function _commitChip(chip, idealLeft, idealTop) {
        overlay.appendChild(chip);
        const cw = chip.offsetWidth  || 100;
        const ch = chip.offsetHeight || 18;
        chip.style.left = Math.max(L, Math.min(R - cw, Math.round(idealLeft))) + "px";
        chip.style.top  = Math.max(4, Math.min(H - ch - 4, Math.round(idealTop)))  + "px";
    }

    function _placeLocator(nodeData, isPrev) {
        const cyNode = cy.$id(nodeData.id);
        if (!cyNode || !cyNode.length) return;
        cyNode.addClass(isPrev ? "nav-prev" : "nav-next");
        cyNode.removeClass("dimmed");

        const label    = nodeData.label || nodeData.id;
        const rp       = cyNode.renderedPosition();
        const onScreen = rp.x >= L + M && rp.x <= R - M && rp.y >= M && rp.y <= H - M;

        if (onScreen) {
            // Place chip adjacent to the node in the flow direction.
            // We append first to read real dimensions, then set the clamped position.
            const bb   = cyNode.renderedBoundingBox({ includeLabels: false });
            const chip = _makeChip(isPrev, label);
            chip.style.position = "absolute";
            overlay.appendChild(chip);
            const cw = chip.offsetWidth  || 100;
            const ch = chip.offsetHeight || 18;

            let left, top;
            if (dir === "LR") {
                // horizontal flow: prev → left of node, next → right
                top  = Math.max(4, Math.min(H - ch - 4, Math.round((bb.y1 + bb.y2) / 2 - ch / 2)));
                left = isPrev
                    ? Math.max(L, Math.min(R - cw, Math.round(bb.x1 - 6 - cw)))
                    : Math.max(L, Math.min(R - cw, Math.round(bb.x2 + 6)));
            } else {
                // vertical flow: prev → above node, next → below
                left = Math.max(L, Math.min(R - cw, Math.round((bb.x1 + bb.x2) / 2 - cw / 2)));
                top  = isPrev
                    ? Math.max(4, Math.min(H - ch - 4, Math.round(bb.y1 - 6 - ch)))
                    : Math.max(4, Math.min(H - ch - 4, Math.round(bb.y2 + 6)));
            }
            chip.style.left = left + "px";
            chip.style.top  = top  + "px";
        } else {
            // Off-screen: edge-of-viewport triangle + chip clamped to visible graph area.
            const dx    = rp.x - cx, dy = rp.y - cy_c;
            const angle = Math.atan2(dy, dx);
            const pt    = _rayRectIntersect(cx, cy_c, dx, dy, L + M, R - M, M, H - M);

            const tri = document.createElement("div");
            tri.className = "nav-locator " + (isPrev ? "nav-locator-prev" : "nav-locator-next");
            tri.style.left = Math.round(pt.x) + "px";
            tri.style.top  = Math.round(pt.y) + "px";
            tri.style.transform = `translate(-50%,-50%) rotate(${(angle + Math.PI / 2).toFixed(4)}rad)`;
            tri.style.pointerEvents = "all";
            tri.title = label;
            tri.addEventListener("click", () => {
                cy.animate({ fit: { eles: cyNode, padding: 120 }, duration: 350 });
            });
            overlay.appendChild(tri);

            const OFF   = 28;
            const chip  = _makeChip(isPrev, label);
            chip.style.position      = "absolute";
            chip.style.pointerEvents = "none";
            _commitChip(
                chip,
                pt.x - Math.cos(angle) * OFF - 50,
                pt.y - Math.sin(angle) * OFF - 9
            );
            const cw = chip.offsetWidth  || 100;
            const ch = chip.offsetHeight || 18;
            chip.style.left = Math.max(L, Math.min(R - cw, Math.round(pt.x - Math.cos(angle) * OFF - cw / 2))) + "px";
            chip.style.top  = Math.max(4, Math.min(H - ch - 4, Math.round(pt.y - Math.sin(angle) * OFF - ch / 2))) + "px";
        }
    }

    if (prevNodeData) _placeLocator(prevNodeData, true);
    if (nextNodeData) _placeLocator(nextNodeData, false);
}

function _clearNavLocators() {
    if (cy) cy.nodes().removeClass("nav-prev nav-next nav-current");
    const overlay = document.getElementById("nav-locators");
    if (overlay) { overlay.innerHTML = ""; overlay.style.display = "none"; }
}

// ─── bending logic ───────────────────────────────────────────────────────────

// Update has_bending on cytoscape nodes in-place (no layout re-run, no viewport change).
function _refreshBentNodeStyles() {
    if (!currentGraphData) return;
    const bentTargets = new Set();
    const macroTargets = new Set();
    _bendingBindings.forEach(b => {
        (b.nodes || [b.node]).forEach(n => bentTargets.add(n));
        if (b.bp_links && Object.keys(b.bp_links).length > 0)
            (b.nodes || [b.node]).forEach(n => macroTargets.add(n));
    });
    currentGraphData.nodes.forEach(n => {
        const isBent  = bentTargets.has(n.id)  || (n.target && bentTargets.has(n.target));
        const isMacro = macroTargets.has(n.id) || (n.target && macroTargets.has(n.target));
        n.has_bending = isBent;
        n.has_macro   = isMacro;
        const cyNode = cy.getElementById(n.id);
        if (cyNode.length) {
            cyNode.data("has_bending", isBent  ? true : null);
            cyNode.data("has_macro",   isMacro ? true : null);
        }
    });
    updateStats(currentGraphData);
    renderActivationList(currentGraphData);
}

// Fetch active bindings from server and refresh all bending UI.
function _syncBendingState(d) {
    if (d.bindings      != null) _bendingBindings = d.bindings;
    if (d.bending_params != null) _bendingParams  = d.bending_params;
}

async function _fetchBendings() {
    try {
        const r = await fetch("/api/bending/");
        if (!r.ok) return;
        const d = await r.json();
        _syncBendingState(d);
        _bendingUpdateMode  = d.update_mode || "auto";
        _syncUpdateModeBar();
        _bendingAutoThreshMs = d.auto_threshold_ms || 500;
        // Reconcile node bent/macro styling from the freshly-fetched cache.
        // No-op until the graph is rendered (guarded inside), so on reload this
        // re-applies bending visuals once both the graph and bindings are ready.
        _refreshBentNodeStyles();
        _refreshAllBendingUI();
    } catch (_) {}
}

// Fetch available callback types (cached).
async function _ensureAvailableCallbacks() {
    if (_availableCallbacks !== null) return _availableCallbacks;
    try {
        const r = await fetch("/api/bending/callbacks/");
        if (!r.ok) return [];
        const d = await r.json();
        _availableCallbacks = d.callbacks || [];
    } catch (_) {
        _availableCallbacks = [];
    }
    return _availableCallbacks;
}

// Lightweight update after a param PATCH — avoids rebuilding / resetting sliders.
// Sync slider + text-input values in-place across every rendered location for a binding.
function _syncParamSlidersInPlace(bid, paramValues) {
    // Rows with data-bid in details panel (bop-row) and pin-bending-sections (pin-bop-row),
    // and the bending pin card (pin-card-bending[data-bid]).
    document.querySelectorAll(`[data-bid="${bid}"] [data-param-name]`).forEach(slider => {
        if (slider.type !== "range") return;
        const pname = slider.dataset.paramName;
        if (paramValues[pname] == null) return;
        const v = paramValues[pname];
        if (v < parseFloat(slider.min)) slider.min = v - Math.max(Math.abs(v), 1);
        if (v > parseFloat(slider.max)) slider.max = v + Math.max(Math.abs(v), 1);
        slider.value = v;
        const inp = slider.nextElementSibling;
        if (inp && inp.type === "text") {
            inp.value = Number.isInteger(v) ? String(v) : Number(v).toFixed(4);
        }
    });
}

// Update eye button states in-place after a vis_muted toggle — no slider rebuild.
function _applyVisMutedInPlace() {
    _bendingBindings.forEach(b => {
        // details-panel bop rows
        document.querySelectorAll(`.bop-row[data-bid="${b.id}"] .bop-eye-btn`).forEach(btn => {
            btn.className = "bop-eye-btn" + (b.vis_muted ? " muted" : "");
            btn.title = b.vis_muted ? "Unmute (re-activate for viz)" : "Mute for viz";
            btn.textContent = b.vis_muted ? "○" : "●";
        });
        // pin-card bop rows
        document.querySelectorAll(`.pin-bop-row[data-bid="${b.id}"] .pin-bop-eye`).forEach(btn => {
            btn.className = "pin-bop-eye" + (b.vis_muted ? " muted" : "");
            btn.title = b.vis_muted ? "Unmute for viz" : "Mute for viz";
            btn.textContent = b.vis_muted ? "○" : "●";
        });
        // params summary in bop-row header
        document.querySelectorAll(`.bop-row[data-bid="${b.id}"] .bop-params-summary`).forEach(el => {
            el.textContent = _formatParamsSummary(b.params);
        });
    });
    // show-bent overlay
    if (_showBentMode) _applyShowBent();
    // act-modal bend view if open
    if (document.getElementById("act-mode-bend").classList.contains("active"))
        _renderBendActModalView();
}

function _postParamUpdateUI(updatedBid) {
    const lbl = document.getElementById("bending-mode-label");
    if (lbl) lbl.textContent = _bendingUpdateMode;
    // update timing notes in-place wherever they exist
    const note = _bendingLastMs != null ? `last update: ${_bendingLastMs}ms` : "";
    document.querySelectorAll(".bop-timing-note, .abl-timing-note, .pbc-timing-note")
        .forEach(el => { el.textContent = note; });
    // sync slider values across all locations for the updated binding
    if (updatedBid) {
        const b = _bendingBindings.find(x => x.id === updatedBid);
        if (b) _syncParamSlidersInPlace(updatedBid, b.params);
    }
    _refreshBendingPinCards();
}

// Rebuild every piece of bending UI that may be currently visible.
function _refreshAllBendingUI() {
    // details panel ops section
    if (currentVizNode) renderBendingOpsSection(currentVizNode);
    // act-modal bend view
    if (document.getElementById("act-mode-bend").classList.contains("active"))
        _renderBendActModalView();
    // mode indicator in details panel
    const lbl = document.getElementById("bending-mode-label");
    if (lbl) lbl.textContent = _bendingUpdateMode;
    // show-bent overlay (graph may now have new bent nodes)
    if (_showBentMode) _applyShowBent();
    // pin dashboard cards
    _refreshPinCardBendingSections();
    _refreshBendingPinCards();
    _refreshBpPinCards();
    // act-modal BP rows
    _refreshActModalBpRows();
    // a recalled snapshot is a bending: its bars and marks follow the bindings
    _syncSnapBars();
    _syncRecalledMarks();
}

// ── bending ops section (details panel) ──────────────────────────────────────

function renderBendingOpsSection(nodeData) {
    const section = document.getElementById("bending-ops-section");
    if (!section) return;

    const bendTarget = nodeData.op === "get_attr" ? nodeData.target : nodeData.id;
    const nodeBnds   = _bendingBindings.filter(b =>
        (b.nodes || [b.node]).some(n => n === bendTarget || n === nodeData.id)
    );

    const countEl = document.getElementById("bending-ops-count");
    const addBtn  = document.getElementById("bending-ops-add-btn");

    section.style.display = "";
    if (nodeBnds.length > 0) {
        countEl.textContent  = String(nodeBnds.length);
        countEl.style.display = "";
    } else {
        countEl.style.display = "none";
    }

    // add-bend button
    addBtn.onclick = () => _openBendDialog(nodeData);

    // mode indicator
    const lbl = document.getElementById("bending-mode-label");
    if (lbl) {
        lbl.textContent = _bendingUpdateMode;
        document.getElementById("bending-update-mode-indicator").onclick = _cycleBendingMode;
    }

    const list = document.getElementById("bending-ops-list");
    list.innerHTML = "";

    nodeBnds.forEach((b, i) => {
        const row = document.createElement("div");
        row.className = "bop-row";
        row.dataset.bid = b.id;

        const header = document.createElement("div");
        header.className = "bop-row-header";

        const badge = document.createElement("span");
        badge.className = "bop-type-badge";
        badge.textContent = b.callback_type;
        badge.appendChild(_bendInfoIcon(b));

        const nameInp = document.createElement("input");
        nameInp.type = "text";
        nameInp.className = "bop-name-input";
        nameInp.placeholder = "label…";
        nameInp.value = b.name || "";
        nameInp.title = "Optional label for this bending";
        nameInp.addEventListener("click", e => e.stopPropagation());
        nameInp.addEventListener("change", async () => {
            const r = await fetch(`/api/bending/${b.id}/`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: nameInp.value }),
            });
            const d = await r.json();
            if (r.ok) _syncBendingState(d);
        });

        const nodeLbl = document.createElement("span");
        nodeLbl.className = "bop-node-label";
        nodeLbl.textContent = `[${b.fn}]`;

        const summary = document.createElement("span");
        summary.className = "bop-params-summary";
        summary.textContent = _formatParamsSummary(b.params);

        const expandBtn = document.createElement("button");
        expandBtn.className = "bop-expand-btn";
        expandBtn.title = "Expand / collapse";
        expandBtn.textContent = "▸";

        const eyeBtn = document.createElement("button");
        eyeBtn.className = "bop-eye-btn" + (b.vis_muted ? " muted" : "");
        eyeBtn.title = b.vis_muted ? "Unmute (re-activate for viz)" : "Mute for viz";
        eyeBtn.textContent = b.vis_muted ? "○" : "●";
        eyeBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const live = _bendingBindings.find(x => x.id === b.id);
            const nowMuted = live ? live.vis_muted : b.vis_muted;
            const r = await fetch(`/api/bending/${b.id}/`, {
                method: "PATCH",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({vis_muted: !nowMuted}),
            });
            const d = await r.json();
            if (!r.ok) { showToast("error", d.error, d.traceback); return; }
            _syncBendingState(d);
            _applyVisMutedInPlace();
            if (hasAnyInput() && currentVizActId)
                fetchActivation(currentFn, currentVizActId);
            _refreshActivationPins(currentFn);
        });

        const pinBtn = document.createElement("button");
        pinBtn.className = "bop-pin-btn";
        pinBtn.title = "Pin this bending to dashboard";
        pinBtn.textContent = "⊕";
        pinBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            addBendingPin(b.id);
        });

        const upBtn = document.createElement("button");
        upBtn.className = "bop-move-btn";
        upBtn.title = "Move up";
        upBtn.textContent = "▲";
        upBtn.disabled = i === 0;
        upBtn.addEventListener("click", async (e) => { e.stopPropagation(); await _reorderBinding(b.id, nodeBnds, i, -1); });

        const downBtn = document.createElement("button");
        downBtn.className = "bop-move-btn";
        downBtn.title = "Move down";
        downBtn.textContent = "▼";
        downBtn.disabled = i === nodeBnds.length - 1;
        downBtn.addEventListener("click", async (e) => { e.stopPropagation(); await _reorderBinding(b.id, nodeBnds, i, +1); });

        const removeBtn = document.createElement("button");
        removeBtn.className = "bop-remove-btn";
        removeBtn.title = "Remove this bending";
        removeBtn.textContent = "✕";
        removeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            _removeBending(b.id);
        });

        header.appendChild(badge);
        header.appendChild(nameInp);
        header.appendChild(nodeLbl);
        header.appendChild(summary);
        header.appendChild(expandBtn);
        header.appendChild(eyeBtn);
        header.appendChild(pinBtn);
        header.appendChild(upBtn);
        header.appendChild(downBtn);
        header.appendChild(removeBtn);

        const body = document.createElement("div");
        body.className = "bop-body";
        _buildParamControls(body, b, "bop");

        header.addEventListener("click", (e) => {
            if (e.target === removeBtn) return;
            const open = body.classList.toggle("expanded");
            expandBtn.textContent = open ? "▾" : "▸";
            summary.style.display = open ? "none" : "";
        });

        row.appendChild(header);
        row.appendChild(body);
        list.appendChild(row);
    });
}

async function _reorderBinding(bid, nodeBnds, idx, dir) {
    const swapIdx = idx + dir;
    if (swapIdx < 0 || swapIdx >= nodeBnds.length) return;
    const swapBid = nodeBnds[swapIdx].id;
    const newOrder = _bendingBindings.map(b => b.id);
    const posA = newOrder.indexOf(bid), posB = newOrder.indexOf(swapBid);
    if (posA < 0 || posB < 0) return;
    [newOrder[posA], newOrder[posB]] = [newOrder[posB], newOrder[posA]];
    const r = await fetch("/api/bending/reorder/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ order: newOrder }),
    });
    const d = await r.json();
    if (r.ok) _syncBendingState(d);
    else showToast("error", d.error, d.traceback);
}

// ── context menu ──────────────────────────────────────────────────────────────

let _ctxMenu = null;

let _ctxInfo = null;

function _showNodeContextMenu(evt, nodeData) {
    _hideContextMenu();
    // a copy has no identity of its own — every action belongs to the original
    if (nodeData.is_alias) {
        const orig = cy.$id(nodeData.alias_of);
        if (orig.length) nodeData = orig.data();
    }
    // Select the node so details panel / currentVizData are populated for menu actions
    const cyNode = cy.getElementById(nodeData.id);
    if (cyNode && !cyNode.empty()) onNodeClick(cyNode);

    const menu = document.createElement("div");
    menu.id = "ctx-menu";
    menu.className = "ctx-menu";

    const hasBendable = ["call_function", "call_module", "call_method", "get_attr"].includes(nodeData.op);
    const fullNode = (currentGraphData && currentGraphData.nodes.find(n => n.id === nodeData.id)) || nodeData;

    // Top to bottom: pin, expand, then bend (yellow) sitting right next to
    // the mouse -- the menu is centred on it vertically, below -- then
    // show-in-list, add-alias, source underneath.
    const items = [
        { label: "⊕ pin",    action: () => _pinNodeToPage(fullNode, currentPinPage) },
        { label: "⊞ expand", action: () => openExpandModal() },
    ];
    if (hasBendable) {
        items.push({ label: "⚡ bend", action: () => _openBendDialog(fullNode), cls: "ctx-menu-item-bend" });
    }
    items.push({ label: "☰ show in list", action: () => _focusNodeInActList(nodeData.id) });
    items.push({ label: "# add alias", action: () => {
        const addBtn = document.querySelector("#detail-tags-section .detail-tag-add-btn");
        if (addBtn) addBtn.click();
    }});
    if (hasBendable) {
        items.push({ label: "{ } source", action: () => openSourceModal(currentFn, nodeData.id) });
    }
    if (fullNode.is_chain) {
        items.push({ label: "⤢ expand chain", action: () => _expandChain(fullNode.id) });
    } else if (fullNode.in_chain) {
        items.push({ label: "⤡ collapse chain", action: () => _collapseChainOf(fullNode) });
    }

    items.forEach(({ label, action, cls }) => {
        const item = document.createElement("div");
        item.className = "ctx-menu-item" + (cls ? " " + cls : "");
        item.textContent = label;
        item.addEventListener("click", () => { _hideContextMenu(); action(); });
        menu.appendChild(item);
    });

    document.body.appendChild(menu);
    _ctxMenu = menu;

    const info = _buildNodeCtxInfo(cyNode, fullNode);
    document.body.appendChild(info);
    _ctxInfo = info;

    // The menu's left edge sits at the mouse, vertically centred on it.
    const mx = evt.clientX, my = evt.clientY;
    const mw = menu.offsetWidth  || 140;
    const mh = menu.offsetHeight || 64;
    const menuLeft = Math.min(mx, window.innerWidth  - mw - 8);
    const menuTop  = Math.max(8, Math.min(my - mh / 2, window.innerHeight - mh - 8));
    menu.style.left = menuLeft + "px";
    menu.style.top  = menuTop  + "px";

    // The info card sits beside the menu -- to its right if there's room,
    // otherwise to the left of the mouse instead of overlapping the menu.
    const iw = info.offsetWidth  || 200;
    const ih = info.offsetHeight || 100;
    const gap = 6;
    let infoLeft = menuLeft + mw + gap;
    if (infoLeft + iw > window.innerWidth - 8) {
        infoLeft = Math.max(8, mx - gap - iw);
    }
    const infoTop = Math.max(8, Math.min(menuTop, window.innerHeight - ih - 8));
    info.style.left = infoLeft + "px";
    info.style.top  = infoTop  + "px";

    const dismiss = (e) => { if (!menu.contains(e.target) && !info.contains(e.target)) { _hideContextMenu(); } };
    document.addEventListener("mousedown", dismiss, { once: true });
    document.addEventListener("keydown",   (e) => { if (e.key === "Escape") _hideContextMenu(); }, { once: true });
}

function _hideContextMenu() {
    if (_ctxMenu)  { _ctxMenu.remove();  _ctxMenu  = null; }
    if (_ctxInfo)  { _ctxInfo.remove();  _ctxInfo  = null; }
}

//: The shape/op/target/etc info card shown beside the context menu. Read-only
//: -- it exists so the menu doesn't have to be opened just to look something
//: up, and it closes with it.
function _buildNodeCtxInfo(cyNode, d) {
    const info = document.createElement("div");
    info.className = "node-ctx-info";

    const title = document.createElement("div");
    title.className = "node-ctx-info-name";
    title.textContent = d.label || d.id;
    info.appendChild(title);

    const shapeOf = (nd) => (nd.shape && nd.shape.length) ? `[${nd.shape.join("×")}]` : "—";

    function row(label, value) {
        if (value === null || value === undefined || value === "") return;
        const r = document.createElement("div");
        r.className = "node-ctx-info-row";
        const l = document.createElement("span");
        l.className = "node-ctx-info-label";
        l.textContent = label;
        const v = document.createElement("span");
        v.className = "node-ctx-info-value";
        v.textContent = value;
        r.appendChild(l);
        r.appendChild(v);
        info.appendChild(r);
    }

    row("shape", shapeOf(d));
    row("op", d.op);
    row("target", d.target || "—");

    function ioSection(heading, cyNodes) {
        if (!cyNodes || !cyNodes.length) return;
        const section = document.createElement("div");
        section.className = "node-ctx-info-section";
        const hdr = document.createElement("div");
        hdr.className = "node-ctx-info-label";
        hdr.textContent = heading;
        section.appendChild(hdr);
        cyNodes.forEach(cn => {
            const nd = cn.data();
            const line = document.createElement("div");
            line.className = "node-ctx-info-io";
            const name = document.createElement("span");
            name.textContent = nd.label || nd.id;
            const shape = document.createElement("span");
            shape.textContent = shapeOf(nd);
            line.appendChild(name);
            line.appendChild(shape);
            section.appendChild(line);
        });
        info.appendChild(section);
    }

    if (cyNode && !cyNode.empty()) {
        ioSection("inputs", cyNode.incomers("node"));
        ioSection("outputs", cyNode.outgoers("node"));
    }

    const memberLists = Object.values(_nodeLists).filter(l => l.nodes.includes(d.label || d.id));
    if (memberLists.length) {
        const section = document.createElement("div");
        section.className = "node-ctx-info-section";
        const hdr = document.createElement("div");
        hdr.className = "node-ctx-info-label";
        hdr.textContent = "lists";
        section.appendChild(hdr);
        memberLists.forEach(l => {
            const line = document.createElement("div");
            line.className = "node-ctx-info-io";
            line.textContent = l.name;
            section.appendChild(line);
        });
        info.appendChild(section);
    }

    return info;
}

let _applyExistingMenu = null;
function _showApplyExistingMenu(triggerEvt, nodeData) {
    if (_applyExistingMenu) { _applyExistingMenu.remove(); _applyExistingMenu = null; }

    const isWeight   = nodeData.op === "get_attr";
    const bendTarget = isWeight ? (nodeData.target || nodeData.id) : nodeData.id;

    const menu = document.createElement("div");
    menu.className = "ctx-menu";
    menu.style.zIndex = "3100";

    _bendingBindings.forEach(b => {
        const item = document.createElement("div");
        item.className = "ctx-menu-item";
        item.textContent = (b.name ? `${b.name} — ` : "") + `${b.callback_type} @ ${b.node}`;
        item.addEventListener("click", async () => {
            menu.remove(); _applyExistingMenu = null;
            try {
                const r = await fetch("/api/bending/", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        fn: currentFn, node: bendTarget,
                        callback_type: b.callback_type, params: {},
                        existing_id: b.id,
                    }),
                });
                const d = await r.json();
                if (!r.ok) { const _e = new Error(d.error || r.statusText); _e.traceback = d.traceback || ""; throw _e; }
                _syncBendingState(d);
                _refreshBentNodeStyles();
                _refreshAllBendingUI();
                showToast("ok", `Applied "${b.callback_type}" to ${bendTarget}`);
            } catch (err) { showToast("error", err.message, err.traceback, err.traceback); }
        });
        menu.appendChild(item);
    });

    document.body.appendChild(menu);
    _applyExistingMenu = menu;

    const mx = triggerEvt.clientX || 0, my = triggerEvt.clientY || 0;
    menu.style.left = mx + "px"; menu.style.top = my + "px";
    requestAnimationFrame(() => {
        const r = menu.getBoundingClientRect();
        if (r.right  > window.innerWidth)  menu.style.left = (mx - r.width)  + "px";
        if (r.bottom > window.innerHeight) menu.style.top  = (my - r.height) + "px";
    });
    const dismiss = (e) => { if (!menu.contains(e.target)) { menu.remove(); _applyExistingMenu = null; } };
    setTimeout(() => document.addEventListener("mousedown", dismiss, { once: true }), 0);
}

//: Open the activation pane on the current graph, with `nodeId` selected and
//: scrolled into view -- the same selection state a normal click on its row
//: would leave, just reached from the node itself (the context menu's
//: "show in list") rather than by finding the row yourself.
function _focusNodeInActList(nodeId) {
    const data = _actModalData || currentGraphData;
    if (!data) return;
    openActModal(data);
    _setActModalMode("list");
    _actSelection.clear();
    _actSelection.add(nodeId);
    _actLastSelLabel = nodeId;
    _updateSelectionBar();
    requestAnimationFrame(() => {
        const row = document.querySelector(
            `#act-modal-list .act-modal-row[data-name="${CSS.escape(nodeId)}"]`);
        if (row) {
            row.classList.add("act-selected");
            row.scrollIntoView({ block: "center" });
        }
    });
}

// ── bend dialog ───────────────────────────────────────────────────────────────

async function _openBendDialog(nodeData) {
    _bendDialogNode = nodeData;
    const cbs = await _ensureAvailableCallbacks();
    if (!cbs.length) { showToast("error", "No bending callbacks available"); return; }

    // Filter by compatibility
    const isWeight   = nodeData.op === "get_attr";
    const compatible = cbs.filter(cb =>
        isWeight ? cb.weight_compatible : cb.activation_compatible
    );
    if (!compatible.length) {
        showToast("warn", "No compatible callbacks for this node type");
        return;
    }

    const modal    = document.getElementById("bend-modal");
    const nodeName = document.getElementById("bend-modal-node-name");
    const cbSelect = document.getElementById("bend-cb-select");   // hidden
    const paramsEl = document.getElementById("bend-params-section");
    const existChk = document.getElementById("bend-add-to-existing"); // hidden
    const existSel = document.getElementById("bend-existing-select"); // hidden

    nodeName.textContent = nodeData.id || nodeData.label;

    // Sync hidden select for _applyBendDialog compatibility
    cbSelect.innerHTML = "";
    compatible.forEach(cb => {
        const opt = document.createElement("option");
        opt.value = cb.name; opt.textContent = cb.name;
        cbSelect.appendChild(opt);
    });
    existChk.checked = false;
    existSel.innerHTML = "";

    function _makeBendParamRow(name, pd, isExtra, allowBpLink) {
        const row = document.createElement("div");
        row.className = "bend-param-row" + (isExtra ? " bend-extra-row" : "");
        const lbl = document.createElement("span");
        lbl.className = "bend-param-label";
        lbl.textContent = (pd.label || name) + (isExtra && pd.required ? " *" : "");
        row.appendChild(lbl);
        const isInt = pd.type === "int";
        const isNullable = isExtra && !pd.required && pd.default === null;
        if (pd.type === "ints") {
            // a list of axes, typed as "1, 2"; empty keeps the constructor's default
            const inp = document.createElement("input");
            inp.type        = "text";
            inp.className   = "bend-param-input";
            inp.dataset.name = name;
            inp.dataset.kind = "ints";
            inp.value       = Array.isArray(pd.default) ? pd.default.join(", ")
                            : (pd.default != null ? String(pd.default) : "");
            inp.placeholder = "e.g. 1, 2";
            if (pd.description) inp.title = pd.description;
            row.appendChild(document.createElement("span"));
            row.appendChild(inp);
        } else if (isNullable && isInt) {
            // Optional int with no default: show a plain number input, empty = None
            const inp = document.createElement("input");
            inp.type        = "number";
            inp.step        = "1";
            inp.className   = "bend-param-input";
            inp.dataset.name = name;
            inp.placeholder = "none";
            inp.value       = "";
            row.appendChild(document.createElement("span"));
            row.appendChild(inp);
        } else if (pd.widget === "slider" || pd.widget === "number" || isInt ||
            pd.type === "float") {
            const [rmin, rmax] = pd.range || [null, null];
            const defVal     = pd.default != null ? pd.default : 0;
            const rangeBasis = Math.abs(defVal);
            const fallback   = Math.max(rangeBasis * 3, 2);
            const slider = document.createElement("input");
            slider.type  = "range";
            slider.className = "bend-param-slider";
            slider.dataset.name = name;
            slider.min   = rmin != null ? rmin : -fallback;
            slider.max   = rmax != null ? rmax :  fallback;
            slider.step  = isInt ? 1 : "any";
            slider.value = defVal;
            const inp = document.createElement("input");
            inp.type  = "text";
            inp.className = "bend-param-input";
            inp.dataset.name = name;
            inp.value = isInt ? String(Math.round(defVal)) : Number(defVal).toFixed(4);
            slider.addEventListener("input", () => {
                const v = isInt ? Math.round(parseFloat(slider.value)) : parseFloat(slider.value);
                inp.value = isInt ? String(v) : Number(v).toFixed(4);
            });
            inp.addEventListener("change", () => {
                const v = isInt ? Math.round(parseFloat(inp.value)) : parseFloat(inp.value);
                if (!isNaN(v)) {
                    if (v < parseFloat(slider.min)) slider.min = v - Math.max(Math.abs(v), 1);
                    if (v > parseFloat(slider.max)) slider.max = v + Math.max(Math.abs(v), 1);
                    slider.value = v;
                    inp.value = isInt ? String(v) : Number(v).toFixed(4);
                }
            });
            row.appendChild(slider);
            row.appendChild(inp);
        } else if (pd.widget === "toggle") {
            const chk = document.createElement("input");
            chk.type  = "checkbox";
            chk.className = "bend-param-toggle";
            chk.dataset.name = name;
            chk.checked = !!pd.default;
            row.appendChild(chk);
            row.appendChild(document.createElement("span"));
        } else if (pd.choices) {
            const sel = document.createElement("select");
            sel.className = "bend-param-select";
            sel.dataset.name = name;
            pd.choices.forEach(c => {
                const opt = document.createElement("option");
                opt.value = c; opt.textContent = c;
                if (c === pd.default) opt.selected = true;
                sel.appendChild(opt);
            });
            row.appendChild(document.createElement("span"));
            row.appendChild(sel);
        } else {
            const inp = document.createElement("input");
            inp.type  = "text";
            inp.className = "bend-param-input";
            inp.dataset.name = name;
            inp.value = pd.default != null ? pd.default : "";
            row.appendChild(document.createElement("span"));
            row.appendChild(inp);
        }
        // BP link dropdown — shown when BPs exist or always (has "+ new macro…")
        if (allowBpLink) {
            const bpSel = document.createElement("select");
            bpSel.className = "bend-param-bp-select";
            bpSel.dataset.bpLinkFor = name;
            bpSel.title = "Link to BendingParameter (macro)";

            function _rebuildBpOptions(selectVal) {
                bpSel.innerHTML = "";
                const noneOpt = document.createElement("option");
                noneOpt.value = ""; noneOpt.textContent = "— macro";
                bpSel.appendChild(noneOpt);
                // The callback refuses a macro whose type doesn't match this param,
                // so show each macro's type and flag the ones that won't link.
                const want = pd.type || null;
                _bendingParams.forEach(bp => {
                    const t = _bpType(bp);
                    const bad = want && want !== t;
                    const opt = document.createElement("option");
                    opt.value = bp.name;
                    opt.textContent = `${bp.name} (${t})` + (bad ? " ⚠" : "");
                    if (bad) opt.title = `macro is ${t}, but '${name}' expects ${want}`;
                    bpSel.appendChild(opt);
                });
                const newOpt = document.createElement("option");
                newOpt.value = "__new__"; newOpt.textContent = "+ new macro…";
                bpSel.appendChild(newOpt);
                if (selectVal != null) bpSel.value = selectVal;
            }
            _rebuildBpOptions(null);

            bpSel.addEventListener("change", () => {
                if (bpSel.value !== "__new__") {
                    bpSel.classList.toggle("linked", !!bpSel.value);
                    return;
                }
                bpSel.value = "";   // reset to "— macro" while panel is open
                // Read current slider value and param range for pre-filling
                const slider = row.querySelector(".bend-param-slider");
                const curVal = slider ? parseFloat(slider.value) : (pd.default != null ? Number(pd.default) : 0);
                const rng = pd.range || [null, null];
                _showBpCreatePanel(bpSel, name, curVal, rng, (bpName) => {
                    _rebuildBpOptions(bpName);
                    bpSel.classList.add("linked");
                }, pd.type || "float");
            });
            row.appendChild(bpSel);
        }
        // Per-param description hint
        if (pd.description) {
            const hint = document.createElement("div");
            hint.className = "bend-param-desc";
            hint.textContent = pd.description;
            const wrap = document.createElement("div");
            wrap.className = "bend-param-wrap";
            wrap.appendChild(row);
            wrap.appendChild(hint);
            return wrap;
        }
        return row;
    }

    function renderParams() {
        const cbName = cbSelect.value;
        const cbDesc = compatible.find(c => c.name === cbName) || {};

        // ── description / compat header ───────────────────────────────────────
        const descEl = document.getElementById("bend-cb-desc");
        if (descEl) {
            descEl.innerHTML = "";
            if (cbDesc.description) {
                const txt = document.createElement("p");
                txt.className = "bend-cb-desc-text";
                txt.textContent = cbDesc.description;
                descEl.appendChild(txt);
            }
            const badges = document.createElement("div");
            badges.className = "bend-cb-badges";
            if (cbDesc.weight_compatible) {
                const b = document.createElement("span");
                b.className = "bend-cb-badge bend-cb-badge-weight";
                b.textContent = "weights";
                badges.appendChild(b);
            }
            if (cbDesc.activation_compatible) {
                const b = document.createElement("span");
                b.className = "bend-cb-badge bend-cb-badge-act";
                b.textContent = "activations";
                badges.appendChild(b);
            }
            if (cbDesc.jit_compatible) {
                const b = document.createElement("span");
                b.className = "bend-cb-badge bend-cb-badge-jit";
                b.textContent = "TorchScript";
                badges.appendChild(b);
            }
            if (badges.children.length) descEl.appendChild(badges);
        }

        // ── parameters ────────────────────────────────────────────────────────
        paramsEl.innerHTML = "";
        const paramEntries = Object.entries(cbDesc.params || {});
        if (paramEntries.length) {
            const hdr = document.createElement("div");
            hdr.className = "bend-section-title";
            hdr.textContent = "Parameters";
            paramsEl.appendChild(hdr);
            paramEntries.forEach(([name, pd]) => {
                paramsEl.appendChild(_makeBendParamRow(name, pd, false, true));
            });
        }
        const extras = cbDesc.extra_init_params || {};
        if (Object.keys(extras).length) {
            const sep = document.createElement("div");
            sep.className = "bend-extra-sep";
            sep.textContent = "init options";
            paramsEl.appendChild(sep);
            Object.entries(extras).forEach(([name, spec]) => {
                paramsEl.appendChild(_makeBendParamRow(name, spec, true, false));
            });
        }
    }
    // ── Build callback list (left column of "New" pane) ───────────────────────
    const cbListEl = document.getElementById("bend-cb-list");
    cbListEl.innerHTML = "";
    compatible.forEach((cb, i) => {
        const row = document.createElement("div");
        row.className = "bend-cb-list-row" + (i === 0 ? " active" : "");
        row.dataset.cbName = cb.name;

        const badge = document.createElement("span");
        badge.className = "bend-cb-list-badge";
        badge.textContent = cb.name;

        row.appendChild(badge);
        row.addEventListener("click", () => {
            cbListEl.querySelectorAll(".bend-cb-list-row").forEach(r => r.classList.remove("active"));
            row.classList.add("active");
            cbSelect.value = cb.name;
            renderParams();
        });
        cbListEl.appendChild(row);
    });

    // Select first callback
    if (compatible.length > 0) {
        cbSelect.value = compatible[0].name;
        renderParams();
    }

    // ── Build "Bend with…" pane ───────────────────────────────────────────────
    function buildExistingPane() {
        const list = document.getElementById("bend-existing-pane-list");
        list.innerHTML = "";
        if (!_bendingBindings.length) {
            list.innerHTML = '<div class="bend-existing-empty">No active bindings yet. Create one first.</div>';
            return;
        }
        _bendingBindings.forEach(b => {
            const row = document.createElement("div");
            row.className = "bend-existing-row";

            const badge = document.createElement("span");
            badge.className = "bop-type-badge";
            badge.textContent = b.callback_type;

            const info = document.createElement("div");
            info.className = "bend-existing-info";
            const nameLine = document.createElement("div");
            nameLine.className = "bend-existing-name";
            nameLine.textContent = (b.name ? `${b.name} — ` : "") + b.node;
            const paramsLine = document.createElement("div");
            paramsLine.className = "bend-existing-params";
            paramsLine.textContent = Object.entries(b.params || {})
                .map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(3) : v}`)
                .join("  ·  ");

            info.appendChild(nameLine);
            info.appendChild(paramsLine);
            row.appendChild(badge);
            row.appendChild(info);

            row.addEventListener("click", async () => {
                existChk.checked = true;
                existSel.innerHTML = "";
                const opt = document.createElement("option");
                opt.value = b.id; opt.selected = true;
                existSel.appendChild(opt);
                await _applyBendDialog();
            });
            list.appendChild(row);
        });
    }

    // ── Tab switching ─────────────────────────────────────────────────────────
    const tabBtns = modal.querySelectorAll(".bend-modal-tab");
    tabBtns.forEach(btn => {
        btn.onclick = () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const tab = btn.dataset.tab;
            modal.querySelectorAll(".bend-pane").forEach(p => {
                p.style.display = p.id === `bend-pane-${tab}` ? "" : "none";
            });
            if (tab === "existing") buildExistingPane();
        };
    });
    // Reset to "New" tab
    modal.querySelectorAll(".bend-pane").forEach(p => {
        p.style.display = p.id === "bend-pane-new" ? "" : "none";
    });
    tabBtns.forEach(b => b.classList.toggle("active", b.dataset.tab === "new"));

    document.getElementById("bend-modal-cancel").onclick = _closeBendDialog;
    document.getElementById("bend-modal-backdrop").onclick = _closeBendDialog;
    document.getElementById("bend-modal-close").onclick   = _closeBendDialog;
    document.getElementById("bend-modal-apply").onclick   = _applyBendDialog;

    modal.style.display = "flex";
}

function _closeBendDialog() {
    document.getElementById("bend-modal").style.display = "none";
    _bendDialogNode = null;
}

async function _applyBendDialog() {
    if (!_bendDialogNode) return;
    const nodeData   = _bendDialogNode;
    const isWeight   = nodeData.op === "get_attr";
    const bendTarget = isWeight ? nodeData.target : nodeData.id;
    const cbType     = document.getElementById("bend-cb-select").value;
    const existChk   = document.getElementById("bend-add-to-existing");
    const existSel   = document.getElementById("bend-existing-select");

    // Gather params from dialog controls — type-aware
    const cbDesc = (_availableCallbacks || []).find(c => c.name === cbType) || {};
    const allParamDescs = Object.assign({}, cbDesc.params || {}, cbDesc.extra_init_params || {});
    const params = {};
    document.querySelectorAll("#bend-params-section [data-name]").forEach(el => {
        const name = el.dataset.name;
        const pd   = allParamDescs[name] || {};
        const isInt = pd.type === "int";
        if (el.dataset.kind === "ints") {
            const axes = el.value.split(/[\s,]+/).filter(Boolean).map(Number);
            if (axes.length && axes.every(Number.isFinite)) params[name] = axes.map(Math.round);
            // empty (or unreadable): omitted, so the constructor's default applies
        } else if (el.type === "checkbox") {
            params[name] = el.checked ? 1 : 0;
        } else if (el.tagName === "SELECT") {
            params[name] = el.value;
        } else if (el.type === "number" || el.type === "range" || el.type === "text") {
            const raw = parseFloat(el.value);
            if (isNaN(raw) || el.value.trim() === "") {
                // Empty optional extra init param → omit (Python uses None default)
                const spec = (cbDesc.extra_init_params || {})[name];
                if (!spec || spec.required) params[name] = 0; // required: fall back to 0
                // else: skip entirely → Python constructor uses its default (None)
            } else {
                params[name] = isInt ? Math.round(raw) : raw;
            }
        }
    });

    const body = {
        fn:            currentFn,
        node:          bendTarget,
        callback_type: cbType,
        params,
        existing_id:   (existChk && existChk.checked && existSel) ? existSel.value : null,
    };

    const applyBtn = document.getElementById("bend-modal-apply");
    applyBtn.disabled = true;
    try {
        const r = await fetch("/api/bending/", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body:    JSON.stringify(body),
        });
        const d = await r.json();
        if (!r.ok) { const _e = new Error(d.error || r.statusText); _e.traceback = d.traceback || ""; throw _e; }
        _syncBendingState(d);
        _bendingUpdateMode = d.update_mode || _bendingUpdateMode;
        _syncUpdateModeBar();

        // Link params to BendingParameters if selected in the dialog.
        // Only applies when creating a NEW binding (not when reusing an existing one).
        const newBid = d.id;
        const usingExisting = existChk && existChk.checked;
        if (newBid && !usingExisting) {
            const linkPairs = [];
            document.querySelectorAll("#bend-params-section .bend-param-bp-select").forEach(sel => {
                if (!sel.value) return;
                const paramName = sel.getAttribute("data-bp-link-for");
                if (paramName) linkPairs.push({ paramName, bpName: sel.value });
            });
            for (const { paramName, bpName } of linkPairs) {
                try {
                    const r2 = await fetch(`/api/bending/${encodeURIComponent(newBid)}/link/`, {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({ param_name: paramName, bp_name: bpName }),
                    });
                    const d2 = await r2.json();
                    if (!r2.ok) showToast("warn", `Could not link ${paramName} → ${bpName}: ${d2.error}`);
                    else _syncBendingState(d2);
                } catch (err) { showToast("warn", `Link failed: ${err.message}`); }
            }
        }

        _closeBendDialog();
        _refreshBentNodeStyles();
        _refreshAllBendingUI();
        await _refreshLiveViews();
        showToast("ok", `${cbType} applied to ${bendTarget}`);
    } catch (err) {
        showToast("error", "Bend failed: " + err.message, err.traceback);
    } finally {
        applyBtn.disabled = false;
    }
}

// ── manual mode: one queue, one Apply ────────────────────────────────────────
// In manual mode nothing recomputes until asked. Every held change — a bending
// slider or toggle, a macro — waits here, the latest value per parameter, and
// the one Apply button in the top bar sends them all and recomputes once. One
// button rather than one per bending: applying a single bending left the others'
// sliders showing values the graph was not using, and macros had no button at
// all (their held update was dropped).
const _manualPending = new Map();   // key → held update (async fn), latest wins

function _holdManual(key, doIt) {
    _manualPending.set(key, doIt);
    _syncApplyAllBtn();
}

function _dropManual(prefix) {
    for (const k of [..._manualPending.keys()]) if (k.startsWith(prefix)) _manualPending.delete(k);
    _syncApplyAllBtn();
}

function _syncApplyAllBtn() {
    const btn = document.getElementById("apply-all-btn");
    if (!btn) return;
    const n = _manualPending.size;
    btn.style.display = (_bendingUpdateMode === "manual" && n) ? "" : "none";
    btn.textContent = `\u27F3 apply ${n} change${n === 1 ? "" : "s"}`;
}

async function _applyAllPending() {
    if (!_manualPending.size) return;
    const btn = document.getElementById("apply-all-btn");
    const jobs = [..._manualPending.values()];
    _manualPending.clear();
    if (btn) { btn.disabled = true; btn.textContent = "\u23F3 applying…"; }
    try {
        // in order: each one syncs the session state the next one builds on
        for (const job of jobs) await job();
        if (hasAnyInput()) _bendingLastMs = await _refreshLiveViews();
    } finally {
        if (btn) btn.disabled = false;
        _syncApplyAllBtn();
    }
}

// ── param update ──────────────────────────────────────────────────────────────

function _scheduleBendingParamUpdate(bid, params, isActivation) {
    clearTimeout(_bendingDebounceTimer);
    const doIt = async () => {
        try {
            const r = await fetch(`/api/bending/${bid}/`, {
                method:  "PATCH",
                headers: {"Content-Type": "application/json"},
                body:    JSON.stringify(params),
            });
            const d = await r.json();
            if (!r.ok) {
                showToast("error", d.error || "Invalid value");
                _postParamUpdateUI(bid);   // revert controls to last-good value
                return;
            }
            _syncBendingState(d);
            _postParamUpdateUI(bid);

            // Refresh unless in manual mode
            if (_bendingUpdateMode !== "manual") {
                _bendingLastMs = await _refreshLiveViews();
                if (_bendingUpdateMode === "auto" && _bendingLastMs >= _bendingAutoThreshMs)
                    showToast("warn", `Slow update (${_bendingLastMs}ms) — switch to manual mode if needed`);
            }
        } catch (err) {
            if (_isAbort(err)) return;   // superseded by a newer value
            showToast("error", "Param update failed: " + err.message, err.traceback);
        }
    };

    if (_bendingUpdateMode === "manual") {
        // held until the global Apply (see _applyAllPending)
        _holdManual(`b:${bid}:${Object.keys(params).join(",")}`, doIt);
        return null;
    }
    _bendingDebounceTimer = setTimeout(doIt, _BEND_DEBOUNCE_MS);
    return null;
}

function _scheduleBpUpdate(bpName, value) {
    clearTimeout(_bendingDebounceTimer);
    const doIt = async () => {
        try {
            const r = await fetch(`/api/bending_params/${encodeURIComponent(bpName)}/`, {
                method: "PATCH",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ value }),
            });
            const d = await r.json();
            if (!r.ok) { showToast("error", d.error, d.traceback); return; }
            _syncBendingState(d);
            // sync param sliders for every binding whose params are driven by this BP
            _bendingBindings.forEach(b => {
                if (b.bp_links && Object.values(b.bp_links).includes(bpName))
                    _syncParamSlidersInPlace(b.id, b.params);
            });
            _postParamUpdateUI(null);
            if (_bendingUpdateMode !== "manual") {
                _bendingLastMs = await _refreshLiveViews();
                if (_bendingUpdateMode === "auto" && _bendingLastMs >= _bendingAutoThreshMs)
                    showToast("warn", `Slow update (${_bendingLastMs}ms) — switch to manual mode if needed`);
            }
        } catch (err) {
            if (_isAbort(err)) return;   // superseded by a newer macro value
            showToast("error", err.message, err.traceback, err.traceback);
        }
    };
    if (_bendingUpdateMode === "manual") {
        _holdManual(`m:${bpName}`, doIt);   // held until the global Apply
        return null;
    }
    _bendingDebounceTimer = setTimeout(doIt, _BEND_DEBOUNCE_MS);
    return null;
}

async function _removeBending(bid) {
    try {
        _dropManual(`b:${bid}:`);
        const r = await fetch(`/api/bending/${bid}/`, { method: "DELETE" });
        const d = await r.json();
        if (!r.ok) { showToast("error", d.error, d.traceback); return; }
        _syncBendingState(d);
        _refreshBentNodeStyles();
        _refreshAllBendingUI();
        await _refreshLiveViews();
        showToast("ok", "Bending removed");
    } catch (err) {
        showToast("error", "Remove failed: " + err.message, err.traceback);
    }
}

async function _setBendingMode(mode) {
    // leaving manual mode applies what was held -- first, while still in
    // manual mode, so the held changes do not each refresh the views on top of
    // the batch's own single refresh
    if (_bendingUpdateMode === "manual" && mode !== "manual" && _manualPending.size > 0)
        await _applyAllPending();
    try {
        const r = await fetch("/api/bending/mode/", {
            method:  "PATCH",
            headers: {"Content-Type": "application/json"},
            body:    JSON.stringify({ update_mode: mode }),
        });
        if (!r.ok) return;
        const d = await r.json();
        _bendingUpdateMode  = d.update_mode;
        _syncUpdateModeBar();
        _bendingAutoThreshMs = d.auto_threshold_ms || _bendingAutoThreshMs;
        _refreshAllBendingUI();
        _syncApplyAllBtn();
    } catch (_) {}
}

// The update mode, in the top bar: when a bending or macro change recomputes.
const _UPDATE_MODE_HELP = {
    auto:   "auto — recompute on every change while that stays fast; past the time threshold, only once a change settles",
    live:   "live — recompute on every change, however long it takes",
    manual: "manual — hold every change until you apply them, all at once",
};
function _syncUpdateModeBar() {
    document.querySelectorAll("#update-mode-bar .update-mode-btn").forEach(b => {
        const on = b.dataset.mode === _bendingUpdateMode;
        b.classList.toggle("active", on);
        b.title = _UPDATE_MODE_HELP[b.dataset.mode] + (on ? "  (current)" : "");
    });
    const lbl = document.getElementById("bending-mode-label");
    if (lbl) lbl.textContent = _bendingUpdateMode;
}

function _cycleBendingMode() {
    const next = { auto: "live", live: "manual", manual: "auto" }[_bendingUpdateMode] || "auto";
    _setBendingMode(next);
}

// ── parameter controls builder (shared by details-panel and act-modal) ────────

function _snapToChoices(v, choices) {
    if (!choices || !choices.length) return v;
    return choices.reduce((best, c) => Math.abs(c - v) < Math.abs(best - v) ? c : best, choices[0]);
}

function _buildParamControls(container, binding, prefix) {
    const desc = binding.descriptor || {};
    const ps   = desc.params || {};
    const isActivation = !((desc.weight_compatible) && !(desc.activation_compatible));

    Object.entries(ps).forEach(([name, pd]) => {
        // Skip invisible params
        if (pd.visible === false) return;

        const isLinked = !!(binding.bp_links && binding.bp_links[name]);
        const currentVal = binding.params[name];
        const row = document.createElement("div");
        row.className = `${prefix}-param-row`;

        const lbl = document.createElement("span");
        lbl.className = `${prefix}-param-label`;
        lbl.textContent = pd.label || name;
        row.appendChild(lbl);

        const choices = pd.choices && pd.choices.length ? pd.choices : null;
        const isInt   = pd.type === "int";
        const initVal = currentVal != null ? currentVal : (pd.default != null ? pd.default : 0);
        const _fmtVal = v => isInt ? String(Math.round(v)) : Number(v).toFixed(4);

        const _dispatch = (v) => {
            _scheduleBendingParamUpdate(binding.id, {[name]: v}, isActivation);
        };

        if (pd.widget === "toggle") {
            const chk = document.createElement("input");
            chk.type    = "checkbox";
            chk.checked = initVal ? !!initVal : false;
            chk.addEventListener("change", () => {
                // held like any other change in manual mode
                _scheduleBendingParamUpdate(binding.id, {[name]: chk.checked ? 1 : 0}, isActivation);
            });
            row.appendChild(chk);
            row.appendChild(document.createElement("span"));

        } else if (pd.widget === "select" && choices) {
            const sel = document.createElement("select");
            sel.className = `${prefix}-param-select`;
            sel.dataset.paramName = name;
            choices.forEach(c => {
                const opt = document.createElement("option");
                opt.value = c; opt.textContent = c;
                if (c == initVal) opt.selected = true;
                sel.appendChild(opt);
            });
            sel.addEventListener("change", () => _dispatch(isInt ? parseInt(sel.value) : parseFloat(sel.value)));
            row.appendChild(sel);
            row.appendChild(document.createElement("span"));

        } else if (pd.widget === "field") {
            const inp = document.createElement("input");
            inp.type  = isInt ? "number" : "text";
            inp.className = `${prefix}-param-input`;
            inp.dataset.paramName = name;
            inp.value = _fmtVal(initVal);
            if (isInt) { inp.step = "1"; }
            if (pd.placeholder) inp.placeholder = pd.placeholder;
            inp.addEventListener("change", () => {
                let v = isInt ? parseInt(inp.value) : parseFloat(inp.value);
                if (isNaN(v)) return;
                if (choices) v = _snapToChoices(v, choices);
                inp.value = _fmtVal(v);
                _dispatch(v);
            });
            row.appendChild(inp);
            row.appendChild(document.createElement("span"));

        } else {
            // slider (default) or int
            const [rmin, rmax] = pd.range || [null, null];
            const rangeBasis = pd.default != null ? Math.abs(pd.default) : Math.abs(initVal);
            const fallback   = Math.max(rangeBasis * 3, 2);

            const slider = document.createElement("input");
            slider.type  = "range";
            slider.className = `${prefix}-param-slider`;
            slider.dataset.paramName = name;
            slider.min   = rmin != null ? rmin : (choices ? Math.min(...choices) : -fallback);
            slider.max   = rmax != null ? rmax : (choices ? Math.max(...choices) :  fallback);
            slider.step  = pd.step != null ? pd.step : (isInt ? 1 : "any");
            slider.value = initVal;

            const inp = document.createElement("input");
            inp.type  = "text";
            inp.className = `${prefix}-param-input`;
            inp.value = _fmtVal(initVal);

            const _readSlider = () => {
                let v = isInt ? Math.round(parseFloat(slider.value)) : parseFloat(slider.value);
                if (choices) v = _snapToChoices(v, choices);
                return v;
            };

            slider.addEventListener("input", () => {
                const v = _readSlider();
                slider.value = v;   // snap back if choices
                inp.value = _fmtVal(v);
                _dispatch(v);
            });
            inp.addEventListener("change", () => {
                let v = isInt ? Math.round(parseFloat(inp.value)) : parseFloat(inp.value);
                if (isNaN(v)) return;
                if (choices) {
                    v = _snapToChoices(v, choices);
                } else {
                    if (v < parseFloat(slider.min)) slider.min = v - Math.max(Math.abs(v), 1);
                    if (v > parseFloat(slider.max)) slider.max = v + Math.max(Math.abs(v), 1);
                }
                slider.value = v;
                inp.value = _fmtVal(v);
                _dispatch(v);
            });

            row.appendChild(slider);
            row.appendChild(inp);
        }

        // BP link / unlink indicator (added after all control elements)
        if (isLinked) {
            row.querySelectorAll("input, select").forEach(el => { el.disabled = true; });
            const bpName = binding.bp_links[name];
            const badge = document.createElement("span");
            badge.className = `${prefix}-bp-badge`;
            badge.title = `Linked to BendingParameter "${bpName}"`;
            badge.textContent = `⊕ ${bpName}`;

            // The macro reads 0…1; this is the range *this attachment* maps it
            // onto. Editing it re-links, which swaps in new arithmetic — other
            // params driven by the same macro keep their own ranges.
            const mapped = (binding.bp_maps || {})[name];
            const mapRow = document.createElement("span");
            mapRow.className = `${prefix}-bp-map`;
            if (mapped) {
                const mk = (idx, ph) => {
                    const i = document.createElement("input");
                    i.type = "text"; i.className = `${prefix}-bp-map-inp`;
                    i.value = mapped[idx]; i.placeholder = ph;
                    i.title = idx === 0 ? "value the macro sends at 0"
                                        : "value the macro sends at 1";
                    return i;
                };
                const loInp = mk(0, "at 0"), hiInp = mk(1, "at 1");
                const commit = async () => {
                    const lo = parseFloat(loInp.value), hi = parseFloat(hiInp.value);
                    if (!isFinite(lo) || !isFinite(hi) || lo === hi) {
                        loInp.value = mapped[0]; hiInp.value = mapped[1];
                        return;
                    }
                    try { await _linkBpParam(binding, name, bpName, [lo, hi]); }
                    catch (err) { showToast("error", err.message, err.traceback, err.traceback); }
                };
                loInp.addEventListener("change", commit);
                hiInp.addEventListener("change", commit);
                mapRow.appendChild(document.createTextNode("0…1→"));
                mapRow.appendChild(loInp);
                mapRow.appendChild(hiInp);
            }
            const unlinkBtn = document.createElement("button");
            unlinkBtn.className = `${prefix}-bp-unlink-btn`;
            unlinkBtn.title = "Unlink from BendingParameter";
            unlinkBtn.textContent = "×";
            unlinkBtn.addEventListener("click", async (e) => {
                e.stopPropagation();
                await _unlinkBpParam(binding, name);
            });
            const pinBpBtn = document.createElement("button");
            pinBpBtn.className = `${prefix}-bp-link-btn`;
            pinBpBtn.title = `Pin BendingParameter "${bpName}" to dashboard`;
            pinBpBtn.textContent = "⊞";
            pinBpBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                addBendingParamPin(bpName);
            });
            row.appendChild(badge);
            if (mapRow.childNodes.length) row.appendChild(mapRow);
            row.appendChild(unlinkBtn);
            row.appendChild(pinBpBtn);
        } else {
            const linkBtn = document.createElement("button");
            linkBtn.className = `${prefix}-bp-link-btn`;
            linkBtn.title = "Promote to BendingParameter";
            linkBtn.textContent = "⊕";
            linkBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                _openBpLinkDialog(binding, name, currentVal);
            });
            row.appendChild(linkBtn);
        }

        container.appendChild(row);
    });


    // Timing note
    const timingNote = document.createElement("div");
    timingNote.className = `${prefix}-timing-note`;
    if (_bendingLastMs != null)
        timingNote.textContent = `last update: ${_bendingLastMs}ms`;
    container.appendChild(timingNote);
}

// ── act-modal bending view ────────────────────────────────────────────────────

// ── BendingParameter "macro" rows in the act-modal list ──────────────────────

function _renderActModalBpRows(listEl) {
    // Remove any existing BP rows
    listEl.querySelectorAll(".act-modal-bp-sep, .act-modal-bp-row").forEach(el => el.remove());
    if (_bendingParams.length === 0) return;

    const sep = document.createElement("div");
    sep.className = "act-modal-bp-sep";
    sep.textContent = "macro parameters";
    listEl.appendChild(sep);

    _bendingParams.forEach(bp => {
        const row = document.createElement("div");
        row.className = "act-modal-bp-row";
        row.dataset.bpName = bp.name;

        const badge = document.createElement("span");
        badge.className = "act-modal-badge act-modal-macro-badge";
        badge.textContent = "macro";

        const name = document.createElement("span");
        name.className = "act-modal-name";
        name.textContent = bp.name;

        const valSpan = document.createElement("span");
        valSpan.className = "act-modal-bp-val";
        valSpan.textContent = _bpFmt(bp, bp.value);

        const linkedSpan = document.createElement("span");
        linkedSpan.className = "act-modal-shape";
        if (bp.linked && bp.linked.length > 0)
            linkedSpan.textContent = `→ ${bp.linked.map(l => l.node).join(", ")}`;

        const pinBtn = document.createElement("button");
        pinBtn.className = "act-modal-pin-btn";
        pinBtn.title = "Pin to dashboard";
        pinBtn.textContent = "⊕";
        pinBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            addBendingParamPin(bp.name);
        });

        row.appendChild(badge);
        row.appendChild(name);
        row.appendChild(valSpan);
        row.appendChild(linkedSpan);
        row.appendChild(pinBtn);
        listEl.appendChild(row);
    });
}

// Refresh BP rows in act-modal list if it's currently open
function _refreshActModalBpRows() {
    const list = document.getElementById("act-modal-list");
    if (!list || document.getElementById("act-modal").style.display === "none") return;
    _renderActModalBpRows(list);
    // Apply current search filter to the new rows
    const q = (document.getElementById("act-modal-search") || {}).value || "";
    _filterActModalBpRows(q);
}

function _filterActModalBpRows(q) {
    const list = document.getElementById("act-modal-list");
    if (!list) return;
    const lq = q.trim().toLowerCase();
    list.querySelectorAll(".act-modal-bp-row").forEach(row => {
        const name = (row.dataset.bpName || "").toLowerCase();
        row.style.display = (!lq || name.includes(lq)) ? "" : "none";
    });
    const sep = list.querySelector(".act-modal-bp-sep");
    if (sep) {
        const anyVisible = [...list.querySelectorAll(".act-modal-bp-row")].some(r => r.style.display !== "none");
        sep.style.display = anyVisible ? "" : "none";
    }
}

// ── inline BendingParameter creation panel ────────────────────────────────────

function _showBpCreatePanel(anchor, paramName, curVal, paramRange, onCreated, paramType) {
    document.querySelectorAll(".bp-create-panel").forEach(el => el.remove());

    const panel = document.createElement("div");
    panel.className = "bp-create-panel";
    panel.addEventListener("click", e => e.stopPropagation());

    function _row(labelText, ...inputs) {
        const r = document.createElement("div");
        r.className = "bp-create-row";
        const lbl = document.createElement("span");
        lbl.className = "bp-create-lbl";
        lbl.textContent = labelText;
        r.appendChild(lbl);
        inputs.forEach(i => r.appendChild(i));
        return r;
    }
    function _inp(placeholder, val, type = "text") {
        const i = document.createElement("input");
        i.type = type; i.className = "bp-create-inp";
        i.placeholder = placeholder;
        if (val != null && val !== "") i.value = val;
        return i;
    }

    const nameInp = _inp("macro name", `macro_${paramName}`);

    // Type selector — determines the Python type of the BendingParameter
    const typeSel = document.createElement("select");
    typeSel.className = "bp-create-inp";
    typeSel.style.flex = "1";
    ["float", "int", "bool"].forEach(t => {
        const o = document.createElement("option");
        o.value = t; o.textContent = t;
        if (t === (paramType || "float")) o.selected = true;
        typeSel.appendChild(o);
    });

    // When type is bool, hide numeric fields; when int, use integer step
    const valInp  = _inp("initial value", "", "number");
    const minInp  = _inp("min", "", "number");
    const maxInp  = _inp("max", "", "number");
    const rangeRow = document.createElement("div");
    rangeRow.className = "bp-create-row";

    function _syncType() {
        const t = typeSel.value;
        const isBool = t === "bool";
        valInp.step = t === "int" ? "1" : "any";
        valInp.value = isBool ? "0" : (curVal != null ? (t === "int" ? Math.round(curVal) : Number(curVal).toFixed(4)) : "0");
        rangeRow.style.display = isBool ? "none" : "";
        minInp.step = maxInp.step = t === "int" ? "1" : "any";
        minInp.value = (!isBool && paramRange && paramRange[0] != null) ? paramRange[0] : "";
        maxInp.value = (!isBool && paramRange && paramRange[1] != null) ? paramRange[1] : "";
    }
    typeSel.addEventListener("change", _syncType);
    _syncType();

    panel.appendChild(_row("name", nameInp));
    panel.appendChild(_row("type", typeSel));
    panel.appendChild(_row("value", valInp));

    const rangeLbl = document.createElement("span");
    rangeLbl.className = "bp-create-lbl"; rangeLbl.textContent = "range";
    minInp.style.flex = "1"; maxInp.style.flex = "1";
    const sep = document.createElement("span");
    sep.textContent = "—"; sep.style.cssText = "color:var(--text-muted);flex-shrink:0;font-size:10px";
    rangeRow.appendChild(rangeLbl); rangeRow.appendChild(minInp);
    rangeRow.appendChild(sep); rangeRow.appendChild(maxInp);
    panel.appendChild(rangeRow);

    const btns = document.createElement("div");
    btns.className = "bp-create-btns";
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "bp-create-cancel"; cancelBtn.textContent = "Cancel";
    const createBtn = document.createElement("button");
    createBtn.className = "bp-create-ok"; createBtn.textContent = "Create";

    cancelBtn.addEventListener("click", () => panel.remove());
    createBtn.addEventListener("click", async () => {
        const bpName = nameInp.value.trim();
        if (!bpName) { nameInp.focus(); return; }
        const pt  = typeSel.value;
        const val = pt === "bool" ? (parseFloat(valInp.value) !== 0 ? 1 : 0)
                  : pt === "int"  ? Math.round(parseFloat(valInp.value) || 0)
                  : parseFloat(valInp.value) || 0;
        const mn  = rangeRow.style.display !== "none" && minInp.value.trim() !== "" ? parseFloat(minInp.value) : null;
        const mx  = rangeRow.style.display !== "none" && maxInp.value.trim() !== "" ? parseFloat(maxInp.value) : null;
        createBtn.disabled = true;
        try {
            const r = await fetch("/api/bending_params/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: bpName, value: val, param_type: pt, range_min: mn, range_max: mx }),
            });
            const d = await r.json();
            if (!r.ok) { showToast("error", d.error || "Could not create macro"); createBtn.disabled = false; return; }
            _syncBendingState(d);
            panel.remove();
            onCreated(bpName);
            showToast("ok", `Created macro "${bpName}" (${pt})`);
        } catch (err) { showToast("error", err.message, err.traceback, err.traceback); createBtn.disabled = false; }
    });

    btns.appendChild(cancelBtn); btns.appendChild(createBtn);
    panel.appendChild(btns);
    document.body.appendChild(panel);

    // Position below the anchor, flip up if near viewport bottom
    const rect = anchor.getBoundingClientRect();
    panel.style.position = "fixed";
    panel.style.left = rect.left + "px";
    panel.style.top  = (rect.bottom + 4) + "px";
    requestAnimationFrame(() => {
        const pr = panel.getBoundingClientRect();
        if (pr.right > window.innerWidth - 8)
            panel.style.left = Math.max(8, window.innerWidth - pr.width - 8) + "px";
        if (pr.bottom > window.innerHeight - 8)
            panel.style.top = (rect.top - pr.height - 4) + "px";
    });

    // Dismiss on outside click
    const dismiss = (e) => {
        if (!panel.contains(e.target)) { panel.remove(); document.removeEventListener("mousedown", dismiss); }
    };
    setTimeout(() => document.addEventListener("mousedown", dismiss), 0);
    nameInp.focus(); nameInp.select();
}

// ── BendingParameter helpers ──────────────────────────────────────────────────

// Macros are typed (float / int / bool) and the backend enforces it: a bool one
// refuses a numeric value outright and an int one rounds. These helpers keep the
// widgets honest so the user never sends a value the macro cannot hold.
function _bpType(bp) { return (bp && bp.param_type) || "float"; }

// value → the string shown in the macro's text box
function _bpFmt(bp, v) {
    const t = _bpType(bp);
    if (t === "bool") return (v === true || Number(v) !== 0) ? "true" : "false";
    if (t === "int")  return String(Math.round(Number(v) || 0));
    return Number(v || 0).toFixed(4);
}

// text/slider input → the value to send, or null when it isn't a valid one
function _bpParse(bp, raw) {
    const t = _bpType(bp);
    if (t === "bool") {
        if (typeof raw === "boolean") return raw;
        const s = String(raw).trim().toLowerCase();
        if (["true", "1", "yes", "on"].includes(s)) return true;
        if (["false", "0", "no", "off", ""].includes(s)) return false;
        return null;
    }
    const v = parseFloat(raw);
    if (isNaN(v)) return null;
    return t === "int" ? Math.round(v) : v;
}

function _bpStep(bp) { return _bpType(bp) === "int" ? "1" : "any"; }

async function _unlinkBpParam(binding, paramName) {
    try {
        const r = await fetch(`/api/bending/${binding.id}/unlink/`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ param_name: paramName }),
        });
        const d = await r.json();
        if (!r.ok) { showToast("error", d.error, d.traceback); return; }
        _syncBendingState(d);
        _refreshAllBendingUI();
    } catch (err) {
        showToast("error", "Unlink failed: " + err.message, err.traceback);
    }
}

// `range` — [lo, hi] this attachment should map the macro's 0…1 onto, or null to
// let the server pick the default (the parameter's own declared range). Passing
// it again later is how a link is re-mapped: the callback gets a fresh derived
// parameter with the new arithmetic in place of the old one.
async function _linkBpParam(binding, paramName, bpName, range) {
    const body = { param_name: paramName, bp_name: bpName };
    if (range !== undefined) {
        body.range_min = range ? range[0] : null;
        body.range_max = range ? range[1] : null;
    }
    const r = await fetch(`/api/bending/${binding.id}/link/`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!r.ok) { const _e = new Error(d.error || r.statusText); _e.traceback = d.traceback || ""; throw _e; }
    _syncBendingState(d);
    _refreshAllBendingUI();
}

function _openBpLinkDialog(binding, paramName, currentVal) {
    // Remove any existing dialog
    const existing = document.getElementById("bp-link-dialog");
    if (existing) existing.remove();

    const overlay = document.createElement("div");
    overlay.id = "bp-link-dialog";
    overlay.className = "bp-link-dialog-overlay";

    const panel = document.createElement("div");
    panel.className = "bp-link-dialog-panel";

    const hdr = document.createElement("div");
    hdr.className = "bp-link-dialog-header";
    hdr.innerHTML = `<span>⊕ Promote <strong>${paramName}</strong></span>`;
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "×";
    closeBtn.className = "bp-link-dialog-close";
    closeBtn.addEventListener("click", () => overlay.remove());
    hdr.appendChild(closeBtn);
    panel.appendChild(hdr);

    // the param's declared type — a macro of another type won't link
    const wantType = ((binding.descriptor || {}).params || {})[paramName]
        ? binding.descriptor.params[paramName].type : null;

    // The range this attachment will map a normalised macro's 0…1 onto. Defaults
    // to what the parameter itself declares, and is the user's to change — it
    // belongs to the link, so the same macro can span something else elsewhere.
    const declared = (((binding.descriptor || {}).params || {})[paramName] || {}).range || [null, null];
    let mapLo = declared[0], mapHi = declared[1];

    // Existing BPs section
    if (_bendingParams.length > 0) {
        const sec = document.createElement("div");
        sec.className = "bp-link-section";
        const secTitle = document.createElement("div");
        secTitle.className = "bp-link-section-title";
        secTitle.textContent = "Link to existing:";
        sec.appendChild(secTitle);

        // range picker, shown only where it means something (a float param
        // driven by a normalised macro)
        const rangeRow = document.createElement("div");
        rangeRow.className = "bp-link-map-row";
        if (wantType === "float" || wantType == null) {
            const lbl = document.createElement("span");
            lbl.className = "bp-link-map-lbl";
            lbl.textContent = "macro 0…1 maps to";
            const mk = (init, ph, set) => {
                const i = document.createElement("input");
                i.type = "text"; i.className = "bp-link-map-inp";
                i.value = init != null ? init : ""; i.placeholder = ph;
                i.addEventListener("change", () => {
                    const v = parseFloat(i.value);
                    set(isFinite(v) ? v : null);
                });
                return i;
            };
            rangeRow.appendChild(lbl);
            rangeRow.appendChild(mk(mapLo, "at 0", (v) => { mapLo = v; }));
            rangeRow.appendChild(mk(mapHi, "at 1", (v) => { mapHi = v; }));
            sec.appendChild(rangeRow);
        }
        _bendingParams.forEach(bp => {
            const row = document.createElement("div");
            row.className = "bp-link-existing-row";
            const info = document.createElement("span");
            info.className = "bp-link-existing-name";
            const mismatch = wantType && wantType !== _bpType(bp);
            info.textContent = `${bp.name}  (${_bpType(bp)}: ${_bpFmt(bp, bp.value)})`
                             + (mismatch ? "  ⚠" : "");
            if (mismatch) info.title = `macro is ${_bpType(bp)}, but '${paramName}' expects ${wantType}`;
            const linkBtn = document.createElement("button");
            linkBtn.textContent = "link";
            linkBtn.className = "bp-link-use-btn";
            linkBtn.addEventListener("click", async () => {
                try {
                    const rng = (mapLo != null && mapHi != null && mapLo !== mapHi)
                        ? [mapLo, mapHi] : null;
                    await _linkBpParam(binding, paramName, bp.name, rng);
                    overlay.remove();
                    showToast("ok", `Linked ${paramName} → ${bp.name}`
                                    + (rng ? ` over ${rng[0]}…${rng[1]}` : ""));
                } catch (err) {
                    showToast("error", err.message, err.traceback, err.traceback);
                }
            });
            row.appendChild(info);
            row.appendChild(linkBtn);
            sec.appendChild(row);
        });
        panel.appendChild(sec);
    }

    // Create new BP section
    const newSec = document.createElement("div");
    newSec.className = "bp-link-section";
    const newTitle = document.createElement("div");
    newTitle.className = "bp-link-section-title";
    newTitle.textContent = "Create new:";
    newSec.appendChild(newTitle);

    function _field(label, type, value, placeholder) {
        const wrap = document.createElement("div");
        wrap.className = "bp-link-field";
        const lbl = document.createElement("label");
        lbl.textContent = label;
        const inp = document.createElement("input");
        inp.type = type;
        inp.value = value !== undefined ? value : "";
        if (placeholder) inp.placeholder = placeholder;
        if (type === "number") inp.step = "any";
        wrap.appendChild(lbl);
        wrap.appendChild(inp);
        newSec.appendChild(wrap);
        return inp;
    }

    const nameInp  = _field("Name",      "text",   "", "e.g. gain");

    // The macro must be created with the *param's* type, or the link that follows
    // is rejected — a float macro cannot drive an int argument.
    const newType = wantType || "float";
    const typeWrap = document.createElement("div");
    typeWrap.className = "bp-link-field";
    const typeLbl = document.createElement("label");
    typeLbl.textContent = "Type";
    const typeVal = document.createElement("span");
    typeVal.className = "bp-link-type-fixed";
    typeVal.textContent = newType;
    typeVal.title = `taken from '${paramName}'`;
    typeWrap.appendChild(typeLbl); typeWrap.appendChild(typeVal);
    newSec.appendChild(typeWrap);

    const isBool   = newType === "bool";
    const defVal   = currentVal != null ? currentVal : (newType === "int" ? 0 : 0.5);
    const valInp   = _field("Value",     isBool ? "checkbox" : "number",
                            isBool ? undefined : (newType === "int" ? Math.round(defVal) : defVal));
    if (isBool) valInp.checked = Number(currentVal) !== 0;
    if (newType === "int") valInp.step = "1";
    const minInp   = _field("Range min", "number", declared[0] != null ? declared[0] : "", "none");
    const maxInp   = _field("Range max", "number", declared[1] != null ? declared[1] : "", "none");
    if (isBool) {
        minInp.parentElement.style.display = "none";
        maxInp.parentElement.style.display = "none";
    } else if (newType === "int") {
        minInp.step = maxInp.step = "1";
    }

    const createBtn = document.createElement("button");
    createBtn.textContent = "Create & link";
    createBtn.className = "bp-link-create-btn";
    createBtn.addEventListener("click", async () => {
        const bpName = nameInp.value.trim();
        if (!bpName) { showToast("error", "Name is required"); return; }
        const _num = (s) => newType === "int" ? Math.round(parseFloat(s)) : parseFloat(s);
        const bpVal  = isBool ? (valInp.checked ? 1 : 0) : _num(valInp.value);
        const bpMin  = (!isBool && minInp.value.trim() !== "") ? _num(minInp.value) : null;
        const bpMax  = (!isBool && maxInp.value.trim() !== "") ? _num(maxInp.value) : null;
        try {
            // Create the BP first
            const cr = await fetch("/api/bending_params/", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ name: bpName, value: bpVal, param_type: newType,
                                       range_min: bpMin, range_max: bpMax }),
            });
            const cd = await cr.json();
            if (!cr.ok) throw new Error(cd.error || cr.statusText);
            _syncBendingState(cd);
            // Then link, over the range the picker above is showing
            const rng = (bpMin != null && bpMax != null && bpMin !== bpMax)
                ? [bpMin, bpMax] : null;
            await _linkBpParam(binding, paramName, bpName, rng);
            overlay.remove();
            showToast("ok", `Created & linked ${paramName} → ${bpName}`);
        } catch (err) {
            showToast("error", err.message, err.traceback, err.traceback);
        }
    });
    newSec.appendChild(createBtn);
    panel.appendChild(newSec);

    overlay.appendChild(panel);
    document.body.appendChild(overlay);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
    nameInp.focus();
}

function _renderBendingParamsSection(container) {
    let sec = container.querySelector(".act-bend-params-section");
    if (!sec) {
        sec = document.createElement("div");
        sec.className = "act-bend-params-section";
        // Insert after the toolbar, before the bindings list
        const toolbar = container.querySelector("#act-bend-toolbar");
        if (toolbar) toolbar.after(sec);
        else container.insertBefore(sec, container.firstChild);
    }
    sec.innerHTML = "";

    const hdr = document.createElement("div");
    hdr.className = "act-bend-params-header";
    const title = document.createElement("span");
    title.className = "act-bend-params-title";
    title.textContent = "Macros";
    const addBtn = document.createElement("button");
    addBtn.className = "act-bend-params-add-btn";
    addBtn.textContent = "+ new";
    addBtn.title = "Create a new BendingParameter";
    addBtn.addEventListener("click", () => _openCreateBpDialog());
    hdr.appendChild(title);
    hdr.appendChild(addBtn);
    sec.appendChild(hdr);

    if (_bendingParams.length === 0) {
        const empty = document.createElement("div");
        empty.className = "act-bend-params-empty";
        empty.textContent = "No macros yet — use the ⊕ button or create one from a param's dropdown in the bend dialog.";
        sec.appendChild(empty);
        return;
    }

    _bendingParams.forEach(bp => {
        const card = document.createElement("div");
        card.className = "abp-card";

        // ── Header row: name + actions ────────────────────────────────────────
        const cardHdr = document.createElement("div");
        cardHdr.className = "abp-card-hdr";
        const nameLbl = document.createElement("span");
        nameLbl.className = "abp-name";
        nameLbl.textContent = bp.name;
        const typeLbl = document.createElement("span");
        typeLbl.className = "abp-type";
        typeLbl.textContent = _bpType(bp);
        typeLbl.title = `macro type: ${_bpType(bp)}`;
        const actions = document.createElement("div");
        actions.className = "abp-actions";
        const pinBtn = document.createElement("button");
        pinBtn.className = "abp-btn"; pinBtn.title = "Pin to dashboard"; pinBtn.textContent = "⊕";
        pinBtn.addEventListener("click", () => addBendingParamPin(bp.name));
        const delBtn = document.createElement("button");
        delBtn.className = "abp-btn abp-btn-del"; delBtn.title = "Delete"; delBtn.textContent = "×";
        delBtn.addEventListener("click", async () => {
            if (!confirm(`Delete macro "${bp.name}"?`)) return;
            _dropManual(`m:${bp.name}`);
            const r = await fetch(`/api/bending_params/${encodeURIComponent(bp.name)}/`, { method: "DELETE" });
            const d = await r.json();
            if (!r.ok) { showToast("error", d.error, d.traceback); return; }
            _syncBendingState(d); _refreshAllBendingUI();
        });
        actions.appendChild(pinBtn); actions.appendChild(delBtn);
        cardHdr.appendChild(nameLbl); cardHdr.appendChild(typeLbl); cardHdr.appendChild(actions);
        card.appendChild(cardHdr);

        // ── Value widget (typed) ──────────────────────────────────────────────
        const isBool = _bpType(bp) === "bool";
        const sliderRow = document.createElement("div");
        sliderRow.className = "abp-slider-row";
        const _upd = (v) => _scheduleBpUpdate(bp.name, v);
        if (isBool) {
            // a bool macro is a toggle — a slider would send values it can't hold
            const chk = document.createElement("input");
            chk.type = "checkbox"; chk.className = "abp-toggle";
            chk.checked = bp.value === true || Number(bp.value) !== 0;
            const stateLbl = document.createElement("span");
            stateLbl.className = "abp-val-input abp-val-static";
            stateLbl.textContent = chk.checked ? "true" : "false";
            chk.addEventListener("change", () => {
                stateLbl.textContent = chk.checked ? "true" : "false";
                _upd(chk.checked);
            });
            sliderRow.appendChild(chk); sliderRow.appendChild(stateLbl);
        } else {
            const slider = document.createElement("input");
            slider.type = "range"; slider.className = "abp-slider";
            slider.min  = bp.min_clamp != null ? bp.min_clamp : -2;
            slider.max  = bp.max_clamp != null ? bp.max_clamp :  2;
            slider.step = _bpStep(bp); slider.value = bp.value;
            const valInp = document.createElement("input");
            valInp.type = "text"; valInp.className = "abp-val-input";
            valInp.value = _bpFmt(bp, bp.value);
            // a normalised macro reads 0…1; show what that currently maps to
            const mapLbl = document.createElement("span");
            mapLbl.className = "abp-mapped";
            const _mapped = (v) => {
                if (!bp.target_range) return "";
                const [lo, hi] = bp.target_range;
                const m = lo + Number(v) * (hi - lo);
                return "→ " + (Number.isInteger(m) ? m : m.toFixed(3));
            };
            mapLbl.textContent = _mapped(bp.value);
            mapLbl.title = "The value this macro's position sends to its targets";
            slider.addEventListener("input", () => {
                const v = _bpParse(bp, slider.value);
                if (v === null) return;
                valInp.value = _bpFmt(bp, v); mapLbl.textContent = _mapped(v); _upd(v);
            });
            valInp.addEventListener("change", () => {
                const v = _bpParse(bp, valInp.value);
                if (v === null) { valInp.value = _bpFmt(bp, slider.value); return; }
                if (v < parseFloat(slider.min)) slider.min = v - Math.max(Math.abs(v), 1);
                if (v > parseFloat(slider.max)) slider.max = v + Math.max(Math.abs(v), 1);
                slider.value = v; valInp.value = _bpFmt(bp, v); _upd(v);
            });
            sliderRow.appendChild(slider); sliderRow.appendChild(valInp);
            if (bp.target_range) sliderRow.appendChild(mapLbl);
        }
        card.appendChild(sliderRow);

        // ── Range row (meaningless for a bool macro) ──────────────────────────
        // A normalised macro always reads 0…1, so this row is not its clamp — it
        // is the range that 0…1 spans on whatever the macro drives.
        const norm = !!bp.normalized;
        const shown = norm ? (bp.target_range || [null, null])
                           : [bp.min_clamp, bp.max_clamp];
        const rangeRow = document.createElement("div");
        rangeRow.className = "abp-range-row";
        if (isBool) rangeRow.style.display = "none";
        const rangeLbl = document.createElement("span");
        rangeLbl.className = "abp-range-lbl";
        rangeLbl.textContent = norm ? "maps to" : "range";
        if (norm) rangeLbl.title = "This macro reads 0…1; these are the values that span.";
        const minInp = document.createElement("input");
        minInp.type = "text"; minInp.className = "abp-range-inp"; minInp.placeholder = "min";
        minInp.value = shown[0] != null ? shown[0] : "";
        const maxInp = document.createElement("input");
        maxInp.type = "text"; maxInp.className = "abp-range-inp"; maxInp.placeholder = "max";
        maxInp.value = shown[1] != null ? shown[1] : "";
        async function _patchRange() {
            // clamps are typed like the macro itself (integer bounds for an int)
            const mn = minInp.value.trim() === "" ? null : _bpParse(bp, minInp.value);
            const mx = maxInp.value.trim() === "" ? null : _bpParse(bp, maxInp.value);
            const r = await fetch(`/api/bending_params/${encodeURIComponent(bp.name)}/`, {
                method: "PATCH", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ min_clamp: mn, max_clamp: mx, clamp: mn != null && mx != null }),
            });
            const d = await r.json();
            if (!r.ok) { showToast("error", d.error, d.traceback); return; }
            _syncBendingState(d);
            // a normalised macro keeps its 0…1 slider — only what it maps to moved
            const slider = sliderRow.querySelector(".abp-slider");
            if (slider && !norm) { if (mn != null) slider.min = mn; if (mx != null) slider.max = mx; }
            if (norm) _refreshAllBendingUI();
            if (mn != null) minInp.value = mn;
            if (mx != null) maxInp.value = mx;
        }
        minInp.addEventListener("change", _patchRange); maxInp.addEventListener("change", _patchRange);
        rangeRow.appendChild(rangeLbl); rangeRow.appendChild(minInp); rangeRow.appendChild(maxInp);
        card.appendChild(rangeRow);

        // ── Linked bindings ───────────────────────────────────────────────────
        if (bp.linked && bp.linked.length > 0) {
            const linkedDiv = document.createElement("div");
            linkedDiv.className = "abp-linked";
            linkedDiv.textContent = "→ " + bp.linked.map(l =>
                (l.callback_type || "?") + (l.param ? `(${l.param})` : "") + " @ " + (l.node || "?")
            ).join("  ·  ");
            card.appendChild(linkedDiv);
        }

        sec.appendChild(card);
    });
}

function _openCreateBpDialog() {
    // Create-only dialog (no binding to link to immediately)
    const existing = document.getElementById("bp-link-dialog");
    if (existing) existing.remove();

    const overlay = document.createElement("div");
    overlay.id = "bp-link-dialog";
    overlay.className = "bp-link-dialog-overlay";

    const panel = document.createElement("div");
    panel.className = "bp-link-dialog-panel";

    const hdr = document.createElement("div");
    hdr.className = "bp-link-dialog-header";
    hdr.innerHTML = `<span>⊕ New BendingParameter</span>`;
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "×";
    closeBtn.className = "bp-link-dialog-close";
    closeBtn.addEventListener("click", () => overlay.remove());
    hdr.appendChild(closeBtn);
    panel.appendChild(hdr);

    const sec = document.createElement("div");
    sec.className = "bp-link-section";

    function _field(label, type, value, placeholder) {
        const wrap = document.createElement("div");
        wrap.className = "bp-link-field";
        const lbl = document.createElement("label");
        lbl.textContent = label;
        const inp = document.createElement("input");
        inp.type = type;
        inp.value = value !== undefined ? value : "";
        if (placeholder) inp.placeholder = placeholder;
        if (type === "number") inp.step = "any";
        wrap.appendChild(lbl);
        wrap.appendChild(inp);
        sec.appendChild(wrap);
        return inp;
    }

    const nameInp = _field("Name",      "text",   "", "e.g. gain");

    // Type selector
    const typeWrap = document.createElement("div"); typeWrap.className = "bp-link-field";
    const typeLbl = document.createElement("label"); typeLbl.textContent = "Type";
    const typeSel = document.createElement("select"); typeSel.className = "bp-link-type-sel";
    ["float", "int", "bool"].forEach(t => {
        const o = document.createElement("option"); o.value = t; o.textContent = t;
        typeSel.appendChild(o);
    });
    typeWrap.appendChild(typeLbl); typeWrap.appendChild(typeSel);
    sec.appendChild(typeWrap);

    const valInp  = _field("Value",     "number", 0);
    const minInp  = _field("Range min", "number", "", "none");
    const maxInp  = _field("Range max", "number", "", "none");

    typeSel.addEventListener("change", () => {
        const isBool = typeSel.value === "bool";
        valInp.step = typeSel.value === "int" ? "1" : "any";
        minInp.parentElement.style.display = isBool ? "none" : "";
        maxInp.parentElement.style.display = isBool ? "none" : "";
    });

    const createBtn = document.createElement("button");
    createBtn.textContent = "Create";
    createBtn.className = "bp-link-create-btn";
    createBtn.addEventListener("click", async () => {
        const bpName = nameInp.value.trim();
        if (!bpName) { showToast("error", "Name is required"); return; }
        const pt    = typeSel.value;
        const bpVal = pt === "bool" ? (parseFloat(valInp.value) !== 0 ? 1 : 0)
                    : pt === "int"  ? Math.round(parseFloat(valInp.value) || 0)
                    : parseFloat(valInp.value) || 0;
        const bpMin = minInp.value.trim() !== "" && pt !== "bool" ? parseFloat(minInp.value) : null;
        const bpMax = maxInp.value.trim() !== "" && pt !== "bool" ? parseFloat(maxInp.value) : null;
        try {
            const r = await fetch("/api/bending_params/", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ name: bpName, value: bpVal, param_type: pt, range_min: bpMin, range_max: bpMax }),
            });
            const d = await r.json();
            if (!r.ok) { const _e = new Error(d.error || r.statusText); _e.traceback = d.traceback || ""; throw _e; }
            _syncBendingState(d);
            _refreshAllBendingUI();
            overlay.remove();
            showToast("ok", `Created BendingParameter "${bpName}"`);
        } catch (err) {
            showToast("error", err.message, err.traceback, err.traceback);
        }
    });
    sec.appendChild(createBtn);
    panel.appendChild(sec);

    overlay.appendChild(panel);
    document.body.appendChild(overlay);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
    nameInp.focus();
}

function addBendingParamPin(bpName) {
    const bp = _bendingParams.find(x => x.name === bpName);
    if (!bp) { showToast("error", "BendingParameter not found"); return; }
    for (const page of pinPages) {
        if (page.some(p => p.type === "bp" && p.bpName === bpName)) {
            showToast("info", "Already pinned"); return;
        }
    }
    const pin = { id: genId(), type: "bp", bpName, label: bpName };
    pinPages[currentPinPage].push(pin);
    renderPinTabs();
    renderCurrentPage();
    _updatePinCount();
    showToast("info", `Pinned parameter "${bpName}"`);
}

function _makeBpPinCard(pin) {
    const bp = _bendingParams.find(x => x.name === pin.bpName) || {};
    const card = document.createElement("div");
    card.id = `pin-card-${pin.id}`;
    card.className = "pin-card pin-card-bp";
    if (pin.collapsed) card.classList.add("pin-card--collapsed");

    const hdr = document.createElement("div");
    hdr.className = "pin-card-header";
    const lbl = document.createElement("span");
    lbl.className = "pin-card-label";
    lbl.innerHTML = `<span class="pin-bp-icon">⊕</span> ${pin.bpName}`;
    const closeBtn = document.createElement("button");
    closeBtn.className = "pin-card-close";
    closeBtn.textContent = "×";
    closeBtn.addEventListener("click", () => removePin(pin.id));
    hdr.appendChild(lbl);
    hdr.appendChild(closeBtn);
    card.appendChild(hdr);

    // Drag support
    let dragging = false, ox = 0, oy = 0;
    hdr.addEventListener("mousedown", (e) => {
        if (e.target.tagName === "BUTTON") return;
        if (e.detail >= 2) { pin.collapsed = card.classList.toggle("pin-card--collapsed"); e.preventDefault(); return; }
        const r = card.getBoundingClientRect();
        ox = e.clientX - r.left; oy = e.clientY - r.top;
        dragging = true;
        e.preventDefault();
    });
    document.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const cont = document.getElementById("pin-items");
        const cr = cont ? cont.getBoundingClientRect() : { left: 0, top: 0 };
        pin.x = e.clientX - cr.left - ox;
        pin.y = Math.max(0, e.clientY - cr.top  - oy);
        card.style.left = pin.x + "px";
        card.style.top  = pin.y + "px";
    });
    document.addEventListener("mouseup", () => { dragging = false; });

    // Body
    const body = document.createElement("div");
    body.className = "pin-card-bp-body";

    const sliderMin = bp.min_clamp != null ? bp.min_clamp : -2;
    const sliderMax = bp.max_clamp != null ? bp.max_clamp :  2;
    const curVal    = bp.value != null ? bp.value : 0;
    const isBool    = _bpType(bp) === "bool";

    const sliderWrap = document.createElement("div");
    sliderWrap.className = "pin-bp-slider-wrap";
    const _updateBp = (v) => _scheduleBpUpdate(pin.bpName, v);

    if (isBool) {
        const chk = document.createElement("input");
        chk.type = "checkbox";
        chk.className = "pin-bp-toggle";
        chk.checked = curVal === true || Number(curVal) !== 0;
        const stateLbl = document.createElement("span");
        stateLbl.className = "pin-bp-val-input pin-bp-val-static";
        stateLbl.textContent = chk.checked ? "true" : "false";
        chk.addEventListener("change", () => {
            stateLbl.textContent = chk.checked ? "true" : "false";
            _updateBp(chk.checked);
        });
        sliderWrap.appendChild(chk);
        sliderWrap.appendChild(stateLbl);
    } else {
        const slider = document.createElement("input");
        slider.type = "range";
        slider.className = "pin-bp-slider";
        slider.min  = sliderMin;
        slider.max  = sliderMax;
        slider.step = _bpStep(bp);
        slider.value = curVal;
        const valInp = document.createElement("input");
        valInp.type  = "text";
        valInp.className = "pin-bp-val-input";
        valInp.value = _bpFmt(bp, curVal);

        slider.addEventListener("input", () => {
            const v = _bpParse(bp, slider.value);
            if (v === null) return;
            valInp.value = _bpFmt(bp, v);
            _updateBp(v);
        });
        valInp.addEventListener("change", () => {
            const v = _bpParse(bp, valInp.value);
            if (v === null) { valInp.value = _bpFmt(bp, slider.value); return; }
            if (v < parseFloat(slider.min)) slider.min = v - Math.max(Math.abs(v), 1);
            if (v > parseFloat(slider.max)) slider.max = v + Math.max(Math.abs(v), 1);
            slider.value = v;
            valInp.value = _bpFmt(bp, v);
            _updateBp(v);
        });

        sliderWrap.appendChild(slider);
        sliderWrap.appendChild(valInp);
    }
    body.appendChild(sliderWrap);

    // Editable range row. For a normalised macro (which always reads 0…1) this
    // is not its clamp but the range that 0…1 spans on whatever it drives.
    const norm = !!bp.normalized;
    const shownRange = norm ? (bp.target_range || [null, null])
                            : [bp.min_clamp, bp.max_clamp];
    const rangeRow = document.createElement("div");
    rangeRow.className = "pin-bp-range-row";
    if (isBool) rangeRow.style.display = "none";   // no range for a bool macro
    const rangeLbl = document.createElement("span");
    rangeLbl.className = "pin-bp-range-lbl";
    rangeLbl.textContent = norm ? "maps to" : "range";
    if (norm) rangeLbl.title = "This macro reads 0…1; these are the values that spans.";
    const minInp = document.createElement("input");
    minInp.type = "text"; minInp.className = "pin-bp-range-inp";
    minInp.placeholder = "min"; minInp.title = norm ? "value at 0" : "min clamp";
    minInp.value = shownRange[0] != null ? shownRange[0] : "";
    const maxInp = document.createElement("input");
    maxInp.type = "text"; maxInp.className = "pin-bp-range-inp";
    maxInp.placeholder = "max"; maxInp.title = norm ? "value at 1" : "max clamp";
    maxInp.value = shownRange[1] != null ? shownRange[1] : "";

    async function _patchRange() {
        // clamps are typed like the macro itself (integer bounds for an int)
        const mn = minInp.value.trim() === "" ? null : _bpParse(bp, minInp.value);
        const mx = maxInp.value.trim() === "" ? null : _bpParse(bp, maxInp.value);
        try {
            const r = await fetch(`/api/bending_params/${encodeURIComponent(pin.bpName)}/`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ min_clamp: mn, max_clamp: mx, clamp: mn != null && mx != null }),
            });
            const d = await r.json();
            if (!r.ok) { showToast("error", d.error, d.traceback); return; }
            _syncBendingState(d);
            // a normalised macro keeps its 0…1 slider — only its mapping moved
            const slider = sliderWrap.querySelector(".pin-bp-slider");
            if (slider && !norm) {
                if (mn != null) slider.min = mn;
                if (mx != null) slider.max = mx;
            }
            if (norm) _refreshAllBendingUI();
            if (mn != null) minInp.value = mn;
            if (mx != null) maxInp.value = mx;
        } catch (err) { showToast("error", err.message, err.traceback, err.traceback); }
    }
    minInp.addEventListener("change", _patchRange);
    maxInp.addEventListener("change", _patchRange);
    rangeRow.appendChild(rangeLbl);
    rangeRow.appendChild(minInp);
    rangeRow.appendChild(maxInp);
    body.appendChild(rangeRow);

    const linkedDiv = document.createElement("div");
    linkedDiv.className = "pin-bp-linked";
    if (bp.linked && bp.linked.length > 0) {
        linkedDiv.textContent = bp.linked.map(l =>
            (l.callback_type || "?") + (l.param ? `(${l.param})` : "") + " @ " + (l.node || "?")
        ).join("  ·  ");
    } else {
        linkedDiv.style.display = "none";
    }
    body.appendChild(linkedDiv);

    card.appendChild(body);
    return card;
}

// Update all BP pin card value displays after a BP update
function _refreshBpPinCards() {
    (pinPages[currentPinPage] || []).forEach(pin => {
        if (pin.type !== "bp") return;
        const card = document.getElementById(`pin-card-${pin.id}`);
        if (!card) return;
        const bp = _bendingParams.find(x => x.name === pin.bpName);
        if (!bp) return;
        // Value display (typed: toggle for bool, integer text for int)
        const slider = card.querySelector(".pin-bp-slider");
        const toggle = card.querySelector(".pin-bp-toggle");
        const inp    = card.querySelector(".pin-bp-val-input");
        if (slider) slider.value = bp.value;
        if (toggle) toggle.checked = bp.value === true || Number(bp.value) !== 0;
        if (inp) {
            const txt = _bpFmt(bp, bp.value);
            if (inp.tagName === "INPUT") inp.value = txt; else inp.textContent = txt;
        }
        // Linked-targets display
        const linkedDiv = card.querySelector(".pin-bp-linked");
        if (linkedDiv) {
            if (bp.linked && bp.linked.length > 0) {
                linkedDiv.textContent = bp.linked.map(l =>
                    (l.callback_type || "?") + (l.param ? `(${l.param})` : "") + " @ " + (l.node || "?")
                ).join("  ·  ");
                linkedDiv.style.display = "";
            } else {
                linkedDiv.style.display = "none";
            }
        }
        // Range inputs
        const minInp = card.querySelector(".pin-bp-range-inp[title='min clamp']");
        const maxInp = card.querySelector(".pin-bp-range-inp[title='max clamp']");
        if (minInp) minInp.value = bp.min_clamp != null ? bp.min_clamp : "";
        if (maxInp) maxInp.value = bp.max_clamp != null ? bp.max_clamp : "";
    });
}

function _renderBendActModalView() {
    const list = document.getElementById("act-bend-list");
    if (!list) return;
    list.innerHTML = "";

    const modeSelect = document.getElementById("act-bend-mode-select");
    if (modeSelect) modeSelect.value = _bendingUpdateMode;

    const threshInp = document.getElementById("act-bend-threshold");
    if (threshInp) threshInp.value = _bendingAutoThreshMs;

    if (_bendingBindings.length === 0) {
        list.innerHTML = `<div style="padding:8px 12px;color:var(--text-muted);font-size:11px;">
            No active bendings. Right-click a node and choose ⚡ bend.</div>`;
    }

    _bendingBindings.forEach(b => {
        const row = document.createElement("div");
        row.className = "abl-row";

        const header = document.createElement("div");
        header.className = "abl-header";

        const badge = document.createElement("span");
        badge.className = "abl-type-badge";
        badge.textContent = b.callback_type;
        badge.appendChild(_bendInfoIcon(b));

        const nodeLink = document.createElement("span");
        nodeLink.className = "abl-node";
        nodeLink.textContent = `${b.fn}:${b.node}`;
        nodeLink.title = "Jump to node";
        nodeLink.addEventListener("click", () => {
            document.getElementById("act-modal").style.display = "none";
            const cyNode = cy && cy.getElementById(b.node);
            if (cyNode && cyNode.length) {
                cy.animate({ fit: { eles: cyNode, padding: 80 }, duration: 400 });
                onNodeClick(cyNode);
            }
        });

        const removeBtn = document.createElement("button");
        removeBtn.className = "abl-remove-btn";
        removeBtn.title = "Remove";
        removeBtn.textContent = "✕";
        removeBtn.addEventListener("click", () => _removeBending(b.id));

        header.appendChild(badge);
        header.appendChild(nodeLink);
        header.appendChild(removeBtn);
        row.appendChild(header);

        const params = document.createElement("div");
        params.className = "abl-params";
        _buildParamControls(params, b, "abl");
        row.appendChild(params);

        list.appendChild(row);
    });

    // BendingParameters section at bottom
    const bendView = document.getElementById("act-modal-bend-view");
    if (bendView) _renderBendingParamsSection(bendView);
}

// ── helpers ───────────────────────────────────────────────────────────────────

function _formatParamsSummary(params) {
    return Object.entries(params || {})
        .map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(3) : v}`)
        .join("  ");
}

// ─── pane fold toggle buttons ──────────────────────────────────────────────────

function _initPaneFoldBtns() {
    const sidebar     = document.getElementById("sidebar");
    const details     = document.getElementById("details-panel");
    const sidebarBtn  = document.getElementById("sidebar-fold-btn");
    const detailsBtn  = document.getElementById("details-fold-btn");
    if (!sidebar || !details || !sidebarBtn || !detailsBtn) return;

    function togglePane(panel, btn, openArrow, closedArrow) {
        const folded = panel.classList.toggle("pane-folded");
        btn.classList.toggle("pane-folded", folded);
        btn.textContent = folded ? closedArrow : openArrow;
    }

    sidebarBtn.addEventListener("click", () =>
        togglePane(sidebar, sidebarBtn, "◀", "▶"));
    detailsBtn.addEventListener("click", () =>
        togglePane(details, detailsBtn, "▶", "◀"));
}

// ─── boot ─────────────────────────────────────────────────────────────────────
// Leaving for play mode (or anywhere): write the bench now — the debounced
// save may not have run. Coming back through the back button can restore this
// page from the browser's cache, as it was; reload so it reads the bench (and
// device, and method) play mode may have changed.
window.addEventListener("pagehide", () => { try { _saveBenchNow(currentModelName); } catch (_) {} });
window.addEventListener("pageshow", (e) => { if (e.persisted) location.reload(); });

document.addEventListener("DOMContentLoaded", () => {
    _initErrorLog();
    _initActSelectionBar();
    initCytoscape();
    loadCallbacks();
    loadInterfaceOptions();
    loadDevices();

    // method tabs (initial set from server-rendered HTML)
    document.querySelectorAll(".method-btn").forEach((btn) => {
        btn.addEventListener("click", () => loadGraph(btn.dataset.fn));
    });

    document.getElementById("graph-device").addEventListener("change", (e) => setDevice(e.target.value));

    document.getElementById("apply-all-btn").addEventListener("click", _applyAllPending);

    // bench batch
    document.getElementById("bench-batch-on").addEventListener("change", (e) => {
        _benchBatch.on = e.target.checked;
        _benchBatchChanged();
    });
    document.getElementById("bench-batch-mode").addEventListener("change", (e) => {
        _benchBatch.mode = e.target.value;
        _benchBatchChanged();
    });
    _syncBenchBatchBar();

    // the bench as a large window
    document.getElementById("expand-input-panel").addEventListener("click", () =>
        setInputPanelExpanded(!document.getElementById("input-panel").classList.contains("expanded")));
    document.getElementById("input-panel-backdrop").addEventListener("click", () => setInputPanelExpanded(false));
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && document.getElementById("input-panel").classList.contains("expanded")) {
            e.stopPropagation();
            setInputPanelExpanded(false);
        }
    }, true);

    // model picker
    renderModelPicker(INITIAL_MODELS);
    document.getElementById("model-picker-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        toggleModelPicker();
    });
    // close picker when clicking outside
    document.addEventListener("click", (e) => {
        if (_pickerOpen && !document.getElementById("model-picker").contains(e.target)) {
            toggleModelPicker(false);
        }
    });

    // fit button
    document.getElementById("fit-btn").addEventListener("click", () => {
        cy.animate({ fit: { padding: 30 } }, { duration: 300 });
    });

    const resetBtn = document.getElementById("reset-layout-btn");
    if (resetBtn) resetBtn.addEventListener("click", () => {
        _forgetPositions(currentFn);
        runLayout();
        showToast("info", "Layout reset");
    });

    // prune unreachable toggle
    document.getElementById("prune-btn").addEventListener("click", () => {
        _pruneUnreachable = !_pruneUnreachable;
        document.getElementById("prune-btn").classList.toggle("active", _pruneUnreachable);
        loadGraph(currentFn);
    });

    // simplification switches — each refetches, the way prune does, because the
    // folding rewires edges and only the server can do that correctly
    ["shape_calc", "unpack", "collapse"].forEach((key) => {
        const cb = document.getElementById("gopt-simplify-" + key);
        if (!cb) return;
        cb.addEventListener("change", () => {
            _simplify[key] = cb.checked;
            if (key === "collapse" && !cb.checked) _expandedChains.clear();
            loadGraph(currentFn);
        });
    });

    const depthSel = document.getElementById("module-depth-select");
    if (depthSel) depthSel.addEventListener("change", () => {
        _moduleDepth = depthSel.value === "auto" ? "auto"
                     : (parseInt(depthSel.value, 10) || 0);
        // leaving depth mode leaves the module you were inside with it
        if (_moduleDepth === 0) _scope = "";
        loadGraph(currentFn);
    });

    const aliasCb = document.getElementById("gopt-alias-nodes");
    if (aliasCb) aliasCb.addEventListener("change", () => {
        _aliasEnabled = aliasCb.checked;
        // canvas-only, so the graph data in hand is still correct — just redraw
        if (currentGraphData) renderGraph(currentGraphData);
    });

    const expandAllBtn = document.getElementById("gopt-chains-expand");
    if (expandAllBtn) expandAllBtn.addEventListener("click", () => {
        (currentGraphData ? currentGraphData.nodes : [])
            .filter(n => n.is_chain)
            .forEach(n => _expandedChains.add(n.id));
        loadGraph(currentFn);
    });
    const collapseAllBtn = document.getElementById("gopt-chains-collapse");
    if (collapseAllBtn) collapseAllBtn.addEventListener("click", () => {
        _expandedChains.clear();
        loadGraph(currentFn);
    });

    // show-bent toggle
    document.getElementById("show-bent-btn").addEventListener("click", _toggleShowBent);

    document.getElementById("save-config-btn").addEventListener("click", async () => {
        const btn = document.getElementById("save-config-btn");
        btn.disabled = true;
        try {
            const r = await fetch("/api/bending/config/tbconfig/");
            if (!r.ok) { const d = await r.json(); { const _e = new Error(d.error || r.statusText); _e.traceback = d.traceback || ""; throw _e; } }
            const blob = await r.blob();
            const url  = URL.createObjectURL(blob);
            const a = Object.assign(document.createElement("a"), { href: url, download: "bending_config.tbconfig" });
            document.body.appendChild(a); a.click(); document.body.removeChild(a);
            setTimeout(() => URL.revokeObjectURL(url), 1000);
            showToast("ok", "BendingConfig saved");
        } catch (err) { showToast("error", "Save failed: " + err.message, err.traceback); }
        finally { btn.disabled = false; }
    });

    document.getElementById("export-script-btn").addEventListener("click", async () => {
        const btn = document.getElementById("export-script-btn");
        btn.disabled = true; btn.textContent = "⏳…";
        try {
            const r = await fetch("/api/bending/export/", { method: "POST" });
            if (!r.ok) { const d = await r.json(); { const _e = new Error(d.error || r.statusText); _e.traceback = d.traceback || ""; throw _e; } }
            const blob = await r.blob();
            const url  = URL.createObjectURL(blob);
            const a = Object.assign(document.createElement("a"), { href: url, download: "bended_module.pt" });
            document.body.appendChild(a); a.click(); document.body.removeChild(a);
            setTimeout(() => URL.revokeObjectURL(url), 1000);
            showToast("ok", "TorchScript exported");
        } catch (err) { showToast("error", "Export failed: " + err.message, err.traceback); }
        finally { btn.disabled = false; btn.textContent = "script ⬇"; }
    });

    // keep locators in sync while panning/zooming
    cy.on("viewport", () => {
        if (_showBentMode) _updateBentLocators();
        _updateNodeInfoOverlay();
    });

    // ── edge hover tooltip ────────────────────────────────────────────────────
    let _edgeTip = null;
    cy.on("mouseover", "edge", (evt) => {
        const e = evt.target;
        const src = e.source().data("label") || e.source().id();
        const tgt = e.target().data("label") || e.target().id();
        if (!_edgeTip) {
            _edgeTip = document.createElement("div");
            _edgeTip.id = "edge-tooltip";
            document.body.appendChild(_edgeTip);
        }
        const lbl = e.data("label");
        _edgeTip.textContent = lbl ? `${src} →[${lbl}]→ ${tgt}` : `${src} → ${tgt}`;
        _edgeTip.style.display = "block";
    });
    cy.on("mousemove", "edge", (evt) => {
        if (!_edgeTip) return;
        _edgeTip.style.left = (evt.originalEvent.clientX + 12) + "px";
        _edgeTip.style.top  = (evt.originalEvent.clientY + 8)  + "px";
    });
    cy.on("mouseout", "edge", () => { if (_edgeTip) _edgeTip.style.display = "none"; });

    // ── node info tooltip (hover mode) ────────────────────────────────────────
    let _nodeTip = null;
    cy.on("mouseover", "node", (evt) => {
        const n = evt.target;
        if (n.data("is_compound")) return;
        const d = n.data();
        const lines = [];
        if (_goptNodeInfoMode === "hover" && d.shape && d.shape.length)
            lines.push(`<span class="nit-shape">[${d.shape.join("×")}]</span>`);
        // what the model says this is — shown whatever the node-info setting
        _annotationsFor(d.id).forEach(a => {
            const head = _annotTitle(a);
            const desc = a.description && a.description !== head ? a.description : "";
            lines.push(`<span class="nit-annot">✎ <b>${_escAnnot(head)}</b>${desc ? "<br>" + _escAnnot(desc) : ""}</span>`);
        });
        if (!lines.length) return;
        if (!_nodeTip) {
            _nodeTip = document.createElement("div");
            _nodeTip.id = "node-info-tooltip";
            document.body.appendChild(_nodeTip);
        }
        _nodeTip.innerHTML = lines.join("<br>");
        _nodeTip.style.display = "block";
    });
    cy.on("mousemove", "node", (evt) => {
        if (!_nodeTip) return;
        _nodeTip.style.left = (evt.originalEvent.clientX + 14) + "px";
        _nodeTip.style.top  = (evt.originalEvent.clientY + 8)  + "px";
    });
    cy.on("mouseout", "node", () => { if (_nodeTip) _nodeTip.style.display = "none"; });

    // ── graph options panel ───────────────────────────────────────────────────
    const _gOptBtn   = document.getElementById("graph-options-btn");
    const _gOptPanel = document.getElementById("graph-options-panel");
    _gOptBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const open = _gOptPanel.style.display === "none";
        _gOptPanel.style.display = open ? "" : "none";
        _gOptBtn.classList.toggle("active", open);
    });
    document.addEventListener("click", (e) => {
        if (!_gOptPanel.contains(e.target) && e.target !== _gOptBtn) {
            _gOptPanel.style.display = "none";
            _gOptBtn.classList.remove("active");
        }
    });

    document.getElementById("gopt-edge-labels").addEventListener("change", function () {
        if (this.checked) {
            cy.style().selector("edge").style({ "label": "data(label)", "font-size": "8px", "color": "#6e6e73", "text-background-opacity": 0, "text-margin-y": -6 }).update();
        } else {
            cy.style().selector("edge").style({ "label": "" }).update();
        }
    });

    document.getElementById("gopt-node-info").addEventListener("change", function () {
        const wasPermament = _goptNodeInfoMode === "permanent";
        _goptNodeInfoMode = this.value;
        if (_nodeTip) _nodeTip.style.display = "none";
        _updateNodeInfoOverlay();
        // Spacing changes when entering or leaving permanent mode — re-run layout.
        if (wasPermament !== (_goptNodeInfoMode === "permanent")) runLayout();
    });
    document.getElementById("gopt-node-info-shape").addEventListener("change", function () {
        _goptNodeInfoShape = this.checked;
        _updateNodeInfoOverlay();
    });

    // compound block opacity slider
    (function () {
        const sl  = document.getElementById("gopt-compound-opacity");
        const lbl = document.getElementById("gopt-compound-opacity-val");
        if (!sl) return;
        sl.addEventListener("input", () => {
            const v = parseFloat(sl.value);
            lbl.textContent = Math.round(v * 100) + "%";
            if (cy) {
                cy.nodes("[?is_compound]").style("background-opacity", v);
            }
        });
    })();

    // layout direction
    document.getElementById("layout-select").addEventListener("change", runLayout);

    // module depth
    document.getElementById("depth-select").addEventListener("change", () => {
        cy.nodes('[?is_compound]').remove();
        addModulePanes();
    });

    // op filters
    document.querySelectorAll("#op-filters input").forEach((inp) => {
        inp.addEventListener("change", applyFilters);
    });

    // expand modal
    document.querySelectorAll("#update-mode-bar .update-mode-btn").forEach(b =>
        b.addEventListener("click", () => { if (b.dataset.mode !== _bendingUpdateMode) _setBendingMode(b.dataset.mode); }));
    _syncUpdateModeBar();
    document.getElementById("viz-expand-btn").addEventListener("click", openExpandModal);
    document.getElementById("viz-save-btn").addEventListener("click", (e) => {
        if (!currentVizNode) return;
        _showActSaveMenu(e, currentVizNode.id || currentVizNode.label);
    });
    document.getElementById("viz-modal-close").addEventListener("click", () => {
        document.getElementById("viz-modal").style.display = "none";
        _purgePlotlyIn(document.getElementById("viz-modal-wrap"));
    });
    document.getElementById("viz-modal-backdrop").addEventListener("click", () => {
        document.getElementById("viz-modal").style.display = "none";
        _purgePlotlyIn(document.getElementById("viz-modal-wrap"));
    });

    // input panel
    document.getElementById("input-panel-btn").addEventListener("click", toggleInputPanel);
    const staleBtn = document.getElementById("iface-retrace");
    if (staleBtn) staleBtn.addEventListener("click", () => {
        if (!hasAnyInput()) { toggleInputPanel(); showToast("info", "Add an input, then retrace"); return; }
        _ifaceStale = false; _syncIfaceStale();
        retrace(currentFn);
    });

    const cbBtn = document.getElementById("callback-panel-btn");
    if (cbBtn) cbBtn.addEventListener("click", () => toggleCallbackPanel());
    const cbClose = document.getElementById("close-callback-panel");
    if (cbClose) cbClose.addEventListener("click", () => toggleCallbackPanel(false));
    document.getElementById("close-input-panel").addEventListener("click", () => {
        setInputPanelExpanded(false);      // closing the window closes it for good
        document.getElementById("input-panel").classList.add("hidden");
        document.getElementById("input-panel-btn").classList.remove("active");
    });
    document.getElementById("retrace-btn").addEventListener("click", () => retrace(currentFn));
    document.getElementById("topbar-retrace-btn").addEventListener("click", () => {
        if (!hasAnyInput()) { toggleInputPanel(); return; }
        if (currentFn) retrace(currentFn);
    });

    // pin dashboard
    document.getElementById("viz-pin-btn").addEventListener("click", () => {
        if (!currentVizData) return;
        addPin(currentVizData.label, currentVizData.data, currentVizData.phNodeId || null);
        togglePinDashboard(true);
    });
    document.getElementById("pin-backdrop").addEventListener("click", () => togglePinDashboard(false));
    document.getElementById("pin-dashboard-close").addEventListener("click", () => togglePinDashboard(false));
    document.getElementById("pin-sync-batch-btn").addEventListener("click", function () {
        _syncBatchEnabled = !_syncBatchEnabled;
        this.classList.toggle("active", _syncBatchEnabled);
        this.title = _syncBatchEnabled ? "Batch sync ON — click to disable" : "Sync batch indices across all cards";
    });
    document.getElementById("pin-retrace-btn").addEventListener("click", () => {
        if (currentFn) retrace(currentFn);
    });
    document.getElementById("pin-clear-btn").addEventListener("click", clearPins);
    const _pinZoomSlider = document.getElementById("pin-tile-zoom");
    const _pinZoomVal    = document.getElementById("pin-tile-zoom-val");
    _pinZoomSlider.addEventListener("input", () => {
        _pinTileZoom = parseFloat(_pinZoomSlider.value);
        _pinZoomVal.textContent = _pinTileZoom.toFixed(2).replace(/\.?0+$/, "") + "×";
        clearTimeout(_pinZoomSlider._debounce);
        _pinZoomSlider._debounce = setTimeout(_rerenderAllGridCards, 80);
    });
    document.getElementById("pin-items").addEventListener("contextmenu", (e) => {
        e.preventDefault();
        if (Date.now() - _lastCardDragEndTime > 250)
            _openPinCtxMenu(e.clientX, e.clientY);
    });

    // ── auto-show panes on edge hover ─────────────────────────────────────────
    _initPaneFoldBtns();

    // activation modal
    document.getElementById("act-modal-close").addEventListener("click", () => {
        document.getElementById("act-modal").style.display = "none";
    });
    document.getElementById("act-modal-backdrop").addEventListener("click", () => {
        document.getElementById("act-modal").style.display = "none";
    });

    // ── arrow-key graph navigation ─────────────────────────────────────────────
    let _navNodes = []; // ordered non-compound nodes for arrow navigation

    function _navTopoNodes() {
        if (!currentGraphData) return [];
        const skipWeights = document.getElementById("gopt-nav-skip-weights");
        const skip = skipWeights ? skipWeights.checked : true;
        let base = currentGraphData.nodes.filter(n => !n.is_compound && !(skip && n.op === "get_attr"));
        // An active search is a navigation scope: the arrows walk its matches
        // and nothing else, so the prev/next hints in the panel stay honest
        // about where the keys will actually take you.
        if (_searchMatchSet) base = base.filter(n => _searchMatchSet.has(n.label));
        return _computeNavNodes(base);
    }

    // The scope moved under us; the stored index no longer points anywhere real.
    _navResetForScope = function () {
        if (_navIdx < 0) return;
        const current = _navNodes[_navIdx];
        _navNodes = _navTopoNodes();
        const idx = current ? _navNodes.findIndex(nd => nd.id === current.id) : -1;
        if (idx >= 0) { _navIdx = idx; _navShow(idx); return; }
        // where we were is outside the new scope: step back to unpositioned
        _navIdx = -1;
        _navHide();
    };

    // Jump to a node by id, honouring the scope: outside it, ask before leaving.
    _navGoToNode = function (nodeId, nodeLabel) {
        const label = nodeLabel || nodeId;
        if (!_inSearchScope(label)) {
            const ok = confirm(
                `'${label}' is outside the current search ("${_searchQuery}").\n\n`
                + `Clear the search and go there?`);
            if (!ok) return;
            const inp = document.querySelector(".act-search-input");
            if (inp) { inp.value = ""; inp.dispatchEvent(new Event("input")); }
            else _setSearchScope(null, "");
        }
        const target = cy.$id(nodeId);
        if (!target || target.empty()) return;
        cy.animate({ center: { eles: target }, zoom: cy.zoom() }, { duration: 250 });
        _navNodes = _navTopoNodes();
        const idx = _navNodes.findIndex(nd => nd.id === nodeId);
        if (idx >= 0) { _navIdx = idx; _navShow(idx); }
        onNodeClick(target);
    };

    function _navIsForward(code) {
        const rankDir = (document.getElementById("layout-select") || {}).value || "TB";
        return rankDir === "LR"
            ? (code === "ArrowRight" || code === "ArrowDown")
            : (code === "ArrowDown"  || code === "ArrowRight");
    }
    function _navIsBackward(code) {
        const rankDir = (document.getElementById("layout-select") || {}).value || "TB";
        return rankDir === "LR"
            ? (code === "ArrowLeft" || code === "ArrowUp")
            : (code === "ArrowUp"   || code === "ArrowLeft");
    }

    function _navShow(idx) {
        _navIdx = idx;
        const n = _navNodes[idx];
        if (!n) return;

        const prevNode = _navNodes[idx - 1];
        const nextNode = _navNodes[idx + 1];

        // Pan to center the current node in the *visible* graph area — zoom stays fixed.
        // #cy fills the full window; sidebar (175px) and details panel (230px) overlay it.
        // The nav panel floats near the node (not blocking it), so exclude it from visH.
        const cyNode = cy.$id(n.id);
        if (cyNode && !cyNode.empty()) {
            const cyEl        = cy.container();
            const cyRect      = cyEl.getBoundingClientRect();
            const visH        = cyRect.height;
            const totalW = cyEl.offsetWidth;
            const sidebarFolded = document.getElementById("sidebar")?.classList.contains("pane-folded");
            const detailsFolded = document.getElementById("details-panel")?.classList.contains("pane-folded");
            const leftInset  = sidebarFolded  ? 0 : 175;
            const rightInset = detailsFolded  ? 0 : 230;
            const visCenterX = leftInset + (totalW - leftInset - rightInset) / 2;
            const zoom   = cy.zoom();
            const bb     = cyNode.boundingBox();
            const modelCx = (bb.x1 + bb.x2) / 2;
            const modelCy = (bb.y1 + bb.y2) / 2;
            cy.animate({
                pan: { x: visCenterX - modelCx * zoom, y: visH / 2 - modelCy * zoom },
            }, { duration: 150 });
            // onNodeClick updates the details panel and fetches activation, but also
            // undims ancestors/descendants. Re-apply a clean nav dimming pass afterwards:
            // only the current node (and prev/next, undimmed by _placeLocator) stay bright.
            onNodeClick(cyNode);
            cy.elements().addClass("dimmed");
            cy.elements().removeClass("ancestor-node ancestor-edge descendant-node descendant-edge selected");
            cyNode.removeClass("dimmed");
            cyNode.connectedEdges().removeClass("dimmed");
            cyNode.addClass("nav-current");
        }

        // Build floating panel content
        const panel = document.getElementById("graph-nav-panel");
        document.getElementById("graph-nav-label").textContent = n.label || n.id;
        const shapeEl = document.getElementById("graph-nav-shape");
        shapeEl.textContent = n.shape && n.shape.length ? `[${n.shape.join(" × ")}]` : "";
        document.getElementById("graph-nav-idx").textContent = `${idx + 1}/${_navNodes.length}`;
        document.getElementById("graph-nav-op").textContent = n.op || "";
        document.getElementById("graph-nav-target").textContent = n.target || "";

        // Prev / next context hints
        const prevHintEl = document.getElementById("graph-nav-prev-hint");
        const nextHintEl = document.getElementById("graph-nav-next-hint");
        prevHintEl.style.display = "none";
        nextHintEl.style.display = "none";
        if (prevNode) {
            const argIdx = _findArgIdx(n.args, prevNode.id);
            _buildNavHint(prevHintEl, prevNode, "prev",
                argIdx !== null ? { op: n.op || "", arg: `arg${argIdx}` } : null);
        }
        if (nextNode) {
            const argIdx = _findArgIdx(nextNode.args, n.id);
            _buildNavHint(nextHintEl, nextNode, "next",
                argIdx !== null ? { op: nextNode.op || "", arg: `arg${argIdx}` } : null);
        }

        const argsEl = document.getElementById("graph-nav-args");
        argsEl.innerHTML = "";
        const nodeById = {};
        if (currentGraphData) currentGraphData.nodes.forEach(nd => { nodeById[nd.id] = nd; });

        function _argRow(label, arg) {
            if (!arg) return;
            const row = document.createElement("div");
            row.className = "graph-nav-arg";
            const nm = document.createElement("span");
            nm.className = "graph-nav-arg-name";
            nm.textContent = label;
            const sh = document.createElement("span");
            sh.className = "graph-nav-arg-shape";
            // a node argument is somewhere you can go — clicking walks there,
            // asking first if it sits outside the active search
            const _goBtn = (nodeId, label) => {
                const b = document.createElement("button");
                b.className = "graph-nav-goto";
                b.textContent = "→";
                b.title = _inSearchScope(label)
                    ? `Go to ${label}`
                    : `Go to ${label} (outside the current search)`;
                if (!_inSearchScope(label)) b.classList.add("out-of-scope");
                b.addEventListener("click", (e) => {
                    e.stopPropagation();
                    if (_navGoToNode) _navGoToNode(nodeId, label);
                });
                return b;
            };

            if (arg.type === "node") {
                const src = nodeById[arg.name];
                const nodeName = src ? (src.label || arg.name) : arg.name;
                const shapeStr = src && src.shape && src.shape.length ? ` [${src.shape.join("×")}]` : "";
                sh.textContent = nodeName + shapeStr;
                row.appendChild(nm);
                row.appendChild(sh);
                row.appendChild(_goBtn(arg.name, nodeName));
                argsEl.appendChild(row);
                return;
            } else if (arg.type === "list") {
                const nodeItems = (arg.items || []).filter(i => i.type === "node");
                const shapes = nodeItems
                    .map(i => { const s = nodeById[i.name]; return s && s.shape && s.shape.length ? `[${s.shape.join("×")}]` : i.name; });
                sh.textContent = shapes.length ? shapes.join(", ") : "list";
                row.appendChild(nm);
                row.appendChild(sh);
                nodeItems.forEach((i) => {
                    const src = nodeById[i.name];
                    row.appendChild(_goBtn(i.name, src ? (src.label || i.name) : i.name));
                });
                argsEl.appendChild(row);
                return;
            } else {
                sh.textContent = String(arg.value || "");
            }
            row.appendChild(nm);
            row.appendChild(sh);
            argsEl.appendChild(row);
        }

        (n.args || []).forEach((arg, i) => _argRow(`[${i}]`, arg));
        Object.entries(n.kwargs || {}).forEach(([k, arg]) => _argRow(k, arg));

        panel.style.display = "flex";
        // update nav locators and position panel after a tick so layout is stable
        requestAnimationFrame(() => {
            _updateNavLocators(prevNode || null, nextNode || null);
            _navPositionPanel(cyNode);
        });

        // optional inline code preview
        _navShowCode(n);
    }

    // Reposition next chip and nav locators on pan/zoom (attach once cy exists)
    let _navViewportBound = false;
    function _ensureNavViewport() {
        if (_navViewportBound || !cy) return;
        _navViewportBound = true;
        cy.on("viewport", () => {
            if (_navIdx < 0) return;
            _updateNavLocators(_navNodes[_navIdx - 1] || null, _navNodes[_navIdx + 1] || null);
            const curNode = _navNodes[_navIdx];
            if (curNode) _navPositionPanel(cy.$id(curNode.id));
        });
    }

    let _navCodeOpen = false;  // tracks whether the user wants the code panel open

    function _navSetCodeBtn(active) {
        const btn = document.getElementById("graph-nav-code-btn");
        if (btn) btn.classList.toggle("active", active);
    }

    function _navShowCode(n, forceOpen) {
        const checkbox = document.getElementById("gopt-nav-show-code");
        const codeWrap = document.getElementById("graph-nav-code");
        const panel    = document.getElementById("graph-nav-panel");
        if (!codeWrap) return;
        if (forceOpen !== undefined) _navCodeOpen = forceOpen;
        const shouldShow = _navCodeOpen || (checkbox && checkbox.checked);
        _navSetCodeBtn(shouldShow);
        if (!shouldShow) {
            codeWrap.style.display = "none";
            if (panel) panel.classList.remove("code-open");
            return;
        }
        if (!n) {
            codeWrap.style.display = "none";
            if (panel) panel.classList.remove("code-open");
            return;
        }
        if (panel) { panel.classList.add("code-open"); _navResetPanelPosition(); }
        codeWrap.style.display = "flex";
        codeWrap.innerHTML = '<span class="viz-loading">…</span>';
        fetch(`/api/node_source/${encodeURIComponent(currentFn)}/${encodeURIComponent(n.id)}/`)
            .then(r => r.json())
            .then(d => {
                if (!d || d.error || !d.lines || !d.lines.length) {
                    codeWrap.style.display = "none";
                    if (panel) panel.classList.remove("code-open");
                    _navSetCodeBtn(false);
                    return;
                }
                const header = document.createElement("div");
                header.className = "graph-nav-code-header";
                const nameSpan = document.createElement("span");
                nameSpan.textContent = n.label || n.id;
                const fileSpan = document.createElement("span");
                fileSpan.className = "graph-nav-code-file";
                fileSpan.textContent = d.file ? d.file.replace(/.*[/\\]/, "") + (d.highlight_line ? `:${d.highlight_line}` : "") : "";
                header.appendChild(nameSpan);
                header.appendChild(fileSpan);
                codeWrap.innerHTML = "";
                codeWrap.appendChild(header);
                const body = document.createElement("div");
                body.className = "graph-nav-code-body";
                codeWrap.appendChild(body);
                _renderSourceBlock(d, body);
            })
            .catch(() => { codeWrap.style.display = "none"; if (panel) panel.classList.remove("code-open"); _navSetCodeBtn(false); });
    }

    document.getElementById("graph-nav-code-btn").addEventListener("click", () => {
        if (_navIdx < 0 || !_navNodes[_navIdx]) return;
        _navShowCode(_navNodes[_navIdx], !_navCodeOpen);
    });

    function _navHide() {
        ["graph-nav-panel", "graph-nav-code"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = "none";
        });
        const panel = document.getElementById("graph-nav-panel");
        if (panel) { panel.classList.remove("code-open"); _navResetPanelPosition(); }
        _navSetCodeBtn(false);
        _navCodeOpen = false;
        _navIdx = -1;
        _clearNavLocators();
    }

    // spacebar toggles pin dashboard; Tab opens activation browser; arrow keys navigate detail view
    document.addEventListener("keydown", (e) => {
        const tag = document.activeElement && document.activeElement.tagName;
        const typing = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";

        if (e.code === "Escape") {
            document.getElementById("act-modal").style.display = "none";
            document.getElementById("help-modal").style.display = "none";
            _closeNodeCtxMenu();
            _closeBendDialog();
            const srcModal = document.getElementById("source-modal");
            if (srcModal) srcModal.style.display = "none";
            _navHide();
            return;
        }

        // graph arrow-key navigation (only when no modal is open and not typing)
        const _anyModalOpen = document.getElementById("act-modal").style.display !== "none";
        if (!typing && !_anyModalOpen && cy && currentGraphData) {
            const fwd = _navIsForward(e.code);
            const bwd = _navIsBackward(e.code);
            if (fwd || bwd) {
                e.preventDefault();
                _ensureNavViewport();
                _navNodes = _navTopoNodes();
                if (_navNodes.length === 0) return;
                if (_navIdx < 0) {
                    // Find currently selected node in cy to seed index
                    const sel = cy.$(".selected").first();
                    if (sel && !sel.empty()) {
                        const idx = _navNodes.findIndex(nd => nd.id === sel.data("id"));
                        _navIdx = idx >= 0 ? idx : 0;
                    } else {
                        _navIdx = fwd ? 0 : _navNodes.length - 1;
                    }
                } else {
                    _navIdx = fwd
                        ? Math.min(_navNodes.length - 1, _navIdx + 1)
                        : Math.max(0, _navIdx - 1);
                }
                _navShow(_navIdx);
                return;
            }
        }

        if (e.code === "Tab" && !e.ctrlKey && !e.metaKey && !typing) {
            e.preventDefault();
            const actModal = document.getElementById("act-modal");
            if (actModal.style.display !== "none") {
                actModal.style.display = "none";
            } else if (currentGraphData) {
                openActModal(currentGraphData);
            }
            return;
        }

        // arrow navigation in detail mode
        const actModal = document.getElementById("act-modal");
        if (_actModalMode === "detail" && actModal.style.display !== "none" && !typing) {
            if (e.code === "ArrowLeft" || e.code === "ArrowUp") {
                e.preventDefault();
                _actDetailIdx = Math.max(0, _actDetailIdx - 1);
                _showActDetail(_actDetailIdx);
                return;
            }
            if (e.code === "ArrowRight" || e.code === "ArrowDown") {
                e.preventDefault();
                _actDetailIdx = Math.min(_actFiltered.length - 1, _actDetailIdx + 1);
                _showActDetail(_actDetailIdx);
                return;
            }
        }

        if (e.code !== "Space") return;
        if (typing) return;
        e.preventDefault();
        togglePinDashboard();
    });

    // help modal
    document.getElementById("help-btn").addEventListener("click", () => {
        document.getElementById("help-modal").style.display = "flex";
    });
    document.getElementById("help-modal-close").addEventListener("click", () => {
        document.getElementById("help-modal").style.display = "none";
    });
    document.getElementById("help-modal").addEventListener("click", (e) => {
        if (e.target === document.getElementById("help-modal"))
            document.getElementById("help-modal").style.display = "none";
    });

    // persist pin pages to localStorage on unload; also flush to server via sendBeacon
    window.addEventListener("beforeunload", () => {
        clearTimeout(_clientStatePushTimer);
        _persistPinPages();
        if (currentModelName) {
            const toSave = pinPages.map(page =>
                page.map(({ data, originalData, ...rest }) => rest)
            );
            const state = {
                pins:      { pages: toSave, current: currentPinPage },
                favs:      [..._favNodes],
                tags:      _nodeTags,
                bookmarks: _loadBookmarks(),
                vizStates: _vizStatesForSave(),
                inputs:    _serializeInputSetsSync(),
            };
            navigator.sendBeacon(`/api/client-state/?name=${encodeURIComponent(currentModelName)}`,
                new Blob([JSON.stringify(state)], { type: "application/json" }));
        }
    });

    // Play mode reads the shared input store on load; the debounced write may
    // still be pending when the link is clicked. Flush it first, then navigate,
    // so a sound loaded a moment ago is not lost on the way over.
    // Generate mode reads the same store, so its link gets the same flush.
    ["play-mode-btn", "generate-mode-btn"].forEach((id) => {
        const btn = document.getElementById(id);
        if (!btn) return;
        btn.addEventListener("click", (e) => {
            if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) return;  // new tab
            e.preventDefault();
            const href = btn.getAttribute("href") || "/play/";
            clearTimeout(_persistInputsTimer);
            Promise.resolve(_doPersistInputsForPlay(currentModelName))
                .catch(() => {})
                .finally(() => { window.location.href = href; });
        });
    });

    // wire up TBViews state-change hook so viz preferences are persisted
    if (window.TBViews) TBViews.onStateChange = () => _scheduleClientStatePush();

    const _selClear = document.getElementById("sel-scope-clear");
    if (_selClear) _selClear.addEventListener("click", _clearScope);

    // nav scope selector
    document.querySelectorAll(".nav-scope-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".nav-scope-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            _navScope.mode = btn.dataset.scope;
            _navScope.listId = null;
            _navIdx = -1;
            _navHide();
            _scopeChanged(false);
        });
    });
    const _navScopeListSel = document.getElementById("nav-scope-list-sel");
    if (_navScopeListSel) {
        _navScopeListSel.addEventListener("change", () => {
            _navScope.listId = _navScopeListSel.value || null;
            _navIdx = -1;
            _scopeChanged(false);
        });
    }

    // restore pinned nodes from previous session then fetch server state;
    // chain loadGraph after client state so saved inputs don't race with _applyDefaultInputs.
    _loadPinPages(currentModelName);
    renderPinTabs();
    renderCurrentPage();
    _updatePinCount();
    _fetchAndApplyClientState(currentModelName)
        .then(() => _restoreSharedInputs(currentModelName))
        .then(() => {
            if (INITIAL_FN) loadGraph(INITIAL_FN);
        });

    // bending: fetch initial state + wire up act-modal bending view controls
    _fetchBendings();

    document.getElementById("act-bend-mode-select").addEventListener("change", (e) => {
        _setBendingMode(e.target.value);
    });
    document.getElementById("act-bend-threshold").addEventListener("change", async (e) => {
        const v = parseFloat(e.target.value);
        if (!isNaN(v) && v > 0) {
            _bendingAutoThreshMs = v;
            try {
                await fetch("/api/bending/mode/", {
                    method:  "PATCH",
                    headers: {"Content-Type": "application/json"},
                    body:    JSON.stringify({ auto_threshold_ms: v }),
                });
            } catch (_) {}
        }
    });
    document.getElementById("act-bend-export-btn").addEventListener("click", async () => {
        const btn = document.getElementById("act-bend-export-btn");
        btn.disabled = true;
        btn.textContent = "⏳ scripting…";
        try {
            const r = await fetch("/api/bending/export/", { method: "POST" });
            if (!r.ok) {
                const d = await r.json();
                { const _e = new Error(d.error || r.statusText); _e.traceback = d.traceback || ""; throw _e; }
            }
            const blob = await r.blob();
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement("a");
            a.href     = url;
            a.download = "bended_module.pt";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            setTimeout(() => URL.revokeObjectURL(url), 1000);
            showToast("ok", "TorchScript exported");
        } catch (err) {
            showToast("error", "Export failed: " + err.message, err.traceback);
        } finally {
            btn.disabled = false;
            btn.textContent = "⬇ export TorchScript";
        }
    });

    // ── bending config save / load (.tbconfig) ────────────────────────────────
    document.getElementById("act-bend-save-btn").addEventListener("click", async () => {
        try {
            const r = await fetch("/api/bending/config/tbconfig/");
            if (!r.ok) throw new Error((await r.json()).error || r.statusText);
            const blob = await r.blob();
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement("a");
            a.href = url; a.download = "bending_config.tbconfig";
            document.body.appendChild(a); a.click(); document.body.removeChild(a);
            setTimeout(() => URL.revokeObjectURL(url), 1000);
            showToast("ok", "Config saved (.tbconfig)");
        } catch (err) { showToast("error", "Save failed: " + err.message, err.traceback); }
    });

    const _loadInput = document.getElementById("act-bend-load-input");
    _loadInput.accept = ".tbconfig";
    document.getElementById("act-bend-load-btn").addEventListener("click", () => _loadInput.click());
    _loadInput.addEventListener("change", async () => {
        const file = _loadInput.files[0];
        if (!file) return;
        _loadInput.value = "";
        try {
            const bytes = await file.arrayBuffer();
            const r = await fetch("/api/bending/config/tbconfig/import/", {
                method: "POST",
                headers: { "Content-Type": "application/octet-stream" },
                body: bytes,
            });
            const d = await r.json();
            if (!r.ok) { const _e = new Error(d.error || r.statusText); _e.traceback = d.traceback || ""; throw _e; }
            _syncBendingState(d);
            _refreshBentNodeStyles();
            _refreshAllBendingUI();
            showToast("ok", `Config loaded — ${d.bindings.length} binding(s)`);
        } catch (err) { showToast("error", "Load failed: " + err.message, err.traceback); }
    });
});
