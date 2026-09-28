/* TorchBend — shared node search (TBSearch)
 *
 * The editor's activation list grew a real query language: field filters, shape
 * patterns, aliases, favourites and saved bookmarks. Play mode needs to find a
 * node too — to bend it — and re-inventing a lesser search there would mean two
 * ways to say the same thing. This is the one implementation, plus a ready-made
 * picker widget built on it.
 *
 * Favourites, tags and bookmarks live under the same localStorage keys the
 * editor uses, so a node starred in one place is starred in the other.
 */
(function () {
  "use strict";
  const TBSearch = {};

  function el(tag, cls, txt) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }

  // ── shared storage ───────────────────────────────────────────────────────────
  const FAV_KEY = "torchbend_fav_nodes";
  const TAG_KEY = "tb_act_tags";
  const BM_KEY  = "tb_act_bookmarks";
  TBSearch.FAV_KEY = FAV_KEY;
  TBSearch.TAG_KEY = TAG_KEY;
  TBSearch.BM_KEY = BM_KEY;

  TBSearch.loadFavs = function () {
    try { return new Set(JSON.parse(localStorage.getItem(FAV_KEY) || "[]")); }
    catch (e) { return new Set(); }
  };
  TBSearch.saveFavs = function (set) {
    try { localStorage.setItem(FAV_KEY, JSON.stringify([...set])); } catch (e) { /* quota */ }
  };
  TBSearch.isFav = (label) => TBSearch.loadFavs().has(label);
  TBSearch.toggleFav = function (label) {
    const favs = TBSearch.loadFavs();
    if (favs.has(label)) favs.delete(label); else favs.add(label);
    TBSearch.saveFavs(favs);
    return favs.has(label);
  };
  TBSearch.loadTags = function () {
    try { return JSON.parse(localStorage.getItem(TAG_KEY) || "{}") || {}; }
    catch (e) { return {}; }
  };
  TBSearch.loadBookmarks = function () {
    try { return JSON.parse(localStorage.getItem(BM_KEY) || "[]"); }
    catch (e) { return []; }
  };
  TBSearch.saveBookmarks = function (list) {
    try { localStorage.setItem(BM_KEY, JSON.stringify(list)); } catch (e) { /* quota */ }
  };

  // ── shape patterns ───────────────────────────────────────────────────────────
  TBSearch.shapeStr = (n) => (n.shape && n.shape.length ? n.shape.join("×") : "");

  TBSearch.isShapePattern = function (tok) {
    if (/^\d+$/.test(tok.trim())) return true;   // bare integer → "any dim equals N"
    return /[xX×]/.test(tok) || tok.includes("...") || tok.includes("?") || tok.includes("|");
  };

  // dimensions separated by x / X / × ;  ? = one dim, ... = any number, A|B = either
  function matchShapePattern(shape, pat) {
    if (!shape || !shape.length) return false;
    const parts = pat.trim().split(/\s*[xX×]\s*/);
    function match(si, pi) {
      if (pi >= parts.length) return si >= shape.length;
      const p = parts[pi].trim();
      if (p === "...") {
        for (let skip = 0; si + skip <= shape.length; skip++) {
          if (match(si + skip, pi + 1)) return true;
        }
        return false;
      }
      if (si >= shape.length) return false;
      if (p === "?") return match(si + 1, pi + 1);
      const opts = p.split("|").map((x) => parseInt(x.trim(), 10));
      if (!opts.some((o) => o === shape[si])) return false;
      return match(si + 1, pi + 1);
    }
    return match(0, 0);
  }

  TBSearch.matchShapeQuery = function (shape, query) {
    if (!shape || !shape.length) return false;
    const q = query.trim();
    if (/^\d+$/.test(q)) {                       // bare integer matches any dim
      const n = parseInt(q, 10);
      return shape.some((d) => d === n);
    }
    return matchShapePattern(shape, q);
  };

  // ── numeric comparisons (in:>3, ndim:<=2, …) ─────────────────────────────────
  TBSearch.parseNumericCmp = function (raw) {
    if (!raw) return null;
    const m = raw.match(/^(>=|<=|>|<|=)?(\d+(?:\.\d+)?)$/);
    return m ? { op: m[1] || "=", val: parseFloat(m[2]) } : null;
  };
  TBSearch.testNumericCmp = function (cmp, actual) {
    if (cmp === null || actual === undefined || actual === null) return false;
    switch (cmp.op) {
      case ">":  return actual > cmp.val;
      case ">=": return actual >= cmp.val;
      case "<":  return actual < cmp.val;
      case "<=": return actual <= cmp.val;
      case "=":  return actual === cmp.val;
      default:   return false;
    }
  };

  // ── query parsing ────────────────────────────────────────────────────────────
  TBSearch.parse = function (q) {
    if (!q) return [];
    return q.split(/\s+/).filter(Boolean).map((tok) => {
      if (tok.startsWith("#")) return { field: "alias", alias: tok.slice(1), re: null };
      const m = tok.match(/^(\w+):(.*)$/);
      if (m) {
        const field = m[1].toLowerCase();
        const raw = m[2];
        let re = null;
        try { re = raw ? new RegExp(raw, "i") : null; } catch (e) { /* literal */ }
        return { field, re, raw };
      }
      // shape patterns are recognised without a prefix: 1x?x1024, 2|4×512, …
      if (/^(?:\d+|\?|\.{3}|\d+(?:\|\d+)*)(?:[xX×](?:\d+|\?|\.{3}|\d+(?:\|\d+)*))+$/.test(tok)) {
        return { field: "shape", re: null, raw: tok };
      }
      let re = null;
      try { re = new RegExp(tok, "i"); } catch (e) { /* literal */ }
      return { field: "name", re };
    });
  };

  // ── matching ─────────────────────────────────────────────────────────────────
  // ctx: { aliases, tags, recurrence } — everything the caller knows that a bare
  // node object does not carry.
  function nodeId(n) { return n.id || n.label || n.name; }
  function nodeLabel(n) { return n.label || n.name || n.id; }

  function tagsFor(ctx, id) { return ((ctx && ctx.tags) || {})[id] || []; }

  TBSearch.nodeAliases = function (n, ctx) {
    const id = nodeId(n);
    const names = new Set();
    const aliases = (ctx && ctx.aliases) || null;
    if (aliases) {
      Object.entries(aliases).forEach(([name, members]) => {
        if (members.includes(id)) names.add(name);
      });
    }
    tagsFor(ctx, id).forEach((t) => names.add(t));
    return [...names];
  };

  function matchField(n, field, re, raw, ctx) {
    if (field === "shape" && raw && TBSearch.isShapePattern(raw)) {
      return TBSearch.matchShapeQuery(n.shape, raw);
    }
    if (!re) return true;
    switch (field) {
      case "name":   return re.test(nodeLabel(n));
      case "target": return re.test(n.target || "");
      case "shape":  return re.test(TBSearch.shapeStr(n));
      case "op":     return re.test(n.op || "");
      case "src":    return re.test(n.source_file || "") || re.test(n.source_fn || "");
      // `mod:attn` catches every submodule whose path contains it, so filtering
      // by a parent module keeps its children — which is what you want when you
      // ask for "everything in block 5".
      case "mod":    return re.test(n.module_path || "");
      case "tag":    return tagsFor(ctx, nodeId(n)).some((t) => re.test(t));
      default:       return re.test(nodeLabel(n)) || re.test(n.target || "");
    }
  }

  TBSearch.match = function (n, terms, ctx) {
    ctx = ctx || {};
    return terms.every((t) => {
      if (t.field === "bent" && !t.re) return !!n.has_bending;
      if (t.field === "fav" && !t.re) return TBSearch.isFav(nodeLabel(n));
      if (t.field === "alias") {
        let re = null;
        try { re = t.alias ? new RegExp(t.alias, "i") : null; } catch (e) { /* literal */ }
        const aliases = ctx.aliases || null;
        if (aliases) {
          for (const [name, members] of Object.entries(aliases)) {
            if ((!re || re.test(name)) && members.includes(nodeId(n))) return true;
          }
        }
        return tagsFor(ctx, nodeId(n)).some((tag) => (re ? re.test(tag) : true));
      }
      // `raw` is absent when a caller builds a term by hand rather than parsing
      const raw = (t.raw || "").toLowerCase();
      if (t.field === "trivial") return !!n.is_trivial === (raw !== "no");
      if (t.field === "change")  return !!n.shape_changed === (raw !== "no");
      if (t.field === "in")   return TBSearch.testNumericCmp(TBSearch.parseNumericCmp(t.raw), n.in_degree ?? 0);
      if (t.field === "out")  return TBSearch.testNumericCmp(TBSearch.parseNumericCmp(t.raw), n.out_degree ?? 0);
      if (t.field === "deg")  return TBSearch.testNumericCmp(TBSearch.parseNumericCmp(t.raw),
                                                             (n.in_degree ?? 0) + (n.out_degree ?? 0));
      if (t.field === "recur") {
        const cnt = (n.target && (ctx.recurrence || {})[n.target]) || 1;
        return TBSearch.testNumericCmp(TBSearch.parseNumericCmp(t.raw), cnt);
      }
      if (t.field === "ndim") {
        return TBSearch.testNumericCmp(TBSearch.parseNumericCmp(t.raw), n.shape ? n.shape.length : 0);
      }
      if (t.field === "has_mod") {
        return !!n.module_path === (raw !== "no");
      }
      if (t.field === "has_alias") {
        return (TBSearch.nodeAliases(n, ctx).length > 0) === (raw !== "no");
      }
      return matchField(n, t.field, t.re, t.raw, ctx);
    });
  };

  // A module contains its submodules: selecting `transformer.h.0` keeps
  // `transformer.h.0.attn.c_attn`.  Prefix alone is not enough -- `h.1` must not
  // swallow `h.10` -- so the boundary dot is part of the test.
  TBSearch.inModule = function (path, scope) {
    if (!scope) return true;
    if (!path) return false;
    return path === scope || path.startsWith(scope + ".");
  };

  // Every module path present in a graph payload, with how many nodes sit
  // directly in each, deepest paths included even when nothing is directly in them.
  TBSearch.moduleIndex = function (data) {
    const counts = new Map();
    (data && data.nodes || []).forEach((n) => {
      if (n.is_compound) { if (n.module_path) counts.set(n.module_path, counts.get(n.module_path) || 0); return; }
      if (!n.module_path) return;
      counts.set(n.module_path, (counts.get(n.module_path) || 0) + 1);
    });
    return [...counts.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([path, direct]) => ({ path, direct, depth: path.split(".").length }));
  };

  TBSearch.filter = function (nodes, query, ctx) {
    const terms = TBSearch.parse(query);
    if (!terms.length) return nodes.slice();
    return nodes.filter((n) => {
      try { return TBSearch.match(n, terms, ctx); } catch (e) { return false; }
    });
  };

  TBSearch.SYNTAX_HINT =
    "name · target:conv · op:call_module · mod:transformer.h.5 · shape:1x?x512 · "
    + "1x?x512 · ndim:>2 · in:>1 · out:0 · #alias · tag:mine · src:model.py";

  // ── picker widget ────────────────────────────────────────────────────────────
  // Built from the editor's own activation-browser markup — same ids, same
  // classes, same stylesheet — so picking a node in play mode is the same
  // interface, not a lookalike. What it drops is only what play mode has no use
  // for: the detail viz, the bend view and pinning, which belong to the graph.
  //
  // opts: { nodes, ctx, onPick(node), placeholder, limit }
  TBSearch.picker = function (host, opts) {
    opts = opts || {};
    host.innerHTML = "";
    const mk = (tag, id, cls) => {
      const e = el(tag, cls);
      if (id) e.id = id;
      host.appendChild(e);
      return e;
    };
    const bar = mk("div", "act-modal-search-bar");
    const searchEl = el("input");
    searchEl.id = "act-modal-search";
    searchEl.type = "text";
    searchEl.spellcheck = false;
    searchEl.autocomplete = "off";
    const groupBtn = el("button", null, "⊞ group");
    groupBtn.id = "act-sort-btn";
    const filterBtn = el("button", null, "⊟ filters");
    filterBtn.id = "act-filter-toggle";
    bar.appendChild(searchEl); bar.appendChild(groupBtn); bar.appendChild(filterBtn);
    return TBSearch.bind({
      searchEl, groupBtn, filterBtn,
      bookmarkEl: mk("div", "act-bookmark-bar"),
      panelEl:    mk("div", "act-filter-panel"),
      listEl:     mk("div", "act-modal-list"),
      countEl:    mk("div", null, "act-modal-count"),
      nodes: opts.nodes, ctx: opts.ctx, onPick: opts.onPick,
      placeholder: opts.placeholder, limit: opts.limit,
    });
  };

  // Bind the browser's behaviour to elements that already exist — the shared
  // markup in _act_browser.html, or the ones picker() just made.
  TBSearch.bind = function (o) {
    let nodes = o.nodes || [];
    const ctx = o.ctx || {};
    const limit = o.limit || 400;
    const search = o.searchEl, panel = o.panelEl, list = o.listEl;
    const bmBar = o.bookmarkEl, count = o.countEl || null;
    let favOnly = false, bentOnly = false, grouped = false;
    const activeOps = new Set();
    const collapsed = new Set();
    let chosen = null;

    if (search) {
      search.value = "";
      search.title = TBSearch.SYNTAX_HINT;
      if (o.placeholder) search.placeholder = o.placeholder;
    }

    function renderBookmarks() {
      if (!bmBar) return;
      bmBar.innerHTML = "";
      const bms = TBSearch.loadBookmarks();
      bms.forEach((bm, i) => {
        const chip = el("span", "act-bm-chip");
        const go = el("button", "act-bm-go", bm.name || bm.query);
        go.title = bm.query;
        go.addEventListener("click", () => { search.value = bm.query; render(); });
        const del = el("button", "act-bm-del", "×");
        del.title = "Delete bookmark";
        del.addEventListener("click", () => {
          const next = TBSearch.loadBookmarks();
          next.splice(i, 1);
          TBSearch.saveBookmarks(next);
          renderBookmarks();
        });
        chip.appendChild(go); chip.appendChild(del);
        bmBar.appendChild(chip);
      });
      const add = el("button", "act-bm-add", "＋ save search");
      add.title = "Remember the current search";
      add.addEventListener("click", () => {
        const q = (search.value || "").trim();
        if (!q) return;
        const name = (prompt("Name this search:", q) || "").trim();
        if (!name) return;
        const next = TBSearch.loadBookmarks();
        next.push({ name, query: q });
        TBSearch.saveBookmarks(next);
        renderBookmarks();
      });
      bmBar.appendChild(add);
      bmBar.style.display = "";
    }

    function renderPanel() {
      if (!panel) return;
      panel.innerHTML = "";
      const opRow = el("div", "act-filter-row");
      [...new Set(nodes.map((n) => n.op).filter(Boolean))].sort().forEach((op) => {
        const lbl = el("label", "act-op-filter");
        const chk = el("input");
        chk.type = "checkbox";
        chk.checked = !activeOps.size || activeOps.has(op);
        chk.addEventListener("change", () => {
          // an empty set means "everything"; the first click narrows from there
          if (!activeOps.size) [...new Set(nodes.map((n) => n.op))].forEach((x) => activeOps.add(x));
          if (chk.checked) activeOps.add(op); else activeOps.delete(op);
          render();
        });
        lbl.appendChild(chk);
        lbl.appendChild(document.createTextNode(" " + op));
        opRow.appendChild(lbl);
      });
      panel.appendChild(opRow);

      const chips = el("div", "act-filter-chips");
      const chip = (text, on, color, onClick, title) => {
        const c = el("button", "act-filter-chip" + (on ? " active" : ""), text);
        if (color) c.style.setProperty("--chip-color", color);
        if (title) c.title = title;
        c.addEventListener("click", onClick);
        chips.appendChild(c);
      };
      chip("★ fav", favOnly, "#e6a817", () => { favOnly = !favOnly; renderPanel(); render(); },
           "Only nodes you starred — the editor shares these");
      chip("bent", bentOnly, "#F0AD4E", () => { bentOnly = !bentOnly; renderPanel(); render(); },
           "Only nodes that already carry a bending");
      Object.keys(ctx.aliases || {}).sort().forEach((a) => {
        const tok = "#" + a;
        const active = (search.value || "").includes(tok);
        chip(tok, active, "#89dceb", () => {
          const parts = (search.value || "").split(/\s+/).filter(Boolean);
          search.value = active ? parts.filter((p) => p !== tok).join(" ")
                                : parts.concat(tok).join(" ");
          renderPanel(); render();
        }, `Nodes marked '${a}'`);
      });
      panel.appendChild(chips);
    }

    function makeRow(n) {
      const label = nodeLabel(n);
      const row = el("div", "act-modal-row" + (chosen === label ? " act-selected" : "")
                            + (n.has_bending ? " act-list-bent" : ""));
      row.dataset.name = label;

      const star = el("button", "act-modal-star-btn" + (TBSearch.isFav(label) ? " starred" : ""), "★");
      star.title = "Star this node — the editor's list shares these";
      star.addEventListener("click", (e) => { e.stopPropagation(); TBSearch.toggleFav(label); render(); });
      row.appendChild(star);

      const main = el("div", "act-modal-row-main");
      main.appendChild(el("span", "act-modal-badge", (n.op || "").replace("call_", "")));
      main.appendChild(el("span", "act-modal-name", label));
      TBSearch.nodeAliases(n, ctx).forEach((a) =>
        main.appendChild(el("span", "act-tag-pill", "#" + a)));
      if (n.shape) main.appendChild(el("span", "act-modal-shape", "[" + n.shape.join(", ") + "]"));
      row.appendChild(main);

      const sub = el("div", "act-modal-row-sub");
      sub.textContent = [n.target && n.target !== label ? n.target : null,
                         n.module_path || null,
                         n.source_fn ? `in ${n.source_fn}()` : null].filter(Boolean).join("  ·  ");
      if (sub.textContent) row.appendChild(sub);

      row.addEventListener("click", () => {
        chosen = label;
        render();
        if (o.onPick) o.onPick(n);
      });
      return row;
    }

    function render() {
      const hits = TBSearch.filter(nodes, (search.value || "").trim(), ctx)
        .filter((n) => !favOnly || TBSearch.isFav(nodeLabel(n)))
        .filter((n) => !bentOnly || !!n.has_bending)
        .filter((n) => !activeOps.size || activeOps.has(n.op));
      // starred float to the top, as in the editor
      hits.sort((a, b) => (TBSearch.isFav(nodeLabel(b)) ? 1 : 0) - (TBSearch.isFav(nodeLabel(a)) ? 1 : 0));

      list.innerHTML = "";
      const shown = hits.slice(0, limit);
      if (grouped) {
        const byOp = {};
        shown.forEach((n) => { (byOp[n.op] = byOp[n.op] || []).push(n); });
        Object.keys(byOp).sort().forEach((op) => {
          const hdr = el("div", "act-group-header");
          hdr.appendChild(el("span", "act-group-arrow", collapsed.has(op) ? "▸" : "▾"));
          hdr.appendChild(document.createTextNode(` ${op}  (${byOp[op].length})`));
          hdr.addEventListener("click", () => {
            if (collapsed.has(op)) collapsed.delete(op); else collapsed.add(op);
            render();
          });
          list.appendChild(hdr);
          if (!collapsed.has(op)) byOp[op].forEach((n) => list.appendChild(makeRow(n)));
        });
      } else {
        shown.forEach((n) => list.appendChild(makeRow(n)));
      }
      if (!hits.length) list.appendChild(el("div", "act-modal-empty", "nothing matches"));
      if (count) {
        count.textContent = hits.length > limit
          ? `${limit} of ${hits.length} shown — narrow the search`
          : `${hits.length} of ${nodes.length}`;
      }
    }

    if (search) search.oninput = render;
    if (o.groupBtn) o.groupBtn.onclick = () => {
      grouped = !grouped;
      o.groupBtn.classList.toggle("active", grouped);
      render();
    };
    if (o.filterBtn) o.filterBtn.onclick = () => {
      panel.classList.toggle("hidden");
      o.filterBtn.classList.toggle("active", !panel.classList.contains("hidden"));
    };

    renderBookmarks();
    renderPanel();
    render();

    return {
      focus: () => search && search.focus(),
      value: () => chosen,
      refresh(next) { nodes = next || nodes; renderPanel(); render(); },
    };
  };

  // ── the activation browser, in list mode ─────────────────────────────────────
  // Drives the very markup the editor uses (templates/graph_viewer/_act_browser
  // .html, included by both pages) so this *is* the graph-mode browser, not a
  // second one that looks like it. The modes play mode has no use for — detail
  // viz, the bend view, pinning, multi-select — are hidden rather than rebuilt:
  // they need the graph, the pin dashboard and the activation fetches.
  //
  // opts: { nodes, ctx, title, onPick(node), onClose() }
  TBSearch.openBrowser = function (opts) {
    opts = opts || {};
    const modal = document.getElementById("act-modal");
    if (!modal) return null;

    const byId = (id) => document.getElementById(id);
    const hide = (id) => { const e = byId(id); if (e) e.style.display = "none"; };

    // list mode only
    ["act-modal-mode-btns", "act-modal-detail", "act-modal-bend-view",
     "act-selection-bar", "act-pin-all-btn", "act-add-to-list-btn"].forEach(hide);

    const title = byId("act-modal-title");
    if (title) title.textContent = opts.title || "Choose a node";

    const close = () => {
      modal.style.display = "none";
      document.removeEventListener("keydown", onKey);
      if (opts.onClose) opts.onClose();
    };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    document.addEventListener("keydown", onKey);

    const closeBtn = byId("act-modal-close");
    if (closeBtn) closeBtn.onclick = close;
    const backdrop = byId("act-modal-backdrop");
    if (backdrop) backdrop.onclick = close;

    // the search bar, filter panel, bookmark bar and list are already in the
    // markup — bind them rather than build them
    const ctl = TBSearch.bind({
      searchEl:   byId("act-modal-search"),
      groupBtn:   byId("act-sort-btn"),
      filterBtn:  byId("act-filter-toggle"),
      bookmarkEl: byId("act-bookmark-bar"),
      panelEl:    byId("act-filter-panel"),
      listEl:     byId("act-modal-list"),
      nodes: opts.nodes || [],
      ctx: opts.ctx || {},
      onPick: (n) => { if (opts.onPick) opts.onPick(n); },
    });

    modal.style.display = "";
    ctl.focus();
    return { close, controller: ctl };
  };

  window.TBSearch = TBSearch;
})();
