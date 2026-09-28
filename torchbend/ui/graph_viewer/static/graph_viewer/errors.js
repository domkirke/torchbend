/* TorchBend — shared trace / run error panel (TBTraceError)
 *
 * A tracing or forward failure raises deep inside torch, so its message alone
 * says nothing about which line of the *model* caused it. The server locates the
 * innermost frame in user code (torchbend/tracing/code.py, `describe_exception`)
 * and reports the last node the tracer got through; this renders both, with the
 * source around the failing line.
 *
 * Shared by the editor (graph.js) and play mode (play.js) so a failure reads the
 * same wherever the model was run from.
 */
(function () {
  "use strict";
  const TBTraceError = {};

  function el(tag, cls, txt) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }

  // Does this payload carry located-in-the-model detail, or is it a bare message?
  // Only the former is worth a full-screen panel.
  TBTraceError.isLocated = function (data) {
    return !!(data && (data.location || data.last_node ||
                       (data.user_frames && data.user_frames.length)));
  };

  // opts: { onCopyPath(text), log(msg, traceback) }
  TBTraceError.show = function (title, data, opts) {
    if (!data) return null;
    opts = opts || {};
    document.querySelectorAll(".trace-error-overlay").forEach((e) => e.remove());

    const overlay = el("div", "trace-error-overlay");
    const panel = el("div", "trace-error-panel");
    overlay.appendChild(panel);

    // header
    const hdr = el("div", "trace-error-header");
    hdr.appendChild(el("span", "trace-error-title", `⚠ ${title}`));
    const close = el("button", "trace-error-close", "×");
    close.addEventListener("click", () => overlay.remove());
    hdr.appendChild(close);
    panel.appendChild(hdr);

    const body = el("div", "trace-error-body");
    panel.appendChild(body);

    // exception
    body.appendChild(el("div", "trace-error-exc",
      `${data.error_type || "Error"}: ${data.error || ""}`));

    // where in the model's code
    const loc = data.location;
    if (loc) {
      const where = el("div", "trace-error-section");
      where.appendChild(el("div", "trace-error-label", "where"));
      const path = el("div", "trace-error-path");
      path.textContent = `${loc.file}:${loc.line}`;
      path.title = "click to copy";
      path.addEventListener("click", () => {
        navigator.clipboard?.writeText(`${loc.file}:${loc.line}`);
        if (opts.onCopyPath) opts.onCopyPath(`${loc.file}:${loc.line}`);
      });
      where.appendChild(path);
      where.appendChild(el("div", "trace-error-fn", `in ${loc.function}()`
        + (loc.internal ? "  — inside torch/torchbend, not model code" : "")));
      // the op that actually threw, when it lives in torch rather than the model
      if (data.raised_in) {
        where.appendChild(el("div", "trace-error-fn",
          `raised in ${data.raised_in.basename}:${data.raised_in.line} `
          + `(${data.raised_in.function}) — ${data.raised_in.code}`));
      }
      if (loc.context && loc.context.length) {
        const pre = el("pre", "trace-error-code");
        loc.context.forEach(([n, text, isTarget]) => {
          const line = el("div", "trace-error-code-line" + (isTarget ? " target" : ""));
          line.appendChild(el("span", "trace-error-lineno", String(n)));
          line.appendChild(el("span", "trace-error-linetext", text));
          pre.appendChild(line);
        });
        where.appendChild(pre);
      }
      body.appendChild(where);
    }

    // the graph node whose generated code failed (a *run*, not a trace)
    const gf = data.graph_frame;
    if (gf && (gf.node || gf.code)) {
      const sec = el("div", "trace-error-section");
      sec.appendChild(el("div", "trace-error-label", "failing graph node"));
      const line = el("div", "trace-error-node");
      if (gf.node) line.appendChild(el("code", null, gf.node));
      if (gf.code) line.appendChild(el("span", "trace-error-node-meta", "  " + gf.code));
      sec.appendChild(line);
      sec.appendChild(el("div", "trace-error-hint",
        gf.origin
          ? `Running the traced graph, not the model's own forward — this node was `
            + `built from ${gf.origin.basename}:${gf.origin.line} in ${gf.origin.function}().`
          : "Running the traced graph, not the model's own forward — the line above "
            + "is fx's generated code for this node."));
      body.appendChild(sec);
    }

    // how far the trace got
    const ln = data.last_node;
    if (ln) {
      const sec = el("div", "trace-error-section");
      sec.appendChild(el("div", "trace-error-label", "last node traced"));
      const line = el("div", "trace-error-node");
      line.appendChild(el("code", null, ln.name));
      const meta = [ln.op, ln.target && ln.target !== ln.name ? `→ ${ln.target}` : null,
                    ln.shape ? `shape ${ln.shape}` : null,
                    ln.module_path ? `in ${ln.module_path}` : null,
                    ln.code].filter(Boolean).join("  ·  ");
      line.appendChild(el("span", "trace-error-node-meta", "  " + meta));
      sec.appendChild(line);
      sec.appendChild(el("div", "trace-error-hint",
        `${ln.traced} node(s) traced before the failure — the graph stops just after this one.`));
      body.appendChild(sec);
    }

    // user-code call chain
    if (data.user_frames && data.user_frames.length > 1) {
      const sec = el("div", "trace-error-section");
      sec.appendChild(el("div", "trace-error-label", "model call chain"));
      const list = el("div", "trace-error-frames");
      data.user_frames.forEach((f) => {
        const row = el("div", "trace-error-frame");
        row.appendChild(el("span", "trace-error-frame-loc", `${f.basename}:${f.line}`));
        row.appendChild(el("span", "trace-error-frame-fn", ` ${f.function}()`));
        if (f.code) row.appendChild(el("code", "trace-error-frame-code", f.code));
        list.appendChild(row);
      });
      sec.appendChild(list);
      body.appendChild(sec);
    }

    // full traceback, collapsed
    if (data.traceback) {
      const det = document.createElement("details");
      det.className = "trace-error-tb";
      const sum = document.createElement("summary");
      sum.textContent = "full traceback";
      det.appendChild(sum);
      det.appendChild(el("pre", null, data.traceback));
      body.appendChild(det);
    }

    document.body.appendChild(overlay);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
    // also keep it wherever the host page logs errors, so it survives the close
    if (opts.log) {
      opts.log(`${data.error_type || "Error"}: ${data.error}`
        + (loc ? `  (${loc.basename}:${loc.line} in ${loc.function})` : ""),
        data.traceback || "");
    }
    return overlay;
  };

  window.TBTraceError = TBTraceError;
})();
