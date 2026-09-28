/* TorchBend — shared node-view renderer (TBViews)
 *
 * Single client-side counterpart to the server `node_views` package. Renders a
 * view payload (keyed by `payload.view`) into a container, and builds the
 * per-node view picker (dropdown + option widgets) from `_view_meta`.
 *
 * Used by both the editor (graph.js) and play mode (play.js).
 */
(function () {
  "use strict";

  const TBViews = {};

  // ── small DOM / math helpers ──────────────────────────────────────────────
  function el(tag, cls, txt) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }
  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  function purgePlotly(node) {
    if (!window.Plotly || !node) return;
    node.querySelectorAll(".js-plotly-plot").forEach((d) => { try { window.Plotly.purge(d); } catch (e) { /* */ } });
  }
  const HUES = ["#5CB85C", "#4A90D9", "#F0AD4E", "#9B59B6", "#E74C3C",
                "#1ABC9C", "#E67E22", "#2C3E50", "#16A085", "#C0392B"];

  // magma-ish colormap for heatmaps/spectrograms (t in [0,1] -> [r,g,b])
  function magma(t) {
    t = clamp01(t);
    const r = clamp01(1.30 * t - 0.10) * 255;
    const g = clamp01(1.20 * t * t - 0.05) * 255;
    const b = clamp01(0.6 * Math.sin(Math.PI * t) + 0.25 * t) * 255;
    return [r, g, b];
  }

  // ── primitive canvases ────────────────────────────────────────────────────
  function lineCanvas(rows, opts) {
    opts = opts || {};
    const w = Math.min(opts.maxW || 1100, Math.max(220, (rows[0] || []).length));
    const h = opts.h || 110;
    const c = el("canvas", "tbv-canvas");
    c.width = w; c.height = h;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, w, h);
    let lo = Infinity, hi = -Infinity;
    rows.forEach((r) => r.forEach((v) => { if (v < lo) lo = v; if (v > hi) hi = v; }));
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    const span = hi - lo || 1;
    rows.forEach((data, ri) => {
      ctx.strokeStyle = HUES[ri % HUES.length]; ctx.lineWidth = 1; ctx.beginPath();
      const n = data.length;
      data.forEach((v, i) => {
        const x = (i / (n - 1 || 1)) * (w - 2) + 1;
        const y = h - 2 - ((v - lo) / span) * (h - 4);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
    return c;
  }

  function scatterCanvas(rows, opts) {
    opts = opts || {};
    const w = Math.min(opts.maxW || 520, 520), h = opts.h || 160;
    const c = el("canvas", "tbv-canvas");
    c.width = w; c.height = h;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, w, h);
    let lo = Infinity, hi = -Infinity;
    rows.forEach((r) => r.forEach((v) => { if (v < lo) lo = v; if (v > hi) hi = v; }));
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    const span = hi - lo || 1;
    rows.forEach((data, ri) => {
      ctx.fillStyle = HUES[ri % HUES.length];
      const n = data.length;
      data.forEach((v, i) => {
        const x = (i / (n - 1 || 1)) * (w - 6) + 3;
        const y = h - 3 - ((v - lo) / span) * (h - 6);
        ctx.beginPath(); ctx.arc(x, y, 2, 0, 2 * Math.PI); ctx.fill();
      });
    });
    return c;
  }

  function barCanvas(values, opts) {
    opts = opts || {};
    const n = values.length;
    const bw = Math.max(3, Math.min(24, Math.floor((opts.maxW || 520) / Math.max(1, n))));
    const w = Math.max(120, n * bw), h = opts.h || 140;
    const c = el("canvas", "tbv-canvas");
    c.width = w; c.height = h;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, w, h);
    let lo = Math.min(0, ...values), hi = Math.max(0, ...values);
    const span = (hi - lo) || 1;
    const zeroY = h - 2 - ((0 - lo) / span) * (h - 4);
    values.forEach((v, i) => {
      const x = i * bw;
      const y = h - 2 - ((v - lo) / span) * (h - 4);
      ctx.fillStyle = (opts.highlight === i) ? "#F0AD4E" : "#4A90D9";
      ctx.fillRect(x + 1, Math.min(y, zeroY), bw - 2, Math.max(1, Math.abs(zeroY - y)));
    });
    return c;
  }

  function heatmapCanvas(mat, opts) {
    opts = opts || {};
    const rows = mat.length, cols = (mat[0] || []).length;
    const c = el("canvas", "tbv-canvas");
    c.width = cols; c.height = rows;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(cols, rows);
    const cmap = opts.colormap;
    for (let y = 0; y < rows; y++)
      for (let x = 0; x < cols; x++) {
        const v = clamp01(mat[y][x]);
        const idx = (y * cols + x) * 4;
        if (cmap) { const [r, g, b] = cmap(v); img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; }
        else { const g = v * 255; img.data[idx] = g; img.data[idx + 1] = g; img.data[idx + 2] = g; }
        img.data[idx + 3] = 255;
      }
    ctx.putImageData(img, 0, 0);
    if (opts.target) { c.style.width = opts.target + "px"; c.style.imageRendering = "pixelated"; }
    else { c.style.maxWidth = "100%"; }
    c.style.height = "auto";
    return c;
  }

  function imageCanvas(chw, type, target) {
    if (!Array.isArray(chw) || !Array.isArray(chw[0]) || !Array.isArray(chw[0][0])) return el("div");
    const C = chw.length, H = chw[0].length, W = chw[0][0].length;
    const c = el("canvas", "tbv-canvas-img");
    c.width = W; c.height = H;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(W, H);
    const px = (v) => clamp01(v) * 255;
    const gray = type === "gray" || C === 1;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const idx = (y * W + x) * 4;
        let r, g, b, a = 255;
        if (gray) { r = g = b = px(chw[0][y][x]); }
        else {
          r = px(chw[0][y][x]);
          g = px(chw[1] ? chw[1][y][x] : chw[0][y][x]);
          b = px(chw[2] ? chw[2][y][x] : chw[0][y][x]);
          if (type === "rgba" && chw[3]) a = px(chw[3][y][x]);
        }
        img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = a;
      }
    ctx.putImageData(img, 0, 0);
    const tgt = target || 200;
    c.style.width = Math.round(W * (tgt / Math.max(W, H))) + "px";
    c.style.height = "auto";
    return c;
  }

  // ── axis engine ────────────────────────────────────────────────────────────
  // Views are iterated along up to two axes (batch, channel), each with a mode:
  //   navigate (one index) · superimpose (overlay, 1-D only) · list (stacked).
  function range(n) { return Array.from({ length: n }, (_, i) => i); }
  function fmt(v) { const n = Number(v); return Number.isInteger(n) ? String(n) : n.toFixed(4); }

  const LINE_VIEWS = { line: 1, scatter: 1, bar: 1, channel_lines: 1 };
  function isLineContent(p) {
    return !!LINE_VIEWS[p.view] || (p.view === "audio");
  }
  function channelCount(p, audioMode) {
    if (p.view === "channel_lines" || p.view === "channel_grid") return p.n_channels || 1;
    if (p.view === "audio" && audioMode !== "spec") return p.n_channels || 1;
    if (p.view === "image") return (p.shape && p.shape[1]) || 1;
    return 1;
  }
  function superAllowed(p, audioMode) {
    // overlay only makes sense for 1-D content …
    if (p.view === "audio") return audioMode !== "spec";
    // … except for images, where "superimpose" merges the channel axis into a
    // single rgb/rgba picture (the only mode that shows colour).
    if (p.view === "image") return true;
    return !!LINE_VIEWS[p.view];
  }

  // sequences as [B][C][N] for line-family views (channel = 1 when absent)
  function lineSeqs(p, audioMode) {
    if (p.view === "channel_lines") return p.batches || [];
    if (p.view === "audio") return p.waveform || [];
    return (p.lines || []).map((row) => [row]);   // line/scatter/bar
  }

  function buildLinePanels(seqs, st) {
    const B = seqs.length;
    let bgroups;
    if (st.batchMode === "navigate") bgroups = [[Math.min(st.batchIdx, B - 1)]];
    else if (st.batchMode === "list") bgroups = range(B).map((b) => [b]);
    else bgroups = [range(B)];                     // superimpose
    const panels = [];
    bgroups.forEach((bg) => {
      const C = (seqs[bg[0]] || []).length;
      let cgroups;
      if (st.chanMode === "navigate") cgroups = [[Math.min(st.chanIdx, C - 1)]];
      else if (st.chanMode === "list") cgroups = range(C).map((c) => [c]);
      else cgroups = [range(C)];
      cgroups.forEach((cg) => {
        const ls = [];
        bg.forEach((b) => cg.forEach((c) => {
          const d = (seqs[b] || [])[c];
          if (d) ls.push({ label: `b${b}${C > 1 ? " c" + c : ""}`, data: d });
        }));
        let title = "";
        if (st.batchMode === "list") title += "batch " + bg[0];
        if (st.chanMode === "list" && C > 1) title += (title ? " · " : "") + "ch " + cg[0];
        panels.push({ title, lines: ls });
      });
    });
    return panels;
  }

  // How many interactive (Plotly) plots one view may build. Beyond this the
  // canvas renderer takes over: the cards are thumbnails anyway, and the cost of
  // a Plotly instance is paid again on every refresh.
  const MAX_PLOTLY_PANELS = 8;

  const _PLOTLY_CFG = { displaylogo: false, responsive: true,
                        modeBarButtonsToRemove: ["select2d", "lasso2d"] };
  function _plotDiv(host, h) {
    const div = el("div"); div.style.width = "100%"; div.style.height = (h || 230) + "px";
    host.appendChild(div); return div;
  }
  function plotLines(panel, host, detailed, mode) {
    if (detailed && window.Plotly) {
      const traces = panel.lines.map((ln) => ({ y: ln.data, mode: mode || "lines", type: "scattergl",
                                                name: ln.label, line: { width: 1 }, marker: { size: 4 } }));
      window.Plotly.newPlot(_plotDiv(host), traces,
        { margin: { l: 40, r: 10, t: 8, b: 28 }, showlegend: panel.lines.length > 1 && panel.lines.length <= 12,
          legend: { font: { size: 9 } }, paper_bgcolor: "#fff", plot_bgcolor: "#fff" }, _PLOTLY_CFG);
    } else if (mode === "markers") {
      host.appendChild(scatterCanvas(panel.lines.map((l) => l.data)));
    } else {
      host.appendChild(lineCanvas(panel.lines.map((l) => l.data), {}));
    }
  }
  const _GRAY_SCALE = [[0, "#000"], [1, "#fff"]];
  function plotHeat(mat, host, detailed, gray) {
    if (detailed && window.Plotly) {
      window.Plotly.newPlot(_plotDiv(host, 260),
        [{ z: mat, type: "heatmap", colorscale: gray ? _GRAY_SCALE : "Magma", showscale: false }],
        { margin: { l: 34, r: 10, t: 8, b: 28 }, paper_bgcolor: "#fff", plot_bgcolor: "#fff" }, _PLOTLY_CFG);
    } else { host.appendChild(specCanvas(mat, gray)); }
  }
  function plotBar(values, host, detailed, highlight, names) {
    if (detailed && window.Plotly) {
      const colors = values.map((_, i) => (i === highlight ? "#F0AD4E" : "#4A90D9"));
      const x = (names && names.length === values.length) ? names : values.map((_, i) => i);
      window.Plotly.newPlot(_plotDiv(host, 220),
        [{ type: "bar", x, y: values, marker: { color: colors } }],
        { margin: { l: 40, r: 10, t: 8, b: 30 }, paper_bgcolor: "#fff", plot_bgcolor: "#fff" }, _PLOTLY_CFG);
    } else { host.appendChild(barCanvas(values, { highlight })); }
  }
  // multi-series bars (one series per superimposed line)
  function plotBarPanel(panel, host, detailed) {
    const series = panel.lines || [];
    if (series.length <= 1) { plotBar(series[0] ? series[0].data : [], host, detailed); return; }
    if (detailed && window.Plotly) {
      const traces = series.map((ln) => ({ type: "bar", y: ln.data, name: ln.label }));
      window.Plotly.newPlot(_plotDiv(host, 220), traces,
        // grouped, with batches sitting tightly next to each other at each index
        { barmode: "group", bargap: 0.25, bargroupgap: 0.0,
          margin: { l: 40, r: 10, t: 8, b: 30 }, paper_bgcolor: "#fff", plot_bgcolor: "#fff",
          showlegend: series.length <= 12, legend: { font: { size: 9 } } }, _PLOTLY_CFG);
    } else {
      // simple preview: stack one mini bar chart per series
      series.forEach((ln) => host.appendChild(barCanvas(ln.data, {})));
    }
  }

  // ── audio transport ─────────────────────────────────────────────────────────
  // The WAV is rendered server-side on demand. It is fetched for the batch and
  // channel currently on screen (otherwise you hear batch 0 while looking at
  // batch 3), cached per selection so replaying costs nothing, and its object
  // URL is revoked when replaced.
  function audioSelection(p, st) {
    const b = (st.batchMode === "navigate") ? Math.min(st.batchIdx || 0, (p.n_batches || 1) - 1) : 0;
    const c = (st.chanMode === "navigate") ? Math.min(st.chanIdx || 0, (p.n_channels || 1) - 1) : -1;
    return { batch: Math.max(0, b), channel: c };
  }

  function audioTransport(p, st, opts) {
    const bar = el("div", "tbv-audio-bar");
    const sel = audioSelection(p, st);
    const info = el("span", "tbv-dim", st.audioMode === "spec"
      ? `sr ${p.sample_rate} · n_fft ${p.n_fft} · hop ${p.hop}`
      : `sr ${p.sample_rate} · ${p.n_channels}ch · ${p.length} samples`);

    if (typeof opts.audioFetch !== "function") {
      bar.appendChild(info);
      return bar;
    }

    const slot = el("span", "tbv-audio-slot");
    const btn = el("button", "tbv-btn tbv-audio-load", "▶ load audio");
    btn.title = `Render this output as audio${sel.channel >= 0 ? ` (channel ${sel.channel})` : ""}`
              + `${(p.n_batches || 1) > 1 ? `, batch ${sel.batch}` : ""}`;

    // one cache per view instance, keyed by what is being played
    const cache = (st.__audio || (st.__audio = {}));
    const key = `${sel.batch}:${sel.channel}`;

    const mount = (url) => {
      slot.innerHTML = "";
      const audio = el("audio");
      audio.controls = true;
      audio.className = "tbv-audio";
      audio.preload = "auto";
      audio.src = url;
      slot.appendChild(audio);
      audio.play().catch(() => {});   // autoplay may be blocked; controls remain
    };

    btn.addEventListener("click", async () => {
      if (cache[key]) { mount(cache[key]); return; }
      btn.disabled = true;
      btn.textContent = "rendering…";
      try {
        const blob = await opts.audioFetch(sel);
        // release the URL this slot held before, so repeated renders don't leak
        if (cache[key]) URL.revokeObjectURL(cache[key]);
        cache[key] = URL.createObjectURL(blob);
        mount(cache[key]);
      } catch (e) {
        btn.disabled = false;
        btn.textContent = "▶ load audio";
        btn.title = "Could not render audio: " + (e && e.message ? e.message : e);
        btn.classList.add("tbv-audio-failed");
      }
    });

    slot.appendChild(btn);
    bar.appendChild(slot);
    bar.appendChild(info);
    // already rendered for this selection: show the player straight away
    if (cache[key]) mount(cache[key]);
    return bar;
  }

  // ── audio clips ─────────────────────────────────────────────────────────────
  // One strip per batch item — the waveform (or spectrogram) you are looking at
  // *is* the play button. Auditioning a batch of files used to mean stepping the
  // batch axis and pressing "load audio" once per item; here they are all on
  // screen and a click plays one, with a playhead riding over it.

  // One <audio> per view instance: starting a clip stops whatever was playing,
  // which is what you want when comparing a batch item against the next.
  const CLIP_H = 46;              // strip height, shared by waveform and spectrogram

  // A cheap signature of a payload's *audio content*. The rendered WAVs are
  // cached per batch/channel, but they belong to one particular output: move a
  // macro or change an input and the same key now names different audio. This is
  // what tells the two apart. The waveform the server already sends is a
  // downsample of exactly the samples the WAV would contain, so sampling it is
  // enough to notice any change that matters.
  function audioFingerprint(p) {
    if (!p) return "";
    let h = `${p.view}|${p.length}|${p.sample_rate}|${p.n_batches}|${p.n_channels}`;
    const w = p.waveform;
    if (Array.isArray(w)) {
      for (let b = 0; b < w.length; b++) {
        const chans = w[b] || [];
        for (let c = 0; c < chans.length; c++) {
          const d = chans[c] || [];
          let acc = 0;
          for (let i = 0; i < d.length; i += 17) acc += d[i] * (i + 1);
          h += "|" + acc.toFixed(5);
        }
      }
    }
    return h;
  }

  // Drop every rendered WAV this view is holding (and stop the player).
  function dropAudioCache(st) {
    Object.values(st.__audio || {}).forEach((u) => {
      try { URL.revokeObjectURL(u); } catch (e) { /* already gone */ }
    });
    st.__audio = {};
    const pl = st.__player;
    if (pl && pl.audio) {
      pl.audio.pause();
      pl.audio.removeAttribute("src");
      if (pl.rows) pl.rows.forEach((r) => r.setPlaying(false));
    }
  }

  function clipPlayer(st) {
    const pl = st.__player || (st.__player = { audio: new Audio(), rows: [] });
    pl.audio.onended = () => pl.rows.forEach((r) => r.setPlaying(false));
    pl.rows = [];                       // rebuilt on every redraw
    pl.stopAll = () => pl.rows.forEach((r) => r.setPlaying(false));
    return pl;
  }

  function audioClipRow(p, st, opts, b, chan, player) {
    const row = el("div", "tbv-clip");
    const btn = el("button", "tbv-clip-play", "▶");
    // What this strip is: the batch item, the channel, or both. With one of each
    // there is nothing to distinguish and the label would only take up room.
    const parts = [];
    if ((p.n_batches || 1) > 1) parts.push("b" + b);
    if (chan >= 0 && (p.n_channels || 1) > 1) parts.push("ch" + chan);
    const label = parts.join(" ");
    btn.title = `Play${label ? " " + label : ""}${chan >= 0 ? ` · channel ${chan}` : ""}`;
    row.appendChild(btn);
    if (label) row.appendChild(el("span", "tbv-clip-label", label));

    const viz = el("div", "tbv-clip-viz");
    let strip = null;
    if (st.audioMode === "spec") {
      const m = (p.spectrogram || [])[b];
      if (m) strip = heatmapCanvas(m, { colormap: magma });
    } else {
      const chans = (p.waveform || [])[b] || [];
      const rows = (chan >= 0 ? [chans[chan]] : chans).filter(Boolean);
      strip = lineCanvas(rows.length ? rows : [[0, 0]], { h: CLIP_H });
    }
    if (strip) {
      // the canvas helpers size themselves inline (max-width, height:auto), which
      // a stylesheet cannot override — state the strip geometry here instead, so
      // waveform and spectrogram rows line up at the same height
      strip.style.width = "100%";
      strip.style.maxWidth = "none";
      strip.style.height = CLIP_H + "px";
      viz.appendChild(strip);
    }
    const head = el("div", "tbv-clip-head");     // playhead, positioned in %
    viz.appendChild(head);
    row.appendChild(viz);

    const secs = p.sample_rate ? (p.length || 0) / p.sample_rate : 0;
    row.appendChild(el("span", "tbv-clip-dur", secs ? secs.toFixed(2) + "s" : ""));

    const cache = (st.__audio || (st.__audio = {}));
    const key = `${b}:${chan}`;
    const audio = player.audio;
    let raf = null;

    const setPlaying = (on) => {
      row.classList.toggle("playing", on);
      btn.textContent = on ? "■" : "▶";
      if (!on) {
        head.style.display = "none";
        if (raf) { cancelAnimationFrame(raf); raf = null; }
      } else {
        head.style.display = "block";
        const tick = () => {
          if (!row.classList.contains("playing")) return;
          const d = audio.duration || 0;
          head.style.left = (d ? (audio.currentTime / d) * 100 : 0) + "%";
          raf = requestAnimationFrame(tick);
        };
        tick();
      }
    };
    const me = { setPlaying };
    player.rows.push(me);

    // clicking the strip starts from where you clicked, so a long clip does not
    // have to be listened to from the top
    async function play(fromFrac) {
      if (row.classList.contains("playing") && fromFrac == null) {
        audio.pause(); player.stopAll(); return;
      }
      if (!cache[key]) {
        btn.disabled = true; btn.textContent = "…";
        try {
          cache[key] = URL.createObjectURL(await opts.audioFetch({ batch: b, channel: chan }));
        } catch (e) {
          btn.disabled = false; btn.textContent = "▶";
          row.classList.add("failed");
          row.title = "Could not render audio: " + ((e && e.message) || e);
          return;
        }
        btn.disabled = false; btn.textContent = "▶";
      }
      player.stopAll();
      if (audio.src !== cache[key]) audio.src = cache[key];
      const start = () => {
        try { audio.currentTime = (fromFrac || 0) * (audio.duration || 0); } catch (_) {}
        audio.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
      };
      if (audio.readyState >= 1) start();
      else audio.addEventListener("loadedmetadata", start, { once: true });
    }

    btn.addEventListener("click", (e) => { e.stopPropagation(); play(null); });
    viz.addEventListener("click", (e) => {
      const r = viz.getBoundingClientRect();
      play(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)));
    });
    return row;
  }

  //: A listed channel axis on a many-channel activation would ask for hundreds
  //: of strips, each of which can start a server render. Past this the list is
  //: cut and says so — the channel navigator reaches the rest one at a time.
  const MAX_CLIP_CHANNELS = 32;

  // Which channels get a strip of their own.
  //
  // "navigate" plays the one being looked at, "list" plays each in turn — a
  // multichannel activation is usually worth hearing channel by channel, which
  // is the whole point of listing them. Anything else is the mixdown (-1).
  function audioChannels(p, st) {
    const C = p.n_channels || 1;
    // The spectrogram the server sends is one per batch, over the channel
    // mixdown: splitting it per channel would repeat the same picture under
    // every strip. The channel axis is a waveform affair.
    if (st.audioMode === "spec") return [-1];
    if (st.chanMode === "navigate") return [Math.min(st.chanIdx || 0, C - 1)];
    if (st.chanMode === "list" && C > 1) return range(Math.min(C, MAX_CLIP_CHANNELS));
    return [-1];
  }

  function audioClips(p, st, opts, host) {
    const B = p.n_batches || 1;
    const C = p.n_channels || 1;
    const bsel = st.batchMode === "navigate" ? [Math.min(st.batchIdx || 0, B - 1)] : range(B);
    const csel = audioChannels(p, st);
    const player = clipPlayer(st);
    const box = el("div", "tbv-clips");
    bsel.forEach((b) => csel.forEach(
      (c) => box.appendChild(audioClipRow(p, st, opts, b, c, player))));
    host.appendChild(box);
    if (st.chanMode === "list" && st.audioMode !== "spec" && C > MAX_CLIP_CHANNELS) {
      host.appendChild(el("div", "tbv-dim tbv-perf-note",
        `first ${MAX_CLIP_CHANNELS} of ${C} channels — use ▭ navigate for the rest`));
    }
    host.appendChild(el("div", "tbv-dim", st.audioMode === "spec"
      ? `sr ${p.sample_rate} · n_fft ${p.n_fft} · hop ${p.hop} · click a strip to play`
      : `sr ${p.sample_rate} · ${p.n_channels}ch · ${p.length} samples · click a strip to play`));
  }

  // ── per-view drawing (given axis state st) ──────────────────────────────────
  function drawView(p, host, st, opts, detailed) {
    switch (p.view) {
      case "scalar":
        host.appendChild(el("div", "tbv-scalar", fmt(p.value))); return;

      case "line": case "scatter": case "bar": case "channel_lines": {
        const panels = buildLinePanels(lineSeqs(p, st.audioMode), st);
        // Channels default to "list" mode, so a 64-channel activation asks for
        // 64 plots. Each Plotly instance costs ~15 ms to build and holds its own
        // context — that is what makes a pinned channel_lines card take seconds,
        // and it re-pays it on every macro move. Past a handful of panels use
        // the canvas renderer instead (~0.1 ms each).
        const rich = detailed && panels.length <= MAX_PLOTLY_PANELS;
        if (detailed && !rich) {
          const note = el("div", "tbv-dim",
            `${panels.length} plots — drawn lightweight; switch the channel axis to `
            + `▭ navigate or ⧉ superimpose for interactive ones`);
          note.className = "tbv-dim tbv-perf-note";
          host.appendChild(note);
        }
        panels.forEach((pan) => {
          if (pan.title) host.appendChild(el("div", "tbv-sub", pan.title));
          if (p.view === "scatter") plotLines(pan, host, rich, "markers");
          else if (p.view === "bar") plotBarPanel(pan, host, rich);
          else plotLines(pan, host, rich);
        });
        return;
      }

      case "audio": {
        // Playable strips whenever we can actually render audio. Superimpose is
        // the exception: overlaying batches is about comparing their shape, not
        // listening to one of them, so it keeps the plot and the single
        // transport row.
        if (typeof opts.audioFetch === "function" && st.batchMode !== "superimpose") {
          audioClips(p, st, opts, host);
          return;
        }
        // One transport row on top, plot below. The player replaces the load
        // button once the audio is there, instead of sitting empty underneath.
        host.appendChild(audioTransport(p, st, opts));
        if (st.audioMode === "spec") drawRaster((p.spectrogram || []).map((m) => [m]), st, host, detailed, false);
        else drawView({ view: "channel_lines", batches: p.waveform, n_batches: p.n_batches, n_channels: p.n_channels },
                       host, st, opts, detailed);
        return;
      }

      // feature maps shown as black & white (not a colour image)
      case "heatmap":      drawRaster((p.batches || []).map((m) => [m]), st, host, detailed, true); return;
      case "channel_grid": drawGrid(p.batches || [], st, host); return;

      case "image": {
        const imgs = p.images || [];
        const B = imgs.length;
        const C = (imgs[0] && imgs[0].length) || 1;
        const bsel = st.batchMode === "navigate" ? [Math.min(st.batchIdx, B - 1)] : range(B);
        const box = el("div", "tbv-grid");
        const tgt = Math.round(300 * (st.imgZoom || 1));
        bsel.forEach((b) => {
          if (!imgs[b]) return;
          if (B > 1 && st.batchMode === "list") box.appendChild(el("div", "tbv-sub", "batch " + b));
          if (C <= 1 || st.chanMode !== "list") {
            // composite image (all channels merged) or navigate-to-single-channel
            const chw = (st.chanMode === "navigate" && C > 1)
              ? [imgs[b][Math.min(st.chanIdx, C - 1)]]
              : imgs[b];
            box.appendChild(imageCanvas(chw, st.chanMode === "navigate" ? "gray" : (p.image_type || "gray"), tgt));
          } else {
            // list mode: one grayscale tile per channel
            range(C).forEach((c) => {
              if (C > 1) box.appendChild(el("div", "tbv-sub", "ch " + c));
              box.appendChild(imageCanvas([imgs[b][c]], "gray", tgt));
            });
          }
        });
        host.appendChild(box);
        return;
      }

      case "category": {
        const probs = p.probs || [], argmax = p.argmax || [];
        const names = p.names && p.names.length ? p.names : null;
        const sel = st.batchMode === "navigate" ? [Math.min(st.batchIdx, probs.length - 1)] : range(probs.length);
        sel.forEach((b) => {
          const am = argmax[b];
          const card = el("div", "tbv-cat");
          if (st.batchMode === "list") card.appendChild(el("div", "tbv-sub", "batch " + b));
          plotBar(probs[b] || [], card, detailed, am, names);
          card.appendChild(el("div", "tbv-cat-label", `argmax: ${names && names[am] != null ? names[am] : am}`));
          host.appendChild(card);
        });
        return;
      }

      case "text": {
        // what the ids stand for. `from_logits` means these are the model's
        // own choices at each position, not something it was given.
        const texts = p.texts || [];
        if (p.error) { host.appendChild(el("div", "tbv-unsupported", p.error)); return; }
        const sel = st.batchMode === "navigate"
          ? [Math.min(st.batchIdx, texts.length - 1)] : range(texts.length);
        // The per-position alternatives are only worth showing where there is
        // room to read them — expanded or pinned, not in a sidebar card.
        const explorable = detailed && p.positions && p.positions.length;
        sel.forEach((b) => {
          if (st.batchMode === "list" && texts.length > 1)
            host.appendChild(el("div", "tbv-sub", "batch " + b));
          if (explorable && p.positions[b]) host.appendChild(buildTokenExplorer(p, b, st));
          else host.appendChild(el("div", "tbv-text", texts[b] || ""));
        });
        host.appendChild(el("div", "tbv-dim",
          `${p.n_tokens} tokens${p.from_logits ? " · argmax over the vocabulary" : ""}`
          + (explorable ? " · click a token for its runners-up" : "")));
        return;
      }

      case "tokens": {
        const ids = p.ids || [];
        const pieces = p.pieces || null;
        const names = p.names && p.names.length ? p.names : null;
        const sel = st.batchMode === "navigate" ? [Math.min(st.batchIdx, ids.length - 1)] : range(ids.length);
        sel.forEach((b) => {
          if (st.batchMode === "list") host.appendChild(el("div", "tbv-sub", "batch " + b));
          const row = el("div", "tbv-tokens");
          (ids[b] || []).forEach((id) => {
            // a pasted vocabulary first, then the model's own decoder, then the
            // bare number — the id is always on the tooltip either way
            const label = names && names[id] != null ? names[id]
                        : pieces && pieces[String(id)] != null ? pieces[String(id)]
                        : String(id);
            const tok = el("span", "tbv-token", label);
            tok.title = `id ${id}`; row.appendChild(tok);
          });
          host.appendChild(row);
        });
        host.appendChild(el("div", "tbv-dim", `${p.n_tokens} tokens · vocab ${p.vocab}`));
        return;
      }

      default:
        host.appendChild(el("div", "tbv-unsupported",
          `${p.view || "?"} · shape [${(p.shape || []).join(", ")}]${p.error ? " — " + p.error : " — no preview"}`));
    }
  }

  // ── token probability explorer ─────────────────────────────────────────────
  // Each position of a language model's output is a distribution, and the text
  // you read is only its argmax. This shows what else was in contention, and
  // lets you put a runner-up in its place — at which point the model is asked
  // again, so the tokens after your edit are the ones it would really produce.
  //
  // The end of the sequence is a token like any other: it sits in the strip
  // where the tokenizer's marker fell, and any position can be made the end
  // instead.
  function seqKey(p, b) {
    const row = (p.positions || [])[b] || [];
    return [p.n_tokens, row.length, row.length ? row[0].id : -1].join(":");
  }

  function buildTokenExplorer(p, b, st) {
    const wrap = el("div", "tbv-explore");
    const key = seqKey(p, b);
    const all = (st.__seq || (st.__seq = {}));
    // a fresh activation replaces the sequence; an edit within one survives
    // re-renders (batch switch, option change)
    if (!all[b] || all[b].key !== key) {
      all[b] = {
        key: key,
        positions: ((p.positions || [])[b] || []).map((x) => Object.assign({}, x)),
        seps: (p.seps || []).slice(),
        edited: {},
        end: ((p.end_at || [])[b] == null ? null : p.end_at[b]),
        busy: false, error: null, recomputedFrom: null,
      };
    }
    const seq = all[b];

    const render = () => {
      wrap.innerHTML = "";
      const n = seq.positions.length;
      const stop = (seq.end == null ? n : seq.end);

      const line = el("div", "tbv-text tbv-explore-line");
      line.textContent = seq.positions.slice(0, stop)
        .map((pos, j) => (j ? (seq.seps[j] || "") : "") + pos.piece).join("");
      wrap.appendChild(line);

      const strip = el("div", "tbv-explore-strip");
      seq.positions.forEach((pos, j) => {
        const past = (seq.end != null && j > seq.end);
        const isEnd = (j === seq.end);
        const chip = el("button", "tbv-tok"
          + (seq.edited[j] ? " tbv-tok-swapped" : "")
          + (seq.recomputedFrom != null && j > seq.recomputedFrom ? " tbv-tok-fresh" : "")
          + (past ? " tbv-tok-past" : "") + (isEnd ? " tbv-tok-end" : ""));
        chip.appendChild(el("span", "tbv-tok-piece", pos.piece || "␣"));
        const bar = el("div", "tbv-tok-bar");
        const fill = el("div", "tbv-tok-fill");
        fill.style.width = Math.max(2, Math.round(pos.prob * 100)) + "%";
        bar.appendChild(fill);
        chip.appendChild(bar);
        chip.title = `${pos.piece} · ${(pos.prob * 100).toFixed(1)}%`
                   + (seq.edited[j] ? " (yours)" : "")
                   + (isEnd ? " — the end of the sequence" : past ? " (after the end)" : "");
        chip.addEventListener("click", () => {
          st.__textOpen = (st.__textOpen === j && st.__textBatch === b) ? null : j;
          st.__textBatch = b;
          render();
        });
        if (st.__textOpen === j && st.__textBatch === b) chip.classList.add("tbv-tok-open");
        strip.appendChild(chip);
      });
      wrap.appendChild(strip);

      const open = (st.__textBatch === b) ? st.__textOpen : null;
      if (open != null && seq.positions[open]) {
        wrap.appendChild(buildAlternatives(seq, open, render));
      }

      if (seq.busy) wrap.appendChild(el("div", "tbv-explore-busy", "asking the model…"));
      if (seq.error) wrap.appendChild(el("div", "tbv-end-miss", seq.error));

      const note = el("div", "tbv-explore-note");
      const parts = [];
      if (seq.recomputedFrom != null) {
        parts.push(`Everything after position ${seq.recomputedFrom} was predicted again `
                   + "from your edit"
                   + (seq.stopped === "eos" ? ", stopping at the model's end token."
                                            : `, for ${p.continue_steps || 16} tokens.`));
      } else if (Object.keys(seq.edited).length) {
        parts.push("Showing your choice; the rest is unchanged.");
      } else {
        parts.push("One distribution per position; the text is its argmax.");
      }
      if (seq.end != null) parts.push(`The sequence ends at position ${seq.end}.`);
      note.textContent = parts.join(" ");
      wrap.appendChild(note);
    };

    // Put a runner-up in place, then ask the model what follows *that* — the
    // whole point, since leaving the tail alone would show a continuation the
    // model never made.
    const choose = async (j, alt) => {
      const prev = seq.positions[j];
      seq.positions[j] = { id: alt.id, piece: alt.piece, prob: alt.prob, alts: prev.alts };
      seq.edited[j] = true;
      // How far to continue is its own question, not "however many tokens the
      // original happened to have left" — that made an edit near the end
      // produce almost nothing, and an edit on the last token produce nothing
      // at all. It runs for `continue for` tokens, or until the model's end
      // token, whichever comes first.
      const steps = Math.max(1, p.continue_steps || 16);

      seq.busy = true; seq.error = null; render();
      try {
        const r = await fetch("/api/text/continue/", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prefix: seq.positions.slice(0, j + 1).map((x) => x.id),
            steps: steps,
            topk: (prev.alts || []).length || 5,
          }),
        });
        const d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || r.statusText);
        // the sequence is now the prefix plus what the model produced from it —
        // longer or shorter than before, which is the truth of the matter
        seq.positions = seq.positions.slice(0, j + 1).concat(d.positions);
        seq.seps = (d.seps || []).slice(0, seq.positions.length);
        seq.recomputedFrom = j;
        seq.stopped = d.stopped;
        // the end is wherever the model actually ended, if it did
        seq.end = (d.eos_at == null) ? null : d.eos_at;
      } catch (e) {
        seq.error = "could not continue from here: " + (e.message || e)
                  + " — the tokens after your edit are still the model's originals";
      }
      seq.busy = false; render();
    };

    const setEnd = (j) => { seq.end = (seq.end === j) ? null : j; render(); };

    wrap.__choose = choose;
    wrap.__setEnd = setEnd;
    render();
    return wrap;

    function buildAlternatives(seq, j, render) {
      const panel = el("div", "tbv-alts");
      const head = el("div", "tbv-alts-head");
      head.appendChild(el("span", null, `position ${j}`));

      const actions = el("div", "tbv-alts-actions");
      // the end of the sequence is set here, on the token itself
      const endBtn = el("button", "tbv-alts-reset",
                        seq.end === j ? "✕ not the end" : "⊣ end here");
      endBtn.title = seq.end === j
        ? "Stop treating this token as the end of the sequence"
        : "Treat this token as the end — the text stops before it";
      endBtn.addEventListener("click", () => setEnd(j));
      actions.appendChild(endBtn);
      if (seq.edited[j]) {
        const reset = el("button", "tbv-alts-reset", "↺ model's choice");
        reset.addEventListener("click", () => choose(j, seq.positions[j].alts[0]));
        actions.appendChild(reset);
      }
      head.appendChild(actions);
      panel.appendChild(head);

      (seq.positions[j].alts || []).forEach((alt, k) => {
        const isPick = alt.id === seq.positions[j].id;
        const line = el("button", "tbv-alt" + (isPick ? " tbv-alt-active" : ""));
        line.appendChild(el("span", "tbv-alt-piece", alt.piece || "␣"));
        const bar = el("div", "tbv-alt-bar");
        const fill = el("div", "tbv-alt-fill");
        fill.style.width = Math.max(1, Math.round(alt.prob * 100)) + "%";
        bar.appendChild(fill);
        line.appendChild(bar);
        line.appendChild(el("span", "tbv-alt-prob", (alt.prob * 100).toFixed(1) + "%"));
        line.title = `id ${alt.id}`;
        line.addEventListener("click", () => choose(j, alt));
        panel.appendChild(line);
      });
      return panel;
    }
  }

  // single big map per batch (heatmap / spectrogram). gray=true → black & white.
  function drawRaster(maps, st, host, detailed, gray) {
    const B = maps.length;
    const bsel = st.batchMode === "navigate" ? [Math.min(st.batchIdx, B - 1)] : range(B);
    bsel.forEach((b) => {
      const m = (maps[b] || [])[0];
      if (st.batchMode === "list" && B > 1) host.appendChild(el("div", "tbv-sub", "batch " + b));
      if (m) plotHeat(m, host, detailed, gray);
    });
  }

  // grid of small channel-map tiles (channel_grid); always black & white, zoomable.
  function drawGrid(maps, st, host) {
    const B = maps.length;
    const bsel = st.batchMode === "navigate" ? [Math.min(st.batchIdx, B - 1)] : range(B);
    const tile = Math.round(96 * (st.imgZoom || 1));
    bsel.forEach((b) => {
      const chans = maps[b] || [];
      const csel = st.chanMode === "navigate" ? [Math.min(st.chanIdx, chans.length - 1)] : range(chans.length);
      if (st.batchMode === "list" && B > 1) host.appendChild(el("div", "tbv-sub", "batch " + b));
      const row = el("div", "tbv-grid");
      csel.forEach((c) => { if (chans[c]) row.appendChild(heatmapCanvas(chans[c], { target: tile })); });
      host.appendChild(row);
    });
  }

  function specCanvas(mat, gray) {
    const c = heatmapCanvas(mat, gray ? {} : { colormap: magma });
    c.style.width = "100%"; c.style.maxWidth = "640px";
    return c;
  }

  // ── axis control widget ─────────────────────────────────────────────────────
  const MODE_ICON = { navigate: "▭", superimpose: "⧉", list: "☰" };
  const MODE_TIP = { navigate: "navigate (one at a time)", superimpose: "superimpose (overlay)", list: "list (stacked)" };
  const IMAGE_CHAN_TIP = { navigate: "navigate (one channel, grayscale)",
                           superimpose: "composite (merge channels — colour)",
                           list: "list (one grayscale tile per channel)" };
  function axisControl(label, n, st, modeKey, idxKey, allowSuper, redraw, tips) {
    const wrap = el("span", "tbv-axis");
    wrap.appendChild(el("span", "tbv-dim", label));
    const seg = el("span", "tbv-seg");
    const modes = allowSuper ? ["navigate", "superimpose", "list"] : ["navigate", "list"];
    const num = el("input"); num.type = "number"; num.className = "tbv-batch-num";
    num.min = 0; num.max = n - 1; num.value = st[idxKey] || 0;
    const syncNum = () => { num.style.display = st[modeKey] === "navigate" ? "" : "none"; };
    modes.forEach((m) => {
      const b = el("button", "tbv-seg-btn tbv-icon" + (st[modeKey] === m ? " active" : ""), MODE_ICON[m]);
      b.title = (tips && tips[m]) || MODE_TIP[m];
      b.addEventListener("click", () => {
        st[modeKey] = m;
        [...seg.children].forEach((x) => x.classList.remove("active"));
        b.classList.add("active"); syncNum(); redraw();
        if (TBViews.onStateChange) TBViews.onStateChange();
      });
      seg.appendChild(b);
    });
    num.addEventListener("change", () => {
      st[idxKey] = Math.max(0, Math.min(n - 1, parseInt(num.value, 10) || 0));
      num.value = st[idxKey]; redraw();
      if (TBViews.onStateChange) TBViews.onStateChange();
    });
    wrap.appendChild(seg); wrap.appendChild(num); syncNum();
    return wrap;
  }

  // ── public API ─────────────────────────────────────────────────────────────
  TBViews.render = function (payload, container, opts) {
    opts = opts || {};
    purgePlotly(container);
    container.innerHTML = "";
    if (container.style) container.style.height = "";
    if (!payload) return;
    // interactive (Plotly) plots by default wherever Plotly is loaded; callers may
    // force it off with opts.detailed === false (e.g. tiny thumbnails).
    const plotly = (typeof window.Plotly !== "undefined");
    const detailed = (opts.detailed === undefined) ? plotly : (!!opts.detailed && plotly);

    // Persist the view state across re-renders (zoom slider, realtime refresh, and
    // — via opts.stateKey — play-mode runs that recreate the output card) so the
    // user's batch/channel mode, index and audio toggle stick. Reset only when the
    // view type itself changes (e.g. a different view was picked).
    const store = opts.stateKey ? (TBViews._states || (TBViews._states = {})) : null;
    const prev = store ? store[opts.stateKey] : container._tbvState;
    const nB = payload.n_batches || 1;
    const st = (prev && prev.__view === payload.view) ? prev
             : { showOrig: false, audioMode: "wave",
                 // raster/image-like views: navigate batches; 1-D views: list small batches
                 batchMode: /^(image|heatmap|channel_grid|channel_lines)$/.test(payload.view) ? "navigate"
                          : (nB > 1 && nB <= 16) ? "list" : "navigate",
                 batchIdx: 0, chanMode: null, chanIdx: 0, __view: payload.view };
    if (opts.imgZoom !== undefined) st.imgZoom = opts.imgZoom;     // always take the latest zoom
    else if (st.imgZoom === undefined) st.imgZoom = 1;
    // the channels option (gray/rgb/rgba) doesn't change the view type, but it does
    // change what the channel axis means — re-derive its default mode when it moves.
    if (payload.view === "image" && st.__imgType !== payload.image_type) {
      st.__imgType = payload.image_type;
      st.chanMode = null;
    }
    if (store) store[opts.stateKey] = st; else container._tbvState = st;

    // The view state deliberately survives a re-render (so the batch mode and
    // audio toggle stick across runs) — but the audio it cached must not: a
    // bending or an input change makes every rendered WAV stale under an
    // unchanged key, and the player would happily keep serving the last run.
    // opts.audioEpoch, when the host provides it, is exact: play mode bumps it
    // once per forward pass, so "is this the same audio?" needs no guessing.
    // Without one (the editor's pin cards), fall back to the payload's content.
    const fp = (opts.audioEpoch != null) ? "e" + opts.audioEpoch : audioFingerprint(payload);
    if (st.__audioFp !== fp) { dropAudioCache(st); st.__audioFp = fp; }

    const ctrl = el("div", "tbv-ctrl");
    const host = el("div");
    container.appendChild(ctrl);
    container.appendChild(host);

    const cur = () => (st.showOrig && opts.original) ? opts.original : payload;

    const redraw = () => {
      const p = cur();
      purgePlotly(host);          // release old Plotly instances before clearing
      host.innerHTML = "";
      try { drawView(p, host, st, opts, detailed); }
      catch (e) { host.appendChild(el("div", "tbv-unsupported", "render error: " + e.message)); }
    };

    const buildControls = () => {
      const p = cur();
      if (st.chanMode === null) st.chanMode =
        // rgb/rgba images composite their channels (colour); everything else lists them
        (p.view === "image") ? ((p.image_type && p.image_type !== "gray") ? "superimpose" : "list")
        : (p.view === "channel_lines" || p.view === "channel_grid") ? "list"
        : isLineContent(p) ? "superimpose" : "navigate";
      ctrl.innerHTML = "";

      if (opts.original) {
        const btn = el("button", "tbv-btn", st.showOrig ? "orig" : "bent");
        btn.title = "Toggle bent / original";
        btn.addEventListener("click", () => { st.showOrig = !st.showOrig; buildControls(); redraw(); if (TBViews.onStateChange) TBViews.onStateChange(); });
        ctrl.appendChild(btn);
      }
      if (p.view === "audio") {
        const seg = el("div", "tbv-seg");
        [["wave", "∿", "waveform"], ["spec", "▦", "spectrogram"]].forEach(([m, icon, tip]) => {
          const b = el("button", "tbv-seg-btn tbv-icon" + (st.audioMode === m ? " active" : ""), icon);
          b.title = tip;
          b.addEventListener("click", () => { st.audioMode = m; buildControls(); redraw(); if (TBViews.onStateChange) TBViews.onStateChange(); });
          seg.appendChild(b);
        });
        ctrl.appendChild(seg);
      }
      const nB = p.n_batches || 1;
      if (nB > 1) ctrl.appendChild(axisControl("batch", nB, st, "batchMode", "batchIdx", superAllowed(p, st.audioMode), redraw));
      const nC = channelCount(p, st.audioMode);
      if (nC > 1) ctrl.appendChild(axisControl("channel", nC, st, "chanMode", "chanIdx", superAllowed(p, st.audioMode), redraw,
        p.view === "image" ? IMAGE_CHAN_TIP : null));

      if (!ctrl.children.length) ctrl.style.display = "none";
      else ctrl.style.display = "";
    };

    buildControls();
    redraw();
  };

  // Build the view picker (dropdown of compatible views + option widgets).
  // meta = _view_meta; onChange(viewName, optionValues) is called on any change.
  TBViews.renderPicker = function (meta, onChange) {
    const wrap = el("div", "tbv-picker");
    if (!meta || !meta.compatible || meta.compatible.length === 0) return wrap;

    const sel = el("select", "tbv-picker-select");
    meta.compatible.forEach((c) => {
      const o = el("option", null, c.label || c.name);
      o.value = c.name;
      if (c.name === meta.current) o.selected = true;
      sel.appendChild(o);
    });
    wrap.appendChild(el("span", "tbv-picker-label", "view"));
    wrap.appendChild(sel);

    const optBox = el("span", "tbv-picker-opts");
    wrap.appendChild(optBox);

    const collectOpts = () => {
      const vals = {};
      optBox.querySelectorAll("[data-opt]").forEach((inp) => {
        const name = inp.dataset.opt, type = inp.dataset.type;
        if (type === "bool") vals[name] = inp.checked;
        else if (type === "int") vals[name] = parseInt(inp.value, 10);
        else if (type === "float") vals[name] = parseFloat(inp.value);
        else if (type === "str_list") vals[name] = inp.value;
        else vals[name] = inp.value;
      });
      return vals;
    };

    const buildOpts = (options, values) => {
      optBox.innerHTML = "";
      (options || []).forEach((o) => {
        const v = (values && values[o.name] != null) ? values[o.name] : o.default;
        const lab = el("label", "tbv-opt");
        lab.title = o.description || "";
        lab.appendChild(el("span", "tbv-opt-name", o.label || o.name));
        let inp;
        if (o.type === "bool") {
          inp = el("input"); inp.type = "checkbox"; inp.checked = !!v;
        } else if (o.type === "choice") {
          inp = el("select");
          (o.choices || []).forEach((ch) => {
            const op = el("option", null, ch); op.value = ch; if (ch === v) op.selected = true;
            inp.appendChild(op);
          });
        } else if (o.type === "int" || o.type === "float") {
          inp = el("input"); inp.type = "number"; inp.value = v != null ? v : "";
          if (o.type === "int") inp.step = 1;
          if (o.range) { if (o.range[0] != null) inp.min = o.range[0]; if (o.range[1] != null) inp.max = o.range[1]; }
        } else { // str / str_list
          inp = el("input"); inp.type = "text";
          inp.value = Array.isArray(v) ? v.join(", ") : (v != null ? v : "");
          if (o.type === "str_list") inp.placeholder = "comma,separated";
        }
        inp.className = "tbv-opt-input";
        inp.dataset.opt = o.name; inp.dataset.type = o.type;
        inp.addEventListener("change", () => onChange(sel.value, collectOpts()));
        lab.appendChild(inp);
        optBox.appendChild(lab);
      });
    };

    buildOpts(meta.options, meta.option_values);

    sel.addEventListener("change", () => {
      // options schema is per-view; we don't have other views' schemas client-side,
      // so switching view sends empty options (server fills defaults) and the next
      // meta refresh rebuilds the option widgets.
      optBox.innerHTML = "";
      onChange(sel.value, {});
    });

    return wrap;
  };

  window.TBViews = TBViews;
})();
