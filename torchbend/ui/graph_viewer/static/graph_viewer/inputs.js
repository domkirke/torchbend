/* TorchBend — shared image input widget (TBImage)
 *
 * A small reusable widget + processor for image file inputs, used by both the
 * editor (graph.js) and play mode (play.js) so they behave the same. Supports
 * a target size, channel count, and two modes:
 *   resize — scale the whole image to the target size (default)
 *   crop   — pick an x/y/w/h region, scaled to the target size
 *
 * The processed PNG is written to `entry.croppedFile`.
 */
(function () {
  "use strict";
  const TBImage = {};

  function el(tag, cls, txt) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }

  function _loadImg(entry) {
    if (entry._img) return Promise.resolve(entry._img);
    return new Promise((res, rej) => {
      const url = entry.thumbUrl || URL.createObjectURL(entry.file);
      entry.thumbUrl = url;
      const img = new Image();
      img.onload = () => { entry._img = img; res(img); };
      img.onerror = rej;
      img.src = url;
    });
  }

  // detect natural channel count from pixel data (1 / 3 / 4)
  TBImage.detectChannels = function (img) {
    const sw = Math.min(img.naturalWidth, 64), sh = Math.min(img.naturalHeight, 64);
    const c = el("canvas"); c.width = sw; c.height = sh;
    const ctx = c.getContext("2d"); ctx.drawImage(img, 0, 0, sw, sh);
    const px = ctx.getImageData(0, 0, sw, sh).data;
    let hasColor = false, hasAlpha = false;
    for (let i = 0; i < px.length; i += 4) {
      if (px[i] !== px[i + 1] || px[i] !== px[i + 2]) hasColor = true;
      if (px[i + 3] < 255) hasAlpha = true;
      if (hasColor && hasAlpha) break;
    }
    return hasAlpha ? 4 : (hasColor ? 3 : 1);
  };

  // initialise entry.* fields (target size / channels / crop region) from a shape hint
  TBImage.init = async function (entry, shape) {
    const img = await _loadImg(entry);
    entry.srcW = img.naturalWidth; entry.srcH = img.naturalHeight;
    const detected = TBImage.detectChannels(img);
    const nodeCh = (shape && shape.length >= 3) ? shape[shape.length - 3] : null;
    if (entry.imgChannels == null) entry.imgChannels = nodeCh || detected;
    if (entry.imgTargetW == null) entry.imgTargetW = (shape && shape.length >= 1) ? shape[shape.length - 1] : img.naturalWidth;
    if (entry.imgTargetH == null) entry.imgTargetH = (shape && shape.length >= 2) ? shape[shape.length - 2] : img.naturalHeight;
    if (entry.imgMode == null) entry.imgMode = "resize";
    if (entry.cropX == null) { entry.cropX = 0; entry.cropY = 0; entry.cropW = img.naturalWidth; entry.cropH = img.naturalHeight; }
    return img;
  };

  // process entry.file → entry.croppedFile (PNG) applying mode / size / channels
  TBImage.process = async function (entry) {
    if (!entry || !entry.file) return;
    const img = await _loadImg(entry);
    const sW = img.naturalWidth, sH = img.naturalHeight;
    entry.srcW = sW; entry.srcH = sH;
    const tw = Math.max(1, parseInt(entry.imgTargetW, 10) || sW);
    const th = Math.max(1, parseInt(entry.imgTargetH, 10) || sH);
    const nch = entry.imgChannels || 3;
    const canvas = el("canvas"); canvas.width = tw; canvas.height = th;
    const ctx = canvas.getContext("2d");
    if (entry.imgMode === "crop") {
      const cx = Math.max(0, Math.min(entry.cropX || 0, sW - 1));
      const cy = Math.max(0, Math.min(entry.cropY || 0, sH - 1));
      const cw = Math.max(1, Math.min(entry.cropW || sW, sW - cx));
      const ch = Math.max(1, Math.min(entry.cropH || sH, sH - cy));
      ctx.drawImage(img, cx, cy, cw, ch, 0, 0, tw, th);
    } else {                                  // resize: whole image → target
      ctx.drawImage(img, 0, 0, sW, sH, 0, 0, tw, th);
    }
    if (nch === 1) {
      const id = ctx.getImageData(0, 0, tw, th);
      for (let i = 0; i < id.data.length; i += 4) {
        const g = Math.round(0.299 * id.data[i] + 0.587 * id.data[i + 1] + 0.114 * id.data[i + 2]);
        id.data[i] = id.data[i + 1] = id.data[i + 2] = g;
      }
      ctx.putImageData(id, 0, 0);
    }
    await new Promise((res) => canvas.toBlob((blob) => {
      const name = (entry.file.name || "image").replace(/\.[^.]+$/, ".png");
      entry.croppedFile = new File([blob], name, { type: "image/png" });
      res();
    }, "image/png"));
  };

  // Build the full widget (thumbnail + size + channels + mode + crop) into `host`.
  // opts: { shape, onChange }. Mutates `entry`; calls onChange() after each process.
  TBImage.widget = async function (entry, host, opts) {
    opts = opts || {};
    await TBImage.init(entry, opts.shape);
    host.innerHTML = "";
    const pane = el("div", "img-pane");

    const thumb = el("img", "img-thumb"); thumb.src = entry.thumbUrl; pane.appendChild(thumb);
    const controls = el("div", "img-pane-controls");
    const status = el("span", "crop-status");

    const commit = async () => {
      status.textContent = "…";
      await TBImage.process(entry);
      status.textContent = "✓";
      setTimeout(() => { status.textContent = ""; }, 1000);
      if (opts.onChange) opts.onChange();
    };

    // Row 1: target size + channels + mode
    const r1 = el("div", "img-pane-row");
    r1.appendChild(el("span", "crop-lbl", "size"));
    const wInp = el("input"); wInp.type = "number"; wInp.className = "crop-input"; wInp.min = 1; wInp.value = entry.imgTargetW; wInp.title = "Target width";
    const xSep = el("span", "crop-sep", "×");
    const hInp = el("input"); hInp.type = "number"; hInp.className = "crop-input"; hInp.min = 1; hInp.value = entry.imgTargetH; hInp.title = "Target height";
    const chSel = el("select", "crop-input");
    [["auto", "auto"], ["1", "gray"], ["3", "RGB"], ["4", "RGBA"]].forEach(([v, t]) => {
      const o = el("option", null, t); o.value = v;
      if ((v === "auto" && false) || parseInt(v, 10) === entry.imgChannels) o.selected = true;
      chSel.appendChild(o);
    });
    chSel.title = "Channels";
    wInp.addEventListener("change", () => { entry.imgTargetW = parseInt(wInp.value, 10) || 1; commit(); });
    hInp.addEventListener("change", () => { entry.imgTargetH = parseInt(hInp.value, 10) || 1; commit(); });
    chSel.addEventListener("change", () => {
      entry.imgChannels = chSel.value === "auto" ? TBImage.detectChannels(entry._img) : parseInt(chSel.value, 10);
      commit();
    });
    r1.appendChild(wInp); r1.appendChild(xSep); r1.appendChild(hInp); r1.appendChild(chSel); r1.appendChild(status);
    controls.appendChild(r1);

    // Row 2: mode toggle (resize / crop)
    const r2 = el("div", "img-pane-row");
    r2.appendChild(el("span", "crop-lbl", "mode"));
    const seg = el("span", "tbv-seg");
    const cropRow = el("div", "img-pane-row");   // built below; toggled by mode
    const syncMode = () => { cropRow.style.display = entry.imgMode === "crop" ? "" : "none"; };
    [["resize", "resize"], ["crop", "crop"]].forEach(([m, label]) => {
      const b = el("button", "tbv-seg-btn" + (entry.imgMode === m ? " active" : ""), label);
      b.addEventListener("click", () => {
        entry.imgMode = m;
        [...seg.children].forEach((x) => x.classList.remove("active"));
        b.classList.add("active"); syncMode(); commit();
      });
      seg.appendChild(b);
    });
    r2.appendChild(seg);
    r2.appendChild(el("span", "crop-info", `${entry.srcW}×${entry.srcH}`));
    controls.appendChild(r2);

    // Row 3: crop region (only in crop mode)
    cropRow.appendChild(el("span", "crop-lbl", "crop"));
    const mk = (key, ph, title) => {
      const inp = el("input"); inp.type = "number"; inp.className = "crop-input img-crop-small";
      inp.min = (key === "cropW" || key === "cropH") ? 1 : 0;
      inp.value = entry[key]; inp.placeholder = ph; inp.title = title;
      inp.addEventListener("change", () => { entry[key] = parseInt(inp.value, 10) || (inp.min); commit(); });
      return inp;
    };
    cropRow.appendChild(mk("cropX", "x", "Crop X"));
    cropRow.appendChild(mk("cropY", "y", "Crop Y"));
    cropRow.appendChild(mk("cropW", "w", "Crop width"));
    cropRow.appendChild(mk("cropH", "h", "Crop height"));
    controls.appendChild(cropRow);
    syncMode();

    pane.appendChild(controls);
    host.appendChild(pane);

    await commit();   // produce the initial processed file
  };

  window.TBImage = TBImage;
})();


/* TorchBend — shared audio input widget (TBAudio)
 *
 * The counterpart of TBImage for audio files: decode once, then offer a
 * waveform with a draggable crop selection, power-of-2 length presets, and
 * normalize / mono / resample options. Used by both the editor (graph.js) and
 * play mode (play.js) so a file behaves the same wherever it is loaded.
 *
 * The processed WAV is written to `entry.croppedFile`.
 */
(function () {
  "use strict";
  const TBAudio = {};

  function el(tag, cls, txt) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }

  TBAudio.isAudioFile = function (f) {
    return !!f && ((f.type || "").startsWith("audio/") ||
                   /\.(wav|mp3|ogg|flac|m4a|aiff|aif)$/i.test(f.name || ""));
  };

  // ── power-of-2 helpers (crop-length presets) ───────────────────────────────
  function _nextP2(n) {
    if (n <= 1) return 1;
    n--;
    n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
    return n + 1;
  }
  function _nearestP2(n) {
    const hi = _nextP2(n);
    const lo = hi >> 1;
    return (n - lo) <= (hi - n) ? lo : hi;
  }

  function _bufferToWavBlob(buf) {
    const nCh = buf.numberOfChannels, sr = buf.sampleRate, len = buf.length;
    const dataBytes = nCh * len * 2; // 16-bit PCM
    const ab = new ArrayBuffer(44 + dataBytes);
    const v = new DataView(ab);
    const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    ws(0, 'RIFF'); v.setUint32(4, 36 + dataBytes, true); ws(8, 'WAVE');
    ws(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
    v.setUint16(22, nCh, true); v.setUint32(24, sr, true);
    v.setUint32(28, sr * nCh * 2, true); v.setUint16(32, nCh * 2, true); v.setUint16(34, 16, true);
    ws(36, 'data'); v.setUint32(40, dataBytes, true);
    let off = 44;
    for (let i = 0; i < len; i++) for (let ch = 0; ch < nCh; ch++) {
      const s = Math.max(-1, Math.min(1, buf.getChannelData(ch)[i]));
      v.setInt16(off, s < 0 ? s * 32768 : s * 32767, true); off += 2;
    }
    return new Blob([ab], { type: 'audio/wav' });
  }

  function _buildPeaks(buffer, numBins) {
    const nch = buffer.numberOfChannels;
    const len = buffer.length;
    const binSize = len / numBins;
    const peaks = new Float32Array(numBins);
    for (let b = 0; b < numBins; b++) {
      const s0 = Math.floor(b * binSize);
      const s1 = Math.min(Math.ceil((b + 1) * binSize), len);
      let peak = 0;
      for (let ch = 0; ch < nch; ch++) {
        const d = buffer.getChannelData(ch);
        for (let i = s0; i < s1; i++) peak = Math.max(peak, Math.abs(d[i]));
      }
      peaks[b] = peak;
    }
    return peaks;
  }

  TBAudio.draw = function (canvas, entry) {
    const W = canvas.width, H = canvas.height;
    if (!W || !H || !entry.audioBuffer) return;
    if (!entry._wfPeaks || entry._wfPeaksLen !== W) {
      entry._wfPeaks = _buildPeaks(entry.audioBuffer, W);
      entry._wfPeaksLen = W;
    }
    const peaks = entry._wfPeaks;
    const total = entry.totalSamples;
    const startFrac = (entry.cropStart || 0) / total;
    const endFrac = (entry.cropEnd || total) / total;
    const sx = startFrac * W, ex = endFrac * W;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#0f0f0f';
    ctx.fillRect(0, 0, W, H);

    // selection highlight
    ctx.fillStyle = 'rgba(74, 144, 217, 0.13)';
    ctx.fillRect(sx, 0, ex - sx, H);

    // waveform bars
    const mid = H / 2;
    for (let x = 0; x < W; x++) {
      const barH = Math.max(1, peaks[x] * (mid - 2));
      ctx.fillStyle = (x >= sx && x <= ex) ? '#4A90D9' : '#383838';
      ctx.fillRect(x, mid - barH, 1, barH * 2);
    }

    // handle lines + knobs
    ctx.strokeStyle = '#4A90D9'; ctx.lineWidth = 1.5;
    [sx, ex].forEach(hx => {
      ctx.beginPath(); ctx.moveTo(hx, 0); ctx.lineTo(hx, H); ctx.stroke();
      ctx.fillStyle = '#4A90D9';
      ctx.beginPath(); ctx.arc(hx, H / 2, 4, 0, Math.PI * 2); ctx.fill();
    });
  };

  // Channels the placeholder's traced shape asks for: (B, C, N) names it
  // outright; anything flatter is a single channel. null when there is no shape
  // to go on, in which case the file keeps whatever it has.
  TBAudio.wantedChannels = function (shape) {
    if (!shape || !shape.length) return null;
    return shape.length >= 3 ? shape[shape.length - 2] : 1;
  };

  // Decode `entry.file` and seed the crop / processing fields. Idempotent.
  // `shape` (the placeholder's traced shape) sets the default selection length,
  // so a model expecting 65536 samples starts on exactly that many — and, when
  // it asks for one channel, mixes a stereo file down by default rather than
  // handing the model twice the channels it was traced with.
  TBAudio.init = async function (entry, shape) {
    if (entry.audioBuffer) return entry;
    const actx = new (window.AudioContext || window.webkitAudioContext)();
    try {
      const decoded = await actx.decodeAudioData(await entry.file.arrayBuffer());
      entry.audioBuffer = decoded;
      entry.totalSamples = decoded.length;
      entry.sampleRate = decoded.sampleRate;
      entry.numChannels = decoded.numberOfChannels;
      if (entry.targetSR == null) entry.targetSR = decoded.sampleRate;
      if (entry.normalize == null) entry.normalize = false;
      if (entry.toMono == null) {
        const want = TBAudio.wantedChannels(shape);
        entry.toMono = want === 1 && decoded.numberOfChannels > 1;
        entry.monoByShape = entry.toMono;     // so the widget can say why
      }
      if (entry.cropStart == null) entry.cropStart = 0;
      if (entry.cropEnd == null) {
        const expected = (shape && shape.length) ? shape[shape.length - 1] : null;
        entry.cropEnd = expected ? Math.min(expected, decoded.length) : decoded.length;
      }
    } finally {
      try { await actx.close(); } catch (_) { /* already closed */ }
    }
    return entry;
  };

  // Apply crop → resample → mono → normalize, writing entry.croppedFile (WAV).
  TBAudio.process = async function (entry) {
    if (!entry.audioBuffer) { entry.croppedFile = entry.file; return; }
    try {
      const src = entry.audioBuffer;
      const total = entry.totalSamples;
      const start = Math.max(0, entry.cropStart || 0);
      const end = Math.min(total, entry.cropEnd || total);
      const len = Math.max(1, end - start);
      const srcSR = src.sampleRate;
      const targetSR = entry.targetSR || srcSR;
      const nchIn = src.numberOfChannels;

      // 1. Slice into a new buffer (no re-decode — uses the cached audioBuffer)
      const tmpCtx = new OfflineAudioContext(nchIn, len, srcSR);
      const sliced = tmpCtx.createBuffer(nchIn, len, srcSR);
      for (let ch = 0; ch < nchIn; ch++)
        sliced.getChannelData(ch).set(src.getChannelData(ch).subarray(start, end));

      let processed = sliced;

      // 2. Resample (OfflineAudioContext does high-quality resampling)
      if (targetSR !== srcSR) {
        const resampLen = Math.ceil(len * targetSR / srcSR);
        const offCtx = new OfflineAudioContext(nchIn, resampLen, targetSR);
        const bufSrc = offCtx.createBufferSource();
        bufSrc.buffer = sliced;
        bufSrc.connect(offCtx.destination);
        bufSrc.start(0);
        processed = await offCtx.startRendering();
      }

      // 3. Mix to mono (manual average — avoids OfflineAudioContext level issues)
      if (entry.toMono && processed.numberOfChannels > 1) {
        const nch = processed.numberOfChannels;
        const plen = processed.length;
        const psr = processed.sampleRate;
        const monoData = new Float32Array(plen);
        for (let ch = 0; ch < nch; ch++) {
          const d = processed.getChannelData(ch);
          for (let i = 0; i < plen; i++) monoData[i] += d[i] / nch;
        }
        const monoCtx = new OfflineAudioContext(1, plen, psr);
        const monoBuf = monoCtx.createBuffer(1, plen, psr);
        monoBuf.getChannelData(0).set(monoData);
        processed = monoBuf;
      }

      // 4. Normalize to peak 1.0
      if (entry.normalize) {
        const nch = processed.numberOfChannels;
        const plen = processed.length;
        const psr = processed.sampleRate;
        let peak = 0;
        for (let ch = 0; ch < nch; ch++) {
          const d = processed.getChannelData(ch);
          for (let i = 0; i < plen; i++) peak = Math.max(peak, Math.abs(d[i]));
        }
        if (peak > 0 && Math.abs(peak - 1) > 1e-4) {
          const normCtx = new OfflineAudioContext(nch, plen, psr);
          const normBuf = normCtx.createBuffer(nch, plen, psr);
          for (let ch = 0; ch < nch; ch++) {
            const src_d = processed.getChannelData(ch);
            const dst_d = normBuf.getChannelData(ch);
            for (let i = 0; i < plen; i++) dst_d[i] = src_d[i] / peak;
          }
          processed = normBuf;
        }
      }

      const blob = _bufferToWavBlob(processed);
      const base = (entry.file && entry.file.name) || "audio";
      entry.croppedFile = new File([blob], base.replace(/\.[^.]+$/, '.wav'), { type: 'audio/wav' });
    } catch (e) {
      console.warn('[TBAudio] process error:', e);
      entry.croppedFile = entry.file;
    }
  };

  // Build the full widget (waveform + crop + presets + options) into `host`.
  // opts: { shape, onChange }. Mutates `entry`; calls onChange() after each process.
  TBAudio.widget = async function (entry, host, opts) {
    opts = opts || {};
    await TBAudio.init(entry, opts.shape);
    host.innerHTML = "";
    const pane = el("div", "audio-pane");

    // ── Waveform canvas ──────────────────────────────────────────────────────
    const canvasWrap = el("div", "audio-waveform-wrap");
    const canvas = el("canvas", "audio-waveform");
    canvas.height = 60;
    canvasWrap.appendChild(canvas);
    pane.appendChild(canvasWrap);

    const status = el("span", "crop-status audio-pane-status");

    let _commitTimer = null;
    function scheduleCommit() {
      clearTimeout(_commitTimer);
      _commitTimer = setTimeout(async () => {
        status.textContent = "…";
        await TBAudio.process(entry);
        status.textContent = "✓";
        setTimeout(() => { status.textContent = ""; }, 1200);
        if (opts.onChange) opts.onChange();
      }, 0);
    }

    function syncInputsFromEntry() {
      startInp.value = entry.cropStart || 0;
      endInp.value = entry.cropEnd || entry.totalSamples;
      updateInfo();
      TBAudio.draw(canvas, entry);
    }

    // ── Mouse drag on the waveform ───────────────────────────────────────────
    let _drag = null;
    const _xFrac = (e) => {
      const r = canvas.getBoundingClientRect();
      return Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    };
    const _fracToSample = (f) => Math.round(f * entry.totalSamples);
    const _sampleToX = (s) => (s / entry.totalSamples) * canvas.width;

    canvasWrap.addEventListener("mousedown", (e) => {
      const px = _xFrac(e) * canvas.width;
      const eps = 10;
      if (Math.abs(px - _sampleToX(entry.cropStart || 0)) < eps) {
        _drag = 'start';
      } else if (Math.abs(px - _sampleToX(entry.cropEnd || entry.totalSamples)) < eps) {
        _drag = 'end';
      } else {
        entry.cropStart = _fracToSample(_xFrac(e));
        entry.cropEnd = entry.cropStart;
        _drag = 'end';
      }
      syncInputsFromEntry();
      e.preventDefault();
    });

    const _onMove = (e) => {
      if (!_drag) return;
      const s = _fracToSample(_xFrac(e));
      if (_drag === 'start') {
        entry.cropStart = Math.max(0, Math.min(s, (entry.cropEnd || entry.totalSamples) - 1));
      } else {
        entry.cropEnd = Math.min(entry.totalSamples, Math.max(s, (entry.cropStart || 0) + 1));
      }
      syncInputsFromEntry();
    };
    const _onUp = () => {
      if (!_drag) return;
      _drag = null;
      scheduleCommit();
    };
    window.addEventListener("mousemove", _onMove);
    window.addEventListener("mouseup", _onUp);
    // drop the global listeners once the pane leaves the DOM
    const _cleanup = new MutationObserver(() => {
      if (!pane.isConnected) {
        window.removeEventListener("mousemove", _onMove);
        window.removeEventListener("mouseup", _onUp);
        _cleanup.disconnect();
      }
    });
    _cleanup.observe(document.body, { childList: true, subtree: true });

    // ── Row 1: start / end / info / status ───────────────────────────────────
    const row1 = el("div", "audio-pane-row");
    row1.appendChild(el("span", "crop-lbl", "crop"));

    const startInp = el("input"); startInp.type = "number"; startInp.className = "crop-input";
    startInp.min = 0; startInp.max = entry.totalSamples - 1;
    startInp.value = entry.cropStart || 0; startInp.title = "Start sample";

    const endInp = el("input"); endInp.type = "number"; endInp.className = "crop-input";
    endInp.min = 1; endInp.max = entry.totalSamples;
    endInp.value = entry.cropEnd || entry.totalSamples; endInp.title = "End sample";

    const infoSpan = el("span", "crop-info");
    function updateInfo() {
      const s = parseInt(startInp.value) || 0;
      const e = parseInt(endInp.value) || entry.totalSamples;
      const len = Math.max(0, e - s);
      const sr = entry.sampleRate || 1;
      infoSpan.textContent = `${len} · ${(len / sr).toFixed(2)}s`;
    }
    updateInfo();

    startInp.addEventListener("change", () => {
      entry.cropStart = Math.max(0, parseInt(startInp.value) || 0);
      syncInputsFromEntry(); scheduleCommit();
    });
    endInp.addEventListener("change", () => {
      entry.cropEnd = Math.min(entry.totalSamples, parseInt(endInp.value) || entry.totalSamples);
      syncInputsFromEntry(); scheduleCommit();
    });

    row1.appendChild(startInp);
    row1.appendChild(el("span", "crop-sep", "–"));
    row1.appendChild(endInp);
    row1.appendChild(infoSpan);
    row1.appendChild(status);
    pane.appendChild(row1);

    // ── Row 2: power-of-2 length presets ─────────────────────────────────────
    const row2 = el("div", "audio-pane-row audio-pane-presets");
    const _setLength = (plen) => {
      const center = ((entry.cropStart || 0) + (entry.cropEnd || entry.totalSamples)) / 2;
      entry.cropStart = Math.max(0, Math.round(center - plen / 2));
      entry.cropEnd = entry.cropStart + plen;
      if (entry.cropEnd > entry.totalSamples) {
        entry.cropEnd = entry.totalSamples;
        entry.cropStart = Math.max(0, entry.cropEnd - plen);
      }
      syncInputsFromEntry(); scheduleCommit();
    };

    const snapBtn = el("button", "audio-preset-btn audio-snap-btn", "snap p2");
    snapBtn.title = "Snap selection length to nearest power of 2 (centered)";
    snapBtn.addEventListener("click", () =>
      _setLength(_nearestP2((entry.cropEnd || entry.totalSamples) - (entry.cropStart || 0))));
    row2.appendChild(snapBtn);

    const p2Lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
    const p2Labels = ['512', '1k', '2k', '4k', '8k', '16k', '32k', '64k'];
    p2Lengths.forEach((plen, i) => {
      if (plen > entry.totalSamples) return;
      const btn = el("button", "audio-preset-btn", p2Labels[i]);
      btn.title = `Set selection to ${plen} samples`;
      btn.addEventListener("click", () => _setLength(plen));
      row2.appendChild(btn);
    });

    const fullBtn = el("button", "audio-preset-btn", "full");
    fullBtn.title = "Select entire file";
    fullBtn.addEventListener("click", () => {
      entry.cropStart = 0; entry.cropEnd = entry.totalSamples;
      syncInputsFromEntry(); scheduleCommit();
    });
    row2.appendChild(fullBtn);
    pane.appendChild(row2);

    // ── Row 3: processing options ────────────────────────────────────────────
    const row3 = el("div", "audio-pane-row audio-pane-opts");
    function _optLabel(text, checked, onChange) {
      const lbl = el("label", "audio-opt-label");
      const cb = el("input");
      cb.type = "checkbox"; cb.checked = checked;
      cb.addEventListener("change", () => { onChange(cb.checked); scheduleCommit(); });
      lbl.appendChild(cb);
      lbl.appendChild(document.createTextNode(" " + text));
      return lbl;
    }
    row3.appendChild(_optLabel("normalize", entry.normalize || false, v => { entry.normalize = v; }));
    if (entry.numChannels && entry.numChannels > 1) {
      const monoLbl = _optLabel(`mono (${entry.numChannels}ch)`, entry.toMono || false,
                                v => { entry.toMono = v; entry.monoByShape = false; });
      monoLbl.title = entry.monoByShape
        ? "On by default: this input was traced with a single channel, so the "
          + "file is mixed down to match. Untick to send it as it is."
        : "Mix every channel down to one";
      row3.appendChild(monoLbl);
    }

    row3.appendChild(el("span", "crop-lbl", "sr"));
    const srInp = el("input");
    srInp.type = "number"; srInp.className = "crop-input audio-sr-input";
    srInp.min = 100; srInp.value = entry.targetSR || entry.sampleRate;
    srInp.title = "Target sample rate — resamples on export";
    srInp.addEventListener("change", () => {
      entry.targetSR = Math.max(100, parseInt(srInp.value) || entry.sampleRate);
      scheduleCommit();
    });
    row3.appendChild(srInp);
    pane.appendChild(row3);

    // ── Initial waveform draw (after layout) ─────────────────────────────────
    const ro = new ResizeObserver(() => {
      const w = canvasWrap.clientWidth;
      if (w > 0 && canvas.width !== w) {
        canvas.width = w;
        entry._wfPeaks = null;
        TBAudio.draw(canvas, entry);
      }
    });
    ro.observe(canvasWrap);
    requestAnimationFrame(() => {
      canvas.width = canvasWrap.clientWidth || 240;
      TBAudio.draw(canvas, entry);
    });

    host.appendChild(pane);
    if (!entry.croppedFile) await TBAudio.process(entry);
    return pane;
  };

  window.TBAudio = TBAudio;
})();


/* TorchBend — shared input helpers (TBInputs) */
(function () {
  "use strict";
  const TBInputs = {};

  TBInputs.isImageFile = function (f) {
    return !!f && ((f.type || "").startsWith("image/") ||
                   /\.(png|jpe?g|gif|bmp|webp|tiff?)$/i.test(f.name || ""));
  };

  // Which input pane a placeholder deserves.
  //
  // Shape alone gets this wrong in the two ways that matter most. An integer
  // tensor is never a waveform or a picture however it is shaped — token ids
  // are `[B, T]` exactly like mono audio — and a 3D float tensor is only audio
  // if its channel dimension is a channel count: VITS hands its vocoder a
  // `[1, 192, T]` latent, which is not something you can drop a .wav onto.
  //
  // So dtype is consulted first, then the dimensions are read for what they
  // actually have to be. `opts` carries `{dtype, name}`.
  const _INT_DTYPE = /^(u?int|long|short|byte|char)/;
  // Integer tensors all look alike; their names do not. These are the argument
  // names the transformer world settled on, and they say more about what a
  // tensor is for than its shape ever could — `attention_mask` and `input_ids`
  // are both `[B, T]` of int64 and are not remotely the same thing.
  const _MASK_NAME  = /mask/i;
  const _INDEX_NAME = /(^|_)(position_ids|token_type_ids|type_ids|idx|index|indices|lengths?|offsets?)$/i;
  const _TOKEN_NAME = /(^|_)(input_ids|ids|tokens?|token_ids|codes?|codebook)$/i;
  const _IMAGE_CHANNELS = new Set([1, 2, 3, 4]);   // grey, grey+α, rgb, rgba
  const _AUDIO_CHANNELS = new Set([1, 2]);         // mono, stereo
  const _MIN_AUDIO_LEN = 256;

  TBInputs.guessType = function (shape, opts) {
    const o = opts || {};
    const dtype = (o.dtype || "").toLowerCase();
    // An interface that declares how to fill this input has already answered.
    // A declared input mode says the interface knows how to *fill* this input,
    // not what the numbers are: Bark's codec decoder takes a prompt and a float
    // latent. The mode bar names itself, so dtype and shape answer this as usual.
    const name = o.name || "";
    if (dtype === "bool") return "mask";
    if (_INT_DTYPE.test(dtype)) {
      // Integral either way: an expression the user types, never a file they
      // drop. Which of the three it is comes down to the name.
      if (_MASK_NAME.test(name)) return "mask";
      if (_INDEX_NAME.test(name)) return "indices";
      if (_TOKEN_NAME.test(name)) return "tokens";
      return "indices";
    }
    // A float tensor called a mask is still a mask.
    if (_MASK_NAME.test(name)) return "mask";
    if (!shape || shape.length === 0) return "numeric";
    if (shape.length >= 4) {
      // [B, C, H, W] — C has to be a plausible channel count
      return _IMAGE_CHANNELS.has(shape[shape.length - 3]) ? "image" : "tensor";
    }
    if (shape.length === 3) {
      // [B, C, T] — a channel count and something long enough to hear
      return (_AUDIO_CHANNELS.has(shape[1]) && shape[2] >= _MIN_AUDIO_LEN)
        ? "audio" : "tensor";
    }
    if (shape.length === 2) {
      // [B, T] of floats: a batch of mono waveforms, if T is long enough
      return shape[1] >= 512 ? "audio" : "tensor";
    }
    if (shape.length === 1) return shape[0] >= 512 ? "audio" : "numeric";
    return "numeric";
  };

  window.TBInputs = TBInputs;
})();
