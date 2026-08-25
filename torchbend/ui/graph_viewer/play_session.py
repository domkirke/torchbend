"""Play mode: a fast-interaction runtime for a bended model.

The graph viewer is for *building* a bending (inspecting activations, attaching
callbacks, wiring macros).  Play mode is for *performing* with it: it freezes
the current bending into the fastest callable it can produce and exposes only
the defined macros (BendingParameters) plus an input bench.

Two runtimes are supported, chosen automatically by :meth:`PlaySession.compile`:

* ``"scripted"`` — ``bended_module.script()`` (a TorchScript module).  Macro
  changes go through generated ``set_<name>`` setters and never rebuild the
  graph, so interaction is as fast as the model itself.  This is preferred but
  not always possible (some callbacks / models do not script).
* ``"eager"`` — the bended ``GraphModule``.  Activation-macro changes are free
  (the callback runs inside the graph and shares the live BendingParameter);
  weight-macro changes require rebuilding the module's parameter copy, which is
  flagged so :meth:`run` rebuilds lazily.

  The eager path goes through the same demand-driven activation cache as the
  editor: moving a macro invalidates only the bended node and what reads it, so
  the next run resumes from the nearest clean ancestor instead of re-running the
  whole model.  With the macro late in the graph this is the difference between
  a usable slider and a full forward pass per pixel of travel.

Both runtimes operate on their own copy of the weights, so moving them to an
accelerator never disturbs the model used by the rest of the graph viewer.
"""
import io
import logging
import time
import torch

from torchbend.tracing import activation_log as actlog


# ── device discovery ────────────────────────────────────────────────────────

def available_devices() -> list:
    """Return the torch devices usable for play mode (always includes 'cpu')."""
    devices = ["cpu"]
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
    except Exception:
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    return devices


def _device_label(dev: str) -> str:
    if dev == "cpu":
        return "CPU"
    if dev == "mps":
        return "MPS (Apple)"
    if dev.startswith("cuda"):
        try:
            idx = int(dev.split(":")[1]) if ":" in dev else 0
            return f"{torch.cuda.get_device_name(idx)} ({dev})"
        except Exception:
            return f"CUDA ({dev})"
    return dev


def device_options() -> list:
    return [{"value": d, "label": _device_label(d)} for d in available_devices()]


# ── PlaySession ──────────────────────────────────────────────────────────────

class PlaySession:
    """Holds the compiled runtime for one model + method, for fast interaction."""

    def __init__(self):
        self.fn: str = "forward"
        self.device: str = "cpu"
        self.mode: str | None = None          # "scripted" | "eager" | None
        self.error: str | None = None          # scripting error (when fell back)
        self.last_run_ms: float | None = None

        self._scripted = None                  # TorchScript module (scripted mode)
        self._bm = None                        # the source BendedModule
        self._weight_macros: set = set()       # macro names driving weight callbacks
        self._output_nodes: list = []          # graph nodes feeding the output
        self._orig_device: str = "cpu"         # module device before play moved it
        self._eager_gm = None                  # compiled eager runtime (fallback)
        self._needs_rebuild: bool = False      # a weight macro moved: rebuild it
        self._session = None                   # editor BendingSession (macro source)
        self._cache = None                     # ActivationCache for partial reruns
        self._bent_module = None               # bent module the cache slices target
        self._bent_graph = None
        self._macro_node_cache = None          # macro name → nodes it drives
        self._macro_resume = None              # nodes to keep so moves resume mid-graph
        # BendingParameters pulled in by name from the editor. Everything the
        # module or the editor session already knows is exposed automatically;
        # this holds the ones asked for explicitly, and survives a recompile.
        self._imported: dict = {}

    # ── compile ──────────────────────────────────────────────────────────────

    def compile(self, bended_module, fn: str, device: str,
                prefer_scripted: bool = False, session=None) -> dict:
        """(Re)build the runtime for *fn* on *device*.

        The default ``"eager"`` runtime evaluates the bended graph through
        ``get_activations`` on the output node(s); this reliably reflects live
        macro (BendingParameter) changes.  ``prefer_scripted=True`` attempts a
        TorchScript build for raw speed — but note that, with the current
        ``as_input`` handling, scripted macros may not propagate; it is offered
        as an experimental fast path.
        """
        if device not in available_devices():
            raise ValueError(f"Device '{device}' is not available")
        if fn not in getattr(bended_module, "_graphs", {}):
            raise ValueError(f"Method '{fn}' is not traced")

        # restore any previous device move before switching
        self._restore_device()

        self.fn = fn
        self.device = device
        self.error = None
        self._scripted = None
        self._bm = bended_module
        self._session = session or self._session
        self._weight_macros = self._detect_weight_macros(bended_module, self._session)
        self._output_nodes = self._compute_output_nodes(bended_module, fn)
        self._eager_gm = None
        self._needs_rebuild = False
        # note: self._imported is deliberately *not* cleared — a macro the user
        # asked for should still be there after switching method or device
        # cached activations belong to one (fn, device, bending topology) — a
        # recompile changes at least one of those, so start clean
        self._reset_cache()

        t0 = time.perf_counter()
        if prefer_scripted:
            try:
                scripted = bended_module.script()
                scripted = scripted.to(device)
                scripted.eval()
                self._scripted = scripted
                self.mode = "scripted"
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
        if self._scripted is None:
            self.mode = "eager"
            # eager runs get_activations on the live module — move it to the
            # requested device (restored on the next compile / release).
            self._move_device(device)
        compile_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "mode": self.mode,
            "fn": self.fn,
            "device": self.device,
            "scripting_error": self.error,
            "compile_ms": round(compile_ms, 1),
        }

    @staticmethod
    def _compute_output_nodes(bended_module, fn) -> list:
        """Names of the bended graph node(s) that feed the output op for *fn*.

        Uses the *bended* graph so terminal ``<node>_bended`` callback nodes are
        captured — those carry the final, macro-driven values.
        """
        import torch.fx as _fx
        try:
            graph = bended_module.bend_graph(fn=fn)
        except Exception:
            return []
        names = []
        for node in graph.nodes:
            if node.op != "output":
                continue
            ret = node.args[0] if node.args else None
            for n in (ret if isinstance(ret, (tuple, list)) else [ret]):
                if isinstance(n, _fx.Node):
                    names.append(n.name)
        return names

    def _move_device(self, device: str) -> None:
        if device == "cpu" or self._bm is None:
            self._orig_device = "cpu"
            return
        try:
            params = list(self._bm._module.parameters())
            self._orig_device = str(params[0].device) if params else "cpu"
            self._bm._module.to(device)
        except Exception:
            self._orig_device = "cpu"

    def _restore_device(self) -> None:
        if self._bm is not None and getattr(self, "device", "cpu") != "cpu" \
                and self._orig_device == "cpu":
            try:
                self._bm._module.to("cpu")
            except Exception:
                pass

    def release(self) -> None:
        """Restore the module to its original device (call when leaving play mode)."""
        self._restore_device()
        self._scripted = None
        self._eager_gm = None

    def invalidate(self) -> None:
        """Force the eager runtime to be rebuilt (bendings changed under us)."""
        self._eager_gm = None
        self._reset_cache()

    # ── activation cache ─────────────────────────────────────────────────────

    def _reset_cache(self) -> None:
        """Drop every cached activation, compiled slice and bent-module handle.

        Needed whenever the graph itself changes (recompile, binding added), as
        the compiled slices hold a reference to the module they were built for.
        """
        from .activation_cache import ActivationCache
        if self._cache is None:
            try:
                from django.conf import settings as _ds
                max_bytes = getattr(_ds, "ACTIVATION_CACHE_MAX_BYTES", 512 * 1024 * 1024)
            except Exception:
                max_bytes = 512 * 1024 * 1024
            self._cache = ActivationCache(max_bytes=max_bytes)
        else:
            self._cache.clear()
            self._cache.clear_slices()
        self._bent_module = None
        self._bent_graph = None
        self._macro_node_cache = None
        self._macro_resume = None

    def _bent(self):
        """The bent module + graph the cache slices are built against."""
        if self._bent_module is None or self._bent_graph is None or self._needs_rebuild:
            with actlog.step("play.bend_module", "fn=%s" % self.fn, level=logging.INFO):
                self._bent_graph = self._bm.bend_graph(fn=self.fn)
                self._bent_module = self._bm.bend_module(fn=self.fn)
            # everything built against the previous module is stale
            self._cache.clear_slices(self.fn)
            self._eager_gm = None
            self._needs_rebuild = False
        return self._bent_module, self._bent_graph

    def mark_dirty(self, nodes) -> None:
        """Invalidate cached activations downstream of *nodes*."""
        if self._cache is None or not nodes:
            return
        from .bending_session import mark_nodes_dirty
        mark_nodes_dirty(self._cache, self._bent_graph, self.fn, nodes)

    def cache_stats(self) -> dict:
        return self._cache.stats() if self._cache is not None else {}

    def _eager_runtime(self):
        """The graph trimmed to the output node(s), compiled once and reused.

        Rebuilding it per run — which is what going through
        ``BendedModule.get_activations`` every time amounts to — re-copies the
        parameters, re-inserts the callbacks and re-runs the fx codegen before
        any arithmetic happens. Activation macros are shared callback objects,
        so they stay live inside the compiled graph; only weight macros, which
        rewrite the module's parameter copy, need a rebuild.
        """
        from torchbend.tracing.graph import graph_get_activations
        from torchbend.tracing.graphmodule import BendedGraphModule

        if self._eager_gm is not None and not self._needs_rebuild:
            return self._eager_gm
        # built from the same bent module the cache slices target, so a weight
        # macro rebuilds both together
        module, graph = self._bent()
        with actlog.step("play.compile_eager",
                         "fn=%s  outputs=%s" % (self.fn, actlog.fmt_names(self._output_nodes)),
                         level=logging.INFO):
            trimmed = graph_get_activations(graph, self._output_nodes)
            self._eager_gm = BendedGraphModule(module, **{self.fn: trimmed})
            actlog.log("compiled %s", actlog.fmt_graph(trimmed))
        return self._eager_gm

    def _detect_weight_macros(self, bended_module, session=None) -> set:
        """Names of BendingParameters that drive at least one weight callback.

        Read straight from the module, so it works whether the bending was set
        up in Python or through the graph-viewer session.
        """
        from .bending_session import _cb_holds
        weight_macros = set()
        try:
            controllables = self.all_macros(bended_module, session)
            for _param, cbs in bended_module.bended_weights.items():
                for cb in cbs:
                    for name, bp in controllables.items():
                        # _cb_holds also sees a parameter *derived* from the macro,
                        # which is what a range-mapped link installs
                        if _cb_holds(cb, bp):
                            weight_macros.add(name)
        except Exception:
            pass
        return weight_macros

    @property
    def compiled(self) -> bool:
        return self.mode is not None and self._bm is not None

    # ── macros ─────────────────────────────────────────────────────────────────

    def all_macros(self, bended_module, session=None) -> dict:
        """All manipulable BendingParameters, as ``{name: BendingParameter}``.

        ``bended_module.controllables()`` only knows a macro once it has been
        *linked* to a callback param, so a macro created in the editor but not
        wired to anything yet would be invisible here. The graph-viewer session
        holds those too, and it is the source of truth for the UI — union both,
        module first (a linked macro is the same object in either), then anything
        imported by name through :meth:`import_macro`.
        """
        # The editor session holds the macro objects themselves; the module may
        # hold a parameter *derived* from one (a link maps a 0…1 macro onto its
        # target's range). Session first, so a name always resolves to the macro
        # the user drives rather than to one of its mapped views.
        macros = dict(getattr(session, "bending_params", {}) or {})
        try:
            for name, bp in bended_module.controllables().items():
                macros.setdefault(name, bp)
        except Exception:
            pass
        for name, bp in dict(self._imported).items():
            macros.setdefault(name, bp)
        return macros

    @staticmethod
    def promotable_params(session=None) -> list:
        """Callback parameters in the editor that no macro drives yet.

        A binding's parameters (an ``Affine``'s ``scale``, a mask's ``prob``) are
        plain values until one is *promoted*: given a BendingParameter and linked
        to it. Only a promoted parameter can be moved from play mode, so this is
        the catalogue of the ones still waiting — the picker offers them, and the
        promotion happens here instead of through a detour in the editor.
        """
        out = []
        try:
            bindings = session.list_bindings()
        except Exception:
            return out
        for b in bindings:
            desc = (b.get("descriptor") or {}).get("params") or {}
            links = b.get("bp_links") or {}
            values = b.get("params") or {}
            for pname, pd in desc.items():
                if pd.get("visible") is False or pname in links:
                    continue
                rng = list(pd.get("range") or [None, None])
                out.append({
                    "binding":       b.get("id"),
                    "param":         pname,
                    "label":         pd.get("label") or pname,
                    "node":          b.get("node") or "",
                    "callback_type": b.get("callback_type") or "",
                    "param_type":    pd.get("type") or "float",
                    "value":         values.get(pname, pd.get("default")),
                    "min":           rng[0] if len(rng) > 0 else None,
                    "max":           rng[1] if len(rng) > 1 else None,
                })
        return out

    def import_macro(self, bended_module, name: str, session=None) -> bool:
        """Expose the BendingParameter *name* in play mode. True if it was added.

        Most parameters are picked up on their own; this is for the ones that are
        not — created in the editor after this runtime was compiled, or living on
        a callback the module does not track. Looks the name up wherever it can
        be found and pins it, so it survives the next recompile too.
        """
        session = session or self._session
        if name in self.all_macros(bended_module, session):
            return False
        bp = dict(getattr(session, "bending_params", {}) or {}).get(name)
        if bp is None:
            raise KeyError(f"No BendingParameter named '{name}' in the editor")
        self._imported[name] = bp
        self._macro_node_cache = None      # it may drive nodes we have not mapped
        return True

    def known_params(self, bended_module, session=None) -> list:
        """Every BendingParameter the editor knows, flagged with whether play mode
        already exposes it — the catalogue behind the macro import picker."""
        from .bending_session import param_python_value, param_type_str
        exposed = set(self.all_macros(bended_module, session or self._session))
        out = {}
        for src in (dict(getattr(session or self._session, "bending_params", {}) or {}),
                    self._imported):
            for name, bp in src.items():
                if name in out:
                    continue
                out[name] = {
                    "name": name,
                    "value": param_python_value(bp),
                    "param_type": param_type_str(bp),
                    "min": bp.min_clamp,
                    "max": bp.max_clamp,
                    "in_play": name in exposed,
                }
        return sorted(out.values(), key=lambda d: d["name"])

    def list_macros(self, bended_module, session=None) -> list:
        """Manipulable macros = the module's BendingParameters (``controllables``)
        plus any the editor session defines but has not linked yet.

        Each entry reports the value, range and whether changing it is "fast"
        (no graph rebuild) under the active runtime.
        """
        from .bending_session import param_python_value, param_type_str, is_normalized
        macros = []
        sess = session or self._session
        ranges = dict(getattr(sess, "bp_ranges", {}) or {})
        controllables = self.all_macros(bended_module, sess)
        for name, bp in controllables.items():
            type_str = param_type_str(bp)
            is_weight = name in self._weight_macros
            # Eager re-evaluates the graph each run, so every macro is "live".
            fast = (self.mode == "scripted") or (not is_weight)
            # what this macro's 0…1 actually spans, for the readout next to the
            # slider — the link's own mapping when there is exactly one, else the
            # range the macro was created with
            # the range belongs to each link; a single shared answer exists only
            # when every link agrees (or when there are none, and the macro's own
            # creation range is what a new link would get)
            attachments = self.macro_links(sess, name)
            links = [tuple(a["range"]) for a in attachments if a["range"]]
            uniq = set(links)
            span = (links[0] if len(uniq) == 1
                    else (ranges.get(name) if not attachments else None))
            macros.append({
                "name":       name,
                # native python value, so the client sees a bool as a bool
                "value":      param_python_value(bp),
                "param_type": type_str,
                "min":        bp.min_clamp,
                "max":        bp.max_clamp,
                "normalized": is_normalized(bp),
                "target_range": list(span) if span else None,
                "n_links":    len(attachments),
                # every derived parameter this macro feeds, for the readout panel
                "links":      attachments,
                "drives_weight": is_weight,
                "fast":       fast,
            })
        return macros

    @staticmethod
    def macro_links(session, name: str) -> list:
        """Every attachment this macro drives — where it lands, over what range,
        and what the target is reading right now.

        One macro can span a different range on each of its links, so "the value
        of this macro" is really several values. This is all of them.
        """
        from .bending_session import _read_param
        out = []
        for bid, b in (getattr(session, "bindings", {}) or {}).items():
            for pname, bpname in (b.get("bp_links") or {}).items():
                if bpname != name:
                    continue
                cb = b.get("callback")
                rng = (b.get("bp_maps") or {}).get(pname)
                out.append({
                    "binding":       bid,
                    "node":          b.get("node") or "",
                    "param":         pname,
                    "callback_type": type(cb).__name__ if cb is not None else "",
                    "range":         [float(rng[0]), float(rng[1])] if rng else None,
                    "value":         _read_param(cb, pname) if cb is not None else None,
                })
        return sorted(out, key=lambda d: (d["node"], d["param"]))

    @classmethod
    def _macro_link_ranges(cls, session, name: str) -> list:
        """The ``[lo, hi]`` each of this macro's links maps its 0…1 onto."""
        return [tuple(l["range"]) for l in cls.macro_links(session, name) if l["range"]]

    def set_macro(self, bended_module, name: str, value, session=None) -> dict:
        """Set a macro value as fast as possible. Returns {value, needs_rebuild}."""
        from .bending_session import param_python_value
        controllables = self.all_macros(bended_module, session or self._session)
        if name not in controllables:
            raise KeyError(f"Macro '{name}' not found")
        bp = controllables[name]
        coerced = self._coerce_macro(bp, value)

        # Always update the live BendingParameter on the module: the eager
        # runtime reads it via get_activations, and it keeps the panel in sync.
        # `update` refreshes every callback the module has registered for it;
        # a macro the module doesn't track (created in the editor, not linked to
        # a callback yet) is set on the parameter itself — set_value already
        # notifies whatever callbacks are attached to it.
        try:
            bended_module.update(name, coerced)
        except Exception:
            bp.set_value(coerced)
        if self.mode == "scripted" and self._scripted is not None:
            setter = getattr(self._scripted, f"set_{name}", None)
            if setter is not None:
                setter(coerced)
        else:
            # Invalidate only what this macro actually drives — the bended nodes
            # and their descendants. Everything upstream keeps its cached value
            # and becomes the frontier the next run resumes from.
            self.mark_dirty(self._macro_nodes(bended_module, name, bp))
            if name in self._weight_macros:
                # weight callbacks rewrote the parameters; the runtime holds its
                # own copy, so it has to be rebuilt on the next run
                self._needs_rebuild = True

        return {"value": param_python_value(bp),
                "needs_rebuild": self._needs_rebuild}

    def _macro_nodes(self, bended_module, name: str, bp) -> list:
        """Bended nodes / weights driven by macro *name* (memoised per compile)."""
        if self._macro_node_cache is None:
            self._macro_node_cache = {}
        if name not in self._macro_node_cache:
            from .bending_session import nodes_driven_by
            self._macro_node_cache[name] = nodes_driven_by(bended_module, self.fn, bp)
            actlog.log("macro    %s drives %s", name,
                       actlog.fmt_names(self._macro_node_cache[name]))
        return self._macro_node_cache[name]

    @staticmethod
    def _coerce_macro(bp, value):
        from .bending_session import coerce_param_value
        return coerce_param_value(bp, value)

    # ── run ──────────────────────────────────────────────────────────────────

    def _ordered_placeholders(self) -> list:
        graph = self._bm.graph(fn=self.fn, bended=True)
        return [n.name for n in graph.nodes if n.op == "placeholder"]

    def run(self, kwargs: dict):
        """Run a forward pass and return the raw output (tensors stay on device).

        ``kwargs`` is keyed by placeholder name; inputs are moved to the active
        device.  Sets :attr:`last_run_ms`.
        """
        if not self.compiled:
            raise RuntimeError("Play session is not compiled")
        dev = self.device
        moved = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()}

        t0 = time.perf_counter()
        with actlog.step("play.run", "mode=%s  fn=%s  device=%s" % (self.mode, self.fn, dev),
                         level=logging.INFO):
            with torch.no_grad():
                if self.mode == "scripted" and self._scripted is not None:
                    order = self._ordered_placeholders()
                    args = [moved[name] for name in order if name in moved]
                    fnc = self._scripted if self.fn == "forward" else getattr(self._scripted, self.fn)
                    actlog.log("scripted call — no graph rebuild")
                    out = fnc(*args)
                else:
                    out = self._run_eager(moved)
        # mps is async; sync for honest timing
        try:
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            elif dev == "mps":
                torch.mps.synchronize()
        except Exception:
            pass
        self.last_run_ms = (time.perf_counter() - t0) * 1000.0
        return out

    def _resume_points(self, graph, bended_acts) -> list:
        """Nodes to materialise so a macro move can resume mid-graph.

        A callback bound to X is inserted as a separate ``X_bended`` node, so X
        itself keeps its value when the macro moves — it is exactly the point the
        next run should restart from. Nothing else requests X though, so unless
        it is asked for here the cache holds only the output and every move
        recomputes the model from the inputs.
        """
        if self._macro_resume is not None:
            return self._macro_resume
        names = {n.name for n in graph.nodes}
        outs = set(self._output_nodes)
        self._macro_resume = [n for n in sorted(bended_acts)
                              if f"{n}_bended" in names and n not in outs]
        actlog.log("play     resume points %s", actlog.fmt_names(self._macro_resume))
        return self._macro_resume

    def _run_eager(self, moved: dict):
        """Compute the output node(s) through the activation cache.

        Only what a macro actually invalidated is recomputed: the cache resumes
        from the nearest clean ancestor of each output node. Falls back to the
        whole trimmed graph if the cache cannot serve the outputs (no cache slot
        for them, or a slicing failure).
        """
        from .activation_cache import run_activations_with_cache

        cache_exc = None
        if self._output_nodes and self._cache is not None:
            try:
                module, graph = self._bent()
                bended = set(self._bm.bended_activations(self.fn).keys())
                targets = self._resume_points(graph, bended) + list(self._output_nodes)
                acts, warn = run_activations_with_cache(
                    self._bm, self.fn, moved, self._cache,
                    target_nodes=targets,
                    bended_nodes=bended,
                    # outputs and resume points are what play mode lives on —
                    # evict anything else first
                    pinned_nodes=set(targets),
                    bent_module=module, bent_graph=graph,
                )
                if warn:
                    actlog.log("cache    %s", warn, level=logging.WARNING)
                outs = [acts[n] for n in self._output_nodes if n in acts]
                if len(outs) == len(self._output_nodes):
                    return outs[0] if len(outs) == 1 else tuple(outs)
                actlog.log("cache    served %d/%d output node(s) — running the full graph",
                           len(outs), len(self._output_nodes), level=logging.WARNING)
            except Exception as exc:
                cache_exc = exc
                actlog.log("cache    FAILED (%s: %s) — running the full graph",
                           type(exc).__name__, exc, level=logging.WARNING)

        # fallback: the compiled output-node graph, trimmed like get_activations
        # would but built once
        gm = self._eager_runtime()
        fn_method = getattr(gm, self.fn)
        inputs_obj = self._bm.inputs_for_fn(fn_method, dict(moved))
        try:
            return fn_method(*inputs_obj, **inputs_obj)
        except Exception:
            # Both routes run the same model on the same inputs, so when the
            # cache path already failed it holds the root cause — the fallback's
            # own error is a consequence of it and says nothing useful.
            if cache_exc is not None:
                raise cache_exc
            raise

    def run_to_cpu(self, kwargs: dict):
        """Like :meth:`run` but detaches/moves every output tensor to CPU."""
        out = self.run(kwargs)
        return _to_cpu(out), self.last_run_ms


def _to_cpu(out):
    if torch.is_tensor(out):
        return out.detach().to("cpu")
    if isinstance(out, (list, tuple)):
        return type(out)(_to_cpu(o) for o in out)
    if isinstance(out, dict):
        return {k: _to_cpu(v) for k, v in out.items()}
    return out


def flatten_outputs(out) -> list:
    """Flatten a model output into an ordered list of (label, tensor) pairs."""
    pairs = []

    def _walk(obj, label):
        if torch.is_tensor(obj):
            pairs.append((label, obj))
        elif isinstance(obj, (list, tuple)):
            for i, o in enumerate(obj):
                _walk(o, f"{label}[{i}]" if label else f"out[{i}]")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{label}.{k}" if label else str(k))
        # silently ignore non-tensor leaves (e.g. distributions) for viz

    _walk(out, "")
    if len(pairs) == 1 and not pairs[0][0]:
        pairs = [("output", pairs[0][1])]
    return pairs
