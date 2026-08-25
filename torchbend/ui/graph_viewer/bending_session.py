import io
import logging
import uuid
import inspect
import torch
from torchbend.bending.callback import BendingCallback
from torchbend.tracing import activation_log as actlog
from .activation_cache import ActivationCache, run_activations_with_cache

logger = logging.getLogger("torchbend.bending_session")

_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)

# ── callback discovery ─────────────────────────────────────────────────────────

_CALLBACK_CLASSES = None


def _discover_callback_classes():
    import torchbend.bending as _tb
    classes = {}
    for name in dir(_tb):
        obj = getattr(_tb, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BendingCallback)
            and obj is not BendingCallback
        ):
            classes[name] = obj
    return classes


def get_callback_classes() -> dict:
    global _CALLBACK_CLASSES
    if _CALLBACK_CLASSES is None:
        _CALLBACK_CLASSES = _discover_callback_classes()
    return _CALLBACK_CLASSES


def get_available_callbacks() -> list:
    """Serialisable list of ui_descriptor dicts for ui_compatible callbacks only."""
    return [
        cls.ui_descriptor()
        for cls in get_callback_classes().values()
        if getattr(cls, "ui_compatible", True) and cls.controllable_params
    ]


def _find_callback_class(name: str) -> type:
    classes = get_callback_classes()
    if name not in classes:
        available = list(classes.keys())
        raise KeyError(f"Unknown callback type '{name}'. Available: {available}")
    return classes[name]


def _coerce(val, type_str):
    """Coerce a value coming from JSON to the right Python type."""
    if type_str == "int":
        try:
            return int(round(float(val)))
        except (TypeError, ValueError):
            return 0
    elif type_str == "float":
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0
    return val


def param_type_str(bp) -> str:
    """The declared python type of a BendingParameter, as a string."""
    from torchbend.bending.parameter import BendingParamType
    return BendingParamType.param_hash().get(bp.param_type, "float")


def param_python_value(bp):
    """JSON-safe current value of a BendingParameter, keeping its python type."""
    val = bp.get_python_value()
    return val.tolist() if torch.is_tensor(val) else val


def coerce_param_value(bp, value):
    """Cast *value* to the type declared by *bp*, or raise ValueError.

    ``BendingParameter.set_value`` is strict: a bool parameter refuses a float
    outright, and an int one truncates silently. Values arriving from the UI are
    JSON numbers with no type attached, so every entry point that sets a macro
    has to come through here.
    """
    type_str = param_type_str(bp)
    try:
        if type_str == "bool":
            if isinstance(value, str):
                return value.strip().lower() not in ("", "0", "false", "no", "off")
            return bool(value)
        if type_str == "int":
            return int(round(float(value)))
        if type_str == "float":
            return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"macro '{bp.name}' is of type {type_str}; cannot take value {value!r}"
        ) from exc
    return value


def is_normalized(bp) -> bool:
    """True for a macro that lives in 0…1 — the form the editor creates.

    A BendingParameter built directly in Python may carry any range; it keeps it,
    and its range stays its clamp rather than a mapping. Checking the clamp (not
    just the type) is what tells the two apart.
    """
    return (param_type_str(bp) == "float"
            and bp.min_clamp is not None and bp.max_clamp is not None
            and float(bp.min_clamp) == 0.0 and float(bp.max_clamp) == 1.0)


def param_types_for(cb, param_name) -> list:
    """Type names a callback param accepts, mirroring ``_check_controllable_type``.

    An empty list means the callback declares no type hint for it, in which case
    anything is allowed.
    """
    from torchbend.bending.parameter import BendingParamType
    from torchbend.utils import checktuple
    hint = type(cb).controllable_params.get(param_name, (None, None))[0]
    if hint is None:
        return []
    names = []
    for t in checktuple(hint):
        try:
            names.append(BendingParamType._str_from_type(t))
        except Exception:
            pass
    return names


def mark_nodes_dirty(cache, graph, fn: str, nodes) -> None:
    """Invalidate *nodes* and their descendants in *cache*.

    ``nodes`` mixes graph node names and weight parameter paths ("conv1.weight"),
    which are not graph nodes and have to be resolved first. Shared by the editor
    session and by play mode, which keeps its own cache but invalidates it on the
    very same events.
    """
    module_to_act: dict = {}
    attr_to_node: dict = {}
    graph_node_names: set = set()
    if graph is not None:
        graph_node_names = {n.name for n in graph.nodes}
        for n in graph.nodes:
            if n.op == "call_module":
                module_to_act[str(n.target)] = n.name
            elif n.op == "get_attr":
                # weights show up as get_attr nodes whose name has the dots
                # of the parameter path turned into underscores
                attr_to_node[str(n.target)] = n.name
                attr_to_node[n.name] = n.name

    # (node, is_weight) — a weight change alters the activation itself,
    # an activation callback only alters what reads it.
    effective_nodes = []
    for node in nodes:
        if graph is not None and node not in graph_node_names and '.' in node:
            # A weight param name (e.g. "conv1.weight") is not a graph node.
            # It reaches the graph either as the call_module node of its
            # parent module, or as its own get_attr node ("conv1_weight").
            parent = node.rsplit('.', 1)[0]
            act_node = module_to_act.get(parent)
            if act_node:
                actlog.log("resolve  weight '%s' → activation node '%s'", node, act_node)
                effective_nodes.append((act_node, True))
                continue
            attr_node = attr_to_node.get(node) or attr_to_node.get(node.replace('.', '_'))
            if attr_node:
                actlog.log("resolve  weight '%s' → get_attr node '%s'", node, attr_node)
                effective_nodes.append((attr_node, True))
                continue
            actlog.log("resolve  weight '%s' matches no graph node — nothing invalidated",
                       node, level=logging.WARNING)
        effective_nodes.append((node, False))

    for node, is_weight in effective_nodes:
        # An activation callback inserted as its own "<node>_bended" node
        # leaves <node>'s value untouched, so its cached tensor survives and
        # can serve as the frontier for the recomputation. A weight change,
        # or a callback rewriting the node in place (applied_to_node, which
        # produces no "_bended" node), does change the node itself.
        keeps_value = (not is_weight) and (f"{node}_bended" in graph_node_names)
        cache.mark_dirty(fn, node, graph, include_self=not keeps_value)


def nodes_driven_by(bended_module, fn: str, bp) -> list:
    """Bended activations / weights whose value depends on the macro *bp*.

    Read from the module, so it covers bendings set up in Python as well as
    through the editor session.
    """
    nodes = []
    try:
        for act, cbs in bended_module.bended_activations(fn).items():
            if any(_cb_holds(cb, bp) for cb in cbs):
                nodes.append(act)
    except Exception:
        pass
    try:
        for param, cbs in bended_module.bended_weights.items():
            if any(_cb_holds(cb, bp) for cb in cbs):
                nodes.append(param)
    except Exception:
        pass
    return nodes


def _cb_holds(cb, bp) -> bool:
    """Does *cb* read from macro *bp* — directly, or through a derived parameter?

    Linking a normalised macro hands the callback ``macro * span + lo``: a
    different object that shares the macro's value *and* its name. Identity alone
    would miss it, and then nothing would know which nodes the macro drives — the
    activation cache would keep serving the value from before it moved.
    """
    try:
        if bp in cb:           # BendingCallback.__contains__
            return True
    except Exception:
        pass
    try:
        name = bp.name
        return any(getattr(c, "name", None) == name
                   for c in cb._controllables.values())
    except Exception:
        return False


def _read_param(cb, name):
    """Read the current value of a controllable param from a callback."""
    try:
        val = cb.get(name)
        if val is None:
            val = getattr(cb, name, None)
    except Exception:
        val = getattr(cb, name, None)
    if torch.is_tensor(val):
        if val.numel() == 1:
            raw = val.item()
            return int(raw) if val.dtype in _INT_DTYPES else float(raw)
        return val.tolist()
    return val


def _write_param(cb, name, value):
    """Update a controllable param in-place on a callback."""
    try:
        attr = cb.get(name)
        if attr is None:
            attr = getattr(cb, name, None)
        if torch.is_tensor(attr) and attr.numel() == 1:
            if attr.dtype in _INT_DTYPES:
                value = int(round(float(value)))
            attr.fill_(float(value))
            return
    except Exception:
        pass
    try:
        tensor = torch.tensor(float(value))
        cb.register_buffer(name, tensor)
    except Exception:
        setattr(cb, name, value)


# ── BendingSession ─────────────────────────────────────────────────────────────

class BendingSession:
    """Manages the set of active BendingCallback bindings and BendingParameters for one model."""

    def __init__(self):
        self.bindings: dict = {}
        self.bending_params: dict = {}   # name → BendingParameter (float ones live in 0…1)
        # name → (lo, hi): the range a float macro's 0…1 should span when it is
        # linked to a target that declares none of its own
        self.bp_ranges: dict = {}
        self.vis_muted: set = set()
        self.update_mode: str = "auto"
        self.auto_threshold_ms: float = 500.0
        # live per-node display view overrides: {(fn, node): {"view", "options"}}
        self.node_views: dict = {}
        # Lazy activation cache; max_bytes read from Django settings when first used
        self._activation_cache: ActivationCache | None = None
        # Cached bent module and graph per fn — avoids deep-copying parameters on every
        # cache miss. Invalidated when bindings change (add/remove/rebuild).
        self._bent_module_cache: dict = {}
        self._bent_graph_cache: dict = {}

    # ── node view selection ──────────────────────────────────────────────────

    def get_node_view(self, fn: str, node: str):
        """Return the live view selection {'view','options'} for a node, or None."""
        return self.node_views.get((fn, node))

    def set_node_view(self, fn: str, node: str, view: str, options: dict | None = None) -> None:
        self.node_views[(fn, node)] = {"view": view, "options": dict(options or {})}

    def _get_cache(self) -> ActivationCache:
        if self._activation_cache is None:
            try:
                from django.conf import settings as _ds
                max_bytes = getattr(_ds, "ACTIVATION_CACHE_MAX_BYTES", 512 * 1024 * 1024)
            except Exception:
                max_bytes = 512 * 1024 * 1024
            self._activation_cache = ActivationCache(max_bytes=max_bytes)
        return self._activation_cache

    # ── bent module / graph cache ─────────────────────────────────────────────

    def _invalidate_bent_module(self, fn: str = None) -> None:
        """Discard the cached bent module/graph (e.g. after bindings change).

        Compiled slices are built against that module and indexed by node name,
        so they go with it.
        """
        if fn is None:
            self._bent_module_cache.clear()
            self._bent_graph_cache.clear()
        else:
            self._bent_module_cache.pop(fn, None)
            self._bent_graph_cache.pop(fn, None)
        self._get_cache().clear_slices(fn)

    def _get_bent_module_and_graph(self, bended_module, fn: str):
        """Return (bent_module, bent_graph), building and caching them on first call.

        For activation-only bendings the module is reused across requests because
        callback objects are shared by reference — in-place parameter changes
        (slider moves) are already reflected in the cached module.
        The cache is invalidated whenever the binding topology changes.
        """
        if fn not in self._bent_module_cache or fn not in self._bent_graph_cache:
            _t0 = actlog.tick()
            m = bended_module.bend_module(fn=fn)
            g = bended_module.bend_graph(fn=fn)
            self._bent_module_cache[fn] = m
            self._bent_graph_cache[fn] = g
            actlog.log("rebuild  bent module + graph for fn=%s  %s  %s",
                       fn, actlog.fmt_graph(g), actlog.tock(_t0))
        return self._bent_module_cache[fn], self._bent_graph_cache[fn]

    # ── cache helpers ──────────────────────────────────────────────────────────

    def _mark_dirty_nodes(self, bended_module, fn: str, nodes) -> None:
        """Mark nodes and their descendants dirty, releasing cached tensors."""
        # Reuse the session's bent graph: dirty marking only needs the node
        # topology, and rebuilding it here costs milliseconds on every slider
        # move. When it is not cached yet (a binding just changed), build one.
        graph = self._bent_graph_cache.get(fn)
        if graph is None:
            try:
                graph = bended_module.bend_graph(fn=fn)
            except Exception:
                pass
        mark_nodes_dirty(self._get_cache(), graph, fn, nodes)

    def _bended_nodes_for_fn(self, fn: str) -> set:
        nodes = set()
        for b in self.bindings.values():
            if b["fn"] == fn:
                nodes.update(b.get("nodes", [b["node"]]))
        return nodes

    def get_cached_activations(
        self,
        bended_module,
        fn: str,
        kwargs: dict,
        target_nodes=None,
        pinned_nodes=None,
    ):
        """Compute and return activations for *target_nodes* only.

        target_nodes: list of node names to (lazily) compute, or None to
            return only what is already clean in the cache.
        Returns (activations_dict, warning_or_None).
        """
        cache = self._get_cache()
        bended_nodes = self._bended_nodes_for_fn(fn)
        bent_module, bent_graph = self._get_bent_module_and_graph(bended_module, fn)
        return run_activations_with_cache(
            bended_module, fn, kwargs, cache,
            target_nodes=target_nodes,
            bended_nodes=bended_nodes,
            pinned_nodes=set(pinned_nodes or []),
            bent_module=bent_module,
            bent_graph=bent_graph,
        )

    def clear_cache(self, fn: str = None) -> None:
        self._get_cache().clear(fn=fn)

    def cache_stats(self) -> dict:
        return self._get_cache().stats()

    def set_cache_max_bytes(self, max_bytes: int) -> None:
        self._get_cache().max_bytes = max_bytes

    # ── helpers ────────────────────────────────────────────────────────────────

    def _make_cb_kwargs(self, b: dict) -> dict:
        """Build constructor kwargs, substituting BendingParameter objects for linked params."""
        bp_links = b.get("bp_links", {})
        kwargs = {}
        for pname, val in b["params"].items():
            bp_name = bp_links.get(pname)
            if bp_name and bp_name in self.bending_params:
                kwargs[pname] = self.bending_params[bp_name]
            else:
                kwargs[pname] = val
        return kwargs

    # ── introspection ──────────────────────────────────────────────────────────

    def list_bindings(self) -> list:
        result = []
        for bid, b in self.bindings.items():
            cb = b["callback"]
            param_values = {name: _read_param(cb, name)
                            for name in cb.controllable_params}
            b["params"].update(param_values)
            result.append({
                "id":            bid,
                "fn":            b["fn"],
                "node":          b["node"],
                "nodes":         b.get("nodes", [b["node"]]),
                "name":          b.get("name", ""),
                "callback_type": type(cb).__name__,
                "repr":          repr(cb),
                "params":        param_values,
                "descriptor":    type(cb).ui_descriptor(),
                "vis_muted":     bid in self.vis_muted,
                "bp_links":      b.get("bp_links", {}),
                # {param: [lo, hi]} — what a linked macro's 0…1 spans here
                "bp_maps":       b.get("bp_maps", {}),
            })
        return result

    def verbose(self) -> None:
        """Log all active bindings and bending params at INFO level."""
        lines = ["── BendingSession ─────────────────────────────────────────"]
        if not self.bindings and not self.bending_params:
            lines.append("  (no bindings)")
        for bid, b in self.bindings.items():
            cb = b["callback"]
            param_strs = []
            for pname in cb.controllable_params:
                val = _read_param(cb, pname)
                bp_name = b.get("bp_links", {}).get(pname)
                param_strs.append(f"{pname}={val}" + (f" [→{bp_name}]" if bp_name else ""))
            muted = "muted " if bid in self.vis_muted else ""
            nodes = b.get("nodes", [b["node"]])
            node_str = ", ".join(nodes) if len(nodes) > 1 else nodes[0]
            lines.append(f"  [{bid}] {muted}{b['fn']}:{node_str}  ←  {type(cb).__name__}({', '.join(param_strs)})")
        if self.bending_params:
            lines.append("  params:")
            for name, bp in self.bending_params.items():
                lines.append(f"    {name} = {float(bp.get_python_value()):.4g}  [{bp.min_clamp}, {bp.max_clamp}]")
        lines.append("───────────────────────────────────────────────────────")
        logger.info("\n".join(lines))

    # ── BendingParameter management ────────────────────────────────────────────

    def list_bending_params(self) -> list:
        result = []
        for name, bp in self.bending_params.items():
            type_str = param_type_str(bp)
            linked = []
            for bid, b in self.bindings.items():
                for pname, bpname in b.get("bp_links", {}).items():
                    if bpname == name:
                        linked.append({
                            "bid":           bid,
                            "param":         pname,
                            "callback_type": type(b["callback"]).__name__,
                            "node":          b["node"],
                        })
            tr = self.bp_ranges.get(name)
            result.append({
                "name":       name,
                # native python value, so the client sees a bool as a bool
                "value":      param_python_value(bp),
                "param_type": type_str,
                "min_clamp":  bp.min_clamp,
                "max_clamp":  bp.max_clamp,
                "clamp":      bp.clamp,
                "weight":     float(bp.weight),
                "bias":       float(bp.bias),
                # a float macro created here is normalised: it reads 0…1 and each
                # link maps that onto whatever its target wants. One built in
                # Python keeps whatever range it was given.
                "normalized": is_normalized(bp),
                "target_range": list(tr) if tr else None,
                "linked":     linked,
            })
        return result

    def create_bending_param(self, bended_module, name: str, value: float,
                              range_min=None, range_max=None,
                              param_type: str = "float") -> None:
        from torchbend.bending.parameter import BendingParameter
        if name in self.bending_params:
            raise ValueError(f"BendingParameter '{name}' already exists")
        # Cast the initial value to the declared type so BendingParameter infers
        # the correct param_type for TorchScript export.
        _type_map = {"float": float, "int": int, "bool": bool}
        if param_type not in _type_map:
            raise ValueError(
                f"Unknown macro type '{param_type}' (expected one of "
                f"{', '.join(_type_map)})")
        cast = _type_map[param_type]
        try:
            typed_value = cast(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"macro '{name}' is of type {param_type}; cannot take value {value!r}"
            ) from exc
        if param_type == "bool":
            # BendingParameter fixes a bool's range to [0, 1] itself and warns
            # about any range/clamp keyword — don't pass one.
            bp = BendingParameter(name=name, value=typed_value)
        elif param_type == "int":
            # An int macro keeps its integer range: normalising it would make it
            # a float, and a float cannot drive an int param.
            range_min = None if range_min is None else int(round(float(range_min)))
            range_max = None if range_max is None else int(round(float(range_max)))
            bp = BendingParameter(
                name=name,
                value=typed_value,
                range=(range_min, range_max),
                clamp=(range_min is not None and range_max is not None),
            )
        else:
            # A float macro is a *normalised* control: it always lives in 0…1, so
            # every macro slider means the same thing. The range a target
            # actually wants is applied when the two are linked, by parameter
            # arithmetic (``macro * span + lo``) — see :meth:`_mapped_param`.
            target = None
            if range_min is not None and range_max is not None \
                    and float(range_max) != float(range_min):
                target = (float(range_min), float(range_max))
                # the requested value keeps its meaning: it names a point in the
                # target range, which is that same point in 0…1
                typed_value = (typed_value - target[0]) / (target[1] - target[0])
            typed_value = min(1.0, max(0.0, float(typed_value)))
            bp = BendingParameter(name=name, value=typed_value, range=(0., 1.), clamp=True)
            if target is not None:
                self.bp_ranges[name] = target
        self.bending_params[name] = bp
        bended_module._module.register_module(f"_bp_{name}", bp)

    def target_range_for(self, cb, param_name: str, bp_name: str,
                         explicit=None, binding_id: str = None):
        """The ``[lo, hi]`` this *link* maps the macro's 0…1 onto.

        The range belongs to the attachment, not to the macro: one macro can
        drive a scale over ±10 and a mix over 0…1 at the same time. In order of
        authority:

        1. what the caller passed in — the user just chose it;
        2. what this link already had — a re-link must not silently re-map it;
        3. the parameter's own declared range, the natural default;
        4. the range the macro was created with, for a parameter that declares
           none.

        ``None`` when the macro is not a normalised float, or when nothing names
        a range — either way it drives its target directly.
        """
        bp = self.bending_params.get(bp_name)
        if bp is None or not is_normalized(bp):
            return None
        candidates = [explicit]
        if binding_id is not None:
            b = self.bindings.get(binding_id) or {}
            candidates.append((b.get("bp_maps") or {}).get(param_name))
        try:
            pd = ((type(cb).ui_descriptor() or {}).get("params") or {}).get(param_name) or {}
            candidates.append(pd.get("range"))
        except Exception:
            pass
        candidates.append(self.bp_ranges.get(bp_name))
        for cand in candidates:
            if not cand:
                continue
            lo, hi = (list(cand) + [None, None])[:2]
            if lo is None or hi is None or float(lo) == float(hi):
                continue
            return (float(lo), float(hi))
        return None

    @staticmethod
    def _mapped_param(bp, target):
        """``bp`` seen through *target*: a derived parameter reading
        ``value * (hi - lo) + lo`` off the same value the macro holds.

        This is the macro arithmetic doing the range adaptation — the object the
        callback receives moves with the macro but speaks the target's units.
        """
        if target is None:
            return bp
        lo, hi = target
        span = hi - lo
        if span == 0:
            return bp
        return bp * span + lo

    def rename_bending_param(self, bended_module, name: str, new_name: str) -> str:
        """Rename a macro everywhere it is known.

        The name is not a label: it keys this session, the module's controllables
        and every link, and it is what each driven callback's generated forward
        calls its argument. They all move together, and the callbacks are rebuilt
        so their signature follows.
        """
        import re as _re
        if name not in self.bending_params:
            raise KeyError(f"BendingParameter '{name}' not found")
        new_name = (new_name or "").strip()
        if not new_name or new_name == name:
            return name
        if not _re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", new_name):
            raise ValueError(
                f"'{new_name}' cannot be a macro name — it becomes an argument of "
                f"the generated forward, so it has to be a plain identifier: "
                f"letters, digits and underscores, not starting with a digit.")
        if new_name in self.bending_params:
            raise ValueError(f"A macro named '{new_name}' already exists")

        bp = self.bending_params.pop(name)
        bp._rename(new_name)
        self.bending_params[new_name] = bp
        if name in self.bp_ranges:
            self.bp_ranges[new_name] = self.bp_ranges.pop(name)

        # the module keeps macros both as submodules and as controllables, by name
        try:
            mods = bended_module._module._modules
            if f"_bp_{name}" in mods:
                mods[f"_bp_{new_name}"] = mods.pop(f"_bp_{name}")
        except Exception:
            pass
        for attr in ("_controllables", "_controllable_hash"):
            try:
                store = getattr(bended_module, attr)
                if name in store:
                    store[new_name] = store.pop(name)
            except Exception:
                pass

        touched = []
        for b in self.bindings.values():
            links = b.get("bp_links") or {}
            for pname, bpname in list(links.items()):
                if bpname == name:
                    links[pname] = new_name
                    touched.append(b["callback"])
        for cb in touched:
            if getattr(cb, "_initialized", False):
                cb._init_forward_callback()
                cb._init_static_controllable_callback(cb.controllable_params)
        return new_name

    def set_macro_range(self, bended_module, name: str, lo, hi) -> None:
        """Set the range a normalised macro's 0…1 maps onto, and re-apply it.

        The macro itself does not move — it stays where it is in 0…1 — but every
        link it drives is rebuilt against the new range, so the targets follow.
        """
        if name not in self.bending_params:
            raise KeyError(f"BendingParameter '{name}' not found")
        rng = None
        if lo is not None and hi is not None and float(lo) != float(hi):
            rng = (float(lo), float(hi))
        if rng is None:
            self.bp_ranges.pop(name, None)
        else:
            self.bp_ranges[name] = rng
        # every link this macro drives takes the new range: editing it from the
        # macro is the "all of them" gesture, editing one link is the other
        for bid, b in list(self.bindings.items()):
            for pname, bpname in list((b.get("bp_links") or {}).items()):
                if bpname == name:
                    self.link_param(bid, pname, name, bended_module=bended_module,
                                    range=rng, _replace_range=True)

    def set_link_range(self, bended_module, binding_id: str, param_name: str,
                       lo, hi) -> None:
        """Re-map one link: the macro stays put, its target follows a new range.

        Rebuilding the link is what applies it — a fresh derived parameter with
        different arithmetic replaces the one the callback was holding.
        """
        b = self.bindings.get(binding_id)
        if b is None:
            raise KeyError(f"Binding '{binding_id}' not found")
        bp_name = (b.get("bp_links") or {}).get(param_name)
        if not bp_name:
            raise KeyError(f"Param '{param_name}' is not driven by a macro")
        rng = None if (lo is None or hi is None) else (float(lo), float(hi))
        self.link_param(binding_id, param_name, bp_name,
                        bended_module=bended_module, range=rng, _replace_range=True)

    def link_param(self, binding_id: str, param_name: str, bp_name: str,
                   bended_module=None, range=None, _replace_range: bool = False) -> None:
        if bp_name not in self.bending_params:
            raise KeyError(f"BendingParameter '{bp_name}' not found")
        b = self.bindings.get(binding_id)
        if b is None:
            raise KeyError(f"Binding '{binding_id}' not found")
        if param_name not in b["params"]:
            raise KeyError(f"Param '{param_name}' not in binding '{binding_id}'")

        cb = b["callback"]
        bp = self.bending_params[bp_name]

        # Type check before touching anything: the callback asserts on a mismatch
        # deep inside register_controllable, and an assertion is both unhelpful to
        # show and stripped under python -O.
        allowed = param_types_for(cb, param_name)
        bp_type = param_type_str(bp)
        if allowed and bp_type not in allowed:
            raise ValueError(
                f"Macro '{bp_name}' is of type {bp_type}, but param '{param_name}' "
                f"of {type(cb).__name__} expects {' or '.join(allowed)}")

        # Link directly on the existing callback instance (no rebuild needed).
        # What the callback gets is the macro seen through the target's range;
        # it shares the macro's value, so moving the macro moves it.
        # One macro cannot drive two params of the *same* callback: the generated
        # forward names its arguments after the macro, so a second one collides
        # ("duplicate argument" from the generated source, or — worse — both
        # params silently reading the same value). Refuse it here, where the
        # reason can be explained, rather than let codegen fail.
        clash = [p for p, n in (b.get("bp_links") or {}).items()
                 if n == bp_name and p != param_name]
        if clash:
            raise ValueError(
                f"Macro '{bp_name}' already drives '{clash[0]}' on this "
                f"{type(cb).__name__}. A callback cannot take the same macro "
                f"twice — give '{param_name}' a macro of its own, or bend the "
                f"node with a second callback.")

        if _replace_range:
            # an explicit re-map: this link's previous range must not outvote it
            b.get("bp_maps", {}).pop(param_name, None)
        target = self.target_range_for(cb, param_name, bp_name,
                                       explicit=range, binding_id=binding_id)
        # whatever this link was reading through, it no longer does
        prev = b.setdefault("bp_objs", {}).pop(param_name, None)
        if prev is not None and prev is not bp:
            bp._release_derived(prev)

        linked = self._mapped_param(bp, target)
        cb.register_controllable(param_name, linked)
        b["bp_objs"][param_name] = linked

        if "bp_links" not in b:
            b["bp_links"] = {}
        b["bp_links"][param_name] = bp_name
        b.setdefault("bp_maps", {})
        if target is None:
            b["bp_maps"].pop(param_name, None)
        else:
            b["bp_maps"][param_name] = [target[0], target[1]]
        b["params"][param_name] = float(linked.get_python_value())

        # Reconcile the link into the BendedModule so the parameter shows up in
        # bended_module.controllables() — needed for scripting (set_<name>
        # accessors) and play mode, which read controllables from the module.
        if bended_module is not None:
            try:
                bended_module._register_controllables(cb)
                # …but the name must resolve to the macro itself, not to the
                # derived parameter the callback holds: reading it back is how
                # the UI shows the macro's own 0…1 position.
                bended_module._controllables[bp_name] = bp
            except Exception:
                pass

        # For weight callbacks, re-apply immediately
        if cb.weight_compatible and len(cb._bending_targets) > 0:
            cb.apply(update=False)

    def unlink_param(self, binding_id: str, param_name: str) -> None:
        b = self.bindings.get(binding_id)
        if b is None:
            raise KeyError(f"Binding '{binding_id}' not found")
        bp_links = b.get("bp_links", {})
        if param_name not in bp_links:
            return  # already plain, no-op

        cb = b["callback"]
        bp_name = bp_links.pop(param_name)
        b.get("bp_maps", {}).pop(param_name, None)
        prev = b.get("bp_objs", {}).pop(param_name, None)
        parent = self.bending_params.get(bp_name)
        if prev is not None and parent is not None and prev is not parent:
            parent._release_derived(prev)
        current_val = b["params"].get(param_name, 0.0)

        # Remove from _controllables and re-register as a plain buffer, keeping
        # the param's declared type (a bool/int param must not become a float one)
        if param_name in cb._controllables:
            del cb._controllables[param_name]
        # register_controllable also set the parameter as a plain attribute on the
        # callback; nn.Module refuses to register a buffer over a name that is
        # still taken, so clear it before putting the frozen value back
        for store in (cb.__dict__, getattr(cb, "_parameters", {}), getattr(cb, "_modules", {})):
            store.pop(param_name, None)
        types = param_types_for(cb, param_name)
        if "bool" in types:
            frozen = torch.tensor(bool(current_val))
        elif "int" in types:
            frozen = torch.tensor(int(round(float(current_val))))
        else:
            frozen = torch.tensor(float(current_val))
        cb.register_buffer(param_name, frozen)
        # register_controllable regenerates the callback's forward when a macro
        # is attached; taking one away has to do the same. The generated forward
        # carries one argument per controllable, named after it — leaving a stale
        # one behind makes the next graph build fail looking for a macro that no
        # longer exists (KeyError on the name with its last segment stripped).
        if getattr(cb, "_initialized", False):
            cb._init_forward_callback()
            cb._init_static_controllable_callback(cb.controllable_params)

        if cb.weight_compatible and len(cb._bending_targets) > 0:
            cb.apply(update=False)

    def update_bending_param(self, name: str, value, bended_module=None) -> None:
        bp = self.bending_params.get(name)
        if bp is None:
            raise KeyError(f"BendingParameter '{name}' not found")
        bp.set_value(coerce_param_value(bp, value))
        # set_value already triggers cb.update() for each linked callback;
        # for weight callbacks we also need apply()
        for b in self.bindings.values():
            for pname, bpname in b.get("bp_links", {}).items():
                if bpname == name:
                    cb = b["callback"]
                    is_weight_cb = cb.weight_compatible and len(cb._bending_targets) > 0
                    if is_weight_cb:
                        cb.apply(update=False)
                    b["params"][pname] = bp.get_python_value()
                    if bended_module is not None:
                        self._mark_dirty_nodes(bended_module, b["fn"], b.get("nodes", [b["node"]]))
                    if is_weight_cb:
                        # the bent module's parameter copy is now stale
                        self._invalidate_bent_module(b["fn"])

    def delete_bending_param(self, bended_module, name: str) -> None:
        if name not in self.bending_params:
            raise KeyError(f"BendingParameter '{name}' not found")
        # Unlink from every binding that references it
        for bid in list(self.bindings.keys()):
            bp_links = self.bindings[bid].get("bp_links", {})
            for pname in [k for k, v in list(bp_links.items()) if v == name]:
                self.unlink_param(bid, pname)
        del self.bending_params[name]
        self.bp_ranges.pop(name, None)
        # Remove the submodule from the nn.Module
        try:
            mod_key = f"_bp_{name}"
            if mod_key in bended_module._module._modules:
                del bended_module._module._modules[mod_key]
        except Exception:
            pass

    # ── binding mutation ───────────────────────────────────────────────────────

    def add_binding(self, bended_module, fn: str, node: str,
                    callback_type_name: str, params: dict,
                    existing_id: str | None = None) -> str:
        if existing_id and existing_id in self.bindings:
            b = self.bindings[existing_id]
            nodes_list = b.get("nodes", [b["node"]])
            if node not in nodes_list:
                is_weight = any(
                    n.op == "get_attr"
                    for n in bended_module.graph(fn=fn).nodes
                    if n.name == node
                )
                try:
                    bended_module._bend(b["callback"], node, fn=fn,
                                        bend_param=is_weight, bend_graph=not is_weight)
                except Exception as exc:
                    raise ValueError(f"Could not add '{node}' to existing binding: {exc}") from exc
                b["nodes"] = nodes_list + [node]
                # the graph gained a callback node: drop the stale bent module
                # first, so dirty marking sees the binding that was just added
                self._invalidate_bent_module(fn)
                self._mark_dirty_nodes(bended_module, fn, [node])
            return existing_id

        cb_class = _find_callback_class(callback_type_name)
        descriptor = cb_class.ui_descriptor()

        cb_kwargs: dict = {}
        stored_params: dict = {}
        for name, param_desc in descriptor["params"].items():
            val = params.get(name, param_desc["default"])
            if val is None:
                val = param_desc["default"]
            if val is None:
                val = 0
            val = _coerce(val, param_desc.get("type", "float"))
            cb_kwargs[name] = val
            stored_params[name] = val

        for name, spec in descriptor.get("extra_init_params", {}).items():
            raw = params.get(name, spec.get("default"))
            if raw is None and spec.get("required"):
                raise ValueError(
                    f"Missing required init param '{name}' for {callback_type_name}"
                )
            if raw is not None:
                cb_kwargs[name] = _coerce(raw, spec.get("type") or "")

        callback = cb_class(**cb_kwargs)
        try:
            bended_module.bend(callback, node, fn=fn)
        except Exception as exc:
            raise ValueError(
                f"Could not bend '{node}' with {callback_type_name}: {exc}"
            ) from exc

        bid = str(uuid.uuid4())[:8]
        self.bindings[bid] = {
            "fn":       fn,
            "node":     node,
            "nodes":    [node],
            "callback": callback,
            "params":   stored_params,
            "bp_links": {},
        }
        logger.debug("[add_binding] %s:%s  ←  %s(%s)  id=%s",
                     fn, node, callback_type_name,
                     ", ".join(f"{k}={v}" for k, v in stored_params.items()), bid)
        # invalidate first: dirty marking reads the graph to decide whether the
        # bended node keeps its own value, and must see the new callback node
        self._invalidate_bent_module(fn)
        self._mark_dirty_nodes(bended_module, fn, [node])
        return bid

    def update_param(self, bended_module, binding_id: str,
                     param_name: str, value) -> None:
        b = self.bindings.get(binding_id)
        if b is None:
            raise KeyError(f"Binding '{binding_id}' not found")
        if param_name in b.get("bp_links", {}):
            # Forward change directly to the linked BendingParameter
            self.update_bending_param(b["bp_links"][param_name], value, bended_module=bended_module)
            return
        cb = b["callback"]
        param_desc = (type(cb).ui_descriptor().get("params", {}).get(param_name) or {})
        type_str = param_desc.get("type", "float")
        value = _coerce(value, type_str)

        param_ui = (getattr(type(cb), "_param_ui", None) or {}).get(param_name, {})
        factory = param_ui.get("factory")
        if callable(factory):
            value = factory(value)
            value = _coerce(value, type_str)

        choices = param_desc.get("choices")
        if choices:
            value = min(choices, key=lambda c: abs(c - value))

        guard = param_ui.get("guard")
        if callable(guard):
            try:
                n_params = sum(
                    1 for p in inspect.signature(guard).parameters.values()
                    if p.default is inspect.Parameter.empty
                )
                result = guard(value, cb) if n_params >= 2 else guard(value)
            except Exception as exc:
                raise ValueError(str(exc)) from exc
            if result is not True:
                msg = str(result) if isinstance(result, Exception) else f"Invalid value for '{param_name}': {value}"
                raise ValueError(msg)

        _write_param(cb, param_name, value)
        b["params"][param_name] = value
        logger.debug("[update_param] binding=%s  %s=%s", binding_id, param_name, value)
        cb.update()
        is_weight_cb = cb.weight_compatible and len(cb._bending_targets) > 0
        if is_weight_cb:
            cb.apply(update=False)
        self._mark_dirty_nodes(bended_module, b["fn"], b.get("nodes", [b["node"]]))
        if is_weight_cb:
            # re-applying a weight callback rewrites the parameters, but the bent
            # module holds its own copy made when it was built — drop it, unlike
            # activation callbacks which are shared by reference and stay live
            self._invalidate_bent_module(b["fn"])

    def _rebuild_bindings(self, bended_module) -> None:
        """Reset module bending and re-apply all non-muted bindings."""
        if logger.isEnabledFor(logging.DEBUG):
            active = [
                f"{b['fn']}:{b['node']}({type(b['callback']).__name__})"
                for bid, b in self.bindings.items() if bid not in self.vis_muted
            ]
            logger.debug("[_rebuild_bindings] %d active: %s", len(active), ", ".join(active) or "none")
        self._invalidate_bent_module()
        bended_module.reset_bending()
        for bid, b in self.bindings.items():
            cb_class = type(b["callback"])
            fresh_cb = b["callback"]
            try:
                fresh_cb = cb_class(**self._make_cb_kwargs(b))
                b["callback"] = fresh_cb
            except Exception:
                pass
            if bid not in self.vis_muted:
                try:
                    bended_module.bend(fresh_cb, b["node"], fn=b["fn"])
                except Exception:
                    pass

    def set_vis_muted(self, bended_module, binding_id: str, muted: bool) -> None:
        if binding_id not in self.bindings:
            raise KeyError(f"Binding '{binding_id}' not found")
        b = self.bindings[binding_id]
        if muted:
            self.vis_muted.add(binding_id)
        else:
            self.vis_muted.discard(binding_id)
        self._mark_dirty_nodes(bended_module, b["fn"], b.get("nodes", [b["node"]]))
        self._rebuild_bindings(bended_module)

    def get_original_activations(self, bended_module, fn: str, kwargs: dict, capture_fn):
        bended_module.reset_bending()
        try:
            return capture_fn(bended_module, fn, kwargs)
        finally:
            self._rebuild_bindings(bended_module)

    def reorder_bindings(self, bended_module, order: list) -> None:
        """Re-apply bindings in the given order (list of binding IDs)."""
        new_bindings = {bid: self.bindings[bid] for bid in order if bid in self.bindings}
        for bid, b in self.bindings.items():
            if bid not in new_bindings:
                new_bindings[bid] = b
        self.bindings = new_bindings
        # Reordering changes callback chain — invalidate all bended nodes
        for b in self.bindings.values():
            self._mark_dirty_nodes(bended_module, b["fn"], b.get("nodes", [b["node"]]))
        self._rebuild_bindings(bended_module)

    def remove_binding(self, bended_module, binding_id: str) -> None:
        if binding_id not in self.bindings:
            raise KeyError(f"Binding '{binding_id}' not found")
        b = self.bindings.pop(binding_id)
        logger.debug("[remove_binding] %s:%s  id=%s", b["fn"], b["node"], binding_id)
        self.vis_muted.discard(binding_id)
        self._mark_dirty_nodes(bended_module, b["fn"], b.get("nodes", [b["node"]]))
        self._rebuild_bindings(bended_module)

    # ── config save / load ─────────────────────────────────────────────────────

    def session_to_json(self) -> dict:
        """Serialise current session state to a JSON-safe dict."""
        bindings = []
        for bid, b in self.bindings.items():
            cb = b["callback"]
            bindings.append({
                "fn":            b["fn"],
                "node":          b["node"],
                "callback_type": type(cb).__name__,
                "params":        dict(b["params"]),
                "bp_links":      dict(b.get("bp_links", {})),
                "bp_maps":       dict(b.get("bp_maps", {})),
                "vis_muted":     bid in self.vis_muted,
            })
        bps = []
        for name, bp in self.bending_params.items():
            bps.append({
                "name":       name,
                # keep the declared type and the native value: reloading a macro
                # as a float would break every link to an int / bool param
                "param_type": param_type_str(bp),
                "value":      param_python_value(bp),
                "min_clamp":  bp.min_clamp,
                "max_clamp":  bp.max_clamp,
                "clamp":      bp.clamp,
                # a normalised macro's clamp is always 0…1, so the range it maps
                # onto has to be saved in its own right
                "normalized": is_normalized(bp),
                "target_range": list(self.bp_ranges[name]) if name in self.bp_ranges else None,
            })
        return {"bindings": bindings, "bending_params": bps}

    def session_from_json(self, bended_module, data: dict) -> None:
        """Restore session from a dict produced by :meth:`session_to_json`."""
        bended_module.reset_bending()
        self.bindings.clear()
        self.bending_params.clear()
        self.bp_ranges.clear()
        self.vis_muted.clear()
        self._get_cache().clear()
        self._invalidate_bent_module()
        # Re-create BendingParameters first so bindings can link to them
        for bp_data in data.get("bending_params", []):
            try:
                ptype = bp_data.get("param_type") or "float"
                target = bp_data.get("target_range")
                if ptype == "float":
                    # The saved value is already the macro's own 0…1 reading, so
                    # pass no range here (which would re-map it) and restore the
                    # target separately. A session written before macros were
                    # normalised saves a raw value with its real range: let that
                    # one go through the normalising path instead.
                    normalized = bp_data.get("normalized")
                    if normalized is None:
                        normalized = (bp_data.get("min_clamp") == 0
                                      and bp_data.get("max_clamp") == 1)
                    if normalized:
                        self.create_bending_param(
                            bended_module, bp_data["name"],
                            bp_data.get("value", 0.0), None, None, ptype)
                        if target:
                            self.bp_ranges[bp_data["name"]] = (float(target[0]),
                                                               float(target[1]))
                        continue
                self.create_bending_param(
                    bended_module,
                    bp_data["name"],
                    bp_data.get("value", 0.0),
                    bp_data.get("min_clamp"),
                    bp_data.get("max_clamp"),
                    ptype,
                )
            except Exception as exc:
                logger.warning("could not restore macro %r: %s",
                               bp_data.get("name"), exc)
        # Re-create bindings
        for b_data in data.get("bindings", []):
            try:
                bid = self.add_binding(
                    bended_module,
                    b_data["fn"],
                    b_data["node"],
                    b_data["callback_type"],
                    b_data.get("params", {}),
                )
                if b_data.get("vis_muted"):
                    self.vis_muted.add(bid)
                saved_maps = b_data.get("bp_maps") or {}
                for param_name, bp_name in b_data.get("bp_links", {}).items():
                    if bp_name in self.bending_params:
                        try:
                            # a link's range is its own — restore the saved one
                            # rather than re-deriving a default for it
                            self.link_param(bid, param_name, bp_name,
                                            bended_module=bended_module,
                                            range=saved_maps.get(param_name))
                        except Exception:
                            pass
            except Exception:
                pass

    def session_from_tbconfig(self, bended_module, data: bytes) -> None:
        """Restore session from dill-serialised BendingConfig bytes."""
        import dill
        config = dill.loads(data)
        bended_module.reset_bending()
        self.bindings.clear()
        self.bending_params.clear()
        self.vis_muted.clear()
        self._get_cache().clear()
        self._invalidate_bent_module()
        for item in config._bendings:
            if not item:
                continue
            cb, *nodes = item
            if not nodes:
                continue
            fn = "forward"
            first_node = nodes[0] if isinstance(nodes[0], str) else str(nodes[0])
            try:
                bended_module.bend(cb, first_node, fn=fn)
                bid = str(uuid.uuid4())[:8]
                params = {name: _read_param(cb, name) for name in cb.controllable_params}
                self.bindings[bid] = {
                    "fn": fn, "node": first_node, "nodes": [first_node],
                    "callback": cb, "params": params, "bp_links": {},
                }
                for node in nodes[1:]:
                    node = node if isinstance(node, str) else str(node)
                    try:
                        bended_module._bend(cb, node, fn=fn, bend_param=False, bend_graph=True)
                        self.bindings[bid]["nodes"].append(node)
                    except Exception:
                        pass
            except Exception:
                pass

    def export_bending_config(self, bended_module) -> bytes:
        """Serialise the current session as a BendingConfig and return dill bytes."""
        import dill
        import io
        from torchbend.bending.config import BendingConfig
        # Build (callback, node) tuples for BendingConfig
        tuples = []
        for b in self.bindings.values():
            cb = b["callback"]
            node = b["node"]
            tuples.append((cb, node))
        config = BendingConfig(*tuples) if tuples else BendingConfig()
        buf = io.BytesIO()
        dill.dump(config, buf)
        return buf.getvalue()

    # ── export ─────────────────────────────────────────────────────────────────

    def export_torchscript(self, bended_module) -> bytes:
        scripted = bended_module.script()
        buf = io.BytesIO()
        torch.jit.save(scripted, buf)
        return buf.getvalue()
