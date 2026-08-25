"""Lazy, demand-driven activation cache for the graph viewer.

Architecture
------------
Activations are never computed speculatively.  The cache is a dict of dicts:

    _entries[(fn, input_id)][node_name] = state

where *state* is one of:
    None       — never computed for this input
    _DIRTY     — was computed, but an ancestor was bended; tensor was released
    Tensor     — clean, valid

Computation is triggered only when the caller explicitly requests a node
(``target_nodes``).  For each missing/dirty target the smallest possible
subgraph is run:

    1. Walk backward from the dirty/missing node in the *bended* graph.
    2. Find the nearest **clean** ancestor (a cached tensor).
    3. Call ``bended_module.from_activations`` with that ancestor as input
       (or ``get_activations`` starting from the original inputs if no clean
       ancestor exists), trimmed to only output the requested node(s).
    4. Store the result; the slot becomes clean.

When a node is bended
---------------------
``mark_dirty(fn, node, graph)`` is called.  It sets every clean tensor that is
a descendant of *node* (in the bended graph) to ``_DIRTY``, releasing the
tensor memory.  On the next request for any of those nodes, step 1–4 above
fires.

Memory budget
-------------
``max_bytes`` caps clean tensor memory.  Before storing we evict clean tensors
with this priority:
    1. Clean tensors from a different (fn, input_id) than the current call
    2. Clean tensors of nodes that are not pinned
    3. Any remaining clean tensor (last resort)
Dirty / None slots cost no memory and are never evicted for space.
If the required bytes exceed ``max_bytes`` even after a full eviction the call
returns ``(result, error_str)`` so the caller can alert the user.
"""

import hashlib
import logging
import torch
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

from torchbend.tracing import activation_log as actlog


_DEFAULT_MAX_BYTES = 512 * 1024 * 1024  # 512 MB

# Compiled slices are cheap (they share the bent module's parameters by
# identity — only the fx-generated code is new), so we can keep a good few.
_DEFAULT_MAX_SLICES = 64

# Sentinel: slot was computed but is now stale (tensor already released).
_DIRTY = object()


# ── input hashing ──────────────────────────────────────────────────────────────

def _hash_inputs(kwargs: dict) -> str:
    h = hashlib.md5()
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        h.update(k.encode())
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu()
            h.update(str(t.shape).encode())
            h.update(str(t.dtype).encode())
            flat = t.contiguous().view(-1)
            n = min(flat.numel(), 4096)
            h.update(flat[:n].numpy().tobytes())
        else:
            h.update(str(v).encode())
    return h.hexdigest()[:16]


# ── graph topology helpers ────────────────────────────────────────────────────

def _get_descendants(graph, node_name: str) -> Set[str]:
    """All node names reachable from node_name via users edges."""
    node_map = {n.name: n for n in graph.nodes}
    root = node_map.get(node_name)
    if root is None:
        return set()
    visited: Set[str] = set()
    queue = list(root.users)
    while queue:
        n = queue.pop()
        if n.name in visited:
            continue
        visited.add(n.name)
        queue.extend(n.users)
    return visited


def _find_clean_frontier(
    graph,
    target_names: List[str],
    cache_entry: dict,
) -> Optional[List[str]]:
    """Walk backward from target_names and return the nearest clean ancestors.

    A node is a valid frontier candidate only when its cache slot holds an
    actual ``torch.Tensor`` (clean).  Dirty and None slots are skipped.
    Returns None when no clean ancestor exists (must run from original inputs).

    The walk goes through ``all_input_nodes``, not ``args``: an operand can be
    a keyword argument, or sit inside a list/tuple argument (``torch.cat([a,
    b])``). Iterating ``args`` and keeping only bare Nodes misses both, and the
    walk then dead-ends on such a node and reports no frontier at all — making
    every request recompute from the inputs.
    """
    node_map = {n.name: n for n in graph.nodes}
    target_set = set(target_names)
    frontier: Set[str] = set()

    visited: Set[str] = set()
    queue = list(target_names)

    while queue:
        name = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)

        node = node_map.get(name)
        if node is None:
            continue

        for arg in node.all_input_nodes:
            if arg.name in target_set:
                if arg.name not in visited:
                    queue.append(arg.name)
                continue
            if isinstance(cache_entry.get(arg.name), torch.Tensor):
                # Clean cached ancestor — use as frontier
                frontier.add(arg.name)
            elif arg.op == "placeholder":
                pass  # original input; subgraph will keep this placeholder
            else:
                # Neither clean nor placeholder — keep walking back
                if arg.name not in visited:
                    queue.append(arg.name)

    return list(frontier) if frontier else None


# ── slice compilation ─────────────────────────────────────────────────────────

def _compiled_slice(cache, fn: str, module, graph,
                    roots: List[str], targets: List[str], callbacks=None):
    """Return the fx module computing *targets* out of *roots*, reusing it if seen.

    This is the same slicing ``BendedModule.call`` performs — ``graph_subset``
    when the run is re-rooted, ``graph_get_activations`` when it starts from the
    original inputs — but against the session's already-built bent module and
    graph, and memoised, so a repeated request pays neither the rewrite nor the
    fx codegen again.
    """
    from torchbend.tracing.graph import graph_get_activations, graph_subset
    from torchbend.tracing.graphmodule import BendedGraphModule

    key = (fn, tuple(sorted(roots)), tuple(targets))
    gm = cache.get_slice(key)
    if gm is not None:
        actlog.log("slice    reused  roots=%s → targets=%s  (no rewrite, no codegen)",
                   actlog.fmt_names(sorted(roots)), actlog.fmt_names(targets))
        return gm

    _t0 = actlog.tick()
    if roots:
        sub_graph = graph_subset(graph, list(roots), list(targets),
                                 remove_placeholders=True,
                                 parse_inputs_from_callbacks=callbacks)
    else:
        sub_graph = graph_get_activations(graph, list(targets))
    gm = BendedGraphModule(module, **{fn: sub_graph})  # type: ignore[arg-type]
    cache.store_slice(key, gm)
    actlog.log("slice    compiled  %s  %s", actlog.fmt_graph(sub_graph), actlog.tock(_t0))
    return gm


# ── subgraph execution from a clean frontier ──────────────────────────────────

def _compute_from_frontier(
    bended_module,
    fn: str,
    target_names: List[str],
    frontier_names: List[str],
    frontier_tensors: Dict[str, torch.Tensor],
    original_kwargs: dict,
    bent_module=None,
    bent_graph=None,
    cache=None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Run the minimal subgraph from frontier_names to target_names.

    Slices the bended graph in a single ``graph_subset`` pass — the frontier
    nodes become input placeholders, so nothing upstream of them is computed,
    and nothing outside the frontier→target span is built at all. The compiled
    slice is memoised, so re-requesting it costs only the forward.
    Returns None on any failure so the caller can fall back to a full run.
    """
    with actlog.step("cache.from_frontier",
                     "targets=%s  frontier=%s" % (actlog.fmt_names(target_names),
                                                  actlog.fmt_names(frontier_names))) as step:
        try:
            actlog.log_tensors(frontier_tensors, prefix="reuse  ")

            bended_graph = bent_graph if bent_graph is not None else bended_module.bend_graph(fn=fn)
            module = bent_module if bent_module is not None else bended_module.bend_module(fn=fn)

            gm = _compiled_slice(cache, fn, module, bended_graph,
                                 list(frontier_names), list(target_names))
            fn_method = getattr(gm, fn)

            all_inputs = dict(original_kwargs)
            all_inputs.update(frontier_tensors)
            _t0 = actlog.tick()
            inputs_obj = bended_module.inputs_for_fn(fn_method, all_inputs)
            outs = fn_method(*inputs_obj, **inputs_obj)
            actlog.log("run FROM FRONTIER %s  %s",
                       actlog.fmt_names(sorted(frontier_names)), actlog.tock(_t0))

            if isinstance(outs, torch.Tensor):
                result = {target_names[0]: outs}
            elif isinstance(outs, (tuple, list)):
                result = {target_names[i]: outs[i] for i in range(min(len(target_names), len(outs)))}
            else:
                result = {}

            result = {k: v.detach() for k, v in result.items() if isinstance(v, torch.Tensor)}
            actlog.log_tensors(result, prefix="→ ")
            step.result("computed %d node(s) without recomputing upstream", len(result))
            return result
        except Exception as e:
            actlog.log("FAILED: %s: %s — falling back to a full run",
                       type(e).__name__, e, level=logging.WARNING)
            step.result("failed")
            warnings.warn(f"[ActivationCache] from-frontier execution failed ({e}), falling back to full run")
            return None


# ── ActivationCache ────────────────────────────────────────────────────────────

class ActivationCache:
    """Demand-driven, dirty-flag activation cache.

    Slots start as None.  When requested, a node is computed via the minimal
    subgraph and stored as a clean Tensor.  When its ancestor is bended the
    slot is set to _DIRTY (tensor released) and recomputed lazily on next
    request.
    """

    def __init__(self, max_bytes: int = _DEFAULT_MAX_BYTES,
                 max_slices: int = _DEFAULT_MAX_SLICES):
        self.max_bytes = max_bytes
        self._entries: Dict[Tuple[str, str], Dict[str, object]] = {}
        self._pinned: Set[str] = set()
        self._bended_nodes: Set[str] = set()
        # Compiled roots→targets slices, keyed by (fn, roots, targets). Moving a
        # slider re-requests the very same slice, so keeping the fx-compiled
        # module spares both the graph rewrite and the codegen every time.
        self.max_slices = max_slices
        self._slices: "OrderedDict[tuple, object]" = OrderedDict()
        self._slice_hits = 0
        self._slice_misses = 0
        # Activation-name index per fn — the "?.*" scan behind it is not free.
        self._act_names: Dict[str, List[str]] = {}

    # ── compiled slice reuse ───────────────────────────────────────────────────

    def get_slice(self, key: tuple):
        """Return the compiled slice for *key*, or None (LRU-touched on hit)."""
        gm = self._slices.get(key)
        if gm is None:
            self._slice_misses += 1
            return None
        self._slices.move_to_end(key)
        self._slice_hits += 1
        return gm

    def store_slice(self, key: tuple, gm) -> None:
        self._slices[key] = gm
        self._slices.move_to_end(key)
        while len(self._slices) > self.max_slices:
            self._slices.popitem(last=False)

    def clear_slices(self, fn: Optional[str] = None) -> None:
        """Drop compiled slices and the name index — the graph itself changed.

        Must be called whenever the bent module/graph is rebuilt: a compiled
        slice holds a reference to the module it was built against. Parameter
        changes (slider moves) do *not* need this — callbacks are shared by
        reference, so the compiled slice already sees them.
        """
        n = len(self._slices)
        if fn is None:
            self._slices.clear()
            self._act_names.clear()
        else:
            for k in [k for k in self._slices if k[0] == fn]:
                del self._slices[k]
            self._act_names.pop(fn, None)
        if n:
            actlog.log("slices   dropped %d compiled slice(s) (graph changed)", n)

    def activation_names(self, bended_module, fn: str, bended_graph) -> List[str]:
        """Names of every activation of *fn*, minus the output node (memoised).

        Recomputed only when :meth:`clear_slices` invalidates it, since the
        underlying ``activations("?.*")`` scan costs a few ms on large graphs
        and would otherwise run on every single request.
        """
        names = self._act_names.get(fn)
        if names is None:
            output_names = {n.name for n in bended_graph.nodes if n.op == "output"}
            names = [
                name
                for name in bended_module.activations("?.*", fn=fn, with_bended=True).keys()
                if name not in output_names
            ]
            self._act_names[fn] = names
        return names

    # ── slot state queries ─────────────────────────────────────────────────────

    def is_clean(self, fn: str, input_id: str, node: str) -> bool:
        """True only when the slot holds a valid Tensor."""
        return isinstance(self._entries.get((fn, input_id), {}).get(node), torch.Tensor)

    def is_dirty(self, fn: str, input_id: str, node: str) -> bool:
        return self._entries.get((fn, input_id), {}).get(node) is _DIRTY

    def get_clean(self, fn: str, input_id: str, node: str) -> Optional[torch.Tensor]:
        t = self._entries.get((fn, input_id), {}).get(node)
        return t if isinstance(t, torch.Tensor) else None

    # ── memory accounting ──────────────────────────────────────────────────────

    def used_bytes(self) -> int:
        total = 0
        for entry in self._entries.values():
            for t in entry.values():
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
        return total

    def stats(self) -> dict:
        total_slots = sum(len(e) for e in self._entries.values())
        clean = sum(1 for e in self._entries.values()
                    for t in e.values() if isinstance(t, torch.Tensor))
        dirty = sum(1 for e in self._entries.values()
                    for t in e.values() if t is _DIRTY)
        return {
            "max_bytes":    self.max_bytes,
            "max_mb":       round(self.max_bytes / 1024 / 1024, 1),
            "used_bytes":   self.used_bytes(),
            "used_mb":      round(self.used_bytes() / 1024 / 1024, 1),
            "entries":      len(self._entries),
            "total_slots":  total_slots,
            "clean_slots":  clean,
            "dirty_slots":  dirty,
            "none_slots":   total_slots - clean - dirty,
            "pinned":       len(self._pinned),
            "slices":       len(self._slices),
            "max_slices":   self.max_slices,
            "slice_hits":   self._slice_hits,
            "slice_misses": self._slice_misses,
        }

    # ── slot management ────────────────────────────────────────────────────────

    def init_slots(self, fn: str, input_id: str, node_names: List[str]) -> None:
        """Ensure all node_names have a slot; never overwrites existing state."""
        key = (fn, input_id)
        if key not in self._entries:
            self._entries[key] = {name: None for name in node_names}
        else:
            for name in node_names:
                if name not in self._entries[key]:
                    self._entries[key][name] = None

    def store(
        self,
        fn: str,
        input_id: str,
        tensors: Dict[str, torch.Tensor],
    ) -> Tuple[bool, Optional[str]]:
        """Store a batch of tensors, evicting clean entries as needed.

        Returns (True, None) on success or (False, error_message) if even a
        full eviction cannot make enough room.
        """
        needed = sum(t.numel() * t.element_size() for t in tensors.values())
        if needed > self.max_bytes:
            actlog.log("store    REFUSED %s — %s needed, limit is %s. Nothing is cached, "
                       "so every later request recomputes from the inputs.",
                       actlog.fmt_names(list(tensors)), actlog.fmt_bytes(needed),
                       actlog.fmt_bytes(self.max_bytes), level=logging.WARNING)
            return False, (
                f"{', '.join(tensors)} needs {actlog.fmt_bytes(needed)} but the activation "
                f"cache limit is {actlog.fmt_bytes(self.max_bytes)}, so it was not cached — "
                f"every later request will recompute it from the inputs. Raise it with "
                f"graph_viewer.run(..., cache_size=\"16GB\")."
            )

        evicted = 0
        while self.used_bytes() + needed > self.max_bytes:
            if not self._evict_one(current_fn=fn, current_input_id=input_id):
                actlog.log("store    FAILED — cache full at %s, %d evicted, needed %s",
                           actlog.fmt_bytes(self.used_bytes()), evicted,
                           actlog.fmt_bytes(needed), level=logging.WARNING)
                return False, (
                    f"Activation cache full ({actlog.fmt_bytes(self.used_bytes())} used of "
                    f"{actlog.fmt_bytes(self.max_bytes)}) — could not free enough room for "
                    f"{actlog.fmt_bytes(needed)}. Raise it with "
                    f"graph_viewer.run(..., cache_size=\"16GB\")."
                )
            evicted += 1

        key = (fn, input_id)
        if key not in self._entries:
            self._entries[key] = {}
        for name, t in tensors.items():
            self._entries[key][name] = t
        actlog.log("store    %s (+%s) → %s / %s  (%d evicted)",
                   actlog.fmt_names(list(tensors)), actlog.fmt_bytes(needed),
                   actlog.fmt_bytes(self.used_bytes()), actlog.fmt_bytes(self.max_bytes),
                   evicted)
        return True, None

    # ── dirty-flag invalidation ───────────────────────────────────────────────

    def mark_dirty(self, fn: str, node: str, graph=None, include_self: bool = True) -> None:
        """Mark node and all its descendants as dirty, releasing their tensors.

        Slots that are already None stay None (no-op).
        Only clean Tensor slots are downgraded to _DIRTY.

        ``include_self=False`` keeps *node*'s own cached value: bending an
        activation through an inserted ``<node>_bended`` callback does not
        change the node itself, only what reads it. Keeping it clean lets the
        next request resume right at the bend instead of recomputing the whole
        prefix — which is the difference between a slider being usable or not.
        """
        affected = {node} if include_self else set()
        if graph is not None:
            affected |= _get_descendants(graph, node)

        released, freed = [], 0
        for (efn, _), entry in self._entries.items():
            if efn == fn:
                for name in affected:
                    slot = entry.get(name)
                    if isinstance(slot, torch.Tensor):
                        freed += slot.numel() * slot.element_size()
                        released.append(name)
                        entry[name] = _DIRTY  # release tensor
        actlog.log("dirty    fn=%s  root=%s%s  → %d node(s), released %s from %s",
                   fn, node, "" if include_self else " (value kept)",
                   len(affected), actlog.fmt_bytes(freed),
                   actlog.fmt_names(sorted(set(released))), level=logging.INFO)

    def clear(self, fn: Optional[str] = None) -> None:
        if fn is None:
            self._entries.clear()
        else:
            for k in [k for k in self._entries if k[0] == fn]:
                del self._entries[k]

    # ── pins & bended metadata ────────────────────────────────────────────────

    def set_pinned(self, nodes: Optional[Set[str]]) -> None:
        self._pinned = set(nodes) if nodes else set()

    def set_bended_nodes(self, nodes: Optional[Set[str]]) -> None:
        self._bended_nodes = set(nodes) if nodes else set()

    # ── eviction (clean tensors only) ─────────────────────────────────────────

    def _evict_one(self, current_fn: Optional[str] = None, current_input_id: Optional[str] = None) -> bool:
        """Evict one clean tensor.  Dirty / None slots cost no memory — skip them."""
        current_key = (current_fn, current_input_id)

        # Priority 1: clean tensors from a different input
        for key, entry in self._entries.items():
            if key != current_key:
                for node, t in entry.items():
                    if isinstance(t, torch.Tensor):
                        entry[node] = None
                        return True

        # Priority 2: clean tensors that are not pinned
        for key, entry in self._entries.items():
            for node, t in entry.items():
                if isinstance(t, torch.Tensor) and node not in self._pinned:
                    entry[node] = None
                    return True

        # Priority 3: any remaining clean tensor (pinned, last resort)
        for key, entry in self._entries.items():
            for node, t in entry.items():
                if isinstance(t, torch.Tensor):
                    entry[node] = None
                    return True

        return False  # nothing left to free


# ── high-level fetch ──────────────────────────────────────────────────────────

def run_activations_with_cache(
    bended_module,
    fn: str,
    kwargs: dict,
    cache: ActivationCache,
    target_nodes: Optional[List[str]] = None,
    bended_nodes: Optional[Set[str]] = None,
    pinned_nodes: Optional[Set[str]] = None,
    bent_module=None,
    bent_graph=None,
) -> Tuple[Dict[str, torch.Tensor], Optional[str]]:
    """Lazily compute and return activations for *target_nodes* only.

    Parameters
    ----------
    target_nodes:
        Explicit list of node names to compute.  Each missing/dirty node is
        computed via the minimal subgraph (from its nearest clean ancestor).
        If None, no new computation is triggered — only already-clean entries
        are returned.
    """
    input_id = _hash_inputs(kwargs)
    cache.set_bended_nodes(bended_nodes or set())
    cache.set_pinned(pinned_nodes or set())

    with actlog.step("cache.request",
                     "fn=%s  targets=%s  input=%s"
                     % (fn, actlog.fmt_names(target_nodes or []) if target_nodes else "(cached only)",
                        input_id), level=logging.INFO) as step:
        actlog.log_tensors({k: v for k, v in kwargs.items() if torch.is_tensor(v)}, prefix="input  ")
        if bended_nodes:
            actlog.log("bended   %s", actlog.fmt_names(sorted(bended_nodes)))

        # Use pre-built graph if provided; otherwise build it now.
        bended_graph = bent_graph if bent_graph is not None else bended_module.bend_graph(fn=fn)

        all_act_names = cache.activation_names(bended_module, fn, bended_graph)

        # Initialise slots for any node we don't know about yet.
        cache.init_slots(fn, input_id, all_act_names)

        warn_msg = None
        freshly_computed: Dict[str, torch.Tensor] = {}

        if target_nodes:
            # Only compute nodes that are requested and not yet clean.
            to_compute = [n for n in target_nodes
                          if n in all_act_names and not cache.is_clean(fn, input_id, n)]

            if actlog.enabled():
                hits = [n for n in target_nodes if cache.is_clean(fn, input_id, n)]
                unknown = [n for n in target_nodes if n not in all_act_names]
                stats = cache.stats()
                actlog.log("slots    hit=%s  recompute=%s%s  (clean=%d dirty=%d none=%d, %s / %s)",
                           actlog.fmt_names(hits), actlog.fmt_names(to_compute),
                           "  unknown=%s" % actlog.fmt_names(unknown) if unknown else "",
                           stats["clean_slots"], stats["dirty_slots"], stats["none_slots"],
                           actlog.fmt_bytes(stats["used_bytes"]), actlog.fmt_bytes(stats["max_bytes"]))

            if to_compute:
                cache_entry = cache._entries.get((fn, input_id), {})
                frontier = _find_clean_frontier(bended_graph, to_compute, cache_entry)

                computed: Optional[Dict[str, torch.Tensor]] = None
                if frontier:
                    actlog.log("frontier %s  → resume from nearest clean ancestor(s)",
                               actlog.fmt_names(sorted(frontier)))
                    frontier_tensors = {
                        n: t for n in frontier
                        if (t := cache.get_clean(fn, input_id, n)) is not None
                    }
                    computed = _compute_from_frontier(
                        bended_module, fn, to_compute, frontier, frontier_tensors, kwargs,
                        bent_module=bent_module, bent_graph=bended_graph, cache=cache,
                    )
                elif actlog.enabled():
                    # Say why, so a cache that never hits is diagnosable: the
                    # usual causes are an input that changes between requests
                    # (different input id => different entry) or ancestors that
                    # were computed but then invalidated.
                    entry = cache._entries.get((fn, input_id), {})
                    n_clean = sum(1 for v in entry.values() if isinstance(v, torch.Tensor))
                    n_dirty = sum(1 for v in entry.values() if v is _DIRTY)
                    others = [iid for (efn, iid) in cache._entries if efn == fn and iid != input_id]
                    actlog.log("frontier none  → full run from the original inputs")
                    actlog.log("         entry %s holds %d clean / %d dirty tensor(s)%s",
                               input_id, n_clean, n_dirty,
                               ";  %d other input id(s) cached: %s — the inputs are changing "
                               "between requests, so nothing can be reused"
                               % (len(others), actlog.fmt_names(others)) if others and not n_clean else "")

                if not computed:
                    # No usable frontier — run from original inputs.
                    # Use pre-built module/graph when available to avoid redundant deep-copy.
                    m = bent_module if bent_module is not None else bended_module.bend_module(fn=fn)
                    with actlog.step("cache.full_run",
                                     "targets=%s" % actlog.fmt_names(to_compute)):
                        try:
                            gm = _compiled_slice(cache, fn, m, bended_graph, [], to_compute)
                            fn_method = getattr(gm, fn)
                            _t0 = actlog.tick()
                            inputs_obj = bended_module.inputs_for_fn(fn_method, kwargs)
                            outs = fn_method(*inputs_obj, **inputs_obj)
                            actlog.log("run FROM INPUTS (whole prefix recomputed)  %s",
                                       actlog.tock(_t0))
                            if isinstance(outs, torch.Tensor):
                                computed = {to_compute[0]: outs.detach()}
                            elif isinstance(outs, (tuple, list)):
                                computed = {to_compute[i]: outs[i].detach()
                                            for i in range(min(len(to_compute), len(outs)))
                                            if isinstance(outs[i], torch.Tensor)}
                            else:
                                computed = {}
                        except Exception as e:
                            actlog.log("FAILED: %s: %s — retrying via BendedModule.get_activations",
                                       type(e).__name__, e, level=logging.WARNING)
                            warnings.warn(f"[ActivationCache] BendedGraphModule run failed ({e}), falling back to get_activations")
                            try:
                                raw = bended_module.get_activations(*to_compute, fn=fn, **kwargs)
                            except Exception:
                                # The retry is the same computation by another
                                # route. If it fails too the fault is the model's,
                                # not the compiled slice's — and the retry's own
                                # error (wrapped several layers deep) would only
                                # bury the one the user can act on.
                                raise e
                            computed = {k: v.detach() for k, v in raw.items() if isinstance(v, torch.Tensor)}
                        actlog.log_tensors(computed, prefix="→ ")

                ok, err = cache.store(fn, input_id, computed)
                if not ok:
                    warn_msg = err
                    freshly_computed = computed  # return even if we couldn't cache
            elif actlog.enabled():
                unknown = [n for n in target_nodes if n not in all_act_names]
                actlog.log("nothing to compute — %s",
                           "no requested node is a known activation" if unknown and len(unknown) == len(target_nodes)
                           else "every target is already clean")

        # Build return dict: target nodes (or all known) from cache + freshly computed.
        nodes_to_return = target_nodes if target_nodes is not None else all_act_names
        result = {}
        for name in nodes_to_return:
            t = cache.get_clean(fn, input_id, name)
            if t is None:
                t = freshly_computed.get(name)
            if t is not None:
                result[name] = t
        step.result("returned %d activation(s), %s",
                    len(result), actlog.fmt_bytes(actlog.tensor_bytes(result)))
        return result, warn_msg
