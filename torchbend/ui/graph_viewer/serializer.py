import torch.fx


def _collect_arg_nodes(arg):
    if isinstance(arg, torch.fx.Node):
        yield arg
    elif isinstance(arg, (list, tuple)):
        for item in arg:
            yield from _collect_arg_nodes(item)


def _serialize_target(target):
    if isinstance(target, str):
        return target
    if hasattr(target, "__qualname__"):
        return target.__qualname__
    if hasattr(target, "__name__"):
        return target.__name__
    return str(target)


def _is_mark_tensor(node):
    try:
        return node.target._name in ("torchbend::mark_tensor", "torchbend::mark_tensor_pre")
    except AttributeError:
        return False


def _get_arg_label(src_node, target_node):
    """Return the positional index or kwarg name of src_node in target_node's call."""
    for i, arg in enumerate(target_node.args):
        if arg is src_node:
            return str(i)
        if isinstance(arg, (list, tuple)):
            for a in arg:
                if a is src_node:
                    return str(i)
    for k, v in target_node.kwargs.items():
        if v is src_node:
            return k
        if isinstance(v, (list, tuple)):
            for a in v:
                if a is src_node:
                    return k
    return ""


def _serialize_args(args):
    result = []
    for arg in args:
        if isinstance(arg, torch.fx.Node):
            result.append({"type": "node", "name": arg.name})
        elif isinstance(arg, (list, tuple)):
            result.append({
                "type": "list",
                "items": [
                    {"type": "node", "name": a.name} if isinstance(a, torch.fx.Node)
                    else {"type": "value", "value": repr(a)}
                    for a in arg
                ],
            })
        else:
            result.append({"type": "value", "value": repr(arg)})
    return result


def _build_module_groups(graph, activations=None):
    """Return (compound_nodes, node_parent_map).

    Priority order:
    1. module_path field on ActivationProperties (populated during tracing via stack inspection)
    2. call_module node targets (standard torch.fx high-level trace)
    3. get_attr parameter name inference (ATen-level fallback)
    """
    seen = {}
    node_parent = {}

    # ── 1. module_path from tracing ────────────────────────────────────────────
    if activations:
        node_module = {}
        for node in graph.nodes:
            act = activations.get(node.name)
            path = getattr(act, 'module_path', None) if act else None
            if path:
                node_module[node.name] = path
        if node_module:
            return _groups_from_node_module(node_module)

    # ── 2. call_module nodes ───────────────────────────────────────────────────
    for node in graph.nodes:
        if node.op != "call_module":
            continue
        target = str(node.target)
        parts = target.split(".")
        for depth, _ in enumerate(parts):
            path = ".".join(parts[: depth + 1])
            if path not in seen:
                parent_path = ".".join(parts[:depth]) if depth > 0 else None
                seen[path] = _make_compound(path, parts[depth], parent_path)
        node_parent[node.name] = f"__mod__{target}"

    if seen:
        return list(seen.values()), node_parent

    # ── 3. get_attr parameter-name inference (ATen-level fallback) ─────────────
    node_module = {}
    for node in graph.nodes:
        if node.op != "get_attr":
            continue
        parts = str(node.target).split(".")
        if len(parts) >= 2:
            node_module[node.name] = ".".join(parts[:-1])

    if not node_module:
        return [], {}

    # propagate assignments to direct consumers from a single module
    changed = True
    while changed:
        changed = False
        for node in graph.nodes:
            if node.op not in ("call_function", "call_method") or node.name in node_module:
                continue
            hits = {node_module[a.name] for a in node.args
                    if isinstance(a, torch.fx.Node) and a.name in node_module}
            if len(hits) == 1:
                node_module[node.name] = hits.pop()
                changed = True

    return _groups_from_node_module(node_module)


def _groups_from_node_module(node_module):
    seen = {}
    all_paths = set(node_module.values())
    for path in list(all_paths):
        parts = path.split(".")
        for depth in range(len(parts)):
            all_paths.add(".".join(parts[:depth + 1]))
    for path in sorted(all_paths):
        parts = path.split(".")
        parent_path = ".".join(parts[:-1]) if len(parts) > 1 else None
        seen[path] = _make_compound(path, parts[-1], parent_path)
    node_parent = {name: f"__mod__{path}" for name, path in node_module.items()}
    return list(seen.values()), node_parent


def _make_compound(path, label, parent_path):
    return {
        "id": f"__mod__{path}",
        "label": label,
        "op": "module",
        "module_path": path,
        "is_compound": True,
        "parent": f"__mod__{parent_path}" if parent_path else None,
        "shape": None,
        "has_bending": False,
        "bending_callbacks": [],
        "args": [],
    }


def _reachable_from_output(graph):
    """Return the set of node names that are ancestors of any output node."""
    output_nodes = [n for n in graph.nodes if n.op == "output"]
    visited = set()
    queue = list(output_nodes)
    while queue:
        node = queue.pop()
        if node.name in visited:
            continue
        visited.add(node.name)
        for arg in _collect_arg_nodes(node.args):
            if arg.name not in visited:
                queue.append(arg)
    return visited


# Targets (call_method name or call_function qualname) that don't compute anything
# learned — they only reshape, cast, or copy data.
_TRIVIAL_TARGETS = frozenset({
    # call_method names
    "detach", "detach_", "contiguous", "clone", "copy_",
    "view", "reshape", "flatten", "unflatten",
    "squeeze", "unsqueeze", "expand", "expand_as", "broadcast_to",
    "permute", "transpose", "t", "movedim", "moveaxis",
    "to", "float", "half", "double", "long", "int", "short", "bool", "byte",
    "type", "type_as", "cuda", "cpu", "pin_memory", "requires_grad_",
    # call_function qualnames
    "getitem", "getattr",
    "flatten", "reshape", "squeeze", "unsqueeze", "permute", "transpose",
    "Tensor.detach", "Tensor.view", "Tensor.reshape", "Tensor.flatten",
    "Tensor.squeeze", "Tensor.unsqueeze", "Tensor.expand", "Tensor.permute",
    "Tensor.transpose", "Tensor.contiguous", "Tensor.clone", "Tensor.to",
    "Tensor.float", "Tensor.half", "Tensor.long", "Tensor.int",
    "Tensor.type_as", "Tensor.cuda", "Tensor.cpu",
})


def _is_trivial_node(node) -> bool:
    """Return True if the node is a no-op (reshape/cast/copy) with no learned parameters."""
    if node.op in ("placeholder", "output", "get_attr"):
        return True
    target = _serialize_target(node.target) if node.target is not None else ""
    # Strip common module prefixes (torch., operator., etc.) for matching
    base = target.rsplit(".", 1)[-1] if "." in target else target
    return base in _TRIVIAL_TARGETS or target in _TRIVIAL_TARGETS


def serialize_graph(bended_module, fn="forward", prune_unreachable=True):
    # use the original traced graph (bended=True), not the bent graph which adds
    # internal callback nodes and rewrites ops — those aren't useful for display
    graph = bended_module.graph(fn=fn, bended=True)

    try:
        activations = bended_module.activations("?.*", fn=fn)
    except Exception:
        activations = {}

    bended_acts = {}
    try:
        bended_acts = bended_module._bended_activations.get(fn, {})
    except Exception:
        pass

    reachable = _reachable_from_output(graph) if prune_unreachable else None

    # Build passthrough map for mark_tensor nodes (pure identity ops to hide)
    skip_nodes = set()
    passthrough = {}   # hidden node name → real upstream node name
    for node in graph.nodes:
        if _is_mark_tensor(node):
            skip_nodes.add(node.name)
            if node.args and isinstance(node.args[0], torch.fx.Node):
                passthrough[node.name] = node.args[0].name

    def _resolve(name, depth=0):
        if depth > 20 or name not in passthrough:
            return name
        return _resolve(passthrough[name], depth + 1)

    compound_nodes, node_parent = _build_module_groups(graph, activations=activations)
    nodes = list(compound_nodes)   # compound nodes first so Cytoscape resolves parents
    edges = []
    seen_edges = set()

    for node in graph.nodes:
        if node.name in skip_nodes:
            continue
        if reachable is not None and node.name not in reachable:
            continue
        act = activations.get(node.name)
        shape = None
        if act is not None and hasattr(act, "shape") and act.shape is not None:
            try:
                shape = [int(s) for s in act.shape]
            except Exception:
                shape = None

        # For get_attr (weight) nodes, fall back to reading the tensor shape directly
        if node.op == "get_attr" and shape is None:
            try:
                obj = bended_module._module
                for attr in str(node.target).split("."):
                    obj = getattr(obj, attr)
                if hasattr(obj, "shape"):
                    shape = [int(s) for s in obj.shape]
            except Exception:
                pass

        # source location from tracer code frame
        source_file = source_line = source_fn = None
        if act is not None and getattr(act, "code", None) is not None:
            source_file = getattr(act.code, "source_file", None)
            source_line = getattr(act.code, "source_line", None)
            source_fn   = getattr(act.code, "source_fn", None)

        bending_cbs = bended_acts.get(node.name, [])

        # Degree / shape-change metadata -----------------------------------------
        parent_nodes = []
        for pn in _collect_arg_nodes(node.args):
            real = _resolve(pn.name)
            if real in skip_nodes:
                continue
            if reachable is not None and real not in reachable:
                continue
            parent_nodes.append(real)
        in_degree = len(parent_nodes)

        out_degree = 0
        for user in node.users:
            if user.name in skip_nodes:
                continue
            if reachable is not None and user.name not in reachable:
                continue
            out_degree += 1

        # shape_changed: True when our shape differs from every parent's shape
        # (if no parents have a shape, we leave it False to avoid false positives)
        shape_changed = False
        if shape is not None and parent_nodes:
            parent_shapes_with_data = []
            for pname in parent_nodes:
                pact = activations.get(pname)
                if pact is not None and hasattr(pact, "shape") and pact.shape is not None:
                    try:
                        parent_shapes_with_data.append(tuple(int(s) for s in pact.shape))
                    except Exception:
                        pass
            if parent_shapes_with_data:
                our_shape = tuple(shape)
                shape_changed = all(our_shape != ps for ps in parent_shapes_with_data)

        nodes.append({
            "id": node.name,
            "label": node.name,
            "op": node.op,
            "target": _serialize_target(node.target) if node.target is not None else None,
            "shape": shape,
            "has_shape": shape is not None,
            "has_bending": len(bending_cbs) > 0,
            "bending_callbacks": [str(cb) for cb in bending_cbs],
            "args": _serialize_args(node.args),
            "parent": node_parent.get(node.name),
            "source_file": source_file,
            "source_line": source_line,
            "source_fn":   source_fn,
            "is_trivial": _is_trivial_node(node),
            "shape_changed": shape_changed,
            "in_degree": in_degree,
            "out_degree": out_degree,
        })

        for src in _collect_arg_nodes(node.args):
            label = _get_arg_label(src, node)
            real_src = _resolve(src.name)
            if real_src in skip_nodes:
                continue
            edge_key = (real_src, node.name)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append({
                    "id": f"{real_src}__{node.name}",
                    "source": real_src,
                    "target": node.name,
                    "label": label,
                    "target_op": node.op,
                    "target_fn": _serialize_target(node.target) if node.target is not None else "",
                })

    try:
        module_type = type(bended_module._module).__name__
    except Exception:
        module_type = type(bended_module).__name__

    aliases = {}
    try:
        raw = bended_module.aliases(fn)
        for name, node_names in raw.items():
            aliases[name] = list(node_names)
    except Exception:
        pass

    return {"nodes": nodes, "edges": edges, "fn": fn, "module_type": module_type, "aliases": aliases}


def get_available_methods(bended_module):
    try:
        return list(bended_module.traced_methods)
    except Exception:
        return []


# Per-module-type dimension labels for weight tensors.
# Index matches the actual tensor dimension order.
_WEIGHT_DIM_LABELS = {
    "Linear":           ["filter", "in"],
    "Bilinear":         ["filter", "in1", "in2"],
    "Conv1d":           ["filter", "in_ch", "kW"],
    "Conv2d":           ["filter", "in_ch", "kH", "kW"],
    "Conv3d":           ["filter", "in_ch", "kD", "kH", "kW"],
    "ConvTranspose1d":  ["in_ch", "filter", "kW"],
    "ConvTranspose2d":  ["in_ch", "filter", "kH", "kW"],
    "ConvTranspose3d":  ["in_ch", "filter", "kD", "kH", "kW"],
    "Embedding":        ["token", "dim"],
    "EmbeddingBag":     ["token", "dim"],
    "MultiheadAttention": ["out", "in"],
}


def serialize_tensor_for_viz(tensor, as_image=False, module_type=None, param_name=None):
    import torch
    import torch.nn.functional as F

    # Dimension labels for weight tensors, keyed by module type.
    # Truncated/extended to match the actual ndim at return time.
    dim_labels = list(_WEIGHT_DIM_LABELS.get(module_type, [])) if module_type else None

    def _dl(n=None):
        """Return dim_labels trimmed/padded to n entries, or None."""
        if not dim_labels:
            return None
        if n is None:
            return dim_labels
        labels = dim_labels[:n]
        while len(labels) < n:
            labels.append(f"d{len(labels)}")
        return labels

    if not isinstance(tensor, torch.Tensor):
        try:
            tensor = torch.tensor(tensor)
        except Exception:
            return {"error": "not a tensor", "shape": [], "data": None, "ndim": 0}

    tensor = tensor.detach().float().cpu()
    shape = [int(s) for s in tensor.shape]
    ndim = len(shape)

    def norm(t):
        mn, mx = float(t.min()), float(t.max())
        rng = mx - mn
        return (t - mn) / rng if rng > 1e-8 else t - mn

    if ndim == 0:
        return {"shape": [], "data": float(tensor.item()), "ndim": 0, "kind": "scalar",
                "is_audio_compatible": False}

    if ndim == 1:
        t = tensor
        n = shape[0]
        if n > 1024:
            t = F.interpolate(t.view(1, 1, -1), size=1024, mode="linear", align_corners=False).view(-1)
        r = {"shape": shape, "data": t.tolist(), "ndim": 1, "kind": "line",
             "is_audio_compatible": n > 1024}
        if _dl(1): r["dim_labels"] = _dl(1)
        return r

    if ndim == 2:
        B, T = shape
        # treat as batch-of-sequences when T >> B and T looks like a sequence length
        if T > B and T > 32:
            n_b = min(B, 64)
            t = tensor[:n_b]
            if T > 1024:
                t = F.interpolate(t.unsqueeze(0), size=1024, mode="linear",
                                  align_corners=False).squeeze(0)
            r = {"shape": shape, "data": t.tolist(), "ndim": 2, "kind": "lineset",
                 "n_batches": B, "is_audio_compatible": T > 1024}
            if _dl(2): r["dim_labels"] = _dl(2)
            return r
        t = tensor
        if shape[0] > 256 or shape[1] > 256:
            t = F.interpolate(t.unsqueeze(0).unsqueeze(0),
                              size=(min(shape[0], 256), min(shape[1], 256)),
                              mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        r = {"shape": shape, "data": norm(t).tolist(), "ndim": 2, "kind": "heatmap",
             "is_audio_compatible": False}
        if _dl(2): r["dim_labels"] = _dl(2)
        return r

    if ndim == 3:
        c, h, w = shape
        # Temporal: [batch, channels, time] where time >> channels
        if w > h and w > 32:
            n_b, n_c, T = c, h, w
            n_b_show = min(n_b, 4)
            n_c_show = min(n_c, 64)
            T_show = min(T, 1024)
            t = tensor[:n_b_show, :n_c_show]
            if T > T_show:
                t = F.interpolate(
                    t.reshape(n_b_show * n_c_show, 1, T),
                    size=T_show, mode="linear", align_corners=False,
                ).reshape(n_b_show, n_c_show, T_show)
            r = {"shape": shape, "data": t.tolist(), "ndim": 3,
                 "kind": "temporal_3d", "n_batches": n_b, "n_channels": n_c,
                 "is_audio_compatible": T > 1024}
            if _dl(3): r["dim_labels"] = _dl(3)
            return r
        if c == 1:
            # Single-channel: heatmap
            t = norm(tensor.squeeze(0))
            if h > 256 or w > 256:
                t = F.interpolate(t.unsqueeze(0).unsqueeze(0),
                                  size=(min(h, 256), min(w, 256)),
                                  mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
            r = {"shape": shape, "data": t.tolist(), "ndim": 3, "kind": "heatmap",
                 "is_audio_compatible": False}
            if _dl(3): r["dim_labels"] = _dl(3)
            return r
        if c == 3 and h > 4 and w > 4 and not dim_labels:
            # RGB image (skip for weight tensors that happen to have 3 in dim 0)
            t = tensor[:3]
            if h > 256 or w > 256:
                t = F.interpolate(t.unsqueeze(0), size=(min(h, 256), min(w, 256)),
                                  mode="bilinear", align_corners=False).squeeze(0)
            return {"shape": shape, "data": norm(t).tolist(), "ndim": 3, "kind": "rgb",
                    "is_audio_compatible": False}
        # Multi-channel grid or 1-D filters — show up to 16 channels as heatmaps
        n_show = min(c, 16)
        t = tensor[:n_show]
        if h > 64 or w > 64:
            t = F.interpolate(t.unsqueeze(0), size=(min(h, 64), min(w, 64)),
                              mode="bilinear", align_corners=False).squeeze(0)
        r = {"shape": shape, "data": norm(t).tolist(), "ndim": 3, "kind": "grid",
             "shown": n_show, "is_audio_compatible": False}
        if _dl(3): r["dim_labels"] = _dl(3)
        return r

    if ndim == 4:
        n, c, h, w = shape
        if c in (1, 3, 4) and not dim_labels:
            # Only treat as image batch when not a known weight tensor
            n_show = min(n, 16)
            t = tensor[:n_show]
            if h > 256 or w > 256:
                t = F.interpolate(t, size=(min(h, 256), min(w, 256)),
                                  mode='bilinear', align_corners=False)
            mn, mx = float(t.min()), float(t.max())
            rng = mx - mn
            t = (t - mn) / rng if rng > 1e-8 else torch.zeros_like(t)
            image_type = {1: "gray", 3: "rgb", 4: "rgba"}[c]
            return {
                "shape": shape, "ndim": 4, "kind": "image_batch",
                "image_type": image_type,
                "n_batches": n, "shown_batches": n_show,
                "data": t.tolist(),
                "is_audio_compatible": False,
            }
        n_show = min(n, 8)
        c_show = min(c, 64)
        t = tensor[:n_show, :c_show]
        if h > 64 or w > 64:
            flat = t.reshape(n_show * c_show, 1, h, w)
            flat = F.interpolate(flat, size=(min(h, 64), min(w, 64)),
                                 mode="bilinear", align_corners=False)
            t = flat.reshape(n_show, c_show, min(h, 64), min(w, 64))
        result_data = []
        for bi in range(n_show):
            batch_slices = []
            for ci in range(c_show):
                sl = t[bi, ci]
                mn2, mx2 = float(sl.min()), float(sl.max())
                rng2 = mx2 - mn2
                sl = (sl - mn2) / rng2 if rng2 > 1e-8 else sl - mn2
                batch_slices.append(sl.tolist())
            result_data.append(batch_slices)
        r = {"shape": shape, "data": result_data, "ndim": 4, "kind": "grid_4d",
             "n_batches": n, "n_channels": c,
             "shown_batches": n_show, "shown_channels": c_show,
             "is_audio_compatible": False}
        if _dl(4): r["dim_labels"] = _dl(4)
        return r

    return {"shape": shape, "data": None, "ndim": ndim, "kind": "unsupported",
            "is_audio_compatible": False}
