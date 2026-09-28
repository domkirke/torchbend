from dataclasses import dataclass, replace

import torch.fx

from . import node_kinds


def _collect_arg_nodes(arg):
    """Yield every fx.Node nested anywhere inside an argument.

    Operands hide in dicts as well as lists: a module returning
    ``{'logits': tensor}`` puts its only real operand inside a dict, and a
    traversal that skips dicts sees an output node with no inputs at all.
    Prefer ``node.all_input_nodes`` when you want a node's predecessors --
    it covers kwargs too. This helper is for walking one specific argument.
    """
    if isinstance(arg, torch.fx.Node):
        yield arg
    elif isinstance(arg, dict):
        for item in arg.values():
            yield from _collect_arg_nodes(item)
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


def _contains_node(container, src_node):
    """True if src_node is the container or is nested anywhere inside it."""
    return any(n is src_node for n in _collect_arg_nodes(container))


def _get_arg_label(src_node, target_node):
    """Return the positional index or kwarg name of src_node in target_node's call.

    A dict argument is labelled by its key rather than its position: for an
    output node returning ``{'logits': ...}``, "logits" says more than "0".
    """
    for i, arg in enumerate(target_node.args):
        if isinstance(arg, dict):
            for k, v in arg.items():
                if _contains_node(v, src_node):
                    return str(k)
        if _contains_node(arg, src_node):
            return str(i)
    for k, v in target_node.kwargs.items():
        if _contains_node(v, src_node):
            return k
    return ""


def _serialize_args(args):
    result = []
    for arg in args:
        if isinstance(arg, torch.fx.Node):
            result.append({"type": "node", "name": arg.name})
        elif isinstance(arg, dict):
            result.append({
                "type": "dict",
                "items": [
                    {"key": str(k),
                     **({"type": "node", "name": v.name} if isinstance(v, torch.fx.Node)
                        else {"type": "value", "value": repr(v)})}
                    for k, v in arg.items()
                ],
            })
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
        # A weight has no activation, so the stack-inspection pass above never
        # sees one -- but a `get_attr`'s target says exactly which module it
        # belongs to. Without this every weight in the model is unfoldable, and
        # depth mode draws all of them however far you fold: 541 of them on
        # Bark's pipeline, which is most of what was on the canvas.
        for node in graph.nodes:
            if node.op != "get_attr" or node.name in node_module:
                continue
            target = str(node.target)
            if "." in target:
                node_module[node.name] = target.rsplit(".", 1)[0]
        if node_module:
            return _groups_from_node_module(_drop_boundary_ops(graph, node_module))

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
        return list(seen.values()), _drop_boundary_ops(graph, node_parent)

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

    return _groups_from_node_module(_drop_boundary_ops(graph, node_module))


#: The method's signature, not part of any submodule.  A tracer that attributes
#: nodes by walking the call stack sees whatever ``forward`` happened to be
#: running when the proxy was made, which for an interface that traces from
#: inside a module puts the model's own inputs in a submodule -- and from there
#: they get folded away with it.
_TOP_LEVEL_OPS = ("placeholder", "output")


def _drop_boundary_ops(graph, node_module):
    """Strip module attributions from placeholder and output nodes."""
    for node in graph.nodes:
        if node.op in _TOP_LEVEL_OPS:
            node_module.pop(node.name, None)
    return node_module


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


def _dtype_of(node):
    """The node's dtype as a bare string ('int64'), or None if unknown.

    Without it the bench suggests ``torch.randn`` for everything, which is a
    wrong answer for an integer input like ``input_ids`` — it looks plausible
    and fails on the embedding lookup.
    """
    for key in ("val", "tensor_meta"):
        try:
            meta = node.meta.get(key)
        except AttributeError:
            return None
        dtype = getattr(meta, "dtype", None)
        if dtype is not None:
            return str(dtype).replace("torch.", "")
    return None


def _module_path_of(parent_id):
    """``'__mod__transformer.h.0'`` -> ``'transformer.h.0'``."""
    if not parent_id:
        return None
    return parent_id[len("__mod__"):] if parent_id.startswith("__mod__") else parent_id


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
        # all_input_nodes, not args: operands also arrive as kwargs and nested
        # inside dicts/lists, and missing one prunes everything behind it.
        for arg in node.all_input_nodes:
            if arg.name not in visited:
                queue.append(arg)
    return visited


# ── display simplification ────────────────────────────────────────────────────
#
# None of this touches the traced graph.  It decides what the *viewer* draws:
# an ATen trace spends most of its nodes reading symbolic sizes, destructuring
# tuples and relabelling views, and drawing all of it buries the model.  Nodes
# folded away here keep their names, their activations and their bendability --
# they are reachable through the details panel and through search.


@dataclass(frozen=True)
class DisplayOptions:
    """What the viewer folds away.  Purely presentational."""
    prune_unreachable: bool = True
    hide_shape_calc:   bool = True
    merge_unpack:      bool = True
    collapse_layout:   bool = True
    expanded:          frozenset = frozenset()   # chain ids the user opened
    #: Depth mode: show the graph as modules rather than as every operation.
    #: `scope` is the module the view is inside ("" = the whole model), and
    #: `module_depth` is how many levels below it stay separate before their
    #: contents are folded into one node per module.
    scope:             str = ""
    module_depth:      int = 0                   # 0 = off, show every node
    #: Pick `module_depth` from the graph instead: the most detail that still
    #: fits in `max_nodes`. A small graph then stays fully expanded and a large
    #: one opens as modules, without anyone choosing a number.
    auto_depth:        bool = False
    max_nodes:         int = 60


#: The unsimplified view.  ``sync`` fingerprints the serialized graph, so it has
#: to hash the trace itself and not whatever the display defaults happen to be
#: this week -- otherwise changing a default invalidates every saved session.
RAW = DisplayOptions(hide_shape_calc=False, merge_unpack=False, collapse_layout=False)

#: Everything the trace contains, folded nowhere and pruned nothing: what the
#: activation search looks through. The drawn graph is a subset of this — a node
#: inside a collapsed module is still a node someone can go looking for.
FULL = DisplayOptions(prune_unreachable=False, hide_shape_calc=False,
                      merge_unpack=False, collapse_layout=False,
                      scope="", module_depth=0, auto_depth=False)


def _module_levels(path):
    return path.split(".") if path else []


def _in_scope(path, scope):
    """True if `path` is `scope` or inside it. Scope "" contains everything."""
    if not scope:
        return True
    if not path:
        return False
    return path == scope or path.startswith(scope + ".")


def _group_path(path, scope, depth):
    """The module a node is folded into, or None to leave it alone.

    ``depth`` counts levels below the scope: at 1, the scope's immediate
    children are the groups. Measuring from the scope's own level instead made
    a descent skip a rung whenever the intervening module owned no nodes
    directly — scoping into ``transformer.h`` showed ``ln_1``/``attn`` rather
    than the blocks ``0`` and ``1``.
    """
    if not path or depth <= 0:
        return None
    group_level = len(_module_levels(scope)) + depth
    levels = _module_levels(path)
    if len(levels) < group_level:
        return None
    return ".".join(levels[:group_level])


#: How deep auto mode will go before giving up on fitting the budget.
_MAX_AUTO_DEPTH = 8


def _auto_depth(graph, scope, reachable, node_parent, target):
    """The deepest grouping that still fits `target` nodes, or 0 to show them all.

    Node count rises monotonically with depth — finer grouping is always more
    nodes — so the first depth that overflows ends the search, and the one
    before it is the most detail that fits. When even the coarsest grouping
    overflows, that coarsest one is still the best answer available.
    """
    paths = []
    for node in graph.nodes:
        if reachable is not None and node.name not in reachable:
            continue
        if _is_mark_tensor(node):
            continue
        path = _module_path_of(node_parent.get(node.name))
        if not _in_scope(path, scope):
            continue
        paths.append(path)

    if not paths:
        return 0
    if len(paths) <= target:
        return 0                     # it all fits; group nothing

    best = 1
    for depth in range(1, _MAX_AUTO_DEPTH + 1):
        groups, plain = set(), 0
        for path in paths:
            group = _group_path(path, scope, depth)
            if group is None:
                plain += 1
            else:
                groups.add(group)
        if plain + len(groups) > target:
            break
        best = depth
    return best


class _DisplayGraph:
    """Which nodes are drawn, and what stands in for the ones that are not."""

    def __init__(self):
        self.kinds = {}          # node name → display kind
        self.hidden = set()      # node names that are not emitted
        self.passthrough = {}    # hidden name → id that takes its place on edges
        self.unpack_index = {}   # merged getitem name → index into its producer
        self.merged = {}         # producer name → [{"name", "index"}]
        self.chains = {}         # chain id → {"members", "parent"}
        self.expanded_members = {}  # member name → chain id, for chains left open
        self.groups = {}         # group id → {"path", "members"} in depth mode
        self.boundary = {}       # node name → "in" | "out" | "both", at a scope edge
        self.global_inputs = set()  # the method's placeholders, kept in a scoped view

    def resolve(self, name, limit=64):
        """Follow passthrough links to the id that actually gets drawn."""
        seen = 0
        while name in self.passthrough and seen < limit:
            name = self.passthrough[name]
            seen += 1
        return name


def _protected_nodes(bended_acts, aliases):
    """Nodes that must stay individually visible.

    Folding away a node the user has already bent -- or one they reach by an
    alias they declared with ``mark()`` -- would hide their own work from them.
    """
    protected = set(bended_acts or {})
    for members in (aliases or {}).values():
        protected.update(members)
    return protected


def _collapse_chains(graph, dg, opts, protected, node_parent):
    """Fold runs of pure layout ops into one node each.

    A chain is a maximal run of ``layout`` nodes threaded one-to-one: every
    hand-off inside it has a single producer and a single consumer, so the whole
    run is a single reshaping step no matter how many ATen calls it took.  A run
    is cut wherever it would straddle two module panes, since ``addModulePanes``
    sizes a pane from its members' positions and a node belonging to both gives
    it no answer.
    """
    def visible(n):
        return n.name not in dg.hidden

    def vis_users(n):
        return [u for u in n.users if visible(u)]

    def vis_inputs(n):
        return [a for a in n.all_input_nodes if visible(a)]

    def chainable(n):
        return (visible(n)
                and dg.kinds.get(n.name) == "layout"
                and n.name not in protected
                and n.name not in dg.chains
                and n.name not in dg.passthrough)

    for node in graph.nodes:
        if not chainable(node):
            continue
        # Start only at a head: if our single producer is itself an unclaimed
        # layout node feeding only us, it will open the chain instead.
        ins = vis_inputs(node)
        if len(ins) == 1 and chainable(ins[0]) and len(vis_users(ins[0])) == 1:
            continue

        members = [node]
        cur = node
        while True:
            users = vis_users(cur)
            if len(users) != 1:
                break
            nxt = users[0]
            if not chainable(nxt) or len(vis_inputs(nxt)) != 1:
                break
            if node_parent.get(nxt.name) != node_parent.get(cur.name):
                break
            members.append(nxt)
            cur = nxt

        if len(members) < 2:
            continue
        # The collapsed node is the run's *last* member, not a synthetic id: it
        # is a real traced node, so its name, its activation and its bendability
        # are the run's own.  Everything keyed on node names -- favourites,
        # lists, selection, the activation browser -- keeps working untouched.
        chain_id = members[-1].name
        if chain_id in opts.expanded:
            # Drawn in full, but still remembered: every member knows which
            # chain it belongs to, so any of them can fold it back up.
            for mem in members:
                dg.expanded_members[mem.name] = chain_id
            continue

        dg.chains[chain_id] = {"members": members, "parent": node_parent.get(members[0].name)}
        for mem in members[:-1]:
            dg.passthrough[mem.name] = chain_id
            dg.hidden.add(mem.name)


def _collapse_modules(graph, dg, opts, node_parent):
    """Depth mode: show modules, not every operation inside them.

    Two things happen. Anything outside the scope is dropped, so descending into
    a module gives that module's graph rather than the whole model with the rest
    dimmed. And anything deeper than `module_depth` below the scope is folded
    into the one module node that represents it — so the picture is modules and
    the edges between them, at whatever grain is asked for.

    The method's placeholders are the exception: they are the model's inputs,
    not the displayed module's, so they survive every scope. Dropping them left
    a submodule view showing only the ports where values happen to enter it,
    which reads as though the block had inputs of its own.
    """
    scope, depth = opts.scope, opts.module_depth

    if scope:
        dg.global_inputs = {n.name for n in graph.nodes if n.op == "placeholder"}

    # What is inside, so what touches it from outside can be recognised.
    inside = {n.name for n in graph.nodes
              if _in_scope(_module_path_of(node_parent.get(n.name)), scope)}

    for node in graph.nodes:
        if node.name in dg.global_inputs:
            continue
        path = _module_path_of(node_parent.get(node.name))
        if not _in_scope(path, scope):
            # A block whose inputs and outputs are hidden looks like it comes
            # from nowhere and goes nowhere. Anything immediately outside that
            # feeds it or reads from it is kept, drawn as a port.
            feeds = any(u.name in inside for u in node.users)
            reads = any(a.name in inside for a in node.all_input_nodes)
            if feeds or reads:
                dg.boundary[node.name] = ("in" if feeds and not reads
                                          else "out" if reads and not feeds
                                          else "both")
                continue
            dg.hidden.add(node.name)
            continue
        group = _group_path(path, scope, depth)
        if group is None:
            continue
        gid = "__group__%s" % group
        entry = dg.groups.setdefault(gid, {"path": group, "members": []})
        entry["members"].append(node.name)
        dg.passthrough[node.name] = gid
        dg.hidden.add(node.name)


def _group_outputs(by_name, members, reachable):
    """The members of a folded module that anything outside it reads.

    Usually one -- the module's result. A block whose members are all consumed
    internally still has to answer for something, so the last member in
    topological order stands in.
    """
    inside = set(members)
    outs = []
    for name in members:
        node = by_name.get(name)
        if node is None or (reachable is not None and name not in reachable):
            continue
        if any(u.name not in inside for u in node.users) or not list(node.users):
            outs.append(name)
    if outs:
        return outs
    drawn = [m for m in members
             if m in by_name and (reachable is None or m in reachable)]
    return drawn[-1:] if drawn else []


def _build_display_graph(graph, opts, bended_acts, aliases, node_parent):
    dg = _DisplayGraph()
    dg.kinds = node_kinds.classify_graph(
        graph, lambda n: _serialize_target(n.target) if n.target is not None else "")
    protected = _protected_nodes(bended_acts, aliases)

    # mark_tensor: identity markers inserted by mark(), never worth drawing.
    # Unconditional, and not subject to `protected` -- this is what the aliases
    # resolve *through*.
    for node in graph.nodes:
        if _is_mark_tensor(node):
            dg.hidden.add(node.name)
            if node.args and isinstance(node.args[0], torch.fx.Node):
                dg.passthrough[node.name] = node.args[0].name

    # Shape arithmetic.  Dropped outright rather than passed through: a size is
    # not a tensor, so there is no data edge to preserve behind it.
    if opts.hide_shape_calc:
        for node in graph.nodes:
            if dg.kinds.get(node.name) == "shape_calc" and node.name not in protected:
                dg.hidden.add(node.name)

    # Tuple destructuring folds back into the op that produced the tuple.
    if opts.merge_unpack:
        for node in graph.nodes:
            if dg.kinds.get(node.name) != "unpack" or node.name in protected:
                continue
            if node.name in dg.hidden:
                continue
            producer = node.all_input_nodes[0]
            if producer.name in dg.hidden:
                continue
            index = node.args[1] if len(node.args) > 1 else None
            dg.hidden.add(node.name)
            dg.passthrough[node.name] = producer.name
            dg.unpack_index[node.name] = index
            dg.merged.setdefault(producer.name, []).append(
                {"name": node.name, "index": index})

    if opts.collapse_layout:
        _collapse_chains(graph, dg, opts, protected, node_parent)

    # last, so it folds whatever the earlier passes left standing
    if opts.module_depth > 0 or opts.scope:
        _collapse_modules(graph, dg, opts, node_parent)

    return dg


def serialize_graph(bended_module, fn="forward", prune_unreachable=True, display=None):
    """Build the JSON payload the viewer draws.

    Read-only: the traced graph is never modified, and every node keeps its real
    name, so bending and activation lookups are unaffected by what is folded
    away here.  Pass ``display=RAW`` for the unsimplified graph.
    """
    if display is None:
        display = DisplayOptions(prune_unreachable=prune_unreachable)

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

    aliases = {}
    try:
        raw = bended_module.aliases(fn)
        for name, node_names in raw.items():
            aliases[name] = list(node_names)
    except Exception:
        pass

    reachable = _reachable_from_output(graph) if display.prune_unreachable else None

    compound_nodes, node_parent = _build_module_groups(graph, activations=activations)
    if display.auto_depth:
        display = replace(display, module_depth=_auto_depth(
            graph, display.scope, reachable, node_parent, display.max_nodes))
    dg = _build_display_graph(graph, display, bended_acts, aliases, node_parent)

    def _drawn(name):
        """True if `name` reaches the canvas (after passthrough, and not pruned)."""
        real = dg.resolve(name)
        if real in dg.hidden:
            return False
        # a group id is not a traced node name, so it is not in `reachable` —
        # without this every edge leaving a folded module is dropped and the
        # groups float unconnected
        return reachable is None or real in reachable or real in dg.groups

    def _shape_of(node):
        act = activations.get(node.name)
        if act is not None and getattr(act, "shape", None) is not None:
            try:
                return [int(s) for s in act.shape]
            except Exception:
                pass
        # For get_attr (weight) nodes, fall back to reading the tensor shape directly
        if node.op == "get_attr":
            try:
                obj = bended_module._module
                for attr in str(node.target).split("."):
                    obj = getattr(obj, attr)
                if hasattr(obj, "shape"):
                    return [int(s) for s in obj.shape]
            except Exception:
                pass
        return None

    by_name = {n.name: n for n in graph.nodes}
    # In depth mode the module panes would duplicate the group nodes themselves.
    if display.module_depth > 0 or display.scope:
        compound_nodes = [c for c in compound_nodes
                          if _in_scope(c["module_path"], display.scope)
                          and _group_path(c["module_path"], display.scope,
                                          display.module_depth) is None]

    # A module has no tensor of its own -- what it produces are the members
    # anything outside it reads. Named here so acting on a module (pinning it,
    # showing what it made) has real nodes to act on rather than the pane's
    # internal id, which no activation answers to.
    for compound in compound_nodes:
        members = [name for name, parent in node_parent.items()
                   if _in_scope(_module_path_of(parent), compound["module_path"])]
        compound["output_nodes"] = _group_outputs(by_name, members, reachable)
    nodes = list(compound_nodes)   # compound nodes first so Cytoscape resolves parents
    edges = []
    seen_edges = set()
    order = 0

    for node in graph.nodes:
        if node.name in dg.hidden:
            continue
        if reachable is not None and node.name not in reachable:
            continue

        chain = dg.chains.get(node.name)
        emit_id = node.name
        # A collapsed run takes its inputs from where it started; its own value
        # needs no adjusting, since it *is* the last node in the run.
        head = chain["members"][0] if chain else node

        shape = _shape_of(node)

        # source location from tracer code frame
        source_file = source_line = source_fn = None
        act = activations.get(head.name)
        if act is not None and getattr(act, "code", None) is not None:
            source_file = getattr(act.code, "source_file", None)
            source_line = getattr(act.code, "source_line", None)
            source_fn   = getattr(act.code, "source_fn", None)

        bending_cbs = list(bended_acts.get(node.name, []))

        # Degree / shape-change metadata -----------------------------------------
        parent_nodes = []
        for pn in head.all_input_nodes:
            real = dg.resolve(pn.name)
            if real == emit_id or not _drawn(pn.name):
                continue
            if real not in parent_nodes:
                parent_nodes.append(real)
        in_degree = len(parent_nodes)

        out_degree = 0
        for user in node.users:
            if dg.resolve(user.name) == emit_id or not _drawn(user.name):
                continue
            out_degree += 1

        # shape_changed: True when our shape differs from every parent's shape
        # (if no parents have a shape, we leave it False to avoid false positives)
        shape_changed = False
        if shape is not None and parent_nodes:
            parent_shapes_with_data = []
            for pname in parent_nodes:
                pact = activations.get(pname)
                if pact is not None and getattr(pact, "shape", None) is not None:
                    try:
                        parent_shapes_with_data.append(tuple(int(s) for s in pact.shape))
                    except Exception:
                        pass
            if parent_shapes_with_data:
                our_shape = tuple(shape)
                shape_changed = all(our_shape != ps for ps in parent_shapes_with_data)

        target = _serialize_target(node.target) if node.target is not None else None
        kind = dg.kinds.get(node.name, "compute")
        chain_label = None
        if chain:
            chain_label = "→".join(
                node_kinds.op_name(_serialize_target(m.target)) for m in chain["members"])
            # `target` is what the list and details show; the run reads better
            # there than the single aten op the tail happens to be
            target = chain_label

        entry = {
            "id": emit_id,
            # Never the composite text: labels are identity across the viewer
            # (favourites, lists, selection, activation rows), so they have to
            # stay unique and stay equal to the node's real name.
            "label": node.name,
            # Chains keep their members' op so every op-keyed path in the viewer
            # — colours, filters, "is this bendable" — keeps working unchanged.
            "op": node.op,
            "target": target,
            "op_name": node_kinds.op_name(target) if target else None,
            "kind": kind,
            "shape": shape,
            "has_shape": shape is not None,
            "dtype": _dtype_of(node),
            "has_bending": len(bending_cbs) > 0,
            "bending_callbacks": [str(cb) for cb in bending_cbs],
            "args": _serialize_args(head.args),
            "parent": node_parent.get(node.name),
            # the readable path behind `parent`, so the viewer can search on it
            "module_path": _module_path_of(node_parent.get(node.name)),
            "source_file": source_file,
            "source_line": source_line,
            "source_fn":   source_fn,
            "is_trivial": node_kinds.is_trivial(kind),
            "shape_changed": shape_changed,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "order": order,
        }
        order += 1

        if chain:
            entry["is_chain"] = True
            entry["chain_label"] = chain_label
            entry["members"] = [
                {"name": m.name,
                 "target": _serialize_target(m.target),
                 "shape": _shape_of(m)}
                for m in chain["members"]
            ]
            entry["in_shape"] = _shape_of(chain["members"][0])
        # a packed `tb.loop` node stands for several iterations of a body the
        # graph does not show; say how many, and which. Its carry outputs are
        # the loop's state at the end of those iterations. (Stamped at trace
        # time by `torchbend.tracing.loop.stamp_loop_meta`.)
        loop_meta = getattr(node, "meta", {}).get("torchbend_loop")
        if loop_meta:
            start, stop = loop_meta.get("iters", (0, 0))
            entry["loop"] = dict(loop_meta)
            if loop_meta.get("carry"):
                entry["is_loop_carry"] = True
                entry["loop_badge"] = "\u27f3 %s \u00b7 after step %d" % (loop_meta["name"], stop - 1)
            else:
                entry["is_loop"] = True
                entry["loop_badge"] = "\u27f3 %s \u00d7%d \u00b7 steps %d\u2013%d" % (
                    loop_meta["name"], stop - start, start, stop - 1)
        if node.name in dg.boundary:
            entry["is_boundary"] = True
            entry["boundary_dir"] = dg.boundary[node.name]
        if node.name in dg.global_inputs:
            entry["is_global_input"] = True
        if node.name in dg.expanded_members:
            entry["in_chain"] = dg.expanded_members[node.name]
        if node.name in dg.merged:
            entry["merged"] = [
                {"name": m["name"], "index": m["index"],
                 "shape": _shape_of(by_name[m["name"]])}
                for m in dg.merged[node.name]
            ]

        nodes.append(entry)

        for src in head.all_input_nodes:
            real_src = dg.resolve(src.name)
            if real_src == emit_id or not _drawn(src.name):
                continue
            # An unpacked producer can feed the same consumer twice through two
            # different outputs; the index has to be part of the identity or the
            # second operand silently vanishes.
            src_index = dg.unpack_index.get(src.name)
            edge_key = (real_src, emit_id, src_index)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            suffix = "" if src_index is None else "_%s" % src_index
            edges.append({
                "id": f"{real_src}{suffix}__{emit_id}",
                "source": real_src,
                "target": emit_id,
                "label": _get_arg_label(src, head),
                "src_index": src_index,
                "target_op": node.op,
                "target_fn": _serialize_target(node.target) if node.target is not None else "",
            })

    try:
        module_type = type(bended_module._module).__name__
    except Exception:
        module_type = type(bended_module).__name__

    # one node per folded module, carrying what it stands for
    for gid, group in dg.groups.items():
        members = group["members"]
        drawn = [m for m in members if reachable is None or m in reachable]
        if not drawn:
            continue
        leaf = group["path"].split(".")[-1]
        # A group has no activation of its own, but what leaves it does: borrow
        # the shape of the member the outside reads. Without this the group is
        # shapeless, and the "no shape data" dimmer greys out every folded
        # module the moment an input is loaded.
        group_outputs = _group_outputs(by_name, members, reachable)
        out_node = by_name.get(group_outputs[0]) if group_outputs else None
        group_shape = _shape_of(out_node) if out_node is not None else None
        nodes.append({
            "id": gid, "label": gid, "op": "call_module",
            "target": group["path"], "op_name": leaf, "kind": "compute",
            "shape": group_shape, "has_shape": group_shape is not None,
            "dtype": _dtype_of(out_node) if out_node is not None else None,
            "has_bending": any(m in bended_acts for m in members),
            "bending_callbacks": [], "args": [],
            "parent": None, "module_path": group["path"],
            "source_file": None, "source_line": None, "source_fn": None,
            "is_trivial": False, "shape_changed": False,
            "in_degree": 0, "out_degree": 0, "order": order,
            "is_module_group": True, "group_label": leaf,
            "n_members": len(drawn),
            # A group is a stand-in with no tensor of its own. What it actually
            # produces are the members read from outside it -- so clicking one,
            # or following an edge that leaves it, has something real to show.
            "output_nodes": group_outputs,
        })
        order += 1

    for gid, group in dg.groups.items():
        for name in group["members"]:
            node = by_name.get(name)
            if node is None or (reachable is not None and name not in reachable):
                continue
            for src in node.all_input_nodes:
                real_src = dg.resolve(src.name)
                if real_src == gid or not _drawn(src.name):
                    continue
                key = (real_src, gid, None)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append({"id": "%s__%s" % (real_src, gid), "source": real_src,
                              "target": gid, "label": "", "src_index": None,
                              "target_op": "call_module", "target_fn": group["path"]})

    # ── the model's inputs, seen from inside a submodule ──────────────────────
    # The path from a placeholder to the block usually runs through nodes the
    # scope dropped, so the kept placeholders would float unattached. Join each
    # one to the block's entry points -- the drawn nodes nothing drawn feeds --
    # and the inputs sit at the top of the picture where they belong. The edge
    # says the value gets there, not that it arrives untouched, so it is marked
    # indirect and the viewer dashes it.
    if dg.global_inputs:
        drawn_ids = {n["id"] for n in nodes}
        with_incoming = {e["target"] for e in edges}

        def _feeding_inputs(start):
            """Global inputs reaching `start` through nodes this view does not draw."""
            stack, seen, found = [start], {start}, set()
            while stack:
                node = by_name.get(stack.pop())
                if node is None:
                    continue
                for src in node.all_input_nodes:
                    if src.name in dg.global_inputs:
                        found.add(src.name)
                    elif src.name not in drawn_ids and src.name not in seen:
                        seen.add(src.name)
                        stack.append(src.name)
            return found

        for entry in list(nodes):
            name = entry["id"]
            if (name in dg.global_inputs or entry.get("is_compound")
                    or name in with_incoming):
                continue
            # A folded module is an entry point like any other: what feeds it is
            # what feeds any of the members it stands for.
            starts = ([m for m in dg.groups[name]["members"] if m in by_name]
                      if name in dg.groups else
                      [name] if name in by_name else [])
            if not starts:
                continue
            feeding = set()
            for start in starts:
                feeding |= _feeding_inputs(start)
            for ph in sorted(feeding):
                key = (ph, name, None)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append({"id": "%s__%s" % (ph, name), "source": ph,
                              "target": name, "label": "", "src_index": None,
                              "is_indirect": True,
                              "target_op": entry.get("op", ""), "target_fn": ""})

    # A port earns its place by connecting to something. One whose only path
    # into the block ran through a node the folds removed connects to nothing,
    # and would sit on the canvas as an unexplained orphan.  Same for an input
    # the block never reads.
    if dg.boundary or dg.global_inputs:
        touched = set()
        for edge in edges:
            touched.add(edge["source"])
            touched.add(edge["target"])
        nodes = [n for n in nodes
                 if not (n.get("is_boundary") or n.get("is_global_input"))
                 or n["id"] in touched]

    hidden_counts = {"shape_calc": 0, "unpack": 0, "layout": 0, "chains": len(dg.chains)}
    for name in dg.hidden:
        kind = dg.kinds.get(name)
        if kind in hidden_counts:
            hidden_counts[kind] += 1

    # The method's inputs, whatever the view is showing. The input bench is keyed
    # by these: it describes the model, so it must not change because someone
    # descended into a submodule or folded one away.
    placeholders = [
        {"id": n.name, "label": n.name, "op": "placeholder",
         "target": _serialize_target(n.target) if n.target is not None else None,
         "shape": _shape_of(n), "dtype": _dtype_of(n),
         "module_path": None}
        for n in graph.nodes
        if n.op == "placeholder" and (reachable is None or n.name in reachable)
    ]

    # What mark() said about the values, by node -- and, for each element on the
    # canvas, the annotated nodes it stands for: the node itself, or the module
    # box it is folded into. Ordered as the graph runs, so read in sequence the
    # annotations tell the model's process.
    annotations, annotation_targets = {}, {}
    try:
        raw_annotations = bended_module.annotations(fn)
    except Exception:
        raw_annotations = {}
    drawn_ids = {n["id"] for n in nodes}
    order_of = {n.name: i for i, n in enumerate(graph.nodes)}
    for name in sorted(raw_annotations, key=lambda k: order_of.get(k, 1 << 30)):
        entry = dict(raw_annotations[name], node=name, order=order_of.get(name))
        real = dg.resolve(name)
        entry["drawn"] = real if real in drawn_ids else None
        annotations[name] = entry
        if entry["drawn"]:
            annotation_targets.setdefault(entry["drawn"], []).append(name)

    return {"nodes": nodes, "edges": edges, "fn": fn, "module_type": module_type,
            "aliases": aliases, "hidden_counts": hidden_counts,
            "annotations": annotations, "annotation_targets": annotation_targets,
            "placeholders": placeholders,
            "scope": display.scope,
            "module_depth": display.module_depth,
            "auto_depth": display.auto_depth,
            "display": {"prune": display.prune_unreachable,
                        "shape_calc": display.hide_shape_calc,
                        "unpack": display.merge_unpack,
                        "collapse": display.collapse_layout,
                        "expanded": sorted(display.expanded)}}


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
