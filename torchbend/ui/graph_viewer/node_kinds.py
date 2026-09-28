"""Classify fx nodes into display kinds.

An ATen-level trace spends most of its nodes on bookkeeping: reading a symbolic
dimension out of a tensor, destructuring a multi-output op, relabelling a view.
None of that is what someone opening the graph came to look at, and on GPT-2 it
is 60% of the nodes.  This module puts a name on each category so the serializer
can decide what to fold away; it makes no decisions of its own.

The kinds:

``io``          placeholder / output
``weight``      get_attr
``unpack``      getitem destructuring a tuple/list-valued producer
``shape_calc``  arithmetic on sizes -- the node's value is a SymInt, not a tensor
``layout``      relabels a tensor without selecting or computing anything
``compute``     everything else, i.e. the actual model
"""

import torch


#: Ops that hand back a different view of the same data.  ``slice`` is
#: deliberately absent: slicing selects a subset, which is a real operation, not
#: a relabelling.
_LAYOUT_OPS = frozenset({
    "view", "_unsafe_view", "reshape", "transpose", "t", "permute",
    "expand", "expand_as", "broadcast_to",
    "squeeze", "unsqueeze", "flatten", "unflatten",
    "contiguous", "clone", "detach", "detach_", "alias",
    "movedim", "moveaxis", "to", "_to_copy", "type_as",
})

#: Only consulted when ``node.meta['val']`` is missing -- the vanilla (Python
#: level) tracer records no fake values, so the value-based test below cannot
#: fire and we have to fall back on names.  Kept to ops that can *only* yield a
#: size: ``add`` and friends are missing on purpose, since a name alone cannot
#: tell ``batch + 1`` from a residual connection.  Arithmetic reaches
#: ``shape_calc`` through propagation in :func:`classify_graph` instead.
_SHAPE_CALC_OPS = frozenset({
    "sym_size", "sym_numel", "sym_stride", "sym_float",
    "size", "dim", "ndim", "numel", "len",
})

#: Arithmetic that becomes shape arithmetic when fed only sizes.  ``getitem`` is
#: here for the vanilla tracer, where ``b, c, t = h.shape`` destructures a plain
#: ``torch.Size`` and there is no recorded value to give the game away.
_ARITH_OPS = frozenset({
    "add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "neg",
    "min", "max", "eq", "ne", "lt", "le", "gt", "ge", "index", "getitem",
})

#: Attributes that hand back a shape rather than data, for ``getattr`` nodes.
_SHAPE_ATTRS = frozenset({"shape", "ndim"})

#: Values a node can carry that mean "this is a size, not data".
_SCALARISH = (int, float, bool, str)


def op_name(target: str) -> str:
    """Reduce a serialized target to its bare op name.

    ``'aten::sym_size.int'`` -> ``'sym_size'``, ``'Tensor.view'`` -> ``'view'``,
    ``'getitem'`` -> ``'getitem'``.  ATen op names never contain a dot, so
    everything after the first one is an overload selector and can go; Python
    qualnames are the other way round and keep only their last component.
    """
    if not target:
        return ""
    if "::" in target:
        return target.split("::", 1)[1].split(".", 1)[0]
    return target.rsplit(".", 1)[-1]


def _val(node):
    """``node.meta['val']``, or the sentinel ``NotImplemented`` when absent.

    ``None`` is a legitimate value for a node to carry, so it cannot double as
    "no value recorded".
    """
    try:
        return node.meta.get("val", NotImplemented)
    except AttributeError:
        return NotImplemented


def _is_tensor_valued(node) -> bool:
    return isinstance(_val(node), torch.Tensor)


def is_unpack(node, name: str) -> bool:
    """True for a getitem that destructures a multi-output op.

    Every getitem in a GPT-2 trace is one of these -- ``native_layer_norm``,
    ``split`` and the flash-attention kernel all return tuples that immediately
    get taken apart.  A getitem indexing a *tensor* is real indexing and must
    not be folded into its producer.
    """
    if name != "getitem":
        return False
    inputs = node.all_input_nodes
    if not inputs:
        return False
    return isinstance(_val(inputs[0]), (tuple, list))


def is_shape_calc(node, name: str) -> bool:
    """True for nodes whose value is a size rather than data.

    Prefer the recorded value over the op name: it is exact, and it stays
    correct for ops this module has never heard of.  Only when no value was
    recorded -- the vanilla tracer -- does the name matter.
    """
    if node.op not in ("call_function", "call_method"):
        return False
    val = _val(node)
    if val is not NotImplemented:
        if isinstance(val, (torch.Tensor, tuple, list)):
            return False
        return isinstance(val, _SCALARISH) or type(val).__name__.startswith("Sym")
    if name in _SHAPE_CALC_OPS:
        return True
    # `h.shape` traces as getattr(h, 'shape'); the sizes come out of it
    return (name == "getattr"
            and len(node.args) > 1
            and node.args[1] in _SHAPE_ATTRS)


def classify(node, target: str) -> str:
    """Return the display kind of ``node``.  ``target`` is the serialized target."""
    if node.op in ("placeholder", "output"):
        return "io"
    if node.op == "get_attr":
        return "weight"

    name = op_name(target)
    if is_unpack(node, name):
        return "unpack"
    if is_shape_calc(node, name):
        return "shape_calc"
    # Restricted to calls so a submodule that happens to be named `to` or
    # `expand` is never mistaken for a reshape.  Without a recorded value (the
    # vanilla tracer) the name has to stand on its own.
    if (node.op in ("call_function", "call_method")
            and name in _LAYOUT_OPS
            and (_is_tensor_valued(node) or _val(node) is NotImplemented)):
        return "layout"
    return "compute"


def classify_graph(graph, target_of) -> dict:
    """Classify every node in ``graph``.  ``target_of(node)`` gives the target string.

    Runs :func:`classify` per node, then propagates ``shape_calc`` forwards:
    arithmetic fed only by sizes is itself size arithmetic.  With
    ``proxy_tensor`` the recorded values already settle this, so the fixpoint
    changes nothing; it is what lets the vanilla tracer -- which records no fake
    values -- reach the same answer.
    """
    kinds = {}
    names = {}
    for node in graph.nodes:
        target = target_of(node)
        names[node.name] = op_name(target)
        kinds[node.name] = classify(node, target)

    changed = True
    while changed:
        changed = False
        for node in graph.nodes:
            if kinds[node.name] != "compute" or names[node.name] not in _ARITH_OPS:
                continue
            inputs = node.all_input_nodes
            if inputs and all(kinds.get(a.name) == "shape_calc" for a in inputs):
                kinds[node.name] = "shape_calc"
                changed = True

    return kinds


#: Kinds that compute nothing the model learned.  Drives the ``trivial:`` search
#: filter, which used to be backed by a name table that never matched an ATen
#: target.
TRIVIAL_KINDS = frozenset({"io", "weight", "unpack", "shape_calc", "layout"})


def is_trivial(kind: str) -> bool:
    return kind in TRIVIAL_KINDS
