"""Tests for the graph viewer's display simplification.

The simplification is view-only: it decides what the canvas draws and must never
change what was traced.  These tests pin the two properties that matter -- the
raw view is untouched, and every simplified view is still a well-formed graph.
"""

import pytest
import torch
import torch.nn as nn

import torchbend as tb
from torchbend.ui.graph_viewer import node_kinds, serializer as S


TRACE_METHODS = ["vanilla", "proxy_tensor"]


class ReshapingNet(nn.Module):
    """Exercises every category: real ops, weights, shape arithmetic, view runs."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 8, 3, padding=1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        h = self.conv(x)
        b, c, t = h.shape
        h = h.transpose(1, 2).reshape(b * t, c).contiguous()
        return self.fc(h).view(b, t, 2)


@pytest.fixture(params=TRACE_METHODS)
def traced(request):
    module = tb.BendedModule(ReshapingNet())
    module.trace("forward", trace_method=request.param, x=torch.randn(3, 4, 16))
    return module


def real_nodes(payload):
    return [n for n in payload["nodes"] if not n.get("is_compound")]


def assert_well_formed(payload):
    """Every edge lands on a drawn node, nothing points at itself, nothing duplicates."""
    ids = {n["id"] for n in real_nodes(payload)}
    for edge in payload["edges"]:
        assert edge["source"] in ids, "edge from undrawn node %s" % edge["source"]
        assert edge["target"] in ids, "edge to undrawn node %s" % edge["target"]
        assert edge["source"] != edge["target"], "self-edge on %s" % edge["source"]
    keys = [(e["source"], e["target"], e.get("src_index")) for e in payload["edges"]]
    assert len(keys) == len(set(keys)), "duplicate edges"


def assert_reaches_output(payload):
    """No node is left stranded: each one still feeds the output."""
    nodes = real_nodes(payload)
    preds = {}
    for edge in payload["edges"]:
        preds.setdefault(edge["target"], []).append(edge["source"])
    stack = [n["id"] for n in nodes if n["op"] == "output"]
    seen = set()
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(preds.get(name, []))
    assert {n["id"] for n in nodes} == seen


# ── op_name ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("target,expected", [
    ("aten::sym_size.int", "sym_size"),
    ("aten::view", "view"),
    ("aten::add.Tensor", "add"),
    ("aten::_scaled_dot_product_flash_attention_for_cpu",
     "_scaled_dot_product_flash_attention_for_cpu"),
    ("Tensor.view", "view"),
    ("getitem", "getitem"),
    ("", ""),
])
def test_op_name(target, expected):
    assert node_kinds.op_name(target) == expected


# ── the raw view is the traced graph ──────────────────────────────────────────

def test_raw_is_the_traced_graph(traced):
    """RAW draws exactly the traced nodes -- no folding, no synthetic ids.

    `sync` fingerprints this payload (`sync._graph_hash`), so any drift here
    silently invalidates every saved session on disk.  The only nodes RAW is
    allowed to drop are the mark_tensor markers, which it has always dropped.
    """
    raw = S.serialize_graph(traced, display=S.RAW)
    drawn = {n["id"] for n in real_nodes(raw)}
    graph = traced.graph(fn="forward", bended=True)
    traced_names = {n.name for n in graph.nodes if not S._is_mark_tensor(n)}
    assert drawn == traced_names


def test_raw_folds_nothing(traced):
    raw = S.serialize_graph(traced, display=S.RAW)
    assert raw["hidden_counts"]["shape_calc"] == 0
    assert raw["hidden_counts"]["unpack"] == 0
    assert raw["hidden_counts"]["chains"] == 0
    assert not any(n.get("is_chain") for n in raw["nodes"])
    assert_well_formed(raw)


def test_simplification_does_not_touch_the_trace(traced):
    """Serializing must leave the fx graph exactly as it was."""
    before = [(n.name, n.op, str(n.target)) for n in traced.graph(fn="forward", bended=True).nodes]
    S.serialize_graph(traced)
    S.serialize_graph(traced, display=S.RAW)
    after = [(n.name, n.op, str(n.target)) for n in traced.graph(fn="forward", bended=True).nodes]
    assert before == after


# ── every combination stays a valid graph ─────────────────────────────────────

@pytest.mark.parametrize("shape_calc", [True, False])
@pytest.mark.parametrize("unpack", [True, False])
@pytest.mark.parametrize("collapse", [True, False])
def test_every_combination_is_well_formed(traced, shape_calc, unpack, collapse):
    payload = S.serialize_graph(traced, display=S.DisplayOptions(
        hide_shape_calc=shape_calc, merge_unpack=unpack, collapse_layout=collapse))
    assert_well_formed(payload)
    assert_reaches_output(payload)


def test_labels_stay_unique(traced):
    """Labels are identity across the viewer, so no two drawn nodes may share one.

    A collapsed run is drawn as `view→transpose`; putting that in `label` made
    several chains indistinguishable and broke the activation browser's row
    lookup, which keys on it.
    """
    for opts in (S.RAW, S.DisplayOptions()):
        payload = S.serialize_graph(traced, display=opts)
        labels = [n["label"] for n in real_nodes(payload)]
        assert len(labels) == len(set(labels))
        for node in real_nodes(payload):
            assert node["label"] == node["id"]


def test_simplification_removes_nodes(traced):
    raw = S.serialize_graph(traced, display=S.RAW)
    simple = S.serialize_graph(traced)
    assert len(real_nodes(simple)) < len(real_nodes(raw))
    assert simple["hidden_counts"]["shape_calc"] > 0


# ── chains ────────────────────────────────────────────────────────────────────

def test_chain_reports_its_members(traced):
    payload = S.serialize_graph(traced)
    chains = [n for n in real_nodes(payload) if n.get("is_chain")]
    assert chains, "the reshape run should have collapsed"
    for chain in chains:
        assert len(chain["members"]) >= 2
        assert chain["chain_label"] == "→".join(
            node_kinds.op_name(m["target"]) for m in chain["members"])
        # a collapsed run IS its last node: same name, so its activation and
        # its bendability are reachable by that name like any other node's
        assert chain["id"] == chain["members"][-1]["name"]
        assert chain["label"] == chain["id"]
        assert chain["shape"] == chain["members"][-1]["shape"]


def test_expanding_a_chain_restores_its_members(traced):
    payload = S.serialize_graph(traced)
    chain = next(n for n in real_nodes(payload) if n.get("is_chain"))
    members = [m["name"] for m in chain["members"]]

    opened = S.serialize_graph(traced, display=S.DisplayOptions(
        expanded=frozenset({chain["id"]})))
    ids = {n["id"] for n in real_nodes(opened)}
    assert set(members) <= ids
    assert not any(n.get("is_chain") for n in real_nodes(opened) if n["id"] == chain["id"])
    # each member knows which chain it belongs to, so any of them can re-fold it
    for name in members:
        node = next(n for n in real_nodes(opened) if n["id"] == name)
        assert node["in_chain"] == chain["id"]
    assert_well_formed(opened)
    assert_reaches_output(opened)


# ── bent nodes are never folded away ──────────────────────────────────────────

def test_bent_nodes_stay_visible(traced):
    """Folding away a node the user has bent would hide their own work."""
    payload = S.serialize_graph(traced)
    chain = next(n for n in real_nodes(payload) if n.get("is_chain"))
    victim = chain["members"][-1]["name"]

    traced.bend(tb.Bias(bias=0.1), victim, fn="forward")
    try:
        after = S.serialize_graph(traced)
    finally:
        traced.reset()

    ids = {n["id"] for n in real_nodes(after)}
    assert victim in ids, "%s was bent and must stay drawn" % victim
    assert_well_formed(after)


# ── the search index ──────────────────────────────────────────────────────────
# The activation search reads `FULL`, not the drawn graph: a node folded inside
# a module or hidden by a toggle is still a node someone can go looking for.


def test_full_covers_every_simplified_view(traced):
    """Whatever the view folds, the index still has it."""
    full = {n["id"] for n in real_nodes(S.serialize_graph(traced, display=S.FULL))}
    for opts in (S.DisplayOptions(),
                 S.RAW,
                 S.DisplayOptions(module_depth=1),
                 S.DisplayOptions(auto_depth=True, max_nodes=8)):
        drawn = set()
        for n in real_nodes(S.serialize_graph(traced, display=opts)):
            if n.get("is_module_group"):
                continue
            drawn.add(n["id"])
            for m in (n.get("members") or []) + (n.get("merged") or []):
                drawn.add(m["name"])
        missing = drawn - full
        assert not missing, "drawn but absent from the index: %s" % sorted(missing)


def test_full_folds_and_prunes_nothing(traced):
    payload = S.serialize_graph(traced, display=S.FULL)
    nodes = real_nodes(payload)
    assert not any(n.get("is_module_group") for n in nodes)
    assert not any(n.get("is_chain") for n in nodes)
    # it is a superset of the pruned raw view, and of the default one
    for opts in (S.RAW, S.DisplayOptions()):
        assert len(nodes) >= len(real_nodes(S.serialize_graph(traced, display=opts)))


def test_index_entries_carry_where_they_live(traced):
    """`module_path` is what a search result needs to offer to go there."""
    nodes = real_nodes(S.serialize_graph(traced, display=S.FULL))
    for n in nodes:
        assert "module_path" in n
        assert "kind" in n and n["kind"] is not None
    # this model has real submodules, so something must sit inside one
    assert any(n["module_path"] for n in nodes)


def test_a_folded_node_is_reachable_by_scoping_to_its_module(traced):
    """Scoping to a node's own module is what the jump does — it has to work."""
    folded = S.serialize_graph(traced, display=S.DisplayOptions(module_depth=1))
    drawn = {n["id"] for n in real_nodes(folded)}
    index = real_nodes(S.serialize_graph(traced, display=S.FULL))
    hidden = [n for n in index if n["id"] not in drawn and n["module_path"]]
    if not hidden:
        pytest.skip("nothing folded away at this depth")
    target = hidden[0]
    opened = S.serialize_graph(
        traced, display=S.DisplayOptions(scope=target["module_path"]))
    assert target["id"] in {n["id"] for n in real_nodes(opened)}


# ── inputs belong to the method, not the view ─────────────────────────────────
# Scoping into a module draws a graph with no placeholders of its own. The
# editor used to read the model's inputs off whatever was drawn, concluded they
# were all stale and deleted them -- so every activation inside a submodule
# failed with "No valid inputs provided". These pin the payload facts that fix
# relies on.


def test_a_scoped_view_keeps_the_methods_inputs(traced):
    """The model's inputs are the model's, not the displayed module's.

    Scoped, a placeholder that reaches the block is still drawn -- marked as a
    global input, and joined to the block's entry points by an indirect edge,
    since the ops between them are outside the view. What is *not* drawn is a
    placeholder the block never reads: it would be an orphan on the canvas.
    """
    full = real_nodes(S.serialize_graph(traced, display=S.FULL))
    paths = sorted({n["module_path"] for n in full if n["module_path"]})
    if not paths:
        pytest.skip("no submodules in this model")
    real_inputs = {n["id"] for n in full if n["op"] == "placeholder"}
    assert real_inputs

    for path in paths:
        scoped = S.serialize_graph(traced, display=S.DisplayOptions(scope=path))
        drawn = {n["id"] for n in real_nodes(scoped) if n["op"] == "placeholder"}
        assert drawn <= real_inputs, path
        for n in real_nodes(scoped):
            if n["op"] == "placeholder":
                # a port is what enters the block; these are the method's inputs
                assert n.get("is_global_input"), "%s: %s" % (path, n["id"])
                assert not n.get("is_boundary"), "%s: %s" % (path, n["id"])
        # nothing floats: every drawn input is wired to something
        touched = {e["source"] for e in scoped["edges"]}
        assert drawn <= touched, path


def test_a_scoped_view_reports_the_methods_inputs(traced):
    """The bench reads this list, so it cannot depend on what is drawn."""
    unscoped = S.serialize_graph(traced, display=S.DisplayOptions())
    listed = {p["id"] for p in unscoped["placeholders"]}
    assert listed
    full = real_nodes(S.serialize_graph(traced, display=S.FULL))
    paths = sorted({n["module_path"] for n in full if n["module_path"]})
    for path in paths:
        scoped = S.serialize_graph(traced, display=S.DisplayOptions(scope=path))
        assert {p["id"] for p in scoped["placeholders"]} == listed, path


def test_the_methods_inputs_are_never_inside_a_module(traced):
    """A placeholder folds away with a submodule if it is attributed to one."""
    full = real_nodes(S.serialize_graph(traced, display=S.FULL))
    for n in full:
        if n["op"] in ("placeholder", "output"):
            assert n["module_path"] is None, n["id"]


def test_the_index_still_carries_the_real_placeholders(traced):
    """...and this is what the editor reads instead."""
    full = real_nodes(S.serialize_graph(traced, display=S.FULL))
    phs = [n for n in full if n["op"] == "placeholder"]
    assert phs, "the model has inputs; the index has to show them"
    unscoped = S.serialize_graph(traced, display=S.DisplayOptions())
    drawn_phs = {n["id"] for n in real_nodes(unscoped) if n["op"] == "placeholder"}
    assert drawn_phs <= {n["id"] for n in phs}


def test_placeholders_do_not_depend_on_the_scope(traced):
    """Whatever the view is inside, the method's inputs are the same set."""
    full = real_nodes(S.serialize_graph(traced, display=S.FULL))
    phs = {n["id"] for n in full if n["op"] == "placeholder"}
    for path in {n["module_path"] for n in full if n["module_path"]}:
        scoped_full = S.serialize_graph(
            traced, display=S.DisplayOptions(
                prune_unreachable=False, hide_shape_calc=False, merge_unpack=False,
                collapse_layout=False, scope="", module_depth=0, auto_depth=False))
        assert {n["id"] for n in real_nodes(scoped_full)
                if n["op"] == "placeholder"} == phs, path


# ── what a folded module stands for ───────────────────────────────────────────
# A group node has no tensor of its own. Asking the server for
# `__group__decoder.layers` gets "no activation for ..." -- so the payload has to
# name the members anything outside the group reads, and the editor shows those.


def test_a_group_names_the_members_read_from_outside(traced):
    payload = S.serialize_graph(traced, display=S.DisplayOptions(module_depth=1))
    groups = [n for n in real_nodes(payload) if n.get("is_module_group")]
    if not groups:
        pytest.skip("nothing folds at this depth")
    real = {n["id"] for n in real_nodes(
        S.serialize_graph(traced, display=S.FULL))}
    for g in groups:
        outs = g.get("output_nodes")
        assert outs, "%s stands for nothing" % g["id"]
        for name in outs:
            assert name in real, "%s names %s, which is not a node" % (g["id"], name)
            assert not name.startswith("__group__")


def test_a_pane_names_them_too(traced):
    """A module drawn as a pane has no tensor either.

    Pinning one used to ask the server for `__mod__decoder`, which no activation
    answers to, and put an empty card on the board. What a module produces are
    the members read from outside it — the same answer a folded group gives.
    """
    payload = S.serialize_graph(traced)
    panes = [n for n in payload["nodes"] if n.get("is_compound")]
    if not panes:
        pytest.skip("no submodules in this model")
    real = {n["id"] for n in real_nodes(S.serialize_graph(traced, display=S.FULL))}
    for pane in panes:
        outs = pane.get("output_nodes")
        assert outs, "%s stands for nothing" % pane["id"]
        for name in outs:
            assert name in real, "%s names %s, which is not a node" % (pane["id"], name)
            assert not name.startswith("__mod__")


def test_a_panes_outputs_come_from_inside_it(traced):
    payload = S.serialize_graph(traced, display=S.FULL)
    by_name = {n["id"]: n for n in real_nodes(payload)}
    for pane in [n for n in payload["nodes"] if n.get("is_compound")]:
        path = pane["module_path"]
        for name in pane.get("output_nodes") or []:
            mod = by_name[name]["module_path"] or ""
            assert mod == path or mod.startswith(path + "."), \
                "%s names %s, which lives in %r" % (pane["id"], name, mod)


def test_every_edge_into_output_resolves_to_something_real(traced):
    """The reported bug: the output node was fed by a group and showed nothing."""
    payload = S.serialize_graph(traced, display=S.DisplayOptions(module_depth=1))
    nodes = {n["id"]: n for n in real_nodes(payload)}
    out = next((n for n in nodes.values() if n["op"] == "output"), None)
    if out is None:
        pytest.skip("no output node")
    feeders = [e["source"] for e in payload["edges"] if e["target"] == out["id"]]
    assert feeders, "the output has to be fed by something"
    for f in feeders:
        node = nodes.get(f)
        if node and node.get("is_module_group"):
            assert node.get("output_nodes"), \
                "%s feeds the output but names no node to show" % f


def test_group_outputs_survive_every_depth(traced):
    for depth in (1, 2, 3):
        payload = S.serialize_graph(traced, display=S.DisplayOptions(module_depth=depth))
        for g in real_nodes(payload):
            if g.get("is_module_group"):
                assert g.get("output_nodes"), "depth %d: %s" % (depth, g["id"])


# ── weights belong to their module ────────────────────────────────────────────
# A weight has no activation, so the stack-inspection pass that assigns module
# paths never sees one. Left unattributed they can never be folded, and depth
# mode draws every weight in the model however far you fold -- 541 of them on
# Bark's pipeline, which was most of what reached the canvas.


def test_weights_carry_the_module_they_live_in(traced):
    nodes = real_nodes(S.serialize_graph(traced, display=S.FULL))
    weights = [n for n in nodes if n["op"] == "get_attr" and "." in str(n["target"])]
    if not weights:
        pytest.skip("no nested weights in this model")
    for w in weights:
        assert w["module_path"], "%s has no module" % w["target"]
        assert str(w["target"]).startswith(w["module_path"] + ".")


def test_folding_a_module_folds_its_weights_too(traced):
    payload = S.serialize_graph(traced, display=S.DisplayOptions(module_depth=1))
    drawn = real_nodes(payload)
    grouped = set()
    for n in drawn:
        if n.get("is_module_group"):
            grouped.add(n["module_path"])
    if not grouped:
        pytest.skip("nothing folds at depth 1")
    for n in drawn:
        if n.get("is_module_group") or n["op"] != "get_attr":
            continue
        path = n.get("module_path") or ""
        assert not any(path == g or path.startswith(g + ".") for g in grouped), \
            "%s is inside a folded module but drawn anyway" % n["id"]
