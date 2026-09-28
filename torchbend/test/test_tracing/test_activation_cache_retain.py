"""The viewer's activation cache keeps only what was asked for, unless the
interface names joints worth keeping (``Method(retain=...)``).

A run passes through many nodes on the way to the one requested. Without help
they are all thrown away, so looking at the node just upstream of the output --
or bending near the end -- recomputes the whole prefix. With ``retain``, the
joints a run passes through are kept at no extra computation.
"""
import torch
import torch.nn as nn

import torchbend as tb
from torchbend.ui.graph_viewer.activation_cache import (
    ActivationCache, run_activations_with_cache)


class _ThreeStages(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b, self.c = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)

    def forward(self, x):
        return self.c(torch.relu(self.b(torch.relu(self.a(x)))))


def _setup():
    torch.manual_seed(0)
    model = _ThreeStages()
    bm = tb.BendedModule(model)
    x = torch.randn(1, 4)
    bm.trace("forward", x=x)
    graph = bm.graph(fn="forward", bended=True)
    linears = [n.name for n in graph.nodes if n.name.startswith("addmm")]
    assert len(linears) == 3
    return bm, model, x, linears


def _ask(bm, cache, x, targets, retain=None):
    return run_activations_with_cache(bm, "forward", {"x": x.clone()}, cache,
                                      target_nodes=list(targets), retain=retain)[0]


def _id(x):
    from torchbend.ui.graph_viewer.activation_cache import _hash_inputs
    return _hash_inputs({"x": x})


def test_without_retain_only_the_request_is_kept():
    bm, _, x, (first, second, third) = _setup()
    cache = ActivationCache()
    _ask(bm, cache, x, [third])
    assert cache.is_clean("forward", _id(x), third)
    assert not cache.is_clean("forward", _id(x), first)
    assert not cache.is_clean("forward", _id(x), second)


def test_retained_joints_on_the_way_are_kept_and_exact():
    bm, model, x, (first, second, third) = _setup()
    cache = ActivationCache()
    got = _ask(bm, cache, x, [third], retain={first, second})
    assert set(got) == {third}, "extras are kept, not returned"
    for node in (first, second):
        assert cache.is_clean("forward", _id(x), node)
    # what was kept is what the model computes there
    with torch.no_grad():
        expected = model.b(torch.relu(model.a(x)))
    assert torch.allclose(cache.get_clean("forward", _id(x), second), expected, atol=1e-6)
    # ... so looking at it later costs nothing new
    again = _ask(bm, cache, x, [second])
    assert torch.allclose(again[second], expected, atol=1e-6)


def test_joints_the_run_did_not_pass_through_are_not_computed():
    """Retaining is free because it only keeps what a run computed anyway: a
    joint upstream of a clean node the run resumed from is not recomputed."""
    bm, _, x, (first, second, third) = _setup()
    cache = ActivationCache()
    _ask(bm, cache, x, [second])                       # `second` is now clean; `first` is not
    assert not cache.is_clean("forward", _id(x), first)
    _ask(bm, cache, x, [third], retain={first, second})
    assert not cache.is_clean("forward", _id(x), first), \
        "the run resumed from `second`; `first` was never computed"


def test_retained_joints_are_pinned_against_eviction():
    bm, _, x, (first, second, third) = _setup()
    cache = ActivationCache()
    _ask(bm, cache, x, [third], retain={first})
    assert first in cache._pinned


def test_a_joint_that_does_not_fit_is_dropped_not_the_answer():
    bm, _, x, (first, second, third) = _setup()
    cache = ActivationCache(max_bytes=16 * 4)          # room for one [1, 4] tensor... barely
    got = _ask(bm, cache, x, [third], retain={first, second})
    assert third in got, "the requested node is always returned"


def test_the_callers_inputs_are_not_consumed():
    """`inputs_for_fn` used to pop from the dict it was given, so a caller that
    reused its inputs for the next request found them gone."""
    bm, _, x, (first, second, third) = _setup()
    cache = ActivationCache()
    kwargs = {"x": x}
    run_activations_with_cache(bm, "forward", kwargs, cache, target_nodes=[third])
    assert "x" in kwargs


# ── a retrace replaces the graph ─────────────────────────────────────────────

def test_graph_changed_drops_everything_built_from_the_old_graph():
    from torchbend.ui.graph_viewer.bending_session import BendingSession
    bm, _, x, (first, second, third) = _setup()
    session = BendingSession()
    session.get_cached_activations(bm, "forward", {"x": x}, target_nodes=[third])
    assert "forward" in session._bent_graph_cache and session._get_cache()._entries
    session.graph_changed("forward")
    assert "forward" not in session._bent_graph_cache
    assert "forward" not in session._bent_module_cache
    assert not session._get_cache()._entries


def test_a_retraces_own_output_is_kept_for_the_inputs_it_ran_on():
    """The retrace ran the model on the bench's inputs already; the next view
    of the output is served from that, not computed a second time."""
    from torchbend.ui.graph_viewer.bending_session import BendingSession
    from torchbend.ui.graph_viewer.activation_cache import _hash_inputs
    bm, model, x, (first, second, third) = _setup()
    session = BendingSession()
    traced = bm.trace("forward", x=x, _return_out=True)
    session.graph_changed("forward")
    kept = session.retain_output(bm, "forward", {"x": x}, traced[1])
    assert kept == [third]
    stored = session._get_cache().get_clean("forward", _hash_inputs({"x": x}), third)
    with torch.no_grad():
        assert torch.allclose(stored, model(x), atol=1e-6)


def test_a_bent_model_does_not_keep_the_retraces_unbent_output():
    from torchbend.bending import Mask
    from torchbend.ui.graph_viewer.bending_session import BendingSession
    bm, _, x, _ = _setup()
    bm.bend(Mask(prob=0.0), "a.weight")
    traced = bm.trace("forward", x=x, _return_out=True)
    assert BendingSession().retain_output(bm, "forward", {"x": x}, traced[1]) == []
