"""Tests for `tb.loop` — compact loops in traced graphs.

The wider suite is red at baseline, so everything here is self-contained:
each test builds its own module and compares packed tracing against an
unrolled reference traced the same way, rather than against recorded values.
"""
import pytest
import torch
import torch.nn as nn

import torchbend as tb
from torchbend.bending import Mask
from torchbend.tracing.loop import BendingLoopError, is_loop_node


N_ITER = 12


class Refine(nn.Module):
    """A fixed-shape carry looped N times — the shape `tb.loop` is built for.

    The body is deliberately several layers deep: unrolling costs
    iterations × body, and a one-op body would not show that.
    """

    def __init__(self, dim=8, n_iter=N_ITER):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                   nn.Linear(dim, dim), nn.ReLU(),
                                   nn.Linear(dim, dim))
        self.n_iter = n_iter

    def forward(self, x):
        def step(i, carry):
            h, = carry
            return (torch.tanh(self.lin(h) + self.block(h)),)
        h, = tb.loop(step, (x,), self.n_iter, name="refine",
                     modules=[self.lin, self.block])
        return h


class Grow(nn.Module):
    """Violates the fixed-shape carry contract on purpose."""

    def __init__(self, dim=8, n_iter=N_ITER):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.n_iter = n_iter

    def forward(self, x):
        def step(i, carry):
            h, = carry
            return (torch.cat([h, h[:, :1]], dim=1),)
        h, = tb.loop(step, (x,), self.n_iter, name="grow", modules=[self.lin])
        return h[:, :8]


GRANULARITIES = {
    "unroll":   {"mode": "unroll"},
    "pack_1":   {"mode": "pack", "pack": 1},
    "pack_3":   {"mode": "pack", "pack": 3},
    "pack_all": {"mode": "pack", "pack": N_ITER},
    "window":   {"mode": "pack", "pack": 4, "unroll_range": (0, 2)},
}


@pytest.fixture
def state_dict():
    torch.manual_seed(0)
    return Refine().state_dict()


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(2, 8)


def _traced(state_dict, x, policy, cls=Refine):
    bm = tb.BendedModule(cls())
    bm._module.load_state_dict(state_dict, strict=False)
    bm.trace("forward", x=x, _loop_policy=policy)
    return bm


def _n_nodes(bm):
    return len(list(bm.graph(fn="forward", bended=True).nodes))


def test_transparent_outside_tracing(state_dict, x):
    """Untraced, `loop` is just a Python loop — safe to leave in model code."""
    module = Refine()
    module.load_state_dict(state_dict)
    expected = x
    for _ in range(N_ITER):
        expected = torch.tanh(module.lin(expected) + module.block(expected))
    assert torch.allclose(module(x), expected, atol=1e-6)


@pytest.mark.parametrize("name", list(GRANULARITIES))
def test_output_matches_unrolled(state_dict, x, name):
    reference = _traced(state_dict, x, GRANULARITIES["unroll"])(x)
    assert torch.allclose(_traced(state_dict, x, GRANULARITIES[name])(x),
                          reference, atol=1e-6)


@pytest.mark.parametrize("name", list(GRANULARITIES))
def test_bent_weights_reach_packed_loop(state_dict, x, name):
    """The crux: a packed loop is opaque, but its parameters arrive as an
    explicit argument sourced from the already-bent module copy."""
    reference = _traced(state_dict, x, GRANULARITIES["unroll"])
    unbent = reference(x)
    reference.bend(Mask(prob=0.5, seed=3), "lin.weight")
    bent_reference = reference(x)
    assert not torch.allclose(unbent, bent_reference), "bending changed nothing"

    bm = _traced(state_dict, x, GRANULARITIES[name])
    bm.bend(Mask(prob=0.5, seed=3), "lin.weight")
    assert torch.allclose(bm(x), bent_reference, atol=1e-6)


def test_packing_shrinks_the_graph(state_dict, x):
    counts = {k: _n_nodes(_traced(state_dict, x, p)) for k, p in GRANULARITIES.items()}
    assert counts["pack_all"] < counts["pack_3"] < counts["pack_1"] < counts["unroll"]
    # the point of the exercise: unrolling costs iterations × body, packing
    # costs iterations (a loop node, its carry, and its parameter reads)
    assert counts["pack_1"] < counts["unroll"] / 2
    assert counts["pack_all"] < counts["unroll"] / 10


def test_auto_packs_only_past_the_threshold(state_dict, x):
    short = _traced(state_dict, x, {"mode": "auto", "max_unroll": N_ITER + 1})
    long = _traced(state_dict, x, {"mode": "auto", "max_unroll": 2})
    n_loop = lambda bm: sum(1 for n in bm.graph(fn="forward", bended=True).nodes
                            if is_loop_node(n))
    assert n_loop(short) == 0
    assert n_loop(long) == N_ITER          # auto packs at pack=1 by default


def test_carry_between_packs_is_bendable(state_dict, x):
    """Pack size is the temporal resolution of activation bending."""
    bm = _traced(state_dict, x, {"mode": "pack", "pack": 4})
    activations = bm.graph(fn="forward", bended=True).activations

    # the loop node itself has no shape; its carry children do
    assert activations["loop_fwd"].shape is None
    assert list(activations["getitem"].shape) == [2, 8]
    assert activations["getitem"].loop["carry"] is True
    assert activations["loop_fwd"].loop["iters"] == (0, 4)

    base = bm(x)
    bm.bend(Mask(prob=0.5, seed=5), "getitem")
    assert not torch.allclose(bm(x), base)


def test_growing_carry_is_rejected(state_dict, x):
    """The contract is enforced where it can be seen. The body never runs under
    fake tensors, but `_make_fx_raw` runs the forward eagerly once before
    tracing, so a violation is caught at trace time rather than much later."""
    with pytest.raises(BendingLoopError, match="carry slot 0"):
        _traced(state_dict, x, {"mode": "pack", "pack": 4}, cls=Grow)


def test_captured_tensor_is_rejected(state_dict, x):
    """A body that closes over a tensor must be refused, not silently replayed
    with the value that tensor had while tracing.

    This is a regression test: BLIP's decode loop captured its vision features
    that way, and the packed loop happily replayed the *fake* trace-time
    tensors, returning a FakeTensor full of unbacked symbols instead of a
    caption. Failing loudly at trace time is the only safe behaviour.
    """
    class Captures(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            offset = x * 2                      # a tensor the graph computes

            def step(i, carry):
                h, = carry
                return (torch.tanh(self.lin(h)) + offset,)   # captured!

            h, = tb.loop(step, (x,), N_ITER, name="captures", modules=[self.lin])
            return h

    bm = tb.BendedModule(Captures())
    with pytest.raises(BendingLoopError, match="closes over tensor"):
        bm.trace("forward", x=x, _loop_policy={"mode": "pack", "pack": 4})


def test_captured_tensor_is_fine_when_unrolled(state_dict, x):
    """Unrolled there is no replay, so capturing is harmless — and allowed."""
    class Captures(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            offset = x * 2

            def step(i, carry):
                h, = carry
                return (torch.tanh(self.lin(h)) + offset,)

            h, = tb.loop(step, (x,), N_ITER, name="captures2", modules=[self.lin])
            return h

    module = Captures()
    bm = tb.BendedModule(Captures())
    bm._module.load_state_dict(module.state_dict())
    bm.trace("forward", x=x, _loop_policy={"mode": "unroll"})
    assert torch.allclose(bm(x), module(x), atol=1e-6)


def test_carried_tensor_is_the_supported_route(state_dict, x):
    """Passing the same tensor through the carry packs correctly."""
    class Carries(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            offset = x * 2

            def step(i, carry):
                h, off = carry
                return (torch.tanh(self.lin(h)) + off, off)

            h, _ = tb.loop(step, (x, offset), N_ITER, name="carries",
                           modules=[self.lin])
            return h

    module = Carries()
    bm = tb.BendedModule(Carries())
    bm._module.load_state_dict(module.state_dict())
    bm.trace("forward", x=x, _loop_policy={"mode": "pack", "pack": 4})
    out = bm(x)
    assert not isinstance(out, torch._subclasses.fake_tensor.FakeTensor)
    assert torch.allclose(out, module(x), atol=1e-6)


def test_unrolled_carry_keeps_full_granularity(state_dict, x):
    """Unrolled, every inner activation is still individually addressable."""
    bm = _traced(state_dict, x, GRANULARITIES["unroll"])
    tanh_nodes = [n for n in bm.activation_names() if "tanh" in n]
    assert len(tanh_nodes) == N_ITER


def test_registry_is_stable_across_retraces(state_dict, x):
    from torchbend.tracing.loop import _BODY_REGISTRY
    bm = tb.BendedModule(Refine())
    bm._module.load_state_dict(state_dict)
    for _ in range(3):
        bm.trace("forward", x=x, _loop_policy={"mode": "pack", "pack": 4})
    keys = [k for k in _BODY_REGISTRY if k.startswith("refine")]
    assert keys == ["refine"], "re-tracing should reuse the key, not mint new ones"


def test_policy_survives_trace_config(state_dict, x):
    """`api_retrace` replays `_trace_config`, so packing must live there."""
    bm = _traced(state_dict, x, {"mode": "pack", "pack": 4})
    assert bm.trace_config("forward")["_loop_policy"] == {"mode": "pack", "pack": 4}
