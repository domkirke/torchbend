"""A re-trace has to reproduce how a method was configured, not just its inputs.

Tracing takes two kinds of keyword: the tensors a method runs on, and the flags
that decide what it does. A bench can supply the first; only the original trace
knows the second. GPT-2 is the case that forced this — it must be traced with
``use_cache=False``, because a transformers ``Cache`` cannot pass through fx,
and re-tracing without it fails on the cache rather than on anything the user
touched.
"""

import pytest
import torch
import torch.nn as nn

import torchbend as tb


class Configurable(nn.Module):
    """A method whose behaviour changes with a non-tensor keyword."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x, doubled: bool = False, scale: float = 1.0):
        out = self.fc(x) * scale
        return out * 2 if doubled else out


@pytest.fixture(params=["vanilla", "proxy_tensor"])
def traced(request):
    module = tb.BendedModule(Configurable())
    module.trace("forward", trace_method=request.param,
                 x=torch.randn(2, 4), doubled=True, scale=3.0)
    return module


def test_non_tensor_keywords_are_remembered(traced):
    assert traced.trace_kwargs("forward") == {"doubled": True, "scale": 3.0}


def test_tensors_are_not_remembered(traced):
    """Inputs are what a re-trace replaces; keeping them would defeat the point."""
    assert "x" not in traced.trace_kwargs("forward")


def test_untraced_method_has_no_kwargs(traced):
    assert traced.trace_kwargs("nope") == {}


def test_returns_a_copy(traced):
    """Callers merge into this; mutating it must not rewrite the record."""
    traced.trace_kwargs("forward")["doubled"] = "clobbered"
    assert traced.trace_kwargs("forward")["doubled"] is True


def test_retracing_with_them_reproduces_the_configuration(traced):
    """The round trip the viewer's retrace endpoint performs."""
    remembered = traced.trace_kwargs("forward")
    traced.trace(fn="forward", **{**remembered, "x": torch.randn(5, 4)})
    assert traced.trace_kwargs("forward") == {"doubled": True, "scale": 3.0}
    shapes = [n.meta["val"].shape for n in traced.graph("forward").nodes
              if n.op == "placeholder" and hasattr(n.meta.get("val"), "shape")]
    assert shapes and shapes[0][0] == 5


def test_bench_inputs_win_over_remembered_keywords(traced):
    """Merge order matters: the bench overrides, it does not get overridden."""
    remembered = traced.trace_kwargs("forward")
    traced.trace(fn="forward", **{**remembered, "x": torch.randn(2, 4), "scale": 9.0})
    assert traced.trace_kwargs("forward")["scale"] == 9.0


def test_save_as_records_under_the_saved_name():
    module = tb.BendedModule(Configurable())
    module.trace("forward", trace_method="proxy_tensor",
                 x=torch.randn(2, 4), doubled=True, _save_as="doubled_forward")
    assert module.trace_kwargs("doubled_forward") == {"doubled": True}


def test_a_copy_remembers_how_it_was_traced(traced):
    """A copy keeps the graphs, so it has to keep their configuration too."""
    copy = tb.BendedModule.copy(traced)
    assert copy.trace_kwargs("forward") == {"doubled": True, "scale": 3.0}


# ── tracer configuration ──────────────────────────────────────────────────────
# `trace_kwargs` remembers what the *model* was called with. The tracer's own
# settings are configuration too, and a retrace that loses them is not the same
# trace -- a graph that needed `_wrap_recurrent` to trace cannot re-trace.


class _Recurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(4, 4, batch_first=True)
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(h)


def test_trace_config_records_the_tracer_settings():
    m = tb.BendedModule(_Recurrent())
    m.trace("forward", x=torch.randn(2, 6, 4), _wrap_recurrent=True)
    cfg = m.trace_config("forward")
    assert cfg["_wrap_recurrent"] is True
    assert cfg["trace_method"] in (None, "proxy_tensor", "vanilla")


def test_trace_config_is_empty_for_an_untraced_method():
    m = tb.BendedModule(_Recurrent())
    assert m.trace_config("forward") == {}


def test_retracing_with_the_recorded_config_reproduces_the_graph():
    m = tb.BendedModule(_Recurrent())
    m.trace("forward", x=torch.randn(2, 6, 4), _wrap_recurrent=True)
    before = len(m.graph("forward").nodes)
    # exactly what api_retrace does: replay the config, swap the inputs
    m.trace(fn="forward", **m.trace_config("forward"),
            **{**m.trace_kwargs("forward"), "x": torch.randn(2, 11, 4)})
    assert len(m.graph("forward").nodes) == before


# Not tested here: that `_trace_config` survives `copy.copy`. It is in
# `__copy_attrs__` beside `_trace_kwargs`, so it travels the same way -- but
# `copy.copy` on *any* BendedModule currently raises RecursionError, which
# predates this and has nothing to do with the tracer config.
