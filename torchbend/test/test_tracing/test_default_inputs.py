"""Seeding the input bench with an expression rather than a value.

``_default_inputs`` normally holds tensors, and a tensor reaches the browser as
the literal it serialises to — a real input arrives as a hundred kilobytes of
digits that nobody can read or edit, on every graph request. ``Expr`` carries
the source instead, which is what the bench wanted all along.
"""

import json

import pytest
import torch

from torchbend.ui.graph_viewer import Expr, as_bench_value
from torchbend.ui.graph_viewer.sync import _serialize_inputs, _deserialize_inputs


# ── the wrapper ───────────────────────────────────────────────────────────────

def test_it_is_its_own_source():
    e = Expr("torch.randn(1, 4, 1500)")
    assert e.source == "torch.randn(1, 4, 1500)"
    assert str(e) == "torch.randn(1, 4, 1500)"


def test_surrounding_space_is_dropped():
    assert Expr("  torch.zeros(3)  ").source == "torch.zeros(3)"


def test_two_of_the_same_source_are_equal():
    assert Expr("torch.zeros(3)") == Expr("torch.zeros(3)")
    assert Expr("torch.zeros(3)") != Expr("torch.ones(3)")
    assert Expr("torch.zeros(3)") != "torch.zeros(3)"
    assert len({Expr("torch.zeros(3)"), Expr("torch.zeros(3)")}) == 1


@pytest.mark.parametrize("bad", ["", "   ", None, 3, torch.zeros(3)])
def test_it_refuses_anything_that_is_not_source(bad):
    with pytest.raises(ValueError):
        Expr(bad)


# ── what reaches the bench ────────────────────────────────────────────────────

def test_an_expr_reaches_the_bench_as_its_source():
    assert as_bench_value(Expr("torch.zeros(1, 4)")) == "torch.zeros(1, 4)"


def test_a_tensor_still_reaches_it_as_a_literal():
    out = as_bench_value(torch.zeros(2, 2))
    assert json.loads(out) == [[0.0, 0.0], [0.0, 0.0]]


def test_an_expr_is_a_fraction_of_the_size_of_the_value():
    """The reason for the wrapper, in numbers."""
    t = torch.zeros(1, 4, 1500)
    assert len(as_bench_value(Expr("torch.zeros(1, 4, 1500)"))) < 40
    assert len(as_bench_value(t)) > 10_000


def test_anything_else_is_left_alone():
    assert as_bench_value(3) == 3
    assert as_bench_value("a prompt") == "a prompt"
    assert as_bench_value(None) is None


# ── surviving a saved session ─────────────────────────────────────────────────

def test_it_round_trips_through_sync_as_source():
    orig = {"t": Expr("torch.randn(1, 4, 1500)")}
    raw = _serialize_inputs(orig)
    assert raw["t"] == {"__type__": "expr", "value": "torch.randn(1, 4, 1500)"}
    assert _deserialize_inputs(raw) == orig


def test_the_saved_form_is_small():
    """A tensor default is saved in full; an Expr is saved as the one line."""
    expr = _serialize_inputs({"x": Expr("torch.zeros(1, 4, 1500)")})
    value = _serialize_inputs({"x": torch.zeros(1, 4, 1500)})
    assert len(str(expr)) * 100 < len(str(value))


def test_the_other_kinds_still_round_trip():
    orig = {"w": torch.zeros(2), "n": 3, "s": "hello"}
    back = _deserialize_inputs(_serialize_inputs(orig))
    assert torch.equal(back["w"], orig["w"])
    assert back["n"] == 3 and back["s"] == "hello"
