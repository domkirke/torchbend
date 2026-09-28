"""`tb.mark`: aliases are optional; a description and metadata ride along.

Whatever a mark says about a value survives tracing, with either tracer, and
reaches the graph viewer: `BendedModule.annotations()` lists it by node, and the
serializer resolves each annotation to the element drawn for it.
"""
import pytest
import torch
import torch.nn as nn

import torchbend as tb
from torchbend.tracing.mark import decode_info, encode_info, mark


class _Marked(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 2)

    def forward(self, x):
        h = mark(self.a(x), name="hidden", description="the first projection", meta={"stage": 1})
        h = mark(torch.relu(h), description="rectified", meta={"stage": 2})   # no alias
        return mark(self.b(h), name="out", meta={"axes": "batch, features",
                                                 "device": torch.device("cpu")})   # no description


@pytest.fixture(params=["proxy_tensor", "vanilla"])
def traced(request):
    bm = tb.BendedModule(_Marked())
    bm.trace("forward", x=torch.randn(2, 4), trace_method=request.param)
    return bm


def test_aliases_are_optional(traced):
    aliases = traced.aliases()
    assert set(aliases) == {"hidden", "out"}        # the unnamed mark made none
    annotations = traced.annotations()
    unnamed = [a for a in annotations.values() if a["description"] == "rectified"]
    assert len(unnamed) == 1 and unnamed[0]["alias"] is None and unnamed[0]["aliases"] == []


def test_descriptions_and_metadata_survive_tracing(traced):
    by_alias = {a["alias"]: a for a in traced.annotations().values() if a["alias"]}
    hidden = by_alias["hidden"]
    assert hidden["description"] == "the first projection" and hidden["meta"] == {"stage": 1}
    out = by_alias["out"]
    assert out["description"] is None
    assert out["meta"]["axes"] == "batch, features"
    assert out["meta"]["device"] == "device(type='cpu')"      # not JSON: kept as repr


def test_an_alias_still_bends(traced):
    assert traced.activations("#hidden")
    assert traced.forward(x=torch.randn(3, 4)).shape == (3, 2)


def test_annotations_are_a_copy(traced):
    traced.annotations().clear()
    assert traced.annotations()


def test_mark_is_a_passthrough_outside_tracing():
    x = torch.randn(3)
    assert mark(x, description="just a tensor", meta={"unit": "dB"}) is x


def test_decorator_forms_carry_the_description():
    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)

        @mark(name="decoded", description="what decode returns", meta={"step": 3})
        def forward(self, x):
            return self.a(x)

    bm = tb.BendedModule(Dec())
    bm.trace("forward", x=torch.randn(2, 4))
    (entry,) = [a for a in bm.annotations().values() if a["alias"] == "decoded"]
    assert entry["description"] == "what decode returns" and entry["meta"] == {"step": 3}


def test_a_node_marked_twice_keeps_both():
    class Twice(nn.Module):
        def forward(self, x):
            y = torch.tanh(x)
            y = mark(y, name="first", description="one", meta={"a": 1})
            return y * 2

    from torchbend.tracing.mark import add_annotation
    table = {}
    add_annotation(table, "n", "first", encode_info("one", {"a": 1}))
    add_annotation(table, "n", "second", encode_info("two", {"b": 2}))
    assert table["n"]["aliases"] == ["first", "second"] and table["n"]["alias"] == "first"
    assert table["n"]["description"] == "two" and table["n"]["meta"] == {"a": 1, "b": 2}


def test_info_round_trip():
    assert encode_info(None, {}) is None
    assert decode_info(None) == {"description": None, "meta": {}}
    assert decode_info(encode_info("d", {"k": [1, 2]})) == {"description": "d", "meta": {"k": [1, 2]}}
    assert decode_info("not json") == {"description": "not json", "meta": {}}


def test_the_viewer_gets_annotations_in_graph_order(traced):
    from torchbend.ui.graph_viewer.serializer import serialize_graph
    data = serialize_graph(traced, fn="forward")
    entries = list(data["annotations"].values())
    assert [e["description"] for e in entries] == ["the first projection", "rectified", None]
    assert all(e["drawn"] for e in entries)
    for e in entries:
        assert e["node"] in data["annotation_targets"][e["drawn"]]


def test_a_model_that_marks_with_metadata_still_scripts():
    """TorchScript compiles mark()'s signature: it must stay scriptable."""
    import torchbend.test.test_tracing._scriptable_mark as m
    model = m.ScriptableMarked()
    scripted = torch.jit.script(model)
    x = torch.randn(2, 4)
    # scripted, a mark is the identity
    assert torch.equal(scripted(x), model.a(x))
