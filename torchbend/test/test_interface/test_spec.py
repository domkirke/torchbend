"""`torchbend.interfaces.spec`: what an interface declares about itself.

Everything a script or a UI asks an interface -- its input modes, options,
callbacks, token decoding, which devices and batching its methods support,
which activations are worth keeping -- is declared with these classes and read
through ``iface.spec``. These tests pin down the declarations' validation, the
answers ``iface.spec`` gives, and the JSON a frontend is handed.
"""
import json

import pytest
import torch
import torch.nn as nn

from torchbend.interfaces.base import Interface
from torchbend.interfaces.spec import (
    Audio, Bool, Callback, Choice, Float, Image, Input, InputMode, Int, Method,
    Option, Ref, SpecError, Str, Tensor, Text, Tokens, Value)


class _Tiny(nn.Module):
    def forward(self, x):
        return torch.tanh(x) * 2

    def decode(self, x):
        return x + 1


class _Tokenizer:
    eos_token_id = 7


class _Iface(Interface):
    """A small interface declaring a bit of everything."""

    methods = {
        "forward": Method(
            inputs={"x": Input("torch.randn(1, 4)", mode=InputMode(
                Text(default=Ref("prompt"), placeholder="say something"),
                encode="encode_text", also_fills=["mask"], label="prompt"))},
            outputs=[Audio(sample_rate=Ref("rate"))],
            batch=False,
            devices={"mps": False, "cuda": False},
            retain="_joints",
            before_retrace="_before"),
    }
    options = {
        "n_tokens": Option(Int(range=(1, 64)), attr="n_tokens", needs="retrace"),
        "side": Option(Choice(["left", "right"]), get="get_side", set="set_side",
                       label="padding side"),
        "loop": Option(Int(), attr="n_tokens", when="full"),
    }
    callbacks = {
        "generate": Callback(returns=Text(), args={
            "prompt": Text(placeholder="prompt…"),
            "n_tokens": Int(range=(1, 256)),
        }),
    }
    tokens = Tokens(decode="decode_ids", logits="logits_of")

    def __init__(self):
        self.n_tokens = 8
        self._side = "left"
        self.full = True
        self.rate = 16000
        self.prompt = "hello"
        self.tokenizer = _Tokenizer()
        super().__init__(_Tiny())

    def bend_model(self, model):
        model.trace("forward", x=torch.randn(1, 4))
        model.trace("decode", x=torch.randn(1, 4))

    def encode_text(self, text):
        texts = [text] if isinstance(text, str) else list(text)
        width = max(len(t.split()) for t in texts)
        return torch.zeros(len(texts), width), torch.ones(len(texts), width)

    def get_side(self):
        return self._side

    def set_side(self, value):
        self._side = value

    def generate(self, prompt: str = "hi", n_tokens: int = 12, seed=None):
        """Continue ``prompt`` for a while."""
        return "%s (%d)" % (prompt, n_tokens)

    def decode_ids(self, ids, skip_special_tokens=True):
        return "one string"

    def logits_of(self, ids):
        return torch.zeros(ids.shape[0], ids.shape[1], 10)

    def _joints(self, fn, graph):
        return {"out": [n for n in graph.nodes if n.op == "output"][0].all_input_nodes[0].name}

    def _before(self, fn):
        return {"_loop_policy": {"mode": "pack", "pack": self.n_tokens}}


@pytest.fixture
def iface():
    return _Iface()


# ── class-time validation ────────────────────────────────────────────────────

def _declare(**attrs):
    return type("_Declared", (Interface,), attrs)


@pytest.mark.parametrize("attrs, expected", [
    ({"options": {"x": Option(Choice([]), attr="x")}}, "no choices"),
    ({"options": {"x": Option(Int(range=(1, 2, 3)), attr="x")}}, "(min, max)"),
    ({"options": {"x": Option(Int(range=(5, 1)), attr="x")}}, "min above max"),
    ({"options": {"x": Option(Int(), attr="x", get="g", set="s")}}, "pick one"),
    ({"options": {"x": Option(Int())}}, "either `attr`"),
    ({"options": {"x": Option(Int(), attr="x", needs="coffee")}}, "needs"),
    ({"options": {"x": {"type": "int"}}}, "must be a Option"),
    ({"callbacks": {"c": Callback(returns=Int())}}, "medium"),
    ({"callbacks": {"c": Callback(args={"a": "text"})}}, "must be a Value"),
    ({"methods": {"f": Method(inputs={"x": Input(mode=InputMode(Int(), encode="e"))})}}, "medium"),
    ({"methods": {"f": Method(devices={"mps": "no"})}}, "device type: bool"),
    ({"methods": {"f": Method(inputs={"x": "torch.zeros(1)"})}}, "must be an Input"),
    ({"methods": {"f": Method(retain=3)}}, "method name or a callable"),
    ({"tokens": "decode"}, "must be a Tokens"),
])
def test_malformed_declarations_fail_when_the_class_is_created(attrs, expected):
    with pytest.raises(SpecError) as exc:
        _declare(**attrs)
    assert expected in str(exc.value)


@pytest.mark.parametrize("old", ["_input_modes_", "_options_", "_callbacks_", "_device_compat_",
                                 "_batch_compat_", "_token_decoder_", "_token_logits_",
                                 "_default_inputs"])
def test_retired_attributes_are_an_error_not_a_silent_no_op(old):
    with pytest.raises(SpecError) as exc:
        _declare(**{old: {}})
    assert "retired" in str(exc.value)


def test_a_value_class_is_accepted_for_an_instance():
    cls = _declare(callbacks={"c": Callback(returns=Text)})
    assert isinstance(cls.callbacks["c"].returns, Text)


# ── validation against the instance ──────────────────────────────────────────

class _Named(_Iface):
    pass


@pytest.mark.parametrize("override, expected", [
    ({"options": {"x": Option(Int(), attr="missing_attr")}}, "missing_attr"),
    ({"options": {"x": Option(Int(), get="nope", set="set_side")}}, "nope"),
    ({"callbacks": {"missing": Callback()}}, "missing"),
    ({"callbacks": {"generate": Callback(args={"temperature": Float()})}}, "temperature"),
    ({"methods": {"forward": Method(inputs={"x": Input(mode=InputMode(Text(), encode="nope"))})}},
     "nope"),
    ({"methods": {"forward": Method(retain="nope")}}, "nope"),
    ({"tokens": Tokens(decode="nope")}, "nope"),
])
def test_names_that_do_not_exist_fail_on_first_use(iface, override, expected):
    for attr, value in override.items():
        setattr(iface, attr, value)
    with pytest.raises(SpecError) as exc:
        iface.spec.validate()
    assert expected in str(exc.value)


def test_an_instance_can_swap_a_declaration(iface):
    assert iface.spec.method("forward").batch is False
    iface.methods = {"forward": Method()}
    assert iface.spec.method("forward").batch is True


# ── methods ──────────────────────────────────────────────────────────────────

def test_an_undeclared_method_gets_the_permissive_defaults(iface):
    decode = iface.spec.method("decode")
    assert not decode.declared
    assert decode.batch is True
    assert decode.runs_on("mps") and decode.runs_on("cuda")
    assert decode.input_modes() == {} and decode.default_inputs() == {}
    assert decode.retained(iface.model.graph(fn="decode")) == {}
    assert decode.before_retrace() == {}


def test_batch_and_devices(iface):
    forward = iface.spec.method("forward")
    assert forward.batch is False
    assert forward.runs_on("cpu") is True
    assert forward.runs_on("mps") is False
    # matched by device type, whatever form it comes in
    assert forward.runs_on(torch.device("mps")) is False
    assert forward.runs_on("cuda:0") is False
    assert forward.devices(["cpu", "mps"]) == {"cpu": True, "mps": False}


def test_retain_and_before_retrace_hooks(iface):
    forward = iface.spec.method("forward")
    graph = iface.model.graph(fn="forward")
    joints = forward.retained(graph)
    assert set(joints) == {"out"} and joints["out"] in {n.name for n in graph.nodes}
    iface.n_tokens = 3
    assert forward.before_retrace() == {"_loop_policy": {"mode": "pack", "pack": 3}}


def test_callable_hooks_take_the_interface_first(iface):
    iface.methods = {"forward": Method(retain=lambda i, fn, graph: {"fn": fn},
                                       before_retrace=lambda i, fn: {"seen": i.n_tokens})}
    assert iface.spec.method("forward").retained(None) == {"fn": "forward"}
    assert iface.spec.method("forward").before_retrace() == {"seen": 8}


def test_an_audio_output_names_its_node_rate(iface):
    forward = iface.spec.method("forward")
    out = forward.output_nodes()
    assert len(out) == 1
    assert iface.sample_rate_for(out[0], fn="forward") == 16000
    assert iface.sample_rate_for(out[0]) == 16000            # any method
    assert iface.sample_rate_for("x", fn="forward") is None   # not an output
    iface.rate = 8000                                        # a Ref follows the attribute
    assert iface.sample_rate_for(out[0], fn="forward") == 8000


def test_default_inputs_may_be_computed(iface):
    assert iface.spec.method("forward").default_inputs() == {"x": "torch.randn(1, 4)"}
    iface.methods = {"forward": Method(inputs={
        "x": Input(lambda i: "torch.zeros(1, %d)" % i.n_tokens)})}
    assert iface.spec.method("forward").default_inputs() == {"x": "torch.zeros(1, 8)"}


# ── input modes ──────────────────────────────────────────────────────────────

def test_input_mode_description(iface):
    modes = iface.spec.method("forward").input_modes()
    assert modes == {"x": {
        "name": "x", "fn": "forward", "type": "text", "label": "prompt",
        "encode": "encode_text", "fills": ["x", "mask"],
        "placeholder": "say something", "default": "hello", "doc": ""}}
    iface.prompt = "changed"        # the default is a Ref: read when asked
    assert iface.spec.method("forward").input_modes()["x"]["default"] == "changed"


def test_encoder_returning_a_sequence_follows_fills(iface):
    out = iface.spec.method("forward").encode("x", "one two three")
    assert set(out) == {"x", "mask"} and out["x"].shape == (1, 3)


def test_encoder_receives_a_list_when_several_values_are_posted(iface):
    """Several prompts reach the encoder as one list: padding them together is
    the only way the ids and the mask agree on a width."""
    out = iface.spec.method("forward").encode("x", ["one", "one two three four"])
    assert out["x"].shape == (2, 4)


@pytest.mark.parametrize("produced, expected", [
    (lambda i, v: {"x": 1}, None),
    (lambda i, v: 1, "single value"),
    (lambda i, v: (1, 2, 3), "3 values"),
    (lambda i, v: {"x": 1, "other": 2}, "other"),
])
def test_encoder_shapes(iface, produced, expected):
    iface.methods = {"forward": Method(inputs={"x": Input(mode=InputMode(
        Text(), encode=produced, also_fills=["mask"]))})}
    method = iface.spec.method("forward")
    if expected is None:
        assert method.encode("x", "a") == {"x": 1}
    else:
        with pytest.raises(SpecError) as exc:
            method.encode("x", "a")
        assert expected in str(exc.value)


def test_a_single_value_fills_a_mode_that_fills_only_itself(iface):
    iface.methods = {"forward": Method(inputs={"x": Input(mode=InputMode(
        Text(), encode=lambda i, v: v.upper()))})}
    assert iface.spec.method("forward").encode("x", "a") == {"x": "A"}


def test_no_mode_is_refused(iface):
    with pytest.raises(SpecError):
        iface.spec.method("decode").encode("x", "a")


# ── options ──────────────────────────────────────────────────────────────────

def test_options_round_trip(iface):
    spec = iface.spec
    assert spec.set_option("n_tokens", "12") == 12 and iface.n_tokens == 12
    assert spec.set_option("side", "right") == "right" and iface._side == "right"
    with pytest.raises(SpecError):
        spec.set_option("side", "sideways")
    with pytest.raises(SpecError):
        spec.set_option("nope", 1)


@pytest.mark.parametrize("value, raw, expected", [
    (Int(), "12", 12), (Int(), "3.0", 3), (Float(), "0.5", 0.5),
    (Bool(), "true", True), (Bool(), "0", False), (Bool(), True, True), (Str(), 4, "4"),
])
def test_values_coerce_what_a_ui_posts(value, raw, expected):
    assert value.coerce(raw) == expected


def test_set_returns_what_the_interface_actually_holds(iface):
    iface.set_side = lambda value: None        # a setter that refuses
    assert iface.spec.set_option("side", "right") == "left"


def test_when_hides_an_option_that_does_not_apply(iface):
    assert "loop" in iface.spec.options
    iface.full = False
    assert "loop" not in iface.spec.options
    with pytest.raises(SpecError):
        iface.spec.set_option("loop", 3)


def test_option_description_carries_the_current_value(iface):
    rows = {r["name"]: r for r in iface.spec.describe_options()}
    assert rows["n_tokens"] == {"type": "int", "range": [1, 64], "name": "n_tokens",
                                "label": "n_tokens", "doc": "", "needs": "retrace", "value": 8}
    assert rows["side"]["choices"] == ["left", "right"] and rows["side"]["label"] == "padding side"


# ── callbacks ────────────────────────────────────────────────────────────────

def test_callback_description(iface):
    (generate,) = iface.spec.describe_callbacks()
    assert generate["returns"] == "text"
    assert generate["doc"] == "Continue prompt for a while."      # RST unwrapped
    args = {a["name"]: a for a in generate["args"]}
    # `seed` is accepted by the method but not declared, so not offered
    assert set(args) == {"prompt", "n_tokens"}
    # an argument with no declared default takes the signature's
    assert args["prompt"]["default"] == "hi" and args["n_tokens"]["default"] == 12
    assert args["n_tokens"]["range"] == [1, 256]


def test_run_callback(iface):
    assert iface.spec.run_callback("generate", prompt="x", n_tokens=3) == "x (3)"
    with pytest.raises(SpecError):
        iface.spec.run_callback("generate", seed=1)          # not declared
    with pytest.raises(SpecError):
        iface.spec.run_callback("render")                    # no such callback


# ── tokens ───────────────────────────────────────────────────────────────────

def test_tokens(iface):
    tokens = iface.spec.tokens
    assert tokens.decode(torch.zeros(1, 3)) == ["one string"]   # a str becomes a row
    assert tokens.has_logits and tokens.logits(torch.zeros(2, 3)).shape == (2, 3, 10)
    assert tokens.eos == 7                                      # from the tokenizer
    iface.tokens = None
    assert iface.spec.tokens is None


# ── everything, for any frontend ─────────────────────────────────────────────

def test_describe_is_plain_json(iface):
    described = iface.spec.describe()
    json.dumps(described)
    assert described["methods"]["forward"]["batch"] is False
    assert described["methods"]["forward"]["outputs"][0] == {"type": "audio", "sample_rate": 16000.0}
    assert described["tokens"] == {"logits": True, "eos": 7}


def test_ui_hints_reach_only_their_frontend(iface):
    iface.options = {"n_tokens": Option(Int(), attr="n_tokens",
                                        ui={"graph_viewer": {"slider": True}})}
    (row,) = iface.spec.describe_options("graph_viewer")
    assert row["ui"] == {"slider": True}
    (row,) = iface.spec.describe_options("max")
    assert "ui" not in row


# ── the graph viewer's side ──────────────────────────────────────────────────

class _Midi(Value):
    kind = "midi"
    media = True


@pytest.fixture
def viewer(iface, monkeypatch):
    from torchbend.ui.graph_viewer import views
    monkeypatch.setattr(views, "get_interface", lambda: iface)
    return views


def test_viewer_reads_batch_and_devices_from_the_spec(viewer):
    assert viewer._batch_supported("forward") is False
    assert viewer._batch_supported("decode") is True
    assert "not expected to work" in viewer._device_refusal("forward", "mps")
    assert viewer._device_refusal("forward", "cpu") is None
    assert viewer._device_refusal("decode", "mps") is None


def test_viewer_offers_only_the_input_modes_it_can_collect(iface, viewer):
    from torchbend.ui.graph_viewer import spec_adapters
    iface.methods = {"forward": Method(inputs={
        "x": Input(mode=InputMode(Text(), encode="encode_text", also_fills=["mask"])),
        "notes": Input(mode=InputMode(_Midi(), encode=lambda i, v: v))})}
    assert set(viewer._declared_input_modes("forward")) == {"x"}

    # a frontend supports a new value type by registering how to read it
    spec_adapters.register_reader(_Midi)(lambda request, name: request.POST.getlist(name))
    try:
        assert set(viewer._declared_input_modes("forward")) == {"x", "notes"}
    finally:
        spec_adapters._READERS.pop(_Midi)


def test_a_subclass_is_read_as_its_parent_until_it_registers():
    from torchbend.ui.graph_viewer import spec_adapters

    class _Prose(Text):
        pass

    assert spec_adapters.reader_for(_Prose()) is spec_adapters.reader_for(Text())
    assert spec_adapters.reader_for(Image()) is None
    assert spec_adapters.reader_for(Tensor()) is None


# ── on_inputs ────────────────────────────────────────────────────────────────
# A callback receives only its own declared arguments, so without this hook an
# interface cannot act on *what the graph is currently running on*. BLIP
# captions the image sitting in the input bench; it learns which one from here.

def test_on_inputs_defaults_to_doing_nothing(iface):
    assert iface.on_inputs("forward", {"x": 1}) is None


def test_a_failing_on_inputs_does_not_break_the_run(iface, viewer):
    def rude(fn, kwargs):
        raise RuntimeError("interfaces may misbehave")
    iface.on_inputs = rude
    viewer._notify_interface_of_inputs("forward", {"x": 1})   # must not raise
