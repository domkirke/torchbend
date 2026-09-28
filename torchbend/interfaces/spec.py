"""What an interface declares about itself, in one place.

An interface knows things about its model that no inspection recovers: that
GPT-2's ``input_ids`` is really a prompt, that XTTS's decoder fails on MPS, that
its ``forward`` speaks one utterance at a time, which nodes are the joints worth
keeping. Every one of those is declared here, with the classes below, on four
attributes of the :class:`~torchbend.interfaces.base.Interface` subclass::

    from torchbend.interfaces.spec import (Method, Input, InputMode, Option,
                                           Callback, Tokens, Ref, Text, Audio,
                                           Int, Float, Choice, retain_aliases)

    class BendedSomething(Interface):
        methods = {                        # the traced graphs
            "forward": Method(
                inputs={"input_ids": Input(
                    default="torch.randint(0, 100, (1, 8))",
                    mode=InputMode(Text(default="hello"), encode="encode_text",
                                   also_fills=["attention_mask"], label="prompt"))},
                outputs=[Audio(sample_rate=Ref("sample_rate"))],
                batch=False,               # one input at a time
                devices={"mps": False},    # a backend it fails on
                retain=retain_aliases,     # activations worth keeping
            ),
        }
        options = {                        # settings that outlive a call
            "trace_tokens": Option(Int(range=(1, 64)), attr="trace_tokens",
                                   needs="retrace"),
        }
        callbacks = {                      # interface methods a UI can run
            "generate": Callback(args={"prompt": Text()}, returns=Text()),
        }
        tokens = Tokens(decode="decode", logits="logits")

Anything not declared has the permissive default: an undeclared method batches,
runs on every device, has no input modes and keeps nothing extra.

The declarations say what things *are*, never how a particular UI shows them.
Every consumer -- a script, the graph viewer, any later frontend -- reads them
through ``iface.spec`` (an :class:`InterfaceSpec`), which validates them against
the interface and answers questions (``spec.method("forward").runs_on("mps")``)
or hands out a plain-JSON description (``spec.describe()``). A frontend decides
how to show a :class:`Value` by its class, in its own registry; ``ui=`` on any
declaration carries per-frontend hints (``ui={"graph_viewer": {...}}``) that
only that frontend reads.

Declarations are checked twice: their shape when the class is created (a range
that is not ``(min, max)`` fails at import), and the names they refer to (an
encoder, an attribute, a callback's arguments) the first time ``iface.spec`` is
used on an instance, since some of those only exist once ``__init__`` has run.

**Varying per instance.** An interface whose graphs are chosen at construction
(Bark picks a stage; XTTS may or may not hold its sampling loop) assigns
``self.methods = {...}`` in ``__init__``, or uses ``when=`` on an option. Any
field documented as *dynamic* also accepts a :class:`Ref` or a callable taking
the interface, read each time the spec is resolved rather than once.
"""

import inspect
import re
from collections import OrderedDict


__all__ = [
    "BendingInterfaceException", "SpecError",
    "Ref", "Value", "Int", "Float", "Bool", "Str", "Choice",
    "Text", "Audio", "Image", "Tensor",
    "InputMode", "Input", "Method", "Option", "Callback", "Tokens",
    "retain_aliases", "InterfaceSpec", "BoundMethod",
]


class BendingInterfaceException(Exception):
    pass


class SpecError(BendingInterfaceException):
    """A declaration that is malformed, or names something that is not there."""


class _Missing:
    def __repr__(self):
        return "MISSING"

    def __bool__(self):
        return False


MISSING = _Missing()


# ── dynamic values ────────────────────────────────────────────────────────────

class Ref:
    """An attribute of the interface, read when the spec is resolved.

    For a value that follows the interface's state -- a default that is an
    option's current value, a sample rate only known once the model is loaded::

        InputMode(Audio(default=Ref("reference")), encode="encode_reference")
        Audio(sample_rate=Ref("sample_rate"))

    A callable taking the interface does the same for anything computed.
    """

    __slots__ = ("attr",)

    def __init__(self, attr: str):
        if not isinstance(attr, str) or not attr:
            raise SpecError("Ref takes an attribute name, got %r" % (attr,))
        self.attr = attr

    def __call__(self, iface):
        value = getattr(iface, self.attr)
        return value() if callable(value) else value

    def __repr__(self):
        return "Ref(%r)" % self.attr


def _dynamic(value, iface):
    """Resolve a dynamic field: a Ref or a callable is asked, anything else kept."""
    if isinstance(value, Ref):
        return value(iface)
    if callable(value) and not isinstance(value, type):
        return value(iface)
    return value


def _owner_name(owner):
    return getattr(owner, "__name__", type(owner).__name__)


def _hook(iface, hook, what):
    """A hook given as a method name, as a bound callable."""
    if isinstance(hook, str):
        fn = getattr(iface, hook, None)
        if not callable(fn):
            raise SpecError("%s names %r, which %s does not have"
                            % (what, hook, _owner_name(iface)))
        return fn
    return hook


class _Spec:
    """Shared repr: the fields that differ from their default."""

    _fields = ()

    def __repr__(self):
        parts = []
        for name in self._fields:
            value = getattr(self, name)
            if value in (None, MISSING, (), [], {}, "") or value is False and name == "optional":
                continue
            parts.append("%s=%r" % (name, value))
        return "%s(%s)" % (type(self).__name__, ", ".join(parts))

    def _check_ui(self, where):
        if not isinstance(self.ui, dict) or not all(isinstance(v, dict) for v in self.ui.values()):
            raise SpecError("%s: ui must be {frontend: {hint: value}}, got %r" % (where, self.ui))


def _ui_for(ui, frontend):
    return dict((ui or {}).get(frontend) or {}) if frontend else {}


# ── values ────────────────────────────────────────────────────────────────────
#
# What an argument, an option, an input mode or an output *is*. The class is the
# type: a frontend picks a widget by it, and a new kind of value is a subclass
# (with its own `kind`) that a frontend then registers a widget for.

class Value(_Spec):
    """Base of every value type.

    ``default`` (dynamic), ``label`` and ``doc`` are for whoever presents it;
    ``optional`` says leaving it blank means "do not pass it" rather than
    "pass the default".
    """

    #: The type's name on the wire. Subclasses set it.
    kind = None
    #: Media are what a model consumes or produces (and a UI renders as such);
    #: the others are plain controls.
    media = False
    _fields = ("default", "label", "doc", "optional", "ui")

    def __init__(self, *, default=MISSING, label=None, doc="", optional=False, ui=None):
        self.default = default
        self.label = label
        self.doc = doc
        self.optional = bool(optional)
        self.ui = dict(ui or {})

    def check(self, where):
        """Structural validation; raises SpecError."""
        if self.kind is None:
            raise SpecError("%s: %s declares no `kind`" % (where, type(self).__name__))
        self._check_ui(where)

    def coerce(self, raw):
        """A value posted by a UI (often a string), as this type."""
        return raw

    def describe(self, iface=None, frontend=None) -> dict:
        out = {"type": self.kind}
        default = _dynamic(self.default, iface) if iface is not None else self.default
        if default is not MISSING:
            out["default"] = default
        if self.label:
            out["label"] = self.label
        if self.doc:
            out["doc"] = self.doc
        if self.optional:
            out["optional"] = True
        ui = _ui_for(self.ui, frontend)
        if ui:
            out["ui"] = ui
        return out


class _Ranged(Value):
    _fields = ("range", "step") + Value._fields

    def __init__(self, *, range=None, step=None, **common):
        super().__init__(**common)
        self.range = range
        self.step = step

    def check(self, where):
        super().check(where)
        if self.range is not None:
            try:
                lo, hi = tuple(self.range)
            except (TypeError, ValueError):
                raise SpecError("%s: range must be (min, max), got %r" % (where, self.range))
            if lo > hi:
                raise SpecError("%s: range %r has min above max" % (where, self.range))

    def describe(self, iface=None, frontend=None):
        out = super().describe(iface, frontend)
        if self.range is not None:
            out["range"] = [self.range[0], self.range[1]]
        if self.step is not None:
            out["step"] = self.step
        return out


class Int(_Ranged):
    kind = "int"

    def coerce(self, raw):
        return int(float(raw))


class Float(_Ranged):
    kind = "float"

    def coerce(self, raw):
        return float(raw)


class Bool(Value):
    kind = "bool"

    def coerce(self, raw):
        return raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")


class Str(Value):
    kind = "str"
    _fields = ("placeholder",) + Value._fields

    def __init__(self, *, placeholder="", **common):
        super().__init__(**common)
        self.placeholder = placeholder

    def coerce(self, raw):
        return str(raw)

    def describe(self, iface=None, frontend=None):
        out = super().describe(iface, frontend)
        if self.placeholder:
            out["placeholder"] = self.placeholder
        return out


class Choice(Value):
    """One of a fixed list: ``Choice(["left", "right"])``."""

    kind = "choice"
    _fields = ("choices",) + Value._fields

    def __init__(self, choices, **common):
        super().__init__(**common)
        self.choices = list(choices or [])

    def check(self, where):
        super().check(where)
        if not self.choices:
            raise SpecError("%s: a choice lists no choices" % where)

    def coerce(self, raw):
        value = str(raw)
        if value not in [str(c) for c in self.choices]:
            raise SpecError("%r is not one of %s" % (value, ", ".join(str(c) for c in self.choices)))
        # hand back the declared choice itself, which need not be a string
        return next(c for c in self.choices if str(c) == value)

    def describe(self, iface=None, frontend=None):
        out = super().describe(iface, frontend)
        out["choices"] = list(self.choices)
        return out


class Text(Str):
    """Prose -- a prompt, a caption -- rather than merely a string."""

    kind = "text"
    media = True


class Audio(Value):
    """A recording. ``sample_rate`` (dynamic) says at what rate, where known.

    As an input mode, the encoder receives what the user gave: a string (a path,
    typically) or, for a loaded file, ``(waveform [C, L] float tensor, rate)``.
    As an output, it names the rate the graph's output node is audio at -- which
    is also what makes a UI open it as audio.
    """

    kind = "audio"
    media = True
    _fields = ("sample_rate", "placeholder") + Value._fields

    def __init__(self, *, sample_rate=None, placeholder="", **common):
        super().__init__(**common)
        self.sample_rate = sample_rate
        self.placeholder = placeholder

    def rate(self, iface):
        rate = _dynamic(self.sample_rate, iface)
        return float(rate) if rate else None

    def describe(self, iface=None, frontend=None):
        out = super().describe(iface, frontend)
        if self.placeholder:
            out["placeholder"] = self.placeholder
        if iface is not None and self.sample_rate is not None:
            try:
                out["sample_rate"] = self.rate(iface)
            except Exception:
                pass
        return out


class Image(Value):
    kind = "image"
    media = True


class Tensor(Value):
    """A tensor with no more specific meaning (token ids a UI may decode, say)."""

    kind = "tensor"
    media = True


def _as_value(value, where, media_only=False):
    """A Value instance from a Value or a Value class."""
    if isinstance(value, type) and issubclass(value, Value):
        value = value()
    if not isinstance(value, Value):
        raise SpecError("%s must be a Value (Int, Text, Audio, ...), got %r" % (where, value))
    if media_only and not value.media:
        raise SpecError("%s must be a medium (Text, Audio, Image, Tensor), got %s"
                        % (where, type(value).__name__))
    value.check(where)
    return value


# ── methods: the traced graphs ────────────────────────────────────────────────

class InputMode(_Spec):
    """Another way to fill a placeholder than an expression or a file.

    GPT-2's ``input_ids`` is really a prompt; turning prose into ids is the
    interface's job, not the bench's. ``value`` is what the user provides (a
    :class:`Text`, an :class:`Audio`, ...), ``encode`` the interface method (or
    a callable ``f(iface, value)``) that turns it into tensors, and
    ``also_fills`` the other placeholders the same call decides -- one prompt
    fixes the sequence length, so the mask made from it has to come along.

    The encoder returns a tensor (for the placeholder alone), a sequence in the
    order ``[placeholder, *also_fills]``, or a ``{placeholder: tensor}`` dict.
    """

    _fields = ("value", "encode", "also_fills", "label", "doc", "ui")

    def __init__(self, value, encode, *, also_fills=(), label=None, doc="", ui=None):
        self.value = value
        self.encode = encode
        self.also_fills = list(also_fills or ())
        self.label = label
        self.doc = doc
        self.ui = dict(ui or {})

    @property
    def kind(self):
        return self.value.kind

    def check(self, where):
        self.value = _as_value(self.value, where + ".value", media_only=True)
        if not (isinstance(self.encode, str) or callable(self.encode)):
            raise SpecError("%s: encode must be a method name or a callable" % where)
        self._check_ui(where)

    def fills(self, name):
        return [name] + [n for n in self.also_fills if n != name]

    def bind(self, iface, where):
        if isinstance(self.encode, str):
            _hook(iface, self.encode, where + ".encode")

    def run(self, iface, name, raw) -> dict:
        """Encode ``raw``, returning ``{placeholder: tensor}``."""
        encoder = self.encode
        if isinstance(encoder, str):
            produced = getattr(iface, encoder)(raw)
            encoder_name = encoder
        else:
            produced = encoder(iface, raw)
            encoder_name = getattr(encoder, "__name__", "encoder")
        fills = self.fills(name)
        if isinstance(produced, dict):
            unexpected = set(produced) - set(fills)
            if unexpected:
                raise SpecError("encoder %s produced %s, which the mode does not fill"
                                % (encoder_name, ", ".join(sorted(unexpected))))
            return dict(produced)
        if isinstance(produced, (tuple, list)):
            if len(produced) != len(fills):
                raise SpecError("encoder %s returned %d values for %d filled inputs (%s)"
                                % (encoder_name, len(produced), len(fills), ", ".join(fills)))
            return dict(zip(fills, produced))
        if len(fills) != 1:
            raise SpecError("encoder %s returned a single value, but the mode fills %s"
                            % (encoder_name, ", ".join(fills)))
        return {name: produced}

    def describe(self, iface, fn, name, frontend=None) -> dict:
        value = self.value.describe(iface, frontend)
        out = {
            "name": name,
            "fn": fn,
            "type": self.kind,
            "label": self.label or self.value.label or self.kind,
            "encode": self.encode if isinstance(self.encode, str)
                      else getattr(self.encode, "__name__", "encoder"),
            "fills": self.fills(name),
            "placeholder": value.get("placeholder", ""),
            "default": value.get("default", ""),
            "doc": self.doc or self.value.doc,
        }
        ui = {**value.get("ui", {}), **_ui_for(self.ui, frontend)}
        if ui:
            out["ui"] = ui
        return out


class Input(_Spec):
    """One placeholder of a traced method.

    ``default`` (dynamic) is the expression the bench starts from -- a string
    such as ``"torch.randn(1, 80, 300)"`` -- so the graph opens on something
    runnable. ``mode`` is an optional :class:`InputMode`.
    """

    _fields = ("default", "mode", "label", "doc", "ui")

    def __init__(self, default=None, *, mode=None, label=None, doc="", ui=None):
        self.default = default
        self.mode = mode
        self.label = label
        self.doc = doc
        self.ui = dict(ui or {})

    def check(self, where):
        if self.mode is not None:
            if not isinstance(self.mode, InputMode):
                raise SpecError("%s.mode must be an InputMode, got %r" % (where, self.mode))
            self.mode.check(where + ".mode")
        self._check_ui(where)


def retain_aliases(iface, fn, graph) -> dict:
    """``Method(retain=retain_aliases)``: keep the activations the model's
    ``mark()`` aliases tag -- usually exactly the joints between its stages."""
    joints = {}
    for label, nodes in iface.model.aliases(fn=fn).items():
        if len(nodes) == 1:
            joints[label] = nodes[0]
        else:
            joints.update({"%s[%d]" % (label, i): n for i, n in enumerate(nodes)})
    return joints


class Method(_Spec):
    """Everything about one traced method (``forward``, ``decode``, ...).

    inputs
        ``{placeholder: Input}``. Placeholders left out still work; they only
        get no default and no mode.
    outputs
        Values for the graph's outputs, in order (``None`` to skip one). An
        :class:`Audio` with a ``sample_rate`` is what makes that output audio.
    batch
        ``False`` when the method takes one input at a time, so a UI does not
        offer to stack several (running them one after another stays fine).
    devices
        ``{device type: False}`` for backends it is known to fail on --
        ``"mps"``, ``"cuda"``; a backend-level fact, not a per-GPU one.
    retain
        The activations worth keeping, as ``{label: node}``: a method name
        ``(fn, graph) -> dict``, or a callable ``(iface, fn, graph) -> dict``
        such as :func:`retain_aliases`. A UI keeps these whenever it computes
        them anyway, so looking at the output does not throw away what led
        there, and a bending downstream resumes from them.
    before_retrace
        Called just before a UI retraces the method: a method name ``(fn) ->
        dict`` or a callable ``(iface, fn) -> dict``. It applies options that
        only take effect at trace time, and returns tracer settings
        (``_loop_policy``, ...) that replace the recorded ones.
    """

    _fields = ("inputs", "outputs", "batch", "devices", "retain", "before_retrace",
               "label", "doc", "ui")

    def __init__(self, *, inputs=None, outputs=None, batch=True, devices=None,
                 retain=None, before_retrace=None, label=None, doc="", ui=None):
        self.inputs = OrderedDict(inputs or {})
        self.outputs = list(outputs or [])
        self.batch = bool(batch)
        self.devices = dict(devices or {})
        self.retain = retain
        self.before_retrace = before_retrace
        self.label = label
        self.doc = doc
        self.ui = dict(ui or {})

    def check(self, where):
        for name, spec in self.inputs.items():
            if not isinstance(spec, Input):
                raise SpecError("%s.inputs[%r] must be an Input, got %r" % (where, name, spec))
            spec.check("%s.inputs[%r]" % (where, name))
        self.outputs = [None if v is None else _as_value(v, "%s.outputs[%d]" % (where, i))
                        for i, v in enumerate(self.outputs)]
        for dev, ok in self.devices.items():
            if not isinstance(dev, str) or not isinstance(ok, bool):
                raise SpecError("%s.devices must be {device type: bool}, got %r"
                                % (where, self.devices))
        for hook in ("retain", "before_retrace"):
            value = getattr(self, hook)
            if value is not None and not (isinstance(value, str) or callable(value)):
                raise SpecError("%s.%s must be a method name or a callable" % (where, hook))
        self._check_ui(where)


# ── interface-level declarations ─────────────────────────────────────────────

class Option(_Spec):
    """A setting that outlives any one call.

    ``value`` is its type (``Int(range=(1, 64))``, ``Choice([...])``, ...). It is
    read and written either as an attribute (``attr=``) or through accessor
    methods (``get=``, ``set=``) -- a setter may clamp or refuse, and the value
    read back afterwards is what a UI shows. ``needs`` is what changing it
    invalidates (``"retrace"``, ``"reload"``; ``None`` takes effect at once).
    ``when`` hides it when it does not apply: an attribute name or a callable
    taking the interface, e.g. ``when=lambda iface: iface.full``.
    """

    EFFECTS = ("retrace", "reload")
    _fields = ("value", "attr", "get", "set", "needs", "when", "label", "doc", "ui")

    def __init__(self, value, *, attr=None, get=None, set=None, needs=None, when=None,
                 label=None, doc=None, ui=None):
        self.value = value
        self.attr = attr
        self.get = get
        self.set = set
        self.needs = needs
        self.when = when
        self.label = label
        self.doc = doc
        self.ui = dict(ui or {})

    def check(self, where):
        self.value = _as_value(self.value, where + ".value")
        if self.attr and (self.get or self.set):
            raise SpecError("%s gives both an attribute and accessors; pick one" % where)
        if not self.attr and not (self.get and self.set):
            raise SpecError("%s needs either `attr`, or both `get` and `set`" % where)
        if self.needs is not None and self.needs not in self.EFFECTS:
            raise SpecError("%s needs %r (expected one of %s)"
                            % (where, self.needs, ", ".join(self.EFFECTS)))
        self._check_ui(where)

    def bind(self, iface, where):
        if self.attr:
            if not hasattr(iface, self.attr):
                raise SpecError("%s reads attribute %r, which %s does not have"
                                % (where, self.attr, _owner_name(iface)))
        else:
            _hook(iface, self.get, where + ".get")
            _hook(iface, self.set, where + ".set")

    def applies(self, iface) -> bool:
        if self.when is None:
            return True
        if isinstance(self.when, str):
            return bool(getattr(iface, self.when))
        return bool(self.when(iface))

    def read(self, iface):
        return getattr(iface, self.attr) if self.attr else _hook(iface, self.get, "get")()

    def write(self, iface, raw):
        """Set from a posted value; returns what the interface then holds."""
        try:
            value = self.value.coerce(raw)
        except SpecError:
            raise
        except (TypeError, ValueError) as exc:
            raise SpecError("%r is not a valid %s: %s" % (raw, self.value.kind, exc))
        if self.attr:
            setattr(iface, self.attr, value)
        else:
            _hook(iface, self.set, "set")(value)
        return self.read(iface)

    def describe(self, iface, name, frontend=None) -> dict:
        out = self.value.describe(iface, frontend)
        out.pop("default", None)
        out.update({"name": name, "label": self.label or self.value.label or name,
                    "doc": self.doc if self.doc is not None else self.value.doc,
                    "needs": self.needs})
        ui = {**out.get("ui", {}), **_ui_for(self.ui, frontend)}
        if ui:
            out["ui"] = ui
        return out


def _first_doc_line(fn):
    """The method's summary line, with RST literal markup unwrapped for display."""
    line = (fn.__doc__ or "").strip().split("\n")[0]
    return re.sub(r"``([^`]+)``", r"\1", line)


class Callback(_Spec):
    """An interface method a UI can run -- a whole operation around the graph
    (GPT-2's generate, a TTS's speak) that cannot be traced itself.

    ``args`` are the arguments offered, ``{name: Value}``; anything the
    signature accepts and ``args`` leaves out is not offered, and keeps the
    method's own default. A value with no ``default`` takes the signature's.
    ``returns`` is the medium of the result.
    """

    _fields = ("args", "returns", "label", "doc", "ui")

    def __init__(self, *, args=None, returns=None, label=None, doc=None, ui=None):
        self.args = OrderedDict(args or {})
        self.returns = returns
        self.label = label
        self.doc = doc
        self.ui = dict(ui or {})

    def check(self, where):
        self.args = OrderedDict(
            (name, _as_value(v, "%s.args[%r]" % (where, name))) for name, v in self.args.items())
        if self.returns is not None:
            self.returns = _as_value(self.returns, where + ".returns", media_only=True)
        self._check_ui(where)

    def bind(self, iface, name, where):
        fn = _hook(iface, name, where)
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return
        for arg in self.args:
            if arg not in signature.parameters:
                raise SpecError("%s declares argument %r, which %s does not accept"
                                % (where, arg, name))

    def _signature(self, iface, name):
        try:
            return inspect.signature(getattr(iface, name)).parameters
        except (TypeError, ValueError):
            return {}

    def describe(self, iface, name, frontend=None) -> dict:
        params = self._signature(iface, name)
        args = []
        for arg, value in self.args.items():
            row = value.describe(iface, frontend)
            param = params.get(arg)
            if "default" not in row and param is not None and param.default is not param.empty:
                row["default"] = param.default
            row["name"] = arg
            args.append(row)
        out = {"name": name, "label": self.label or name,
               "doc": self.doc if self.doc is not None else _first_doc_line(getattr(iface, name)),
               "returns": self.returns.kind if self.returns is not None else None,
               "args": args}
        ui = _ui_for(self.ui, frontend)
        if ui:
            out["ui"] = ui
        return out


class Tokens(_Spec):
    """How token ids read as text: what lets a UI show ids and logits as words.

    ``decode`` turns ``[B, T]`` ids into strings (``(ids, skip_special_tokens)``);
    ``logits`` takes ``[B, T]`` ids to ``[B, T, V]`` logits, which lets a UI
    continue a sequence after an edit; ``eos`` is the end-of-sequence id
    (dynamic; by default the interface's ``tokenizer.eos_token_id``). Method
    names, or callables taking the interface first.
    """

    _fields = ("decode", "logits", "eos")

    def __init__(self, decode, *, logits=None, eos=None):
        self.decode = decode
        self.logits = logits
        self.eos = eos

    def check(self, where):
        for field in ("decode", "logits"):
            value = getattr(self, field)
            if value is not None and not (isinstance(value, str) or callable(value)):
                raise SpecError("%s.%s must be a method name or a callable" % (where, field))
        if self.decode is None:
            raise SpecError("%s names no decoder" % where)


# ── class-time validation ────────────────────────────────────────────────────

#: What the declarations replaced. Declaring one of these is an error, not a
#: silent no-op: the new form is one line away, and the old one does nothing.
_RETIRED = {
    "_input_modes_": "methods = {fn: Method(inputs={name: Input(mode=InputMode(...))})}",
    "_options_": "options = {name: Option(...)}",
    "_callbacks_": "callbacks = {name: Callback(...)}",
    "_device_compat_": "methods = {fn: Method(devices={...})}",
    "_batch_compat_": "methods = {fn: Method(batch=False)}",
    "_token_decoder_": "tokens = Tokens(decode=..., logits=...)",
    "_token_logits_": "tokens = Tokens(decode=..., logits=...)",
    "_default_inputs": "methods = {fn: Method(inputs={name: Input(default=...)})}",
}


def check_declarations(owner, methods, options, callbacks, tokens):
    """Validate the four declarations' shape. ``owner`` names them in errors."""
    where = _owner_name(owner)
    for attr, table, kind in (("methods", methods, Method), ("options", options, Option),
                              ("callbacks", callbacks, Callback)):
        if not isinstance(table, dict):
            raise SpecError("%s.%s must be a dict of {name: %s}, got %r"
                            % (where, attr, kind.__name__, type(table).__name__))
        for name, spec in table.items():
            if not isinstance(spec, kind):
                raise SpecError("%s.%s[%r] must be a %s, got %r"
                                % (where, attr, name, kind.__name__, spec))
            spec.check("%s.%s[%r]" % (where, attr, name))
    if tokens is not None:
        if not isinstance(tokens, Tokens):
            raise SpecError("%s.tokens must be a Tokens, got %r" % (where, tokens))
        tokens.check(where + ".tokens")


def check_class(cls):
    """Called for every Interface subclass as it is created."""
    for attr, replacement in _RETIRED.items():
        if attr in cls.__dict__:
            raise SpecError("%s declares %s, which is retired: use %s (see torchbend.interfaces.spec)"
                            % (cls.__name__, attr, replacement))
    check_declarations(cls, cls.__dict__.get("methods", {}), cls.__dict__.get("options", {}),
                       cls.__dict__.get("callbacks", {}), cls.__dict__.get("tokens"))


# ── resolution against an instance ───────────────────────────────────────────

class BoundMethod:
    """One method's declaration, answered for one interface."""

    def __init__(self, iface, fn, method):
        self.iface = iface
        self.fn = fn
        self.method = method
        self._outputs_cache = (None, [])

    @property
    def declared(self) -> bool:
        return self.fn in self.iface.methods

    @property
    def batch(self) -> bool:
        return self.method.batch

    @property
    def inputs(self):
        return self.method.inputs

    def runs_on(self, device) -> bool:
        """Whether the method is expected to work on ``device`` (default: yes)."""
        import torch
        return bool(self.method.devices.get(torch.device(str(device)).type, True))

    def devices(self, candidates) -> dict:
        return {d: self.runs_on(d) for d in candidates}

    def default_inputs(self) -> dict:
        out = OrderedDict()
        for name, spec in self.method.inputs.items():
            value = _dynamic(spec.default, self.iface)
            if value is not None:
                out[name] = value
        return out

    def modes(self) -> dict:
        return OrderedDict((n, s.mode) for n, s in self.method.inputs.items() if s.mode)

    def input_modes(self, frontend=None) -> dict:
        """``{placeholder: description}`` of the placeholders with an input mode."""
        return OrderedDict((n, m.describe(self.iface, self.fn, n, frontend))
                           for n, m in self.modes().items())

    def encode(self, name, raw) -> dict:
        """Run ``name``'s input mode on what the user gave: ``{placeholder: tensor}``."""
        mode = self.modes().get(name)
        if mode is None:
            raise SpecError("no input mode for %s.%s" % (self.fn, name))
        return mode.run(self.iface, name, raw)

    def retained(self, graph) -> dict:
        """``{label: node}`` of the activations worth keeping."""
        hook = self.method.retain
        if hook is None:
            return {}
        if isinstance(hook, str):
            return dict(_hook(self.iface, hook, "retain")(self.fn, graph) or {})
        return dict(hook(self.iface, self.fn, graph) or {})

    def before_retrace(self) -> dict:
        hook = self.method.before_retrace
        if hook is None:
            return {}
        if isinstance(hook, str):
            return dict(_hook(self.iface, hook, "before_retrace")(self.fn) or {})
        return dict(hook(self.iface, self.fn) or {})

    def output_nodes(self) -> list:
        """The names of the nodes feeding the traced graph's output, in order."""
        try:
            graph = self.iface.model.graph(fn=self.fn)
        except Exception:
            return []
        key, nodes = self._outputs_cache
        if key is graph:
            return nodes
        out = next((n for n in graph.nodes if n.op == "output"), None)
        nodes = [n.name for n in out.all_input_nodes] if out is not None else []
        self._outputs_cache = (graph, nodes)
        return nodes

    def output_rate(self, node):
        """The declared audio rate of ``node`` if it is one of the outputs."""
        audio = [(i, v) for i, v in enumerate(self.method.outputs)
                 if isinstance(v, Audio) and v.sample_rate is not None]
        if not audio:
            return None
        nodes = self.output_nodes()
        for i, value in audio:
            if i < len(nodes) and nodes[i] == node:
                return value.rate(self.iface)
        return None

    def describe(self, frontend=None) -> dict:
        out = {"name": self.fn, "label": self.method.label or self.fn,
               "doc": self.method.doc, "batch": self.batch,
               "devices": dict(self.method.devices),
               "inputs": OrderedDict(
                   (n, {"default": _dynamic(s.default, self.iface), "label": s.label or n,
                        "doc": s.doc}) for n, s in self.method.inputs.items()),
               "input_modes": self.input_modes(frontend),
               "outputs": [None if v is None else v.describe(self.iface, frontend)
                           for v in self.method.outputs]}
        ui = _ui_for(self.method.ui, frontend)
        if ui:
            out["ui"] = ui
        return out


class BoundTokens:
    def __init__(self, iface, tokens):
        self.iface = iface
        self.tokens = tokens

    def _call(self, fn, *args, **kwargs):
        if isinstance(fn, str):
            return getattr(self.iface, fn)(*args, **kwargs)
        return fn(self.iface, *args, **kwargs)

    def decode(self, ids, skip_special_tokens: bool = True) -> list:
        """Decode ``[B, T]`` (or ``[T]``) ids into one string per row."""
        out = self._call(self.tokens.decode, ids, skip_special_tokens=skip_special_tokens)
        return [out] if isinstance(out, str) else list(out)

    @property
    def has_logits(self) -> bool:
        return self.tokens.logits is not None

    def logits(self, ids):
        """Logits ``[B, T, V]`` for a batch of ids ``[B, T]``."""
        if self.tokens.logits is None:
            raise SpecError("%s declares no token logits" % _owner_name(self.iface))
        return self._call(self.tokens.logits, ids)

    @property
    def eos(self):
        if self.tokens.eos is not None:
            return _dynamic(self.tokens.eos, self.iface)
        return getattr(getattr(self.iface, "tokenizer", None), "eos_token_id", None)


class InterfaceSpec:
    """An interface's declarations, validated against it and ready to query.

    Obtained as ``iface.spec``. Everything a script or a UI asks an interface
    about itself goes through here.
    """

    def __init__(self, iface):
        self.iface = iface
        self._bound = False
        self._methods = {}

    # the declarations, as the instance currently holds them
    def _declared(self):
        i = self.iface
        return i.methods, i.options, i.callbacks, i.tokens

    def validate(self):
        """Check every name the declarations refer to exists. Idempotent."""
        if self._bound:
            return self
        iface = self.iface
        methods, options, callbacks, tokens = self._declared()
        # instance-level declarations have not been through class-time checks
        check_declarations(iface, methods, options, callbacks, tokens)
        where = _owner_name(iface)
        for fn, method in methods.items():
            for name, spec in method.inputs.items():
                if spec.mode is not None:
                    spec.mode.bind(iface, "%s.methods[%r].inputs[%r].mode" % (where, fn, name))
            for hook in ("retain", "before_retrace"):
                value = getattr(method, hook)
                if isinstance(value, str):
                    _hook(iface, value, "%s.methods[%r].%s" % (where, fn, hook))
        for name, option in options.items():
            option.bind(iface, "%s.options[%r]" % (where, name))
        for name, callback in callbacks.items():
            callback.bind(iface, name, "%s.callbacks[%r]" % (where, name))
        if tokens is not None:
            for field in ("decode", "logits"):
                value = getattr(tokens, field)
                if isinstance(value, str):
                    _hook(iface, value, "%s.tokens.%s" % (where, field))
        self._bound = True
        return self

    # -- methods --

    def method(self, fn) -> BoundMethod:
        """``fn``'s declaration; an undeclared method gets the defaults."""
        self.validate()
        declared = self.iface.methods.get(fn)
        source, bound = self._methods.get(fn, (None, None))
        if bound is None or source is not declared:
            bound = BoundMethod(self.iface, fn, declared if declared is not None else Method())
            self._methods[fn] = (declared, bound)
        return bound

    @property
    def methods(self) -> dict:
        return OrderedDict((fn, self.method(fn)) for fn in self.iface.methods)

    # -- options --

    @property
    def options(self) -> dict:
        """The options that apply now, ``{name: Option}``."""
        self.validate()
        return OrderedDict((n, o) for n, o in self.iface.options.items() if o.applies(self.iface))

    def option(self, name) -> Option:
        option = self.options.get(name)
        if option is None:
            raise SpecError("no option %r on %s" % (name, _owner_name(self.iface)))
        return option

    def get_option(self, name):
        return self.option(name).read(self.iface)

    def set_option(self, name, value):
        """Set one option, returning the value that actually took effect."""
        return self.option(name).write(self.iface, value)

    def describe_options(self, frontend=None, values=True) -> list:
        rows = []
        for name, option in self.options.items():
            row = option.describe(self.iface, name, frontend)
            if values:
                try:
                    row["value"] = option.read(self.iface)
                except Exception as exc:
                    row["value"] = None
                    row["error"] = str(exc)
            rows.append(row)
        return rows

    # -- callbacks --

    @property
    def callbacks(self) -> dict:
        self.validate()
        return OrderedDict(self.iface.callbacks)

    def callback(self, name) -> Callback:
        callback = self.callbacks.get(name)
        if callback is None:
            raise SpecError("no callback %r on %s" % (name, _owner_name(self.iface)))
        return callback

    def run_callback(self, name, **kwargs):
        """Run a declared callback, passing only arguments it declares.

        Undeclared arguments are refused rather than forwarded: the declaration
        is what a UI was built from, so anything outside it did not come from
        one of its widgets.
        """
        callback = self.callback(name)
        unknown = set(kwargs) - set(callback.args)
        if unknown:
            raise SpecError("callback %s does not declare argument(s) %s"
                            % (name, ", ".join(sorted(unknown))))
        return getattr(self.iface, name)(**kwargs)

    def describe_callbacks(self, frontend=None) -> list:
        return [c.describe(self.iface, n, frontend) for n, c in self.callbacks.items()]

    # -- tokens --

    @property
    def tokens(self):
        """A :class:`BoundTokens`, or None when the interface declares none."""
        self.validate()
        tokens = self.iface.tokens
        return BoundTokens(self.iface, tokens) if tokens is not None else None

    # -- everything --

    def describe(self, frontend=None) -> dict:
        """Every declaration as plain JSON-able data, for any frontend.

        ``frontend`` names the one asking, whose ``ui=`` hints are included.
        """
        tokens = self.tokens
        return {
            "interface": _owner_name(self.iface),
            "methods": OrderedDict((fn, m.describe(frontend)) for fn, m in self.methods.items()),
            "options": self.describe_options(frontend),
            "callbacks": self.describe_callbacks(frontend),
            "tokens": None if tokens is None else {"logits": tokens.has_logits,
                                                   "eos": tokens.eos},
        }
