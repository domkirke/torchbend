"""Which nodes are audio, and at what rate.

Declared by the model -- an :class:`~torchbend.interfaces.base.Interface` or a
:class:`~torchbend.tracing.module.BendedModule` -- and read by anything that
plays a tensor back. Lives on its own so both can use it: the interfaces import
the tracing package, so the shared piece cannot live in either.
"""
import inspect
from collections import OrderedDict


# The model is the only thing that knows. A codec's latent runs at one frame per
# hop, a mel at one per STFT hop, a waveform at the model's own rate -- and
# nothing about a tensor's shape tells the viewer which of those it is looking
# at. Working it out from the ratio of tensor lengths is right for a uniformly
# strided convnet and confident nonsense everywhere else, so the model says.
#
#     _sample_rates_ = 44100                          # every node, every method
#     _sample_rates_ = {"audio": 44100, "z": 21.53}   # per node, any method
#     _sample_rates_ = {"decode": {"audio": 44100}}   # per method, then per node
#     _sample_rates_ = "rate_of"                      # asked, not recorded
#
# A value is a number, a callable, or the name of a method on the owner. The
# callable form is for a rate that is not known at construction -- one that
# follows an option the user can move, or a node whose stride depends on the
# config. It is called as ``f(node)``, and additionally given ``fn`` and
# ``shape`` if it declares them; returning ``None`` means "no opinion", and the
# next, less specific declaration is asked instead.
#
# ``"*"`` as a node name is the catch-all for a method (or, at the top level,
# for the whole model).


class SampleRateError(Exception):
    pass


def _normalize_rate(owner, fn, node, spec):
    """One declared rate: a positive number, or something to call for one."""
    where = "%s.%s" % (fn or "*", node)
    owner_name = getattr(owner, "__name__", type(owner).__name__)
    if isinstance(spec, str):
        if not callable(getattr(owner, spec, None)):
            raise SampleRateError(
                "sample rate %s names %r, which %s does not have"
                % (where, spec, owner_name))
        return spec
    if callable(spec):
        # A callback is called as ``f(node)``, so it has to be able to take one
        # positional argument. The usual slip is a nested function written as
        # though it were a method -- ``def rate(self, node, ...)`` -- which then
        # binds the node name to `self` and fails for a missing `node`, at read
        # time, deep inside a viewer request. Refuse it here instead.
        try:
            inspect.signature(spec).bind("node")
        except TypeError as exc:
            raise SampleRateError(
                "sample rate %s names a callable that cannot be called as "
                "f(node): %s. A plain function passed here is not a method -- "
                "drop the leading `self` parameter, or pass the bound method."
                % (where, exc))
        except ValueError:
            pass                      # no readable signature; trust it
        return spec
    try:
        rate = float(spec)
    except (TypeError, ValueError):
        raise SampleRateError(
            "sample rate %s is %r, which is neither a number nor a callable"
            % (where, spec))
    if not rate > 0:
        raise SampleRateError("sample rate %s is %r, which is not positive"
                              % (where, spec))
    return rate


def normalize_sample_rates(owner, raw):
    """Validate ``_sample_rates_`` into ``{fn | None: {node | "*": spec}}``.

    ``None`` as the method means "whatever method is running". A dict value
    names a method and holds that method's nodes; anything else is a node's
    rate, good for every method.
    """
    out = OrderedDict()
    if raw is None or raw == {}:
        return out
    if not isinstance(raw, dict):
        out[None] = OrderedDict({"*": _normalize_rate(owner, None, "*", raw)})
        return out
    for key, value in raw.items():
        if isinstance(value, dict):
            per_node = OrderedDict(
                (str(node), _normalize_rate(owner, key, node, spec))
                for node, spec in value.items())
            if per_node:
                out.setdefault(str(key), OrderedDict()).update(per_node)
        else:
            out.setdefault(None, OrderedDict())[str(key)] = _normalize_rate(
                owner, None, key, value)
    return out


def _ask_rate(owner, spec, node, fn, shape):
    """Call a declared callable, giving it the arguments it can take.

    A callback is free to want nothing but the node. It is also free to be
    reached through a wrapper: a method named on a bare ``BendedModule``
    resolves to the wrapped module's, through a ``(*args, **kwargs)`` forwarder
    whose signature says nothing about what the real function accepts. Reading
    that signature literally passed neither ``fn`` nor ``shape``, and a callback
    that keyed on shape quietly answered ``None`` for everything. So a signature
    that is unreadable, or that swallows what it is given, is offered the lot.
    """
    f = getattr(owner, spec) if isinstance(spec, str) else spec
    extras = {"fn": fn, "shape": shape}
    try:
        params = inspect.signature(f).parameters
    except (TypeError, ValueError):
        params = None
    if params is not None and not any(
            p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            for p in params.values()):
        extras = {k: v for k, v in extras.items() if k in params}
    return f(node, **extras)


def resolve_sample_rate(owner, declared, node, fn=None, shape=None):
    """The declared rate for one node, or ``None`` if nothing declares it.

    Most specific first: this method's entry for this node, then a node entry
    good for any method, then this method's catch-all, then the model's.
    """
    if not declared:
        return None
    for table, key in ((declared.get(fn), node), (declared.get(None), node),
                       (declared.get(fn), "*"),  (declared.get(None), "*")):
        if not table or key not in table:
            continue
        spec = table[key]
        if isinstance(spec, float):
            return spec
        try:
            rate = _ask_rate(owner, spec, node, fn, shape)
        except Exception as exc:
            raise SampleRateError(
                "asking %s for the sample rate of %r failed: %s"
                % (getattr(owner, "__name__", type(owner).__name__), node, exc))
        if rate is None:
            continue                     # no opinion; try a broader declaration
        return _normalize_rate(owner, fn, node, rate)
    return None


class SampleRateMixin:
    """Reads ``_sample_rates_``. Shared by Interface and BendedModule."""

    #: Which nodes are audio, and at what rate. See above.
    _sample_rates_ = {}

    #: Opt in to "every node is the same timeline at a different stride".
    #:
    #: True of a fully convolutional audio model -- a codec, a vocoder -- where
    #: a tensor half as long covers the same span of time at half the rate, so
    #: an undeclared node can be placed from its length alone. False of anything
    #: with attention, a frequency axis, or a ``[B, T, C]`` layout, where that
    #: reasoning produces a confident wrong answer. Off by default: a model
    #: says this about itself, rather than a viewer assuming it.
    _strided_audio_ = False

    def sample_rates(self, fn=None):
        """The declared rates, validated.

        With ``fn`` given, the entries that apply to it -- its own, plus the
        ones declared for any method. Without, the whole table keyed by method.
        """
        # `self`, not `type(self)`: an interface whose graph is chosen at
        # construction sets this on the instance, like `methods`.
        table = normalize_sample_rates(self, self._sample_rates_)
        if fn is None:
            return table
        merged = OrderedDict(table.get(None) or {})
        merged.update(table.get(fn) or {})
        return merged

    def set_sample_rates(self, rates):
        """Declare the rates from outside the class, for a model built inline."""
        normalize_sample_rates(self, rates)      # fail here, not at read time
        self._sample_rates_ = rates

    def sample_rate_for(self, node, fn=None, shape=None):
        """The rate ``node`` is audio at, or ``None`` if nothing says it is.

        Override this for a model whose rates are easier computed than listed;
        the declaration is the default implementation, not the only route.
        """
        return resolve_sample_rate(self, normalize_sample_rates(self, self._sample_rates_),
                                   node, fn=fn, shape=shape)
