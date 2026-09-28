"""What a model offers the input bench to start from.

``_default_inputs`` seeds the bench so a graph opens on something runnable. It
normally holds tensors — and a tensor has to reach the browser as the literal it
serialises to, which for anything the size of a real input is unreadable,
uneditable, and rides on every graph request.
"""


class Expr:
    """A default input given as the expression that makes it, not as a value.

    The bench shows the source, evaluates it, and re-evaluates it if you edit
    it — so a generated input stays something you can read and change rather
    than a wall of digits::

        model._default_inputs = {
            "t":   tb.ui.Expr("torch.linspace(0, 3, 1500)[None, None]"),
            "idx": tb.ui.Expr("torch.zeros(1, 4, 1500)"),
        }

    Evaluated where every other bench expression is, with ``torch`` and
    ``numpy`` in scope. A bare string has always meant the same thing and still
    does; this says so, and leaves room for a bare string to one day mean a
    literal value — which is what it would have to mean for a text input.
    """

    __slots__ = ("source",)

    def __init__(self, source):
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "Expr takes the source of an expression, got %r" % (source,))
        self.source = source.strip()

    def __str__(self):
        return self.source

    def __repr__(self):
        return "Expr(%r)" % self.source

    def __eq__(self, other):
        return isinstance(other, Expr) and other.source == self.source

    def __hash__(self):
        return hash(self.source)


def as_bench_value(value):
    """How one default input reaches the browser.

    Everything the bench receives is an expression, so this is where a value
    becomes one: an :class:`Expr` is its own source, a tensor is the literal it
    serialises to, and anything else is left as it is.
    """
    import json
    import torch

    if isinstance(value, Expr):
        return value.source
    if isinstance(value, torch.Tensor):
        return json.dumps(value.tolist())
    return value
