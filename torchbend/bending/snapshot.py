"""Recalling a saved activation into the graph.

A snapshot is a copy of an activation, kept by name (the graph viewer saves
them). :class:`Snapshot` puts one back: bent onto a node, it replaces the node's
value by the saved one -- so everything computed after the node follows -- or
mixes the two, which is the simplest activation interpolation there is.
"""
import math
from collections import OrderedDict
from typing import Optional

import torch

from .callback import BendingCallback
from .parameter import BendingParameter


def fit_to(saved: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """``saved`` shaped as ``like``: size-1 dimensions broadcast, longer ones
    cropped, shorter ones zero-padded -- a snapshot taken on another input (a
    longer caption, another batch) still has somewhere to go."""
    s = saved.to(device=like.device, dtype=like.dtype)
    if s.shape == like.shape:
        return s
    if s.ndim != like.ndim:
        try:
            return s.expand_as(like)
        except RuntimeError:
            raise ValueError("snapshot of shape %s cannot stand for an activation of shape %s"
                             % (list(saved.shape), list(like.shape)))
    for d in range(like.ndim):
        have, want = s.shape[d], like.shape[d]
        if have == want:
            continue
        if have == 1:
            s = s.expand(*[want if i == d else -1 for i in range(s.ndim)])
        elif have > want:
            s = s.narrow(d, 0, want)
        else:
            pad = [0] * (2 * s.ndim)
            pad[2 * (s.ndim - 1 - d) + 1] = want - have
            s = torch.nn.functional.pad(s, pad)
    return s


class Snapshot(BendingCallback):
    """Replaces an activation by a saved one, or mixes the two: ``mix`` = 1 is
    the snapshot, 0 the live activation, anything between an interpolation."""

    weight_compatible = False
    activation_compatible = True
    jit_compatible = False
    nntilde_compatible = False
    # needs a saved tensor, so it is not offered as a plain bending: the graph
    # viewer creates it when a view is recalled to a snapshot
    ui_compatible = False
    controllable_params = OrderedDict({'mix': (None, 1.)})
    _param_ui = {
        'mix': {
            'range': [0., 1.],
            'step': 0.01,
            'widget': 'slider',
            'description': "1 = the snapshot, 0 = the live activation; in between, "
                           "a crossfade of the two.",
        },
    }

    def __init__(self, snapshot: Optional[torch.Tensor] = None,
                 mix: float | torch.Tensor | BendingParameter = 1., snapshot_name: Optional[str] = None):
        super().__init__(mix=mix)
        self.snapshot_name = snapshot_name
        self._snapshot = None if snapshot is None else snapshot.detach().to("cpu")

    def __repr__(self):
        return "Snapshot(%s, mix=%.2f)" % (self.snapshot_name or "?", float(self.get('mix')))

    def set_snapshot(self, tensor: torch.Tensor, name: Optional[str] = None):
        self._snapshot = tensor.detach().to("cpu")
        if name is not None:
            self.snapshot_name = name

    def apply_to_param(self, idx, param, cache=None):
        pass

    def bend_input(self, x: torch.Tensor, mix: torch.Tensor | None = None, name: str | None = None):
        if self._snapshot is None:
            return x
        saved = fit_to(self._snapshot, x)
        if mix is None:
            return saved
        m = torch.as_tensor(mix).to(device=x.device, dtype=x.dtype)
        return x + (saved - x) * m


# ── mixing several ───────────────────────────────────────────────────────────

MIX_MODES = ("linear", "cosine", "slerp", "max", "min", "sweep")

MIX_MODE_HELP = {
    "linear": "Weighted average of the sources.",
    "cosine": "Equal-power crossfade: each source's share of the weights, p, "
              "becomes a gain sin(p·π/2) -- with two sources, the classic "
              "cos/sin crossfade. Unlike linear, the middle of a crossfade does "
              "not dip in level.",
    "slerp":  "Mixes directions, not values: along the target dimension each "
              "source is normalised, the weighted directions are averaged and "
              "renormalised, and given the weighted average norm -- the result "
              "does not shrink the way an average of opposite vectors does.",
    "max":    "For each slice along the target dimension (each channel, by "
              "default), the source with the largest amplitude (RMS over the "
              "other dimensions) wins. Weights scale the amplitudes; 0 drops a "
              "source.",
    "min":    "As max, with the smallest amplitude.",
    "sweep":  "A crossfade along the target dimension: from the first source at "
              "its start to the last at its end, through the others in order. "
              "Sources weighted 0 are skipped.",
}


def _norm_dim(dim, ndim):
    if dim is None:
        return None
    dim = int(dim)
    if not -ndim <= dim < ndim:
        raise ValueError("mix dimension %d is out of range for a %d-d activation" % (dim, ndim))
    return dim % ndim


def mix_tensors(sources, weights, mode="linear", dim=None, live=None):
    """Mix same-shaped tensors. ``weights`` is a 1-d tensor, one per source;
    ``dim`` is the target dimension (None: the whole tensor, or per element
    for max / min, the last dimension for sweep). ``live`` is returned when
    every weight is 0."""
    if mode not in MIX_MODES:
        raise ValueError("unknown mix mode %r (have: %s)" % (mode, ", ".join(MIX_MODES)))
    stack = torch.stack(list(sources))                          # [n, *shape]
    n, shape = stack.shape[0], stack.shape[1:]
    ndim = len(shape)
    w = weights.to(device=stack.device, dtype=stack.dtype).reshape(n)
    wv = w.view(n, *([1] * ndim))
    d = _norm_dim(dim, ndim)
    eps = torch.finfo(stack.dtype).eps if stack.dtype.is_floating_point else 1e-8
    if mode != "sweep" and bool((w.abs().sum() <= eps).item()):
        return live if live is not None else stack.mean(0)

    if mode == "linear":
        return (wv * stack).sum(0) / w.sum()

    if mode == "cosine":
        # equal power: the gains' squares sum to about 1 wherever the weights sit
        share = w / w.sum()
        gains = torch.sin(share * (math.pi / 2)).view(n, *([1] * ndim))
        return (gains * stack).sum(0)

    if mode == "slerp":
        reduce = (d + 1,) if d is not None else tuple(range(1, ndim + 1))
        norms = stack.norm(dim=reduce, keepdim=True).clamp_min(eps)
        direction = (wv * stack / norms).sum(0)
        direction = direction / direction.norm(dim=tuple(r - 1 for r in reduce), keepdim=True).clamp_min(eps)
        magnitude = (wv * norms).sum(0) / w.sum()
        return direction * magnitude

    if mode in ("max", "min"):
        if d is None:
            amp = stack.abs()                                       # per element
        else:
            others = tuple(i + 1 for i in range(ndim) if i != d)
            amp = stack.pow(2).mean(dim=others, keepdim=True).sqrt() if others else stack.abs()
        keep = (w > 0).view(n, *([1] * ndim))
        if mode == "max":
            score = torch.where(keep, amp * wv, torch.full_like(amp, -float("inf")))
            pick = score.argmax(0, keepdim=True)
        else:
            score = torch.where(keep, amp / wv.clamp_min(eps), torch.full_like(amp, float("inf")))
            pick = score.argmin(0, keepdim=True)
        return torch.gather(stack, 0, pick.expand(1, *shape))[0]

    # sweep: through the sources in order, along the target dimension
    order = [i for i in range(n) if float(w[i]) > 0] or list(range(n))
    stack = stack[order]
    n = stack.shape[0]
    if n == 1:
        return stack[0]
    d = ndim - 1 if d is None else d
    length = shape[d]
    pos = torch.linspace(0, n - 1, length, device=stack.device, dtype=stack.dtype)
    lo = pos.floor().clamp(max=n - 2).long()
    frac = pos - lo.to(stack.dtype)
    view = [1] * ndim
    view[d] = length
    lo_idx = lo.view(1, *view).expand(1, *shape)
    a = torch.gather(stack, 0, lo_idx)[0]
    b = torch.gather(stack, 0, lo_idx + 1)[0]
    f = frac.view(*view)
    return a * (1 - f) + b * f


class Mix(BendingCallback):
    """Mixes several saved activations -- and, if asked, the live one -- into
    the node, in one of several ways (see MIX_MODE_HELP), along a target
    dimension. One weight per source, each a controllable parameter."""

    weight_compatible = False
    activation_compatible = True
    jit_compatible = False
    nntilde_compatible = False
    ui_compatible = False          # built from snapshots, not offered on its own
    controllable_params = OrderedDict()
    _param_ui = {}

    @classmethod
    def build(cls, sources, names, mode="linear", dim=1, weights=None):
        """A Mix over ``sources`` (tensors; ``None`` for the live activation),
        with a weight parameter ``w_<i>`` per source, labelled by ``names``.

        A mix has as many weights as sources, and a callback's parameters are
        declared on its class -- so each mix gets a subclass of its own."""
        if len(sources) != len(names) or not sources:
            raise ValueError("a mix needs sources, one name each")
        if mode not in MIX_MODES:
            raise ValueError("unknown mix mode %r (have: %s)" % (mode, ", ".join(MIX_MODES)))
        params = OrderedDict(("w_%d" % i, (None, 1.)) for i in range(len(names)))
        ui = {"w_%d" % i: {"range": [0., 1.], "step": 0.01, "widget": "slider",
                           "label": "w · " + str(n),
                           "description": "How much of %s goes into the mix." % n}
              for i, n in enumerate(names)}
        klass = type("Mix", (cls,), {"controllable_params": params, "_param_ui": ui,
                                      "__module__": cls.__module__, "__doc__": cls.__doc__})
        weights = list(weights) if weights is not None else [1.] * len(names)
        return klass(sources, names, mode=mode, dim=dim,
                     **{"w_%d" % i: float(v) for i, v in enumerate(weights)})

    def __init__(self, sources=None, names=None, mode="linear", dim=1, **weights):
        super().__init__(**weights)
        self._sources = [None if s is None else s.detach().to("cpu") for s in (sources or [])]
        self.names = list(names or [])
        self.mode = mode
        self.dim = dim

    def __repr__(self):
        return "Mix(%s, mode=%s, dim=%s)" % (", ".join(self.names), self.mode, self.dim)

    def apply_to_param(self, idx, param, cache=None):
        pass

    def bend_input(self, x: torch.Tensor, name: str | None = None, **weights):
        if not self._sources:
            return x
        sources = [x if s is None else fit_to(s, x) for s in self._sources]
        w = torch.stack([torch.as_tensor(weights.get("w_%d" % i, 1.), dtype=x.dtype).reshape(())
                         for i in range(len(sources))]).to(x.device)
        return mix_tensors(sources, w, self.mode, self.dim, live=x)
