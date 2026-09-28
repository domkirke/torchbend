"""Recalling a saved activation into the graph.

A snapshot is a copy of an activation, kept by name (the graph viewer saves
them). :class:`Snapshot` puts one back: bent onto a node, it replaces the node's
value by the saved one -- so everything computed after the node follows -- or
mixes the two, which is the simplest activation interpolation there is.
"""
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
