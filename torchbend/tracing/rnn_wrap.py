"""
Recurrent module wrappers for make_fx tracing.

When tracing with make_fx (proxy_tensor mode), nn.LSTM/GRU/RNN are decomposed
into per-timestep cell operations, producing thousands of nodes for long sequences.
This module provides atomic wrappers backed by custom torch.library ops that appear
as a single node in the traced graph.

Usage (automatic, via BendedModule.trace):
    bm.trace(fn='forward', x=x, _wrap_recurrent=True)

Usage (manual):
    with wrap_recurrent_modules(my_module):
        traced_gm = make_fx(my_module)(x)
"""

import contextlib
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Custom op registration — one library, registered once at import time.
# The ops are intentionally not in the aten decomp table so make_fx captures
# them as single nodes.
# ---------------------------------------------------------------------------

_lib = torch.library.Library("torchbend_rnn", "DEF")

# ── LSTM ──────────────────────────────────────────────────────────────────
_lib.define(
    "lstm_fwd("
    "Tensor input, Tensor[] hx, Tensor[] params, "
    "bool has_biases, int num_layers, float dropout, "
    "bool train, bool bidirectional, bool batch_first"
    ") -> (Tensor, Tensor, Tensor)"
)

@torch.library.impl(_lib, "lstm_fwd", "CPU")
def _lstm_fwd_cpu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    out = torch.ops.aten.lstm.input(
        input, list(hx), list(params),
        has_biases, num_layers, dropout, train, bidirectional, batch_first,
    )
    return out[0], out[1], out[2]

@torch.library.impl(_lib, "lstm_fwd", "Meta")
def _lstm_fwd_meta(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    nd = 2 if bidirectional else 1
    bs = input.size(0) if batch_first else input.size(1)
    sq = input.size(1) if batch_first else input.size(0)
    # hx[0] = h_0: [..., real_hidden_size];  hx[1] = c_0: [..., hidden_size]
    real_h = hx[0].size(-1)
    full_h = hx[1].size(-1)
    out_sz = (bs, sq, real_h * nd) if batch_first else (sq, bs, real_h * nd)
    h_sz   = (num_layers * nd, bs, real_h)
    c_sz   = (num_layers * nd, bs, full_h)
    return input.new_empty(out_sz), input.new_empty(h_sz), input.new_empty(c_sz)


# ── GRU ──────────────────────────────────────────────────────────────────
_lib.define(
    "gru_fwd("
    "Tensor input, Tensor hx, Tensor[] params, "
    "bool has_biases, int num_layers, float dropout, "
    "bool train, bool bidirectional, bool batch_first"
    ") -> (Tensor, Tensor)"
)

@torch.library.impl(_lib, "gru_fwd", "CPU")
def _gru_fwd_cpu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    out = torch.ops.aten.gru.input(
        input, hx, list(params),
        has_biases, num_layers, dropout, train, bidirectional, batch_first,
    )
    return out[0], out[1]

@torch.library.impl(_lib, "gru_fwd", "Meta")
def _gru_fwd_meta(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    nd = 2 if bidirectional else 1
    bs = input.size(0) if batch_first else input.size(1)
    sq = input.size(1) if batch_first else input.size(0)
    hs = hx.size(-1)
    out_sz = (bs, sq, hs * nd) if batch_first else (sq, bs, hs * nd)
    return input.new_empty(out_sz), input.new_empty((num_layers * nd, bs, hs))


# ── RNN (tanh / relu) ─────────────────────────────────────────────────────
_lib.define(
    "rnn_fwd("
    "Tensor input, Tensor hx, Tensor[] params, "
    "bool has_biases, int num_layers, float dropout, "
    "bool train, bool bidirectional, bool batch_first, "
    "bool use_tanh"
    ") -> (Tensor, Tensor)"
)

@torch.library.impl(_lib, "rnn_fwd", "CPU")
def _rnn_fwd_cpu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first, use_tanh):
    op = torch.ops.aten.rnn_tanh.input if use_tanh else torch.ops.aten.rnn_relu.input
    out = op(input, hx, list(params), has_biases, num_layers, dropout, train, bidirectional, batch_first)
    return out[0], out[1]

@torch.library.impl(_lib, "rnn_fwd", "Meta")
def _rnn_fwd_meta(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first, use_tanh):
    nd = 2 if bidirectional else 1
    bs = input.size(0) if batch_first else input.size(1)
    sq = input.size(1) if batch_first else input.size(0)
    hs = hx.size(-1)
    out_sz = (bs, sq, hs * nd) if batch_first else (sq, bs, hs * nd)
    return input.new_empty(out_sz), input.new_empty((num_layers * nd, bs, hs))


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------

class _LSTMAtomicWrapper(nn.Module):
    """Drop-in nn.LSTM replacement that emits a single torchbend_rnn::lstm_fwd node."""

    def __init__(self, lstm: nn.LSTM):
        super().__init__()
        self._rnn = lstm          # kept as submodule → params stay as get_attr nodes
        self._hidden_size  = lstm.hidden_size
        self._num_layers   = lstm.num_layers
        self._has_biases   = lstm.bias
        self._batch_first  = lstm.batch_first
        self._dropout      = float(lstm.dropout)
        self._bidirectional = lstm.bidirectional
        self._proj_size    = lstm.proj_size

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        nd = 2 if self._bidirectional else 1
        real_h = self._proj_size if self._proj_size > 0 else self._hidden_size

        if hx is None:
            bs = input.size(0) if self._batch_first else input.size(1)
            h0 = torch.zeros(self._num_layers * nd, bs, real_h,
                             dtype=input.dtype, device=input.device)
            c0 = torch.zeros(self._num_layers * nd, bs, self._hidden_size,
                             dtype=input.dtype, device=input.device)
            hx = (h0, c0)

        out, h_n, c_n = torch.ops.torchbend_rnn.lstm_fwd(
            input, list(hx), self._rnn._flat_weights,
            self._has_biases, self._num_layers, self._dropout,
            self.training, self._bidirectional, self._batch_first,
        )
        return out, (h_n, c_n)


class _GRUAtomicWrapper(nn.Module):
    """Drop-in nn.GRU replacement that emits a single torchbend_rnn::gru_fwd node."""

    def __init__(self, gru: nn.GRU):
        super().__init__()
        self._rnn = gru
        self._hidden_size   = gru.hidden_size
        self._num_layers    = gru.num_layers
        self._has_biases    = gru.bias
        self._batch_first   = gru.batch_first
        self._dropout       = float(gru.dropout)
        self._bidirectional = gru.bidirectional

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nd = 2 if self._bidirectional else 1

        if hx is None:
            bs = input.size(0) if self._batch_first else input.size(1)
            hx = torch.zeros(self._num_layers * nd, bs, self._hidden_size,
                             dtype=input.dtype, device=input.device)

        out, h_n = torch.ops.torchbend_rnn.gru_fwd(
            input, hx, self._rnn._flat_weights,
            self._has_biases, self._num_layers, self._dropout,
            self.training, self._bidirectional, self._batch_first,
        )
        return out, h_n


class _RNNAtomicWrapper(nn.Module):
    """Drop-in nn.RNN replacement that emits a single torchbend_rnn::rnn_fwd node."""

    def __init__(self, rnn: nn.RNN):
        super().__init__()
        self._rnn = rnn
        self._hidden_size   = rnn.hidden_size
        self._num_layers    = rnn.num_layers
        self._has_biases    = rnn.bias
        self._batch_first   = rnn.batch_first
        self._dropout       = float(rnn.dropout)
        self._bidirectional = rnn.bidirectional
        self._use_tanh      = (rnn.nonlinearity == "tanh")

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nd = 2 if self._bidirectional else 1

        if hx is None:
            bs = input.size(0) if self._batch_first else input.size(1)
            hx = torch.zeros(self._num_layers * nd, bs, self._hidden_size,
                             dtype=input.dtype, device=input.device)

        out, h_n = torch.ops.torchbend_rnn.rnn_fwd(
            input, hx, self._rnn._flat_weights,
            self._has_biases, self._num_layers, self._dropout,
            self.training, self._bidirectional, self._batch_first, self._use_tanh,
        )
        return out, h_n


_RECURRENT_TYPES = (nn.LSTM, nn.GRU, nn.RNN)

_WRAPPER_MAP = {
    nn.LSTM: _LSTMAtomicWrapper,
    nn.GRU:  _GRUAtomicWrapper,
    nn.RNN:  _RNNAtomicWrapper,
}


def _make_rnn_wrapper(module: nn.Module) -> nn.Module:
    cls = _WRAPPER_MAP.get(type(module))
    if cls is None:
        raise TypeError(f"No atomic wrapper for {type(module)}")
    return cls(module)


@contextlib.contextmanager
def wrap_recurrent_modules(module: nn.Module):
    """Context manager: temporarily replace nn.LSTM/GRU/RNN with atomic wrappers.

    Inside the context, each recurrent submodule is swapped for a lightweight
    wrapper whose ``forward`` calls a single custom aten-level op.  This makes
    ``make_fx`` (proxy_tensor tracing) capture the recurrent layer as one graph
    node instead of unrolling every timestep.

    Example::

        with wrap_recurrent_modules(my_module):
            gm = make_fx(my_module)(x)
    """
    replacements: list = []
    for name, submod in list(module.named_modules()):
        if not isinstance(submod, _RECURRENT_TYPES):
            continue
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = module.get_submodule(parent_name)
        else:
            parent, child_name = module, name
        wrapper = _make_rnn_wrapper(submod)
        replacements.append((parent, child_name, submod))
        setattr(parent, child_name, wrapper)
    try:
        yield
    finally:
        for parent, child_name, original in replacements:
            setattr(parent, child_name, original)


__all__ = ["wrap_recurrent_modules", "_LSTMAtomicWrapper", "_GRUAtomicWrapper", "_RNNAtomicWrapper"]
