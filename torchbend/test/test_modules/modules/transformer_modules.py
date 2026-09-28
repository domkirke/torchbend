"""Toy transformers for exercising the text interaction path without a download.

``TinyGPT`` is GPT-2's shape at 1/1000th the size: same module names
(``transformer.h.0.attn.c_attn``, ``ln_f``, ``lm_head``), same
``{"logits": ...}`` return, same ``(input_ids, attention_mask, position_ids)``
signature. Anything the viewer does to GPT-2 — module scoping, reshape-chain
folding, the prompt input mode — it does to this, in milliseconds and offline.

``TinyAudioGen`` is the text2sound counterpart: a text conditioner feeding a
temporal decoder feeding a synthesiser, which is AudioGen's shape even though
the synthesiser here is a harmonic oscillator bank rather than a neural codec.
It exists because text-to-*audio* is the case GPT-2 cannot cover, and it is what
puts a declared ``returns: "audio"`` callback under a real waveform.

Attention is written out (matmul, mask, softmax) instead of calling
``scaled_dot_product_attention``: the fused kernel collapses into one opaque
node with a tuple output, and the point of a test model is to have things worth
bending in the graph.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module_config import ModuleTestConfig


__all__ = ["TinyGPT", "TinyAudioGen", "TinyGPTConfig"]


class TinyGPTConfig:
    """Small enough to trace instantly, wide enough to be a real transformer."""

    def __init__(self, vocab_size=98, n_layer=2, n_head=2, n_embd=32, block_size=128):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size

    # names GPT-2's interface reads off its own config
    @property
    def n_positions(self):
        return self.block_size


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x, attn_bias):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att + attn_bias
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, attn_bias):
        x = x + self.attn(self.ln_1(x), attn_bias)
        x = x + self.mlp(self.ln_2(x))
        return x


class _Transformer(nn.Module):
    """The stack itself, shared by the text model and the audio conditioner."""

    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

    def forward(self, input_ids, attn_bias, position_ids):
        x = self.wte(input_ids) + self.wpe(position_ids)
        for block in self.h:
            x = block(x, attn_bias)
        return self.ln_f(x)


def _attention_bias(input_ids, attention_mask):
    """Additive mask: causal, plus whatever padding the batch carries.

    Additive rather than boolean because ``masked_fill`` on a symbolic-shaped
    bool tensor is where traces tend to go data-dependent; adding a large
    negative number is arithmetic all the way down.
    """
    T = input_ids.shape[-1]
    causal = torch.ones(T, T, device=input_ids.device).tril()
    bias = (1.0 - causal) * -1e9                      # [T, T]
    bias = bias[None, None, :, :]                     # [1, 1, T, T]
    if attention_mask is not None:
        pad = (1.0 - attention_mask.to(bias.dtype)) * -1e9
        bias = bias + pad[:, None, None, :]           # [B, 1, 1, T]
    return bias


def _default_positions(attention_mask, input_ids):
    """Positions that skip left padding, so a padded batch stays aligned."""
    if attention_mask is None:
        T = input_ids.shape[-1]
        return torch.arange(T, device=input_ids.device).expand_as(input_ids)
    return (attention_mask.cumsum(-1) - 1).clamp(min=0)


class TinyGPT(nn.Module):
    """A GPT-2 shaped language model, small enough to trace in milliseconds."""

    def __init__(self, vocab_size=98, n_layer=2, n_head=2, n_embd=32, block_size=128):
        super().__init__()
        self.config = TinyGPTConfig(vocab_size, n_layer, n_head, n_embd, block_size)
        self.transformer = _Transformer(self.config)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        if position_ids is None:
            position_ids = _default_positions(attention_mask, input_ids)
        bias = _attention_bias(input_ids, attention_mask)
        hidden = self.transformer(input_ids, bias, position_ids)
        # a dict, like GPT-2: it is also the shape that used to defeat the
        # viewer's output discovery, so keeping it here keeps that covered
        return {"logits": self.lm_head(hidden)}


class HarmonicSynth(nn.Module):
    """A DDSP-style oscillator bank: f0 and per-harmonic amplitudes in, audio out.

    An untrained neural vocoder produces noise, and noise tells you nothing
    about whether a bending did anything. Additive synthesis from predicted
    controls is audible from the first random initialisation — bend the layer
    that predicts f0 and the pitch moves.
    """

    def __init__(self, n_harmonics=16, sample_rate=16000, hop=256,
                 f0_min=60.0, f0_max=1000.0):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.hop = hop
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.register_buffer("harmonics",
                             torch.arange(1, n_harmonics + 1).float()[None, :, None])

    def forward(self, f0_logits, amp_logits):
        # f0_logits: [B, 1, n_frames]   amp_logits: [B, n_harmonics, n_frames]
        f0 = self.f0_min + (self.f0_max - self.f0_min) * torch.sigmoid(f0_logits)
        amps = torch.softmax(amp_logits, dim=1)

        f0 = F.interpolate(f0, scale_factor=float(self.hop), mode="linear",
                           align_corners=False)
        amps = F.interpolate(amps, scale_factor=float(self.hop), mode="linear",
                             align_corners=False)

        # phase per harmonic, integrated over time
        freqs = f0 * self.harmonics                        # [B, n_harmonics, T]
        phase = torch.cumsum(2 * math.pi * freqs / self.sample_rate, dim=-1)
        # harmonics above Nyquist would alias back down as tones that are not there
        amps = amps * (freqs < (self.sample_rate / 2)).to(amps.dtype)
        audio = (torch.sin(phase) * amps).sum(dim=1, keepdim=True)
        return torch.tanh(audio)


class TinyAudioGen(nn.Module):
    """Text to sound, in AudioGen's shape: conditioner → decoder → synthesiser."""

    def __init__(self, vocab_size=98, n_layer=2, n_head=2, n_embd=32,
                 block_size=128, n_frames=32, n_harmonics=16,
                 sample_rate=16000, hop=256):
        super().__init__()
        self.config = TinyGPTConfig(vocab_size, n_layer, n_head, n_embd, block_size)
        self.n_frames = n_frames
        self.sample_rate = sample_rate
        # text side
        self.conditioner = _Transformer(self.config)
        # audio side: a learned frame query per output step, conditioned on text
        self.frame_pos = nn.Parameter(torch.randn(1, n_frames, n_embd) * 0.02)
        self.decoder = nn.ModuleList([Block(self.config) for _ in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        self.to_f0 = nn.Linear(n_embd, 1)
        self.to_amps = nn.Linear(n_embd, n_harmonics)
        self.synth = HarmonicSynth(n_harmonics, sample_rate, hop)

    def forward(self, input_ids, attention_mask=None):
        position_ids = _default_positions(attention_mask, input_ids)
        bias = _attention_bias(input_ids, attention_mask)
        hidden = self.conditioner(input_ids, bias, position_ids)

        # pool the prompt into one conditioning vector, ignoring padding
        if attention_mask is not None:
            weights = attention_mask.to(hidden.dtype)[..., None]
            cond = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        else:
            cond = hidden.mean(1)

        frames = self.frame_pos + cond[:, None, :]
        # the decoder attends over frames only, so no mask is needed here
        no_bias = torch.zeros(1, 1, 1, 1, device=input_ids.device)
        for block in self.decoder:
            frames = block(frames, no_bias)
        frames = self.ln_out(frames)

        f0 = self.to_f0(frames).transpose(1, 2)       # [B, 1, n_frames]
        amps = self.to_amps(frames).transpose(1, 2)   # [B, n_harmonics, n_frames]
        return self.synth(f0, amps)


# ── test registry ─────────────────────────────────────────────────────────────

_IDS = torch.randint(3, 90, (2, 12))
_MASK = torch.ones(2, 12, dtype=torch.long)

# Deliberately NOT named `modules_to_test`: the package __init__ globs for that
# name, and these would join a matrix whose bending tests are already red for
# most modules (`test_weight_bending` fails for every config at the time of
# writing). Adding two more sets of failures to that would make the suite harder
# to reason about, not easier. Splice this in when the matrix is green:
#
#     from .transformer_modules import transformer_modules_to_test
#     modules_to_test.extend(transformer_modules_to_test)
#
# The models themselves are exercised by test_transformer_modules.py.
transformer_modules_to_test = [
    ModuleTestConfig(
        TinyGPT,
        (tuple(), dict(vocab_size=98, n_layer=2, n_head=2, n_embd=32)),
        {
            "forward": (
                tuple(),
                {"input_ids": _IDS, "attention_mask": _MASK},
                ["transformer.h.0.attn.c_attn.weight", "lm_head.weight"],
                # ATen-level node names, not module paths: the trace is taken
                # below nn.Module. They must also be tensor-valued —
                # `native_layer_norm` returns a tuple, has no single shape, and
                # so cannot carry a bending.
                ["embedding", "addmm"],
                False,
            ),
        },
    ),
    ModuleTestConfig(
        TinyAudioGen,
        (tuple(), dict(vocab_size=98, n_layer=2, n_head=2, n_embd=32, n_frames=16)),
        {
            "forward": (
                tuple(),
                {"input_ids": _IDS, "attention_mask": _MASK},
                ["to_f0.weight", "to_amps.weight"],
                ["embedding", "addmm"],
                False,
            ),
        },
    ),
]
