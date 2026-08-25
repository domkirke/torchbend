"""
Complex models for debugging torchbend tracing and bending.

Each model exercises a different challenging pattern:
  - skip connections & residual paths
  - variational reparameterization
  - self-attention with dynamic reshaping
  - encoder-decoder with skip concat (UNet)
  - gated activations (WaveNet-style)
  - deeply nested submodules
  - multi-input / multi-output
  - LSTM-based recurrent encode
  - conditional (class-conditioned) processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module_config import ModuleTestConfig


# ---------------------------------------------------------------------------
# 1.  Residual network (skip connections + BatchNorm)
# ---------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ResNet1D(nn.Module):
    """Stack of residual blocks with a projecting stem and output head."""
    __bended_methods__ = ['forward']

    def __init__(self, in_channels: int = 1, channels: int = 16, n_blocks: int = 3, out_dim: int = 8):
        super().__init__()
        self.stem    = nn.Conv1d(in_channels, channels, 7, padding=3)
        self.blocks  = nn.ModuleList([ResidualBlock1D(channels) for _ in range(n_blocks)])
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Linear(channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x))
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ---------------------------------------------------------------------------
# 2.  Convolutional VAE encoder (reparameterization trick)
# ---------------------------------------------------------------------------

class ConvVAEEncoder(nn.Module):
    """
    Encodes a 1-D signal into (z, mu, log_var).
    Tests reparameterization and multi-output tensors.
    """
    __bended_methods__ = ['forward', 'encode']

    def __init__(self, in_channels: int = 1, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.flatten_dim = 64 * 8
        self.fc_mu      = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim, latent_dim)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        h = h.reshape(h.shape[0], -1)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


# ---------------------------------------------------------------------------
# 3.  Multi-head self-attention (dynamic reshape / transpose patterns)
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Minimal MHSA operating on (batch, seq, dim).
    Tests view/reshape with runtime-dependent dimensions.
    """
    __bended_methods__ = ['forward']

    def __init__(self, embed_dim: int = 32, n_heads: int = 4):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads   = n_heads
        self.head_dim  = embed_dim // n_heads
        self.q_proj    = nn.Linear(embed_dim, embed_dim)
        self.k_proj    = nn.Linear(embed_dim, embed_dim)
        self.v_proj    = nn.Linear(embed_dim, embed_dim)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)
        self.scale     = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out  = torch.matmul(attn, v)
        out  = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# 4.  Transformer block (attention + FFN + LayerNorm)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block. Tests LayerNorm and sequential sub-modules."""
    __bended_methods__ = ['forward']

    def __init__(self, embed_dim: int = 32, n_heads: int = 4, ff_mult: int = 4):
        super().__init__()
        self.norm1  = nn.LayerNorm(embed_dim)
        self.attn   = MultiHeadSelfAttention(embed_dim, n_heads)
        self.norm2  = nn.LayerNorm(embed_dim)
        self.ff     = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Linear(embed_dim * ff_mult, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# 5.  1-D UNet (encoder-decoder with skip connections via list)
# ---------------------------------------------------------------------------

class UNetEncoder1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels[i], channels[i+1], 3, stride=2, padding=1), nn.ReLU())
            for i in range(len(channels) - 1)
        ])

    def forward(self, x: torch.Tensor):
        skips = []
        for layer in self.layers:
            x = layer(x)
            skips.append(x)
        return x, skips


class UNetDecoder1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(channels[i] * 2, channels[i+1], 4, stride=2, padding=1),
                nn.ReLU()
            )
            for i in range(len(channels) - 1)
        ])

    def forward(self, x: torch.Tensor, skips):
        for i, layer in enumerate(self.layers):
            skip = skips[-(i + 1)]
            x = layer(torch.cat([x, skip], dim=1))
        return x


class UNet1D(nn.Module):
    """
    Encoder-decoder with skip connections collected in a list.
    Tests list-building in the forward pass.
    """
    __bended_methods__ = ['forward']

    def __init__(self, in_channels: int = 1, base: int = 16, depth: int = 3):
        super().__init__()
        enc_ch = [in_channels] + [base * (2 ** i) for i in range(depth)]
        dec_ch = list(reversed(enc_ch[1:]))

        self.stem    = nn.Conv1d(in_channels, enc_ch[1], 1)
        self.encoder = UNetEncoder1D(enc_ch[1:])
        self.bottleneck = nn.Sequential(
            nn.Conv1d(enc_ch[-1], enc_ch[-1], 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = UNetDecoder1D(dec_ch)
        self.head    = nn.Conv1d(dec_ch[-1], in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skips)
        return self.head(x)


# ---------------------------------------------------------------------------
# 6.  Gated convolutional network (WaveNet-style)
# ---------------------------------------------------------------------------

class GatedConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv_f = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv_g = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.res    = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor):
        f = torch.tanh(self.conv_f(x))
        g = torch.sigmoid(self.conv_g(x))
        activated = f * g
        return self.res(activated) + x, activated


class GatedConvNet(nn.Module):
    """
    Stack of gated conv blocks with exponentially growing dilations.
    Each block returns two tensors; skip outputs are summed.
    Tests multi-return submodules and accumulator patterns.
    """
    __bended_methods__ = ['forward']

    def __init__(self, channels: int = 16, n_layers: int = 4, in_channels: int = 1):
        super().__init__()
        self.input_conv = nn.Conv1d(in_channels, channels, 1)
        self.blocks = nn.ModuleList([
            GatedConvBlock(channels, dilation=2 ** i)
            for i in range(n_layers)
        ])
        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv1d(channels, in_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        skip_sum = torch.zeros_like(x)
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        return self.output_conv(skip_sum)


# ---------------------------------------------------------------------------
# 7.  Deeply nested submodules (tests recursive parameter/activation access)
# ---------------------------------------------------------------------------

class InnerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.relu(self.fc(x)))


class MiddleBlock(nn.Module):
    def __init__(self, dim: int, n_inner: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([InnerBlock(dim) for _ in range(n_inner)])
        self.proj   = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)


class HierarchicalMLP(nn.Module):
    """
    Three-level nesting: outer -> middle -> inner.
    Tests that parameter flattening and activation capture work
    through multiple levels of nn.ModuleList.
    """
    __bended_methods__ = ['forward']

    def __init__(self, dim: int = 32, n_middle: int = 2, n_inner: int = 2):
        super().__init__()
        self.embed   = nn.Linear(dim, dim)
        self.stages  = nn.ModuleList([MiddleBlock(dim, n_inner) for _ in range(n_middle)])
        self.out     = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.embed(x))
        for stage in self.stages:
            x = x + stage(x)
        return self.out(x)


# ---------------------------------------------------------------------------
# 8.  Multi-input, multi-output module
# ---------------------------------------------------------------------------

class MultiIOModule(nn.Module):
    """
    Takes two separate 1-D signals, processes them independently then jointly.
    Returns three tensors: fused, branch_a, branch_b.
    Tests multi-arg forward and tuple returns.
    """
    __bended_methods__ = ['forward']

    def __init__(self, channels: int = 16, dim: int = 32):
        super().__init__()
        self.branch_a = nn.Sequential(
            nn.Conv1d(1, channels, 3, padding=1), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1), nn.ReLU(),
        )
        self.branch_b = nn.Sequential(
            nn.Conv1d(1, channels, 5, padding=2), nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=2), nn.ReLU(),
        )
        self.fuse = nn.Conv1d(channels * 2, channels, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        fa = self.branch_a(a)
        fb = self.branch_b(b)
        fused = F.relu(self.fuse(torch.cat([fa, fb], dim=1)))
        out   = self.head(self.pool(fused).squeeze(-1))
        return out, fa, fb


# ---------------------------------------------------------------------------
# 9.  LSTM-based recurrent encoder
# ---------------------------------------------------------------------------

class RecurrentEncoder(nn.Module):
    """
    Convolutional front-end -> bidirectional LSTM -> projection.
    Tests LSTM's tuple hidden-state handling.
    """
    __bended_methods__ = ['forward', 'encode']

    def __init__(self, in_channels: int = 1, hidden: int = 32, latent: int = 16):
        super().__init__()
        self.cnn  = nn.Sequential(
            nn.Conv1d(in_channels, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 4, stride=2, padding=1),          nn.ReLU(),
        )
        self.lstm = nn.LSTM(32, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, latent)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x).permute(0, 2, 1)
        out, _ = self.lstm(feat)
        return self.proj(out[:, -1, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


# ---------------------------------------------------------------------------
# 10.  Class-conditioned module (FiLM / affine conditioning)
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: scale + shift conditioned on a label."""
    def __init__(self, n_classes: int, channels: int):
        super().__init__()
        self.gamma = nn.Embedding(n_classes, channels)
        self.beta  = nn.Embedding(n_classes, channels)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        g = self.gamma(labels).unsqueeze(-1)
        b = self.beta(labels).unsqueeze(-1)
        return g * x + b


class ConditionalConvNet(nn.Module):
    """
    1-D ConvNet conditioned on a class label via FiLM at every block.
    Tests conditioning paths, Embedding lookup, and interleaved sub-modules.
    """
    __bended_methods__ = ['forward']

    def __init__(self, n_classes: int = 4, in_channels: int = 1, channels: int = 16, n_layers: int = 3):
        super().__init__()
        self.stem  = nn.Conv1d(in_channels, channels, 1)
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, 3, padding=1)
            for _ in range(n_layers)
        ])
        self.films = nn.ModuleList([
            FiLMLayer(n_classes, channels)
            for _ in range(n_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for conv, film in zip(self.convs, self.films):
            x = F.relu(film(conv(x), labels))
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ---------------------------------------------------------------------------
# ModuleTestConfig registrations
# ---------------------------------------------------------------------------

_B, _T, _L = 1, 16, 1024   # batch, sequence length (for transformers), signal length

# ── Reusable default-input expression strings ────────────────────────────────
_SINE_1D   = "torch.sin(2 * torch.pi * torch.arange(1024) / 512)[None, None]"
_CHIRP_1D  = ("torch.sin(2 * torch.pi * torch.arange(1024).float() "
              "* (1 + torch.arange(1024).float() / 1024) / 256)[None, None]")
_SINE_64   = "torch.sin(2 * torch.pi * torch.arange(64) / 16.0)[None, None]"
_PE_16_32  = ("torch.stack([torch.sin(torch.arange(32).float() * k * 0.3) "
              "for k in range(16)], dim=0)[None]")   # [1, 16, 32] sinusoidal PE
_RAMP_32   = "torch.linspace(-1, 1, 32)[None]"       # [1, 32] linear ramp

modules_to_test = [

    ModuleTestConfig(
        ResNet1D,
        (tuple(), dict(in_channels=1, channels=16, n_blocks=3, out_dim=8)),
        {'forward': (
            tuple(),
            {'x': torch.randn(_B, 1, _L)},
            ['?.*weight'],
            ['stem', 'blocks_0_conv1', 'head'],
            False,
        )},
        default_inputs={'x': _SINE_1D},
    ),

    ModuleTestConfig(
        ConvVAEEncoder,
        (tuple(), dict(in_channels=1, latent_dim=16)),
        {
            'encode': (
                tuple(),
                {'x': torch.randn(_B, 1, 64)},
                ['?.*weight'],
                ['encoder_0', 'fc_mu'],
                False,
            ),
            'forward': (
                tuple(),
                {'x': torch.randn(_B, 1, 64)},
                ['?.*weight'],
                ['encoder_0', 'fc_mu'],
                False,
            ),
        },
        default_inputs={'x': _SINE_64},
    ),

    ModuleTestConfig(
        MultiHeadSelfAttention,
        (tuple(), dict(embed_dim=32, n_heads=4)),
        {'forward': (
            tuple(),
            {'x': torch.randn(_B, _T, 32)},
            ['?.*weight'],
            ['q_proj', 'out_proj'],
            False,
        )},
        default_inputs={'x': _PE_16_32},
    ),

    ModuleTestConfig(
        TransformerBlock,
        (tuple(), dict(embed_dim=32, n_heads=4, ff_mult=4)),
        {'forward': (
            tuple(),
            {'x': torch.randn(_B, _T, 32)},
            ['?.*weight'],
            ['attn_q_proj', 'ff_0'],
            False,
        )},
        default_inputs={'x': _PE_16_32},
    ),

    ModuleTestConfig(
        UNet1D,
        (tuple(), dict(in_channels=1, base=8, depth=3)),
        {'forward': (
            tuple(),
            {'x': torch.randn(_B, 1, _L)},
            ['?.*weight'],
            ['stem', 'bottleneck_0'],
            False,
        )},
        default_inputs={'x': _SINE_1D},
    ),

    ModuleTestConfig(
        GatedConvNet,
        (tuple(), dict(channels=16, n_layers=4, in_channels=1)),
        {'forward': (
            tuple(),
            {'x': torch.randn(_B, 1, _L)},
            ['?.*weight'],
            ['input_conv', 'blocks_0_res'],
            False,
        )},
        default_inputs={'x': _CHIRP_1D},
    ),

    ModuleTestConfig(
        HierarchicalMLP,
        (tuple(), dict(dim=32, n_middle=2, n_inner=2)),
        {'forward': (
            tuple(),
            {'x': torch.randn(_B, 32)},
            ['?.*weight'],
            ['embed', 'stages_0_blocks_0_fc'],
            False,
        )},
        default_inputs={'x': _RAMP_32},
    ),

    ModuleTestConfig(
        MultiIOModule,
        (tuple(), dict(channels=16, dim=32)),
        {'forward': (
            tuple(),
            {'a': torch.randn(_B, 1, _L), 'b': torch.randn(_B, 1, _L)},
            ['?.*weight'],
            ['branch_a_0', 'fuse'],
            False,
        )},
        default_inputs={
            'a': _SINE_1D,
            'b': _CHIRP_1D,
        },
    ),

    ModuleTestConfig(
        RecurrentEncoder,
        (tuple(), dict(in_channels=1, hidden=16, latent=8)),
        {
            'encode': (
                tuple(),
                {'x': torch.randn(_B, 1, _L)},
                ['?.*weight'],
                ['cnn_0', 'proj'],
                False,
            ),
            'forward': (
                tuple(),
                {'x': torch.randn(_B, 1, _L)},
                ['?.*weight'],
                ['cnn_0', 'proj'],
                False,
            ),
        },
        default_inputs={'x': _SINE_1D},
        wrap_recurrent=True,
    ),

    ModuleTestConfig(
        ConditionalConvNet,
        (tuple(), dict(n_classes=4, in_channels=1, channels=16, n_layers=3)),
        {'forward': (
            tuple(),
            {'x': torch.randn(_B, 1, _L), 'labels': torch.randint(0, 4, (_B,))},
            ['?.*weight'],
            ['stem', 'convs_0', 'films_0_gamma'],
            False,
        )},
        default_inputs={
            'x':      _SINE_1D,
            'labels': 'torch.zeros(1, dtype=torch.long)',
        },
    ),

]
