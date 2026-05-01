#!/usr/bin/env python3
"""Batch DAC pairwise audio bending script.

Expects a root folder containing one sub-folder per audio set.
All file pairs within each sub-folder are processed with every bending op.

Usage:
    python bend_dac_pairs.py --input_dir data/sets --output_dir outs/bend_pairs
"""

import random
from itertools import combinations
from pathlib import Path

import torch
import torchaudio
from absl import app, flags

import torchbend
from torchbend.interfaces.descript_interface import BendedDescriptAudioCodec
from torchbend.utils import load_audio

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', None, 'Root folder with one sub-folder per audio set', required=True)
flags.DEFINE_string('output_dir', 'outs/bend_pairs', 'Output root folder')
flags.DEFINE_string('model_type', '44khz', 'DAC model type: 44khz | 24khz | 16khz')
flags.DEFINE_string('device', 'cuda', 'Torch device')
flags.DEFINE_integer('n_seeds', 4, 'Number of random seeds for stochastic ops')
flags.DEFINE_integer('seed', 42, 'Global random seed')
flags.DEFINE_boolean('force', False, 'Recompute and overwrite files that already exist')


# ---------------------------------------------------------------------------
# z-domain pair primitives
# ---------------------------------------------------------------------------

def _mix_z(z1, z2, idx):
    out = torch.empty_like(z1)
    out[:, :idx, :] = z1[:, :idx, :]
    out[:, idx:, :] = z2[:, idx:, :]
    return out


def _blend_z(z1, z2, seed=0):
    gen = torch.Generator(device=z1.device).manual_seed(seed)
    mask = torch.randint(0, 2, (z1.shape[-2],), device=z1.device, generator=gen)
    return z1 * mask.reshape(1, -1, 1).to(z1.dtype) + z2 * (1 - mask).reshape(1, -1, 1).to(z2.dtype)


def _stochastic_mix_z(z1, z2, prob=0.5, seed=0):
    gen = torch.Generator(device=z1.device).manual_seed(seed)
    mask = torch.bernoulli(torch.full((z1.shape[-2],), prob, device=z1.device), generator=gen)
    mask = mask.reshape(1, -1, 1).to(z1.dtype)
    return z1 * mask + z2 * (1 - mask)


def _lerp_z(z1, z2, alpha=0.5):
    return alpha * z1 + (1 - alpha) * z2


def _sum_max_z(z1, z2, prob=0.5):
    k = max(1, int(prob * z1.shape[-2]))
    a1, a2 = z1.abs().mean(-1).mean(0), z2.abs().mean(-1).mean(0)
    mask1 = torch.zeros(z1.shape[-2], device=z1.device, dtype=z1.dtype)
    mask2 = torch.zeros(z2.shape[-2], device=z2.device, dtype=z2.dtype)
    mask1[a1.topk(k).indices] = 1
    mask2[a2.topk(k).indices] = 1
    return z1 * mask1.reshape(1, -1, 1) + z2 * mask2.reshape(1, -1, 1)


# ---------------------------------------------------------------------------
# codes-domain pair primitives
# ---------------------------------------------------------------------------

def _mix_codes(c1, c2, idx):
    out = torch.empty_like(c1)
    out[:, :idx, :] = c1[:, :idx, :]
    out[:, idx:, :] = c2[:, idx:, :]
    return out


def _blend_codes(c1, c2, seed=0):
    gen = torch.Generator(device=c1.device).manual_seed(seed)
    mask = torch.randint(0, 2, (c1.shape[-2],), device=c1.device, generator=gen)
    mask = mask.reshape(1, -1, 1)
    return torch.where(mask.bool(), c1, c2)


# ---------------------------------------------------------------------------
# activation blend callbacks (encoder / decoder from_activations)
# ---------------------------------------------------------------------------

def _act_max(a1, a2): return torch.where(a1 > a2, a1, a2)
def _act_min(a1, a2): return torch.where(a1 < a2, a1, a2)
def _act_diff(a1, a2): return (a1 - a2) / 2


def _act_stochastic(a1, a2, prob=0.5, seed=0):
    gen = torch.Generator(device=a1.device).manual_seed(seed)
    mask = torch.bernoulli(torch.full((a1.shape[-2],), prob, device=a1.device), generator=gen)
    mask = mask.reshape(1, -1, 1).to(a1.dtype)
    return a1 * mask + a2 * (1 - mask)


# ---------------------------------------------------------------------------
# Normalized PairBendingOp hierarchy
# ---------------------------------------------------------------------------

class PairBendingOp:
    """Base class. apply(interface, audio1, audio2, z1, z2, codes1, codes2) -> Tensor[C, T]."""

    def __init__(self, name: str, target: str, **params):
        self.name = name
        self.target = target
        self.params = params

    @property
    def param_str(self) -> str:
        return '__'.join(f'{k}={v}' for k, v in self.params.items())

    def out_path(self, output_dir: Path, subdir: str, stem1: str, stem2: str) -> Path:
        fname = f"{stem1}_x_{stem2}__{self.name}__{self.param_str}.wav"
        return output_dir / subdir / self.target / fname

    def apply(self, interface, audio1, audio2, z1, z2, codes1, codes2) -> torch.Tensor:
        raise NotImplementedError


class ZPairOp(PairBendingOp):
    def __init__(self, name, fn, **params):
        super().__init__(name, 'z', **params)
        self._fn = fn

    def apply(self, interface, audio1, audio2, z1, z2, codes1, codes2):
        return interface.decode(self._fn(z1, z2, **self.params)).cpu()[0]


class CodesPairOp(PairBendingOp):
    def __init__(self, name, fn, **params):
        super().__init__(name, 'codes', **params)
        self._fn = fn

    def apply(self, interface, audio1, audio2, z1, z2, codes1, codes2):
        codes_bent = self._fn(codes1, codes2, **self.params)
        z_bent, _, _ = interface.model.quantizer.from_codes(codes_bent)
        return interface.decode(z_bent).cpu()[0]


class EncoderPairOp(PairBendingOp):
    def __init__(self, name, act_idx, act_name, blend_fn, **params):
        super().__init__(name, 'encoder', act_idx=act_idx, **params)
        self._act_name = act_name
        self._blend_fn = blend_fn

    def apply(self, interface, audio1, audio2, z1, z2, codes1, codes2):
        act1 = interface.get_activations(self._act_name, audio_data=audio1, fn="encode")[self._act_name]
        act2 = interface.get_activations(self._act_name, audio_data=audio2, fn="encode")[self._act_name]
        mixed = self._blend_fn(act1, act2, **self.params)
        z_bent, *_ = interface.from_activations(
            self._act_name, audio_data=audio1, fn="encode", **{self._act_name: mixed}
        )
        return interface.decode(z_bent).cpu()[0]


class DecoderPairOp(PairBendingOp):
    def __init__(self, name, act_idx, act_name, blend_fn, **params):
        super().__init__(name, 'decoder', act_idx=act_idx, **params)
        self._act_name = act_name
        self._blend_fn = blend_fn

    def apply(self, interface, audio1, audio2, z1, z2, codes1, codes2):
        act1 = interface.get_activations(self._act_name, z=z1, fn="decode")[self._act_name]
        act2 = interface.get_activations(self._act_name, z=z2, fn="decode")[self._act_name]
        mixed = self._blend_fn(act1, act2, **self.params)
        return interface.from_activations(
            self._act_name, z=z1, fn="decode", **{self._act_name: mixed}
        ).cpu()[0]


# ---------------------------------------------------------------------------
# Build ops (shapes determined from a reference encoding)
# ---------------------------------------------------------------------------

def _audio_acts(interface, fn, pattern=r"?add_\d+"):
    return [
        name for name, prop in interface.activations(pattern, fn=fn).items()
        if len(prop.shape) >= 1 and isinstance(prop.shape[-1], torch.SymInt)
    ]


def build_pair_ops(interface, seeds, z_dim, n_quantizers):
    ops = []

    act_blend_fns = [
        ('max',         _act_max,        {}),
        ('min',         _act_min,        {}),
        ('diff',        _act_diff,       {}),
        ('stoch_p0.3',  _act_stochastic, {'prob': 0.3, 'seed': seeds[0]}),
        ('stoch_p0.5',  _act_stochastic, {'prob': 0.5, 'seed': seeds[0]}),
        ('stoch_p0.7',  _act_stochastic, {'prob': 0.7, 'seed': seeds[0]}),
    ]

    # --- z pair ops ---
    for idx in [z_dim // 4, z_dim // 2, 3 * z_dim // 4]:
        ops.append(ZPairOp('mix_z', _mix_z, idx=idx))
    for s in seeds:
        ops.append(ZPairOp('blend_z', _blend_z, seed=s))
    for prob in [0.2, 0.5, 0.8]:
        for s in seeds[:2]:
            ops.append(ZPairOp('stoch_z', _stochastic_mix_z, prob=prob, seed=s))
    for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
        ops.append(ZPairOp('lerp_z', _lerp_z, alpha=alpha))
    for prob in [0.2, 0.5, 0.8]:
        ops.append(ZPairOp('sum_max_z', _sum_max_z, prob=prob))

    # --- codes pair ops ---
    for idx in range(1, n_quantizers):
        ops.append(CodesPairOp('mix_codes', _mix_codes, idx=idx))
    for s in seeds:
        ops.append(CodesPairOp('blend_codes', _blend_codes, seed=s))

    # --- encoder pair ops ---
    enc_acts = _audio_acts(interface, fn="encode")
    target_enc = [(i, enc_acts[i]) for i in [5, 10, 20, 30] if i < len(enc_acts)]
    for idx, act in target_enc:
        for bname, bfn, bparams in act_blend_fns:
            ops.append(EncoderPairOp(f'enc_{bname}', idx, act, bfn, **bparams))

    # --- decoder pair ops ---
    dec_acts = _audio_acts(interface, fn="decode")
    target_dec = [(i, dec_acts[i]) for i in [1, 5, 10, 20] if i < len(dec_acts)]
    for idx, act in target_dec:
        for bname, bfn, bparams in act_blend_fns:
            ops.append(DecoderPairOp(f'dec_{bname}', idx, act, bfn, **bparams))

    return ops


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv):
    random.seed(FLAGS.seed)
    torch.set_grad_enabled(False)

    device = torch.device(FLAGS.device)
    input_dir = Path(FLAGS.input_dir)
    output_dir = Path(FLAGS.output_dir)

    subdirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if not subdirs:
        raise ValueError(f"No sub-folders found in {input_dir}")

    trace_n_samples = max(load_audio(str(d)).max_size() for d in subdirs)

    interface = BendedDescriptAudioCodec(
        model_type=FLAGS.model_type,
        device=device,
        trace_n_samples=trace_n_samples,
        trace_n_batches=1,
    )

    # probe shapes from a dummy encoding
    dummy = torch.zeros(1, 1, trace_n_samples, device=device)
    ref_z, ref_codes, *_ = interface.encode(dummy)
    z_dim, n_quantizers = ref_z.shape[-2], ref_codes.shape[-2]
    del dummy, ref_z, ref_codes

    seeds = [random.randrange(10000) for _ in range(FLAGS.n_seeds)]
    ops = build_pair_ops(interface, seeds, z_dim, n_quantizers)
    print(f"{len(ops)} pairwise bending ops")

    for subdir in subdirs:
        collection = load_audio(str(subdir))
        xs = collection.as_list(channels=1, sr=interface.sample_rate, size=trace_n_samples)
        items = list(zip(collection.keys(), xs))
        pairs = list(combinations(range(len(items)), 2))
        print(f"\n{subdir.name}: {len(items)} files → {len(pairs)} pairs × {len(ops)} ops")

        encoded = []
        for fname, x in items:
            audio = x[None].to(device)
            z, codes, *_ = interface.encode(audio)
            encoded.append((fname, audio, z, codes))

        for i, j in pairs:
            fname1, audio1, z1, codes1 = encoded[i]
            fname2, audio2, z2, codes2 = encoded[j]
            stem1, stem2 = Path(fname1).stem, Path(fname2).stem

            for op in ops:
                out_path = op.out_path(output_dir, subdir.name, stem1, stem2)
                if out_path.exists() and not FLAGS.force:
                    print(f"  skip  {out_path.relative_to(output_dir)} (exists)")
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.cuda.empty_cache()
                try:
                    out_audio = op.apply(interface, audio1, audio2, z1, z2, codes1, codes2)
                    torchaudio.save(str(out_path), out_audio, interface.sample_rate)
                    print(f"  saved {out_path.relative_to(output_dir)}")
                except Exception as e:
                    print(f"  error {op.name} ({stem1} × {stem2}): {e}")

        for _, _, z, codes in encoded:
            del z, codes
        torch.cuda.empty_cache()


if __name__ == '__main__':
    app.run(main)
