#!/usr/bin/env python3
"""Batch DAC audio bending script.

Usage:
    python scripts/bend_dac.py --input_dir data/test --output_dir outs/bend
"""

import random
from pathlib import Path

import torch
import torchaudio
from absl import app, flags

import torchbend
from torchbend.interfaces.descript_interface import BendedDescriptAudioCodec
from torchbend.utils import load_audio

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', None, 'Input folder with audio files', required=True)
flags.DEFINE_string('output_dir', 'outs/bend', 'Output root folder')
flags.DEFINE_string('model_type', '44khz', 'DAC model type: 44khz | 24khz | 16khz')
flags.DEFINE_string('device', 'cuda', 'Torch device')
flags.DEFINE_integer('n_seeds', 4, 'Number of random seeds for stochastic ops')
flags.DEFINE_integer('seed', 42, 'Global random seed')
flags.DEFINE_list('scales', '-1.,-0.1,0.1,1.,1.2,2.,4.', 'Comma-separated scale values for affine ops')
flags.DEFINE_list('biases', '-1.,-0.1,0.1,1.,2.', 'Comma-separated bias values for affine ops')
flags.DEFINE_boolean('force', False, 'Recompute and overwrite files that already exist')


# ---------------------------------------------------------------------------
# z-domain primitives
# ---------------------------------------------------------------------------

def _mask_dim(z, prob, seed=0):
    gen = torch.Generator(device=z.device).manual_seed(seed)
    mask = torch.bernoulli(torch.full((z.shape[-2],), prob, device=z.device), generator=gen)
    return z * mask.reshape(1, -1, 1)


def _mask_extreme_dims(z, prob, largest=True):
    k = max(1, int(prob * z.shape[-2]))
    indices = z.abs().mean(-1).mean(0).topk(k, largest=largest).indices
    mask = torch.ones(z.shape[-2], device=z.device, dtype=z.dtype)
    mask[indices] = 0
    return z * mask.reshape(1, -1, 1)


def _permute_dims(z, seed=0):
    gen = torch.Generator(device=z.device).manual_seed(seed)
    perm = torch.randperm(z.shape[-2], device=z.device, generator=gen)
    return z[:, perm, :]


def _noise_dims(z, prob, std=1.0, seed=0):
    gen = torch.Generator(device=z.device).manual_seed(seed)
    mask = torch.bernoulli(torch.full((z.shape[-2],), prob, device=z.device), generator=gen)
    noise = torch.randn(z.shape, device=z.device, generator=gen) * std
    return z + noise * mask.reshape(1, -1, 1)


def _affine(z, scale=1.0, bias=0.0):
    return scale * z + bias


# ---------------------------------------------------------------------------
# codes-domain primitives
# ---------------------------------------------------------------------------

def _zero_codes_up_to(codes, idx):
    out = codes.clone()
    out[:, :idx, :] = 0
    return out


def _zero_codes_down_to(codes, idx):
    out = codes.clone()
    out[:, idx:, :] = 0
    return out


def _stretch_z(z, factor):
    return torch.nn.functional.interpolate(z, scale_factor=factor, mode='linear')


def _corrupt_codes(codes, prob, n_codes=1024, seed=0):
    gen = torch.Generator(device=codes.device).manual_seed(seed)
    mask = torch.bernoulli(
        torch.full(codes.shape, prob, device=codes.device, dtype=torch.float),
        generator=gen,
    ).bool()
    random_codes = torch.randint(0, n_codes, codes.shape, device=codes.device, generator=gen)
    return torch.where(mask, random_codes, codes)


# ---------------------------------------------------------------------------
# Normalized BendingOp hierarchy
# ---------------------------------------------------------------------------

class BendingOp:
    """Base class. Subclasses implement apply(interface, audio, z, codes) -> Tensor[C,T]."""

    def __init__(self, name: str, target: str, **params):
        self.name = name
        self.target = target
        self.params = params

    @property
    def param_str(self) -> str:
        return '__'.join(f'{k}={v}' for k, v in self.params.items())

    def out_path(self, output_dir: Path, stem: str) -> Path:
        fname = f"{stem}__{self.name}__{self.param_str}.wav"
        return output_dir / self.target / fname

    def apply(self, interface, audio, z, codes) -> torch.Tensor:
        raise NotImplementedError


class ZOp(BendingOp):
    """Transforms z then decodes."""

    def __init__(self, name, fn, **params):
        super().__init__(name, 'z', **params)
        self._fn = fn

    def apply(self, interface, audio, z, codes):
        return interface.decode(self._fn(z, **self.params)).cpu()[0]


class CodesOp(BendingOp):
    """Transforms codes, re-quantizes then decodes."""

    def __init__(self, name, fn, **params):
        super().__init__(name, 'codes', **params)
        self._fn = fn

    def apply(self, interface, audio, z, codes):
        codes_bent = self._fn(codes, **self.params)
        z_bent, _, _ = interface.model.quantizer.from_codes(codes_bent)
        return interface.decode(z_bent).cpu()[0]


class EncoderOp(BendingOp):
    """Bends an intermediate encoder activation via torchbend callback."""

    def __init__(self, name, act_idx, act_name, cb, **params):
        super().__init__(name, 'encoder', act_idx=act_idx, **params)
        self._act_name = act_name
        self._cb = cb

    def apply(self, interface, audio, z, codes):
        interface.reset()
        interface.bend(self._cb, self._act_name, fn="encode")
        z_bent, *_ = interface.encode(audio)
        out = interface.decode(z_bent).cpu()[0]
        interface.reset()
        return out


class DecoderOp(BendingOp):
    """Bends an intermediate decoder activation via torchbend callback."""

    def __init__(self, name, act_idx, act_name, cb, **params):
        super().__init__(name, 'decoder', act_idx=act_idx, **params)
        self._act_name = act_name
        self._cb = cb

    def apply(self, interface, audio, z, codes):
        interface.reset()
        interface.bend(self._cb, self._act_name, fn="decode")
        out = interface.decode(z).cpu()[0]
        interface.reset()
        return out


# ---------------------------------------------------------------------------
# Build the full op sweep
# ---------------------------------------------------------------------------

def _audio_acts(interface, fn, pattern=r"?add_\d+"):
    return [
        name for name, prop in interface.activations(pattern, fn=fn).items()
        if len(prop.shape) >= 1 and isinstance(prop.shape[-1], torch.SymInt)
    ]


def build_ops(interface, seeds, scales, biases):
    ops = []
    scale_bias_grid = [(s, b) for s in scales for b in biases]

    # --- z ops ---
    for prob in [0.1, 0.5, 0.8]:
        ops.append(ZOp('mask_dim', _mask_dim, prob=prob, seed=seeds[0]))
    for prob in [0.1, 0.5, 0.8]:
        ops.append(ZOp('mask_extreme_max', _mask_extreme_dims, prob=prob, largest=True))
        ops.append(ZOp('mask_extreme_min', _mask_extreme_dims, prob=prob, largest=False))
    for s in seeds:
        ops.append(ZOp('permute', _permute_dims, seed=s))
    for prob, std in [(0.3, 0.5), (0.7, 1.0), (1.0, 2.0)]:
        ops.append(ZOp('noise', _noise_dims, prob=prob, std=std, seed=seeds[0]))
    for scale, bias in scale_bias_grid:
        ops.append(ZOp('affine', _affine, scale=scale, bias=bias))

    # --- codes ops ---
    for prob in [0.1, 0.5, 0.8]:
        ops.append(CodesOp('corrupt', _corrupt_codes, prob=prob, seed=seeds[0]))
    n_quantizers = 9  # DAC 44khz default
    for idx in range(1, n_quantizers):
        ops.append(CodesOp('zero_up_to', _zero_codes_up_to, idx=idx))
    for idx in range(1, n_quantizers):
        ops.append(CodesOp('zero_down_to', _zero_codes_down_to, idx=idx))

    # --- z time-stretch ops ---
    for factor in [0.5, 1.0, 2.0, 4.0, 8.0]:
        ops.append(ZOp('stretch', _stretch_z, factor=factor))

    # --- encoder ops ---
    enc_acts = _audio_acts(interface, fn="encode")
    target_enc = [(i, enc_acts[i]) for i in [3, 5, 7, 10, 15, 24, 30, 45] if i < len(enc_acts)]
    for idx, act in target_enc:
        for prob in [0.1, 0.5, 0.8]:
            p = torchbend.BendingParameter(name="prob", value=prob)
            cb = torchbend.ThresholdActivation(threshold=p, dim=-2)
            ops.append(EncoderOp('enc_threshold', idx, act, cb, prob=prob))
        for scale, bias in scale_bias_grid:
            sp = torchbend.BendingParameter(name="scale", value=scale)
            bp = torchbend.BendingParameter(name="bias", value=bias)
            ops.append(EncoderOp('enc_affine', idx, act, torchbend.Affine(scale=sp, bias=bp), scale=scale, bias=bias))

    # --- decoder ops ---
    dec_acts = _audio_acts(interface, fn="decode")

    # static-across-time channel noise on decoder's direct input (first activation)
    if dec_acts:
        for std in [0.1, 0.5, 1.0, 2.0, 5.0]:
            cb = torchbend.Normal(std=torchbend.BendingParameter(name="std", value=std), seed=seeds[0], dim=-2)
            ops.append(DecoderOp('static_channel_noise', 0, dec_acts[0], cb, std=std))

    target_dec = [(i, dec_acts[i]) for i in [1, 3, 7, 10, 13, 15, 17, 20] if i < len(dec_acts)]
    for idx, act in target_dec:
        for prob in [0.1, 0.5, 0.8]:
            p = torchbend.BendingParameter(name="prob", value=prob)
            cb = torchbend.ThresholdActivation(threshold=p, dim=-2)
            ops.append(DecoderOp('dec_threshold', idx, act, cb, prob=prob))
        for scale, bias in scale_bias_grid:
            sp = torchbend.BendingParameter(name="scale", value=scale)
            bp = torchbend.BendingParameter(name="bias", value=bias)
            ops.append(DecoderOp('dec_affine', idx, act, torchbend.Affine(scale=sp, bias=bp), scale=scale, bias=bias))

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

    audio_files = load_audio(str(input_dir))
    trace_n_samples = audio_files.max_size()

    interface = BendedDescriptAudioCodec(
        model_type=FLAGS.model_type,
        device=device,
        trace_n_samples=trace_n_samples,
        trace_n_batches=1,
    )

    seeds = [random.randrange(10000) for _ in range(FLAGS.n_seeds)]
    scales = [float(v) for v in FLAGS.scales]
    biases = [float(v) for v in FLAGS.biases]
    ops = build_ops(interface, seeds, scales, biases)
    print(f"{len(ops)} bending ops × {len(audio_files)} files = {len(ops) * len(audio_files)} generations")

    xs = audio_files.as_list(channels=1, sr=interface.sample_rate, size=trace_n_samples)

    for (fname, _, _), x in zip(audio_files, xs):
        stem = Path(fname).stem
        audio = x[None].to(device)
        z, codes, *_ = interface.encode(audio)

        for op in ops:
            out_path = op.out_path(output_dir, stem)
            if out_path.exists() and not FLAGS.force:
                print(f"  skip  {out_path.relative_to(output_dir)} (exists)")
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.empty_cache()
            try:
                out_audio = op.apply(interface, audio, z, codes)
                torchaudio.save(str(out_path), out_audio, interface.sample_rate)
                print(f"  saved {out_path.relative_to(output_dir)}")
            except Exception as e:
                print(f"  error {op.name} ({fname}): {e}")

        del z, codes, audio


if __name__ == '__main__':
    app.run(main)
