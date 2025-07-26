import os, sys, math
import torch, torch.nn as nn

import cached_conv as cc
import nn_tilde
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple

import rave
import rave.blocks
import rave.core
import rave.resampler
from rave.prior import model as prior

from torchbend import mark


class DumbPrior(nn.Module):
    def forward(self, x: torch.Tensor):
        return x


        
@torch.fx.wrap
def get_noise(z: torch.Tensor, full_latent_size: int):
    return torch.randn(
            z.shape[0],
            full_latent_size - z.shape[1],
            z.shape[-1],
    ).type_as(z)




def script_rave_model(pretrained, **kwargs):
    cc.use_cached_conv(True)
    if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
        script_class = VariationalScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
        script_class = DiscreteScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
        script_class = WasserteinScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.SphericalEncoder):
        script_class = SphericalScriptedRAVE
    else:
        raise ValueError(f"Encoder type {type(pretrained.encoder)} "
                        "not supported for export.")
    scripted_model = script_class(pretrained=pretrained, **kwargs)
    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)
    return scripted_model



def post_process_variational(self, z):
    z = z - self.model.latent_mean.unsqueeze(-1)
    z = F.conv1d(z, self.model.latent_pca.unsqueeze(-1))
    return z

def pre_process_variational(self, z):
    z = F.conv1d(z, self.model.latent_pca.T.unsqueeze(-1))
    z = z + self.model.latent_mean.unsqueeze(-1)
    return z

def post_process_discrete(self, z):
    z = self.model.encoder.rvq.encode(z)
    return z.float()

def pre_process_discrete(self, z):
    z = torch.clamp(z, 0,
                    self.encoder.rvq.layers[0].codebook_size - 1).long()
    z = self.model.encoder.rvq.decode(z)
    if self.model.encoder.noise_augmentation:
        noise = torch.randn(z.shape[0], self.model.encoder.noise_augmentation,
                            z.shape[-1]).type_as(z)
        z = torch.cat([z, noise], 1)
    return z

def post_process_ws(self, z):
    return z

def pre_process_ws(self, z):
    if self.model.encoder.noise_augmentation:
        noise = torch.randn(z.shape[0], self.model.encoder.noise_augmentation,
                            z.shape[-1]).type_as(z)
        z = torch.cat([z, noise], 1)
    return z

def post_process_sph(self, z):
    return rave.blocks.unit_norm_vector_to_angles(z)

def pre_process_sph(self, z):
    return rave.blocks.angles_to_unit_norm_vector(z)

pre_process_fn = {rave.blocks.VariationalEncoder: pre_process_variational, 
                  rave.blocks.DiscreteEncoder: pre_process_discrete, 
                  rave.blocks.WasserteinEncoder: pre_process_ws, 
                  rave.blocks.SphericalEncoder: pre_process_sph
                 }

post_process_fn = {rave.blocks.VariationalEncoder: post_process_variational, 
                  rave.blocks.DiscreteEncoder: post_process_discrete, 
                  rave.blocks.WasserteinEncoder: post_process_ws, 
                  rave.blocks.SphericalEncoder: post_process_sph
                 }            



class ScriptedRAVE(nn_tilde.Module):

    _attributes_for_tb_scripting = ['projections_names', 'projections_types']

    def __init__(self,
                 pretrained: rave.RAVE,
                 channels: Optional[int] = None,
                 fidelity: float = .95,
                 target_sr: bool = None, 
                 prior: prior.Prior = None,
                 latent_size: int = None) -> None:

        super().__init__()
        self.pqmf = pretrained.pqmf
        self.sr = pretrained.sr
        self.spectrogram = pretrained.spectrogram
        self.resampler = None
        self.input_mode = pretrained.input_mode
        self.output_mode = pretrained.output_mode
        self.n_channels = pretrained.n_channels
        self.target_channels = channels or self.n_channels
        self.stereo_mode = False

        if target_sr is not None:
            if target_sr != self.sr:
                assert not target_sr % self.sr, "Incompatible target sampling rate"
                self.resampler = rave.resampler.Resampler(target_sr, self.sr)
                self.sr = target_sr

        self.full_latent_size = pretrained.latent_size
        self.is_using_adain = False
        for m in self.modules():
            if isinstance(m, rave.blocks.AdaptiveInstanceNormalization):
                self.is_using_adain = True
                break
        if self.is_using_adain and (self.n_channels != self.target_channels):
            raise ValueError("AdaIN requires the original number of channels")

        self.register_attribute("learn_target", False)
        self.register_attribute("reset_target", False)
        self.register_attribute("learn_source", False)
        self.register_attribute("reset_source", False)
        self.register_attribute("temperature", 1.0)
        self.register_buffer("_temperature", torch.tensor(1.))

        # projection handling
        self.register_attribute("projection", "default")
        self._has_projections = hasattr(pretrained, "projections")
        self.projections_names = ['default']
        self.projections = nn.ParameterList([torch.cat([pretrained.latent_mean[None], pretrained.fidelity[None], pretrained.latent_pca], 0)])
        self.projections_types = [0]
        if self._has_projections:
            for k, v in pretrained.projections.items():
                if k.startswith('pca'):
                    self.projections_types.append(0)
                elif k.startswith('ica'):
                    self.projections_types.append(1)
                elif k.startswith('mapper'):
                    self.projections_types.append(2)
                self.projections_names.append(k)
                self.projections.append(v)

        receptive_field = getattr(pretrained, "receptive_field", None)
        self.register_attribute("receptive_field", (int(receptive_field[0]), int(receptive_field[1])))
        self.register_buffer("latent_pca", pretrained.latent_pca)
        self.register_buffer("latent_mean", pretrained.latent_mean)
        self.register_buffer("fidelity", pretrained.fidelity)

        if latent_size is None:
            if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
                if fidelity is None:
                    latent_size = self.full_latent_size
                else:
                    latent_size = max(
                        np.argmax(pretrained.fidelity.numpy() > fidelity), 1)
                    latent_size = 2**math.ceil(math.log2(latent_size))

            elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
                latent_size = pretrained.encoder.num_quantizers

            elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
                latent_size = pretrained.latent_size

            elif isinstance(pretrained.encoder, rave.blocks.SphericalEncoder):
                latent_size = pretrained.latent_size - 1

            else:
                raise ValueError(
                    f'Encoder type {pretrained.encoder.__class__.__name__} not supported'
                )
        else:
            assert latent_size > 0, "latent_size must be positive."

        self.latent_size = latent_size
        self.fake_adain = rave.blocks.AdaptiveInstanceNormalization(0)

        # have to init cached conv before graphing
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder
        x_len = 2**14
        x = torch.zeros(1, self.n_channels, x_len)
        z = self.encode(x)
        self.ratio_encode = x_len // z.shape[-1]

        # configure encoder
        if (pretrained.input_mode == "pqmf") or (pretrained.output_mode == "pqmf"):
            # scripting fails if cached conv is not initialized
            self.pqmf(torch.zeros(1, 1, x_len))

        self.prior = prior
        self._has_prior = prior is not None


    def post_process_latent(self, z):
        raise NotImplementedError

    def pre_process_latent(self, z):
        raise NotImplementedError

    def init_cache(self, x):
        self.forward(x)
        if self.pqmf is not None:
            u = self.pqmf(x.reshape(-1, 1, x.shape[-1]))
            v = self.pqmf.inverse(u)

    def update_adain(self):
        for m in self.modules():
            if isinstance(m, rave.blocks.AdaptiveInstanceNormalization):
                m.learn_x.zero_()
                m.learn_y.zero_()

                if self.learn_target[0]:
                    m.learn_y.add_(1)
                if self.learn_source[0]:
                    m.learn_x.add_(1)

                if self.reset_target[0]:
                    m.reset_y()
                if self.reset_source[0]:
                    m.reset_x()

        self.reset_source = False,
        self.reset_target = False,

    @torch.jit.export
    def set_stereo_mode(self, stereo):
        self.stereo_mode = bool(stereo);

    @torch.jit.export
    def set_projection(self, projection: str) -> int:
        # if not self._has_projections: 
            # return -1
        for i, k in enumerate(self.projections_names):
            if projection == k:
                self.projection = projection,
                return 0
        return -1
        
    def get_projection(self) -> str:
        return self.projection

    @torch.jit.export
    def encode(self, x):
        if self.stereo_mode:
            if self.n_channels == 1:
                x = x[:, 0].unsqueeze(0)
            elif self.n_channels > 2:
                raise RuntimeError("stereo mode is not available when n_channels > 2")

        if self.is_using_adain:
            self.update_adain()

        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        batch_size = x.shape[:-2]
        if self.input_mode == "pqmf":
            x = x.reshape(-1, 1, x.shape[-1])
            x = self.pqmf(x)
            x = x.reshape(batch_size + (-1, x.shape[-1]))
        elif self.input_mode == "mel":
            if self.spectrogram is not None:
                x = self.spectrogram(x)[..., :-1]
                x = torch.log1p(x).reshape(batch_size + (-1, x.shape[-1]))
            else:
                raise RuntimeError()
        z = self.encoder(x)
        z = self.post_process_latent(z)
        return z


    @torch.jit.export
    def decode(self, z, from_forward: bool = False):
        if self.is_using_adain and not from_forward:
            self.update_adain()
        n_batch = z.shape[0]
        if self.stereo_mode:
            n_batch = int(n_batch / 2)

        if self.target_channels > self.n_channels:
            z = z.repeat(math.ceil(self.target_channels / self.n_channels), 1, 1)[:self.target_channels]

        z = self.pre_process_latent(z)
        y = self.decoder(z)

        batch_size = z.shape[:-2]
        if self.output_mode == "pqmf":
            y = y.reshape(y.shape[0] * self.n_channels, -1, y.shape[-1])
            y = self.pqmf.inverse(y)
            y = y.reshape(batch_size+(self.n_channels, -1))

        if self.resampler is not None:
            y = self.resampler.from_model_sampling_rate(y)

        # if (output-) padding is scrambled
        if y.shape[-1] > z.shape[-1] * self.ratio_encode:
            y = y[..., :z.shape[-1] * self.ratio_encode]

        if self.stereo_mode:
            y = torch.cat([y[:n_batch], y[n_batch:]], 1)
        elif self.target_channels > self.n_channels:
            y = torch.cat(y.chunk(self.target_channels, 0), 1)
        elif self.target_channels < self.n_channels:
            y = y[:, :self.target_channels]
        return y

    def forward(self, x):
        return self.decode(self.encode(x), from_forward=True)

    @torch.jit.export
    def get_learn_target(self) -> bool:
        return self.learn_target[0]

    @torch.jit.export
    def set_learn_target(self, learn_target: bool) -> int:
        self.learn_target = (learn_target, )
        return 0

    @torch.jit.export
    def get_learn_source(self) -> bool:
        return self.learn_source[0]

    @torch.jit.export
    def set_learn_source(self, learn_source: bool) -> int:
        self.learn_source = (learn_source, )
        return 0

    @torch.jit.export
    def get_reset_target(self) -> bool:
        return self.reset_target[0]

    @torch.jit.export
    def set_reset_target(self, reset_target: bool) -> int:
        self.reset_target = (reset_target, )
        return 0

    @torch.jit.export
    def get_reset_source(self) -> bool:
        return self.reset_source[0]

    @torch.jit.export
    def set_reset_source(self, reset_source: bool) -> int:
        self.reset_source = (reset_source, )
        return 0
    
    @torch.jit.export
    def get_temperature(self) -> Tuple[float]:
        return float(self._temperature),

    @torch.jit.export 
    def set_temperature(self, temperature: float) -> int:
        if temperature < 0: 
            return -1
        self._temperature.set_(torch.tensor(temperature))
        return 0

    @torch.jit.export
    def get_receptive_field(self) -> Tuple[int, int]:
        return self.receptive_field

    @torch.jit.export 
    def set_receptive_field(self, dimred: str) -> int:
        return -1


    @torch.jit.export
    def prior(self, temp: torch.Tensor):
        if self._has_prior:
            return self.prior_module.forward(temp)
        else:
            return torch.tensor(0)
        


class VariationalScriptedRAVE(ScriptedRAVE):

    def __init__(self, *args, use_pca=True, **kwargs):
        self.use_pca = use_pca
        super(VariationalScriptedRAVE, self).__init__(*args, **kwargs)
        if self.use_pca:
            assert self.latent_size <= self.full_latent_size, "latent_size must be <= %d"%(self.full_latent_size)

    @torch.jit.export
    def encode_full(self, x):
        if self.stereo_mode:
            if self.n_channels == 1:
                x = x[:, 0].unsqueeze(0)
            elif self.n_channels > 2:
                raise RuntimeError("stereo mode is not available when n_channels > 2")

        if self.is_using_adain:
            self.update_adain()

        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        batch_size = x.shape[:-2]
        if self.input_mode == "pqmf":
            x = x.reshape(-1, 1, x.shape[-1])
            x = self.pqmf(x)
            x = x.reshape(batch_size + (-1, x.shape[-1]))
        elif self.input_mode == "mel":
            if self.spectrogram is not None:
                x = self.spectrogram(x)[..., :-1]
                x = torch.log1p(x).reshape(batch_size + (-1, x.shape[-1]))
            else:
                raise RuntimeError()
        z = self.encoder(x)
        z = self.post_process_latent_full(z)
        return z


    @torch.jit.export
    def decode_full(self, z, from_forward: bool = False):

        z = self.pre_process_latent_full(z)

        n_batch = z.shape[0]
        if self.stereo_mode:
            n_batch = int(n_batch / 2)

        y = self.decoder(z)

        batch_size = z.shape[:-2]
        if self.output_mode == "pqmf":
            y = y.reshape(y.shape[0] * self.n_channels, -1, y.shape[-1])
            y = self.pqmf.inverse(y)
            y = y.reshape(batch_size+(self.n_channels, -1))

        if self.resampler is not None:
            y = self.resampler.from_model_sampling_rate(y)

        # if (output-) padding is scrambled
        if y.shape[-1] > z.shape[-1] * self.ratio_encode:
            y = y[..., :z.shape[-1] * self.ratio_encode]

        if self.stereo_mode:
            y = torch.cat([y[:n_batch], y[n_batch:]], 1)
        elif self.target_channels > self.n_channels:
            y = torch.cat(y.chunk(self.target_channels, 0), 1)
        elif self.target_channels < self.n_channels:
            y = y[:, :self.target_channels]
        return y

    
    def forward(self, x):
        return self.decode_full(self.encode_full(x), from_forward=True)

    @torch.jit.export
    def encode_dist(self, x):
        if self.stereo_mode:
            if self.n_channels == 1:
                x = x[:, 0].unsqueeze(0)
            elif self.n_channels > 2:
                raise RuntimeError("stereo mode is not available when n_channels > 2")

        if self.is_using_adain:
            self.update_adain()

        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        batch_size = x.shape[:-2]
        if self.input_mode == "pqmf":
            x = x.reshape(-1, 1, x.shape[-1])
            x = self.pqmf(x)
            x = x.reshape(batch_size + (-1, x.shape[-1]))
        elif self.input_mode == "mel":
            if self.spectrogram is not None:
                x = self.spectrogram(x)[..., :-1]
                x = torch.log1p(x).reshape(batch_size + (-1, x.shape[-1]))
            else:
                raise RuntimeError()
        z = self.encoder(x)

        # parse mean and scale
        z, scale = z.chunk(2, 1)
        std = nn.functional.softplus(scale) + 1e-4
        z = z - self.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, self.latent_pca.unsqueeze(-1))
        z = z[:, :self.latent_size]
        std = F.conv1d(std*std, self.latent_pca.unsqueeze(-1).pow(2)).sqrt()
        std = std[:, :self.latent_size]
        return torch.cat([z, std], dim=-2)

    def post_process_latent_full(self, z):
        z = self.encoder.reparametrize(z, temperature=self._temperature)[0]
        if self.use_pca:
            z = z - self.latent_mean.unsqueeze(-1)
            z = F.conv1d(z, self.latent_pca.unsqueeze(-1))
        return z

    def post_process_latent(self, z):
        z = self.post_process_latent_full(z)
        z = z[:, :self.latent_size]
        return z

    def pre_process_latent_full(self, z):
        if self.use_pca:
            z = F.conv1d(z, self.latent_pca.T.unsqueeze(-1))
            z = z + self.latent_mean.unsqueeze(-1)
        return z

    def pre_process_latent(self, z):
        if z.shape[1] < self.full_latent_size:
            noise = get_noise(z, self.full_latent_size)
            z = torch.cat([z, noise * self._temperature], 1)
        z = self.pre_process_latent_full(z) 
        return z



class DiscreteScriptedRAVE(ScriptedRAVE):

    def __init__(self, *args, **kwargs):
        super(DiscreteScriptedRAVE, self).__init__(*args,**kwargs)
        assert self.latent_size < self.full_latent_size, "latent_size must be < %d"%(self.full_latent_size)



class WasserteinScriptedRAVE(ScriptedRAVE):

    def __init__(self, *args, **kwargs):
        super(WasserteinScriptedRAVE, self).__init__(*args,**kwargs)
        assert self.latent_size < self.full_latent_size, "latent_size must be < %d"%(self.full_latent_size)


class SphericalScriptedRAVE(ScriptedRAVE):

    def __init__(self, *args, **kwargs):
        super(SphericalScriptedRAVE, self).__init__(*args,**kwargs)
        assert self.latent_size < self.full_latent_size, "latent_size must be < %d"%(self.full_latent_size)


class TraceModel(nn.Module):
    def __init__(self, pretrained: prior.Prior, model: rave.RAVE):
        super().__init__()
        pretrained._jit_is_scripting = True
        self.pretrained = pretrained
        self.latent_size = pretrained.latent_size

        x = torch.zeros(1, self.pretrained.n_channels, 2**14)
        z = model.encode(x)
        z = pretrained.post_process_latent(z)
        self.ratio = x.shape[-1] // z.shape[-1]

        # self.register_buffer(
        #     "forward_params",
        #     torch.tensor([1, self.ratio, self.latent_size, self.ratio]),
        # )

        self.pretrained.synth = None

        self.register_buffer(
            "previous_step",
            self.pretrained.quantized_normal.encode(
                torch.zeros(1, self.latent_size, 1)),
        )

        self.pre_diag_cache = cc.CachedPadding1d(self.latent_size - 1)
        self.pre_diag_cache(z)
        self.pre_diag_cache = torch.jit.script(self.pre_diag_cache)

    def step_forward(self, temp):
        # PREDICT NEXT STEP
        x = self.pretrained.forward(self.previous_step)
        x = x / temp
        x = self.pretrained.post_process_prediction(x, argmax=False)
        # self.previous_step.copy_(x.clone())
        # self.previous_step = x.clone()

        # DECODE AND SHIFT PREDICTION
        x = self.pretrained.quantized_normal.decode(x)
        x = self.pre_diag_cache(x)
        x = self.pretrained.diagonal_shift.inverse(x)
        return x

    def forward(self, temp: torch.Tensor):
        x = torch.zeros(
            temp.shape[0],
            self.latent_size,
            temp.shape[-1],
        ).to(temp)

        temp = temp.mean(-1, keepdim=True)
        temp = nn.functional.softplus(temp) / math.log(2)

        for i in range(x.shape[-1]):
            x[..., i:i + 1] = self.step_forward(temp)

        return x