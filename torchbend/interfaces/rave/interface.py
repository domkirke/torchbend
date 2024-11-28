import torch
from types import MethodType
import copy
import os
import torchaudio
import re
from typing import Union, Optional
import rave as ravelib
from ..base import Interface
from ...tracing import BendedModule
from .scripting import *
from .nntilde import ScriptableRAVE, _zero_cache
import cached_conv as cc
import gin

class BendingRAVEException(Exception):
    pass


def script_rave_model(pretrained, **kwargs):
    cc.use_cached_conv(True)
    if isinstance(pretrained.encoder, ravelib.blocks.VariationalEncoder):
        script_class = VariationalScriptedRAVE
    elif isinstance(pretrained.encoder, ravelib.blocks.DiscreteEncoder):
        script_class = DiscreteScriptedRAVE
    elif isinstance(pretrained.encoder, ravelib.blocks.WasserteinEncoder):
        script_class = WasserteinScriptedRAVE
    elif isinstance(pretrained.encoder, ravelib.blocks.SphericalEncoder):
        script_class = SphericalScriptedRAVE
    else:
        raise ValueError(f"Encoder type {type(pretrained.encoder)} "
                        "not supported for export.")
    scripted_model = script_class(pretrained=pretrained, **kwargs)
    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)
    return scripted_model

def _scripted_model_to_nntilde(self):
    return ScriptableRAVE(self)


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



class BendedRAVE(Interface):
    _imported_callbacks_ = []
    _proxied_buffers = ['.*pad', '.*cache', 'latent_mean', 'latent_pca']

    def __init__(self, model_path, batch_size=4):
        model = self.load_model(model_path)
        self.batch_size = batch_size
        model(torch.zeros(self.batch_size, 1, 8192))
        self.model = model
        self.pre_process_latent = MethodType(pre_process_fn[type(self.model.encoder)], self)
        self.post_process_latent = MethodType(post_process_fn[type(self.model.encoder)], self)

    @staticmethod
    def load_model(model_path):
        try:
            if (not os.path.isfile(model_path)) or (os.path.splitext(model_path)[1] == ".ckpt"):
                return BendedRAVE.load_checkpoint(model_path)
            else:
                return BendedRAVE.load_scripted(model_path)
        except Exception:
            raise BendingRAVEException("Could not load model %s ; does not seem to be a valid file."%model_path)

    @staticmethod
    def load_checkpoint(model_path):
        cc.use_cached_conv(True)
        config_path = ravelib.core.search_for_config(model_path)
        if config_path is None:
            raise BendingRAVEException('config not found in folder %s'%model_path)
        gin.parse_config_file(config_path)
        model = ravelib.RAVE()
        run = ravelib.core.search_for_run(model_path)
        if run is None:
            raise BendingRAVEException("run not found in folder %s"%model_path)
        model = model.load_from_checkpoint(run)
        return model

    @staticmethod
    def load_scripted(model_path):
        model = torch.jit.load(model_path)
        return model

    def _bend_model(self, model):
        self._model = BendedModule(model)
        self._import_methods(self._model)
        self._model.trace("forward", x=torch.zeros(4, 1, 48000),  _proxied_buffers=self._proxied_buffers)
        _, (decoder_out,) = self._model.trace("encode", x=torch.zeros(4, 1, 48000), _proxied_buffers=self._proxied_buffers, _return_out=True)
        latent_out = self._model.encoder.reparametrize(decoder_out)[:2][0]
        self._model.trace("decode", z=latent_out, _proxied_buffers=self._proxied_buffers)

    def load_audio(self, path: str):
        audio, sr = torchaudio.load(path)
        if sr != self._model.sr:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        return audio

    @property
    def sample_rate(self):
        return self._model.sr

    def write_audio(self, path, audio):
        torchaudio.save(path, audio, self.sample_rate)

    def forward(self, x: Union[torch.Tensor, str], out: Optional[str] = None):
        if isinstance(x, str):
            x = self.load_audio(x)
        audio = self._model.forward(x)
        if out is not None: self.write_audio(out, audio[0])
        return audio

    @property
    def encoder(self):
        return BendedModule(self._model.encoder)
    @property
    def decoder(self):
        return BendedModule(self._model.decoder)
    @property
    def discriminator(self):
        return BendedModule(self._model.discriminator)

    @property
    def latent_size(self):
        return self.model.latent_size

    def pre_process_latent(self, z):
        raise NotImplementedError 

    def post_process_latent(self, z):
       raise NotImplementedError 

    def get_dims_for_fidelity(self, fidelity: float):
        latent_size = max(np.argmax(self.model.fidelity.numpy() > fidelity), 1)
        return latent_size

    def get_fidelity_for_dims(self, dims: int):
        return self.model.fidelity[dims]

    def encode(self, x: Union[torch.Tensor, str], postprocess=False):
        if isinstance(x, str):
            x = self.load_audio(x)
        decoder_out = self._model.encode(x)
        z = self._model.encoder.reparametrize(decoder_out)[:2][0]
        if postprocess: 
            z = self.post_process_latent(z)
        return z

    def decode(self, z: torch.Tensor, out: Optional[str] = None, preprocess=False):
        if preprocess: 
            z = self.pre_process_latent(z)
        audio = self._model.decode(z)
        if out is not None: self.write_audio(out, audio[0])
        return audio

    def script(self):
        scripted_model = script_rave_model(self._model)
        module = BendedModule(scripted_model)
        module.trace("encode", x=torch.randn(self.batch_size, 1, 8192), _proxied_buffers=self._proxied_buffers, _no_tensor_for_args=True)#, ".*cache.pad"])
        module.trace("decode", z=torch.randn(self.batch_size, scripted_model.latent_size, 8), _proxied_buffers=self._proxied_buffers, _no_tensor_for_args=True)#, 'decode_params', ".*cache.pad"])
        module.trace("forward", x=torch.randn(self.batch_size, 1, 8192), _proxied_buffers=self._proxied_buffers, _no_tensor_for_args=True)#, ".*cache.pad"])
        _zero_cache(module)
        setattr(module, "nntilde", MethodType(_scripted_model_to_nntilde, module))
        return module


__all__ = ['BendedRAVE', 'script_rave_model']