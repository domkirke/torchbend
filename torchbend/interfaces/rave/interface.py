import torch
import copy
import os
import torchaudio
import re
from typing import Union, Optional
import rave as ravelib
from ..base import Interface
from ...tracing import BendedModule
from .scripting import *
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


def _zero_cache(module, filters=[r".*cache", r".*pad"]):
    for k, v in module.named_buffers():
        if True in [re.match(f, k) is not None for f in filters]:
            v.data.zero_()

class BendedRAVE(Interface):
    _imported_callbacks_ = []
    _proxied_buffers = ['.*pad', '.*cache', 'latent_mean', 'latent_pca']

    def __init__(self, model_path, batch_size=4):
        model = self.load_model(model_path)
        self.batch_size = batch_size
        model(torch.zeros(self.batch_size, 1, 8192))
        self.model = model

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

    def encode(self, x: Union[torch.Tensor, str]):
        if isinstance(x, str):
            x = self.load_audio(x)
        decoder_out = self._model.encode(x)
        return self._model.encoder.reparametrize(decoder_out)[:2][0]

    def decode(self, z: torch.Tensor, out: Optional[str] = None):
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
        return module


__all__ = ['BendedRAVE', 'script_rave_model']