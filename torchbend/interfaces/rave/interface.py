import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from types import MethodType
import copy
import os
import torchaudio
import re
from typing import Union, Optional
import rave as ravelib
from ..base import Interface
from ...tracing import BendedModule
from .scripting import pre_process_fn, post_process_fn, script_rave_model
from .nntilde import ScriptableRAVE, _zero_cache
import cached_conv as cc
import gin


class BendingRAVEException(Exception):
    pass


def _scripted_model_to_nntilde(self):
    return ScriptableRAVE(self)


class BendedRAVEImportException(Exception):
    def __init__(self, path, msg=None):
        super().__init__()
        self.path = path
        self.msg = msg
    def __repr__(self):
        s = f"RAVEImportException(path={self.path}"
        if self.reason is not None:
            return s+f", reason={self.msg})"
        else:
            return s+"p)"


def _rave_get_model_paths_from_ckpt(path):
    ckpt_path = path
    ckpt_dir = Path(path).parent
    if (ckpt_dir / "config.gin").exists(): 
        return {'ckpt': str(ckpt_path), 'config': str(ckpt_dir / "config.gin")}
    elif (ckpt_dir / ".." / "config.gin").exists(): 
        return {'ckpt': str(ckpt_path), 'config': str(ckpt_dir / ".." / "config.gin")}
    elif (ckpt_dir / ".." / ".." / "config.gin").exists(): 
        return {'ckpt': str(ckpt_path), 'config': str(ckpt_dir / ".." / ".." / "config.gin")}
    else:
        raise BendedRAVEImportException(path, msg="config file not found")
    
def _rave_get_model_paths_from_folder(path):
    # is it directly a folder containing checkpoints? 
    path = Path(path)
    ckpt_files = list(path.glob('*.ckpt')) + list(path.glob('checkpoints/*.ckpt'))
    # version dirs
    version_dirs = list(filter(lambda x: x.is_dir(), path.glob('version_*')))
    version_dirs = list(filter(lambda x: re.match(r'version_\d+', x.name) is not None, version_dirs))
    if len(ckpt_files):
        ckpt_files.sort(key=lambda x: os.path.getmtime(x))
        return _rave_get_model_paths_from_ckpt(ckpt_files[-1])
    elif version_dirs:
        # maybe it is a folder containing version
        if len(version_dirs) == 0: raise BendedRAVEImportException(path, "could not fetch any checkpoint from path")
        max_version = max([int(v.name.split('_')[-1]) for v in version_dirs])
        return _rave_get_model_paths_from_folder(path / f"version_{max_version}")
    else:
        raise BendedRAVEImportException(path, "could not fin any checkpoint in dir %s"%path)
    

def rave_get_model_paths(path):
    path = Path(path)
    # is it directly a checkpoint? 
    if path.is_file() and path.suffix in ['.ckpt']:
        return _rave_get_model_paths_from_ckpt(path)
    elif path.is_file():
        raise BendedRAVEImportException(path)
    else:
        return _rave_get_model_paths_from_folder(path)


class BendedRAVE(Interface):
    _imported_callbacks_ = []
    _proxied_buffers = ['.*pad', '.*cache', 'latent_mean', 'latent_pca']

    def __init__(self, model_path, strict=True, batch_size=4):
        model = self.load_model(model_path, strict=strict)
        self.batch_size = batch_size

        # warmup model cache
        model(torch.zeros(self.batch_size, model.n_channels, 8192))

        self.model = model
        self.pre_process_latent = MethodType(pre_process_fn[type(self.model.encoder)], self)
        self.post_process_latent = MethodType(post_process_fn[type(self.model.encoder)], self)

    @staticmethod
    def is_loadable(path):
        try: 
            model_paths = rave_get_model_paths(path)
            return True
        except BendedRAVEImportException as e:
            return False

    @staticmethod
    def load_model(model_path, strict=True, device="cpu"):
        if (not os.path.isfile(model_path)) or (os.path.splitext(model_path)[1] == ".ckpt"):
            return BendedRAVE.load_checkpoint(model_path, strict=strict, device=device)
        else:
            raise NotImplementedError
            # return BendedRAVE.load_scripted(model_path)

    @staticmethod
    def load_checkpoint(model_path, strict=True, device="cpu"):
        assert BendedRAVE.is_loadable(model_path)
        cc.use_cached_conv(True)
        paths = rave_get_model_paths(model_path)

        config_path = paths['config']
        if config_path is None:
            raise BendedRAVEImportException(model_path)
        gin.parse_config_file(config_path)
        model = ravelib.RAVE()
        
        run = paths['ckpt']
        if run is None:
            raise BendedRAVEImportException(model_path)
        model = model.load_from_checkpoint(run, strict=strict, map_location=device)

        for m in model.modules():
            if hasattr(m, "weight_g"):
                nn.utils.remove_weight_norm(m)        

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
    def channels(self):
        return self._model.n_channels

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

    @property
    def receptive_field(self):
        return self._model.receptive_field

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