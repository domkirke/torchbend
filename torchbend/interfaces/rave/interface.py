import torch
from functools import cached_property
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
from ..base import Interface, _export_to_module, _overload_module
from ...tracing import BendedModule, NNBendedMethodAttributes
from .scripting import pre_process_fn, post_process_fn, script_rave_model, VariationalScriptedRAVE
import cached_conv as cc
import gin

_VALID_AUDIO_EXT = ['.wav', '.aif', '.aiff', '.mp3']

class BendingRAVEException(Exception):
    pass


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

    def __init__(self, model_path, strict=True, scriptable=True, **kwargs):
        model = self.load_model(model_path, strict=strict)

        self.scriptable = scriptable
        if scriptable:
            self.model = script_rave_model(model, **kwargs)
        else:
            self.model = model

        # warmup model cache
        # self.pre_process_latent = MethodType(pre_process_fn[type(self.model.encoder)], self)
        # self.post_process_latent = MethodType(post_process_fn[type(self.model.encoder)], self)

    @property
    @_export_to_module
    def _panel_render_type_(self):
        return "audio"

    @property
    def _proxied_buffers(self):
        if self.scriptable:
            return ['.*pad', '.*cache', 'latent_mean', 'latent_pca', 'decode_params', 'encode_params', 'forward_params']
        else: 
            return ['.*pad', '.*cache', 'latent_mean', 'latent_pca']

    @staticmethod
    def is_loadable(path):
        try: 
            model_paths = rave_get_model_paths(path)
            return True
        except BendedRAVEImportException as e:
            return False

    @staticmethod
    def load_model(model_path, strict: bool = True, scriptable: bool = False,  device: str | torch.device ="cpu"):
        if (not os.path.isfile(model_path)) or (os.path.splitext(model_path)[1] == ".ckpt"):
            return BendedRAVE.load_checkpoint(model_path, strict=strict, device=device, scriptable=scriptable)
        else:
            raise NotImplementedError()
            # return BendedRAVE.load_scripted(model_path)

    @staticmethod
    def load_checkpoint(model_path, strict=True, scriptable: bool = True, device: str | torch.device = "cpu", load_ema: bool = False):

        model, model_path = ravelib.load_rave_checkpoint(model_path, name=None, ema=load_ema)
        
        if scriptable: model = script_rave_model(model)

        _ = model.pqmf(torch.zeros(cc.MAX_BATCH_SIZE, model.pqmf.forward_conv.weight.shape[1], 8192))
        _ = model.pqmf.inverse(torch.zeros(cc.MAX_BATCH_SIZE, model.pqmf.inverse_conv.weight.shape[1], 8192))
        model(torch.zeros(1, model.n_channels, 8192))

        for m in model.modules():
            if hasattr(m, "weight_g"):
                nn.utils.remove_weight_norm(m)        

        return model

    @staticmethod
    def load_scripted(model_path):
        model = torch.jit.load(model_path)
        return model

    def _bend_model(self, model: BendedModule):
        model.trace("forward", x=torch.zeros(1, 1, 65536),  _proxied_buffers=self._proxied_buffers)
        is_variational = isinstance(model._module, VariationalScriptedRAVE)
        _, (decoder_out,) = model.trace("encode", x=torch.zeros(1, 1, 65536), _proxied_buffers=self._proxied_buffers, _return_out=True)
        if is_variational:
            _, (decoder_out_full,) = model.trace("encode_full", x=torch.zeros(1, 1, 65536), _proxied_buffers=self._proxied_buffers, _return_out=True)
            _ = model.trace("encode_dist", x=torch.zeros(1, 1, 65536), _proxied_buffers=self._proxied_buffers, _return_out=True)
        if self.scriptable:
            latent_out = decoder_out
        else:
            latent_out = model.encoder.reparametrize(decoder_out)[:2][0]
        model.trace("decode", z=latent_out, _proxied_buffers=self._proxied_buffers)
        if is_variational:
            model.trace("decode_full", z=decoder_out_full, _proxied_buffers=self._proxied_buffers)
        

    def _load_single_audio(self, path: str):
        audio, sr = torchaudio.load(path)
        if sr != self._model.sr:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        if audio.shape[0] >= self.channels:
            audio = audio[:self.channels]
        else:
            raise BendingRAVEException("model needs at least %d channels, but %s seems to have only %d"%(self.channels, path, audio.shape[0]))
        return audio

    def _stack_audio(self, path_list, **kwargs):
        stack = kwargs.get('stack', False)
        audios = [self._load_single_audio(p) for p in path_list]
        if stack:
            max_length = max([a.shape[-1] for a in audios])
            for i, a in enumerate(audios):
                if a.shape[-1] < max_length:
                    a = torch.nn.functional.pad(a, max_length - a.shape[-1], mode="constant", value=0.)
                    audios[i] = a
            audios = torch.stack(audios)
        return audios

    def load_audio(self, path: str, return_files=False, **kwargs):
        if os.path.isfile(path):
            audios = self._load_single_audio(path)
            paths = path
        elif os.path.isdir(path):
            valid_audio_files = []
            for r, d, f in os.walk(path):
                f = list(filter(lambda x: os.path.splitext(x)[1] in _VALID_AUDIO_EXT, f))
                f = list(map(lambda x, root=r: os.path.join(root, x), f))
                valid_audio_files.extend(f)
            audios = self._stack_audio(valid_audio_files, **kwargs)
            paths = valid_audio_files
        if return_files:
            return audios, paths
        else:
            return audios
            

    @property
    def channels(self):
        return self._model.n_channels

    @property
    def sample_rate(self):
        return self._model.sr

    def write_audio(self, path, audio):
        torchaudio.save(path, audio, self.sample_rate)

    def clear_cache(self): 
        for k, v in self._model.named_buffers(): 
            if re.match(r'.*(pad|cache)', k):
                v.data.zero_()

    def forward(self, x: Union[torch.Tensor, str], out: Optional[str] = None, clear_cache:  bool = False, **kwargs):
        if isinstance(x, str):
            x = self.load_audio(x)
        if clear_cache: self.clear_cache()
        audio = self._model.forward(x, **kwargs)
        if out is not None: self.write_audio(out, audio[0])
        return audio
    
    @property
    def latent_size(self):
        return self.model.latent_size
    @property
    def receptive_field(self):
        return self.model.receptive_field

    @property
    def encoder(self):
        return BendedModule(self._model.encoder)
    @property
    def decoder(self):
        return BendedModule(self._model.decoder)
    @property
    def discriminator(self):
        return BendedModule(self._model.discriminator)

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
        if self.scriptable:
            z = self._model.encode(x)
            if postprocess: 
                z = self.post_process_latent(z)
        else:
            decoder_out = self._model.encode(x)
            z = self._model.encoder.reparametrize(decoder_out)[:2][0]
            if postprocess: 
                z = self.post_process_latent(z)
        return z

    def decode(self, z: torch.Tensor, out: Optional[str] = None, preprocess=False):
        if self.scriptable:
            if preprocess: 
                z = self.pre_process_latent(z)
            audio = self._model.decode(z, preprocess=preprocess)
        else:
            if preprocess: 
                z = self.pre_process_latent(z)
            audio = self._model.decode(z)
        if out is not None: self.write_audio(out, audio[0])
        return audio


    # nntilde-related callbacks
    @_export_to_module
    def nn_tilde_methods(self):
        methods = {'encode': NNBendedMethodAttributes(
                    in_channels=self.channels,
                    in_ratio=1,
                    out_channels=self.latent_size,
                    out_ratio=self.model.ratio_encode,
                    input_labels=['(signal) Channel %d'%d for d in range(1, self.model.target_channels+1)],
                    output_labels=[
                        f'(signal) Latent dimension {i + 1}'
                        for i in range(self.latent_size)
                        ], 
                    ), 
                    'decode': NNBendedMethodAttributes(
                        in_channels=self.latent_size,
                        in_ratio=self.model.ratio_encode,
                        out_channels=self.model.target_channels,
                        out_ratio=1,
                        input_labels=[
                            f'(signal) Latent dimension {i+1}'
                            for i in range(self.latent_size)
                        ],
                        output_labels=['(signal) Channel %d'%d for d in range(1, self.model.target_channels+1)]
                    ), 
                    'forward': NNBendedMethodAttributes(
                        in_channels=self.channels,
                        in_ratio=1,
                        out_channels=self.model.target_channels,
                        out_ratio=1,
                        input_labels=['(signal) Channel %d'%d for d in range(1, self.channels + 1)],
                        output_labels=['(signal) Channel %d'%d for d in range(1, self.model.target_channels+1)]
                    )}
        if isinstance(self.model._module, VariationalScriptedRAVE): 
            methods['encode_full'] = NNBendedMethodAttributes(
                    in_channels=self.model.n_channels,
                    in_ratio=methods['encode'].in_ratio,
                    out_channels=self.model.full_latent_size,
                    out_ratio=methods['encode'].out_ratio,
                    input_labels=['(signal) Channel %d'%d for d in range(1, self.channels+1)],
                    output_labels=[
                        f'(signal) Latent dimension {i + 1}'
                        for i in range(self.model.full_latent_size)
                        ]
            )
            methods['decode_full'] = NNBendedMethodAttributes(
                    in_channels=self.model.full_latent_size,
                    in_ratio=methods['decode'].out_ratio,
                    out_channels=self.model.target_channels,
                    out_ratio=methods['decode'].out_ratio,
                    input_labels=[
                        f'(signal) Latent dimension {i + 1}'
                        for i in range(self.model.full_latent_size)
                        ],
                    output_labels=['(signal) Channel %d'%d for d in range(1, self.model.target_channels+1)],
            )
            methods['encode_dist'] = NNBendedMethodAttributes(
                    in_channels=self.model.n_channels,
                    in_ratio=methods['encode'].in_ratio,
                    out_channels=self.model.latent_size * 2,
                    out_ratio=methods['encode'].out_ratio,
                    input_labels=['(signal) Channel %d'%d for d in range(1, self.model.target_channels+1)],
                    output_labels=[
                            f'(signal) Latent mean {i + 1}'
                            for i in range(self.model.latent_size)
                        ] + [
                            f'(signal) Latent std {i + 1}'
                            for i in range(self.model.latent_size) 
                        ]
            )
            
        if self.model._has_prior:
            methods['prior'] = NNBendedMethodAttributes(
                in_channels=1,
                in_ratio=self.prior.ratio,
                out_channels = self.latent_size,
                out_ratio=self.prior.ratio
            )
        return methods

    @_export_to_module
    def register_nntilde_attributes(self, nn_module):
        nn_module.register_attribute('temperature', 1.)
        nn_module.register_attribute('projection', 'default')
        nn_module.register_attribute("learn_target", False)
        nn_module.register_attribute("reset_target", False)
        nn_module.register_attribute("learn_source", False)
        nn_module.register_attribute("reset_source", False)

    @_overload_module
    def script(self, *args, **kwargs):
        assert self.scriptable, "BendedRAVE must be initialized with scriptable=True to allow jit scripting"
        return self.model.script(*args, **kwargs)

    @_overload_module
    def nntilde(self, *args, **kwargs):
        assert self.scriptable, "BendedRAVE must be initialized with scriptable=True to allow jit scripting"
        self.clear_cache()
        return self.model.nntilde(*args, **kwargs)


__all__ = ['BendedRAVE']