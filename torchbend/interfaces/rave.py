import torch
import torchaudio
from typing import LiteralString, Union, Optional
import rave as ravelib
from .base import Interface
from ..tracing import BendedModule
import gin

class BendingRAVEException(Exception):
    pass


class BendedRAVE(Interface):
    _imported_callbacks_ = []

    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    @staticmethod
    def load_model(model_path):
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

    def _bend_model(self, model):
        self._model = BendedModule(model)
        self._model.trace("forward", x=torch.zeros(1, 1, 48000))
        _, (decoder_out,) = self._model.trace("encode", x=torch.zeros(1, 1, 48000), _return_out=True)
        latent_out = self._model.encoder.reparametrize(decoder_out)[:2][0]
        self._model.trace("decode", z=latent_out)

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

    def forward(self, x: Union[torch.Tensor, LiteralString], out: Optional[LiteralString] = None):
        if isinstance(x, str):
            x = self.load_audio(x)
        audio = self._model.forward(x)
        if out is not None: self.write_audio(out, audio)
        return audio

    def encode(self, x: Union[torch.Tensor, LiteralString]):
        if isinstance(x, str):
            x = self.load_audio(x)
        decoder_out = self._model.encode(x)
        return self._model.encoder.reparametrize(decoder_out)[:2][0]

    def decode(self, z: torch.Tensor, out: Optional[LiteralString] = None):
        audio = self._model.decode(z)
        if out is not None: self.write_audio(out, audio)
        return audio
        


__all__ = ['BendedRAVE']