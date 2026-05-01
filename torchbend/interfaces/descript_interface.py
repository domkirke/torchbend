import os
from pathlib import Path
import torchaudio
import torch, os, random
from .base import Interface, _export_to_module
from .utils import get_random_hash
import dac
from dac.utils import __MODEL_LATEST_TAGS__, __MODEL_URLS__
from .. import _TORCHBEND_DEFAULT_MODEL_DIR, ScriptableState

_IMPORT_AS_INTERFACE_ = True

"""
TODO : bug list

- scripting the snake activations makes cuda unavailable

"""

def custom_dac_download(
    model_path: Path, model_type: str = "44khz", model_bitrate: str = "8kbps", tag: str = "latest"
):
    
    model_type = model_type.lower()
    tag = tag.lower()

    assert model_type in [
        "44khz",
        "24khz",
        "16khz",
    ], "model_type must be one of '44khz', '24khz', or '16khz'"

    assert model_bitrate in [
        "8kbps",
        "16kbps",
    ], "model_bitrate must be one of '8kbps', or '16kbps'"

    if tag == "latest":
        tag = __MODEL_LATEST_TAGS__[(model_type, model_bitrate)]

    download_link = __MODEL_URLS__.get((model_type, tag, model_bitrate), None)

    if download_link is None:
        raise ValueError(
            f"Could not find model with tag {tag} and model type {model_type}"
        )

    local_path = (
        model_path 
        / f"weights_{model_type}_{model_bitrate}_{tag}.pth"
    )
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the model
        import requests

        response = requests.get(download_link)

        if response.status_code != 200:
            raise ValueError(
                f"Could not download model. Received response code {response.status_code}"
            )
        local_path.write_bytes(response.content)

    return local_path


class BendedDescriptAudioCodec(Interface):

    def __init__(self, 
                 model_dir: Path | str | None = None, 
                 trace_n_batches: int = 4,
                 trace_n_samples: int | float = 4.0,
                 device: torch.device = torch.device('cpu'),
                 **kwargs):
        model_dir = Path(model_dir or _TORCHBEND_DEFAULT_MODEL_DIR) / "dac"
        model_path = custom_dac_download(model_dir, **kwargs)
        model = dac.DAC.load(str(model_path)).eval().to(device)
        self._sr = model.metadata['kwargs']['sample_rate']
        self.trace_n_samples = self._parse_trace_len(trace_n_samples, self._sr)
        self.trace_n_batches = trace_n_batches
        self.device= device
        super(BendedDescriptAudioCodec, self).__init__(model)

    def _parse_trace_len(self, trace_len: int | float, sr: int) -> int:
        if isinstance(trace_len, float):
            return int(trace_len * sr)
        else: 
            return trace_len

    def bend_model(self, model):
        x = torch.randn(self.trace_n_batches, 1, self.trace_n_samples, device=self.device)
        _, (z, codes, latents, _, _) = model.trace(fn="encode", audio_data=x, _return_out=True)
        y = model.trace(fn="decode", z=z, _return_out=True)

    def encode(self, x: torch.Tensor):
        x = self.original_model.preprocess(x, self.sample_rate).to(self.device)
        return self._model.encode(x)

    def decode(self, z: torch.Tensor):
        out = self._model.decode(z.to(self.device))
        return out

    @property
    def sample_rate(self) -> int: 
        return self._sr
    
    @property
    def audio_channels(self) -> int: 
        return 1

    @property
    def scriptable(self): 
        return ScriptableState.Unknown

    @property
    def nntilde_compatible(self):
        return ScriptableState.NotScriptable
        
