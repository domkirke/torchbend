"""Bending interface for SpeechT5 text-to-speech.

SpeechT5 splits cleanly in two. Text to mel-spectrogram is autoregressive — a
decoder that emits frames until it decides to stop — and so cannot be traced.
The **HiFiGAN vocoder** that turns mel into audio is a single convolutional
pass, and traces to 372 nodes.

That vocoder is the bendable graph here, as in
:class:`~torchbend.interfaces.vits.BendedVits` and much like RAVE's decoder:
:meth:`speak` runs the full model but routes the final step through the bended
vocoder, so a bending is heard in the speech.

SpeechT5 is speaker-conditioned by a 512-dim x-vector. One is generated from a
fixed seed so the class is usable with no extra download; pass your own to
:meth:`speak` (or set :attr:`speaker_embedding`) for a specific voice.

One quirk worth knowing: SpeechT5's speech-decoder prenet applies dropout at
*inference* too — deliberately, it is what gives the prosody its variety — so
two identical calls produce speech of different lengths. Pass ``seed`` to pin
it, which is the only way a bent run and a clean one can be compared.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..spec import Audio, Callback, Int, Option, Text
from ..base import Interface
from ...tracing import ScriptableState


__all__ = ["BendedSpeechT5", "BendingSpeechT5Exception"]


_DEFAULT_MODEL = "microsoft/speecht5_tts"
_DEFAULT_VOCODER = "microsoft/speecht5_hifigan"
_SPEAKER_DIM = 512


class BendingSpeechT5Exception(Exception):
    pass


class _VocoderShim(nn.Module):
    """Stands in for the model's vocoder so the bended graph is on the path."""

    def __init__(self, bended):
        super().__init__()
        self._bended = bended

    def forward(self, spectrogram):
        return self._bended.forward(spectrogram=spectrogram)


class BendedSpeechT5(Interface):
    """Text to speech, with the HiFiGAN vocoder as the bendable graph."""

    _imported_callbacks_ = []
    _panel_render_type_ = "audio"

    options = {
        "speaker_seed": Option(
            Int(range=(0, 9999)), attr="speaker_seed", label="speaker seed",
            doc="Which generated x-vector to speak with. A different seed is a "
                "different synthetic voice."),
        "trace_frames": Option(
            Int(range=(16, 512)), attr="trace_frames", needs="retrace", label="trace frames",
            doc="Mel length the vocoder is traced on. The trace is symbolic in "
                "it, so this rarely needs changing."),
    }

    callbacks = {
        "speak": Callback(returns=Audio(), args={
            "text": Text(default="hello, this is a test", placeholder="something to say…"),
            "seed": Int(optional=True, doc="Pins the prenet dropout, which is on at "
                                           "inference and otherwise varies the length."),
        }),
    }

    def __init__(self, model_path: str = _DEFAULT_MODEL,
                 vocoder_path: str = _DEFAULT_VOCODER,
                 speaker_seed: int = 0, trace_frames: int = 120,
                 device=torch.device("cpu"), **kwargs):
        self.device = device
        self.speaker_seed = speaker_seed
        self.trace_frames = trace_frames
        tts, vocoder, processor = self.load_model(model_path, vocoder_path,
                                                  device=device, **kwargs)
        self._tts = tts
        self._processor = processor
        super().__init__(vocoder)

    # -- loading --

    @staticmethod
    def load_model(model_path=_DEFAULT_MODEL, vocoder_path=_DEFAULT_VOCODER,
                   device="cpu", **kwargs):
        try:
            from transformers import (SpeechT5ForTextToSpeech, SpeechT5HifiGan,
                                      SpeechT5Processor)
        except ImportError as exc:                      # pragma: no cover
            raise BendingSpeechT5Exception(
                "SpeechT5 needs `transformers`: pip install transformers") from exc
        try:
            processor = SpeechT5Processor.from_pretrained(str(model_path))
            tts = SpeechT5ForTextToSpeech.from_pretrained(str(model_path), **kwargs)
            vocoder = SpeechT5HifiGan.from_pretrained(str(vocoder_path))
        except Exception as exc:
            raise BendingSpeechT5Exception(
                "could not load SpeechT5 %s / %s, got: %s"
                % (model_path, vocoder_path, exc))
        return tts.eval().to(device), vocoder.eval().to(device), processor

    @staticmethod
    def is_loadable(path) -> bool:
        try:
            BendedSpeechT5.load_model(path)
            return True
        except Exception:
            return False

    # -- tracing --

    def bend_model(self, model):
        """Trace the vocoder on a mel of the traced length.

        Text-to-mel is deliberately not traced: it emits frames until it decides
        to stop, which is a data-dependent loop.
        """
        model.trace("forward", spectrogram=self._trace_mel())

    def _trace_mel(self):
        dim = int(self._model._module.config.model_in_dim)
        return torch.randn(self.trace_frames, dim, device=self.device)

    # -- properties --

    @property
    def config(self):
        return self._tts.config

    @property
    def processor(self):
        return self._processor

    @property
    def sample_rate(self) -> int:
        return int(self._model._module.config.sampling_rate)

    @property
    def speaker_embedding(self) -> torch.Tensor:
        """A synthetic x-vector, fixed by `speaker_seed` so a voice is repeatable."""
        gen = torch.Generator().manual_seed(int(self.speaker_seed))
        vec = torch.randn(1, _SPEAKER_DIM, generator=gen)
        return torch.nn.functional.normalize(vec, dim=-1).to(self.device)

    # -- inference --

    def mel(self, text: str, speaker_embedding: Optional[torch.Tensor] = None,
            seed: Optional[int] = None):
        """The mel-spectrogram for some text — what the bendable graph runs on."""
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self._processor(text=text, return_tensors="pt")
        with torch.no_grad():
            return self._tts.generate_speech(
                inputs["input_ids"].to(self.device),
                self.speaker_embedding if speaker_embedding is None else speaker_embedding,
            )

    def speak(self, text: Union[str, List[str]] = "hello, this is a test",
              speaker_embedding: Optional[torch.Tensor] = None,
              seed: Optional[int] = None):
        """Synthesise speech, through the bended vocoder.

        Without a ``seed`` the mel comes out a different length every call (see
        the module docstring), so a bent run cannot be compared with a clean one.
        """
        if not isinstance(text, str):
            text = " ".join(text)
        spectrogram = self.mel(text, speaker_embedding, seed=seed)
        with torch.no_grad():
            waveform = self._model.forward(spectrogram=spectrogram)
        while waveform.ndim < 3:
            waveform = waveform.unsqueeze(0)
        return waveform

    # -- export --

    @property
    def scriptable(self):
        return ScriptableState.NotScriptable

    @property
    def nntilde_compatible(self):
        return ScriptableState.NotScriptable
