"""Bending interface for VITS / MMS-TTS text-to-speech.

VITS is end-to-end -- text encoder, duration predictor, flow, HiFiGAN vocoder
-- and this makes the whole thing one bendable graph. That costs something:
the length regulator, between the duration predictor and the flow, normally
sizes its output from ``predicted_lengths.max()`` -- a number decided by the
model's own prediction, not by any input's shape. A symbolic tracer cannot
carry that as an unbacked size past the first op that needs to compare it
against something else, which turns out to be deep inside the flow's WaveNet
layers (``fused_add_tanh_sigmoid_multiply``): ``GuardOnDataDependentSymNode``.

So this pins a maximum instead: ``max_frames``, a plain Python int rather
than a traced value. The length regulator produces exactly that many frames
every time, masking whatever a shorter sentence didn't need rather than
sizing the tensor to it. That is what lets the trace go through.

It is not free. The flow's convolutions see ``max_frames`` of context on
every call, including the zero-padded tail past what a short sentence used,
and that tail leaks backward through the receptive field into frames that
would otherwise be exact. Checked against ``VitsModel.forward`` directly, a
13-frame sentence traced at ``max_frames=200`` comes back with a max sample
error of 0.24 on a waveform in ``[-1, 1]`` -- audible, not a rounding
difference. A sentence whose real length would exceed ``max_frames`` is cut
off outright: the tail past the cap was never computed. Raise ``max_frames``
for longer sentences; it retraces.

The alternative -- one graph on each side of the length regulator, bit-exact
and with no cap on length -- is not what this class does; this trades that
exactness and that unbounded length for having every VITS weight reachable
in a single graph, viewable and bendable end to end.

``facebook/mms-tts-*`` covers a thousand-odd languages at ~290 MB each; the
class works with any single-speaker VITS checkpoint ``transformers`` can load.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..spec import (Audio, Callback, Float, Input, InputMode, Int, Method,
                    Option, Text)
from ..base import Interface
from ...tracing import ScriptableState


__all__ = ["BendedVits", "BendingVitsException"]


_DEFAULT_MODEL = "facebook/mms-tts-eng"


class BendingVitsException(Exception):
    pass


class _VitsPipeline(nn.Module):
    """Text encoder, duration predictor, flow and HiFiGAN, as one module.

    Every learned weight VITS has lives here, in the order the real forward
    runs them. The length regulator between the duration predictor and the
    flow has none of its own -- it is ``VitsModel.forward``'s alignment block,
    copied in below with one change: ``max_frames`` is a fixed constant
    rather than ``predicted_lengths.max()``, which is what lets this trace at
    all. See the module docstring for what that change costs.
    """

    def __init__(self, text_encoder, duration_predictor, flow, decoder, max_frames: int,
                speaking_rate: float, noise_scale: float, noise_scale_duration: float):
        super().__init__()
        self.text_encoder = text_encoder
        self.duration_predictor = duration_predictor
        self.flow = flow
        self.decoder = decoder
        self.max_frames = max_frames
        self.speaking_rate = speaking_rate
        self.noise_scale = noise_scale
        self.noise_scale_duration = noise_scale_duration

    def forward(self, input_ids, padding_mask, attention_mask=None):
        out = self.text_encoder(input_ids=input_ids, padding_mask=padding_mask,
                                attention_mask=attention_mask, return_dict=True)
        hidden_states = out.last_hidden_state.transpose(1, 2)
        mask = padding_mask.transpose(1, 2)
        log_duration = self.duration_predictor(
            hidden_states, mask, None, reverse=True,
            noise_scale=self.noise_scale_duration)

        # -- length regulator: VitsModel.forward's alignment block, with
        # `max_frames` standing in for `predicted_lengths.max()` --
        length_scale = 1.0 / self.speaking_rate
        duration = torch.ceil(torch.exp(log_duration) * mask * length_scale)
        predicted_lengths = torch.clamp(
            torch.clamp_min(torch.sum(duration, [1, 2]), 1).long(), max=self.max_frames)
        indices = torch.arange(self.max_frames, dtype=predicted_lengths.dtype,
                               device=predicted_lengths.device)
        output_padding_mask = (indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
                               ).unsqueeze(1).to(mask.dtype)
        attn_mask = torch.unsqueeze(mask, 2) * torch.unsqueeze(output_padding_mask, -1)
        batch_size, _, output_length, input_length = attn_mask.shape
        cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
        frame_indices = torch.arange(output_length, dtype=duration.dtype, device=duration.device)
        valid_indices = (frame_indices.unsqueeze(0) < cum_duration
                         ).to(attn_mask.dtype).view(batch_size, input_length, output_length)
        padded_indices = valid_indices - nn.functional.pad(valid_indices, [0, 0, 1, 0, 0, 0])[:, :-1]
        attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask
        prior_means = torch.matmul(attn.squeeze(1), out.prior_means).transpose(1, 2)
        prior_log_variances = torch.matmul(attn.squeeze(1), out.prior_log_variances).transpose(1, 2)
        prior_latents = (prior_means + torch.randn_like(prior_means)
                        * torch.exp(prior_log_variances) * self.noise_scale)
        # -- flow + vocoder --
        spectrogram = self.flow(prior_latents, output_padding_mask, None, reverse=True) * output_padding_mask
        waveform = self.decoder(spectrogram, None)
        return waveform, predicted_lengths


class BendedVits(Interface):
    """Text to speech, the whole pipeline as one bendable graph."""

    _imported_callbacks_ = []
    _panel_render_type_ = "audio"

    #: Every one of these is read by `_VitsPipeline.forward`, which is the
    #: traced graph -- so a plain Python float or int read there is baked
    #: into the trace, and changing it needs a retrace to be heard.
    #:
    #: The graph viewer's own retrace button re-traces the *live pipeline
    #: module* as it currently stands rather than calling back into
    #: `bend_model`, so these use `get`/`set` accessors rather than `attr`:
    #: the setter writes through to the pipeline module immediately, and
    #: whichever retrace path runs next picks up the new value because it was
    #: already sitting on the module, not because anything asked it to.
    options = {
        "speaking_rate": Option(
            Float(range=(0.25, 4.0)), needs="retrace",
            get="get_speaking_rate", set="set_speaking_rate", label="speaking rate",
            doc="Higher is faster. Scales the predicted durations."),
        "noise_scale": Option(
            Float(range=(0.0, 2.0)), needs="retrace",
            get="get_noise_scale", set="set_noise_scale", label="noise scale",
            doc="How much the sampled latent wanders around the flow's prior. "
                "0 gives the same reading every time."),
        "noise_scale_duration": Option(
            Float(range=(0.0, 2.0)), needs="retrace",
            get="get_noise_scale_duration", set="set_noise_scale_duration",
            label="duration noise scale",
            doc="How much the stochastic duration predictor wanders."),
        "trace_tokens": Option(
            Int(range=(1, 64)), attr="trace_tokens", needs="retrace", label="trace tokens",
            doc="Token length the graph is seeded with at trace time. The trace "
                "is symbolic in it, so this rarely needs changing."),
        "max_frames": Option(
            Int(range=(8, 1024)), needs="retrace",
            get="get_max_frames", set="set_max_frames", label="max frames",
            doc="Latent frames the length regulator produces, always -- unlike "
                "the token count, this is a hard cap, not a seed: sentences "
                "needing more are cut off. Raise it for longer sentences. See "
                "the module docstring for why it exists."),
    }

    callbacks = {
        "speak": Callback(returns=Audio(), args={
            "text": Text(default="hello, this is a test", placeholder="something to say…"),
            "seed": Int(optional=True),
        }),
    }

    #: A prompt fills `input_ids`, `padding_mask` and `attention_mask` together
    #: -- one prompt fixes the token count, so the three have to agree.
    methods = {
        "forward": Method(inputs={
            "input_ids": Input(mode=InputMode(
                Text(default="hello, this is a test", placeholder="something to say…"),
                encode="encode_inputs", also_fills=["padding_mask", "attention_mask"],
                label="text",
                doc="Tokenized and run through the whole graph -- text encoder, "
                    "duration predictor, length regulator, flow and vocoder.")),
        }),
    }

    def __init__(self, model_path: str = _DEFAULT_MODEL,
                 trace_tokens: int = 12, max_frames: int = 200,
                 device=torch.device("cpu"), model=None, tokenizer=None, **kwargs):
        self.device = device
        self.trace_tokens = trace_tokens
        self.max_frames = max_frames
        if model is None or tokenizer is None:
            model, tokenizer = self.load_model(model_path, device=device, **kwargs)
        self._full = model
        self._tokenizer = tokenizer
        self.speaking_rate = float(getattr(model.config, "speaking_rate", 1.0))
        self.noise_scale = float(getattr(model.config, "noise_scale", 0.667))
        self.noise_scale_duration = float(getattr(model.config, "noise_scale_duration", 0.8))
        pipeline = _VitsPipeline(model.text_encoder, model.duration_predictor,
                                 model.flow, model.decoder, max_frames,
                                 self.speaking_rate, self.noise_scale, self.noise_scale_duration)
        super().__init__(pipeline)

    # -- loading --

    @staticmethod
    def load_model(model_path=_DEFAULT_MODEL, device="cpu", **kwargs):
        try:
            from transformers import AutoTokenizer, VitsModel
        except ImportError as exc:                      # pragma: no cover
            raise BendingVitsException(
                "VITS needs `transformers`: pip install transformers") from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = VitsModel.from_pretrained(str(model_path), **kwargs)
        except Exception as exc:
            raise BendingVitsException(
                "could not load VITS model %s, got: %s" % (model_path, exc))
        return model.eval().to(device), tokenizer

    @staticmethod
    def is_loadable(path) -> bool:
        try:
            BendedVits.load_model(path)
            return True
        except Exception:
            return False

    # -- tracing --

    def bend_model(self, model):
        """Trace the pipeline module on the current options' values.

        `max_frames`, `speaking_rate`, `noise_scale` and `noise_scale_duration`
        are read as plain attributes inside `_VitsPipeline.forward`, which is
        what gets traced -- so they live on that module, not on this
        interface, and a changed option has to be copied across before a
        retrace picks it up.
        """
        pipeline = model._module
        pipeline.max_frames = self.max_frames
        pipeline.speaking_rate = self.speaking_rate
        pipeline.noise_scale = self.noise_scale
        pipeline.noise_scale_duration = self.noise_scale_duration
        input_ids = torch.randint(0, self._full.config.vocab_size,
                                  (1, self.trace_tokens), device=self.device)
        padding_mask = torch.ones(1, self.trace_tokens, 1, device=self.device)
        attention_mask = torch.ones(1, self.trace_tokens, device=self.device)
        model.trace("forward", input_ids=input_ids, padding_mask=padding_mask,
                   attention_mask=attention_mask)

    # -- options --
    #
    # Each setter writes through to the live pipeline module (`self._model`
    # exists by the time any of these can be called from the UI, since options
    # are only reachable once construction -- and so `bend_model` -- has run).
    # See the `options` comment for why that matters.

    def get_speaking_rate(self) -> float:
        return self.speaking_rate

    def set_speaking_rate(self, value) -> float:
        self.speaking_rate = float(value)
        self._model._module.speaking_rate = self.speaking_rate
        return self.speaking_rate

    def get_noise_scale(self) -> float:
        return self.noise_scale

    def set_noise_scale(self, value) -> float:
        self.noise_scale = float(value)
        self._model._module.noise_scale = self.noise_scale
        return self.noise_scale

    def get_noise_scale_duration(self) -> float:
        return self.noise_scale_duration

    def set_noise_scale_duration(self, value) -> float:
        self.noise_scale_duration = float(value)
        self._model._module.noise_scale_duration = self.noise_scale_duration
        return self.noise_scale_duration

    def get_max_frames(self) -> int:
        return self.max_frames

    def set_max_frames(self, value) -> int:
        self.max_frames = int(value)
        self._model._module.max_frames = self.max_frames
        return self.max_frames

    # -- properties --

    @property
    def config(self):
        return self._full.config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def sample_rate(self) -> int:
        return int(self._full.config.sampling_rate)

    @property
    def hop_length(self) -> int:
        """Audio samples per latent frame -- the vocoder's total upsampling."""
        hop = 1
        for rate in self._full.config.upsample_rates:
            hop *= rate
        return int(hop)

    # -- text to the graph's inputs --

    def encode_inputs(self, text: Union[str, List[str]]) -> dict:
        """Tokenize a prompt into the placeholders the graph takes."""
        enc = self._tokenizer(text if isinstance(text, str) else list(text),
                              return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.device)
        mask_dtype = self._full.text_encoder.embed_tokens.weight.dtype
        padding_mask = attention_mask.unsqueeze(-1).to(mask_dtype)
        return {"input_ids": input_ids, "padding_mask": padding_mask,
                "attention_mask": attention_mask}

    # -- inference --

    def speak(self, text: Union[str, List[str]] = "hello, this is a test",
              seed: Optional[int] = None):
        """Synthesise speech through the bended graph, cropped to its real length.

        The graph always produces `max_frames` of latent, masked past what a
        sentence actually needed (see the module docstring) -- so the
        waveform it returns is cropped back to `predicted_lengths * hop`
        rather than handed back at the padded length.
        """
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.encode_inputs(text)
        with torch.no_grad():
            waveform, predicted_lengths = self._model.forward(**inputs)
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)
        n_samples = int(predicted_lengths.max().item()) * self.hop_length
        return waveform[..., :n_samples]

    # -- export --

    @property
    def scriptable(self):
        return ScriptableState.NotScriptable

    @property
    def nntilde_compatible(self):
        return ScriptableState.NotScriptable
