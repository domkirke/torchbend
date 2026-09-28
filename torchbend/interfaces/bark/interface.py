"""Bending interface for Bark, Suno's text-to-audio model.

Bark is four models in a row: a *semantic* transformer turns text into semantic
tokens, a *coarse* one turns those into the first two EnCodec codebooks, a
*fine* one fills in the remaining six, and EnCodec decodes the lot to audio.

Three of the four trace, and each is a different thing to bend:

``semantic``   what is said — bend it and the words wander
``coarse``     the broad acoustic shape — bend it and the voice changes
``fine``       the detail EnCodec reconstructs — bend it and the timbre frays

The codec itself does not trace (``GuardOnDataDependentSymNode``), and neither
does generation as a whole: each stage samples token by token, which is a
data-dependent loop. So generation is offered as a callback, exactly as with
GPT-2, and one stage at a time is the bendable graph.

**What a bending reaches.** Each stage drives its own sampling loop, so
``generate`` runs the model's code rather than the traced graph. Weight
bendings are carried into it — they are written into the live stage before the
call — but activation bendings apply to the traced graph only, and will not be
heard in ``generate``. Bend weights if you want it audible.

**Feeding a stage from a prompt.** Every stage's forward takes ``input_ids``,
but only the semantic one takes *text* there: coarse reads semantic tokens and
fine reads coarse codebooks. So the prompt input mode does whatever that stage
needs — tokenize for semantic, run the pipeline up to this stage for the other
two — and hands over the tensor the stage would really have been given. That
makes the prompt slow for coarse (a few seconds) and slower for fine (it has to
generate the coarse codes first), so results are cached per prompt.
"""

import contextlib
from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..spec import (Audio, Callback, Float, Input, InputMode, Int, Method,
                    Option, Ref, Str, Text)
from ..base import Interface
from ...tracing import ScriptableState


__all__ = ["BendedBark", "BendingBarkException", "BARK_STAGES"]


_DEFAULT_MODEL = "suno/bark-small"

#: The stages that trace, and where each lives on the full model.
#:
#: The three transformers predict tokens, so their graphs end in logits. The
#: fourth entry is EnCodec's *decoder* -- the one part of the codec that traces
#: -- and its graph ends in a waveform, which makes it the only Bark graph
#: where an activation bending is audible.
BARK_STAGES = {
    "semantic": "semantic",
    "coarse":   "coarse_acoustics",
    "fine":     "fine_acoustics",
    "codec":    "codec_model.decoder",
}

#: The stages that predict tokens, as opposed to the one that makes sound.
TOKEN_STAGES = ("semantic", "coarse", "fine")

#: EnCodec keeps its convolution geometry in buffers rather than as ints, so the
#: padding amounts come out as tensors and make_fx turns them into unbacked
#: symbols -- ``Eq(u0, 1)``, and the trace stops. The values are derived from
#: the config and never learned, so as plain ints they are the same numbers.
_CONV_GEOMETRY = ("kernel_size", "stride", "padding_total")


def _resolve(root, path):
    for part in path.split("."):
        root = getattr(root, part)
    return root


def _extra_padding_int(self, hidden_states):
    """``_get_extra_padding_for_conv1d`` in integer arithmetic.

    The original runs ``torch.ceil`` on a tensor. ``ceil(a/s) + 1 - 1`` is just
    ``ceil(a/s)``, and integer ceil-division gives it exactly -- on a plain int
    or a SymInt, so the traced graph stays generic in the frame count.
    """
    length = hidden_states.shape[-1]
    a = length - self.kernel_size + self.padding_total
    n_frames = -((-a) // self.stride)
    ideal = n_frames * self.stride + self.kernel_size - self.padding_total
    return ideal - length


def _freeze_conv_geometry(module):
    """Make EnCodec's conv geometry plain ints, permanently.

    Done once when the interface is built rather than around each trace: a
    retrace goes through the tracer directly and would otherwise hit the same
    unbacked symbol the first trace was helped past. The values are the ones
    the buffers held, so the module computes exactly what it did before -- only
    now in a form make_fx can follow.
    """
    for m in module.modules():
        if hasattr(m, "_get_extra_padding_for_conv1d"):
            m._get_extra_padding_for_conv1d = _extra_padding_int.__get__(m)
        for name in _CONV_GEOMETRY:
            val = getattr(m, "_buffers", {}).get(name)
            if isinstance(val, torch.Tensor):
                del m._buffers[name]
                setattr(m, name, int(val))
    return module


@contextlib.contextmanager
def _static_conv_geometry(module):
    """The same, undone afterwards. Kept for callers that want it scoped."""
    saved, patched = [], []
    for m in module.modules():
        if hasattr(m, "_get_extra_padding_for_conv1d"):
            patched.append(m)
            m._get_extra_padding_for_conv1d = _extra_padding_int.__get__(m)
        for name in _CONV_GEOMETRY:
            val = getattr(m, "_buffers", {}).get(name)
            if isinstance(val, torch.Tensor):
                saved.append((m, name, val))
                del m._buffers[name]
                setattr(m, name, int(val))
    try:
        yield
    finally:
        for m in patched:
            del m._get_extra_padding_for_conv1d
        for m, name, val in saved:
            m.__dict__.pop(name, None)
            m.register_buffer(name, val, persistent=False)


class _DecoderShim(nn.Module):
    """Stands in for EnCodec's decoder so the bended graph is on the audio path.

    Bark calls ``self.decoder(embeddings)`` from inside ``_decode_frame``.
    Swapping this in for the duration of a call is what makes a bending audible
    -- bending a copy nothing calls would change nothing you could hear.
    """

    def __init__(self, bended):
        super().__init__()
        self._bended = bended

    def forward(self, hidden_states):
        out = self._bended.forward(hidden_states=hidden_states)
        return out if torch.is_tensor(out) else out["output"]


#: What a prompt has to go through to reach each stage's `input_ids`, in the
#: words the bench shows next to the field.
_PROMPT_MODES = {
    "semantic": {
        "label": "prompt",
        "placeholder": "type something to say…",
        "default": "hello, this is bark speaking",
        "doc": "Tokenized straight into the semantic model's input_ids.",
    },
    "coarse": {
        "label": "prompt",
        "placeholder": "type something to say… (runs the semantic stage)",
        "default": "hello, this is bark speaking",
        "doc": "The coarse model reads semantic tokens, not text, so the "
               "prompt is run through the semantic stage first. Takes a "
               "few seconds.",
    },
    "fine": {
        "label": "prompt",
        "placeholder": "type something to say… (runs semantic + coarse)",
        "default": "hello, this is bark speaking",
        "doc": "The fine model reads coarse codebooks, so the prompt is run "
               "through the semantic and coarse stages first. Slow — the "
               "result is cached per prompt.",
    },
    "codec": {
        "label": "prompt",
        "placeholder": "type something to say… (runs all three transformers)",
        "default": "hello, this is bark speaking",
        "doc": "The decoder reads the quantizer's embeddings, so the prompt is "
               "run through the whole pipeline first. Slowest of the four — "
               "the result is cached per prompt.",
    },
}


class BendingBarkException(Exception):
    pass


class _CaptureDone(Exception):
    """Raised to stop generation once the stage has been handed its input."""


class BendedBark(Interface):
    """Text to speech through Bark, with one stage as the bendable graph."""

    _imported_callbacks_ = []
    _panel_render_type_ = "audio"

    options = {
        "voice_preset": Option(
            Str(), attr="voice_preset", label="voice preset",
            doc="A Bark speaker, e.g. v2/en_speaker_6. Empty for none."),
        "temperature": Option(
            Float(range=(0.1, 2.0)), attr="temperature",
            doc="Sampling temperature for the stages that sample."),
        "trace_tokens": Option(
            Int(range=(8, 256)), attr="trace_tokens", needs="retrace", label="trace tokens",
            doc="Sequence length the stage is traced on. The trace is symbolic "
                "in it, so this rarely needs changing."),
    }

    callbacks = {
        "speak": Callback(returns=Audio(), args={
            "text": Text(default="hello, this is bark speaking", placeholder="something to say…"),
            "temperature": Float(range=(0.1, 2.0), step=0.05),
            "seed": Int(optional=True),
        }),
    }

    def __init__(self, model_path: str = _DEFAULT_MODEL, stage: str = "fine",
                 voice_preset: str = "v2/en_speaker_6", trace_tokens: int = 64,
                 device=torch.device("cpu"), model=None, processor=None, **kwargs):
        if stage not in BARK_STAGES:
            raise BendingBarkException(
                "unknown stage %r (have: %s)" % (stage, ", ".join(BARK_STAGES)))
        if stage == "codec":
            # the decoder is the only Bark graph whose output is a waveform
            self._panel_render_type_ = "audio"
            self._freeze_geometry = True
        self.device = device
        self.stage = stage
        self.voice_preset = voice_preset
        self.temperature = 1.0
        self.trace_tokens = trace_tokens
        # `model`/`processor` let several stages share one loaded Bark; four
        # graphs over four copies is 1.6 GB for no reason.
        if model is None or processor is None:
            model, processor = self.load_model(model_path, device=device, **kwargs)
        self._full = model
        self._processor = processor
        self._prompt_cache = {}
        self._codec_dim = int(model.codec_model.config.hidden_size)
        if getattr(self, "_freeze_geometry", False):
            _freeze_conv_geometry(model.codec_model.decoder)
        # Seed the bench. `input_ids` has no signature default and the fine
        # stage's `codebook_idx` has none either, so without these the viewer
        # opens on a graph it cannot run and no clue what would be valid. The
        # prompt input mode replaces `input_ids` the moment anyone types.
        # `input_ids` means something different in each stage, so the method
        # has to be declared once the stage is known rather than on the class.
        prompt = _PROMPT_MODES[stage]
        mode = InputMode(Text(default=prompt["default"], placeholder=prompt["placeholder"]),
                         encode="encode_inputs", label=prompt["label"], doc=prompt["doc"])
        if stage == "codec":
            inputs = {"hidden_states": Input(
                "torch.randn(1, %d, %d)" % (self._codec_dim, trace_tokens), mode=mode)}
        elif stage == "fine":
            inputs = {"input_ids": Input("torch.randint(0, 512, (1, %d, 8))" % trace_tokens,
                                         mode=mode),
                      # 0 and 1 come from the coarse model; the fine model fills 2..7
                      "codebook_idx": Input("2")}
        else:
            inputs = {"input_ids": Input("torch.randint(0, 512, (1, %d))" % trace_tokens,
                                         mode=mode)}
        # the decoder is the only Bark graph whose output is a waveform; naming
        # its rate is what says "this node is audio" -- every other 3-D tensor
        # in there is a feature map, so it cannot be keyed on rank
        outputs = [Audio(sample_rate=Ref("sample_rate"))] if stage == "codec" else []
        self.methods = {"forward": Method(inputs=inputs, outputs=outputs)}
        super().__init__(_resolve(model, BARK_STAGES[stage]))

    # -- loading --

    @staticmethod
    def load_model(model_path=_DEFAULT_MODEL, device="cpu", **kwargs):
        try:
            from transformers import AutoProcessor, BarkModel
        except ImportError as exc:                      # pragma: no cover
            raise BendingBarkException(
                "Bark needs `transformers`: pip install transformers") from exc
        try:
            processor = AutoProcessor.from_pretrained(str(model_path))
            model = BarkModel.from_pretrained(str(model_path), **kwargs)
        except Exception as exc:
            raise BendingBarkException(
                "could not load Bark model %s, got: %s" % (model_path, exc))
        return model.eval().to(device), processor

    @staticmethod
    def is_loadable(path) -> bool:
        try:
            BendedBark.load_model(path)
            return True
        except Exception:
            return False

    # -- tracing --

    def bend_model(self, model):
        """Trace the chosen stage.

        ``use_cache=False`` for the two that sample: a transformers ``Cache``
        object cannot pass through fx, and leaving it on stops the trace before
        it starts. The fine model does not use one.
        """
        if self.stage == "codec":
            # `_wrap_recurrent` keeps the decoder's LSTM as one node; unrolled it
            # bakes in the frame count (2376 nodes hard-wired to one length)
            # instead of the 202-node graph that runs on any.
            emb = torch.randn(1, self._codec_dim, self.trace_tokens, device=self.device)
            model.trace("forward", hidden_states=emb, _wrap_recurrent=True)
            return
        tokens = torch.randint(0, 512, (1, self.trace_tokens), device=self.device)
        if self.stage == "fine":
            codes = torch.randint(0, 1024, (1, self.trace_tokens, 8), device=self.device)
            model.trace("forward", codebook_idx=2, input_ids=codes)
        else:
            model.trace("forward", input_ids=tokens, use_cache=False)

    # -- properties --

    @property
    def config(self):
        return self._full.config

    @property
    def processor(self):
        return self._processor

    @property
    def sample_rate(self) -> int:
        return int(self._full.generation_config.sample_rate)

    # -- feeding the graph from a prompt --

    def _processor_inputs(self, text):
        enc = self._processor(
            text, **({"voice_preset": self.voice_preset} if self.voice_preset else {}))
        return {k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in enc.items()}

    def _capture_stage_input(self, text):
        """The tensor this stage is handed when Bark really speaks `text`.

        Coarse and fine read the stage before them, not text. Rather than
        rebuilding that plumbing here -- it is `transformers` internals and it
        moves -- the real pipeline is run and the stage's own forward is spied
        on, then generation is cut short the moment the input arrives.
        """
        live = _resolve(self._full, BARK_STAGES[self.stage])
        original = live.forward
        grabbed = {}
        arg = self._prompt_arg

        def spy(*args, **kwargs):
            if not grabbed:
                # the samplers take `input_ids` first; the fine model takes the
                # codebook index first; the decoder takes `hidden_states`
                positional = ("codebook_idx", "input_ids") if self.stage == "fine" \
                    else (arg,)
                for i, a in enumerate(args[:len(positional)]):
                    grabbed[positional[i]] = a
                if arg in kwargs:
                    grabbed[arg] = kwargs[arg]
                raise _CaptureDone()
            return original(*args, **kwargs)

        live.forward = spy
        try:
            with torch.no_grad():
                self._full.generate(**self._processor_inputs(text), do_sample=True,
                                    temperature=float(self.temperature))
        except _CaptureDone:
            pass
        finally:
            live.forward = original

        ids = grabbed.get(arg)
        if ids is None:
            raise BendingBarkException(
                "could not capture what the %s stage is fed for this prompt"
                % self.stage)
        return ids

    @property
    def _prompt_arg(self):
        """Which placeholder the prompt fills, for this stage."""
        return "hidden_states" if self.stage == "codec" else "input_ids"

    def encode_inputs(self, text: Union[str, List[str]]) -> dict:
        """Turn a prompt into the input this stage's forward takes.

        Only the semantic model reads text; the other two read the stage before
        them, so for those the prompt is run that far through the pipeline
        first. Cached per prompt, because for the fine stage that is most of a
        generation.
        """
        if not isinstance(text, str):
            text = " ".join(text)
        key = (self.stage, text, self.voice_preset)
        if key not in self._prompt_cache:
            if self.stage == "semantic":
                # generation pre-embeds and passes `inputs_embeds`, but the
                # graph was traced on `input_ids` -- which is the same thing one
                # embedding earlier, and what the tokenizer produces.
                ids = self._processor_inputs(text)["input_ids"]
            else:
                ids = self._capture_stage_input(text)
            self._prompt_cache[key] = ids
        return {self._prompt_arg: self._prompt_cache[key]}

    # -- inference --

    def _apply_weight_bendings(self):
        """Write the bended weights into the stage generation actually runs.

        Each stage drives its own sampling loop over its own module, so the
        traced copy is never called. Copying the bended weights across is what
        makes a weight bending audible; an activation bending has no equivalent
        here and stays confined to the graph.
        """
        live = _resolve(self._full, BARK_STAGES[self.stage])
        try:
            bended = self._model.bended_state_dict()
        except Exception:
            return None
        original = {k: v.detach().clone() for k, v in live.state_dict().items()
                    if k in bended}
        live.load_state_dict(bended, strict=False)
        return original

    def speak(self, text: Union[str, List[str]] = "hello, this is bark speaking",
              temperature: float = 1.0, seed: Optional[int] = None):
        """Generate speech for some text, with this stage's weight bendings applied."""
        kwargs = {}
        if self.voice_preset:
            kwargs["voice_preset"] = self.voice_preset
        inputs = self._processor(text, **kwargs)
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                  for k, v in inputs.items()}

        restore = self._apply_weight_bendings()
        # On the codec stage the graph *is* the audio path, so swapping the
        # bended decoder in makes an activation bending audible -- the one place
        # in Bark where that is true.
        codec, original_decoder = None, None
        if self.stage == "codec":
            codec = self._full.codec_model
            original_decoder = codec.decoder
            codec.decoder = _DecoderShim(self._model)
        # Seed *after* the bendings are resolved: a random callback (a mask, a
        # noise) draws when its state dict is built, so seeding first leaves the
        # generator in a different place for a bent run than for a clean one —
        # and the two stop being comparable, which is the whole point of a seed.
        if seed is not None:
            torch.manual_seed(seed)
        try:
            with torch.no_grad():
                audio = self._full.generate(**inputs, do_sample=True,
                                            temperature=float(temperature))
        finally:
            if codec is not None:
                codec.decoder = original_decoder
            if restore:
                _resolve(self._full, BARK_STAGES[self.stage]).load_state_dict(
                    restore, strict=False)
        return audio if audio.ndim == 3 else audio.unsqueeze(1)

    # -- export --

    @property
    def scriptable(self):
        return ScriptableState.NotScriptable

    @property
    def nntilde_compatible(self):
        return ScriptableState.NotScriptable
