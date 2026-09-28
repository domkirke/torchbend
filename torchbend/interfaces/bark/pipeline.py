"""Bark end to end: text in, audio out, with the decoder as the bendable graph.

:class:`~torchbend.interfaces.bark.BendedBark` gives you one of Bark's parts at
a time. That is the right shape for studying a stage, and the wrong shape for
actually making sound: to hear anything you have to run the other three parts
yourself.

This joins them. :class:`BarkPipeline` is a module holding Bark's four models
and EnCodec, whose ``forward`` is the *decoder* -- the one piece that traces,
and the one whose output is a waveform. The rest of the pipeline hangs off it as
methods, because each transformer samples in a Python loop and no graph can hold
that.

The arrangement has a property the per-stage interface does not:

    **everything you bend is audible.**

The graph is literally the last step of generation, so ``speak()`` routes the
decode through the bended copy. Weight bendings and activation bendings both
come out of the speaker -- no state-dict copying, no "this one is silent".

Three ways in, matching the three places you might want to start:

``codes_from_text``   text → the fine codebooks Bark decided on
``audio_from_codes``  codes → audio, through the bended decoder
``audio_from_text``   both at once

The middle one is the one to reach for while working: generating the codes takes
tens of seconds, decoding them takes a moment, and the codes are cached. So you
generate once, then bend and re-decode as often as you like.

**How much of Bark is in the graph** is a choice, because it costs something.

``include_fine=True`` (the default) puts the **fine transformer** in as well --
Bark's one language model that a graph can hold. It writes codebooks 2..7 in six
full passes over the sequence, a fixed count rather than anything the data
decides, so it unrolls: 5664 nodes, 648 of them the transformer's. Its codes are
exact -- every entry of all eight codebooks matches what Bark's own fine stage
produces. The *audio* is very slightly not, because the fine model works on a
padded 1024-frame window and the graph decodes the whole of it before the
utterance is cut back out, where Bark decodes the utterance alone. That is worth
``mean|Δ| = 0.0038`` on the waveform.

``include_fine=False`` starts at the finished codes: 245 nodes, decoding in
0.06 s, and bit-identical to ``BarkModel.generate``. No language model in it.

The semantic and coarse transformers cannot join either way. They sample one
token at a time and stop when they decide to, and no graph holds a loop whose
length is decided while it runs.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from .interface import (BARK_STAGES, BendingBarkException, _CaptureDone,
                        _freeze_conv_geometry)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..spec import (Audio, Callback, Float, Input, InputMode, Int, Method,
                    Option, Ref, Str, Text)
from ..base import Interface
from ...tracing import ScriptableState


__all__ = ["BarkPipeline", "BendedBarkPipeline", "BendingBarkException"]


_DEFAULT_MODEL = "suno/bark-small"

#: EnCodec frames per audio sample, for trimming the graph's padded output.
_SAMPLES_PER_FRAME = 320


def _cfg(section_owner, section, key, default):
    """Read a nested generation-config value, dict or object."""
    sec = getattr(section_owner, section, None)
    if isinstance(sec, dict):
        return sec.get(key, default)
    return getattr(sec, key, default)


class _FineFiller(nn.Module):
    """The fine transformer writing codebooks 2..7, as a module of its own.

    A module rather than a loop in ``forward`` so that its nodes -- the six
    passes and the argmax/where/concat between them -- carry a module path and
    fold together in depth mode. Ops written straight into ``forward`` belong to
    nothing and can never be folded, which on this graph meant 600-odd nodes the
    canvas had to draw whatever depth you asked for.
    """

    def __init__(self, fine, n_coarse, n_fine, codebook_size):
        super().__init__()
        self.fine = fine
        self.n_coarse, self.n_fine = n_coarse, n_fine
        self.codebook_size = codebook_size

    def forward(self, buf, write_mask=None):
        for i in range(self.n_coarse, self.n_fine):
            logits = self.fine(i, buf).logits
            pred = torch.argmax(logits[:, :, : self.codebook_size], dim=-1)
            if write_mask is not None:
                pred = torch.where(write_mask.bool(), pred, buf[:, :, i])
            buf = torch.cat([buf[:, :, :i], pred.unsqueeze(-1), buf[:, :, i + 1:]], dim=-1)
        return buf


class _Quantizer(nn.Module):
    """EnCodec's quantizer decode, as a module.

    ``quantizer.decode`` is a method, so the codebook lookups it performs are
    attributed to no module and stay unfoldable. Calling it through a module
    gives them a home.
    """

    def __init__(self, quantizer):
        super().__init__()
        self.quantizer = quantizer

    def forward(self, codes):
        return self.quantizer.decode(codes)


class BarkPipeline(nn.Module):
    """Bark's generation and EnCodec's decoder as one module.

    Only the decoder is a registered child, so it is the only thing traced,
    copied or state-dicted -- 7.4 M parameters rather than the full 404 M. The
    rest of Bark is held by reference, deliberately outside the module tree: the
    pipeline needs to *call* it, not to bend it.
    """

    def __init__(self, model, processor, include_fine: bool = True,
                 include_samplers: bool = False):
        super().__init__()
        # The two samplers are registered only when asked for: they are 300 MB
        # of parameters the BendedModule would otherwise copy for nothing, and
        # driving generation through them is slow (see `semantic_step`).
        self.include_samplers = include_samplers
        if include_samplers:
            self.semantic = model.semantic
            self.coarse = model.coarse_acoustics
        # Registered children are what the graph contains. The quantizer's eight
        # codebook lookups and the decoder always; the fine transformer too,
        # unless asked otherwise -- it is six forward passes and most of the
        # node count, but it is also the one language model Bark has that a
        # graph can hold.
        self.include_fine = include_fine
        gen = model.generation_config
        self.n_coarse = int(_cfg(gen, "coarse_acoustics_config", "n_coarse_codebooks", 2))
        self.n_fine = int(_cfg(gen, "fine_acoustics_config", "n_fine_codebooks", 8))
        self.codebook_size = int(getattr(gen, "codebook_size", 1024))
        if include_fine:
            self.fine_filler = _FineFiller(model.fine_acoustics, self.n_coarse,
                                           self.n_fine, self.codebook_size)
        self.quantizer = _Quantizer(model.codec_model.quantizer)
        self.decoder = _freeze_conv_geometry(model.codec_model.decoder)
        self.max_fine_history = int(
            _cfg(gen, "fine_acoustics_config", "max_fine_history_length", 512))
        #: Where the last utterance sits inside the fine model's window. The
        #: padded buffer does not say, and the graph's audio has to be cut back
        #: to it: [history | what was said | padding].
        self.last_n_frames = None
        self.last_n_history = 0
        self.last_write_mask = None
        #: The BendedModule wrapping this pipeline, set by the interface, so the
        #: samplers can be routed through the graph with its bendings applied.
        self._bended = None
        #: Cap on semantic tokens when generation is routed through the graphs.
        self.sampler_max_tokens = 48
        # `object.__setattr__` keeps these out of `_modules`, so a BendedModule
        # copy shares the live Bark rather than duplicating 400 MB of it.
        object.__setattr__(self, "_full", model)
        object.__setattr__(self, "_processor", processor)

    # -- the traceable part --

    def forward(self, codes, write_mask=None):
        """Codes to waveform. This is the graph, and it is the whole of it.

        With ``include_fine``, ``codes`` is ``[B, T, 8]`` holding the two coarse
        codebooks, and the graph starts by having the fine transformer write the
        other six -- six full passes over the sequence, all of them here to be
        looked at and bent. Without it, ``codes`` is ``[B, 8, T]`` and the graph
        starts at the codebook lookups.

        Either way the last node is audio: generation is not wrapped around the
        graph, the graph is what generates.
        """
        if self.include_fine:
            # The window the fine model works on is [history | coarse | padding],
            # and the padding is written as `codebook_size` — one past the last
            # legal code. Wrapping brings it back in range so the quantizer can
            # look it up; the padded region is nonsense either way and gets
            # trimmed off the audio afterwards.
            filled = self.fine_filler(codes, write_mask) % self.codebook_size
            codes = filled.permute(2, 0, 1)
        else:
            codes = codes.transpose(0, 1)
        return self.decoder(self.quantizer(codes))

    def _fill_fine(self, buf, write_mask=None):
        """The fine transformer's six passes. See :class:`_FineFiller`."""
        return self.fine_filler(buf, write_mask)

    # -- the samplers, one step at a time --
    #
    # Their *loops* cannot be traced: each runs until it decides to stop, and a
    # graph is a fixed sequence of operations. One step can be, though -- and a
    # loop that calls the traced step is a loop that runs through the graph. So
    # these are what turn a prompt into codes, and bending them changes what
    # gets said rather than only how it sounds.

    def semantic_step(self, input_ids):
        """One pass of the semantic transformer, **from text tokens**.

        Traced on ids rather than embeddings so the text embedding table is in
        the graph: `input_embeds_layer.weight` and the lookup that reads it.
        This is the text→latent path, and bending anything in it changes what
        the model is about to say.
        """
        return self.semantic(input_ids=input_ids, use_cache=False).logits

    def semantic_step_embeds(self, inputs_embeds):
        """The same pass, entered one step later.

        Bark embeds the prompt itself -- it concatenates the voice preset's
        semantic history, which has no token ids -- and hands the model
        `inputs_embeds`. So generation's first call cannot go through
        `semantic_step`, and this is the entry it uses instead. Same weights,
        same layers; only the text embedding is missing from the front.
        """
        return self.semantic(inputs_embeds=inputs_embeds, use_cache=False).logits

    def coarse_step(self, input_ids):
        """One pass of the coarse transformer: semantic tokens to codebook logits."""
        return self.coarse(input_ids=input_ids, use_cache=False).logits

    def _route_samplers(self, bended):
        """Point Bark's sampling loops at the bended step graphs.

        Returns a restore callable. ``bended`` is the BendedModule, so each step
        of generation goes through the graph with whatever is bent in it --
        which is the only way a bending on the semantic model can change the
        words. The cost is the KV cache: the traced step does not keep one, so
        every token re-reads the whole sequence.
        """
        sem, coa = self._full.semantic, self._full.coarse_acoustics
        sem_fwd, coa_fwd = sem.forward, coa.forward
        # The *graph modules*, not the bended module's methods: those dispatch
        # back to this very module and the shim would call itself forever. Built
        # once here rather than per step, since each build re-walks the graph.
        sem_gm = bended.graph_module(fn="semantic_step")
        sem_emb_gm = bended.graph_module(fn="semantic_step_embeds")
        coa_gm = bended.graph_module(fn="coarse_step")

        def sem_shim(*args, **kw):
            # ids when generation has them (every step after the first), the
            # embeds entry for the prompt itself -- both are the same weights,
            # so a bending on either graph's layers is felt throughout
            ids = kw.get("input_ids")
            if ids is not None:
                logits = sem_gm.semantic_step(input_ids=ids)
            else:
                logits = sem_emb_gm.semantic_step_embeds(
                    inputs_embeds=kw.get("inputs_embeds"))
            return CausalLMOutputWithPast(logits=logits, past_key_values=None)

        def coa_shim(*args, **kw):
            ids = kw.get("input_ids")
            if ids is None and args:
                ids = args[0]
            return CausalLMOutputWithPast(
                logits=coa_gm.coarse_step(input_ids=ids), past_key_values=None)

        sem.forward, coa.forward = sem_shim, coa_shim

        def restore():
            sem.forward, coa.forward = sem_fwd, coa_fwd
        return restore

    # -- the parts that cannot be a graph --

    @property
    def sample_rate(self) -> int:
        return int(self._full.generation_config.sample_rate)

    @property
    def codec_dim(self) -> int:
        return int(self._full.codec_model.config.hidden_size)

    def _inputs(self, text, voice_preset=None):
        kwargs = {"voice_preset": voice_preset} if voice_preset else {}
        enc = self._processor(text, **kwargs)
        device = next(self.decoder.parameters()).device
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}

    def codes_from_text(self, text: Union[str, List[str]], voice_preset=None,
                        temperature: float = 1.0, seed: Optional[int] = None):
        """Run the transformers and hand back what the graph starts from.

        Where that is depends on how much of the pipeline is in the graph. With
        the fine transformer in it, generation is stopped one stage earlier --
        at the buffer the fine model is about to fill -- so the graph does that
        filling itself. Otherwise it runs to the finished codes.

        Either way the value is taken by intercepting the real pipeline rather
        than rebuilding it, so the graph gets exactly what Bark would have used.
        """
        if not isinstance(text, str):
            text = " ".join(text)
        inputs = self._inputs(text, voice_preset)
        grabbed = {}

        if self.include_fine:
            # Two things are needed. `generate` is handed the coarse output, and
            # its length is the real one -- the buffer that reaches `forward` is
            # padded to the fine model's window, so the graph would otherwise
            # return more audio than was said.
            live = self._full.fine_acoustics
            original, original_gen = live.forward, live.generate

            def gen_spy(coarse_output, *a, **kw):
                grabbed["n_frames"] = int(coarse_output.shape[-1]) // self.n_coarse
                # ...and how much voice-preset history sits in front of it, so
                # the audio can be trimmed back to the part that was said
                # a BatchFeature, not a dict -- an isinstance check on dict
                # silently reports no history, and the audio then gets trimmed
                # to the voice preset instead of to what was said
                hp = kw.get("history_prompt")
                fine_hist = None
                if hp is not None:
                    try:
                        fine_hist = hp["fine_prompt"]
                    except Exception:
                        fine_hist = None
                grabbed["n_history"] = (
                    min(int(fine_hist.shape[-1]), self.max_fine_history)
                    if fine_hist is not None else 0)
                return original_gen(coarse_output, *a, **kw)

            def spy(*args, **kwargs):
                buf = kwargs.get("input_ids")
                if buf is None and len(args) > 1:
                    buf = args[1]
                grabbed["codes"] = buf
                raise _CaptureDone()

            live.generate, live.forward = gen_spy, spy

            def restore():
                live.forward, live.generate = original, original_gen
        else:
            original = self._full.codec_decode

            def spy(fine_output, output_lengths=None):
                grabbed["codes"] = fine_output
                raise _CaptureDone()

            self._full.codec_decode = spy
            restore = lambda: delattr(self._full, "codec_decode")

        # With the samplers in the graph, generation runs through them: this is
        # what makes bending the semantic or coarse model change the words.
        # Without a KV cache it is slow, which is why it is opt-in.
        route = (self._route_samplers(self._bended)
                 if self.include_samplers and self._bended is not None else None)
        gen_kwargs = {}
        if route:
            # No KV cache through the graph, so every token re-reads the whole
            # sequence: seconds each, and Bark will happily produce hundreds of
            # them. Bounded, or a prompt never comes back.
            gen_kwargs = {"use_cache": False,
                          "semantic_max_new_tokens": int(self.sampler_max_tokens)}

        if seed is not None:
            torch.manual_seed(seed)
        try:
            with torch.no_grad():
                self._full.generate(**inputs, do_sample=True,
                                    temperature=float(temperature), **gen_kwargs)
        except _CaptureDone:
            pass
        finally:
            if route:
                route()
            restore()
        if grabbed.get("codes") is None:
            raise BendingBarkException("Bark produced no codes for this prompt")
        # remembered rather than returned, so the caller's signature stays a
        # tensor in and a tensor out
        self.last_n_frames = grabbed.get("n_frames") or self.n_frames_of(grabbed["codes"])
        self.last_n_history = grabbed.get("n_history", 0)
        self.last_write_mask = self.write_mask_for(grabbed["codes"], self.last_n_history)
        return grabbed["codes"]

    def embeddings_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """The quantizer's view of the codes -- what the decoder graph takes."""
        with torch.no_grad():
            return self._full.codec_model.quantizer.decode(codes.transpose(0, 1))

    def write_mask_for(self, codes: torch.Tensor, n_history: int) -> torch.Tensor:
        """True where the fine model may write: everything after the history."""
        if not self.include_fine:
            return None
        b, t = codes.shape[0], codes.shape[1]
        mask = torch.zeros(b, t, dtype=torch.bool, device=codes.device)
        mask[:, int(n_history):] = True
        return mask

    def n_frames_of(self, codes: torch.Tensor) -> int:
        """How many EnCodec frames these codes carry, whichever layout they are in."""
        return int(codes.shape[1] if self.include_fine else codes.shape[-1])

    def audio_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Codes to audio -- the graph, shaped ``[batch, 1, samples]``.

        The fast half: no sampling, just the codebooks and the decoder.
        """
        with torch.no_grad():
            out = self(codes)
        out = out if torch.is_tensor(out) else out["output"]
        while out.ndim < 3:
            out = out.unsqueeze(0)
        return out

    def audio_from_text(self, text: Union[str, List[str]], voice_preset=None,
                        temperature: float = 1.0, seed: Optional[int] = None):
        """Text to audio, the whole way."""
        return self.audio_from_codes(
            self.codes_from_text(text, voice_preset, temperature, seed))


class BendedBarkPipeline(Interface):
    """Text to speech through Bark, with the decoder bendable and audible."""

    _imported_callbacks_ = []
    _panel_render_type_ = "audio"

    options = {
        "voice_preset": Option(
            Str(), attr="voice_preset", label="voice preset",
            doc="A Bark speaker, e.g. v2/en_speaker_6. Empty for none."),
        "temperature": Option(
            Float(range=(0.1, 2.0)), attr="temperature",
            doc="Sampling temperature for the three transformers."),
        "sampler_max_tokens": Option(
            Int(range=(8, 256)), attr="sampler_max_tokens", label="sampler tokens",
            doc="With the samplers in the graph, how many semantic tokens to "
                "generate. Each is a full pass through a 900-node graph with no "
                "KV cache, so this is a time budget more than a length. Ignored "
                "when the samplers are not in the graph."),
        "trace_frames": Option(
            Int(range=(8, 512)), attr="trace_frames", needs="retrace", label="trace frames",
            doc="Frame count the decoder is traced on. The trace is generic in "
                "it, so this rarely needs changing."),
    }

    #: One callback, and it is a convenience rather than the way in: the graph
    #: takes a prompt on `codes` and ends in audio, so exploring, bending and
    #: generating all happen in the graph. This is here for a one-liner.
    callbacks = {
        "speak": Callback(
            returns=Audio(), label="speak (text → audio)",
            args={"text": Text(default="hello, this is bark speaking",
                               placeholder="something to say…"),
                  "seed": Int(optional=True)},
            doc="Runs the transformers for a prompt and puts the codes through "
                "the graph. Exactly what typing that prompt into the `codes` "
                "input does."),
    }

    def __init__(self, model_path: str = _DEFAULT_MODEL,
                 voice_preset: str = "v2/en_speaker_6", trace_frames: int = 64,
                 include_fine: bool = True, include_samplers: bool = False,
                 device=torch.device("cpu"), model=None, processor=None, **kwargs):
        self.device = device
        self.voice_preset = voice_preset
        self.temperature = 1.0
        self.trace_frames = trace_frames
        if model is None or processor is None:
            model, processor = self.load_model(model_path, device=device, **kwargs)
        pipeline = BarkPipeline(model, processor, include_fine=include_fine,
                                include_samplers=include_samplers)
        self.include_fine = include_fine
        self.include_samplers = include_samplers
        self._pipeline = pipeline
        self._codes = None            # what the graph runs on
        self._n_frames = None         # how long the utterance really was
        self._n_history = 0           # how much voice-preset history precedes it
        self._write_mask = None       # where the fine model may write
        self._prompt_cache = {}
        codes_default = (
            ("torch.cat([torch.randint(0, 1024, (1, %d, 2)), "
             "torch.zeros(1, %d, 6, dtype=torch.long)], dim=-1)"
             % (trace_frames, trace_frames)) if include_fine
            else "torch.randint(0, 1024, (1, 8, %d))" % trace_frames)
        inputs = {"codes": Input(codes_default, mode=InputMode(
            Text(default="hello, this is bark speaking",
                 placeholder="type something to say… (runs Bark's transformers)"),
            encode="encode_inputs", also_fills=["write_mask"] if include_fine else [],
            label="prompt",
            doc="Runs Bark's samplers so the graph gets the codes it decided "
                "on. Slow the first time, then cached — so bend, re-run, and "
                "hear the same utterance re-voiced."))}
        if include_fine:
            inputs["write_mask"] = Input("torch.ones(1, %d, dtype=torch.bool)" % trace_frames)
        # the graph's output is the waveform: naming its rate is what makes the
        # viewer open it as audio and play it at that rate
        self.methods = {"forward": Method(inputs=inputs,
                                          outputs=[Audio(sample_rate=Ref("sample_rate"))])}
        super().__init__(pipeline)
        pipeline._bended = self._model

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
            BendedBarkPipeline.load_model(path)
            return True
        except Exception:
            return False

    # -- tracing --

    def bend_model(self, model):
        """Trace the decoder.

        Two things are needed, both explained in ``interface.py``: EnCodec's
        conv geometry has to be plain ints for make_fx to get past it, and the
        LSTM has to stay one node or the graph bakes in the frame count.
        """
        if self.include_fine:
            # [B, T, 8]: the two coarse codebooks set, the six the fine model
            # writes still zero -- the buffer it is really handed
            codes = torch.cat([
                torch.randint(0, 1024, (1, self.trace_frames, 2), device=self.device),
                torch.zeros(1, self.trace_frames, 6, dtype=torch.long, device=self.device),
            ], dim=-1)
            model.trace("forward", codes=codes, _wrap_recurrent=True,
                        write_mask=torch.ones(1, self.trace_frames, dtype=torch.bool,
                                              device=self.device))
        else:
            codes = torch.randint(0, 1024, (1, 8, self.trace_frames), device=self.device)
            model.trace("forward", codes=codes, _wrap_recurrent=True)

        if self.include_samplers:
            # One graph per sampler, each a single step. The viewer lists them
            # beside `forward`, so the whole path from prompt to audio is
            # something you can open and bend rather than something that
            # happens out of sight.
            hidden = int(model._module.semantic.config.hidden_size)
            # the text→latent graph: token ids in, embedding table included
            model.trace("semantic_step",
                        input_ids=torch.randint(0, 512, (1, self.trace_frames),
                                                device=self.device))
            # and the entry generation uses for the prompt, which arrives
            # already embedded (see `semantic_step_embeds`)
            model.trace("semantic_step_embeds",
                        inputs_embeds=torch.randn(1, self.trace_frames, hidden,
                                                  device=self.device))
            model.trace("coarse_step",
                        input_ids=torch.randint(0, 512, (1, self.trace_frames),
                                                device=self.device))


    # -- properties --

    @property
    def config(self):
        return self._pipeline._full.config

    @property
    def processor(self):
        return self._pipeline._processor

    @property
    def sample_rate(self) -> int:
        return self._pipeline.sample_rate

    @property
    def sampler_max_tokens(self) -> int:
        return self._pipeline.sampler_max_tokens

    @sampler_max_tokens.setter
    def sampler_max_tokens(self, value):
        self._pipeline.sampler_max_tokens = int(value)

    @property
    def last_codes(self) -> Optional[torch.Tensor]:
        """The fine codebooks from the last prompt, if there was one.

        Not called `codes`: that name is the callback that *makes* them.
        """
        return self._codes

    # -- feeding the graph --

    def encode_inputs(self, text: Union[str, List[str]]) -> dict:
        """A prompt, as the codes the graph takes.

        This is what makes the graph generative: type a prompt into `codes` and
        the transformers run, the graph runs on what they produced, and every
        node in it -- codebook lookups, decoder layers, the output waveform --
        is there to inspect and bend. Cached, so bending and re-running does not
        pay for generation again.
        """
        if not isinstance(text, str):
            text = " ".join(text)
        key = (text, self.voice_preset, self.temperature)
        if key not in self._prompt_cache:
            self._prompt_cache[key] = (
                self._pipeline.codes_from_text(text, self.voice_preset, self.temperature),
                self._pipeline.last_n_frames, self._pipeline.last_n_history,
                self._pipeline.last_write_mask)
        (self._codes, self._n_frames, self._n_history,
         self._write_mask) = self._prompt_cache[key]
        filled = {"codes": self._codes}
        if self._write_mask is not None:
            filled["write_mask"] = self._write_mask
        return filled

    # -- callbacks --

    def _audio_from_codes(self, codes):
        """Codes to audio **through the bended graph**.

        `BarkPipeline.audio_from_codes` runs the module itself, which is the
        original -- `BendedModule` bends a copy. Going through `self._model` is
        what puts the bendings on the path.
        """
        kwargs = {"codes": codes}
        if self.include_fine:
            kwargs["write_mask"] = (
                self._write_mask if self._write_mask is not None
                else self._pipeline.write_mask_for(codes, self._n_history))
        with torch.no_grad():
            out = self._model.forward(**kwargs)
        out = out if torch.is_tensor(out) else out["output"]
        while out.ndim < 3:
            out = out.unsqueeze(0)
        # The fine model works on a window of [history | said | padding], so the
        # graph returns more audio than was said. Cut back to the middle part.
        if self._n_frames:
            start = self._n_history * _SAMPLES_PER_FRAME
            stop = start + self._n_frames * _SAMPLES_PER_FRAME
            if 0 <= start < stop <= out.shape[-1]:
                out = out[..., start:stop]
        return out

    def speak(self, text: Union[str, List[str]] = "hello, this is bark speaking",
              seed: Optional[int] = None):
        """Text to audio: run the transformers, then the graph.

        The same thing typing the prompt into the `codes` input does — this
        just does it in one call and hands back the waveform.
        """
        self._codes = self._pipeline.codes_from_text(
            text, self.voice_preset, self.temperature, seed)
        self._n_frames = self._pipeline.last_n_frames
        self._n_history = self._pipeline.last_n_history
        self._write_mask = self._pipeline.last_write_mask
        return self._audio_from_codes(self._codes)

    # -- export --

    @property
    def scriptable(self):
        return ScriptableState.NotScriptable

    @property
    def nntilde_compatible(self):
        return ScriptableState.NotScriptable
