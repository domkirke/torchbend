"""Bending interface for XTTS-v2 — text and a voice in, speech out.

XTTS is two models in a trenchcoat. A **GPT** predicts audio tokens from text
tokens plus a speaker conditioning, one token at a time; a **HiFiGAN decoder**
turns the latents that GPT produced into a waveform.

Only the second can be a graph. The GPT samples in a loop and stops when it
decides to, and no graph holds a loop whose length is decided while it runs
(the same limitation Bark's semantic and coarse stages hit — see
``torchbend.interfaces.bark.pipeline``). The decoder, by contrast, is one
fixed-shape feedforward pass, and it is the piece whose output *is* the sound.

That arrangement has the property worth having:

    **everything you bend is audible.**

``speak()`` runs XTTS's own inference, but with the decoder swapped for the
bended graph for the duration of the call, so the waveform you hear is the one
the bending produced — not a copy computed beside it. Weight bendings and
activation bendings both come out of the speaker.

Generating the latents is the slow part (it is an autoregressive language
model) and decoding them is quick, so the latents of the last utterance are
kept: say something once with :meth:`speak`, then bend and call
:meth:`decode_latents` as often as you like without re-generating anything.

The graph's inputs are the decoder's own: ``latents`` ``[B, T, 1024]`` from the
GPT, and ``g`` ``[B, 512, 1]``, the speaker embedding that carries the cloned
voice. Bending ``g`` is worth a try on its own — it is the voice, isolated
from what is being said.

**The decoder is only the last third.** ``full=True`` adds the rest -- the
GPT that decides what is said and the encoders that make the voice -- to the
*same* graph, ``forward``, caption in and speech out; see
:mod:`~torchbend.interfaces.xtts.pipeline` for how a sampling loop becomes
part of a graph, and what it costs. ``decode`` stays a graph of its own too:
latents and a voice to speech, without the GPT.

Needs coqui's ``TTS`` package (``pip install TTS``, which pins
``python<3.12``) and the XTTS-v2 checkpoint, which is downloaded from the Hub
on first use if no local directory is given.
"""

import os
from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..base import Interface
from ..spec import (Audio, Choice, Float, Input, InputMode, Int, Method,
                    Option, Ref, Str, Text, retain_aliases)
from ...tracing import ScriptableState
from .pipeline import XTTSPipeline


__all__ = ["BendedXTTS", "BendingXTTSException", "XTTS_LANGUAGES"]


_DEFAULT_MODEL = "coqui/XTTS-v2"

#: What XTTS-v2 was trained on. ``zh`` is spelled ``zh-cn``.
XTTS_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
                  "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]


#: Shapes ``forward`` is traced on. They are examples: the trace is symbolic in
#: every length below, so other lengths still run.
_TRACE_TEXT = 14      # text tokens
_TRACE_MEL = 300      # reference mel frames
_TRACE_AUDIO = 48000  # reference samples at 16 kHz


class BendingXTTSException(Exception):
    pass


def read_audio(path, sampling_rate):
    """A recording as a mono ``[1, T]`` tensor at ``sampling_rate``, via soundfile.

    Stands in for TTS's ``load_audio``, which reads through torchaudio, which
    since 2.9 reads through torchcodec, which needs FFmpeg's shared libraries
    next to the environment and cannot load without them. A reference wav needs
    none of that.
    """
    import soundfile as sf
    import torchaudio
    data, rate = sf.read(path, dtype="float32", always_2d=True)      # [T, C]
    audio = torch.from_numpy(data).mean(dim=1, keepdim=True).T
    if rate != sampling_rate:
        audio = torchaudio.functional.resample(audio, rate, sampling_rate)
    return audio.clip_(-1, 1)


class _DecoderShim(nn.Module):
    """Stands in for XTTS's HiFiGAN decoder so the bended graph is on the path.

    ``Xtts.inference`` calls ``self.hifigan_decoder(latents, g=speaker_embedding)``
    and returns whatever comes back. Swapping this in for the duration of a
    call is what makes a bending audible -- bending a copy nothing calls would
    change nothing you could hear.
    """

    def __init__(self, bended):
        super().__init__()
        self._bended = bended

    def forward(self, latents, g=None):
        out = self._bended.forward(latents=latents, g=g)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


class BendedXTTS(Interface):
    """Voice-cloning TTS, with XTTS's HiFiGAN decoder as the bendable graph.

    ``full=True`` makes the *whole process* the graph: ``forward`` takes a
    caption and a reference recording and returns speech, through the two voice
    encoders, the GPT sampling its audio tokens in a loop, and the decoder --
    all bendable, all audible. See :mod:`~torchbend.interfaces.xtts.pipeline`
    for how a loop that decides its own length becomes a graph, and what that
    costs. ``decode`` is a graph of its own too, for the last link alone;
    everything before it (the encoders, the sampling loop) is addressed by
    the aliases its main outputs are tagged with -- ``#conditioning``,
    ``#voice``, ``#tokens``, ``#latents``, ``#speech`` -- not by a graph of
    their own.
    """

    _imported_callbacks_ = []
    _panel_render_type_ = "audio"

    options = {
        "language": Option(
            Choice(XTTS_LANGUAGES), attr="language",
            doc="Language of the text to speak."),
        "speaker": Option(
            Str(), attr="speaker", when=lambda self: not self.full,
            doc="Built-in XTTS speaker to clone, e.g. 'Claribel Dervla'. "
                "Ignored when a reference wav is given to `speak`. "
                "`speakers` lists what the checkpoint ships with."),
        # with `full`, sampling temperature is an input of the `forward` graph
        # (on the bench) instead
        "temperature": Option(
            Float(range=(0.1, 1.5)), attr="temperature", when=lambda self: not self.full,
            doc="Sampling temperature of the GPT that writes the audio tokens."),
        "reference": Option(
            Str(), attr="reference", label="reference audio",
            doc="Path to a recording of the voice to clone, used by `speak`, "
                "and the default of `forward`'s two reference inputs (`mel`, "
                "the style; `audio`, the voice) -- each of which can be set to "
                "a different recording on the bench."),
        "seed": Option(
            Int(range=(-1, 1000000)), attr="seed",
            doc="Seed for sampling the audio tokens; -1 for a fresh draw each "
                "time. Fix it to bend and generate again from the same starting "
                "point: without it, what you hear changes because the sampling "
                "did, not only because of the bending."),
        # the rest only shape the sampling loop, which only `full` has
        "trace_tokens": Option(
            Int(range=(16, 605)), attr="trace_tokens", needs="retrace", when="full",
            label="audio tokens",
            doc="How many audio tokens the GPT's sampling loop runs for -- about "
                "21 per second of speech, so 125 is six seconds. A graph is a "
                "fixed sequence of steps: the speech ends where the GPT stops, "
                "but the loop always runs this long, so it is also the time "
                "each generation takes."),
        "loop_pack": Option(
            Int(range=(0, 605)), attr="loop_pack", needs="retrace", when="full",
            label="steps per node",
            doc="Sampling steps folded into one node of the graph. 0 (the "
                "default) is the whole loop in one node. Smaller splits it into "
                "several, each bendable at its boundary (the token buffer so "
                "far) -- but every node takes its own copy of the GPT's weights "
                "as inputs, so 5 nodes draw the GPT 5 times. Weight bendings "
                "reach every step regardless."),
        "loop_open": Option(
            Int(range=(0, 8)), attr="loop_open", needs="retrace", when="full",
            label="steps opened",
            doc="How many of the first sampling steps are drawn out in full -- "
                "every layer of the GPT, every activation bendable. Activations "
                "inside the other steps are not in the graph. Each opened step "
                "is about 2000 raw ops loose in the main graph, not grouped into "
                "a GPT block -- 0 (the default) keeps the whole loop, including "
                "its first step, inside the opaque loop node, so it reads as one "
                "block."),
    }

    def __init__(self, model_path: str = _DEFAULT_MODEL, language: str = "en",
                 speaker: str = "Claribel Dervla", temperature: float = 0.75,
                 trace_frames: int = 128, device=torch.device("cpu"),
                 model=None, config=None, full: bool = False, trace_tokens: int = 125,
                 loop_pack: int = 0, loop_open: int = 0, reference: str = "", **kwargs):
        self.device = device
        self.full = bool(full)
        self.trace_tokens = int(trace_tokens)
        self.loop_pack = int(loop_pack)
        self.loop_open = int(loop_open)
        self.reference = reference
        self.seed = -1
        self.language = language
        self.speaker = speaker
        self.temperature = float(temperature)
        self.trace_frames = int(trace_frames)
        if model is None:
            model, config = self.load_model(model_path, device=device, **kwargs)
        self._full = model
        self._config = config
        self._latents = None            # (gpt_latents, speaker_embedding) of the last utterance
        args = model.args
        self._latent_dim = int(getattr(args, "decoder_input_dim", 1024))
        self._speaker_dim = int(getattr(args, "d_vector_dim", 512))
        if self.full:
            pipeline = XTTSPipeline(model, n_tokens=self.trace_tokens)
            self._pipeline = pipeline
            module = pipeline
        else:
            module = model.hifigan_decoder
        self.methods = self._declare_methods()
        super().__init__(module)

    def _declare_methods(self):
        """`forward` is the decoder alone, or -- with `full` -- the whole
        process; `decode` exists only in the second case."""
        speech = Audio(sample_rate=Ref("sample_rate"))
        # The GPT and the two voice encoders run fine on MPS -- measured on the
        # real model. The HiFiGAN decoder does not: MPS's convolution kernel
        # hits `NotImplementedError: Output channels > 65536` once the traced
        # sequence is long enough (confirmed length-dependent: short latents
        # pass, anything near a real utterance's length does not) -- a backend
        # limitation, not something bendable away. `forward` with `full` ends
        # in the same decoder, so it inherits the same limit.
        no_mps = {"mps": False}
        decoder = Method(
            inputs={"latents": Input("torch.randn(1, %d, %d)" % (self.trace_frames, self._latent_dim)),
                    "g": Input("torch.randn(1, %d, 1)" % self._speaker_dim,
                               doc="The speaker embedding: the voice, isolated "
                                   "from what is being said.")},
            outputs=[speech], devices=no_mps)
        if not self.full:
            return {"forward": decoder}
        # Each of `forward`'s inputs on its own: the caption, the recording the
        # style is taken from (`mel`, for the conditioning encoder), and the one
        # the voice is taken from (`audio`, for the speaker encoder). They are
        # independent -- one recording's delivery with another's timbre is a
        # perfectly good thing to try. The two reference defaults follow the
        # `reference` option as it is now, not as it was at construction.
        reference_doc = ("A recording: load a file, or type a path (the box "
                         "starts with the `reference` option's).")
        recording = dict(default=Ref("reference"), placeholder="path to a recording, or load a file")
        forward = Method(
            inputs={
                "text_ids": Input(
                    "torch.randint(0, 200, (1, %d))" % _TRACE_TEXT,
                    mode=InputMode(
                        Text(default="the graph is the instrument",
                             placeholder="type something to say…"),
                        encode="encode_text", label="caption",
                        doc="The caption to say, tokenized with XTTS's own "
                            "tokenizer in the language set in the options.")),
                "mel": Input(
                    "torch.randn(1, 80, %d)" % _TRACE_MEL,
                    mode=InputMode(
                        Audio(**recording), encode="encode_style_reference",
                        label="style reference",
                        doc="The recording the delivery is taken from: its "
                            "first six seconds as a mel spectrogram, for the "
                            "conditioning encoder (#conditioning). " + reference_doc)),
                "audio": Input(
                    "torch.randn(1, %d)" % _TRACE_AUDIO,
                    mode=InputMode(
                        Audio(**recording), encode="encode_voice_reference",
                        label="voice reference",
                        doc="The recording the timbre is taken from: up to "
                            "thirty seconds at 16 kHz, for the speaker encoder "
                            "(#voice). " + reference_doc)),
                "temperature": Input("torch.tensor([%s])" % self.temperature,
                                     doc="Sampling temperature of the GPT."),
            },
            outputs=[speech], devices=no_mps,
            # the sampling loop's token buffer, KV cache and stop logic are all
            # written for one utterance, and the text and reference encoders
            # take one each; several at once is running them one after another
            batch=False,
            # the joints between stages are exactly the `mark()` aliases the
            # pipeline tags its main blocks' outputs with -- keeping them means
            # looking at the speech does not throw away the latents behind it,
            # and bending the decoder resumes from them instead of resampling
            retain=retain_aliases,
            before_retrace="_before_retrace")
        return {"forward": forward, "decode": decoder}

    # -- loading --

    @staticmethod
    def load_model(model_path: str = _DEFAULT_MODEL, device="cpu", **kwargs):
        """Load XTTS-v2 from a checkpoint directory, or from the Hub.

        A local directory must hold ``config.json``, ``model.pth`` and
        ``vocab.json``; anything else is treated as a Hub repo id and fetched.
        """
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError as exc:                     # pragma: no cover
            raise BendingXTTSException(
                "could not import coqui TTS (%s: %s). It needs python<3.12 and a "
                "transformers it supports: the maintained fork is `pip install "
                "coqui-tts`, the original `TTS` 0.22 breaks on transformers>=5"
                % (type(exc).__name__, exc)
            ) from exc

        import os
        directory = str(model_path)
        if not os.path.isdir(directory):
            try:
                from huggingface_hub import snapshot_download
                directory = snapshot_download(repo_id=directory)
            except Exception as exc:
                raise BendingXTTSException(
                    "could not obtain XTTS checkpoint %s, got: %s" % (model_path, exc))

        try:
            config = XttsConfig()
            config.load_json(os.path.join(directory, "config.json"))
            model = Xtts.init_from_config(config)
            model.load_checkpoint(config, checkpoint_dir=directory, eval=True, **kwargs)
        except Exception as exc:
            raise BendingXTTSException(
                "could not load XTTS model from %s, got: %s" % (directory, exc))
        return model.to(device), config

    @staticmethod
    def is_loadable(path) -> bool:
        try:
            BendedXTTS.load_model(path)
            return True
        except Exception:
            return False

    # -- tracing --

    def bend_model(self, model):
        """Trace the decoder: GPT latents and a speaker embedding, to a waveform.

        On its own the GPT is deliberately not traced -- it writes one token at
        a time, and how many depends on what it has written so far. With
        ``full`` its single *step* is, and the loop that calls it stays here
        (see ``pipeline.py``), along with the voice encoders and the pass that
        turns the finished tokens into latents.
        """
        dev = self.device
        decoder_inputs = dict(
            latents=torch.randn(1, self.trace_frames, self._latent_dim, device=dev),
            g=torch.randn(1, self._speaker_dim, 1, device=dev))
        if not self.full:
            model.trace("forward", **decoder_inputs)
            return

        text = torch.randint(0, 200, (1, _TRACE_TEXT), device=dev)
        mel = torch.randn(1, 80, _TRACE_MEL, device=dev)
        audio = torch.randn(1, _TRACE_AUDIO, device=dev)
        model.trace("forward", text_ids=text, mel=mel, audio=audio,
                    temperature=torch.tensor([self.temperature], device=dev),
                    _loop_policy=self._loop_policy())
        model.trace("decode", **decoder_inputs)

    def _loop_policy(self) -> dict:
        """How the sampling loop is drawn: folded into nodes of `loop_pack`
        steps (0: the whole loop, one node), the first `loop_open` of them
        spelled out in full."""
        pack = self.loop_pack if self.loop_pack > 0 else self.trace_tokens
        policy = {"mode": "pack", "pack": pack}
        if self.loop_open > 0:
            policy["unroll_range"] = (0, min(self.loop_open, self.trace_tokens))
        return policy

    def _before_retrace(self, fn):
        """The loop's length lives in the pipeline and its drawing in the trace
        settings; the options only hold the numbers. Push both before `forward`
        is traced again, or a changed `audio tokens` would retrace the old loop."""
        self._pipeline.n_tokens = int(self.trace_tokens)
        return {"_loop_policy": self._loop_policy()}

    # -- properties --

    @property
    def config(self):
        return self._config

    @property
    def sample_rate(self) -> int:
        """The decoder's output rate (24 kHz for XTTS-v2), read by the viewer
        to play an audio-returning callback's result at the right speed."""
        return int(self._full.hifigan_decoder.output_sample_rate)

    @property
    def speakers(self) -> List[str]:
        """The speaker ids this checkpoint ships with."""
        manager = getattr(self._full, "speaker_manager", None)
        if manager is None:
            return []
        return list(manager.speakers.keys())

    @property
    def latents(self):
        """The last utterance's ``(gpt_latents, speaker_embedding)``, if any."""
        return self._latents

    # -- conditioning --

    def conditioning(self, speaker_wav: Optional[Union[str, List[str]]] = None):
        """``(gpt_cond_latent, speaker_embedding)`` for a reference wav, or for
        the configured built-in speaker when none is given."""
        if speaker_wav:
            paths = [speaker_wav] if isinstance(speaker_wav, str) else list(speaker_wav)
            if self.full:
                paths = [os.path.expanduser(str(p)) for p in paths]
                missing = [p for p in paths if not os.path.isfile(p)]
                if missing:
                    raise BendingXTTSException("reference audio not found: %s" % ", ".join(missing))
            return self._full.get_conditioning_latents(audio_path=paths)
        manager = getattr(self._full, "speaker_manager", None)
        if manager is None or self.speaker not in getattr(manager, "speakers", {}):
            raise BendingXTTSException(
                "no reference wav given and speaker %r is not in this checkpoint "
                "(have: %s)" % (self.speaker, ", ".join(self.speakers[:8]) or "none"))
        return tuple(manager.speakers[self.speaker].values())

    # -- inference --

    def speak(self, text: str = "the graph is the instrument",
              speaker_wav: Optional[str] = None, seed: Optional[int] = None) -> torch.Tensor:
        """Say ``text``, decoding through the bended graph.

        With ``full`` this runs the ``forward`` graph -- the encoders, the GPT
        and the decoder -- so a bending anywhere on the way is heard.
        """
        speaker_wav = speaker_wav or self.reference or None
        seed = seed if seed is not None else (self.seed if self.seed >= 0 else None)
        if self.full:
            return self._speak_full(text, speaker_wav, seed)
        gpt_cond_latent, speaker_embedding = self.conditioning(speaker_wav)
        if seed is not None:
            torch.manual_seed(seed)

        original = self._full.hifigan_decoder
        self._full.hifigan_decoder = _DecoderShim(self._model)
        try:
            out = self._full.inference(
                text=text, language=self.language,
                gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding,
                temperature=self.temperature)
        finally:
            self._full.hifigan_decoder = original

        latents = torch.as_tensor(out["gpt_latents"])
        self._latents = (latents, torch.as_tensor(speaker_embedding))
        return self._as_audio(out["wav"])

    def _speak_full(self, text, speaker_wav, seed) -> torch.Tensor:
        inputs = self.encode_utterance(text, speaker_wav)
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            wav, latents, g, codes = self._model.forward(**inputs)
        # the graph ran for `trace_tokens` steps and silenced what came after
        # the GPT's stop; here the silence is cut off, and the latents kept for
        # `decode_latents` are the utterance's own
        n, _ = self._pipeline.utterance_length(codes)
        self._latents = (latents[:, :n].detach().cpu(), g.detach().cpu())
        keep = int(self._pipeline.samples_for(n))
        return self._as_audio(wav.reshape(1, -1)[:, :keep])

    _REF_RATE = 22050

    def _reference(self, value):
        """A recording at 22.05 kHz, mono, up to thirty seconds -- what both
        encoders' front-ends start from, as XTTS's ``get_conditioning_latents``
        does. ``value`` is a path, or ``(waveform [C, L], sample_rate)`` as the
        bench hands over an uploaded file; empty means the `reference` option."""
        import torchaudio
        if isinstance(value, list):
            if len(value) != 1:
                raise BendingXTTSException("one reference recording at a time, got %d" % len(value))
            value = value[0]
        rate = self._REF_RATE
        if isinstance(value, tuple) and len(value) == 2 and torch.is_tensor(value[0]):
            wav, sr = value
            wav = wav.float().reshape(-1, wav.shape[-1]).mean(dim=0, keepdim=True)   # mono [1, L]
            if int(sr) != rate:
                wav = torchaudio.functional.resample(wav, int(sr), rate)
            audio = wav.clip(-1, 1)
        else:
            path = os.path.expanduser(str(value or "").strip() or self.reference or "")
            if not path:
                raise BendingXTTSException(
                    "no reference recording: load a file, give a path, or set "
                    "the `reference` option")
            if not os.path.isfile(path):
                raise BendingXTTSException("reference audio not found: %s" % path)
            audio = read_audio(path, rate)
        return audio[:, :rate * 30].to(self.device)

    def encode_style_reference(self, path) -> torch.Tensor:
        """A recording to ``mel``: the mel spectrogram of its first six seconds,
        normalized as the conditioning encoder expects. The front-end (STFT,
        mel norms) is the part that stays outside the graph."""
        import TTS.tts.models.xtts as xtts_module
        rate = self._REF_RATE
        chunk = self._reference(path)[:, :rate * 6]
        if chunk.shape[-1] < rate * 0.33:
            raise BendingXTTSException(
                "reference audio too short (%.2fs, XTTS needs at least 0.33s)" % (chunk.shape[-1] / rate))
        mel = xtts_module.wav_to_mel_cloning(
            chunk, mel_norms=self._full.mel_stats.cpu(), n_fft=2048, hop_length=256,
            win_length=1024, power=2, normalized=False, sample_rate=rate,
            f_min=0, f_max=8000, n_mels=80)
        return mel.to(self.device)

    def encode_voice_reference(self, path) -> torch.Tensor:
        """A recording to ``audio``: up to thirty seconds at 16 kHz, for the
        speaker encoder."""
        import torchaudio
        return torchaudio.functional.resample(self._reference(path), self._REF_RATE, 16000)

    def reference_inputs(self, path):
        """A recording as both encoders take it: ``(mel, audio)``."""
        return self.encode_style_reference(path), self.encode_voice_reference(path)

    def encode_utterance(self, text, speaker_wav: Optional[str] = None):
        """A caption and the reference voice to the ``forward`` graph's inputs.

        What the caption input on ``forward`` runs. Everything it produces is
        the outside-the-graph part: tokenizing, and preparing the recording.
        """
        if not self.full:
            raise BendingXTTSException(
                "a caption can only be run through the whole pipeline with full=True; "
                "this interface holds just the decoder")
        wav = speaker_wav or self.reference
        if not wav:
            raise BendingXTTSException(
                "the full pipeline starts from a reference recording: set the "
                "`reference` option to the path of one")
        mel, audio = self.reference_inputs(wav)
        return {"text_ids": self.encode_text(text), "mel": mel, "audio": audio,
                "temperature": torch.tensor([self.temperature], device=self.device)}

    def encode_text(self, text) -> torch.Tensor:
        """Text to the ``[1, n]`` token ids XTTS's GPT is fed."""
        if not isinstance(text, str):
            text = list(text)
            if len(text) != 1:
                raise BendingXTTSException("XTTS reads one sentence at a time, got %d" % len(text))
            text = text[0]
        ids = self._full.tokenizer.encode(text.strip().lower(), lang=self.language.split("-")[0])
        if len(ids) >= self._full.args.gpt_max_text_tokens:
            raise BendingXTTSException(
                "XTTS can only read %d text tokens, this is %d"
                % (self._full.args.gpt_max_text_tokens - 1, len(ids)))
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def decode_latents(self) -> torch.Tensor:
        """Re-decode the last utterance's latents through the bended graph.

        The point of keeping them: bend, hear it, bend again, without paying
        for the language model each time.
        """
        if self._latents is None:
            raise BendingXTTSException(
                "nothing to decode yet — call `speak` once first; its latents "
                "are kept so you can bend and re-decode them")
        latents, speaker_embedding = self._latents
        decode = self._model.decode if self.full else self._model.forward
        with torch.no_grad():
            wav = decode(latents=latents.to(self.device), g=speaker_embedding.to(self.device))
        return self._as_audio(wav)

    @staticmethod
    def _as_audio(wav) -> torch.Tensor:
        """A ``[1, T]`` waveform, whatever shape the decoder handed back."""
        wav = torch.as_tensor(wav)
        if isinstance(wav, (tuple, list)):
            wav = wav[0]
        wav = wav.detach().float().squeeze()
        return wav.reshape(1, -1)

    # -- export --

    @property
    def scriptable(self):
        return ScriptableState.NotScriptable

    @property
    def nntilde_compatible(self):
        return ScriptableState.NotScriptable
