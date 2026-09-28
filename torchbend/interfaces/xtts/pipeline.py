"""XTTS end to end: a caption and a reference voice in, speech out, as one graph.

XTTS is a chain -- a pair of encoders that turn a reference recording into a
voice, a GPT that writes audio tokens for a text one at a time, and a HiFiGAN
decoder that turns those tokens' latents into a waveform -- and bending it means
being able to reach any link. :class:`XTTSPipeline.forward` is the whole chain
as a single traced graph::

    text ids, reference mel, reference audio, temperature
        ├─ style ──────────────► #conditioning ─┐
        ├─ speaker ─────────────► #voice ───────┼──────────────┐
        └─ GPT sampling loop ► #tokens ─► #latents ─► decoder ─► #speech

so a bending anywhere on it -- the encoders, the language model deciding what
is said, the decoder -- is heard, and the graph viewer shows all of it at once.
The ``#name``s are aliases -- see :func:`torchbend.mark` -- tagging each main
block's output as it flows through the one graph, so they read like the
diagram above and are addressable as bending targets on their own
(``bend(cb, "#latents")``) without hunting for the node fx happened to name it.

**The loop is the hard part.** The GPT writes its tokens one at a time and
stops when it decides to, and a graph is a fixed sequence of operations. Two
things make it fit. The loop runs for a *fixed* number of steps (``n_tokens``),
each step rewriting a preallocated token buffer, with tokens after the stop
token forced to be the stop token; the audio is cut back to where the stop fell
once the graph has run. And it is written with :func:`torchbend.loop`, which
puts one opaque node in the graph per *pack* of steps rather than a copy of the
GPT per step -- unrolled, 100 steps of a 30-layer transformer would be 200 000
nodes. What that costs is visibility: activations *inside* a packed step are not
in the graph. Weight bendings reach every step regardless, the carry between
packs is bendable, and ``open`` steps (the first few, by default) are inlined
completely, so a GPT activation can be bent in them.

**It still has a KV cache**, despite that. Step 0 is a real pass over
``[conditioning | text | start token]`` -- there is nothing to attend to yet --
and it also seeds a cache of every layer's keys and values, as fixed-size
buffers threaded through the loop's carry (so they fit the same "shape and
dtype survive an iteration" contract the token buffer already meets). Every
step after that computes only the one new token, attending to the cache instead
of recomputing the tokens before it; causal attention makes the two give the
same answer, so this is not an approximation, just not redoing work whose
result cannot have changed. This costs nothing in bendability beyond what
packing already costs: a packed step's internals were already invisible, and
an *opened* step past the first is now the price of one token, not of the
whole prefix over again -- opening more of them got much cheaper too.

``decode`` is also a graph of its own -- latents and a voice to a waveform,
without the GPT -- for bending just the last link, or for a caller (see the
interface) that already has latents from somewhere else. ``style``, ``speaker``
and ``gpt_latents`` are plain methods, not separate graphs: what they compute
only exists as part of ``forward``'s single trace, addressed by the aliases
above rather than by a graph of their own.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from ...tracing import mark
from ...tracing.loop import current_loop_policy, loop


__all__ = ["XTTSPipeline"]


class _GPT(nn.Module):
    """The parts of XTTS's ``GPT`` that generation and the latent pass use.

    Holds the *same* modules as the original, not copies, so this is a view of
    the language model with its unused halves (the text head, the training-only
    pieces, the HF inference wrapper that aliases the same weights) left out.
    What is registered here is what shows up in the graph and can be bent.
    """

    def __init__(self, gpt):
        super().__init__()
        self.text_embedding = gpt.text_embedding
        self.text_pos_embedding = gpt.text_pos_embedding
        self.mel_embedding = gpt.mel_embedding
        self.mel_pos_embedding = gpt.mel_pos_embedding
        self.gpt = gpt.gpt
        self.final_norm = gpt.final_norm
        self.mel_head = gpt.mel_head
        self.model_dim = int(gpt.model_dim)
        self.n_layer = int(gpt.gpt.config.n_layer)
        self.n_head = int(gpt.gpt.config.n_head)
        self.head_dim = int(gpt.gpt.config.n_embd) // self.n_head
        #: The model's own absolute cap on text length -- not this call's text
        #: length, which varies. The incremental cache is sized against this so
        #: one trace serves any caption without knowing its length in advance.
        self.max_text_tokens = int(gpt.text_pos_embedding.seq_len)
        self.start_text = int(gpt.start_text_token)
        self.stop_text = int(gpt.stop_text_token)
        self.start_audio = int(gpt.start_audio_token)
        self.stop_audio = int(gpt.stop_audio_token)
        self.n_audio_tokens = int(gpt.mel_head.out_features)

    def hidden(self, cond_latents, text_ids, mel_ids, use_cache: bool = False):
        """The transformer over ``[conditioning | text | audio tokens]``.

        ``text_ids`` are raw tokenizer ids: the start and stop markers are
        added here. ``mel_ids`` are used as given -- the caller decides what
        precedes the first token. Returns the normed hidden states of the audio
        part only, or ``(hidden, past_key_values)`` with ``use_cache`` -- the
        per-layer keys/values for every position just processed, used to seed
        :meth:`step_cached` (see the module docstring's second half).
        """
        text_ids = F.pad(F.pad(text_ids, (0, 1), value=self.stop_text),
                         (1, 0), value=self.start_text)
        text_emb = self.text_embedding(text_ids) + self.text_pos_embedding(text_ids)
        mel_emb = self.mel_embedding(mel_ids) + self.mel_pos_embedding(mel_ids)
        emb = torch.cat([cond_latents, text_emb, mel_emb], dim=1)
        out = self.gpt(inputs_embeds=emb, use_cache=use_cache, return_dict=True)
        hidden = self.final_norm(out.last_hidden_state[:, -mel_emb.shape[1]:])
        return (hidden, out.past_key_values) if use_cache else hidden

    def step_cached(self, key_cache, val_cache, valid_len: int, position: int, token):
        """One new mel token's hidden state, attending to ``valid_len`` cached
        positions instead of recomputing them.

        ``token`` ``[B, 1]`` is the one new audio token; ``position`` its index
        within the mel sequence (for its positional embedding -- position 0 is
        the start token, already covered by the prefill). ``key_cache``/
        ``val_cache`` are the fixed-size buffers :meth:`hidden`'s cache seeded;
        only their first ``valid_len`` slots are read. Returns ``(hidden, new_key,
        new_value)`` -- the new position's own hidden state and its keys/values,
        for the caller to write into the buffers for the next call.
        """
        emb = self.mel_embedding(token) + self.mel_pos_embedding.get_fixed_embedding(
            position, token.device)
        past = DynamicCache.from_legacy_cache(
            [(key_cache[l, :, :, :valid_len], val_cache[l, :, :, :valid_len])
             for l in range(self.n_layer)])
        out = self.gpt(inputs_embeds=emb, past_key_values=past, use_cache=True,
                       return_dict=True)
        hidden = self.final_norm(out.last_hidden_state)
        new_key = torch.stack([out.past_key_values[l][0][:, :, -1, :]
                               for l in range(self.n_layer)])
        new_val = torch.stack([out.past_key_values[l][1][:, :, -1, :]
                               for l in range(self.n_layer)])
        return hidden, new_key, new_val

    def modules_used_by_sampling(self):
        """What one sampling step calls: the modules whose weights a packed
        loop has to be handed. Not the conditioning encoders, which run once."""
        return [self.text_embedding, self.text_pos_embedding, self.mel_embedding,
                self.mel_pos_embedding, self.gpt, self.final_norm, self.mel_head]


class _Style(nn.Module):
    """The reference mel to the GPT's conditioning latents.

    The checkpoint nests this inside the GPT module; it is pulled out here
    so it traces as its own block (a distinct box in the graph) instead of
    disappearing into "gpt" -- it is the voice, extracted, not the language
    model reading it.
    """

    def __init__(self, gpt):
        super().__init__()
        self.conditioning_encoder = gpt.conditioning_encoder
        self.use_perceiver = bool(gpt.use_perceiver_resampler)
        if self.use_perceiver:
            self.conditioning_perceiver = gpt.conditioning_perceiver

    def forward(self, mel):
        """``[B, 80, S]`` reference mel to ``[B, 1024, 32]`` conditioning."""
        conds = self.conditioning_encoder(mel)
        if self.use_perceiver:
            conds = self.conditioning_perceiver(conds.permute(0, 2, 1)).transpose(1, 2)
        return conds


class XTTSPipeline(nn.Module):
    """XTTS's voice encoders, GPT and HiFiGAN decoder as one module.

    Two graphs: ``forward`` (caption and reference voice to speech) and
    ``decode`` (decoder latents and a voice to speech, alone). Everything
    ``forward`` does before the decoder -- the encoders, the sampling loop --
    is plain methods rather than graphs of their own; their outputs are
    tagged with :func:`torchbend.mark` instead (see the module docstring), so
    they stay addressable without a separate trace to hold them. The XTTS
    model itself is held by reference, outside the module tree: the pipeline
    needs its tokenizer and mel statistics, not to bend them.
    """

    #: XTTS's own sampling settings. Baked into the graph; ``temperature`` is an
    #: input of it instead, since it is the one worth turning while listening.
    top_k = 50
    top_p = 0.85
    repetition_penalty = 10.0

    def __init__(self, xtts, n_tokens: int = 125):
        super().__init__()
        self.gpt = _GPT(xtts.gpt)
        self.style_encoder = _Style(xtts.gpt)
        # the speaker encoder stays reached through the decoder, not aliased to
        # a second top-level attribute: a get_attr node's target is the
        # parameter's *canonical* registered path, not however the call site
        # reached it, so a second attribute pointing at the same submodule
        # would put its weights under "decoder.speaker_encoder" while its
        # compute ops (grouped by the actual call site) landed under
        # "speaker_encoder" -- two different boxes for one block. Nested is
        # consistent; aliased is not. `module_depth >= 2` already draws
        # "decoder > speaker_encoder" as its own box.
        self.decoder = xtts.hifigan_decoder
        #: Audio tokens the sampling loop runs for -- about 21 per second. A
        #: graph is a fixed sequence of steps, so this is decided at trace time.
        self.n_tokens = int(n_tokens)
        object.__setattr__(self, "_full", xtts)

    # -- the whole process --

    def forward(self, text_ids, mel, audio, temperature):
        """Caption and reference voice to speech.

        ``text_ids`` ``[1, T]`` are the tokenized caption; ``mel`` ``[1, 80, S]``
        and ``audio`` ``[1, A]`` (16 kHz) are the reference recording, as the
        two encoders take it. ``temperature`` ``[1]`` is the sampling
        temperature. Returns ``(wav, latents, g, codes)``: the waveform, then
        what led to it -- the decoder's two inputs, and the audio tokens the GPT
        chose -- which is what lets the utterance be re-decoded and trimmed
        without running the graph again.

        ``wav`` is ``n_tokens`` worth of audio however early the GPT stopped;
        what it decided to say ends at the first stop token in ``codes``.
        """
        g = self.speaker(audio).unsqueeze(-1)                       # [1, 512, 1]
        cond = self.style(mel).transpose(1, 2)                      # [1, 32, 1024]
        codes = self.sample(cond, text_ids, temperature)            # [1, n_tokens]
        latents = self.gpt_latents(cond, text_ids, codes)
        wav = self.decode(latents, g)
        # what the GPT decided to say ends at its first stop token; what the
        # decoder made of the stop-token filler after it is silenced, so the
        # waveform is the utterance and not the utterance plus a tail
        stops = codes == self.gpt.stop_audio
        said = torch.where(stops.any(-1), stops.float().argmax(-1) + 1,
                           torch.full_like(codes[:, 0], codes.shape[-1]))
        keep = self.samples_for(said)
        heard = torch.arange(wav.shape[-1], device=wav.device) < keep.view(-1, 1, 1)
        heard = mark(wav * heard.to(wav.dtype),
                     description="What is heard: the waveform, silenced after the GPT's "
                                 "stop token. The graph always runs every sampling step; "
                                 "this is where the speech it decided on ends.",
                     meta={"step": 6, "title": "utterance", "axes": "batch, channel, samples",
                           "sample_rate": self.sample_rate})
        return heard, latents, g, codes

    def sample(self, cond_latents, text_ids, temperature):
        """The sampling loop: ``n_tokens`` audio tokens for a text, from the GPT.

        Step 0 is a real pass over ``[conditioning | text | start token]`` --
        there is nothing to cache before it exists -- and it seeds a key/value
        cache. Every step after it computes only the *one new token*, attending
        to that cache instead of recomputing everything before it; causal
        attention makes the two give the same answer (see the module
        docstring) -- this is just not redoing work whose result cannot have
        changed. After the stop token every token is the stop token.

        Whether a step's own computation lands as bendable graph nodes or inside
        an opaque packed one is exactly :func:`torchbend.loop`'s ``pack``/
        ``unroll_range`` policy, unaffected by any of this: a packed step was
        already invisible to activation bending before it had a cache, so
        giving it one costs nothing there. What changes is that an *opened*
        step past the first is now the cost of one token, not of the whole
        prefix again -- opening more of them is far cheaper than it was.
        """
        gpt, n = self.gpt, self.n_tokens
        vocab, start, stop = gpt.n_audio_tokens, gpt.start_audio, gpt.stop_audio
        B = cond_latents.shape[0]
        cache_len = cond_latents.shape[1] + gpt.max_text_tokens + 2 + n + 1

        # every slot starts as the stop token: once the GPT has stopped, each
        # later step would write exactly that, so a step skipped after the stop
        # (see `step`) leaves the buffer as running it would have
        buffer = torch.cat([
            torch.full((1, 1), start, dtype=torch.long, device=text_ids.device),
            torch.full((1, n), stop, dtype=torch.long, device=text_ids.device)], dim=1)
        # what `generate` penalizes: the placeholder id 1 XTTS hands it for the
        # prefix, and the start token -- then every token drawn
        ids = torch.arange(vocab, device=text_ids.device)
        seen = (ids == 1) | (ids == start)
        done = torch.zeros(1, dtype=torch.bool, device=text_ids.device)
        key_cache = torch.zeros(gpt.n_layer, B, gpt.n_head, cache_len, gpt.head_dim,
                                dtype=cond_latents.dtype, device=text_ids.device)
        val_cache = torch.zeros_like(key_cache)

        def step(i, carry):
            buffer, seen, done, cond, text, temp, key_cache, val_cache = carry
            # The loop has a fixed length (a graph is a fixed sequence of
            # steps), but once the GPT has said its stop token, every step left
            # forces the stop token again and changes nothing that is read
            # afterwards. When the step is really running -- a packed node
            # replaying it, not a trace drawing it, where `done` is not a value
            # yet -- skip it: generation then costs what was said, not the
            # maximum length.
            if current_loop_policy() is None and bool(done):
                return carry
            prefix_len = cond.shape[1] + text.shape[1] + 2   # + start/stop text tokens
            if i == 0:
                # nothing exists to attend to yet: a real pass over the whole
                # prefix, which also seeds the cache for every step after it
                hidden, past = gpt.hidden(cond, text, buffer[:, :1], use_cache=True)
                logits = gpt.mel_head(hidden)[0, -1]
                seed_len = prefix_len + 1                        # + mel position 0
                idx = torch.arange(seed_len, device=key_cache.device)
                keys = torch.stack([past[l][0] for l in range(gpt.n_layer)])   # [L,B,H,seed_len,d]
                vals = torch.stack([past[l][1] for l in range(gpt.n_layer)])
                key_cache = key_cache.index_copy(3, idx, keys)
                val_cache = val_cache.index_copy(3, idx, vals)
            else:
                # buffer[i] (mel position i) was drawn at the end of the
                # previous step; attend to the prefix + mel[0..i-1] already in
                # the cache, then extend the cache by this one new position
                ctx_len = prefix_len + i
                hidden, new_key, new_val = gpt.step_cached(
                    key_cache, val_cache, ctx_len, i, buffer[:, i:i + 1])
                logits = gpt.mel_head(hidden)[0, -1]
                pos = torch.tensor([ctx_len], device=key_cache.device)
                key_cache = key_cache.index_copy(3, pos, new_key.unsqueeze(3))
                val_cache = val_cache.index_copy(3, pos, new_val.unsqueeze(3))
            token = self._draw(logits, seen, temp, self.top_k, self.top_p,
                               self.repetition_penalty)
            token = torch.where(done, torch.full_like(token, stop), token)
            buffer = torch.cat([buffer[:, :i + 1], token.view(1, 1), buffer[:, i + 2:]], dim=1)
            seen = seen | (torch.arange(vocab, device=seen.device) == token)
            done = done | (token == stop)
            return (buffer, seen, done, cond, text, temp, key_cache, val_cache)

        buffer = loop(step, (buffer, seen, done, cond_latents, text_ids, temperature,
                            key_cache, val_cache),
                      n, name="gpt_sample", modules=gpt.modules_used_by_sampling())[0]
        return mark(buffer[:, 1:], name="tokens",
                    description="The speech as discrete tokens, chosen one at a time by "
                                "the GPT: each step reads the conditioning, the text and "
                                "the tokens so far, and draws the next one. After the stop "
                                "token, every position is the stop token.",
                    meta={"step": 3, "title": "audio tokens", "axes": "batch, token",
                          "tokens_per_second": self.tokens_per_second,
                          "top_k": self.top_k, "top_p": self.top_p,
                          "repetition_penalty": self.repetition_penalty,
                          "stop_token": int(self.gpt.stop_audio),
                          "sampling": "temperature is the `temperature` input"})

    @staticmethod
    def _draw(logits, seen, temperature, top_k, top_p, repetition_penalty):
        """One token from ``[V]`` logits, as ``transformers``' ``generate`` would
        under XTTS's settings: repetition penalty over the ``seen`` mask, then
        temperature, top-k, top-p, then a draw. Written without indexing by
        value, so it traces."""
        logits = logits.float()
        penalized = torch.where(logits < 0, logits * repetition_penalty,
                                logits / repetition_penalty)
        logits = torch.where(seen, penalized, logits)
        logits = logits / torch.as_tensor(temperature).clamp_min(1e-5)
        neg_inf = torch.full_like(logits, -float("inf"))
        if 0 < top_k < logits.numel():
            logits = torch.where(logits < torch.topk(logits, top_k).values[-1], neg_inf, logits)
        if top_p < 1.0:
            ordered, order = logits.sort(descending=False)
            drop = ordered.softmax(-1).cumsum(-1) <= (1.0 - top_p)
            last = torch.arange(logits.numel(), device=logits.device) == logits.numel() - 1
            drop = drop & ~last                      # always keep the likeliest
            drop = torch.zeros_like(drop).scatter(0, order, drop)
            logits = torch.where(drop, neg_inf, logits)
        return torch.multinomial(logits.softmax(-1), 1)

    # -- the pieces, each a graph of its own --

    def decode(self, latents, g=None):
        """Decoder latents and a voice to a waveform: the last link, alone."""
        return mark(self.decoder(latents, g=g), name="speech",
                    description="The waveform: the HiFiGAN decoder turns the latents into "
                                "sound, in the voice given by the speaker embedding.",
                    meta={"step": 5, "title": "speech", "axes": "batch, channel, samples",
                          "sample_rate": self.sample_rate})

    def style(self, mel):
        """Reference mel to the GPT's conditioning latents ``[B, 1024, 32]``."""
        return mark(self.style_encoder(mel), name="conditioning",
                    description="How it is said: the style reference's mel spectrogram, "
                                "summarised into 32 latent frames the GPT reads before the "
                                "text -- pace, intonation, recording colour.",
                    meta={"step": 2, "title": "delivery", "axes": "batch, features, latent frames",
                          "source": "mel (style reference)"})

    def speaker(self, audio):
        """16 kHz reference audio ``[B, T]`` to the speaker embedding ``[B, 512]``."""
        return mark(self.decoder.speaker_encoder(audio, l2_norm=True), name="voice",
                    description="Who speaks: a fingerprint of the speaker's timbre, from "
                                "the voice reference. Only the decoder reads it -- the "
                                "words and the delivery come from elsewhere.",
                    meta={"step": 1, "title": "voice", "axes": "batch, speaker features",
                          "source": "audio (voice reference, 16 kHz)"})

    def gpt_latents(self, cond_latents, text_ids, codes):
        """The finished audio tokens ``[B, n]`` to decoder latents ``[B, n, 1024]``.

        XTTS reads its own tokens back through the transformer to get the
        latents the decoder wants: the hidden state at each position, having
        seen the tokens before it. That is this, with the start token in front
        and the last token (the stop) dropped, since position ``i`` is the one
        that predicted token ``i``.
        """
        start = torch.full_like(codes[:, :1], self.gpt.start_audio)
        mel_ids = torch.cat([start, codes[:, :-1]], dim=1)
        return mark(self.gpt.hidden(cond_latents, text_ids, mel_ids), name="latents",
                    description="The GPT reads its own tokens back: the hidden state at "
                                "each position -- a continuous version of the tokens, with "
                                "the text and the delivery folded in -- is what the decoder "
                                "turns into sound.",
                    meta={"step": 4, "title": "latents", "axes": "batch, token, features"})

    # -- what a graph cannot hold --

    @property
    def sample_rate(self) -> int:
        return int(self.decoder.output_sample_rate)

    @property
    def tokens_per_second(self) -> float:
        """Audio tokens per second of speech (about 21.5 for XTTS-v2): plain
        arithmetic on the decoder's settings, so reading it while tracing adds
        nothing to the graph (unlike `samples_for`, which is tensors)."""
        d = self.decoder
        per_token = d.ar_mel_length_compression * (d.output_sample_rate / d.input_sample_rate)
        return round(d.output_sample_rate / per_token, 2)

    def samples_for(self, n_tokens):
        """How many waveform samples ``n_tokens`` audio tokens decode to.

        The decoder resamples its latents twice and then upsamples by its hop
        length, flooring at each step, so the length is not proportional to the
        token count -- this is the decoder's own arithmetic. Takes an int or a
        tensor; returns a tensor, so it can be part of the graph.
        """
        d = self.decoder
        n = torch.as_tensor(n_tokens).double()
        n = torch.floor(n * (d.ar_mel_length_compression / d.output_hop_length))
        if d.output_sample_rate != d.input_sample_rate:
            n = torch.floor(n * (d.output_sample_rate / d.input_sample_rate))
        return (n * d.output_hop_length).long()

    @property
    def stop_audio(self) -> int:
        return self.gpt.stop_audio

    def utterance_length(self, codes) -> Tuple[int, int]:
        """``(n, of)``: the GPT stopped after ``n`` of ``of`` tokens, the stop
        token counted, as XTTS counts it. Without a stop, all of them."""
        codes = codes.reshape(-1)
        stops = (codes == self.gpt.stop_audio).nonzero()
        return (int(stops[0]) + 1 if len(stops) else codes.numel()), codes.numel()
