"""Interfaces over the toy transformers, for exercising the interaction path.

These declare what a real interface declares — ``methods`` (with input modes),
``options``, ``callbacks``, ``tokens`` — over models that load instantly and need
no network. :class:`BendedTinyGPT` is the text→text case GPT-2 covers;
:class:`BendedTinyAudioGen` is the text→audio one it cannot, and is what puts a
declared ``returns: "audio"`` callback under an actual waveform.

Use them from the viewer::

    import torchbend as tb
    from torchbend.test.test_modules.interfaces import BendedTinyGPT, BendedTinyAudioGen
    tb.ui.graph_viewer.run({"tinygpt": BendedTinyGPT(), "tinyaudio": BendedTinyAudioGen()})
"""

from typing import List, Optional, Union

import torch

from ...interfaces.base import Interface
from ...interfaces.spec import (Audio, Callback, Choice, Float, Input, InputMode, Int,
                                Method, Option, Text, Tokens)
from ...tracing import ScriptableState
from .modules.transformer_modules import TinyAudioGen, TinyGPT
from .tokenizers import CharTokenizer, WordTokenizer


__all__ = ["BendedTinyGPT", "BendedTinyAudioGen", "TOKENIZERS"]


TOKENIZERS = {"char": CharTokenizer, "word": WordTokenizer}


class _ToyTextInterface(Interface):
    """What the two share: a tokenizer, prompt encoding, and padding as an option."""

    _imported_callbacks_ = []

    # the inverse of `encode_inputs`, so ids and logits can be read as text.
    # Continuing a sequence is *not* declared here: it belongs to the model that
    # predicts tokens, and the audio one does not.
    tokens = Tokens(decode="decode")

    # `padding_side` is live — it changes what `encode_inputs` produces on the
    # next run. `trace_n_tokens` only matters the next time the graph is taken.
    options = {
        "padding_side": Option(
            Choice(["left", "right"]), get="get_padding_side", set="set_padding_side",
            label="padding side",
            doc="Which side batched prompts are padded on. Left keeps the "
                "last token last, which is what continuation needs."),
        "trace_n_tokens": Option(
            Int(range=(4, 64)), attr="trace_n_tokens", needs="retrace",
            label="trace tokens", doc="Sequence length the graph is traced on."),
    }

    def __init__(self, tokenizer: str = "char", trace_n_batches: int = 2,
                 trace_n_tokens: int = 12, device=torch.device("cpu"), **kwargs):
        if tokenizer not in TOKENIZERS:
            raise ValueError("unknown tokenizer %r (have: %s)"
                             % (tokenizer, ", ".join(TOKENIZERS)))
        self.device = device
        self.tokenizer_name = tokenizer
        self._tokenizer = TOKENIZERS[tokenizer]()
        self.trace_n_batches = trace_n_batches
        self.trace_n_tokens = trace_n_tokens
        super().__init__(self._build_model(**kwargs))

    def _build_model(self, **kwargs):
        raise NotImplementedError

    # -- options --

    def get_padding_side(self) -> str:
        return self._tokenizer.padding_side

    def set_padding_side(self, value: str) -> None:
        self._tokenizer.padding_side = value

    # -- tokenization --

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def encode(self, text: Union[str, List[str]]):
        enc = self._tokenizer(text)
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        return ids, mask, self._position_ids(mask)

    def decode(self, tokens: torch.Tensor, skip_special_tokens: bool = True):
        if tokens.ndim == 1:
            tokens = tokens[None]
        return self._tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens)

    @staticmethod
    def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        return (attention_mask.cumsum(-1) - 1).clamp(min=0)

    def _trace_inputs(self):
        """Random ids of the traced size — the shapes are what tracing records."""
        ids = torch.randint(3, self.vocab_size,
                            (self.trace_n_batches, self.trace_n_tokens), device=self.device)
        mask = torch.ones_like(ids)
        return ids, mask

    # -- export --

    @property
    def scriptable(self):
        return ScriptableState.NotScriptable

    @property
    def nntilde_compatible(self):
        return ScriptableState.NotScriptable


class BendedTinyGPT(_ToyTextInterface):
    """A GPT-2 shaped toy: prompt in, continuation out.

    Generation is a Python loop over the traced ``forward``, exactly as in
    :class:`~torchbend.interfaces.gpt2.BendedGPT2` — the loop is data dependent
    and cannot be traced, which is why it is offered as a callback instead.
    """

    _panel_render_type_ = "text"
    # this one predicts tokens, so an edit in the decoded view can be continued
    tokens = Tokens(decode="decode", logits="token_logits")

    methods = {
        "forward": Method(inputs={"input_ids": Input(mode=InputMode(
            Text(default="the sound of", placeholder="type a prompt to tokenize…"),
            encode="encode_inputs", also_fills=["attention_mask", "position_ids"],
            label="prompt"))}),
    }

    callbacks = {
        "generate": Callback(returns=Text(), args={
            "prompt": Text(default="the sound of", placeholder="prompt the model…"),
            "n_tokens": Int(range=(1, 128)),
            "temperature": Float(range=(0.0, 2.0), step=0.05),
            "top_k": Int(range=(0, 64)),
            "seed": Int(optional=True),
        }),
    }

    def _build_model(self, **kwargs):
        kwargs.setdefault("vocab_size", self.vocab_size)
        return TinyGPT(**kwargs)

    def bend_model(self, model):
        ids, mask = self._trace_inputs()
        # position_ids is passed explicitly: left over as None it is computed
        # inside forward and baked into the graph, and a padded batch then has
        # positions that start at 0 where they should not.
        model.trace("forward", input_ids=ids, attention_mask=mask,
                    position_ids=self._position_ids(mask))

    # -- input mode --

    def encode_inputs(self, text: Union[str, List[str]]) -> dict:
        """Tokenize text into the placeholders ``forward`` takes."""
        input_ids, attention_mask, position_ids = self.encode(text)
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids}

    # -- inference --

    def token_logits(self, input_ids):
        """Logits for a bare batch of ids — every position attended to."""
        return self.logits(input_ids, torch.ones_like(input_ids))

    def logits(self, input_ids, attention_mask):
        out = self._model.forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  position_ids=self._position_ids(attention_mask))
        return out["logits"] if isinstance(out, dict) else out

    def generate(self,
                 prompt: Union[str, List[str]] = "the sound of",
                 n_tokens: int = 24,
                 temperature: float = 1.0,
                 top_k: Optional[int] = 16,
                 seed: Optional[int] = None):
        """Sample n_tokens continuations of prompt through the bended graph."""
        if seed is not None:
            torch.manual_seed(seed)
        input_ids, attention_mask, _ = self.encode(prompt)
        window = self._model._module.config.block_size
        for _ in range(n_tokens):
            ids = input_ids[:, -window:]
            mask = attention_mask[:, -window:]
            logits = self.logits(ids, mask)[:, -1]
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k:
                    k = min(int(top_k), logits.shape[-1])
                    kth = logits.topk(k, dim=-1).values[..., -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(logits.softmax(-1), num_samples=1)
            input_ids = torch.cat([input_ids, nxt], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(nxt)], dim=-1)
        return self.decode(input_ids)


class BendedTinyAudioGen(_ToyTextInterface):
    """Text to sound: a prompt conditions a synthesiser, one forward pass.

    Unlike :class:`BendedTinyGPT` there is no sampling loop — the whole thing is
    the traced graph — so this is also the case where a callback's result is a
    tensor rather than a string.
    """

    _panel_render_type_ = "audio"

    methods = {
        "forward": Method(inputs={"input_ids": Input(mode=InputMode(
            Text(default="a low rain on water", placeholder="describe a sound…"),
            encode="encode_inputs", also_fills=["attention_mask"], label="prompt"))}),
    }

    callbacks = {
        "render": Callback(returns=Audio(), args={
            "prompt": Text(default="a low rain on water", placeholder="describe a sound…"),
        }),
    }

    def _build_model(self, **kwargs):
        kwargs.setdefault("vocab_size", self.vocab_size)
        return TinyAudioGen(**kwargs)

    def bend_model(self, model):
        ids, mask = self._trace_inputs()
        model.trace("forward", input_ids=ids, attention_mask=mask)

    @property
    def sample_rate(self) -> int:
        return self._model._module.sample_rate

    def encode_inputs(self, text: Union[str, List[str]]) -> dict:
        input_ids, attention_mask, _ = self.encode(text)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def render(self, prompt: Union[str, List[str]] = "a low rain on water"):
        """Synthesise audio for a prompt, through the bended graph."""
        input_ids, attention_mask, _ = self.encode(prompt)
        return self._model.forward(input_ids=input_ids, attention_mask=attention_mask)
