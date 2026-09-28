import os
from pathlib import Path
from typing import List, Optional, Union

import torch
import transformers
from transformers import AutoTokenizer, GPT2LMHeadModel

from ..spec import (Callback, Choice, Float, Input, InputMode, Int, Method,
                    Option, Text, Tokens)
from ..base import Interface
from ...tracing import BendedModule, ScriptableState, mark
from ...tracing import tracing_experimental as _tbe


_DEFAULT_MODEL = "gpt2"


class BendingGPT2Exception(Exception):
    pass


def _patch_tracer_globals():
    """Make ``transformers`` resolvable from the proxy_tensor tracing backend.

    That backend rebuilds the traced signature in a temporary module and exec's
    it with ``tracing_experimental``'s globals; GPT2's ``forward`` annotations
    name ``transformers.cache_utils.Cache``, so the module has to be visible
    there or the generated def raises NameError before tracing even starts.
    """
    if not hasattr(_tbe, "transformers"):
        _tbe.transformers = transformers


def mark_gpt2_activations(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    """Alias the transformer's semantic tensors before tracing.

    proxy_tensor traces down to ATen, so activations come out named ``view_64``
    / ``add_112``. Marking the modules whose forward returns a single tensor
    gives the layer-level targets one actually wants to bend, reachable with the
    ``#name`` syntax (``#block_5``, ``#mlp_0``, ``#ln_f``).
    """
    for i, block in enumerate(model.transformer.h):
        mark(block, name="block_%d" % i)
        mark(block.mlp, name="mlp_%d" % i)
        mark(block.ln_1, name="ln_1_%d" % i)
        mark(block.ln_2, name="ln_2_%d" % i)
    mark(model.transformer.ln_f, name="ln_f")
    return model


class BendedGPT2(Interface):
    """Bending interface for GPT-2 style causal language models.

    The traced graph is symbolic in both batch and sequence length, so a single
    trace covers any prompt; generation runs token by token through the *bended*
    graph, which is what makes weight and activation bendings audible in the
    sampled text.

    KV caching is disabled: the cache is a ``transformers.Cache`` object that fx
    cannot carry through a graph. Generation is therefore quadratic in length --
    fine for the short prompts bending experiments use, slow for long ones.
    """

    _imported_callbacks_ = []
    _panel_render_type_ = "text"
    # `decode` is the inverse of `encode_inputs`, which is what lets the viewer
    # show ids and logits as text; `logits` accepts a tensor of ids directly,
    # so it is the continuation hook
    tokens = Tokens(decode="decode", logits="logits")
    _download_subdir = "gpt2"

    methods = {
        "forward": Method(inputs={
            # `input_ids` is a prompt wearing a tensor's clothes. Offering the
            # bench a text mode for it means the tokenizer does the conversion,
            # rather than the user hand-writing token ids. It fills the mask and
            # the positions too: one prompt fixes the sequence length, so all
            # three have to agree.
            "input_ids": Input(mode=InputMode(
                Text(default="i would like to know if",
                     placeholder="type a prompt to tokenize…"),
                encode="encode_inputs", also_fills=["attention_mask", "position_ids"],
                label="prompt")),
        }),
    }

    # Settings that outlive any one call. `padding_side` changes what `encode`
    # produces for a batch and so takes effect on the next run; the trace sizes
    # only matter the next time the model is traced, which is what `needs` says.
    options = {
        "padding_side": Option(
            Choice(["left", "right"]), get="get_padding_side", set="set_padding_side",
            label="padding side",
            doc="Which side batched prompts are padded on. Left keeps the "
                "last token last, which is what continuation needs."),
        "trace_n_batches": Option(
            Int(range=(2, 8)), attr="trace_n_batches", needs="retrace",
            label="trace batches",
            doc="Batch size the graph is traced on. Must stay above 1, or the "
                "batch dimension is baked into the graph."),
        "trace_n_tokens": Option(
            Int(range=(1, 64)), attr="trace_n_tokens", needs="retrace",
            label="trace tokens", doc="Sequence length the graph is traced on."),
    }

    # What a UI needs to know about `generate` that its signature cannot say:
    # `prompt` is prose rather than merely a string, and the result is prose
    # too.  `return_tokens` and `out` are plumbing and stay unlisted, so no
    # widget offers them.
    callbacks = {
        "generate": Callback(returns=Text(), args={
            "prompt": Text(default="i would like to know if", placeholder="prompt the model…"),
            "n_tokens": Int(range=(1, 256)),
            "temperature": Float(range=(0.0, 2.0), step=0.05),
            "top_k": Int(range=(0, 200)),
            "top_p": Float(range=(0.0, 1.0), step=0.01),
            "seed": Int(optional=True),
        }),
    }

    def __init__(self,
                 model_path: str | Path = _DEFAULT_MODEL,
                 trace_n_batches: int = 2,
                 trace_n_tokens: int = 8,
                 mark_activations: bool = True,
                 device: torch.device = torch.device('cpu'),
                 **kwargs):
        self.device = device
        self.trace_n_batches = trace_n_batches
        self.trace_n_tokens = trace_n_tokens
        model, tokenizer = self.load_model(model_path, device=device, **kwargs)
        if mark_activations:
            model = mark_gpt2_activations(model)
        self._tokenizer = tokenizer
        super(BendedGPT2, self).__init__(model)

    # -- loading --

    @staticmethod
    def _resolve_pretrained(model_path: str | Path) -> str:
        """Hub ids ("gpt2", "openai-community/gpt2-xl") are not filesystem paths,
        so they go to transformers untouched; local checkpoints are resolved."""
        if os.path.exists(str(model_path)):
            return str(Path(model_path).resolve())
        if isinstance(model_path, Path):
            raise BendingGPT2Exception("model path %s does not exist" % model_path)
        return str(model_path)

    @staticmethod
    def is_loadable(path) -> bool:
        try:
            BendedGPT2.load_model(path)
            return True
        except Exception:
            return False

    @staticmethod
    def load_model(model_path: str | Path = _DEFAULT_MODEL,
                   device: str | torch.device = "cpu",
                   **kwargs):
        _patch_tracer_globals()
        model_path = BendedGPT2._resolve_pretrained(model_path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = GPT2LMHeadModel.from_pretrained(model_path, **kwargs)
        except Exception as e:
            raise BendingGPT2Exception("could not load GPT2 model %s, got : %s" % (model_path, e))
        # GPT2 ships no pad token; generation batches need one to left-pad with.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = model.eval().to(device)
        return model, tokenizer

    # -- tracing --

    def get_inputs(self, n_batches: int = None, n_tokens: int = None):
        n_batches = n_batches or self.trace_n_batches
        n_tokens = n_tokens or self.trace_n_tokens
        input_ids = torch.randint(0, self.vocab_size, (n_batches, n_tokens), device=self.device)
        attention_mask = torch.ones(n_batches, n_tokens, dtype=torch.long, device=self.device)
        position_ids = torch.arange(n_tokens, device=self.device).expand(n_batches, n_tokens)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

    def bend_model(self, model: BendedModule):
        # Two things the example inputs have to get right:
        #  - position_ids must be a real graph input, because left-padded
        #    batches need positions that do not start at 0, and anything left
        #    as None at trace time is baked into the graph as None;
        #  - trace_n_batches must be > 1, because a batch of 1 is specialized
        #    away into the reshapes and the graph then only accepts one prompt.
        assert self.trace_n_batches > 1, \
            "trace_n_batches must be > 1, otherwise the batch dimension is baked into the graph"
        model.trace("forward", **self.get_inputs(), use_cache=False)
        # `generate` is deliberately not traced.  Two things stop it: the second
        # positional of BendedModule.trace is `trace_method`, so a prompt passed
        # there is read as a backend name; and transformers' own generate cannot
        # be traced at all -- make_fx stops at
        # `GuardOnDataDependentSymNode: Eq(u0, 1)`, because sampling and the
        # stopping criteria branch on values that only exist at run time.
        # Generation is a Python loop over the traced `forward` instead, which
        # is what `self.generate` is, and every bending applies at each step.

    # -- properties --

    @property
    def config(self):
        # Interface.config resolves to the BendedModule's *bending* config; here
        # the useful one is the transformers config of the wrapped model.
        return self.original_model.config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def context_length(self) -> int:
        return self.config.n_positions

    @property
    def n_layers(self) -> int:
        return self.config.n_layer

    @property
    def n_heads(self) -> int:
        return self.config.n_head

    @property
    def hidden_size(self) -> int:
        return self.config.n_embd

    # -- tokenization --

    def encode(self, text: Union[str, List[str]]):
        """Tokenize text into ``(input_ids, attention_mask, position_ids)``."""
        encoded = self._tokenizer(text if isinstance(text, list) else [text],
                                  return_tensors="pt", padding=True)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        return input_ids, attention_mask, self._position_ids(attention_mask)

    # -- options --

    def get_padding_side(self) -> str:
        return self._tokenizer.padding_side

    def set_padding_side(self, value: str) -> None:
        self._tokenizer.padding_side = value

    def encode_inputs(self, text: Union[str, List[str]]) -> dict:
        """Tokenize text into the placeholders ``forward`` takes.

        The bench's text input mode calls this. It returns a mapping rather than
        a tuple so the three stay tied to their names — a prompt sets the
        sequence length, and a mask or positions left over from the trace would
        no longer match it.
        """
        input_ids, attention_mask, position_ids = self.encode(text)
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids}

    def decode(self, tokens: torch.Tensor, skip_special_tokens: bool = True):
        """Detokenize a ``(batch, length)`` id tensor back to text."""
        if tokens.ndim == 1:
            tokens = tokens[None]
        return self._tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens)

    @staticmethod
    def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        """Positions that skip left padding, so a padded batch stays aligned."""
        return (attention_mask.cumsum(-1) - 1).clamp(min=0)

    def write_text(self, path: str, text: Union[str, List[str]]):
        path = Path(path).resolve()
        os.makedirs(path.parent, exist_ok=True)
        with open(path, 'w+') as f:
            f.write("\n".join(text) if isinstance(text, list) else text)

    # -- inference --

    @staticmethod
    def _logits_from_out(out) -> torch.Tensor:
        if isinstance(out, dict):
            return out['logits']
        if isinstance(out, (tuple, list)):
            return out[0]
        return getattr(out, "logits", out)

    def forward(self, x: Union[str, List[str], torch.Tensor], **kwargs) -> torch.Tensor:
        """Run the bended graph on text or token ids, and return the logits."""
        if isinstance(x, (str, list)):
            input_ids, attention_mask, position_ids = self.encode(x)
        else:
            input_ids = x.to(self.device)
            attention_mask = kwargs.pop('attention_mask', None)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(self.device)
            position_ids = kwargs.pop('position_ids', None)
            if position_ids is None:
                position_ids = self._position_ids(attention_mask)
        out = self._model.forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  position_ids=position_ids.to(self.device),
                                  use_cache=False,
                                  **kwargs)
        return self._logits_from_out(out)

    def logits(self, x, **kwargs) -> torch.Tensor:
        return self.forward(x, **kwargs)

    @staticmethod
    def _filter_logits(logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
        if top_k:
            k = min(int(top_k), logits.shape[-1])
            kth = logits.topk(k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            probs = sorted_logits.softmax(-1)
            # keep the smallest prefix whose mass exceeds top_p (always >= 1 token)
            remove = (probs.cumsum(-1) - probs) > top_p
            logits = logits.masked_fill(remove.scatter(-1, sorted_idx, remove), float("-inf"))
        return logits

    def generate(self,
                 prompt: Union[str, List[str]],
                 n_tokens: int = 32,
                 temperature: float = 1.0,
                 top_k: Optional[int] = 50,
                 top_p: Optional[float] = 1.0,
                 seed: Optional[int] = None,
                 return_tokens: bool = False,
                 out: Optional[str] = None):
        """Sample ``n_tokens`` continuations of ``prompt`` through the bended graph.

        ``temperature=0`` switches to greedy decoding. Sampling runs on the
        graph traced by :meth:`bend_model`, so every active bending applies at
        each decoding step.

        Use this rather than transformers' own ``generate``: ``self.model``
        still exposes the wrapped module's ``generate``, but it drives the
        module outside the traced graph and ignores every bending.
        """
        if seed is not None:
            torch.manual_seed(seed)
        input_ids, attention_mask, _ = self.encode(prompt)
        for _ in range(n_tokens):
            # the trace is symbolic in length, but the model's own position
            # table is not: crop to the context window as it fills up.
            window_ids = input_ids[:, -self.context_length:]
            window_mask = attention_mask[:, -self.context_length:]
            logits = self.forward(window_ids, attention_mask=window_mask)[:, -1]
            if temperature == 0:
                next_token = logits.argmax(-1, keepdim=True)
            else:
                logits = self._filter_logits(logits / temperature, top_k, top_p)
                next_token = torch.multinomial(logits.softmax(-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=-1)
        text = self.decode(input_ids)
        if out is not None:
            self.write_text(out, text)
        if return_tokens:
            return text, input_ids
        return text

    # -- export --

    @property
    def scriptable(self):
        return ScriptableState.NotScriptable

    @property
    def nntilde_compatible(self):
        return ScriptableState.NotScriptable


__all__ = ['BendedGPT2', 'BendingGPT2Exception', 'mark_gpt2_activations']
