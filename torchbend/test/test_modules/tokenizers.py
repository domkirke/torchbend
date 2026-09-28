"""Toy tokenizers with a HuggingFace-shaped surface.

The point is not tokenization quality — these are deliberately trivial — but the
*interface*: ``__call__`` returning ``input_ids``/``attention_mask``,
``batch_decode``, ``padding_side``, ``pad_token``, ``vocab_size``. That is the
surface :class:`~torchbend.interfaces.gpt2.BendedGPT2` talks to, so anything
built against these swaps to a real ``transformers`` tokenizer unchanged, and
anything that breaks here would have broken there.

Two flavours, because they fail differently and both are worth having under the
UI:

``CharTokenizer``  every string round-trips exactly, so what you typed is what
                   the model saw. Sequences are long.
``WordTokenizer``  short sequences and legible output, at the cost of ``<unk>``
                   for anything outside its small lexicon.
"""

from typing import Dict, List, Union

import torch


__all__ = ["CharTokenizer", "WordTokenizer", "ToyTokenizer"]


class ToyTokenizer:
    """Shared HF-shaped behaviour: specials, padding, batching, decoding."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self, padding_side: str = "left"):
        self.padding_side = padding_side
        self._itos: List[str] = []
        self._stoi: Dict[str, int] = {}

    # -- vocabulary ------------------------------------------------------------

    def _build(self, symbols):
        # specials first so their ids are stable no matter what the corpus is
        self._itos = [self.pad_token, self.eos_token, self.unk_token] + list(symbols)
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    @property
    def pad_token_id(self) -> int:
        return self._stoi[self.pad_token]

    @property
    def eos_token_id(self) -> int:
        return self._stoi[self.eos_token]

    @property
    def unk_token_id(self) -> int:
        return self._stoi[self.unk_token]

    # -- subclass hooks --------------------------------------------------------

    def _split(self, text: str) -> List[str]:
        raise NotImplementedError

    def _join(self, symbols: List[str]) -> str:
        raise NotImplementedError

    # -- HF-shaped API ---------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        return [self._stoi.get(s, self.unk_token_id) for s in self._split(text)]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        specials = {self.pad_token_id, self.eos_token_id}
        out = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in specials:
                continue
            out.append(self._itos[i] if 0 <= i < len(self._itos) else self.unk_token)
        return self._join(out)

    def batch_decode(self, ids, skip_special_tokens: bool = True) -> List[str]:
        if isinstance(ids, torch.Tensor) and ids.ndim == 1:
            ids = ids[None]
        return [self.decode(row, skip_special_tokens) for row in ids]

    def __call__(self, text: Union[str, List[str]], return_tensors: str = "pt",
                 padding: bool = True) -> Dict[str, torch.Tensor]:
        """Tokenize and pad a batch, the way a HF tokenizer would.

        An empty prompt still yields one token: a zero-length sequence gives the
        model nothing to attend to, and every downstream shape would be 0.
        """
        texts = [text] if isinstance(text, str) else list(text)
        rows = [self.encode(t) or [self.eos_token_id] for t in texts]
        width = max(len(r) for r in rows)

        ids, mask = [], []
        for row in rows:
            pad = [self.pad_token_id] * (width - len(row))
            keep = [0] * (width - len(row))
            if self.padding_side == "left":
                ids.append(pad + row)
                mask.append(keep + [1] * len(row))
            else:
                ids.append(row + pad)
                mask.append([1] * len(row) + keep)

        if return_tensors != "pt":
            return {"input_ids": ids, "attention_mask": mask}
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long)}


class CharTokenizer(ToyTokenizer):
    """Character level over printable ASCII: any string survives the round trip."""

    def __init__(self, padding_side: str = "left"):
        super().__init__(padding_side)
        self._build([chr(c) for c in range(32, 127)])

    def _split(self, text):
        return list(text)

    def _join(self, symbols):
        return "".join(symbols)


#: A small lexicon, chosen so an untrained model still emits something readable.
_TOY_WORDS = (
    "the a an and or but if then than that this these those of to in on at by "
    "for with from into over under about through while when where how why what "
    "i you he she it we they me him her us them my your his its our their "
    "is are was were be been being am do does did done have has had "
    "can could will would shall should may might must "
    "say says said know knew think thought see saw hear heard feel felt "
    "make made take took give gave find found want wanted need needed "
    "go goes went come came look looked seem seemed become became "
    "time day night year world life hand eye head place work word thing "
    "sound light water fire air earth wind rain sun moon star sky sea "
    "good bad new old great small long short high low far near same other "
    "not no yes very much many more most less least just only also even still "
    "here there now then always never often sometimes again once "
    "like as so because although however therefore meanwhile perhaps maybe"
).split()


class WordTokenizer(ToyTokenizer):
    """Word level over a small fixed lexicon; anything else becomes ``<unk>``.

    Punctuation is split off rather than glued to a word, so "sound," and
    "sound" are the same token — with a lexicon this small, every collapse of
    two surface forms into one is worth having.
    """

    _PUNCT = ".,;:!?'\"()-"

    def __init__(self, padding_side: str = "left", words=None):
        super().__init__(padding_side)
        self._build(list(words or _TOY_WORDS) + list(self._PUNCT))

    def _split(self, text):
        out = []
        for raw in text.lower().split():
            lead = ""
            while raw and raw[0] in self._PUNCT:
                out.append(raw[0]); raw = raw[1:]
            trail = []
            while raw and raw[-1] in self._PUNCT:
                trail.insert(0, raw[-1]); raw = raw[:-1]
            if raw:
                out.append(raw)
            out.extend(trail)
            del lead
        return out

    def _join(self, symbols):
        out = ""
        for s in symbols:
            if not out:
                out = s
            elif s in self._PUNCT:
                out += s
            else:
                out += " " + s
        return out
