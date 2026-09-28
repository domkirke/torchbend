"""The toy transformers and their tokenizers.

These exist so the text interaction path — prompt in, tokens through a traced
graph, text or audio out — can be exercised without downloading GPT-2. That
makes them worth testing on their own: if the toy stops being GPT-2 shaped, it
stops being evidence about GPT-2.
"""

import pytest
import torch

import torchbend as tb
from torchbend.test.test_modules.modules.transformer_modules import (
    TinyAudioGen, TinyGPT,
)
from torchbend.test.test_modules.tokenizers import CharTokenizer, WordTokenizer


TOKENIZERS = [CharTokenizer, WordTokenizer]


# ── tokenizers ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", TOKENIZERS)
def test_specials_come_first(cls):
    """Special ids stay put whatever the vocabulary is, so padding is stable."""
    tok = cls()
    assert tok.pad_token_id == 0
    assert tok.eos_token_id == 1
    assert tok.unk_token_id == 2


def test_char_tokenizer_round_trips_exactly():
    tok = CharTokenizer()
    text = "the sound of water, and the light."
    assert tok.batch_decode(tok(text)["input_ids"])[0] == text


def test_word_tokenizer_round_trips_known_words():
    tok = WordTokenizer()
    text = "the sound of water, and the light."
    assert tok.batch_decode(tok(text)["input_ids"])[0] == text


def test_word_tokenizer_marks_what_it_does_not_know():
    tok = WordTokenizer()
    out = tok.batch_decode(tok("zzzz the sea")["input_ids"], skip_special_tokens=False)[0]
    assert "<unk>" in out and "sea" in out


@pytest.mark.parametrize("cls", TOKENIZERS)
def test_padding_side_moves_the_mask(cls):
    """Left padding keeps the last token last, which is what continuation needs."""
    tok = cls(padding_side="left")
    left = tok(["short", "a much longer prompt"])["attention_mask"]
    tok.padding_side = "right"
    right = tok(["short", "a much longer prompt"])["attention_mask"]
    assert left[0][0] == 0 and left[0][-1] == 1
    assert right[0][0] == 1 and right[0][-1] == 0


@pytest.mark.parametrize("cls", TOKENIZERS)
def test_empty_prompt_still_yields_a_token(cls):
    """A zero-length sequence gives the model nothing to attend to."""
    assert cls()("")["input_ids"].shape[-1] == 1


@pytest.mark.parametrize("cls", TOKENIZERS)
def test_batches_are_rectangular(cls):
    tok = cls()
    enc = tok(["one", "a considerably longer one here"])
    assert enc["input_ids"].shape == enc["attention_mask"].shape
    assert enc["input_ids"].shape[0] == 2


# ── models ────────────────────────────────────────────────────────────────────

@pytest.fixture
def prompt_batch():
    tok = CharTokenizer()
    enc = tok(["the sound of water", "a longer prompt about the sea"])
    return tok, enc


def test_tinygpt_returns_a_dict_like_gpt2(prompt_batch):
    """The dict return is the shape that used to defeat output discovery."""
    tok, enc = prompt_batch
    out = TinyGPT(vocab_size=tok.vocab_size)(**enc)
    assert isinstance(out, dict) and set(out) == {"logits"}
    B, T = enc["input_ids"].shape
    assert list(out["logits"].shape) == [B, T, tok.vocab_size]


def test_tinygpt_is_causal(prompt_batch):
    """Changing a later token must not move an earlier position's logits."""
    tok, _ = prompt_batch
    torch.manual_seed(0)
    model = TinyGPT(vocab_size=tok.vocab_size).eval()
    ids = torch.randint(3, tok.vocab_size, (1, 8))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        a = model(input_ids=ids, attention_mask=mask)["logits"]
        ids2 = ids.clone()
        ids2[0, -1] = (ids2[0, -1] + 1) % tok.vocab_size
        b = model(input_ids=ids2, attention_mask=mask)["logits"]
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5)
    assert not torch.allclose(a[0, -1], b[0, -1], atol=1e-5)


def test_tinygpt_ignores_padded_positions(prompt_batch):
    """A left-padded short prompt must give what the unpadded one gives."""
    tok, _ = prompt_batch
    torch.manual_seed(0)
    model = TinyGPT(vocab_size=tok.vocab_size).eval()
    ids = torch.randint(3, tok.vocab_size, (1, 5))
    mask = torch.ones_like(ids)
    pad_ids = torch.cat([torch.zeros(1, 3, dtype=torch.long), ids], dim=1)
    pad_mask = torch.cat([torch.zeros(1, 3, dtype=torch.long), mask], dim=1)
    with torch.no_grad():
        plain = model(input_ids=ids, attention_mask=mask)["logits"][0, -1]
        padded = model(input_ids=pad_ids, attention_mask=pad_mask)["logits"][0, -1]
    assert torch.allclose(plain, padded, atol=1e-4)


def test_tinyaudiogen_makes_audio(prompt_batch):
    tok, enc = prompt_batch
    model = TinyAudioGen(vocab_size=tok.vocab_size, n_frames=16)
    with torch.no_grad():
        audio = model(**enc)
    assert audio.ndim == 3 and audio.shape[1] == 1
    assert audio.shape[0] == enc["input_ids"].shape[0]
    assert float(audio.abs().max()) <= 1.0        # tanh-bounded


def test_tinyaudiogen_is_tonal_not_noise(prompt_batch):
    """Additive synthesis is audible untrained; a neural vocoder would not be."""
    tok, enc = prompt_batch
    torch.manual_seed(0)
    model = TinyAudioGen(vocab_size=tok.vocab_size, n_frames=16)
    with torch.no_grad():
        audio = model(**enc)[0, 0]
    spec = torch.fft.rfft(audio).abs()
    # a tone concentrates its energy; white noise would spread it evenly
    assert float(spec.max() / spec.mean()) > 20


def test_tinyaudiogen_prompt_changes_the_sound(prompt_batch):
    tok, _ = prompt_batch
    torch.manual_seed(0)
    model = TinyAudioGen(vocab_size=tok.vocab_size, n_frames=16).eval()
    with torch.no_grad():
        a = model(**tok("a low rain on water"))
        b = model(**tok("a bright metal bell"))
    assert not torch.allclose(a, b, atol=1e-4)


# ── tracing ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_cls,kwargs", [
    (TinyGPT, {}),
    (TinyAudioGen, {"n_frames": 16}),
])
def test_models_trace(model_cls, kwargs):
    tok = CharTokenizer()
    enc = tok(["the sound of water", "a longer prompt about the sea"])
    module = tb.BendedModule(model_cls(vocab_size=tok.vocab_size, **kwargs))
    module.trace("forward", **enc)
    nodes = list(module.graph("forward").nodes)
    assert len(nodes) > 50
    assert any(n.op == "output" for n in nodes)


def test_traced_graph_keeps_gpt2_module_paths():
    """The viewer's module scoping is exercised through these names."""
    tok = CharTokenizer()
    enc = tok(["the sound of water", "another prompt entirely"])
    module = tb.BendedModule(TinyGPT(vocab_size=tok.vocab_size))
    module.trace("forward", **enc)
    weights = set(module.weights("?.*"))
    for expected in ("transformer.wte.weight",
                     "transformer.h.0.attn.c_attn.weight",
                     "transformer.h.0.mlp.c_fc.weight",
                     "transformer.ln_f.weight",
                     "lm_head.weight"):
        assert expected in weights, expected


def test_trace_is_symbolic_in_batch_and_length():
    """One trace has to serve any prompt, or the prompt input mode is useless."""
    tok = CharTokenizer()
    module = tb.BendedModule(TinyGPT(vocab_size=tok.vocab_size))
    module.trace("forward", **tok(["aaaa", "bbbb"]))
    for text in ("a much longer prompt than the traced one", "hi"):
        out = module.forward(**tok(text))
        logits = out["logits"] if isinstance(out, dict) else out
        assert logits.shape[1] == tok(text)["input_ids"].shape[1]
