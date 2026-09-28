"""Reading tokens back as text.

An input mode turns text into ids; this is the inverse, and it is what makes a
language model legible in the viewer — `input_ids` shown as the prompt, and a
vocabulary-wide tensor shown as the words it argmaxes to, rather than either
appearing as a grid of numbers.
"""

import pytest
import torch

from torchbend.test.test_modules.interfaces import BendedTinyGPT
from torchbend.ui.graph_viewer.node_views import serialize_node


@pytest.fixture(scope="module")
def iface():
    return BendedTinyGPT(tokenizer="word")


def views_of(payload):
    return [c["name"] for c in payload["_view_meta"]["compatible"]]


def test_interface_declares_the_inverse_of_its_encoder(iface):
    assert (iface.spec.tokens is not None)
    ids = iface.spec.method("forward").encode("input_ids", "the sound of water")["input_ids"]
    assert iface.spec.tokens.decode(ids) == ["the sound of water"]


def test_ids_render_as_the_prompt(iface):
    ids = iface.spec.method("forward").encode("input_ids", "the sound of water")["input_ids"]
    payload = serialize_node(ids, node="input_ids", decode=iface.spec.tokens.decode)
    assert payload["view"] == "text"
    assert payload["texts"] == ["the sound of water"]
    assert payload["from_logits"] is False


def test_a_batch_decodes_row_by_row(iface):
    ids = iface.spec.method("forward").encode("input_ids", ["the sound of water", "a bright light"])["input_ids"]
    payload = serialize_node(ids, node="input_ids", decode=iface.spec.tokens.decode)
    assert len(payload["texts"]) == 2
    assert payload["texts"][0].endswith("water")


def test_logits_are_argmaxed_before_decoding(iface):
    """This is the case that answers 'what did the model just say'."""
    logits = torch.randn(2, 5, iface.vocab_size)
    payload = serialize_node(logits, node="logits", decode=iface.spec.tokens.decode)
    assert payload["view"] == "text"
    assert payload["from_logits"] is True
    assert len(payload["texts"]) == 2
    assert all(isinstance(t, str) and t for t in payload["texts"])


def test_the_view_is_not_offered_without_a_decoder(iface):
    """A model that cannot invert its tokens has nothing to show here."""
    ids = iface.spec.method("forward").encode("input_ids", "the sound of water")["input_ids"]
    payload = serialize_node(ids, node="input_ids", decode=None)
    assert payload["view"] != "text"
    assert "text" not in views_of(payload)


def test_the_view_is_offered_with_one(iface):
    ids = iface.spec.method("forward").encode("input_ids", "the sound of water")["input_ids"]
    payload = serialize_node(ids, node="input_ids", decode=iface.spec.tokens.decode)
    assert "text" in views_of(payload)


def test_float_rank_two_is_not_mistaken_for_ids(iface):
    """A [B, N] float tensor is some other quantity, not a token sequence."""
    payload = serialize_node(torch.randn(2, 16), node="input_ids",
                             decode=iface.spec.tokens.decode)
    assert payload["view"] != "text"
    assert "text" not in views_of(payload)


def test_the_decoder_does_not_leak_between_calls(iface):
    """It is published per serialization; the next call must not inherit it."""
    ids = iface.spec.method("forward").encode("input_ids", "the sound of water")["input_ids"]
    serialize_node(ids, node="input_ids", decode=iface.spec.tokens.decode)
    after = serialize_node(ids, node="input_ids", decode=None)
    assert "text" not in views_of(after)


def test_max_tokens_bounds_what_is_decoded(iface):
    ids = torch.randint(3, iface.vocab_size, (1, 300))
    payload = serialize_node(ids, node="input_ids", decode=iface.spec.tokens.decode,
                             session_sel={"view": "text", "options": {"max_tokens": 16}})
    assert payload["n_tokens"] == 16


def test_a_failing_tokenizer_is_reported_not_raised(iface):
    """A tokenizer is user code; a view must not take the request down with it."""
    def broken(ids, skip_special_tokens=True):
        raise RuntimeError("no vocabulary loaded")

    ids = iface.spec.method("forward").encode("input_ids", "the sound of water")["input_ids"]
    payload = serialize_node(ids, node="input_ids", decode=broken)
    assert payload["view"] == "text"
    assert "no vocabulary loaded" in payload["error"]


def test_tokens_view_labels_chips_with_decoded_pieces(iface):
    logits = torch.randn(1, 6, iface.vocab_size)
    payload = serialize_node(logits, node="logits", decode=iface.spec.tokens.decode,
                             session_sel={"view": "tokens", "options": {}})
    assert payload["view"] == "tokens"
    pieces = payload["pieces"]
    assert pieces, "ids should carry their decoded pieces"
    for row in payload["ids"]:
        for i in row:
            assert str(i) in pieces


def test_a_pasted_vocabulary_still_wins_over_the_decoder(iface):
    """An explicit `names` list is the user overriding the model."""
    logits = torch.randn(1, 4, iface.vocab_size)
    payload = serialize_node(
        logits, node="logits", decode=iface.spec.tokens.decode,
        session_sel={"view": "tokens", "options": {"names": ["a", "b", "c"]}})
    assert payload["names"] == ["a", "b", "c"]
    assert payload["pieces"] is None


# ── per-position probabilities (the expanded explorer) ────────────────────────

def positions_of(iface, logits, **options):
    sel = {"view": "text", "options": options} if options else None
    return serialize_node(logits, node="logits", decode=iface.spec.tokens.decode,
                          session_sel=sel)


def test_logits_carry_per_position_alternatives(iface):
    payload = positions_of(iface, torch.randn(1, 6, iface.vocab_size))
    rows = payload["positions"]
    assert len(rows) == 1 and len(rows[0]) == 6
    for pos in rows[0]:
        assert pos["piece"] == pos["alts"][0]["piece"]      # the shown token leads
        assert pos["alts"] == sorted(pos["alts"], key=lambda a: -a["prob"])
        assert 0.0 <= pos["prob"] <= 1.0


def test_alternatives_are_probabilities_not_logits(iface):
    """A distribution: each within [0,1], and the top-k a prefix of one."""
    payload = positions_of(iface, torch.randn(1, 4, iface.vocab_size), topk=10)
    for pos in payload["positions"][0]:
        assert all(0.0 <= a["prob"] <= 1.0 for a in pos["alts"])
        assert sum(a["prob"] for a in pos["alts"]) <= 1.0 + 1e-6


def test_topk_controls_how_many_are_sent(iface):
    payload = positions_of(iface, torch.randn(1, 3, iface.vocab_size), topk=3)
    assert all(len(p["alts"]) == 3 for p in payload["positions"][0])


def test_topk_zero_sends_none(iface):
    """The alternatives are the expensive half of this payload; opt out cleanly."""
    payload = positions_of(iface, torch.randn(1, 3, iface.vocab_size), topk=0)
    assert payload["positions"] is None
    assert payload["texts"]                       # the text itself still arrives


def test_ids_carry_no_alternatives(iface):
    """Ids are what was chosen; there is no distribution behind them."""
    ids = iface.spec.method("forward").encode("input_ids", "the sound of water")["input_ids"]
    payload = serialize_node(ids, node="input_ids", decode=iface.spec.tokens.decode)
    assert "positions" not in payload


def test_long_sequences_are_capped(iface):
    """A whole context window of alternatives would dwarf the tensor itself."""
    from torchbend.ui.graph_viewer.node_views.builtin import _MAX_INTERACTIVE_POS
    payload = positions_of(iface, torch.randn(1, _MAX_INTERACTIVE_POS + 50,
                                              iface.vocab_size))
    assert len(payload["positions"][0]) == _MAX_INTERACTIVE_POS


def test_pieces_reassemble_into_the_decoded_text(iface):
    """The join hint is what stops a reassembled sentence being onelongword.

    Compared against a decode that also keeps special tokens: pieces are always
    decoded with them, so that a position where the model nearly stopped shows
    `<eos>` rather than an empty chip. Skipping on one side only would be
    comparing two different sentences.
    """
    torch.manual_seed(0)
    payload = positions_of(iface, torch.randn(1, 6, iface.vocab_size),
                           skip_special=False)
    pieces = [p["piece"] for p in payload["positions"][0]]
    seps = payload["seps"]
    joined = "".join((seps[j] if j else "") + pieces[j] for j in range(len(pieces)))
    assert joined == payload["texts"][0]


def test_pieces_keep_special_tokens_even_when_the_text_drops_them(iface):
    """Seeing that the model nearly emitted <eos> is the point of the strip."""
    ids = torch.tensor([[3, iface.tokenizer.eos_token_id, 4]])
    one_hot = torch.zeros(1, 3, iface.vocab_size)
    for j, i in enumerate(ids[0]):
        one_hot[0, j, int(i)] = 10.0
    payload = positions_of(iface, one_hot, skip_special=True)
    pieces = [p["piece"] for p in payload["positions"][0]]
    assert iface.tokenizer.eos_token in pieces
    assert iface.tokenizer.eos_token not in payload["texts"][0]


def test_separators_are_per_position_not_per_sequence():
    """A word takes a space before it, a comma does not — in the same sequence.

    Deriving one separator for the whole sequence got this wrong whenever the
    sampled pair happened to be word+punctuation.
    """
    from torchbend.test.test_modules.interfaces import BendedTinyGPT as G
    word = G(tokenizer="word")
    ids = word.spec.method("forward").encode("input_ids", "the sound, and light")["input_ids"]
    one_hot = torch.zeros(1, ids.shape[1], word.vocab_size)
    for j, i in enumerate(ids[0]):
        one_hot[0, j, int(i)] = 10.0
    payload = positions_of(word, one_hot, skip_special=False)
    seps = payload["seps"]
    pieces = [p["piece"] for p in payload["positions"][0]]
    assert "" in seps[1:] and " " in seps[1:], (pieces, seps)
    joined = "".join((seps[j] if j else "") + pieces[j] for j in range(len(pieces)))
    assert joined == payload["texts"][0]


def test_char_pieces_need_no_separators():
    from torchbend.test.test_modules.interfaces import BendedTinyGPT as G
    char = G(tokenizer="char")
    payload = positions_of(char, torch.randn(1, 4, char.vocab_size))
    assert set(payload["seps"]) == {""}


def test_probabilities_survive_a_grad_tracking_tensor(iface):
    """Activations can arrive still attached to the graph."""
    logits = torch.randn(1, 4, iface.vocab_size, requires_grad=True)
    payload = positions_of(iface, logits)
    assert len(payload["positions"][0]) == 4


# ── the end of the sequence ───────────────────────────────────────────────────
# Not a setting to type: the end is a token in the strip, marked where the
# tokenizer's own marker falls, and any position can be made the end instead.

def decoded(iface, ids, **options):
    sel = {"view": "text", "options": options} if options else None
    return serialize_node(ids, node="input_ids", decode=iface.spec.tokens.decode,
                          eos_id=iface.spec.tokens.eos, session_sel=sel)


@pytest.fixture
def sentence(iface):
    return iface.spec.method("forward").encode("input_ids", "the sound of water and the light")["input_ids"]


def test_the_end_is_not_an_option_to_type(iface, sentence):
    """It is a token you click, so it must not also be a text field."""
    payload = decoded(iface, sentence)
    assert "end_token" not in [o["name"] for o in payload["_view_meta"]["options"]]


def test_the_tokenizer_s_own_marker_is_found(iface):
    ids = torch.tensor([[3, 127, iface.tokenizer.eos_token_id, 16, 129]])
    payload = decoded(iface, ids)
    assert payload["eos_id"] == iface.tokenizer.eos_token_id
    assert payload["end_at"] == [2]
    assert payload["texts"] == ["the sound"]


def test_a_sequence_without_the_marker_has_no_end(iface, sentence):
    payload = decoded(iface, sentence)
    assert payload["end_at"] == [None]
    assert payload["texts"] == ["the sound of water and the light"]


def test_rows_find_their_own_end(iface):
    eos = iface.tokenizer.eos_token_id
    ids = torch.tensor([[3, eos, 16, 129], [3, 127, 16, eos]])
    payload = decoded(iface, ids)
    assert payload["end_at"] == [1, 3]


def test_a_model_without_an_eos_marks_nothing(iface, sentence):
    payload = serialize_node(sentence, node="input_ids", decode=iface.spec.tokens.decode,
                             eos_id=None)
    assert payload["eos_id"] is None
    assert payload["end_at"] == [None]


# ── continuing from an edit ───────────────────────────────────────────────────
# Changing a word has to change what follows it, or the view shows a
# continuation the model never made.

def test_interface_declares_a_way_to_continue(iface):
    assert iface.spec.tokens.has_logits
    ids = iface.spec.method("forward").encode("input_ids", "the sound of")["input_ids"]
    logits = iface.spec.tokens.logits(ids)
    assert list(logits.shape) == [1, ids.shape[1], iface.vocab_size]


def greedy(iface, prefix, steps):
    """What the continuation endpoint does, without the HTTP around it."""
    ids = torch.tensor([list(prefix)], dtype=torch.long)
    out = []
    with torch.no_grad():
        for _ in range(steps):
            nxt = int(iface.spec.tokens.logits(ids)[0, -1].argmax())
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long)], dim=-1)
    return out


def test_a_different_prefix_continues_differently(iface):
    """The whole point: the tail depends on the token you changed."""
    base = iface.spec.method("forward").encode("input_ids", "the sound of")["input_ids"][0].tolist()
    a = greedy(iface, base, 5)
    swapped = base[:-1] + [base[-1] + 1]
    b = greedy(iface, swapped, 5)
    assert a != b


def test_continuing_is_deterministic(iface):
    """Greedy, so the same edit twice gives the same tail — no hidden sampling."""
    base = iface.spec.method("forward").encode("input_ids", "the sound of")["input_ids"][0].tolist()
    assert greedy(iface, base, 4) == greedy(iface, base, 4)


def test_continuing_grows_the_sequence_by_the_steps_asked(iface):
    base = iface.spec.method("forward").encode("input_ids", "the sound")["input_ids"][0].tolist()
    assert len(greedy(iface, base, 6)) == 6


def test_a_model_that_cannot_predict_tokens_does_not_claim_to():
    """The audio model shares the text one's tokenizer, not its head.

    Declaring the hook on the shared base made it claim a continuation it
    cannot produce; it belongs to the model that predicts tokens.
    """
    from torchbend.test.test_modules.interfaces import BendedTinyAudioGen
    audio = BendedTinyAudioGen(tokenizer="word")
    assert (audio.spec.tokens is not None)          # it can still read ids back
    assert not audio.spec.tokens.has_logits       # but it cannot continue them


# ── how far a continuation runs ───────────────────────────────────────────────
# It used to run for however many tokens the original sequence happened to have
# left, which made an edit near the end produce almost nothing and an edit on
# the last token produce nothing at all. It is now its own length, ended by the
# model's end token rather than by a leftover count.

class _StopsAfter:
    """A model that emits `eos` once the sequence reaches a given length."""

    def __init__(self, vocab, eos, at):
        self.vocab, self.eos, self.at = vocab, eos, at

    def token_logits(self, ids):
        nxt = self.eos if ids.shape[1] >= self.at else 7
        out = torch.zeros(1, ids.shape[1], self.vocab)
        out[0, -1, nxt] = 10.0
        return out


def _pieces(ids, skip_special_tokens=True):
    return ["<t%d>" % int(i) for i in ids[:, 0]] if ids.shape[1] == 1 else \
           [" ".join("<t%d>" % int(i) for i in row) for row in ids]


def test_a_continuation_runs_for_the_length_asked():
    from torchbend.ui.graph_viewer.views import continue_greedy
    model = _StopsAfter(vocab=16, eos=1, at=10_000)      # never stops on its own
    positions, ids, stopped = continue_greedy(model.token_logits, _pieces, [3, 4], steps=9, topk=2)
    assert len(positions) == 9
    assert stopped == "steps"
    assert len(ids) == 2 + 9


def test_a_continuation_stops_at_the_end_token():
    """The model ending the sequence beats the count."""
    from torchbend.ui.graph_viewer.views import continue_greedy
    model = _StopsAfter(vocab=16, eos=1, at=5)
    positions, ids, stopped = continue_greedy(model.token_logits, _pieces, [3, 4], steps=40, topk=2,
                                              eos=1)
    assert stopped == "eos"
    assert len(positions) < 40
    assert positions[-1]["id"] == 1                      # the marker is included


def test_without_an_eos_it_runs_to_the_count():
    """A tokenizer with no end token cannot stop early — the count is the net."""
    from torchbend.ui.graph_viewer.views import continue_greedy
    model = _StopsAfter(vocab=16, eos=1, at=5)
    positions, _ids, stopped = continue_greedy(model.token_logits, _pieces, [3, 4], steps=12, topk=2,
                                               eos=None)
    assert stopped == "steps"
    assert len(positions) == 12


def test_editing_the_last_token_still_continues():
    """It used to compute zero steps there, so nothing happened at all."""
    from torchbend.ui.graph_viewer.views import continue_greedy
    model = _StopsAfter(vocab=16, eos=1, at=10_000)
    positions, _ids, _stopped = continue_greedy(model.token_logits, _pieces, [3], steps=6, topk=2)
    assert len(positions) == 6


def test_the_continuation_length_travels_with_the_payload(iface):
    """The client needs it to know how far to ask for."""
    payload = serialize_node(torch.randn(1, 4, iface.vocab_size), node="logits",
                             decode=iface.spec.tokens.decode, eos_id=iface.spec.tokens.eos)
    assert payload["continue_steps"] == 16
    tuned = serialize_node(torch.randn(1, 4, iface.vocab_size), node="logits",
                           decode=iface.spec.tokens.decode, eos_id=iface.spec.tokens.eos,
                           session_sel={"view": "text", "options": {"continue_steps": 5}})
    assert tuned["continue_steps"] == 5


def test_an_untrained_model_simply_never_ends(iface):
    """Why the toy shows no end token: eos is essentially never its argmax.

    Not a bug in the view — a property of random weights. The end can still be
    placed by hand from the strip.
    """
    ids = iface.spec.method("forward").encode("input_ids",
                             "the sound of water and the light")["input_ids"]
    argmax = iface.spec.tokens.logits(ids)[0].argmax(-1).tolist()
    assert iface.tokenizer.eos_token_id not in argmax
