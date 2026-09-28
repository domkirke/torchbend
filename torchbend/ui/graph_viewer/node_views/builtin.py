"""Built-in view types, organised by (batched) tensor rank.

Payloads are JSON-safe and downsampled (long sequences interpolated, channels/
batches capped) so they stay light over the wire. Every ``serialize`` receives a
tensor that is already batched (ndim >= 2).
"""
import torch
import torch.nn.functional as F

from .base import ViewType, ViewOption, token_decoder, token_eos_id, token_vocab_size

# display caps
_MAX_B = 16          # batches for 1-D / image views
_MAX_B_2D = 8        # batches for heatmap / spectrogram / grid
_MAX_C = 64          # channels
_MAX_LEN = 1024      # samples along a sequence axis
_MAX_IMG = 256       # image side
_MAX_MAP = 64        # channel-grid map side


# ── helpers ──────────────────────────────────────────────────────────────────

def _norm(t):
    mn, mx = float(t.min()), float(t.max())
    rng = mx - mn
    return (t - mn) / rng if rng > 1e-8 else t - mn


def _interp_seq(t, size):
    """Interpolate the last dim of a [B, N] tensor to `size` (linear)."""
    if t.shape[-1] <= size:
        return t
    return F.interpolate(t.unsqueeze(1).float(), size=size, mode="linear",
                         align_corners=False).squeeze(1)


def _f(t):
    return t.detach().float().cpu()


# Floats one line-family payload may carry (~2.5 MB of JSON). A [8, 128, 44100]
# activation would otherwise ship 10 MB per card — more time in json.dumps and
# JSON.parse than the forward pass itself, paid again on every refresh. A pin
# card is ~340 px wide, so even the floor below is oversampled for display.
_MAX_POINTS = 120_000


def _fit_budget(b, c, n, floor=256, budget=_MAX_POINTS):
    """Return (batches, length) that keep b*c*length within *budget*.

    Length goes first — a card is a few hundred pixels wide, so a sequence is
    already oversampled at 1024 points — then the number of batches. Channels
    are never dropped here: they are the content (``_MAX_C`` already caps them).
    """
    n = min(n, _MAX_LEN)
    if b * c * n <= budget:
        return b, n
    n = max(floor, budget // max(1, b * c))
    if b * c * n <= budget:
        return b, min(n, _MAX_LEN)
    return max(1, budget // max(1, c * n)), min(n, _MAX_LEN)


# ── rank 2 : B x N ───────────────────────────────────────────────────────────

class LineView(ViewType):
    name = "line"; label = "line plot"; priority = 60; ranks = (2,)

    def accepts(self, shape, dtype=None):
        return len(shape) == 2 and shape[0] * shape[1] > 1

    def serialize(self, t, opts, ctx):
        t = _interp_seq(_f(t)[:_MAX_B], _MAX_LEN)
        return {"view": "line", "shape": list(t.shape), "lines": t.tolist(),
                "n_batches": int(t.shape[0]), "is_audio_compatible": False}


class ScatterView(ViewType):
    name = "scatter"; label = "scatter plot"; priority = 25; ranks = (2,)

    def serialize(self, t, opts, ctx):
        t = _interp_seq(_f(t)[:_MAX_B], _MAX_LEN)
        return {"view": "scatter", "shape": list(t.shape), "lines": t.tolist(),
                "n_batches": int(t.shape[0])}


class BarView(ViewType):
    name = "bar"; label = "bar plot"; priority = 20; ranks = (2,)

    def accepts(self, shape, dtype=None):
        return len(shape) == 2 and shape[1] <= 256

    def serialize(self, t, opts, ctx):
        t = _f(t)[:_MAX_B]
        return {"view": "bar", "shape": list(t.shape), "lines": t.tolist(),
                "n_batches": int(t.shape[0])}


class CategoryView(ViewType):
    name = "category"; label = "category / one-hot"; priority = 30; ranks = (2,)
    options = [
        ViewOption("softmax", "bool", False, label="softmax",
                   description="Apply softmax over the class axis before display."),
        ViewOption("names", "str_list", None, label="class names",
                   description="Optional class labels (comma-separated)."),
    ]

    def accepts(self, shape, dtype=None):
        return len(shape) == 2 and shape[1] >= 2

    def name_hint(self, name, shape, dtype=None):
        n = (name or "").lower()
        return any(k in n for k in ("logit", "prob", "class", "categ", "softmax", "onehot"))

    def serialize(self, t, opts, ctx):
        t = _f(t)[:_MAX_B]
        probs = torch.softmax(t, dim=-1) if opts.get("softmax") else t
        argmax = probs.argmax(dim=-1)
        return {"view": "category", "shape": list(t.shape),
                "probs": probs.tolist(), "argmax": argmax.tolist(),
                "names": opts.get("names") or None,
                "n_classes": int(t.shape[1]), "n_batches": int(t.shape[0])}


# ── rank 3 : B x C x N ───────────────────────────────────────────────────────

class ChannelLinesView(ViewType):
    name = "channel_lines"; label = "channel line plots"; priority = 60; ranks = (3,)

    def serialize(self, t, opts, ctx):
        full = list(t.shape)
        t = _f(t)[:_MAX_B_2D, :_MAX_C]
        B, C, N = t.shape
        keep_b, size = _fit_budget(int(B), int(C), int(N))
        if keep_b < B:
            t = t[:keep_b]
        if size < N:
            t = F.interpolate(t, size=size, mode="linear", align_corners=False)
        return {"view": "channel_lines", "shape": [int(s) for s in full],
                "batches": t.tolist(),
                "n_batches": int(t.shape[0]), "n_channels": int(C),
                "points": int(t.shape[-1]),
                "is_audio_compatible": N > 1024}


class AudioView(ViewType):
    """Audio view carrying both a waveform and a spectrogram; the client toggles
    between the two display modes without re-fetching."""
    name = "audio"; label = "audio (waveform / spectrogram)"; priority = 50; ranks = (3,)
    options = [
        ViewOption("sample_rate", "int", 44100, label="sample rate",
                   range=[1, 192000], description="Playback / spectrogram sample rate."),
        ViewOption("channel", "int", -1, label="channel",
                   description="Channel to play (-1 = all / mixdown)."),
        ViewOption("n_fft", "int", 1024, label="n_fft", range=[16, 8192],
                   description="Spectrogram FFT size."),
        ViewOption("hop", "int", 256, label="hop", range=[1, 4096],
                   description="Spectrogram hop length."),
    ]

    def accepts(self, shape, dtype=None):
        return len(shape) == 3 and shape[-1] >= 256

    def name_hint(self, name, shape, dtype=None):
        n = (name or "").lower()
        return any(k in n for k in ("audio", "wav", "waveform", "sound", "synth", "spec"))

    def serialize(self, t, opts, ctx):
        t = _f(t)[:_MAX_B_2D, :_MAX_C]
        B, C, N = t.shape
        # The payload carries a waveform *and* a spectrogram per batch; give each
        # half the budget. One 256² spectrogram is already 65k floats, so rather
        # than dropping batches (which would take batch navigation away) the
        # spectrograms get coarser as the batch count grows — still well above
        # what a card can show.
        half = _MAX_POINTS // 2
        keep_b, size = _fit_budget(int(B), int(C), int(N), budget=half)
        n_fft = max(16, min(int(opts.get("n_fft") or 1024), int(N)))
        hop = int(opts.get("hop") or 256)
        spec_side = int(max(96, min(_MAX_IMG, (half / max(1, keep_b)) ** 0.5)))

        t = t[:keep_b]
        B = int(t.shape[0])
        # waveform (downsampled for transport, within the payload budget)
        disp = t
        if size < N:
            disp = F.interpolate(disp, size=size, mode="linear", align_corners=False)
        # spectrogram per batch (channel mixdown → mono)
        mono = t.mean(dim=1)
        win = torch.hann_window(n_fft)
        specs = []
        for b in range(B):
            stft = torch.stft(mono[b], n_fft=n_fft, hop_length=hop, window=win,
                              return_complex=True, center=True)
            db = (20.0 * torch.log10(stft.abs() + 1e-6)).flipud()   # freq low→high
            if db.shape[0] > spec_side or db.shape[1] > spec_side:
                db = F.interpolate(db.unsqueeze(0).unsqueeze(0),
                                   size=(min(db.shape[0], spec_side), min(db.shape[1], spec_side)),
                                   mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
            specs.append(_norm(db).tolist())
        return {"view": "audio", "shape": [int(B), int(C), int(N)],
                "waveform": disp.tolist(), "spectrogram": specs,
                "sample_rate": int(opts.get("sample_rate") or 44100),
                "channel": int(opts.get("channel", -1)),
                "n_fft": n_fft, "hop": hop,
                "n_batches": int(B), "n_channels": int(C), "length": int(N),
                "is_audio_compatible": True}


class Heatmap3dView(ViewType):
    name = "heatmap"; label = "heatmap (C×N)"; priority = 30; ranks = (3,)

    def serialize(self, t, opts, ctx):
        t = _f(t)[:_MAX_B_2D]
        B, C, N = t.shape
        out = t
        if C > _MAX_IMG or N > _MAX_IMG:
            out = F.interpolate(t.unsqueeze(1), size=(min(C, _MAX_IMG), min(N, _MAX_IMG)),
                                mode="bilinear", align_corners=False).squeeze(1)
        batches = [_norm(out[b]).tolist() for b in range(out.shape[0])]
        return {"view": "heatmap", "shape": [int(B), int(C), int(N)],
                "batches": batches, "n_batches": int(B)}


def _find_eos(ids, eos_id):
    """Where each row hits the tokenizer's end-of-sequence marker, if at all."""
    blank = {"at": [None] * int(ids.shape[0]), "id": eos_id}
    if eos_id is None:
        return blank
    at = []
    for row in ids.tolist():
        at.append(next((j for j, i in enumerate(row) if int(i) == int(eos_id)), None))
    return {"at": at, "id": eos_id}


def _separators(decode, ids, pieces):
    """What goes *before* each position when the pieces are reassembled.

    Some tokenizers carry the separator inside the piece (GPT-2's BPE decodes
    " the", leading space and all) and some do not — and for the ones that do
    not, whether a space belongs there depends on the neighbours: a word takes
    one, a comma does not. So this is asked per adjacent pair rather than once
    for the sequence, which is what a single sampled pair got wrong whenever it
    happened to land on punctuation.

    One batched decode of every adjacent pair, not one call per position.
    """
    # only as far as the positions that were actually sent: the alternatives are
    # capped well below the sequence length, and walking past them would index
    # off the end of `pieces`
    row = [int(i) for i in ids[0].tolist()][:len(pieces)]
    seps = [""] * len(row)
    if decode is None or len(row) < 2:
        return seps
    try:
        pairs = decode(torch.tensor([[row[j - 1], row[j]] for j in range(1, len(row))],
                                    dtype=torch.long), skip_special_tokens=False)
    except Exception:
        return seps
    for j, pair in enumerate(pairs, start=1):
        left, right = pieces[j - 1], pieces[j]
        if pair == left + right:
            seps[j] = ""
        elif pair.startswith(left) and pair.endswith(right):
            seps[j] = pair[len(left):len(pair) - len(right)]
        else:
            seps[j] = ""
    return seps


#: Positions carrying alternatives in one payload. Each costs `topk` ids, pieces
#: and probabilities; a whole context window of them would dwarf the tensor.
_MAX_INTERACTIVE_POS = 128


def _decode_pieces(ids, ctx):
    """``{id: piece}`` for the ids present, or None when nothing can decode them.

    Decoding is per distinct id rather than per position: a long sequence
    repeats its ids heavily, and a tokenizer call each time is the difference
    between instant and noticeable.
    """
    decode = ctx.get("decode") or token_decoder()
    if decode is None:
        return None
    uniq = sorted({int(i) for row in ids.tolist() for i in row})
    if not uniq:
        return None
    try:
        # one row per id, so each decodes independently of its neighbours
        pieces = decode(torch.tensor(uniq, dtype=torch.long).unsqueeze(1),
                        skip_special_tokens=False)
    except Exception:
        return None
    return {str(i): str(p) for i, p in zip(uniq, pieces)}


class TextView(ViewType):
    """Token ids as the text they stand for.

    Accepts both sides of the round trip: a ``[B, T]`` tensor of ids, and a
    ``[B, T, V]`` tensor of logits, which is argmaxed first — that second case
    is how you read what a language model just generated instead of squinting
    at a vocabulary-wide grid of floats.

    Offered only when the interface declares a token decoder; without one there
    is nothing to invert the ids with.
    """

    name = "text"; label = "decoded text"; priority = 60; ranks = (2, 3)
    options = [
        ViewOption("skip_special", "bool", True, label="skip special tokens",
                   description="Hide padding and end-of-sequence markers."),
        ViewOption("max_tokens", "int", 512, label="max tokens", range=[16, 4096],
                   description="How many positions to decode per row."),
        ViewOption("topk", "int", 5, label="alternatives", range=[0, 10],
                   description="How many runner-up tokens to carry per position, "
                               "for the expanded view. 0 sends none."),
        ViewOption("continue_steps", "int", 16, label="continue for", range=[1, 64],
                   description="How many tokens to predict after you change one. "
                               "Stops earlier if the model produces its end token."),
    ]

    @staticmethod
    def _looks_like_vocab(width):
        """Is `width` a plausible vocabulary size for the active decoder?

        Without this a `[B, N, C]` tensor is "logits" purely because C is wide
        enough — which any vision transformer's patch embeddings satisfy too
        (768-wide is completely ordinary). Argmaxing those and decoding the
        result as tokens doesn't error, it just produces near-random ids —
        mostly a tokenizer's reserved `[unusedN]` slots, since those sit at the
        low end of the vocabulary and an out-of-range argmax lands there
        disproportionately often. When the real vocab size is known, requiring
        a close match rules that out; some slack covers the few extra rows a
        model's output head is often padded to.
        """
        vocab = token_vocab_size()
        if vocab:
            return abs(width - vocab) <= max(64, int(vocab * 0.01))
        return width >= 8

    def accepts(self, shape, dtype=None):
        if token_decoder() is None:
            return False
        if len(shape) == 2:
            # a row of ids; floats here are some other quantity entirely
            return dtype is not None and not dtype.is_floating_point
        if len(shape) == 3:
            return self._looks_like_vocab(shape[-1])
        return False

    def name_hint(self, name, shape, dtype=None):
        n = (name or "").lower()
        if any(k in n for k in ("input_ids", "token", "logit", "text", "prompt")):
            return True
        # the graph's own output, when it is wide enough to be a vocabulary —
        # that is the tensor someone means by "the generated text"
        return "output" in n and len(shape) == 3 and self._looks_like_vocab(shape[-1])

    def serialize(self, t, opts, ctx):
        decode = ctx.get("decode") or token_decoder()
        if decode is None:
            return {"view": "text", "shape": [int(s) for s in t.shape],
                    "texts": [], "error": "this model declares no token decoder"}

        ids = t.argmax(dim=-1) if t.ndim == 3 else t
        ids = ids[:_MAX_B_2D]
        limit = max(16, int(opts.get("max_tokens") or 512))
        ids = ids[:, :limit].long().cpu()
        skip = bool(opts.get("skip_special", True))

        # the sequence's own end-of-sequence marker, if the tokenizer has one and
        # it occurs: a default the user can move, not a setting they must find
        end = _find_eos(ids, token_eos_id())
        try:
            texts = self._decode_rows(ids, end["at"], skip, decode)
        except Exception as exc:                      # a tokenizer is user code
            return {"view": "text", "shape": [int(s) for s in t.shape],
                    "texts": [], "error": str(exc)}

        payload = {"view": "text", "shape": [int(s) for s in t.shape],
                   "texts": [str(x) for x in texts],
                   "from_logits": bool(t.ndim == 3),
                   "n_tokens": int(ids.shape[-1]), "n_batches": int(ids.shape[0]),
                   # where each row was cut, and whether the token was there at
                   # all — a request that matched nothing must say so rather
                   # than look like a sequence that simply never ended
                   "end_at": end["at"], "end_id": end["id"],
                   "eos_id": token_eos_id(),
                   "continue_steps": int(opts.get("continue_steps") or 16)}
        if t.ndim == 3:
            positions = self._positions(t, ids, opts, decode)
            payload["positions"] = positions
            if positions:
                payload["seps"] = _separators(decode, ids,
                                              [p["piece"] for p in positions[0]])
        return payload

    @staticmethod
    def _decode_rows(ids, cuts, skip, decode):
        """Decode each row, stopping where its end token was found.

        Rows are decoded together while none of them is cut — that is the common
        case and one call is cheaper than B — and one at a time once their
        lengths diverge.
        """
        if not any(c is not None for c in cuts):
            return decode(ids, skip_special_tokens=skip)
        out = []
        for b, cut in enumerate(cuts):
            row = ids[b, :cut] if cut is not None else ids[b]
            if row.numel() == 0:
                out.append("")
                continue
            out.append(decode(row.unsqueeze(0), skip_special_tokens=skip)[0])
        return out

    def _positions(self, logits, ids, opts, decode):
        """Per position: what was chosen, how sure, and what nearly won.

        Only for the expanded view — a sequence's worth of alternatives is far
        more than a sidebar card can show, so it is capped hard here rather than
        sent in full and thrown away by the client.
        """
        topk = int(opts.get("topk") if opts.get("topk") is not None else 5)
        if topk <= 0:
            return None
        topk = min(topk, 10, int(logits.shape[-1]))
        n_pos = min(int(ids.shape[-1]), _MAX_INTERACTIVE_POS)

        # detach: an activation captured under grad would otherwise warn on every
        # single probability read out of it
        probs = torch.softmax(logits[:, :n_pos].detach().float(), dim=-1)
        top_p, top_i = probs.topk(topk, dim=-1)

        pieces = _decode_pieces(top_i.reshape(top_i.shape[0], -1), {"decode": decode})
        if pieces is None:
            return None

        out = []
        for b in range(top_i.shape[0]):
            row = []
            for j in range(n_pos):
                alts = [{"id": int(top_i[b, j, k]),
                         "piece": pieces.get(str(int(top_i[b, j, k])), ""),
                         "prob": round(float(top_p[b, j, k]), 5)}
                        for k in range(topk)]
                row.append({"id": alts[0]["id"], "piece": alts[0]["piece"],
                            "prob": alts[0]["prob"], "alts": alts})
            out.append(row)
        return out


class TokensView(ViewType):
    name = "tokens"; label = "token sequence (B×T×C)"; priority = 20; ranks = (3,)
    options = [
        ViewOption("topk", "int", 1, label="top-k", range=[1, 10],
                   description="Show the top-k token ids per position."),
        ViewOption("names", "str_list", None, label="vocabulary",
                   description="Optional vocabulary labels (comma-separated)."),
    ]

    def name_hint(self, name, shape, dtype=None):
        n = (name or "").lower()
        return any(k in n for k in ("token", "logit", "vocab", "seq"))

    def serialize(self, t, opts, ctx):
        # interpret as [B, T, C]; argmax over the last (vocab) axis
        t = _f(t)[:_MAX_B_2D]
        B, T, C = t.shape
        T_show = min(T, 512)
        tt = t[:, :T_show, :]
        topk = max(1, min(int(opts.get("topk") or 1), 10, C))
        vals, idx = tt.topk(topk, dim=-1)
        ids = idx[..., 0]
        return {"view": "tokens", "shape": [int(B), int(T), int(C)],
                "ids": ids.tolist(),                         # [B, T_show] argmax ids
                "topk_ids": idx.tolist() if topk > 1 else None,
                # a hand-pasted vocabulary still wins; otherwise ask the model
                "names": opts.get("names") or None,
                "pieces": None if opts.get("names") else _decode_pieces(ids, ctx),
                "n_tokens": int(T_show), "vocab": int(C), "n_batches": int(B)}


# ── rank 4 : B x C x H x W ───────────────────────────────────────────────────

class ImageView(ViewType):
    name = "image"; label = "image"; priority = 60; ranks = (4,)
    options = [
        ViewOption("channels", "choice", "auto", label="channels",
                   choices=["auto", "gray", "rgb", "rgba"],
                   description="How to interpret the channel axis."),
    ]

    def accepts(self, shape, dtype=None):
        return len(shape) == 4 and shape[1] in (1, 3, 4)

    def name_hint(self, name, shape, dtype=None):
        n = (name or "").lower()
        return any(k in n for k in ("image", "img", "rgb", "frame", "pixel"))

    def serialize(self, t, opts, ctx):
        t = _f(t)[:_MAX_B]
        B, C, H, W = t.shape
        if H > _MAX_IMG or W > _MAX_IMG:
            t = F.interpolate(t, size=(min(H, _MAX_IMG), min(W, _MAX_IMG)),
                              mode="bilinear", align_corners=False)
        mn, mx = float(t.min()), float(t.max())
        rng = mx - mn
        t = (t - mn) / rng if rng > 1e-8 else torch.zeros_like(t)
        ch = opts.get("channels", "auto")
        image_type = ch if ch in ("gray", "rgb", "rgba") else {1: "gray", 3: "rgb", 4: "rgba"}[C]
        return {"view": "image", "shape": [int(B), int(C), int(H), int(W)],
                "images": t.tolist(), "image_type": image_type, "n_batches": int(B)}


class ChannelGridView(ViewType):
    name = "channel_grid"; label = "channel grid"; priority = 40; ranks = (4,)

    def serialize(self, t, opts, ctx):
        t = _f(t)[:_MAX_B_2D, :_MAX_C]
        B, C, H, W = t.shape
        if H > _MAX_MAP or W > _MAX_MAP:
            flat = F.interpolate(t.reshape(B * C, 1, H, W),
                                 size=(min(H, _MAX_MAP), min(W, _MAX_MAP)),
                                 mode="bilinear", align_corners=False)
            t = flat.reshape(B, C, min(H, _MAX_MAP), min(W, _MAX_MAP))
        batches = [[_norm(t[b, c]).tolist() for c in range(t.shape[1])] for b in range(t.shape[0])]
        return {"view": "channel_grid", "shape": [int(B), int(C), int(H), int(W)],
                "batches": batches, "n_batches": int(B), "n_channels": int(C)}


# ── scalar fallback ─────────────────────────────────────────────────────────

class ScalarView(ViewType):
    name = "scalar"; label = "scalar"; priority = 100; ranks = (2, 3, 4)

    def accepts(self, shape, dtype=None):
        n = 1
        for s in shape:
            n *= int(s)
        return n == 1

    def serialize(self, t, opts, ctx):
        return {"view": "scalar", "shape": list(t.shape), "value": float(_f(t).reshape(-1)[0])}


ALL_VIEWS = [
    ScalarView,
    LineView, ScatterView, BarView, CategoryView,
    ChannelLinesView, AudioView, Heatmap3dView, TokensView, TextView,
    ImageView, ChannelGridView,
]
