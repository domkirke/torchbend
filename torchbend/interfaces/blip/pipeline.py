"""BLIP end to end: image in, caption out, with the text decoder in the graph too.

:class:`~torchbend.interfaces.blip.BendedBlip` traces only the vision tower --
the text decoder samples one token at a time and stops when it decides to,
and no graph holds a loop whose length is decided while it runs. That is the
same limitation Bark's semantic and coarse transformers hit (see
``torchbend.interfaces.bark.pipeline``), and the same fix applies: a decoder
that instead runs a *fixed* number of steps, unconditionally, unrolls into
the graph just fine -- it is a fixed sequence of operations again, just a
longer one. That is what :class:`BlipCaptionPipeline` does: greedy, always
``max_new_tokens`` steps, no early stop while tracing.

The payoff is the same as Bark's pipeline:

    **the graph is literally the last step of generation.**

``caption()`` reads the bended graph's own output -- there is no separate
``generate()`` call and no vision-shim swap. Bend a node, anywhere in the
vision tower *or* the decoder, and the caption changes because it was
computed by the bending, not beside it.

The cost is the one described above: no beam search (only greedy unrolls
cleanly) and a fixed length regardless of when ``[SEP]`` actually shows up
(trimmed for display, not for tracing -- shortening the graph on the fly is
exactly the "loop whose length is decided while it runs" problem this exists
to avoid). Reach for :class:`~torchbend.interfaces.blip.BendedBlip` instead
when you want beam search or a graph limited to the vision tower alone.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .interface import BendedBlip, BendingBlipException, _DEFAULT_MODEL
from ..spec import Callback, Ref, Input, Int, Method, Option, Tensor, Text, Tokens
from ..base import Interface
from ...tracing import ScriptableState
from ...tracing.loop import loop as tb_loop


__all__ = ["BlipCaptionPipeline", "BendedBlipPipeline", "BendingBlipException"]


class BlipCaptionPipeline(nn.Module):
    """BLIP's vision tower and text decoder chained into one module: pixel
    values in, caption token ids out, greedy, always ``max_new_tokens`` steps.

    This is what gets traced -- see the module docstring for why the decoder
    has to run a fixed length to be traceable at all.

    The decode loop runs through :func:`torchbend.loop`, so how much of it the
    graph actually holds is a trace-time choice (``_loop_policy``) rather than
    a fact about this code: unrolled it is ~18.5k nodes, packed it is a few
    dozen. That is why the caption buffer is **preallocated to its full length
    and written into** rather than grown by ``torch.cat`` -- a packed loop
    needs a carry whose shape survives an iteration unchanged.

    Reading ``logits[:, i]`` off the full-length buffer is exactly equivalent
    to reading ``logits[:, -1]`` off a sequence truncated at ``i``: the
    decoder is causal, so position ``i`` cannot attend to the not-yet-written
    positions after it. Verified token-for-token against the grown-ids
    formulation in ``test/test_interface/test_blip_pipeline.py``.
    """

    def __init__(self, full_model, max_new_tokens: int = 20):
        super().__init__()
        self.vision_model = full_model.vision_model
        self.text_decoder = full_model.text_decoder
        self.max_new_tokens = int(max_new_tokens)
        self.bos_token_id = int(full_model.config.text_config.bos_token_id)
        self.pad_token_id = int(getattr(full_model.config.text_config, "pad_token_id", 0) or 0)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vis_out = self.vision_model(pixel_values=pixel_values)
        vis_feats = vis_out.last_hidden_state if hasattr(vis_out, "last_hidden_state") else vis_out[0]
        vis_mask = torch.ones(vis_feats.shape[:-1], dtype=torch.long, device=vis_feats.device)

        device = pixel_values.device
        batch = pixel_values.shape[0]
        length = self.max_new_tokens + 1
        ids = torch.full((batch, length), self.pad_token_id, dtype=torch.long, device=device)
        ids = ids.index_copy(
            1, torch.zeros(1, dtype=torch.long, device=device),
            torch.full((batch, 1), self.bos_token_id, dtype=torch.long, device=device))

        # The vision features travel *through* the loop rather than being
        # captured by `step`: a packed body is replayed from the graph, so a
        # captured tensor would be frozen at its trace-time value. Carrying
        # them also makes them bendable between decode steps.
        def step(i, carry):
            ids, feats, mask = carry
            out = self.text_decoder(input_ids=ids, encoder_hidden_states=feats,
                                     encoder_attention_mask=mask, use_cache=False)
            # logits at position i predict position i+1
            next_id = out.logits[:, i, :].argmax(dim=-1, keepdim=True)
            index = torch.full((1,), i + 1, dtype=torch.long, device=ids.device)
            return (ids.index_copy(1, index, next_id), feats, mask)

        ids, _, _ = tb_loop(step, (ids, vis_feats, vis_mask), self.max_new_tokens,
                            name="blip_decode", modules=[self.text_decoder])
        return ids


class BendedBlipPipeline(Interface):
    """Image captioning with the *whole* path -- vision tower and text decoder
    both -- as the bendable graph. See the module docstring for the tradeoff
    against :class:`~torchbend.interfaces.blip.BendedBlip`."""

    _imported_callbacks_ = []
    _panel_render_type_ = "text"
    tokens = Tokens(decode="decode_caption_tokens", logits="caption_logits",
                    eos=Ref("eos_token_id"))

    options = {
        "max_new_tokens": Option(
            Int(range=(4, 64)), attr="max_new_tokens", needs="retrace", label="max tokens",
            doc="Fixed number of decoder steps the graph holds. Changing this "
                "reshapes the graph, so it needs a retrace."),
        "pack": Option(
            Int(range=(0, 64)), attr="pack", needs="retrace", label="decode steps per node",
            doc="How many decode steps each graph node covers. 0 unrolls the "
                "decoder completely — every activation inside it becomes "
                "individually bendable, at ~18.5k nodes. 1 gives one node per "
                "step, keeping the caption-so-far bendable between steps. "
                "Higher packs further. Weight bending works at any setting."),
    }

    callbacks = {
        "caption": Callback(
            returns=Text(), label="caption (image → text)",
            doc="Decode the bended graph's own output. Bend a node -- vision "
                "tower or text decoder -- and run this again to hear what "
                "changed; there is no separate generation step to route "
                "around, the graph's output *is* the caption."),
        "caption_tokens": Callback(
            returns=Tensor(), label="caption tokens (editable)",
            doc="Same caption as `caption`, as the raw token ids so the viewer "
                "can show it as editable text with next-token probabilities "
                "(from the unbent model — see `caption_logits`)."),
    }

    methods = {"forward": Method(inputs={"pixel_values": Input(
        lambda self: "torch.rand(1, 3, %d, %d)" % (self._image_size, self._image_size))})}

    def __init__(self, model_path: str = _DEFAULT_MODEL, max_new_tokens: int = 20,
                 pack: int = 1, device=torch.device("cpu"), model=None,
                 processor=None, **kwargs):
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.pack = int(pack)
        if model is None or processor is None:
            model, processor = BendedBlip.load_model(model_path, device=device, **kwargs)
        self._full = model
        self._processor = processor
        self._pixels = None
        size = int(model.config.vision_config.image_size)
        self._image_size = size
        pipeline = BlipCaptionPipeline(model, max_new_tokens=self.max_new_tokens)
        super().__init__(pipeline)

    # -- tracing --

    def bend_model(self, model):
        model.trace("forward", pixel_values=self._blank_image(),
                    _loop_policy=self._loop_policy())

    def _loop_policy(self) -> dict:
        """`pack=0` means "put the whole decoder in the graph"; anything else
        packs that many decode steps per node."""
        if self.pack <= 0:
            return {"mode": "unroll"}
        return {"mode": "pack", "pack": self.pack}

    def _blank_image(self):
        return torch.rand(1, 3, self._image_size, self._image_size, device=self.device)

    # -- what the bench is running on --

    def on_inputs(self, fn, kwargs):
        pixels = kwargs.get("pixel_values")
        if torch.is_tensor(pixels):
            self._pixels = pixels.detach()

    # -- properties --

    @property
    def config(self):
        return self._full.config

    @property
    def processor(self):
        return self._processor

    @property
    def tokenizer(self):
        return getattr(self._processor, "tokenizer", None)

    @property
    def eos_token_id(self):
        """BLIP's tokenizer is BERT-based and has no `eos_token` — captions end
        on `[SEP]` instead. See BendedBlip.eos_token_id for why this matters."""
        tok = self.tokenizer
        if tok is None:
            return None
        eos = getattr(tok, "eos_token_id", None)
        return eos if eos is not None else getattr(tok, "sep_token_id", None)

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def pixel_values(self) -> Optional[torch.Tensor]:
        return self._pixels

    def encode_image(self, image) -> torch.Tensor:
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image).convert("RGB")
        return self._processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)

    # -- inference --

    def _current_pixels(self):
        if self._pixels is None:
            raise BendingBlipException(
                "no image yet — drop one on `pixel_values` in the input bench; "
                "the caption describes whatever the graph last ran on")
        return self._pixels

    def caption_tokens(self, pixel_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Caption ids `[1, T]`, computed by the bended graph itself."""
        pixels = self._current_pixels() if pixel_values is None else pixel_values
        with torch.no_grad():
            return self._model.forward(pixel_values=pixels)

    def caption(self, pixel_values: Optional[torch.Tensor] = None) -> List[str]:
        """Describe the image, reading the caption straight off the bended graph."""
        ids = self.caption_tokens(pixel_values)
        return self._processor.batch_decode(ids, skip_special_tokens=True)

    def decode_caption_tokens(self, ids: torch.Tensor, skip_special_tokens: bool = True):
        return self._processor.batch_decode(ids, skip_special_tokens=skip_special_tokens)

    def caption_logits(self, ids: torch.Tensor) -> torch.Tensor:
        """Logits `[B, T, V]` for a candidate caption, from the *unbent* model --
        there is no separate teacher-forcing entry point once decoding is inside
        the bended graph, so this previews continuations independently of
        whatever bending is currently applied, the same as BendedBlip's."""
        pixels = self._current_pixels()
        with torch.no_grad():
            out = self._full(pixel_values=pixels, input_ids=ids)
        return out.logits if hasattr(out, "logits") else out[0]

    # -- export --

    @property
    def scriptable(self):
        return ScriptableState.NotScriptable

    @property
    def nntilde_compatible(self):
        return ScriptableState.NotScriptable
