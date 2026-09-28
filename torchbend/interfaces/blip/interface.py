"""Bending interface for BLIP image captioning — a picture in, a sentence out.

BLIP is two models. A **vision transformer** turns an image into 577 patch
embeddings, and a **text decoder** reads those and writes a sentence about them,
one token at a time.

Only the first can be a graph: the decoder samples in a loop, and a loop whose
length depends on what the model just said is not a fixed sequence of
operations. So the vision tower is what this interface makes bendable — 846
nodes from pixels to patch embeddings — and captioning runs *through* it.

That is the point of the arrangement. The caption is not computed beside the
graph, it is computed from the graph's output, so:

    **bend the vision and the sentence changes.**

Scale an attention head down and the model stops noticing the thing it was
describing; mask a patch embedding and it invents something else. You can watch
that happen: bend a node, re-run `caption`, read the new sentence.

The image is an ordinary graph input — drop a file on `pixel_values` — and the
callback captions whatever is currently in the bench, which it learns through
:meth:`on_inputs`.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..spec import (Callback, Ref, Image, Input, Int, Method, Option, Str, Tensor,
                    Text, Tokens)
from ..base import Interface
from ...tracing import ScriptableState


__all__ = ["BendedBlip", "BendingBlipException"]


_DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"


class BendingBlipException(Exception):
    pass


class _VisionShim(nn.Module):
    """Stands in for BLIP's vision tower so the bended graph is on the path.

    ``generate`` calls ``self.vision_model(pixel_values, ...)`` and hands the
    result to the decoder. Swapping this in for the duration of a call is what
    makes a bending show up in the sentence -- bending a copy nothing calls
    would change nothing you could read.
    """

    def __init__(self, bended):
        super().__init__()
        self._bended = bended

    def forward(self, pixel_values, *args, **kwargs):
        out = self._bended.forward(pixel_values=pixel_values)
        if torch.is_tensor(out):
            return (out,)
        if isinstance(out, dict):
            hidden = out.get("last_hidden_state", next(iter(out.values())))
            return (hidden,)
        return out


class BendedBlip(Interface):
    """Image captioning, with the vision tower as the bendable graph."""

    _imported_callbacks_ = []
    _panel_render_type_ = "text"

    #: Lets the viewer decode a raw id tensor (``caption_tokens``) into the text it
    #: stands for, and edit it token by token — see :meth:`decode_caption_tokens` /
    #: :meth:`caption_logits`.
    tokens = Tokens(decode="decode_caption_tokens", logits="caption_logits",
                    eos=Ref("eos_token_id"))

    options = {
        "prompt": Option(
            Str(), attr="prompt",
            doc="Text the caption must continue, e.g. 'a photograph of'. "
                "Empty lets BLIP start on its own."),
        "max_new_tokens": Option(
            Int(range=(4, 128)), attr="max_new_tokens", label="max tokens",
            doc="How long the caption may get."),
        "num_beams": Option(
            Int(range=(1, 8)), attr="num_beams", label="beams",
            doc="1 is greedy. More beams read better and take longer."),
    }

    callbacks = {
        "caption": Callback(
            returns=Text(), label="caption (image → text)",
            args={"prompt": Text(optional=True, placeholder="optional lead-in, e.g. 'a photo of'"),
                  "seed": Int(optional=True)},
            doc="Describe the image currently in the input bench, reading the "
                "vision features off the bended graph. Bend a node and run this "
                "again to hear what changed."),
        "caption_tokens": Callback(
            returns=Tensor(), label="caption tokens (editable)",
            args={"prompt": Text(optional=True, placeholder="optional lead-in, e.g. 'a photo of'"),
                  "seed": Int(optional=True)},
            doc="Same caption as `caption`, as the raw token ids instead of "
                "text — the viewer decodes and shows it as editable tokens, "
                "with next-token probabilities from `caption_logits`, so you "
                "can swap a word and see how the rest would have continued."),
        "describe_patches": Callback(
            returns=Image(), label="patch attention (what it looked at)",
            doc="The norm of each patch embedding, laid back out on the image "
                "grid — roughly where the vision tower put its weight. Bend the "
                "graph and watch this move."),
    }

    methods = {"forward": Method(inputs={"pixel_values": Input(
        lambda self: "torch.rand(1, 3, %d, %d)" % (self._image_size, self._image_size))})}

    def __init__(self, model_path: str = _DEFAULT_MODEL,
                 prompt: str = "", max_new_tokens: int = 30, num_beams: int = 1,
                 device=torch.device("cpu"), model=None, processor=None, **kwargs):
        self.device = device
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        if model is None or processor is None:
            model, processor = self.load_model(model_path, device=device, **kwargs)
        self._full = model
        self._processor = processor
        self._pixels = None          # what the bench is currently showing
        size = int(model.config.vision_config.image_size)
        self._image_size = size
        super().__init__(model.vision_model)

    # -- loading --

    @staticmethod
    def load_model(model_path=_DEFAULT_MODEL, device="cpu", **kwargs):
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:                      # pragma: no cover
            raise BendingBlipException(
                "BLIP needs `transformers`: pip install transformers") from exc
        try:
            processor = BlipProcessor.from_pretrained(str(model_path))
            model = BlipForConditionalGeneration.from_pretrained(str(model_path), **kwargs)
        except Exception as exc:
            raise BendingBlipException(
                "could not load BLIP model %s, got: %s" % (model_path, exc))
        return model.eval().to(device), processor

    @staticmethod
    def is_loadable(path) -> bool:
        try:
            BendedBlip.load_model(path)
            return True
        except Exception:
            return False

    # -- tracing --

    def bend_model(self, model):
        """Trace the vision tower: pixels to patch embeddings.

        The decoder is deliberately not traced -- it writes one token at a time,
        and how many depends on what it has written so far.
        """
        model.trace("forward", pixel_values=self._blank_image())

    def _blank_image(self):
        return torch.rand(1, 3, self._image_size, self._image_size, device=self.device)

    # -- what the bench is running on --

    def on_inputs(self, fn, kwargs):
        """Remember the image, so `caption` describes the one on screen."""
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
        """BLIP's tokenizer is BERT-based and has no ``eos_token`` — captions end on
        ``[SEP]`` instead (every caption's last id is its ``sep_token_id``). Without
        this override the base class's generic lookup finds nothing, and the
        viewer's token-continuation feature never stops: it keeps greedily
        predicting past the caption's real end, into context BLIP was never trained
        to continue, which is what produces a stream of near-random ``[unusedNNN]``
        placeholder tokens instead of stopping cleanly."""
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
        """The image the graph last ran on, if there was one."""
        return self._pixels

    def encode_image(self, image) -> torch.Tensor:
        """A PIL image, path or array as the tensor the graph takes."""
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image).convert("RGB")
        return self._processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)

    # -- inference --

    def _current_pixels(self):
        if self._pixels is None:
            raise BendingBlipException(
                "no image yet — drop one on `pixel_values` in the input bench, "
                "or call `encode_image` and pass it in; the caption describes "
                "whatever the graph last ran on")
        return self._pixels

    def features(self, pixel_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Patch embeddings ``[B, 577, 768]`` **from the bended graph**."""
        pixels = self._current_pixels() if pixel_values is None else pixel_values
        with torch.no_grad():
            out = self._model.forward(pixel_values=pixels)
        if torch.is_tensor(out):
            return out
        if isinstance(out, dict):
            return out.get("last_hidden_state", next(iter(out.values())))
        return out[0]

    def _generate_caption_ids(self, prompt: Optional[str] = None, seed: Optional[int] = None,
                               pixel_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Caption ids ``[B, T]``, reading vision features off the bended graph.

        Shared by :meth:`caption` (decodes to text) and :meth:`caption_tokens`
        (returns the ids as-is, for the viewer's editable-token display).
        """
        pixels = self._current_pixels() if pixel_values is None else pixel_values
        lead = self.prompt if prompt is None else prompt

        inputs = {"pixel_values": pixels}
        if lead:
            enc = self._processor(text=lead, return_tensors="pt")
            inputs["input_ids"] = enc["input_ids"].to(self.device)
            inputs["attention_mask"] = enc["attention_mask"].to(self.device)

        original = self._full.vision_model
        self._full.vision_model = _VisionShim(self._model)
        if seed is not None:
            torch.manual_seed(seed)
        try:
            with torch.no_grad():
                ids = self._full.generate(
                    **inputs, max_new_tokens=int(self.max_new_tokens),
                    num_beams=int(self.num_beams))
        finally:
            self._full.vision_model = original
        return ids

    def caption(self, prompt: Optional[str] = None, seed: Optional[int] = None,
                pixel_values: Optional[torch.Tensor] = None) -> List[str]:
        """Describe the image, reading its features off the bended graph."""
        ids = self._generate_caption_ids(prompt=prompt, seed=seed, pixel_values=pixel_values)
        return self._processor.batch_decode(ids, skip_special_tokens=True)

    def caption_tokens(self, prompt: Optional[str] = None, seed: Optional[int] = None,
                        pixel_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Caption ids ``[1, T]`` — same as :meth:`caption`, but as the raw tensor
        so the viewer can show it as editable text (see :meth:`decode_caption_tokens`)
        with next-token probabilities (see :meth:`caption_logits`)."""
        return self._generate_caption_ids(prompt=prompt, seed=seed, pixel_values=pixel_values)

    def decode_caption_tokens(self, ids: torch.Tensor, skip_special_tokens: bool = True):
        """Token ids back to text, one string per row — BLIP's tokenizer doing what
        the viewer needs to show ``caption_tokens`` as text and let it be edited."""
        return self._processor.batch_decode(ids, skip_special_tokens=skip_special_tokens)

    def caption_logits(self, ids: torch.Tensor) -> torch.Tensor:
        """Logits ``[B, T, V]`` for a candidate caption sequence, conditioned on the
        image currently in the bench. One teacher-forced pass, not a generation loop:
        this is what lets the viewer preview what BLIP would say next after a token
        in ``caption_tokens`` is edited, the same way GPT-2's own continuation does."""
        pixels = self._current_pixels()
        with torch.no_grad():
            out = self._full(pixel_values=pixels, input_ids=ids)
        return out.logits if hasattr(out, "logits") else out[0]

    def describe_patches(self, pixel_values: Optional[torch.Tensor] = None):
        """Per-patch embedding norm, back on the image grid.

        Not an attention map -- it is where the tower's activations are large,
        which is a rougher thing. Useful because it *moves* when you bend, so
        you can see which part of the picture a bending took away.
        """
        feats = self.features(pixel_values)
        patches = feats[:, 1:, :]        # drop the CLS token
        n = patches.shape[1]
        side = int(round(n ** 0.5))
        if side * side != n:
            raise BendingBlipException(
                "%d patches do not make a square grid" % n)
        norms = patches.norm(dim=-1)                       # [B, n]
        grid = norms.reshape(patches.shape[0], 1, side, side)
        lo, hi = grid.amin(), grid.amax()
        return (grid - lo) / (hi - lo + 1e-8)

    # -- export --

    @property
    def scriptable(self):
        return ScriptableState.NotScriptable

    @property
    def nntilde_compatible(self):
        return ScriptableState.NotScriptable
