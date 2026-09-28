"""Bending interface for an image → sound pipeline — a picture in, a sound out.

Two models chained together. **BLIP**'s vision tower turns an image into patch
embeddings and its text decoder reads those into a caption (exactly the
arrangement in :class:`~torchbend.interfaces.blip.BendedBlip` — only the
vision tower is a fixed-shape graph, so only it is traced and bendable).
That caption becomes the text prompt for **AudioGen** or **MusicGen**, which
render it into a few seconds of sound.

The bendable graph is BLIP's vision tower, inherited as-is:

    **bend the vision and the sound changes**,

because the caption driving the sound generator is read off that graph, the
same way BLIP's own caption is. The sound generator itself is not part of the
graph — like BLIP's text decoder, it samples in a loop of unpredictable
length, so it runs as a plain call after the graph, on whatever caption the
(possibly bent) vision tower currently produces.

The image is an ordinary graph input, same as in ``BendedBlip`` — drop a file
on ``pixel_values``.
"""

from typing import List, Optional, Union

import torch

from ..spec import Audio, Callback, Float, Int, Option, Text
from ..blip.interface import BendedBlip, BendingBlipException, _DEFAULT_MODEL as _DEFAULT_BLIP_MODEL


__all__ = ["BendedImg2Sound", "BendingImg2SoundException"]


_DEFAULT_SOUND_MODELS = {
    "audiogen": "facebook/audiogen-medium",
    "musicgen": "facebook/musicgen-small",
}


class BendingImg2SoundException(Exception):
    pass


class BendedImg2Sound(BendedBlip):
    """Image to sound: BLIP captions the image (bendable vision tower, inherited from
    BendedBlip), and the caption is rendered into sound by AudioGen or MusicGen."""

    _panel_render_type_ = "audio"

    options = dict(BendedBlip.options, **{
        "duration": Option(
            Float(range=(1., 30.)), attr="duration", label="duration (s)",
            doc="Length of the generated sound, in seconds."),
        "temperature": Option(
            Float(range=(0.1, 2.0)), attr="gen_temperature", label="gen. temperature",
            doc="Sampling temperature for the sound generator (not BLIP's captioning)."),
    })

    callbacks = dict(BendedBlip.callbacks, **{
        "sonify": Callback(
            returns=Audio(), label="sonify (image → sound)",
            args={"prompt": Text(optional=True,
                                 placeholder="optional lead-in for the caption, e.g. 'a photo of'"),
                  "sound_prompt": Text(optional=True, placeholder="override the caption entirely"),
                  "seed": Int(optional=True)},
            doc="Caption the image (reading vision features off the bended graph) "
                "and render that caption into sound. Bend a node and run this "
                "again to hear what changed."),
    })

    def __init__(self, model_path: str = _DEFAULT_BLIP_MODEL, generator: str = "audiogen",
                 sound_model_path: Optional[str] = None,
                 prompt: str = "", max_new_tokens: int = 30, num_beams: int = 1,
                 duration: float = 5.0,
                 device=torch.device("cpu"), model=None, processor=None,
                 sound_model=None, **kwargs):
        if generator not in _DEFAULT_SOUND_MODELS:
            raise BendingImg2SoundException(
                "generator must be one of %s, got %r" % (list(_DEFAULT_SOUND_MODELS), generator))
        self.generator = generator
        self.duration = float(duration)
        self.gen_temperature = 1.0
        self._last_caption = None
        if sound_model is None:
            sound_model = self.load_sound_model(
                generator, sound_model_path or _DEFAULT_SOUND_MODELS[generator], device=device)
        self._sound_model = sound_model
        self._sound_model.set_generation_params(duration=self.duration, temperature=self.gen_temperature)
        super().__init__(model_path=model_path, prompt=prompt, max_new_tokens=max_new_tokens,
                          num_beams=num_beams, device=device, model=model, processor=processor, **kwargs)

    # -- loading --

    @staticmethod
    def load_sound_model(generator: str, model_path: str, device="cpu"):
        try:
            from audiocraft.models import AudioGen, MusicGen
        except ImportError as exc:                      # pragma: no cover
            raise BendingImg2SoundException(
                "img2sound needs `audiocraft`: pip install audiocraft") from exc
        cls = AudioGen if generator == "audiogen" else MusicGen
        try:
            model = cls.get_pretrained(model_path, device=device)
        except Exception as exc:
            raise BendingImg2SoundException(
                "could not load %s model %s, got: %s" % (generator, model_path, exc))
        return model

    # -- properties --

    @property
    def sample_rate(self) -> int:
        """The sound generator's output rate — read generically by the graph viewer
        to play back an audio-returning callback's result at the right speed."""
        return self._sound_model.sample_rate

    @property
    def sound_model(self):
        return self._sound_model

    @property
    def last_caption(self) -> Optional[str]:
        """The caption `sonify` last rendered into sound."""
        return self._last_caption

    # -- inference --

    def sonify(self, prompt: Optional[str] = None, sound_prompt: Optional[str] = None,
               seed: Optional[int] = None, pixel_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Caption the image through the (possibly bent) vision tower, then render
        that caption into sound. `sound_prompt` bypasses captioning entirely, e.g. to
        compare what the bent vision tower says against a fixed reference prompt."""
        if sound_prompt:
            caption = sound_prompt
        else:
            captions = self.caption(prompt=prompt, seed=seed, pixel_values=pixel_values)
            caption = captions[0]
        self._last_caption = caption
        self._sound_model.set_generation_params(duration=self.duration, temperature=self.gen_temperature)
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            audio = self._sound_model.generate([caption])
        return audio
