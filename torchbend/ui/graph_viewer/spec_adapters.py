"""The graph viewer's side of :mod:`torchbend.interfaces.spec`.

An interface declares what its values *are* (a :class:`~torchbend.interfaces.
spec.Text` prompt, an :class:`~torchbend.interfaces.spec.Audio` recording); how
they are shown and collected is each frontend's business, decided here by the
value's class. The browser draws a widget per ``type`` string; this module is
the server half: how a request's payload for an input mode becomes the raw value
handed to the interface's encoder.

A new value type is supported by registering a reader for it::

    from torchbend.ui.graph_viewer.spec_adapters import register_reader

    @register_reader(MyImage)
    def read_image(request, name):
        return [decode(f) for f in request.FILES.getlist(name)]

A reader is found along the value's MRO, so a subclass of ``Audio`` is read as
audio until it registers its own. An input mode whose value has no reader is
not offered by the viewer (and says why in the log) rather than offered broken.
"""

import io
import logging

import torch

from ...interfaces.spec import Audio, Text


#: The name `ui=` hints are keyed by for this frontend.
FRONTEND = "graph_viewer"

_READERS = {}


def register_reader(value_cls):
    """Register ``fn(request, name) -> [raw value, ...]`` for a Value class."""
    def deco(fn):
        _READERS[value_cls] = fn
        return fn
    return deco


def reader_for(value):
    for cls in type(value).__mro__:
        if cls in _READERS:
            return _READERS[cls]
    return None


def supported(mode) -> bool:
    """Whether the viewer can collect this input mode's value."""
    if reader_for(mode.value) is None:
        logging.getLogger(__name__).warning(
            "input mode of type %r has no reader in the graph viewer; not offered",
            mode.kind)
        return False
    return True


def read_mode_input(mode, request, name) -> list:
    """What the request carries for ``name`` under its input mode."""
    return reader_for(mode.value)(request, name)


def _posted_strings(request, name):
    return [r for r in request.POST.getlist(name) if str(r).strip() != ""]


@register_reader(Text)
def _read_text(request, name):
    return _posted_strings(request, name)


@register_reader(Audio)
def _read_audio(request, name):
    # typed paths, and uploaded recordings decoded to (waveform, rate) -- the
    # interface prepares either its own way
    return _posted_strings(request, name) + [
        decode_audio_upload(f) for f in request.FILES.getlist(name)]


def decode_audio_upload(f):
    """An uploaded recording as ``(waveform [C, L] float32, sample_rate)``.

    soundfile first: it reads wav/flac/ogg/aiff with no system libraries,
    where `torchaudio.load` needs torchcodec and FFmpeg's shared libraries and
    fails to load at all without them. torchaudio is only the fallback, for
    formats soundfile does not know (mp3 on older libsndfile).
    """
    data = f.read()
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)   # [L, C]
        return torch.from_numpy(audio.T.copy()), int(sr)
    except Exception as sf_exc:
        try:
            import torchaudio
            t, sr = torchaudio.load(io.BytesIO(data))
            return t.float(), int(sr)
        except Exception:
            raise ValueError("could not read %r as audio (%s)"
                             % (getattr(f, "name", "upload"), sf_exc)) from sf_exc
