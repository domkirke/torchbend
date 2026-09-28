"""What rate an activation is played back at.

Listening to an intermediate activation only means anything if the rate is
right, and the rate is not in the tensor: a codec's latent runs at one frame per
hop, a mel at one per STFT hop, a waveform at the model's own rate, and nothing
about a shape tells them apart. The model declares it — see
:mod:`torchbend.sample_rates` — and the viewer only guesses where nothing has
been said, and only as far as the model has agreed it may.
"""

import io
import wave

import pytest
import torch

from torchbend.sample_rates import (SampleRateError, SampleRateMixin,
                                    normalize_sample_rates)
from torchbend.ui.graph_viewer import views as V


# ── the declaration ───────────────────────────────────────────────────────────

class Model(SampleRateMixin):
    """Stands in for an interface or a BendedModule; both read the same table."""
    def __init__(self, rates=None):
        if rates is not None:
            self._sample_rates_ = rates
        self.asked = []

    def rate_of(self, node, fn=None, shape=None):
        self.asked.append((node, fn, shape))
        return 16000 if node.startswith("audio") else None


def test_nothing_declared_says_nothing():
    assert Model().sample_rate_for("audio") is None


def test_a_bare_number_covers_every_node():
    m = Model(44100)
    assert m.sample_rate_for("audio") == 44100
    assert m.sample_rate_for("anything", fn="decode") == 44100


def test_a_dict_names_nodes():
    m = Model({"audio": 44100, "z": 21.53})
    assert m.sample_rate_for("audio") == 44100
    assert m.sample_rate_for("z") == pytest.approx(21.53)
    assert m.sample_rate_for("hidden") is None


def test_a_nested_dict_names_methods():
    m = Model({"decode": {"audio": 44100}, "encode": {"z": 21.5}})
    assert m.sample_rate_for("audio", fn="decode") == 44100
    assert m.sample_rate_for("audio", fn="encode") is None
    assert m.sample_rate_for("z", fn="encode") == 21.5


def test_the_method_entry_beats_the_general_one():
    m = Model({"audio": 44100, "decode": {"audio": 22050}})
    assert m.sample_rate_for("audio") == 44100
    assert m.sample_rate_for("audio", fn="decode") == 22050


def test_a_star_is_the_catch_all():
    m = Model({"decode": {"*": 48000, "z": 375}})
    assert m.sample_rate_for("whatever", fn="decode") == 48000
    assert m.sample_rate_for("z", fn="decode") == 375
    assert m.sample_rate_for("whatever", fn="encode") is None


# ── the dynamic form ──────────────────────────────────────────────────────────

def test_a_method_name_is_called_for_the_rate():
    m = Model("rate_of")
    assert m.sample_rate_for("audio_out", fn="forward", shape=[1, 1, 4]) == 16000
    assert m.asked == [("audio_out", "forward", [1, 1, 4])]


def test_a_callable_is_called_too():
    m = Model({"z": lambda node: 375})
    assert m.sample_rate_for("z") == 375


def test_a_callback_reached_through_a_wrapper_still_gets_the_shape():
    """Naming a method on a bare BendedModule resolves to the wrapped module's.

    That arrives through a ``(*args, **kwargs)`` forwarder, so reading its
    signature literally passed neither `fn` nor `shape` — and a callback that
    keyed on shape answered None for every node.
    """
    seen = []

    def forwarder(*args, **kwargs):
        seen.append((args, kwargs))
        return 375

    m = Model({"z": forwarder})
    assert m.sample_rate_for("z", fn="decode", shape=[1, 8, 4]) == 375
    assert seen == [(("z",), {"fn": "decode", "shape": [1, 8, 4]})]


def test_a_callback_is_given_only_what_it_asks_for():
    seen = []
    m = Model({"z": lambda node: seen.append(node) or 375})
    assert m.sample_rate_for("z", fn="decode", shape=[1, 8, 4]) == 375
    assert seen == ["z"]


def test_none_from_a_callback_means_no_opinion():
    """...so the next, less specific declaration gets its turn."""
    m = Model({"rate_of": None})   # placeholder, replaced below
    m._sample_rates_ = {"decode": {"*": "rate_of"}, "z": 375}
    assert m.sample_rate_for("audio_x", fn="decode") == 16000   # the callback answered
    assert m.sample_rate_for("z", fn="decode") == 375           # it declined; the dict has it


def test_a_declaration_can_be_set_from_outside():
    m = Model()
    m.set_sample_rates({"audio": 44100})
    assert m.sample_rate_for("audio") == 44100


def test_the_rates_for_a_method_merge_the_general_ones():
    m = Model({"audio": 44100, "decode": {"z": 21.5}})
    assert dict(m.sample_rates(fn="decode")) == {"audio": 44100.0, "z": 21.5}


# ── declarations that cannot work are refused at declaration time ─────────────

def test_a_named_method_that_does_not_exist_is_refused():
    with pytest.raises(SampleRateError) as exc:
        normalize_sample_rates(Model(), {"z": "no_such_method"})
    assert "no_such_method" in str(exc.value)


def test_a_callable_that_cannot_take_the_node_is_refused():
    """The usual slip: a nested function written as though it were a method.

    ``def rate(self, node, ...)`` passed as a plain callable binds the node name
    to `self` and dies for a missing `node` — at read time, inside a viewer
    request, where the only symptom was that the rate did nothing.
    """
    def looks_like_a_method(self, node, fn=None, shape=None):
        return 375

    with pytest.raises(SampleRateError) as exc:
        normalize_sample_rates(Model(), looks_like_a_method)
    assert "f(node)" in str(exc.value)

    with pytest.raises(SampleRateError):
        Model().set_sample_rates({"z": looks_like_a_method})


def test_a_bound_method_is_accepted():
    """The same function, bound, takes the node just fine."""
    m = Model()
    m.set_sample_rates(m.rate_of)
    assert m.sample_rate_for("audio_x") == 16000


def test_a_rate_that_is_not_a_number_is_refused():
    with pytest.raises(SampleRateError):
        normalize_sample_rates(Model(), {"z": "  "})


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_rate_is_refused(bad):
    with pytest.raises(SampleRateError):
        normalize_sample_rates(Model(), {"z": bad})


def test_set_sample_rates_refuses_a_broken_table():
    m = Model()
    with pytest.raises(SampleRateError):
        m.set_sample_rates({"z": "no_such_method"})
    assert m.sample_rate_for("z") is None


# ── declared on the nn.Module itself ──────────────────────────────────────────
#
# The natural place to put it, and the one that silently did nothing: the
# wrapper's own empty `_sample_rates_` shadowed the model's, so BendedModule's
# attribute forwarding never fired.

import torch.nn as nn                                             # noqa: E402
import torchbend as tb                                            # noqa: E402


def _traced(**attrs):
    body = {
        "__init__": lambda self: (nn.Module.__init__(self),
                                  setattr(self, "lin", nn.Linear(4, 4)))[0],
        "forward": lambda self, x: torch.tanh(self.lin(x)),
    }
    bm = tb.BendedModule(type("M", (nn.Module,), {**body, **attrs})())
    bm.trace(x=torch.randn(1, 4))
    return bm


def test_a_declaration_on_the_module_is_read_through():
    bm = _traced(_sample_rates_={"tanh": 375})
    assert bm.sample_rate_for("tanh") == 375
    assert bm.sample_rate_for("lin_weight") is None


def test_a_module_may_answer_with_its_own_method():
    bm = _traced(sample_rate_for=lambda self, node, fn=None, shape=None:
                 375 if node == "tanh" else None)
    assert bm.sample_rate_for("tanh") == 375
    assert bm.sample_rate_for("addmm") is None


def test_a_named_method_resolves_on_the_module():
    bm = _traced(_sample_rates_="node_rate",
                 node_rate=lambda self, node, fn=None, shape=None:
                 375 if shape and shape[-1] == 4 else None)
    assert bm.sample_rate_for("tanh", shape=[1, 4]) == 375
    assert bm.sample_rate_for("tanh", shape=[1, 8]) is None


def test_the_wrapper_overrides_the_module():
    bm = _traced(_sample_rates_={"tanh": 375})
    bm.set_sample_rates({"tanh": 750})
    assert bm.sample_rate_for("tanh") == 750


def test_strided_audio_on_the_module_is_read_through():
    assert V._strided_audio(_traced(_strided_audio_=True))
    assert not V._strided_audio(_traced())


# ── what the viewer does with it ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """No audio uploads and no interface, unless a test says otherwise."""
    monkeypatch.setattr(V, "_last_audio_sr", {})
    monkeypatch.setattr(V, "get_interface", lambda: None)


class _Iface(SampleRateMixin):
    """Stands in for an audio interface, which knows its own rate."""
    sample_rate = 24000


def test_a_declared_rate_is_the_answer(monkeypatch):
    """Loudly: nothing else is consulted, however much else is known."""
    iface = _Iface()
    iface._sample_rates_ = {"z": 375}
    monkeypatch.setattr(V, "get_interface", lambda: iface)
    monkeypatch.setattr(V, "_last_audio_sr", {"x": 16000})
    kwargs = {"x": torch.zeros(1, 1, 48000)}
    assert V._infer_output_sr(torch.zeros(1, 8, 1500), kwargs, fn="forward", node="z") == 375


def test_the_module_is_asked_when_the_interface_says_nothing():
    class _Module(SampleRateMixin):
        _sample_rates_ = {"z": 375}
    sr = V._infer_output_sr(torch.zeros(1, 8, 1500), {},
                            bended_module=_Module(), node="z")
    assert sr == 375


def test_a_broken_declaration_does_not_break_the_run():
    class _Module(SampleRateMixin):
        def sample_rate_for(self, node, fn=None, shape=None):
            raise RuntimeError("boom")
    sr = V._infer_output_sr(torch.zeros(1, 1, 4096), {},
                            bended_module=_Module(), node="z")
    assert sr == V._DEFAULT_AUDIO_SR


def test_a_declared_node_is_audio(monkeypatch):
    iface = _Iface()
    iface._sample_rates_ = {"z": 375}
    monkeypatch.setattr(V, "get_interface", lambda: iface)
    assert V._declares_audio("z", "forward", None)
    assert not V._declares_audio("hidden", "forward", None)


# ── the fallback, for a model that declares nothing ───────────────────────────

def test_nothing_known_falls_back_to_the_default():
    assert V._infer_output_sr(torch.zeros(1, 1, 4096), {}) == V._DEFAULT_AUDIO_SR


def test_the_interface_rate_is_used_when_no_audio_came_in(monkeypatch):
    monkeypatch.setattr(V, "get_interface", lambda: _Iface())
    assert V._infer_output_sr(torch.zeros(1, 1, 4096), {}) == 24000


def test_an_audio_input_is_the_reference(monkeypatch):
    monkeypatch.setattr(V, "_last_audio_sr", {"x": 16000})
    kwargs = {"x": torch.zeros(1, 1, 48000)}
    assert V._infer_output_sr(torch.zeros(1, 2, 48000), kwargs) == 16000


def test_the_input_that_this_run_carries_wins(monkeypatch):
    """Two audio inputs remembered; only one is in this run's kwargs."""
    monkeypatch.setattr(V, "_last_audio_sr", {"other": 8000, "x": 16000})
    kwargs = {"x": torch.zeros(1, 1, 48000)}
    assert V._infer_output_sr(torch.zeros(1, 2, 48000), kwargs) == 16000


def test_a_shorter_tensor_is_not_placed_on_the_timeline_by_default(monkeypatch):
    """The guess the model has not agreed to: it does not run."""
    monkeypatch.setattr(V, "_last_audio_sr", {"x": 16000})
    kwargs = {"x": torch.zeros(1, 1, 48000)}
    assert V._infer_output_sr(torch.zeros(1, 8, 1500), kwargs) == 16000


def test_a_strided_model_places_it(monkeypatch):
    """A fully convolutional codec has said its graph is one timeline."""
    class _Codec(SampleRateMixin):
        _strided_audio_ = True
    monkeypatch.setattr(V, "_last_audio_sr", {"x": 16000})
    kwargs = {"x": torch.zeros(1, 1, 48000)}
    sr = V._infer_output_sr(torch.zeros(1, 8, 1500), kwargs, bended_module=_Codec())
    assert sr == 500          # 1500 frames over the same 3 seconds


def test_a_strided_model_reads_its_own_output_when_nothing_came_in(monkeypatch):
    class _Codec(_Iface):
        _strided_audio_ = True
    monkeypatch.setattr(V, "get_interface", lambda: _Codec())
    monkeypatch.setattr(V, "_traced_output_length", lambda bm, fn: 48000)
    sr = V._infer_output_sr(torch.zeros(1, 8, 1500), {},
                            bended_module=object(), fn="forward")
    assert sr == 750          # 1500 frames over the 2 s the output covers


def test_a_model_that_is_not_strided_is_never_measured(monkeypatch):
    """The gate is `_strided_audio_`; without it nothing is even looked up."""
    called = []
    monkeypatch.setattr(V, "_traced_output_length",
                        lambda bm, fn: called.append(1) or 48000)
    sr = V._infer_output_sr(torch.zeros(1, 8, 1500), {},
                            bended_module=object(), fn="forward")
    assert sr == V._DEFAULT_AUDIO_SR
    assert not called, "the reference length is only wanted for scaling"


def test_a_bare_module_can_say_what_rate_it_runs_at():
    """Asking only the interface left a wrapped nn.Module with no way to say."""
    class _Model:
        sample_rate = 22050
    class _Wrapper(SampleRateMixin):
        _module = _Model()
    assert V._declared_sample_rate(_Wrapper()) == 22050

    class _SrModel:
        sr = 16000                      # the other common spelling
    class _SrWrapper(SampleRateMixin):
        _module = _SrModel()
    assert V._declared_sample_rate(_SrWrapper()) == 16000


def test_a_strided_bare_module_scales_with_no_audio_input(monkeypatch):
    """The reported case: indices in, audio out, `_strided_audio_` set.

    Nothing audio is fed and there is no interface, so the reference is the
    model's own rate and the length of what it returns. Before, the missing
    interface ended the search at the default and every node read 44100
    however short it was.
    """
    class _Model:
        sr = 44100
    class _Wrapper(SampleRateMixin):
        _strided_audio_ = True
        _module = _Model()
    monkeypatch.setattr(V, "_traced_output_length", lambda bm, fn: 48000)
    bm = _Wrapper()
    assert V._infer_output_sr(torch.zeros(1, 1, 48000), {},
                              bended_module=bm, fn="forward") == 44100
    assert V._infer_output_sr(torch.zeros(1, 8, 1500), {},
                              bended_module=bm, fn="forward") == 1378


# ── the file that comes back ──────────────────────────────────────────────────

def _read(wav_bytes):
    w = wave.open(io.BytesIO(wav_bytes))
    return w.getnchannels(), w.getframerate(), w.getnframes()


def test_a_channel_is_rendered_on_its_own():
    wav, sr = V._tensor_to_wav(torch.zeros(2, 4, 16000), 16000, batch=1, channel=2)
    assert _read(wav) == (1, 16000, 16000)
    assert sr == 16000


def test_the_mixdown_keeps_every_channel():
    wav, _ = V._tensor_to_wav(torch.zeros(1, 4, 16000), 16000, channel=-1)
    assert _read(wav)[0] == 4


def test_a_sub_audio_rate_is_raised_but_the_duration_is_not():
    """A latent at 500 frames a second is a WAV no browser will decode."""
    wav, sr = V._tensor_to_wav(torch.zeros(1, 1, 1500), 500)   # 3 s at 500 Hz
    n_ch, rate, frames = _read(wav)
    assert sr == rate == V._MIN_PLAYABLE_SR
    assert frames / rate == pytest.approx(3.0, rel=1e-3)


def test_a_playable_rate_is_left_alone():
    wav, sr = V._tensor_to_wav(torch.zeros(1, 1, 16000), 16000)
    assert sr == 16000
    assert _read(wav)[2] == 16000
