"""Tests for the XTTS-v2 interface.

coqui `TTS` and the 2GB checkpoint are usually not around, so the tests that
matter here run against a **stub** standing in for `Xtts`'s surface: the same
attributes and the same `inference()` call shape, read off TTS v0.22.0's
source. That is enough to exercise everything the interface itself is
responsible for — tracing the decoder, routing `inference` through the bended
graph via the shim, caching the latents, and the audio that comes out — without
downloading anything.

`test_real_model_*` covers the parts only real XTTS can answer and skips when
it is unavailable.
"""
import pytest
import torch
import torch.nn as nn

from torchbend.interfaces.xtts import BendedXTTS, BendingXTTSException
from torchbend.interfaces.xtts.pipeline import XTTSPipeline


LATENT_DIM, SPEAKER_DIM, FRAMES, SR = 32, 16, 24, 24000

TTS_AVAILABLE = False
try:
    import TTS  # noqa: F401
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


# ── a stand-in for Xtts ──────────────────────────────────────────────────────

class _StubDecoder(nn.Module):
    """Same contract as TTS's HifiDecoder: [B, T, C] + [B, D, 1] -> [B, 1, T']."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(LATENT_DIM, 8)
        self.cond = nn.Linear(SPEAKER_DIM, 8)
        self.output_sample_rate = SR

    def forward(self, latents, g=None):
        h = self.proj(latents)                       # [B, T, 8]
        if g is not None:
            h = h + self.cond(g.squeeze(-1)).unsqueeze(1)
        return torch.tanh(h).reshape(latents.shape[0], 1, -1)


class _StubArgs:
    decoder_input_dim = LATENT_DIM
    d_vector_dim = SPEAKER_DIM


class _StubSpeakerManager:
    def __init__(self):
        self.speakers = {
            "Test Voice": {"gpt_cond_latent": torch.randn(1, FRAMES, LATENT_DIM),
                           "speaker_embedding": torch.randn(1, SPEAKER_DIM, 1)},
        }


class _StubXtts(nn.Module):
    """Mimics the bits of `Xtts` the interface touches."""

    def __init__(self):
        super().__init__()
        self.hifigan_decoder = _StubDecoder()
        self.args = _StubArgs()
        self.speaker_manager = _StubSpeakerManager()
        self.inference_calls = 0

    def get_conditioning_latents(self, audio_path=None, **kwargs):
        return torch.randn(1, FRAMES, LATENT_DIM), torch.randn(1, SPEAKER_DIM, 1)

    def inference(self, text, language, gpt_cond_latent, speaker_embedding, **kwargs):
        """Shaped like the real one: the decoder is called *from here*, which is
        what the interface's shim relies on to make bendings audible."""
        self.inference_calls += 1
        latents = torch.randn(1, FRAMES, LATENT_DIM)
        wav = self.hifigan_decoder(latents, g=speaker_embedding)
        return {"wav": wav.detach().cpu().squeeze().numpy(),
                "gpt_latents": latents.numpy(),
                "speaker_embedding": speaker_embedding}


@pytest.fixture
def iface():
    torch.manual_seed(0)
    return BendedXTTS(model=_StubXtts(), config=None, trace_frames=FRAMES,
                      speaker="Test Voice")


# ── what the interface itself is responsible for ─────────────────────────────

def test_the_decoder_is_the_graph(iface):
    """The bendable graph is the decoder — latents and a voice in, audio out."""
    names = iface.model.activation_names()
    assert len(names) > 3
    assert iface.sample_rate == SR
    assert iface.speakers == ["Test Voice"]


def test_speak_routes_through_the_bended_graph(iface):
    """`inference` must call *our* graph, or bendings would be inaudible."""
    audio = iface.speak(text="hello")
    assert audio.ndim == 2 and audio.shape[0] == 1 and audio.shape[1] > 0
    assert iface._full.inference_calls == 1


def test_latents_are_kept_for_redecoding(iface):
    """Generating latents is slow; decoding is not. The interface keeps them."""
    assert iface.latents is None
    iface.speak(text="hello")
    latents, speaker = iface.latents
    assert latents.shape == (1, FRAMES, LATENT_DIM)
    assert speaker.shape == (1, SPEAKER_DIM, 1)

    audio = iface.decode_latents()
    assert audio.ndim == 2 and audio.shape[1] > 0
    assert iface._full.inference_calls == 1, "re-decoding must not re-run the GPT"


def test_decode_before_speak_explains_itself(iface):
    with pytest.raises(BendingXTTSException, match="call `speak` once first"):
        iface.decode_latents()


def test_bending_is_audible(iface):
    """The point of the arrangement: bend a decoder weight, hear it change."""
    from torchbend.bending import Mask
    iface.speak(text="hello")
    before = iface.decode_latents()
    iface.model.bend(Mask(prob=0.0), "proj.weight")
    after = iface.decode_latents()
    assert not torch.allclose(before, after)


def test_unknown_speaker_is_refused(iface):
    iface.speaker = "Nobody"
    with pytest.raises(BendingXTTSException, match="not in this checkpoint"):
        iface.conditioning()


def test_no_ui_callbacks_declared(iface):
    """`speak`/`decode_latents` still exist as plain methods (scripts use them),
    but are no longer offered as UI callbacks -- the graph does that now."""
    assert iface.spec.callbacks == {}
    assert callable(iface.speak) and callable(iface.decode_latents)


def test_decoder_only_mode_has_no_aliases_to_retain():
    """Nothing is marked outside `full` -- there is no encoder/GPT in this
    graph for `#conditioning`/`#voice`/etc. to tag."""
    stub = _StubXtts()
    iface_decoder_only = BendedXTTS(model=stub, config=None, trace_frames=FRAMES,
                                    speaker="Test Voice")
    graph = iface_decoder_only.model.graph(fn="forward")
    assert iface_decoder_only.spec.method("forward").retained(graph) == {}


def test_reference_wav_takes_precedence(iface):
    """A reference wav clones that voice instead of the configured speaker."""
    gpt_cond, speaker = iface.conditioning(speaker_wav="/some/voice.wav")
    assert gpt_cond.shape == (1, FRAMES, LATENT_DIM)
    assert speaker.shape == (1, SPEAKER_DIM, 1)


# ── only real XTTS can answer these ──────────────────────────────────────────

@pytest.mark.skipif(not TTS_AVAILABLE, reason="coqui TTS not installed")
def test_real_model_loads_and_speaks():
    try:
        iface = BendedXTTS()
    except BendingXTTSException as exc:
        pytest.skip("XTTS checkpoint unavailable: %s" % exc)
    assert iface.sample_rate == 24000
    audio = iface.speak(text="the graph is the instrument")
    assert audio.shape[1] > iface.sample_rate // 4


# ── the sampling step, which needs no model ─────────────────────────────────

def _draw(logits, seen=(), temperature=1.0, top_k=0, top_p=1.0, penalty=1.0):
    logits = torch.tensor(logits, dtype=torch.float32)
    mask = torch.zeros(logits.numel(), dtype=torch.bool)
    mask[list(seen)] = True
    return int(XTTSPipeline._draw(logits, mask, torch.tensor([temperature]),
                                  top_k, top_p, penalty))


def test_top_k_one_is_greedy():
    for _ in range(10):
        assert _draw([0.1, 3.0, 0.2, 2.9], top_k=1) == 1


def test_top_p_keeps_only_the_nucleus():
    """With one token holding nearly all the mass, a small top_p leaves only it."""
    for _ in range(10):
        assert _draw([8.0, 0.0, 0.0, 0.0], top_p=0.5) == 0


def test_repetition_penalty_moves_a_seen_token_down():
    """Positive scores are divided, negative multiplied -- both make it less likely."""
    assert _draw([2.0, 1.9], seen=[0], penalty=10.0, top_k=1) == 1
    assert _draw([-1.0, -1.1], seen=[0], penalty=10.0, top_k=1) == 1


def test_temperature_is_an_input_not_a_constant():
    """It has to be: it is what the graph takes at run time. A tiny one is greedy."""
    for _ in range(10):
        assert _draw([1.0, 1.2, 0.9], temperature=1e-3) == 1


# ── only real XTTS can answer these ──────────────────────────────────────────

@pytest.mark.skipif(not TTS_AVAILABLE, reason="coqui TTS not installed")
def test_real_model_full_process_is_one_graph(monkeypatch):
    """`forward` is caption + reference voice in, speech out, through the
    encoders, the GPT's sampling loop and the decoder -- and it says what XTTS's
    own inference says for the same seed."""
    import glob
    import os
    import TTS.tts.models.xtts as xtts_module
    from torchbend.interfaces.xtts.interface import read_audio
    # stock XTTS reads references through torchcodec, which needs FFmpeg libs
    monkeypatch.setattr(xtts_module, "load_audio", read_audio)
    try:
        iface = BendedXTTS(full=True, trace_tokens=80, loop_pack=20, loop_open=1)
        from huggingface_hub import snapshot_download
        samples = glob.glob(os.path.join(snapshot_download("coqui/XTTS-v2"), "samples", "en_sample.wav"))
    except Exception as exc:
        pytest.skip("XTTS checkpoint unavailable: %s" % exc)
    if not samples:
        pytest.skip("no reference recording available")
    iface.reference = samples[0]
    assert set(iface.model.traced_methods) == {"forward", "decode"}

    # everything is in `forward`: the loop is packed nodes, and the opened first
    # step has the GPT's layers in it to bend
    from torchbend.tracing.loop import is_loop_node
    graph = iface.model.graph(fn="forward")
    assert any(is_loop_node(n) for n in graph.nodes)
    assert sum(n.name.startswith("native_layer_norm") for n in graph.nodes) > 60

    # the main blocks' outputs are marked, not scattered across separate graphs
    aliases = iface.model.aliases(fn="forward")
    assert set(aliases) == {"conditioning", "voice", "tokens", "latents", "speech"}
    joints = iface.spec.method("forward").retained(graph)
    assert set(joints) == {"conditioning", "voice", "tokens", "latents", "speech"}
    node_names = {n.name for n in graph.nodes}
    assert all(name in node_names for name in joints.values())
    # bendable by alias, the whole point of tagging them
    by_alias = iface.model.activations("#latents", fn="forward")
    assert set(by_alias) == {joints["latents"]}

    # the blocks are distinct top-level modules, and each block's weights sit
    # under the same name as its computation -- a submodule reachable from two
    # attributes gets its weights named by the first, its ops by the call site,
    # and draws as two boxes
    acts = iface.model.activations("?.*", fn="forward")
    tops = {(getattr(p, "module_path", None) or "").split(".")[0] for p in acts.values()}
    assert tops - {""} == {"gpt", "style_encoder", "decoder"}
    style_weights = [str(n.target) for n in graph.nodes
                     if n.op == "get_attr" and "conditioning_encoder" in str(n.target)]
    assert style_weights and all(t.startswith("style_encoder.") for t in style_weights)

    # same seed, same speech as XTTS itself
    audio = iface.speak(text="the graph", seed=1)
    xtts = iface._full
    cond, g = xtts.get_conditioning_latents(audio_path=[samples[0]])
    torch.manual_seed(1)
    with torch.no_grad():
        stock = torch.as_tensor(xtts.inference(
            text="the graph", language="en", gpt_cond_latent=cond, speaker_embedding=g,
            temperature=iface.temperature)["wav"]).reshape(1, -1)
    assert audio.shape == stock.shape
    assert torch.allclose(audio, stock, atol=1e-3)

    # the bench route: each input on its own -- a caption, a style reference,
    # a voice reference (empty: the `reference` option), a temperature
    modes = iface.spec.method("forward").input_modes()
    assert {k: v["fills"] for k, v in modes.items()} == {
        "text_ids": ["text_ids"], "mel": ["mel"], "audio": ["audio"]}
    assert modes["mel"]["default"] == samples[0], "reference inputs default to the option"
    filled = {}
    filled.update(iface.spec.method("forward").encode("text_ids", "the graph"))
    filled.update(iface.spec.method("forward").encode("mel", ""))
    filled.update(iface.spec.method("forward").encode("audio", ""))
    filled["temperature"] = torch.tensor([iface.temperature])
    torch.manual_seed(1)
    with torch.no_grad():
        wav = iface.model.forward(**filled)[0].reshape(1, -1)
    assert torch.allclose(wav[:, :audio.shape[1]], audio, atol=1e-5)
    assert wav[:, audio.shape[1]:].abs().max() == 0, "the tail after the stop is silent"

    # the reference inputs are audio modes: a loaded file (the bench hands
    # over (waveform, rate)) encodes exactly like typing its path
    assert modes["mel"]["type"] == modes["audio"]["type"] == "audio"
    import soundfile as sf
    data, sr = sf.read(samples[0], dtype="float32", always_2d=True)
    uploaded = (torch.from_numpy(data.T.copy()), sr)
    for name in ("mel", "audio"):
        by_path = iface.spec.method("forward").encode(name, samples[0])[name]
        by_file = iface.spec.method("forward").encode(name, uploaded)[name]
        assert torch.equal(by_path, by_file)

    # temperature is the graph's input in full mode, not an option too
    assert "temperature" not in iface.spec.options

    # and without any reference, the reference inputs say so
    iface.reference = ""
    with pytest.raises(BendingXTTSException, match="reference"):
        iface.spec.method("forward").encode("mel", "")
