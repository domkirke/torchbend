"""BLIP end-to-end pipeline: the decode loop packed into the graph.

The claim under test is that how much of the decoder the graph holds is a
*trace-time* choice with no effect on what the model says: unrolled, or packed
at any granularity, the caption must be the same token for token. That is not
obvious, because packing required rewriting the decode loop to write into a
preallocated buffer instead of growing one — see `BlipCaptionPipeline`.

Skipped when transformers or the BLIP weights are unavailable.
"""
import pytest
import torch

BLIP_AVAILABLE = False
try:
    from torchbend.interfaces.blip import BendedBlipPipeline
    BLIP_AVAILABLE = True
except ModuleNotFoundError:
    BLIP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not BLIP_AVAILABLE,
                                 reason="transformers / BLIP not available")

N_TOKENS = 6


@pytest.fixture(scope="module")
def image():
    torch.manual_seed(7)
    return torch.rand(1, 3, 384, 384)


def _pipeline(pack, image):
    try:
        m = BendedBlipPipeline(max_new_tokens=N_TOKENS, pack=pack)
    except Exception as exc:                       # weights not downloaded
        pytest.skip("could not load BLIP: %s" % exc)
    m.on_inputs("forward", {"pixel_values": image[:, :, :m.image_size, :m.image_size]})
    return m


@pytest.mark.parametrize("pack", [1, 3, N_TOKENS])
def test_packed_decode_matches_unrolled(pack, image):
    """Packing is a graph-shape decision, not a behavioural one."""
    reference = _pipeline(0, image)
    expected = reference.caption_tokens()

    packed = _pipeline(pack, image)
    ids = packed.caption_tokens()
    assert isinstance(ids, torch.Tensor) and not ids.is_meta
    assert torch.equal(ids, expected), (
        "pack=%d changed the caption: %r vs %r"
        % (pack, packed.caption(), reference.caption()))


def test_packing_shrinks_the_graph(image):
    """The decoder leaves the graph; the vision tower stays (and stays bendable)."""
    n = lambda m: len(list(m.model.graph(fn="forward", bended=True).nodes))
    unrolled = n(_pipeline(0, image))
    packed = n(_pipeline(N_TOKENS, image))
    assert packed < unrolled / 5


def test_bending_reaches_the_packed_decoder(image):
    """A weight inside the opaque packed loop still changes what BLIP says.

    Zeroing the weight outright rather than thinning it: a caption is a greedy
    argmax, so it is quite capable of surviving a partial mask unchanged, and
    that would make this test flaky rather than strict.
    """
    from torchbend.bending import Mask
    m = _pipeline(2, image)
    before = m.caption()
    weight = [w for w in m.model.weight_names
              if "text_decoder" in w and w.endswith("weight")][3]
    m.model.bend(Mask(prob=0.0), weight)
    assert m.caption() != before
