"""The graph editor's device switch (`BendingSession.set_device`), and the
`devices` module it and play mode share.

Only CPU is guaranteed to exist in CI, so the device-*move* behaviour is
tested against CPU→CPU (a no-op that still exercises the code path) and the
device-*compat* / cache-*invalidation* behaviour is tested independently of
which accelerators happen to be present.
"""
import torch
import torch.nn as nn

import torchbend as tb
from torchbend.interfaces.base import Interface
from torchbend.ui.graph_viewer.bending_session import BendingSession
from torchbend.ui.graph_viewer.devices import device_compat_for_methods


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)

    def forward(self, x):
        return self.a(x)


def _bm():
    bm = tb.BendedModule(_Tiny())
    bm.trace("forward", x=torch.randn(1, 4))
    return bm


def test_set_device_moves_the_module():
    bm = _bm()
    session = BendingSession()
    session.set_device(bm, "cpu")   # a no-op move, but exercises the real path
    assert session.device == "cpu"
    assert next(bm._module.parameters()).device.type == "cpu"


def test_unavailable_device_is_refused():
    bm = _bm()
    session = BendingSession()
    try:
        session.set_device(bm, "not-a-real-device")
        assert False, "should have raised"
    except ValueError as exc:
        assert "not-a-real-device" in str(exc)
    assert session.device == "cpu"   # unchanged


def test_setting_the_same_device_again_is_a_cheap_no_op():
    """Nothing moved, so nothing already cached needs to go."""
    bm = _bm()
    session = BendingSession()
    cache = session._get_cache()
    session.get_cached_activations(bm, "forward", {"x": torch.randn(1, 4)},
                                   target_nodes=["addmm"])
    assert cache._entries
    session.set_device(bm, "cpu")   # already cpu
    assert cache._entries           # untouched


def test_switching_to_a_real_other_device_clears_the_activation_cache():
    import pytest
    from torchbend.ui.graph_viewer.devices import available_devices
    other = next((d for d in available_devices() if d != "cpu"), None)
    if other is None:
        pytest.skip("no non-CPU device available on this machine")
    bm = _bm()
    session = BendingSession()
    cache = session._get_cache()
    session.get_cached_activations(bm, "forward", {"x": torch.randn(1, 4)},
                                   target_nodes=["addmm"])
    assert cache._entries and cache.used_bytes() > 0
    session.set_device(bm, other)
    assert cache._entries == {}
    session.restore_device(bm)


def test_restore_device_is_a_noop_at_cpu():
    bm = _bm()
    session = BendingSession()
    session.restore_device(bm)   # never moved -- must not raise
    assert session.device == "cpu"


def test_device_compat_for_methods_defaults_to_true_without_an_interface():
    compat = device_compat_for_methods(None, ["forward", "decode"])
    assert compat["forward"]["cpu"] is True
    assert compat["decode"]["cpu"] is True


def test_device_compat_for_methods_reads_the_interface():
    from torchbend.interfaces.spec import Method

    class _Picky(Interface):
        methods = {"forward": Method(devices={"mps": False})}
        def __init__(self):
            super().__init__(_Tiny())

    compat = device_compat_for_methods(_Picky().spec, ["forward", "decode"])
    assert compat["forward"]["cpu"] is True
    assert compat["decode"]["cpu"] is True
    if "mps" in compat["forward"]:
        assert compat["forward"]["mps"] is False
        assert compat["decode"]["mps"] is True   # undeclared -> default True
