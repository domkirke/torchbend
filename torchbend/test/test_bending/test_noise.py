import sys, os
import torch, torch.nn as nn
import pytest
import torchbend as tb
from functools import partial

testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)
from test_modules import modules_to_test, ModuleTestConfig


@pytest.mark.parametrize('cb_class', [tb.Normal])
@pytest.mark.parametrize('module_config', modules_to_test)
def test_mask_weight(cb_class, module_config):
    mod = module_config.get_bended_module()

    for method, (args, kwargs, weight_targets, activation_targets) in module_config:
        mod.reset()
        out_orig = getattr(mod, method)(*args, **kwargs)

        std = tb.bending.BendingParameter('scale', 0.)
        mask_callback = cb_class(std=std)

        mod.bend(mask_callback, *weight_targets)
        out_nomask = getattr(mod, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_nomask))

        std.set_value(1.)
        out_masked = getattr(mod, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(out_orig, out_masked))


@pytest.mark.parametrize('cb_class', [tb.Normal])
@pytest.mark.parametrize('module_config', modules_to_test)
def test_mask_activations(cb_class, module_config):
    mod = module_config.get_bended_module()

    for method, (args, kwargs, weight_targets, activation_targets) in module_config:
        mod.reset()
        mod.trace(method, **kwargs)
        out_orig = getattr(mod, method)(*args, **kwargs)

        std = tb.bending.BendingParameter('scale', 0.)
        mask_callback = cb_class(std=std)

        mod.bend(mask_callback, *activation_targets)
        out_nomask = getattr(mod, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_nomask))

        std.set_value(1.)
        out_masked = getattr(mod, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(out_orig, out_masked))


@pytest.mark.parametrize('cb_class', [tb.Normal])
@pytest.mark.parametrize('module_config', modules_to_test)
def test_mask_script(cb_class, module_config):
    mod = module_config.get_bended_module()

    for method, (args, kwargs, weight_targets, activation_targets) in module_config:
        mod.reset()
        mod.trace(method, **kwargs)
        args, kwargs, _, _ = module_config.get_method_args(method)

        std = tb.bending.BendingParameter('scale', 1.)
        mask_callback = cb_class(std=std)
        mod.bend(mask_callback, *activation_targets)
        out_orig = getattr(mod, method)(*args, **kwargs)
        
        scripted = mod.script(script=True)
        out_scripted = getattr(scripted, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_scripted))