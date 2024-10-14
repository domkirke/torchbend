import sys, os
import torch, torch.nn as nn
import pytest
import torchbend as tb
from functools import partial

testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)
from test_modules.module_test_modules import modules_to_test, ModuleTestConfig





@pytest.mark.parametrize('cb_class', [tb.Mask, partial(tb.Mask, dim=-2), tb.OrderedMask, partial(tb.OrderedMask, dim=-2)])
@pytest.mark.parametrize('module_config', modules_to_test)
def test_mask_weight(cb_class, module_config):
    mod = module_config.get_module()
    mod = tb.BendedModule(mod)

    for method, (args, kwargs, weight_targets, activation_targets) in module_config:
        mod.reset()
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(mod, method)(*args, **kwargs)

        prob = tb.bending.BendingParameter('mask', 1.)
        mask_callback = cb_class(prob=prob)

        mod.bend(mask_callback, *weight_targets)
        out_nomask = getattr(mod, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_nomask))

        prob.set_value(0.)
        out_masked = getattr(mod, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(out_orig, out_masked))



@pytest.mark.parametrize('cb_class', [tb.Mask, partial(tb.Mask, dim=-2), tb.OrderedMask, partial(tb.OrderedMask, dim=-2)])
@pytest.mark.parametrize('module_config', modules_to_test)
def test_mask_activation(cb_class, module_config):
    mod = module_config.get_module()
    mod = tb.BendedModule(mod)

    for method, (args, kwargs, weight_targets, activation_targets) in module_config:
        mod.reset()
        mod.trace(method, **kwargs)
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(mod, method)(*args, **kwargs)

        prob = tb.bending.BendingParameter('mask', 1.)
        mask_callback = cb_class(prob=prob)

        mod.bend(mask_callback, fn=method, *activation_targets)
        out_nomask = getattr(mod, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_nomask))

        prob.set_value(0.)
        out_masked = getattr(mod, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(out_orig, out_masked))


@pytest.mark.parametrize('cb_class', [tb.Mask, partial(tb.Mask, dim=-2), tb.OrderedMask, partial(tb.OrderedMask, dim=-2)])
@pytest.mark.parametrize('module_config', modules_to_test)
def test_mask_script(cb_class, module_config):
    mod = module_config.get_module()
    mod = tb.BendedModule(mod)

    for method, (args, kwargs, weight_targets, activation_targets) in module_config.scriptable():
        mod.reset()
        mod.trace(method, **kwargs)
        out_orig = getattr(mod, method)(*args, **kwargs)

        prob = tb.bending.BendingParameter('mask', 1.)
        mask_callback = cb_class(prob=prob)

        mod.bend(mask_callback, *weight_targets, *activation_targets)
        mod_scripted = mod.script(script=False)
        mod_scripted._set_bending_control('mask', 1.)
        out_scripted = getattr(mod_scripted, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_scripted))

        mod_scripted._set_bending_control('mask', 0.)
        out_scripted = getattr(mod_scripted, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(out_orig, out_scripted))