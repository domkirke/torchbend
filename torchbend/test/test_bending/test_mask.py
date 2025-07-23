import sys, os
import torch, torch.nn as nn
import pytest
import torchbend as tb
from functools import partial

testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)
from test_modules import modules_to_test, ModuleTestConfig



@pytest.mark.parametrize('cb_class', [tb.Mask, partial(tb.Mask, dim=-1), tb.OrderedMask, partial(tb.OrderedMask, dim=-1)])
@pytest.mark.parametrize('module_config', modules_to_test)
def test_mask_weight(cb_class, module_config):
    torch.set_grad_enabled(False)
    mod = module_config.get_bended_module()

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



@pytest.mark.parametrize('cb_class', [tb.Mask, partial(tb.Mask, dim=-1), tb.OrderedMask, partial(tb.OrderedMask, dim=-1)])
@pytest.mark.parametrize('as_input', [True, False])
@pytest.mark.parametrize('module_config', modules_to_test)
def test_mask_activation(cb_class, module_config, as_input):
    torch.set_grad_enabled(False)
    mod = module_config.get_bended_module()

    for method, (args, kwargs, weight_targets, activation_targets) in module_config:
        if len(activation_targets) == 0: continue
        mod.reset()
        mod.trace(method, **kwargs)
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(mod, method)(*args, **kwargs)

        prob = tb.bending.BendingParameter('mask', 1., as_input=as_input)
        mask_callback = cb_class(prob=prob)

        kwargs_nomask = dict(kwargs)
        kwargs_masked = dict(kwargs)
        mod.bend(mask_callback, fn=method, *activation_targets)
        if as_input: 
            for i, a in enumerate(activation_targets):
                if len(activation_targets) == 1:
                    kwargs_nomask['mask'] = torch.Tensor([1.])
                    kwargs_masked['mask'] = torch.Tensor([0.])
                else:
                    kwargs_nomask['mask_%d'%i] = torch.Tensor([1.])
                    kwargs_masked['mask_%d'%i] = torch.Tensor([0.])
        out_nomask = getattr(mod, method)(*args, **kwargs_nomask)
        assert bool(tb.compare_outs(out_orig, out_nomask))

        prob.set_value(0.)
        out_masked = getattr(mod, method)(*args, **kwargs_masked)
        assert not bool(tb.compare_outs(out_orig, out_masked))


@pytest.mark.parametrize('cb_class', [tb.Mask, partial(tb.Mask, dim=-1), tb.OrderedMask, partial(tb.OrderedMask, dim=-1)])
@pytest.mark.parametrize('module_config', modules_to_test)
@pytest.mark.parametrize('as_controllable', [True, False])
@pytest.mark.parametrize('as_input', [True, False])
@pytest.mark.parametrize('jit', [True, False])
def test_mask_script(cb_class, module_config, jit, as_controllable, as_input):
    mod = module_config.get_bended_module()
    if not as_controllable and as_input: pytest.skip(reason="as_controllable must be True if as_input")
    for method, (args, kwargs, weight_targets, activation_targets) in module_config.scriptable():
        mod.reset()
        mod.trace(method, **kwargs)
        out_orig = getattr(mod, method)(*args, **kwargs)

        if as_controllable:
            prob = tb.bending.BendingParameter('mask', 1., as_input=as_input)
        else:
            prob = 1.
        mask_callback = cb_class(prob=prob)
        mask_callback.get('prob')
        if len(weight_targets) > 0: mod.bend(mask_callback, *weight_targets, bend_graph=False)
        if len(activation_targets) > 0: mod.bend(mask_callback, *activation_targets, bend_param=False)

        kwargs_nomask = dict(kwargs)
        kwargs_masked = dict(kwargs)
        if as_input and len(activation_targets) > 0: 
            targets = list(mod.activations(*activation_targets, fn=method, with_bended=False).keys())
            for i, a in enumerate(targets):
                if len(targets) == 1:
                    kwargs_nomask['mask'] = torch.Tensor([1.])
                    kwargs_masked['mask'] = torch.Tensor([0.])
                else:
                    kwargs_nomask['mask_%d'%i] = torch.Tensor([1.])
                    kwargs_masked['mask_%d'%i] = torch.Tensor([0.])

        mod_scripted = mod.script(script=jit)
        if len(weight_targets) > 0 and as_controllable: mod_scripted._set_bending_control('mask', 1.)
        out_scripted = getattr(mod_scripted, method)(*args, **kwargs_nomask)
        assert bool(tb.compare_outs(out_orig, out_scripted))

        if as_controllable:
            mod_scripted._set_bending_control('mask', 0.)
            out_scripted = getattr(mod_scripted, method)(*args, **kwargs_masked)
            assert not bool(tb.compare_outs(out_orig, out_scripted))


@pytest.mark.parametrize('cb_class', [partial(tb.ThresholdActivation, invert=False)])
@pytest.mark.parametrize('module_config', modules_to_test)
@pytest.mark.parametrize('as_input', [True, False])
@pytest.mark.parametrize('jit', [True, False])
def test_threshold_activation(cb_class, module_config, jit, as_input): 
    mod = module_config.get_bended_module()

    for method, (args, kwargs, _, activation_targets) in module_config.scriptable():
        mod.reset()
        mod.trace(method, **kwargs)
        out_orig = getattr(mod, method)(*args, **kwargs)

        threshold = tb.bending.BendingParameter('threshold', 1., as_input=as_input)
        mask_callback = cb_class(threshold=threshold)
        if len(activation_targets) > 0: mod.bend(mask_callback, *activation_targets, bend_param=False)

        kwargs_nomask = dict(kwargs)
        kwargs_masked = dict(kwargs)
        if as_input and len(activation_targets) > 0: 
            targets = list(mod.activations(*activation_targets, fn=method, with_bended=False).keys())
            for i, a in enumerate(targets):
                if len(targets) == 1:
                    kwargs_nomask['threshold'] = torch.Tensor([1.])
                else:
                    kwargs_nomask['threshold_%d'%i] = torch.Tensor([1.])

        mod_scripted = mod.script(script=jit)
        out_scripted = getattr(mod_scripted, method)(*args, **kwargs_nomask)
        assert bool(tb.compare_outs(out_orig, out_scripted))

        out_scripted = getattr(mod_scripted, method)(*args, **kwargs_masked)
        assert not bool(tb.compare_outs(out_orig, out_scripted))


