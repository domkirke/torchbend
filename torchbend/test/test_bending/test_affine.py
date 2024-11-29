import os, sys
import torch, torch.nn as nn
import pytest
import torchbend as tb


testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)
from test_modules import modules_to_test, ModuleTestConfig


@pytest.mark.parametrize('cb_class', [tb.Bias, tb.Affine, tb.Scale])
@pytest.mark.parametrize('module_config', modules_to_test)
def test_affine_activation(cb_class, module_config):
    mod = module_config.get_bended_module()

    for method, (args, kwargs, weight_targets, activation_targets) in module_config:
        mod.reset()
        mod.trace(method, **kwargs)
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(mod, method)(*args, **kwargs)

        scale = tb.bending.BendingParameter('scale', 1.)
        bias = tb.bending.BendingParameter('bias', 0.)
        cb_kwargs = {}
        if "scale" in cb_class.controllable_params:
            cb_kwargs['scale'] = scale
        if "bias" in cb_class.controllable_params:
            cb_kwargs['bias'] = bias
        affine_callback = cb_class(**cb_kwargs)

        mod.bend(affine_callback, fn=method, *activation_targets)
        out_nomask = getattr(mod, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_nomask))

        scale.set_value(0.)
        bias.set_value(1.)
        out_masked = getattr(mod, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(out_orig, out_masked))


@pytest.mark.parametrize('cb_class', [tb.Bias, tb.Affine, tb.Scale])
@pytest.mark.parametrize('module_config', modules_to_test)
def test_affine_weights(cb_class, module_config):
    mod = module_config.get_bended_module()

    for method, (args, kwargs, weight_targets, activation_targets) in module_config:
        mod.reset()
        mod.trace(method, **kwargs)
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(mod, method)(*args, **kwargs)

        scale = tb.bending.BendingParameter('scale', 1.)
        bias = tb.bending.BendingParameter('bias', 0.)
        cb_kwargs = {}
        if "scale" in cb_class.controllable_params:
            cb_kwargs['scale'] = scale
        if "bias" in cb_class.controllable_params:
            cb_kwargs['bias'] = bias
        affine_callback = cb_class(**cb_kwargs)

        mod.bend(affine_callback, *weight_targets)
        out_nomask = getattr(mod, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_nomask))

        scale.set_value(0.)
        bias.set_value(1.)
        out_masked = getattr(mod, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(out_orig, out_masked))



@pytest.mark.parametrize('cb_class', [tb.Bias, tb.Affine, tb.Scale])
@pytest.mark.parametrize('module_config', modules_to_test)
def test_affine_script(cb_class, module_config):
    mod = module_config.get_bended_module()

    for method, (args, kwargs, weight_targets, activation_targets) in module_config:
        mod.reset()
        mod.trace(method, **kwargs)
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(mod, method)(*args, **kwargs)

        scale = tb.bending.BendingParameter('scale', 1.)
        bias = tb.bending.BendingParameter('bias', 0.)
        cb_kwargs = {}
        if "scale" in cb_class.controllable_params:
            cb_kwargs['scale'] = scale
        if "bias" in cb_class.controllable_params:
            cb_kwargs['bias'] = bias
        affine_callback = cb_class(**cb_kwargs)

        mod.bend(affine_callback, *weight_targets)
        scripted_model = mod.script()

        out_nomask = getattr(scripted_model, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_nomask))


"""
@pytest.mark.parametrize('cb_class', [tb.Scale, tb.Bias, tb.Affine])
@pytest.mark.parametrize('module_class', [Foo])
def test_affine_activation(cb_class, module_class):
    mod = tb.BendedModule(module_class())
    mod_input = torch.randn(module_class._input_shape)
    mod.trace(x=mod_input)
    out_orig = mod.forward(mod_input)

    scale = tb.bending.BendingParameter('scale', 1.)
    bias = tb.bending.BendingParameter('bias', 0.)
    kwargs = {}
    if "scale" in cb_class.controllable_params:
        kwargs['scale'] = scale
    if "bias" in cb_class.controllable_params:
        kwargs['bias'] = bias
    affine_callback = cb_class(**kwargs)

    mod.bend(affine_callback, *mod.activations())
    out_nobend = mod.forward(mod_input)
    assert bool(tb.compare_outs(out_orig, out_nobend))

    scale.set_value(0.)
    bias.set_value(1.)
    out_bend = mod.forward(mod_input)
    assert not bool(tb.compare_outs(out_orig, out_bend))

@pytest.mark.parametrize('cb_class', [tb.Scale, tb.Bias, tb.Affine])
@pytest.mark.parametrize('module_class', [Foo])
def test_affine_script(cb_class, module_class):
    mod = tb.BendedModule(module_class())
    mod_input = torch.randn(module_class._input_shape)
    mod.trace(x=mod_input)
    out_orig = mod.forward(mod_input)

    scale = tb.bending.BendingParameter('scale', 1.)
    bias = tb.bending.BendingParameter('bias', 0.)
    kwargs = {}
    if "scale" in cb_class.controllable_params:
        kwargs['scale'] = scale
    if "bias" in cb_class.controllable_params:
        kwargs['bias'] = bias
    affine_callback = cb_class(**kwargs)
    mod.bend(affine_callback, *mod.weights)
    
    mod_scripted = mod.script()
    out_scripted = mod_scripted.forward(mod_input)
    assert bool(tb.compare_outs(out_orig, out_scripted))



"""