import os, sys
import logging
import torch, torch.nn as nn
import pytest
import torchbend as tb


testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)
from test_modules import modules_to_test, scriptable_modules_to_test, ModuleTestConfig

@pytest.mark.parametrize('cb_class', [tb.Bias, tb.Affine, tb.Scale])
@pytest.mark.parametrize('module_config', modules_to_test)
@pytest.mark.parametrize('as_input', [False, True])
def test_affine_activation(cb_class, module_config, as_input, caplog):
    mod = module_config.get_bended_module()

    for method, (args, kwargs, weight_targets, activation_targets) in module_config:
        if len(activation_targets) == 0: continue
        mod.reset()
        mod.trace(method, **kwargs)
        out_orig = getattr(mod, method)(*args, **kwargs)

        scale = tb.bending.BendingParameter('scale', 1., as_input=as_input)
        bias = tb.bending.BendingParameter('bias', 0., as_input=as_input)
        cb_kwargs = {}
        if "scale" in cb_class.controllable_params:
            cb_kwargs['scale'] = scale
        if "bias" in cb_class.controllable_params:
            cb_kwargs['bias'] = bias
        affine_callback = cb_class(**cb_kwargs)

        mod.bend(affine_callback, fn=method, *activation_targets)

        # create additional inputs if as_input
        if as_input: 
            bended_params = affine_callback._bending_shapes
            if "scale" in cb_class.controllable_params:
                if len(bended_params) == 1: kwargs['scale'] = torch.ones(mod.activation_shape(activation_targets[0]))
                if len(bended_params) > 1: [kwargs.update({'scale_%d'%i: torch.ones(mod.activation_shape(activation_targets[i]))}) for i in range(len(bended_params))]
            if "bias" in cb_class.controllable_params:
                if len(bended_params) == 1: kwargs['bias'] = torch.zeros(mod.activation_shape(activation_targets[0]))
                if len(bended_params) > 1: [kwargs.update({'bias_%d'%i: torch.zeros(mod.activation_shape(activation_targets[i]))}) for i in range(len(bended_params))]

        out_nomask = getattr(mod, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_nomask))

        if as_input:
            if "scale" in cb_class.controllable_params:
                if len(bended_params) == 1: kwargs['scale'] = torch.zeros(mod.activation_shape(activation_targets[0]))
                if len(bended_params) > 1: [kwargs.update({'scale_%d'%i: torch.zeros(mod.activation_shape(activation_targets[i]))}) for i in range(len(bended_params))]
            if "bias" in cb_class.controllable_params:
                if len(bended_params) == 1: kwargs['bias'] = torch.ones(mod.activation_shape(activation_targets[0]))
                if len(bended_params) > 1: [kwargs.update({'bias_%d'%i: torch.ones(mod.activation_shape(activation_targets[i]))}) for i in range(len(bended_params))]
        else:
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


#TODO why is the pytest.mark.parametrize on as_input provokes a torchscript error???
def affine_test_script(cb_class, module_config, as_input):

    mod = module_config.get_bended_module(trace=True)
    
    for method, (args, kwargs, weight_targets, activation_targets) in module_config:
        mod.reset()
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(mod, method)(*args, **kwargs)

        scale = tb.bending.BendingParameter('scale', 1., as_input=as_input)
        bias = tb.bending.BendingParameter('bias', 0., as_input=as_input)
        cb_kwargs = {}
        if "scale" in cb_class.controllable_params:
            cb_kwargs['scale'] = scale
        if "bias" in cb_class.controllable_params:
            cb_kwargs['bias'] = bias
        affine_callback = cb_class(**cb_kwargs)

        mod.bend(affine_callback, *weight_targets, *activation_targets)

        if as_input: 
            bended_params = affine_callback.bended_activations(fn=method)
            if "scale" in cb_class.controllable_params:
                if len(bended_params) == 1: kwargs['scale'] = torch.zeros(mod.activation_shape(activation_targets[0], fn=method))
                if len(bended_params) > 1: [kwargs.update({'scale_%d'%i: torch.zeros(mod.activation_shape(b, fn=method)) for i, b in enumerate(bended_params)})]
            if "bias" in cb_class.controllable_params:
                if len(bended_params) == 1: kwargs['bias'] = torch.ones(mod.activation_shape(activation_targets[0], fn=method))
                if len(bended_params) > 1: [kwargs.update({'bias_%d'%i: torch.ones(mod.activation_shape(b, fn=method)) for i, b in enumerate(bended_params)})]
        else:
            scale.set_value(0.)
            bias.set_value(1.)

        scripted_model = mod.script()
        # scripted_model = torch.jit.script(mod)

        out_nomask = getattr(scripted_model, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(out_orig, out_nomask))

@pytest.mark.parametrize('cb_class', [tb.Bias, tb.Affine, tb.Scale])
@pytest.mark.parametrize('module_config', scriptable_modules_to_test)
def test_affine_script(cb_class, module_config):
    affine_test_script(cb_class, module_config, False)

@pytest.mark.parametrize('cb_class', [tb.Bias, tb.Affine, tb.Scale])
@pytest.mark.parametrize('module_config', scriptable_modules_to_test)
def test_affine_script_asinput(cb_class, module_config):
    affine_test_script(cb_class, module_config, True)
