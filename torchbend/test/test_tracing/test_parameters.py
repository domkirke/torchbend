import torch
import torchbend as tb
import pytest
from torchbend.test.conftest import *


@pytest.mark.parametrize("module_config", modules_to_test)
def test_bending_parameters_weights(module_config):
    module = module_config.get_module()
    bended_module = tb.BendedModule(module)
    zero_callback = tb.Mask(tb.BendingParameter("mask", 1.))

    for method in module_config.get_methods():
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(module, method)(*args, **kwargs)

        bended_module.reset()
        bended_module.trace(method, **kwargs)
        bended_module.bend(zero_callback, *bended_module.weights, fn=method, verbose=True)

        bended_module.update("mask", 0.)
        out_bended = getattr(bended_module, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(out_orig, out_bended))

        bended_module.update("mask", 1.)
        out_unbended = getattr(bended_module, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_unbended))

@pytest.mark.parametrize("module_config", modules_to_test)
def test_bending_parameters_activations(module_config):
    module = module_config.get_module()
    bended_module = tb.BendedModule(module)

    for method in module_config.get_methods():
        zero_callback = tb.Mask(prob=tb.BendingParameter("param_1", 1.))
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(module, method)(*args, **kwargs)

        bended_module.reset()
        bended_module.trace(method, **kwargs)
        bended_module.bend(zero_callback, *bended_module.activations(method, op="call_module"), fn=method, verbose=True)

        bended_module.update("param_1", 0.)
        out_bended = getattr(bended_module, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(out_orig, out_bended))

        bended_module.update("param_1", 1.)
        out_unbended = getattr(bended_module, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_unbended)) 
