import torch
import torchbend as tb
import pytest
from test_modules.module_test_modules import modules_to_test


@pytest.mark.parametrize("module_config", modules_to_test)
def test_bending_parameters_weights(module_config):
    module, bended_module = module_config.get_modules()
    zero_callback = tb.Mask(tb.BendingParameter("mask", 1.))

    for method in module_config.get_methods():
        args, kwargs, weights, _ = module_config.get_method_args(method)
        out_orig = getattr(module, method)(*args, **kwargs)

        bended_module.reset()
        bended_module.trace(method, **kwargs)
        bended_module.bend(zero_callback, *weights, fn=method, verbose=True)

        bended_module.update("mask", 0.)
        out_bended = getattr(bended_module, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(out_orig, out_bended))

        bended_module.update("mask", 1.)
        out_unbended = getattr(bended_module, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_unbended))


@pytest.mark.parametrize("module_config", modules_to_test)
def test_bending_parameters_activations(module_config):
    module, bended_module = module_config.get_modules()

    for method in module_config.get_methods():
        zero_callback = tb.Mask(prob=tb.BendingParameter("param_1", 1.))
        args, kwargs, _, acts = module_config.get_method_args(method)
        out_orig = getattr(module, method)(*args, **kwargs)

        bended_module.reset()
        bended_module.trace(method, **kwargs)
        bended_module.bend(zero_callback, *acts, fn=method, verbose=True)

        bended_module.update("param_1", 0.)
        out_bended = getattr(bended_module, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(out_orig, out_bended))

        bended_module.update("param_1", 1.)
        out_unbended = getattr(bended_module, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_unbended)) 

