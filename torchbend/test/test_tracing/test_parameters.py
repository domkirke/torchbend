import torch
import torchbend as tb
import pytest
from torchbend.bending.parameter import BendingParameterException
from test_modules.module_test_modules import modules_to_test, scriptable_modules_to_test 

@pytest.mark.parametrize("module_config", scriptable_modules_to_test)
def test_int_parameter(module_config):
    module, bended_module = module_config.get_modules()

    def _test_parameter(bended_module, method, bias, ok=[], not_ok=[]):
        bended_module.reset()
        assert int(bias) == bias.get_value()
        assert isinstance(bias.get_python_value(), int)
        bias_callback = tb.Bias(bias=bias)
        args, kwargs, weights, acts = module_config.get_method_args(method)
        bended_module.bend(bias_callback, *weights, *acts, fn=method)
        assert len(ok) + len(not_ok) > 0, "at least one value must be given in either ok or not_ok"
        
        scripted_module = bended_module.script(script=False) 

        # try out ok values
        for value in ok:
            bias.set_value(value)
            getattr(bended_module, method)(*args, **kwargs)
            scripted_module.set_bias(value)

        # try out not ok values
        for value in not_ok:
            try: 
                bias.set_value(value)
                assert False, "setting value %s in parameter %s should raise an exception"
            except BendingParameterException:
                pass

            try: 
                scripted_module.set_bias(value)
                assert False, "setting value %s in parameter %s should raise an exception"
            except BendingParameterException:
                pass

    for method in module_config.get_methods():

        bias = tb.BendingParameter("bias", 0)
        _test_parameter(bended_module, method, bias, ok=[0])

        bias = tb.BendingParameter("bias", 0, range=[0, None])
        _test_parameter(bended_module, method, bias, ok=[0, 3], not_ok=[-1])

        bias = tb.BendingParameter("bias", 0, range=[None, 3])
        _test_parameter(bended_module, method, bias, ok=[0, 3], not_ok=[5])

        bias = tb.BendingParameter("bias", 0, range=[0, 3])
        _test_parameter(bended_module, method, bias, ok=[0, 3], not_ok=[-2, 5])



@pytest.mark.parametrize("module_config", scriptable_modules_to_test)
def test_float_parameter(module_config):
    module, bended_module = module_config.get_modules()

    def _test_parameter(bended_module, method, bias, ok=[], not_ok=[]):
        bended_module.reset()
        assert int(bias) == bias.get_value()
        assert isinstance(bias.get_python_value(), float)
        bias_callback = tb.Normal(std=bias)
        args, kwargs, weights, acts = module_config.get_method_args(method)
        bended_module.bend(bias_callback, *weights, *acts, fn=method)
        assert len(ok) + len(not_ok) > 0, "at least one value must be given in either ok or not_ok"
        
        scripted_module = bended_module.script(script=False) 

        # try out ok values
        for value in ok:
            bias.set_value(value)
            getattr(bended_module, method)(*args, **kwargs)
            scripted_module.set_bias(value)

        # try out not ok values
        for value in not_ok:
            try: 
                bias.set_value(value)
                assert False, "setting value %s in parameter %s should raise an exception"
            except BendingParameterException:
                pass

            try: 
                scripted_module.set_bias(value)
                assert False, "setting value %s in parameter %s should raise an exception"
            except BendingParameterException:
                pass

    for method in module_config.get_methods():

        bias = tb.BendingParameter("bias", 0.)
        _test_parameter(bended_module, method, bias, ok=[0])

        bias = tb.BendingParameter("bias", 0., range=[0, None])
        _test_parameter(bended_module, method, bias, ok=[0, 3], not_ok=[-1])

        bias = tb.BendingParameter("bias", 0., range=[None, 3])
        _test_parameter(bended_module, method, bias, ok=[0, 3], not_ok=[5])

        bias = tb.BendingParameter("bias", 0., range=[0, 3])
        _test_parameter(bended_module, method, bias, ok=[0, 3], not_ok=[-2, 5])


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

