import torch
import copy
import torchbend as tb
import pytest
from torchbend.utils import PArgs
from torchbend.bending.parameter import BendingParameterException
from test_modules import modules_to_test, scriptable_modules_to_test 




@pytest.mark.parametrize("module_config", scriptable_modules_to_test)
def test_int_parameter(module_config):
    module, bended_module = module_config.get_modules(trace=True)

    def _test_parameter(bended_module, method, control_args, ok=[], not_ok=[]):
        bended_module.reset()

        # test with normal behaviour
        bias = tb.BendingParameter(*control_args, **control_args)
        # test with as_input=True, that should add the parameter in the graph's signature 
        control_args_input = control_args.copy()
        control_args_input[0] = control_args_input[0] + "_input"
        bias_input = tb.BendingParameter(*control_args_input, as_input=True, **control_args_input)
        assert int(bias) == bias.get_value()
        assert isinstance(bias.get_python_value(), int)

        bias_callback = tb.Bias(bias=bias)
        bias_callback_input = tb.Bias(bias=bias_input)
        args, kwargs, weights, acts = module_config.get_method_args(method)
        bended_module.bend(bias_callback, *weights, *acts, fn=method)
        bended_module.bend(bias_callback_input, *weights, *acts, fn=method)
        assert len(ok) + len(not_ok) > 0, "at least one value must be given in either ok or not_ok"
        
        scripted_module = bended_module.script(script=False) 

        # try out ok values
        inputs = bended_module.bend_graph(fn=method).inputs
        for value in ok:
            current_kwargs = dict(kwargs)
            for i in inputs: 
                if i.name not in current_kwargs:
                    current_kwargs[i.name] = value
            bias.set_value(value)
            getattr(bended_module, method)(*args, **current_kwargs)
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

        bias = PArgs("bias", 0)
        _test_parameter(bended_module, method, bias, ok=[0])

        bias = PArgs("bias", 0, range=[0, None])
        _test_parameter(bended_module, method, bias, ok=[0, 3], not_ok=[-1])

        bias = PArgs("bias", 0, range=[None, 3])
        _test_parameter(bended_module, method, bias, ok=[0, 3], not_ok=[5])

        bias = PArgs("bias", 0, range=[0, 3])
        _test_parameter(bended_module, method, bias, ok=[0, 3], not_ok=[-2, 5])




@pytest.mark.parametrize("module_config", scriptable_modules_to_test)
def test_float_parameter(module_config):
    module, bended_module = module_config.get_modules(trace=True)

    def _test_parameter(bended_module, method, control_args, ok=[], not_ok=[]):
        bended_module.reset()

        # test with normal behaviour
        bias = tb.BendingParameter(*control_args, **control_args)
        # test with as_input=True, that should add the parameter in the graph's signature 
        control_args_input = control_args.copy()
        control_args_input[0] = control_args_input[0] + "_input"
        bias_input = tb.BendingParameter(*control_args_input, as_input=True, **control_args_input)
        assert int(bias) == bias.get_value()
        assert isinstance(bias.get_python_value(), float)

        bias_callback = tb.Bias(bias=bias)
        bias_callback_input = tb.Bias(bias=bias_input)
        args, kwargs, weights, acts = module_config.get_method_args(method)
        bended_module.bend(bias_callback, *weights, *acts, fn=method)
        bended_module.bend(bias_callback_input, *weights, *acts, fn=method)
        assert len(ok) + len(not_ok) > 0, "at least one value must be given in either ok or not_ok"
        
        scripted_module = bended_module.script(script=False) 

        # try out ok values
        inputs = bended_module.bend_graph(fn=method).inputs
        for value in ok:
            current_kwargs = dict(kwargs)
            for i in inputs: 
                if i.name not in current_kwargs:
                    current_kwargs[i.name] = value
            bias.set_value(value)
            getattr(bended_module, method)(*args, **current_kwargs)
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

        bias = PArgs("bias", 0.)
        _test_parameter(bended_module, method, bias, ok=[0])

        bias = PArgs("bias", 0., range=[0, None])
        _test_parameter(bended_module, method, bias, ok=[0, 3], not_ok=[-1])

        bias = PArgs("bias", 0., range=[None, 3])
        _test_parameter(bended_module, method, bias, ok=[0, 3], not_ok=[5])

        bias = PArgs("bias", 0., range=[0, 3])
        _test_parameter(bended_module, method, bias, ok=[0, 3], not_ok=[-2, 5])



@pytest.mark.skip(reason="to program")
def test_bool_parameter(module_config):
    pass


@pytest.mark.skip(reason="to program")
def test_tensor_parameter(module_config):
    pass


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
        if len(acts) == 0: continue
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


@pytest.mark.parametrize("module_config", modules_to_test)
def test_parameters_as_inputs(module_config):
    module, bended_module = module_config.get_modules(trace=True)
    
    for method in module_config.get_methods(): 
        bended_module.reset()
        args, kwargs, weights, acts = module_config.get_method_args(method)
        if len(acts) == 0:
            pytest.skip(reason="no activation to bend")
        param = tb.BendingParameter(name="param", value=0., as_input=True)
        bias = tb.Bias(bias=param)

        # test non-scripted
        bended_module.bend(bias, *acts, fn=method, verbose=True)
        bended_module.bend(bias, *acts, fn=method, verbose=True)

        bended_idx = 0
        untouched_kwargs = dict(kwargs)
        touched_kwargs = dict(kwargs)
        for i in [0, 1]:
            for act in acts: 
                untouched_kwargs[f'{param.name}_{bended_idx}'] = torch.zeros(bended_module.activation_shape(act, symbolic=False))
                touched_kwargs[f'{param.name}_{bended_idx}'] = torch.ones(bended_module.activation_shape(act, symbolic=False))
                bended_idx += 1
        getattr(bended_module, method)(*args, **kwargs)

        # test scripted
        if module_config.is_scriptable:
            scripted_module = bended_module.script()
            out1 = getattr(scripted_module, method)(*args, **untouched_kwargs)
            out2 = getattr(scripted_module, method)(*args, **touched_kwargs)
            assert not tb.compare_outs(out1, out2)
            torch.jit.save(scripted_module, '/tmp/test.ts' )


