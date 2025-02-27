import torch
import torchbend as tb
from torchbend.utils import checktuple
import pytest
import sys, os
from types import MethodType

testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)

from test_modules import scriptable_modules_to_test 


def get_scriptable_methods(module):
    methods = []
    if isinstance(getattr(module, "forward", None), MethodType):
        methods.append("forward")
    for m in dir(module):
        obj = getattr(module, m)
        if isinstance(obj, MethodType) and getattr(obj, "_torchscript_modifier", None) == torch._jit_internal.FunctionModifiers.EXPORT:
            methods.append(m)
    return methods


@pytest.mark.parametrize("module_config", scriptable_modules_to_test)
def test_scripting(module_config):
    """test if module (without bending) is scripted"""
    module = module_config.get_module()
    scriptable = module.script()
    scripted = torch.jit.script(scriptable)
    for method in get_scriptable_methods(scriptable):
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(module, method)(*args, **kwargs)
        out_scripted = getattr(scripted, method)(*args, **kwargs)
        assert tb.compare_outs(out_orig, out_scripted)


@pytest.mark.parametrize("module_config", scriptable_modules_to_test)
@pytest.mark.parametrize("nntilde", [False])
def test_bendable_scripting(module_config, nntilde: bool):
    """test if bended modules can be scripted"""
    module, bended = module_config.get_modules()
    # trace scriptable methods
    for method in get_scriptable_methods(module):
        args, kwargs, _, _ = module_config.get_method_args(method)
        bended.trace(fn=method, *args, **kwargs)
    # script module
    if nntilde: 
        scripted = bended.nntilde()
    else:
        scripted = bended.script()
    # compare outs
    for method in get_scriptable_methods(module):
        out_orig = getattr(module, method)(*args, **kwargs)
        out_scripted = getattr(scripted, method)(*args, **kwargs)
        assert tb.compare_outs(out_orig, out_scripted)


@pytest.mark.parametrize("module_config", scriptable_modules_to_test)
@pytest.mark.parametrize("nntilde", [False])
def test_bended_scripting(module_config, nntilde: bool):
    module, bended = module_config.get_modules()
    # trace scriptable methods
    for method in get_scriptable_methods(module):
        args, kwargs, bended_weights, bended_acts = module_config.get_method_args(method)
        bended.trace(fn=method, *args, **kwargs)
        # bend module
        cb = tb.Mask(0.)
        bended.bend(cb, fn=method, *bended_weights, *bended_acts)

    # script module
    if nntilde: 
        scripted = bended.nntilde()
    else:
        scripted = bended.script()
    
    # compare outs
    for method in get_scriptable_methods(module):
        out_orig = getattr(module, method)(*args, **kwargs)
        out_scripted = getattr(scripted, method)(*args, **kwargs)
        assert not tb.compare_outs(out_orig, out_scripted)


@pytest.mark.parametrize("module_config", scriptable_modules_to_test)
def test_controlled_bended_scripting(module_config):
    module, bended = module_config.get_modules()
    param = tb.BendingParameter("mask", 1.)

    # trace scriptable methods
    for method in get_scriptable_methods(module):
        args, kwargs, bended_weights, bended_acts = module_config.get_method_args(method)
        bended.trace(fn=method, *args, **kwargs)
        # bend module
        cb = tb.Mask(param)
        bended.bend(cb, *bended_weights, *bended_acts)

    def check_bending_param(param, val, tol=1.e-5):
        return  (param > val - tol) and (param < val + tol)

    # script module
    scripted = bended.script()
    # compare outs

    torch.set_grad_enabled(False)
    for method in get_scriptable_methods(module):
        # compare with unmasked weights & activations
        param.set_value(1.)
        scripted.set_mask(1.)
        assert check_bending_param(scripted.get_mask(), 1.)
        out_orig = getattr(module, method)(*args, **kwargs)
        out_scripted = getattr(scripted, method)(*args, **kwargs)
        assert tb.compare_outs(out_orig, out_scripted)

        # compare with masked weights & activations
        param.set_value(0.)
        scripted.set_mask(0.)
        assert check_bending_param(scripted.get_mask(), 0.)
        out_orig = getattr(module, method)(*args, **kwargs)
        out_scripted = getattr(scripted, method)(*args, **kwargs)
        assert not tb.compare_outs(out_orig, out_scripted)


@pytest.mark.parametrize("module_config", scriptable_modules_to_test) 
@pytest.mark.parametrize("jit", [True, False]) 
def test_scripting_graph_split(module_config, jit):
    module, bended = module_config.get_modules()
    for method in get_scriptable_methods(module):
        args, kwargs, _, bended_acts = module_config.get_method_args(method)
        bended.trace(**kwargs, fn=method)
        for activation in bended_acts:
            out_act = bended.get_activations(activation, **kwargs, fn=method, _save_as_method=f"{method}_get_{activation}")
            out = bended.from_activations(activation, **out_act, fn=method, _save_as_method=f"{method}_from_{activation}")
    
    scripted = bended.script(script=jit)
    for method in get_scriptable_methods(module):
        for activation in bended_acts:
            args, kwargs, _, bended_acts = module_config.get_method_args(method)
            out_act = checktuple(getattr(scripted, f"{method}_get_{activation}")(**kwargs))
            out = getattr(scripted, f"{method}_from_{activation}")(*out_act)


