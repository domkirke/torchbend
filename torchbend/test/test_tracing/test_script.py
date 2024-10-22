import torch
import torchbend as tb
from torchbend.tracing import unmatching_ids, get_model_copy, clone_parameters
from torchbend.utils import checktuple, get_parameter
import pytest
import sys, os
from conftest import get_log_file, log_to_file
from types import MethodType

testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)

from test_modules.module_test_modules import scriptable_modules_to_test 


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
    module = module_config.get_module()
    scriptable = module.script()
    scripted = torch.jit.script(scriptable)
    for method in get_scriptable_methods(scriptable):
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(module, method)(*args, **kwargs)
        out_bended = getattr(scripted, method)(*args, **kwargs)
        assert tb.compare_outs(out_orig, out_bended)


@pytest.mark.parametrize("module_config", scriptable_modules_to_test)
def test_bended_scripting(module_config):
    module = module_config.get_module()
    scripted = module.script()
    scripted = torch.jit.script(scripted)
    for method in module_config.get_methods():
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(module, method)(*args, **kwargs)
        out_bended = getattr(scripted, method)(*args, **kwargs)
        assert tb.compare_outs(out_orig, out_bended)


        
    


