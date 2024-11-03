import sys, os
import torch, torch.nn as nn
import pytest
import torchbend as tb
from torchbend.tracing.utils import get_kwargs_from_gm

testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)
from test_modules.module_test_modules import modules_to_test, ModuleTestConfig


@pytest.mark.parametrize('module_config', modules_to_test)
def test_interpolation(module_config, n=8):
    mod = module_config.get_bended_module()
    for method, (args, kwargs, weight_targets, activation_targets) in module_config:    
        mod.reset()
        mod.trace(func=method, **kwargs)
        args, kwargs, _, _ = module_config.get_method_args(method)
        cb = tb.InterpolateActivation()
        for target in activation_targets:
            mod.bend(cb, target)
            outs = mod.get_activations(target, **kwargs, fn=method)
            gm = mod.bend_activation_as_input(target, fn=method)
            # unbatched
            interp = torch.randn(outs[target].shape[0])
            outs_interpolated = gm(**get_kwargs_from_gm(gm, **kwargs, **outs), interp_weights=interp)
            # batched
            interp = torch.randn(4, outs[target].shape[0])
            outs_interpolated = gm(**get_kwargs_from_gm(gm, **kwargs, **outs), interp_weights=interp)
            
