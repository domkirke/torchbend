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

        # # single activations
        for target in activation_targets:
            cb = tb.InterpolateActivation()
            mod.reset()
            mod.bend(cb, target)
            outs = mod.get_activations(target, **kwargs, fn=method, _filter_bended=True)
            # unbatched
            interp = torch.randn(outs[target].shape[0])
            outs_interpolated = mod.from_activations(*activation_targets, fn=method, **kwargs, **outs, interp_weights=interp)
            # batched
            interp = torch.randn(4, outs[target].shape[0])
            outs_interpolated = mod.from_activations(*activation_targets, fn=method, **kwargs, **outs, interp_weights=interp)

        # full activations
        if len(activation_targets) > 1:
            mod.reset()
            cb = tb.InterpolateActivation()
            mod.bend(cb, *activation_targets, fn=method)
            outs = mod.get_activations(*activation_targets, **kwargs, fn=method, _filter_bended=True)
            # unbatched
            interp = {f"{t}_interp_weights": torch.randn(outs[t].shape[0]) for t in activation_targets}
            outs_interpolated = mod.from_activations(*activation_targets, fn=method, **kwargs, **outs, **interp)
            # batched
            interp = {f"{t}_interp_weights": torch.randn(4, outs[t].shape[0]) for t in activation_targets}
            outs_interpolated = mod.from_activations(*activation_targets, fn=method, **kwargs, **outs, **interp)

            
