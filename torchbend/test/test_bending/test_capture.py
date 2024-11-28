import sys, os
import torch, torch.nn as nn
import pytest
import torchbend as tb
from functools import partial


testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)
from test_modules.module_test_modules import modules_to_test, ModuleTestConfig


@pytest.mark.parametrize('module_config', modules_to_test)
def test_capture(module_config, n = 4):
    for method, (args, kwargs, weight_targets, activation_targets) in module_config:
        mod = module_config.get_bended_module()
        mod.trace(fn=method, **kwargs)
        cb = tb.Capture()
        mod.bend(cb, *activation_targets)
        args, kwargs, _, _ = module_config.get_method_args(method)
        cb.capture()
        for _ in range(n):
            _ = getattr(mod, method)(*args, **kwargs)
        cb.stop()
        # pass through a second time
        cb.capture()
        for _ in range(n):
            _ = getattr(mod, method)(*args, **kwargs)
        cb.stop()
        assert list(cb.captures.keys()) == [f"{method}:{t}" for t in mod.bended_keys(fn=method)]


@pytest.mark.parametrize('module_config', modules_to_test)
def test_capture_env(module_config, n = 4):
    for method, (args, kwargs, weight_targets, activation_targets) in module_config:
        mod = module_config.get_bended_module()
        mod.trace(fn=method, **kwargs)
        cb = tb.Capture()
        cb2 = tb.Capture()
        mod.bend(cb, *activation_targets)
        mod.bend(cb2, *activation_targets)
        args, kwargs, _, _ = module_config.get_method_args(method)
        with mod.capture():
            for _ in range(n):
                _ = getattr(mod, method)(*args, **kwargs)
        # pass through a second time
        with mod.capture(cb):
            for _ in range(n):
                _ = getattr(mod, method)(*args, **kwargs)
        assert list(cb.captures.keys()) == [f"{method}:{t}" for t in mod.bended_keys(fn=method)]


@pytest.mark.parametrize('module_config', modules_to_test)
def test_interpolation_env(module_config, n=8):
    mod = module_config.get_bended_module()
    for method, (args, kwargs, weight_targets, activation_targets) in module_config:    
        mod.reset()
        mod.trace(fn=method, **kwargs)
        for target in activation_targets:
            cb = tb.InterpolationFromCapture()
            mod.bend(cb, target)
            args, kwargs, _, _ = module_config.get_method_args(method)
            with mod.capture():
                for _ in range(n):
                    for k, v in kwargs.items(): kwargs[k] = torch.randn_like(v)
                    _ = getattr(mod, method)(*args, **kwargs)
            onehot = torch.randn(4, cb.captures[f"{method}:{target}"].shape[0])
            outs = mod.from_activations(f"{target}", **{target: onehot}, **kwargs, fn=method)