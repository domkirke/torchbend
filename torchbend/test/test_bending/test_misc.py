import sys, os
import torch, torch.nn as nn
import pytest
import torchbend as tb
from functools import partial

testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)
from test_modules import modules_to_test, ModuleTestConfig
from torchbend.utils import PArgs



@pytest.mark.parametrize('cb_class, cb_args', [(tb.Permute, tb.PArgs(dim=0))])
@pytest.mark.parametrize('module_config', modules_to_test)
@pytest.mark.parametrize('as_controllable', [True, False])
@pytest.mark.parametrize('jit', [True, False])
def test_permute(cb_class, cb_args, module_config, jit, as_controllable):

    mod = module_config.get_bended_module()

    for method, (args, kwargs, weight_targets, activation_targets) in module_config.scriptable():
        mod.reset()
        mod.trace(method, **kwargs)
        out_orig = getattr(mod, method)(*args, **kwargs)

        if as_controllable:
            seed = tb.bending.BendingParameter('perm_seed', -1)
        else:
            seed = -1

        permute_callback = cb_class(*cb_args, seed=seed, **cb_args)
        permute_callback.get('seed')
        if len(weight_targets) > 0: mod.bend(permute_callback, *weight_targets, bend_graph=False)
        if len(activation_targets) > 0: mod.bend(permute_callback, *activation_targets, bend_param=False)

        mod_scripted = mod.script(script=jit)
        out_scripted = getattr(mod_scripted, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_scripted))

        if as_controllable:
            mod_scripted._set_bending_control('perm_seed', 0)
            out_scripted = getattr(mod_scripted, method)(*args, **kwargs)
            assert not bool(tb.compare_outs(out_orig, out_scripted))

            mod_scripted._set_bending_control('perm_seed', 1234)
            out_scripted_2 = getattr(mod_scripted, method)(*args, **kwargs)
            assert not bool(tb.compare_outs(out_scripted_2, out_scripted))



@pytest.mark.parametrize('cb_class, cb_args', [(tb.Reverse, tb.PArgs(dim=-1))])
@pytest.mark.parametrize('module_config', modules_to_test)
@pytest.mark.parametrize('as_controllable', [True, False])
@pytest.mark.parametrize('jit', [True, False])
def test_reverse(cb_class, cb_args, module_config, jit, as_controllable):

    mod = module_config.get_bended_module()

    for method, (args, kwargs, _, activation_targets) in module_config.scriptable():
        if len(activation_targets) == 0: continue
        mod.reset()
        mod.trace(method, **kwargs)
        out_orig = getattr(mod, method)(*args, **kwargs)

        if as_controllable:
            reverse = tb.bending.BendingParameter('reverse', 0)
        else:
            reverse = False 

        reverse_callback = cb_class(*cb_args, reverse = reverse, **cb_args)
        reverse_callback.get('reverse')
        if len(activation_targets) > 0: mod.bend(reverse_callback, *activation_targets, bend_param=False)

        mod_scripted = mod.script(script=jit)
        out_scripted = getattr(mod_scripted, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_scripted))

        if as_controllable:
            mod_scripted._set_bending_control('reverse', 1)
            out_scripted = getattr(mod_scripted, method)(*args, **kwargs)
            assert not bool(tb.compare_outs(out_orig, out_scripted))

            mod_scripted._set_bending_control('reverse', 0)
            out_scripted = getattr(mod_scripted, method)(*args, **kwargs)
            assert bool(tb.compare_outs(out_orig, out_scripted))


def scale_scalar(x: torch.Tensor, factor: float = 1.):
    return x * factor

def scale_tensor(x: torch.Tensor, factor: torch.Tensor | None = None):
    if factor is None: 
        return x
    return x * factor


def affine_tensor(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor):
    return x * scale + bias


lambda_fns = [
    (scale_scalar, [("factor", PArgs("scale", 1.))]),
    (scale_tensor, [("factor", None)]),
    (scale_tensor, [("factor", PArgs("scale", torch.tensor(1.)))]),
    (affine_tensor, [("factor", PArgs("scale", torch.tensor(1.))), ("bias", PArgs("bias", torch.tensor(0.)))]),
]

@pytest.mark.parametrize('fn,params', lambda_fns)
@pytest.mark.parametrize('module_config', modules_to_test)
@pytest.mark.parametrize('jit', [True])
def test_lambda_activation(fn, params, module_config, jit):
    mod = module_config.get_bended_module()
    for method, (args, kwargs, _, activation_targets) in module_config.scriptable():
        if len(activation_targets) == 0: continue
        mod.reset()
        mod.trace(method, **kwargs)
        out_orig = getattr(mod, method)(*args, **kwargs)

        param_dict = {}
        for (name, pargs) in params:
            if pargs is not None: param_dict[name] = tb.BendingParameter(*pargs, **pargs)
        lambda_cb = tb.Lambda(fn, **param_dict)
        mod.bend(lambda_cb, *activation_targets, bend_param=False)

        mod_scripted = mod.script(script=jit)
        out_scripted = getattr(mod_scripted, method)(*args, **kwargs)
        assert bool(tb.compare_outs(out_orig, out_scripted))

        # mod_scripted._set_bending_control('reverse', 1)
        # out_scripted = getattr(mod_scripted, method)(*args, **kwargs)
        # assert not bool(tb.compare_outs(out_orig, out_scripted))

