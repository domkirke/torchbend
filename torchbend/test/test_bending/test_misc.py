import sys, os
import torch, torch.nn as nn
import pytest
import torchbend as tb
from functools import partial

testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)
from test_modules import modules_to_test, ModuleTestConfig



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


def test_split_and_bend():
    #TODO
    pytest.skip("to do")