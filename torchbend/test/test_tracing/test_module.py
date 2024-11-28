import torch
import torchbend as tb
from torchbend.tracing import BendedWrapper, unmatching_ids, get_model_copy, clone_parameters
from torchbend.tracing.utils import state_dict, named_parameters, get_kwargs_from_gm
from torchbend.utils import checktuple, get_parameter
import pytest
import sys, os
from conftest import get_log_file, log_to_file
from typing import Optional

testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)
from test_modules.module_test_modules import modules_to_test, ModuleTestConfig


@pytest.mark.parametrize("module_config", modules_to_test)
def test_model_copy(module_config: ModuleTestConfig):
    torch.set_grad_enabled(False)

    module, bended = module_config.get_modules()
    weights = bended.bendable_keys(*module_config.weight_targets()) 
    assert id(module) == id(bended.module)
    assert not unmatching_ids(module, bended, weights, data=True)

    if isinstance(bended, BendedWrapper):
        module = bended.wrapped_module

    # test model copy
    module_copy = get_model_copy(module, copy_parameters=False)
    assert not unmatching_ids(module, module_copy, weights)
    assert not unmatching_ids(module, module_copy, weights, data=True)
    module_copy = get_model_copy(module, copy_parameters=True)
    assert unmatching_ids(module, module_copy, weights)
    assert not unmatching_ids(module, module_copy, weights, data=True)
    
    # test replacements
    weights = bended.bendable_keys(*module_config.weight_targets())
    module_copy = get_model_copy(module, copy_parameters=True)
    for w in weights:
        if (get_parameter(module_copy, w).data == 0).all(): continue
        get_parameter(module_copy, w).zero_()
        assert (get_parameter(module, w) == get_parameter(module_copy, w)).any()
    module_copy = get_model_copy(module, copy_parameters=True)
    clone_parameters(module_copy, weights)
    for w in weights:
        if (get_parameter(module_copy, w).data == 0).all(): continue
        get_parameter(module_copy, w).zero_()
        assert not (get_parameter(module, w) == get_parameter(module_copy, w)).any()

@pytest.mark.parametrize("module_config", modules_to_test)
def test_bending_config(module_config):
    module = module_config.get_module()
    for method in module_config.get_methods():
        bended = module_config.get_bended_module(module)
        bendable_params = module_config.weight_targets(method)
        bending_config = tb.BendingConfig((tb.Mask(0.3), *bendable_params), (tb.Mask(0.2), *bendable_params))
        bended.bend(bending_config)
        
        bending_config.bind(bended)
        recorded_config = bended.bending_config()
        assert bending_config == recorded_config


@pytest.mark.parametrize("module_config", modules_to_test)
def test_methods(module_config: ModuleTestConfig):
    module, bended_module = module_config.get_modules()
    for method in module_config.get_methods():
        args, kwargs, _, _ = module_config.get_method_args(method)
        out = getattr(module, method)(*args, **kwargs)
        out_bended = getattr(bended_module, method)(*args, **kwargs)
        is_equal = tb.compare_outs(out, out_bended)
        assert bool(is_equal), "outputs are not equal for method %s ; got %s"%(method, is_equal)


@pytest.mark.parametrize("module_config", modules_to_test)
def test_weights(module_config):
    module, bended_module = module_config.get_modules()
    named_params = dict(named_parameters(module))

    assert bended_module.weight_names == list(named_params.keys())
    bended_module.print_weights()
    weight_names = bended_module.weight_names
    for k, v in named_params.items():
        assert k in weight_names
        assert bended_module.weight_shape(k) == v.shape
    assert tb.compare_state_dict_tensors(state_dict(module), bended_module.state_dict())
    bended_module.print_weights(out=get_log_file(__file__))


@pytest.mark.parametrize("module_config", modules_to_test)
def test_graph(module_config):
    module, bended_module = module_config.get_modules()
    with open(get_log_file(__file__), 'w+') as f:
        for method in module_config.get_methods():
            f.write('-'*10)
            f.write('\ncurrent method : %s'%method)
            args, kwargs, _, _ = module_config.get_method_args(method)
            out = checktuple(getattr(module, method)(*args, **kwargs))
            graph, out_bended = bended_module.trace(method, *args, **kwargs, _return_out=True)
            assert bended_module.is_traced(method)
            assert bended_module.graph(method)
            for i, o in enumerate(out_bended):
                is_equal = tb.compare_outs(out[i], o)
                assert bool(is_equal), "outputs are not equal for method %s ; got %s"%(method, is_equal)
            for act in bended_module.activation_names(method):
                shape = bended_module.activation_shape(act, fn=method)
            log_to_file(f, "graph", graph)
            log_to_file(f, "activations", bended_module.print_activations(method))


@pytest.mark.parametrize("module_config", modules_to_test)
def test_weight_bending(module_config):
    """
    Safe bending adds the bending callbacks in a list, and modifies the state dict of a module
    safely at call.
    """
    module, bended_module = module_config.get_modules()
    for method in module_config.get_methods():
        # remote bending
        zero_callback = tb.Mask(0.)
        bended_module.bend(zero_callback, *module_config.weight_targets(method), verbose=True, fn=method)
        assert tb.compare_state_dict_tensors(state_dict(module), bended_module.state_dict()), "module's state dicts have been affected by bending."
        assert len(bended_module.bending_callbacks) == 1
        assert len(bended_module.bended_params) != 0
        unbended_dict = bended_module.state_dict()
        bended_dict = bended_module.bended_state_dict()
        for k in bended_module.bended_keys():
            assert torch.allclose(bended_dict[k], torch.zeros_like(bended_dict[k]))
            assert not bended_dict[k].eq(unbended_dict[k]).any()

        # try calls
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(module, method)(*args, **kwargs)
        out_bended = getattr(bended_module, method)(*args, **kwargs)
        assert tb.compare_state_dict_tensors(state_dict(module), bended_module.state_dict()), "module's state dicts have been affected by bending."
        assert not bool(tb.compare_outs(out_orig, out_bended))

        # reset
        bended_module.reset()
        assert len(bended_module.bending_callbacks) == 0
        assert len(bended_module.bended_params) == 0
        for method in module_config.get_methods():
            args, kwargs, _, _ = module_config.get_method_args(method)
            out_orig = getattr(module, method)(*args, **kwargs)
            out_bended = getattr(bended_module, method)(*args, **kwargs)
            assert bool(tb.compare_outs(out_orig, out_bended))


@pytest.mark.parametrize("module_config", modules_to_test)
def test_versions(module_config):
    #TODO something modifies original module ; check
    module = module_config.get_module()
    module2 = module_config.get_module()
    for method in module_config.get_methods():
        bended_module = module_config.get_bended_module(module)
        zero_callback = tb.Mask(0.)
        args, kwargs, _, _ = module_config.get_method_args(method)
        weight_targets = module_config.weight_targets(method)
        bended_module.bend(zero_callback, *weight_targets, verbose=True)
        bended_module.write("bended")
        for target in bended_module.bendable_keys(*weight_targets):
            assert not tb.compare_outs(bended_module.state_dict('_default')[target], bended_module.state_dict('bended')[target])
            assert tb.compare_outs(bended_module.state_dict()[target], bended_module.state_dict('bended')[target])
        # revert to original version
        bended_module.version = None
        # remove bendings of original version
        bended_module.reset()
        assert tb.compare_outs(bended_module.state_dict()[target], bended_module.state_dict('_default')[target])

        # test version context managers
        bended_module.create_version("imported", module2)
        with bended_module.set_version("bended"):
            out_bended = getattr(bended_module, method)(*args, **kwargs)
        with bended_module.set_version("imported"):
            out_imported = getattr(bended_module, method)(*args, **kwargs)
        with bended_module.set_version():
            out = getattr(bended_module, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(out, out_bended))
        assert not bool(tb.compare_outs(out, out_imported))
        break

@pytest.mark.parametrize("module_config", modules_to_test)
def test_weight_interpolation(module_config):
    module = module_config.get_module()
    module2 = module_config.get_module()
    # test interpolation with bended version
    for method in module_config.get_methods():
        bended_module = module_config.get_bended_module(module)
        args, kwargs, w_targets, a_targets = module_config.callback_with_args[method][:4]
        bended_module.create_version("imported", module2)
        out_unbended = getattr(bended_module, method)(*args, **kwargs)
        bended_module.bend(tb.Mask(0.), *w_targets, *a_targets)
        bended_module.write("bended")
        out_bended = getattr(bended_module, method)(*args, **kwargs)
        with bended_module.interpolate(1., bended=1., imported=1.):
            out_interpolated = getattr(bended_module, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(out_unbended, out_bended))
        assert not bool(tb.compare_outs(out_unbended, out_interpolated))
        assert not bool(tb.compare_outs(out_bended, out_interpolated))


@pytest.mark.parametrize("module_config", modules_to_test)
def test_activation_getter(module_config):
    module, bended_module = module_config.get_modules()
    cb = tb.Mask(prob=0.)
    for method in module_config.get_methods():
        args, kwargs, _, _ = module_config.get_method_args(method)
        bended_module.trace(*args, **kwargs, fn=method)
        targets = module_config.activation_targets(method)
        outs = bended_module.get_activations(*targets, **kwargs, fn=method)
        assert set(targets) == set(outs.keys()) 
        bended_module.bend(cb, *targets)
        outs_bended = bended_module.get_activations(*targets, **kwargs, fn=method, _save_as_method="getter") 
        outs_bended_2 = getattr(bended_module, "getter")(**kwargs)
        for t in targets:
            assert bool(tb.compare_outs(outs[t], outs_bended[t]))
            assert bool(tb.compare_outs(outs[t], outs_bended_2[t]))
        #TODO bend and check
        


@pytest.mark.parametrize("module_config", modules_to_test)
def test_activation_bending(module_config):
    """
    Safe bending adds the bending callbacks in a list, and modifies the state dict of a module
    safely at call.
    """
    module, bended_module = module_config.get_modules()
    zero_callback = tb.Mask(0.)
    zero_callback_2 = tb.Mask(0.)

    for method in module_config.get_methods():
        args, kwargs, _, bended_activations = module_config.get_method_args(method)
        bended_module.trace(method, *args, **kwargs)
        bended_module.print_activations(method)

        for t in bended_activations:
            bended_module.bend(zero_callback, t, fn=method, verbose=True)
            bended_module.bend(zero_callback_2, t, fn=method, verbose=True)

        outs_orig = getattr(module, method)(*args, **kwargs)
        outs = getattr(bended_module, method)(*args, **kwargs)
        assert not bool(tb.compare_outs(outs, outs_orig))

        for t in bended_activations:
            outs = bended_module.get_activations(t, **kwargs, fn=method)
            outs_bended = bended_module.get_activations(f"{t}_bended", **kwargs, fn=method)
            assert tb.compare_outs(outs_bended[f"{t}_bended"], torch.zeros_like(outs[t]))


@pytest.mark.parametrize("module_config", modules_to_test)
def test_bending_callbacks_as_inputs(module_config):

    class Dumb1(tb.BendingCallback):
        def forward(self, x: torch.Tensor, name: Optional[str] = None, param1: int = 2, param2: str = "plpl"):
            x = torch.ones_like(x) * param1
            return x
    class Dumb2(tb.BendingCallback):
        def forward(self, x: torch.Tensor, name: Optional[str] = None, param1: int = -1, param3: str = "plpl"):
            x = x * param1
            return x

    module, bended_module = module_config.get_modules()
    cb1 = Dumb1()
    cb2 = Dumb2()

    for method in module_config.get_methods():
        args, kwargs, _, bended_activations = module_config.get_method_args(method)
        bended_module.trace(method, *args, **kwargs)

        # test with 1 activations
        for t in bended_activations:
            bended_module.reset()
            bended_module.bend(cb1, t, fn=method, verbose=True)
            bended_module.bend(cb2, t, fn=method)
            acts = bended_module.get_activations(f"{t}$", fn=method, **kwargs)
            out = bended_module.from_activations(t, fn=method, **acts, **kwargs, param1 = [1, 1], param2="coucou", param3="bonjour")
            

# @pytest.mark.parametrize("module_config", modules_to_test)
# def test_weight_bending_inplace(module_config):
#     """
#     In place bending bends in place the parameter dict, without altering the original module to enable bending reset.
#     """
#     for method in module_config.get_methods():
#         module = module_config.get_module()
#         bended_module = tb.BendedModule(module)
#         args, kwargs, _, _ = module_config.get_method_args(method)
#         out_orig = getattr(module, method)(*args, **kwargs)
#         zero_callback = tb.Mask(0.)

#         # in-place bending
#         bended_module.bend_(zero_callback, 
#                             *module_config.weight_targets(), 
#                             *module_config.activation_targets()) 

#         # compare parameters
#         for k in bended_module.bended_params.keys():
#             # bended module's state dict is not changed ; inner module is modified in place
#             assert torch.allclose(v, bended_module.state_dict()[k])
#             assert not torch.allclose(v, bended_module.module.state_dict()[k])
#             assert torch.allclose(bended_module.bended_state_dict()[k], bended_module.module.state_dict()[k])
#             assert not torch.allclose(bended_module.bended_state_dict()[k], bended_module.state_dict()[k])

#         # try calls
#         out_bended = getattr(bended_module, method)(*args, **kwargs)
#         assert not bool(tb.compare_outs(out_orig, out_bended))

#         # reset
#         bended_module.reset()
#         for k, v in module.state_dict().items():
#             # if v.eq(0).all():
#             #    continue 
#             # bended module's state dict is not changed ; inner module is modified in place
#             assert torch.allclose(v, bended_module.state_dict()[k])
#             assert torch.allclose(v, bended_module.module.state_dict()[k])
#             assert torch.allclose(bended_module.bended_state_dict()[k], bended_module.module.state_dict()[k])
#             assert torch.allclose(bended_module.bended_state_dict()[k], bended_module.state_dict()[k])

#         args, kwargs, _, _ = module_config.get_method_args(method)
#         out_orig = getattr(module, method)(*args, **kwargs)
#         out_bended = getattr(bended_module, method)(*args, **kwargs)
#         assert bool(tb.compare_outs(out_orig, out_bended))

