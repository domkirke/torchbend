import torch
import torchbend as tb
import pytest
from conftest import *

# def get_module(module_config):
#     module_type, module_args, module_kwargs = module_config
#     return module_type(*module_args, **module_kwargs)



@pytest.mark.parametrize("module_config", modules_to_test)
def test_calls(module_config: ModuleTestConfig):
    module = module_config.get_module()
    bended_module = tb.BendedModule(module)
    for method in module_config.get_methods():
        args, kwargs, _, _ = module_config.get_method_args(method)
        out = getattr(module, method)(*args, **kwargs)
        out_bended = getattr(bended_module, method)(*args, **kwargs)
        is_equal = compare_outs(out, out_bended)
        assert bool(is_equal), "outputs are not equal for method %s ; got %s"%(method, is_equal)


@pytest.mark.parametrize("module_config", modules_to_test)
def test_weights(module_config):
    module = module_config.get_module()
    named_params = dict(module.named_parameters())
    bended_module = tb.BendedModule(module)

    assert bended_module.weights == list(named_params.keys())
    bended_module.print_weights()
    for k, v in named_params.items():
        assert bended_module.param_shape(k) == v.shape

    assert compare_state_dict_tensors(module.state_dict(), bended_module.state_dict())

@pytest.mark.parametrize("module_config", modules_to_test)
def test_weight_bending(module_config):
    """
    Safe bending adds the bending callbacks in a list, and modifies the state dict of a module
    safely at call.
    """
    module = module_config.get_module()
    bended_module = tb.BendedModule(module)

    # remote bending
    zero_callback = tb.Mask(0.)
    bended_module.bend(zero_callback, *module_config.weight_targets(), verbose=True)
    assert compare_state_dict_tensors(module.state_dict(), bended_module.state_dict()), "module's state dicts have been affected by bending."
    assert len(bended_module.bending_callbacks) == 1
    assert len(bended_module.bended_params) != 0
    unbended_dict = bended_module.state_dict()
    bended_dict = bended_module.bended_state_dict()
    for k, v in module.state_dict().items():
        if v.eq(0).all():
            continue
        assert torch.allclose(bended_dict[k], torch.zeros_like(bended_dict[k]))
        assert not bended_dict[k].eq(unbended_dict[k]).any()

    # try calls
    for method in module_config.get_methods():
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(module, method)(*args, **kwargs)
        out_bended = getattr(bended_module, method)(*args, **kwargs)
        assert not bool(compare_outs(out_orig, out_bended))

    # reset
    bended_module.reset()
    assert len(bended_module.bending_callbacks) == 0
    assert len(bended_module.bended_params) == 0
    for method in module_config.get_methods():
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(module, method)(*args, **kwargs)
        out_bended = getattr(bended_module, method)(*args, **kwargs)
        assert bool(compare_outs(out_orig, out_bended))
   

@pytest.mark.parametrize("module_config", modules_to_test)
def test_weight_bending_inplace(module_config):
    """
    In place bending does not modify the BendedModule state dict, but instead modifies the inner module in place.
    This way, it can be reverted to saved state anyway, but still be bended in active calls (like inside wrappers.)
    """
    module = module_config.get_module()
    bended_module = tb.BendedModule(module)
    zero_callback = tb.Mask(0.)

    # in-place bending
    bended_module.bend_(zero_callback, *module_config.weight_targets(), verbose=True)

    # compare parameters
    for k, v in module.state_dict().items():
        # if v.eq(0).all():
        #    continue 
        # bended module's state dict is not changed ; inner module is modified in place
        assert torch.allclose(v, bended_module.state_dict()[k])
        assert not torch.allclose(v, bended_module.module.state_dict()[k])
        assert torch.allclose(bended_module.bended_state_dict()[k], bended_module.module.state_dict()[k])
        assert not torch.allclose(bended_module.bended_state_dict()[k], bended_module.state_dict()[k])

    # try calls
    for method in module_config.get_methods():
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(module, method)(*args, **kwargs)
        out_bended = getattr(bended_module, method)(*args, **kwargs)
        assert not bool(compare_outs(out_orig, out_bended))

    # reset
    bended_module.reset()
    for k, v in module.state_dict().items():
        # if v.eq(0).all():
        #    continue 
        # bended module's state dict is not changed ; inner module is modified in place
        assert torch.allclose(v, bended_module.state_dict()[k])
        assert torch.allclose(v, bended_module.module.state_dict()[k])
        assert torch.allclose(bended_module.bended_state_dict()[k], bended_module.module.state_dict()[k])
        assert torch.allclose(bended_module.bended_state_dict()[k], bended_module.state_dict()[k])

    for method in module_config.get_methods():
        args, kwargs, _, _ = module_config.get_method_args(method)
        out_orig = getattr(module, method)(*args, **kwargs)
        out_bended = getattr(bended_module, method)(*args, **kwargs)
        assert bool(compare_outs(out_orig, out_bended))


@pytest.mark.parametrize("module_config", modules_to_test)
def test_activation_bending(module_config):
    """
    Safe bending adds the bending callbacks in a list, and modifies the state dict of a module
    safely at call.
    """
    module = module_config.get_module()
    bended_module = tb.BendedModule(module)
    zero_callback = tb.Mask(0.)

    for method in module_config.get_methods():
        args, kwargs, _, bended_activations = module_config.get_method_args(method)
        bended_module.trace(method, *args, **kwargs)
        bended_module.print_activations(method)

        for t in bended_activations:
            bended_module.bend(zero_callback, t, fn=method, verbose=True)

        outs_orig = getattr(module, method)(*args, **kwargs)
        outs = getattr(bended_module, method)(*args, **kwargs)
        assert not bool(compare_outs(outs, outs_orig))

        for t in bended_activations:
            # t = t + "_bended"
            outs = bended_module.get_activations(t, **kwargs, fn=method)
            assert compare_outs(outs[t], torch.zeros_like(outs[t]))