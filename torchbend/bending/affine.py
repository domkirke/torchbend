import torch
from copy import copy
import math
from typing import Optional
from .parameter import BendingParameter
from collections import OrderedDict
from .callback import BendingCallback, BendingCallbackAttributeException


def _parse_affine_control(inp, ctrl):
    assert ctrl.shape[0] == 1 or ctrl.shape[0] == inp.shape[0]
    if ctrl.shape[1] != 1:
        if ctrl.shape[1] < inp.shape[1]:
            ctrl = torch.cat([ctrl, torch.zeros(ctrl.shape[0], inp.shape[1] - ctrl.shape[1], ctrl.shape[2])])
        elif ctrl.shape[1] > inp.shape[1]: 
            ctrl = ctrl[:, :inp.shape[1]]
    if ctrl.shape[-1] != inp.shape[-1]:
        ctrl = torch.nn.functional.interpolate(ctrl, inp.shape[-1], mode="nearest")
    return ctrl


class Bias(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = OrderedDict({'bias': (None, 0.)})

    def __init__(self, bias: float | torch.Tensor | BendingParameter = 0.):
        super().__init__(bias=bias)

    def __repr__(self):
        return f"Bias(bias={self.get('bias'):.4f})"

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: Optional[torch.Tensor] = None):
        assert cache is not None
        bias = self.get("bias")
        if bias is not None:
            if cache is not None: 
                param.set_(cache + bias.to(param.device))

    def bend_input(self, x: torch.Tensor, bias: torch.Tensor | None = None, name: Optional[str] = None):
        if bias is None: 
            bias = torch.tensor(0.)
        else:
            if self._bias_as_input and self._for_nntilde: bias = _parse_affine_control(x, bias)
        return x + bias.to(x.device)


class Scale(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = OrderedDict({'scale': (None, 1.)})

    def __init__(self, scale: float | torch.Tensor | BendingParameter  = 1.):
        super().__init__(scale=scale)

    def __repr__(self):
        return f"Scale(scale={self.get('scale'):.4f})"
    
    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: Optional[torch.Tensor] = None):
        scale = self.get('scale')
        if scale is not None:
            assert cache is not None
            param.set_(cache * scale.to(param.device))

    def bend_input(self, x: torch.Tensor, scale: torch.Tensor | None = None, name: str | None = None):
        if scale is None: 
            scale = torch.tensor(1.)
        if self._scale_as_input and self._for_nntilde: scale = _parse_affine_control(x, scale)
        return x * scale.to(x.device)
        

class Affine(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = OrderedDict({'scale': (None, 1.0), 'bias': (None, 0.0)})

    def __init__(self, bias: float | torch.Tensor | BendingParameter = 0., scale: float = 1.):
        super().__init__(scale=scale, bias=bias)

    def __getstate__(self):
        return super().__getstate__()

    def __setstate__(self, state):
        return super().__setstate__(state)

    def __repr__(self):
        return f"Affine(scale={(self.get('scale')):.4f}, bias={self.get('bias'):.4f})"

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: Optional[torch.Tensor] = None):
        if cache is not None: 
            scale = self.get("scale")
            if scale is not None: 
                cache = cache * scale.to(param.device)
            bias = self.get("bias")
            if bias is not None: 
                cache = cache * bias.to(param.device)
            param.set_(cache)

    def bend_input(self, x: torch.Tensor, scale: torch.Tensor | None = None, bias: torch.Tensor | None = None, name: Optional[str] = None):
        if scale is None:
            scale = torch.tensor(1.)
        else:
            if self._scale_as_input and self._for_nntilde: scale = _parse_affine_control(x, scale)
        if bias is  None: 
            bias = torch.tensor(0.)
        else:
            if self._bias_as_input and self._for_nntilde: bias = _parse_affine_control(x, bias)
        return x * scale.to(x.device) + bias.to(x.device)

        
__all__ = ['Scale', 'Affine', 'Bias']