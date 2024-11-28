import torch
from typing import Optional
from .parameter import BendingParamType
from .base import BendingCallback, BendingCallbackAttributeException


class Bias(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = {'bias': None}

    # bias: int | float | complex

    def __init__(self, bias: float = 0.):
        super().__init__()
        self.bias = bias

    def __repr__(self):
        return f"Bias(bias={self.bias:.4f})"

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: Optional[torch.Tensor] = None):
        assert cache is not None
        param.set_(cache + self.get('bias'))

    def bend_input(self, param: torch.Tensor, name: Optional[str] = None):
        return param + self.get('bias')


class Scale(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = {'scale': None}

    def __init__(self, scale: float = 1.):
        super().__init__()
        self.scale = scale

    def __repr__(self):
        return f"Scale(scale={self.scale:.4f})"
    
    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor):
        param.set_(cache * self.get('scale'))

    def bend_input(self, param: torch.Tensor, name: Optional[str] = None):
        return param * self.get('scale')
        

class Affine(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = {'scale': None, 'bias': None}

    def __init__(self, bias: float = 0., scale: float = 1.):
        super().__init__()
        self.bias = bias
        self.scale = scale

    def __repr__(self):
        return f"Affine(scale={self.scale:.4f}, bias={self.bias:.4f})"

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor):
        param.set_(cache * self.get('scale') + self.get('bias'))

    def bend_input(self, param: torch.Tensor, name: Optional[str] = None):
        return param * self.get('scale') + self.get('bias')

        
        
__all__ = ['Scale', 'Affine', 'Bias']