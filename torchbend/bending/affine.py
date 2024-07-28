import torch
from typing import Optional
from .base import BendingCallback


class Bias(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = ['bias']
    def __init__(self, bias: float = 0.):
        super().__init__()
        self.bias = bias

    def __repr__(self):
        return f"Bias(bias={self.bias:.4f})"

    def _apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor):
        param.set_(cache + self.get('bias'))

    def forward(self, param: torch.Tensor, name: Optional[str] = None):
        return param + self.get('bias')


class Scale(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = ['scale']

    def __init__(self, scale: float = 1.):
        super().__init__()
        self.scale = scale

    def __repr__(self):
        return f"Scale(scale={self.scale:.4f})"
    
    def _apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor):
        param.set_(cache * self.get('scale'))

    def forward(self, param: torch.Tensor, name: Optional[str] = None):
        return param * self.get('scale')
        

class Affine(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = ['scale', 'bias']

    def __init__(self, bias: float = 0., scale: float = 1.):
        super().__init__()
        self.bias = bias
        self.scale = scale

    def __repr__(self):
        return f"Affine(scale={self.scale:.4f}, bias={self.bias:.4f})"

    def _apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor):
        param.set_(cache * self.get('scale') + self.get('bias'))

    def forward(self, param: torch.Tensor, name: Optional[str] = None):
        return param * self.get('scale') + self.get('bias')

        
        
__all__ = ['Scale', 'Affine', 'Bias']