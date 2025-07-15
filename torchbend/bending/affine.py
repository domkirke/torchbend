import torch
from typing import Optional
from .parameter import BendingParameter
from .callback import BendingCallback, BendingCallbackAttributeException


class Bias(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = {'bias': (None, 0.)}

    def __init__(self, bias: float | torch.Tensor | BendingParameter = 0.):
        super().__init__(bias=bias)

    def __repr__(self):
        return f"Bias(bias={self.get('bias'):.4f})"

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: Optional[torch.Tensor] = None):
        assert cache is not None
        param.set_(cache + self.get('bias'))

    def bend_input(self, x: torch.Tensor, bias: torch.Tensor | None = None, name: Optional[str] = None):
        if bias is None: 
            bias = self.get('bias')
        return x + bias


class Scale(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = {'scale': (None, 1.)}

    def __init__(self, scale: float | torch.Tensor | BendingParameter  = 1.):
        super().__init__(scale=scale)

    def __repr__(self):
        return f"Scale(scale={self.get('scale'):.4f})"
    
    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: Optional[torch.Tensor] = None):
        assert cache is not None
        param.set_(cache * self.get('scale'))

    def bend_input(self, x: torch.Tensor, scale: torch.Tensor | None = None, name: str | None = None):
        if scale is None:
            scale = self.get('scale')
        return x * scale
        

class Affine(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = {'scale': (None, 1.0), 'bias': (None, 0.0)}

    def __init__(self, bias: float | torch.Tensor | BendingParameter = 0., scale: float = 1.):
        super().__init__(scale=scale, bias=bias)

    def __repr__(self):
        return f"Affine(scale={(self.get('scale')):.4f}, bias={self.get('bias'):.4f})"

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: Optional[torch.Tensor] = None):
        assert cache is not None
        param.set_(cache * self.get('scale') + self.get('bias'))

    def bend_input(self, x: torch.Tensor, scale: torch.Tensor | None = None, bias: torch.Tensor | None = None, name: Optional[str] = None):
        if scale is None: scale = self.get('scale')
        if bias is None: bias = self.get('bias')
        return x * scale + bias

        
        
__all__ = ['Scale', 'Affine', 'Bias']