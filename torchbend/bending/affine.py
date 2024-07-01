import torch
from typing import Optional
from .base import BendingCallback


class Bias(BendingCallback):
    def __init__(self, bias: float = 0.):
        super().__init__()
        self.bias = bias

    def __repr__(self):
        return f"Bias(bias={self.bias:.4f})"

    def forward(self, param: torch.Tensor, name: Optional[str] = None):
        return param + self.bias


class Scale(BendingCallback):
    def __init__(self, scale: float = 1.):
        super().__init__()
        self.scale = scale

    def __repr__(self):
        return f"Bias(scale={self.scale:.4f})"

    def forward(self, param: torch.Tensor, name: Optional[str] = None):
        return param * self.scale


class Affine(BendingCallback):
    def __init__(self, bias: float = 0., scale: float = 1.):
        super().__init__()
        self.bias = bias
        self.scale = scale

    def __repr__(self):
        return f"Affine(scale={self.scale:.4f}, bias={self.bias:.4f})"

    def forward(self, param: torch.Tensor, name: Optional[str] = None):
        return param * self.scale + self.bias

        
                  

        
__all__ = ['Scale', 'Affine', 'Bias']