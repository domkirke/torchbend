from typing import Callable, Optional
import torch
from .base import BendingCallback

class Lambda(BendingCallback):
    weight_compatible = True 
    activation_compatible = True 
    controllable_params = []

    def __init__(self, fn: Callable):
        super().__init__()
        self._callable = fn

    def bend_input(self, x, name: Optional[str] = None):
        return self._callable(x)

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor = None) -> None:
        param.set_(self._callable(cache))
