from typing import Callable, Optional
import torch
from .callback import BendingCallback

class Lambda(BendingCallback):
    weight_compatible = True 
    activation_compatible = True 
    controllable_params = {'param': (torch.Tensor, None), 'seed': (int, 0)}

    def __init__(self, fn: Callable, param: torch.Tensor | int | float | None = None, seed: int | None = None):
        super().__init__(param=param, seed=seed)
        self._callable: Callable = torch.jit.Attribute(fn, Callable)

    def bend_input(self, x, param: torch.Tensor | None = None, seed: torch.Tensor | None = None, name: Optional[str] = None):
        if seed is not None:
            torch.manual_seed(int(seed))
        if torch.jit.is_scripting():
            return self._callable(x)#, param)
        else:
            return self._callable.value(x)#, param)

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor = None) -> None:
        if torch.jit.is_scripting():
            new_param = self._callable(cache)#, param)
        else:
            new_param = self._callable.value(cache)#, param)
        param.set_(new_param)
