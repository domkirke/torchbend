import torch
from typing import Optional, Union, List
from .parameter import BendingParameter
from .callback import BendingCallback



class Reverse(BendingCallback):
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = {'reverse': (bool, False)}

    def __init__(self, dim: int, reverse: bool = False):
        super().__init__(reverse=reverse)
        self.dim = dim

    def __repr__(self):
        return f"Permute(dim={self.dim})"

    def bend_input(self, x: torch.Tensor, reverse: Optional[torch.Tensor] = None, name: Optional[str] = None):
        perform = True
        if reverse is None: 
            perform = False
        else:
            if not bool(reverse.item()):
                perform = False
        if perform:
            dim = self.dim if self.dim >= 0 else x.ndim + self.dim
            if dim >= x.ndim:
                return x
            idx = torch.arange(x.shape[self.dim])
            idx = torch.flip(idx, [0])
            return torch.index_select(x, self.dim, idx)
        else:
            return x

