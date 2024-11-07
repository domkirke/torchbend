from typing import Optional
import torch
from .base import BendingCallback

class InterpolateActivation(BendingCallback):
    weight_compatible = False
    activation_compatible = True
    jit_compatible = False
    nntilde_compatible = False
    controllable_params = []

    def _interp_activations(self, x, interp_weights, softmax=False):
        assert interp_weights.shape[-1] == x.shape[0]
        if softmax: interp_weights = torch.nn.functional.softmax(interp_weights, dim=-1)
        interp_weights = interp_weights.reshape(interp_weights.shape + (1, ) * (x.ndim - 1))
        out = ((interp_weights * x).sum(-interp_weights.ndim+1))
        if out.ndim == x.ndim - 1:
            out = out[None]
        return out
        
    def forward(self, x: torch.Tensor, name: Optional[str] = None, interp_weights: Optional[torch.Tensor] = None, softmax: bool = False):
        if (interp_weights is None):
            return x
        else:
            return self._interp_activations(x, interp_weights, softmax=softmax)
