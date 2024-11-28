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
        if interp_weights.ndim == 1:
            interp_weights = interp_weights.unsqueeze(0)
        x_r = x.reshape((1,) * (interp_weights.ndim - 1) + x.shape)
        interp_weights_r = interp_weights.reshape(interp_weights.shape + (1, ) * (x.ndim - 1))
        out = ((interp_weights_r * x_r).sum(-interp_weights_r.ndim+1))
        return out
        
    def forward(self, x: torch.Tensor, name: Optional[str] = None, interp_weights: Optional[torch.Tensor] = None, softmax: bool = False):
        if (interp_weights is None):
            return x
        else:
            return self._interp_activations(x, interp_weights, softmax=softmax)
