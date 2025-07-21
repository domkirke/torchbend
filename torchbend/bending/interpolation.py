from typing import Optional
import torch
from .parameter import BendingParameter
from .callback import BendingCallback

class InterpolateActivation(BendingCallback):
    weight_compatible = False
    activation_compatible = True
    jit_compatible = False
    nntilde_compatible = False
    controllable_params = {'interp_weights': (torch.FloatTensor, None), 'softmax': (bool, False)}

    def __init__(self, interp_weights: BendingParameter | torch.FloatTensor | None = None, **kwargs):
        # by default, put interp_weights as additional argument using a placeholder with as_input=True
        if interp_weights is None: 
            interp_weights = BendingParameter('interp_weights', torch.FloatTensor([[1.]]), as_input=True)
        super().__init__(interp_weights = interp_weights, **kwargs)

    def _interp_activations(self, x, interp_weights, softmax: Optional[torch.Tensor] = None):
        assert interp_weights.shape[-1] == x.shape[0]
        if softmax is None:
            use_softmax = False
        else:
            use_softmax = bool(softmax.item())
        if use_softmax: interp_weights = torch.nn.functional.softmax(interp_weights, dim=-1)
        if interp_weights.ndim == 1:
            interp_weights = interp_weights.unsqueeze(0)
        x_r = x.reshape((1,) * (interp_weights.ndim - 1) + x.shape)
        interp_weights_r = interp_weights.reshape(interp_weights.shape + (1, ) * (x.ndim - 1))
        out = ((interp_weights_r * x_r).sum(-interp_weights_r.ndim+1))
        return out

    def bend_input(self, x, interp_weights: Optional[torch.Tensor] = None, softmax: Optional[torch.Tensor] = None, name: Optional[str] = None):
        if (interp_weights is None):
            return x
        else:
            return self._interp_activations(x, interp_weights, softmax=softmax)
