from ..utils import can_concatenate
from .callback import BendingCallback, BendingCallbackException
from typing import Optional
import torch
import torch.nn as nn

class Capture(BendingCallback):
    weight_compatible = False 
    activation_compatible = True
    jit_compatible = False
    nntilde_compatible = False

    def __init__(self):
        super().__init__()
        self.clear()
    
    def _concatenate_buffers(self, x): 
        if len(x) > 0:
            return torch.cat(x, 0)

    def get_capture(self, name):
        for n, capture in self._captures.items():
            if n == name:
                return capture

    @property 
    def captures(self):
        return self._captures

    @property
    def capturing(self): 
        return self._is_capturing

    @property
    def is_ready(self) -> bool:
        return self._is_initialized or self._is_capturing

    def _n_captures_from_activation(self, name: str, dim=0):
        captures = self._captures[name]
        n = 0
        for c in captures:
            n += c.shape[dim]
        return n

    def n_captures(self, name: str | None = None, dim = 0):
        if name is None: 
            n_captures = {}
            for k, v in self._captures.items(): 
                n_captures[k] = self.n_captures_from_activation(name, dim=dim)
        else:
            n_captures = self._n_captures_from_activation(name, dim=dim)
        return n_captures

    def register_weight(self, parameter, name=None, cache = True):
        name = super().register_weight(parameter, name=name, cache=cache)
        self._buffer_tmp[name] = []

    def register_activation(self, name, shape):
        name = super().register_activation(name, shape)
        self._buffer_tmp[name] = []
        return name

    def record_buffer(self, x, name):
        self._buffer_tmp[name].append(x)

    def clear(self):
        self._captures = nn.ParameterDict()
        self._buffer_tmp = {} 
        self._is_initialized = False

    def stop(self):
        #TODO make batched and non-batched version
        super().stop()
        for k, v in self._buffer_tmp.items():
            if k in self._captures:
                self._captures[k] = [self._captures[k]]
            else:
                self._captures[k] = v
            self._buffer_tmp[k] = []
        self._is_initialized = True

    def bend_input_with_capture(self, x: torch.Tensor, name: Optional[str] = None): 
        """Callback to override for specific behavior with captured content"""
        return x

    def bend_input(self, x: torch.Tensor, name: Optional[str] = None):
        """applies transformation to an input (typically activations)"""
        if self._is_capturing:
            assert name is not None
            self.record_buffer(x, name)
            return x
        else:
            if not self.is_ready: 
                return x
            else:
                return self.bend_input_with_capture(x, name=name)


class InterpolationFromCapture(Capture):

    def __init__(self, *args, dim=0, **kwargs):
        super(InterpolationFromCapture, self).__init__(*args, **kwargs)
        self.dim = dim   

    @property
    def different_input(self):
        return self._is_ready

    def stop(self): 
        super(InterpolationFromCapture, self).stop()
        for k, v in self._captures.items():
            if len(v) > 0:
                assert can_concatenate(v, self.dim), "captures for activation {k} are not concatenable"

    def bend_input_with_capture(self, x: torch.Tensor, name: Optional[str] = None):
        # x : b x b_c
        # captures: b_c x (...)
        # captures -> : 1 x b_c x (...)
        # x: b x b_c x (1,) * ...
        if name not in self._captures: 
            raise BendingCallbackException('capture for activation %s seems empty. Did you record anything?')
        captures = torch.cat(self._captures[name], dim=self.dim).unsqueeze(0)
        x = x.reshape(x.shape + (1, ) * (captures.ndim - 2))
        return (captures * torch.nn.functional.softmax(x, dim=1)).sum(1)

        