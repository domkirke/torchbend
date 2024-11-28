from .base import BendingCallback, BendingCallbackException
from typing import Optional
import torch
import torch.nn as nn

class Capture(BendingCallback):
    weight_compatible = False 
    activation_compatible = True
    jit_compatible = False
    nntilde_compatible = False
    controllable_params = []

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
    def is_ready(self) -> bool:
        return self._is_initialized or self._is_capturing

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
                self._captures[k] = self._concatenate_buffers([self._captures[k]] + v)
            else:
                self._captures[k] = self._concatenate_buffers(v)
            self._buffer_tmp[k] = []
        self._is_initialized = True

    def bend_input(self, x: torch.Tensor, name: str | None = None):
        return x

    def forward(self, x: torch.Tensor, name: Optional[str] = None):
        """applies transformation to an input (typically activations)"""
        if self._is_capturing:
            assert name is not None
            self.record_buffer(x, name)
            return x
        else:
            if not self.is_ready: raise BendingCallbackException(self._not_ready_str)
            return self.bend_input(x, name=name)


class InterpolationFromCapture(Capture):

    def bend_input(self, x: torch.Tensor, name: Optional[str] = None):
        # x : b x b_c
        # captures: b_c x (...)
        # captures -> : 1 x b_c x (...)
        # x: b x b_c x (1,) * ...
        if name not in self._captures: 
            raise BendingCallbackException('capture for activation %s seems empty. Did you record anything?')
        captures = self._captures[name].unsqueeze(0)
        x = x.reshape(x.shape + (1, ) * (captures.ndim - 2))
        return (captures * torch.nn.functional.softmax(x, dim=1)).sum(1)

    def forward(self, x: torch.Tensor, name: Optional[str] = None):
        """applies transformation to an input (typically activations)"""
        if not self.is_ready: raise BendingCallbackException(self._not_ready_str)
        if self._is_capturing:
            assert name is not None
            self.record_buffer(x, name)
            return x
        else:
            return self.bend_input(x, name=name)
        