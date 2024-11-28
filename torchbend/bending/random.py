import torch
from typing import Optional, List
from .base import BendingCallback, BendingParamType, BendingCallbackException


class Normal(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    valid_ops = ['add', 'mul']
    controllable_params = {'std': (BendingParamType['float'], BendingParamType['int'])}

    def __init__(self, std: float = 0.3, seed: int = None, dim=None, op = "add"):
        super().__init__(seed=seed)
        self.register_controllable('std', std)
        assert op in self.valid_ops
        self.op = op
        self.dim = dim

        # init masks
        self._noises = torch.nn.ParameterList()
        self._noise_names = []
        self._noise_shapes = torch.jit.Attribute([], List[List[int]])

    def __repr__(self):
        rp =  f"{type(self).__name__}(std={float(self.std):.3f}"
        if self.seed is not None:
            rp += f", seed={int(self.seed)}"
        rp+=")"
        return rp

    def _get_rnd_shape(self, shape: List[int]):
        dim = self.dim
        if dim is None:
            return shape
        rnd_shape = [1] * len(shape)
        if isinstance(dim, int):
            rnd_shape[dim] = shape[dim]
        elif isinstance(dim, list):
            for d in dim:
                rnd_shape[d] = shape[d]
        return rnd_shape
    
    def _init_rnd_(self, shape: List[int]):
        assert shape is not None, "mask preinit must be given target shape"
        if torch.jit.is_scripting():
            noise = torch.randn(self._get_rnd_shape(shape))
        else:
            noise = torch.randn(self._get_rnd_shape(shape))
        return noise

    def _add_noise(self, name, shape):
        noise = self._init_rnd_(shape)
        self._noises.append(noise)
        # disable gradient
        self._noises[-1].requires_grad_(False)
        self._noise_names.append(name)
        self._noise_shapes.value.append(shape)

    def _noise_from_name(self, name: str) -> torch.Tensor:
        for i, m in enumerate(self._masks):
            if self._noise_names[i] == name:
                return m
        raise RuntimeError('does not have mask for name %s'%name)

    def register_weight(self, parameter: List[torch.nn.Parameter], name=None, cache: bool = True):
        name = super().register_weight(parameter, name=name, cache=cache)
        self._add_noise(name, parameter.shape)

    def register_activation(self, name, shape):
        super(Normal, self).register_activation(name, shape)
        name = name.replace('.', '_')
        self._add_noise(name, shape)

    def _noise_from_name(self, name: str) -> torch.Tensor:
        for i, m in enumerate(self._noises):
            if self._noise_names[i] == name:
                return m
        raise RuntimeError('does not have noise for name %s'%name)

    def get_noise(self, param, name: Optional[str]) -> torch.Tensor:
        if name is not None:
            mask = self._noise_from_name(name)
        else:
            mask = torch.randn_like(param).to(param)
        return mask

    def get_noise_from_id(self, idx: int) -> torch.nn.Parameter:
        #grrrr
        for i, v in enumerate(self._noises):
            if i == idx:
                return v
        raise BendingCallbackException('%s not present in masks'%idx)

    def update(self):
        for i, v in enumerate(self._noises):
            v.set_(self._init_rnd_(v.shape))

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache:torch.Tensor) -> None:
        if self.op == "mul":
            param.set_(self.get_noise_from_id(idx) * cache * self.get('std'))
        else: 
            param.set_(self.get_noise_from_id(idx) * self.get('std') + cache)

    def bend_input(self, param: torch.Tensor, name: Optional[str] = None):
        noise = self.get_noise(param, name).to(param)
        if self.op == "mul":
            return param * (noise * self.get('std'))
        elif self.op == "add":
            return param + (noise * self.get('std'))
        else:
            raise BendingCallbackException('op %s not known'%(self.op))