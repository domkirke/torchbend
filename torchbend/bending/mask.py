import torch
import copy
from operator import mul
from functools import reduce
from typing import Optional, List, Tuple, Iterable, Union

from torch.nn.parameter import Parameter as Parameter
from .base import BendingCallback, BendingCallbackException
from .parameter import BendingParamType
from .utils import prod



#TODO : (philippe's idea), thresholding activations keeping N% of maximum amplitudes
#TODO : register a quantile function to have dynamic masking from the prob parameter (either ordered permutation, or beta distribution over mean and selection trough parameter)

class Mask(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = {'prob': None}

    def __init__(self, prob: float = 0.3, seed: int = None, dim: Optional[Union[int, List[int]]]=None):
        super().__init__(seed=seed)
        # register paramters
        self.register_controllable('prob', prob)
        self.dim = dim
        # init masks
        self._masks = torch.nn.ParameterList()
        self._mask_names = []
        self._mask_shapes = torch.jit.Attribute([], List[List[int]])

    def script(self):
        mod = copy.copy(self)
        return mod

    def __repr__(self):
        return f"Mask(prob={float(self.prob):.3f})"

    def _get_mask_shape(self, shape: List[int]):
        dim = self.dim
        if dim is None:
            return shape
        if len(shape) == 0:
            return tuple()
        mask_shape = [1] * len(shape)
        if isinstance(dim, int):
            mask_shape[dim] = shape[dim]
        elif isinstance(dim, list):
            for d in dim:
                mask_shape[d] = shape[d]
        return mask_shape
    
    def _init_mask(self, shape: List[int]):
        #TODO generator not scriptable
        prob = self.get('prob')
        mask_shape = self._get_mask_shape(shape)
        if torch.jit.is_scripting():
            mask = torch.bernoulli(torch.full(size=mask_shape, fill_value=prob)).requires_grad_(False)#, generator=self.generator)
        else:
            mask = torch.bernoulli(torch.full(size=mask_shape, fill_value=prob)).requires_grad_(False)
        return mask

    def _add_mask(self, name, shape):
        mask = self._init_mask(shape)
        self._masks.append(mask)
        # disable gradient
        self._masks[-1].requires_grad_(False)
        self._mask_names.append(name)
        self._mask_shapes.value.append(shape)

    def _mask_from_name(self, name: str) -> torch.Tensor:
        for i, m in enumerate(self._masks):
            if self._mask_names[i] == name:
                return m
        raise RuntimeError('does not have mask for name %s'%name)
        
    def register_activation(self, name, shape):
        name = super(Mask, self).register_activation(name, shape)
        self._add_mask(name, shape)

    def register_weight(self, parameter: List[Parameter], name=None, cache: bool = True):
        name = super().register_weight(parameter, name=name, cache=cache)
        self._add_mask(name, parameter.shape)

    def get_mask(self, param, name: Optional[str]) -> torch.Tensor:
        if name is not None:
            return self._mask_from_name(name)
        else:
            mask = torch.bernoulli(torch.full_like(param, fill_value=float(self.prob))).to(param)
        return mask

    def get_mask_from_id(self, idx: int) -> torch.nn.Parameter:
        #grrrr
        for i, v in enumerate(self._masks):
            if i == idx:
                return v
        raise BendingCallbackException('%s not present in masks'%idx)

    def update(self):
        for i, v in enumerate(self._masks):
            v.set_(self._init_mask(v.shape))

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor) -> None:
        param.set_(self.get_mask_from_id(idx) * cache)

    def bend_input(self, param: torch.Tensor, name: Optional[str] = None):
        mask = self.get_mask(param, name).to(param)
        return param * mask
        
                  
class OrderedMask(Mask): 

    def __repr__(self): 
        return f"Mask(prob={float(self.prob):.3f})"

    def _init_mask(self, shape: List[int]):
        mask_shape = self._get_mask_shape(shape)
        numel = prod(mask_shape)
        if torch.jit.is_scripting():
            mask = torch.randperm(numel)
        else:
            mask = torch.randperm(numel, requires_grad=False)
        # stupid but otherwise cannot be added to ParameterList
        return mask.float()

    def _update_mask(self, name: str, shape: List[int]):
        new_mask = self._init_mask(shape).requires_grad_(False)
        good_idx = -1
        for idx, mask_name in enumerate(self._mask_names):
            if mask_name == name:
                good_idx = idx
                break
        for i, mask in enumerate(self._masks):
            if i == good_idx: 
                mask.set_(new_mask)
        if torch.jit.is_scripting(): 
            self._mask_shapes[good_idx] = shape
        else:
            self._mask_shapes.value[good_idx] = shape
        return mask

    def _mask_from_randperm(self, perm, prob, shape: List[int]):
        mask_shape = self._get_mask_shape(shape)
        numel = prod(mask_shape)
        idx = int(prob * numel)
        mask = torch.zeros(numel)
        mask.index_put_((perm[:idx].long(),), torch.full((idx,), 1.))
        return mask.reshape(mask_shape)

    def get_mask(self, param, name: Optional[str]) -> torch.Tensor:
        if name is not None:
            mask_idx = self._mask_from_name(name)
            mask = self._mask_from_randperm(mask_idx, self.prob.get_value(), param.shape).to(param)
        else:
            mask = torch.bernoulli(torch.full_like(param, fill_value=float(self.prob))).to(param)
        return mask
    
    def get_mask_from_id(self, idx: int, cached: torch.Tensor) -> torch.nn.Parameter:
        #grrrr
        for i, v in enumerate(self._masks):
            if i == idx:
                return self._mask_from_randperm(v, self.prob.get_value(), cached.shape).to(cached)
        raise BendingCallbackException('%s not present in masks'%idx)

    def update(self):
        pass

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor):
        param.set_(self.get_mask_from_id(idx, cache) * cache)


class ThresholdActivation(BendingCallback):
    activation_compatible = True
    controllable_params = {'threshold': None}
    def __init__(self, threshold: float = 0.5, dim: Union[int, List[int], None]=None, invert: bool = False):
        super().__init__()
        self.register_controllable('threshold', threshold)
        self.dim = dim
        self.invert = invert

    def forward(self, x: torch.Tensor, name: Optional[str] = None):
        dim = self.dim
        if dim < 0:
            dim =  x.ndim + dim
        threshold = self.get('threshold')

        vals = x.mean(tuple(range(dim+1, x.ndim)))

        idx = torch.argsort(vals, dim=-1, descending=not self.invert)
        idx_sorted = idx[..., :int(threshold * idx.shape[-1])]

        mask = torch.zeros_like(x)
        index = []
        for d in mask.shape[:(dim)]:
            d = torch.arange(d) 
            for _ in range(dim+1):
                d = d.unsqueeze(-1)
            index.append(d)
            index += [idx_sorted]
        mask.__setitem__(index, 1)
        return x * mask
