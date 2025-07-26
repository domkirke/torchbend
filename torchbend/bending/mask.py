import torch
import math
import copy
from operator import mul
from functools import reduce
from typing import Optional, List, Tuple, Iterable, Union

from torch.nn.parameter import Parameter as Parameter
from .callback import BendingCallback, BendingCallbackException
from .parameter import BendingParamType, BendingParameter
from .utils import prod
from ..utils import checklist




class Mask(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = {'prob': ((float, torch.Tensor), 1.), 'seed': (int, 0)}

    def __init__(self, prob: BendingParameter | float | None = None, seed: int = None, dim: Optional[Union[int, List[int]]]=None, learnable: bool = False):
        super().__init__(seed=seed, prob=prob)
        # register paramters
        # self.register_controllable('prob', prob)
        self.dim = dim
        self.learnable = learnable
        # init masks
        self._masks = torch.nn.ParameterList()
        self._mask_names = []
        self._mask_shapes = torch.jit.Attribute([], List[List[int]])

    def script(self):
        mod = copy.copy(self)
        return mod

    def __repr__(self):
        return f"Mask(prob={float(self.prob):.3f})"

    def _get_mask_shape(self, shape: List[int]) -> List[int]:
        dim = self.dim
        if dim is None:
            return shape
        if len(shape) == 0:
            return []
        mask_shape = [1] * len(shape)
        if isinstance(dim, int):
            mask_shape[dim] = int(shape[dim])
        elif isinstance(dim, list):
            for d in dim:
                mask_shape[d] = int(shape[d])
        return mask_shape
    
    def _init_mask(self, shape: List[int]):
        prob = float(self.get('prob'))
        mask_shape = self._get_mask_shape(shape)
        #TODO goddamn generator is not pickable.
        torch.manual_seed(int(self.get("seed")))
        mask = torch.bernoulli(torch.full(size=mask_shape, fill_value=prob)).requires_grad_(self.learnable)
        return mask

    def _add_mask(self, name, shape):
        mask = self._init_mask(shape)
        self._masks.append(mask)
        # disable gradient
        self._mask_names.append(name)
        self._mask_shapes.value.append(shape)

    def _mask_from_name(self, name: str) -> torch.Tensor:
        ## for torchscript integration
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

    def get_mask_from_id(self, idx: int) -> torch.nn.Parameter:
        #grrrr
        for i, v in enumerate(self._masks):
            if i == idx:
                return v
        raise BendingCallbackException('%s not present in masks'%idx)
    
    def get_mask(self, param, prob: torch.Tensor | None = None, name: str | None = None) -> torch.Tensor:
        torch.manual_seed(int(self.get("seed")))
        generator = None
        if prob is None or not self._prob_as_input: 
            if name is None:
                return torch.bernoulli(torch.full_like(param, fill_value=float(self.get('prob'))), generator=generator).to(param)
            else:
                return self._mask_from_name(name)
        else:
            if isinstance(prob, float):
                return torch.bernoulli(torch.full_like(param, fill_value=float(self.get("prob"))), generator=generator).to(param)
            elif isinstance(prob, torch.Tensor):
                #TODO perform some broadcast? 
                return torch.bernoulli(prob.expand_as(param), generator=generator).to(param)
            else:
                raise TypeError('wrong type for prob : %s'%type(prob))

    def update(self):
        for i, v in enumerate(self._masks):
            with torch.no_grad():
                for j, s in enumerate(self._mask_shapes.value):
                    if i == j:
                        v.set_(self._init_mask(s))

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: Optional[torch.Tensor] = None) -> None:
        if cache is not None:
            with torch.no_grad():
                param.set_(self.get_mask_from_id(idx) * cache)

    def bend_input(self, x: torch.Tensor, prob: torch.Tensor | None = None, seed: torch.Tensor | None = None, name: str | None = None):
        mask = self.get_mask(x, prob, name)
        return x * mask
        
                  
class OrderedMask(Mask): 

    def __repr__(self): 
        return f"OrderedMask(prob={float(self.prob):.3f})"

    def _get_mask_shape(self, shape: List[int]) -> List[int]:
        dim = self.dim
        if dim is None:
            return shape
        if len(shape) == 0:
            return []
        mask_shape = [1] * len(shape)
        if isinstance(dim, int):
            mask_shape[dim] = int(shape[dim])
        elif isinstance(dim, list):
            for d in dim:
                mask_shape[d] = int(shape[d])
        return mask_shape

    def _init_mask(self, shape: List[int]):
        torch.manual_seed(int(self.get("seed")))
        mask_shape = self._get_mask_shape(shape)
        numel = prod(mask_shape)
        if torch.jit.is_scripting():
            mask = torch.randperm(numel).requires_grad_(self.learnable)
        else:
            mask = torch.randperm(numel).requires_grad_(self.learnable)
        # stupid but otherwise cannot be added to ParameterList
        return mask.float()

    def _update_mask(self, name: str, shape: List[int]):
        new_mask = self._init_mask(shape).requires_grad_(self.learnable)
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

    def _mask_from_randperm(self, perm: torch.Tensor, prob: torch.Tensor | None, shape: List[int]):
        mask_shape = self._get_mask_shape(shape)
        numel = prod(mask_shape)
        if prob is None: 
            prob = self.get('prob')
        idx = int(prob * numel)
        mask = torch.zeros(numel)
        mask.index_put_((perm[:idx].long(),), torch.full((idx,), 1.))
        return mask.reshape(mask_shape)

    def get_mask(self, param, prob: torch.Tensor | None, name: str | None) -> torch.Tensor:
        torch.manual_seed(int(self.get("prob")))
        if name is not None:
            mask_idx = self._mask_from_name(name)
            mask = self._mask_from_randperm(mask_idx, prob, param.shape).to(param)
        else:
            mask = torch.bernoulli(torch.full_like(param, fill_value=float(self.get("prob")))).to(param)
        return mask
    
    def get_mask_from_id(self, idx: int, cached: torch.Tensor) -> torch.nn.Parameter:
        #grrrr
        for i, v in enumerate(self._masks):
            if i == idx:
                return self._mask_from_randperm(v, self.get("prob"), cached.shape).to(cached)
        raise BendingCallbackException('%s not present in masks'%idx)

    def update(self):
        if torch.jit.is_scripting(): 
            mask_shapes = self._mask_shapes
        else:
            mask_shapes = self._mask_shapes.value
        for i, v in enumerate(self._masks):
            with torch.no_grad():
                for j, s in enumerate(mask_shapes):
                    if i == j:
                        v.set_(self._init_mask(s))

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor | None = None):
        assert cache is not None
        param.set_(self.get_mask_from_id(idx, cache) * cache)


class ThresholdActivation(BendingCallback):
    activation_compatible = True
    controllable_params = {'threshold': (None, 0.5)}
    def __init__(self, threshold: BendingParameter | float | None = None, dim: Union[int, List[int], None] = [1, 2], invert: bool = False):
        super().__init__(threshold = threshold)
        self.dim = checklist(dim)
        self.invert = invert

    def bend_input(self, x: torch.Tensor, threshold: torch.Tensor, name: Optional[str] = None):

        dims = self._get_operative_dims(self.dim, x)

        mean_dims: List[int] = []
        for n in range(x.ndim):
            if n not in dims:
                mean_dims.append(n)

        vals = x.mean(mean_dims)

        values = torch.sort(vals.flatten()).values
        idx_limit = int(math.floor(threshold * (values.numel() - 1)))
        threshold_value = values.flatten()[idx_limit]

        if self.invert:
            idx = torch.nonzero(vals >= threshold_value)
        else:
            idx = torch.nonzero(vals <= threshold_value)

        mask = torch.zeros_like(vals)
        mask.index_put_(list(idx.t()), torch.tensor(1.))
        for i in range(x.ndim):
            if i not in dims:
                mask = mask.unsqueeze(i)
        
        return x * mask
