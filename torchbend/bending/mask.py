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
    """Randomly zeros out elements of the tensor with probability (1 − prob). Works on weights and activations. Use dim to restrict masking to a single axis."""
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = {'prob': ((float, torch.Tensor), 1.), 'seed': (int, 0)}
    _extra_init_params = {
        "dim": {"type": "int", "default": None, "required": False, "label": "dim (axis)",
                "description": "Axis along which masking is applied. Leave empty to mask all elements independently."},
    }
    _param_ui = {
        'prob': {
            'range': [0., 1.],
            'step':  0.01,
            'description': "Keep probability: 1.0 = no masking, 0.0 = zero out everything.",
            'guard': lambda v, cb: True if 0. <= v <= 1. else ValueError(f"prob must be in [0, 1], got {v:.4f}"),
        },
        'seed': {
            'range':  [-1, 999],
            'widget': 'int',
            'description': "Random seed for reproducible masks. -1 = draw a fresh mask on every forward pass.",
            'guard':  lambda v, cb: True if v >= -1 else ValueError("seed must be ≥ -1  (-1 = random)"),
        },
    }

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
        prob = self.get('prob')
        seed = self.get('seed')
        if prob is None: prob = torch.tensor(1.)
        if seed is None: seed = torch.tensor(0)
        mask_shape = self._get_mask_shape(shape)
        #TODO goddamn generator is not pickable.
    
        torch.manual_seed(int(seed))
        mask = torch.bernoulli(torch.full(size=mask_shape, fill_value=float(prob))).requires_grad_(self.learnable)
        return mask

    def _add_mask(self, name, shape):
        mask = self._init_mask(shape)
        if name not in self._mask_names:
            self._mask_shapes.value.append(shape)
        else:
            self._mask_shapes.value[self._mask_names.index(name)] = shape
        self._upsert_buffer(self._masks, self._mask_names, name, mask)

    def _mask_from_name(self, name: str) -> torch.Tensor:
        ## for torchscript integration
        for i, m in enumerate(self._masks):
            if self._mask_names[i] == name:
                return m
        raise RuntimeError('does not have mask for name %s'%name)
        
    def register_activation(self, name, shape):
        name, shape = super(Mask, self).register_activation(name, shape)
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
        seed = self.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
        generator = None
        if prob is None or not self._prob_as_input: 
            prob = self.get('prob')
            if prob is None: prob  = torch.tensor(1.)
            if name is None:
                return torch.bernoulli(torch.full_like(param, fill_value=float(prob)), generator=generator).to(param)
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

    def update(self) -> None:
        if torch.jit.is_scripting(): 
            mask_shapes = self._mask_shapes
        else:
            mask_shapes = self._mask_shapes.value
        for i, v in enumerate(self._masks):
            with torch.no_grad():
                for j, s in enumerate(mask_shapes):
                    if i == j:
                        v.set_(self._init_mask(s))

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: Optional[torch.Tensor] = None) -> None:
        if cache is not None:
            with torch.no_grad():
                param.set_(self.get_mask_from_id(idx).to(param.device) * cache)

    def bend_input(self, x: torch.Tensor, prob: torch.Tensor | None = None, seed: torch.Tensor | None = None, name: str | None = None):
        mask = self.get_mask(x, prob, name)
        return x * mask.to(x.device)
        
                  
class Binary(BendingCallback):
    """Zeros out whole slices along `dim` with probability (1 - prob) — one keep/drop coin flip
    per index along the axis, not per element (leave dim empty to gate every element
    independently, like Mask). If `fixed` is True the gate pattern is drawn once per registered
    weight/activation and reused on every call; if False a fresh gate is drawn every call."""
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = {'prob': (float, 1.0)}
    _param_ui = {
        'prob': {
            'range': [0., 1.],
            'step':  0.01,
            'description': "Keep probability per index along dim: 1 = no gating, 0 = zero out everything.",
            'guard': lambda v: True if 0. <= v <= 1. else ValueError(f"prob must be in [0, 1], got {v:.4f}"),
        },
    }
    _extra_init_params = {
        "dim": {"type": "int", "default": None, "required": False, "label": "dim (axis)",
                "description": "Axis the gate is drawn along (one keep/drop decision per index). Leave empty to gate every element independently."},
        "fixed": {"type": "bool", "default": True, "required": False,
                  "description": "If True, the gate pattern is drawn once per registered weight/activation and reused; if False, a fresh gate is drawn on every call."},
    }

    def __init__(self, prob: Union[float, BendingParameter] = 1.0, dim: Optional[int] = None, fixed: bool = True):
        super().__init__(prob=prob)
        self.dim = dim
        self.fixed = bool(fixed)
        self._gates = torch.nn.ParameterList()
        self._gate_keys = []
        self._gate_shapes = torch.jit.Attribute([], List[List[int]])

    def __repr__(self):
        return f"Binary(prob={float(self.get('prob')):.3f}, dim={self.dim}, fixed={self.fixed})"

    def _gate_shape(self, shape: List[int]) -> List[int]:
        dim = self.dim
        if dim is None or len(shape) == 0:
            return shape
        d = dim if dim >= 0 else len(shape) + dim
        gshape = [1] * len(shape)
        if d < len(shape):
            gshape[d] = int(shape[d])
        return gshape

    def _draw_gate(self, shape: List[int], prob: float) -> torch.Tensor:
        gshape = self._gate_shape(shape)
        return torch.bernoulli(torch.full(gshape, prob))

    def _init_gate(self, name, shape: List[int]):
        if name not in self._gate_keys:
            self._gate_shapes.value.append(shape)
        else:
            self._gate_shapes.value[self._gate_keys.index(name)] = shape
        if self.fixed:
            prob = self.get('prob')
            prob = 1. if prob is None else float(prob)
            gate = self._draw_gate(shape, prob)
        else:
            gate = torch.zeros(0)
        self._upsert_buffer(self._gates, self._gate_keys, name, gate)

    def register_weight(self, parameter: List[Parameter], name=None, cache: bool = True):
        name = super().register_weight(parameter, name=name, cache=cache)
        self._init_gate(name, list(parameter.shape))

    def register_activation(self, name, shape):
        name, shape = super().register_activation(name, shape)
        self._init_gate(name, list(shape))

    def _gate_from_name(self, name: str) -> torch.Tensor:
        for i, g in enumerate(self._gates):
            if self._gate_keys[i] == name:
                return g
        raise BendingCallbackException('name %s not present in binary gates' % name)

    def _gate_from_id(self, idx: int) -> torch.Tensor:
        for i, g in enumerate(self._gates):
            if i == idx:
                return g
        raise BendingCallbackException('%s not present in binary gates' % idx)

    def update(self):
        if not self.fixed:
            return
        prob = self.get('prob')
        prob = 1. if prob is None else float(prob)
        for i in range(len(self._gates)):
            shape = self._gate_shapes.value[i]
            self._gates[i].data = self._draw_gate(shape, prob).to(self._gates[i].device)

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor) -> None:
        prob = self.get('prob')
        if prob is None:
            return
        with torch.no_grad():
            if self.fixed:
                gate = self._gate_from_id(idx)
            else:
                gate = self._draw_gate(list(cache.shape), float(prob)).to(cache.device)
            param.set_(cache * gate.to(cache))

    def bend_input(self, x: torch.Tensor, prob: Optional[torch.Tensor] = None, name: Optional[str] = None):
        if prob is None:
            return x
        if self.fixed and name is not None:
            gate = self._gate_from_name(name)
        else:
            gate = self._draw_gate(list(x.shape), float(prob))
        return x * gate.to(x)


class OrderedMask(Mask):
    """Like Mask, but elements are removed in a fixed per-seed order as prob decreases
    (deterministic progressive thinning instead of i.i.d. masking)."""
    _param_ui = {
        'prob': {
            'range': [0., 1.],
            'step':  0.01,
            'guard': lambda v: True if 0. <= v <= 1. else ValueError(f"prob must be in [0, 1], got {v:.4f}"),
        },
        'seed': {
            'range':  [-1, 999],
            'widget': 'int',
            'guard':  lambda v: True if v >= -1 else ValueError("seed must be ≥ -1  (-1 = random)"),
        },
    }

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
        seed = self.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
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
        if prob is None: 
            raise ValueError("prob cannot be None")

        idx = int(prob * numel)
        mask = torch.zeros(numel)
        mask.index_put_((perm[:idx].long(),), torch.full((idx,), 1.))
        return mask.reshape(mask_shape)

    def get_mask(self, param, prob: torch.Tensor | None, name: str | None) -> torch.Tensor:
        if name is not None:
            mask_idx = self._mask_from_name(name)
            mask = self._mask_from_randperm(mask_idx, prob, param.shape).to(param)
        else:
            if prob is None: 
                raise ValueError("prob cannot be None")
            mask = torch.bernoulli(torch.full_like(param, fill_value=float(prob))).to(param)
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
                        v.set_(self._init_mask(s).to(v.device))

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor | None = None):
        assert cache is not None
        param.set_(self.get_mask_from_id(idx, cache) * cache)


class ThresholdActivation(BendingCallback):
    """Keeps only the activation slices (along dim) whose mean is below the
    threshold quantile (invert=True keeps the ones above). Activation-only."""
    activation_compatible = True
    controllable_params = {'threshold': (None, 0.5)}
    _extra_init_params = {
        "dim": {"type": "ints", "default": [1, 2], "required": False, "label": "dim (axes)",
                "description": "Axes the slices run along: the mean is taken over every "
                               "other axis, and whole slices are kept or dropped. One axis "
                               "or several, e.g. 1 or 1, 2. Empty: 1, 2."},
        "invert": {"type": "bool", "widget": "toggle", "default": False, "required": False,
                   "label": "invert",
                   "description": "Keep the slices above the threshold instead of below."},
    }
    _param_ui = {
        'threshold': {
            'range': [0., 1.],
            'step':  0.01,
            'guard': lambda v: True if 0. <= v <= 1. else ValueError(f"threshold must be in [0, 1], got {v:.4f}"),
        },
    }
    def __init__(self, threshold: BendingParameter | float | None = None, dim: Union[int, List[int], None] = [1, 2], invert: bool = False):
        super().__init__(threshold = threshold)
        self.dim = checklist(dim)
        self.invert = invert

    def bend_input(self, x: torch.Tensor, threshold: torch.Tensor | None = None, name: Optional[str] = None):

        if threshold is None: 
            return x

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
