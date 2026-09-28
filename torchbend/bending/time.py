import math
import torch
from typing import Optional, Union, List
from .parameter import BendingParameter
from .callback import BendingCallback, BendingCallbackException



class Reverse(BendingCallback):
    """Flips the tensor along the specified dimension. Typically used to reverse the time axis of activations (e.g. playing audio backwards)."""
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = {'reverse': (bool, 0)}
    _param_ui = {
        'reverse': {
            'widget': 'toggle',
            'description': "Enable/disable the reversal. When off, the tensor passes through unchanged.",
        },
    }

    def __init__(self, dim: int, reverse: int = 0):
        super().__init__(reverse=reverse)
        self.dim = dim

    def __repr__(self):
        return f"Reverse(reverse={self.get('reverse')})"

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


class Unphase(BendingCallback):
    """Circularly rolls the tensor along `dim`, using an independent shift amount for every 1D slice
    taken along that axis (every combination of the other indices, i.e. every "channel"). This decorrelates
    the slices from each other ("unphasing"). In "increment" mode each channel's shift grows by a fixed
    step (`shift`); in "random" mode each channel gets an independent random shift drawn between 0 and
    `shift` (or centered around 0 if `centered` is set)."""
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    valid_modes = ['increment', 'random']
    controllable_params = {'shift': (Optional[int], None), 'centered': (bool, False), 'seed': (int, 0)}
    _param_ui = {
        'shift': {
            'widget': 'int',
            'range': [0, 4096],
            'description': "increment mode: step added per channel. random mode: max shift magnitude. None disables the effect.",
        },
        'centered': {
            'widget': 'toggle',
            'description': "Center the shift range/increments around zero instead of starting at zero.",
        },
        'seed': {
            'range':  [0, 999],
            'widget': 'int',
            'description': "Seed for the random shifts. Only used when mode='random' and fixed=True.",
            'guard':  lambda v: True if v >= 0 else ValueError("seed must be ≥ 0"),
        },
    }
    _extra_init_params = {
        "dim": {"type": "int", "default": 0, "required": True, "label": "dim (axis)",
                "description": "The tensor dimension that gets circularly shifted (e.g. the time axis)."},
        "mode": {"type": "str", "default": "increment", "required": False, "choices": ["increment", "random"],
                 "description": "increment: deterministic, growing shift per channel. random: independent random shift per channel."},
        "fixed": {"type": "bool", "default": True, "required": False,
                  "description": "random mode only: if True, shifts are drawn once per registered weight/activation and reused; if False, fresh random shifts are drawn on every call."},
    }

    def __init__(self, dim: int, shift: Optional[int] = None, mode: str = "increment",
                 seed: Union[int, BendingParameter] = 0, centered: Union[bool, BendingParameter] = False,
                 fixed: bool = True):
        super().__init__(shift=shift, centered=centered, seed=seed)
        assert mode in self.valid_modes, f"mode must be one of {self.valid_modes}, got {mode}"
        self.mode = mode
        self.fixed = bool(fixed)
        self.register_buffer('dim', torch.tensor(dim).int())
        self._shifts = torch.nn.ParameterList()
        self._shift_keys: List[str] = []
        self._shift_shapes: List[List[int]] = torch.jit.Attribute([], List[List[int]])

    def __repr__(self):
        return f"Unphase(dim={int(self.dim)}, mode={self.mode}, fixed={self.fixed})"

    def _resolve_dim(self, ndim: int) -> int:
        d = int(self.dim)
        return ndim + d if d < 0 else d

    def _num_channels(self, shape: List[int], dim: int) -> int:
        c = 1
        for i in range(len(shape)):
            if i != dim:
                c = c * shape[i]
        return c

    def _increment_shifts(self, C: int, pad: int, centered: bool) -> torch.Tensor:
        idx = torch.arange(C, dtype=torch.float32)
        if centered:
            idx = idx - float(C) / 2.
        return (idx * pad).round().long()

    def _random_shifts(self, C: int, shift: int, centered: bool) -> torch.Tensor:
        if centered:
            low = -int(math.ceil(shift / 2))
            high = int(math.floor(shift / 2))
        else:
            low = 0
            high = shift
        return torch.randint(low, high + 1, (C,))

    def _compute_shifts_for_shape(self, shape: List[int], shift: int, centered: bool) -> torch.Tensor:
        if len(shape) == 0:
            return torch.zeros(0, dtype=torch.long)
        dim = self._resolve_dim(len(shape))
        if dim >= len(shape):
            return torch.zeros(0, dtype=torch.long)
        C = self._num_channels(shape, dim)
        if self.mode == "increment":
            return self._increment_shifts(C, shift, centered)
        else:
            return self._random_shifts(C, shift, centered)

    def _roll_with_shifts(self, x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        if shifts.numel() == 0:
            return x
        dim = self._resolve_dim(x.ndim)
        if x.ndim == 0 or dim >= x.ndim:
            return x
        L = x.shape[dim]
        if L == 0:
            return x
        x_perm = x.movedim(dim, -1)
        perm_shape = x_perm.shape
        x_flat = x_perm.reshape(-1, L)
        shifts = shifts.to(device=x.device, dtype=torch.long) % L
        ar = torch.arange(L, device=x.device)
        idx = (ar.unsqueeze(0) - shifts.unsqueeze(1)) % L
        out_flat = torch.gather(x_flat, 1, idx)
        return out_flat.reshape(perm_shape).movedim(-1, dim)

    # per-target buffered shifts (mode="random", fixed=True)
    def _get_shift_from_id(self, idx: int) -> torch.Tensor:
        for i, v in enumerate(self._shifts):
            if i == idx:
                return v
        raise BendingCallbackException('%s not present in unphase shifts' % idx)

    def _get_shift_from_name(self, name: str) -> torch.Tensor:
        for i, v in enumerate(self._shifts):
            if self._shift_keys[i] == name:
                return v
        raise BendingCallbackException('name %s not present in unphase shifts' % name)

    def _init_shifts_(self, name: str, shape: List[int]):
        if name not in self._shift_keys:
            self._shift_shapes.value.append(shape)
        else:
            self._shift_shapes.value[self._shift_keys.index(name)] = shape
        not_buffered = (self.mode == "random" and not self.fixed)
        shift = self.get('shift')
        if not_buffered or shift is None:
            shifts = torch.zeros(0, dtype=torch.long)
        else:
            centered = bool(self.get('centered'))
            if self.mode == "random":
                seed = self.get('seed')
                if seed is not None:
                    torch.manual_seed(int(seed))
            shifts = self._compute_shifts_for_shape(shape, int(shift), centered)
        self._upsert_buffer(self._shifts, self._shift_keys, name, shifts)

    def update(self):
        if self.mode == "random" and not self.fixed:
            return
        shift = self.get('shift')
        centered = bool(self.get('centered'))
        if self.mode == "random":
            seed = self.get('seed')
            if seed is not None:
                torch.manual_seed(int(seed))
        for i in range(len(self._shifts)):
            shape = self._shift_shapes.value[i]
            if shift is None:
                new_shifts = torch.zeros(0, dtype=torch.long)
            else:
                new_shifts = self._compute_shifts_for_shape(shape, int(shift), centered)
            self._shifts[i].data = new_shifts.to(self._shifts[i].device)

    def register_weight(self, parameter, name=None, cache: bool = True):
        name = super().register_weight(parameter, name=name, cache=cache)
        name = name.replace('.', '_')
        self._init_shifts_(name, list(parameter.shape))

    def register_activation(self, name, shape):
        name, shape = super().register_activation(name, shape)
        name = name.replace('.', '_')
        self._init_shifts_(name, list(shape))

    def _expected_num_channels(self, x_ndim: int, shape: List[int]) -> int:
        dim = self._resolve_dim(x_ndim)
        if x_ndim == 0 or dim >= x_ndim:
            return 0
        return self._num_channels(shape, dim)

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor) -> None:
        shift = self.get('shift')
        if shift is None:
            return
        centered = bool(self.get('centered'))
        with torch.no_grad():
            if self.mode == "random" and not self.fixed:
                shifts = self._compute_shifts_for_shape(list(cache.shape), int(shift), centered)
            else:
                shifts = self._get_shift_from_id(idx)
                # the registered target's shape may no longer match `cache` (e.g. a dynamic
                # axis whose length differs from the one seen at register_weight time) — a
                # stale buffer would otherwise crash the reshape in _roll_with_shifts.
                if shifts.numel() != self._expected_num_channels(cache.ndim, list(cache.shape)):
                    shifts = self._compute_shifts_for_shape(list(cache.shape), int(shift), centered)
            if shifts.numel() == 0:
                return
            param.set_(self._roll_with_shifts(cache, shifts.to(cache.device)))

    def bend_input(self, x: torch.Tensor, shift: Optional[torch.Tensor] = None, centered: Optional[torch.Tensor] = None,
                   seed: Optional[torch.Tensor] = None, name: Optional[str] = None):
        if shift is None:
            return x
        shift_i = int(shift)
        centered_b = bool(centered) if centered is not None else False
        if (self.mode == "random" and not self.fixed) or name is None:
            shifts = self._compute_shifts_for_shape(list(x.shape), shift_i, centered_b)
        else:
            shifts = self._get_shift_from_name(name)
            # same staleness guard as apply_to_param: a dynamic/variable-length axis can make
            # the buffered shift count no longer match this call's actual shape.
            if shifts.numel() != self._expected_num_channels(x.ndim, list(x.shape)):
                shifts = self._compute_shifts_for_shape(list(x.shape), shift_i, centered_b)
        if shifts.numel() == 0:
            return x
        return self._roll_with_shifts(x, shifts.to(x.device))

