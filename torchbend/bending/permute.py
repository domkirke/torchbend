import torch
from typing import Optional, Union, List
from .parameter import BendingParameter
from .callback import BendingCallback



class Permute(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    jit_compatible = True
    nntilde_compatible = True
    controllable_params = {'seed': (int, -1)}

    def __getstate__(self):
        out_dict = dict(self.__dict__)
        del out_dict["_generator"]
        return out_dict

    def __setstate__(self, obj):
        self.__dict__.update(obj)
        self._generator = torch.Generator()
        if obj.get('seed'):
            self._set_seed(int(obj.get('seed')))

    def __init__(self, dim: int, seed: int = -1):
        super().__init__(seed=seed)
        if isinstance(self.seed, BendingParameter):
            if self.seed.as_input: 
                raise ValueError('seed cannot be fed as input in graph. Please set as_input=False')
        self._generator = torch.Generator()
        self._set_seed(int(self.get('seed')))
        self.register_buffer('dim', torch.tensor(dim).int())
        self._perms = torch.nn.ParameterList()
        self._perm_keys = []

    def __repr__(self):
        return f"Permute(dim={self.dim})"

    def _get_perm_from_name(self, name: str):
        for i, v in enumerate(self._perms):
            if self._perm_keys[i] == name:
                return v
        raise KeyError("not found in permutations : %s"%name)

    def _get_perm_from_id(self, idx: int):
        for i, v in enumerate(self._perms):
            if i == idx:
                return v
        raise KeyError(f"invalid idx: {idx}")
    
    def _init_permute_(self, name, shape):
        assert shape is not None, "mask preinit must be given target shape"
        self.dim = len(shape) + self.dim if self.dim < 0 else self.dim
        if self.dim < len(shape): 
            perm = torch.randperm(shape[self.dim], generator=self._generator, requires_grad=False)
            self._perms.append(torch.nn.Parameter(perm, requires_grad=False))
            self._perm_keys.append(name)
        else: 
            # no perm
            self._perms.append(torch.nn.Parameter(torch.Tensor([]), requires_grad=False))
            self._perm_keys.append(name)


    def _set_seed(self, seed: int | None = -1):
        if seed is None: seed = -1
        if seed == -1: 
            self._bypass = True
        else: 
            self._bypass = False
            self._generator.manual_seed(seed)
    
    def update(self):
        self._set_seed(int(self.get('seed')))
        if self._bypass:
            return
        for i, perm in enumerate(self._perms):
            if perm.numel() != 0:
                with torch.no_grad():
                    perm.set_(torch.randperm(perm.shape[0], generator=self._generator))

    def register_weight(self, parameter, name=None, cache: bool = True):
        name = super().register_weight(parameter, name=name, cache=cache) 
        name = name.replace('.', '_')
        self._init_permute_(name, parameter.shape)
            
    def register_activation(self, name, shape):
        name = super().register_activation(name, shape)
        name = name.replace('.', '_')
        self._init_permute_(name, shape)
    
    def get_permutation(self, param, name: Optional[str]) -> torch.Tensor:
        if name is not None:
            prm = self._get_perm_from_name(name)
        else:
            prm = torch.randperm(param.shape[int(self.dim)]).to(device=param.device)
        return prm 

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor) -> None:
        with torch.no_grad():
            perm = self._get_perm_from_id(idx)
            if perm.numel() == 0: return
            param.set_(torch.index_select(cache, self.dim, perm))

    def bend_input(self, x: torch.Tensor, seed: Optional[torch.Tensor] = None, name: Optional[str] = None):
        permute = self.get_permutation(x, name).to(device=x.device)
        if permute.numel() == 0 or self._bypass: 
            return x
        else:
            return torch.index_select(x, self.dim, permute)

