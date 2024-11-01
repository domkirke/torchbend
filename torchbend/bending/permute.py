import torch
from typing import Optional
from .base import BendingCallback



class Permute(BendingCallback):

    def __getstate__(self):
        out_dict = dict(self.__dict__)
        del out_dict["generator"]
        return out_dict

    def __setstate__(self, obj):
        self.__dict__.update(obj)
        self.generator = torch.Generator()
        if obj.get('seed'):
            self.generator.manual_seed(obj.get('seed'))

    def __init__(self, dim: int, seed: int = None):
        super().__init__()
        self.register_buffer('dim', torch.tensor(dim).int())
        self.seed = seed
        self.generator = torch.Generator()
        if self.seed is not None:
            self.generator.manual_seed(self.seed)

    def __repr__(self):
        return f"Permute(dim={self.dim})"
    
    def _init_permute_(self, name, shape):
        assert shape is not None, "mask preinit must be given target shape"
        buffer_name = "prm_"+name.replace('.', '_')
        perm = torch.randperm(shape[self.dim], generator=self.generator)
        self.register_buffer(buffer_name, perm.int())

    def add_bending_target(self, name, shape=None):
        super(Permute, self).add_bending_target(name, shape=shape)
        if shape is not None:
            self._init_permute_(name, shape)
    
    def get_permutation(self, param, name: Optional[str]) -> torch.Tensor:
        if name is not None:
            prm = self._buffers["prm_"+name]
        else:
            prm = torch.randperm(param.shape[self.dim]).to(device=param.device)
        return prm 

    def bend_input(self, param: torch.Tensor, name: Optional[str] = None):
        permute = self.get_permutation(param, name).to(device=param.device)
        dim = self.dim.to(device=param.device)
        return torch.index_select(param, dim, permute)
        
        