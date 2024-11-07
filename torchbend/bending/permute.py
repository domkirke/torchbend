import torch
from typing import Optional, Union, List
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

    def register_parameter(self, parameter, name=None):
        name = super().register_parameter(parameter, name=name)      
        self._init_permute_(name, parameter.shape)
            
    def register_activation(self, name, shape):
        name = super().register_activation(name, shape)
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



class ThresholdPermute(BendingCallback):
    activation_compatible = True
    controllable_params = ['threshold']
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

        
        