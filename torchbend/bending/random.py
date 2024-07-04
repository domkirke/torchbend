import torch
from typing import Optional
from .base import BendingCallback


class Normal(BendingCallback):
    valid_ops = ['add', 'mul']

    def __getstate__(self):
        out_dict = dict(self.__dict__)
        del out_dict["generator"]
        return out_dict

    def __setstate__(self, obj):
        self.__dict__.update(obj)
        self.generator = torch.Generator()
        if obj.get('seed'):
            self.generator.manual_seed(obj.get('seed'))

    def __init__(self, std: float = 0.3, seed: int = None, op = "add"):
        super().__init__()
        self.std = std
        self.seed = seed
        assert op in self.valid_ops
        self.op = op
        self.generator = torch.Generator()
        if self.seed is not None:
            self.generator.manual_seed(self.seed)

    def __repr__(self):
        rp =  f"Normal(std={self.std:.3f}"
        if self.seed is not None:
            rp += f", seed={self.seed}"
        rp+=")"
        return rp
    
    def _init_rnd_(self, name, shape):
        assert shape is not None, "mask preinit must be given target shape"
        buffer_name = "rnd_"+name.replace('.', '_')
        mask = torch.randn(shape, generator=self.generator) * self.std
        self.register_buffer(buffer_name, mask)

    def add_bending_target(self, name, shape=None):
        super(Normal, self).add_bending_target(name, shape=shape)
        if shape is not None:
            self._init_rnd_(name, shape)
    
    def get_noise(self, param, name: Optional[str]) -> torch.Tensor:
        if name is not None:
            mask = self._buffers["rnd_"+name]
        else:
            mask = torch.randn_like(param).to(param) * self.std
        return mask

    def forward(self, param: torch.Tensor, name: Optional[str] = None):
        noise = self.get_noise(param, name).to(param)
        if self.op == "mul":
            return param * noise
        elif self.op == "add":
            return param + noise