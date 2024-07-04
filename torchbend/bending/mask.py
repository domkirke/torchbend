import torch
from typing import Optional
from .base import BendingCallback



class Mask(BendingCallback):

    def __getstate__(self):
        out_dict = dict(self.__dict__)
        del out_dict["generator"]
        return out_dict


    def __setstate__(self, obj):
        self.__dict__.update(obj)
        self.generator = torch.Generator()
        self.generator.manual_seed(obj.get('seed'))

    def __init__(self, prob: float = 0.3, seed: int = None):
        super().__init__()
        self.prob = prob
        self.seed = seed
        self.generator = torch.Generator()
        if self.seed is not None:
            self.generator.manual_seed(self.seed)

    def __repr__(self):
        return f"Mask(prob={self.prob:.3f})"
    
    def _init_mask_(self, name, shape):
        assert shape is not None, "mask preinit must be given target shape"
        buffer_name = "mask_"+name.replace('.', '_')
        mask = torch.bernoulli(torch.full(size=shape, fill_value=self.prob), generator=self.generator)
        self.register_buffer(buffer_name, mask)

    def add_bending_target(self, name, shape=None):
        super(Mask, self).add_bending_target(name, shape=shape)
        if shape is not None:
            self._init_mask_(name, shape)
    
    def get_mask(self, param, name: Optional[str]) -> torch.Tensor:
        if name is not None:
            mask = self._buffers["mask_"+name]
        else:
            mask = torch.bernoulli(torch.full_like(param, fill_value=self.prob)).to(param)
        return mask

    def forward(self, param: torch.Tensor, name: Optional[str] = None):
        mask = self.get_mask(param, name).to(param)
        return param * mask
        
        
                  

        