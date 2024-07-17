import torch
from typing import Optional, List, Tuple

from torch.nn.parameter import Parameter as Parameter
from .base import BendingCallback



#TODO : (philippe's idea), thresholding activations keeping N% of maximum amplitudes
#TODO : register a quantile function to have dynamic masking from the prob parameter (either ordered permutation, or beta distribution over mean and selection trough parameter)

class Mask(BendingCallback):
    weight_compatible = True
    activation_compatible = True
    nntilde_compatible = True

    def __init__(self, prob: float = 0.3, seed: int = None):
        super().__init__()
        self._register_controllable_param("prob", prob)
        self.seed = seed
        self.generator = torch.Generator()
        self._masks = torch.nn.ParameterList()
        self._mask_names = []
        self._hash = {}
        if self.seed is not None:
            self.generator.manual_seed(self.seed)

    def __getstate__(self):
        out_dict = dict(self.__dict__)
        del out_dict["generator"]
        return out_dict

    def __setstate__(self, obj):
        self.__dict__.update(obj)
        self.generator = torch.Generator()
        if obj.get('seed'):
            self.generator.manual_seed(obj.get('seed'))

    def __repr__(self):
        return f"Mask(prob={float(self.prob):.3f})"
    
    def _init_mask(self, shape: List[int]):
        #TODO generator not scriptable
        mask = torch.bernoulli(torch.full(size=shape, fill_value=float(self.prob)))#, generator=self.generator)
        return mask

    def _add_mask(self, name, mask):
        self._masks.append(mask)
        self._mask_names.append(name)

    def _mask_from_name(self, name: str) -> torch.Tensor:
        for i, m in enumerate(self._masks):
            if self._mask_names[i] == name:
                return m
        raise RuntimeError('does not have mask for name %s'%name)
        
    def _register_shape(self, name, shape):
        super(Mask, self)._register_shape(name, shape)
        name = name.replace('.', '_')
        mask = self._init_mask(shape)
        self._add_mask(name, mask)

    def _register_parameter(self, parameter: List[Parameter], name=None):
        super()._register_parameter(parameter)
        mask = self._init_mask(parameter.shape)
        if name is None:
            name = self._generate_parameter_name()
        else:
            name = name.replace(".", "_")
        self._add_mask(name, mask)

    def get_mask(self, param, name: Optional[str]) -> torch.Tensor:
        if name is not None:
            return self._mask_from_name(name)
        else:
            mask = torch.bernoulli(torch.full_like(param, fill_value=float(self.prob))).to(param)
        return mask

    def update(self):
        for i, v in enumerate(self._masks):
            v.set_(self._init_mask(v.shape))

    def apply(self, update:bool=True):
        if update:
            self.update()
        prob = self.prob.get_value()

    def forward(self, param: torch.Tensor, name: Optional[str] = None):
        mask = self.get_mask(param, name).to(param)
        return param * mask
        
        
                  

        