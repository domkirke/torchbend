import os
import torch, torch.nn as nn
import rave
import gin
from .utils import ModuleTestConfig


class Foo(nn.Module):
    def __init__(self, nlayers=3):
        super().__init__()
        self.nlayers = nlayers
        self.pre_conv = nn.Conv1d(1, 16, 1)
        self.conv_modules = nn.ModuleList([nn.Conv1d(16, 16, 3) for _ in range(nlayers)])
        self.activations = nn.ModuleList([nn.Tanh() for _ in range(nlayers)])

    def forward(self, x):
        out = self.pre_conv(x)
        for i in range(self.nlayers):
            out = self.conv_modules[i](out)
            out = self.activations[i](out)
        return torch.distributions.Normal(out, torch.ones_like(out))

    def forward_nodist(self, x):
        out = self.pre_conv(x)
        for i in range(self.nlayers):
            out = self.conv_modules[i](out)
            out = self.activations[i](out)
        return out


config = os.path.join(os.path.dirname(rave.__file__), "configs", "v2.gin")
class RAVETest(rave.RAVE):
    def __new__(cls, *args, **kwargs):
        gin.parse_config_file(config)
        return super().__new__(cls, *args, **kwargs)
        
    def forward(self, x):
        return super().forward(x)


modules_to_test = [
    ModuleTestConfig(Foo, 
                     (tuple(), dict()), 
                     {
                     'forward': (
                         tuple(),
                         {"x": torch.randn(1, 1, 128)},
                         [".*weight"],
                         ["conv_modules_1"],
                         True
                     ), 
                     'forward_nodist': (
                         tuple(),
                         {"x": torch.randn(1, 1, 128)},
                         [".*weight"],
                         ["conv_modules_1"],
                         True
                     )
                    }
                    )
]


modules_to_compare = [
     ModuleTestConfig(Foo, 
                     (tuple(), dict()), 
                     {'forward_nodist': (
                         tuple(),
                         {"x": torch.randn(1, 1, 128)},
                     )}
     ), 
     ModuleTestConfig(RAVETest,
                      (tuple(), dict()),
                      {'forward': (
                          tuple(),
                          {'x': torch.randn(1, 1, 4096)},
                      )}
    )
]


__all__ = ['Foo', 'RAVETest']