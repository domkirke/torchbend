import os
import torch, torch.nn as nn
import rave
import gin
from .utils import ModuleTestConfig


class Foo(nn.Module):
    __bended_methods__ = ['forward', 'forward_dist']
    def __init__(self, nlayers=3):
        super().__init__()
        self.nlayers = nlayers
        self.pre_conv = nn.Conv1d(1, 16, 1)
        modules = []
        for i in range(nlayers):
            modules.append(nn.Conv1d(16, 16, 3))
            modules.append(nn.Tanh())
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        out = self.pre_conv(x)
        for i, mod in enumerate(self.module_list):
            out = mod(out)
        return out

    def forward_dist(self, x):
        out = self.pre_conv(x)
        for i, mod in enumerate(self.module_list):
            out = mod(out)
        return torch.distributions.Normal(out, torch.ones_like(out))

    def script(self):
        return self


class WrappedFoo(object):

    __bended_methods__ = ['forward', 'forward_dist']

    def __init__(self, nlayers=3):
        self._foo1 = Foo(nlayers)
        self._foo2 = Foo(nlayers)

    def preprocess(self, x):
        x = (x - x.mean()) / x.std()
        return x

    def forward(self, x):
        x = self.preprocess(x)
        out1 = self._foo1(x)
        out2 = self._foo2(x)
        return out1, out2

    def forward_dist(self, x):
        x = self.preprocess(x)
        out1 = self._foo1.forward_dist(x)
        out2 = self._foo2.forward_dist(x)
        return out1, out2



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
                         {"x": torch.randn(4, 1, 128)},
                         [".*weight"],
                         ["module_list_1"],
                         True
                     ), 
                     'forward_dist': (
                         tuple(),
                         {"x": torch.randn(4, 1, 128)},
                         [".*weight"],
                         ["module_list_1"],
                         True
                     )
                    }
                    ),
    ModuleTestConfig(WrappedFoo, 
                     (tuple(), dict()), 
                     {
                     'forward': (
                         tuple(),
                         {"x": torch.randn(1, 1, 128)},
                         [".*weight"],
                         ["_foo1_module_list_1", "_foo2_module_list_1"],
                         True
                     ), 
                     'forward_dist': (
                         tuple(),
                         {"x": torch.randn(1, 1, 128)},
                         [".*weight"],
                         ["_foo1_module_list_1", "_foo2_module_list_1"],
                         True
                     )
                    }
                    )
]

scriptable_modules_to_test = list(filter(lambda x: x.is_scriptable, modules_to_test))


modules_to_compare = [
     ModuleTestConfig(Foo, 
                     (tuple(), dict()), 
                     {'forward_dist': (
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


__all__ = ['Foo', 'RAVETest', 'modules_to_test', 'scriptable_modules_to_test', 'modules_to_compare']