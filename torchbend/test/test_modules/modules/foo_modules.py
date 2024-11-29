import os
import typing
import torch, torch.nn as nn
import gin
from ..module_config import ModuleTestConfig


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

    # @torch.jit.export
    def forward_dist(self, x):
        out = self.pre_conv(x)
        for i, mod in enumerate(self.module_list):
            out = mod(out)
        return torch.distributions.Normal(out, torch.ones_like(out))

    def script(self):
        return self


class ShapedFoo(Foo):
    __bended_methods__ = ['forward', 'return_shape', 'loop_on_shape']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param = nn.Parameter(torch.tensor(1.))

    @torch.jit.export 
    def return_shape(self, x):
        out = self.pre_conv(x)
        for i, mod in enumerate(self.module_list):
            out = mod(out)
        out = out + out.shape[1] * self.param 
        return out

    @torch.jit.export
    def loop_on_shape(self, x):
        out = self.pre_conv(x)
        for i, mod in enumerate(self.module_list):
            out = mod(out)
        for i in range(out.shape[0]):
            out = out + (i+1) * self.param
        return out


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
                    ),
    ModuleTestConfig(ShapedFoo, 
                     (tuple(), dict()), 
                     {
                     'forward': (
                         tuple(),
                         {"x": torch.randn(1, 1, 128)},
                         [".*weight"],
                         ["module_list_1"],
                         True
                     ),
                     'return_shape': (
                         tuple(),
                         {"x": torch.randn(1, 1, 128)},
                         [".*weight", "param"],
                         ["module_list_1"],
                         True
                     ), 
                     'loop_on_shape': (
                         tuple(),
                         {"x": torch.randn(1, 1, 128)},
                         [".*weight", "param"],
                         ["module_list_1"],
                         True
                    )
        }
    )
]


modules_to_compare = [
     ModuleTestConfig(Foo, 
                     (tuple(), dict()), 
                     {'forward_dist': (
                         tuple(),
                         {"x": torch.randn(1, 1, 128)}, 
                         [], [], True
                     )}
     ), 
]