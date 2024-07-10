from typing import Literal
import torch
import os, sys
from enum import Enum

libpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..", "..", "..")))
if libpath not in sys.path:
    sys.path.append(libpath)
import torchbend as tb
from test_modules import Foo, RAVETest

class ModuleTestConfig():
    def __init__(self, module_class, init_args=(tuple(), dict()), callbacks_with_args={}):
        self.module_class = module_class
        self.init_args = init_args
        self.callback_with_args = callbacks_with_args

    def get_module(self):
        return self.module_class(*self.init_args[0], **self.init_args[1])

    def get_methods(self):
        return list(self.callback_with_args.keys())

    def get_method_args(self, method):
        return self.callback_with_args[method]

    def activation_targets(self, fn="forward"):
        return self.callback_with_args[fn][3]

    def weight_targets(self, fn="forward"):
        return self.callback_with_args[fn][2]


modules_to_test = [
    ModuleTestConfig(Foo, 
                     (tuple(), dict()), 
                     {
                     'forward': (
                         tuple(),
                         {"x": torch.randn(1, 1, 128)},
                         [".*"],
                         ["conv_modules_1"]
                     ), 
                     'forward_nodist': (
                         tuple(),
                         {"x": torch.randn(1, 1, 128)},
                         [".*"],
                         ["conv_modules_1"]

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