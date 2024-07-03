from typing import Literal
import torch
import os, sys
from enum import Enum

libpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..", "..", "..")))
if libpath not in sys.path:
    sys.path.append(libpath)
import torchbend as tb
from test_modules import Foo, RAVETest


class StateDictException():
    pass

def tensor_eq(t1, t2):
    return t2.eq(t1).all()

def tensor_allclose(t1, t2):
    return torch.allclose(t1, t2)

def compare_state_dict_tensors(dict1, dict2):
    keys1, keys2 = set(dict1.keys()), set(dict2.keys())
    if keys1 != keys2: raise StateDictException("dict keys do not match with difference : %s", keys1.difference(keys2))
    result = True
    for k, v in dict1.items():
        result = result and (tensor_allclose(v, dict2[k]))
    return result

class ComparisonResult(Enum):
    Equal = 0
    DifferentTypes = 1
    DifferentLength = 2
    DifferentKeys = 3
    def __bool__(self) -> Literal[True]:
        return self == ComparisonResult.Equal
    

def compare_outs(out1, out2, allow_subclasses=False):
    if isinstance(out1, (list, tuple)):
        if not isinstance(out2, type(out1)): return ComparisonResult.DifferentTypes
        if len(out1) != len(out2): return ComparisonResult.DifferentLength
        return False not in [compare_outs(out1[i], out2[i]) for i in range(len(out1))]
    elif isinstance(out1, dict):
        if not isinstance(out2, dict): return ComparisonResult.DifferentTypes
        keys1, keys2 = set(out1.keys()), set(out2.keys())
        if len(keys1.difference(keys2)) + len(keys2.difference(keys1)) != 0: return ComparisonResult.DifferentKeys
        return False not in [compare_outs(out1[k], out2[k]) for k in keys1]
    elif isinstance(out1, (tb.distributions.Distribution, torch.distributions.Distribution)):
        if not isinstance(out2, type(out1)): return ComparisonResult.DifferentTypes
        return compare_outs(tb.distributions.utils.convert_from_torch(out1).as_tuple(), 
                            tb.distributions.utils.convert_from_torch(out2).as_tuple())
    else:
        if allow_subclasses:
            if not isinstance(out1, type(out2)): return ComparisonResult.DifferentTypes
        else:
            if (type(out1) != type(out2)):return ComparisonResult.DifferentTypes
        
        if torch.is_tensor(out1):
            return tensor_allclose(out1, out2)
        elif isinstance(out1, (float, int, str, complex, bool, bytes, bytearray)):
            return out1 == out2
        

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
                     {'forward': (
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

                     )}
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