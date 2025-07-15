import os
from copy import deepcopy
import torch.nn as nn
import torchbend as tb

class ModuleTestConfig():
    def __init__(self, module_class, init_args=(tuple(), dict()), callbacks_with_args=None):
        self.module_class = module_class
        self.init_args = init_args
        self.callback_with_args = callbacks_with_args or {}

    def __iter__(self):
        return iter({m: self.get_method_args(m) for m in self.get_methods()}.items())

    def __repr__(self):
        return "TestConfig(module_class=%s)"%self.module_class

    @property
    def is_scriptable(self):
        return hasattr(self.module_class, "script")

    def scriptable_methods(self):
        assert self.scriptable, "module %s is not scriptable"%type(self.module_class.__name__)

    def scriptable(self):
        outs = {m: self.callback_with_args[m] for m in self.get_methods()}.items()
        outs = list(filter(lambda x: x[0][4], outs))
        return iter({m: v[:4] for m, v in dict(outs).items()}.items())

    def get_module(self):
        return deepcopy(self.module_class)(*self.init_args[0], **self.init_args[1])

    def get_bended_module(self, module=None, trace=False):
        module = module or self.get_module()
        if isinstance(module, nn.Module):
            module = tb.BendedModule(module)
        else:
            module = tb.BendedWrapper(module)
        if trace: self.trace_module(module)
        return module

    def trace_module(self, module):
        for method in self.get_methods(): 
            args, kwargs, weights, acts = self.get_method_args(method)
            module.trace(*args, **kwargs, fn=method)

    def get_modules(self, trace=False):
        module = self.get_module()
        return module, self.get_bended_module(module, trace=trace)

    def get_methods(self):
        return list(self.callback_with_args.keys())

    def get_method_args(self, method):
        return deepcopy(self.callback_with_args[method][:4])

    def activation_targets(self, fn="forward"):
        return deepcopy(self.callback_with_args[fn][3])

    def weight_targets(self, fn=None):
        if fn is None:
            fn = list(self.callback_with_args.keys())[0]
        return deepcopy(self.callback_with_args[fn][2])

