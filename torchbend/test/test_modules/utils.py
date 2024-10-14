import os

class ModuleTestConfig():
    def __init__(self, module_class, init_args=(tuple(), dict()), callbacks_with_args=None):
        self.module_class = module_class
        self.init_args = init_args
        self.callback_with_args = callbacks_with_args or {}

    def __iter__(self):
        return iter({m: self.get_method_args(m) for m in self.get_methods()}.items())

    def scriptable(self):
        outs = {m: self.callback_with_args[m] for m in self.get_methods()}.items()
        outs = list(filter(lambda x: x[0][4], outs))
        return iter({m: v[:4] for m, v in dict(outs).items()}.items())

    def get_module(self):
        return self.module_class(*self.init_args[0], **self.init_args[1])

    def get_methods(self):
        return list(self.callback_with_args.keys())

    def get_method_args(self, method):
        return self.callback_with_args[method][:4]

    def activation_targets(self, fn="forward"):
        return self.callback_with_args[fn][3]

    def weight_targets(self, fn="forward"):
        return self.callback_with_args[fn][2]

