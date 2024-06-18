import torch.nn as nn
from collections import OrderedDict


class BendingCallback(nn.Module):
    def __init__(self):
        super().__init__()
        self._bending_targets = OrderedDict()

    def add_bending_target(self, name, shape=None):
        self._bending_targets[name] = shape

    def reset(self):
        for k in self._bending_targets.keys():
            self._bending_targets[k] = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class CallbackChain(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.callbacks = nn.ModuleList(*args)

    def forward(self, x, name=None, **kwargs):
        for i, m in enumerate(self.callbacks):
            x = m(x, name=name, **kwargs)
        return x 

