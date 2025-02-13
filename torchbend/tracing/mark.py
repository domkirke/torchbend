import torch.fx as fx
from .proxy import BendingProxy

@fx.wrap
def mark(obj, name=None):
    if isinstance(obj, BendingProxy):
        obj.tracer.register_alias(obj.node, name)
    return obj