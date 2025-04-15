import functools
import torch.fx as fx
import torch.nn as nn
from .proxy import BendingProxy

def register_alias(obj, name, mode):
    obj.tracer.register_alias(obj.node, name, mode)
    return obj

def mark(*args, obj=None, name=None, mode="post"):
    if len(args) == 1:
        assert obj is None
        obj = args[0]
    if len(args) == 2: 
        assert obj is None and name is None
        obj, name = args
    if len(args) == 3:
        obj, name, mode = args
    if len(args) > 3: raise RuntimeError('mark must be given obj, or obj+name')
    if obj is not None:
        if isinstance(obj, BendingProxy):
            assert obj is not None
            return register_alias(obj, name, mode)
        elif isinstance(obj, nn.Module):
            obj.__tb_register_forward_in_alias = name, mode
            return obj
        else:
            return obj
    else: 
        # used as a decorator
        def mark_decorator(fn):
            if callable(fn):
                @functools.wraps(fn)
                def wrapper(*args, **kwargs):
                    out = fn(*args, **kwargs)
                    out = mark(obj=out, name=name)
                    return out
                return wrapper
            else:
                raise RuntimeError()

        return mark_decorator
