import functools
import torch.fx as fx
from .proxy import BendingProxy

def register_alias(obj, name):
    obj.tracer.register_alias(obj.node, name)
    return obj

def mark(*args, obj=None, name=None):
    if len(args) == 1:
        assert obj is None
        obj = args[0]
    if len(args) == 2: 
        assert obj is None and name is None
        obj, name = args
    if len(args) > 2: raise RuntimeError('mark must be given obj, or obj+name')
    if obj is not None:
        if isinstance(obj, BendingProxy):
            assert obj is not None
            return register_alias(obj, name)
        else:
            return obj
    else: 
        # used as a decorator
        def mark_decorator(fn):
            if isinstance(fn, type):
                raise NotImplemented
            elif callable(fn):
                @functools.wraps(fn)
                def wrapper(*args, **kwargs):
                    out = fn(*args, **kwargs)
                    out = mark(obj=out, name=name)
                    return out
                return wrapper
            else:
                raise RuntimeError()

        return mark_decorator
