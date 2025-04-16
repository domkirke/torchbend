import functools
from typing import Any, Optional
import torch
import torch.fx as fx
import torch.nn as nn
from .proxy import BendingProxy

def register_alias(obj, name, mode):
    obj.tracer.register_alias(obj.node, name, mode)
    return obj

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


@torch.jit.ignore
def mark(obj: Optional[Any] = None, name: Optional[str] = None, mode: Optional[str] = "post") -> Any:
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
        return mark_decorator
