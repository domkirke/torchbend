import functools
from typing import Any, Optional
import torch
import torch.fx as fx
import torch.nn as nn
from .proxy import BendingProxy

def register_alias(obj, name, mode):
    obj.tracer.register_alias(obj.node, name, mode)
    return obj

def register_alias_from_nodes(tracer, nodes, name):
    tracer.register_alias_from_node(tuple(n.node for n in nodes), name)



@torch.jit.ignore
def mark_fn(obj: Optional[Any] = None, name: Optional[str] = None, mode: Optional[str] = "post") -> Any:
    if obj is not None:
        if isinstance(obj, BendingProxy):
            assert obj is not None
            return register_alias(obj, name, mode)
        if isinstance(obj, tuple):
            return 
        elif isinstance(obj, nn.Module):
            obj.__tb_register_forward_in_alias = name, mode
            return obj
        elif callable(obj):
            obj.__tb_register_forward_in_alias = name, mode
            return obj
        else:
            return obj
    else: 
        if mode == "post":
            def mark_decorator(fn):
                if callable(fn):
                    @functools.wraps(fn)
                    def wrapper(*args, **kwargs):
                        out = fn(*args, **kwargs)
                        out = mark_fn(obj=out, name=name, mode=mode)
                        return out
                    return wrapper
                else:
                    raise RuntimeError()
        elif mode == "pre":
            def mark_decorator(fn):
                if callable(fn):
                    @functools.wraps(fn)
                    def wrapper(*args, **kwargs):
                        if len(args) > 0: 
                            register_alias_from_nodes(args[0].tracer, args, name=name)
                        out = fn(*args, **kwargs)
                        return out
                    return wrapper
                else:
                    raise RuntimeError()
        # used as a decorator
        return mark_decorator


def mark(obj: Optional[Any] = None, name: Optional[str] = None, mode: Optional[str] = "post") -> torch.Tensor:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        if torch.jit.isinstance(obj, torch.Tensor):
            return obj
        else: 
            raise RuntimeError("when scripting, torchbend marking is only scriptable with tensors")
    else: 
        return mark_fn(obj=obj, name=name, mode=mode)
