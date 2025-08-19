import functools
from typing import Any, Optional
import torch
import torch.fx as fx
import torch.nn as nn
from .proxy import BendingProxy
from torch._subclasses.fake_tensor import FakeTensor


    


def register_alias(obj, name, mode):
    if isinstance(obj, BendingProxy):
        obj.tracer.register_alias(obj.node, name, mode)
    elif isinstance(obj, FakeTensor):
        pass
    else:
        raise TypeError('register_alias got %s'%type(obj))
    return obj

def register_alias_from_nodes(tracer, nodes, name):
    tracer.register_alias_from_node(tuple(n.node for n in nodes), name)

from typing import Sequence, List

@torch.library.custom_op("torchbend::mark_tensor", mutates_args=())
def mark_tensor(obj: torch.Tensor, name:Optional[str] = None) -> torch.Tensor:
    return obj.clone()

@mark_tensor.register_fake
def _(obj: torch.Tensor, name:Optional[str] = None) -> torch.Tensor:
    return torch.empty_like(obj)

@torch.library.custom_op("torchbend::mark_tensor_pre", mutates_args=())
def mark_tensor_pre(obj: Sequence[torch.Tensor], name:Optional[str] = None) -> List[torch.Tensor]:
    return obj.clone()

@mark_tensor_pre.register_fake
def _(obj: Sequence[torch.Tensor], name:Optional[str] = None) -> Sequence[torch.Tensor]:
    return tuple([torch.empty_like(o) for o in obj])


@torch.jit.ignore
def mark_fn(obj: Optional[Any] = None, name: Optional[str] = None, mode: Optional[str] = "post") -> Any:
    if obj is not None:
        if isinstance(obj, BendingProxy):
            assert obj is not None
            return register_alias(obj, name, mode)
        elif isinstance(obj, FakeTensor):
            return mark_tensor(obj, name)
        if isinstance(obj, tuple):
            return 
        elif isinstance(obj, type):
            assert issubclass(obj, nn.Module), "mark class decorator only works with nn.Module subclasses"
            # obj.__tb_register_forward_in_alias = name, mode
            _original_forward = obj.forward
            @functools.wraps(obj.forward)
            def _fn_wrapper_post(*args, **kwargs):
                return mark_tensor(_original_forward(*args, **kwargs), name=name)
            @functools.wraps(obj.forward)
            def _fn_wrapper_pre(self, *args, **kwargs):
                args = mark_tensor_pre(args, name=name)
                return _original_forward(self, *args, **kwargs)
            # obj.__tb_register_forward_in_alias = name, mode
            if mode == "pre": obj.forward = _fn_wrapper_pre
            if mode == "post": obj.forward = _fn_wrapper_post
            return obj
        elif isinstance(obj, nn.Module):
            # obj.__tb_register_forward_in_alias = name, mode
            _original_forward = obj.forward
            @functools.wraps(obj.forward)
            def _fn_wrapper_post(*args, **kwargs):
                return mark_tensor(_original_forward(*args, **kwargs), name=name)
            @functools.wraps(obj.forward)
            def _fn_wrapper_pre(*args, **kwargs):
                args = mark_tensor_pre(args, name=name)
                return _original_forward(*args, **kwargs)
            # obj.__tb_register_forward_in_alias = name, mode
            if mode == "pre": obj.forward = _fn_wrapper_pre
            if mode == "post": obj.forward = _fn_wrapper_post
            return obj
        elif callable(obj):
            @functools.wraps(obj)
            def _fn_wrapper_post(*args, **kwargs):
                return mark_tensor(obj(*args, **kwargs), name=name)
            @functools.wraps(obj)
            def _fn_wrapper_pre(*args, **kwargs):
                args = mark_tensor_pre(args, name=name)
                return obj(*args, **kwargs)
            # obj.__tb_register_forward_in_alias = name, mode
            if mode == "pre": return _fn_wrapper_pre
            if mode == "post": return _fn_wrapper_post
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



def mark(obj: Optional[torch.Tensor] = None, name: Optional[str] = None, mode: Optional[str] = "post") -> torch.Tensor:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        if torch.jit.isinstance(obj, torch.Tensor):
            return obj
        else: 
            raise RuntimeError("when scripting, torchbend marking is only scriptable with tensors")
    else: 
        if obj is None: 
            return functools.partial(mark, name=name, mode=mode)
        return mark_fn(obj=obj, name=name, mode=mode)
