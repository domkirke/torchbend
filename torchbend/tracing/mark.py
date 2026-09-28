import functools
from typing import Any, Optional
import torch
import torch.fx as fx
import torch.nn as nn
from .proxy import BendingProxy
from .utils import register_torchscript_dispatch
from torch._subclasses.fake_tensor import FakeTensor


    


def encode_info(description=None, meta=None) -> Optional[str]:
    """A mark's description and metadata, as the JSON string the op carries.

    The op's schema only takes strings, and a mark has to survive tracing as an
    op argument -- so what it says about a tensor travels as one string. A
    value JSON cannot hold is kept as its ``repr``.
    """
    if description is None and not meta:
        return None
    import json
    return json.dumps({"description": description, "meta": dict(meta or {})}, default=repr)


def decode_info(info) -> dict:
    """``{"description", "meta"}`` from what :func:`encode_info` made."""
    if not info:
        return {"description": None, "meta": {}}
    import json
    try:
        data = json.loads(info)
    except (TypeError, ValueError):
        return {"description": str(info), "meta": {}}
    return {"description": data.get("description"), "meta": dict(data.get("meta") or {})}


def add_annotation(annotations: dict, node_name: str, alias=None, info=None, mode="post"):
    """Record one mark on ``node_name`` in a graph's ``annotations`` table.

    ``annotations`` maps a node to ``{"alias", "aliases", "description",
    "meta", "mode"}``. A node marked twice keeps every alias, the last
    description given, and the union of the metadata.
    """
    decoded = decode_info(info)
    entry = annotations.setdefault(node_name, {"alias": None, "aliases": [], "description": None,
                                               "meta": {}, "mode": mode})
    if alias and alias not in entry["aliases"]:
        entry["aliases"].append(alias)
        entry["alias"] = entry["alias"] or alias
    if decoded["description"]:
        entry["description"] = decoded["description"]
    entry["meta"].update(decoded["meta"])
    return entry


def register_alias(obj, name, mode, info=None):
    if isinstance(obj, BendingProxy):
        obj.tracer.register_alias(obj.node, name, mode, info=info)
    elif isinstance(obj, FakeTensor):
        pass
    else:
        raise TypeError('register_alias got %s'%type(obj))
    return obj

def register_alias_from_nodes(tracer, nodes, name, info=None):
    tracer.register_alias_from_node(tuple(n.node for n in nodes), name, info=info)

from typing import Sequence, List

# `info` is the mark's description and metadata as JSON (see encode_info): it
# rides along as an op argument, so it is in the traced graph like the name.
@torch.library.custom_op("torchbend::mark_tensor", mutates_args=())
def mark_tensor(obj: torch.Tensor, name: Optional[str] = None, info: Optional[str] = None) -> torch.Tensor:
    return obj.clone()

@mark_tensor.register_fake
def _(obj: torch.Tensor, name: Optional[str] = None, info: Optional[str] = None) -> torch.Tensor:
    return torch.empty_like(obj)

@register_torchscript_dispatch(mark_tensor)
def mark_ts(obj: torch.Tensor, name: Optional[str] = None, info: Optional[str] = None) -> torch.Tensor:
    return obj



@torch.library.custom_op("torchbend::mark_tensor_pre", mutates_args=())
def mark_tensor_pre(obj: Sequence[torch.Tensor], name: Optional[str] = None, info: Optional[str] = None) -> List[torch.Tensor]:
    return type(obj)([o.clone() for o in obj])

@mark_tensor_pre.register_fake
def _(obj: Sequence[torch.Tensor], name: Optional[str] = None, info: Optional[str] = None) -> List[torch.Tensor]:
    return tuple([torch.empty_like(o) for o in obj])

@register_torchscript_dispatch(mark_tensor_pre)
def mark_pre(obj: List[torch.Tensor], name: Optional[str] = None, info: Optional[str] = None) -> List[torch.Tensor]:
    return obj


@torch.jit.ignore
def mark_fn(obj: Optional[Any] = None, name: Optional[str] = None, mode: Optional[str] = "post",
            info: Optional[str] = None) -> Any:
    if obj is not None:
        if isinstance(obj, BendingProxy):
            assert obj is not None
            return register_alias(obj, name, mode, info=info)
        elif isinstance(obj, FakeTensor):
            return mark_tensor(obj, name, info)
        if isinstance(obj, tuple):
            return 
        elif isinstance(obj, type):
            assert issubclass(obj, nn.Module), "mark class decorator only works with nn.Module subclasses"
            # obj.__tb_register_forward_in_alias = name, mode
            _original_forward = obj.forward
            @functools.wraps(obj.forward)
            def _fn_wrapper_post(*args, **kwargs):
                return mark_tensor(_original_forward(*args, **kwargs), name=name, info=info)
            @functools.wraps(obj.forward)
            def _fn_wrapper_pre(self, *args, **kwargs):
                args = mark_tensor_pre(args, name=name, info=info)
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
                return mark_tensor(_original_forward(*args, **kwargs), name=name, info=info)
            @functools.wraps(obj.forward)
            def _fn_wrapper_pre(*args, **kwargs):
                args = mark_tensor_pre(args, name=name, info=info)
                return _original_forward(*args, **kwargs)
            # obj.__tb_register_forward_in_alias = name, mode
            if mode == "pre": obj.forward = _fn_wrapper_pre
            if mode == "post": obj.forward = _fn_wrapper_post
            return obj
        elif callable(obj):
            @functools.wraps(obj)
            def _fn_wrapper_post(*args, **kwargs):
                return mark_tensor(obj(*args, **kwargs), name=name, info=info)
            @functools.wraps(obj)
            def _fn_wrapper_pre(*args, **kwargs):
                args = mark_tensor_pre(args, name=name, info=info)
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
                        out = mark_fn(obj=out, name=name, mode=mode, info=info)
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
                            register_alias_from_nodes(args[0].tracer, args, name=name, info=info)
                        out = fn(*args, **kwargs)
                        return out
                    return wrapper
                else:
                    raise RuntimeError()
        # used as a decorator
        return mark_decorator



def mark(obj: Optional[torch.Tensor] = None, name: Optional[str] = None,
         mode: Optional[str] = "post", description: Optional[str] = None,
         meta: Any = None) -> torch.Tensor:
    """Tag a value so it survives tracing: an alias, a description, metadata.

    Usable from inside model code in three ways::

        z = mark(z, name="latent")          # tag a tensor

        @mark(name="decoded")               # decorator: tags the return value
        def decode(self, z): ...

        @mark(name="inputs", mode="pre")    # decorator: tags the arguments

    Every part is optional:

    name
        An alias. The value can then be bent or filtered as ``#name``, and
        ``BendedModule.aliases()`` lists it. Leave it out to annotate a value
        without making it a target.
    description
        What the value is, in words -- shown next to it in the graph viewer.
    meta
        Anything else worth knowing about it, as a dict: a stage, a unit, a
        rate, what its axes mean -- ``meta={"step": 2, "axes": "batch, tokens,
        features"}``. Kept as JSON; a value JSON cannot hold is kept as its
        ``repr``. (A dict rather than keyword arguments, so that a model
        calling ``mark`` can still be scripted: TorchScript compiles this
        signature, and has no ``**kwargs``.) The graph viewer reads ``title``
        as the value's short name and ``step`` as its place in the process.

    ``BendedModule.annotations()`` returns everything marked, by node, with its
    alias (if any), description and metadata.

    With the proxy_tensor backend the call is recorded as a custom op
    (``torchbend::mark_tensor``) and converted at finalization; with the
    vanilla tracer the proxy's node is registered directly. Outside tracing the
    function is a passthrough (and a tensor-only no-op under TorchScript), so it
    is safe to leave in production model code.
    """
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        if torch.jit.isinstance(obj, torch.Tensor):
            return obj
        else: 
            raise RuntimeError("when scripting, torchbend marking is only scriptable with tensors")
    else: 
        if obj is None: 
            return functools.partial(mark, name=name, mode=mode, description=description, meta=meta)
        return mark_fn(obj=obj, name=name, mode=mode, info=encode_info(description, meta))
