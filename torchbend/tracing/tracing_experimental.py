import torch
from types import NoneType
from ..utils import _resolve_code, _import_defs_from_tmpfile
import re

from typing import Sequence, _BaseGenericAlias
import inspect
import functools

from torch._ops import OpOverload
from .tracing import ActivationProperties
from .graph import BendedGraph
from .graphmodule import BendedGraphModule
from torch.fx.experimental.proxy_tensor import make_fx as tfe_make_fx
from torch._subclasses.fake_tensor import extract_tensor_metadata

from ..utils import _resolve_code, _import_defs_from_tmpfile



def is_sequence(typing_obj):
    if issubclass(type(typing_obj), _BaseGenericAlias):
        return typing_obj._name in ["List", "Sequence", "Tuple"]
    else:
        return False


def rewire_to_original_module(module, gm, traced_obj, fn="forward"):
    """renames input names and activations using graph, and also import annotations for correct TorchScript parsing"""

    def _rewire_node(node, param_name):
        node.target = param_name
        node.name = param_name.replace('.', '_')

    id_dict = {}
    params_and_buffers = {**dict(module.named_parameters()), **dict(module.named_buffers())}
    for k, v in params_and_buffers.items():
        n_id = id(v)
        id_dict[n_id] = k 
    unmatched_params = {}
    input_names = []
    input_types = []
    input_nodes = list(filter(lambda x: x.op == "placeholder", gm.graph.nodes))
    def n_args_for_input(i):
        outs = list(filter(lambda x, n=i: re.match(rf"arg{i}\_(\d+)", x.name), input_nodes))
        return len(outs)
    for i, (k, v) in enumerate(dict(inspect.signature(traced_obj).parameters).items()):
        if is_sequence(v.annotation):
            input_names.extend([f"{k}_{j}" for j in range(n_args_for_input(i))])
        else:
            input_names.append(k)
        input_types.append(v.annotation)
        
    for n in gm.graph.nodes: 
        if n.op == "placeholder": 
            current_input_param = input_names.pop(0)
            current_input_type = input_types.pop(0)
            n.target = current_input_param
            n.name = current_input_param
            if current_input_type != inspect._empty:
                n.type = current_input_type
        elif n.op == "get_attr":
            target = n.target
            param = getattr(gm, target)
            current_id = id(param)
            if len(n.meta) == 0:
                n.meta = {'val': param, 'tensor_meta': extract_tensor_metadata(param)}
            if current_id in id_dict: 
                _rewire_node(n, id_dict[current_id])
            else:
                new_name = f"{fn}_{n.target}"
                unmatched_params[new_name] = getattr(gm, n.target)
                n.target = new_name
        elif n.op == "call_module":
            pass

    gm.recompile()
    return unmatched_params

def is_mark(n):
    if not isinstance(n.target, OpOverload): return False
    return n.target._name in ["torchbend::mark_tensor", "torchbend::mark_tensor_pre"]



make_fx_closure_pattern = """
import torch
from typing import * 

def fn({{SIGNATURE}}){{RETURN_ANNOTATION}}:
    return module.{{FN_NAME}}({{ARGUMENTS}})
"""

ALLOW_TENSORS_IN_SIGNATURE = False
NATIVE_TYPES_HASH = [str, bool, int, NoneType, float, complex, bytes]

def _type_ok_for_signature(obj):
    if type(obj) in NATIVE_TYPES_HASH:
        return True
    elif torch.is_tensor(obj):
        if obj.numel() == 1: 
            return True
        else:
            return ALLOW_TENSORS_IN_SIGNATURE
    else:
        return False


def _value_for_signature(obj):
    if type(obj) in NATIVE_TYPES_HASH:
        return obj
    elif torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.item()
        else:
            if ALLOW_TENSORS_IN_SIGNATURE: 
                return obj.tolist()
    raise ValueError('value %s cannot be formatted for signature.'%obj)


def _parse_fn_args(obj, inputs):

    kwargs = dict(**inputs)
    new_args = [] # arguments for tracing
    new_kwargs = {} # kwarguments for tracing
    new_signature = [] # signature of closure
    new_arguments = [] # arguments of callback inside closure
    
    if isinstance(obj, torch.nn.Module):
        obj = obj.forward
    has_varargs = False
    has_varkwargs = False
    return_annotation = inspect.signature(obj).return_annotation
    return_annotation = "" if return_annotation == inspect._empty else " -> " + inspect.formatannotation(return_annotation)
    for i, (name, param) in enumerate(dict(inspect.signature(obj).parameters).items()):
        annotation = "" if param.annotation == inspect._empty else ": "+inspect.formatannotation(param.annotation) 
        if param.kind in [param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD]: 
            if name in kwargs:
                value = kwargs.pop(name)
                new_args.append(value)
                if param.default == inspect._empty:
                    new_signature.append(f"{name}{annotation}")
                    new_arguments.append(name)
                else:
                    if _type_ok_for_signature(value):
                        new_signature.append(f"{name}{annotation} = {_value_for_signature(value)}")
                    else:
                        new_signature.append(f"{name}{annotation} = {param.default}")
                    new_arguments.append(f"{name} = {name}")
            else:
                new_args.append(param.default)
                if param.default == inspect._empty:
                    raise TypeError('missing argument : %s'%name)
                else:
                    new_signature.append(f"{name}{annotation}={param.default}")
                    new_arguments.append(f"{name}={name}")

        elif param.kind == param.KEYWORD_ONLY:
            if name in kwargs:
                # new_kwargs[param.name] = kwargs.pop(name)
                value = kwargs.pop(name)
                new_args.append(value)
                if _type_ok_for_signature(value):
                    new_signature.append(f"{name}{annotation} = {_value_for_signature(value)}")
                else:
                    new_signature.append(f"{name}{annotation} = {param.default}")
                new_arguments.append(f"{name} = {name}")
            else:
                new_args.append(param.default)
                if param.default == inspect._empty:
                    # should not happen, but who knows
                    pass
                else:
                    new_signature.append(f"{name}{annotation} = {param.default}")
                    new_arguments.append(f"{name} = {name}")
        elif param.kind == param.VAR_POSITIONAL:
            has_varargs = True
        elif param.kind == param.VAR_KEYWORD:
            has_varkwargs = True
            new_signature.append("**kwargs")
            new_arguments.append("**kwargs")

    if has_varkwargs:
        new_kwargs.update(kwargs)

    return tuple(new_args), new_kwargs, new_signature, new_arguments, return_annotation 


def make_fx(module, inputs, fn="forward"):
    # if fn == "forward":
    #     obj_to_trace = module
    #     args, kwargs, _, _ = _parse_fn_args(obj_to_trace, inputs)
    # else:
    # @functools.wraps(functools.partial(getattr(module, fn), self=module))
    # def _closure(*args, **kwargs):
    #     return getattr(module, fn)(*args, **kwargs)
    # obj_to_trace = _closure
    signature = []
    arguments = []
    args, kwargs, signature, arguments, return_ann = _parse_fn_args(getattr(module, fn), inputs)
    codes = _resolve_code(make_fx_closure_pattern, 
                                    signature = ", ".join(signature), 
                                    fn_name = fn, 
                                    arguments = ", ".join(arguments), 
                                    return_annotation = return_ann)
    gl = globals() 
    gl['module'] = module
    funcs = _import_defs_from_tmpfile(codes, gl=gl, lo=locals())
    obj_to_trace = funcs['fn']
        
    traced_gm = tfe_make_fx(obj_to_trace, tracing_mode="symbolic", _allow_non_fake_inputs=True, _allow_fake_constant=True, record_module_stack=True)(*args, **kwargs)
    unmatched_params = rewire_to_original_module(module, traced_gm, obj_to_trace, fn)
    
    graph = BendedGraph(from_graph=traced_gm.graph)
    graph.add_unmatched_params(unmatched_params)
    graph._original_func_name = fn
    env = {}
    for n in traced_gm.graph.nodes:
        new_node = graph.node_copy(n, lambda x: env[x.name])
        env[n.name] = new_node
    activations = {k: ActivationProperties.from_node(v, fn=fn) for k, v in env.items()}
    graph._from_backend = "proxy_tensor"
    graph.activations = activations

    traced_gm = BendedGraphModule(module, forward=graph)

    # parse aliases
    aliases = {}
    for n in traced_gm.graph['forward'].nodes:
        if is_mark(n):
            alias_name = "aliases" if len(n.args) < 2 else n.args[1]
            if isinstance(n.args[0], Sequence):
                aliases[alias_name] = aliases.get(alias_name, []) + [[x.name for x in n.args[0]]]
            else:
                aliases[alias_name] = aliases.get(alias_name, []) + [n.args[0].name]

    traced_gm.graph['forward'].aliases = aliases
    return traced_gm, activations

