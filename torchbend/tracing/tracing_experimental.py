import torch
from typing import Sequence
import inspect
from .tracing import ActivationProperties
from .input import Inputs
from .graph import BendedGraph
from torch.fx.experimental.proxy_tensor import make_fx as tfe_make_fx
from torch._subclasses.fake_tensor import FakeTensor


def _parse_fn_args(obj, inputs):
    kwargs = dict(**inputs)
    new_args = []
    new_kwargs = {}
    if isinstance(obj, torch.nn.Module):
        obj = obj.forward
    has_varargs = False
    has_varkwargs = False
    for i, (name, param) in enumerate(dict(inspect.signature(obj).parameters).items()):
        if param.kind in [param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD]: 
            if name in kwargs:
                new_args.append(kwargs.pop(name))
            else:
                if param.kind == param.POSITIONAL_ONLY:
                    raise TypeError('missing argument : %s'%name)
                else:
                    new_args.append(param.default)
        elif param.kind == param.KEYWORD_ONLY:
            if name in kwargs:
                new_kwargs[param.name] = kwargs.pop(name)
            else:
                new_kwargs[param.name] = param.default
        elif param.kind == param.VAR_POSITIONAL:
            has_varargs = True
        elif param.kind == param.VAR_KEYWORD:
            has_varkwargs = True

    if has_varkwargs:
        new_kwargs.update(kwargs)

    return tuple(new_args), new_kwargs



def rewire_to_original_module(module, gm, fn="forward"):

    def _rewire_node(node, param_name):
        node.target = param_name
        node.name = param_name.replace('.', '_')

    id_dict = {}
    params_and_buffers = {**dict(module.named_parameters()), **dict(module.named_buffers())}
    for k, v in params_and_buffers.items():
        n_id = id(v)
        id_dict[n_id] = k 
    unmatched_params = []
    input_names = list(inspect.signature(getattr(module, fn)).parameters)
    for n in gm.graph.nodes: 
        if n.op == "placeholder": 
            current_input_param = input_names.pop(0)
            n.target = current_input_param
            n.name = current_input_param
        elif n.op == "get_attr":
            target = n.target
            current_id = id(getattr(gm, target))
            if current_id in id_dict: 
                _rewire_node(n, id_dict[current_id])
            else:
                unmatched_params.append(n.target)
    gm.recompile()
    return torch.fx.GraphModule(module, gm.graph)

from torch._ops import OpOverload

def is_mark(n):
    if not isinstance(n.target, OpOverload): return False
    return n.target._name in ["torchbend::mark_tensor", "torchbend::mark_tensor_pre"]


def make_fx(module, inputs, fn="forward"):
    if fn == "forward":
        obj_to_trace = module
        args, kwargs = _parse_fn_args(obj_to_trace, inputs)
    else:
        def _closure(*args, **kwargs):
            return getattr(module, fn)(*args, **kwargs)
        obj_to_trace = _closure
        args, kwargs = _parse_fn_args(getattr(module, fn), inputs)
    
    traced_gm = tfe_make_fx(obj_to_trace, tracing_mode="symbolic", _allow_non_fake_inputs=True, record_module_stack=True)(*args, **kwargs)
    rewire_to_original_module(module, traced_gm)
    # traced_gm = humanize_node_names(traced_gm)
    graph = BendedGraph(from_graph=traced_gm.graph)
    graph._original_func_name = fn
    env = {}
    for n in traced_gm.graph.nodes:
        new_node = graph.node_copy(n, lambda x: env[x.name])
        env[n.name] = new_node
    activations = {k: ActivationProperties.from_node(v, fn=fn) for k, v in env.items()}
    graph._from_backend = "proxy_tensor"
    traced_gm.graph = graph
    traced_gm.graph.activations = activations

    # parse aliases
    aliases = {}
    for n in traced_gm.graph.nodes:
        if is_mark(n):
            alias_name = "aliases" if len(n.args) < 2 else n.args[1]
            if isinstance(n.args[0], Sequence):
                aliases[alias_name] = aliases.get(alias_name, []) + [[x.name for x in n.args[0]]]
            else:
                aliases[alias_name] = aliases.get(alias_name, []) + [n.args[0].name]

    traced_gm.graph.aliases = aliases
    return traced_gm, activations

