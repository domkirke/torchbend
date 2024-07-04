import copy
import weakref
from collections import OrderedDict
import torch
from .. import distributions as dist


## Utils
def set_callback(x, env):
    return env[x.name]


__COPY_LIST = [
    "_parameters",
    "_buffers",
    "_is_full_backward",
    "_modules",
    "_versions"
]

__COPY_HOOKS = [
    "_backward_pre_hooks",
    "_backward_hooks",
    "_forward_hooks",
    "_forward_hooks_with_kw",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kw",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
]


def get_model_copy(model, copy_parameters=False):
    """Make a bendable copy of a model, just copying internal dicts of submodules
       without deep-copying parameters"""
    model_copy = copy.copy(model)
    for attr in dir(model_copy):
        if attr == "_modules":
            continue
        if attr in __COPY_HOOKS:
            hook_dict = copy.copy(getattr(model_copy, attr))
            for k in hook_dict.keys():
                # hook_dict[k].module = weakref.ref(model_copy)
                hook_dict[k].module = lambda: model_copy
            setattr(model_copy, attr, hook_dict)
        elif attr in __COPY_LIST: 
            setattr(model_copy, attr, getattr(model_copy, attr).copy())

    if copy_parameters:
        parameters = OrderedDict([(k, copy.copy(m)) for k, m in model._parameters.items()])
        model_copy._parameters = parameters
    else:
        model_copy._parameters = model._parameters.copy()
    
    model_copy._buffers = model._buffers.copy()
    model_copy._modules = {}
    for name, mod in model._modules.items():
        if mod is not None:
            model_copy._modules[name] = get_model_copy(mod, copy_parameters=copy_parameters)
    return model_copy



class BendingError(RuntimeError):
    pass


def bend_graph(graph, callbacks, verbose=False):
    new_graph = torch.fx.Graph()
    env = {}
    bended_lookup = {}
    name_hash = {}
    for node in graph.nodes:
        new_node = new_graph.node_copy(node, lambda x: env[x.name])
        # check arguments to replace by bended node in case
        new_args = list(new_node.args)
        for i, arg in enumerate(new_args):
            if isinstance(arg, torch.fx.Node):
                if arg.name in bended_lookup:
                    new_args[i] = bended_lookup[arg.name]
        new_node.args = tuple(new_args)
        env[node.name] = new_node
        # add bending layer to graph
        #TODO using inserting_after??
        if node.name in callbacks:
            if verbose:
                print('bending activation %s with function %s...'%(node.name, callbacks[node.name]))
            # add placeholder
            # add callback
            bended_node_name = node.name+"_bended"
            hack_obj_name = node.name + "_callback"
            # if hack_obj_name not in name_hash:
            #     name_hash[hack_obj_name] = 0
            #     hack_obj_name += "_0"
            # else:
            #     idx = name_hash[hack_obj_name]
            #     name_hash[hack_obj_name] += 1
            #     hack_obj_name += f"_{idx}"
            bended_node = new_graph.create_node("call_module", hack_obj_name, args=(new_node,), kwargs={}, name=bended_node_name)
            env[bended_node_name] = bended_node 
            bended_lookup[node.name] = bended_node 
    return new_graph





# Probability handling for scripting / tracing

def _bernoulli2tensor(x: dist.Bernoulli):# -> torch.Tensor:
    return x.probs

def _normal2tensor(x: dist.Normal, temperature: float = 0.):# -> torch.Tensor:
    return x.mean + temperature * x.stddev * torch.randn_like(x.mean)

def _categorical2tensor(x: dist.Normal, return_probs: bool = False, sample: bool = False):# -> torch.Tensor:
    if return_probs:
        return x.probs
    else:
        if sample:
            return x.probs.sample()
        else:
            return x.probs.max(1)

def dist_to_tensor(target):
    if target == dist.Bernoulli:
        return _bernoulli2tensor
    elif target == dist.Normal:
        return _normal2tensor
    elif target == dist.Categorical:
        return _categorical2tensor
    else:
        raise NotImplementedError


def _get_weight_properties(args):
    name, value = args
    try:
       minval = value.min()
       maxval = value.max() 
    except ValueError:
       minval = torch.nan
       maxval = torch.nan
    try:
       meanval = value.mean()
       stdval = value.std() 
    except ValueError:
       meanval = torch.nan
       stdval = torch.nan    
    return [name, value.shape, value.dtype, minval, maxval, meanval, stdval]


def _import_to_interface(fn):
    fn.__import_to_interface = True
    return fn


