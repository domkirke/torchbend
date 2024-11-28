import copy, re, os
from IPython.core.display import HTML
from IPython import display as ipython_display
import pandas as pd
from typing import Any, Union, Dict, List
import random
from enum import Enum
from typing import Literal
import weakref
from collections import OrderedDict
import torch
from .. import distributions as dist
from ..bending.parameter import BendingParameter, get_param_type
from ..bending.config import BendingConfig
from ..utils import get_parameter


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
            obj = getattr(model_copy, attr)
            if isinstance(obj, torch.jit._script.OrderedDictWrapper):
                setattr(model_copy, attr, copy.copy(obj))
            else:
                setattr(model_copy, attr, obj.copy())

    if copy_parameters:
        parameters = OrderedDict([(k, copy.copy(m)) for k, m in model._parameters.items()])
        model_copy._parameters = parameters
    else:
        model_copy._parameters = model._parameters.copy()
    
    if isinstance(model_copy._buffers, torch.jit._script.OrderedDictWrapper):
        model_copy._buffers = copy.copy(model_copy._buffers)
    else:
        model_copy._buffers = model._buffers.copy()

    modules = {}
    for name, mod in model._modules.items():
        if mod is not None:
            modules[name] = get_model_copy(mod, copy_parameters=copy_parameters)
    if isinstance(model_copy._modules, torch.jit._script.OrderedModuleDict):
        model_copy._modules = torch.jit._script.OrderedModuleDict(model_copy._c, modules)
    else:
        model_copy._modules = modules
    return model_copy



class BendingError(RuntimeError):
    pass





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
       
    meanval = value.float().mean()
    stdval = torch.nan if value.numel() == 1 else value.float().std() 
    return [name, value.shape, value.dtype, minval, maxval, meanval, stdval]


def _import_to_interface(fn):
    fn.__import_to_interface = True
    return fn



class StateDictException(Exception):
    pass

def tensor_eq(t1, t2):
    return t2.eq(t1).all()

def tensor_allclose(t1, t2, rtol=1e-05, atol=1e-08):
    return torch.allclose(t1, t2, rtol=rtol, atol=atol)


class ComparisonResult(Enum):
    NotEqual = 0
    Equal = 1
    AlmostEqual = 2
    DifferentTypes = 10
    DifferentLength = 11
    DifferentKeys = 12
    SimilarTypes = 21
    def __bool__(self) -> Literal[True]:
        return self in [ComparisonResult.AlmostEqual, ComparisonResult.Equal, ComparisonResult.SimilarTypes]

ALMOST_EQUAL_THRESHOLD = 1.e-5

def compare_outs(out1, out2, allow_subclasses=False, allow_almost_equal=False):
    if isinstance(out1, (list, tuple)):
        if not isinstance(out2, type(out1)): return ComparisonResult.DifferentTypes
        if len(out1) != len(out2): return ComparisonResult.DifferentLength
        if ComparisonResult.NotEqual in [compare_outs(out1[i], out2[i], allow_subclasses=allow_subclasses, allow_almost_equal=allow_almost_equal) for i in range(len(out1))]:
            return ComparisonResult.NotEqual
        else:
            return ComparisonResult.Equal
    elif isinstance(out1, dict):
        if not isinstance(out2, dict): return ComparisonResult.DifferentTypes
        keys1, keys2 = set(out1.keys()), set(out2.keys())
        if len(keys1.difference(keys2)) + len(keys2.difference(keys1)) != 0: return ComparisonResult.DifferentKeys
        if ComparisonResult.NotEqual in [compare_outs(out1[k], out2[k], allow_subclasses=allow_subclasses, allow_almost_equal=allow_almost_equal) for k in keys1]:
            return ComparisonResult.NotEqual
        else:
            return ComparisonResult.Equal
    elif isinstance(out1, (dist.Distribution, torch.distributions.Distribution)):
        if not isinstance(out2, type(out1)): 
            out1 = dist.convert_to_torch(out1)
            out2 = dist.convert_to_torch(out2)
            res = compare_outs(dist.utils.convert_from_torch(out1).as_tuple(), 
                                dist.utils.convert_from_torch(out2).as_tuple(),
                                allow_subclasses=allow_subclasses, allow_almost_equal=allow_almost_equal)
            if bool(res):
                return ComparisonResult.SimilarTypes
            else:
                return ComparisonResult.NotEqual
        else:
            return compare_outs(dist.utils.convert_from_torch(out1).as_tuple(), 
                                dist.utils.convert_from_torch(out2).as_tuple(), 
                                allow_subclasses=allow_subclasses, allow_almost_equal=allow_almost_equal)
    else:
        if allow_subclasses:
            if not isinstance(out1, type(out2)): return ComparisonResult.DifferentTypes
        else:
            if (type(out1) != type(out2)):return ComparisonResult.DifferentTypes
        
        if torch.is_tensor(out1):
            if tensor_allclose(out1, out2):
                return ComparisonResult.Equal
            elif (tensor_allclose(out1, out2, atol=ALMOST_EQUAL_THRESHOLD) and allow_almost_equal):
                return ComparisonResult.AlmostEqual
            else:
                return ComparisonResult.NotEqual
        elif isinstance(out1, (float, int, str, complex, bool, bytes, bytearray)):
            if out1 == out2:
                return ComparisonResult.Equal
            elif (tensor_allclose(out1, out2, atol=ALMOST_EQUAL_THRESHOLD) and allow_almost_equal):
                return ComparisonResult.AlmostEqual
            else:
                return ComparisonResult.NotEqual
        
def compare_state_dict_tensors(dict1, dict2):
    keys1, keys2 = set(dict1.keys()), set(dict2.keys())
    if keys1 != keys2: raise StateDictException("dict keys do not match with difference : %s", keys1.difference(keys2))
    result = True
    for k, v in dict1.items():
        result = result and bool(compare_outs(v, dict2[k]))
    return result

def identity(x: Any) -> Any:
    return x

    
def make_graph_jit_compatible(graph: torch.fx.Graph):
    #TODO
    nodes = {}
    # new_graph = torch.fx.Graph()
    # _, out = new_graph.graph_copy(graph, nodes, return_output_node=True)
    # new_graph.output(out)
    for n in graph.nodes:
        if n.op == "call_function" and n.target == dist.convert_to_torch:
            n.target = identity
    return graph 

def _bending_config_from_dicts(*dicts, module=None):
    bending_config = BendingConfig(module=module)
    for d in dicts:
        for key, cbs in d.items():
            for cb in cbs:
                bending_config.append((cb, key))
    return bending_config

def clone_parameters(module_or_dict: Union[Dict, torch.nn.Module], params: List[str]):
    if isinstance(module_or_dict, torch.nn.Module):
        for p in params:
            get_parameter(module_or_dict, p).set_(get_parameter(module_or_dict, p).data.clone())
    elif isinstance(module_or_dict, dict):
        for p in params:
            module_or_dict[p] = module_or_dict[p].clone()
    else:
        raise BendingError('clone_parameters only takes module or dictionaries as inputs, got : %s'%(type(module_or_dict)))

def state_dict(obj):
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()
    else:
        state_dict = {}
        for attr_name in dir(obj):
            submodule = getattr(obj, attr_name)
            if not isinstance(submodule, torch.nn.Module): continue
            state_dict.update({attr_name+"."+k: v for k, v in submodule.state_dict().items()})
        return state_dict
    
def named_parameters(obj):
    if isinstance(obj, torch.nn.Module):
        return dict(obj.named_parameters())
    else:
        named_parameters = {}
        for attr_name in dir(obj):
            submodule = getattr(obj, attr_name)
            if not isinstance(submodule, torch.nn.Module): continue
            named_parameters.update({attr_name+"."+k: v for k, v in dict(submodule.named_parameters()).items()})
        return named_parameters


def get_kwargs_from_gm(gm, **kwargs):
    target_kwargs = list(filter(lambda x: x.op == "placeholder", gm.graph.nodes))
    target_kwargs_names = [n.name for n in target_kwargs]
    missing_kwargs = list(filter(lambda x: x.name not in kwargs.keys() and len(x.args) == 0, target_kwargs))
    if len(missing_kwargs) > 0: raise RuntimeError('missing kwargs : %s'%missing_kwargs)
    out_kwargs = {}
    for k in target_kwargs:
        if k.name in kwargs:
            out_kwargs[k.name] = kwargs[k.name]
    return out_kwargs


# Create a sample DataFrame with more rows for better scrolling demonstration


# Function to display a scrollable DataFrame
def display_table_for_jupyter(table, columns=None, max_height=300, display=False):
    df = pd.DataFrame(table, columns=columns)
    # Convert DataFrame to HTML with styles
    html = df.to_html(classes='mystyle', index=False)
    # Define scrolling CSS
    scroll_css = f"""
    <style>
    .mystyle tbody {{
        display:block;
        overflow-y:scroll;
        max-height:{max_height}px;
    }}
    .mystyle thead, .mystyle tbody tr {{
        display:table;
        width:100%;
        table-layout:fixed;
    }}
    </style>
    """
    # Display with CSS
    obj = HTML(scroll_css + html)
    if display: 
        ipython_display(obj)
    return obj
