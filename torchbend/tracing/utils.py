import copy, re, os
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
            bended_node = new_graph.create_node("call_module", hack_obj_name, args=(new_node,), kwargs={'name': node.name}, name=bended_node_name)
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
       meanval = value.float().mean()
       stdval = value.float().std() 
    except ValueError:
       meanval = torch.nan
       stdval = torch.nan    
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

def _resolve_code(code, **kwargs):
    pattern = str(code)
    for k, v in kwargs.items():
        pattern = re.compile(r'\{\{%s\}\}'%(k.upper()))
        iterations = list(pattern.finditer(code))
        while len(iterations) > 0:
            start, end = iterations[0].start(), iterations[0].end()
            code = code[:start] + str(v) + code[end:]
            iterations = list(pattern.finditer(code))
    #TODO check if no {{}} left
    return code

def _defs_from_template(template,  **kwargs):
    code = template
    local_namespace = {}
    for k, v in kwargs.items():
        pattern = re.compile(r'\{\{%s\}\}'%(k.upper()))
        iterations = list(pattern.finditer(code))
        while len(iterations) > 0:
            start, end = iterations[0].start(), iterations[0].end()
            code = code[:start] + str(v) + code[end:]
            iterations = list(pattern.finditer(code))
    
    code_compiled = compile(code, __file__, 'exec')
    exec(code_compiled, {}, local_namespace)
    return local_namespace


def get_random_hash(n=8):
    return "".join([chr(random.randrange(97,122)) for i in range(n)])

def _import_defs_from_tmpfile(code, gl=None, lo=None, tmpdir="/tmp/torchbend/jit"):
    gl = gl or globals() 
    os.makedirs(tmpdir, exist_ok=True)
    file = os.path.join(tmpdir, get_random_hash()+".py")
    with open(file, 'w+') as f:
        f.write(code)
    lo = lo or {}
    code_compiled = compile(code, file, 'exec')
    exec(code_compiled, gl, lo)
    return lo 

def _template_from_param(template: str, param: BendingParameter, **kwargs):
    kwargs['name'] = kwargs.get('name', param.name)
    if param.param_type == get_param_type("float"):
        kwargs['dtype'] = kwargs.get('dtype', torch.float32)
        kwargs['type_expr'] = kwargs.get('type_expr', "float")
        return _defs_from_template(template, **kwargs)
    elif param.param_type == get_param_type("int"):
        kwargs['dtype'] = kwargs.get('dtype', torch.int64)
        kwargs['type_expr'] = kwargs.get('type_expr', "int")
        return _defs_from_template(template, **kwargs)


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