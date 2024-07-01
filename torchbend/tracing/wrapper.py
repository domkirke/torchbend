import torch, inspect
from types import MethodType
import os, pathlib
from tabulate import tabulate
import re
from torch.fx.proxy import TraceError
from typing import Union, Callable, Type, Tuple
from .module import BendedModule
from .utils import _get_weight_properties
from .tracing import BendingTracer
from .input import Inputs
import copy
from ..utils import checklist


class BendingWrappingException(Exception):
    pass

def bendable_filter_fn(attr_name, obj):
    return attr_name.startswith('__') or \
           inspect.isfunction(getattr(obj, attr_name)) or \
           inspect.ismethod(getattr(obj, attr_name)) or \
           inspect.isclass(getattr(obj, attr_name)) or \
           inspect.isgenerator(getattr(obj, attr_name)) or \
           inspect.ismodule(getattr(obj, attr_name))

def class_import_filter_fn(attr_name, attr):
    return not attr_name.startswith('__')
    
def get_valid_attributes(obj):
    # remove attrs
    obj_attrs = dir(obj)
    obj_attrs = filter(lambda x: not bendable_filter_fn(x, obj), obj_attrs)
    obj_attrs = list(obj_attrs)
    return [(a, getattr(obj, a)) for a in obj_attrs]

def extract_objs_from_type(obj, obj_type):
    target_objs = {}
    for attr, attr_obj in get_valid_attributes(obj):
        if isinstance(attr_obj, obj_type):
            target_objs[attr] = attr_obj
    return target_objs

def _import_to_interface(fn):
    fn.__import_to_interface = True
    return fn


class BendingWrapper(object):
    _bendable_types = [torch.nn.Module]
    def __init__(self, obj, target_objs=None):
        self.__bending_submodules = self._fetch_submodules(obj, target_objs)
        self._import_attrs_and_methods(obj)

    def _import_attrs_and_methods(self, obj, verbose=False):
        type_obj = type(obj)
        current_attrs = dir(self)
        for name, obj in inspect.getmembers(obj):
            if class_import_filter_fn(name, obj):
                if name in current_attrs:
                    print('[Warning] tried to override %s in BendingWrapper. May cause conflict'%name)
                if verbose:
                    print('importing attribute %s...'%name)
                if name in self.__bending_submodules:
                    obj = BendedModule(obj)
                    self.__bending_submodules[name] = obj
                if inspect.ismethod(obj):
                    bounded_method = MethodType(_import_to_interface(getattr(type_obj, name)), self)
                    # bounded_method = _import_to_interface(bounded_method)
                    # bounded_method.__wrapped_from_bending__ = True
                    setattr(self, name, bounded_method)
                else:
                    setattr(self, name, obj)

    def _fetch_submodules(self, obj, target_objs=None):
        submodules = {}
        if target_objs is None:
            for t in self._bendable_types:
                submodules.update(extract_objs_from_type(obj, t))
        else:
            target_objs = checklist(target_objs)
            for target_name in target_objs:
                if hasattr(obj, target_name):
                    module = getattr(obj, target_name)
                    assert True in [isinstance(module, t) for t in self._bendable_types], "object %s not in bendable types : %s"%self._bendable_types
                    submodules[target_name] = module
                else:
                    raise TraceError("submodule %s is absent from %s"%(target_name, obj))
        return submodules

    @property
    @_import_to_interface
    def bended_modules(self):
        return dict(self.__bending_submodules)

    # parameters
    @_import_to_interface
    def parameters(self):
        parameters = []
        for k, v in self.bended_modules:
            parameters.extend(v.parameters())
        return parameters

    @_import_to_interface
    def named_parameters(self):
        named_parameters = dict()
        for k, v in self.bended_modules.items():
            named_parameters.update({f"{k}.{k_m}": v_m for k_m, v_m in v.named_parameters()})
        return named_parameters

    @_import_to_interface
    def reset(self):
        for v in self.bended_modules.values():
            v.reset()

    def _resolve_parameters(self, *weights):
        """get valid weight names from a regexp"""
        valid_weights = []
        for weight in self.weights:
            current_weight = []
            for w in weights:
                if re.match(w, weight) is not None:
                    current_weight.append(weight)
            if len(current_weight) > 0:
                valid_weights.extend(current_weight)
        return valid_weights

    def _resolve_activations(self, *activations):
        """get valid activation names from a regexp"""
        valid_acts = []
        for act in self.activation_names:
            current_act = []
            for a in activations:
                if re.match(a, act) is not None:
                    current_act.append(act)
            if len(current_act) > 0:
                valid_acts.extend(current_act)
        return valid_acts


    @property 
    @_import_to_interface
    def weights(self):
        weights = []
        for k, v in self.bended_modules.items():
            weights.extend([f"{k}.{param}" for param in v.weights])
        return weights

    @property
    @_import_to_interface
    def activations(self):
        activations = []
        for k, v in self.bended_modules.items():
            activations.extend([f"{k}.{param}" for param in v.activations])
        return activations

    @_import_to_interface
    def print_weights(self, flt=r".*", out=None):
        """print / export weights"""
        parameters = dict(filter(lambda v: re.match(flt, v[0]), dict(self.named_parameters()).items()))
        pretty_weights = tabulate(map(_get_weight_properties, parameters.items()))
        if out is None:
            print(pretty_weights)
        else:
            out = pathlib.Path(out)
            os.makedirs(out.parent, exist_ok=True)
            with open(out, 'w+') as f:
                f.write(pretty_weights)

    @_import_to_interface
    def print_activations(self):
        raise NotImplementedError

    @_import_to_interface
    def param_shape(self, param):
        raise NotImplementedError
    @_import_to_interface
    def activation_shape(self, param):
        raise NotImplementedError

    @_import_to_interface
    def state_dict(self, with_versions=False):
        raise NotImplementedError
        # if with_versions:
        #     return dict(self._param_dict)
        # else:
        #     return self._param_dict[self._version]

    @_import_to_interface
    def bended_state_dict(self):
        raise NotImplementedError

    @_import_to_interface
    def bend(self, callback, *params, **kwargs):
        params = self._resolve_parameters(*params)
        for p in params:
            target_module, param = p.split('.')[0], p.split('.')[:-1]
            if len(param) == 0: raise BendingWrappingException('%s does not seem to be a parameter.'%param)
            param = ".".join(param)
            if not target_module in self.bended_modules: raise BendingWrappingException('%s is not a bended submodule of wrapper.'%target_module)
            getattr(self, target_module).bend(callback, param, **kwargs)
            
    @_import_to_interface
    def bend_(self, callback, *params, **kwargs):
        params = self._resolve_parameters(*params)
        for p in params:
            target_module, param = p.split('.')[0], p.split('.')[1:]
            if len(param) == 0: raise BendingWrappingException('%s does not seem to be a parameter.'%param)
            param = ".".join(param)
            if not target_module in self.bended_modules: raise BendingWrappingException('%s is not a bended submodule of wrapper.'%target_module)
            getattr(self, target_module).bend_(callback, param, **kwargs)

    @_import_to_interface
    def trace(self, func, *args, **kwargs):
        """Updates inner graph with the target method and inputs"""
        self._inputs = Inputs(*args, **kwargs)
        tracer = BendingTracer(func=func)
        self.graph = tracer.trace(self.module, self._inputs)
        self._activations = tracer._activations


def is_within_class_definition(classname):
    current_frame = inspect.currentframe()
    for _ in range(2):
        f = current_frame.f_back
        assert f is not None
    code_obj = current_frame.f_code
    return code_obj.co_name != classname


#TODO only before instantiation ; how could we do after? 
def wrapmethod(classname, methodname):
    torch.fx._symbolic_trace._wrapped_methods_to_patch.append((classname, methodname))

def wrapmodule(obj):
    setattr(obj, "__torchbend_wrap__", True)
    # obj.forward = torch.fx.wrap(obj.forward)








__all__ = ['BendingWrapper', 'wrapmethod', 'wrapmodule']