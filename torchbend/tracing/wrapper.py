import torch, inspect
from torch.fx.proxy import TraceError
from typing import Union, Callable, Type, LiteralString, Tuple
from .module import BendedModule
from ..utils import checklist

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

class BendingWrapper(object):
    _bendable_types = [torch.nn.Module]
    def __init__(self, obj, target_objs=None):
        self.__bending_submodules = self._fetch_submodules(obj, target_objs)
        self._import_attrs_and_methods(obj)

    def _import_attrs_and_methods(self, obj, verbose=False):
        current_attrs = dir(self)
        for name, obj in inspect.getmembers(obj):
            if class_import_filter_fn(name, obj):
                if name in current_attrs:
                    print('[Warning] tried to override %s in BendingWrapper. May cause conflict'%name)
                if verbose:
                    print('importing attribute %s...'%name)
                if name in self.__bending_submodules:
                    obj = BendedModule(obj)
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
    def bended_modules(self):
        return dict(self.__bending_submodules)

    # parameters
    def parameters(self):
        raise NotImplementedError
    def named_parameters(self):
        raise NotImplementedError

    # @property 
    # def weights(self):
    #     raise NotImplementedError
    # @property
    # def activations(self):
    #     raise NotImplementedError

    def print_weights(self):
        raise NotImplementedError
    def print_activations(self):
        raise NotImplementedError

    def param_shape(self, param):
        raise NotImplementedError
    def activation_shape(self, param):
        raise NotImplementedError

    def state_dict(self, with_versions=False):
        raise NotImplementedError
        # if with_versions:
        #     return dict(self._param_dict)
        # else:
        #     return self._param_dict[self._version]
    def bended_state_dict(self):
        raise NotImplementedError

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