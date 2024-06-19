import torch, inspect
from .module import BendedModule

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
    def __init__(self, obj):
        self.__bending_submodules = self._fetch_submodules(obj)
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

    def _fetch_submodules(self, obj):
        submodules = {}
        for t in self._bendable_types:
            submodules.update(extract_objs_from_type(obj, t))
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


__all__ = ['BendingWrapper']