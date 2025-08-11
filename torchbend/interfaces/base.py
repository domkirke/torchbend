from inspect import ismethod
import functools
import torch
from typing import NoReturn
from collections import OrderedDict
import abc
from ..tracing import BendedWrapper, BendedModule

def wrap_model_method(ext, func, hook=None):
    @functools.wraps(func)
    def wrapped_function(*args, **kwargs):
        out = getattr(ext, func)(*args, **kwargs)
        if hook is not None:
            out = hook(out, (args, kwargs))
        return out
    return wrapped_function


def _export_to_module(fn):
    fn.__export_to_module = True
    return fn

def _overload_module(fn):
    fn.__overload_module = True
    return fn


class BendingInterfaceException(Exception):
    pass


class Interface(object):
    _imported_callbacks_ = []

    def __init__(self, model):
        self.model = model
        self._import_callbacks_()

    def _getmodel_(self):
        return self._model
    def _setmodel_(self, model):
        self._model = self._import_model(model)
        self._import_methods(self._model)
        self._bend_model(self._model)
        
    def _delmodel_(self):
        raise BendingInterfaceException('cannot delete model of interface')
    model = property(_getmodel_, _setmodel_, _delmodel_)

    def to(self, device):
        return self._model.to(device)

    def _import_model(self, model):
        if isinstance(model, torch.nn.Module):
            return BendedModule(model, _wrapped_methods = self._imported_callbacks_)
        else: 
            return BendedWrapper(model, _wrapped_methods = self._imported_callbacks_)

    def _import_callbacks_(self):
        for cb in self._imported_callbacks_:
            if cb not in dir(self._model):
                assert "method %s not present in base class %s"%(cb, type(self._model))
            if not hasattr(self, cb):
                setattr(self, cb, wrap_model_method(self._model, cb))

    def _retrieve_exported_methods(self):
        exported_methods = OrderedDict()
        # list methods to export
        for attr_name in dir(self):
            if attr_name in type(self).__dict__:
                # retrieving property callbacks instead of direct values.
                attr = type(self).__dict__[attr_name]
            else:
                attr = getattr(self, attr_name)
            if isinstance(attr, property):
                if hasattr(attr.fget, "__export_to_module"):
                    exported_methods[attr_name] = attr
            else:
                if hasattr(attr, "__export_to_module"):
                    exported_methods[attr_name] = attr
        return exported_methods

    def get_activations_hook(self, out, original_args):
        if original_args[1].get("_save_as_method"):
            method_name = original_args[1]["_save_as_method"]
            setattr(self, method_name, wrap_model_method(self._model, method_name))
        return out

    def from_activations_hook(self, out, original_args):
        if original_args[1].get("_save_as_method"):
            method_name = original_args[1]["_save_as_method"]
            setattr(self, method_name, wrap_model_method(self._model, method_name))
        return out

    def _import_methods(self, model):
        assert isinstance(model, (BendedWrapper, BendedModule))
        exported_methods = self._retrieve_exported_methods()        
        # import methods from module to interface
        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if ismethod(attr) and (hasattr(attr,"__import_to_interface")):
                if hasattr(self, attr_name):
                    if not getattr(getattr(self, attr_name), "__overload_module", False):
                        raise BendingInterfaceException("method %s seems in conflict with original module method. Add _overload_module decorator, or remove"%(attr_name,))
                hook = getattr(self, f"{attr_name}_hook", None)
                if getattr(attr, "__import_to_interface"):
                    setattr(self, attr_name, wrap_model_method(self._model, attr_name, hook=hook))
        # export methods to module
        for attr_name, attr in exported_methods.items():
            if hasattr(model, attr_name):
                # print('[Warning]: tried to export method %s to original module, but conflicting with existing attribute. Skipping'%attr_name)
                continue
            if attr_name in self.__dict__:
                setattr(model, attr_name, self.__dict__[attr_name])
            else:
                setattr(model, attr_name, getattr(self, attr_name))
                        
    @abc.abstractmethod
    def _bend_model(self, model):
        pass

    def _register_method_from_graph(self, graph, fn, method_name) -> NoReturn:
        self._model._register_method_from_graph(graph, fn, method_name)
        setattr(self, method_name, wrap_model_method(self._model, method_name))


    # nntilde-related callbacks
    @abc.abstractmethod
    def register_nntilde_methods(self, model):
        raise NotImplementedError

    @abc.abstractmethod
    def register_nntilde_attributes(self, model):
        raise NotImplementedError


    