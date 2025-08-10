
import torch.nn as nn
from typing import NoReturn
from . import BendedModule, BendingError
from .module import _copy_attrs

class BendedWrapperModule(nn.Module):
    def __init__(self, obj):
        nn.Module.__init__(self)
        self._import_attrs(obj)

    def _import_attrs(self, obj):
        for k, v in obj.__dict__.items():
            if not k in self.__dict__:
                self.__setattr__(k, v)

def make_wrapper_class_dict(obj):
    class_dict = {}
    imported_attrs = filter(lambda x: not x.startswith('__'), dir(obj))
    for k in imported_attrs:
        class_dict[k] = getattr(obj, k)
    for k, v in BendedWrapperModule.__dict__.items():
        if k in imported_attrs:
            print('[Warning] method %s is being overrided'%k)
        class_dict[k] = v
    if '__dict__' in class_dict:
        del class_dict['__dict__']
    return class_dict

def make_wrapper_class_for_obj(obj, wrapped_methods):
    name = obj.__qualname__ + "_wrapper"
    class_dict = make_wrapper_class_dict(obj)
    for w in wrapped_methods: 
        class_dict[w] = getattr(obj, w)
    return type(name, (BendedWrapperModule,), class_dict)

def make_wrapper_for_obj(obj, wrapped_methods):
    return make_wrapper_class_for_obj(type(obj), wrapped_methods)(obj)


class BendedWrapper(BendedModule):
    __copy_attrs__ = BendedModule.__copy_attrs__ + ['__original_obj__']
    def __init__(self, module, _wrapped_methods=[]):
        super().__init__(make_wrapper_for_obj(module, _wrapped_methods), _wrapped_methods=_wrapped_methods)
        self.__original_obj__ = module
        # no __call__ callback for BendingWrapper

    def __repr__(self):
        return "BendedWrapper(%s)"%(type(self.__original_obj__).__name__)
    
    def _getmodule_(self) -> object:
        return self.__original_obj__
    def _setmodule_(self, module) -> NoReturn:
        raise BendingError('Cannot set module of BendedModule after initialization.')
    def _delmodule_(self) -> NoReturn:
        raise BendingError('Cannot delete module of BendedModule')
    module = property(_getmodule_, _setmodule_, _delmodule_)

    @property
    def wrapped_module(self):
        return self._module

    @classmethod
    def copy(cls, module):
        module_copy = BendedWrapper(module.__original_obj__)
        _copy_attrs(module, module_copy, cls.__copy_attrs__)
        return module_copy

    def create_version(self, name, obj, strict=True):
        assert issubclass(type(obj), type(self.module)), "cannot create version for type %s from type %s"%(type(self.module), type(self.obj))
        state_dict = {}
        for submodule in self._module._modules:
            state_dict.update({submodule+"."+k: v for k, v in getattr(obj, submodule).state_dict().items()})
        super(BendedWrapper, self).create_version(name, state_dict, strict=strict)


__all__ = ['BendedWrapper']