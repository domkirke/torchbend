
import torch.nn as nn
from typing import NoReturn
from . import BendedModule, BendingError

class BendedWrapperModule(nn.Module):
    def __init__(self, obj):
        nn.Module.__init__(self)
        
        self._import_attrs(obj)

    def _import_attrs(self, obj):
        for k, v in obj.__dict__.items():
            if not k in self.__dict__:
                self.__setattr__(k, v)

def make_wrapper_class_dict(obj):
    class_dict = dict(obj.__dict__)
    for k, v in BendedWrapperModule.__dict__.items():
        if k in class_dict:
            print('[Warning] method %s is being overrided'%k)
        class_dict[k] = v
    del class_dict['__dict__']
    return class_dict

def make_wrapper_class_for_obj(obj):
    name = obj.__qualname__ + "_wrapper"
    class_dict = make_wrapper_class_dict(obj)
    return type(name, (BendedWrapperModule,), class_dict)

def make_wrapper_for_obj(obj):
    return make_wrapper_class_for_obj(type(obj))(obj)


class BendedWrapper(BendedModule):
    def __init__(self, module):
        super().__init__(make_wrapper_for_obj(module))
        self.__original_obj__ = module
    
    def _getmodule_(self) -> object:
        return self.__original_obj__
    def _setmodule_(self, module) -> NoReturn:
        raise BendingError('Cannot set module of BendedModule after initaliazation.')
    def _delmodule_(self) -> NoReturn:
        raise BendingError('Cannot delete module of BendedModule')
    module = property(_getmodule_, _setmodule_, _delmodule_)

    @property
    def wrapped_module(self):
        return self._module

    def create_version(self, name, obj, strict=True):
        assert issubclass(type(obj), type(self.module)), "cannot create version for type %s from type %s"%(type(self.module), type(self.obj))
        state_dict = {}
        for submodule in self._module._modules:
            state_dict.update({submodule+"."+k: v for k, v in getattr(obj, submodule).state_dict().items()})
        super(BendedWrapper, self).create_version(name, state_dict, strict=strict)


__all__ = ['BendedWrapper']