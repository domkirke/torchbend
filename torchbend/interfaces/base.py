from inspect import ismethod
import abc
from ..tracing import BendingWrapper

def wrap_model_method(ext, func):
    def wrapped_function(*args, **kwargs):
        return getattr(ext, func)(*args, **kwargs)
    return wrapped_function


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
        self._bend_model(model)
    def _delmodel_(self):
        raise BendingInterfaceException('cannot delete model of interface')
    model = property(_getmodel_, _setmodel_, _delmodel_)

    def _import_callbacks_(self):
        for cb in self._imported_callbacks_:
            if not cb in dir(self._model):
                assert "method %s not present in base class %s"%(cb, type(self._model))
            setattr(self, cb, wrap_model_method(self._model, cb))

    def _import_wrapper_methods(self, model):
        assert isinstance(model, BendingWrapper)
        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if ismethod(attr) and (hasattr(attr,"__import_to_interface")):
                if hasattr(self, attr_name):
                    print('[Warning]: method %s seems in conflict with original module method. May cause discrepencies')
                if getattr(attr, "__import_to_interface"):
                    setattr(self, attr_name, wrap_model_method(self._model, attr_name))
                        
    @abc.abstractmethod
    def _bend_model(self, model):
        pass