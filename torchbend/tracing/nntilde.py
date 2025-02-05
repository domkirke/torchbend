import nn_tilde
from types import MethodType
from .module import BendedModule
from .script import ScriptedBendedModule, ScriptedBendedException

class NNBendedModuleException(Exception):
    pass

class NNBendedModule(nn_tilde.Module, ScriptedBendedModule):
    def __init__(self, model):
        assert isinstance(model, BendedModule), "NNBendedModule must be initialized with a BendedModule"

        self._get_set_candidates = {}
        ScriptedBendedModule.__init__(self, model)
        self._search_for_getter_and_setters(model.module)

        if getattr(getattr(model, "register_nntilde_attributes"), "__isabstractmethod__", False):
            raise ScriptedBendedException('nntilde_register_attributes is not defined for class %s'%type(model))
        model.register_nntilde_attributes(self)

        if getattr(getattr(model, "register_nntilde_methods"), "__isabstractmethod__", False):
            raise ScriptedBendedException('register_nntilde_methods is not defined for class %s'%type(model))
        model.register_nntilde_methods(self)

        self._reset_get_set_candidates()

    def _search_for_getter_and_setters(self, module):
        _candidates = {}
        for attr_name in dir(module):
            if not (attr_name.startswith("set_") or attr_name.startswith("get_")): continue
            if not isinstance(getattr(module, attr_name), MethodType): continue
            _candidates[attr_name] = getattr(module, attr_name)
        self._get_set_candidates = _candidates

    def _reset_get_set_candidates(self):
        self._get_set_candidates = {}
        
    def _register_controllable(self, controllable, controllables_hash):
        super()._register_controllable(controllable, controllables_hash)
        self.register_attribute(controllable.name, controllable.get_python_value())

    def register_attribute(self, attribute_name: str, values: nn_tilde.Any | nn_tilde.Tuple[nn_tilde.Any]):
        getter_name = "get_"+attribute_name
        setter_name = "set_"+attribute_name
        if not hasattr(self, getter_name):
            if getter_name not in self._get_set_candidates:
                raise NNBendedModuleException(f"getter for attribute {attribute_name} not found.")
            setattr(self, getter_name, self._get_set_candidates[getter_name])
        if not hasattr(self, setter_name):
            if setter_name not in self._get_set_candidates:
                raise NNBendedModuleException(f"setter for attribute {attribute_name} not found.")
            setattr(self, setter_name, self._get_set_candidates[setter_name])
        nn_tilde.Module.register_attribute(self, attribute_name, values)