import nn_tilde
from .script import ScriptedBendedModule, ScriptedBendedException

class NNBendedModule(ScriptedBendedModule, nn_tilde.Module):
    def __init__(self, model):
        self._methods = []
        self._attributes = ["none"]

        ScriptedBendedModule.__init__(self, model)
        if getattr(getattr(self, "register_methods"), "__isabstractmethod__", False):
            raise ScriptedBendedException('register_methods is not defined for class %s'%type(self))
        self.register_methods(model)

    def _register_controllable(self, controllable, controllables_hash):
        super()._register_controllable(controllable, controllables_hash)
        self.register_attribute(controllable.name, controllable.get_python_value())
