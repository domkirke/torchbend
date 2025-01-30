from .input import *
from .proxy import *
from .graph import *
from .tracing import *
from .module import *
from .interp import * 

from .nntilde import *
def script_method(self, script=True, export_for_nn: bool = False, **kwargs):
    # mod = self if not hasattr(self, "get_scriptable") else self.get_scriptable(**kwargs)
    if export_for_nn:
        mod = NNBendedModule(self)
    else:
        mod = ScriptedBendedModule(self)
    # for m in checklist(methods):
    #     setattr(mod, m, torch.jit.export(getattr(mod, m)))
    if script: 
        return torch.jit.script(mod)
    else: 
        return mod
BendedModule.script = script_method

from .wrapper import *

