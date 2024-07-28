from .input import *
from .proxy import *
from .tracing import *
from .module import *

from .script import *
def script_method(self, script=True, methods=[]):
    mod = ScriptedBendedModule(self)
    # for m in checklist(methods):
    #     setattr(mod, m, torch.jit.export(getattr(mod, m)))
    if script: 
        return torch.jit.script(mod)
    else: 
        return mod
BendedModule.script = script_method

from .wrapper import *
# from .nntilde import BendableNNTildeModule

