import torch
CONTROLLABLE_TYPES = int | bool | float | torch.Tensor

from .mark import *
from .input import *
from .proxy import *
from .graph import *
from .tracing import *
from .module import *
from .module import _import_to_interface
from .interp import * 
from .script import *
from .nntilde import *


def script_method(self, script=True, **kwargs):
    mod = ScriptedBendedModule(self, **kwargs)
    if script: mod = torch.jit.script(mod)
    return mod
BendedModule.script = _import_to_interface(script_method)

def script_method_for_nntilde(self, script=True, **kwargs):
    mod = NNBendedModule(self, **kwargs)
    if script: mod = torch.jit.script(mod)
    return mod
BendedModule.nntilde = _import_to_interface(script_method_for_nntilde)

from .wrapper import *

