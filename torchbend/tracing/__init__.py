import torch
CONTROLLABLE_TYPES = int | bool | float | torch.Tensor | None

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
    if bool(self.scriptable):
        mod = ScriptedBendedModule(self, **kwargs)
        if script: mod = torch.jit.script(mod)
        return mod
    else:
        raise ScriptedBendedException("%s cannot be scripted"%self)

def default_scriptable(self): 
    return ScriptableState.Unknown


BendedModule.script = _import_to_interface(script_method)
BendedModule.scriptable = property(_import_to_interface(default_scriptable))


def default_nntildable(self): 
    if self.scriptable: 
        return ScriptableState.Unknown
    else:
        return ScriptableState.NotScriptable

def script_method_for_nntilde(self, script=True, **kwargs):
    if bool(self.nntilde_compatible): 
        mod = NNBendedModule(self, **kwargs)
        if script: mod = torch.jit.script(mod)
        return mod
    else:
        raise ScriptedBendedException("%s cannot be scripted"%self)

BendedModule.nntilde = _import_to_interface(script_method_for_nntilde)
BendedModule.nntilde_compatible = property(_import_to_interface(default_scriptable))

from .wrapper import *

