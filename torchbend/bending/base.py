import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Union, List, Optional
from .parameter import BendingParameter


class BendingCallbackException(Exception):
    pass


class BendingCallback(nn.Module):
    weight_compatible = False
    activation_compatible = False
    nntilde_compatible = False

    def __init__(self):
        super().__init__()
        # controllables points to the dynamic controls used. 
        self._controllables = nn.ModuleDict()
        # targets and shapes are used for weight activation ; copies internally 
        # parameter to cache for dynamic bending
        self._bending_targets = nn.ParameterList()
        self._cache = []
        # bending shapes are used for activation bending, where just shapes are needed. 
        self._bending_shapes = OrderedDict()
        self._parameter_idx = 0

    def __contains__(self, i: BendingParameter):
        """checks if a parameter is used by the callback instance"""
        return i in list(self._controllables.values())

    def _register_controllable_param(self, name, value):
        if isinstance(value, BendingParameter):
            self._controllables[name] = value
        setattr(self, name, value)

    def _register_parameter(self, parameter: List[nn.Parameter], name=None) :
        if not isinstance(parameter, nn.Parameter):
            raise BendingCallbackException("tried to register a parameter, but got type %s"%type(parameter))
        #TODO do not make this automatic? make "cache_parameter" function using weakrefs? 
        self._cache.append(parameter.data.clone())
        self._bending_targets.append(parameter)

    def update_parameter(self, parameter, new_parameter):
        """is used to replace reference from a parameter to another"""
        try:
            parameter_idx = list(map(id, self._bending_targets)).index(id(parameter))
        except IndexError:
            raise BendingCallbackException('parameter with id %s not found in callback %s'%(parameter, self))
        self._bending_targets[parameter_idx] = new_parameter

    def _generate_parameter_name(self):
        name = "parameter_%d"%self._parameter_idx 
        self._parameter_idx += 1
        return name

    def _register_shape(self, name, shape):
        self._bending_shapes[name] = shape

    def add_bending_target(self, name, parameter=None, shape=None):
        if (parameter is None) and (shape is None):
            raise BendingCallbackException("add_bending_target must be given a parameter or shape attribute")
        if shape is not None:
            self._register_shape(name, shape)
        if parameter is not None:
            self._register_parameter(parameter, name)

    def register_controllable(self, control):
        self._controllables[control.name] = control

    def parse_bending_parameter(self, param, name=None):
        if isinstance(param, (int, float)):
            assert name is not None, "BendingParameter must have a name"
            return BendingParameter(name="prob", value=torch.tensor(float(param)))
        elif isinstance(param, torch.Tensor):
            assert name is not None, "BendingParameter must have a name"
            return BendingParameter(name="prob", value=param)
        elif isinstance(param, BendingParameter):
            self.register_controllable(param)
            return param
        else:
            raise TypeError('received invalid prod argument of type %s'%type(param))

    # ---------------------------------
    # callback-specific methods

    def update(self):
        """updates internal state from controllables."""
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        """applies transformation to an input (typically activations)"""
        raise NotImplementedError()

    def apply(self, *args, **kwargs):
        """applies in place a transformation to a parameter."""
        raise NotImplementedError()



class CallbackChain(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.callbacks = nn.ModuleList(*args)

    def forward(self, x, name: Optional[str] = None):
        for i, m in enumerate(self.callbacks):
            x = m(x, name=name)
        return x 

