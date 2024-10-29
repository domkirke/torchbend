import torch
from functools import reduce
import copy
import torch.nn as nn
from collections import OrderedDict
from typing import Union, List, Optional, Callable
from .parameter import BendingParameter, _VALID_PARAM_TYPES


class BendingCallbackException(Exception):
    pass

class BendingCallbackAttributeException(Exception):
    pass


class BendingCallback(nn.Module):
    weight_compatible = False
    activation_compatible = False
    jit_compatible = False
    nntilde_compatible = False
    controllable_params = []

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

    def __setattr__(self, name, value):
        if isinstance(value, BendingParameter):
            self.register_controllable(name, value)
        else:
            super().__setattr__(name, value)

    @property
    def controllables(self):
        return self._controllables

    def script(self):
        return self
    
    def get_cache(self, i: int) -> torch.Tensor:
        "Don't judge me, this is because torch.jit only allows literal indexing..."
        assert i < len(self._cache)
        for j, c in enumerate(self._cache):
            if i == j: return c
        raise BendingCallbackException('cache %d does not exist'%i)

    def register_controllable(self, name, value):
        assert name in self.controllable_params, "tried to register controllable value %s, but not compatible with %s"%(name, type(self))
        if isinstance(value, BendingParameter):
            setattr(super().__getattr__('_controllables'), name, value)
            value._register_callback(self, name)
        else:
            value = torch.tensor(value)
            self.register_buffer(name, value)
        super().__setattr__(name, value)

    def register_parameter(self, parameter: List[nn.Parameter], name=None, cache=True) -> str:
        if not isinstance(parameter, nn.Parameter):
            raise BendingCallbackException("tried to register a parameter, but got type %s"%type(parameter))
        #TODO do not make this automatic? make "cache_parameter" function using weakrefs? 
        if cache:
            self._cache.append(parameter.data.clone())
        else:
            self._cache.append(None)
        self._bending_targets.append(parameter)
        name = self._generate_parameter_name() if name is None else name.replace(".", "_")
        return name

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

    def register_activation(self, name, shape):
        name = name.replace('.', '_')
        self._bending_shapes[name] = shape
        return name

    def add_bending_target(self, name, parameter=None, shape=None, cache=True):
        if (parameter is None) and (shape is None):
            raise BendingCallbackException("add_bending_target must be given a parameter or shape attribute")
        if shape is not None:
            self.register_activation(name, shape)
        if parameter is not None:
            self.register_parameter(parameter, name, cache=cache)

    # def register_controllable(self, control):
    #     self._controllables[control.name] = control

    def get(self, name: str):
        if torch.jit.is_scripting():
            # assert name in self.controllable_params, "%s has no controllable value %s"%(type(self), name)
            for i, v in self._controllables.items():
                if i==name:
                    return v.get_value()
            for i, b in dict(self.named_buffers()).items():
                if i == name:
                    return b
            raise BendingCallbackAttributeException(name)
        else:
            if name in self._controllables: 
                return self._controllables[name].get_value()
            elif name in dict(self.named_buffers()).keys():
                return dict(self.named_buffers())[name]
            else:
                return getattr(self, name)

    def parse_bending_parameter(self, param, name=None):
        if isinstance(param, (int, float)):
            assert name is not None, "BendingParameter must have a name"
            return BendingParameter(name="prob", value=torch.tensor(float(param)))
        elif isinstance(param, torch.Tensor):
            assert name is not None, "BendingParameter must have a name"
            return BendingParameter(name="prob", value=param)
        elif isinstance(param, BendingParameter):
            self.register_controllable(param.name, param)
            return param
        else:
            raise TypeError('received invalid prod argument of type %s'%type(param))

    # ---------------------------------
    # callback-specific methods

    def update(self):
        """updates internal state from controllables."""
        pass

    def forward(self, *args, **kwargs):
        """applies transformation to an input (typically activations)"""
        raise NotImplementedError()
    
    def cache_from_id(self, idx: int) -> torch.nn.Parameter:
        #grrrr
        for i, v in enumerate(self._cache):
            if i == idx:
                if v is None:
                    raise BendingCallbackException('cache with idx %s has not been cached.'%idx)
                else:
                    return v
        raise BendingCallbackException('%s not present in masks'%idx)

    def apply_to_param(self, idx: int, param: nn.Parameter, cache: Optional[torch.Tensor] = None):
        pass

    def apply(self, update: bool = True):
        """applies in place a transformation to cached parameters."""
        if update:
            self.update()
        for i, v in enumerate(self._bending_targets):
            v_cached = self.cache_from_id(i).data
            self.apply_to_param(i, v, v_cached)

    def __rshift__(self, obj):
        if isinstance(obj, CallbackChain):
            return CallbackChain(self, *obj.callbacks)
        elif isinstance(obj, BendingCallback):
            return CallbackChain(self, obj)
        else:
            raise TypeError('%s can only be added to CallbackChain or BendingCallback objects'%(type(self).__name__))


class CallbackChain(nn.Module):
    def __init__(self, *args):
        super().__init__()
        full_controllables = {}
        for i, c in enumerate(args):
            assert isinstance(c, (BendingCallback, CallbackChain)), "CallbackChain only takes BendingCallback or CallbackChain as arguments"
            controllables = c.controllables
            for k, c in controllables.items():
                if (k in full_controllables) and id(c) != id(full_controllables[k]):
                    #TODO merge?
                    raise BendingCallbackException('BendingParameter with name %s is present multiplie times, but with different objects.')
                full_controllables[k] = c
        self.callbacks = nn.ModuleList(args)
        self._controllables = nn.ModuleDict(full_controllables)
        self._controllable_params: List[str] = torch.jit.Attribute(list(self._controllables.keys()), List[str])
        self._init_compatibility_attributes()

    def _init_compatibility_attributes(self):
        for attr in ['weight', 'activation', 'jit', 'nntilde']:
            res = reduce(lambda x, y : x and y, [getattr(c, f"{attr}_compatible") for c in self.callbacks], True)
            setattr(self, f"{attr}_compatible", torch.jit.Attribute(res, bool))

    @property
    def controllable_params(self) -> List[str]:
        return self._controllable_params

    def script(self):
        scripted = copy.copy(self)
        scripted.callbacks = nn.ModuleList([s.script() for s in scripted.callbacks])
        return scripted

    def add_bending_target(self, name, parameter=None, shape=None, cache=True):
        for i, m in enumerate(self.callbacks):
            m.add_bending_target(name, parameter=parameter, shape=shape, cache=cache)

    def apply_to_param(self, idx: int, param: nn.Parameter, cache: Optional[torch.Tensor] = None):
        for i, m in enumerate(self.callbacks):
            if i == 0:
                m.apply_to_param(idx, param, cache)
            else:
                m.apply_to_param(idx, param, param.data)

    def update(self):
        for i, m in enumerate(self.callbacks):
            m.update()

    @torch.jit.export
    def apply(self, update: bool = True):
        """applies in place a transformation to cached parameters."""
        for i, m in enumerate(self.callbacks):
            m.apply(update)

    @torch.jit.export
    def forward(self, x, name: Optional[str] = None):
        for i, m in enumerate(self.callbacks):
            x = m(x, name=name)
        return x 

    def __rshift__(self, obj):
        if isinstance(obj, CallbackChain):
            return CallbackChain(*self.callbacks, *obj.callbacks)
        elif isinstance(obj, BendingCallback):
            return CallbackChain(*self.callbacks, obj)
        else:
            raise TypeError('%s can only be added to CallbackChain or BendingCallback objects'%(type(self).__name__))

def is_bending_callback(obj):
    return isinstance(obj, (BendingCallback, CallbackChain))

