import torch, re
from collections import OrderedDict
import inspect
from functools import reduce
import copy
import torch.nn as nn
from types import MethodType, UnionType
from collections import OrderedDict
from typing import Union, List, Optional, Callable, Dict, Tuple
from .parameter import BendingParameter, _VALID_PARAM_TYPES, BendingParamType, BendingParameterException
from ..utils import _import_defs_from_tmpfile, _replace_placeholders, checktuple

_native_callback_arguments = ['x', 'name']

call_pattern = """
\tx = self.callbacks[{{ITERATION}}].forward(x, name=name, {{CALLBACK_SIGS[]}})
"""

forward_pattern = """
@torch.jit.export
def forward(self, x: Optional[torch.Tensor] = None, name: Optional[str] = None, {{ADDITIONAL_ARGS}}):
\tassert x is not None, "at least a value must be given"
{{CALL_PATTERN:LOOP}}
\treturn x 
"""

class BendingCallbackException(Exception):
    pass

class BendingCallbackAttributeException(Exception):
    pass

def _make_default_arg(v):
    if isinstance(v, str):
        return f'\"{v}\"'
    else:
        return str(v)


def _annotation_siganture_str(annotation):
    name = annotation.__name__
    if name == "Optional":
        return re.match(r'.*\[(.*)\]', str(annotation)).groups()[0]
    else:
        return name

def _check_param_type(param, valid_types):
    if isinstance(valid_types, type):
        assert BendingParamType.param_hash()[param.param_type] == str(valid_types.__name__)
    elif isinstance(valid_types, UnionType):
        valid_types = list(map(lambda x: x.__name__, valid_types.__args__))
        assert BendingParamType.param_hash()[param.param_type] in list(valid_types)
    else:
        raise BendingParameterException('Could not verify parameter %s with type %s'%(param, valid_types))

def _create_forward_function(additional_args):
    is_list = OrderedDict()
    defaults = OrderedDict()
    types = OrderedDict()
    callback_sigs = []
    for a in additional_args:
        for c in a:
            if c[0] is not None:
                is_list[c[0]] = is_list.get(c[0], 0) + 1
                defaults[c[0]] = defaults.get(c[0], []) + [c[1]._default]
                if c[1].annotation is not None:
                    types[c[0]] = types.get(c[0], []) + [_annotation_siganture_str(c[1].annotation)]
                else:
                    types[c[0]] = types.get(c[0], []) + ['torch.Tensor']

    parsed_add_args = []
    for k, v in is_list.items():
        if v == 1:
            parsed_add_args.append(f"{k}: Optional[{types[k][0]}] = {_make_default_arg(defaults[k][0])}")
        else:
            types[k] = ",".join(types[k])
            defaults[k] = "[" + ", ".join(map(_make_default_arg, defaults[k])) + "]"
            parsed_add_args.append(f"{k}: Optional[Tuple[{types[k]}]] = {defaults[k]}")
        # annotation = c[1].annotation.__name__ if is_list.get(c[0]) == 0 else "List[%s]"%(c[1].annotation.__name__)
        # parsed_add_args.append(f"{c[1].name}: {annotation} = {c[1].}")
    
    for a in additional_args:
        add_kwargs = []
        for c in a:
            if c[0] is None: continue
            if is_list[c[0]] == 1:
                add_kwargs.append(f"{c[1].name}={c[1].name}")
            else:
                add_kwargs.append(f"{c[1].name}={c[1].name}[{c[2]}]")
        callback_sigs.append(", ".join(add_kwargs))
    parsed_add_args = ", ".join(parsed_add_args)
    codes = _replace_placeholders(forward_pattern, additional_args=parsed_add_args, call_pattern=call_pattern, callback_sigs=callback_sigs, _call_pattern_loop=len(additional_args))
    funcs = _import_defs_from_tmpfile(codes, gl=globals(), lo=locals())
    return funcs['forward']



class BendingCallback(nn.Module):
    weight_compatible = False
    activation_compatible = False
    jit_compatible = False
    nntilde_compatible = False
    controllable_params = {}

    def __init__(self, seed=None):
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
        self._is_capturing = False
        self._not_ready_str = "BendingCallback is not ready"

        # stochasticity management
        self._seed = seed

    def __contains__(self, i: BendingParameter):
        """checks if a parameter is used by the callback instance"""
        return i in list(self._controllables.values())

    def __setattr__(self, name, value):
        if isinstance(value, BendingParameter):
            self.register_controllable(name, value, valid_types=self.__class__.__annotations__.get(name))
        else:
            super().__setattr__(name, value)

    @property
    def is_ready(self) -> bool:
        return True

    # capture and stop
    def capture(self) -> None:
        self._is_capturing = True

    def stop(self) -> None:
        self._is_capturing = False

    # manage torch seeds
    def _reset_seed(self):
        if self._seed is not None: torch.manual_seed(self._seed)

    # controllables
    @property
    def controllables(self):
        return self._controllables

    def _check_controllable_type(self, name, value):
        if self.controllable_params[name] is None:
            return 
        target_type = checktuple(self.controllable_params[name])
        if isinstance(value, BendingParameter):
            assert value.param_type in target_type

    def register_controllable(self, name, value, valid_types=None):
        assert name in self.controllable_params, "tried to register controllable value %s, but not compatible with %s"%(name, type(self))
        self._check_controllable_type(name, value)
        if isinstance(value, BendingParameter):
            setattr(super().__getattr__('_controllables'), name, value)
            value._register_callback(self, name)
        else:
            value = torch.tensor(value)
            self.register_buffer(name, value)
        super().__setattr__(name, value)
    
    def get(self, name: str) -> torch.Tensor:
        if torch.jit.is_scripting():
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
    
    # weights
    def _generate_parameter_name(self):
        name = "parameter_%d"%self._parameter_idx 
        self._parameter_idx += 1
        return name

    def get_cache(self, i: int) -> torch.Tensor:
        """
        Get a given parameter cache from its index.
        Don't judge me, this is because torch.jit only allows literal indexing..."""
        assert i < len(self._cache)
        for j, c in enumerate(self._cache):
            if i == j: return c
        raise BendingCallbackException('cache %d does not exist'%i)

    def cache_from_id(self, idx: int) -> torch.nn.Parameter:
        #grrrr
        for i, v in enumerate(self._cache):
            if i == idx:
                if v is None:
                    raise BendingCallbackException('cache with idx %s has not been cached.'%idx)
                else:
                    return v
        raise BendingCallbackException('%s not present in masks'%idx)

    def register_weight(self, parameter: List[nn.Parameter], name=None, cache=True) -> str:
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

    def update_weight(self, parameter, new_parameter):
        """is used to replace reference from a parameter to another"""
        try:
            parameter_idx = list(map(id, self._bending_targets)).index(id(parameter))
        except IndexError:
            raise BendingCallbackException('parameter with id %s not found in callback %s'%(parameter, self))
        self._bending_targets[parameter_idx] = new_parameter

    
    # activations
    def register_activation(self, name, shape):
        name = name.replace('.', '_')
        self._bending_shapes[name] = shape
        return name

    # generic callback for bending targets
    def add_bending_target(self, name, parameter=None, shape=None, cache=True):
        if (parameter is None) and (shape is None):
            raise BendingCallbackException("add_bending_target must be given a parameter or shape attribute")
        if shape is not None:
            self.register_activation(name, shape)
        if parameter is not None:
            self.register_weight(parameter, name, cache=cache)


    # ---------------------------------
    # callback-specific methods

    def get_shape(self, shape):
        """returns the shape of the activation after processing. 
        Used in CallbackChain to register piped activation bending"""
        return shape

    def update(self):
        """updates internal state from controllables."""
        pass

    def apply_to_param(self, idx: int, param: nn.Parameter, cache: Optional[torch.Tensor] = None):
        raise NotImplementedError()

    def apply(self, update: bool = True, _reset_seed: bool = True):
        """applies in place a transformation to cached parameters."""
        if update:
            self.update()
        self._reset_seed()
        for i, v in enumerate(self._bending_targets):
            v_cached = self.cache_from_id(i).data
            self.apply_to_param(i, v, v_cached)
    
    def bend_input(self,  x: torch.Tensor, name: Optional[str] = None):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor, name: Optional[str] = None):
        """applies transformation to an input (typically activations)"""
        if not self.is_ready: raise BendingCallbackException(self._not_ready_str)
        return self.bend_input(x, name=name)

    # script
    def script(self):
        return self

    # ops
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
        self._controllable_params: Dict[str, int] = torch.jit.Attribute(list(self._controllables.keys()), List[str])
        self._init_compatibility_attributes()
        self._additional_args = self._init_forward_function()

    def _init_forward_function(self) -> List[Tuple[List[str], List[int]]]:
        additional_args = [list() for _ in self.callbacks]
        fn_counter = {}
        for i, callback in enumerate(self.callbacks):
            sig = inspect.signature(getattr(callback, "forward"))
            for name, param in dict(sig.parameters).items():
                if name in _native_callback_arguments: continue
                if name not in fn_counter: fn_counter[name] = 0
                additional_args[i].append((name, param, fn_counter[name]))
                fn_counter[name] += 1
        # create new forward function with additional arguments
        if len(fn_counter) != 0: 
            func = _create_forward_function(additional_args) 
            setattr(self, "forward", MethodType(func, self))
        return additional_args

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
        """applies in place a transformation to cached parameters during jit export."""
        for i, m in enumerate(self.callbacks):
            m.apply(update)

    @torch.jit.export
    def forward(self, x, name: Optional[str] = None):
        """applies the transformation to a given input"""
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

    def register_weight(self, parameter: List[nn.Parameter], name=None, cache=True) -> str:
        for c in self.callbacks:
            c.register_weight(parameter, name=name, cache=cache)

    def register_activation(self, name, shape):
        for c in self.callbacks:
            # TODO make a get_shape function for 
            c.register_activation(name, shape)
            shape = c.get_shape(shape)
    

def is_bending_callback(obj):
    return isinstance(obj, (BendingCallback, CallbackChain))

