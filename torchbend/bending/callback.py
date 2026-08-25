
import torch, re
import copy

import logging
import abc
from collections import OrderedDict
import inspect
from functools import reduce
import copy
import torch.nn as nn
from types import MethodType, UnionType, NoneType
from collections import OrderedDict
from typing import Union, List, Optional, Any, Iterable
from .parameter import BendingParameter, _VALID_PARAM_TYPES, BendingParamType, BendingParameterException, get_param_type
from ..utils import _import_defs_from_tmpfile, _replace_placeholders, checktuple

# _param_ui  —  per-parameter UI metadata
# ─────────────────────────────────────────────────────────────────────────────
# Define a class-level `_param_ui` dict on any BendingCallback subclass to
# control how each controllable parameter is presented in the graph viewer.
# Only the keys you need are required; omit the rest for defaults.
#
# Supported keys per parameter entry:
#
#   range      : [min, max]  — slider / int-spinner bounds.
#                              Use None for one-sided bounds, e.g. [0., None].
#
#   step       : float       — slider step increment (default: auto).
#
#   label      : str         — display label (default: param name).
#
#   widget      : str        — force a specific widget type:
#                  "slider"  — continuous float slider (default for floats)
#                  "int"     — integer spinner
#                  "toggle"  — on/off checkbox (for bool params)
#                  "select"  — dropdown; requires `choices`
#                  "field"   — free-text numeric input
#
#   placeholder : str        — hint text shown inside a "field" widget when
#                              the input is empty (default: none).
#
#   choices    : list        — restrict value to this list.
#                              With widget="select" → dropdown.
#                              Without → nearest-value snap (client + server).
#
#   visible    : bool        — set False to hide the param from the UI entirely
#                              (default: True).
#
# Class-level flag (not per-parameter):
#
#   ui_compatible : bool     — set False on the class to hide the entire callback
#                              from the graph viewer bend dialog (default: True).
#                              Useful for callbacks whose interface is purely
#                              programmatic (e.g. tensor-input-only).
#
#   factory    : callable    — server-side transform applied to the incoming
#                              value after type coercion, before guard.
#                              Signature: factory(value) -> value.
#                              Serialised to the frontend as has_factory=True.
#
#   guard      : callable    — server-side validator.
#                              Signature: guard(value) -> True | Exception
#                              or guard(value, cb) -> True | Exception.
#                              If the guard declares 2+ required positional params,
#                              the callback instance is passed as the second arg.
#                              Return True to accept; return (or raise) an
#                              Exception whose str() is shown as a toast error.
#                              Serialised to the frontend as has_guard=True.
#
# Example:
#
#   _param_ui = {
#       'prob': {
#           'range': [0., 1.],
#           'step':  0.01,
#           'guard': lambda v: True if 0. <= v <= 1.
#                              else ValueError(f"prob must be in [0,1], got {v}"),
#       },
#       'mode': {
#           'widget':  'select',
#           'choices': ['add', 'mul'],
#       },
#       'internal_state': {'visible': False},
#   }
# ─────────────────────────────────────────────────────────────────────────────


attribute_setter_pattern = """
\t{{ATTRIBUTE[]}} = BendingParamType._to_tensor(self.{{ATTRIBUTE[]}}, {{ATTRIBUTE_TYPE[]}})"""

controllable_setter_pattern = """
\t{{CONTROLLABLE_NAME[]}} = self.get("{{ARG_NAME[]}}")
\tif {{CONTROLLABLE_NAME[]}} is not None: 
\t\t{{CONTROLLABLE_NAME[]}}={{CONTROLLABLE_NAME[]}}.to(x.device)
"""

input_controllable_setter_pattern = """
\tif {{INPUT_CONTROLLABLE_NAME[]}} is None:
\t\t{{INPUT_CONTROLLABLE_NAME[]}} = self.get("{{INPUT_ARG_NAME[]}}")
\t\tif {{INPUT_CONTROLLABLE_NAME[]}} is not None: 
\t\t\t{{INPUT_CONTROLLABLE_NAME[]}} = {{INPUT_CONTROLLABLE_NAME[]}}.to(x.device)
\telse:
\t\t{{INPUT_CONTROLLABLE_NAME[]}} = self.parse_controllable({{INPUT_CONTROLLABLE_NAME[]}}, {{INPUT_CONTROLLABLE_TYPE[]}})
"""

comment_pattern = """
\t#{{COMMENT[]}}
"""

forward_pattern = """
import torch
import typing
from typing import Optional
from torchbend import BendingParamType
from torchbend.bending.parameter import _VALID_PARAM_TYPES

@torch.jit.export
def dynamic_forward(self, x, name: Optional[str] = None, {{CALLBACK_ARGS_SIG}}):
{{COMMENT_PATTERN:LOOP}}
\t{{ATTRIBUTE_SETTER:LOOP}}
\t{{CONTROLLABLE_SETTER:LOOP}}
\t{{CONTROLLABLE_INPUT_SETTER:LOOP}}
\tif not self.is_ready: raise BendingCallbackException(self._not_ready_str)
\tif self.applied_to_node:
\t\treturn x
\telse:
\t\tout = self.{{BEND_INPUT_NAME}}(x=x, name=name, {{CALLBACK_ARGS}})
\t\treturn out
"""


def get_param_type_from_callback(module, name):
    if getattr(module, name) is not None:
        param_type = BendingParamType.param_type_from_type(type(getattr(module, name)))
    else:
        raw_param_type = _extract_type_if_optional(module.controllable_params[name][0])
        param_type = BendingParamType.param_type_from_type(raw_param_type)
    return param_type


#TODO: so far is_input=True is enforced; remove opposite case when confirmed
def create_callback_forward_function(module):
    controllables = module.controllables
    attributes = []
    attributes_types = []
    noinput_controllables = []
    noinput_controllable_names = []
    noinput_arg_names = []
    input_controllable_names = []
    input_args_names = []
    callback_args_sig = []
    callback_args = []
    callback_args_type = []
    input_controllables = []
    for name, _ in module.controllable_params.items():
        if name in controllables:
            c = controllables[name]
            # if not c.as_input: 
            #     noinput_controllables.append(c)
            #     noinput_arg_names.append(name)
            #     noinput_controllable_names.append(c.name)
            # else:
            #     input_controllables.append(c)
            #     input_controllable_names.append(c.name)
            #     input_args_names.append(name)
            #     callback_args_sig.append(
            #         f"{c.name}: Optional[torch.Tensor] = None"
            #     )
            #     callback_args_type.append(f"{c.param_type}")
            # callback_args.append(f'{name}={c.name}')
            input_controllables.append(c)
            input_controllable_names.append(c.name)
            input_args_names.append(name)
            callback_args_sig.append(
                f"{c.name}: Optional[torch.Tensor] = None"
            )
            callback_args_type.append(f"{c.param_type}")
            callback_args.append(f'{name}={c.name}')
        else:
            attributes.append(name)
            attributes_types.append(get_param_type_from_callback(module, name))
            callback_args.append(f'{name}={name}')

    callback_args_sig = ", ".join(callback_args_sig)
    callback_args = ", ".join(callback_args)
    comments = [f"generated from {module.__class__.__name__}"]
    codes = _replace_placeholders(forward_pattern,
                                  callback_args=callback_args, callback_args_sig = callback_args_sig, input_controllable_type=callback_args_type,
                                  attribute_setter = attribute_setter_pattern, attribute = attributes, attribute_type = attributes_types,
                                  _attribute_setter_loop = len(attributes), 
                                  controllable_setter = controllable_setter_pattern, controllable_name = noinput_controllable_names, arg_name = noinput_arg_names,
                                  _controllable_setter_loop = len(noinput_controllables), 
                                  controllable_input_setter = input_controllable_setter_pattern, input_controllable_name = input_controllable_names, input_arg_name = input_args_names,
                                  _controllable_input_setter_loop = len(input_controllables),
                                  comment_pattern = comment_pattern, comment = comments, _comment_pattern_loop = len(comments), 
                                  bend_input_name = module.bend_input_callback)
    funcs = _import_defs_from_tmpfile(codes, gl=globals(), lo=locals())
    return funcs['dynamic_forward']

static_controllable_pattern = """
import torch
from torchbend import BendingParamType

@torch.jit.export
def static_getter(self, name: str) -> torch.Tensor | None:
{{RETURN_SETTER:LOOP}}
\traise ValueError("controllable %s not present in callback")
"""

static_controllable_return_pattern = """
\tif name == "{{C_NAME[]}}": return BendingParamType._to_tensor(self.{{C_NAME[]}}, {{C_TYPE[]}})
"""

def _extract_type_if_optional(type_obj):
    if type(type_obj) == UnionType:
        no_none_types = list(filter(lambda x: x != NoneType, type_obj.__args__))
        if len(no_none_types) == 1: return no_none_types[0]
        else: raise TypeError('Could not infer type from Union : %s'%type_obj)
    else:
        return type_obj

def create_static_controllable_callback(module, controllable_dict):
    controllable_names = []
    controllable_types = []
    for name, c in controllable_dict.items():
        if isinstance(getattr(module, name), BendingParameter): continue
        controllable_names.append(name)
        
        controllable_types.append(get_param_type_from_callback(module, name))

    codes = _replace_placeholders(static_controllable_pattern, 
                                  return_setter = static_controllable_return_pattern, c_name=controllable_names, c_type=controllable_types, 
                                  _return_setter_loop = len(controllable_names))
    funcs = _import_defs_from_tmpfile(codes, gl=globals(), lo=locals())
    return funcs['static_getter']
                                


class BendingCallbackException(Exception):
    pass

class BendingCallbackAttributeException(Exception):
    pass

class BendingCallback(nn.Module):
    """Base class of every bending operation.

    A callback is an ``nn.Module`` that transforms weights and/or activations
    of a ``BendedModule``. Subclasses declare their capabilities through class
    attributes and implement the transformation methods:

    Class attributes:
        weight_compatible / activation_compatible: may bend weights / activations.
        jit_compatible / nntilde_compatible: survives ``script()`` / ``nntilde()``.
        controllable_params: dict ``{name: (allowed_types, default)}`` of init
            args that accept a plain value or a :class:`BendingParameter`.
        applied_to_node: True for graph-surgery callbacks (``apply_to_node``
            rewrites the fx node instead of transforming its value).
        ui_compatible / _param_ui / _extra_init_params: UI integration hints.

    Methods to override:
        bend_input(x, name=None): transform an activation value (also used for
            weights outside jit); ``name`` retrieves per-target state.
        apply_to_param(idx, param, cache): weight bending under jit — write the
            transformed ``cache`` (original value) into ``param`` in place.
        register_weight(parameter, name=None, cache=True): called once per
            bended weight; allocate per-target buffers here.
        register_activation(name, shape): called once per bended activation
            with the shape recorded at trace time.
        update(): recompute internal state after a BendingParameter change.

    Runtime API: ``get(name)`` (current value of a controllable, jit-safe),
    ``get_cache(i)`` (cached original weights), ``capture()`` / ``stop()``
    (record mode), ``apply()`` (in-place application to registered weights),
    ``param in callback`` (BendingParameter membership).
    """
    ui_compatible = True
    applied_to_node = False
    jit_compatible = False
    nntilde_compatible = False
    compatibility_attributes_keys = ['weight', 'activation', 'jit', 'nntilde']
    controllable_params = {}

    # some tags to indicate if the domain of the callback for ins / outs are the same.
    different_input = False
    different_output = False

    bend_input_callback = "bend_input"
    apply_to_param_callback = "apply_to_param"

    def __init__(self, **controllables):
        super().__init__()
        self._init_compatibility_attributes()
        # controllables points to the dynamic controls used. 
        self._controllables = nn.ModuleDict()
        self._init_controllables(**controllables)
        # targets and shapes are used for weight activation ; copies internally 
        # parameter to cache for dynamic bending
        self._bending_targets = nn.ParameterList()
        self._cache = []
        # bending shapes are used for activation bending, where just shapes are needed. 
        self._bending_shapes = OrderedDict()
        self._parameter_idx = 0
        self._is_capturing = False
        self._not_ready_str = "BendingCallback is not ready"
        self._init_forward_callback()
        self._for_nntilde = False
        # Once constructed, later controllable links (register_controllable) must
        # regenerate the forward; during __init__ the call above already covers it.
        self._initialized = True

    @torch.jit.ignore
    def bended_activations(self, fn = None):
        activations = list(self._bending_shapes.keys())
        if fn is not None: 
            activations = list(filter(lambda x: x.split(':')[0] == fn, activations))
        return activations

    def bended_params(self):
        return list(self._bending_shapes.keys())

    def _init_controllables(self,  **controllables):
        for k, v in self.controllable_params.items(): 
            setattr(self, f"_{k}_as_input", True)
            if k in controllables:
                if controllables.get(k) is None:
                    logging.info('parameter %s seems uninitialised. Taking default value : %s'%(k, v[1]))
                    self.__setattr__(k, v[1])
                else:
                    self.__setattr__(k, controllables[k])
                    if isinstance(controllables[k], BendingParameter):
                        if controllables[k].as_input: 
                            setattr(self, f"_{k}_as_input", True)
            else:
                self.__setattr__(k, v[1])

        self._init_static_controllable_callback(self.controllable_params)
        self._init_forward_callback()

    

    def _init_forward_callback(self):
        self.forward = MethodType(create_callback_forward_function(self), self)

    def _init_compatibility_attributes(self):
        if not hasattr(type(self), "weight_compatible"):
            self.weight_compatible = not hasattr(self.apply_to_param, "abs")
        if not hasattr(type(self), "activation_compatible"):
            self.activation_compatible = not hasattr(self.bend_input, "abs")
        for attr in self.compatibility_attributes_keys:
            attr_name = f"{attr}_compatible"
            setattr(self, attr_name, getattr(type(self), attr_name, False))
        setattr(self, "applied_to_node", getattr(type(self), "applied_to_node", False))

    def _init_static_controllable_callback(self, controllable_dict):
        setattr(self, "_get_static_controllable", MethodType(create_static_controllable_callback(self, controllable_dict), self))

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

    @property
    def needs_insertion(self) -> bool:
        """True when the callback is inserted as a graph node (vs. node rewriting)."""
        return (self.activation_compatible) and (not self.applied_to_node)

    # capture and stop
    def capture(self) -> None:
        """Switch to capture (record) mode; see the Capture callback."""
        self._is_capturing = True

    def stop(self) -> None:
        """Leave capture mode."""
        self._is_capturing = False

    def _get_operative_dims(self, dims: List[int], x: torch.Tensor) -> List[int]:
        """resolves negative dim indexes with a concrete tensor dimension."""
        operative_dims: List[int] = []
        for i, d in enumerate(dims):
            if d < 0:
                d =  x.ndim + d 
            operative_dims.append(d)
        return operative_dims

    # controllables
    @property
    def controllables(self):
        return self._controllables

    def _check_controllable_type(self, name, value):
        target_type = self.controllable_params[name][0]
        if target_type is None:
            return 
        target_type = checktuple(target_type)
        target_type_ids = [BendingParamType.param_type_from_type(t) for t in target_type]
        if isinstance(value, BendingParameter):
            assert value.param_type in target_type_ids, f"got value {value}, but param of types {target_type}"
        else:
            assert type(value) == BendingParamType.param_hash()[target_type]

    def register_controllable(self, name, value, valid_types=None):
        """Attach a value or BendingParameter to a declared controllable param.

        Called automatically by ``__setattr__`` when a BendingParameter is
        assigned; plain values are registered as buffers.
        """
        assert name in self.controllable_params, "tried to register controllable value %s, but not compatible with %s"%(name, type(self))
        self._check_controllable_type(name, value)
        if isinstance(value, BendingParameter):
            setattr(super().__getattr__('_controllables'), name, value)
            value._register_callback(self, name)
            # Regenerate dynamic_forward so it routes through self.get() for this
            # param instead of reading the stale instance-dict float value.
            # Required when a BendingParameter is linked to an already-built
            # callback (e.g. graph-viewer link_param); without it the forward
            # keeps the static _to_tensor(self.<param>) path and crashes once
            # self.<param> becomes a BendingParameter.
            super().__setattr__(name, value)
            # Only regenerate when linking to an already-built callback (e.g. the
            # graph-viewer link_param flow). During __init__ the constructor's own
            # _init_forward_callback handles it, and regenerating mid-construction
            # produces a broken forward that breaks scripting.
            if getattr(self, "_initialized", False):
                self._init_forward_callback()
                self._init_static_controllable_callback(self.controllable_params)
        else:
            value = torch.tensor(value)
            self.register_buffer(name, value)
        super().__setattr__(name, value)

    def parse_controllable(self, value: _VALID_PARAM_TYPES, param_type: int) -> torch.Tensor | None:
        return BendingParamType._to_tensor(value, param_type)
    
    def get(self, name: str) -> torch.Tensor | None:
        """Return the current value of a controllable parameter (jit-safe).

        Resolves, in order: attached BendingParameters (through their
        weight/bias/clamp transform), registered buffers, then static values.
        """
        if torch.jit.is_scripting():
            for i, v in self._controllables.items():
                if i==name:
                    return v.get_value()
            for i, b in dict(self.named_buffers()).items():
                if i == name:
                    return b
            return self._get_static_controllable(name)
        else:
            if name in self._controllables: 
                return self._controllables[name].get_value()
            elif name in dict(self.named_buffers()).keys():
                return dict(self.named_buffers())[name]
            else:
                return getattr(self, name)

    def input_controllables(self): 
        controllables = {}
        for k, v in self.controllables.items():
            # if v.as_input:
            controllables[k] = v
        return controllables
    
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
        """Register a parameter as a bending target (called by ``BendedModule.bend``).

        Caches the original value (when ``cache=True``) for revertible and
        dynamic bending, and returns the normalized target name. Subclasses
        allocating per-target state must call ``super().register_weight(...)``.
        """
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
    def apply_to_node(self, node):
        raise BendingCallbackException('apply_to_node called with callback of type %s, but not available'%type(self))

    def copy_activation(self, origin, target):
        self._bending_shapes[target] = self._bending_shapes[origin]

    def register_activation(self, name, shape):
        """Register an activation (with its traced shape) as a bending target.

        Returns ``(normalized_name, shape)``. Subclasses allocating shape-
        dependent state (masks, noise buffers...) must call the super method.
        """
        name = name.replace('.', '_')
        shape = list(shape)
        for i, a in enumerate(shape): 
            if isinstance(a, torch.SymInt):
                shape[i] = int(copy.deepcopy(a))
        self._bending_shapes[name] = shape
        return name, shape

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

    @abc.abstractmethod
    def apply_to_param(self, idx: int, param: nn.Parameter, cache: Optional[torch.Tensor] = None):
        """callback-specific method to apply the bending on a weight"""
        if not self.weight_compatible:
            pass
        else:
            raise NotImplementedError()

    def apply(self, update: bool = True, _reset_seed: bool = True):
        """applies in place a transformation to cached parameters."""
        if update:
            self.update()
        with torch.no_grad():
            for i, v in enumerate(self._bending_targets):
                v_cached = self.cache_from_id(i).data
                self.apply_to_param(i, v, v_cached)
    
    @abc.abstractmethod
    def bend_input(self,  x: torch.Tensor, name: Optional[str] = None):
        raise NotImplementedError()

    # def forward(self, x: torch.Tensor, name: Optional[str] = None):
    #     """applies transformation to an input (typically activations)"""
    #     if not self.is_ready: raise BendingCallbackException(self._not_ready_str)
    #     if self.applied_to_node:
    #         return x
    #     else:
    #         out = self._forward_callback(x, name=name)
    #         return out 
    #         # return self.bend_input(x, name=name)
            
    # script
    def script(self):
        """script callback is called when scripting a BendedModule to have a scriptable version of the callback.
        By default, it returns itself"""
        return self

    def nntilde(self):
        """script callback is called when scripting a BendedModule to have a nntilde version of the callback.
        By default, it returns itself"""
        self._for_nntilde = True

    @classmethod
    @torch.jit.ignore
    def ui_descriptor(cls) -> dict:
        """Return a UI-facing description of this callback's parameters.

        Ignored by TorchScript. Subclasses may override to provide richer hints
        (custom ranges, widget types, display labels).
        """
        from types import UnionType, NoneType as _NoneType
        params = {}
        for name, (type_hint, default) in cls.controllable_params.items():
            if type_hint is None:
                if isinstance(default, bool):
                    type_str = "bool"
                elif isinstance(default, int):
                    type_str = "int"
                else:
                    type_str = "float"
            else:
                try:
                    raw = type_hint
                    if type(raw) is UnionType:
                        real = [t for t in raw.__args__ if t is not _NoneType]
                        raw = real[0] if real else float
                    if isinstance(raw, tuple):
                        raw = raw[0]
                    type_str = BendingParamType._str_from_type(raw)
                except Exception:
                    type_str = "float"

            if type_str == "bool":
                widget, def_val = "toggle", bool(default) if default is not None else False
            elif type_str == "int":
                widget, def_val = "number", int(default) if isinstance(default, (int, float)) else 0
            elif type_str == "tensor":
                widget, def_val = "tensor", None
            else:
                widget, def_val = "slider", float(default) if isinstance(default, (int, float)) else 0.0

            rng = getattr(cls, f"_{name}_range", None)
            if rng is not None:
                rng = [float(rng[0]) if rng[0] is not None else None,
                       float(rng[1]) if rng[1] is not None else None]
            else:
                rng = [None, None]

            entry = {
                "type":        type_str,
                "default":     def_val,
                "range":       rng,
                "widget":      widget,
                "label":       name,
                "description": "",
                "visible":     True,
                "choices":     None,
                "placeholder": None,
                "has_factory": False,
                "has_guard":   False,
            }

            # Merge _param_ui overrides (factory/guard are server-side only)
            ui = (getattr(cls, "_param_ui", None) or {}).get(name, {})
            for k, v in ui.items():
                if k == "factory":
                    entry["has_factory"] = callable(v)
                elif k == "guard":
                    entry["has_guard"] = callable(v)
                elif k == "range":
                    entry["range"] = [
                        float(v[0]) if v[0] is not None else None,
                        float(v[1]) if v[1] is not None else None,
                    ]
                else:
                    entry[k] = v

            # choices imply select widget if widget not explicitly set and no other override
            if entry["choices"] is not None and "widget" not in ui:
                entry["widget"] = "select"

            params[name] = entry

        extra_init = {}
        for name, spec in (getattr(cls, "_extra_init_params", None) or {}).items():
            entry = dict(spec)
            entry.setdefault("label", name)
            entry.setdefault("required", False)
            extra_init[name] = entry
        return {
            "name": cls.__name__,
            "description": (cls.__doc__ or "").strip(),
            "params": params,
            "extra_init_params": extra_init,
            "weight_compatible": bool(getattr(cls, "weight_compatible", False)),
            "activation_compatible": bool(getattr(cls, "activation_compatible", False)),
            "jit_compatible": bool(getattr(cls, "jit_compatible", False)),
            "ui_compatible": bool(getattr(cls, "ui_compatible", True)),
        }

