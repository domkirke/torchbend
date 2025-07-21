import torch, re
from collections import OrderedDict
import inspect
from functools import reduce
import copy
import torch.nn as nn
from types import MethodType, UnionType
from collections import OrderedDict
from typing import Union, List, Optional, Callable, Dict, Tuple
from .callback import BendingCallback, BendingCallbackException, BendingParameterException
from .parameter import BendingParameter, _VALID_PARAM_TYPES, BendingParamType, BendingParameterException, get_param_type
from ..utils import _import_defs_from_tmpfile, _replace_placeholders, checktuple

_native_callback_arguments = ['x', 'name']

controllable_pattern = """
\t
"""

call_pattern = """
\tx = self.callbacks[{{ITERATION}}].forward(x, name=name, {{CALLBACK_SIGS[]}})
"""

comment_pattern = """
\t# {{COMMENT[]}}
"""

forward_pattern = """
@torch.jit.export
def forward(self, x: torch.Tensor, name: Optional[str] = None, {{ADDITIONAL_ARGS}}):
{{COMMENT_PATTERN:LOOP}}
\tassert x is not None, "at least a value must be given"
{{CALL_PATTERN:LOOP}}
\treturn x 
"""


def _make_default_arg(v):
    if isinstance(v, str):
        return f'\"{v}\"'
    else:
        return str(v)


def _annotation_signature_str(annotation):
    name = annotation.__name__
    if name == "Optional":
        # return re.match(r'.*\[(.*)\]', str(annotation)).groups()[0]
        return str(annotation).replace('typing.', '')
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

def _create_forward_function(additional_args, additional_controllables, comments=[]):
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
                    types[c[0]] = types.get(c[0], []) + [_annotation_signature_str(c[1].annotation)]
                else:
                    types[c[0]] = types.get(c[0], []) + ['torch.Tensor']

    parsed_add_args = []
    controllables_names = [v[0] for v in additional_controllables.values()]
    for k, v in is_list.items():

        if v == 1:
            if types[k][0] != "_empty":
                parsed_add_args.append(f"{k}: {types[k][0]}= {_make_default_arg(defaults[k][0])}")
            else:
                parsed_add_args.append(f"{k} = {_make_default_arg(defaults[k][0])}")
        else:
            if k not in controllables_names:
                types[k] = ",".join(types[k])
                defaults_tmp = list(map(_make_default_arg, defaults[k]))
                if None in defaults_tmp:
                    defaults[k] = "None"
                else:
                    defaults[k] = "[" + ", ".join(defaults_tmp) + "]"
                parsed_add_args.append(f"{k}: Tuple[{types[k]}] = {defaults[k]}")
            # annotation = c[1].annotation.__name__ if is_list.get(c[0]) == 0 else "List[%s]"%(c[1].annotation.__name__)
            # parsed_add_args.append(f"{c[1].name}: {annotation} = {c[1].}")
            else:
                for i in range(v):
                    # for controllables, a separate entry is made for every recorded bended activation
                    if types[k][i] != "_empty":
                        parsed_add_args.append(f"{k}_{i}: {types[k][i]} = {_make_default_arg(defaults[k][i])}")
                    else:
                        parsed_add_args.append(f"{k}_{i} = {_make_default_arg(defaults[k][0])}")
    
    for a in additional_args:
        add_kwargs = []
        for c in a:
            if c[0] is None: continue
            if (is_list[c[0]] == 1):
                add_kwargs.append(f"{c[1].name}={c[1].name}")
            else:
                if c[0] in controllables_names:
                    add_kwargs.append(f"{c[1].name}={c[1].name}_{c[2]}")
                else:
                    add_kwargs.append(f"{c[1].name}={c[1].name}[{c[2]}]")
        callback_sigs.append(", ".join(add_kwargs))
    parsed_add_args = ", ".join(parsed_add_args)

    codes = _replace_placeholders(forward_pattern, 
                                  additional_args=parsed_add_args,
                                  comment_pattern=comment_pattern, comment=comments, _comment_pattern_loop=len(comments),
                                  call_pattern=call_pattern, callback_sigs=callback_sigs, _call_pattern_loop=len(additional_args))
    funcs = _import_defs_from_tmpfile(codes, gl=globals(), lo=locals())
    return funcs['forward']


class CallbackChain(nn.Module):
    native_callback_arguments = _native_callback_arguments
    instance_idx = 0
    def __init__(self, *args):
        super().__init__()
        full_controllables = {}
        for i, c in enumerate(args):
            assert isinstance(c, (BendingCallback, CallbackChain)), "CallbackChain only takes BendingCallback or CallbackChain as arguments"
            controllables = c.controllables
            for k, c in controllables.items():
                c_name = c.name
                if (c_name in full_controllables) and id(c) != id(full_controllables[c_name]):
                    raise BendingCallbackException('BendingParameter with name %s is present multiplie times, but with different objects.')
                full_controllables[c_name] = c
        self.callbacks = nn.ModuleList(args)
        self._controllables = nn.ModuleDict(full_controllables)
        self._controllable_params: Dict[str, int] = torch.jit.Attribute(list(self._controllables.keys()), List[str])
        self._init_compatibility_attributes()
        self._additional_args = self._init_forward_function()

    @classmethod
    def create(cls, *args, _jit_compatible: bool = False, name: str = None, **kwargs):
        if _jit_compatible:
            if name is None: name = f"{cls.__name__}_{cls.instance_idx}"
            new_type = type(name, (CallbackChain,), {})
            args = list(args)
            for i, a in enumerate(args):
                new_callback_type = type(f"{name}_{i}", (a.__class__,), {'original_class': a.__class__})
                args[i].__class__ = new_callback_type
            cls.instance_idx += 1
            return new_type(*args, **kwargs)
        else:
            return cls(*args, **kwargs)
        
    def __getitem__(self, idx: int):
        return self.callbacks[idx]

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
        additional_controllables = {}
        for name, param in self.input_controllables().items():
            if name not in fn_counter: fn_counter[name] = 0
            additional_controllables[name] = (param.name, param, fn_counter[name])
        # create new forward function with additional arguments
        comments = [f"{i}: {type(self.callbacks[i]).__name__}: {inspect.signature(self.callbacks[i].forward)}" for i in range(len(self.callbacks))]
        # comments = []
        if len(fn_counter) != 0: 
            func = _create_forward_function(additional_args, additional_controllables, comments) 
            setattr(self, "forward", MethodType(func, self))
        return additional_args

    def _init_compatibility_attributes(self):
        for attr in BendingCallback.compatibility_attributes_keys:
            res = reduce(lambda x, y : x and y, [getattr(c, f"{attr}_compatible") for c in self.callbacks], True)
            setattr(self, f"{attr}_compatible", torch.jit.Attribute(res, bool))
        self.applied_to_node = True in [c.applied_to_node for c in self.callbacks]

    @property
    def needs_insertion(self) -> bool:
        needs_insertion = False
        for c in self.callbacks:
            needs_insertion = needs_insertion or c.needs_insertion
        return needs_insertion

    @property
    def controllables(self):
        return self._controllables

    @property
    def controllable_params(self) -> List[str]:
        return self._controllable_params

    def input_controllables(self): 
        inp_c = {}
        for i in self.callbacks:
            inp_c.update(i.input_controllables())
        return inp_c

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

    def apply_to_node(self, node):
        for i, m in enumerate(self.callbacks):
            if m.applied_to_node:
                m.apply_to_node(node)

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
