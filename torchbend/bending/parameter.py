from typing import Union, Optional, Any, Tuple, Callable, Dict, NoReturn
import numbers
import torch
import torch.nn as nn
import typing
from numbers import Real, Integral
from enum import Enum
from torchbend import log_error, log_warning
from torchbend.utils import checktensor
from types import UnionType, NoneType


class BendingParameterException(Exception):
    pass


_VALID_PARAM_TYPES = Union[float, int, bool, torch.Tensor, None]


def _extract_type_if_optional(type_obj):
    if type(type_obj) in (UnionType, typing._UnionGenericAlias):
        no_none_types = list(filter(lambda x: x != NoneType, type_obj.__args__))
        if len(no_none_types) == 1: return no_none_types[0]
        else: raise TypeError('Could not infer type from Union : %s'%type_obj)
    else:
        return type_obj



class BendingParamType():

    @staticmethod
    def get_param_type(param_type: str):
        #damn torchscipt, don't judge me
        if param_type not in BendingParamType._param_types:
            raise BendingParameterException('param_type %s not handled'%param_type)
        return BendingParamType.param_types()[param_type]

    @classmethod
    def __class_getitem__(cls, idx: str):
        return cls.param_types()[idx.lower()]

    @staticmethod
    def param_types():
        return {'float': 2, 'int': 1, 'bool': 0, 'complex': 7, 'tensor': 4, 'str': 3}
    @staticmethod
    def param_hash():
        return {v: k for k, v in BendingParamType.param_types().items()}

    @staticmethod
    def _get_default_tensor_type() -> type:
        dtype = torch.get_default_dtype()
        if dtype in [torch.bool]:
            return torch.ByteTensor
        if dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
            return torch.LongTensor
        elif dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
            return torch.FloatTensor
        elif dtype in [torch.complex, torch.complex32, torch.complex64, torch.complex128]: 
            raise NotImplementedError
        else:
            raise TypeError('cannot parse tensor %s as a native python value'%dtype)

    @staticmethod 
    def _str_from_type(obj: type) -> str:
        # if issubclass(obj, torch.Tensor):
        #     if obj == torch.Tensor: obj = BendingParamType._get_default_tensor_type()
        obj = _extract_type_if_optional(obj)
        if issubclass(obj, bool):
            return 'bool'
        elif issubclass(obj, (int, numbers.Integral)):
            return 'int'
        elif issubclass(obj, (float, numbers.Real)):
            return 'float'
        elif issubclass(obj, (complex, numbers.Complex)):
            raise NotImplementedError()
        elif getattr(torch, obj.__name__, None) == obj:
            return 'tensor'
            # if obj.numel() == 1:
            #     if issubclass(obj, torch.FloatTensor):
            #         return 'float'
            #     elif issubclass(obj, (torch.LongTensor, torch.IntTensor)):
            #         return 'int'
            #     elif issubclass(obj, (torch.ByteTensor)):
            #         return 'bool'
            # else:
            #     return 'tensor'
        raise TypeError('could not get type string for type : %s'%obj)
            

    @staticmethod
    def param_type_from_type(obj: type) -> int:
        return BendingParamType.param_types()[BendingParamType._str_from_type(obj)]
    
    @staticmethod
    def _param_type_from_obj(obj: _VALID_PARAM_TYPES) -> int:
        #damn torchscipt, don't judge me
        # if torch.is_tensor(obj):
        if torch.jit.isinstance(obj, torch.Tensor):
            # assert obj.numel() == 1, "Got non-scalar tensor for BendingParameter value"
            # if obj.numel() == 1:
            #     if obj.dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
            #         return BendingParamType.param_types()['float']
            #     elif obj.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
            #         return BendingParamType.param_types()['int']
            #     elif obj.dtype in [torch.complex, torch.complex32, torch.complex64, torch.complex128]:
            #         raise NotImplementedError
            #         return BendingParamType.param_types()['complex']
            #     elif obj.dtype in [torch.bool]:
            #         return BendingParamType.param_types()['bool']
            #     else:
            #         raise BendingParameterException("tensor dtype not handled : %s"%obj.dtype)
            # else: 
            return BendingParamType.param_types()['tensor'] 
        elif isinstance(obj, bool):
            return BendingParamType.param_types()['bool']
        elif isinstance(obj, numbers.Integral):
            return BendingParamType.param_types()['int']
        elif isinstance(obj, numbers.Real):
            return BendingParamType.param_types()['float']
        elif isinstance(obj, numbers.Complex):
            raise NotImplementedError
            return BendingParamType.param_types()['complex']
        elif isinstance(obj, str):
            return BendingParamType.param_types()['str']
        else:
            raise BendingParameterException('cannot retrieve param type from type : %s'%(type(obj)))

    @staticmethod
    def get_type(type_obj) -> int:
        return BendingParamType.param_types()[type_obj]

    @staticmethod
    def _parse_tensor_value(value: _VALID_PARAM_TYPES, param_type: int) -> torch.Tensor | None:
        if torch.jit.isinstance(value, torch.Tensor):
            if (param_type == BendingParamType.param_types()['bool']):
                return value.byte()
            elif (param_type == BendingParamType.param_types()['float']):
                return value.float()
            elif (param_type == BendingParamType.param_types()['int']):
                return value.int()
            elif (param_type == BendingParamType.param_types()['complex']):
                raise NotImplementedError
                return torch.view_as_complex(value)
            elif (param_type == BendingParamType.param_types()['tensor']):
                return value
            else:
                raise BendingParameterException('Wrong ParamType: %s'%param_type)
        else:
            raise BendingParameterException('value is not tensor')

    @staticmethod
    def _to_tensor(value: _VALID_PARAM_TYPES, param_type: int) -> torch.Tensor | None:
        #damn torchscipt, don't judge me
        #TODO make a generative code for param_types?
        # _param_types = {'float': 1, 'int': 2} 
        #TODO handle general float types
        if value is None: return None
        if torch.jit.isinstance(value, torch.Tensor):
            return BendingParamType._parse_tensor_value(value, param_type)            
        else:
            if (param_type == BendingParamType.param_types()['bool']):
                if isinstance(value, bool):
                    return torch.tensor(value)
                else:
                    raise BendingParameterException('Cannot make byte tensor out of %s'%value)
            if (param_type == BendingParamType.param_types()['float']):
                if isinstance(value, (int, float)):
                    return torch.tensor(float(value))
                else:
                    raise BendingParameterException('Cannot make float tensor out of %s'%value)
            elif (param_type == BendingParamType.param_types()['int']):
                if isinstance(value, (int, float)):
                    return torch.tensor(int(value))
                else:
                    raise BendingParameterException('Cannot make int tensor out of %s'%value)
            elif (param_type == BendingParamType.param_types()['complex']):
                raise NotImplementedError
                if isinstance(value, (int, float, complex)):
                    return torch.tensor(complex(value))
                else:
                    raise BendingParameterException('Cannot make complex tensor out of %s'%value)
            elif (param_type == BendingParamType.param_types()['tensor']):
                if isinstance(value, int): 
                    return BendingParamType._parse_tensor_value(torch.tensor(int(value)), param_type)            
                elif isinstance(value, float):  
                    return BendingParamType._parse_tensor_value(torch.tensor(float(value)), param_type)            
                elif isinstance(value, int): 
                    return BendingParamType._parse_tensor_value(torch.tensor(bool(value)), param_type)            
                else:
                    raise BendingParameterException('Cannot make tensor out of %s'%value)
            else:
                raise BendingParameterException('Wrong ParamType: %s'%param_type)

    @staticmethod
    def _from_tensor(tensor) -> _VALID_PARAM_TYPES:
        if tensor.numel() == 0: 
            raise ValueError('got empty tensor in _from_tensor')
        elif tensor.numel() == 1:
            if tensor.dtype in [torch.bool]:
                return bool(tensor) 
            if tensor.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
                return int(tensor)
            elif tensor.dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
                return float(tensor)
            elif tensor.dtype in [torch.complex, torch.complex32, torch.complex64, torch.complex128]: 
                raise NotImplementedError
                return complex(tensor)
            else:
                raise TypeError('cannot parse tensor %s as a native python value'%tensor)
        else:
            return tensor

def get_param_type(param_type: str):
    #damn torchscipt, don't judge me
    if param_type not in BendingParamType.param_types():
        raise BendingParameterException('param_type %s not handled'%param_type)
    return BendingParamType.param_types()[param_type]


class BendingParameter(nn.Module):
    """Named macro controlling one or several callback parameters dynamically.

    Pass a BendingParameter wherever a callback accepts a controllable value::

        c = BendingParameter("amount", value=1., range=[0., 2.])
        bended.bend(Scale(c), "?decoder\\..*weight")
        bended.update("amount", 0.5)        # all dependent callbacks update

    Arithmetic on the object (``2 * c``, ``c + 1.``) builds derived parameters
    sharing the same underlying value with adjusted ``weight`` / ``bias``, so a
    single macro can drive several callbacks at different scales. On jit /
    nn~ export, each parameter becomes ``get_<name>()`` / ``set_<name>(value)``
    accessors with range checking.

    Args:
        name: macro name (used by ``BendedModule.update`` and export setters).
        value: initial value (float / int / bool / tensor; sets ``param_type``).
        as_input: if True, the parameter becomes a *graph placeholder* — an
            extra argument of the bended forward — instead of a stored value
            (used for signal-rate control in nn~).
        weight / bias: affine read transform: ``get_value() = value * weight + bias``.
        range: ``[min, max]`` bounds, enforced on ``set_value`` (and in
            scripted modules) when ``clamp`` is set.
        clamp: clamp ``get_value()`` into ``range``.
    """

    def __init__(self,
                 name: str,
                 value: Any,
                 as_input: bool = False,
                 weight: Optional[float] = None,
                 bias: Optional[float] = None,
                 range: Tuple[Optional[float], Optional[float]] = [None, None], 
                 clamp: Optional[bool] = None,
                 **kwargs):
        super().__init__()
        self._name : str = torch.jit.Attribute(name, str)
        self.param_type: int = BendingParamType._param_type_from_obj(value)
        self.value : Any = nn.Parameter(self._to_tensor(value), requires_grad=False)
        self.as_input = as_input
        if self.param_type in [BendingParamType.get_type('bool')]:
            self._make_init_warnings_for_bool(weight=weight, bias=bias, min_range=range[0], max_range=range[1], clamp=clamp)
            self.min_clamp = 0
            self.max_clamp = 1
            self.clamp = True
            self.register_buffer("weight", torch.tensor(1.))
            self.register_buffer("bias", torch.tensor(0.))
        else:
            self.register_buffer("weight", checktensor(weight if weight is not None else 1.))
            self.register_buffer("bias", checktensor(bias if bias is not None else 0.))
            self.min_clamp = range[0]
            self.max_clamp = range[1]
            self.clamp = clamp or False
        self._nodes = {}
        self._kwargs = kwargs
        self._callbacks = []
        # parameters built from this one by arithmetic. They share this object's
        # value tensor, but they have their own callback list, so an update has
        # to reach them explicitly. Kept off nn.Module's child registry on
        # purpose: a derived parameter is a *view*, not a submodule to save.
        object.__setattr__(self, "_derived", [])

    def _make_init_warnings_for_str(self, **attributes):
        for name, val in attributes:
            if val is not None:
                log_warning("provided keyword %s for BendingParameter of type str"%name)

    def _make_init_warnings_for_bool(self, **attributes):
        for name, val in attributes.items():
            if val is not None:
                log_warning("provided keyword %s for BendingParameter of type bool"%name)

    def as_node(self, graph=None):
        if hash(graph) not in self._nodes:
            self._nodes[hash(graph)] = graph.create_node("placeholder", self.name, (self.value,), type_expr=float)
        return self._nodes[hash(graph)]

    @property
    def name(self) -> str:
        if torch.jit.is_scripting():
            return str(self._name)
        else:
            return str(self._name.value)

    def _register_callback(self, cb, name):
        self._callbacks.append(cb)

    def get_value(self) -> torch.Tensor:
        value = self.value.data
        # if self.param_type == BendingParamType.get_type('bool'):
        #     return value
        if self.value.dtype == torch.bool:
            return value
        if self.clamp:
            return self._clamp(value * self.weight.to(value) + self.bias.to(value))
        else:
            return value * self.weight.to(value) + self.bias.to(value)

    def get_python_value(self) -> _VALID_PARAM_TYPES:
        if self.param_type == BendingParamType.get_type('tensor'):
            return self.get_value()
        else:
            return BendingParamType._from_tensor(self.get_value())

    def set_value(self, value: _VALID_PARAM_TYPES, update: bool = True) -> None:
        if value is not None: 
            value = self._to_tensor(value)
            if value is not None: 
                if self.clamp:
                    value = self._clamp(value)
                else:
                    val_real = value if not torch.is_complex(value) else value.abs()
                    if self.min_clamp is not None:
                        if (val_real < self.min_clamp).any():
                            raise BendingParameterException(f'tried to set value < min_clamp = {self.min_clamp}, but got {value}')
                    if self.max_clamp is not None:
                        if (val_real > self.max_clamp).any():
                            raise BendingParameterException(f'tried to set value > max_clamp = {self.max_clamp}, but got {value}')

                if torch.jit.is_scripting():
                    self.value.set_(value.to(self.value))
                else:
                    # Write through the existing storage. Parameters derived by
                    # arithmetic share this tensor — rebinding `.data` would give
                    # this object a fresh one and silently strand them on the old
                    # value, which is the whole premise of `2 * macro`.
                    tgt = self.value.data
                    if tgt.shape == value.shape and tgt.dtype == value.dtype:
                        tgt.copy_(value)
                    else:
                        self.value.data = value
                if not torch.jit.is_scripting():
                    # a device move may have unshared them since the last write
                    self._sync_derived()
                    if update:
                        self._update_callbacks()

    def _update_callbacks(self) -> None:
        for i, cb in enumerate(self._callbacks):
            cb.update()
        # derived parameters read the same value but hold their own callbacks
        for child in getattr(self, "_derived", []):
            child._update_callbacks()

    def _derive(self, weight, bias) -> "BendingParameter":
        """A parameter reading ``value * weight + bias`` off *this* one's value.

        Unclamped by construction: the clamp belongs to the macro's own range
        (what you may set it to), while a derived parameter's job is to map that
        range onto whatever the target actually wants.
        """
        child = BendingParameter(name=self.name, value=self.value,
                                 weight=weight, bias=bias,
                                 range=[self.min_clamp, self.max_clamp])
        getattr(self, "_derived").append(child)
        return child

    def _rename(self, new_name: str) -> None:
        """Rename this parameter and everything derived from it.

        A derived parameter answers to the same name — it is a view onto this
        one's value — and that name is what the callbacks' generated forward
        calls its argument, so they have to move together.
        """
        self._name = torch.jit.Attribute(str(new_name), str)
        for child in getattr(self, "_derived", []) or []:
            child._rename(new_name)

    def _sync_derived(self) -> None:
        """Re-link everything derived from this parameter to its value tensor.

        Arithmetic makes a derived parameter share the *same* value tensor — that
        sharing is what makes ``2 * macro`` follow the macro. It survives writing
        through the tensor, but not rebinding it, and ``Module.to()`` rebinds:
        it replaces ``value.data`` with a tensor on the new device rather than
        copying into the old one. The two then come apart silently, and the
        derived parameter keeps answering with whatever it last held — which is a
        macro that moves on screen and does nothing to the model.

        Re-pointing the children at the parent's tensor restores the link, and
        puts them on the parent's device while it is at it.
        """
        for child in getattr(self, "_derived", None) or []:
            try:
                src = self.value.data
                dst = child.value.data
                if dst.data_ptr() != src.data_ptr() or dst.device != src.device:
                    child.value.data = src
            except Exception:
                pass
            child._sync_derived()

    def _apply(self, *args, **kwargs):
        """``.to()`` / ``.cuda()`` land here; re-link the derived parameters after."""
        out = super()._apply(*args, **kwargs)
        if not torch.jit.is_scripting():
            self._sync_derived()
        return out

    def _release_derived(self, child) -> bool:
        """Stop driving *child*, and any intermediate left with nothing to feed.

        ``macro * span + lo`` evaluates in two steps, so what a target ends up
        holding is a grandchild: the multiply hangs off the macro, the add hangs
        off the multiply. Walk the branch to find it — releasing only direct
        children would leave the old arithmetic on the update path for good.

        Returns True when it was found.
        """
        derived = getattr(self, "_derived", None)
        if not derived:
            return False
        for c in list(derived):
            if c is child:
                derived.remove(c)
                return True
            if c._release_derived(child):
                # keep the intermediate only while something still reads it
                if not getattr(c, "_derived", None) and not c._callbacks:
                    derived.remove(c)
                return True
        return False

    def _clamp(self, value: torch.Tensor):
        if self.min_clamp is None and self.max_clamp is None:
            return value
        else:
            return torch.clamp(value, self.min_clamp, self.max_clamp)

    def _to_tensor(self, obj: _VALID_PARAM_TYPES) -> torch.Tensor | None:
        if obj is None: 
            return None
        else:
            if isinstance(obj, (int, float, bool, torch.Tensor, torch.nn.Parameter)):
                return BendingParamType._to_tensor(obj, self.param_type)
            else:
                raise TypeError('BendingParameter values can only be int or float')

    @torch.jit.export
    def __float__(self):
        return float(self.value)

    @torch.jit.export
    def __int__(self):
        return int(self.value)

    @torch.jit.export
    def __complex__(self):
        return complex(self.value)

    @torch.jit.export
    def __bool__(self):
       return bool(self.value)
    
    @torch.jit.export
    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return "BendingParameter(name=%s, value=%s)"%(self.name, self.get_value().data)

    def _check_arithmetics_available(self) -> NoReturn:
        # str and bool are the types *without* arithmetics; float/int/tensor have it
        if self.param_type in [BendingParamType.get_type('str'), BendingParamType.get_type('bool')]:
            raise TypeError("BendingParameter of type str or bool cannot have arithmetics")

    def __add__(self , obj):
        self._check_arithmetics_available()
        if not isinstance(obj, (int, float)):
            raise TypeError('BendingParameter can only be added to int, float, or scalars')
        return self._derive(self.weight, self.bias + obj)

    def __radd__(self, obj):
        return self.__add__(obj)

    def __sub__(self, obj):
        self._check_arithmetics_available()
        if not isinstance(obj, (int, float)):
            raise TypeError('BendingParameter can only be subtracted by int, float, or scalars')
        return self._derive(self.weight, self.bias - obj)

    def __rsub__(self, obj):
        self._check_arithmetics_available()
        if not isinstance(obj, (int, float)):
            raise TypeError('BendingParameter can only be subtracted from int, float, or scalars')
        return self._derive(-self.weight, obj - self.bias)

    def __mul__(self, obj):
        self._check_arithmetics_available()
        if not isinstance(obj, (int, float)):
            raise TypeError('BendingParameter can only be multiplied by int, float, or scalars')
        return self._derive(self.weight * obj, self.bias * obj)

    def __rmul__(self, obj):
        return self.__mul__(obj)

    def __call__(self):
        return self.get_value()
