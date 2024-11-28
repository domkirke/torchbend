from typing import Union, Optional, Any, Tuple, Callable, Dict, NoReturn
import numbers
import torch
import torch.nn as nn
from numbers import Real, Integral
from enum import Enum
from torchbend import log_error, log_warning
from torchbend.utils import checktensor


class BendingParameterException(Exception):
    pass


_VALID_PARAM_TYPES = Union[float, int, bool, complex, torch.Tensor]
_VALID_PARAM_NATIVE_TYPES = Union[float, int, str, bool, complex]

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
        return {'float': 1, 'int': 2, 'bool': 3, 'complex': 4}
    @staticmethod
    def param_hash():
        return {v: k for k, v in BendingParamType.param_types().items()}
    
    @staticmethod
    def _param_type_from_obj(obj: _VALID_PARAM_TYPES) -> int:
        #damn torchscipt, don't judge me
        # if torch.is_tensor(obj):
        if torch.jit.isinstance(obj, torch.Tensor):
            assert obj.numel() == 1, "Got non-scalar tensor for BendingParameter value"
            if obj.dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
                return BendingParamType.param_types()['float']
            elif obj.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
                return BendingParamType.param_types()['int']
            elif obj.dtype in [torch.complex, torch.complex32, torch.complex64, torch.complex128]:
                return BendingParamType.param_types()['complex']
            elif obj.dtype in [torch.bool]:
                return BendingParamType.param_types()['bool']
            else:
                raise BendingParameterException("tensor dtype not handled : %s"%obj.dtype)
        elif isinstance(obj, bool):
            return BendingParamType.param_types()['bool']
        elif isinstance(obj, numbers.Integral):
            return BendingParamType.param_types()['int']
        elif isinstance(obj, numbers.Real):
            return BendingParamType.param_types()['float']
        elif isinstance(obj, numbers.Complex):
            return BendingParamType.param_types()['complex']
        elif isinstance(obj, str):
            return BendingParamType.param_types()['str']
        else:
            raise BendingParameterException('cannot retrieve param type from type : %s'%(type(obj)))

    @staticmethod
    def get_type(type_obj) -> int:
        return BendingParamType.param_types()[type_obj]

    @staticmethod
    def _to_tensor(value: _VALID_PARAM_TYPES, param_type: int) -> torch.Tensor:
        #damn torchscipt, don't judge me
        #TODO make a generative code for param_types?
        # _param_types = {'float': 1, 'int': 2} 
        #TODO handle general float types
        if torch.jit.isinstance(value, torch.Tensor):
            if (param_type == BendingParamType.param_types()['bool']):
                return value.byte()
            elif (param_type == BendingParamType.param_types()['float']):
                return value.float()
            elif (param_type == BendingParamType.param_types()['int']):
                return value.int()
            elif (param_type == BendingParamType.param_types()['complex']):
                return torch.view_as_complex(value)
            else:
                raise BendingParameterException('Wrong ParamType: %s'%param_type)
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
                if isinstance(value, (int, float, complex)):
                    return torch.tensor(complex(value))
                else:
                    raise BendingParameterException('Cannot make complex tensor out of %s'%value)
            else:
                raise BendingParameterException('Wrong ParamType: %s'%param_type)

    @staticmethod
    def _from_tensor(tensor) -> _VALID_PARAM_NATIVE_TYPES:
        if tensor.dtype in [torch.bool]:
            return bool(tensor) 
        if tensor.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
            return int(tensor)
        elif tensor.dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
            return float(tensor)
        elif tensor.dtype in [torch.complex, torch.complex32, torch.complex64, torch.complex128]: 
            return complex(tensor)
        else:
            raise TypeError('cannot parse tensor %s as a native python value'%tensor)

def get_param_type(param_type: str):
    #damn torchscipt, don't judge me
    if param_type not in BendingParamType.param_types():
        raise BendingParameterException('param_type %s not handled'%param_type)
    return BendingParamType.param_types()[param_type]


class BendingParameter(nn.Module):
    
    def __init__(self, 
                 name: str,
                 value: Any,
                 weight: Optional[float] = None,
                 bias: Optional[float] = None,
                 range: Tuple[Optional[float], Optional[float]] = [None, None], 
                 clamp: Optional[bool] = None,
                 **kwargs):
        super().__init__()
        self._name : str = torch.jit.Attribute(name, str)
        self.param_type: int = BendingParamType._param_type_from_obj(value)
        # if self.param_type in [BendingParamType.get_type('str')]:
        #     self.value: str = value
        #     self._make_init_warnings_for_str(weight=weight, bias=bias, min_range=range[0], max_range=range[1], clamp=clamp)
        # else:
        self.value : Any = nn.Parameter(self._to_tensor(value), requires_grad=False)
        if self.param_type in [BendingParamType.get_type('bool')]:
            self._make_init_warnings_for_bool(weight=weight, bias=bias, min_range=range[0], max_range=range[1], clamp=clamp)
        else:
            self.register_buffer("weight", checktensor(weight if weight is not None else 1.))
            self.register_buffer("bias", checktensor(bias if bias is not None else 0.))
            self.min_clamp = range[0]
            self.max_clamp = range[1]
            self.clamp = clamp or False
            self._nodes = {}
            self._kwargs = kwargs
            self._callbacks = {}

    def _make_init_warnings_for_str(self, **attributes):
        for name, val in attributes:
            if val is not None:
                log_warning("provided keyword %s for BendingParameter of type str"%name)

    def _make_init_warnings_for_bool(self, **attributes):
        for name, val in attributes:
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
        self._callbacks[cb] = name

    def get_value(self) -> torch.Tensor:
        if self.clamp:
            return self._clamp(self.value * self.weight + self.bias)
        else:
            return self.value

    def get_python_value(self) -> _VALID_PARAM_NATIVE_TYPES:
        return BendingParamType._from_tensor(self.get_value())

    def set_value(self, value: _VALID_PARAM_TYPES, update: bool = True) -> None:
        # if str, just set value
        # if not str, check and clamp
        if not isinstance(value, torch.Tensor):
            value = self._to_tensor(value)
        if self.clamp:
            value = self._clamp(value)
        else:
            val_real = value if not torch.is_complex(value) else value.abs()
            if self.min_clamp is not None:
                if val_real < self.min_clamp:
                    raise BendingParameterException(f'tried to set value < min_clamp = {self.min_clamp}, but got {value}')
            if self.max_clamp is not None:
                if val_real > self.max_clamp:
                    raise BendingParameterException(f'tried to set value > max_clamp = {self.max_clamp}, but got {value}')

        if torch.jit.is_scripting():
            self.value.set_(value)
        else:
            self.value.data = value
            #TODO not compatible with scripting yet. Find a solution?
            self._update_callbacks()

    def _update_callbacks(self):
        for cb, name in self._callbacks.items():
            cb.update()

    def _clamp(self, value: torch.Tensor):
        if self.min_clamp is None and self.max_clamp is None:
            return value
        else:
            return torch.clamp(value, self.min_clamp, self.max_clamp)

    def _to_tensor(self, obj: _VALID_PARAM_TYPES) -> torch.Tensor:
        if isinstance(obj, (int, float, bool, complex, torch.Tensor, torch.nn.Parameter)):
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
        if self.param_type not in [BendingParamType.get_type('str'), BendingParamType.get_type('bool')]:
            raise TypeError("BendingParameter of type str or bool cannot have arithmetics")

    def __add__(self , obj):
        if not isinstance(obj, (int, float)):
            raise TypeError('BendingParameter can only be added to int, float, or scalars')
        self._check_arithmetics_available()
        return BendingParameter(name=self.name, value=self.value, weight=self.weight, bias=self.bias+obj, range=[self.min_clamp, self.max_clamp])

    def __radd__(self, obj):
        return self.__add__(obj)

    def __sub__(self, obj):
        return BendingParameter(name=self.name, value=self.value, weight=self.weight, bias=self.bias-obj, range=[self.min_clamp, self.max_clamp])

    def __rsub__(self, obj):
        self._check_arithmetics_available()
        return BendingParameter(name=self.name, value=self.value, weight=-self.weight, bias=self.bias+obj, range=[self.min_clamp, self.max_clamp])

    def __mul__(self, obj):
        self._check_arithmetics_available()
        if not isinstance(obj, (int, float)):
            raise TypeError('BendingParameter can only be added to int, float, or scalars')
        return BendingParameter(name=self.name, value=self.value, weight=self.weight * obj, bias=self.bias, range=[self.min_clamp, self.max_clamp])

    def __rmul__(self, obj):
        self._check_arithmetics_available()
        return self.__mul__(obj)

    def __call__(self):
        return self.get_value()
