from typing import Union, Optional, Any, Tuple, Callable, Dict
import torch
import torch.nn as nn
from numbers import Real, Integral
from enum import Enum
from torchbend.utils import checktensor


class BendingParameterException(Exception):
    pass

_VALID_PARAM_TYPES = Union[float, int, torch.Tensor]


class BendingParamType():
    
    @staticmethod
    def get_param_type(param_type: str):
        #damn torchscipt, don't judge me
        _param_types = {'float': 1, 'int': 2} 
        if param_type not in _param_types:
            raise BendingParameterException('param_type %s not handled'%param_type)
        return _param_types[param_type]

    @staticmethod
    def _param_type_from_obj(obj: _VALID_PARAM_TYPES) -> int:
        #damn torchscipt, don't judge me
        _param_types = {'float': 1, 'int': 2} 
        # if torch.is_tensor(obj):
        if torch.jit.isinstance(obj, torch.Tensor):
            assert obj.numel() == 1, "Got non-scalar tensor for BendingParameter value"
            if obj.dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
                return _param_types['float']
            elif obj.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
                return _param_types['int']
            else:
                raise BendingParameterException("tensor dtype not handled : %s"%obj.dtype)
        elif isinstance(obj, int):
            return _param_types['int']
        elif isinstance(obj, float):
            return _param_types['float']
        else:
            raise BendingParameterException('cannot retrieve param type from type : %s'%(type(obj)))

    @staticmethod
    def _to_tensor(value: _VALID_PARAM_TYPES, param_type: int) -> torch.Tensor:
        #damn torchscipt, don't judge me
        #TODO make a generative code for param_types?
        _param_types = {'float': 1, 'int': 2} 
        #TODO handle general float types
        if torch.jit.isinstance(value, torch.Tensor):
            if (param_type == _param_types['float']):
                return value.float()
            elif (param_type == _param_types['int']):
                return value.int()
            else:
                raise BendingParameterException('Wrong ParamType: %s'%param_type)
        else:
            if (param_type == _param_types['float']):
                value_float: float = float(value)
                return torch.tensor(value_float)
            elif (param_type == _param_types['int']):
                value_int: int = int(value)
                return torch.tensor(value_int)
            else:
                raise BendingParameterException('Wrong ParamType: %s'%param_type)



def get_param_type(param_type: str):
    #damn torchscipt, don't judge me
    _param_types = {'float': 1, 'int': 2}
    if param_type not in _param_types:
        raise BendingParameterException('param_type %s not handled'%param_type)
    return _param_types[param_type]

"""
def _param_type_from_obj(obj):
    #damn torchscipt, don't judge me
    _param_types = {'float': 1, 'int': 2}
    if torch.is_tensor(obj):
        assert obj.numel() == 1, "Got non-scalar tensor for BendingParameter value"
        if obj.dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
            return _param_types['float']
        elif obj.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
            return _param_types['int']
        else:
            raise BendingParameterException("tensor dtype not handled : %s"%obj.dtype)
    elif isinstance(obj, Integral):
        return _param_types['int']
    elif isinstance(obj, Real):
        return _param_types['float']
    else:
        raise BendingParameterException('cannot retrieve param type from type : %s'%(type(obj)))

def _dtype_from_param_type(param_type):
    #TODO handle general float types
    if (param_type == param_types['float']):
        return torch.float32
    elif (param_type == param_types['int']):
        return torch.int64
    else:
        raise BendingParameterException('Wrong ParamType: %s'%param_type)
    
"""

class BendingParameter(nn.Module):
    def __init__(self, 
                 name: str,
                 value: Any,
                 weight: float = 1.0,
                 bias: float = 0.0,
                 range: Tuple[Union[float, None], Union[float, None]] = [None, None], 
                 clamp: bool = False,
                 **kwargs):
        super().__init__()
        self._name : str = torch.jit.Attribute(name, str)
        self.param_type: int = BendingParamType._param_type_from_obj(value)
        self.value : Any = nn.Parameter(self._to_tensor(value), requires_grad=False)
        self.register_buffer("weight", checktensor(weight))
        self.register_buffer("bias", checktensor(bias))
        self.min_clamp = range[0]
        self.max_clamp = range[1]
        self.clamp = clamp
        self._nodes = {}
        self._kwargs = kwargs

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

    # def slider(self, **add_kwargs):
    #     kwargs = self._kwargs
    #     kwargs.update(add_kwargs)
    #     kwargs['start'] = kwargs.get('start', self.min_clamp)
    #     kwargs['end'] = kwargs.get('end', self.max_clamp)
    #     kwargs['name'] = self.name
    #     kwargs['value'] = float(self.value)
    #     slider = pn.widgets.FloatSlider(**kwargs)
    #     return slider

    def get_value(self) -> torch.Tensor:
        if self.clamp:
            return self._clamp(self.value * self.weight + self.bias)
        else:
            return self.value

    def get_python_value(self) -> Union[int, float]:
        if self.value.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
            return int(self.get_value())
        elif self.value.dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
            return float(self.get_value())
        else:
            raise TypeError('cannot parse tensor %s as a native python value'%self.get_value())

    def set_value(self, value: _VALID_PARAM_TYPES) -> None:
        if not isinstance(value, torch.Tensor):
            value = self._to_tensor(value)
        if self.clamp:
            value = self._clamp(value)
        else:
            if self.min_clamp is not None:
                if value < self.min_clamp:
                    raise BendingParameterException(f'tried to set value < min_clamp = {self.min_clamp}, but got {value}')
            if self.max_clamp is not None:
                if value > self.min_clamp:
                    raise BendingParameterException(f'tried to set value > max_clamp = {self.max_clamp}, but got {value}')

        if torch.jit.is_scripting():
            self.value.set_(value)
        else:
            self.value.data = value

    def _clamp(self, value: torch.Tensor):
        if self.min_clamp is None and self.max_clamp is None:
            return value
        else:
            return torch.clamp(value, self.min_clamp, self.max_clamp)

    def _to_tensor(self, obj: Union[float, int]) -> torch.Tensor:
        if isinstance(obj, (int, float)):
            return BendingParamType._to_tensor(obj, self.param_type)
        else:
            raise TypeError('BendingParameter values can only be int or float')

    @torch.jit.export
    def __float__(self):
        return float(self.value)

    @torch.jit.export
    def __int__(self):
        return int(self.value)


    def __repr__(self):
        return "BendingParameter(name=%s, value=%s)"%(self.name, self.get_value())

    def __add__(self , obj):
        if not isinstance(obj, (int, float)):
            raise TypeError('BendingParameter can only be added to int, float, or scalars')
        return BendingParameter(name=self.name, value=self.value, weight=self.weight, bias=self.bias+obj, range=[self.min_clamp, self.max_clamp])

    def __radd__(self, obj):
        return self.__add__(obj)

    def __sub__(self, obj):
        return BendingParameter(name=self.name, value=self.value, weight=self.weight, bias=self.bias-obj, range=[self.min_clamp, self.max_clamp])

    def __rsub__(self, obj):
        return BendingParameter(name=self.name, value=self.value, weight=-self.weight, bias=self.bias+obj, range=[self.min_clamp, self.max_clamp])

    def __mul__(self, obj):
        if not isinstance(obj, (int, float)):
            raise TypeError('BendingParameter can only be added to int, float, or scalars')
        return BendingParameter(name=self.name, value=self.value, weight=self.weight * obj, bias=self.bias, range=[self.min_clamp, self.max_clamp])

    def __rmul__(self, obj):
        return self.__mul__(obj)

    def __call__(self):
        return self.get_value()

# ___________________________________________________________________________________________
# parameter setter


def float_parameter_setter(parameter: BendingParameter, name: Optional[str] = None) -> Callable:
    if name is None:
        parameter_name = parameter.name
    else:
        parameter_name = str(name)

    def float_parameter_clamped(self, value: float, parameter_name: str) -> int:
        if parameter.min_clamp is not None:
            if value < parameter.min_clamp:
                return -1
        if parameter.max_clamp is not None:
            if value > parameter.min_clamp:
                return -1
        self._set_bending_control(parameter_name, torch.tensor(value))
        return 0

    def float_parameter_unclamped(self, value: float, parameter_name: str) -> int:
        self._set_bending_control(parameter_name, torch.tensor(value))
        return 0

    if parameter.clamp:
        return float_parameter_clamped
    else:
        return float_parameter_unclamped


def int_parameter_setter(parameter: BendingParameter, name: Optional[str] = None) -> Callable:
    if name is None:
        parameter_name = parameter.name
    else:
        parameter_name = str(name)

    def int_parameter_clamped(self, value: int, parameter_name: str) -> int:
        if parameter.min_clamp is not None:
            if value < parameter.min_clamp:
                return -1
        if parameter.max_clamp is not None:
            if value > parameter.min_clamp:
                return -1
        self._set_bending_control(parameter_name, torch.tensor(value, dtype=torch.int))
        return 0

    def int_parameter_unclamped(self, value: int, paramater_name: str) -> int:
        self._set_bending_control(parameter_name, torch.tensor(value, dtype=torch.int))
        return 0

    if parameter.clamp:
        return int_parameter_clamped
    else:
        return int_parameter_unclamped


def parameter_setter(parameter: BendingParameter, name: Optional[str] = None)  -> Callable:
    return float_parameter_setter(parameter, name=name)

# ___________________________________________________________________________________________
# parameter getter

def float_parameter_getter(parameter: BendingParameter, name: Optional[str] = None) -> Callable:
    parameter_name = parameter.name if name is None else name
    def float_getter(self, parameter_name: str) -> float:
        return float(self._get_bending_control(parameter_name)) 
    return float_getter


def parameter_getter(parameter: BendingParameter, name: Optional[str] = None) -> Callable:
    func = float_parameter_getter(parameter, name=name)
    func.__globals__ = globals()
    return func