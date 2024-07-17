from typing import Union, Optional, Any, Tuple
import torch
import torch.nn as nn
from torchbend.utils import checktensor


class BendingParameter(nn.Module):
    def __init__(self, 
                 name: str,
                 value: Any,
                 weight: float = 1.0,
                 bias: float = 0.0,
                 range: Tuple[Union[float, None], Union[float, None]] = [None, None], 
                 **kwargs):
        super().__init__()
        self.name : str = name
        self.value : Any = nn.Parameter(self._to_tensor(value), requires_grad=False)
        self.register_buffer("weight", checktensor(weight))
        self.register_buffer("bias", checktensor(bias))
        self.min_clamp = range[0]
        self.max_clamp = range[1]
        self._nodes = {}
        self._kwargs = kwargs

    def as_node(self, graph=None):
        if hash(graph) not in self._nodes:
            self._nodes[hash(graph)] = graph.create_node("placeholder", self.name, (self.value,), type_expr=float)
        return self._nodes[hash(graph)]

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
        return self._clamp(self.value * self.weight + self.bias)

    def get_python_value(self) -> Union[int, float]:
        if self.value.dtype in [torch.int16, torch.int32, torch.int64]:
            return int(self.get_value())
        elif self.value.dtype in [torch.float16, torch.float32, torch.float64]:
            return float(self.get_value())
        else:
            raise TypeError('cannot parse tensor %s as a native python value'%self.get_value())

    def set_value(self, value: torch.Tensor) -> None:
        if torch.jit.is_scripting():
            self.value.set_(value)
        else:
            if isinstance(value, float):
                value = torch.tensor(value)
            self.value.data = torch.Tensor(value)

    def _clamp(self, value):
        if self.min_clamp is None and self.max_clamp is None:
            return value
        else:
            return torch.clamp(value, self.min_clamp, self.max_clamp)

    def _to_tensor(self, obj: Any):
        if isinstance(obj, (int, float)):
            return torch.tensor(obj)
        elif torch.is_tensor(obj):
            return obj
        else:
            raise TypeError('BendingParameter values can only be int or float')

    def __float__(self):
        return float(self.value)

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


