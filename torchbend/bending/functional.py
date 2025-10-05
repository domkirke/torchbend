from typing import Callable, Optional
from types import MethodType
import inspect
import torch
from .callback import BendingCallback
from ..utils import _replace_placeholders, _import_defs_from_tmpfile


bend_input_pattern = """

{{FN_CODE}}

def bend_input(fn):
    def bend_input(self, x, name: Optional[str] = None, seed: int = 0, {{SIG_PARAMS}}):
        if seed is not None:
            torch.manual_seed(int(seed))
        return fn(x, {{PARAMS_CALL}})
    return bend_input

def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor = None, {{SIG_PARAMS}}) -> None:
    new_param = {{FN}}(cache, {{PARAMS_CALL}})
    param.set_(new_param)
"""

class Lambda(BendingCallback):
    weight_compatible = True 
    activation_compatible = True 
    controllable_params = {'seed': (int, 0)}

    def __init__(self, fn: Callable, seed: int = 0, **kwargs):
        self.controllable_params = {**self.retrieve_params_from_callback(fn), **self.controllable_params}
        bend_input_code = _replace_placeholders(
            bend_input_pattern, 
            sig_params=self.get_params_for_signature(),
            fn=fn.__name__, 
            fn_code = inspect.getsource(fn),
            params_call=self.get_params_for_call()
        )
        dynamic_calls = _import_defs_from_tmpfile(bend_input_code, globals(), locals())
        self.bend_input = MethodType(dynamic_calls['bend_input'](fn), self)
        self.apply_to_param = MethodType(dynamic_calls['apply_to_param'], self)
        super().__init__(seed=seed, **kwargs)

    def get_params_for_signature(self):
        params = []
        for name, (type_expr, default) in self.controllable_params.items(): 
            if name == "seed": continue
            type_expr_str = getattr(type_expr, "__name__", None) or type_expr.__repr__()
            params.append(f"{name}: {type_expr_str} = {default}")
        return ", ".join(params)

    def get_params_for_call(self):
        params = []
        for name, (type_expr, default) in self.controllable_params.items(): 
            if name == "seed": continue
            params.append(f"{name}={name}")
        return ", ".join(params)

    @classmethod
    def retrieve_params_from_callback(cls, fn):
        signature = inspect.signature(fn)
        controllable_params = {}
        for name, param in dict(signature.parameters).items():
            if name in ["self", "x"]: continue
            type_expr = param.annotation
            if type_expr == inspect._empty: type_expr = torch.Tensor
            default = param.default
            if default == inspect._empty:
                type_expr = Optional[type_expr]
                default = None
            controllable_params[name] = (type_expr, default)
        return controllable_params

    # def bend_input(self, x, name: Optional[str] = None, seed: torch.Tensor | None = None, **kwargs):
    #     if seed is not None:
    #         torch.manual_seed(int(seed))
    #     return self._callable(x, **kwargs)#, param)

    # def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor = None, **kwargs) -> None:
    #     new_param = self._callable.value(cache, **kwargs)#, param)
    #     param.set_(new_param)
