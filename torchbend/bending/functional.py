import torch, torch.nn as nn
import torchbend as tb
import inspect
from torchbend.utils import _replace_placeholders, _import_defs_from_tmpfile

def type_as_str(type_expr):
    return getattr(type_expr, "__name__", None) or type_expr.__repr__()

def retrieve_params_from_callback(fn):
        signature = inspect.signature(fn)
        controllable_params = {}
        for name, param in dict(signature.parameters).items():
            if name in ["self", "x", "name"]: continue
            type_expr = param.annotation
            if type_expr == inspect._empty: type_expr = torch.Tensor
            default = param.default
            if default == inspect._empty:
                type_expr = type_expr | None
                default = None
            controllable_params[name] = (type_expr, default)
        return controllable_params

def get_params_for_signature(fn_params):
    params = []
    for name, (type_expr, default) in fn_params.items(): 
        type_expr_str = type_as_str(type_expr)
        params.append(f"{name}: {type_expr_str} = {default}")
    return ", ".join(params)

def get_params_for_call(fn_params):
    params = []
    for name, (type_expr, default) in fn_params.items(): 
        params.append(f"{name}={name}")
    return ", ".join(params)

def get_controllables_from_params(fn_params): 
    return "{" + ", ".join([f"\"{k}\": ({type_as_str(v[0])}, {str(v[1])})" for k, v in fn_params.items()]) + "}"


lambda_code = """
def generate_class(fn, **kwargs):
    import torchbend as tb
    from torch import Tensor
    from typing import Optional
    class Lambda(tb.BendingCallback): 
        activation_compatible = True
        jit_compatible = True
        controllable_params = {{FN_CLASS_PARAMS}}
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def __repr__(self): 
            return f"Lambda(fn={fn.__name__})"

        def bend_input(self, x, name: Optional[str] = None, {{FN_FORWARD_SIG}}):
            return fn(x, {{FN_FORWARD_ARGS}})

        def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor = None, {{FN_FORWARD_SIG}}) -> None:
            new_param = fn(cache, {{FN_FORWARD_ARGS}})
            param.set_(new_param)
    return Lambda(**kwargs)
"""

def Lambda(fn, **kwargs):
    params = retrieve_params_from_callback(fn)
    resolved_code = _replace_placeholders(
        lambda_code, 
        fn_class_params=get_controllables_from_params(params),
        fn_forward_args=get_params_for_call(params),
        fn_forward_sig=get_params_for_signature(params)
    )
    generator = _import_defs_from_tmpfile(resolved_code)['generate_class']
    return generator(fn, **kwargs)

__all__ = ["Lambda"]