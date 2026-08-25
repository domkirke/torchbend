import torch, torch.nn as nn
import torchbend as tb
from torchbend.bending.callback import BendingCallback
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

class Lambda(BendingCallback):
    """Wraps an arbitrary Python function as a bending callback.

    Usage: Lambda(fn, **param_defaults)

    The function signature determines the controllable params. Cannot be
    created from the graph viewer UI (ui_compatible=False); use it
    programmatically and it will appear in the bindings list.
    """
    ui_compatible = True 
    activation_compatible = True
    weight_compatible = True
    jit_compatible = True
    controllable_params = {}
    _param_ui = {
        'prob': {
            'widget': 'field',
            'guard': lambda v, cb: True if type(cb)._check_valid_fn(v, cb) else ValueError(f"prob must be in [0, 1], got {v:.4f}"),
            'factory': lambda x: Lambda._create_function(x)
        }
    }

    @classmethod
    def _check_valid_fn(cls, value, callback):
        return True

    @classmethod
    def _create_function(cls, fn, **kwargs):
        params = retrieve_params_from_callback(fn)
        resolved_code = _replace_placeholders(
            lambda_code,
            fn_class_params=get_controllables_from_params(params),
            fn_forward_args=get_params_for_call(params),
            fn_forward_sig=get_params_for_signature(params)
        )
        generator = _import_defs_from_tmpfile(resolved_code)['generate_class']
        # generate_class builds and returns a fully initialised instance;
        # since it is not an instance of Lambda, Python skips Lambda.__init__.
        return generator(fn, **kwargs)


    def __new__(cls, fn, **kwargs):
        return cls._create_function(fn, **kwargs)

__all__ = ["Lambda"]