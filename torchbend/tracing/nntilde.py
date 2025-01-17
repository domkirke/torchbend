from typing import List, Dict, Callable
import re
from types import MethodType, FunctionType
from functools import partialmethod, partial
import abc
import torch
import torch.fx as fx
from torch import nn
from ..bending.parameter import BendingParameter, get_param_type
from .module import BendedModule
import nn_tilde


class BendingNNTildeException(Exception):
    pass


attribute_template = """
def get_{{NAME}}(self) -> {{TYPE_EXPR}}:
    return float(self._get_bending_control(\"{{NAME}}\")) 

def set_{{NAME}}(self, value: {{TYPE_EXPR}}) -> int:
    try: 
        self._set_bending_control(\"{{NAME}}\", torch.tensor(value, dtype={{DTYPE}}))
        return 0
    except BendingParameterException:
        return -1
"""


def _defs_from_template(template, **kwargs):
    code = template
    local_namespace = {}
    for k, v in kwargs.items():
        pattern = re.compile(r'\{\{%s\}\}'%(k.upper()))
        iterations = list(pattern.finditer(code))
        while len(iterations) > 0:
            start, end = iterations[0].start(), iterations[0].end()
            code = code[:start] + str(v) + code[end:]
            iterations = list(pattern.finditer(code))
    #TODO check if no {{}} left
    exec(code, {}, local_namespace)
    return local_namespace


def _template_from_param(param: BendingParameter, **kwargs):
    kwargs['name'] = kwargs.get('name', param.name)
    if param.param_type == get_param_type("float"):
        kwargs['dtype'] = kwargs.get('dtype', torch.float32)
        kwargs['type_expr'] = kwargs.get('type_expr', "float")
        return _defs_from_template(attribute_template, **kwargs)
    elif param.param_type == get_param_type("int"):
        kwargs['dtype'] = kwargs.get('dtype', torch.int64)
        kwargs['type_expr'] = kwargs.get('type_expr', "int")
        return _defs_from_template(attribute_template, **kwargs)


class BendableNNTildeModule(nn_tilde.Module):

    scripted_methods = []

    def __init__(self, model):
        super().__init__()
        self._import_model(model)
        if getattr(getattr(self, "_register_methods"), "__isabstractmethod__", False):
            raise BendingNNTildeException('_register_methods is not defined for class %s'%type(self))
        self._register_methods(model)
        self._import_bending(model) 

    def _import_model(self, model):
        bended_module = model.bend_module()
        self._bended_modules = []
        for method in self.scripted_methods:
            module = model.graph_module(method, module=bended_module)
            setattr(self, f"_{method}", module)
            self._bended_modules.append(getattr(self, f"_{method}"))

    @abc.abstractmethod
    def _register_methods(self, model):
        pass

    # ____________________________________________________________
    # bending methods

    def _full_param_dict(self):
        param_dict = {}
        for module in self._bended_modules:
            for k, v in dict(module.named_parameters()).items():
                if k in param_dict:
                    if id(param_dict[k]) != id(v):
                        print('[Warning] param %s does not coincide between modules')
                else:
                    param_dict[k] = v
        return param_dict

    def _set_attribute_callbacks(self, param: BendingParameter) -> Dict[str, Callable]:
        funcs = _template_from_param(param)
        for name, func in funcs.items():
            setattr(self, name, MethodType(func, self))

    def _import_bending_ops(self, model):
        self._controllables = nn.ModuleList(model.controllables.values())
        self._bending_callbacks = nn.ModuleList([m.script() for m in model._bending_callbacks])
        self._controllables_hash = torch.jit.Attribute({}, Dict[str, List[int]])
        for v in self._controllables:
            for i, b in enumerate(self._bending_callbacks):
                if v in b:
                    self._controllables_hash.value[v.name] = self._controllables_hash.value.get(v.name, []) + [i]
            self._set_attribute_callbacks(v)
            self.register_attribute(v.name, v.get_python_value())
                
    def _update_bended_weights(self, model):
        param_dict = self._full_param_dict()
        model_param_dict = dict(model.named_parameters())
        for param, cb_list in model.bended_params.items():
            if param not in param_dict:
                print('[Warning] Bended parameter %s not found in current module.'%param)
                continue
            for cb in cb_list:
                # cb.update_weight(model.get_parameter(param), param_dict[param])
                cb.update_weight(model_param_dict[param], param_dict[param])

    def _update_bended_activations(self, model):
        pass

    def _import_bending(self, model):
        self._import_bending_ops(model)
        self._update_bended_weights(model)
        self._update_bended_activations(model)

    # ____________________________________________________________
    # operational methods

    def _update_weights(self, name: str):
        with torch.no_grad():
            callbacks = self._controllables_hash[name]
            for i, c in enumerate(self._bending_callbacks):
                for j in callbacks:
                    if i == j: c.apply()

    @torch.jit.export
    def _get_bending_control(self, name: str) -> torch.Tensor:
        """returns value of a bending control by name"""
        # grrr
        for i, v in enumerate(self._controllables):
            if v.name == name:
                return v.value.data
        raise ModuleNotFoundError("No bending control named %s in model %s"%(name, self))

    @torch.jit.export
    def _set_bending_control(self, name: str, value: torch.Tensor) -> None:
        """set a bending control with name and value"""
        for v in self._controllables:
            if v.name == name:
                v.set_value(value)
        self._update_weights(name)