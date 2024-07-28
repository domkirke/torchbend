import inspect
from typing import List, Dict
from types import MethodType
import torch, torch.nn as nn
from .utils import _defs_from_template, _resolve_code, _import_defs_from_tmpfile

class ScriptedBendedException(Exception):
    pass

method_template = """
@torch.jit.export
def {{NAME}}{{SIGNATURE}}:
    return self._{{NAME}}{{OUTS}}
"""

class ScriptedBendedModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._import_model(model)
        self._import_bending(model) 

    def _register_imported_methods(self, methods: List[str]):
        codes = []
        for m in methods:
            signature = inspect.signature(getattr(self, "_"+m).forward)
            signature_str = "(self, " + str(signature)[1:]
            outs = "(" + ",".join([f"{i}={i}" for i in signature.parameters]) + ")"
            codes.append(_resolve_code(method_template, name=m, signature=signature_str, outs=outs))
        codes = "\n".join(codes)
        methods_defs = _import_defs_from_tmpfile(codes, gl=globals())
        for k, v in methods_defs.items():
            setattr(self, k, MethodType(v, self))

    def _import_model(self, model):
        bended_module = model.bend_module()
        self._bended_modules = []
        for method in model._graphs.keys():
            module = model.graph_module(method, module=bended_module, make_jit_compatible=True)
            setattr(self, f"_{method}", module)
            self._bended_modules.append(getattr(self, f"_{method}"))
        self._register_imported_methods(model._graphs.keys())

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

    def _import_bending_ops(self, model):
        self._controllables = nn.ModuleList(model.controllables.values())
        self._bending_callbacks = nn.ModuleList(model._bending_callbacks)
        self._controllables_hash = torch.jit.Attribute({}, Dict[str, List[int]])
        for v in self._controllables:
            for i, b in enumerate(self._bending_callbacks):
                if v in b:
                    self._controllables_hash.value[v.name] = self._controllables_hash.value.get(v.name, []) + [i]
            # self._set_attribute_callbacks(v)
            # self.register_attribute(v.name, v.get_python_value())
                
    def _update_bended_weights(self, model):
        param_dict = self._full_param_dict()
        model_param_dict = dict(model.named_parameters())
        for param, cb_list in model.bended_params.items():
            if param not in param_dict:
                print('[Warning] Bended parameter %s not found in current module.'%param)
                continue
            for cb in cb_list:
                # cb.update_parameter(model.get_parameter(param), param_dict[param])
                cb.update_parameter(model_param_dict[param], param_dict[param])

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