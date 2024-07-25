from typing import List, Dict
import abc
import torch
import torch.fx as fx
from torch import nn
from .module import BendedModule
import nn_tilde

class BendingNNTildeException(Exception):
    pass


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

    def _import_bending_ops(self, model):
        self._controllables = nn.ModuleList(model.controllables.values())
        self._bending_callbacks = nn.ModuleList(model._bending_callbacks)
        self._controllables_hash = torch.jit.Attribute({}, Dict[str, List[int]])
        for v in self._controllables:
            for i, b in enumerate(self._bending_callbacks):
                if v in b:
                    self._controllables_hash.value[v.name] = self._controllables_hash.value.get(v.name, []) + [i]
            self.register_attribute(v.name, float(v.value))
                
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