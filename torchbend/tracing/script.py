import inspect
from abc import abstractmethod
from typing import List, Dict, Callable, Optional
from types import MethodType
import torch, torch.nn as nn
from ..bending import BendingParameter, get_param_type, BendingCallback, CallbackChain
from .module import BendedModule
from .graphmodule import BendedGraphModule
from . import CONTROLLABLE_TYPES
from ..utils import _resolve_code, _import_defs_from_tmpfile
import nn_tilde

class ScriptedBendedException(Exception):
    pass

former_method_template = """
@torch.jit.export
def {{METHOD_NAME}}{{SIGNATURE}}:
    return self._{{CALLBACK_NAME}}{{INS}}
"""

method_template = """
@torch.jit.export
def {{METHOD_NAME}}{{SIGNATURE}}:
    return self.graph_module.{{CALLBACK_NAME}}{{INS}}
"""

attribute_template = """
@torch.jit.export
def get_{{NAME}}(self) -> {{TYPE_EXPR}}:
    return {{TYPE_EXPR}}(self._get_bending_control(\"{{NAME}}\")) 

@torch.jit.export
def set_{{NAME}}(self, value: {{TYPE_EXPR}}) -> int:
    return self._set_bending_control(\"{{NAME}}\", torch.tensor(value, dtype={{DTYPE}}))
"""

def _template_from_param(param: BendingParameter, template=attribute_template, **kwargs):
    kwargs['name'] = kwargs.get('name', param.name)
    if param.param_type == get_param_type("float"):
        kwargs['dtype'] = kwargs.get('dtype', torch.float32)
        kwargs['type_expr'] = kwargs.get('type_expr', "float")
        return _resolve_code(template, **kwargs)
    elif param.param_type == get_param_type("int"):
        kwargs['dtype'] = kwargs.get('dtype', torch.int64)
        kwargs['type_expr'] = kwargs.get('type_expr', "int")
        return _resolve_code(template, **kwargs)
    elif param.param_type == get_param_type("bool"):
        kwargs['dtype'] = kwargs.get('dtype', torch.uint8)
        kwargs['type_expr'] = kwargs.get('type_expr', "bool")
        return _resolve_code(template, **kwargs)
    else:
        raise TypeError('Type not handled by automatic attribute writing : %s'%(param.param_type))
        

class ScriptedBendedModule(nn.Module):
    method_template = method_template
    attribute_template = attribute_template

    def __init__(self, model: BendedModule, enable_grad: bool = False):
        """
        ScriptedBendedModule is a extension of the nntile.Module that allows : 
        - automatic scripting / graphing of traced methods
        - automatic parsing of BendingParameters, and making attribute callbacks
        - automatic import of BendingCallbacks and corresponding parameters / attributes. 

        To allow dynamic registering of attributes with jit, temporary files are created to export the code in TorchScript.
        """
        super().__init__()
        assert isinstance(model, BendedModule), "ScriptedBendedModule must be initialized with a BendedModule"
        self._original_class = type(model).__name__

        if not hasattr(self, "scripted_methods"):
            setattr(self, "scripted_methods", list(model._graphs.keys()))
        self._import_model(model)
        self._import_bending(model)
        if not enable_grad:
            self._disable_parameter_grad()

    def __repr__(self):
        return f"{type(self).__name__}(original_class={self._original_class}, methods={self._methods}, attributes={self._attributes})"

    @property
    def available_methods(self):
        return self._available_methods
    
    def _import_model(self, model):
        """Import all the registered methods of a BendedModule into GraphModule calls."""
        self._bended_modules = []
        self._available_methods = []
        self.graph_module = model.graph_module(jit_compatible=True)
        self._import_attributes(model)
        self._available_methods = list(self.graph_module.graph.keys())

        for attr in dir(model):
            if hasattr(getattr(model, attr), "_export_to_module"):
                assert attr not in dir(self)
                setattr(self, attr, getattr(model, attr))
        self._register_imported_methods(model._graphs.keys())

    def _import_attributes(self, model, import_buffers=True):
        _attrs_to_import = getattr(model, "_attributes_for_tb_scripting", [])
        for attr in _attrs_to_import:
            current_attr = getattr(self.graph_module, attr, None)
            if current_attr is not None:
                setattr(self.graph_module, attr, current_attr)
            else: 
                current_attr = getattr(model, attr, None)
                if current_attr is not None:
                    setattr(self.graph_module, attr, current_attr)

        # for name, buff in dict(model.named_buffers()).items():
        #     setattr(self.graph_module, name, buff)
    
    def _make_method(self, method_name: str, callback_name: Optional[str] = None):
        callback_name = callback_name or method_name
        signature = inspect.signature(getattr(self.graph_module, callback_name))
        new_params = dict(signature.parameters)
        for k, v in dict(new_params).items():
            if hasattr(v.annotation, "__module__"):
                if v.annotation.__module__ == "typing":
                    # v._annotation = str(v.annotation)
                    new_params[k] = v
        signature._parameters = new_params
            
        signature_str = "(self, " + str(signature)[1:]
        ins = "(" + ",".join([f"{i}={i}" for i in signature.parameters]) + ")"

        code = _resolve_code(self.method_template,
                             method_name=method_name, 
                             callback_name=callback_name, 
                             signature=signature_str, 
                             ins=ins, 
                             _import_modules=['typing'])
        return code

    def _register_imported_methods(self, methods: List[str]):
        codes = []
        for m in methods:
            codes.append(self._make_method(m))
            # if m == "forward":
            #     codes.append(self._make_method("__call__", m))
        codes = "\n".join(codes)
        methods_defs = _import_defs_from_tmpfile(codes, gl=globals())
        for k, v in methods_defs.items():
            if not callable(v): continue
            setattr(self, k, MethodType(v, self))
    
    def _import_bending(self, model):
        """parse and registerbending callbacks and controllables for attribute registereing"""
        self._import_bending_ops(model)
        self._update_bended_weights(model)
        self._update_bended_activations(model)

    def _import_bending_ops(self, model):
        self._controllables = nn.ModuleList(model.controllables().values())
        self._bending_callbacks = nn.ModuleList([m.script() for m in model._bending_callbacks])
        _controllables_hash = torch.jit.Attribute({}, Dict[str, List[int]])
        for v in self._controllables:
            # if not v.as_input:
            self._register_controllable(v, _controllables_hash)
        self._controllables_hash = _controllables_hash
                
    def _update_bended_weights(self, model):
        param_dict = self._full_param_dict()
        model_param_dict = dict(model.named_parameters())
        for param, cb_list in model.bended_params.items():
            if param not in param_dict:
                print('[Warning] Bended parameter %s not found in current module.'%param)
                continue
            for cb in cb_list:
                cb.update_weight(model_param_dict[param], param_dict[param])

    def _update_bended_activations(self, model):
        for k, v in self.graph_module._modules.items():
            if isinstance(v, (CallbackChain, BendingCallback)):
                self.graph_module._modules.__setitem__(k, v.script())

    def _register_controllable(self, controllable, controllables_hash):
        for i, b in enumerate(self._bending_callbacks):
            if controllable in b:
                controllables_hash.value[controllable.name] = controllables_hash.value.get(controllable.name, []) + [i]
        self._set_attribute_callbacks(controllable)
    
    def _set_attribute_callbacks(self, param: BendingParameter) -> Dict[str, Callable]:
        codes = _template_from_param(param, template=self.attribute_template, cls_self=type(self).__name__)
        funcs = _import_defs_from_tmpfile(codes, gl=globals(), lo=locals())
        setattr(self, "set_"+param.name, MethodType(funcs["set_"+param.name], self))
        setattr(self, "get_"+param.name, MethodType(funcs["get_"+param.name], self))
    
    def _full_param_dict(self):
        return dict(self.graph_module.named_parameters())

    def _disable_parameter_grad(self):
        for gm in self._bended_modules:
            for param in gm.parameters():
                param.requires_grad_(False)

    def _get_graph_for_method(self, method):
        assert method in self._available_methods
        method_graph = self.graph_module.graph[method]
        return method_graph

    def _get_placeholders_for_method(self, method):
        method_graph = self._get_graph_for_method(method)
        if not getattr(method_graph, "activations", None): raise ScriptedBendedException("Cannot extract activations from graph for method %s"%method)
        input_placeholders = list(filter(lambda x: x.op == "placeholder", method_graph.nodes))
        return input_placeholders

    def _get_outputs_for_method(self, method):
        method_graph = self._get_graph_for_method(method)
        if not getattr(method_graph, "activations", None): raise ScriptedBendedException("Cannot extract activations from graph for method %s"%method)
        output_placeholders = list(filter(lambda x: x.op == "output", method_graph.nodes))
        return output_placeholders

    # ____________________________________________________________
    # operational methods

    def _update_weights(self, name: str):
        with torch.no_grad():
            if torch.jit.is_scripting():
                callbacks = self._controllables_hash[name]
            else:
                callbacks = self._controllables_hash.value[name]
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
    def _set_bending_control(self, name: str, value: CONTROLLABLE_TYPES) -> int:
        """set a bending control with name and value"""
        if isinstance(value, (int, float)):
            value = torch.full((1,), value)
        elif isinstance(value, bool):
            value = torch.full((1,), int(value)).to(torch.bool)
        for v in self._controllables:
            if v.name == name:
                v.set_value(value)
        self._update_weights(name)
        return 0