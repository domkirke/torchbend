import torch
from dataclasses import dataclass
import re
import math
import inspect
import math
from typing import List, Optional, Any, Tuple
from types import MethodType
import nn_tilde
from .module import BendedModule
from .script import ScriptedBendedModule, ScriptedBendedException
from .utils import tmp_file_session
from ..utils import _resolve_code


method_template = """
@torch.jit.export
def {{METHOD_NAME}}{{SIGNATURE}}:
    return self.graph_module.{{CALLBACK_NAME}}({{INS}})
"""

# specific method templates that split a single input to multi-input.
method_template_n_args = """
@torch.jit.export
def {{METHOD_NAME}}{{SIGNATURE}}:
    {{INS}} = torch.split(x, {{SECTIONS}}, dim=-2)
{{SR_CONVERSION}}
    return self.graph_module.{{CALLBACK_NAME}}({{INS_PARSED}})
"""

class ListAttribute(torch.jit.Attribute):

    def __getitem__(self, item):
        self.value.__getitem__(item)

    def append(self, x): 
        self.value.append(x)


class NNBendedModuleException(Exception):
    pass

@dataclass
class NNBendedMethodAttributes:
    in_channels: int
    in_ratio: int 
    out_channels: int
    out_ratio: int
    input_labels: List[str]
    output_labels: List[str]
    test_method: bool = False

    def as_dict(self): 
        return self.__getstate__()

class NNBendedModule(nn_tilde.Module, ScriptedBendedModule):
    def __init__(self, model, enable_grad: bool = False, force_default: bool = False, sr: int | None = None):
        assert isinstance(model, BendedModule), "NNBendedModule must be initialized with a BendedModule"
        self._init_nntilde_module(sr=sr)
        # self._methods = ListAttribute([], List[str])
        self._attributes = ListAttribute([], List[str])
        self._get_set_candidates = {}
        ScriptedBendedModule.__init__(self, model, enable_grad=enable_grad)
        self._search_for_getter_and_setters(model.module)

        if hasattr(model, "register_nntilde_attributes"):
            if not getattr(getattr(model, "register_nntilde_attributes"), "__isabstractmethod__", False):
                model.register_nntilde_attributes(self)

        self._register_methods(model, force_default)
        self._reset_get_set_candidates()

    def _init_nntilde_module(self, sr = None):
        self._methods = []
        self._attributes = []
        self._buffer_attributes = torch.jit.Attribute([], List[str])
        self.tmp_file_session = tmp_file_session(self)
        self._ready = False
        self.sr = torch.jit.Attribute(sr, int | None) 

    def _check_input_type_for_export(self, x):
        if x.type == Optional[torch.Tensor]:
            return True
        else:
            return (x.type is None) or issubclass(x.type, torch.Tensor)

    @torch.jit.export
    def _adjust_sr(self, x, factor: Optional[float] = 1.):
        return torch.nn.functional.interpolate(x, scale_factor=factor)

    def method_template(self, n_args=1):
        if n_args == 1:
            return method_template
        else:
            return method_template_n_args

    def _make_method(self, method_name: str, callback_name: Optional[str] = None):
        callback_name = callback_name or method_name
        signature = inspect.signature(getattr(self.graph_module, callback_name))
        new_params = dict(signature.parameters)
        for k, v in dict(new_params).items():
            if hasattr(v.annotation, "__module__"):
                if v.annotation.__module__ == "typing":
                    v._annotation = str(v.annotation)
                    new_params[k] = v
        signature._parameters = new_params
        signature_str = "(self, x)"

        # parse channel split
        channel_split = [s[-2] for s in self._get_input_shapes_from_method(callback_name)]
        input_args = self.graph_module.graph[method_name].find_nodes(op="placeholder")
        input_args = input_args + list(filter(lambda x: hasattr(x, "from_callback"), input_args))
        ins = ", ".join([f"in{i}" for i in range(len(channel_split))])
        ins_parsed = ", ".join([f"{input_args[i].name}=in{i}" for i in range(len(channel_split))])
        sections = "(" + ", ".join([str(sp) for sp in channel_split]) + ",)"

        # parse upsampling 
        n_samples = [s[-1] for s in self._get_input_shapes_from_method(callback_name)]
        sr_conversion = []
        for i, n in enumerate(n_samples):
            if i == 0: continue
            if n > n_samples[0] or n < n_samples[0]:
                factor = n / n_samples[0]
                sr_conversion.append(f"    in{i} = self._adjust_sr(in{i}, {factor})")
        sr_conversion = "\n".join(sr_conversion)

        n_inputs = len(channel_split)
        code = _resolve_code(self.method_template(n_inputs),
                             method_name=method_name, 
                             callback_name=callback_name, 
                             signature=signature_str, 
                             ins=ins if n_inputs > 1 else "x", 
                             ins_parsed = ins_parsed,
                             sr_conversion=sr_conversion,
                             sections=sections,
                             _import_modules=['typing']) 
        return code

    
    def _get_input_shapes_from_method(self, method):
        input_placeholders = self._get_placeholders_for_method(method)
        input_placeholders = list(filter(lambda x: self._check_input_type_for_export(x), input_placeholders))
        method_graph = self._get_graph_for_method(method)
        input_shapes = []
        for i in input_placeholders:
            activation = method_graph.activations.get(i.name)
            if activation is None: 
                if hasattr(i, "from_callback"):
                    for k, p in i.from_callback.input_controllables().items(): 
                        if re.match(rf'{i.name}((_\d)*)?$', p.name):
                            if p.value is not None: 
                                input_shapes.append(p.value.shape)
                            else:
                                raise ValueError('Could not obtain shape for input %s'%i.name)                                
            else:
                input_shapes.append(activation.shape)
        return input_shapes

    def _default_method_attributes(self, method):
        method_graph = self._get_graph_for_method(method)
        if not getattr(method_graph, "activations", None): raise ScriptedBendedException("Cannot extract activations from graph for method %s"%method)

        # get inputs
        input_placeholders = list(filter(lambda x: x.op == "placeholder", method_graph.nodes))
        input_placeholders = list(filter(lambda x: self._check_input_type_for_export(x), input_placeholders))
        input_shapes = self._get_input_shapes_from_method(method)
        # if len(set(input_shapes)) != 1:
        #     raise ScriptedBendedException("Found different input shapes for method %s. Multi-input is available if sharing same shapes"%method)
        input_shape = input_shapes[0]
        assert len(input_shape) == 3
        input_shape = input_shape[-1]
             
        # get outputs
        output_placeholder = list(filter(lambda x: x.op == "output", method_graph.nodes))[0]
        output_nodes = output_placeholder.args
        output_shapes = []
        for o in output_nodes:
            current_act = method_graph.activations.get(o.name)
            if current_act is None:
                raise ScriptedBendedException("Could not find output activation %s for method %s."%(o.name, method))
            if current_act.shape is None: 
                raise ScriptedBendedException("shape for activation %s not found, or None."%(current_act.shape))
            output_shapes.append(current_act.shape)
        if len(set(output_shapes)) != 1:
            raise ScriptedBendedException("Found different output shapes for method %s. Multi-input is available if sharing same shapes"%method)
        output_shape = output_shapes[0]
        assert len(output_shape) == 3
        output_shape = output_shape[-1]

        # retrieve channels and labels
        in_channels = 0
        out_channels = 0
        in_labels = []
        out_labels = []
        for i, p in enumerate(input_placeholders):
            in_channels += input_shapes[i][-2]
            in_labels += ["input %s, channel %d"%(p.name, j) for j in range(input_shapes[i][-2])]
        for i, p in enumerate(output_nodes):
            out_channels += output_shapes[i][-2]
            out_labels += ["output %d, channel %d"%(i, j) for j in range(output_shapes[i][-2])]

        if input_shape > output_shape:
            ratio = input_shape / output_shape
            if ratio % 2 != 0: print("[Warning] got ratio %f, may cause discrepencies"%ratio)
            ratio = round(ratio)
            in_ratio, out_ratio = 1, ratio 
        elif input_shape < output_shape:
            ratio = output_shape / input_shape
            if ratio % 2 != 0: print("[Warning] got ratio %f, may cause discrepencies"%ratio)
            ratio = round(ratio)
            in_ratio, out_ratio = ratio, 1
        else:
            in_ratio = out_ratio = 1
        
        # self.register_method(
        #    method, 
        return NNBendedMethodAttributes(
            in_channels=in_channels,
            in_ratio=in_ratio,
            out_channels=out_channels,
            out_ratio=out_ratio,
            input_labels=in_labels,
            output_labels=out_labels, 
            test_method=False
        )

    def _update_method_attributes(self, method, attributes):
        input_shapes = self._get_input_shapes_from_method(method) 
        input_nodes = self.graph_module.graph[method].find_nodes(op="placeholder")
        input_nodes = list(filter(lambda x: len(x.users) > 0, input_nodes))
        pre_annotated_channels = attributes.in_channels
        channel_count = 0
        for i, shape in enumerate(input_shapes): 
            channel_count += shape[1]
            if channel_count > pre_annotated_channels: 
                labels = [f"(signal) {input_nodes[i].name} #{j}" for j in range(shape[1])]
                attributes.input_labels.extend(labels)
        attributes.in_channels = channel_count
        if attributes.in_channels != len(attributes.input_labels):
            pass
        return attributes

    def _register_methods(self, model, force_default: bool = False):
        method_attributes = getattr(model, "nn_tilde_methods", None)
        if method_attributes is None: 
            method_attributes = {}
        else: 
            method_attributes = method_attributes()
        for method in self._available_methods:
            if method in method_attributes and not force_default:
                # self._register_method(method, self._update_method_attributes(method_attributes[method]))
                attrs = self._update_method_attributes(method, method_attributes[method])
            else:
                attrs = self._default_method_attributes(method)
            self.register_method(method, **attrs.as_dict())

    def _search_for_getter_and_setters(self, module):
        _candidates = {}
        for attr_name in dir(module):
            if not (attr_name.startswith("set_") or attr_name.startswith("get_")): continue
            if not isinstance(getattr(module, attr_name), MethodType): continue
            _candidates[attr_name] = getattr(module, attr_name)
        self._get_set_candidates = _candidates

    def _reset_get_set_candidates(self):
        self._get_set_candidates = {}
        
    def _register_controllable(self, controllable, controllables_hash):
        super()._register_controllable(controllable, controllables_hash)
        self.register_attribute(controllable.name, controllable.get_python_value())

    def _retrieve_act_sr_from_method(self, method):

        def fill_with_children(n, obj):
            assert isinstance(obj, list)
            assert isinstance(n, torch.fx.Node)
            if len(n.users) == 0: 
                return
            else:
                obj.extend(list(n.users))
                for k in n.users:
                    fill_with_children(k, obj)

        graph = getattr(self, f"_{method}").graph
        activations = graph.activations
        if not activations: 
            raise ScriptedBendedException("Could not extract activations from graph for method %s."%method)
        # graph_nodes = {k.name: k for k in list(graph.nodes)}
        input_placeholder = list(filter(lambda x: x.op == "placeholder", graph.nodes))[0]

        input_children = []
        fill_with_children(input_placeholder, input_children)
        input_shape = activations[input_placeholder.name].shape[-1]
        downsamplings = {}

        for node in input_children:
            current_shape = activations[node.name].shape[-1]
            downsamplings[node] = current_shape

        return downsamplings


    def _parse_exportable_activations(self):
        """retrieve exportable activations and corresponding sampling ratios for graph splitting"""
        for method in self._methods.value:
            self._retrieve_act_sr_from_method(method)
        # pass
            

    def register_attribute(self, attribute_name: str, values: Any | Tuple[Any]):
        getter_name = "get_"+attribute_name
        setter_name = "set_"+attribute_name
        if not hasattr(self, getter_name):
            if getter_name not in self._get_set_candidates:
                raise NNBendedModuleException(f"getter for attribute {attribute_name} not found.")
            setattr(self, getter_name, self._get_set_candidates[getter_name])
        if not hasattr(self, setter_name):
            if setter_name not in self._get_set_candidates:
                raise NNBendedModuleException(f"setter for attribute {attribute_name} not found.")
            setattr(self, setter_name, self._get_set_candidates[setter_name])
        nn_tilde.Module.register_attribute(self, attribute_name, values)

