from tabulate import tabulate
import types
import pathlib, os
import re
import copy
import torch
import inspect
from torch import nn
from torch.fx import Graph
from torch.fx.proxy import TraceError
from typing import Union , NoReturn
from .input import Inputs
from .tracing import BendingTracer
from .utils import BendingError, get_model_copy, bend_graph, _get_weight_properties, _import_to_interface
from ..utils import checklist
from ..bending import BendingCallback, CallbackChain


def _get_activations_properties(args):
    name, act_prop = args
    return [name, act_prop.op, act_prop.shape]


def _get_wrapped_module_forward_call(fn):
    def _wrapped_module_forward_call(self, *args, **kwargs):
        module = self.bend_module()
        if self._graphs.get(fn) is None:
            return getattr(module, fn)(*args, **kwargs)
        else:
            # bend activations
            graph = self.bend_graph(fn=fn)
            graph_module = torch.fx.GraphModule(module, graph)
            return graph_module(*args, **kwargs)
    return _wrapped_module_forward_call


class BendedModule(object):
    _default_version_key = "_default"

    # -- Module property --
    def _setmodule_(self, module) -> NoReturn:
        if isinstance(module, nn.Module) or nn.Module is None:
            self._init_module_(module)
        else:
            raise TraceError('cannot set graph to value of type %s'%type(module))
    def _getmodule_(self) -> Union[torch.fx.Graph, None]:
        return self._module
    def _delmodule_(self) -> NoReturn:
        del self._module
    module = property(_getmodule_, _setmodule_, _delmodule_)

    # -- Version property --
    def _setversion_(self, version) -> NoReturn:
        if version is None: version = self._default_version_key
        if version not in self.state_dict(True): raise BendingError('BendedModule has no version %s'%version)
        self._version = version
    def _getversion_(self) -> Union[torch.fx.Graph, None]:
        return self._version
    def _delversion_(self) -> NoReturn:
        self._version = self._default_version_key
    version = property(_getversion_, _setversion_, _delversion_)

    
    def __init__(self, module):
        self._graphs = {}
        self._activations = {}
        self._bending_callbacks = []
        self._bended_params = {}
        self._bended_activations = {}
        self.module: nn.Module = module

    def __getattr__(self, attr_name):
        if attr_name in dir(self):
            super(BendedModule, self).__getattribute__(attr_name)
        else:
            # import attribute from current module
            return getattr(self.module, attr_name)

    def _init_module_(self, module):
        #TODO Noooo
        self._module = get_model_copy(module, copy_parameters=True)
        self._module = module
        self._original_module = None
        self._version = self._default_version_key
        self._param_dict = {'_default': {}}
        for k, v in self._module.state_dict().items():
            if isinstance(v, nn.Parameter):
                self._param_dict[self._default_version_key][k] = v.data
            else:
                self._param_dict[self._default_version_key][k] = v

    def _register_forward_call(self, func):
        setattr(self, func, types.MethodType(_get_wrapped_module_forward_call(func), self))
    
    def trace(self, func="forward", *args, _return_out=False, **kwargs):
        """Updates inner graph with the target method and inputs"""
        inputs = Inputs(*args, **kwargs)
        tracer = BendingTracer(func=func)
        tracer_out = tracer.trace(self.module, inputs, return_out=_return_out)
        graph = tracer_out[0] if _return_out else tracer_out
        self._graphs[func] = graph
        self._activations[func] = tracer._activations
        self._bended_activations[func] = dict()
        if func != "forward":
            self._register_forward_call(func)
        if _return_out:
            return graph, tracer_out[1]
        else:
            return graph
    
    # parameters
    def parameters(self):
        """return model parameters"""
        return self.module.parameters()
    def named_parameters(self):
        """return model's named parameters"""
        return self.module.named_parameters()

    @property
    def weights(self):
        """returns weights names"""
        if self.module is None:
            raise TraceError('BendedGraph has no weights since module as not been initialized')
        return list(dict(self.module.named_parameters()).keys())

    @_import_to_interface
    def print_weights(self, flt=r".*", out=None):
        """print / export weights"""
        parameters = dict(filter(lambda v: re.match(flt, v[0]), dict(self.named_parameters()).items()))
        pretty_weights = tabulate(map(_get_weight_properties, parameters.items()))
        if out is None:
            print(pretty_weights)
        else:
            out = pathlib.Path(out)
            os.makedirs(out.parent, exist_ok=True)
            with open(out, 'w+') as f:
                f.write(pretty_weights)

    def param_shape(self, param):
        return self.module.state_dict()[param].shape

    def activation_shape(self, param, fn="forward"):
        return self.activations(fn)[param].shape

    # activations
    def activations(self, fn):
        return self._activations[fn]

    def activation_names(self, fn="forward"):
        return list(self._activations[fn].keys())

    def is_traced(self, fn):
        return fn in self._graphs

    @_import_to_interface
    def print_activations(self, fn="forward", op=None, flt=r".*", out=None):
        activations = self.activations(fn)
        if op is not None:
            op = checklist(op)
            activations = dict(filter(lambda obj: obj[1].op in op, activations.items()))
        activations = dict(filter(lambda obj: re.match(flt, obj[0]), activations.items()))
        act_txt = tabulate(map(_get_activations_properties, activations.items()))
        if out is None:
            print(act_txt)
        else:
            out = pathlib.Path(out)
            os.makedirs(out.parent, exist_ok=True)
            with open(out, 'w+') as f:
                f.write(act_txt)


    def _resolve_parameters(self, *weights):
        """get valid weight names from a regexp"""
        valid_weights = []
        for weight in self.weights:
            current_weight = []
            for w in weights:
                if re.match(w, weight) is not None:
                    current_weight.append(weight)
            if len(current_weight) > 0:
                valid_weights.extend(current_weight)
        return valid_weights

    def _resolve_activations(self, *activations, fn="forward"):
        """get valid activation names from a regexp"""
        valid_acts = []
        for act in self.activation_names(fn):
            current_act = []
            for a in activations:
                if re.match(a, act) is not None:
                    current_act.append(act)
            if len(current_act) > 0:
                valid_acts.extend(current_act)
        return valid_acts

    def state_dict(self, with_versions=False):
        if with_versions:
            return dict(self._param_dict)
        else:
            return self._param_dict[self._version]


    def save(self, name):
        """save bended dictionary with current wieght callbacks"""
        #TODO
        pass


    # Bending callbacks
    def bended_state_dict(self):
        state_dict = copy.copy(self.module.state_dict())
        for k, v in self._param_dict[self._default_version_key].items():
            if k not in state_dict:
                state_dict[k] = v
            if k in self._bended_params:
                for bc in self._bended_params[k]:
                    state_dict[k] = bc(v, name=k.replace(".", "_"))
        return state_dict

    def _clone_bended_parameters(self, module, params):
        """makes copy of internal parameter values of a module"""
        for p in params:
            module.get_parameter(p).data = module.get_parameter(p).data.clone()

    @property
    def bending_callbacks(self):
        return list(self._bending_callbacks)

    @property
    def bended_params(self):
        return {k: list(v) for k, v in self._bended_params.items()}

    def bended_activations(self, fn):
        return {k: list(v) for k, v in self._bended_activations[fn].items()}

    def bend_module(self, fn="forward"):
        with torch.no_grad():
            state_dict = self.bended_state_dict()
            self._clone_bended_parameters(self.module, self._bended_params)
            # clone module with deep-copying parameters
            module = get_model_copy(self.module, copy_parameters=True)
            # copy target weights, as load_state_dict method replaces in place
            # loaded bended dict
            module.load_state_dict(state_dict)
            # add activation callbacks
            if self._graphs.get(fn) is not None:
                for k, v in self.bended_activations(fn).items():
                    setattr(module, k+'_callback', CallbackChain(v))
            return module

    def bend_graph(self, fn="forward"):
        return bend_graph(self._graphs[fn], {k: CallbackChain(v) for k, v in self._bended_activations[fn].items()})

    def _bend_parameter(self, parameter, callback):
        assert parameter in self.weights, "parameter %s not found in module"%parameter
        if callback not in self._bending_callbacks:
            self._bending_callbacks.append(callback)
        self._bended_params[parameter] = self._bended_params.get(parameter, []) + [callback]
        callback.add_bending_target(parameter, self.param_shape(parameter))

    def _bend_activation(self, parameter, callback, fn="forward"):
        if callback not in self._bending_callbacks:
            self._bending_callbacks.append(callback)
        if fn not in self._bended_activations: self._bended_activations[fn] = {}
        self._bended_activations[fn][parameter] = self._bended_activations[fn].get(parameter, []) + [callback]
        try: 
            callback.add_bending_target(parameter, self.activation_shape(parameter, fn=fn))
        except Exception as e:
            print('Cannot bend activation %s with callback %s.\nException : %s\n Proceeding'%(parameter, callback, e))

    @_import_to_interface
    def bend(self, callback, *params, fn="forward", verbose=False):
        assert isinstance(callback, BendingCallback), "callback must be a BendingCallback instance"
        target_params = self._resolve_parameters(*params)
        target_activations = []
        if self._graphs.get(fn) is not None:
            target_activations = self._resolve_activations(*params, fn=fn)
        if len(target_params) + len(target_activations) == 0:
            raise BendingError('Could not find bendable elements with specification %s'%params)
        # register bending
        for param in target_params:
            if verbose: 
                print('bending parameter %s with %s...'%(param, callback))
            self._bend_parameter(param, callback)
        for activation in target_activations:
            self._bend_activation(activation, callback, fn=fn)
            if verbose: 
                print('bending activation %s with %s...'%(activation, callback))

    @_import_to_interface
    def bend_(self, *args, fn="forward", **kwargs):
        self.bend(*args, **kwargs)
        self._module = self.bend_module()
        if self._graphs.get(fn):
            self._graphs[fn+"_orig"] = self._graphs[fn]
            self._graphs[fn] = self.bend_graph(fn)
        self._reset_bending()

    def _reset_bending(self):
        self._bending_callbacks = []
        self._bended_params = {}
        self._bended_activations = {k: {} for k in self._bended_activations.keys()}

    @_import_to_interface
    def reset(self, version=None):
        self._reset_bending()
        for fn in list(self._graphs.keys()):
            if  fn+"_orig" not in self._graphs:
                print('[Warning] could not recover graph for method %s'%fn)
                continue
            self._graphs[fn] = self._graphs[fn+"_orig"]
            del self._graphs[fn+"_orig"]
        #TODO : callbacks are in modules for activations, handle it in proper way
        if version is None:
            self._module.load_state_dict(self.state_dict(True)[self.version], strict=False)
        else:
            self.version = version
            self._module.load_state_dict(self.state_dict(True)[version], strict=False)

    # callbacks
    def forward(self, *args, **kwargs):
        """call the module with current input."""
        module = self.bend_module()
        if self._graphs.get('forward') is None:
            return module(*args, **kwargs)
        else:
            # bend activations
            graph = self.bend_graph()
            graph_module = torch.fx.GraphModule(module, graph)
            return graph_module(*args, **kwargs)

    def _get_bended_activations(self, activations, fn="forward"):
        bended_activations = []
        for act in activations:
            if act in self._bended_activations[fn]:
                bended_activations.append(act+"_bended")
            else:
                bended_activations.append(act)
        return bended_activations

    def get_activations(self, *activations, fn="forward", bended=True, **inputs):
        """return target activations from given inputs."""
        # modify graph
        module = self.bend_module()
        graph = self.bend_graph(fn=fn)

        activations = self._resolve_activations(*activations)
        if bended:
            activations = self._get_bended_activations(activations, fn=fn)

        out_nodes = {}
        for node in list(graph.nodes):
            if node.op == "output":
                graph.erase_node(node)
            else:
                if node.name in activations:
                    out_nodes[node.name] = node

        out_node = graph.create_node("call_function", dict, kwargs=out_nodes, name="out_dict")
        graph.output(out_node)

        # forward
        gm = torch.fx.GraphModule(module, graph)
        outs = gm(**inputs)
        outs = {k.replace('_bended', ''): v for k, v in outs.items()}
        return outs
