from tabulate import tabulate
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
from .utils import BendingError, get_model_copy, bend_graph, _get_weight_properties
from ..bending import BendingCallback, CallbackChain


def _get_activations_properties(args):
    name, act_prop = args
    return [name, act_prop.op, act_prop.shape]


class BendedModule(object):
    _default_version_key = "_default"

    # -- Graph property --
    def _setgraph_(self, graph) -> NoReturn:
        if isinstance(graph, Graph) or graph is None:
            self._init_graph_(graph)
        else:
            raise TraceError('cannot set graph to value of type %s'%type(graph))
    def _getgraph_(self) -> Union[torch.fx.Graph, None]:
        # if self._graph is None:
        #     raise TraceError('BendedModule has not been traced ; please use trace function to build internal graph')
        return self._graph
    def _delgraph_(self) -> NoReturn:
        raise RuntimeError('cannot delete graph of BendedGraph')
    graph = property(_getgraph_, _setgraph_, _delgraph_)

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
        self._graph = None
        self._inputs = None
        self._activations = []
        self._bending_callbacks = []
        self._bended_params = {}
        self._bended_activations = {}
        self.module: nn.Module = module
        print(self._inputs)

    def __getattr__(self, attr_name):
        # fr = inspect.currentframe()
        if attr_name in dir(self):
            super(BendedModule, self).__getattribute__(attr_name)
        else:
            # import attribute from current module
            return getattr(self.module, attr_name)

    def _init_graph_(self, graph):
        self._graph = graph

    def _init_module_(self, module):
        self._module = module
        self._original_module = None
        self._version = self._default_version_key
        self._param_dict = {'_default': {}}
        for k, v in self._module.state_dict().items():
            if isinstance(v, nn.Parameter):
                self._param_dict[self._default_version_key][k] = v.data
            else:
                self._param_dict[self._default_version_key][k] = v
    
    def trace(self, func, *args, **kwargs):
        """Updates inner graph with the target method and inputs"""
        self._inputs = Inputs(*args, **kwargs)
        tracer = BendingTracer(func=func)
        self.graph = tracer.trace(self.module, self._inputs)
        self._activations = tracer._activations
    
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

    def activation_shape(self, param):
        return self.activations[param].shape

    # activations
    @property
    def activations(self):
        return self._activations

    @property
    def activation_names(self):
        return list(self._activations.keys())

    def print_activations(self):
        print(tabulate(map(_get_activations_properties, self.activations.items())))

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

    def _resolve_activations(self, *activations):
        """get valid activation names from a regexp"""
        valid_acts = []
        for act in self.activation_names:
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
        state_dict = copy.copy(self.state_dict())
        for k, v in self._param_dict[self._default_version_key].items():
            if k not in state_dict:
                state_dict[k] = v
            if k in self._bended_params:
                for bc in self._bended_params[k]:
                    state_dict[k] = bc(v, name=k.replace(".", "_"))
        return state_dict

    def _clone_bended_parameters(self, module, params):
        for p in params:
            module.get_parameter(p).data = module.get_parameter(p).data.clone()

    def bend_module(self):
        state_dict = self.bended_state_dict()
        self._clone_bended_parameters(self.module, self._bended_params)
        # clone module with deep-copying parameters
        module = get_model_copy(self.module, copy_parameters=True)
        # copy target weights, as load_state_dict method replaces in place
        # loaded bended dict
        module.load_state_dict(state_dict)
        # add activation callbacks
        if self.graph is not None:
            for k, v in self._bended_activations.items():
                setattr(module, k+'_callback', CallbackChain(v))
        return module

    def bend_graph(self):
        return bend_graph(self.graph, {k: CallbackChain(v) for k, v in self._bended_activations.items()})

    def _bend_parameter(self, parameter, callback):
        assert parameter in self.weights, "parameter %s not found in module"%parameter
        if callback not in self._bending_callbacks:
            self._bending_callbacks.append(callback)
        self._bended_params[parameter] = self._bended_params.get(parameter, []) + [callback]
        callback.add_bending_target(parameter, self.param_shape(parameter))

    def _bend_activation(self, parameter, callback):
        if callback not in self._bending_callbacks:
            self._bending_callbacks.append(callback)
        self._bended_activations[parameter] = self._bended_activations.get(parameter, []) + [callback]
        callback.add_bending_target(parameter, self.activation_shape(parameter))

    def bend(self, callback, *params, verbose=False):
        assert isinstance(callback, BendingCallback), "callback must be a BendingCallback instance"
        target_params = self._resolve_parameters(*params)
        target_activations = []
        if self.graph is not None:
            target_activations = self._resolve_activations(*params)
        if len(target_params) + len(target_activations) == 0:
            raise BendingError('Could not find bendable elements with specification %s'%params)
        # register bending
        for param in target_params:
            if verbose: 
                print('bending parameter %s with %s...'%(param, callback))
            self._bend_parameter(param, callback)
        for activation in target_activations:
            self._bend_activation(activation, callback)
            if verbose: 
                print('bending activation %s with %s...'%(param, callback))

    def bend_(self, *args, **kwargs):
        self.bend(*args, **kwargs)
        self._module = self.bend_module()

    def _reset_bending(self):
        self._bending_callbacks = []
        self._bended_params = {}
        self._bended_activations = {}

    def reset(self, version=None):
        self._reset_bending()
        if version is None:
            self._module.load_state_dict(self.state_dict(True)[self.version])
        else:
            self.version = version
            self._module.load_state_dict(self.state_dict(True)[version])
        return

    # callbacks
    def __call__(self, *args, **kwargs):
        """call the module with current input."""
        module = self.bend_module()
        if self._graph is None:
            return module(*args, **kwargs)
        else:
            # bend activations
            graph = self.bend_graph()
            graph_module = torch.fx.GraphModule(module, graph)
            return graph_module(*args, **kwargs)

    def _get_bended_activations(self, activations):
        bended_activations = []
        for act in activations:
            if act in self._bended_activations:
                bended_activations.append(act+"_bended")
            else:
                bended_activations.append(act)
        return act

    def get_activations(self, *activations, bended=True, **inputs):
        """return target activations from given inputs."""
        # modify graph
        module = self.bend_module()
        graph = self.bend_graph()

        activations = self._resolve_activations(*activations)
        if bended:
            activations = self._get_bended_activations(activations)

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
        return outs
