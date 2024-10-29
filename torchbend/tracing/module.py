from tabulate import tabulate
from io import TextIOWrapper
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
from . import interp
from .input import Inputs
from .tracing import BendingTracer
from .utils import BendingError, get_model_copy, bend_graph, _get_weight_properties
from .utils import _import_to_interface, make_graph_jit_compatible, clone_parameters, _bending_config_from_dicts
from ..utils import checklist, get_parameter, print_tensor_ids
from ..bending import BendingCallback, CallbackChain, BendingConfig


def _get_activations_properties(args):
    name, act_prop = args
    return [name, act_prop.op, act_prop.shape]


def _get_wrapped_module_forward_call(fn, bend=True):
    def _wrapped_bended_module_forward_call(self, *args, **kwargs):
        module = self.bend_module(fn=fn)
        if self._graphs.get(fn) is None:
            return getattr(module, fn)(*args, **kwargs)
        else:
            # bend activations
            graph = self.bend_graph(fn=fn)
            graph_module = torch.fx.GraphModule(module, graph)
            return graph_module(*args, **kwargs)
    def _wrapped_module_forward_call(self, *args, **kwargs):
        return getattr(self._module, fn)(*args, **kwargs)
    return _wrapped_bended_module_forward_call if bend else _wrapped_module_forward_call 


class BendedModuleVersionEnv:
    def __init__(self, *args, version=None):
        self._modules = args
        self._init_versions = [mod.version for mod in args]
        self.version = version

    def __enter__(self):
        for mod in self._modules: 
            mod.version = self.version

    def __exit__(self, *args):
        for i, mod in enumerate(self._modules): 
            self._modules[i].version = self._init_versions[i]


class BendedModuleInterpolationEnv(object):
    def __init__(self, module, interp_dict=None, interp_func=interp.linear):
        self._module = module 
        self._interp_dict = interp_dict
        self._interp_func = interp_func

    def __enter__(self):
        self._module.set_interpolation_weights(self._interp_dict, self._interp_func)

    def __exit__(self, *args):
        self._module.remove_interpolation_weights()


class BendedModule(object):
    _default_version_key = "_default"
    _wrapped_methods = ['forward']

    # -- Module property --
    def _init_module_(self, module):
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
    def _setmodule_(self, module) -> NoReturn:
        raise BendingError('Cannot set module of BendedModule after initaliazation.')
        #TODO : allow? why? good idea?
        # if isinstance(module, nn.Module):
        #     self._init_module_(module)
        # else:
        #     raise TraceError('cannot set graph to value of type %s'%type(module))
    def _getmodule_(self) -> Union[torch.fx.Graph, None]:
        return self._module
    def _delmodule_(self) -> NoReturn:
        raise BendingError('Cannot delete module of BendedModule')
    # def get_module(self):
    #     return self._module
    module = property(_getmodule_, _setmodule_, _delmodule_)

    # -- Version property --
    def _setversion_(self, version) -> NoReturn:
        if version is None: version = self._default_version_key
        if version not in self.state_dict(with_versions=True): raise BendingError('BendedModule has no version %s'%version)
        self._version = version
    def _getversion_(self) -> Union[torch.fx.Graph, None]:
        return self._version
    def _delversion_(self) -> NoReturn:
        self._version = self._default_version_key
    version = property(_getversion_, _setversion_, _delversion_)

    def to(self, *args, _no_warning: bool = False, **kwargs):
        if not _no_warning: 
            print('[Warning]to is an experimental feature for BendingModule ;\nsetting original module to target device before wrapping is advised.')
            print('call with _no_warning=True to remove this warning.')
        # make shallow copy to change module.
        #TODO should be tested more 
        self_copy = copy.copy(self)
        if self_copy._module is not None:
            self_copy._module = self_copy._module.to(*args, **kwargs)
        return self_copy

    # -- init --
    def __init__(self, module, _wrapped_methods=[]):
        self._graphs = {}
        self._activations = {}
        # callback, parameters and activations
        self._bending_callbacks = []
        self._bended_params = {self._default_version_key: {}}
        self._bended_params_history = {self._default_version_key: []}
        self._bended_activations = {}
        # interpolation parameters
        self._interp_dict = None
        self._interp_func = None
        # controllables parameters
        self._controllables = {}
        self._controllable_hash = {}
        self._wrapped_methods.extend(_wrapped_methods)
        if issubclass(type(module), nn.Module):
            self._init_module_(module)
        else:
            raise TypeError('module must be a nn.Module subclass, got : %s'%(type(module).__name__))

    # -- getattr --
    def __getattr__(self, attr_name):
        if attr_name in dir(self):
            return super(BendedModule, self).__getattribute__(attr_name)
        else:
            # import attribute from current module
            attr = getattr(self._module, attr_name)
            if isinstance(attr, types.MethodType):
                _is_bended = (attr.__name__ in self._wrapped_methods) or (attr.__name__ in getattr(type(self._module), "__bended_methods__", []))
                self._register_forward_call(attr_name, _with_bended=_is_bended)
                return super(BendedModule, self).__getattribute__(attr_name)
            else:
                return attr

    # -- parameters & weights --
    def _resolve_parameters(self, *weights):
        """get valid weight names from a regexp"""
        valid_weights = []
        for weight in self.weight_names:
            current_weight = []
            for w in weights:
                if re.match(w, weight) is not None:
                    current_weight.append(weight)
            if len(current_weight) > 0:
                valid_weights.extend(current_weight)
        return valid_weights

    @property
    @_import_to_interface
    def weight_names(self):
        """returns weights names"""
        if self._module is None:
            raise TraceError('BendedGraph has no weights since module as not been initialized')
        return list(dict(self._module.named_parameters()).keys())

    @_import_to_interface
    def weight_shape(self, param):
        return self._module.state_dict()[param].shape

    @_import_to_interface
    def print_weights(self, flt=r".*", out=None) -> str:
        """print / export weights"""
        parameters = dict(filter(lambda v: re.match(flt, v[0]), dict(self.named_parameters()).items()))
        pretty_weights = tabulate(map(_get_weight_properties, parameters.items()), headers=['name', 'shape', 'dtype', 'min', 'max', 'mean', 'stddev'])
        if out is None:
            print(pretty_weights)
        elif isinstance(out, TextIOWrapper):
            out.write(pretty_weights)
        else:
            out = pathlib.Path(out)
            os.makedirs(out.parent, exist_ok=True)
            with open(out, 'w+') as f:
                f.write(pretty_weights)
        return pretty_weights

    @_import_to_interface
    def parameters(self):
        """return model parameters"""
        return self._module.parameters()

    @_import_to_interface
    def named_parameters(self):
        """return model's named parameters"""
        return self._module.named_parameters()
    
    @_import_to_interface
    def state_dict(self, version=None, with_versions=False):
        if with_versions:
            assert version is None, "either version or with_versions must be true."
            return dict(self._param_dict)
        else:
            version = version or self._version
            return self._param_dict[version]

    # -- activations --
    def _resolve_activations(self, *activations, fn="forward", _raise_notfound=False):
        """get valid activation names from a regexp"""
        valid_acts = {}
        for act in self.activation_names(fn):
            for a in activations:
                if re.match(a, act) is not None:
                    if not a in valid_acts: valid_acts[a] = []
                    valid_acts[a].append(act)
        if _raise_notfound:
            for a in activations:
                if not a in valid_acts: raise BendingError('request %s could not be found in graph for function %s'%(a, fn))
        valid_acts = list(set(sum(list(valid_acts.values()), [])))
        return valid_acts

    @_import_to_interface
    def activations(self, fn="forward", op=None, flt=r".*"):
        activations = self._activations[fn] 
        if op is not None:
            op = checklist(op)
            activations = dict(filter(lambda obj: obj[1].op in op, activations.items()))
        activations = dict(filter(lambda obj: re.match(flt, obj[0]), activations.items()))
        return activations

    @_import_to_interface
    def activation_names(self, fn="forward"):
        return list(self._activations[fn].keys())

    @_import_to_interface
    def activation_shape(self, param, fn="forward"):
        return self.activations(fn)[param].shape

    @_import_to_interface
    def print_activations(self, fn="forward", op=None, flt=r".*", out=None) -> str:
        activations = self.activations(fn, op=op, flt=flt)
        act_txt = tabulate(map(_get_activations_properties, activations.items()))
        if out is None:
            print(act_txt)
        elif isinstance(out, TextIOWrapper):
            out.write(act_txt)
        else:
            out = pathlib.Path(out)
            os.makedirs(out.parent, exist_ok=True)
            with open(out, 'w+') as f:
                f.write(act_txt)
        return act_txt

    # -- tracing -- 
    def is_traced(self, fn):
        return fn in self._graphs

    def _register_forward_call(self, func, _with_bended=False):
        setattr(self, func, types.MethodType(_get_wrapped_module_forward_call(func, _with_bended), self))

    def trace(self, func="forward", *args, _return_out=False, _proxied_buffers=[], _no_tensor_for_args=None, **kwargs):
        """Updates inner graph with the target method and inputs"""
        #TODO general split between kwargs with _ at the beginning for tracer
        inputs = Inputs(*args, **kwargs)
        tracer = BendingTracer(func=func, _no_tensor_for_args=_no_tensor_for_args)
        tracer_out = tracer.trace(self._module, inputs, return_out=_return_out, proxied_buffers=_proxied_buffers)
        graph = tracer_out[0] if _return_out else tracer_out
        self._graphs[func] = graph
        self._activations[func] = tracer._activations
        self._bended_activations[func] = dict()
        if func != "forward":
            self._register_forward_call(func, True)
        if _return_out:
            return graph, tracer_out[1]
        else:
            return graph

    @_import_to_interface
    def graph(self, fn="forward"):
        if not fn in self._graphs: raise BendingError('function %s is not graphed'%fn)
        return self._graphs[fn]
    

    # -- callbacks --
    @_import_to_interface
    def __call__(self, *args, **kwargs):
        """call the module with current input."""
        module = self.bend_module()
        if self._graphs.get('forward') is None:
            return module(*args, **kwargs)
        else:
            # bend activations
            graph = self.bend_graph()
            graph_module = torch.fx.GraphModule(module, graph)
            return graph_module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)



    #  -- Bending callbacks --
    def _bended_state_dict_from_version(self, version=None):
        version = version or self.version
        state_dict = copy.copy(self.state_dict(version=version))
        for k, v in self._param_dict[self._default_version_key].items():
            if k not in state_dict:
                state_dict[k] = v
            clone_parameters(state_dict, [k])
            if k in self._bended_params[version]:
                for bc in self._bended_params[version][k]:
                    state_dict[k] = bc(v, name=k.replace(".", "_"))
        return state_dict

    def _bended_state_dict_from_interp(self):
        dicts = {}
        for version, weight in self._interp_dict.items():
            bended_dict = self._bended_state_dict_from_version(version)
            dicts[version] = (bended_dict, weight)
        return self._interp_func(self, dicts)
    
    @_import_to_interface
    def bendable_keys(self, *request, fn="forward"):
        keys = self._resolve_parameters(*request)
        if self.is_traced(fn):
            keys = keys + self._resolve_activations(*request, fn=fn)
        return keys

    @_import_to_interface
    def bended_state_dict(self, version=None):
        if self._interp_dict is None:
            return self._bended_state_dict_from_version(version)
        else:
            # assert version is not None, "cannot specify version when interpolation weights are defined. Call remove_interpolation_weights to remove"
            return self._bended_state_dict_from_interp()

    @property
    @_import_to_interface
    def bending_callbacks(self):
        return list(self._bending_callbacks)

    @property
    @_import_to_interface
    def bended_params(self):
        return {k: list(v) for k, v in self._bended_params[self.version].items()}

    @_import_to_interface
    def bended_activations(self, fn):
        return {k: list(v) for k, v in self._bended_activations[fn].items()}

    @_import_to_interface
    def bended_keys(self, fn=None, version=None):
        version = version or self.version
        bended_params = list({k: list(v) for k, v in self._bended_params[version].items()}.keys())
        if (fn is None) and ('forward' in self._graphs): fn = "forward"
        if (fn is not None) and fn in self._bended_activations:
            return bended_params + list(self.bended_activations(fn).keys())
        else:
            return bended_params

    @_import_to_interface
    def bending_config(self, fn="forward", version=None):
        version = version or self.version
        #TODO reconstruct tree or ?
        bending_config = _bending_config_from_dicts(self._bended_params[version], self._bended_activations.get(fn, {}), module=self)
        print(bending_config)
        return bending_config

    def _bend_parameter(self, parameter, callback, version=None):
        version = version or self.version
        assert parameter in self.weight_names, "parameter %s not found in module"%parameter
        if callback not in self._bending_callbacks:
            self._bending_callbacks.append(callback)
        self._bended_params[version][parameter] = self._bended_params[version].get(parameter, []) + [callback]
        #TODO register parameter in callback
        callback.add_bending_target(parameter, parameter=self._module.get_parameter(parameter))

    def _bend_activation(self, parameter, callback, fn="forward"):
        if callback not in self._bending_callbacks:
            self._bending_callbacks.append(callback)
        if fn not in self._bended_activations: self._bended_activations[fn] = {}
        self._bended_activations[fn][parameter] = self._bended_activations[fn].get(parameter, []) + [callback]
        try: 
            #TODO register activation shape
            callback.add_bending_target(parameter, shape=self.activation_shape(parameter, fn=fn))
        except Exception as e:
            print('Cannot bend activation %s with callback %s.\nException : %s\n Proceeding'%(parameter, callback, e))
    
    @_import_to_interface
    def bend_module(self, fn="forward", version=None):
        version = version or self.version
        with torch.no_grad():
            # clone module with deep-copying parameters
            module = get_model_copy(self._module, copy_parameters=True)
            # module = copy.deepcopy(self.module)
            state_dict = self.bended_state_dict()
            # copy target weights, as load_state_dict method replaces in place
            clone_parameters(module, list(self._bended_params[version].keys()) + self._bended_params_history[self.version])
            # loaded bended dict
            module.load_state_dict(state_dict, assign=True)
            # add activation callbacks
            if self._graphs.get(fn) is not None:
                for k, v in self.bended_activations(fn).items():
                    setattr(module, k+'_callback', CallbackChain(v))
            return module

    @_import_to_interface
    def bend_graph(self, fn="forward"):
        return bend_graph(self._graphs[fn], {k: CallbackChain(v) for k, v in self._bended_activations[fn].items()})

    @_import_to_interface
    def graph_module(self, fn="forward", module=None, make_jit_compatible: bool = False):
        if module is None:
            module = self.bend_module(fn=fn)
        graph = self.bend_graph(fn=fn)
        if make_jit_compatible:
            graph = make_graph_jit_compatible(graph)
        return torch.fx.GraphModule(module, graph)

    @_import_to_interface
    def bend(self, *args, fn=None, verbose=False, bend_param=True, bend_graph=True):
        if len(args) == 1:
            bended_config = args[0]
            assert isinstance(bended_config, BendingConfig)
            for c, *w in args[0]:
                self.bend(c, *w)
            return 
        callback, *params = args
        assert isinstance(callback, BendingCallback), "callback must be a BendingCallback instance"
        if fn is None:
            fn = list(self._graphs.keys())
        else:
            fn = checklist(fn)

        # bend weights
        
        target_params = [] if not bend_param else self._resolve_parameters(*params)
        target_activations = {method: [] if not bend_graph else self._resolve_activations(*params, fn=method) for method in fn}
        if len(target_params) + sum(map(len, target_activations.values())) == 0:
            raise BendingError('Could not find bendable elements with specification %s'%params)
        for param in target_params:
            if verbose: 
                print('bending parameter %s with %s...'%(param, callback))
            self._bend_parameter(param, callback)

        # bend activations
        for method in fn:
            for activation in target_activations[method]:
                self._bend_activation(activation, callback, fn=method)
                if verbose: 
                    print('bending activation %s with %s...'%(activation, callback))
        # extract controllables in case
        self._register_controllables(callback)

    # @_import_to_interface
    # def bend_(self, *args, fn="forward", **kwargs):
    #     self.bend(*args, **kwargs)
    #     self._module = self.bend_module()
    #     if self._graphs.get(fn):
    #         self._graphs[fn+"_orig"] = self._graphs[fn]
    #         self._graphs[fn] = self.bend_graph(fn)
    #     self._reset_bending()

    def _reset_bending(self, version=None):
        version = version or self.version
        self._bending_callbacks = []
        self._bended_params[version] = {}
        self._bended_activations = {k: {} for k in self._bended_activations.keys()}

    @_import_to_interface
    def reset(self, version=None):
        self._reset_bending(version=version)
        for fn in list(self._graphs.keys()):
            if  fn+"_orig" in self._graphs:
                self._graphs[fn] = self._graphs[fn+"_orig"]
                del self._graphs[fn+"_orig"]
        #TODO : callbacks are in modules for activations, handle it in proper way
        if version is None:
            self._module.load_state_dict(self.state_dict(self.version), strict=False)
        else:
            self.version = version
            self._module.load_state_dict(self.state_dict(version), strict=False)

    # -- controllables --
    def _register_controllables(self, callback):
        #TODO be sure that controllables does not have the same name at creation
        for k, v in callback._controllables.items():
            if k not in self._controllables:
                self._controllables[v.name] = v
                self._controllable_hash[v.name] = self._controllable_hash.get(v.name, []) + [self._bending_callbacks.index(callback)]

    @property
    @_import_to_interface
    def controllables(self):
        return copy.copy(self._controllables)

    def update(self, param_name, value):
        """updates value of a given BendingParameter object"""
        if param_name not in self._controllables:
            raise BendingError("parameter %s not present in BendingModule"%param_name)
        self._controllables[param_name].set_value(value)
        for i in self._controllable_hash[param_name]:
            self._bending_callbacks[i].update()


    # -- activation retrival -- 
    def _get_bended_activations(self, activations, fn="forward"):
        bended_activations = []
        for act in activations:
            if act in self._bended_activations[fn]:
                bended_activations.append(act+"_bended")
            else:
                bended_activations.append(act)
        return bended_activations

    @_import_to_interface
    def get_activations(self, *activations, fn="forward", bended=True, **inputs):
        """return target activations from given inputs."""
        # modify graph
        module = self.bend_module()
        graph = self.bend_graph(fn=fn)

        activations = self._resolve_activations(*activations, _raise_notfound=True)
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


    #  -- version & interpolation handling --
    def _write_bendings(self) -> None:
        # write weight bendings
        self._param_dict[self.version] = self.bended_state_dict()
        self.reset()

    def _write_bendings_as_new(self, _version, force: bool = False, deep: bool = False, clear: bool = True) -> None:
        if (_version in self._param_dict) and (not force):
            raise BendingError(f'Version {_version} already exists. Please pass force=True as a keyword to erase previous configuration')
        self._param_dict[_version] = self.bended_state_dict()
        self._bended_params_history[_version] = list(self._bended_params[self.version].keys()) + list(self._bended_params_history[self.version])
        self._bended_params[_version] = {}
        if deep:
            self._param_dict[_version] = copy.deepcopy(self._param_dict[_version])
        if clear: 
            self.reset(self.version)

    @_import_to_interface
    def write(self, version=None, force: bool = False, deep: bool = False, clear: bool = True):
        if version is None: 
            self._write_bendings()
        else:
            self._write_bendings_as_new(version, force=force, deep=deep, clear = clear)
        self.version = version

    @_import_to_interface
    def set_version(self, version=None):
        return BendedModuleVersionEnv(self, version=version)

    @_import_to_interface
    def create_version(self, name, state_dict, strict=True):
        if isinstance(state_dict, nn.Module):
            state_dict = state_dict.state_dict()
        if strict == True and not state_dict.keys() == self._param_dict[self._default_version_key].keys():
            raise BendingError('given state dict has different keys ; you can bypass this warning by setting strict=False, at your own risk')
        unmatched_keys = list(set(state_dict.keys()).difference(self._param_dict[self._default_version_key].keys()))
        self._param_dict[name] = self.bended_state_dict()
        self._param_dict[name].update(state_dict)
        self._bended_params[name] = {}
        self._bended_params_history[name] = []
        return unmatched_keys

    # -- interpolation -- 

    @_import_to_interface
    def interpolate(self, *args, **weights):
        if len(args) > 1: raise BendingError("interpolate takes a single optional positional argument for default weight, got %d"%(len(args)))
        if len(args) == 1: weights[self._default_version_key] = float(args[0])
        return BendedModuleInterpolationEnv(self, interp_dict=weights)

    @_import_to_interface
    def set_interpolation_weights(self, interp_dict, interp_func):
        for k in interp_dict.keys():
            assert k in self._param_dict, "version %s not set"%k
            interp_dict[k] = float(interp_dict[k])
        self._interp_dict = interp_dict
        self._interp_func = interp_func

    @_import_to_interface
    def remove_interpolation_weights(self):
        self._interp_dict = None
        self._interp_func = None


def unmatching_ids(module1, module2, weights, data=False):
    def _get(module, w):
        if hasattr(module, "__getitem__"):
            return module[w]            
        else:
            return get_parameter(module, w)
            
    unmatched = []
    for w in weights:
        if data: 
            res_tmp = id(_get(module1, w).data) == id(_get(module2, w).data)
        else:
            res_tmp = id(_get(module1, w)) == id(_get(module2, w))
        if not res_tmp: unmatched.append(w)
    return unmatched
