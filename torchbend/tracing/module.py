from tabulate import tabulate
import typing
from collections import OrderedDict
from functools import partial
from itertools import product
from types import MethodType
from io import TextIOWrapper
import types
import pathlib, os
import re
import copy
import torch
import inspect
from torch import nn
from torch.fx import Graph, GraphModule
from torch.fx.proxy import TraceError
from typing import Union , NoReturn, Optional, Tuple, List
from .. import get_output, TorchbendOutput
from . import interp
from .input import Inputs
from .graphmodule import BendedGraphModule
from .tracing import BendingTracer, ActivationProperties, BendedGraph
from .utils import BendingError, get_model_copy, _get_weight_properties, _get_signature_from_graph, _get_graph_inputs
from .utils import _import_to_interface, make_graph_jit_compatible, clone_parameters, display_table_for_jupyter, get_kwargs_from_gm
from .graph import graph_insert_callbacks, graph_get_activations, graph_from_activations, graph_transform_nodes
from ..utils import checklist, checktuple, get_parameter, _resolve_code
from ..bending import BendingCallback, CallbackChain, is_bending_callback, BendingConfig, BendingParameter

_DEFAULT_ACT_EXCLUDE_LIST = ['getattr.*', 'cat.*', 'getitem.*', 'copy.*', 'reshape.*']
_DEFAULT_ACTIVATION_FIELDS = ['name', 'op', 'target', 'shape', 'args', 'kwargs']

def _get_activations_properties(act_prop, fields=None):
    fields = fields or _DEFAULT_ACTIVATION_FIELDS
    return [getattr(act_prop, n) for n in fields]

def _get_wrapped_module_forward_call(fn, bend=True):
    def _wrapped_bended_module_forward_call(self, *args, **kwargs):
        module = self.bend_module(fn=fn)
        if self._graphs.get(fn) is None:
            return getattr(module, fn)(*args, **kwargs)
        else:
            # bend activations
            graph = self.bend_graph(fn=fn)
            graph_module = BendedGraphModule(module, **{fn: graph})
            return getattr(graph_module, fn)(*args, **kwargs)
    def _wrapped_module_forward_call(self, *args, **kwargs):
        return getattr(self._module, fn)(*args, **kwargs)
    return _wrapped_bended_module_forward_call if bend else _wrapped_module_forward_call 

def _get_method_from_graph(module, name):
    method_template = """def method_closure(self, {{SIGNATURE}}):\n\tgm = self.graph_module(fn=\"{{NAME}}\")\n\treturn gm.{{NAME}}({{INPUTS}})"""
    signature = _get_signature_from_graph(module.graph(fn=name))
    graph_inputs = _get_graph_inputs(module.graph(fn=name))
    method_code = _resolve_code(method_template, signature=signature, inputs=graph_inputs, name=name)
    exec(method_code)
    if "method_closure" not in locals():
        raise BendingError(f"Got an error exporting method {name} as a method.")
    return locals()['method_closure'] 

def _get_bended_activation_from_callaback(bended_activations, callback):
    res = []
    for k, v in bended_activations.items():
        if callback in v:
            res.append(k)
    return res

def _copy_attrs(orig, new, attrs):
    for a in attrs:
        setattr(new, a, getattr(orig, a))


class BendedModuleBendingEnv:
    def __init__(self, module, config):
        self.module = module
        self.config = config
        self._previous_bended_config = BendingConfig(
            self.module.bending_config(self.config)
        )

    def __enter__(self):
        # set current bending config to X
        pass
        # self.module.set_

    def __exit__(self, *args):
        # revert previous bending config 
        self.module.save_config(self.config, self._previous_bended_config)


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

class BendedModuleCaptureContext(object):
    def __init__(self, module, callbacks=None):
        self._module = module 
        self._callbacks = callbacks

    def __enter__(self):
        self._module.enable_capture(*(self._callbacks or tuple()))

    def __exit__(self, *args):
        self._module.disable_capture(*(self._callbacks or tuple()))

class BendedModule(object):
    _default_version_key = "_default"
    _default_bending_key = "_default"
    _wrapped_methods = ['forward']

    __copy_attrs__ = [
        "_graphs", "_activations",
        "_bending_callbacks", "_bended_params", 
        "_bended_params_history", "_bended_activations",
        "_interp_dict", "_interp_func",
        "_controllables", "_controllable_hash"
    ]

    # -- Module property --
    def _init_module_(self, module):
        self._module = get_model_copy(module, copy_parameters=True)
        self._module = module
        self._original_module = None
        self._version = self._default_version_key
        self._param_dict = {self._default_version_key: {}}
        self._config = self._default_bending_key
        self._bconfig_dict = {self._default_bending_key: BendingConfig()}
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
    def _getmodule_(self) -> Union[BendedGraph, None]:
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
    def _getversion_(self) -> Union[str, None]:
        return self._version
    def _delversion_(self) -> NoReturn:
        self._version = self._default_version_key
    version = property(_getversion_, _setversion_, _delversion_)

    # -- Config property --
    def _getconfig_(self):
        return str(self._config)
    def _setconfig_(self, config): 
        if config == None:
            self.set_config(self._default_bending_key)
        elif isinstance(config, str):
            self.set_config(config)
        else:
            raise BendingError('Cannot set config to %s'%config)
    def _delconfig_(self):
        if self._config == self._default_bending_key:
            raise BendingError("Cannot erase default bending configuration.")
        self.set_config(self._default_bending_key)
    config = property(_getconfig_, _setconfig_, _delconfig_)

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
                self._register_forward_call(attr_name, with_bended=_is_bended)
                return super(type(self), self).__getattribute__(attr_name)
            else:
                return attr
            
    def __repr__(self):
        module_repr = type(self.module).__name__
        return "BendedModule(%s)"%module_repr

    # -- copy & to
    @classmethod
    def copy(cls, module):
        module_copy = BendedModule(module.module)
        _copy_attrs(module, module_copy, cls.__copy_attrs__)
        return module_copy

    def to(self, *args, _no_warning: bool = False, **kwargs):
        if not _no_warning: 
            print('[Warning]to on bended modules is an experimental feature of torchbend ;\nsetting original module to target device before wrapping is advised.')
            print('call with _no_warning=True to remove this warning.')
        # make shallow copy to change module.
        obj = type(self).copy(self)
        obj._module = obj._module.to(*args, **kwargs)
        for k, v in obj._param_dict.items():
            for kk, vv in v.items():
                obj._param_dict[k][kk] = vv.to(*args, **kwargs)
        return obj

    # -- parameters & weights --
    def weights(self, *flt, exclude=None):
        """get valid weight names from a regexp"""
        parameters = OrderedDict(self.named_parameters())
        valid_parameters = {}
        if len(flt) > 0:
            for f in flt:
                valid_parameters.update(dict(filter(lambda x, r=f: re.match(r, x[0]) is not None, parameters.items())))
        else:
            valid_parameters = parameters
        if exclude is not None:
            for e in checklist(exclude):
                valid_parameters = dict(filter(lambda x, r=e: re.match(r, x[0]) is None, valid_parameters.items())) 
        return valid_parameters

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
    def print_weights(self, *flt, exclude=None, out=None) -> str:
        """print / export weights"""
        parameters = self.weights(*flt, exclude=exclude)
        pretty_weights = list(map(_get_weight_properties, parameters.items()))
        pretty_weights_txt = tabulate(pretty_weights, headers=['name', 'shape', 'dtype', 'min', 'max', 'mean', 'stddev'])
        if out is None:
            if get_output() == TorchbendOutput.RAW:
                print(pretty_weights_txt)
            elif get_output() == TorchbendOutput.NOTEBOOK:
                display_table_for_jupyter(pretty_weights, columns=['name', 'shape', 'dtype', 'min', 'max', 'mean', 'stddev'], display=True)
        elif isinstance(out, TextIOWrapper):
            out.write(pretty_weights_txt)
        else:
            out = pathlib.Path(out)
            os.makedirs(out.parent, exist_ok=True)
            with open(out, 'w+') as f:
                f.write(pretty_weights_txt)
        return pretty_weights_txt

    @_import_to_interface
    def print_aliases(self, out=None):
        return self.print_activations(*sum(self.aliases().values(), tuple()), out=out)

    # -- overrides from nn.Module -- 
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

    @_import_to_interface
    def all_activations(self, with_bended: bool = True):
        activations = {}
        for n, v in self._activations.items():
            activations.update({f"{n}:{k}": v_tp for k, v_tp in v.items()})
        if with_bended:
            for n, v in self._bended_activations.items():
                activations.update({f"{n}:{k}_bended": ActivationProperties(name=k+"_bended", op="bended", fn=n) for k, v_tp in v.items()})
        return activations

    def _parse_aliases(self, flt):
        methods, name = flt.split(':')
        if not name.startswith("#"): 
            return [flt]
        if not methods.startswith('('): methods = f"({methods})"%methods
        methods = re.match(r"\(?([\w|]*)\)?$", methods).groups()[0].split('|')
        flt_out = []
        for fn_tmp in methods:
            graph = self._graphs[fn_tmp]
            if not hasattr(graph, "aliases"): 
                print('[Warning] aliases not found for function %s'%fn_tmp)
            method, name = flt.split(':')
            if name.startswith("#"):
                name = name[1:]
                if name not in graph.aliases: continue
                flt_out.extend(list(map(lambda x, m=method: f"{m}:{x}$", graph.aliases[name])))
            else:
                flt_out.append(flt)
        return flt_out

    @_import_to_interface
    def activations(self, *flt, fn=None, op=None, exclude=None, with_bended: bool = True, _with_fn: bool = False, _raise_notfound: bool = False):

        if len(flt) == 0: 
            raise ValueError('BendedModule.activations take at least one regexp (give ".*" to retrieve everything)')

        if exclude is not None: 
            exclude = checklist(exclude)
        if isinstance(fn, (tuple, list)) or fn is None:
            _with_fn = True
        
        # add callback to regexp if needed
        if fn is None:
            if len(self._activations) == 0: return {}
            fn = list(self._activations.keys())
        fn = checklist(fn)
        flt = list(flt)

        # parse aliases
        for i, k in enumerate(list(flt)):
            flt_with_aliases = []
            if k[0] == "#" and k[1:] in self.aliases():
                for f in fn:
                    aliases = [f"{f}:{x}" for x in self.aliases(fn=f).get(k[1:], [])]
                    flt_with_aliases.extend(aliases)
            else:
                flt_with_aliases.append(k)
        flt = flt_with_aliases

        # add prefix for method
        for i, f in enumerate(flt):
            if ":" not in f: 
                if flt[i].startswith('?'):
                    if len(flt) > 1:
                        if flt[1] == "^": flt = f"?{flt[2:]}"
                    flt[i] = f"?^({'|'.join(fn)}):{f[1:]}$"
                else:
                    flt[i] = f"?^({'|'.join(fn)}):{f}$"
        if exclude:
            for i, e in enumerate(exclude):
                if ":" not in e: exclude[i] = f"({'|'.join(fn)}):{e}"
        # flt = sum([self._parse_aliases(f) for f in flt], [])
        
        # exclude = sum([self._parse_aliases(f) for f in exclude], [])
        activations = self.all_activations(with_bended=with_bended)
        if op is not None:
            op = checklist(op)
            activations = dict(filter(lambda obj: obj[1].op in op, activations.items()))

        valid_activations = {}
        if len(flt) == 0: flt = [f"{f}:" for f in fn]
        for f in checklist(flt):
            if f.startswith('?'):
                f = f[1:]
                valid_activations.update(dict(filter(lambda x, r=f: re.match(r, x[0]) is not None, activations.items())))
            else:
                valid_activations.update(dict(filter(lambda x, r=f: x[0]==r is not None, activations.items())))

        if exclude is not None:
            for e in checklist(exclude):
                valid_activations = dict(filter(lambda x, r=e: re.match(r, x[0]) is None, valid_activations.items())) 
                
        if len(valid_activations) == 0 and _raise_notfound:
            raise BendingError(f"No corresponding key have been found for flt={flt}, fn={fn}, op={op}, exclude={exclude}")

        if not _with_fn:
            valid_activations = {k.split(':')[1]: v for k, v in valid_activations.items()}
        return valid_activations

    @_import_to_interface
    def activation_names(self, *flt, **kwargs):
        if len(flt) == 0: flt = [".*"]
        names = list(self.activations(*flt, **kwargs).keys()) 
        return names

    @_import_to_interface
    def activation_shape(self, param, fn="forward"):
        if ":" in param:
            fn, param = param.split(":")
        if fn not in self._activations: raise BendingError("function %s does not exist or not traced yet"%(fn))
        return self._activations[fn][param].shape


    # @_import_to_interface
    # def print_graph(self,  fn="forward", op=None, flt=r".*", exclude=None, out=None) -> str:
    #     graph = self._graphs[fn]
    #     if op is not None: op = checklist(op)
    #     graph_parsed = [[n.op, n.name, n.target, n.args, n.kwargs]
    #                   for n in graph.nodes]
    #     if op is not None:
    #         graph_parsed = list(filter(lambda x: x[0] in op, graph_parsed))
    #     if flt is not None:
    #         for f in checklist(flt):
    #             graph_parsed = list(filter(lambda x, r=f: re.match(r, x[1]) is not None, graph_parsed)) 
    #     if exclude is not None:
    #         for e in checklist(exclude):
    #             graph_parsed = list(filter(lambda x, r=e: re.match(r, x[1]) is None, graph_parsed)) 
    #     graph_txt = tabulate(graph_parsed,
    #           headers=['opcode', 'name', 'target', 'args', 'kwargs'])
    #     if out is None:
    #         if get_output() == TorchbendOutput.RAW:
    #             print(graph_txt)
    #         elif get_output() == TorchbendOutput.NOTEBOOK:
    #             display_table_for_jupyter(graph_parsed, columns=['opcode', 'name', 'target', 'args', 'kwargs'], display=True)
    #     elif isinstance(out, TextIOWrapper):
    #         out.write(graph_txt)
    #     else:
    #         out = pathlib.Path(out)
    #         os.makedirs(out.parent, exist_ok=True)
    #         with open(out, 'w+') as f:
    #             f.write(graph_txt)
    #     return graph_txt

    @_import_to_interface
    def print_activations(self, *flt, fn="forward", op=None, exclude=None, out=None, fields=None, _with_fn: bool = False) -> str:
        if len(flt) == 0: flt = [".*"]
        activations = self.activations(*flt, fn=fn, op=op, exclude=exclude, _with_fn=_with_fn)
        fields = fields or _DEFAULT_ACTIVATION_FIELDS
        act_parsed = list(map(partial(_get_activations_properties, fields=fields), activations.values()))
        act_txt = tabulate(act_parsed)
        if out is None:
            if get_output() == TorchbendOutput.RAW:
                print(act_txt)
            elif get_output() == TorchbendOutput.NOTEBOOK:
                display_table_for_jupyter(act_parsed, columns=fields, display=True)
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

    @property
    @_import_to_interface
    def traced_methods(self):
        return list(self._graphs.keys())

    def _register_forward_call(self, func, with_bended=False):
        setattr(self, func, types.MethodType(_get_wrapped_module_forward_call(func, with_bended), self))

    def trace(self, fn="forward", *args, _return_out=False, _proxied_buffers=[], _no_tensor_for_args=None, **kwargs):
        """Updates inner graph with the target method and inputs"""
        #TODO general split between kwargs with _ at the beginning for tracer
        inputs = Inputs(*args, **kwargs)
        tracer = BendingTracer(func=fn, _no_tensor_for_args=_no_tensor_for_args)
        tracer_out = tracer.trace(self._module, inputs, return_out=_return_out)#, proxied_buffers=_proxied_buffers)
        graph = tracer_out[0] if _return_out else tracer_out
        self._graphs[fn] = graph
        self._activations[fn] = tracer._activations
        self._bended_activations[fn] = dict()
        if fn != "forward":
            self._register_forward_call(fn, True)
        if _return_out:
            return graph, tracer_out[1]
        else:
            return graph

    @_import_to_interface
    def graph(self, fn="forward", bended: bool = False):
        if not fn in self._graphs: raise BendingError('function %s is not graphed'%fn)
        if bended:
            return self._graphs[fn]
        else:
            return self.bend_graph(fn)

    @_import_to_interface
    def aliases(self, fn="forward"):
        alias_dict = self.graph(fn=fn).aliases
        alias_out = {}
        for k, v in alias_dict.items():
            alias_out[k] = sum(list(map(checktuple, v)), tuple())
        return alias_out


    # -- callbacks --
    @_import_to_interface
    def __call__(self, *args, **kwargs):
        """call the module"""
        module = self.bend_module()
        if self._graphs.get('forward') is None:
            return module(*args, **kwargs)
        else:
            # bend activations
            graph = self.bend_graph()
            graph_module = BendedGraphModule(module, forward=graph)
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
                    state_dict[k] = bc(state_dict[k], name=k.replace(".", "_"))
        return state_dict

    def _bended_state_dict_from_interp(self):
        dicts = {}
        for version, weight in self._interp_dict.items():
            bended_dict = self._bended_state_dict_from_version(version)
            dicts[version] = (bended_dict, weight)
        return self._interp_func(self, dicts)
    
    @_import_to_interface
    def bendable_keys(self, *flt, exclude=None, fn="forward", return_weights=True, return_activations=True):
        keys = []
        if return_weights:
            keys = self.weights(*flt, exclude=exclude)
        if self.is_traced(fn) and return_activations:
            acts = self.activations(*flt, fn=fn, exclude=exclude)
            common_keys = set(keys.keys()).intersection(set(acts.keys()))
            if len(common_keys):
                raise BendingError('found common keys between activations and weights : %s. Please specify better your request'%(common_keys))
            keys.update(acts) 
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
    def bended_activations(self, fn="forward"):
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
    def bending_config(self, config=None):
        config = config or self._config
        assert config in self._bconfig_dict, BendingError('config %s not found in module'%config)
        return self._bconfig_dict[config]
        #TODO reconstruct tree or ?
        # bending_config = _bending_config_from_dicts(self._bended_params[version], self._bended_activations.get(fn, {}), module=self)
        # return bending_config

    def _bend_parameter(self, parameter, callback, version=None):
        version = version or self.version
        assert parameter in self.weight_names, "parameter %s not found in module"%parameter
        if callback not in self._bending_callbacks:
            self._bending_callbacks.append(callback)
        self._bended_params[version][parameter] = self._bended_params[version].get(parameter, []) + [callback]
        #TODO register parameter in callback
        callback.register_weight(self._module.get_parameter(parameter), name=parameter)

    def _bend_activation(self, parameter, callback):
        if ":" not in parameter:
            for k in self._graphs.keys(): self._bend_activation(f"{k}:{parameter}", callback)
        else:
            fn, parameter = parameter.split(":")
            if callback not in self._bending_callbacks:
                self._bending_callbacks.append(callback)
            if fn not in self._bended_activations: self._bended_activations[fn] = {}
            self._bended_activations[fn][parameter] = self._bended_activations[fn].get(parameter, []) + [callback]
            try: 
                callback.register_activation(f"{fn}:{parameter}", shape=self.activation_shape(parameter, fn=fn))
            except Exception as e:
                raise BendingError('Cannot bend activation %s with callback %s.\nException : %s\n Proceeding'%(parameter, callback, e))
        
    @_import_to_interface
    def bend_module(self, fn=None, version=None, copy_parameters=True, jit_compatible: bool = False):
        version = version or self.version
        with torch.no_grad():
            # clone module with deep-copying parameters
            module = get_model_copy(self._module, copy_parameters=copy_parameters)
            state_dict = self.bended_state_dict()
            # copy target weights, as load_state_dict method replaces in place
            clone_parameters(module, list(self._bended_params[version].keys()) + self._bended_params_history[self.version])
            #TODO concrete implications of putting strict=False? (buffers seem to escape model copy)
            # loaded bended dict
            module.load_state_dict(state_dict, assign=True, strict=False)
            # add activation callbacks
            fn = list(self._graphs.keys()) if fn is None else checklist(fn)
            for f in fn:
                if self._graphs.get(f) is not None:
                    for k, v in self.bended_activations(f).items():
                        setattr(module, f"{f}_{k}_callback", CallbackChain.create(*v, _jit_compatible=jit_compatible, name=f"CallbackChain_{f}_{k}"))
                    for k, v in self._graphs[f].get_attached_callbacks().items():
                        setattr(module, f"{f}_{k}", v)
                        
            return module

    @_import_to_interface
    def bend_graph(self, fn="forward"):
        callbacks = {k: CallbackChain(*v) for k, v in self._bended_activations[fn].items()}
        graph = graph_transform_nodes(self._graphs[fn], callbacks)
        graph = graph_insert_callbacks(graph, callbacks)
        return graph

    @_import_to_interface
    def graph_module(self, fn=None, module=None, jit_compatible=False):
        if module is None:
            module = self.bend_module(fn=fn, jit_compatible=jit_compatible)
        if fn is None:
            graphs = {k: self.bend_graph(fn=k) for k in self._graphs.keys()}
        else:
            graphs = {fn: self.bend_graph(fn=fn)}
        if jit_compatible:
            graphs = {k: make_graph_jit_compatible(g) for k, g in graphs.items()}
        graph_module = BendedGraphModule(module, **graphs)
        return graph_module

    @_import_to_interface
    def _bend(self, *args, fn=None, verbose=False, bend_param=True, bend_graph=True):

        callback, *params = args
        target_params = [] if not bend_param else list(self.weights(*params).keys())

        # get default target functions
        assert is_bending_callback(callback), "callback must be a BendingCallback instance"
        if fn is None:
            fn = list(self._graphs.keys())
        else:
            fn = checklist(fn)
            fn = list(filter(lambda x: x in self._graphs, fn))

        # get activations
        target_activations = []
        if bend_graph:
            for p in params:
                if ":" in p: 
                    act_fn, p = p.split(':')
                    act_fn = [act_fn]
                else:
                    act_fn = fn
                for f in act_fn: 
                    target_activations.extend(self.activations(p, fn=f, _with_fn=True, with_bended=False))

        # bend weights
        if len(target_params) + len(target_activations) == 0:
            raise BendingError('Could not find bendable elements with specification %s'%params)
        for param in target_params:
            if verbose: 
                print('bending parameter %s with %s...'%(param, callback))
            self._bend_parameter(param, callback)

        # bend activations
        for target_act in target_activations:
            self._bend_activation(target_act, callback)
            if verbose: 
                print('bending activation %s with %s...'%(target_act, callback))

        # extract controllables in case
        self._register_controllables(callback)
        return target_params + target_activations

    def save_config_as(self, new_config_name, force: bool = False):
        if (new_config_name in self._bconfig_dict) and (not force):
            raise BendingError('configuration %s already exists. Provide force=True keyword to enforce.')
        self._bconfig_dict[new_config_name] = BendingConfig(self.bending_config())
    
    def set_config(self, config_name, bend_graph=True, fn=None, bend_param=True):
        assert config_name in self._bconfig_dict
        self._config = config_name
        self.reset_bending(_erase_config=False)
        bended_config = BendingConfig(self._bconfig_dict[config_name])
        # bended_config.bind(self, bend_graph=bend_graph, bend_param=bend_param)
        for b in bended_config:
            self._bend(*b, bend_graph=bend_graph, fn=fn, bend_param=bend_param)

    def save_config(self, config_name, bending_config):
        assert isinstance(bending_config, BendingConfig)
        self._bconfig_dict[config_name] = BendingConfig(bending_config)
        if config_name == self._config:
            self.set_config(config_name)

    def _prepend_fn_to_acts(self, args, fn):
        named_acts = []
        for a, f in product(args, fn):
            named_acts.append(f"{f}:{a}") 
        return named_acts

    @_import_to_interface
    def bend(self, *args, fn=None, config=None, **kwargs):
        config = config or self._config
        if fn is not None: fn = checklist(fn)

        if len(args) == 1:
            bended_config = args[0]
            assert isinstance(args[0], BendingConfig)
        else:
            bended_config = BendingConfig(args)

        bending_env = BendedModuleBendingEnv(self, config)
        # bended_config.bind(self, fn=fn, bend_graph=bend_graph, bend_param=bend_param)
        for b in bended_config:
            _bended_keys = self._bend(*b, fn=fn, **kwargs)
            self._bconfig_dict[config].append((b[0], *_bended_keys))
        return bending_env

    def reset_bending(self, version=None, _erase_config=True):
        version = version or self.version
        self._bending_callbacks = []
        self._bended_params[version] = {}
        self._bended_activations = {k: {} for k in self._bended_activations.keys()}

        self._controllables = {}
        self._controllable_hash = {}
        if _erase_config:
            self._bconfig_dict[self._config] = BendingConfig()

    @_import_to_interface
    def reset(self, version=None):
        self.reset_bending(version=version)
        self._bconfig_dict = {self._default_bending_key: BendingConfig()}
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

    @_import_to_interface
    def controllables(self) -> List[BendingParameter]:
        return copy.copy(self._controllables)

    def update(self, param_name, value):
        """updates value of a given BendingParameter object"""
        if param_name not in self._controllables:
            print("controllables :", self._controllables)
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
    
    def _register_method_from_graph(self, activations, graph, fn, method_name) -> NoReturn:
        graph.change_target_bending_method(method_name)
        self._graphs[method_name] = graph
        self._activations[method_name] = {}
        self._bended_activations[method_name] = {}
        for node in graph.nodes:
            if node.name.endswith('_bended'):
               continue 
            if node.name in self._activations[fn]:
                self._activations[method_name][node.name] = self._activations[fn][node.name]
            if node.name in self._bended_activations[fn]: 
                # self._bended_activations[method_name][node.name] = self._bended_activations[fn][node.name]
                bending_callbacks = self._bended_activations[fn][node.name]
                for cb in bending_callbacks:
                    cb.copy_activation(f"{fn}:{node.name}", f"{method_name}:{node.name}")
                for k, v in self.bended_activations().items():
                    self._graphs[method_name].attach_bending_callback(f"{k}_callback", CallbackChain(*v))
        # node_names = [n.name for n in graph.nodes]
        # for act, bendings in self._bended_activations[fn].items():
        #     if f"{act}_bended" not in node_names: continue
        #     for b in bendings:
        #         # self.bend(b, f"{act}$", fn=method_name)
        setattr(self, method_name, types.MethodType(_get_method_from_graph(self, method_name), self))

    def inputs_for(self, method_name, **kwargs):
        model_signature = inspect.signature(getattr(self, method_name))
        inputs = {}
        for arg in dict(model_signature.parameters).keys():
            if arg in kwargs: inputs[arg] = kwargs[arg]
        return inputs

    @_import_to_interface
    def get_activations(self, 
                        *activations, 
                        fn="forward", 
                        as_dict=True,
                        # bended=False,
                        _return_graph=False, 
                        _save_as_method=None, 
                        **inputs):
        """return target activations from given inputs."""
        if len(activations) == 0:
            raise BendingError("please provide activations to get_activations method")
        # modify graph
        module = self.bend_module(fn=fn)
        graph = self.bend_graph(fn=fn)

        # if bended: 
        #     activations = list(activations)
        #     bended_activations = self.bended_activations(fn)
        #     for i, a in enumerate(activations):
        #         if a in bended_activations: activations[i] += "_bended"

        activations = list(self.activations(*activations, _raise_notfound=True, fn=fn, with_bended = True).keys())
        #TODO parse node's children and remove then to get minimal graphs? 
        new_graph = graph_get_activations(graph, activations)

        # forward
        gm = BendedGraphModule(module, **{fn: new_graph})
        try:
            outs = getattr(gm, fn)(**inputs)
        except Exception as e:
            raise BendingError('Error by forwarding graph module. Caught error: \n %s'%e)

        if as_dict:
            outs = checktuple(outs)
            outs = {activations[i]: outs[i] for i in range(len(outs))}
        if _save_as_method: 
            self._register_method_from_graph(activations, new_graph, fn, _save_as_method)

        if _return_graph:
            return outs, new_graph
        else:
            return outs

    @_import_to_interface
    def from_activations(self,
                        *activations: Optional[Tuple[str]], 
                        callbacks: Optional[Tuple[BendingCallback]] = None, 
                        fn: str = "forward", 
                        _return_graph = False, 
                        _save_as_method=None, 
                        **inputs):
        #TODO add method to target name of callbacks in activation bending
        assert fn in self._graphs, "method %s is not accessible or isn't traced yet."%(fn)
        if len(activations) == 0:
            assert callbacks is not None, "bend_activation_as_input must be given a recorded callback if activation is not given."
            activations = []
            for callback in callbacks: 
                if not isinstance(callback, BendingCallback): raise TypeError("callback must be a BendingCallback, got : %s"%(type(callback).__name__))
                activations = _get_bended_activation_from_callaback(self._bended_activations[fn], callback)
                assert len(activations) != 0, "given callback does not seem to be bending any activation for method %s.\nCallback : %s"%(fn, callback)

        activations = list(self.activations(*activations, _raise_notfound=True, fn=fn, with_bended = False).keys())
        graph = self.bend_graph(fn=fn)
        bended_activations = list(filter(lambda a: a in self._bended_activations[fn], activations))
        callbacks = {a: CallbackChain(*self._bended_activations[fn][a]) for a in bended_activations}
        new_graph = graph_from_activations(graph, activations, remove_placeholders=True, parse_inputs_from_callbacks=callbacks)
        gm =  BendedGraphModule(self.bend_module(fn=fn), **{fn: new_graph})
        outs = getattr(gm, fn)(**get_kwargs_from_gm(gm, fn=fn, **inputs))
        if _save_as_method:
            self._register_method_from_graph(activations, new_graph, fn, _save_as_method)
        if _return_graph:
            return outs, graph
        else:
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
    
    @_import_to_interface
    def enable_capture(self, *callbacks):
        for c in callbacks:
            assert c in self._bending_callbacks
        if len(callbacks) == 0: callbacks = self._bending_callbacks
        for c in callbacks:
            c.capture()

    @_import_to_interface
    def disable_capture(self, *callbacks):
        for c in callbacks:
            assert c in self._bending_callbacks
        if len(callbacks) == 0: callbacks = self._bending_callbacks
        for c in callbacks:
            if c.capturing: c.stop()

    @_import_to_interface
    def capture(self, *callbacks):
        return BendedModuleCaptureContext(self, callbacks)

    @_import_to_interface
    def interpolate_bending(self, *bending_parameters, **inputs):
        assert not False in [lambda x: x in self._controllables, bending_parameters]



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
