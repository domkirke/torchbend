import re, copy
from enum import Enum
import inspect
import sys
import torch
import functools
from types import FunctionType
from typing import Union, Union,  Callable, Optional, Any, Dict, List, Type
from torch._C import ScriptObject  # type: ignore[attr-defined]
from torch.fx._symbolic_trace import _proxyable_classes, Tracer, _Patcher, _autowrap_check, _patch_wrapped_functions
from torch.fx.proxy import Proxy, TraceError, TracerBase, ParameterProxy
from torch.fx.node import Argument, Node
from torch.fx.graph import Graph
from .. import distributions as dist, DEBUG
from .proxy import *
from .input import Inputs
from ..utils import checklist
from .utils import dist_to_tensor

_orig_module_call: Callable = torch.nn.Module.__call__
_orig_module_getattr: Callable = torch.nn.Module.__getattr__
_is_fx_tracing_flag = False

def is_fx_tracing():
    return _is_fx_tracing_flag

"""
torchbend's tracing philosophy :

default torch.fx is limited in several aspects, often chosen by design to extend the flexibility of the framework.

However, for our purpose, we would like to reduce this flexibility and improve the capabilities of the tracer, to 
open graph bending to most neural networks, at the cost of generality (a little like scripting for instance.)

# Trace with execution.

to this, we couple graph tracing with actual execution of the graph, allowing to populate the actual size of the
inner activations. This allows to access proxies shapes in real-time, and hence to significantly open the diversity
of traced code (for exemple, no more bugs with things like "n_dim = x.shape[2]".) This is at the cost that graph is
conceived for a specified input shape, and possibly values for control flow (see below).

This done by the overloading of the proxy() function, that is called when a node is created. 



# Increased traced types

## Distributions
torch.fx was also not working with torch.Distribution. We redesigned our own Distribution, as they were not scriptable either,
and made them scriptable by adding an "as_tuple" callback that allows dynamic Distribution instantiation through call_function call from
a given and fixed set of args.

## Control flow
??

## Loops 
?? (idea : making a "LoopProxy" that would allow a maximum lenght, and specific graphing)


"""


## TRACING / PATCHING ADDITIONAL DEFINITIONS

def catch_value(obj: Any):
    if isinstance(obj, BendingProxyInt):
        return int(obj)
    else:
        return obj

def wrapped_range(*args):
    args = tuple(map(catch_value, args))
    return range(*args)

def wrapped_reversed(*args):
    if len(args) > 1: raise TypeError("reversed expected 1 argument, got 2")
    if not hasattr(args[0], "__iter__"): TypeError("'%s' object is not reversible"%(type(args[0])))
    return reversed(list(iter(args[0])))

def _patch_iterators(patcher: _Patcher, frame_dict: Dict[str, Any]):
    patcher.patch(frame_dict, "range", wrapped_range)
    patcher.patch(frame_dict, "reversed", wrapped_reversed)


class FlowStep():
    def __init__(self, obj):
        if isinstance(obj, (Proxy, BendingProxy)):
            # recording proxy
            self.obj = obj
            self.frame = obj.tracer._find_user_frame()
        elif isinstance(obj, Node):
            raise NotImplementedError()
        else:
            raise TypeError(f'{self.__class__.__name__} only takes BendingProxy or Nodes as arguments')

    def __repr__(self):
        return f"LogicalFlowStep(name={self.obj.node.name}, value={self.obj.value}, file={get_code_pos_from_frame(self.frame)})"

class LogicalFlowStep(FlowStep):
    pass

class LoopFlowStep(FlowStep):
    def __init__(self, obj):
        super(LoopFlowStep, self).__init__(obj)
        self.frame = obj.tracer._find_user_frame()
    def __repr__(self):
        return f"LoopFlowStep(name={self.obj.node.name}, file={get_code_pos_from_frame(self.frame)})"
        

class ActivationProperties():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TracingContext():
    
    count = 0
    def __init__(self, tracer: Tracer, state: TracingState):
        self._tracer = tracer
        self._state = state
        self._id = TracingContext.get_id()

    @property
    def state(self):
        return self._state

    @classmethod
    def get_id(cls):
        cls.count += 1
        return int(cls.count)

    def __enter__(self):
        self._tracer.set_current_context(self)

    def __exit__(self, *args, **kwargs):
        self._tracer.remove_context(self)





class BendingTracer(torch.fx.Tracer):
    _dist_count_hash = {}
    # TODO better handling of this
    proxy_buffer_attributes = False 
    _no_tensor_for_args = False

    def __init__(self, *args, func="forward", _no_tensor_for_args=None, **kwargs):
        super(BendingTracer, self).__init__(*args, **kwargs)
        self._no_tensor_for_args = _no_tensor_for_args if _no_tensor_for_args is not None else self._no_tensor_for_args
        self.traced_func_name = func 
        self._active_contexts = []
        self._current_context = None


    def _check_input_values(self, root, inputs):
        signature = inspect.Signature.from_callable(getattr(root, self.traced_func_name))
        for name, arg in dict(signature.parameters).items():
            if name not in inputs:
                if arg.default != inspect._empty:
                    inputs.update_(**{name: arg.default})
                else:
                    print('[Warning] input %s not provided and has not default ; may trigger discrepencies among tracing.'%name)
                # inputs.update(name)
        for i in inputs.keys():
            if i not in signature.parameters:
                print('[Warning] found value for input %s, but not in signature for function %s'%(i, self.traced_func_name))
        return inputs


    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        inputs: Inputs, 
        concrete_args: Optional[Dict[str, Any]] = {},
        proxied_buffers = [],
        return_out: bool = False, 
        **kwargs
    ):
        inputs.update_(**concrete_args)
        inputs = self._check_input_values(root, inputs)

        # initialize all input args of the function, both from concrete_args and inputs
        # TODO no difference between concrete_args and inputs? remove concrete_args?    
        self._input_args = inputs.update_(**concrete_args)

        self._model = root
        self._activations = {}
        self._proxied_buffers = []
        self._values = {k.replace('.', '_'): v.data for k, v in root.named_parameters()}
        # concrete_flow_steps is used for boolean control flow that stay fixed during tracing.
        self._concrete_flow_steps = []

        buffers = dict(root.named_buffers())
        for p in proxied_buffers:
            buffers_to_proxy = list(filter(lambda x, u=p: re.match(u, x), buffers.keys()))
            self._proxied_buffers.extend(buffers_to_proxy)
            for proxied_buffer in buffers_to_proxy:
                self._values[proxied_buffer.replace('.', '_')] = root.get_buffer(proxied_buffer)

        graph = self._trace(root, concrete_args)
        graph.flow_steps = self._concrete_flow_steps

        if return_out:
            out_node = list(filter(lambda x: x.op == "output", self.graph.nodes))[0]
            outs = []
            for o in out_node.args:
                if isinstance(o, (Node)):
                    outs.append(self._values[o.name])
                else:
                    outs.append(None)
            del self._values, self._input_args
            return graph, tuple(outs)
        else:
            del self._values, self._input_args
            return graph

    ## ____________________________________________________________________________________________________________________
    ## CREATE ARGUMENTS



    """
    create_arg callback is called when it is needed to prepare values as arguments for nodes.

    from original torch.fx : 
    '''
    By default, the behavior includes:

        #. Iterate through collection types (e.g. tuple, list, dict) and recursively
           call ``create_args`` on the elements.
        #. Given a Proxy object, return a reference to the underlying IR ``Node``
        #. Given a non-Proxy Tensor object, emit IR for various cases:
            * For a Parameter, emit a ``get_attr`` node referring to that Parameter
            * For a non-Parameter Tensor, store the Tensor away in a special
              attribute referring to that attribute.
    '''

    here we extend as follows : 
        - nothing hehehe

    """

    def _tensor_to_scalar(self, arg): 
        if not torch.is_tensor(arg):
            raise TraceError('Got type %s for _tensor_to_scalar, but torch.tensor needed'%(type(arg)))
        if arg.dtype in (torch.float16, torch.float32, torch.float64, torch.float):
            return float(arg)
        elif arg.dtype in (torch.int16, torch.int8, torch.int64, torch.int32, torch.int):
            return int(arg)
        elif arg.dtype in (torch.complex, torch.complex32, torch.complex64, torch.complex128):
            return float(arg.real) + float(arg.imag) * 1j
        else:
            raise TraceError('Cannot transform type %s to python scalar'%(type(arg)))


    def create_arg(self, a: Any) -> "Argument":
        if DEBUG: print('creating arg for', a)
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node("get_attr", n, (), {})
            raise NameError("parameter is not a member of this module")
        elif isinstance(a, torch.Tensor):
            for n_, p_ in self.root.named_buffers():
                if a is p_:
                    return self.create_node("get_attr", n_, (), {})
        elif isinstance(a, torch.nn.Module):
            for n_, p_ in self.root.named_modules():
                if a is p_:
                    return self.create_node("get_attr", n_, (), {}, concrete_value=p_)
        elif isinstance(a, (dist.Distribution, torch.distributions.Distribution)):
            a = self._create_proxy_for_dist(a)

        elif isinstance(a, tuple) and hasattr(a, "_fields"):
            args = tuple(self.create_arg(elem) for elem in a)
            return self.create_node("call_function", a.__class__, args, {})

        elif isinstance(a, (torch.Tensor, ScriptObject)):
            qualname: Optional[str] = self.tensor_attrs.get(a)
            if not qualname:
                i = 0
                while True:
                    qualname = f"_tensor_constant{i}"
                    if not hasattr(self.root, qualname):
                        break
                    i += 1
                self.tensor_attrs[a] = qualname
                setattr(self.root, qualname, a)
            # we pass the concrete_value to populate the tensor's shape during tracing
            return self.create_node("get_attr", qualname, (), {}, concrete_value=a)

        # elif isinstance(a, ShapeAttribute):
        #     if a.has_static:
        #         return a
        #     else:
        #         raise TraceError('tried to convert shape attribute as arg')

        elif type(a) in _proxyable_classes:
            # This is an instance of a proxyable class for which we did not
            # witness its construction. Intern this as a constant attribute
            i = 0
            while True:
                qualname = f"_{a.__class__.__name__}_constant_{i}"
                if not hasattr(self.root, qualname):
                    break
                i += 1
            setattr(self.root, qualname, a)

            return self.create_node("get_attr", qualname, (), {})

        arg = TracerBase.create_arg(self, a)
        if self._no_tensor_for_args and torch.is_tensor(arg):
            if len(arg.shape) == 0:
                arg = self._tensor_to_scalar(arg)
            else:
                raise TraceError('tracer with _no_tensor_for_args=True encoutered a non scalar argument value, that is not handled yet.')
        return arg
 

    ## ____________________________________________________________________________________________________________________
    ## CALLBACKS


    """
    Below are implemented executions corresponding to graph tracing. This way, each time a node is created,
    it also executed using the callbacks below to allow shape propagation at tracing. 
    """

    def _replace_with_value(self, val):
        if isinstance(val, Node):
            if val.name in self._values:
                return self._values[val.name]
            else:
                raise TraceError("bending proxy %s has no value"%val)
        else:
            return val
          
    def _replace_args(self, args):
        """replaces proxy and nodes with actual values."""
        new_args = list(args)
        for i, n in enumerate(args):
            if isinstance(n, (tuple, list)):
                new_args[i] = self._replace_args(n)
            elif isinstance(n, (Proxy, Node)):
                if isinstance(n, ShapeAttribute):
                    print('coucuo')
                if n.name in self._values:
                    new_args[i] = self._values[n.name]
                elif hasattr(n, "concrete_value"):
                    if n.concrete_value:
                        new_args[i] = n.concrete_value
            elif isinstance(n, slice):
                new_args[i] = slice(
                    self._replace_with_value(n.start),
                    self._replace_with_value(n.stop),
                    self._replace_with_value(n.step)
                )
        return type(args)(new_args)
    
    def _replace_kwargs(self, kwargs):
        """replaces proxy and nodes with actual values."""
        kwargs = dict(kwargs)
        for k, n in kwargs.items():
            if isinstance(n, (Proxy, Node)):
                if n.name in self._values:
                    kwargs[k] = self._values[n.name]
        return kwargs

    def placeholder(self, n: Node) -> Any:
        """returns the actual value of a placeholder's input"""
        if n.name in self._input_args:
            return self._input_args[n.name]
        else:
            # for concrete args
            names = list(filter(lambda x: re.match(f"{x}_\d+", n.name), self._input_args.keys()))
            if len(names) > 0:
                return self._input_args[names[0]]
            else:
                return None
    
    def get_attr(self, n: Node) -> Any:
        args = self._replace_args(n.args)
        if n.name in self._values:
            if len(args) > 0:
                return getattr(self._values[n.target], args[0])
            else:
                # this happends when....? (it occurs, check why)
                return self._values[n.name]
        else:
            return 

    def call_function(self, n: Node) -> Any:
        args = self._replace_args(n.args)
        kwargs = self._replace_kwargs(n.kwargs)
        try: 
            return n.target(*args, **kwargs)
        except Exception as e:
            if (n.target == getattr) and (args[0] is None) and (args[1] == "shape"):
                print('Try to access shape attribute of None; try providing input for the placeholder %s'%(n.args[0]))
                raise TraceError('Try to access shape attribute of None ; check input values?')#; try providing input for the placeholder %s'%(n.args[0]))
            else:
                raise e

    def call_method(self, n: Node) -> Any:
        target = n.target
        args = self._replace_args(n.args)
        kwargs = self._replace_kwargs(n.kwargs)
        self_obj, *args_tail = args
        if isinstance(self_obj, Node):
            if hasattr(self_obj, "concrete_value"):
                self_obj = self_obj.concrete_value
            else:
                self_obj = self.run_node(self_obj)
                if self_obj is None:
                    raise TraceError('Cannot call method on node')
        return getattr(self_obj, target)(*args_tail, **kwargs)

    def call_module_run(self, n: Node) -> Any:
        target = n.target
        args = self._replace_args(n.args)
        kwargs = self._replace_kwargs(n.kwargs)
        module = self._get_unwrapped_submodule(self._model, target)
        if DEBUG: print('calling module', n, n.target, n.args)
        with self.create_context(TracingState.RUNNING):
            return type(module).forward(module, *args, **kwargs)

    def call_function_run(self, n: Node) -> Any:
        target = n.target
        args = self._replace_args(n.args)
        kwargs = self._replace_kwargs(n.kwargs)
        if DEBUG: print('calling function', n, n.target, n.args)
        with self.create_context(TracingState.RUNNING):
            args = self._replace_args(args)
            return target(*args, **kwargs)
    
    def run_node(self, n):
        if n.op == "call_module":
            return self.call_module_run(n)
        elif n.op == "call_function":
            return self.call_function_run(n)
        else:
            return getattr(self, n.op)(n)

    def output(self, n: Node) -> Any:
        return self._replace_args(n.args)[0]

    ## ____________________________________________________________________________________________________________________
    ## NODES

    def create_node(self, kind : str, target, args, kwargs, name = None, type_expr = None, dist_args = {}, concrete_value = None) -> Node:
        """
        original torch.fx documentation : 
            '''
            Inserts a graph node given target, args, kwargs, and name.

            This method can be overridden to do extra checking, validation, or
            modification of values used in node creation. For example, one might
            want to disallow in-place operations from being recorded.
            '''
        """

        if DEBUG: print('creating node', kind, target, args, kwargs, name)
        # create node
        if kind == "output":
            new_args, new_type_expr = self._check_output_args(args, type_expr=type_expr, dist_args=dist_args)
            node = super(BendingTracer, self).create_node(kind, target, tuple(new_args), kwargs, name=name)#, type_expr = new_type_expr)
        else:
            node = super(BendingTracer, self).create_node(kind, target, args, kwargs, name=name, type_expr = type_expr)
        node.concrete_value = concrete_value
        return node

    ## ____________________________________________________________________________________________________________________
    ## PROXIES

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None):
        """
        original torch.fx documentation : 
            '''
            Create a Node from the given arguments, then return the Node
            wrapped in a Proxy object.

            If kind = 'placeholder', then we're creating a Node that
            represents the parameter of a function. If we need to encode
            a default parameter, we use the ``args`` tuple. ``args`` is
            otherwise empty for ``placeholder`` Nodes.
            '''
        """
        if DEBUG: print("creating proxy : ", kind, target, args, kwargs, name)
        if (type_expr == int) and (proxy_factory_fn is None):
            proxy_factory_fn = functools.partial(self.proxy, proxy_type=BendingProxyInt, type_expr=int)
        else:
            proxy_factory_fn = functools.partial(self.proxy, proxy_type=BendingProxy, type_expr=type_expr)
        return super().create_proxy(kind, target, args, kwargs, name=name, type_expr=type_expr, proxy_factory_fn=proxy_factory_fn)

    def proxy(self, node: Node, proxy_type=BendingProxy, type_expr=None) -> 'BendingProxy':
        """
        this is the default callback to call a proxy after the node creation, if not specified
        with node_factory when calling create_proxy.
        This is where we record various activations shapes, and corresponding values. 
        """
        if DEBUG: print("creating proxy for node : ", node)
        # proxys are "empty values" that are populated among tracing.
        if node.op == "placeholder":
            out = self._input_args.get(node.name)
        else:
            if node.concrete_value is not None:
                out = node.concrete_value
            else: 
                out = self.run_node(node)

        self._values[node.name] = out
        shape = self._get_shape(out)
        self._activations[node.name] = ActivationProperties(op=node.op, shape=shape)
        proxy = proxy_type(node, self, value=out, type_expr=type_expr)
        return proxy

    def dynamic_shape_proxy(self, node: Node) -> 'ShapeAttribute':
        if DEBUG: print("creating dynamic shape attribute for node : ", node)
        # proxys are "empty values" that are populated among tracing.
        if node.concrete_value is not None:
            out = node.concrete_value
        else: 
            out = self.run_node(node)

        self._values[node.name] = out
        shape = self._get_shape(out)
        self._activations[node.name] = ActivationProperties(op=node.op, shape=shape)
        proxy = ShapeAttribute(node, self, value=out)
        return proxy

    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        """
        Method that specifies the behavior of this ``Tracer`` when we call getattr
        on a call to an ``nn.Module`` instance.

        By default, the behavior is to return a proxy value for the attribute. It
        also stores the proxy value in the ``parameter_proxy_cache``, so that future
        calls will reuse the proxy rather than creating a new one.

        This method can be overridden to --for example-- not return proxies when
        querying parameters.

        Args:

            attr (str): The name of the attribute being queried
            attr_val (Any): The value of the attribute
            parameter_proxy_cache (Dict[str, Any]): A cache of attr names to proxies

        Return:

            The return value from the getattr call.
        """
        def maybe_get_proxy_for_attr(
            attr_val, collection_to_search, parameter_proxy_cache
        ):
            for n, p in collection_to_search:
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        kwargs = {}
                        if (
                            "proxy_factory_fn"
                            in inspect.signature(self.create_proxy).parameters
                        ):
                            kwargs["proxy_factory_fn"] = (
                                None
                                if not self.param_shapes_constant
                                else lambda node: ParameterProxy(
                                    self, node, n, attr_val
                                )
                            )
                        val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)  # type: ignore[arg-type]
                        parameter_proxy_cache[n] = val_proxy
                    self._values[attr] = attr_val
                    return parameter_proxy_cache[n]
            return None

        if isinstance(attr_val, torch.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_parameters(), parameter_proxy_cache
            )
            if maybe_parameter_proxy is not None:
                return maybe_parameter_proxy

        if len(self.module_stack) > 0:
            full_attr_name = f"{list(self.module_stack)[-1]}.{attr}"
        else:
            full_attr_name = attr
        if (self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor)) or (full_attr_name in self._proxied_buffers):
            if attr in self._proxied_buffers and self.get_current_context_state() == TracingState.RUNNING:
                return attr_val
            else:
                maybe_buffer_proxy = maybe_get_proxy_for_attr(
                        attr_val, self.root.named_buffers(), parameter_proxy_cache
                    )
                if maybe_buffer_proxy is not None:
                    return maybe_buffer_proxy

        return attr_val


    ## ____________________________________________________________________________________________________________________
    ## UTILS

    def record_loop(self, obj):
        self._concrete_flow_steps.append(LoopFlowStep(obj))

    def _get_dist_count(self, dist_name: str):
        if not dist_name in self._dist_count_hash:
            self._dist_count_hash[dist_name] = 0
        dist_count = int(self._dist_count_hash[dist_name])
        self._dist_count_hash[dist_name] += 1
        return dist_count
    
    def _check_output_args(self, a, type_expr = None, dist_args = {}, force_tensor_out = False):
        if not isinstance(a, (tuple, list, torch.fx.Node)):
            return a, type_expr
        if isinstance(a, (tuple, list)):
            if type_expr is None:
                type_expr = [None]*len(a)
            else:
                type_expr = checklist(type_expr, n=len(a))
                assert len(type_expr) == len(a)
            return tuple(zip(*[self._check_output_args(a[i], type_expr=type_expr[i], dist_args=dist_args) for i in range(len(a))]))
        if a.name.startswith('_dist'):
            if force_tensor_out:
                a = self.create_node("call_function", dist_to_tensor(a.target), args=(a,), kwargs = dist_args, name=a.name+"_tensor")#, type_expr="torch.Tensor")
                type_expr = None
            else:
                a = self.create_node("call_function", dist.convert_to_torch, args=(a,), kwargs = dist_args, name=a.name+"_tensor")#, type_expr="torch.Tensor")
                type_expr = None
            self._values[a.name] = self.run_node(a)
        return a, type_expr
    
    def _get_shape(self, x):
        if torch.is_tensor(x):
            return x.shape
        elif type(x) == (list, tuple):
            return type(x)([self._get_shape(x_i) for x_i in x])
        elif isinstance(x, BendingProxy):
            return self._get_shape(x._value)
        elif hasattr(x, "__len__"):
            return x.__len__()

    def _get_unwrapped_submodule(self, module, address):
        paths = address.split('.')
        for p in paths:
            try: 
                p = int(p)
            except:
                pass
            if isinstance(p, int):
                module = module.__getitem__(p)
            else:
                module = module._modules[p]
        return module

    def _retrieve_proxyless_parameters(self, target):
        matching_keys = {}
        for k, v in self._values.items():
            if re.match(target.replace('.', '_'), k):
                matching_keys[re.sub(target+'_', '', k).replace('_', '.')] = v
        return matching_keys

    def _create_proxy_for_dist(self, a):
        """
        this is where we override creation of distribution, by making a proxy with call_function. 
        arguments have been previously parsed using an "as_tuple" code, and given as arguments for object creation.
        """

        #TODO i don't remember why I did if/elif for each type, I remember this has to do with scripting. Let's check taht
        if isinstance(a, dist.Bernoulli):
            a = a.as_tuple()
            a = self.create_proxy("call_function", dist.Bernoulli, args=a, kwargs={}, name=f"_dist_Bernoulli_{self._get_dist_count('Bernoulli')}")
        elif isinstance(a, dist.Normal):
            a = a.as_tuple()
            a = self.create_proxy("call_function", dist.Normal, args=a, kwargs={}, name=f"_dist_Normal_{self._get_dist_count('Normal')}")
        elif isinstance(a, dist.Categorical):
            a = a.as_tuple()
            a = self.create_proxy("call_function", dist.Categorical, args=a, kwargs={}, name=f"_dist_Categorical_{self._get_dist_count('Categorical')}")
        else:
            a = dist.convert_from_torch(a)
            return self._create_proxy_for_dist(a)
        return a

    def to_bool(self, obj: 'BendingProxy') -> bool:
        if getattr(obj, "_value", None) is not None:
            # trace steps where control flow was made concrete
            self._concrete_flow_steps.append(LogicalFlowStep(obj))
            return bool(obj._value)
        else:
            raise TraceError('symbolically traced variables cannot be used as inputs to control flow')

    ## ____________________________________________________________________________________________________________________
    ##  overriden trace function

    def _find_user_frame(self):
        """
        Find the Python stack frame executing the user code during
        symbolic tracing.
        """
        # We have to do a little dance here. Basically, walk up the callstack and
        # record the first frame not in the pytorch source. This is the frame executing
        # the user code during tracing.
        frame = inspect.currentframe()

        pt_files = ['torch/fx/proxy.py',
                    'torch/fx/_symbolic_trace.py',
                    'torch/fx/experimental/proxy_tensor.py',
                    'torch/_ops.py',
                    'torch/_tensor.py',
                    'torch/utils/_python_dispatch.py',
                    'torch/_prims_common/wrappers.py',
                    'torch/_refs/__init__.py',
                    'torch/_refs/nn/functional/__init__.py',
                    'torch/utils/_stats.py',
                    'torchbend/tracing/tracing.py',
                    'torchbend/tracing/proxy.py',
                    'torchbend/tracing/module.py',
                    ]
        while frame:
            frame = frame.f_back
            if frame and all(not frame.f_code.co_filename.endswith(file) for file in pt_files):
                break

        if not frame:
            return None

        return frame

    def set_current_context(self, context):
        self._active_contexts.append(context)
        self._current_context = context

    def remove_context(self, context):
        try:
            ctx_idx = self._active_contexts.index(context)
            del self._active_contexts[ctx_idx]
        except ValueError:
            pass
        if len(self._active_contexts) > 0:
            self._current_context = self._active_contexts[-1]
        else:
            self._current_context = None

    def create_context(self, state):
        return TracingContext(self, state)

    def get_current_context_state(self):
        if self._current_context is None:
            return TracingState.TRACING
        else:
            return self._current_context.state

    def _trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
    ) -> Graph:
        """
        Trace ``root`` and return the corresponding FX ``Graph`` representation. ``root``
        can either be an ``nn.Module`` instance or a Python callable.

        Note that after this call, ``self.root`` may be different from the ``root`` passed
        in here. For example, when a free function is passed to ``trace()``, we will
        create an ``nn.Module`` instance to use as the root and add embedded constants
        to.


        Args:

            root (Union[Module, Callable]): Either a ``Module`` or a function to be
                traced through. Backwards-compatibility for this parameter is
                guaranteed.
            concrete_args (Optional[Dict[str, any]]): Concrete arguments that should
                not be treated as Proxies. This parameter is experimental and
                its backwards-compatibility is *NOT* guaranteed.

        Returns:

            A ``Graph`` representing the semantics of the passed-in ``root``.
        """
        global _is_fx_tracing_flag
        old_is_fx_tracing_flag = _is_fx_tracing_flag
        _is_fx_tracing_flag = True

        try:
            if isinstance(root, torch.jit._script.RecursiveScriptModule):
                self.root = root 
                fn = getattr(root, self.traced_func_name)
                self.root_module_name = root._get_name()
                self.submodule_paths = {mod: name for name, mod in root.named_modules()}

            elif isinstance(root, torch.nn.Module):
                self.root = root

                assert hasattr(
                    type(root), self.traced_func_name
                ), f"traced_func_name={self.traced_func_name} doesn't exist in {type(root).__name__}"

                fn = getattr(type(root), self.traced_func_name)
                self.root_module_name = root._get_name()
                self.submodule_paths = {mod: name for name, mod in root.named_modules()}
            else:
                self.root = torch.nn.Module()
                fn = root

            tracer_cls: Optional[Type[Tracer]] = getattr(self, "__class__", None)
            self.graph = Graph(tracer_cls=tracer_cls)
            if hasattr(fn, '__code__'):
                code = fn.__code__
                self.graph._co_fields = {
                    'co_name': code.co_name,
                    'co_filename': code.co_filename,
                    'co_firstlineno': code.co_firstlineno,
                }

            # When we encounter a Tensor value that's not a parameter, we look if it
            # is some other attribute on the model. Construct a dict mapping Tensor
            # values to the qualified name here for efficiency. This is used downstream
            # in create_arg
            self.tensor_attrs: Dict[Union[torch.Tensor, ScriptObject], str] = {}

            def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: List[str]):
                for k, v in m.__dict__.items():
                    if isinstance(v, (torch.Tensor, ScriptObject)):
                        self.tensor_attrs[v] = ".".join(prefix_atoms + [k])
                for k, v in m.named_children():
                    collect_tensor_attrs(v, prefix_atoms + [k])

            collect_tensor_attrs(self.root, [])

            if isinstance(fn, FunctionType):
                fn_globals = fn.__globals__  # run before it gets patched
                fn, args = self.create_args_for_root(
                    fn, isinstance(root, torch.nn.Module), concrete_args
                )
            elif isinstance(fn, torch.ScriptMethod):
                fn_globals = {}
                fn, args = self.create_args_for_root(
                    fn, isinstance(root, torch.nn.Module), concrete_args
                )
            else:
                assert TraceError('Cannot trace function %s'%fn)
            
            parameter_proxy_cache: Dict[
                str, Proxy
            ] = {}  # Reduce number of get_attr calls

            # Method dispatch on parameters is not recorded unless it's directly used.
            # Thus, we need to insert a proxy when __getattr__ requests a parameter.
            @functools.wraps(_orig_module_getattr)
            def module_getattr_wrapper(mod, attr):
                attr_val = _orig_module_getattr(mod, attr)
                if self.get_current_context_state() == TracingState.RUNNING:
                    return attr_val
                else:
                    return self.getattr(attr, attr_val, parameter_proxy_cache)

            @functools.wraps(_orig_module_call)
            def module_call_wrapper(mod, *args, **kwargs):
                def forward(*args, **kwargs):
                    return _orig_module_call(mod, *args, **kwargs)

                _autowrap_check(
                    patcher,
                    getattr(getattr(mod, "forward", mod), "__globals__", {}),
                    self._autowrap_function_ids,
                )
                return self.call_module(mod, forward, args, kwargs)

            with _Patcher() as patcher:
                # allow duplicate patches to support the case of nested calls
                patcher.patch_method(
                    torch.nn.Module,
                    "__getattr__",
                    module_getattr_wrapper,
                    deduplicate=False,
                )
                patcher.patch_method(
                    torch.nn.Module, "__call__", module_call_wrapper, deduplicate=False
                )
                _patch_wrapped_functions(patcher)
                _patch_iterators(patcher, fn_globals)
                _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
                for module in self._autowrap_search:
                    _autowrap_check(
                        patcher, module.__dict__, self._autowrap_function_ids
                    )
                self.create_node(
                    "output",
                    "output",
                    (self.create_arg(fn(*args)),),
                    {},
                    type_expr=fn.__annotations__.get("return", None),
                )

            self.submodule_paths = None
        finally:
            _is_fx_tracing_flag = old_is_fx_tracing_flag
        return self.graph


__all__ = ["BendingTracer"]