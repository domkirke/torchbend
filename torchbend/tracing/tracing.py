import re, copy
import inspect
import sys
import torch
from typing import Union, Union,  Callable, Optional, Any, Dict
from torch._C import ScriptObject  # type: ignore[attr-defined]
from torch.fx._symbolic_trace import _proxyable_classes
from torch.fx.proxy import Proxy, TraceError, TracerBase
from torch.fx.node import Argument, Node
from .. import distributions as dist
from .proxy import BendingProxy, ShapeAttribute
from .input import Inputs
from ..utils import checklist
from .utils import dist_to_tensor


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



DEBUG = True

class ActivationProperties():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class BendingTracer(torch.fx.Tracer):
    _dist_count_hash = {}

    def __init__(self, *args, func="forward", **kwargs):
        super(BendingTracer, self).__init__(*args, **kwargs)
        self.traced_func_name = func 

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
        **kwargs
    ):
        # how could we prevent model recursion when shape cannot be fetched? making a special object that
        # could be put in _input_args? 
        # kwargs = {}
        inputs.update_(**concrete_args)
        inputs = self._check_input_values(root, inputs)

        # initialize all input args of the function, both from concrete_args and inputs
        # TODO no difference between concrete_args and inputs? remove concrete_args?    
        
        self._input_args = inputs.update_(**concrete_args)

        self._values = {k.replace('.', '_'): v.data for k, v in root.named_parameters()}

        # concrete_flow_steps is used for boolean control flow that stay fixed during tracing.
        self._concrete_flow_steps = []

        self._model = root
        self._activations = {}
        graph = super(BendingTracer, self).trace(root, concrete_args)
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
        return TracerBase.create_arg(self, a)
 
 

    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        # if isinstance(attr_val, torch.nn.Parameter):
        #     # by default, torch.fx creates a proxy for model parameters. 
        #     # this, however, make shape propagation impossible during tracing.
        #     #TODO ayayayay
        #     return attr_val
        # else:
            return super(BendingTracer, self).getattr(attr, attr_val, parameter_proxy_cache)

    ## ____________________________________________________________________________________________________________________
    ## CALLBACKS


    """
    Below are implemented executions corresponding to graph tracing. This way, each time a node is created,
    it also executed using the callbacks below to allow shape propagation at tracing. 
    """
          
    def _replace_args(self, args):
        """replaces proxy and nodes with actual values."""
        new_args = list(args)
        for i, n in enumerate(args):
            if isinstance(n, (tuple, list)):
                new_args[i] = self._replace_args(n)
            elif isinstance(n, (Proxy, Node)):
                if n.name in self._values:
                    new_args[i] = self._values[n.name]
                elif hasattr(n, "concrete_value"):
                    if n.concrete_value:
                        new_args[i] = n.concrete_value
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
        return type(module).forward(module, *args, **kwargs)

    def call_function_run(self, n: Node) -> Any:
        target = n.target
        args = self._replace_args(n.args)
        kwargs = self._replace_kwargs(n.kwargs)
        if DEBUG: print('calling function', n, n.target, n.args)
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
        return super().create_proxy(kind, target, args, kwargs, name=name, type_expr=type_expr, proxy_factory_fn=proxy_factory_fn)

    def proxy(self, node: Node) -> 'Proxy':
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
        proxy = BendingProxy(node, self, value=out)
        return proxy
    
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
        if obj._value is not None:
            # trace steps where control flow was made concrete
            self._concrete_flow_steps.append(obj)
            return bool(obj._value)
        else:
            raise TraceError('symbolically traced variables cannot be used as inputs to control flow')


    ## ____________________________________________________________________________________________________________________
    ## UTILS

    def _get_dist_count(self, dist_name: str):
        if not dist_name in self._dist_count_hash:
            self._dist_count_hash[dist_name] = 0
        dist_count = int(self._dist_count_hash[dist_name])
        self._dist_count_hash[dist_name] += 1
        return dist_count
    
    def _check_output_args(self, a, type_expr = None, dist_args = {}):
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
            a = self.create_node("call_function", dist_to_tensor(a.target), args=(a,), kwargs = dist_args, name=a.name+"_tensor")#, type_expr="torch.Tensor")
            type_expr = None
        return a, type_expr
    
    def _get_shape(self, x):
        if torch.is_tensor(x):
            return x.shape
        elif type(x) in (list, tuple):
            return type(x)([self._get_shape(x_i) for x_i in x])
        elif hasattr(x, "__len__"):
            return len(x)

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


__all__ = ["BendingTracer"]