import re, copy
import torch
from typing import Union, Union,  Callable, Optional, Any, Dict
from torch._C import ScriptObject  # type: ignore[attr-defined]
from torch.fx._symbolic_trace import _proxyable_classes
from torch.fx.proxy import Proxy, TraceError
from torch.fx.node import Argument, Node
from .. import distributions as dist
from .proxy import ShapedProxy, ShapeAttribute
from .input import Inputs
from ..utils import checklist
from .utils import dist_to_tensor


class ActivationProperties():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class BendingTracer(torch.fx.Tracer):
    _dist_count_hash = {}

    def __init__(self, *args, func="forward", **kwargs):
        super(BendingTracer, self).__init__(*args, **kwargs)
        self.traced_func_name = func 

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
        self._input_args = inputs.update_(**concrete_args)
        self._values = {k.replace('.', '_'): v.data for k, v in root.named_parameters()}
        self._concrete_flow_steps = []
        # self._model = _get_model_copy(root, copy_parameters=True)
        #TODO noooooooOOOOO (how to copy the model before tracing to remove all decorators?)
        # _get_inputs(input_shape, inspect.getfullargspec(getattr(module, func)))
        self._model = copy.deepcopy(root)
        # self._model = root
        self._activations = {}
        graph = super(BendingTracer, self).trace(root, concrete_args)
        del self._values, self._input_args
        # graph._activations = self._activations
        # graph.get_activations = activation_callback(graph)
        return graph

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

    def _create_proxy_for_dist(self, a):
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

    def create_arg(self, a: Any) -> "Argument":

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
                    return self.create_node("get_attr", n_, (), {})
        elif isinstance(a, (dist.Distribution, torch.distributions.Distribution)):
            a = self._create_proxy_for_dist(a)

        if isinstance(a, tuple) and hasattr(a, "_fields"):
            args = tuple(self.create_arg(elem) for elem in a)
            return self.create_node("call_function", a.__class__, args, {})

        if isinstance(a, (torch.Tensor, ScriptObject)):
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

        if type(a) in _proxyable_classes:
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

        return super().create_arg(a)
    
    def _replace_args(self, args):
        new_args = list(args)
        for i, n in enumerate(args):
            if isinstance(n, (tuple, list)):
                new_args[i] = self._replace_args(n)
            elif isinstance(n, (Proxy, Node)):
                if n.name in self._values:
                    new_args[i] = self._values[n.name]
        return type(args)(new_args)

    def _replace_kwargs(self, kwargs):
        kwargs = dict(kwargs)
        for k, n in kwargs.items():
            if isinstance(n, (Proxy, Node)):
                if n.name in self._values:
                    kwargs[k] = self._values[n.name]
        return kwargs
    
    def _get_shape(self, x):
        if torch.is_tensor(x):
            return x.shape
        elif type(x) in (list, tuple):
            return type(x)([self._get_shape(x_i) for x_i in x])
        elif hasattr(x, "__len__"):
            return len(x)

    def placeholder(self, n: Node) -> Any:
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
        args = tuple(n.args)
        args = self._replace_args(n.args)
        if n.name in self._values:
            if len(args) > 0:
                return getattr(self._values[n.target], args[0])
            else:
                return self._values[n.name]
        else:
            return 

    def call_function(self, n: Node) -> Any:
        args = self._replace_args(n.args)
        kwargs = self._replace_kwargs(n.kwargs)
        return n.target(*args, **kwargs)

    def call_method(self, n: Node) -> Any:
        target = n.target
        args = self._replace_args(n.args)
        kwargs = self._replace_kwargs(n.kwargs)
        self_obj, *args_tail = args
        return getattr(self_obj, target)(*args_tail, **kwargs)

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

    def call_module_run(self, n: Node) -> Any:
        target = n.target
        args = self._replace_args(n.args)
        kwargs = self._replace_kwargs(n.kwargs)
        module = self._get_unwrapped_submodule(self._model, target)
        #TODO this shouldn't be here, how to bypass proxysation of parameters
        # module.load_state_dict(self._retrieve_proxyless_parameters(target))
        return type(module).forward(module, *args, **kwargs)

    def output(self, n: Node) -> Any:
        return self._replace_args(n.args)[0]

    def run_node(self, n):
        if n.op == "call_module":
            return self.call_module_run(n)
        return getattr(self, n.op)(n)

    def create_node(self, kind : str, target, args, kwargs, name = None, type_expr = None, dist_args = {}, concrete_value = None) -> Node:
        # create node
        if kind == "output":
            new_args, new_type_expr = self._check_output_args(args, type_expr=type_expr, dist_args=dist_args)
            node = super(BendingTracer, self).create_node(kind, target, tuple(new_args), kwargs, name=name)#, type_expr = new_type_expr)
        else:
            node = super(BendingTracer, self).create_node(kind, target, args, kwargs, name=name, type_expr = type_expr)
        node.concrete_value = concrete_value
        return node

    def proxy(self, node: Node) -> 'Proxy':
        # populate shape
        if node.concrete_value is not None:
            out = node.concrete_value
        else: 
           out = self.run_node(node)
        self._values[node.name] = out
        shape = self._get_shape(out)
        self._activations[node.name] = ActivationProperties(op=node.op, shape=shape)
        proxy = ShapedProxy(node, self, value=out)
        return proxy

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None):
        return super().create_proxy(kind, target, args, kwargs, name=name, type_expr=type_expr, proxy_factory_fn=proxy_factory_fn)

    def to_bool(self, obj: 'ShapedProxy') -> bool:
        if obj._value is not None:
            # trace steps where control flow was made concrete
            self._concrete_flow_steps.append(obj)
            return bool(obj._value)
        else:
            raise TraceError('symbolically traced variables cannot be used as inputs to control flow')






__all__ = ["BendingTracer"]