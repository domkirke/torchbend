import torch
import torch.nn
import collections
from .tracing import ActivationProperties
from types import MethodType
from torch.fx import Graph, GraphModule
from torch.fx.graph_module import _CodeOnlyModule, _copy_attr, _assign_attr, _method_from_src, _WrappedCall
from torch.fx.graph import _custom_builtins, _is_from_torch, _PyTreeCodeGen, Graph, PythonCode
from torch.fx._compatibility import compatibility
from typing import Dict, Union, Any, Dict, List, Callable, Type


def _create_poly_forward_for_gm(forward_args):
    def _poly_forward(self, *args, **kwargs):
        return BendedGraphModule.forward(self, *args, **{k: kwargs.get(k) for k in forward_args})
    return _poly_forward



class BendedGraphModule(GraphModule):


    def __new__(cls: "Type[BendedGraphModule]", *args, **kwargs):
        # each instance of a graph module needs its own forward method
        # so create a new singleton class for each instance.
        # it is a subclass of the user-defined class, the only difference
        # is an extra layer to install the forward method

        # address issue described at https://github.com/pytorch/pytorch/issues/63883
        # in other words, traverse class hierarchy to fix the redundant class definition problem
        for t in cls.__mro__:
            c = t.__qualname__.split(".")[-1]
            if c != "BendedGraphModuleImpl":
                cls = t
                break

        class BendedGraphModuleImpl(cls):  # type: ignore[misc, valid-type]
            pass
        
        setattr(BendedGraphModuleImpl, "graph", BendedGraphModule.graph)
        return torch.nn.Module.__new__(BendedGraphModuleImpl)


    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        class_name: str = "BendedGraphModule",
        # _create_poly_signature: bool = False,
        **kwargs
    ):
        """
        Construct a GraphModule.

        Args:

            root (Union[torch.nn.Module, Dict[str, Any]):
                ``root`` can either be an nn.Module instance or a Dict mapping strings to any attribute type.
                In the case that ``root`` is a Module, any references to Module-based objects (via qualified
                name) in the Graph's Nodes' ``target`` field will be copied over from the respective place
                within ``root``'s Module hierarchy into the GraphModule's module hierarchy.
                In the case that ``root`` is a dict, the qualified name found in a Node's ``target`` will be
                looked up directly in the dict's keys. The object mapped to by the Dict will be copied
                over into the appropriate place within the GraphModule's module hierarchy.

            graph (Graph): ``graph`` contains the nodes this GraphModule should use for code generation

            class_name (str): ``name`` denotes the name of this GraphModule for debugging purposes. If it's unset, all
                error messages will report as originating from ``GraphModule``. It may be helpful to set this
                to ``root``'s original name or a name that makes sense within the context of your transform.
        """
        torch.nn.Module.__init__(self)
        self.__class__.__name__ = class_name


        for k, v in kwargs.items():
            assert isinstance(v, torch.fx.Graph), f"GraphModule must be initialized with a valid sequence of Graph; got {type(v)} for callback {k}"


        #TODO check if graphs are graphing the same object
        graph_nodes = sum([list(g.nodes) for g in kwargs.values()], [])

        if isinstance(root, torch.nn.Module):
            if hasattr(root, "training"):
                self.training = root.training

            # When we pickle/unpickle graph module, we don't want to drop any module or attributes.
            if isinstance(root, _CodeOnlyModule):
                for k, _ in root.named_children():
                    _copy_attr(root, self, k)

                for k, _ in root.named_buffers():
                    _copy_attr(root, self, k)

                for k, _ in root.named_parameters():
                    _copy_attr(root, self, k)

            for node in graph_nodes:
                if node.op in ["get_attr", "call_module"]:
                    assert isinstance(node.target, str)
                    _copy_attr(root, self, node.target)

        elif isinstance(root, dict):
            targets_to_copy = []
            for node in graph_nodes:
                if node.op in ["get_attr", "call_module"]:
                    assert isinstance(node.target, str)
                    if node.target not in root:
                        raise RuntimeError(
                            "Node "
                            + str(node)
                            + " referenced target "
                            + node.target
                            + " but that target was not provided in ``root``!"
                        )
                    targets_to_copy.append(node.target)
            targets_to_copy.sort(key=lambda t: t.count("."))
            for target_to_copy in targets_to_copy:
                _assign_attr(root[target_to_copy], self, target_to_copy)
        else:
            raise RuntimeError("Unsupported type " + str(root) + " passed for root!")

        for k, v in kwargs.items():
            assert isinstance(v, torch.fx.Graph), f"GraphModule must be initialized with a valid sequence of Graph; got {type(v)} for callback {k}"

        self._activations: Dict[str, Dict[str, ActivationProperties] | None] = {}
        self.graph = kwargs 

        # import tracer cls
        tracers = set([getattr(t, "_tracer_cls") for t in self.graph.values()])
        tracers = None if len(tracers) != 1 else tracers.pop()
        self._tracer_cls = None
        if (
            tracers
            and "<locals>" not in tracers.__qualname__
        ):
            self._tracer_cls = tracers

        #TODO wtf is that
        # self._tracer_extras = {}
        # if self.graph._tracer_extras:
        #     self._tracer_extras = self.graph._tracer_extras

        # Dictionary to store metadata
        self.meta: Dict[str, Any] = {}
        self._replace_hooks: List[Callable] = []
        self._create_node_hooks: List[Callable] = []
        self._erase_node_hooks: List[Callable] = []

        for k, v in self._graph.items():
            if hasattr(v, "_attached_bending_callbacks"):
                for name, callback in v.get_attached_callbacks().items(): 
                    setattr(root, f"{k}_{name}", callback)

    __jit_ignored_attributes__ = ["graph", "graphs"]

    @property
    def graph(self) -> Graph | None:
        return self._graph

    def __str__(self) -> str:
        rep = ""
        for k, g in self._graph.items(): 
            rep += f'Graph for {k} : \n', '-'*10
            rep += str(g)
            rep += "\n"
        return rep

    @graph.setter
    def graph(self, graphs: Dict[str, Graph]) -> None:
        """
        
        """
        for method, g in graphs.items():
            assert isinstance(g, Graph), f"Expected a Graph instance, but got {type(g)}"
            g.owning_module = self
            g.lint()
            g.eliminate_dead_code()
            self._activations[method] = getattr(g, "activations", None)
            # self.recompile()
        self._graph = graphs
        self.recompile()
        # self._forward_args = []   
        # self._has_poly_signature = _create_poly_signature
        # if self._has_poly_signature:
        #     self._init_poly_signature()

    @compatibility(is_backward_compatible=True)
    def recompile(self) -> PythonCode:
        """
        Recompile this GraphModule from its ``graph`` attribute. This should be
        called after editing the contained ``graph``, otherwise the generated
        code of this ``GraphModule`` will be out of date.
        """
        self._in_spec = {}
        self._out_spec = {}

        self._code = ""
        self._lineno_map = collections.OrderedDict()
        _lineno_offset = 0

        for method, graph in self._graph.items():

            if isinstance(graph._codegen, _PyTreeCodeGen):
                self._in_spec[method] = graph._codegen.pytree_info.in_spec
                self._out_spec[method] = graph._codegen.pytree_info.out_spec

            python_code = graph.python_code(root_module="self")
            self._code += python_code.src
            self._lineno_map.update({k + _lineno_offset: v + _lineno_offset for k, v in python_code._lineno_map.items()})

            cls = type(self)
            co_fields = graph._co_fields if hasattr(graph, "_co_fields") else {}
            setattr(cls, method, _method_from_src(method, python_code.src, python_code.globals, co_fields))

            if method == "forward":
                cls_call = cls.__call__ if "__call__" in vars(cls) else None
                if "_wrapped_call" not in vars(cls):
                    cls._wrapped_call = _WrappedCall(cls, cls_call)  # type: ignore[attr-defined]

                def call_wrapped(self, *args, **kwargs):
                    return self._wrapped_call(self, *args, **kwargs)

                cls.__call__ = call_wrapped  # type: ignore[method-assign]

        return self._code


    def _activations_as_dict(self) -> Dict[str, Dict[str, Any]]:
        # activations = torch.jit.annotate(Dict[str, Dict[str, Any]], {})
        # for k, v in activations.items():
        #     activations[k] = {}
        # return activations
        return {}

    @property 
    def activations(self) -> Dict[str, Dict[str, Any]]:
        if torch.jit.is_scripting():
            return self._activations_as_dict()
        else:
            return self._activations

    def _init_poly_signature(self):
        graph = self.graph
        placeholders = list(filter(lambda n: n.op == "placeholder", graph.nodes))
        self._forward_args = [p.name for p in placeholders]
        setattr(self, "forward", MethodType(_create_poly_forward_for_gm(self._forward_args), self)) 
