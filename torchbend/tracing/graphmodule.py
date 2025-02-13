import torch
from types import MethodType
from torch.fx import Graph, GraphModule
from typing import Dict, Union, Any, Dict


def _create_poly_forward_for_gm(forward_args):
    def _poly_forward(self, *args, **kwargs):
        return BendedGraphModule.forward(self, *args, **{k: kwargs.get(k) for k in forward_args})
    return _poly_forward


class BendedGraphModule(GraphModule):
    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, _create_poly_signature: bool = False, **kwargs):
        super().__init__(root, graph, **kwargs)
        self._activations = getattr(graph, "activations", None)
        self._forward_args = []   
        self._has_poly_signature = _create_poly_signature
        if self._has_poly_signature:
            self._init_poly_signature()

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

