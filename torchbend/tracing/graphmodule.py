import torch
from types import MethodType


def _create_poly_forward_for_gm(forward_args):
    def _poly_forward(self, *args, **kwargs):
        return BendedGraphModule.forward(self, *args, **{k: kwargs.get(k) for k in forward_args})
    return _poly_forward


class BendedGraphModule(torch.fx.GraphModule):
    def __init__(self, *args, _create_poly_signature=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._forward_args = []   
        self._has_poly_signature = _create_poly_signature
        if self._has_poly_signature:
            self._init_poly_signature()

    def _init_poly_signature(self):
        graph = self.graph
        placeholders = list(filter(lambda n: n.op == "placeholder", graph.nodes))
        self._forward_args = [p.name for p in placeholders]
        setattr(self, "forward", MethodType(_create_poly_forward_for_gm(self._forward_args), self)) 

