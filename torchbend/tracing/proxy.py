import torch
from typing import Union, Optional, Iterable, Any
from torch.fx.proxy import Attribute, Proxy, TraceError
from torch.fx.node import Argument, Node


class ShapeAttribute(Attribute):
    def __init__(self, root: Proxy, attr: str):
        super().__init__(root, attr)
        if hasattr(root, "_value"):
            self._static_shape = root._value.shape
        elif hasattr(root.node, "shape"):
            self._static_shape = root.node.shape
        else:
            self._static_shape = None

    def __iter__(self):
        if hasattr(self, "_static_shape"):
            return iter(self._static_shape)
        else:
            return super().__iter__()

    def __len__(self):
        if hasattr(self, "_static_shape"):
            return len(self._static_shape)
        else:
            return super().__len__()

    def __repr__(self):
        return "ShapeAttribute(root=%s, value=%s)"%(self.root, self.attr)

    def __call__(self, *args, **kwargs):
        return super(ShapeAttribute, self).__call__(*args, **kwargs)


class BendingProxy(torch.fx.Proxy):
    def __init__(self, node: Node, tracer = None, value: Optional[Any] = None):
        super(BendingProxy, self).__init__(node, tracer)
        self._value = value

    # def __len__(self):
        # return ShapeAttribute(self, "shape")[0]

    def __repr__(self):
        return "BendingProxy(%s)"%self.node#, value=%s)"%(self.node, self._value if self._value is None else self._value.shape)

    def __getattr__(self, k) -> Union[Attribute, Iterable[int]]:
        if k == "shape":
            return ShapeAttribute(self, k)
        if k == "device":
           return None
        else:
            return Attribute(self, k)


__all__  = ['ShapeAttribute', 'BendingProxy']