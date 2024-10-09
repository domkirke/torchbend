import torch
from operator import setitem, getitem
from typing import Union, Optional, Iterable, Any, Dict
from torch.fx.proxy import Attribute, Proxy, TraceError
from enum import Enum
from torch.fx.node import Argument, Node



class CodePosition():
    def __init__(self, frame):
        self.frame = frame

    @property
    def code(self):
        return self.frame.f_code

    @property
    def description(self):
        code = self.code
        desc = f"{code.co_filename}:"
        desc += f"{code.co_name}"
        desc += f".{code.co_firstlineno})"
        return desc

def get_code_pos_from_frame(frame):
    return CodePosition(frame).description


class TracingState(Enum):
    TRACING = 0
    RUNNING = 1

class ShapeAttribute(Attribute):
    #TODO make several behaviors for this.
        
    def __init__(self, root: Proxy, attr: str, static_shape=None, _sub_idxs=None):
        super().__init__(root, attr)
        self._sub_idxs = _sub_idxs
        if static_shape is None:
            if hasattr(root, "_value"):
                self._static_shape = root._value.shape
            elif hasattr(root.node, "shape"):
                self._static_shape = root.node.shape
            else:
                self._static_shape = None
        else:
            self._static_shape is None

    def __iter__(self):
        if hasattr(self, "_static_shape"):
            tracer = self.tracer
            if self.tracer.get_current_context_state() == TracingState.TRACING:
                out = [tracer.create_proxy('call_function', getitem, (self, i,), {},
                                            name=tracer.graph._target_to_str(getitem)) for i in range(len(self.static_shape))]
                return iter(out)
            elif self.tracer.get_current_context_state() == TracingState.RUNNING:
                return iter(self._static_shape)
        #     return iter(self._static_shape)
        # else:
        #     return super().__iter__()

    @property
    def static_shape(self):
        if self._sub_idxs is None:
            return self._static_shape
        else:
            return self._static_shape.__getitem__(*self._sub_idxs)

    def __getitem__(self, *args):
        # static_shape = None if self._static_shape is None else self._static_shape.__getitem__(*args)
        return self.tracer.create_proxy('call_function', getitem, (self, *args), {},
                                        name=self.tracer.graph._target_to_str(getitem), type_expr=int)
        #                                 proxy_factory_fn=self.tracer.dynamic_shape_proxy)
        # return ShapeAttribute(self.root, self.attr, static_shape=self._static_shape, _sub_idxs=args)

    def __len__(self):
        if hasattr(self, "_static_shape"):
            return len(self.static_shape)
        else:
            return super().__len__()

    def __repr__(self):
        return "ShapeAttribute(root=%s, value=%s)"%(self.root, self.attr)

    def __call__(self, *args, **kwargs):
        return super(ShapeAttribute, self).__call__(*args, **kwargs)


class BendingProxy(torch.fx.Proxy):
    def __init__(self, node: Node, tracer = None, value: Optional[Any] = None, type_expr=None):
        super(BendingProxy, self).__init__(node, tracer)
        self._code_pos = CodePosition(self.tracer._find_user_frame())
        self._value = value
        self._type_expr = type_expr

    def __len__(self):
        if self.tracer.get_current_context_state() == TracingState.TRACING:
            if torch.is_tensor(self.value):
                return ShapeAttribute(self, "shape")[0]
            elif self.value is not None:
                return len(self.value)

        elif self.tracer.get_current_context_state() == TracingState.RUNNING:
            return len(self.value)

    def __repr__(self):
        return "BendingProxy(%s)"%self.node#, value=%s)"%(self.node, self._value if self._value is None else self._value.shape)

    @property
    def type_expr(self):
        return self._type_expr

    @property
    def value(self):
        return self._value

    @property
    def code(self):
        return self._code_pos.get_code()

    def __iter__(self):
        tracer = self.tracer
        if not hasattr(self.value, "__iter__"):
            raise TraceError('Tried to iterate over BendyProxy with non-iterable value of type %s'%type(self.value))
        if self.tracer.get_current_context_state() == TracingState.TRACING:
            out = [tracer.create_proxy('call_function', getitem, (self, i,), {},
                                        name=tracer.graph._target_to_str(getitem)) for i in range(len(self.value))]
            tracer.record_loop(self)
            return iter(out)
        elif self.tracer.get_current_context_state() == TracingState.RUNNING:
            return iter(self.value)

    def __getattr__(self, k) -> Union[Attribute, Iterable[int]]:
        if k == "shape":
            return ShapeAttribute(self, k)
        if k == "device":
           return None

        #TODO GENERAL BEHAVIOR
        if k == "is_cuda":
            return self._value.is_cuda
        else:
            return Attribute(self, k)

    def __setitem__(self, idx, val):
        tracer = self.tracer
        return tracer.create_proxy('call_function', setitem, (self, idx, val), {},
                                    name=tracer.graph._target_to_str(setitem))


class BendingProxyInt(BendingProxy):

    def __repr__(self):
        return "BendingProxyInt(%s)"%self.node#, value=%s)"%(self.node, self._value if self._value is None else self._value.shape)

    def __int__(self):
        return int(self._value)

__all__  = ["BendingProxy", "BendingProxyInt", "ShapeAttribute", "TracingState", "get_code_pos_from_frame"]