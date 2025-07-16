from . import BendingCallback, CallbackChain


def _callback_pipe(self, obj):
    if isinstance(obj, CallbackChain):
        return CallbackChain(self, *obj.callbacks)
    elif isinstance(obj, BendingCallback):
        return CallbackChain(self, obj)
    else:
        raise TypeError('%s can only be added to CallbackChain or BendingCallback objects'%(type(self).__name__))


def add_bending_ops(cls):
    cls.__rshift__ = _callback_pipe
    return cls

__all__ = ['add_bending_ops']