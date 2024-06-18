import torch
import copy


class Inputs(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def __repr__(self):
        string = "Inputs("
        if len(self.args) > 0:
            string+= "args=%s"%([a.shape for a in self.args],)
        if len(self.kwargs) > 0:
            string+="kwargs=%s"%([f"{k}:{v.shape}" for k, v in self.kwargs.items()],)
        string += ")"
        return string
    def __call__(self):
        return self.args, self.kwargs
    def __contains__(self, elt):
        if isinstance(elt, int):
            return elt < len(self.args)
        else:
            return elt in self.kwargs
    def __getitem__(self, elt):
        if isinstance(elt, int):
            if elt < len(self.args):
                return self.args[elt]
            else:
                raise IndexError('Input does not contain index %s'%elt) 
        else:
            try:
                return self.kwargs[elt]
            except KeyError:
                raise KeyError('Input does not contains elt %s', elt)
    def update_(self, **kwargs):
        self.kwargs.update(kwargs)
        return self
    def update(self, **kwargs):
        obj = copy.copy(self).update_(**kwargs)
        return obj
    def keys(self):
        return self.kwargs.keys()
    def items(self):
        return self.kwargs.items()
    def values(self):
        return self.kwargs.values()
