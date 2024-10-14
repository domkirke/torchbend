from types import FunctionType
import copy

"""
cb = cb.Mask(1)
bending_config = BendingConfig(
    (cb, weights),
    ...,
)
"""


class BendingConfig(object):
    def __init__(self, *args, module=None):
        self._bendings = []
        self._cb_hash = {}
        self._weight_hash = {}
        self._module = module 
        for a in args:
            assert isinstance(a, tuple), "BendingConfig arguments must be tuples (Callable, *str)"
            self._bendings.append(a)
        if module is not None:
            self.bind(module)

    def __repr__(self):
        out = "BendingConfig("
        if self.is_binded:
            out += "\nmodule = %s(id = %d)"%(type(self._module.module).__name__, id(self._module.module))
            for c, v in self._cb_hash.items():
                out += f"\n\t{c}: {v}"
        else: 
            for b in self._bendings:
                out += f"\n\t{b}"
        if len(self._bendings) != 0: out += "\n"
        out += ")"
        return out

    def append(self, bend):
        assert type(bend) == tuple, "tuple needed"
        assert len(bend) > 1, "tuple must be (callback, *bended_keys)"
        assert hasattr(bend[0], "__call__"), "tuple must be (callback, *bended_keys)"
        assert set(map(type, bend[1:])) == {str}, "tuple must be (callback, *bended_keys)"
        self._bendings.append(bend)
        if self.is_binded:
            self._bind_bending(bend)

    @property
    def module(self):
        return self._module

    @property
    def is_binded(self):
        return self._module is not None

    def _bind_bending(self, bending):
        assert self.is_binded
        c, *w = bending
        resolved_weights = sum([self.module.bendable_keys(w_tmp) for w_tmp in w], [])
        for r_w in resolved_weights:
            if not r_w in self._weight_hash: self._weight_hash[r_w] = set()
            self._weight_hash[r_w].union({c})
        if not c in self._cb_hash: self._cb_hash[c] = []
        self._cb_hash[c].extend(resolved_weights)

    def bind(self, module):
        self._module = module
        for b in self._bendings:
            self._bind_bending(b)

    def copy_and_bind(self, module):
        obj = copy.deepcopy(self)
        obj.bind(module)
        return obj
        
    def __contains__(self, key_or_cb):
        if isinstance(key_or_cb, str):
            return key_or_cb in self._weight_hash
        elif isinstance(key_or_cb, FunctionType) or hasattr(key_or_cb, "__call__"):
            return key_or_cb in self._cb_hash
        else:
            raise TypeError('BendingConfig can only contain strings or callables.')

    def __iter__(self):
        return iter(self._bendings)

    def __eq__(self, obj):
        if not isinstance(obj, BendingConfig): raise TypeError("BendingConfig can be compared to BendingConfig, not %s"%(type(obj)))
        if self.is_binded and obj.is_binded: 
            return self._weight_hash == obj._weight_hash
        else:
            return set(self._bendings) == set(obj._bendings)

    def __add__(self, obj):
        if not isinstance(obj, BendingConfig): raise TypeError("BendingConfig can be added to BendingConfig, not %s"%(type(obj)))
        if self.module is not None:
            assert self.module == obj.module, "For concatenations BendingConfig objects must have the same bounded module"
        return BendingConfig(*self._bendings, *obj._bendings, module=self.module)



        

    
