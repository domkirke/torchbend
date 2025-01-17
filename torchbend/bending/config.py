from types import FunctionType
import dill
import copy

"""
cb = cb.Mask(1)
bending_config = BendingConfig(
    (cb, weights),
    ...,
)
"""

class BendingConfigException(Exception):
    pass


class BendingConfig(object):
    def __init__(self, *args, module=None):
        if (len(args) == 1) and isinstance(args[0], BendingConfig):
            self._import_config(args[0])
        else:
            self._bendings = []
            self._cb_hash = {}
            self._weight_hash = {}
            self._module = module 
            for a in args:
                if not isinstance(a, tuple): raise BendingConfigException("BendingConfig arguments must be tuples (Callable, *str)")
                self._bendings.append(a)
            if module is not None:
                self.bind(module)

    def _import_config(self, config):
        self._bendings = list(config._bendings)
        self._cb_hash = dict(config._cb_hash)
        self._weight_hash = dict(config._weight_hash)
        self._module = config._module

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

    def op_from_key(self, key: str):
        assert self.is_binded, ""
        return self._weight_hash[key]

    def append(self, bend):
        if not type(bend) == tuple: raise BendingConfigException("tuple needed")
        if not len(bend) > 1: raise BendingConfigException("tuple must be (callback, *bended_keys)")
        if not hasattr(bend[0], "__call__"): raise BendingConfigException("tuple must be (callback, *bended_keys)")
        if not set(map(type, bend[1:])) == {str}: raise BendingConfigException("tuple must be (callback, *bended_keys)")
        self._bendings.append(bend)
        if self.is_binded:
            self._bind_bending(bend)

    def extend(self, bends):
        for b in bends: self.append(b)

    @property
    def module(self):
        return self._module

    @property
    def is_binded(self):
        return self._module is not None

    @property
    def binded_fn(self):
        assert self.is_binded, "BendedConfig must be binded to provide binded_fn"
        binded_fn =[]
        for k in self._weight_hash.keys():
            if ":" in k: 
                fn, _ = k.split(':')
                binded_fn.append(fn)
        return binded_fn

    def _bind_bending(self, bending, fn=None, bend_graph=True, bend_param=True):
        assert self.is_binded
        c, *w = bending
        resolved_keys = []
        if bend_param:
            resolved_keys.extend(sum([self.module.resolve_parameters(w_tmp) for w_tmp in w], []))
        if bend_graph:
            resolved_keys.extend(sum([self.module.resolve_activations(w_tmp, fn=fn, _with_fn=True) for w_tmp in w], []))
        for r_w in resolved_keys:
            if r_w not in self._weight_hash: self._weight_hash[r_w] = []
            self._weight_hash[r_w].append(c)
        if c not in self._cb_hash: self._cb_hash[c] = []
        self._cb_hash[c].extend(resolved_keys)

    def bind(self, module, **kwargs):
        self._module = module
        self._cb_hash = {}
        self._weight_hash = {}
        for b in self._bendings:
            self._bind_bending(b, **kwargs)

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
        if self.module: 
            for k, v in self._cb_hash.items():
                yield (k, *v)
        else:
            for b in self._bendings:
                yield b

    def __eq__(self, obj):
        if not isinstance(obj, BendingConfig): raise TypeError("BendingConfig can be compared to BendingConfig, not %s"%(type(obj)))
        if self.is_binded and obj.is_binded: 
            return self._weight_hash == obj._weight_hash
        else:
            return set(self._bendings) == set(obj._bendings)

    def __add__(self, obj):
        if not isinstance(obj, BendingConfig): raise TypeError("BendingConfig can be added to BendingConfig, not %s"%(type(obj)))
        if self.module is not None:
            if not self.module == obj.module: raise BendingConfigException("For concatenations BendingConfig objects must have the same bounded module")
        return BendingConfig(*self._bendings, *obj._bendings, module=self.module)

    def _pickle_obj(self):
        return {
            'bendings': self._bendings, 
            'cb_hash': self._cb_hash, 
            'weight_hash': self._weight_hash, 
            'is_binded': self.is_binded, 
            'obj_type': None if self._module is None else type(self._module).__name__
        }

    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump(self._pickle_obj(), f)

    @classmethod
    def load(self, path, module=None):
        with open(path, 'rb') as f: 
            obj = dill.load(f)
        config = BendingConfig()
        if obj['is_binded']: 
            if type(module).__name__ != obj['obj_type']:
                raise BendingConfigException('BendingConfig was bounded to object of type %s, got %s'%(obj['obj_type'], type(module).__name__))
            config._bendings = obj['bendings']
            if module is None:
                print('[Warning] BendingConfig was bounded to object of type %s, but was not provided.')
                return config
            config._module = module
            config._cb_hash = obj['cb_hash']
            config._weight_hash = obj['weight_hash']
        else:
            config._bendings = obj['bendings']
        return config





        

    
