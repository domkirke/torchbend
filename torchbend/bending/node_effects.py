import enum
import torch
from .base import BendingCallback, BendingCallbackException

class CopyArg():
    pass

class ChangeNodeTargetTokens(enum.Enum):
    copy = 0

class ChangeNodeTarget(BendingCallback):
    activation_compatible = True 
    jit_compatible = True
    applied_to_node = True
    tokens = ChangeNodeTargetTokens
    __valid_kwargs__ = {'op', 'target', 'args', 'kwargs', 'name'}
    __valid_ops__ = ['call_method', 'call_module', 'call_function', 'get_attr']


    def __init__(self, **kwargs):
        super().__init__()
        self._check_input_kwargs(**kwargs)
        self.kwargs = kwargs 

    @property
    def copy_arg(self):
        return CopyArg()

    def _check_input_kwargs(self, **kwargs):
        _unvalid_keys = []
        if len(kwargs) == 0: raise BendingCallbackException('ChangeNodeTarget must be given at least one keyword among : %s'%self.__valid_kwargs__)
        for k, v in kwargs.items():
            if k not in self.__valid_kwargs__: _unvalid_keys.append(k)
            if k == "args":
                if not isinstance(v, tuple): raise BendingCallbackException('invalid type for args keyword in ChangeNodeTarget: expected tuple, not %s'%type(v))
            if k == "kwargs":
                if not isinstance(v, dict): raise BendingCallbackException('invalid type for kwargs keyword in ChangeNodeTarget: expected dict, not %s'%type(v))
            if k == "op":
                if not isinstance(v, str): raise BendingCallbackException('invalid type for op in ChangeNodeTarget: expected str, not %s'%type(v))
                if v not in self.__valid_ops__: raise BendingCallbackException('op %s invalid for ChangeNodeTarget.'%v)
            if k == "name":
                if not isinstance(v, str): raise BendingCallbackException('invalid type for name in ChangeNodeTarget: expected str, not %s'%type(v))

        if len(_unvalid_keys) > 0: 
            raise BendingCallbackException("ChangeNodeTarget must be initiliased with one of following arguments : %s. Got unvalid keys : %s"%(self.__valid_kwargs__, _unvalid_keys))

    def _is_kwarg_controllable(self, k, v):
        if isinstance(v, (int, float, bool)): 
            return True
        elif torch.is_tensor(v) and v.ndim == 0: 
            return True
        else:
            return False

    def register_activation(self, name, shape):
        name = super().register_activation(name, shape)
        self._init_permute_(name, shape)

    def _parse_controllable_params(self, **kwargs):
        valid_params = {}
        for k, v in kwargs.items():
            if self._is_kwarg_controllable(k, v): valid_params[k] = v
        return valid_params

    def apply_to_node(self, node):
        for k, v in self.kwargs.items():
            if k == "args":
                new_args = list(v)
                for i, v_tmp in enumerate(v):
                    if v_tmp == ChangeNodeTargetTokens.copy:
                        assert i < len(node.args), "tried to copy argument #%d, but got %d arguments in original node"%(i, len(node.args))
                        new_args[i] = node.args[i]
                v = new_args
            setattr(node, k, v)
        return node

