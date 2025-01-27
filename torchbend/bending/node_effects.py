import enum
import torch
from .base import BendingCallback, BendingCallbackException

class CopyArg():
    pass

class ChangeNodeTokens(enum.Enum):
    copy = 0

class ChangeNodeActivationPointer(object):
    def __init__(self, name):
        self.name = name

class ChangeNode(BendingCallback):
    activation_compatible = True 
    jit_compatible = True
    applied_to_node = True
    copy = ChangeNodeTokens.copy
    __valid_kwargs__ = {'op', 'target', 'args', 'kwargs', 'name'}
    __valid_ops__ = ['call_method', 'call_module', 'call_function', 'get_attr']


    def __init__(self, **kwargs):
        super().__init__()
        self._check_input_kwargs(**kwargs)
        self.kwargs = kwargs 

    @property
    def copy_arg(self):
        return CopyArg()

    @staticmethod
    def activation(name):
        return ChangeNodeActivationPointer(name)

    @staticmethod
    def expression(expression):
        #TODO evaluation expression on target nodes.
        raise NotImplementedError

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

    def retrieve_activation_from_graph(self, graph, name):
        for n in graph.nodes:
            if n.name == name:
                return n

    def apply_to_node(self, node):
        for k, v in self.kwargs.items():
            if k == "args":
                new_args = list(v)
                for i, v_tmp in enumerate(v):
                    if v_tmp == ChangeNodeTokens.copy:
                        assert i < len(node.args), "tried to copy argument #%d, but got %d arguments in original node"%(i, len(node.args))
                        new_args[i] = node.args[i]
                    elif isinstance(v_tmp, ChangeNodeActivationPointer):
                        target_activation = self.retrieve_activation_from_graph(node.graph, v_tmp.name)
                        if target_activation is None:
                            raise BendingCallbackException('node %s not found in graph.'%v_tmp.name)
                        new_args[i] = target_activation
                v = new_args
            elif k == "kwargs":
                new_kwargs = dict(v)
                for k_tmp, v_tmp in v.items():
                    if v_tmp == ChangeNodeTokens.copy:
                        assert k in node.kwargs, "tried to copy key %s, but absent from original nodes kwargs."%k_tmp
                        new_kwargs[k_tmp] = node.kwargs[k_tmp]
                    elif isinstance(v_tmp, ChangeNodeActivationPointer):
                        target_activation = self.retrieve_activation_from_graph(node.graph, v_tmp.name)
                        if target_activation is None:
                            raise BendingCallbackException('node %s not found in graph.'%v_tmp.name)
                        new_kwargs[k_tmp] = target_activation
                v = new_kwargs
            setattr(node, k, v)
        return node

