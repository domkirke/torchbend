import os, enum
import torch
import typing as tp
import inspect
import ast
from .callback import BendingCallback, BendingCallbackException
from ..tracing.proxy import BendingProxy


class CopyArg():
    def __init__(self):
        """adding empty method for TorchScript"""
        pass

class ChangeNodeTokens(enum.Enum):
    copy = 0

class ChangeNodeActivationPointer(object):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return "ChangeNodeActivationPointer(name=\"%s\")"%self.name

def get_atoms(node):
    if isinstance(node, ast.BinOp):
        return get_atoms(node.left) + get_atoms(node.right)
    elif isinstance(node, ast.UnaryOp):
        return get_atoms(node.operand)
    elif isinstance(node, ast.Call):
        atoms = get_atoms(node.func)
        for arg in node.args:
            atoms.extend(get_atoms(arg))
        return atoms
    elif isinstance(node, ast.Name):
        # Variable name
        return [node.id]
    elif isinstance(node, ast.Constant):
        return []
    elif isinstance(node, ast.Attribute):
        # Attribute of an object
        return get_atoms(node.value)# + [node.attr]
    elif isinstance(node, ast.Subscript):
        # Subscript operation (e.g., indexing)
        return get_atoms(node.value) + get_atoms(node.slice)
    elif isinstance(node, ast.List) or isinstance(node, ast.Tuple):
        atoms = []
        for elt in node.elts:
            atoms.extend(get_atoms(elt))
        return atoms
    elif isinstance(node, ast.Dict):
        atoms = []
        for key in node.keys:
            atoms.extend(get_atoms(key))
        for value in node.values:
            atoms.extend(get_atoms(value))
        return atoms
    else:
        # For simplicity, ignore other complex nodes
        return []

class ChangeNodeExpressionPointer(object):
    def __init__(self, expression: str, gl_dict: tp.Optional[tp.Dict[str, tp.Any]] = None, lc_dict: tp.Optional[tp.Dict[str, tp.Any]] = None):
        self.expression = expression
        self.globals = gl_dict 
        self.locals = lc_dict
        
    def __repr__(self):
        return "ChangeNodeExpressionPointer(expression=\"%s\")"%self.expression

    def _get_nodes_from_expression(self, graph):
        tree = ast.parse(self.expression, mode="eval")
        expression_atoms = get_atoms(tree.body)
        nodes_to_add = list(filter(lambda x: x.name in expression_atoms, graph.nodes))
        return {n.name : BendingProxy(n) for n in nodes_to_add}

    def _make_execution_scope(self, node):
        target_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tracing", "module.py"))
        target_frame = None
        for frame_info in reversed(inspect.stack()):
            if frame_info.filename == target_file:
                target_frame = frame_info.frame
                break
        if target_frame is None: 
            raise BendingCallbackException('Could not find execution frame for expression %s'%self)
        target_frame = target_frame.f_back
        f_locals = target_frame.f_locals
        f_globals = target_frame.f_globals

        # include nodes to locals
        nodes_to_include = self._get_nodes_from_expression(node.graph)
        for name, node in nodes_to_include.items():
            if name in f_locals:
                #TODO resolve conflict
                raise BendingCallbackException('Conflict : activation name %s already exists in original scope.'%n.name)
            f_locals[name] = node

        return f_locals, f_globals

    def parse_expression(self, node: torch.fx.Node):
        f_locals, f_globals = self._make_execution_scope(node)
        new_proxy = eval(self.expression, f_globals, f_locals)
        # new_node = node.graph.node_copy(new_proxy.node)
        new_node = new_proxy.node
        return new_node

class ChangeNode(BendingCallback):
    """Graph surgery callback: rewrites the targeted fx node itself.

    Instead of transforming the activation value, ChangeNode modifies the
    node's ``op``, ``target``, ``args``, ``kwargs`` or ``name`` when the
    bended graph is built (``applied_to_node=True``)::

        # swap torch.sin for torch.cos, keeping the original argument
        cb = ChangeNode(op="call_function", target=torch.cos,
                        args=(ChangeNode.copy,))
        bended.bend(cb, "sin")

    Helpers for arguments:
        ChangeNode.copy: keep the original node's argument at this position.
        ChangeNode.activation(name): point to another node of the graph.
        ChangeNode.expression(expr): evaluate an expression in graph scope.

    Scalar kwargs are automatically promoted to controllable parameters.
    """
    needs_insertion = False
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
        return ChangeNodeExpressionPointer(expression)

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
                    elif isinstance(v_tmp, ChangeNodeExpressionPointer):
                        expression_node = v_tmp.parse_expression(node)
                        new_args[i] = expression_node
                v = tuple(new_args)
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
                v = dict(new_kwargs)
            setattr(node, k, v)
        return node



"""
1. **Module-level**
   - `visit_Module`: This handles the entire module, essentially an entire file of Python code.
   - `visit_Interactive`: For interactive mode, often not used directly.

2. **Statements**
   - `visit_Expr`: A generic expression statement.
   - `visit_Assign`: A basic assignment `a = b`.
   - `visit_AugAssign`: An augmented assignment `a += b`.
   - `visit_AnnAssign`: An annotated assignment `a: int = 3`.
   - `visit_Return`: A return statement `return x`.
   - `visit_If`: An if statement.
   - `visit_For`: A for loop.
   - `visit_While`: A while loop.
   - `visit_Continue`: A continue statement.
   - `visit_Break`: A break statement.
   - `visit_FunctionDef`: A function definition `def func(...):`.
   - `visit_ClassDef`: A class definition `class MyClass:`.
   - `visit_Import`: An import statement `import xyz`.
   - `visit_ImportFrom`: A from import statement `from abc import xyz`.

3. **Expressions**
   - `visit_BinOp`: A binary operation, like `a + b`.
   - `visit_UnaryOp`: A unary operation, like `-a`.
   - `visit_BoolOp`: A boolean operation, like `a and b`.
   - `visit_Compare`: A comparison, like `a < b`.
   - `visit_Call`: A function or method call.
   - `visit_Attribute`: Accessing an attribute, like `object.attr`.
   - `visit_Subscript`: Handling subscripting or slicing, like `a[1]`.

4. **Literals and Constants**
   - `visit_Constant`: Python 3.8+ node for constants (replaces `visit_Str`, `visit_Num`, etc.).
   - `visit_List`: A list literal, like `[1, 2, 3]`.
   - `visit_Tuple`: A tuple literal, like `(a, b, c)`.
   - `visit_Dict`: A dictionary literal, like `{key: value}`.
   - `visit_Set`: A set literal, like `{1, 2, 3}`.

5. **Comprehensions and Generators**
   - `visit_ListComp`: A list comprehension, like `[x for x in iterable]`.
   - `visit_SetComp`: A set comprehension.
   - `visit_DictComp`: A dictionary comprehension.
   - `visit_GeneratorExp`: A generator expression `(x for x in iterable)`.

6. **Other Constructs**
   - `visit_Lambda`: A lambda expression.
   - `visit_IfExp`: A conditional expression, like `a if condition else b`.
   - `visit_Global`: The global statement `global x, y`.
   - `visit_Nonlocal`: The nonlocal statement `nonlocal x, y`.
   - `visit_Name`: A variable name.
   - `visit_Starred`: Elements in unpacking, like `*a`.
   
"""