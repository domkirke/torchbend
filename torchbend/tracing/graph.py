import re
from collections import OrderedDict
import inspect
import torch
from torch.fx import Graph, Node
from torch.fx._compatibility import compatibility
# from torch.fx.graph_module import GraphModule
from typing import List, Dict, Any, Optional, Type
from .tracing import BendedGraph, TraceError, ActivationProperties
from . import activation_log as actlog


# _GRAPH_COPY_ATTR = ['activations', 'aliases']
# def _import_attr_from_original_graph(graph, new_graph):
#     for attr in _GRAPH_COPY_ATTR:
#         setattr(new_graph, attr, getattr(graph, attr, None))


def graph_transform_nodes(graph, callbacks, verbose=False):
    """Apply node-rewriting callbacks (``applied_to_node=True``, e.g. ChangeNode).

    Copies ``graph`` and lets each callback rewrite its target node in place
    (op / target / args / kwargs) via ``apply_to_node``. The result is linted;
    a ``TraceError`` is raised when the rewritten graph is inconsistent.
    """
    new_graph = BendedGraph(from_graph=graph)
    env = {}
    # then insert
    for node in graph.nodes:
        # check arguments to replace by bended node in case
        new_node = new_graph.node_copy(node, lambda x: env[x.name])
        env[node.name] = new_node
        if node.name in callbacks and callbacks[node.name].applied_to_node:
            if verbose:
                print('bending activation %s with function %s...'%(node.name, callbacks[node.name]))
            with new_graph.inserting_before(new_node):
                callbacks[node.name].apply_to_node(new_node)
    try:
        new_graph.lint()
    except RuntimeError as e:
        raise TraceError("Lint failed after node transformation. Caught error : %s"%(e))
    # _import_attr_from_original_graph(graph, new_graph)
    return new_graph



def _get_additional_inputs_from_cb(cb):
    add_inputs = []
    param_shapes = {v.name: v.value.shape for k, v in cb.input_controllables().items()}
    add_shapes = []
    for name, param in dict(inspect.signature(cb.forward).parameters).items():
        if name in cb.native_callback_arguments: continue
        add_inputs.append(name)
        if name in param_shapes:
            add_shapes.append(param_shapes[name])
        else:
            add_shapes.append(param_shapes["_".join(name.split('_')[:-1])])
    return add_inputs, add_shapes
        

def graph_insert_callbacks(graph, callbacks, verbose=False, fn=None, make_bending_placeholders=False):
    """Insert activation-bending callbacks into a copy of ``graph``.

    For every node present in ``callbacks`` (``{node_name: CallbackChain}``):

    1. a ``call_module`` node named ``<node>_bended`` is appended, targeting a
       submodule ``<fn>_<node>_callback`` that the executing GraphModule owns;
    2. every downstream consumer of the original node is rerouted to the
       bended node;
    3. controllables declared ``as_input=True`` get extra placeholder inputs
       (this is how nn~ exports expose bending controls as inlets).

    Returns the new ``BendedGraph``; the input graph is left untouched.
    """
    def node_name_from(input_dict, node_name):
        # if re.match(r'.*\_(\d+)$', node_name):
            # get_node_name = (lambda x: "_".join(node_name.split('_')[:-1]))
        #     name = get_node_name(node_name)
        #     n_occur = len(list(filter(lambda x: x == name, map(get_node_name, input_dict.keys())))) - 1
        #     return f"{name}_{n_occur}"
        # else:
        #     name = node_name
        #     n_occur = 0
        #     return name
        return 
    
    def _get_node_name(node_name):
        if re.match(r'.*\_(\d+)$', node_name):
            node_name = "_".join(node_name.split('_')[:-1])
        return node_name

    def _update_bending_ph_names(bended, original):
        ph_hash = {}
        for act_name, bending_list in bended.items(): 
            for b in bending_list:
                b_name = _get_node_name(b)
                if b_name not in ph_hash:
                    ph_hash[b_name].append(act_name)

    def _parse_placeholder_names(graph):
        placeholders = list(filter(lambda x: x.op == "placeholder", graph.nodes))
        placeholders_names = [n.name for n in placeholders]
        placeholder_count = {}
        for p in placeholders_names:
            root_name = _get_node_name(p)
            if root_name in placeholder_count:
                placeholder_count[root_name] += 1
            else:
                placeholder_count[root_name] = 1
        ph_with_shared_names = list(filter(lambda k: placeholder_count[k] != 1, placeholder_count))
        for p_name in ph_with_shared_names:
            current_count = 0
            for p in placeholders:
                if _get_node_name(p.name) == p_name:
                    p.name = f"{_get_node_name(p.name)}_{current_count}"
                    p.target = f"{_get_node_name(p.name)}_{current_count}"
                    current_count += 1
            

    new_graph = BendedGraph(from_graph=graph)
    env = {}
    bended_lookup = {}
    fn_name = fn or graph.fn

    def _replace_with_bended(arg, bended_lookup):
        if isinstance(arg, torch.fx.Node):
            if arg.name in bended_lookup:
                return bended_lookup[arg.name]
            else:
                return arg
        elif isinstance(arg, (tuple, list)):
            return type(arg)([_replace_with_bended(a, bended_lookup) for a in arg])
        elif isinstance(arg, dict):
            return {k: _replace_with_bended(v, bended_lookup) for k, v in arg.items()}
        else:
            return arg
        

    # then insert
    last_input = None
    for node in graph.nodes:
        new_node = new_graph.node_copy(node, lambda x: env[x.name])
        if node.op == "placeholder":
            last_input = new_node
        # check arguments to replace by bended node in case
        new_node.args = _replace_with_bended(new_node.args, bended_lookup)
        env[node.name] = new_node
        if node.name in callbacks:
            if verbose:
                print('bending activation %s with function %s...'%(node.name, callbacks[node.name]))
            if callbacks[node.name].needs_insertion:
                add_inputs, add_shapes = _get_additional_inputs_from_cb(callbacks[node.name])
                callback_kwargs = {'name': f"{fn_name}:{node.name}"}
                for i, k in enumerate(add_inputs):
                    # node_name = node_name_from(inputs, k)
                    with new_graph.inserting_after(last_input):
                        additional_node = new_graph.create_node("placeholder", k, (None,), type_expr = Optional[torch.Tensor])
                        additional_node.meta['shape_from_controllable'] = add_shapes[i]
                        additional_node.meta['from_callback'] = callbacks[node.name]
                        callback_kwargs[k] = additional_node
                        last_input = additional_node
                bended_node_name = node.name+"_bended"
                hack_obj_name = f"{fn_name}_{node.name}_callback"
                bended_node = new_graph.create_node("call_module", hack_obj_name, args=(env[node.name],), kwargs=callback_kwargs, name=bended_node_name)
                env[bended_node_name] = bended_node
                bended_lookup[node.name] = bended_node
                #TODO how could we get shape? 
                new_graph.activations[bended_node_name] = ActivationProperties(op=node.op, 
                                                            target=node.target, 
                                                            type=node.type, 
                                                            name=node.name, 
                                                            args=node.args,
                                                            kwargs=node.kwargs,
                                                            fn=fn)
    # _import_attr_from_original_graph(graph, new_graph)
    # regularize placeholder names
    _parse_placeholder_names(new_graph)
    return new_graph


def _get_new_node_args(env, node):
    if isinstance(node, (list, tuple)):
        return type(node)([_get_new_node_args(env, n) for n in node])
    elif isinstance(node, torch.fx.Node):
        return env[node.name]
    elif isinstance(node, slice):
        return slice(_get_new_node_args(env, node.start),
                     _get_new_node_args(env, node.stop), 
                     _get_new_node_args(env, node.step))
    else:
        return node

def graph_get_activations(graph: BendedGraph, activations: List[str]):
    """Return a copy of ``graph`` truncated at the given activations.

    The output node is replaced by a tuple of the requested nodes; everything
    downstream of them is dropped, so running the graph computes only what is
    needed. Used by ``BendedModule.get_activations``.
    """
    _t0 = actlog.tick()
    out_graph = BendedGraph(from_graph=graph)
    env = OrderedDict()
    out_nodes = {}
    for node in list(graph.nodes):
        if node.op != "output":
            args = tuple([_get_new_node_args(env, n) for n in node.args])
            kwargs = {k: _get_new_node_args(env, v) for k, v in node.kwargs.items()}
            env[node.name] = out_graph.create_node(node.op, node.target, args, kwargs, name=node.name, type_expr=node.type)
            if node.name in activations:
                out_nodes[node.name] = env[node.name]
        if list(out_nodes.keys()) == activations:
            # consider that all needed operations are copied to the amputed graph
            break
    out_nodes = tuple(out_nodes[a] for a in activations)
    # out_node = out_graph.call_function(dict, kwargs=out_nodes, type_expr=Dict[str, torch.Tensor])
    if len(out_nodes) == 1:
        out_graph.output(out_nodes[0])
    else:
        out_graph.output(out_nodes)

    # filter unused nodes 
    for k, n in reversed(env.items()):
        if len(n.users) == 0: 
            out_graph.erase_node(n)

    if graph.activations is not None:
        out_graph.activations = {k: graph.activations.get(k) for k in env.keys()}
    actlog.log("graph_get_activations   targets=%s  %s  %s",
               actlog.fmt_names(activations),
               actlog.fmt_graph_delta(graph, out_graph), actlog.tock(_t0))
    return out_graph

def get_single_users(node, out):
    for n in node.args:
        if not isinstance(n, torch.fx.Node): continue
        if len(n.users) == 1:
            out.append(n)
            get_single_users(n, out)

def graph_from_activations(graph, activations, remove_placeholders=True, parse_inputs_from_callbacks=None):
    """Return a copy of ``graph`` re-rooted at the given activations.

    The requested activations become placeholder (input) nodes and every node
    that only fed them is dropped (with ``remove_placeholders=True``, original
    inputs that became unused are removed too). ``parse_inputs_from_callbacks``
    maps bended activation names to their CallbackChain so callback inputs
    survive the cut. Used by ``BendedModule.from_activations``.
    """
    _t0 = actlog.tick()
    new_graph = BendedGraph(from_graph=graph)
    env = {}
    node_act = list(filter(lambda x: x.name in activations, graph.nodes))
    nodes_to_remove = []
    for n in node_act:
        single_users = []
        get_single_users(n, single_users)
        single_users = list(filter(lambda x: x.op != "placeholder" or remove_placeholders, single_users))
        nodes_to_remove.extend(single_users)
    nodes_to_remove = [n.name for n in nodes_to_remove]
    conflicting_nodes = set(activations).intersection(set(nodes_to_remove))
    if len(conflicting_nodes) > 0:
        raise TraceError('conflicting nodes found : %s. Could not parse new graph'%list(conflicting_nodes))

    # add placeholders
    ph_orig = list(filter(lambda x: x.op == "placeholder" and x.name not in nodes_to_remove, graph.nodes))
    # organize placeholders well
    nondefault_placeholders = list(filter(lambda x: len(x.args) == 0, ph_orig))
    default_placeholders = list(filter(lambda x: len(x.args) > 0, ph_orig))
    for p in nondefault_placeholders: env[p.name] = new_graph.placeholder(p.name, p.type)
    for a in activations: env[a] = new_graph.placeholder(a, torch.Tensor) 
    for p in default_placeholders: env[p.name] = new_graph.placeholder(p.name, p.type, *p.args)

    # parse inputs
    additional_inputs = {}
    for n in graph.nodes:
        if n.name.endswith('_bended') and n.name.replace('_bended', '') in activations and parse_inputs_from_callbacks:
            bended_activation_name = n.name.replace('_bended', '')
            cb = parse_inputs_from_callbacks[bended_activation_name]
            signature = inspect.signature(cb.forward)
            # args = tuple()
            # new_kwargs = OrderedDict()
            kwargs = {k: _get_new_node_args(env, v) for k, v in n.kwargs.items()}
            for name, param in dict(signature.parameters).items():
                # bypass current positional arguments
                if name == "x": pass
                elif name in kwargs: pass
                else:
                    if not name in additional_inputs: additional_inputs[name] = []
                    additional_inputs[name].append((bended_activation_name, param))
            

    placeholder_map = {}
    for k, v in additional_inputs.items():
        if len(v) > 1:
            for bended_name, param in v: placeholder_map[f"{bended_name}_{k}"] = param
        else:
            placeholder_map[k] = v[0][1]
    # create additional placeholders
    for k, v in placeholder_map.items():
        env[k] = new_graph.placeholder(k, v.annotation, v.default)
        
    # copy graph
    for n in graph.nodes:
        if n.op == "placeholder": continue
        if n.name not in nodes_to_remove and n.name not in activations:
            args = tuple([_get_new_node_args(env, n) for n in n.args])
            kwargs = {k: _get_new_node_args(env, v) for k, v in n.kwargs.items()}
            if n.name.endswith('_bended') and n.name.replace('_bended', '') in activations and parse_inputs_from_callbacks:
                bended_activation_name = n.name.replace('_bended', '')
                cb = parse_inputs_from_callbacks[bended_activation_name]
                signature = inspect.signature(cb.forward)
                args = tuple()
                new_kwargs = OrderedDict()
                for name, param in dict(signature.parameters).items():
                    # bypass current positional arguments
                    if name == "x":
                        new_kwargs[name] = env[bended_activation_name]
                    elif name in kwargs: 
                        new_kwargs[name] = kwargs[name]
                    else:
                        if f"{bended_activation_name}_{name}" in env:
                            new_kwargs[name] = env[f"{bended_activation_name}_{name}"]
                        else:
                            new_kwargs[name] = env[name]
                kwargs = new_kwargs
            env[n.name] = new_graph.create_node(n.op, n.target, args, kwargs, name = n.name)

    if graph.activations is not None:
        new_graph.activations = {k: graph.activations.get(k) for k in env.keys()}
    new_graph.__from_op = "from_activations"

    actlog.log("graph_from_activations  roots=%s  %s  %s  (dropped %d upstream node(s)%s)",
               actlog.fmt_names(activations),
               actlog.fmt_graph_delta(graph, new_graph), actlog.tock(_t0),
               len(nodes_to_remove),
               ", +%d callback input(s)" % len(placeholder_map) if placeholder_map else "")
    return new_graph


def _callback_placeholders(nodes, activations, callbacks):
    """Map the callback arguments no node provides to graph inputs.

    Returns an ordered ``{placeholder_name: inspect.Parameter}`` for the
    ``<activation>_bended`` nodes among ``nodes``; an argument name claimed by
    several bended activations is prefixed by the activation it belongs to.
    """
    additional_inputs = OrderedDict()
    for n in nodes:
        if not n.name.endswith('_bended'): continue
        bended_activation_name = n.name.replace('_bended', '')
        if bended_activation_name not in activations: continue
        signature = inspect.signature(callbacks[bended_activation_name].forward)
        for name, param in dict(signature.parameters).items():
            # bypass current positional arguments
            if name == "x": continue
            if name in n.kwargs: continue
            additional_inputs.setdefault(name, []).append((bended_activation_name, param))
    placeholder_map = OrderedDict()
    for k, v in additional_inputs.items():
        if len(v) > 1:
            for bended_name, param in v: placeholder_map[f"{bended_name}_{k}"] = param
        else:
            placeholder_map[k] = v[0][1]
    return placeholder_map


def _callback_node_kwargs(node, env, bended_activation_name, callback):
    """Rewire a ``<activation>_bended`` node whose input became a placeholder."""
    kwargs = {k: _get_new_node_args(env, v) for k, v in node.kwargs.items()}
    new_kwargs = OrderedDict()
    for name, param in dict(inspect.signature(callback.forward).parameters).items():
        if name == "x":
            new_kwargs[name] = env[bended_activation_name]
        elif name in kwargs:
            new_kwargs[name] = kwargs[name]
        elif f"{bended_activation_name}_{name}" in env:
            new_kwargs[name] = env[f"{bended_activation_name}_{name}"]
        else:
            new_kwargs[name] = env[name]
    return new_kwargs


def graph_subset(graph: BendedGraph,
                 inputs: List[str],
                 outputs: List[str],
                 remove_placeholders: bool = True,
                 parse_inputs_from_callbacks: Optional[Dict[str, Any]] = None):
    """Return the minimal graph computing ``outputs`` out of ``inputs``.

    Same result as re-rooting the graph at ``inputs`` with
    ``graph_from_activations`` then truncating it at ``outputs`` with
    ``graph_get_activations``, but in a single pass : the graph is walked
    backwards from ``outputs``, stopping at ``inputs``, and only the nodes met
    on the way are copied. Nothing upstream of ``inputs``, nothing downstream
    of ``outputs``, and no side branch neither of them needs is ever built,
    instead of being copied then pruned.

    ``inputs`` become placeholders, so they are read as the values they hold
    *before* being bended : a bended input still goes through its callback.

    Args:
        graph (BendedGraph): graph to slice, typically obtained from
            ``BendedModule.bend_graph``.
        inputs (List[str]): activation names to feed the graph with. Kept in
            the signature even when ``outputs`` turn out not to need them.
            Empty means keeping the original placeholders.
        outputs (List[str]): activation names to return, in output order.
        remove_placeholders (bool): drop the original placeholders that no kept
            node depends on. ``False`` keeps the original signature intact.
        parse_inputs_from_callbacks (dict | None): maps bended activation names
            to their ``CallbackChain``, as in ``graph_from_activations``.

    Returns:
        BendedGraph: outputs a single value if one output is given, else a
        tuple ordered as ``outputs``. An output that does not depend on any
        input is computed from the original placeholders.

    Raises:
        TraceError: if no output is given, a name is not in the graph or is its
            output node, or the resulting graph does not lint.
    """
    _t0 = actlog.tick()
    inputs, outputs = list(inputs), list(outputs)
    if len(outputs) == 0:
        raise TraceError("graph_subset needs at least one output activation")
    node_hash = {n.name: n for n in graph.nodes}
    unknown_nodes = list(filter(lambda x: x not in node_hash, inputs + outputs))
    if len(unknown_nodes) > 0:
        raise TraceError('activations not found in graph : %s'%unknown_nodes)
    output_nodes = list(filter(lambda x: node_hash[x].op == "output", inputs + outputs))
    if len(output_nodes) > 0:
        raise TraceError('%s is the output node of the graph, and cannot be used as an activation. Use graph_from_activations to keep the original outputs.'%output_nodes[0])

    # walk backwards from outputs, cutting at inputs : this is the only set of
    # nodes the subset needs.
    needed = set()
    stack = [node_hash[o] for o in outputs]
    while len(stack) > 0:
        node = stack.pop()
        if node.name in needed: continue
        needed.add(node.name)
        if node.name in inputs: continue
        stack.extend([n for n in node.all_input_nodes if n.name not in needed])

    # inputs that are already placeholders need no re-rooting
    activation_inputs = list(filter(lambda x: node_hash[x].op != "placeholder", inputs))
    kept_placeholders = list(filter(lambda x: x.op == "placeholder" and (x.name in needed or x.name in inputs or not remove_placeholders), graph.nodes))
    # placeholders without default must come first, new inputs having none
    nondefault_placeholders = list(filter(lambda x: len(x.args) == 0, kept_placeholders))
    default_placeholders = list(filter(lambda x: len(x.args) > 0, kept_placeholders))

    new_graph = BendedGraph(from_graph=graph)
    env = OrderedDict()
    for p in nondefault_placeholders: env[p.name] = new_graph.node_copy(p)
    for a in activation_inputs: env[a] = new_graph.placeholder(a, torch.Tensor)
    for p in default_placeholders: env[p.name] = new_graph.node_copy(p)
    if parse_inputs_from_callbacks:
        kept_nodes = filter(lambda x: x.name in needed, graph.nodes)
        for k, v in _callback_placeholders(kept_nodes, activation_inputs, parse_inputs_from_callbacks).items():
            env[k] = new_graph.placeholder(k, v.annotation, v.default)

    # copy kept nodes, in original (topological) order
    for node in graph.nodes:
        if node.op == "placeholder": continue
        if node.name not in needed or node.name in activation_inputs: continue
        bended_activation_name = node.name.replace('_bended', '') if node.name.endswith('_bended') else None
        if parse_inputs_from_callbacks and bended_activation_name in activation_inputs:
            kwargs = _callback_node_kwargs(node, env, bended_activation_name, parse_inputs_from_callbacks[bended_activation_name])
            env[node.name] = new_graph.create_node(node.op, node.target, tuple(), kwargs, name=node.name)
        else:
            env[node.name] = new_graph.node_copy(node, lambda x: env[x.name])

    out_nodes = tuple(env[o] for o in outputs)
    if len(out_nodes) == 1:
        new_graph.output(out_nodes[0])
    else:
        new_graph.output(out_nodes)

    if graph.activations is not None:
        new_graph.activations = {k: graph.activations.get(k) for k in env.keys()}
    try:
        new_graph.lint()
    except RuntimeError as e:
        raise TraceError("Lint failed after graph subset. Caught error : %s"%(e))
    actlog.log("graph_subset            roots=%s → targets=%s  %s  %s  (kept %d of %d node(s))",
               actlog.fmt_names(inputs), actlog.fmt_names(outputs),
               actlog.fmt_graph_delta(graph, new_graph), actlog.tock(_t0),
               len(needed), sum(1 for _ in graph.nodes))
    return new_graph


def stitch_return_node(*args):
    if len(args) > 1:
        return args
    else:
        return args[0]


def stitch_graph(target_graph, stitched_graph, input_nodes, out_node, env):
    """stitches a target graph with another graph, representing a target op"""
    placeholder_map = list(input_nodes[0]) + list(input_nodes[1].values())
    new_env = {"arg%d_1"%i: placeholder_map[i] for i in range(len(placeholder_map))}
    output_node = None
    for n in stitched_graph.nodes:
        if n.op == "placeholder": continue
        elif n.op == "output": 
            if isinstance(n.args[0], tuple):
                output_node = target_graph.create_node("call_function", stitch_return_node, (tuple(new_env[i.name] for i in n.args[0]),))
            else:
                output_node = target_graph.create_node("call_function", stitch_return_node, tuple(new_env[i.name] for i in n.args[0]))
            env[out_node.name] = output_node
        else:
            new_node = target_graph.node_copy(n, lambda x: new_env[x.name])
            new_env[new_node.name] = new_node
            env[new_node.name] = new_node
    return target_graph, env

