import re
from collections import OrderedDict
import inspect
import torch
from torch.fx import Graph, Node
from torch.fx._compatibility import compatibility
# from torch.fx.graph_module import GraphModule
from typing import List, Dict, Any, Optional, Type
from .tracing import BendedGraph, TraceError, ActivationProperties


# _GRAPH_COPY_ATTR = ['activations', 'aliases']
# def _import_attr_from_original_graph(graph, new_graph):
#     for attr in _GRAPH_COPY_ATTR:
#         setattr(new_graph, attr, getattr(graph, attr, None))


def graph_transform_nodes(graph, callbacks, verbose=False):
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
        

def graph_insert_callbacks(graph, callbacks, verbose=False, fn=None):
    
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
            

    """inserts bending operation into a graph"""
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
    return out_graph

def get_single_users(node, out):
    for n in node.args:
        if not isinstance(n, torch.fx.Node): continue
        if len(n.users) == 1:
            out.append(n)
            get_single_users(n, out)

def graph_from_activations(graph, activations, remove_placeholders=True, parse_inputs_from_callbacks=None):
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

