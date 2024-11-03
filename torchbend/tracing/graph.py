import re
from collections import OrderedDict
import inspect
import torch
from torch.fx import Graph
from typing import List, Dict
from .tracing import TraceError

def graph_insert_callbacks(graph, callbacks, verbose=False, _fn_name="forward"):
    """inserts bending operation into a graph"""
    new_graph = torch.fx.Graph()
    env = {}
    bended_lookup = {}
    name_hash = {}
    for node in graph.nodes:
        new_node = new_graph.node_copy(node, lambda x: env[x.name])
        # check arguments to replace by bended node in case
        new_args = list(new_node.args)
        for i, arg in enumerate(new_args):
            if isinstance(arg, torch.fx.Node):
                if arg.name in bended_lookup:
                    new_args[i] = bended_lookup[arg.name]
        new_node.args = tuple(new_args)
        env[node.name] = new_node
        #TODO using inserting_after??
        if node.name in callbacks:
            if verbose:
                print('bending activation %s with function %s...'%(node.name, callbacks[node.name]))
            bended_node_name = node.name+"_bended"
            hack_obj_name = node.name + "_callback"
            bended_node = new_graph.create_node("call_module", hack_obj_name, args=(new_node,), kwargs={'name': f"{_fn_name}:{node.name}"}, name=bended_node_name)
            env[bended_node_name] = bended_node 
            bended_lookup[node.name] = bended_node 
    return new_graph

def graph_get_activations(graph: torch.fx.Graph, activations: List[str]):
    out_graph = torch.fx.Graph()
    env = {}
    out_nodes = {}
    for node in list(graph.nodes):
        if node.op != "output":
            args = tuple([n if not isinstance(n, torch.fx.Node) else env[n.name] for n in node.args])
            kwargs = {k: v if not isinstance(v, torch.fx.Node) else env[v.name] for k, v in node.kwargs.items()}
            env[node.name] = out_graph.create_node(node.op, node.target, args, kwargs, name=node.name, type_expr=node.type)
            if node.name in activations:
                out_nodes[node.name] = env[node.name]
        if list(out_nodes.keys()) == activations:
            # consider that all needed operations are copied to the amputed graph
            break
    out_node = out_graph.call_function(dict, kwargs=out_nodes, type_expr=Dict[str, torch.Tensor])
    out_graph.output(out_node)
    return out_graph

def get_single_users(node, out):
    for n in node.args:
        if len(n.users) == 1:
            out.append(n)
            get_single_users(n, out)
    

def graph_from_activations(graph, activations, remove_placeholders=False, parse_inputs_from_callbacks=None):
    new_graph = torch.fx.Graph()
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

    for p in ph_orig: env[p.name] = new_graph.placeholder(p.name, p.type, *p.args)
    for a in activations: env[a] = new_graph.placeholder(a, )

    # parse inputs
    additional_inputs = {}
    for n in graph.nodes:
        if n.name.endswith('_bended') and n.name.replace('_bended', '') in activations and parse_inputs_from_callbacks:
            bended_activation_name = n.name.replace('_bended', '')
            cb = parse_inputs_from_callbacks[bended_activation_name]
            signature = inspect.signature(cb.forward)
            # args = tuple()
            # new_kwargs = OrderedDict()
            kwargs = {k: v if not isinstance(v, torch.fx.Node) else  env[v.name] for k, v in n.kwargs.items()}
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
            args = tuple([n if not isinstance(n, torch.fx.Node) else env[n.name] for n in n.args])
            kwargs = {k: v if not isinstance(v, torch.fx.Node) else  env[v.name] for k, v in n.kwargs.items()}
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
    return new_graph


# def graph_split(graph: torch.fx.Graph, *activations):
#     """
#     graph_split returns two graphs : 
#     - one with outputs corresponding to activations, 
#     - one with inputs corresponding to activations + other needed inputs.
#     """
#     graph_before = graph_get_activations(graph, activations)
#     graph_after = graph_from_activations(graph, activations)
#     return graph_before, graph_after



    # assert activation in [n.name for n in graph.nodes], f"{activation} not present in given graph"
    # graph_before = torch.fx.Graph()
    # graph_after = torch.fx.Graph()
    # env_before = {}
    # env_after = {}

    # current_graph = graph_before
    # current_env = env_before
    # for node in list(graph.nodes):
    #     current_env[node.name] = current_graph.node_copy(node, lambda n, e = current_env: e[n.name])
    #     if node.name == activation:
    #         # create dict output 
    #         current_graph.output(current_env[node.name])
    #         current_graph = graph_after
    #         current_env = env_after
    #         current_env[node.name] = current_graph.placeholder(node.name)#, type_expr = node.type_expr)
    # return graph_before, graph_after
