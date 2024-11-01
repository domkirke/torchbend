import pytest, os
from typing import List
import torch, torch.nn as nn
import torchbend as tb
from test_modules import trace_test_modules as ttm

outdir = os.path.join(os.path.dirname(__file__), "outs")
os.makedirs(outdir, exist_ok=True)

def get_log_file():
    test_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    return os.path.join(outdir, test_name+"_out.txt")

def log_to_file(f, label, value):
    f.write(f"{label} : \n{value}\n\n{'-' * 16}")
    

@pytest.mark.parametrize("module,method", ttm.ReshapeFoo.tests())
def test_dynamic_shape(module, method):
    module = tb.BendedModule(module())
    graph, out = module.trace(method, **getattr(module, f'{method}_inputs'), _return_out=True)
    with open(get_log_file(), 'w+') as f:
        log_to_file(f, "out", out)
        log_to_file(f, "graph", graph)
    return True

@pytest.mark.parametrize("module,method", ttm.LogicalFlowFoo.tests())
def test_logical_flow(module, method):
    module = tb.BendedModule(module())
    inputs = getattr(module, f'{method}_inputs')
    with open(get_log_file(), 'w+') as f:
        for inp in inputs:
            graph, out = module.trace(method, **inp, _return_out=True)
            log_to_file(f, "input", inp)
            log_to_file(f, "out", out)
            log_to_file(f, "graph", graph)
            log_to_file(f, "flow", graph.flow_steps)
    return True


@pytest.mark.parametrize("module,method", ttm.LoopFoo.tests())
def test_loop(module, method):
    module = tb.BendedModule(module())
    inputs = getattr(module, f'{method}_inputs')
    with open(get_log_file(), 'w+') as f:
        for inp in inputs:
            graph, out = module.trace(method, **inp, _return_out=True)
            log_to_file(f, "input", inp)
            log_to_file(f, "out", out)
            log_to_file(f, "graph", graph)
            log_to_file(f, "flow", graph.flow_steps)
    return True

@pytest.mark.parametrize("module", ttm.split_graph_test_modules)
def test_graph_split(module):
    module = tb.BendedModule(module)
    kwargs = module.forward_inputs
    module.trace(**kwargs)
    for t in module.forward_targets:
        out_mid = module.get_activations(t, **kwargs)
        out, graph = module.from_activations(t, **kwargs, **out_mid, _return_graph=True)


