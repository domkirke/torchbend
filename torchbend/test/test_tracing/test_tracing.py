import pytest, os
import torch, torch.nn as nn
import torchbend as tb

outdir = os.path.join(os.path.dirname(__file__), "outs")
os.makedirs(outdir, exist_ok=True)

class ReshapeFoo(nn.Module):
    test_reshape_inputs = {'x': torch.zeros(3, 5, 7)}
    test_control_inputs = {'x': torch.zeros(3, 5, 7)}
    
    @staticmethod
    def tests():
        return [(ReshapeFoo, 'test_reshape'),
                (ReshapeFoo, 'test_control')]

    def test_reshape(self, x):
        x = x.reshape(*x.shape[:-1], x.shape[-1])
        return x

    def test_control(self, x):
        if (len(x.shape) > 2):
            return torch.tensor(1)
        else:
            return torch.tensor(0)


class LogicalFlowFoo(nn.Module):
    test_logical_tensor_if_inputs = [{'x': torch.zeros(3)}, {'x': torch.ones(3)}]
    test_logical_int_if_inputs = [{'x': 0}, {'x': 1010}]

    @staticmethod
    def tests():
        return [(LogicalFlowFoo, 'test_logical_tensor_if'), 
                (LogicalFlowFoo, 'test_logical_int_if')]

    def test_logical_tensor_if(self, x: torch.Tensor):
        tensor = (x == 0)
        if tensor.all():
            return torch.tensor(0)
        else:
            return torch.tensor(1)

    def test_logical_int_if(self, x: int):
        if x == 0:
            return 0
        else:
            return 1


def get_log_file():
    test_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    return os.path.join(outdir, test_name+"_out.txt")


def log_to_file(f, label, value):
    f.write(f"{label} : \n{value}\n\n{'-' * 16}")
    

@pytest.mark.parametrize("module,method", ReshapeFoo.tests())
def test_dynamic_shape(module, method):
    module = tb.BendedModule(module())
    graph, out = module.trace(method, **getattr(module, f'{method}_inputs'), _return_out=True)
    with open(get_log_file(), 'w+') as f:
        log_to_file(f, "out", out)
        log_to_file(f, "graph", graph)
    return True

@pytest.mark.parametrize("module,method", LogicalFlowFoo.tests())
def test_logical_flow(module, method):
    module = tb.BendedModule(module())
    inputs = getattr(module, f'{method}_inputs')
    with open(get_log_file(), 'w+') as f:
        for inp in inputs:
            graph, out = module.trace(method, **inp, _return_out=True)
            log_to_file(f, "input", inp)
            log_to_file(f, "out", out)
            log_to_file(f, "graph", graph)
            log_to_file(f, "logic", graph.logical_steps)
    return True

