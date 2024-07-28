import pytest, os
import torch, torch.nn as nn
import torchbend as tb

outdir = os.path.join(os.path.dirname(__file__), "outs")
os.makedirs(outdir, exist_ok=True)

class ReshapeFoo(nn.Module):
    test_reshape_inputs = {'x': torch.zeros(3, 5, 7)}
    test_control_inputs = {'x': torch.zeros(3, 5, 7)}

    def test_reshape(self, x):
        x = x.reshape(*x.shape[:-1], x.shape[-1])
        return x

    def test_control(self, x):
        if (len(x.shape) > 2):
            return torch.tensor(1)
        else:
            return torch.tensor(0)


class ControlFlowFoo(nn.Module):
    def forward(self, x):
        if (x == 0).all():
            return torch.tensor(0)
        else:
            return torch.tensor(1)


# def test_control_flow():
#     foo = ControlFlowFoo()
#     assert foo 

@pytest.mark.parametrize("module,method", [(ReshapeFoo, 'test_reshape'),
                                           (ReshapeFoo, 'test_control')])
def test_dynamic_shape(module, method):
    test_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    module = tb.BendedModule(module())
    # graph, out = module.trace(method, **getattr(module, f'{method}_inputs'), _return_out=True)
    module.trace(method, **getattr(module, f'{method}_inputs'), _return_out=True)
    # with open(os.path.join(outdir, test_name+"_out.txt"), 'w+') as f:
    #     f.write(out)
