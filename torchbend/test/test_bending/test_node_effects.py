import os, sys
import torch, torch.nn as nn
import pytest
import torchbend as tb


testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)
from test_modules import modules_to_test, ModuleTestConfig

def pow_2_whatever(x):
    return x ** 2

class NodeEffectsTester(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10, bias=False)
        
    def forward(self, x):
        out = self.linear1(x)
        out = pow_2_whatever(out)
        out = out * out.shape[-1]
        out = torch.sigmoid(out)
        return out

@pytest.mark.parametrize("activation,cb_args", [
    ("linear1", {'target': 'linear2'}),
    ("pow_1", {'target': globals()['__builtins__']['abs'], 'args': (tb.ChangeNode.copy,)}),
    ("sigmoid", {'target': 'linear2', 'op': 'call_module', 'name': 'linear2'}),
    ("sigmoid", {'target': torch.split, 'op': 'call_function', 'args': (tb.ChangeNode.copy, 1), 'kwargs': {'dim': 0}, 'name': 'split'}),
    ("mul", {'target': torch.mul, 'op': 'call_function', 'args': (tb.ChangeNode.copy, tb.ChangeNode.activation("pow_1"))})
])
def test_node_change_target(activation, cb_args):
    module = NodeEffectsTester()
    x = torch.randn(16, 10)

    out = module(x)
    bended = tb.BendedModule(module)
    bended.trace(x=x)

    cb = tb.ChangeNode(**cb_args)
    bended.bend(cb, activation)
    #TODO expression evaluation from activation, like "pow_1.shape"
    out_bended = bended(x)

    if torch.is_tensor(out_bended):
        assert not torch.allclose(out, out_bended)
    


