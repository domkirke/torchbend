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

    def script(self):
        return self

@pytest.mark.parametrize("activation,cb_args", [
    ("linear1", {'target': 'linear2'}),
    ("pow_1", {'target': globals()['__builtins__']['abs'], 'args': (tb.ChangeNode.copy,)}),
    ("sigmoid", {'target': 'linear2', 'op': 'call_module', 'name': 'linear2'}),
    ("sigmoid", {'target': torch.split, 'op': 'call_function', 'args': (tb.ChangeNode.copy, 1), 'kwargs': {'dim': 0}, 'name': 'split'}),
    ("mul", {'target': torch.mul, 'op': 'call_function', 'args': (tb.ChangeNode.copy, tb.ChangeNode.activation("pow_1"))}),
    ("pow_1", {'target': torch.mul, 'op': 'call_function', 'args': (tb.ChangeNode.expression("linear1 ** 2 - linear1.mean()"), 1)})
])
@pytest.mark.parametrize("jit", [True, False])
def test_node_change_target(activation, cb_args, jit):
    module = NodeEffectsTester()
    x = torch.randn(16, 10)

    out = module(x)
    bended = tb.BendedModule(module)
    bended.trace(x=x)

    cb = tb.ChangeNode(**cb_args)
    bended.bend(cb, activation)
    out_bended = bended(x)

    if torch.is_tensor(out_bended):
        assert not torch.allclose(out, out_bended)

    scripted = bended.script(script=jit)
    out_scripted = scripted(x)
    if torch.is_tensor(out_bended):
        assert not torch.allclose(out, out_scripted)
        assert torch.allclose(out_bended, out_scripted)

    
