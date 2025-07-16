import sys, os, pytest
import torch
import torch.nn as nn
testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)
from test_modules import modules_to_test, ModuleTestConfig 
from torchbend import BendedModule, mark




@pytest.mark.parametrize("test_mode", ["function", "module"])
def test_mark_tracing(test_mode):

    if test_mode == "function":
        @mark(name="nn_lin", mode="post")
        def nn_lin(x):
            return x + x.abs()

        @mark(name="nn_sum", mode="pre")
        def nn_sum(x): 
            return (x.sum() - x.abs().sum()).cos()

        class MarkFoo(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 30)
                self.linear3 = nn.Linear(30, 40)
                self.linear4 = nn.Linear(40, 50)

            def forward(self, x):
                out1 = mark(self.linear1(x))
                out2 = mark(self.linear2(out1))
                out3 = mark(self.linear3(out2))
                out_nn = nn_lin(out3)
                out4 = mark(self.linear4(out_nn), mode="pre")
                out = nn_sum(out4)
                return out

    elif test_mode == "module":
        @mark(name="nn_lin", mode="post")
        class ActModule(nn.Module):
            def forward(self, x):
                return x + x.abs()

        @mark(name="nn_sum", mode="pre")
        class SumModule(nn.Module):
            def forward(self, x): 
                return (x.sum() - x.abs().sum()).cos()
    
        class MarkFoo(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 30)
                self.nn_lin = ActModule()
                self.linear3 = nn.Linear(30, 40)
                self.linear4 = nn.Linear(40, 50)
                self.nn_sum = SumModule()

            def forward(self, x):
                out1 = mark(self.linear1(x))
                out2 = mark(self.linear2(out1))
                out3 = mark(self.linear3(out2))
                out_nn = self.nn_lin(out3)
                out4 = mark(self.linear4(out_nn), mode="pre")
                out = self.nn_sum(out4)
                return out

    obj = MarkFoo()

    # test scripting
    obj_scripted = torch.jit.script(obj)
    obj_scripted(torch.randn(4, 10))

    # test aliases
    obj = BendedModule(obj)
    obj.trace(x=torch.Tensor(4, 10))
    aliases = obj.graph().aliases
    print(aliases)
