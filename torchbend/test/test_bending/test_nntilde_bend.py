import os
from pathlib import Path
import torch, torch.nn as nn
import torchbend as tb

outdir = Path(__file__).parent / "nn_tests"
os.makedirs(outdir, exist_ok=True)



class PermuteFoo(nn.Module):
    def forward(self, x):
        out = x + 0
        out_1 = out + 0
        return out_1
    

def test_permute_nn(n_channels=4):
    foo = PermuteFoo()
    bended = tb.BendedModule(foo)
    x = torch.randn(1, n_channels, 8192)
    bended.trace(x=x)

    seed_param = tb.BendingParameter("seed", -1, range=[-1, None])
    cb = tb.Permute(seed=seed_param, dim=-2)
    bended.bend(cb, "add")
    bended.forward(x)

    scripted = bended.nntilde()

    test_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    out_path = outdir / f"{test_name}.ts"
    torch.jit.save(scripted, out_path)

def test_reverse_nn(n_channels=4):
    foo = PermuteFoo()
    bended = tb.BendedModule(foo)
    x = torch.randn(1, n_channels, 8192)
    bended.trace(x=x)

    bypass = tb.BendingParameter("reverse", 0)
    cb = tb.Reverse(dim=-1, reverse=bypass)
    bended.bend(cb, "add")
    bended.forward(x)

    scripted = bended.nntilde()

    test_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    out_path = outdir / f"{test_name}.ts"
    torch.jit.save(scripted, out_path)


