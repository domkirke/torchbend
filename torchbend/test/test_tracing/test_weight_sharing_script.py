import os
from pathlib import Path
import torch
import torchbend as tb
import torch.nn as nn


TMP_PATH = Path("/tmp")


class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.huge_linear = nn.Linear(8192, 8192)

    @torch.jit.export
    def a(self, x):
        return self.huge_linear(x)


class Foo2(nn.Module):
    def __init__(self):
        super().__init__()
        self.huge_linear = nn.Linear(8192, 8192)

    @torch.jit.export
    def a(self, x):
        return self.huge_linear(x)
    
    @torch.jit.export
    def b(self, x):
        return self.huge_linear(x)

    @torch.jit.export
    def c(self, x):
        return self.huge_linear(x)

    @torch.jit.export
    def d(self, x):
        out = self.huge_linear(x)
        out2 = self.huge_linear(x)
        return out2

def test_weight_sharing_jit():
    module1 = Foo()
    module2 = Foo2()
    scripted1 = torch.jit.script(module1)
    scripted2 = torch.jit.script(module2)
    path1 = TMP_PATH / "foo1.ts"
    path2 = TMP_PATH / "foo2.ts"
    torch.jit.save(scripted1, TMP_PATH / "foo1.ts")
    torch.jit.save(scripted2, TMP_PATH / "foo2.ts")
    size1 = os.path.getsize(path1)
    size2 = os.path.getsize(path2)
    assert size1 // size2 < 2 or size1 // size2 > 0.5 


def test_weight_sharing_as_export():
    module1 = tb.BendedModule(Foo2())
    module2 = tb.BendedModule(Foo2())
    x = torch.randn(1, 8192)
    module1.trace("a", x=x)
    for fn in ['a', 'b', 'c', 'd']:
        module2.trace(fn, x=x)

    path1 = TMP_PATH / "module1.ts"
    path2 = TMP_PATH / "module2.ts"
   
    scripted1 = module1.script()
    scripted2 = module2.script()

    assert 'a' in scripted1._available_methods
    for fn in ['a', 'b', 'c', 'd']:
        assert fn in scripted2._available_methods
    
    torch.jit.save(scripted1, path1)
    torch.jit.save(scripted2, path2)

    size1 = os.path.getsize(path1)
    size2 = os.path.getsize(path2)
    assert size2 / size1 < 2, "module weights are duplicated"
    print(size1, size2)



    



    
    