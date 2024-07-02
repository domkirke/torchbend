import pytest
import torch
from conftest import compare_outs

const_tensor = torch.rand(2, 2)

@pytest.mark.parametrize("comp_simple_args", [
    (1, 1, True),
    (3.4, 3.4, True), 
    (2, 2.1, False), 
    (torch.tensor(1.), 1., False), 
    ({'a': torch.tensor(1.)}, {'a': torch.tensor(1.).clone()}, True), 
    ({'a': torch.tensor(1.)}, {'a': torch.tensor(1.).clone(), 'b': 2}, False), 
    ([1, 2, 3], [1, 2, 3], True),
    ([1], [1, 2, 3], False)
])
def test_eq(comp_simple_args):
    arg1, arg2, result = comp_simple_args
    out = compare_outs(arg1, arg2)
    assert bool(out) == result, "got wrong result: expected %s, got %s"%(result, out)