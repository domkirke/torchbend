import os, sys
import torch, torch.nn as nn
import pytest
import torchbend as tb


testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)
from test_modules.module_test_modules import modules_to_test, ModuleTestConfig

@pytest.mark.parametrize("test_modules", modules_to_test)
@pytest.mark.parametrize("callbacks,is_equal", [((tb.Scale(2.), tb.Scale(0.5)), True),
                                                ((tb.Bias(1.), tb.Bias(-1.)), True),
                                                ((tb.Mask(0.4), tb.Mask(0.4)), False)])

def test_piping(test_modules, callbacks, is_equal):
    
    for m in test_modules.get_methods():
        module, bended = test_modules.get_modules()
        bended.reset()
        args, kwargs, weight_targets, act_targets  = test_modules.get_method_args(m)
        bended.trace(func=m, **kwargs)
        out_orig = getattr(module, m)(*args, **kwargs)

        pipe = callbacks[0] >> callbacks[1]
        # bended.bend(pipe, *act_targets, *weight_targets)
        bended.bend(pipe, bended.bendable_keys(weight_targets[0])[1])
        out_bended = getattr(bended, m)(*args, **kwargs)

        assert bool(tb.compare_outs(out_orig, out_bended, allow_almost_equal=True)) == is_equal

        # test scripting
        if pipe.jit_compatible:
            scripted = torch.jit.script(pipe)