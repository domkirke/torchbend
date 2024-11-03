import sys, os, pytest
import torch
testpath = os.path.abspath((os.path.join(os.path.dirname(__file__), "..")))
if testpath not in sys.path:
    sys.path.append(testpath)
from test_modules.module_test_modules import modules_to_test, ModuleTestConfig 


devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices.append(torch.device('cuda'))
# if torch.mps.device_count() > 0:
#     devices.append(torch.device('mps'))

def to(args, kwargs, device):
    args = list(args)
    kwargs = dict(kwargs)
    for i, a in enumerate(args):
        if torch.is_tensor(a): args[i] = args[i].to(device)      
    for k, v in kwargs.items():
        if torch.is_tensor(v): kwargs[k] = v.to(device)
    return tuple(args), kwargs

@pytest.mark.parametrize("module_config", modules_to_test)
@pytest.mark.parametrize("device", [torch.device('cpu'), torch.device('mps')])
def test_to(module_config, device):
    bended = module_config.get_bended_module()
    for m in module_config.get_methods():
        args, kwargs, _, _ = module_config.get_method_args(m)    
        args, kwargs = to(args, kwargs, device)
        bended = bended.to(device)
        out = getattr(bended, m)(*args, **kwargs)
