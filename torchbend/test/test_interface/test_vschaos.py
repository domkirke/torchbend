import os
import random
from pathlib import Path
import itertools
import torch
import pytest
import torchbend as tb

VSCHAOS2_AVAILABLE = False
try:
    import vschaos
    from torchbend.interfaces.vschaos2 import BendedVSChaos
    VSCHAOS2_AVAILABLE = True
except ModuleNotFoundError:
    VSCHAOS2_AVAILABLE = False

VSCHAOS2_MODEL_PATHS = [Path("models/vschaos2/fm/version_10/fm_epoch=9757.vs")]
VSCHAOS2_STRICT_LOADING = False
VSCHAOS2_MODEL_DL_PATH = None
rave_test_activations = {
    'forward': [], 
    'encode': [], 
    'decode': []
}

def get_test_name(): 
    return os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

def check_rave_models():
    valid_paths = []
    for d in VSCHAOS2_MODEL_PATHS:
        if Path(d).exists(): 
            valid_paths.append(d)
        else:
            print(f'[Warning] path {d} not valid for RAVE tests')
    return valid_paths

        
@pytest.mark.skipif(not VSCHAOS2_AVAILABLE, reason="rave not available")
@pytest.mark.parametrize("model_path", VSCHAOS2_MODEL_PATHS)
@pytest.mark.parametrize("scriptable", [True, False])
def test_import(model_path, scriptable):
    check_rave_models()
    model, config, transform = BendedVSChaos.load_model(model_path, scriptable=scriptable)
    assert model
    assert config
    assert transform


@pytest.mark.skipif(not VSCHAOS2_AVAILABLE, reason="rave not available")
@pytest.mark.parametrize("model_path", VSCHAOS2_MODEL_PATHS)
@pytest.mark.parametrize("scriptable", [True, False])
def test_tracing(model_path, scriptable):
    check_rave_models()
    model = BendedVSChaos(model_path, scriptable=scriptable)
