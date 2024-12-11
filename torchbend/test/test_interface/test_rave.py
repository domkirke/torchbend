from pathlib import Path
import itertools
import torch
import pytest
from torchbend.interfaces.rave import BendedRAVE

RAVE_AVAILABLE = False
try:
    import rave
    RAVE_AVAILABLE = True
except ModuleNotFoundError:
    exit()

RAVE_MODEL_PATHS = [Path("models/rave/test")]
RAVE_STRICT_LOADING = False
RAVE_MODEL_DL_PATH = None
rave_test_activations = {
    'forward': [], 
    'encode': [], 
    'decode': []
}

def check_rave_models():
    valid_paths = []
    for d in RAVE_MODEL_PATHS:
        if Path(d).exists(): 
            valid_paths.extend(filter(lambda x: BendedRAVE.is_loadable(x), d.iterdir()))
        else:
            print(f'[Warning] path {d} not valid for RAVE tests')
    return valid_paths

RAVE_MODEL_PATHS = check_rave_models()
        
@pytest.mark.skipif(not RAVE_AVAILABLE, reason="rave not available")
@pytest.mark.parametrize("model_path", RAVE_MODEL_PATHS)
def test_import(model_path):
    check_rave_models()
    model = BendedRAVE.load_model(model_path, strict=RAVE_STRICT_LOADING)

@pytest.mark.skipif(not RAVE_AVAILABLE, reason="rave not available")
@pytest.mark.parametrize("model_path", RAVE_MODEL_PATHS)
@pytest.mark.parametrize("batch_size", (1, 4))
def test_callbacks(model_path, batch_size):
    model = BendedRAVE(model_path, batch_size=batch_size, strict=RAVE_STRICT_LOADING)
    x = torch.randn(batch_size, model.channels, 2048)

    # test attributes
    sample_rate = model.sample_rate
    channels = model.channels

    # test interface methods
    z = model.encode(x)
    z_nopostprocess = model.encode(x, postprocess=False)

    x_rec = model.decode(z)
    x_rec_nopreprocess = model.decode(z_nopostprocess, preprocess=False)

    # test helper methods
    fid = model.get_fidelity_for_dims(8)
    dims = model.get_dims_for_fidelity(0.8)
    rf = model.receptive_field

    assert x_rec.shape == x_rec_nopreprocess.shape    
    assert z.shape == z_nopostprocess.shape
    assert z.shape[-2] == model.latent_size
    assert z_nopostprocess.shape[-2] == model.latent_size


@pytest.mark.skipif(not RAVE_AVAILABLE, reason="rave not available")
@pytest.mark.parametrize("model_path", RAVE_MODEL_PATHS)
@pytest.mark.parametrize("batch_size", (1, 4)) 
def test_tracing(model_path, batch_size):
    model = BendedRAVE(model_path, batch_size=batch_size, strict=RAVE_STRICT_LOADING)
    assert "encode" in model._model._graphs
    assert "decode" in model._model._graphs
    assert "forward" in model._model._graphs


    # test variable batch sizes
    batch_sizes = (1, 4)
    for b in batch_sizes:
        out = model.forward(torch.zeros(b, model.channels, 8192))
        model.print_weights(out=Path(model_path)/ "weights.txt")
        for method in rave_test_activations: 
            model.print_activations(fn=method, out=Path(model_path) / f"activations_{method}.txt")

    def locate_channel_amount_change(activations, init_channels=1):
        current_n_channels = 1
        acts = []
        for n, a in activations.items():
            if a.op == "placeholder": continue
            if not isinstance(a.shape, (tuple, torch.Size)): continue
            if len(a.shape) < 3: continue
            if a.shape[-2] != current_n_channels:
                acts.append(n)
                current_n_channels = a.shape[-2]
        return acts

    # test encode
    encode_acts = locate_channel_amount_change(model.activations(fn="encode"))
    decode_acts = locate_channel_amount_change(model.activations(fn="decode"), model.latent_size)
    forward_acts = locate_channel_amount_change(model.activations(fn="forward"))

    x = torch.zeros(batch_size, model.channels, 8192)
    z = model.encode(x)

    for e_act in encode_acts:
        acts = model.get_activations(e_act, x=x, fn="encode")
        out = model.from_activations(e_act, **acts, x=x, fn="encode")

    for d_act in decode_acts:
        acts = model.get_activations(d_act, z=z, fn="decode")
        out = model.from_activations(d_act, **acts, z=z, fn="decode")

    for f_act in forward_acts:
        acts = model.get_activations(f_act, x=x, fn="forward")
        out = model.from_activations(f_act, **acts, x=x, fn="forward")
        

def test_script():
    pass

def test_export():
    pass