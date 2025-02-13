import os
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


def locate_channel_amount_change(activations, init_channels=1):
    current_n_channels = init_channels
    acts = []
    for n, a in activations.items():
        if a.op == "placeholder": continue
        if "getitem" in a.name: continue
        if "cat" in a.name: continue
        if "copy" in a.name: continue
        if not isinstance(a.shape, (tuple, torch.Size)): continue
        if len(a.shape) < 3: continue
        if a.shape[-2] != current_n_channels:
            acts.append(n)
            current_n_channels = a.shape[-2]
    return acts


RAVE_MODEL_PATHS = check_rave_models()
RAVE_TEST_BATCH_SIZE = (1, 4)
        
@pytest.mark.skipif(not RAVE_AVAILABLE, reason="rave not available")
@pytest.mark.parametrize("model_path", RAVE_MODEL_PATHS)
@pytest.mark.parametrize("scriptable", [True, False])
def test_import(model_path, scriptable):
    check_rave_models()
    model = BendedRAVE.load_model(model_path, scriptable=scriptable, strict=RAVE_STRICT_LOADING)
    assert model

@pytest.mark.skipif(not RAVE_AVAILABLE, reason="rave not available")
@pytest.mark.parametrize("model_path", RAVE_MODEL_PATHS)
@pytest.mark.parametrize("batch_size", RAVE_TEST_BATCH_SIZE)
@pytest.mark.parametrize("scriptable", [True, False])
def test_callbacks(model_path, batch_size, scriptable):
    model = BendedRAVE(model_path, scriptable=scriptable, strict=RAVE_STRICT_LOADING)
    x = torch.randn(batch_size, model.channels, 2048)

    # test attributes
    assert model.sample_rate
    assert model.channels

    # test interface methods
    z = model.encode(x, postprocess=True)
    z_nopostprocess = model.encode(x, postprocess=False)

    x_rec = model.decode(z, preprocess=True)
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
@pytest.mark.parametrize("scriptable", (True, False)) 
def test_tracing(model_path, scriptable):
    model = BendedRAVE(model_path, scriptable=scriptable, strict=RAVE_STRICT_LOADING)

    # test variable batch sizes
    batch_sizes = RAVE_TEST_BATCH_SIZE
    for b in batch_sizes:
        out = model.forward(torch.zeros(b, model.channels, 8192))

    # print weights & activations
    model.print_weights(out=Path(model_path)/ "weights.txt")
    for method in rave_test_activations: 
        model.print_activations(fn=method, out=Path(model_path) / f"activations_{method}.txt")

    # test encode
    encode_acts = locate_channel_amount_change(model.activations(fn="encode"))[:2]
    decode_acts = locate_channel_amount_change(model.activations(fn="decode"), model.latent_size)[:2]
    forward_acts = locate_channel_amount_change(model.activations(fn="forward"))[:2]

    x = torch.zeros(1, model.channels, 8192)
    z = model.encode(x)

    for e_act in encode_acts:
        acts = model.get_activations(f"{e_act}$", x=x, fn="encode")
        out = model.from_activations(f"{e_act}$", **acts, x=x, fn="encode")

    for d_act in decode_acts:
        acts = model.get_activations(f"{d_act}$", z=z, fn="decode")
        out = model.from_activations(f"{d_act}$", **acts, z=z, fn="decode")

    for f_act in forward_acts:
        acts = model.get_activations(f"{f_act}$", x=x, fn="forward")
        out = model.from_activations(f"{f_act}$", **acts, x=x, fn="forward")

        

@pytest.mark.skipif(not RAVE_AVAILABLE, reason="rave not available")
@pytest.mark.parametrize("model_path", RAVE_MODEL_PATHS)
def test_script_export(model_path):
    model = BendedRAVE(model_path, scriptable=True, strict=RAVE_STRICT_LOADING)
    x = torch.zeros(1, model.channels, 8192)
    model = model.script(script=True)
    torch.jit.save(model, '.test.ts')
    os.remove('.test.ts')

    out = model(x)
    out = model.forward(x)
    z = model.encode(x)
    out = model.decode(z)


@pytest.mark.skipif(not RAVE_AVAILABLE, reason="rave not available")
@pytest.mark.parametrize("model_path", RAVE_MODEL_PATHS)
def test_nntilde_export(model_path):
    model = BendedRAVE(model_path, scriptable=True, strict=RAVE_STRICT_LOADING)
    x = torch.zeros(1, model.channels, 8192)
    model = model.nntilde(script=True)
    torch.jit.save(model, '.test.ts')
    os.remove('.test.ts')

    out = model(x)
    out = model.forward(x)
    z = model.encode(x)
    out = model.decode(z)




@pytest.mark.skipif(not RAVE_AVAILABLE, reason="rave not available")
@pytest.mark.parametrize("model_path", RAVE_MODEL_PATHS)
def test_nntilde_split(model_path):
    model = BendedRAVE(model_path, scriptable=True, strict=RAVE_STRICT_LOADING)
    x = torch.zeros(1, model.channels, 8192)

    forward_acts = "add_44"

    out = model.get_activations(forward_acts, x=x, _save_as_method=f"get_{forward_acts}")
    out = model.from_activations(forward_acts, x=x, **out, _save_as_method=f"from_{forward_acts}")

    # check obtained methods
    out_act = getattr(model, f"get_{forward_acts}")(x)
    out = getattr(model, f"from_{forward_acts}")(x, out_act)

    model = model.nntilde(script=True, force_default=True)

    out_act = getattr(model, f"get_{forward_acts}")(x)
    out_act = torch.nn.functional.interpolate(out_act, size=x.shape[-1])
    from_input = torch.cat([x, out_act], -2)
    out = getattr(model, f"from_{forward_acts}")(from_input)
    
    torch.jit.save(model, '.test.ts')
    os.remove('.test.ts')

    # out = model(x)
    # out = model.forward(x)
    # z = model.encode(x)
    # out = model.decode(z)

