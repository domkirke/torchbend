import os
import random
from pathlib import Path
import itertools
import torch
import pytest
import torchbend as tb

RAVE_AVAILABLE = False
try:
    import rave
    import cached_conv
    from torchbend.interfaces.rave import BendedRAVE
    RAVE_AVAILABLE = True
except ModuleNotFoundError:
    RAVE_AVAILABLE = False

RAVE_MODEL_PATHS = [Path("models/rave/test")]
RAVE_STRICT_LOADING = False
RAVE_MODEL_DL_PATH = None
rave_test_activations = {
    'forward': [], 
    'encode': [], 
    'decode': []
}

RAVE_TS_DIR = Path("outs") / "nntilde" / "rave"

def save_scripted(obj, test_name):
    target_dir = (Path(__file__).parent / RAVE_TS_DIR).resolve()
    os.makedirs(target_dir, exist_ok=True)
    torch.jit.save(obj, target_dir / f"{test_name}.ts")

def get_test_name(): 
    return os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

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

if RAVE_AVAILABLE:
    RAVE_MODEL_PATHS = check_rave_models()
else:
    RAVE_MODEL_PATHS = []
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
@pytest.mark.parametrize("cached", [True, False])
@pytest.mark.parametrize("scriptable", [True, False])
def test_callbacks(model_path, batch_size, cached, scriptable):
    cached_conv.use_cached_conv(cached)
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

MAX_ACTIVATION_TESTS = 4

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
    encode_acts = model.activations('#encoder_act', fn="encode")
    decode_acts = model.activations('#decoder_act', fn="decode")
    forward_acts = model.activations('#encoder_act', '#decoder_act', fn="forward")

    x = torch.zeros(1, model.channels, 8192)
    z = model.encode(x)

    for i, e_act in enumerate(encode_acts):
        acts = model.get_activations(f"{e_act}", x=x, fn="encode")
        out = model.from_activations(f"{e_act}", **acts, x=x, fn="encode")
        if i >= MAX_ACTIVATION_TESTS: break

    for i, d_act in enumerate(decode_acts):
        acts = model.get_activations(f"{d_act}", z=z, fn="decode")
        out = model.from_activations(f"{d_act}", **acts, z=z, fn="decode")
        if i >= MAX_ACTIVATION_TESTS: break

    for i, f_act in enumerate(forward_acts):
        acts = model.get_activations(f"{f_act}", x=x, fn="forward")
        out = model.from_activations(f"{f_act}", **acts, x=x, fn="forward")
        if i >= MAX_ACTIVATION_TESTS: break

        

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
    torch.jit.save(model, f'.rave_test_{get_test_name()}.ts')
    os.remove(f'.rave_test_{get_test_name()}.ts')

    out = model(x)
    out = model.forward(x)
    z = model.encode(x)
    out = model.decode(z)

    out_full = model.encode_full(x)
    out = model.encode_dist(x)
    out = model.decode_full(out_full)





@pytest.mark.skipif(not RAVE_AVAILABLE, reason="rave not available")
@pytest.mark.parametrize("model_path", RAVE_MODEL_PATHS)
@pytest.mark.parametrize("jit", [True, False])
def test_nntilde_split(model_path, jit):
    model = BendedRAVE(model_path, scriptable=True, strict=RAVE_STRICT_LOADING)
    forward_acts = model.aliases()['encoder_act'][0]
    x = torch.zeros(1, model.channels, 8192)

    bending_op = tb.Mask(prob=tb.BendingParameter("mask", 0.), dim=-2)
    model.bend(bending_op, *model.aliases()['encoder_act'])

    out = model.get_activations(f"{forward_acts}", x=x, _save_as_method=f"get_{forward_acts}")
    out = model.from_activations(f"{forward_acts}", x=x, **out, _save_as_method=f"from_{forward_acts}")

    # check obtained methods
    out_act = getattr(model, f"get_{forward_acts}")(x)
    out = getattr(model, f"from_{forward_acts}")(x, out_act)

    model = model.nntilde(script=jit, force_default=True)

    out_act = getattr(model, f"get_{forward_acts}")(x)
    out_act = torch.nn.functional.interpolate(out_act, size=x.shape[-1])
    from_input = torch.cat([x, out_act], -2)
    out = getattr(model, f"from_{forward_acts}")(from_input)
    
    if jit:
        torch.jit.save(model, f'.rave_test_{get_test_name()}.ts')
        os.remove(f'.rave_test_{get_test_name()}.ts')

    # out = model(x)
    # out = model.forward(x)
    # z = model.encode(x)
    # out = model.decode(z)

@pytest.mark.skipif(not RAVE_AVAILABLE, reason="rave not available")
@pytest.mark.parametrize("model_path", RAVE_MODEL_PATHS)
def test_latent_steering_with_input_controllables(model_path):
    test_name = get_test_name()
    model = BendedRAVE(model_path, scriptable=True, strict=RAVE_STRICT_LOADING)

    x = torch.full((1, 1, 8192), 0.)
    out_orig = model.forward(x, clear_cache=True)
    out_orig2 = model.forward(x, clear_cache=True) 

    target_activations = model.activations('#latent_pca')
    target_activations_b = [f"{t}_bended" for t in target_activations]
    activation_shape = model.activation_shape(list(target_activations.keys())[0])

    scale_value = torch.full((1, *activation_shape[1:]), 1.)
    scale_in = tb.BendingParameter('latent_scale', scale_value, True)
    bias_value = torch.full((1, *activation_shape[1:]), 0.)
    bias_in = tb.BendingParameter('latent_bias', bias_value, True)
    affine_cb = tb.Affine(scale=scale_in, bias=bias_in)
    affine_cb.nntilde()
    model.bend(affine_cb, *target_activations, fn="forward")

    for m in ['forward', 'encode']:
        target_act = model.activation_names('#latent_pca', fn=m)[0]
        out = model.get_activations(f"{target_act}_bended", x=x, latent_scale=torch.zeros_like(scale_value), latent_bias=bias_value, fn=m)
        assert tb.compare_outs(out[f"{target_act}_bended"], torch.zeros_like(out[f"{target_act}_bended"]))

    scripted = model.nntilde()
    x_scripted = torch.cat([
        x, 
        torch.full((x.shape[0], model.latent_size, x.shape[-1]), 1.), 
        torch.full((x.shape[0], model.latent_size, x.shape[-1]), 0.), 
    ], dim=-2)
    out = scripted.encode(x_scripted)
    out = scripted.forward(x_scripted)
    save_scripted(scripted, test_name)


@pytest.mark.skipif(not RAVE_AVAILABLE, reason="rave not available")
@pytest.mark.parametrize("model_path", RAVE_MODEL_PATHS)
def test_modulation_with_input_controllables(model_path):
    test_name = get_test_name()
    model = BendedRAVE(model_path, scriptable=True, strict=RAVE_STRICT_LOADING)

    x = torch.full((1, 1, 8192), 0.)
    out_orig = model.forward(x, clear_cache=True)
    out_orig2 = model.forward(x, clear_cache=True) 

    target_activations = model.activation_names('#decoder_act', fn="decode")
    target_fw_activations = model.activation_names('#decoder_act', fn="forward")

    for i, t in enumerate(target_activations):
        t = [f"decode:{t}", f"forward:{target_fw_activations[i]}"]
        activation_shape = model.activation_shape(t[0])
        scale_value = torch.full((1, 1, activation_shape[2]), 1.)
        scale_in = tb.BendingParameter(f'mod_{i}', scale_value, True)
        affine_cb = tb.Scale(scale=scale_in)
        affine_cb.nntilde()
        model.bend(affine_cb, *t)

    out_encoder = model.encode(x)
    for i, t in enumerate(target_activations):
        out = model.get_activations(f"{t}_bended", z=out_encoder, **{f'mod_{i}': torch.zeros_like(scale_value)}, fn="decode")
        assert tb.compare_outs(out[f"{t}_bended"], torch.zeros_like(out[f"{t}_bended"]))
    for i, t in enumerate(target_fw_activations):
        out = model.get_activations(f"{t}_bended", x=x, **{f'mod_{i}': torch.zeros_like(scale_value)}, fn="forward")
        assert tb.compare_outs(out[f"{t}_bended"], torch.zeros_like(out[f"{t}_bended"]))


    scripted = model.nntilde()
    z_scripted = torch.cat([
        out_encoder, 
        torch.full((out_encoder.shape[0], len(target_activations), out_encoder.shape[-1]), 1.), 
    ], dim=-2)
    out = scripted.decode(z_scripted)

    x_scripted = torch.cat([
        x, 
        torch.full((x.shape[0], len(target_activations), x.shape[-1]), 1.), 
    ], dim=-2)
    out = scripted.forward(x_scripted)
    save_scripted(scripted, test_name)
























