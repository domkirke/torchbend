import os
import torchvision
import torch
import urllib.request

from pathlib import Path
import torchbend as tb
import pytest

STYLEGAN_AVAILABLE = False
try: 
    from torchbend.interfaces.stylegan import BendedStyleGAN
    STYLEGAN_AVAILABLE = True
except ModuleNotFoundError:
    STYLEGAN_AVAILABLE = False
 
model_dl_dir = (Path(__file__).parent / "models" / "sg3").resolve()
model_links = {
    'stylegan2-cifar10-32x32.pkl': 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/stylegan2/1/files?redirect=true&path=stylegan2-cifar10-32x32.pkl'
}

from torchvision.transforms.functional import to_pil_image


IMAGE_OUT_DIR = (Path(__file__) / ".." / "outs" / "stylegan3").resolve()

N_BATCHES = 4
   

def get_test_name(): 
    return os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]



def save_image_example(out, model_args, test_name, suffix):
    model_name = Path(list(model_args)[0]).stem
    out_dir = IMAGE_OUT_DIR / model_name 
    if not out_dir.exists(): os.makedirs(out_dir)
    out_path = out_dir / f"{test_name}_{suffix}.png"
    out = (out.clamp(-1, 1) + 1) / 2
    out = to_pil_image(torchvision.utils.make_grid(out))
    # torchvision.io.write_png(out.detach().cpu(), str(out_path))
    out.save(str(out_path))


def get_model_args():
    os.makedirs(str(model_dl_dir), exist_ok=True)
    models = []
    for k, v in model_links.items():
        model_path = model_dl_dir / k
        if not model_path.exists(): 
            urllib.request.urlretrieve(v, str(model_path))
        models.append((BendedStyleGAN, tb.PArgs(model_path)))
    return models

def get_inputs(model, n_inputs = 1):
    inputs = []
    z = torch.randn(n_inputs, model.latent_dim)
    inputs.append(z)
    if model.conditioning_dim:
        # c = torch.nn.functional.one_hot()
        c = torch.randint(0, model.conditioning_dim, (n_inputs, ))
        c = torch.nn.functional.one_hot(c, model.conditioning_dim)
        inputs.append(c)
    return inputs
    

@pytest.mark.parametrize("model_class,model_args", get_model_args())
def test_sg3_generation(model_class, model_args, n_images = N_BATCHES):
    model = model_class(*model_args, **model_args)
    inputs = get_inputs(model, n_images)
    out = model.forward(*inputs)
    save_image_example(out, model_args, get_test_name(), "original")
    prob = tb.BendingParameter('mask', 1.0)
    cb = tb.Mask(prob=prob)

    model.bend(cb, "?.*weight")
    prob.set_value(0.6)
    out_bended = model.forward(*inputs)
    save_image_example(out_bended, model_args, get_test_name(), "bended")
    
