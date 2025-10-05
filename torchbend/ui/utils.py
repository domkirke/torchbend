import os
import torch
import torchvision.transforms as tv_transforms
import torchaudio
from pathlib import Path

from ..utils import get_random_hash


TB_DEFAULT_GENERATION_DIR = Path(os.environ.get("TB_DEFAULT_GENERATION_DIR") or Path(os.getcwd()) / "generations")

def tensor_to_image(input_tensor, filename=None, upscale=None, norm_fn=None, out=None):
    out = out or TB_DEFAULT_GENERATION_DIR
    filename = filename or os.path.join(TB_DEFAULT_GENERATION_DIR, get_random_hash(n=8)+".png")
    # First convert back to cpu and detach from computational graph if linked
    tensor = input_tensor.to('cpu').detach()
    if norm_fn is not None:
        tensor = norm_fn(tensor)
    if upscale:
        assert isinstance(upscale, int), "upscale keyword argument must be an int"
        tensor = torch.nn.functional.interpolate(tensor[None], scale_factor=upscale)[0]

    # Convert tensor to PIL Image
    transform = tv_transforms.ToPILImage()
    image = transform(tensor)

    # Save Image
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    image.save(filename)
    return filename



def tensor_to_audio(input_tensor, filename=None, sr=None, out=None):
    filename = filename or os.path.join(TB_DEFAULT_GENERATION_DIR, get_random_hash(n=8)+".wav")
    # First convert back to cpu and detach from computational graph if linked
    tensor = input_tensor.to('cpu').detach()
    torchaudio.save(filename, tensor, sample_rate=sr)
    return filename
