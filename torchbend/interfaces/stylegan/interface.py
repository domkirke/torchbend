import torchbend as tb
import requests, zipfile, io
from pathlib import Path
from ..base import Interface
import sys
import numpy
import os, torch

STYLEGAN_REPO_LINK = os.environ.get('TB_STYLEGAN_LINK', 'https://github.com/NVlabs/stylegan3/archive/refs/heads/main.zip')
STYLEGAN_PATH = Path(__file__).parent / "stylegan3-main"

def check_stylegan(sg_link = STYLEGAN_REPO_LINK, sg_path = STYLEGAN_PATH):
    sg_path = Path(sg_path)
    if not sg_path.exists():
        response = requests.get(str(sg_link))
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract all files to current directory
            z.extractall(path=str(sg_path.parent))
        #TODO brows for correct dir
        return sg_path / "stylegan3-main"
    else:
        return sg_path


class BendedStyleGAN(Interface):
    _imported_callbacks_ = ['forward']                        

    def __init__(self, 
                 pretrained, 
                 *args, 
                 device=torch.device('cpu'), 
                 repository_link: Path | str = STYLEGAN_REPO_LINK, 
                 repository_path: Path | str = STYLEGAN_PATH,
                 **kwargs):
        repository_path = check_stylegan(repository_link, repository_path)
        sys.path.insert(0, str(repository_path.absolute()))
        pretrained = numpy.load(str(pretrained), allow_pickle=True)
        super(BendedStyleGAN, self).__init__(pretrained['G'].to(device))

    def get_inputs(model, n_batches = 1):
        z = torch.randn(n_batches, model.latent_dim)
        if model.conditioning_dim:
            # c = torch.nn.functional.one_hot()
            c = torch.randint(0, model.conditioning_dim, (n_batches, ))
            c = torch.nn.functional.one_hot(c, model.conditioning_dim)
        return {'z': z, 'c': c}

    def _bend_model(self, model, n_batches=1):
        model.trace("forward", **self.get_inputs(n_batches))

    @property
    def latent_dim(self):
        return self._model.mapping.z_dim

    @property
    def conditioning_dim(self): 
        return getattr(self._model.mapping, "c_dim", None)

    @property
    def intermediate_latent_dim(self):
        return getattr(self._model.mapping, "w_dim", None)
