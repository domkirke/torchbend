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
                 reinit_module: Path | str = None,
                 load_ema: bool = False,
                 **kwargs):
        repository_path = check_stylegan(repository_link, repository_path)
        sys.path.insert(0, str(repository_path.absolute()))
        pretrained = numpy.load(str(pretrained), allow_pickle=True)
        pickle_key = "G" if not load_ema else "G_ema"
        if reinit_module is not None: 
            sys.path.insert(0, str(repository_path / "training"))
            assert reinit_module in ['sg2', 'sg3']
            if reinit_module == "sg2":
                from networks_stylegan2 import Generator
            else:
                from networks_stylegan3 import Generator
            module = Generator(**pretrained[pickle_key].init_kwargs) 
            module.load_state_dict(pretrained[pickle_key].state_dict())
        else:
            module = pretrained['G'].to(device)
        super(BendedStyleGAN, self).__init__(module)
        del sys.path[0]

    def get_inputs(model, n_batches = 4):
        z = torch.randn(n_batches, model.latent_dim)
        if model.conditioning_dim:
            # c = torch.nn.functional.one_hot()
            c = torch.randint(0, model.conditioning_dim, (n_batches, ))
            c = torch.nn.functional.one_hot(c, model.conditioning_dim)
        else:
            c = None
        return {'z': z, 'c': c}

    def bend_model(self, model, n_batches=4):
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
