import urllib.parse
import torchbend as tb
import urllib
import requests, zipfile, io
from pathlib import Path
from ..base import Interface
import sys
import numpy
import os, torch
from ..base import _overload_module


#TODO embed a wheel? 
STYLEGAN_REPO_LINK = os.environ.get('TB_STYLEGAN_LINK', 'https://github.com/domkirke/stylegan3/archive/refs/heads/video.zip')
STYLEGAN_PATH = Path(__file__).parent / "stylegan3-video"
TB_DEFAULT_MODEL_DIR = tb._TORCHBEND_DEFAULT_MODEL_DIR / "stylegan3"

def check_stylegan(sg_link = STYLEGAN_REPO_LINK, sg_path = STYLEGAN_PATH):
    sg_path = Path(sg_path)
    if not sg_path.exists():
        response = requests.get(str(sg_link))
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract all files to current directory
            z.extractall(path=str(sg_path.parent))
        #TODO brows for correct dir
        return sg_path 
    else:
        return sg_path



class BendedStyleGAN(Interface):
    _imported_callbacks_ = ['forward']                        
    _panel_render_type_ = "image"
    _download_subdir = "sg3"
    traced_methods = ['forward']

    @property
    def _panel_out_norm_fn(self): 
        return lambda x: (x.clamp(-1, 1) + 1) / 2


    def __init__(self, 
                 pretrained_path, 
                 n_batches=1,
                 *args, 
                 device=torch.device('cpu'), 
                 repository_link: Path | str = STYLEGAN_REPO_LINK, 
                 repository_path: Path | str = STYLEGAN_PATH,
                 reinit_module: Path | str = None,
                 load_ema: bool = False,
                 **kwargs):
        
        repository_path = check_stylegan(repository_link, repository_path)
        sys.path.insert(0, str(repository_path.absolute()))
        pretrained_path = self.get_model_path(pretrained_path)
        pretrained = numpy.load(str(pretrained_path), allow_pickle=True)
        # pretrained = pickle.load()
        pickle_key = "G" if not load_ema else "G_ema"
        self.device = device
        if reinit_module is None: 
            if "stylegan2" in pretrained_path.stem: reinit_module = "sg2"
            if "stylegan3" in pretrained_path.stem: reinit_module = "sg3"

        if reinit_module is not None: 
            sys.path.insert(0, str(repository_path / "training"))
            assert reinit_module in ['sg2', 'sg3']
            if reinit_module == "sg2":
                from networks_stylegan2 import Generator
            else:
                from networks_stylegan3 import Generator
            module = Generator(**pretrained[pickle_key].init_kwargs)
            module.load_state_dict(pretrained[pickle_key].state_dict())
            module = module.to(device)
        else:
            logging.warning("Could not fetch from filename if imported from StyleGAN2 or StyleGAN3. Activations not marked.")
            module = pretrained['G'].to(device)
        self.n_batches = n_batches
        super(BendedStyleGAN, self).__init__(module)
        del sys.path[0]

    def get_inputs(model, n_batches = None):
        n_batches = n_batches or model.n_batches
        z = torch.randn(n_batches, model.latent_dim).to(model.device) 
        if model.conditioning_dim:
            # c = torch.nn.functional.one_hot()
            c = torch.randint(0, model.conditioning_dim, (n_batches, ))
            c = torch.nn.functional.one_hot(c, model.conditioning_dim).to(model.device)
        else:
            c = None
        return {'z': z, 'c': c}

    def bend_model(self, model):
        model.trace("forward", **self.get_inputs())

    @property
    def latent_dim(self):
        return self._model.mapping.z_dim

    @property
    def conditioning_dim(self): 
        return getattr(self._model.mapping, "c_dim", None)

    @property
    def intermediate_latent_dim(self):
        return getattr(self._model.mapping, "w_dim", None)

    @property
    def channels(self): 
        return self._model.synthesis.channels

    @property
    def n_layers(self): 
        return self._model.synthesis.num_layers

    @_overload_module
    def script(self): 
        scripted = super().script()
        scripted.latent_dim = self.latent_dim
        return scripted