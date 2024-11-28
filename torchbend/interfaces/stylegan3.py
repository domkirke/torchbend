import torchbend as tb
from .base import Interface
import sys
import numpy
import os, torch

class BendedStyleGAN3(Interface):
    def __init__(self, pretrained, *args, cache_dir='.cache/stylegan3', device=torch.device('cpu'), **kwargs):
        sys.path.append(cache_dir)
        pretrained = numpy.load(pretrained, allow_pickle=True)
        super(BendedStyleGAN3, self).__init__(pretrained['G'].to(device))

    def _bend_model(self, model):
        self._model = tb.BendedModule(model)
        self._import_methods(self._model)
    
    @property
    def latent_dim(self):
        return self._model.mapping.z_dim
