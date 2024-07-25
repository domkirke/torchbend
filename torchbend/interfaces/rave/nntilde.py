from typing import Dict, List
import re
from ...bending.parameter import BendingParameter
from ...tracing.nntilde import BendableNNTildeModule
import torch
import torch.nn as nn


def _zero_cache(module, filters=[r".*cache", r".*pad"]):
    for k, v in module.named_buffers():
        if True in [re.match(f, k) is not None for f in filters]:
            v.data.zero_()

class ScriptableRAVE(BendableNNTildeModule):

    scripted_methods = ['encode', 'decode', 'forward']

    def _import_model(self, model):
        super()._import_model(model)
        for mod in self._bended_modules:
            _zero_cache(mod)

    def _register_methods(self, model):
        self.register_method(
            "encode",
            in_channels=model.n_channels,
            in_ratio=1,
            out_channels=model.latent_size,
            out_ratio=model.encode_params[3],
            input_labels=['(signal) Channel %d'%d for d in range(1, model.n_channels+1)],
            output_labels=[
                f'(signal) Latent dimension {i + 1}'
                for i in range(model.latent_size)
            ],
            test_method=False
        )
        self.register_method(
            "decode",
            in_channels=model.latent_size,
            in_ratio=model.encode_params[3],
            out_channels=model.n_channels,
            out_ratio=1,
            input_labels=[
                f'(signal) Latent dimension {i+1}'
                for i in range(model.latent_size)
            ],
            output_labels=['(signal) Channel %d'%d for d in range(1, model.n_channels+1)],
            test_method=False
        )

        self.register_method(
            "forward",
            in_channels=model.n_channels,
            in_ratio=1,
            out_channels=model.n_channels,
            out_ratio=1,
            input_labels=['(signal) Channel %d'%d for d in range(1, model.n_channels + 1)],
            output_labels=['(signal) Channel %d'%d for d in range(1, model.n_channels+1)],
            test_method=False
        )

    @torch.jit.export
    def encode(self, x):
        return self._encode(x)

    @torch.jit.export
    def decode(self, z):
        return self._decode(z)

    @torch.jit.export
    def forward(self, x):
        return self._forward(x)