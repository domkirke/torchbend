
import torch
import torch.nn as nn
import nn_tilde

class ScriptableRAVE(nn_tilde.Module):
    def __init__(self, model):
        super().__init__()
        self._init_controllables(model) 
        model.trace("encode", x=torch.randn(1, 1, 8192), _proxied_buffers=['latent_mean', 'latent_pca'])
        model.trace("decode", z=torch.randn(1, 16, 8), _proxied_buffers=['latent_mean', 'latent_pca', 'decode_params'])
        model.trace("forward", x=torch.randn(1, 1, 8192), _proxied_buffers=['latent_mean', 'latent_pca', 'decode_params'])
        self._encode = model.graph_module('encode')
        self._decode = model.graph_module('decode')
        self._forward = model.graph_module('forward')
        self.register_methods(model)

    def _init_controllables(self, model):
        self._controllables = nn.ModuleList(model.controllables.values())
        self._bending_callbacks = nn.ModuleList(model._bending_callbacks)
        self._controllables_hash = {}
        for v in self._controllables:
            for i, b in enumerate(self._bending_callbacks):
                if v in b:
                    self._controllables_hash[v.name] = self._controllables_hash.get(v.name, []) + [i]

    def _update_weights(self, name: str):
        with torch.no_grad():
            callbacks = self._controllables_hash[name]
            for i, c in enumerate(self._bending_callbacks):
                for j in callbacks:
                    if i == j: c.apply()

    @torch.jit.export
    def _get_bending_control(self, name: str) -> torch.Tensor:
        """returns value of a bending control by name"""
        # grrr
        for i, v in enumerate(self._controllables):
            if v.name == name:
                return v.value.data
        raise RuntimeError()

    @torch.jit.export
    def _set_bending_control(self, name: str, value: torch.Tensor) -> None:
        """set a bending control with name and value"""
        for v in self._controllables:
            if v.name == name:
                v.set_value(value)
        self._update_weights(name)


    def register_methods(self, model):
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
            output_labels=['(signal) Channel %d'%d for d in range(1, model.n_channels+1)]
        )

        self.register_method(
            "forward",
            in_channels=model.n_channels,
            in_ratio=1,
            out_channels=model.n_channels,
            out_ratio=1,
            input_labels=['(signal) Channel %d'%d for d in range(1, model.n_channels + 1)],
            output_labels=['(signal) Channel %d'%d for d in range(1, model.n_channels+1)]
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