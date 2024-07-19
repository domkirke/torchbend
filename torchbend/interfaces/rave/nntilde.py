
import torch
import torch.nn as nn
import nn_tilde

class ScriptableRAVE(nn_tilde.Module):

    def __init__(self, model):
        super().__init__()
        self._import_model(model)
        self._register_methods(model)
        self._import_bending(model) 

    def _import_model(self, model):
        # model.trace("encode", x=torch.randn(1, 1, 8192), _proxied_buffers=['latent_mean', 'latent_pca'])
        # model.trace("decode", z=torch.randn(1, 16, 8), _proxied_buffers=['latent_mean', 'latent_pca', 'decode_params'])
        # model.trace("forward", x=torch.randn(1, 1, 8192), _proxied_buffers=['latent_mean', 'latent_pca', 'decode_params'])
        bended_module = model.bend_module()

        self._encode = model.graph_module('encode', module=bended_module)
        self._decode = model.graph_module('decode', module=bended_module)
        self._forward = model.graph_module('forward', module=bended_module)
        self._bended_modules = [self._encode, self._decode, self._forward]

    def _full_param_dict(self):
        param_dict = {}
        for module in self._bended_modules:
            for k, v in dict(module.named_parameters()).items():
                if k in param_dict:
                    if id(param_dict[k]) != id(v):
                        print('[Warning] param %s does not coincide between modules')
                else:
                    param_dict[k] = v
        return param_dict

    def _import_bending_ops(self, model):
        self._controllables = nn.ModuleList(model.controllables.values())
        self._bending_callbacks = nn.ModuleList(model._bending_callbacks)
        self._controllables_hash = {}
        for v in self._controllables:
            for i, b in enumerate(self._bending_callbacks):
                if v in b:
                    self._controllables_hash[v.name] = self._controllables_hash.get(v.name, []) + [i]
            self.register_attribute(v.name, float(v.value))
                
    def _update_bended_weights(self, model):
        param_dict = self._full_param_dict()
        model_param_dict = dict(model.named_parameters())
        for param, cb_list in model.bended_params.items():
            if param not in param_dict:
                print('[Warning] Bended parameter %s not found in current module.'%param)
                continue
            for cb in cb_list:
                # cb.update_parameter(model.get_parameter(param), param_dict[param])
                cb.update_parameter(model_param_dict[param], param_dict[param])

    def _update_bended_activations(self, model):
        pass
        # for bended_act in model._bended_activations:
        #     print()

    def _import_bending(self, model):
        self._import_bending_ops(model)
        self._update_bended_weights(model)
        self._update_bended_activations(model)

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
        raise ModuleNotFoundError("No bending control named %s in model %s"%(name, self))

    @torch.jit.export
    def _set_bending_control(self, name: str, value: torch.Tensor) -> None:
        """set a bending control with name and value"""
        for v in self._controllables:
            if v.name == name:
                v.set_value(value)
        self._update_weights(name)


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