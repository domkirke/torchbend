import numpy as np
import torch.nn.functional as F
import torch, torch.nn as nn
from einops import rearrange
from sklearn.decomposition import PCA, FastICA


        
@torch.fx.wrap
def get_noise(z: torch.Tensor, full_latent_size: int):
    return torch.randn(
            z.shape[0],
            full_latent_size - z.shape[1],
            z.shape[-1],
    ).type_as(z)


class RAVEProjection(nn.Module):

    def __init__(self, name: str | None = None):
        super().__init__()
        self.name = torch.jit.Attribute(name, str)

    def __repr__(self): 
        return "%s(name=%s)"%(type(self).__name__, self.name.value)
    
    def forward(self, x): 
        return x

    @torch.jit.export
    def inverse(self, x):
        return x



class PCAProjection(RAVEProjection): 
    def __init__(self, name: str | None = None, seed: int = 0):
        super().__init__(name=name)
        self.seed = seed
        self.latent_mean = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.latent_var = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.components = nn.Parameter(torch.tensor(0), requires_grad=False)

    @staticmethod
    def from_params(latent_mean, latent_var, components, name=None):
        self = PCAProjection(name=name)
        self.latent_mean.data = latent_mean
        self.latent_var.data = latent_var
        self.components.data = components
        return self

    @staticmethod
    def from_buffer(buffer, name=None):
        self = PCAProjection(name=name)
        latent_mean, latent_var, components = torch.split(buffer, (1, 1, buffer.shape[0]-2), dim=0)
        self.latent_mean.data = latent_mean[0]
        self.latent_var.data = latent_var[0]
        self.components.data = components
        return self


    def fit(self, z):
        z = rearrange(z, "b c t -> (b t) c")
        z_mean = z.mean(0)
        z = z - z_mean
        
        pca = PCA(z.shape[-1], random_state = self.seed).fit(z)
        components = pca.components_
        components = torch.from_numpy(components)

        var = pca.explained_variance_ / np.sum(pca.explained_variance_)
        var = torch.from_numpy(np.cumsum(var))

        self.latent_mean = z_mean
        self.latent_var = var
        self.components = components
        return components, z_mean, var

    def forward(self, z: torch.Tensor, latent_size: int | None = None):
        z = z - self.latent_mean.unsqueeze(-1).to(z)
        z = F.conv1d(z, self.components.unsqueeze(-1).to(z))
        if latent_size is not None: 
            z = z[:, :latent_size]
        return z

    def inverse(self, z: torch.Tensor, latent_size: int | None = None, temperature: float | None = None):
        proj_dim = self.components.shape[0]
        if temperature is None :
            temperature = 1.
        if z.shape[1] < proj_dim:
            noise = get_noise(z, proj_dim)
            z = torch.cat([z, noise * temperature], 1)
        z = F.conv1d(z, self.components.T.unsqueeze(-1).to(z))
        z = z + self.latent_mean.unsqueeze(-1).to(z)
        return z



class ICAProjection(RAVEProjection): 
    def __init__(self, name: str | None = None, seed: int = 0):
        super().__init__(name=name)
        self.seed = seed
        self.latent_mean = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.components = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.mixing = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.whitening = nn.Parameter(torch.tensor(0), requires_grad=False)

    @staticmethod
    def from_params(latent_mean, components, mixing, whitening, name=None):
        self = ICAProjection(name=name)
        self.latent_mean.data = latent_mean
        self.components.data = components
        self.mixing.data = mixing 
        self.whitening.data = whitening
        return self

    @staticmethod
    def from_buffer(buffer, name=None):
        self = ICAProjection(name=name)
        if buffer.shape[0] == 4:
            latent_mean, components, mixing, whitening = torch.split(buffer, (1, 1, 1, 1), dim=0)
        else:
            latent_mean = torch.Tensor([[0.]])
            components, mixing, whitening = torch.split(buffer, (1, 1, 1), dim=0)
        self.latent_mean.data = latent_mean[0]
        self.components.data = components[0]
        self.mixing.data = mixing[0]
        self.whitening.data = whitening[0]
        return self


    def fit(self, z):
        z = rearrange(z, "b c t -> (b t) c")
        
        pca = FastICA(z.shape[-1], whiten='unit-variance', random_state=self.seed).fit(z)
        components = pca.components_
        mixing = pca.mixing_
        mean = pca.mean_

        mean = torch.from_numpy(mean)[None]
        components = torch.from_numpy(components)
        mixing = torch.from_numpy(mixing)
        whitening = torch.from_numpy(pca.whitening_)

        return torch.stack([mean, components, mixing, whitening])

    def forward(self, z: torch.Tensor, latent_size: int | None = None):
        z = z - self.latent_mean.unsqueeze(-1).to(z)
        z = torch.bmm(z.permute(0, 2, 1), self.components.unsqueeze(0).float()).permute(0, 2, 1)
        if latent_size is not None: 
            z = z[:, :latent_size]
        return z

    def inverse(self, z: torch.Tensor, latent_size: int | None = None, temperature: float | None = None):
        proj_dim = self.components.shape[0]
        if temperature is None :
            temperature = 1.
        if z.shape[1] < proj_dim:
            noise = get_noise(z, proj_dim)
            z = torch.cat([z, noise * temperature], 1)
        z = torch.bmm(z.permute(0, 2, 1), self.whitening.unsqueeze(0).float()).permute(0, 2, 1)
        z = z + self.latent_mean.unsqueeze(-1).to(z)
        return z

    

# LATENT MAPPER

class Mapper(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=300, hidden_layers=3):
        super().__init__()
        layer_shapes = [input_size] + [hidden_size] * hidden_layers + [output_size]
        layers = []
        for i in range(len(layer_shapes) - 1):
            layers.append(nn.Linear(layer_shapes[i], layer_shapes[i+1]))
            if i != len(layer_shapes) - 2:
                layers.append(nn.BatchNorm1d(layer_shapes[i+1]))
                layers.append(nn.ReLU())
        self._module = nn.Sequential(*layers)

    def forward(self, x):
        return self._module(x)



class MapperProjection(RAVEProjection):
    def __init__(self, mapper_file, name: str | None = None):
        super().__init__(name=name)
        self.encoder = Mapper(128, 8)
        self.decoder = Mapper(8, 128)
        state_dict = torch.load(mapper_file, map_location="cpu")['state_dict']
        encoder_state_dict = {}
        decoder_state_dict = {}
        for k, v in state_dict.items(): 
            param_name = k.split('.')
            module, param_name = param_name[0], ".".join(param_name[1:])
            if module == "encoder": encoder_state_dict[param_name] = v
            if module == "decoder": decoder_state_dict[param_name] = v
        self.encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)

    def forward(self, z: torch.Tensor, latent_size: int | None = None):
        n_batch, n_channels, n_steps = z.shape
        z = self.encoder(z.permute(0, 2, 1).reshape(-1, n_channels))
        z = z.reshape(n_batch, n_steps, -1).permute(0, 2, 1)
        if latent_size is not None: 
            if latent_size < z.shape[1]:
                z = z[:, :latent_size]
            elif latent_size > z.shape[1]:
                missing_dims = latent_size - n_channels
                z = torch.cat([z, torch.zeros(z.shape[0], missing_dims, z.shape[2])], dim=1)
        return z

    def inverse(self, z: torch.Tensor, latent_size: int | None = None, temperature: float | None = None):
        n_batch, n_channels, n_steps = z.shape
        if temperature is None: temperature = 0.
        z = z + torch.randn_like(z) * temperature
        z = self.decoder(z.permute(0, 2, 1).reshape(-1, n_channels))
        z = z.reshape(n_batch, n_steps, -1).permute(0, 2, 1)
        if latent_size is not None: 
            if latent_size < z.shape[1]:
                missing_dims = latent_size - n_channels
                z = torch.cat([z, torch.zeros(z.shape[0], missing_dims, z.shape[2])], dim=1)
            elif latent_size > z.shape[1]:
                z = z[:, :latent_size]
        return z

    
