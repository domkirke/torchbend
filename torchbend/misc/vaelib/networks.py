import os
import numpy as np
import torch 
import torch.nn as nn 
from tqdm import tqdm

from torchbend import mark


class Encoder(nn.Module) :
    def __init__(
        self, in_size, in_channel, kernels, channels, hdim, nmlp, zdim
        ) :
        super().__init__()
        self.net = nn.Sequential()
        self.sizes = [in_size]
        for n,kernel in enumerate(kernels) :
            in_chan = in_channel if n==0 else channels[n-1]
            out_chan = channels[n]
            stride = (kernel-1)//2
            padding = (kernel-1)//2
            self.net.add_module(f'conv{n}', nn.Conv2d(in_chan, 
                                                   out_chan, 
                                                   kernel, 
                                                   stride,
                                                   padding))
            self.net.add_module(f'bn{n}', mark(nn.BatchNorm2d(out_chan), name="encoder"))
            self.net.add_module(f'act{n}', nn.ReLU())
            self.net.add_module(f'dr{n}', nn.Dropout2d(0.2))
            h_out = ((self.sizes[-1][0]+2*padding-(kernel-1)-1)//stride)+1
            w_out = ((self.sizes[-1][1]+2*padding-(kernel-1)-1)//stride)+1
            self.sizes.append([h_out, w_out])
        self.net.add_module('flatten', nn.Flatten())
        
        for n in range(nmlp) :
            in_feat = self.sizes[-1][0]*self.sizes[-1][-1]*channels[-1] if n==0 else hdim
            self.net.add_module(f'fc{n}',nn.Linear(in_feat, hdim)) 
            self.net.add_module(f'bn{n+len(kernels)}', nn.BatchNorm1d(hdim))
            self.net.add_module(f'act{n+len(kernels)}', nn.ReLU())
        
        self.mu = nn.Linear(hdim, zdim)
        self.log_var = nn.Linear(hdim, zdim)
        self.softplus = nn.Softplus()
        
    def forward(self, x) :
        x = x.unsqueeze(1) if len(x.shape) < 4 else x
        x = self.net(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, self.softplus(log_var)+1e-4
    
            
class Decoder(nn.Module) :
    def __init__(
        self, conv_sizes, in_channel, kernels, channels, hdim, nmlp, zdim
        ) :       
        super().__init__()
        self.net = nn.Sequential()
        for n in range(nmlp) :
            in_feat = zdim if n == 0 else hdim
            out_feat = channels[-1]*np.prod(conv_sizes[-1]) if n==nmlp-1 else hdim
            self.net.add_module(f'fc{n}',nn.Linear(in_feat, out_feat))
            self.net.add_module(f'bn{n}', nn.BatchNorm1d(out_feat))
            self.net.add_module(f'act{n}', nn.ReLU())
        self.net.append(nn.Unflatten(dim=1, 
                                     unflattened_size=(channels[-1], conv_sizes[-1][0], 
                                                       conv_sizes[-1][1])))
        
        for n in range(len(kernels)) :
            kernel = kernels[-n-1]
            stride = (kernel-1)//2
            padding = (kernel-1)//2
            size = conv_sizes[-n-2]
            out_pad = ((size[0]-1) % stride,
                       (size[1]-1) % stride)
            in_chan = channels[-n-1]
            out_chan = in_channel if n == len(kernels)-1 else channels[-n-2]
            self.net.add_module(f'convt{n}', nn.ConvTranspose2d(in_chan, 
                                                                out_chan,
                                                                kernel,
                                                                stride,
                                                                padding,
                                                                out_pad))
            self.net.add_module(f'bn{n+nmlp}', mark(nn.BatchNorm2d(out_chan), name="decoder"))
            if n < len(kernels)-1 :
                self.net.add_module(f'act{n+nmlp}', nn.ReLU())
                self.net.add_module(f'dr{n}', nn.Dropout2d(0.2))
        self.net.add_module('final', nn.Sigmoid())
        
    def forward(self, z) :
        return self.net(z)  


class VAE(nn.Module) :
    def __init__(self, encoder, decoder) :
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def encode(self, x) :
        return self.encoder(x)
    
    def decode(self, z) :
        return self.decoder(z)
    
    def reparametrize(self, mu, var) :
        std = torch.sqrt(var)
        eps = torch.randn_like(var)
        z = std*eps+mu
        kl = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var, dim=-1).mean(dim=0)
        return z, kl
    
    def forward(self, x) :
        mu, logvar = self.encode(x)
        z, kl = self.reparametrize(mu, torch.exp(logvar))
        xhat = self.decode(z)
        return xhat
    
    def rec_loss(self, x, xhat) :
        crit = nn.MSELoss(reduction='none')
        return crit(x, xhat).mean(dim=0).sum()   
    
    def _init_optimizer(self, lr, betas=(0.5,0.999)) :
        self.opt = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
    
    def train_step(self, x) :
        self.encoder.train()
        self.decoder.train()
        xhat, z, kl = self(x)
        rec_loss = self.rec_loss(x, xhat)
        tot_loss = rec_loss + .2*kl
        self.opt.zero_grad()
        tot_loss.backward()
        self.opt.step()
    
    def _init_weights(self) :
        for mod in self.modules() :
            if mod.__class__ in [nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                                 nn.ConvTranspose1d, nn.ConvTranspose2d ,nn.ConvTranspose3d]:
                nn.init.xavier_normal_(mod.weight.data)
                if mod.bias is not None:
                    nn.init.normal_(mod.bias.data)
            elif mod.__class__ in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
                nn.init.normal_(mod.weight.data, mean=1, std=0.02)
                nn.init.constant_(mod.bias.data, 0)
            elif mod.__class__ in [nn.Linear]:
                nn.init.xavier_normal_(mod.weight.data)
                nn.init.normal_(mod.bias.data) 
    
    def sample_from_prior(self,n=1, device='cpu') :
        z = torch.randn((n, self.decoder.zdim)).to(device)
        return self.decode(z)

    def fit(self, 
            trainloader,  
            lr, 
            tot_step,
            device, 
            save_path=None) :
        
        self._init_weights()
        self._init_optimizer(lr)
        epochs = tot_step//len(trainloader)
        
        for epoch in range(epochs) :
            for i,(x,_) in enumerate(tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')) :
                x = x.to(device)
                cur_step = i+epoch*len(trainloader)
                self.train_step(x)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.state_dict(), save_path+'/final.ckpt')

def make_mnist_vae():
    in_size = [28, 28]
    in_channels = 1
    kernels = [5, 5, 5, 5]
    channels = [16, 32, 64, 128]
    hdim = 256
    nmlp = 2
    zdim = 64
    encoder = Encoder(in_size, in_channels, kernels, channels, hdim, nmlp, zdim)
    out_sizes = encoder.sizes
    decoder = Decoder(out_sizes, in_channels, kernels, channels, hdim, nmlp, zdim)
    vae = VAE(encoder, decoder)
    vae.eval()
    return vae

def load_mnist_vae(checkpoint_path=None):
    checkpoint_path = checkpoint_path or os.path.join(os.path.dirname(__file__), "mnist_vae", "final.ckpt")
    model = make_mnist_vae()
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model