import torch, torch.nn as nn

class Foo(nn.Module):
    def __init__(self, nlayers=3):
        super().__init__()
        self.nlayers = nlayers
        self.pre_conv = nn.Conv1d(1, 16, 1)
        self.conv_modules = nn.ModuleList([nn.Conv1d(16, 16, 3) for _ in range(nlayers)])
        self.activations = nn.ModuleList([nn.Tanh() for _ in range(nlayers)])

    def forward(self, x):
        out = self.pre_conv(x)
        for i in range(self.nlayers):
            out = self.conv_modules[i](out)
            out = self.activations[i](out)
        return torch.distributions.Normal(out, torch.ones_like(out))

    def forward_nodist(self, x):
        out = self.pre_conv(x)
        for i in range(self.nlayers):
            out = self.conv_modules[i](out)
            out = self.activations[i](out)
        return out

__all__ = ['Foo']