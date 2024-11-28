import torch
from .normal import Normal

def isotropic_gaussian(shape, device=torch.device('cpu')):
    return  Normal(torch.zeros(*shape, device=device), torch.ones(*shape, device=device))
