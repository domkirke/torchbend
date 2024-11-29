import torch
import torch.distributions as tdist
from .base import Distribution
from .bernoulli import Bernoulli
from .categorical import Categorical
from .uniform import Uniform
from .normal import Normal


def convert_to_torch(distribution):
    if isinstance(distribution, tdist.Distribution):
        return distribution
    elif isinstance(distribution, Bernoulli):
        return tdist.Bernoulli(probs=distribution.probs)
    elif isinstance(distribution, Categorical):
        return tdist.Categorical(logits=distribution.logits)
    elif isinstance(distribution, Normal):
        return tdist.Normal(distribution.mean, distribution.stddev)
    elif isinstance(distribution, Uniform):
        return tdist.Uniform(distribution.low, distribution.high)
    else:
        raise NotImplementedError

def convert_from_torch(distribution):
    if isinstance(distribution, Distribution):
        return distribution
    if isinstance(distribution, tdist.Bernoulli):
        return Bernoulli(probs=distribution.probs)
    elif isinstance(distribution, tdist.Categorical):
        return Categorical(logits=distribution.logits)
    elif isinstance(distribution, tdist.Normal):
        return Normal(distribution.mean, distribution.stddev)
    elif isinstance(distribution, tdist.Uniform):
        return Uniform(distribution.low, distribution.high)
    else:
        raise NotImplementedError

dist_hash = {'Normal': Normal, 'Bernoulli': Bernoulli, 'Categorical': Categorical}
    
def checkdist(obj):
    if obj is None:
        return obj
    elif isinstance(obj, str):
        return dist_hash[obj]
    elif issubclass(obj, Distribution):
        return obj
    else:
        raise TypeError('obj %s does not seem to be a distribution')


def trace_distribution(distribution, name="", scatter_dim=False):
    if name != "":
        name = name + "_"
    if isinstance(distribution, Normal):
        if scatter_dim:
            return {**{f'{name}mean/dim_{i}': distribution.mean[..., i] for i in range(distribution.mean.shape[-1])},
                    **{f'{name}std/dim_{i}': distribution.stddev[..., i] for i in range(distribution.stddev.shape[-1])}}
        else:
            return {f"{name}mean": distribution.mean, f"{name}std": distribution.stddev}
    elif isinstance(distribution, (Bernoulli, Categorical)):
        if scatter_dim:
            return {**{f'{name}probs/dim_{i}': distribution.probs[..., i] for i in range(distribution.probs.shape[-1])}}
        else:
            return {f"{name}probs": distribution.probs}
    elif torch.is_tensor(distribution):
        return {f"{name}": distribution}
