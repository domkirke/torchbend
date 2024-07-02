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
        return tdist.Bernoulli(distribution.probs)
    elif isinstance(distribution, Categorical):
        return tdist.Categorical(logits=distribution.logits)
    elif isinstance(distribution, Normal):
        return tdist.Normal(distribution.mean, distribution.stddev)
    else:
        raise NotImplementedError

def convert_from_torch(distribution):
    if isinstance(distribution, Distribution):
        return distribution
    if isinstance(distribution, tdist.Bernoulli):
        return Bernoulli(distribution.probs)
    elif isinstance(distribution, tdist.Categorical):
        return Categorical(logits=distribution.logits)
    elif isinstance(distribution, tdist.Normal):
        return Normal(distribution.mean, distribution.stddev)
    else:
        raise NotImplementedError