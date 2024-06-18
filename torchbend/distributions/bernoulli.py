import torch, torch.nn as nn
from collections import namedtuple
from typing import Union, Dict, Tuple
from torch.nn.functional import binary_cross_entropy_with_logits, softmax
from .base import Distribution

__all__ = ["Bernoulli"]

def logits_to_probs(logits, is_binary: bool=False):
    if is_binary:
        return torch.sigmoid(logits)
    return softmax(logits, dim=-1)

def probs_to_logits(probs, is_binary: bool=False, eps: float=1.e-7):
    ps_clamped = probs.clamp(min=eps, max=1 - eps)
    if is_binary:
        return torch.log(ps_clamped) - torch.log1p(-ps_clamped)
    return torch.log(ps_clamped)


class Bernoulli(Distribution):
    def __init__(self, probs: torch.Tensor):
        self.probs: torch.Tensor = probs
        self.logits = probs_to_logits(self.probs)
        self._batch_shape = probs.size() 
        self._event_shape = torch.Size([0])

    def as_tuple(self) -> Tuple[torch.Tensor]:
        return (self.probs,)
 
    @property 
    def mean(self) -> torch.Tensor:
        return self.probs

    @property
    def mode(self):
        mode = (self.probs >= 0.5).to(self.probs)
        mode[self.probs == 0.5] = torch.nan
        return mode

    @property
    def variance(self):
        return self.probs * (1 - self.probs)

    def sample(self):
        with torch.no_grad():
            return torch.bernoulli(self.probs)

    def log_prob(self, value):
        return -binary_cross_entropy_with_logits(self.logits, value, reduction='none')

    @property
    def _natural_params(self):
        return (torch.log(self.probs / (1 - self.probs)), )

    def entropy(self):
        return binary_cross_entropy_with_logits(self.logits, self.probs, reduction='none')