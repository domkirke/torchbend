import torch

DEBUG = False
def debug(val):
    global DEBUG
    DEBUG = bool(val)

Model = torch.nn.Module
from .utils import *
from . import distributions
from .bending import *
from .tracing import *
from .tracing.utils import compare_outs, compare_state_dict_tensors