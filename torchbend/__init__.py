import torch

DEBUG = False
def debug(val):
    global DEBUG
    DEBUG = bool(val)

def log_warning(*args):
    print('[Warning]', *args)

def log_error(*args):
    print('[Error]', *args)

import enum
class TorchbendOutput(enum.Enum):
    RAW = 0
    NOTEBOOK = 1
TB_OUTPUT = TorchbendOutput.RAW
def set_output(output):
    assert isinstance(output, (str, TorchbendOutput))
    if isinstance(output, str):
        output = getattr(TorchbendOutput, output.upper())
    global TB_OUTPUT
    TB_OUTPUT = output
def get_output():
    global TB_OUTPUT
    return TB_OUTPUT

Model = torch.nn.Module
from .utils import *
from . import distributions
from .bending import *
from .tracing import *
from .tracing.utils import compare_outs, compare_state_dict_tensors