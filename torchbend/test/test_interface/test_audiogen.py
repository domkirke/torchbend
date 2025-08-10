import os
import random
from pathlib import Path
import itertools
import torch
import pytest
import torchbend as tb

AUDIOCRAFT_AVAILABLE = False
try:
    from torchbend.interfaces.audiocraft_interface import BendedAudioGen, BendedMusicGen
    AUDIOCRAFT_AVAILABLE = True
except ModuleNotFoundError:
    AUDIOCRAFT_AVAILABLE = False

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


N_SAMPLES_TEST = 131072

@pytest.mark.skipif(not AUDIOCRAFT_AVAILABLE, reason="audiocraft not available")
@pytest.mark.parametrize('model_class, model_args', [(BendedAudioGen, tb.PArgs("facebook/audiogen-medium")),
                                                     (BendedMusicGen, tb.PArgs('facebook/musicgen-small'))])
def test_unconditional_generation(model_class, model_args):
    model = model_class(*model_args, **model_args).to(device)
    out = model.generate_unconditional(N_SAMPLES_TEST)

