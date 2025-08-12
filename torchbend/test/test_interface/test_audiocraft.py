import os
import random
from pathlib import Path
import itertools
import torch
import pytest
import torchaudio
import torchbend as tb

AUDIOCRAFT_AVAILABLE = False
try:
    from torchbend.interfaces.audiocraft_interface import BendedAudioGen, BendedMusicGen
    AUDIOCRAFT_AVAILABLE = True
except ModuleNotFoundError:
    AUDIOCRAFT_AVAILABLE = False

N_BATCHES_TEST = 1
AUDIO_OUT_DIR = (Path(__file__) / ".." / "outs" / "audiocraft").resolve()

def get_test_name(): 
    return os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]


def save_audio_example(out, model, suffix):
    out_dir = AUDIO_OUT_DIR / type(model).__name__ / suffix
    if not out_dir.exists(): os.makedirs(out_dir)
    for i, o in enumerate(out):
        out_path = out_dir / f"{get_test_name()}_{i}.wav"
        torchaudio.save(str(out_path), o.detach().cpu(), sample_rate = model.sample_rate)


@pytest.mark.skipif(not AUDIOCRAFT_AVAILABLE, reason="audiocraft not available")
@pytest.mark.parametrize('model_class, model_args', [(BendedAudioGen, tb.PArgs("facebook/audiogen-medium")),
                                                     (BendedMusicGen, tb.PArgs('facebook/musicgen-small'))])
def test_unconditional_generation(model_class, model_args):
    model = model_class(*model_args, **model_args)
    model.set_generation_params(duration=5.0)
    with torch.no_grad():
        out = model.generate_unconditional(N_BATCHES_TEST, seed=0)
        save_audio_example(out, model, "unconditional")
        prob = tb.bending.BendingParameter('mask', 1.)
        cb = tb.Mask(prob=prob)
        model.bend(cb, r"?.*weight_v")
        out_unmasked = model.generate_unconditional(N_BATCHES_TEST, seed=0)
        prob.set_value(0.)
        out_masked = model.generate_unconditional(N_BATCHES_TEST, seed=0)
        assert bool(tb.compare_outs(out, out_unmasked))
        assert not bool(tb.compare_outs(out, out_masked))

@pytest.mark.skipif(not AUDIOCRAFT_AVAILABLE, reason="audiocraft not available")
@pytest.mark.parametrize('model_class, model_args', [(BendedAudioGen, tb.PArgs("facebook/audiogen-medium")),
                                                     (BendedMusicGen, tb.PArgs('facebook/musicgen-small'))])
@pytest.mark.parametrize('prompt', [['do what you do man']])
def test_prompt_generation(prompt, model_class, model_args):
    model = model_class(*model_args, **model_args)
    with torch.no_grad():
        out = model.generate(prompt)
    save_audio_example(out, model, "prompt")