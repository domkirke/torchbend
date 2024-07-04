import os
import torchaudio
import torch, audiocraft, os, random
from audiocraft.models import MusicGen, AudioGen
from .base import Interface
from ..tracing import BendingWrapper
from .utils import get_random_hash



class BendedMusicGen(Interface):

    _imported_callbacks_ = ['generate', 'generate_unconditional', 'generate_audio']

    def __init__(self, *args, cache_dir=None, **kwargs):
        if cache_dir is not None:
            os.environ['AUDIOCRAFT_CACHE_DIR'] = cache_dir
        # import model
        model = self.get_pretrained(*args, **kwargs)
        # init interfaces
        super(BendedMusicGen, self).__init__(model)

    def _bend_model(self, model):
        self._model = BendingWrapper(model)
        self._import_wrapper_methods(model)

    def get_pretrained(self, *args, **kwargs):
        model = MusicGen.get_pretrained(*args, **kwargs)

        # flattened temporal modules
        for i, t in enumerate(model.compression_model.decoder._modules['model']):
            if isinstance(t, audiocraft.modules.lstm.StreamableLSTM):
                t.lstm.flatten_parameters()

        # set default generation params
        model.set_generation_params(
            duration=30,
            cfg_coef=3.,
            top_k=250,
            top_p=0.,
            temperature=1.,
            use_sampling=True,
            extend_stride=18
        )

        return model

    @property
    def sample_rate(self):
        return self._model.compression_model.sample_rate

    def get_filename_from_args(prompt, callbacks=None):
        name = prompt.replace(" ", "_")
        return f"{name}_{get_random_hash(10)}" 

    def set_generation_params(self, **kwargs):
        self.model.set_generation_params(**kwargs)



class BendedAudioGen(BendedMusicGen):

    _imported_callbacks_ = ['generate', 'generate_continuation']

    def _bend_model(self, model):
        self._model = BendingWrapper(model)
        self._import_wrapper_methods(self._model)

    def get_pretrained(self, *args, **kwargs):
        model = AudioGen.get_pretrained(*args, **kwargs)
        model.set_generation_params(
            use_sampling = True, top_k = 250,
            top_p = 0.0, temperature = 1.0,
            duration = 10.0, cfg_coef = 3.0,
            two_step_cfg = False, extend_stride = 2
        )
        return model


__all__ = ['BendedMusicGen', 'BendedAudioGen']

'''


torch.default_generator.manual_seed(438)

out_dir = "generations"
os.makedirs(out_dir, exist_ok=True)

model.compression_model = tb.BendedModule(model.compression_model)
model.compression_model.print_weights("decoder.*", "descriptions/"+card.split('/')[-1])
out_dir_sess = os.path.join(out_dir, get_random_hash(6))
os.makedirs(out_dir_sess, exist_ok=True)

prompts = [
    "monkey kick",
    "heavy funk drums with frenetic bass",
    "bassline for dubstep neurofunk", 
    "heavy and dark keyboards"
]

hacks = [
    # (tb.Bias(-0.2), 'decoder.model...lstm.weight_.h_l.'),
    # (tb.Bias(-0.1), 'decoder.model...lstm.weight_.h_l.'),
    # (tb.Bias(-0.3), 'decoder.model...lstm.weight_.h_l.'),
    # (callback, 'decoder.model.4.block.\d+.conv.conv.*'),
    # (callback, 'decoder.model.7.block.\d+.conv.conv.*'),
    # (callback, 'decoder.model.10.block.\d+.conv.conv.*'),
    # (callback, 'decoder.model.13.block.\d+.conv.conv.*')
]
for i in range(20): hacks.append((tb.Bias(-i/10.), 'decoder.model...lstm.weight_.h_l.'))

# original
model.compression_model.reset()

outs, tokens = model.generate(prompts, return_tokens=True)
for i, out in enumerate(outs):
    filepath = os.path.join(out_dir_sess, get_filename_from_args(prompts[i]))
    torchaudio.save(filepath + ".wav", out.cpu(), sample_rate=model.compression_model.sample_rate)

for i, current_hack in enumerate(hacks):
    model.compression_model.reset()
    model.compression_model.bend_(*current_hack, verbose=True)
    outs = model.generate_audio(tokens)
    for j, out in enumerate(outs):
        filepath = os.path.join(out_dir_sess, get_filename_from_args(prompts[j]))
        torchaudio.save(filepath + "_" + str(i) + ".wav", out.cpu(), sample_rate=model.compression_model.sample_rate)

print("all exported at %s"%out_dir_sess)



# tb.wrapmethod(audiocraft.models.encodec.EncodecModel, "decode_latent")
# model = tb.BendingWrapper(model, ['compression_model'])
# token_file = 'tokens.pt'
# if not os.path.isfile(token_file):
#     descriptions = [None]
#     attributes, prompt_tokens = model._prepare_tokens_and_attributes(descriptions, None)
#     tokens = model._generate_tokens(attributes, prompt_tokens, False)
#     torch.save(tokens, token_file)
# else:
#     tokens = torch.load(token_file)

# decoder = model.compression_model
# tb.wrapmodule(decoder.quantizer)


# model =  BendingWrapper(MusicGen.get_pretrained('facebook/musicgen-small'))
# model.lm.trace('forward', prompt="caca")

# how to use forward instead of generate for lm? (streaming would mess the scripting)
# lm = model.lm


# try:
#     lm.trace('generate', prompt=prompt)
# except RecursionError:
#     print('max recursion reached')

# model.compression_model.print_params()
# out = model.generate_unconditional(1)
# torchaudio.save('test.wav', out[0], sample_rate=44100)

'''