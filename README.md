# torchbend

`torchbend` is a library grounded on `torch.fx` focused on generative neural networks analysis and creative bending. This library allows you to:

- [✔︎] extend the tracing abilities of `torch.fx` with augmented parsers and proxies
    - dynamic parsing (wrapping un-traceable functions, shape propagation)
    - tracing torch distributions (currently implemented : `Bernoulli`, `Normal`, `Categorical`)
- [✔︎] easily parse and analyze model's graphs 
- [︎✔︎] bend model's weights and activations
- [✕︎] adapt the library to specific generative models, and provide handy interfaces for python notebooks
    - [✕︎] handful classes for image, text, and sound
    - [✕︎] panel implementation for real-time bending
    - [✕︎] model analysis UI
- [✕︎] script generative models with JIT additional bending inputs (for use in [nn~] for example)

`torchbend` provides end-to-end examples and interfaces for the following libraries:

| Model                | Weights | Activation | Script |
| :------------------- | :-----: | :--------: | :----: |
| **Audio** | | | |
| vschaos              | ◇       | ◇          | ◇      |
| RAVE                 | ✔︎        | ✔︎          | ◇      |
| MusicGen             | ✔︎    | ✕︎       | ✕︎   |
| AudioGen             | ✔︎    | ✕︎       | ✕︎   |
| **Image** | | | |
| StyleGAN3            | ◇       | ✕︎       | ✕︎   |
| StableDiffusion      | ◇       | ✕︎       | ✕︎   |
| **Text**                 |         |            |        |
| GPT-2                | ◇       | ✕︎       | ✕︎   |
| Llama                | ◇       | ✕︎       | ✕︎   |

<small>✔︎: tested ; ✕︎ : not working ; ◇ : to try out</small>


## Parse and analyse model's graphs

```python
import torch, torchbend

# make dumb module to test
module = torchbend.TestModule()
module_in = torch.randn(1, 1, 512)

# init BendedModule with the module, and trace target functions with given inputs
bended_module = torchbend.BendedModule(module)
bended_module.trace("forward", x=module_in)

# print weights and activations
print("weights : ")
bended_module.print_weights()
print("activations : ")
bended_module.print_activations()

outs = bended_module.get_activations('pre_conv', x=module_in)
print("pre_conv activation min and max : ", outs['pre_conv'].min(), outs['pre_conv'].max())
```


## Bending weights and activations

```python
import torch, torchbend

# make dumb module to test
module = torchbend.TestModule()
module_in = torch.randn(1, 1, 512)

# init BendedModule with the module, and trace target functions with given inputs
bended_module = torchbend.BendedModule(module)
bended_module.trace("forward", x=module_in)

# bend target weights and make forward pass
bended_module.bend(torchbend.bending.Mask(0.), "pre_conv.weight")
outs = bended_module(x=module_in)
print("pre_conv bended weight std : ", bended_module.bended_state_dict()['pre_conv.weight'].std())
print("pre_conv original weight std : ", bended_module.module.state_dict()['pre_conv.weight'].std())

# reset bending
bended_module.reset()

# bend target activation 
bended_module.bend(torchbend.bending.Mask(0.), "pre_conv")
outs = bended_module.get_activations('pre_conv', x=module_in, bended=True)
print("pre_conv activation min and max : ", outs['pre_conv'].min(), outs['pre_conv'].max())
```

## Bending MusicGen

```python
import torchaudio
import torchbend as tb
from audiocraft.models import MusicGen

card = 'facebook/musicgen-small'
model = MusicGen.get_pretrained(card)
model.set_generation_params(duration=4) 

model.compression_model = tb.BendedModule(model.compression_model)
model.compression_model.print_weights("decoder.*", "descriptions/"+card.split('/')[-1])
model.compression_model.bend_(tb.Mask(0.7), 'decoder.model.\d+.lstm.weight_hh_l.', verbose=True)

out = model.generate(['monkey kick'])
torchaudio.save('test.wav', out[0], sample_rate=44100)
```