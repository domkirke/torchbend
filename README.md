# torchbend

Welcome to `torchbend`, a high-level framework for dissecting, analyzing and bending machine learning models programmed with [Pytorch](https://pytorch.org/docs/stable/index.html). This framework extends `torch.fx` and proposes convenient methods to target certain activations of a network, bend its parameters or internal values, and easily perform some [active divergence](https://arxiv.org/pdf/2107.05599) techniques to unbound a co-creative approache to generative ML.

### Warning
`torchbend` is still a beta library, and is likely to change a lot in the future months. Do not hestitate to add issues on github, but sustainability can not be ensured before version 1 release!

<!--
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
-->

## Installation 

*Installing pytorch.* This repository assumes that you have a dedicated python environement (for example through [miniconda](https://docs.anaconda.com/miniconda/install/)) with an installed torch version. If you don't, find the appropriate version of PyTorch on the [official website](https://pytorch.org/). 

*Installing torchbend*. Installing torchbend requires so far to clone the repository and install the dependencies with `pip` : 
```sh
git clone https://github.com/acids-ircam/torchbend.git
cd torchbend
pip install .
```

If the environment targets to bend some of the interfaces, additional requirements may be required for specific interfaces as RAVE, that can be installed by precising extra configurations : 
```sh
git clone https://github.com/acids-ircam/torchbend.git
cd torchbend
pip install ".[rave]"
pip install "git+https://github.com/acids-ircam/RAVE.git" --no-deps
```
