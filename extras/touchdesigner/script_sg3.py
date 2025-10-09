import sys
from absl import app, logging, flags
import torch
import pickle
from pathlib import Path

orig_path = str(Path(__file__).parent.parent.parent.absolute())
sys.path.append(orig_path)
import torchbend as tb
from torchbend.interfaces.stylegan import BendedStyleGAN

FLAGS = flags.FLAGS
flags.DEFINE_string('path', required=True, default=None, help="path of the model to script")
flags.DEFINE_string('config', default=None, help="path of bending config")
flags.DEFINE_string('out', default="models/sg3", help="out path")
flags.DEFINE_string('device', default="cpu", help="device id")

def bend_model(module):
    act_layers = module.aliases()['layer_out']
    for i, target_act in enumerate(act_layers):
        c1 = tb.BendingParameter(f"mask_{i}", 1., range=[0., 1.])
        c2 = tb.BendingParameter(f"seed_{i}", 0, range=[0, 1024])
        cb_mask_kernels = tb.OrderedMask(prob = c1, seed = c2, dim=1)
        c3 = tb.BendingParameter(f"scale_{i}", 1., range=[-5, 5])
        c4 = tb.BendingParameter(f"bias_{i}", 0., range=[-5, 5])
        cb_affine_kernels = tb.Affine(scale=c3, bias=c4)
        c5 = tb.BendingParameter(f"noise_{i}", 0., range=[0., 5.])
        cb_noise_kernels = tb.Normal(std=c5)
        c6 = tb.BendingParameter(f"permute_seed_{i}", -1, range=[-1, 1024])
        cb_permute_kernels = tb.Permute(seed=c6, dim=1)
        module.bend(cb_mask_kernels, target_act, bend_param=False)
        module.bend(cb_affine_kernels, target_act, bend_param=False)
        module.bend(cb_noise_kernels, target_act, bend_param=False)
        module.bend(cb_permute_kernels, target_act, bend_param=False)


def main(argv):
    path = Path(FLAGS.path)
    target_path = Path(FLAGS.out) / f"{path.stem}_{FLAGS.device}.ts"
    device = torch.device(FLAGS.device)

    logging.info('loading model...')
    module = BendedStyleGAN(path, device=torch.device(device))

    logging.info('bending model...')
    bend_model(module)
    #TODO loading config makes model not scriptable...
    # if FLAGS.config: 
    #     config = tb.BendingConfig.load(FLAGS.config)
    # module.bend(config)

    controllable_dict = {}
    for name, param in module.controllables().items():
        controllable_dict[name] = {'type': tb.BendingParamType.param_hash()[param.param_type], 
                                   'range': [param.min_clamp, param.max_clamp]} 

    # bending units 
    metadata = {
        'latent_dim': module.latent_dim, 
        'conditioning_dim': module.conditioning_dim, 
        'controllables': controllable_dict
    }

    extra_files = {'td_metadata': pickle.dumps(metadata)}
    
    bended = module.script()
    torch.jit.save(bended, str(target_path), _extra_files=extra_files)

    loaded_meta = {k: '' for k in extra_files.keys()}
    loaded = torch.jit.load(str(target_path), _extra_files=loaded_meta)
    loaded_meta['td_metadata'] = pickle.loads(loaded_meta['td_metadata'])

    logging.info('testing model...')
    inputs = module.get_inputs(1)
    for name, param in module.controllables().items():
        param_type = tb.BendingParamType.param_hash()[param.param_type]
        getattr(loaded, f"get_{name}")()


    loaded(**inputs)



if __name__== "__main__":
    app.run(main)