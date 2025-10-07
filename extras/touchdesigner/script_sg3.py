import torchbend as tb
from absl import app, logging, flags
import torch
import pickle
from pathlib import Path


from torchbend.interfaces.stylegan import BendedStyleGAN

FLAGS = flags.FLAGS
flags.DEFINE_string('path', required=True, default=None, help="path of the model to script")
flags.DEFINE_string('config', default=None, help="path of bending config")
flags.DEFINE_string('out', default="models/sg3", help="out path")
flags.DEFINE_string('device', default="cpu", help="device id")


def main(argv):
    path = Path(FLAGS.path)
    target_path = Path(FLAGS.out) / f"{path.stem}_{FLAGS.device}.ts"
    device = torch.device(FLAGS.device)

    module = BendedStyleGAN("stylegan2-cifar10-32x32.pkl", device=torch.device(device))
    metadata = {
        'latent_dim': module.latent_dim, 
        'conditioning_dim': module.conditioning_dim, 
    }
    if FLAGS.config is not None: 
        metadata['bending_config'] = FLAGS.config

    extra_files = {'td_metadata': pickle.dumps(metadata)}
    
    bended = module.script()
    torch.jit.save(bended, str(target_path), _extra_files=extra_files)

    loaded_meta = {k: '' for k in extra_files.keys()}
    loaded = torch.jit.load(str(target_path), _extra_files=loaded_meta)

    logging.info('testing model...')
    inputs = module.get_inputs(1)
    loaded(**inputs)



if __name__== "__main__":
    app.run(main)