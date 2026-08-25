
import re
from pathlib import Path
import torch
from ..base import Interface
import vschaos
from ...tracing import BendedModule
import acids_transforms
from vschaos.models.scriptable_ae import ScriptableSpectralAutoEncoder
from torch import _dynamo as torchdynamo

torch.fx._symbolic_trace._wrapped_methods_to_patch.extend([
    (acids_transforms.OverlapAdd, "forward"), 
    (acids_transforms.OverlapAdd, "invert")
])

class BendedVSChaos(Interface):
    
    def __init__(self, model_path, scriptable=True, **kwargs):
        self.model, self.config, self.transform = self.load_model(model_path, scriptable=scriptable, **kwargs)
        
        # warmup model cache
        # self.pre_process_latent = MethodType(pre_process_fn[type(self.model.encoder)], self)
        # self.post_process_latent = MethodType(post_process_fn[type(self.model.encoder)], self)

    @staticmethod
    def load_model(model_path, scriptable: bool = True,  device: str | torch.device ="cpu", **kwargs):
        model_path = Path(model_path).resolve()
        assert model_path.suffix == ".vs", "BendedVSChaos must be given direct path to .ts inside the checkpoint repository"
        name = model_path.stem
        version = re.match("version_(\d+)", model_path.parent.stem)
        if version is None: 
            raise FileNotFoundError('could not retrive version from path %s'%model_path)
        version = int(version.groups()[0])
        run_path = model_path.parent.parent
        model, config, transform  = vschaos.utils.load.load_model_from_run(str(run_path), version=version, name=name, map_location=device)
        model.eval()

        if scriptable: 
            model = ScriptableSpectralAutoEncoder(model, transform, inversion_mode="keep_input", use_oa=True, use_dimred=False)

        return model, config, transform

    def _bend_model(self, model: BendedModule):
        gms = []
        #todo : make false values with conditioning
        def compiler(gm, example_inputs):
            gms.append(gm)
            return gm.forward
        # forward = torch.compile(self.model.forward, backend=compiler, fullgraph=True)
        forward_dynamo = torchdynamo.optimize(compiler)(model.forward)
        forward_dynamo(torch.randn(1, int(model.forward_params[0]), 16384))
        print("h")