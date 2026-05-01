import torch
import random

def get_random_hash(n=8):
    return "".join([chr(random.randrange(97,122)) for i in range(n)])


def retrieve_audio_activations(bended_module, fn="forward"):

    def _arg_in_attribute(n, used_attributes):
        if not isinstance(n, torch.fx.Node): return False
        return n.name in used_attributes

    activations = bended_module.activations(r"?.*", fn=fn)
    used_attributes = []
    audio_activations = []
    for act_name, act_obj in activations.items() :
        if act_obj.op == "get_attr":
            used_attributes.append(act_name)
        else:
            use_attr = len(list(filter(lambda n: _arg_in_attribute(n, used_attributes), act_obj.args))) != 0 \
                        or len(list(filter(lambda n: _arg_in_attribute(n, used_attributes), act_obj.kwargs.values()))) != 0 
            if (use_attr):
                if act_obj.shape is not None:
                    if len(act_obj.shape) == 3 and act_obj.shape[-1] > 1:
                        audio_activations.append(act_obj)
    return audio_activations