from typing import List
import importlib
import os    


def import_hacks_from_file(source_path):
    config_name = os.path.basename(source_path).replace('.py', '')  # Gets the filename without extension
    spec = importlib.util.spec_from_file_location(config_name, source_path)
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)
    assert hasattr(config_file, "callbacks")
    assert hasattr(config_file, "params") or hasattr(config_file, "activations")
    return {'callbacks': config_file.callbacks, 'params': config_file.params}


def prod(x: List[int], start: int =1):
    for i, x_tmp in enumerate(x):
        start = start * x_tmp
    return start