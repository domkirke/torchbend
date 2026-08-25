import os, json

_SERVER_INIT_DICT = {
      "models": [

      ], 
}

def _init_config_file(config_file):
    config_dir = os.path.dirname(config_file)
    os.makedirs(config_dir, exist_ok=True)
    with open(config_file, "w+") as f:
        json.dump(_SERVER_INIT_DICT, f)
        