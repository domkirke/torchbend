from .utils import *
from .module_config import ModuleTestConfig

import os, sys, uuid
import importlib

import os
import importlib
from glob import glob

# Assume subpackage is located one level below this __init__.py
subpackage_path = os.path.join(os.path.dirname(__file__), 'modules')
module_files = glob(os.path.join(subpackage_path, '*.py'))

modules_to_test = []
modules_to_compare = []

for module_file in module_files:
    module_name = os.path.basename(module_file)[:-3]  # Remove .py
    if module_name == "__init__":
        continue
    
    full_module_name = f'test_modules.modules.{module_name}'
    module = importlib.import_module(full_module_name)
    
    if hasattr(module, 'modules_to_test'):
        modules_to_test.extend(getattr(module, 'modules_to_test', []))
    if hasattr(module, 'modules_to_compare'):
        modules_to_compare.extend(getattr(module, 'modules_to_compare', []))


# You can now use `test_modules_list` as needed within this package
# __all__ = ['test_modules_list']  # Optionally make it part of the package API

scriptable_modules_to_test = list(filter(lambda x: x.is_scriptable, modules_to_test))
__all__ = ["modules_to_test", "modules_to_compare", "scriptable_modules_to_test"]


# from . import module_test_modules
# from . import trace_test_modules

