import glob
import importlib
import os
from configs.__base__.cfg_common import cfg_common
from configs.__base__.cfg_dataset_default import cfg_dataset_default


# Get absolute path to this directory
__base_dir = os.path.dirname(os.path.abspath(__file__))
files = glob.glob(os.path.join(__base_dir, '[!_]*.py'))
for file in files:
    # Convert file path to module name
    module_name = 'configs.__base__.' + os.path.basename(file).replace('.py', '')
    model_lib = importlib.import_module(module_name)
    for obj_name in dir(model_lib):
        if obj_name.startswith("cfg_model"):
            globals()[obj_name] = getattr(model_lib, obj_name)
