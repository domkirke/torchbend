import os
import pickle
import numpy as np
import time
from threading import Lock, Thread
import torch
import sys
from pathlib import Path

sourcefile = os.path.abspath(me.parent().op('gan_script_file').par.file.eval())
torchbend_path = str(Path(sourcefile).parent.parent.parent)
if str(torchbend_path) not in sys.path: sys.path.append(str(torchbend_path))

import torchbend as tb
from torchbend.interfaces.stylegan import BendedStyleGAN
torch.set_grad_enabled(False)

init_kernels_with_path = str(Path(torchbend_path) / "models" / "sg3" / "stylegan3-80sComm-000208.pkl")

DEBUG = False

class ModelFake(object):
    def forward(self, z): 
        time.sleep(1)
        return torch.zeros(3, 256, 256)
    

def dbg(*args, **kwargs):
    if DEBUG: print(*args, **kwargs)

def generate_image(handler, model, z, device):
    z = torch.from_numpy(z)
    z = z.permute(2, 0, 1)[[0]].reshape(1, -1).to(torch.device(device))
    if handler._conditioning_dim:
        c = torch.zeros(1, 10).to(torch.device(device))
        c[0, 4] = 1
    else:
        c = None
    with handler._model_lock:
        image = model.forward(z, c)
    image = (image.clamp(-1, 1) + 1) / 2
    image = image.flip(-2)
    image = image[0].permute(1, 2, 0).contiguous()
    dbg('setting cache')
    handler.set_cache(image)
    with handler._image_available_lock:
        dbg('setting availability')
        handler._image_available = True



class SG3Handler(object): 
    def __init__(self): 
        self._ready = False
        self._path = None
        self._model = None
        self._model_lock = Lock()
        self._latent_dim = None
        self._conditioning_dim = None
        self._generation_thread = None
        self._device = None
        self._image_available = False
        self._image_available_lock = Lock()
        self._cache = None
        self._cache_lock = Lock()
        self._param_map_dict = {}

    @property
    def has_model(self): 
        return self._model is not None

    def _device_from_model_path(self, path): 
        path = os.path.splitext(os.path.basename(path))[0]
        if path.endswith("cuda"): 
            device = "cuda"
        elif path.endswith("mps"): 
            device = "mps"
        else:
            device = "cpu"
        return device

    def load_model(self, path):
        dbg('loading model : ', path)
        try:
            meta_dict = {'td_metadata': ''}
            self._device = self._device_from_model_path(path)
            BendedStyleGAN(init_kernels_with_path, device=self._device)
            self._path = path
            with self._model_lock:
                self._model = torch.jit.load(path, _extra_files=meta_dict, map_location=torch.device(self._device))
                self._model = self._model.to(torch.device(self._device))
            try:
                meta_dict = pickle.loads(meta_dict['td_metadata'])
            except Exception as e: 
                print('[Warning] Could not load metadata from ts file. Bending unavailable')
                meta_dict = {}
            self._latent_dim = meta_dict.get('latent_dim', 512)
            self._conditioning_dim = meta_dict.get('conditioning_dim', 0)
            self._controllables = meta_dict.get('controllables', {})
            # pass
        except Exception as e:
            self._path = path
            raise e

    def set_cache(self, image):
        with self._cache_lock:
            self._cache = image

    def get_cache(self): 
        with self._cache_lock:
            return self._cache.clone()

    def has_image_available(self):
        with self._image_available_lock:
            return bool(self._image_available)

    def get_bending_param_from_alias(self, alias):
        if not alias in self._param_map_dict:
            return None
        if self._model is None: 
            return None
        name, partype = self._param_map_dict[alias]
        with self._model_lock:
            val = getattr(self._model, f"get_{name}")()
        if partype == "int": val = int(val)
        if partype == "float": val = float(val)
        if partype == "bool": val = bool(val)
        if partype == "tensor": val = torch.Tensor(val)
        return val

    def start_thread(self, z):
        if self._generation_thread is not None:
            if self._generation_thread.is_alive():
                return 
            else:
                self._generation_thread.join()
        self._generation_thread = Thread(target=generate_image, args=(self, self._model, z, self._device))
        self._generation_thread.start()

    def generate(self, z):
        dbg('starting thread')
        self.start_thread(z)
        with self._image_available_lock:
            dbg('fetching availability')
            available = self._image_available
        dbg('got availability', available)
        if available:
            dbg('cache available')
            image = self.get_cache()
            dbg('cache loaded')
            # self._generation_thread.join()
            with self._image_available_lock:
                dbg('set avaiblity', False)
                self._image_available = False
            dbg('starting thread...')
            self.start_thread(z)
            dbg('image', image.shape)
            return image

    def clear_mapping(self): 
        self._param_map_dict = {}

    def map_parameter(self, name, alias, partype):
        self._param_map_dict[alias] = (name, partype)
        
    def set_bending_parameter(self, name, val):
        if name not in self._param_map_dict:
            print(f"{name} not known")
        name, partype = self._param_map_dict[name]
        if partype == "int": val = int(val)
        if partype == "float": val = float(val)
        if partype == "bool": val = bool(val)
        if partype == "tensor": val = torch.Tensor(val)
        try:
            with self._model_lock:
                res = getattr(self._model, f"set_{name}")(val)
                if res != 0:
                    print("could not set parameter :", name)
        except Exception as e: 
            print("ERROR : %s"%e)
        

sg3_handler = SG3Handler()

def updateModel(path):
    if os.path.exists(path):
        sg3_handler.load_model(path)
    else: 
        print('no model at path : %s'%path) 

def updateBendingParameter(name, val):
    sg3_handler.set_bending_parameter(name, val)

def updatePage():
    scriptop = me.parent().op('gan_handler')
    scriptop.destroyCustomPars()
    sg3_handler.clear_mapping()
    page = scriptop.appendCustomPage('StyleGAN3')
    model_p = page.appendFile('Model', label="SG3 torchscript path")
    print("set to path : ", sg3_handler._path)
    model_p.val = sg3_handler._path
    for name, props in sg3_handler._controllables.items():
        partype = props.get('type')
        parrange = props.get('range')
        alias = "Bending"+(name.replace('_', ''))
        print(alias)
        sg3_handler.map_parameter(name, alias, partype)
        if partype == "float":
            p = page.appendFloat(alias, label=f"Bending: {name}")
        elif partype == "int": 
            p = page.appendInt(alias, label=f"Bending: {name}")
        elif partype == "bool":
            p = page.appendToggle(alias, label=f"Bending: {name}")
        elif partype == "tensor": 
            p = page.appendSequence(alias, label=f"Bending: {name}")
        if parrange[0] is not None: 
            p.normMin = parrange[0]
            print("setting min of", name, "to", parrange[0])
        if parrange[1] is not None: 
            p.normMax = parrange[1]
            print("setting max of", name, "to", parrange[1])


def onSetupParameters(scriptOp):
    page = scriptOp.appendCustomPage('StyleGAN3')
    p = page.appendFile('Model', label="SG3 torchscript path")

# called whenever custom pulse parameter is pushed
def onPulse(par):
    me.module.onCook(me)

def updateBendingParamVals(scriptOp):
    bending_params = scriptOp.pars("Bending*")
    for param in bending_params:
        val = sg3_handler.get_bending_param_from_alias(param.name)
        if val is not None:
            print(param.name, val)
            param.val = val


def checkModel(scriptOp): 
    model_param = scriptOp.par['Model'].eval()
    dbg('found model path :', model_param)
    if os.path.isfile(str(model_param)):
        sg3_handler.load_model(model_param)
        updatePage()
        updateBendingParamVals(scriptOp)

def onCook(scriptOp):
    if not sg3_handler.has_model: 
        dbg('no model found; checking...')
        checkModel(scriptOp)
        if not sg3_handler.has_model: return 
    noise_in = scriptOp.inputs[0].numpyArray(delayed=True, writable=True)
    if noise_in is not None:
        out = sg3_handler.generate(noise_in)
        if out is not None: 
            scriptOp.copyNumpyArray(out.cpu().numpy())

def onGetCookLevel(scriptOp):
    """
    sets the scriptOp's cook level, the conditions necessary to cause a cook.

    Return one of the following:
        CookLevel.AUTOMATIC - inputs changed and output being used. TD default behavior.
        CookLevel.ON_CHANGE - inputs changed, output used or not.
        CookLevel.WHEN_USED - every frame when output is being used
        CookLevel.ALWAYS - every frame
    """

    return CookLevel.AUTOMATIC
