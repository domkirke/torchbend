from inspect import ismethod
import inspect
import urllib
import re
import functools
import torch
from typing import NoReturn
from collections import OrderedDict
import abc
import os
from ..tracing import BendedWrapper, BendedModule, ScriptableState
from .. import _TORCHBEND_DEFAULT_MODEL_DIR

def wrap_model_method(ext, func, hook=None):
    @functools.wraps(func)
    def wrapped_function(*args, **kwargs):
        out = getattr(ext, func)(*args, **kwargs)
        if hook is not None:
            out = hook(out, (args, kwargs))
        return out
    return wrapped_function


def _export_to_module(fn):
    fn.__export_to_module = True
    return fn

def _overload_module(fn):
    fn.__overload_module = True
    return fn

from pathlib import Path



class BendingInterfaceException(Exception):
    pass

def _get_name_from_url(url):
    out = re.match("^.*=(.+).(pkl|ckpt|pth)", urllib.parse.urlparse(url).query)
    if out is None:
        return None
    else:
        return f"{out.groups()[0]}.{out.groups()[1]}"

def name_from_type(cls):
    name = getattr(cls, "_download_subdir", None)
    if name is not None: return name
    name = getattr(cls, "__file__", inspect.getsourcefile(cls))
    if name is not None:
        if Path(name).suffix == ".py":
            if Path(name).parent.stem == "interfaces":
                return Path(name).stem
            else:
                return Path(name).parent.stem
    name = getattr(cls, "__name__", None)
    return name

    

class Interface(object):
    _imported_callbacks_ = []

    def __init__(self, model):
        self.model = model
        self._import_callbacks_()

    def _getmodel_(self):
        return self._model
    def _setmodel_(self, model):
        self._model = self._import_model(model)
        self._import_methods(self._model)
        self.bend_model(self._model)
    def _delmodel_(self):
        raise BendingInterfaceException('cannot delete model of interface')
    model = property(_getmodel_, _setmodel_, _delmodel_)

    @classmethod
    def _get_download_location(cls, url, name: str = None):
        loc = Path(_TORCHBEND_DEFAULT_MODEL_DIR / name_from_type(cls)).resolve()
        os.makedirs(loc, exist_ok=True)
        loc = loc / (name or _get_name_from_url(url))
        return loc

    @classmethod
    def _download_model_to(cls, url, download_location) -> Path:
        assert download_location.parent.exists()
        try:
            res = urllib.request.urlretrieve(url, str(download_location))
        except Exception as e: 
            raise BendingInterfaceException("could not download model, got : %s"%e)
        return res

    @classmethod
    def get_model_path(cls, model_path_or_url: str | Path, force_download: bool = False, download_to: Path | str | None = None):
        assert isinstance(model_path_or_url, (str, Path))
        if isinstance(model_path_or_url, str):
            parsed_url = urllib.parse.urlparse(model_path_or_url)
            if parsed_url.scheme == "":
                model_path_or_url = Path(model_path_or_url)
            else:
                try:
                    destination = Path(download_to or cls._get_download_location(model_path_or_url))
                    if (not destination.exists()) or force_download:
                        cls._download_model_to(model_path_or_url, destination)
                    return destination
                except BendingInterfaceException as e:
                    raise e

        if isinstance(model_path_or_url, Path):
            if model_path_or_url.exists():
                return model_path_or_url
            path_relative_to_model_dir = _TORCHBEND_DEFAULT_MODEL_DIR / name_from_type(cls) / model_path_or_url
            if path_relative_to_model_dir.exists():
                return path_relative_to_model_dir.resolve()
        
        raise BendingInterfaceException("could not find or download : %s"%model_path_or_url)


    def _getoriginalmodel_(self):
        return self._model._module
    def _setoriginalmodel_(self): 
        raise BendingInterfaceException("originalmodel cannot be set directly. Set model instead")
    def _deloriginal_model_(self): 
        raise BendingInterfaceException('cannot delete model of interface')
    original_model = property(_getoriginalmodel_, _setoriginalmodel_, _deloriginal_model_)


    def to(self, device):
        return self._model.to(device)

    def _import_model(self, model):
        if isinstance(model, torch.nn.Module):
            return BendedModule(model, _wrapped_methods = self._imported_callbacks_)
        else: 
            return BendedWrapper(model, _wrapped_methods = self._imported_callbacks_)

    def _import_callbacks_(self):
        for cb in self._imported_callbacks_:
            if cb not in dir(self._model):
                assert "method %s not present in base class %s"%(cb, type(self._model))
            if not hasattr(self, cb):
                setattr(self, cb, wrap_model_method(self._model, cb))

    def _retrieve_exported_methods(self):
        exported_methods = OrderedDict()
        # list methods to export
        _not_to_import_methods = []
        for attr_name in dir(self):
            if attr_name in type(self).__dict__:
                # retrieving property callbacks instead of direct values.
                attr = type(self).__dict__[attr_name]
            else:
                attr = getattr(self, attr_name)
            if isinstance(attr, property):
                if hasattr(attr.fget, "__export_to_module") and not hasattr(attr.fget, "_overload_module"):
                    exported_methods[attr_name] = attr
            else:
                if hasattr(attr, "__export_to_module"):
                    exported_methods[attr_name] = attr
        return exported_methods

    def get_activations_hook(self, out, original_args):
        if original_args[1].get("_save_as_method"):
            method_name = original_args[1]["_save_as_method"]
            setattr(self, method_name, wrap_model_method(self._model, method_name))
        return out

    def from_activations_hook(self, out, original_args):
        if original_args[1].get("_save_as_method"):
            method_name = original_args[1]["_save_as_method"]
            setattr(self, method_name, wrap_model_method(self._model, method_name))
        return out

    def _import_methods(self, model):
        assert isinstance(model, (BendedWrapper, BendedModule))
        exported_methods = self._retrieve_exported_methods()        
        # import methods from module to interface
        for attr_name in dir(model):
            if attr_name.startswith('__'): continue
            attr = getattr(model, attr_name, None)
            if ismethod(attr) and (hasattr(attr,"__import_to_interface")):
                if hasattr(self, attr_name):
                    if not getattr(getattr(self, attr_name), "__overload_module", False):
                        raise BendingInterfaceException("method %s seems in conflict with original module method. Add _overload_module decorator, or remove"%(attr_name,))
                hook = getattr(self, f"{attr_name}_hook", None)
                if getattr(attr, "__import_to_interface"):
                    setattr(self, attr_name, wrap_model_method(self._model, attr_name, hook=hook))
        # export methods to module
        for attr_name, attr in exported_methods.items():
            if hasattr(model, attr_name):
                # print('[Warning]: tried to export method %s to original module, but conflicting with existing attribute. Skipping'%attr_name)
                continue
            if attr_name in self.__dict__:
                setattr(model, attr_name, self.__dict__[attr_name])
            else:
                setattr(model, attr_name, getattr(self, attr_name))


    @abc.abstractmethod
    def bend_model(self, model):
        pass


    def trace(self, fn = "forward", *args, _save_as=None, **kwargs):
        outs = self.model.trace(fn=fn, _save_as=_save_as, **kwargs)
        method_name = fn if _save_as is None else _save_as
        if not hasattr(self, method_name):
            setattr(self, method_name, wrap_model_method(self.model, method_name))
        return outs

    @_overload_module
    def _register_method_from_graph(self, graph, fn, method_name) -> NoReturn:
        self._model._register_method_from_graph(graph, fn, method_name)
        setattr(self, method_name, wrap_model_method(self._model, method_name))

    @property
    def scriptable(self): 
        return ScriptableState.Unknown

    @property
    def nntilde_compatible(self):
        return ScriptableState.NotScriptable

    @_overload_module
    def script(self, *args, **kwargs):
        assert bool(self.scriptable), "BendedRAVE must be initialized with scriptable=True to allow jit scripting"
        return self.model.script(*args, **kwargs)

    # nntilde-related callbacks
    @abc.abstractmethod
    def register_nntilde_methods(self, model):
        raise NotImplementedError

    @abc.abstractmethod
    def register_nntilde_attributes(self, model):
        raise NotImplementedError

    @_overload_module
    def nntilde(self, *args, **kwargs):
        assert self.scriptable, "BendedRAVE must be initialized with scriptable=True to allow jit scripting"
        self.clear_cache()
        return self.model.nntilde(*args, **kwargs)
