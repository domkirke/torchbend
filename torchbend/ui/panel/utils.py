import math, os
import torchaudio
import torch
from collections import OrderedDict
import torchvision.transforms as transforms
import panel as pn
from ... import  BendingParamType
from ...utils import get_random_hash


_TB_DEFAULT_PANEL_OUT = "/tmp/torchbend/ui/panel"


def get_widget_from_controllable(ctrl):
    widget_type = BendingParamType.param_hash()[ctrl.param_type]
    if widget_type == "float":
        assert ctrl.min_clamp is not None, "minimum value must be defined for panel interfaces"
        assert ctrl.max_clamp is not None, "maximum value must be defined for panel interfaces"
        widget = pn.widgets.FloatSlider(
                name = ctrl.name,
                start = ctrl.min_clamp,
                end = ctrl.max_clamp,
                step = (ctrl.max_clamp - ctrl.min_clamp) / 1000,
                value = ctrl.get_python_value()
        )
    elif widget_type == "int":
        assert ctrl.min_clamp is not None, "minimum value must be defined for panel interfaces"
        assert ctrl.max_clamp is not None, "maximum value must be defined for panel interfaces"
        widget = pn.widgets.IntSlider(
                name = ctrl.name,
                start = ctrl.min_clamp,
                end = ctrl.max_clamp,
                value = ctrl.get_python_value()
        )
    elif widget_type == "bool":
        widget = pn.widgets.Checkbox(
            name = ctrl.name
        )
    return widget

def get_widgets_from_controllables(controllables):
    widgets = OrderedDict()
    for name, ctrl in controllables.items():
        widgets[name] = get_widget_from_controllable(ctrl)
    return widgets

def get_frame_dim(n, max_row=None, max_col=None):
    if max_row is None and max_col is None:
        max_col = _DEFAULT_MAX_COLUMNS
    if max_col is None:
        rows = math.floor(n / max_row)
        cols = n % max_col
    elif max_row is None:
        cols = math.floor(n / max_col)
        rows = n % max_col
    return rows, cols

def batched(iterator, batch):
    iterator = list(iterator)
    for i in range(math.ceil(len(iterator) / batch)):
        if len(iterator) == 0:
            raise StopIteration
        result, iterator = iterator[:batch], iterator[batch:]
        yield result
