_DEFAULT_PANEL_OUT = "/tmp/torchbend/ui/panel"

import panel as pn
from . import utils
from .generation import panel_generation_ui
from .bend import panel_bending_ui
from .search import panel_search_ui


def panel_ui(bended_module, **kwargs):
    panel_context = {}
    return pn.Tabs(
        ('Explore', panel_search_ui(bended_module, context=panel_context)),
        # ('Bend', panel_bending_ui(bended_module, context=panel_context)),
        ('Play', panel_generation_ui(bended_module, context=panel_context, **kwargs))
    )