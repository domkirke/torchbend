try:
    import panel as pn
    from . import panel
except ImportError:
    panel = None

from . import graph_viewer
from .graph_viewer.node_views import NodeView  # tb.ui.NodeView — per-node display spec
