try:
    import panel as pn
    from . import panel
except ImportError:
    panel = None

from . import graph_viewer
from .graph_viewer.node_views import NodeView  # tb.ui.NodeView — per-node display spec
from .graph_viewer.default_inputs import Expr  # tb.ui.Expr — a default input as its source
