try:
    import panel as pn
    from . import panel
except ImportError:
    panel = None
