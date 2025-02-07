import panel as pn
from ... import bending

def get_bending_callbacks():
    callback_classes = {}
    for attr_name, attr in bending.__dict__.items():
        if not isinstance(attr, type): continue
        if issubclass(attr, bending.BendingCallback) and attr != bending.BendingCallback:
            callback_classes[attr_name] = attr
    return callback_classes


def panel_bending_ui(
        bended_module, 
        context = None
):
    callbacks = get_bending_callbacks()
    fav_activations = context.get('marked_activations', [])

    callbacks_menu = pn.widgets.MenuButton(name="Callback", items = list(callbacks.keys()), width=500)

    return pn.Column(
        callbacks_menu
    )