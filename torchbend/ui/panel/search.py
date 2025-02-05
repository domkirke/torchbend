import panel as pn


def get_filter_from_module(bended_module):
    pn.widgets.AutocompleteInput()

def get_fn_filters_from_module(bended_module):
    methods = bended_module.traced_methods
    default_choice = ["forward"] if "forward" in methods else []
    return pn.widgets.CheckBoxGroup(
        name = "traced functions",
        value = default_choice,
        options = methods, 
        inline = True
    )

def panel_search_ui(
        bended_module
):
    fn_checkboxes = get_fn_filters_from_module(bended_module)

    return pn.Column(fn_checkboxes)
    # filter_input = get_filter_from_module(bended_module)

