from ...tracing.tracing import ActivationProperties
from functools import partial
import pandas as pd
import panel as pn


def get_fn_filters_from_module(bended_module):
    methods = bended_module.traced_methods
    default_choice = ["forward"] if "forward" in methods else []
    return pn.widgets.CheckBoxGroup(
        name = "traced functions",
        value = default_choice,
        options = methods, 
        inline = True
    )


def get_activation_names(bended_module, fn):
    with_fn = (len(fn) >= 2)
    if len(fn) == 1: fn = fn[0]
    print("fn", fn, with_fn)
    activations = bended_module.activations(fn=fn, _with_fn=with_fn)
    return list(activations.keys())


def get_filter_from_module(bended_module, fn_checkboxes):
    fn_checkboxes = fn_checkboxes.value
    print(fn_checkboxes, len(fn_checkboxes))
    if len(fn_checkboxes) == 0:
        return pn.widgets.TextInput(disabled=True)
    activations = get_activation_names(bended_module, fn=fn_checkboxes)
    return pn.widgets.AutocompleteInput(options=activations, placeholder="Filter activations using regexps...")

def update_activation_completions(bended_module, fn_checkboxes, filter_input):
    if len(fn_checkboxes.value) > 0:
        activations = get_activation_names(bended_module, fn=fn_checkboxes.value)
        filter_input.param.update(options=activations)

def get_activation_list(fn_checkboxes, filter_input, bended_module=None, fields=None):
    assert bended_module is not None
    fn = fn_checkboxes.value
    flt = filter_input
    if len(fn) == 0: return pn.pane.DataFrame()
    if len(fn) == 1: fn = fn[0]
    with_fn = isinstance(fn, list)

    fields = fields or ActivationProperties._default_panel_fields()
    flt = tuple() if flt.value_input == "" else (flt.value_input,)
    activations = bended_module.activations(*flt, fn=fn, _with_fn=with_fn)
    parsed_activations = pd.DataFrame(
        {f: [str(getattr(p, f)) for p in activations.values()] for f in fields},
    )
    return parsed_activations

def update_activation_list(data_frame, fn_checkboxes, *args, **kwargs):
    if len(fn_checkboxes.value) == 0:
        data_frame.param.update(object=pd.DataFrame())
    else:
        activations = get_activation_list(fn_checkboxes, *args, **kwargs)
        data_frame.param.update(object=activations)

def update_code_editor(activation, code_editor):
    code = activation.code
    print(code.code)
    filename = code.code.co_filename
    with open(filename, 'r') as f:
        code = f.read()
    code_editor.param.update(value=code, readonly=True)


def click_list_callback(event, bended_module=None, data_frame=None, code_editor=None):
    assert bended_module is not None
    assert data_frame is not None
    print(event, event.row)
    clicked_activation = data_frame.value['name'][event.row]
    activation = bended_module.activations(clicked_activation)[clicked_activation]
    if code_editor:
        update_code_editor(activation, code_editor)

    
def get_activation_list_widget(fn_checkboxes, filter_input, bended_module=None, fields=None):
    parsed_activations = get_activation_list(fn_checkboxes, filter_input, bended_module=bended_module, fields=fields)
    out = pn.widgets.Tabulator(
        parsed_activations, 
        disabled = True,
        max_height = 400,
        sizing_mode = "stretch_both",
    )
    return out

def update_search(bended_module, fn_checkboxes, filter_input, data_frame, event):
    if event.obj == fn_checkboxes:
        filter_input.param.update(value="")
        update_activation_completions(bended_module, fn_checkboxes, filter_input)
        update_activation_list(data_frame, fn_checkboxes, filter_input, bended_module=bended_module)
    elif event.obj == filter_input:
        update_activation_list(data_frame, fn_checkboxes, filter_input, bended_module=bended_module)
    elif event.obj == data_frame:
        pass

def panel_search_ui(
        bended_module, 
        context = None
):

    fn_checkboxes = get_fn_filters_from_module(bended_module)
    filter_input = get_filter_from_module(bended_module=bended_module, fn_checkboxes=fn_checkboxes)
    activations_frame = get_activation_list_widget(fn_checkboxes, filter_input, bended_module=bended_module)
    # make watchers
    update_callback = partial(update_search, bended_module, fn_checkboxes, filter_input, activations_frame)
    fn_checkboxes.param.watch(update_callback, ['value'], onlychanged=True)
    filter_input.param.watch(update_callback, ['value_input'], onlychanged=True)

    # activations_frame.watch(click_list_callback, ['selection'], onlychanged=True)

    # w1 = pn.widgets.CodeEditor(annotations=["row"], readonly=True, width = 200, height=300)
    # w1 = pn.widgets.CodeEditor(width = 200, height = 300)
    # w1 = "Hello!"

    code_editor = pn.widgets.CodeEditor(sizing_mode='stretch_width', readonly=True, language='python', width = 500, height=300)
    config = {"headerControls": {"close": "remove"}, "theme": "light"}
    floatpanel = pn.layout.FloatPanel(code_editor, name='code', margin=20, config=config)

    activations_frame.on_click(partial(click_list_callback, bended_module=bended_module, data_frame=activations_frame, code_editor=code_editor))

    return pn.Column(
        fn_checkboxes, 
        filter_input, 
        activations_frame, 
        floatpanel,
        height=600, 
        sizing_mode="stretch_width"
    ).servable()

