import panel as pn
import functools
from .utils import tensor_to_audio, tensor_to_image, batched, get_widgets_from_controllables

_DEFAULT_MAX_COLUMNS = 4
_DEFAULT_MAX_IMAGE_COLUMNS = 4

def generate_realtime_image(bended_module=None,
                            max_columns=None,
                            max_rows=None, 
                            gen_params=None, 
                            **widget_dict):
    assert bended_module is not None
    for name, param in widget_dict.items():
        bended_module.update(name, param)
    in_args, in_kwargs = gen_params.get('input', (tuple(), dict()))
    upscale = gen_params.get('upscale')
    callback = gen_params.get('fn', 'forward')
    outs = getattr(bended_module, callback)(*in_args, **in_kwargs)
    assert outs.ndim in [3, 4], "image generation model must have either 3 or 4 dimensions"
    if outs.ndim == 3:
        outs = outs[None]
    max_columns = max_columns or _DEFAULT_MAX_IMAGE_COLUMNS
    outs = [pn.pane.image.Image(tensor_to_image(o, upscale=upscale)) for o in outs]
    outs = [pn.Row(*w) for w in batched(outs, max_columns)]
    return pn.Column(*outs)


def generate_offline_image(event,
                           bended_module=None,
                           widget=None,
                           max_columns=None,
                           max_rows=None, 
                           gen_params=None, 
                           **widget_dict):
    assert bended_module is not None
    assert widget is not None
    for name, param in widget_dict.items():
        bended_module.update(name, param.value)
    in_args, in_kwargs = gen_params.get('input', (tuple(), dict()))
    upscale = gen_params.get('upscale')
    callback = gen_params.get('fn', 'forward')
    outs = getattr(bended_module, callback)(*in_args, **in_kwargs)
    assert outs.ndim in [3, 4], "image generation model must have either 3 or 4 dimensions"
    if outs.ndim == 3:
        outs = outs[None]
    max_columns = max_columns or _DEFAULT_MAX_COLUMNS
    outs = [pn.pane.image.Image(tensor_to_image(o, upscale=upscale)) for o in outs]
    outs = [pn.Row(*w) for w in batched(outs, max_columns)]
    outs = pn.Column(*outs)
    widget.clear()
    widget.append(outs)


def generate_offline_audio(event,
                           widget=None,
                           bended_module=None,
                           max_columns=None,
                           max_rows=None, 
                           gen_params=None, 
                           **widget_dict):
    assert bended_module is not None
    assert widget is not None
    for param_name, param_widget in widget_dict.items():
        bended_module.update(param_name, param_widget.value)
    in_args, in_kwargs = gen_params.get('input', (tuple(), dict()))
    callback = gen_params.get('fn', 'forward')
    outs = getattr(bended_module, callback)(*in_args, **in_kwargs)
    assert outs.ndim in [3, 2], "image generation model must have either 3 or 4 dimensions"
    if outs.ndim == 2:
        outs = outs[None]
    max_columns = max_columns or _DEFAULT_MAX_COLUMNS
    sr = gen_params.get('sr') or bended_module.sample_rate
    if sr is None: sr= bended_module.sample_rate
    outs = [tensor_to_audio(o, sr=sr) for o in outs]
    outs = [pn.pane.Audio(o, sample_rate=sr) for o in outs]
    outs = [pn.Row(*w) for w in batched(outs, max_columns)]
    outs = pn.Column(*outs)
    widget.clear()
    widget.append(outs)       


def generate_realtime(bended_module, 
                      render_type=None,
                      max_columns = None, 
                      max_rows = None, 
                      **kwargs):
    render_type = render_type or getattr(bended_module, "_panel_render_type_")
    if render_type == "image":
        return functools.partial(generate_realtime_image, 
                                 bended_module=bended_module, 
                                 max_columns = max_columns,
                                 max_rows = max_rows,
                                 gen_params=kwargs)
    else:
        raise NotImplementedError("render_type %s not handled by panel interfaces"%render_type)


def generate_offline(bended_module, 
                     parameters, 
                     widget,
                     render_type = None,
                     max_columns = None, 
                     max_rows = None, 
                     **kwargs):
    render_type = render_type or getattr(bended_module, "_panel_render_type_")                     
    if render_type == "image": 
        return functools.partial(generate_offline_image, 
                                 widget=widget,
                                 bended_module=bended_module, 
                                 max_columns = max_columns,
                                 max_rows = max_rows,
                                 gen_params=kwargs, 
                                 **parameters)
    elif render_type == "audio":
        return functools.partial(generate_offline_audio, 
                                 widget=widget,
                                 bended_module=bended_module, 
                                 max_columns = max_columns,
                                 max_rows = max_rows,
                                 gen_params=kwargs, 
                                 **parameters)
    else:
        raise NotImplementedError("render_type %s not handled by panel interfaces"%render_type)

                                 
def get_generation_ui(bended_module, controllable_widgets, realtime, **kwargs):
    max_image_columns = kwargs.get('max_image_columns') or _DEFAULT_MAX_IMAGE_COLUMNS
    if realtime:
        gen_block = pn.bind(
            generate_realtime(
                bended_module=bended_module, 
                max_columns=max_image_columns,
                **kwargs),
            **controllable_widgets
        )

    else:
        gen_block = pn.layout.WidgetBox()
        generate_callback = generate_offline(
            bended_module=bended_module, 
            parameters=controllable_widgets, 
            widget=gen_block,
            max_columns=max_image_columns,
            **kwargs
        )
        generate_button = pn.widgets.Button(name="Generate", 
                                            button_type="primary")
        generate_button.on_click(generate_callback)

        gen_block = pn.layout.Column(
            gen_block, 
            pn.Row(pn.Spacer(), generate_button)
        )
    return gen_block


def panel_generation_ui(
        bended_module, 
        realtime=True,
        max_columns=None,
        max_image_columns=None,
        max_rows=None,
        context = None,
        **kwargs
):
    controllables = bended_module.controllables
    controllable_widgets = get_widgets_from_controllables(controllables)
    if max_rows is None and max_columns is None:
        max_columns = _DEFAULT_MAX_COLUMNS
    widgets_columns = [pn.Row(*w) for w in batched(controllable_widgets.values(), max_columns)]
    widgets_block = pn.Column(*widgets_columns)

    gen_block = get_generation_ui(bended_module, controllable_widgets, realtime, max_image_columns=max_image_columns, **kwargs)
    full_widget = pn.Column(widgets_block, gen_block)
    return full_widget

__all__ = ['panel_generation_ui']