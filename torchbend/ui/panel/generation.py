import panel as pn
import functools
import torch
import logging
from .utils import batched, get_widgets_from_controllables, _TB_DEFAULT_PANEL_OUT
from ..utils import tensor_to_audio, tensor_to_image

_DEFAULT_MAX_COLUMNS = 4
_DEFAULT_MAX_IMAGE_COLUMNS = 4


def _get_norm_fn_for_out(bended_module, gen_params, fn):
    norm_fn = gen_params.get('norm_fn')
    if norm_fn is None: 
        if hasattr(bended_module, "_panel_out_norm_fn"):
            norm_fn = bended_module._panel_out_norm_fn
            if isinstance(norm_fn, dict):
                norm_fn  = norm_fn.get('fn')
                if norm_fn is None:
                    logging.warning('could not get normalisation function for callback %s'%fn)
    norm_fn = norm_fn or (lambda x: x)
    return norm_fn


def generate_realtime_image(bended_module=None,
                            max_columns=None,
                            max_rows=None, 
                            gen_params=None, 
                            out=None,
                            **widget_dict):
    assert bended_module is not None
    for name, param in widget_dict.items():
        if isinstance(bended_module, torch.jit._script.RecursiveScriptModule):
            bended_module._set_bending_control(name, torch.tensor(param))
        else:
            bended_module.update(name, param)
    in_args, in_kwargs = gen_params.get('input', (tuple(), dict()))
    upscale = gen_params.get('upscale')
    callback = gen_params.get('fn', 'forward')
    norm_fn = gen_params.get('norm_fn')
    outs = getattr(bended_module, callback)(*in_args, **in_kwargs)
    assert outs.ndim in [3, 4], "image generation model must have either 3 or 4 dimensions"
    if outs.ndim == 3:
        outs = outs[None]
    max_columns = max_columns or _DEFAULT_MAX_IMAGE_COLUMNS
    outs = [pn.pane.image.Image(tensor_to_image(o, upscale=upscale, norm_fn = norm_fn, out = _TB_DEFAULT_PANEL_OUT)) for o in outs]
    outs = [pn.Row(*w) for w in batched(outs, max_columns)]
    return pn.Column(*outs)


def generate_offline_image(event,
                           bended_module=None,
                           widget=None,
                           max_columns=None,
                           max_rows=None, 
                           norm_fn=None,
                           gen_params=None, 
                           **widget_dict):
    assert bended_module is not None
    assert widget is not None
    for name, param in widget_dict.items():
        if isinstance(bended_module, torch.jit._script.RecursiveScriptModule):
            bended_module._set_bending_control(name, torch.tensor(param))
        else:
            bended_module.update(name, param)
    in_args, in_kwargs = gen_params.get('input', (tuple(), dict()))
    
    upscale = gen_params.get('upscale')
    callback = gen_params.get('fn', 'forward')
    norm_fn = _get_norm_fn_for_out(bended_module, gen_params, callback)

    outs = getattr(bended_module, callback)(*in_args, **in_kwargs)
    assert outs.ndim in [3, 4], "image generation model must have either 3 or 4 dimensions"
    if outs.ndim == 3:
        outs = outs[None]
    max_columns = max_columns or _DEFAULT_MAX_COLUMNS
    outs = [pn.pane.image.Image(tensor_to_image(o, upscale=upscale, norm_fn=norm_fn, out=_TB_DEFAULT_PANEL_OUT)) for o in outs]
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
                           script=False,
                           **widget_dict):
    assert bended_module is not None
    assert widget is not None
    for name, param in widget_dict.items():
        if isinstance(bended_module, torch.jit._script.RecursiveScriptModule):
            bended_module._set_bending_control(name, param)
        else:
            bended_module.update(name, param)
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


    def catch_result(fn):
        def _closure(*args, **kwargs):
            out = fn(*args, out=_closure.out, **kwargs)
            _closure.out = out
            return out
        _closure.out = None
        return _closure
    if render_type == "image":
        return functools.partial(catch_result(generate_realtime_image), 
                                    bended_module=bended_module, 
                                    max_columns = max_columns,
                                    max_rows = max_rows,
                                    gen_params=kwargs)
    elif render_type is None: 
        raise ValueError('could not get render type from model; please provide render_type keyword')
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

                                 
def get_generation_ui(bended_module, controllable_widgets, realtime, script: bool = False, **kwargs):
    max_image_columns = kwargs.get('max_image_columns') or _DEFAULT_MAX_IMAGE_COLUMNS
    render_type = kwargs.get('render_type') or getattr(bended_module, "_panel_render_type_")
    callback = kwargs.get('fn')
    kwargs['norm_fn'] = kwargs.get('norm_fn', _get_norm_fn_for_out(bended_module, kwargs, callback))
    if script: 
        bended_module = bended_module.script()
    if realtime:
        gen_block = pn.bind(
            generate_realtime(
                bended_module=bended_module, 
                render_type=render_type,
                max_columns=max_image_columns,
                **kwargs),
            **controllable_widgets
        )

    else:
        gen_block = pn.layout.WidgetBox()
        if script: 
            bended_module = bended_module.script()
        generate_callback = generate_offline(
            bended_module=bended_module, 
            render_type=render_type,
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
        script=False,
        **kwargs
):
    controllables = bended_module.controllables()
    controllable_widgets = get_widgets_from_controllables(controllables)
    if max_rows is None and max_columns is None:
        max_columns = _DEFAULT_MAX_COLUMNS
    widgets_columns = [pn.Row(*w) for w in batched(controllable_widgets.values(), max_columns)]
    widgets_block = pn.Column(*widgets_columns)

    gen_block = get_generation_ui(bended_module, controllable_widgets, realtime, max_image_columns=max_image_columns, script=script, **kwargs)
    full_widget = pn.Column(widgets_block, gen_block)
    return full_widget

__all__ = ['panel_generation_ui']