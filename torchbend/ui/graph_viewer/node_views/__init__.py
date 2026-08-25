"""Modular node-view system for the graph viewer.

Single source of truth for *how a node's tensor is displayed*: a catalog of view
types keyed by (batched) tensor rank, shape-based inference of a default, per-node
configuration from ``graph_viewer.run(views=...)``, and live user overrides.

Public:
    NodeView          — user-facing per-node spec for run(views=...).
    serialize_node    — resolve + serialize a tensor for a given node.
    infer_views       — {default, compatible} for a shape.
    describe          — inference + option schemas, for the UI picker.
    resolve           — (ViewType, option_values) for a node.
"""
from .base import NodeView, ViewOption, ViewType, as_batched
from . import registry
from .registry import register_view, get_view, all_views, infer_views, describe, resolve
from .builtin import ALL_VIEWS

for _vt in ALL_VIEWS:
    register_view(_vt)


def serialize_node(tensor, fn=None, node=None, *, session_sel=None, run_cfg=None,
                   rank_defaults=None, sr_hint=None, role=None) -> dict:
    """Resolve the view for *node* and return its render payload + ``_view_meta``.

    Resolution precedence (in :func:`resolve`): session selection > run() config >
    inferred default. The returned payload always carries a ``view`` field and a
    ``_view_meta`` block describing the current/default/compatible views and the
    current view's option schema + values, so the client can render the picker
    without a second request.
    """
    import torch
    if not torch.is_tensor(tensor):
        try:
            tensor = torch.as_tensor(tensor)
        except Exception:
            return {"view": "error", "error": "not a tensor", "shape": [],
                    "_view_meta": {"current": None, "default": None, "compatible": [],
                                   "options": [], "option_values": {}}}

    t = as_batched(tensor)
    shape = [int(s) for s in t.shape]
    dtype = t.dtype

    vt, opt_values = resolve(shape, dtype=dtype, name=node,
                             session_sel=session_sel, run_cfg=run_cfg,
                             rank_defaults=rank_defaults)
    ctx = {"sample_rate": sr_hint, "fn": fn, "node": node, "role": role}

    if vt is None:
        payload = {"view": "unsupported", "shape": shape, "data": None}
    else:
        try:
            payload = vt.serialize(t, opt_values, ctx)
        except Exception as exc:  # never let a view crash the request
            payload = {"view": "error", "shape": shape, "error": str(exc)}

    desc = describe(shape, dtype=dtype, name=node)
    cur = vt.name if vt is not None else None
    cur_view = get_view(cur)
    payload["_view_meta"] = {
        "current": cur,
        "default": desc["default"],
        "compatible": [{"name": c["name"], "label": c["label"]} for c in desc["compatible"]],
        "options": cur_view.options_schema() if cur_view else [],
        "option_values": opt_values,
    }
    return payload


__all__ = ["NodeView", "ViewOption", "ViewType", "as_batched",
           "serialize_node", "infer_views", "describe", "resolve",
           "register_view", "get_view", "all_views", "registry"]
