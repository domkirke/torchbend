"""View registry: registration, shape-based inference, and per-node resolution."""
from .base import NodeView

_REGISTRY: dict = {}


def register_view(view_type) -> None:
    inst = view_type() if isinstance(view_type, type) else view_type
    _REGISTRY[inst.name] = inst


def get_view(name: str):
    return _REGISTRY.get(name)


def all_views() -> list:
    return list(_REGISTRY.values())


def _compatible(shape, dtype=None) -> list:
    """ViewType instances accepting *shape*, ordered by descending priority."""
    vts = [vt for vt in _REGISTRY.values() if _safe_accepts(vt, shape, dtype)]
    vts.sort(key=lambda vt: vt.priority, reverse=True)
    return vts


def _safe_accepts(vt, shape, dtype) -> bool:
    try:
        return bool(vt.accepts(shape, dtype))
    except Exception:
        return False


def infer_views(shape, dtype=None, name=None) -> dict:
    """Return ``{"default": <name>, "compatible": [<name>, ...]}`` for a batched shape."""
    vts = _compatible(shape, dtype)
    if not vts:
        return {"default": None, "compatible": []}
    default = vts[0]
    # name-hint nudge: a hinted view (e.g. "audio"→audio) wins if it's compatible.
    if name:
        for vt in vts:
            try:
                if vt.name_hint(name, shape, dtype):
                    default = vt
                    break
            except Exception:
                pass
    return {"default": default.name, "compatible": [vt.name for vt in vts]}


def describe(shape, dtype=None, name=None) -> dict:
    """Inference + per-view option schemas, for the UI picker."""
    info = infer_views(shape, dtype, name)
    compat = []
    for n in info["compatible"]:
        vt = _REGISTRY[n]
        compat.append({"name": vt.name, "label": vt.label, "options": vt.options_schema()})
    return {"default": info["default"], "compatible": compat}


def resolve(shape, dtype=None, name=None, session_sel=None, run_cfg=None, rank_defaults=None):
    """Resolve (view_name, option_values) for a node.

    Precedence: session selection > run() per-node config (NodeView) >
    run() per-rank default (``rank_defaults``: {ndim: view_name}) > inferred default.
    Returns (view_type, option_values_dict). Falls back gracefully if a chosen
    view name is unknown or no longer accepts the shape.
    """
    info = infer_views(shape, dtype, name)
    chosen_name = None
    chosen_opts = {}

    # 1. live session selection
    if session_sel:
        chosen_name = session_sel.get("view")
        chosen_opts = dict(session_sel.get("options") or {})

    # 2. run() per-node config (NodeView)
    if not chosen_name and run_cfg is not None:
        try:
            nv = NodeView.coerce(run_cfg)
            chosen_name = nv.view
            chosen_opts = dict(nv.options)
        except Exception:
            chosen_name = None

    # 3. run() per-rank default, e.g. {3: "audio"}
    if not chosen_name and rank_defaults:
        chosen_name = rank_defaults.get(len(shape))

    # validate the chosen view exists and accepts the shape; else fall back
    vt = _REGISTRY.get(chosen_name) if chosen_name else None
    if vt is None or not _safe_accepts(vt, shape, dtype):
        if vt is not None and not _safe_accepts(vt, shape, dtype):
            # configured view incompatible with this shape — drop its options too
            chosen_opts = {}
        vt = _REGISTRY.get(info["default"])

    if vt is None:
        return None, {}
    return vt, vt.coerce_options(chosen_opts)
