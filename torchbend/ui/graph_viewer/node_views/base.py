"""Core abstractions for the modular node-view system.

A *view* describes how a node's tensor should be displayed. Every input/output is
treated as **batched**: a tensor is only unsqueezed to reach the minimum batched
rank 2, so rank maps directly to a taxonomy — ``2D = B×N``, ``3D = B×C×N``,
``4D = B×C×H×W``.
"""
import torch


def as_batched(t: torch.Tensor) -> torch.Tensor:
    """Return a view of *t* with a leading batch dim, ndim >= 2.

    scalar -> [1, 1]; [N] -> [1, N]; rank >= 2 is left untouched (taken at face
    value, per the batched convention).
    """
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    if t.ndim == 0:
        return t.reshape(1, 1)
    if t.ndim == 1:
        return t.unsqueeze(0)
    return t


# ── option schema ──────────────────────────────────────────────────────────────

_OPTION_TYPES = ("int", "float", "bool", "str", "choice", "str_list")


class ViewOption:
    """One configurable parameter of a view (drives both validation and the UI widget)."""

    def __init__(self, name, type, default=None, label=None, choices=None,
                 range=None, description=""):
        assert type in _OPTION_TYPES, f"unknown option type {type!r}"
        self.name = name
        self.type = type
        self.default = default
        self.label = label or name
        self.choices = choices
        self.range = range
        self.description = description

    def coerce(self, value):
        """Coerce a JSON value to this option's python type; fall back to default."""
        if value is None:
            return self.default
        try:
            if self.type == "int":
                return int(round(float(value)))
            if self.type == "float":
                return float(value)
            if self.type == "bool":
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "on")
                return bool(value)
            if self.type == "str":
                return str(value)
            if self.type == "choice":
                v = value
                if self.choices and v not in self.choices:
                    return self.default
                return v
            if self.type == "str_list":
                if isinstance(value, str):
                    # comma-separated convenience form
                    return [s.strip() for s in value.split(",") if s.strip()]
                return [str(s) for s in (value or [])]
        except (TypeError, ValueError):
            return self.default
        return value

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "label": self.label,
            "choices": self.choices,
            "range": self.range,
            "description": self.description,
        }


# ── view type ────────────────────────────────────────────────────────────────

class ViewType:
    """Base class for a view. Subclasses set class attrs and implement ``serialize``.

    Class attributes:
        name      : stable identifier used in payloads / config (e.g. "audio").
        label     : human label for the UI.
        priority  : inference ordering; higher wins when several views accept a shape.
        options   : list[ViewOption].
        ranks     : iterable of accepted (batched) ranks, used by the default
                    ``accepts``; subclasses may override ``accepts`` for finer control.
    """

    name = "base"
    label = "base"
    priority = 0
    options: list = []
    ranks: tuple = ()

    def accepts(self, shape, dtype=None) -> bool:
        return len(shape) in self.ranks

    def option_defaults(self) -> dict:
        return {o.name: o.default for o in self.options}

    def coerce_options(self, opts: dict | None) -> dict:
        """Return a full option dict: declared options coerced, with defaults filled in."""
        opts = opts or {}
        out = {}
        for o in self.options:
            out[o.name] = o.coerce(opts.get(o.name, o.default))
        return out

    def options_schema(self) -> list:
        return [o.to_dict() for o in self.options]

    # name-hint nudge: return True if this view should be preferred as default for
    # a node called *name*. Overridden by views that have obvious name cues.
    def name_hint(self, name, shape, dtype=None) -> bool:
        return False

    def serialize(self, t, opts: dict, ctx: dict) -> dict:
        """Return a JSON-safe payload. Must include ``"view": self.name``.

        *t* is already batched (ndim >= 2). *opts* is the full, coerced option dict.
        *ctx* carries hints (e.g. ``sample_rate``).
        """
        raise NotImplementedError


# ── user-facing config object ──────────────────────────────────────────────────

class NodeView:
    """User-facing per-node view spec, passed to ``graph_viewer.run(views=...)``.

    ``NodeView("spectrogram", sample_rate=16000)`` or, as sugar, the bare string
    ``"spectrogram"``. Validation against the named view's option schema is performed
    lazily (the registry is consulted) via :meth:`resolve_options`.
    """

    def __init__(self, view: str, **options):
        if not isinstance(view, str):
            raise TypeError("NodeView(view, **options): view must be a string name")
        self.view = view
        self.options = dict(options)

    @classmethod
    def coerce(cls, spec) -> "NodeView":
        if isinstance(spec, NodeView):
            return spec
        if isinstance(spec, str):
            return cls(spec)
        if isinstance(spec, dict):
            d = dict(spec)
            name = d.pop("view", None) or d.pop("name", None)
            if not name:
                raise ValueError("view dict spec must include a 'view' key")
            return cls(name, **d)
        raise TypeError(f"cannot interpret view spec: {spec!r}")

    def __repr__(self):
        if self.options:
            return f"NodeView({self.view!r}, {self.options})"
        return f"NodeView({self.view!r})"
