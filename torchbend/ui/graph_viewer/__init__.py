import threading

from .node_views import NodeView   # re-exported as the public per-node view spec

default_app_config = "torchbend.ui.graph_viewer.apps.GraphViewerConfig"

_REGISTRY = None
_DEFAULT_INPUTS: dict = {}
_VIEWS: dict = {}              # global default {node: NodeView}, applied to all models
_VIEW_DEFAULTS: dict = {}      # global default {ndim: view_name}, applied to all models
_SYNC_MANAGER = None           # SyncManager instance when sync= is passed to run()


def _normalize_views(spec):
    """Coerce a ``{node: str|dict|NodeView}`` mapping into ``{node: NodeView}``."""
    if not spec:
        return {}
    out = {}
    for node, v in dict(spec).items():
        try:
            out[node] = NodeView.coerce(v)
        except Exception:
            pass
    return out


def _normalize_view_defaults(spec):
    """Coerce a ``{ndim: str|NodeView}`` mapping into ``{int: view_name}``."""
    if not spec:
        return {}
    out = {}
    for rank, v in dict(spec).items():
        try:
            name = v.view if isinstance(v, NodeView) else (
                v.get("view") if isinstance(v, dict) else str(v))
            out[int(rank)] = name
        except Exception:
            pass
    return out


class ModelEntry:
    """One slot in the multi-model registry — may be pre-loaded or lazily loaded."""

    def __init__(self, name, factory=None, module=None, memory_hint_mb=None,
                 default_inputs=None, views=None, view_defaults=None):
        self.name = name
        self.factory = factory          # () -> BendedModule, or None if pre-loaded
        self.module = module            # cached BendedModule
        self.memory_hint_mb = memory_hint_mb
        self.default_inputs = default_inputs or {}
        self.views = _normalize_views(views)   # {node: NodeView} for this model
        self.view_defaults = _normalize_view_defaults(view_defaults)   # {ndim: view_name}
        self._lock = threading.Lock()
        self.session = None             # BendingSession, created on first use

    @property
    def is_loaded(self):
        return self.module is not None

    @property
    def status(self):
        return "loaded" if self.module is not None else "lazy"

    def param_mb(self):
        if self.module is not None:
            try:
                total = sum(
                    p.numel() * p.element_size()
                    for p in self.module._module.parameters()
                )
                return round(total / (1024 * 1024), 1)
            except Exception:
                pass
        return self.memory_hint_mb


class ModelRegistry:
    """Registry of named models, loaded lazily with an optional memory guard."""

    def __init__(self, entries):
        from typing import Optional, Callable
        self._entries = {e.name: e for e in entries}
        self._current_name: Optional[str] = next(iter(self._entries)) if self._entries else None
        self._post_load_hook: Optional[Callable] = None  # callable(name, entry) fired after lazy load

    @property
    def current_name(self):
        return self._current_name

    @property
    def current(self):
        if self._current_name is None:
            return None
        entry = self._entries.get(self._current_name)
        return entry.module if entry else None

    def list_entries(self):
        return [
            {
                "name": e.name,
                "status": e.status,
                "param_mb": e.param_mb(),
                "current": e.name == self._current_name,
            }
            for e in self._entries.values()
        ]

    def select(self, name, min_free_mb=256):
        """Load (if needed) and activate a model.

        Returns ``(True, None)`` on success or ``(False, error_string)`` on failure.
        The model is cached after first load; switching back is free.
        """
        if name not in self._entries:
            return False, f"Model '{name}' not found"
        entry = self._entries[name]
        with entry._lock:
            if entry.module is None:
                if entry.factory is None:
                    return False, f"No factory registered for '{name}'"
                # Memory guard — requires psutil; silently skipped if not installed.
                try:
                    import psutil
                    avail_mb = psutil.virtual_memory().available / (1024 * 1024)
                    needed_mb = entry.memory_hint_mb or 0
                    threshold = max(float(min_free_mb), needed_mb * 3.0)
                    if avail_mb < threshold:
                        return False, (
                            f"Insufficient memory to load '{name}': "
                            f"{avail_mb:.0f} MB free, ~{threshold:.0f} MB needed"
                        )
                except ImportError:
                    pass
                try:
                    result = entry.factory()
                    # Accept BendedModule or Interface; unwrap Interface to BendedModule.
                    bm = _unwrap(result)
                    if bm is None:
                        return False, (
                            f"Factory for '{name}' returned {type(result)}, "
                            "expected BendedModule or Interface"
                        )
                    entry.module = bm
                except Exception as exc:
                    return False, f"Failed to load '{name}': {exc}"
                if self._post_load_hook is not None:
                    try:
                        self._post_load_hook(name, entry)
                    except Exception as exc:
                        import warnings
                        warnings.warn(f"[sync] post-load hook failed for '{name}': {exc}")
        self._current_name = name
        return True, None


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_bended_module(val):
    return hasattr(val, "_module") and hasattr(val, "_graphs")

def _is_interface(val):
    """True for torchbend Interface objects (wrap a BendedModule as ._model)."""
    return hasattr(val, "_model") and _is_bended_module(getattr(val, "_model", None))

def _unwrap(val):
    """Return the BendedModule from a BendedModule or an Interface; else None."""
    if _is_bended_module(val):
        return val
    if _is_interface(val):
        return val._model
    return None

def _module_class_name(bm):
    try:
        return type(bm._module).__name__
    except Exception:
        return type(bm).__name__


def _to_registry(arg):
    """Normalise the ``modules`` argument of :func:`run` into a :class:`ModelRegistry`.

    Accepted forms
    --------------
    * ``bm``                                single BendedModule or Interface (backward compat)
    * ``[bm1, bm2]``                        list of BendedModules / Interfaces (auto-named)
    * ``[(name, bm), ...]``                 list of (name, BendedModule/Interface) tuples
    * ``[(name, factory), ...]``            list of (name, callable) tuples — lazy
    * ``[(name, factory, hint_mb), ...]``   with memory hint
    * ``{name: bm_or_factory, ...}``        dict form
    """
    if isinstance(arg, dict):
        entries = []
        for name, val in arg.items():
            bm = _unwrap(val)
            if bm is not None:
                entries.append(ModelEntry(name, module=bm))
            elif callable(val):
                di = getattr(val, '_default_inputs', None)
                vi = getattr(val, '_views', None)
                vd = getattr(val, '_view_defaults', None)
                entries.append(ModelEntry(name, factory=val, default_inputs=di, views=vi, view_defaults=vd))
            else:
                raise TypeError(
                    f"Expected BendedModule, Interface, or callable factory for '{name}', got {type(val)}"
                )
        return ModelRegistry(entries)

    if isinstance(arg, (list, tuple)) and _unwrap(arg) is None:
        entries = []
        for item in arg:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                name = item[0]
                val  = item[1]
                hint = item[2] if len(item) >= 3 else None
                bm = _unwrap(val)
                if bm is not None:
                    entries.append(ModelEntry(name, module=bm, memory_hint_mb=hint))
                elif callable(val):
                    di = getattr(val, '_default_inputs', None)
                    vi = getattr(val, '_views', None)
                    vd = getattr(val, '_view_defaults', None)
                    entries.append(ModelEntry(name, factory=val, memory_hint_mb=hint,
                                              default_inputs=di, views=vi, view_defaults=vd))
                else:
                    raise TypeError(f"Expected BendedModule, Interface, or callable for '{name}'")
            else:
                bm = _unwrap(item)
                if bm is not None:
                    entries.append(ModelEntry(_module_class_name(bm), module=bm))
                else:
                    raise TypeError(f"Unexpected item in modules list: {type(item)}")
        return ModelRegistry(entries)

    # Single BendedModule or Interface (backward compat)
    bm = _unwrap(arg)
    if bm is not None:
        return ModelRegistry([ModelEntry(_module_class_name(bm), module=bm)])

    raise TypeError(f"Unsupported modules argument type: {type(arg)}")


# ── public API ────────────────────────────────────────────────────────────────

def set_registry(registry):
    global _REGISTRY
    _REGISTRY = registry


def get_registry():
    return _REGISTRY


def set_sync_manager(sm):
    global _SYNC_MANAGER
    _SYNC_MANAGER = sm


def get_sync_manager():
    return _SYNC_MANAGER


def set_module(bended_module):
    """Backward-compatible single-module setter."""
    global _REGISTRY
    try:
        name = type(bended_module._module).__name__
    except Exception:
        name = type(bended_module).__name__
    _REGISTRY = ModelRegistry([ModelEntry(name, module=bended_module)])


def get_module():
    """Return the currently active BendedModule (backward-compatible)."""
    if _REGISTRY is None:
        return None
    return _REGISTRY.current


def set_default_inputs(defaults):
    global _DEFAULT_INPUTS
    _DEFAULT_INPUTS = dict(defaults) if defaults else {}


def get_default_inputs():
    return _DEFAULT_INPUTS


def get_current_default_inputs():
    """Return per-model default inputs, falling back to the global defaults only when
    there is no registry entry (single-model backward-compat path)."""
    if _REGISTRY is not None:
        entry = _REGISTRY._entries.get(_REGISTRY.current_name)
        if entry is not None:
            return entry.default_inputs   # may be {} — don't inject another model's globals
    return _DEFAULT_INPUTS


def set_views(views):
    """Set the global default per-node views ({node: str|dict|NodeView})."""
    global _VIEWS
    _VIEWS = _normalize_views(views)


def get_views():
    return _VIEWS


def get_current_views():
    """Return the active model's per-node views, falling back to the global defaults."""
    if _REGISTRY is not None:
        entry = _REGISTRY._entries.get(_REGISTRY.current_name)
        if entry and entry.views:
            return entry.views
    return _VIEWS


def set_view_defaults(view_defaults):
    """Set the global default views by tensor rank ({ndim: view_name})."""
    global _VIEW_DEFAULTS
    _VIEW_DEFAULTS = _normalize_view_defaults(view_defaults)


def get_view_defaults():
    return _VIEW_DEFAULTS


def get_current_view_defaults():
    """Return the active model's per-rank view defaults, falling back to the globals."""
    if _REGISTRY is not None:
        entry = _REGISTRY._entries.get(_REGISTRY.current_name)
        if entry and entry.view_defaults:
            return entry.view_defaults
    return _VIEW_DEFAULTS


def _is_per_model_views(views, registry):
    """True if *views* looks like ``{model_name: {node: spec}}`` rather than ``{node: spec}``."""
    if not isinstance(views, dict) or not views:
        return False
    names = set(registry._entries.keys())
    # every top-level key is a known model AND maps to a dict of node specs
    return all(k in names and isinstance(v, dict) for k, v in views.items())


def _confirm_erase(sm):
    """Prompt the user before wiping the sync directory."""
    from pathlib import Path
    p = Path(sm.sync_dir)
    try:
        ans = input(f"\n[graph_viewer] Erase sync dir '{p}'? All saved sessions and settings will be lost. [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"
        print()
    if ans == "y":
        sm.erase()
        print(f"[graph_viewer] Sync dir erased.")
    else:
        print(f"[graph_viewer] Erase skipped — keeping existing sync data.")


_SIZE_UNITS = {"": 1, "B": 1, "K": 1024, "KB": 1024, "M": 1024 ** 2, "MB": 1024 ** 2,
               "G": 1024 ** 3, "GB": 1024 ** 3, "T": 1024 ** 4, "TB": 1024 ** 4}


def parse_size(value) -> int:
    """Turn ``"16GB"`` / ``"512 MB"`` / ``2_000_000`` into a byte count.

    A bare number is already a byte count. Raises ValueError on anything else.
    """
    if isinstance(value, bool):
        raise ValueError("cache size must be a size, not a bool")
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().upper().replace(" ", "")
    for suffix in sorted(_SIZE_UNITS, key=len, reverse=True):
        if suffix and text.endswith(suffix):
            number = text[: -len(suffix)]
            break
    else:
        number, suffix = text, ""
    try:
        return int(float(number) * _SIZE_UNITS[suffix])
    except (ValueError, KeyError):
        raise ValueError(
            f"could not read '{value}' as a size — use a number of bytes or "
            f"a string like '512MB', '16GB'"
        ) from None


def set_verbose(level=True, stream=None):
    """Turn the graph viewer's activation trace on (or off).

    When on, every activation computation is printed as a nested, timed
    transcript: which UI action asked for what, which cache slots were hit,
    whether the run resumed from a clean ancestor (``from_activations``) or
    started from the inputs (``get_activations``), how much of the graph each
    rewrite actually cut, how long the run took and the shape/stats of every
    tensor produced.

    ``level`` accepts ``True`` (DEBUG), ``False`` (off), a level name such as
    ``"info"`` — which keeps only the coarse-grained lines — or an int level.
    Callable at any time, including while the server is running::

        tb.ui.graph_viewer.set_verbose(True)
    """
    from torchbend.tracing import activation_log as actlog
    return actlog.configure(level, stream=stream)


def run(modules, host="127.0.0.1", port=8765, open_browser=False, default_inputs=None,
        views=None, view_defaults=None, sync=None, erase=False, force=False, verbose=False,
        cache_size=None):
    """Launch the interactive graph viewer for one or multiple models.

    ``modules`` accepts a single ``BendedModule`` (backward compatible) or a
    collection of models for the multi-model picker.  Factories are called
    lazily — only when the user selects that model in the UI.

    Examples::

        # single model (backward compat)
        tb.ui.graph_viewer.run(bm)

        # dict form — BendedModules are pre-loaded, callables are lazy
        tb.ui.graph_viewer.run({
            "ResNet":  bm_resnet,
            "UNet":    lambda: load_and_trace_unet(),
        })

        # list of (name, factory, memory_hint_mb) tuples
        tb.ui.graph_viewer.run([
            ("ResNet",  bm_resnet),
            ("UNet",    make_unet, 120),   # hint: ~120 MB; skip if RAM too low
        ])

    ``views`` configures how nodes are displayed (see ``tb.ui.NodeView``)::

        # single model: {node_name: spec}, spec = view name or NodeView
        tb.ui.graph_viewer.run(rave, views={
            "audio_out": "audio",
            "z":         tb.ui.NodeView("spectrogram", sample_rate=16000),
            "logits":    tb.ui.NodeView("category", names=["cat", "dog"]),
        })

        # multiple models: {model_name: {node_name: spec}}
        tb.ui.graph_viewer.run({"A": bmA, "B": bmB},
                               views={"A": {"out": "audio"}, "B": {"out": "image"}})

    ``view_defaults`` sets the default view per tensor rank (ndim), applied to any
    node without a per-node ``views`` entry::

        tb.ui.graph_viewer.run(model,
                               view_defaults={2: "category", 3: "audio", 4: "image"})

    ``cache_size`` caps the activation cache, as a byte count or a string::

        tb.ui.graph_viewer.run(sg3, cache_size="16GB")

    It defaults to 4 GB (or ``$TORCHBEND_ACTIVATION_CACHE_MAX_BYTES``). Raise it
    for large generative models: an activation too big to store is not cached at
    all, so every later request has nothing to resume from and recomputes the
    whole prefix. The verbose trace reports each refusal.

    ``verbose`` prints a nested, timed trace of every activation computation —
    each ``get_activations`` / ``from_activations`` call, the graph rewrites they
    perform, the cache hits that avoided them, and the tensors produced.  Accepts
    ``True``, a level name (``"info"`` for the coarse lines only) or an int
    level; see :func:`set_verbose` to toggle it while the server runs.
    """
    import os
    import django
    from django.conf import settings as django_settings

    if verbose:
        set_verbose(verbose)

    if cache_size is not None:
        # Must land in the environment before django.setup() reads the settings
        # module; also applied directly when settings are already configured.
        max_bytes = parse_size(cache_size)
        os.environ["TORCHBEND_ACTIVATION_CACHE_MAX_BYTES"] = str(max_bytes)
        if django_settings.configured:
            django_settings.ACTIVATION_CACHE_MAX_BYTES = max_bytes
        print(f"[graph_viewer] Activation cache limit: {max_bytes / 1024 ** 3:.2f} GB")

    registry = _to_registry(modules)
    set_registry(registry)

    if default_inputs:
        set_default_inputs(default_inputs)
        # Propagate global defaults into pre-loaded entries that don't have their own.
        # Lazy factories should declare their own via _default_inputs; if they don't,
        # an empty dict is intentional (no cross-model injection of global inputs).
        for _e in registry._entries.values():
            if _e.is_loaded and not _e.default_inputs:
                _e.default_inputs = dict(default_inputs)

    if views:
        if _is_per_model_views(views, registry):
            for mname, mviews in views.items():
                registry._entries[mname].views = _normalize_views(mviews)
        else:
            # flat {node: spec} → global default for every model
            set_views(views)

    if view_defaults:
        set_view_defaults(view_defaults)

    # Sync: initialise manager, run pre-flight checks (interactive CLI prompts),
    # and register a post-load hook for lazy models.
    _sm = None
    if sync is not None:
        sync_dir = sync  # avoid shadowing by the `.sync` submodule import below
        from .sync import SyncManager, startup_sync_check, post_load_sync_check
        _sm = SyncManager(sync_dir)
        if erase:
            if force:
                _sm.erase()
            else:
                _confirm_erase(_sm)
        set_sync_manager(_sm)
        registry._post_load_hook = lambda n, e: post_load_sync_check(n, e, _sm)

    # Pre-load only the first model so the page has something to show on boot.
    # All other lazy entries remain unloaded until the user selects them.
    first_entry = next(iter(registry._entries.values()), None)
    if first_entry and not first_entry.is_loaded and first_entry.factory is not None:
        ok, err = registry.select(first_entry.name)
        if not ok:
            import warnings
            warnings.warn(f"[graph_viewer] Could not pre-load first model: {err}")

    # Sync: startup checks run after eager models are loaded (fingerprint needs module).
    if _sm is not None:
        startup_sync_check(registry, _sm)

    os.environ["DJANGO_SETTINGS_MODULE"] = "torchbend.ui.graph_viewer.settings"
    if not django_settings.configured:
        django.setup()

    if open_browser:
        import threading as _threading
        import webbrowser

        def _open():
            import time
            time.sleep(1.5)
            webbrowser.open(f"http://{host}:{port}/")

        _threading.Thread(target=_open, daemon=True).start()

    from django.core.management import execute_from_command_line
    execute_from_command_line(["manage.py", "runserver", f"{host}:{port}", "--noreload"])
