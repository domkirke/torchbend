"""Structured trace logging for activation computation.

Every path that actually *runs a graph* — ``BendedModule.get_activations`` /
``from_activations``, the graph-level ``graph_get_activations`` /
``graph_from_activations`` rewrites, and the graph viewer's lazy activation
cache — emits a line here.  The result is a nested, timed transcript of what
was really computed, as opposed to what was requested::

    [act] ▸ cache.request  fn=forward  targets=[relu_1]  input=a3f9c2
    [act]     inputs   x = [1×3×224×224] float32
    [act]     slots    hit=[]  miss=[relu_1]  (clean=3 dirty=2 none=41, 12.3 MB / 512.0 MB)
    [act]     frontier conv1_bended  (nearest clean ancestor)
    [act]   ▸ cache.from_frontier  targets=[relu_1]  frontier=[conv1_bended]
    [act]       conv1_bended = [1×64×112×112] float32 μ=0.41 σ=0.59
    [act]       graph_from_activations  roots=[conv1_bended]  48 nodes → 44 nodes (-4)  0.81 ms
    [act]       graph_get_activations   targets=[relu_1]  44 nodes → 6 nodes (-38)  0.42 ms
    [act]       run BendedGraphModule.forward  6 nodes  2.10 ms
    [act]       relu_1 = [1×64×112×112] float32 μ=0.33 σ=0.44
    [act]   ◂ cache.from_frontier  3.51 ms
    [act]     store    +12.2 MB → 24.5 MB / 512.0 MB  (0 evicted)
    [act] ◂ cache.request  3.77 ms

Nothing is formatted unless the ``torchbend.activations`` logger is enabled at
the relevant level, so the instrumentation is free when verbose mode is off.

Turn it on with :func:`configure` (or ``graph_viewer.run(..., verbose=True)``).
"""

import logging
import threading
import time
from collections import Counter
from contextlib import contextmanager

import torch


LOGGER_NAME = "torchbend.activations"

#: Loggers covered by :func:`configure` — the whole graph-viewer verbose surface.
TORCHBEND_LOGGERS = (
    LOGGER_NAME,
    "torchbend.bending_session",
    "torchbend.bending_module",
)

_logger = logging.getLogger(LOGGER_NAME)
_state = threading.local()

_INDENT = "  "


# ── enablement ───────────────────────────────────────────────────────────────

def enabled(level: int = logging.DEBUG) -> bool:
    """True when the activation logger would emit at *level* (cheap guard)."""
    return _logger.isEnabledFor(level)


def _parse_level(level) -> int:
    if isinstance(level, bool):
        return logging.DEBUG if level else logging.WARNING
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        parsed = logging.getLevelName(level.upper())
        if isinstance(parsed, int):
            return parsed
    return logging.DEBUG


def configure(level=logging.DEBUG, stream=None, names=None, fmt="[%(name)s] %(message)s"):
    """Attach a stream handler to the torchbend loggers and set their level.

    ``level`` accepts a bool, an int level or a level name ("debug", "info").
    Passing ``False`` mutes them again.  Idempotent: re-calling only updates
    the level, it never stacks handlers.
    """
    lvl = _parse_level(level)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt))
    for name in (names or TORCHBEND_LOGGERS):
        lg = logging.getLogger(name)
        if not any(getattr(h, "_torchbend_verbose", False) for h in lg.handlers):
            handler._torchbend_verbose = True
            lg.addHandler(handler)
            # our own handler already prints it; don't let the root logger echo
            lg.propagate = False
        lg.setLevel(lvl)
    return lvl


# ── nesting & timing ─────────────────────────────────────────────────────────

def _depth() -> int:
    return getattr(_state, "depth", 0)


def log(msg, *args, level=logging.DEBUG):
    """Emit one indented trace line (no-op when the logger is disabled)."""
    if not _logger.isEnabledFor(level):
        return
    _logger.log(level, _INDENT * _depth() + (msg % args if args else msg))


def tick():
    """Start a timer, or return None when logging is off."""
    return time.perf_counter() if _logger.isEnabledFor(logging.DEBUG) else None


def tock(t0) -> str:
    """Format the elapsed time since :func:`tick` (empty string if not timing)."""
    if t0 is None:
        return ""
    return "%.2f ms" % ((time.perf_counter() - t0) * 1000.0)


class _Step:
    """Handle yielded by :func:`step`; carries a summary shown on the exit line."""

    def __init__(self):
        self._tail = ""

    def note(self, msg, *args, level=logging.DEBUG):
        log(msg, *args, level=level)

    def result(self, msg, *args):
        self._tail = (msg % args if args else msg)


@contextmanager
def step(label, detail="", level=logging.DEBUG):
    """Log an indented, timed ``▸ label … ◂ label  N ms`` block."""
    if not _logger.isEnabledFor(level):
        yield _Step()
        return
    log("▸ %s%s", label, ("  " + detail) if detail else "", level=level)
    _state.depth = _depth() + 1
    t0 = time.perf_counter()
    handle = _Step()
    try:
        yield handle
    finally:
        elapsed = (time.perf_counter() - t0) * 1000.0
        _state.depth = _depth() - 1
        log("◂ %s  %.2f ms%s", label, elapsed,
            ("  " + handle._tail) if handle._tail else "", level=level)


# ── formatting helpers ───────────────────────────────────────────────────────

def fmt_names(names, limit: int = 8) -> str:
    """``[a, b, c, … +12]`` — a bounded rendering of a node-name list."""
    names = list(names)
    if len(names) > limit:
        shown = ", ".join(str(n) for n in names[:limit])
        return "[%s, … +%d]" % (shown, len(names) - limit)
    return "[%s]" % ", ".join(str(n) for n in names)


def fmt_tensor(value) -> str:
    """``[1×64×112×112] float32 cuda:0 μ=0.41 σ=0.59 ∈[-2.1, 3.4]``."""
    if not torch.is_tensor(value):
        return type(value).__name__ if value is not None else "None"
    shape = "×".join(str(s) for s in value.shape) if value.dim() else "scalar"
    out = "[%s] %s" % (shape, str(value.dtype).replace("torch.", ""))
    if str(value.device) != "cpu":
        out += " " + str(value.device)
    try:
        if value.numel() and value.is_floating_point():
            flat = value.detach().float()
            out += " μ=%.4g" % flat.mean().item()
            if value.numel() > 1:
                out += " σ=%.4g" % flat.std().item()
            out += " ∈[%.4g, %.4g]" % (flat.min().item(), flat.max().item())
    except Exception:
        pass
    return out


def log_tensors(tensors, prefix="", level=logging.DEBUG):
    """Log ``name = <tensor summary>`` for each entry of a name→tensor dict."""
    if not _logger.isEnabledFor(level):
        return
    for name, value in tensors.items():
        log("%s%s = %s", prefix, name, fmt_tensor(value), level=level)


def fmt_graph(graph) -> str:
    """``48 nodes (2 placeholder, 30 call_function, 15 call_module, 1 output)``."""
    try:
        ops = Counter(n.op for n in graph.nodes)
    except Exception:
        return "<graph>"
    total = sum(ops.values())
    detail = ", ".join("%d %s" % (count, op) for op, count in sorted(ops.items()))
    return "%d nodes (%s)" % (total, detail)


def fmt_graph_delta(before, after) -> str:
    """``48 nodes → 6 nodes (-42)`` — how much of the graph was actually cut."""
    try:
        n_before = sum(1 for _ in before.nodes)
        n_after = sum(1 for _ in after.nodes)
    except Exception:
        return ""
    return "%d nodes → %d nodes (%+d)" % (n_before, n_after, n_after - n_before)


def fmt_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return "%.2f GB" % (n / 1024 ** 3)
    if n >= 1024 ** 2:
        return "%.1f MB" % (n / 1024 ** 2)
    if n >= 1024:
        return "%.1f kB" % (n / 1024)
    return "%d B" % n


def tensor_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size()
               for t in tensors.values() if torch.is_tensor(t))
