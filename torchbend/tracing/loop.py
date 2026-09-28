"""Compact loops for traced graphs.

A hardcoded Python ``for`` loop in a model's forward is *unrolled* by tracing:
N iterations become N independent copies of the body. That is often what you
want -- every iteration's activations are individually bendable -- and it is
sometimes ruinous: a 10-step decode loop through a transformer produces an
18,500-node graph.

:func:`loop` gives a dial between the two. Written in place of a raw ``for``,
it emits one opaque ``torchbend_loop::loop_fwd`` node per *pack* of iterations
instead of inlining the body, so the graph stays the size of the loop rather
than the size of the loop times its body.

**What stays bendable.** Weight bending is unaffected at any setting: it
happens at the state-dict level, before the graph runs, and the packed op
receives its parameters as an explicit ``Tensor[]`` argument sourced from
``get_attr`` nodes of the already-bent module copy. The *carry* flowing
between packs is a real graph tensor, so it is bendable too (bend the
``getitem`` children of the loop node, which carry the shapes -- the loop node
itself has none). Only activations *inside* a packed body are invisible.

So: **pack size is the temporal resolution of activation bending.**
``pack=1`` keeps every iteration boundary bendable; ``pack=n_iter`` collapses
the loop to a single node; ``mode="unroll"`` is the old behaviour, with every
inner activation exposed.

Contract: **the carry's shapes and dtypes must survive an iteration
unchanged.** That is what lets the fake kernel be trivial, and it is why the
body is never executed under fake tensors while tracing (which is where the
cost of unrolling goes). Preallocate buffers and write into them rather than
growing a tensor. A violation raises :class:`BendingLoopError` on the first
real execution, naming the offending carry slot.

Outside tracing -- and under TorchScript -- :func:`loop` is transparent: it
just runs the Python loop, so it is safe to leave in production model code.
Packed graphs are not scriptable or exportable (the body lives in a process
registry), and the op has no autograd formula, so this is an inference-time
tool.

Usage::

    def step(i, carry):
        h, = carry
        return (self.block(h) + h,)

    h, = tb.loop(step, (h,), 32, name="refine", modules=[self.block])

and at trace time::

    bm.trace("forward", x=x,
             _loop_policy={"mode": "auto", "max_unroll": 8, "pack": 1})
"""

import contextlib
import contextvars
import itertools
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from torch.nn.utils.stateless import _reparametrize_module
except ImportError:                                   # pragma: no cover
    _reparametrize_module = None


__all__ = ["loop", "loop_policy", "current_loop_policy", "BendingLoopError",
           "DEFAULT_LOOP_POLICY", "LOOP_OP_NAME", "is_loop_node"]


LOOP_OP_NAME = "torchbend_loop::loop_fwd"


class BendingLoopError(Exception):
    pass


#: Applied to any :func:`loop` reached while tracing, unless ``trace`` is given
#: a ``_loop_policy``. Packing only kicks in past ``max_unroll`` iterations, and
#: then one node per iteration -- compact, but every iteration boundary is still
#: a bendable tensor.
DEFAULT_LOOP_POLICY = {"mode": "auto", "max_unroll": 8, "pack": 1}

_LOOP_POLICY: contextvars.ContextVar = contextvars.ContextVar(
    "torchbend_loop_policy", default=None)

# A custom op cannot take a Python callable, so bodies live here and travel
# through the graph as a readable string key instead.
_BODY_REGISTRY: Dict[str, "_LoopBody"] = {}
_KEY_COUNTER = itertools.count()


# ── the packed op ────────────────────────────────────────────────────────────

class _LoopBody:
    """One packed node's replay: the body, plus the modules whose parameters
    arrive as the op's ``params`` argument and must be substituted before the
    body runs (the body closes over the *original* modules; the op is handed
    the *bent* tensors)."""

    def __init__(self, body: Callable, modules: Sequence[nn.Module]):
        self.body = body
        self.specs: List[Tuple[nn.Module, List[str]]] = []
        for module in modules:
            names = list(dict(module.named_parameters()).keys())
            names += list(dict(module.named_buffers()).keys())
            self.specs.append((module, names))

    def flat_params(self) -> List[torch.Tensor]:
        """Every registered module's parameters and buffers, flattened in the
        order :meth:`run` expects to unpack them."""
        out: List[torch.Tensor] = []
        for module, names in self.specs:
            table = {**dict(module.named_parameters()),
                     **dict(module.named_buffers())}
            out.extend(table[n] for n in names)
        return out

    def run(self, carry: Tuple[torch.Tensor, ...], params: Sequence[torch.Tensor],
            n_iter: int, start: int) -> Tuple[torch.Tensor, ...]:
        with contextlib.ExitStack() as stack:
            index = 0
            for module, names in self.specs:
                substitution = {n: params[index + j] for j, n in enumerate(names)}
                index += len(names)
                if substitution and _reparametrize_module is not None:
                    stack.enter_context(_reparametrize_module(module, substitution))
            for i in range(start, start + n_iter):
                carry = _as_carry(self.body(i, carry))
        return carry


def _as_carry(carry) -> Tuple[torch.Tensor, ...]:
    if isinstance(carry, (tuple, list)):
        return tuple(carry)
    return (carry,)


def _check_carry(before: Sequence[torch.Tensor], after: Sequence[torch.Tensor],
                 loop_key: str) -> None:
    """The fixed-shape contract, enforced where it can actually be seen."""
    if len(after) != len(before):
        raise BendingLoopError(
            "loop %r returned a carry of %d tensor(s), but was given %d. A packed "
            "loop's body must return the carry it received, unchanged in structure."
            % (loop_key, len(after), len(before)))
    for i, (was, now) in enumerate(zip(before, after)):
        if tuple(now.shape) != tuple(was.shape) or now.dtype != was.dtype:
            raise BendingLoopError(
                "loop %r: carry slot %d changed from %s/%s to %s/%s across the "
                "iteration. Packed loops need a fixed-shape carry — preallocate "
                "the buffer and write into it (rather than growing a tensor), or "
                "trace this loop with mode='unroll'."
                % (loop_key, i, tuple(was.shape), was.dtype,
                   tuple(now.shape), now.dtype))


@torch.library.custom_op(LOOP_OP_NAME, mutates_args=())
def loop_fwd(carry: List[torch.Tensor], params: List[torch.Tensor],
             loop_key: str, n_iter: int, start: int) -> List[torch.Tensor]:
    entry = _BODY_REGISTRY.get(loop_key)
    if entry is None:
        raise BendingLoopError(
            "no loop body registered under %r. A packed loop only replays inside "
            "the process that traced it — re-trace after reloading." % loop_key)
    # cloned: the op declares no mutation, and a body is free to write in place
    out = entry.run(tuple(c.clone() for c in carry), params, n_iter, start)
    _check_carry(carry, out, loop_key)
    return list(out)


@loop_fwd.register_fake
def _(carry: List[torch.Tensor], params: List[torch.Tensor],
      loop_key: str, n_iter: int, start: int) -> List[torch.Tensor]:
    # The whole point: shapes come from the contract, not from running the body,
    # so tracing never pays for the loop at all.
    return [torch.empty_like(c) for c in carry]


def is_loop_node(node) -> bool:
    """True if an fx node is a packed-loop call."""
    target = getattr(node, "target", None)
    return getattr(target, "_name", None) == LOOP_OP_NAME


def stamp_loop_meta(graph) -> None:
    """Record which loop and which iterations each packed node stands for.

    Read straight off the call's own arguments, so nothing has to be threaded
    through the tracer. Node *names* are deliberately left alone — they are
    bending targets and viewer identities — so a label like ``refine[0:4]``
    is the display layer's business, built from this metadata.
    """
    import operator

    for node in graph.nodes:
        if not is_loop_node(node):
            continue
        args = node.args
        if len(args) < 5:
            continue
        loop_key, n_iter, start = args[2], args[3], args[4]
        meta = {"name": loop_key, "start": int(start), "n_iter": int(n_iter),
                "iters": (int(start), int(start) + int(n_iter))}
        node.meta["torchbend_loop"] = dict(meta)
        for user in node.users:
            if user.target in (operator.getitem, getattr(operator, "getitem")):
                slot = user.args[1] if len(user.args) > 1 else None
                user.meta["torchbend_loop"] = dict(meta, slot=slot, carry=True)


# ── policy ───────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def loop_policy(policy: Optional[dict] = None):
    """Install the packing policy every :func:`loop` in scope reads.

    Set by :meth:`BendedModule.trace` around tracing; outside it no policy is
    installed and :func:`loop` stays a plain Python loop.
    """
    resolved = dict(DEFAULT_LOOP_POLICY)
    resolved.update(policy or {})
    token = _LOOP_POLICY.set(resolved)
    try:
        yield resolved
    finally:
        _LOOP_POLICY.reset(token)


def current_loop_policy() -> Optional[dict]:
    """The policy in force, or None when not tracing."""
    return _LOOP_POLICY.get()


def _resolve(policy: dict, name: str, mode, pack, unroll_range) -> dict:
    """Per-loop trace config beats the call site, which beats the global default."""
    per_loop = (policy.get("loops") or {}).get(name, {})
    glob_range = policy.get("unroll_range")
    if not isinstance(glob_range, (tuple, list)):
        glob_range = None
    return {
        "mode": per_loop.get("mode", mode if mode is not None
                             else policy.get("mode", "auto")),
        "pack": per_loop.get("pack", pack if pack is not None
                             else policy.get("pack", 1)),
        "max_unroll": per_loop.get("max_unroll", policy.get("max_unroll", 8)),
        "unroll_range": per_loop.get("unroll_range", unroll_range
                                     if unroll_range is not None else glob_range),
    }


def _same_site(a: Callable, b: Callable) -> bool:
    """Whether two body callables are the same loop *site*.

    Compared by code object, not identity: a body written as a closure inside
    ``forward`` is a fresh function object on every call, yet it is the same
    loop every time — keyed by identity, re-tracing would mint a new key on
    each trace and grow the registry without bound.
    """
    code_a, code_b = getattr(a, "__code__", None), getattr(b, "__code__", None)
    if code_a is not None and code_b is not None:
        return code_a is code_b
    return a is b


def _register(name: str, body: Callable, modules: Sequence[nn.Module]) -> str:
    key = name
    existing = _BODY_REGISTRY.get(key)
    if existing is not None and not _same_site(existing.body, body):
        # genuinely different loops colliding on one name — keep both replayable
        key = "%s#%d" % (name, next(_KEY_COUNTER))
    _BODY_REGISTRY[key] = _LoopBody(body, modules)
    return key


# ── front end ────────────────────────────────────────────────────────────────

def _run_inline(body: Callable, carry: Tuple[torch.Tensor, ...],
                start: int, n_iter: int) -> Tuple[torch.Tensor, ...]:
    for i in range(start, start + n_iter):
        carry = _as_carry(body(i, carry))
    return carry


def _captured_tensors(body: Callable) -> Dict[str, torch.Tensor]:
    """Tensors the body closes over — which a packed loop cannot use.

    A packed body is *replayed* later, from the graph, so everything it reads
    has to arrive through the op. A tensor captured in the closure is instead
    frozen at the value it had while tracing (a fake tensor, in the
    proxy_tensor backend), and the loop would quietly compute nonsense from
    it. Modules are fine — those come in through ``modules=``.
    """
    captured: Dict[str, torch.Tensor] = {}
    names = getattr(getattr(body, "__code__", None), "co_freevars", ()) or ()
    cells = getattr(body, "__closure__", None) or ()
    for name, cell in zip(names, cells):
        try:
            value = cell.cell_contents
        except ValueError:                             # empty cell
            continue
        if isinstance(value, torch.Tensor):
            captured[name] = value
    return captured


def loop(body: Callable, carry, n_iter: int, *, name: Optional[str] = None,
         modules: Optional[Sequence[nn.Module]] = None, pack: Optional[int] = None,
         unroll_range: Optional[Tuple[int, int]] = None,
         mode: Optional[str] = None):
    """Run ``body`` ``n_iter`` times, compactly when traced.

    Args:
        body: ``body(i, carry) -> carry``, where carry is a tuple of tensors
            whose shapes and dtypes it must preserve (see the module docstring).
        carry: the initial carry — a tensor or a tuple of them.
        n_iter: how many iterations. A fixed Python int; this is not a
            data-dependent loop.
        name: the loop's name in the graph and in the trace policy. Defaults to
            the body's qualname.
        modules: the modules ``body`` calls. Their parameters are passed into
            the packed op explicitly, which is what lets **bent weights reach a
            packed loop**. Omit only for a body that touches no module.
        pack: iterations per graph node. 1 (the default when packing) keeps
            every iteration boundary bendable; ``n_iter`` collapses the loop to
            one node.
        unroll_range: ``(a, b)`` — inline iterations a..b fully and pack the
            rest, to open up the first few steps of a long loop.
        mode: ``"auto"`` (pack past ``max_unroll`` iterations), ``"pack"``, or
            ``"unroll"``. The trace policy overrides this.

    Returns:
        The final carry, as a tuple.
    """
    carry = _as_carry(carry)
    n_iter = int(n_iter)
    if n_iter <= 0:
        return carry

    policy = _LOOP_POLICY.get()
    if policy is None or torch.jit.is_scripting() or torch.jit.is_tracing():
        # not tracing (or not a graph we can pack into): plain Python loop
        return _run_inline(body, carry, 0, n_iter)

    name = name or getattr(body, "__qualname__", None) or "loop"
    resolved = _resolve(policy, name, mode, pack, unroll_range)
    if resolved["mode"] == "auto":
        resolved["mode"] = "pack" if n_iter > resolved["max_unroll"] else "unroll"
    if resolved["mode"] == "unroll":
        return _run_inline(body, carry, 0, n_iter)

    captured = _captured_tensors(body)
    if captured:
        raise BendingLoopError(
            "loop %r closes over tensor(s) %s. A packed loop replays its body from "
            "the graph, so a captured tensor would be frozen at its trace-time value "
            "— pass them through the carry instead (and return them unchanged), or "
            "trace this loop with mode='unroll'."
            % (name, ", ".join(sorted(captured))))
    if modules is None:
        warnings.warn(
            "loop %r is being packed with modules=None: if its body calls a "
            "module, that module's bent weights will not reach the packed op. "
            "Pass modules=[...] (or modules=[] to silence this)." % name)
    key = _register(name, body, modules or [])
    params = _BODY_REGISTRY[key].flat_params()

    step = max(1, int(resolved["pack"]))
    window = resolved["unroll_range"]
    i = 0
    while i < n_iter:
        if window is not None and window[0] <= i < window[1]:
            carry = _run_inline(body, carry, i, 1)
            i += 1
            continue
        size = min(step, n_iter - i)
        if window is not None and i < window[0]:
            size = min(size, window[0] - i)
        out = torch.ops.torchbend_loop.loop_fwd(list(carry), params, key, size, i)
        carry = tuple(out)
        i += size
    return carry
