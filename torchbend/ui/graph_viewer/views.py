import io
import os
import json
import logging as _logging
import traceback as _tb
import torch
from typing import Optional
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from torchbend.tracing import activation_log as _actlog
from . import (get_module, get_interface, get_registry, get_current_default_inputs,
               get_current_views, get_current_view_defaults, get_sync_manager)
from . import serializer as _serializer
from .default_inputs import Expr as _Expr, as_bench_value as _as_bench_value
from .serializer import serialize_graph, get_available_methods, serialize_tensor_for_viz
from . import spec_adapters
from .spec_adapters import FRONTEND
from .node_views import serialize_node, describe as describe_views, resolve as resolve_view


def _error_json(exc, status=400):
    """Return a JsonResponse with the error message and full traceback."""
    return JsonResponse({
        "error": str(exc),
        "traceback": _tb.format_exc(),
    }, status=status)


def _trace_error_json(exc, bended_module=None, fn=None, status=500):
    """Error payload for a failed trace / run, located in the *model's* code.

    ``str(exc)`` alone is close to useless for a tracing failure: the message
    comes from deep inside torch and says nothing about which line of the model
    produced it. This adds the innermost user frame with its surrounding source,
    the user-code call chain, and — from the partial graph the tracer left
    behind — the last node that made it through.
    """
    from torchbend.tracing.code import describe_exception
    payload = describe_exception(exc)
    payload["last_node"] = _last_traced_node(bended_module)
    if fn:
        payload["fn"] = fn
    return JsonResponse(payload, status=status)


def _last_traced_node(bended_module):
    """The last activation the tracer recorded before it stopped, if any."""
    tracer = getattr(bended_module, "_last_tracer", None)
    acts = getattr(tracer, "_activations", None) if tracer is not None else None
    if not acts:
        return None
    name = list(acts)[-1]
    props = acts[name]
    code = getattr(props, "code", None)
    return {
        "name":        name,
        "op":          getattr(props, "op", None),
        "target":      str(getattr(props, "target", "") or ""),
        "module_path": str(getattr(props, "module_path", "") or ""),
        "shape":       str(getattr(props, "shape", "") or ""),
        "code":        code.description if code is not None else None,
        "traced":      len(acts),
    }


def _get_bending_session():
    """Return (and lazily create) the BendingSession for the active model."""
    from .bending_session import BendingSession
    from .sync import _json_to_node_views
    registry = get_registry()
    if registry is None:
        return None
    entry = registry._entries.get(registry.current_name)
    if entry is None:
        return None
    if entry.session is None:
        entry.session = BendingSession()
        # snapshots before the bindings: a recalled snapshot is a binding that
        # needs its tensor to be restored
        sm = get_sync_manager()
        if sm is not None:
            entry.session.snapshots = sm.load_snapshots(entry.name)
            if entry.session.snapshots:
                print(f"[sync] Restored {len(entry.session.snapshots)} snapshot(s) for '{entry.name}'.")
        pending = getattr(entry, "_pending_sync_session", None)
        if pending is not None:
            bm = entry.module
            if bm is not None:
                try:
                    entry.session.session_from_json(bm, pending.get("bending", {}))
                    entry.session.update_mode = pending.get("update_mode", entry.session.update_mode)
                    entry.session.auto_threshold_ms = pending.get(
                        "auto_threshold_ms", entry.session.auto_threshold_ms)
                    entry.session.node_views = _json_to_node_views(pending.get("node_views", {}))
                    print(f"[sync] Restored session for '{entry.name}'.")
                except Exception as exc:
                    print(f"[sync] Warning: could not restore session for '{entry.name}': {exc}")
            entry._pending_sync_session = None
    return entry.session

def _get_play_session():
    """Return (and lazily create) the PlaySession for the active model."""
    from .play_session import PlaySession
    registry = get_registry()
    if registry is None:
        return None
    entry = registry._entries.get(registry.current_name)
    if entry is None:
        return None
    if getattr(entry, "play_session", None) is None:
        entry.play_session = PlaySession()
    return entry.play_session


def _invalidate_play_session():
    """Tell a compiled play session that the bending topology changed.

    Play mode caches a bent module, its compiled slices and its activations; all
    three are built against a fixed set of bendings, so anything that adds,
    removes or re-links one has to drop them. (Entering play mode recompiles
    anyway — this covers the editor and play being open side by side.)
    """
    try:
        play = _get_play_session()
        if play is not None and play.compiled:
            play.invalidate()
    except Exception:
        pass


def _play_mark_dirty(nodes):
    """Invalidate only *nodes* (and their descendants) in the play cache.

    The counterpart to :func:`_invalidate_play_session` for value changes: a
    slider move in the editor must not cost play mode its whole cache.
    """
    if not nodes:
        return
    try:
        play = _get_play_session()
        if play is not None and play.compiled:
            play.mark_dirty(list(nodes))
    except Exception:
        pass


def _sync_save_session():
    """Persist the current session to the sync folder if sync is enabled."""
    sm = get_sync_manager()
    if sm is None:
        return
    registry = get_registry()
    if registry is None:
        return
    name = registry.current_name
    entry = registry._entries.get(name)
    if entry is None or entry.session is None:
        return
    sm.save_session(name, entry.session)


# ── node view resolution ───────────────────────────────────────────────────────

def _view_base_node(node: str) -> str:
    """Strip the ``_bended`` suffix so a view chosen for a node applies to both."""
    return node[:-7] if node.endswith("_bended") else node


def _node_view_args(fn: str, node: str):
    """Return (session_selection, run_config NodeView) for resolving a node's view."""
    base = _view_base_node(node)
    session = _get_bending_session()
    session_sel = session.get_node_view(fn, base) if session else None
    run_cfg = None
    try:
        run_cfg = get_current_views().get(base)
    except Exception:
        run_cfg = None
    return session_sel, run_cfg


# ── the active interface's declarations ──────────────────────────────────────
# Everything the viewer knows about an interface comes from `iface.spec` (see
# torchbend.interfaces.spec). A bare module has none, and gets the permissive
# defaults: every method batches, runs everywhere, has no input modes.

def _spec():
    """The active interface's InterfaceSpec, or None for a bare module."""
    iface = get_interface()
    return iface.spec if iface is not None else None


def _method_spec(fn):
    """`fn`'s BoundMethod, or None when there is no interface or it is invalid."""
    spec = _spec()
    if spec is None:
        return None
    try:
        return spec.method(fn)
    except Exception as exc:
        _warn_once("declarations of %s are invalid: %s" % (type(spec.iface).__name__, exc))
        return None


def _tokens():
    spec = _spec()
    if spec is None:
        return None
    try:
        return spec.tokens
    except Exception as exc:
        _warn_once("declarations of %s are invalid: %s" % (type(spec.iface).__name__, exc))
        return None


def _token_eos_id():
    tokens = _tokens()
    try:
        return tokens.eos if tokens is not None else None
    except Exception:
        return None


def _token_decoder():
    """The active interface's token decoder, if it declares one.

    This is what lets a tensor of ids — or of logits — be read as the text it
    stands for, rather than as a grid of numbers.
    """
    tokens = _tokens()
    return tokens.decode if tokens is not None else None


def _token_vocab_size():
    """The active interface's tokenizer vocabulary size, if it has one.

    Lets the "decoded text" view tell an actual id/logits tensor apart from
    any other tensor a token decoder alone can't rule out — a vision model's
    patch embeddings, say — by size rather than by guessing from rank alone.
    """
    iface = get_interface()
    if iface is None:
        return None
    try:
        tok = getattr(iface, "tokenizer", None)
        return len(tok) if tok is not None else None
    except Exception:
        return None


def _serialize_activation(t, fn, node, sr_hint=None, declared_audio=False):
    """Serialize an activation tensor through the modular view system."""
    session_sel, run_cfg = _node_view_args(fn, node)
    try:
        rank_defaults = get_current_view_defaults()
    except Exception:
        rank_defaults = None
    # A node the model gives a sample rate to *is* audio — saying so is the
    # point of declaring one, so it opens as audio rather than as the channel
    # plots its rank would otherwise infer. Run config, so it outranks the
    # shape-inferred default and yields to a view the user has actually picked.
    if declared_audio and run_cfg is None and not session_sel:
        run_cfg = {"view": "audio"}
    return serialize_node(t, fn=fn, node=_view_base_node(node),
                          session_sel=session_sel, run_cfg=run_cfg,
                          rank_defaults=rank_defaults, sr_hint=sr_hint,
                          decode=_token_decoder(), eos_id=_token_eos_id(),
                          vocab_size=_token_vocab_size())


# Last known audio sample rates per placeholder name (updated on each audio upload)
_last_audio_sr: dict = {}

# (fn, placeholder) -> (raw expression, evaluated tensor). Keeps a generated
# input stable across requests; see _parse_inputs.
_last_eval: dict = {}


def _remember_fn(fn):
    """The method last opened on the current model, by either page -- so the
    graph editor and play mode open on the same one after a switch."""
    registry = get_registry()
    entry = registry._entries.get(registry.current_name) if registry else None
    if entry is not None and fn:
        entry.last_fn = fn


def _current_fn(methods):
    """The method a page should open on: the one last used, if it still exists."""
    registry = get_registry()
    entry = registry._entries.get(registry.current_name) if registry else None
    last = getattr(entry, "last_fn", None) if entry is not None else None
    if last in methods:
        return last
    return methods[0] if methods else ""


def index(request):
    registry = get_registry()
    bended_module = get_module()
    methods = get_available_methods(bended_module) if bended_module else []
    default_fn = _current_fn(methods)
    try:
        module_type = type(bended_module._module).__name__ if bended_module else "Unknown"
    except Exception:
        module_type = "Unknown"
    models = registry.list_entries() if registry else []
    import time as _time
    return render(request, "graph_viewer/index.html", {
        "methods": methods,
        "methods_json": json.dumps(methods),
        "default_fn": default_fn,
        "module_type": module_type,
        "static_v": int(_time.time()),
        "models_json": json.dumps(models),
    })


def api_methods(request):
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    return JsonResponse({"methods": get_available_methods(bended_module)})


def _batch_supported(fn):
    """Whether the active interface lets `fn` take a stacked batch (default yes)."""
    method = _method_spec(fn)
    return method.batch if method is not None else True


def _refuse_batch(fn):
    iface = get_interface()
    raise ValueError(
        "%s: '%s' takes one input at a time, so the bench's entries cannot be "
        "stacked into a batch -- turn batch off%s"
        % (type(iface).__name__, fn,
           " (or, in play mode, use the sequential mode to run them one after "
           "another)"))


def api_devices(request):
    """GET → devices this process has, the current one, and which methods the
    active interface expects to work on each (see ``Interface.device_compat``).

    Same device list play mode uses (`.devices`); the graph editor moves the
    module itself rather than a compiled runtime, through the session below.
    """
    from .devices import device_compat_for_methods, device_options
    bended_module = get_module()
    methods = get_available_methods(bended_module) if bended_module else []
    session = _get_bending_session()
    return JsonResponse({
        "devices": device_options(),
        "current": session.device if session is not None else "cpu",
        "compat": device_compat_for_methods(_spec(), methods),
    })


def _device_refusal(fn, device):
    """Why `fn` should not be moved to `device`, or None when nothing says so."""
    method = _method_spec(fn)
    if method is None or method.runs_on(device):
        return None
    return ("%s says %r is not expected to work on %r (its Method(devices=...))"
            % (type(method.iface).__name__, fn, device))


@csrf_exempt
def api_device_set(request):
    """POST {device} → move the active module there for graph exploration.

    Refused (400) when the active interface says the current method does not
    expect to work on that device (`Interface.device_compat`) -- the point is
    to fail here, with a clear reason, rather than on the first request that
    actually runs the graph.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No session"}, status=500)
    try:
        body = json.loads(request.body) if request.body else {}
    except Exception:
        body = {}
    device = body.get("device") or "cpu"
    fn = body.get("fn") or (get_available_methods(bended_module) or [""])[0]
    refused = _device_refusal(fn, device)
    if refused:
        return JsonResponse({"error": refused}, status=400)
    try:
        session.set_device(bended_module, device)
    except Exception as exc:
        return _error_json(exc, 400)
    return JsonResponse({"ok": True, "device": session.device})


def _display_opts(request):
    """Read the viewer's simplification switches off the query string.

    All of these are presentational: they change what the payload draws, never
    what was traced.  Defaults match the checkboxes in the graph options panel.
    """
    def flag(name, default=True):
        raw = request.GET.get(name)
        if raw is None:
            return default
        return raw.lower() not in ("0", "false", "no")

    expanded = request.GET.get("expand", "")
    # absent or "auto" means: let the graph decide how much detail fits
    raw_depth = request.GET.get("depth")
    auto_depth = raw_depth is None or raw_depth == "auto"
    try:
        module_depth = 0 if auto_depth else max(0, int(raw_depth))
    except (TypeError, ValueError):
        module_depth, auto_depth = 0, True
    return _serializer.DisplayOptions(
        prune_unreachable=flag("prune"),
        hide_shape_calc=flag("shape_calc"),
        merge_unpack=flag("unpack"),
        collapse_layout=flag("collapse"),
        expanded=frozenset(x for x in expanded.split(",") if x),
        scope=request.GET.get("scope", "") or "",
        module_depth=module_depth,
        auto_depth=auto_depth,
    )


def api_graph(request, fn):
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    try:
        data = serialize_graph(bended_module, fn=fn, display=_display_opts(request))
        _remember_fn(fn)
        data["input_modes"] = _declared_input_modes(fn)
        data["batch_supported"] = _batch_supported(fn)
        raw_inputs = get_current_default_inputs(fn)
        data["default_inputs"] = {k: _as_bench_value(v) for k, v in raw_inputs.items()}
        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


#: What a node-index entry carries. Deliberately small: the index covers every
#: node in the trace, and the browser holds all of it in memory to search.
_INDEX_FIELDS = ("id", "label", "op", "target", "op_name", "kind",
                 "shape", "dtype", "module_path", "order")


def api_node_index(request, fn):
    """Every node in the trace, whatever the view is currently showing.

    The activation search reads this instead of the drawn graph. Depth mode
    folds a module's contents into one node and the simplification toggles hide
    whole classes of node, but they are all still there in the trace and still
    worth searching for — so the search looks through this, and a result that is
    not currently drawn carries the `module_path` needed to go and open it.
    """
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    try:
        data = serialize_graph(bended_module, fn=fn, display=_serializer.FULL)
        nodes = [{k: n.get(k) for k in _INDEX_FIELDS}
                 for n in data.get("nodes", [])
                 if not n.get("is_compound") and not n.get("is_module_group")]
        return JsonResponse({"nodes": nodes, "aliases": data.get("aliases", {})})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ── continuing an edited sequence ─────────────────────────────────────────────
# Changing a word in a decoded sequence has to change what follows it, or the
# view shows a continuation the model never made. That means running the model
# again from the edit — one greedy step at a time, exactly as generation does —
# rather than re-reading logits that were computed from the original tokens.


#: A continuation is one forward pass per step; a long one on a real model is
#: a long wait, so the request is bounded rather than trusted.
_MAX_CONTINUE_STEPS = 64


def _text_positions(logits_row, decode, topk):
    """One position's worth of payload: what was chosen, and what nearly was."""
    probs = torch.softmax(logits_row.detach().float(), dim=-1)
    top_p, top_i = probs.topk(min(topk, probs.shape[-1]), dim=-1)
    ids = top_i.tolist()
    pieces = _serializer_pieces(ids, decode)
    return {"id": ids[0], "piece": pieces.get(ids[0], ""),
            "prob": round(float(top_p[0]), 5),
            "alts": [{"id": i, "piece": pieces.get(i, ""), "prob": round(float(p), 5)}
                     for i, p in zip(ids, top_p.tolist())]}


def _serializer_pieces(ids, decode):
    """``{id: piece}`` for a handful of ids, decoded one per row."""
    uniq = sorted(set(int(i) for i in ids))
    if not uniq:
        return {}
    try:
        out = decode(torch.tensor(uniq, dtype=torch.long).unsqueeze(1),
                     skip_special_tokens=False)
    except Exception:
        return {}
    return {i: str(p) for i, p in zip(uniq, out)}


def continue_greedy(logits_of, decode, prefix, steps, topk, eos=None):
    """Extend ``prefix`` greedily, stopping at ``eos`` or after ``steps``.

    Returns ``(positions, ids, stopped)``. Greedy on purpose: the same edit has
    to give the same continuation, or you could not tell what your change did
    from what the sampler did.
    """
    ids = torch.tensor([list(prefix)], dtype=torch.long)
    positions, stopped = [], "steps"
    with torch.no_grad():
        for _ in range(steps):
            logits = logits_of(ids)
            if isinstance(logits, dict):
                logits = logits.get("logits", next(iter(logits.values())))
            pos = _text_positions(logits[0, -1], decode, topk)
            positions.append(pos)
            ids = torch.cat([ids, torch.tensor([[pos["id"]]], dtype=torch.long)], dim=-1)
            # a sequence ends when the model says it does; the count is only the
            # safety net behind that
            if eos is not None and pos["id"] == int(eos):
                stopped = "eos"
                break
    return positions, ids[0].tolist(), stopped


@csrf_exempt
def api_text_continue(request):
    """POST {prefix, steps, topk} → greedily continue a sequence of token ids.

    This is what makes an edit in the decoded-text view honest: the tokens after
    the one you changed are predicted from your version of the sequence, not
    left over from the model's.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    iface = get_interface()
    tokens = _tokens()
    if tokens is None or not tokens.has_logits:
        return JsonResponse(
            {"error": "this model declares no way to continue a sequence"}, status=404)
    decode = _token_decoder()
    if decode is None:
        return JsonResponse({"error": "this model declares no token decoder"}, status=404)

    try:
        body = json.loads(request.body.decode() or "{}")
        prefix = [int(i) for i in (body.get("prefix") or [])]
        steps = max(0, min(int(body.get("steps") or 0), _MAX_CONTINUE_STEPS))
        topk = max(1, min(int(body.get("topk") or 5), 10))
    except Exception as exc:
        return JsonResponse({"error": "bad request: %s" % exc}, status=400)
    if not prefix:
        return JsonResponse({"error": "prefix is empty"}, status=400)

    eos = _token_eos_id() if body.get("stop_at_eos", True) else None

    import time as _time
    started = _time.time()
    try:
        positions, full, stopped = continue_greedy(tokens.logits, decode, prefix, steps, topk, eos)
    except Exception as exc:
        return _error_json(exc)
    pieces = _serializer_pieces(full, decode)
    return JsonResponse({
        "ok": True,
        "positions": positions,
        "ids": full,
        # the separators for the whole edited sequence, so the client can
        # reassemble it the way the tokenizer would
        "seps": _continue_separators(decode, full, pieces),
        # why it stopped, and where the end landed if the model produced one
        "stopped": stopped,
        "eos_at": (len(full) - 1) if stopped == "eos" else None,
        "run_ms": round((_time.time() - started) * 1000, 1),
    })


def _continue_separators(decode, ids, pieces):
    """What goes before each piece — asked per adjacent pair, in one decode."""
    seps = [""] * len(ids)
    if len(ids) < 2:
        return seps
    try:
        pairs = decode(torch.tensor([[ids[j - 1], ids[j]] for j in range(1, len(ids))],
                                    dtype=torch.long), skip_special_tokens=False)
    except Exception:
        return seps
    for j, pair in enumerate(pairs, start=1):
        left, right = pieces.get(ids[j - 1], ""), pieces.get(ids[j], "")
        if pair == left + right:
            seps[j] = ""
        elif left and right and pair.startswith(left) and pair.endswith(right):
            seps[j] = pair[len(left):len(pair) - len(right)]
    return seps


# ── interface options ─────────────────────────────────────────────────────────
# Settings the interface owns: how its tokenizer pads, what sizes it traces on.
# They persist across runs, and some of them make the current trace stale, which
# is what a spec's `needs` says and what the response repeats back.


def _option_specs():
    """``(spec, {name: Option})`` for the active interface, or ``(None, {})``."""
    spec = _spec()
    if spec is None:
        return None, {}
    try:
        return spec, spec.options
    except Exception as exc:
        _logging.getLogger(__name__).warning(
            "options for %s are invalid: %s", type(spec.iface).__name__, exc)
        return spec, {"__error__": str(exc)}


def api_options(request):
    """GET → the active interface's options and their current values."""
    spec, options = _option_specs()
    if spec is None:
        return JsonResponse({"options": [], "interface": None})
    if "__error__" in options:
        return JsonResponse({"error": options["__error__"]}, status=500)
    return JsonResponse({
        "interface": type(spec.iface).__name__,
        "options": spec.describe_options(FRONTEND),
    })


@csrf_exempt
def api_option_set(request, name):
    """POST {value} → set one option, returning the value that took effect."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    spec, options = _option_specs()
    if spec is None:
        return JsonResponse(
            {"error": "The active model is not an Interface, so it declares no options"},
            status=404)
    if "__error__" in options:
        return JsonResponse({"error": options["__error__"]}, status=500)
    if name not in options:
        return JsonResponse({"error": "no option %r" % name}, status=404)

    try:
        body = json.loads(request.body.decode() or "{}")
        value = body.get("value")
    except Exception:
        value = request.POST.get("value")

    try:
        applied = spec.set_option(name, value)
    except Exception as exc:
        return _error_json(exc)
    return JsonResponse({
        "ok": True,
        "name": name,
        # what the interface actually holds now, which a setter may have
        # clamped or normalized away from what was sent
        "value": applied,
        "needs": options[name].needs,
    })


# ── interface callbacks ───────────────────────────────────────────────────────
# A traced graph is only part of what a model can do.  An interface also carries
# whole operations — GPT-2's generate, RAVE's audio round trip — that run Python
# around the graph and cannot be traced themselves.  `callbacks` is where an
# interface declares those, and these two endpoints are how the viewer offers
# them: one to ask what exists, one to run it.


def _callback_specs():
    """``(iface, {name: description})`` for the active interface's callbacks."""
    spec = _spec()
    if spec is None:
        return None, {}
    try:
        return spec.iface, {c["name"]: c for c in spec.describe_callbacks(FRONTEND)}
    except Exception as exc:
        _logging.getLogger(__name__).warning(
            "callback spec for %s is invalid: %s", type(spec.iface).__name__, exc)
        return spec.iface, {"__error__": str(exc)}


def api_callbacks(request):
    """GET → the active interface's declared callbacks, or an empty list."""
    iface, specs = _callback_specs()
    if iface is None:
        return JsonResponse({"callbacks": [], "interface": None})
    if "__error__" in specs:
        return JsonResponse({"error": specs["__error__"]}, status=500)
    return JsonResponse({
        "interface": type(iface).__name__,
        "callbacks": list(specs.values()),
    })


def _coerce_callback_arg(spec, raw):
    """Turn one posted value into what the method expects.

    Everything arrives as a string over JSON/FormData; an empty string for an
    optional argument means "leave it out" rather than "pass an empty value",
    which is how `seed` stays None unless someone actually sets it.
    """
    kind = spec.get("type")
    if raw is None:
        return None if spec.get("optional") else spec.get("default")
    if isinstance(raw, str) and raw.strip() == "" and kind != "text":
        if spec.get("optional"):
            raise _Skip
        return spec.get("default")
    if kind in ("text", "str", "choice"):
        return raw
    if kind == "bool":
        return raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
    if kind == "int":
        return int(float(raw))
    if kind == "float":
        return float(raw)
    return raw


class _Skip(Exception):
    """This argument was left blank and has no business being passed."""


def _render_callback_result(iface, name, result, returns):
    """Shape a callback's return value for the browser, by declared medium."""
    if returns == "text":
        if isinstance(result, (list, tuple)):
            items = [str(x) for x in result]
        else:
            items = [str(result)]
        return {"kind": "text", "items": items}

    if isinstance(result, torch.Tensor):
        # Go through the same view system as an activation, rather than a
        # second renderer: a callback's audio then gets the audio view, its
        # picker, and a view choice that persists — all of it already built.
        # The synthetic node name is what that choice is remembered against.
        sr_hint = None
        if returns == "audio":
            declared = iface.spec.callback(name).returns
            sr_hint = (declared.rate(iface) if getattr(declared, "sample_rate", None) is not None
                       else getattr(iface, "sample_rate", None))
        fn_ns, node_ns = _callback_view_fn(iface), "__callback__%s" % name
        session_sel, run_cfg = _node_view_args(fn_ns, node_ns)
        # The declared medium picks the view when nothing else has — a model
        # that says it returns audio should come up as audio, not as a line
        # plot the user has to correct every time. It goes in as run config, so
        # it outranks the shape-inferred default but yields to a choice the
        # user has actually made.
        if run_cfg is None and not session_sel:
            run_cfg = _MEDIUM_VIEWS.get(returns)
        payload = serialize_node(result, fn=fn_ns, node=node_ns,
                                 session_sel=session_sel, run_cfg=run_cfg,
                                 rank_defaults=_safe_view_defaults(),
                                 sr_hint=sr_hint, decode=_token_decoder(),
                                 eos_id=_token_eos_id(), vocab_size=_token_vocab_size())
        # `medium` is what the interface declared; `view` is how it is being
        # drawn. Spreading the payload last would silently overwrite the first
        # with the second, and the declaration would never reach the client.
        # The namespace travels too, so the view picker knows where to record a
        # choice — a callback's output is not a node and has no name of its own.
        return {"medium": returns, "_view_fn": fn_ns, "_view_node": node_ns, **payload}

    return {"kind": "repr", "items": [repr(result)]}


#: Which view a declared medium asks for, when the user has not chosen one.
_MEDIUM_VIEWS = {"audio": {"view": "audio"}, "image": {"view": "image"}}


def _safe_view_defaults():
    try:
        return get_current_view_defaults()
    except Exception:
        return None


def _callback_view_fn(iface):
    """A namespace for callback view choices, kept apart from any method's."""
    return "__callbacks__%s" % type(iface).__name__


@csrf_exempt
def api_callback_run(request, name):
    """POST {args} → run a declared callback and return its result."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    iface, specs = _callback_specs()
    if iface is None:
        return JsonResponse({"error": "The active model is not an Interface, so it declares no callbacks"}, status=404)
    if "__error__" in specs:
        return JsonResponse({"error": specs["__error__"]}, status=500)
    if name not in specs:
        return JsonResponse({"error": "no callback %r" % name}, status=404)

    try:
        posted = json.loads(request.body.decode() or "{}")
    except Exception:
        posted = {k: v for k, v in request.POST.items()}

    kwargs = {}
    for spec in specs[name]["args"]:
        key = spec["name"]
        if key not in posted:
            continue
        try:
            kwargs[key] = _coerce_callback_arg(spec, posted[key])
        except _Skip:
            continue
        except (TypeError, ValueError):
            return JsonResponse(
                {"error": "argument %r: %r is not a valid %s" % (key, posted[key], spec.get("type"))},
                status=400)

    import time as _time
    started = _time.time()
    try:
        result = iface.spec.run_callback(name, **kwargs)
    except Exception as exc:
        return _error_json(exc)
    return JsonResponse({
        "ok": True,
        "name": name,
        "run_ms": round((_time.time() - started) * 1000, 1),
        "args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v))
                 for k, v in kwargs.items()},
        "result": _render_callback_result(iface, name, result, specs[name].get("returns")),
    })


def api_weights(request, fn, node):
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    try:
        graph = bended_module.graph(fn=fn)
        target = next((n for n in graph.nodes if n.name == node and n.op == "get_attr"), None)
        if target is None:
            return JsonResponse({"error": f"get_attr node '{node}' not found"}, status=404)

        target_parts = str(target.target).split(".")
        param_name = target_parts[-1]          # e.g. "weight", "bias"
        module_path = target_parts[:-1]        # e.g. ["conv1"] or ["encoder", "0"]

        module_type = None
        try:
            mod = bended_module._module
            for part in module_path:
                mod = getattr(mod, part)
            module_type = type(mod).__name__
        except Exception:
            pass

        def _read_weight():
            obj = bended_module._module
            for attr in target_parts:
                obj = getattr(obj, attr)
            return serialize_tensor_for_viz(obj, module_type=module_type, param_name=param_name)

        if request.GET.get("original", "false").lower() == "true":
            session = _get_bending_session()
            if session and session.bindings:
                bended_module.reset_bending()
                try:
                    result = _read_weight()
                finally:
                    session._rebuild_bindings(bended_module)
                return JsonResponse(result)

        return JsonResponse(_read_weight())
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_activate(request, fn):
    """Run the model and return activations.

    POST body may include a JSON ``nodes`` list to lazily compute only those
    nodes (and serve the rest from cache).  Omit ``nodes`` to return whatever
    is already clean in the cache without triggering new computation.

    Example body::

        { "nodes": ["conv1", "relu_1"], "original": false }
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    try:
        kwargs = _parse_inputs(request, bended_module, fn)
        if not kwargs:
            return JsonResponse({"error": "No valid inputs provided"}, status=400)
        session = _get_bending_session()

        # Optional list of specific nodes to compute (lazy mode).
        # Accepted as JSON body {"nodes": [...]} OR as a FormData field "nodes"
        # (the latter is used when inputs are sent as multipart/form-data).
        target_nodes = None
        try:
            body = json.loads(request.body)
            if isinstance(body.get("nodes"), list):
                target_nodes = body["nodes"]
        except Exception:
            pass
        if target_nodes is None and request.POST.get("nodes"):
            try:
                target_nodes = json.loads(request.POST["nodes"])
            except Exception:
                pass

        original = request.POST.get("original", "false").lower() == "true"
        if original:
            if session and session.bindings:
                # Strip _bended suffixes: the reset module has no bended nodes.
                orig_targets = (
                    [n for n in target_nodes if not n.endswith("_bended")]
                    if target_nodes else None
                )
                captured = session.get_original_activations(
                    bended_module, fn, kwargs,
                    lambda m, f, kw: _run_capture_activations(m, f, kw, target_nodes=orig_targets),
                )
            else:
                captured = _run_capture_activations(bended_module, fn, kwargs, session,
                                                    target_nodes=target_nodes)
        else:
            captured = _run_capture_activations(bended_module, fn, kwargs, session,
                                                target_nodes=target_nodes)

        # per tensor, not once for the run: a strided activation plays at its
        # own rate, and the card has to say the rate its clip will actually use
        return JsonResponse({
            name: _serialize_activation(
                t, fn, name,
                sr_hint=_infer_output_sr(t, kwargs, bended_module=bended_module,
                                         fn=fn, node=name),
                declared_audio=_declares_audio(name, fn, bended_module))
            for name, t in captured.items()
        })
    except Exception as e:
        # running the model can fail inside the user's forward exactly like
        # tracing does — locate it the same way
        return _trace_error_json(e, bended_module, fn)


@csrf_exempt
def api_activate_node(request, fn, node):
    """Lazily fetch a single node's activation, using the cache.

    Only the subgraph up to *node* is executed, starting from the nearest
    clean cached ancestor when one is available.

    Returns the same JSON structure as a single entry from ``api_activate``.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    try:
        kwargs = _parse_inputs(request, bended_module, fn)
        if not kwargs:
            return JsonResponse({"error": "No valid inputs provided"}, status=400)
        session = _get_bending_session()
        captured = _run_capture_activations(bended_module, fn, kwargs, session,
                                            target_nodes=[node])
        t = captured.get(node)
        if t is None:
            return JsonResponse({"error": f"No activation computed for '{node}'"}, status=404)
        return JsonResponse(_serialize_activation(
            t, fn, node,
            sr_hint=_infer_output_sr(t, kwargs, bended_module=bended_module,
                                     fn=fn, node=node),
            declared_audio=_declares_audio(node, fn, bended_module)))
    except Exception as e:
        return _error_json(e)


@csrf_exempt
def api_retrace(request, fn):
    """Re-trace the current model with user-supplied inputs and return the updated graph."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    try:
        kwargs = _parse_inputs(request, bended_module, fn)
        if not kwargs:
            return JsonResponse({"error": "No valid inputs provided"}, status=400)
        # Re-trace under the same configuration the method was first traced with.
        # Those keywords are not data the bench can supply — GPT-2 is traced with
        # `use_cache=False` because a Cache object cannot pass through fx, and
        # dropping it fails on the cache rather than on anything the user did.
        # Anything the bench does provide still wins.
        traced_with = bended_module.trace_kwargs(fn)
        # Say plainly which inputs the bench is missing, before the tracer does
        # it as "TypeError: missing argument" -- a placeholder with no default
        # of its own needs a value from somewhere.
        required = [n.name for n in bended_module.graph(fn=fn).nodes
                    if n.op == "placeholder" and not n.args]
        missing = [name for name in required if name not in kwargs and name not in traced_with]
        if missing:
            return JsonResponse({
                "error": "no input for %s on the bench -- add %s (or keep its "
                         "default) before retracing"
                         % (", ".join(missing), "one" if len(missing) == 1 else "them"),
            }, status=400)
        # ...and under the same tracer settings. A graph that needed
        # `_wrap_recurrent` to trace needs it to re-trace.
        traced_as = bended_module.trace_config(fn)
        # ...unless an option changed since: the interface applies it to the
        # model and says which of those settings it replaces
        method = _method_spec(fn)
        if method is not None:
            traced_as = {**traced_as, **method.before_retrace()}
        traced = bended_module.trace(fn=fn, reinit_bending_state=True, _return_out=True,
                                     **traced_as, **{**traced_with, **kwargs})
        session = _get_bending_session()
        if session:
            # the graph was replaced: nothing built from the old one may be
            # reused -- then keep what the trace just computed on these inputs,
            # so the first view after a retrace is not a second full run
            session.graph_changed(fn)
            output = traced[1] if isinstance(traced, tuple) and len(traced) == 2 else None
            if output is not None:
                try:
                    session.retain_output(bended_module, fn, kwargs, output)
                except Exception as exc:
                    _actlog.log("retain   trace output not kept: %s: %s",
                                type(exc).__name__, exc, level=_logging.WARNING)
        data = serialize_graph(bended_module, fn=fn, display=_display_opts(request))
        # The client swaps this in for the whole graph payload, so it has to
        # carry everything api_graph carries — without the declaration the
        # prompt toggle vanishes the moment a retrace happens.
        data["input_modes"] = _declared_input_modes(fn)
        return JsonResponse(data)
    except Exception as e:
        return _trace_error_json(e, bended_module, fn)


@csrf_exempt
def api_eval_expr(request, fn, node):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    try:
        body = json.loads(request.body)
        expr = body.get("expr", "").strip()
        if not expr:
            return JsonResponse({"error": "empty expression"}, status=400)
        graph = bended_module.graph(fn=fn)
        ph = next((n for n in graph.nodes if n.name == node), None)
        if ph is None:
            return JsonResponse({"error": f"node '{node}' not found"}, status=404)
        t = _eval_expr(expr, _node_scope(bended_module, ph, fn))
        return JsonResponse(serialize_tensor_for_viz(t))
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def _expected_image_channels(bended_module, ph_node, fn):
    """Return the expected number of image channels for a placeholder, or None."""
    try:
        acts = bended_module.activations("?.*", fn=fn)
        act = acts.get(ph_node.name)
        if act is not None and hasattr(act, "shape"):
            shape = list(act.shape)
            if len(shape) >= 3:
                return int(shape[-3])
    except Exception:
        pass
    return None


def _expected_audio_channels(bended_module, ph_node, fn):
    """Return the channel count a ``[B, C, L]`` audio placeholder was traced on.

    ``None`` when the trace left the channel axis symbolic (any count is fine)
    or the placeholder is not 3-D.
    """
    try:
        acts = bended_module.activations("?.*", fn=fn)
        act = acts.get(ph_node.name)
        if act is not None and hasattr(act, "shape"):
            shape = list(act.shape)
            if len(shape) == 3:
                return int(shape[-2])      # raises if the dim is a symbol
    except Exception:
        pass
    return None


# ── input modes ───────────────────────────────────────────────────────────────
# A placeholder whose interface declares one can be fed the interface's way —
# GPT-2's `input_ids` as prose, tokenized on the way in — instead of as an
# expression or a file. The bench marks such a field with a companion
# `__mode__<name>` so the server knows to encode rather than eval.

_INPUT_MODE_PREFIX = "__mode__"


def _declared_input_modes(fn):
    """The active interface's input modes for `fn` that this viewer can
    collect, as the bench reads them; {} if it declares none."""
    method = _method_spec(fn)
    if method is None:
        return {}
    try:
        return {name: desc for name, desc in method.input_modes(FRONTEND).items()
                if spec_adapters.supported(method.modes()[name])}
    except Exception as exc:
        _logging.getLogger(__name__).warning(
            "input modes for %s are invalid: %s", type(method.iface).__name__, exc)
        return {}


def _encode_input_modes(request, fn):
    """Run every input-mode field the request carries.

    Returns ``(values, claimed)``: the tensors to use, and the placeholder names
    they cover. A mode that fills its siblings owns them — the bench may still
    hold an expression for `attention_mask`, but a prompt has just decided the
    sequence length and the two would not agree.
    """
    modes = _declared_input_modes(fn)
    if not modes:
        return {}, set()
    method = _method_spec(fn)
    values, claimed = {}, set()
    for name, spec in modes.items():
        marker = request.POST.get(_INPUT_MODE_PREFIX + name)
        if not marker or marker != spec["type"]:
            continue
        # what the request carries, read the way this value type is collected
        # (see spec_adapters): typed strings, decoded uploads, ...
        raws = spec_adapters.read_mode_input(method.modes()[name], request, name)
        if not raws:
            continue
        # Play mode stacks a placeholder's entries along the batch dimension, so
        # several prompts are one batched encode — the tokenizer pads them
        # together, which is the only way the mask and the ids can agree.
        produced = method.encode(name, raws[0] if len(raws) == 1 else raws)
        for key, tensor in produced.items():
            values[key] = tensor
        claimed.update(spec["fills"])
        _actlog.log("input    %s encoded from %d text value(s) via %s",
                    name, len(raws), spec["encode"])
    return values, claimed


def _parse_inputs(request, bended_module, fn):
    """Parse POST/FILES into a kwargs dict keyed by placeholder name.

    Input expressions are evaluated once and remembered per placeholder: a
    random one like ``torch.randn(1, 512)`` must yield the *same* tensor on
    every request, or each click would silently run the model on a different
    input — activations shown side by side would not belong to the same pass,
    and the activation cache, keyed by input, could never reuse anything.
    Re-evaluation happens when the expression itself changes, or on an explicit
    ``resample=true``.

    Side-effect: updates _last_audio_sr for any audio files found."""
    graph = bended_module.graph(fn=fn)
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    resample = str(request.POST.get("resample", "")).lower() == "true"
    kwargs, claimed = _encode_input_modes(request, fn)
    # Batch: every entry the bench sends for a placeholder is stacked along the
    # batch dimension, reconciled the way play mode does it (pad / loop /
    # stack). `sequential` has no meaning here -- the graph shows one pass.
    batch_on = str(request.POST.get("batch", "")).lower() in ("1", "true", "yes")
    if batch_on and not _batch_supported(fn):
        _refuse_batch(fn)
    batch_mode = (request.POST.get("batch_mode") or "pad").lower()
    if batch_mode not in ("pad", "loop", "stack"):
        batch_mode = "pad"
    for ph in placeholders:
        name = ph.name
        if name in claimed:
            continue          # an input mode already produced this one
        if batch_on:
            tensors = _batch_entries(request, bended_module, ph, fn, resample)
            if tensors:
                kwargs[name] = _combine_play_entries(name, tensors, batch_mode)
                _actlog.log("input    %s batched from %d entr%s (%s)", name, len(tensors),
                            "y" if len(tensors) == 1 else "ies", batch_mode)
            continue
        if name in request.FILES:
            expected_ch = _expected_image_channels(bended_module, ph, fn)
            expected_ac = _expected_audio_channels(bended_module, ph, fn)
            t, sr = _file_to_tensor(request.FILES[name], expected_channels=expected_ch,
                                    expected_audio_channels=expected_ac)
            if t is not None:
                kwargs[name] = t
                if sr is not None:
                    _last_audio_sr[name] = sr
        elif name in request.POST:
            raw = request.POST[name].strip()
            expect = _expected_input_kind(bended_module, ph, fn)
            t = None
            try:
                t = _coerce_input(json.loads(raw), expect)
            except Exception:
                pass
            if t is None:
                key = (fn, name)
                prev = _last_eval.get(key)
                if prev is not None and prev[0] == raw and not resample:
                    t = prev[1]          # same expression as last time: same tensor
                    _actlog.log("input    %s reused (expression unchanged)", name)
                else:
                    try:
                        t = _eval_expr(raw, _node_scope(bended_module, ph, fn), expect)
                        if t is not None:
                            _last_eval[key] = (raw, t)
                            _actlog.log("input    %s evaluated from '%s'", name, raw)
                    except Exception:
                        pass
            if t is not None:
                kwargs[name] = t
    _notify_interface_of_inputs(fn, kwargs)
    return kwargs


def _batch_entries(request, bended_module, ph, fn, resample):
    """Every entry the bench sent for one placeholder, as tensors, in order.

    Files first, then expressions -- each expression evaluated once and
    remembered under its position, as the single-entry path does, so a random
    one does not change between runs.
    """
    name = ph.name
    files, exprs = [], []
    if name in request.FILES:
        expected_ch = _expected_image_channels(bended_module, ph, fn)
        expected_ac = _expected_audio_channels(bended_module, ph, fn)
        for f in request.FILES.getlist(name):
            t, sr = _file_to_tensor(f, expected_channels=expected_ch,
                                    expected_audio_channels=expected_ac)
            if t is None:
                raise ValueError("Input '%s': could not decode file %r"
                                 % (name, getattr(f, "name", "?")))
            files.append(t)
            if sr is not None:
                _last_audio_sr[name] = sr
    expect = _expected_input_kind(bended_module, ph, fn)
    for i, raw in enumerate(request.POST.getlist(name)):
        raw = (raw or "").strip()
        if not raw:
            continue
        t = None
        try:
            t = _coerce_input(json.loads(raw), expect)
        except Exception:
            pass
        if t is None:
            key = ("batch", fn, name, i)
            prev = _last_eval.get(key)
            if prev is not None and prev[0] == raw and not resample:
                t = prev[1]
            else:
                t = _eval_expr(raw, _node_scope(bended_module, ph, fn), expect)
                if t is not None:
                    _last_eval[key] = (raw, t)
        if t is None:
            raise ValueError("Input '%s': could not evaluate %r" % (name, raw))
        exprs.append(t)
    # back into the bench's order, when it says what it was ("fef": file,
    # expression, file); files first otherwise
    order = request.POST.get("__order__" + name, "")
    if order and len(order) == len(files) + len(exprs):
        fi, ei = iter(files), iter(exprs)
        return [next(fi) if kind == "f" else next(ei) for kind in order]
    return files + exprs


def _notify_interface_of_inputs(fn, kwargs):
    """Tell the active interface what this run is on, if it wants to know.

    Best-effort by design: an interface that mishandles the notification must
    not take the run down with it.
    """
    iface = get_interface()
    if iface is None:
        return
    try:
        iface.on_inputs(fn, kwargs)
    except Exception as exc:
        _logging.getLogger(__name__).warning(
            "%s.on_inputs failed: %s", type(iface).__name__, exc)


#: Python types a placeholder can want instead of a tensor.
_SCALAR_KINDS = {"int": int, "float": float, "bool": bool, "str": str}


def _expected_input_kind(bended_module, ph, fn):
    """What this placeholder wants: ``"int"``, ``"float"``, … or a torch dtype.

    An argument the model uses as an *index* must arrive as a Python int, not as
    a tensor: Bark's fine model slices with ``codebook_idx + 1``, and a
    ``tensor(2.)`` there raises "only integer tensors of a single element can be
    converted to an index". Equally, token ids handed over as float32 fail in
    the embedding lookup. So the declared type decides, not a blanket cast.
    """
    try:
        param = _param_for_placeholder(ph.name, _fn_signature_params(bended_module, fn))
    except Exception:
        param = None
    try:
        act = bended_module.activations("?.*", fn=fn).get(ph.name)
    except Exception:
        act = None
    shape = getattr(act, "shape", None)
    arg_type = _placeholder_arg_type(param, getattr(act, "type", None), shape, None)
    if arg_type in _SCALAR_KINDS:
        return arg_type
    dtype = None
    try:
        dtype = getattr(act, "dtype", None) or _serializer._dtype_of(ph)
    except Exception:
        pass
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.replace("torch.", ""), None)
    return dtype if isinstance(dtype, torch.dtype) else None


def _coerce_input(value, expect):
    """Shape a parsed value to what the placeholder declared."""
    if expect in _SCALAR_KINDS:
        if isinstance(value, torch.Tensor):
            value = value.item()
        return _SCALAR_KINDS[expect](value)
    if isinstance(value, torch.Tensor):
        return value.to(expect) if isinstance(expect, torch.dtype) else value
    # a traced integer tensor stays integer; anything unknown keeps the old float32
    return torch.tensor(value, dtype=expect if isinstance(expect, torch.dtype)
                        else torch.float32)


def _node_scope(bended_module, ph_node, fn):
    shape = None
    try:
        acts = bended_module.activations("?.*", fn=fn)
        act = acts.get(ph_node.name)
        if act is not None and hasattr(act, "shape"):
            shape = [int(s) for s in act.shape]
    except Exception:
        pass
    node = {"name": ph_node.name, "op": ph_node.op, "shape": shape}
    return {"node": node, "shape": shape}


def _eval_expr(expr, scope, expect=None):
    import numpy as np
    import builtins
    safe_builtins = {k: getattr(builtins, k) for k in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float",
        "int", "len", "list", "map", "max", "min", "range",
        "round", "sum", "tuple", "type", "zip",
    )}
    ns = {"__builtins__": safe_builtins, "torch": torch,
          "np": np, "numpy": np, **scope}
    result = eval(expr, ns)  # noqa: S307 — local dev tool
    if isinstance(result, torch.Tensor) and expect not in _SCALAR_KINDS:
        return result
    return _coerce_input(result, expect)


def _get_output_feeder_names(bended_module, fn):
    """Return names of the nodes that directly feed the FX output op."""
    try:
        graph = bended_module.graph(fn=fn)
        names = []
        for node in graph.nodes:
            if node.op != "output":
                continue
            ret = node.args[0] if node.args else None
            if isinstance(ret, torch.fx.Node):
                names.append(ret.name)
            elif isinstance(ret, (tuple, list)):
                for item in ret:
                    if isinstance(item, torch.fx.Node):
                        names.append(item.name)
        return names
    except Exception:
        return []


def _run_capture_activations(bended_module, fn, kwargs, session=None, target_nodes=None):
    """Compute activations for *target_nodes* only, using the cache when available.

    target_nodes=None  — default: compute only the output-feeder node(s) if not
                         already cached, then return all clean cached entries.
                         This seeds the cache on the first call without computing
                         every intermediate activation.
    target_nodes=list  — lazily compute exactly those nodes (from the nearest
                         clean ancestor) and return them.
    """
    if _actlog.enabled(_logging.INFO):
        # name the endpoint that triggered this, so the trace reads as
        # "UI action → calculations performed"
        import sys as _sys
        _actlog.log("── %s(fn=%s, targets=%s)", _sys._getframe(1).f_code.co_name,
                    fn, _actlog.fmt_names(target_nodes) if target_nodes else "auto",
                    level=_logging.INFO)

    # When no explicit targets are requested, default to seeding the output
    # feeder(s) so the cache is never completely empty after the first call.
    effective_targets = target_nodes
    if effective_targets is None:
        effective_targets = _get_output_feeder_names(bended_module, fn) or None
        _actlog.log("auto targets = output feeder(s) %s",
                    _actlog.fmt_names(effective_targets or []))

    cached_exc = None
    if session is not None:
        try:
            result, warn = session.get_cached_activations(
                bended_module, fn, kwargs, target_nodes=effective_targets
            )
            if warn:
                import warnings as _warnings
                _warnings.warn(f"[graph_viewer] Activation cache: {warn}")
            return result
        except Exception as exc:
            import traceback as _tb_mod, warnings as _warnings
            cached_exc = exc
            _warnings.warn(
                "[graph_viewer] Activation cache error (falling back to full run):\n"
                + _tb_mod.format_exc()
            )

    # Uncached fallback (no session, or cache raised an unexpected error).
    if not effective_targets:
        _actlog.log("capture  no targets — nothing to compute")
        return {}
    _actlog.log("capture  uncached path (no session / cache error) — full get_activations")
    try:
        result = bended_module.get_activations(*effective_targets, fn=fn, **kwargs)
    except Exception:
        # Both paths run the same model on the same inputs. When the cached one
        # already failed it holds the root cause, located in the model's own
        # code; get_activations wraps its own failure in a BendingError raised
        # from here, which would point the error panel at this file.
        if cached_exc is not None:
            raise cached_exc
        raise
    return {k: v.detach() for k, v in result.items() if isinstance(v, torch.Tensor)}


#: What audio plays back at when nothing knows better. Matches the AudioView
#: option default -- the card's "sr N" label and the WAV it renders have to
#: agree, or the sound is the wrong speed and the label does not say so.
_DEFAULT_AUDIO_SR = 44100

#: Below this, browsers refuse to decode a WAV. A latent at one frame per 2048
#: samples is a legitimate thing to listen to, so rather than emit a file that
#: silently will not play, the clip is resampled up to here and keeps its
#: duration -- what you hear is the activation's envelope over the right span.
_MIN_PLAYABLE_SR = 8000


#: What an object calls its own rate. ``sample_rate`` is the convention; ``sr``
#: is common enough in audio code to be worth reading too.
_SR_ATTRS = ("sample_rate", "sr")


def _declared_sample_rate(bended_module=None):
    """The rate the model itself works at, if anything says so.

    An audio interface (RAVE, Bark, VITS, ...) knows its own rate. So does a
    bare ``nn.Module`` that carries one -- and asking only the interface meant a
    model wrapped without one had no way to say what rate it ran at, so every
    activation played at the generic default.
    """
    module = getattr(bended_module, "_module", None)
    for owner in (get_interface(), bended_module, module):
        if owner is None:
            continue
        for attr in _SR_ATTRS:
            try:
                sr = int(getattr(owner, attr, None) or 0)
            except (TypeError, ValueError):
                continue
            if sr > 0:
                return sr
    return None


def _input_audio_reference(input_kwargs):
    """(rate, length) of the audio actually fed in, or (None, None).

    Prefers a placeholder this run really carries: with several audio inputs
    remembered, the one that is not in ``input_kwargs`` says nothing about the
    lengths in this run.
    """
    fallback_sr = None
    for ph_name, sr in _last_audio_sr.items():
        t_in = input_kwargs.get(ph_name) if input_kwargs else None
        if torch.is_tensor(t_in):
            return sr, int(t_in.shape[-1])
        if fallback_sr is None:
            fallback_sr = sr
    return fallback_sr, None


def _traced_output_length(bended_module, fn):
    """How long the method's own output is, from the trace.

    The other end of the reference: a model whose input is text (Bark, a TTS
    stack) has no audio going in to measure an activation against, but what it
    returns is audio at the declared rate. Anything shorter is then a strided
    version of that, and plays over the same span.
    """
    if bended_module is None or not fn:
        return None
    try:
        acts = bended_module.activations("?.*", fn=fn)
        graph = bended_module.graph(fn=fn)
    except Exception:
        return None
    best = None
    for node in graph.nodes:
        if node.op != "output":
            continue
        for src in node.all_input_nodes:
            shape = getattr(acts.get(src.name), "shape", None)
            if shape:
                try:
                    n = int(shape[-1])
                except (TypeError, ValueError):
                    continue
                if best is None or n > best:
                    best = n
    return best


#: Complaints already made, so a per-request failure is reported once and not
#: on every activation of every run.
_warned: set = set()


def _warn_once(message):
    if message in _warned:
        return
    _warned.add(message)
    _logging.getLogger(__name__).warning("[torchbend] %s", message)


def _declared_node_sr(node, fn=None, shape=None, bended_module=None):
    """The rate the model says this node runs at, or None.

    The model is the only thing that actually knows -- see
    :mod:`torchbend.sample_rates`. The interface is asked first: it is the
    higher-level object and may compute a rate the bare module has no way to.

    The name is the one the model declared against: bending a node appends
    ``_bended`` to it, and a declaration must not stop applying the moment the
    node it names is the one you are bending.
    """
    if not node:
        return None
    node = _view_base_node(node)
    for owner in (get_interface(), bended_module):
        if owner is None:
            continue
        ask = getattr(owner, "sample_rate_for", None)
        if not callable(ask):
            continue
        try:
            rate = ask(node, fn=fn, shape=shape)
        except Exception as exc:
            # A broken declaration must not break the run -- but it must not
            # vanish either. Swallowed silently, the only symptom was "the
            # sample rate does nothing", with nothing anywhere to say why.
            _warn_once("sample rate for %r: %s" % (node, exc))
            continue
        if rate:
            return float(rate)
    return None


def _declares_audio(node, fn=None, bended_module=None):
    """True when the model names a rate for this node — so it is audio."""
    return _declared_node_sr(node, fn=fn, bended_module=bended_module) is not None


def _strided_audio(bended_module):
    """Does the model say every node is one timeline at a different stride?"""
    for owner in (get_interface(), bended_module):
        if owner is not None and getattr(owner, "_strided_audio_", False):
            return True
    return False


def _infer_output_sr(output_tensor, input_kwargs, default_sr=None,
                     bended_module=None, fn=None, node=None):
    """The rate to play a tensor back at.

    A rate the model declares for this node is the answer, and nothing else is
    consulted: :mod:`torchbend.sample_rates` is the only source that actually
    knows which nodes are audio and at what rate.

    Undeclared, the reference is whatever says what rate this model's audio runs
    at — the audio file fed in, or the interface's own ``sample_rate``. A node
    is placed on that timeline only for a model that has said it is uniformly
    strided (``_strided_audio_``): halve the length, halve the rate. Without
    that, the reference rate is used as it stands, because nothing has said the
    tensor is a resampled version of anything.
    """
    declared = _declared_node_sr(
        node, fn=fn, bended_module=bended_module,
        shape=(list(output_tensor.shape) if torch.is_tensor(output_tensor) else None))
    if declared:
        return declared

    strided = _strided_audio(bended_module)
    sr_in, len_in = _input_audio_reference(input_kwargs)
    if sr_in is None:
        # No audio came in, so the reference is whatever the model says its own
        # rate is, and failing that the generic default. The default is still a
        # rate to scale *from*: a strided model that nothing has been fed is the
        # ordinary case for a graph driven by indices or embeddings, and bailing
        # out here left every node at the default however short it was.
        sr_in = _declared_sample_rate(bended_module) or default_sr or _DEFAULT_AUDIO_SR
        # the reference length is only wanted for scaling, and only a model that
        # says it is uniformly strided gets scaled
        len_in = _traced_output_length(bended_module, fn) if strided else None
    if not len_in or not torch.is_tensor(output_tensor):
        return sr_in
    len_out = int(output_tensor.shape[-1])
    if len_out == len_in or not strided:
        return sr_in
    return max(1, round(sr_in * len_out / len_in))


@csrf_exempt
def api_activate_audio(request, fn, node):
    """Run model, extract activation for `node`, return it as a WAV file."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    try:
        kwargs = _parse_inputs(request, bended_module, fn)
        if not kwargs:
            return JsonResponse({"error": "No valid inputs provided"}, status=400)
        captured = _run_capture_activations(bended_module, fn, kwargs, _get_bending_session(),
                                            target_nodes=[node])
        t = captured.get(node)
        if t is None:
            return JsonResponse({"error": f"no activation for '{node}'"}, status=404)
        sr = _infer_output_sr(t, kwargs, bended_module=bended_module, fn=fn, node=node)
        b, c = _audio_selection(request)
        wav, enc_sr = _tensor_to_wav(t, sr, batch=b, channel=c)
        return HttpResponse(wav, content_type="audio/wav",
                            headers={"Content-Disposition": f'inline; filename="{node}.wav"',
                                     "X-Sample-Rate": str(enc_sr)})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def api_models(request):
    registry = get_registry()
    if registry is None:
        return JsonResponse({"models": []})
    return JsonResponse({"models": registry.list_entries()})


@csrf_exempt
def api_select_model(request, name):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    registry = get_registry()
    if registry is None:
        return JsonResponse({"error": "No registry"}, status=500)
    ok, err = registry.select(name)
    if not ok:
        return JsonResponse({"error": err}, status=400)
    bended_module = get_module()
    methods = get_available_methods(bended_module) if bended_module else []
    try:
        module_type = type(bended_module._module).__name__ if bended_module else name
    except Exception:
        module_type = name
    return JsonResponse({
        "name": name,
        "module_type": module_type,
        "methods": methods,
        "models": registry.list_entries(),
    })


def _find_call_line(src_lines, start_line, target_node):
    """Heuristically find the call-site line within a function's source lines.
    Returns a 1-based absolute file line number."""
    import re as _re
    op = target_node.op
    target = target_node.target

    patterns = []

    if op == "call_module":
        t = str(target)
        patterns.append(f"self.{t}(")
        last = t.split(".")[-1]
        if last != t:
            patterns.append(f"self.{last}(")

    else:  # call_function / call_method
        # Collect candidate names from multiple sources, longest-first so we
        # try the most specific match before the generic fallback.
        names = []

        # 1. Direct __name__ / __qualname__ (works for Python callables)
        for attr in ("__name__", "__qualname__"):
            v = getattr(target, attr, None)
            if isinstance(v, str) and 2 < len(v) < 30 and not v.startswith("<"):
                names.append(v.split(".")[-1])

        # 2. Parse the string representation — handles ATen ops such as
        #    "aten.relu.default", "torch._C._nn.linear", etc.
        t_str = str(target) if not isinstance(target, str) else target
        _skip = {"aten", "torch", "ops", "default", "tensor", "out",
                 "self", "int", "bool", "float", "none"}
        for word in _re.findall(r'[a-z][a-z0-9_]+', t_str):
            if word not in _skip and 2 < len(word) < 25:
                names.append(word)

        # 3. Node name itself — make_fx names nodes after the op:
        #    "relu_1" → try "relu", "conv2d_3" → try "conv2d"
        node_base = _re.sub(r'_\d+$', '', target_node.name)
        if node_base and 2 < len(node_base) < 25:
            names.append(node_base)

        # De-duplicate while preserving order
        seen: set = set()
        for name in names:
            if name not in seen:
                seen.add(name)
                if op == "call_method":
                    patterns.append(f".{name}(")
                else:
                    patterns.append(f"{name}(")

    # For torchbend mark ops, also search for the public mark() call
    t_str_full = str(target) if not isinstance(target, str) else target
    if "mark_tensor" in t_str_full or "mark_tensor_pre" in t_str_full:
        patterns = [p for p in patterns if p not in ("mark_tensor(", "mark_tensor_pre(")]
        patterns = [".mark(", "mark("] + patterns

    # Search body lines (skip the "def …" line at index 0)
    for i, raw in enumerate(src_lines):
        if i == 0:
            continue
        text = raw if isinstance(raw, str) else ""
        for pat in patterns:
            if pat in text:
                return start_line + i

    return start_line   # fallback: def line


def api_node_source(request, fn, node):
    """Return source for a graph node — whole file + highlight_line for the JS to scroll to."""
    import inspect
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    try:
        graph = bended_module.graph(fn=fn)
        target_node = next((n for n in graph.nodes if n.name == node), None)
        if target_node is None:
            return JsonResponse({"error": f"node '{node}' not found"}, status=404)

        src_file = None
        highlight_line = None
        name = None

        _TORCHBEND_INTERNAL = ("torchbend/tracing/mark.py",)

        # Priority 1: exact source location from custom tracer
        try:
            acts = bended_module.activations("?.*", fn=fn)
            act = acts.get(node)
            if act is not None and getattr(act, "code", None) is not None:
                sf = getattr(act.code, "source_file", None)
                sl = getattr(act.code, "source_line", None)
                sfn = getattr(act.code, "source_fn", None)
                # Skip if source points to a torchbend internal (e.g. mark.py):
                # the user frame was not properly resolved; fall through to Priority 2.
                if sf and sl and not any(sf.replace("\\", "/").endswith(p) for p in _TORCHBEND_INTERNAL):
                    src_file, highlight_line, name = sf, sl, sfn
        except Exception:
            pass

        # Priority 2: show the traced method and search within it for the call site
        if src_file is None:
            try:
                obj = getattr(type(bended_module._module), fn)
                fn_lines, fn_start = inspect.getsourcelines(obj)
                src_file = inspect.getfile(obj)
                name = getattr(obj, "__qualname__", fn)
                highlight_line = _find_call_line(fn_lines, fn_start, target_node)
            except (OSError, TypeError) as e:
                return JsonResponse({"error": f"Could not read source: {e}"}, status=404)

        if src_file is None:
            return JsonResponse({"error": "No source available for this node"}, status=404)

        # Return the whole file so the browser can show it with full context
        try:
            with open(src_file, "r", errors="replace") as fh:
                all_lines = fh.readlines()
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

        lines = [{"no": i + 1, "text": l.rstrip("\n\r")} for i, l in enumerate(all_lines)]
        return JsonResponse({
            "file": src_file,
            "highlight_line": highlight_line,
            "lines": lines,
            "name": name or node,
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def api_source(request, _file=None, _line=None):
    """Return source lines around a given file:line for display in the UI."""
    file_path = _file or request.GET.get("file", "").strip()
    try:
        line = _line if _line is not None else int(request.GET.get("line", 0))
    except (ValueError, TypeError):
        return JsonResponse({"error": "invalid line"}, status=400)
    try:
        context = min(20, max(1, int(request.GET.get("context", 6))))
    except (ValueError, TypeError):
        context = 6

    if not file_path:
        return JsonResponse({"error": "no file"}, status=400)
    try:
        with open(file_path, "r", errors="replace") as f:
            all_lines = f.readlines()
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    start = max(0, line - 1 - context)
    end = min(len(all_lines), line + context)
    lines = [
        {"no": start + i + 1, "text": all_lines[start + i].rstrip(), "highlight": start + i + 1 == line}
        for i in range(end - start)
    ]
    return JsonResponse({"file": file_path, "line": line, "lines": lines})


def _match_audio_channels(t, expected):
    """Coerce an ``[1, C, L]`` audio tensor to ``expected`` channels.

    A model traced on mono (RAVE) throws a Conv1d channel error the moment a
    stereo file is fed in; downmix to what the graph was traced for instead.
    """
    if expected is None or t is None or t.ndim != 3:
        return t
    c = t.shape[1]
    if c == expected or expected < 1:
        return t
    if expected == 1:
        return t.mean(dim=1, keepdim=True)
    if c == 1:
        return t.expand(-1, expected, -1).contiguous()
    if c > expected:
        return t[:, :expected, :].contiguous()
    # c < expected: pad by repeating the last channel
    pad = t[:, -1:, :].expand(-1, expected - c, -1)
    return torch.cat([t, pad], dim=1).contiguous()


def _file_to_tensor(f, expected_channels=None, expected_audio_channels=None):
    """Return (tensor, sample_rate_or_None). sample_rate is set for audio files."""
    data = f.read()
    ct = f.content_type or ""
    try:
        if ct.startswith("image/"):
            from PIL import Image
            import torchvision.transforms.functional as TF
            img = Image.open(io.BytesIO(data))
            # Detect natural channel count from PIL image mode
            _mode_channels = {'1': 1, 'L': 1, 'I': 1, 'F': 1, 'RGB': 3, 'P': 3,
                               'YCbCr': 3, 'LAB': 3, 'HSV': 3, 'RGBA': 4, 'CMYK': 3}
            detected_ch = _mode_channels.get(img.mode, 3)
            n_ch = expected_channels if expected_channels in (1, 3, 4) else detected_ch
            if n_ch == 1:
                img = img.convert("L")
            elif n_ch == 4:
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")
            return TF.to_tensor(img).unsqueeze(0), None    # [1, C, H, W]
        if ct.startswith("audio/"):
            try:
                import torchaudio
                t, sr = torchaudio.load(io.BytesIO(data))
            except Exception:
                import soundfile as sf
                audio, sr = sf.read(io.BytesIO(data))
                t = torch.from_numpy(
                    audio.T if audio.ndim > 1 else audio[None]
                ).float()
            t = _match_audio_channels(t.unsqueeze(0), expected_audio_channels)
            return t, int(sr)                              # [1, C, L]
    except Exception:
        pass
    return None, None


def _audio_selection(request):
    """(batch, channel) the client is currently looking at; channel -1 = all.

    ``batch_idx``, not ``batch`` — the run form already uses ``batch`` as the
    "stack the bench entries" flag.
    """
    def _int(name, default):
        try:
            return int(request.POST.get(name, request.GET.get(name, default)))
        except (TypeError, ValueError):
            return default
    return _int("batch_idx", 0), _int("channel", -1)


def _to_playable_rate(t, sr):
    """Raise a sub-audio rate to something a browser will decode.

    The clip keeps its duration: an activation at 21 frames per second becomes
    8 kHz audio of the same length, so the playhead and the "N s" label stay
    honest. Returns ``(tensor, encoded_rate)``.
    """
    sr = int(sr)
    if sr >= _MIN_PLAYABLE_SR or t.shape[-1] < 2:
        return t, max(1, sr)
    factor = _MIN_PLAYABLE_SR / max(1, sr)
    size = max(2, int(round(t.shape[-1] * factor)))
    try:
        import torch.nn.functional as F
        t = F.interpolate(t.unsqueeze(0), size=size, mode="linear",
                          align_corners=False).squeeze(0)
    except Exception:
        return t, max(1, sr)
    return t, _MIN_PLAYABLE_SR


def _tensor_to_wav(tensor, sr, batch=0, channel=-1):
    """Convert a tensor to WAV bytes. Handles [B,C,L], [C,L], [L] shapes.

    *batch* / *channel* pick the slice the user is viewing, so what plays is what
    is drawn — exporting batch 0 regardless made the two disagree as soon as the
    bench held more than one entry.

    Returns ``(wav_bytes, encoded_rate)``: the rate written into the file, which
    is the one asked for unless it was too low to be decodable.
    """
    t = tensor.detach().float().cpu()
    if t.ndim == 3:
        t = t[max(0, min(int(batch), t.shape[0] - 1))]      # → [C, L]
    if t.ndim == 1:
        t = t.unsqueeze(0)  # [1, L]
    if channel is not None and int(channel) >= 0 and t.shape[0] > 1:
        t = t[max(0, min(int(channel), t.shape[0] - 1))].unsqueeze(0)
    # clamp to avoid clipping artifacts
    t = t.clamp(-1.0, 1.0)
    t, sr = _to_playable_rate(t, sr)
    buf = io.BytesIO()
    try:
        import torchaudio
        # 16-bit PCM explicitly: torchaudio defaults to 32-bit float WAV for a
        # float tensor, which several browsers refuse to decode — the player
        # would just sit there silent.
        torchaudio.save(buf, t, sr, format="wav",
                        encoding="PCM_S", bits_per_sample=16)
        return buf.getvalue(), sr
    except Exception:
        import wave, numpy as np
        arr = t.numpy()
        n_ch, _ = arr.shape
        pcm = (arr * 32767).astype(np.int16)
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(n_ch)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            # interleave channels
            wf.writeframes(pcm.T.flatten().tobytes())
        return buf.getvalue(), sr


# ── bending API ───────────────────────────────────────────────────────────────

@csrf_exempt
def api_activate_save(request, fn, node):
    """POST → save activation as tensor (.pt), image (.png), or audio (.wav)."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    try:
        fmt = request.POST.get("format", "tensor")
        sr_raw = request.POST.get("sample_rate")
        kwargs = _parse_inputs(request, bended_module, fn)
        if not kwargs:
            return JsonResponse({"error": "No valid inputs provided"}, status=400)
        captured = _run_capture_activations(bended_module, fn, kwargs, _get_bending_session(),
                                            target_nodes=[node])
        t = captured.get(node)
        if t is None:
            return JsonResponse({"error": f"No activation for '{node}'"}, status=404)

        if fmt == "audio":
            # the client rarely knows the rate; the same inference the player
            # uses gives the file the speed it was auditioned at
            try:
                sr = (int(sr_raw) if sr_raw else
                      _infer_output_sr(t, kwargs, bended_module=bended_module,
                                       fn=fn, node=node))
            except (TypeError, ValueError):
                sr = _infer_output_sr(t, kwargs, bended_module=bended_module,
                                      fn=fn, node=node)
            wav, enc_sr = _tensor_to_wav(t, sr)
            return HttpResponse(wav, content_type="audio/wav",
                                headers={"Content-Disposition": f'attachment; filename="{node}.wav"',
                                         "X-Sample-Rate": str(enc_sr)})

        if fmt in ("image", "images"):
            from PIL import Image as _Image
            import numpy as _np
            tt = t.detach().float().cpu()
            # Normalise to [0,1]
            mn, mx = float(tt.min()), float(tt.max())
            rng = mx - mn or 1.0
            tt = (tt - mn) / rng
            # Flatten to a list of 2-D or RGB images
            frames = []
            def _to_frame(arr):
                if arr.ndim == 2:
                    return _Image.fromarray((arr.numpy() * 255).astype(_np.uint8), mode="L")
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                    arr = arr.permute(1, 2, 0).numpy()
                    mode = {1: "L", 3: "RGB", 4: "RGBA"}[arr.shape[2]]
                    return _Image.fromarray((arr * 255).astype(_np.uint8).squeeze(), mode=mode)
                return None
            if tt.ndim == 2: frames.append(_to_frame(tt))
            elif tt.ndim == 3: frames.extend([_to_frame(tt[i]) for i in range(min(tt.shape[0], 32))])
            elif tt.ndim == 4: frames.extend([_to_frame(tt[b, c]) for b in range(min(tt.shape[0], 4)) for c in range(min(tt.shape[1], 8))])
            frames = [f for f in frames if f is not None]
            if len(frames) == 1:
                buf = io.BytesIO(); frames[0].save(buf, format="PNG")
                return HttpResponse(buf.getvalue(), content_type="image/png",
                                    headers={"Content-Disposition": f'attachment; filename="{node}.png"'})
            # Multiple frames → GIF
            buf = io.BytesIO()
            frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], loop=0, duration=100)
            return HttpResponse(buf.getvalue(), content_type="image/gif",
                                headers={"Content-Disposition": f'attachment; filename="{node}.gif"'})

        # Default: PyTorch tensor
        buf = io.BytesIO()
        torch.save(t.detach().cpu(), buf)
        return HttpResponse(buf.getvalue(), content_type="application/octet-stream",
                            headers={"Content-Disposition": f'attachment; filename="{node}.pt"'})
    except Exception as exc:
        return _error_json(exc)


def api_bending_callbacks(request):
    """List all available BendingCallback types with their ui_descriptor."""
    from .bending_session import get_available_callbacks
    return JsonResponse({"callbacks": get_available_callbacks()})


@csrf_exempt
def api_bendings(request):
    """GET → list active bindings.  POST → add a new binding."""
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)

    if request.method == "GET":
        return JsonResponse({
            "bindings": session.list_bindings(),
            "bending_params": session.list_bending_params(),
            "update_mode": session.update_mode,
            "auto_threshold_ms": session.auto_threshold_ms,
        })

    if request.method == "POST":
        try:
            body = json.loads(request.body)
            fn = body.get("fn", "forward")
            node = body.get("node", "")
            cb_type = body.get("callback_type", "")
            params = body.get("params", {})
            existing_id = body.get("existing_id") or None
            bid = session.add_binding(bended_module, fn, node, cb_type, params,
                                      existing_id=existing_id)
            _invalidate_play_session()
            _sync_save_session()
            return JsonResponse({
                "id": bid,
                "bindings": session.list_bindings(),
                "bending_params": session.list_bending_params(),
                "update_mode": session.update_mode,
            })
        except Exception as exc:
            return _error_json(exc, 400)

    return JsonResponse({"error": "GET or POST required"}, status=405)


@csrf_exempt
def api_bending_detail(request, bid):
    """PATCH → update param values.  DELETE → remove the binding."""
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)

    if request.method == "PATCH":
        try:
            body = json.loads(request.body)
            if "vis_muted" in body:
                session.set_vis_muted(bended_module, bid, bool(body.pop("vis_muted")))
            if "name" in body:
                b = session.bindings.get(bid)
                if b is not None:
                    b["name"] = str(body.pop("name"))
            for param_name, value in body.items():
                session.update_param(bended_module, bid, param_name, float(value))
            # a value moved: invalidate only what that binding feeds, exactly as
            # the editor's own cache does — dropping everything would make play
            # mode recompute the whole model on every slider move
            _b = session.bindings.get(bid)
            _play_mark_dirty(_b.get("nodes", [_b["node"]]) if _b else [])
            _sync_save_session()
            return JsonResponse({
                "ok": True,
                "bindings": session.list_bindings(),
                "bending_params": session.list_bending_params(),
            })
        except Exception as exc:
            return _error_json(exc, 400)

    if request.method == "DELETE":
        try:
            session.remove_binding(bended_module, bid)
            _invalidate_play_session()
            _sync_save_session()
            return JsonResponse({
                "ok": True,
                "bindings": session.list_bindings(),
                "bending_params": session.list_bending_params(),
            })
        except Exception as exc:
            return _error_json(exc, 400)

    return JsonResponse({"error": "PATCH or DELETE required"}, status=405)


@csrf_exempt
def api_bending_mode(request):
    """PATCH → update update_mode and/or auto_threshold_ms."""
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)
    if request.method != "PATCH":
        return JsonResponse({"error": "PATCH required"}, status=405)
    try:
        body = json.loads(request.body)
        if "update_mode" in body:
            mode = body["update_mode"]
            if mode not in ("auto", "live", "manual"):
                return JsonResponse({"error": "invalid update_mode"}, status=400)
            session.update_mode = mode
        if "auto_threshold_ms" in body:
            session.auto_threshold_ms = float(body["auto_threshold_ms"])
        _sync_save_session()
        return JsonResponse({
            "update_mode": session.update_mode,
            "auto_threshold_ms": session.auto_threshold_ms,
        })
    except Exception as exc:
        return _error_json(exc, 400)


@csrf_exempt
def api_bending_export(request):
    """POST → export the current bended module as a TorchScript .pt file."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)
    try:
        data = session.export_torchscript(bended_module)
        return HttpResponse(
            data,
            content_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="bended_module.pt"'},
        )
    except Exception as exc:
        return _error_json(exc, 500)


# ── BendingParameter API ──────────────────────────────────────────────────────

@csrf_exempt
def api_bending_params(request):
    """GET → list BendingParameters.  POST → create a new one."""
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)

    if request.method == "GET":
        return JsonResponse({
            "bending_params": session.list_bending_params(),
            "bindings": session.list_bindings(),
        })

    if request.method == "POST":
        try:
            body = json.loads(request.body)
            name = body.get("name", "").strip()
            if not name:
                return JsonResponse({"error": "name is required"}, status=400)
            param_type = body.get("param_type", "float")
            # create_bending_param casts the value to the declared type (and
            # rejects an unknown type or an uncastable value) — don't pre-swallow
            # the error here, the client needs to hear about it.
            value = body.get("value", 0)
            range_min = body.get("range_min", None)
            range_max = body.get("range_max", None)
            if range_min is not None:
                range_min = float(range_min)
            if range_max is not None:
                range_max = float(range_max)
            session.create_bending_param(bended_module, name, value, range_min, range_max, param_type)
            _invalidate_play_session()
            _sync_save_session()
            return JsonResponse({
                "ok": True,
                "bending_params": session.list_bending_params(),
                "bindings": session.list_bindings(),
            })
        except Exception as exc:
            return _error_json(exc, 400)

    return JsonResponse({"error": "GET or POST required"}, status=405)


@csrf_exempt
def api_bending_param_detail(request, name):
    """PATCH → update BP value/range/weight/bias.  DELETE → remove BP."""
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)

    if request.method == "PATCH":
        try:
            body = json.loads(request.body)
            bp = session.bending_params.get(name)
            if bp is None:
                return JsonResponse({"error": f"BendingParameter '{name}' not found"}, status=404)
            # renaming first: everything below is keyed by the macro's name
            if "name" in body:
                name = session.rename_bending_param(bended_module, name, body["name"])
                bp = session.bending_params.get(name)
                # the driven callbacks' generated forward changed with it
                _invalidate_play_session()
            from .bending_session import param_type_str
            bp_type = param_type_str(bp)
            # clamps follow the macro's own type
            def _clamp_val(v):
                if v is None:
                    return None
                return int(round(float(v))) if bp_type == "int" else float(v)

            if "value" in body:
                # coerced against the declared type (raises on an impossible value)
                session.update_bending_param(name, body["value"], bended_module=bended_module)
            # A float macro is normalised: its own clamp is 0…1 and stays that
            # way. The range the client sends is the range it should *map onto*,
            # so it goes to the links, not to the clamp — editing the clamp would
            # quietly break the one invariant the whole scheme rests on.
            if bp_type == "float":
                if "min_clamp" in body or "max_clamp" in body:
                    lo = body.get("min_clamp", (session.bp_ranges.get(name) or (None, None))[0])
                    hi = body.get("max_clamp", (session.bp_ranges.get(name) or (None, None))[1])
                    session.set_macro_range(bended_module, name, lo, hi)
            # a bool macro's range is fixed to [0, 1] by BendingParameter itself —
            # leave its clamping alone whatever the client sends
            elif bp_type != "bool":
                if "min_clamp" in body:
                    bp.min_clamp = _clamp_val(body["min_clamp"])
                if "max_clamp" in body:
                    bp.max_clamp = _clamp_val(body["max_clamp"])
                if "clamp" in body:
                    bp.clamp = bool(body["clamp"])
            if "weight" in body:
                bp.weight.fill_(float(body["weight"]))
            if "bias" in body:
                bp.bias.fill_(float(body["bias"]))
            _play_mark_dirty([
                n for b in session.bindings.values()
                if name in (b.get("bp_links") or {}).values()
                for n in b.get("nodes", [b["node"]])
            ])
            _sync_save_session()
            return JsonResponse({
                "ok": True,
                "bending_params": session.list_bending_params(),
                "bindings": session.list_bindings(),
            })
        except Exception as exc:
            return _error_json(exc, 400)

    if request.method == "DELETE":
        try:
            session.delete_bending_param(bended_module, name)
            _invalidate_play_session()
            _sync_save_session()
            return JsonResponse({
                "ok": True,
                "bending_params": session.list_bending_params(),
                "bindings": session.list_bindings(),
            })
        except Exception as exc:
            return _error_json(exc, 400)

    return JsonResponse({"error": "PATCH or DELETE required"}, status=405)


@csrf_exempt
def api_bending_link(request, bid):
    """POST → link a callback param to a BendingParameter."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)
    try:
        body = json.loads(request.body)
        param_name = body.get("param_name", "")
        bp_name = body.get("bp_name", "")
        if not param_name or not bp_name:
            return JsonResponse({"error": "param_name and bp_name are required"}, status=400)
        # The range belongs to this attachment: sent when the user picks one at
        # link time, and again whenever they change it afterwards. Re-linking is
        # what applies it — the callback gets a fresh derived parameter with the
        # new arithmetic in place of the old one.
        rmin, rmax = body.get("range_min"), body.get("range_max")
        rng = None if (rmin is None or rmax is None) else (float(rmin), float(rmax))
        session.link_param(bid, param_name, bp_name, bended_module=bended_module,
                           range=rng, _replace_range=("range_min" in body or "range_max" in body))
        _invalidate_play_session()
        _sync_save_session()
        return JsonResponse({
            "ok": True,
            "bindings": session.list_bindings(),
            "bending_params": session.list_bending_params(),
        })
    except Exception as exc:
        return _error_json(exc, 400)


def api_bending_config_export(request):
    """GET → return the current session state as a JSON config."""
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)
    return JsonResponse({"config": session.session_to_json()})


@csrf_exempt
def api_bending_config_tbconfig_import(request):
    """POST → restore session from a .tbconfig dill file (raw bytes in body)."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)
    try:
        session.session_from_tbconfig(bended_module, request.body)
        _invalidate_play_session()
        _sync_save_session()
        return JsonResponse({
            "ok": True,
            "bindings": session.list_bindings(),
            "bending_params": session.list_bending_params(),
            "update_mode": session.update_mode,
        })
    except Exception as exc:
        return _error_json(exc, 400)


def api_bending_config_tbconfig(request):
    """GET → export the current session as a BendingConfig dill file (.tbconfig)."""
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)
    try:
        data = session.export_bending_config(bended_module)
        return HttpResponse(
            data,
            content_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="bending_config.tbconfig"'},
        )
    except Exception as exc:
        return _error_json(exc, 500)


@csrf_exempt
def api_bending_config_import(request):
    """POST → restore session state from a previously exported JSON config."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)
    try:
        body = json.loads(request.body)
        config = body.get("config", body)
        session.session_from_json(bended_module, config)
        _invalidate_play_session()
        _sync_save_session()
        return JsonResponse({
            "ok": True,
            "bindings": session.list_bindings(),
            "bending_params": session.list_bending_params(),
            "update_mode": session.update_mode,
        })
    except Exception as exc:
        return _error_json(exc, 400)


@csrf_exempt
def api_bending_reorder(request):
    """POST → reorder all bindings by supplying a new ordered list of IDs."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)
    try:
        body = json.loads(request.body)
        order = body.get("order", [])
        session.reorder_bindings(bended_module, order)
        return JsonResponse({
            "ok": True,
            "bindings": session.list_bindings(),
            "bending_params": session.list_bending_params(),
        })
    except Exception as exc:
        return _error_json(exc, 400)


@csrf_exempt
def api_bending_unlink(request, bid):
    """POST → unlink a callback param from its BendingParameter."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)
    try:
        body = json.loads(request.body)
        param_name = body.get("param_name", "")
        if not param_name:
            return JsonResponse({"error": "param_name is required"}, status=400)
        session.unlink_param(bid, param_name)
        return JsonResponse({
            "ok": True,
            "bindings": session.list_bindings(),
            "bending_params": session.list_bending_params(),
        })
    except Exception as exc:
        return _error_json(exc, 400)


# ── activation cache API ───────────────────────────────────────────────────────

@csrf_exempt
def api_cache(request):
    """GET → cache stats.  POST → update settings (max_mb, clear)."""
    session = _get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)

    if request.method == "GET":
        return JsonResponse(session.cache_stats())

    if request.method == "POST":
        try:
            body = json.loads(request.body) if request.body else {}
        except Exception:
            body = {}
        if "max_mb" in body:
            try:
                max_bytes = int(float(body["max_mb"])) * 1024 * 1024
                if max_bytes <= 0:
                    return JsonResponse({"error": "max_mb must be positive"}, status=400)
                session.set_cache_max_bytes(max_bytes)
            except (TypeError, ValueError) as exc:
                return JsonResponse({"error": str(exc)}, status=400)
        if body.get("clear"):
            clear_fn: Optional[str] = body.get("fn") or None
            session.clear_cache(fn=clear_fn)
        return JsonResponse({"ok": True, **session.cache_stats()})

    return JsonResponse({"error": "GET or POST required"}, status=405)


# ── play mode API ──────────────────────────────────────────────────────────────

# Placeholder value types the bench drives with a plain widget instead of a
# tensor expression — sent through as native Python, never wrapped in a tensor.
_PLAY_SCALAR_TYPES = ("bool", "int", "float", "str")


def _fn_signature_params(bended_module, fn):
    """``{param_name: inspect.Parameter}`` for the traced method, or ``{}``."""
    import inspect
    try:
        return dict(inspect.signature(getattr(bended_module._module, fn)).parameters)
    except Exception:
        return {}


def _param_for_placeholder(name, params):
    """The signature parameter a placeholder came from.

    fx sanitises placeholder names and may suffix a counter (``x`` → ``x_1``),
    so an exact hit is tried first and a de-suffixed one after.
    """
    import re
    if name in params:
        return params[name]
    stripped = re.sub(r"_\d+$", "", name)
    return params.get(stripped)


def _placeholder_arg_type(param, traced_type, shape, default):
    """The widget type for a placeholder: a scalar name, or ``"tensor"``.

    Read from the annotation first (it is what the author declared), then the
    type recorded while tracing, then the default's type. Anything with a traced
    shape — or nothing to go on — is a tensor.
    """
    import inspect
    for candidate in (getattr(param, "annotation", inspect._empty) if param else inspect._empty,
                      traced_type):
        if candidate in (None, inspect._empty):
            continue
        if candidate is bool: return "bool"
        if candidate is int: return "int"
        if candidate is float: return "float"
        if candidate is str: return "str"
        if candidate is torch.Tensor: return "tensor"
    if shape is None and default is not None:
        if isinstance(default, bool): return "bool"
        if isinstance(default, int): return "int"
        if isinstance(default, float): return "float"
        if isinstance(default, str): return "str"
    return "tensor"


def _play_placeholders(bended_module, fn):
    """Ordered placeholder metadata for the input bench.

    Each entry carries the traced shape, the seed value the editor uses
    (``default``), and — from the traced method's signature — whether the
    argument is optional and what kind of widget it deserves. Optional
    arguments are the ones the bench can leave out entirely: omitting them makes
    the model fall back to the very default it was traced with.
    """
    import inspect
    try:
        graph = bended_module.graph(fn=fn, bended=True)
    except Exception:
        return []
    try:
        acts = bended_module.activations("?.*", fn=fn)
    except Exception:
        acts = {}
    try:
        defaults = get_current_default_inputs(fn) or {}
    except Exception:
        defaults = {}
    params = _fn_signature_params(bended_module, fn)
    out = []
    for n in graph.nodes:
        if n.op != "placeholder":
            continue
        shape = None
        act = acts.get(n.name)
        if act is not None and getattr(act, "shape", None) is not None:
            try:
                shape = [int(s) for s in act.shape]
            except Exception:
                shape = None
        default = defaults.get(n.name)
        if default is not None:
            default = _as_bench_value(default)
            if not isinstance(default, str):
                default = str(default)

        param = _param_for_placeholder(n.name, params)
        sig_default = getattr(param, "default", inspect._empty) if param else inspect._empty
        optional = sig_default is not inspect._empty
        arg_type = _placeholder_arg_type(
            param, getattr(act, "type", None), shape,
            None if not optional else sig_default)
        # only the JSON-safe defaults are sent as values; the rest as a label
        if not optional:
            arg_default = None
        elif isinstance(sig_default, (bool, int, float, str)) or sig_default is None:
            arg_default = sig_default
        else:
            arg_default = repr(sig_default)
        out.append({
            "name": n.name,
            "shape": shape,
            # so the bench can suggest an expression the input will accept —
            # `torch.randn` into an integer placeholder fails at the embedding
            "dtype": _serializer._dtype_of(n),
            "default": default,
            "optional": optional,
            "arg_type": arg_type,
            "arg_default": arg_default,
            # A scalar argument that took part in control flow (or that an
            # ATen-level trace constant-folded) has no users left in the graph:
            # feeding it does nothing until the method is re-traced on the value
            # you want. The bench says so rather than letting you chase a dead
            # control.
            "used": len(n.users) > 0,
        })
    return out


#: Graph ops that can carry an activation bending — the same set the editor
#: offers on its node menu.
_BENDABLE_OPS = ("call_function", "call_module", "call_method", "get_attr")


def _bendable_nodes(bended_module, fn):
    """Nodes of *fn* a bending can be attached to.

    Carries the same fields the editor's activation list filters on — shape,
    target, degrees, source position — so play mode's picker can offer the very
    same query language instead of a lesser search of its own.
    """
    try:
        graph = bended_module.graph(fn=fn, bended=True)
    except Exception:
        return []
    try:
        acts = bended_module.activations("?.*", fn=fn)
    except Exception:
        acts = {}
    out = []
    for n in graph.nodes:
        if n.op not in _BENDABLE_OPS:
            continue
        act = acts.get(n.name)
        shape = None
        if act is not None and getattr(act, "shape", None) is not None:
            try:
                shape = [int(s) for s in act.shape]
            except Exception:
                shape = None
        code = getattr(act, "code", None) if act is not None else None
        out.append({
            "name":        n.name,
            "op":          n.op,
            "target":      str(getattr(n, "target", "") or ""),
            "shape":       shape,
            "module_path": str(getattr(act, "module_path", "") or "") if act is not None else "",
            "in_degree":   len(n.all_input_nodes),
            "out_degree":  len(n.users),
            "source_file": getattr(code, "source_file", None) if code is not None else None,
            "source_fn":   getattr(code, "source_fn", None) if code is not None else None,
        })
    return out


def _graph_aliases(bended_module, fn):
    """``{alias: [node, ...]}`` for *fn* — what ``#alias`` searches against."""
    try:
        return {k: list(v) for k, v in bended_module.aliases(fn=fn).items()}
    except Exception:
        return {}


def play(request, page_mode="play"):
    """Render the play-mode page (also generate mode's: same bench, other panes)."""
    import time as _time
    registry = get_registry()
    bended_module = get_module()
    methods = get_available_methods(bended_module) if bended_module else []
    default_fn = _current_fn(methods)
    try:
        module_type = type(bended_module._module).__name__ if bended_module else "Unknown"
    except Exception:
        module_type = "Unknown"
    # current registry model name — used to read the inputs shared by the editor
    current_model = registry.current_name if registry else ""
    return render(request, "graph_viewer/play.html", {
        "methods": methods,
        "methods_json": json.dumps(methods),
        "default_fn": default_fn,
        "module_type": module_type,
        "current_model": current_model or "",
        "page_mode": page_mode,
        "static_v": int(_time.time()),
    })


def api_play_devices(request):
    """Same answer as the editor's /api/devices/: one device per model, shared."""
    return api_devices(request)


@csrf_exempt
def api_play_release(request):
    """POST → drop play's compiled runtimes (call on leaving play mode)."""
    play = _get_play_session()
    if play is not None:
        try:
            play.release()
        except Exception:
            pass
    return JsonResponse({"ok": True})


@csrf_exempt
def api_play_compile(request):
    """POST {fn, device, scripted} → (re)compile the play runtime."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    play = _get_play_session()
    if play is None:
        return JsonResponse({"error": "No play session"}, status=500)
    try:
        body = json.loads(request.body) if request.body else {}
    except Exception:
        body = {}
    fn = body.get("fn") or "forward"
    _remember_fn(fn)
    prefer_scripted = bool(body.get("scripted", False))
    # One device per model, shared with the graph editor: a device asked for
    # here goes through the same session (and the same compatibility check),
    # and the runtime compiles wherever the model then is.
    session = _get_bending_session()
    device = body.get("device") or (session.device if session else "cpu")
    refused = _device_refusal(fn, device)
    if refused:
        return JsonResponse({
            "error": refused,
            "device": session.device if session else "cpu",
        }, status=400)
    if session is not None:
        try:
            session.set_device(bended_module, device)
        except Exception as exc:
            return _error_json(exc, 400)
        device = session.device
    try:
        status = play.compile(bended_module, fn, device,
                              prefer_scripted=prefer_scripted,
                              session=_get_bending_session())
        return JsonResponse({
            "ok": True,
            **status,
            # the editor session also holds macros not yet linked to a callback
            "macros": play.list_macros(bended_module, _get_bending_session()),
            # the editor's whole parameter catalogue, so the macro picker is
            # populated without a second round trip
            "available_params": play.known_params(bended_module, _get_bending_session()),
            # …and the bendable parameters no macro drives yet, so the picker can
            # offer to promote one
            "promotable_params": play.promotable_params(_get_bending_session()),
            # the active bendings, so play mode can drive their plain parameters
            # directly instead of insisting everything become a macro first
            "bindings": (_get_bending_session().list_bindings()
                         if _get_bending_session() else []),
            "bendable_nodes": _bendable_nodes(bended_module, fn),
            "aliases": _graph_aliases(bended_module, fn),
            "placeholders": _play_placeholders(bended_module, fn),
            "input_modes": _declared_input_modes(fn),
            "batch_supported": _batch_supported(fn),
        })
    except Exception as exc:
        # building the runtime re-enters the model's own code (bend_graph,
        # scripting) — locate the failure there rather than echoing torch
        return _trace_error_json(exc, bended_module, fn, status=400)


@csrf_exempt
def api_play_macro(request):
    """POST {name, value} → set a macro on the compiled runtime (fast path)."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    play = _get_play_session()
    if play is None or not play.compiled:
        return JsonResponse({"error": "Play session not compiled"}, status=409)
    try:
        body = json.loads(request.body)
        name = body.get("name", "")
        if not name:
            return JsonResponse({"error": "name is required"}, status=400)
        result = play.set_macro(bended_module, name, body.get("value"),
                                session=_get_bending_session())
        return JsonResponse({"ok": True, **result})
    except Exception as exc:
        # a bending callback is user code too — locate a failure inside it
        return _trace_error_json(exc, bended_module, play.fn, status=400)


def _coerce_play_scalar(raw, ptype, name):
    """Parse one bench field into the Python scalar the signature declares."""
    raw = str(raw).strip()
    try:
        if ptype == "bool":
            return raw.lower() in ("1", "true", "yes", "on")
        if ptype == "int":
            return int(float(raw))
        if ptype == "float":
            return float(raw)
    except Exception as exc:
        raise ValueError(f"Input '{name}': {raw!r} is not a valid {ptype}") from exc
    return raw


#: How several entries on one placeholder become a run.
#:
#: ``pad``        — the default: zero-pad the other dimensions up to the largest
#:                  entry, then concatenate. Entries that already agree are
#:                  untouched, so this is ``stack`` plus a safety net rather than
#:                  a different behaviour.
#: ``loop``       — same, but tiles the material instead of padding with zeros.
#: ``stack``      — strict ``torch.cat`` along dim 0; mismatched shapes are an
#:                  error rather than something to reconcile.
#: ``sequential`` — do not batch at all: one forward pass per entry, outputs
#:                  gathered afterwards. The fallback for anything a batch
#:                  cannot express (different ranks, a model that hard-codes its
#:                  batch size, or simply not wanting the shapes touched).
_PLAY_BATCH_MODES = ("pad", "loop", "stack", "sequential")
_PLAY_DEFAULT_BATCH_MODE = "pad"


def _align_batch_tensors(tensors, mode, name):
    """Reconcile everything but dim 0 so *tensors* can be concatenated.

    ``pad``  — zero-pad the trailing edge of every short dimension.
    ``loop`` — tile the material cyclically until it fills the dimension, then
               trim. For audio (and any 1-D signal) this repeats the sound
               rather than appending silence, which is usually what "make them
               the same length" was meant to achieve.
    """
    ranks = {t.dim() for t in tensors}
    if len(ranks) > 1:
        shapes = ", ".join(str(tuple(t.shape)) for t in tensors)
        raise ValueError(
            f"Input '{name}': entries have different ranks ({shapes}) — padding "
            f"reconciles sizes, it cannot add or drop dimensions. Switch the "
            f"batch mode to 'sequential' to feed them one after another.")
    rank = next(iter(ranks))
    target = [max(t.shape[d] for t in tensors) for d in range(rank)]
    out = []
    for t in tensors:
        for d in range(1, rank):          # dim 0 is the batch dim: never aligned
            if t.shape[d] == target[d]:
                continue
            if mode == "loop":
                reps = [1] * rank
                reps[d] = -(-target[d] // t.shape[d])       # ceil division
                t = t.repeat(*reps).narrow(d, 0, target[d])
            else:
                # F.pad counts dimensions from the last, in (before, after) pairs
                pad = [0] * (2 * rank)
                pad[2 * (rank - 1 - d) + 1] = target[d] - t.shape[d]
                t = torch.nn.functional.pad(t, pad)
        out.append(t)
    return out


def _combine_play_entries(name, tensors, mode):
    """Fold one placeholder's entries into the single tensor a batched run wants."""
    if len(tensors) == 1:
        return tensors[0]
    if mode in ("pad", "loop"):
        tensors = _align_batch_tensors(tensors, mode, name)
    try:
        return torch.cat(tensors, dim=0)
    except Exception as exc:
        shapes = ", ".join(
            str(tuple(t.shape)) if torch.is_tensor(t) else "?" for t in tensors)
        raise ValueError(
            f"Input '{name}': cannot stack batch entries along dim 0 "
            f"(entry shapes: {shapes}). Set the batch mode to 'pad' or 'loop' to "
            f"reconcile the sizes, or to 'sequential' to run them one at a time."
        ) from exc


def _entry_labels(request, name, n):
    """The labels the client sent for a placeholder's ``n`` entries, in the
    order the server reads them (files first, then typed values); ``name[i]``
    for any it did not."""
    try:
        labels = [str(x) for x in json.loads(request.POST.get("__labels__" + name) or "[]")]
    except Exception:
        labels = []
    return [labels[i] if i < len(labels) and labels[i] else "%s%d" % (name, i)
            for i in range(n)]


def _mode_recording(raw):
    """The recording an audio input mode was given, as ``(waveform [C, L], rate)``:
    an upload arrives decoded, a typed path is read. None when it is neither."""
    if isinstance(raw, tuple) and len(raw) == 2 and torch.is_tensor(raw[0]):
        return raw[0], int(raw[1])
    path = os.path.expanduser(str(raw or "").strip())
    if not path or not os.path.isfile(path):
        return None
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), int(sr)
    except Exception:
        return None


def _encode_input_modes_each(request, fn):
    """Like :func:`_encode_input_modes`, one encode per entry instead of one
    batched encode: ``({placeholder: [(label, {filled: tensor}, recording)]},
    claimed)`` -- ``recording`` is ``(waveform, rate)`` for an audio mode.

    Generate mode runs the entries one at a time, and names its files after
    them, so it needs them apart rather than padded into a batch.
    """
    modes = _declared_input_modes(fn)
    if not modes:
        return {}, set()
    method = _method_spec(fn)
    entries, claimed = {}, set()
    for name, spec in modes.items():
        marker = request.POST.get(_INPUT_MODE_PREFIX + name)
        if not marker or marker != spec["type"]:
            continue
        raws = spec_adapters.read_mode_input(method.modes()[name], request, name)
        if not raws:
            continue
        labels = _entry_labels(request, name, len(raws))
        audio = spec["type"] == "audio"
        entries[name] = [(label, method.encode(name, raw), _mode_recording(raw) if audio else None)
                         for label, raw in zip(labels, raws)]
        claimed.update(spec["fills"])
    return entries, claimed


def _collect_play_inputs(request, bended_module, fn, per_entry=False):
    """Read the bench into ``(scalars, {placeholder: [tensor, ...]})``.

    Entries are kept apart at this stage; how they become a run (batched or
    sequential) is decided by the caller. With ``per_entry``, the second item
    is ``{placeholder: [(label, {placeholder: tensor}, recording), ...]}``
    instead -- every entry on its own, input modes encoded one by one, and
    ``recording`` the ``(waveform, rate)`` of an entry that came in as audio.

    Like the editor's ``_parse_inputs``, an expression is evaluated once and
    remembered: ``torch.randn(...)`` must yield the *same* tensor on every run,
    or moving a macro would silently re-draw the input — you could never tell
    the macro's effect from the noise, and the activation cache (keyed by input)
    could never reuse anything. Re-evaluation happens when the expression
    changes, or on an explicit ``resample=true``.
    """
    metas = _play_placeholders(bended_module, fn)
    arg_types = {p["name"]: p.get("arg_type") for p in metas}
    resample = str(request.POST.get("resample", "")).lower() in ("1", "true", "yes")
    scalars, per_name, rates = {}, {}, {}
    if per_entry:
        mode_entries, claimed = _encode_input_modes_each(request, fn)
    else:
        encoded, claimed = _encode_input_modes(request, fn)
        for key, tensor in encoded.items():
            per_name[key] = [tensor]
    for name in [p["name"] for p in metas]:
        if name in claimed:
            continue          # an input mode already produced this one
        # scalar arguments (a temperature, a flag, a length) are driven by a plain
        # widget, not a tensor expression — pass the Python value straight through
        ptype = arg_types.get(name)
        if ptype in _PLAY_SCALAR_TYPES:
            raws = [r for r in request.POST.getlist(name) if str(r).strip() != ""]
            if raws:
                scalars[name] = _coerce_play_scalar(raws[0], ptype, name)
            continue
        tensors = []
        # files (may be several)
        if name in request.FILES:
            _ph = next(n for n in bended_module.graph(fn=fn).nodes
                       if n.name == name)
            expected_ch = _expected_image_channels(bended_module, _ph, fn)
            expected_ac = _expected_audio_channels(bended_module, _ph, fn)
            for f in request.FILES.getlist(name):
                t, sr = _file_to_tensor(f, expected_channels=expected_ch,
                                        expected_audio_channels=expected_ac)
                if t is None:
                    raise ValueError(
                        f"Input '{name}': could not decode file "
                        f"{getattr(f, 'name', '?')!r}")
                tensors.append(t)
                rates.setdefault(name, []).append(sr)
                if sr is not None:
                    _last_audio_sr[name] = sr
        # expressions / JSON (may be several)
        for i, raw in enumerate(request.POST.getlist(name)):
            raw = (raw or "").strip()
            if not raw:
                continue
            t = None
            err = None
            try:
                t = torch.tensor(json.loads(raw), dtype=torch.float32)
            except Exception:
                pass
            if t is None:
                key = ("play", fn, name, i)
                prev = _last_eval.get(key)
                if prev is not None and prev[0] == raw and not resample:
                    t = prev[1]          # same expression as last time: same tensor
                    _actlog.log("input    %s[%d] reused (expression unchanged)", name, i)
                else:
                    try:
                        ph = next(n for n in bended_module.graph(fn=fn).nodes if n.name == name)
                        t = _eval_expr(raw, _node_scope(bended_module, ph, fn))
                        if t is not None:
                            _last_eval[key] = (raw, t)
                            _actlog.log("input    %s[%d] evaluated from '%s'", name, i, raw)
                    except Exception as exc:
                        err = exc
            if t is None:
                raise ValueError(
                    f"Input '{name}': could not evaluate {raw!r}"
                    + (f" — {err}" if err else ""))
            tensors.append(t)
        if tensors:
            per_name[name] = tensors
    if per_entry:
        entries = dict(mode_entries)
        for name, tensors in per_name.items():
            labels = _entry_labels(request, name, len(tensors))
            srs = rates.get(name, [])
            entries[name] = [(labels[i], {name: t},
                              (t, srs[i]) if i < len(srs) and srs[i] else None)
                             for i, t in enumerate(tensors)]
        return scalars, entries
    return scalars, per_name


def missing_inputs(bended_module, fn, provided, hint=""):
    """Raise a readable error when a required input has no value.

    Without this the model runs with the input set to None, and fails on
    whichever op first touches it -- "Expected a proper Tensor but got None
    for argument #0 'input'", from deep inside the model, naming nothing.
    """
    required = [p["name"] for p in _play_placeholders(bended_module, fn)
                if not p.get("optional")]
    missing = [name for name in required if name not in provided]
    if missing:
        raise ValueError("no input for %s -- add %s on the bench%s"
                         % (", ".join(missing), "one" if len(missing) == 1 else "them", hint))


def _parse_play_input_sets(request, bended_module, fn):
    """Parse the bench into ``(list of kwargs, mode)`` — one dict per forward pass.

    Every mode but ``sequential`` yields exactly one pass. ``sequential`` yields
    one per entry: a placeholder with fewer entries than the longest reuses its
    last, so "one varying latent against a fixed condition" needs no ceremony.
    """
    mode = (request.POST.get("batch_mode") or _PLAY_DEFAULT_BATCH_MODE).lower()
    if mode not in _PLAY_BATCH_MODES:
        mode = _PLAY_DEFAULT_BATCH_MODE
    batch_on = request.POST.get("batch", "0").lower() in ("1", "true", "yes")
    if batch_on and mode != "sequential" and not _batch_supported(fn):
        _refuse_batch(fn)
    scalars, per_name = _collect_play_inputs(request, bended_module, fn)
    missing_inputs(bended_module, fn, set(scalars) | set(per_name))
    if not batch_on:
        # batching off: a placeholder keeps its first entry and nothing else
        mode = "stack"
        per_name = {k: v[:1] for k, v in per_name.items()}

    if mode == "sequential":
        n_runs = max((len(v) for v in per_name.values()), default=0)
        sets = []
        for i in range(max(n_runs, 1)):
            kw = dict(scalars)
            for name, tensors in per_name.items():
                kw[name] = tensors[min(i, len(tensors) - 1)]
            sets.append(kw)
        return sets, mode

    kwargs = dict(scalars)
    for name, tensors in per_name.items():
        kwargs[name] = _combine_play_entries(name, tensors, mode)
    return [kwargs], mode


def _suggest_macro_name(session, node: str, param: str) -> str:
    """A free, identifier-safe macro name for ``<node>.<param>``."""
    import re
    base = re.sub(r"[^0-9a-zA-Z_]+", "_", f"{node}_{param}").strip("_") or "macro"
    if not (base[0].isalpha() or base[0] == "_"):
        base = "m_" + base
    existing = set(getattr(session, "bending_params", {}) or {})
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def _promote_binding_param(session, bended_module, play, binding_id, param_name, name=None):
    """Give a callback parameter a BendingParameter and link the two.

    Exactly what the editor's ⊕ button does, in one call, so a bendable parameter
    can become a live macro without leaving play mode. The new macro takes the
    parameter's current value and declared range, so the slider starts where the
    bending already is.
    """
    entry = next((p for p in play.promotable_params(session)
                  if p["binding"] == binding_id and p["param"] == param_name), None)
    if entry is None:
        raise KeyError(
            f"'{param_name}' is not a promotable parameter of binding "
            f"'{binding_id}' — it may already be driven by a macro")
    name = (name or "").strip() or _suggest_macro_name(session, entry["node"], param_name)
    value = entry["value"] if entry["value"] is not None else 0
    session.create_bending_param(bended_module, name, value,
                                 entry["min"], entry["max"], entry["param_type"])
    try:
        session.link_param(binding_id, param_name, name, bended_module=bended_module)
    except Exception:
        # a refused link must not leave a dangling macro behind
        try:
            session.delete_bending_param(bended_module, name)
        except Exception:
            pass
        raise
    play.import_macro(bended_module, name, session)
    return name


@csrf_exempt
def api_play_join(request):
    """POST {targets:[{binding,param},...], name?, range?} → drive them all from
    one macro.

    The gesture behind this is dragging one parameter onto another. The first
    target is the reference: it keeps the value it has, and the macro is set to
    whatever position produces it — so the thing you dragged does not jump, and
    the others come to meet it.

    A parameter already driven by a macro lends that macro to the rest instead of
    a new one being made.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = _get_bending_session()
    play = _get_play_session()
    if session is None or play is None:
        return JsonResponse({"error": "No bending session"}, status=500)
    try:
        body = json.loads(request.body) if request.body else {}
        targets = [t for t in (body.get("targets") or [])
                   if t.get("binding") and t.get("param")]
        if len(targets) < 2:
            return JsonResponse({"error": "Two parameters are needed to join"},
                                status=400)
        # One macro cannot drive two params of the same callback (the generated
        # forward names its arguments after the macro), so say that here rather
        # than half-apply the join and fail on the second link.
        seen = set()
        for t in targets:
            if t["binding"] in seen:
                b = session.bindings.get(t["binding"]) or {}
                return JsonResponse(
                    {"error": f"Both parameters belong to the same bending on "
                              f"'{b.get('node', '?')}' — one macro cannot drive a "
                              f"callback twice. Join parameters of different "
                              f"bendings instead."}, status=400)
            seen.add(t["binding"])

        name = _play_join(session, bended_module, play, targets, body.get("name"))
        _invalidate_play_session()
        _sync_save_session()
        return JsonResponse({
            "ok": True,
            "macro": name,
            "macros": play.list_macros(bended_module, session),
            "available": play.known_params(bended_module, session),
            "promotable": play.promotable_params(session),
            "bindings": session.list_bindings(),
        })
    except Exception as exc:
        return _error_json(exc, 400)


def _play_join(session, bended_module, play, targets, name=None):
    """Link every target to one macro, keeping the first one's current value."""
    entries = []
    for t in targets:
        b = session.bindings.get(t["binding"])
        if b is None:
            raise KeyError(f"Binding '{t['binding']}' not found")
        cb = b["callback"]
        pd = ((type(cb).ui_descriptor() or {}).get("params") or {}).get(t["param"]) or {}
        if pd.get("type", "float") != "float":
            raise ValueError(
                f"'{t['param']}' is a {pd.get('type')} parameter — only float "
                f"parameters can share a normalised macro")
        rng = (b.get("bp_maps") or {}).get(t["param"]) or pd.get("range") or [None, None]
        entries.append({"binding": t["binding"], "param": t["param"], "b": b,
                        "range": list(rng),
                        "value": b["params"].get(t["param"], pd.get("default", 0.0)),
                        "linked": (b.get("bp_links") or {}).get(t["param"])})

    # an existing macro on any target is the one to reuse
    name = (name or "").strip() or next((e["linked"] for e in entries if e["linked"]), None)
    ref = entries[0]
    if not name:
        name = _suggest_macro_name(session, ref["b"].get("node", ""), ref["param"])
    if name not in session.bending_params:
        lo, hi = ref["range"]
        # the reference keeps its value: the macro takes the position that yields it
        session.create_bending_param(bended_module, name, ref["value"], lo, hi, "float")
    for e in entries:
        lo, hi = e["range"]
        session.link_param(e["binding"], e["param"], name, bended_module=bended_module,
                           range=None if (lo is None or hi is None) else (lo, hi),
                           _replace_range=True)
    play.import_macro(bended_module, name, session)
    return name


@csrf_exempt
def api_play_macros(request):
    """GET → what play mode exposes, plus everything it *could* expose.
    POST → bring one in, and return the refreshed lists.

    Two kinds of thing can become a macro, and the POST body says which:

    * ``{name}`` — a BendingParameter the editor already has. Most are picked up
      on their own; this is for the ones that are not.
    * ``{binding, param, name?}`` — a callback parameter that no macro drives
      yet. It is promoted first (a BendingParameter is created and linked), then
      exposed, which is the only way such a parameter can be moved from here.

    Everything is read live, so a bending made in the editor after this runtime
    was compiled shows up without recompiling it.
    """
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    play = _get_play_session()
    if play is None:
        return JsonResponse({"error": "No play session"}, status=500)
    session = _get_bending_session()

    added, promoted = False, None
    if request.method == "POST":
        try:
            body = json.loads(request.body) if request.body else {}
            binding_id = (body.get("binding") or "").strip()
            param_name = (body.get("param") or "").strip()
            if binding_id and param_name:
                if session is None:
                    return JsonResponse({"error": "No bending session"}, status=500)
                promoted = _promote_binding_param(
                    session, bended_module, play, binding_id, param_name,
                    body.get("name"))
                added = True
                # the bending now reads its value from a BendingParameter — the
                # compiled runtime and its cached activations predate that
                _invalidate_play_session()
                _sync_save_session()
            else:
                name = (body.get("name") or "").strip()
                if not name:
                    return JsonResponse(
                        {"error": "name, or binding + param, is required"}, status=400)
                added = play.import_macro(bended_module, name, session)
        except Exception as exc:
            return _error_json(exc, 400)

    return JsonResponse({
        "ok": True,
        "added": added,
        "promoted": promoted,
        "macros": play.list_macros(bended_module, session),
        "available": play.known_params(bended_module, session),
        "promotable": play.promotable_params(session),
        "bindings": session.list_bindings() if session else [],
    })


def _parse_play_inputs(request, bended_module, fn):
    """The single-pass kwargs for this request (first set; see
    :func:`_parse_play_input_sets`)."""
    sets, _ = _parse_play_input_sets(request, bended_module, fn)
    return sets[0] if sets else {}


def _merge_play_runs(runs):
    """Fold N sequential runs into one ``[(label, tensor, slot), ...]``.

    Outputs that agree on everything but dim 0 are concatenated, so the result
    looks exactly like a batched run and every batch-aware view keeps working.
    Mismatched ones stay apart and say which run they came from — forcing them
    together would misrepresent what the model produced. ``slot`` is the output
    position, kept so each entry still maps to its graph node.
    """
    if len(runs) == 1:
        return [(label, t, i) for i, (label, t) in enumerate(runs[0])]
    merged = []
    for slot in range(max(len(r) for r in runs)):
        items = [r[slot] for r in runs if slot < len(r)]
        label = items[0][0]
        tensors = [t for _, t in items]
        if len({(t.dim(), tuple(t.shape[1:])) for t in tensors}) == 1:
            try:
                merged.append((label, torch.cat(tensors, dim=0), slot))
                continue
            except Exception:
                pass
        for i, t in enumerate(tensors):
            merged.append((f"{label} · run {i + 1}", t, slot))
    return merged


def _run_play_sets(play, sets):
    """Run every input set and return ``(merged outputs, total ms)``."""
    from .play_session import flatten_outputs
    runs, total_ms = [], 0.0
    for kwargs in sets:
        out, run_ms = play.run_to_cpu(kwargs)
        runs.append(flatten_outputs(out))
        total_ms += run_ms or 0.0
    return _merge_play_runs(runs), total_ms


@csrf_exempt
def api_play_run(request):
    """POST (FormData inputs) → run the compiled model, return serialized outputs."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    play = _get_play_session()
    if play is None or not play.compiled:
        return JsonResponse({"error": "Play session not compiled"}, status=409)
    try:
        sets, batch_mode = _parse_play_input_sets(request, bended_module, play.fn)
        if not sets or not sets[0]:
            return JsonResponse({"error": "No valid inputs provided"}, status=400)
        pairs, run_ms = _run_play_sets(play, sets)
        # map each output to its feeder node name so a view chosen for that node
        # applies in both the editor and play mode
        out_nodes = getattr(play, "_output_nodes", []) or []
        outputs = []
        for label, t, slot in pairs:
            node = out_nodes[slot] if slot < len(out_nodes) else label
            # same rate the run_audio endpoint will render this output at
            base = _view_base_node(node)
            payload = _serialize_activation(
                t, play.fn, node,
                sr_hint=_infer_output_sr(t, sets[0], bended_module=bended_module,
                                         fn=play.fn, node=base),
                declared_audio=_declares_audio(base, play.fn, bended_module))
            outputs.append({"label": label, "node": _view_base_node(node), **payload})
        return JsonResponse({
            "ok": True,
            "run_ms": round(run_ms, 2) if run_ms is not None else None,
            "mode": play.mode,
            "device": play.device,
            "batch_mode": batch_mode,
            "runs": len(sets),
            "outputs": outputs,
        })
    except Exception as exc:
        # a play run fails inside the user's forward exactly like tracing does —
        # locate it in the model's code the same way the editor does
        return _trace_error_json(exc, bended_module, play.fn)


@csrf_exempt
def api_play_run_audio(request, idx):
    """POST (FormData inputs) → run and return output #idx as a WAV file."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    play = _get_play_session()
    if play is None or not play.compiled:
        return JsonResponse({"error": "Play session not compiled"}, status=409)
    try:
        sets, _ = _parse_play_input_sets(request, bended_module, play.fn)
        if not sets or not sets[0]:
            return JsonResponse({"error": "No valid inputs provided"}, status=400)
        # idx indexes the outputs the client is looking at, which is the merged
        # list — so a sequential run's per-run cards each render their own audio
        pairs, _ = _run_play_sets(play, sets)
        i = int(idx)
        if i < 0 or i >= len(pairs):
            return JsonResponse({"error": f"output index {i} out of range"}, status=404)
        t = pairs[i][1]
        out_nodes = getattr(play, "_output_nodes", []) or []
        slot = pairs[i][2]
        node = out_nodes[slot] if slot < len(out_nodes) else pairs[i][0]
        sr = _infer_output_sr(t, sets[0], bended_module=bended_module, fn=play.fn,
                              node=_view_base_node(node))
        b, c = _audio_selection(request)
        wav, enc_sr = _tensor_to_wav(t, sr, batch=b, channel=c)
        return HttpResponse(wav, content_type="audio/wav",
                            headers={"Content-Disposition": f'inline; filename="output_{i}.wav"',
                                     "X-Sample-Rate": str(enc_sr)})
    except Exception as exc:
        return _trace_error_json(exc, bended_module, play.fn)


# ── node view API ──────────────────────────────────────────────────────────────

def _batched_shape(shape):
    """Pad a traced activation shape to the batched minimum (ndim >= 2)."""
    s = [int(x) for x in shape]
    if len(s) == 0:
        return [1, 1]
    if len(s) == 1:
        return [1] + s
    return s


def _view_meta_for(fn, node):
    """Build the picker metadata (current/default/compatible/options) for a node."""
    base = _view_base_node(node)
    bended_module = get_module()
    shape = None
    if bended_module is not None:
        try:
            shape = bended_module.activation_shape(base, fn=fn)
        except Exception:
            shape = None
    if not shape:
        return {"node": base, "fn": fn, "shape": None, "current": None,
                "default": None, "compatible": [], "options": [], "option_values": {}}
    shape = _batched_shape(shape)
    session_sel, run_cfg = _node_view_args(fn, node)
    try:
        rank_defaults = get_current_view_defaults()
    except Exception:
        rank_defaults = None
    vt, optvals = resolve_view(shape, name=base, session_sel=session_sel,
                               run_cfg=run_cfg, rank_defaults=rank_defaults)
    desc = describe_views(shape, name=base)
    return {
        "node": base, "fn": fn, "shape": shape,
        "current": vt.name if vt else None,
        "default": desc["default"],
        "compatible": desc["compatible"],
        "options": vt.options_schema() if vt else [],
        "option_values": optvals,
    }


@csrf_exempt
def api_views(request, fn, node):
    """GET → view metadata for a node.  POST {view, options} → set the live view."""
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)

    if request.method == "GET":
        return JsonResponse(_view_meta_for(fn, node))

    if request.method == "POST":
        session = _get_bending_session()
        if session is None:
            return JsonResponse({"error": "No bending session"}, status=500)
        try:
            body = json.loads(request.body) if request.body else {}
            view = body.get("view")
            if not view:
                return JsonResponse({"error": "view is required"}, status=400)
            options = body.get("options") or {}
            session.set_node_view(fn, _view_base_node(node), view, options)
            _sync_save_session()
            return JsonResponse({"ok": True, **_view_meta_for(fn, node)})
        except Exception as exc:
            return _error_json(exc, 400)

    return JsonResponse({"error": "GET or POST required"}, status=405)


@csrf_exempt
def api_client_state(request):
    """GET → return saved client UI state (pins, favs, tags, bookmarks) for the
    current model.  POST {pins, favs, tags, bookmarks} → persist it.

    The model name is taken from the ``name`` query-parameter when supplied
    (preferred — avoids race conditions during model switches) and falls back
    to ``registry.current_name`` for backward compatibility.
    """
    sm = get_sync_manager()
    if sm is None:
        if request.method == "GET":
            return JsonResponse({})
        return JsonResponse({"ok": True})   # no-op when sync is off

    registry = get_registry()
    name = request.GET.get("name") or (registry.current_name if registry else None)
    if not name:
        return JsonResponse({"error": "No model selected"}, status=400)

    if request.method == "GET":
        state = sm.load_client_state(name) or {}
        return JsonResponse(state)

    if request.method == "POST":
        try:
            state = json.loads(request.body)
            sm.save_client_state(name, state)
            return JsonResponse({"ok": True})
        except Exception as exc:
            return _error_json(exc, 400)

    return JsonResponse({"error": "GET or POST required"}, status=405)
