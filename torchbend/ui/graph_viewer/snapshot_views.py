"""Activation snapshots: saved copies of an activation, by name.

A view in the graph editor (the activation panel, the expand modal, a dashboard
card) can save what it shows, and later be recalled to a snapshot -- frozen, it
no longer follows the graph -- or let go of it to follow the graph again. The
tensors are kept whole, on the server, in the model's session: they are what
activation interpolation will work from, not only something to look at.

    GET    /api/snapshots/?fn=&node=   → {snapshots: [...], default_name}
    POST   /api/snapshots/             bench inputs + fn, node, name → save
    GET    /api/snapshots/<name>/      → the snapshot, serialised like an activation
    DELETE /api/snapshots/<name>/                  (also takes it off every node)
    POST   /api/snapshots/<name>/recall/  {fn, node}  → into the graph at node
    POST   /api/snapshots/release/        {fn, node}  → that node follows the graph again

A snapshot is *recalled* into the graph, not only into a view: it becomes a
:class:`~torchbend.bending.Snapshot` bending on the node, so the node takes the
saved value and everything computed after it follows. Its ``mix`` parameter
crossfades back to the live value, and can be driven by a macro.
"""
import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from . import get_module, get_registry, get_sync_manager
from . import views as V


def _save_to_sync(session):
    sm = get_sync_manager()
    registry = get_registry()
    if sm is not None and registry is not None:
        sm.save_snapshots(registry.current_name, session.snapshots)


@csrf_exempt
def api_snapshots(request):
    """GET → the snapshots (of one node with ?node=); POST → save one."""
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = V._get_bending_session()
    if session is None:
        return JsonResponse({"error": "No bending session"}, status=500)

    if request.method == "GET":
        fn, node = request.GET.get("fn") or None, request.GET.get("node") or None
        node = V._view_base_node(node) if node else None
        out = {"snapshots": session.list_snapshots(fn=fn, node=node)}
        if node:
            out["default_name"] = session.default_snapshot_name(node)
        return JsonResponse(out)

    if request.method != "POST":
        return JsonResponse({"error": "GET or POST required"}, status=405)
    fn = request.POST.get("fn") or "forward"
    node = V._view_base_node(request.POST.get("node") or "")
    if not node:
        return JsonResponse({"error": "which node?"}, status=400)
    name = (request.POST.get("name") or "").strip() or session.default_snapshot_name(node)
    overwrite = request.POST.get("overwrite", "").lower() in ("1", "true", "yes")
    if name in session.snapshots and not overwrite:
        return JsonResponse({"error": "there is already a snapshot named %r" % name,
                             "exists": True}, status=409)
    try:
        kwargs = V._parse_inputs(request, bended_module, fn)
        if not kwargs:
            return JsonResponse({"error": "No valid inputs provided"}, status=400)
        # what the view shows: the bended value when a bending sits on the node
        captured = V._run_capture_activations(bended_module, fn, kwargs, session,
                                              target_nodes=[node, node + "_bended"])
        tensor = captured.get(node + "_bended")
        if tensor is None:
            tensor = captured.get(node)
        if tensor is None:
            return JsonResponse({"error": "no activation for %r" % node}, status=404)
        rate = V._infer_output_sr(tensor, kwargs, bended_module=bended_module, fn=fn, node=node)
        meta = session.save_snapshot(name, fn, node, tensor, sample_rate=rate, overwrite=overwrite)
    except Exception as exc:
        return V._trace_error_json(exc, bended_module, fn)
    _save_to_sync(session)
    return JsonResponse({"ok": True, "snapshot": meta})


@csrf_exempt
def api_snapshot_detail(request, name):
    """GET → one snapshot, rendered like a live activation; DELETE → drop it."""
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    session = V._get_bending_session()
    if session is None or name not in session.snapshots:
        return JsonResponse({"error": "no snapshot named %r" % name}, status=404)

    if request.method == "DELETE":
        released = session.release_snapshot_everywhere(bended_module, name)
        session.delete_snapshot(name)
        _save_to_sync(session)
        if released:
            _bindings_changed()
        return JsonResponse({"ok": True, "snapshots": session.list_snapshots(),
                             "released": released, **_binding_state(session)})
    if request.method != "GET":
        return JsonResponse({"error": "GET or DELETE required"}, status=405)

    snap = session.snapshots[name]
    fn, node = snap["fn"], snap["node"]
    try:
        payload = V._serialize_activation(
            snap["tensor"], fn, node, sr_hint=snap.get("sample_rate"),
            declared_audio=V._declares_audio(node, fn, bended_module))
    except Exception as exc:
        return V._error_json(exc)
    payload["snapshot"] = session.snapshot_meta(name)
    return JsonResponse(payload)


def _binding_state(session):
    """What the editor redraws its bendings from, as the binding endpoints return it."""
    return {"bindings": session.list_bindings(),
            "bending_params": session.list_bending_params()}


def _bindings_changed():
    # the bending topology changed: play mode's compiled runtime is stale, and
    # the session (bindings, links) is saved like any other bending change
    V._invalidate_play_session()
    V._sync_save_session()


def _body(request):
    try:
        return json.loads(request.body or b"{}")
    except Exception:
        return {}


@csrf_exempt
def api_snapshot_recall(request, name):
    """POST {fn, node} → recall snapshot ``name`` into the graph at ``node``."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    session = V._get_bending_session()
    if bended_module is None or session is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    body = _body(request)
    snap = session.snapshots.get(name)
    if snap is None:
        return JsonResponse({"error": "no snapshot named %r" % name}, status=404)
    fn = body.get("fn") or snap["fn"]
    node = V._view_base_node(body.get("node") or snap["node"])
    try:
        bid = session.recall_snapshot(bended_module, name, fn, node)
    except Exception as exc:
        return V._error_json(exc, 400)
    _bindings_changed()
    return JsonResponse({"ok": True, "binding": bid, **_binding_state(session)})


@csrf_exempt
def api_snapshot_release(request):
    """POST {fn, node} → take the recalled snapshot off ``node``."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    bended_module = get_module()
    session = V._get_bending_session()
    if bended_module is None or session is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    body = _body(request)
    fn, node = body.get("fn") or "forward", V._view_base_node(body.get("node") or "")
    released = session.release_snapshot(bended_module, fn, node)
    if released:
        _bindings_changed()
    return JsonResponse({"ok": True, "released": released, **_binding_state(session)})
