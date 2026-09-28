"""Generate mode: the page, and the API that plans and runs generation jobs.

The page is play mode's (same input bench, same compile), with the macro and
output panes swapped for a sweep designer and an export panel -- see
``generate.js``. A job sets the bending values of each run on the live model,
runs it through its own :class:`~.play_session.PlaySession` and writes the
outputs; what it changed is put back when it ends. See :mod:`.generation` for
the planning itself.
"""
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from . import get_module, get_registry
from . import views as V
from .generation import (COMBINE_MODES, MAX_RUNS, Axis, GenerationJob, _field_name,
                         combine, count_runs, render_name, sample_values)


#: Jobs of this process, newest last. One runs at a time: they share the model.
_JOBS = {}
#: Explorer servers launched from here, by output directory.
_EXPLORERS = {}

_SWEEPABLE_TYPES = ("float", "int", "bool")


def generate(request):
    """Render the generate-mode page."""
    return V.play(request, page_mode="generate")


# ── what can vary ────────────────────────────────────────────────────────────

def sweepable(bended_module, session, fn):
    """Every value a generation can vary for ``fn``: the macros, and the
    callback parameters no macro drives (a driven one is varied through its
    macro). Each has a unique ``field`` -- its name in file-name templates."""
    out, taken = [], set()

    def _field(base):
        name, k = _field_name(base), 2
        while name in taken or name in ("index", "n", "fn", "model", "date", "time",
                                        "seed", "input", "output", "batch"):
            name, k = "%s_%d" % (_field_name(base), k), k + 1
        taken.add(name)
        return name

    play = V._get_play_session()
    try:
        macros = play.list_macros(bended_module, session) if play is not None else []
    except Exception:
        macros = []
    for m in macros:
        if m.get("param_type") not in _SWEEPABLE_TYPES:
            continue
        out.append({"key": "macro:" + m["name"], "kind": "macro", "name": m["name"],
                    "label": m["name"], "group_label": "macros", "type": m["param_type"],
                    "min": m.get("min"), "max": m.get("max"), "value": m.get("value"),
                    "choices": None, "field": _field(m["name"])})
    for b in (session.list_bindings() if session is not None else []):
        if b.get("fn") != fn:
            continue
        desc = (b.get("descriptor") or {}).get("params") or {}
        for pname, pd in desc.items():
            if pd.get("visible") is False or pname in (b.get("bp_links") or {}):
                continue
            kind = pd.get("type", "float")
            if kind not in _SWEEPABLE_TYPES:
                continue
            rng = list(pd.get("range") or [None, None]) + [None, None]
            where = b.get("name") or b.get("node") or b["id"]
            out.append({"key": "binding:%s.%s" % (b["id"], pname), "kind": "binding",
                        "binding": b["id"], "param": pname,
                        "label": "%s · %s" % (where, pd.get("label") or pname),
                        "group_label": "%s (%s)" % (where, b.get("callback_type", "")),
                        "type": kind, "min": rng[0], "max": rng[1],
                        "choices": pd.get("choices"),
                        "value": (b.get("params") or {}).get(pname),
                        "field": _field("%s_%s" % (where, pname))})
    return out


def _alias_resolver(bended_module, fn):
    """``alias_of(node, depth)``: the alias of ``node``, or of the nearest aliased
    node at most ``depth`` steps upstream of it."""
    try:
        aliases = bended_module.aliases(fn=fn) or {}
    except Exception:
        aliases = {}
    by_node = {}
    for alias, nodes in aliases.items():
        for node in nodes:
            by_node.setdefault(node, alias)
    try:
        graph_nodes = {n.name: n for n in bended_module.graph(fn=fn).nodes}
    except Exception:
        graph_nodes = {}

    def _alias_of(name, depth=3):
        """The nearest aliased node at most ``depth`` steps upstream: an alias
        names what ``mark()`` was given, and the output is often the mark
        itself, or a step after it (a mask, an unsqueeze)."""
        frontier, seen = [name], {name}
        for _ in range(depth + 1):
            for node in frontier:
                if node in by_node:
                    return by_node[node]
            nxt = []
            for node in frontier:
                for parent in getattr(graph_nodes.get(node), "all_input_nodes", []):
                    if parent.name not in seen and parent.op not in ("placeholder", "get_attr"):
                        seen.add(parent.name)
                        nxt.append(parent.name)
            frontier = nxt
        return None

    return _alias_of


def _describe_node(bended_module, fn, node, label):
    from .generation import guess_format
    audio = V._declares_audio(node, fn, bended_module)
    try:
        shape = list(bended_module.activation_shape(node, fn=fn) or [])
    except Exception:
        shape = []
    fmt = "wav" if audio else (guess_format(torch.empty(shape)) if shape else "pt")
    return {"node": node, "label": ("#" + label) if label else node,
            "field": _field_name(label or node), "audio": audio, "shape": shape, "format": fmt}


def outputs_of(bended_module, fn):
    """What ``fn`` returns, in order: ``[{index, node, label, audio, shape, format}]``.

    Labelled by the model's alias for the node when it has one (XTTS's
    ``#speech``), else by the node; ``format`` is the default a file of it gets.
    """
    from .play_session import PlaySession
    alias_of = _alias_resolver(bended_module, fn)
    out = []
    for i, node in enumerate(PlaySession._compute_output_nodes(bended_module, fn)):
        base = V._view_base_node(node)
        out.append(dict(_describe_node(bended_module, fn, base, alias_of(base)), index=i))
    return out


def describe_target(bended_module, fn, node):
    """An activation to export, described like an output (labelled by its own
    alias only: an arbitrary node is not what an alias upstream of it names)."""
    base = V._view_base_node(node)
    names = {n.name for n in bended_module.graph(fn=fn).nodes}
    if base not in names:
        raise ValueError("no node %r in %s" % (node, fn))
    return _describe_node(bended_module, fn, base, _alias_resolver(bended_module, fn)(base, 0))


def resolve_targets(bended_module, fn, specs):
    """``[(node to compute, field, format)]`` for the export's targets.

    A target is an output or any activation, by node; what is computed is its
    bended value (``<node>_bended``) when a bending sits on it, so the file is
    what the bending made.
    """
    if not specs:
        raise ValueError("nothing selected to export -- add an output or an activation")
    outputs = {o["node"]: o for o in outputs_of(bended_module, fn)}
    bended = {n.name for n in bended_module.bend_graph(fn=fn).nodes}
    resolved, fields = [], set()
    for spec in specs:
        node = V._view_base_node(spec.get("node") or "")
        desc = outputs.get(node) or describe_target(bended_module, fn, node)
        compute = node + "_bended" if node + "_bended" in bended else node
        field, k = desc["field"], 2
        while field in fields:
            field, k = "%s_%d" % (desc["field"], k), k + 1
        fields.add(field)
        resolved.append((compute, field, spec.get("format") or desc["format"]))
    return resolved


def _target(p):
    return ("macro", p["name"]) if p["kind"] == "macro" else ("binding", p["binding"], p["param"])


# ── planning ─────────────────────────────────────────────────────────────────

def build_plan(config, entries, params):
    """``(runs, targets, fields, axes)`` for a generation config.

    ``entries`` is ``{placeholder: [(label, values, rate), ...]}``; ``params``
    what :func:`sweepable` lists. See ``generate.js`` for the config's shape.
    """
    by_key = {p["key"]: p for p in params}
    rng = random.Random(int((config.get("combine") or {}).get("seed") or 0))
    axes, targets, fields = [], {}, {}
    for spec in config.get("params") or []:
        p = by_key.get(spec.get("key"))
        if p is None:
            raise ValueError("%s is no longer there to vary -- was its bending removed?"
                             % spec.get("key"))
        try:
            values = sample_values(spec, kind=p["type"], choices=p.get("choices"), rng=rng)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("%s: %s" % (p["label"], exc))
        axes.append(Axis(p["key"], values, group=(spec.get("group") or "").strip() or None,
                         field=p["field"], label=p["label"]))
        targets[p["key"]] = _target(p)
        fields[p["key"]] = p["field"]
    how = config.get("inputs") or "each"          # each | zip | first
    for name, items in entries.items():
        if len(items) > 1 and how != "first":
            key = "input:" + name
            axes.append(Axis(key, list(range(len(items))),
                             group="inputs" if how == "zip" else key,
                             field="input_" + _field_name(name), label=name))
            targets[key] = ("input", name)
    seed = config.get("seed") or {}
    base, repeats = seed.get("base"), max(1, int(seed.get("repeats") or 1))
    if base not in (None, "") or repeats > 1:
        base = int(base or 0)
        axes.append(Axis("seed", [base + i for i in range(repeats)], group="seed", field="seed"))
        targets["seed"] = ("seed",)
    comb = config.get("combine") or {}
    mode = comb.get("mode") or "product"
    if mode not in COMBINE_MODES:
        raise ValueError("unknown combination %r" % mode)
    runs = combine(axes, mode, comb.get("n"), seed=int(comb.get("seed") or 0))
    return runs, targets, fields, axes


def _preview(config, runs, targets, fields, entries, fn, model, limit=12, output="out"):
    export = config.get("export") or {}
    names = []
    for index, run in enumerate(runs[:limit]):
        chosen = {n: items[run.get("input:" + n, 0)][0] for n, items in entries.items()}
        varying = [n for n in entries if ("input:" + n) in run]
        f = {"index": index, "n": len(runs), "fn": fn, "model": model, "date": "YYYYMMDD",
             "time": "HHMMSS", "seed": run.get("seed", ""), "output": output, "batch": 0,
             "input": "+".join(chosen[n] for n in varying) if varying
             else next(iter(chosen.values()), "input")}
        for name, label in chosen.items():
            f["input_" + _field_name(name)] = label
        for key, value in run.items():
            if targets[key][0] in ("macro", "binding"):
                f[fields[key]] = value
        try:
            names.append(render_name(export.get("name") or "{input}_{index:04d}", f))
        except ValueError as exc:
            return [], str(exc)
    return names, None


def _read(request, fn=None):
    """The config and the bench entries a plan/start request carries."""
    bended_module = get_module()
    if bended_module is None:
        raise LookupError("No module loaded")
    try:
        config = json.loads(request.POST.get("config") or "{}")
    except Exception as exc:
        raise ValueError("bad config: %s" % exc)
    fn = config.get("fn") or fn or V._current_fn(V.get_available_methods(bended_module))
    session = V._get_bending_session()
    scalars, entries = V._collect_play_inputs(request, bended_module, fn, per_entry=True)
    return bended_module, session, config, fn, scalars, entries


def _provided(scalars, entries):
    """Every placeholder the bench gives a value -- an input mode's entry fills
    its siblings too."""
    names = set(scalars)
    for items in entries.values():
        for _label, values, _audio in items:
            names.update(values)
    return names


_UNCHECKED_HINT = " (generate mode leaves unchecked entries out)"


def _model_name():
    registry = get_registry()
    return (registry.current_name if registry else "") or "model"


# ── endpoints ────────────────────────────────────────────────────────────────

def api_generate_params(request):
    """GET ?fn= → what a generation can vary for that method."""
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    fn = request.GET.get("fn") or V._current_fn(V.get_available_methods(bended_module))
    try:
        params = sweepable(bended_module, V._get_bending_session(), fn)
    except Exception as exc:
        return V._error_json(exc)
    try:
        outputs = outputs_of(bended_module, fn)
    except Exception:
        outputs = []
    return JsonResponse({"fn": fn, "params": params, "outputs": outputs,
                         "max_runs": MAX_RUNS, "running": _running_job_id()})


def api_generate_target(request):
    """GET ?fn=&node= → an activation described as an export target."""
    bended_module = get_module()
    if bended_module is None:
        return JsonResponse({"error": "No module loaded"}, status=404)
    fn = request.GET.get("fn") or "forward"
    try:
        return JsonResponse(describe_target(bended_module, fn, request.GET.get("node") or ""))
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)


@csrf_exempt
def api_generate_plan(request):
    """POST (bench + config) → how many runs, and the first file names."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        bended_module, session, config, fn, _scalars, entries = _read(request)
        params = sweepable(bended_module, session, fn)
        runs, targets, fields, axes = build_plan(config, entries, params)
    except LookupError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    try:
        V.missing_inputs(bended_module, fn, _provided(_scalars, entries), _UNCHECKED_HINT)
        missing = None
    except ValueError as exc:
        missing = str(exc)
    try:
        first = resolve_targets(bended_module, fn, (config.get("export") or {}).get("targets"))[0][1]
    except Exception:
        first = "out"
    names, name_error = _preview(config, runs, targets, fields, entries, fn, _model_name(),
                                 output=first)
    return JsonResponse({
        "missing": missing,
        "runs": len(runs),
        "axes": [{"key": a.key, "field": a.field, "label": a.label, "group": a.group,
                  "n": len(a.values), "values": [str(v) for v in a.values[:12]]} for a in axes],
        "inputs": {n: [label for label, _, _ in items] for n, items in entries.items()},
        "names": names, "name_error": name_error,
        "fields": sorted({"index", "n", "fn", "model", "date", "time", "seed", "input",
                          "output", "batch"} | set(fields.values())
                         | {"input_" + _field_name(n) for n in entries}),
    })


def _running_job_id():
    return next((j.id for j in _JOBS.values() if j.state == "running"), None)


@csrf_exempt
def api_generate_start(request):
    """POST (bench + config) → start a generation job in the background."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    running = _running_job_id()
    if running:
        return JsonResponse({"error": "a generation is already running", "id": running},
                            status=409)
    try:
        bended_module, session, config, fn, scalars, entries = _read(request)
        params = sweepable(bended_module, session, fn)
        runs, targets, fields, _axes = build_plan(config, entries, params)
        V.missing_inputs(bended_module, fn, _provided(scalars, entries), _UNCHECKED_HINT)
    except LookupError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    if not runs:
        return JsonResponse({"error": "nothing to generate"}, status=400)

    from .play_session import PlaySession
    runner = PlaySession()
    try:
        runner.compile(bended_module, fn, session.device if session else "cpu", session=session)
        hooks = _hooks(bended_module, session, runner, fn, targets, params)
    except Exception as exc:
        return V._trace_error_json(exc, bended_module, fn)
    export = dict(config.get("export") or {})
    try:
        resolved = resolve_targets(bended_module, fn, export.get("targets"))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    # the runner computes the targets and nothing else; files are named after
    # each one's alias ({output}), in the format asked for it
    runner._output_nodes = [node for node, _field, _fmt in resolved]
    export["outputs"] = None
    export["output_labels"] = {str(i): field for i, (_n, field, _f) in enumerate(resolved)}
    export["formats"] = {str(i): fmt for i, (_n, _field, fmt) in enumerate(resolved)}
    export.setdefault("dir", str(Path("generations") / _model_name() / fn
                                  / time.strftime("%Y%m%d-%H%M%S")))
    by_key = {p["key"]: p for p in params}
    job = GenerationJob(runs, targets, entries, scalars, export, hooks, meta={
        "fn": fn, "model": _model_name(), "config": config, "fields": fields,
        "parameters": {fields[k]: {"label": by_key[k]["label"], "type": by_key[k]["type"],
                                   "key": k} for k in fields},
    })
    _JOBS[job.id] = job
    job.start()
    return JsonResponse({"ok": True, **job.status()})


def _hooks(bended_module, session, runner, fn, targets, params):
    """How a job sets values, runs, and puts everything back -- the parts of a
    generation only the viewer knows."""
    from .bending_session import _read_param, param_python_value
    from .play_session import flatten_outputs

    macros = runner.all_macros(bended_module, session)
    original = {}
    for key, target in targets.items():
        if target[0] == "macro":
            original[key] = param_python_value(macros[target[1]])
        elif target[0] == "binding":
            b = session.bindings[target[1]]
            original[key] = _read_param(b["callback"], target[2])

    def set_value(target, value):
        if target[0] == "macro":
            runner.set_macro(bended_module, target[1], value, session=session)
            return
        _, bid, pname = target
        session.update_param(bended_module, bid, pname, value)
        b = session.bindings[bid]
        runner.mark_dirty(b.get("nodes", [b["node"]]))
        cb = b["callback"]
        if getattr(cb, "weight_compatible", False) and len(getattr(cb, "_bending_targets", []) or []):
            # the runner holds its own copy of the weights: rebuild it
            runner._needs_rebuild = True

    def reseed(seed):
        if seed is None:
            return
        torch.manual_seed(int(seed))
        # cached activations were drawn under another seed
        if runner._cache is not None:
            runner._cache.clear()

    def run(kwargs):
        out, _ms = runner.run_to_cpu(kwargs)
        nodes = runner._output_nodes or []
        result = []
        for i, (label, tensor) in enumerate(flatten_outputs(out)):
            node = V._view_base_node(nodes[i]) if i < len(nodes) else label
            result.append((label, tensor, node, V._declares_audio(node, fn, bended_module)))
        return result

    def rate(tensor, kwargs, node):
        return V._infer_output_sr(tensor, kwargs, bended_module=bended_module, fn=fn, node=node)

    def restore():
        for key, value in original.items():
            try:
                set_value(targets[key], value)
            except Exception:
                pass
        runner.release()
        # whatever the editor and play mode cached was computed under values
        # this job moved
        V._invalidate_play_session()
        try:
            session.graph_changed(fn)
        except Exception:
            pass

    return {"set": set_value, "reseed": reseed, "run": run, "rate": rate,
            "wav": V._tensor_to_wav, "restore": restore}


def _job(request):
    jid = request.GET.get("id") or request.POST.get("id")
    if not jid:
        try:
            jid = json.loads(request.body or b"{}").get("id")
        except Exception:
            jid = None
    if jid:
        return _JOBS.get(jid)
    return next(reversed(list(_JOBS.values())), None)


def api_generate_status(request):
    """GET ?id= → a job's progress (the latest job without one)."""
    job = _job(request)
    if job is None:
        return JsonResponse({"state": "none"})
    return JsonResponse(job.status())


@csrf_exempt
def api_generate_cancel(request):
    """POST {id} → stop a job after the run in progress."""
    job = _job(request)
    if job is None:
        return JsonResponse({"error": "no such job"}, status=404)
    job.cancel()
    return JsonResponse({"ok": True, **job.status()})


# ── the audio explorer, on another port ─────────────────────────────────────

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_open(port):
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


@csrf_exempt
def api_generate_explore(request):
    """POST {dir} → start the audio explorer on a generation's manifest.

    It runs as its own process on its own port: computing the similarity map
    takes a while and needs nothing from this server, and the explorer is a
    Django app of its own.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        body = json.loads(request.body or b"{}")
    except Exception:
        body = {}
    folder = Path(os.path.expanduser(body.get("dir") or "")).resolve()
    manifest = folder / "manifest.json"
    if not manifest.is_file():
        return JsonResponse({"error": "no manifest.json in %s" % folder}, status=404)
    known = _EXPLORERS.get(str(folder))
    if known and known["proc"].poll() is None and not body.get("restart"):
        return JsonResponse({"port": known["port"], "pid": known["proc"].pid, "reused": True})
    if known and known["proc"].poll() is None:
        known["proc"].terminate()
    port = _free_port()
    log = open(folder / "explorer.log", "w")
    # the child starts in the output folder: point it at this torchbend, which
    # need not be installed (a checkout on PYTHONPATH, say)
    import torchbend
    env = dict(os.environ)
    root = str(Path(torchbend.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join([root] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "torchbend.ui.audio_explorer", "--manifest", str(manifest),
         "--port", str(port)] + (["--recompute"] if body.get("recompute") else []),
        stdout=log, stderr=subprocess.STDOUT, cwd=str(folder), env=env)
    _EXPLORERS[str(folder)] = {"proc": proc, "port": port, "log": folder / "explorer.log"}
    return JsonResponse({"port": port, "pid": proc.pid, "reused": False})


def api_generate_explore_status(request):
    """GET ?dir= → whether that explorer is up, still computing, or has died."""
    folder = str(Path(os.path.expanduser(request.GET.get("dir") or "")).resolve())
    known = _EXPLORERS.get(folder)
    if known is None:
        return JsonResponse({"state": "none"})
    code = known["proc"].poll()
    tail = ""
    try:
        tail = Path(known["log"]).read_text()[-2000:]
    except Exception:
        pass
    if code is not None:
        return JsonResponse({"state": "exited", "code": code, "log": tail})
    return JsonResponse({"state": "ready" if _port_open(known["port"]) else "starting",
                         "port": known["port"], "log": tail})


_WAIT_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Audio explorer</title>
<style>body{font:14px system-ui,sans-serif;margin:40px;color:#333;background:#fafafa}
pre{background:#fff;border:1px solid #ddd;padding:10px;max-height:60vh;overflow:auto;
font-size:11px;white-space:pre-wrap}</style></head><body>
<h3>Audio explorer</h3><p id="s">Computing the similarity map for this generation…</p>
<pre id="log"></pre>
<script>
const dir = new URLSearchParams(location.search).get("dir");
async function poll() {
  const r = await fetch("/api/generate/explore/status/?dir=" + encodeURIComponent(dir));
  const d = await r.json();
  document.getElementById("log").textContent = d.log || "";
  if (d.state === "ready") { location.replace(location.protocol + "//" + location.hostname + ":" + d.port + "/"); return; }
  if (d.state === "exited" || d.state === "none") {
    document.getElementById("s").textContent = d.state === "none"
      ? "No explorer was started for this folder." : "The explorer stopped (exit code " + d.code + ").";
    return;
  }
  setTimeout(poll, 1000);
}
poll();
</script></body></html>"""


def generate_explore_wait(request):
    """A tab that waits for the explorer to come up, then goes to it."""
    return HttpResponse(_WAIT_PAGE)
