"""Generate mode: offline batch generation over combinations of bending values.

Play mode is for performing with one set of values; generate mode renders many.
Each thing that varies is an *axis* -- a macro, a callback parameter, the input
bench's entries, a seed -- and a *plan* combines the axes into runs:

    axes = [Axis("macro:gain",  sample_values({"mode": "range", "min": 0, "max": 1, "n": 5})),
            Axis("binding:b1.scale", [0.5, 2.0], group="g"),
            Axis("binding:b2.scale", [1.0, 4.0], group="g")]     # moves with b1.scale
    plan = combine(axes, mode="product")                         # 5 × 2 = 10 runs

Axes that share a ``group`` move in lockstep (zipped); groups are combined with
each other by the plan's mode: ``product`` (every combination), ``zip`` (all
groups in lockstep) or ``random`` (``n`` distinct combinations drawn from the
product). Everything here is plain Python and has no knowledge of Django; the
job runner at the bottom drives a :class:`~.play_session.PlaySession`.
"""

import datetime
import itertools
import json
import math
import os
import random
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

import torch


__all__ = ["Axis", "sample_values", "combine", "render_name", "GenerationJob",
           "MAX_RUNS", "COMBINE_MODES", "SAMPLE_MODES"]

#: A plan larger than this is refused rather than started: it is almost always a
#: range with too many points, and it would fill the disk before anyone noticed.
MAX_RUNS = 20000

SAMPLE_MODES = ("fixed", "values", "range", "random")
COMBINE_MODES = ("product", "zip", "random")


# ── axes ─────────────────────────────────────────────────────────────────────

class Axis:
    """One thing that varies across the runs, and the values it takes.

    ``key`` identifies it (``"macro:<name>"``, ``"binding:<id>.<param>"``,
    ``"input:<placeholder>"``, ``"seed"``); ``field`` is the name it goes by in a
    file-name template; ``group`` zips it with the other axes of that group.
    """

    def __init__(self, key, values, group=None, field=None, label=None):
        self.key = key
        self.values = list(values)
        self.group = group or key
        self.field = field or _field_name(key)
        self.label = label or key
        if not self.values:
            raise ValueError("axis %s takes no values" % key)

    def __repr__(self):
        return "Axis(%r, %d values, group=%r)" % (self.key, len(self.values), self.group)


def _field_name(key):
    """``"binding:b1.scale"`` → ``"b1_scale"``: usable in a format string."""
    name = key.split(":", 1)[-1]
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_") or "value"
    return name if not name[0].isdigit() else "_" + name


def _cast(value, kind, choices=None):
    if kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if kind == "int":
        return int(round(float(value)))
    if choices:
        # a choice is snapped to the nearest one allowed, as the editor does
        return min(choices, key=lambda c: abs(float(c) - float(value)))
    return float(value)


def sample_values(spec, kind="float", choices=None, rng=None):
    """The values one parameter takes, from its sampling spec.

    ``fixed``  -- ``{"value": v}``: one value.
    ``values`` -- ``{"values": [...]}``: exactly these.
    ``range``  -- ``{"min", "max", "n", "scale": "lin"|"log"}``: ``n`` evenly
                  spaced points, ends included.
    ``random`` -- ``{"min", "max", "n"}``: ``n`` uniform draws (``rng`` makes
                  them reproducible).

    Integers are rounded and deduplicated; booleans range over both values;
    a parameter with ``choices`` takes the allowed values in ``[min, max]``.
    """
    mode = spec.get("mode", "fixed")
    if mode not in SAMPLE_MODES:
        raise ValueError("unknown sampling mode %r (expected one of %s)"
                         % (mode, ", ".join(SAMPLE_MODES)))
    rng = rng or random.Random(0)
    if mode == "fixed":
        return [_cast(spec["value"], kind, choices)]
    if mode == "values":
        raw = spec.get("values")
        if isinstance(raw, str):
            raw = [v for v in re.split(r"[,\s]+", raw) if v]
        if not raw:
            raise ValueError("no values given")
        return [_cast(v, kind, choices) for v in raw]
    n = spec.get("n")
    n = 1 if n in (None, "") else int(float(n))
    if n < 1:
        raise ValueError("n must be at least 1")
    if kind == "bool":
        pool = [False, True]
        return pool if mode == "range" else [rng.choice(pool) for _ in range(n)]
    lo, hi = float(spec["min"]), float(spec["max"])
    if choices:
        pool = [c for c in choices if min(lo, hi) <= float(c) <= max(lo, hi)] or list(choices)
        if mode == "range":
            return pool
        return [rng.choice(pool) for _ in range(n)]
    if mode == "range":
        if n == 1:
            points = [lo]
        elif spec.get("scale") == "log":
            if lo <= 0 or hi <= 0:
                raise ValueError("a log range needs min and max above 0")
            points = [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (n - 1))
                      for i in range(n)]
        else:
            points = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    else:
        points = ([rng.randint(int(math.ceil(min(lo, hi))), int(math.floor(max(lo, hi))))
                   for _ in range(n)] if kind == "int"
                  else [rng.uniform(lo, hi) for _ in range(n)])
    # 12 significant digits: a log range's 10.000000000000002 is 10
    out = [_cast(float("%.12g" % v), kind) for v in points]
    if kind == "int" and mode == "range":
        out = list(dict.fromkeys(out))           # rounding may repeat a value
    return out


# ── combining axes into runs ─────────────────────────────────────────────────

def _groups(axes):
    """``[[axis, ...], ...]``: axes sharing a group, in first-seen order."""
    order, by_group = [], {}
    for axis in axes:
        if axis.group not in by_group:
            by_group[axis.group] = []
            order.append(axis.group)
        by_group[axis.group].append(axis)
    return [by_group[g] for g in order]


def _group_rows(group):
    """The rows of one group: its axes zipped, the shorter ones cycled."""
    length = max(len(a.values) for a in group)
    return [{a.key: a.values[i % len(a.values)] for a in group} for i in range(length)]


def count_runs(axes, mode="product", n=None):
    rows = [len(_group_rows(g)) for g in _groups(axes)]
    if not rows:
        return 1
    if mode == "zip":
        return max(rows)
    total = math.prod(rows)
    if mode == "random":
        return min(total, int(n or total))
    return total


def combine(axes, mode="product", n=None, seed=0):
    """The runs a plan makes: a list of ``{axis key: value}``.

    ``product`` takes every combination of the groups; ``zip`` runs the groups
    in lockstep, cycling the shorter ones; ``random`` draws ``n`` distinct
    combinations from the product (all of them when ``n`` covers it), in
    product order so neighbouring files stay related.
    """
    if mode not in COMBINE_MODES:
        raise ValueError("unknown combination mode %r (expected one of %s)"
                         % (mode, ", ".join(COMBINE_MODES)))
    groups = [_group_rows(g) for g in _groups(axes)]
    if not groups:
        return [{}]
    total = count_runs(axes, mode, n)
    if total > MAX_RUNS:
        raise ValueError("this plan makes %d runs, above the limit of %d -- use fewer "
                         "points, or a random combination" % (total, MAX_RUNS))
    if mode == "zip":
        length = max(len(g) for g in groups)
        return [_merge(g[i % len(g)] for g in groups) for i in range(length)]
    if mode == "random":
        sizes = [len(g) for g in groups]
        full = math.prod(sizes)
        picks = sorted(random.Random(seed).sample(range(full), total))
        out = []
        for flat in picks:
            rows = []
            for size, g in zip(reversed(sizes), reversed(groups)):
                flat, i = divmod(flat, size)
                rows.append(g[i])
            out.append(_merge(reversed(rows)))
        return out
    return [_merge(rows) for rows in itertools.product(*groups)]


def _merge(rows):
    out = {}
    for row in rows:
        out.update(row)
    return out


# ── naming ───────────────────────────────────────────────────────────────────

class _Fields(dict):
    """Format fields: a missing one names itself, so a typo shows in the file
    name rather than aborting the whole job."""

    def __missing__(self, key):
        return "{%s}" % key


class _Num(float):
    """A number in a file name: short by default, any format spec on request
    (``{gain}`` → ``0.3``, ``{gain:.3f}`` → ``0.300``)."""

    def __format__(self, spec):
        return float.__format__(self, spec) if spec else _format_value(float(self))


def _format_value(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        text = ("%.4g" % value)
        return text.replace("+", "")
    return str(value)


def _safe(text):
    """One path component: no separators, nothing a filesystem dislikes."""
    text = re.sub(r"[\\/:*?\"<>|\s]+", "-", str(text)).strip("-.")
    return text or "_"


def render_name(template, fields):
    """Fill ``template`` with ``fields``; ``/`` in the template makes folders.

    Values are sanitised so a caption with a slash in it cannot escape the
    output directory; the template itself decides the structure.
    """
    def _field(v):
        if isinstance(v, bool):
            return _format_value(v)
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return _Num(v)
        return _safe(_format_value(v))
    safe = _Fields({k: _field(v) for k, v in fields.items()})
    parts = []
    for part in str(template).replace("\\", "/").split("/"):
        if part in ("", ".", ".."):
            continue
        try:
            text = part.format_map(safe)
        except (ValueError, IndexError, KeyError) as exc:
            raise ValueError("file name template %r: %s" % (template, exc))
        parts.append(_safe(text))
    if not parts:
        raise ValueError("file name template %r names nothing" % template)
    return "/".join(parts)


# ── writing outputs ──────────────────────────────────────────────────────────

def guess_format(tensor, declared_audio=False):
    """``"wav"``, ``"png"`` or ``"pt"`` for one output, when asked for "auto"."""
    if declared_audio:
        return "wav"
    shape = tuple(tensor.shape)
    # an image: [B, C, H, W] or [C, H, W] with colour channels. A single channel
    # 3-D tensor is too often a sequence ([1, frames, features]) to call it one
    if len(shape) == 4 and shape[1] in (1, 3, 4) and min(shape[2:]) >= 16:
        return "png"
    if len(shape) == 3 and shape[0] in (3, 4) and min(shape[1:]) >= 16 and shape[-1] < 4096:
        return "png"
    if 1 <= len(shape) <= 3 and shape[-1] >= 2048:
        return "wav"
    return "pt"


def tensor_to_png(tensor):
    """An image tensor ``[C, H, W]`` or ``[H, W]`` (min-max normalised) as PNG bytes."""
    import io
    import numpy as np
    from PIL import Image
    t = tensor.detach().float().cpu()
    while t.ndim > 3:
        t = t[0]
    lo, hi = float(t.min()), float(t.max())
    t = (t - lo) / ((hi - lo) or 1.0)
    if t.ndim == 3:
        if t.shape[0] not in (1, 3, 4):
            t = t[0]
        else:
            t = t.permute(1, 2, 0)
            if t.shape[-1] == 1:
                t = t[..., 0]
    arr = (t.numpy() * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ── the job ──────────────────────────────────────────────────────────────────

class GenerationJob:
    """One generation, run in a background thread.

    ``targets`` maps an axis key to what setting it means: ``("macro", name)``,
    ``("binding", binding_id, param)``, ``("input", placeholder)`` or
    ``("seed",)``. ``inputs`` holds each placeholder's entries as
    ``[(label, {placeholder: tensor}, recording), ...]`` -- a dict, since an
    input mode's entry fills several (a prompt, and its mask); ``recording``
    is the entry's ``(waveform, rate)`` when it came in as audio, else None;
    ``scalars`` the fixed scalar arguments. ``hooks`` supplies what only the viewer knows: how to
    set a value, run the model, find an output's rate, write audio.
    """

    def __init__(self, runs, targets, inputs, scalars, export, hooks, meta=None):
        self.id = uuid.uuid4().hex[:10]
        self.runs = runs
        self.targets = targets
        self.inputs = inputs
        self.scalars = scalars
        self.export = export
        self.hooks = hooks
        self.meta = dict(meta or {})
        self.state = "pending"            # pending | running | done | cancelled | error
        self.done = 0
        self.files = []                   # manifest rows, in order
        self.errors = []
        self.message = ""
        self.started = None
        self.finished = None
        self._cancel = threading.Event()
        self._thread = None

    # -- control --

    def start(self):
        self._thread = threading.Thread(target=self._run, name="tb-generate-%s" % self.id,
                                        daemon=True)
        self.state = "running"
        self.started = time.time()
        self._thread.start()
        return self

    def cancel(self):
        self._cancel.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def status(self, last=5):
        elapsed = (self.finished or time.time()) - (self.started or time.time())
        eta = (elapsed / self.done * (len(self.runs) - self.done)) if self.done else None
        return {
            "id": self.id, "state": self.state, "done": self.done, "total": len(self.runs),
            "dir": str(self.out_dir), "files": len(self.files),
            "recent": [f["path"] for f in self.files[-last:]],
            "errors": self.errors[-last:], "n_errors": len(self.errors),
            "message": self.message, "elapsed": round(elapsed, 1),
            "eta": round(eta, 1) if eta is not None else None,
        }

    @property
    def out_dir(self):
        return Path(os.path.expanduser(self.export.get("dir") or "generations")).resolve()

    # -- running --

    def _run(self):
        # `state` stays "running" until the values are restored and the manifest
        # is final: "done" is what lets the next job start, and a caller read
        # the manifest -- neither may happen while this job still holds the model
        hooks = self.hooks
        final = "done"
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.sources = self._write_sources()
            previous = {}
            used = set()
            for index, run in enumerate(self.runs):
                if self._cancel.is_set():
                    final = "cancelled"
                    break
                try:
                    kwargs = dict(self.scalars)
                    chosen = {}                      # placeholder -> entry label
                    for key, value in run.items():
                        target = self.targets[key]
                        if target[0] == "input":
                            label, values, _audio = self.inputs[target[1]][value]
                            kwargs.update(values)
                            chosen[target[1]] = label
                        elif target[0] != "seed" and previous.get(key, object()) != value:
                            # only what changed is set, so the activation cache
                            # resumes from whatever the change leaves clean
                            hooks["set"](target, value)
                    for name, entries in self.inputs.items():
                        if name not in chosen:
                            kwargs.update(entries[0][1])
                            chosen[name] = entries[0][0]
                    # a seed makes a run reproducible on its own, which means
                    # drawing everything afresh: reseeded before every run
                    hooks["reseed"](run.get("seed"))
                    previous = run
                    outputs = hooks["run"](kwargs)
                    self._write(index, run, chosen, kwargs, outputs, used)
                except Exception as exc:
                    self.errors.append({"index": index, "error": "%s: %s" % (type(exc).__name__, exc),
                                        "trace": traceback.format_exc(limit=6)})
                    if self.export.get("stop_on_error", True):
                        final = "error"
                        self.message = self.errors[-1]["error"]
                        break
                self.done = index + 1
                if self.done % 10 == 0:
                    self._write_manifest()
        except Exception as exc:
            final = "error"
            self.message = "%s: %s" % (type(exc).__name__, exc)
            self.errors.append({"index": None, "error": self.message,
                                "trace": traceback.format_exc(limit=6)})
        finally:
            try:
                hooks["restore"]()
            except Exception as exc:
                self.errors.append({"index": None, "error": "restoring values: %s" % exc})
            self.finished = time.time()
            try:
                self._write_manifest(final)
            except Exception as exc:
                self.errors.append({"index": None, "error": "manifest: %s" % exc})
            self.state = final

    def _fields(self, index, run, chosen):
        now = datetime.datetime.now()
        fields = {"index": index, "n": len(self.runs), "fn": self.meta.get("fn", ""),
                  "model": self.meta.get("model", ""), "date": now.strftime("%Y%m%d"),
                  "time": now.strftime("%H%M%S"), "seed": run.get("seed", "")}
        varying = [n for n in self.inputs if ("input:" + n) in run]
        fields["input"] = "+".join(chosen[n] for n in varying) if varying else \
            "+".join(chosen[n] for n in list(chosen)[:1]) or "input"
        for name, label in chosen.items():
            fields["input_" + _field_name(name)] = label
        for key, value in run.items():
            target = self.targets[key]
            if target[0] in ("macro", "binding"):
                fields[self.meta["fields"][key]] = value
        return fields

    def _write(self, index, run, chosen, kwargs, outputs, used):
        """Write the selected outputs of one run; record each in the manifest."""
        export = self.export
        wanted = export.get("outputs")            # indices into the flat outputs
        fields = self._fields(index, run, chosen)
        params = {self.meta["fields"][k]: v for k, v in run.items()
                  if self.targets[k][0] in ("macro", "binding")}
        multi_out = len(outputs) > 1 and (wanted is None or len(wanted) > 1)
        formats = export.get("formats") or {}
        names = export.get("output_labels") or {}
        for slot, (label, tensor, node, declared) in enumerate(outputs):
            if wanted is not None and slot not in wanted:
                continue
            label = names.get(str(slot)) or label
            batch = tensor.shape[0] if tensor.ndim >= 2 and export.get("split_batch", True) else 1
            for b in range(batch):
                item = tensor[b:b + 1] if batch > 1 else tensor
                fmt = formats.get(str(slot)) or export.get("format", "auto")
                if fmt == "auto":
                    fmt = guess_format(item, declared)
                f = dict(fields, output=_safe(label), batch=b)
                name = render_name(export.get("name") or "{input}_{index:04d}", f)
                if multi_out and "{output" not in (export.get("name") or ""):
                    name += "_" + _safe(label)
                if batch > 1 and "{batch" not in (export.get("name") or ""):
                    name += "_b%d" % b
                rel = self._unique(name, fmt, used)
                path = self.out_dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                sr = None
                if fmt == "wav":
                    sr = self.hooks["rate"](item, kwargs, node)
                    data, sr = self.hooks["wav"](item, sr)
                    path.write_bytes(data)
                elif fmt == "png":
                    path.write_bytes(tensor_to_png(item))
                else:
                    torch.save(item.detach().cpu(), str(path))
                if export.get("tensors") and fmt != "pt":
                    torch.save(item.detach().cpu(), str(path.with_suffix(".pt")))
                self.files.append({
                    "path": rel, "index": index, "output": label, "batch": b,
                    "source": self._pool(chosen), "type": render_name(
                        export.get("type") or "{fn}", f) if export.get("type") != "" else "",
                    "params": params, "seed": run.get("seed"), "format": fmt,
                    "sample_rate": sr, "inputs": dict(chosen),
                })

    @staticmethod
    def _unique(name, fmt, used):
        rel = "%s.%s" % (name, fmt)
        k = 1
        while rel in used:
            rel = "%s_%d.%s" % (name, k, fmt)
            k += 1
        used.add(rel)
        return rel

    #: The pool of every generation when nothing groups them.
    ALL = "all"

    def _audio_inputs(self):
        """The inputs that came in as audio -- recordings, not tensors."""
        return [n for n, entries in self.inputs.items() if any(e[2] for e in entries)]

    def _pool(self, chosen):
        """Which generations the audio explorer compares with each other.

        By default all of them, in one space. With ``input_as_original``, the
        ones made from the same audio input, measured against that recording;
        with no audio input, that falls back to one space too.
        """
        if self.export.get("input_as_original"):
            audio = self._audio_inputs()
            if audio:
                return "+".join(chosen[n] for n in audio)
        return self.ALL

    def _write_sources(self):
        """With ``input_as_original``, each audio input as the reference of its
        pool: what the audio explorer measures its generations against.

        Written only where a pool has a single recording behind it -- with two
        audio inputs a pool is a pair, and the explorer uses its mean instead.
        """
        out = {}
        audio = self._audio_inputs()
        if not self.export.get("input_as_original") or len(audio) != 1:
            return out
        folder = self.out_dir / "sources"
        for label, _values, recording in self.inputs[audio[0]]:
            if not recording:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            data, _ = self.hooks["wav"](*recording)
            rel = "sources/%s.wav" % _safe(label)
            (self.out_dir / rel).write_bytes(data)
            out[label] = rel
        return out

    def _write_manifest(self, state=None):
        manifest = {
            "torchbend_generation": 1,
            "model": self.meta.get("model", ""), "fn": self.meta.get("fn", ""),
            "created": datetime.datetime.fromtimestamp(self.started or time.time()).isoformat(),
            "state": state or self.state, "total": len(self.runs),
            "config": self.meta.get("config", {}),
            "parameters": self.meta.get("parameters", {}),
            "sources": getattr(self, "sources", {}),
            "files": self.files,
            "errors": [{k: v for k, v in e.items() if k != "trace"} for e in self.errors],
        }
        path = self.out_dir / "manifest.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=1, default=str))
        tmp.replace(path)
