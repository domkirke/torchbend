"""
Sync manager: persist bending session, node views, and default inputs to disk
so they survive server restarts.  One subfolder per model name.

Layout::

    sync_dir/
        ModelName/
            fingerprint.json   # class_name + graph hash + weights hash
            session.json       # bindings, bending_params, node_views, mode
            inputs.json        # serialised default_inputs tensors/scalars
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


# ── serialisation helpers ─────────────────────────────────────────────────────

def _safe_dir_name(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


def _serialize_inputs(inputs: dict) -> dict:
    """Convert {name: tensor|scalar} to a JSON-safe dict."""
    import torch
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = {
                "__type__": "tensor",
                "dtype": str(v.dtype).replace("torch.", ""),
                "shape": list(v.shape),
                "data": v.tolist(),
            }
        elif isinstance(v, (int, float, str, bool, type(None))):
            out[k] = {"__type__": "scalar", "value": v}
        else:
            try:
                out[k] = {"__type__": "json", "value": json.dumps(v)}
            except Exception:
                pass
    return out


def _deserialize_inputs(raw: dict) -> dict:
    import torch
    out = {}
    for k, v in raw.items():
        t = v.get("__type__")
        if t == "tensor":
            dtype = getattr(torch, v.get("dtype", "float32"), torch.float32)
            out[k] = torch.tensor(v["data"], dtype=dtype)
        elif t == "scalar":
            out[k] = v["value"]
        elif t == "json":
            try:
                out[k] = json.loads(v["value"])
            except Exception:
                pass
    return out


def _node_views_to_json(node_views: dict) -> dict:
    """Convert {(fn, node): spec} to a JSON-serialisable flat dict."""
    return {f"{fn}:{node}": spec for (fn, node), spec in node_views.items()}


def _json_to_node_views(data: dict) -> dict:
    out = {}
    for key, spec in data.items():
        parts = key.split(":", 1)
        if len(parts) == 2:
            out[(parts[0], parts[1])] = spec
    return out


def _graph_hash(graph: dict) -> str:
    nodes = sorted(
        (n.get("id", ""), n.get("op", ""), str(n.get("target", "")))
        for n in graph.get("nodes", [])
    )
    edges = sorted(
        (e.get("source", ""), e.get("target", ""))
        for e in graph.get("edges", [])
    )
    sig = json.dumps({"nodes": nodes, "edges": edges}, sort_keys=True)
    return hashlib.md5(sig.encode()).hexdigest()


def _weights_hash(bm) -> str:
    try:
        total = sum(float(p.detach().cpu().sum()) for p in bm._module.parameters())
        return f"{total:.6f}"
    except Exception:
        return ""


# ── SyncManager ───────────────────────────────────────────────────────────────

class SyncManager:
    """Manages per-model state persistence in a sync directory."""

    def __init__(self, sync_dir: str | Path):
        self.sync_dir = Path(sync_dir)
        self.sync_dir.mkdir(parents=True, exist_ok=True)

    def erase(self) -> None:
        """Delete all saved state in the sync directory and recreate it empty."""
        import shutil
        if self.sync_dir.exists():
            shutil.rmtree(self.sync_dir)
        self.sync_dir.mkdir(parents=True, exist_ok=True)

    # ── I/O helpers ───────────────────────────────────────────────────────────

    def model_dir(self, name: str) -> Path:
        d = self.sync_dir / _safe_dir_name(name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def has_saved_state(self, name: str) -> bool:
        return (self.sync_dir / _safe_dir_name(name)).exists()

    def _write(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=2))

    def _read(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    # ── fingerprint ───────────────────────────────────────────────────────────

    def compute_fingerprint(self, bm, graph: dict | None = None) -> dict:
        if graph is None:
            from . import serializer
            graph = serializer.serialize_graph(bm)
        return {
            "class_name": type(bm._module).__name__,
            "graph_hash": _graph_hash(graph),
            "weights_hash": _weights_hash(bm),
        }

    def save_fingerprint(self, name: str, bm, graph: dict | None = None) -> None:
        fp = self.compute_fingerprint(bm, graph)
        self._write(self.model_dir(name) / "fingerprint.json", fp)

    def load_fingerprint(self, name: str) -> dict | None:
        return self._read(self.model_dir(name) / "fingerprint.json")

    def check_fingerprint(self, name: str, bm, graph: dict | None = None) -> str:
        """Return 'new', 'same', or 'changed'."""
        saved = self.load_fingerprint(name)
        if saved is None:
            return "new"
        current = self.compute_fingerprint(bm, graph)
        if (current["class_name"] == saved["class_name"]
                and current["graph_hash"] == saved["graph_hash"]
                and current["weights_hash"] == saved["weights_hash"]):
            return "same"
        return "changed"

    # ── bending session ───────────────────────────────────────────────────────

    def save_session(self, name: str, session) -> None:
        try:
            data = {
                "bending": session.session_to_json(),
                "node_views": _node_views_to_json(session.node_views),
                "update_mode": session.update_mode,
                "auto_threshold_ms": session.auto_threshold_ms,
            }
            self._write(self.model_dir(name) / "session.json", data)
        except Exception as exc:
            print(f"[sync] Warning: could not save session for '{name}': {exc}")

    def load_session(self, name: str) -> dict | None:
        return self._read(self.model_dir(name) / "session.json")

    def clear_session(self, name: str) -> None:
        p = self.model_dir(name) / "session.json"
        if p.exists():
            p.unlink()

    # ── inputs ────────────────────────────────────────────────────────────────

    def save_inputs(self, name: str, inputs: dict) -> None:
        try:
            self._write(self.model_dir(name) / "inputs.json", _serialize_inputs(inputs))
        except Exception as exc:
            print(f"[sync] Warning: could not save inputs for '{name}': {exc}")

    def load_inputs(self, name: str) -> dict | None:
        raw = self._read(self.model_dir(name) / "inputs.json")
        return _deserialize_inputs(raw) if raw is not None else None

    def save_inputs_raw(self, name: str, raw: dict) -> None:
        self._write(self.model_dir(name) / "inputs.json", raw)

    def load_inputs_raw(self, name: str) -> dict | None:
        return self._read(self.model_dir(name) / "inputs.json")

    # ── client-side UI state (pins, favs, tags, bookmarks) ───────────────────

    def save_client_state(self, name: str, state: dict) -> None:
        try:
            self._write(self.model_dir(name) / "client_state.json", state)
        except Exception as exc:
            print(f"[sync] Warning: could not save client state for '{name}': {exc}")

    def load_client_state(self, name: str) -> dict | None:
        return self._read(self.model_dir(name) / "client_state.json")


# ── CLI prompts ───────────────────────────────────────────────────────────────

def _prompt(question: str, choices: list) -> str:
    choices_str = "/".join(choices)
    while True:
        try:
            ans = input(f"  {question} [{choices_str}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = choices[0]
            print(f"\n  (defaulting to '{ans}')")
        if ans in choices:
            return ans
        print(f"  Please enter one of: {choices_str}")


# ── startup logic ─────────────────────────────────────────────────────────────

def startup_sync_check(registry, sync_manager: SyncManager) -> None:
    """
    Run before the server starts (terminal is still interactive).
    For each loaded model: check graph/weights fingerprint and default_inputs.
    Lazy models get only an inputs check; graph check is deferred to first load.
    """
    from . import serializer as _ser

    for name, entry in registry._entries.items():
        if not entry.is_loaded:
            _check_lazy_entry(name, entry, sync_manager)
            continue

        bm = entry.module
        graph = None
        try:
            graph = _ser.serialize_graph(bm)
        except Exception as exc:
            print(f"[sync] Warning: could not serialize graph for '{name}': {exc}")

        if not sync_manager.has_saved_state(name):
            print(f"[sync] New model '{name}' — creating sync folder.")
            if graph is not None:
                sync_manager.save_fingerprint(name, bm, graph)
            sync_manager.save_inputs(name, entry.default_inputs)
            continue

        # ── graph/weights check ──────────────────────────────────────────────
        restore_session = True
        if graph is not None:
            status = sync_manager.check_fingerprint(name, bm, graph)
            if status == "changed":
                saved_fp = sync_manager.load_fingerprint(name)
                current_fp = sync_manager.compute_fingerprint(bm, graph)
                print(f"\n[sync] Model '{name}' has changed since last sync.")
                if saved_fp:
                    print(f"  saved  : class={saved_fp.get('class_name')}  "
                          f"graph={saved_fp.get('graph_hash', '')[:8]}  "
                          f"weights={saved_fp.get('weights_hash')}")
                print(f"  current: class={current_fp['class_name']}  "
                      f"graph={current_fp['graph_hash'][:8]}  "
                      f"weights={current_fp['weights_hash']}")
                print("  [r] replace — overwrite sync data (saved bendings/settings will be lost)")
                print("  [d] discard — keep sync data (bendings may not apply to new model)")
                ans = _prompt("Choice", ["r", "d"])
                if ans == "r":
                    sync_manager.save_fingerprint(name, bm, graph)
                    sync_manager.clear_session(name)
                    sync_manager.save_inputs(name, entry.default_inputs)
                    continue
                # discard: keep saved data; skip session restore to stay safe
                restore_session = False

        # ── inputs check ────────────────────────────────────────────────────
        _check_inputs(name, entry, sync_manager)

        # ── queue session restore ────────────────────────────────────────────
        if restore_session:
            saved_session = sync_manager.load_session(name)
            if saved_session is not None:
                entry._pending_sync_session = saved_session


def _check_lazy_entry(name: str, entry, sync_manager: SyncManager) -> None:
    """Startup inputs check for a model that isn't loaded yet."""
    if not sync_manager.has_saved_state(name):
        print(f"[sync] New lazy model '{name}' — will register fingerprint on first load.")
        if entry.default_inputs:
            sync_manager.save_inputs(name, entry.default_inputs)
        return
    _check_inputs(name, entry, sync_manager)
    saved_session = sync_manager.load_session(name)
    if saved_session is not None:
        entry._pending_sync_session = saved_session


def _check_inputs(name: str, entry, sync_manager: SyncManager) -> None:
    """Compare saved inputs with entry.default_inputs; prompt if keys differ."""
    new_inputs = entry.default_inputs or {}
    raw_saved = sync_manager.load_inputs_raw(name)

    if raw_saved is None:
        # Nothing saved yet — just persist whatever was given
        if new_inputs:
            sync_manager.save_inputs(name, new_inputs)
        return

    saved_keys = set(raw_saved.keys())
    new_keys = set(new_inputs.keys())

    if not new_inputs:
        # No default_inputs provided — silently restore saved ones
        entry.default_inputs = _deserialize_inputs(raw_saved)
        return

    if saved_keys == new_keys:
        # Same keys — prefer saved (user may have tweaked them via UI)
        entry.default_inputs = _deserialize_inputs(raw_saved)
        return

    added = new_keys - saved_keys
    removed = saved_keys - new_keys
    print(f"\n[sync] default_inputs differ for model '{name}'.")
    if added:
        print(f"  New keys:     {sorted(added)}")
    if removed:
        print(f"  Removed keys: {sorted(removed)}")
    print("  [a] add     — merge: add new keys to saved, keep existing values")
    print("  [r] replace — overwrite saved inputs with new default_inputs")
    print("  [b] bypass  — ignore new default_inputs, use saved ones as-is")
    ans = _prompt("Choice", ["a", "r", "b"])

    if ans == "r":
        sync_manager.save_inputs(name, new_inputs)
    elif ans == "a":
        merged = dict(raw_saved)
        for k, v in _serialize_inputs(new_inputs).items():
            if k not in merged:
                merged[k] = v
        sync_manager.save_inputs_raw(name, merged)
        entry.default_inputs = _deserialize_inputs(merged)
    else:  # bypass
        entry.default_inputs = _deserialize_inputs(raw_saved)


# ── post-load check for lazy models ──────────────────────────────────────────

def post_load_sync_check(name: str, entry, sync_manager: SyncManager) -> None:
    """
    Called after a lazy model is first loaded (inside registry.select()).
    Cannot prompt interactively (server already running), so warns to console.
    """
    if not entry.is_loaded:
        return
    bm = entry.module
    graph = None
    try:
        from . import serializer as _ser
        graph = _ser.serialize_graph(bm)
    except Exception:
        pass

    if not sync_manager.has_saved_state(name):
        print(f"[sync] New lazy model '{name}' — creating sync folder.")
        if graph is not None:
            sync_manager.save_fingerprint(name, bm, graph)
        sync_manager.save_inputs(name, entry.default_inputs)
        return

    if graph is not None:
        status = sync_manager.check_fingerprint(name, bm, graph)
        if status == "changed":
            print(f"[sync] WARNING: '{name}' graph/weights differ from saved sync data.")
            print(f"[sync]   Saved bendings will NOT be restored. Restart to get a prompt.")
            entry._pending_sync_session = None
            return
        if status == "new":
            sync_manager.save_fingerprint(name, bm, graph)
            return
    # Same fingerprint (or no graph available) — pending_sync_session is already set
    # by _check_lazy_entry at startup; nothing extra needed here.
