"""Device discovery and per-method compatibility, shared by play mode and the
graph editor -- both let you move the active model to an accelerator, and both
need to know which devices exist and which of the model's methods the active
:class:`~torchbend.interfaces.base.Interface` expects to work on each one
(``Method(devices=...)``, defaulting to "yes" when it says nothing).
"""
import torch


# ── device discovery ────────────────────────────────────────────────────────

def available_devices() -> list:
    """Torch devices usable from this process (always includes ``'cpu'``)."""
    devices = ["cpu"]
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
    except Exception:
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    return devices


def _device_label(dev: str) -> str:
    if dev == "cpu":
        return "CPU"
    if dev == "mps":
        return "MPS (Apple)"
    if dev.startswith("cuda"):
        try:
            idx = int(dev.split(":")[1]) if ":" in dev else 0
            return f"{torch.cuda.get_device_name(idx)} ({dev})"
        except Exception:
            return f"CUDA ({dev})"
    return dev


def device_options() -> list:
    return [{"value": d, "label": _device_label(d)} for d in available_devices()]


# ── per-method compatibility ────────────────────────────────────────────────

def device_compat_for_methods(spec, methods) -> dict:
    """``{method: {device: bool}}`` for every device and every method, from the
    active interface's declarations (``Method(devices=...)``), or all-True when
    there is none -- a bare module has no opinion, and neither does a method
    its interface does not declare."""
    devices = available_devices()
    out = {}
    for fn in methods:
        out[fn] = spec.method(fn).devices(devices) if spec is not None \
            else {d: True for d in devices}
    return out
