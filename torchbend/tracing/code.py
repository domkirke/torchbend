"""Source-position helpers: where in *the user's model* something happened.

Tracing runs the model through several layers of torch and torchbend machinery,
so both the activation metadata and the error reporting need the same thing —
the innermost frame that belongs to neither.
"""
import linecache
import os
import re
import traceback


#: Files that are tracing machinery rather than model code. A frame in one of
#: these is never the answer to "where did this happen?".
INTERNAL_FILES = (
    'torch/fx/proxy.py',
    'torch/fx/_symbolic_trace.py',
    'torch/fx/experimental/proxy_tensor.py',
    'torch/_ops.py',
    'torch/_tensor.py',
    'torch/utils/_python_dispatch.py',
    'torch/_prims_common/wrappers.py',
    'torch/_refs/__init__.py',
    'torch/_refs/nn/functional/__init__.py',
    'torch/utils/_stats.py',
    'torchbend/tracing/tracing.py',
    'torchbend/tracing/tracing_experimental.py',
    'torchbend/tracing/proxy.py',
    'torchbend/tracing/module.py',
    'torchbend/tracing/mark.py',
)


#: Frames that carry no information in a stack shown to the user: torch's own
#: ``Module.__call__`` dispatch. Kept apart from INTERNAL_FILES, which also
#: drives the tracer's activation source positions.
_DISPATCH_FILES = (
    'torch/nn/modules/module.py',
    # the graph viewer's request handler, which is what calls trace() from the UI
    'torchbend/ui/graph_viewer/views.py',
    # and the runners it calls through — play mode's compiled runtime and the
    # demand-driven activation cache both sit between the request and the model,
    # so a failure inside the model must not be blamed on them
    'torchbend/ui/graph_viewer/play_session.py',
    'torchbend/ui/graph_viewer/activation_cache.py',
    'torchbend/ui/graph_viewer/bending_session.py',
)

#: Where torchbend writes the wrapper modules it generates and execs
#: (``utils._import_defs_from_tmpfile``). Real files, but not the user's code.
_GENERATED_DIR = 'torchbend/jit'


def is_internal_file(filename: str) -> bool:
    """True when *filename* is tracing machinery rather than user code."""
    if not filename:
        return False
    norm = filename.replace('\\', '/')
    return any(norm.endswith(p) for p in INTERNAL_FILES)


def _is_library(filename: str) -> bool:
    """True for installed-library code (torch, site-packages) — correct to blame,
    but never the line the user can act on."""
    norm = (filename or "").replace('\\', '/')
    if '/site-packages/' in norm or '/dist-packages/' in norm:
        return True
    try:
        import torch
        torch_dir = os.path.dirname(os.path.abspath(torch.__file__)).replace('\\', '/')
        return norm.startswith(torch_dir + '/')
    except Exception:
        return False


def _is_noise(filename: str) -> bool:
    """Internal, torch dispatch, or code torchbend/fx generated on the fly."""
    if is_internal_file(filename):
        return True
    norm = (filename or "").replace('\\', '/')
    if any(norm.endswith(p) for p in _DISPATCH_FILES):
        return True
    if _GENERATED_DIR in norm:
        return True
    # fx also emits modules under synthetic filenames with nothing on disk
    return not (filename and os.path.exists(filename))


def source_context(filename: str, lineno: int, radius: int = 4) -> list:
    """``[(lineno, text, is_target), ...]`` around *lineno*, for display."""
    if not filename or not lineno:
        return []
    lines = []
    for n in range(max(1, lineno - radius), lineno + radius + 1):
        text = linecache.getline(filename, n)
        if not text and n > lineno:
            break
        if text:
            lines.append((n, text.rstrip('\n'), n == lineno))
    return lines


#: fx names its generated modules ``<eval_with_key>.N from <file>:<line> in <fn>``
#: — no file on disk, but the name itself points back at the model's source.
_GENERATED_FRAME_PREFIX = '<eval_with_key>'
_GENERATED_ORIGIN_RE = re.compile(
    r'^<eval_with_key>[^ ]*\s+from\s+(?P<file>.+):(?P<line>\d+)\s+in\s+(?P<function>\S+)$')


def _describe_generated_frame(fs) -> dict:
    """What a frame of fx-generated code tells us.

    Running a traced model never enters the model's own ``forward`` — fx replaced
    it with generated code — so the stack alone cannot say where the failure was.
    The generated frame can: its source line names the graph node that failed,
    and its (synthetic) filename records the model source the graph was built
    from. Both are worth more to the reader than the frame itself.
    """
    code = (fs.line or "").strip()
    # "l : torch.Tensor = self.l(x);  x = None"  →  node "l"
    node = None
    if "=" in code:
        lhs = code.split("=", 1)[0].split(":", 1)[0].strip()
        if lhs.isidentifier():
            node = lhs
    origin = None
    m = _GENERATED_ORIGIN_RE.match(fs.filename or "")
    if m is not None:
        path = m.group("file")
        if os.path.exists(path) and not _is_noise(path) and not _is_library(path):
            origin = {
                "file":     path,
                "basename": os.path.basename(path),
                "line":     int(m.group("line")),
                "function": m.group("function"),
            }
    return {"node": node, "code": code, "line": fs.lineno, "origin": origin}


def describe_exception(exc, radius: int = 4) -> dict:
    """Locate *exc* in the user's code.

    Returns the exception type/message, the full traceback, and the innermost
    frame that is not tracing machinery — with the surrounding source, so the
    caller can point at the actual line instead of only echoing the message.
    """
    tb = exc.__traceback__
    frames = traceback.extract_tb(tb) if tb is not None else []
    # The frame that actually raised is often inside torch (F.linear complaining
    # about shapes); correct, but the line the user can *act* on is the innermost
    # one in their own code. Report that as the location and keep the raising
    # frame alongside it.
    raised = next((fs for fs in reversed(frames) if not _is_noise(fs.filename)), None)
    chosen = next((fs for fs in reversed(frames)
                   if not _is_noise(fs.filename) and not _is_library(fs.filename)), None)
    # The failure may have happened while *running* a traced graph rather than
    # while tracing: then the model's own frames are not on the stack at all, fx's
    # generated code stands in for them, and without this the panel would blame
    # whatever library function torch raised from.
    generated = next((fs for fs in reversed(frames)
                      if (fs.filename or "").startswith(_GENERATED_FRAME_PREFIX)), None)
    graph_frame = _describe_generated_frame(generated) if generated is not None else None
    if chosen is None and graph_frame is not None and graph_frame["origin"]:
        o = graph_frame["origin"]
        chosen = traceback.FrameSummary(o["file"], o["line"], o["function"])
    if chosen is None:
        chosen = raised
    if chosen is None:
        chosen = next((fs for fs in reversed(frames)
                       if not is_internal_file(fs.filename)), None)
    if chosen is None and frames:
        chosen = frames[-1]

    location = None
    if chosen is not None:
        location = {
            "file":     chosen.filename,
            "basename": os.path.basename(chosen.filename),
            "line":     chosen.lineno,
            "function": chosen.name,
            "code":     (chosen.line or "").strip(),
            "context":  source_context(chosen.filename, chosen.lineno or 0, radius),
            "internal": is_internal_file(chosen.filename),
        }
    raised_in = None
    if raised is not None and raised is not chosen:
        raised_in = {
            "file":     raised.filename,
            "basename": os.path.basename(raised.filename),
            "line":     raised.lineno,
            "function": raised.name,
            "code":     (raised.line or "").strip(),
        }

    return {
        "error":      str(exc),
        "error_type": type(exc).__name__,
        "traceback":  "".join(traceback.format_exception(type(exc), exc, tb)),
        "location":   location,
        # where it was actually raised, when that is library code rather than the
        # model's own — so nothing is hidden by pointing at the actionable line
        "raised_in":  raised_in,
        # the graph node whose generated code failed, when the model was being
        # run rather than traced
        "graph_frame": graph_frame,
        # the model call chain, outermost first — for a model built of nested
        # submodules this is the path down to the failing op, with torch's
        # __call__ dispatch and fx's generated wrappers left out
        "user_frames": [
            {"file": fs.filename, "basename": os.path.basename(fs.filename),
             "line": fs.lineno, "function": fs.name, "code": (fs.line or "").strip()}
            for fs in frames if not _is_noise(fs.filename)
        ],
    }


class CodePosition():
    def __init__(self, frame, tracer=None, node=None, parameter=None):
        self.frame = frame
        self.source_file = None
        self.source_line = None
        self.source_fn = None

    @classmethod
    def from_source(cls, source_file, source_line, source_fn=None):
        obj = object.__new__(cls)
        obj.frame = None
        obj.source_file = source_file
        obj.source_line = source_line
        obj.source_fn = source_fn
        return obj

    @property
    def code(self):
        return self.frame.f_code

    @property
    def description(self):
        if self.source_file is not None:
            import os
            return f"{os.path.basename(self.source_file)}:{self.source_line}"
        code = self.code
        desc = f"{code.co_filename}:"
        desc += f"{code.co_name}"
        desc += f".{code.co_firstlineno})"
        return desc

    def __repr__(self):
        return self.description


def get_code_pos_from_frame(frame):
    return CodePosition(frame).description
