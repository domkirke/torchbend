"""
Auto-discovers optional extras from torchbend/interfaces/requirements/<name>.txt.
Base dependencies are read from requirements.txt via pyproject.toml.

Supported directives (as comments in an interface file):
  # requires-python: >=3.10        warns at install time + adds env markers to all deps
  # submodule: path/to/submodule   warns if not initialized; use `torchbend-init <extra>` to fix
  # [overrides]                    section: version constraints that refine base deps for this extra

Example interfaces/requirements/stylegan.txt:

    # requires-python: >=3.9
    # submodule: torchbend/interfaces/stylegan/stylegan3-video
    some-package>=1.0

    # [overrides]
    numpy>=1.23.3,<2.0   # stricter version range, pip resolves against base

Install with:  pip install torchbend[stylegan]
Install all:   pip install torchbend[all]
"""
import re
import sys
import warnings
from pathlib import Path
from setuptools import setup

_ROOT = Path(__file__).parent
_REQUIREMENTS_DIR = _ROOT / "torchbend" / "interfaces" / "requirements"

_STRIP_COMMENT = re.compile(r"\s*#.*$", re.MULTILINE)
_SECTION_MARKER = re.compile(r"#\s*\[overrides\]", re.IGNORECASE)
_REQUIRES_PYTHON = re.compile(r"#\s*requires-python\s*:\s*([^\n]+)", re.IGNORECASE)
_SUBMODULE = re.compile(r"#\s*submodule\s*:\s*(\S+)", re.IGNORECASE)
_OP_MAP = {">=": ">=", "<=": "<=", ">": ">", "<": "<", "==": "==", "!=": "!=", "~=": ">="}


def _parse_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _parse_interface_file(path: Path) -> dict:
    raw = path.read_text()

    py_match = _REQUIRES_PYTHON.search(raw)
    submodule_paths = [m.group(1) for m in _SUBMODULE.finditer(raw)]

    parts = _SECTION_MARKER.split(raw, maxsplit=1)
    extra_text = _STRIP_COMMENT.sub("", parts[0])
    override_text = _STRIP_COMMENT.sub("", parts[1]) if len(parts) > 1 else ""

    return {
        # regular extra deps (added on top of base)
        "extra_deps": _parse_lines(extra_text),
        # version refinements merged into the extra (pip resolves against base)
        "overrides": _parse_lines(override_text),
        "python_specifier": py_match.group(1).strip() if py_match else None,
        "submodules": submodule_paths,
    }


# ---------------------------------------------------------------------------
# python version check
# ---------------------------------------------------------------------------

def _specifier_to_markers(specifier: str) -> str:
    parts = []
    for clause in specifier.split(","):
        m = re.match(r"([><=!~]+)\s*([\d.]+)", clause.strip())
        if m:
            op = _OP_MAP.get(m.group(1), m.group(1))
            short = ".".join(m.group(2).split(".")[:2])
            parts.append(f'python_version {op} "{short}"')
    return " and ".join(parts) if parts else ""


def _check_python_specifier(name: str, specifier: str) -> None:
    try:
        from packaging.specifiers import SpecifierSet
        current_ver = "%d.%d" % sys.version_info[:2]
        if current_ver not in SpecifierSet(specifier):
            warnings.warn(
                f"\n[torchbend] Extra '{name}' requires Python {specifier}, "
                f"but you are running Python {sys.version.split()[0]}. "
                "Some packages may not install correctly.",
                stacklevel=3,
            )
    except ImportError:
        import operator as op_mod
        mapping = {">=": op_mod.ge, "<=": op_mod.le, ">": op_mod.gt,
                   "<": op_mod.lt, "==": op_mod.eq, "!=": op_mod.ne}
        current = sys.version_info[:2]
        for clause in specifier.split(","):
            m = re.match(r"([><=!~]+)\s*([\d.]+)", clause.strip())
            if m:
                cmp_op = mapping.get(m.group(1))
                target = tuple(int(x) for x in m.group(2).split(".")[:2])
                if cmp_op and not cmp_op(current, target):
                    warnings.warn(
                        f"\n[torchbend] Extra '{name}' requires Python {specifier}, "
                        f"but you are running Python {sys.version.split()[0]}. "
                        "Some packages may not install correctly.",
                        stacklevel=3,
                    )
                    break


def _apply_marker(deps: list[str], marker: str) -> list[str]:
    return [dep if ";" in dep else f"{dep}; {marker}" for dep in deps]


# ---------------------------------------------------------------------------
# submodule check
# ---------------------------------------------------------------------------

def _is_submodule_initialized(rel_path: str) -> bool:
    abs_path = _ROOT / rel_path
    return abs_path.exists() and any(abs_path.iterdir())


def _check_submodules(name: str, submodule_paths: list[str]) -> None:
    missing = [p for p in submodule_paths if not _is_submodule_initialized(p)]
    if missing:
        paths_str = "\n    ".join(missing)
        warnings.warn(
            f"\n[torchbend] Extra '{name}' requires uninitialized git submodule(s):\n"
            f"    {paths_str}\n"
            f"Run:  git submodule update --init " + " ".join(missing),
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# extras discovery
# ---------------------------------------------------------------------------

def _discover_extras() -> dict[str, list[str]]:
    extras: dict[str, list[str]] = {}
    if not _REQUIREMENTS_DIR.is_dir():
        return extras

    for req_file in sorted(_REQUIREMENTS_DIR.glob("*.txt")):
        meta = _parse_interface_file(req_file)
        name = req_file.stem
        deps = meta["extra_deps"] + meta["overrides"]

        if meta["python_specifier"]:
            _check_python_specifier(name, meta["python_specifier"])
            marker = _specifier_to_markers(meta["python_specifier"])
            if marker:
                deps = _apply_marker(deps, marker)

        if meta["submodules"]:
            _check_submodules(name, meta["submodules"])

        extras[name] = deps

    if extras:
        extras["all"] = sorted({dep for deps in extras.values() for dep in deps})
    return extras


setup(extras_require=_discover_extras())
