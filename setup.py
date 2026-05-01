"""
Reads base deps from requirements.txt and discovers optional extras from
torchbend/interfaces/requirements/<name>.txt.

Supported directives (as comments at the top of an interface file):
  # requires-python: >=3.10        warns at install time + adds env markers to all deps
  # submodule: path/to/submodule   warns if not initialized; use `torchbend-init <extra>` to fix
  # [overrides]                    section: lines below replace matching base deps

Example interfaces/requirements/stylegan.txt:

    # requires-python: >=3.9
    # submodule: torchbend/interfaces/stylegan/stylegan3-video
    some-package>=1.0

    # [overrides]
    numpy>=1.23.3,<2.0

Install with:  pip install torchbend[stylegan]
Init submodules: torchbend-init stylegan
"""
import re
import subprocess
import sys
import warnings
from pathlib import Path
from setuptools import setup

_ROOT = Path(__file__).parent
_BASE_REQS = _ROOT / "requirements.txt"
_REQUIREMENTS_DIR = _ROOT / "torchbend" / "interfaces" / "requirements"

_STRIP_COMMENT = re.compile(r"\s*#.*$", re.MULTILINE)
_SECTION_MARKER = re.compile(r"#\s*\[overrides\]", re.IGNORECASE)
_REQUIRES_PYTHON = re.compile(r"#\s*requires-python\s*:\s*([^\n]+)", re.IGNORECASE)
_SUBMODULE = re.compile(r"#\s*submodule\s*:\s*(\S+)", re.IGNORECASE)
_PKG_NAME = re.compile(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)", re.ASCII)
_OP_MAP = {">=": ">=", "<=": "<=", ">": ">", "<": "<", "==": "==", "!=": "!=", "~=": ">="}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pkg_name(dep: str) -> str:
    m = _PKG_NAME.match(dep.strip())
    return m.group(1).lower().replace("-", "_") if m else ""


def _parse_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# interface file parser
# ---------------------------------------------------------------------------

def _parse_interface_file(path: Path) -> dict:
    """Parse an interface requirements file and return all metadata."""
    raw = path.read_text()

    py_match = _REQUIRES_PYTHON.search(raw)
    submodule_paths = [m.group(1) for m in _SUBMODULE.finditer(raw)]

    parts = _SECTION_MARKER.split(raw, maxsplit=1)
    extra_text = _STRIP_COMMENT.sub("", parts[0])
    override_text = _STRIP_COMMENT.sub("", parts[1]) if len(parts) > 1 else ""

    return {
        "extra_deps": _parse_lines(extra_text),
        "overrides": _parse_lines(override_text),
        "python_specifier": py_match.group(1).strip() if py_match else None,
        "submodules": submodule_paths,
    }


# ---------------------------------------------------------------------------
# python version checks
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
# submodule checks
# ---------------------------------------------------------------------------

def _is_submodule_initialized(rel_path: str) -> bool:
    abs_path = _ROOT / rel_path
    if not abs_path.exists():
        return False
    # an uninitialised submodule is an empty directory
    return any(abs_path.iterdir())


def _check_submodules(name: str, submodule_paths: list[str]) -> None:
    missing = [p for p in submodule_paths if not _is_submodule_initialized(p)]
    if missing:
        paths_str = "\n    ".join(missing)
        warnings.warn(
            f"\n[torchbend] Extra '{name}' requires the following git submodule(s) "
            f"which are not yet initialized:\n    {paths_str}\n"
            f"Run:  torchbend-init {name}\n"
            f"  or: git submodule update --init " + " ".join(missing),
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# dependency builders
# ---------------------------------------------------------------------------

def _load_base_deps() -> list[str]:
    if not _BASE_REQS.exists():
        return []
    text = _STRIP_COMMENT.sub("", _BASE_REQS.read_text())
    return _parse_lines(text)


def _apply_overrides(base: list[str], overrides: list[str]) -> list[str]:
    override_names = {_pkg_name(dep) for dep in overrides}
    return [dep for dep in base if _pkg_name(dep) not in override_names] + overrides


def _discover_extras() -> dict[str, list[str]]:
    extras: dict[str, list[str]] = {}
    if not _REQUIREMENTS_DIR.is_dir():
        return extras
    for req_file in sorted(_REQUIREMENTS_DIR.glob("*.txt")):
        meta = _parse_interface_file(req_file)
        name = req_file.stem
        deps = meta["extra_deps"]

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


def _build_base_deps() -> list[str]:
    base = _load_base_deps()
    if _REQUIREMENTS_DIR.is_dir():
        for req_file in sorted(_REQUIREMENTS_DIR.glob("*.txt")):
            meta = _parse_interface_file(req_file)
            if meta["overrides"]:
                base = _apply_overrides(base, meta["overrides"])
    return base


# ---------------------------------------------------------------------------
# torchbend-init entry point
# ---------------------------------------------------------------------------

def _init_submodules(args: list[str] | None = None) -> None:
    """CLI: torchbend-init [extra ...]  — initializes required git submodules."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="torchbend-init",
        description="Initialize git submodules required by torchbend extras.",
    )
    parser.add_argument(
        "extras", nargs="*",
        help="Extra name(s) to initialize (e.g. stylegan). Omit for all.",
    )
    parsed = parser.parse_args(args)

    submodule_map: dict[str, list[str]] = {}
    if _REQUIREMENTS_DIR.is_dir():
        for req_file in sorted(_REQUIREMENTS_DIR.glob("*.txt")):
            meta = _parse_interface_file(req_file)
            if meta["submodules"]:
                submodule_map[req_file.stem] = meta["submodules"]

    targets = parsed.extras if parsed.extras else list(submodule_map)
    for extra in targets:
        if extra not in submodule_map:
            print(f"[torchbend-init] '{extra}' has no submodule requirements.")
            continue
        for path in submodule_map[extra]:
            if _is_submodule_initialized(path):
                print(f"[torchbend-init] {path}: already initialized.")
                continue
            print(f"[torchbend-init] Initializing submodule: {path}")
            result = subprocess.run(
                ["git", "submodule", "update", "--init", path],
                cwd=_ROOT,
            )
            if result.returncode != 0:
                print(f"[torchbend-init] ERROR: failed to initialize {path}", file=sys.stderr)


setup(
    install_requires=_build_base_deps(),
    extras_require=_discover_extras(),
    entry_points={
        "console_scripts": [
            "torchbend-init = setup:_init_submodules",
        ],
    },
)
