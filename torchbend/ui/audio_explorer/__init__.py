"""Audio similarity explorer: maps generated sounds by how they differ.

Each generation is measured against its source recording (or, without one, the
mean of its group) band by band, and the files are laid out on a 2-D map where
neighbours sound alike. Run it on a generation from the graph viewer's
generate mode -- which also launches it for you -- or on folders of files::

    python -m torchbend.ui.audio_explorer --manifest generations/.../manifest.json
    python -m torchbend.ui.audio_explorer --src-dirs originals/ --gen-dirs bended/

It is a Django app of its own, run as its own process: import this package for
``load_manifest`` and the analysis functions without starting anything.
"""
from .explorer import (available_methods, build_source_data, discover_files,  # noqa: F401
                       load_manifest, main, serve)
