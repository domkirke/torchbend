#!/usr/bin/env python3
"""
Audio similarity explorer — local Django server.

Part of torchbend as ``torchbend.ui.audio_explorer``: ``python -m
torchbend.ui.audio_explorer --manifest <generation>/manifest.json`` maps a
generation made in the graph viewer's generate mode, each file coloured by the
bending values it was made with (see ``load_manifest``).

Pipeline per source file:
  1. Load original; find energetic frames (RMS > threshold).
  2. For each bended file: compute per-mel-band mean log-spectral distance
     from original → distance vector (N_BANDS,).
  3. Metric MDS on the distance matrix → MDS_DIMS coords.
  4. Manifold reduction → 2-D embedding.
     Methods: tsne, som, phate, mst, pca, umap, isomap, diffmap,
              ica, sammon, graph, geodesic

Usage:
    python audio_similarity_explorer.py [--src-dirs ...] [--gen-dirs ...]
                                         [--methods tsne,som,phate,mst,pca]
                                         [--sources stem1 stem2 ...]
                                         [--port 8000]
                                         [--duration 10] [--limit N]
                                         [--n-bands 24] [--mds-dims 12]
                                         [--energy-threshold 0.1]
                                         [--cache PATH]
                                         [--manifest PATH ...]
    Then open http://localhost:8000/
"""

import sys
import os
import re
import json
import shutil
import argparse
import warnings
import pickle
import mimetypes
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TRANSFORM     = "nsgt"
DEFAULT_SRC_DIRS = [
    "~/Dropbox/Projets/Kenoma/Daath/popstar Project/Samples/bending/src/solo",
    "~/Dropbox/Projets/Kenoma/Daath/popstar Project/Samples/bending/src/duos",
]
DEFAULT_GEN_DIRS = [
    "/Volumes/Chevre/outs_bended_dac/generations/solo/codes",
    "/Volumes/Chevre/outs_bended_dac/generations/solo/decoder",
    "/Volumes/Chevre/outs_bended_dac/generations/solo/encoder",
    "/Volumes/Chevre/outs_bended_dac/generations/solo/z",
]
#: Every projection `embed_source` knows. All of them by default -- the ones
#: whose library is not installed are skipped at start (see `available_methods`).
ALL_METHODS           = ["tsne", "som", "phate", "mst", "pca", "umap", "isomap",
                         "diffmap", "ica", "sammon", "graph", "geodesic"]
DEFAULT_METHODS       = ALL_METHODS
DEFAULT_PORT          = 8000
DEFAULT_DURATION      = 10
DEFAULT_SR            = 22050
DEFAULT_N_BANDS       = 24
DEFAULT_MDS_DIMS      = 12
DEFAULT_ENERGY_THRESH = 0.1
N_FFT                 = 1024
HOP                   = 256

# ---------------------------------------------------------------------------
# Django: configured when the server starts (see `serve`), not on import, so
# the module can be imported next to another Django app -- the graph viewer.
# ---------------------------------------------------------------------------

def _configure_django():
    import django
    from django.conf import settings
    if not settings.configured:
        settings.configure(
            DEBUG=False,
            SECRET_KEY="audio-explorer-local-only",
            ROOT_URLCONF=__name__,
            ALLOWED_HOSTS=["*"],
            TEMPLATES=[],
            LOGGING_CONFIG=None,
        )
        django.setup()


from django.http import HttpResponse, FileResponse, JsonResponse
from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET

# ---------------------------------------------------------------------------
# Global app state (populated before server starts)
# ---------------------------------------------------------------------------

APP = {
    "records":     None,
    "source_data": None,
    "n_bands":     DEFAULT_N_BANDS,
    "methods":     DEFAULT_METHODS,
    "param_names": {},      # {field: label} of the bending values, from a manifest
}

# ---------------------------------------------------------------------------
# Audio processing helpers
# ---------------------------------------------------------------------------

def mel_band_label(band_idx: int, n_bands: int, sr: int = DEFAULT_SR) -> str:
    try:
        import librosa
        freqs = librosa.mel_frequencies(n_mels=n_bands, fmin=0.0, fmax=sr / 2)
        hz = freqs[band_idx]
        return f"b{band_idx} ({hz:.0f} Hz)" if hz < 1000 else f"b{band_idx} ({hz/1000:.1f} kHz)"
    except Exception:
        return f"band {band_idx}"


_NUMERIC_PLACEHOLDER_NAMES = {'n', 'i', 'idx', 'index', 'num', 'count'}


def _compile_bended_pattern(template):
    """Compile a template like '{originalName}_{part}_{idx}' into a regex.

    {originalName}        → greedy (.+), may contain underscores.
    {idx}/{n}/{i}/…       → \\d+ (digits only, non-capturing).
    Any other {name}      → [^_]+ (no underscores).
    Returns a compiled re.Pattern, or None if template is falsy.
    """
    if not template:
        return None
    parts = re.split(r'\{(\w+)\}', template)
    regex = '^'
    for i, part in enumerate(parts):
        if i % 2 == 0:
            regex += re.escape(part)
        else:
            name = part
            if name == 'originalName':
                regex += f'(?P<{name}>.+)'
            elif name in _NUMERIC_PLACEHOLDER_NAMES:
                regex += r'\d+'          # non-capturing digits
            else:
                regex += f'(?P<{name}>[^_]+)'
    regex += '$'
    return re.compile(regex)


def _match_bended_file(filestem, dir_stem, pattern=None):
    """Return (source_stem, bending_type_override) for a bended filename.

    source_stem          — the original file identifier.
    bending_type_override — extracted from non-originalName/non-numeric pattern
                            groups; None means fall back to directory-based type.

    Returns (None, None) when an explicit pattern is provided but doesn't match
    (caller should treat the file as a potential original, not a bended file).
    """
    if pattern is not None:
        m = pattern.match(filestem)
        if not m or 'originalName' not in m.groupdict():
            return None, None
        source_stem = m.group('originalName')
        extra = [v for k, v in m.groupdict().items()
                 if k != 'originalName' and k not in _NUMERIC_PLACEHOLDER_NAMES]
        btype_override = '_'.join(extra) if extra else None
        return source_stem, btype_override
    # Built-in conventions (no explicit pattern)
    if "__" in filestem:
        return filestem.split("__")[0], None
    mo = re.match(r'^(.+)_\d+$', filestem)
    if mo:
        return mo.group(1), None
    return dir_stem, None


def discover_files(src_dirs, gen_dirs, sources_filter=None, bended_pattern=None):
    originals = []
    for d in src_dirs:
        p = Path(d).expanduser()
        if p.exists():
            originals += sorted(p.glob("*.wav"))

    if sources_filter:
        sources_filter = set(sources_filter)
        originals = [p for p in originals if p.stem in sources_filter]

    known_stems  = {p.stem for p in originals}
    stem_to_path = {p.stem: p for p in originals}

    def _scan(filter_stems, warn):
        """Scan gen_dirs for bended files.

        Returns (bended_list, orig_candidates) where orig_candidates maps
        stem → path for files that don't match the bended pattern (only
        populated when bended_pattern is set and filter_stems is empty).
        """
        result, orig_candidates = [], {}
        for d in gen_dirs:
            p = Path(d).expanduser()
            if not p.exists():
                if warn:
                    print(f"  [warn] not found: {p}")
                continue
            dir_stem = p.name
            for f in sorted(p.rglob("*.wav")):
                rel = f.relative_to(p)
                dir_btype = rel.parts[0] if len(rel.parts) > 1 else p.name
                if filter_stems:
                    # Original-matching mode
                    if bended_pattern:
                        source_stem, btype_override = _match_bended_file(f.stem, None, bended_pattern)
                    else:
                        source_stem = f.stem.split("__")[0] if "__" in f.stem else f.stem
                        btype_override = None
                    if source_stem is None or source_stem not in filter_stems:
                        continue
                else:
                    source_stem, btype_override = _match_bended_file(f.stem, dir_stem, bended_pattern)
                    if source_stem is None:
                        # Explicit pattern didn't match — could be an original file
                        if bended_pattern and f.stem not in orig_candidates:
                            orig_candidates[f.stem] = f
                        continue
                result.append({
                    "path":         f,
                    "source_stem":  source_stem,
                    "bending_type": btype_override if btype_override else dir_btype,
                })
        return result, orig_candidates

    bended, orig_candidates = _scan(known_stems, warn=True)

    # If originals were found but nothing matched, fall back to no-original mode
    if known_stems and not bended:
        print("  No matching originals — extracting source names from filenames.")
        bended, orig_candidates = _scan(set(), warn=False)
        known_stems  = set()
        stem_to_path = {}

    if not known_stems:
        n_found = 0
        for stem in {b["source_stem"] for b in bended}:
            if stem in orig_candidates:
                stem_to_path[stem] = orig_candidates[stem]
                n_found += 1
            else:
                stem_to_path[stem] = None  # use mean spectrogram
        if bended:
            if n_found:
                print(f"  Found {n_found} original(s) in gen-dirs "
                      f"({len(stem_to_path) - n_found} group(s) will use mean reference).")
            else:
                print("  Using mean spectrogram per group as reference.")

    print(f"Found {len(originals)} originals, {len(bended)} bended files.")
    return originals, bended, stem_to_path


def _load(path: Path, sr: int, duration: float) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=sr, duration=duration, mono=True)
    return y


def _nsgt_available():
    try:
        import nsgt  # noqa: F401
        return True
    except ImportError:
        return False


#: The libraries each projection needs beyond numpy / scikit-learn.
_METHOD_DEPS = {"phate": "phate", "som": "minisom", "umap": "umap",
                "mst": "networkx", "graph": "networkx"}


def available_methods(methods):
    """``methods`` without the ones whose library is not installed."""
    import importlib.util
    return [m for m in methods
            if m not in _METHOD_DEPS or importlib.util.find_spec(_METHOD_DEPS[m]) is not None]


def _param_scalars(params):
    """A generation's bending values as scalars: numbers stay, booleans count."""
    out = {}
    for k, v in (params or {}).items():
        if isinstance(v, bool):
            out["param:" + k] = int(v)
        elif isinstance(v, (int, float)):
            out["param:" + k] = float(v)
    return out


def load_manifest(manifest_paths):
    """``(originals, bended, stem_to_path, param_names)`` from generation manifests.

    A manifest (written by torchbend's generate mode) lists every file with the
    input it was made from (``source``), a ``type`` and the bending values
    (``params``); ``sources`` names the reference recording of each input,
    when it had one. The explorer then groups by input and measures each
    generation against its source -- or, without one, against the group's mean.
    """
    bended, stem_to_path, param_names = [], {}, {}
    for mp in manifest_paths:
        mp = Path(mp).expanduser()
        root = mp.parent
        data = json.loads(mp.read_text())
        for field, desc in (data.get("parameters") or {}).items():
            param_names[field] = (desc or {}).get("label") or field
        for label, rel in (data.get("sources") or {}).items():
            path = root / rel
            if path.exists():
                stem_to_path[label] = path
        for f in data.get("files") or []:
            path = root / f["path"]
            if path.suffix.lower() != ".wav" or not path.exists():
                continue
            source = f.get("source") or "all"
            stem_to_path.setdefault(source, None)
            bended.append({
                "path":         path,
                "source_stem":  source,
                "bending_type": f.get("type") or data.get("fn") or "generation",
                "params":       f.get("params") or {},
            })
    originals = [p for p in stem_to_path.values() if p is not None]
    print(f"Found {len(originals)} source(s), {len(bended)} generated file(s) in "
          f"{len(manifest_paths)} manifest(s).")
    return originals, bended, stem_to_path, param_names


def _compute_spectrogram(y, sr, n_bands, transform="mel"):
    if transform == "nsgt":
        try:
            from nsgt import NSGT, MelScale
            scale = MelScale(fmin=27.5, fmax=float(sr) / 2, n_bins=n_bands)
            T = NSGT(scale, sr, len(y), real=True, matrixform=True)
            S = np.abs(np.array(list(T.forward(y)))) ** 2
            if S.shape[0] > n_bands:
                ex = S.shape[0] - n_bands
                S = S[ex // 2: ex // 2 + n_bands]
            elif S.shape[0] < n_bands:
                S = np.vstack([S, np.zeros((n_bands - S.shape[0], S.shape[1]))])
            return S.astype(float)
        except ImportError:
            raise ImportError("pip install nsgt")
        except Exception as e:
            warnings.warn(f"NSGT failed ({e}), falling back to mel")
    import librosa
    return librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_bands, n_fft=N_FFT, hop_length=HOP, power=2.0
    ).astype(float)


def _energy_mask(S, threshold):
    energy = S.sum(axis=0)
    return (energy > threshold * energy.max()
            if energy.max() > 0 else np.ones(S.shape[1], dtype=bool))


def multiband_distance(S_orig, S_bend, emask):
    T    = min(S_orig.shape[1], S_bend.shape[1], len(emask))
    mask = emask[:T]
    if not mask.any():
        mask = np.ones(T, dtype=bool)
    eps = 1e-8
    return np.abs(
        np.log(S_orig[:, :T][:, mask] + eps) -
        np.log(S_bend[:, :T][:, mask] + eps)
    ).mean(axis=1)


def compute_source_distances(orig_path, bended_records, sr, duration,
                              n_bands, energy_threshold, cache, transform):
    version = f"v2|{n_bands}|{int(duration)}|{float(energy_threshold)}|{transform}"

    if orig_path is None:
        import hashlib
        paths_hash = hashlib.md5(
            "|".join(sorted(str(r["path"]) for r in bended_records)).encode()
        ).hexdigest()[:10]
        ref_key  = f"mean:{paths_hash}"
        all_keys = [f"{rec['path']}|{ref_key}|{version}" for rec in bended_records]
        new_entries = 0
        if not all(k in cache for k in all_keys):
            spectrograms = [
                _compute_spectrogram(_load(rec["path"], sr, duration), sr, n_bands, transform)
                for rec in bended_records
            ]
            min_t  = min(S.shape[1] for S in spectrograms)
            S_orig = np.mean(np.stack([S[:, :min_t] for S in spectrograms], axis=0), axis=0)
            emask  = _energy_mask(S_orig, energy_threshold)
            for rec, S_b in zip(bended_records, spectrograms):
                key = f"{rec['path']}|{ref_key}|{version}"
                if key not in cache:
                    cache[key] = multiband_distance(S_orig, S_b, emask)
                    new_entries += 1
        dist_vecs, scalars_list = [], []
        for rec in bended_records:
            vec = cache[f"{rec['path']}|{ref_key}|{version}"]
            dist_vecs.append(vec)
            scalars = {"dist_total": float(vec.mean())}
            for j, v in enumerate(vec):
                scalars[f"dist_b{j}"] = float(v)
            scalars_list.append(scalars)
        return np.array(dist_vecs), scalars_list, new_entries

    y_orig  = _load(orig_path, sr, duration)
    S_orig  = _compute_spectrogram(y_orig, sr, n_bands, transform)
    emask   = _energy_mask(S_orig, energy_threshold)

    dist_vecs, scalars_list, new_entries = [], [], 0
    for rec in bended_records:
        key = f"{rec['path']}|{orig_path}|{version}"
        if key not in cache:
            S_b = _compute_spectrogram(_load(rec["path"], sr, duration), sr, n_bands, transform)
            cache[key] = multiband_distance(S_orig, S_b, emask)
            new_entries += 1
        vec = cache[key]
        dist_vecs.append(vec)
        scalars = {"dist_total": float(vec.mean())}
        for j, v in enumerate(vec):
            scalars[f"dist_b{j}"] = float(v)
        scalars_list.append(scalars)

    return np.array(dist_vecs), scalars_list, new_entries


def embed_source(dist_matrix, mds_dims, method):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([np.zeros((1, dist_matrix.shape[1])), dist_matrix])
    n = len(X)
    if n < 4:
        method = "pca"

    X_scaled = StandardScaler().fit_transform(X)
    n_mds    = min(mds_dims, X_scaled.shape[1], n - 1)
    X_mds    = PCA(n_components=n_mds, random_state=42).fit_transform(X_scaled)
    X_mds   -= X_mds[0]

    if n < 4 or method == "pca":
        proj = PCA(n_components=min(2, n - 1), random_state=42).fit_transform(X_mds)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        perp = min(max(5, n // 5), n - 1)
        import inspect
        # scikit-learn 1.5 renamed n_iter to max_iter (and 1.7 dropped n_iter)
        iters = ("max_iter" if "max_iter" in inspect.signature(TSNE).parameters
                 else "n_iter")
        proj = TSNE(n_components=2, perplexity=perp, random_state=42,
                    learning_rate="auto", init="pca", **{iters: 1000}).fit_transform(X_mds)

    elif method == "phate":
        import phate
        proj = phate.PHATE(n_components=2, knn=min(5, max(2, n // 20)),
                           random_state=42, verbose=False).fit_transform(X_mds)

    elif method == "mst":
        import networkx as nx
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial.distance import cdist
        G   = nx.from_scipy_sparse_array(minimum_spanning_tree(cdist(X_mds, X_mds)))
        pos = nx.kamada_kawai_layout(G)
        proj = np.array([pos[i] for i in range(n)])

    elif method == "som":
        from minisom import MiniSom
        import math
        grid = max(2, math.ceil(math.sqrt(n)))
        som  = MiniSom(grid, grid, X_mds.shape[1], sigma=grid / 3.0,
                       learning_rate=0.5, neighborhood_function="gaussian", random_seed=42)
        som.train(X_mds, num_iteration=500 * n, verbose=False)
        bmu_counts = {}
        rows, cols = [], []
        for vec in X_mds:
            r, c  = som.winner(vec)
            count = bmu_counts.get((r, c), 0)
            bmu_counts[(r, c)] = count + 1
            jit = np.random.default_rng(count * 1000 + r * 100 + c)
            rows.append(r + jit.uniform(-0.35, 0.35))
            cols.append(c + jit.uniform(-0.35, 0.35))
        proj = np.column_stack([cols, rows]).astype(float)

    elif method == "umap":
        import umap as umap_lib
        nn   = min(15, max(2, n // 10))
        proj = umap_lib.UMAP(n_components=2, n_neighbors=nn,
                             random_state=42, min_dist=0.1).fit_transform(X_mds)

    elif method == "isomap":
        from sklearn.manifold import Isomap
        nn   = min(10, max(2, n // 15))
        proj = np.asarray(Isomap(n_components=2, n_neighbors=nn).fit_transform(X_mds))

    elif method == "diffmap":
        from sklearn.manifold import SpectralEmbedding
        gamma = 1.0 / (X_mds.shape[1] * X_mds.var() + 1e-8)
        proj  = SpectralEmbedding(n_components=2, affinity="rbf",
                                   gamma=gamma, random_state=42).fit_transform(X_mds)

    elif method == "ica":
        from sklearn.decomposition import FastICA
        proj = FastICA(n_components=min(2, n - 1), random_state=42,
                       max_iter=1000, tol=1e-4).fit_transform(X_mds)

    elif method == "sammon":
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(X_mds).astype(float)
        np.fill_diagonal(D, 1e-10)
        proj2 = PCA(n_components=2, random_state=42).fit_transform(X_mds).copy()
        scale = D.sum(); lr = 0.5
        for _ in range(400):
            Dp = pairwise_distances(proj2).astype(float)
            np.fill_diagonal(Dp, 1e-10)
            W = (D - Dp) / (D * Dp); np.fill_diagonal(W, 0.0)
            proj2 -= (lr / scale) * (W.sum(axis=1)[:, None] * proj2 - W @ proj2)
        proj = proj2

    elif method == "graph":
        import networkx as nx
        from scipy.spatial.distance import cdist
        D = cdist(X_mds, X_mds); k = min(15, max(2, n // 10))
        G = nx.Graph(); G.add_nodes_from(range(n))
        for i in range(n):
            for j in np.argsort(D[i])[1:k + 1]:
                G.add_edge(int(i), int(j), weight=float(1.0 / (D[i, j] + 1e-8)))
        pos  = nx.spring_layout(G, seed=42, iterations=150, weight="weight")
        proj = np.array([pos[i] for i in range(n)])

    elif method == "geodesic":
        from scipy.sparse.csgraph import shortest_path
        from scipy.spatial.distance import cdist
        D_euc = cdist(X_mds, X_mds); k = min(15, max(3, n // 10))
        adj   = np.zeros_like(D_euc)
        for i in range(n):
            for j in np.argsort(D_euc[i])[1:k + 1]:
                adj[i, j] = adj[j, i] = D_euc[i, j]
        geo = shortest_path(adj, method="D", directed=False, indices=0)
        finite = geo[np.isfinite(geo)]
        geo[~np.isfinite(geo)] = finite.max() if len(finite) else 1.0
        pca2   = PCA(n_components=2, random_state=42).fit_transform(X_mds)
        angles = np.arctan2(pca2[:, 1], pca2[:, 0])
        proj   = np.column_stack([geo * np.cos(angles), geo * np.sin(angles)])

    else:
        raise ValueError(f"Unknown method: {method}")

    proj = np.asarray(proj, dtype=float)
    if proj.ndim == 1:
        proj = proj.reshape(-1, 1)
    if proj.shape[1] < 2:
        proj = np.hstack([proj, np.zeros((len(proj), 2 - proj.shape[1]))])
    proj -= proj[0]
    return proj, X_mds


def build_source_data(originals, bended, stem_to_path,
                      sr, duration, n_bands, mds_dims, energy_threshold,
                      methods, cache_path, transform="mel",
                      proj_cache_path=None, recompute=False):
    import hashlib

    cache = {}
    if not recompute and cache_path and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"  Loaded dist cache ({len(cache)} entries).")

    bended_hash = hashlib.md5(
        "|".join(sorted(str(b["path"]) for b in bended)).encode()
    ).hexdigest()[:10]
    proj_version = (
        f"v1|{n_bands}|{int(duration)}|{float(energy_threshold)}"
        f"|{transform}|{mds_dims}|{'_'.join(sorted(methods))}|{bended_hash}"
    )
    proj_cache = {}
    if not recompute and proj_cache_path and Path(proj_cache_path).exists():
        with open(proj_cache_path, "rb") as f:
            saved = pickle.load(f)
        if saved.get("version") == proj_version:
            proj_cache = saved.get("data", {})
            print(f"  Loaded projection cache ({len(proj_cache)} stems).")
        else:
            print("  Projection cache version mismatch — will recompute projections.")

    source_stems = sorted({b["source_stem"] for b in bended})
    records      = []
    source_data  = {}
    total_new    = 0

    orig_record_idx = {}
    for stem in source_stems:
        orig_path = stem_to_path.get(stem)
        orig_record_idx[stem] = len(records)
        records.append({
            "name":         orig_path.name if orig_path else f"{stem} [mean ref]",
            "path":         orig_path,
            "is_original":  True,
            "source_stem":  stem,
            "bending_type": "original" if orig_path else "mean",
            "dist_total":   0.0,
            "scalars":      {"dist_total": 0.0,
                             **{f"dist_b{j}": 0.0 for j in range(n_bands)}},
        })

    new_proj_stems = 0
    for stem in source_stems:
        stem_bended = [b for b in bended if b["source_stem"] == stem]
        if not stem_bended:
            continue

        dist_matrix, scalars_list, new_entries = compute_source_distances(
            stem_to_path[stem], stem_bended, sr, duration,
            n_bands, energy_threshold, cache, transform
        )
        total_new += new_entries

        bend_start = len(records)
        for i, b in enumerate(stem_bended):
            records.append({
                "name":         b["path"].name,
                "path":         b["path"],
                "is_original":  False,
                "source_stem":  stem,
                "bending_type": b["bending_type"],
                "dist_total":   float(scalars_list[i]["dist_total"]),
                # the values a generation was made with colour it too
                "scalars":      {**scalars_list[i], **_param_scalars(b.get("params"))},
            })

        o_idx   = orig_record_idx[stem]
        b_idxs  = list(range(bend_start, bend_start + len(stem_bended)))
        indices = [o_idx] + b_idxs

        if stem in proj_cache:
            # patch indices (record positions shift each run) then reuse projections
            cached = proj_cache[stem]
            projections = {m: {**v, "indices": indices}
                           for m, v in cached.items() if not m.startswith("_")}
            # restore cluster labels into records
            for key, labels in cached.get("_clusters", {}).items():
                for pos, gi in enumerate(indices):
                    records[gi]["scalars"][key] = labels[pos] if not records[gi]["is_original"] else -1
            print(f"  [{stem}] {len(stem_bended)} files ... cached ({len(projections)} projections)")
        else:
            print(f"  [{stem}] {len(stem_bended)} files ...", end=" ", flush=True)
            projections  = {}
            X_mds_saved  = None
            for method in methods:
                try:
                    proj, X_mds = embed_source(dist_matrix, mds_dims, method)
                    if X_mds_saved is None:
                        X_mds_saved = X_mds
                    projections[method] = {
                        "indices": indices,
                        "x":       proj[:, 0].tolist(),
                        "y":       proj[:, 1].tolist(),
                    }
                except Exception as e:
                    print(f"\n    [{stem}] {method.upper()} failed: {e}")

            clusters_to_cache = {}
            if X_mds_saved is not None and len(X_mds_saved) > 3:
                try:
                    from sklearn.cluster import AgglomerativeClustering
                    for k in [3, 5, 8]:
                        if k >= len(X_mds_saved):
                            continue
                        labels = AgglomerativeClustering(
                            n_clusters=k, linkage="ward"
                        ).fit_predict(X_mds_saved)
                        key = f"cluster_k{k}"
                        clusters_to_cache[key] = [int(l) for l in labels]
                        for pos, gi in enumerate(indices):
                            records[gi]["scalars"][key] = (
                                int(labels[pos]) if not records[gi]["is_original"] else -1
                            )
                except Exception as e:
                    print(f"\n    [{stem}] clustering failed: {e}")

            proj_cache[stem] = {
                "_clusters": clusters_to_cache,
                **{m: {k: v for k, v in p.items() if k != "indices"}
                   for m, p in projections.items()},
            }
            new_proj_stems += 1
            print(f"done ({len(projections)} projections)")

        source_data[stem] = projections

    if cache_path and total_new > 0:
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"  Saved dist cache ({len(cache)} entries) -> {cache_path}")

    if proj_cache_path and new_proj_stems > 0:
        with open(proj_cache_path, "wb") as f:
            pickle.dump({"version": proj_version, "data": proj_cache}, f)
        print(f"  Saved projection cache ({len(proj_cache)} stems) -> {proj_cache_path}")

    return records, source_data


# ---------------------------------------------------------------------------
# Django views
# ---------------------------------------------------------------------------

@require_GET
def view_index(request):
    records      = APP["records"]
    source_data  = APP["source_data"]
    n_bands      = APP["n_bands"]
    methods      = APP["methods"]

    source_stems = sorted(source_data.keys())
    n_orig   = sum(1 for r in records if     r["is_original"])
    n_bended = sum(1 for r in records if not r["is_original"])

    scalar_groups = {
        "distance": {
            "dist_total": "Total distance",
            **{f"dist_b{j}": mel_band_label(j, n_bands) for j in range(n_bands)},
        },
        "clustering": {
            "cluster_k3": "Clusters k=3",
            "cluster_k5": "Clusters k=5",
            "cluster_k8": "Clusters k=8",
        },
    }
    if APP.get("param_names"):
        scalar_groups["bending values"] = {
            "param:" + k: v for k, v in APP["param_names"].items()}
    scalar_optgroups = '<option value="">— color by type —</option>\n' + "\n".join(
        f'<optgroup label="{gname}">\n' +
        "\n".join(f'  <option value="{k}">{v}</option>' for k, v in gitems.items()) +
        "\n</optgroup>"
        for gname, gitems in scalar_groups.items()
    )
    all_scalar_keys = ["dist_total"] + [f"dist_b{j}" for j in range(n_bands)]

    records_js = [
        {
            "idx":          i,
            "name":         r["name"],
            "path":         str(r["path"]) if r["path"] else "",
            "is_original":  r["is_original"],
            "source_stem":  r["source_stem"],
            "bending_type": r["bending_type"],
            "dist_total":   float(r.get("dist_total", 0)),
            "scalars":      r.get("scalars", {}),
        }
        for i, r in enumerate(records)
    ]
    source_data_js = {
        stem: {m: proj for m, proj in md.items()}
        for stem, md in source_data.items()
    }
    default_method = methods[0] if methods else "tsne"

    btn_html = "\n  ".join(
        f'<button id="btn-{m}" class="method-btn" onclick="switchMethod(\'{m}\')">{m.upper()}</button>'
        for m in methods
    )
    sbar_html = "".join(
        f'<div class="sbar-row"><span class="sbar-label">{lbl}</span>'
        f'<div class="sbar-track"><div class="sbar-fill" id="sbar-{k}"></div></div></div>'
        for k, lbl in [("dist_total", "total dist")] +
                       [(f"dist_b{j}", mel_band_label(j, n_bands)) for j in range(n_bands)]
    )
    _btype_palette = [
        "#2196f3", "#43a047", "#ff9800", "#9c27b0", "#e53935",
        "#00acc1", "#f4511e", "#8e24aa", "#3949ab", "#00897b",
        "#c0ca33", "#d81b60", "#6d4c41", "#546e7a", "#fdd835",
    ]
    all_btypes = sorted({r["bending_type"] for r in records if not r["is_original"]})
    btype_color_map = {bt: _btype_palette[i % len(_btype_palette)]
                       for i, bt in enumerate(all_btypes)}
    btype_filter_opts = '<option value="">all</option>\n' + "".join(
        f'<option value="{bt}">{bt}</option>' for bt in all_btypes
    )

    source_opts = "".join(
        f'<option value="{i}">{s}</option>' for i, s in enumerate(source_stems)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Audio Similarity Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
    background: #f0f0f5; color: #1d1d1f;
    height: 100vh; display: flex; flex-direction: column; overflow: hidden;
  }}
  #header {{ padding: 10px 16px 0; flex-shrink: 0; }}
  h1 {{ font-size: 17px; font-weight: 600; display: inline; }}
  .subtitle {{ font-size: 11px; color: #6e6e73; margin-left: 10px; }}
  #controls {{
    display: flex; gap: 7px; align-items: center; flex-wrap: wrap;
    background: #fff; padding: 8px 14px; margin: 8px 16px;
    border-radius: 10px; border: 1px solid #d2d2d7;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05); flex-shrink: 0;
  }}
  .ctrl-label {{
    font-size: 10px; font-weight: 600; color: #6e6e73;
    text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap;
  }}
  .ctrl-sep {{ width: 1px; height: 20px; background: #d2d2d7; margin: 0 2px; }}
  .method-btn {{
    padding: 4px 12px; background: #f0f0f5; color: #0071e3;
    border: 1.5px solid #0071e3; cursor: pointer; border-radius: 16px;
    font-size: 11px; font-weight: 500; transition: background 0.1s, color 0.1s;
  }}
  .method-btn:hover  {{ background: #deeaff; }}
  .method-btn.active {{ background: #0071e3; color: #fff; }}
  select {{
    background: #fff; color: #1d1d1f; border: 1.5px solid #d2d2d7;
    padding: 4px 8px; border-radius: 7px; font-size: 11px;
    font-family: inherit; cursor: pointer;
  }}
  #source-nav {{ display: flex; align-items: center; gap: 8px; margin-left: auto; }}
  .nav-btn {{
    width: 26px; height: 26px; border-radius: 50%;
    background: #f0f0f5; border: 1.5px solid #d2d2d7; cursor: pointer;
    font-size: 14px; display: flex; align-items: center; justify-content: center;
    transition: background 0.1s;
  }}
  .nav-btn:hover {{ background: #deeaff; border-color: #0071e3; }}
  #source-label {{ font-size: 13px; font-weight: 600; min-width: 120px; text-align: center; }}
  #source-count  {{ font-size: 10px; color: #6e6e73; }}
  #plot-wrap {{
    flex: 1; margin: 0 16px 16px; background: #fff;
    border: 1px solid #d2d2d7; border-radius: 12px;
    overflow: hidden; min-height: 0; position: relative;
  }}
  #main-plot {{ width: 100%; height: 100%; }}

  /* Audio panel */
  #audio-panel {{
    position: fixed; bottom: 20px; right: 20px;
    background: #fff; border: 1px solid #d2d2d7;
    padding: 13px 15px; border-radius: 13px; width: 300px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.11); z-index: 100;
  }}
  #panel-header {{
    display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px;
  }}
  #audio-panel h3 {{
    font-size: 9px; font-weight: 600; letter-spacing: 0.8px;
    text-transform: uppercase; color: #6e6e73; margin: 0;
  }}
  #panel-toggle {{
    background: none; border: none; cursor: pointer;
    font-size: 14px; color: #6e6e73; padding: 0 2px; line-height: 1;
  }}
  #panel-toggle:hover {{ color: #0071e3; }}
  #audio-name {{
    font-size: 11px; font-weight: 500; color: #1d1d1f;
    margin-bottom: 7px; word-break: break-all; min-height: 13px;
  }}
  audio {{ width: 100%; height: 30px; }}
  #audio-meta {{ font-size: 10px; color: #6e6e73; margin-top: 5px; min-height: 13px; }}
  #scalar-bars {{ margin-top: 9px; display: flex; flex-direction: column; gap: 3px; }}
  .sbar-row {{ display: flex; align-items: center; gap: 5px; }}
  .sbar-label {{ font-size: 9px; color: #6e6e73; width: 70px; text-align: right; flex-shrink: 0; }}
  .sbar-track {{ flex: 1; height: 4px; background: #e8e8ed; border-radius: 2px; overflow: hidden; }}
  .sbar-fill  {{ height: 100%; width: 0%; border-radius: 2px; transition: width 0.2s ease; }}
  .hint {{ font-size: 9px; color: #aeaeb2; margin-top: 6px; }}

  /* Wishlist add button */
  #wishlist-add-btn {{
    display: none; margin-top: 8px; width: 100%;
    padding: 5px 10px; background: #fff3e0; color: #e65100;
    border: 1.5px solid #ff9800; border-radius: 8px; cursor: pointer;
    font-size: 11px; font-weight: 500; transition: background 0.1s;
  }}
  #wishlist-add-btn:hover {{ background: #ffe0b2; }}
  #wishlist-add-btn.wishlisted {{
    background: #e8f5e9; color: #2e7d32; border-color: #43a047;
  }}

  /* Wishlist panel */
  #wishlist-panel {{
    position: fixed; bottom: 20px; left: 20px;
    background: #fff; border: 1px solid #d2d2d7; border-radius: 13px;
    width: 320px; max-height: 60vh;
    box-shadow: 0 4px 20px rgba(0,0,0,0.11); z-index: 100;
    display: flex; flex-direction: column;
  }}
  #wishlist-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 13px 8px; border-bottom: 1px solid #f0f0f5; flex-shrink: 0;
  }}
  #wishlist-header h3 {{
    font-size: 9px; font-weight: 600; letter-spacing: 0.8px;
    text-transform: uppercase; color: #6e6e73; margin: 0;
  }}
  #wishlist-header-btns {{ display: flex; gap: 5px; align-items: center; }}
  .wl-hbtn {{
    padding: 3px 9px; border-radius: 8px; cursor: pointer; font-size: 10px;
    font-weight: 500; border: 1.5px solid;
  }}
  #wl-export-btn, #wl-copy-btn {{
    color: #0071e3; border-color: #0071e3; background: #f0f0f5;
  }}
  #wl-export-btn:hover, #wl-copy-btn:hover {{ background: #deeaff; }}
  #wl-clear-btn  {{ color: #e53935; border-color: #e53935; background: #fff; }}
  #wl-clear-btn:hover {{ background: #ffebee; }}
  #wishlist-toggle {{
    background: none; border: none; cursor: pointer;
    font-size: 14px; color: #6e6e73; padding: 0 2px; line-height: 1;
  }}
  #wishlist-toggle:hover {{ color: #0071e3; }}
  #wishlist-body {{ overflow-y: auto; padding: 6px 10px 10px; }}
  #wishlist-empty {{
    font-size: 10px; color: #aeaeb2; text-align: center; padding: 12px 0;
  }}
  .wl-source-group {{ margin-bottom: 10px; }}
  .wl-source-title {{
    font-size: 9px; font-weight: 700; color: #6e6e73; letter-spacing: 0.5px;
    text-transform: uppercase; margin-bottom: 4px;
  }}
  .wl-item {{
    display: flex; align-items: center; justify-content: space-between;
    background: #f9f9fb; border: 1px solid #e8e8ed;
    border-radius: 7px; padding: 4px 8px; margin-bottom: 3px;
    font-size: 10px; gap: 6px;
  }}
  .wl-item-name {{
    flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #1d1d1f;
  }}
  .wl-item-type {{
    font-size: 9px; font-weight: 600; padding: 1px 5px; border-radius: 4px; flex-shrink: 0;
  }}
  .wl-remove-btn {{
    background: none; border: none; cursor: pointer; color: #aeaeb2;
    font-size: 12px; line-height: 1; flex-shrink: 0; padding: 0;
  }}
  .wl-remove-btn:hover {{ color: #e53935; }}
  #wl-count-badge {{
    background: #ff9800; color: #fff; border-radius: 10px;
    padding: 1px 6px; font-size: 9px; font-weight: 700; min-width: 18px; text-align: center;
  }}

  /* Right-click context menu */
  #ctx-menu {{
    position: fixed; z-index: 9999;
    background: #fff; border: 1px solid #d2d2d7; border-radius: 9px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.15); padding: 4px 0; min-width: 160px; display: none;
  }}
  .ctx-item {{
    padding: 7px 14px; font-size: 11px; cursor: pointer; color: #1d1d1f; white-space: nowrap;
  }}
  .ctx-item:hover {{ background: #f0f0f5; }}
  .ctx-item.ctx-wl   {{ color: #e65100; }}
  .ctx-item.ctx-unwl {{ color: #2e7d32; }}
  .ctx-sep-line {{ height: 1px; background: #f0f0f5; margin: 3px 0; }}

  /* Copy-files modal */
  #copy-modal {{
    display: none; position: fixed; inset: 0; z-index: 9998;
    background: rgba(0,0,0,0.35); align-items: center; justify-content: center;
  }}
  #copy-modal.open {{ display: flex; }}
  #copy-box {{
    background: #fff; border-radius: 14px; padding: 20px 22px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2); min-width: 360px;
  }}
  #copy-box h4 {{ font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #1d1d1f; }}
  #copy-box p  {{ font-size: 10px; color: #6e6e73; margin-bottom: 10px; }}
  #copy-dest {{
    width: 100%; padding: 6px 10px; border: 1.5px solid #d2d2d7;
    border-radius: 7px; font-size: 11px; font-family: inherit; margin-bottom: 10px;
  }}
  #copy-status {{ font-size: 10px; min-height: 14px; margin-bottom: 8px; color: #6e6e73; }}
  #copy-btns {{ display: flex; gap: 7px; justify-content: flex-end; }}
  .copy-btn {{
    padding: 5px 14px; border-radius: 8px; cursor: pointer;
    font-size: 11px; font-weight: 500; border: 1.5px solid;
  }}
  #copy-cancel {{ color: #6e6e73; border-color: #d2d2d7; background: #fff; }}
  #copy-cancel:hover {{ background: #f0f0f5; }}
  #copy-go {{ color: #fff; border-color: #0071e3; background: #0071e3; }}
  #copy-go:hover {{ background: #005bb5; }}
  #copy-go:disabled {{ background: #a0c4ff; border-color: #a0c4ff; cursor: default; }}
</style>
</head>
<body>
<div id="header">
  <h1>Audio Similarity Explorer</h1>
  <span class="subtitle">
    multi-band spectral distance &middot; {n_orig} originals &middot; {n_bended} bended
  </span>
</div>

<div id="controls">
  <span class="ctrl-label">Method</span>
  {btn_html}
  <div class="ctrl-sep"></div>
  <span class="ctrl-label">Color</span>
  <select id="color-select" onchange="switchColor(this.value)">
    {scalar_optgroups}
  </select>
  <div class="ctrl-sep"></div>
  <span class="ctrl-label">Bends</span>
  <select id="btype-filter" onchange="render()">
    {btype_filter_opts}
  </select>
  <div id="source-nav">
    <button class="nav-btn" onclick="stepSource(-1)">&#8249;</button>
    <div>
      <div id="source-label"></div>
      <div id="source-count"></div>
    </div>
    <button class="nav-btn" onclick="stepSource(1)">&#8250;</button>
    <select id="source-select" onchange="goSource(this.value)" style="max-width:130px">
      {source_opts}
    </select>
  </div>
</div>

<div id="plot-wrap">
  <div id="main-plot"></div>
</div>

<!-- Audio preview panel -->
<div id="audio-panel">
  <div id="panel-header">
    <h3>Audio Preview</h3>
    <button id="panel-toggle" onclick="togglePanel()">&#9662;</button>
  </div>
  <div id="audio-name">click a point to listen</div>
  <audio id="audio-player" controls></audio>
  <div id="audio-meta"></div>
  <div id="panel-body">
    <button id="wishlist-add-btn" onclick="toggleWishlistItem()">&#9829; Add to Wishlist</button>
    <div id="scalar-bars">{sbar_html}</div>
    <div class="hint">left-click = play &middot; right-click = wishlist menu &middot; W = toggle wishlist</div>
  </div>
</div>

<!-- Wishlist panel -->
<div id="wishlist-panel">
  <div id="wishlist-header">
    <h3>Wishlist &nbsp;<span id="wl-count-badge">0</span></h3>
    <div id="wishlist-header-btns">
      <button class="wl-hbtn" id="wl-export-btn" onclick="exportWishlistJSON()">JSON</button>
      <button class="wl-hbtn" id="wl-copy-btn"   onclick="openCopyModal()">Copy Files&hellip;</button>
      <button class="wl-hbtn" id="wl-clear-btn"  onclick="clearWishlist()">Clear</button>
      <button id="wishlist-toggle" onclick="toggleWishlistPanel()">&#9662;</button>
    </div>
  </div>
  <div id="wishlist-body">
    <div id="wishlist-empty">No items yet — right-click a point or click &#9829;</div>
  </div>
</div>

<!-- Right-click context menu -->
<div id="ctx-menu">
  <div class="ctx-item" id="ctx-play" onclick="ctxPlay()">&#9654; Play</div>
  <div class="ctx-sep-line"></div>
  <div class="ctx-item ctx-wl" id="ctx-wl-toggle" onclick="ctxToggleWishlist()">&#9829; Add to Wishlist</div>
</div>

<!-- Copy-files modal -->
<div id="copy-modal">
  <div id="copy-box">
    <h4>Copy Wishlist Files</h4>
    <p>Files will be copied server-side to <code>DEST/&lt;source_stem&gt;/</code>.</p>
    <input id="copy-dest" type="text" placeholder="/absolute/path/to/destination" />
    <div id="copy-status"></div>
    <div id="copy-btns">
      <button class="copy-btn" id="copy-cancel" onclick="closeCopyModal()">Cancel</button>
      <button class="copy-btn" id="copy-go"     onclick="doCopyFiles()">Copy</button>
    </div>
  </div>
</div>

<script>
const RECORDS         = {json.dumps(records_js)};
const SOURCE_DATA     = {json.dumps(source_data_js)};
const STEMS           = {json.dumps(source_stems)};
const ALL_SCALAR_KEYS = {json.dumps(all_scalar_keys)};
const BTYPE_COLORS = {json.dumps(btype_color_map)};

const SCALAR_RANGES = {{}};
ALL_SCALAR_KEYS.forEach(k => {{
  const vals = RECORDS.filter(r => !r.is_original && r.scalars[k] !== undefined)
                       .map(r => r.scalars[k]);
  SCALAR_RANGES[k] = {{ min: Math.min(...vals), max: Math.max(...vals) }};
}});

function norm(val, key) {{
  const {{min, max}} = SCALAR_RANGES[key] || {{ min: 0, max: 1 }};
  return max > min ? (val - min) / (max - min) : 0;
}}

let currentMethod  = "{default_method}";
let currentColor   = "cluster_k5";
let currentStem    = STEMS[0];
let plotReady      = false;
let selectedRecord = null;
let hoveredRecord  = null;
let ctxRecord      = null;   // snapshot at right-click time; survives plotly_unhover
const wishlist = {{}};

const LAYOUT = {{
  paper_bgcolor: "#fff", plot_bgcolor: "#fff",
  font: {{ family: "-apple-system, Helvetica Neue, Arial, sans-serif", size: 11, color: "#1d1d1f" }},
  margin: {{ l: 40, r: 20, t: 20, b: 40 }},
  xaxis: {{ showgrid: true, gridcolor: "#f0f0f5", zeroline: true, zerolinecolor: "#d2d2d7", zerolinewidth: 1.5 }},
  yaxis: {{ showgrid: true, gridcolor: "#f0f0f5", zeroline: true, zerolinecolor: "#d2d2d7", zerolinewidth: 1.5 }},
  legend: {{ bgcolor: "rgba(255,255,255,0.9)", bordercolor: "#d2d2d7", borderwidth: 1, itemsizing: "constant", font: {{size: 11}} }},
  hovermode: "closest",
}};
const CONFIG = {{ responsive: true, displayModeBar: false }};

function buildTraces(stem, method) {{
  const proj = (SOURCE_DATA[stem] || {{}})[method];
  if (!proj) return [];
  const {{indices, x, y}} = proj;
  const btypeFilter = document.getElementById("btype-filter").value;
  const scalarMode  = currentColor !== "";
  const origLocal = [], bendGroups = {{}};
  indices.forEach((gi, li) => {{
    const r = RECORDS[gi];
    if (r.is_original) {{ origLocal.push(li); return; }}
    if (btypeFilter && r.bending_type !== btypeFilter) return;
    (bendGroups[r.bending_type] = bendGroups[r.bending_type] || []).push(li);
  }});

  const rng = SCALAR_RANGES[currentColor] || {{ min: 0, max: 1 }};
  const isCluster = currentColor.startsWith("cluster_");
  const CLUSTER_PAL = ["#e53935","#1e88e5","#43a047","#fb8c00","#8e24aa","#00acc1","#6d4c41","#f06292"];
  const traces = [];

  Object.entries(bendGroups).forEach(([bt, locs]) => {{
    let markerColor, colorscale, cmin, cmax, hoverSuffix;
    if (isCluster) {{
      markerColor = locs.map(li => {{
        const k = (RECORDS[indices[li]].scalars[currentColor] || 0);
        return CLUSTER_PAL[Math.max(0, k) % CLUSTER_PAL.length];
      }});
      hoverSuffix = "<br>cluster: %{{customdata[2]}}";
    }} else if (scalarMode) {{
      markerColor = locs.map(li => RECORDS[indices[li]].scalars[currentColor] || 0);
      colorscale = "Plasma"; cmin = rng.min; cmax = rng.max;
      hoverSuffix = "<br>" + currentColor + ": %{{marker.color:.4f}}";
    }} else {{
      markerColor = BTYPE_COLORS[bt] || "#888";
      hoverSuffix = "<br>dist: %{{customdata[1]:.4f}}";
    }}
    traces.push({{
      type: "scatter", mode: "markers", name: bt,
      x: locs.map(li => x[li]), y: locs.map(li => y[li]),
      marker: {{
        size: 9, color: markerColor, opacity: 0.85, showscale: false,
        ...(colorscale ? {{ colorscale, cmin, cmax }} : {{}}),
        line: {{ width: 0.8, color: "rgba(0,0,0,0.2)" }},
      }},
      text: locs.map(li => RECORDS[indices[li]].name),
      customdata: locs.map(li => {{
        const r = RECORDS[indices[li]];
        return [indices[li], r.dist_total, r.scalars[currentColor] ?? ""];
      }}),
      hovertemplate: "<b>%{{text}}</b>" + hoverSuffix + "<extra></extra>",
    }});
  }});

  origLocal.forEach(li => {{
    traces.push({{
      type: "scatter", mode: "markers", name: "original", showlegend: false,
      x: [x[li]], y: [y[li]],
      marker: {{ size: 22, symbol: "star", color: "#e53935", line: {{ width: 2, color: "#fff" }} }},
      text: [RECORDS[indices[li]].name],
      customdata: [[indices[li], 0, ""]],
      hovertemplate: "<b>%{{text}}</b><br>[ORIGINAL]<extra></extra>",
    }});
  }});

  // Selection highlight — drawn last so it sits on top of everything
  if (selectedRecord) {{
    const li = indices.indexOf(selectedRecord.idx);
    if (li >= 0) {{
      traces.push({{
        type: "scatter", mode: "markers", name: "_sel", showlegend: false,
        x: [x[li]], y: [y[li]],
        marker: {{
          size: 28, symbol: "circle-open",
          color: "#fff", opacity: 1,
          line: {{ width: 3.5, color: "#fff" }},
        }},
        hoverinfo: "skip",
        customdata: [[selectedRecord.idx, selectedRecord.dist_total || 0, ""]],
      }});
      traces.push({{
        type: "scatter", mode: "markers", name: "_sel2", showlegend: false,
        x: [x[li]], y: [y[li]],
        marker: {{
          size: 34, symbol: "circle-open",
          color: "#ff3b30", opacity: 0.8,
          line: {{ width: 2, color: "#ff3b30" }},
        }},
        hoverinfo: "skip",
        customdata: [[selectedRecord.idx, selectedRecord.dist_total || 0, ""]],
      }});
    }}
  }}

  return traces;
}}

function render() {{
  const traces = buildTraces(currentStem, currentMethod);
  if (!plotReady) {{
    Plotly.newPlot("main-plot", traces, LAYOUT, CONFIG);
    const el = document.getElementById("main-plot");
    el.on("plotly_click",   onPointClick);
    el.on("plotly_hover",   ev => {{ hoveredRecord = RECORDS[ev.points[0].customdata[0]]; }});
    el.on("plotly_unhover", ()  => {{ hoveredRecord = null; }});
    el.addEventListener("contextmenu", onCtxMenu);
    plotReady = true;
  }} else {{
    Plotly.react("main-plot", traces, LAYOUT, CONFIG);
  }}
  const idx = STEMS.indexOf(currentStem);
  document.getElementById("source-label").textContent = currentStem;
  document.getElementById("source-count").textContent = (idx + 1) + " / " + STEMS.length;
  document.getElementById("source-select").value = idx;
}}

function playRecord(r) {{
  document.getElementById("audio-name").textContent = r.name;
  const pl = document.getElementById("audio-player");
  pl.src = "/audio/" + r.idx + "/";
  pl.play().catch(() => {{}});
  document.getElementById("audio-meta").textContent = r.is_original
    ? "ORIGINAL (origin)"
    : "bend: " + r.bending_type + "  \xb7  dist: " + r.dist_total.toFixed(4);
  const addBtn = document.getElementById("wishlist-add-btn");
  if (r.is_original) {{ addBtn.style.display = "none"; }}
  else {{ addBtn.style.display = "block"; updateAddBtnState(r); }}
  const palette = ["#e53935","#ff9500","#ff6b35","#9c27b0",
                   "#5c6bc0","#0288d1","#00897b","#43a047","#f9a825","#6d4c41","#455a64","#e91e63"];
  ALL_SCALAR_KEYS.forEach((k, i) => {{
    const fill = document.getElementById("sbar-" + k); if (!fill) return;
    fill.style.width      = Math.min(100, Math.max(0, norm(r.scalars[k] || 0, k) * 100)) + "%";
    fill.style.background = palette[i % palette.length];
  }});
  document.querySelectorAll(".sbar-fill").forEach(f => f.style.opacity = "1");
}}

function onPointClick(ev) {{
  const r = RECORDS[ev.points[0].customdata[0]];
  selectedRecord = r;
  playRecord(r);
  requestAnimationFrame(render);
}}

function togglePanel() {{
  const body = document.getElementById("panel-body"), t = document.getElementById("panel-toggle");
  const h = body.style.display === "none";
  body.style.display = h ? "" : "none"; t.textContent = h ? "▾" : "▸";
}}
function switchMethod(m) {{
  currentMethod = m;
  document.querySelectorAll(".method-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("btn-" + m).classList.add("active");
  render();
}}
function switchColor(c)  {{ currentColor = c; render(); }}
function stepSource(dir) {{
  currentStem = STEMS[(STEMS.indexOf(currentStem) + dir + STEMS.length) % STEMS.length];
  render();
}}
function goSource(idx) {{ currentStem = STEMS[parseInt(idx)]; render(); }}

// ---------------------------------------------------------------------------
// Right-click context menu
// ---------------------------------------------------------------------------
function onCtxMenu(ev) {{
  ev.preventDefault();
  const menu = document.getElementById("ctx-menu");
  if (!hoveredRecord || hoveredRecord.is_original) {{ menu.style.display = "none"; return; }}
  ctxRecord = hoveredRecord;   // snapshot before plotly_unhover can clear it
  const toggle = document.getElementById("ctx-wl-toggle");
  if (isWishlisted(ctxRecord)) {{
    toggle.textContent = "&#9829; Remove from Wishlist"; toggle.className = "ctx-item ctx-unwl";
  }} else {{
    toggle.textContent = "&#9829; Add to Wishlist"; toggle.className = "ctx-item ctx-wl";
  }}
  menu.style.left = ev.clientX + "px"; menu.style.top = ev.clientY + "px";
  menu.style.display = "block";
}}
function ctxPlay() {{
  hideCtxMenu();
  if (!ctxRecord) return;
  selectedRecord = ctxRecord; playRecord(ctxRecord); requestAnimationFrame(render);
}}
function ctxToggleWishlist() {{
  hideCtxMenu();
  if (!ctxRecord || ctxRecord.is_original) return;
  const prev = selectedRecord; selectedRecord = ctxRecord;
  toggleWishlistItem(); selectedRecord = prev;
  if (prev) updateAddBtnState(prev);
}}
function hideCtxMenu() {{ document.getElementById("ctx-menu").style.display = "none"; }}
document.addEventListener("click",   hideCtxMenu);
document.addEventListener("keydown", e => {{ if (e.key === "Escape") hideCtxMenu(); }});

// ---------------------------------------------------------------------------
// Wishlist
// ---------------------------------------------------------------------------
function isWishlisted(r) {{
  return (wishlist[r.source_stem] || []).some(i => i.path === r.path);
}}
function updateAddBtnState(r) {{
  const btn = document.getElementById("wishlist-add-btn");
  if (isWishlisted(r)) {{
    btn.textContent = "&#9829; In Wishlist"; btn.classList.add("wishlisted");
  }} else {{
    btn.textContent = "&#9829; Add to Wishlist"; btn.classList.remove("wishlisted");
  }}
}}
function toggleWishlistItem() {{
  if (!selectedRecord || selectedRecord.is_original) return;
  const r = selectedRecord, stem = r.source_stem;
  if (!wishlist[stem]) wishlist[stem] = [];
  const idx = wishlist[stem].findIndex(i => i.path === r.path);
  if (idx >= 0) {{
    wishlist[stem].splice(idx, 1);
    if (!wishlist[stem].length) delete wishlist[stem];
  }} else {{
    wishlist[stem].push({{ name: r.name, path: r.path, source_stem: stem,
                           bending_type: r.bending_type, dist_total: r.dist_total }});
  }}
  updateAddBtnState(r); renderWishlist();
}}
function removeWishlistItem(stem, path) {{
  if (!wishlist[stem]) return;
  wishlist[stem] = wishlist[stem].filter(i => i.path !== path);
  if (!wishlist[stem].length) delete wishlist[stem];
  if (selectedRecord && selectedRecord.path === path) updateAddBtnState(selectedRecord);
  renderWishlist();
}}
function clearWishlist() {{
  Object.keys(wishlist).forEach(k => delete wishlist[k]);
  if (selectedRecord && !selectedRecord.is_original) updateAddBtnState(selectedRecord);
  renderWishlist();
}}
function renderWishlist() {{
  const body  = document.getElementById("wishlist-body");
  const badge = document.getElementById("wl-count-badge");
  const empty = document.getElementById("wishlist-empty");
  const total = Object.values(wishlist).reduce((s, a) => s + a.length, 0);
  badge.textContent = total;
  const groups = Object.keys(wishlist).sort();
  if (!groups.length) {{ body.innerHTML = ""; body.appendChild(empty); empty.style.display = "block"; return; }}
  empty.style.display = "none"; body.innerHTML = "";
  const TC = BTYPE_COLORS;
  groups.forEach(stem => {{
    const items = wishlist[stem]; if (!items || !items.length) return;
    const grp = document.createElement("div"); grp.className = "wl-source-group";
    const title = document.createElement("div"); title.className = "wl-source-title";
    title.textContent = stem + " (" + items.length + ")"; grp.appendChild(title);
    items.forEach(item => {{
      const row = document.createElement("div"); row.className = "wl-item";
      const nameEl = document.createElement("div"); nameEl.className = "wl-item-name";
      nameEl.title = item.path; nameEl.textContent = item.name;
      const typeEl = document.createElement("span"); typeEl.className = "wl-item-type";
      typeEl.textContent = item.bending_type;
      typeEl.style.background = (TC[item.bending_type] || "#888") + "22";
      typeEl.style.color = TC[item.bending_type] || "#555";
      const rmBtn = document.createElement("button"); rmBtn.className = "wl-remove-btn";
      rmBtn.textContent = "×"; rmBtn.title = "Remove";
      const _s = stem, _p = item.path; rmBtn.onclick = () => removeWishlistItem(_s, _p);
      row.appendChild(nameEl); row.appendChild(typeEl); row.appendChild(rmBtn);
      grp.appendChild(row);
    }});
    body.appendChild(grp);
  }});
}}
function toggleWishlistPanel() {{
  const body = document.getElementById("wishlist-body"), t = document.getElementById("wishlist-toggle");
  const h = body.style.display === "none";
  body.style.display = h ? "" : "none"; t.textContent = h ? "&#9662;" : "&#9656;";
}}
function exportWishlistJSON() {{
  const total = Object.values(wishlist).reduce((s, a) => s + a.length, 0);
  if (!total) {{ alert("Wishlist is empty."); return; }}
  const blob = new Blob([JSON.stringify(wishlist, null, 2)], {{type: "application/json"}});
  const url = URL.createObjectURL(blob), a = document.createElement("a");
  a.href = url; a.download = "wishlist.json";
  document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
}}

// ---------------------------------------------------------------------------
// Copy files (server-side)
// ---------------------------------------------------------------------------
function openCopyModal() {{
  const total = Object.values(wishlist).reduce((s, a) => s + a.length, 0);
  if (!total) {{ alert("Wishlist is empty."); return; }}
  document.getElementById("copy-status").textContent = "";
  document.getElementById("copy-modal").classList.add("open");
  document.getElementById("copy-dest").focus();
}}
function closeCopyModal() {{
  document.getElementById("copy-modal").classList.remove("open");
}}
async function doCopyFiles() {{
  const dest = document.getElementById("copy-dest").value.trim();
  if (!dest) {{ alert("Please enter a destination path."); return; }}
  const btn    = document.getElementById("copy-go");
  const status = document.getElementById("copy-status");
  btn.disabled = true; status.textContent = "Copying…";
  try {{
    const resp = await fetch("/api/copy", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ dest, wishlist }}),
    }});
    const data = await resp.json();
    if (data.errors && data.errors.length) {{
      status.style.color = "#e53935";
      status.textContent = data.copied + " copied, " + data.errors.length + " errors: " +
                           data.errors.map(e => e.name + ": " + e.error).join("; ");
    }} else {{
      status.style.color = "#2e7d32";
      status.textContent = "✓ " + data.copied + " file(s) copied to " + dest;
      setTimeout(closeCopyModal, 1500);
    }}
  }} catch(e) {{
    status.style.color = "#e53935"; status.textContent = "Error: " + e.message;
  }}
  btn.disabled = false;
}}
document.getElementById("copy-modal").addEventListener("click", e => {{
  if (e.target === document.getElementById("copy-modal")) closeCopyModal();
}});
document.addEventListener("keydown", e => {{ if (e.key === "w" || e.key === "W") toggleWishlistItem(); }});

renderWishlist();
switchMethod("{default_method}");
</script>
</body>
</html>"""

    return HttpResponse(html, content_type="text/html; charset=utf-8")


@require_GET
def view_audio(request, idx):
    records = APP["records"]
    try:
        rec  = records[int(idx)]
        if not rec["path"]:
            return HttpResponse("No audio for mean reference", status=404)
        path = Path(rec["path"])
        mime = mimetypes.guess_type(str(path))[0] or "audio/wav"
        return FileResponse(open(str(path), "rb"), content_type=mime)
    except (IndexError, FileNotFoundError, TypeError, ValueError) as e:
        return HttpResponse(f"Not found: {e}", status=404)


@csrf_exempt
@require_POST
def view_copy(request):
    try:
        data     = json.loads(request.body)
        dest     = Path(data["dest"])
        wishlist = data["wishlist"]   # {stem: [{name, path, ...}]}
    except (KeyError, json.JSONDecodeError) as e:
        return JsonResponse({"error": str(e)}, status=400)

    copied, errors = 0, []
    for stem, items in wishlist.items():
        dest_dir = dest / stem
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append({"name": stem, "error": f"mkdir failed: {e}"})
            continue
        for item in items:
            try:
                shutil.copy2(item["path"], dest_dir / item["name"])
                copied += 1
            except Exception as e:
                errors.append({"name": item["name"], "error": str(e)})

    return JsonResponse({"copied": copied, "errors": errors})


urlpatterns = [
    path("",               view_index),
    path("audio/<int:idx>/", view_audio),
    path("api/copy",       view_copy),
]

# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--methods",          default=",".join(DEFAULT_METHODS))
    p.add_argument("--src-dirs",         nargs="+", default=DEFAULT_SRC_DIRS)
    p.add_argument("--gen-dirs",         nargs="+", default=DEFAULT_GEN_DIRS)
    p.add_argument("--port",             type=int,  default=DEFAULT_PORT)
    p.add_argument("--duration",         type=float, default=DEFAULT_DURATION)
    p.add_argument("--n-bands",          type=int,   default=DEFAULT_N_BANDS)
    p.add_argument("--mds-dims",         type=int,   default=DEFAULT_MDS_DIMS)
    p.add_argument("--energy-threshold", type=float, default=DEFAULT_ENERGY_THRESH)
    p.add_argument("--transform",        default=DEFAULT_TRANSFORM, choices=["mel", "nsgt"])
    p.add_argument("--cache",            default=None, metavar="PATH",
                   help="dist cache path (default: <first-gen-dir>/dist_cache_*.pkl)")
    p.add_argument("--proj-cache",       default=None, metavar="PATH",
                   help="projection cache path (default: <first-gen-dir>/proj_cache_*.pkl)")
    p.add_argument("--recompute",        action="store_true",
                   help="ignore existing caches and recompute everything")
    p.add_argument("--limit",            type=int,   default=None)
    p.add_argument("--sources",          nargs="+",  default=None, metavar="STEM")
    p.add_argument("--manifest",         nargs="+", default=None, metavar="PATH",
                   help="manifest.json of torchbend generations: read the files, "
                        "their inputs and bending values from it instead of "
                        "searching --gen-dirs")
    p.add_argument("--bended-regexp",    default=None, metavar="TEMPLATE",
                   help="filename template for extracting source name, e.g. "
                        "'{originalName}_{idx}' or '{originalName}_{part}_{idx}'. "
                        "{originalName} is greedy (may include _); all other "
                        "placeholders match [^_]+ (no underscores).")
    return p.parse_args()


def main():
    args    = parse_args()
    methods = [m.strip() for m in args.methods.split(",")]
    missing = [m for m in methods if m not in available_methods(methods)]
    if missing:
        print(f"  Skipping {', '.join(missing)}: not installed.")
        methods = available_methods(methods) or ["pca"]
    if args.transform == "nsgt" and not _nsgt_available():
        print("  nsgt is not installed -- using the mel transform (pip install nsgt).")
        args.transform = "mel"

    param_names = {}
    if args.manifest:
        originals, bended, stem_to_path, param_names = load_manifest(args.manifest)
        if args.sources:
            keep = set(args.sources)
            bended = [b for b in bended if b["source_stem"] in keep]
        first_gen = Path(args.manifest[0]).expanduser().parent
    else:
        bended_pattern = _compile_bended_pattern(args.bended_regexp)
        if bended_pattern:
            print(f"  Using bended-regexp: {args.bended_regexp!r} → {bended_pattern.pattern}")
        originals, bended, stem_to_path = discover_files(
            args.src_dirs, args.gen_dirs,
            sources_filter=args.sources,
            bended_pattern=bended_pattern,
        )
        first_gen = next(
            (Path(d).expanduser() for d in args.gen_dirs if Path(d).expanduser().exists()),
            Path(".")
        )
    if not originals and not bended:
        print("No audio files found."); sys.exit(1)
    if not bended:
        print("No bended files found."); sys.exit(1)
    suffix = f"{args.n_bands}bands_{int(args.duration)}s"
    cache_path      = args.cache      or str(first_gen / f"dist_cache_{suffix}.pkl")
    proj_cache_path = args.proj_cache or str(first_gen / f"proj_cache_{suffix}.pkl")

    if args.limit:
        import random; random.seed(42)
        bended = random.sample(bended, min(args.limit, len(bended)))
        print(f"  Limited to {len(bended)} bended files.")

    print("\n[1/2] Computing distances + projections...")
    records, source_data = build_source_data(
        originals, bended, stem_to_path,
        sr=DEFAULT_SR, duration=args.duration,
        n_bands=args.n_bands, mds_dims=args.mds_dims,
        energy_threshold=args.energy_threshold,
        methods=methods, cache_path=cache_path,
        transform=args.transform,
        proj_cache_path=proj_cache_path,
        recompute=args.recompute,
    )

    APP["records"]     = records
    APP["source_data"] = source_data
    APP["n_bands"]     = args.n_bands
    APP["methods"]     = methods
    APP["param_names"] = param_names

    serve(args.port)


def serve(port):
    """Serve what ``APP`` holds on ``port`` (blocks)."""
    _configure_django()
    print(f"\n[2/2] Starting server on http://localhost:{port}/")
    print("      Press Ctrl-C to stop.\n")

    from django.core.management import execute_from_command_line
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "")
    execute_from_command_line(["manage.py", "runserver", f"0.0.0.0:{port}", "--noreload"])


if __name__ == "__main__":
    main()
