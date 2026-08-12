"""Corpus storage paths — one place that knows where things live, local or in GCS.

The corpus bucket is `gs://ditloracle-corpus` (us-central1, in the GPU project, colocated with the
A100 quota so training reads incur no cross-region egress). Everything addresses data through this
module so a path never gets hardcoded into a script that later runs somewhere else.

Remote access uses `fsspec`/`gcsfs` when a `gs://` root is in play; a local root needs no extra deps,
which keeps laptop work (planning, tests) dependency-free. Set the root with the DITLORACLE_ROOT env
var, e.g. `export DITLORACLE_ROOT=gs://ditloracle-corpus`.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_BUCKET = "gs://ditloracle-corpus"
ENV_ROOT = "DITLORACLE_ROOT"

# the stable tree laid out in the bucket
SUBDIRS = {
    "organism_weights": "organisms/weights",
    "organism_imgsets": "organisms/imgsets",
    "organism_samples": "organisms/samples",
    "wild_weights": "wild/weights",
    "wild_images": "wild/images",
    "reader_checkpoints": "reader/checkpoints",
    "results": "results",
}


def root() -> str:
    """Corpus root: $DITLORACLE_ROOT, else a local ./assets tree (laptop-friendly default)."""
    return os.environ.get(ENV_ROOT) or "assets"


def is_remote(path: str | None = None) -> bool:
    p = path if path is not None else root()
    return "://" in p


def join(*parts: str) -> str:
    """Join path segments under the corpus root, correct for both gs:// and local roots."""
    base = root().rstrip("/")
    tail = "/".join(str(p).strip("/") for p in parts if str(p) != "")
    return f"{base}/{tail}" if tail else base


def path_for(kind: str, *parts: str) -> str:
    """Path inside a named subtree, e.g. path_for('organism_weights', 'cap__x.safetensors')."""
    if kind not in SUBDIRS:
        raise ValueError(f"unknown storage kind {kind!r}; expected one of {sorted(SUBDIRS)}")
    return join(SUBDIRS[kind], *parts)


def get_fs(path: str | None = None):
    """An fsspec filesystem for `path` (imported lazily so local work needs no gcsfs)."""
    p = path if path is not None else root()
    if not is_remote(p):
        return None
    try:
        import fsspec
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "remote corpus root needs fsspec+gcsfs: pip install 'ditloracle[cloud]'"
        ) from e
    return fsspec.filesystem("gcs")


def ensure_local_dir(path: str) -> str:
    """Create a local directory (no-op for remote paths, where prefixes are implicit)."""
    if not is_remote(path):
        Path(path).mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: str) -> str:
    if is_remote(path):
        import fsspec
        with fsspec.open(path, "rt") as fh:
            return fh.read()
    return Path(path).read_text()


def write_text(path: str, text: str) -> str:
    if is_remote(path):
        import fsspec
        with fsspec.open(path, "wt") as fh:
            fh.write(text)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    return path
