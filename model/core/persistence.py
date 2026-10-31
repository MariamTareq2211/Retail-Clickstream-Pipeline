"""Artifact input/output helpers.

All writes are atomic: data is written to a sibling ``*.tmp`` file and moved
into place with ``os.replace``. A crashed run can therefore never leave a
truncated parquet, npz or joblib file that a later stage would happily load.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import scipy.sparse as sp

from model.core.paths import resolve_path


def _atomic_target(path: str | Path, suffix: str) -> tuple[Path, Path]:
    """Return ``(final_path, temporary_path)`` for an atomic write."""
    final = resolve_path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(final.name + suffix)
    return final, temporary


def save_json(payload: Any, path: str | Path, indent: int = 2) -> Path:
    """Write JSON atomically and return the final path."""
    final, temporary = _atomic_target(path, ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, final)
    return final


def load_json(path: str | Path) -> Any:
    """Read a JSON document."""
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def save_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to parquet atomically and return the final path."""
    final, temporary = _atomic_target(path, ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, final)
    return final


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file."""
    return pd.read_parquet(resolve_path(path))


def save_joblib(payload: Any, path: str | Path) -> Path:
    """Serialise a Python object with joblib atomically."""
    final, temporary = _atomic_target(path, ".tmp")
    joblib.dump(payload, temporary)
    os.replace(temporary, final)
    return final


def load_joblib(path: str | Path) -> Any:
    """Load a joblib artifact."""
    return joblib.load(resolve_path(path))


def save_sparse(matrix: sp.spmatrix, path: str | Path) -> Path:
    """Persist a SciPy sparse matrix in ``.npz`` format atomically."""
    final = resolve_path(path)
    if final.suffix.lower() != ".npz":
        raise ValueError(f"Sparse matrices must be saved as .npz, got {final.name!r}.")
    final.parent.mkdir(parents=True, exist_ok=True)

    # scipy appends '.npz' when the suffix is missing, so the temporary file
    # keeps a real .npz suffix and is only renamed afterwards.
    temporary = final.with_name(final.stem + ".tmp.npz")
    sp.save_npz(temporary, matrix)
    os.replace(temporary, final)
    return final


def load_sparse(path: str | Path) -> sp.spmatrix:
    """Load a sparse matrix written by :func:`save_sparse`."""
    return sp.load_npz(resolve_path(path))


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum of a file, used in the artifact manifest."""
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
