"""Sparse fusion of the text and price blocks (Plan.md 1.8).

The catalogue matrix has one row per product and ``vocabulary_size + 1`` columns:
the TF-IDF block followed by a single scaled-price column. Fusing it with SciPy's
``hstack`` keeps the matrix sparse - densifying 117k x ~3k would already cost
~2.8 GB, and the same structure has to survive into Modules 2 and 3.

Rows are L2-normalised so that cosine similarity is a plain dot product and so
that the price weight expresses a *share of the row norm* rather than an absolute
magnitude. A product with no usable metadata keeps a non-zero vector through the
price dimension; a product with neither stays all-zero and is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from model.config.config_schema import Module1Config
from model.core.logger import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class ItemMatrixArtifacts:
    """Fused, normalised item matrix plus sparsity diagnostics."""

    matrix: sp.csr_matrix
    price_block: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)


def fuse_features(
    text_matrix: sp.spmatrix,
    price_vector: np.ndarray,
    config: Module1Config,
) -> ItemMatrixArtifacts:
    """Concatenate the TF-IDF and price blocks into one CSR matrix.

    Args:
        text_matrix: Sparse TF-IDF matrix with one row per catalogue product.
        price_vector: Scaled prices aligned with the rows of ``text_matrix``.
        config: Validated Module 1 configuration (``fusion.price_weight`` and
            ``fusion.normalize_rows``).

    Returns:
        ``ItemMatrixArtifacts`` holding the fused matrix and its diagnostics.

    Raises:
        ValueError: If the two blocks do not describe the same number of products.
    """
    text_matrix = sp.csr_matrix(text_matrix)
    price_vector = np.asarray(price_vector, dtype=np.float64).ravel()

    if text_matrix.shape[0] != price_vector.shape[0]:
        raise ValueError(
            "Text matrix and price vector describe different numbers of products: "
            f"{text_matrix.shape[0]} vs {price_vector.shape[0]}."
        )

    # The price block is weighted before concatenation; because rows are
    # normalised afterwards, the weight controls the relative share of price.
    weighted_price = price_vector * float(config.fusion.price_weight)
    price_block = sp.csr_matrix(weighted_price.reshape(-1, 1))

    fused = sp.hstack([text_matrix, price_block], format="csr")

    if config.fusion.normalize_rows:
        fused = normalize(fused, norm="l2", axis=1, copy=False).tocsr()

    row_nnz = np.diff(fused.indptr)
    zero_rows = int((row_nnz == 0).sum())
    density = fused.nnz / (fused.shape[0] * fused.shape[1]) if fused.shape[1] else 0.0

    # Share of each row's squared norm contributed by the price dimension. With
    # unit-norm rows this is directly interpretable as "how much price matters".
    price_share = float(np.mean(np.square(fused[:, -1].toarray().ravel())))

    stats: dict[str, Any] = {
        "items": int(fused.shape[0]),
        "features": int(fused.shape[1]),
        "text_features": int(text_matrix.shape[1]),
        "nnz": int(fused.nnz),
        "density": round(float(density), 8),
        "zero_rows": zero_rows,
        "zero_rows_pct": round(zero_rows / fused.shape[0] * 100, 4),
        "mean_nnz_per_row": round(float(row_nnz.mean()), 4),
        "mean_price_share": round(price_share, 6),
        "price_weight": float(config.fusion.price_weight),
        "row_normalised": bool(config.fusion.normalize_rows),
    }

    _LOGGER.info(
        "Item matrix fused: %s x %s | nnz=%s | density=%.6f | zero rows=%s",
        f"{fused.shape[0]:,}",
        f"{fused.shape[1]:,}",
        f"{fused.nnz:,}",
        density,
        f"{zero_rows:,}",
    )
    if zero_rows:
        _LOGGER.warning(
            "%s products have neither text nor price signal and can never be recommended.",
            f"{zero_rows:,}",
        )

    return ItemMatrixArtifacts(matrix=fused, price_block=price_vector, stats=stats)
