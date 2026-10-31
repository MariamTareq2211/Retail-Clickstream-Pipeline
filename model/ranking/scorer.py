"""Cosine scoring and top-K extraction on sparse matrices (Plan.md Module 3.1).

Both matrices are L2-normalised, so cosine similarity is a plain dot product:

    scores = P @ X.T

``P`` has one row per user and ``X`` one row per catalogue item. The full score
matrix is ``users x items`` and cannot be materialised for 700k users, so users
are scored in blocks. Within a block the score matrix is dense (128 x 117k is
about 120 MB) and every business-rule mask is applied before a per-row
``argpartition`` selects the top K.

Rows are processed individually when selecting the top K: a full
``argpartition(..., axis=1)`` would allocate a second 120 MB integer array per
block for no measurable speed gain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from model.core.logger import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class RankingResult:
    """Top-K item rows and scores for every scored user."""

    item_rows: np.ndarray
    scores: np.ndarray
    k: int

    @property
    def users(self) -> int:
        """Number of scored users."""
        return int(self.item_rows.shape[0])

    def to_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(user_row, item_row)`` index arrays for the valid entries."""
        rows, columns = np.nonzero(self.item_rows >= 0)
        return rows, self.item_rows[rows, columns]


def rank_top_k(
    user_matrix: sp.spmatrix,
    item_matrix: sp.spmatrix,
    k: int,
    seen_matrix: sp.spmatrix | None = None,
    allowed_items: np.ndarray | None = None,
    batch_size: int = 128,
) -> RankingResult:
    """Score every user against the catalogue and keep the top K items.

    Args:
        user_matrix: L2-normalised user profiles, one row per scored user.
        item_matrix: L2-normalised catalogue matrix.
        k: Number of recommendations per user.
        seen_matrix: Optional user x item mask of already-interacted items.
            Rows must align with ``user_matrix``.
        allowed_items: Optional boolean mask of items that may be recommended.
        batch_size: Users per dense score block.

    Returns:
        ``RankingResult`` with ``item_rows`` of shape ``(users, k)``; unreachable
        slots are filled with ``-1`` and their score with ``-inf``.

    Raises:
        ValueError: If the inputs disagree on shape or ``k`` is not positive.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if user_matrix.shape[1] != item_matrix.shape[1]:
        raise ValueError(
            "User profiles and item matrix have different feature dimensions: "
            f"{user_matrix.shape[1]} vs {item_matrix.shape[1]}."
        )
    if seen_matrix is not None and seen_matrix.shape[0] != user_matrix.shape[0]:
        raise ValueError(
            "seen_matrix rows must match user_matrix rows: "
            f"{seen_matrix.shape[0]} vs {user_matrix.shape[0]}."
        )

    items = sp.csr_matrix(item_matrix)
    users = sp.csr_matrix(user_matrix)
    seen = sp.csr_matrix(seen_matrix) if seen_matrix is not None else None

    n_users = users.shape[0]
    n_items = items.shape[0]
    k_effective = min(k, n_items)

    if allowed_items is None:
        allowed = np.ones(n_items, dtype=bool)
    else:
        allowed = np.asarray(allowed_items, dtype=bool)
        if allowed.shape[0] != n_items:
            raise ValueError(
                f"allowed_items must have length {n_items}, got {allowed.shape[0]}."
            )

    disallowed = ~allowed

    item_transposed = items.T.tocsr()

    item_rows = np.full((n_users, k_effective), -1, dtype=np.int64)
    scores_out = np.full((n_users, k_effective), -np.inf, dtype=np.float64)

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        block = users[start:end]

        dense = (block @ item_transposed).toarray()

        for local in range(dense.shape[0]):
            row = dense[local]

            if seen is not None:
                global_row = start + local
                row[seen.indices[seen.indptr[global_row] : seen.indptr[global_row + 1]]] = -np.inf

            if disallowed.any():
                row[disallowed] = -np.inf

            candidates = np.argpartition(row, -k_effective)[-k_effective:]
            candidate_scores = row[candidates]

            order = np.argsort(-candidate_scores, kind="stable")
            chosen = candidates[order]
            chosen_scores = candidate_scores[order]

            valid = np.isfinite(chosen_scores)
            count = int(valid.sum())

            item_rows[start + local, :count] = chosen[valid]
            scores_out[start + local, :count] = chosen_scores[valid]

    filled = float(np.mean(item_rows >= 0)) if n_users else 0.0
    _LOGGER.debug(
        "Ranked %s users x %s items with k=%s | %.2f%% of slots filled",
        f"{n_users:,}",
        f"{n_items:,}",
        k_effective,
        filled * 100,
    )

    return RankingResult(item_rows=item_rows, scores=scores_out, k=k_effective)
