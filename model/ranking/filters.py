"""Candidate eligibility and ground-truth construction for ranking.

Business rules from Plan.md Module 3.1 are implemented here, separately from the
scoring loop:

* never recommend an item the user already interacted with in training;
* never recommend an item whose catalogue vector carries no signal;
* evaluate only on interactions the user could not already have seen, otherwise
  a repeated purchase would be counted as a miss by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from model.core.logger import get_logger

_LOGGER = get_logger(__name__)


def item_signal_mask(item_matrix: sp.spmatrix) -> np.ndarray:
    """Return a boolean mask of catalogue rows that carry at least one feature.

    A row with no text token and a zero price feature has an all-zero vector;
    its cosine similarity is zero for every user, so it must never occupy a
    recommendation slot.

    Args:
        item_matrix: Catalogue matrix in CSR/CSC form.

    Returns:
        Boolean array of length ``n_items``.
    """
    matrix = sp.csr_matrix(item_matrix)
    return np.diff(matrix.indptr) > 0


def binarize_interactions(interactions: sp.spmatrix) -> sp.csr_matrix:
    """Turn the weighted user x item matrix into a binary "seen" mask.

    Args:
        interactions: Sparse weight matrix.

    Returns:
        CSR matrix with every stored value equal to one.
    """
    mask = sp.csr_matrix(interactions, copy=True)
    mask.data = np.ones_like(mask.data)
    return mask


def build_ground_truth(
    events: pd.DataFrame,
    item_index: pd.DataFrame,
    user_keys: np.ndarray,
) -> dict[int, np.ndarray]:
    """Collect the distinct relevant items per user for an evaluation split.

    Args:
        events: Validation or test interactions.
        item_index: ``product_id`` -> ``item_row`` mapping from Module 1.
        user_keys: Users that will actually be scored.

    Returns:
        Mapping ``user_key -> sorted unique item rows``. Users without a
        catalogue item are omitted.
    """
    positions = pd.Index(item_index["product_id"].to_numpy())
    frame = events.loc[events["user_key"].isin(user_keys), ["user_key", "product_id"]].copy()
    frame["item_row"] = positions.get_indexer(frame["product_id"].to_numpy())
    frame = frame.loc[frame["item_row"] >= 0]

    grouped = frame.groupby("user_key", observed=True, sort=False)["item_row"].agg(
        lambda values: np.unique(values.to_numpy(dtype=np.int64))
    )

    truth = {int(key): value for key, value in grouped.items()}
    _LOGGER.info(
        "Ground truth: %s users, %.2f relevant items per user on average",
        f"{len(truth):,}",
        float(np.mean([len(value) for value in truth.values()])) if truth else 0.0,
    )
    return truth


def exclude_seen_from_truth(
    truth: dict[int, np.ndarray],
    seen_matrix: sp.spmatrix,
    user_keys: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    """Remove training interactions from the relevance labels.

    An item that the user already interacted with in training is excluded from
    the candidate list, so keeping it as a relevant test label would guarantee a
    miss. Filtering both sides keeps the metric definition consistent.

    Args:
        truth: Mapping produced by :func:`build_ground_truth`.
        seen_matrix: Binary or weighted user x item matrix for ``user_keys``.
        user_keys: Row order of ``seen_matrix``.

    Returns:
        Tuple of the filtered truth and diagnostics describing how many labels
        and users were removed.
    """
    seen = sp.csr_matrix(seen_matrix)
    positions = {int(key): index for index, key in enumerate(np.asarray(user_keys))}

    filtered: dict[int, np.ndarray] = {}
    removed_labels = 0
    removed_users = 0

    for key, items in truth.items():
        index = positions.get(key)
        if index is None:
            continue

        start, end = seen.indptr[index], seen.indptr[index + 1]
        seen_items = seen.indices[start:end]

        remaining = items
        if seen_items.size:
            remaining = np.setdiff1d(items, seen_items, assume_unique=True)

        removed_labels += len(items) - len(remaining)
        if remaining.size:
            filtered[key] = remaining
        else:
            removed_users += 1

    diagnostics = {
        "labels_before": int(sum(len(value) for value in truth.values())),
        "labels_after": int(sum(len(value) for value in filtered.values())),
        "labels_removed_as_seen": int(removed_labels),
        "users_before": int(len(truth)),
        "users_after": int(len(filtered)),
        "users_without_unseen_labels": int(removed_users),
    }

    _LOGGER.info(
        "Seen-item filter: %s users -> %s | %s labels removed (already interacted in train)",
        f"{diagnostics['users_before']:,}",
        f"{diagnostics['users_after']:,}",
        f"{diagnostics['labels_removed_as_seen']:,}",
    )
    return filtered, diagnostics
