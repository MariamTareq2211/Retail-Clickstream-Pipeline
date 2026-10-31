"""User preference vectors built from the training split (Plan.md Module 2.1).

Construction
------------
For every user we aggregate the catalogue vectors of the items they interacted
with in the training window, using the weights produced by
:mod:`model.profiles.interaction_weights`::

    P = W @ X

``W`` is a sparse users x items weight matrix and ``X`` is the Module 1 item
matrix. Because ``X`` rows are already L2-normalised, normalising each row of
``P`` turns a plain dot product between a user and an item into the cosine
similarity required by Module 3.

Why rows are normalised rather than divided by the total weight
--------------------------------------------------------------
A centroid is the weighted *mean* of item vectors, but dividing by the sum of
weights and then L2-normalising is identical in direction to summing and then
normalising. The mean is therefore implicit, which lets the whole computation
stay a single sparse matrix product.

Leakage
-------
Only training rows are passed in. No validation or test interaction can
influence a profile; the caller is responsible for the split, and this module
asserts that the frames it receives are non-empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from model.config.config_schema import PipelineConfig
from model.core.logger import get_logger
from model.models.base import ModelSpec
from model.profiles.interaction_weights import ProfileParams, build_interaction_table, build_weight_matrix
from model.profiles.time_decay import to_utc_ns

_LOGGER = get_logger(__name__)


@dataclass
class ProfileArtifacts:
    """Normalised user profiles plus the interaction matrix they came from."""

    matrix: sp.csr_matrix
    interaction_matrix: sp.csr_matrix
    user_keys: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)


def eligible_user_keys(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    config: PipelineConfig,
) -> np.ndarray:
    """Return users that can be scored fairly on the evaluation split.

    A user is eligible when they have at least ``evaluation.min_train_interactions``
    training rows and at least ``evaluation.min_test_interactions`` rows in the
    evaluation split. Without the training requirement there is no profile to
    score; without the evaluation requirement there is no ground truth.

    Args:
        train: Training interactions.
        evaluation: Validation or test interactions.
        config: Validated pipeline configuration.

    Returns:
        Sorted user identifiers as an ``int64`` array.
    """
    train_counts = train.groupby("user_key", observed=True).size()
    eval_counts = evaluation.groupby("user_key", observed=True).size()

    train_ok = set(train_counts.index[train_counts >= config.evaluation.min_train_interactions])
    eval_ok = set(eval_counts.index[eval_counts >= config.evaluation.min_test_interactions])

    keys = np.array(sorted(train_ok & eval_ok), dtype=np.int64)
    _LOGGER.info(
        "Eligible users: %s (train >= %s, evaluation >= %s) out of %s evaluation users",
        f"{len(keys):,}",
        config.evaluation.min_train_interactions,
        config.evaluation.min_test_interactions,
        f"{len(eval_counts):,}",
    )
    return keys


def sample_user_keys(
    user_keys: np.ndarray,
    max_users: int,
    seed: int,
) -> np.ndarray:
    """Take a deterministic sample of user identifiers.

    Args:
        user_keys: Sorted eligible identifiers.
        max_users: Maximum sample size (the full set is returned when smaller).
        seed: Seed for ``numpy.random.default_rng``.

    Returns:
        Sorted sample, so downstream row order is reproducible.
    """
    if len(user_keys) <= max_users:
        return np.sort(user_keys)

    generator = np.random.default_rng(seed)
    chosen = generator.choice(len(user_keys), size=int(max_users), replace=False)
    return np.sort(user_keys[np.sort(chosen)])


def build_profiles(
    events: pd.DataFrame,
    item_matrix: sp.spmatrix,
    item_index: pd.DataFrame,
    model: ModelSpec,
    params: ProfileParams,
    config: PipelineConfig,
    user_keys: np.ndarray | None = None,
    global_reference_ns: int | None = None,
) -> ProfileArtifacts:
    """Build L2-normalised user profiles for one model configuration.

    Args:
        events: Training interactions (already restricted to ``user_keys`` when
            a sample is used).
        item_matrix: Module 1 catalogue matrix, rows aligned with ``item_index``.
        item_index: Mapping between ``product_id`` and matrix row.
        model: Which of the three progressive models to build.
        params: Event weights, count transform and decay settings.
        config: Validated pipeline configuration.
        user_keys: Explicit user row order. Defaults to the sorted users present
            in ``events``.
        global_reference_ns: Reference instant for global decay. Defaults to the
            latest training timestamp.

    Returns:
        ``ProfileArtifacts`` with the normalised profile matrix, the sparse
        weight matrix (reused later as the "already seen" mask) and diagnostics.

    Raises:
        ValueError: If no user remains after filtering or the item matrix is empty.
    """
    if events.empty:
        raise ValueError(f"Cannot build profiles for {model.key}: training frame is empty.")
    if item_matrix.shape[0] != len(item_index):
        raise ValueError(
            "item_matrix rows and item_index length disagree: "
            f"{item_matrix.shape[0]} vs {len(item_index)}."
        )

    timestamp_column = config.data.timestamp_column

    if user_keys is None:
        user_keys = np.sort(events["user_key"].unique())
        subset = events
    else:
        user_keys = np.asarray(user_keys, dtype=np.int64)
        subset = events[events["user_key"].isin(user_keys)]

    if subset.empty:
        raise ValueError(f"No training interactions remain for {model.key} after sampling.")

    if global_reference_ns is None:
        global_reference_ns = int(to_utc_ns(subset[timestamp_column]).max())

    table = build_interaction_table(
        subset,
        params,
        item_index,
        timestamp_column,
        use_event_weights=model.use_event_weights,
        use_counts=model.use_counts,
        use_decay=model.use_decay,
        global_reference_ns=global_reference_ns,
    )

    interactions = build_weight_matrix(table, user_keys, item_matrix.shape[0])

    profiles = (interactions @ sp.csr_matrix(item_matrix)).tocsr()
    profiles = normalize(profiles, norm="l2", axis=1, copy=False).tocsr()

    row_nnz = np.diff(profiles.indptr)
    empty_rows = int((row_nnz == 0).sum())
    total_weight = np.asarray(interactions.sum(axis=1)).ravel()

    stats: dict[str, Any] = {
        "model": model.key,
        "model_label": model.label,
        "users": int(profiles.shape[0]),
        "features": int(profiles.shape[1]),
        "interaction_pairs": int(interactions.nnz),
        "profile_nnz": int(profiles.nnz),
        "profile_density": round(
            float(profiles.nnz / (profiles.shape[0] * profiles.shape[1])), 10
        ),
        "mean_items_per_user": round(float(interactions.getnnz(axis=1).mean()), 4),
        "mean_profile_nnz_per_user": round(float(row_nnz.mean()), 4),
        "empty_profiles": empty_rows,
        "mean_total_weight": round(float(total_weight.mean()), 6),
        "max_total_weight": round(float(total_weight.max()), 6),
    }

    _LOGGER.info(
        "%s | users=%s | pairs=%s | profile nnz=%s | empty=%s | mean items/user=%.2f",
        model.label,
        f"{profiles.shape[0]:,}",
        f"{interactions.nnz:,}",
        f"{profiles.nnz:,}",
        f"{empty_rows:,}",
        stats["mean_items_per_user"],
    )

    if empty_rows:
        _LOGGER.warning(
            "%s produced %s empty profiles; those users cannot be scored.",
            model.label,
            f"{empty_rows:,}",
        )

    return ProfileArtifacts(
        matrix=profiles,
        interaction_matrix=interactions,
        user_keys=user_keys,
        stats=stats,
    )
