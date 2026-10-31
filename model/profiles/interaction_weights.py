"""Interaction weighting for user-profile construction (Plan.md Module 2.1).

The three benchmarked models share one code path and differ only in three
booleans: whether event types carry a weight, whether the repeat count is
applied, and whether the interaction is exponentially decayed in time. Keeping
the differences as data (``ModelSpec``) instead of three copied functions means
the comparison is guaranteed to isolate exactly one concept at a time.

Every row of the intermediate table answers one question: *how much does user
``u`` value item ``i``?* The table is then folded into a sparse user x item
matrix ``W`` so that the profile matrix is a single sparse product ``W @ X``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from model.config.config_schema import PipelineConfig
from model.core.logger import get_logger
from model.profiles.time_decay import decay_weights, resolve_reference_ns, to_utc_ns

_LOGGER = get_logger(__name__)

COUNT_TRANSFORMS = ("raw", "log1p", "capped", "none")


@dataclass(frozen=True)
class ProfileParams:
    """The complete, explicit set of numbers used to weight one interaction.

    A frozen parameter object is what makes Optuna able to rebuild profiles
    without mutating the validated configuration: a trial produces new
    ``ProfileParams`` and the rest of the code stays identical.
    """

    event_weights: Mapping[int, float]
    count_transform: str
    count_cap: int
    decay_enabled: bool
    decay_reference: str
    decay_rate_per_hour: float

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "ProfileParams":
        """Build the parameters declared in ``config.yaml``."""
        return cls(
            event_weights=dict(config.profiles.weights),
            count_transform=config.dedup.count_transform,
            count_cap=config.dedup.count_cap,
            decay_enabled=config.profiles.decay.enabled,
            decay_reference=config.profiles.decay.reference,
            decay_rate_per_hour=config.profiles.decay.rate_per_hour,
        )

    def with_overrides(self, **overrides: object) -> "ProfileParams":
        """Return a copy with selected fields replaced (used by the tuner)."""
        from dataclasses import replace

        return replace(self, **overrides)  # type: ignore[arg-type]


def apply_count_transform(counts: np.ndarray, transform: str, cap: int) -> np.ndarray:
    """Map raw repeat counts onto the multiplier used in the profile weight.

    Args:
        counts: Positive integer repeat counts, one per interaction row.
        transform: One of ``raw``, ``log1p``, ``capped`` or ``none``.
        cap: Upper bound used by ``capped``.

    Returns:
        Float multiplier aligned with ``counts``.

    Raises:
        ValueError: If the transform name is unknown.
    """
    values = np.asarray(counts, dtype=np.float64)

    if transform == "raw":
        return values
    if transform == "log1p":
        return np.log1p(values)
    if transform == "capped":
        return np.minimum(values, float(cap))
    if transform == "none":
        return np.ones_like(values)
    raise ValueError(
        f"Unknown count transform {transform!r}; expected one of {COUNT_TRANSFORMS}."
    )


def _resolve_event_weights(
    event_types: np.ndarray,
    mapping: Mapping[int, float],
) -> np.ndarray:
    """Translate event-type ids into weights, failing loudly on unknown ids.

    A missing event type would otherwise silently default to zero and erase a
    whole behaviour from every profile, so the contract is enforced here.
    """
    unique = np.unique(event_types)
    missing = [int(value) for value in unique if int(value) not in mapping]
    if missing:
        raise ValueError(
            f"profiles.weights does not define event type(s) {missing}. "
            f"Configured types: {sorted(int(key) for key in mapping)}."
        )
    return pd.Series(event_types).map(mapping).to_numpy(dtype=np.float64)


def build_interaction_table(
    events: pd.DataFrame,
    params: ProfileParams,
    item_index: pd.DataFrame,
    timestamp_column: str,
    *,
    use_event_weights: bool,
    use_counts: bool,
    use_decay: bool,
    global_reference_ns: int | None = None,
) -> pd.DataFrame:
    """Collapse raw interactions into one weighted row per ``(user, item)``.

    Args:
        events: Rows from the training split (or a user subset of it).
        params: Weighting parameters for the current model / trial.
        item_index: Module 1 mapping with ``product_id`` and ``item_row``.
        timestamp_column: Name of the event timestamp column.
        use_event_weights: Multiply by the per-event-type weight.
        use_counts: Multiply by the repeat-count factor.
        use_decay: Multiply by the exponential time-decay factor.
        global_reference_ns: Reference instant for ``reference="global"``.
            Defaults to the latest timestamp in ``events``.

    Returns:
        DataFrame with ``user_key``, ``item_row`` and the aggregated ``weight``.

    Raises:
        ValueError: If an event type has no configured weight, or if the input
            is empty.
    """
    if events.empty:
        raise ValueError("Cannot build an interaction table from an empty frame.")

    positions = pd.Index(item_index["product_id"].to_numpy())
    item_rows = positions.get_indexer(events["product_id"].to_numpy())
    keep = item_rows >= 0

    unmatched = int((~keep).sum())
    if unmatched:
        _LOGGER.warning(
            "%s interactions reference products outside the catalogue and were skipped.",
            f"{unmatched:,}",
        )

    user_keys = events["user_key"].to_numpy()[keep]
    item_rows = item_rows[keep]

    if use_event_weights:
        weights = _resolve_event_weights(
            events["event_type_key"].to_numpy()[keep], params.event_weights
        )
    else:
        # Model 1: every distinct (user, item) pair contributes exactly one unit,
        # i.e. an unweighted centroid over the interacted items.
        weights = np.ones(len(item_rows), dtype=np.float64)

    if use_counts:
        weights = weights * apply_count_transform(
            events["interaction_count"].to_numpy()[keep],
            params.count_transform,
            params.count_cap,
        )

    if use_decay:
        timestamps = to_utc_ns(events[timestamp_column])[keep]
        reference = resolve_reference_ns(
            timestamps,
            user_keys,
            mode=params.decay_reference,
            global_reference_ns=(
                int(timestamps.max()) if global_reference_ns is None else global_reference_ns
            ),
        )
        weights = weights * decay_weights(
            timestamps, reference, params.decay_rate_per_hour
        )

    table = pd.DataFrame(
        {"user_key": user_keys, "item_row": item_rows, "weight": weights}
    )

    if not use_event_weights and not use_counts and not use_decay:
        # Distinct pairs only: repeat events must not turn Model 1 into a
        # frequency-weighted model through the back door.
        table = table.drop_duplicates(subset=["user_key", "item_row"], keep="first")

    aggregated = (
        table.groupby(["user_key", "item_row"], sort=False, observed=True)["weight"]
        .sum()
        .reset_index()
    )

    if not np.isfinite(aggregated["weight"].to_numpy()).all():
        raise ValueError("Interaction weights contain non-finite values.")

    _LOGGER.debug(
        "Interaction table: %s rows -> %s (user, item) pairs",
        f"{len(events):,}",
        f"{len(aggregated):,}",
    )
    return aggregated


def build_weight_matrix(
    table: pd.DataFrame,
    user_keys: np.ndarray,
    n_items: int,
) -> sp.csr_matrix:
    """Fold the interaction table into a sparse ``users x items`` matrix.

    Args:
        table: Output of :func:`build_interaction_table`.
        user_keys: Row order of the matrix (ascending user ids).
        n_items: Number of catalogue rows in the item matrix.

    Returns:
        CSR matrix of shape ``(len(user_keys), n_items)``.

    Raises:
        ValueError: If the table references a user outside ``user_keys``.
    """
    positions = pd.Series(np.arange(len(user_keys)), index=np.asarray(user_keys))
    rows = positions.reindex(table["user_key"].to_numpy()).to_numpy()

    if np.isnan(rows).any():
        raise ValueError("The interaction table contains users outside the profile index.")

    matrix = sp.csr_matrix(
        (
            table["weight"].to_numpy(dtype=np.float64),
            (rows.astype(np.int64), table["item_row"].to_numpy(dtype=np.int64)),
        ),
        shape=(len(user_keys), n_items),
    )
    matrix.sum_duplicates()
    return matrix
