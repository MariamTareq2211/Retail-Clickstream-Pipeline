"""Diversity and novelty diagnostics for recommendation lists (Plan.md post-evaluation).

Accuracy metrics alone reward a system that recommends the same few popular
items to everyone. The blueprint therefore asks for diversity and novelty:

``intra-list diversity``
    ``1 - mean pairwise cosine similarity`` inside a user's list. ``0`` means
    every recommended item is identical in feature space, ``1`` means the items
    are mutually orthogonal.
``catalogue coverage``
    Fraction of the catalogue that appears in at least one list.
``category / brand diversity``
    Average fraction of distinct taxonomy or brand values per list. This is the
    business-facing view of saturation: ten variants of one brand are diverse in
    price but not in assortment.
``novelty``
    Mean self-information ``-log2(p)`` of the recommended items, where ``p`` is
    the training interaction share. Add-one smoothing keeps items that never
    appear in training finite instead of producing ``inf``.

Every function receives item *rows* (matrix positions), never product ids, so
the caller cannot accidentally mix the two identifier spaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from model.core.logger import get_logger

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class DiversityReport:
    """Aggregated diversity and novelty values for one set of lists."""

    users: int
    intra_list_diversity: float
    catalogue_coverage: float
    mean_distinct_categories: float
    mean_category_ratio: float
    mean_distinct_brands: float
    mean_brand_ratio: float
    novelty: float

    def to_dict(self) -> dict[str, float | int]:
        """JSON-ready representation with rounded values."""
        return {
            "users": int(self.users),
            "intra_list_diversity": round(self.intra_list_diversity, 6),
            "catalogue_coverage": round(self.catalogue_coverage, 6),
            "mean_distinct_categories": round(self.mean_distinct_categories, 4),
            "mean_category_ratio": round(self.mean_category_ratio, 6),
            "mean_distinct_brands": round(self.mean_distinct_brands, 4),
            "mean_brand_ratio": round(self.mean_brand_ratio, 6),
            "novelty_bits": round(self.novelty, 6),
        }


def valid_lists(item_rows: np.ndarray) -> list[np.ndarray]:
    """Convert a padded ranking matrix into a list of non-empty item-row arrays.

    Args:
        item_rows: ``(users, k)`` integer matrix where ``-1`` marks an empty slot.

    Returns:
        One array per user, with padding removed and order preserved.
    """
    output: list[np.ndarray] = []
    for row in np.asarray(item_rows):
        values = row[row >= 0]
        output.append(values.astype(np.int64, copy=False))
    return output


def item_popularity(events: pd.DataFrame, item_index: pd.DataFrame) -> np.ndarray:
    """Return the add-one-smoothed training popularity of every catalogue row.

    Args:
        events: Training interactions (only ``product_id`` is read).
        item_index: ``product_id`` -> ``item_row`` mapping.

    Returns:
        Probability vector of length ``len(item_index)`` summing to one.
    """
    counts = events.groupby("product_id", observed=True).size()
    positions = pd.Index(item_index["product_id"].to_numpy())
    item_rows = positions.get_indexer(counts.index.to_numpy())

    values = np.zeros(len(item_index), dtype=np.float64)
    keep = item_rows >= 0
    np.add.at(values, item_rows[keep], counts.to_numpy(dtype=np.float64)[keep])

    smoothed = values + 1.0
    return smoothed / smoothed.sum()


def category_lookup(
    products: pd.DataFrame,
    item_index: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Align category and brand labels with the catalogue row order.

    Args:
        products: Product catalogue containing ``product_id``,
            ``category_level1`` and ``brand``.
        item_index: ``product_id`` -> ``item_row`` mapping.

    Returns:
        Tuple ``(categories, brands)`` of object arrays aligned with item rows.
        Missing labels become ``"Unknown"``.
    """
    columns = ["product_id"]
    for column in ("category_level1", "brand"):
        if column in products.columns:
            columns.append(column)

    lookup = products[columns].drop_duplicates(subset=["product_id"], keep="first")
    aligned = item_index[["product_id"]].merge(lookup, on="product_id", how="left")

    categories = (
        aligned["category_level1"].fillna("Unknown").astype(str).to_numpy()
        if "category_level1" in aligned.columns
        else np.full(len(item_index), "Unknown", dtype=object)
    )
    brands = (
        aligned["brand"].fillna("Unknown").astype(str).to_numpy()
        if "brand" in aligned.columns
        else np.full(len(item_index), "Unknown", dtype=object)
    )
    return categories, brands


def intra_list_diversity(
    item_matrix: sp.spmatrix,
    lists: list[np.ndarray],
) -> tuple[float, int]:
    """Average within-list diversity across users.

    Args:
        item_matrix: L2-normalised catalogue matrix; a dot product is therefore
            the cosine similarity.
        lists: Non-empty item-row arrays.

    Returns:
        Tuple ``(mean diversity, users with at least two items)``. The mean is
        ``0.0`` when no list has two items.
    """
    matrix = sp.csr_matrix(item_matrix)
    total = 0.0
    counted = 0

    for items in lists:
        if items.size < 2:
            continue
        block = matrix[items]
        similarity = np.asarray((block @ block.T).todense(), dtype=np.float64)
        # Exclude the diagonal, whose similarity is one by construction.
        off_diagonal_mean = (similarity.sum() - np.trace(similarity)) / (
            similarity.size - similarity.shape[0]
        )
        total += 1.0 - off_diagonal_mean
        counted += 1

    return (total / counted if counted else 0.0), counted


def _label_diversity(lists: list[np.ndarray], labels: np.ndarray) -> tuple[float, float]:
    """Average distinct-label count and ratio per list.

    Args:
        lists: Non-empty item-row arrays.
        labels: Label array aligned with catalogue rows.

    Returns:
        Tuple ``(mean distinct labels, mean distinct ratio)``.
    """
    distinct_total = 0.0
    ratio_total = 0.0
    counted = 0

    for items in lists:
        if not items.size:
            continue
        distinct = len(np.unique(labels[items]))
        distinct_total += distinct
        ratio_total += distinct / items.size
        counted += 1

    if not counted:
        return 0.0, 0.0
    return distinct_total / counted, ratio_total / counted


def compute_diversity_report(
    item_rows: np.ndarray,
    item_matrix: sp.spmatrix,
    popularity: np.ndarray,
    categories: np.ndarray | None = None,
    brands: np.ndarray | None = None,
    max_users: int | None = None,
) -> DiversityReport:
    """Compute every diversity and novelty diagnostic for a ranking block.

    Args:
        item_rows: ``(users, k)`` padded ranking matrix.
        item_matrix: L2-normalised catalogue matrix.
        popularity: Add-one-smoothed training popularity per item row.
        categories: Optional category label per item row.
        brands: Optional brand label per item row.
        max_users: Deterministic cap on the number of users analysed. ``None``
            analyses every user.

    Returns:
        ``DiversityReport`` with all metrics, including the coverage computed on
        the full block rather than the capped sample.
    """
    row_limit = len(item_rows) if max_users is None else min(int(max_users), len(item_rows))
    sampled_rows = np.asarray(item_rows)[:row_limit]
    lists = valid_lists(sampled_rows)
    non_empty = [items for items in lists if items.size]

    flat = np.concatenate(non_empty) if non_empty else np.array([], dtype=np.int64)
    coverage = float(np.unique(flat).size / item_matrix.shape[0]) if flat.size else 0.0

    diversity_value, _ = intra_list_diversity(item_matrix, non_empty)

    if categories is None:
        mean_categories, category_ratio = 0.0, 0.0
    else:
        mean_categories, category_ratio = _label_diversity(non_empty, categories)

    if brands is None:
        mean_brands, brand_ratio = 0.0, 0.0
    else:
        mean_brands, brand_ratio = _label_diversity(non_empty, brands)

    if flat.size:
        smoothed = np.clip(popularity[flat], np.finfo(np.float64).tiny, None)
        novelty = float(np.mean(-np.log2(smoothed)))
    else:
        novelty = 0.0

    report = DiversityReport(
        users=len(non_empty),
        intra_list_diversity=diversity_value,
        catalogue_coverage=coverage,
        mean_distinct_categories=mean_categories,
        mean_category_ratio=category_ratio,
        mean_distinct_brands=mean_brands,
        mean_brand_ratio=brand_ratio,
        novelty=novelty,
    )
    _LOGGER.info(
        "Diversity: users=%s | intra-list=%.4f | coverage=%.4f | novelty=%.3f bits",
        f"{report.users:,}",
        report.intra_list_diversity,
        report.catalogue_coverage,
        report.novelty,
    )
    return report
