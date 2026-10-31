"""Ranking metrics for the offline evaluation (Plan.md Module 3.2).

Definitions (``R_u`` is the set of relevant items for user ``u``, ``L_u@K`` the
recommended list of length ``K``):

``HitRate@K``   fraction of users with at least one relevant item in ``L_u@K``.
``Recall@K``    ``|R_u intersect L_u@K| / |R_u|`` averaged over users.
``Precision@K`` ``|R_u intersect L_u@K| / K`` averaged over users.
``MRR@K``       mean of ``1 / rank`` for the first relevant item (0 if absent).

Users whose entire relevance set was removed as "already seen" are excluded
rather than scored as zero: they have no achievable target, so including them
would only dilute the metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.core.logger import get_logger
from model.ranking.scorer import RankingResult

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class RankingMetrics:
    """Metric values for one cut-off ``K``."""

    k: int
    users: int
    hit_rate: float
    recall: float
    precision: float
    mrr: float
    coverage: float

    def to_dict(self) -> dict[str, float | int]:
        """JSON-ready representation with rounded values."""
        return {
            "k": int(self.k),
            "users": int(self.users),
            "hit_rate": round(self.hit_rate, 6),
            "recall": round(self.recall, 6),
            "precision": round(self.precision, 6),
            "mrr": round(self.mrr, 6),
            "coverage": round(self.coverage, 6),
        }


def evaluate_at_k(
    rankings: RankingResult,
    truth: dict[int, np.ndarray],
    user_keys: np.ndarray,
    k: int,
    n_items: int,
) -> RankingMetrics:
    """Compute every metric at one cut-off.

    Args:
        rankings: Output of :func:`model.ranking.scorer.rank_top_k`.
        truth: Mapping ``user_key -> relevant item rows``.
        user_keys: User identifiers aligned with the ranking rows.
        k: Cut-off.
        n_items: Catalogue size, used for coverage.

    Returns:
        ``RankingMetrics`` for this cut-off.

    Raises:
        ValueError: If ``rankings`` and ``user_keys`` disagree on length.
    """
    if rankings.users != len(user_keys):
        raise ValueError(
            f"Rankings contain {rankings.users} users but {len(user_keys)} keys were supplied."
        )

    cut = min(k, rankings.item_rows.shape[1])
    evaluated = 0
    hits_total = 0
    recall_total = 0.0
    precision_total = 0.0
    reciprocal_total = 0.0
    recommended: set[int] = set()

    for index, key in enumerate(user_keys):
        relevant = truth.get(int(key))
        if relevant is None or len(relevant) == 0:
            continue

        recommended_row = rankings.item_rows[index, :cut]
        recommended_row = recommended_row[recommended_row >= 0]
        if recommended_row.size == 0:
            evaluated += 1
            continue

        recommended.update(int(item) for item in recommended_row)
        matches = np.isin(recommended_row, relevant, assume_unique=False)
        hits = int(matches.sum())

        evaluated += 1
        if hits:
            hits_total += 1
            recall_total += hits / len(relevant)
            precision_total += hits / cut
            # ``argmax`` on a boolean mask returns the index of the first True.
            reciprocal_total += 1.0 / (int(np.argmax(matches)) + 1)

    denominator = max(evaluated, 1)
    coverage = len(recommended) / n_items if n_items else 0.0

    return RankingMetrics(
        k=cut,
        users=evaluated,
        hit_rate=hits_total / denominator,
        recall=recall_total / denominator,
        precision=precision_total / denominator,
        mrr=reciprocal_total / denominator,
        coverage=coverage,
    )


def evaluate_all_k(
    rankings: RankingResult,
    truth: dict[int, np.ndarray],
    user_keys: np.ndarray,
    k_values: tuple[int, ...],
    n_items: int,
) -> dict[str, RankingMetrics]:
    """Compute metrics for every configured cut-off.

    Args:
        rankings: Ranking output.
        truth: Ground-truth mapping.
        user_keys: Aligned user identifiers.
        k_values: Cut-offs to report.
        n_items: Catalogue size.

    Returns:
        Mapping ``str(k) -> RankingMetrics``.
    """
    metrics = {
        str(int(k)): evaluate_at_k(rankings, truth, user_keys, int(k), n_items)
        for k in sorted(set(k_values))
    }
    _LOGGER.debug(
        "Metrics: %s",
        ", ".join(
            f"K={item.k} recall={item.recall:.4f} hit={item.hit_rate:.4f}"
            for item in metrics.values()
        ),
    )
    return metrics


def select_metric(metric: str, metrics: dict[str, RankingMetrics], k: int) -> float:
    """Extract the scalar Optuna objective from a metric report.

    Args:
        metric: ``recall``, ``hit_rate`` or ``mrr``.
        metrics: Output of :func:`evaluate_all_k`.
        k: Cut-off to read.

    Returns:
        The requested metric value (0.0 when the cut-off is missing).

    Raises:
        ValueError: If the metric name is unknown.
    """
    key = str(int(k))
    if key not in metrics:
        return 0.0
    values = metrics[key]
    if metric == "recall":
        return values.recall
    if metric == "hit_rate":
        return values.hit_rate
    if metric == "mrr":
        return values.mrr
    raise ValueError(f"Unknown metric {metric!r}; expected recall, hit_rate or mrr.")
