"""Evaluation orchestration: one model, several metric cut-offs.

The evaluator is deliberately thin. It scores nothing itself; it combines the
ranking output (:mod:`model.ranking.scorer`) with the relevance labels
(:mod:`model.ranking.filters`) and the metric definitions
(:mod:`model.evaluation.metrics`), and produces a JSON-ready report that can be
persisted without further post-processing.
"""

from __future__ import annotations

import numpy as np

from model.core.logger import get_logger
from model.evaluation.metrics import RankingMetrics, evaluate_all_k
from model.ranking.scorer import RankingResult

_LOGGER = get_logger(__name__)


def evaluate_ranking_result(
    rankings: RankingResult,
    truth: dict[int, np.ndarray],
    user_keys: np.ndarray,
    k_values: tuple[int, ...],
    n_items: int,
) -> dict[str, object]:
    """Evaluate one ranking against a ground-truth mapping.

    Args:
        rankings: Top-K output for ``user_keys``.
        truth: Relevant item rows per user (already seen-filtered).
        user_keys: Aligned user identifiers.
        k_values: Cut-offs to report.
        n_items: Catalogue size for coverage.

    Returns:
        Dictionary with per-cut-off metrics and coverage diagnostics.
    """
    metrics: dict[str, RankingMetrics] = evaluate_all_k(
        rankings, truth, user_keys, k_values, n_items
    )

    evaluated = max((item.users for item in metrics.values()), default=0)
    report: dict[str, object] = {
        "by_k": {key: value.to_dict() for key, value in metrics.items()},
        "users_scored": int(rankings.users),
        "users_evaluated": int(evaluated),
        "users_without_relevant_items": int(rankings.users - evaluated),
    }

    _LOGGER.info(
        "Evaluated %s users at K=%s | recall=%s",
        f"{evaluated:,}",
        ",".join(key for key in sorted(metrics, key=int)),
        ", ".join(f"{metrics[key].recall:.4f}" for key in sorted(metrics, key=int)),
    )
    return report


def aggregate_window_reports(
    reports: list[dict[str, object]],
    k_values: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    """Aggregate per-window metrics into mean and standard deviation.

    The standard deviation across chronological windows is the stability signal
    used to break ties between models in the benchmark: a model with a slightly
    lower mean but a much smaller spread is preferable in production.

    Args:
        reports: One report per window (output of :func:`evaluate_ranking_result`).
        k_values: Cut-offs present in every report.

    Returns:
        ``{str(k): {"recall_mean": ..., "recall_std": ..., ...}}``.
    """
    aggregated: dict[str, dict[str, float]] = {}

    for k in sorted(set(k_values)):
        key = str(int(k))
        rows = [report["by_k"][key] for report in reports if key in report["by_k"]]  # type: ignore[index]
        if not rows:
            continue

        summary: dict[str, float] = {}
        for metric in ("hit_rate", "recall", "precision", "mrr", "coverage"):
            values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
            summary[f"{metric}_mean"] = round(float(values.mean()), 6)
            summary[f"{metric}_std"] = round(float(values.std(ddof=0)), 6)
        aggregated[key] = summary

    return aggregated
