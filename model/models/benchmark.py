"""Three-model stability benchmark on the validation split (Plan.md Module 2.2).

Protocol
--------
1. Select users with enough training and validation history.
2. Build the three profiles **once** from the training split (identical users,
   identical catalogue matrix), so the comparison isolates the weighting scheme.
3. Split the validation period into chronologically ordered windows of equal row
   count. Recommendation lists do not depend on the evaluation window, so one
   ranking pass per model is reused for every window; only the relevance labels
   change.
4. Report mean and standard deviation of every metric across windows. The
   standard deviation is the stability signal: a model that swings between 0.20
   and 0.05 Recall is worse in production than one that stays near 0.13.

Selection rule: highest mean ``Recall@K``; ties are broken by higher mean
``HitRate@K``, then by lower Recall standard deviation, then by registry order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from model.config.config_schema import PipelineConfig
from model.core.logger import get_logger
from model.evaluation.evaluator import aggregate_window_reports, evaluate_ranking_result
from model.models.base import MODEL_SPECS, ModelSpec
from model.models.centroid_models import build_all_profiles
from model.profiles.time_decay import to_utc_ns
from model.profiles.user_profiles import eligible_user_keys, sample_user_keys
from model.ranking.filters import (
    binarize_interactions,
    build_ground_truth,
    exclude_seen_from_truth,
    item_signal_mask,
)
from model.ranking.scorer import RankingResult, rank_top_k

_LOGGER = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Per-model metrics, window stability and the selected winner."""

    report: dict[str, Any]
    winner: ModelSpec
    rankings: dict[str, RankingResult]
    profiles: dict[str, Any]
    user_keys: np.ndarray
    seen_matrix: sp.csr_matrix
    truth: dict[int, np.ndarray]


def _subset_rows(matrix: sp.spmatrix, positions: np.ndarray) -> sp.csr_matrix:
    """Select CSR rows by position without touching the feature dimension."""
    return sp.csr_matrix(matrix)[positions]


def chronological_windows(
    frame: pd.DataFrame,
    timestamp_column: str,
    n_windows: int,
) -> list[pd.DataFrame]:
    """Split a frame into equal-row, chronologically ordered windows.

    Args:
        frame: Evaluation interactions.
        timestamp_column: Timestamp column used for ordering.
        n_windows: Number of windows (clamped to the number of rows).

    Returns:
        List of non-empty windows ordered from earliest to latest.
    """
    if frame.empty:
        return []

    ordered = frame.sort_values(
        [timestamp_column, "user_key", "product_id", "event_type_key"], kind="mergesort"
    ).reset_index(drop=True)

    windows = int(max(1, min(n_windows, len(ordered))))
    boundaries = np.linspace(0, len(ordered), windows + 1).astype(int)

    return [
        ordered.iloc[boundaries[index] : boundaries[index + 1]].reset_index(drop=True)
        for index in range(windows)
        if boundaries[index + 1] > boundaries[index]
    ]


def _window_truth(
    windows: list[pd.DataFrame],
    item_index: pd.DataFrame,
    user_keys: np.ndarray,
    seen: sp.csr_matrix,
    timestamp_column: str,
    exclude_seen: bool,
) -> list[dict[int, np.ndarray]]:
    """Build seen-filtered relevance labels for every window."""
    labels: list[dict[int, np.ndarray]] = []
    for window in windows:
        truth = build_ground_truth(window, item_index, user_keys)
        if exclude_seen:
            truth, _ = exclude_seen_from_truth(truth, seen, user_keys)
        labels.append(truth)
    return labels


def _select_winner(
    model_reports: dict[str, dict[str, Any]],
    k: int,
) -> tuple[ModelSpec, dict[str, Any]]:
    """Apply the documented selection rule and return the winner plus rationale."""
    key = str(int(k))
    candidates: list[dict[str, Any]] = []

    for order, model in enumerate(MODEL_SPECS):
        report = model_reports[model.key]
        stability = report["stability"][key]
        ranked = report["ranking"]["by_k"][key]
        candidates.append(
            {
                "model": model,
                "order": order,
                "recall_mean": stability["recall_mean"],
                "recall_std": stability["recall_std"],
                "hit_rate_mean": stability["hit_rate_mean"],
                "overall_recall": ranked["recall"],
                "overall_hit_rate": ranked["hit_rate"],
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["recall_mean"],
            -item["hit_rate_mean"],
            item["recall_std"],
            item["order"],
        )
    )
    winner = candidates[0]

    ranking_table = [
        {
            "model": item["model"].key,
            "label": item["model"].label,
            "recall_mean": round(float(item["recall_mean"]), 6),
            "recall_std": round(float(item["recall_std"]), 6),
            "hit_rate_mean": round(float(item["hit_rate_mean"]), 6),
            "overall_recall": round(float(item["overall_recall"]), 6),
            "overall_hit_rate": round(float(item["overall_hit_rate"]), 6),
        }
        for item in candidates
    ]

    rationale = {
        "criterion": "max mean Recall@K, then mean HitRate@K, then min Recall std",
        "k": int(k),
        "ranking": ranking_table,
        "winner": winner["model"].key,
    }
    return winner["model"], rationale


def benchmark_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    item_matrix: sp.spmatrix,
    item_index: pd.DataFrame,
    config: PipelineConfig,
    model_specs: tuple[ModelSpec, ...] = MODEL_SPECS,
) -> BenchmarkResult:
    """Run the stability benchmark and select the winning model.

    Args:
        train: Training interactions.
        validation: Validation interactions.
        item_matrix: Module 1 catalogue matrix.
        item_index: ``product_id`` -> ``item_row`` mapping.
        config: Validated pipeline configuration.
        model_specs: Models to compare (all three by default).

    Returns:
        ``BenchmarkResult`` with the JSON-ready report, the winner and the
        reusable rankings/profiles.

    Raises:
        ValueError: If no eligible user exists or a model produces no profiles.
    """
    timestamp_column = config.data.timestamp_column
    k_values = tuple(sorted(set((*config.evaluation.k_values, config.benchmark.k))))
    max_k = max(k_values)

    eligible = eligible_user_keys(train, validation, config)
    if not len(eligible):
        raise ValueError(
            "No user has both training and validation history; the benchmark "
            "cannot run. Check evaluation.min_train_interactions."
        )

    user_keys = sample_user_keys(eligible, config.benchmark.max_users, config.seed)
    train_subset = train.loc[train["user_key"].isin(user_keys)]
    validation_subset = validation.loc[validation["user_key"].isin(user_keys)]

    global_reference = int(to_utc_ns(train_subset[timestamp_column]).max())

    profiles = build_all_profiles(
        train_subset,
        item_matrix,
        item_index,
        config,
        user_keys,
        global_reference_ns=global_reference,
    )

    reference_profile = profiles["unweighted"]
    seen = binarize_interactions(reference_profile.interaction_matrix)
    allowed = item_signal_mask(item_matrix)

    truth = build_ground_truth(validation_subset, item_index, user_keys)
    if config.evaluation.exclude_seen_items:
        truth, seen_filter = exclude_seen_from_truth(truth, seen, user_keys)
    else:
        seen_filter = {}

    scorable = np.array(sorted(truth), dtype=np.int64)
    if not len(scorable):
        raise ValueError(
            "Every validation interaction was already seen in training; there is "
            "nothing left to rank. Review the split or disable exclude_seen_items."
        )

    positions = pd.Series(np.arange(len(user_keys)), index=user_keys)
    scorable_rows = positions.reindex(scorable).to_numpy().astype(np.int64)

    windows = chronological_windows(validation_subset, timestamp_column, config.benchmark.stability_windows)
    window_truths = _window_truth(
        windows, item_index, user_keys, seen, timestamp_column, config.evaluation.exclude_seen_items
    )

    model_reports: dict[str, dict[str, Any]] = {}
    rankings: dict[str, RankingResult] = {}

    for model in model_specs:
        profile = profiles[model.key]
        ranking = rank_top_k(
            _subset_rows(profile.matrix, scorable_rows),
            item_matrix,
            k=max_k,
            seen_matrix=_subset_rows(seen, scorable_rows),
            allowed_items=allowed,
            batch_size=config.evaluation.scoring_batch_size,
        )
        rankings[model.key] = ranking

        overall = evaluate_ranking_result(
            ranking, truth, scorable, k_values, item_matrix.shape[0]
        )

        window_reports: list[dict[str, Any]] = []
        for index, labels in enumerate(window_truths):
            window_report = evaluate_ranking_result(
                ranking, labels, scorable, k_values, item_matrix.shape[0]
            )
            window_reports.append(
                {
                    "window": index,
                    "users_evaluated": window_report["users_evaluated"],
                    "by_k": window_report["by_k"],
                }
            )

        stability = aggregate_window_reports(
            [{"by_k": report["by_k"]} for report in window_reports], k_values
        )
        model_reports[model.key] = {
            "model": model.key,
            "label": model.label,
            "description": model.description,
            "profile": profile.stats,
            "ranking": overall,
            "windows": window_reports,
            "stability": stability,
        }
        _LOGGER.info(
            "%s | overall Recall@%s = %.4f | window mean = %.4f (std %.4f)",
            model.label,
            max_k,
            overall["by_k"][str(max_k)]["recall"],
            stability[str(max_k)]["recall_mean"],
            stability[str(max_k)]["recall_std"],
        )

    winner, rationale = _select_winner(model_reports, config.benchmark.k)

    report: dict[str, Any] = {
        "protocol": {
            "k_values": list(k_values),
            "selection_k": int(config.benchmark.k),
            "users_sampled": int(len(user_keys)),
            "users_scored": int(len(scorable)),
            "windows": len(windows),
            "exclude_seen_items": bool(config.evaluation.exclude_seen_items),
            "seen_filter": seen_filter,
            "sampling_seed": int(config.seed),
            "decay_reference_ns": int(global_reference),
        },
        "models": model_reports,
        "selection": rationale,
    }

    _LOGGER.info(
        "Benchmark winner: %s (mean Recall@%s = %.4f)",
        winner.label,
        config.benchmark.k,
        rationale["ranking"][0]["recall_mean"],
    )

    return BenchmarkResult(
        report=report,
        winner=winner,
        rankings=rankings,
        profiles=profiles,
        user_keys=user_keys,
        seen_matrix=seen,
        truth=truth,
    )
