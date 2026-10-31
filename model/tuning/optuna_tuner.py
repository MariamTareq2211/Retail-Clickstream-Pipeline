"""Optuna study for the winning model (Plan.md Module 2.3).

Objective
---------
Maximise the configured validation metric (``tuning.metric``) at
``benchmark.k``. Every trial rebuilds the catalogue matrix for its TF-IDF and
fusion parameters, rebuilds the user profiles with its event weights, repeat
transform and decay rate, ranks the sampled validation users and scores the
result against the seen-filtered relevance labels.

The study is kept **in memory** on purpose. A persistent sqlite storage would
make repeated runs depend on the previous ones, which undermines the
reproducibility story; the full trial table is exported to CSV instead so the
experiment can still be audited or plotted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp

from model.config.config_schema import PipelineConfig
from model.core.logger import get_logger
from model.evaluation.metrics import evaluate_all_k, select_metric
from model.models.base import ModelSpec
from model.profiles.interaction_weights import ProfileParams, build_interaction_table, build_weight_matrix
from model.profiles.time_decay import to_utc_ns
from model.profiles.user_profiles import build_profiles, eligible_user_keys, sample_user_keys
from model.ranking.filters import (
    binarize_interactions,
    build_ground_truth,
    exclude_seen_from_truth,
    item_signal_mask,
)
from model.ranking.scorer import rank_top_k
from model.tuning.search_space import ItemMatrixFactory, TrialParams, suggest_trial_params

_LOGGER = get_logger(__name__)


@dataclass
class TuningResult:
    """Outcome of one Optuna study."""

    best_params: TrialParams
    best_value: float
    metric: str
    k: int
    trials: pd.DataFrame
    best_trial: int
    completed_trials: int
    study_name: str

    def summary(self) -> dict[str, Any]:
        """JSON-ready summary for the stage manifest."""
        return {
            "study_name": self.study_name,
            "metric": self.metric,
            "k": int(self.k),
            "completed_trials": int(self.completed_trials),
            "best_trial": int(self.best_trial),
            "best_value": round(float(self.best_value), 6),
            "best_params": self.best_params.to_dict(),
        }


@dataclass
class _TuningContext:
    """Pre-computed, trial-independent state for the objective."""

    train_subset: pd.DataFrame
    validation_subset: pd.DataFrame
    user_keys: np.ndarray
    scorable_keys: np.ndarray
    scorable_rows: np.ndarray
    seen: sp.csr_matrix
    truth: dict[int, np.ndarray]
    global_reference_ns: int
    k_values: tuple[int, ...]
    max_k: int


def _select_rows(matrix: sp.spmatrix, positions: np.ndarray) -> sp.csr_matrix:
    """Select CSR rows by position."""
    return sp.csr_matrix(matrix)[positions]


def _prepare_context(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    item_index: pd.DataFrame,
    config: PipelineConfig,
    max_users: int,
) -> _TuningContext:
    """Build the reusable sampling / labelling state of the study."""
    timestamp_column = config.data.timestamp_column
    k_values = tuple(sorted(set((*config.evaluation.k_values, config.benchmark.k))))

    eligible = eligible_user_keys(train, validation, config)
    if not len(eligible):
        raise ValueError("No eligible user for tuning (no training or validation history).")

    user_keys = sample_user_keys(eligible, max_users, config.seed + config.tuning.seed_offset)
    train_subset = train.loc[train["user_key"].isin(user_keys)]
    validation_subset = validation.loc[validation["user_key"].isin(user_keys)]

    global_reference = int(to_utc_ns(train_subset[timestamp_column]).max())

    # The seen mask is the set of distinct (user, item) pairs in training and is
    # independent of the weighting scheme, so it is computed once.
    pair_table = build_interaction_table(
        train_subset,
        ProfileParams.from_config(config),
        item_index,
        timestamp_column,
        use_event_weights=False,
        use_counts=False,
        use_decay=False,
        global_reference_ns=global_reference,
    )
    seen = binarize_interactions(
        build_weight_matrix(pair_table, user_keys, len(item_index))
    )

    truth = build_ground_truth(validation_subset, item_index, user_keys)
    if config.evaluation.exclude_seen_items:
        truth, _ = exclude_seen_from_truth(truth, seen, user_keys)

    scorable_keys = np.array(sorted(truth), dtype=np.int64)
    if not len(scorable_keys):
        raise ValueError(
            "No validation label remains after removing items seen in training; "
            "tuning cannot be evaluated."
        )

    positions = pd.Series(np.arange(len(user_keys)), index=user_keys)
    scorable_rows = positions.reindex(scorable_keys).to_numpy().astype(np.int64)

    return _TuningContext(
        train_subset=train_subset,
        validation_subset=validation_subset,
        user_keys=user_keys,
        scorable_keys=scorable_keys,
        scorable_rows=scorable_rows,
        seen=seen,
        truth=truth,
        global_reference_ns=global_reference,
        k_values=k_values,
        max_k=max(k_values),
    )


def tune_model(
    model: ModelSpec,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    item_index: pd.DataFrame,
    metadata_text: pd.Series,
    train_mask: pd.Series,
    price_vector: np.ndarray,
    config: PipelineConfig,
) -> TuningResult:
    """Run the Optuna study for one model.

    Args:
        model: Winning model from the benchmark.
        train: Training interactions.
        validation: Validation interactions.
        item_index: ``product_id`` -> ``item_row`` mapping.
        metadata_text: Catalogue documents produced by Module 1.
        train_mask: Boolean mask of products with training interactions.
        price_vector: Scaled price column from the Module 1 transformer.
        config: Validated pipeline configuration.

    Returns:
        ``TuningResult`` with the best parameters and the full trial table.

    Raises:
        ValueError: If the tuning sample cannot be built.
        RuntimeError: If no trial completes with a finite objective.
    """
    context = _prepare_context(train, validation, item_index, config, config.tuning.max_users)
    factory = ItemMatrixFactory(metadata_text, train_mask, price_vector, config)

    metric = config.tuning.metric
    selection_k = int(config.benchmark.k)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = suggest_trial_params(trial, model, config)
        item_matrix = factory.build(params)

        profile_params = ProfileParams(
            event_weights=params.event_weights,
            count_transform=params.count_transform,
            count_cap=config.dedup.count_cap,
            decay_enabled=model.use_decay,
            decay_reference=config.profiles.decay.reference,
            decay_rate_per_hour=params.decay_rate_per_hour,
        )

        profiles = build_profiles(
            context.train_subset,
            item_matrix,
            item_index,
            model,
            profile_params,
            config,
            user_keys=context.scorable_keys,
            global_reference_ns=context.global_reference_ns,
        )

        ranking = rank_top_k(
            profiles.matrix,
            item_matrix,
            k=context.max_k,
            seen_matrix=_select_rows(context.seen, context.scorable_rows),
            allowed_items=item_signal_mask(item_matrix),
            batch_size=config.evaluation.scoring_batch_size,
        )

        metrics = evaluate_all_k(
            ranking,
            context.truth,
            context.scorable_keys,
            context.k_values,
            item_matrix.shape[0],
        )
        value = select_metric(metric, metrics, selection_k)

        trial.set_user_attr("ngram_range", f"{params.ngram_range[0]}-{params.ngram_range[1]}")
        trial.set_user_attr("users_evaluated", int(metrics[str(selection_k)].users))
        trial.set_user_attr("catalogue_features", int(item_matrix.shape[1]))
        trial.set_user_attr("catalogue_nnz", int(item_matrix.nnz))

        return float(value) if math.isfinite(value) else 0.0

    def _log_trial(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            _LOGGER.warning("Trial %s ended in state %s", trial.number, trial.state.name)
            return
        _LOGGER.info(
            "Trial %s | %s@%s=%.6f | best=%.6f",
            trial.number,
            metric,
            selection_k,
            float(trial.value),
            float(study.best_value),
        )

    sampler = optuna.samplers.TPESampler(seed=config.seed + config.tuning.seed_offset)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=config.tuning.study_name,
    )

    timeout = config.tuning.timeout_seconds if config.tuning.timeout_seconds > 0 else None
    _LOGGER.info(
        "Starting Optuna study %r for %s | trials=%s | timeout=%ss | users=%s",
        config.tuning.study_name,
        model.label,
        config.tuning.n_trials,
        timeout,
        f"{len(context.scorable_keys):,}",
    )

    study.optimize(
        objective,
        n_trials=config.tuning.n_trials,
        timeout=timeout,
        callbacks=[_log_trial],
        catch=(ValueError,),
    )

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"Optuna produced no completed trial for {model.label}. "
            "Check tuning.search_space and the training vocabulary."
        )

    best_trial = study.best_trial
    best_params = suggest_trial_params(_ReplayTrial(best_trial), model, config)

    trials_frame = study.trials_dataframe(
        attrs=("number", "value", "state", "params", "user_attrs", "duration")
    )

    _LOGGER.info(
        "Optuna finished: %s completed trials | best trial %s | %s@%s=%.6f",
        len(completed),
        best_trial.number,
        metric,
        selection_k,
        float(best_trial.value),
    )

    return TuningResult(
        best_params=best_params,
        best_value=float(best_trial.value),
        metric=metric,
        k=selection_k,
        trials=trials_frame,
        best_trial=int(best_trial.number),
        completed_trials=len(completed),
        study_name=config.tuning.study_name,
    )


class _ReplayTrial:
    """Replay a finished trial's parameters through ``suggest_trial_params``.

    ``study.best_trial.params`` only contains the sampled values; replaying them
    through the same suggestion function guarantees the returned
    ``TrialParams`` (including unsampled defaults and the fixed purchase weight)
    is identical to what the objective actually used.
    """

    def __init__(self, trial: "optuna.trial.FrozenTrial") -> None:
        self._params = dict(trial.params)

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        """Return the recorded categorical value."""
        if name not in self._params:
            raise KeyError(f"Parameter {name!r} was not recorded by the study.")
        return self._params[name]

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        """Return the recorded float value."""
        if name not in self._params:
            raise KeyError(f"Parameter {name!r} was not recorded by the study.")
        return float(self._params[name])
