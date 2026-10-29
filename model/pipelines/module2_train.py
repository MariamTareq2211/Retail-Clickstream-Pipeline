"""Module 2 entry point: profiles, three-model benchmark and Optuna tuning.

Pipeline order
--------------
``load``       Read the Module 1 artifacts (matrix, index, splits, transformers).
``benchmark``  Build Models 1-3 on the same users and score them per validation
               window; select the most accurate and stable model.
``tune``       Run an Optuna study for the winning model (optional).
``rebuild``    Rebuild the catalogue matrix and profiles with the winning
               hyperparameters.
``persist``    Write the final matrix, profiles, seen-item mask and manifests.

Usage (from the project root)::

    venv\\\\Scripts\\\\python.exe -m model.pipelines.module2_train

The stage is deterministic for a given configuration and seed. The exported
``item_matrix.npz`` is the matrix the profiles were actually built from; Module 3
must consume it rather than the Module 1 file, because tuning may have changed
the TF-IDF hyperparameters.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from model.config.config_schema import (
    ConfigError,
    PipelineConfig,
    config_hash,
    config_to_dict,
    load_config,
)
from model.core.logger import configure_logging, get_logger
from model.core.paths import ensure_directory, resolve_path
from model.core.persistence import (
    load_joblib,
    load_json,
    load_parquet,
    load_sparse,
    save_json,
    save_parquet,
    save_sparse,
    sha256_file,
)
from model.core.seeding import set_global_seed
from model.features.price_features import PriceFeatureArtifacts, transform_prices
from model.features.text_features import build_metadata_text
from model.models.base import ModelSpec, describe_models
from model.models.benchmark import BenchmarkResult, benchmark_models
from model.profiles.interaction_weights import ProfileParams
from model.profiles.time_decay import describe_decay, to_utc_ns
from model.profiles.user_profiles import ProfileArtifacts, build_profiles
from model.ranking.filters import binarize_interactions
from model.tuning.optuna_tuner import TuningResult, tune_model
from model.tuning.search_space import ItemMatrixFactory, TrialParams

_LOGGER = get_logger(__name__)


class Module2Error(RuntimeError):
    """Raised when a Module 1 artifact is missing or inconsistent."""


class _Timer:
    """Context manager that records the duration of one pipeline stage."""

    def __init__(self, timings: dict[str, float], stage: str) -> None:
        self._timings = timings
        self._stage = stage
        self._start = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        _LOGGER.info("--- %s ---", self._stage)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        elapsed = time.perf_counter() - self._start
        self._timings[self._stage] = round(elapsed, 3)
        _LOGGER.info("--- %s finished in %.2fs ---", self._stage, elapsed)


@dataclass
class _Module1Bundle:
    """Everything Module 2 consumes from Module 1."""

    item_matrix: sp.csr_matrix
    item_index: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    products: pd.DataFrame
    metadata_text: pd.Series
    train_mask: pd.Series
    price_vector: np.ndarray
    manifest: dict[str, Any]


def _load_module1(config: PipelineConfig) -> _Module1Bundle:
    """Load and cross-check the Module 1 artifact contract.

    Args:
        config: Validated pipeline configuration.

    Returns:
        The loaded bundle.

    Raises:
        Module2Error: If an artifact is missing, empty or shape-inconsistent.
    """
    artifacts = resolve_path(config.paths.artifacts)
    required = (
        artifacts / "item_matrix.npz",
        artifacts / "item_index.parquet",
        artifacts / "splits" / "train.parquet",
        artifacts / "splits" / "val.parquet",
        artifacts / "price_transformer.joblib",
        artifacts / "module1_manifest.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise Module2Error(
            "Module 1 artifacts are missing:\n  - "
            + "\n  - ".join(missing)
            + "\nRun: venv\\Scripts\\python.exe -m model.pipelines.module1_preprocess"
        )

    item_matrix = sp.csr_matrix(load_sparse(artifacts / "item_matrix.npz"))
    item_index = load_parquet(artifacts / "item_index.parquet")
    train = load_parquet(artifacts / "splits" / "train.parquet")
    validation = load_parquet(artifacts / "splits" / "val.parquet")
    manifest = load_json(artifacts / "module1_manifest.json")

    if item_matrix.shape[0] != len(item_index):
        raise Module2Error(
            "item_matrix rows and item_index length disagree: "
            f"{item_matrix.shape[0]} vs {len(item_index)}."
        )
    if train.empty or validation.empty:
        raise Module2Error("The train or validation split is empty; rerun Module 1.")

    products = pd.read_parquet(resolve_path(config.paths.products))
    price_transformer: PriceFeatureArtifacts = load_joblib(artifacts / "price_transformer.joblib")

    metadata_text = build_metadata_text(products, config).reset_index(drop=True)
    train_items = set(train["product_id"].unique())
    train_mask = products["product_id"].isin(train_items).reset_index(drop=True)
    price_vector = transform_prices(
        products[config.price.column], price_transformer, config
    )

    _LOGGER.info(
        "Module 1 bundle: matrix=%s x %s | train=%s | val=%s | products=%s",
        f"{item_matrix.shape[0]:,}",
        f"{item_matrix.shape[1]:,}",
        f"{len(train):,}",
        f"{len(validation):,}",
        f"{len(products):,}",
    )

    return _Module1Bundle(
        item_matrix=item_matrix,
        item_index=item_index,
        train=train,
        validation=validation,
        products=products,
        metadata_text=metadata_text,
        train_mask=train_mask,
        price_vector=price_vector,
        manifest=manifest,
    )


def _default_trial_params(config: PipelineConfig) -> TrialParams:
    """Express the configured (untuned) hyperparameters as ``TrialParams``."""
    return TrialParams(
        max_features=int(config.text.tfidf.max_features),
        ngram_range=tuple(int(value) for value in config.text.tfidf.ngram_range),  # type: ignore[arg-type]
        min_df=int(config.text.tfidf.min_df),
        price_weight=float(config.fusion.price_weight),
        decay_rate_per_hour=float(config.profiles.decay.rate_per_hour),
        count_transform=config.dedup.count_transform,
        event_weights=dict(config.profiles.weights),
    )


def _profile_params(params: TrialParams, model: ModelSpec, config: PipelineConfig) -> ProfileParams:
    """Translate trial hyperparameters into profile-construction parameters."""
    return ProfileParams(
        event_weights=params.event_weights,
        count_transform=params.count_transform,
        count_cap=config.dedup.count_cap,
        decay_enabled=model.use_decay,
        decay_reference=config.profiles.decay.reference,
        decay_rate_per_hour=params.decay_rate_per_hour,
    )


def _all_training_users(train: pd.DataFrame, config: PipelineConfig) -> np.ndarray:
    """Users with enough training history to receive a profile."""
    counts = train.groupby("user_key", observed=True).size()
    eligible = counts.index[counts >= config.evaluation.min_train_interactions]
    return np.sort(eligible.to_numpy(dtype=np.int64))


def _save_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return path


def _write_markdown_report(
    path: Path,
    config: PipelineConfig,
    config_fingerprint: str,
    benchmark: BenchmarkResult,
    tuning: TuningResult | None,
    final_model: ModelSpec,
    final_params: TrialParams,
    final_profiles: ProfileArtifacts,
    final_matrix: sp.csr_matrix,
    timings: dict[str, float],
) -> Path:
    """Render the Module 2 summary for the thesis appendix."""
    selection = benchmark.report["selection"]
    k = int(selection["k"])

    benchmark_rows = [
        f"| {row['label']} | {row['recall_mean']:.4f} | {row['recall_std']:.4f} | "
        f"{row['hit_rate_mean']:.4f} | {row['overall_recall']:.4f} | {row['overall_hit_rate']:.4f} |"
        for row in selection["ranking"]
    ]

    window_rows = []
    for entry in benchmark.report["models"][final_model.key]["windows"]:
        metrics = entry["by_k"][str(k)]
        window_rows.append(
            f"| {entry['window']} | {entry['users_evaluated']:,} | "
            f"{metrics['hit_rate']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['mrr']:.4f} |"
        )

    if tuning is not None:
        tuning_text = (
            f"- Study: `{tuning.study_name}` ({tuning.completed_trials} completed trials)\n"
            f"- Best trial: {tuning.best_trial}\n"
            f"- Best {tuning.metric}@{tuning.k}: {tuning.best_value:.6f}\n"
            f"- Best parameters: `{tuning.best_params.to_dict()}`"
        )
    else:
        tuning_text = "- Tuning disabled; the configured hyperparameters were used."

    decay_text = describe_decay(final_params.decay_rate_per_hour)

    profile_stats = final_profiles.stats
    matrix_nnz = int(final_matrix.nnz)

    content = f"""# Module 2 - Profiles, model selection and tuning

Generated: {datetime.now(timezone.utc).isoformat()}
Configuration fingerprint: `{config_fingerprint}`

## Benchmark protocol

- Users sampled: {benchmark.report['protocol']['users_sampled']:,}
- Users scored: {benchmark.report['protocol']['users_scored']:,}
- Validation windows: {benchmark.report['protocol']['windows']}
- Selection cut-off: K = {k}
- Seen items excluded from candidates and labels: {benchmark.report['protocol']['exclude_seen_items']}

## Model comparison (mean over validation windows)

| model | Recall@K mean | Recall@K std | HitRate@K mean | Recall overall | HitRate overall |
|---|---:|---:|---:|---:|---:|
{chr(10).join(benchmark_rows)}

**Winner:** {final_model.label}

Selection rule: {selection['criterion']}.

## Winner by validation window

| window | users | HitRate@K | Recall@K | Precision@K | MRR@K |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(window_rows)}

## Optuna tuning

{tuning_text}

## Final profile matrix

- Users: {profile_stats['users']:,}
- Features: {profile_stats['features']:,}
- Mean items per user: {profile_stats['mean_items_per_user']:.2f}
- Mean profile non-zeros per user: {profile_stats['mean_profile_nnz_per_user']:.2f}
- Empty profiles: {profile_stats['empty_profiles']:,}
- Catalogue matrix non-zeros: {matrix_nnz:,}

## Decay

- λ = {decay_text['rate_per_hour']:.6f} per hour
- Half-life: {decay_text['half_life_hours']:.2f} hours
- Weight after 24 h: {decay_text['weight_after_24h']:.4f}

## Timings

{chr(10).join(f"- {stage}: {seconds:.2f}s" for stage, seconds in timings.items())}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _LOGGER.info("Wrote report: %s", path)
    return path


def run_module2(config: PipelineConfig) -> dict[str, Any]:
    """Execute Module 2 end to end.

    Args:
        config: Validated pipeline configuration.

    Returns:
        Summary dictionary with counts, metrics, selected model and timings.

    Raises:
        Module2Error: If a required Module 1 artifact is missing.
        ValueError: If profiling, benchmarking or tuning cannot be completed.
    """
    artifacts_dir = ensure_directory(config.paths.artifacts_module2)
    reports_dir = ensure_directory(config.paths.reports_module2)
    timings: dict[str, float] = {}
    fingerprint = config_hash(config)

    with _Timer(timings, "load"):
        bundle = _load_module1(config)

    with _Timer(timings, "benchmark"):
        benchmark = benchmark_models(
            bundle.train,
            bundle.validation,
            bundle.item_matrix,
            bundle.item_index,
            config,
        )

    tuning: TuningResult | None = None
    if config.tuning.enabled:
        with _Timer(timings, "tune"):
            tuning = tune_model(
                benchmark.winner,
                bundle.train,
                bundle.validation,
                bundle.item_index,
                bundle.metadata_text,
                bundle.train_mask,
                bundle.price_vector,
                config,
            )
        final_params = tuning.best_params
    else:
        _LOGGER.info("Tuning disabled; using configured hyperparameters.")
        final_params = _default_trial_params(config)

    final_model = benchmark.winner

    with _Timer(timings, "rebuild"):
        # Tuning may have changed the TF-IDF block, so the catalogue matrix is
        # rebuilt with the winning parameters instead of reusing Module 1's.
        factory = ItemMatrixFactory(
            bundle.metadata_text, bundle.train_mask, bundle.price_vector, config
        )
        final_matrix = sp.csr_matrix(factory.build(final_params))

        final_user_keys = _all_training_users(bundle.train, config)
        global_reference = int(to_utc_ns(bundle.train[config.data.timestamp_column]).max())

        final_profiles = build_profiles(
            bundle.train,
            final_matrix,
            bundle.item_index,
            final_model,
            _profile_params(final_params, final_model, config),
            config,
            user_keys=final_user_keys,
            global_reference_ns=global_reference,
        )

        seen_items = binarize_interactions(final_profiles.interaction_matrix)

    with _Timer(timings, "persist"):
        save_sparse(final_matrix, artifacts_dir / "item_matrix.npz")
        save_parquet(bundle.item_index, artifacts_dir / "item_index.parquet")
        save_sparse(final_profiles.matrix, artifacts_dir / "profiles.npz")
        save_sparse(seen_items, artifacts_dir / "seen_items.npz")

        user_index = pd.DataFrame(
            {
                "user_key": final_profiles.user_keys.astype(np.int64),
                "user_row": np.arange(len(final_profiles.user_keys), dtype=np.int64),
            }
        )
        save_parquet(user_index, artifacts_dir / "user_index.parquet")

        best_payload = {
            "tuned": bool(tuning is not None),
            "model": final_model.key,
            "metric": config.tuning.metric,
            "k": int(config.benchmark.k),
            "best_value": None if tuning is None else round(float(tuning.best_value), 6),
            "best_trial": None if tuning is None else int(tuning.best_trial),
            "parameters": final_params.to_dict(),
            "completed_trials": 0 if tuning is None else int(tuning.completed_trials),
        }
        save_json(best_payload, artifacts_dir / "best_params.json")

        if tuning is not None:
            _save_csv(tuning.trials, artifacts_dir / "tuning_trials.csv")

        winner_report = benchmark.report["models"][final_model.key]
        save_json(benchmark.report, artifacts_dir / "benchmark.json")

        model_spec = {
            "model": {
                "key": final_model.key,
                "label": final_model.label,
                "description": final_model.description,
                "use_event_weights": final_model.use_event_weights,
                "use_counts": final_model.use_counts,
                "use_decay": final_model.use_decay,
            },
            "models_evaluated": describe_models(),
            "selection": benchmark.report["selection"],
            "winner_overall": winner_report["ranking"]["by_k"],
            "winner_stability": winner_report["stability"],
            "parameters": final_params.to_dict(),
            "decay": describe_decay(final_params.decay_rate_per_hour),
            "feature_dimension": int(final_matrix.shape[1]),
            "catalogue_items": int(final_matrix.shape[0]),
            "profile_users": int(final_profiles.matrix.shape[0]),
            "evaluation": {
                "k_values": list(config.evaluation.k_values),
                "exclude_seen_items": bool(config.evaluation.exclude_seen_items),
                "min_train_interactions": int(config.evaluation.min_train_interactions),
            },
            "module1_config_fingerprint": bundle.manifest.get("config_fingerprint"),
            "config_fingerprint": fingerprint,
        }
        save_json(model_spec, artifacts_dir / "model_spec.json")
        save_json(config_to_dict(config), artifacts_dir / "config.resolved.json")

        fingerprints = {
            "item_matrix.npz": sha256_file(artifacts_dir / "item_matrix.npz"),
            "profiles.npz": sha256_file(artifacts_dir / "profiles.npz"),
            "seen_items.npz": sha256_file(artifacts_dir / "seen_items.npz"),
            "item_index.parquet": sha256_file(artifacts_dir / "item_index.parquet"),
            "user_index.parquet": sha256_file(artifacts_dir / "user_index.parquet"),
        }

        manifest = {
            "stage": "module2_train",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": fingerprint,
            "module1_config_fingerprint": bundle.manifest.get("config_fingerprint"),
            "seed": config.seed,
            "selected_model": final_model.key,
            "tuned": bool(tuning is not None),
            "profile_users": int(final_profiles.matrix.shape[0]),
            "catalogue_items": int(final_matrix.shape[0]),
            "feature_dimension": int(final_matrix.shape[1]),
            "timings_seconds": timings,
            "artifacts": [
                {"file": name, "sha256": digest} for name, digest in fingerprints.items()
            ],
            "environment": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scipy": __import__("scipy").__version__,
                "scikit_learn": __import__("sklearn").__version__,
                "optuna": __import__("optuna").__version__,
            },
        }
        save_json(manifest, artifacts_dir / "module2_manifest.json")

        _write_markdown_report(
            reports_dir / "module2_report.md",
            config,
            fingerprint,
            benchmark,
            tuning,
            final_model,
            final_params,
            final_profiles,
            final_matrix,
            timings,
        )

    summary = {
        "selected_model": final_model.key,
        "benchmark_recall": benchmark.report["selection"]["ranking"][0]["recall_mean"],
        "tuned": tuning is not None,
        "best_value": None if tuning is None else tuning.best_value,
        "profile_users": int(final_profiles.matrix.shape[0]),
        "catalogue_items": int(final_matrix.shape[0]),
        "feature_dimension": int(final_matrix.shape[1]),
        "timings_seconds": timings,
        "artifacts_dir": str(artifacts_dir),
        "reports_dir": str(reports_dir),
    }

    _LOGGER.info(
        "Module 2 complete | model=%s | profiles=%s x %s | tuned=%s",
        final_model.key,
        f"{final_profiles.matrix.shape[0]:,}",
        f"{final_profiles.matrix.shape[1]:,}",
        tuning is not None,
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the Module 2 stage."""
    parser = argparse.ArgumentParser(description="Run Module 2 training and tuning.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: model/config/config.yaml).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point.

    Returns:
        ``0`` on success, ``2`` on a configuration, artifact or runtime failure.
    """
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    reports_dir = ensure_directory(config.paths.reports_module2)
    configure_logging(
        config.logging.level,
        reports_dir / "module2.log",
    )
    set_global_seed(config.seed)

    try:
        run_module2(config)
    except (ConfigError, Module2Error, ValueError, KeyError, RuntimeError) as error:
        _LOGGER.error("Module 2 aborted: %s", error)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
