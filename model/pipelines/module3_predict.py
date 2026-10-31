"""Module 3 entry point: candidate scoring, ranking and hidden-test evaluation.

Pipeline order
--------------
``load``       Read Module 2's catalogue matrix, profiles, seen-item mask and
               identifier maps, plus the evaluation split and the catalogue.
``select``     Keep users that have a profile, enough evaluation history and at
               least one target that was not already seen in training.
``rank``       Cosine-score every selected user against the catalogue, apply
               the business-rule masks and keep the top-K items.
``evaluate``   Score the lists against the hidden split with HitRate, Recall,
               Precision and MRR at every configured cut-off.
``diagnose``   Optionally compute diversity and novelty diagnostics.
``persist``    Write recommendations, metrics, manifest and a markdown report.

Usage (from the project root)::

    venv\\\\Scripts\\\\python.exe -m model.pipelines.module3_predict

The stage reads the **Module 2** ``item_matrix.npz`` on purpose. Module 2 may
have changed the TF-IDF hyperparameters during tuning, so the Module 1 matrix is
not guaranteed to share the profile feature space.
"""

from __future__ import annotations

import argparse
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
    load_config,
)
from model.core.logger import configure_logging, get_logger
from model.core.paths import ensure_directory, resolve_path
from model.core.persistence import (
    load_json,
    load_parquet,
    load_sparse,
    save_json,
    save_parquet,
    sha256_file,
)
from model.core.seeding import set_global_seed
from model.evaluation.diversity import (
    category_lookup,
    compute_diversity_report,
    item_popularity,
)
from model.evaluation.evaluator import evaluate_ranking_result
from model.profiles.user_profiles import sample_user_keys
from model.ranking.filters import (
    build_ground_truth,
    exclude_seen_from_truth,
    item_signal_mask,
)
from model.ranking.scorer import RankingResult, rank_top_k

_LOGGER = get_logger(__name__)


class Module3Error(RuntimeError):
    """Raised when a Module 2 artifact is missing or inconsistent."""


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
class _Module2Bundle:
    """Everything Module 3 consumes from Module 2."""

    item_matrix: sp.csr_matrix
    profiles: sp.csr_matrix
    seen_items: sp.csr_matrix
    item_index: pd.DataFrame
    user_index: pd.DataFrame
    best_params: dict[str, Any]
    model_spec: dict[str, Any]
    manifest: dict[str, Any]


def _load_module2(config: PipelineConfig) -> _Module2Bundle:
    """Load and cross-check the Module 2 artifact contract.

    Args:
        config: Validated pipeline configuration.

    Returns:
        The loaded bundle.

    Raises:
        Module3Error: If an artifact is missing or the shapes disagree.
    """
    artifacts = resolve_path(config.paths.artifacts_module2)
    required = (
        artifacts / "item_matrix.npz",
        artifacts / "profiles.npz",
        artifacts / "seen_items.npz",
        artifacts / "item_index.parquet",
        artifacts / "user_index.parquet",
        artifacts / "best_params.json",
        artifacts / "model_spec.json",
        artifacts / "module2_manifest.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise Module3Error(
            "Module 2 artifacts are missing:\n  - "
            + "\n  - ".join(missing)
            + "\nRun: venv\\Scripts\\python.exe -m model.pipelines.module2_train"
        )

    item_matrix = sp.csr_matrix(load_sparse(artifacts / "item_matrix.npz"))
    profiles = sp.csr_matrix(load_sparse(artifacts / "profiles.npz"))
    seen_items = sp.csr_matrix(load_sparse(artifacts / "seen_items.npz"))
    item_index = load_parquet(artifacts / "item_index.parquet")
    user_index = load_parquet(artifacts / "user_index.parquet")
    best_params = load_json(artifacts / "best_params.json")
    model_spec = load_json(artifacts / "model_spec.json")
    manifest = load_json(artifacts / "module2_manifest.json")

    if item_matrix.shape[0] != len(item_index):
        raise Module3Error(
            "item_matrix rows and item_index length disagree: "
            f"{item_matrix.shape[0]} vs {len(item_index)}."
        )
    if profiles.shape[0] != len(user_index):
        raise Module3Error(
            "profiles rows and user_index length disagree: "
            f"{profiles.shape[0]} vs {len(user_index)}."
        )
    if profiles.shape[1] != item_matrix.shape[1]:
        raise Module3Error(
            "profiles and item_matrix have different feature dimensions: "
            f"{profiles.shape[1]} vs {item_matrix.shape[1]}."
        )
    if seen_items.shape != (profiles.shape[0], item_matrix.shape[0]):
        raise Module3Error(
            "seen_items must have shape (profiles, items) = "
            f"{(profiles.shape[0], item_matrix.shape[0])}, got {seen_items.shape}."
        )
    if user_index["user_key"].duplicated().any():
        raise Module3Error("user_index contains duplicate user_key values.")
    if item_index["product_id"].duplicated().any():
        raise Module3Error("item_index contains duplicate product_id values.")

    current = config_hash(config)
    stored = manifest.get("config_fingerprint")
    if stored and stored != current:
        _LOGGER.warning(
            "The current configuration (%s...) differs from the one that produced "
            "Module 2 (%s...). Metrics may not be reproducible.",
            current[:12],
            str(stored)[:12],
        )

    _LOGGER.info(
        "Module 2 bundle: model=%s | profiles=%s x %s | items=%s | seen nnz=%s",
        model_spec.get("model", {}).get("key"),
        f"{profiles.shape[0]:,}",
        f"{profiles.shape[1]:,}",
        f"{item_matrix.shape[0]:,}",
        f"{seen_items.nnz:,}",
    )

    return _Module2Bundle(
        item_matrix=item_matrix,
        profiles=profiles,
        seen_items=seen_items,
        item_index=item_index,
        user_index=user_index,
        best_params=best_params,
        model_spec=model_spec,
        manifest=manifest,
    )


def _user_rows(user_keys: np.ndarray, user_index: pd.DataFrame) -> np.ndarray:
    """Map user identifiers to profile row positions.

    Args:
        user_keys: Identifiers to map.
        user_index: ``user_key`` -> ``user_row`` mapping from Module 2.

    Returns:
        Integer row positions aligned with ``user_keys``.

    Raises:
        Module3Error: If a user is absent from the profile matrix.
    """
    positions = pd.Series(
        user_index["user_row"].to_numpy(dtype=np.int64),
        index=user_index["user_key"].to_numpy(),
    )
    rows = positions.reindex(user_keys).to_numpy()
    if np.isnan(rows).any():
        raise Module3Error("Some selected users are missing from the profile matrix.")
    return rows.astype(np.int64)


def _select_users(
    evaluation: pd.DataFrame,
    user_index: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select users that can be scored and report cold-start coverage.

    Args:
        evaluation: Evaluation split (validation or test).
        user_index: Profile row mapping from Module 2.
        config: Validated pipeline configuration.

    Returns:
        Tuple ``(sampled user keys, diagnostics)``.
    """
    counts = evaluation.groupby("user_key", observed=True).size()
    eligible = counts.index[
        counts >= config.evaluation.min_test_interactions
    ].to_numpy(dtype=np.int64)

    profiled = np.intersect1d(
        user_index["user_key"].to_numpy(dtype=np.int64),
        eligible,
        assume_unique=False,
    )

    total_users = int(evaluation["user_key"].nunique())
    profiled_set = set(user_index["user_key"].tolist())
    cold_mask = ~evaluation["user_key"].isin(profiled_set)

    limit = config.module3.max_users if config.module3.max_users > 0 else len(profiled)
    user_keys = sample_user_keys(profiled, max(limit, 1), config.seed + 3)

    diagnostics: dict[str, Any] = {
        "split": config.module3.evaluation_split,
        "events": int(len(evaluation)),
        "users_in_split": total_users,
        "users_meeting_min_test": int(len(eligible)),
        "users_with_profile": int(len(profiled)),
        "cold_start_users_without_profile": int(total_users - len(profiled)),
        "cold_start_users_pct": round(
            100.0 * (total_users - len(profiled)) / total_users, 4
        )
        if total_users
        else 0.0,
        "cold_start_events_without_profile": int(cold_mask.sum()),
        "cold_start_events_pct": round(100.0 * float(cold_mask.mean()), 4)
        if len(evaluation)
        else 0.0,
        "users_sampled": int(len(user_keys)),
        "max_users": int(config.module3.max_users),
    }

    _LOGGER.info(
        "User selection: %s split users | %s have a profile | %s sampled | "
        "%s cold-start users (%.2f%%)",
        f"{total_users:,}",
        f"{len(profiled):,}",
        f"{len(user_keys):,}",
        f"{diagnostics['cold_start_users_without_profile']:,}",
        diagnostics["cold_start_users_pct"],
    )
    return user_keys, diagnostics


def _unseen_item_diagnostics(
    evaluation: pd.DataFrame,
    train: pd.DataFrame,
) -> dict[str, Any]:
    """Quantify evaluation interactions whose item has no training interaction.

    Even when the user is warm, an item absent from training was never seen by
    the vectorizer; this diagnostic makes that share explicit.

    Args:
        evaluation: Evaluation split.
        train: Training split.

    Returns:
        JSON-ready counts and shares.
    """
    train_items = pd.Index(train["product_id"].unique())
    unseen_mask = ~evaluation["product_id"].isin(train_items)
    return {
        "evaluation_items_absent_from_train": int(
            evaluation.loc[unseen_mask, "product_id"].nunique()
        ),
        "evaluation_events_absent_from_train": int(unseen_mask.sum()),
        "evaluation_events_absent_from_train_pct": round(
            100.0 * float(unseen_mask.mean()), 4
        )
        if len(evaluation)
        else 0.0,
    }


def _export_recommendations(
    ranking: RankingResult,
    user_keys: np.ndarray,
    item_index: pd.DataFrame,
    export_users: int,
    depth: int,
) -> pd.DataFrame:
    """Flatten the top-K block into a tidy recommendation table.

    Args:
        ranking: Ranking output aligned with ``user_keys``.
        user_keys: Scored user identifiers.
        item_index: ``product_id`` -> ``item_row`` mapping.
        export_users: Number of users to export (``0`` exports all).
        depth: Maximum list length to write.

    Returns:
        DataFrame with ``user_key``, ``rank``, ``item_row``, ``product_id`` and
        ``score``, ordered by user and rank.
    """
    limit = len(user_keys) if export_users == 0 else min(export_users, len(user_keys))
    cut = min(depth, ranking.item_rows.shape[1])

    items = ranking.item_rows[:limit, :cut]
    scores = ranking.scores[:limit, :cut]
    user_positions, rank_positions = np.nonzero(items >= 0)

    frame = pd.DataFrame(
        {
            "user_key": np.asarray(user_keys)[user_positions],
            "rank": rank_positions + 1,
            "item_row": items[user_positions, rank_positions],
            "score": scores[user_positions, rank_positions],
        }
    )
    products = item_index["product_id"].to_numpy()
    frame["product_id"] = products[frame["item_row"].to_numpy(dtype=np.int64)]
    frame = frame.sort_values(["user_key", "rank"], kind="mergesort").reset_index(
        drop=True
    )
    return frame


def _write_markdown_report(
    path: Path,
    config: PipelineConfig,
    fingerprint: str,
    bundle: _Module2Bundle,
    evaluation: dict[str, Any],
    diagnostics: dict[str, Any],
    unseen_items: dict[str, Any],
    diversity: dict[str, Any] | None,
    recommendations: pd.DataFrame,
    timings: dict[str, float],
) -> Path:
    """Render the Module 3 summary for the thesis appendix."""
    k_values = sorted(
        (int(key) for key in evaluation["by_k"]), key=int
    )
    metric_rows = [
        f"| {evaluation['by_k'][str(k)]['k']} | "
        f"{evaluation['by_k'][str(k)]['users']:,} | "
        f"{evaluation['by_k'][str(k)]['hit_rate']:.4f} | "
        f"{evaluation['by_k'][str(k)]['recall']:.4f} | "
        f"{evaluation['by_k'][str(k)]['precision']:.4f} | "
        f"{evaluation['by_k'][str(k)]['mrr']:.4f} | "
        f"{evaluation['by_k'][str(k)]['coverage']:.4f} |"
        for k in k_values
    ]

    if diversity is None:
        diversity_text = "- Diversity diagnostics disabled in the configuration."
    else:
        diversity_text = (
            f"- Users analysed: {diversity['users']:,}\n"
            f"- Intra-list diversity (1 - mean cosine): {diversity['intra_list_diversity']:.4f}\n"
            f"- Catalogue coverage: {diversity['catalogue_coverage']:.4f}\n"
            f"- Distinct categories per list: {diversity['mean_distinct_categories']:.2f} "
            f"(ratio {diversity['mean_category_ratio']:.4f})\n"
            f"- Distinct brands per list: {diversity['mean_distinct_brands']:.2f} "
            f"(ratio {diversity['mean_brand_ratio']:.4f})\n"
            f"- Novelty: {diversity['novelty_bits']:.3f} bits"
        )

    sample = recommendations.head(10)
    sample_rows = [
        f"| {int(row.user_key)} | {int(row['rank'])} | {int(row.product_id)} | "
        f"{float(row.score):.4f} |"
        for _, row in sample.iterrows()
    ]

    content = f"""# Module 3 - Ranking and hidden-test evaluation

Generated: {datetime.now(timezone.utc).isoformat()}
Configuration fingerprint: `{fingerprint}`
Selected model: **{bundle.model_spec['model']['label']}**
Module 2 fingerprint: `{bundle.manifest.get('config_fingerprint')}`

## Evaluation protocol

- Evaluation split: `{config.module3.evaluation_split}`
- Events: {diagnostics['events']:,}
- Users in split: {diagnostics['users_in_split']:,}
- Users with a profile: {diagnostics['users_with_profile']:,}
- Users sampled: {diagnostics['users_sampled']:,}
- Users scored after seen-item filtering: {evaluation['users_evaluated']:,}
- Users without an unseen target: {evaluation['users_without_relevant_items']:,}
- Cold-start users (no training profile): {diagnostics['cold_start_users_without_profile']:,} ({diagnostics['cold_start_users_pct']:.2f}%)
- Cold-start events: {diagnostics['cold_start_events_without_profile']:,} ({diagnostics['cold_start_events_pct']:.2f}%)
- Evaluation items absent from training: {unseen_items['evaluation_items_absent_from_train']:,}
- Evaluation events absent from training: {unseen_items['evaluation_events_absent_from_train']:,} ({unseen_items['evaluation_events_absent_from_train_pct']:.2f}%)
- Seen items excluded from candidates and labels: {config.evaluation.exclude_seen_items}

## Ranking metrics

| K | users | HitRate@K | Recall@K | Precision@K | MRR@K | Coverage@K |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

## Diversity and novelty

{diversity_text}

## Sample recommendations

| user_key | rank | product_id | cosine |
|---:|---:|---:|---:|
{chr(10).join(sample_rows)}

## Timings

{chr(10).join(f"- {stage}: {seconds:.2f}s" for stage, seconds in timings.items())}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _LOGGER.info("Wrote report: %s", path)
    return path


def run_module3(config: PipelineConfig) -> dict[str, Any]:
    """Execute Module 3 end to end.

    Args:
        config: Validated pipeline configuration.

    Returns:
        Summary dictionary with metrics, counts and timings.

    Raises:
        Module3Error: If a Module 2 artifact is missing or inconsistent.
        ValueError: If no user can be scored.
    """
    artifacts_dir = ensure_directory(config.paths.artifacts_module3)
    reports_dir = ensure_directory(config.paths.reports_module3)
    timings: dict[str, float] = {}
    fingerprint = config_hash(config)

    with _Timer(timings, "load"):
        bundle = _load_module2(config)
        split_name = config.module3.evaluation_split
        evaluation_split = load_parquet(
            resolve_path(config.paths.artifacts) / "splits" / f"{split_name}.parquet"
        )
        train_split = load_parquet(
            resolve_path(config.paths.artifacts) / "splits" / "train.parquet"
        )

    with _Timer(timings, "select"):
        user_keys, diagnostics = _select_users(
            evaluation_split, bundle.user_index, config
        )
        unseen_items = _unseen_item_diagnostics(evaluation_split, train_split)

        sampled_rows = _user_rows(user_keys, bundle.user_index)
        sampled_seen = bundle.seen_items[sampled_rows]

        truth = build_ground_truth(evaluation_split, bundle.item_index, user_keys)
        seen_filter: dict[str, Any] = {}
        if config.evaluation.exclude_seen_items:
            truth, seen_filter = exclude_seen_from_truth(
                truth, sampled_seen, user_keys
            )

        scorable_keys = np.array(sorted(truth), dtype=np.int64)
        if not len(scorable_keys):
            raise ValueError(
                "No evaluation label remains after removing items seen in "
                "training; there is nothing to rank."
            )
        scorable_rows = _user_rows(scorable_keys, bundle.user_index)

    k_values = tuple(int(k) for k in config.evaluation.k_values)
    depth = max(max(k_values), int(config.module3.export_top_k))
    depth = min(depth, bundle.item_matrix.shape[0])

    with _Timer(timings, "rank"):
        ranking = rank_top_k(
            bundle.profiles[scorable_rows],
            bundle.item_matrix,
            k=depth,
            seen_matrix=bundle.seen_items[scorable_rows],
            allowed_items=item_signal_mask(bundle.item_matrix),
            batch_size=config.evaluation.scoring_batch_size,
        )

    with _Timer(timings, "evaluate"):
        evaluation_report = evaluate_ranking_result(
            ranking,
            truth,
            scorable_keys,
            k_values,
            bundle.item_matrix.shape[0],
        )

    diversity_payload: dict[str, Any] | None = None
    if config.module3.compute_diversity:
        with _Timer(timings, "diversity"):
            products = pd.read_parquet(resolve_path(config.paths.products))
            popularity = item_popularity(train_split, bundle.item_index)
            categories, brands = category_lookup(products, bundle.item_index)
            diversity_payload = compute_diversity_report(
                ranking.item_rows,
                bundle.item_matrix,
                popularity,
                categories=categories,
                brands=brands,
                max_users=config.module3.diversity_users,
            ).to_dict()

    with _Timer(timings, "persist"):
        recommendations = _export_recommendations(
            ranking,
            scorable_keys,
            bundle.item_index,
            config.module3.export_users,
            config.module3.export_top_k,
        )
        save_parquet(recommendations, artifacts_dir / "recommendations.parquet")

        protocol = {
            "evaluation_split": split_name,
            "k_values": list(k_values),
            "ranking_depth": int(depth),
            "export_top_k": int(config.module3.export_top_k),
            "scoring_batch_size": int(config.evaluation.scoring_batch_size),
            "exclude_seen_items": bool(config.evaluation.exclude_seen_items),
            "min_train_interactions": int(config.evaluation.min_train_interactions),
            "min_test_interactions": int(config.evaluation.min_test_interactions),
            "sampling_seed": int(config.seed + 3),
            "user_selection": diagnostics,
            "cold_start": unseen_items,
            "seen_filter": seen_filter,
            "scored_users": int(len(scorable_keys)),
            "ranked_items": int(bundle.item_matrix.shape[0]),
            "profile_features": int(bundle.profiles.shape[1]),
            "selected_model": bundle.model_spec.get("model", {}).get("key"),
            "tuned_parameters": bundle.best_params.get("parameters"),
        }
        save_json(
            {"protocol": protocol, "metrics": evaluation_report},
            artifacts_dir / "evaluation.json",
        )
        if diversity_payload is not None:
            save_json(diversity_payload, artifacts_dir / "diversity.json")

        fingerprints = {
            "recommendations.parquet": sha256_file(
                artifacts_dir / "recommendations.parquet"
            ),
            "evaluation.json": sha256_file(artifacts_dir / "evaluation.json"),
        }
        if diversity_payload is not None:
            fingerprints["diversity.json"] = sha256_file(
                artifacts_dir / "diversity.json"
            )

        manifest = {
            "stage": "module3_predict",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": fingerprint,
            "module2_config_fingerprint": bundle.manifest.get("config_fingerprint"),
            "module1_config_fingerprint": bundle.manifest.get(
                "module1_config_fingerprint"
            ),
            "seed": config.seed,
            "evaluation_split": split_name,
            "selected_model": bundle.model_spec.get("model", {}).get("key"),
            "scored_users": int(len(scorable_keys)),
            "users_sampled": int(len(user_keys)),
            "recommendation_users_exported": int(
                recommendations["user_key"].nunique()
            ),
            "recommendation_rows": int(len(recommendations)),
            "metrics": evaluation_report["by_k"],
            "diversity": diversity_payload,
            "timings_seconds": timings,
            "artifacts": [
                {"file": name, "sha256": digest}
                for name, digest in fingerprints.items()
            ],
            "environment": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scipy": __import__("scipy").__version__,
            },
        }
        save_json(manifest, artifacts_dir / "module3_manifest.json")

        _write_markdown_report(
            reports_dir / "module3_report.md",
            config,
            fingerprint,
            bundle,
            evaluation_report,
            diagnostics,
            unseen_items,
            diversity_payload,
            recommendations,
            timings,
        )

    summary = {
        "evaluation_split": split_name,
        "selected_model": bundle.model_spec.get("model", {}).get("key"),
        "scored_users": int(len(scorable_keys)),
        "users_sampled": int(len(user_keys)),
        "metrics": evaluation_report["by_k"],
        "diversity": diversity_payload,
        "recommendation_rows": int(len(recommendations)),
        "timings_seconds": timings,
        "artifacts_dir": str(artifacts_dir),
        "reports_dir": str(reports_dir),
    }

    _LOGGER.info(
        "Module 3 complete | model=%s | users scored=%s | recall@%s=%.4f",
        summary["selected_model"],
        f"{len(scorable_keys):,}",
        max(k_values),
        evaluation_report["by_k"][str(max(k_values))]["recall"],
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the Module 3 stage."""
    parser = argparse.ArgumentParser(description="Run Module 3 ranking and evaluation.")
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

    reports_dir = ensure_directory(config.paths.reports_module3)
    configure_logging(config.logging.level, reports_dir / "module3.log")
    set_global_seed(config.seed)

    try:
        run_module3(config)
    except (ConfigError, Module3Error, ValueError, KeyError, RuntimeError) as error:
        _LOGGER.error("Module 3 aborted: %s", error)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
