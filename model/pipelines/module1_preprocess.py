"""Module 1 entry point: preprocessing, deduplication, splitting and features.

Pipeline order (each step is a separate, testable function):

``load``        Read the two parquet files and enforce the schema contract.
``validate``    Run hard/soft data checks; stop on contract violations.
``deduplicate`` Collapse duplicates and retain ``interaction_count``.
``split``       Chronological train / validation / test with audit report.
``text``        Namespaced metadata tokens, TF-IDF fitted on train items only.
``price``       Train-median imputation, log1p, RobustScaler fitted on train items.
``fuse``        Sparse CSR concatenation and row normalisation.
``persist``     Write split frames, matrices, transformers and manifests.

Usage (from the project root)::

    venv\\Scripts\\python.exe -m model.pipelines.module1_preprocess

Every artifact written here is deterministic for a given input file and
configuration, and the stage manifest records checksums so Modules 2 and 3 can
prove which matrix they consumed.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from model.config.config_schema import (
    ConfigError,
    Module1Config,
    config_hash,
    config_to_dict,
    load_config,
)
from model.core.logger import configure_logging, get_logger
from model.core.paths import ensure_directory, resolve_path
from model.core.persistence import (
    save_joblib,
    save_json,
    save_parquet,
    save_sparse,
    sha256_file,
)
from model.core.seeding import set_global_seed
from model.data.deduplication import deduplicate_events
from model.data.loader import DataContractError, load_inputs
from model.data.temporal_split import SplitResult, split_events
from model.data.validators import (
    DataValidationError,
    assert_no_failures,
    validate_events,
    validate_products,
)
from model.features.item_matrix import ItemMatrixArtifacts, fuse_features
from model.features.price_features import (
    PriceFeatureArtifacts,
    fit_price_features,
    transform_prices,
)
from model.features.text_features import (
    TextFeatureArtifacts,
    build_metadata_text,
    fit_transform_text,
)

_LOGGER = get_logger(__name__)


class _Timer:
    """Small context manager that records the duration of a pipeline stage."""

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


def _ensure_layout(config: Module1Config) -> tuple[Path, Path]:
    """Create the artifact and report folders and return their paths."""
    artifacts_dir = ensure_directory(config.paths.artifacts)
    reports_dir = ensure_directory(config.paths.reports)
    ensure_directory(artifacts_dir / "splits")
    return artifacts_dir, reports_dir


def _persist_splits(split_result: SplitResult, artifacts_dir: Path) -> dict[str, Path]:
    """Write one parquet file per split and return their paths."""
    targets: dict[str, Path] = {}
    for name, frame in split_result.splits.items():
        path = artifacts_dir / "splits" / f"{name}.parquet"
        save_parquet(frame, path)
        targets[name] = path
        _LOGGER.info("Saved %s split: %s rows -> %s", name, f"{len(frame):,}", path.name)
    return targets


def _write_markdown_report(
    path: Path,
    split_result: SplitResult,
    text_artifacts: TextFeatureArtifacts,
    price_artifacts: PriceFeatureArtifacts,
    item_artifacts: ItemMatrixArtifacts,
    config_fingerprint: str,
) -> Path:
    """Render a human-readable summary for the thesis appendix."""
    report = split_result.report
    split_rows = []
    for name, summary in report["splits"].items():
        split_rows.append(
            f"| {name} | {summary['events']:,} | {summary['share_of_events_pct']:.2f}% | "
            f"{summary['users']:,} | {summary['items']:,} | "
            f"{summary['start'] or '-'} | {summary['end'] or '-'} |"
        )

    cold_rows = []
    for name, summary in report["cold_start_vs_train"].items():
        cold_rows.append(
            f"| {name} | {summary['users_unseen_in_train']:,} "
            f"({summary['users_unseen_in_train_pct']:.2f}%) | "
            f"{summary['items_unseen_in_train']:,} "
            f"({summary['items_unseen_in_train_pct']:.2f}%) | "
            f"{summary['events_on_unseen_items_pct']:.2f}% |"
        )

    text_stats = text_artifacts.stats
    price_stats = price_artifacts.to_dict()
    matrix_stats = item_artifacts.stats

    # Computed outside the f-string so the expression does not have to nest
    # quotes inside a triple-quoted template.
    tokens_preview = ", ".join(text_stats["most_common_tokens"][:12])
    split_rows_text = chr(10).join(split_rows)
    cold_rows_text = chr(10).join(cold_rows)

    # Centering is disabled by default; ``scaler.center_`` is then ``None`` and
    # must not be formatted with ``:.4f``.
    center = price_stats["scaler_center"]
    centering_text = "disabled" if center is None else f"{center:.4f}"
    scale = price_stats["scaler_scale"]
    scale_text = "n/a" if scale is None else f"{scale:.4f}"

    content = f"""# Module 1 - Preprocessing report

Generated: {datetime.now(timezone.utc).isoformat()}
Configuration fingerprint: `{config_fingerprint}`

## Splits

| split | events | share | users | items | start | end |
|---|---:|---:|---:|---:|---|---|
{split_rows_text}

Cutoffs: train < `{report['cutoffs']['train_end']}` <= val < `{report['cutoffs']['val_end']}` <= test.

## Cold start relative to train

| split | users unseen | items unseen | events on unseen items |
|---|---:|---:|---:|
{cold_rows_text}

## Text features (TF-IDF, fitted on train items only)

- Vocabulary size: {text_stats['vocabulary_size']:,}
- Training documents: {text_stats['train_documents']:,} ({text_stats['train_documents_with_text']:,} with usable text)
- Catalogue rows without in-vocabulary tokens: {text_stats['zero_text_rows_catalogue']:,} ({text_stats['zero_text_rows_catalogue_pct']:.2f}%)
- Most frequent tokens: {tokens_preview}

## Price features (fitted on train-linked products only)

- Imputation value (median): {price_stats['impute_value']:.4f}
- log1p applied: {price_stats['log_transform']}
- RobustScaler centering: {centering_text}
- RobustScaler center / scale: {centering_text} / {scale_text}
- Missing prices in training slice: {price_stats['train_missing_prices']:,}

## Fused item matrix

- Shape: {matrix_stats['items']:,} products x {matrix_stats['features']:,} features
- Non-zero entries: {matrix_stats['nnz']:,} (density {matrix_stats['density']:.6f})
- Rows without signal: {matrix_stats['zero_rows']:,} ({matrix_stats['zero_rows_pct']:.2f}%)
- Mean price share of row norm: {matrix_stats['mean_price_share']:.4f}
- Rows L2-normalised: {matrix_stats['row_normalised']}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _LOGGER.info("Wrote report: %s", path)
    return path


def run_module1(config: Module1Config) -> dict[str, Any]:
    """Execute the complete preprocessing stage.

    Args:
        config: Validated Module 1 configuration.

    Returns:
        A summary dictionary describing counts, timings and artifact paths.

    Raises:
        DataContractError: If an input file or column is missing.
        DataValidationError: If a hard data-quality check fails.
        ValueError: If the configuration makes a stage impossible (for example an
            empty training vocabulary).
    """
    artifacts_dir, reports_dir = _ensure_layout(config)
    timings: dict[str, float] = {}
    timestamp_column = config.data.timestamp_column

    # ---- Load -----------------------------------------------------------
    with _Timer(timings, "load"):
        products, events = load_inputs(config)

    # ---- Validate -------------------------------------------------------
    with _Timer(timings, "validate"):
        product_report = validate_products(products, config)
        event_report = validate_events(events, products, config)
        assert_no_failures(product_report)
        assert_no_failures(event_report)
        for report in (product_report, event_report):
            for check in report.warnings:
                _LOGGER.warning("%s: %s", check.name, check.detail)

    # ---- Deduplicate ----------------------------------------------------
    with _Timer(timings, "deduplicate"):
        deduplicated, dedup_stats = deduplicate_events(events, config)
        del events  # release the raw frame before splitting

    # ---- Temporal split -------------------------------------------------
    with _Timer(timings, "split"):
        split_result = split_events(deduplicated, config)
        del deduplicated
        split_paths = _persist_splits(split_result, artifacts_dir)

    train_items = set(split_result.splits["train"]["product_id"].unique())

    # ---- Text features --------------------------------------------------
    with _Timer(timings, "text_features"):
        metadata_text = build_metadata_text(products, config)
        # The mask aligns positionally with the catalogue order used to persist
        # the item matrix, so row i of the matrix is always product i of
        # ``item_index.parquet``.
        train_mask = products["product_id"].isin(train_items).reset_index(drop=True)
        metadata_text = metadata_text.reset_index(drop=True)
        text_artifacts = fit_transform_text(metadata_text, train_mask, config)

    # ---- Price features -------------------------------------------------
    with _Timer(timings, "price_features"):
        price_artifacts = fit_price_features(
            products.loc[train_mask, config.price.column], config
        )
        price_vector = transform_prices(products[config.price.column], price_artifacts, config)

    # ---- Fusion ---------------------------------------------------------
    with _Timer(timings, "fuse_features"):
        item_artifacts = fuse_features(text_artifacts.matrix, price_vector, config)

    # ---- Persist --------------------------------------------------------
    with _Timer(timings, "persist"):
        item_index = pd.DataFrame(
            {
                "product_id": products["product_id"].to_numpy(),
                "item_row": range(len(products)),
            }
        )
        save_parquet(item_index, artifacts_dir / "item_index.parquet")
        save_sparse(item_artifacts.matrix, artifacts_dir / "item_matrix.npz")
        save_joblib(text_artifacts.vectorizer, artifacts_dir / "tfidf_vectorizer.joblib")
        save_joblib(price_artifacts, artifacts_dir / "price_transformer.joblib")

        save_json(config_to_dict(config), artifacts_dir / "config.resolved.json")
        save_json(
            {
                "deduplication": dedup_stats.to_dict(),
                "split": split_result.report,
                "validation": {
                    "products": product_report.to_dict(),
                    "events": event_report.to_dict(),
                },
            },
            artifacts_dir / "split_manifest.json",
        )
        save_json(
            {
                "text": text_artifacts.stats,
                "price": price_artifacts.to_dict(),
                "item_matrix": item_artifacts.stats,
            },
            artifacts_dir / "feature_manifest.json",
        )

        fingerprints = {
            "item_matrix.npz": sha256_file(artifacts_dir / "item_matrix.npz"),
            "item_index.parquet": sha256_file(artifacts_dir / "item_index.parquet"),
            "tfidf_vectorizer.joblib": sha256_file(artifacts_dir / "tfidf_vectorizer.joblib"),
            "price_transformer.joblib": sha256_file(artifacts_dir / "price_transformer.joblib"),
        }

        fingerprint = config_hash(config)
        manifest = {
            "stage": "module1_preprocess",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": fingerprint,
            "seed": config.seed,
            "catalogue_products": int(len(products)),
            "train_products_with_interactions": len(train_items),
            "timestamp_column": timestamp_column,
            "timings_seconds": timings,
            "artifacts": [
                {"file": name, "sha256": digest} for name, digest in fingerprints.items()
            ],
            "environment": {
                "pandas": pd.__version__,
                "numpy": __import__("numpy").__version__,
                "scipy": __import__("scipy").__version__,
                "scikit_learn": __import__("sklearn").__version__,
            },
        }
        save_json(manifest, artifacts_dir / "module1_manifest.json")

        _write_markdown_report(
            reports_dir / "module1_report.md",
            split_result,
            text_artifacts,
            price_artifacts,
            item_artifacts,
            fingerprint,
        )

    summary = {
        "split_rows": {name: int(len(frame)) for name, frame in split_result.splits.items()},
        "catalogue_products": int(len(products)),
        "item_matrix_shape": list(item_artifacts.matrix.shape),
        "vocabulary_size": int(text_artifacts.matrix.shape[1] - 1),
        "timings_seconds": timings,
        "artifacts_dir": str(artifacts_dir),
        "reports_dir": str(reports_dir),
    }

    _LOGGER.info(
        "Module 1 complete | train=%s val=%s test=%s | matrix=%s x %s",
        f"{summary['split_rows']['train']:,}",
        f"{summary['split_rows']['val']:,}",
        f"{summary['split_rows']['test']:,}",
        f"{summary['item_matrix_shape'][0]:,}",
        f"{summary['item_matrix_shape'][1]:,}",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the Module 1 stage."""
    parser = argparse.ArgumentParser(description="Run Module 1 preprocessing.")
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
        ``0`` on success, ``2`` on a configuration or data-contract failure.
    """
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    reports_dir = ensure_directory(config.paths.reports)
    configure_logging(
        config.logging.level,
        reports_dir / (config.logging.file_name or "module1.log"),
    )
    set_global_seed(config.seed)

    try:
        run_module1(config)
    except (DataContractError, DataValidationError, ValueError, KeyError) as error:
        _LOGGER.error("Module 1 aborted: %s", error)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
