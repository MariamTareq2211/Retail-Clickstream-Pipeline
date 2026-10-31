"""Synthetic end-to-end smoke test for Module 1.

Run from the project root:

    venv\\Scripts\\python.exe -m model.tests.smoke_module1

The test builds a small, fully synthetic catalogue and interaction log in a
temporary directory, runs the real Module 1 pipeline against it, and asserts the
properties that matter most:

* deduplication collapses only exact timestamped duplicates;
* the chronological split is disjoint and lands on the expected volumes;
* TF-IDF is fitted on training items only (a token that exists solely in a
  test-only product must not enter the vocabulary);
* price imputation uses the training median;
* the fused matrix is sparse, row-normalised, and keeps a signal for products
  without any metadata.

No real data is read and no repository artifact is written: everything happens
under ``tempfile.TemporaryDirectory``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from model.config.config_schema import ConfigError, load_config
from model.core.logger import configure_logging
from model.core.persistence import load_joblib, load_json, load_sparse
from model.pipelines.module1_preprocess import run_module1

USERS = 100
PRODUCTS = 30
TRAIN_DATES = ("2023-01-01", "2023-01-02", "2023-01-03")
VAL_DATE = "2023-01-04"
TEST_DATE = "2023-01-05"


def _build_products() -> pd.DataFrame:
    """Create a catalogue with known/unknown metadata and one null price."""
    rows = []
    for index in range(PRODUCTS):
        level1 = ("electronics", "apparel", "Unknown")[index % 3]
        level2 = ("phones", "shoes", "Unknown")[index % 3]
        level3 = "Unknown" if index % 4 else "flagship"
        brand = f"brand{index % 5}"
        if index == 29:
            # Leakage probe: this brand exists only in the test period.
            brand = "zebra"
            level1, level2, level3 = "Unknown", "Unknown", "Unknown"
        if index == 4:
            # Metadata-free product: no usable text at all.
            brand, level1, level2, level3 = "Unknown", "Unknown", "Unknown", "Unknown"
        price = float(10 * index + 5)
        if index == 5:
            price = np.nan  # exercise train-median imputation
        rows.append(
            {
                "product_id": 1000 + index,
                "category_code": f"{level1}.{level2}.{level3}",
                "category_level1": level1,
                "category_level2": level2,
                "category_level3": level3,
                "brand": brand,
                "base_price": price,
            }
        )
    return pd.DataFrame(rows)


def _build_events() -> pd.DataFrame:
    """Create 100 interactions per day, plus 40 exact duplicates on day one."""
    rows = []
    for user in range(1, USERS + 1):
        for date in (*TRAIN_DATES, VAL_DATE, TEST_DATE):
            if date == TEST_DATE:
                # Test items (1019-1028) never appear in train; user 1 receives
                # the zebra probe (1029) instead, so every day keeps 100 tuples.
                product_id = 1029 if user == 1 else 1019 + (user % 10)
            else:
                product_id = 1000 + (user % 19)
            event_type = 4 if user % 11 == 0 else (2 if user % 7 == 0 else 1)
            rows.append(
                {
                    "user_key": user,
                    "product_id": product_id,
                    "event_type_key": event_type,
                    "event_ts": pd.Timestamp(date, tz="UTC"),
                }
            )

    events = pd.DataFrame(rows)

    # Duplicate 20 day-one interactions twice: 40 raw rows must collapse into
    # the same 20 keys, leaving the deduplicated volume at 500.
    duplicates = events[(events["event_ts"] == pd.Timestamp(TRAIN_DATES[0], tz="UTC"))].head(20)
    events = pd.concat([events, duplicates, duplicates], ignore_index=True)
    return events


def _check(condition: bool, message: str) -> None:
    """Assert a condition and print a readable progress line."""
    if not condition:
        raise AssertionError(message)
    print(f"  [ok] {message}")


def main() -> int:
    """Run the smoke test and return the process exit code."""
    configure_logging("WARNING", None)

    with tempfile.TemporaryDirectory(prefix="module1_smoke_") as temporary:
        workdir = Path(temporary)
        products_path = workdir / "dim_product_clean.parquet"
        events_path = workdir / "fact_events_clean.parquet"

        products = _build_products()
        events = _build_events()
        products.to_parquet(products_path, index=False)
        events.to_parquet(events_path, index=False)

        config = load_config(
            overrides={
                "paths": {
                    "products": str(products_path),
                    "events": str(events_path),
                    "artifacts": str(workdir / "artifacts"),
                    "reports": str(workdir / "reports"),
                },
                "logging": {"level": "WARNING", "file_name": "smoke.log"},
                "split": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
                "text": {"tfidf": {"min_df": 1, "max_features": 50}},
            }
        )

        print("Running Module 1 on synthetic data...")
        summary = run_module1(config)

        print("Verifying outputs...")
        _check(
            summary["split_rows"] == {"train": 300, "val": 100, "test": 100},
            f"split volumes are chronological and balanced: {summary['split_rows']}",
        )

        artifacts = workdir / "artifacts"
        split_manifest = load_json(artifacts / "split_manifest.json")
        dedup = split_manifest["deduplication"]
        _check(dedup["rows_in"] == 540 and dedup["rows_out"] == 500, "dedup collapsed 40 exact duplicates")
        _check(dedup["max_interaction_count"] == 3, "repeat count retained as interaction_count")

        train = pd.read_parquet(artifacts / "splits" / "train.parquet")
        val = pd.read_parquet(artifacts / "splits" / "val.parquet")
        test = pd.read_parquet(artifacts / "splits" / "test.parquet")
        timestamp = config.data.timestamp_column
        _check(
            train[timestamp].max() < val[timestamp].min()
            and val[timestamp].max() < test[timestamp].min(),
            "splits are temporally disjoint",
        )

        feature_manifest = load_json(artifacts / "feature_manifest.json")
        vectorizer = load_joblib(artifacts / "tfidf_vectorizer.joblib")
        _check(
            "brand_zebra" not in vectorizer.vocabulary_,
            "test-only token excluded from the training vocabulary (no leakage)",
        )
        _check(
            any(token.startswith("brand_") for token in vectorizer.vocabulary_),
            "training brand tokens are present in the vocabulary",
        )

        matrix = load_sparse(artifacts / "item_matrix.npz")
        _check(matrix.shape[0] == PRODUCTS, "one matrix row per catalogue product")
        _check(
            feature_manifest["item_matrix"]["zero_rows"] == 0,
            "every product keeps a signal through the price dimension",
        )

        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1))).ravel()
        _check(bool(np.allclose(norms, 1.0, atol=1e-6)), "all item rows are L2-normalised")

        item_index = pd.read_parquet(artifacts / "item_index.parquet")
        zebra_row = int(item_index.loc[item_index["product_id"] == 1029, "item_row"].iloc[0])
        vocab_size = matrix.shape[1] - 1
        _check(
            matrix[zebra_row, :vocab_size].nnz == 0 and matrix[zebra_row, -1] != 0,
            "test-only product falls back to the price dimension",
        )

        price_transformer = load_joblib(artifacts / "price_transformer.joblib")
        train_prices = products.loc[products["product_id"] < 1019, "base_price"].dropna()
        _check(
            abs(price_transformer.impute_value - float(train_prices.median())) < 1e-9,
            f"price imputation uses the training median ({price_transformer.impute_value:.1f})",
        )

        _check((artifacts / "module1_manifest.json").exists(), "stage manifest written")

        # Configuration guardrails: both a bad ratio and a typo must be rejected.
        for bad_override in ({"split": {"train_ratio": 0.5}}, {"split": {"typo": 1}}):
            try:
                load_config(overrides=bad_override)
            except ConfigError:
                pass
            else:
                raise AssertionError(f"Invalid configuration was accepted: {bad_override}")
        print("  [ok] invalid configurations are rejected")

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
