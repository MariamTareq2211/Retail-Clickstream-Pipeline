"""Synthetic end-to-end smoke test for Module 3.

Run from the project root:

    venv\\\\Scripts\\\\python.exe -m model.tests.smoke_module3

The test reuses the shared synthetic fixture, runs Module 1, Module 2 and Module
3, and verifies:

* every configured metric cut-off is reported for the scored users;
* metric values are finite and inside their theoretical range;
* no recommendation repeats an item the user already interacted with in
  training (the business rule from Plan.md);
* the exported recommendation table respects the configured depth;
* diversity and novelty diagnostics are finite;
* the manifest fingerprints the exported artifacts.

No real data is read and no repository artifact is written.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from model.config.config_schema import load_config
from model.core.logger import configure_logging
from model.core.persistence import load_json, load_parquet
from model.pipelines.module1_preprocess import run_module1
from model.pipelines.module2_train import run_module2
from model.pipelines.module3_predict import run_module3
from model.tests.synthetic_fixture import build_data_files, path_overrides


def _check(condition: bool, message: str) -> None:
    """Assert a condition and print a progress line."""
    if not condition:
        raise AssertionError(message)
    print(f"  [ok] {message}")


def main() -> int:
    """Run the smoke test and return the process exit code."""
    configure_logging("WARNING", None)

    with tempfile.TemporaryDirectory(prefix="module3_smoke_") as temporary:
        workdir = Path(temporary)
        build_data_files(workdir)

        overrides = {
            **path_overrides(workdir),
            "logging": {"level": "WARNING", "file_name": "smoke.log"},
            "split": {
                "mode": "fixed_date",
                "fixed_cutoffs": ["2023-01-04T00:00:00Z", "2023-01-05T00:00:00Z"],
            },
            "text": {"tfidf": {"max_features": 50, "min_df": 1}},
            "benchmark": {"k": 5, "max_users": 30, "stability_windows": 2},
            "tuning": {
                "enabled": True,
                "n_trials": 1,
                "max_users": 25,
                "timeout_seconds": 0,
                "metric": "recall",
                "seed_offset": 1,
                "search_space": {
                    "view_weight": [0.1, 1.0],
                    "cart_weight": [0.5, 2.0],
                    "max_features": [50],
                    "ngram_ranges": [[1, 1]],
                    "min_df": [1],
                    "price_weight": [0.1, 1.0],
                    "decay_rate": [0.01, 0.1],
                    "count_transforms": ["raw", "log1p"],
                },
            },
            "evaluation": {"k_values": [3, 5], "scoring_batch_size": 8},
            "module3": {
                "evaluation_split": "test",
                "max_users": 0,
                "export_top_k": 5,
                "export_users": 0,
                "compute_diversity": True,
                "diversity_users": 40,
            },
        }
        config = load_config(overrides=overrides)

        print("Running Module 1 on synthetic data...")
        run_module1(config)

        print("Running Module 2 on synthetic data...")
        run_module2(config)

        print("Running Module 3 on synthetic data...")
        summary = run_module3(config)

        print("Verifying outputs...")
        artifacts = workdir / "module3"
        evaluation = load_json(artifacts / "evaluation.json")
        manifest = load_json(artifacts / "module3_manifest.json")

        _check(evaluation["protocol"]["evaluation_split"] == "test", "test split evaluated")
        _check(
            set(evaluation["metrics"]["by_k"]) == {"3", "5"},
            "every configured cut-off is reported",
        )
        _check(
            evaluation["metrics"]["users_evaluated"] == summary["scored_users"],
            "the report and the summary agree on the scored user count",
        )
        _check(summary["scored_users"] > 0, "at least one user was scored")

        for key, values in evaluation["metrics"]["by_k"].items():
            for metric in ("hit_rate", "recall", "precision", "mrr", "coverage"):
                value = float(values[metric])
                _check(
                    0.0 <= value <= 1.0,
                    f"{metric}@{key} is a probability ({value:.4f})",
                )

        recommendations = load_parquet(artifacts / "recommendations.parquet")
        _check(len(recommendations) > 0, "recommendations were exported")
        _check(
            int(recommendations["rank"].max()) <= 5,
            "exported lists respect the configured depth",
        )
        _check(
            recommendations["user_key"].nunique() == summary["scored_users"],
            "every scored user has an exported list",
        )

        train = load_parquet(Path(overrides["paths"]["artifacts"]) / "splits" / "train.parquet")
        seen_pairs = set(zip(train["user_key"], train["product_id"]))
        leaked = [
            (int(row.user_key), int(row.product_id))
            for row in recommendations.itertuples()
            if (row.user_key, row.product_id) in seen_pairs
        ]
        _check(not leaked, "no recommendation repeats a training item")

        catalogue = set(load_parquet(
            Path(overrides["paths"]["artifacts"]) / "item_index.parquet"
        )["product_id"])
        _check(
            set(recommendations["product_id"]).issubset(catalogue),
            "every recommended product exists in the catalogue",
        )

        diversity = load_json(artifacts / "diversity.json")
        _check(
            diversity["users"] == summary["scored_users"],
            "diversity covers every scored user",
        )
        _check(
            0.0 <= float(diversity["intra_list_diversity"]) <= 1.0,
            "intra-list diversity is inside [0, 1]",
        )
        _check(
            0.0 < float(diversity["catalogue_coverage"]) <= 1.0,
            "catalogue coverage is inside (0, 1]",
        )
        _check(float(diversity["novelty_bits"]) >= 0.0, "novelty is non-negative")

        _check(
            manifest["scored_users"] == summary["scored_users"],
            "the manifest records the scored user count",
        )
        _check(
            len(manifest["artifacts"]) == 3,
            "the manifest fingerprints every exported artifact",
        )
        _check(
            (workdir / "reports_module3" / "module3_report.md").exists(),
            "the markdown report was written",
        )

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
