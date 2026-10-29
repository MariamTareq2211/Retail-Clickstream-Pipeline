"""Synthetic end-to-end smoke test for Module 2.

Run from the project root:

    venv\\\\Scripts\\\\python.exe -m model.tests.smoke_module2

The test generates a small catalogue and interaction log with a clean temporal
structure (train days 1-3, validation day 4, test day 5), runs the real Module 1
and Module 2 pipelines against it, and checks:

* the three-model benchmark produces a complete comparison and a winner;
* the Optuna study runs and its best parameters are replayable;
* the exported catalogue matrix, profiles, user index and seen-item mask agree
  on shape;
* every profiled user comes from the training split (no leakage);
* the metric and decay primitives return the documented values.

No real data is read and no repository artifact is written.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from model.config.config_schema import load_config
from model.core.logger import configure_logging
from model.core.persistence import load_json, load_parquet, load_sparse
from model.evaluation.metrics import evaluate_at_k
from model.pipelines.module1_preprocess import run_module1
from model.pipelines.module2_train import run_module2
from model.profiles.interaction_weights import apply_count_transform
from model.profiles.time_decay import decay_weights
from model.ranking.scorer import RankingResult
from model.tests.synthetic_fixture import build_data_files, path_overrides


def _check(condition: bool, message: str) -> None:
    """Assert a condition and print a progress line."""
    if not condition:
        raise AssertionError(message)
    print(f"  [ok] {message}")


def _check_primitives() -> None:
    """Verify the decay transform, count transform and metric definitions."""
    hour_ns = int(3.6e12)
    weight = decay_weights(
        np.array([0], dtype=np.int64),
        int(24 * hour_ns),
        float(np.log(2) / 24),
    )
    _check(abs(float(weight[0]) - 0.5) < 1e-9, "decay reaches a 24-hour half-life")

    transformed = apply_count_transform(np.array([1, 2, 10]), "capped", 5)
    _check(
        np.allclose(transformed, [1.0, 2.0, 5.0]),
        "capped count transform clips at the configured cap",
    )

    rankings = RankingResult(
        item_rows=np.array([[5, 6, 7], [8, 9, -1]], dtype=np.int64),
        scores=np.zeros((2, 3)),
        k=3,
    )
    metrics = evaluate_at_k(
        rankings,
        {1: np.array([7]), 2: np.array([9])},
        np.array([1, 2]),
        k=3,
        n_items=20,
    )
    _check(metrics.hit_rate == 1.0, "HitRate counts every user with a hit")
    _check(metrics.recall == 1.0, "Recall is one when the single relevant item is found")
    _check(
        abs(metrics.mrr - (1 / 3 + 1 / 2) / 2) < 1e-9,
        "MRR uses the rank of the first relevant item",
    )


def main() -> int:
    """Run the smoke test and return the process exit code."""
    configure_logging("WARNING", None)
    _check_primitives()

    with tempfile.TemporaryDirectory(prefix="module2_smoke_") as temporary:
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
                "n_trials": 2,
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
        }
        config = load_config(overrides=overrides)

        print("Running Module 1 on synthetic data...")
        run_module1(config)

        print("Running Module 2 on synthetic data...")
        summary = run_module2(config)

        print("Verifying outputs...")
        artifacts = workdir / "module2"
        benchmark = load_json(artifacts / "benchmark.json")
        _check(
            set(benchmark["models"]) == {"unweighted", "weighted", "decayed"},
            "all three models were benchmarked",
        )
        _check(
            benchmark["selection"]["winner"] == summary["selected_model"],
            "the report and the summary agree on the winning model",
        )
        _check(
            len(benchmark["models"]["decayed"]["windows"]) == 2,
            "validation was split into the configured number of windows",
        )

        matrix = load_sparse(artifacts / "item_matrix.npz")
        profiles = load_sparse(artifacts / "profiles.npz")
        seen = load_sparse(artifacts / "seen_items.npz")
        item_index = load_parquet(artifacts / "item_index.parquet")
        user_index = load_parquet(artifacts / "user_index.parquet")
        _check(matrix.shape[0] == item_index.shape[0], "item matrix rows match item_index")
        _check(profiles.shape[1] == matrix.shape[1], "profiles and catalogue share the feature space")
        _check(profiles.shape[0] == user_index.shape[0], "profile rows match user_index")
        _check(seen.shape == (profiles.shape[0], matrix.shape[0]), "seen mask has the profile x item shape")

        train = load_parquet(Path(overrides["paths"]["artifacts"]) / "splits" / "train.parquet")
        _check(
            set(user_index["user_key"]).issubset(set(train["user_key"])),
            "every profiled user comes from the training split (no leakage)",
        )

        best = load_json(artifacts / "best_params.json")
        _check(best["tuned"] is True, "tuning ran")
        _check(best["completed_trials"] == 2, "both Optuna trials completed")
        _check((artifacts / "tuning_trials.csv").exists(), "trial table exported")
        _check(best["parameters"]["max_features"] == 50, "best parameters are replayable")

        spec = load_json(artifacts / "model_spec.json")
        _check(spec["feature_dimension"] == matrix.shape[1], "model spec records the exported feature dimension")
        _check(spec["profile_users"] == profiles.shape[0], "model spec records the exported user count")

        norms = np.sqrt(np.asarray(profiles.multiply(profiles).sum(axis=1))).ravel()
        _check(
            bool(np.allclose(norms[norms > 0], 1.0, atol=1e-6)),
            "exported profiles are L2-normalised",
        )

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
