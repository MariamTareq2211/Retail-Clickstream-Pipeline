"""End-to-end contract test for the demo API.

Run from the project root::

    venv\\Scripts\\python.exe -m app.tests.smoke_api

The test loads the real artefacts through FastAPI's lifespan, exercises every
endpoint, and independently verifies that no recommended product appears in the
user's training seen-item row.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep the test light without changing production defaults.
os.environ.setdefault("DEMO_SAMPLE_USERS", "16")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi.testclient import TestClient

from app.api import app
from app.config import DemoSettings
from model.core.paths import resolve_path


def _assert(condition: bool, message: str) -> None:
    """Raise a readable assertion error."""
    if not condition:
        raise AssertionError(message)


def _check_recommendation_list(payload: dict) -> None:
    """Validate ranking order, score bounds and rank numbering."""
    items = payload["recommendations"]
    _assert(payload["returned"] == len(items), "returned does not match list length.")
    _assert(len(items) <= payload["requested_top_k"], "Returned more than top_k.")
    scores = [item["score"] for item in items]
    _assert(
        all(scores[index] >= scores[index + 1] for index in range(len(scores) - 1)),
        "Recommendations are not sorted by descending score.",
    )
    for position, item in enumerate(items, start=1):
        _assert(item["rank"] == position, "Ranks are not contiguous from 1.")
        _assert(0.0 <= item["score"] <= 1.0, "Cosine score outside [0, 1].")
        _assert(
            abs(item["match_percent"] - round(item["score"] * 100.0, 1)) <= 0.11,
            "match_percent does not track the cosine score.",
        )
        _assert(item["price"] >= 0, "Negative product price.")
        _assert(item["product_id"] > 0, "Invalid product identifier.")


def _load_seen_products() -> tuple[pd.DataFrame, sp.csr_matrix, np.ndarray]:
    """Load the independent artefacts used by the seen-item assertion."""
    root = resolve_path("model/artifacts/module2")
    user_index = pd.read_parquet(root / "user_index.parquet")
    item_index = pd.read_parquet(root / "item_index.parquet")
    seen_items = sp.csr_matrix(sp.load_npz(root / "seen_items.npz"))
    products = item_index.sort_values("item_row")["product_id"].to_numpy(np.int64)
    return user_index, seen_items, products


def main() -> int:
    """Execute the smoke suite and return a process exit code."""
    checks = 0
    started = time.perf_counter()

    with TestClient(app) as client:
        health = client.get("/health")
        _assert(health.status_code == 200, "GET /health failed.")
        health_payload = health.json()
        _assert(health_payload["status"] == "ok", "Health status is not ok.")
        _assert(
            health_payload["model"]["key"] == "decayed",
            "The selected production model is not the time-decayed centroid.",
        )
        _assert(health_payload["catalogue_items"] == 117_286, "Catalogue size changed.")
        _assert(health_payload["feature_dimension"] == 4_001, "Feature dimension changed.")
        checks += 1
        print("PASS /health")

        metadata_response = client.get("/metadata")
        _assert(metadata_response.status_code == 200, "GET /metadata failed.")
        metadata = metadata_response.json()
        _assert(metadata["categories"]["level1"], "No department filters returned.")
        _assert(metadata["categories"]["level2"], "No category filters returned.")
        _assert(metadata["brands"], "No brand filters returned.")
        _assert(metadata["price"]["min"] >= 0, "Invalid catalogue price minimum.")
        _assert(
            metadata["price"]["max"] > metadata["price"]["min"],
            "Invalid catalogue price range.",
        )
        _assert(metadata["evaluation"]["by_k"], "Evaluation metrics are missing.")
        checks += 1
        print("PASS /metadata")

        users_response = client.get("/users", params={"limit": 16})
        _assert(users_response.status_code == 200, "GET /users failed.")
        users_payload = users_response.json()
        _assert(users_payload["users"], "No sample users returned.")
        _assert(
            users_payload["count"] <= 16, "The users limit was not respected."
        )
        for user in users_payload["users"]:
            _assert(isinstance(user["user_key"], str), "user_key is not a string.")
            _assert(
                user["interactions"] >= DemoSettings().sample_min_interactions,
                "Sample user is below the interaction threshold.",
            )
        checks += 1
        print(f"PASS /users ({users_payload['total']} curated users)")

        active = max(users_payload["users"], key=lambda user: user["interactions"])
        user_key = active["user_key"]

        profile_response = client.get(f"/profile/{user_key}")
        _assert(profile_response.status_code == 200, "GET /profile failed.")
        profile = profile_response.json()
        _assert(profile["user_key"] == user_key, "Profile returned a different user.")
        _assert(profile["total_interactions"] > 0, "Profile has no interactions.")
        _assert(profile["top_brands"], "Profile has no top brands.")
        _assert(profile["top_categories"], "Profile has no top categories.")
        checks += 1
        print(
            "PASS /profile "
            f"({profile['total_interactions']:,} interactions, "
            f"{profile['distinct_products']} products)"
        )

        recommend_response = client.get(
            "/recommend", params={"user_key": user_key, "top_k": 10}
        )
        _assert(recommend_response.status_code == 200, "GET /recommend failed.")
        payload = recommend_response.json()
        _check_recommendation_list(payload)
        _assert(payload["user_key"] == user_key, "Recommendation user mismatch.")
        _assert(payload["returned"] > 0, "No recommendations returned.")
        _assert(
            payload["profile"]["total_interactions"]
            == profile["total_interactions"],
            "Recommendation profile disagrees with /profile.",
        )
        _assert(
            "x-process-time-ms" in recommend_response.headers,
            "Process-time header is missing.",
        )
        checks += 1
        print(
            "PASS /recommend "
            f"({payload['returned']} items, {payload['ranking_ms']:.1f} ms engine)"
        )

        user_index, seen_items, item_products = _load_seen_products()
        row = int(
            user_index.loc[user_index["user_key"] == int(user_key), "user_row"].iloc[0]
        )
        seen_rows = seen_items.indices[
            seen_items.indptr[row] : seen_items.indptr[row + 1]
        ]
        seen_products = set(item_products[seen_rows].tolist())
        recommended_products = {
            item["product_id"] for item in payload["recommendations"]
        }
        _assert(
            not (seen_products & recommended_products),
            "A recommendation intersects the user's training seen-item row.",
        )
        checks += 1
        print("PASS seen-item exclusion")

        department = metadata["categories"]["level1"][0]["value"]
        filtered = client.get(
            "/recommend",
            params={
                "user_key": user_key,
                "top_k": 10,
                "category_level1": department,
            },
        )
        _assert(filtered.status_code == 200, "Department filter request failed.")
        filtered_payload = filtered.json()
        _check_recommendation_list(filtered_payload)
        _assert(
            all(
                item["category_level1"] == department
                for item in filtered_payload["recommendations"]
            ),
            "Department filter leaked another department.",
        )
        checks += 1
        print(f"PASS department filter ({department})")

        category_entry = metadata["categories"]["level2"][0]
        filtered = client.get(
            "/recommend",
            params={
                "user_key": user_key,
                "top_k": 10,
                "category_level2": category_entry["value"],
            },
        )
        _assert(filtered.status_code == 200, "Category filter request failed.")
        filtered_payload = filtered.json()
        _check_recommendation_list(filtered_payload)
        _assert(
            all(
                item["category_level2"] == category_entry["value"]
                for item in filtered_payload["recommendations"]
            ),
            "Category filter leaked another category.",
        )
        checks += 1
        print(f"PASS category filter ({category_entry['value']})")

        brand = metadata["brands"][0]["value"]
        filtered = client.get(
            "/recommend",
            params={"user_key": user_key, "top_k": 10, "brand": brand},
        )
        _assert(filtered.status_code == 200, "Brand filter request failed.")
        filtered_payload = filtered.json()
        _check_recommendation_list(filtered_payload)
        _assert(
            all(
                item["brand"] == brand
                for item in filtered_payload["recommendations"]
            ),
            "Brand filter leaked another brand.",
        )
        checks += 1
        print(f"PASS brand filter ({brand})")

        filtered = client.get(
            "/recommend",
            params={
                "user_key": user_key,
                "top_k": 10,
                "min_price": 50,
                "max_price": 300,
            },
        )
        _assert(filtered.status_code == 200, "Price filter request failed.")
        filtered_payload = filtered.json()
        _check_recommendation_list(filtered_payload)
        _assert(
            all(
                50 <= item["price"] <= 300
                for item in filtered_payload["recommendations"]
            ),
            "Price filter returned an out-of-range product.",
        )
        checks += 1
        print("PASS price filter (50-300)")

        combined = client.get(
            "/recommend",
            params={
                "user_key": user_key,
                "top_k": 5,
                "category_level1": department,
                "brand": brand,
                "min_price": 0,
                "max_price": 1000,
            },
        )
        _assert(combined.status_code == 200, "Combined filter request failed.")
        _check_recommendation_list(combined.json())
        checks += 1
        print("PASS combined filters")

        missing = client.get(
            "/recommend", params={"user_key": "-1", "top_k": 10}
        )
        _assert(missing.status_code == 404, "Unknown user did not return 404.")
        _assert("users" in missing.json()["detail"], "404 does not point to /users.")
        checks += 1
        print("PASS unknown user -> 404")

        invalid_key = client.get(
            "/recommend", params={"user_key": "not-a-key", "top_k": 10}
        )
        _assert(invalid_key.status_code == 400, "Malformed key did not return 400.")
        checks += 1

        invalid_depth = client.get(
            "/recommend", params={"user_key": user_key, "top_k": 1}
        )
        _assert(invalid_depth.status_code == 400, "Out-of-range top_k accepted.")
        checks += 1

        invalid_filter = client.get(
            "/recommend",
            params={
                "user_key": user_key,
                "top_k": 10,
                "category_level1": "does-not-exist",
            },
        )
        _assert(
            invalid_filter.status_code == 400,
            "Unknown category did not return 400.",
        )
        checks += 1
        print("PASS request validation (3 cases)")

        timings = []
        for index in range(20):
            start = time.perf_counter()
            response = client.get(
                "/recommend",
                params={
                    "user_key": user_key,
                    "top_k": 20,
                    "min_price": 20 + index,
                    "max_price": 2000,
                },
            )
            timings.append((time.perf_counter() - start) * 1000.0)
            _assert(response.status_code == 200, "Latency request failed.")
        p50 = float(np.percentile(timings, 50))
        p95 = float(np.percentile(timings, 95))
        _assert(p95 < 1000.0, f"p95 latency is not sub-second: {p95:.1f} ms")
        checks += 1
        print(f"PASS latency (p50 {p50:.1f} ms, p95 {p95:.1f} ms)")

    elapsed = time.perf_counter() - started
    print(f"\nAll {checks} checks passed in {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
