"""Latency benchmark for the demo API.

Run from the project root::

    venv\\Scripts\\python.exe -m app.tests.benchmark_api

The benchmark warms the score cache, then measures end-to-end TestClient
latency for unfiltered and filtered top-K requests. It fails if the 95th
percentile is not sub-second.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DEMO_SAMPLE_USERS", "24")

import numpy as np
from fastapi.testclient import TestClient

from app.api import app


def _summarise(label: str, samples: list[float]) -> None:
    """Print a compact latency summary."""
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    p95 = float(np.percentile(ordered, 95))
    p99 = float(np.percentile(ordered, 99))
    print(
        f"{label:<24} n={len(ordered):>3}  "
        f"p50={p50:6.1f} ms  p95={p95:6.1f} ms  "
        f"p99={p99:6.1f} ms  max={max(ordered):6.1f} ms"
    )


def main() -> int:
    """Measure warm recommendation latency and assert the SLA."""
    with TestClient(app) as client:
        metadata = client.get("/metadata").json()
        users = client.get("/users", params={"limit": 10}).json()["users"]
        keys = [user["user_key"] for user in users]
        brand = metadata["brands"][0]["value"]
        department = metadata["categories"]["level1"][0]["value"]

        scenarios = {
            "unfiltered top-10": {"top_k": 10},
            "unfiltered top-20": {"top_k": 20},
            "department filter": {
                "top_k": 10,
                "category_level1": department,
            },
            "brand filter": {"top_k": 10, "brand": brand},
            "price window": {"top_k": 10, "min_price": 50, "max_price": 500},
        }

        # Warm the per-user score cache and the HTTP path.
        for key in keys:
            client.get("/recommend", params={"user_key": key, "top_k": 20})

        failures = []
        for label, params in scenarios.items():
            samples = []
            for index in range(40):
                key = keys[index % len(keys)]
                start = time.perf_counter()
                response = client.get(
                    "/recommend", params={"user_key": key, **params}
                )
                samples.append((time.perf_counter() - start) * 1000.0)
                if response.status_code != 200:
                    failures.append(f"{label}: HTTP {response.status_code}")
            _summarise(label, samples)
            p95 = float(np.percentile(samples, 95))
            if p95 >= 1000.0:
                failures.append(f"{label}: p95 {p95:.1f} ms is not sub-second")

        if failures:
            for failure in failures:
                print(f"FAIL: {failure}", file=sys.stderr)
            return 1

        print("\nSub-second SLO satisfied for every scenario.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
