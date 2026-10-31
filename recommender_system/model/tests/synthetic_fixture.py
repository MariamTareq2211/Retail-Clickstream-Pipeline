"""Shared synthetic catalogue and interaction log for the smoke tests.

The fixture is intentionally tiny but has the temporal structure the pipeline
depends on: training on days 1-3, validation on day 4 and test on day 5. Every
validation/test item is absent from training, so a model cannot score by simply
memorising a seen item; it must generalise through the category and brand tokens
that the splits share.

Keeping the builders in one module means the Module 2 and Module 3 smoke tests
exercise exactly the same data and cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

USERS = 40
TRAIN_PRODUCTS = 30
VAL_PRODUCTS = 15
TEST_PRODUCTS = 15
TOTAL_PRODUCTS = TRAIN_PRODUCTS + VAL_PRODUCTS + TEST_PRODUCTS


def build_products() -> pd.DataFrame:
    """Catalogue where the five category labels repeat across every split."""
    rows = []
    for index in range(TOTAL_PRODUCTS):
        rows.append(
            {
                "product_id": 1000 + index,
                "category_code": f"cat{index % 5}.sub{(index // 5) % 3}.leaf",
                "category_level1": f"cat{index % 5}",
                "category_level2": f"sub{(index // 5) % 3}",
                "category_level3": "Unknown",
                "brand": f"brand{index % 7}",
                "base_price": float(100 + (index % 5) * 10),
            }
        )
    return pd.DataFrame(rows)


def build_events() -> pd.DataFrame:
    """Create train (days 1-3), validation (day 4) and test (day 5) events."""
    rows = []

    for user in range(1, USERS + 1):
        target = (user * 3) % 5
        for day in range(3):
            for offset in range(2):
                product = (user * 3 + day * 7 + offset * 5) % TRAIN_PRODUCTS
                event_type = 4 if (day == 2 and offset == 0) else (2 if day == 1 else 1)
                rows.append(
                    {
                        "user_key": user,
                        "product_id": 1000 + product,
                        "event_type_key": event_type,
                        "event_ts": pd.Timestamp("2023-01-01", tz="UTC")
                        + pd.Timedelta(days=day, hours=user % 24),
                    }
                )

        # Validation and test items are never present in training; they share a
        # category with the user's training items so ranking has real signal.
        rows.append(
            {
                "user_key": user,
                "product_id": 1000 + TRAIN_PRODUCTS + target + 5 * (user % 3),
                "event_type_key": 4,
                "event_ts": pd.Timestamp("2023-01-04", tz="UTC")
                + pd.Timedelta(hours=user % 24),
            }
        )
        rows.append(
            {
                "user_key": user,
                "product_id": 1000 + TRAIN_PRODUCTS + VAL_PRODUCTS + target + 5 * (user % 3),
                "event_type_key": 4,
                "event_ts": pd.Timestamp("2023-01-05", tz="UTC")
                + pd.Timedelta(hours=user % 24),
            }
        )

    return pd.DataFrame(rows)


def build_data_files(workdir: Path) -> tuple[Path, Path]:
    """Write the synthetic parquet inputs and return their paths.

    Args:
        workdir: Temporary directory owned by the calling test.

    Returns:
        Tuple ``(products_path, events_path)``.
    """
    products_path = workdir / "dim_product_clean.parquet"
    events_path = workdir / "fact_events_clean.parquet"
    build_products().to_parquet(products_path, index=False)
    build_events().to_parquet(events_path, index=False)
    return products_path, events_path


def path_overrides(workdir: Path) -> dict[str, dict[str, str]]:
    """Return the ``paths`` section pointing every stage into ``workdir``."""
    return {
        "paths": {
            "products": str(workdir / "dim_product_clean.parquet"),
            "events": str(workdir / "fact_events_clean.parquet"),
            "artifacts": str(workdir / "module1"),
            "reports": str(workdir / "reports_module1"),
            "artifacts_module2": str(workdir / "module2"),
            "reports_module2": str(workdir / "reports_module2"),
            "artifacts_module3": str(workdir / "module3"),
            "reports_module3": str(workdir / "reports_module3"),
        }
    }
