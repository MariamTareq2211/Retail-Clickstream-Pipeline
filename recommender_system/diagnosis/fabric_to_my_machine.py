"""Extract the product catalog and the raw interaction log from Microsoft Fabric.

Blueprint reference: Plan.md - Module 1, item 1.2 (temporal split by ``event_ts``).

This module is the only component allowed to talk to Microsoft Fabric. It pulls
``Gold.dim_product`` (active rows) and ``Gold.fact_events`` (bot-filtered rows)
including the event timestamp, and writes them to ``data/*.parquet`` for the
offline pipeline.

Usage (from the project root, inside the local virtual environment):

    venv\\Scripts\\python.exe diagnosis\\fabric_to_my_machine.py --check-only
    venv\\Scripts\\python.exe diagnosis\\fabric_to_my_machine.py

The script opens an interactive browser window for Azure AD authentication, so
it must be started manually by a human. It is intentionally kept outside
``model/`` and must never be imported or invoked by the training pipeline.

Design notes
------------
* The timestamp column name is discovered at runtime from
  ``INFORMATION_SCHEMA`` (candidates live in ``EVENT_TS_CANDIDATES``) because the
  Gold layer naming is not guaranteed. If discovery fails the script prints the
  available columns instead of guessing.
* Existing parquet files are replaced atomically (temporary file + rename), so a
  failed extraction can never leave a half-written dataset behind.
* No deduplication happens here: raw events are preserved exactly as delivered
  by the warehouse. Module 1 applies the documented aggregation step and logs
  every row it collapses, keeping this script reproducible and dumb on purpose.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyodbc  # noqa: F401  (imported for driver availability checks at runtime)
from azure.identity import InteractiveBrowserCredential
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Connection settings
# ---------------------------------------------------------------------------

FABRIC_TENANT_ID = "6a713771-0d54-4a6c-bbfd-fcba6c85d290"
USER_EMAIL = "2401526@rst.edu.eg"
SERVER = (
    "oe3xc2subvwevo757s5gzbossa-nby64cvvc6benanjye2rzyyu34"
    ".datawarehouse.fabric.microsoft.com"
)
DATABASE = "Retail_Clickstream_Lakehouse"

# SQL Server driver constant for passing an Azure AD access token through pyodbc.
SQL_COPT_SS_ACCESS_TOKEN = 1256

# Product catalogue columns consumed by Module 1. Kept explicit so a schema
# change upstream fails loudly here instead of silently breaking TF-IDF later.
PRODUCT_COLUMNS = (
    "product_id",
    "category_code",
    "category_level1",
    "category_level2",
    "category_level3",
    "brand",
    "base_price",
)

# Interaction columns required by the pipeline, in output order.
EVENT_COLUMNS = ("user_key", "product_id", "event_type_key")

# Candidate names for the event timestamp, ordered by preference. The first one
# present in the warehouse table wins; ``event_ts`` is the documented contract.
EVENT_TS_CANDIDATES = (
    "event_ts",
    "event_time",
    "event_timestamp",
    "timestamp",
    "ts",
    "event_datetime",
    "occurred_at",
    "created_at",
)


def build_engine() -> Engine:
    """Create a SQLAlchemy engine authenticated with an interactive Azure AD token.

    Returns:
        Engine: A SQLAlchemy engine bound to the Fabric SQL analytics endpoint.

    Raises:
        RuntimeError: If the access token cannot be acquired.
    """
    print("[auth] Acquiring Azure AD access token for the Fabric SQL endpoint...")

    credential = InteractiveBrowserCredential(
        tenant_id=FABRIC_TENANT_ID,
        login_hint=USER_EMAIL,
        additionally_allowed_tenants=["*"],
    )
    token = credential.get_token("https://database.windows.net/.default").token

    # pyodbc expects the token as a UTF-16-LE byte buffer with a 4-byte length
    # prefix, not as a plain string.
    token_bytes = token.encode("utf-16-le")
    token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)

    connection_string = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
    )
    quoted = urllib.parse.quote_plus(connection_string)

    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={quoted}",
        connect_args={"attrs_before": {SQL_COPT_SS_ACCESS_TOKEN: token_struct}},
    )


def list_table_columns(engine: Engine, schema: str, table: str) -> list[str]:
    """Return the column names of a warehouse table.

    Args:
        engine: Open SQLAlchemy engine.
        schema: Schema name, e.g. ``"Gold"``.
        table: Table name, e.g. ``"fact_events"``.

    Returns:
        Column names in ordinal order (empty when the table is not visible).
    """
    query = text(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
        ORDER BY ORDINAL_POSITION
        """
    )
    with engine.connect() as connection:
        rows = connection.execute(query, {"schema": schema, "table": table}).fetchall()
    return [row[0] for row in rows]


def resolve_event_timestamp_column(available: list[str]) -> str:
    """Pick the event timestamp column from the warehouse schema.

    Args:
        available: Column names reported by ``INFORMATION_SCHEMA``.

    Returns:
        The first candidate present in ``available``.

    Raises:
        RuntimeError: If none of the candidates exists, with the full column list
            printed so the contract can be updated deliberately.
    """
    lowered = {column.lower(): column for column in available}
    for candidate in EVENT_TS_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    raise RuntimeError(
        "No event timestamp column found in the warehouse table. "
        f"Looked for {list(EVENT_TS_CANDIDATES)}.\n"
        f"Available columns: {available}\n"
        "Update EVENT_TS_CANDIDATES in this script with the real column name."
    )


def build_queries(
    schema: str,
    products_table: str,
    events_table: str,
    timestamp_column: str,
) -> tuple[str, str]:
    """Build the two extraction queries.

    Args:
        schema: Warehouse schema, normally ``"Gold"``.
        products_table: Product dimension table name.
        events_table: Interaction fact table name.
        timestamp_column: Resolved event timestamp column name.

    Returns:
        Tuple of ``(products_query, events_query)``. Both queries sort
        deterministically so repeated extractions are byte-comparable.
    """
    products_query = f"""
        SELECT {", ".join(PRODUCT_COLUMNS)}
        FROM {schema}.{products_table}
        WHERE is_current = 1
        ORDER BY product_id
    """

    event_columns = (*EVENT_COLUMNS, timestamp_column)
    events_query = f"""
        SELECT {", ".join(event_columns)}
        FROM {schema}.{events_table}
        WHERE is_bot = 0
          AND product_id <> -1
        ORDER BY {timestamp_column}, user_key, product_id, event_type_key
    """
    return products_query, events_query


def write_parquet_atomically(frame: pd.DataFrame, destination: Path) -> None:
    """Write a DataFrame to parquet without exposing partial files.

    The data is written to a sibling temporary file and moved into place with
    ``os.replace`` only after the write succeeds.

    Args:
        frame: Data to persist.
        destination: Final parquet path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def extract(engine: Engine, output_dir: Path, schema: str, products_table: str, events_table: str) -> None:
    """Run the full extraction and persist both datasets.

    Args:
        engine: Authenticated SQLAlchemy engine.
        output_dir: Folder that receives the parquet files.
        schema: Warehouse schema to read from.
        products_table: Product dimension table.
        events_table: Interaction fact table.
    """
    columns = list_table_columns(engine, schema, events_table)
    if not columns:
        raise RuntimeError(f"{schema}.{events_table} is not visible with the current credentials.")

    timestamp_column = resolve_event_timestamp_column(columns)
    print(f"[schema] Using event timestamp column: {timestamp_column!r}")
    print(f"[schema] {schema}.{events_table} exposes {len(columns)} columns.")

    products_query, events_query = build_queries(
        schema, products_table, events_table, timestamp_column
    )

    print("[extract] Reading product catalog...")
    with engine.connect() as connection:
        products = pd.read_sql(text(products_query), connection)
    print(f"[extract] Products loaded: {len(products):,}")

    print("[extract] Reading raw interactions (this may take a few minutes)...")
    with engine.connect() as connection:
        events = pd.read_sql(text(events_query), connection)
    print(f"[extract] Events loaded: {len(events):,}")

    # Fail fast when the timestamp column came back empty - a read that silently
    # returns nulls would poison every downstream temporal split.
    if events[timestamp_column].isna().any():
        nulls = int(events[timestamp_column].isna().sum())
        raise RuntimeError(
            f"{nulls:,} events have a null {timestamp_column!r}; "
            "resolve this upstream before running the pipeline."
        )

    events[timestamp_column] = pd.to_datetime(events[timestamp_column], utc=True)
    print(
        "[extract] Event time range: "
        f"{events[timestamp_column].min()} -> {events[timestamp_column].max()}"
    )

    products_path = output_dir / "dim_product_clean.parquet"
    events_path = output_dir / "fact_events_clean.parquet"
    write_parquet_atomically(products, products_path)
    write_parquet_atomically(events, events_path)

    print(f"[save] {products_path}  ({products_path.stat().st_size / 1e6:.2f} MB)")
    print(f"[save] {events_path}  ({events_path.stat().st_size / 1e6:.2f} MB)")
    print(f"[done] Extraction finished at {datetime.now(timezone.utc).isoformat()}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Folder receiving the parquet files (default: <project>/data).",
    )
    parser.add_argument("--schema", default="Gold", help="Warehouse schema (default: Gold).")
    parser.add_argument(
        "--products-table", default="dim_product", help="Product dimension table."
    )
    parser.add_argument(
        "--events-table", default="fact_events", help="Interaction fact table."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print the discovered schema and timestamp column, then exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point used by the command line."""
    args = parse_args(argv)
    engine = build_engine()

    if args.check_only:
        columns = list_table_columns(engine, args.schema, args.events_table)
        print(f"[schema] {args.schema}.{args.events_table} columns ({len(columns)}):")
        for column in columns:
            print(f"  - {column}")
        print(f"[schema] Resolved timestamp column: {resolve_event_timestamp_column(columns)!r}")
        return 0

    extract(engine, args.out_dir, args.schema, args.products_table, args.events_table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
