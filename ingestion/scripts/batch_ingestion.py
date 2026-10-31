
import csv
import getpass
from pathlib import Path

import psycopg


SOURCE_FILE = Path(
    r"D:\Samsung\Graduation Project\Data\Old Data"
    r"\2019-Oct_week1_week2_combined.csv"
)

BATCH_SIZE = 100_000
TARGET_TOTAL = 10_253_804

DB_CONFIG = {
    "host": "localhost",
    "port": 15432,
    "dbname": "ecommerce_db",
    "user": "ecommerce_user",
}

COLUMNS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
    "marketing_channel",
    "platform",
]

COLUMN_SQL = ", ".join(COLUMNS)


def main():
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {SOURCE_FILE}"
        )

    initial_stat = SOURCE_FILE.stat()
    source_path = str(SOURCE_FILE.resolve())
    file_size = initial_stat.st_size
    file_mtime_ns = initial_stat.st_mtime_ns

    password = getpass.getpass("Enter PostgreSQL password: ")

    # Autocommit allows each batch to have its own transaction.
    with psycopg.connect(
        **DB_CONFIG,
        password=password,
        autocommit=True,
    ) as conn:

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS
                monitoring.ingestion_checkpoints (
                    source_path TEXT PRIMARY KEY,
                    file_size_bytes BIGINT NOT NULL,
                    file_mtime_ns BIGINT NOT NULL,
                    last_row_number BIGINT NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            cur.execute("""
                INSERT INTO monitoring.ingestion_checkpoints (
                    source_path,
                    file_size_bytes,
                    file_mtime_ns,
                    last_row_number
                )
                VALUES (%s, %s, %s, 0)
                ON CONFLICT (source_path) DO NOTHING;
            """, (source_path, file_size, file_mtime_ns))

            cur.execute("""
                SELECT file_size_bytes,
                       file_mtime_ns,
                       last_row_number
                FROM monitoring.ingestion_checkpoints
                WHERE source_path = %s;
            """, (source_path,))

            saved_size, saved_mtime, last_row = cur.fetchone()

            if saved_size != file_size or saved_mtime != file_mtime_ns:
                raise RuntimeError(
                    "Source CSV changed since the checkpoint was created."
                )

            # Verify that table contents match the checkpoint.
            cur.execute("""
                SELECT COUNT(*)
                FROM batch.ecommerce_events;
            """)
            table_rows = cur.fetchone()[0]

            if table_rows != last_row:
                raise RuntimeError(
                    f"Safety check failed: table has {table_rows:,} rows, "
                    f"but checkpoint says {last_row:,}. "
                    "No data was loaded."
                )

            if table_rows >= TARGET_TOTAL:
                print(f"Target already reached: {table_rows:,} rows.")
                return

            print(f"Current rows: {table_rows:,}")
            print(f"Target rows: {TARGET_TOTAL:,}")
            print(f"Rows remaining: {TARGET_TOTAL - table_rows:,}")
            print(f"Starting from data row: {last_row + 1:,}")

        with SOURCE_FILE.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:

            reader = csv.reader(csv_file)
            header = next(reader, None)

            if header != COLUMNS:
                raise ValueError(f"Unexpected CSV header: {header}")

            # Skip rows already committed in previous runs.
            for _ in range(last_row):
                if next(reader, None) is None:
                    raise RuntimeError(
                        "CSV ended before the saved checkpoint."
                    )

            while last_row < TARGET_TOTAL:
                rows_to_load = min(
                    BATCH_SIZE,
                    TARGET_TOTAL - last_row,
                )

                batch_start = last_row + 1
                batch_end = last_row
                rows_written = 0

                with conn.transaction():
                    with conn.cursor() as cur:
                        # Lock and verify checkpoint for this batch.
                        cur.execute("""
                            SELECT last_row_number,
                                   file_size_bytes,
                                   file_mtime_ns
                            FROM monitoring.ingestion_checkpoints
                            WHERE source_path = %s
                            FOR UPDATE;
                        """, (source_path,))

                        checkpoint = cur.fetchone()

                        if checkpoint is None:
                            raise RuntimeError("Checkpoint is missing.")

                        db_last_row, db_size, db_mtime = checkpoint

                        if db_last_row != last_row:
                            raise RuntimeError(
                                "Checkpoint changed unexpectedly. "
                                "Stopping to avoid duplicate inserts."
                            )

                        if db_size != file_size or db_mtime != file_mtime_ns:
                            raise RuntimeError(
                                "Source file identity changed. Stopping."
                            )

                        with cur.copy(
                            f"COPY batch.ecommerce_events "
                            f"({COLUMN_SQL}) FROM STDIN"
                        ) as copy:
                            while rows_written < rows_to_load:
                                row = next(reader, None)

                                if row is None:
                                    break

                                if len(row) != len(COLUMNS):
                                    raise ValueError(
                                        f"Invalid column count at data row "
                                        f"{last_row + rows_written + 1:,}"
                                    )

                                copy.write_row(row)
                                rows_written += 1

                        if rows_written != rows_to_load:
                            raise RuntimeError(
                                f"CSV ended early: expected "
                                f"{rows_to_load:,} rows, got "
                                f"{rows_written:,}."
                            )

                        batch_end = last_row + rows_written

                        # Update checkpoint in the same transaction as COPY.
                        cur.execute("""
                            UPDATE monitoring.ingestion_checkpoints
                            SET last_row_number = %s,
                                updated_at = NOW()
                            WHERE source_path = %s;
                        """, (batch_end, source_path))

                # Reached only after the batch transaction commits.
                last_row = batch_end

                print(
                    f"Committed rows {batch_start:,}-{batch_end:,} "
                    f"| Total loaded: {last_row:,}/{TARGET_TOTAL:,}"
                )

                current_stat = SOURCE_FILE.stat()

                if (
                    current_stat.st_size != file_size
                    or current_stat.st_mtime_ns != file_mtime_ns
                ):
                    raise RuntimeError(
                        "Source CSV changed during loading. "
                        "Stopped after the last committed batch."
                    )

        print("\nTarget reached successfully.")
        print(f"Total source rows loaded: {last_row:,}")


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\nBatch loading stopped.")
        print("Check the table count and checkpoint before retrying.")