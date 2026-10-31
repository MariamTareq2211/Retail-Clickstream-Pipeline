
import csv
import getpass
import time
from itertools import islice
from pathlib import Path

import psycopg


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SOURCE_FILE = Path(
    r"D:\Samsung\Graduation Project\Data\Old Data"
    r"\2019-Oct_week1_week2_combined.csv"
)

TARGET_SCHEMA = "streaming"
TARGET_TABLE = "ecommerce_events"

CHECKPOINT_SCHEMA = "monitoring"
CHECKPOINT_TABLE = "streaming_checkpoints"

TOTAL_ROWS = 9_253_804
BATCH_SIZE = 20_000
DELAY_SECONDS = 0

COLUMNS = [
    "event_time",
    "event_type",
    "user_id",
    "product_id",
    "price",
    "marketing_channel",
    "platform",
    "user_session",
]

SOURCE_COLUMNS = COLUMNS.copy()


# --------------------------------------------------
# Database connection
# --------------------------------------------------

def connect_db():
    password = getpass.getpass("PostgreSQL password: ")

    return psycopg.connect(
        host="localhost",
        port=15432,
        dbname="ecommerce_db",
        user="ecommerce_user",
        password=password,
    )


# --------------------------------------------------
# Checkpoint management
# --------------------------------------------------

def get_checkpoint(conn, source_path, file_size, file_mtime_ns):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT file_size_bytes, file_mtime_ns, last_row_number
            FROM {CHECKPOINT_SCHEMA}.{CHECKPOINT_TABLE}
            WHERE source_path = %s
            """,
            (source_path,),
        )
        checkpoint = cur.fetchone()

        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM {TARGET_SCHEMA}.{TARGET_TABLE}
            """
        )
        table_count = cur.fetchone()[0]

    if checkpoint is None:
        if table_count != 0:
            raise RuntimeError(
                f"Target table already contains {table_count} rows, "
                "but no checkpoint exists for this source. "
                "Stopping to avoid accidental duplicate loading."
            )

        return 0

    saved_size, saved_mtime, last_row = checkpoint

    if saved_size != file_size or saved_mtime != file_mtime_ns:
        raise RuntimeError(
            "The source CSV has changed since the checkpoint was saved. "
            "Stopping to avoid resuming against a different file."
        )

    if table_count != last_row:
        raise RuntimeError(
            f"Checkpoint says {last_row} rows, but target table has "
            f"{table_count} rows. Investigate before resuming."
        )

    return last_row


def save_checkpoint(
    conn,
    source_path,
    file_size,
    file_mtime_ns,
    last_row_number,
):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {CHECKPOINT_SCHEMA}.{CHECKPOINT_TABLE}
                (
                    source_path,
                    file_size_bytes,
                    file_mtime_ns,
                    last_row_number,
                    updated_at
                )
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (source_path)
            DO UPDATE SET
                file_size_bytes = EXCLUDED.file_size_bytes,
                file_mtime_ns = EXCLUDED.file_mtime_ns,
                last_row_number = EXCLUDED.last_row_number,
                updated_at = NOW()
            """,
            (
                source_path,
                file_size,
                file_mtime_ns,
                last_row_number,
            ),
        )


# --------------------------------------------------
# Streaming simulation
# --------------------------------------------------

def run_streaming():
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(
            f"Source CSV not found: {SOURCE_FILE}"
        )

    source_path = str(SOURCE_FILE.resolve())
    file_stat = SOURCE_FILE.stat()

    file_size = file_stat.st_size
    file_mtime_ns = file_stat.st_mtime_ns

    print("Connecting to PostgreSQL...")

    with connect_db() as conn:
        last_row = get_checkpoint(
            conn,
            source_path,
            file_size,
            file_mtime_ns,
        )

        if last_row >= TOTAL_ROWS:
            print(
                f"Already processed {last_row} rows. "
                "All rows have been processed. Nothing to do."
            )
            return

        print(f"Previously processed: {last_row:,} rows")
        print(f"Target for this test: {TOTAL_ROWS:,} rows")
        print(f"Remaining: {TOTAL_ROWS - last_row:,} rows")
        print("Starting CSV event replay...")

        with SOURCE_FILE.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as csv_file:

            reader = csv.DictReader(csv_file)

            missing_columns = [
                col for col in SOURCE_COLUMNS
                if col not in (reader.fieldnames or [])
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing source columns: {missing_columns}"
                )

            # Skip previously committed data rows.
            for _ in range(last_row):
                if next(reader, None) is None:
                    raise RuntimeError(
                        "CSV ended before the checkpoint row."
                    )

            insert_sql = """
                INSERT INTO streaming.ecommerce_events (
                    event_time,
                    event_type,
                    user_id,
                    product_id,
                    price,
                    marketing_channel,
                    platform,
                    user_session
                )
                VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
            """

            while last_row < TOTAL_ROWS:
                remaining = TOTAL_ROWS - last_row
                batch_limit = min(BATCH_SIZE, remaining)

                batch = []

                for row in islice(reader, batch_limit):
                    batch.append(
                        tuple(
                            row[column]
                            for column in COLUMNS
                        )
                    )

                if not batch:
                    print("End of CSV reached.")
                    break

                new_last_row = last_row + len(batch)

                # Insert data and update checkpoint atomically.
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.executemany(insert_sql, batch)

                    save_checkpoint(
                        conn,
                        source_path,
                        file_size,
                        file_mtime_ns,
                        new_last_row,
                    )

                last_row = new_last_row

                print(
                    f"Progress: {last_row:,}/{TOTAL_ROWS:,} "
                    f"rows loaded"
                )

                if last_row < TOTAL_ROWS:
                    time.sleep(DELAY_SECONDS)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM streaming.ecommerce_events
                """
            )
            final_count = cur.fetchone()[0]

        print("\nStreaming simulation finished.")
        print(f"Rows in target table: {final_count:,}")
        print(f"Last source row checkpointed: {last_row:,}")


if __name__ == "__main__":
    try:
        run_streaming()
    except KeyboardInterrupt:
        print(
            "\nStopped by user. Successfully committed batches "
            "and checkpoint are preserved for a later resume."
        )
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise
