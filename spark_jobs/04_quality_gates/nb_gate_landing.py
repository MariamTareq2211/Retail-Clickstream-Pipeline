from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

import pyspark.sql.functions as F
import logging
import sys

logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s'
)

MAX_DLQ_RATE = 0.05
MAX_BOT_RATE = 0.20

logging.info(" Inspecting Silver Layer Metrics...")

events = spark.table("Silver.events")
dlq = spark.table("Silver.dlq_events").filter("source = 'batch' AND entity = 'events'")

latest_date_row = events.agg(F.max(F.to_date("ingestion_time")).alias("latest")).first()
latest_date = latest_date_row["latest"] if latest_date_row else None

if not latest_date:
    logging.error("Silver.events is completely empty. No data to process.")
    raise ValueError("[Gate 2 Failed] Silver.events is completely empty. No data to process.")

logging.info(f" Latest batch date found: {latest_date}. Calculating metrics...")

events_today = events.filter(F.to_date("ingestion_time") == latest_date)
dlq_today = dlq.filter(F.to_date("bronze_ingestion_time") == latest_date)

n_good = events_today.count()
n_dlq = dlq_today.count()
n_total = n_good + n_dlq

if n_total == 0:
    logging.error(f"No records found for the latest batch date: {latest_date}")
    raise ValueError(f"[Gate 2 Failed] No records found for the latest batch date: {latest_date}")

n_bots = events_today.filter(F.col("is_bot_like") | F.col("is_bot_session")).count()

dlq_rate = n_dlq / n_total
bot_rate = n_bots / n_good if n_good > 0 else 0

print(f"Batch Date: {latest_date}")
print(f"Total Processed : {n_total:,}")
print(f"Good Records    : {n_good:,}")
print(f"DLQ Records     : {n_dlq:,}")
print(f"Bot Records     : {n_bots:,}")
print(f"DLQ Rate        : {dlq_rate:.2%} (Max Allowed: {MAX_DLQ_RATE:.0%})")
print(f"Bot Rate        : {bot_rate:.2%} (Max Allowed: {MAX_BOT_RATE:.0%})")

if dlq_rate > MAX_DLQ_RATE:
    logging.error(f"DLQ Rate ({dlq_rate:.2%}) exceeded the max threshold of {MAX_DLQ_RATE:.0%}!")
    raise ValueError(f"[Gate 2 Failed] DLQ Rate ({dlq_rate:.2%}) exceeded the max threshold of {MAX_DLQ_RATE:.0%}!")

if bot_rate > MAX_BOT_RATE:
    logging.error(f"Bot Rate ({bot_rate:.2%}) exceeded the max threshold of {MAX_BOT_RATE:.0%}!")
    raise ValueError(f"[Gate 2 Failed] Bot Rate ({bot_rate:.2%}) exceeded the max threshold of {MAX_BOT_RATE:.0%}!")

print("Gate 2 Passed! Data quality is within acceptable limits.")
logging.info(" Gate 2 Passed! Data quality is within acceptable limits.")