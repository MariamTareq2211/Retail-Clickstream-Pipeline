from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from pyspark.sql import functions as F

INJECTED = {"Bot-like Crawling Loops": 53061, "Duplicate Events": 42449, "Impossible Sequences": 42449,
            "Cart Abandonment Scenarios": 31837, "Promotional Purchase Surges": 42449, "Missing Data": 212244}

ev  = spark.table("Silver.events")
dlq = spark.table("Silver.dlq_events").filter("source='batch' AND entity='events'")
dups = (spark.table("Silver.pipeline_metrics")
             .filter("stage LIKE 'silver_events/%' AND metric='duplicates_removed'")
             .agg(F.sum("value")).first()[0] or 0)

a = ev.agg(
    F.sum(F.col("is_bot_like").cast("int")).alias("bot_events"),
    F.countDistinct(F.when(F.col("is_bot_session"), F.col("user_session"))).alias("bot_sessions"),
    F.sum(F.col("is_impossible_sequence").cast("int")).alias("impossible"),
    F.sum((F.col("platform_imputed") | F.col("channel_imputed")).cast("int")).alias("imputed_rows"),
).first().asDict()


dlq_by = {r["dlq_reason"]: r["count"] for r in dlq.groupBy("dlq_reason").count().collect()}
missing_dlq = sum(v for k, v in dlq_by.items() if k.startswith("null_") or k == "malformed_row")

print(f"Bot events          : {a['bot_events']:>10,} (sessions {a['bot_sessions']:,}) | injected {INJECTED['Bot-like Crawling Loops']:,}")
print(f"Duplicates removed  : {int(dups):>10,} | injected {INJECTED['Duplicate Events']:,}")
print(f"Impossible sequences: {a['impossible']:>10,} | injected {INJECTED['Impossible Sequences']:,}")
print(f"Missing data        : DLQ null/malformed {missing_dlq:,} + imputed rows {a['imputed_rows']:,} | injected {INJECTED['Missing Data']:,}")
print(f"Abandoned sessions  : (Moved to Gold Layer) | injected {INJECTED['Cart Abandonment Scenarios']:,}")
print("DLQ by reason:", dlq_by)