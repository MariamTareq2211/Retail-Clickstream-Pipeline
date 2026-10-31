from nb_utils import *   
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
 
spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.ansi.enabled", "false")
 
EH_NAMESPACE = "eh-ecommerce-streaming-2026"
EH_NAME      = "ecommerce-cdc-events"
 
with open("/lakehouse/default/Files/_config/eh_conn.txt") as f:
    EH_CONN = f.read().strip()
 
KAFKA_OPTIONS = {
    "kafka.bootstrap.servers": f"{EH_NAMESPACE}.servicebus.windows.net:9093",
    "subscribe": EH_NAME,
    "kafka.group.id": "cg-fabric",
    "kafka.security.protocol": "SASL_SSL",
    "kafka.sasl.mechanism": "PLAIN",
    "kafka.sasl.jaas.config": (
        'org.apache.kafka.common.security.plain.PlainLoginModule required '
        f'username="$ConnectionString" password="{EH_CONN}";'
    ),
    "kafka.request.timeout.ms": "60000",
    "kafka.session.timeout.ms": "30000",
}
print("Config ready")
 
ALLOWED_TYPES = ["view", "cart", "purchase"]
 
def parse_and_validate(raw_df):
    v = F.col("value").cast("string")
    p = lambda path: F.coalesce(F.get_json_object(v, f"$.payload.{path}"),
                                F.get_json_object(v, f"$.{path}"))
 
    df = raw_df.select(
        v.alias("raw_payload"),
        F.col("partition").alias("eh_partition"),
        F.col("offset").alias("eh_offset"),
        F.col("timestamp").alias("ingestion_time"),
        p("op").alias("op"),
        (p("source.ts_ms").cast("long") / 1000).cast("timestamp").alias("source_commit_time"),
        p("after").alias("after_json"),
        p("after.event_time").alias("event_time_raw"),
        p("after.event_type").alias("event_type_raw"),
        p("after.user_id").alias("user_id_raw"),
        p("after.product_id").alias("product_id_raw"),
        p("after.price").alias("price_raw"),
        p("after.user_session").alias("user_session_raw"),
        p("after.platform").alias("platform_raw"),
        p("after.marketing_channel").alias("channel_raw"),
    )
 
    for c in ["event_time_raw", "event_type_raw", "user_id_raw", "product_id_raw",
              "price_raw", "user_session_raw", "platform_raw", "channel_raw"]:
        df = df.withColumn(c, s(c))
 
    df = (df
        .withColumn("event_time", F.coalesce(
            F.to_timestamp("event_time_raw", "yyyy-MM-dd HH:mm:ss 'UTC'"),
            F.to_timestamp("event_time_raw")))
        .withColumn("event_type", F.lower("event_type_raw"))
        .withColumn("user_id", F.col("user_id_raw").cast("long"))
        .withColumn("product_id", F.col("product_id_raw").cast("long"))
        .withColumn("price", F.round(F.col("price_raw").cast("double"), 2))
        .withColumn("user_session", F.col("user_session_raw")))
 
    return df.withColumn("dlq_reason",
        F.when(F.col("after_json").isNull(), F.lit("malformed_json"))
         .when(F.col("event_time_raw").isNull(), F.lit("null_event_time"))
         .when(F.col("event_type_raw").isNull(), F.lit("null_event_type"))
         .when(F.col("user_session_raw").isNull(), F.lit("null_user_session"))
         .when(F.col("user_id_raw").isNull(), F.lit("null_user_id"))
         .when(F.col("product_id_raw").isNull(), F.lit("null_product_id"))
         .when(F.col("price_raw").isNull(), F.lit("null_price"))
         .when(F.col("user_id").isNull() | F.col("product_id").isNull(), F.lit("malformed_json"))
         .when(F.col("event_time").isNull(), F.lit("invalid_event_time"))
         .when(F.col("price").isNull(), F.lit("invalid_price"))
         .when(~F.col("event_type").isin(ALLOWED_TYPES), F.lit("invalid_event_type"))
         .when(F.col("price") <= 0, F.lit("non_positive_price")))
 
VALID_CHANNELS = ["direct", "email", "organic_search", "paid_search",
                  "referral", "social_media", "seasonal_promotion"]
 
def standardize(df):
    plat = F.lower(F.col("platform_raw"))
    chan = F.regexp_replace(F.lower(F.col("channel_raw")), r"[\s\-]+", "_")
    return (df
        .withColumn("platform_clean",
            F.when(plat == "web", F.lit("web"))
             .when(plat.isin("mobile app", "mobile"), F.lit("mobile")))
        .withColumn("channel_clean", F.when(chan.isin(VALID_CHANNELS), chan))
        .withColumn("platform_imputed", F.col("platform_clean").isNull())
        .withColumn("channel_imputed", F.col("channel_clean").isNull())
        .withColumn("event_date", F.to_date("event_time"))
        .withColumn("event_hour", F.hour("event_time")))
 
spark.sql("""
CREATE TABLE IF NOT EXISTS Silver.stream_events (
  event_time TIMESTAMP, event_date DATE, event_hour INT, event_type STRING,
  product_id BIGINT, user_id BIGINT, user_session STRING, price DOUBLE,
  platform STRING, marketing_channel STRING,
  platform_imputed BOOLEAN, channel_imputed BOOLEAN,
  ingestion_time TIMESTAMP, source_commit_time TIMESTAMP, pipeline_latency_s DOUBLE,
  eh_partition INT, eh_offset BIGINT,
  repeat_gap_s BIGINT, is_duplicate_event BOOLEAN, is_bot_like BOOLEAN,
  is_bot_session BOOLEAN, is_impossible_sequence BOOLEAN, exclude_from_kpis BOOLEAN,
  processed_time TIMESTAMP
) USING DELTA
PARTITIONED BY (event_date)
""")
 
import datetime
from pyspark.sql import Window
from delta.tables import DeltaTable
 
TARGET = "Silver.stream_events"
CHECKPOINT = "Files/_checkpoints/stream_silver"
IMPOSSIBLE_MAX_GAP_S = 1
KEY = ["user_session", "event_time", "event_type", "product_id"]
FLAG_COLS = ["repeat_gap_s", "is_duplicate_event", "is_bot_like",
             "is_bot_session", "is_impossible_sequence", "exclude_from_kpis"]
BASE_COLS = ["event_time", "event_date", "event_hour", "event_type", "product_id",
             "user_id", "user_session", "price", "platform", "marketing_channel",
             "platform_imputed", "channel_imputed", "ingestion_time",
             "source_commit_time", "pipeline_latency_s", "eh_partition", "eh_offset"]
 
 
def process_batch(batch_df, batch_id, stage="stream_silver"):
    run_id = f"{stage}-{batch_id}"
    df = standardize(parse_and_validate(batch_df)).persist()
 
    is_ignored = F.coalesce(F.col("op").isin("u", "d"), F.lit(False))
    n_in = df.count()
    n_ignored = df.filter(is_ignored).count()
    work = df.filter(~is_ignored)
 
    bad = work.filter("dlq_reason IS NOT NULL").select(
        F.lit("stream").alias("source"),
        F.lit("events").alias("entity"),
        F.lit(run_id).alias("run_id"),
        "dlq_reason",
        "raw_payload",
        F.concat_ws(":", F.lit("eventhubs"), "eh_partition", "eh_offset").alias("source_file"),
        F.col("ingestion_time").alias("bronze_ingestion_time"),
        F.current_timestamp().alias("dlq_time"),
        F.lit("new").alias("status"))
    n_bad = bad.count()
    write_dlq(bad)
 
    new_rows = work.filter("dlq_reason IS NULL").select(
        "event_time", "event_date", "event_hour", "event_type", "product_id",
        "user_id", "user_session", "price",
        F.col("platform_clean").alias("platform"),
        F.col("channel_clean").alias("marketing_channel"),
        "platform_imputed", "channel_imputed", "ingestion_time", "source_commit_time",
        (F.col("ingestion_time").cast("double")
         - F.col("source_commit_time").cast("double")).alias("pipeline_latency_s"),
        F.col("eh_partition").cast("int").alias("eh_partition"),
        "eh_offset",
        F.lit(1).alias("_is_new"))
 
    lo, hi = new_rows.agg(F.min("event_date"), F.max("event_date")).first()
    if lo is None:
        write_metrics(stage, [("rows_in", n_in), ("ignored_ops", n_ignored),
                              ("dlq_count", n_bad)], run_id)
        df.unpersist()
        print(f"{run_id}: in={n_in} dlq={n_bad} inserted=0")
        return
 
    lo_ctx = lo - datetime.timedelta(days=1)
    hi_ctx = hi + datetime.timedelta(days=1)
 
    sessions = new_rows.select("user_session").distinct()
    existing = (spark.table(TARGET)
        .filter(F.col("event_date").between(F.lit(lo_ctx), F.lit(hi_ctx)))
        .join(F.broadcast(sessions), "user_session", "left_semi")
        .withColumn("platform", F.when(F.col("platform") != "unknown", F.col("platform")))
        .withColumn("marketing_channel",
                    F.when(F.col("marketing_channel") != "unknown", F.col("marketing_channel")))
        .select(*BASE_COLS)
        .withColumn("_is_new", F.lit(0)))
 
    combined = new_rows.unionByName(existing)
 
    w_key = Window.partitionBy(*KEY).orderBy(
        F.col("_is_new").asc(),
        (F.col("platform").isNull() | F.col("marketing_channel").isNull()).asc(),
        F.col("eh_offset").asc())
    ranked = combined.withColumn("_rn", F.row_number().over(w_key))
    n_dups = ranked.filter("_rn > 1 AND _is_new = 1").count()
    rows = ranked.filter("_rn = 1").drop("_rn")
 
    w_session = Window.partitionBy("user_session")
    rows = (rows
        .withColumn("platform", F.coalesce(
            "platform", F.first("platform", ignorenulls=True).over(w_session), F.lit("unknown")))
        .withColumn("marketing_channel", F.coalesce(
            "marketing_channel", F.first("marketing_channel", ignorenulls=True).over(w_session),
            F.lit("unknown"))))
 
    ts = F.col("event_time").cast("long")
    w_spt = Window.partitionBy("user_session", "product_id", "event_type").orderBy("event_time")
    w_sp_before = (Window.partitionBy("user_session", "product_id").orderBy("event_time")
                   .rowsBetween(Window.unboundedPreceding, -1))
    w_s_ord = Window.partitionBy("user_session").orderBy("event_time")
    w_s_60 = Window.partitionBy("user_session").orderBy(ts).rangeBetween(-59, 0)
 
    rows = (rows
        .withColumn("repeat_gap_s", ts - F.lag(ts).over(w_spt))
        .withColumn("is_duplicate_event",
                    F.coalesce(F.col("repeat_gap_s") == 1, F.lit(False)))
        .withColumn("is_bot_like", F.coalesce(
            (F.col("event_type") == "view") & F.col("repeat_gap_s").between(1, 3), F.lit(False)))
        .withColumn("_last_purchase",
                    F.max(F.when(F.col("event_type") == "purchase", ts)).over(w_sp_before))
        .withColumn("is_impossible_sequence", F.coalesce(
            (F.col("event_type") == "cart")
            & (ts - F.col("_last_purchase")).between(1, IMPOSSIBLE_MAX_GAP_S), F.lit(False)))
        .withColumn("_gap_any", ts - F.lag(ts).over(w_s_ord))
        .withColumn("_views_60s", F.sum(
            F.when(F.col("event_type") == "view", 1).otherwise(0)).over(w_s_60)))
 
    bot_sessions = (rows.groupBy("user_session")
        .agg(F.max("_views_60s").alias("mx"),
             F.sum(F.when(F.col("event_type") == "cart", 1).otherwise(0)).alias("carts"),
             F.percentile_approx("_gap_any", 0.5).alias("med"))
        .filter("mx > 100 AND carts = 0 AND med < 2")
        .select("user_session", F.lit(True).alias("is_bot_session")))
 
    final = (rows.join(bot_sessions, "user_session", "left")
        .withColumn("is_bot_session", F.coalesce("is_bot_session", F.lit(False)))
        .withColumn("exclude_from_kpis",
                    F.col("is_duplicate_event") | F.col("is_bot_like") | F.col("is_bot_session"))
        .withColumn("processed_time", F.current_timestamp())
        .select(*BASE_COLS, *FLAG_COLS, "processed_time", "_is_new")
        .persist())
 
    flag_counts = final.filter("_is_new = 1").agg(
        F.count("*").alias("rows_inserted"),
        F.sum(F.col("is_duplicate_event").cast("int")).alias("flag_duplicate_event"),
        F.sum(F.col("is_bot_like").cast("int")).alias("flag_bot_like"),
        F.sum(F.col("is_bot_session").cast("int")).alias("flag_bot_session"),
        F.sum(F.col("is_impossible_sequence").cast("int")).alias("flag_impossible_sequence"),
        F.sum(F.col("platform_imputed").cast("int")).alias("platform_imputed"),
        F.sum(F.col("channel_imputed").cast("int")).alias("channel_imputed"),
        F.avg("pipeline_latency_s").alias("avg_latency_s")).first().asDict()
 
    cond = (f"t.event_date BETWEEN '{lo_ctx}' AND '{hi_ctx}' AND "
            + " AND ".join(f"t.{k} = s.{k}" for k in KEY))
    update_cols = ["platform", "marketing_channel"] + FLAG_COLS + ["processed_time"]
    changed = " OR ".join(f"NOT (t.{c} <=> s.{c})"
                          for c in ["platform", "marketing_channel"] + FLAG_COLS)
 
    (DeltaTable.forName(spark, TARGET).alias("t")
        .merge(final.drop("_is_new").alias("s"), cond)
        .whenMatchedUpdate(condition=changed, set={c: f"s.{c}" for c in update_cols})
        .whenNotMatchedInsertAll()
        .execute())
 
    write_metrics(stage,
        [("rows_in", n_in), ("ignored_ops", n_ignored), ("dlq_count", n_bad),
         ("duplicates_removed", n_dups)] + list(flag_counts.items()),
        run_id)
 
    final.unpersist()
    df.unpersist()
    print(f"{run_id}: in={n_in} dlq={n_bad} dups={n_dups} "
          f"inserted={flag_counts['rows_inserted']}")
 
for q in spark.streams.active:
    q.stop()
 
stream_df = (spark.readStream
    .format("kafka")
    .options(**KAFKA_OPTIONS)
    .option("startingOffsets", "earliest")
    .option("maxOffsetsPerTrigger", 100000)
    .option("minPartitions", "8")
    .option("failOnDataLoss", "false")
    .load())
 
query = (stream_df.writeStream
    .foreachBatch(process_batch)
    .option("checkpointLocation", CHECKPOINT)
    .trigger(processingTime="20 seconds")
    .queryName("stream_silver")
    .start())
 
print("started")
 
for q in spark.streams.active:
    if q.name == "stream_minute_kpis":
        q.stop()
 
KPI_TABLE = "Silver.stream_minute_kpis"
KPI_CHECKPOINT = "Files/_checkpoints/stream_minute_kpis"
 
kpi_source = (spark.readStream
    .format("delta")
    .option("skipChangeCommits", "true")
    .option("maxFilesPerTrigger", 50)
    .table("Silver.stream_events")
    .filter("NOT exclude_from_kpis"))
 
minute_kpis = (kpi_source
    .withWatermark("event_time", "2 minutes")
    .groupBy(F.window("event_time", "1 minute").alias("w"),
             "platform", "marketing_channel")
    .agg(
        F.count("*").alias("events"),
        F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("views"),
        F.sum(F.when(F.col("event_type") == "cart", 1).otherwise(0)).alias("carts"),
        F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("purchases"),
        F.round(F.sum(F.when(F.col("event_type") == "purchase", F.col("price"))
                       .otherwise(0)), 2).alias("revenue"),
        F.approx_count_distinct("user_session").alias("active_sessions"),
        F.round(F.avg("pipeline_latency_s"), 3).alias("avg_latency_s"))
    .select(F.col("w.start").alias("minute_start"),
            F.col("w.end").alias("minute_end"),
            "platform", "marketing_channel", "events", "views", "carts",
            "purchases", "revenue", "active_sessions", "avg_latency_s"))
 
kpi_query = (minute_kpis.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", KPI_CHECKPOINT)
    .trigger(processingTime="10 seconds")
    .queryName("stream_minute_kpis")
    .toTable(KPI_TABLE))
 
print("stream_minute_kpis started")
 
for q in spark.streams.active:
    p = q.lastProgress
    print(f"{q.name:20} | {q.status['message']:28} | "
          f"batch {p['batchId'] if p else '-':>5} | rows {p['numInputRows'] if p else '-':>7}")
 
spark.streams.awaitAnyTermination()