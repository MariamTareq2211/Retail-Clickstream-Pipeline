from nb_utils import *  
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

import logging
import sys
import uuid

logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s'
)

logging.info(" Initializing Silver Layer Transformations...")

spark.conf.set("spark.sql.ansi.enabled", "false")
spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true")

RUN_ID, CHECKPOINT = uuid.uuid4().hex[:8], "Files/_checkpoints/silver_events"
MAX_FILES_PER_BATCH = 50
LOOKBACK_DAYS = 1
ALLOWED_TYPES    = ["view", "cart", "purchase"]
ALLOWED_CHANNELS = ["direct", "email", "organic_search", "paid_search", "referral",
                    "social_media", "seasonal_promotion"] 
BOT_MIN_VIEWS_60S, BOT_MAX_MEDIAN_GAP = 100, 2
DUP_GAP_S, BOT_REPEAT_MAX_GAP_S, IMPOSSIBLE_MAX_GAP_S = 1, 3, 1  

KEY  = ["user_session", "event_time", "event_type", "product_id"]
BASE = ["event_time", "event_date", "event_hour", "event_type", "product_id", "user_id", "user_session", "price",
        "platform", "marketing_channel", "platform_imputed", "channel_imputed", "ingestion_time", "source_file"]
FLAG_ONLY   = ["repeat_gap_s", "is_duplicate_event", "is_bot_like", "is_bot_session",
               "is_impossible_sequence", "exclude_from_kpis"]
BOOL_FLAGS  = [c for c in FLAG_ONLY if c != "repeat_gap_s"]
SILVER_COLS = BASE + FLAG_ONLY
UPDATABLE   = ["platform", "marketing_channel"] + FLAG_ONLY

def validate(raw):
    has_corrupt = "_corrupt_record" in raw.columns
    cols = ["event_time","event_type","product_id","price","user_id","user_session","platform","marketing_channel"]
    p = (raw.withColumn("payload", F.to_json(F.struct(*cols)))
            .withColumn("ts_str", s("event_time"))
            .withColumn("event_ts", F.coalesce(F.to_timestamp("ts_str", "yyyy-MM-dd HH:mm:ss 'UTC'"),
                                               F.to_timestamp("ts_str")))
            .withColumn("price_s", s("price")).withColumn("price_d", F.col("price_s").cast("double"))
            .withColumn("etype", F.lower(s("event_type"))).withColumn("sess", s("user_session")))
    corrupt = F.col("_corrupt_record").isNotNull() if has_corrupt else F.lit(False)
    reason = (F.when(corrupt, "malformed_row")
               .when(F.col("ts_str").isNull(), "null_event_time")
               .when(F.col("etype").isNull(), "null_event_type")
               .when(F.col("sess").isNull(), "null_user_session")
               .when(F.col("user_id").isNull(), "null_user_id")
               .when(F.col("product_id").isNull(), "null_product_id")
               .when(F.col("price_s").isNull(), "null_price")
               .when(F.col("event_ts").isNull(), "invalid_event_time")
               .when(F.col("price_d").isNull(), "invalid_price")
               .when(~F.col("etype").isin(ALLOWED_TYPES), "invalid_event_type")
               .when(F.col("price_d") <= 0, "non_positive_price"))
    p = p.withColumn("dlq_reason", reason)
    payload = F.coalesce("_corrupt_record", "payload") if has_corrupt else F.col("payload")
    dlq = p.filter("dlq_reason IS NOT NULL").select(
        F.lit("batch").alias("source"), F.lit("events").alias("entity"), F.lit(RUN_ID).alias("run_id"),
        "dlq_reason", payload.alias("raw_payload"), "source_file",
        F.col("ingestion_time").alias("bronze_ingestion_time"),
        F.current_timestamp().alias("dlq_time"), F.lit("new").alias("status"))
    return p.filter("dlq_reason IS NULL"), dlq


def standardize(good):
    p0 = F.regexp_replace(F.lower(s("platform")), r"[\s\-]+", "_")
    c0 = F.regexp_replace(F.lower(s("marketing_channel")), r"[\s\-]+", "_")
    plat = F.when(p0 == "web", "web").when(p0.isin("mobile", "mobile_app"), "mobile")
    chan = F.when(c0.isin(ALLOWED_CHANNELS), c0)
    return (good.withColumn("_p", plat).withColumn("_c", chan).select(
        F.col("event_ts").alias("event_time"), F.to_date("event_ts").alias("event_date"),
        F.hour("event_ts").alias("event_hour"), F.col("etype").alias("event_type"),
        "product_id", "user_id", F.col("sess").alias("user_session"), F.round("price_d", 2).alias("price"),
        F.coalesce("_p", F.lit("unknown")).alias("platform"),
        F.coalesce("_c", F.lit("unknown")).alias("marketing_channel"),
        F.col("_p").isNull().alias("platform_imputed"), F.col("_c").isNull().alias("channel_imputed"),
        "ingestion_time", "source_file"))


def add_flags(ctx):
    is_v, is_c, is_p = (F.col("event_type") == "view"), (F.col("event_type") == "cart"), (F.col("event_type") == "purchase")
    ts = F.col("event_time").cast("long")

    w_rep = Window.partitionBy("user_session", "product_id", "event_type").orderBy("event_time")
    w_buy = (Window.partitionBy("user_session", "product_id").orderBy("event_time")
                   .rowsBetween(Window.unboundedPreceding, -1))
    gap = ts - F.lag(ts).over(w_rep)                                  
    since_buy = ts - F.max(F.when(is_p, ts)).over(w_buy)             

    ctx = (ctx
      .withColumn("repeat_gap_s", gap.cast("int"))
      .withColumn("is_duplicate_event", F.coalesce(F.col("repeat_gap_s") == DUP_GAP_S, F.lit(False)))        
      .withColumn("is_bot_like", is_v & F.coalesce(F.col("repeat_gap_s").between(1, BOT_REPEAT_MAX_GAP_S), F.lit(False)))
      .withColumn("is_impossible_sequence",                                                                   
                  is_c & F.coalesce((since_buy > 0) & (since_buy <= IMPOSSIBLE_MAX_GAP_S), F.lit(False)))
      .persist())

    cand = (ctx.groupBy("user_session")
               .agg(F.sum(is_v.cast("int")).alias("n_view"), F.sum(is_c.cast("int")).alias("n_cart"))
               .filter((F.col("n_view") > BOT_MIN_VIEWS_60S) & (F.col("n_cart") == 0)).select("user_session"))
    v = (ctx.filter(is_v).join(cand, "user_session")
            .withColumn("views_60s", F.count("*").over(Window.partitionBy("user_session").orderBy(ts).rangeBetween(-59, 0)))
            .withColumn("gap_s", ts - F.lag(ts).over(Window.partitionBy("user_session").orderBy("event_time"))))
    bots = (v.groupBy("user_session")
              .agg(F.max("views_60s").alias("max_views_60s"), F.percentile_approx("gap_s", 0.5).alias("median_gap_s"))
              .filter((F.col("max_views_60s") > BOT_MIN_VIEWS_60S) & (F.col("median_gap_s") < BOT_MAX_MEDIAN_GAP))
              .select("user_session", F.lit(True).alias("is_bot_session")))
    return (ctx.join(bots, "user_session", "left")
               .withColumn("is_bot_session", F.coalesce("is_bot_session", F.lit(False)))
               .withColumn("exclude_from_kpis",
                           F.col("is_duplicate_event") | F.col("is_bot_like") | F.col("is_bot_session")))


def build_context(new):
    sessions = new.select("user_session").distinct()
    lo, hi = new.agg(F.min("event_date"), F.max("event_date")).first()
    ctx = new.select(*BASE).withColumn("_src", F.lit(1))
    if spark.catalog.tableExists("Silver.events"):
        old = (spark.table("Silver.events")
                 .filter(F.col("event_date").between(F.date_sub(F.lit(lo), LOOKBACK_DAYS),
                                                     F.date_add(F.lit(hi), LOOKBACK_DAYS)))
                 .join(sessions, "user_session", "left_semi")
                 .select(*BASE).withColumn("_src", F.lit(0)))
        ctx = old.unionByName(ctx)


    ctx = (ctx.withColumn("_rn", F.row_number().over(
                Window.partitionBy(*KEY).orderBy("_src", "ingestion_time", "source_file")))
              .filter("_rn = 1").drop("_rn"))


    w_s = Window.partitionBy("user_session")
    _p = F.when(~F.col("platform_imputed"), F.col("platform"))
    _c = F.when(~F.col("channel_imputed"), F.col("marketing_channel"))
    ctx = (ctx.withColumn("platform", F.coalesce(_p, F.max(_p).over(w_s), F.lit("unknown")))
              .withColumn("marketing_channel", F.coalesce(_c, F.max(_c).over(w_s), F.lit("unknown"))))

    _incomplete = (F.col("platform_imputed") | F.col("channel_imputed")).cast("int")
    ctx = (ctx.withColumn("_rn", F.row_number().over(
               Window.partitionBy(*KEY).orderBy("_src", _incomplete, "ingestion_time", "source_file")))
             .filter("_rn = 1").drop("_rn"))

    return add_flags(ctx).persist()


def merge_events(ctx):
    out = ctx.select(*SILVER_COLS)
    if not spark.catalog.tableExists("Silver.events"):
        out.write.format("delta").partitionBy("event_date").saveAsTable("Silver.events")
        return
    lo, hi = out.agg(F.min("event_date"), F.max("event_date")).first()
    cond = f"t.event_date BETWEEN '{lo}' AND '{hi}' AND " + " AND ".join(f"t.{k} = s.{k}" for k in KEY)
    changed = " OR ".join(f"NOT (t.{c} <=> s.{c})" for c in UPDATABLE)
    (DeltaTable.forName(spark, "Silver.events").alias("t").merge(out.alias("s"), cond)
        .whenMatchedUpdate(condition=changed, set={c: f"s.{c}" for c in UPDATABLE})
        .whenNotMatchedInsertAll().execute())


def process(raw, batch_id):
    logging.info(f"--- Starting processing for Batch {batch_id} ---")
    raw = raw.persist()
    n_in = raw.count()
    logging.info(f" Batch {batch_id}: Read {n_in:,} rows from Bronze.")
    good, dlq = validate(raw)
    good = good.persist(); dlq = dlq.persist()
    n_good = good.count()
    n_dlq = n_in - n_good
    logging.info(f"Validation Done: {n_good:,} Good rows, {n_dlq:,} DLQ rows.")

    reasons = {r["dlq_reason"]: r["count"] for r in dlq.groupBy("dlq_reason").count().collect()}
    write_dlq(dlq)
    
    n_new, fl = 0, {}
    if n_good > 0:
        logging.info(f" Building Context and extracting Flags for Batch {batch_id}...")
        ctx = build_context(standardize(good))
        new_rows = ctx.filter("_src = 1")
        n_new = new_rows.count()
        logging.info(f" Merging {n_new:,} new rows into Silver.events...")
        merge_events(ctx)
        fl = new_rows.agg(*[F.sum(F.col(c).cast("int")).alias(c) for c in BOOL_FLAGS + ["platform_imputed", "channel_imputed"]]).first().asDict()
        ctx.unpersist()
    else:
        logging.info(" No good rows to merge in this batch.")
    
    logging.info(f" Writing metrics to pipeline_metrics table...")
    write_metrics(f"silver_events/b{batch_id}",
                  [("bronze_in", n_in), 
                   ("dlq", n_in - n_good), 
                   ("duplicates_removed", n_good - n_new), 
                   ("inserted", n_new)]
                  + [(f"flag_{k}", v) for k, v in fl.items()] + [(f"dlq_{k}", v) for k, v in reasons.items()], RUN_ID)
                  
    raw.unpersist(); good.unpersist(); dlq.unpersist()
    
    logging.info(f" --- Completed processing for Batch {batch_id} ---")

logging.info(" Starting Silver Events Stream (AvailableNow)...")
query = (spark.readStream
      .option("startingVersion", "0") 
      .option("maxFilesPerTrigger", MAX_FILES_PER_BATCH)
      .table("Bronze.events")
      .writeStream.foreachBatch(process)
      .option("checkpointLocation", CHECKPOINT + "_v2")
      .trigger(availableNow=True).start())

query.awaitTermination()

logging.info(" Silver Layer Transformations finished successfully.")