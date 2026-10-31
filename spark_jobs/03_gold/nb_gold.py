from nb_utils import *   # s(), write_dlq(), write_metrics(), F, Window, DeltaTable, spark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
import uuid
import logging
import sys
from datetime import timedelta
from pyspark.sql import functions as F, Window
from delta.tables import DeltaTable

logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s'
)

logging.info(" Initializing Gold Layer Transformations (Star Schema & Marts)...")

spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.conf.set("spark.sql.ansi.enabled", "false")
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true")

RUN_ID = uuid.uuid4().hex[:8]
WM_NAME, LOOKBACK_HOURS = "gold_events", 6    

PLATFORMS   = {"unknown": 0, "web": 1, "mobile": 2}
CHANNELS    = {"unknown": 0, "direct": 1, "email": 2, "organic_search": 3, "paid_search": 4,
               "referral": 5, "social_media": 6, "seasonal_promotion": 7}
EVENT_TYPES = {"view": 1, "cart": 2, "remove_from_cart": 3, "purchase": 4}   # Silver only emits view/cart/purchase

FE_COLS = ["event_id", "user_session", "session_key", "event_ts", "source_event_ts", "ingestion_ts", "product_key",
           "product_id", "user_key", "event_type_key", "event_date_key", "event_time_key", "ingestion_date_key",
           "ingestion_time_key", "platform_key", "channel_key", "is_bot", "event_price", "revenue", "event_count",
           "ingestion_delay_seconds"]
FS_COLS = ["session_key", "user_session", "user_key", "session_start_date_key", "session_start_time_key",
           "platform_key", "channel_key", "session_start_ts", "session_end_ts", "is_bot", "is_first_session",
           "is_impossible_sequence", "has_view", "has_cart", "has_purchase", "view_count", "cart_count",
           "purchase_count", "events_per_session", "session_duration_seconds", "time_to_first_cart_seconds",
           "session_cart_value", "session_revenue", "last_updated_ts"]

def key_map(d):
    return F.create_map(*[x for k, v in d.items() for x in (F.lit(k), F.lit(v))])
def dkey(c): return F.date_format(F.col(c), "yyyyMMdd").cast("int")
def tkey(c): return F.hour(F.col(c)) * 3600 + F.minute(F.col(c)) * 60 + F.second(F.col(c))
def ukey(c): return F.xxhash64(F.col(c).cast("long"))

def gate(name, bad, detail=""):
    """Data-quality gate: raises so the pipeline run fails and the alert fires."""
    if bad:
        raise ValueError(f"[Gold DQ gate] {name} FAILED: {bad} violating row(s). {detail}".strip())

def recon(name, a, b):
    a, b = [tuple(x or 0 for x in r) for r in (a, b)]
    gate(name, int(a != b), f"gold={a} expected={b}")


spark.sql("CREATE SCHEMA IF NOT EXISTS Gold")

TABLES = {
 "dim_product": ("""product_key BIGINT, product_id BIGINT, category_id BIGINT, category_code STRING,
                    category_level1 STRING, category_level2 STRING, category_level3 STRING, brand STRING,
                    base_price DECIMAL(12,2), valid_from TIMESTAMP, valid_to TIMESTAMP, is_current BOOLEAN""", None),
 "dim_user": ("user_key BIGINT, user_id BIGINT", None),
 "dim_date": ("""date_key INT, full_date DATE, year INT, quarter INT, month INT, month_name STRING, day INT,
                 day_of_week_num INT, day_of_week STRING, iso_week INT, is_weekend BOOLEAN, is_seasonal_peak BOOLEAN""", None),
 "dim_time": ("time_key INT, full_time STRING, hour INT, minute INT, second INT, time_category STRING", None),
 "dim_platform": ("platform_key INT, platform_name STRING", None),
 "dim_marketing_channel": ("channel_key INT, channel_name STRING", None),
 "dim_event_type": ("event_type_key INT, event_type_name STRING", None),
 "fact_events": ("""event_id STRING, user_session STRING, session_key BIGINT, event_ts TIMESTAMP, source_event_ts TIMESTAMP,
                    ingestion_ts TIMESTAMP, product_key BIGINT, product_id BIGINT, user_key BIGINT, event_type_key INT,
                    event_date_key INT, event_time_key INT, ingestion_date_key INT, ingestion_time_key INT,
                    platform_key INT, channel_key INT, is_bot INT, event_price DECIMAL(12,2), revenue DECIMAL(12,2),
                    event_count INT, ingestion_delay_seconds INT""", "event_date_key"),
 "fact_sessions": ("""session_key BIGINT, user_session STRING, user_key BIGINT, session_start_date_key INT,
                      session_start_time_key INT, platform_key INT, channel_key INT, session_start_ts TIMESTAMP,
                      session_end_ts TIMESTAMP, is_bot INT, is_first_session INT, is_impossible_sequence INT,
                      has_view INT, has_cart INT, has_purchase INT, view_count INT, cart_count INT, purchase_count INT,
                      events_per_session INT, session_duration_seconds INT, time_to_first_cart_seconds INT,
                      session_cart_value DECIMAL(12,2), session_revenue DECIMAL(12,2), last_updated_ts TIMESTAMP""",
                   "session_start_date_key"),
 "mart_executive_marketing": ("""date_key INT, platform_key INT, channel_key INT, total_sessions BIGINT, total_orders BIGINT,
                                 total_revenue DECIMAL(14,2), view_events BIGINT, purchase_events BIGINT,
                                 first_sessions BIGINT, returning_sessions BIGINT""", None),
 "mart_funnel_behavior": ("""date_key INT, platform_key INT, channel_key INT, view_sessions BIGINT, cart_sessions BIGINT,
                             purchase_sessions BIGINT, abandoned_cart_sessions BIGINT, no_view_sessions BIGINT,
                             lost_revenue DECIMAL(14,2), sum_dwell_time_seconds BIGINT, sum_events BIGINT,
                             sum_time_to_first_cart_seconds BIGINT""", None),
 "mart_product_brand_performance": ("""date_key INT, platform_key INT, channel_key INT, product_key BIGINT, total_views BIGINT,
                                      total_carts BIGINT, total_purchases BIGINT, total_revenue DECIMAL(14,2),
                                      sum_view_price DECIMAL(16,2), price_band STRING""", None),
 "product_price_band": ("product_id BIGINT, price_band STRING", None),     # helper: detects band changes between runs
 "etl_watermark": ("pipeline STRING, last_ingestion_ts TIMESTAMP, updated_at TIMESTAMP", None),
}
for name, (cols, part) in TABLES.items():
    spark.sql(f"CREATE TABLE IF NOT EXISTS Gold.{name} ({cols}) USING DELTA" + (f" PARTITIONED BY ({part})" if part else ""))

def seed(table, mapping):
    vals = ", ".join(f"({v}, '{k}')" for k, v in mapping.items())
    spark.sql(f"INSERT OVERWRITE Gold.{table} VALUES {vals}")

seed("dim_platform", PLATFORMS); seed("dim_marketing_channel", CHANNELS); seed("dim_event_type", EVENT_TYPES)

if spark.table("Gold.dim_date").limit(1).count() == 0:
    spark.sql("""
    INSERT OVERWRITE Gold.dim_date
    SELECT CAST(date_format(d, 'yyyyMMdd') AS INT), d, year(d), quarter(d), month(d), date_format(d, 'MMMM'), day(d),
           weekday(d) + 1, date_format(d, 'EEEE'), weekofyear(d), weekday(d) >= 5, FALSE
    FROM (SELECT explode(sequence(DATE'2019-01-01', DATE'2030-12-31', INTERVAL 1 DAY)) AS d)""")

if spark.table("Gold.dim_time").limit(1).count() == 0:
    spark.sql("""
    INSERT OVERWRITE Gold.dim_time
    SELECT s,
           concat(lpad(CAST(s DIV 3600 AS STRING), 2, '0'), ':', lpad(CAST((s % 3600) DIV 60 AS STRING), 2, '0'), ':',
                  lpad(CAST(s % 60 AS STRING), 2, '0')),
           CAST(s DIV 3600 AS INT), CAST((s % 3600) DIV 60 AS INT), CAST(s % 60 AS INT),
           CASE WHEN s DIV 3600 BETWEEN 6  AND 11 THEN 'Morning'
                WHEN s DIV 3600 BETWEEN 12 AND 17 THEN 'Peak'
                WHEN s DIV 3600 BETWEEN 18 AND 22 THEN 'Evening' ELSE 'Night' END
    FROM (SELECT explode(sequence(0, 86399)) AS s)""")


spark.sql("CREATE OR REPLACE VIEW Gold.dim_ingestion_date AS SELECT * FROM Gold.dim_date")
spark.sql("CREATE OR REPLACE VIEW Gold.dim_ingestion_time AS SELECT * FROM Gold.dim_time")
spark.sql("""
CREATE OR REPLACE VIEW Gold.vw_hourly_activity AS
SELECT e.event_date_key AS date_key, t.hour AS hour_of_day, e.platform_key, e.channel_key, p.category_level1,
       SUM(CASE WHEN e.event_type_key = 1 THEN 1 ELSE 0 END) AS view_events,
       SUM(CASE WHEN e.event_type_key = 2 THEN 1 ELSE 0 END) AS cart_events,
       SUM(CASE WHEN e.event_type_key = 4 THEN 1 ELSE 0 END) AS purchase_events,
       SUM(e.revenue) AS revenue
FROM Gold.fact_events e
JOIN Gold.dim_time    t ON t.time_key    = e.event_time_key
JOIN Gold.dim_product p ON p.product_key = e.product_key
WHERE e.is_bot = 0
GROUP BY e.event_date_key, t.hour, e.platform_key, e.channel_key, p.category_level1""")

def transform(ev, dim_product):
    """ev = Silver.events rows of complete sessions; dim_product = Gold.dim_product.
    Returns (e, fe, fs0): all Silver rows, fact_events source (incl. duplicates, flagged _delete), fact_sessions without is_first_session."""
    is_v, is_c, is_p = [F.col("event_type") == t for t in ("view", "cart", "purchase")]
    zero = F.lit(0).cast("decimal(12,2)")

    e = (ev.select("event_time", "event_type", "product_id", "user_id", "user_session", "platform", "marketing_channel",
                   "ingestion_time", "is_duplicate_event", "is_bot_like", "is_bot_session", "is_impossible_sequence",
                   F.col("price").cast("decimal(12,2)").alias("price"))
           .withColumn("session_key", F.xxhash64("user_session")))

    
    k = (e.filter(~F.col("is_duplicate_event"))
          .withColumn("_bot", (F.col("is_bot_session") | F.col("is_bot_like")).cast("int"))
          .withColumn("_imp", F.col("is_impossible_sequence").cast("int")))
    agg = k.groupBy("session_key", "user_session").agg(
        F.expr("min_by(user_id, event_time)").alias("_uid"),
        F.min("event_time").alias("session_start_ts"), F.max("event_time").alias("session_end_ts"),
        F.expr("min_by(platform, event_time)").alias("_plat"), F.expr("min_by(marketing_channel, event_time)").alias("_chan"),
        F.max("_bot").alias("is_bot"), F.max("_imp").alias("is_impossible_sequence"),
        F.sum(is_v.cast("int")).alias("view_count"), F.sum(is_c.cast("int")).alias("cart_count"),
        F.sum(is_p.cast("int")).alias("purchase_count"), F.count(F.lit(1)).alias("events_per_session"),
        F.min(F.when(is_c, F.col("event_time"))).alias("_first_cart"),
        F.sum(F.when(is_c, F.col("price")).otherwise(zero)).alias("session_cart_value"),
        F.sum(F.when(is_p, F.col("price")).otherwise(zero)).alias("session_revenue"))

    fs0 = agg.select(
        "session_key", "user_session", ukey("_uid").alias("user_key"),
        dkey("session_start_ts").alias("session_start_date_key"), tkey("session_start_ts").alias("session_start_time_key"),
        key_map(PLATFORMS)[F.col("_plat")].alias("platform_key"), key_map(CHANNELS)[F.col("_chan")].alias("channel_key"),
        "session_start_ts", "session_end_ts", "is_bot", "is_impossible_sequence",
        (F.col("view_count") > 0).cast("int").alias("has_view"), (F.col("cart_count") > 0).cast("int").alias("has_cart"),
        (F.col("purchase_count") > 0).cast("int").alias("has_purchase"),
        F.col("view_count").cast("int").alias("view_count"), F.col("cart_count").cast("int").alias("cart_count"),
        F.col("purchase_count").cast("int").alias("purchase_count"), F.col("events_per_session").cast("int").alias("events_per_session"),
        (F.col("session_end_ts").cast("long") - F.col("session_start_ts").cast("long")).cast("int").alias("session_duration_seconds"),
        (F.col("_first_cart").cast("long") - F.col("session_start_ts").cast("long")).cast("int").alias("time_to_first_cart_seconds"),
        F.col("session_cart_value").cast("decimal(12,2)").alias("session_cart_value"),
        F.col("session_revenue").cast("decimal(12,2)").alias("session_revenue"),
        F.current_timestamp().alias("last_updated_ts"))


    dp = (dim_product.filter("product_key <> -1")
          .select(F.col("product_id").alias("d_pid"), F.col("product_key").alias("d_key"),
                  F.col("valid_from").alias("d_from"), F.col("valid_to").alias("d_to")))
    first_ver = (dp.groupBy("d_pid").agg(F.expr("min_by(d_key, d_from)").alias("f_key"))
                   .withColumnRenamed("d_pid", "f_pid"))
    x = (e.withColumn("_pid", F.col("product_id").cast("long"))
          .join(agg.select("session_key", F.col("is_bot").alias("_is_bot")), "session_key", "left")
          .join(dp, (F.col("_pid") == F.col("d_pid")) & (F.col("event_time") >= F.col("d_from"))
                    & (F.col("event_time") < F.col("d_to")), "left")
          .join(first_ver, F.col("_pid") == F.col("f_pid"), "left"))

    fe = x.select(
        F.sha2(F.concat_ws("|", F.col("user_session"), F.col("event_time").cast("long").cast("string"),
                           F.col("event_type"), F.col("_pid").cast("string")), 256).alias("event_id"),
        "user_session", "session_key",
        F.col("event_time").alias("event_ts"), F.col("event_time").alias("source_event_ts"),   
        F.col("ingestion_time").alias("ingestion_ts"),
        F.coalesce(F.col("d_key"), F.col("f_key"), F.lit(-1).cast("long")).alias("product_key"),
        F.col("_pid").alias("product_id"), ukey("user_id").alias("user_key"),
        key_map(EVENT_TYPES)[F.col("event_type")].alias("event_type_key"),
        dkey("event_time").alias("event_date_key"), tkey("event_time").alias("event_time_key"),
        dkey("ingestion_time").alias("ingestion_date_key"), tkey("ingestion_time").alias("ingestion_time_key"),
        key_map(PLATFORMS)[F.col("platform")].alias("platform_key"),
        key_map(CHANNELS)[F.col("marketing_channel")].alias("channel_key"),
        F.col("_is_bot").cast("int").alias("is_bot"),                    
        F.col("price").alias("event_price"),
        F.when(F.col("event_type") == "purchase", F.col("price")).otherwise(zero).alias("revenue"),
        F.lit(1).alias("event_count"),
        (F.col("ingestion_time").cast("long") - F.col("event_time").cast("long")).cast("int").alias("ingestion_delay_seconds"),
        F.col("is_duplicate_event").alias("_delete"))
    return e, fe, fs0


def add_first_session(fs0, gold_sessions):
    """is_first_session = earliest session of the user across Gold.fact_sessions + this batch.
    Returns (fs, rk): rk also lists the user's older sessions so their flag can be corrected if an earlier session shows up late."""
    cols = ["user_key", "session_key", "session_start_ts", "session_start_date_key"]
    cur = fs0.select(*cols)
    prior = (gold_sessions.join(cur.select("user_key").distinct(), "user_key", "left_semi")
                          .join(cur.select("session_key"), "session_key", "left_anti").select(*cols))
    rk = prior.unionByName(cur).withColumn(
        "is_first_session",
        (F.row_number().over(Window.partitionBy("user_key").orderBy("session_start_ts", "session_key")) == 1).cast("int"))
    fs = fs0.join(rk.select("session_key", "is_first_session"), "session_key").select(*FS_COLS)
    return fs, rk


def mart_exec(base):         
    return base.groupBy(F.col("session_start_date_key").alias("date_key"), "platform_key", "channel_key").agg(
        F.count(F.lit(1)).alias("total_sessions"), F.sum("has_purchase").alias("total_orders"),
        F.sum("session_revenue").cast("decimal(14,2)").alias("total_revenue"),
        F.sum("view_count").alias("view_events"), F.sum("purchase_count").alias("purchase_events"),
        F.sum(F.when(F.col("is_first_session") == 1, 1).otherwise(0)).alias("first_sessions"),
        F.sum(F.when(F.col("is_first_session") == 0, 1).otherwise(0)).alias("returning_sessions"))


def mart_funnel(base):     
    v, c, p = F.col("has_view") == 1, F.col("has_cart") == 1, F.col("has_purchase") == 1
    abandoned = v & c & ~p
    zero = F.lit(0).cast("decimal(12,2)")
    return base.groupBy(F.col("session_start_date_key").alias("date_key"), "platform_key", "channel_key").agg(
        F.sum("has_view").alias("view_sessions"),
        F.sum(F.when(v & c, 1).otherwise(0)).alias("cart_sessions"),
        F.sum(F.when(v & p, 1).otherwise(0)).alias("purchase_sessions"),
        F.sum(F.when(abandoned, 1).otherwise(0)).alias("abandoned_cart_sessions"),
        F.sum(F.when(~v, 1).otherwise(0)).alias("no_view_sessions"),
        F.sum(F.when(abandoned, F.col("session_cart_value")).otherwise(zero)).cast("decimal(14,2)").alias("lost_revenue"),
        F.sum(F.when(v, F.col("session_duration_seconds")).otherwise(0)).alias("sum_dwell_time_seconds"),
        F.sum(F.when(v, F.col("events_per_session")).otherwise(0)).alias("sum_events"),
        F.sum(F.when(v & c, F.coalesce(F.col("time_to_first_cart_seconds"), F.lit(0))).otherwise(0)).alias("sum_time_to_first_cart_seconds"))


def price_bands(events):      
    return (events.filter("is_bot = 0 AND event_type_key = 1").groupBy("product_id")
            .agg(F.avg("event_price").alias("avg_price"))
            .withColumn("_nt", F.ntile(3).over(Window.orderBy("avg_price", "product_id")))
            .select("product_id", F.when(F.col("_nt") == 1, "Low").when(F.col("_nt") == 2, "Mid").otherwise("High").alias("price_band")))


def mart_product(events, bands):   
    zero = F.lit(0).cast("decimal(12,2)")
    vw = F.col("event_type_key") == 1
    return (events.join(bands, "product_id", "left")
            .groupBy(F.col("event_date_key").alias("date_key"), "platform_key", "channel_key", "product_key",
                     F.coalesce(F.col("price_band"), F.lit("Unknown")).alias("price_band"))
            .agg(F.sum(F.when(vw, 1).otherwise(0)).alias("total_views"),
                 F.sum(F.when(F.col("event_type_key") == 2, 1).otherwise(0)).alias("total_carts"),
                 F.sum(F.when(F.col("event_type_key") == 4, 1).otherwise(0)).alias("total_purchases"),
                 F.sum("revenue").cast("decimal(14,2)").alias("total_revenue"),
                 F.sum(F.when(vw, F.col("event_price")).otherwise(zero)).cast("decimal(16,2)").alias("sum_view_price"))
            .select("date_key", "platform_key", "channel_key", "product_key", "total_views", "total_carts",
                    "total_purchases", "total_revenue", "sum_view_price", "price_band"))


def check_inputs():
    for t in ("Silver.events", "Silver.dim_product"):
        gate(f"{t} exists", int(not spark.catalog.tableExists(t)))
    # تم إزالة version_no من التحقق
    need = {"Silver.events": {"event_time", "event_type", "product_id", "user_id", "user_session", "price", "platform",
                              "marketing_channel", "ingestion_time", "is_duplicate_event", "is_bot_like",
                              "is_bot_session", "is_impossible_sequence"},
            "Silver.dim_product": {"product_sk", "product_id", "price", "category_id", "category_code",
                                   "category_l1", "category_l2", "category_l3", "brand", "valid_from", "valid_to", "is_current"}}
    for t, cols in need.items():
        gate(f"{t} missing columns", len(cols - set(spark.table(t).columns)), str(sorted(cols - set(spark.table(t).columns))))
    gate("Silver.dim_product is empty", int(spark.table("Silver.dim_product").limit(1).count() == 0))


def refresh_dim_product():
    """Gold.dim_product mirrors Silver.dim_product (product_key = product_sk) + the -1 Unknown member. Small table: full overwrite."""
    U = F.lit("Unknown")
    dp = spark.table("Silver.dim_product").select(
        F.col("product_sk").cast("long").alias("product_key"), F.col("product_id").cast("long").alias("product_id"),
        F.col("category_id").cast("long").alias("category_id"),
        F.coalesce(F.col("category_code"), U).alias("category_code"),
        F.coalesce(F.col("category_l1"), U).alias("category_level1"), F.coalesce(F.col("category_l2"), U).alias("category_level2"),
        F.coalesce(F.col("category_l3"), U).alias("category_level3"), F.coalesce(F.col("brand"), U).alias("brand"),
        F.col("price").cast("decimal(12,2)").alias("base_price"), "valid_from", "valid_to", "is_current")
   
    unk = spark.sql("""SELECT CAST(-1 AS BIGINT) product_key, CAST(-1 AS BIGINT) product_id,
        CAST(NULL AS BIGINT) category_id, 'Unknown' category_code, 'Unknown' category_level1, 'Unknown' category_level2,
        'Unknown' category_level3, 'Unknown' brand, CAST(NULL AS DECIMAL(12,2)) base_price,
        TIMESTAMP'1970-01-01 00:00:00' valid_from, TIMESTAMP'9999-12-31 00:00:00' valid_to, TRUE is_current""")
    dp = dp.unionByName(unk).persist()
    gate("dim_product: duplicate product_key", dp.count() - dp.select("product_key").distinct().count())
    gate("dim_product: valid_from >= valid_to", dp.filter("valid_from >= valid_to").count())
    gate("dim_product: product without exactly one current version",
         dp.filter("product_id <> -1").groupBy("product_id").agg(F.sum(F.col("is_current").cast("int")).alias("n"))
           .filter("n <> 1").count())
    dp.write.format("delta").mode("overwrite").saveAsTable("Gold.dim_product")
    dp.unpersist()


def load_facts():
    """Incremental: find sessions that got new Silver rows since the watermark, rebuild those sessions completely, MERGE."""
    silver = spark.table("Silver.events")
    row = spark.table("Gold.etl_watermark").filter(F.col("pipeline") == WM_NAME).first()
    wm = row["last_ingestion_ts"] if row else None
    new_wm = silver.agg(F.max("ingestion_time")).first()[0]
    if new_wm is None or (wm is not None and new_wm <= wm):
        print("Nothing new in Silver.events"); return None

    if wm is None:
        ev = silver                                                     
    else:
        aff = silver.filter(F.col("ingestion_time") > F.lit(wm - timedelta(hours=LOOKBACK_HOURS))).select("user_session").distinct()
        ev = silver.join(aff, "user_session", "left_semi")              

    # ---- dim_user (SCD1, business key only) ----
    users = ev.select(F.col("user_id").cast("long").alias("user_id")).distinct().withColumn("user_key", F.xxhash64("user_id"))
    gate("dim_user: null user_id", users.filter("user_id IS NULL").count())
    (DeltaTable.forName(spark, "Gold.dim_user").alias("t").merge(users.alias("s"), "t.user_key = s.user_key")
        .whenNotMatchedInsertAll().execute())

    # ---- transform ----
    e, fe, fs0 = transform(ev, spark.table("Gold.dim_product"))
    e, fe, fs0 = e.persist(), fe.persist(), fs0.persist()
    gs = spark.table("Gold.fact_sessions")
    fs, rk = add_first_session(fs0, gs)
    fs, rk = fs.persist(), rk.persist()
    rk.count()                                                         

    # ---- pre-merge gates ----
    fk = fe.filter(~F.col("_delete"))
    c = fk.agg(F.count(F.lit(1)).alias("n"), F.countDistinct("event_id").alias("n_ids"), F.sum("revenue").alias("rev"),
               F.sum((F.col("platform_key").isNull() | F.col("channel_key").isNull() | F.col("event_type_key").isNull()
                      | F.col("user_key").isNull()).cast("int")).alias("null_keys"),
               F.sum((F.col("event_price") <= 0).cast("int")).alias("bad_price"),
               F.sum((F.col("ingestion_delay_seconds") < 0).cast("int")).alias("neg_delay"),
               F.sum((F.col("product_key") == -1).cast("int")).alias("unknown_product")).first()
    gate("fact_events: duplicate event_id", c["n"] - c["n_ids"])
    gate("fact_events: unmapped platform / channel / event_type / user", c["null_keys"] or 0,
         "a new value appeared in Silver -> add it to PLATFORMS / CHANNELS / EVENT_TYPES")
    gate("fact_events: event_price <= 0", c["bad_price"] or 0)
    gate("fact_events: ingestion_delay_seconds < 0", c["neg_delay"] or 0)
    s = fs.agg(F.count(F.lit(1)).alias("n"), F.countDistinct("session_key").alias("n_keys"),
               F.sum("events_per_session").alias("ev"), F.sum("session_revenue").alias("rev"),
               F.sum(((F.col("events_per_session") < 1) | (F.col("session_duration_seconds") < 0)
                      | (F.col("session_start_ts") > F.col("session_end_ts")) | (F.col("session_cart_value") < 0)
                      | (F.col("session_revenue") < 0)).cast("int")).alias("bad_measures"),
               F.sum((F.col("platform_key").isNull() | F.col("channel_key").isNull()).cast("int")).alias("null_keys")).first()
    gate("fact_sessions: duplicate session_key in batch", s["n"] - s["n_keys"])
    gate("fact_sessions: invalid measures", s["bad_measures"] or 0)
    gate("fact_sessions: unmapped platform / channel", s["null_keys"] or 0)
    recon("batch: fact_events vs fact_sessions", (c["n"], c["rev"]), (s["ev"], s["rev"]))

    # ---- date ranges (partition pruning + which mart dates to rebuild) ----
    e_lo, e_hi = fe.agg(F.min("event_date_key"), F.max("event_date_key")).first()
    n_lo, n_hi = fs.agg(F.min("session_start_date_key"), F.max("session_start_date_key")).first()
    o_lo, o_hi = gs.join(fs.select("session_key"), "session_key", "left_semi") \
                   .agg(F.min("session_start_date_key"), F.max("session_start_date_key")).first()   
    s_lo, s_hi = min(x for x in (n_lo, o_lo) if x is not None), max(x for x in (n_hi, o_hi) if x is not None)

    # ---- MERGE fact_events (duplicates flagged by Silver are removed, changed flags / keys are updated) ----
    if spark.table("Gold.fact_events").limit(1).count() == 0:
        fk.select(*FE_COLS).write.format("delta").mode("append").saveAsTable("Gold.fact_events")
    else:
        upd = ["product_key", "platform_key", "channel_key", "is_bot"]
        changed = " OR ".join(f"NOT (t.{c_} <=> s.{c_})" for c_ in upd)
        (DeltaTable.forName(spark, "Gold.fact_events").alias("t")
            .merge(fe.alias("s"), f"t.event_date_key BETWEEN {e_lo} AND {e_hi} AND t.event_id = s.event_id")
            .whenMatchedDelete(condition="s._delete")
            .whenMatchedUpdate(condition=f"NOT s._delete AND ({changed})", set={c_: f"s.{c_}" for c_ in upd})
            .whenNotMatchedInsert(condition="NOT s._delete", values={c_: f"s.{c_}" for c_ in FE_COLS})
            .execute())

    # ---- MERGE fact_sessions (accumulating snapshot) ----
    if gs.limit(1).count() == 0:
        fs.write.format("delta").mode("append").saveAsTable("Gold.fact_sessions")
    else:
        cols = [c_ for c_ in FS_COLS if c_ != "session_key"]
        changed = " OR ".join(f"NOT (t.{c_} <=> s.{c_})" for c_ in cols if c_ != "last_updated_ts")
        (DeltaTable.forName(spark, "Gold.fact_sessions").alias("t")
            .merge(fs.alias("s"), f"t.session_start_date_key BETWEEN {s_lo} AND {s_hi} AND t.session_key = s.session_key")
            .whenMatchedUpdate(condition=changed, set={c_: f"s.{c_}" for c_ in cols})
            .whenNotMatchedInsertAll().execute())

    # ---- older sessions of the same users whose is_first_session flipped because an earlier session arrived late ----
    chg = (rk.join(fs.select("session_key"), "session_key", "left_anti")
             .join(spark.table("Gold.fact_sessions").select("session_key", F.col("is_first_session").alias("_old")), "session_key")
             .filter(F.col("is_first_session") != F.col("_old"))
             .select("session_key", "session_start_date_key", "is_first_session").persist())
    c_lo = c_hi = None
    if chg.limit(1).count() > 0:
        (DeltaTable.forName(spark, "Gold.fact_sessions").alias("t")
            .merge(chg.alias("s"), "t.session_start_date_key = s.session_start_date_key AND t.session_key = s.session_key")
            .whenMatchedUpdate(set={"is_first_session": "s.is_first_session", "last_updated_ts": "current_timestamp()"}).execute())
        c_lo, c_hi = chg.agg(F.min("session_start_date_key"), F.max("session_start_date_key")).first()
    m_lo, m_hi = min(x for x in (s_lo, c_lo) if x is not None), max(x for x in (s_hi, c_hi) if x is not None)

    # ---- post-merge gates ----
    ks = fs.select("session_key")
    ge = (spark.table("Gold.fact_events").filter(F.col("event_date_key").between(e_lo, e_hi))
               .join(ks, "session_key", "left_semi").agg(F.count(F.lit(1)), F.sum("revenue")).first())
    gsx = (spark.table("Gold.fact_sessions").join(ks, "session_key", "left_semi")
                .agg(F.sum("events_per_session"), F.sum("session_revenue")).first())
    recon("Gold.fact_events vs Gold.fact_sessions (touched sessions)", ge, gsx)
    gate("fact_sessions: duplicate session_key", spark.table("Gold.fact_sessions").groupBy("session_key").count().filter("count > 1").count())
    gate("fact_sessions: user_key missing from dim_user", fs.join(spark.table("Gold.dim_user"), "user_key", "left_anti").count())

    out = dict(new_wm=new_wm, e_lo=e_lo, e_hi=e_hi, m_lo=m_lo, m_hi=m_hi, sessions=s["n"], events=c["n"],
               duplicates_removed=fe.filter("_delete").count(), unknown_product_events=c["unknown_product"] or 0,
               first_flips=chg.count())
    for d in (e, fe, fs0, fs, rk, chg): d.unpersist()
    return out


def replace_range(df, table, lo, hi):
    (df.write.format("delta").mode("overwrite").option("replaceWhere", f"date_key BETWEEN {lo} AND {hi}").saveAsTable(table))


def build_marts(r):
    fsess, fev = spark.table("Gold.fact_sessions"), spark.table("Gold.fact_events")
    lo, hi = r["m_lo"], r["m_hi"]
    base = fsess.filter(f"is_bot = 0 AND session_start_date_key BETWEEN {lo} AND {hi}").persist()
    replace_range(mart_exec(base), "Gold.mart_executive_marketing", lo, hi)
    replace_range(mart_funnel(base), "Gold.mart_funnel_behavior", lo, hi)

    # price bands depend on the whole product mix: if any product changed band, rebuild mart 3 over all dates
    bands = price_bands(fev).persist()
    changed = (bands.join(spark.table("Gold.product_price_band").withColumnRenamed("price_band", "_old"), "product_id", "full_outer")
                    .filter(~F.col("price_band").eqNullSafe(F.col("_old"))).limit(1).count() > 0)
    p_lo, p_hi = (fev.agg(F.min("event_date_key"), F.max("event_date_key")).first() if changed else (r["e_lo"], r["e_hi"]))
    if changed:
        bands.write.format("delta").mode("overwrite").saveAsTable("Gold.product_price_band")
    ev_nb = fev.filter(f"is_bot = 0 AND event_date_key BETWEEN {p_lo} AND {p_hi}")
    replace_range(mart_product(ev_nb, bands), "Gold.mart_product_brand_performance", p_lo, p_hi)

    # ---- mart gates: marts must reconcile with the facts for the rebuilt range ----
    m1 = spark.table("Gold.mart_executive_marketing").filter(F.col("date_key").between(lo, hi)) \
              .agg(F.sum("total_sessions"), F.sum("total_orders"), F.sum("total_revenue")).first()
    recon("mart_executive_marketing vs fact_sessions", m1,
          base.agg(F.count(F.lit(1)), F.sum("has_purchase"), F.sum("session_revenue")).first())
    m2 = spark.table("Gold.mart_funnel_behavior").filter(F.col("date_key").between(lo, hi)) \
              .agg(F.sum(F.col("view_sessions") + F.col("no_view_sessions")), F.sum("purchase_sessions")).first()
    recon("mart_funnel_behavior vs fact_sessions", m2,
          base.agg(F.count(F.lit(1)), F.sum(F.when((F.col("has_view") == 1) & (F.col("has_purchase") == 1), 1).otherwise(0))).first())
    m3 = spark.table("Gold.mart_product_brand_performance").filter(F.col("date_key").between(p_lo, p_hi)) \
              .agg(F.sum("total_revenue"), F.sum("total_views"), F.sum("total_purchases")).first()
    recon("mart_product_brand_performance vs fact_events", m3,
          ev_nb.agg(F.sum("revenue"), F.sum(F.when(F.col("event_type_key") == 1, 1).otherwise(0)),
                    F.sum(F.when(F.col("event_type_key") == 4, 1).otherwise(0))).first())
    base.unpersist(); bands.unpersist()


def save_watermark(ts):
    df = spark.createDataFrame([(WM_NAME, ts)], "pipeline string, last_ingestion_ts timestamp").withColumn("updated_at", F.current_timestamp())
    (DeltaTable.forName(spark, "Gold.etl_watermark").alias("t").merge(df.alias("s"), "t.pipeline = s.pipeline")
        .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())


def run():
    check_inputs()
    refresh_dim_product()
    r = load_facts()
    if r is None:
        return
    build_marts(r)
    save_watermark(r["new_wm"])          
    write_metrics("gold/events", [("sessions_rebuilt", r["sessions"]), ("events_loaded", r["events"]),
                                  ("duplicates_removed", r["duplicates_removed"]), ("unknown_product_events", r["unknown_product_events"]),
                                  ("is_first_session_flips", r["first_flips"])], RUN_ID)


def run():
    logging.info(" Starting Gold pipeline execution...")
    
    logging.info(" Checking inputs and table schemas...")
    check_inputs()
    
    logging.info(" Refreshing Gold.dim_product...")
    refresh_dim_product()
    
    logging.info(" Loading incremental facts (fact_events & fact_sessions)...")
    r = load_facts()
    
    if r is None:
        logging.info(" Pipeline finished early: No new data to process.")
        return
        
    logging.info(f"   -> Processed {r['events']:,} events across {r['sessions']:,} sessions.")
    logging.info(f"   -> Removed {r['duplicates_removed']:,} duplicates.")
    
    if r['unknown_product_events'] > 0:
        logging.warning(f" Found {r['unknown_product_events']:,} events with unknown products (-1).")
        
    logging.info(" Building/Updating Executive Data Marts...")
    build_marts(r)
    
    logging.info(f" Saving ETL Watermark ({r['new_wm']})...")
    save_watermark(r["new_wm"])
    
    logging.info(" Writing pipeline metrics...")
    write_metrics("gold/events", [("sessions_rebuilt", r["sessions"]), ("events_loaded", r["events"]),
                                  ("duplicates_removed", r["duplicates_removed"]), ("unknown_product_events", r["unknown_product_events"]),
                                  ("is_first_session_flips", r["first_flips"])], RUN_ID)
                                  
    logging.info(" Gold Pipeline finished successfully!")

run()