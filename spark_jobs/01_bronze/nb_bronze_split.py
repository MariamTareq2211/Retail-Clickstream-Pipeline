from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

import logging
import sys
from pyspark.sql import functions as F, Window

logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s'
)

logging.info("Starting Bronze Transformation (Catalog & Events Split)...")

logging.info("Building Product Catalog (SCD logic)...")

raw_batch = spark.table("Bronze.batch_raw")

p = (raw_batch.withColumn("event_ts", F.to_timestamp("event_time"))
      .withColumn("price_d", F.col("price").cast("double"))
      .filter(F.col("product_id").isNotNull() & F.col("event_ts").isNotNull() & (F.col("price_d") > 0)))

sig = F.concat_ws("|", F.col("price_d").cast("string"),
                  F.coalesce("brand", F.lit("")),
                  F.coalesce("category_code", F.lit("")),
                  F.coalesce(F.col("category_id").cast("string"), F.lit("")))

w = Window.partitionBy("product_id").orderBy("event_ts")

cat = (p.withColumn("sig", sig).withColumn("prev", F.lag("sig").over(w))
        .filter(F.col("prev").isNull() | (F.col("sig") != F.col("prev")))
        .select("product_id","category_id","category_code","brand",
                F.col("price_d").alias("price"),
                F.col("event_ts").alias("effective_from"), "ingestion_time"))

logging.info("Overwriting 'Bronze.product_catalog' table...")
(cat.write.format("delta")
    .mode("overwrite")
    .saveAsTable("Bronze.product_catalog"))

catalog_count = spark.table("Bronze.product_catalog").count()
logging.info(f" Product Catalog updated successfully. Total unique product states: {catalog_count:,}")

logging.info("Initializing Bronze.events Stream extraction...")
events_checkpoint = "Files/_checkpoints/bronze_events_split"

query = (spark.readStream.format("delta").table("Bronze.batch_raw")
    .select("event_time","event_type","product_id","price","user_id",
            "user_session","platform","marketing_channel","ingestion_time",
            "source_file", "_corrupt_record")
    .withColumn("source", F.lit("batch_workaround"))
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", events_checkpoint)
    .trigger(availableNow=True)
    .toTable("Bronze.events"))

logging.info(" Events Stream started. Waiting for completion...")

query.awaitTermination()

if query.lastProgress:
    rows_processed = query.lastProgress.get('numInputRows', 0)
    logging.info(f" Events Stream completed! Appended {rows_processed:,} rows to 'Bronze.events'.")
else:
    logging.info(" Events Stream completed! No new rows processed.")

logging.info(" Bronze Transformation Pipeline finished successfully.")