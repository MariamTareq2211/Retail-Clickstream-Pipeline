from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

import logging
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import col, from_json, lit, to_timestamp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(" Starting streaming ingestion from Event Hub Capture...")

CHECKPOINT_PATH = "Files/_checkpoints/bronze_events_stream"

payload_schema = StructType([
    StructField("event_time", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("price", StringType(), True),
    StructField("user_id", StringType(), True),   
    StructField("user_session", StringType(), True),
    StructField("platform", StringType(), True),
    StructField("marketing_channel", StringType(), True)
])


cdc_schema = StructType([
    StructField("after", payload_schema, True)
])


raw_avro_schema = (spark.read
    .format("avro")
    .option("recursiveFileLookup", "true")
    .option("pathGlobFilter", "*.avro")
    .load("Files/stream-landing-zone")
    .schema
)


df_raw = (spark.readStream
    .format("avro")
    .schema(raw_avro_schema)
    .option("recursiveFileLookup", "true")
    .option("pathGlobFilter", "*.avro")
    .load("Files/stream-landing-zone")
)


df_final = (df_raw
    .withColumn("json_string", col("Body").cast("string")) 
    .withColumn("data", from_json(col("json_string"), cdc_schema)) 
    .select(
        col("data.after.event_time").alias("event_time"),
        col("data.after.event_type").alias("event_type"),
        col("data.after.product_id").cast("long").alias("product_id"),
        col("data.after.price").alias("price"),
        col("data.after.user_id").cast("long").alias("user_id"),    
        col("data.after.user_session").alias("user_session"),
        col("data.after.platform").alias("platform"),
        col("data.after.marketing_channel").alias("marketing_channel"),
        to_timestamp(col("EnqueuedTimeUtc"), "M/d/yyyy h:mm:ss a").alias("ingestion_time"),
        lit("EventHub_Capture").alias("source_file"),
        lit(None).cast("string").alias("_corrupt_record"),
        lit("Stream").alias("source")
    )
)


query = (df_final.writeStream
    .format("delta")
    .outputMode("append") 
    .option("checkpointLocation", CHECKPOINT_PATH)
    .trigger(availableNow=True)
    .toTable("Bronze.events_stream")
)

query.awaitTermination()
logging.info(" Streaming ingestion completed successfully.")