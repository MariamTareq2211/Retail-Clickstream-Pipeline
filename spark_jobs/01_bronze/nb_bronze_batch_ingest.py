from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

import logging
import sys
from pyspark.sql import functions as F
from pyspark.sql.types import *

logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s'
)

logging.info("Starting Bronze Layer Ingestion...")

landing = "abfss://Retail_Clickstream_Pipeline@onelake.dfs.fabric.microsoft.com/Retail_Clickstream_Lakehouse.Lakehouse/Files/landing-zone/"  
checkpoint = "Files/_checkpoints/bronze_batch_raw"

schema_def = StructType([
    StructField("event_time", StringType()), StructField("event_type", StringType()),
    StructField("product_id", LongType()),   StructField("category_id", LongType()),
    StructField("category_code", StringType()), StructField("brand", StringType()),
    StructField("price", StringType()),   
    StructField("user_id", LongType()),      StructField("user_session", StringType()),
    StructField("marketing_channel", StringType()),   StructField("platform", StringType()),
    StructField("_corrupt_record", StringType()),
])

logging.info(f"Configured landing zone path: {landing}")
logging.info("Initializing Spark readStream...")

query = (spark.readStream.format("csv")
    .option("header", "true")
    .option("mode", "PERMISSIVE")
    .option("columnNameOfCorruptRecord", "_corrupt_record")
    .schema(schema_def)
    .load(landing)
    .withColumn("ingestion_time", F.current_timestamp())
    .withColumn("source_file", F.col("_metadata.file_path"))
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", checkpoint)
    .trigger(availableNow=True)
    .toTable("Bronze.batch_raw"))

logging.info("Stream started. Waiting for 'AvailableNow' trigger to process new files...")

query.awaitTermination()

if query.lastProgress:
    rows_processed = query.lastProgress.get('numInputRows', 0)
    logging.info(f" Batch completed! Processed {rows_processed:,} rows from Landing Zone.")
else:
    logging.info(" Batch completed! No new files/rows found in Landing Zone.")

logging.info(" Bronze Ingestion Pipeline finished successfully.")