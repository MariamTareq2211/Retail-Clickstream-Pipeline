from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from pyspark.sql import functions as F, Window
from delta.tables import DeltaTable

def s(c):
    t = F.trim(F.col(c).cast("string"))
    return F.when(t == "", F.lit(None)).otherwise(t)

def write_dlq(df):
    """df: source, entity, run_id, dlq_reason, raw_payload, source_file, bronze_ingestion_time, dlq_time, status"""
    occ = F.row_number().over(Window.partitionBy("source_file", "raw_payload").orderBy(F.lit(1)))
    df = (df.withColumn("_occ", occ)
            .withColumn("dlq_id", F.sha2(F.concat_ws("|", "source", "entity", "source_file", "raw_payload",
                                                     F.col("_occ").cast("string")), 256))
            .withColumn("dlq_date", F.to_date("bronze_ingestion_time")).drop("_occ"))
    if not spark.catalog.tableExists("Silver.dlq_events"):
        df.limit(0).write.format("delta").partitionBy("dlq_date").saveAsTable("Silver.dlq_events")
    lo, hi = df.agg(F.min("dlq_date"), F.max("dlq_date")).first()
    if lo is None:
        return
    (DeltaTable.forName(spark, "Silver.dlq_events").alias("t")
       .merge(df.alias("s"), f"t.dlq_date BETWEEN '{lo}' AND '{hi}' AND t.dlq_id = s.dlq_id")
       .whenNotMatchedInsertAll().execute())

def write_metrics(stage, rows, run_id):
    schema = "run_id string, stage string, metric string, value double"
    (spark.createDataFrame([(run_id, stage, k, float(v or 0)) for k, v in rows], schema)
          .withColumn("ts", F.current_timestamp())
          .write.format("delta").mode("append").saveAsTable("Silver.pipeline_metrics"))