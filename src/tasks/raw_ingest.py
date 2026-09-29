import argparse

from databricks.sdk.runtime import spark
from pyspark.sql import functions as F

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", required=True)
parser.add_argument("--schema", required=True)
args = parser.parse_args()

spark.sql(f"USE CATALOG `{args.catalog}`")
spark.sql(f"USE SCHEMA `{args.schema}`")

source_path = f"/Volumes/{args.catalog}/{args.schema}/raw_data/mumbai-house-price-data-cleaned.csv"

raw = (
    spark.read.option("header", "true")
    .option("inferSchema", "true")
    .option("multiLine", "true")
    .csv(source_path)
    .withColumn("_source_file", F.lit(source_path))
    .withColumn("_ingested_at", F.current_timestamp())
)

raw.write.mode("overwrite").saveAsTable("raw_house_pricing")
