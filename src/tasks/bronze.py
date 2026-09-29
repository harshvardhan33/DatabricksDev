import argparse

from databricks.sdk.runtime import spark
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", required=True)
parser.add_argument("--schema", required=True)
args = parser.parse_args()

spark.sql(f"USE CATALOG `{args.catalog}`")
spark.sql(f"USE SCHEMA `{args.schema}`")

raw = spark.table("raw_house_pricing")

bronze = raw.select(
    F.trim(F.col("title")).alias("title"),
    F.col("price").cast(DoubleType()).alias("price"),
    F.col("area").cast(DoubleType()).alias("area"),
    F.col("price_per_sqft").cast(DoubleType()).alias("price_per_sqft"),
    F.trim(F.col("locality")).alias("locality"),
    F.trim(F.col("city")).alias("city"),
    F.trim(F.col("property_type")).alias("property_type"),
    F.col("bedroom_num").cast(IntegerType()).alias("bedroom_num"),
    F.col("bathroom_num").cast(IntegerType()).alias("bathroom_num"),
    F.col("balcony_num").cast(IntegerType()).alias("balcony_num"),
    F.trim(F.col("furnished")).alias("furnished"),
    F.col("age").cast(IntegerType()).alias("age"),
    F.col("total_floors").cast(IntegerType()).alias("total_floors"),
    F.col("latitude").cast(DoubleType()).alias("latitude"),
    F.col("longitude").cast(DoubleType()).alias("longitude"),
    F.col("_source_file"),
    F.col("_ingested_at"),
).dropDuplicates()

bronze.write.mode("overwrite").saveAsTable("bronze_house_pricing")
