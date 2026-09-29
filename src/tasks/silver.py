import argparse

from databricks.sdk.runtime import spark
from pyspark.sql import functions as F

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", required=True)
parser.add_argument("--schema", required=True)
args = parser.parse_args()

spark.sql(f"USE CATALOG `{args.catalog}`")
spark.sql(f"USE SCHEMA `{args.schema}`")

bronze = spark.table("bronze_house_pricing")

VALID_PROPERTY_TYPES = ["Apartment", "Villa", "Independent House", "Independent Floor", "Studio Apartment"]
VALID_FURNISHED = ["Unfurnished", "Semi-Furnished", "Furnished"]

silver = (
    bronze.filter(F.col("price") > 0)
    # 99.9th percentile is ~4e8; anything past 5e8 is a scraping/overflow artifact
    # (one row was exactly 2147483647 — INT32_MAX — not a real price).
    .filter(F.col("price") <= 500000000)
    .filter(F.col("area") > 0)
    .filter(F.col("property_type").isin(VALID_PROPERTY_TYPES))
    .filter(F.col("furnished").isin(VALID_FURNISHED))
    .filter(F.col("bedroom_num") >= 0)
    .filter(F.col("bathroom_num") >= 0)
    .filter(F.col("balcony_num") >= 0)
    .filter(F.col("total_floors") >= 1)
    .filter(F.col("age") >= 0)
    .filter(F.col("latitude").between(6.0, 38.0))
    .filter(F.col("longitude").between(68.0, 98.0))
    .withColumn("price_per_sqft", F.round(F.col("price") / F.col("area"), 2))
    .dropDuplicates(["title", "locality", "area", "price", "bedroom_num", "bathroom_num"])
    .withColumn("house_id", F.monotonically_increasing_id())
)

silver.write.mode("overwrite").saveAsTable("silver_house_pricing")
