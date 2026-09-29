import argparse

from databricks.sdk.runtime import spark
from pyspark.sql import functions as F

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", required=True)
parser.add_argument("--schema", required=True)
args = parser.parse_args()

spark.sql(f"USE CATALOG `{args.catalog}`")
spark.sql(f"USE SCHEMA `{args.schema}`")

silver = spark.table("silver_house_pricing")

by_locality = (
    silver.groupBy("locality")
    .agg(
        F.count("*").alias("listing_count"),
        F.round(F.avg("price"), 0).alias("avg_price"),
        F.round(F.avg("price_per_sqft"), 0).alias("avg_price_per_sqft"),
        F.round(F.avg("area"), 0).alias("avg_area"),
    )
    .orderBy(F.desc("listing_count"))
    .limit(30)
)
by_locality.write.mode("overwrite").saveAsTable("eda_by_locality")

by_property_type = (
    silver.groupBy("property_type")
    .agg(
        F.count("*").alias("listing_count"),
        F.round(F.avg("price"), 0).alias("avg_price"),
        F.round(F.avg("price_per_sqft"), 0).alias("avg_price_per_sqft"),
    )
    .orderBy(F.desc("listing_count"))
)
by_property_type.write.mode("overwrite").saveAsTable("eda_by_property_type")

numeric_cols = ["area", "bedroom_num", "bathroom_num", "balcony_num", "age", "total_floors"]
corr_rows = [(c, float(silver.stat.corr(c, "price"))) for c in numeric_cols]
correlations = spark.createDataFrame(corr_rows, ["feature", "correlation_with_price"]).orderBy(
    F.desc(F.abs(F.col("correlation_with_price")))
)
correlations.write.mode("overwrite").saveAsTable("eda_correlations")

print("EDA tables written: eda_by_locality, eda_by_property_type, eda_correlations")
