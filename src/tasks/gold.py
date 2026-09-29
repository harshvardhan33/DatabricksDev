import argparse

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk.runtime import spark
from pyspark.sql import functions as F

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", required=True)
parser.add_argument("--schema", required=True)
args = parser.parse_args()

spark.sql(f"USE CATALOG `{args.catalog}`")
spark.sql(f"USE SCHEMA `{args.schema}`")

silver = spark.table("silver_house_pricing")

# Fort, Mumbai — used as a reference point so location becomes one numeric feature
# instead of raw lat/long alone.
CITY_CENTER_LAT, CITY_CENTER_LON = 18.9750, 72.8258
EARTH_RADIUS_KM = 6371.0

locality_freq = silver.groupBy("locality").agg(F.count("*").alias("locality_listing_count"))

furnished_map = F.create_map(
    F.lit("Unfurnished"), F.lit(0),
    F.lit("Semi-Furnished"), F.lit(1),
    F.lit("Furnished"), F.lit(2),
)

lat1, lon1 = F.radians(F.lit(CITY_CENTER_LAT)), F.radians(F.lit(CITY_CENTER_LON))
lat2, lon2 = F.radians(F.col("latitude")), F.radians(F.col("longitude"))
dlat, dlon = lat2 - lat1, lon2 - lon1
haversine_a = F.sin(dlat / 2) ** 2 + F.cos(lat1) * F.cos(lat2) * F.sin(dlon / 2) ** 2
distance_km = 2 * EARTH_RADIUS_KM * F.asin(F.sqrt(haversine_a))

gold = (
    silver.join(locality_freq, on="locality", how="left")
    .withColumn("furnished_encoded", furnished_map[F.col("furnished")])
    .withColumn("distance_from_center_km", F.round(distance_km, 2))
    .withColumn(
        "bed_bath_ratio",
        F.round(F.col("bedroom_num") / F.greatest(F.col("bathroom_num"), F.lit(1)), 2),
    )
    .withColumn("log_price", F.round(F.log("price"), 4))
    .select(
        "house_id",
        "price",
        "log_price",
        "area",
        "bedroom_num",
        "bathroom_num",
        "balcony_num",
        "total_floors",
        "age",
        "bed_bath_ratio",
        "property_type",
        "furnished_encoded",
        "locality_listing_count",
        "distance_from_center_km",
        "latitude",
        "longitude",
    )
)

fe = FeatureEngineeringClient()
table_name = f"{args.catalog}.{args.schema}.gold_house_pricing"

# house_id is regenerated every run (monotonically_increasing_id() isn't stable across
# jobs), so this is a full refresh each time, not an incremental upsert — drop and
# recreate rather than fe.write_table(mode="merge"), which is the only merge mode the
# Feature Engineering client supports and would just accumulate duplicate rows here.
spark.sql(f"DROP TABLE IF EXISTS {table_name}")
fe.create_table(
    name=table_name,
    primary_keys=["house_id"],
    df=gold,
    description="ML-ready engineered features for Mumbai house price prediction.",
)
