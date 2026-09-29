import argparse
from datetime import datetime, timezone

import mlflow
import numpy as np
from databricks.sdk.runtime import spark
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", required=True)
parser.add_argument("--schema", required=True)
parser.add_argument("--tolerance", type=float, default=0.02)  # challenger may be up to 2% worse
args = parser.parse_args()

spark.sql(f"USE CATALOG `{args.catalog}`")
spark.sql(f"USE SCHEMA `{args.schema}`")

mlflow.set_registry_uri("databricks-uc")
model_name = f"{args.catalog}.{args.schema}.house_price_model"
client = MlflowClient(registry_uri="databricks-uc")

challenger = client.get_model_version_by_alias(model_name, "challenger")
challenger_rmse = float(challenger.tags.get("test_rmse", "nan"))

passed = True
reasons = []

# 1. Sanity check: the challenger must load and score without error, producing
#    finite, positive predictions on a live sample of the feature table.
try:
    model = mlflow.pyfunc.load_model(f"models:/{model_name}@challenger")
    sample = spark.table("gold_house_pricing").drop("house_id", "price", "log_price").limit(20).toPandas()
    preds = np.asarray(model.predict(sample))
    if not np.isfinite(preds).all() or not (preds > 0).all():
        passed = False
        reasons.append("predictions contain non-positive or non-finite values")
except Exception as e:
    passed = False
    reasons.append(f"failed to load/score challenger model: {e}")

# 2. Performance vs the current champion (skipped if no champion exists yet).
champion_rmse = None
try:
    champion = client.get_model_version_by_alias(model_name, "champion")
    champion_rmse = float(champion.tags.get("test_rmse", "nan"))
    if challenger_rmse > champion_rmse * (1 + args.tolerance):
        passed = False
        reasons.append(
            f"challenger test_rmse {challenger_rmse:.0f} worse than champion {champion_rmse:.0f} "
            f"by more than {args.tolerance:.0%}"
        )
except MlflowException:
    reasons.append("no existing champion — bootstrapping")

if passed:
    client.set_registered_model_alias(model_name, "champion", challenger.version)
    reasons.append(f"promoted v{challenger.version} to @champion")

log_df = spark.createDataFrame(
    [(
        model_name,
        int(challenger.version),
        datetime.now(timezone.utc),
        passed,
        challenger_rmse,
        champion_rmse,
        "; ".join(reasons),
    )],
    "model_name string, version int, validated_at timestamp, passed boolean, "
    "challenger_rmse double, champion_rmse double, notes string",
)
log_df.write.mode("append").saveAsTable("model_validation_log")

print(f"Validation {'PASSED' if passed else 'FAILED'}: {'; '.join(reasons)}")

if not passed:
    raise RuntimeError(f"Model validation failed for v{challenger.version}: {'; '.join(reasons)}")
