import argparse

import mlflow
import numpy as np
import pandas as pd
from databricks.sdk.runtime import spark
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", required=True)
parser.add_argument("--schema", required=True)
parser.add_argument("--experiment-path", required=True)
args = parser.parse_args()

spark.sql(f"USE CATALOG `{args.catalog}`")
spark.sql(f"USE SCHEMA `{args.schema}`")

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(args.experiment_path)

pdf = spark.table("gold_house_pricing").toPandas()

TARGET = "price"
CATEGORICAL = ["property_type"]
NUMERIC = [
    "area",
    "bedroom_num",
    "bathroom_num",
    "balcony_num",
    "total_floors",
    "age",
    "bed_bath_ratio",
    "furnished_encoded",
    "locality_listing_count",
    "distance_from_center_km",
    "latitude",
    "longitude",
]
FEATURES = NUMERIC + CATEGORICAL

X = pdf[FEATURES]
y = pdf[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
    remainder="passthrough",
)

MODELS = {
    "linear_regression": LinearRegression(),
    "ridge_regression": Ridge(alpha=1.0, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
    "xgboost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1),
}

mlflow.autolog(log_models=True, silent=True)

results = []
for name, estimator in MODELS.items():
    with mlflow.start_run(run_name=name) as run:
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        mape = float(mean_absolute_percentage_error(y_test, preds))

        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("test_mape", mape)
        mlflow.set_tag("model_family", name)

        results.append({"run_id": run.info.run_id, "model": name, "test_rmse": rmse, "test_r2": r2})

results_df = pd.DataFrame(results).sort_values("test_rmse")
print(results_df.to_string(index=False))
best = results_df.iloc[0]
print(f"Best model: {best['model']} (run_id={best['run_id']}, test_rmse={best['test_rmse']:.0f}, test_r2={best['test_r2']:.4f})")
