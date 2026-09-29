import argparse

import mlflow
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", required=True)
parser.add_argument("--schema", required=True)
parser.add_argument("--experiment-path", required=True)
args = parser.parse_args()

mlflow.set_registry_uri("databricks-uc")
model_name = f"{args.catalog}.{args.schema}.house_price_model"

client = MlflowClient(registry_uri="databricks-uc")
experiment = mlflow.get_experiment_by_name(args.experiment_path)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_rmse ASC"],
    max_results=1,
)
if not runs:
    raise RuntimeError(f"No runs found in experiment {args.experiment_path}")

best_run = runs[0]
run_id = best_run.info.run_id
rmse = best_run.data.metrics.get("test_rmse")
model_family = best_run.data.tags.get("model_family", "unknown")

print(f"Best run: {run_id} ({model_family}), test_rmse={rmse:.0f}")

model_uri = f"runs:/{run_id}/model"
model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

client.set_registered_model_alias(model_name, "challenger", model_version.version)
client.set_model_version_tag(model_name, model_version.version, "model_family", model_family)
client.set_model_version_tag(model_name, model_version.version, "test_rmse", str(rmse))

print(f"Registered {model_name} v{model_version.version} as @challenger")
