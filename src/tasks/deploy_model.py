import argparse

import mlflow
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", required=True)
parser.add_argument("--schema", required=True)
parser.add_argument("--deploy-endpoint", default="false")
args = parser.parse_args()

mlflow.set_registry_uri("databricks-uc")
model_name = f"{args.catalog}.{args.schema}.house_price_model"
endpoint_name = f"{args.schema}_house_price_endpoint"

if args.deploy_endpoint.lower() != "true":
    print(
        f"deploy_endpoint={args.deploy_endpoint!r} — skipping live endpoint deployment. "
        f"{model_name}@champion is registered and ready to serve. Re-run this job with the "
        f"deploy_endpoint parameter set to 'true' to provision the '{endpoint_name}' serving "
        f"endpoint (this creates billable infrastructure)."
    )
else:
    client = MlflowClient(registry_uri="databricks-uc")
    champion = client.get_model_version_by_alias(model_name, "champion")

    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    config = {
        "served_entities": [
            {
                "name": "champion",
                "entity_name": model_name,
                "entity_version": champion.version,
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }
        ],
        "traffic_config": {"routes": [{"served_model_name": "champion", "traffic_percentage": 100}]},
    }

    existing = [e["name"] for e in deploy_client.list_endpoints()]
    if endpoint_name in existing:
        deploy_client.update_endpoint_config(endpoint=endpoint_name, config=config)
        print(f"Updated endpoint {endpoint_name} to serve {model_name}@champion v{champion.version}")
    else:
        deploy_client.create_endpoint(name=endpoint_name, config=config)
        print(f"Created endpoint {endpoint_name} serving {model_name}@champion v{champion.version}")
