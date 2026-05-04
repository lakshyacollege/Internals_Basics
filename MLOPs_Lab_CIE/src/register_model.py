import mlflow
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RESULT_PATH = BASE_DIR / "results" / "step4_s6.json"

model_name = "cloudpulse-response-time-ms-predictor"

mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR / 'mlflow.db'}")
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("cloudpulse-response-time-ms")

if experiment is None:
    raise RuntimeError("Run src/train.py before registering a model.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.project_phase = 'model_selection'",
    order_by=["metrics.rmse ASC"],
    max_results=1,
)

if not runs:
    raise RuntimeError("No model_selection runs found. Run src/train.py first.")

best_run = runs[0]
artifact_path = best_run.data.params.get("model")
run_id = best_run.info.run_id
source_metric_value = best_run.data.metrics["rmse"]
model_uri = f"runs:/{run_id}/{artifact_path}"

result = mlflow.register_model(model_uri, model_name)

output = {
    "registered_model_name": model_name,
    "version": int(result.version),
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": source_metric_value,
}

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULT_PATH, "w") as f:
    import json

    json.dump(output, f, indent=4)

print("Version:", result.version)
