import mlflow

model_name = "cloudpulse-response-time-ms-predictor"

run_id = "ed73113b6bc24728a6df843467fad983"

model_uri = f"runs:/{run_id}/Ridge"

result = mlflow.register_model(model_uri, model_name)

print("Version:", result.version)