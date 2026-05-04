import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "training_data.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
RESULT_PATH = BASE_DIR / "results" / "step1_s1.json"

df = pd.read_csv(DATA_PATH)

X = df.drop("response_time_ms", axis=1)
y = df["response_time_ms"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR / 'mlflow.db'}")
mlflow.set_experiment("cloudpulse-response-time-ms")

results = []


def evaluate(model, name):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.set_tag("project_phase", "model_selection")

        mlflow.sklearn.log_model(model, name)

        return {"name": name, "mae": mae, "rmse": rmse, "r2": r2}


ridge = evaluate(Ridge(), "Ridge")
gbr = evaluate(GradientBoostingRegressor(random_state=42), "GradientBoosting")

results.extend([ridge, gbr])

best = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name": "cloudpulse-response-time-ms",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"],
}

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULT_PATH, "w") as f:
    json.dump(output, f, indent=4)

if best["name"] == "Ridge":
    final_model = Ridge()
else:
    final_model = GradientBoostingRegressor(random_state=42)

final_model.fit(X_train, y_train)
joblib.dump(final_model, MODEL_PATH)
