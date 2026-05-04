import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import numpy as np
import joblib
# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("response_time_ms", axis=1)
y = df["response_time_ms"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.set_tag("project_phase", "model_selection")

        mlflow.sklearn.log_model(model, name)

        return {"name": name, "mae": mae, "rmse": rmse, "r2": r2}

ridge = evaluate(Ridge(), "Ridge")
gbr = evaluate(GradientBoostingRegressor(), "GradientBoosting")

results.extend([ridge, gbr])

best = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name": "cloudpulse-response-time-ms",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

# Save best model
if best["name"] == "Ridge":
    final_model = Ridge()
else:
    final_model = GradientBoostingRegressor()

final_model.fit(X_train, y_train)
joblib.dump(final_model, "models/model.pkl")