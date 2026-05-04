import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, ParameterSampler, cross_validate, train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "training_data.csv"
MODEL_PATH = BASE_DIR / "models" / "tuned_model.pkl"
RESULT_PATH = BASE_DIR / "results" / "step2_s2.json"

df = pd.read_csv(DATA_PATH)

X = df.drop("response_time_ms", axis=1)
y = df["response_time_ms"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_distributions = {
    "n_estimators": [50, 150],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 10],
}

mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR / 'mlflow.db'}")
mlflow.set_experiment("cloudpulse-response-time-ms")

cv = KFold(n_splits=3, shuffle=True, random_state=42)
trials = list(
    ParameterSampler(
        param_distributions,
        n_iter=5,
        random_state=42,
    )
)
trial_results = []

with mlflow.start_run(run_name="tuning-cloudpulse"):
    for trial_number, params in enumerate(trials, start=1):
        with mlflow.start_run(run_name=f"trial-{trial_number}", nested=True):
            model = GradientBoostingRegressor(random_state=42, **params)
            scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "rmse": "neg_root_mean_squared_error",
                    "mae": "neg_mean_absolute_error",
                },
            )
            cv_rmse = float(-np.mean(scores["test_rmse"]))
            cv_mae = float(-np.mean(scores["test_mae"]))

            mlflow.log_params(params)
            mlflow.log_param("trial_number", trial_number)
            mlflow.log_metric("cv_rmse", cv_rmse)
            mlflow.log_metric("cv_mae", cv_mae)

            trial_results.append(
                {
                    "trial_number": trial_number,
                    "params": params,
                    "cv_rmse": cv_rmse,
                    "cv_mae": cv_mae,
                }
            )

    best_trial = min(trial_results, key=lambda trial: trial["cv_rmse"])
    best_model = GradientBoostingRegressor(random_state=42, **best_trial["params"])
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)

    best_mae = mean_absolute_error(y_test, preds)
    best_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mlflow.log_params(best_trial["params"])
    mlflow.log_metric("best_cv_rmse", best_trial["cv_rmse"])
    mlflow.log_metric("best_cv_mae", best_trial["cv_mae"])
    mlflow.log_metric("best_test_rmse", best_rmse)
    mlflow.log_metric("best_test_mae", best_mae)
    mlflow.sklearn.log_model(best_model, "tuned_model")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    output = {
        "search_type": "random",
        "n_folds": 3,
        "total_trials": len(trials),
        "best_params": best_trial["params"],
        "best_mae": best_mae,
        "best_cv_mae": best_trial["cv_mae"],
        "best_rmse": best_rmse,
        "best_cv_rmse": best_trial["cv_rmse"],
        "parent_run_name": "tuning-cloudpulse",
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=4)
