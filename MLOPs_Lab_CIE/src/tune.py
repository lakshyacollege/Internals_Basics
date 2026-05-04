import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import json

df = pd.read_csv("data/training_data.csv")

X = df.drop("response_time_ms", axis=1)
y = df["response_time_ms"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 150],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 10]
}

mlflow.set_experiment("cloudpulse-response-time-ms")

with mlflow.start_run(run_name="tuning-cloudpulse") as parent:

    search = RandomizedSearchCV(
        GradientBoostingRegressor(),
        param_grid,
        cv=3,
        n_iter=5,
        scoring="neg_mean_absolute_error"
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    output = {
        "search_type": "random",
        "n_folds": 3,
        "total_trials": 5,
        "best_params": search.best_params_,
        "best_mae": mae,
        "best_cv_mae": -search.best_score_,
        "parent_run_name": "tuning-cloudpulse"
    }

    with open("results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=4)