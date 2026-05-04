import argparse
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

parser = argparse.ArgumentParser()

parser.add_argument("--request_size_kb", type=float, required=True)
parser.add_argument("--server_load", type=float, required=True)
parser.add_argument("--is_cached", type=int, required=True)
parser.add_argument("--region_latency", type=float, required=True)

args = parser.parse_args()

model = joblib.load(MODEL_PATH)

data = pd.DataFrame(
    [
        {
            "request_size_kb": args.request_size_kb,
            "server_load": args.server_load,
            "is_cached": args.is_cached,
            "region_latency": args.region_latency,
        }
    ]
)

prediction = model.predict(data)[0]

print(prediction)
