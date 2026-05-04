import argparse
import joblib
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--request_size_kb", type=float, required=True)
parser.add_argument("--server_load", type=float, required=True)
parser.add_argument("--is_cached", type=int, required=True)
parser.add_argument("--region_latency", type=float, required=True)

args = parser.parse_args()

# Load model
model = joblib.load("models/model.pkl")

# Prepare input
data = np.array([[ 
    args.request_size_kb,
    args.server_load,
    args.is_cached,
    args.region_latency
]])

# Predict
prediction = model.predict(data)[0]

print(prediction)