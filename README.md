# MLOps Lab CIE – API Response Time Prediction

## Project Overview

This project is part of the **MLOps Lab Continuous Internal Evaluation (CIE)**.
The objective is to build an end-to-end machine learning pipeline to **predict API response time (in milliseconds)** using system-level features.

---

## Problem Statement

Predict the **response time of an API** based on:

* Request Size (KB)
* Server Load
* Cache Status (0 = No, 1 = Yes)
* Region Latency (ms)

---

## Machine Learning Models Used

* **Ridge Regression**
* **Gradient Boosting Regressor**

Best model selected based on **lowest RMSE**

---

## Experiment Tracking

We used **MLflow** for:

* Tracking experiments
* Logging metrics (MAE, RMSE, R²)
* Comparing multiple models
* Model versioning

---

## Project Structure

```
Internals_Basics/
└── MLOPs_Lab_CIE/
    ├── data/
    │   └── training_data.csv
    ├── src/
    │   ├── train.py
    │   ├── tune.py
    │   ├── predict_cli.py
    │   └── register_model.py
    ├── models/
    │   └── model.pkl
    ├── results/
    │   ├── step1_s1.json
    │   ├── step2_s2.json
    │   ├── step3_s3.json
    │   └── step4_s6.json
    ├── Dockerfile
    ├── requirements.txt
```

---

## Steps Implemented

### Task 1: Model Training

* Trained Ridge and Gradient Boosting models
* Logged metrics using MLflow
* Selected best model based on RMSE

---

### Task 2: Hyperparameter Tuning

* Used RandomizedSearchCV
* Optimized Gradient Boosting parameters
* Logged results

---

### Task 3: Docker Deployment

* Created CLI-based prediction script
* Built Docker container
* Ran predictions using input arguments

---

### Task 4: Model Versioning

* Registered best model in MLflow Model Registry
* Created versioned model

---

## Docker Usage

### Build Image

```
docker build -t cloudpulse-predictor:v1 .
```

### Run Prediction

```
docker run cloudpulse-predictor:v1 python src/predict_cli.py \
--request_size_kb 297.3 \
--server_load 0.6 \
--is_cached 0 \
--region_latency 94
```

---

## Sample Output

```
Prediction: 208.16 ms
```

---

## Technologies Used

* Python
* Scikit-learn
* MLflow
* Docker
* Pandas & NumPy

---

## Key Learnings

* End-to-end ML pipeline design
* Experiment tracking with MLflow
* Model comparison & selection
* Containerization using Docker
* Model versioning

---

## Author

**Lakshya Khamesra**
MLOps Lab – BMS College of Engineering

---

## Note

This repository is created as part of an academic evaluation and follows the required structure strictly.
