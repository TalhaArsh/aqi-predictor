"""
AQI Predictor - Phase 3.5: XGBoost Regression (absolute target)
================================================================
Predicts ABSOLUTE AQI directly using XGBoost.

Run:
python -m src.training.train_xg
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn
import dagshub

from src.training.data_loader import get_train_data, FORECAST_HORIZONS
from src.training.evaluate import (
    compute_metrics,
    compute_metrics_by_category,
    print_evaluation_report,
    metrics_summary_row,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = "aqi-predictor"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

EXPERIMENT_NAME = "xgboost_absolute"

PARAM_GRID = [
    {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 10,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
    },
    {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 15,
        "reg_alpha": 2.0,
        "reg_lambda": 10.0,
    }
]


def init_dagshub():
    dagshub.init(
        repo_owner=DAGSHUB_USERNAME,
        repo_name=DAGSHUB_REPO,
        mlflow=True
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(f"MLflow tracking: {mlflow.get_tracking_uri()}")
    logger.info(f"Experiment: {EXPERIMENT_NAME}")


def tune_xgb(X_train, y_train, X_val, y_val):

    best_params = None
    best_rmse = float("inf")

    for params in PARAM_GRID:

        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_val)

        rmse = np.sqrt(np.mean((pred - y_val) ** 2))

        logger.info(
            f"    n_est={params['n_estimators']}, "
            f"depth={params['max_depth']} "
            f"→ val_RMSE={rmse:.3f}"
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    logger.info(f"    → best params: {best_params}")

    return best_params


def train_one_horizon(data, horizon):

    logger.info(f"\n--- Training XGBoost for {horizon}h horizon ---")

    X_train = data[f"X_train_{horizon}"]
    X_val   = data[f"X_val_{horizon}"]
    X_test  = data[f"X_test_{horizon}"]

    y_train = data[f"y_abs_train_{horizon}"]
    y_val   = data[f"y_abs_val_{horizon}"]
    y_test  = data[f"y_abs_test_{horizon}"]

    imputer = SimpleImputer(strategy="median")

    cols = X_train.columns.tolist()

    X_train_i = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=cols,
        index=X_train.index
    )

    X_val_i = pd.DataFrame(
        imputer.transform(X_val),
        columns=cols,
        index=X_val.index
    )

    X_test_i = pd.DataFrame(
        imputer.transform(X_test),
        columns=cols,
        index=X_test.index
    )

    logger.info(
        f"  Shapes: "
        f"train={X_train_i.shape}, "
        f"val={X_val_i.shape}, "
        f"test={X_test_i.shape}"
    )

    logger.info("  Tuning XGBoost hyperparameters:")

    best_params = tune_xgb(
        X_train_i,
        y_train,
        X_val_i,
        y_val
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "xgb",
            XGBRegressor(
                **best_params,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    pipe.fit(X_train, y_train)

    pred_train = pipe.predict(X_train)
    pred_val   = pipe.predict(X_val)
    pred_test  = pipe.predict(X_test)

    pred_train = np.clip(pred_train, 0, 500)
    pred_val   = np.clip(pred_val, 0, 500)
    pred_test  = np.clip(pred_test, 0, 500)

    m_train = compute_metrics(y_train, pred_train)
    m_val   = compute_metrics(y_val, pred_val)
    m_test  = compute_metrics(y_test, pred_test)

    cat_breakdown = compute_metrics_by_category(
        y_test,
        pred_test
    )

    print_evaluation_report(
        f"XGBoost-{best_params}",
        horizon,
        m_train,
        m_val,
        m_test,
        cat_breakdown
    )

    with mlflow.start_run(run_name=f"xgb_{horizon}h"):

        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("horizon_h", horizon)

        for k, v in best_params.items():
            mlflow.log_param(k, v)

        mlflow.log_param("target_type", "absolute")
        mlflow.log_param("n_features", len(X_train.columns))

        for split_name, m in [
            ("train", m_train),
            ("val", m_val),
            ("test", m_test),
        ]:

            mlflow.log_metric(f"{split_name}_RMSE", m["RMSE"])
            mlflow.log_metric(f"{split_name}_MAE", m["MAE"])
            mlflow.log_metric(f"{split_name}_R2", m["R2"])
            mlflow.log_metric(f"{split_name}_n", m["n_samples"])

        cat_csv = MODELS_DIR / f"xgb_{horizon}h_test_by_category.csv"

        cat_breakdown.to_csv(cat_csv)

        mlflow.log_artifact(str(cat_csv))

        mlflow.sklearn.log_model(
            pipe,
            artifact_path=f"xgb_{horizon}h",
            registered_model_name=f"xgb_{horizon}h",
        )

        logger.info(f"  ✅ Logged to MLflow: xgb_{horizon}h")

    return {
        "summary": metrics_summary_row(
            "XGBoost",
            horizon,
            m_train,
            m_val,
            m_test
        )
    }


def main():

    init_dagshub()

    logger.info("\nLoading data...")

    data = get_train_data()

    all_summaries = []

    for h in FORECAST_HORIZONS:

        result = train_one_horizon(data, h)

        all_summaries.append(result["summary"])

    summary_df = pd.DataFrame(all_summaries)

    summary_df.to_csv(
        MODELS_DIR / "xgb_metrics.csv",
        index=False
    )

    print("\n" + "=" * 70)

    print("XGBOOST — ALL HORIZONS SUMMARY")

    print("=" * 70)

    print(
        summary_df[
            [
                "model",
                "horizon_h",
                "val_RMSE",
                "val_MAE",
                "val_R2",
                "test_RMSE",
                "test_MAE",
                "test_R2",
            ]
        ].to_string(index=False)
    )

    print(
        f"\n✅ Models on DagsHub: "
        f"https://dagshub.com/"
        f"{DAGSHUB_USERNAME}/"
        f"{DAGSHUB_REPO}/experiments"
    )


if __name__ == "__main__":
    main()