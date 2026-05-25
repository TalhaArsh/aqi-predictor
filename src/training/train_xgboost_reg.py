import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.training.data_loader_reg import (
    get_train_data
)

# ======================================================
# PARAM GRID
# ======================================================

PARAM_GRID = [

    {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
]

# ======================================================
# METRICS
# ======================================================

def compute_metrics(
    y_true,
    y_pred
):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {

        "RMSE": rmse,

        "MAE": mae,

        "R2": r2
    }


# ======================================================
# PRINT REPORT
# ======================================================

def print_report(

    model_name,

    train_metrics,

    val_metrics,

    test_metrics
):

    print("\n" + "=" * 70)

    print(
        f"{model_name}"
    )

    print("=" * 70)

    print(

        f"\n{'Split':<10}"

        f"{'RMSE':>12}"

        f"{'MAE':>12}"

        f"{'R2':>12}"
    )

    print("-" * 50)

    for split_name, m in [

        ("Train", train_metrics),

        ("Val", val_metrics),

        ("Test", test_metrics)
    ]:

        print(

            f"{split_name:<10}"

            f"{m['RMSE']:>12.3f}"

            f"{m['MAE']:>12.3f}"

            f"{m['R2']:>12.3f}"
        )


# ======================================================
# TUNE MODEL
# ======================================================

def tune_model(

    X_train,
    y_train,

    X_val,
    y_val
):

    best_model = None

    best_rmse = float("inf")

    for params in PARAM_GRID:

        model = XGBRegressor(

            **params,

            objective="reg:squarederror",

            random_state=42,

            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(
            X_val
        )

        metrics = compute_metrics(
            y_val,
            pred
        )

        print(
            f"{params} | "
            f"RMSE={metrics['RMSE']:.3f} | "
            f"R2={metrics['R2']:.3f}"
        )

        if metrics["RMSE"] < best_rmse:

            best_rmse = metrics["RMSE"]

            best_model = model

    return best_model


# ======================================================
# RECONSTRUCT AQI
# ======================================================

def reconstruct_aqi(
    current_aqi,
    pred_delta
):

    pred = (
        current_aqi
        + pred_delta
    )

    pred = np.clip(
        pred,
        0,
        500
    )

    return pred


# ======================================================
# RECURSIVE FORECAST
# ======================================================

def recursive_forecast(
    model,
    initial_row,
    steps=72
):

    current = initial_row.copy()

    forecasts = []

    current_aqi = current["aqi"]

    for step in range(steps):

        X = pd.DataFrame([current])

        pred_delta = model.predict(X)[0]

        next_aqi = (
            current_aqi
            + pred_delta
        )

        next_aqi = np.clip(
            next_aqi,
            0,
            500
        )

        forecasts.append(
            next_aqi
        )

        # ==========================================
        # Update AQI Lags
        # ==========================================

        for lag in [24, 12, 6, 3, 2]:

            current[f"aqi_lag_{lag}"] = (
                current[f"aqi_lag_{lag-1}"]
            )

        current["aqi_lag_1"] = next_aqi

        # ==========================================
        # Update Rolling Features
        # ==========================================

        lag_values = [

            current[f"aqi_lag_{i}"]

            for i in range(1, 25)
        ]

        current["aqi_roll_mean_3"] = (
            np.mean(lag_values[:3])
        )

        current["aqi_roll_mean_6"] = (
            np.mean(lag_values[:6])
        )

        current["aqi_roll_mean_12"] = (
            np.mean(lag_values[:12])
        )

        current["aqi_roll_mean_24"] = (
            np.mean(lag_values[:24])
        )

        current["aqi_roll_std_3"] = (
            np.std(lag_values[:3])
        )

        current["aqi_roll_std_6"] = (
            np.std(lag_values[:6])
        )

        current["aqi_roll_std_12"] = (
            np.std(lag_values[:12])
        )

        current["aqi_roll_std_24"] = (
            np.std(lag_values[:24])
        )

        current_aqi = next_aqi

    return forecasts


# ======================================================
# MAIN
# ======================================================

def main():

    print(
        "\nLoading recursive forecasting dataset..."
    )

    data = get_train_data()

    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"]

    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]

    aqi_train = data["aqi_train"]
    aqi_val   = data["aqi_val"]
    aqi_test  = data["aqi_test"]

    # ==================================================
    # TRAIN MODEL
    # ==================================================

    print("\nTraining XGBoost...")

    model = tune_model(

        X_train,
        y_train,

        X_val,
        y_val
    )

    # ==================================================
    # DELTA PREDICTIONS
    # ==================================================

    pred_train_delta = model.predict(
        X_train
    )

    pred_val_delta = model.predict(
        X_val
    )

    pred_test_delta = model.predict(
        X_test
    )

    # ==================================================
    # RECONSTRUCT AQI
    # ==================================================

    pred_train_abs = reconstruct_aqi(
        aqi_train,
        pred_train_delta
    )

    pred_val_abs = reconstruct_aqi(
        aqi_val,
        pred_val_delta
    )

    pred_test_abs = reconstruct_aqi(
        aqi_test,
        pred_test_delta
    )

    # ==================================================
    # TRUE FUTURE AQI
    # ==================================================

    true_train_abs = (
        aqi_train
        + y_train
    )

    true_val_abs = (
        aqi_val
        + y_val
    )

    true_test_abs = (
        aqi_test
        + y_test
    )

    # ==================================================
    # METRICS
    # ==================================================

    train_metrics = compute_metrics(
        true_train_abs,
        pred_train_abs
    )

    val_metrics = compute_metrics(
        true_val_abs,
        pred_val_abs
    )

    test_metrics = compute_metrics(
        true_test_abs,
        pred_test_abs
    )

    # ==================================================
    # PRINT RESULTS
    # ==================================================

    print_report(

        "Recursive XGBoost "
        "(1h Delta Forecast)",

        train_metrics,

        val_metrics,

        test_metrics
    )

    # ==================================================
    # 72H RECURSIVE FORECAST
    # ==================================================

    print("\n" + "=" * 70)

    print(
        "72 HOUR RECURSIVE FORECAST"
    )

    print("=" * 70)

    sample = X_test.iloc[-1].copy()

    recursive_preds = recursive_forecast(

        model,

        sample,

        steps=72
    )

    for i, pred in enumerate(
        recursive_preds,
        1
    ):

        print(
            f"Hour +{i:02d} | "
            f"AQI = {pred:.2f}"
        )


if __name__ == "__main__":

    main()