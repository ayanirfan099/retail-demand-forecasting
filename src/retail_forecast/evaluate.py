"""Expanding-window evaluation for daily next-day forecasts."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from .features import FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, make_features, validate_sales
from .metrics import regression_metrics


def _mase_scale(train: pd.DataFrame) -> float | None:
    series = train.sort_values(["store", "item", "date"]).groupby(["store", "item"], sort=False)["sales"]
    seasonal_errors = series.apply(lambda values: values.diff(7).abs()).dropna()
    if seasonal_errors.empty:
        return None
    return float(seasonal_errors.mean())


def evaluate_sales(frame: pd.DataFrame, *, test_days: int = 28, model: str = "hist_gradient_boosting"):
    """Train before test_start, then score test days in a rolling one-day-ahead setting.

    For a test date, lags may use actual sales from earlier dates in the test window,
    as those observations would be known to a daily reforecasting process.
    """
    if test_days < 1:
        raise ValueError("test_days must be at least 1")
    raw = validate_sales(frame)
    featured = make_features(raw).dropna(subset=FEATURES + ["sales"])
    if featured.empty:
        raise ValueError("not enough history; each series needs at least 14 prior days")
    test_start = featured["date"].max() - pd.Timedelta(days=test_days - 1)
    train = featured[featured["date"] < test_start]
    test = featured[featured["date"] >= test_start]
    if train.empty or test.empty:
        raise ValueError("time split produced an empty train or test set; add more history or reduce test_days")
    prep = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numeric", "passthrough", NUMERIC_FEATURES),
    ])
    if model == "hist_gradient_boosting":
        # Dense one-hot data is adequate for a portfolio-sized retail dataset.
        prep = ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ])
        estimator = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, l2_regularization=1.0, random_state=42)
    elif model == "random_forest":
        estimator = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42)
    else:
        raise ValueError("model must be hist_gradient_boosting or random_forest")
    pipeline = make_pipeline(prep, estimator)
    pipeline.fit(train[FEATURES], train["sales"])
    prediction = np.maximum(0, pipeline.predict(test[FEATURES]))
    baseline_prediction = np.maximum(0, test["lag_7"].to_numpy(dtype=float))
    scale = _mase_scale(raw[raw["date"] < test_start])
    return {
        "model": model,
        "forecast_mode": "rolling one-day-ahead; model fit once before test window",
        "train_start": str(train["date"].min().date()),
        "train_end": str(train["date"].max().date()),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "series": int(raw[["store", "item"]].drop_duplicates().shape[0]),
        "model_metrics": regression_metrics(test["sales"], prediction, mase_scale=scale),
        "seasonal_naive_lag_7_metrics": regression_metrics(test["sales"], baseline_prediction, mase_scale=scale),
    }
