"""Leakage-aware calendar, lag, and rolling features."""
from __future__ import annotations
import pandas as pd

REQUIRED = {"date", "store", "item", "sales"}
NUMERIC_FEATURES = ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "day_of_week", "month", "day_of_year"]
CATEGORICAL_FEATURES = ["store", "item"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def validate_sales(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
    out = frame[["date", "store", "item", "sales"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["sales"] = pd.to_numeric(out["sales"], errors="coerce")
    if out[["date", "store", "item", "sales"]].isna().any().any():
        raise ValueError("date, store, item, and sales must be present and valid")
    if (out["sales"] < 0).any():
        raise ValueError("sales must be non-negative")
    out["store"] = out["store"].astype(str)
    out["item"] = out["item"].astype(str)
    # Multiple transactions for a store/item/day are summed into daily demand.
    return out.groupby(["date", "store", "item"], as_index=False, sort=True)["sales"].sum()


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build next-day predictors; all demand history features are shifted first."""
    out = validate_sales(frame).sort_values(["store", "item", "date"]).reset_index(drop=True)
    grouped = out.groupby(["store", "item"], sort=False)["sales"]
    for lag in (1, 7, 14):
        out[f"lag_{lag}"] = grouped.shift(lag)
    out["rolling_mean_7"] = out.groupby(["store", "item"], sort=False)["sales"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=7).mean()
    )
    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["day_of_year"] = out["date"].dt.dayofyear
    return out
