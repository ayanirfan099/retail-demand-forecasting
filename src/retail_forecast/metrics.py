"""Forecast metrics with explicit zero-demand behavior."""
from __future__ import annotations
import numpy as np


def regression_metrics(actual, predicted, *, mase_scale: float | None = None):
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("actual and predicted must be non-empty arrays with matching shapes")
    errors = np.abs(y - p)
    result = {"n": int(y.size), "mae": float(errors.mean()),
              "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
              "wape": float(errors.sum() / np.abs(y).sum()) if np.abs(y).sum() else None}
    result["mase"] = float(errors.mean() / mase_scale) if mase_scale and mase_scale > 0 else None
    return result
