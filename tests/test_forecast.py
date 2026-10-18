import unittest
import numpy as np
import pandas as pd
from retail_forecast.features import make_features
from retail_forecast.evaluate import evaluate_sales
from retail_forecast.metrics import regression_metrics

class ForecastTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        self.frame = pd.DataFrame({"date": list(dates) * 2,
                                   "store": ["A"] * 100 + ["B"] * 100,
                                   "item": ["x"] * 200,
                                   "sales": [20 + (d.dayofweek in (4, 5)) * 5 + i * .05 for i, d in enumerate(dates)] * 2})

    def test_current_target_is_not_used_to_build_its_features(self):
        f1 = make_features(self.frame)
        changed = self.frame.copy()
        changed.loc[changed.index == 80, "sales"] += 10000
        f2 = make_features(changed)
        row1 = f1[(f1.store == "A") & (f1.item == "x")].iloc[80]
        row2 = f2[(f2.store == "A") & (f2.item == "x")].iloc[80]
        self.assertEqual(row1["lag_1"], row2["lag_1"])
        self.assertEqual(row1["rolling_mean_7"], row2["rolling_mean_7"])
        self.assertNotEqual(row1["sales"], row2["sales"])

    def test_time_split_and_baseline_metrics_are_reported(self):
        result = evaluate_sales(self.frame, test_days=14)
        self.assertEqual(result["forecast_mode"], "rolling one-day-ahead; model fit once before test window")
        self.assertGreater(result["model_metrics"]["n"], 0)
        self.assertIn("mase", result["seasonal_naive_lag_7_metrics"])
        self.assertLess(result["train_end"], result["test_start"])

    def test_invalid_negative_demand_is_rejected(self):
        bad = self.frame.copy()
        bad.loc[0, "sales"] = -1
        with self.assertRaises(ValueError):
            make_features(bad)

    def test_metric_requires_matching_nonempty_arrays(self):
        with self.assertRaises(ValueError):
            regression_metrics([], [])

if __name__ == "__main__":
    unittest.main()
