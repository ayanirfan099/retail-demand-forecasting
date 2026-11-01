"""Execute the bundled synthetic holdout and plot its actual predictions."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from retail_forecast.evaluate import evaluate_sales

ROOT = Path(__file__).resolve().parents[1]
report = evaluate_sales(pd.read_csv(ROOT / "data/synthetic_demo.csv"), test_days=28, include_predictions=True)
report["dataset"] = "Bundled synthetic daily sales; software smoke only, not a public retail benchmark"
rows = pd.DataFrame(report["predictions"])
first = rows.sort_values(["store", "item"]).iloc[0]
example = rows[(rows.store == first.store) & (rows.item == first.item)].sort_values("date")
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
for column, label in [("sales", "Observed"), ("prediction", "Model"), ("seasonal_naive_lag_7", "Lag-7 baseline")]:
    axes[0].plot(pd.to_datetime(example.date), example[column], label=label)
axes[0].set(title=f"Example: {first.store} / {first.item}", ylabel="Daily units")
axes[0].tick_params(axis="x", rotation=25)
axes[0].legend()
axes[1].hist(rows.sales - rows.prediction, bins=20, color="#267f87", edgecolor="white")
axes[1].axvline(0, color="black", linestyle="--")
axes[1].set(title="Residuals across all holdout series", xlabel="Observed − predicted units", ylabel="Rows")
fig.suptitle("Synthetic smoke data — rolling next-day predictions, model fitted once")
out = ROOT / "reports"
out.mkdir(exist_ok=True)
fig.savefig(out / "synthetic-diagnostics.png", dpi=150)
plt.close(fig)
(out / "synthetic-diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
print("Saved synthetic holdout predictions, metrics and plots in reports/")
