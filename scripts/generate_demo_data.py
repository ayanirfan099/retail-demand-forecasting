"""Generate a clearly synthetic daily demand file for smoke tests and demos."""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(24)
    rows = []
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    for store in ("store-01", "store-02"):
        for item in ("item-A", "item-B"):
            base = 24 if item == "item-A" else 13
            for i, date in enumerate(dates):
                weekly = 5 if date.dayofweek in (4, 5) else 0
                trend = i * 0.035
                promo = 8 if i in (45, 90, 135) else 0
                sales = max(0, round(base + weekly + trend + promo + rng.normal(0, 2.5)))
                rows.append({"date": date.date().isoformat(), "store": store, "item": item, "sales": sales})
    target = Path(__file__).resolve().parents[1] / "data" / "synthetic_demo.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)
    print(f"Wrote synthetic example data: {target}")

if __name__ == "__main__":
    main()
