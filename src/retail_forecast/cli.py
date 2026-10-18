from __future__ import annotations
import argparse
import json
import pandas as pd
from .evaluate import evaluate_sales


def main():
    parser = argparse.ArgumentParser(description="Evaluate one-day-ahead retail demand forecasts")
    parser.add_argument("--csv", required=True, help="CSV with date,store,item,sales columns")
    parser.add_argument("--test-days", type=int, default=28)
    parser.add_argument("--model", choices=["hist_gradient_boosting", "random_forest"], default="hist_gradient_boosting")
    parser.add_argument("--report", default=None, help="optional JSON report path")
    args = parser.parse_args()
    report = evaluate_sales(pd.read_csv(args.csv), test_days=args.test_days, model=args.model)
    output = json.dumps(report, indent=2)
    print(output)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(output + "\n")

if __name__ == "__main__":
    main()
