import argparse
import csv
import json
from pathlib import Path
import _bootstrap  # noqa: F401
from spner.analysis import compare_planned_runs
from spner.utils import file_sha256, save_json


def main():
    parser = argparse.ArgumentParser(description="Matched-run F1 contrasts with Holm adjustment")
    parser.add_argument("runs", type=Path)
    parser.add_argument("--plan", type=Path, default=Path("configs/analysis_plan.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.runs.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    results = compare_planned_runs(rows, plan)
    save_json(args.out.with_suffix(".json"), {"results": results, "plan": plan,
              "runs_sha256": file_sha256(args.runs), "plan_sha256": file_sha256(args.plan),
              "interpretation": "Paired run variability, conditional on fixed evaluation sets. CIs are pointwise."})
    with args.out.with_suffix(".csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Analyzed {len(results)} planned contrasts; outputs: {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
