from __future__ import annotations

from .statistics import paired_run_summary, holm_adjust


def compare_planned_runs(rows: list[dict], plan: dict) -> list[dict]:
    """Require the complete planned family; never silently inner-join missing runs."""
    if not rows:
        raise ValueError("No experimental results supplied")
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate run_id")
    results = []
    expected_budget = float(plan["budget_percent"])
    for target in plan["targets"]:
        for contrast in plan["contrasts"]:
            selected = {}
            for method in (contrast["a"], contrast["b"]):
                method_rows = [row for row in rows if row["target"] == target
                               and float(row["budget_percent"]) == expected_budget
                               and row["method"] == method]
                if not method_rows:
                    raise ValueError(f"Missing planned condition: {target}/{method}/{expected_budget:g}%")
                if any(not row.get("pair_id") for row in method_rows):
                    raise ValueError("Missing pair_id")
                index = {row["pair_id"]: row for row in method_rows}
                if len(index) != len(method_rows):
                    raise ValueError(f"Duplicate pair for {target}/{method}")
                selected[method] = index
            left, right = selected[contrast["a"]], selected[contrast["b"]]
            if set(left) != set(right):
                raise ValueError(f"Unmatched pairs for {target}/{contrast['id']}")
            a, b = [], []
            for pair_id in sorted(left):
                for key in plan["pair_match_fields"] + contrast["extra_match_fields"]:
                    if not str(left[pair_id].get(key, "")).strip() or not str(right[pair_id].get(key, "")).strip():
                        raise ValueError(f"Missing matching metadata {key} in {pair_id}")
                    if str(left[pair_id][key]) != str(right[pair_id][key]):
                        raise ValueError(f"Mismatched {key} in {target}/{pair_id}")
                a.append(float(left[pair_id][contrast["endpoint"]]))
                b.append(float(right[pair_id][contrast["endpoint"]]))
            summary = paired_run_summary(a, b, seed=plan["analysis_seed"], draws=plan["bootstrap_draws"])
            results.append({"plan_id": plan["plan_id"], "plan_status": plan["status"],
                            "contrast": contrast["id"], "target": target,
                            "budget_percent": expected_budget, "a": contrast["a"], "b": contrast["b"],
                            "endpoint": contrast["endpoint"], **summary})
    for row, adjusted in zip(results, holm_adjust([row["p_value"] for row in results])):
        row["p_holm"] = adjusted
        row["reject_at_alpha"] = adjusted <= plan["alpha"]
        row["family_size"] = len(results)
    return results
