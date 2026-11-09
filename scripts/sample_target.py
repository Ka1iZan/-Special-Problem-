import argparse
from pathlib import Path
import _bootstrap  # noqa: F401
from spner.data.io import read_jsonl
from spner.data.sampling import nested_target_subsets
from spner.utils import save_json, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("training", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budgets", nargs="+", type=float, default=[5, 10, 25])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = read_jsonl(args.training)
    subsets = nested_target_subsets(records, args.budgets, args.seed)
    save_json(args.out, {"training_sha256": file_sha256(args.training), "subsets": subsets,
                         "unit": "released_record", "selection": "nested_uniform_random_without_replacement"})
    print(f"Saved fixed subset IDs: {args.out}")


if __name__ == "__main__":
    main()
