import argparse
from pathlib import Path
import _bootstrap  # noqa: F401
from spner.data.io import read_jsonl
from spner.data.audit import audit_records
from spner.utils import file_sha256, save_json


def main():
    parser = argparse.ArgumentParser(description="Counts, duplicate leakage and entity overlap audit")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = [record for path in args.inputs for record in read_jsonl(path)]
    report = audit_records(records)
    report["inputs"] = [{"path": str(p.resolve()), "sha256": file_sha256(p)} for p in args.inputs]
    save_json(args.out, report)
    print(f"Audited {len(records)} records; report: {args.out}")
    if report["cross_split_duplicate_groups"] or report["document_leaks"]:
        print("Unresolved cross-split overlap: inspect report before training.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
