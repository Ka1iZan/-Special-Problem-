import argparse
from pathlib import Path
import _bootstrap  # noqa: F401
from spner.data.tlunified import load_tlunified_parquet
from spner.data.io import write_jsonl
from spner.data.audit import audit_records
from spner.utils import file_sha256, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/public_v1/tlunified_ner"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/tlunified_v1"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new processed version directory")
    all_records, files = [], []
    converted = {}
    for original, split in (("train", "train"), ("validation", "dev"), ("test", "test")):
        path = args.root / f"{original}.parquet"
        records = load_tlunified_parquet(path, split)
        converted[split] = records
        all_records.extend(records)
        files.append({"split": split, "source": str(path.resolve()), "sha256": file_sha256(path)})
    audit = audit_records(all_records)
    if audit["cross_split_duplicate_groups"] or audit["document_leaks"]:
        raise ValueError("Split leakage found; inspect raw data audit")
    for split, records in converted.items():
        write_jsonl(records, args.out / f"{split}.jsonl")
    audit["inputs"] = files
    audit["status"] = "validated_word_level_records; tokenizer_length_audit_pending"
    audit["document_id_policy"] = "Parent documents unavailable; original row ID preserved in example_id only"
    save_json(args.out / "manifest.json", audit)
    print(f"Prepared {len(all_records)} records in official splits: {args.out}")


if __name__ == "__main__":
    main()
